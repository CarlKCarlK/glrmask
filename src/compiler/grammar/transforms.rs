use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use rustc_hash::FxHashSet;

use crate::automata::regex::Expr;
use crate::compiler::glr::analysis::{
    eliminate_right_recursion, has_indirect_left_recursion, merge_identical_nonterminals,
    inline_null_productions, normalize_grammar,
};
use crate::grammar::flat::{GrammarDef, NonterminalID, Terminal};
use crate::grammar::flat::{Rule, Symbol, TerminalID};

const MAX_RUNTIME_REDUCTION_LEN: usize = 5;
const INLINE_PROTECTED_NONTERMINALS_ENV: &str = "GLRMASK_INLINE_PROTECTED_NONTERMINALS";

fn env_var_enabled(key: &str, default: bool) -> bool {
    std::env::var(key)
        .map(|v| {
            let n = v.trim().to_ascii_lowercase();
            !matches!(n.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(default)
}

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn emit_grammar_transform_profile(
    stage: &str,
    elapsed_ms: f64,
    rules_before: usize,
    rules_after: usize,
    extra: &str,
) {
    eprintln!(
        "[glrmask-profile] grammar_transform stage={} ms={:.3} rules_before={} rules_after={}{}",
        stage,
        elapsed_ms,
        rules_before,
        rules_after,
        extra,
    );
}

// ── Nullable terminal expansion ─────────────────────────────────────────────

/// Rewrite grammar rules so that nullable terminals (those matching the empty
/// string) are treated as optional.  Operates in place on owned rule data.
///
/// For each nullable terminal `T`, a fresh nonterminal is allocated with two
/// productions: `NT → ε` and `NT → T`.  Every occurrence of `T` in the
/// existing rules is replaced by `NT`.  The tokenizer's start-state finalizer
/// for `T` is assumed to already be drained before this function is called.
pub(crate) fn expand_nullable_terminals(
    rules: &mut Vec<Rule>,
    start: NonterminalID,
    nullable_terminals: &BTreeSet<TerminalID>,
) {
    if nullable_terminals.is_empty() {
        return;
    }

    // Compute the next available nonterminal ID from both the rule graph and
    // the declared start symbol. A sparse GrammarDef may have a start ID above
    // every nonterminal currently mentioned by a rule.
    let mut next_nt = rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|sym| match sym {
                Symbol::Nonterminal(id) => Some(*id),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .unwrap_or(start)
        .max(start)
        + 1;

    // Map: nullable terminal id → fresh nonterminal id.
    let mut nt_for_terminal = BTreeMap::<TerminalID, NonterminalID>::new();
    let mut extra_rules = Vec::new();

    for &tid in nullable_terminals {
        let fresh_nt = next_nt;
        next_nt += 1;
        nt_for_terminal.insert(tid, fresh_nt);

        // NT → ε
        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![],
        });
        // NT → T
        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![Symbol::Terminal(tid)],
        });
    }

    // Rewrite existing rules in place: replace nullable Terminal(T) with Nonterminal(NT).
    for rule in rules.iter_mut() {
        for sym in rule.rhs.iter_mut() {
            if let Symbol::Terminal(tid) = sym {
                if let Some(&nt) = nt_for_terminal.get(tid) {
                    *sym = Symbol::Nonterminal(nt);
                }
            }
        }
    }

    rules.extend(extra_rules);
}

fn remap_terminal_id(terminal: &Terminal, new_id: TerminalID) -> Terminal {
    match terminal {
        Terminal::Literal { bytes, .. } => Terminal::Literal {
            id: new_id,
            bytes: bytes.clone(),
        },
        Terminal::Pattern { pattern, utf8, .. } => Terminal::Pattern {
            id: new_id,
            pattern: pattern.clone(),
            utf8: *utf8,
        },
        Terminal::Expr { expr, .. } => Terminal::Expr {
            id: new_id,
            expr: expr.clone(),
        },
        Terminal::SpecialToken { token_id, .. } => Terminal::SpecialToken {
            id: new_id,
            token_id: *token_id,
        },
    }
}

fn nullable_terminals_for_grammar(grammar: &GrammarDef) -> BTreeSet<TerminalID> {
    grammar
        .terminals
        .iter()
        .filter_map(|terminal| terminal.is_nullable().then_some(terminal.id()))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TerminalIdentityRef<'a> {
    Literal {
        bytes: &'a [u8],
        is_ignore: bool,
    },
    Pattern {
        pattern: &'a str,
        utf8: bool,
        is_ignore: bool,
    },
    Expr {
        expr: &'a Expr,
        is_ignore: bool,
    },
    SpecialToken {
        token_id: u32,
        is_ignore: bool,
    },
}

fn terminal_identity_ref(terminal: &Terminal, is_ignore: bool) -> TerminalIdentityRef<'_> {
    match terminal {
        Terminal::Literal { bytes, .. } => TerminalIdentityRef::Literal {
            bytes,
            is_ignore,
        },
        Terminal::Pattern { pattern, utf8, .. } => TerminalIdentityRef::Pattern {
            pattern,
            utf8: *utf8,
            is_ignore,
        },
        Terminal::Expr { expr, .. } => TerminalIdentityRef::Expr { expr, is_ignore },
        Terminal::SpecialToken { token_id, .. } => TerminalIdentityRef::SpecialToken {
            token_id: *token_id,
            is_ignore,
        },
    }
}

/// Remove terminals that are no longer referenced by any normalized rule,
/// merge identical terminals within the same explicit lexer partition, and
/// compact the remaining terminal IDs to a dense 0..N-1 range. Distinct
/// partition assignments are part of the compilation structure and must be
/// preserved even when two terminals have identical languages. Mutates the
/// grammar in place.
pub(crate) fn compact_unused_terminals(grammar: &mut GrammarDef) -> bool {
    let terminal_count = grammar.terminals.len();
    let mut used_flags = vec![false; terminal_count];
    for rule in grammar.rules.iter() {
        for symbol in &rule.rhs {
            if let Symbol::Terminal(terminal_id) = symbol {
                let index = *terminal_id as usize;
                assert!(
                    index < terminal_count,
                    "terminal id {terminal_id} referenced by a rule but missing from grammar.terminals",
                );
                used_flags[index] = true;
            }
        }
    }
    if let Some(ignore_terminal) = grammar.ignore_terminal {
        let index = ignore_terminal as usize;
        assert!(
            index < terminal_count,
            "ignore terminal id {ignore_terminal} missing from grammar.terminals",
        );
        used_flags[index] = true;
    }
    if let Some(automaton) = &grammar.direct_regular_automaton {
        for state in &automaton.states {
            for &terminal in state.transitions.keys() {
                let index = terminal as usize;
                assert!(
                    index < terminal_count,
                    "direct regular terminal id {terminal} missing from grammar.terminals",
                );
                used_flags[index] = true;
            }
        }
    }

    // The common imported-grammar case is already compact: every terminal is
    // referenced, terminal IDs match their vector positions, and terminal
    // languages are unique within their lexer partition/isolation class. Prove
    // that with borrowed identities and return before cloning every terminal,
    // rewriting every rule, and rebuilding every direct-regular transition map.
    let already_identity_compaction = used_flags.iter().all(|used| *used)
        && grammar
            .terminals
            .iter()
            .enumerate()
            .all(|(index, terminal)| terminal.id() as usize == index)
        && {
            let mut identities = HashMap::with_capacity(terminal_count);
            grammar.terminals.iter().enumerate().all(|(index, terminal)| {
                let terminal_id = index as TerminalID;
                let identity = (
                    terminal_identity_ref(terminal, grammar.ignore_terminal == Some(terminal_id)),
                    grammar.lexer_partitions.get(&terminal_id).map(String::as_str),
                    grammar.residual_isolation_classes.get(&terminal_id).copied(),
                );
                identities.insert(identity, terminal_id).is_none()
            })
        };
    if already_identity_compaction {
        return false;
    }

    let used = used_flags
        .into_iter()
        .enumerate()
        .filter_map(|(terminal, used)| used.then_some(terminal as TerminalID))
        .collect::<Vec<_>>();

    for &terminal in &used {
        if terminal as usize >= terminal_count {
            unreachable!("used-terminal validation above covers every terminal source");
        }
    }

    let mut remap = BTreeMap::<TerminalID, TerminalID>::new();
    let mut compacted = Vec::with_capacity(used.len());
    {
        // Keep deduplication identities borrowed from the source grammar. The
        // previous fallback path cloned every terminal language (including
        // potentially large Expr trees) solely to own the hash-map key. Source
        // terminals remain immutable until this pass is complete, so borrowed
        // identities are sufficient; clone only each genuinely unique terminal
        // once when constructing the compacted output vector.
        let mut canonical_ids = HashMap::<
            (TerminalIdentityRef<'_>, Option<&str>, Option<u32>),
            TerminalID,
        >::with_capacity(used.len());

        for old_id in used {
            let terminal = grammar.terminals.get(old_id as usize).unwrap_or_else(|| {
                panic!(
                    "terminal id {} referenced by a rule but missing from grammar.terminals",
                    old_id
                )
            });
            let is_ignore = grammar.ignore_terminal == Some(old_id);
            let identity = (
                terminal_identity_ref(terminal, is_ignore),
                grammar.lexer_partitions.get(&old_id).map(String::as_str),
                grammar.residual_isolation_classes.get(&old_id).copied(),
            );
            if let Some(&existing_id) = canonical_ids.get(&identity) {
                remap.insert(old_id, existing_id);
                continue;
            }
            let new_id = compacted.len() as TerminalID;
            canonical_ids.insert(identity, new_id);
            remap.insert(old_id, new_id);
            compacted.push(remap_terminal_id(terminal, new_id));
        }
    }

    for rule in grammar.rules.iter_mut() {
        for symbol in rule.rhs.iter_mut() {
            if let Symbol::Terminal(terminal_id) = symbol {
                *terminal_id = *remap
                    .get(terminal_id)
                    .expect("used terminal must have been assigned a compacted id");
            }
        }
    }

    if let Some(automaton) = &mut grammar.direct_regular_automaton {
        for state in &mut automaton.states {
            let old = std::mem::take(&mut state.transitions);
            for (old_terminal, targets) in old {
                let new_terminal = *remap
                    .get(&old_terminal)
                    .expect("direct regular terminal must have been assigned a compacted id");
                state
                    .transitions
                    .entry(new_terminal)
                    .or_default()
                    .extend(targets);
            }
            for targets in state.transitions.values_mut() {
                targets.sort_unstable();
                targets.dedup();
            }
        }
    }

    grammar.terminals = compacted;
    grammar.ignore_terminal = grammar.ignore_terminal.and_then(|old_id| remap.get(&old_id).copied());
    grammar.terminal_names = remap_terminal_names(&grammar.terminal_names, &remap);
    grammar.lexer_partitions = remap_lexer_partitions(&grammar.lexer_partitions, &remap);
    grammar.residual_isolation_classes =
        remap_residual_isolation_classes(&grammar.residual_isolation_classes, &remap);
    true
}

fn remap_residual_isolation_classes(
    classes: &BTreeMap<TerminalID, u32>,
    remap: &BTreeMap<TerminalID, TerminalID>,
) -> BTreeMap<TerminalID, u32> {
    let mut remapped = BTreeMap::new();
    for (&old_id, &class) in classes {
        let Some(&new_id) = remap.get(&old_id) else {
            continue;
        };
        if let Some(previous) = remapped.insert(new_id, class) {
            assert_eq!(
                previous, class,
                "terminal compaction merged distinct residual isolation classes {previous} and {class}",
            );
        }
    }
    remapped
}

fn remap_lexer_partitions(
    lexer_partitions: &BTreeMap<TerminalID, String>,
    remap: &BTreeMap<TerminalID, TerminalID>,
) -> BTreeMap<TerminalID, String> {
    let mut remapped = BTreeMap::new();
    for (old_id, partition) in lexer_partitions {
        let Some(&new_id) = remap.get(old_id) else {
            continue;
        };
        if let Some(previous) = remapped.insert(new_id, partition.clone()) {
            assert_eq!(
                previous, *partition,
                "terminal compaction merged terminals from distinct lexer partitions {previous:?} and {partition:?}",
            );
        }
    }
    remapped
}

fn remap_terminal_names(
    terminal_names: &BTreeMap<TerminalID, String>,
    remap: &BTreeMap<TerminalID, TerminalID>,
) -> BTreeMap<TerminalID, String> {
    terminal_names
        .iter()
        .filter_map(|(old_id, name)| remap.get(old_id).map(|new_id| (*new_id, name.clone())))
        .collect()
}

struct SingleUseInlineIndexes {
    only_production_index: Vec<Option<usize>>,
    has_multiple_productions: Vec<bool>,
    use_counts: Vec<usize>,
    sole_user_index: Vec<Option<usize>>,
    position_0_edges: FxHashSet<(NonterminalID, NonterminalID)>,
}

impl SingleUseInlineIndexes {
    fn build(rules: &[Rule]) -> Self {
        let max_nonterminal = rules
            .iter()
            .flat_map(|rule| {
                std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                    Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                    Symbol::Terminal(_) => None,
                }))
            })
            .max()
            .unwrap_or(0) as usize;
        let len = max_nonterminal + 1;
        let mut only_production_index = vec![None; len];
        let mut has_multiple_productions = vec![false; len];
        let mut use_counts = vec![0usize; len];
        let mut sole_user_index = vec![None; len];
        let mut position_0_edges = FxHashSet::default();
        position_0_edges.reserve(rules.len());

        for (index, rule) in rules.iter().enumerate() {
            let lhs = rule.lhs as usize;
            if only_production_index[lhs].replace(index).is_some() {
                has_multiple_productions[lhs] = true;
            }
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    let nonterminal = *nonterminal as usize;
                    use_counts[nonterminal] += 1;
                    if use_counts[nonterminal] == 1 {
                        sole_user_index[nonterminal] = Some(index);
                    } else {
                        sole_user_index[nonterminal] = None;
                    }
                }
            }
            if let Some(Symbol::Nonterminal(first_nonterminal)) = rule.rhs.first() {
                position_0_edges.insert((*first_nonterminal, rule.lhs));
            }
        }

        Self {
            only_production_index,
            has_multiple_productions,
            use_counts,
            sole_user_index,
            position_0_edges,
        }
    }

    fn single_production(&self, nonterminal: NonterminalID) -> Option<usize> {
        let index = nonterminal as usize;
        (!self.has_multiple_productions[index])
            .then_some(self.only_production_index[index])
            .flatten()
    }

    fn use_count(&self, nonterminal: NonterminalID) -> usize {
        self.use_counts[nonterminal as usize]
    }

    fn sole_user(&self, nonterminal: NonterminalID) -> Option<usize> {
        self.sole_user_index[nonterminal as usize]
    }

    fn creates_direct_left_recursion(
        &self,
        nonterminal: NonterminalID,
        replacement_first: NonterminalID,
    ) -> bool {
        self.position_0_edges
            .contains(&(nonterminal, replacement_first))
    }

    fn nonterminals(&self) -> impl Iterator<Item = NonterminalID> + '_ {
        (0..self.only_production_index.len()).map(|index| index as NonterminalID)
    }
}

pub(crate) fn inline_single_use_nonterminals(
    rules: &mut Vec<Rule>,
    protected_nonterminals: &BTreeSet<NonterminalID>,
    start: NonterminalID,
) {
    let inline_protected_nonterminals = env_var_enabled(INLINE_PROTECTED_NONTERMINALS_ENV, true);

    loop {
        let indexes = SingleUseInlineIndexes::build(rules);
        let mut inline_candidates = BTreeMap::<NonterminalID, (usize, Vec<Symbol>)>::new();

        for nonterminal in indexes.nonterminals() {
            // Entry identity is an external use, not an RHS use counted by the
            // index. Even when named-symbol inlining is enabled, removing this
            // production would leave GrammarDef.start naming a dead symbol.
            if nonterminal == start {
                continue;
            }
            let Some(production_index) = indexes.single_production(nonterminal) else {
                continue;
            };
            let rule = &rules[production_index];
            if rule.rhs.is_empty()
                || rule
                    .rhs
                    .iter()
                    .any(|symbol| matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal))
            {
                continue;
            }
            let use_count = indexes.use_count(nonterminal);
            if use_count == 0 {
                continue;
            }
            if protected_nonterminals.contains(&nonterminal) && !inline_protected_nonterminals {
                continue;
            }
            if rule.rhs.len() != 1 && use_count != 1 {
                continue;
            }
            if let Some(Symbol::Nonterminal(first)) = rule.rhs.first()
                && indexes.creates_direct_left_recursion(nonterminal, *first)
            {
                continue;
            }
            inline_candidates.insert(nonterminal, (production_index, rule.rhs.clone()));
        }

        if inline_candidates.is_empty() {
            break;
        }
        remove_cyclic_inline_candidates(&mut inline_candidates);
        if inline_candidates.is_empty() {
            break;
        }

        expand_inline_candidates(&mut inline_candidates);

        let remove_indexes: BTreeSet<usize> =
            inline_candidates.values().map(|(index, _)| *index).collect();
        let mut rewritten = Vec::with_capacity(rules.len());
        for (index, rule) in rules.iter().enumerate() {
            if remove_indexes.contains(&index) {
                continue;
            }
            let has_candidate = rule.rhs.iter().any(|symbol| {
                matches!(symbol, Symbol::Nonterminal(id) if inline_candidates.contains_key(id))
            });
            if has_candidate {
                let mut new_rhs = Vec::with_capacity(rule.rhs.len());
                for symbol in &rule.rhs {
                    if let Symbol::Nonterminal(id) = symbol {
                        if let Some((_, replacement_rhs)) = inline_candidates.get(id) {
                            new_rhs.extend(replacement_rhs.iter().cloned());
                            continue;
                        }
                    }
                    new_rhs.push(symbol.clone());
                }
                rewritten.push(Rule {
                    lhs: rule.lhs,
                    rhs: new_rhs,
                });
            } else {
                rewritten.push(rule.clone());
            }
        }
        *rules = rewritten;
    }
}

fn remove_cyclic_inline_candidates(
    inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
) {
    if inline_candidates.is_empty() {
        return;
    }

    // Preserve the old per-candidate reachability semantics, but avoid building
    // ordered dependency maps and allocating a new visited set for each start.
    // Candidate IDs are grammar nonterminal IDs, so one dense index plus visit
    // stamps gives the same search with substantially less bookkeeping.
    let candidate_ids: Vec<NonterminalID> = inline_candidates.keys().copied().collect();
    let max_nonterminal = candidate_ids.last().copied().unwrap_or(0) as usize;
    let mut candidate_index_by_nonterminal = vec![usize::MAX; max_nonterminal + 1];
    for (candidate_index, &nonterminal) in candidate_ids.iter().enumerate() {
        candidate_index_by_nonterminal[nonterminal as usize] = candidate_index;
    }

    let mut visited_at = vec![0usize; candidate_ids.len()];
    let mut search_stamp = 0usize;
    let mut stack = Vec::new();
    let mut cyclic = vec![false; candidate_ids.len()];

    for (start_index, &start_nonterminal) in candidate_ids.iter().enumerate() {
        search_stamp = search_stamp.checked_add(1).expect("cycle-search stamp overflow");
        stack.clear();
        stack.push(start_index);
        visited_at[start_index] = search_stamp;

        let mut reaches_start = false;
        while let Some(current_index) = stack.pop() {
            let current_nonterminal = candidate_ids[current_index];
            let (_, rhs) = inline_candidates
                .get(&current_nonterminal)
                .expect("candidate id must be present");
            for symbol in rhs {
                let Symbol::Nonterminal(child) = symbol else {
                    continue;
                };
                if *child == start_nonterminal {
                    reaches_start = true;
                    break;
                }
                let child_index = candidate_index_by_nonterminal
                    .get(*child as usize)
                    .copied()
                    .unwrap_or(usize::MAX);
                if child_index != usize::MAX && visited_at[child_index] != search_stamp {
                    visited_at[child_index] = search_stamp;
                    stack.push(child_index);
                }
            }
            if reaches_start {
                break;
            }
        }

        cyclic[start_index] = reaches_start;
    }

    for (candidate_index, nonterminal) in candidate_ids.into_iter().enumerate() {
        if cyclic[candidate_index] {
            inline_candidates.remove(&nonterminal);
        }
    }
}


/// Expand acyclic inline candidates bottom-up.
///
/// Candidate cycles are removed before this runs.  The old implementation
/// repeatedly cloned every candidate map snapshot until no replacement was
/// possible.  Scheduling candidates after their dependencies computes the same
/// transitive substitutions once per candidate, without recursive traversal.
fn expand_inline_candidates(
    inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
) {
    if inline_candidates.is_empty() {
        return;
    }

    let candidate_ids: Vec<NonterminalID> = inline_candidates.keys().copied().collect();
    let max_nonterminal = candidate_ids.last().copied().unwrap_or(0) as usize;
    let mut candidate_index_by_nonterminal = vec![usize::MAX; max_nonterminal + 1];
    for (candidate_index, &nonterminal) in candidate_ids.iter().enumerate() {
        candidate_index_by_nonterminal[nonterminal as usize] = candidate_index;
    }

    let mut remaining_dependencies = vec![0usize; candidate_ids.len()];
    let mut dependents = vec![Vec::<usize>::new(); candidate_ids.len()];
    for (candidate_index, &nonterminal) in candidate_ids.iter().enumerate() {
        let (_, rhs) = inline_candidates
            .get(&nonterminal)
            .expect("candidate id must be present");
        let mut dependencies = Vec::new();
        for symbol in rhs {
            let Symbol::Nonterminal(child) = symbol else {
                continue;
            };
            let child_index = candidate_index_by_nonterminal
                .get(*child as usize)
                .copied()
                .unwrap_or(usize::MAX);
            // Self references are intentionally left untouched, matching the
            // legacy loop. They should already have been removed as cycles.
            if child_index != usize::MAX
                && child_index != candidate_index
                && !dependencies.contains(&child_index)
            {
                dependencies.push(child_index);
            }
        }
        remaining_dependencies[candidate_index] = dependencies.len();
        for dependency in dependencies {
            dependents[dependency].push(candidate_index);
        }
    }

    let mut ready: Vec<usize> = remaining_dependencies
        .iter()
        .enumerate()
        .filter_map(|(candidate_index, &count)| (count == 0).then_some(candidate_index))
        .collect();
    let mut expanded_rhs: Vec<Option<Vec<Symbol>>> = vec![None; candidate_ids.len()];
    let mut expanded_count = 0usize;

    while let Some(candidate_index) = ready.pop() {
        let nonterminal = candidate_ids[candidate_index];
        let (_, rhs) = inline_candidates
            .get(&nonterminal)
            .expect("candidate id must be present");
        let mut expanded = Vec::with_capacity(rhs.len());
        for symbol in rhs {
            let replacement_index = match symbol {
                Symbol::Nonterminal(child) => candidate_index_by_nonterminal
                    .get(*child as usize)
                    .copied()
                    .filter(|&index| index != usize::MAX && index != candidate_index),
                Symbol::Terminal(_) => None,
            };
            if let Some(replacement_index) = replacement_index {
                let replacement = expanded_rhs[replacement_index]
                    .as_ref()
                    .expect("dependency must be expanded before its dependent");
                expanded.extend(replacement.iter().cloned());
            } else {
                expanded.push(symbol.clone());
            }
        }
        expanded_rhs[candidate_index] = Some(expanded);
        expanded_count += 1;

        for &dependent in &dependents[candidate_index] {
            remaining_dependencies[dependent] -= 1;
            if remaining_dependencies[dependent] == 0 {
                ready.push(dependent);
            }
        }
    }

    if expanded_count != candidate_ids.len() {
        // `remove_cyclic_inline_candidates` should make this unreachable. Keep
        // the legacy iterative algorithm as an exact fallback if a malformed
        // candidate graph somehow violates that invariant.
        expand_inline_candidates_iteratively(inline_candidates);
        return;
    }

    for (candidate_index, nonterminal) in candidate_ids.into_iter().enumerate() {
        inline_candidates
            .get_mut(&nonterminal)
            .expect("candidate id must be present")
            .1 = expanded_rhs[candidate_index]
            .take()
            .expect("all candidates must have expanded RHS");
    }
}

fn expand_inline_candidates_iteratively(
    inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
) {
    let inline_candidate_ids: BTreeSet<NonterminalID> =
        inline_candidates.keys().copied().collect();
    let mut expanded = true;
    while expanded {
        expanded = false;
        let snapshot: Vec<(NonterminalID, Vec<Symbol>)> = inline_candidates
            .iter()
            .map(|(&nonterminal, (_, rhs))| (nonterminal, rhs.clone()))
            .collect();
        for (nonterminal, rhs) in snapshot {
            if rhs.iter().any(|symbol| {
                matches!(symbol, Symbol::Nonterminal(id) if inline_candidate_ids.contains(id) && *id != nonterminal)
            }) {
                let mut new_rhs = Vec::with_capacity(rhs.len());
                for symbol in &rhs {
                    if let Symbol::Nonterminal(id) = symbol {
                        if *id != nonterminal {
                            if let Some((_, replacement_rhs)) = inline_candidates.get(id) {
                                new_rhs.extend(replacement_rhs.iter().cloned());
                                continue;
                            }
                        }
                    }
                    new_rhs.push(symbol.clone());
                }
                if new_rhs != rhs {
                    inline_candidates
                        .get_mut(&nonterminal)
                        .expect("inline candidate should still exist")
                        .1 = new_rhs;
                    expanded = true;
                }
            }
        }
    }
}

fn inline_post_bound_single_use_nonterminals(
    rules: &mut Vec<Rule>,
    protected_nonterminals: &BTreeSet<NonterminalID>,
    max_rhs_len: usize,
    start: NonterminalID,
) -> bool {
    let inline_protected_nonterminals = env_var_enabled(INLINE_PROTECTED_NONTERMINALS_ENV, true);
    let mut changed = false;

    loop {
        let indexes = SingleUseInlineIndexes::build(rules);
        let mut candidate = None;

        for nonterminal in indexes.nonterminals() {
            if nonterminal == start {
                continue;
            }
            let Some(candidate_rule_index) = indexes.single_production(nonterminal) else {
                continue;
            };
            if indexes.use_count(nonterminal) != 1 {
                continue;
            }
            if protected_nonterminals.contains(&nonterminal) && !inline_protected_nonterminals {
                continue;
            }
            let candidate_rule = &rules[candidate_rule_index];
            if candidate_rule.rhs.is_empty()
                || candidate_rule
                    .rhs
                    .iter()
                    .any(|symbol| matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal))
            {
                continue;
            }
            if let Some(Symbol::Nonterminal(first)) = candidate_rule.rhs.first()
                && indexes.creates_direct_left_recursion(nonterminal, *first)
            {
                continue;
            }

            let user_index = indexes
                .sole_user(nonterminal)
                .expect("one use must have a recorded user rule");
            let user_rule = &rules[user_index];
            let mut new_rhs = Vec::with_capacity(user_rule.rhs.len() + candidate_rule.rhs.len());
            for symbol in &user_rule.rhs {
                if let Symbol::Nonterminal(id) = symbol
                    && *id == nonterminal
                {
                    new_rhs.extend(candidate_rule.rhs.iter().cloned());
                    continue;
                }
                new_rhs.push(symbol.clone());
            }
            if new_rhs.len() > max_rhs_len {
                continue;
            }
            candidate = Some((candidate_rule_index, user_index, new_rhs));
            break;
        }

        let Some((candidate_rule_index, user_index, new_rhs)) = candidate else {
            break;
        };
        changed = true;
        let mut rewritten = Vec::with_capacity(rules.len().saturating_sub(1));
        for (index, rule) in rules.iter().enumerate() {
            if index == candidate_rule_index {
                continue;
            }
            if index == user_index {
                rewritten.push(Rule {
                    lhs: rule.lhs,
                    rhs: new_rhs.clone(),
                });
            } else {
                rewritten.push(rule.clone());
            }
        }
        *rules = rewritten;
    }
    changed
}

pub(crate) fn bound_runtime_reduction_length(
    grammar: &mut GrammarDef,
    max_rhs_len: usize,
) -> bool {
    if max_rhs_len < 2 || grammar.rules.iter().all(|rule| rule.rhs.len() <= max_rhs_len) {
        return false;
    }

    let mut next_nt = grammar.num_nonterminals();
    let mut rewritten = Vec::with_capacity(grammar.rules.len());

    for rule in grammar.rules.drain(..) {
        if rule.rhs.len() <= max_rhs_len {
            rewritten.push(rule);
            continue;
        }

        let lhs_name = grammar
            .nonterminal_names
            .get(&rule.lhs)
            .cloned()
            .unwrap_or_else(|| format!("N{}", rule.lhs));
        let mut symbols = rule.rhs;
        let mut lhs = rule.lhs;

        if symbols[0] == Symbol::Nonterminal(lhs) {
            let tail_nt = next_nt;
            next_nt += 1;
            grammar.nonterminal_names.insert(tail_nt, format!("{lhs_name}__tail"));

            rewritten.push(Rule {
                lhs,
                rhs: vec![Symbol::Nonterminal(lhs), Symbol::Nonterminal(tail_nt)],
            });

            symbols = symbols[1..].to_vec();
            lhs = tail_nt;

            if symbols.len() <= max_rhs_len {
                rewritten.push(Rule { lhs, rhs: symbols });
                continue;
            }
        }

        let first_chunk_len = max_rhs_len.min(symbols.len());
        let mut consumed = first_chunk_len;
        let mut stage = 0usize;

        let first_helper = next_nt;
        next_nt += 1;
        stage += 1;
        grammar
            .nonterminal_names
            .entry(first_helper)
            .or_insert_with(|| format!("{lhs_name}__prefix_{stage}"));
        rewritten.push(Rule {
            lhs: first_helper,
            rhs: symbols[..first_chunk_len].to_vec(),
        });

        let mut prefix_nt = first_helper;
        while symbols.len() - consumed > max_rhs_len - 1 {
            let helper = next_nt;
            next_nt += 1;
            stage += 1;
            grammar
                .nonterminal_names
                .entry(helper)
                .or_insert_with(|| format!("{lhs_name}__prefix_{stage}"));

            let take = max_rhs_len - 1;
            let mut rhs = Vec::with_capacity(max_rhs_len);
            rhs.push(Symbol::Nonterminal(prefix_nt));
            rhs.extend(symbols[consumed..consumed + take].iter().cloned());
            rewritten.push(Rule { lhs: helper, rhs });
            prefix_nt = helper;
            consumed += take;
        }

        let mut final_rhs = Vec::with_capacity(1 + symbols.len() - consumed);
        final_rhs.push(Symbol::Nonterminal(prefix_nt));
        final_rhs.extend(symbols[consumed..].iter().cloned());
        rewritten.push(Rule {
            lhs,
            rhs: final_rhs,
        });
    }

    grammar.rules = rewritten;
    true
}

fn collect_protected_nonterminals(grammar: &GrammarDef) -> BTreeSet<NonterminalID> {
    grammar
        .nonterminal_names
        .keys()
        .copied()
        .chain(std::iter::once(grammar.start))
        .collect()
}

/// Run only the grammar transforms without building the tokenizer.
/// The caller is responsible for calling `build_tokenizer` on the result.
pub(crate) fn prepare_grammar_transforms_only(grammar: GrammarDef) -> GrammarDef {
    let profiling = compile_profile_enabled();
    let nullable_rules_before = grammar.rules.len();
    let nullable_started_at = profiling.then(Instant::now);
    let nullable_terminals = nullable_terminals_for_grammar(&grammar);
    if let Some(started_at) = nullable_started_at {
        emit_grammar_transform_profile(
            "nullable_terminals_for_grammar",
            elapsed_ms(started_at),
            nullable_rules_before,
            nullable_rules_before,
            &format!(" nullable_terminals={}", nullable_terminals.len()),
        );
    }
    let mut normalized = grammar;
    prepare_grammar_transforms_impl(&mut normalized, &nullable_terminals, profiling);
    std::mem::take(&mut normalized)
}

/// Dynamic compilation can discard rules unreachable from the declared start
/// before running the expensive normalization fixed point. The shared pipeline
/// already removes the same rules during normalization, so this only moves a
/// semantics-preserving prune earlier and avoids transforming dead frontend
/// scaffolding.
pub(crate) fn prepare_dynamic_grammar_transforms_only(grammar: GrammarDef) -> GrammarDef {
    let profiling = compile_profile_enabled();
    let mut normalized = grammar;
    let rules_before = normalized.rules.len();
    let prune_started_at = profiling.then(Instant::now);
    normalized.rules = prune_unreachable_rules(&normalized.rules, normalized.start);
    if let Some(started_at) = prune_started_at {
        emit_grammar_transform_profile(
            "early_prune_unreachable_rules",
            elapsed_ms(started_at),
            rules_before,
            normalized.rules.len(),
            " dynamic=true",
        );
    }
    let nullable_rules_before = normalized.rules.len();
    let nullable_started_at = profiling.then(Instant::now);
    let nullable_terminals = nullable_terminals_for_grammar(&normalized);
    if let Some(started_at) = nullable_started_at {
        emit_grammar_transform_profile(
            "nullable_terminals_for_grammar",
            elapsed_ms(started_at),
            nullable_rules_before,
            nullable_rules_before,
            &format!(" nullable_terminals={}", nullable_terminals.len()),
        );
    }
    prepare_grammar_transforms_impl(&mut normalized, &nullable_terminals, profiling);
    normalized
}

/// Dynamic-runtime preparation for the mandatory native template parser.
/// Right-recursive reductions can consume an arbitrarily long concrete stack;
/// normalize them exactly before requiring an acyclic action characterization.
pub(crate) fn prepare_dynamic_glr_transforms_only(grammar: GrammarDef) -> GrammarDef {
    let profiling = compile_profile_enabled();
    let nullable_terminals = nullable_terminals_for_grammar(&grammar);
    let mut normalized = grammar;
    expand_nullable_terminals(
        &mut normalized.rules,
        normalized.start,
        &nullable_terminals,
    );

    let started = profiling.then(Instant::now);
    let before = normalized.rules.len();
    normalize_grammar(&mut normalized.rules, normalized.start);
    if let Some(started) = started {
        emit_grammar_transform_profile(
            "normalize_grammar",
            elapsed_ms(started),
            before,
            normalized.rules.len(),
            "",
        );
    }

    let protected_nonterminals = collect_protected_nonterminals(&normalized);
    inline_single_use_nonterminals(&mut normalized.rules, &protected_nonterminals, normalized.start);
    let max_reduction_len = std::env::var("GLRMASK_MAX_RUNTIME_REDUCTION_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_RUNTIME_REDUCTION_LEN);
    bound_runtime_reduction_length(&mut normalized, max_reduction_len);
    inline_post_bound_single_use_nonterminals(
        &mut normalized.rules,
        &protected_nonterminals,
        max_reduction_len,
        normalized.start,
    );
    compact_unused_terminals(&mut normalized);
    normalized
}

/// Dynamic-compile shared prefix: perform only the work needed to establish a
/// stable terminal domain before parser-only normalization. This lets tokenizer
/// construction overlap the parser normalization/analysis/table lane.
pub(crate) fn prepare_dynamic_shared_terminal_domain(mut grammar: GrammarDef) -> GrammarDef {
    let nullable_terminals = nullable_terminals_for_grammar(&grammar);
    expand_nullable_terminals(&mut grammar.rules, grammar.start, &nullable_terminals);
    grammar.rules = prune_unreachable_rules(&grammar.rules, grammar.start);
    compact_unused_terminals(&mut grammar);
    grammar
}

/// Parser-only continuation after `prepare_dynamic_shared_terminal_domain`.
pub(crate) fn prepare_dynamic_parser_after_terminal_domain(
    mut normalized: GrammarDef,
) -> (GrammarDef, bool) {
    normalize_grammar(&mut normalized.rules, normalized.start);
    let protected_nonterminals = collect_protected_nonterminals(&normalized);
    inline_single_use_nonterminals(&mut normalized.rules, &protected_nonterminals, normalized.start);
    let max_reduction_len = std::env::var("GLRMASK_MAX_RUNTIME_REDUCTION_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_RUNTIME_REDUCTION_LEN);
    bound_runtime_reduction_length(&mut normalized, max_reduction_len);
    inline_post_bound_single_use_nonterminals(
        &mut normalized.rules,
        &protected_nonterminals,
        max_reduction_len,
        normalized.start,
    );
    let terminal_domain_changed = compact_unused_terminals(&mut normalized);
    (normalized, terminal_domain_changed)
}

/// Prepare exactly the grammar structure needed by VocabPartition.
///
/// Unlike full static compilation, VocabPartition does not build an LR table
/// and therefore does not need parser-runtime normal form.  It does need:
///
/// * nullable terminals represented explicitly in the CFG before the lexer
///   drains their zero-byte matches;
/// * nullable nonterminal productions expanded once so terminal adjacency is
///   exposed cheaply to FIRST/FOLLOW analysis;
/// * rules unreachable from the declared start removed, otherwise dead rules
///   create spurious terminal follows and keep dead terminals alive; and
/// * the surviving terminal domain compacted/deduplicated before tokenizer and
///   L1/L2P work.
///
/// The heavier recursion/reduction-length transforms in
/// `prepare_grammar_transforms_only` exist for parser construction and are
/// deliberately omitted here.
pub(crate) fn prepare_grammar_for_vocab_partition(grammar: GrammarDef) -> GrammarDef {
    let nullable_terminals = nullable_terminals_for_grammar(&grammar);
    let mut prepared = grammar;
    expand_nullable_terminals(
        &mut prepared.rules,
        prepared.start,
        &nullable_terminals,
    );
    let num_nonterminals = prepared.num_nonterminals();
    prepared.rules = inline_null_productions(&prepared.rules, num_nonterminals);
    prepared.rules = prune_unreachable_rules(&prepared.rules, prepared.start);
    compact_unused_terminals(&mut prepared);
    prepared
}

fn prune_unreachable_rules(rules: &[Rule], start: NonterminalID) -> Vec<Rule> {
    let max_nonterminal = rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .unwrap_or(start)
        .max(start) as usize;
    let mut rules_by_lhs = vec![Vec::<usize>::new(); max_nonterminal + 1];
    for (index, rule) in rules.iter().enumerate() {
        rules_by_lhs[rule.lhs as usize].push(index);
    }

    let mut reachable = vec![false; max_nonterminal + 1];
    let mut worklist = vec![start];
    while let Some(nonterminal) = worklist.pop() {
        let nonterminal = nonterminal as usize;
        if reachable[nonterminal] {
            continue;
        }
        reachable[nonterminal] = true;
        for &rule_index in &rules_by_lhs[nonterminal] {
            for symbol in &rules[rule_index].rhs {
                if let Symbol::Nonterminal(next) = symbol
                    && !reachable[*next as usize]
                {
                    worklist.push(*next);
                }
            }
        }
    }

    rules
        .iter()
        .filter(|rule| reachable[rule.lhs as usize])
        .cloned()
        .collect()
}

/// The shared grammar transform steps (without tokenizer build).
fn prepare_grammar_transforms_impl(
    normalized: &mut GrammarDef,
    nullable_terminals: &BTreeSet<TerminalID>,
    profiling: bool,
) {
    let expand_rules_before = normalized.rules.len();
    let expand_started_at = profiling.then(Instant::now);
    expand_nullable_terminals(&mut normalized.rules, normalized.start, nullable_terminals);
    if let Some(started_at) = expand_started_at {
        emit_grammar_transform_profile(
            "expand_nullable_terminals",
            elapsed_ms(started_at),
            expand_rules_before,
            normalized.rules.len(),
            &format!(" nullable_terminals={}", nullable_terminals.len()),
        );
    }

    let normalize_rules_before = normalized.rules.len();
    let normalize_started_at = profiling.then(Instant::now);
    normalize_grammar(&mut normalized.rules, normalized.start);
    if let Some(started_at) = normalize_started_at {
        emit_grammar_transform_profile(
            "normalize_grammar",
            elapsed_ms(started_at),
            normalize_rules_before,
            normalized.rules.len(),
            "",
        );
    }

    let protected_rules_before = normalized.rules.len();
    let protected_started_at = profiling.then(Instant::now);
    let protected_nonterminals = collect_protected_nonterminals(normalized);
    if let Some(started_at) = protected_started_at {
        emit_grammar_transform_profile(
            "collect_protected_nonterminals",
            elapsed_ms(started_at),
            protected_rules_before,
            normalized.rules.len(),
            &format!(" protected_nonterminals={}", protected_nonterminals.len()),
        );
    }

    let inline_rules_before = normalized.rules.len();
    let inline_started_at = profiling.then(Instant::now);
    inline_single_use_nonterminals(&mut normalized.rules, &protected_nonterminals, normalized.start);
    if let Some(started_at) = inline_started_at {
        emit_grammar_transform_profile(
            "inline_single_use_nonterminals",
            elapsed_ms(started_at),
            inline_rules_before,
            normalized.rules.len(),
            " pass=pre_bound",
        );
    }

    let skip_identical_merges = std::env::var("GLRMASK_SKIP_IDENTICAL_NT_MERGE")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"));
    if !skip_identical_merges {
        let mut merge_iteration = 0usize;
        loop {
            merge_iteration += 1;
            let prev_len = normalized.rules.len();
            let merge_started_at = profiling.then(Instant::now);
            normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
            if let Some(started_at) = merge_started_at {
                emit_grammar_transform_profile(
                    "merge_identical_nonterminals",
                    elapsed_ms(started_at),
                    prev_len,
                    normalized.rules.len(),
                    &format!(" pass=pre_bound iteration={merge_iteration}"),
                );
            }
            if normalized.rules.len() == prev_len {
                break;
            }
        }
    }

    let max_reduction_len = std::env::var("GLRMASK_MAX_RUNTIME_REDUCTION_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_RUNTIME_REDUCTION_LEN);

    let bound_rules_before = normalized.rules.len();
    let bound_started_at = profiling.then(Instant::now);
    let bound_changed = bound_runtime_reduction_length(normalized, max_reduction_len);
    if let Some(started_at) = bound_started_at {
        emit_grammar_transform_profile(
            "bound_runtime_reduction_length",
            elapsed_ms(started_at),
            bound_rules_before,
            normalized.rules.len(),
            &format!(" max_reduction_len={max_reduction_len}"),
        );
    }

    let post_inline_rules_before = normalized.rules.len();
    let post_inline_started_at = profiling.then(Instant::now);
    let post_inline_changed = inline_post_bound_single_use_nonterminals(
        &mut normalized.rules,
        &protected_nonterminals,
        max_reduction_len,
        normalized.start,
    );
    if let Some(started_at) = post_inline_started_at {
        emit_grammar_transform_profile(
            "inline_post_bound_single_use_nonterminals",
            elapsed_ms(started_at),
            post_inline_rules_before,
            normalized.rules.len(),
            &format!(" max_reduction_len={max_reduction_len}"),
        );
    }

    if (bound_changed || post_inline_changed) && !skip_identical_merges {
        let mut final_merge_iteration = 0usize;
        loop {
            final_merge_iteration += 1;
            let prev_len = normalized.rules.len();
            let merge_started_at = profiling.then(Instant::now);
            normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
            if let Some(started_at) = merge_started_at {
                emit_grammar_transform_profile(
                    "merge_identical_nonterminals",
                    elapsed_ms(started_at),
                    prev_len,
                    normalized.rules.len(),
                    &format!(" pass=post_bound iteration={final_merge_iteration}"),
                );
            }
            if normalized.rules.len() == prev_len {
                break;
            }
        }
    } else if profiling {
        emit_grammar_transform_profile(
            "merge_identical_nonterminals",
            0.0,
            normalized.rules.len(),
            normalized.rules.len(),
            " pass=post_bound skipped=unchanged",
        );
    }

    let post_merge_rr_rules_before = normalized.rules.len();
    let post_merge_rr_started_at = profiling.then(Instant::now);
    let mut next_nt = normalized.num_nonterminals();
    let mut fresh_nt = || {
        let id = next_nt;
        next_nt += 1;
        id
    };
    eliminate_right_recursion(&mut normalized.rules, &mut fresh_nt);
    // Inlining and identical-nonterminal merging can recreate indirect left
    // recursion after the initial normalization pass. Most grammars do not,
    // so pay for the full fixed-point normalization only when the cheap
    // left-reachability check finds such a cycle.
    if has_indirect_left_recursion(&normalized.rules) {
        normalize_grammar(&mut normalized.rules, normalized.start);
    }
    if let Some(started_at) = post_merge_rr_started_at {
        emit_grammar_transform_profile(
            "post_merge_normalize",
            elapsed_ms(started_at),
            post_merge_rr_rules_before,
            normalized.rules.len(),
            " pass=post_merge",
        );
    }

    let compact_rules_before = normalized.rules.len();
    let compact_started_at = profiling.then(Instant::now);
    compact_unused_terminals(normalized);
    if let Some(started_at) = compact_started_at {
        emit_grammar_transform_profile(
            "compact_unused_terminals",
            elapsed_ms(started_at),
            compact_rules_before,
            normalized.rules.len(),
            "",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nt(id: NonterminalID) -> Symbol {
        Symbol::Nonterminal(id)
    }

    fn t(id: TerminalID) -> Symbol {
        Symbol::Terminal(id)
    }

    #[test]
    fn dynamic_split_preparation_matches_monolithic_when_terminal_domain_stabilizes() {
        let grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![nt(2), t(0)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![],
                },
                // Unreachable parser structure and terminal must disappear in both paths.
                Rule {
                    lhs: 1,
                    rhs: vec![t(1)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"live".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
            ],
            ..GrammarDef::default()
        };

        let monolithic = prepare_dynamic_glr_transforms_only(grammar.clone());
        let shared = prepare_dynamic_shared_terminal_domain(grammar);
        let (split, terminal_domain_changed) =
            prepare_dynamic_parser_after_terminal_domain(shared);

        assert!(!terminal_domain_changed);
        assert_eq!(
            serde_json::to_vec(&split).unwrap(),
            serde_json::to_vec(&monolithic).unwrap()
        );
    }

    #[test]
    fn dynamic_native_preparation_normalizes_recursive_unary_in_both_schedules() {
        let grammar = GrammarDef {
            start: 0,
            rules: vec![
                Rule { lhs: 0, rhs: vec![nt(1), t(2)] },
                Rule { lhs: 1, rhs: vec![t(0), nt(1)] },
                Rule { lhs: 1, rhs: vec![t(1)] },
            ],
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"!".to_vec() },
                Terminal::Literal { id: 1, bytes: b"x".to_vec() },
                Terminal::Literal { id: 2, bytes: b";".to_vec() },
            ],
            ..GrammarDef::default()
        };
        let monolithic = prepare_dynamic_glr_transforms_only(grammar.clone());
        let shared = prepare_dynamic_shared_terminal_domain(grammar);
        let (split, domain_changed) = prepare_dynamic_parser_after_terminal_domain(shared);
        assert!(!domain_changed);
        assert_eq!(serde_json::to_vec(&monolithic).unwrap(), serde_json::to_vec(&split).unwrap());
        let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_grammar_def(&monolithic);
        analyzed.check_recursion_boundedness().unwrap();
    }

    #[test]
    fn dynamic_native_recursive_unary_preserves_full_masks_and_unbounded_completion() {
        let vocab = crate::Vocab::new((0..=255).map(|id| (id, vec![id as u8])).collect());
        let grammar = "start p; nt p ::= unary ';'; nt unary ::= 'x' | '!' unary;";
        let constraint = crate::DynamicConstraint::compile(crate::Grammar::glrm(grammar), &vocab).unwrap();
        let backend = crate::__private::dynamic_parser_backend_report(&constraint);
        assert!(backend.as_array().unwrap().iter().all(|v|
            v["backend"] == "acyclic-template-dfa" && v["lr_table_present"] == false));
        for depth in [0, 1, 2, 17, 257, 1024] {
            let mut state = constraint.start();
            let mut mask = vec![0; constraint.mask_len()];
            let mut expected = vec![0; constraint.mask_len()];
            for token in [b'!', b'x'] { expected[token as usize / 32] |= 1 << (token % 32); }
            for _ in 0..depth {
                state.fill_mask(&mut mask);
                assert_eq!(mask, expected, "depth={depth}: unary prefix");
                assert!(!state.is_accepting());
                state.commit_token(b'!' as u32).unwrap();
            }
            state.fill_mask(&mut mask);
            assert_eq!(mask, expected, "depth={depth}: atom");
            state.commit_token(b'x' as u32).unwrap();
            expected.fill(0);
            expected[b';' as usize / 32] |= 1 << (b';' % 32);
            state.fill_mask(&mut mask);
            assert_eq!(mask, expected, "depth={depth}: terminator");
            assert!(!state.is_accepting());
            let mut rejected = state.clone();
            assert!(rejected.commit_token(b'!' as u32).is_err());
            state.commit_token(b';' as u32).unwrap();
            assert!(state.is_accepting());
            state.fill_mask(&mut mask);
            assert!(mask.iter().all(|&word| word == 0), "depth={depth}: complete");
        }
    }

    #[test]
    fn nullable_terminal_expansion_allocates_above_sparse_start_id() {
        let mut rules = vec![Rule {
            lhs: 0,
            rhs: vec![t(0)],
        }];
        let nullable = BTreeSet::from([0]);

        expand_nullable_terminals(&mut rules, 10, &nullable);

        let helper_lhs = rules
            .iter()
            .filter(|rule| rule.lhs != 0)
            .map(|rule| rule.lhs)
            .collect::<BTreeSet<_>>();
        assert_eq!(helper_lhs, BTreeSet::from([11]));
        assert_eq!(rules[0].rhs, vec![nt(11)]);
    }

    #[test]
    fn vocab_partition_prepare_prunes_dead_rules_before_terminal_compaction() {
        let grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![t(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![t(1)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"live".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
            ],
            ..GrammarDef::default()
        };

        let prepared = prepare_grammar_for_vocab_partition(grammar);

        assert_eq!(prepared.rules.len(), 1);
        assert_eq!(prepared.rules[0].lhs, 0);
        assert_eq!(prepared.rules[0].rhs, vec![t(0)]);
        assert_eq!(prepared.terminals.len(), 1);
        assert_eq!(prepared.terminals[0].id(), 0);
    }

    fn remove_cyclic_inline_candidates_reference(
        inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
    ) {
        fn reaches_start(
            start: NonterminalID,
            current: NonterminalID,
            deps: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
            seen: &mut BTreeSet<NonterminalID>,
        ) -> bool {
            if !seen.insert(current) {
                return false;
            }
            deps.get(&current).is_some_and(|nexts| {
                nexts
                    .iter()
                    .any(|&next| next == start || reaches_start(start, next, deps, seen))
            })
        }

        let inline_candidate_ids: BTreeSet<NonterminalID> =
            inline_candidates.keys().copied().collect();
        let deps: BTreeMap<NonterminalID, BTreeSet<NonterminalID>> = inline_candidates
            .iter()
            .map(|(&nt, (_, rhs))| {
                let rhs_deps = rhs
                    .iter()
                    .filter_map(|symbol| match symbol {
                        Symbol::Nonterminal(id) if inline_candidate_ids.contains(id) => Some(*id),
                        _ => None,
                    })
                    .collect();
                (nt, rhs_deps)
            })
            .collect();

        let cyclic: Vec<NonterminalID> = inline_candidate_ids
            .iter()
            .copied()
            .filter(|&nt| reaches_start(nt, nt, &deps, &mut BTreeSet::new()))
            .collect();
        for nt in cyclic {
            inline_candidates.remove(&nt);
        }
    }

    #[test]
    fn compact_unused_terminals_remaps_lexer_partitions_with_terminal_ids() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![t(1), t(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"unused-zero".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"first".to_vec(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"unused-middle".to_vec(),
                },
                Terminal::Literal {
                    id: 3,
                    bytes: b"second".to_vec(),
                },
            ],
            lexer_partitions: BTreeMap::from([
                (1, "literal-family".to_string()),
                (3, "other-family".to_string()),
            ]),
            ..GrammarDef::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(grammar.rules[0].rhs, vec![t(0), t(1)]);
        assert_eq!(
            grammar.lexer_partitions,
            BTreeMap::from([
                (0, "literal-family".to_string()),
                (1, "other-family".to_string()),
            ]),
        );
    }

    #[test]
    fn compact_unused_terminals_preserves_distinct_residual_isolation_classes() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![t(0), t(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"same".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"same".to_vec(),
                },
            ],
            lexer_partitions: BTreeMap::from([
                (0, "same-partition".to_string()),
                (1, "same-partition".to_string()),
            ]),
            residual_isolation_classes: BTreeMap::from([(0, 41), (1, 42)]),
            ..GrammarDef::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(grammar.terminals.len(), 2);
        assert_eq!(grammar.rules[0].rhs, vec![t(0), t(1)]);
        assert_eq!(
            grammar.residual_isolation_classes,
            BTreeMap::from([(0, 41), (1, 42)]),
        );
    }

    #[test]
    fn inline_candidate_expansion_matches_reference_on_varied_graphs() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *seed
        }

        let mut seed = 0x1a11_e7a0_5eed_u64;
        for case_index in 0..512 {
            let nonterminals = (next(&mut seed) % 10 + 1) as u32;
            let mut candidates = BTreeMap::new();
            for nonterminal in 0..nonterminals {
                if next(&mut seed) % 4 == 0 {
                    continue;
                }
                let rhs_len = (next(&mut seed) % 5) as usize;
                let mut rhs = Vec::with_capacity(rhs_len);
                for _ in 0..rhs_len {
                    if next(&mut seed) & 1 == 0 {
                        rhs.push(Symbol::Terminal((next(&mut seed) % 5) as u32));
                    } else {
                        rhs.push(Symbol::Nonterminal(
                            (next(&mut seed) % u64::from(nonterminals + 2)) as u32,
                        ));
                    }
                }
                candidates.insert(nonterminal, (nonterminal as usize, rhs));
            }

            remove_cyclic_inline_candidates(&mut candidates);
            let mut expected = candidates.clone();
            expand_inline_candidates_iteratively(&mut expected);
            expand_inline_candidates(&mut candidates);
            assert_eq!(candidates, expected, "case_index={case_index}");
        }
    }

    #[test]
    fn inline_cycle_pruning_matches_reference_on_varied_candidate_graphs() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *seed
        }

        let mut seed = 0x1a11_c7c1_e55_u64;
        for case_index in 0..512 {
            let nonterminals = (next(&mut seed) % 9 + 1) as u32;
            let mut candidates = BTreeMap::new();
            for nonterminal in 0..nonterminals {
                if next(&mut seed) % 4 == 0 {
                    continue;
                }
                let rhs_len = (next(&mut seed) % 5) as usize;
                let mut rhs = Vec::with_capacity(rhs_len);
                for _ in 0..rhs_len {
                    if next(&mut seed) & 1 == 0 {
                        rhs.push(Symbol::Terminal((next(&mut seed) % 5) as u32));
                    } else {
                        rhs.push(Symbol::Nonterminal(
                            (next(&mut seed) % u64::from(nonterminals + 2)) as u32,
                        ));
                    }
                }
                candidates.insert(nonterminal, (nonterminal as usize, rhs));
            }

            let mut expected = candidates.clone();
            remove_cyclic_inline_candidates_reference(&mut expected);
            remove_cyclic_inline_candidates(&mut candidates);
            assert_eq!(candidates, expected, "case_index={case_index}");
        }
    }

    #[test]
    fn inlining_preserves_recursively_referenced_entry_alias() {
        // The external entry is also referenced by an inner recursive branch.
        // It must not be discarded merely because its single alias production
        // can be substituted into that branch: GrammarDef.start still names it.
        let original = vec![
            Rule { lhs: 0, rhs: vec![nt(1)] },
            Rule { lhs: 1, rhs: vec![t(0), nt(0), t(1)] },
            Rule { lhs: 1, rhs: vec![t(2)] },
        ];
        let mut ordinary = original.clone();
        inline_single_use_nonterminals(&mut ordinary, &BTreeSet::from([0]), 0);
        assert!(ordinary.iter().any(|rule| rule.lhs == 0), "entry lost by ordinary inlining");
        let mut bounded = original;
        inline_post_bound_single_use_nonterminals(&mut bounded, &BTreeSet::from([0]), 5, 0);
        assert!(bounded.iter().any(|rule| rule.lhs == 0), "entry lost by post-bound inlining");
    }

    #[test]
    fn static_preparation_retains_recursive_entry_identity() {
        let source = GrammarDef {
            start: 0,
            rules: vec![
                Rule { lhs: 0, rhs: vec![nt(1)] },
                Rule { lhs: 1, rhs: vec![t(0), nt(0), t(1)] },
                Rule { lhs: 1, rhs: vec![t(2)] },
            ],
            terminals: (0..3).map(|id| Terminal::Literal { id, bytes: vec![b'a'+id as u8] }).collect(),
            nonterminal_names: BTreeMap::from([(0,"entry".into()),(1,"body".into())]),
            ..GrammarDef::default()
        };
        let prepared = prepare_grammar_transforms_only(source);
        assert_eq!(prepared.start, 0);
        assert!(prepared.rules.iter().any(|rule| rule.lhs == prepared.start));
        let shared = prepare_dynamic_shared_terminal_domain(prepared);
        let (dynamic, _) = prepare_dynamic_parser_after_terminal_domain(shared);
        assert!(!dynamic.rules.is_empty());
        assert!(dynamic.rules.iter().any(|rule| rule.lhs == dynamic.start));
    }

    #[test]
    fn inline_single_use_nonterminals_skips_candidate_cycles() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![nt(1), t(0)],
            },
            Rule {
                lhs: 1,
                rhs: vec![nt(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![nt(1)],
            },
        ];

        inline_single_use_nonterminals(&mut rules, &BTreeSet::new(), 0);

        assert_eq!(
            rules,
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![nt(1), t(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![nt(2)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![nt(1)],
                },
            ]
        );
    }

    #[test]
    fn inline_single_use_nonterminals_still_expands_acyclic_chains() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![nt(1), t(2)],
            },
            Rule {
                lhs: 1,
                rhs: vec![nt(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![t(0), t(1)],
            },
        ];

        inline_single_use_nonterminals(&mut rules, &BTreeSet::new(), 0);

        assert_eq!(
            rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![t(0), t(1), t(2)],
            }]
        );
    }
}
