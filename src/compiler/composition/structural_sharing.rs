use super::*;
use crate::compiler::glr::table::row::{ActionRow, GotoRow};
use crate::grammar::flat::{NonterminalID, Symbol};

// -------------------------------------------------------------------------
// Exact structural sharing for independently compiled subgrammars
// -------------------------------------------------------------------------
//
// Composition historically kept every child parser state disjoint.  That is
// correct but unnecessarily preserves copies of the same parser machine.  The
// quotient below is deliberately *not* a grammar-language-equivalence oracle.
// It uses only sufficient, mechanically checkable congruences:
//
//  * terminals may be related only when their retained lexer Expr values are
//    exactly equal, or when they are the same local terminal of the exact same
//    compiled artifact reused at multiple call sites, and neither terminal has
//    control/skip/ignore/special-token semantics;
//  * nonterminals are related by greatest structural bisimulation of their
//    production equations modulo the terminal relation, with the augmented
//    start and boundary/non-boundary role kept distinct. This relation is a
//    *candidate-isomorphism certificate* for independently compiled children;
//    nonterminal IDs are not physically identified by the ordinary quotient,
//    because equal generated languages do not imply equal caller goto columns;
//  * LR states are related by greatest row bisimulation modulo terminal
//    aliases, concrete nonterminal IDs, and the state relation itself. Every
//    observable guard set is
//    included in the initial state colour, so no quotient can make a guarded
//    stack predicate less discriminating.
//
// The materialized table keeps every original terminal and nonterminal ID. A
// merged state is simply given the union of the terminal aliases from its
// members, and the refinement proves that all aliases in one terminal-language
// class have the same quotient action under concrete nonterminal semantics.
// Existing parser DWAs then need no recompilation: the already supported
// local-state -> composed-state relation is just post-composed with the LR
// quotient map.
//
// A full proof is kept in docs/subgrammar-structural-sharing-proof.md.  Keep
// the implementation conservative when adding new table metadata: if a field
// can observe LR-state identity, it must either be transported through the
// quotient or included as an observation in the refinement.

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct StructuralSharingReport {
    pub(super) terminal_aliases: usize,
    pub(super) terminal_structural_matches: usize,
    pub(super) terminal_exact_checks: usize,
    pub(super) terminal_exact_unknown: usize,
    pub(super) nonterminals_before: usize,
    pub(super) nonterminal_classes: usize,
    pub(super) contextual_candidate_groups: usize,
    pub(super) contextual_states_saved: usize,
    pub(super) states_before: usize,
    pub(super) states_after: usize,
}

#[derive(Debug, Clone)]
pub(super) struct TerminalClassAnalysis {
    pub(super) classes: Vec<u32>,
    pub(super) structural_matches: usize,
    pub(super) exact_checks: usize,
    pub(super) exact_unknown: usize,
}

#[derive(Debug, Clone, Copy)]
struct TerminalRepresentative {
    component_index: usize,
    local_terminal: u32,
    global_terminal: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TerminalEquivalencePrefilter {
    byte_support: U8Set,
    scalar_prefix_fingerprint: Option<u64>,
    /// Algebraic fingerprint of original-vocabulary token IDs admitted by
    /// possible-matches from tokenizer reset. Equal sets necessarily have the
    /// same fingerprint. Unequal sets can collide, which only causes an extra
    /// exact automata check; the fingerprint is never an equivalence proof.
    reset_tokens: OriginalTokenSetFingerprint,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
struct OriginalTokenSetFingerprint {
    count: u64,
    sum: u128,
    sum_squares: u128,
    xor: u32,
}

impl OriginalTokenSetFingerprint {
    fn from_originals(originals: &[u32]) -> Self {
        let mut out = Self::default();
        for &token in originals {
            let token128 = token as u128;
            out.count += 1;
            out.sum += token128;
            out.sum_squares += token128 * token128;
            out.xor ^= token;
        }
        out
    }

    fn add_group(&mut self, group: Self) {
        self.count += group.count;
        self.sum += group.sum;
        self.sum_squares += group.sum_squares;
        self.xor ^= group.xor;
    }
}

fn terminal_equivalence_prefilter(
    component: &Constraint,
    terminal: u32,
    token_group_fingerprints: &[OriginalTokenSetFingerprint],
    prefix_depth: u8,
) -> Option<TerminalEquivalencePrefilter> {
    let byte_support = component.tokenizer.terminal_byte_support(terminal)?;
    let scalar_prefix_fingerprint = component
        .tokenizer
        .terminal_scalar_prefix_fingerprint(terminal, prefix_depth);
    let mut internal_tokens = RangeSetBlaze::<u32>::new();
    if let Some(weight) = component.possible_matches.get(&terminal) {
        for &tsid in component
            .internal_tsids_for_state(component.tokenizer.initial_state())
        {
            internal_tokens |= weight.tokens_for_tsid(tsid);
        }
    }
    let mut reset_tokens = OriginalTokenSetFingerprint::default();
    for internal_token in internal_tokens.iter() {
        reset_tokens.add_group(*token_group_fingerprints.get(internal_token as usize)?);
    }
    Some(TerminalEquivalencePrefilter {
        byte_support,
        scalar_prefix_fingerprint,
        reset_tokens,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum StructuralSymbolSignature {
    Terminal(u32),
    Nonterminal(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NonterminalRefinementSignature {
    previous_class: u32,
    productions: Vec<Vec<StructuralSymbolSignature>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructuralStateSignature {
    previous_class: u32,
    /// A state ID here intentionally makes the row a singleton.  This is used
    /// only when one physical row assigns observably different behavior to two
    /// terminal/nonterminal aliases that our symbol proof otherwise relates.
    alias_conflict_nonce: Option<u32>,
    guard_memberships: Vec<u32>,
    actions: Vec<(u32, Action, bool)>,
    gotos: Vec<(u32, u32, bool)>,
    advance: Vec<u32>,
    wide_frontiers: Vec<(u32, Vec<u32>)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ComponentActionSignature {
    Shift(u32, bool),
    StackShifts(Vec<(u32, Vec<u32>)>),
    GuardedStackShifts(Vec<(Vec<(u32, Vec<u32>)>, u32, Vec<u32>)>),
    Reduce(u32, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(u32, u32)>,
        accept: bool,
    },
    Accept,
    ReplaceShifts(Vec<u32>),
    Skip,
}

fn component_action_signature(action: &Action) -> ComponentActionSignature {
    match action {
        Action::Shift(target, replace) => ComponentActionSignature::Shift(*target, *replace),
        Action::StackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| (shift.pop, shift.pushes.clone()))
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            ComponentActionSignature::StackShifts(shifts)
        }
        Action::GuardedStackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    let mut guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states = guard.states.clone();
                            states.sort_unstable();
                            states.dedup();
                            (guard.pop, states)
                        })
                        .collect::<Vec<_>>();
                    guards.sort();
                    guards.dedup();
                    (
                        guards,
                        shift.pop,
                        shift.pushes.clone(),
                    )
                })
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            ComponentActionSignature::GuardedStackShifts(shifts)
        }
        Action::Reduce(nonterminal, len) => {
            ComponentActionSignature::Reduce(*nonterminal, *len)
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => ComponentActionSignature::Split {
            shift: *shift,
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => ComponentActionSignature::Accept,
        Action::ReplaceShifts(targets) => {
            ComponentActionSignature::ReplaceShifts(targets.iter().copied().collect())
        }
        Action::Skip => ComponentActionSignature::Skip,
    }
}

fn component_action_signature_remapped(
    action: &Action,
    state_classes: &[u32],
    nonterminal_classes: &[u32],
) -> ComponentActionSignature {
    let map_state = |state: u32| {
        state_classes
            .get(state as usize)
            .copied()
            .unwrap_or(state)
    };
    let map_nonterminal = |nonterminal: u32| {
        nonterminal_classes
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(nonterminal)
    };
    match action {
        Action::Shift(target, replace) => {
            ComponentActionSignature::Shift(map_state(*target), *replace)
        }
        Action::StackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    (
                        shift.pop,
                        shift.pushes.iter().map(|&state| map_state(state)).collect(),
                    )
                })
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            ComponentActionSignature::StackShifts(shifts)
        }
        Action::GuardedStackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    let mut guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states = guard
                                .states
                                .iter()
                                .map(|&state| map_state(state))
                                .collect::<Vec<_>>();
                            states.sort_unstable();
                            states.dedup();
                            (guard.pop, states)
                        })
                        .collect::<Vec<_>>();
                    guards.sort();
                    guards.dedup();
                    (
                        guards,
                        shift.pop,
                        shift.pushes.iter().map(|&state| map_state(state)).collect(),
                    )
                })
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            ComponentActionSignature::GuardedStackShifts(shifts)
        }
        Action::Reduce(nonterminal, len) => {
            ComponentActionSignature::Reduce(map_nonterminal(*nonterminal), *len)
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            let mut reduces = reduces
                .iter()
                .map(|&(nonterminal, len)| (map_nonterminal(nonterminal), len))
                .collect::<Vec<_>>();
            reduces.sort_unstable();
            reduces.dedup();
            ComponentActionSignature::Split {
                shift: shift.map(|(target, replace)| (map_state(target), replace)),
                reduces,
                accept: *accept,
            }
        }
        Action::Accept => ComponentActionSignature::Accept,
        Action::ReplaceShifts(targets) => {
            let mut targets = targets.iter().map(|&state| map_state(state)).collect::<Vec<_>>();
            targets.sort_unstable();
            targets.dedup();
            ComponentActionSignature::ReplaceShifts(targets)
        }
        Action::Skip => ComponentActionSignature::Skip,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ComponentStateSignature {
    previous_class: u32,
    actions: Vec<(u32, ComponentActionSignature, bool)>,
    gotos: Vec<(u32, u32, bool)>,
    advance: Vec<u32>,
}

pub(super) fn structural_sharing_enabled() -> bool {
    std::env::var_os("GLRMASK_COMPOSE_DISABLE_STRUCTURAL_SHARING").is_none()
}

pub(super) fn composition_terminal_classes(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    composed: &ComposedTable,
) -> TerminalClassAnalysis {
    let num_terminals = composed.table.num_terminals as usize;
    let mut classes = (0..num_terminals as u32).collect::<Vec<_>>();
    let mut ineligible = BitSet::new(num_terminals);
    for &terminal in &composed.table.control_terminals {
        ineligible.set(terminal as usize);
    }
    for &terminal in &composed.table.skip_terminals {
        ineligible.set(terminal as usize);
    }
    for child in children {
        // Placeholder terminals are linker controls even on the legacy splice
        // path where they are removed rather than retained in control_terminals.
        ineligible.set(child.placeholder_terminal as usize);
    }

    let components = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let automata_fallback_enabled = std::env::var("GLRMASK_COMPOSE_TERMINAL_AUTOMATA_EQUIV")
        .ok()
        .is_some_and(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        });
    let needs_automata_fallback = automata_fallback_enabled
        && components
            .iter()
            .any(|component| component.retained_terminal_exprs().is_none());
    let token_group_fingerprints = needs_automata_fallback.then(|| {
        components
            .iter()
            .map(|component| {
                component
                    .internal_token_groups()
                    .unwrap_or(&[])
                    .iter()
                    .map(|originals| OriginalTokenSetFingerprint::from_originals(originals))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    for (component_index, component) in components.iter().enumerate() {
        let offset = composed.terminal_offsets[component_index];
        if let Some(ignore) = component.ignore_terminal {
            ineligible.set((offset + ignore) as usize);
        }
        for special in &component.special_token_terminals {
            ineligible.set((offset + special.terminal_id) as usize);
        }
    }

    // Fast exact proofs are tried first. Current v11 artifacts retain terminal
    // Exprs, so the production path does not need to rediscover language
    // equivalence after load. For legacy/mixed artifacts, expensive compiled-
    // automaton rediscovery is explicitly opt-in; without proof metadata the
    // default is simply to leave terminals distinct.
    let mut representative_by_expr = FxHashMap::<Expr, u32>::default();
    let mut representative_by_artifact_terminal = FxHashMap::<(usize, u32), u32>::default();
    let mut representatives_by_prefilter =
        FxHashMap::<TerminalEquivalencePrefilter, Vec<TerminalRepresentative>>::default();
    let pair_limit = std::env::var("GLRMASK_COMPOSE_TERMINAL_EQUIV_PAIR_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1_024);
    let transition_work_limit =
        std::env::var("GLRMASK_COMPOSE_TERMINAL_EQUIV_TRANSITION_WORK_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(200_000);
    let prefix_depth = std::env::var("GLRMASK_COMPOSE_TERMINAL_EQUIV_PREFIX_DEPTH")
        .ok()
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(10);
    let certificate_state_limit =
        std::env::var("GLRMASK_COMPOSE_TERMINAL_CERTIFICATE_STATE_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(20_000);
    let certificate_transition_limit =
        std::env::var("GLRMASK_COMPOSE_TERMINAL_CERTIFICATE_TRANSITION_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(1_000_000);
    let semantic_fallback = std::env::var("GLRMASK_COMPOSE_TERMINAL_SEMANTIC_EQUIV")
        .ok()
        .is_some_and(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        });
    let mut structural_certificate_cache =
        FxHashMap::<(usize, u32), Option<Arc<[u64]>>>::default();
    let mut structural_matches = 0usize;
    let mut exact_checks = 0usize;
    let mut exact_unknown = 0usize;

    for (component_index, component) in components.iter().enumerate() {
        let offset = composed.terminal_offsets[component_index];
        for local_terminal in 0..component.tokenizer.num_terminals() {
            let global_terminal = offset + local_terminal;
            if global_terminal as usize >= num_terminals
                || ineligible.contains(global_terminal as usize)
            {
                continue;
            }
            // Reusing one compiled child artifact at multiple call sites is an
            // exact identity proof, including legacy loaded artifacts that do
            // not carry the v11 terminal-Expr sidecar.
            let artifact_key = (*component as *const Constraint as usize, local_terminal);
            let artifact_representative = representative_by_artifact_terminal
                .get(&artifact_key)
                .copied();
            let expr = component.retained_terminal_expr(local_terminal);
            let expr_representative = expr
                .and_then(|expr| representative_by_expr.get(expr).copied());
            let mut representative = artifact_representative.or(expr_representative);
            let prefilter = if needs_automata_fallback && representative.is_none() {
                terminal_equivalence_prefilter(
                    component,
                    local_terminal,
                    &token_group_fingerprints
                        .as_ref()
                        .expect("fallback fingerprints were prepared")[component_index],
                    prefix_depth,
                )
            } else {
                None
            };

            if representative.is_none()
                && let Some(prefilter) = prefilter.as_ref()
                && let Some(candidates) = representatives_by_prefilter.get(prefilter)
            {
                let current_key = (component_index, local_terminal);
                let current_certificate = if let Some(cached) =
                    structural_certificate_cache.get(&current_key)
                {
                    cached.clone()
                } else {
                    let built = component
                        .tokenizer
                        .terminal_scalar_structural_certificate(
                            local_terminal,
                            certificate_state_limit,
                            certificate_transition_limit,
                        )
                        .map(Arc::<[u64]>::from);
                    structural_certificate_cache.insert(current_key, built.clone());
                    built
                };
                for candidate in candidates {
                    let candidate_component = components[candidate.component_index];
                    let candidate_key = (candidate.component_index, candidate.local_terminal);
                    let candidate_certificate = if let Some(cached) =
                        structural_certificate_cache.get(&candidate_key)
                    {
                        cached.clone()
                    } else {
                        let built = candidate_component
                            .tokenizer
                            .terminal_scalar_structural_certificate(
                                candidate.local_terminal,
                                certificate_state_limit,
                                certificate_transition_limit,
                            )
                            .map(Arc::<[u64]>::from);
                        structural_certificate_cache.insert(candidate_key, built.clone());
                        built
                    };
                    if current_certificate.is_some()
                        && current_certificate == candidate_certificate
                    {
                        structural_matches += 1;
                        representative = Some(candidate.global_terminal);
                        break;
                    }
                    if semantic_fallback {
                        exact_checks += 1;
                        match component.tokenizer.terminal_language_equivalent_bounded(
                            local_terminal,
                            &candidate_component.tokenizer,
                            candidate.local_terminal,
                            pair_limit,
                            transition_work_limit,
                        ) {
                            Some(true) => {
                                representative = Some(candidate.global_terminal);
                                break;
                            }
                            Some(false) => {}
                            None => exact_unknown += 1,
                        }
                    }
                }
            }

            let representative = representative.unwrap_or(global_terminal);
            classes[global_terminal as usize] = representative;
            representative_by_artifact_terminal
                .entry(artifact_key)
                .or_insert(representative);
            if let Some(expr) = expr {
                representative_by_expr
                    .entry(expr.clone())
                    .or_insert(representative);
            }
            if representative == global_terminal
                && let Some(prefilter) = prefilter
            {
                representatives_by_prefilter
                    .entry(prefilter)
                    .or_default()
                    .push(TerminalRepresentative {
                        component_index,
                        local_terminal,
                        global_terminal,
                    });
            }
        }
    }
    TerminalClassAnalysis {
        classes,
        structural_matches,
        exact_checks,
        exact_unknown,
    }
}

fn dense_initial_nonterminal_classes(
    table: &crate::compiler::glr::table::GLRTable,
    boundary_nonterminals: &BTreeSet<NonterminalID>,
) -> Vec<u32> {
    let num_nonterminals = table.nonterminal_display_names.len();
    let augmented_start = table.rules.first().map(|rule| rule.lhs);
    let mut class_by_anchor = BTreeMap::<(bool, bool), u32>::new();
    let mut classes = Vec::with_capacity(num_nonterminals);
    for nonterminal in 0..num_nonterminals as u32 {
        let anchor = (
            Some(nonterminal) == augmented_start,
            boundary_nonterminals.contains(&nonterminal),
        );
        let next = class_by_anchor.len() as u32;
        let class = *class_by_anchor.entry(anchor).or_insert(next);
        classes.push(class);
    }
    classes
}

pub(super) fn structural_nonterminal_classes(
    table: &crate::compiler::glr::table::GLRTable,
    terminal_classes: &[u32],
    boundary_nonterminals: &BTreeSet<NonterminalID>,
) -> Vec<u32> {
    let started_at = compose_profile_enabled().then(Instant::now);
    let num_nonterminals = table.nonterminal_display_names.len();
    if num_nonterminals == 0 {
        return Vec::new();
    }
    let mut productions = vec![Vec::<Vec<Symbol>>::new(); num_nonterminals];
    for rule in &table.rules {
        if let Some(slot) = productions.get_mut(rule.lhs as usize) {
            slot.push(rule.rhs.clone());
        }
    }

    let mut classes = dense_initial_nonterminal_classes(table, boundary_nonterminals);
    loop {
        let signatures = (0..num_nonterminals)
            .into_par_iter()
            .map(|nonterminal| {
                let mut normalized_productions = productions[nonterminal]
                    .iter()
                    .map(|rhs| {
                        rhs.iter()
                            .map(|symbol| match *symbol {
                                Symbol::Terminal(terminal) => StructuralSymbolSignature::Terminal(
                                    terminal_classes
                                        .get(terminal as usize)
                                        .copied()
                                        .unwrap_or(terminal),
                                ),
                                Symbol::Nonterminal(child) => {
                                    StructuralSymbolSignature::Nonterminal(
                                        classes.get(child as usize).copied().unwrap_or(child),
                                    )
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                // Rule order is not part of CFG semantics. Preserve duplicate
                // productions so this remains at least as strict as structural
                // equality of the derivation graph.
                normalized_productions.sort();
                NonterminalRefinementSignature {
                    previous_class: classes[nonterminal],
                    productions: normalized_productions,
                }
            })
            .collect::<Vec<_>>();
        let mut class_by_signature = FxHashMap::<NonterminalRefinementSignature, u32>::default();
        let mut next_classes = Vec::with_capacity(num_nonterminals);
        for signature in signatures {
            let next = class_by_signature.len() as u32;
            let class = *class_by_signature.entry(signature).or_insert(next);
            next_classes.push(class);
        }
        if next_classes == classes {
            if let Some(started_at) = started_at {
                eprintln!(
                    "[glrmask/profile][constraint_structural_nonterminal_classes] nonterminals={} classes={} ms={:.3}",
                    num_nonterminals,
                    next_classes.iter().copied().max().map_or(0, |class| class as usize + 1),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return classes;
        }
        classes = next_classes;
    }
}

fn remap_action_for_structural_quotient(
    action: &Action,
    state_map: &[u32],
    nonterminal_map: &[u32],
) -> Action {
    let map_state = |state: u32| state_map.get(state as usize).copied().unwrap_or(state);
    let map_nonterminal = |nonterminal: u32| {
        nonterminal_map
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(nonterminal)
    };
    match action {
        Action::Shift(target, replace) => Action::Shift(map_state(*target), *replace),
        Action::ReplaceShifts(targets) => {
            let mut targets = targets.iter().map(|&target| map_state(target)).collect::<Vec<_>>();
            targets.sort_unstable();
            targets.dedup();
            Action::ReplaceShifts(targets.into())
        }
        Action::StackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| crate::compiler::glr::table::StackShift {
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| map_state(state)).collect(),
                })
                .collect::<Vec<_>>();
            shifts.sort_by(|left, right| {
                left.pop
                    .cmp(&right.pop)
                    .then_with(|| left.pushes.cmp(&right.pushes))
            });
            shifts.dedup();
            Action::StackShifts(shifts)
        }
        Action::GuardedStackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    let mut guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states = guard
                                .states
                                .iter()
                                .map(|&state| map_state(state))
                                .collect::<Vec<_>>();
                            states.sort_unstable();
                            states.dedup();
                            crate::compiler::glr::table::StackShiftGuard {
                                pop: guard.pop,
                                states,
                            }
                        })
                        .collect::<Vec<_>>();
                    guards.sort();
                    guards.dedup();
                    crate::compiler::glr::table::GuardedStackShift {
                        guards,
                        pop: shift.pop,
                        pushes: shift.pushes.iter().map(|&state| map_state(state)).collect(),
                    }
                })
                .collect::<Vec<_>>();
            shifts.sort();
            shifts.dedup();
            Action::GuardedStackShifts(shifts)
        }
        Action::Reduce(nonterminal, len) => {
            Action::Reduce(map_nonterminal(*nonterminal), *len)
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            let mut reduces = reduces
                .iter()
                .map(|&(nonterminal, len)| (map_nonterminal(nonterminal), len))
                .collect::<Vec<_>>();
            reduces.sort_unstable();
            reduces.dedup();
            Action::Split {
                shift: shift.map(|(target, replace)| (map_state(target), replace)),
                reduces,
                accept: *accept,
            }
        }
        Action::Accept => Action::Accept,
        Action::Skip => Action::Skip,
    }
}

fn guarded_state_memberships(
    table: &crate::compiler::glr::table::GLRTable,
) -> Vec<Vec<u32>> {
    let mut memberships = vec![Vec::<u32>::new(); table.num_states as usize];
    let mut guard_index = 0u32;
    for row in &table.action {
        for (_, action) in row.iter() {
            if let Action::GuardedStackShifts(shifts) = action {
                for shift in shifts {
                    for guard in &shift.guards {
                        for &state in &guard.states {
                            if let Some(slot) = memberships.get_mut(state as usize) {
                                slot.push(guard_index);
                            }
                        }
                        guard_index += 1;
                    }
                }
            }
        }
    }
    memberships
}

fn structural_state_signature(
    table: &crate::compiler::glr::table::GLRTable,
    state: u32,
    previous_classes: &[u32],
    terminal_classes: &[u32],
    nonterminal_classes: &[u32],
    guard_memberships: &[Vec<u32>],
) -> StructuralStateSignature {
    let state_index = state as usize;
    let mut alias_conflict = false;
    let mut action_by_terminal_class = BTreeMap::<u32, (Action, bool)>::new();
    for (terminal, action) in table.action[state_index].iter() {
        let terminal_class = terminal_classes
            .get(terminal as usize)
            .copied()
            .unwrap_or(terminal);
        let normalized = remap_action_for_structural_quotient(
            action,
            previous_classes,
            nonterminal_classes,
        );
        let forwarded = table.forwarded_shifts.contains(&(state, terminal));
        match action_by_terminal_class.entry(terminal_class) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((normalized, forwarded));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get() != &(normalized, forwarded) {
                    alias_conflict = true;
                }
            }
        }
    }

    let mut goto_by_nonterminal_class = BTreeMap::<u32, (u32, bool)>::new();
    for (&nonterminal, &(target, replace)) in table.goto[state_index].iter() {
        let nonterminal_class = nonterminal_classes
            .get(nonterminal as usize)
            .copied()
            .unwrap_or(nonterminal);
        let normalized = (
            previous_classes
                .get(target as usize)
                .copied()
                .unwrap_or(target),
            replace,
        );
        match goto_by_nonterminal_class.entry(nonterminal_class) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(normalized);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get() != &normalized {
                    alias_conflict = true;
                }
            }
        }
    }

    let mut advance = if table.advance.len() == table.num_states as usize {
        table.advance[state_index]
            .iter()
            .map(|terminal| {
                terminal_classes
                    .get(terminal)
                    .copied()
                    .unwrap_or(terminal as u32)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    advance.sort_unstable();
    advance.dedup();

    let mut wide_by_terminal_class = BTreeMap::<u32, Vec<u32>>::new();
    for descriptor in table
        .direct_regular_wide_frontiers
        .iter()
        .filter(|descriptor| descriptor.source_state == state)
    {
        let terminal_class = terminal_classes
            .get(descriptor.terminal as usize)
            .copied()
            .unwrap_or(descriptor.terminal);
        let mut targets = descriptor
            .target_states
            .iter()
            .map(|&target| {
                previous_classes
                    .get(target as usize)
                    .copied()
                    .unwrap_or(target)
            })
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        match wide_by_terminal_class.entry(terminal_class) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(targets);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get() != &targets {
                    alias_conflict = true;
                }
            }
        }
    }

    StructuralStateSignature {
        previous_class: previous_classes[state_index],
        alias_conflict_nonce: alias_conflict.then_some(state),
        guard_memberships: guard_memberships
            .get(state_index)
            .cloned()
            .unwrap_or_default(),
        actions: action_by_terminal_class
            .into_iter()
            .map(|(terminal, (action, forwarded))| (terminal, action, forwarded))
            .collect(),
        gotos: goto_by_nonterminal_class
            .into_iter()
            .map(|(nonterminal, (target, replace))| (nonterminal, target, replace))
            .collect(),
        advance,
        wide_frontiers: wide_by_terminal_class.into_iter().collect(),
    }
}

fn structural_state_classes(
    table: &crate::compiler::glr::table::GLRTable,
    terminal_classes: &[u32],
    nonterminal_map: &[u32],
) -> Vec<u32> {
    let num_states = table.num_states as usize;
    if num_states == 0 {
        return Vec::new();
    }
    // Guard membership is a direct observation of a stack-state identity.  By
    // colouring states with exact membership in every guard set before any
    // refinement, every final guard set is a union of quotient classes.
    let guard_memberships = guarded_state_memberships(table);
    // Parser state 0 is not merely another language-equivalent LR row: it is
    // the distinguished entry row consumed by later subgrammar composition.
    // Keep that compositional role in the initial colour so the ordinary
    // quotient can never identify it with an internal state.
    let mut classes = vec![0u32; num_states];
    classes[0] = 1;
    loop {
        let mut class_by_signature = FxHashMap::<StructuralStateSignature, u32>::default();
        let mut next_classes = Vec::with_capacity(num_states);
        for state in 0..num_states as u32 {
            let signature = structural_state_signature(
                table,
                state,
                &classes,
                terminal_classes,
                nonterminal_map,
                &guard_memberships,
            );
            let next = class_by_signature.len() as u32;
            let class = *class_by_signature.entry(signature).or_insert(next);
            next_classes.push(class);
        }
        if next_classes == classes {
            return classes;
        }
        classes = next_classes;
    }
}

/// Find state correspondences in the independently compiled child tables,
/// before caller-specific continuation terminals are spliced into their rows.
///
/// This is the crucial distinction from quotienting the already-composed LR
/// table. Two copies of the same submachine can have different return
/// lookaheads after linking even though their standalone machines are
/// structurally identical. The resulting groups are only *candidates*; the
/// table-level contextual quotient subsequently proves that the caller can be
/// recovered exactly from the predecessor state before accepting a merge.
fn component_structural_state_groups(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    composed: &ComposedTable,
    terminal_classes: &[u32],
    nonterminal_classes: &[u32],
) -> Vec<Vec<u32>> {
    if children.len() < 2 {
        return Vec::new();
    }

    let parent_nonterminals = parent.table.nonterminal_display_names.len() as u32;
    let mut child_nonterminal_offsets = Vec::with_capacity(children.len());
    let mut next_nonterminal = parent_nonterminals;
    for child in children {
        child_nonterminal_offsets.push(next_nonterminal);
        next_nonterminal += child.constraint.table.nonterminal_display_names.len() as u32;
    }

    let mut virtual_state_offsets = Vec::with_capacity(children.len());
    let mut next_virtual_state = 0u32;
    for child in children {
        virtual_state_offsets.push(next_virtual_state);
        next_virtual_state += child.constraint.table.num_states;
    }
    let virtual_state_count = next_virtual_state as usize;
    if virtual_state_count == 0 {
        return Vec::new();
    }

    let child_local_nonterminal_classes = children
        .iter()
        .enumerate()
        .map(|(component_index, child)| {
            let nonterminal_offset = child_nonterminal_offsets[component_index];
            (0..child.constraint.table.nonterminal_display_names.len())
                .map(|nonterminal| {
                    nonterminal_classes
                        .get(nonterminal_offset as usize + nonterminal)
                        .copied()
                        .unwrap_or(nonterminal_offset + nonterminal as u32)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Distinguish each component's start role from internal rows, but allow
    // starts from different components to be compared with one another.
    let mut classes = vec![0u32; virtual_state_count];
    for (component_index, child) in children.iter().enumerate() {
        if child.constraint.table.num_states != 0 {
            classes[virtual_state_offsets[component_index] as usize] = 1;
        }
    }

    loop {
        let signatures_by_component = children
            .par_iter()
            .enumerate()
            .map(|(component_index, child)| {
                let table = &child.constraint.table;
                let state_offset = virtual_state_offsets[component_index];
                let terminal_offset = composed.terminal_offsets[component_index + 1];
                let local_state_classes = &classes[state_offset as usize
                    ..state_offset as usize + table.num_states as usize];
                let local_nonterminal_classes = &child_local_nonterminal_classes[component_index];

                (0..table.num_states)
                    .map(|local_state| {
                        let mut actions = table.action[local_state as usize]
                            .iter()
                            .map(|(terminal, action)| {
                                let terminal_class = if terminal < table.num_terminals {
                                    terminal_classes
                                        .get((terminal_offset + terminal) as usize)
                                        .copied()
                                        .unwrap_or(terminal_offset + terminal)
                                } else {
                                    terminal
                                };
                                (
                                    terminal_class,
                                    component_action_signature_remapped(
                                        action,
                                        local_state_classes,
                                        local_nonterminal_classes,
                                    ),
                                    table.forwarded_shifts.contains(&(local_state, terminal)),
                                )
                            })
                            .collect::<Vec<_>>();
                        actions.sort();

                        let mut gotos = table.goto[local_state as usize]
                            .iter()
                            .map(|(&nonterminal, &(target, replace))| {
                                (
                                    local_nonterminal_classes
                                        .get(nonterminal as usize)
                                        .copied()
                                        .unwrap_or(nonterminal),
                                    local_state_classes[target as usize],
                                    replace,
                                )
                            })
                            .collect::<Vec<_>>();
                        gotos.sort_unstable();

                        let mut advance = if table.advance.len() == table.num_states as usize {
                            table.advance[local_state as usize]
                                .iter()
                                .map(|terminal| {
                                    if terminal < table.num_terminals as usize {
                                        terminal_classes
                                            .get(terminal_offset as usize + terminal)
                                            .copied()
                                            .unwrap_or(terminal_offset + terminal as u32)
                                    } else {
                                        terminal as u32
                                    }
                                })
                                .collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        };
                        advance.sort_unstable();
                        advance.dedup();

                        ComponentStateSignature {
                            previous_class: classes[(state_offset + local_state) as usize],
                            actions,
                            gotos,
                            advance,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut class_by_signature = FxHashMap::<ComponentStateSignature, u32>::default();
        let mut next_classes = vec![0u32; virtual_state_count];
        for (component_index, signatures) in signatures_by_component.into_iter().enumerate() {
            let state_offset = virtual_state_offsets[component_index];
            for (local_state, signature) in signatures.into_iter().enumerate() {
                let virtual_state = state_offset as usize + local_state;
                let next = class_by_signature.len() as u32;
                next_classes[virtual_state] =
                    *class_by_signature.entry(signature).or_insert(next);
            }
        }
        if next_classes == classes {
            break;
        }
        classes = next_classes;
    }

    let class_count = classes.iter().copied().max().map_or(0, |class| class + 1) as usize;
    let mut members = vec![Vec::<(usize, u32)>::new(); class_count];
    for (component_index, child) in children.iter().enumerate() {
        let state_offset = virtual_state_offsets[component_index];
        for local_state in 0..child.constraint.table.num_states {
            members[classes[(state_offset + local_state) as usize] as usize]
                .push((component_index, local_state));
        }
    }

    let mut out = Vec::new();
    for class_members in members {
        let distinct_components = class_members
            .iter()
            .map(|&(component, _)| component)
            .collect::<BTreeSet<_>>();
        if distinct_components.len() < 2 {
            continue;
        }
        let mut composed_states = Vec::<u32>::new();
        for (component_index, local_state) in class_members {
            if let Some(targets) = composed.state_relations[component_index + 1]
                .get(local_state as usize)
            {
                composed_states.extend_from_slice(targets);
            }
        }
        composed_states.sort_unstable();
        composed_states.dedup();
        if composed_states.len() >= 2 {
            out.push(composed_states);
        }
    }
    out
}

fn remap_composed_state_relations(composed: &mut ComposedTable, state_map: &[u32]) {
    for relation in &mut composed.state_relations {
        for targets in relation {
            for target in targets.iter_mut() {
                *target = state_map
                    .get(*target as usize)
                    .copied()
                    .unwrap_or(*target);
            }
            targets.sort_unstable();
            targets.dedup();
        }
    }
}

pub(super) fn contextually_share_composed_states(
    composed: &mut ComposedTable,
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_classes: &[u32],
    nonterminal_classes: &[u32],
) -> (usize, usize) {
    if terminal_classes
        .iter()
        .enumerate()
        .all(|(terminal, &class)| terminal as u32 == class)
    {
        return (0, 0);
    }
    let groups_started_at = compose_profile_enabled().then(Instant::now);
    let groups = component_structural_state_groups(
        parent,
        children,
        composed,
        terminal_classes,
        nonterminal_classes,
    );
    let groups_ms = groups_started_at.map_or(0.0, |started_at| {
        started_at.elapsed().as_secs_f64() * 1000.0
    });
    if groups.is_empty() {
        return (0, 0);
    }
    let before = composed.table.num_states as usize;
    let share_started_at = compose_profile_enabled().then(Instant::now);
    let state_map = composed
        .table
        .share_context_distinguishable_states_exact(&groups);
    let share_ms = share_started_at.map_or(0.0, |started_at| {
        started_at.elapsed().as_secs_f64() * 1000.0
    });
    let remap_started_at = compose_profile_enabled().then(Instant::now);
    remap_composed_state_relations(composed, &state_map);
    let remap_ms = remap_started_at.map_or(0.0, |started_at| {
        started_at.elapsed().as_secs_f64() * 1000.0
    });
    let after = composed.table.num_states as usize;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_contextual_structural_sharing] groups={} saved={} group_ms={groups_ms:.3} share_ms={share_ms:.3} relation_remap_ms={remap_ms:.3}",
            groups.len(),
            before.saturating_sub(after),
        );
    }
    (groups.len(), before.saturating_sub(after))
}

pub(super) fn quotient_composed_table_structurally(
    composed: &mut ComposedTable,
    terminal_analysis: &TerminalClassAnalysis,
    nonterminal_classes: &[u32],
) -> Result<StructuralSharingReport, String> {
    let states_before = composed.table.num_states as usize;
    let nonterminals_before = composed.table.nonterminal_display_names.len();
    if states_before <= 1 {
        return Ok(StructuralSharingReport {
            states_before,
            states_after: states_before,
            nonterminals_before,
            nonterminal_classes: nonterminals_before,
            ..StructuralSharingReport::default()
        });
    }

    let terminal_classes = &terminal_analysis.classes;
    let terminal_aliases = terminal_classes
        .iter()
        .enumerate()
        .filter(|&(terminal, &class)| terminal as u32 != class)
        .count();
    if terminal_aliases == 0 {
        return Ok(StructuralSharingReport {
            terminal_aliases,
            terminal_structural_matches: terminal_analysis.structural_matches,
            terminal_exact_checks: terminal_analysis.exact_checks,
            terminal_exact_unknown: terminal_analysis.exact_unknown,
            nonterminals_before,
            nonterminal_classes: nonterminals_before,
            states_before,
            states_after: states_before,
            ..StructuralSharingReport::default()
        });
    }

    let nonterminal_class_count = nonterminal_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    // This whole-table bisimulation is only a secondary cleanup after the
    // caller-sensitive quotient. On large linked tables it repeatedly scans
    // every row and can cost far more build time than the few additional
    // states it removes. Skipping it is semantics-preserving: it merely leaves
    // some exact states unmerged. Keep it for smaller tables where it is cheap,
    // with an environment override for measurement.
    let ordinary_max_states = std::env::var("GLRMASK_COMPOSE_ORDINARY_STRUCTURAL_MAX_STATES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4_096);
    if states_before > ordinary_max_states {
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_ordinary_structural_quotient] skipped=true states={} limit={} saved=0 state_classes_ms=0.000 materialize_ms=0.000",
                states_before,
                ordinary_max_states,
            );
        }
        return Ok(StructuralSharingReport {
            terminal_aliases,
            terminal_structural_matches: terminal_analysis.structural_matches,
            terminal_exact_checks: terminal_analysis.exact_checks,
            terminal_exact_unknown: terminal_analysis.exact_unknown,
            nonterminals_before,
            nonterminal_classes: nonterminal_class_count,
            states_before,
            states_after: states_before,
            ..StructuralSharingReport::default()
        });
    }
    // Structural nonterminal equivalence is intentionally *not* used by this
    // ordinary LR quotient. Equal nonterminal languages do not imply equal
    // goto behavior in one caller state. Keep concrete nonterminal identity
    // here; the stronger cross-child sharing path uses the structural relation
    // only to propose candidates and then preserves source behavior with
    // stack-context guards.
    let nonterminal_identity = (0..nonterminals_before as u32).collect::<Vec<_>>();
    let state_classes_started_at = compose_profile_enabled().then(Instant::now);
    let state_classes =
        structural_state_classes(&composed.table, &terminal_classes, &nonterminal_identity);
    let state_classes_ms = state_classes_started_at.map_or(0.0, |started_at| {
        started_at.elapsed().as_secs_f64() * 1000.0
    });
    let state_class_count = state_classes
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    if state_class_count == states_before {
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_ordinary_structural_quotient] saved=0 state_classes_ms={state_classes_ms:.3} materialize_ms=0.000",
            );
        }
        return Ok(StructuralSharingReport {
            terminal_aliases,
            terminal_structural_matches: terminal_analysis.structural_matches,
            terminal_exact_checks: terminal_analysis.exact_checks,
            terminal_exact_unknown: terminal_analysis.exact_unknown,
            nonterminals_before,
            nonterminal_classes: nonterminal_class_count,
            states_before,
            states_after: states_before,
            ..StructuralSharingReport::default()
        });
    }

    let mut members = vec![Vec::<u32>::new(); state_class_count];
    for (state, &class) in state_classes.iter().enumerate() {
        members[class as usize].push(state as u32);
    }

    let materialize_started_at = compose_profile_enabled().then(Instant::now);
    let old_action = std::mem::take(&mut composed.table.action);
    let old_goto = std::mem::take(&mut composed.table.goto);
    let old_advance = std::mem::take(&mut composed.table.advance);
    let has_advance = old_advance.len() == states_before;
    let mut new_action = Vec::<ActionRow>::with_capacity(state_class_count);
    let mut new_goto = Vec::<GotoRow>::with_capacity(state_class_count);
    let mut new_advance = has_advance
        .then(|| Vec::<BitSet>::with_capacity(state_class_count))
        .unwrap_or_default();

    for class_members in &members {
        let mut action_row = ActionRow::default();
        let mut goto_row = GotoRow::default();
        let mut advance_row = BitSet::new(composed.table.num_terminals as usize);
        for &old_state in class_members {
            for (terminal, action) in old_action[old_state as usize].iter() {
                let remapped = remap_action_for_structural_quotient(
                    action,
                    &state_classes,
                    &nonterminal_identity,
                );
                if let Some(previous) = action_row.insert(terminal, remapped.clone())
                    && previous != remapped
                {
                    return Err(format!(
                        "structural-sharing proof violation: quotient state has incompatible actions for terminal {terminal}",
                    ));
                }
            }
            for (&nonterminal, &(target, replace)) in old_goto[old_state as usize].iter() {
                let canonical_nonterminal = nonterminal;
                let remapped = (state_classes[target as usize], replace);
                if let Some(previous) = goto_row.insert(canonical_nonterminal, remapped)
                    && previous != remapped
                {
                    return Err(format!(
                        "structural-sharing proof violation: quotient state members {class_members:?} have incompatible gotos for nonterminal {canonical_nonterminal}; old_state={old_state} original_nonterminal={nonterminal} previous={previous:?} remapped={remapped:?}",
                    ));
                }
            }
            if has_advance {
                for terminal in old_advance[old_state as usize].iter() {
                    advance_row.set(terminal);
                }
            }
        }
        new_action.push(action_row);
        new_goto.push(goto_row);
        if has_advance {
            new_advance.push(advance_row);
        }
    }

    composed.table.action = new_action;
    composed.table.goto = new_goto;
    composed.table.advance = new_advance;
    composed.table.num_states = state_class_count as u32;

    composed.table.forwarded_shifts = composed
        .table
        .forwarded_shifts
        .iter()
        .map(|&(state, terminal)| (state_classes[state as usize], terminal))
        .collect();
    for descriptor in &mut composed.table.direct_regular_wide_frontiers {
        descriptor.source_state = state_classes[descriptor.source_state as usize];
        for target in &mut descriptor.target_states {
            *target = state_classes[*target as usize];
        }
        descriptor.target_states.sort_unstable();
        descriptor.target_states.dedup();
    }
    for relation in &mut composed.state_relations {
        for targets in relation {
            for target in targets.iter_mut() {
                *target = state_classes[*target as usize];
            }
            targets.sort_unstable();
            targets.dedup();
        }
    }
    composed.table.rebuild_guarded_shift_index();
    composed.table.compress_default_action_rows();
    let materialize_ms = materialize_started_at.map_or(0.0, |started_at| {
        started_at.elapsed().as_secs_f64() * 1000.0
    });
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_ordinary_structural_quotient] saved={} state_classes_ms={state_classes_ms:.3} materialize_ms={materialize_ms:.3}",
            states_before.saturating_sub(state_class_count),
        );
    }

    Ok(StructuralSharingReport {
        terminal_aliases,
        terminal_structural_matches: terminal_analysis.structural_matches,
        terminal_exact_checks: terminal_analysis.exact_checks,
        terminal_exact_unknown: terminal_analysis.exact_unknown,
        nonterminals_before,
        nonterminal_classes: nonterminal_class_count,
        states_before,
        states_after: state_class_count,
        ..StructuralSharingReport::default()
    })
}
