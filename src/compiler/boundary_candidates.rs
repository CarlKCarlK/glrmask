//! Grammar-aware reusable candidate summaries for composition boundaries.
//!
//! The summary answers one deliberately narrow question in original model-token
//! coordinates: can a non-empty proper byte prefix of this model token reach a
//! component's public outward interface from some component-local history?
//! It is an upper bound. Every uncertainty widens; none is interpreted as an
//! empty language.

use std::collections::{BTreeMap, BTreeSet};

use crate::automata::lexer::Lexer;
use crate::grammar::flat::{Rule, Symbol, TerminalID};
use crate::runtime::{
    BoundaryCandidateFingerprint, BoundaryCandidateSummary, Constraint, OriginalTokenSet,
    SummaryPrecision, SummaryUnavailable,
};

// Version 4's r=2 child algebra could lose a validated nullable empty word.
// Its persisted exclusion certificates must widen rather than be reused.
const BOUNDARY_CANDIDATE_ALGORITHM_VERSION: u16 = 5;
const DEFAULT_MAX_FRONTIER_PAIRS: usize = 250_000;
// Frontier steps manipulate grammar-position sets rather than raw bytes, so
// one "step" is materially more expensive than a tokenizer transition. Keep
// the reusable-summary budget intentionally small: on large components the
// safe fallback (all multibyte model tokens) is preferable to making a later
// composition pay seconds to save a modest fraction of the vocabulary.
const DEFAULT_MAX_BYTE_STEPS: usize = 10_000;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BoundaryCandidateStats {
    pub(crate) input_tokens: usize,
    pub(crate) candidate_tokens: usize,
    pub(crate) history_positions: usize,
    pub(crate) initial_frontier_pairs: usize,
    pub(crate) byte_steps: usize,
    pub(crate) widened_subtrees: usize,
    pub(crate) peak_frontier_pairs: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Frontier {
    by_lexer_state: BTreeMap<u32, BTreeSet<usize>>,
}

impl Frontier {
    fn insert_controls(&mut self, lexer_state: u32, controls: &BTreeSet<usize>) {
        if controls.is_empty() {
            return;
        }
        self.by_lexer_state
            .entry(lexer_state)
            .or_default()
            .extend(controls.iter().copied());
    }

    fn pair_count(&self) -> usize {
        self.by_lexer_state.values().map(BTreeSet::len).sum()
    }

    fn is_empty(&self) -> bool {
        self.by_lexer_state.is_empty()
    }
}

struct GrammarMachine<'a> {
    rules: &'a [Rule],
    rule_offsets: Vec<usize>,
    position_rule: Vec<usize>,
    position_dot: Vec<usize>,
    entries: BTreeMap<u32, Vec<usize>>,
    returns: BTreeMap<u32, Vec<usize>>,
    outward_terminals: BTreeSet<TerminalID>,
    skip_terminals: BTreeSet<TerminalID>,
    control_terminals: BTreeSet<TerminalID>,
}

impl<'a> GrammarMachine<'a> {
    fn new(
        rules: &'a [Rule],
        outward_terminals: BTreeSet<TerminalID>,
        skip_terminals: BTreeSet<TerminalID>,
        control_terminals: BTreeSet<TerminalID>,
    ) -> Result<Self, SummaryUnavailable> {
        if rules.is_empty() {
            return Err(SummaryUnavailable::MissingGrammarMetadata);
        }
        let mut rule_offsets = Vec::with_capacity(rules.len());
        let mut position_rule = Vec::new();
        let mut position_dot = Vec::new();
        let mut entries = BTreeMap::<u32, Vec<usize>>::new();
        for (rule_index, rule) in rules.iter().enumerate() {
            let offset = position_rule.len();
            rule_offsets.push(offset);
            entries.entry(rule.lhs).or_default().push(offset);
            for dot in 0..=rule.rhs.len() {
                position_rule.push(rule_index);
                position_dot.push(dot);
            }
        }
        let mut returns = BTreeMap::<u32, Vec<usize>>::new();
        for (rule_index, rule) in rules.iter().enumerate() {
            for (dot, symbol) in rule.rhs.iter().enumerate() {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    returns
                        .entry(*nonterminal)
                        .or_default()
                        .push(rule_offsets[rule_index] + dot + 1);
                }
            }
        }
        Ok(Self {
            rules,
            rule_offsets,
            position_rule,
            position_dot,
            entries,
            returns,
            outward_terminals,
            skip_terminals,
            control_terminals,
        })
    }

    fn symbol_at(&self, position: usize) -> Option<&Symbol> {
        let rule_index = *self.position_rule.get(position)?;
        let dot = *self.position_dot.get(position)?;
        self.rules.get(rule_index)?.rhs.get(dot)
    }

    fn next_position(&self, position: usize) -> usize {
        position + 1
    }

    fn is_augmented_completion(&self, position: usize) -> bool {
        self.position_rule.get(position).copied() == Some(0)
            && self
                .position_dot
                .get(position)
                .copied()
                .is_some_and(|dot| dot == self.rules[0].rhs.len())
    }

    /// Epsilon-close parser control after a committed terminal. Public outward
    /// portals are reported and not traversed. Nonterminal return is deliberately
    /// context-insensitive (K=0): it is an over-approximation, never a pruning
    /// assumption.
    fn operational_closure(&self, seeds: impl IntoIterator<Item = usize>) -> (BTreeSet<usize>, bool) {
        let mut stable = BTreeSet::new();
        let mut pending = seeds.into_iter().collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        let mut outward = false;
        while let Some(position) = pending.pop() {
            if !seen.insert(position) {
                continue;
            }
            let rule_index = self.position_rule[position];
            let dot = self.position_dot[position];
            let rule = &self.rules[rule_index];
            if dot == rule.rhs.len() {
                if self.is_augmented_completion(position) {
                    outward = true;
                } else if let Some(continuations) = self.returns.get(&rule.lhs) {
                    pending.extend(continuations.iter().copied());
                }
                continue;
            }
            match &rule.rhs[dot] {
                Symbol::Nonterminal(nonterminal) => {
                    if let Some(entries) = self.entries.get(nonterminal) {
                        pending.extend(entries.iter().copied());
                    }
                }
                Symbol::Terminal(terminal) if self.outward_terminals.contains(terminal) => {
                    outward = true;
                }
                Symbol::Terminal(terminal) if self.control_terminals.contains(terminal) => {
                    pending.push(self.next_position(position));
                }
                Symbol::Terminal(_) => {
                    stable.insert(position);
                }
            }
        }
        (stable, outward)
    }

    /// Parser positions at which an arbitrary *later* model token may start.
    /// Previous ordinary terminals and unresolved slots are traversed as
    /// history edges. An unresolved slot therefore does not hide positions that
    /// become reachable after a previously completed child invocation.
    fn history_starts(&self) -> BTreeSet<usize> {
        let mut stable = BTreeSet::new();
        let mut pending = vec![self.rule_offsets[0]];
        let mut seen = BTreeSet::new();
        while let Some(position) = pending.pop() {
            if !seen.insert(position) {
                continue;
            }
            let rule_index = self.position_rule[position];
            let dot = self.position_dot[position];
            let rule = &self.rules[rule_index];
            if dot == rule.rhs.len() {
                if !self.is_augmented_completion(position) {
                    if let Some(continuations) = self.returns.get(&rule.lhs) {
                        pending.extend(continuations.iter().copied());
                    }
                }
                continue;
            }
            match &rule.rhs[dot] {
                Symbol::Nonterminal(nonterminal) => {
                    if let Some(entries) = self.entries.get(nonterminal) {
                        pending.extend(entries.iter().copied());
                    }
                }
                Symbol::Terminal(terminal) if self.control_terminals.contains(terminal) => {
                    pending.push(self.next_position(position));
                }
                Symbol::Terminal(terminal) if self.outward_terminals.contains(terminal) => {
                    // A prior token may already have traversed the child. Do not
                    // count the portal itself as a zero-offset candidate, but
                    // retain its grammar position as a possible token-start
                    // control. A real local Skip/IGNORE may consume positive
                    // bytes while the grammar remains parked immediately
                    // before this portal; that positive prefix then witnesses
                    // the first outward event. `initial_frontier` admits this
                    // seed only when the lexer can actually scan the expected
                    // terminal or a local skip from the raw state.
                    stable.insert(position);
                    pending.push(self.next_position(position));
                }
                Symbol::Terminal(_) => {
                    stable.insert(position);
                    pending.push(self.next_position(position));
                }
            }
        }
        stable
    }

    fn expected_terminal(&self, position: usize) -> Option<TerminalID> {
        match self.symbol_at(position) {
            Some(Symbol::Terminal(terminal)) => Some(*terminal),
            _ => None,
        }
    }

    fn advance_terminal(&self, position: usize, terminal: TerminalID) -> Option<usize> {
        (self.expected_terminal(position) == Some(terminal)).then(|| self.next_position(position))
    }
}

struct CandidateMachine<'a> {
    tokenizer: &'a crate::automata::lexer::tokenizer::Tokenizer,
    grammar: GrammarMachine<'a>,
    reset_states: Vec<u32>,
    max_frontier_pairs: usize,
    max_byte_steps: usize,
    history_positions: usize,
    initial_frontier_pairs: usize,
    byte_steps: usize,
    peak_frontier_pairs: usize,
}

impl<'a> CandidateMachine<'a> {
    #[inline]
    fn raw_can_consume_byte(&self, raw: u32, byte: u8) -> bool {
        if !self.tokenizer.state_has_epsilon_transitions(raw) {
            return self.tokenizer.state_first_bytes(raw).contains(byte);
        }
        self.tokenizer
            .execute_from_state_end_only(&[], raw)
            .iter()
            .any(|&state| self.tokenizer.state_first_bytes(state).contains(byte))
    }

    fn indexed_initial_seed(&mut self, starts: &BTreeSet<usize>) -> Result<Frontier, ()> {
        // Index grammar positions by the terminal they are waiting for.  The
        // previous implementation probed every raw lexer state against every
        // history position and therefore needed a pessimistic Cartesian-product
        // budget.  Large composed components could trip that budget even when
        // the *actual* compatible frontier was tiny.  A position has exactly
        // one expected terminal, so visiting only the state's live future
        // terminals constructs the identical relation without the quadratic
        // probe space.
        let mut positions_by_terminal = BTreeMap::<TerminalID, Vec<usize>>::new();
        for &position in starts {
            if let Some(terminal) = self.grammar.expected_terminal(position) {
                positions_by_terminal
                    .entry(terminal)
                    .or_default()
                    .push(position);
            }
        }

        let mut frontier = Frontier::default();
        let mut frontier_pairs = 0usize;
        let mut overflowed = false;
        for raw in 0..self.tokenizer.num_states() {
            let futures = self.tokenizer.possible_future_terminals(raw);
            let can_skip = self
                .grammar
                .skip_terminals
                .iter()
                .any(|&terminal| (terminal as usize) < futures.len() && futures.get(terminal as usize));
            let controls = if can_skip {
                // A local skip can consume positive bytes without advancing
                // the parser, so every grammar history position remains a
                // possible control while that skip is live.
                starts.clone()
            } else {
                let mut controls = BTreeSet::new();
                for terminal in self.tokenizer.possible_future_terminals_iter(raw) {
                    if let Some(positions) = positions_by_terminal.get(&terminal) {
                        controls.extend(positions.iter().copied());
                    }
                }
                controls
            };
            frontier_pairs = frontier_pairs.saturating_add(controls.len());
            if frontier_pairs > self.max_frontier_pairs {
                // Finish counting the exact logical relation without retaining
                // any more rows. This keeps the widening path memory-bounded
                // while making diagnostics distinguish "barely above cap"
                // from a genuinely huge initial relation.
                overflowed = true;
            }
            if !overflowed {
                frontier.insert_controls(raw, &controls);
            }
        }
        self.initial_frontier_pairs = frontier_pairs;
        self.peak_frontier_pairs = self.peak_frontier_pairs.max(frontier_pairs);
        if overflowed {
            return Err(());
        }
        self.observe_frontier(&frontier)?;
        Ok(frontier)
    }

    fn initial_frontier(&mut self) -> Result<Frontier, ()> {
        let starts = self.grammar.history_starts();
        self.history_positions = starts.len();
        let frontier = self.indexed_initial_seed(&starts)?;
        self.initial_frontier_pairs = frontier.pair_count();
        // Accepting lexer states may commit exactly at the model-token boundary.
        // Such an offset-zero outward event is not itself a candidate, but an
        // internal commit can change the parser/lexer state from which byte 0
        // of this token is consumed. Close those internal commits here.
        self.close_zero_commits(frontier)
    }

    fn observe_frontier(&mut self, frontier: &Frontier) -> Result<(), ()> {
        let pairs = frontier.pair_count();
        self.peak_frontier_pairs = self.peak_frontier_pairs.max(pairs);
        (pairs <= self.max_frontier_pairs && self.byte_steps <= self.max_byte_steps)
            .then_some(())
            .ok_or(())
    }

    fn close_zero_commits(&mut self, mut frontier: Frontier) -> Result<Frontier, ()> {
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot = frontier.by_lexer_state.clone();
            for (raw, controls) in snapshot {
                let matched = self.tokenizer.matched_terminals_iter(raw).collect::<Vec<_>>();
                if matched.is_empty() {
                    continue;
                }
                for terminal in matched {
                    let mut advanced = BTreeSet::new();
                    for &position in &controls {
                        if self.grammar.skip_terminals.contains(&terminal) {
                            // Skip consumes a lexer terminal without advancing
                            // the grammar symbol, but it can expose an outward
                            // portal (or another internal nullable/control
                            // path) at the same grammar position. Offset zero
                            // outward is intentionally not a candidate and is
                            // not traversed; retain any simultaneous internal
                            // stable branches conservatively.
                            let (closed, _outward) =
                                self.grammar.operational_closure([position]);
                            advanced.extend(closed);
                        }
                        if let Some(next) = self.grammar.advance_terminal(position, terminal) {
                            let (closed, outward) = self.grammar.operational_closure([next]);
                            // Outward at offset zero is deliberately ignored and
                            // not traversed. Internal closure remains live.
                            let _ = outward;
                            advanced.extend(closed);
                        }
                    }
                    if advanced.is_empty() {
                        continue;
                    }
                    for &reset in &self.reset_states {
                        let before = frontier
                            .by_lexer_state
                            .get(&reset)
                            .map_or(0, BTreeSet::len);
                        frontier.insert_controls(reset, &advanced);
                        let after = frontier
                            .by_lexer_state
                            .get(&reset)
                            .map_or(0, BTreeSet::len);
                        changed |= after != before;
                    }
                }
            }
            self.observe_frontier(&frontier)?;
        }
        Ok(frontier)
    }

    fn consume(&mut self, frontier: &Frontier, byte: u8) -> Result<(Frontier, bool), ()> {
        let mut next = Frontier::default();
        let mut outward = false;
        for (&raw, controls) in &frontier.by_lexer_state {
            // Charge only exact epsilon-closed state/byte transitions that can
            // do real work.  The old counter charged every frontier state for
            // every trie edge, so a 25k-state component exhausted a 10k budget
            // before its first byte even though almost all states were dead on
            // that byte.
            if !self.raw_can_consume_byte(raw, byte) {
                continue;
            }
            self.byte_steps = self.byte_steps.saturating_add(1);
            if self.byte_steps > self.max_byte_steps {
                return Err(());
            }
            let targets = self.tokenizer.step_all(&[raw], byte);
            for target in targets {
                // The terminal need not commit here; retain the ordinary
                // continuing lexer branch.
                next.insert_controls(target, controls);
                let matched = self
                    .tokenizer
                    .matched_terminals_iter(target)
                    .collect::<Vec<_>>();
                for terminal in matched {
                    let mut advanced = BTreeSet::new();
                    for &position in controls {
                        if self.grammar.skip_terminals.contains(&terminal) {
                            let (closed, did_outward) =
                                self.grammar.operational_closure([position]);
                            outward |= did_outward;
                            advanced.extend(closed);
                        }
                        if let Some(after_terminal) =
                            self.grammar.advance_terminal(position, terminal)
                        {
                            let (closed, did_outward) =
                                self.grammar.operational_closure([after_terminal]);
                            outward |= did_outward;
                            advanced.extend(closed);
                        }
                    }
                    for &reset in &self.reset_states {
                        next.insert_controls(reset, &advanced);
                    }
                }
            }
        }
        self.observe_frontier(&next)?;
        Ok((next, outward))
    }
}

fn fingerprint(
    constraint: &Constraint,
    vocab: &crate::Vocab,
    rules: &[Rule],
) -> Result<BoundaryCandidateFingerprint, SummaryUnavailable> {
    let vocabulary = crate::compiler::compile::vocab_content_digest(vocab);

    // A provider-native composed constraint intentionally does not retain one
    // flattened top-level terminal-expression array: its immutable semantic
    // object is the immediate component tree plus typed call links. Fingerprint
    // exactly that structure so a summary prepared before flattening survives
    // save/load and later compiler-view materialization without trusting an
    // opaque cache entry.
    if let Some(overlay) = constraint
        .static_dynamic_overlay
        .as_ref()
        .filter(|overlay| !overlay.segmented_parser_components.is_empty())
    {
        let mut semantics = blake3::Hasher::new();
        semantics.update(b"glrmask-boundary-component-semantics-v2-composite\0");
        semantics.update(&(overlay.segmented_parser_components.len() as u64).to_le_bytes());
        for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
            let child = fingerprint_for_constraint(component.constraint.as_ref(), vocab)?;
            semantics.update(&(component_index as u64).to_le_bytes());
            semantics.update(&child.component_semantics);
            semantics.update(&child.public_interface);
        }
        let mut links = overlay.segmented_parser_links.iter().collect::<Vec<_>>();
        links.sort_unstable_by_key(|link| {
            (
                link.parent_component,
                link.slot_terminal,
                link.child_component,
                link.child_start,
                link.return_pop,
                link.child_start_nullable,
            )
        });
        semantics.update(&(links.len() as u64).to_le_bytes());
        for link in &links {
            semantics.update(&link.parent_component.to_le_bytes());
            semantics.update(&link.slot_terminal.to_le_bytes());
            semantics.update(&link.child_component.to_le_bytes());
            semantics.update(&link.child_start.to_le_bytes());
            semantics.update(&link.return_pop.to_le_bytes());
            semantics.update(&[u8::from(link.child_start_nullable)]);
        }

        // Over-validation is intentional here: make the public-interface digest a
        // function of the complete immediate composition identity plus any still
        // explicitly exported slots on the wrapper. Binding an old exported slot
        // necessarily produces a new composition/fingerprint.
        let semantics_digest = *semantics.finalize().as_bytes();
        let mut interface = blake3::Hasher::new();
        interface.update(b"glrmask-boundary-public-interface-v2-composite\0");
        interface.update(&semantics_digest);
        for (name, terminal) in &constraint.unbound_grammar_placeholders {
            interface.update(&(name.len() as u64).to_le_bytes());
            interface.update(name.as_bytes());
            interface.update(&terminal.to_le_bytes());
        }
        let mut late = constraint
            .late_grammar_slots
            .iter()
            .map(|slot| (slot.name.as_str(), slot.terminal_id))
            .collect::<Vec<_>>();
        late.sort_unstable();
        for (name, terminal) in late {
            interface.update(&(name.len() as u64).to_le_bytes());
            interface.update(name.as_bytes());
            interface.update(&terminal.to_le_bytes());
        }

        return Ok(BoundaryCandidateFingerprint {
            algorithm_version: BOUNDARY_CANDIDATE_ALGORITHM_VERSION,
            component_semantics: semantics_digest,
            public_interface: *interface.finalize().as_bytes(),
            vocabulary,
        });
    }

    // Ordinary leaf component: hash its retained grammar + terminal
    // definitions exactly as before.
    if let Some(exprs) = constraint.retained_terminal_exprs() {
        let mut semantics = blake3::Hasher::new();
        semantics.update(b"glrmask-boundary-component-semantics-v2-leaf\0");
        semantics.update(
            &bincode::serialize(rules).map_err(|_| SummaryUnavailable::MalformedMetadata)?,
        );
        semantics.update(
            &bincode::serialize(exprs).map_err(|_| SummaryUnavailable::MalformedMetadata)?,
        );
        semantics.update(&constraint.ignore_terminal.unwrap_or(u32::MAX).to_le_bytes());
        for terminal in constraint.parser_skip_terminals() {
            semantics.update(&terminal.to_le_bytes());
        }
        for terminal in &constraint.parser_control_terminals() {
            semantics.update(&terminal.to_le_bytes());
        }
        let mut semantic_specials = constraint
            .special_token_terminals
            .iter()
            .map(|special| (special.terminal_id, special.token_id))
            .collect::<Vec<_>>();
        semantic_specials.sort_unstable();
        for (terminal, token) in semantic_specials {
            semantics.update(&terminal.to_le_bytes());
            semantics.update(&token.to_le_bytes());
        }

        let mut interface = blake3::Hasher::new();
        interface.update(b"glrmask-boundary-public-interface-v2-leaf\0");
        for (name, terminal) in &constraint.unbound_grammar_placeholders {
            interface.update(&(name.len() as u64).to_le_bytes());
            interface.update(name.as_bytes());
            interface.update(&terminal.to_le_bytes());
        }
        let mut late = constraint
            .late_grammar_slots
            .iter()
            .map(|slot| (slot.name.as_str(), slot.terminal_id))
            .collect::<Vec<_>>();
        late.sort_unstable();
        for (name, terminal) in late {
            interface.update(&(name.len() as u64).to_le_bytes());
            interface.update(name.as_bytes());
            interface.update(&terminal.to_le_bytes());
        }
        let mut specials = constraint
            .special_token_terminals
            .iter()
            .filter(|special| {
                constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                    || constraint.token_bytes_for_id(special.token_id).is_none()
            })
            .map(|special| (special.terminal_id, special.token_id))
            .collect::<Vec<_>>();
        specials.sort_unstable();
        for (terminal, token) in specials {
            interface.update(&terminal.to_le_bytes());
            interface.update(&token.to_le_bytes());
        }

        return Ok(BoundaryCandidateFingerprint {
            algorithm_version: BOUNDARY_CANDIDATE_ALGORITHM_VERSION,
            component_semantics: *semantics.finalize().as_bytes(),
            public_interface: *interface.finalize().as_bytes(),
            vocabulary,
        });
    }

    Err(SummaryUnavailable::MissingGrammarMetadata)
}

fn outward_terminals(constraint: &Constraint) -> BTreeSet<TerminalID> {
    constraint
        .unbound_grammar_placeholders
        .values()
        .copied()
        .chain(constraint.late_grammar_slots.iter().map(|slot| slot.terminal_id))
        .chain(
            constraint
                .special_token_terminals
                .iter()
                .filter(|special| {
                    constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                        || constraint.token_bytes_for_id(special.token_id).is_none()
                })
                .map(|special| special.terminal_id),
        )
        .collect()
}

fn add_descendant_ids(
    entries: &[(u32, &[u8])],
    prefix_len: usize,
    candidates: &mut BTreeSet<u32>,
) {
    candidates.extend(
        entries
            .iter()
            .filter_map(|(id, bytes)| (bytes.len() > prefix_len).then_some(*id)),
    );
}

fn scan_group(
    machine: &mut CandidateMachine<'_>,
    entries: &[(u32, &[u8])],
    depth: usize,
    frontier: &Frontier,
    candidates: &mut BTreeSet<u32>,
    widened_subtrees: &mut usize,
) {
    let mut index = 0usize;
    while index < entries.len() {
        while index < entries.len() && entries[index].1.len() <= depth {
            index += 1;
        }
        if index == entries.len() {
            return;
        }
        let byte = entries[index].1[depth];
        let start = index;
        index += 1;
        while index < entries.len()
            && entries[index].1.len() > depth
            && entries[index].1[depth] == byte
        {
            index += 1;
        }
        let group = &entries[start..index];
        match machine.consume(frontier, byte) {
            Err(()) => {
                *widened_subtrees += 1;
                add_descendant_ids(group, depth + 1, candidates);
            }
            Ok((next, outward)) => {
                let prefix_len = depth + 1;
                if outward {
                    add_descendant_ids(group, prefix_len, candidates);
                } else if !next.is_empty() {
                    scan_group(
                        machine,
                        group,
                        prefix_len,
                        &next,
                        candidates,
                        widened_subtrees,
                    );
                }
            }
        }
    }
}

fn compute_summary(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (BoundaryCandidateSummary, BoundaryCandidateStats) {
    let mut stats = BoundaryCandidateStats {
        input_tokens: vocab.len(),
        ..BoundaryCandidateStats::default()
    };
    let rules = match constraint.retained_table_rules() {
        Ok(rules) if !rules.is_empty() => rules,
        _ => {
            return (
                BoundaryCandidateSummary::Unknown {
                    reason: SummaryUnavailable::MissingGrammarMetadata,
                },
                stats,
            );
        }
    };
    let fp = match fingerprint(constraint, vocab, rules) {
        Ok(fp) => fp,
        Err(reason) => return (BoundaryCandidateSummary::Unknown { reason }, stats),
    };

    // First-principles interface-tail envelope.  This deliberately quantifies
    // away parser history and raw residual lexer identity: a real positive
    // proper-prefix boundary witness must have some final byte immediately
    // before the outward event, and this r=1 algebra computes a sound set of
    // such bytes while preserving known parent postambles through opaque
    // interiors.  It is both dramatically smaller and cheaper than the old
    // raw-state × grammar-position frontier on composed constraints.
    if let Ok(probe) = crate::compiler::boundary_tail::build_boundary_tail_r1(constraint, vocab) {
        let ids = probe.candidate_ids;
        stats.candidate_tokens = ids.len();
        stats.widened_subtrees = usize::from(probe.fixed_point_widened);
        let precision = if probe.fixed_point_widened {
            SummaryPrecision::BudgetWidenedUpperBound
        } else {
            SummaryPrecision::RegularUpperBound
        };
        return (
            BoundaryCandidateSummary::Known {
                fingerprint: fp,
                tokens: OriginalTokenSet::from_sorted_unique(ids, vocab.max_token_id()),
                precision,
            },
            stats,
        );
    }

    if constraint.uses_compact_segmented_parser_runtime() {
        match constraint.recursive_parser_layout() {
            Ok(Some(layout))
                if constraint.tokenizer.num_states() != layout.total_tokenizer_states =>
            {
                return (
                    BoundaryCandidateSummary::Unknown {
                        reason: SummaryUnavailable::Deferred,
                    },
                    stats,
                );
            }
            Err(_) => {
                return (
                    BoundaryCandidateSummary::Unknown {
                        reason: SummaryUnavailable::MalformedMetadata,
                    },
                    stats,
                );
            }
            Ok(_) => {}
        }
    }
    let mut local_skip_terminals = constraint.parser_skip_terminals().clone();
    if let Some(ignore) = constraint.ignore_terminal {
        local_skip_terminals.insert(ignore);
    }
    let grammar = match GrammarMachine::new(
        rules,
        outward_terminals(constraint),
        local_skip_terminals,
        constraint.parser_control_terminals(),
    ) {
        Ok(grammar) => grammar,
        Err(reason) => return (BoundaryCandidateSummary::Unknown { reason }, stats),
    };
    let reset_states = constraint
        .tokenizer
        .deterministic_reset_states()
        .into_iter()
        .collect::<Vec<_>>();
    let mut machine = CandidateMachine {
        tokenizer: &constraint.tokenizer,
        grammar,
        reset_states,
        max_frontier_pairs: DEFAULT_MAX_FRONTIER_PAIRS,
        max_byte_steps: DEFAULT_MAX_BYTE_STEPS,
        history_positions: 0,
        initial_frontier_pairs: 0,
        byte_steps: 0,
        peak_frontier_pairs: 0,
    };
    let initial = match machine.initial_frontier() {
        Ok(frontier) => frontier,
        Err(()) => {
            stats.widened_subtrees = 1;
            stats.candidate_tokens = vocab.iter().filter(|(_, bytes)| bytes.len() >= 2).count();
            stats.history_positions = machine.history_positions;
            stats.initial_frontier_pairs = machine.initial_frontier_pairs;
            stats.byte_steps = machine.byte_steps;
            stats.peak_frontier_pairs = machine.peak_frontier_pairs;
            return (
                BoundaryCandidateSummary::Known {
                    fingerprint: fp,
                    tokens: OriginalTokenSet::AllByteTokensAtLeastTwo,
                    precision: SummaryPrecision::BudgetWidenedUpperBound,
                },
                stats,
            );
        }
    };

    let mut entries = vocab.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| {
        left.1
            .cmp(right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut candidates = BTreeSet::new();
    let mut widened_subtrees = 0usize;
    if !initial.is_empty() {
        scan_group(
            &mut machine,
            &entries,
            0,
            &initial,
            &mut candidates,
            &mut widened_subtrees,
        );
    }
    let ids = candidates.into_iter().collect::<Vec<_>>();
    stats.candidate_tokens = ids.len();
    stats.history_positions = machine.history_positions;
    stats.initial_frontier_pairs = machine.initial_frontier_pairs;
    stats.byte_steps = machine.byte_steps;
    stats.peak_frontier_pairs = machine.peak_frontier_pairs;
    stats.widened_subtrees = widened_subtrees;
    let precision = if widened_subtrees == 0 {
        SummaryPrecision::RegularUpperBound
    } else {
        SummaryPrecision::BudgetWidenedUpperBound
    };
    (
        BoundaryCandidateSummary::Known {
            fingerprint: fp,
            tokens: OriginalTokenSet::from_sorted_unique(ids, vocab.max_token_id()),
            precision,
        },
        stats,
    )
}

pub(crate) fn fingerprint_for_constraint(
    constraint: &Constraint, vocab: &crate::Vocab,
) -> Result<BoundaryCandidateFingerprint, SummaryUnavailable> {
    if constraint.static_dynamic_overlay.as_ref()
        .is_some_and(|overlay| !overlay.segmented_parser_components.is_empty()) {
        // The composite digest is defined by validated immediate components and
        // typed links. Its branch does not consume flattened wrapper rules.
        if constraint.template_parser.as_ref().is_some_and(|parser| parser.link_grammar.is_some()) {
            return fingerprint(constraint, vocab, &[]);
        }
    }
    let rules = constraint.retained_table_rules()
        .map_err(|_| SummaryUnavailable::MissingGrammarMetadata)?;
    if rules.is_empty() { return Err(SummaryUnavailable::MissingGrammarMetadata); }
    fingerprint(constraint, vocab, rules)
}

/// The same immutable immediate inputs used by the eager native linker. This
/// recipe is issued only for a validated fresh composition, and is never read
/// from a wire artifact or inferred from an absent certificate.
#[derive(Debug)]
pub(crate) struct DeferredCompositionSummary {
    // Keep the auto-trait boundary finite for public ParserProgram users.
    // The concrete typed recipe still owns the immutable component inputs.
    recipe: Box<dyn DeferredBoundaryRecipe>,
    completed: std::sync::OnceLock<Option<BoundaryCandidateSummary>>,
}

trait DeferredBoundaryRecipe: std::fmt::Debug + Send + Sync {
    fn compute(&self) -> Option<BoundaryCandidateSummary>;
}

#[derive(Debug)]
struct NativeCompositionBoundaryRecipe {
    components: Vec<std::sync::Arc<Constraint>>,
    slots: Vec<Vec<u32>>,
    vocab: crate::Vocab,
    fingerprint: BoundaryCandidateFingerprint,
}

impl DeferredBoundaryRecipe for NativeCompositionBoundaryRecipe {
    fn compute(&self) -> Option<BoundaryCandidateSummary> {
        let bindings = self.slots.iter().enumerate().flat_map(|(child, slots)|
            slots.iter().map(move |&slot| (slot, self.components[child + 1].as_ref())))
            .collect::<Vec<_>>();
        let proof = crate::compiler::boundary_tail::build_composition_boundary_tail_r2(
            &self.components[0], &bindings, &self.vocab,
        ).map(|proof| (proof.candidate_ids, proof.fixed_point_widened))
            .or_else(|_| crate::compiler::boundary_tail::build_composition_boundary_tail_r1(
                &self.components[0], &bindings, &self.vocab,
            ).map(|proof| (proof.candidate_ids, proof.fixed_point_widened))).ok()?;
        let mut ids = proof.0;
        ids.sort_unstable(); ids.dedup();
        Some(BoundaryCandidateSummary::Known {
            fingerprint: self.fingerprint.clone(),
            tokens: OriginalTokenSet::from_sorted_unique(ids, self.vocab.max_token_id()),
            precision: if proof.1 { SummaryPrecision::BudgetWidenedUpperBound }
                else { SummaryPrecision::RegularUpperBound },
        })
    }
}

pub(crate) fn defer_composition_boundary_candidate_summary(
    constraint: &mut Constraint, vocab: &crate::Vocab,
    components: &[std::sync::Arc<Constraint>], slots: &[Vec<u32>],
) -> bool {
    let Ok(fingerprint) = fingerprint_for_constraint(constraint, vocab) else { return false; };
    let Some(grammar) = constraint.template_parser.as_ref()
        .and_then(|parser| parser.link_grammar.as_ref()) else { return false; };
    grammar.deferred_boundary_summary.set(std::sync::Arc::new(
        DeferredCompositionSummary { recipe: Box::new(NativeCompositionBoundaryRecipe {
            components: components.to_vec(), slots: slots.to_vec(), vocab: vocab.clone(), fingerprint }),
            completed: std::sync::OnceLock::new() })).is_ok()
}

pub(crate) fn materialize_deferred_composition_boundary_summary(constraint: &Constraint) {
    if constraint.boundary_candidate_summary.get().is_some() { return; }
    let Some(recipe) = constraint.template_parser.as_ref()
        .and_then(|parser| parser.link_grammar.as_ref())
        .and_then(|grammar| grammar.deferred_boundary_summary.get()) else { return; };
    let summary = recipe.completed.get_or_init(|| recipe.recipe.compute());
    if let Some(summary) = summary {
        let _ = constraint.boundary_candidate_summary.set(summary.clone());
    }
}

pub(crate) fn install_precomputed_boundary_candidate_ids(
    constraint: &mut Constraint,
    vocab: &crate::Vocab,
    ids: &[u32],
    widened: bool,
) -> Result<(), String> {
    let fp = fingerprint_for_constraint(constraint, vocab)
        .map_err(|reason| format!("cannot fingerprint precomputed boundary summary: {reason:?}"))?;
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let summary = BoundaryCandidateSummary::Known {
        fingerprint: fp,
        tokens: OriginalTokenSet::from_sorted_unique(ids, vocab.max_token_id()),
        precision: if widened {
            SummaryPrecision::BudgetWidenedUpperBound
        } else {
            SummaryPrecision::RegularUpperBound
        },
    };
    constraint
        .boundary_candidate_summary
        .set(summary)
        .map_err(|_| "boundary candidate summary was already installed".to_owned())
}

pub(crate) fn boundary_candidate_summary(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (BoundaryCandidateSummary, BoundaryCandidateStats) {
    materialize_deferred_composition_boundary_summary(constraint);
    let wanted = fingerprint_for_constraint(constraint, vocab).ok();
    if let Some(existing) = constraint.boundary_candidate_summary.get() {
        if wanted
            .as_ref()
            .is_some_and(|fp| existing.known_tokens_for(fp).is_some())
        {
            let count = match existing {
                BoundaryCandidateSummary::Known { tokens, .. } => tokens
                    .canonical_ids(vocab.iter())
                    .len(),
                BoundaryCandidateSummary::Unknown { .. } => 0,
            };
            return (
                existing.clone(),
                BoundaryCandidateStats {
                    input_tokens: vocab.len(),
                    candidate_tokens: count,
                    ..BoundaryCandidateStats::default()
                },
            );
        }
        if matches!(
            existing,
            BoundaryCandidateSummary::Unknown {
                reason: SummaryUnavailable::Disabled
            }
        ) {
            return (
                existing.clone(),
                BoundaryCandidateStats {
                    input_tokens: vocab.len(),
                    ..BoundaryCandidateStats::default()
                },
            );
        }
    }
    let (summary, stats) = compute_summary(constraint, vocab);
    // A loaded legacy artifact may retain an unchanged-resave byte cache.  Do
    // not install a newly computed summary behind that cache: doing so would
    // make the in-memory metadata differ from save() without a mutable cache
    // invalidation path.  Current artifacts already carry their summary; fresh
    // compiler-owned constraints have no cache here and may memoize normally.
    if constraint.serialized_artifact_cache.is_none() {
        let _ = constraint.boundary_candidate_summary.set(summary.clone());
    }
    (summary, stats)
}

pub(crate) fn restricted_boundary_vocab(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (crate::Vocab, BoundaryCandidateSummary, BoundaryCandidateStats) {
    let (summary, mut stats) = boundary_candidate_summary(constraint, vocab);
    let entries = match &summary {
        BoundaryCandidateSummary::Known { tokens, .. } => vocab
            .iter()
            .filter(|(id, bytes)| tokens.contains(*id, bytes))
            .map(|(id, bytes)| (id, bytes.to_vec()))
            .collect::<Vec<_>>(),
        BoundaryCandidateSummary::Unknown { .. } => vocab
            .iter()
            .map(|(id, bytes)| (id, bytes.to_vec()))
            .collect::<Vec<_>>(),
    };
    stats.candidate_tokens = entries.len();
    (crate::Vocab::new(entries), summary, stats)
}

pub(crate) fn prepare_boundary_candidate_summary(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> BoundaryCandidateStats {
    boundary_candidate_summary(constraint, vocab).1
}

/// Explicit component-only preparation for a mutable compiled artifact.
///
/// Unlike a read-only query, this entry point may replace deferred/stale
/// metadata and invalidate unchanged-resave bytes. It never sees a caller,
/// sibling component, or link graph. The same fingerprint and widening rules
/// remain authoritative; Disabled is preserved rather than silently enabled.
pub(crate) fn persist_boundary_candidate_summary(
    constraint: &mut Constraint,
    vocab: &crate::Vocab,
) -> BoundaryCandidateStats {
    let (summary, stats) = boundary_candidate_summary(constraint, vocab);
    constraint.serialized_artifact_cache = None;
    let _ = constraint.boundary_candidate_summary.take();
    constraint.boundary_candidate_summary.set(summary)
        .expect("exclusive mutable preparation leaves the summary cell empty");
    stats
}

/// Return the checked original-ID candidate set for one component. `None`
/// means the summary is unavailable and the caller must widen to the full
/// vocabulary; `Some(empty)` is a proved empty proper-prefix candidate set.
pub(crate) fn boundary_candidate_ids(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (Option<Vec<u32>>, BoundaryCandidateStats) {
    let (summary, stats) = boundary_candidate_summary(constraint, vocab);
    let ids = match summary {
        BoundaryCandidateSummary::Known { tokens, .. } => {
            Some(tokens.canonical_ids(vocab.iter()))
        }
        BoundaryCandidateSummary::Unknown { .. } => None,
    };
    (ids, stats)
}

/// Read a candidate certificate that was deliberately persisted while the
/// component still owned its grammar/table metadata. Template-parser
/// conversion physically drops that metadata, so a later composition cannot
/// re-fingerprint/recompute the summary. The component itself is immutable at
/// that point; vocabulary identity is still checked here before trusting the
/// persisted original-token set.
pub(crate) fn persisted_boundary_candidate_ids(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> Result<Option<Vec<u32>>, String> {
    fn public_interface_matches(
        constraint: &Constraint,
        fingerprint: &BoundaryCandidateFingerprint,
    ) -> bool {
        let composite = constraint
            .static_dynamic_overlay
            .as_ref()
            .is_some_and(|overlay| !overlay.segmented_parser_components.is_empty());
        let mut interface = blake3::Hasher::new();
        if composite {
            interface.update(b"glrmask-boundary-public-interface-v2-composite\0");
            interface.update(&fingerprint.component_semantics);
        } else {
            interface.update(b"glrmask-boundary-public-interface-v2-leaf\0");
        }
        for (name, terminal) in &constraint.unbound_grammar_placeholders {
            interface.update(&(name.len() as u64).to_le_bytes());
            interface.update(name.as_bytes());
            interface.update(&terminal.to_le_bytes());
        }
        let mut late = constraint
            .late_grammar_slots
            .iter()
            .map(|slot| (slot.name.as_str(), slot.terminal_id))
            .collect::<Vec<_>>();
        late.sort_unstable();
        for (name, terminal) in late {
            interface.update(&(name.len() as u64).to_le_bytes());
            interface.update(name.as_bytes());
            interface.update(&terminal.to_le_bytes());
        }
        if !composite {
            let mut specials = constraint
                .special_token_terminals
                .iter()
                .filter(|special| {
                    constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                        || constraint.token_bytes_for_id(special.token_id).is_none()
                })
                .map(|special| (special.terminal_id, special.token_id))
                .collect::<Vec<_>>();
            specials.sort_unstable();
            for (terminal, token) in specials {
                interface.update(&terminal.to_le_bytes());
                interface.update(&token.to_le_bytes());
            }
        }
        *interface.finalize().as_bytes() == fingerprint.public_interface
    }

    if !constraint.token_bytes_match_vocab(vocab) {
        return Ok(None);
    }
    let Some(summary) = constraint.retained_boundary_candidate_summary_for_compilation()? else {
        return Ok(None);
    };
    match summary {
        BoundaryCandidateSummary::Known { fingerprint, tokens, .. }
            if fingerprint.algorithm_version == BOUNDARY_CANDIDATE_ALGORITHM_VERSION
                && fingerprint.vocabulary
                    == crate::compiler::compile::vocab_content_digest(vocab)
                && public_interface_matches(constraint, &fingerprint) =>
        {
            let ids = tokens.canonical_ids(vocab.iter());
            let max_token_id = vocab.max_token_id();
            // Unique nonnegative IDs with this cardinality fill 0..=max exactly.
            // Sparse byte domains still require the original membership check;
            // exact-only IDs are excluded from both len() and max_token_id().
            let dense_byte_domain = vocab.len() as u64 == u64::from(max_token_id) + 1;
            if ids.windows(2).any(|pair| pair[0] >= pair[1])
                || ids.iter().any(|id| {
                    if dense_byte_domain { *id > max_token_id }
                    else { !vocab.entries_map().contains_key(id) }
                })
            {
                return Ok(None);
            }
            Ok(Some(ids))
        }
        BoundaryCandidateSummary::Unknown { .. } => Ok(None),
        BoundaryCandidateSummary::Known { .. } => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_indexed_initial_seed_matches_bruteforce(constraint: &Constraint) {
        let rules = constraint.retained_table_rules().unwrap();
        let mut local_skip_terminals = constraint.parser_skip_terminals().clone();
        if let Some(ignore) = constraint.ignore_terminal {
            local_skip_terminals.insert(ignore);
        }
        let grammar = GrammarMachine::new(
            rules,
            outward_terminals(constraint),
            local_skip_terminals,
            constraint.parser_control_terminals(),
        )
        .unwrap();
        let reset_states = constraint
            .tokenizer
            .deterministic_reset_states()
            .into_iter()
            .collect::<Vec<_>>();
        let mut machine = CandidateMachine {
            tokenizer: &constraint.tokenizer,
            grammar,
            reset_states,
            max_frontier_pairs: usize::MAX,
            max_byte_steps: usize::MAX,
            history_positions: 0,
            initial_frontier_pairs: 0,
            byte_steps: 0,
            peak_frontier_pairs: 0,
        };
        let starts = machine.grammar.history_starts();
        let indexed = machine.indexed_initial_seed(&starts).unwrap();

        let mut reference = Frontier::default();
        for raw in 0..machine.tokenizer.num_states() {
            let futures = machine.tokenizer.possible_future_terminals(raw);
            let can_skip = machine
                .grammar
                .skip_terminals
                .iter()
                .any(|&terminal| {
                    (terminal as usize) < futures.len() && futures.get(terminal as usize)
                });
            let controls = starts
                .iter()
                .filter_map(|&position| {
                    let expected = machine.grammar.expected_terminal(position)?;
                    (((expected as usize) < futures.len() && futures.get(expected as usize))
                        || can_skip)
                        .then_some(position)
                })
                .collect::<BTreeSet<_>>();
            reference.insert_controls(raw, &controls);
        }
        assert_eq!(indexed, reference);
    }

    #[test]
    fn indexed_initial_frontier_matches_bruteforce_with_and_without_skip() {
        let vocab = crate::Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b" ".to_vec()),
            (3, b"ab".to_vec()),
            (4, b" a".to_vec()),
        ]);
        let plain = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "a" SUB | "b" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        assert_indexed_initial_seed_matches_bruteforce(&plain);

        let ignored = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "a" SUB | "b" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        assert_indexed_initial_seed_matches_bruteforce(&ignored);
    }

    #[test]
    fn proper_prefix_widening_excludes_token_ending_at_prefix() {
        let entries = vec![
            (10u32, b"ab".as_slice()),
            (11u32, b"abc".as_slice()),
            (12u32, b"abd".as_slice()),
            (13u32, b"ab".as_slice()),
        ];
        let mut out = BTreeSet::new();
        add_descendant_ids(&entries, 2, &mut out);
        assert_eq!(out, BTreeSet::from([11, 12]));
    }

    #[test]
    fn root_completion_requires_proper_prefix_and_preserves_duplicate_ids() {
        let vocab = crate::Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, stats) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1, 2]));
        assert_eq!(stats.input_tokens, 4);
        assert_eq!(stats.candidate_tokens, 2);
    }

    #[test]
    fn live_out_of_vocab_special_slot_is_an_outward_portal() {
        let vocab = crate::Vocab::new(vec![
            (0, b"m".to_vec()),
            (1, b"mg".to_vec()),
            (2, b"mx".to_vec()),
            (3, b"g".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(998);
                nt document ::= "m" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1, 2]));
    }

    #[test]
    fn stale_known_empty_fingerprint_is_recomputed_not_trusted() {
        let vocab = crate::Vocab::new(vec![
            (0, b"m".to_vec()),
            (1, b"mg".to_vec()),
            (2, b"mx".to_vec()),
            (3, b"g".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(998);
                nt document ::= "m" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        constraint
            .boundary_candidate_summary
            .set(BoundaryCandidateSummary::Known {
                fingerprint: BoundaryCandidateFingerprint {
                    algorithm_version: BOUNDARY_CANDIDATE_ALGORITHM_VERSION,
                    component_semantics: [0; 32],
                    public_interface: [0; 32],
                    vocabulary: [0; 32],
                },
                tokens: OriginalTokenSet::Empty,
                precision: SummaryPrecision::RegularUpperBound,
            })
            .unwrap();

        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(
            ids,
            Some(vec![1, 2]),
            "fingerprint mismatch must widen/recompute instead of trusting stale empty",
        );
    }

    #[test]
    fn leading_component_ignore_can_precede_positive_outward_cut() {
        let vocab = crate::Vocab::new(vec![
            (0, b" X a".to_vec()),
            (1, b"X a".to_vec()),
            (2, b" ".to_vec()),
            (3, b"X".to_vec()),
            (4, b"a".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB;
            "#,
            &vocab,
        )
        .unwrap();

        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        let ids = ids.expect("component summary should be known");
        assert!(ids.contains(&0), "leading ignore must not hide the later outward cut");
        assert!(ids.contains(&1), "direct outward-prefix token must remain a candidate");
    }

    #[test]
    fn ignore_commit_can_expose_outward_cut_before_next_byte() {
        let vocab = crate::Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" \ta".to_vec()),
            (2, b" ".to_vec()),
            (3, b"\t".to_vec()),
            (4, b"a".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB;
            "#,
            &vocab,
        )
        .unwrap();

        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert!(
            ids.expect("component summary should be known").contains(&1),
            "committing WS after its positive-byte prefix must expose SUB before the following byte",
        );
    }

    #[test]
    fn multiple_local_terminal_commits_can_precede_outward_cut() {
        let vocab = crate::Vocab::new(vec![
            (0, b"ab".to_vec()),
            (1, b"abx".to_vec()),
            (2, b"ax".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "a" "b" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1]));
    }

    #[test]
    fn utf8_literal_prefix_uses_byte_offsets_without_rounding() {
        let vocab = crate::Vocab::new(vec![
            (0, "é".as_bytes().to_vec()),
            (1, "éx".as_bytes().to_vec()),
            (2, b"x".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "é" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1]));
    }
    #[test]
    fn composed_interface_tail_summary_survives_save_load() {
        use crate::compiler::constraint_compose::{
            CompiledSubgrammarInput, SegmentedBoundaryBackend,
            compose_constraints_owned_parent_segmented,
        };

        let vocab = crate::Vocab::new(vec![
            (0, b")x".to_vec()),
            (1, b"})x".to_vec()),
            (2, b"a)x".to_vec()),
            (3, b")".to_vec()),
            (4, b"x)".to_vec()),
            (5, b"}".to_vec()),
            (6, b"p(".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "p(" SUB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "}";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "SUB")
            .expect("SUB terminal") as u32;
        let direct_r2 = crate::compiler::boundary_tail::build_composition_boundary_tail_r2(
            &parent,
            &[(slot, &child)],
            &vocab,
        )
        .unwrap();
        assert_eq!(direct_r2.candidate_ids, vec![0, 1]);
        let composed = compose_constraints_owned_parent_segmented(
            parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: slot,
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;

        let (fresh, _) = boundary_candidate_ids(&composed, &vocab);
        assert_eq!(fresh, Some(vec![0, 1]));
        let mut loaded = Constraint::load(&composed.save()).unwrap();
        assert!(
            loaded.boundary_candidate_summary.get().is_none(),
            "current artifacts deliberately defer composition-link metadata",
        );
        loaded
            .materialize_composition_link_metadata_for_compilation()
            .unwrap();
        let (reloaded, stats) = boundary_candidate_ids(&loaded, &vocab);
        assert_eq!(reloaded, fresh);
        assert_eq!(stats.candidate_tokens, 2);
    }

}

#[cfg(test)]
mod preparation_tests {
    use super::*;

    fn fixture() -> (Constraint, crate::Vocab) {
        let vocab=crate::Vocab::new(vec![(0,b"a".to_vec()),(1,b"ab".to_vec()),
            (2,b"ax".to_vec()),(3,b"b".to_vec()),(4,b"ba".to_vec())]);
        let component=Constraint::from_glrm_grammar(r#"
            start document; t SUB ::= @token(999);
            nt document ::= "a" SUB | "b" SUB;
        "#,&vocab).unwrap();
        (component,vocab)
    }

    #[test]
    fn explicit_preparation_persists_checked_summary_after_reload() {
        let (mut source,vocab)=fixture();
        source.boundary_candidate_summary.take();
        source.boundary_candidate_summary.set(BoundaryCandidateSummary::Unknown {
            reason:SummaryUnavailable::Deferred,
        }).unwrap();
        let original=source.save();
        let expected=boundary_candidate_ids(&source,&vocab).0.unwrap();
        let mut loaded=Constraint::load_with_vocab(&original,&vocab).unwrap();
        loaded.prepare_for_composition_internal(&vocab).unwrap();
        assert!(loaded.boundary_candidate_summary.get().unwrap().is_known());
        assert_eq!(boundary_candidate_ids(&loaded,&vocab).0,Some(expected.clone()));
        assert!(loaded.serialized_artifact_cache.is_none());
        let saved=loaded.save();
        let mut reloaded=Constraint::load_with_vocab(&saved,&vocab).unwrap();
        reloaded.materialize_composition_link_metadata_for_compilation().unwrap();
        assert!(reloaded.boundary_candidate_summary.get().unwrap().is_known());
        assert_eq!(boundary_candidate_ids(&reloaded,&vocab).0,Some(expected));
        assert_eq!(reloaded.save(),saved,"unchanged reload must retain the newly prepared artifact");
    }

    #[test]
    fn explicit_preparation_preserves_disabled_policy_after_reload() {
        let (mut source,vocab)=fixture();
        source.boundary_candidate_summary.take();
        source.boundary_candidate_summary.set(BoundaryCandidateSummary::Unknown {
            reason:SummaryUnavailable::Disabled,
        }).unwrap();
        let mut loaded=Constraint::load_with_vocab(&source.save(),&vocab).unwrap();
        loaded.prepare_for_composition_internal(&vocab).unwrap();
        assert!(matches!(loaded.boundary_candidate_summary.get(),Some(BoundaryCandidateSummary::Unknown{reason:SummaryUnavailable::Disabled})));
        let mut reloaded=Constraint::load_with_vocab(&loaded.save(),&vocab).unwrap();
        reloaded.materialize_composition_link_metadata_for_compilation().unwrap();
        assert!(matches!(reloaded.boundary_candidate_summary.get(),Some(BoundaryCandidateSummary::Unknown{reason:SummaryUnavailable::Disabled})));
        assert!(boundary_candidate_ids(&reloaded,&vocab).0.is_none());
    }

    #[test]
    fn persisted_candidates_survive_physical_table_removal_and_reload() {
        let vocab = crate::Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ax".to_vec()),
            (3, b"b".to_vec()),
            (4, b"ba".to_vec()),
        ]);
        for optimization in [crate::Optimization::FastRuntime, crate::Optimization::FastBuild] {
            let source = crate::Grammar::from_glrm(
                r#"glrm 1; start document; nt document = "a";"#,
            )
            .compile_with(
                &vocab,
                crate::BuildOptions::default()
                    .optimization(optimization)
                    .parser_backend(crate::ParserBackend::TemplateDfa),
            )
            .unwrap();
            assert!(!source.table.is_present());
            let expected = persisted_boundary_candidate_ids(&source, &vocab)
                .unwrap()
                .expect("public template compilation must persist its boundary certificate");
            assert!(!expected.is_empty());
            assert!(expected.len() < vocab.len());

            let loaded = Constraint::load(&source.save()).unwrap();
            assert!(!loaded.table.is_present());
            assert!(loaded.boundary_candidate_summary.get().is_none());
            assert!(loaded.deferred_composition_metadata_blob.is_some());
            assert_eq!(persisted_boundary_candidate_ids(&loaded, &vocab).unwrap(), Some(expected));
            assert!(loaded.boundary_candidate_summary.get().is_none(),
                "read-only proof lookup must not materialize link metadata");
            assert!(loaded.deferred_composition_metadata_blob.is_some(),
                "read-only proof lookup must retain deferred compiler metadata");
        }
    }

    #[test]
    fn persisted_pre_nullable_fix_proof_is_rejected_before_composition() {
        let vocab = crate::Vocab::new(vec![(0, b"X[a]!".to_vec()),
            (1, b"a]!".to_vec()), (2, b"[]!".to_vec()), (3, b"X[]!".to_vec())]);
        let options = crate::BuildOptions::default().optimization(crate::Optimization::FastBuild)
            .parser_backend(crate::ParserBackend::TemplateDfa);
        let leaf = crate::Grammar::from_glrm(r#"glrm 1; start value; nt value = "a"?;"#)
            .compile_with(&vocab, options.clone()).unwrap();
        let mut middle = crate::Grammar::from_glrm(r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#)
            .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap()
            .link_with(options.clone()).unwrap();
        materialize_deferred_composition_boundary_summary(&middle);
        let BoundaryCandidateSummary::Known { mut fingerprint, precision, .. } =
            middle.boundary_candidate_summary.take().unwrap() else { panic!("expected known proof"); };
        // Version 4 could omit []! because it lost the nullable leaf's empty
        // ordinary word. Preserve that historical certificate through save/load.
        fingerprint.algorithm_version = 4;
        middle.boundary_candidate_summary.set(BoundaryCandidateSummary::Known {
            fingerprint, tokens: OriginalTokenSet::Sparse(std::sync::Arc::from([0, 1])), precision,
        }).unwrap();
        middle.serialized_artifact_cache = None;
        let loaded = Constraint::load_with_vocab(middle.save(), &vocab).unwrap();
        assert!(persisted_boundary_candidate_ids(&loaded, &vocab).unwrap().is_none(),
            "pre-fix nullable candidate proof must widen before it can exclude []!");
        let root = crate::Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "X" middle "!";"#)
            .compile_unlinked(&vocab).unwrap().bind("middle", &loaded).unwrap()
            .link_with(options).unwrap();
        let mut state = root.start(); state.commit_bytes(b"X").unwrap();
        assert_ne!(state.mask()[0] & (1 << 2), 0);
        state.commit_token(2).unwrap(); assert!(state.is_accepting());
    }

    #[test]
    fn persisted_candidates_reject_stale_or_malformed_proofs_but_keep_known_empty() {
        let (mut source, vocab) = fixture();
        persist_boundary_candidate_summary(&mut source, &vocab);
        let original = source.boundary_candidate_summary.take().unwrap();
        let BoundaryCandidateSummary::Known {
            fingerprint: original_fingerprint,
            tokens: original_tokens,
            precision,
        } = original else {
            panic!("fixture should produce a known certificate")
        };

        let install = |constraint: &mut Constraint,
                       fingerprint: BoundaryCandidateFingerprint,
                       tokens: OriginalTokenSet| {
            let _ = constraint.boundary_candidate_summary.take();
            constraint
                .boundary_candidate_summary
                .set(BoundaryCandidateSummary::Known {
                    fingerprint,
                    tokens,
                    precision,
                })
                .unwrap();
        };

        let mut stale_algorithm = original_fingerprint;
        stale_algorithm.algorithm_version = BOUNDARY_CANDIDATE_ALGORITHM_VERSION.wrapping_sub(1);
        install(&mut source, stale_algorithm, original_tokens.clone());
        assert!(persisted_boundary_candidate_ids(&source, &vocab).unwrap().is_none());

        let mut stale_vocab = original_fingerprint;
        stale_vocab.vocabulary[0] ^= 0x80;
        install(&mut source, stale_vocab, original_tokens.clone());
        assert!(persisted_boundary_candidate_ids(&source, &vocab).unwrap().is_none());

        install(&mut source, original_fingerprint, original_tokens.clone());
        source
            .unbound_grammar_placeholders
            .insert("changed-after-proof".to_owned(), 0);
        assert!(persisted_boundary_candidate_ids(&source, &vocab).unwrap().is_none());
        source.unbound_grammar_placeholders.remove("changed-after-proof");

        install(
            &mut source,
            original_fingerprint,
            OriginalTokenSet::Sparse(std::sync::Arc::from([u32::MAX])),
        );
        assert!(persisted_boundary_candidate_ids(&source, &vocab).unwrap().is_none());

        let empty_vocab = crate::Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let mut empty = Constraint::from_glrm_grammar(
            r#"glrm 1; start document; nt document = "a";"#,
            &empty_vocab,
        )
        .unwrap();
        persist_boundary_candidate_summary(&mut empty, &empty_vocab);
        assert_eq!(persisted_boundary_candidate_ids(&empty, &empty_vocab).unwrap(), Some(vec![]));
    }

    #[test]
    fn persisted_candidates_reject_sparse_byte_domain_holes_even_when_exact_only() {
        let vocab = crate::Vocab::new_with_exact_token_ids(
            vec![(0, b"a".to_vec()), (2, b"ab".to_vec())], [1],
        );
        let mut source = Constraint::from_glrm_grammar(
            r#"glrm 1; start document; nt document = "a";"#, &vocab,
        ).unwrap();
        persist_boundary_candidate_summary(&mut source, &vocab);
        assert_eq!(persisted_boundary_candidate_ids(&source, &vocab).unwrap(), Some(vec![2]));
        assert!(vocab.contains_exact_token_id(1));
        let mut summary = source.boundary_candidate_summary.take().unwrap();
        let BoundaryCandidateSummary::Known { tokens, .. } = &mut summary else {
            panic!("expected a checked component certificate");
        };
        *tokens = OriginalTokenSet::Sparse(std::sync::Arc::from([1]));
        source.boundary_candidate_summary.set(summary).unwrap();
        assert!(persisted_boundary_candidate_ids(&source, &vocab).unwrap().is_none(),
            "an exact-only control ID is not a byte-backed boundary candidate");
    }
}
