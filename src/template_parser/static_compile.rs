//! Compile a data-only parser into the ordinary static masking artifact.
//!
//! The lexer stage is deliberately conservative about parser properties: no
//! terminal colouring, no rule-derived follow exclusions and no inferred LR
//! admission facts. The supplied stack relations then make the parser product
//! exact. Possible-match and terminal equivalence are reconciled independently.

use super::*;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::constraint_possible_matches as pm;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::MappedArtifact;
use crate::compiler::stages::id_map_and_terminal_dwa as tdwa;
use crate::compiler::stages::parser_dwa::{
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table,
    normalize_weighted_stack_predicate_for_symbol_count,
};
use crate::compiler::stages::templates::Templates;
use crate::ds::bitset::BitSet;
use crate::runtime::ConstraintRuntimeBackend;
use glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Bound representation growth, not the accepted language. Exceeding any
/// bound is an explicit build error, never a truncation or runtime fallback.
#[derive(Default)]
struct ExpansionBudget {
    states: usize,
    edges: usize,
    subset_members: usize,
    work: usize,
}

impl ExpansionBudget {
    fn charge(&mut self, states: usize, edges: usize, members: usize, work: usize) -> Result<()> {
        self.states = self.states.saturating_add(states);
        self.edges = self.edges.saturating_add(edges);
        self.subset_members = self.subset_members.saturating_add(members);
        self.work = self.work.saturating_add(work);
        if self.states > 131_072
            || self.edges > 1_048_576
            || self.subset_members > 2_097_152
            || self.work > 33_554_432
        {
            return Err(Error::Compilation(
                "static template expansion exceeded its representation/work budget; the parser relation was not truncated".into(),
            ));
        }
        Ok(())
    }
}

/// Convert a phase graph to an ordinary action-word NFA. DEFAULT must be
/// resolved *before* any epsilon/subset union: merging default branches first
/// can let another branch's concrete edge shadow the wrong alternative.
fn concrete_action_nfa(
    split: &CommitTemplateDfas,
    symbols: u32,
    budget: &mut ExpansionBudget,
) -> Result<NFA> {
    let read_offset = split.pop.states.len();
    let push_offset = read_offset + split.read.states.len();
    let fixed = push_offset + split.push.states.len();
    budget.charge(fixed, 0, 0, fixed)?;
    let mut nfa = NFA::new_empty();
    nfa.states.resize_with(fixed, Default::default);
    nfa.start_states.push(split.pop.start_state);
    for (id, state) in split.pop.states.iter().enumerate() {
        nfa.states[id].is_accepting = state.is_accepting;
        for (&label, &target) in &state.transitions {
            if label == DEFAULT_LABEL {
                // Even a concrete edge into a dead state shadows DEFAULT.
                for symbol in 0..symbols {
                    budget.charge(0, 0, 0, 1)?;
                    if !state.transitions.contains_key(&(symbol as i32)) {
                        budget.charge(0, 1, 0, 0)?;
                        nfa.add_transition(id as u32, symbol as i32, target);
                    }
                }
            } else {
                budget.charge(0, 1, 0, 1)?;
                nfa.add_transition(id as u32, label, target);
            }
        }
        for (links, offset) in [
            (&split.pop_to_read, read_offset),
            (&split.pop_to_push, push_offset),
        ] {
            if let Some(target) = links.get(id).copied().flatten() {
                budget.charge(0, 1, 0, 1)?;
                nfa.add_epsilon(id as u32, offset as u32 + target);
            }
        }
    }
    for (id, state) in split.read.states.iter().enumerate() {
        let from = (read_offset + id) as u32;
        nfa.states[from as usize].is_accepting = state.is_accepting;
        for (&label, &target) in &state.transitions {
            budget.charge(1, 2, 0, 1)?;
            let restore = nfa.add_state();
            nfa.add_transition(from, label, restore);
            nfa.add_transition(
                restore,
                encode_negative_label(label as u32),
                read_offset as u32 + target,
            );
        }
        if let Some(target) = split.read_to_push.get(id).copied().flatten() {
            budget.charge(0, 1, 0, 1)?;
            nfa.add_epsilon(from, push_offset as u32 + target);
        }
    }
    for (id, state) in split.push.states.iter().enumerate() {
        let from = (push_offset + id) as u32;
        nfa.states[from as usize].is_accepting = state.is_accepting;
        for (&label, &target) in &state.transitions {
            budget.charge(0, 1, 0, 1)?;
            nfa.add_transition(from, label, push_offset as u32 + target);
        }
    }
    Ok(nfa)
}

/// Exact subset construction with explicit accounting before growing its
/// buckets, graph and retained subset keys. The ordinary unweighted compiler
/// assumes trusted finite inputs; the public data-only entry point must also
/// handle a small NFA whose deterministic representation is exponential.
fn bounded_determinize(nfa: &NFA, budget: &mut ExpansionBudget) -> Result<DFA> {
    fn closure(nfa: &NFA, seeds: &[u32], budget: &mut ExpansionBudget) -> Result<Vec<u32>> {
        let mut seen = BTreeSet::new();
        let mut pending = Vec::new();
        for &q in seeds {
            budget.charge(0, 0, 0, 1)?;
            if seen.insert(q) {
                pending.push(q);
            }
        }
        while let Some(q) = pending.pop() {
            for &to in &nfa.states[q as usize].epsilons {
                budget.charge(0, 0, 0, 1)?;
                if seen.insert(to) {
                    pending.push(to);
                }
            }
        }
        Ok(seen.into_iter().collect())
    }
    let start = closure(nfa, &nfa.start_states, budget)?;
    budget.charge(1, 0, start.len(), 0)?;
    let mut dfa = DFA::new();
    let mut known = FxHashMap::from_iter([(start.clone(), 0u32)]);
    let mut pending = VecDeque::from([(0u32, start)]);
    while let Some((id, subset)) = pending.pop_front() {
        let mut targets = BTreeMap::<i32, Vec<u32>>::new();
        for q in subset {
            dfa.states[id as usize].is_accepting |= nfa.states[q as usize].is_accepting;
            for (&label, destinations) in &nfa.states[q as usize].transitions {
                budget.charge(0, 0, 0, destinations.len())?;
                targets
                    .entry(label)
                    .or_default()
                    .extend_from_slice(destinations);
            }
        }
        for (label, seeds) in targets {
            let key = closure(nfa, &seeds, budget)?;
            let target = if let Some(&target) = known.get(&key) {
                target
            } else {
                budget.charge(1, 0, key.len(), 0)?;
                let target = dfa.add_state();
                known.insert(key.clone(), target);
                pending.push_back((target, key));
                target
            };
            budget.charge(0, 1, 0, 1)?;
            dfa.add_transition(id, label, target);
        }
    }
    Ok(dfa)
}

/// Compatibility context for the existing lexical compiler. It contains no
/// grammar rules, nonterminals, productions or parser automaton. Empty follow
/// certificates and global observation prohibit grammar-specific shortcuts;
/// all real parser semantics come exclusively from the supplied templates.
fn lexical_context(terminals: u32) -> AnalyzedGrammar {
    let mut protected = BitSet::new(terminals as usize);
    for terminal in 0..terminals {
        protected.set(terminal as usize);
    }
    AnalyzedGrammar {
        rules: Vec::new(),
        num_terminals: terminals,
        terminal_display_names: (0..terminals).map(|t| format!("terminal_{t}")).collect(),
        protected_shift_terminals: protected,
        num_nonterminals: 0,
        nonterminal_display_names: Vec::new(),
        residual_isolation_classes: BTreeMap::new(),
        requires_global_terminal_observation: true,
        direct_regular_automaton: None,
        nullable: BTreeSet::new(),
        first: Vec::new(),
        follow: Vec::new(),
        rules_by_lhs: Vec::new(),
    }
}

pub(super) fn compile(
    program: &ParserProgram,
    tokenizer: crate::automata::lexer::tokenizer::Tokenizer,
    ignore_terminal: Option<u32>,
    vocab: &Vocab,
) -> Result<crate::runtime::Constraint> {
    let context = lexical_context(program.parser.terminal_count);
    let mut budget = ExpansionBudget::default();
    let mut terminal_templates = BTreeMap::new();
    for (terminal, split) in program.templates.iter().enumerate() {
        let split = split
            .as_ref()
            .ok_or_else(|| Error::Compilation("missing custom terminal relation".into()))?;
        let nfa = concrete_action_nfa(split, program.parser.state_count, &mut budget)?;
        terminal_templates.insert(terminal as u32, bounded_determinize(&nfa, &mut budget)?);
    }
    let templates = Templates::from_terminal_dfas(terminal_templates);
    let (terminal, _, _) = tdwa::build_id_map_and_terminal_dwa(
        &tokenizer,
        vocab,
        &tdwa::types::TerminalColoring::identity(context.num_terminals as usize),
        false,
        ignore_terminal,
        &context,
        &BTreeMap::new(),
        None,
    );
    let (terminal_dwa, terminal_ids) = terminal.into_parts();
    let parser_dwa = match build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table(
        &terminal_dwa, context.num_terminals, &templates, true,
    ) {
        Some(mut signed) => {
            resolve_negative_codes_in_nwa(&mut signed, false);
            normalize_weighted_stack_predicate_for_symbol_count(program.parser.state_count, &signed)
        }
        None => DWA::new(terminal_ids.num_tsids(), terminal_ids.max_internal_token_id()),
    };
    let possible = pm::compute_constraint_possible_matches_for_vocab(
        &tokenizer,
        vocab,
        pm::ConstraintPossibleMatchesConfig::EAGER,
    );
    let mut mapped = MappedArtifact::from((
        MappedArtifact::new(parser_dwa, terminal_ids),
        possible.mapped_possible_matches,
    ));
    mapped.compact_dimensions();
    let ((parser_dwa, possible_matches), ids) = mapped.into_parts();
    let mut inner =
        crate::dynamic_constraint::DynamicConstraint::from_template_runtime_parts_unfinalized(
            tokenizer,
            context.terminal_display_names,
            ignore_terminal,
            program.templates.iter().cloned().collect(),
            Arc::clone(&program.parser),
            vocab,
            possible.runtime_dynamic_vocab.vocab,
        );
    inner.runtime_backend = ConstraintRuntimeBackend::Static;
    inner.parser_dwa = parser_dwa.share_exact_transition_rows_owned();
    inner.possible_matches = possible_matches;
    inner.possible_matches_complete = possible.complete;
    inner.state_to_internal_tsid = ids.tokenizer_states.original_to_internal.clone();
    inner.internal_tsid_to_states = ids.tokenizer_states.internal_to_originals_vecs();
    inner.state_internal_tsid_offsets = vec![u32::MAX];
    inner.original_token_to_internal = ids.vocab_tokens.original_to_internal.clone();
    inner.internal_token_to_tokens = ids.vocab_tokens.internal_to_originals_vecs();
    inner.internal_token_bytes =
        pm::build_internal_token_bytes_from_groups(vocab, &ids.vocab_tokens.internal_to_originals);
    inner.rebuild_runtime_caches();
    assert!(
        !inner.table.is_present(),
        "custom static compilation retained an LR table"
    );
    Ok(inner)
}
