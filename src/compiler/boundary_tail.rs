//! Compositional fixed-vocabulary interface-tail summaries.
//!
//! A reusable boundary whitelist only asks whether some positive proper prefix
//! of a model token can reach this component's current public interface.  The
//! history before the token is existential, so carrying `(raw lexer state,
//! parser history position)` through every byte preserves far more information
//! than that observation requires.
//!
//! This module instead propagates typed normal-return/public-exit tails through
//! the grammar.  The production hierarchy currently keeps one-byte tails for
//! ordinary leaf preparation and a two-byte compositional refinement. Unknown
//! interiors widen locally; known parent postambles remain exact.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rayon::prelude::*;

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::tokenizer::Lexer;
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};
use crate::runtime::Constraint;

const DEFAULT_FIXED_POINT_LIMIT: usize = 256;
const FINITE_EXPR_WORD_CAP: usize = 4096;

fn finite_literal_words(expr: &Expr, cap: usize) -> Option<Vec<Vec<u8>>> {
    fn combine(left: Vec<Vec<u8>>, right: Vec<Vec<u8>>, cap: usize) -> Option<Vec<Vec<u8>>> {
        if left.len().saturating_mul(right.len()) > cap {
            return None;
        }
        let mut out = Vec::with_capacity(left.len().saturating_mul(right.len()));
        for a in &left {
            for b in &right {
                let mut joined = Vec::with_capacity(a.len() + b.len());
                joined.extend_from_slice(a);
                joined.extend_from_slice(b);
                out.push(joined);
            }
        }
        out.sort_unstable();
        out.dedup();
        (out.len() <= cap).then_some(out)
    }

    match expr {
        Expr::U8Seq(bytes) => Some(vec![bytes.clone()]),
        Expr::U8Class(class) => {
            let words = class.iter().map(|byte| vec![byte]).collect::<Vec<_>>();
            (words.len() <= cap).then_some(words)
        }
        Expr::Epsilon => Some(vec![Vec::new()]),
        Expr::Shared(inner) => finite_literal_words(inner, cap),
        Expr::Seq(parts) => {
            let mut words = vec![Vec::new()];
            for part in parts {
                words = combine(words, finite_literal_words(part, cap)?, cap)?;
            }
            Some(words)
        }
        Expr::Choice(options) => {
            let mut words = Vec::new();
            for option in options {
                words.extend(finite_literal_words(option, cap)?);
                if words.len() > cap {
                    return None;
                }
            }
            words.sort_unstable();
            words.dedup();
            (words.len() <= cap).then_some(words)
        }
        Expr::Repeat { expr, min, max: Some(max) }
            if *max <= 16 && max.saturating_sub(*min) <= 8 => {
            let unit = finite_literal_words(expr, cap)?;
            let mut all = Vec::new();
            for count in *min..=*max {
                let mut words = vec![Vec::new()];
                for _ in 0..count {
                    words = combine(words, unit.clone(), cap)?;
                }
                all.extend(words);
                if all.len() > cap {
                    return None;
                }
            }
            all.sort_unstable();
            all.dedup();
            (all.len() <= cap).then_some(all)
        }
        Expr::Dfa(_)
        | Expr::Intersect { .. }
        | Expr::Exclude { .. }
        | Expr::Repeat { .. } => None,
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ByteSet([u64; 4]);

impl ByteSet {
    fn full() -> Self {
        Self([u64::MAX; 4])
    }

    fn insert(&mut self, byte: u8) {
        self.0[byte as usize >> 6] |= 1u64 << (byte & 63);
    }

    fn contains(&self, byte: u8) -> bool {
        self.0[byte as usize >> 6] & (1u64 << (byte & 63)) != 0
    }

    fn union_with(&mut self, other: Self) {
        for (left, right) in self.0.iter_mut().zip(other.0) {
            *left |= right;
        }
    }

    fn count(&self) -> usize {
        self.0.iter().map(|word| word.count_ones() as usize).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ByteLanguage {
    epsilon: bool,
    bytes: ByteSet,
}

impl ByteLanguage {
    fn empty() -> Self {
        Self::default()
    }

    fn epsilon() -> Self {
        Self { epsilon: true, bytes: ByteSet::default() }
    }

    fn top() -> Self {
        Self { epsilon: true, bytes: ByteSet::full() }
    }

    fn union_with(&mut self, other: Self) {
        self.epsilon |= other.epsilon;
        self.bytes.union_with(other.bytes);
    }

    fn concat_tail(self, right: Self) -> Self {
        // r=1 suffix algebra. Any one-byte observation produced by the right
        // operand hides the left. Only an epsilon completion of the right can
        // expose the left operand's final byte.
        let mut bytes = right.bytes;
        if right.epsilon {
            bytes.union_with(self.bytes);
        }
        Self { epsilon: self.epsilon && right.epsilon, bytes }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BytePhaseSummary {
    historical_productive: bool,
    normal: ByteLanguage,
    event_from_entry: ByteLanguage,
    tail_to_return: ByteLanguage,
    tail_to_event: ByteLanguage,
}

impl BytePhaseSummary {
    fn bottom() -> Self {
        Self {
            historical_productive: false,
            normal: ByteLanguage::empty(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::empty(),
            tail_to_event: ByteLanguage::empty(),
        }
    }

    fn identity() -> Self {
        Self {
            historical_productive: true,
            normal: ByteLanguage::epsilon(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::empty(),
            tail_to_event: ByteLanguage::empty(),
        }
    }

    fn union_with(&mut self, other: Self) {
        self.historical_productive |= other.historical_productive;
        self.normal.union_with(other.normal);
        self.event_from_entry.union_with(other.event_from_entry);
        self.tail_to_return.union_with(other.tail_to_return);
        self.tail_to_event.union_with(other.tail_to_event);
    }

    fn concat(self, right: Self) -> Self {
        let normal = self.normal.concat_tail(right.normal);
        let mut event_from_entry = self.event_from_entry;
        event_from_entry.union_with(self.normal.concat_tail(right.event_from_entry));

        let mut tail_to_return = self.tail_to_return.concat_tail(right.normal);
        if self.historical_productive {
            tail_to_return.union_with(right.tail_to_return);
        }

        let mut tail_to_event = self.tail_to_event;
        tail_to_event.union_with(self.tail_to_return.concat_tail(right.event_from_entry));
        if self.historical_productive {
            tail_to_event.union_with(right.tail_to_event);
        }

        Self {
            historical_productive: self.historical_productive && right.historical_productive,
            normal,
            event_from_entry,
            tail_to_return,
            tail_to_event,
        }
    }
}


fn expr_last_byte_language(expr: &Expr) -> ByteLanguage {
    match expr {
        Expr::U8Seq(bytes) => {
            if let Some(&last) = bytes.last() {
                let mut set = ByteSet::default();
                set.insert(last);
                ByteLanguage { epsilon: false, bytes: set }
            } else {
                ByteLanguage::epsilon()
            }
        }
        Expr::U8Class(class) => {
            let mut set = ByteSet::default();
            for byte in class.iter() {
                set.insert(byte);
            }
            ByteLanguage { epsilon: false, bytes: set }
        }
        Expr::Dfa(dfa) => {
            // `Expr::Dfa` is normally epsilon-free. If a hand-built or legacy
            // expression carries epsilon edges, direct transition/finalizer
            // inspection is not enough to certify its final byte. Widen the
            // lexical atom rather than risk an under-approximation.
            if dfa.has_epsilon_transitions() {
                return ByteLanguage::top();
            }
            let mut set = ByteSet::default();
            for state in 0..dfa.num_states() as u32 {
                for (byte, target) in dfa.transitions(state) {
                    if !dfa.finalizers(target).is_empty() {
                        set.insert(byte);
                    }
                }
            }
            ByteLanguage {
                epsilon: !dfa.finalizers(0).is_empty(),
                bytes: set,
            }
        }
        Expr::Intersect { expr, intersect } => {
            let left = expr_last_byte_language(expr);
            let right = expr_last_byte_language(intersect);
            let mut bytes = ByteSet::default();
            for i in 0..4 {
                bytes.0[i] = left.bytes.0[i] & right.bytes.0[i];
            }
            ByteLanguage {
                epsilon: left.epsilon && right.epsilon,
                bytes,
            }
        }
        Expr::Exclude { expr, exclude } => {
            let base = expr_last_byte_language(expr);
            // Exclusion can only remove words, so retaining the base last-byte
            // set is a sound upper bound. Nullability can be decided exactly.
            ByteLanguage {
                epsilon: base.epsilon && !exclude.is_nullable(),
                bytes: base.bytes,
            }
        }
        Expr::Seq(parts) => {
            let mut out = ByteLanguage::epsilon();
            for part in parts {
                out = out.concat_tail(expr_last_byte_language(part));
            }
            out
        }
        Expr::Choice(options) => {
            let mut out = ByteLanguage::empty();
            for option in options {
                out.union_with(expr_last_byte_language(option));
            }
            out
        }
        Expr::Repeat { expr, min, max } => {
            if *max == Some(0) {
                return ByteLanguage::epsilon();
            }
            let inner = expr_last_byte_language(expr);
            ByteLanguage {
                epsilon: *min == 0 || inner.epsilon,
                bytes: inner.bytes,
            }
        }
        Expr::Shared(inner) => expr_last_byte_language(inner),
        Expr::Epsilon => ByteLanguage::epsilon(),
    }
}

fn byte_terminal_summary(
    constraint: &Constraint,
    terminal: TerminalID,
    outward: &BTreeSet<TerminalID>,
    child_overrides: &BTreeMap<TerminalID, BytePhaseSummary>,
) -> BytePhaseSummary {
    if let Some(child) = child_overrides.get(&terminal) {
        return *child;
    }
    if outward.contains(&terminal) {
        return BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::empty(),
            event_from_entry: ByteLanguage::epsilon(),
            tail_to_return: ByteLanguage::epsilon(),
            tail_to_event: ByteLanguage::epsilon(),
        };
    }
    if constraint.parser_is_control_terminal(terminal) {
        return BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::epsilon(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::epsilon(),
            tail_to_event: ByteLanguage::empty(),
        };
    }
    if constraint
        .special_token_terminals
        .iter()
        .any(|special| special.terminal_id == terminal)
    {
        return BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::top(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::top(),
            tail_to_event: ByteLanguage::empty(),
        };
    }

    let language = constraint
        .retained_terminal_expr(terminal)
        .map(expr_last_byte_language)
        .unwrap_or_else(ByteLanguage::top);
    BytePhaseSummary {
        historical_productive: language.epsilon || language.bytes.count() != 0,
        normal: language,
        event_from_entry: ByteLanguage::empty(),
        // A cut at the lexeme end contributes epsilon; any nonempty residual
        // completion has a final byte in the terminal's complete last-byte set.
        tail_to_return: ByteLanguage {
            epsilon: true,
            bytes: language.bytes,
        },
        tail_to_event: ByteLanguage::empty(),
    }
}

fn byte_rule_summary(
    rule: &Rule,
    nt: &BTreeMap<NonterminalID, BytePhaseSummary>,
    terminal: &[BytePhaseSummary],
) -> BytePhaseSummary {
    rule.rhs.iter().fold(BytePhaseSummary::identity(), |acc, symbol| {
        let right = match symbol {
            Symbol::Terminal(id) => terminal
                .get(*id as usize)
                .copied()
                .unwrap_or_else(BytePhaseSummary::bottom),
            Symbol::Nonterminal(id) => nt.get(id).copied().unwrap_or_else(BytePhaseSummary::bottom),
        };
        acc.concat(right)
    })
}

fn summarize_rules_module_r1(
    constraint: &Constraint,
    child_overrides: &BTreeMap<TerminalID, BytePhaseSummary>,
    bound_slots: &BTreeSet<TerminalID>,
) -> Result<(BytePhaseSummary, usize, bool), String> {
    let rules = constraint.retained_table_rules()?;
    if rules.is_empty() {
        return Err("r1 interface-tail summary requires retained grammar rules".to_owned());
    }
    let mut outward = outward_terminals(constraint);
    for terminal in bound_slots {
        outward.remove(terminal);
    }
    let terminals = (0..constraint.tokenizer.num_terminals())
        .map(|terminal| {
            byte_terminal_summary(
                constraint,
                terminal,
                &outward,
                child_overrides,
            )
        })
        .collect::<Vec<_>>();

    let mut nts = BTreeMap::<NonterminalID, BytePhaseSummary>::new();
    for rule in rules {
        nts.entry(rule.lhs).or_insert_with(BytePhaseSummary::bottom);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(id) = symbol {
                nts.entry(*id).or_insert_with(BytePhaseSummary::bottom);
            }
        }
    }

    let mut converged = false;
    let mut iterations = 0usize;
    for iteration in 0..DEFAULT_FIXED_POINT_LIMIT {
        iterations = iteration + 1;
        let snapshot = nts.clone();
        let mut next = snapshot.clone();
        for rule in rules {
            let summary = byte_rule_summary(rule, &snapshot, &terminals);
            next.entry(rule.lhs)
                .or_insert_with(BytePhaseSummary::bottom)
                .union_with(summary);
        }
        if next == nts {
            converged = true;
            break;
        }
        nts = next;
    }
    if !converged {
        let top = ByteLanguage::top();
        return Ok((
            BytePhaseSummary {
                historical_productive: true,
                normal: top,
                event_from_entry: top,
                tail_to_return: top,
                tail_to_event: top,
            },
            iterations,
            true,
        ));
    }

    let mut module = byte_rule_summary(&rules[0], &nts, &terminals);
    // Native compiler rules may have their empty productions removed while
    // the validated embedding retains exact root nullability. Child summaries
    // feed the r=2 concatenation algebra: omitting this empty ordinary word
    // loses a caller's adjacent byte pair across a nullable child (e.g. []).
    if constraint.composition_start_nullable()? {
        module.historical_productive = true;
        module.normal.epsilon = true;
        module.tail_to_return.epsilon = true;
    }

    // Local Skip/IGNORE is modeled conservatively at module exits. This keeps
    // real ignore-before-public-event/root-return witnesses while avoiding any
    // assumption about a future parent's reset domain.
    let mut ignore_ids = constraint.parser_skip_terminals().clone();
    if let Some(ignore) = constraint.ignore_terminal {
        ignore_ids.insert(ignore);
    }
    for ignore in ignore_ids {
        let bytes = if let Some(child) = child_overrides.get(&ignore) {
            child.tail_to_return.bytes
        } else if let Some(expr) = constraint.retained_terminal_expr(ignore) {
            expr_last_byte_language(expr).bytes
        } else {
            ByteSet::full()
        };
        module.tail_to_return.bytes.union_with(bytes);
        if !module.event_from_entry.bytes.0.iter().all(|word| *word == 0)
            || module.event_from_entry.epsilon
            || !module.tail_to_event.bytes.0.iter().all(|word| *word == 0)
            || module.tail_to_event.epsilon
        {
            module.tail_to_event.bytes.union_with(bytes);
        }
    }
    Ok((module, iterations, false))
}

fn has_recursive_public_exit(constraint: &Constraint) -> bool {
    if !outward_terminals(constraint).is_empty() {
        return true;
    }
    constraint
        .static_dynamic_overlay
        .as_ref()
        .is_some_and(|overlay| {
            overlay
                .segmented_parser_components
                .iter()
                .any(|component| has_recursive_public_exit(component.constraint.as_ref()))
        })
}

fn opaque_child_r1(constraint: &Constraint) -> BytePhaseSummary {
    let top = ByteLanguage::top();
    let public = has_recursive_public_exit(constraint);
    BytePhaseSummary {
        historical_productive: true,
        normal: top,
        event_from_entry: if public { top } else { ByteLanguage::empty() },
        tail_to_return: top,
        tail_to_event: if public { top } else { ByteLanguage::empty() },
    }
}

fn summarize_composition_envelope_r1(
    parent: &Constraint,
    bindings: &[(TerminalID, &Constraint)],
) -> Result<(BytePhaseSummary, usize, bool), String> {
    let mut overrides = BTreeMap::<TerminalID, BytePhaseSummary>::new();
    let mut bound_slots = BTreeSet::<TerminalID>::new();
    for &(slot, child) in bindings {
        bound_slots.insert(slot);
        overrides.insert(slot, opaque_child_r1(child));
    }
    summarize_rules_module_r1(parent, &overrides, &bound_slots)
}

fn summarize_constraint_module_r1(
    constraint: &Constraint,
) -> Result<(BytePhaseSummary, usize, bool), String> {
    let Some(overlay) = constraint.static_dynamic_overlay.as_ref() else {
        return summarize_rules_module_r1(constraint, &BTreeMap::new(), &BTreeSet::new());
    };
    if overlay.segmented_parser_components.is_empty() {
        return summarize_rules_module_r1(constraint, &BTreeMap::new(), &BTreeSet::new());
    }

    let parent_component = overlay
        .segmented_parser_components
        .first()
        .ok_or_else(|| "composed constraint is missing root component".to_owned())?;
    let parent = parent_component.constraint.as_ref();
    let mut child_by_component = BTreeMap::<u32, (BytePhaseSummary, usize, bool)>::new();
    let mut overrides = BTreeMap::<TerminalID, BytePhaseSummary>::new();
    let mut bound_slots = BTreeSet::<TerminalID>::new();
    let mut iterations = 0usize;
    let mut widened = false;

    for link in &overlay.segmented_parser_links {
        if link.parent_component != 0 {
            // Immediate-level overlays should only point from the local parent;
            // nested structure lives inside the child constraint itself. If a
            // future artifact violates that shape, do not publish a partially
            // interpreted summary: widen every observable field explicitly.
            let top = ByteLanguage::top();
            return Ok((
                BytePhaseSummary {
                    historical_productive: true,
                    normal: top,
                    event_from_entry: top,
                    tail_to_return: top,
                    tail_to_event: top,
                },
                iterations,
                true,
            ));
        }
        let child = child_by_component.entry(link.child_component).or_insert_with(|| {
            overlay
                .segmented_parser_components
                .get(link.child_component as usize)
                .and_then(|component| summarize_constraint_module_r1(component.constraint.as_ref()).ok())
                .unwrap_or((
                    BytePhaseSummary {
                        historical_productive: true,
                        normal: ByteLanguage::top(),
                        event_from_entry: ByteLanguage::top(),
                        tail_to_return: ByteLanguage::top(),
                        tail_to_event: ByteLanguage::top(),
                    },
                    0,
                    true,
                ))
        });
        iterations = iterations.saturating_add(child.1);
        widened |= child.2;
        bound_slots.insert(link.slot_terminal);
        overrides.insert(link.slot_terminal, child.0);
    }

    let (module, local_iterations, local_widened) =
        summarize_rules_module_r1(parent, &overrides, &bound_slots)?;
    iterations = iterations.saturating_add(local_iterations);
    widened |= local_widened;
    Ok((module, iterations, widened))
}

fn summarize_root_r1(constraint: &Constraint) -> Result<(ByteLanguage, usize, bool), String> {
    let (module, iterations, widened) = summarize_constraint_module_r1(constraint)?;
    let mut exits = module.tail_to_return;
    exits.union_with(module.tail_to_event);
    Ok((exits, iterations, widened))
}

fn candidate_ids_for_r1(vocab: &crate::Vocab, language: ByteLanguage) -> Vec<u32> {
    let mut ids = Vec::new();
    for (id, bytes) in vocab.iter() {
        if bytes.len() < 2 {
            continue;
        }
        if bytes[..bytes.len() - 1]
            .iter()
            .copied()
            .any(|byte| language.bytes.contains(byte))
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[cfg(test)]
#[test]
fn nullable_nested_return_keeps_the_empty_child_crossing_token() {
    use crate::{BuildOptions, Grammar, Optimization, ParserBackend, Vocab};
    let vocab = Vocab::new(vec![(0, b"X[a]!".to_vec()), (1, b"a]!".to_vec()),
        (2, b"[]!".to_vec()), (3, b"X[]!".to_vec())]);
    for mode in [Optimization::FastRuntime, Optimization::FastBuild] {
        let options = || BuildOptions::default().optimization(mode).parser_backend(ParserBackend::TemplateDfa);
        let leaf = Grammar::from_glrm(r#"glrm 1; start value; nt value = "a"?;"#)
            .compile_with(&vocab, options()).unwrap();
        let parent = Grammar::from_glrm(r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#)
            .compile_unlinked(&vocab).unwrap();
        let middle = parent.bind("leaf", &leaf).unwrap().link_with(options()).unwrap();
        let overlay = middle.static_dynamic_overlay.as_ref().unwrap();
        let local = &overlay.segmented_parser_components[0].constraint;
        let slot = overlay.segmented_parser_links[0].slot_terminal;
        let refined = build_composition_boundary_tail_r2(local, &[(slot, &leaf)], &vocab).unwrap();
        assert!(refined.candidate_ids.contains(&2),
            "two-byte tail lost nullable []! return: mode={mode:?} candidates={:?}", refined.candidate_ids);
        assert!(summarize_constraint_module_r1(&leaf).unwrap().0.normal.epsilon,
            "a validated nullable native root must include an empty ordinary word");
        let candidates = build_boundary_tail_r1(&middle, &vocab).unwrap().candidate_ids;
        assert!(candidates.contains(&2), "nullable child return excludes []!: mode={mode:?} candidates={candidates:?}");
    }
}

pub(crate) fn build_composition_boundary_tail_r1(
    parent: &Constraint,
    bindings: &[(TerminalID, &Constraint)],
    vocab: &crate::Vocab,
) -> Result<BoundaryTailR1Result, String> {
    let summary_started = Instant::now();
    let (module, fixed_point_iterations, fixed_point_widened) =
        summarize_composition_envelope_r1(parent, bindings)?;
    let mut language = module.tail_to_return;
    language.union_with(module.tail_to_event);
    let summary_ms = summary_started.elapsed().as_secs_f64() * 1000.0;
    let map_started = Instant::now();
    let candidate_ids = candidate_ids_for_r1(vocab, language);
    let map_ms = map_started.elapsed().as_secs_f64() * 1000.0;
    Ok(BoundaryTailR1Result {
        candidate_ids,
        exit_byte_count: language.bytes.count(),
        fixed_point_iterations,
        fixed_point_widened,
        summary_ms,
        map_ms,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct BoundaryTailR1Result {
    pub(crate) candidate_ids: Vec<u32>,
    pub(crate) exit_byte_count: usize,
    pub(crate) fixed_point_iterations: usize,
    pub(crate) fixed_point_widened: bool,
    pub(crate) summary_ms: f64,
    pub(crate) map_ms: f64,
}

pub(crate) fn build_boundary_tail_r1(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> Result<BoundaryTailR1Result, String> {
    let summary_started = Instant::now();
    let (language, fixed_point_iterations, fixed_point_widened) = summarize_root_r1(constraint)?;
    let summary_ms = summary_started.elapsed().as_secs_f64() * 1000.0;
    let map_started = Instant::now();
    let candidate_ids = candidate_ids_for_r1(vocab, language);
    let map_ms = map_started.elapsed().as_secs_f64() * 1000.0;
    Ok(BoundaryTailR1Result {
        candidate_ids,
        exit_byte_count: language.bytes.count(),
        fixed_point_iterations,
        fixed_point_widened,
        summary_ms,
        map_ms,
    })

}

/// A finite cover of nonempty prefixes of every word in an expression's
/// language. A cover word need not be a complete lexeme. `None` deliberately
/// means unrestricted, including any nullable or unsupported expression.
fn nonempty_entry_prefix_cover(expr: &Expr) -> Option<Vec<Vec<u8>>> {
    if let Some(words) = finite_literal_words(expr, FINITE_EXPR_WORD_CAP) {
        return (!words.is_empty() && words.iter().all(|word| !word.is_empty()))
            .then_some(words);
    }
    match expr {
        Expr::Shared(inner) => nonempty_entry_prefix_cover(inner),
        Expr::Repeat { expr, min, .. } if *min > 0 => nonempty_entry_prefix_cover(expr),
        Expr::Seq(parts) => {
            for part in parts {
                if let Some(words) = finite_literal_words(part, FINITE_EXPR_WORD_CAP) {
                    if !words.is_empty() && words.iter().all(Vec::is_empty) {
                        continue;
                    }
                }
                return nonempty_entry_prefix_cover(part);
            }
            None
        }
        Expr::Choice(options) => {
            let mut words = Vec::new();
            for option in options {
                words.extend(nonempty_entry_prefix_cover(option)?);
                if words.len() > FINITE_EXPR_WORD_CAP {
                    return None;
                }
            }
            words.sort_unstable();
            words.dedup();
            (!words.is_empty()).then_some(words)
        }
        _ => None,
    }
}

/// Upper-bound the first nonempty lexeme of an entered component. Syntactic
/// nullable nonterminals expose later FIRST terminals. A byte-empty terminal,
/// unresolved slot, or unsupported expression makes the entire entry unknown,
/// so it cannot hide a subsequent first byte. All descendant ignores are
/// included even when local policy would disallow them at this entry: extra
/// candidates are safe, whereas forgetting an inherited/global ignore is not.
fn component_entry_prefix_cover(constraint: &Constraint) -> Option<Vec<Vec<u8>>> {
    if constraint.composition_start_nullable().ok()? {
        return None;
    }
    let rules = constraint.retained_table_rules().ok()?;
    let first_rule = rules.first()?;
    let mut by_lhs = BTreeMap::<NonterminalID, Vec<&Rule>>::new();
    for rule in rules {
        by_lhs.entry(rule.lhs).or_default().push(rule);
    }
    let mut nullable = BTreeSet::new();
    loop {
        let previous = nullable.len();
        for rule in rules {
            if rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Nonterminal(id) => nullable.contains(id),
                Symbol::Terminal(_) => false,
            }) {
                nullable.insert(rule.lhs);
            }
        }
        if nullable.len() == previous {
            break;
        }
    }
    if nullable.contains(&first_rule.lhs) {
        return None;
    }
    let mut pending = vec![first_rule.lhs];
    let mut seen = BTreeSet::new();
    let mut first = BTreeSet::new();
    while let Some(nonterminal) = pending.pop() {
        if !seen.insert(nonterminal) {
            continue;
        }
        for rule in by_lhs.get(&nonterminal)? {
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(id) => {
                        first.insert(*id);
                        break;
                    }
                    Symbol::Nonterminal(id) => {
                        pending.push(*id);
                        if !nullable.contains(id) {
                            break;
                        }
                    }
                }
            }
        }
    }
    let outward = outward_terminals(constraint);
    let mut prefixes = Vec::new();
    for terminal in first {
        // A placeholder may retain a synthetic byte expression for compiler
        // coordinates. It does not constrain a future child's first bytes.
        // Special/control terminals likewise must not be treated as lexemes.
        if outward.contains(&terminal)
            || constraint.parser_is_control_terminal(terminal)
            || constraint.special_token_terminals.iter()
                .any(|special| special.terminal_id == terminal)
        {
            return None;
        }
        prefixes.extend(nonempty_entry_prefix_cover(
            constraint.retained_terminal_expr(terminal)?,
        )?);
        if prefixes.len() > FINITE_EXPR_WORD_CAP {
            return None;
        }
    }
    let mut pending = vec![constraint];
    let mut seen = BTreeSet::new();
    while let Some(component) = pending.pop() {
        // Local traversal dedup only; borrowed constraints remain alive for
        // this entire query. No address-keyed result is retained or published.
        if !seen.insert(component as *const Constraint as usize) {
            continue;
        }
        for terminal in component.parser_skip_terminals().iter().copied()
            .chain(component.ignore_terminal)
        {
            prefixes.extend(nonempty_entry_prefix_cover(
                component.retained_terminal_expr(terminal)?,
            )?);
            if prefixes.len() > FINITE_EXPR_WORD_CAP {
                return None;
            }
        }
        if let Some(overlay) = &component.static_dynamic_overlay {
            pending.extend(overlay.segmented_parser_components.iter()
                .map(|component| component.constraint.as_ref()));
        }
    }
    prefixes.sort_unstable();
    prefixes.dedup();
    (!prefixes.is_empty()).then_some(prefixes)
}

pub(crate) struct RootCallCandidateResult {
    pub(crate) candidate_ids: Vec<u32>,
    pub(crate) exit_byte_count: usize,
    pub(crate) entry_prefix_count: Option<usize>,
    pub(crate) summary_ms: f64,
    pub(crate) map_ms: f64,
}

/// The original exact root-CALL vocabulary predicate. Kept as an independent
/// validator and as the fallback for an unsorted prefix input.
fn root_call_candidate_ids_reference(
    vocab: &crate::Vocab,
    exits: ByteSet,
    entries: Option<&[Vec<u8>]>,
) -> Vec<u32> {
    let mut ids = Vec::new();
    for (id, bytes) in vocab.iter() {
        if (1..bytes.len()).any(|cut| {
            exits.contains(bytes[cut - 1]) && entries.is_none_or(|prefixes| {
                let suffix = &bytes[cut..];
                prefixes.iter().any(|prefix| {
                    suffix.starts_with(prefix) || prefix.starts_with(suffix)
                })
            })
        }) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Nonempty comparable byte strings must have the same first byte. Index the
/// already-sorted entry cover once, then compare only that byte's contiguous
/// prefix range at each positive, proper token cut. This changes no proof or
/// admission rule and makes no assumptions about text encoding or token IDs.
fn root_call_candidate_ids_indexed(
    vocab: &crate::Vocab,
    exits: ByteSet,
    entries: Option<&[Vec<u8>]>,
) -> Vec<u32> {
    let Some(prefixes) = entries else {
        return root_call_candidate_ids_reference(vocab, exits, None);
    };
    if prefixes.iter().any(Vec::is_empty)
        || prefixes.windows(2).any(|pair| pair[0] > pair[1])
    {
        return root_call_candidate_ids_reference(vocab, exits, entries);
    }
    if prefixes.is_empty() || exits.count() == 0 {
        return Vec::new();
    }
    let mut ranges = [(0usize, 0usize); 256];
    let mut cursor = 0;
    while cursor < prefixes.len() {
        let first = prefixes[cursor][0] as usize;
        let start = cursor;
        cursor += 1;
        while cursor < prefixes.len() && prefixes[cursor][0] as usize == first {
            cursor += 1;
        }
        ranges[first] = (start, cursor);
    }
    let mut ids = Vec::new();
    for (id, bytes) in vocab.iter() {
        if bytes.windows(2).enumerate().any(|(offset, pair)| {
            if !exits.contains(pair[0]) { return false; }
            let (lo, hi) = ranges[pair[1] as usize];
            if lo == hi { return false; }
            let suffix = &bytes[offset + 1..];
            prefixes[lo..hi].iter().any(|prefix| {
                suffix.starts_with(prefix) || prefix.starts_with(suffix)
            })
        }) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Link-local root refinement, never reusable component metadata. A flat
/// graph root has no caller: its first ownership crossing must be a CALL,
/// not root completion. At the SAME positive proper-prefix cut, the remaining
/// bytes must also be comparable with a prefix of an entered child's first
/// lexeme. We union child entries, deliberately dropping call-site correlation
/// in the widening direction. Full exact L2P equivalence still runs for every
/// retained vocabulary entry; no approximate equivalence is substituted.
pub(crate) fn build_root_call_candidates(
    parent: &Constraint,
    children: &[&Constraint],
    call_terminals: &[TerminalID],
    vocab: &crate::Vocab,
) -> Result<RootCallCandidateResult, String> {
    if parent.static_dynamic_overlay.as_ref().is_some_and(|overlay| {
        !overlay.segmented_parser_components.is_empty()
    }) {
        return Err("root CALL refinement requires a flat parent component".to_owned());
    }
    let started = Instant::now();
    // Use the actual link's terminal IDs, not only the parent's reusable
    // placeholder metadata. A byte-backed special token can become a CALL
    // when explicitly bound even if it was not an outward event beforehand.
    let calls = call_terminals.iter().map(|&terminal| (terminal, BytePhaseSummary {
        historical_productive: true,
        normal: ByteLanguage::empty(),
        event_from_entry: ByteLanguage::epsilon(),
        tail_to_return: ByteLanguage::epsilon(),
        tail_to_event: ByteLanguage::epsilon(),
    })).collect();
    let (module, _, _) = summarize_rules_module_r1(parent, &calls, &BTreeSet::new())?;
    let language = module.tail_to_event;
    let mut entries = Some(Vec::new());
    for child in children {
        let Some(prefixes) = component_entry_prefix_cover(child) else {
            entries = None;
            break;
        };
        let collected = entries.as_mut().expect("unknown entry stops iteration");
        collected.extend(prefixes);
        if collected.len() > FINITE_EXPR_WORD_CAP {
            entries = None;
            break;
        }
    }
    if let Some(prefixes) = &mut entries {
        prefixes.sort_unstable();
        prefixes.dedup();
    }
    let summary_ms = started.elapsed().as_secs_f64() * 1000.0;
    let map_started = Instant::now();
    let indexed = crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_ROOT_PREFIX_INDEX");
    let candidate_ids = if indexed {
        root_call_candidate_ids_indexed(vocab, language.bytes, entries.as_deref())
    } else {
        root_call_candidate_ids_reference(vocab, language.bytes, entries.as_deref())
    };
    if indexed && std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_ROOT_PREFIX_INDEX").is_some() {
        assert_eq!(candidate_ids,
            root_call_candidate_ids_reference(vocab, language.bytes, entries.as_deref()),
            "indexed root CALL candidates differ from the original full prefix scan");
        eprintln!("[glrmask/validate][boundary_root_prefix_index] exact=true candidates={}", candidate_ids.len());
    }
    Ok(RootCallCandidateResult {
        candidate_ids,
        exit_byte_count: language.bytes.count(),
        entry_prefix_count: entries.as_ref().map(Vec::len),
        summary_ms,
        map_ms: map_started.elapsed().as_secs_f64() * 1000.0,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PairSet {
    // Row indexed by first byte; each row contains possible second bytes.
    rows: Box<[ByteSet; 256]>,
}

impl Default for PairSet {
    fn default() -> Self {
        Self {
            rows: Box::new([ByteSet::default(); 256]),
        }
    }
}

impl PairSet {
    fn insert(&mut self, first: u8, second: u8) {
        self.rows[first as usize].insert(second);
    }

    fn contains(&self, first: u8, second: u8) -> bool {
        self.rows[first as usize].contains(second)
    }

    fn union_with(&mut self, other: &Self) {
        for (left, right) in self.rows.iter_mut().zip(other.rows.iter()) {
            left.union_with(*right);
        }
    }

    fn cartesian(left: ByteSet, right: ByteSet) -> Self {
        let mut out = Self::default();
        for first in 0u16..=255 {
            let first = first as u8;
            if left.contains(first) {
                out.rows[first as usize] = right;
            }
        }
        out
    }

    fn any_first_ending_in(last: ByteSet) -> Self {
        Self {
            rows: Box::new([last; 256]),
        }
    }

    fn count(&self) -> usize {
        self.rows.iter().map(ByteSet::count).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Tail2Language {
    epsilon: bool,
    /// Last byte of every non-empty word.
    last1: ByteSet,
    /// Exact one-byte words. Needed for proper-prefix cuts shorter than r.
    len1: ByteSet,
    /// Last two bytes of every word of length >= 2.
    last2: PairSet,
}

impl Tail2Language {
    fn empty() -> Self {
        Self::default()
    }

    fn epsilon() -> Self {
        Self {
            epsilon: true,
            ..Self::default()
        }
    }

    fn from_words(words: &[Vec<u8>]) -> Self {
        let mut out = Self::empty();
        for word in words {
            match word.as_slice() {
                [] => out.epsilon = true,
                [byte] => {
                    out.last1.insert(*byte);
                    out.len1.insert(*byte);
                }
                _ => {
                    let n = word.len();
                    out.last1.insert(word[n - 1]);
                    out.last2.insert(word[n - 2], word[n - 1]);
                }
            }
        }
        out
    }

    /// Maximally precise r=2 upper envelope available from an r=1 language
    /// alone: preserve epsilon and final-byte support, but allow every possible
    /// preceding byte for words of length >=2 and conservatively allow a
    /// one-byte word for every supported final byte.
    fn from_r1_upper(r1: ByteLanguage) -> Self {
        Self {
            epsilon: r1.epsilon,
            last1: r1.bytes,
            len1: r1.bytes,
            last2: PairSet::any_first_ending_in(r1.bytes),
        }
    }

    fn union_with(&mut self, other: &Self) {
        self.epsilon |= other.epsilon;
        self.last1.union_with(other.last1);
        self.len1.union_with(other.len1);
        self.last2.union_with(&other.last2);
    }

    fn concat_tail(&self, right: &Self) -> Self {
        let epsilon = self.epsilon && right.epsilon;

        let mut last1 = right.last1;
        if right.epsilon {
            last1.union_with(self.last1);
        }

        let mut len1 = ByteSet::default();
        if self.epsilon {
            len1.union_with(right.len1);
        }
        if right.epsilon {
            len1.union_with(self.len1);
        }

        let mut last2 = right.last2.clone();
        // If the right word has exactly one byte, the final pair is the last
        // byte of a non-empty left word followed by that one right byte.
        last2.union_with(&PairSet::cartesian(self.last1, right.len1));
        if right.epsilon {
            last2.union_with(&self.last2);
        }

        Self {
            epsilon,
            last1,
            len1,
            last2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PhaseSummary2 {
    historical_productive: bool,
    normal: Tail2Language,
    event_from_entry: Tail2Language,
    tail_to_return: Tail2Language,
    tail_to_event: Tail2Language,
}

impl PhaseSummary2 {
    fn bottom() -> Self {
        Self {
            historical_productive: false,
            normal: Tail2Language::empty(),
            event_from_entry: Tail2Language::empty(),
            tail_to_return: Tail2Language::empty(),
            tail_to_event: Tail2Language::empty(),
        }
    }

    fn identity() -> Self {
        Self {
            historical_productive: true,
            normal: Tail2Language::epsilon(),
            event_from_entry: Tail2Language::empty(),
            tail_to_return: Tail2Language::empty(),
            tail_to_event: Tail2Language::empty(),
        }
    }

    fn from_r1_upper(summary: BytePhaseSummary) -> Self {
        Self {
            historical_productive: summary.historical_productive,
            normal: Tail2Language::from_r1_upper(summary.normal),
            event_from_entry: Tail2Language::from_r1_upper(summary.event_from_entry),
            tail_to_return: Tail2Language::from_r1_upper(summary.tail_to_return),
            tail_to_event: Tail2Language::from_r1_upper(summary.tail_to_event),
        }
    }

    fn union_with(&mut self, other: &Self) {
        self.historical_productive |= other.historical_productive;
        self.normal.union_with(&other.normal);
        self.event_from_entry.union_with(&other.event_from_entry);
        self.tail_to_return.union_with(&other.tail_to_return);
        self.tail_to_event.union_with(&other.tail_to_event);
    }

    fn concat(&self, right: &Self) -> Self {
        let normal = self.normal.concat_tail(&right.normal);

        let mut event_from_entry = self.event_from_entry.clone();
        event_from_entry.union_with(&self.normal.concat_tail(&right.event_from_entry));

        let mut tail_to_return = self.tail_to_return.concat_tail(&right.normal);
        if self.historical_productive {
            tail_to_return.union_with(&right.tail_to_return);
        }

        let mut tail_to_event = self.tail_to_event.clone();
        tail_to_event.union_with(&self.tail_to_return.concat_tail(&right.event_from_entry));
        if self.historical_productive {
            tail_to_event.union_with(&right.tail_to_event);
        }

        Self {
            historical_productive: self.historical_productive && right.historical_productive,
            normal,
            event_from_entry,
            tail_to_return,
            tail_to_event,
        }
    }
}

fn terminal_summary2(
    constraint: &Constraint,
    terminal: TerminalID,
    outward: &BTreeSet<TerminalID>,
    child_overrides: &BTreeMap<TerminalID, PhaseSummary2>,
) -> PhaseSummary2 {
    if let Some(child) = child_overrides.get(&terminal) {
        return child.clone();
    }
    if outward.contains(&terminal) {
        let eps = Tail2Language::epsilon();
        return PhaseSummary2 {
            historical_productive: true,
            normal: Tail2Language::empty(),
            event_from_entry: eps.clone(),
            tail_to_return: eps.clone(),
            tail_to_event: eps,
        };
    }
    if constraint.parser_is_control_terminal(terminal) {
        let eps = Tail2Language::epsilon();
        return PhaseSummary2 {
            historical_productive: true,
            normal: eps.clone(),
            event_from_entry: Tail2Language::empty(),
            tail_to_return: eps,
            tail_to_event: Tail2Language::empty(),
        };
    }
    if let Some(expr) = constraint.retained_terminal_expr(terminal) {
        if let Some(words) = finite_literal_words(expr, FINITE_EXPR_WORD_CAP) {
            let normal = Tail2Language::from_words(&words);
            let mut tails = Tail2Language::epsilon();
            for word in &words {
                for cut in 0..word.len() {
                    let suffix = word[cut..].to_vec();
                    tails.union_with(&Tail2Language::from_words(&[suffix]));
                }
            }
            return PhaseSummary2 {
                historical_productive: !words.is_empty(),
                normal,
                event_from_entry: Tail2Language::empty(),
                tail_to_return: tails,
                tail_to_event: Tail2Language::empty(),
            };
        }
    }
    // Unsupported lexical atom: retain the r=1 facts and widen only the
    // unknown byte to the left.
    PhaseSummary2::from_r1_upper(byte_terminal_summary(
        constraint,
        terminal,
        outward,
        &BTreeMap::new(),
    ))
}

fn rule_summary2(
    rule: &Rule,
    nt: &BTreeMap<NonterminalID, PhaseSummary2>,
    terminals: &[PhaseSummary2],
) -> PhaseSummary2 {
    rule.rhs.iter().fold(PhaseSummary2::identity(), |acc, symbol| {
        let right = match symbol {
            Symbol::Terminal(id) => terminals
                .get(*id as usize)
                .cloned()
                .unwrap_or_else(PhaseSummary2::bottom),
            Symbol::Nonterminal(id) => nt.get(id).cloned().unwrap_or_else(PhaseSummary2::bottom),
        };
        acc.concat(&right)
    })
}

fn summarize_rules_module_r2(
    constraint: &Constraint,
    child_overrides: &BTreeMap<TerminalID, PhaseSummary2>,
    bound_slots: &BTreeSet<TerminalID>,
) -> Result<(PhaseSummary2, usize, bool), String> {
    let rules = constraint.retained_table_rules()?;
    if rules.is_empty() {
        return Err("r2 interface-tail summary requires retained grammar rules".to_owned());
    }
    let mut outward = outward_terminals(constraint);
    for terminal in bound_slots {
        outward.remove(terminal);
    }
    let terminals = (0..constraint.tokenizer.num_terminals())
        .map(|terminal| terminal_summary2(constraint, terminal, &outward, child_overrides))
        .collect::<Vec<_>>();
    let mut nts = BTreeMap::<NonterminalID, PhaseSummary2>::new();
    for rule in rules {
        nts.entry(rule.lhs).or_insert_with(PhaseSummary2::bottom);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(id) = symbol {
                nts.entry(*id).or_insert_with(PhaseSummary2::bottom);
            }
        }
    }
    let mut iterations = 0usize;
    for iteration in 0..DEFAULT_FIXED_POINT_LIMIT {
        iterations = iteration + 1;
        let snapshot = nts.clone();
        let mut next = snapshot.clone();
        for rule in rules {
            let summary = rule_summary2(rule, &snapshot, &terminals);
            next.entry(rule.lhs)
                .or_insert_with(PhaseSummary2::bottom)
                .union_with(&summary);
        }
        if next == nts {
            let module = rule_summary2(&rules[0], &next, &terminals);
            return Ok((module, iterations, false));
        }
        nts = next;
    }
    Ok((
        PhaseSummary2::from_r1_upper(BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::top(),
            event_from_entry: ByteLanguage::top(),
            tail_to_return: ByteLanguage::top(),
            tail_to_event: ByteLanguage::top(),
        }),
        iterations,
        true,
    ))
}

fn candidate_ids_for_r2(vocab: &crate::Vocab, language: &Tail2Language) -> Vec<u32> {
    let mut ids = Vec::new();
    'tokens: for (id, bytes) in vocab.iter() {
        if bytes.len() < 2 {
            continue;
        }
        // k=1 (<r): the entire one-byte prefix must be an exact exit word.
        if language.len1.contains(bytes[0]) {
            ids.push(id);
            continue;
        }
        // k>=2: the final two bytes immediately before the proper cut must be
        // a supported exit tail.
        for cut in 2..bytes.len() {
            if language.last2.contains(bytes[cut - 2], bytes[cut - 1]) {
                ids.push(id);
                continue 'tokens;
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub(crate) fn build_composition_boundary_tail_r2(
    parent: &Constraint,
    bindings: &[(TerminalID, &Constraint)],
    vocab: &crate::Vocab,
) -> Result<BoundaryTailR2Result, String> {
    // The r=1 layer explicitly widens for local Skip/IGNORE consumption before
    // an outward event. The r=2 pair algebra below intentionally models only
    // grammar-symbol concatenation, so decline rather than risk forgetting an
    // ignore byte between the historical cut and the interface. Composition
    // automatically falls back to the sound and faster r=1 summary.
    if parent.ignore_terminal.is_some() || !parent.parser_skip_terminals().is_empty() {
        return Err("r2 interface-tail summary declines parent skip/ignore; use r1".to_owned());
    }
    let started = Instant::now();
    let child_started = Instant::now();
    // One child may be called from several parent terminals.  Its module
    // summary is independent of the call site, so compute it once and fan the
    // result back out to every bound slot.
    let mut unique_children = BTreeMap::<usize, (&Constraint, Vec<TerminalID>)>::new();
    for &(slot, child) in bindings {
        unique_children
            .entry(child as *const Constraint as usize)
            .or_insert_with(|| (child, Vec::new()))
            .1
            .push(slot);
    }
    let child_rows = unique_children
        .into_values()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(child, slots)| {
            let (r1, _, widened) = summarize_constraint_module_r1(child)?;
            let mut r2 = PhaseSummary2::from_r1_upper(r1);
            if widened {
                r2.historical_productive = true;
            }
            Ok::<_, String>((slots, r2))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let child_summary_ms = child_started.elapsed().as_secs_f64() * 1000.0;
    let mut overrides = BTreeMap::<TerminalID, PhaseSummary2>::new();
    let mut bound_slots = BTreeSet::<TerminalID>::new();
    for (slots, r2) in child_rows {
        for slot in slots {
            bound_slots.insert(slot);
            overrides.insert(slot, r2.clone());
        }
    }
    let (module, fixed_point_iterations, fixed_point_widened) =
        summarize_rules_module_r2(parent, &overrides, &bound_slots)?;
    let mut exits = module.tail_to_return;
    exits.union_with(&module.tail_to_event);
    let summary_ms = started.elapsed().as_secs_f64() * 1000.0;
    let map_started = Instant::now();
    let candidate_ids = candidate_ids_for_r2(vocab, &exits);
    let map_ms = map_started.elapsed().as_secs_f64() * 1000.0;
    Ok(BoundaryTailR2Result {
        candidate_ids,
        exit_last1_count: exits.last1.count(),
        exit_pair_count: exits.last2.count(),
        fixed_point_iterations,
        fixed_point_widened,
        child_summary_ms,
        summary_ms,
        map_ms,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct BoundaryTailR2Result {
    pub(crate) candidate_ids: Vec<u32>,
    pub(crate) exit_last1_count: usize,
    pub(crate) exit_pair_count: usize,
    pub(crate) fixed_point_iterations: usize,
    pub(crate) fixed_point_widened: bool,
    pub(crate) child_summary_ms: f64,
    pub(crate) summary_ms: f64,
    pub(crate) map_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(entries: &[(u32, &[u8])]) -> crate::Vocab {
        crate::Vocab::new(
            entries
                .iter()
                .map(|(id, bytes)| (*id, bytes.to_vec()))
                .collect(),
        )
    }

    fn probe_root_call_candidates(
        parent: &Constraint,
        children: &[&Constraint],
        vocab: &crate::Vocab,
    ) -> RootCallCandidateResult {
        let call = parent.terminal_display_names.iter()
            .position(|name| name == "SUB").expect("fixture CALL terminal") as u32;
        build_root_call_candidates(parent, children, &[call], vocab).unwrap()
    }

    #[test]
    fn r1_root_completion_is_strict_proper_prefix_and_preserves_duplicate_ids() {
        let vocab = vocab(&[(0, b"a"), (1, b"ab"), (2, b"ab"), (3, b"b")]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let probe = build_boundary_tail_r1(&constraint, &vocab).unwrap();
        assert!(
            !probe.candidate_ids.contains(&0),
            "token-end completion is not a proper-prefix event"
        );
        assert!(probe.candidate_ids.contains(&1));
        assert!(probe.candidate_ids.contains(&2));
    }

    #[test]
    fn root_call_index_preserves_generated_binary_vocabulary_and_fallbacks() {
        let mut seed = 0x8a6137c5u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize
        };
        let alphabet = [0u8, 9, 10, 32, 34, 97, 98, 127, 128, 255];
        for case in 0..256 {
            let mut words = vec![(1u32, Vec::new()), (8, vec![0]),
                (22, vec![255, 0, 128]), (31, vec![255, 0, 128])];
            for n in 0..96 {
                let len = next() % 33;
                words.push((100 + n * 7, (0..len).map(|_| alphabet[next() % alphabet.len()]).collect()));
            }
            let vocab = crate::Vocab::new(words);
            let mut exits = ByteSet::default();
            for &byte in &alphabet { if next() % 2 == 0 { exits.insert(byte); } }
            let count = next() % 16;
            let mut prefixes: Vec<Vec<u8>> = (0..count).map(|_| {
                let len = 1 + next() % 8;
                (0..len).map(|_| alphabet[next() % alphabet.len()]).collect()
            }).collect();
            if case % 11 == 0 { prefixes.push(Vec::new()); }
            prefixes.sort();
            if case % 7 == 0 { prefixes.reverse(); }
            for cover in [None, Some(prefixes.as_slice()), Some(&[][..])] {
                assert_eq!(root_call_candidate_ids_indexed(&vocab, exits, cover),
                    root_call_candidate_ids_reference(&vocab, exits, cover), "case {case}");
            }
        }
    }

    #[test]
    fn root_call_index_checks_both_prefix_directions_at_the_same_proper_cut() {
        let vocab = vocab(&[
            (0, b"a"), (1, b"ac"), (2, b"ac"), (3, b"acat"),
            (4, b"acaterpillar"), (5, b"xa"), (6, b"axcat"),
            (7, b"xac"), (8, b"adog"), (9, b"acow"),
            (10, b"cat"), (11, b""),
        ]);
        let mut exits = ByteSet::default(); exits.insert(b'a');
        let prefixes = vec![b"cat".to_vec(), b"cot".to_vec(), b"dog".to_vec()];
        assert_eq!(root_call_candidate_ids_indexed(&vocab, exits, Some(&prefixes)),
            vec![1, 2, 3, 4, 7, 8]);
        assert_eq!(root_call_candidate_ids_indexed(&vocab, exits, Some(&[Vec::new()])),
            root_call_candidate_ids_reference(&vocab, exits, None));
        assert!(root_call_candidate_ids_indexed(&vocab, exits, Some(&[])).is_empty());
        assert!(root_call_candidate_ids_indexed(&vocab, ByteSet::default(), Some(&prefixes)).is_empty());
    }

    #[test]
    fn root_call_candidates_match_exit_and_entry_at_the_same_cut() {
        let vocab = vocab(&[
            (0, b"a"), (1, b"ac"), (2, b"ac"), (3, b"ax"),
            (4, b"bc"), (5, b"acatb"), (6, b"aac"), (7, b"xc"),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"start document; t SUB ::= @token(999);
               nt document ::= "a" SUB "b";"#, &vocab,
        ).unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"start document; nt document ::= "cat";"#, &vocab,
        ).unwrap();
        let reusable = build_boundary_tail_r1(&parent, &vocab).unwrap();
        let refined = probe_root_call_candidates(&parent, &[&child], &vocab);
        assert_eq!(refined.entry_prefix_count, Some(1));
        assert_eq!(refined.candidate_ids, vec![1, 2, 5, 6]);
        assert!(reusable.candidate_ids.contains(&4),
            "the reusable component must retain its possible outward RETURN");
        assert_eq!(build_boundary_tail_r1(&parent, &vocab).unwrap().candidate_ids,
            reusable.candidate_ids, "the root-local proof must not mutate reusable metadata");
    }

    #[test]
    fn root_call_candidates_preserve_parent_and_entered_child_ignores() {
        let vocab = vocab(&[
            (0, b" cat"), (1, b" dog"), (2, b" \tcat"),
            (3, b" \tdog"), (4, b" "), (5, b"c cat"),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"start document; ignore WS; t WS ::= " "+;
               t SUB ::= @token(999); nt document ::= SUB;"#, &vocab,
        ).unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"start document; ignore TAB; t TAB ::= "\t"+;
               nt document ::= "cat";"#, &vocab,
        ).unwrap();
        let result = probe_root_call_candidates(&parent, &[&child], &vocab);
        assert!(result.candidate_ids.contains(&0));
        assert!(result.candidate_ids.contains(&2));
        assert!(result.candidate_ids.contains(&3),
            "a conservative first-ignore prefix must not infer the later parser continuation");
        assert!(!result.candidate_ids.contains(&1));
        assert!(!result.candidate_ids.contains(&4));
    }

    #[test]
    fn root_call_candidates_widen_unknown_and_nullable_entries() {
        let vocab = vocab(&[(0, b"ac"), (1, b"ax"), (2, b"a"), (3, b"bc")]);
        let parent = Constraint::from_glrm_grammar(
            r#"start document; t SUB ::= @token(999);
               nt document ::= "a" SUB;"#, &vocab,
        ).unwrap();
        let nullable = Constraint::from_glrm_grammar(
            r#"start document; nt document ::= "cat"?;"#, &vocab,
        ).unwrap();
        let unknown = Constraint::from_glrm_grammar(
            r#"start document; t LATER ::= @token(998);
               nt document ::= LATER;"#, &vocab,
        ).unwrap();
        for child in [&nullable, &unknown] {
            let result = probe_root_call_candidates(&parent, &[child], &vocab);
            assert_eq!(result.entry_prefix_count, None);
            assert_eq!(result.candidate_ids, vec![0, 1]);
        }
    }

    #[test]
    fn root_call_candidates_include_both_sides_of_nullable_first_nonterminal() {
        let vocab = vocab(&[(0, b"ac"), (1, b"ax"), (2, b"ad"), (3, b"axy")]);
        let parent = Constraint::from_glrm_grammar(
            r#"start document; t SUB ::= @token(999);
               nt document ::= "a" SUB;"#, &vocab,
        ).unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"start document; nt document ::= "x"? "cat";"#, &vocab,
        ).unwrap();
        let result = probe_root_call_candidates(&parent, &[&child], &vocab);
        assert!(result.candidate_ids.contains(&0));
        assert!(result.candidate_ids.contains(&1));
        assert!(result.candidate_ids.contains(&3));
        if result.entry_prefix_count.is_some() {
            assert!(!result.candidate_ids.contains(&2));
        }
    }

    #[test]
    fn root_call_candidates_use_actual_byte_backed_bound_slot() {
        let vocab = vocab(&[(0, b"ac"), (1, b"ax"), (2, b"a"), (999, b"?")]);
        let parent = Constraint::from_glrm_grammar(
            r#"start document; t SUB ::= @token(999);
               nt document ::= "a" SUB "b";"#, &vocab,
        ).unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"start document; nt document ::= "cat";"#, &vocab,
        ).unwrap();
        let result = probe_root_call_candidates(&parent, &[&child], &vocab);
        assert_eq!(result.candidate_ids, vec![0]);
    }

    #[test]
    fn r1_ignore_before_public_slot_keeps_positive_ignore_prefix() {
        let vocab = vocab(&[(0, b" x"), (1, b" xq"), (2, b"xq"), (3, b" ")]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let probe = build_boundary_tail_r1(&constraint, &vocab).unwrap();
        assert!(probe.candidate_ids.contains(&0));
        assert!(probe.candidate_ids.contains(&1));
        assert!(
            !probe.candidate_ids.contains(&3),
            "one-byte token has no proper internal cut"
        );
    }

    #[test]
    fn r1_opaque_bound_child_preserves_mandatory_postamble() {
        let vocab = vocab(&[
            (0, b")x"),
            (1, b"abc)x"),
            (2, b"abcx"),
            (3, b")"),
            (4, b"x)"),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "prefix(" SUB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "anything";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "SUB")
            .unwrap() as u32;
        let probe =
            build_composition_boundary_tail_r1(&parent, &[(slot, &child)], &vocab).unwrap();
        assert_eq!(probe.exit_byte_count, 1);
        assert!(probe.candidate_ids.contains(&0));
        assert!(probe.candidate_ids.contains(&1));
        assert!(!probe.candidate_ids.contains(&2));
        assert!(
            !probe.candidate_ids.contains(&3),
            "token-end ')' is not a proper-prefix cut"
        );
        assert!(
            !probe.candidate_ids.contains(&4),
            "final ')' is not an internal cut"
        );
    }

    #[test]
    fn r2_child_final_byte_plus_postamble_prunes_unrelated_pairs() {
        let vocab = vocab(&[
            (0, b")x"),
            (1, b"})x"),
            (2, b"a)x"),
            (3, b"}x"),
            (4, b")"),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= SUB ")";
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
            .unwrap() as u32;
        let probe =
            build_composition_boundary_tail_r2(&parent, &[(slot, &child)], &vocab).unwrap();
        assert!(
            probe.candidate_ids.contains(&0),
            "token may start in the postamble itself"
        );
        assert!(
            probe.candidate_ids.contains(&1),
            "child-final '}}' followed by ')' is a real two-byte tail"
        );
        assert!(
            !probe.candidate_ids.contains(&2),
            "unrelated byte before ')' must be pruned"
        );
        assert!(
            !probe.candidate_ids.contains(&3),
            "child return alone has not yet reached parent root"
        );
        assert!(
            !probe.candidate_ids.contains(&4),
            "token-end ')' is not a proper-prefix cut"
        );
    }
    #[test]
    fn r2_parent_ignore_declines_to_r1_layer() {
        let vocab = vocab(&[(0, b" x"), (1, b" )x"), (2, b")x")]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= SUB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "anything";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "SUB")
            .unwrap() as u32;
        assert!(build_composition_boundary_tail_r2(&parent, &[(slot, &child)], &vocab).is_err());
    }

}
