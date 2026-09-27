use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::analysis::{AnalyzedGrammar, EOF};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

pub(crate) mod action;
mod build;
mod compose;
mod embedding;

pub use embedding::EmbeddedStart;
mod optimize;
pub(crate) mod row;

pub use action::{Action, GuardedStackShift, StackShift, StackShiftGuard};
#[allow(unused_imports)]
pub use compose::{
    ComposedTable, SubgrammarTableInput, compose_subgrammar_tables,
    compose_subgrammar_tables_explicit, compose_subgrammar_tables_explicit_with_rules,
    compose_subgrammar_tables_with_rules, empty_terminals_in_composed_table,
    subgrammar_child_return_pop,
};

use build::{build_table, build_table_with_default_construction, Item, PendingAction};
#[allow(unused_imports)]
pub use optimize::ControlEliminationReport;
#[allow(unused_imports)]
pub use optimize::control_elimination_budget_exhausted;
#[allow(unused_imports)]
pub use optimize::control_elimination_run_count;
use optimize::merge_same_core_lr1_states;

use row::{ActionRow, GotoRow};

const DISABLE_DEFAULT_ACTION_ROWS_ENV: &str = "GLRMASK_DISABLE_DEFAULT_ACTION_ROWS";

fn default_action_rows_enabled() -> bool {
    !std::env::var(DISABLE_DEFAULT_ACTION_ROWS_ENV)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardedShiftCellIndex {
    pub guard_pops: Box<[u32]>,
    pub by_guard_key: FxHashMap<(u32, u32), Box<[u32]>>,
    pub guard_counts: Box<[u16]>,
    pub unguarded_indices: Box<[u32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdmissionPolicy {
    /// Cheap row-presence admission is exact for this table.
    RowPresenceExact,
    /// Dynamically simulate reductions/gotos before admitting a terminal.
    ExactSimulation,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self::RowPresenceExact
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GlrTableConstruction {
    LegacyRowBisim,
    Lalr,
    ExperimentalCoreMerged,
}

impl Default for GlrTableConstruction {
    fn default() -> Self {
        Self::LegacyRowBisim
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<ActionRow>,
    pub goto: Vec<GotoRow>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
    #[serde(default)]
    pub nonterminal_display_names: Vec<String>,
    pub embedded_start: EmbeddedStart,
    #[serde(default)]
    pub construction: GlrTableConstruction,
    #[serde(default)]
    pub admission_policy: AdmissionPolicy,
    /// Terminal support used by cheap admission/mask queries.
    ///
    /// `action` is the optimized execution table. Some execution actions are
    /// guarded stack effects, whose guards must be evaluated when executing the
    /// action. This side table is captured before guard-producing stack-effect
    /// lowering and is kept in sync across state remapping/merging. A bit set in
    /// this vector answers only the admission question: "can a reachable parser
    /// path with this top state advance on this terminal?"  That lets
    /// `stack_may_advance_on*` be pure row-presence checks without inspecting an
    /// optimized action body.
    #[serde(default)]
    pub advance: Vec<BitSet>,
    /// Runtime-derived subset of `advance` whose current-row action is
    /// unconditionally admissible without inspecting deeper parser stack.
    /// Rebuilt after load/final table mutation; never serialized.
    #[serde(skip, default)]
    pub unconditional_advance: Vec<BitSet>,
    /// Set of (state, terminal) pairs where the shift was created by the
    /// transfer mechanism. The characterization should treat these as
    /// non-replace to avoid creating pop-0 reduces in the template NFA.
    pub forwarded_shifts: FxHashSet<(u32, TerminalID)>,
    /// Linker-internal zero-width terminal labels. These are never emitted by
    /// the lexer; parser execution computes their closure before and after a
    /// real terminal.
    #[serde(default)]
    pub control_terminals: BTreeSet<TerminalID>,
    /// Terminals whose table action is parser identity in a subset of states.
    /// This metadata preserves scoped ignore ownership across nested compiled
    /// composition; the actions themselves remain the execution authority.
    #[serde(default)]
    pub skip_terminals: BTreeSet<TerminalID>,
    #[serde(skip)]
    pub guarded_shift_index: Vec<FxHashMap<TerminalID, GuardedShiftCellIndex>>,
    /// Maximum-width direct-regular frontier descriptors captured while rows
    /// are constructed, before the expanded table would need to be rescanned.
    #[serde(default)]
    pub direct_regular_wide_frontiers: Vec<DirectRegularWideFrontierDescriptor>,
}

/// Version-scoped artifact serialization for GLR tables.
///
/// Sectioned artifacts replace the in-core table with a one-byte placeholder
/// and serialize its rows and semantic metadata independently. The codec owns
/// compact admission rows, chunked actions and deferred source-rule backing.
pub mod artifact_serde;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DirectRegularWideFrontierDescriptor {
    pub source_state: u32,
    pub terminal: TerminalID,
    pub target_states: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TableAmbiguityKind {
    Split,
    StackShifts,
    GuardedStackShifts,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableAmbiguity {
    pub state: u32,
    pub terminal: TerminalID,
    pub kind: TableAmbiguityKind,
    pub alternatives: usize,
}

fn guarded_stack_shift_constraints(
    guards: &[StackShiftGuard],
) -> Option<BTreeMap<u32, BTreeSet<u32>>> {
    let mut constraints: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for guard in guards {
        let states: BTreeSet<u32> = guard.states.iter().copied().collect();
        if states.is_empty() {
            return None;
        }
        if let Some(existing) = constraints.get_mut(&guard.pop) {
            existing.retain(|state| states.contains(state));
            if existing.is_empty() {
                return None;
            }
        } else {
            constraints.insert(guard.pop, states);
        }
    }
    Some(constraints)
}

fn guarded_stack_shifts_overlap(left: &GuardedStackShift, right: &GuardedStackShift) -> bool {
    let Some(left_constraints) = guarded_stack_shift_constraints(&left.guards) else {
        return false;
    };
    let Some(right_constraints) = guarded_stack_shift_constraints(&right.guards) else {
        return false;
    };

    for (pop, left_states) in &left_constraints {
        if let Some(right_states) = right_constraints.get(pop)
            && left_states.is_disjoint(right_states)
        {
            return false;
        }
    }

    true
}

fn guarded_stack_shifts_are_ambiguous(shifts: &[GuardedStackShift]) -> bool {
    for first in 0..shifts.len() {
        for second in (first + 1)..shifts.len() {
            if guarded_stack_shifts_overlap(&shifts[first], &shifts[second]) {
                return true;
            }
        }
    }
    false
}

fn action_ambiguity(action: &Action) -> Option<(TableAmbiguityKind, usize)> {
    match action {
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            let alternatives = usize::from(shift.is_some()) + reduces.len() + usize::from(*accept);
            (alternatives > 1).then_some((TableAmbiguityKind::Split, alternatives))
        }
        Action::StackShifts(shifts) => {
            (shifts.len() > 1).then_some((TableAmbiguityKind::StackShifts, shifts.len()))
        }
        Action::GuardedStackShifts(shifts) => {
            (guarded_stack_shifts_are_ambiguous(shifts))
                .then_some((TableAmbiguityKind::GuardedStackShifts, shifts.len()))
        }
        _ => None,
    }
}

impl GLRTable {
    /// Minimal table metadata for a runtime that executes a retained direct-regular
    /// automaton instead of materialized LR action rows.
    pub fn direct_regular_runtime_stub(num_states: u32, num_terminals: u32) -> Self {
        Self {
            action: Vec::new(),
            goto: Vec::new(),
            num_states,
            num_terminals,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            embedded_start: Default::default(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        }
    }



    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        build_table(grammar)
    }

    pub fn build_with_default_construction(
        grammar: &AnalyzedGrammar,
        default_construction: GlrTableConstruction,
    ) -> Self {
        build_table_with_default_construction(grammar, default_construction)
    }

    #[inline]
    fn terminal_bit(&self, terminal: TerminalID) -> Option<usize> {
        if terminal == EOF {
            Some(self.num_terminals as usize)
        } else if terminal < self.num_terminals {
            Some(terminal as usize)
        } else {
            None
        }
    }

    #[inline]
    fn has_advance_rows(&self) -> bool {
        self.advance.len() == self.num_states as usize
    }

    pub fn rebuild_advance_rows_from_actions(&mut self) {
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            self.advance = action_presence_rows(&self.action, self.num_terminals);
        } else {
            use rayon::prelude::*;
            let num_terminals = self.num_terminals;
            self.advance = self
                .action
                .par_iter()
                .map(|row| action_presence_row(row, num_terminals))
                .collect();
        }
    }

    pub fn rebuild_unconditional_advance_rows(&mut self) {
        use rayon::prelude::*;

        let terminal_count = self.num_terminals as usize;
        let build_row = |row: &ActionRow| {
            let unconditional = |terminal: TerminalID, action: &Action| match action {
                Action::Shift(..) | Action::ReplaceShifts(_) | Action::Skip => true,
                Action::Split { shift, accept, .. } => {
                    shift.is_some() || (*accept && terminal == crate::glr::analysis::EOF)
                }
                Action::StackShifts(_)
                | Action::GuardedStackShifts(_)
                | Action::Reduce(..)
                | Action::Accept => false,
            };
            match row {
                ActionRow::Sparse(row) => {
                    let mut admitted = BitSet::new(terminal_count);
                    for (terminal, action) in row.iter() {
                        if (*terminal as usize) < terminal_count
                            && unconditional(*terminal, action)
                        {
                            admitted.set(*terminal as usize);
                        }
                    }
                    admitted
                }
                ActionRow::Indexed { .. } => {
                    let mut admitted = BitSet::new(terminal_count);
                    for (terminal, action) in row.iter() {
                        if (terminal as usize) < terminal_count && unconditional(terminal, action) {
                            admitted.set(terminal as usize);
                        }
                    }
                    admitted
                }
                ActionRow::Default {
                    default,
                    exceptions,
                    num_terminals,
                } if *num_terminals as usize == terminal_count => {
                    let default_unconditional = match default {
                        Action::Shift(..) | Action::ReplaceShifts(_) | Action::Skip => true,
                        Action::Split { shift, .. } => shift.is_some(),
                        Action::StackShifts(_)
                        | Action::GuardedStackShifts(_)
                        | Action::Reduce(..)
                        | Action::Accept => false,
                    };
                    let mut admitted = if default_unconditional {
                        BitSet::all(terminal_count)
                    } else {
                        BitSet::new(terminal_count)
                    };
                    for (terminal, action) in exceptions.iter() {
                        let is_unconditional = action
                            .as_ref()
                            .is_some_and(|action| unconditional(*terminal, action));
                        if is_unconditional {
                            admitted.set(*terminal as usize);
                        } else {
                            admitted.clear(*terminal as usize);
                        }
                    }
                    admitted
                }
                ActionRow::Default { .. } => {
                    // Composition can retain a default row over a component's
                    // original terminal domain inside a wider composed table.
                    // In that case the default applies only to the row's own
                    // domain; preserve the generic iterator semantics rather
                    // than extending the default to newly added terminals.
                    let mut admitted = BitSet::new(terminal_count);
                    for (terminal, action) in row.iter() {
                        if (terminal as usize) < terminal_count && unconditional(terminal, action) {
                            admitted.set(terminal as usize);
                        }
                    }
                    admitted
                }
            }
        };
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            self.unconditional_advance = self.action.iter().map(build_row).collect();
        } else {
            self.unconditional_advance = self.action.par_iter().map(build_row).collect();
        }
    }

    #[inline]
    pub fn unconditional_advance_row(&self, state: u32) -> Option<&BitSet> {
        (self.unconditional_advance.len() == self.num_states as usize)
            .then(|| self.unconditional_advance.get(state as usize))
            .flatten()
    }

    pub fn rebuild_guarded_shift_index(&mut self) {
        let build_row = |row: &ActionRow| {
            let mut index_row = FxHashMap::default();

            for (terminal, action) in row {
                let Action::GuardedStackShifts(shifts) = action else {
                    continue;
                };

                let mut guard_pops = BTreeSet::new();
                let mut by_guard_key: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
                let mut guard_counts = vec![0u16; shifts.len()];
                let mut unguarded_indices = Vec::new();

                for (shift_index, shift) in shifts.iter().enumerate() {
                    if shift.guards.is_empty() {
                        unguarded_indices.push(shift_index as u32);
                        continue;
                    }

                    for guard in &shift.guards {
                        guard_pops.insert(guard.pop);
                        guard_counts[shift_index] += 1;
                        for &guard_state in &guard.states {
                            by_guard_key
                                .entry((guard.pop, guard_state))
                                .or_default()
                                .push(shift_index as u32);
                        }
                    }
                }

                index_row.insert(
                    terminal,
                    GuardedShiftCellIndex {
                        guard_pops: guard_pops.into_iter().collect::<Vec<_>>().into_boxed_slice(),
                        by_guard_key: by_guard_key
                            .into_iter()
                            .map(|(key, shift_indices)| (key, shift_indices.into_boxed_slice()))
                            .collect(),
                        guard_counts: guard_counts.into_boxed_slice(),
                        unguarded_indices: unguarded_indices.into_boxed_slice(),
                    },
                );
            }

            index_row
        };
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            self.guarded_shift_index = self.action.iter().map(build_row).collect();
        } else {
            use rayon::prelude::*;
            self.guarded_shift_index = self.action.par_iter().map(build_row).collect();
        }
    }

    /// Whether any parser cell actually contains a guarded stack shift.
    ///
    /// Dynamic artifacts deliberately omit `guarded_shift_index` because it is
    /// derived. Most ordinary schema tables contain no guarded shifts at all;
    /// in that common case rebuilding an all-empty Vec<FxHashMap<..>> after
    /// every load is pure allocation work. A linear action scan is both much
    /// cheaper and sufficient to decide whether the index is needed.
    pub fn has_guarded_stack_shifts(&self) -> bool {
        self.action.iter().any(|row| {
            row.iter()
                .any(|(_, action)| matches!(action, Action::GuardedStackShifts(_)))
        })
    }

    #[inline]
    pub fn guarded_shift_index(
        &self,
        state: u32,
        terminal: TerminalID,
    ) -> Option<&GuardedShiftCellIndex> {
        self.guarded_shift_index
            .get(state as usize)
            .and_then(|row| row.get(&terminal))
    }

    #[inline]
    pub fn advance_row_allows(&self, state: u32, terminal: TerminalID) -> bool {
        if self.has_advance_rows() {
            let Some(bit) = self.terminal_bit(terminal) else {
                return false;
            };
            return self
                .advance
                .get(state as usize)
                .is_some_and(|row| row.contains(bit));
        }

        // Compatibility fallback for hand-built test tables and older serialized
        // artifacts that do not carry the side table. Newly compiled tables build
        // `advance` before guard-producing optimizations run.
        self.action(state, terminal).is_some()
    }

    #[inline]
    pub fn advance_row(&self, state: u32) -> Option<&BitSet> {
        self.has_advance_rows()
            .then(|| self.advance.get(state as usize))
            .flatten()
    }

    #[inline]
    pub fn advance_row_intersects(&self, state: u32, terminals: &BitSet) -> bool {
        if self.has_advance_rows()
            && let Some(row) = self.advance.get(state as usize)
        {
            return row
                .words()
                .iter()
                .zip(terminals.words())
                .any(|(left, right)| (*left & *right) != 0);
        }

        self.action.get(state as usize).is_some_and(|actions| {
            actions.keys().any(|terminal| {
                self.terminal_bit(terminal)
                    .is_some_and(|bit| terminals.contains(bit))
            })
        })
    }

    pub fn compress_default_action_rows(&mut self) {
        use rayon::prelude::*;
        let num_terminals = self.num_terminals;
        if rayon::current_num_threads() == 1 || self.action.len() < 128 {
            for row in &mut self.action {
                row.compress_default(num_terminals);
            }
        } else {
            self.action
                .par_iter_mut()
                .for_each(|row| row.compress_default(num_terminals));
        }
    }

    #[inline]
    pub fn action(&self, state: u32, terminal: TerminalID) -> Option<&Action> {
        self.action
            .get(state as usize)
            .and_then(|by_terminal| by_terminal.get(&terminal))
    }

    pub fn ambiguous_actions(&self) -> Vec<TableAmbiguity> {
        let mut ambiguities = Vec::new();
        for (state, row) in self.action.iter().enumerate() {
            for (terminal, action) in row {
                if let Some((kind, alternatives)) = action_ambiguity(action) {
                    ambiguities.push(TableAmbiguity {
                        state: state as u32,
                        terminal,
                        kind,
                        alternatives,
                    });
                }
            }
        }
        ambiguities
    }

    pub fn has_ambiguity(&self) -> bool {
        self.action.iter().enumerate().any(|(_, row)| {
            row.into_iter()
                .any(|(_, action)| action_ambiguity(action).is_some())
        })
    }

    #[inline]
    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<(u32, bool)> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
    }

    pub(super) fn validate_structure(&self, context: &str) {
        let expected_len = self.num_states as usize;
        if self.action.len() != expected_len {
            panic!(
                "{context}: action row count {} does not match num_states {}",
                self.action.len(),
                self.num_states,
            );
        }
        if self.goto.len() != expected_len {
            panic!(
                "{context}: goto row count {} does not match num_states {}",
                self.goto.len(),
                self.num_states,
            );
        }
        if !self.advance.is_empty() && self.advance.len() != expected_len {
            panic!(
                "{context}: advance row count {} does not match num_states {}",
                self.advance.len(),
                self.num_states,
            );
        }

        let validate_target = |source_state: u32,
                               label_kind: &str,
                               label_value: u32,
                               path: &str,
                               target: u32| {
            if target >= self.num_states {
                panic!(
                    "{context}: state {} {} {} has invalid {} target {} >= num_states {}",
                    source_state,
                    label_kind,
                    label_value,
                    path,
                    target,
                    self.num_states,
                );
            }
        };

        for (source_state, row) in self.action.iter().enumerate() {
            let source_state = source_state as u32;
            for (terminal, action) in row.iter() {
                match action {
                    Action::Shift(target, _) => {
                        validate_target(source_state, "terminal", terminal, "Action::Shift", *target);
                    }
                    Action::ReplaceShifts(targets) => {
                        for (target_idx, &target) in targets.iter().enumerate() {
                            validate_target(
                                source_state,
                                "terminal",
                                terminal,
                                &format!("Action::ReplaceShifts[{target_idx}]"),
                                target,
                            );
                        }
                    }
                    Action::StackShifts(shifts) => {
                        for (shift_idx, shift) in shifts.iter().enumerate() {
                            for (push_idx, &target) in shift.pushes.iter().enumerate() {
                                validate_target(
                                    source_state,
                                    "terminal",
                                    terminal,
                                    &format!("Action::StackShifts[{shift_idx}].pushes[{push_idx}]"),
                                    target,
                                );
                            }
                        }
                    }
                    Action::GuardedStackShifts(shifts) => {
                        for (shift_idx, shift) in shifts.iter().enumerate() {
                            for (guard_idx, guard) in shift.guards.iter().enumerate() {
                                for (state_idx, &target) in guard.states.iter().enumerate() {
                                    validate_target(
                                        source_state,
                                        "terminal",
                                        terminal,
                                        &format!("Action::GuardedStackShifts[{shift_idx}].guards[{guard_idx}].states[{state_idx}]"),
                                        target,
                                    );
                                }
                            }
                            for (push_idx, &target) in shift.pushes.iter().enumerate() {
                                validate_target(
                                    source_state,
                                    "terminal",
                                    terminal,
                                    &format!("Action::GuardedStackShifts[{shift_idx}].pushes[{push_idx}]"),
                                    target,
                                );
                            }
                        }
                    }
                    Action::Split { shift: Some((target, _)), .. } => {
                        validate_target(
                            source_state,
                            "terminal",
                            terminal,
                            "Action::Split.shift",
                            *target,
                        );
                    }
                    Action::Split { shift: None, .. }
                    | Action::Reduce(..)
                    | Action::Accept
                    | Action::Skip => {}
                }
            }
        }

        for (source_state, row) in self.goto.iter().enumerate() {
            let source_state = source_state as u32;
            for (&nonterminal, &(target, _)) in row.iter() {
                validate_target(source_state, "nonterminal", nonterminal, "goto", target);
            }
        }

        for &(state, terminal) in &self.forwarded_shifts {
            if state >= self.num_states {
                panic!(
                    "{context}: forwarded_shifts has invalid state {} for terminal {} >= num_states {}",
                    state,
                    terminal,
                    self.num_states,
                );
            }
        }
    }

    #[inline]
    pub fn nonterminal_display_name(&self, nt: NonterminalID) -> Option<&str> {
        self.nonterminal_display_names
            .get(nt as usize)
            .map(String::as_str)
    }
}

fn action_presence_rows(action: &[ActionRow], num_terminals: u32) -> Vec<BitSet> {
    let mut rows = Vec::with_capacity(action.len());
    for action_row in action {
        rows.push(action_presence_row(action_row, num_terminals));
    }
    rows
}

fn action_presence_row(action_row: &ActionRow, num_terminals: u32) -> BitSet {
    let mut row = BitSet::new(num_terminals as usize + 1);
    for terminal in action_row.keys() {
        let bit = if terminal == EOF {
            num_terminals as usize
        } else if terminal < num_terminals {
            terminal as usize
        } else {
            continue;
        };
        row.set(bit);
    }
    row
}

impl GLRTable {
    pub fn extend_advance_rows_from_actions(&mut self) {
        if self.advance.is_empty() {
            return;
        }

        for action_row in self.action.iter().skip(self.advance.len()) {
            self.advance
                .push(action_presence_row(action_row, self.num_terminals));
        }
    }
}

pub mod testing {
    use super::row::{ActionRow, GotoRow};
    use super::{Action, AdmissionPolicy, GlrTableConstruction, GLRTable};
    use crate::grammar::flat::{NonterminalID, TerminalID};

    pub fn build_test_table(
        num_states: u32,
        num_terminals: u32,
        action_rows: &[&[(TerminalID, Action)]],
        goto_rows: &[&[(NonterminalID, (u32, bool))]],
    ) -> GLRTable {
        let action: Vec<_> = action_rows
            .iter()
            .map(|row| ActionRow::from_iter(row.iter().cloned()))
            .collect();
        let advance = super::action_presence_rows(&action, num_terminals);
        GLRTable {
            action,
            goto: goto_rows
                .iter()
                .map(|row| GotoRow::from_iter(row.iter().cloned()))
                .collect(),
            num_states,
            num_terminals,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            embedded_start: Default::default(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance,
            unconditional_advance: Vec::new(),
            forwarded_shifts: Default::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        }
    }
}

#[cfg(test)]
mod ambiguity_tests {
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::grammar::ast::{lower, GrammarExpr, Quantifier, NamedGrammar, NamedRule};
    use crate::grammar::expr_nfa::ExprNfaBuilder;
    use crate::grammar::glrm::from_glrm;

    use super::testing::build_test_table;
    use super::{Action, GLRTable, GuardedStackShift, StackShift, TableAmbiguity, TableAmbiguityKind};

    fn build_table_from_glrm(glrm: &str) -> GLRTable {
        let named = from_glrm(glrm).unwrap();
        build_table_from_named_grammar(&named)
    }

    fn build_table_from_named_grammar(named: &NamedGrammar) -> GLRTable {
        let grammar = lower(&named).unwrap();
        let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
        GLRTable::build(&analyzed)
    }

    #[test]
    #[ignore]
    fn dump_tiny_subgrammar_splice_oracle() {
        fn dump(label: &str, source: &str) {
            let named = from_glrm(source).unwrap();
            let grammar = lower(&named).unwrap();
            let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
            let table = GLRTable::build(&analyzed);
            eprintln!("\n=== {label} ===");
            eprintln!(
                "terminals={:?}",
                analyzed.terminal_display_names,
            );
            eprintln!(
                "nonterminals={:?}",
                analyzed.nonterminal_display_names,
            );
            eprintln!("rules={:?}", table.rules);
            for state in 0..table.num_states as usize {
                let actions = table.action[state]
                    .iter()
                    .map(|(terminal, action)| (terminal, action.clone()))
                    .collect::<Vec<_>>();
                let gotos = table.goto[state]
                    .iter()
                    .map(|(nonterminal, target)| (*nonterminal, *target))
                    .collect::<Vec<_>>();
                eprintln!("state {state}: action={actions:?} goto={gotos:?}");
            }
        }

        dump(
            "child",
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
        );
        dump(
            "parent-pseudo",
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">";
            "#,
        );
        dump(
            "flattened",
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">";
            "#,
        );
    }

    fn build_expr_nfa_optional_pair_suffix_grammar_with_value(
        value_symbol: GrammarExpr,
        extra_rules: Vec<NamedRule>,
    ) -> NamedGrammar {
        let mut builder = ExprNfaBuilder::new();

        let start = builder.start_state();
        let after_a = builder.add_state();
        let after_av = builder.add_state();
        let before_b_or_c = builder.add_state();
        let after_b = builder.add_state();
        let after_bv = builder.add_state();
        let before_c = builder.add_state();
        let after_c = builder.add_state();
        let after_cv = builder.add_state();
        let accept = builder.add_state();

        builder.add_epsilon(start, before_b_or_c);
        builder.add_transition(start, GrammarExpr::Literal(b"a".to_vec()), after_a);
        builder.add_transition(after_a, value_symbol.clone(), after_av);
        builder.add_transition(after_av, GrammarExpr::Literal(b",".to_vec()), before_b_or_c);

        builder.add_epsilon(before_b_or_c, before_c);
        builder.add_transition(before_b_or_c, GrammarExpr::Literal(b"b".to_vec()), after_b);
        builder.add_transition(after_b, value_symbol.clone(), after_bv);
        builder.add_transition(after_bv, GrammarExpr::Literal(b",".to_vec()), before_c);

        builder.add_transition(before_c, GrammarExpr::Literal(b"c".to_vec()), after_c);
        builder.add_transition(after_c, value_symbol, after_cv);
        builder.add_transition(after_cv, GrammarExpr::Literal(b"$".to_vec()), accept);
        builder.set_accepting(accept);

        let expr_nfa = builder.build().into_determinized_and_minimized();
        let mut rules = vec![NamedRule {
            name: "start".into(),
            expr: GrammarExpr::ExprNFA(Box::new(expr_nfa)),
            is_terminal: false,
            is_internal: false,
        }];
        rules.extend(extra_rules);
        NamedGrammar {
            rules,
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        }
    }

    fn glrm_recursive_array_colorpalette_minimal_stackshift_mre() -> &'static str {
        r#"
            start start;
            t JSON_ITEM_SEPARATOR ::= /(?:, )/;
            nt json_array ::= "[" (json_value (JSON_ITEM_SEPARATOR json_value)*)? "]";
            nt json_value ::= json_array;
            nt start ::= "{" "\"icons\": " "{" "\"ColorPalette\": " "{" ("\"k\": " json_value)* "}" "}" "}";
        "#
    }

    fn build_direct_recursive_array_colorpalette_minimal_grammar() -> NamedGrammar {
        let start_expr = GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"{".to_vec()),
            GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"\"icons\": ".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"{".to_vec()),
                    GrammarExpr::SeparatedSequence {
                        items: vec![(
                            GrammarExpr::Sequence(vec![
                                GrammarExpr::Literal(b"\"ColorPalette\": ".to_vec()),
                                GrammarExpr::Sequence(vec![
                                    GrammarExpr::Literal(b"{".to_vec()),
                                    GrammarExpr::SeparatedSequence {
                                        items: vec![(
                                            GrammarExpr::Sequence(vec![
                                                GrammarExpr::Literal(b"\"k\": ".to_vec()),
                                                GrammarExpr::Ref("json_value".into()),
                                            ]),
                                            Some(Quantifier::ZeroPlus),
                                        )],
                                        separator: Box::new(GrammarExpr::Ref("JSON_ITEM_SEPARATOR".into())),
                                        allow_empty: true,
                                    },
                                    GrammarExpr::Literal(b"}".to_vec()),
                                ]),
                            ]),
                            Some(Quantifier::Optional),
                        )],
                        separator: Box::new(GrammarExpr::Ref("JSON_ITEM_SEPARATOR".into())),
                        allow_empty: true,
                    },
                    GrammarExpr::Literal(b"}".to_vec()),
                ]),
            ]),
            GrammarExpr::Literal(b"}".to_vec()),
        ]);

        NamedGrammar {
            rules: vec![
                NamedRule {
                    name: "JSON_ITEM_SEPARATOR".into(),
                    expr: GrammarExpr::RawRegex("(?:, )".into()),
                    is_terminal: true,
                    is_internal: false,
                },
                NamedRule {
                    name: "json_array".into(),
                    expr: GrammarExpr::Sequence(vec![
                        GrammarExpr::Literal(b"[".to_vec()),
                        GrammarExpr::Quantified(Box::new(GrammarExpr::Sequence(vec![
                            GrammarExpr::Ref("json_value".into()),
                            GrammarExpr::Quantified(Box::new(GrammarExpr::Sequence(vec![
                                GrammarExpr::Ref("JSON_ITEM_SEPARATOR".into()),
                                GrammarExpr::Ref("json_value".into()),
                            ])), Quantifier::ZeroPlus),
                        ])), Quantifier::Optional),
                        GrammarExpr::Literal(b"]".to_vec()),
                    ]),
                    is_terminal: false,
                    is_internal: false,
                },
                NamedRule {
                    name: "json_value".into(),
                    expr: GrammarExpr::Ref("json_array".into()),
                    is_terminal: false,
                    is_internal: false,
                },
                NamedRule {
                    name: "start".into(),
                    expr: start_expr,
                    is_terminal: false,
                    is_internal: false,
                },
            ],
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        }
    }

    fn assert_table_has_all_pop1_stack_shift_ambiguity(table: &GLRTable) {
        let ambiguities = table.ambiguous_actions();
        let Some(ambiguity) = ambiguities.iter().find(|ambiguity| {
            ambiguity.kind == TableAmbiguityKind::StackShifts
                && matches!(
                    table.action(ambiguity.state, ambiguity.terminal),
                    Some(Action::StackShifts(shifts))
                        if shifts.len() > 1 && shifts.iter().all(|shift| shift.pop == 1)
                )
        }) else {
            panic!("expected all-pop1 StackShifts ambiguity, but table had none: {:?}", table.ambiguous_actions());
        };

        let action = table
            .action(ambiguity.state, ambiguity.terminal)
            .expect("ambiguous action should still be present in table");
        match action {
            Action::StackShifts(shifts) => {
                assert!(shifts.iter().all(|shift| shift.pop == 1));
            }
            other => panic!("expected StackShifts action, found {:?}", other),
        }
    }

    fn assert_table_has_no_ambiguity(table: &GLRTable) {
        let ambiguities = table.ambiguous_actions();
        if let Some(TableAmbiguity {
            state,
            terminal,
            kind,
            alternatives,
        }) = ambiguities.first()
        {
            let action = table
                .action(*state, *terminal)
                .expect("ambiguous action should still be present in table");
            panic!(
                "expected unambiguous table, found ambiguity at state={} terminal={} kind={:?} alternatives={} action={:?}",
                state,
                terminal,
                kind,
                alternatives,
                action
            );
        }
    }

    fn assert_table_has_no_all_pop1_stack_shift_ambiguity(table: &GLRTable) {
        let ambiguities = table.ambiguous_actions();
        if let Some(ambiguity) = ambiguities.iter().find(|ambiguity| {
            ambiguity.kind == TableAmbiguityKind::StackShifts
                && matches!(
                    table.action(ambiguity.state, ambiguity.terminal),
                    Some(Action::StackShifts(shifts))
                        if shifts.len() > 1 && shifts.iter().all(|shift| shift.pop == 1)
                )
        }) {
            let action = table
                .action(ambiguity.state, ambiguity.terminal)
                .expect("ambiguous action should still be present in table");
            panic!(
                "expected unambiguous table, found all-pop1 StackShifts ambiguity at state={} terminal={} alternatives={} action={:?}",
                ambiguity.state,
                ambiguity.terminal,
                ambiguity.alternatives,
                action,
            );
        }
    }

    #[test]
    fn ambiguous_actions_reports_split_and_stack_shift_fanout() {
        let token = 0;
        let table = build_test_table(
            4,
            1,
            &[
                &[(token, Action::Shift(1, false))],
                &[(
                    token,
                    Action::Split {
                        shift: Some((2, false)),
                        reduces: vec![(0, 1)],
                        accept: false,
                    },
                )],
                &[(
                    token,
                    Action::StackShifts(vec![
                        StackShift {
                            pop: 1,
                            pushes: vec![2],
                        },
                        StackShift {
                            pop: 2,
                            pushes: vec![3],
                        },
                    ]),
                )],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![
                        GuardedStackShift {
                            guards: Vec::new(),
                            pop: 1,
                            pushes: vec![2],
                        },
                        GuardedStackShift {
                            guards: Vec::new(),
                            pop: 1,
                            pushes: vec![3],
                        },
                    ]),
                )],
            ],
            &[&[], &[], &[], &[]],
        );

        let ambiguities = table.ambiguous_actions();
        assert!(table.has_ambiguity());
        assert_eq!(ambiguities.len(), 3);
        assert_eq!(ambiguities[0].kind, TableAmbiguityKind::Split);
        assert_eq!(ambiguities[0].alternatives, 2);
        assert_eq!(ambiguities[1].kind, TableAmbiguityKind::StackShifts);
        assert_eq!(ambiguities[1].alternatives, 2);
        assert_eq!(ambiguities[2].kind, TableAmbiguityKind::GuardedStackShifts);
        assert_eq!(ambiguities[2].alternatives, 2);
    }

    #[test]
    fn guarded_stack_shifts_with_disjoint_guards_are_not_ambiguous() {
        let token = 0;
        let table = build_test_table(
            1,
            1,
            &[&[(
                token,
                Action::GuardedStackShifts(vec![
                    GuardedStackShift {
                        guards: vec![super::StackShiftGuard {
                            pop: 1,
                            states: vec![7],
                        }],
                        pop: 2,
                        pushes: vec![5],
                    },
                    GuardedStackShift {
                        guards: vec![super::StackShiftGuard {
                            pop: 1,
                            states: vec![10],
                        }],
                        pop: 2,
                        pushes: vec![8],
                    },
                ]),
            )]],
            &[&[]],
        );

        assert!(!table.has_ambiguity());
        assert!(table.ambiguous_actions().is_empty());
    }

    #[test]
    #[should_panic(expected = "validator test action target")]
    fn validate_structure_panics_on_invalid_action_target() {
        let token = 0;
        let table = build_test_table(
            2,
            1,
            &[&[(token, Action::Shift(2, false))], &[]],
            &[&[], &[]],
        );

        table.validate_structure("validator test action target");
    }

    #[test]
    fn expr_nfa_optional_pair_suffixes_with_value_ref_have_no_table_ambiguity() {
        let grammar = build_expr_nfa_optional_pair_suffix_grammar_with_value(
            GrammarExpr::Ref("value".into()),
            vec![NamedRule {
                name: "value".into(),
                expr: GrammarExpr::Literal(b"v".to_vec()),
                is_terminal: false,
                is_internal: false,
            }],
        );

        let table = build_table_from_named_grammar(&grammar);

        assert_table_has_no_ambiguity(&table);
    }

    #[test]
    fn glrm_recursive_array_colorpalette_minimal_does_not_reproduce_all_pop1_stack_shifts() {
        let table = build_table_from_glrm(glrm_recursive_array_colorpalette_minimal_stackshift_mre());

        assert_table_has_no_all_pop1_stack_shift_ambiguity(&table);
    }

    #[test]
    fn direct_recursive_array_colorpalette_minimal_avoids_all_pop1_stack_shifts() {
        let grammar = build_direct_recursive_array_colorpalette_minimal_grammar();
        let table = build_table_from_named_grammar(&grammar);

        assert_table_has_no_all_pop1_stack_shift_ambiguity(&table);
    }

}
