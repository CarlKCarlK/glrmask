//! Construction-only native parser pipeline.
//!
//! The table is borrowed, never retained by a compiled program. Action identity
//! interning avoids repeatedly hashing large reduction vectors while forming
//! terminal signatures. One GOTO predecessor index serves ordinary terminals,
//! standalone completion, and finite embedding return construction.
//!
//! Exact characterization interning is local to one synchronous build. A task
//! that loses an interning race records an alias without waiting. Every owner
//! finishes before the parallel collection joins, so reading result cells after
//! that join cannot block a Rayon worker.
//!
//! Fingerprints select equality buckets only. Complete characterization equality
//! is the sole authority for sharing.

mod program;
pub use program::ProgramCompiler;

use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::table::{Action, GLRTable, StackShiftGuard};
use crate::grammar::flat::{NonterminalID, Symbol, TerminalID};

use super::characterize::{
    FinishEndpointPolicy, FinishTransfer, InitialEscape, InitialReduce,
    NtEscape, NtRereduce, StackMatcher, TerminalCharacterization,
};

fn flag(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        )
    })
}

fn parallel() -> bool {
    !super::macro_parallelism_disabled() && rayon::current_num_threads() > 1
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCompileError {
    pub terminal: TerminalID,
    pub message: String,
}

impl std::fmt::Display for NativeCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "terminal {} template compilation failed: {}; refusing table fallback",
            self.terminal, self.message
        )
    }
}

impl std::error::Error for NativeCompileError {}

pub struct CompiledGroup<T> {
    pub terminals: Vec<TerminalID>,
    pub program: Arc<T>,
}

pub struct CompiledSelection<T> {
    pub groups: Vec<CompiledGroup<T>>,
    pub action_signature_groups: usize,
    pub exact_characterization_groups: usize,
}

/// Immutable, construction-only index tied to one borrowed table.
pub struct NativeTableIndex<'a> {
    table: &'a GLRTable,
    selected: Vec<bool>,
    action_states: Vec<Vec<u32>>,
    signatures: Vec<Vec<(u32, u32, bool)>>,
    predecessors: Vec<Vec<(u32, NonterminalID, bool)>>,
    eof_states: Vec<u32>,
    accepting_eof_states: Vec<u32>,
    valid_entries: BTreeSet<TerminalID>,
    unique_actions: usize,
    logical_actions: usize,
}

impl<'a> NativeTableIndex<'a> {
    pub fn new(table: &'a GLRTable, selected: &[bool]) -> Result<Self, String> {
        if selected.len() != table.num_terminals as usize {
            return Err("selected template mask must cover the terminal domain".into());
        }
        if table.action.len() > table.num_states as usize
            || table.goto.len() > table.num_states as usize
        {
            return Err("compiler table contains rows outside its state coordinate".into());
        }

        let mut action_states = vec![Vec::new(); selected.len()];
        let mut signatures = vec![Vec::new(); selected.len()];
        let mut valid_entries = vec![true; selected.len()];
        let mut eof_states = Vec::new();
        let mut accepting_eof_states = Vec::new();

        // Pointer identity is used only to avoid hashing the same immutable
        // physical Action repeatedly, notably for DEFAULT-compressed rows.
        // Content identity, not pointer identity, assigns the action ID.
        let mut physical = FxHashMap::<usize, u32>::default();
        let mut content = FxHashMap::<&'a Action, u32>::default();
        let mut logical_actions = 0usize;

        for (source, row) in table.action.iter().enumerate() {
            let source = source as u32;
            for (terminal, action) in row.iter() {
                if terminal == EOF {
                    eof_states.push(source);
                    if matches!(
                        action,
                        Action::Accept | Action::Split { accept: true, .. }
                    ) {
                        accepting_eof_states.push(source);
                    }
                }
                let Some(valid) = valid_entries.get_mut(terminal as usize) else {
                    continue;
                };
                if *valid {
                    *valid = match action {
                        Action::Shift(..) | Action::Reduce(..) => true,
                        Action::Split {
                            shift: Some(_),
                            reduces,
                            accept: false,
                        } => reduces.is_empty(),
                        Action::Split {
                            shift: None,
                            reduces,
                            accept: false,
                        } => !reduces.is_empty(),
                        _ => false,
                    };
                }
                if !selected[terminal as usize] {
                    continue;
                }
                logical_actions = logical_actions.saturating_add(1);
                let address = action as *const Action as usize;
                let action_id = if let Some(&id) = physical.get(&address) {
                    id
                } else {
                    let next = u32::try_from(content.len())
                        .ok()
                        .filter(|&id| id != u32::MAX)
                        .ok_or("too many distinct compiler actions")?;
                    let id = *content.entry(action).or_insert(next);
                    physical.insert(address, id);
                    id
                };
                let forwarded = table.forwarded_shifts.contains(&(source, terminal));
                action_states[terminal as usize].push(source);
                signatures[terminal as usize].push((source, action_id, forwarded));
            }
        }

        for &(source, terminal) in &table.forwarded_shifts {
            if let Some(valid) = valid_entries.get_mut(terminal as usize) {
                *valid = false;
            }
            if source >= table.num_states {
                return Err("forwarded shift lies outside the compiler state coordinate".into());
            }
            if selected.get(terminal as usize).copied().unwrap_or(false)
                && table.action(source, terminal).is_none()
            {
                signatures[terminal as usize].push((source, u32::MAX, true));
            }
        }
        for signature in &mut signatures {
            signature.sort_unstable();
            signature.dedup();
        }

        let mut predecessors = vec![Vec::new(); table.num_states as usize];
        for (source, row) in table.goto.iter().enumerate() {
            for (&nt, &(target, replace)) in row.iter() {
                let Some(targets) = predecessors.get_mut(target as usize) else {
                    return Err(format!(
                        "compiler GOTO from state {source} targets missing state {target}"
                    ));
                };
                targets.push((source as u32, nt, replace));
            }
        }

        let unique_actions = content.len();
        Ok(Self {
            table,
            selected: selected.to_vec(),
            action_states,
            signatures,
            predecessors,
            eof_states,
            accepting_eof_states,
            valid_entries: valid_entries
                .into_iter()
                .enumerate()
                .filter_map(|(terminal, valid)| valid.then_some(terminal as u32))
                .collect(),
            unique_actions,
            logical_actions,
        })
    }

    pub fn table(&self) -> &'a GLRTable {
        self.table
    }

    pub fn valid_slot_entry_terminals(&self) -> &BTreeSet<TerminalID> {
        &self.valid_entries
    }

    pub fn accepting_eof_states(&self) -> &[u32] {
        &self.accepting_eof_states
    }

    pub fn unique_action_count(&self) -> usize {
        self.unique_actions
    }

    pub fn logical_action_count(&self) -> usize {
        self.logical_actions
    }

    /// The same canonical return-depth rule as subgrammar_child_return_pop,
    /// using the EOF inventory already collected by the shared index.
    pub fn canonical_return_pop(&self) -> Result<u32, String> {
        let Some(augmented) = self.table.rules.first() else {
            return Err("standalone child table contains no augmented-start rule".into());
        };
        let root = match augmented.rhs.as_slice() {
            [Symbol::Nonterminal(root)] => *root,
            rhs => {
                return Err(format!(
                    "standalone child augmented-start rule must contain exactly one nonterminal, found {rhs:?}"
                ));
            }
        };
        let pure_accepts = self.eof_states.iter().copied().filter(|&state| {
            matches!(self.table.action(state, EOF), Some(Action::Accept))
        }).collect::<Vec<_>>();
        let accept = match pure_accepts.as_slice() {
            [state] => *state,
            _ => {
                return Err(format!(
                    "standalone child table must have exactly one EOF accept state, found {}",
                    pure_accepts.len()
                ));
            }
        };
        let Some((target, replace)) = self.table.goto_target(0, root) else {
            return Err("child start row has no goto for its root nonterminal".into());
        };
        if target != accept {
            return Err(format!(
                "child root goto targets state {target}, expected accept state {accept}"
            ));
        }
        Ok(if replace { 1 } else { 2 })
    }

    fn action_groups(&self) -> Vec<Vec<TerminalID>> {
        if flag("GLRMASK_DISABLE_CHARACTERIZATION_QUOTIENT") {
            return self.selected.iter().enumerate()
                .filter_map(|(terminal, &selected)| {
                    selected.then_some(vec![terminal as u32])
                })
                .collect();
        }
        let mut groups = FxHashMap::<&[(u32, u32, bool)], Vec<u32>>::default();
        for (terminal, &selected) in self.selected.iter().enumerate() {
            if selected {
                groups.entry(self.signatures[terminal].as_slice())
                    .or_default().push(terminal as u32);
            }
        }
        let mut groups = groups.into_values().collect::<Vec<_>>();
        groups.sort_unstable_by_key(|group| group[0]);
        groups
    }

    fn validate_sparse_signature(&self, terminal: u32) {
        let indexed = self.signatures[terminal as usize].iter()
            .map(|&(state, id, forwarded)| {
                let action = if id == u32::MAX {
                    None
                } else {
                    Some(self.table.action(state, terminal)
                        .expect("indexed compiler action exists"))
                };
                (state, action, forwarded)
            })
            .collect::<Vec<_>>();
        let dense = (0..self.table.num_states).filter_map(|state| {
            let action = self.table.action(state, terminal);
            let forwarded = self.table.forwarded_shifts.contains(&(state, terminal));
            (action.is_some() || forwarded).then_some((state, action, forwarded))
        }).collect::<Vec<_>>();
        assert_eq!(indexed, dense, "native sparse action signature terminal {terminal}");
    }

    pub fn completion(&self) -> Result<TerminalCharacterization, String> {
        if !self.table.control_terminals.is_empty() {
            return Err(
                "standalone completion templates do not support linker-control closure".into()
            );
        }
        let result = Context { index: self, mode: Mode::Completion }.characterize()?;
        if super::compiler_template_validation_enabled() {
            let expected = super::characterize::characterize_completion_domain(self.table)?;
            assert_eq!(result, expected, "borrowed EOF projection changed completion");
        }
        Ok(result)
    }

    pub fn finish(&self, policy: FinishEndpointPolicy) -> Result<FinishTransfer, String> {
        if policy.return_pop != 1 && policy.return_pop != 2 {
            return Err(format!(
                "finish transfer requires canonical return_pop 1|2, got {}",
                policy.return_pop
            ));
        }
        if let Some(start) = policy.nullable_child_start
            && start >= self.table.num_states
        {
            return Err(format!(
                "finish transfer nullable child start {start} outside {} states",
                self.table.num_states
            ));
        }
        for &state in &self.eof_states {
            if matches!(
                self.table.action(state, EOF),
                Some(Action::Split { accept: true, .. })
            ) {
                return Err(format!(
                    "finish transfer unsupported: child state {state} has a mixed accepting EOF Split"
                ));
            }
        }
        if self.table.forwarded_shifts.iter().any(|&(_, terminal)| terminal == EOF) {
            return Err(
                "finish transfer unsupported: forwarded shift involving EOF needs provider-conformance audit"
                    .into()
            );
        }

        let has_local_eof_effects = self.eof_states.iter().any(|&state| {
            match self.table.action(state, EOF) {
                Some(Action::Accept | Action::Reduce(..)) => false,
                Some(Action::Split { shift: None, reduces, accept: false }) => {
                    reduces.is_empty()
                }
                Some(_) => true,
                None => false,
            }
        });
        let characterization = Context { index: self, mode: Mode::Finish(policy) }
            .characterize()?;
        if super::compiler_template_validation_enabled() {
            let expected = super::characterize::characterize_finish_transfer(
                self.table, &policy,
            )?;
            assert_eq!(characterization, expected.characterization);
            assert_eq!(has_local_eof_effects, expected.has_local_eof_effects);
        }
        Ok(FinishTransfer { characterization, has_local_eof_effects })
    }

    /// Characterize, exactly intern, and fully compile selected terminal groups.
    ///
    /// `build` must depend only on the characterization and immutable
    /// parser-wide facts. It must not attach representative-terminal identity.
    pub fn compile_selected<T, F>(
        &self,
        build: F,
    ) -> Result<CompiledSelection<T>, NativeCompileError>
    where
        T: Send + Sync,
        F: Fn(&TerminalCharacterization) -> Result<T, String> + Send + Sync,
    {
        type Cell<T> = Arc<OnceLock<Result<Arc<T>, String>>>;
        struct Entry<T> {
            characterization: Arc<TerminalCharacterization>,
            result: Cell<T>,
        }
        struct Row<T> {
            terminals: Vec<u32>,
            characterization: Arc<TerminalCharacterization>,
            result: Cell<T>,
        }

        let groups = self.action_groups();
        let action_signature_groups = groups.len();
        let validate_quotient = super::compiler_template_validation_enabled()
            || flag("GLRMASK_VALIDATE_CHARACTERIZATION_QUOTIENT");
        let validate_signatures = super::compiler_template_validation_enabled()
            || flag("GLRMASK_VALIDATE_SPARSE_ACTION_SIGNATURES");
        let cache = Mutex::new(FxHashMap::<u64, Vec<Entry<T>>>::default());

        let run = |terminals: &Vec<u32>| -> Result<Row<T>, NativeCompileError> {
            let representative = terminals[0];
            if validate_signatures {
                for &terminal in terminals {
                    self.validate_sparse_signature(terminal);
                }
            }
            let characterization = Context {
                index: self,
                mode: Mode::Terminal(representative),
            }.characterize().map_err(|message| NativeCompileError {
                terminal: representative,
                message,
            })?;

            if validate_quotient {
                for &terminal in terminals.iter().skip(1) {
                    let direct = Context {
                        index: self,
                        mode: Mode::Terminal(terminal),
                    }.characterize().map_err(|message| NativeCompileError {
                        terminal,
                        message,
                    })?;
                    assert_eq!(
                        characterization, direct,
                        "native action quotient member terminal {terminal}"
                    );
                }
            }

            // Hash outside the lock. Equality is still exact inside the
            // selected bucket. Only unique characterizations remain owned.
            let fingerprint = characterization_fingerprint(&characterization);
            let (characterization, result, owner) = {
                let mut cache = cache.lock().expect("native compiler interner poisoned");
                let bucket = cache.entry(fingerprint).or_default();
                if let Some(entry) = bucket.iter().find(|entry| {
                    entry.characterization.as_ref() == &characterization
                }) {
                    (
                        Arc::clone(&entry.characterization),
                        Arc::clone(&entry.result),
                        false,
                    )
                } else {
                    let characterization = Arc::new(characterization);
                    let result = Arc::new(OnceLock::new());
                    bucket.push(Entry {
                        characterization: Arc::clone(&characterization),
                        result: Arc::clone(&result),
                    });
                    (characterization, result, true)
                }
            };
            if owner {
                let compiled = build(&characterization).map(Arc::new);
                assert!(result.set(compiled).is_ok(), "one compiler owner per exact key");
            }
            // A losing task never waits for the owner here.
            Ok(Row {
                terminals: terminals.clone(),
                characterization,
                result,
            })
        };

        let rows = if parallel() {
            groups.par_iter().map(run).collect::<Vec<_>>()
        } else {
            groups.iter().map(run).collect::<Vec<_>>()
        };

        // All owners have now finished. Select the smallest failing terminal,
        // independently of task completion order and interner winner identity.
        let mut failure: Option<NativeCompileError> = None;
        for row in &rows {
            let error = match row {
                Err(error) => Some(error.clone()),
                Ok(row) => match row.result.get().expect("owner completed before join") {
                    Ok(_) => None,
                    Err(message) => Some(NativeCompileError {
                        terminal: row.terminals[0],
                        message: message.clone(),
                    }),
                },
            };
            if let Some(error) = error
                && failure.as_ref().is_none_or(|old| error.terminal < old.terminal)
            {
                failure = Some(error);
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        let rows = rows.into_iter().map(|row| {
            row.unwrap_or_else(|_| unreachable!("failures handled after the join"))
        }).collect::<Vec<_>>();

        if validate_quotient {
            let reference =
                super::characterize::try_characterize_selected_terminals_for_terminal_count(
                    self.table, self.table.num_terminals, &self.selected,
                ).map_err(|message| NativeCompileError { terminal: 0, message })?;
            for row in &rows {
                for &terminal in &row.terminals {
                    assert_eq!(
                        reference.get(&terminal),
                        Some(row.characterization.as_ref()),
                        "native characterization differs from established relation for terminal {terminal}"
                    );
                }
            }
        }

        let mut group_by_cell = FxHashMap::<usize, usize>::default();
        let mut output = Vec::<CompiledGroup<T>>::new();
        for row in rows {
            let identity = Arc::as_ptr(&row.result) as usize;
            if let Some(&group) = group_by_cell.get(&identity) {
                output[group].terminals.extend(row.terminals);
            } else {
                let program = match row.result.get().expect("joined owner") {
                    Ok(program) => Arc::clone(program),
                    Err(_) => unreachable!("failures handled above"),
                };
                group_by_cell.insert(identity, output.len());
                output.push(CompiledGroup { terminals: row.terminals, program });
            }
        }
        for group in &mut output {
            group.terminals.sort_unstable();
        }
        output.sort_unstable_by_key(|group| group.terminals[0]);
        let exact_characterization_groups = output.len();
        Ok(CompiledSelection {
            groups: output,
            action_signature_groups,
            exact_characterization_groups,
        })
    }
}

fn characterization_fingerprint(c: &TerminalCharacterization) -> u64 {
    let mut hash = FxHasher::default();
    c.escapes.hash(&mut hash);
    c.reduces.hash(&mut hash);
    c.nt_escapes.hash(&mut hash);
    c.nt_rereduces.hash(&mut hash);
    c.all_nts.hash(&mut hash);
    hash.finish()
}

#[derive(Clone, Copy)]
enum Mode {
    Terminal(u32),
    Completion,
    Finish(FinishEndpointPolicy),
}

#[derive(Clone, Copy)]
enum Source {
    Initial,
    Nonterminal(u32),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Config {
    input: Vec<StackMatcher>,
    segment: Vec<u32>,
}

#[derive(Default)]
struct Output {
    escapes: Vec<InitialEscape>,
    reduces: Vec<InitialReduce>,
    nt_escapes: Vec<NtEscape>,
    nt_rereduces: Vec<NtRereduce>,
}

impl Output {
    fn extend(&mut self, other: Self) {
        self.escapes.extend(other.escapes);
        self.reduces.extend(other.reduces);
        self.nt_escapes.extend(other.nt_escapes);
        self.nt_rereduces.extend(other.nt_rereduces);
    }

    fn escape(&mut self, source: Source, pop: Vec<StackMatcher>, pushes: Vec<u32>) {
        match source {
            Source::Initial => self.escapes.push(InitialEscape { pop, pushes }),
            Source::Nonterminal(source_nonterminal) => self.nt_escapes.push(
                NtEscape { source_nonterminal, pop, pushes }
            ),
        }
    }

    fn reduce(&mut self, source: Source, pop: Vec<StackMatcher>, nt: u32) {
        debug_assert!(!pop.is_empty());
        match source {
            Source::Initial => self.reduces.push(InitialReduce { pop, nonterminal: nt }),
            Source::Nonterminal(source_nonterminal) => self.nt_rereduces.push(
                NtRereduce {
                    source_nonterminal,
                    pop,
                    target_nonterminal: nt,
                }
            ),
        }
    }

    fn finish(mut self) -> TerminalCharacterization {
        self.escapes.sort_unstable();
        self.escapes.dedup();
        self.reduces.sort_unstable();
        self.reduces.dedup();
        self.nt_escapes.sort_unstable();
        self.nt_escapes.dedup();
        self.nt_rereduces.sort_unstable();
        self.nt_rereduces.dedup();
        let mut all_nts = BTreeSet::new();
        all_nts.extend(self.reduces.iter().map(|r| r.nonterminal));
        all_nts.extend(self.nt_escapes.iter().map(|r| r.source_nonterminal));
        for r in &self.nt_rereduces {
            all_nts.insert(r.source_nonterminal);
            all_nts.insert(r.target_nonterminal);
        }
        TerminalCharacterization {
            escapes: self.escapes,
            reduces: self.reduces,
            nt_escapes: self.nt_escapes,
            nt_rereduces: self.nt_rereduces,
            all_nts,
        }
    }
}

struct Context<'i, 't> {
    index: &'i NativeTableIndex<'t>,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum UnitResolution {
    Pending,
    Resolved(Option<(u32, bool)>),
}

struct UnitSplit<'a> {
    shift: Option<(u32, bool)>,
    reduces: &'a [(u32, u32)],
    units: FxHashMap<u32, Vec<usize>>,
    others: Vec<usize>,
}

impl<'a> UnitSplit<'a> {
    fn new(action: &'a Action) -> Option<Self> {
        let Action::Split { shift, reduces, .. } = action else {
            return None;
        };
        let units_count = reduces.iter().filter(|(_, len)| *len == 1).count();
        if reduces.len() < 16
            || units_count < 16
            || units_count < reduces.len() - reduces.len() / 4
        {
            return None;
        }
        let mut units = FxHashMap::<u32, Vec<usize>>::default();
        let mut others = Vec::new();
        for (position, &(lhs, len)) in reduces.iter().enumerate() {
            if len == 1 {
                units.entry(lhs).or_default().push(position);
            } else {
                others.push(position);
            }
        }
        Some(Self { shift: *shift, reduces, units, others })
    }

    fn positions(&self, table: &GLRTable, revealed: u32) -> Vec<usize> {
        let mut positions = self.others.clone();
        if let Some(row) = table.goto.get(revealed as usize) {
            for (&lhs, _) in row.iter() {
                if let Some(found) = self.units.get(&lhs) {
                    positions.extend_from_slice(found);
                }
            }
        }
        positions.sort_unstable();
        positions
    }
}

impl Context<'_, '_> {
    fn tops(&self) -> &[u32] {
        match self.mode {
            Mode::Terminal(terminal) => &self.index.action_states[terminal as usize],
            Mode::Completion | Mode::Finish(_) => &self.index.eof_states,
        }
    }

    fn action(&self, top: u32) -> Option<&Action> {
        match self.mode {
            Mode::Terminal(terminal) => self.index.table.action(top, terminal),
            Mode::Finish(_) => self.index.table.action(top, EOF),
            Mode::Completion => {
                if !self.index.table.advance_row_allows(top, EOF) {
                    return None;
                }
                match self.index.table.action(top, EOF) {
                    action @ Some(Action::Accept | Action::Reduce(..)) => action,
                    action @ Some(Action::Split { accept: true, .. }) => action,
                    action @ Some(Action::Split { reduces, .. }) if !reduces.is_empty() => action,
                    _ => None,
                }
            }
        }
    }

    fn identity(top: u32) -> Config {
        Config {
            input: vec![StackMatcher::State(top)],
            segment: vec![top],
        }
    }

    fn after_goto(revealed: u32, target: u32, replace: bool) -> Config {
        Config {
            input: vec![StackMatcher::State(revealed)],
            segment: if replace { vec![target] } else { vec![revealed, target] },
        }
    }

    fn reduce(
        &self,
        source: Source,
        config: &Config,
        lhs: u32,
        len: usize,
        output: &mut Output,
        seen: &mut FxHashSet<Config>,
        pending: &mut VecDeque<Config>,
    ) {
        if len >= config.segment.len() && len != 0 {
            let mut pop = config.input.clone();
            pop.extend(std::iter::repeat_n(
                StackMatcher::Any, len - config.segment.len(),
            ));
            output.reduce(source, pop, lhs);
            return;
        }
        let Some(reveal_index) = config.segment.len().checked_sub(len + 1) else {
            return;
        };
        let revealed = config.segment[reveal_index];
        let Some((target, replace)) = self.index.table.goto_target(revealed, lhs) else {
            return;
        };
        let mut segment = config.segment[..=reveal_index].to_vec();
        if replace {
            segment.pop();
        }
        segment.push(target);
        let next = Config { input: config.input.clone(), segment };
        if seen.insert(next.clone()) {
            pending.push_back(next);
        }
    }

    fn emit(
        source: Source,
        config: &Config,
        pop: usize,
        pushes: &[u32],
        guards: &[StackShiftGuard],
        output: &mut Output,
    ) {
        let known = config.segment.len();
        let base = config.input.len();
        let mut input = config.input.clone();
        let mut previous = None;
        let mut revealed_guarded = false;

        for guard in guards {
            let depth = guard.pop as usize;
            let allowed = guard.states.as_slice();
            if previous.is_some_and(|previous| depth < previous)
                || depth > pop
                || allowed.is_empty()
            {
                return;
            }
            previous = Some(depth);
            if depth < known {
                if allowed.binary_search(&config.segment[known - 1 - depth]).is_err() {
                    return;
                }
            } else {
                let at = base + depth - known;
                if input.len() <= at {
                    input.resize(at + 1, StackMatcher::Any);
                }
                if !constrain(&mut input[at], allowed) {
                    return;
                }
            }
            revealed_guarded |= depth == pop;
        }

        let unknown_popped = pop.saturating_sub(known);
        if input.len() < base + unknown_popped {
            input.resize(base + unknown_popped, StackMatcher::Any);
        }
        if pop < known {
            let mut next = config.segment[..known - pop].to_vec();
            next.extend_from_slice(pushes);
            output.escape(source, input, next);
        } else if revealed_guarded {
            let at = base + unknown_popped;
            if input.len() <= at {
                input.resize(at + 1, StackMatcher::Any);
            }
            let states = match &input[at] {
                StackMatcher::State(state) => vec![*state],
                StackMatcher::States(states) => states.clone(),
                StackMatcher::Any => unreachable!("a revealed guard is finite"),
            };
            for state in states {
                let mut branch = input.clone();
                branch[at] = StackMatcher::State(state);
                let mut next = Vec::with_capacity(pushes.len() + 1);
                next.push(state);
                next.extend_from_slice(pushes);
                output.escape(source, branch, next);
            }
        } else {
            output.escape(source, input, pushes.to_vec());
        }
    }

    fn process(
        &self,
        source: Source,
        config: &Config,
        action: Option<&Action>,
        adjustment: usize,
        output: &mut Output,
        seen: &mut FxHashSet<Config>,
        pending: &mut VecDeque<Config>,
    ) {
        if matches!(self.mode, Mode::Completion) {
            match action {
                Some(Action::Accept | Action::Split { accept: true, .. }) => {
                    Self::emit(source, config, 0, &[], &[], output);
                }
                Some(Action::Reduce(lhs, len)) => {
                    self.reduce(source, config, *lhs, *len as usize, output, seen, pending);
                }
                Some(Action::Split { reduces, .. }) => {
                    for &(lhs, len) in reduces {
                        self.reduce(source, config, lhs, len as usize, output, seen, pending);
                    }
                }
                _ => {}
            }
            return;
        }

        match action {
            Some(Action::Accept) => {
                if let Mode::Finish(policy) = self.mode {
                    Self::emit(
                        source, config, policy.return_pop as usize, &[], &[], output,
                    );
                }
            }
            Some(Action::Shift(target, replace)) => {
                Self::emit(
                    source, config, usize::from(*replace),
                    std::slice::from_ref(target), &[], output,
                );
            }
            Some(Action::ReplaceShifts(targets)) => {
                for target in targets.iter() {
                    Self::emit(
                        source, config, 1, std::slice::from_ref(target), &[], output,
                    );
                }
            }
            Some(Action::StackShifts(shifts)) => {
                for shift in shifts {
                    Self::emit(
                        source, config, shift.pop as usize, &shift.pushes, &[], output,
                    );
                }
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                for shift in shifts {
                    Self::emit(
                        source, config, shift.pop as usize,
                        &shift.pushes, &shift.guards, output,
                    );
                }
            }
            Some(Action::Skip) => Self::emit(source, config, 0, &[], &[], output),
            Some(Action::Reduce(lhs, len)) => {
                self.reduce(
                    source, config, *lhs, *len as usize + adjustment,
                    output, seen, pending,
                );
            }
            Some(Action::Split { shift, reduces, .. }) => {
                if let Some((target, replace)) = shift {
                    Self::emit(
                        source, config, usize::from(*replace),
                        std::slice::from_ref(target), &[], output,
                    );
                }
                for &(lhs, len) in reduces {
                    self.reduce(
                        source, config, lhs, len as usize + adjustment,
                        output, seen, pending,
                    );
                }
            }
            None => {}
        }
        if let Mode::Finish(policy) = self.mode
            && policy.nullable_child_start == config.segment.last().copied()
        {
            Self::emit(source, config, 1, &[], &[], output);
        }
    }

    fn drain(
        &self,
        source: Source,
        output: &mut Output,
        seen: &mut FxHashSet<Config>,
        pending: &mut VecDeque<Config>,
    ) {
        while let Some(config) = pending.pop_front() {
            let Some(&top) = config.segment.last() else { continue; };
            self.process(
                source, &config, self.action(top), 0, output, seen, pending,
            );
        }
    }

    fn seed_has_effect(
        &self, action: Option<&Action>, revealed: u32, top: u32, replace: bool,
    ) -> bool {
        if matches!(self.mode, Mode::Finish(_)) {
            return true;
        }
        let reduction = |lhs, len| {
            if len >= if replace { 1 } else { 2 } {
                true
            } else {
                self.index.table.goto_target(
                    if len == 0 { top } else { revealed }, lhs,
                ).is_some()
            }
        };
        match action {
            None => false,
            Some(Action::Accept) => matches!(self.mode, Mode::Completion),
            Some(Action::Reduce(lhs, len)) => reduction(*lhs, *len),
            Some(Action::Split { shift, reduces, accept }) => {
                (matches!(self.mode, Mode::Completion) && *accept)
                    || (!matches!(self.mode, Mode::Completion) && shift.is_some())
                    || reduces.iter().any(|&(lhs, len)| reduction(lhs, len))
            }
            _ => true,
        }
    }

    fn resolve_unit(
        &self,
        revealed: u32,
        top: u32,
        cache: &mut FxHashMap<(u32, u32), UnitResolution>,
    ) -> Option<(u32, bool)> {
        const MAX_MEMO_ENTRIES: usize = 16_384;
        let mut trail = Vec::new();
        let mut current = top;
        let result = loop {
            match cache.get(&(revealed, current)).copied() {
                Some(UnitResolution::Resolved(result)) => break result,
                Some(UnitResolution::Pending) => break None,
                None => {}
            }
            if cache.len() >= MAX_MEMO_ENTRIES {
                break Some((current, false));
            }
            cache.insert((revealed, current), UnitResolution::Pending);
            trail.push(current);
            let Some(Action::Reduce(lhs, 1)) = self.action(current) else {
                break Some((current, false));
            };
            match self.index.table.goto_target(revealed, *lhs) {
                Some((next, false)) => current = next,
                Some((next, true)) => break Some((next, true)),
                None => break None,
            }
        };
        for visited in trail {
            cache.insert((revealed, visited), UnitResolution::Resolved(result));
        }
        result
    }

    fn continuations(&self, tops: &[u32]) -> Output {
        let mut output = Output::default();
        let mut unit_cache = FxHashMap::default();
        for &top in tops {
            let predecessors = &self.index.predecessors[top as usize];
            let action = self.action(top);
            let can_index = !matches!(self.mode, Mode::Finish(_))
                && !matches!(
                    (self.mode, action),
                    (Mode::Completion, Some(Action::Split { accept: true, .. }))
                );
            let split_index = if can_index && predecessors.len() >= 4 {
                action.and_then(UnitSplit::new)
            } else {
                None
            };
            for &(revealed, nt, replace) in predecessors {
                let (seed_top, seed_replace) = if can_index && !replace
                    && matches!(action, Some(Action::Reduce(_, 1)))
                {
                    let Some(resolved) = self.resolve_unit(revealed, top, &mut unit_cache)
                    else { continue; };
                    resolved
                } else {
                    (top, replace)
                };
                let seed_action = self.action(seed_top);
                if !self.seed_has_effect(seed_action, revealed, seed_top, seed_replace) {
                    continue;
                }
                let config = Self::after_goto(revealed, seed_top, seed_replace);
                let source = Source::Nonterminal(nt);
                let mut seen = FxHashSet::default();
                seen.insert(config.clone());
                let mut pending = VecDeque::new();

                if seed_top == top && !seed_replace
                    && let Some(index) = &split_index
                {
                    if !matches!(self.mode, Mode::Completion)
                        && let Some((target, replace)) = index.shift
                    {
                        Self::emit(
                            source, &config, usize::from(replace), &[target], &[], &mut output,
                        );
                    }
                    for position in index.positions(self.index.table, revealed) {
                        let (lhs, len) = index.reduces[position];
                        self.reduce(
                            source, &config, lhs, len as usize,
                            &mut output, &mut seen, &mut pending,
                        );
                    }
                } else {
                    pending.push_back(config);
                }
                self.drain(source, &mut output, &mut seen, &mut pending);
            }
        }
        output
    }

    fn characterize(&self) -> Result<TerminalCharacterization, String> {
        let mut output = Output::default();
        for &top in self.tops() {
            let action = self.action(top);
            if action.is_none() && !matches!(self.mode, Mode::Finish(_)) {
                continue;
            }
            let config = Self::identity(top);
            let mut seen = FxHashSet::default();
            seen.insert(config.clone());
            let mut pending = VecDeque::new();
            let adjustment = match self.mode {
                Mode::Terminal(terminal) => usize::from(
                    self.index.table.forwarded_shifts.contains(&(top, terminal))
                ),
                _ => 0,
            };
            self.process(
                Source::Initial, &config, action, adjustment,
                &mut output, &mut seen, &mut pending,
            );
            self.drain(Source::Initial, &mut output, &mut seen, &mut pending);
        }

        if let Mode::Finish(policy) = self.mode
            && let Some(start) = policy.nullable_child_start
            && self.index.table.action(start, EOF).is_none()
        {
            output.escape(
                Source::Initial, vec![StackMatcher::State(start)], Vec::new(),
            );
        }

        if parallel() && self.tops().len() >= 256 {
            let chunks = self.tops().par_chunks(64)
                .map(|tops| self.continuations(tops))
                .collect::<Vec<_>>();
            for chunk in chunks {
                output.extend(chunk);
            }
        } else {
            output.extend(self.continuations(self.tops()));
        }

        let result = output.finish();
        if let Some(cycle) = result.find_cycle() {
            return Err(match self.mode {
                Mode::Terminal(terminal) => format!(
                    "terminal characterization for terminal {terminal} contains a reduction cycle: {cycle:?}"
                ),
                Mode::Completion => format!(
                    "completion template requires a cyclic reduction relation: {cycle:?}"
                ),
                Mode::Finish(_) => format!(
                    "finish transfer has a nonterminal re-reduction cycle (boundedness not certified): {cycle:?}"
                ),
            });
        }
        Ok(result)
    }
}

fn constrain(matcher: &mut StackMatcher, allowed: &[u32]) -> bool {
    let states = match matcher {
        StackMatcher::Any => allowed.to_vec(),
        StackMatcher::State(state) => {
            if allowed.binary_search(state).is_err() {
                return false;
            }
            return true;
        }
        StackMatcher::States(states) => {
            let mut out = Vec::new();
            let (mut i, mut j) = (0, 0);
            while i < states.len() && j < allowed.len() {
                match states[i].cmp(&allowed[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        out.push(states[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            out
        }
    };
    match states.as_slice() {
        [] => false,
        [state] => {
            *matcher = StackMatcher::State(*state);
            true
        }
        _ => {
            *matcher = StackMatcher::States(states);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use crate::compiler::glr::table::testing::build_test_table;

    #[test]
    fn streaming_exact_quotient_compiles_once_without_waiting_on_aliases() {
        let table = build_test_table(
            2, 5,
            &[
                &[
                    (0, Action::Shift(1, false)),
                    (1, Action::Shift(1, false)),
                    (2, Action::Shift(1, false)),
                    (4, Action::Accept),
                ],
                &[(2, Action::Accept), (EOF, Action::Accept)],
            ],
            &[&[], &[]],
        );
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads).build().unwrap();
            let index = NativeTableIndex::new(&table, &[true; 5]).unwrap();
            let builds = AtomicUsize::new(0);
            let result = pool.install(|| index.compile_selected(|c| {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok(c.escapes.len())
            })).unwrap();
            assert_eq!(result.action_signature_groups, 4);
            assert_eq!(result.exact_characterization_groups, 2);
            assert_eq!(builds.load(Ordering::Relaxed), 2);
            assert_eq!(result.groups[0].terminals, vec![0, 1, 2]);
            assert_eq!(result.groups[1].terminals, vec![3, 4]);
            assert_eq!(*result.groups[0].program, 1);
            assert_eq!(*result.groups[1].program, 0);
        }
    }

    #[test]
    fn borrowed_completion_and_finish_match_established_projection() {
        for replace in [false, true] {
            let table = build_test_table(
                4, 2,
                &[
                    &[(0, Action::Shift(1, false))],
                    &[(EOF, Action::Reduce(0, 1))],
                    &[(EOF, Action::Accept)],
                    &[(1, Action::Skip)],
                ],
                &[&[(0, (2, replace))], &[], &[], &[]],
            );
            let index = NativeTableIndex::new(&table, &[true, true]).unwrap();
            assert_eq!(
                index.completion().unwrap(),
                super::super::characterize::characterize_completion_domain(&table).unwrap()
            );
            for nullable in [None, Some(0)] {
                let policy = FinishEndpointPolicy {
                    return_pop: if replace { 1 } else { 2 },
                    nullable_child_start: nullable,
                };
                let actual = index.finish(policy).unwrap();
                let expected =
                    super::super::characterize::characterize_finish_transfer(&table, &policy)
                        .unwrap();
                assert_eq!(actual.characterization, expected.characterization);
                assert_eq!(actual.has_local_eof_effects, expected.has_local_eof_effects);
            }
        }
    }

    #[test]
    fn entry_certificate_includes_absent_slots_and_excludes_forwarding() {
        let mut table = build_test_table(
            2, 6,
            &[
                &[
                    (0, Action::Shift(1, false)),
                    (1, Action::Reduce(0, 1)),
                    (2, Action::Skip),
                    (3, Action::Split {
                        shift: Some((1, false)),
                        reduces: vec![(0, 1)],
                        accept: false,
                    }),
                ],
                &[],
            ],
            &[&[], &[]],
        );
        table.forwarded_shifts.insert((0, 0));
        let index = NativeTableIndex::new(&table, &[false; 6]).unwrap();
        assert_eq!(
            index.valid_slot_entry_terminals(),
            &BTreeSet::from([1, 4, 5])
        );
    }

    #[test]
    fn smallest_compilation_error_is_independent_of_interner_owner() {
        let table = build_test_table(
            1, 8,
            &[&[
                (0, Action::Skip), (1, Action::Skip),
                (2, Action::Accept), (3, Action::Accept),
                (4, Action::Skip), (5, Action::Accept),
                (6, Action::Skip), (7, Action::Accept),
            ]],
            &[&[]],
        );
        let index = NativeTableIndex::new(&table, &[true; 8]).unwrap();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads).build().unwrap();
            for _ in 0..8 {
                let error = pool.install(|| index.compile_selected::<(), _>(|_| {
                    Err("deliberate construction failure".into())
                })).err().unwrap();
                assert_eq!(error.terminal, 0);
            }
        }
    }

    #[test]
    fn action_identity_interns_default_physical_values_by_content() {
        let mut table = build_test_table(2, 24, &[&[], &[]], &[&[], &[]]);
        for source in 0..2usize {
            for terminal in 0..24u32 {
                table.action[source].insert(terminal, Action::Reduce(7, 2));
            }
            table.action[source].compress_default(24);
        }
        let index = NativeTableIndex::new(&table, &[true; 24]).unwrap();
        assert_eq!(index.unique_action_count(), 1);
        assert_eq!(index.logical_action_count(), 48);
        assert_eq!(index.action_groups(), vec![(0..24).collect::<Vec<_>>()]);
        for terminal in 0..24 {
            index.validate_sparse_signature(terminal);
        }
    }

    #[test]
    fn cyclic_relations_are_errors_not_bounded_approximations() {
        let table = build_test_table(
            5, 2,
            &[
                &[],
                &[(0, Action::Reduce(0, 2)), (1, Action::Reduce(0, 2))],
                &[],
                &[(0, Action::Shift(4, false)), (1, Action::Shift(4, false))],
                &[],
            ],
            &[&[(0, (1, false))], &[], &[(0, (3, false))], &[], &[]],
        );
        let index = NativeTableIndex::new(&table, &[true, true]).unwrap();
        let error = index.compile_selected(|_| Ok(())).err().unwrap();
        assert_eq!(error.terminal, 0);
        assert!(error.message.contains("reduction cycle"));
    }
}
