use super::*;
use crate::ds::bitset::BitSet;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use rustc_hash::FxHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

const DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV: &str =
    "GLRMASK_DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION";
const DISABLE_RECOGNIZER_SUFFIX_QUOTIENT_ENV: &str =
    "GLRMASK_DISABLE_RECOGNIZER_SUFFIX_QUOTIENT";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_STATES_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_STATES";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_ALTS_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_ALTS";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_WIDTH_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_WIDTH";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_DEPTH_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_DEPTH";
const MAX_GUARDED_STACK_EFFECTS_ENV: &str = "GLRMASK_MAX_GUARDED_STACK_EFFECTS";
const UNIT_INLINE_WORK_MAX_WALL_MS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_MS";
const UNIT_INLINE_WORK_MAX_ITERATIONS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_ITERATIONS";
const UNIT_INLINE_WORK_MAX_CELLS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_CELLS";
const UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES_ENV: &str =
    "GLRMASK_UNIT_REDUCTION_INLINE_MAX_SYNTHETIC_STATES";
const UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_STACK_VISITS";
const UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS_ENV: &str =
    "GLRMASK_UNIT_REDUCTION_INLINE_STATE_MAX_STACK_VISITS";
const DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS: u128 = 5_000;
const DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS: usize = 64;
const DEFAULT_UNIT_INLINE_WORK_MAX_CELLS: usize = 2_000_000;
const DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES: usize = 4_096;
const DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS: usize = 100_000;
const DEFAULT_UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS: usize = 512;
const UNIT_INLINE_COMPLETED_ABORT_ENV: &str =
    "GLRMASK_UNIT_INLINE_COMPLETED_WORK_ABORT";

fn completed_work_abort_eligible(construction: GlrTableConstruction) -> bool {
    // The measured benefit is in the exact core-merged builder. Keep legacy
    // and LALR scheduling untouched instead of charging successful static
    // passes for a certificate they have not been shown to benefit from.
    construction == GlrTableConstruction::ExperimentalCoreMerged
}

// Most unit-inlining states are singleton constituents. Keeping those
// inline avoids thousands of tiny BTreeSet allocations before any synthetic
// merged state is actually needed. Subsets remain sorted and deduplicated.
type StateSubset = SmallVec<[u32; 2]>;
// Runtime predecessors are always kept sorted and deduplicated. Most parser
// states have at most a couple of predecessors, so an inline vector avoids
// the per-state tree allocation and pointer chasing of BTreeSet.
type PredecessorSet = SmallVec<[u32; 2]>;
type RuntimePredecessors = Vec<PredecessorSet>;

fn stack_shift_predecessor_canonicalization_enabled() -> bool {
    !env_flag_enabled(DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV, false)
}

fn recognizer_suffix_quotient_enabled() -> bool {
    !env_flag_enabled(DISABLE_RECOGNIZER_SUFFIX_QUOTIENT_ENV, false)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_u128(name: &str, default: u128) -> u128 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(default)
}

fn unit_inline_state_max_stack_effect_visits() -> usize {
    env_usize(
        UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS_ENV,
        DEFAULT_UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS,
    )
}

fn max_guarded_stack_effects() -> Option<usize> {
    std::env::var(MAX_GUARDED_STACK_EFFECTS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackEffectVisitKey {
    state: u32,
    tid: TerminalID,
    action_tag: u8,
    frame: StackEffectFrame,
}

#[derive(Clone)]
struct StackEffectResult {
    effects: Vec<GuardedStackShift>,
    origin_dependent: bool,
}

#[derive(Debug, Clone)]
pub(super) struct UnitReductionInliningReport {
    pub(super) aborted: bool,
    pub(super) reason: Option<&'static str>,
    pub(super) iterations: usize,
    pub(super) cells: usize,
    pub(super) synthetic_states: usize,
    pub(super) stack_effect_visits: usize,
    pub(super) elapsed_ms: f64,
    pub(super) changed_original_states: Vec<u32>,
}

/// Number of real control-elimination runs process-wide. No-op calls on
/// control-free tables do not count. The prepared static linker's
/// signed-transfer boundary compiler asserts this does not move across a link.
static CONTROL_ELIMINATION_RUNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Process-wide count of real `eliminate_control_terminals_exact` runs.
pub fn control_elimination_run_count() -> u64 {
    CONTROL_ELIMINATION_RUNS.load(std::sync::atomic::Ordering::SeqCst)
}

#[derive(Debug, Clone)]
pub struct ControlEliminationReport {
    pub states: usize,
    pub controls: usize,
    pub compiled_cells: usize,
    pub closure_frames: usize,
    pub stack_effect_visits: usize,
    /// Real terminal columns whose consuming action changed while compiling
    /// zero-width control closure. Component-local templates remain exact for
    /// every other terminal and can therefore be transported directly.
    pub changed_terminals: Vec<TerminalID>,
    pub elapsed_ms: f64,
}

struct UnitInlineUndo {
    original_num_states: u32,
    original_action_len: usize,
    original_goto_len: usize,
    original_advance_len: usize,
    original_cells: FxHashMap<(usize, TerminalID), Option<Action>>,
    original_advance_rows: FxHashMap<usize, BitSet>,
}

impl UnitInlineUndo {
    fn new(table: &GLRTable) -> Self {
        Self {
            original_num_states: table.num_states,
            original_action_len: table.action.len(),
            original_goto_len: table.goto.len(),
            original_advance_len: table.advance.len(),
            original_cells: FxHashMap::default(),
            original_advance_rows: FxHashMap::default(),
        }
    }

    fn record_cell(&mut self, table: &GLRTable, state: usize, tid: TerminalID) {
        if state >= self.original_num_states as usize {
            return;
        }
        self.original_cells
            .entry((state, tid))
            .or_insert_with(|| table.action[state].get(&tid).cloned());
        if self.original_advance_len == self.original_action_len {
            self.original_advance_rows
                .entry(state)
                .or_insert_with(|| table.advance[state].clone());
        }
    }

    fn changed_original_states(&self) -> Vec<u32> {
        let mut states = self
            .original_cells
            .keys()
            .map(|&(state, _)| state as u32)
            .collect::<Vec<_>>();
        states.sort_unstable();
        states.dedup();
        states
    }

    fn rollback(self, table: &mut GLRTable) {
        table.action.truncate(self.original_action_len);
        table.goto.truncate(self.original_goto_len);
        table.advance.truncate(self.original_advance_len);
        table.num_states = self.original_num_states;
        for ((state, tid), action) in self.original_cells {
            match action {
                Some(action) => {
                    table.action[state].insert(tid, action);
                }
                None => {
                    table.action[state].remove(&tid);
                }
            }
        }
        for (state, advance) in self.original_advance_rows {
            table.advance[state] = advance;
        }
    }
}

struct UnitInlineBudget {
    started_at: std::time::Instant,
    max_ms: u128,
    max_iterations: usize,
    max_cells: usize,
    max_synthetic_states: usize,
    max_stack_effect_visits: usize,
    iterations: std::sync::atomic::AtomicUsize,
    cells: std::sync::atomic::AtomicUsize,
    synthetic_states: std::sync::atomic::AtomicUsize,
    stack_effect_visits: std::sync::atomic::AtomicUsize,
    // 0 = not aborted; otherwise a small code identifying the reason. Using an
    // atomic lets the budget be shared across the parallel stack-effect
    // precompute below while keeping serial behaviour byte-for-byte identical.
    abort_code: std::sync::atomic::AtomicU8,
}

/// A lower bound on work the serial fold will charge, not a shared child
/// budget. Only completed jobs publish here, so a scheduling-dependent partial
/// analysis can never cause a successful reference optimization to be lost.
/// No production table changes until the normal serial phase or rollback.
struct CompletedStackEffectWork {
    visits: std::sync::atomic::AtomicUsize,
    limit: usize,
}

impl CompletedStackEffectWork {
    fn new(budget: &UnitInlineBudget) -> Option<Self> {
        // In particular, leave the uncapped correctness oracle unchanged.
        (budget.max_stack_effect_visits != usize::MAX).then(|| Self {
            visits: std::sync::atomic::AtomicUsize::new(budget.stack_effect_visits()),
            limit: budget.max_stack_effect_visits,
        })
    }

    fn record_completed(&self, visits: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        // One uncontended/local-budget total per job, not an atomic per visit.
        // Saturation preserves the exceeded predicate even at usize::MAX-1.
        let _ = self.visits.fetch_update(Relaxed, Relaxed, |total| {
            Some(total.saturating_add(visits).min(self.limit + 1))
        });
    }

    fn exceeded(&self) -> bool {
        self.visits.load(std::sync::atomic::Ordering::Relaxed) > self.limit
    }
}

const ABORT_NONE: u8 = 0;
const ABORT_ELAPSED: u8 = 1;
const ABORT_ITERATIONS: u8 = 2;
const ABORT_CELLS: u8 = 3;
const ABORT_SYNTHETIC_STATES: u8 = 4;
const ABORT_STACK_EFFECT_VISITS: u8 = 5;

/// Convergence caps for exact control elimination (see
/// `exact_control_elimination`). Legitimate tables use a handful of visits
/// (small segmented fixtures: 8–13) and sub-millisecond walls; the
/// pathological cyclic case burns millions of visits without converging, so
/// 100k visits (~0.6 s observed) fails fast with >1000x headroom.
const EXACT_CONTROL_ELIMINATION_MAX_STACK_EFFECT_VISITS: usize = 100_000;
const EXACT_CONTROL_ELIMINATION_MAX_WALL_MS: u128 = 10_000;

fn abort_reason_str(code: u8) -> Option<&'static str> {
    match code {
        ABORT_ELAPSED => Some("elapsed_ms"),
        ABORT_ITERATIONS => Some("iterations"),
        ABORT_CELLS => Some("cells"),
        ABORT_SYNTHETIC_STATES => Some("synthetic_states"),
        ABORT_STACK_EFFECT_VISITS => Some("stack_effect_visits"),
        _ => None,
    }
}

impl UnitInlineBudget {
    fn from_env() -> Self {
        use std::sync::atomic::{AtomicU8, AtomicUsize};
        Self {
            started_at: std::time::Instant::now(),
            max_ms: env_u128(UNIT_INLINE_WORK_MAX_WALL_MS_ENV, DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS),
            max_iterations: env_usize(
                UNIT_INLINE_WORK_MAX_ITERATIONS_ENV,
                DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS,
            ),
            max_cells: env_usize(UNIT_INLINE_WORK_MAX_CELLS_ENV, DEFAULT_UNIT_INLINE_WORK_MAX_CELLS),
            max_synthetic_states: env_usize(
                UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES_ENV,
                DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES,
            ),
            max_stack_effect_visits: env_usize(
                UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV,
                DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS,
            ),
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        }
    }

    /// Exact control elimination is a semantic compilation pass, not an
    /// optional optimizer. It must not succeed or fail according to the
    /// environment limits used to bound speculative unit-reduction inlining,
    /// so the convergence caps below are hardcoded, not env-tunable.
    ///
    /// The caps exist because the symbolic traversal is not guaranteed to
    /// terminate: on cyclic reduce graphs every lap accumulates a distinct
    /// guard set, so the backtracking `visiting` set never repeats a key and
    /// the DFS tree grows without bound (observed: a 7947-state composed
    /// table burning 170k visits/s with no end). The visits cap bounds total
    /// traversal work deterministically; the wall backstop bounds residual
    /// non-visit work (frame cloning, cache-hit closure steps). Exhaustion
    /// reports `reason=stack_effect_visits` / `reason=elapsed_ms`; callers
    /// detect it with `control_elimination_budget_exhausted` and fall back
    /// to dynamic boundary shards (static linking is an accelerator — the
    /// dynamic backend always remains exact).
    fn exact_control_elimination() -> Self {
        use std::sync::atomic::{AtomicU8, AtomicUsize};
        Self {
            started_at: std::time::Instant::now(),
            max_ms: EXACT_CONTROL_ELIMINATION_MAX_WALL_MS,
            max_iterations: usize::MAX,
            max_cells: usize::MAX,
            max_synthetic_states: usize::MAX,
            max_stack_effect_visits: EXACT_CONTROL_ELIMINATION_MAX_STACK_EFFECT_VISITS,
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        }
    }

    fn iterations(&self) -> usize {
        self.iterations.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn cells(&self) -> usize {
        self.cells.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn synthetic_states(&self) -> usize {
        self.synthetic_states.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn stack_effect_visits(&self) -> usize {
        self.stack_effect_visits.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn is_aborted(&self) -> bool {
        self.abort_code.load(std::sync::atomic::Ordering::Relaxed) != ABORT_NONE
    }

    fn report(&self) -> UnitReductionInliningReport {
        let code = self.abort_code.load(std::sync::atomic::Ordering::Relaxed);
        UnitReductionInliningReport {
            aborted: code != ABORT_NONE,
            reason: abort_reason_str(code),
            iterations: self.iterations(),
            cells: self.cells(),
            synthetic_states: self.synthetic_states(),
            stack_effect_visits: self.stack_effect_visits(),
            elapsed_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
            changed_original_states: Vec::new(),
        }
    }

    fn abort(&self, code: u8) {
        let _ = self.abort_code.compare_exchange(
            ABORT_NONE,
            code,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn check_elapsed(&self) -> bool {
        if self.is_aborted() {
            return false;
        }
        if self.started_at.elapsed().as_millis() > self.max_ms {
            self.abort(ABORT_ELAPSED);
            return false;
        }
        true
    }

    fn record_iteration(&self) -> bool {
        let value = self.iterations.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_iterations {
            self.abort(ABORT_ITERATIONS);
            return false;
        }
        self.check_elapsed()
    }

    fn record_cell(&self) -> bool {
        let value = self.cells.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_cells {
            self.abort(ABORT_CELLS);
            return false;
        }
        self.check_elapsed()
    }

    fn record_synthetic_state(&self) -> bool {
        let value = self.synthetic_states.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_synthetic_states {
            self.abort(ABORT_SYNTHETIC_STATES);
            return false;
        }
        self.check_elapsed()
    }

    fn record_stack_effect_visit(&self) -> bool {
        let value = self.stack_effect_visits.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_stack_effect_visits {
            self.abort(ABORT_STACK_EFFECT_VISITS);
            return false;
        }
        if value & 0x3ff == 0 {
            return self.check_elapsed();
        }
        !self.is_aborted()
    }

    /// Create a budget that shares this one's limits and start instant but owns
    /// independent counters. Used by the parallel stack-effect precompute so
    /// each task increments private (uncontended) atomics; counts are folded
    /// back into the parent afterwards. The shared `started_at`/`max_ms`
    /// preserves the global wall-clock cap across the parallel phase.
    fn child(&self) -> UnitInlineBudget {
        use std::sync::atomic::{AtomicU8, AtomicUsize};
        UnitInlineBudget {
            started_at: self.started_at,
            max_ms: self.max_ms,
            max_iterations: self.max_iterations,
            max_cells: self.max_cells,
            max_synthetic_states: self.max_synthetic_states,
            max_stack_effect_visits: self.max_stack_effect_visits,
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        }
    }

    /// Create one independently bounded parser-state analysis. A state that
    /// exceeds this local stack-effect limit is skipped, while sibling states
    /// and the enclosing fixed-point iteration may still complete.
    fn child_with_stack_effect_visit_limit(&self, limit: usize) -> UnitInlineBudget {
        let child = self.child();
        UnitInlineBudget {
            max_stack_effect_visits: child.max_stack_effect_visits.min(limit),
            ..child
        }
    }

    fn fold_in_counts(&self, child: &UnitInlineBudget) {
        use std::sync::atomic::Ordering::Relaxed;
        self.synthetic_states
            .fetch_add(child.synthetic_states(), Relaxed);
        let total = self
            .stack_effect_visits
            .fetch_add(child.stack_effect_visits(), Relaxed)
            + child.stack_effect_visits();
        if total > self.max_stack_effect_visits {
            self.abort(ABORT_STACK_EFFECT_VISITS);
        }
    }

    /// Fold a child budget's consumption back into this one and re-check the
    /// global caps, so total work across the parallel tasks is still bounded.
    fn fold_in(&self, child: &UnitInlineBudget) {
        use std::sync::atomic::Ordering::Relaxed;
        self.fold_in_counts(child);
        let child_code = child.abort_code.load(Relaxed);
        if child_code != ABORT_NONE {
            self.abort(child_code);
        }
    }

    /// Fold one core-merged parser-state analysis. Local stack-effect
    /// exhaustion rejects only that state, but its work still contributes to
    /// the aggregate cap. Other abort causes remain global.
    fn fold_in_isolated_state(&self, child: &UnitInlineBudget) {
        use std::sync::atomic::Ordering::Relaxed;
        self.fold_in_counts(child);
        let child_code = child.abort_code.load(Relaxed);
        if child_code != ABORT_NONE && child_code != ABORT_STACK_EFFECT_VISITS {
            self.abort(child_code);
        }
    }
}

/// Certify that zero-width control closure never needs to discover an unknown
/// parser state below the represented stack prefix.
///
/// Reductions are the only consuming actions whose effect depends on goto
/// predecessors. A control may also expose an unknown predecessor when it pops
/// without pushing a replacement top. In every other case the post-control top
/// is explicit in the action, so exact elimination can use an empty predecessor
/// relation.
fn control_elimination_has_known_tops(
    table: &GLRTable,
    controls: &BTreeSet<TerminalID>,
) -> bool {
    let mut queue = VecDeque::<u32>::new();
    let mut visited = vec![false; table.num_states as usize];
    for (state, row) in table.action.iter().enumerate() {
        if controls.iter().any(|control| row.contains_key(control)) {
            queue.push_back(state as u32);
        }
    }

    while let Some(state) = queue.pop_front() {
        let Some(visited_state) = visited.get_mut(state as usize) else {
            return false;
        };
        if *visited_state {
            continue;
        }
        *visited_state = true;
        let Some(row) = table.action.get(state as usize) else {
            return false;
        };

        for (terminal, action) in row.iter() {
            if !controls.contains(&terminal) {
                if action.reduce_count() != 0 {
                    return false;
                }
                continue;
            }

            match action {
                Action::Shift(target, _) => queue.push_back(*target),
                Action::ReplaceShifts(targets) => queue.extend(targets.iter().copied()),
                Action::StackShifts(shifts) => {
                    for shift in shifts {
                        if let Some(&target) = shift.pushes.last() {
                            queue.push_back(target);
                        } else if shift.pop != 0 {
                            return false;
                        } else {
                            queue.push_back(state);
                        }
                    }
                }
                Action::GuardedStackShifts(shifts) => {
                    for shift in shifts {
                        if let Some(&target) = shift.pushes.last() {
                            queue.push_back(target);
                        } else if shift.pop != 0 {
                            return false;
                        } else {
                            queue.push_back(state);
                        }
                    }
                }
                Action::Split { shift, reduces, .. } => {
                    if !reduces.is_empty() {
                        return false;
                    }
                    if let Some((target, _)) = shift {
                        queue.push_back(*target);
                    }
                }
                Action::Skip => queue.push_back(state),
                Action::Accept => {}
                Action::Reduce(..) => return false,
            }
        }
    }
    true
}

impl GLRTable {
    pub(super) fn canonicalize_stack_shift_predecessors(&mut self) {
        let protected = BitSet::new(self.num_terminals as usize);
        self.canonicalize_stack_shift_predecessors_except(&protected);
    }

    pub(super) fn canonicalize_stack_shift_predecessors_except(&mut self, protected: &BitSet) {
        self.canonicalize_stack_shift_predecessors_with_enabled(
            stack_shift_predecessor_canonicalization_enabled(),
            protected,
        );
    }

    fn canonicalize_stack_shift_predecessors_with_enabled(
        &mut self,
        enabled: bool,
        protected: &BitSet,
    ) {
        if !enabled {
            return;
        }

        for state in 0..self.num_states as usize {
            let terminals: Vec<TerminalID> = self.action[state].keys().collect();
            for terminal in terminals {
                if protected.get(terminal as usize) {
                    continue;
                }
                let Some(Action::StackShifts(shifts)) = self.action[state].get(&terminal).cloned() else {
                    continue;
                };

                let mut shifts = shifts;
                canonicalize_stack_shift_predecessors_by_goto_superset(self, &mut shifts);
                let Some(action) = stack_shift_action(shifts) else {
                    self.action[state].remove(&terminal);
                    if self.advance.len() == self.num_states as usize
                        && let Some(bit) = self.terminal_bit(terminal)
                    {
                        self.advance[state].clear(bit);
                    }
                    continue;
                };
                self.action[state].insert(terminal, action);
            }
        }
    }

    /// Merge states with identical (action, goto) rows.
    /// Iterates until no more merges are possible, since remapping targets
    /// can reveal new equivalences.
    pub(super) fn merge_identical_rows(&mut self) {
        let profile_detail = std::env::var("GLRMASK_PROFILE_GLR_ROW_MERGE_DETAIL")
            .map(|value| value == "1")
            .unwrap_or(false);
        if !env_flag_enabled("GLRMASK_FORCE_ITERATIVE_ROW_MERGE", false) {
            self.merge_identical_rows_by_partition_refinement(profile_detail);
            return;
        }
        let mut iteration = 0usize;
        loop {
            iteration += 1;
            let states_before = self.num_states;
            let scan_started_at = profile_detail.then(std::time::Instant::now);
            // Almost every row fingerprint is unique. Keep the first
            // representative inline and allocate a collision list only when a
            // fingerprint actually contains distinct rows. This preserves the
            // old first-representative search order exactly.
            let mut sig_to_first_rep: FxHashMap<u64, u32> = FxHashMap::default();
            let mut collision_reps: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
            let mut remap: Vec<u32> = (0..self.num_states).collect();
            let mut changed = false;
            let mut fingerprint_collisions = 0usize;
            let mut equality_checks = 0usize;
            let mut matched_rows = 0usize;

            let has_advance_rows = self.advance.len() == self.num_states as usize;
            for state in 0..self.num_states as usize {
                let advance_row = has_advance_rows.then(|| &self.advance[state]);
                let fingerprint = row_fingerprint(&self.action[state], &self.goto[state], advance_row);
                let Some(&first_rep) = sig_to_first_rep.get(&fingerprint) else {
                    sig_to_first_rep.insert(fingerprint, state as u32);
                    continue;
                };
                fingerprint_collisions += 1;
                let row_matches = |rep: u32| {
                    rows_equal(
                        &self.action[state],
                        &self.goto[state],
                        advance_row,
                        &self.action[rep as usize],
                        &self.goto[rep as usize],
                        has_advance_rows.then(|| &self.advance[rep as usize]),
                    )
                };
                equality_checks += 1;
                if row_matches(first_rep) {
                    remap[state] = first_rep;
                    changed = true;
                    matched_rows += 1;
                    continue;
                }

                let reps = collision_reps.entry(fingerprint).or_default();
                let mut matching_rep = None;
                for &rep in reps.iter() {
                    equality_checks += 1;
                    if row_matches(rep) {
                        matching_rep = Some(rep);
                        break;
                    }
                }
                if let Some(rep) = matching_rep {
                    remap[state] = rep;
                    changed = true;
                    matched_rows += 1;
                } else {
                    reps.push(state as u32);
                }
            }

            if !changed {
                if let Some(scan_started_at) = scan_started_at {
                    eprintln!(
                        "[glrmask/profile][row_merge] iteration={} states_before={} states_after={} changed=false fingerprints={} fingerprint_collisions={} equality_checks={} matched_rows={} scan_ms={:.3} remap_ms=0.000",
                        iteration,
                        states_before,
                        states_before,
                        sig_to_first_rep.len(),
                        fingerprint_collisions,
                        equality_checks,
                        matched_rows,
                        scan_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                break;
            }

            let remap_started_at = profile_detail.then(std::time::Instant::now);
            // Build old_to_new: compose remap (merge) with sequential renumbering
            let mut new_id = 0u32;
            let mut rep_to_new: FxHashMap<u32, u32> = FxHashMap::default();
            let mut kept: Vec<u32> = Vec::new();
            for state in 0..self.num_states as usize {
                if remap[state] == state as u32 {
                    rep_to_new.insert(state as u32, new_id);
                    kept.push(state as u32);
                    new_id += 1;
                }
            }
            let mapping: Vec<u32> = (0..self.num_states as usize)
                .map(|s| rep_to_new[&remap[s]])
                .collect();

            // Move surviving rows forward instead of cloning every action,
            // goto, and advance row on every refinement round.  The remap is
            // still applied to exactly the same representatives in the same
            // state order.
            let old_action = std::mem::take(&mut self.action);
            let old_goto = std::mem::take(&mut self.goto);
            let mut old_advance = has_advance_rows.then(|| {
                std::mem::take(&mut self.advance)
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>()
            });
            let mut new_action = Vec::with_capacity(kept.len());
            let mut new_goto = Vec::with_capacity(kept.len());
            let mut new_advance = Vec::with_capacity(kept.len());
            for (state, (mut action_row, mut goto_row)) in
                old_action.into_iter().zip(old_goto).enumerate()
            {
                if remap[state] != state as u32 {
                    continue;
                }
                remap_action_row_targets_in_place(&mut action_row, &mapping);
                remap_goto_row_targets_in_place(&mut goto_row, &mapping);
                new_action.push(action_row);
                new_goto.push(goto_row);
                if let Some(old_advance) = &mut old_advance {
                    new_advance.push(
                        old_advance[state]
                            .take()
                            .expect("kept state has an advance row"),
                    );
                }
            }

            self.action = new_action;
            self.goto = new_goto;
            if has_advance_rows {
                self.advance = new_advance;
            }
            self.forwarded_shifts = self.forwarded_shifts
                .iter()
                .map(|&(state, terminal)| (mapping[state as usize], terminal))
                .collect();
            self.num_states = kept.len() as u32;
            if let (Some(scan_started_at), Some(remap_started_at)) =
                (scan_started_at, remap_started_at)
            {
                eprintln!(
                    "[glrmask/profile][row_merge] iteration={} states_before={} states_after={} changed=true fingerprints={} fingerprint_collisions={} equality_checks={} matched_rows={} scan_ms={:.3} remap_ms={:.3}",
                    iteration,
                    states_before,
                    self.num_states,
                    sig_to_first_rep.len(),
                    fingerprint_collisions,
                    equality_checks,
                    matched_rows,
                    remap_started_at.duration_since(scan_started_at).as_secs_f64() * 1000.0,
                    remap_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
    }

    fn merge_identical_rows_by_partition_refinement(&mut self, profile_detail: bool) {
        let states_before = self.num_states as usize;
        let has_advance_rows = self.advance.len() == states_before;
        // A direct row fingerprint is exact for the identity partition unless
        // an action requires target-remap normalization. Canonical LR rows have
        // only shift/reduce/split/accept actions, so this avoids needless map
        // lookups during the expensive first quotient pass. Post-unit rows can
        // contain stack-effect actions and retain the conservative path.
        let identity_fingerprint_is_exact = self.action.iter().all(|row| {
            row.values().all(|action| {
                !matches!(action, Action::StackShifts(_) | Action::GuardedStackShifts(_))
            })
        });
        let mut state_to_group = (0..self.num_states).collect::<Vec<_>>();
        let mut representatives = (0..states_before).collect::<Vec<_>>();
        let mut iteration = 0usize;

        loop {
            iteration += 1;
            let iteration_started_at = profile_detail.then(std::time::Instant::now);
            let mut fingerprint_to_first_group: FxHashMap<u64, u32> = FxHashMap::default();
            let mut collision_groups: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
            let mut group_to_next_group = vec![0u32; representatives.len()];
            let mut next_representatives = Vec::with_capacity(representatives.len());
            let mut fingerprint_collisions = 0usize;
            let mut equality_checks = 0usize;

            // Fingerprints for the current representatives are independent of
            // each other; compute them in parallel. The serial grouping pass
            // below still visits representatives in order, so the canonical
            // first-occurrence group numbering is preserved exactly.
            let fingerprints: Vec<u64> = if iteration == 1 && identity_fingerprint_is_exact {
                representatives
                    .par_iter()
                    .map(|&state| {
                        row_fingerprint(
                            &self.action[state],
                            &self.goto[state],
                            has_advance_rows.then(|| &self.advance[state]),
                        )
                    })
                    .collect()
            } else {
                representatives
                    .par_iter()
                    .map(|&state| {
                        let advance = has_advance_rows.then(|| &self.advance[state]);
                        remapped_row_fingerprint(
                            &self.action[state],
                            &self.goto[state],
                            advance,
                            &state_to_group,
                        )
                    })
                    .collect()
            };

            for (group, &state) in representatives.iter().enumerate() {
                let advance = has_advance_rows.then(|| &self.advance[state]);
                let fingerprint = fingerprints[group];
                let Some(&first_group) = fingerprint_to_first_group.get(&fingerprint) else {
                    let next_group = next_representatives.len() as u32;
                    fingerprint_to_first_group.insert(fingerprint, next_group);
                    next_representatives.push(state);
                    group_to_next_group[group] = next_group;
                    continue;
                };

                fingerprint_collisions += 1;
                let mut matching_group = None;
                equality_checks += 1;
                if rows_equal_after_remap(
                    &self.action[state],
                    &self.goto[state],
                    advance,
                    &self.action[next_representatives[first_group as usize]],
                    &self.goto[next_representatives[first_group as usize]],
                    has_advance_rows
                        .then(|| &self.advance[next_representatives[first_group as usize]]),
                    &state_to_group,
                ) {
                    matching_group = Some(first_group);
                } else {
                    for &candidate_group in collision_groups
                        .get(&fingerprint)
                        .into_iter()
                        .flatten()
                    {
                        equality_checks += 1;
                        if rows_equal_after_remap(
                            &self.action[state],
                            &self.goto[state],
                            advance,
                            &self.action[next_representatives[candidate_group as usize]],
                            &self.goto[next_representatives[candidate_group as usize]],
                            has_advance_rows.then(|| {
                                &self.advance[next_representatives[candidate_group as usize]]
                            }),
                            &state_to_group,
                        ) {
                            matching_group = Some(candidate_group);
                            break;
                        }
                    }
                }

                if let Some(next_group) = matching_group {
                    group_to_next_group[group] = next_group;
                } else {
                    let next_group = next_representatives.len() as u32;
                    collision_groups
                        .entry(fingerprint)
                        .or_default()
                        .push(next_group);
                    next_representatives.push(state);
                    group_to_next_group[group] = next_group;
                }
            }

            let next_group_count = next_representatives.len();
            if let Some(iteration_started_at) = iteration_started_at {
                eprintln!(
                    "[glrmask/profile][row_merge_partition] iteration={} states_before={} states_after={} fingerprint_collisions={} equality_checks={} refine_ms={:.3}",
                    iteration,
                    representatives.len(),
                    next_group_count,
                    fingerprint_collisions,
                    equality_checks,
                    iteration_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            if next_group_count == representatives.len() {
                break;
            }
            for group in &mut state_to_group {
                *group = group_to_next_group[*group as usize];
            }
            representatives = next_representatives;
        }

        if representatives.len() == states_before {
            return;
        }

        let apply_started_at = profile_detail.then(std::time::Instant::now);
        debug_assert!(representatives.windows(2).all(|pair| pair[0] < pair[1]));
        let mut keep = vec![false; states_before];
        for &state in &representatives {
            keep[state] = true;
        }
        let old_action = std::mem::take(&mut self.action);
        let old_goto = std::mem::take(&mut self.goto);
        let mut old_advance = has_advance_rows.then(|| {
            std::mem::take(&mut self.advance)
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>()
        });
        let mut new_action = Vec::with_capacity(representatives.len());
        let mut new_goto = Vec::with_capacity(representatives.len());
        let mut new_advance = Vec::with_capacity(representatives.len());
        for (state, (mut action_row, mut goto_row)) in
            old_action.into_iter().zip(old_goto).enumerate()
        {
            if !keep[state] {
                continue;
            }
            remap_action_row_targets_in_place(&mut action_row, &state_to_group);
            remap_goto_row_targets_in_place(&mut goto_row, &state_to_group);
            new_action.push(action_row);
            new_goto.push(goto_row);
            if let Some(old_advance) = &mut old_advance {
                new_advance.push(
                    old_advance[state]
                        .take()
                        .expect("kept state has an advance row"),
                );
            }
        }
        self.action = new_action;
        self.goto = new_goto;
        if has_advance_rows {
            self.advance = new_advance;
        }
        self.forwarded_shifts = self
            .forwarded_shifts
            .iter()
            .map(|&(state, terminal)| (state_to_group[state as usize], terminal))
            .collect();
        self.num_states = representatives.len() as u32;
        if let Some(apply_started_at) = apply_started_at {
            eprintln!(
                "[glrmask/profile][row_merge_partition_apply] states_before={} states_after={} apply_ms={:.3}",
                states_before,
                self.num_states,
                apply_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }


    /// Incrementally coarsen an already-minimized table after a local row
    /// mutation. Rows not changed and not reverse-dependent on a newly merged
    /// target cannot become equivalent, so only the dirty frontier is tested.
    ///
    /// Callers must use this only after an exact full quotient. The method is
    /// exact under that precondition and returns false when no quotient occurs.
    pub(super) fn merge_identical_rows_from_dirty(&mut self, dirty: &[u32]) -> bool {
        let nstates = self.num_states as usize;
        if dirty.is_empty() || nstates <= 1 || dirty.iter().any(|&state| state as usize >= nstates) {
            return false;
        }
        let profile_detail = std::env::var("GLRMASK_PROFILE_GLR_ROW_MERGE_DETAIL")
            .map(|value| value == "1")
            .unwrap_or(false);
        let has_advance_rows = self.advance.len() == nstates;
        let setup_started_at = profile_detail.then(std::time::Instant::now);
        let reverse_started_at = profile_detail.then(std::time::Instant::now);

        let mut reverse = vec![SmallVec::<[u32; 2]>::new(); nstates];
        let mut targets = Vec::new();
        for state in 0..nstates {
            targets.clear();
            for action in self.action[state].values() {
                collect_action_targets(action, &mut targets);
            }
            targets.extend(self.goto[state].values().map(|&(target, _)| target));
            targets.sort_unstable();
            targets.dedup();
            for target in &targets {
                reverse[*target as usize].push(state as u32);
            }
        }

        let reverse_ms = reverse_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let init_started_at = profile_detail.then(std::time::Instant::now);
        let mut parent = (0..nstates as u32).collect::<Vec<_>>();
        // The exact quotient starts with singleton classes. Keep those
        // inline: only the handful of classes that actually merge need spill
        // storage.
        let mut members: Vec<SmallVec<[u32; 1]>> = (0..nstates as u32)
            .map(|state| smallvec![state])
            .collect();
        let mut target_group = parent.clone();
        let mut active = dirty.iter().copied().collect::<BTreeSet<_>>();
        let mut merged_any = false;
        let mut iteration = 0usize;

        fn find(parent: &mut [u32], mut state: u32) -> u32 {
            while parent[state as usize] != state {
                let next = parent[state as usize];
                let root = parent[next as usize];
                parent[state as usize] = root;
                state = next;
            }
            state
        }

        // Post-unit rows can contain stack-effect actions whose canonical
        // identity form is produced by target remapping. Unlike ordinary
        // shifts, their direct hash is not a safe candidate fingerprint even
        // under the identity map, so retain the remapped representation here.
        let mut fingerprint_by_state = (0..nstates)
            .into_par_iter()
            .map(|state| {
                remapped_row_fingerprint(
                    &self.action[state],
                    &self.goto[state],
                    has_advance_rows.then(|| &self.advance[state]),
                    &target_group,
                )
            })
            .collect::<Vec<_>>();
        let mut fingerprints = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        for (state, &fingerprint) in fingerprint_by_state.iter().enumerate() {
            fingerprints.entry(fingerprint).or_default().push(state as u32);
        }
        let mut representative_count = nstates;
        let mut first_iteration = true;
        if let Some(setup_started_at) = setup_started_at {
            let total_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[glrmask/profile][row_merge_incremental_setup] states={} dirty={} reverse_ms={:.3} init_ms={:.3} total_ms={:.3}",
                nstates,
                dirty.len(),
                reverse_ms,
                total_ms - reverse_ms,
                total_ms,
            );
        }

        while !active.is_empty() {
            iteration += 1;
            let iteration_started_at = profile_detail.then(std::time::Instant::now);
            let active_now = std::mem::take(&mut active);
            let active_count = active_now.len();
            let mut active_representatives = BTreeSet::new();
            for state in active_now {
                active_representatives.insert(find(&mut parent, state));
            }

            if !first_iteration {
                for &state in &active_representatives {
                    if parent[state as usize] != state {
                        continue;
                    }
                    let old_fingerprint = fingerprint_by_state[state as usize];
                    let remove_bucket = {
                        let bucket = fingerprints
                            .get_mut(&old_fingerprint)
                            .expect("representative has a fingerprint bucket");
                        bucket.retain(|entry| *entry != state);
                        bucket.is_empty()
                    };
                    if remove_bucket {
                        fingerprints.remove(&old_fingerprint);
                    }
                    let fingerprint = remapped_row_fingerprint(
                        &self.action[state as usize],
                        &self.goto[state as usize],
                        has_advance_rows.then(|| &self.advance[state as usize]),
                        &target_group,
                    );
                    fingerprint_by_state[state as usize] = fingerprint;
                    fingerprints.entry(fingerprint).or_default().push(state);
                }
            }
            first_iteration = false;

            let mut pairs = Vec::new();
            let mut comparisons = 0usize;
            for state in active_representatives {
                if parent[state as usize] != state {
                    continue;
                }
                let fingerprint = fingerprint_by_state[state as usize];
                let Some(candidates) = fingerprints.get(&fingerprint) else {
                    continue;
                };
                for &candidate in candidates {
                    let candidate = find(&mut parent, candidate);
                    if candidate == state {
                        continue;
                    }
                    comparisons += 1;
                    if rows_equal_after_remap(
                        &self.action[state as usize],
                        &self.goto[state as usize],
                        has_advance_rows.then(|| &self.advance[state as usize]),
                        &self.action[candidate as usize],
                        &self.goto[candidate as usize],
                        has_advance_rows.then(|| &self.advance[candidate as usize]),
                        &target_group,
                    ) {
                        pairs.push((state, candidate));
                    }
                }
            }

            let mut next_active = BTreeSet::new();
            let mut merged_groups = 0usize;
            for (left, right) in pairs {
                let left = find(&mut parent, left);
                let right = find(&mut parent, right);
                if left == right {
                    continue;
                }
                let (winner, loser) = if left < right { (left, right) } else { (right, left) };
                let loser_fingerprint = fingerprint_by_state[loser as usize];
                let remove_bucket = {
                    let bucket = fingerprints
                        .get_mut(&loser_fingerprint)
                        .expect("losing representative has a fingerprint bucket");
                    bucket.retain(|entry| *entry != loser);
                    bucket.is_empty()
                };
                if remove_bucket {
                    fingerprints.remove(&loser_fingerprint);
                }
                let loser_members = std::mem::take(&mut members[loser as usize]);
                for member in loser_members {
                    parent[member as usize] = winner;
                    target_group[member as usize] = winner;
                    for &source in &reverse[member as usize] {
                        next_active.insert(find(&mut parent, source));
                    }
                    members[winner as usize].push(member);
                }
                representative_count -= 1;
                merged_groups += 1;
                merged_any = true;
            }

            if let Some(iteration_started_at) = iteration_started_at {
                eprintln!(
                    "[glrmask/profile][row_merge_incremental] iteration={} representatives={} active={} comparisons={} merged_groups={} elapsed_ms={:.3}",
                    iteration,
                    representative_count + merged_groups,
                    active_count,
                    comparisons,
                    merged_groups,
                    iteration_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            if merged_groups == 0 {
                break;
            }
            active = next_active;
        }

        if !merged_any {
            return false;
        }

        let apply_started_at = profile_detail.then(std::time::Instant::now);
        let representatives = (0..nstates as u32)
            .filter(|&state| parent[state as usize] == state)
            .collect::<Vec<_>>();
        let mut final_mapping = vec![0u32; nstates];
        for (new_state, &representative) in representatives.iter().enumerate() {
            for &member in &members[representative as usize] {
                final_mapping[member as usize] = new_state as u32;
            }
        }

        let old_action = std::mem::take(&mut self.action);
        let old_goto = std::mem::take(&mut self.goto);
        let mut old_advance = has_advance_rows.then(|| {
            std::mem::take(&mut self.advance)
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>()
        });
        let mut new_action = Vec::with_capacity(representatives.len());
        let mut new_goto = Vec::with_capacity(representatives.len());
        let mut new_advance = Vec::with_capacity(representatives.len());
        for (state, (mut action_row, mut goto_row)) in old_action.into_iter().zip(old_goto).enumerate() {
            if parent[state] != state as u32 {
                continue;
            }
            remap_action_row_targets_in_place(&mut action_row, &final_mapping);
            remap_goto_row_targets_in_place(&mut goto_row, &final_mapping);
            new_action.push(action_row);
            new_goto.push(goto_row);
            if let Some(old_advance) = &mut old_advance {
                new_advance.push(old_advance[state].take().expect("kept state has advance row"));
            }
        }
        self.action = new_action;
        self.goto = new_goto;
        if has_advance_rows {
            self.advance = new_advance;
        }
        self.forwarded_shifts = self
            .forwarded_shifts
            .iter()
            .map(|&(state, terminal)| (final_mapping[state as usize], terminal))
            .collect();
        self.num_states = representatives.len() as u32;
        if let Some(apply_started_at) = apply_started_at {
            eprintln!(
                "[glrmask/profile][row_merge_incremental_apply] states_before={} states_after={} apply_ms={:.3}",
                nstates,
                self.num_states,
                apply_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        true
    }


    pub(super) fn prune_unreachable_states(&mut self) {
        let _ = self.prune_unreachable_states_impl(false);
    }

    /// Remove unreachable states and return the old-state -> new-state map.
    /// Unreachable states map to `u32::MAX`.
    ///
    /// Composition-time structural sharing needs this map to transport the
    /// already-compiled parser-DWA label relation through a table compaction.
    pub fn prune_unreachable_states_with_mapping(&mut self) -> Vec<u32> {
        self.prune_unreachable_states_impl(true)
            .unwrap_or_default()
    }

    fn prune_unreachable_states_impl(&mut self, return_mapping: bool) -> Option<Vec<u32>> {
        if self.num_states == 0 {
            return return_mapping.then(Vec::new);
        }

        let mut reachable = vec![false; self.num_states as usize];
        let mut stack = vec![0u32];
        reachable[0] = true;

        while let Some(state) = stack.pop() {
            for action in self.action[state as usize].values() {
                push_action_targets(action, &mut reachable, &mut stack);
            }
            for &(target, _) in self.goto[state as usize].values() {
                push_reachable_state(target, &mut reachable, &mut stack);
            }
        }

        if reachable.iter().all(|&is_reachable| is_reachable) {
            return return_mapping.then(|| (0..self.num_states).collect());
        }

        let mut mapping = vec![u32::MAX; self.num_states as usize];
        let mut kept = Vec::new();
        for (state, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                mapping[state] = kept.len() as u32;
                kept.push(state as u32);
            }
        }

        self.action = kept
            .iter()
            .map(|&state| {
                self.action[state as usize]
                    .iter()
                    .map(|(terminal, action)| (terminal, remap_action_targets(action, &mapping)))
                    .collect()
            })
            .collect();
        self.goto = kept
            .iter()
            .map(|&state| {
                self.goto[state as usize]
                    .iter()
                    .map(|(&nonterminal, &(target, replace))| {
                        (nonterminal, (mapping[target as usize], replace))
                    })
                    .collect()
            })
            .collect();
        if self.advance.len() == reachable.len() {
            self.advance = kept
                .iter()
                .map(|&state| self.advance[state as usize].clone())
                .collect();
        }
        self.forwarded_shifts = self.forwarded_shifts
            .iter()
            .filter_map(|&(state, terminal)| {
                reachable[state as usize].then_some((mapping[state as usize], terminal))
            })
            .collect();
        self.direct_regular_wide_frontiers = self
            .direct_regular_wide_frontiers
            .iter()
            .filter_map(|descriptor| {
                if !reachable[descriptor.source_state as usize]
                    || descriptor
                        .target_states
                        .iter()
                        .any(|&state| !reachable[state as usize])
                {
                    return None;
                }
                Some(super::DirectRegularWideFrontierDescriptor {
                    source_state: mapping[descriptor.source_state as usize],
                    terminal: descriptor.terminal,
                    target_states: descriptor
                        .target_states
                        .iter()
                        .map(|&state| mapping[state as usize])
                        .collect(),
                })
            })
            .collect();
        self.num_states = kept.len() as u32;
        return_mapping.then_some(mapping)
    }

    /// Replace several context-distinguishable LR states by one physical LR
    /// state while retaining the old source identity as a guard on the state
    /// immediately below it on the parser stack.
    ///
    /// This is an exact quotient, not an approximation. `groups` are merely
    /// candidates. A group is accepted only when every member has a non-empty,
    /// pairwise-disjoint set of possible immediate predecessors, those
    /// predecessor states remain unmerged, and predecessor provenance is known
    /// complete (no complex stack-effect edge can enter the member). Each
    /// member action is compiled to an equivalent macro stack effect with a
    /// depth-one predecessor guard. Consequently, on every reachable stack,
    /// exactly the original member's behavior remains enabled after the top
    /// state is replaced by the shared state.
    ///
    /// Returns an old-state -> final-state map covering the state domain that
    /// existed on entry. A no-op returns the identity map.
    pub fn share_context_distinguishable_states_exact(
        &mut self,
        groups: &[Vec<u32>],
    ) -> Vec<u32> {
        let debug_context_share =
            std::env::var_os("GLRMASK_DEBUG_CONTEXT_STATE_SHARING").is_some();
        let original_num_states = self.num_states;
        if original_num_states <= 1
            || groups.is_empty()
            || self.admission_policy != super::AdmissionPolicy::ExactSimulation
        {
            return (0..original_num_states).collect();
        }

        let original_len = original_num_states as usize;
        let constituent_sets: Vec<StateSubset> = (0..original_num_states)
            .map(|state| smallvec![state])
            .collect();
        let predecessors = build_runtime_state_predecessors(
            self,
            original_num_states,
            &constituent_sets,
        );

        // The predecessor recurrence above is exact for ordinary Shift/Split
        // shift and goto edges. If a state can be entered by a precompiled
        // arbitrary stack effect, its immediate predecessor may depend on a
        // deeper stack value that recurrence intentionally does not model.
        // Mark such targets unsafe, then propagate unsafety through replace
        // edges (which inherit their source predecessor). Non-replace edges
        // establish their source state as the new immediate predecessor and
        // therefore stop the propagation.
        let mut unsafe_predecessor = vec![false; original_len];
        for row in &self.action[..original_len] {
            for action in row.values() {
                match action {
                    Action::StackShifts(shifts) => {
                        for shift in shifts {
                            for &target in &shift.pushes {
                                if let Some(slot) = unsafe_predecessor.get_mut(target as usize) {
                                    *slot = true;
                                }
                            }
                        }
                    }
                    Action::GuardedStackShifts(shifts) => {
                        for shift in shifts {
                            for &target in &shift.pushes {
                                if let Some(slot) = unsafe_predecessor.get_mut(target as usize) {
                                    *slot = true;
                                }
                            }
                        }
                    }
                    Action::ReplaceShifts(targets) => {
                        for &target in targets.iter() {
                            if let Some(slot) = unsafe_predecessor.get_mut(target as usize) {
                                *slot = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for source in 0..original_len {
                if !unsafe_predecessor[source] {
                    continue;
                }
                for action in self.action[source].values() {
                    let target = match action {
                        Action::Shift(target, true) => Some(*target),
                        Action::Split {
                            shift: Some((target, true)),
                            ..
                        } => Some(*target),
                        _ => None,
                    };
                    if let Some(target) = target
                        && let Some(slot) = unsafe_predecessor.get_mut(target as usize)
                        && !*slot
                    {
                        *slot = true;
                        changed = true;
                    }
                }
                for &(target, replace) in self.goto[source].values() {
                    if replace
                        && let Some(slot) = unsafe_predecessor.get_mut(target as usize)
                        && !*slot
                    {
                        *slot = true;
                        changed = true;
                    }
                }
            }
        }

        let mut protected = vec![false; original_len];
        protected[0] = true;
        // Existing guards observe concrete LR-state identity. Keep every state
        // they can inspect distinct; otherwise remapping a guard set could
        // broaden its predicate.
        for row in &self.action[..original_len] {
            for action in row.values() {
                if let Action::GuardedStackShifts(shifts) = action {
                    for shift in shifts {
                        for guard in &shift.guards {
                            for &state in &guard.states {
                                if let Some(slot) = protected.get_mut(state as usize) {
                                    *slot = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        // These descriptors are runtime observations with their own state
        // semantics. Preserve them unchanged rather than extending this first
        // exact quotient with another proof obligation.
        for descriptor in &self.direct_regular_wide_frontiers {
            if let Some(slot) = protected.get_mut(descriptor.source_state as usize) {
                *slot = true;
            }
            for &state in &descriptor.target_states {
                if let Some(slot) = protected.get_mut(state as usize) {
                    *slot = true;
                }
            }
        }
        // Reduction macro compilation may need every LR state having a goto
        // for a reduced nonterminal after the reconstructed suffix is popped
        // completely. Build that exact inverse index once; repeatedly scanning
        // all goto rows inside each speculative reduction is O(groups × rows)
        // and dominated large subgrammar compositions.
        let mut goto_sources_by_nonterminal =
            vec![Vec::<u32>::new(); self.nonterminal_display_names.len()];
        for (state, row) in self.goto[..original_len].iter().enumerate() {
            for (&nonterminal, _) in row.iter() {
                if nonterminal as usize >= goto_sources_by_nonterminal.len() {
                    goto_sources_by_nonterminal
                        .resize_with(nonterminal as usize + 1, Vec::new);
                }
                goto_sources_by_nonterminal[nonterminal as usize].push(state as u32);
            }
        }

        // Cheap first pass: any state used as immediate-predecessor provenance
        // by a structurally viable candidate cannot itself be shared. Compute
        // that necessary protection before compiling any suffix-effect macro
        // rows. This may conservatively protect a predecessor for a candidate
        // whose later macro compilation would fail, but can only lose an
        // optimization; it cannot make an invalid merge admissible. In the
        // common duplicated-subgrammar case this removes a large fraction of
        // speculative interior/caller groups before the expensive exact row
        // compilation.
        let mut early_provenance_protected = vec![false; original_len];
        for input_group in groups {
            let mut group = input_group
                .iter()
                .copied()
                .filter(|&state| state < original_num_states)
                .collect::<Vec<_>>();
            group.sort_unstable();
            group.dedup();
            if group.len() < 2
                || group.iter().any(|&state| {
                    protected[state as usize]
                        || unsafe_predecessor[state as usize]
                        || predecessors[state as usize].is_empty()
                })
            {
                continue;
            }
            let Some(plan) = context_share_provenance_plan(&group, &predecessors) else {
                continue;
            };
            let first_goto = &self.goto[group[0] as usize];
            let goto_ok = group[1..].iter().all(|&state| {
                let row = &self.goto[state as usize];
                row.len() == first_goto.len()
                    && first_goto
                        .iter()
                        .all(|(&nonterminal, &target)| row.get(&nonterminal) == Some(&target))
            });
            if !goto_ok {
                continue;
            }
            // Accept cannot be represented by a predecessor-guarded macro, so
            // reject such groups before allowing them to reserve provenance.
            if group.iter().any(|&state| {
                self.action[state as usize]
                    .values()
                    .any(|action| matches!(action, Action::Accept | Action::Split { accept: true, .. }))
            }) {
                continue;
            }
            for member_predecessors in &plan.predecessors_by_member {
                for &predecessor in member_predecessors {
                    early_provenance_protected[predecessor as usize] = true;
                }
            }
        }
        // First establish which proposed classes are individually viable.
        // Only *those* classes are allowed to reserve their predecessor IDs as
        // provenance tags. Reserving predecessors for every speculative class
        // is overly conservative: an unmergeable return/caller class can use a
        // perfectly mergeable interior state as its predecessor and would then
        // freeze the useful state for no semantic reason.
        let mut viable = Vec::<ContextStateSharePlan>::new();
        for input_group in groups {
            let mut group = input_group
                .iter()
                .copied()
                .filter(|&state| state < original_num_states)
                .collect::<Vec<_>>();
            group.sort_unstable();
            group.dedup();
            let blocked_member = group.iter().copied().find(|&state| {
                protected[state as usize]
                    || unsafe_predecessor[state as usize]
                    || predecessors[state as usize].is_empty()
            });
            if group.len() < 2 || blocked_member.is_some() {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=member group={group:?} blocked={blocked_member:?} protected={:?} unsafe={:?} predecessors={:?}",
                        group
                            .iter()
                            .map(|&state| protected[state as usize])
                            .collect::<Vec<_>>(),
                        group
                            .iter()
                            .map(|&state| unsafe_predecessor[state as usize])
                            .collect::<Vec<_>>(),
                        group
                            .iter()
                            .map(|&state| predecessors[state as usize].to_vec())
                            .collect::<Vec<_>>(),
                    );
                }
                continue;
            }
            let Some(mut plan) = context_share_provenance_plan(&group, &predecessors) else {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=overlapping_predecessors group={group:?}",
                    );
                }
                continue;
            };

            // Gotos are not terminal actions and therefore cannot carry the
            // provenance guards. Require the concrete goto rows to be exactly
            // equal; a disjoint goto key would otherwise become spuriously
            // available from every source represented by the shared state.
            let first_goto = &self.goto[group[0] as usize];
            let goto_ok = group[1..].iter().all(|&state| {
                let row = &self.goto[state as usize];
                row.len() == first_goto.len()
                    && first_goto
                        .iter()
                        .all(|(&nonterminal, &target)| row.get(&nonterminal) == Some(&target))
            });
            if !goto_ok {
                if debug_context_share {
                    eprintln!("[glrmask/debug][context_state_share] reject=goto group={group:?}");
                }
                continue;
            }

            if group
                .iter()
                .any(|&state| early_provenance_protected[state as usize])
            {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=early_provenance group={group:?}",
                    );
                }
                continue;
            }

            // Build the exact row against the unmodified table. Each initial
            // frame reconstructs the concrete member only when the immediate
            // predecessor belongs to that member's certified, disjoint
            // predecessor set. The existing suffix-effect compiler then
            // executes reductions to the eventual consuming effect.
            let mut terminals = BTreeSet::<TerminalID>::new();
            for &state in &group {
                terminals.extend(self.action[state as usize].keys());
            }
            let mut action_row = ActionRow::default();
            let mut row_ok = true;
            for terminal in terminals {
                let mut effects = Vec::<GuardedStackShift>::new();
                let mut accepts = 0usize;
                for (member_index, &state) in group.iter().enumerate() {
                    let Some(action) = self.action[state as usize].get(&terminal) else {
                        continue;
                    };
                    let frame = context_share_frame_for_member(
                        state,
                        &plan.predecessors_by_member[member_index],
                    );
                    if collect_context_share_effects_from_frame(
                        self,
                        terminal,
                        state,
                        action,
                        frame,
                        &goto_sources_by_nonterminal,
                        &mut effects,
                        &mut accepts,
                    )
                    .is_err()
                    {
                        row_ok = false;
                        break;
                    }
                }
                // Accept has no guard-bearing representation. Keep any group
                // containing an accepting action separate.
                if !row_ok || accepts != 0 {
                    row_ok = false;
                    break;
                }
                normalize_guarded_effects_for_suffix_quotient(&mut effects);
                for effect in &effects {
                    for guard in &effect.guards {
                        plan.generated_guard_states.extend_from_slice(&guard.states);
                    }
                }
                if !effects.is_empty() {
                    action_row.insert(terminal, Action::GuardedStackShifts(effects));
                }
            }
            if !row_ok {
                if debug_context_share {
                    eprintln!("[glrmask/debug][context_state_share] reject=row group={group:?}");
                }
                continue;
            }

            if debug_context_share {
                eprintln!(
                    "[glrmask/debug][context_state_share] viable group={group:?} predecessors={:?}",
                    plan.predecessors_by_member,
                );
            }
            plan.generated_guard_states.sort_unstable();
            plan.generated_guard_states.dedup();

            // State 0 is the table's distinguished subgrammar entry state. A
            // later composition overlays that row onto each caller instead of
            // copying state 0 as an ordinary child state. Therefore an
            // internal macro action that observes state 0 in a stack guard is
            // exact for this standalone table but is not representable by the
            // current subgrammar-splice ABI without expanding the observation
            // to caller states (and preserving caller correlation). Keep the
            // structural quotient closed under further composition by simply
            // rejecting such an optimization. This can only leave states
            // unmerged; it cannot change the accepted language.
            if plan.generated_guard_states.binary_search(&0).is_ok() {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=compositional_start_guard group={group:?}",
                    );
                }
                continue;
            }

            plan.action_row = action_row;
            viable.push(plan);
        }

        // Every predecessor state used by a provenance guard is a runtime tag.
        // Give that role priority over sharing the tag itself. This also makes
        // all guard predicates stable under the later old->shared state map.
        let mut provenance_protected = vec![false; original_len];
        for plan in &viable {
            for member_predecessors in &plan.predecessors_by_member {
                for &guard_state in member_predecessors {
                    if let Some(slot) = provenance_protected.get_mut(guard_state as usize) {
                        *slot = true;
                    }
                }
            }
            for &guard_state in &plan.generated_guard_states {
                if let Some(slot) = provenance_protected.get_mut(guard_state as usize) {
                    *slot = true;
                }
            }
        }
        let mut claimed = vec![false; original_len];
        let mut accepted = Vec::<ContextStateSharePlan>::new();
        for plan in viable {
            let group = &plan.group;
            if group.iter().any(|&state| {
                provenance_protected[state as usize] || claimed[state as usize]
            }) {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=provenance group={group:?} protected={:?}",
                        group
                            .iter()
                            .map(|&state| provenance_protected[state as usize])
                            .collect::<Vec<_>>(),
                    );
                }
                continue;
            }
            for &state in group {
                claimed[state as usize] = true;
            }
            accepted.push(plan);
        }

        if accepted.is_empty() {
            return (0..original_num_states).collect();
        }

        // The exact macro rows were already compiled during viability. Once
        // the accepted classes are fixed, transport their embedded state IDs
        // through the final sharing map rather than recomputing the same
        // reduction/suffix closure a second time.
        let mut share_map = (0..original_num_states).collect::<Vec<_>>();
        let shared_base = original_num_states;
        for (index, plan) in accepted.iter().enumerate() {
            let shared = shared_base + index as u32;
            for &state in &plan.group {
                share_map[state as usize] = shared;
            }
        }
        share_map.extend((shared_base..shared_base + accepted.len() as u32).collect::<Vec<_>>());

        let had_advance_rows = self.advance.len() == original_len;
        let mut shared_rows = Vec::<(ActionRow, GotoRow, Option<BitSet>)>::new();
        for plan in &accepted {
            let group = &plan.group;
            let mut action_row = plan.action_row.clone();
            remap_action_row_targets_in_place(&mut action_row, &share_map);
            action_row.for_each_value_mut(|action| {
                if let Action::GuardedStackShifts(effects) = action {
                    normalize_guarded_effects_for_suffix_quotient(effects);
                }
            });

            let goto_row = self.goto[group[0] as usize]
                .iter()
                .map(|(&nonterminal, &(target, replace))| {
                    (nonterminal, (share_map[target as usize], replace))
                })
                .collect();
            let advance = had_advance_rows.then(|| union_advance_rows(self, &group.iter().copied().collect()));
            shared_rows.push((action_row, goto_row, advance));
        }

        // Redirect every incoming table edge before pruning the now-unreachable
        // concrete member states.
        for row in &mut self.action[..original_len] {
            remap_action_row_targets_in_place(row, &share_map);
        }
        for row in &mut self.goto[..original_len] {
            remap_goto_row_targets_in_place(row, &share_map);
        }
        for (action, goto, advance) in shared_rows {
            self.action.push(action);
            self.goto.push(goto);
            if let Some(advance) = advance {
                self.advance.push(advance);
            }
            self.num_states += 1;
        }
        self.forwarded_shifts = self
            .forwarded_shifts
            .iter()
            .filter_map(|&(state, terminal)| {
                (share_map[state as usize] == state).then_some((state, terminal))
            })
            .collect();

        let prune_map = self.prune_unreachable_states_with_mapping();
        let mut final_map = vec![u32::MAX; original_len];
        for old in 0..original_len {
            let shared = share_map[old] as usize;
            final_map[old] = prune_map.get(shared).copied().unwrap_or(u32::MAX);
        }
        debug_assert!(final_map.iter().all(|&state| state != u32::MAX));
        self.rebuild_guarded_shift_index();
        self.compress_default_action_rows();
        final_map
    }

    /// Partially evaluate linker-internal zero-width controls into ordinary
    /// consuming parser actions.
    ///
    /// For every source LR state and every real terminal reachable after zero
    /// or more control actions, this compiles the exact guarded stack relation
    /// `control* ; terminal` into one `StackShifts`/`GuardedStackShifts` cell.
    /// The next consuming cell performs its own leading closure, so no runtime
    /// control closure is required between tokens. EOF is represented by row
    /// presence, matching `stacks_finished` semantics.
    pub fn eliminate_control_terminals_exact(
        &mut self,
    ) -> Result<ControlEliminationReport, String> {
        if self.control_terminals.is_empty() {
            return Ok(ControlEliminationReport {
                states: self.num_states as usize,
                controls: 0,
                compiled_cells: 0,
                closure_frames: 0,
                stack_effect_visits: 0,
                changed_terminals: Vec::new(),
                elapsed_ms: 0.0,
            });
        }
        // Proof invariant for the prepared static linker: the new signed-transfer
        // boundary compiler must never invoke control elimination. Count every
        // real elimination run (no-op calls on control-free tables do not count)
        // so link tests can assert the counter does not move across the link.
        CONTROL_ELIMINATION_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let started_at = std::time::Instant::now();
        let source = self.clone();
        let controls = source.control_terminals.clone();
        let predecessor_started_at = std::time::Instant::now();
        let predecessor_free = control_elimination_has_known_tops(&source, &controls);
        let predecessors = if predecessor_free {
            Vec::new()
        } else if std::env::var_os("GLRMASK_DISABLE_CONTROL_PREDECESSOR_DEMAND").is_none() {
            let origins = (0..source.num_states)
                .filter(|&state| controls.iter().any(|control| source.action[state as usize].contains_key(control)))
                .collect::<Vec<_>>();
            build_control_elimination_predecessors_demand(&source, &origins)?
        } else {
            build_control_elimination_predecessors(&source)?
        };
        let predecessor_ms = predecessor_started_at.elapsed().as_secs_f64() * 1000.0;
        let budget = UnitInlineBudget::exact_control_elimination();
        let mut compiled_rows = Vec::with_capacity(source.num_states as usize);
        let mut compiled_cells = 0usize;
        let mut closure_frames = 0usize;
        let mut copied_row_count = 0usize;
        let mut symbolic_row_count = 0usize;
        let mut changed_terminals = BTreeSet::<TerminalID>::new();
        let mut slowest_rows = Vec::<(u64, u32, usize, usize)>::new();
        let profile_control = std::env::var_os("GLRMASK_PROFILE_CONTROL_ELIMINATION").is_some();

        for origin_state in 0..source.num_states {
            let row_started_at = profile_control.then(std::time::Instant::now);
            let visits_before = budget.stack_effect_visits();
            let source_row = &source.action[origin_state as usize];
            if !controls
                .iter()
                .any(|control| source_row.contains_key(control))
            {
                // A leading zero-width control closure can only begin with an
                // action in the current top-state row. With no control cell,
                // the closure is exactly identity and every ordinary action is
                // already the required compiled result. Copy it verbatim;
                // symbolically recompiling every child-table row dominates
                // realistic multi-schema composition for no semantic gain.
                compiled_cells = compiled_cells.saturating_add(source_row.len());
                compiled_rows.push(source_row.clone());
                closure_frames = closure_frames.saturating_add(1);
                copied_row_count += 1;
                continue;
            }
            symbolic_row_count += 1;

            let mut depth_cache = FxHashMap::default();
            let closure = control_effect_closure(
                &source,
                &predecessors,
                origin_state,
                &controls,
                StackEffectFrame {
                    pop: 0,
                    pushes: Vec::new(),
                    guards: Vec::new(),
                },
                &mut depth_cache,
                &budget,
            )?;
            closure_frames = closure_frames.saturating_add(closure.len());

            let mut candidates = BTreeSet::<TerminalID>::new();
            for frame in &closure {
                for (top_state, _) in frame_top_branches(
                    &predecessors,
                    origin_state,
                    frame,
                    &mut depth_cache,
                    &budget,
                )? {
                    candidates.extend(
                        source.action[top_state as usize]
                            .keys()
                            .filter(|terminal| !controls.contains(terminal)),
                    );
                }
            }

            let mut row = ActionRow::default();
            for terminal in candidates {
                if terminal == EOF {
                    if matches!(source.action(origin_state, EOF), Some(Action::Accept)) {
                        row.insert(EOF, Action::Accept);
                        compiled_cells += 1;
                        continue;
                    }
                    let mut effects = Vec::<GuardedStackShift>::new();
                    for frame in &closure {
                        for (top_state, top_frame) in frame_top_branches(
                            &predecessors,
                            origin_state,
                            frame,
                            &mut depth_cache,
                            &budget,
                        )? {
                            let Some(action) = source.action[top_state as usize].get(&EOF) else {
                                continue;
                            };
                            effects.extend(eof_accept_effects_for_action(
                                &source,
                                &predecessors,
                                origin_state,
                                top_state,
                                action,
                                top_frame,
                                &mut depth_cache,
                                &mut FxHashSet::default(),
                                &budget,
                            )?);
                        }
                    }
                    if let Some(action) = control_stack_effect_action(&source, effects) {
                        row.insert(EOF, action);
                        compiled_cells += 1;
                    }
                    continue;
                }

                let mut effects = Vec::<GuardedStackShift>::new();
                for frame in &closure {
                    for (top_state, top_frame) in frame_top_branches(
                        &predecessors,
                        origin_state,
                        frame,
                        &mut depth_cache,
                        &budget,
                    )? {
                        let Some(action) = source.action[top_state as usize].get(&terminal) else {
                            continue;
                        };
                        let Some(result) = stack_effects_for_action(
                            &source,
                            &predecessors,
                            origin_state,
                            terminal,
                            top_state,
                            action,
                            top_frame,
                            &mut depth_cache,
                            &mut FxHashSet::default(),
                            &mut Vec::new(),
                            &budget,
                        ) else {
                            return Err(control_elimination_failure(
                                &budget,
                                origin_state,
                                terminal,
                                "real-terminal stack-effect traversal failed",
                            ));
                        };
                        effects.extend(result.effects);
                    }
                }

                if let Some(action) = control_stack_effect_action(&source, effects) {
                    row.insert(terminal, action);
                    compiled_cells += 1;
                }
            }
            for terminal in source_row.keys().chain(row.keys()) {
                if controls.contains(&terminal) {
                    continue;
                }
                if source_row.get(&terminal) != row.get(&terminal) {
                    changed_terminals.insert(terminal);
                }
            }
            compiled_rows.push(row);
            if let Some(row_started_at) = row_started_at {
                let elapsed_ns = row_started_at.elapsed().as_nanos() as u64;
                let visits = budget.stack_effect_visits().saturating_sub(visits_before);
                slowest_rows.push((elapsed_ns, origin_state, closure.len(), visits));
            }
        }

        if budget.is_aborted() {
            return Err(control_elimination_failure(
                &budget,
                0,
                EOF,
                "control elimination exceeded its compile budget",
            ));
        }

        self.action = compiled_rows;
        self.control_terminals.clear();
        self.forwarded_shifts.clear();
        self.admission_policy = AdmissionPolicy::RowPresenceExact;
        self.rebuild_advance_rows_from_actions();
        self.rebuild_unconditional_advance_rows();
        self.rebuild_guarded_shift_index();

        if profile_control {
            slowest_rows.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
            slowest_rows.truncate(20);
            eprintln!(
                "[glrmask/profile][control_elimination] states={} controls={} copied_rows={} symbolic_rows={} compiled_cells={} closure_frames={} stack_effect_visits={} predecessor_free={} predecessor_ms={predecessor_ms:.3} total_ms={:.3} slowest_rows={:?}",
                source.num_states,
                controls.len(),
                copied_row_count,
                symbolic_row_count,
                compiled_cells,
                closure_frames,
                budget.stack_effect_visits(),
                predecessor_free,
                started_at.elapsed().as_secs_f64() * 1000.0,
                slowest_rows,
            );
        }

        Ok(ControlEliminationReport {
            states: self.num_states as usize,
            controls: controls.len(),
            compiled_cells,
            closure_frames,
            stack_effect_visits: budget.stack_effect_visits(),
            changed_terminals: changed_terminals.into_iter().collect(),
            elapsed_ms: started_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Collapse unit reductions by inlining their destination actions.
    ///
    /// When inlining produces multiple shift destinations, create a synthetic
    /// merged state whose row is the union of its constituents' rows. This
    /// keeps the parser representation unchanged: every action cell still has
    /// at most one shift slot, but that shift target may be a merged state.
    #[cfg(test)]
    pub(super) fn collapse_sr_unit_reductions_for_correctness_oracle(
        &mut self,
    ) -> UnitReductionInliningReport {
        use std::sync::atomic::{AtomicU8, AtomicUsize};

        let budget = UnitInlineBudget {
            started_at: std::time::Instant::now(),
            max_ms: u128::MAX,
            max_iterations: usize::MAX,
            max_cells: usize::MAX,
            max_synthetic_states: usize::MAX,
            max_stack_effect_visits: usize::MAX,
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        };
        let mut undo = UnitInlineUndo::new(self);
        let protected = BitSet::new(self.num_terminals as usize);
        self.collapse_sr_unit_reductions_with_compatible_gotos_inner(
            &budget,
            &mut undo,
            &protected,
        );
        let mut report = budget.report();
        if report.aborted {
            undo.rollback(self);
        } else {
            report.changed_original_states = undo.changed_original_states();
        }
        report
    }

    pub(super) fn collapse_sr_unit_reductions_with_compatible_gotos(
        &mut self,
    ) -> UnitReductionInliningReport {
        let protected = BitSet::new(self.num_terminals as usize);
        self.collapse_sr_unit_reductions_with_compatible_gotos_except(&protected)
    }

    pub(super) fn collapse_sr_unit_reductions_with_compatible_gotos_except(
        &mut self,
        protected_terminals: &BitSet,
    ) -> UnitReductionInliningReport {
        debug_assert_eq!(protected_terminals.len(), self.num_terminals as usize);
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let journal_started_at = profile_enabled.then(std::time::Instant::now);
        let mut undo = UnitInlineUndo::new(self);
        let journal_ms = journal_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        let inner_started_at = profile_enabled.then(std::time::Instant::now);
        let budget = UnitInlineBudget::from_env();
        self.collapse_sr_unit_reductions_with_compatible_gotos_inner(
            &budget,
            &mut undo,
            protected_terminals,
        );
        let mut report = budget.report();
        let inner_ms = inner_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        if report.aborted {
            undo.rollback(self);
        } else {
            report.changed_original_states = undo.changed_original_states();
        }
        if let (Some(journal_ms), Some(inner_ms)) = (journal_ms, inner_ms) {
            eprintln!(
                "[glrmask/profile][unit_reduction_inlining] outcome={} reason={} iterations={} cells={} synthetic_states={} stack_effect_visits={} journal_ms={:.3} inner_ms={:.3} elapsed_ms={:.3} max_ms={} max_iterations={} max_cells={} max_synthetic_states={} max_stack_effect_visits={}",
                if report.aborted { "aborted" } else { "committed" },
                report.reason.unwrap_or("none"),
                report.iterations,
                report.cells,
                report.synthetic_states,
                report.stack_effect_visits,
                journal_ms,
                inner_ms,
                report.elapsed_ms,
                budget.max_ms,
                budget.max_iterations,
                budget.max_cells,
                budget.max_synthetic_states,
                budget.max_stack_effect_visits,
            );
        }
        report
    }

    fn collapse_sr_unit_reductions_with_compatible_gotos_inner(
        &mut self,
        budget: &UnitInlineBudget,
        undo: &mut UnitInlineUndo,
        protected_terminals: &BitSet,
    ) {
        self.collapse_sr_unit_reductions_with_completed_work_abort(
            budget,
            undo,
            protected_terminals,
            completed_work_abort_eligible(self.construction)
                && env_flag_enabled(UNIT_INLINE_COMPLETED_ABORT_ENV, true),
        );
    }

    fn collapse_sr_unit_reductions_with_completed_work_abort(
        &mut self,
        budget: &UnitInlineBudget,
        undo: &mut UnitInlineUndo,
        protected_terminals: &BitSet,
        stop_proven_abort_work: bool,
    ) {
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let original_num_states = self.num_states;
        let mut constituent_sets: Vec<StateSubset> = (0..self.num_states)
            .map(|state| smallvec![state])
            .collect();
        // Singleton subsets map directly to their state ID, so only synthetic
        // (multi-state) subsets need hash-map entries.
        let mut subset_to_state: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
        let mut failed_subsets: FxHashSet<Vec<u32>> = FxHashSet::default();
        let mut dirty_original_states: BTreeSet<u32> = BTreeSet::new();
        let mut cached_predecessors: Option<RuntimePredecessors> = None;
        let mut next_worklist: Option<Vec<UnitInlineCell>> = None;
        let mut dependencies = UnitInlineDependencies::default();

        loop {
            if !budget.record_iteration() {
                break;
            }
            let iteration = budget.iterations();
            let worklist = next_worklist.take();
            let using_worklist = worklist.is_some();
            let cells_before = budget.cells();
            let visits_before = budget.stack_effect_visits();
            let refresh_started_at = profile_enabled.then(std::time::Instant::now);
            if !using_worklist && !dirty_original_states.is_empty() {
                refresh_merged_states_depending_on(
                    self,
                    original_num_states,
                    &mut constituent_sets,
                    &mut subset_to_state,
                    &mut failed_subsets,
                    &dirty_original_states,
                    budget,
                );
                if budget.is_aborted() {
                    break;
                }
                dirty_original_states.clear();
            }

            let refresh_ms = refresh_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let predecessors_started_at = profile_enabled.then(std::time::Instant::now);
            if !using_worklist {
                cached_predecessors = Some(build_runtime_state_predecessors(
                    self,
                    original_num_states,
                    &constituent_sets,
                ));
            }
            let predecessors = cached_predecessors
                .as_ref()
                .expect("first unit-inlining pass constructs predecessors");
            let predecessors_ms = if using_worklist {
                0.0
            } else {
                predecessors_started_at
                    .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0)
            };
            // The scan below computes updates against a stable snapshot of the
            // current original rows for this iteration. Delayed synthetic
            // states may be appended, but existing rows are not mutated until
            // after the scan.
            let nstates = original_num_states as usize;
            let worklist_cells = worklist.as_ref().map(|cells| {
                cells.iter().copied().collect::<FxHashSet<UnitInlineCell>>()
            });
            let synthetic_states_before_scan = self.num_states;
            let mut pending_updates: Vec<(usize, TerminalID, CellUpdate)> = Vec::new();
            let scan_started_at = profile_enabled.then(std::time::Instant::now);

            // Phase 1 (parallel, read-only): stack-shift inlining is a pure
            // function of the stable table snapshot and the predecessors graph.
            // Schedule only states that actually have a candidate cell. The old
            // implementation spawned a task, a local cache, and a child budget
            // for every state, even though most rows contain only shifts.
            let candidate_states = if let Some(cells) = &worklist_cells {
                let mut states = cells
                    .iter()
                    .map(|&(state, _)| state as usize)
                    .collect::<Vec<_>>();
                states.sort_unstable();
                states.dedup();
                states
            } else {
                (0..nstates)
                    .filter(|&state| {
                        self.action[state]
                            .iter()
                            .any(|(terminal, action)| {
                                !protected_terminals.contains(terminal as usize)
                                    && action_has_inlinable_reductions(action)
                            })
                    })
                    .collect::<Vec<_>>()
            };
            let stack_phase_started_at = profile_enabled.then(std::time::Instant::now);
            let isolate_state_stack_effect_budget =
                self.construction == GlrTableConstruction::ExperimentalCoreMerged;
            let state_stack_effect_visit_limit = unit_inline_state_max_stack_effect_visits();
            // Core-merged children stop at the local cap plus one failed visit.
            // If even every child exhausting that cap cannot exceed the parent
            // limit, no certificate is possible in this phase: pay no atomics.
            let certificate_possible = !isolate_state_stack_effect_budget
                || candidate_states.len().saturating_mul(
                    state_stack_effect_visit_limit
                        .min(budget.max_stack_effect_visits)
                        .saturating_add(1),
                ) > budget.max_stack_effect_visits
                    .saturating_sub(budget.stack_effect_visits());
            let completed_work = (stop_proven_abort_work && certificate_possible)
                .then(|| CompletedStackEffectWork::new(budget))
                .flatten();
            let per_state: Vec<(
                usize,
                Vec<(TerminalID, StackInlineAttempt)>,
                Vec<(UnitInlineCell, UnitInlineCell)>,
                UnitInlineBudget,
            )> = {
                let table: &GLRTable = self;
                let predecessors = &predecessors;
                candidate_states
                    .par_iter()
                    .map(|&state| {
                        let local_budget = if isolate_state_stack_effect_budget {
                            budget.child_with_stack_effect_visit_limit(
                                state_stack_effect_visit_limit,
                            )
                        } else {
                            budget.child()
                        };
                        if completed_work.as_ref().is_some_and(|work| work.exceeded()) {
                            // A complete result slot retains indexed collection
                            // order; this empty job contributes no actual work.
                            return (state, Vec::new(), Vec::new(), local_budget);
                        }
                        let mut local_depth_cache: FxHashMap<(u32, u32), Option<StateSubset>> =
                            FxHashMap::default();
                        let mut reads = Vec::new();
                        let rows = table.action[state]
                            .iter()
                            .filter(|(tid, action)| {
                                !protected_terminals.contains(*tid as usize)
                                    && action_has_inlinable_reductions(action)
                                    && worklist_cells.as_ref().is_none_or(|cells| {
                                        cells.contains(&(state as u32, *tid))
                                    })
                            })
                            .map(|(tid, action)| {
                                let inlined = if local_budget.is_aborted() {
                                    StackInlineAttempt {
                                        action: None,
                                        outcome: StackInlineOutcomeKind::TraversalFailed,
                                    }
                                } else if is_trivial_origin_dependent_one_push(
                                    table,
                                    predecessors,
                                    state as u32,
                                    tid,
                                    action,
                                    &mut reads,
                                ) {
                                    StackInlineAttempt {
                                        action: None,
                                        outcome: StackInlineOutcomeKind::TrivialOriginDependentOnePush,
                                    }
                                } else {
                                    try_inline_action_to_stack_shifts_detailed(
                                        table,
                                        predecessors,
                                        state as u32,
                                        tid,
                                        action,
                                        &mut local_depth_cache,
                                        &mut reads,
                                        &local_budget,
                                    )
                                };
                                (tid, inlined)
                            })
                            .collect();
                        if let Some(work) = &completed_work {
                            work.record_completed(local_budget.stack_effect_visits());
                        }
                        (state, rows, reads, local_budget)
                    })
                    .collect()
            };

            let stack_phase_ms = stack_phase_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            if completed_work.as_ref().is_some_and(|work| work.exceeded()) {
                // Every completed child's nonnegative count is charged by both
                // reference fold policies, including locally exhausted states.
                // The reference therefore must abort before committing this
                // iteration. Account for all completed work and use the same
                // outer undo journal to restore earlier iterations as well.
                for (_, _, _, child) in &per_state {
                    budget.fold_in_counts(child);
                }
                budget.abort(ABORT_STACK_EFFECT_VISITS);
                if profile_enabled {
                    eprintln!(
                        "[glrmask/profile][unit_reduction_completed_abort] candidates={} nonempty_results={} stack_phase_ms={stack_phase_ms:.3} completed_stack_visits={} max_stack_visits={}",
                        candidate_states.len(),
                        per_state.iter().filter(|(_, rows, _, _)| !rows.is_empty()).count(),
                        budget.stack_effect_visits(),
                        budget.max_stack_effect_visits,
                    );
                }
                break;
            }
            let unit_phase_started_at = profile_enabled.then(std::time::Instant::now);
            let mut stack_effect_candidates = 0usize;
            let mut unit_candidates = 0usize;
            let mut stack_effect_resolved = 0usize;
            let mut stack_effect_updates = 0usize;
            let mut unit_updates = 0usize;
            let mut candidate_kind_counts = [0usize; StackInlineCandidateKind::COUNT];
            let mut resolved_kind_counts = [0usize; StackInlineCandidateKind::COUNT];
            let mut direct_unit_candidates = 0usize;
            let mut direct_unit_resolved = 0usize;
            let mut stack_outcome_counts = [0usize; StackInlineOutcomeKind::COUNT];
            // Phase 2 (serial): apply stack-shift results in canonical state
            // order and run unit-reduction inlining (which may synthesize
            // shared states). `candidate_states` is sorted and Rayon preserves
            // indexed-iterator collection order.
            if !budget.is_aborted() {
                'scan: for (state, row_stack_shifts, reads, local_budget) in per_state {
                    if isolate_state_stack_effect_budget {
                        budget.fold_in_isolated_state(&local_budget);
                    } else {
                        budget.fold_in(&local_budget);
                    }
                    dependencies.record_reads(reads);
                    if budget.is_aborted() {
                        break 'scan;
                    }
                    for (tid, stack_attempt) in row_stack_shifts {
                        let StackInlineAttempt {
                            action: stack_inlined,
                            outcome: stack_outcome,
                        } = stack_attempt;
                        let Some(action) = self.action[state].get(&tid).cloned() else {
                            continue;
                        };
                        stack_effect_candidates += 1;
                        stack_outcome_counts[stack_outcome.index()] += 1;
                        let candidate_kind = stack_inline_candidate_kind(&action);
                        candidate_kind_counts[candidate_kind.index()] += 1;
                        let is_unit_candidate = action_needs_unit_reduction_inlining(&action);
                        unit_candidates += usize::from(is_unit_candidate);
                        let is_direct_unit_candidate = profile_enabled
                            && matches!(
                                &action,
                                Action::Reduce(nonterminal, 1)
                                    if unit_reduce_destination(
                                        self,
                                        predecessors,
                                        state as u32,
                                        *nonterminal,
                                    )
                                    .is_some()
                            );
                        direct_unit_candidates += usize::from(is_direct_unit_candidate);
                        let resolved_by_stack_effect = stack_inlined.is_some();
                        direct_unit_resolved += usize::from(
                            is_direct_unit_candidate && resolved_by_stack_effect,
                        );
                        stack_effect_resolved += usize::from(resolved_by_stack_effect);
                        if resolved_by_stack_effect {
                            resolved_kind_counts[candidate_kind.index()] += 1;
                        }
                        if !budget.record_cell() {
                            break 'scan;
                        }

                        let update = if let Some(inlined) = stack_inlined {
                            Some(CellUpdate::Set(inlined))
                        } else if is_unit_candidate {
                            let mut visiting = BTreeSet::new();
                            match try_inline_unit_reductions_for_cell_inner(
                                self,
                                predecessors,
                                state as u32,
                                tid,
                                &action,
                                &mut constituent_sets,
                                &mut subset_to_state,
                                &mut failed_subsets,
                                &mut visiting,
                                (state as u32, tid),
                                &mut dependencies,
                                budget,
                            ) {
                                Ok(update) => update,
                                Err(()) => continue,
                            }
                        } else {
                            None
                        };
                        if budget.is_aborted() {
                            break 'scan;
                        }

                        match update {
                            Some(CellUpdate::Set(new_action)) if new_action != action => {
                                if resolved_by_stack_effect {
                                    stack_effect_updates += 1;
                                } else {
                                    unit_updates += 1;
                                }
                                pending_updates.push((state, tid, CellUpdate::Set(new_action)));
                            }
                            Some(CellUpdate::Remove) => {
                                if resolved_by_stack_effect {
                                    stack_effect_updates += 1;
                                } else {
                                    unit_updates += 1;
                                }
                                pending_updates.push((state, tid, CellUpdate::Remove));
                            }
                            _ => {}
                        }
                    }
                }
            }


            let unit_phase_ms = unit_phase_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            if pending_updates.is_empty() || budget.is_aborted() {
                if profile_enabled {
                    eprintln!(
                        "[glrmask/profile][unit_reduction_inlining_iteration] iteration={} refresh_ms={:.3} predecessors_ms={:.3} stack_phase_ms={:.3} unit_phase_ms={:.3} scan_ms={:.3} apply_ms=0.000 candidates={} unit_candidates={} stack_resolved={} stack_updates={} unit_updates={} kinds={:?} resolved_kinds={:?} direct_unit_candidates={} direct_unit_resolved={} outcomes={:?} scanned_cells={} stack_effect_visits={} pending_updates={} terminal={}",
                        iteration,
                        refresh_ms,
                        predecessors_ms,
                        stack_phase_ms,
                        unit_phase_ms,
                        scan_started_at
                            .expect("profile start exists when profiling is enabled")
                            .elapsed()
                            .as_secs_f64()
                            * 1000.0,
                        stack_effect_candidates,
                        unit_candidates,
                        stack_effect_resolved,
                        stack_effect_updates,
                        unit_updates,
                        candidate_kind_counts,
                        resolved_kind_counts,
                        direct_unit_candidates,
                        direct_unit_resolved,
                        stack_outcome_counts,
                        budget.cells() - cells_before,
                        budget.stack_effect_visits() - visits_before,
                        pending_updates.len(),
                        if budget.is_aborted() { "abort" } else { "stable" },
                    );
                }
                break;
            }

            let update_count = pending_updates.len();
            let scan_ms = scan_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let apply_started_at = profile_enabled.then(std::time::Instant::now);
            let mut changed_cells = Vec::with_capacity(update_count);
            let mut predecessor_edges_changed = 0usize;
            for (state, tid, update) in pending_updates {
                undo.record_cell(self, state, tid);
                let old_action = self.action[state].get(&tid).cloned();
                match update {
                    CellUpdate::Set(new_action) => {
                        predecessor_edges_changed += usize::from(
                            old_action
                                .as_ref()
                                .and_then(runtime_predecessor_transition)
                                != runtime_predecessor_transition(&new_action),
                        );
                        self.action[state].insert(tid, new_action);
                    }
                    CellUpdate::Remove => {
                        predecessor_edges_changed += usize::from(old_action.as_ref().and_then(runtime_predecessor_transition).is_some());
                        self.action[state].remove(&tid);
                        if self.advance.len() == self.num_states as usize
                            && let Some(bit) = self.terminal_bit(tid)
                        {
                            self.advance[state].clear(bit);
                        }
                    }
                }
                dirty_original_states.insert(state as u32);
                changed_cells.push((state as u32, tid));
            }
            let can_use_worklist = predecessor_edges_changed == 0
                && self.num_states == synthetic_states_before_scan
                && self.num_states == original_num_states;
            if can_use_worklist {
                next_worklist = Some(dependencies.worklist_for_changes(changed_cells.iter().copied()));
            } else {
                cached_predecessors = None;
                next_worklist = None;
            }
            if profile_enabled {
                eprintln!(
                    "[glrmask/profile][unit_reduction_inlining_iteration] iteration={} refresh_ms={:.3} predecessors_ms={:.3} stack_phase_ms={:.3} unit_phase_ms={:.3} scan_ms={:.3} apply_ms={:.3} candidates={} unit_candidates={} stack_resolved={} stack_updates={} unit_updates={} kinds={:?} resolved_kinds={:?} direct_unit_candidates={} direct_unit_resolved={} outcomes={:?} scanned_cells={} stack_effect_visits={} pending_updates={} terminal=continue",
                    iteration,
                    refresh_ms,
                    predecessors_ms,
                    stack_phase_ms,
                    unit_phase_ms,
                    scan_ms,
                    apply_started_at
                        .expect("profile start exists when profiling is enabled")
                        .elapsed()
                        .as_secs_f64()
                        * 1000.0,
                    stack_effect_candidates,
                    unit_candidates,
                    stack_effect_resolved,
                    stack_effect_updates,
                    unit_updates,
                    candidate_kind_counts,
                    resolved_kind_counts,
                    direct_unit_candidates,
                    direct_unit_resolved,
                    stack_outcome_counts,
                    budget.cells() - cells_before,
                    budget.stack_effect_visits() - visits_before,
                    update_count,
                );
            }
        }
    }

    /// Merge states that are equivalent for recognition purposes.
    ///
    /// Unlike `merge_identical_rows` which requires exact action/goto match,
    /// this treats two Reduce actions as equivalent when they have the same
    /// `(lhs, rhs_len)`, since the parser only uses those two fields.
    /// It also merges goto columns for nonterminals that become equivalent.
    /// Iterates until stable.
    pub(super) fn merge_recognizer_equivalent(&mut self) {
        loop {
            let prev_states = self.num_states;

            // Step 1: With Reduce(nt, len) representation, reduces are already
            // canonicalized by (lhs, rhs_len). Just merge identical rows.

            // Step 2: Merge states with identical rows.
            self.merge_identical_rows();

            // Step 3: Merge goto columns for nonterminals whose goto vectors
            // are identical across all states (i.e., they always land in the
            // same state, or are both absent).
            let nstates = self.num_states as usize;
            let mut all_nts: BTreeSet<NonterminalID> = BTreeSet::new();
            let mut columns_by_nt: FxHashMap<NonterminalID, Vec<(u32, (u32, bool))>> =
                FxHashMap::default();
            for (state, goto_row) in self.goto.iter().enumerate() {
                for (&nt, &target) in goto_row {
                    all_nts.insert(nt);
                    columns_by_nt
                        .entry(nt)
                        .or_default()
                        .push((state as u32, target));
                }
            }

            // Build sparse goto signatures for each nonterminal and group by them.
            let mut column_to_canon: FxHashMap<Vec<(u32, (u32, bool))>, NonterminalID> =
                FxHashMap::default();
            let mut nt_remap: FxHashMap<NonterminalID, NonterminalID> = FxHashMap::default();
            for &nt in &all_nts {
                let col = columns_by_nt.remove(&nt).unwrap_or_default();
                if let Some(&canon) = column_to_canon.get(&col) {
                    nt_remap.insert(nt, canon);
                } else {
                    column_to_canon.insert(col, nt);
                }
            }

            if !nt_remap.is_empty() {
                // Rewrite goto entries: merge columns.
                for state in 0..nstates {
                    let old = std::mem::take(&mut self.goto[state]);
                    let mut new_goto = GotoRow::default();
                    for (&nt, &target) in old.iter() {
                        let canon_nt = nt_remap.get(&nt).copied().unwrap_or(nt);
                        // All remapped NTs should have the same target; just insert.
                        new_goto.insert(canon_nt, target);
                    }
                    self.goto[state] = new_goto;
                }

                // Rewrite nonterminal IDs in action entries (Reduce and Split reduces).
                for state in 0..nstates {
                    let old = std::mem::take(&mut self.action[state]);
                    let new_action: ActionRow = old
                        .iter()
                        .map(|(tid, action)| {
                            let remapped = match action {
                                Action::Reduce(nt, len) => {
                                    let canon = nt_remap.get(&nt).copied().unwrap_or(*nt);
                                    Action::Reduce(canon, *len)
                                }
                                Action::StackShifts(shifts) => Action::StackShifts(shifts.clone()),
                                Action::GuardedStackShifts(shifts) => Action::GuardedStackShifts(shifts.clone()),
                                Action::Split { shift, reduces, accept } => {
                                    let reduces = reduces
                                        .into_iter()
                                        .map(|(nt, len)| {
                                            let canon = nt_remap.get(nt).copied().unwrap_or(*nt);
                                            (canon, *len)
                                        })
                                        .collect();
                                    Action::Split { shift: *shift, reduces, accept: *accept }
                                }
                                other => other.clone(),
                            };
                            (tid, remapped)
                        })
                        .collect();
                    self.action[state] = new_action;
                }

                // Rewrite rule LHS to use canonical NTs.
                for rule in &mut self.rules {
                    if let Some(&canon) = nt_remap.get(&rule.lhs) {
                        rule.lhs = canon;
                    }
                }

                // Rewrite rule RHS nonterminals to use canonical NTs.
                for rule in &mut self.rules {
                    for sym in &mut rule.rhs {
                        if let Symbol::Nonterminal(nt) = sym {
                            if let Some(&canon) = nt_remap.get(nt) {
                                *nt = canon;
                            }
                        }
                    }
                }

                // Merge identical rows again after NT merging.
                self.merge_identical_rows();
            }

            // Step 4: Local split collapsing.
            // For each remaining Split action, check if all reduces land in the
            // same goto target from every predecessor state.  If so, the split
            // is invisible to a recognizer and we can collapse it.
            //
            // Two sub-passes:
            //  4a (original) — immediate goto-target equality from all static predecessors.
            //  4b (new)      — speculative reduce-chain convergence: simulate
            //      each alternative reduce for up to MAX_SPEC_DEPTH steps,
            //      collecting the set of (top-state) the chain reaches.
            //      If all alternatives converge to the same set, collapse.
            let nstates2 = self.num_states as usize;

            // Build predecessor map: for each state, which states can be
            // "goto_from" after a rhs_len=K pop.
            // For rhs_len=1: predecessor is any state X such that
            //   goto[X][*] == this_state  OR  shift in action[X][*] -> this_state
            let mut predecessors: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); nstates2];
            for x in 0..nstates2 {
                for (_, action) in &self.action[x] {
                    if let Some(target) = action.shift_target() {
                        predecessors[target as usize].insert(x as u32);
                    }
                }
                for (_, &(target, _)) in &self.goto[x] {
                    predecessors[target as usize].insert(x as u32);
                }
            }

            let mut collapsed_any = false;
            let mut collapses: Vec<(usize, TerminalID, (NonterminalID, u32))> = Vec::new();
            for state in 0..nstates2 {
                for (tid, action) in &self.action[state] {
                    if let Action::Split { shift, reduces, accept } = action {
                        // Only handle pure-reduce splits (no shift, no accept).
                        if shift.is_some() || *accept {
                            continue;
                        }
                        if reduces.is_empty() {
                            continue;
                        }
                        // Check: do all reduces have the same rhs_len?
                        let (_, rhs_len) = reduces[0];
                        if reduces.iter().any(|&(_, l)| l != rhs_len) {
                            continue;
                        }
                        // For rhs_len=K, find all states that are K levels
                        // up in the stack (predecessors^K).
                        let mut candidate_froms: BTreeSet<u32> = BTreeSet::new();
                        candidate_froms.insert(state as u32);
                        for _ in 0..rhs_len {
                            let mut next = BTreeSet::new();
                            for &s in &candidate_froms {
                                if let Some(preds) = predecessors.get(s as usize) {
                                    next.extend(preds);
                                }
                            }
                            candidate_froms = next;
                        }
                        if candidate_froms.is_empty() {
                            continue;
                        }
                        // Check if all reduces lead to the same goto target
                        // from every predecessor.
                        let lhss: Vec<NonterminalID> = reduces
                            .iter()
                            .map(|&(nt, _)| nt)
                            .collect();
                        let mut all_same = true;
                        'pred_loop: for &pred in &candidate_froms {
                            let first_target = self.goto[pred as usize].get(&lhss[0]).map(|&(t, _)| t);
                            for &lhs in &lhss[1..] {
                                let target = self.goto[pred as usize].get(&lhs).map(|&(t, _)| t);
                                if target != first_target {
                                    all_same = false;
                                    break 'pred_loop;
                                }
                            }
                        }
                        if all_same {
                            collapses.push((state, tid, reduces[0]));
                        }
                    }
                }
            }

            for (state, tid, reduce_info) in collapses {
                self.action[state].insert(tid, Action::Reduce(reduce_info.0, reduce_info.1));
                collapsed_any = true;
            }

            // Step 4b: Deep split collapsing via stack-relative chain following.
            //
            // For pure R/R splits not collapsed in 4a, simulate the full reduce
            // chain for each alternative.  Track predecessor depth relative to
            // the ORIGINAL split state S (not intermediate chain states).
            //
            // The stack at the split: …→ preds^K(S) →…→ S (top)
            //
            // After alternative reduce Ri (pop=rhs_len(Ri)):
            //   - Expose state at depth rhs_len(Ri) below S
            //   - goto from that state with lhs(Ri) → push T1
            //   - If T1 has another reduce on the same terminal, follow it:
            //     pop rhs_len from T1's position, which goes further below S
            //   - Continue until we reach a non-reduce action
            //
            // If all alternatives' chains converge to the same final state
            // (same goto target from preds^(total_depth) of S), collapse.
            //
            // Two sub-passes:
            //  4b-i: filter out split-state predecessors (handles circular deps)
            //  4b-ii: deep chain following for remaining unconverged splits
            let mut spec_collapses: Vec<(usize, TerminalID, (NonterminalID, u32))> = Vec::new();

            // Build set of (state, terminal) pairs that have pure R/R splits
            let pure_rr_splits: BTreeSet<(usize, TerminalID)> = {
                let mut set = BTreeSet::new();
                for s in 0..nstates2 {
                    for (t, a) in &self.action[s] {
                        if let Action::Split { shift, reduces: _, accept } = a {
                            if shift.is_none() && !*accept {
                                set.insert((s, t));
                            }
                        }
                    }
                }
                set
            };

            for state in 0..nstates2 {
                for (tid, action) in &self.action[state] {
                    let Action::Split { shift, reduces, accept } = action else { continue };
                    if shift.is_some() || *accept { continue }
                    if reduces.is_empty() { continue }

                    let (_, rhs_len) = reduces[0];
                    if reduces.iter().any(|&(_, l)| l != rhs_len) {
                        continue;
                    }
                    let reduces = reduces.clone();

                    // Compute candidate_froms (predecessors^K of the split state)
                    let mut candidate_froms: BTreeSet<u32> = BTreeSet::new();
                    candidate_froms.insert(state as u32);
                    for _ in 0..rhs_len {
                        let mut next = BTreeSet::new();
                        for &s in &candidate_froms {
                            if let Some(preds) = predecessors.get(s as usize) {
                                next.extend(preds);
                            }
                        }
                        candidate_froms = next;
                    }
                    if candidate_froms.is_empty() { continue }

                    // 4b-i: Filter out predecessors that are themselves split states
                    let filtered: BTreeSet<u32> = candidate_froms.iter()
                        .filter(|&&p| !pure_rr_splits.contains(&(p as usize, tid)))
                        .copied()
                        .collect();

                    if filtered.is_empty() {
                        spec_collapses.push((state, tid, reduces[0]));
                        continue;
                    }

                    // Simple check: do all reduces converge from filtered preds?
                    let lhss: Vec<NonterminalID> = reduces
                        .iter()
                        .map(|&(nt, _)| nt)
                        .collect();
                    let mut simple_converge = true;
                    'pred_simple: for &pred in &filtered {
                        let first_target = self.goto[pred as usize].get(&lhss[0]).map(|&(t, _)| t);
                        for &lhs in &lhss[1..] {
                            if self.goto[pred as usize].get(&lhs).map(|&(t, _)| t) != first_target {
                                simple_converge = false;
                                break 'pred_simple;
                            }
                        }
                    }
                    if simple_converge {
                        spec_collapses.push((state, tid, reduces[0]));
                        continue;
                    }


                    // 4b-ii: Deep chain following.
                    // For each alternative, simulate the reduce chain and track
                    // the total depth popped from the original split state S.
                    //
                    // Stack model: After initial reduce Ri (pop=K) from S:
                    //   base_depth = K (below S)
                    //   goto_from = preds^K(S)
                    //   push T1 = goto[goto_from][lhs(Ri)]
                    //   T1 sits at depth K-1 (one above goto_from)
                    //
                    // After follow-up reduce Rj (pop=M) from T1:
                    //   We pop M items from T1's position. T1 is at K-1.
                    //   Popping 1 removes T1 itself (back to K).
                    //   Popping M total goes to depth K + M - 1.
                    //   base_depth = K + M - 1
                    //   goto_from = preds^(K+M-1)(S)
                    //   push T2, sits at K + M - 2
                    //
                    // In general: after n reduces with pop values K1,K2,...,Kn,
                    //   base_depth = K1 + K2 + ... + Kn - (n-1)
                    //   = sum(Ki) - n + 1
                    //
                    // The chain terminates when the action at the pushed state
                    // is not a Reduce on terminal T.
                    //
                    // All alternatives converge if they reach the same
                    // (base_depth, final_lhs) and goto[preds^base_depth][lhs]
                    // agrees for all preds.
                    const MAX_CHAIN: usize = 32;

                    // Follow one alternative's chain.  Returns (base_depth, final_lhs)
                    // or None if the chain diverges or is too deep.
                    let follow = |first_nt: NonterminalID, _first_len: u32| -> Option<(usize, NonterminalID)> {
                        let mut depth = rhs_len as usize; // after initial reduce

                        // Compute goto targets from preds^depth(state) with lhs
                        let preds_at_depth = |d: usize| -> BTreeSet<u32> {
                            let mut s = BTreeSet::new();
                            s.insert(state as u32);
                            for _ in 0..d {
                                let mut next = BTreeSet::new();
                                for &st in &s {
                                    if let Some(ps) = predecessors.get(st as usize) {
                                        next.extend(ps);
                                    }
                                }
                                s = next;
                            }
                            s
                        };

                        let mut current_lhs = first_nt;
                        for _ in 0..MAX_CHAIN {
                            let preds = preds_at_depth(depth);
                            if preds.is_empty() { return None }

                            // Get goto targets
                            let mut goto_targets: BTreeSet<u32> = BTreeSet::new();
                            for &p in &preds {
                                if let Some(&(gt, _)) = self.goto[p as usize].get(&current_lhs) {
                                    goto_targets.insert(gt);
                                }
                            }
                            if goto_targets.is_empty() { return None }

                            // Check action at goto targets on terminal tid
                            let mut next_reduce: Option<(NonterminalID, u32)> = None;
                            let mut all_reduce = true;
                            for &gt in &goto_targets {
                                match self.action.get(gt as usize).and_then(|r| r.get(&tid)) {
                                    Some(Action::Reduce(nt, len)) => {
                                        let info = (*nt, *len);
                                        match next_reduce {
                                            None => next_reduce = Some(info),
                                            Some(nr) if nr == info => {}
                                            _ => { all_reduce = false; break }
                                        }
                                    }
                                    _ => {
                                        // Chain terminates
                                        return Some((depth, current_lhs));
                                    }
                                }
                            }
                            if !all_reduce { return None }

                            // Follow the next reduce
                            let (next_nt, next_len) = next_reduce.unwrap();
                            depth = depth + next_len as usize - 1;
                            current_lhs = next_nt;
                        }
                        None // Too deep
                    };

                    // Follow all alternatives
                    let mut first_result: Option<(usize, NonterminalID)> = None;
                    let mut chain_converge = true;
                    for &(nt, len) in &reduces {
                        match follow(nt, len) {
                            Some(result) => {
                                match first_result {
                                    None => first_result = Some(result),
                                    Some(prev) if prev == result => {}
                                    _ => { chain_converge = false; break }
                                }
                            }
                            None => { chain_converge = false; break }
                        }
                    }

                    if !chain_converge { continue }
                    let Some((final_depth, final_lhs)) = first_result else { continue };

                    // All alternatives converge to (final_depth, final_lhs).
                    // Check: from preds^final_depth(state), do all gotos agree?
                    let mut final_preds = BTreeSet::new();
                    final_preds.insert(state as u32);
                    for _ in 0..final_depth {
                        let mut next = BTreeSet::new();
                        for &s in &final_preds {
                            if let Some(ps) = predecessors.get(s as usize) {
                                next.extend(ps);
                            }
                        }
                        final_preds = next;
                    }

                    let mut goto_target_val: Option<Option<u32>> = None;
                    let mut targets_agree = true;
                    for &pred in &final_preds {
                        let target = self.goto[pred as usize].get(&final_lhs).map(|&(t, _)| t);
                        match goto_target_val {
                            None => goto_target_val = Some(target),
                            Some(prev) if prev == target => {}
                            _ => { targets_agree = false; break }
                        }
                    }

                    if targets_agree {
                        spec_collapses.push((state, tid, reduces[0]));
                    }
                }
            }

            for (state, tid, reduce_info) in spec_collapses {
                self.action[state].insert(tid, Action::Reduce(reduce_info.0, reduce_info.1));
                collapsed_any = true;
            }

            if collapsed_any {
                self.merge_identical_rows();
            }

            if self.num_states == prev_states {
                break;
            }
        }
    }

    /// Collapse recognizer-equivalent stack-effect alternatives by replacing a
    /// set of pushed LR stack suffixes with one synthetic suffix state.
    ///
    /// A synthetic suffix state denotes a finite union of concrete LR stack
    /// suffixes over the same lower stack. Its rows are compiled from those
    /// concrete suffixes into ordinary `StackShifts`/`GuardedStackShifts`, so the
    /// runtime still consumes the same table representation. The pass is purely
    /// table-level: it does not change import, grammar lowering, or parser
    /// runtime behavior.
    pub(super) fn quotient_recognizer_stack_suffixes(&mut self) {
        let protected = BitSet::new(self.num_terminals as usize);
        self.quotient_recognizer_stack_suffixes_except(&protected);
    }

    pub(super) fn quotient_recognizer_stack_suffixes_except(&mut self, protected: &BitSet) {
        if !recognizer_suffix_quotient_enabled() {
            return;
        }

        let mut quotient = SuffixQuotient::new();
        let original_states = self.num_states as usize;
        let mut changed = false;

        for state in 0..original_states {
            let terminals: Vec<TerminalID> = self.action[state].keys().collect();
            for terminal in terminals {
                if protected.get(terminal as usize) {
                    continue;
                }
                let Some(action) = self.action[state].get(&terminal).cloned() else {
                    continue;
                };
                let Some(new_action) = quotient.normalize_action(self, action.clone()) else {
                    continue;
                };
                if new_action != action {
                    self.action[state].insert(terminal, new_action);
                    changed = true;
                }
            }
        }

        if changed || quotient.created_states > 0 {
            self.extend_advance_rows_from_actions();
            self.validate_structure("recognizer suffix quotient before prune");
            self.prune_unreachable_states();
            self.merge_identical_rows();
            self.validate_structure("recognizer suffix quotient after merge");
        }
    }
}

#[derive(Debug)]
struct SuffixQuotient {
    suffix_to_state: FxHashMap<Vec<Vec<u32>>, u32>,
    failed_suffixes: FxHashSet<Vec<Vec<u32>>>,
    max_states: usize,
    max_alts: usize,
    max_width: usize,
    max_depth: usize,
    build_depth: usize,
    created_states: usize,
}

impl SuffixQuotient {
    fn new() -> Self {
        Self {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            // Do not lower this default to mask correctness or crash bugs.
            // If a schema fails only above this cap, fix the quotient/table
            // invariant that fails at scale instead of hiding the failure.
            max_states: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_STATES_ENV, 4096),
            max_alts: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_ALTS_ENV, 64),
            max_width: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_WIDTH_ENV, 8),
            // Synthetic rows can recursively require further synthetic rows.
            // The quotient is optional and transactional, so aborting a deeply
            // nested attempt preserves the exact unquotiented table while
            // avoiding a native stack overflow on adversarial grammars.
            max_depth: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_DEPTH_ENV, 128),
            build_depth: 0,
            created_states: 0,
        }
    }

    fn normalize_action(&mut self, table: &mut GLRTable, action: Action) -> Option<Action> {
        match action {
            Action::StackShifts(shifts) => {
                let mut effects = shifts
                    .into_iter()
                    .map(|shift| GuardedStackShift {
                        guards: Vec::new(),
                        pop: shift.pop,
                        pushes: shift.pushes,
                    })
                    .collect::<Vec<_>>();
                let normalized = self.quotient_effect_groups(table, &mut effects).ok()?;
                if Self::action_has_multi_stack_shifts(&normalized) {
                    return None;
                }
                Some(normalized)
            }
            Action::GuardedStackShifts(mut effects) => {
                let normalized = self.quotient_effect_groups(table, &mut effects).ok()?;
                if Self::action_has_multi_stack_shifts(&normalized) {
                    return None;
                }
                Some(normalized)
            }
            Action::Split {
                shift,
                reduces,
                accept,
            } if reduces.is_empty() && !accept => {
                shift.map(|(target, replace)| Action::Shift(target, replace))
            }
            _ => None,
        }
    }

    fn quotient_effect_groups(
        &mut self,
        table: &mut GLRTable,
        effects: &mut Vec<GuardedStackShift>,
    ) -> Result<Action, ()> {
        normalize_guarded_effects_for_suffix_quotient(effects);
        if effects.is_empty() {
            return Err(());
        }

        let mut groups: BTreeMap<(Vec<StackShiftGuard>, u32), Vec<Vec<u32>>> = BTreeMap::new();
        for effect in effects.iter() {
            groups
                .entry((effect.guards.clone(), effect.pop))
                .or_default()
                .push(effect.pushes.clone());
        }

        let mut out = Vec::new();
        for ((guards, pop), mut suffixes) in groups {
            normalize_suffixes(&mut suffixes);
            if suffixes.len() > 1
                && suffixes.iter().all(|suffix| !suffix.is_empty())
                && let Ok(target) = self.ensure_suffix_state(table, suffixes.clone())
            {
                out.push(GuardedStackShift {
                    guards,
                    pop,
                    pushes: vec![target],
                });
            } else {
                for pushes in suffixes {
                    out.push(GuardedStackShift {
                        guards: guards.clone(),
                        pop,
                        pushes,
                    });
                }
            }
        }

        normalize_guarded_effects_for_suffix_quotient(&mut out);
        if out.iter().all(|effect| effect.guards.is_empty()) {
            let shifts = out
                .into_iter()
                .map(|effect| StackShift {
                    pop: effect.pop,
                    pushes: effect.pushes,
                })
                .collect();
            return stack_shift_action(shifts).ok_or(());
        }
        Ok(Action::GuardedStackShifts(out))
    }

    fn ensure_suffix_state(
        &mut self,
        table: &mut GLRTable,
        mut suffixes: Vec<Vec<u32>>,
    ) -> Result<u32, ()> {
        normalize_suffixes(&mut suffixes);
        if suffixes.is_empty() || suffixes.iter().any(|suffix| suffix.is_empty()) {
            return Err(());
        }
        if suffixes.len() > self.max_alts || suffixes.iter().any(|suffix| suffix.len() > self.max_width) {
            return Err(());
        }
        if suffixes.len() == 1 && suffixes[0].len() == 1 {
            return Ok(suffixes[0][0]);
        }
        if let Some(&state) = self.suffix_to_state.get(&suffixes) {
            return Ok(state);
        }
        if self.failed_suffixes.contains(&suffixes) || self.created_states >= self.max_states {
            return Err(());
        }
        if self.build_depth >= self.max_depth {
            return Err(());
        }

        let rollback_state = table.num_states;
        let rollback_created_states = self.created_states;
        let had_advance_rows = table.advance.len() == table.num_states as usize;
        let state = rollback_state;
        table.num_states += 1;
        table.action.push(ActionRow::default());
        table.goto.push(GotoRow::default());
        if had_advance_rows {
            table
                .advance
                .push(super::action_presence_row(&ActionRow::default(), table.num_terminals));
        }
        self.created_states += 1;
        self.suffix_to_state.insert(suffixes.clone(), state);
        self.build_depth += 1;
        let built = (|| {
            let action = self.build_suffix_action_row(table, &suffixes)?;
            let goto = self.build_suffix_goto_row(table, &suffixes)?;
            Ok::<_, ()>((action, goto))
        })();
        self.build_depth -= 1;

        match built {
            Ok((action, goto)) => {
                if Self::action_row_has_multi_stack_shifts(&action) {
                    self.suffix_to_state.retain(|_, mapped_state| *mapped_state < rollback_state);
                    table.action.truncate(rollback_state as usize);
                    table.goto.truncate(rollback_state as usize);
                    if had_advance_rows {
                        table.advance.truncate(rollback_state as usize);
                    }
                    table.num_states = rollback_state;
                    self.created_states = rollback_created_states;
                    self.failed_suffixes.insert(suffixes);
                    return Err(());
                }
                table.action[state as usize] = action;
                table.goto[state as usize] = goto;
                if had_advance_rows {
                    table.advance[state as usize] =
                        super::action_presence_row(&table.action[state as usize], table.num_terminals);
                }
                Ok(state)
            }
            Err(()) => {
                self.suffix_to_state.retain(|_, mapped_state| *mapped_state < rollback_state);
                table.action.truncate(rollback_state as usize);
                table.goto.truncate(rollback_state as usize);
                if had_advance_rows {
                    table.advance.truncate(rollback_state as usize);
                }
                table.num_states = rollback_state;
                self.created_states = rollback_created_states;
                self.failed_suffixes.insert(suffixes);
                Err(())
            }
        }
    }

    fn build_suffix_action_row(
        &mut self,
        table: &mut GLRTable,
        suffixes: &[Vec<u32>],
    ) -> Result<ActionRow, ()> {
        let mut terminals = BTreeSet::new();
        for suffix in suffixes {
            let top = *suffix.last().ok_or(())?;
            for terminal in table.action[top as usize].keys() {
                terminals.insert(terminal);
            }
        }

        let mut row = ActionRow::default();
        for terminal in terminals {
            let mut effects = Vec::new();
            let mut accepts = 0usize;
            for suffix in suffixes {
                let top = *suffix.last().ok_or(())?;
                let Some(action) = table.action[top as usize].get(&terminal).cloned() else {
                    continue;
                };
                self.collect_effects_for_suffix_action(table, suffix, terminal, &action, &mut effects, &mut accepts)?;
            }

            if accepts > 0 {
                if !effects.is_empty() {
                    return Err(());
                }
                row.insert(terminal, Action::Accept);
                continue;
            }

            if effects.is_empty() {
                continue;
            }
            let action = self.quotient_effect_groups(table, &mut effects)?;
            row.insert(terminal, action);
        }
        Ok(row)
    }

    fn build_suffix_goto_row(
        &mut self,
        table: &mut GLRTable,
        suffixes: &[Vec<u32>],
    ) -> Result<GotoRow, ()> {
        let mut nts = BTreeSet::new();
        for suffix in suffixes {
            let top = *suffix.last().ok_or(())?;
            for &nt in table.goto[top as usize].keys() {
                nts.insert(nt);
            }
        }

        let mut row = GotoRow::default();
        for nt in nts {
            let mut result_suffixes = Vec::new();
            for suffix in suffixes {
                let top = *suffix.last().ok_or(())?;
                let Some(&(target, replace)) = table.goto[top as usize].get(&nt) else {
                    continue;
                };
                result_suffixes.push(apply_goto_to_suffix(suffix, target, replace));
            }
            normalize_suffixes(&mut result_suffixes);
            if result_suffixes.is_empty() {
                continue;
            }
            let target = self.ensure_suffix_target(table, result_suffixes)?;
            row.insert(nt, (target, true));
        }
        Ok(row)
    }


fn action_row_has_multi_stack_shifts(row: &ActionRow) -> bool {
    row.values().any(|action| matches!(action, Action::StackShifts(shifts) if shifts.len() > 1))
}

fn action_has_multi_stack_shifts(action: &Action) -> bool {
    matches!(action, Action::StackShifts(shifts) if shifts.len() > 1)
}
    fn ensure_suffix_target(
        &mut self,
        table: &mut GLRTable,
        mut suffixes: Vec<Vec<u32>>,
    ) -> Result<u32, ()> {
        normalize_suffixes(&mut suffixes);
        match suffixes.as_slice() {
            [only] if only.len() == 1 => Ok(only[0]),
            _ => self.ensure_suffix_state(table, suffixes),
        }
    }

    fn collect_effects_for_suffix_action(
        &mut self,
        table: &GLRTable,
        suffix: &[u32],
        terminal: TerminalID,
        action: &Action,
        effects: &mut Vec<GuardedStackShift>,
        accepts: &mut usize,
    ) -> Result<(), ()> {
        if suffix.is_empty() {
            return Err(());
        }
        let Some(&state) = suffix.last() else {
            return Err(());
        };
        let frame = StackEffectFrame {
            pop: 1,
            pushes: suffix.to_vec(),
            guards: Vec::new(),
        };
        collect_suffix_effects_from_frame(
            table,
            terminal,
            state,
            action,
            frame,
            None,
            &mut FxHashSet::default(),
            effects,
            accepts,
        )
    }
}

fn normalize_suffixes(suffixes: &mut Vec<Vec<u32>>) {
    suffixes.sort();
    suffixes.dedup();
}

fn normalize_guarded_effects_for_suffix_quotient(effects: &mut Vec<GuardedStackShift>) {
    for effect in effects.iter_mut() {
        for guard in &mut effect.guards {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
    effects.sort();
    effects.dedup();
}

fn apply_goto_to_suffix(suffix: &[u32], target: u32, replace: bool) -> Vec<u32> {
    let mut out = suffix.to_vec();
    if replace {
        if let Some(top) = out.last_mut() {
            *top = target;
        } else {
            out.push(target);
        }
    } else {
        out.push(target);
    }
    out
}

fn unguarded_suffix_effect(
    suffix: &[u32],
    pop: u32,
    pushes: &[u32],
) -> Result<GuardedStackShift, ()> {
    suffix_effect(suffix, Vec::new(), pop, pushes)
}

fn guarded_suffix_effect(
    suffix: &[u32],
    shift: &GuardedStackShift,
) -> Result<Option<GuardedStackShift>, ()> {
    let suffix_len = suffix.len() as u32;
    let mut translated_guards = Vec::new();

    for guard in &shift.guards {
        if guard.pop < suffix_len {
            let index = suffix.len() - 1 - guard.pop as usize;
            if guard.states.binary_search(&suffix[index]).is_err() {
                return Ok(None);
            }
        } else {
            translated_guards.push(StackShiftGuard {
                pop: 1 + (guard.pop - suffix_len),
                states: guard.states.clone(),
            });
        }
    }

    let effect = suffix_effect(suffix, translated_guards, shift.pop, &shift.pushes)?;
    Ok(Some(effect))
}

fn suffix_effect(
    suffix: &[u32],
    mut guards: Vec<StackShiftGuard>,
    pop: u32,
    pushes: &[u32],
) -> Result<GuardedStackShift, ()> {
    if suffix.is_empty() {
        return Err(());
    }
    let suffix_len = suffix.len() as u32;
    let (macro_pop, macro_pushes) = if pop <= suffix_len {
        let keep = suffix.len() - pop as usize;
        let mut out = suffix[..keep].to_vec();
        out.extend_from_slice(pushes);
        (1, out)
    } else {
        (1 + (pop - suffix_len), pushes.to_vec())
    };

    guards.sort_by_key(|guard| guard.pop);
    guards.dedup();
    debug_assert!(guards.iter().all(|guard| guard.pop <= macro_pop));

    Ok(GuardedStackShift {
        guards,
        pop: macro_pop,
        pushes: macro_pushes,
    })
}

// Compile one concrete action row of a synthetic suffix state into an
// equivalent macro stack effect.
//
// Invariant.  A synthetic suffix state Q_S denotes the finite set S of concrete
// LR stack suffixes.  If the lower stack is alpha, a concrete member suffix s in
// S denotes alpha·s, while the synthetic stack denotes alpha·Q_S.  A macro
// effect produced here is correct when applying it to alpha·Q_S yields exactly
// the same lower-stack result as applying the original LR action sequence to
// alpha·s, for every alpha satisfying its guards.
//
// Shifts and existing stack effects are translated by replacing the synthetic
// state with the concrete suffix, applying the concrete pop/push operation, and
// then re-encoding the result as one macro pop plus pushes.  Reductions are the
// only subtle case: the reduce pop may expose a state inside the represented
// suffix, or it may cross below Q_S into alpha.  In the first case the exposed
// predecessor is known statically.  In the second case the predecessor is not
// known until runtime, so we enumerate table goto sources for the reduced
// nonterminal and add a guard at the exposed depth.  The guarded alternatives
// are disjoint by predecessor state, and states without the goto simply produce
// no branch, matching ordinary LR reduce failure.  After goto, we recursively
// continue on the same terminal until a consuming shift/stack-effect/accept is
// reached, exactly as the GLR interpreter does for reductions.
fn collect_suffix_effects_from_frame(
    table: &GLRTable,
    terminal: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    goto_sources_by_nonterminal: Option<&[Vec<u32>]>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    effects: &mut Vec<GuardedStackShift>,
    accepts: &mut usize,
) -> Result<(), ()> {
    let key = StackEffectVisitKey {
        state,
        tid: terminal,
        action_tag: stack_effect_action_tag(action),
        frame: frame.clone(),
    };
    if !visiting.insert(key.clone()) {
        return Err(());
    }

    let result = (|| {
        match action {
            Action::Skip => {
                effects.push(frame_to_guarded_shift(frame));
                Ok(())
            }
            Action::Shift(target, replace) => {
                let mut frame = frame;
                let effective_replace = *replace && !table.forwarded_shifts.contains(&(state, terminal));
                push_transition_to_frame(&mut frame, *target, effective_replace);
                effects.push(frame_to_guarded_shift(frame));
                Ok(())
            }
            Action::ReplaceShifts(targets) => {
                for &target in targets.iter() {
                    let mut frame = frame.clone();
                    pop_frame(&mut frame, 1);
                    frame.pushes.push(target);
                    effects.push(frame_to_guarded_shift(frame));
                }
                Ok(())
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    let mut frame = frame.clone();
                    pop_frame(&mut frame, shift.pop);
                    frame.pushes.extend_from_slice(&shift.pushes);
                    effects.push(frame_to_guarded_shift(frame));
                }
                Ok(())
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    if let Some(frame) = compose_guarded_shift_with_frame(frame.clone(), shift).ok_or(())? {
                        effects.push(frame_to_guarded_shift(frame));
                    }
                }
                Ok(())
            }
            Action::Reduce(nt, len) => {
                for frame in reduce_suffix_frame(
                    table,
                    frame,
                    *nt,
                    *len,
                    goto_sources_by_nonterminal,
                )? {
                    let Some(&next_state) = frame.pushes.last() else {
                        continue;
                    };
                    let Some(next_action) = table.action[next_state as usize].get(&terminal) else {
                        continue;
                    };
                    collect_suffix_effects_from_frame(
                        table,
                        terminal,
                        next_state,
                        next_action,
                        frame,
                        goto_sources_by_nonterminal,
                        visiting,
                        effects,
                        accepts,
                    )?;
                }
                Ok(())
            }
            Action::Split { shift, reduces, accept } => {
                if *accept {
                    *accepts += 1;
                }
                if let Some((target, replace)) = shift {
                    let shift_action = Action::Shift(*target, *replace);
                    collect_suffix_effects_from_frame(
                        table,
                        terminal,
                        state,
                        &shift_action,
                        frame.clone(),
                        goto_sources_by_nonterminal,
                        visiting,
                        effects,
                        accepts,
                    )?;
                }
                for &(nt, len) in reduces {
                    let reduce_action = Action::Reduce(nt, len);
                    collect_suffix_effects_from_frame(
                        table,
                        terminal,
                        state,
                        &reduce_action,
                        frame.clone(),
                        goto_sources_by_nonterminal,
                        visiting,
                        effects,
                        accepts,
                    )?;
                }
                Ok(())
            }
            Action::Accept => {
                *accepts += 1;
                Ok(())
            }
        }
    })();

    visiting.remove(&key);
    result
}

/// Contextual-sharing specialization of `collect_suffix_effects_from_frame`.
///
/// Nonrecursive consuming actions are exact one-step stack algebra and cannot
/// revisit an action/frame pair, so allocating a fresh cycle-detection hash set
/// for them is pure overhead. Reduction-bearing actions retain the generic
/// recursive compiler unchanged.
fn collect_context_share_effects_from_frame(
    table: &GLRTable,
    terminal: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    goto_sources_by_nonterminal: &[Vec<u32>],
    effects: &mut Vec<GuardedStackShift>,
    accepts: &mut usize,
) -> Result<(), ()> {
    match action {
        Action::Skip => {
            effects.push(frame_to_guarded_shift(frame));
            Ok(())
        }
        Action::Shift(target, replace) => {
            let mut frame = frame;
            let effective_replace =
                *replace && !table.forwarded_shifts.contains(&(state, terminal));
            push_transition_to_frame(&mut frame, *target, effective_replace);
            effects.push(frame_to_guarded_shift(frame));
            Ok(())
        }
        Action::ReplaceShifts(targets) => {
            for &target in targets.iter() {
                let mut next = frame.clone();
                pop_frame(&mut next, 1);
                next.pushes.push(target);
                effects.push(frame_to_guarded_shift(next));
            }
            Ok(())
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                let mut next = frame.clone();
                pop_frame(&mut next, shift.pop);
                next.pushes.extend_from_slice(&shift.pushes);
                effects.push(frame_to_guarded_shift(next));
            }
            Ok(())
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                if let Some(next) =
                    compose_guarded_shift_with_frame(frame.clone(), shift).ok_or(())?
                {
                    effects.push(frame_to_guarded_shift(next));
                }
            }
            Ok(())
        }
        Action::Split {
            shift: Some((target, replace)),
            reduces,
            accept: false,
        } if reduces.is_empty() => {
            let mut frame = frame;
            let effective_replace =
                *replace && !table.forwarded_shifts.contains(&(state, terminal));
            push_transition_to_frame(&mut frame, *target, effective_replace);
            effects.push(frame_to_guarded_shift(frame));
            Ok(())
        }
        _ => collect_suffix_effects_from_frame(
            table,
            terminal,
            state,
            action,
            frame,
            Some(goto_sources_by_nonterminal),
            &mut FxHashSet::default(),
            effects,
            accepts,
        ),
    }
}

fn reduce_suffix_frame(
    table: &GLRTable,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
    goto_sources_by_nonterminal: Option<&[Vec<u32>]>,
) -> Result<Vec<StackEffectFrame>, ()> {
    pop_frame(&mut frame, len);

    let goto_froms: Vec<(u32, Option<Vec<u32>>)> = if let Some(&state) = frame.pushes.last() {
        vec![(state, None)]
    } else if let Some(goto_sources_by_nonterminal) = goto_sources_by_nonterminal {
        goto_sources_by_nonterminal
            .get(nt as usize)
            .into_iter()
            .flatten()
            .copied()
            .map(|state| (state, Some(vec![state])))
            .collect()
    } else {
        table
            .goto
            .iter()
            .enumerate()
            .filter(|(_, row)| row.contains_key(&nt))
            .map(|(state, _)| (state as u32, Some(vec![state as u32])))
            .collect()
    };

    if goto_froms.is_empty() {
        return Ok(Vec::new());
    }

    let guard_pop = frame.pop;
    let mut out = Vec::new();
    for (goto_from, guard_states) in goto_froms {
        let Some((target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            continue;
        };
        let mut next = frame.clone();
        if let Some(states) = guard_states
            && !add_guard_to_frame(&mut next, guard_pop, states)
        {
            continue;
        }
        push_transition_to_frame(&mut next, target, replace);
        out.push(next);
    }
    out.sort();
    out.dedup();
    Ok(out)
}

#[inline]
fn unordered_entry_hash<T: Hash>(entry: &T) -> u64 {
    let mut hasher = FxHasher::default();
    entry.hash(&mut hasher);
    hasher.finish()
}

/// Stable, order-independent bucket hash for a sparse row.
///
/// `SparseRow` can iterate inline entries or an `FxHashMap`; neither iteration
/// order is a semantic property. We therefore combine independently hashed
/// entries commutatively instead of allocating and sorting every row. This hash
/// only selects equality candidates: `rows_equal` remains the exact criterion.
#[inline]
fn unordered_row_hash<K: Hash, V: Hash>(entries: impl Iterator<Item = (K, V)>, len: usize) -> (u64, u64) {
    let mut sum = 0u64;
    let mut mixed_xor = 0u64;
    for (key, value) in entries {
        let entry_hash = unordered_entry_hash(&(key, value));
        sum = sum.wrapping_add(entry_hash);
        mixed_xor ^= entry_hash.rotate_left((entry_hash >> 58) as u32);
    }
    (sum, mixed_xor ^ (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn remapped_stack_shift_cmp(
    left: &StackShift,
    right: &StackShift,
    mapping: &[u32],
) -> std::cmp::Ordering {
    left.pop.cmp(&right.pop).then_with(|| {
        left.pushes
            .iter()
            .map(|&state| mapping[state as usize])
            .cmp(right.pushes.iter().map(|&state| mapping[state as usize]))
    })
}

/// `remap_action_targets` normalizes `StackShifts` after target IDs coalesce.
/// Keep that exact set semantics without allocating a remapped `Action` for
/// every row-fingerprint/equality probe.
fn normalized_remapped_stack_shifts<'a>(
    shifts: &'a [StackShift],
    mapping: &[u32],
) -> SmallVec<[&'a StackShift; 2]> {
    let mut normalized = shifts.iter().collect::<SmallVec<[&StackShift; 2]>>();
    normalized.sort_unstable_by(|left, right| remapped_stack_shift_cmp(left, right, mapping));
    normalized.dedup_by(|left, right| {
        remapped_stack_shift_cmp(*left, *right, mapping) == std::cmp::Ordering::Equal
    });
    normalized
}

fn remapped_stack_shifts_hash(shifts: &[StackShift], mapping: &[u32]) -> u64 {
    let normalized = normalized_remapped_stack_shifts(shifts, mapping);
    let mut h = FxHasher::default();
    4u8.hash(&mut h);
    normalized.len().hash(&mut h);
    for shift in normalized {
        shift.pop.hash(&mut h);
        shift.pushes.len().hash(&mut h);
        for &target in &shift.pushes {
            mapping[target as usize].hash(&mut h);
        }
    }
    h.finish()
}

fn remapped_stack_shifts_equal(
    left: &[StackShift],
    right: &[StackShift],
    mapping: &[u32],
) -> bool {
    let left = normalized_remapped_stack_shifts(left, mapping);
    let right = normalized_remapped_stack_shifts(right, mapping);
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.pop == right.pop
                && left
                    .pushes
                    .iter()
                    .map(|&target| mapping[target as usize])
                    .eq(right.pushes.iter().map(|&target| mapping[target as usize]))
        })
}

#[inline]
fn remapped_action_hash(action: &Action, mapping: &[u32]) -> u64 {
    let mut h = FxHasher::default();
    match action {
        Action::Shift(target, replace) => {
            0u8.hash(&mut h);
            mapping[*target as usize].hash(&mut h);
            replace.hash(&mut h);
        }
        Action::Reduce(nt, len) => {
            1u8.hash(&mut h);
            nt.hash(&mut h);
            len.hash(&mut h);
        }
        Action::Split { shift, reduces, accept } => {
            2u8.hash(&mut h);
            shift.map(|(target, replace)| (mapping[target as usize], replace)).hash(&mut h);
            reduces.hash(&mut h);
            accept.hash(&mut h);
        }
        Action::Accept => 3u8.hash(&mut h),
        Action::Skip => 7u8.hash(&mut h),
        Action::StackShifts(shifts) => return remapped_stack_shifts_hash(shifts, mapping),
        Action::ReplaceShifts(_) | Action::GuardedStackShifts(_) => {
            return unordered_entry_hash(&remap_action_targets(action, mapping));
        }
    }
    h.finish()
}

#[inline]
fn actions_equal_after_remap(left: &Action, right: &Action, mapping: &[u32]) -> bool {
    match (left, right) {
        (Action::Shift(lt, lr), Action::Shift(rt, rr)) => lr == rr && mapping[*lt as usize] == mapping[*rt as usize],
        (Action::Reduce(ln, ll), Action::Reduce(rn, rl)) => ln == rn && ll == rl,
        (Action::Split { shift: ls, reduces: lrs, accept: la }, Action::Split { shift: rs, reduces: rrs, accept: ra }) => {
            ls.map(|(target, replace)| (mapping[target as usize], replace)) == rs.map(|(target, replace)| (mapping[target as usize], replace)) && lrs == rrs && la == ra
        }
        (Action::Accept, Action::Accept) => true,
        (Action::Skip, Action::Skip) => true,
        (Action::StackShifts(left), Action::StackShifts(right)) => {
            remapped_stack_shifts_equal(left, right, mapping)
        }
        (Action::ReplaceShifts(_), Action::ReplaceShifts(_))
        | (Action::GuardedStackShifts(_), Action::GuardedStackShifts(_)) => {
            remap_action_targets(left, mapping) == remap_action_targets(right, mapping)
        }
        _ => false,
    }
}

fn row_fingerprint(
    action_row: &ActionRow,
    goto_row: &GotoRow,
    advance_row: Option<&BitSet>,
) -> u64 {
    let (action_sum, action_xor) = unordered_row_hash(
        action_row.iter().map(|(terminal, action)| (terminal, action)),
        action_row.len(),
    );
    let (goto_sum, goto_xor) = unordered_row_hash(
        goto_row.iter().map(|(nonterminal, target)| (*nonterminal, target)),
        goto_row.len(),
    );
    let mut hasher = FxHasher::default();
    action_row.len().hash(&mut hasher);
    action_sum.hash(&mut hasher);
    action_xor.hash(&mut hasher);
    goto_row.len().hash(&mut hasher);
    goto_sum.hash(&mut hasher);
    goto_xor.hash(&mut hasher);
    if let Some(advance_row) = advance_row {
        advance_row.hash(&mut hasher);
    }
    hasher.finish()
}

fn remapped_row_fingerprint(
    action_row: &ActionRow,
    goto_row: &GotoRow,
    advance_row: Option<&BitSet>,
    state_to_group: &[u32],
) -> u64 {
    let (action_sum, action_xor) = unordered_row_hash(
        action_row.iter().map(|(terminal, action)| {
            (terminal, remapped_action_hash(action, state_to_group))
        }),
        action_row.len(),
    );
    let (goto_sum, goto_xor) = unordered_row_hash(
        goto_row.iter().map(|(nonterminal, &(target, replace))| {
            (*nonterminal, (state_to_group[target as usize], replace))
        }),
        goto_row.len(),
    );
    let mut hasher = FxHasher::default();
    action_row.len().hash(&mut hasher);
    action_sum.hash(&mut hasher);
    action_xor.hash(&mut hasher);
    goto_row.len().hash(&mut hasher);
    goto_sum.hash(&mut hasher);
    goto_xor.hash(&mut hasher);
    if let Some(advance_row) = advance_row {
        advance_row.hash(&mut hasher);
    }
    hasher.finish()
}

fn rows_equal(
    action_a: &ActionRow,
    goto_a: &GotoRow,
    advance_a: Option<&BitSet>,
    action_b: &ActionRow,
    goto_b: &GotoRow,
    advance_b: Option<&BitSet>,
) -> bool {
    advance_a == advance_b
        && action_a.len() == action_b.len()
        && goto_a.len() == goto_b.len()
        && action_a
            .iter()
            .all(|(terminal, action)| action_b.get(&terminal) == Some(action))
        && goto_a
            .iter()
            .all(|(&nonterminal, target)| goto_b.get(&nonterminal) == Some(target))
}

fn rows_equal_after_remap(
    action_a: &ActionRow,
    goto_a: &GotoRow,
    advance_a: Option<&BitSet>,
    action_b: &ActionRow,
    goto_b: &GotoRow,
    advance_b: Option<&BitSet>,
    state_to_group: &[u32],
) -> bool {
    advance_a == advance_b
        && action_a.len() == action_b.len()
        && goto_a.len() == goto_b.len()
        && action_a.iter().all(|(terminal, action)| {
            action_b
                .get(&terminal)
                .is_some_and(|other| actions_equal_after_remap(action, other, state_to_group))
        })
        && goto_a.iter().all(|(&nonterminal, &(target, replace))| {
            goto_b.get(&nonterminal).is_some_and(|&(other_target, other_replace)| {
                replace == other_replace
                    && state_to_group[target as usize] == state_to_group[other_target as usize]
            })
        })
}

fn push_reachable_state(state: u32, reachable: &mut [bool], stack: &mut Vec<u32>) {
    let Some(slot) = reachable.get_mut(state as usize) else {
        return;
    };
    if !*slot {
        *slot = true;
        stack.push(state);
    }
}

fn collect_action_targets(action: &Action, out: &mut Vec<u32>) {
    match action {
        Action::Shift(target, _) => out.push(*target),
        Action::ReplaceShifts(targets) => out.extend(targets.iter().copied()),
        Action::StackShifts(shifts) => {
            for shift in shifts {
                out.extend(shift.pushes.iter().copied());
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                for guard in &shift.guards {
                    out.extend(guard.states.iter().copied());
                }
                out.extend(shift.pushes.iter().copied());
            }
        }
        Action::Split { shift, .. } => {
            if let Some((target, _)) = shift {
                out.push(*target);
            }
        }
        Action::Reduce(_, _) | Action::Accept | Action::Skip => {}
    }
}

fn push_action_targets(action: &Action, reachable: &mut [bool], stack: &mut Vec<u32>) {
    match action {
        Action::Shift(target, _) => push_reachable_state(*target, reachable, stack),
        Action::ReplaceShifts(targets) => {
            for &state in targets.iter() {
                push_reachable_state(state, reachable, stack);
            }
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                for &state in &shift.pushes {
                    push_reachable_state(state, reachable, stack);
                }
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                for guard in &shift.guards {
                    for &state in &guard.states {
                        push_reachable_state(state, reachable, stack);
                    }
                }
                for &state in &shift.pushes {
                    push_reachable_state(state, reachable, stack);
                }
            }
        }
        Action::Reduce(_, _) | Action::Accept | Action::Skip => {}
        Action::Split { shift, .. } => {
            if let Some((target, _)) = shift {
                push_reachable_state(*target, reachable, stack);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CellUpdate {
    Set(Action),
    Remove,
}

/// Only actions that contain at least one non-accepting reduction can change
/// under stack-effect or unit-reduction inlining. In particular, a plain shift
/// has no reduction to resolve and the recursive unit inliner must return it
/// unchanged, so visiting it is pure overhead.
fn action_has_inlinable_reductions(action: &Action) -> bool {
    match action {
        Action::Reduce(_, _) => true,
        Action::Split {
            reduces,
            accept: false,
            ..
        } => !reduces.is_empty(),
        _ => false,
    }
}

/// The unit-reduction pass adds work beyond stack-effect inlining only when an
/// action has both a shift and a unit reduction. A reduction-only action is
/// either handled by the stack-effect pass or cannot gain a shift by this
/// phase; this matches the old dispatch exactly.
#[derive(Clone, Copy)]
enum StackInlineCandidateKind {
    ReduceZero,
    ReduceOne,
    ReduceMany,
    Split,
}

impl StackInlineCandidateKind {
    const COUNT: usize = 4;

    fn index(self) -> usize {
        match self {
            Self::ReduceZero => 0,
            Self::ReduceOne => 1,
            Self::ReduceMany => 2,
            Self::Split => 3,
        }
    }
}

fn stack_inline_candidate_kind(action: &Action) -> StackInlineCandidateKind {
    match action {
        Action::Reduce(_, 0) => StackInlineCandidateKind::ReduceZero,
        Action::Reduce(_, 1) => StackInlineCandidateKind::ReduceOne,
        Action::Reduce(_, _) => StackInlineCandidateKind::ReduceMany,
        Action::Split { .. } => StackInlineCandidateKind::Split,
        _ => unreachable!("only stack-inlining candidates are classified"),
    }
}

fn action_needs_unit_reduction_inlining(action: &Action) -> bool {
    matches!(
        action,
        Action::Split {
            shift: Some(_),
            reduces,
            accept: false,
        } if !reduces.is_empty()
    )
}

type UnitInlineCell = (u32, TerminalID);

#[derive(Default)]
struct UnitInlineDependencies {
    reverse: FxHashMap<UnitInlineCell, Vec<UnitInlineCell>>,
}

impl UnitInlineDependencies {
    fn record_read(&mut self, owner: UnitInlineCell, read: UnitInlineCell) {
        if owner != read {
            self.reverse.entry(read).or_default().push(owner);
        }
    }

    fn record_reads(&mut self, reads: impl IntoIterator<Item = (UnitInlineCell, UnitInlineCell)>) {
        for (owner, read) in reads {
            self.record_read(owner, read);
        }
    }

    fn worklist_for_changes(&self, changed: impl IntoIterator<Item = UnitInlineCell>) -> Vec<UnitInlineCell> {
        let mut queued = FxHashSet::default();
        for changed_cell in changed {
            queued.insert(changed_cell);
            if let Some(dependents) = self.reverse.get(&changed_cell) {
                queued.extend(dependents.iter().copied());
            }
        }
        let mut worklist = queued.into_iter().collect::<Vec<_>>();
        worklist.sort_unstable();
        worklist
    }
}

/// A cheap sufficient test for the common rejected stack-effect shape:
/// reducing one symbol and then taking only non-forwarded replace shifts. The
/// general stack-effect evaluator would classify this as origin-dependent with
/// one pushed state and deliberately leave the reduce untouched, so skipping
/// it is semantically exact. Every action row inspected here is recorded for
/// fixed-point invalidation because a later rewrite can make the shortcut
/// inapplicable.
fn is_trivial_origin_dependent_one_push(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
) -> bool {
    let Action::Reduce(nonterminal, 1) = action else {
        return false;
    };
    let owner = (state, tid);
    let mut saw_effect = false;
    for &predecessor in &predecessors[state as usize] {
        let Some((next_state, _)) = table.goto[predecessor as usize].get(nonterminal).copied() else {
            continue;
        };
        let Some(next_action) = table.action[next_state as usize].get(&tid) else {
            continue;
        };
        dependencies.push((owner, (next_state, tid)));
        let is_replace_shift = match next_action {
            Action::Shift(_, replace) => {
                *replace && !table.forwarded_shifts.contains(&(next_state, tid))
            }
            Action::Split {
                shift: Some((_, replace)),
                reduces,
                accept: false,
            } => {
                reduces.is_empty()
                    && *replace
                    && !table.forwarded_shifts.contains(&(next_state, tid))
            }
            _ => false,
        };
        if !is_replace_shift {
            return false;
        }
        saw_effect = true;
    }
    saw_effect
}

fn runtime_predecessor_transition(action: &Action) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, replace) => Some((*target, *replace)),
        Action::Split { shift: Some((target, replace)), .. } => Some((*target, *replace)),
        _ => None,
    }
}

fn build_runtime_state_predecessors(
    table: &GLRTable,
    original_num_states: u32,
    constituent_sets: &[StateSubset],
) -> RuntimePredecessors {
    let mut predecessors = vec![PredecessorSet::new(); table.num_states as usize];

    // Collect direct non-replace edges first, then normalize each compact row
    // once. This preserves the sorted set iteration used by the old BTreeSet
    // implementation without paying a tree insertion for every edge.
    for src in 0..table.num_states as usize {
        for action in table.action[src].values() {
            match action {
                Action::Shift(dst, false) => {
                    predecessors[*dst as usize].extend_from_slice(&constituent_sets[src]);
                }
                Action::Split { shift: Some((dst, false)), .. } => {
                    predecessors[*dst as usize].extend_from_slice(&constituent_sets[src]);
                }
                _ => {}
            }
        }
        for &(dst, replace) in table.goto[src].values() {
            if !replace {
                predecessors[dst as usize].extend_from_slice(&constituent_sets[src]);
            }
        }
    }
    for predecessors_for_state in &mut predecessors {
        predecessors_for_state.sort_unstable();
        predecessors_for_state.dedup();
    }

    // Replacement edges inherit the predecessor set of their source.  Rows
    // stay sorted between passes, so each propagation has deterministic set
    // semantics and the fixed point matches the old BTreeSet version exactly.
    let mut changed = true;
    while changed {
        changed = false;
        for src in 0..original_num_states as usize {
            let src_preds = predecessors[src].clone();
            for action in table.action[src].values() {
                let destination = match action {
                    Action::Shift(dst, true) => Some(*dst),
                    Action::Split { shift: Some((dst, true)), .. } => Some(*dst),
                    _ => None,
                };
                let Some(dst) = destination else {
                    continue;
                };
                let destination_preds = &mut predecessors[dst as usize];
                let before = destination_preds.len();
                destination_preds.extend_from_slice(&src_preds);
                destination_preds.sort_unstable();
                destination_preds.dedup();
                changed |= destination_preds.len() != before;
            }
            for &(dst, replace) in table.goto[src].values() {
                if !replace {
                    continue;
                }
                let destination_preds = &mut predecessors[dst as usize];
                let before = destination_preds.len();
                destination_preds.extend_from_slice(&src_preds);
                destination_preds.sort_unstable();
                destination_preds.dedup();
                changed |= destination_preds.len() != before;
            }
        }
    }

    predecessors
}

fn control_elimination_failure(
    budget: &UnitInlineBudget,
    state: u32,
    terminal: TerminalID,
    message: &str,
) -> String {
    let reason = budget.report().reason.unwrap_or("semantic traversal");
    format!(
        "{message}: state={state} terminal={terminal} reason={reason} stack_effect_visits={}",
        budget.stack_effect_visits(),
    )
}

/// True when a control-elimination error reports convergence-budget
/// exhaustion (`reason=stack_effect_visits` / `reason=elapsed_ms`) rather
/// than a semantic traversal failure. Budget exhaustion is a *decline*:
/// the table stays controlled and the caller falls back to dynamic
/// boundary shards. Any other elimination error is a hard failure.
pub fn control_elimination_budget_exhausted(error: &str) -> bool {
    error.contains("reason=stack_effect_visits") || error.contains("reason=elapsed_ms")
}

fn for_each_action_stack_rewrite(
    action: &Action,
    mut visit: impl FnMut(u32, &[u32]),
) {
    match action {
        Action::Shift(target, replace) => {
            visit(u32::from(*replace), std::slice::from_ref(target));
        }
        Action::ReplaceShifts(targets) => {
            for target in targets.iter() {
                visit(1, std::slice::from_ref(target));
            }
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                visit(shift.pop, &shift.pushes);
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                visit(shift.pop, &shift.pushes);
            }
        }
        Action::Split { shift, .. } => {
            if let Some((target, replace)) = shift {
                visit(u32::from(*replace), std::slice::from_ref(target));
            }
        }
        Action::Skip | Action::Reduce(..) | Action::Accept => {}
    }
}

#[cfg(test)]
fn action_stack_rewrites(action: &Action) -> Vec<(u32, Vec<u32>)> {
    let mut rewrites = Vec::new();
    for_each_action_stack_rewrite(action, |pop, pushes| {
        rewrites.push((pop, pushes.to_vec()));
    });
    rewrites
}

fn predecessor_states_at_depth(
    predecessors: &[PredecessorSet],
    origin: u32,
    depth: u32,
) -> PredecessorSet {
    let mut states = smallvec![origin];
    for _ in 0..depth {
        let mut next = PredecessorSet::new();
        for state in states {
            if let Some(preds) = predecessors.get(state as usize) {
                next.extend_from_slice(preds);
            }
        }
        next.sort_unstable();
        next.dedup();
        if next.is_empty() {
            return next;
        }
        states = next;
    }
    states
}

struct DensePredecessorMatrix {
    bit_len: usize,
    words_per_row: usize,
    words: Vec<u64>,
}

impl DensePredecessorMatrix {
    fn new(rows: usize, bit_len: usize) -> Self {
        let words_per_row = bit_len.div_ceil(64);
        Self {
            bit_len,
            words_per_row,
            words: vec![0; rows.saturating_mul(words_per_row)],
        }
    }

    fn set(&mut self, row: u32, bit: u32) -> bool {
        let index = row as usize * self.words_per_row + bit as usize / 64;
        let mask = 1u64 << (bit % 64);
        let was_set = self.words[index] & mask != 0;
        self.words[index] |= mask;
        !was_set
    }

    fn row_words(&self, row: u32) -> &[u64] {
        let start = row as usize * self.words_per_row;
        &self.words[start..start + self.words_per_row]
    }

    fn union_row_into(&mut self, source: u32, destination: u32) -> bool {
        if source == destination {
            return false;
        }
        let source_start = source as usize * self.words_per_row;
        let destination_start = destination as usize * self.words_per_row;
        let mut changed = false;
        for offset in 0..self.words_per_row {
            let source_word = self.words[source_start + offset];
            let destination_word = &mut self.words[destination_start + offset];
            let merged = *destination_word | source_word;
            changed |= merged != *destination_word;
            *destination_word = merged;
        }
        changed
    }

    fn union_words_into(&mut self, destination: u32, source: &[u64]) -> bool {
        debug_assert_eq!(source.len(), self.words_per_row);
        let destination_start = destination as usize * self.words_per_row;
        let mut changed = false;
        for (offset, &source_word) in source.iter().enumerate() {
            let destination_word = &mut self.words[destination_start + offset];
            let merged = *destination_word | source_word;
            changed |= merged != *destination_word;
            *destination_word = merged;
        }
        changed
    }

}

fn for_each_set_bit(words: &[u64], bit_len: usize, mut visit: impl FnMut(u32)) {
    for (word_index, &word) in words.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let state = word_index * 64 + bit;
            if state < bit_len {
                visit(state as u32);
            }
            remaining &= remaining - 1;
        }
    }
}

fn condense_pop_one_predecessor_graph(
    adjacency: &[PredecessorSet],
) -> (Vec<u32>, Vec<PredecessorSet>, Vec<u32>) {
    let state_count = adjacency.len();
    let mut visited = vec![false; state_count];
    let mut finish_order = Vec::with_capacity(state_count);

    for start in 0..state_count {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((state, next_index)) = stack.last_mut() {
            if *next_index < adjacency[*state].len() {
                let destination = adjacency[*state][*next_index] as usize;
                *next_index += 1;
                if !visited[destination] {
                    visited[destination] = true;
                    stack.push((destination, 0));
                }
            } else {
                finish_order.push(*state);
                stack.pop();
            }
        }
    }

    let mut reverse = vec![Vec::<u32>::new(); state_count];
    for (source, destinations) in adjacency.iter().enumerate() {
        for &destination in destinations {
            reverse[destination as usize].push(source as u32);
        }
    }

    visited.fill(false);
    let mut state_to_component = vec![0u32; state_count];
    let mut component_count = 0u32;
    for &start in finish_order.iter().rev() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start as u32];
        while let Some(state) = stack.pop() {
            state_to_component[state as usize] = component_count;
            for &predecessor in &reverse[state as usize] {
                if !visited[predecessor as usize] {
                    visited[predecessor as usize] = true;
                    stack.push(predecessor);
                }
            }
        }
        component_count += 1;
    }

    let mut component_edges = vec![PredecessorSet::new(); component_count as usize];
    for (source, destinations) in adjacency.iter().enumerate() {
        let source_component = state_to_component[source];
        for &destination in destinations {
            let destination_component = state_to_component[destination as usize];
            if source_component != destination_component {
                component_edges[source_component as usize].push(destination_component);
            }
        }
    }
    for edges in &mut component_edges {
        edges.sort_unstable();
        edges.dedup();
    }

    let mut indegree = vec![0usize; component_count as usize];
    for edges in &component_edges {
        for &destination in edges {
            indegree[destination as usize] += 1;
        }
    }
    let mut queue = VecDeque::new();
    for (component, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(component as u32);
        }
    }
    let mut topological_order = Vec::with_capacity(component_count as usize);
    while let Some(component) = queue.pop_front() {
        topological_order.push(component);
        for &destination in &component_edges[component as usize] {
            indegree[destination as usize] -= 1;
            if indegree[destination as usize] == 0 {
                queue.push_back(destination);
            }
        }
    }
    debug_assert_eq!(topological_order.len(), component_count as usize);

    (state_to_component, component_edges, topological_order)
}

fn propagate_component_predecessors(
    matrix: &mut DensePredecessorMatrix,
    component_edges: &[PredecessorSet],
    topological_order: &[u32],
    dirty_components: &mut [bool],
) -> usize {
    let mut changed_edges = 0usize;
    for &source in topological_order {
        if !dirty_components[source as usize] {
            continue;
        }
        dirty_components[source as usize] = false;
        for &destination in &component_edges[source as usize] {
            if matrix.union_row_into(source, destination) {
                changed_edges += 1;
                dirty_components[destination as usize] = true;
            }
        }
    }
    changed_edges
}

fn dense_predecessor_states_at_depth(
    matrix: &DensePredecessorMatrix,
    state_to_component: &[u32],
    origin_component: u32,
    depth: u32,
    component_marks: &mut [u32],
    mark_generation: &mut u32,
) -> Vec<u64> {
    debug_assert!(depth >= 1);
    let mut current = matrix.row_words(origin_component).to_vec();
    let mut next = vec![0u64; matrix.words_per_row];
    let mut components = Vec::<u32>::new();

    for _ in 1..depth {
        next.fill(0);
        components.clear();
        *mark_generation = mark_generation.wrapping_add(1);
        if *mark_generation == 0 {
            component_marks.fill(0);
            *mark_generation = 1;
        }
        let mark = *mark_generation;
        for_each_set_bit(&current, matrix.bit_len, |state| {
            let component = state_to_component[state as usize];
            if component_marks[component as usize] != mark {
                component_marks[component as usize] = mark;
                components.push(component);
            }
        });
        for component in components.iter().copied() {
            for (destination, source) in next.iter_mut().zip(matrix.row_words(component)) {
                *destination |= *source;
            }
        }
        std::mem::swap(&mut current, &mut next);
        if current.iter().all(|word| *word == 0) {
            break;
        }
    }

    current
}


/// Demand-driven exact predecessor support for control elimination.
///
/// Control closure starts only in rows containing a control action. Rather than
/// solve and materialize the predecessor relation for every parser state, this
/// solver starts from those origins and monotonically discovers exactly the
/// rows needed by copy and deep-pop equations encountered on their predecessor
/// chains. Empty rows outside that demand closure are never observed by control
/// elimination.
fn build_control_elimination_predecessors_demand(
    table: &GLRTable,
    origins: &[u32],
) -> Result<RuntimePredecessors, String> {
    let started_at = std::time::Instant::now();
    let state_count = table.num_states as usize;
    let mut direct = vec![PredecessorSet::new(); state_count];
    let mut copy_sources = vec![PredecessorSet::new(); state_count];
    let mut deep_sources = vec![Vec::<(u32, u32)>::new(); state_count];
    let mut rewrite_count = 0usize;

    for source in 0..table.num_states {
        let mut record_rewrite = |pop: u32, pushes: &[u32]| {
            rewrite_count += 1;
            if pushes.is_empty() {
                return;
            }
            for pair in pushes.windows(2) {
                direct[pair[1] as usize].push(pair[0]);
            }
            let destination = pushes[0] as usize;
            match pop {
                0 => direct[destination].push(source),
                1 => copy_sources[destination].push(source),
                _ => deep_sources[destination].push((source, pop)),
            }
        };
        for action in table.action[source as usize].values() {
            for_each_action_stack_rewrite(action, &mut record_rewrite);
        }
        for &(target, replace) in table.goto[source as usize].values() {
            record_rewrite(u32::from(replace), std::slice::from_ref(&target));
        }
    }
    for row in &mut direct {
        row.sort_unstable();
        row.dedup();
    }
    for row in &mut copy_sources {
        row.sort_unstable();
        row.dedup();
    }
    for row in &mut deep_sources {
        row.sort_unstable();
        row.dedup();
    }
    let extraction_ms = started_at.elapsed().as_secs_f64() * 1000.0;

    let mut rows = vec![None::<BitSet>; state_count];
    let mut needed = vec![false; state_count];
    let mut dependency_queue = VecDeque::<u32>::new();
    let mut needed_states = Vec::<u32>::new();
    let mark_needed = |state: u32,
                       rows: &mut [Option<BitSet>],
                       needed: &mut [bool],
                       dependency_queue: &mut VecDeque<u32>,
                       needed_states: &mut Vec<u32>| {
        let index = state as usize;
        if index >= state_count || needed[index] {
            return false;
        }
        needed[index] = true;
        let mut row = BitSet::new(state_count);
        for &predecessor in &direct[index] {
            row.set(predecessor as usize);
        }
        rows[index] = Some(row);
        dependency_queue.push_back(state);
        needed_states.push(state);
        true
    };
    for &origin in origins {
        mark_needed(
            origin,
            &mut rows,
            &mut needed,
            &mut dependency_queue,
            &mut needed_states,
        );
    }

    fn union_optional_row(rows: &mut [Option<BitSet>], source: u32, destination: u32) -> bool {
        if source == destination {
            return false;
        }
        let source = source as usize;
        let destination = destination as usize;
        if source < destination {
            let (left, right) = rows.split_at_mut(destination);
            let Some(source_row) = left[source].as_ref() else {
                return false;
            };
            right[0]
                .as_mut()
                .expect("needed destination row exists")
                .union_with_changed(source_row)
        } else {
            let (left, right) = rows.split_at_mut(source);
            let Some(source_row) = right[0].as_ref() else {
                return false;
            };
            left[destination]
                .as_mut()
                .expect("needed destination row exists")
                .union_with_changed(source_row)
        }
    }

    let pass_limit = state_count.saturating_mul(state_count).saturating_add(2);
    let mut rounds = 0usize;
    let mut dynamic_dependencies = 0usize;
    let mut changed_unions = 0usize;
    for _ in 0..pass_limit {
        rounds += 1;
        while let Some(destination) = dependency_queue.pop_front() {
            for &source in &copy_sources[destination as usize] {
                dynamic_dependencies += usize::from(mark_needed(
                    source,
                    &mut rows,
                    &mut needed,
                    &mut dependency_queue,
                    &mut needed_states,
                ));
            }
            for &(source, _) in &deep_sources[destination as usize] {
                dynamic_dependencies += usize::from(mark_needed(
                    source,
                    &mut rows,
                    &mut needed,
                    &mut dependency_queue,
                    &mut needed_states,
                ));
            }
        }

        let mut changed = false;
        let round_states = needed_states.clone();
        for destination in round_states {
            for &source in &copy_sources[destination as usize] {
                if union_optional_row(&mut rows, source, destination) {
                    changed = true;
                    changed_unions += 1;
                }
            }
            for &(source, pop) in &deep_sources[destination as usize] {
                let Some(mut current) = rows[source as usize].clone() else {
                    continue;
                };
                for _ in 1..pop {
                    let current_states = current.iter_ones().map(|state| state as u32).collect::<Vec<_>>();
                    let mut next = BitSet::new(state_count);
                    for state in current_states {
                        dynamic_dependencies += usize::from(mark_needed(
                            state,
                            &mut rows,
                            &mut needed,
                            &mut dependency_queue,
                            &mut needed_states,
                        ));
                        if let Some(row) = rows[state as usize].as_ref() {
                            next.union_with(row);
                        }
                    }
                    current = next;
                    if current.is_zero() {
                        break;
                    }
                }
                let destination_row = rows[destination as usize]
                    .as_mut()
                    .expect("needed destination row exists");
                if destination_row.union_with_changed(&current) {
                    changed = true;
                    changed_unions += 1;
                }
            }
        }

        if !changed && dependency_queue.is_empty() {
            // Control traversal may pop deeper than any one predecessor
            // equation used while solving the origin rows. Close the demand
            // set under the completed predecessor relation so every state that
            // can appear below a control origin has its exact row available.
            let closure_states = needed_states.clone();
            let mut closure_added = false;
            for state in closure_states {
                let predecessors = rows[state as usize]
                    .as_ref()
                    .expect("needed predecessor row exists")
                    .iter_ones()
                    .map(|predecessor| predecessor as u32)
                    .collect::<Vec<_>>();
                for predecessor in predecessors {
                    if mark_needed(
                        predecessor,
                        &mut rows,
                        &mut needed,
                        &mut dependency_queue,
                        &mut needed_states,
                    ) {
                        dynamic_dependencies += 1;
                        closure_added = true;
                    }
                }
            }
            if closure_added {
                continue;
            }
            let materialize_started_at = std::time::Instant::now();
            let mut result = vec![PredecessorSet::new(); state_count];
            let mut predecessor_edges = 0usize;
            let mut max_predecessors = 0usize;
            for &state in &needed_states {
                let row = rows[state as usize]
                    .as_ref()
                    .expect("needed predecessor row exists");
                for predecessor in row.iter_ones() {
                    result[state as usize].push(predecessor as u32);
                }
                predecessor_edges += result[state as usize].len();
                max_predecessors = max_predecessors.max(result[state as usize].len());
            }
            let materialize_ms = materialize_started_at.elapsed().as_secs_f64() * 1000.0;

            if std::env::var_os("GLRMASK_VALIDATE_CONTROL_PREDECESSOR_DEMAND").is_some() {
                let reference = build_control_elimination_predecessors(table)?;
                for &state in &needed_states {
                    assert_eq!(
                        result[state as usize],
                        reference[state as usize],
                        "demand predecessor row differs for state {state}",
                    );
                }
                eprintln!(
                    "[glrmask/validate][control_predecessor_demand] origins={} needed_states={} exact=true",
                    origins.len(),
                    needed_states.len(),
                );
            }
            if std::env::var_os("GLRMASK_PROFILE_CONTROL_ELIMINATION").is_some() {
                eprintln!(
                    "[glrmask/profile][control_predecessors_demand] states={} origins={} needed_states={} rewrites={} rounds={} dynamic_dependencies={} changed_unions={} predecessor_edges={} max_predecessors={} extraction_ms={extraction_ms:.3} materialize_ms={materialize_ms:.3} total_ms={:.3}",
                    state_count,
                    origins.len(),
                    needed_states.len(),
                    rewrite_count,
                    rounds,
                    dynamic_dependencies,
                    changed_unions,
                    predecessor_edges,
                    max_predecessors,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return Ok(result);
        }
    }

    Err(format!(
        "demand control-elimination predecessor propagation did not converge: states={} origins={}",
        state_count,
        origins.len(),
    ))
}

/// Conservative-exact predecessor support for arbitrary stack rewrites.
/// Extra predecessor candidates are harmless because generated guarded effects
/// test the concrete stack state before applying; missing candidates would be
/// unsound. Iterate stack-shift propagation to a fixed point so copied-child
/// return actions contribute their caller context.
fn build_control_elimination_predecessors(
    table: &GLRTable,
) -> Result<RuntimePredecessors, String> {
    let started_at = std::time::Instant::now();
    let mut predecessors = vec![PredecessorSet::new(); table.num_states as usize];
    let mut pop_one_destinations = vec![PredecessorSet::new(); table.num_states as usize];
    let mut deep_rules = Vec::<(u32, u32, u32)>::new();
    let mut rewrite_count = 0usize;
    let mut direct_edge_count = 0usize;

    // Extract the monotone predecessor equations once. The previous
    // implementation rebuilt every action rewrite, cloned the complete
    // predecessor relation, and rescanned the whole table on every fixed-point
    // round. Realistic composed tables have tens of thousands of states but
    // only a small number of rounds and stack-rewrite shapes, so that repeated
    // table work dominated control elimination by orders of magnitude.
    //
    // Each delayed rule `(source, pop, destination)` means:
    //
    //     pred[destination] += pred^pop[source]
    //
    // Direct stack adjacency inside a pushed sequence and pop-zero rules are
    // seeded immediately. The equations are monotone over finite state sets;
    // asynchronous iteration below therefore reaches the same least fixed
    // point as the old synchronous snapshot iteration.
    for source in 0..table.num_states {
        let mut record_rewrite = |pop: u32, pushes: &[u32]| {
            rewrite_count += 1;
            if pushes.is_empty() {
                return;
            }
            for pair in pushes.windows(2) {
                predecessors[pair[1] as usize].push(pair[0]);
                direct_edge_count += 1;
            }
            let destination = pushes[0];
            if pop == 0 {
                predecessors[destination as usize].push(source);
                direct_edge_count += 1;
            } else if pop == 1 {
                pop_one_destinations[source as usize].push(destination);
            } else {
                deep_rules.push((source, pop, destination));
            }
        };

        for action in table.action[source as usize].values() {
            for_each_action_stack_rewrite(action, &mut record_rewrite);
        }
        for &(target, replace) in table.goto[source as usize].values() {
            record_rewrite(u32::from(replace), std::slice::from_ref(&target));
        }
    }
    let extraction_ms = started_at.elapsed().as_secs_f64() * 1000.0;

    for row in &mut predecessors {
        row.sort_unstable();
        row.dedup();
    }
    for row in &mut pop_one_destinations {
        row.sort_unstable();
        row.dedup();
    }
    deep_rules.sort_unstable();
    deep_rules.dedup();
    let normalized_ms = started_at.elapsed().as_secs_f64() * 1000.0;

    // `pop == 1` equations are pure row-copy edges. Collapse their SCCs, then
    // propagate predecessor bitsets once over the condensation DAG. This turns
    // hundreds of thousands of individual sorted-vector insertions into dense
    // word-wise unions while preserving the exact least fixed point.
    let scc_started_at = std::time::Instant::now();
    let (state_to_component, component_edges, topological_order) =
        condense_pop_one_predecessor_graph(&pop_one_destinations);
    let component_count = topological_order.len();
    let scc_ms = scc_started_at.elapsed().as_secs_f64() * 1000.0;

    let matrix_started_at = std::time::Instant::now();
    let mut component_predecessors =
        DensePredecessorMatrix::new(component_count, table.num_states as usize);
    let mut dirty_components = vec![false; component_count];
    for (state, row) in predecessors.iter().enumerate() {
        let component = state_to_component[state];
        for &predecessor in row {
            if component_predecessors.set(component, predecessor) {
                dirty_components[component as usize] = true;
            }
        }
    }
    let initial_changed_component_edges = propagate_component_predecessors(
        &mut component_predecessors,
        &component_edges,
        &topological_order,
        &mut dirty_components,
    );
    let initial_propagation_ms = matrix_started_at.elapsed().as_secs_f64() * 1000.0;

    for (source, _, destination) in &mut deep_rules {
        *source = state_to_component[*source as usize];
        *destination = state_to_component[*destination as usize];
    }
    deep_rules.sort_unstable();
    deep_rules.dedup();
    let pass_limit = (table.num_states as usize)
        .saturating_mul(table.num_states as usize)
        .saturating_add(2);

    let mut deep_rounds = 0usize;
    let mut changed_deep_rules_total = 0usize;
    let mut changed_component_edges_total = initial_changed_component_edges;
    let mut component_marks = vec![0u32; component_count];
    let mut mark_generation = 0u32;
    for _ in 0..pass_limit {
        deep_rounds += 1;
        let mut changed_deep_rules = 0usize;
        let mut depth_cache = FxHashMap::<(u32, u32), Vec<u64>>::default();
        for &(source_component, pop, destination_component) in &deep_rules {
            let key = (source_component, pop);
            if !depth_cache.contains_key(&key) {
                let bases = dense_predecessor_states_at_depth(
                    &component_predecessors,
                    &state_to_component,
                    source_component,
                    pop,
                    &mut component_marks,
                    &mut mark_generation,
                );
                depth_cache.insert(key, bases);
            }
            let bases = depth_cache.get(&key).expect("depth cache was populated");
            if component_predecessors.union_words_into(destination_component, bases) {
                changed_deep_rules += 1;
                dirty_components[destination_component as usize] = true;
            }
        }

        if changed_deep_rules == 0 {
            let materialize_started_at = std::time::Instant::now();
            let mut result = Vec::with_capacity(table.num_states as usize);
            let mut predecessor_edges = 0usize;
            let mut max_predecessors = 0usize;
            for state in 0..table.num_states as usize {
                let mut row = PredecessorSet::new();
                for_each_set_bit(
                    component_predecessors.row_words(state_to_component[state]),
                    table.num_states as usize,
                    |predecessor| row.push(predecessor),
                );
                predecessor_edges += row.len();
                max_predecessors = max_predecessors.max(row.len());
                result.push(row);
            }
            let materialize_ms = materialize_started_at.elapsed().as_secs_f64() * 1000.0;
            if std::env::var_os("GLRMASK_PROFILE_CONTROL_ELIMINATION").is_some() {
                let pop_one_rules = pop_one_destinations.iter().map(|row| row.len()).sum::<usize>();
                let component_edge_count = component_edges.iter().map(|row| row.len()).sum::<usize>();
                eprintln!(
                    "[glrmask/profile][control_predecessors] states={} components={} rewrites={} direct_edges={} pop_one_rules={} component_edges={} deep_rules={} deep_rounds={} changed_component_edges={} changed_deep_rules={} predecessor_edges={} max_predecessors={} matrix_mib={:.3} extraction_ms={extraction_ms:.3} normalize_ms={:.3} scc_ms={scc_ms:.3} initial_propagation_ms={initial_propagation_ms:.3} materialize_ms={materialize_ms:.3} total_ms={:.3}",
                    table.num_states,
                    component_count,
                    rewrite_count,
                    direct_edge_count,
                    pop_one_rules,
                    component_edge_count,
                    deep_rules.len(),
                    deep_rounds,
                    changed_component_edges_total,
                    changed_deep_rules_total,
                    predecessor_edges,
                    max_predecessors,
                    component_predecessors.words.len() as f64 * 8.0 / (1024.0 * 1024.0),
                    normalized_ms - extraction_ms,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return Ok(result);
        }
        changed_deep_rules_total += changed_deep_rules;
        changed_component_edges_total += propagate_component_predecessors(
            &mut component_predecessors,
            &component_edges,
            &topological_order,
            &mut dirty_components,
        );
    }

    Err(format!(
        "control-elimination predecessor propagation did not converge: states={}",
        table.num_states,
    ))
}

#[cfg(test)]
fn build_control_elimination_predecessors_synchronous_reference(
    table: &GLRTable,
) -> Result<RuntimePredecessors, String> {
    let mut predecessors = vec![PredecessorSet::new(); table.num_states as usize];
    let pass_limit = (table.num_states as usize)
        .saturating_mul(table.num_states as usize)
        .saturating_add(2);

    for _ in 0..pass_limit {
        let snapshot = predecessors.clone();
        let mut changed = false;

        for source in 0..table.num_states {
            let mut rewrites = Vec::<(u32, Vec<u32>)>::new();
            for action in table.action[source as usize].values() {
                rewrites.extend(action_stack_rewrites(action));
            }
            rewrites.extend(
                table.goto[source as usize]
                    .values()
                    .map(|&(target, replace)| (u32::from(replace), vec![target])),
            );

            for (pop, pushes) in rewrites {
                if pushes.is_empty() {
                    continue;
                }
                for pair in pushes.windows(2) {
                    let destination = pair[1] as usize;
                    let before = predecessors[destination].len();
                    predecessors[destination].push(pair[0]);
                    predecessors[destination].sort_unstable();
                    predecessors[destination].dedup();
                    changed |= predecessors[destination].len() != before;
                }

                let bases = if pop == 0 {
                    smallvec![source]
                } else {
                    predecessor_states_at_depth(&snapshot, source, pop)
                };
                let first = pushes[0] as usize;
                let before = predecessors[first].len();
                predecessors[first].extend_from_slice(&bases);
                predecessors[first].sort_unstable();
                predecessors[first].dedup();
                changed |= predecessors[first].len() != before;
            }
        }

        if !changed {
            return Ok(predecessors);
        }
    }

    Err(format!(
        "reference control-elimination predecessor propagation did not converge: states={}",
        table.num_states,
    ))
}

fn frame_top_branches(
    predecessors: &[PredecessorSet],
    origin_state: u32,
    frame: &StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Result<Vec<(u32, StackEffectFrame)>, String> {
    if let Some(&state) = frame.pushes.last() {
        return Ok(vec![(state, frame.clone())]);
    }

    let Some(states) = states_at_depth(
        predecessors,
        origin_state,
        frame.pop,
        states_at_depth_cache,
        budget,
    )
    .cloned()
    else {
        if budget.is_aborted() {
            return Err(control_elimination_failure(
                budget,
                origin_state,
                EOF,
                "top-state traversal failed",
            ));
        }
        return Ok(Vec::new());
    };

    if states.len() == 1 {
        return Ok(vec![(states[0], frame.clone())]);
    }

    let mut branches = Vec::with_capacity(states.len());
    for state in states {
        let mut branch = frame.clone();
        if add_guard_to_frame(&mut branch, frame.pop, [state]) {
            branches.push((state, branch));
        }
    }
    Ok(branches)
}

fn control_effect_closure(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    controls: &BTreeSet<TerminalID>,
    seed: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Result<Vec<StackEffectFrame>, String> {
    let mut seen = BTreeSet::<StackEffectFrame>::new();
    let mut queue = VecDeque::<StackEffectFrame>::new();
    seen.insert(seed.clone());
    queue.push_back(seed);
    let frame_limit = (table.num_states as usize)
        .saturating_mul(controls.len().max(1))
        .saturating_mul(64)
        .saturating_add(256);

    let mut closure_pops: u64 = 0;
    while let Some(frame) = queue.pop_front() {
        closure_pops += 1;
        // Cache-hit closure steps burn no visits; check the wall backstop
        // directly so a giant sparse-control closure cannot stall.
        if closure_pops % 16_384 == 0 && !budget.check_elapsed() {
            return Err(control_elimination_failure(
                budget,
                origin_state,
                EOF,
                "control stack-effect closure exceeded its compile budget",
            ));
        }
        for (top_state, top_frame) in frame_top_branches(
            predecessors,
            origin_state,
            &frame,
            states_at_depth_cache,
            budget,
        )? {
            for &control in controls {
                let Some(action) = table.action[top_state as usize].get(&control) else {
                    continue;
                };
                let Some(result) = stack_effects_for_action(
                    table,
                    predecessors,
                    origin_state,
                    control,
                    top_state,
                    action,
                    top_frame.clone(),
                    states_at_depth_cache,
                    &mut FxHashSet::default(),
                    &mut Vec::new(),
                    budget,
                ) else {
                    return Err(control_elimination_failure(
                        budget,
                        origin_state,
                        control,
                        "control stack-effect traversal failed",
                    ));
                };
                for effect in result.effects {
                    let next = StackEffectFrame {
                        pop: effect.pop,
                        pushes: effect.pushes,
                        guards: effect.guards,
                    };
                    if seen.insert(next.clone()) {
                        if seen.len() > frame_limit {
                            return Err(format!(
                                "control stack-effect closure did not converge: origin_state={origin_state} controls={controls:?} frames={} limit={frame_limit}",
                                seen.len(),
                            ));
                        }
                        queue.push_back(next);
                    }
                }
            }
        }
    }

    Ok(seen.into_iter().collect())
}

fn continue_eof_accept_from_frames(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    frames: Vec<StackEffectFrame>,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    budget: &UnitInlineBudget,
) -> Result<Vec<GuardedStackShift>, String> {
    let mut out = Vec::new();
    for frame in frames {
        for (top_state, top_frame) in frame_top_branches(
            predecessors,
            origin_state,
            &frame,
            states_at_depth_cache,
            budget,
        )? {
            let Some(next) = table.action[top_state as usize].get(&EOF) else {
                continue;
            };
            out.extend(eof_accept_effects_for_action(
                table,
                predecessors,
                origin_state,
                top_state,
                next,
                top_frame,
                states_at_depth_cache,
                visiting,
                budget,
            )?);
        }
    }
    Ok(out)
}

/// Compile an EOF reduction/control path into the stack rewrite which reaches
/// the table's unique canonical `Accept` state. EOF is never committed as a
/// model token, so shifts and stack shifts on this column are zero-width
/// completion rewrites and must be followed until `Accept` rather than treated
/// as consuming endpoints. Keeping only the canonical state as `Action::Accept`
/// preserves the standalone-table embedding contract for nested composition.
fn eof_accept_effects_for_action(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    budget: &UnitInlineBudget,
) -> Result<Vec<GuardedStackShift>, String> {
    if !budget.record_stack_effect_visit() {
        return Err(control_elimination_failure(
            budget,
            origin_state,
            EOF,
            "EOF stack-effect traversal exceeded its compile budget",
        ));
    }
    let key = StackEffectVisitKey {
        state,
        tid: EOF,
        action_tag: stack_effect_action_tag(action),
        frame: frame.clone(),
    };
    if !visiting.insert(key.clone()) {
        // A cyclic completion branch contributes no new accepting effect. Any
        // acyclic sibling branch remains live and is collected independently.
        return Ok(Vec::new());
    }

    let result = (|| -> Result<Vec<GuardedStackShift>, String> {
        match action {
            Action::Accept => Ok(vec![frame_to_guarded_shift(frame)]),
            Action::Reduce(nonterminal, len) => {
                let frames = match apply_reduce_to_frame(
                    table,
                    predecessors,
                    origin_state,
                    frame,
                    *nonterminal,
                    *len,
                    states_at_depth_cache,
                    budget,
                ) {
                    Some(ReduceFrameResult::Dead) => return Ok(Vec::new()),
                    Some(ReduceFrameResult::Frames { frames, .. }) => frames,
                    None => {
                        return Err(control_elimination_failure(
                            budget,
                            origin_state,
                            EOF,
                            "EOF reduction traversal failed",
                        ));
                    }
                };
                continue_eof_accept_from_frames(
                    table,
                    predecessors,
                    origin_state,
                    frames,
                    states_at_depth_cache,
                    visiting,
                    budget,
                )
            }
            Action::Split {
                shift,
                reduces,
                accept,
            } => {
                let mut out = Vec::new();
                if *accept {
                    out.push(frame_to_guarded_shift(frame.clone()));
                }
                if let Some((target, replace)) = shift {
                    let mut shifted = frame.clone();
                    let effective_replace =
                        *replace && !table.forwarded_shifts.contains(&(state, EOF));
                    push_transition_to_frame(&mut shifted, *target, effective_replace);
                    out.extend(continue_eof_accept_from_frames(
                        table,
                        predecessors,
                        origin_state,
                        vec![shifted],
                        states_at_depth_cache,
                        visiting,
                        budget,
                    )?);
                }
                for &(nonterminal, len) in reduces {
                    out.extend(eof_accept_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        state,
                        &Action::Reduce(nonterminal, len),
                        frame.clone(),
                        states_at_depth_cache,
                        visiting,
                        budget,
                    )?);
                }
                Ok(out)
            }
            Action::Shift(target, replace) => {
                let mut shifted = frame;
                let effective_replace =
                    *replace && !table.forwarded_shifts.contains(&(state, EOF));
                push_transition_to_frame(&mut shifted, *target, effective_replace);
                continue_eof_accept_from_frames(
                    table,
                    predecessors,
                    origin_state,
                    vec![shifted],
                    states_at_depth_cache,
                    visiting,
                    budget,
                )
            }
            Action::ReplaceShifts(targets) => {
                let mut frames = Vec::with_capacity(targets.len());
                for &target in targets.iter() {
                    let mut shifted = frame.clone();
                    pop_frame(&mut shifted, 1);
                    shifted.pushes.push(target);
                    frames.push(shifted);
                }
                continue_eof_accept_from_frames(
                    table,
                    predecessors,
                    origin_state,
                    frames,
                    states_at_depth_cache,
                    visiting,
                    budget,
                )
            }
            Action::StackShifts(shifts) => {
                let frames = shifts
                    .iter()
                    .map(|shift| {
                        let mut shifted = frame.clone();
                        pop_frame(&mut shifted, shift.pop);
                        shifted.pushes.extend_from_slice(&shift.pushes);
                        shifted
                    })
                    .collect();
                continue_eof_accept_from_frames(
                    table,
                    predecessors,
                    origin_state,
                    frames,
                    states_at_depth_cache,
                    visiting,
                    budget,
                )
            }
            Action::GuardedStackShifts(shifts) => {
                let mut frames = Vec::new();
                for shift in shifts {
                    if let Some(Some(shifted)) =
                        compose_guarded_shift_with_frame(frame.clone(), shift)
                    {
                        frames.push(shifted);
                    }
                }
                continue_eof_accept_from_frames(
                    table,
                    predecessors,
                    origin_state,
                    frames,
                    states_at_depth_cache,
                    visiting,
                    budget,
                )
            }
            Action::Skip => continue_eof_accept_from_frames(
                table,
                predecessors,
                origin_state,
                vec![frame],
                states_at_depth_cache,
                visiting,
                budget,
            ),
        }
    })();

    visiting.remove(&key);
    let mut effects = result?;
    effects.sort();
    effects.dedup();
    Ok(effects)
}

fn control_stack_effect_action(
    table: &GLRTable,
    mut effects: Vec<GuardedStackShift>,
) -> Option<Action> {
    for effect in &mut effects {
        for guard in &mut effect.guards {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
    effects.sort();
    effects.dedup();
    if effects.is_empty() {
        return None;
    }

    if effects.iter().all(|effect| effect.guards.is_empty()) {
        let mut shifts = effects
            .into_iter()
            .map(|effect| StackShift {
                pop: effect.pop,
                pushes: effect.pushes,
            })
            .collect::<Vec<_>>();
        normalize_stack_shifts(&mut shifts);
        if shifts.len() == 1 && shifts[0].pop == 0 && shifts[0].pushes.is_empty() {
            return Some(Action::Skip);
        }
        if stack_shift_predecessor_canonicalization_enabled() {
            canonicalize_stack_shift_predecessors_by_goto_superset(table, &mut shifts);
        }
        return stack_shift_action(shifts);
    }

    // This is an exact linker compilation pass, not an optional table-size
    // optimization. Environment caps used by generic reduction inlining must
    // never turn a live subgrammar path into a missing action.
    Some(Action::GuardedStackShifts(effects))
}

fn subset_key(subset: &StateSubset) -> Vec<u32> {
    subset.iter().copied().collect()
}

fn union_state_subsets(
    states: impl IntoIterator<Item = u32>,
    constituent_sets: &[StateSubset],
) -> StateSubset {
    let mut out = StateSubset::new();
    for state in states {
        out.extend_from_slice(&constituent_sets[state as usize]);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn merge_shift_into_pending(
    pending: &mut PendingAction,
    target: u32,
    replace: bool,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<(), ()> {
    match pending.shift {
        None => {
            pending.shift = Some((target, replace));
            Ok(())
        }
        Some((existing_target, existing_replace)) => {
            if existing_target == target {
                return if existing_replace == replace { Ok(()) } else { Err(()) };
            }
            if existing_replace != replace {
                return Err(());
            }
            let merged_subset = union_state_subsets([existing_target, target], constituent_sets);
            let merged_target = ensure_subset_state(
                table,
                &merged_subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
                budget,
            )?;
            pending.shift = Some((merged_target, replace));
            Ok(())
        }
    }
}

fn merge_action_into_pending(
    pending: &mut PendingAction,
    action: &Action,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<(), ()> {
    match action {
        Action::Shift(target, replace) => merge_shift_into_pending(
            pending,
            *target,
            *replace,
            table,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        ),
        Action::StackShifts(_) | Action::ReplaceShifts(_) => Err(()),
        Action::GuardedStackShifts(_) | Action::Skip => Err(()),
        Action::Reduce(nt, len) => {
            pending.push_reduce(*nt, *len);
            Ok(())
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                merge_shift_into_pending(
                    pending,
                    *target,
                    *replace,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    budget,
                )?;
            }
            for &(nt, len) in reduces {
                pending.push_reduce(nt, len);
            }
            if *accept {
                pending.push_accept();
            }
            Ok(())
        }
        Action::Accept => {
            pending.push_accept();
            Ok(())
        }
    }
}

fn build_merged_action_row(
    table: &mut GLRTable,
    subset: &StateSubset,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<ActionRow, ()> {
    let mut terminals = BTreeSet::new();
    for &state in subset {
        for tid in table.action[state as usize].keys() {
            terminals.insert(tid);
        }
    }

    let mut row = ActionRow::default();
    for tid in terminals {
        if !budget.record_cell() {
            return Err(());
        }
        let mut pending = PendingAction::default();
        for &state in subset {
            if let Some(action) = table.action[state as usize].get(&tid).cloned() {
                merge_action_into_pending(
                    &mut pending,
                    &action,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    budget,
                )?;
            }
        }
        if let Some(action) = pending.maybe_finish() {
            row.insert(tid, action);
        }
    }

    Ok(row)
}

fn build_merged_goto_row(
    table: &mut GLRTable,
    subset: &StateSubset,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<GotoRow, ()> {
    let mut nts = BTreeSet::new();
    for &state in subset {
        for &nt in table.goto[state as usize].keys() {
            nts.insert(nt);
        }
    }

    let mut row = GotoRow::default();
    for nt in nts {
        if !budget.record_cell() {
            return Err(());
        }
        let mut replace: Option<bool> = None;
        let mut target_subset = StateSubset::new();
        let mut saw_target = false;

        for &state in subset {
            if let Some(&(target, is_replace)) = table.goto[state as usize].get(&nt) {
                saw_target = true;
                match replace {
                    None => replace = Some(is_replace),
                    Some(existing) if existing == is_replace => {}
                    Some(_) => return Err(()),
                }
                target_subset.extend_from_slice(&constituent_sets[target as usize]);
            }
        }

        if !saw_target {
            continue;
        }

        target_subset.sort_unstable();
        target_subset.dedup();
        target_subset.sort_unstable();
        target_subset.dedup();
        let merged_target = ensure_subset_state(
            table,
            &target_subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        )?;
        row.insert(nt, (merged_target, replace.unwrap()));
    }

    Ok(row)
}

fn union_advance_rows(table: &GLRTable, subset: &StateSubset) -> BitSet {
    let mut out = BitSet::new(table.num_terminals as usize + 1);
    if table.advance.len() == table.num_states as usize {
        for &state in subset {
            out.union_with(&table.advance[state as usize]);
        }
    } else {
        for &state in subset {
            for terminal in table.action[state as usize].keys() {
                let bit = if terminal == EOF {
                    table.num_terminals as usize
                } else if terminal < table.num_terminals {
                    terminal as usize
                } else {
                    continue;
                };
                out.set(bit);
            }
        }
    }
    out
}

fn ensure_subset_state(
    table: &mut GLRTable,
    subset: &StateSubset,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<u32, ()> {
    debug_assert!(!subset.is_empty());
    if subset.len() == 1 {
        return Ok(subset[0]);
    }

    let key = subset_key(subset);
    if let Some(&state) = subset_to_state.get(&key) {
        return Ok(state);
    }
    if failed_subsets.contains(&key) {
        return Err(());
    }

    if !budget.record_synthetic_state() {
        return Err(());
    }
    let had_advance_rows = table.advance.len() == table.num_states as usize;
    let advance_row = had_advance_rows.then(|| union_advance_rows(table, subset));

    let state = table.num_states;
    table.num_states += 1;
    table.action.push(ActionRow::default());
    table.goto.push(GotoRow::default());
    if let Some(advance_row) = advance_row {
        table.advance.push(advance_row);
    }
    constituent_sets.push(subset.clone());
    subset_to_state.insert(key.clone(), state);

    let built = (|| {
        let action_row = build_merged_action_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        )?;
        let goto_row = build_merged_goto_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        )?;
        Ok::<_, ()>((action_row, goto_row))
    })();

    match built {
        Ok((action_row, goto_row)) => {
            table.action[state as usize] = action_row;
            table.goto[state as usize] = goto_row;
            Ok(state)
        }
        Err(()) => {
            subset_to_state.remove(&key);
            failed_subsets.insert(key);
            table.action.pop();
            table.goto.pop();
            if had_advance_rows {
                table.advance.pop();
            }
            table.num_states -= 1;
            constituent_sets.pop();
            Err(())
        }
    }
}

fn refresh_merged_states_depending_on(
    table: &mut GLRTable,
    original_num_states: u32,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    dirty_original_states: &BTreeSet<u32>,
    budget: &UnitInlineBudget,
) {
    let mut state = original_num_states as usize;
    while state < table.num_states as usize {
        if !budget.check_elapsed() {
            break;
        }
        // Merged-state rows depend only on their flattened original
        // constituents, so only subsets intersecting changed original rows
        // need to be rebuilt.
        if !constituent_sets[state]
            .iter()
            .any(|member| dirty_original_states.contains(member))
        {
            state += 1;
            continue;
        }

        let subset = constituent_sets[state].clone();
        let rebuilt = (|| {
            let action_row = build_merged_action_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
                budget,
            )?;
            let goto_row = build_merged_goto_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
                budget,
            )?;
            Ok::<_, ()>((action_row, goto_row))
        })();

        if let Ok((action_row, goto_row)) = rebuilt {
            table.action[state] = action_row;
            table.goto[state] = goto_row;
            if table.advance.len() == table.num_states as usize {
                table.advance[state] = union_advance_rows(table, &subset);
            }
        }

        state += 1;
    }
}

fn unit_reduce_destination(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    lhs: NonterminalID,
) -> Option<u32> {
    let preds = &predecessors[state as usize];
    if preds.is_empty() {
        return None;
    }

    let relevant_preds: Vec<u32> = preds
        .iter()
        .copied()
        .filter(|&pred| table.goto[pred as usize].contains_key(&lhs))
        .collect();
    if relevant_preds.is_empty() {
        return None;
    }

    let mut reduce_dst: Option<u32> = None;
    for pred in relevant_preds {
        let (dst, is_replace) = table.goto[pred as usize][&lhs];
        if is_replace {
            return None;
        }
        if table.goto[dst as usize] != table.goto[state as usize] {
            return None;
        }
        match reduce_dst {
            None => reduce_dst = Some(dst),
            Some(existing) if existing == dst => {}
            Some(_) => return None,
        }
    }

    reduce_dst
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StackEffectFrame {
    pop: u32,
    pushes: Vec<u32>,
    guards: Vec<StackShiftGuard>,
}

#[derive(Debug, Clone)]
struct ContextStateSharePlan {
    group: Vec<u32>,
    predecessors_by_member: Vec<Vec<u32>>,
    generated_guard_states: Vec<u32>,
    /// Exact macro row compiled against the original table during viability.
    /// Once the final sharing map is known, remapping its embedded state IDs is
    /// sufficient; recompiling the same suffix effects would be redundant.
    action_row: ActionRow,
}

fn context_share_provenance_plan(
    group: &[u32],
    predecessors: &[StateSubset],
) -> Option<ContextStateSharePlan> {
    let predecessors_by_member = group
        .iter()
        .map(|&state| predecessors.get(state as usize).map(|set| set.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    if predecessors_by_member.iter().any(Vec::is_empty) {
        return None;
    }
    let mut seen = BTreeSet::<u32>::new();
    for member_predecessors in &predecessors_by_member {
        for &predecessor in member_predecessors {
            if !seen.insert(predecessor) {
                return None;
            }
        }
    }
    Some(ContextStateSharePlan {
        group: group.to_vec(),
        predecessors_by_member,
        generated_guard_states: Vec::new(),
        action_row: ActionRow::default(),
    })
}

fn context_share_frame_for_member(state: u32, predecessors: &[u32]) -> StackEffectFrame {
    StackEffectFrame {
        pop: 1,
        pushes: vec![state],
        guards: vec![StackShiftGuard {
            pop: 1,
            states: predecessors.to_vec(),
        }],
    }
}

enum ReduceFrameResult {
    Dead,
    Frames {
        frames: Vec<StackEffectFrame>,
        origin_dependent: bool,
    },
}

/// Union one predecessor frontier without sorting a potentially much larger
/// multiset of repeated targets. Only representation changes: results retain
/// the same ascending IDs, and the caller still charges one visit per depth.
fn predecessor_frontier_union(
    predecessors: &[PredecessorSet],
    states: &[u32],
    bitmap: &mut Vec<u64>,
    bitmap_enabled: bool,
) -> Option<StateSubset> {
    fn sorted_union(predecessors: &[PredecessorSet], states: &[u32]) -> Option<StateSubset> {
        let mut next = StateSubset::new();
        for &state in states {
            next.extend_from_slice(predecessors.get(state as usize)?);
        }
        next.sort_unstable();
        next.dedup();
        Some(next)
    }

    const MAX_BITMAP_STATES: usize = 65_536;
    if !bitmap_enabled || states.len() < 4 || predecessors.len() > MAX_BITMAP_STATES {
        return sorted_union(predecessors, states);
    }
    let words = predecessors.len().div_ceil(64);
    let mut edges = 0usize;
    for &state in states {
        edges = edges.saturating_add(predecessors.get(state as usize)?.len());
    }
    // Sparse unions keep the original small-vector path. The threshold pays
    // for clearing/enumerating the bounded bitmap as well as the second scan.
    if edges < words.saturating_mul(4).max(64) {
        return sorted_union(predecessors, states);
    }
    bitmap.resize(words, 0);
    bitmap.fill(0);
    for &state in states {
        for &target in &predecessors[state as usize] {
            let Some(word) = bitmap.get_mut(target as usize / 64) else {
                // A malformed graph can contain a dangling target that the
                // reference retains at its last depth. Never silently discard
                // it or assume every ID fits the valid-state bitmap.
                return sorted_union(predecessors, states);
            };
            *word |= 1u64 << (target % 64);
        }
    }
    let mut next = StateSubset::new();
    for (index, &word) in bitmap.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            next.push((index * 64 + word.trailing_zeros() as usize) as u32);
            word &= word - 1;
        }
    }
    Some(next)
}


fn states_at_depth<'a>(
    predecessors: &[PredecessorSet],
    origin_state: u32,
    depth: u32,
    cache: &'a mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Option<&'a StateSubset> {
    states_at_depth_with_bitmap(predecessors, origin_state, depth, cache, budget, true)
}

fn states_at_depth_with_bitmap<'a>(
    predecessors: &[PredecessorSet],
    origin_state: u32,
    depth: u32,
    cache: &'a mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
    bitmap_enabled: bool,
) -> Option<&'a StateSubset> {
    let cache_key = (origin_state, depth);
    if !cache.contains_key(&cache_key) {
        let mut states = smallvec![origin_state];
        let mut bitmap = Vec::new();
        for _ in 0..depth {
            if !budget.record_stack_effect_visit() {
                return None;
            }
            let next = predecessor_frontier_union(predecessors, &states, &mut bitmap, bitmap_enabled)?;
            if next.is_empty() {
                cache.insert(cache_key, None);
                return None;
            }
            states = next;
        }

        cache.insert(cache_key, Some(states));
    }

    cache.get(&cache_key).and_then(Option::as_ref)
}

fn normalize_states(mut states: Vec<u32>) -> Vec<u32> {
    states.sort_unstable();
    states.dedup();
    states
}

fn add_guard_to_frame(
    frame: &mut StackEffectFrame,
    pop: u32,
    states: impl IntoIterator<Item = u32>,
) -> bool {
    let states = normalize_states(states.into_iter().collect());
    if states.is_empty() {
        return false;
    }

    if let Some(existing) = frame.guards.iter_mut().find(|guard| guard.pop == pop) {
        let wanted: BTreeSet<u32> = states.into_iter().collect();
        existing.states.retain(|state| wanted.contains(state));
        return !existing.states.is_empty();
    }

    frame.guards.push(StackShiftGuard { pop, states });
    frame.guards.sort_by_key(|guard| guard.pop);
    true
}

fn pop_frame(frame: &mut StackEffectFrame, pop: u32) {
    if pop as usize <= frame.pushes.len() {
        let keep = frame.pushes.len() - pop as usize;
        frame.pushes.truncate(keep);
    } else {
        frame.pop += pop - frame.pushes.len() as u32;
        frame.pushes.clear();
    }
}

fn push_transition_to_frame(frame: &mut StackEffectFrame, target: u32, replace: bool) {
    if replace {
        if let Some(top) = frame.pushes.last_mut() {
            *top = target;
        } else {
            frame.pop += 1;
            frame.pushes.push(target);
        }
    } else {
        frame.pushes.push(target);
    }
}

fn frame_to_guarded_shift(frame: StackEffectFrame) -> GuardedStackShift {
    GuardedStackShift {
        guards: frame.guards,
        pop: frame.pop,
        pushes: frame.pushes,
    }
}

fn stack_effect_action_tag(action: &Action) -> u8 {
    match action {
        Action::Shift(..) => 0,
        Action::StackShifts(_) => 1,
        Action::GuardedStackShifts(_) => 2,
        Action::Reduce(..) => 3,
        Action::Split { .. } => 4,
        Action::Accept => 5,
        Action::ReplaceShifts(_) => 6,
        Action::Skip => 7,
    }
}

fn apply_reduce_to_frame(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Option<ReduceFrameResult> {
    pop_frame(&mut frame, len);

    let mut origin_dependent = false;
    let direct_goto_from;
    let goto_froms = if let Some(&state) = frame.pushes.last() {
        direct_goto_from = smallvec![state];
        &direct_goto_from
    } else {
        origin_dependent = true;
        states_at_depth(
            predecessors,
            origin_state,
            frame.pop,
            states_at_depth_cache,
            budget,
        )?
    };

    if goto_froms.len() == 1 {
        let goto_from = goto_froms
            .first()
            .copied()
            .expect("a singleton set has one element");
        let Some((target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            return Some(ReduceFrameResult::Dead);
        };
        push_transition_to_frame(&mut frame, target, replace);
        return Some(ReduceFrameResult::Frames {
            frames: vec![frame],
            origin_dependent,
        });
    }

    let guard_pop = frame.pop;
    let mut by_target: BTreeMap<(u32, bool), BTreeSet<u32>> = BTreeMap::new();
    let mut missing = 0usize;
    for &goto_from in goto_froms {
        let Some((next_target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            missing += 1;
            continue;
        };
        by_target
            .entry((next_target, replace))
            .or_default()
            .insert(goto_from);
    }

    if missing > 0 && by_target.is_empty() {
        return Some(ReduceFrameResult::Dead);
    }

    let needs_guard = missing > 0 || by_target.len() > 1;
    let mut frames = Vec::new();
    for ((target, replace), froms) in by_target {
        let mut next_frame = frame.clone();
        if needs_guard && !add_guard_to_frame(&mut next_frame, guard_pop, froms.into_iter()) {
            continue;
        }
        push_transition_to_frame(&mut next_frame, target, replace);
        frames.push(next_frame);
    }

    if frames.is_empty() {
        Some(ReduceFrameResult::Dead)
    } else {
        frames.sort();
        frames.dedup();
        Some(ReduceFrameResult::Frames {
            frames,
            origin_dependent,
        })
    }
}

fn compose_guarded_shift_with_frame(
    mut frame: StackEffectFrame,
    shift: &GuardedStackShift,
) -> Option<Option<StackEffectFrame>> {
    let pushed_len = frame.pushes.len() as u32;

    for guard in &shift.guards {
        if guard.states.is_empty() {
            return Some(None);
        }

        if guard.pop < pushed_len {
            let idx = (pushed_len - 1 - guard.pop) as usize;
            let known_state = frame.pushes[idx];
            if guard.states.binary_search(&known_state).is_err() {
                return Some(None);
            }
        } else {
            let translated_pop = frame.pop + (guard.pop - pushed_len);
            if !add_guard_to_frame(&mut frame, translated_pop, guard.states.iter().copied()) {
                return Some(None);
            }
        }
    }

    if shift.pop < shift.guards.iter().map(|guard| guard.pop).max().unwrap_or(0) {
        return None;
    }

    pop_frame(&mut frame, shift.pop);
    frame.pushes.extend_from_slice(&shift.pushes);
    Some(Some(frame))
}

fn stack_effects_for_action(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    tid: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
    budget: &UnitInlineBudget,
) -> Option<StackEffectResult> {
    if !budget.record_stack_effect_visit() {
        return None;
    }
    let key = StackEffectVisitKey {
        state,
        tid,
        action_tag: stack_effect_action_tag(action),
        frame: frame.clone(),
    };
    if !visiting.insert(key.clone()) {
        return None;
    }

    let result = (|| {
        let mut out = Vec::new();
        let mut origin_dependent = false;
        match action {
            Action::Skip => out.push(frame_to_guarded_shift(frame)),
            Action::Shift(target, replace) => {
                let mut frame = frame;
                let effective_replace = *replace && !table.forwarded_shifts.contains(&(state, tid));
                push_transition_to_frame(&mut frame, *target, effective_replace);
                out.push(frame_to_guarded_shift(frame));
            }
            Action::ReplaceShifts(targets) => {
                for &target in targets.iter() {
                    let mut frame = frame.clone();
                    pop_frame(&mut frame, 1);
                    frame.pushes.push(target);
                    out.push(frame_to_guarded_shift(frame));
                }
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    let mut frame = frame.clone();
                    pop_frame(&mut frame, shift.pop);
                    frame.pushes.extend_from_slice(&shift.pushes);
                    out.push(frame_to_guarded_shift(frame));
                }
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    match compose_guarded_shift_with_frame(frame.clone(), shift)? {
                        None => {}
                        Some(frame) => out.push(frame_to_guarded_shift(frame)),
                    }
                }
            }
            Action::Reduce(nt, len) => {
                let frames = match apply_reduce_to_frame(
                    table,
                    predecessors,
                    origin_state,
                    frame,
                    *nt,
                    *len,
                    states_at_depth_cache,
                    budget,
                )? {
                    ReduceFrameResult::Dead => {
                        return Some(StackEffectResult {
                            effects: Vec::new(),
                            origin_dependent,
                        })
                    }
                    ReduceFrameResult::Frames {
                        frames,
                        origin_dependent: reduce_origin_dependent,
                    } => {
                        origin_dependent |= reduce_origin_dependent;
                        frames
                    }
                };
                for frame in frames {
                    let Some(&next_state) = frame.pushes.last() else {
                        continue;
                    };
                    let Some(next) = table.action[next_state as usize].get(&tid) else {
                        continue;
                    };
                    dependencies.push(((origin_state, tid), (next_state, tid)));
                    let next_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        next_state,
                        next,
                        frame,
                        states_at_depth_cache,
                        visiting,
                        dependencies,
                        budget,
                    )?;
                    origin_dependent |= next_result.origin_dependent;
                    out.extend(next_result.effects);
                }
            }
            Action::Split { shift, reduces, accept } => {
                if *accept {
                    return None;
                }
                if let Some((target, replace)) = shift {
                    let shift_action = Action::Shift(*target, *replace);
                    let shift_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &shift_action,
                        frame.clone(),
                        states_at_depth_cache,
                        visiting,
                        dependencies,
                        budget,
                    )?;
                    origin_dependent |= shift_result.origin_dependent;
                    out.extend(shift_result.effects);
                }
                for &(nt, len) in reduces {
                    let reduce_action = Action::Reduce(nt, len);
                    let reduce_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &reduce_action,
                        frame.clone(),
                        states_at_depth_cache,
                        visiting,
                        dependencies,
                        budget,
                    )?;
                    origin_dependent |= reduce_result.origin_dependent;
                    out.extend(reduce_result.effects);
                }
            }
            Action::Accept => return None,
        }

        out.sort();
        out.dedup();
        Some(StackEffectResult {
            effects: out,
            origin_dependent,
        })
    })();

    visiting.remove(&key);
    result
}

fn normalize_guarded_effects(effects: &mut Vec<GuardedStackShift>) {
    for effect in effects.iter_mut() {
        for guard in effect.guards.iter_mut() {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
    effects.retain(|effect| !effect.pushes.is_empty());
    effects.sort();
    effects.dedup();
}

fn stack_effect_action(table: &GLRTable, mut effects: Vec<GuardedStackShift>) -> Option<Action> {
    normalize_guarded_effects(&mut effects);
    if effects.is_empty() {
        return None;
    }
    // Core-merged predecessor graphs can encode guarded effects very
    // compactly in the LR table but expand them into runtime guard scans. The
    // reduction interpreter is cheaper for those origin-dependent actions.
    // Unguarded effects are deterministic stack rewrites and remain profitable.
    if table.construction == GlrTableConstruction::ExperimentalCoreMerged
        && effects.iter().any(|effect| !effect.guards.is_empty())
    {
        return None;
    }
    if effects.iter().all(|effect| effect.guards.is_empty()) {
        let mut shifts: Vec<_> = effects
            .into_iter()
            .map(|effect| StackShift {
                pop: effect.pop,
                pushes: effect.pushes,
            })
            .collect();
        if stack_shift_predecessor_canonicalization_enabled() {
            canonicalize_stack_shift_predecessors_by_goto_superset(table, &mut shifts);
        }
        return stack_shift_action(shifts);
    }
    // Opt-in diagnostic knob only. Do not use this to hide correctness or
    // compile-time bugs in guarded stack-effect lowering.
    if max_guarded_stack_effects().is_some_and(|limit| effects.len() > limit) {
        return None;
    }
    Some(Action::GuardedStackShifts(effects))
}

#[derive(Clone, Copy, Debug)]
enum StackInlineOutcomeKind {
    NotCandidate,
    TraversalFailed,
    EmptyEffects,
    OriginDependentOnePush,
    TrivialOriginDependentOnePush,
    CannotRepresent,
    Inlined,
}

impl StackInlineOutcomeKind {
    const COUNT: usize = 7;

    fn index(self) -> usize {
        match self {
            Self::NotCandidate => 0,
            Self::TraversalFailed => 1,
            Self::EmptyEffects => 2,
            Self::OriginDependentOnePush => 3,
            Self::TrivialOriginDependentOnePush => 4,
            Self::CannotRepresent => 5,
            Self::Inlined => 6,
        }
    }
}

struct StackInlineAttempt {
    action: Option<Action>,
    outcome: StackInlineOutcomeKind,
}

fn try_inline_action_to_stack_shifts_detailed(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
    budget: &UnitInlineBudget,
) -> StackInlineAttempt {
    let has_reductions = match action {
        Action::Reduce(..) => true,
        Action::Split {
            reduces,
            accept: false,
            ..
        } => !reduces.is_empty(),
        _ => false,
    };
    if !has_reductions {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::NotCandidate,
        };
    }

    let Some(result) = stack_effects_for_action(
        table,
        predecessors,
        state,
        tid,
        state,
        action,
        StackEffectFrame {
            pop: 0,
            pushes: Vec::new(),
            guards: Vec::new(),
        },
        states_at_depth_cache,
        &mut FxHashSet::default(),
        dependencies,
        budget,
    ) else {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::TraversalFailed,
        };
    };
    let effects = result.effects;
    if result.origin_dependent
        && matches!(action, Action::Reduce(_, 1))
        && !effects.is_empty()
        && effects.iter().all(|effect| effect.pushes.len() == 1)
    {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::OriginDependentOnePush,
        };
    }
    if effects.is_empty() {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::EmptyEffects,
        };
    }
    match stack_effect_action(table, effects) {
        Some(action) => StackInlineAttempt {
            action: Some(action),
            outcome: StackInlineOutcomeKind::Inlined,
        },
        None => StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::CannotRepresent,
        },
    }
}

fn try_inline_action_to_stack_shifts(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
    budget: &UnitInlineBudget,
) -> Option<Action> {
    try_inline_action_to_stack_shifts_detailed(
        table,
        predecessors,
        state,
        tid,
        action,
        states_at_depth_cache,
        dependencies,
        budget,
    )
    .action
}

fn normalize_stack_shifts(shifts: &mut Vec<StackShift>) {
    shifts.sort_by(|a, b| a.pop.cmp(&b.pop).then_with(|| a.pushes.cmp(&b.pushes)));
    shifts.dedup();
}

fn canonicalize_stack_shift_predecessors_by_goto_superset(table: &GLRTable, shifts: &mut [StackShift]) {
    for idx in 0..shifts.len() {
        if shifts[idx].pushes.len() < 2 {
            continue;
        }

        for pos in 0..shifts[idx].pushes.len() - 1 {
            for rep_idx in 0..idx {
                if shifts[idx].pop != shifts[rep_idx].pop
                    || shifts[idx].pushes.len() != shifts[rep_idx].pushes.len()
                    || shifts[idx].pushes[..pos] != shifts[rep_idx].pushes[..pos]
                    || shifts[idx].pushes[pos + 1..] != shifts[rep_idx].pushes[pos + 1..]
                {
                    continue;
                }

                // This pushed state is buried below an identical pushed suffix
                // and below the current top state. Once buried, it can only be
                // observed by a later reduction querying its goto row, so
                // prefer a predecessor whose goto row is a compatible superset
                // and let the otherwise identical stack paths merge.
                let pred = shifts[idx].pushes[pos];
                let rep = shifts[rep_idx].pushes[pos];
                if goto_row_is_target_compatible_subset(table, pred, rep) {
                    shifts[idx].pushes[pos] = rep;
                    break;
                }
                if goto_row_is_target_compatible_subset(table, rep, pred) {
                    shifts[rep_idx].pushes[pos] = pred;
                    break;
                }
            }
        }
    }
}

fn goto_row_is_target_compatible_subset(table: &GLRTable, subset: u32, superset: u32) -> bool {
    let subset_row = &table.goto[subset as usize];
    let superset_row = &table.goto[superset as usize];
    !subset_row.is_empty()
        && subset_row.iter().all(|(&nt, &target)| {
            superset_row
                .get(&nt)
                .is_some_and(|&superset_target| superset_target == target)
        })
}

fn stack_shift_action(mut shifts: Vec<StackShift>) -> Option<Action> {
    normalize_stack_shifts(&mut shifts);
    if shifts.is_empty() {
        return None;
    }
    if shifts.len() == 1 {
        let shift = &shifts[0];
        if shift.pushes.len() == 1 {
            match shift.pop {
                0 => return Some(Action::Shift(shift.pushes[0], false)),
                1 => return Some(Action::Shift(shift.pushes[0], true)),
                _ => {}
            }
        }
    }
    Some(Action::StackShifts(shifts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::accumulator::TerminalsDisallowed;
    use crate::glr::parser::{ParserGSS, advance_stacks};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var(name).ok();
            unsafe {
                std::env::set_var(name, value);
            }
            Self { name, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(previous) = &self.previous {
                    std::env::set_var(self.name, previous);
                } else {
                    std::env::remove_var(self.name);
                }
            }
        }
    }

    fn default_unit_inline_budget() -> UnitInlineBudget {
        UnitInlineBudget {
            started_at: std::time::Instant::now(),
            max_ms: DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS,
            max_iterations: DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS,
            max_cells: DEFAULT_UNIT_INLINE_WORK_MAX_CELLS,
            max_synthetic_states: DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES,
            max_stack_effect_visits: DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS,
            iterations: std::sync::atomic::AtomicUsize::new(0),
            cells: std::sync::atomic::AtomicUsize::new(0),
            synthetic_states: std::sync::atomic::AtomicUsize::new(0),
            stack_effect_visits: std::sync::atomic::AtomicUsize::new(0),
            abort_code: std::sync::atomic::AtomicU8::new(ABORT_NONE),
        }
    }

    #[test]
    fn bitmap_predecessor_frontiers_match_literal_sort_with_duplicates_and_dangling_ids() {
        fn reference(predecessors: &[PredecessorSet], states: &[u32]) -> Option<StateSubset> {
            let mut out = StateSubset::new();
            for &state in states { out.extend_from_slice(predecessors.get(state as usize)?); }
            out.sort_unstable();
            out.dedup();
            Some(out)
        }
        let mut seed = 948752_u64;
        let mut random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };
        let mut bitmap = Vec::new();
        for n in [0, 1, 2, 7, 63, 64, 65, 257, 1024] {
            for _ in 0..12 {
                let mut predecessors = vec![PredecessorSet::new(); n];
                for row in &mut predecessors {
                    for _ in 0..(random() as usize % 80) {
                        row.push((random() as usize % n) as u32);
                    }
                }
                let mut states: Vec<u32> = (0..n.min(20) as u32).rev().collect();
                for _ in 0..2 {
                    assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                               reference(&predecessors, &states));
                    states.reverse();
                }
                if n > 0 {
                    // A dangling target is returned at the last depth, not
                    // dropped even when another branch uses the bitmap.
                    predecessors[0].extend_from_slice(&[n as u32, u32::MAX, 0, 0]);
                    assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                               reference(&predecessors, &states));
                }
                states.push(n as u32);
                assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true), None);
            }
        }
        let mut predecessors = vec![PredecessorSet::new(); 65];
        for row in predecessors.iter_mut().take(8) {
            row.extend((0..65).rev().chain(0..65));
        }
        let states: Vec<_> = (0..8).collect();
        bitmap.clear();
        assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                   reference(&predecessors, &states));
        assert_eq!(bitmap.len(), 2, "fixture must actually use dense accumulation");
        // Bits above the last valid state still represent literal dangling IDs.
        predecessors[0].push(127);
        assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                   reference(&predecessors, &states));
        let mut disabled = Vec::new();
        predecessor_frontier_union(&predecessors, &states, &mut disabled, false);
        assert_eq!(disabled.capacity(), 0);
        let huge = vec![PredecessorSet::new(); 65_537];
        let mut oversized = Vec::new();
        assert_eq!(predecessor_frontier_union(&huge, &[0, 1, 2, 3], &mut oversized, true), Some(StateSubset::new()));
        assert_eq!(oversized.capacity(), 0);
    }

    #[test]
    fn bitmap_depth_queries_preserve_cache_failure_and_exact_visit_accounting() {
        let mut graphs = Vec::new();
        for n in [1, 8, 65, 257] {
            let mut graph = vec![PredecessorSet::new(); n];
            for (state, row) in graph.iter_mut().enumerate() {
                for offset in 0..n.min(40) {
                    row.push(((state + offset) % n) as u32);
                    if offset % 3 == 0 { row.push(((state + offset) % n) as u32); }
                }
            }
            graphs.push(graph.clone());
            graph[0].push(u32::MAX);
            graphs.push(graph);
        }
        graphs.push(vec![PredecessorSet::new(); 8]);
        for graph in graphs {
            for limit in [1, 4, 100_000, usize::MAX] {
                let mut caches = [FxHashMap::default(), FxHashMap::default()];
                let mut budgets = [default_unit_inline_budget(), default_unit_inline_budget()];
                for b in &mut budgets { b.max_ms = u128::MAX; b.max_stack_effect_visits = limit; }
                for (origin, depth) in [(0, 0), (0, 1), (0, 3), (0, 3), (1, 2), (0, 8), (u32::MAX, 0), (u32::MAX, 1)] {
                    let left = states_at_depth_with_bitmap(&graph, origin, depth, &mut caches[0], &budgets[0], false).cloned();
                    let right = states_at_depth_with_bitmap(&graph, origin, depth, &mut caches[1], &budgets[1], true).cloned();
                    assert_eq!(left, right, "origin={origin} depth={depth} limit={limit}");
                    assert_eq!(caches[0], caches[1]);
                    assert_eq!(budgets[0].stack_effect_visits(), budgets[1].stack_effect_visits());
                    assert_eq!(budgets[0].is_aborted(), budgets[1].is_aborted());
                    assert_eq!(budgets[0].report().reason, budgets[1].report().reason);
                }
            }
        }
    }

    #[test]
    fn completed_work_certificate_is_strict_monotone_and_overflow_safe() {
        let mut budget = default_unit_inline_budget();
        budget.max_stack_effect_visits = 10;
        budget.stack_effect_visits.store(3, std::sync::atomic::Ordering::Relaxed);
        let work = CompletedStackEffectWork::new(&budget).unwrap();
        assert!(!work.exceeded());
        work.record_completed(7);
        assert!(!work.exceeded(), "work exactly at the cap must continue");
        work.record_completed(1);
        assert!(work.exceeded());
        work.record_completed(usize::MAX);
        assert!(work.exceeded(), "addition must not wrap below the cap");

        budget.max_stack_effect_visits = usize::MAX - 1;
        let work = CompletedStackEffectWork::new(&budget).unwrap();
        work.record_completed(usize::MAX);
        assert!(work.exceeded());
        budget.max_stack_effect_visits = usize::MAX;
        assert!(CompletedStackEffectWork::new(&budget).is_none());
    }

    #[test]
    fn completed_work_production_policy_keeps_legacy_and_lalr_scheduling() {
        assert!(completed_work_abort_eligible(GlrTableConstruction::ExperimentalCoreMerged));
        assert!(!completed_work_abort_eligible(GlrTableConstruction::LegacyRowBisim));
        assert!(!completed_work_abort_eligible(GlrTableConstruction::Lalr));
    }

    #[test]
    fn completed_work_certificate_counts_locally_exhausted_states_concurrently() {
        let mut budget = default_unit_inline_budget();
        budget.max_stack_effect_visits = 63;
        let work = CompletedStackEffectWork::new(&budget).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            (0..16).into_par_iter().for_each(|_| {
                let child = budget.child_with_stack_effect_visit_limit(3);
                for _ in 0..3 { assert!(child.record_stack_effect_visit()); }
                assert!(!child.record_stack_effect_visit());
                work.record_completed(child.stack_effect_visits());
            });
        });
        assert!(work.exceeded());
        // The certificate never aborts or spends the parent's live budget.
        // The read-only phase must finish before folding/rollback is selected.
        assert!(!budget.is_aborted());
        assert_eq!(budget.stack_effect_visits(), 0);
    }

    #[test]
    fn completed_work_abort_preserves_successful_and_rolled_back_tables() {
        fn table(groups: usize, construction: GlrTableConstruction) -> GLRTable {
            let n = groups * 5;
            let mut action = vec![ActionRow::default(); n];
            let mut goto = vec![GotoRow::default(); n];
            for group in 0..groups {
                let base = group * 5;
                action[base + 1].insert(0, Action::Shift((base + 2) as u32, false));
                action[base + 2].insert(0, Action::Split {
                    shift: Some(((base + 4) as u32, false)),
                    reduces: vec![(10, 1)], accept: false,
                });
                action[base + 3].insert(0, Action::Shift((base + 4) as u32, false));
                goto[base + 1].insert(10, ((base + 3) as u32, true));
            }
            let mut table = GLRTable {
                action, goto, num_states: n as u32, num_terminals: 1,
                num_rules: 0, rules: Vec::new(), nonterminal_display_names: Vec::new(),
                construction, admission_policy: AdmissionPolicy::ExactSimulation,
                advance: Vec::new(), unconditional_advance: Vec::new(),
                forwarded_shifts: FxHashSet::default(),
                control_terminals: Default::default(), skip_terminals: Default::default(),
                guarded_shift_index: Vec::new(), direct_regular_wide_frontiers: Vec::new(),
            };
            table.rebuild_advance_rows_from_actions();
            table
        }

        let mut aborts = 0;
        let mut successes = 0;
        let mut changed = 0;
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            for construction in [GlrTableConstruction::LegacyRowBisim,
                                 GlrTableConstruction::ExperimentalCoreMerged] {
                for groups in [1, 8, 64] {
                    for limit in [1, 8, 64, 100_000, usize::MAX] {
                        let input = table(groups, construction);
                        let protected = BitSet::new(1);
                        let mut results = Vec::new();
                        for stop in [false, true] {
                            let mut result = input.clone();
                            let mut budget = default_unit_inline_budget();
                            budget.max_ms = u128::MAX;
                            budget.max_stack_effect_visits = limit;
                            let mut undo = UnitInlineUndo::new(&result);
                            if limit == 1 {
                                // Model a prior iteration that already changed
                                // an original row: certified abort must undo it,
                                // not merely leave the current phase untouched.
                                undo.record_cell(&result, 0, 0);
                                result.action[0].insert(0, Action::Shift(1, false));
                                result.advance[0].set(0);
                                budget.stack_effect_visits.store(
                                    1, std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            pool.install(|| {
                                result.collapse_sr_unit_reductions_with_completed_work_abort(
                                    &budget, &mut undo, &protected, stop,
                                );
                            });
                            if budget.is_aborted() { undo.rollback(&mut result); }
                            results.push((result, budget.is_aborted()));
                        }
                        assert_eq!(results[0].1, results[1].1,
                                   "abort policy threads={threads} groups={groups} limit={limit}");
                        assert_eq!(bincode::serialize(&results[0].0).unwrap(),
                                   bincode::serialize(&results[1].0).unwrap(),
                                   "complete table threads={threads} groups={groups} limit={limit}");
                        if results[0].1 {
                            aborts += 1;
                            assert_eq!(results[0].0.action, input.action);
                            assert_eq!(results[0].0.goto, input.goto);
                        } else {
                            successes += 1;
                            changed += usize::from(results[0].0.action != input.action);
                        }
                    }
                }
            }
        }
        assert!(aborts > 0 && successes > 0 && changed > 0,
                "must cover real aborts and successful nontrivial optimization");
    }

    #[test]
    fn control_predecessor_propagation_matches_synchronous_reference() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        fn random_pushes(seed: &mut u64, nstates: u32) -> Vec<u32> {
            let len = (next(seed) % 3 + 1) as usize;
            (0..len)
                .map(|_| (next(seed) % u64::from(nstates)) as u32)
                .collect()
        }

        fn random_action(seed: &mut u64, nstates: u32) -> Action {
            match next(seed) % 8 {
                0 => Action::Shift(
                    (next(seed) % u64::from(nstates)) as u32,
                    next(seed) & 1 != 0,
                ),
                1 => Action::ReplaceShifts(
                    (0..(next(seed) % 3 + 1))
                        .map(|_| (next(seed) % u64::from(nstates)) as u32)
                        .collect(),
                ),
                2 => Action::StackShifts(
                    (0..(next(seed) % 3 + 1))
                        .map(|_| StackShift {
                            pop: (next(seed) % 4) as u32,
                            pushes: random_pushes(seed, nstates),
                        })
                        .collect(),
                ),
                3 => Action::GuardedStackShifts(
                    (0..(next(seed) % 3 + 1))
                        .map(|_| GuardedStackShift {
                            guards: Vec::new(),
                            pop: (next(seed) % 4) as u32,
                            pushes: random_pushes(seed, nstates),
                        })
                        .collect(),
                ),
                4 => Action::Split {
                    shift: Some((
                        (next(seed) % u64::from(nstates)) as u32,
                        next(seed) & 1 != 0,
                    )),
                    reduces: Vec::new(),
                    accept: false,
                },
                5 => Action::Skip,
                6 => Action::Reduce((next(seed) % 4) as u32, (next(seed) % 4) as u32),
                _ => Action::Accept,
            }
        }

        let mut seed = 0xD1CE_F00D_5EED_BAADu64;
        for case in 0..256 {
            let nstates = (next(&mut seed) % 9 + 2) as usize;
            let mut action = vec![ActionRow::default(); nstates];
            let mut goto = vec![GotoRow::default(); nstates];

            for state in 0..nstates {
                for terminal in 0..4 {
                    if next(&mut seed) % 3 != 0 {
                        action[state].insert(
                            terminal,
                            random_action(&mut seed, nstates as u32),
                        );
                    }
                }
                for nonterminal in 0..4 {
                    if next(&mut seed) & 1 != 0 {
                        goto[state].insert(
                            nonterminal,
                            (
                                (next(&mut seed) % nstates as u64) as u32,
                                next(&mut seed) & 1 != 0,
                            ),
                        );
                    }
                }
            }

            let table = GLRTable {
                action,
                goto,
                num_states: nstates as u32,
                num_terminals: 4,
                num_rules: 0,
                rules: Vec::new(),
                nonterminal_display_names: Vec::new(),
                construction: GlrTableConstruction::LegacyRowBisim,
                admission_policy: AdmissionPolicy::RowPresenceExact,
                advance: Vec::new(),
                unconditional_advance: Vec::new(),
                forwarded_shifts: FxHashSet::default(),
                control_terminals: Default::default(),
                skip_terminals: Default::default(),
                guarded_shift_index: Vec::new(),
                direct_regular_wide_frontiers: Vec::new(),
            };

            let expected = build_control_elimination_predecessors_synchronous_reference(&table)
                .unwrap_or_else(|error| panic!("reference failed for case {case}: {error}"));
            let actual = build_control_elimination_predecessors(&table)
                .unwrap_or_else(|error| panic!("optimized failed for case {case}: {error}"));
            assert_eq!(actual, expected, "case {case}");

            let mut origins = (0..nstates as u32)
                .filter(|_| next(&mut seed) & 3 == 0)
                .collect::<Vec<_>>();
            if origins.is_empty() {
                origins.push((next(&mut seed) % nstates as u64) as u32);
            }
            origins.sort_unstable();
            origins.dedup();
            let demand = build_control_elimination_predecessors_demand(&table, &origins)
                .unwrap_or_else(|error| panic!("demand solver failed for case {case}: {error}"));
            let mut required = vec![false; nstates];
            let mut queue = VecDeque::from(origins.clone());
            while let Some(state) = queue.pop_front() {
                if std::mem::replace(&mut required[state as usize], true) {
                    continue;
                }
                queue.extend(expected[state as usize].iter().copied());
            }
            for state in 0..nstates {
                if required[state] || !demand[state].is_empty() {
                    assert_eq!(
                        demand[state], expected[state],
                        "materialized demand row differs for case {case}, state {state}, origins {origins:?}",
                    );
                }
            }
        }
    }

    fn guarded_effect(pop: u32, guard_count: u32) -> GuardedStackShift {
        GuardedStackShift {
            guards: (0..guard_count)
                .map(|guard_pop| StackShiftGuard {
                    pop: guard_pop,
                    states: vec![0],
                })
                .collect(),
            pop: pop.max(guard_count.saturating_sub(1)),
            pushes: vec![1],
        }
    }

    fn table_with_stack_shifts(
        shifts: Vec<StackShift>,
        goto_rows: &[(u32, &[(NonterminalID, (u32, bool))])],
    ) -> GLRTable {
        let num_states = 8;
        let mut action = vec![ActionRow::default(); num_states];
        action[0].insert(0, Action::StackShifts(shifts));

        let mut goto = vec![GotoRow::default(); num_states];
        for &(state, row) in goto_rows {
            for &(nt, target) in row {
                goto[state as usize].insert(nt, target);
            }
        }

        GLRTable {
            action,
            goto,
            num_states: num_states as u32,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
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

    fn stack_shifts_at_start(table: &GLRTable) -> Vec<StackShift> {
        match table.action(0, 0).expect("expected action at state 0 terminal 0") {
            Action::StackShifts(shifts) => shifts.clone(),
            action => panic!("expected stack shifts, got {action:?}"),
        }
    }

    #[test]
    fn contextual_state_sharing_compacts_ambiguous_top_with_distinct_stack_contexts() {
        // Two reachable parser paths have different caller states (1 and 2)
        // but structurally corresponding tops (3 and 4). Sharing 3~4 should
        // turn the GSS frontier from two top LR IDs into one while retaining
        // the caller distinction one frame below as an exact guard.
        let mut action = vec![ActionRow::default(); 6];
        action[0].insert(10, Action::Shift(1, false));
        action[0].insert(11, Action::Shift(2, false));
        action[1].insert(20, Action::Shift(3, false));
        action[2].insert(21, Action::Shift(4, false));
        action[3].insert(0, Action::Shift(5, true));
        action[4].insert(0, Action::Shift(5, true));
        action[5].insert(1, Action::Shift(5, true));
        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 6],
            num_states: 6,
            num_terminals: 22,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::ExactSimulation,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();
        let baseline = table.clone();
        let mapping = table.share_context_distinguishable_states_exact(&[vec![3, 4]]);

        assert_eq!(mapping[3], mapping[4]);
        assert!(table.num_states < baseline.num_states);

        let acc = TerminalsDisallowed::new();
        let baseline_before = ParserGSS::from_stacks(&[
            (vec![0, 1, 3], acc.clone()),
            (vec![0, 2, 4], acc.clone()),
        ]);
        let shared_before = ParserGSS::from_stacks(&[
            (
                vec![mapping[0], mapping[1], mapping[3]],
                acc.clone(),
            ),
            (vec![mapping[0], mapping[2], mapping[4]], acc),
        ]);
        assert_eq!(baseline_before.peek_values().len(), 2);
        assert_eq!(shared_before.peek_values().len(), 1);

        let baseline_after = advance_stacks(&baseline, &baseline_before, 0);
        let shared_after = advance_stacks(&table, &shared_before, 0);
        let mut expected = baseline_after
            .to_stacks(32)
            .unwrap()
            .into_iter()
            .map(|(stack, acc)| {
                (
                    stack
                        .into_iter()
                        .map(|state| mapping[state as usize])
                        .collect::<Vec<_>>(),
                    acc,
                )
            })
            .collect::<Vec<_>>();
        let mut actual = shared_after.to_stacks(32).unwrap();
        expected.sort_by(|left, right| left.0.cmp(&right.0));
        actual.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(actual, expected);
    }

    #[test]
    fn state_local_stack_effect_exhaustion_does_not_abort_parent_budget() {
        let parent = default_unit_inline_budget();
        let child = parent.child_with_stack_effect_visit_limit(2);

        assert!(child.record_stack_effect_visit());
        assert!(child.record_stack_effect_visit());
        assert!(!child.record_stack_effect_visit());
        assert_eq!(
            child.abort_code.load(std::sync::atomic::Ordering::Relaxed),
            ABORT_STACK_EFFECT_VISITS,
        );

        parent.fold_in_isolated_state(&child);

        assert_eq!(parent.stack_effect_visits(), 3);
        assert!(!parent.is_aborted());
    }

    #[test]
    fn state_local_work_still_counts_towards_global_stack_effect_limit() {
        let mut parent = default_unit_inline_budget();
        parent.max_stack_effect_visits = 4;

        let first = parent.child_with_stack_effect_visit_limit(2);
        assert!(first.record_stack_effect_visit());
        assert!(first.record_stack_effect_visit());
        assert!(!first.record_stack_effect_visit());
        parent.fold_in_isolated_state(&first);
        assert!(!parent.is_aborted());

        let second = parent.child_with_stack_effect_visit_limit(2);
        assert!(second.record_stack_effect_visit());
        assert!(second.record_stack_effect_visit());
        parent.fold_in_isolated_state(&second);

        assert!(parent.is_aborted());
        assert_eq!(parent.report().reason, Some("stack_effect_visits"));
    }

    #[test]
    fn core_merged_stack_effect_lowering_accepts_only_unguarded_effects() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(MAX_GUARDED_STACK_EFFECTS_ENV, "1000000");
        let mut table = table_with_stack_shifts(Vec::new(), &[]);
        table.construction = GlrTableConstruction::ExperimentalCoreMerged;

        let unguarded = vec![GuardedStackShift {
            guards: Vec::new(),
            pop: 2,
            pushes: vec![2],
        }];
        assert!(matches!(
            stack_effect_action(&table, unguarded),
            Some(Action::StackShifts(shifts)) if shifts.len() == 1
        ));

        assert!(stack_effect_action(&table, vec![guarded_effect(1, 1)]).is_none());
    }

    #[test]
    fn legacy_stack_effect_lowering_still_accepts_guarded_effects() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(MAX_GUARDED_STACK_EFFECTS_ENV, "1000000");
        let table = table_with_stack_shifts(Vec::new(), &[]);

        assert!(matches!(
            stack_effect_action(&table, vec![guarded_effect(1, 1)]),
            Some(Action::GuardedStackShifts(shifts)) if shifts.len() == 1
        ));
    }

    #[test]
    fn row_fingerprint_is_independent_of_sparse_row_insertion_order() {
        let mut action_left = ActionRow::default();
        action_left.insert(3, Action::Reduce(7, 1));
        action_left.insert(1, Action::Shift(4, false));
        action_left.insert(9, Action::Accept);
        let mut action_right = ActionRow::default();
        action_right.insert(9, Action::Accept);
        action_right.insert(1, Action::Shift(4, false));
        action_right.insert(3, Action::Reduce(7, 1));

        let mut goto_left = GotoRow::default();
        goto_left.insert(5, (2, false));
        goto_left.insert(1, (6, true));
        let mut goto_right = GotoRow::default();
        goto_right.insert(1, (6, true));
        goto_right.insert(5, (2, false));

        assert!(rows_equal(
            &action_left,
            &goto_left,
            None,
            &action_right,
            &goto_right,
            None,
        ));
        assert_eq!(
            row_fingerprint(&action_left, &goto_left, None),
            row_fingerprint(&action_right, &goto_right, None),
        );
    }

    #[test]
    fn incremental_row_merge_matches_full_quotient_on_small_generated_tables() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        fn action(seed: &mut u64, nstates: u32) -> Action {
            match next(seed) % 5 {
                0 => Action::Shift((next(seed) % nstates as u64) as u32, next(seed) & 1 != 0),
                1 => Action::Reduce((next(seed) % 3) as u32, (next(seed) % 3) as u32),
                2 => Action::Split {
                    shift: (next(seed) & 1 != 0)
                        .then(|| ((next(seed) % nstates as u64) as u32, next(seed) & 1 != 0)),
                    reduces: vec![((next(seed) % 3) as u32, (next(seed) % 3) as u32)],
                    accept: next(seed) & 1 != 0,
                },
                3 => Action::StackShifts(vec![
                    StackShift {
                        pop: (next(seed) % 3) as u32,
                        pushes: vec![(next(seed) % nstates as u64) as u32],
                    },
                    StackShift {
                        pop: (next(seed) % 3) as u32,
                        pushes: vec![(next(seed) % nstates as u64) as u32],
                    },
                ]),
                _ => Action::Accept,
            }
        }

        let mut seed = 0xA3B1_C2D3_E4F5_6789u64;
        for _case in 0..128 {
            let nstates = 6usize;
            let mut action_rows = vec![ActionRow::default(); nstates];
            let mut goto_rows = vec![GotoRow::default(); nstates];
            for state in 0..nstates {
                for terminal in 0..3 {
                    if next(&mut seed) % 4 != 0 {
                        action_rows[state].insert(terminal, action(&mut seed, nstates as u32));
                    }
                }
                for nonterminal in 0..3 {
                    if next(&mut seed) & 1 != 0 {
                        goto_rows[state].insert(
                            nonterminal,
                            ((next(&mut seed) % nstates as u64) as u32, next(&mut seed) & 1 != 0),
                        );
                    }
                }
            }

            let mut base = GLRTable {
                action: action_rows,
                goto: goto_rows,
                num_states: nstates as u32,
                num_terminals: 3,
                num_rules: 0,
                rules: Vec::new(),
                nonterminal_display_names: Vec::new(),
                construction: GlrTableConstruction::LegacyRowBisim,
                admission_policy: AdmissionPolicy::RowPresenceExact,
                advance: Vec::new(),
                unconditional_advance: Vec::new(),
                forwarded_shifts: FxHashSet::default(),
                control_terminals: Default::default(),
                skip_terminals: Default::default(),
                guarded_shift_index: Vec::new(),
                direct_regular_wide_frontiers: Vec::new(),
            };
            base.merge_identical_rows();
            if base.num_states == 0 {
                continue;
            }

            let dirty = (next(&mut seed) % base.num_states as u64) as u32;
            let terminal = (next(&mut seed) % 3) as u32;
            base.action[dirty as usize].insert(terminal, action(&mut seed, base.num_states));

            let mut expected = base.clone();
            expected.merge_identical_rows();
            let mut incremental = base;
            incremental.merge_identical_rows_from_dirty(&[dirty]);

            assert_eq!(incremental.num_states, expected.num_states);
            assert_eq!(incremental.action, expected.action);
            assert_eq!(incremental.goto, expected.goto);
            assert_eq!(incremental.advance, expected.advance);
            assert_eq!(incremental.forwarded_shifts, expected.forwarded_shifts);
        }
    }

    #[test]
    fn incremental_row_merge_matches_full_quotient_for_noncanonical_stack_shifts() {
        let mut action = vec![ActionRow::default(); 4];
        action[0].insert(
            0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 2,
                    pushes: vec![2],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![3],
                },
            ]),
        );
        // Initially distinct from state 0, so the starting table is already
        // fully exact-minimized before the local mutation below.
        action[1].insert(
            0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![3],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![3],
                },
            ]),
        );
        action[2].insert(0, Action::Shift(2, false));
        action[3].insert(0, Action::Accept);

        let table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 4],
            num_states: 4,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };

        let mut base = table.clone();
        base.merge_identical_rows();
        assert_eq!(base.num_states, 4);

        // Stack-effect equality remaps and normalizes alternatives. The two
        // rows are exact equals under the identity map, while their direct
        // Action hashes differ because their alternative order differs.
        base.action[1].insert(
            0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![3],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![2],
                },
            ]),
        );
        let identity = (0..base.num_states).collect::<Vec<_>>();
        assert!(rows_equal_after_remap(
            &base.action[0],
            &base.goto[0],
            None,
            &base.action[1],
            &base.goto[1],
            None,
            &identity,
        ));
        assert_ne!(
            row_fingerprint(&base.action[0], &base.goto[0], None),
            row_fingerprint(&base.action[1], &base.goto[1], None),
        );

        let mut expected = base.clone();
        expected.merge_identical_rows();
        assert_eq!(expected.num_states, 3);

        let mut incremental = base;
        assert!(incremental.merge_identical_rows_from_dirty(&[1]));
        assert_eq!(incremental.num_states, expected.num_states);
        assert_eq!(incremental.action, expected.action);
        assert_eq!(incremental.goto, expected.goto);
        assert_eq!(incremental.advance, expected.advance);
        assert_eq!(incremental.forwarded_shifts, expected.forwarded_shifts);
    }

    #[test]
    fn unit_reduction_inlining_budget_abort_keeps_original_table() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV, "1");

        let mut action = vec![ActionRow::default(); 5];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, false)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(4, false));

        let mut goto = vec![GotoRow::default(); 5];
        goto[1].insert(10, (3, true));

        let mut table = GLRTable {
            action,
            goto,
            num_states: 5,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let original_action = format!("{:?}", table.action);
        let original_goto = format!("{:?}", table.goto);
        let original_num_states = table.num_states;

        let report = table.collapse_sr_unit_reductions_with_compatible_gotos();

        assert!(report.aborted);
        assert_eq!(report.reason, Some("stack_effect_visits"));
        assert_eq!(table.num_states, original_num_states);
        assert_eq!(format!("{:?}", table.action), original_action);
        assert_eq!(format!("{:?}", table.goto), original_goto);
    }

    #[test]
    fn canonicalizes_stack_shift_predecessor_to_goto_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            }]
        );
    }

    #[test]
    fn leaves_protected_stack_shift_terminal_unchanged() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );
        let mut protected = BitSet::new(table.num_terminals as usize);
        protected.set(0);

        table.canonicalize_stack_shift_predecessors_except(&protected);

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn leaves_stack_shift_predecessors_unchanged_when_canonicalization_is_disabled() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors_with_enabled(
            false,
            &BitSet::new(table.num_terminals as usize),
        );

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn does_not_canonicalize_stack_shift_predecessors_when_shared_goto_target_differs() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (22, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn does_not_canonicalize_empty_goto_row_to_nonempty_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[(1, &[(10, (20, true))])],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn canonicalizes_buried_middle_stack_shift_predecessor_to_goto_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![StackShift {
                pop: 1,
                pushes: vec![9, 1, 3, 4],
            }]
        );
    }

    #[test]
    fn does_not_canonicalize_top_pushed_state_even_when_goto_rows_are_compatible() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 1],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 2],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 1],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 2],
                },
            ]
        );
    }

    #[test]
    fn reduce_frame_allows_origin_dependent_multiple_goto_targets() {
        let mut table = table_with_stack_shifts(Vec::new(), &[
            (1, &[(10, (3, false))]),
            (2, &[(10, (4, false))]),
        ]);
        table.num_states = 6;
        table.action.resize(6, ActionRow::default());
        table.goto.resize(6, GotoRow::default());

        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[5] = smallvec![1, 2];
        let budget = default_unit_inline_budget();

        let result = apply_reduce_to_frame(
            &table,
            &predecessors,
            5,
            StackEffectFrame {
                pop: 0,
                pushes: Vec::new(),
                guards: Vec::new(),
            },
            10,
            1,
            &mut FxHashMap::default(),
            &budget,
        );

        let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
            panic!("expected frames");
        };
        assert!(origin_dependent);
        assert_eq!(
            frames,
            vec![
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![3],
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![1],
                    }],
                },
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![4],
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![2],
                    }],
                },
            ]
        );
    }

    #[test]
    fn reduce_frame_allows_origin_dependent_single_goto_target() {
        let mut table = table_with_stack_shifts(Vec::new(), &[
            (1, &[(10, (3, false))]),
            (2, &[(10, (3, false))]),
        ]);
        table.num_states = 6;
        table.action.resize(6, ActionRow::default());
        table.goto.resize(6, GotoRow::default());

        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[5] = smallvec![1, 2];
        let budget = default_unit_inline_budget();

        let result = apply_reduce_to_frame(
            &table,
            &predecessors,
            5,
            StackEffectFrame {
                pop: 0,
                pushes: Vec::new(),
                guards: Vec::new(),
            },
            10,
            1,
            &mut FxHashMap::default(),
            &budget,
        );

        let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
            panic!("expected frames");
        };
        assert!(origin_dependent);
        assert_eq!(
            frames,
            vec![
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![3],
                    guards: Vec::new(),
                }
            ]
        );
    }

    #[test]
    fn in_place_action_target_remap_matches_copy_remap() {
        let mapping = vec![3, 0, 2, 1, 4, 5];
        let actions = vec![
            Action::Shift(1, false),
            Action::Reduce(7, 2),
            Action::Split {
                shift: Some((3, true)),
                reduces: vec![(4, 1), (5, 2)],
                accept: false,
            },
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![0, 3],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![0, 1],
                },
            ]),
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![0, 3],
                }],
                pop: 2,
                pushes: vec![1, 4],
            }]),
            Action::Accept,
        ];
        for action in actions {
            let expected = remap_action_targets(&action, &mapping);
            let mut actual = action;
            remap_action_targets_in_place(&mut actual, &mapping);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn trivial_origin_dependent_one_push_shortcut_matches_generic_refusal() {
        let mut action = vec![ActionRow::default(); 6];
        action[3].insert(0, Action::Shift(5, true));
        action[4].insert(0, Action::Shift(5, true));
        let mut goto = vec![GotoRow::default(); 6];
        goto[1].insert(10, (3, false));
        goto[2].insert(10, (4, false));
        let table = GLRTable {
            action,
            goto,
            num_states: 6,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[0] = smallvec![1, 2];
        let reduce = Action::Reduce(10, 1);
        let mut shortcut_reads = Vec::new();
        assert!(is_trivial_origin_dependent_one_push(
            &table,
            &predecessors,
            0,
            0,
            &reduce,
            &mut shortcut_reads,
        ));
        assert_eq!(shortcut_reads, vec![((0, 0), (3, 0)), ((0, 0), (4, 0))]);

        let budget = default_unit_inline_budget();
        let generic = try_inline_action_to_stack_shifts_detailed(
            &table,
            &predecessors,
            0,
            0,
            &reduce,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );
        assert!(generic.action.is_none());
        assert!(matches!(
            generic.outcome,
            StackInlineOutcomeKind::OriginDependentOnePush
        ));
    }

    #[test]
    fn trivial_origin_dependent_one_push_shortcut_rejects_nonreplace_continuation() {
        let mut action = vec![ActionRow::default(); 4];
        action[3].insert(0, Action::Shift(3, false));
        let mut goto = vec![GotoRow::default(); 4];
        goto[1].insert(10, (3, false));
        let table = GLRTable {
            action,
            goto,
            num_states: 4,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 4];
        predecessors[0].push(1);
        assert!(!is_trivial_origin_dependent_one_push(
            &table,
            &predecessors,
            0,
            0,
            &Action::Reduce(10, 1),
            &mut Vec::new(),
        ));
    }

    #[test]
    fn inline_action_to_stack_shifts_keeps_multishift_replacement_reduce_chain() {
        let mut action = vec![ActionRow::default(); 5];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, false)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(4, false));

        let mut goto = vec![GotoRow::default(); 5];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 5,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 5];
        predecessors[2].push(1);
        let budget = default_unit_inline_budget();

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );

        let Some(Action::StackShifts(shifts)) = result else {
            panic!("expected multi-stack-shift action, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                StackShift {
                    pop: 0,
                    pushes: vec![4],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![3, 4],
                },
            ]
        );
    }

    #[test]
    fn inline_action_to_stack_shifts_handles_replace_shift_and_replace_goto() {
        let mut action = vec![ActionRow::default(); 6];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, true)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(5, true));

        let mut goto = vec![GotoRow::default(); 6];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 6,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[2].push(1);
        let budget = default_unit_inline_budget();

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );

        let Some(Action::StackShifts(shifts)) = result else {
            panic!("expected replacement stack shifts, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![4],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![5],
                },
            ]
        );
    }

    #[test]
    fn inline_action_to_stack_shifts_guards_divergent_replace_gotos_by_predecessor() {
        let mut action = vec![ActionRow::default(); 9];
        action[2].insert(0, Action::Reduce(10, 1));
        action[3].insert(0, Action::Shift(7, false));
        action[4].insert(0, Action::Shift(8, false));

        let mut goto = vec![GotoRow::default(); 9];
        goto[1].insert(10, (3, true));
        goto[6].insert(10, (4, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 9,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 9];
        predecessors[2].extend_from_slice(&[1, 6]);
        let budget = default_unit_inline_budget();

        let action = table.action(2, 0).expect("expected reduce action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );

        let Some(Action::GuardedStackShifts(shifts)) = result else {
            panic!("expected guarded replacement stack shifts, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![1],
                    }],
                    pop: 2,
                    pushes: vec![3, 7],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![6],
                    }],
                    pop: 2,
                    pushes: vec![4, 8],
                },
            ]
        );
    }

    #[test]
    fn unit_reduce_destination_without_predecessors_is_not_inlineable() {
        let table = GLRTable {
            action: vec![ActionRow::default(); 2],
            goto: vec![GotoRow::default(); 2],
            num_states: 2,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::ExperimentalCoreMerged,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let predecessors = vec![PredecessorSet::new(); 2];

        assert_eq!(unit_reduce_destination(&table, &predecessors, 1, 10), None);
    }

    #[test]
    fn compatible_goto_unit_destination_still_refuses_replace_goto() {
        let action = vec![ActionRow::default(); 4];
        let mut goto = vec![GotoRow::default(); 4];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 4,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 4];
        predecessors[2].push(1);

        assert_eq!(unit_reduce_destination(&table, &predecessors, 2, 10), None);
    }

    #[test]
    fn suffix_quotient_collapses_same_pop_stack_shift_fanout() {
        let token0 = 0;
        let token1 = 1;
        let mut action = vec![ActionRow::default(); 8];
        action[0].insert(
            token0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 2],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![3, 4],
                },
            ]),
        );
        action[2].insert(
            token1,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![5],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![6],
                },
            ]),
        );
        action[4].insert(
            token1,
            Action::StackShifts(vec![StackShift {
                pop: 2,
                pushes: vec![7],
            }]),
        );

        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 8],
            num_states: 8,
            num_terminals: 2,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        assert!(matches!(table.action(0, token0), Some(Action::Shift(_, true))));
        assert!(
            table.ambiguous_actions().is_empty(),
            "{:#?}",
            table.ambiguous_actions()
        );
    }


    #[test]
    fn suffix_quotient_builds_synthetic_rows_through_reductions() {
        let produce = 0;
        let consume = 1;
        let mut action = vec![ActionRow::default(); 9];
        action[0].insert(
            produce,
            Action::StackShifts(vec![
                StackShift { pop: 0, pushes: vec![1] },
                StackShift { pop: 0, pushes: vec![2] },
            ]),
        );
        action[1].insert(consume, Action::Reduce(10, 1));
        action[2].insert(consume, Action::Reduce(11, 1));
        action[3].insert(consume, Action::Shift(7, false));
        action[4].insert(consume, Action::Shift(8, false));

        let mut goto = vec![GotoRow::default(); 9];
        goto[0].insert(10, (3, false));
        goto[0].insert(11, (4, false));

        let mut table = GLRTable {
            action,
            goto,
            num_states: 9,
            num_terminals: 2,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        let synthetic = match table.action(0, produce) {
            Some(Action::StackShifts(producer_shifts)) => {
                assert_eq!(producer_shifts.len(), 1);
                assert_eq!(producer_shifts[0].pop, 0);
                assert_eq!(producer_shifts[0].pushes.len(), 1);
                producer_shifts[0].pushes[0]
            }
            Some(Action::Shift(target, replace)) => {
                assert!(!replace);
                *target
            }
            other => panic!("expected producer to be rewritten to one target action, got {other:?}"),
        };
        let Some(Action::GuardedStackShifts(shifts)) = table.action(synthetic, consume) else {
            panic!("expected synthetic row to compile reductions into guarded stack effects");
        };
        assert!(!shifts.is_empty());
        assert!(shifts.iter().all(|shift| shift.pop == 1));
        assert!(shifts.iter().all(|shift| shift.guards.len() == 1));
        assert!(shifts.iter().all(|shift| shift.guards[0].pop == 1));
        assert!(shifts.iter().all(|shift| shift.guards[0].states == vec![0]));
    }

    #[test]
    fn suffix_quotient_preserves_guarded_stack_shift_guards() {
        let token = 0;
        let guard = StackShiftGuard {
            pop: 1,
            states: vec![9],
        };
        let mut action = vec![ActionRow::default(); 12];
        action[0].insert(
            token,
            Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![guard.clone()],
                    pop: 1,
                    pushes: vec![1, 2],
                },
                GuardedStackShift {
                    guards: vec![guard.clone()],
                    pop: 1,
                    pushes: vec![3, 4],
                },
            ]),
        );

        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 12],
            num_states: 12,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        let Some(Action::GuardedStackShifts(shifts)) = table.action(0, token) else {
            panic!("expected one guarded stack-shift action");
        };
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0].guards.len(), 1);
        assert_eq!(shifts[0].guards[0].pop, guard.pop);
        assert!(!shifts[0].guards[0].states.is_empty());
        assert_eq!(shifts[0].pop, 1);
        assert_eq!(shifts[0].pushes.len(), 1);
        assert!(
            table.ambiguous_actions().is_empty(),
            "{:#?}",
            table.ambiguous_actions()
        );
    }

    #[test]
    fn suffix_quotient_rolls_back_when_nested_construction_hits_depth_bound() {
        let outer_suffixes = vec![vec![10, 1], vec![10, 2]];

        let mut table = GLRTable {
            action: vec![ActionRow::default(); 11],
            goto: vec![GotoRow::default(); 11],
            num_states: 11,
            num_terminals: 0,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.goto[1].insert(0, (3, false));
        table.goto[2].insert(0, (4, false));
        table.rebuild_advance_rows_from_actions();

        let original_num_states = table.num_states;
        let original_action = table.action.clone();
        let original_goto = table.goto.clone();
        let original_advance = table.advance.clone();

        let mut quotient = SuffixQuotient {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            max_states: 4096,
            max_alts: 8,
            max_width: 8,
            max_depth: 1,
            build_depth: 0,
            created_states: 0,
        };

        assert_eq!(
            quotient.ensure_suffix_state(&mut table, outer_suffixes.clone()),
            Err(())
        );
        assert_eq!(table.num_states, original_num_states);
        assert_eq!(table.action, original_action);
        assert_eq!(table.goto, original_goto);
        assert_eq!(table.advance, original_advance);
        assert_eq!(quotient.created_states, 0);
        assert_eq!(quotient.build_depth, 0);
        assert!(quotient.failed_suffixes.contains(&outer_suffixes));
        assert!(quotient
            .suffix_to_state
            .values()
            .all(|&state| state < original_num_states));
    }

    #[test]
    fn suffix_quotient_rolls_back_nested_created_states_on_outer_failure() {
        let outer_suffixes = vec![vec![10, 1], vec![10, 2]];

        let mut table = GLRTable {
            action: vec![ActionRow::default(); 11],
            goto: vec![GotoRow::default(); 11],
            num_states: 11,
            num_terminals: 0,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.goto[1].insert(0, (3, false));
        table.goto[2].insert(0, (4, false));
        table.goto[1].insert(1, (5, false));
        table.goto[2].insert(1, (6, false));
        table.rebuild_advance_rows_from_actions();

        let original_num_states = table.num_states;
        let original_action_len = table.action.len();
        let original_goto_len = table.goto.len();
        let original_advance_len = table.advance.len();

        let mut quotient = SuffixQuotient {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            max_states: 2,
            max_alts: 8,
            max_width: 8,
            max_depth: 128,
            build_depth: 0,
            created_states: 0,
        };

        assert_eq!(
            quotient.ensure_suffix_state(&mut table, outer_suffixes.clone()),
            Err(())
        );
        assert_eq!(table.num_states, original_num_states);
        assert_eq!(table.action.len(), original_action_len);
        assert_eq!(table.goto.len(), original_goto_len);
        assert_eq!(table.advance.len(), original_advance_len);
        assert_eq!(quotient.created_states, 0);
        assert!(quotient.failed_suffixes.contains(&outer_suffixes));
        assert!(
            quotient
                .suffix_to_state
                .values()
                .all(|&state| state < original_num_states)
        );
    }
}

fn try_inline_unit_reductions_for_cell_inner(
    table: &mut GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    visiting: &mut BTreeSet<(u32, TerminalID)>,
    owner: UnitInlineCell,
    dependencies: &mut UnitInlineDependencies,
    budget: &UnitInlineBudget,
) -> Result<Option<CellUpdate>, ()> {
    if !budget.record_stack_effect_visit() {
        return Err(());
    }
    if !visiting.insert((state, tid)) {
        return Ok(None);
    }

    let mut pending = PendingAction::default();
    let mut reduces: Vec<(NonterminalID, u32)> = Vec::new();

    match action {
        Action::Shift(target, replace) => pending.push_shift(*target, *replace),
        Action::StackShifts(_) | Action::ReplaceShifts(_) => return Ok(None),
        Action::GuardedStackShifts(_) | Action::Skip => return Ok(None),
        Action::Reduce(nt, len) => reduces.push((*nt, *len)),
        Action::Split {
            shift,
            reduces: action_reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                pending.push_shift(*target, *replace);
            }
            reduces.extend(action_reduces.iter().copied());
            if *accept {
                pending.push_accept();
            }
        }
        Action::Accept => pending.push_accept(),
    }

    let mut changed = false;
    for (lhs, pop_len) in reduces {
        if pop_len != 1 {
            pending.push_reduce(lhs, pop_len);
            continue;
        }

        let Some(reduce_dst) = unit_reduce_destination(table, predecessors, state, lhs) else {
            pending.push_reduce(lhs, pop_len);
            continue;
        };

        dependencies.record_read(owner, (reduce_dst, tid));
        match table.action[reduce_dst as usize].get(&tid).cloned() {
            None => {
                pending.push_reduce(lhs, pop_len);
            }
            Some(inline_action) => {
                let resolved_inline = match try_inline_unit_reductions_for_cell_inner(
                    table,
                    predecessors,
                    reduce_dst,
                    tid,
                    &inline_action,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    visiting,
                    owner,
                    dependencies,
                    budget,
                )? {
                    Some(CellUpdate::Set(action)) => Some(action),
                    Some(CellUpdate::Remove) => None,
                    None => Some(inline_action),
                };

                let Some(resolved_inline) = resolved_inline else {
                    changed = true;
                    continue;
                };

                merge_action_into_pending(
                    &mut pending,
                    &resolved_inline,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    budget,
                )?;
                changed = true;
            }
        }
    }

    let result = if !changed {
        Ok(None)
    } else {
        Ok(match pending.maybe_finish() {
            Some(action) => Some(CellUpdate::Set(action)),
            None => Some(CellUpdate::Remove),
        })
    };
    visiting.remove(&(state, tid));
    result
}

fn remap_action_row_targets_in_place(action_row: &mut ActionRow, mapping: &[u32]) {
    action_row.for_each_value_mut(|action| remap_action_targets_in_place(action, mapping));
}

/// Remap action targets without cloning reductions, shift lists, or guarded
/// alternatives. This is used while compacting a quotient; it preserves the
/// exact normalization performed by `remap_action_targets` for actions whose
/// target IDs can collide after a merge.
fn remap_action_targets_in_place(action: &mut Action, mapping: &[u32]) {
    match action {
        Action::Shift(target, _) => {
            *target = mapping[*target as usize];
        }
        Action::ReplaceShifts(targets) => {
            let mut remapped = targets
                .iter()
                .map(|&target| mapping[target as usize])
                .collect::<Vec<_>>();
            remapped.sort_unstable();
            remapped.dedup();
            *targets = remapped.into();
        }
        Action::StackShifts(shifts) => {
            for shift in shifts.iter_mut() {
                for target in &mut shift.pushes {
                    *target = mapping[*target as usize];
                }
            }
            normalize_stack_shifts(shifts);
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts.iter_mut() {
                for guard in shift.guards.iter_mut() {
                    for state in &mut guard.states {
                        *state = mapping[*state as usize];
                    }
                    guard.states.sort_unstable();
                    guard.states.dedup();
                }
                for target in &mut shift.pushes {
                    *target = mapping[*target as usize];
                }
            }
        }
        Action::Reduce(_, _) | Action::Accept | Action::Skip => {}
        Action::Split { shift, .. } => {
            if let Some((target, _)) = shift {
                *target = mapping[*target as usize];
            }
        }
    }
}

fn remap_goto_row_targets_in_place(goto_row: &mut GotoRow, mapping: &[u32]) {
    goto_row.for_each_value_mut(|(target, _)| {
        *target = mapping[*target as usize];
    });
}

fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {
    match action {
        Action::Shift(target, replace) => Action::Shift(mapping[*target as usize], *replace),
        Action::ReplaceShifts(targets) => {
            let mut targets = targets
                .iter()
                .map(|&target| mapping[target as usize])
                .collect::<Vec<_>>();
            targets.sort_unstable();
            targets.dedup();
            Action::ReplaceShifts(targets.into())
        }
        Action::StackShifts(shifts) => {
            let mut remapped = shifts
                .iter()
                .map(|shift| StackShift {
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| mapping[state as usize]).collect(),
                })
                .collect();
            normalize_stack_shifts(&mut remapped);
            Action::StackShifts(remapped)
        }
        Action::GuardedStackShifts(shifts) => Action::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| GuardedStackShift {
                    guards: shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states: Vec<u32> = guard
                                .states
                                .iter()
                                .map(|&state| mapping[state as usize])
                                .collect();
                            states.sort_unstable();
                            states.dedup();
                            StackShiftGuard {
                                pop: guard.pop,
                                states,
                            }
                        })
                        .collect(),
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| mapping[state as usize]).collect(),
                })
                .collect(),
        ),
        Action::Reduce(nt, len) => Action::Reduce(*nt, *len),
        Action::Split {
            shift,
            reduces,
            accept,
        } => Action::Split {
            shift: shift.map(|(target, replace)| (mapping[target as usize], replace)),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => Action::Accept,
        Action::Skip => Action::Skip,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ActionSig {
    Shift(u32, bool),
    ReplaceShifts(Vec<u32>),
    StackShifts(Vec<(u32, Vec<u32>)>),
    GuardedStackShifts(Vec<(Vec<(u32, Vec<u32>)>, u32, Vec<u32>)>),
    Reduce(NonterminalID, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(NonterminalID, u32)>,
        accept: bool,
    },
    Accept,
    Skip,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowSignature {
    core_class: u32,
    action: Vec<(TerminalID, ActionSig)>,
    goto: Vec<(NonterminalID, (u32, bool))>,
}

fn remap_action_to_partition(action: &Action, partition: &[u32]) -> ActionSig {
    match action {
        Action::Shift(target, replace) => ActionSig::Shift(partition[*target as usize], *replace),
        Action::ReplaceShifts(targets) => {
            let mut targets = targets
                .iter()
                .map(|&target| partition[target as usize])
                .collect::<Vec<_>>();
            targets.sort_unstable();
            targets.dedup();
            ActionSig::ReplaceShifts(targets)
        }
        Action::StackShifts(shifts) => ActionSig::StackShifts(
            shifts
                .iter()
                .map(|shift| {
                    (
                        shift.pop,
                        shift.pushes.iter().map(|&state| partition[state as usize]).collect(),
                    )
                })
                .collect(),
        ),
        Action::GuardedStackShifts(shifts) => ActionSig::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| {
                    let guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states: Vec<u32> = guard
                                .states
                                .iter()
                                .map(|&state| partition[state as usize])
                                .collect();
                            states.sort_unstable();
                            states.dedup();
                            (guard.pop, states)
                        })
                        .collect();
                    let pushes = shift
                        .pushes
                        .iter()
                        .map(|&state| partition[state as usize])
                        .collect();
                    (guards, shift.pop, pushes)
                })
                .collect(),
        ),
        Action::Reduce(nt, len) => ActionSig::Reduce(*nt, *len),
        Action::Split {
            shift,
            reduces,
            accept,
        } => ActionSig::Split {
            shift: shift.map(|(target, replace)| (partition[target as usize], replace)),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => ActionSig::Accept,
        Action::Skip => ActionSig::Skip,
    }
}

fn core_classes(core_keys: &[Vec<Item>]) -> Vec<u32> {
    // Use the first state in each LR(0)-core class as its label. These labels
    // are sparse, but they let refinement touch only non-singleton classes;
    // an overwhelmingly LR(0)-unique grammar such as BFCL catalog-512 should
    // not build and hash row signatures for every singleton state.
    let mut class_of = Vec::with_capacity(core_keys.len());
    let mut key_to_representative: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    for (state, key) in core_keys.iter().enumerate() {
        let representative = *key_to_representative
            .entry(key.clone())
            .or_insert(state as u32);
        class_of.push(representative);
    }
    class_of
}

fn nontrivial_partition_members(partition: &[u32]) -> FxHashMap<u32, Vec<usize>> {
    let mut members = FxHashMap::<u32, Vec<usize>>::default();
    for (state, &representative) in partition.iter().enumerate() {
        if representative == state as u32 {
            continue;
        }
        members
            .entry(representative)
            .or_insert_with(|| vec![representative as usize])
            .push(state);
    }
    members
}

fn refine_same_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {
    let profile_detail = std::env::var("GLRMASK_PROFILE_GLR_CORE_MERGE_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false);
    let core_classes_started_at = profile_detail.then(std::time::Instant::now);
    let nstates = table.num_states as usize;
    let has_advance_rows = table.advance.len() == nstates;
    let core_class_of = core_classes(core_keys);
    let core_classes_ms = core_classes_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let mut partition = core_class_of.clone();
    let mut iteration = 0usize;

    loop {
        let members_by_class = nontrivial_partition_members(&partition);
        if members_by_class.is_empty() {
            return partition;
        }
        iteration += 1;
        let iteration_started_at = profile_detail.then(std::time::Instant::now);
        let mut active_states = members_by_class
            .values()
            .flat_map(|members| members.iter().copied())
            .collect::<Vec<_>>();
        active_states.sort_unstable();

        // The initial core partition forbids every singleton state from
        // merging. Signatures are needed only for the small set of members in
        // still-nontrivial core classes. Retain parallel work for genuinely
        // ambiguous grammars, but avoid Rayon setup on the common tiny set.
        let make_signature = |&state: &usize| {
            let action = table.action[state]
                .iter()
                .map(|(terminal, action)| {
                    (terminal, remap_action_to_partition(action, &partition))
                })
                .collect();
            let goto = table.goto[state]
                .iter()
                .map(|(&nt, &(target, replace))| (nt, (partition[target as usize], replace)))
                .collect();
            let signature = RowSignature {
                core_class: core_class_of[state],
                action,
                goto,
            };
            let mut hasher = FxHasher::default();
            signature.hash(&mut hasher);
            if has_advance_rows {
                table.advance[state].hash(&mut hasher);
            }
            (state, hasher.finish(), signature)
        };
        let signatures = if active_states.len() >= 256 {
            active_states
                .par_iter()
                .map(make_signature)
                .collect::<Vec<_>>()
        } else {
            active_states.iter().map(make_signature).collect::<Vec<_>>()
        };

        let mut fingerprint_to_classes: FxHashMap<u64, Vec<(usize, u32, RowSignature)>> =
            FxHashMap::default();
        let mut next_partition = partition.clone();
        let mut action_entries = 0usize;
        let mut goto_entries = 0usize;
        for (state, fingerprint, signature) in signatures {
            action_entries += table.action[state].len();
            goto_entries += table.goto[state].len();
            let bucket = fingerprint_to_classes.entry(fingerprint).or_default();
            let mut class = None;
            for (rep_state, representative, rep_signature) in bucket.iter() {
                if core_class_of[*rep_state] == core_class_of[state]
                    && rep_signature == &signature
                    && (!has_advance_rows || table.advance[*rep_state] == table.advance[state])
                {
                    class = Some(*representative);
                    break;
                }
            }
            let class = class.unwrap_or_else(|| {
                let representative = state as u32;
                bucket.push((state, representative, signature));
                representative
            });
            next_partition[state] = class;
        }

        let changed = next_partition != partition;
        if let Some(iteration_started_at) = iteration_started_at {
            let unique_partitions = nstates - nontrivial_partition_members(&next_partition)
                .values()
                .map(|members| members.len() - 1)
                .sum::<usize>();
            eprintln!(
                "[glrmask/profile][same_core_refine] iteration={} states={} active_states={} core_classes_ms={:.3} unique_partitions={} action_entries={} goto_entries={} changed={} elapsed_ms={:.3}",
                iteration,
                nstates,
                active_states.len(),
                core_classes_ms,
                unique_partitions,
                action_entries,
                goto_entries,
                changed,
                iteration_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if !changed {
            return partition;
        }
        partition = next_partition;
    }
}

pub(super) fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {
    let partition = refine_same_core_partition(&table, core_keys);
    let nstates = table.num_states as usize;

    // Sparse representative labels keep the refinement cheap. Compact them in
    // first-source-state order only when a real quotient is needed; this is
    // the same canonical ordering that the prior dense partition produced.
    let mut group_to_new: FxHashMap<u32, u32> = FxHashMap::default();
    let mut representatives = Vec::with_capacity(nstates);
    let mut mapping = Vec::with_capacity(nstates);
    for (state, &group) in partition.iter().enumerate() {
        let new_group = match group_to_new.get(&group) {
            Some(&existing) => existing,
            None => {
                let next = representatives.len() as u32;
                group_to_new.insert(group, next);
                representatives.push(state as u32);
                next
            }
        };
        mapping.push(new_group);
    }
    if representatives.len() == nstates {
        return table;
    }

    let action = representatives
        .iter()
        .map(|&rep| {
            table.action[rep as usize]
                .iter()
                .map(|(terminal, action)| (terminal, remap_action_targets(action, &mapping)))
                .collect()
        })
        .collect();
    let goto = representatives
        .iter()
        .map(|&rep| {
            table.goto[rep as usize]
                .iter()
                .map(|(&nt, &(target, replace))| (nt, (mapping[target as usize], replace)))
                .collect()
        })
        .collect();
    let advance = if table.advance.len() == nstates {
        representatives
            .iter()
            .map(|&rep| table.advance[rep as usize].clone())
            .collect()
    } else {
        Vec::new()
    };

    let forwarded_shifts: FxHashSet<(u32, TerminalID)> = table
        .forwarded_shifts
        .iter()
        .map(|&(state, terminal)| (mapping[state as usize], terminal))
        .collect();
    let direct_regular_wide_frontiers = table
        .direct_regular_wide_frontiers
        .iter()
        .map(|descriptor| {
            let mut target_states = descriptor
                .target_states
                .iter()
                .map(|&state| mapping[state as usize])
                .collect::<Vec<_>>();
            target_states.sort_unstable();
            target_states.dedup();
            DirectRegularWideFrontierDescriptor {
                source_state: mapping[descriptor.source_state as usize],
                terminal: descriptor.terminal,
                target_states,
            }
        })
        .collect();

    GLRTable {
        action,
        goto,
        num_states: representatives.len() as u32,
        num_terminals: table.num_terminals,
        num_rules: table.num_rules,
        rules: table.rules,
        nonterminal_display_names: table.nonterminal_display_names,
        construction: table.construction,
        admission_policy: table.admission_policy,
        advance,
        unconditional_advance: Vec::new(),
        forwarded_shifts,
        control_terminals: table.control_terminals,
        skip_terminals: table.skip_terminals,
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers,
    }
}
