//! Exact lazy runtime for the intersection of two large bounded repetitions.
//!
//! Each repetition body is required to have a deterministic, non-nullable,
//! prefix-free DFA.  A component residual is therefore exactly
//! `(completed_copies, body_state)`.  The product residual is the pair of
//! component residuals.  We intern product residuals only when runtime input
//! reaches them; the two regex bounds never determine an allocation size.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::dfa::DFA;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

const VIRTUAL_STATE_LIMIT: u32 = 1 << 31;
const DEAD_TRANSITION: u32 = u32::MAX;
/// Exact future-liveness analysis works on the Cartesian product of the two
/// repeat-body DFAs. Keep that grammar-derived product bounded as well: giant
/// repeat counts must never be replaced by a different accidental OOM vector.
const MAX_BODY_PRODUCT_STATES: usize = 65_536;
/// The finite one-token quotient is an optimization, not the authoritative
/// lexer. Refuse pathological horizons/future stencils before they can turn a
/// lazy giant repeat back into a large eager allocation; callers fall back to
/// exact dynamic scanning when projection construction returns `None`.
const MAX_MASK_PROJECTION_STATES: usize = 262_144;

/// Shared allocator for exact virtual tokenizer-state handles. Multiple lazy
/// repeat products in one tokenizer share one allocator, so their lazily
/// interned state IDs can never collide even when their residuals are reached
/// in an arbitrary interleaving at runtime.
#[derive(Debug)]
pub(super) struct VirtualStateAllocator {
    next: AtomicU32,
}

impl VirtualStateAllocator {
    pub(super) fn new(first: u32) -> Option<Self> {
        (first < VIRTUAL_STATE_LIMIT).then(|| Self {
            next: AtomicU32::new(first),
        })
    }

    pub(super) fn allocate(&self) -> Option<u32> {
        self.next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                (next < VIRTUAL_STATE_LIMIT).then(|| next + 1)
            })
            .ok()
    }
}

/// Shared direct owner index for a family of virtual tokenizer runtimes.
/// Physical proxy roots are fixed at construction time; lazily allocated
/// states use a dense owner vector indexed from the allocator's first virtual
/// state. The vector therefore grows only with states actually reached at
/// runtime, never with any declared repeat bound.
#[derive(Debug)]
pub(super) struct VirtualRuntimeStateOwners {
    physical_state_count: u32,
    root_owners: FxHashMap<u32, u32>,
    virtual_owners: Mutex<Vec<u32>>,
}

impl VirtualRuntimeStateOwners {
    pub(super) fn new(physical_state_count: u32, roots: &[u32]) -> Option<Self> {
        let mut root_owners = FxHashMap::default();
        for (owner, &root) in roots.iter().enumerate() {
            if root >= physical_state_count
                || root_owners.insert(root, u32::try_from(owner).ok()?).is_some()
            {
                return None;
            }
        }
        Some(Self {
            physical_state_count,
            root_owners,
            virtual_owners: Mutex::new(Vec::new()),
        })
    }

    pub(super) fn register_virtual(&self, state: u32, owner: u32) -> Option<()> {
        let index = state.checked_sub(self.physical_state_count)? as usize;
        let mut owners = self.virtual_owners.lock().unwrap();
        if owners.len() <= index {
            owners.resize(index + 1, u32::MAX);
        }
        let slot = &mut owners[index];
        if *slot != u32::MAX && *slot != owner {
            return None;
        }
        *slot = owner;
        Some(())
    }

    pub(super) fn owner_index(&self, state: u32) -> Option<usize> {
        let owner = if state < self.physical_state_count {
            *self.root_owners.get(&state)?
        } else {
            let index = state.checked_sub(self.physical_state_count)? as usize;
            *self.virtual_owners.lock().unwrap().get(index)?
        };
        (owner != u32::MAX).then_some(owner as usize)
    }
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualBoundedRepeatSpec {
    pub base_dfa: Arc<DFA>,
    pub min: u32,
    pub max: u32,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualBinaryRepeatIntersectionDescriptor {
    pub left: VirtualBoundedRepeatSpec,
    pub right: VirtualBoundedRepeatSpec,
    pub byte_support: U8Set,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RepeatResidual {
    completed: u32,
    body_state: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ProductResidual {
    left: RepeatResidual,
    right: RepeatResidual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaskRepeatResidual {
    /// Copies still required to satisfy the lower bound, truncated at the
    /// finite-token lower-bound horizon. The top value is the deep prefix
    /// class: one model token cannot reach acceptance from it.
    required: u32,
    /// Remaining copies, truncated at `horizon + 1`; the top value means
    /// "farther than one vocabulary token can reach".
    remaining: u32,
    body_state: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MaskProductResidual {
    left: MaskRepeatResidual,
    right: MaskRepeatResidual,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualBinaryRepeatIntersectionMaskProjection {
    runtime: Arc<VirtualBinaryRepeatIntersectionRuntime>,
    far: u32,
    lower_far: u32,
    mask_state_by_residual: Arc<FxHashMap<MaskProductResidual, u32>>,
}

impl VirtualBinaryRepeatIntersectionMaskProjection {
    /// Canonical finite observation transport, independent of reached states
    /// and the hash-map iteration order of the exact runtime's interner.
    #[doc(hidden)]
    pub fn observation_descriptor(&self) -> Vec<u8> {
        let mut entries = self.mask_state_by_residual.iter().map(|(key, &state)|
            (key.left.required, key.left.remaining, key.left.body_state,
                key.right.required, key.right.remaining, key.right.body_state, state))
            .collect::<Vec<_>>();
        entries.sort_unstable();
        bincode::serialize(&(self.far, self.lower_far, self.runtime.terminal,
            self.runtime.physical_state_count, self.runtime.root_state,
            self.runtime.left.min, self.runtime.left.max,
            self.runtime.right.min, self.runtime.right.max, entries))
            .expect("finite product descriptor serialization")
    }

    #[inline]
    pub fn project(&self, full_state: u32) -> Option<u32> {
        if full_state < self.runtime.physical_state_count {
            return Some(full_state);
        }
        let store = self.runtime.store.lock().unwrap();
        let exact = self.runtime.residual_for_state_locked(&store, full_state)?;
        let projected = self
            .runtime
            .mask_residual(exact, self.far, self.lower_far);
        self.mask_state_by_residual.get(&projected).copied()
    }
}

#[derive(Debug, Default)]
struct LazyProductStore {
    states: Vec<ProductResidual>,
    state_by_residual: FxHashMap<ProductResidual, u32>,
    state_index: FxHashMap<u32, u32>,
    transitions: Vec<SmallVec<[(u8, u32); 8]>>,
    root_transitions: SmallVec<[(u8, u32); 8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BodyProductEdge {
    target: u32,
    left_completed: u32,
    right_completed: u32,
}

#[derive(Debug)]
pub(super) struct VirtualBinaryRepeatIntersectionRuntime {
    runtime_index: u32,
    left: VirtualBoundedRepeatSpec,
    right: VirtualBoundedRepeatSpec,
    byte_support: U8Set,
    terminal: TerminalID,
    physical_state_count: u32,
    root_state: u32,
    state_allocator: Arc<VirtualStateAllocator>,
    state_owners: Arc<VirtualRuntimeStateOwners>,
    accepting: BitSet,
    live: BitSet,
    dead: BitSet,
    accepting_list: Box<[TerminalID]>,
    /// For each pair of body-DFA states, the Pareto-minimal numbers of
    /// additional completed copies on the left/right needed by a *nonempty*
    /// common byte path to return both bodies to their repeat boundary.
    ///
    /// This is the exact future-language certificate for the product.  Counts
    /// are independent of the declared repeat bounds and are finite because a
    /// resource-feasible path can have cycles removed without increasing
    /// either completed-copy count.
    future_requirements: Arc<[Box<[(u32, u32)]>]>,
    future_requirement_cap: u32,
    store: Mutex<LazyProductStore>,
}

impl VirtualBinaryRepeatIntersectionRuntime {
    pub(super) fn new(
        descriptor: VirtualBinaryRepeatIntersectionDescriptor,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        let synchronized_self_repeat = descriptor.left.min == descriptor.right.min
            && descriptor.left.max == descriptor.right.max
            && Arc::ptr_eq(&descriptor.left.base_dfa, &descriptor.right.base_dfa);
        if descriptor.byte_support.is_empty()
            || terminal >= num_terminals
            || physical_state_count == 0
            || root_state >= physical_state_count
            || state_owners.owner_index(root_state) != Some(runtime_index as usize)
            || physical_state_count >= VIRTUAL_STATE_LIMIT
            || descriptor.left.min > descriptor.left.max
            || descriptor.right.min > descriptor.right.max
            || descriptor.left.base_dfa.num_states() == 0
            || descriptor.right.base_dfa.num_states() == 0
            || !descriptor
                .left
                .base_dfa
                .possible_future_group_ids(0)
                .contains(0)
            || !descriptor
                .right
                .base_dfa
                .possible_future_group_ids(0)
                .contains(0)
            || ((descriptor.left.min != 0 || descriptor.right.min != 0)
                && !synchronized_self_repeat)
        {
            return None;
        }
        let (future_requirements, future_requirement_cap) =
            Self::compute_future_requirements(&descriptor.left, &descriptor.right)?;
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        Some(Self {
            runtime_index,
            left: descriptor.left,
            right: descriptor.right,
            byte_support: descriptor.byte_support,
            terminal,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            accepting,
            live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
            future_requirements,
            future_requirement_cap,
            store: Mutex::new(LazyProductStore::default()),
        })
    }

    #[inline]
    fn pair_index(&self, left_state: u32, right_state: u32) -> Option<usize> {
        (left_state as usize)
            .checked_mul(self.right.base_dfa.num_states())?
            .checked_add(right_state as usize)
    }

    fn body_step(spec: &VirtualBoundedRepeatSpec, state: u32, byte: u8) -> Option<(u32, u32)> {
        let target = spec.base_dfa.step(state, byte)?;
        if spec.base_dfa.finalizers(target).contains(0) {
            return Some((0, 1));
        }
        if !spec.base_dfa.possible_future_group_ids(target).contains(0) {
            return None;
        }
        Some((target, 0))
    }

    fn insert_pareto(front: &mut Vec<(u32, u32)>, candidate: (u32, u32)) -> bool {
        if front
            .iter()
            .any(|&(left, right)| left <= candidate.0 && right <= candidate.1)
        {
            return false;
        }
        front.retain(|&(left, right)| !(candidate.0 <= left && candidate.1 <= right));
        front.push(candidate);
        true
    }

    fn compute_future_requirements(
        left: &VirtualBoundedRepeatSpec,
        right: &VirtualBoundedRepeatSpec,
    ) -> Option<(Arc<[Box<[(u32, u32)]>]>, u32)> {
        let left_states = left.base_dfa.num_states();
        let right_states = right.base_dfa.num_states();
        let pair_count = left_states.checked_mul(right_states)?;
        if pair_count == 0
            || pair_count > u32::MAX as usize
            || pair_count > MAX_BODY_PRODUCT_STATES
        {
            return None;
        }

        let mut edges = vec![Vec::<BodyProductEdge>::new(); pair_count];
        for left_state in 0..left_states as u32 {
            for right_state in 0..right_states as u32 {
                let source = left_state as usize * right_states + right_state as usize;
                for byte in 0u16..=255 {
                    let byte = byte as u8;
                    let Some((next_left, left_completed)) = Self::body_step(left, left_state, byte)
                    else {
                        continue;
                    };
                    let Some((next_right, right_completed)) =
                        Self::body_step(right, right_state, byte)
                    else {
                        continue;
                    };
                    let target = next_left as usize * right_states + next_right as usize;
                    let edge = BodyProductEdge {
                        target: target as u32,
                        left_completed,
                        right_completed,
                    };
                    if !edges[source].contains(&edge) {
                        edges[source].push(edge);
                    }
                }
            }
        }

        // Fixed point of Pareto-minimal resource costs for a nonempty path to
        // pair state (0,0). We deliberately do not seed (0,0) with cost (0,0):
        // tokenizer `possible_future` means that at least one more byte can be
        // consumed before the terminal matches again.
        let accept = 0u32;
        let mut requirements = vec![Vec::<(u32, u32)>::new(); pair_count];
        loop {
            let snapshot = requirements.clone();
            let mut changed = false;
            for source in 0..pair_count {
                for edge in &edges[source] {
                    if edge.target == accept {
                        changed |= Self::insert_pareto(
                            &mut requirements[source],
                            (edge.left_completed, edge.right_completed),
                        );
                    }
                    for &(left_cost, right_cost) in &snapshot[edge.target as usize] {
                        let Some(left_cost) = left_cost.checked_add(edge.left_completed) else {
                            return None;
                        };
                        let Some(right_cost) = right_cost.checked_add(edge.right_completed) else {
                            return None;
                        };
                        changed |= Self::insert_pareto(
                            &mut requirements[source],
                            (left_cost, right_cost),
                        );
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let cap = requirements
            .iter()
            .flat_map(|front| front.iter())
            .fold(0u32, |cap, &(left, right)| cap.max(left).max(right));
        let requirements = requirements
            .into_iter()
            .map(|mut front| {
                front.sort_unstable();
                front.into_boxed_slice()
            })
            .collect::<Vec<_>>();
        Some((Arc::from(requirements.into_boxed_slice()), cap))
    }

    fn residual_has_future(&self, residual: ProductResidual) -> bool {
        if self.is_synchronized_nonzero_repeat() {
            if residual.left != residual.right {
                return false;
            }
            let component = residual.left;
            return component.completed < self.left.max
                && (component.body_state == 0
                    || self
                        .left
                        .base_dfa
                        .possible_future_group_ids(component.body_state)
                        .contains(0));
        }
        let Some(index) = self.pair_index(residual.left.body_state, residual.right.body_state)
        else {
            return false;
        };
        let left_remaining = self.left.max.saturating_sub(residual.left.completed);
        let right_remaining = self.right.max.saturating_sub(residual.right.completed);
        self.future_requirements[index]
            .iter()
            .any(|&(left, right)| left <= left_remaining && right <= right_remaining)
    }

    pub(super) fn root_has_future(&self) -> bool {
        self.residual_has_future(self.initial_residual())
    }

    #[inline]
    fn is_synchronized_self_repeat(&self) -> bool {
        self.left.min == self.right.min
            && self.left.max == self.right.max
            && Arc::ptr_eq(&self.left.base_dfa, &self.right.base_dfa)
    }

    #[inline]
    fn is_synchronized_nonzero_repeat(&self) -> bool {
        self.left.min != 0 && self.is_synchronized_self_repeat()
    }

    #[inline]
    pub(super) fn terminal(&self) -> TerminalID {
        self.terminal
    }

    #[inline]
    pub(super) fn byte_support(&self) -> U8Set {
        self.byte_support
    }

    #[inline]
    pub(super) fn physical_state_count(&self) -> u32 {
        self.physical_state_count
    }

    #[inline]
    pub(super) fn root_state(&self) -> u32 {
        self.root_state
    }

    #[inline]
    fn initial_residual(&self) -> ProductResidual {
        ProductResidual {
            left: RepeatResidual {
                completed: 0,
                body_state: 0,
            },
            right: RepeatResidual {
                completed: 0,
                body_state: 0,
            },
        }
    }

    fn residual_for_state_locked(
        &self,
        store: &LazyProductStore,
        state: u32,
    ) -> Option<ProductResidual> {
        if state == self.root_state {
            return Some(self.initial_residual());
        }
        let index = *store.state_index.get(&state)? as usize;
        store.states.get(index).copied()
    }

    fn intern_locked(
        &self,
        store: &mut LazyProductStore,
        residual: ProductResidual,
    ) -> Option<u32> {
        if residual == self.initial_residual() {
            return Some(self.root_state);
        }
        if let Some(&state) = store.state_by_residual.get(&residual) {
            return Some(state);
        }
        let index = u32::try_from(store.states.len()).ok()?;
        let state = self.state_allocator.allocate().expect(
            "exact virtual tokenizer state-id space exhausted below the dynamic-NFA high-bit tag",
        );
        self.state_owners
            .register_virtual(state, self.runtime_index)
            .expect("virtual repeat state owner index must follow shared allocator");
        store.states.push(residual);
        store.transitions.push(SmallVec::new());
        store.state_by_residual.insert(residual, state);
        store.state_index.insert(state, index);
        Some(state)
    }

    fn step_component(
        spec: &VirtualBoundedRepeatSpec,
        residual: RepeatResidual,
        byte: u8,
    ) -> Option<RepeatResidual> {
        // `completed` counts finished copies. A partial next copy can only
        // exist while completed < max.
        if residual.completed >= spec.max {
            return None;
        }
        let target = spec.base_dfa.step(residual.body_state, byte)?;
        if spec.base_dfa.finalizers(target).contains(0) {
            return Some(RepeatResidual {
                completed: residual.completed.checked_add(1)?,
                body_state: 0,
            });
        }
        if !spec.base_dfa.possible_future_group_ids(target).contains(0) {
            return None;
        }
        Some(RepeatResidual {
            completed: residual.completed,
            body_state: target,
        })
    }

    fn step_residual(&self, residual: ProductResidual, byte: u8) -> Option<ProductResidual> {
        Some(ProductResidual {
            left: Self::step_component(&self.left, residual.left, byte)?,
            right: Self::step_component(&self.right, residual.right, byte)?,
        })
    }

    #[inline]
    fn mask_component(
        spec: &VirtualBoundedRepeatSpec,
        residual: RepeatResidual,
        far: u32,
        lower_far: u32,
    ) -> MaskRepeatResidual {
        MaskRepeatResidual {
            required: spec
                .min
                .saturating_sub(residual.completed)
                .min(lower_far),
            remaining: spec.max.saturating_sub(residual.completed).min(far),
            body_state: residual.body_state,
        }
    }

    #[inline]
    fn mask_residual(
        &self,
        residual: ProductResidual,
        far: u32,
        lower_far: u32,
    ) -> MaskProductResidual {
        MaskProductResidual {
            left: Self::mask_component(&self.left, residual.left, far, lower_far),
            right: Self::mask_component(&self.right, residual.right, far, lower_far),
        }
    }

    fn step_mask_component(
        spec: &VirtualBoundedRepeatSpec,
        residual: MaskRepeatResidual,
        far: u32,
        lower_far: u32,
        byte: u8,
    ) -> Option<MaskRepeatResidual> {
        if residual.remaining == 0 {
            return None;
        }
        let target = spec.base_dfa.step(residual.body_state, byte)?;
        if spec.base_dfa.finalizers(target).contains(0) {
            let required = if residual.required == 0 {
                0
            } else if residual.required == lower_far {
                lower_far
            } else {
                residual.required.checked_sub(1)?
            };
            let remaining = if residual.remaining == far {
                far
            } else {
                residual.remaining.checked_sub(1)?
            };
            return Some(MaskRepeatResidual {
                required,
                remaining,
                body_state: 0,
            });
        }
        if !spec.base_dfa.possible_future_group_ids(target).contains(0) {
            return None;
        }
        Some(MaskRepeatResidual {
            required: residual.required,
            remaining: residual.remaining,
            body_state: target,
        })
    }

    fn step_mask_residual(
        &self,
        residual: MaskProductResidual,
        far: u32,
        lower_far: u32,
        byte: u8,
    ) -> Option<MaskProductResidual> {
        Some(MaskProductResidual {
            left: Self::step_mask_component(&self.left, residual.left, far, lower_far, byte)?,
            right: Self::step_mask_component(
                &self.right,
                residual.right,
                far,
                lower_far,
                byte,
            )?,
        })
    }

    fn mask_accepting(residual: MaskProductResidual) -> bool {
        residual.left.required == 0
            && residual.right.required == 0
            && residual.left.body_state == 0
            && residual.right.body_state == 0
    }

    fn mask_residual_has_future(&self, residual: MaskProductResidual) -> bool {
        if self.is_synchronized_nonzero_repeat() {
            return residual.left == residual.right && residual.left.remaining > 0;
        }
        let Some(index) = self.pair_index(residual.left.body_state, residual.right.body_state)
        else {
            return false;
        };
        self.future_requirements[index].iter().any(|&(left, right)| {
            left <= residual.left.remaining && right <= residual.right.remaining
        })
    }

    /// Build the finite observation DFA used for one model-token walk.
    ///
    /// `remaining = horizon + 1` is a mathematically exact finite-horizon
    /// abstraction: a token of at most `horizon` bytes cannot finish enough
    /// non-nullable repetitions to reach the upper bound from that class.
    pub(super) fn build_mask_projection(
        self: &Arc<Self>,
        horizon: usize,
        mut dfa: DFA,
        num_terminals: u32,
    ) -> Option<(DFA, VirtualBinaryRepeatIntersectionMaskProjection)> {
        let horizon = u32::try_from(horizon).ok()?;
        let synchronized_nonzero = self.is_synchronized_nonzero_repeat();
        let lower_far = horizon.checked_add(1)?;
        // `future_requirement_cap` bounds a cycle-free witness for every live
        // body-state pair. Keep another `horizon` copies exact so a model token
        // can consume at most `horizon` non-nullable repetitions and still end
        // in a state whose unbounded future observation is exact. The top value
        // is the merged deep-interior class and self-loops on copy completion.
        //
        // A synchronized self-product has one exact repeat counter, so its
        // future language depends only on whether upper budget remains. It
        // therefore needs only the same one-token margin on both lower and
        // upper boundaries, independent of body-product requirement costs.
        let far = if synchronized_nonzero {
            lower_far
        } else {
            self.future_requirement_cap.checked_add(horizon)?
        };
        let physical_state_count = self.physical_state_count;
        if (dfa.num_states() as u32) < physical_state_count {
            return None;
        }
        if (self.left.min != 0 || self.right.min != 0) && !synchronized_nonzero {
            return None;
        }

        // Every exact committed residual must have a projection, including
        // upper-tail states that are not reachable from the reset-side `far`
        // class. For a general zero-minimum product, seed every remaining-count
        // pair at a copy boundary. For a synchronized nonzero-minimum repeat,
        // only one counter is reachable, so seed the finite lower/upper stencils
        // on that diagonal instead of allocating a quadratic cross-product.
        let mut state_by_residual = FxHashMap::<MaskProductResidual, u32>::default();
        let mut queue = VecDeque::<MaskProductResidual>::new();
        if synchronized_nonzero {
            let stencil_side = usize::try_from(lower_far).ok()?.checked_add(1)?;
            let boundary_seed_upper_bound = stencil_side.checked_mul(2)?.checked_add(1)?;
            if dfa
                .num_states()
                .checked_add(boundary_seed_upper_bound)?
                > MAX_MASK_PROJECTION_STATES
            {
                return None;
            }
            let mut seed_completed = |completed: u32, dfa: &mut DFA| {
                let exact = ProductResidual {
                    left: RepeatResidual {
                        completed,
                        body_state: 0,
                    },
                    right: RepeatResidual {
                        completed,
                        body_state: 0,
                    },
                };
                let residual = self.mask_residual(exact, far, lower_far);
                if state_by_residual.contains_key(&residual) {
                    return Some(());
                }
                let state = dfa.add_state();
                state_by_residual.insert(residual, state);
                queue.push_back(residual);
                Some(())
            };
            seed_completed(0, &mut dfa)?;
            for distance in 0..=lower_far {
                seed_completed(self.left.min.saturating_sub(distance), &mut dfa)?;
                seed_completed(self.left.max.saturating_sub(distance), &mut dfa)?;
            }
        } else {
            let boundary_side = usize::try_from(far).ok()?.checked_add(1)?;
            let boundary_seed_states = boundary_side.checked_mul(boundary_side)?;
            if dfa
                .num_states()
                .checked_add(boundary_seed_states)?
                > MAX_MASK_PROJECTION_STATES
            {
                return None;
            }
            for left_remaining in 0..=far {
                for right_remaining in 0..=far {
                    let residual = MaskProductResidual {
                        left: MaskRepeatResidual {
                            required: 0,
                            remaining: left_remaining,
                            body_state: 0,
                        },
                        right: MaskRepeatResidual {
                            required: 0,
                            remaining: right_remaining,
                            body_state: 0,
                        },
                    };
                    let state = dfa.add_state();
                    state_by_residual.insert(residual, state);
                    queue.push_back(residual);
                }
            }
        }

        while let Some(residual) = queue.pop_front() {
            let state = state_by_residual[&residual];
            let mut finalizers = BitSet::new(num_terminals as usize);
            if Self::mask_accepting(residual) {
                finalizers.set(self.terminal as usize);
            }
            let mut futures = BitSet::new(num_terminals as usize);
            if synchronized_nonzero && self.mask_residual_has_future(residual) {
                futures.set(self.terminal as usize);
            }
            dfa.overwrite_state_metadata(state, finalizers, futures);

            for byte in self.byte_support.iter() {
                let Some(next) = self.step_mask_residual(residual, far, lower_far, byte) else {
                    continue;
                };
                let target = if let Some(&target) = state_by_residual.get(&next) {
                    target
                } else {
                    if dfa.num_states() >= MAX_MASK_PROJECTION_STATES {
                        return None;
                    }
                    let target = dfa.add_state();
                    state_by_residual.insert(next, target);
                    queue.push_back(next);
                    target
                };
                dfa.add_transition(state, byte, target);
            }
        }

        // The physical proxy root is the special drained zero-byte state. Its
        // first byte must enter the abstract residual that results from the
        // exact all-zero counter pair, but the root itself stays non-final.
        let exact_root = self.initial_residual();
        for byte in self.byte_support.iter() {
            let Some(next) = self.step_residual(exact_root, byte) else {
                continue;
            };
            let next = self.mask_residual(next, far, lower_far);
            let target = *state_by_residual.get(&next)?;
            dfa.add_transition(self.root_state, byte, target);
        }

        // Existing zero-minimum products retain their exact graph-derived
        // future metadata. The synchronized nonzero-minimum quotient cannot
        // use graph liveness because its deep-lower class intentionally
        // self-loops for one-token observation even though acceptance may lie
        // beyond that finite horizon; those futures were assigned analytically
        // above instead.
        if !synchronized_nonzero {
            dfa.recompute_possible_futures();
        }

        Some((
            dfa,
            VirtualBinaryRepeatIntersectionMaskProjection {
                runtime: Arc::clone(self),
                far,
                lower_far,
                mask_state_by_residual: Arc::new(state_by_residual),
            },
        ))
    }

    pub(super) fn handles_state(&self, state: u32) -> bool {
        self.state_owners.owner_index(state) == Some(self.runtime_index as usize)
    }

    pub(super) fn owner_index(&self, state: u32) -> Option<usize> {
        self.state_owners.owner_index(state)
    }

    fn step_locked(
        &self,
        store: &mut LazyProductStore,
        state: u32,
        residual: ProductResidual,
        byte: u8,
    ) -> Option<u32> {
        let cached = if state == self.root_state {
            store
                .root_transitions
                .iter()
                .find_map(|&(cached_byte, target)| (cached_byte == byte).then_some(target))
        } else {
            let index = *store.state_index.get(&state)? as usize;
            store.transitions[index]
                .iter()
                .find_map(|&(cached_byte, target)| (cached_byte == byte).then_some(target))
        };
        if let Some(target) = cached {
            return (target != DEAD_TRANSITION).then_some(target);
        }

        let target = self
            .step_residual(residual, byte)
            .and_then(|target| self.intern_locked(store, target))
            .unwrap_or(DEAD_TRANSITION);
        if state == self.root_state {
            store.root_transitions.push((byte, target));
        } else {
            let index = *store.state_index.get(&state)? as usize;
            store.transitions[index].push((byte, target));
        }
        (target != DEAD_TRANSITION).then_some(target)
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let mut store = self.store.lock().unwrap();
        let residual = self.residual_for_state_locked(&store, state)?;
        self.step_locked(&mut store, state, residual, byte)
    }

    fn component_accepting(spec: &VirtualBoundedRepeatSpec, residual: RepeatResidual) -> bool {
        residual.body_state == 0 && residual.completed >= spec.min
    }

    fn observation(&self, state: u32) -> Option<(bool, bool)> {
        let store = self.store.lock().unwrap();
        let residual = self.residual_for_state_locked(&store, state)?;
        // The physical proxy is the drained zero-byte configuration. Even if
        // both repetitions are nullable through min=0, it must not report a
        // terminal match before input has been consumed.
        let accepting = state != self.root_state
            && Self::component_accepting(&self.left, residual.left)
            && Self::component_accepting(&self.right, residual.right);
        let live = self.residual_has_future(residual);
        Some((accepting, live))
    }

    pub(super) fn finalizers(&self, state: u32) -> Option<&BitSet> {
        let (accepting, _) = self.observation(state)?;
        Some(if accepting { &self.accepting } else { &self.dead })
    }

    pub(super) fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        let (accepting, _) = self.observation(state)?;
        Some(if accepting {
            self.accepting_list.as_ref()
        } else {
            &[]
        })
    }

    pub(super) fn futures(&self, state: u32) -> Option<&BitSet> {
        let (_, live) = self.observation(state)?;
        Some(if live { &self.live } else { &self.dead })
    }

    pub(super) fn transitions(&self, state: u32) -> Option<Vec<(u8, u32)>> {
        if !self.handles_state(state) {
            return None;
        }
        let mut store = self.store.lock().unwrap();
        let residual = self.residual_for_state_locked(&store, state)?;
        let mut transitions = Vec::new();
        for byte in self.byte_support.iter() {
            if let Some(target) = self.step_locked(&mut store, state, residual, byte) {
                transitions.push((byte, target));
            }
        }
        Some(transitions)
    }

    pub(super) fn interned_state_count(&self) -> usize {
        self.store.lock().unwrap().states.len()
    }
}
