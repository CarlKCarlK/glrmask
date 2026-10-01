//! Exact runtime representation for a bounded repetition whose body consumes
//! exactly one byte.
//!
//! The physical tokenizer contains only its reset state. Reached positive
//! repetition counts are encoded arithmetically in a disjoint logical-state
//! interval, so storage remains constant regardless of `max` or input length.

use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

/// Dynamic NFA configurations reserve the high bit of a raw tokenizer state.
/// Virtual tokenizer ids must therefore remain below it.
const VIRTUAL_STATE_LIMIT: u32 = 1 << 31;

/// Whether an arithmetic unit-repeat runtime with `max` positive residuals
/// fits beside `physical_state_count` materialized tokenizer states. Dynamic
/// compilation uses this before mutating the tokenizer so an oversized unit
/// lane can be routed to the exact repeat-product runtime instead.
pub(super) fn virtual_unit_repeat_state_ids_fit(
    max: usize,
    physical_state_count: u32,
) -> bool {
    let Ok(virtual_count) = u32::try_from(max) else {
        return false;
    };
    physical_state_count > 0
        && physical_state_count < VIRTUAL_STATE_LIMIT
        && physical_state_count
            .checked_add(virtual_count)
            .is_some_and(|end| end <= VIRTUAL_STATE_LIMIT)
}

#[derive(Debug)]
pub(super) struct VirtualZeroMinUnitRepeatRuntime {
    body: U8Set,
    min: usize,
    max: usize,
    physical_state_count: u32,
    root_state: u32,
    terminal: TerminalID,
    accepting: BitSet,
    live: BitSet,
    dead: BitSet,
    accepting_list: Box<[TerminalID]>,
}

/// Exact finite-token projection for the arithmetic repeat coordinate.
///
/// Positive exact states encode the consumed-byte count arithmetically after
/// the physical-state prefix. The mask tokenizer keeps the physical states,
/// finite lower/upper boundary stencils, and at most one deep class on each
/// side. This object is deliberately independent of any runtime interner:
/// projection is closed-form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct VirtualZeroMinUnitRepeatMaskProjection {
    full_min: u32,
    full_max: u32,
    full_physical_state_count: u32,
    mask_state_count: u32,
    deep_lower_state: u32,
    lower_start: u32,
    lower_offset: u32,
    interior_state: u32,
    upper_start: u32,
    upper_offset: u32,
}

impl VirtualZeroMinUnitRepeatMaskProjection {
    /// Stable construction metadata; excludes mutable runtime history.
    #[doc(hidden)]
    pub fn observation_descriptor(&self) -> Vec<u8> {
        bincode::serialize(&[self.full_min, self.full_max, self.full_physical_state_count,
            self.mask_state_count, self.deep_lower_state, self.lower_start, self.lower_offset,
            self.interior_state, self.upper_start, self.upper_offset])
            .expect("fixed unit-repeat descriptor serialization")
    }

    pub(super) fn new(
        full_min: usize,
        full_max: usize,
        horizon: usize,
        full_physical_state_count: u32,
    ) -> Option<Self> {
        let full_min = u32::try_from(full_min).ok()?;
        let full_max = u32::try_from(full_max).ok()?;
        let horizon = u32::try_from(horizon).ok()?;
        if full_min > full_max || full_max == 0 {
            return None;
        }

        // Positive exact counts are partitioned into four finite-horizon
        // regions. Counts more than H copies below `min` form one deep-lower
        // class; the next H counts are exact. Accepted counts more than H
        // copies below `max` form one interior class; the final H+1 counts are
        // exact. The lower/upper exact stencils may touch but never overlap in
        // the state layout because lower states stop at min-1.
        let lower_start = full_min.saturating_sub(horizon).max(1);
        let lower_count = full_min.saturating_sub(lower_start);
        let has_deep_lower = lower_start > 1;
        let accepted_start = full_min.max(1);
        let upper_start = full_max.saturating_sub(horizon).max(accepted_start);
        let has_interior = accepted_start < upper_start;
        let upper_count = full_max.checked_sub(upper_start)?.checked_add(1)?;

        let mut next = full_physical_state_count;
        let deep_lower_state = if has_deep_lower {
            let state = next;
            next = next.checked_add(1)?;
            state
        } else {
            u32::MAX
        };
        let lower_offset = next;
        next = next.checked_add(lower_count)?;
        let interior_state = if has_interior {
            let state = next;
            next = next.checked_add(1)?;
            state
        } else {
            u32::MAX
        };
        let upper_offset = next;
        next = next.checked_add(upper_count)?;
        (next <= VIRTUAL_STATE_LIMIT).then_some(Self {
            full_min,
            full_max,
            full_physical_state_count,
            mask_state_count: next,
            deep_lower_state,
            lower_start,
            lower_offset,
            interior_state,
            upper_start,
            upper_offset,
        })
    }

    #[inline]
    pub fn project(self, full_state: u32) -> Option<u32> {
        if full_state < self.full_physical_state_count {
            return Some(full_state);
        }
        let consumed = full_state
            .checked_sub(self.full_physical_state_count)?
            .checked_add(1)?;
        if consumed > self.full_max {
            return None;
        }
        if consumed < self.full_min {
            if self.deep_lower_state != u32::MAX && consumed < self.lower_start {
                return Some(self.deep_lower_state);
            }
            return self
                .lower_offset
                .checked_add(consumed.checked_sub(self.lower_start)?);
        }
        if self.interior_state != u32::MAX && consumed < self.upper_start {
            return Some(self.interior_state);
        }
        self.upper_offset
            .checked_add(consumed.checked_sub(self.upper_start)?)
    }

    #[inline]
    pub fn mask_state_count(self) -> u32 {
        self.mask_state_count
    }

    pub fn multiplicities(self) -> Vec<usize> {
        let mut counts = vec![1usize; self.mask_state_count() as usize];
        if self.deep_lower_state != u32::MAX {
            counts[self.deep_lower_state as usize] = self.lower_start.saturating_sub(1) as usize;
        }
        if self.interior_state != u32::MAX {
            counts[self.interior_state as usize] = self
                .upper_start
                .saturating_sub(self.full_min.max(1)) as usize;
        }
        counts
    }

    pub fn unique_full_states(self) -> Vec<u32> {
        let mut unique = vec![u32::MAX; self.mask_state_count() as usize];
        for (state, slot) in unique[..self.full_physical_state_count as usize]
            .iter_mut()
            .enumerate()
        {
            *slot = state as u32;
        }
        if self.deep_lower_state != u32::MAX && self.lower_start == 2 {
            unique[self.deep_lower_state as usize] = self.full_physical_state_count;
        }
        for consumed in self.lower_start..self.full_min {
            let mask_state = self.lower_offset + consumed - self.lower_start;
            unique[mask_state as usize] = self.full_physical_state_count + consumed - 1;
        }
        if self.interior_state != u32::MAX
            && self.upper_start.saturating_sub(self.full_min.max(1)) == 1
        {
            unique[self.interior_state as usize] =
                self.full_physical_state_count + self.full_min.max(1) - 1;
        }
        for consumed in self.upper_start..=self.full_max {
            let mask_state = self.upper_offset + consumed - self.upper_start;
            unique[mask_state as usize] = self.full_physical_state_count + consumed - 1;
        }
        unique
    }

    #[inline]
    pub(super) fn first_positive_state(self) -> Option<u32> {
        let full_state = self.full_physical_state_count;
        self.project(full_state)
    }

    #[inline]
    pub(super) fn deep_lower_state(self) -> Option<u32> {
        (self.deep_lower_state != u32::MAX).then_some(self.deep_lower_state)
    }

    #[inline]
    pub(super) fn lower_range(self) -> std::ops::Range<u32> {
        self.lower_offset..self.lower_offset + self.full_min.saturating_sub(self.lower_start)
    }

    #[inline]
    pub(super) fn interior_state(self) -> Option<u32> {
        (self.interior_state != u32::MAX).then_some(self.interior_state)
    }

    #[inline]
    pub(super) fn upper_range(self) -> std::ops::Range<u32> {
        self.upper_offset..self.mask_state_count
    }

    #[inline]
    pub(super) fn full_consumed_for_exact_mask_state(self, state: u32) -> Option<u32> {
        if self.lower_range().contains(&state) {
            return Some(self.lower_start + state - self.lower_offset);
        }
        if self.upper_range().contains(&state) {
            return Some(self.upper_start + state - self.upper_offset);
        }
        None
    }
}

impl VirtualZeroMinUnitRepeatRuntime {
    pub(super) fn new(
        body: U8Set,
        min: usize,
        max: usize,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
    ) -> Option<Self> {
        if body.is_empty()
            || max == 0
            || min > max
            || physical_state_count == 0
            || root_state >= physical_state_count
            || !virtual_unit_repeat_state_ids_fit(max, physical_state_count)
            || terminal >= num_terminals
        {
            return None;
        }
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        Some(Self {
            body,
            min,
            max,
            physical_state_count,
            root_state,
            terminal,
            accepting,
            live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
        })
    }

    #[inline]
    pub(super) fn body(&self) -> U8Set {
        self.body
    }

    #[inline]
    pub(super) fn max(&self) -> usize {
        self.max
    }

    #[inline]
    pub(super) fn min(&self) -> usize {
        self.min
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
    pub(super) fn handles_state(&self, state: u32) -> bool {
        state == self.root_state || self.is_virtual_state(state)
    }

    #[inline]
    pub(super) fn terminal(&self) -> TerminalID {
        self.terminal
    }

    #[inline]
    fn consumed(&self, state: u32) -> Option<usize> {
        if state == self.root_state {
            return Some(0);
        }
        if state < self.physical_state_count {
            return None;
        }
        let index = state.checked_sub(self.physical_state_count)? as usize;
        (index < self.max).then_some(index + 1)
    }

    fn encode_positive(&self, consumed: usize) -> Option<u32> {
        if consumed == 0 || consumed > self.max {
            return None;
        }
        let index = consumed - 1;
        let id = self
            .physical_state_count
            .checked_add(u32::try_from(index).ok()?)?;
        (id < VIRTUAL_STATE_LIMIT).then_some(id)
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let consumed = self.consumed(state)?;
        if consumed >= self.max || !self.body.contains(byte) {
            return None;
        }
        self.encode_positive(consumed.checked_add(1)?)
    }

    pub(super) fn is_virtual_state(&self, state: u32) -> bool {
        state >= self.physical_state_count && self.consumed(state).is_some()
    }

    pub(super) fn finalizers(&self, state: u32) -> Option<&BitSet> {
        (state >= self.physical_state_count).then(|| {
            if self
                .consumed(state)
                .is_some_and(|consumed| consumed >= self.min)
            {
                &self.accepting
            } else {
                &self.dead
            }
        })
    }

    pub(super) fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        (state >= self.physical_state_count).then(|| {
            if self
                .consumed(state)
                .is_some_and(|consumed| consumed >= self.min)
            {
                self.accepting_list.as_ref()
            } else {
                &[]
            }
        })
    }

    pub(super) fn futures(&self, state: u32) -> Option<&BitSet> {
        if state != self.root_state && state < self.physical_state_count {
            return None;
        }
        let Some(consumed) = self.consumed(state) else {
            return Some(&self.dead);
        };
        Some(if consumed < self.max { &self.live } else { &self.dead })
    }

    pub(super) fn transition_target(&self, state: u32) -> Option<u32> {
        let consumed = self.consumed(state)?;
        (consumed < self.max)
            .then(|| self.encode_positive(consumed + 1))
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::{virtual_unit_repeat_state_ids_fit, VIRTUAL_STATE_LIMIT};

    #[test]
    fn virtual_unit_repeat_state_id_boundary_is_exact() {
        let physical = 17u32;
        let last_fitting = (VIRTUAL_STATE_LIMIT - physical) as usize;
        assert!(virtual_unit_repeat_state_ids_fit(last_fitting, physical));
        assert!(!virtual_unit_repeat_state_ids_fit(
            last_fitting + 1,
            physical,
        ));
    }
}
