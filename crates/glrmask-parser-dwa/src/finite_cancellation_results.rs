//! Private cancellation memo-result storage, not parser graph semantics.
//!
//! Common empty/singleton answers fit in the existing 64-bit memo value slot.
//! Only multi-target answers allocate an immutable sorted slice. Handles are
//! private: decoding, query completion order, and weight operations are exact.
use super::{FastBoundaryDerivedRow, FastBoundaryWeightId};

type Pair = (u32, FastBoundaryWeightId);
const MAX_PAIRS: usize = 4_000_000;

pub(super) trait ResultRows {
    type Handle: Copy + Eq + std::fmt::Debug;
    const EMPTY: Self::Handle;
    fn new() -> Self;
    fn push(&mut self, row: FastBoundaryDerivedRow) -> Option<Self::Handle>;
    fn row_len(&self, handle: Self::Handle) -> usize;
    fn pair(&self, handle: Self::Handle, index: usize) -> Pair;
    fn storage_bytes(&self) -> usize;
}

/// Independent historical representation for differential testing/timing.
pub(super) struct VectorRows(Vec<Vec<Pair>>);
impl ResultRows for VectorRows {
    type Handle = usize;
    const EMPTY: usize = 0;
    fn new() -> Self { Self(vec![Vec::new()]) }
    #[inline] fn push(&mut self, row: FastBoundaryDerivedRow) -> Option<usize> {
        let mut row = row.into_entries();
        row.sort_unstable_by_key(|&(target, _)| target);
        let id = self.0.len();
        self.0.push(row);
        Some(id)
    }
    #[inline] fn row_len(&self, id: usize) -> usize { self.0[id].len() }
    #[inline] fn pair(&self, id: usize, index: usize) -> Pair { self.0[id][index] }
    fn storage_bytes(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<Vec<Pair>>()
            + self.0.iter().map(|r| r.capacity() * std::mem::size_of::<Pair>()).sum::<usize>()
    }
}

/// Empty=0, singleton=(target<<32)|nonzero_weight, multi=tag|len<<32|offset.
/// The existing graph/weight/result budgets fit strictly within these fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedHandle(u64);
impl PackedHandle {
    const MULTI: u64 = 1 << 63;
    fn singleton(target: u32, weight: u32) -> Option<Self> {
        if target >= (1 << 31) || weight == 0 { return None; }
        Some(Self((u64::from(target) << 32) | u64::from(weight)))
    }
    fn span(start: usize, len: usize, limit: usize) -> Option<Self> {
        if len < 2 || len >= (1usize << 31) || start.checked_add(len)? > limit.min(MAX_PAIRS) {
            return None;
        }
        let start = u32::try_from(start).ok()?;
        Some(Self(Self::MULTI | ((len as u64) << 32) | u64::from(start)))
    }
    #[inline] fn len(self) -> usize {
        if self.0 & Self::MULTI != 0 { ((self.0 >> 32) & 0x7fff_ffff) as usize }
        else { usize::from(self.0 != 0) }
    }
}

pub(super) struct PackedRows { entries: Vec<Pair>, pair_limit: usize }
impl ResultRows for PackedRows {
    type Handle = PackedHandle;
    const EMPTY: PackedHandle = PackedHandle(0);
    fn new() -> Self { Self { entries: Vec::new(), pair_limit: MAX_PAIRS } }
    #[inline] fn push(&mut self, row: FastBoundaryDerivedRow) -> Option<PackedHandle> {
        match row {
            FastBoundaryDerivedRow::Empty => Some(Self::EMPTY),
            FastBoundaryDerivedRow::One(target, weight) => PackedHandle::singleton(target, weight),
            FastBoundaryDerivedRow::Many(entries) => {
                // These two branches also handle deliberately noncanonical
                // test inputs. The solver normally constructs Many with >=2.
                if entries.is_empty() { return Some(Self::EMPTY); }
                if entries.len() == 1 {
                    let (target, weight) = entries.into_iter().next()?;
                    return PackedHandle::singleton(target, weight);
                }
                let start = self.entries.len();
                let handle = PackedHandle::span(start, entries.len(), self.pair_limit)?;
                // All shape/budget checks precede append and handle publication.
                self.entries.extend(entries);
                self.entries[start..].sort_unstable_by_key(|&(target, _)| target);
                Some(handle)
            }
        }
    }
    #[inline] fn row_len(&self, handle: PackedHandle) -> usize { handle.len() }
    #[inline] fn pair(&self, handle: PackedHandle, index: usize) -> Pair {
        assert!(index < handle.len());
        if handle.0 & PackedHandle::MULTI != 0 {
            self.entries[handle.0 as u32 as usize + index]
        } else { ((handle.0 >> 32) as u32, handle.0 as u32) }
    }
    fn storage_bytes(&self) -> usize { self.entries.capacity() * std::mem::size_of::<Pair>() }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make(index: usize) -> FastBoundaryDerivedRow {
        match index % 5 {
            0 | 1 => FastBoundaryDerivedRow::Empty,
            2 => FastBoundaryDerivedRow::One(index as u32, 19),
            _ => FastBoundaryDerivedRow::Many((0..(index % 29 + 2)).rev()
                .map(|q| (q as u32 * 13, (q + 1) as u32)).collect()),
        }
    }
    fn decoded<R: ResultRows>(rows: &R, handle: R::Handle) -> Vec<Pair> {
        (0..rows.row_len(handle)).map(|i| rows.pair(handle, i)).collect()
    }
    #[test]
    fn packed_answers_preserve_all_rows_across_growth_without_descriptor_ids() {
        let mut reference = VectorRows::new();
        let mut packed = PackedRows::new();
        let mut handles = Vec::new();
        for i in 0..4096 {
            let a = reference.push(make(i)).unwrap();
            let c = packed.push(make(i)).unwrap();
            assert_eq!(decoded(&reference, a), decoded(&packed, c));
            handles.push((a, c));
        }
        for (a, c) in handles {
            assert_eq!(decoded(&reference, a), decoded(&packed, c));
        }
        assert_eq!(std::mem::size_of::<PackedHandle>(), 8);
        assert!(packed.storage_bytes() < reference.storage_bytes());
    }
    #[test]
    fn empty_and_target_zero_singletons_are_distinct_and_allocate_nothing() {
        let mut rows = PackedRows::new();
        for (target, weight) in [(0, 1), (0, u32::MAX), ((1 << 31) - 1, u32::MAX), (12, 39)] {
            let h = rows.push(FastBoundaryDerivedRow::One(target, weight)).unwrap();
            assert_ne!(h, PackedRows::EMPTY);
            assert_eq!(rows.row_len(h), 1);
            assert_eq!(rows.pair(h, 0), (target, weight));
        }
        for _ in 0..4096 {
            assert_eq!(rows.push(FastBoundaryDerivedRow::Empty), Some(PackedRows::EMPTY));
        }
        assert_eq!(rows.storage_bytes(), 0);
        assert_eq!(rows.row_len(PackedRows::EMPTY), 0);
        assert!(rows.push(FastBoundaryDerivedRow::One(1 << 31, 1)).is_none());
        assert!(rows.push(FastBoundaryDerivedRow::One(0, 0)).is_none());
    }
    #[test]
    fn failed_packed_span_never_appends_or_changes_existing_answers() {
        let mut rows = PackedRows { entries: Vec::new(), pair_limit: 3 };
        let two = || FastBoundaryDerivedRow::Many([(10, 2), (3, 1)].into_iter().collect());
        let h = rows.push(two()).unwrap();
        assert_eq!(decoded(&rows, h), vec![(3, 1), (10, 2)]);
        let before = rows.entries.clone();
        assert!(rows.push(two()).is_none());
        assert_eq!(rows.entries, before);
        assert_eq!(decoded(&rows, h), before);
        assert!(PackedHandle::span(usize::MAX, 2, usize::MAX).is_none());
        assert!(PackedHandle::span(0, 1, MAX_PAIRS).is_none());
        assert!(PackedHandle::span(0, 1usize << 31, usize::MAX).is_none());
        assert!(PackedHandle::span(MAX_PAIRS - 1, 2, MAX_PAIRS).is_none());
    }
}
