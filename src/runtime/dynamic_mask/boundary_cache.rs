//! Exact physical-liveness memoization, local to one vocabulary walk.

const SLOTS: usize = 16;
const PROMOTE_COLLISIONS: u16 = 64;
const MAX_DENSE_BYTES: usize = 64 * 1024;

struct Row {
    cells: [u64; SLOTS],
    dense: Option<Box<[u8]>>,
    collisions: u16,
}

/// Complete lexer tags and disjoint parser rows prevent false cache hits.
/// Thrashing rows may retain the same predicate in dense storage; the total
/// additional dense allocation is bounded independently of parser-node growth.
pub(super) struct FullWalkBoundaryDirectCache {
    rows: Vec<Row>,
    width: usize,
    dense_bytes: usize,
}

impl FullWalkBoundaryDirectCache {
    #[cfg(test)]
    pub(super) fn new() -> Self { Self::with_width(0) }

    pub(super) fn with_width(width: usize) -> Self {
        Self { rows: Vec::new(), width, dense_bytes: 0 }
    }

    pub(super) fn compact_row_bytes() -> usize { std::mem::size_of::<Row>() }

    pub(super) fn push_row(&mut self) {
        self.rows.push(Row { cells: [0; SLOTS], dense: None, collisions: 0 });
    }

    #[inline(always)]
    pub(super) fn get(&self, parser: usize, lexer: u32) -> u8 {
        let row = &self.rows[parser];
        if let Some(dense) = row.dense.as_ref() {
            return dense.get(lexer as usize).copied().unwrap_or(0);
        }
        let cell = row.cells[lexer as usize & (SLOTS - 1)];
        if cell >> 2 == u64::from(lexer) { (cell & 3) as u8 } else { 0 }
    }

    /// Read a coordinate already proved to belong to the walk's fixed domain.
    ///
    /// # Safety
    /// `parser` must name a retained row. If that row is dense, `lexer` must
    /// lie in its fixed transition domain. This is the same invariant used by
    /// the caller's pre-existing unchecked dense-row representation.
    #[inline(always)]
    pub(super) unsafe fn get_physical(&self, parser: usize, lexer: u32) -> u8 {
        debug_assert!(parser < self.rows.len());
        let row = unsafe { self.rows.get_unchecked(parser) };
        if let Some(dense) = row.dense.as_ref() {
            debug_assert!((lexer as usize) < dense.len());
            return unsafe { *dense.get_unchecked(lexer as usize) };
        }
        let cell = row.cells[lexer as usize & (SLOTS - 1)];
        if cell >> 2 == u64::from(lexer) { (cell & 3) as u8 } else { 0 }
    }

    #[inline(always)]
    pub(super) fn set(&mut self, parser: usize, lexer: u32, value: u8) {
        debug_assert!(value == 1 || value == 2);
        let row = &mut self.rows[parser];
        if let Some(dense) = row.dense.as_mut() {
            if let Some(cell) = dense.get_mut(lexer as usize) { *cell = value; }
            return;
        }
        let slot = lexer as usize & (SLOTS - 1);
        let previous = row.cells[slot];
        if previous & 3 != 0 && previous >> 2 != u64::from(lexer) {
            row.collisions = row.collisions.saturating_add(1);
        }
        row.cells[slot] = (u64::from(lexer) << 2) | u64::from(value);
        if row.collisions >= PROMOTE_COLLISIONS
            && self.width != 0
            && (lexer as usize) < self.width
            && self.width <= MAX_DENSE_BYTES.saturating_sub(self.dense_bytes)
        {
            let mut dense = vec![0; self.width].into_boxed_slice();
            for &cell in &row.cells {
                let retained = (cell & 3) as u8;
                if retained != 0 {
                    if let Some(target) = dense.get_mut((cell >> 2) as usize) {
                        *target = retained;
                    }
                }
            }
            self.dense_bytes += self.width;
            row.dense = Some(dense);
        }
    }

    pub(super) fn storage_bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<Row>() + self.dense_bytes
    }

    pub(super) fn dense_row_count(&self) -> usize {
        self.rows.iter().filter(|row| row.dense.is_some()).count()
    }

    /// The caller invokes this before publishing the first dense row pointer.
    /// No pointer into a compact or adaptively promoted row ever escapes.
    pub(super) fn expand_into(self, rows: &mut [Vec<u8>], width: usize) {
        assert_eq!(self.rows.len(), rows.len());
        for (row, cached) in rows.iter_mut().zip(self.rows) {
            if let Some(dense) = cached.dense {
                assert_eq!(dense.len(), width);
                *row = dense.into_vec();
            } else {
                row.resize(width, 0);
                for cell in cached.cells {
                    let value = (cell & 3) as u8;
                    if value != 0 {
                        let lexer = (cell >> 2) as usize;
                        assert!(lexer < width, "only cache states in this walk's fixed domain");
                        row[lexer] = value;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_tags_preserve_high_ids_without_cross_parser_aliases() {
        let mut cache = FullWalkBoundaryDirectCache::new();
        cache.push_row(); cache.push_row();
        cache.set(0, 0, 2); cache.set(1, 0, 1);
        cache.set(0, 16, 1);
        assert_eq!(cache.get(0, 0), 0);
        assert_eq!(cache.get(0, 16), 1);
        assert_eq!(cache.get(1, 0), 1);
        cache.set(0, u32::MAX, 2);
        assert_eq!(cache.get(0, u32::MAX), 2);
        assert_eq!(cache.get(0, 15), 0);
        assert_eq!(cache.dense_row_count(), 0);
        assert_eq!(cache.storage_bytes(), 2 * FullWalkBoundaryDirectCache::compact_row_bytes());
    }

    #[test]
    fn collision_promotion_preserves_both_predicates_and_disjoint_rows() {
        let mut cache = FullWalkBoundaryDirectCache::with_width(257);
        cache.push_row(); cache.push_row();
        for lexer in 0..96 {
            cache.set(0, lexer, 1 + (lexer % 2) as u8);
            cache.set(1, lexer, 2 - (lexer % 2) as u8);
        }
        assert_eq!(cache.dense_row_count(), 2);
        for lexer in 80..96 {
            assert_eq!(cache.get(0, lexer), 1 + (lexer % 2) as u8);
            assert_eq!(cache.get(1, lexer), 2 - (lexer % 2) as u8);
        }
        assert_eq!(cache.get(0, 256), 0);
        assert_eq!(cache.get(0, u32::MAX), 0);
        cache.set(0, 95, 1);
        assert_eq!(cache.get(0, 95), 1);
        assert_eq!(cache.get(1, 95), 1);
    }

    #[test]
    fn dense_budget_remains_bounded_when_parser_rows_grow() {
        let mut cache = FullWalkBoundaryDirectCache::with_width(1025);
        for parser in 0..100 {
            cache.push_row();
            for lexer in 0..128 { cache.set(parser, lexer, 1 + (lexer % 2) as u8); }
        }
        assert_eq!(cache.dense_row_count(), MAX_DENSE_BYTES / 1025);
        assert!(cache.dense_bytes <= MAX_DENSE_BYTES);
        for parser in 0..100 {
            for lexer in 0..1025 {
                let cached = cache.get(parser, lexer);
                assert_eq!(unsafe { cache.get_physical(parser, lexer) }, cached);
                if cached != 0 { assert_eq!(cached, 1 + (lexer % 2) as u8); }
            }
        }
        assert_eq!(cache.storage_bytes(), 100 * std::mem::size_of::<Row>() + cache.dense_bytes);
    }

    #[test]
    fn promotion_requires_real_collisions_and_a_bounded_domain() {
        for width in [0, MAX_DENSE_BYTES + 1, usize::MAX] {
            let mut cache = FullWalkBoundaryDirectCache::with_width(width);
            cache.push_row();
            for lexer in 0..256 { cache.set(0, lexer, 2); }
            assert_eq!(cache.dense_row_count(), 0);
            assert_eq!(cache.get(0, 255), 2);
        }
        let mut cache = FullWalkBoundaryDirectCache::with_width(257);
        cache.push_row();
        for _ in 0..1000 { cache.set(0, 7, 2); }
        assert_eq!(cache.dense_row_count(), 0);
    }

    #[test]
    fn mixed_compact_and_dense_rows_expand_before_pointer_publication() {
        let mut cache = FullWalkBoundaryDirectCache::with_width(257);
        cache.push_row(); cache.push_row();
        for lexer in 0..96 { cache.set(0, lexer, 1 + (lexer % 2) as u8); }
        cache.set(1, 3, 1); cache.set(1, 200, 2);
        let expected: Vec<Vec<u8>> = (0..2).map(|parser|
            (0..257).map(|lexer| cache.get(parser, lexer)).collect()).collect();
        let mut rows = vec![Vec::new(); 2];
        cache.expand_into(&mut rows, 257);
        assert_eq!(rows, expected);
    }
}
