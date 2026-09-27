use super::*;

#[test]
fn sparse_entries_cover_u32_model_ids_without_growing() {
    assert_eq!(std::mem::size_of::<SparseEntry>(), 8);
    assert_eq!(std::mem::align_of::<SparseEntry>(), 4);
    assert_eq!(std::mem::size_of::<(u32, u32)>(), 8);
    let ids = vec![vec![0, (1 << 21) + 31, u32::MAX]];
    let entries = compute_token_entries(&ids, (u32::MAX / 32) as usize + 1);
    let actual: Vec<_> = entries[0].iter().map(|e| (e.word_idx(), e.mask())).collect();
    assert_eq!(actual, vec![(0, 1), (65536, 1 << 31), (u32::MAX / 32, 1 << 31)]);
}

#[test]
fn high_word_masks_match_union_for_every_selection_and_dirty_buffer() {
    let groups = vec![
        vec![0, (1 << 21) + 31],
        vec![31, (1 << 21) + 32],
        vec![7, (1 << 21) - 1],
        vec![(1 << 21) + 30],
    ];
    let words = 65539;
    let mapping = FinalMaskMapping::new(&groups, words);
    for selected in 0..16u64 {
        for poison in [0, 0x5555_5555u32] {
            let mut expected = vec![poison; words];
            for (index, originals) in groups.iter().enumerate() {
                if selected & (1 << index) != 0 {
                    for &original in originals {
                        expected[original as usize / 32] |= 1 << (original % 32);
                    }
                }
            }
            let mut fast = vec![poison; words];
            mapping.or_dense_to_buf_fast(&[selected], &mut fast, poison == 0);
            assert_eq!(fast, expected, "fast path selection={selected:#x} poison={poison:#x}");
            let mut profiled = vec![poison; words];
            mapping.or_dense_to_buf(&[selected], &mut profiled, poison == 0);
            assert_eq!(profiled, expected, "profiled path selection={selected:#x} poison={poison:#x}");
        }
    }
}
