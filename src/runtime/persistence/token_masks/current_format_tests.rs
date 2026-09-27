use super::*;
use std::sync::Arc;

fn dense_cache_wire(magic: &[u8; 4], mask_words: u32, word_index: u32) -> Vec<u8> {
    let seeds = crate::runtime::artifact::SeedTerminalDenseMasks::default();
    // An optional dense group makes the serialized tail deliberately unaligned.
    let dense_groups = [None];
    let mut tail = Vec::new();
    if magic == b"TMC9" {
        let pooled = SeedTerminalDenseCompact::from_map(&seeds);
        bincode::serialize_into(&mut tail, &TokenMaskCachePooledSeedsRef {
            guarded_shift_index: &[],
            seed_terminal_dense: &pooled,
            seed_universe_dense: &[],
            quad_group_sparse_masks: &[],
            quad_group_dense_masks: &dense_groups,
            byte_group_sparse_masks: &[],
            byte_group_dense_masks: &[],
        }).unwrap();
    } else {
        bincode::serialize_into(&mut tail, &TokenMaskCacheIrregularRef {
            guarded_shift_index: &[],
            seed_terminal_dense: &seeds,
            seed_universe_dense: &[],
            quad_group_sparse_masks: &[],
            quad_group_dense_masks: &dense_groups,
            byte_group_sparse_masks: &[],
            byte_group_dense_masks: &[],
        }).unwrap();
    }
    let mut wire = magic.to_vec();
    for field in [tail.len() as u32, mask_words, 1, 1, 2] {
        wire.extend_from_slice(&field.to_le_bytes());
    }
    wire.extend_from_slice(&tail);
    for field in [0u32, 1, word_index, 1 << 31] {
        wire.extend_from_slice(&field.to_le_bytes());
    }
    while wire.len() % 4 != 0 {
        wire.push(0);
    }
    let mut rows = vec![0u32; mask_words as usize * 2];
    if word_index < mask_words {
        rows[mask_words as usize + word_index as usize] = 1 << 31;
    }
    for word in rows {
        wire.extend_from_slice(&word.to_le_bytes());
    }
    wire
}

fn assert_high_word(cache: TokenMaskCacheArtifact) {
    let TokenMaskCacheArtifact::Fast {
        word_group_sparse_masks, word_group_prefix_buf_masks, ..
    } = cache else { panic!("expected the dense-prefix cache"); };
    assert_eq!(word_group_sparse_masks, vec![vec![(65536, 1 << 31)]]);
    assert_eq!(word_group_prefix_buf_masks.len(), 2);
    let last = word_group_prefix_buf_masks.iter().nth(1).unwrap();
    assert_eq!(last[65536], 1 << 31);
    assert!(last[..65536].iter().all(|&word| word == 0));
}

#[test]
fn current_dense_formats_keep_high_words_owned_and_backed() {
    for magic in [b"TMC8", b"TMC9"] {
        let wire = dense_cache_wire(magic, 65537, 65536);
        assert_high_word(decode_token_mask_cache(&wire).unwrap());
        for padding in 0..4 {
            let mut bytes = vec![0u8; padding];
            bytes.extend_from_slice(&wire);
            let backing = Arc::new(bytes);
            let decoded = decode_token_mask_cache_impl(
                &backing, Some((Arc::clone(&backing), 0)),
            ).unwrap();
            assert_high_word(decoded);
        }
    }
}

#[test]
fn retired_cache_formats_are_rejected_with_or_without_padding() {
    for magic in [b"TMC3", b"TMC4", b"TMC5", b"TMC6", b"TMC7"] {
        let mut wire = dense_cache_wire(b"TMC8", 2, 0);
        wire[..4].copy_from_slice(magic);
        for padding in 0..4 {
            let mut bytes = vec![0u8; padding];
            bytes.extend_from_slice(&wire);
            assert!(decode_token_mask_cache(&bytes).is_err());
        }
    }
}

#[test]
fn current_cache_rejects_invalid_sparse_coordinates_and_offsets() {
    assert!(decode_token_mask_cache(&dense_cache_wire(b"TMC8", 2, 65536)).is_err());
    let mut wire = dense_cache_wire(b"TMC8", 2, 0);
    let tail_len = u32::from_le_bytes(wire[4..8].try_into().unwrap()) as usize;
    wire[24 + tail_len + 4..24 + tail_len + 8].copy_from_slice(&2u32.to_le_bytes());
    assert!(decode_token_mask_cache(&wire).is_err());
}

#[test]
fn current_cache_rejects_padding_corruption_and_backing_offset_overflow() {
    let mut wire = dense_cache_wire(b"TMC8", 2, 0);
    let tail_len = u32::from_le_bytes(wire[4..8].try_into().unwrap()) as usize;
    let padding_start = 24 + tail_len + 8 + 8;
    assert_ne!(padding_start % 4, 0, "fixture must exercise internal alignment");
    wire[padding_start] = 1;
    assert!(decode_token_mask_cache(&wire).is_err());

    let mut padded = vec![0];
    padded.extend_from_slice(&dense_cache_wire(b"TMC8", 2, 0));
    let backing = Arc::new(padded);
    let error = decode_token_mask_cache_impl(
        &backing, Some((Arc::clone(&backing), usize::MAX)),
    ).err().expect("invalid backing offset must return an error");
    assert!(error.contains("backing offset overflow"));
}

#[test]
fn current_cache_rejects_truncated_headers_and_trailing_bytes() {
    let mut wire = dense_cache_wire(b"TMC9", 2, 0);
    for len in 0..24 {
        assert!(decode_token_mask_cache(&wire[..len]).is_err());
    }
    assert!(decode_token_mask_cache(&wire[..wire.len() - 1]).is_err());
    wire.push(0);
    assert!(decode_token_mask_cache(&wire).is_err());
}
