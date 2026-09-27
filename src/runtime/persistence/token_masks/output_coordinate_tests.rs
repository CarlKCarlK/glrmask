use super::*;
use std::sync::Arc;

fn slab(magic: &[u8; 4]) -> Vec<u8> {
    let mut bytes = magic.to_vec();
    for value in [2u32, 3, 0, 2, 3, 0, 1, 65536, 1 << 31, u32::MAX / 32, 4] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn word_sparse_wire_keeps_full_output_coordinates() {
    let decoded = decode_word_sparse_token_mask_cache(&slab(b"TWS2")).unwrap();
    assert_eq!(decoded, vec![vec![(0, 1), (65536, 1 << 31)], vec![(u32::MAX / 32, 4)]]);
    assert!(decode_word_sparse_token_mask_cache(&slab(b"TWS1")).is_err());
}

#[test]
fn native_wire_keeps_high_indices_owned_backed_and_unaligned() {
    let expected = vec![(0, 1), (65536, 1 << 31), (u32::MAX / 32, 4)];
    let wire = slab(b"IBM3");
    let owned = decode_internal_token_buf_masks(&wire, None).unwrap();
    assert_eq!(owned.offsets.as_ref(), &[0, 2, 3]);
    assert_eq!(owned.flat.iter().map(|e| (e.word_idx, e.mask)).collect::<Vec<_>>(), expected);
    assert_eq!(std::mem::size_of::<PackedInternalTokenBufMask>(), 8);
    for prefix in 0..4 {
        let mut bytes = vec![0u8; prefix];
        bytes.extend_from_slice(&wire);
        let backing = Arc::new(bytes);
        let loaded = decode_internal_token_buf_masks(
            &backing[prefix..], Some((Arc::clone(&backing), prefix)),
        ).unwrap();
        let entries = loaded.backed.as_ref().unwrap();
        let mut actual = Vec::new();
        entries.for_each_range(0, entries.len(), |word, mask| actual.push((word, mask)));
        assert_eq!(actual, expected, "backing offset={prefix}");
        let mut saved = Vec::new();
        entries.append_wire_bytes(&mut saved);
        assert_eq!(saved, wire[24..]);
    }
    assert!(decode_internal_token_buf_masks(&slab(b"IBM2"), None).is_err());
    assert!(decode_internal_token_buf_masks(&slab(b"IBM1"), None).is_err());
}
