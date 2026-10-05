use super::*;

fn old_runtime_wire(
    metadata: &ConstraintArtifactCurrentRuntimeRef<'_>,
    residual: &[u8],
) -> Vec<u8> {
    let mut meta = Vec::with_capacity(256 * 1024);
    bincode::serialize_into(&mut meta, metadata).unwrap();
    let mut out = Vec::with_capacity(CURRENT_RUNTIME_HEADER_LEN + meta.len() + residual.len());
    out.extend_from_slice(&CURRENT_RUNTIME_MAGIC);
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    out.extend_from_slice(&(residual.len() as u64).to_le_bytes());
    out.extend_from_slice(&meta);
    out.extend_from_slice(residual);
    out
}

#[test]
fn direct_runtime_metadata_writer_preserves_every_byte() {
    let terminal_live_states = vec![vec![0, 3, 1000], vec![], vec![7]];
    let ids = vec![1u32, 4, 7];
    let rows = vec![0u64, u64::MAX, 0x1234_5678_9abc_def0];

    let metadata = ConstraintArtifactCurrentRuntimeRef {
        template_dynamic_proofs: None,
        terminal_live_states: &terminal_live_states,
        segmented_runtime: None,
        dynamic_mask_vocab: None,
        virtual_runtimes: Vec::new(),
        static_virtual_residual_mask: None,
        packed_dwa_dense_mask_ids: &ids,
        packed_dwa_dense_mask_rows: &rows,
    };

    for size in [0usize, 1, 31, 4096, 300_000] {
        let residual = (0..size).map(|index| index as u8).collect::<Vec<_>>();
        assert_eq!(
            encode_current_runtime_wire(&metadata, &residual),
            old_runtime_wire(&metadata, &residual),
        );
    }
}

#[test]
fn runtime_writer_still_decodes_with_the_existing_reader() {
    let metadata = ConstraintArtifactCurrentRuntimeRef {
        template_dynamic_proofs: None,
        terminal_live_states: &[],
        segmented_runtime: None,
        dynamic_mask_vocab: None,
        virtual_runtimes: Vec::new(),
        static_virtual_residual_mask: None,
        packed_dwa_dense_mask_ids: &[],
        packed_dwa_dense_mask_rows: &[],
    };

    let bytes = encode_current_runtime_wire(&metadata, &[]);
    let backing = Arc::new(bytes);
    let (decoded, residual) =
        decode_current_runtime_wire(backing.as_slice(), Arc::clone(&backing)).unwrap();
    assert!(decoded.template_dynamic_proofs.is_none());
    assert!(decoded.terminal_live_states.is_empty());
    assert!(decoded.virtual_runtimes.is_empty());
    assert!(residual.is_none());
}

#[test]
fn runtime_section_length_corruption_remains_an_error() {
    let metadata = ConstraintArtifactCurrentRuntimeRef {
        template_dynamic_proofs: None,
        terminal_live_states: &[],
        segmented_runtime: None,
        dynamic_mask_vocab: None,
        virtual_runtimes: Vec::new(),
        static_virtual_residual_mask: None,
        packed_dwa_dense_mask_ids: &[],
        packed_dwa_dense_mask_rows: &[],
    };

    for range in [4..12, 12..20] {
        let mut bytes = encode_current_runtime_wire(&metadata, &[]);
        bytes[range].copy_from_slice(&u64::MAX.to_le_bytes());
        let backing = Arc::new(bytes);
        assert!(
            decode_current_runtime_wire(backing.as_slice(), Arc::clone(&backing))
                .is_err()
        );
    }
}

#[test]
fn ordinary_composition_metadata_unwrap_has_identical_bytes() {
    let vocab = crate::Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let constraint = crate::Grammar::glrm("start root; nt root ::= 'a' 'b'?;")
        .compile_with(
            &vocab,
            crate::BuildOptions::default()
                .optimization(crate::Optimization::FastBuild),
        )
        .unwrap();

    let base = encode_composition_metadata_base_for_save(&constraint);
    let expected = if let Some(wire) =
        crate::compiler::boundary_precomputed_completion::saved_wire(&constraint)
    {
        crate::compiler::boundary_precomputed_completion::wrap_envelope(base, wire)
            .unwrap()
    } else {
        crate::compiler::boundary_precomputed_completion::split_envelope(&base)
            .unwrap()
            .0
            .to_vec()
    };
    assert_eq!(encode_composition_metadata_for_save(&constraint), expected);
}
