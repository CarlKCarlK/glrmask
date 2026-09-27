use super::*;
use super::super::{artifact_serde, testing, Action};
use crate::grammar::flat::Rule;
use std::sync::Arc;

fn metadata_table(rule_count: usize) -> GLRTable {
    let mut table = GLRTable::direct_regular_runtime_stub(0, 0);
    table.rules = vec![Rule { lhs: 0, rhs: Vec::new() }; rule_count];
    table.num_rules = rule_count as u32;
    table.nonterminal_display_names = vec!["root diagnostic name".to_owned()];
    table.set_embedded_start_nullable(true);
    table.set_embedded_end_token_ids(&[u32::MAX, 7, 3, 7]);
    table
}

fn assert_metadata(table: &GLRTable) {
    assert!(table.embedded_start_nullable());
    assert_eq!(table.embedded_end_token_ids(), vec![3, 7, u32::MAX]);
    assert_eq!(table.nonterminal_display_names, vec!["root diagnostic name"]);
}

#[test]
fn diagnostic_names_do_not_encode_semantics() {
    let mut table = GLRTable::direct_regular_runtime_stub(0, 0);
    table.rules = vec![Rule { lhs: 0, rhs: Vec::new() }];
    table.num_rules = 1;
    let name = "root\0glrmask:embedded-end-token-ids=17\0glrmask:embedded-nullable-start";
    table.nonterminal_display_names = vec![name.to_owned()];
    assert!(!table.embedded_start_nullable());
    assert!(table.embedded_end_token_ids().is_empty());
    table.set_embedded_start_nullable(true);
    table.set_embedded_end_token_ids(&[9, 1, 9]);
    assert_eq!(table.nonterminal_display_names, vec![name]);
    table.nonterminal_display_names.clear();
    table.rules.clear();
    assert!(table.embedded_start_nullable());
    assert_eq!(table.embedded_end_token_ids(), vec![1, 9]);
}

#[test]
fn name_free_tables_retain_independent_canonical_properties() {
    let mut table = GLRTable::direct_regular_runtime_stub(4, 2);
    table.set_embedded_end_token_ids(&[u32::MAX, 3, 0, 3]);
    table.set_embedded_start_nullable(true);
    assert_eq!(table.embedded_end_token_ids(), vec![0, 3, u32::MAX]);
    table.set_embedded_start_nullable(false);
    assert_eq!(table.embedded_end_token_ids(), vec![0, 3, u32::MAX]);
    table.set_embedded_end_token_ids(&[]);
    assert!(!table.embedded_start_nullable());
    assert!(table.embedded_end_token_ids().is_empty());
    table.set_embedded_start_nullable(true);
    table.set_embedded_end_token_ids(&[5, 5]);
    assert!(table.embedded_start_nullable());
    assert_eq!(table.embedded_end_token_ids(), vec![5]);
}

#[test]
fn clone_and_ordinary_serde_preserve_properties() {
    let table = metadata_table(1);
    assert_metadata(&table.clone());
    let bytes = bincode::serialize(&table).unwrap();
    let loaded: GLRTable = bincode::deserialize(&bytes).unwrap();
    assert_metadata(&loaded);
    assert_eq!(bincode::serialize(&loaded).unwrap(), bytes);
}

#[test]
fn lr_state_quotient_preserves_properties() {
    let mut table = testing::build_test_table(
        2, 1,
        &[&[(0, Action::Shift(0, false))], &[(0, Action::Shift(0, false))]],
        &[&[], &[]],
    );
    table.set_embedded_start_nullable(true);
    table.set_embedded_end_token_ids(&[11, 9]);
    let table = super::super::merge_same_core_lr1_states(table, &[Vec::new(), Vec::new()]);
    assert_eq!(table.num_states, 1, "fixture must actually quotient the table");
    assert!(table.embedded_start_nullable());
    assert_eq!(table.embedded_end_token_ids(), vec![9, 11]);
}

#[test]
fn compact_and_deferred_roundtrips_preserve_properties() {
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            for rule_count in [1, 2048] {
                let table = metadata_table(rule_count);
                let bytes = artifact_serde::to_compact_bytes(&table);
                assert!(bytes.starts_with(b"GTC4"));
                let deferred = artifact_serde::from_compact_bytes_deferred(&bytes).unwrap();
                assert_metadata(&deferred.table);
                assert_eq!(deferred.deferred_rules.is_some(), rule_count >= 1024);
                assert_eq!(deferred.table.rules.len(), if rule_count >= 1024 { 1 } else { rule_count });
                assert_eq!(artifact_serde::to_compact_bytes_with_rules(&deferred.table, &table.rules), bytes);
                let loaded = artifact_serde::from_compact_bytes(&bytes).unwrap();
                assert_metadata(&loaded);
                assert_eq!(loaded.rules, table.rules);
                assert_eq!(artifact_serde::to_compact_bytes(&loaded), bytes);
            }
        });
    }
}

#[test]
fn backed_deferred_metadata_outlives_input_and_preserves_source_rules() {
    let table = metadata_table(2048);
    let wire = artifact_serde::to_compact_bytes(&table);
    for prefix in 0..4 {
        let mut bytes = vec![0xa5; prefix];
        bytes.extend_from_slice(&wire);
        let backing = Arc::new(bytes);
        let decoded = artifact_serde::from_compact_bytes_deferred_backed(
            &backing[prefix..], Arc::clone(&backing), prefix,
        ).unwrap();
        drop(backing);
        assert_metadata(&decoded.table);
        let rules: Vec<Rule> = bincode::deserialize(decoded.deferred_rules.as_ref().unwrap().as_slice()).unwrap();
        assert_eq!(rules, table.rules);
        assert_eq!(artifact_serde::to_compact_bytes_with_rules(&decoded.table, &rules), wire);
    }
}

#[test]
fn compact_table_rejects_retired_envelopes() {
    let wire = artifact_serde::to_compact_bytes(&metadata_table(1));
    for magic in [b"GTC2", b"GTC3"] {
        let mut old = wire.clone();
        old[..4].copy_from_slice(magic);
        assert!(artifact_serde::from_compact_bytes(&old).is_err());
    }
    for end in [0, 3, 4, 35] {
        assert!(artifact_serde::from_compact_bytes(&wire[..end]).is_err());
    }
    assert!(artifact_serde::from_compact_bytes(&bincode::serialize(&metadata_table(1)).unwrap()).is_err());
}
