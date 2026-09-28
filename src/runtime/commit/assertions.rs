//! Optional commit oracles and diagnostic runtime configuration.

use super::controls::advance_special_token_paths;
use super::controls::merge_special_token_paths;
use super::engine::finish_token_commit;
use super::frontier::coalesce_uniform_runtime_source_states;
use super::frontier::expand_runtime_product_states;
use super::profiled::commit_bytes_impl_profiled;
use super::token_bytes_for_id;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::CommitBuffers;
use crate::runtime::state::ConstraintState;
use crate::runtime::state::ParserStateMap;
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub(super) const COMMIT_ASSERT_MASK_EQUIVALENCE: u8 = 1 << 0;

pub(super) const COMMIT_ASSERT_FAST_PATH_EQUIVALENCE: u8 = 1 << 1;

pub(super) fn commit_assertion_flags() -> u8 {
    static FLAGS: OnceLock<u8> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        let mut flags = 0;
        if cfg!(debug_assertions)
            || std::env::var("GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE")
                .map(|value| {
                    let normalized = value.trim().to_ascii_lowercase();
                    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
                })
                .unwrap_or(false)
        {
            flags |= COMMIT_ASSERT_MASK_EQUIVALENCE;
        }
        if std::env::var("GLRMASK_ASSERT_COMMIT_FAST_PATH_EQUIVALENCE")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
        {
            flags |= COMMIT_ASSERT_FAST_PATH_EQUIVALENCE;
        }
        flags
    })
}

pub(super) fn canonical_commit_state_for_equivalence_assert(
    state: &ParserStateMap,
) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
    let mut grouped = BTreeMap::<u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>>::new();
    for (&tokenizer_state, gss) in state.iter() {
        let out = grouped.entry(tokenizer_state).or_default();
        out.extend(
            gss.to_stacks(100_000)
                .expect("stack enumeration exceeded explicit limit")
                .into_iter()
                .map(|(stack, terminals_disallowed)| {
                    let disallowed = terminals_disallowed
                        .iter()
                        .map(|(&state, terminals)| {
                            (state, terminals.iter().copied().collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>();
                    (stack, disallowed)
                }),
        );
    }
    for stacks in grouped.values_mut() {
        stacks.sort();
        stacks.dedup();
    }
    grouped.into_iter().collect()
}

pub(super) fn profile_allow_fast_paths() -> bool {
    std::env::var("GLRMASK_PROFILE_ALLOW_FAST_PATHS")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub(super) fn token_in_mask(mask: &[u32], token_id: u32) -> bool {
    let word_idx = token_id as usize / 32;
    let bit_idx = token_id as usize % 32;
    word_idx < mask.len() && ((mask[word_idx] >> bit_idx) & 1) != 0
}

pub(super) fn snapshot_mask_membership(
    state: &ConstraintState<'_>,
    token_id: u32,
    assertion_flags: u8,
) -> Option<bool> {
    if assertion_flags & COMMIT_ASSERT_MASK_EQUIVALENCE == 0 {
        return None;
    }
    let mut mask = vec![0u32; state.constraint.mask_len()];
    state.fill_mask(&mut mask);
    Some(token_in_mask(&mask, token_id))
}

pub(super) fn format_token_bytes(token_bytes: &[u8]) -> String {
    let mut escaped = String::new();
    for byte in token_bytes {
        for ch in std::ascii::escape_default(*byte) {
            escaped.push(ch as char);
        }
    }
    format!("b\"{}\"", escaped)
}

pub(super) fn format_optional_token_bytes(token_bytes: Option<&[u8]>) -> String {
    token_bytes.map(format_token_bytes).unwrap_or_else(|| "<no vocabulary bytes>".to_owned())
}

#[inline]
pub(super) fn assert_commit_oracles(
    constraint: &Constraint,
    token_id: u32,
    token_bytes: Option<&[u8]>,
    was_in_mask: Option<bool>,
    fast_path_reference: Option<ParserStateMap>,
    actual_state: &ParserStateMap,
    commit_succeeded: bool,
) {
    if let Some(was_in_mask) = was_in_mask {
        assert!(
            commit_succeeded == was_in_mask,
            "commit/mask mismatch for token_id {} bytes {}: token_in_mask={} commit_succeeded={}",
            token_id,
            format_optional_token_bytes(token_bytes),
            was_in_mask,
            commit_succeeded,
        );
    }
    if let Some(reference_state) = fast_path_reference {
        assert_commit_fast_path_equivalence(
            constraint,
            reference_state,
            token_id,
            actual_state,
            commit_succeeded,
        );
    }
}

#[cold]
pub(super) fn commit_token_no_fast_path_reference(
    constraint: &Constraint,
    state: &mut ParserStateMap,
    token_id: u32,
) -> Result<(), String> {
    let bytes = token_bytes_for_id(constraint, token_id);
    let has_special = constraint.has_special_token_id(token_id);
    if bytes.is_none() && !has_special {
        return Err(format!(
            "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
        ));
    }

    expand_runtime_product_states(constraint, state);
    let special_paths =
        has_special.then(|| advance_special_token_paths(constraint, state, token_id)).flatten();
    if let Some(bytes) = bytes.filter(|piece| !piece.is_empty()) {
        let mut buffers = CommitBuffers::default();
        if commit_bytes_impl_profiled(constraint, state, bytes, &mut buffers, None, false).is_err()
        {
            state.clear();
        }
    } else {
        state.clear();
    }
    merge_special_token_paths(constraint, state, special_paths)?;
    coalesce_uniform_runtime_source_states(constraint, state);
    finish_token_commit(state)
}

#[cold]
pub(super) fn assert_commit_fast_path_equivalence(
    constraint: &Constraint,
    mut reference_state: ParserStateMap,
    token_id: u32,
    actual_state: &ParserStateMap,
    actual_succeeded: bool,
) {
    let reference_result =
        commit_token_no_fast_path_reference(constraint, &mut reference_state, token_id);
    assert_eq!(
        actual_succeeded,
        reference_result.is_ok(),
        "commit fast-path result mismatch for token_id {token_id}: actual_succeeded={} reference={:?}",
        actual_succeeded,
        reference_result,
    );
    let actual_canonical = canonical_commit_state_for_equivalence_assert(actual_state);
    let reference_canonical = canonical_commit_state_for_equivalence_assert(&reference_state);
    if actual_canonical != reference_canonical {
        // Bounded flat frontiers deliberately preserve lexer/parser
        // correlation that the map-only reference normalizes away. The fast
        // state can therefore be a strict internal refinement of the reference
        // without changing the accepted continuation language. Keep exact
        // state equality as the strongest oracle, but fall back to the public
        // semantic observations when the reference has merged correlations.
        let mut actual_semantic = constraint.start();
        actual_semantic.state = actual_state.clone();
        let mut reference_semantic = constraint.start();
        reference_semantic.state = reference_state;
        assert_eq!(
            actual_semantic.is_accepting(),
            reference_semantic.is_accepting(),
            "commit fast-path completion mismatch for token_id {token_id} bytes {}\nactual={actual_canonical:?}\nreference={reference_canonical:?}",
            format_optional_token_bytes(token_bytes_for_id(constraint, token_id)),
        );
        assert_eq!(
            actual_semantic.mask(),
            reference_semantic.mask(),
            "commit fast-path successor-mask mismatch for token_id {token_id} bytes {}\nactual={actual_canonical:?}\nreference={reference_canonical:?}",
            format_optional_token_bytes(token_bytes_for_id(constraint, token_id)),
        );
    }
}
