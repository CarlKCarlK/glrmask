//! Certify full-to-synthesized state maps using structural or vocabulary-relative proofs.
//! A smaller mask automaton never substitutes for the exact commit language without a proof.

use crate::Vocab;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::bitset::BitSet;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use super::{Regex, compile_with_plan, compile_with_plan_internal};
use super::bounded_repeat::{
    build_bounded_repeat_dfa_from_base,
    collect_suffix_bytes,
    compile_direct_bounded_repeat_base_dfa_unconditionally,
    prepend_literal_prefix_to_dfa,
};
use super::deferred::{DeferredDfa, try_compile_with_plan_deferred_dense};
use super::factor::{seq_from_parts, unwrap_shared};
use super::nfa::{compile_expr_to_dfa, expr_u8set};
use super::plan::{
    build_exclusion_compile_plan,
    expr_profile_summary,
    lift_single_nested_intersection,
    rebuild_single_visible_group_expression,
};
use super::product::{
    ProductBuildTrace,
    ProductComponent,
    ProductComponentClassTransitions,
    ProductStateTuple,
    ProductStateTuples,
    build_product_class_transitions,
    certify_supplied_dfa_state_homomorphism,
    compute_product_equivalence_classes,
    deterministic_component_homomorphism_state_map,
    product_component_state_flags,
    product_state_metadata,
    product_state_single_visible_finalizer,
    pure_binary_intersection,
};
use super::repeat_horizon::VocabularyRepeatHorizonCache;
use super::repeat_suffix::{
    ZeroMinRepeatSuffixState,
    build_zero_min_repeat_suffix_dominance_dfa_internal,
    close_zero_min_repeat_suffix_state,
};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct CertifiedVocabularyExactStateCandidates {
    primary: Vec<u32>,
    candidates_by_full_state: Vec<Arc<[u32]>>,
}

impl CertifiedVocabularyExactStateCandidates {
    pub fn new(primary: Vec<u32>, candidates_by_full_state: Vec<Arc<[u32]>>) -> Self {
        Self { primary, candidates_by_full_state }
    }

    pub fn primary(&self) -> &[u32] {
        &self.primary
    }

    pub fn visit_candidates(
        &self,
        full_state: u32,
        mut visit: impl FnMut(u32) -> bool,
    ) -> bool {
        self.candidates_by_full_state[full_state as usize]
            .iter()
            .copied()
            .any(&mut visit)
    }
}

pub type VocabularyExactStateCertifier = fn(
    &Tokenizer,
    &Tokenizer,
    &Vocab,
    Option<&[bool]>,
) -> Option<CertifiedVocabularyExactStateCandidates>;

static VOCABULARY_EXACT_STATE_CERTIFIER: OnceLock<VocabularyExactStateCertifier> = OnceLock::new();

pub fn install_vocabulary_exact_state_certifier(certifier: VocabularyExactStateCertifier) {
    if let Some(existing) = VOCABULARY_EXACT_STATE_CERTIFIER.get() {
        assert!(
            std::ptr::fn_addr_eq(*existing, certifier),
            "vocabulary exact-state certifier installed more than once with different implementations",
        );
        return;
    }
    let _ = VOCABULARY_EXACT_STATE_CERTIFIER.set(certifier);
}

fn product_component_mapping_dfa(component: &ProductComponent) -> Option<DFA> {
    match component {
        ProductComponent::Materialized(dfa)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
            Some(dfa.as_ref().clone())
        }
        ProductComponent::VirtualFixedSequence { .. } => None,
        ProductComponent::VirtualBoundedRepeat {
            base_dfa,
            min,
            max,
        } => build_bounded_repeat_dfa_from_base(base_dfa, *min as usize, *max as usize),
    }
}

fn exact_combined_dfa_byte_representatives(
    full: &DFA,
    synthesized: &DFA,
    relevant_bytes: &[u8],
) -> Vec<u8> {
    const HASH_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;
    let mut bytes = relevant_bytes.to_vec();
    bytes.sort_unstable();
    bytes.dedup();
    if bytes.len() <= 1 {
        return bytes;
    }

    let mut hashes = FxHashMap::<u64, Vec<u8>>::default();
    for &byte in &bytes {
        let mut hash = 0u64;
        for dfa in [synthesized, full] {
            for state in 0..dfa.num_states() as u32 {
                hash = hash
                    .wrapping_mul(HASH_MULTIPLIER)
                    .wrapping_add(dfa.step(state, byte).unwrap_or(u32::MAX) as u64);
            }
        }
        hashes.entry(hash).or_default().push(byte);
    }

    let columns_equal = |left: u8, right: u8| {
        [synthesized, full].into_iter().all(|dfa| {
            (0..dfa.num_states() as u32)
                .all(|state| dfa.step(state, left) == dfa.step(state, right))
        })
    };
    let mut representatives = Vec::new();
    for candidates in hashes.into_values() {
        let mut local = Vec::<u8>::new();
        for byte in candidates {
            if local
                .iter()
                .copied()
                .any(|representative| columns_equal(byte, representative))
            {
                continue;
            }
            local.push(byte);
        }
        representatives.extend(local);
    }
    representatives.sort_unstable();
    representatives
}

fn exact_kbounded_single_group_state_map(
    full: &DFA,
    synthesized: &DFA,
    depth: usize,
    relevant_bytes: &[u8],
) -> Option<Vec<u32>> {
    if full.num_groups() != synthesized.num_groups() {
        return None;
    }
    let relevant_bytes =
        exact_combined_dfa_byte_representatives(full, synthesized, relevant_bytes);
    let synthesized_states = synthesized.num_states();
    let full_states = full.num_states();
    let label = |dfa: &DFA, state: u32| {
        (
            dfa.finalizers(state).clone(),
            dfa.possible_future_group_ids(state).clone(),
        )
    };
    let synthesized_labels = (0..synthesized_states as u32)
        .map(|state| label(synthesized, state))
        .collect::<Vec<_>>();
    let full_labels = (0..full_states as u32)
        .map(|state| label(full, state))
        .collect::<Vec<_>>();

    let mut label_classes = FxHashMap::<(BitSet, BitSet), u32>::default();
    let mut synthesized_classes = synthesized_labels
        .iter()
        .map(|state_label| {
            let next = label_classes.len() as u32;
            *label_classes.entry(state_label.clone()).or_insert(next)
        })
        .collect::<Vec<_>>();
    let mut full_classes = full_labels
        .iter()
        .map(|state_label| label_classes.get(state_label).copied())
        .collect::<Option<Vec<_>>>()?;

    let mut synthesized_signature = vec![0u32; 1 + relevant_bytes.len()];
    for _ in 0..depth {
        let mut signature_to_class = FxHashMap::<Vec<u32>, u32>::default();
        let mut next_synthesized = vec![0u32; synthesized_states];
        for state in 0..synthesized_states as u32 {
            synthesized_signature[0] = label_classes[&synthesized_labels[state as usize]];
            for (slot, &byte) in relevant_bytes.iter().enumerate() {
                synthesized_signature[slot + 1] = synthesized
                    .step(state, byte)
                    .map_or(u32::MAX, |target| synthesized_classes[target as usize]);
            }
            let next = signature_to_class.len() as u32;
            next_synthesized[state as usize] = *signature_to_class
                .entry(synthesized_signature.clone())
                .or_insert(next);
        }

        let next_full = (0..full_states as u32)
            .into_par_iter()
            .map_init(
                || vec![0u32; 1 + relevant_bytes.len()],
                |signature, state| {
                    signature[0] = label_classes[&full_labels[state as usize]];
                    for (slot, &byte) in relevant_bytes.iter().enumerate() {
                        signature[slot + 1] = full
                            .step(state, byte)
                            .map_or(u32::MAX, |target| full_classes[target as usize]);
                    }
                    signature_to_class.get(signature).copied()
                },
            )
            .collect::<Option<Vec<_>>>()?;
        synthesized_classes = next_synthesized;
        full_classes = next_full;
    }

    let class_count = synthesized_classes
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    let mut synthesized_for_class = vec![u32::MAX; class_count];
    for (state, &class) in synthesized_classes.iter().enumerate() {
        synthesized_for_class[class as usize] = state as u32;
    }
    full_classes
        .into_iter()
        .map(|class| {
            synthesized_for_class
                .get(class as usize)
                .copied()
                .filter(|&state| state != u32::MAX)
        })
        .collect()
}

struct DirectBoundedSuffixShape<'a> {
    prefix: Vec<u8>,
    body: &'a Expr,
    min: usize,
    max: usize,
    suffix: Vec<u8>,
}

pub(super) struct ZeroMinRepeatSuffixComponentTrace {
    pub(super) dfa: Arc<DFA>,
    pub(super) prefix_len: usize,
    pub(super) body_dfa: DFA,
    pub(super) suffix_dfa: DFA,
    pub(super) max: usize,
    pub(super) tail_states: Vec<ZeroMinRepeatSuffixState>,
    pub(super) tail_state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
}

fn dfa_has_same_numbered_layout(left: &DFA, right: &DFA) -> bool {
    left.num_groups() == right.num_groups()
        && left.num_states() == right.num_states()
        && (0..left.num_groups()).all(|group| {
            left.group_id_to_u8set(group as u32)
                == right.group_id_to_u8set(group as u32)
        })
        && left
            .states()
            .iter()
            .zip(right.states())
            .enumerate()
            .all(|(state, (left_state, right_state))| {
                left_state.transitions == right_state.transitions
                    && left_state.epsilon_transitions == right_state.epsilon_transitions
                    && left_state.finalizers == right_state.finalizers
                    && left.possible_future_group_ids(state as u32)
                        == right.possible_future_group_ids(state as u32)
            })
}

pub(super) fn zero_min_repeat_suffix_component_trace(
    expr: &Expr,
) -> Option<ZeroMinRepeatSuffixComponentTrace> {
    let expr = unwrap_shared(expr);
    let Expr::Seq(parts) = expr else {
        return None;
    };
    let mut flat_parts = Vec::<Expr>::new();
    for part in parts {
        match unwrap_shared(part) {
            Expr::Seq(inner) => flat_parts.extend(inner.iter().cloned()),
            _ => flat_parts.push(part.clone()),
        }
    }

    for repeat_index in 0..flat_parts.len().saturating_sub(1) {
        let Expr::Repeat {
            expr: body,
            min: 0,
            max: Some(max),
        } = unwrap_shared(&flat_parts[repeat_index])
        else {
            continue;
        };
        if *max == 0 {
            continue;
        }
        let prefix = if repeat_index == 0 {
            Vec::new()
        } else {
            collect_suffix_bytes(&flat_parts[..repeat_index])?
        };
        let suffix_expr = seq_from_parts(flat_parts[repeat_index + 1..].to_vec());
        let body_dfa = compile_expr_to_dfa(body);
        let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
        if body_dfa.num_states() == 0
            || body_dfa.num_states() > 256
            || body_dfa.finalizers(0).contains(0)
            || suffix_dfa.num_states() == 0
            || suffix_dfa.finalizers(0).contains(0)
        {
            continue;
        }
        let built = build_zero_min_repeat_suffix_dominance_dfa_internal(
            &body_dfa,
            &suffix_dfa,
            *max,
            true,
        )?;
        let mut dfa = prepend_literal_prefix_to_dfa(&prefix, built.dfa)?;
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        return Some(ZeroMinRepeatSuffixComponentTrace {
            dfa: Arc::new(dfa),
            prefix_len: prefix.len(),
            body_dfa,
            suffix_dfa,
            max: *max,
            tail_states: built.states,
            tail_state_by_key: built.state_by_key,
        });
    }
    None
}

pub(super) struct ZeroMinRepeatSuffixStateMap {
    primary: Vec<u32>,
    full_trace: Arc<ZeroMinRepeatSuffixComponentTrace>,
    synthesized_trace: Arc<ZeroMinRepeatSuffixComponentTrace>,
    body_state_map: Arc<[u32]>,
    suffix_state_map: Arc<[u32]>,
    crossed_boundaries: u32,
    interior_representative: u32,
    full_max: u32,
    synthesized_max: u32,
}

impl ZeroMinRepeatSuffixStateMap {
    fn primary(&self) -> &[u32] {
        &self.primary
    }

    fn visit_candidates(&self, full_state: u32, mut visit: impl FnMut(u32) -> bool) -> bool {
        let primary = self.primary[full_state as usize];
        if visit(primary) {
            return true;
        }
        let Some(state) = (full_state as usize)
            .checked_sub(self.full_trace.prefix_len)
            .and_then(|index| self.full_trace.tail_states.get(index))
        else {
            return false;
        };
        let Some(minimum_completed) = state
            .body_min_counts
            .iter()
            .copied()
            .filter(|&count| count != u32::MAX)
            .min()
        else {
            return false;
        };
        let maximum_completed = state
            .body_min_counts
            .iter()
            .copied()
            .filter(|&count| count != u32::MAX)
            .max()
            .unwrap_or(minimum_completed);
        let live_count_span = maximum_completed.saturating_sub(minimum_completed);
        let live_count_headroom = live_count_span.max(1);
        let Some(distance_to_upper) = self.full_max.checked_sub(minimum_completed) else {
            return false;
        };
        if distance_to_upper <= self.crossed_boundaries {
            return false;
        }

        let mapped_minimum = self.interior_representative;
        let mut seen = SmallVec::<[u32; 8]>::from_slice(&[primary]);
        for alternative_minimum in 0..=self.synthesized_max {
            if alternative_minimum == mapped_minimum {
                continue;
            }
            if alternative_minimum
                .checked_add(live_count_headroom)
                .and_then(|count| count.checked_add(self.crossed_boundaries))
                .is_none_or(|required| required > self.synthesized_max)
            {
                continue;
            }
            let Some(alternative) = zero_min_repeat_suffix_candidate(
                state,
                Some(minimum_completed),
                Some(alternative_minimum),
                &self.body_state_map,
                &self.suffix_state_map,
                &self.synthesized_trace,
                self.synthesized_max,
            ) else {
                continue;
            };
            if self.full_trace.dfa.finalizers(full_state)
                != self.synthesized_trace.dfa.finalizers(alternative)
                || self.full_trace.dfa.possible_future_group_ids(full_state)
                    != self
                        .synthesized_trace
                        .dfa
                        .possible_future_group_ids(alternative)
                || seen.contains(&alternative)
            {
                continue;
            }
            seen.push(alternative);
            if visit(alternative) {
                return true;
            }
        }
        false
    }
}

fn zero_min_repeat_suffix_candidate(
    state: &ZeroMinRepeatSuffixState,
    minimum_completed: Option<u32>,
    mapped_minimum: Option<u32>,
    body_state_map: &[u32],
    suffix_state_map: &[u32],
    synthesized_trace: &ZeroMinRepeatSuffixComponentTrace,
    synthesized_max: u32,
) -> Option<u32> {
    let mut body_min_counts = vec![u32::MAX; synthesized_trace.body_dfa.num_states()];
    for (full_body_state, &completed) in state.body_min_counts.iter().enumerate() {
        if completed == u32::MAX {
            continue;
        }
        let minimum = minimum_completed?;
        let mapped_completed = mapped_minimum?.checked_add(completed.checked_sub(minimum)?)?;
        if mapped_completed > synthesized_max {
            return None;
        }
        let mapped_body_state = body_state_map[full_body_state] as usize;
        body_min_counts[mapped_body_state] =
            body_min_counts[mapped_body_state].min(mapped_completed);
    }
    let mut suffix_states = state
        .suffix_states
        .iter()
        .map(|&state| suffix_state_map[state as usize])
        .collect::<Vec<u32>>();
    close_zero_min_repeat_suffix_state(
        &mut body_min_counts,
        &mut suffix_states,
        &synthesized_trace.body_dfa,
        &synthesized_trace.suffix_dfa,
        synthesized_trace.max,
    );
    let mapped_key = ZeroMinRepeatSuffixState {
        body_min_counts: body_min_counts.into_boxed_slice(),
        suffix_states: suffix_states.into_boxed_slice(),
    };
    synthesized_trace
        .tail_state_by_key
        .get(&mapped_key)
        .copied()
        .map(|tail| synthesized_trace.prefix_len as u32 + tail)
}

pub(super) fn zero_min_repeat_suffix_state_map(
    full_expr: &Expr,
    synthesized_expr: &Expr,
    full: &DFA,
    synthesized: &DFA,
    full_trace: Option<Arc<ZeroMinRepeatSuffixComponentTrace>>,
    synthesized_trace: Option<Arc<ZeroMinRepeatSuffixComponentTrace>>,
    max_token_len: usize,
    vocab: Option<&Vocab>,
    repeat_horizons: Option<&VocabularyRepeatHorizonCache>,
) -> Option<ZeroMinRepeatSuffixStateMap> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let full_trace = full_trace
        .or_else(|| zero_min_repeat_suffix_component_trace(full_expr).map(Arc::new))?;
    let synthesized_trace = if let Some(trace) = synthesized_trace {
        trace
    } else if let Some(trace) = zero_min_repeat_suffix_component_trace(synthesized_expr) {
        Arc::new(trace)
    } else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=synthesized_trace full_states={} synthesized_states={}",
                full.num_states(), synthesized.num_states(),
            );
        }
        return None;
    };
    let full_layout = dfa_has_same_numbered_layout(&full_trace.dfa, full);
    let synthesized_layout =
        dfa_has_same_numbered_layout(&synthesized_trace.dfa, synthesized);
    let body_state_map = deterministic_component_homomorphism_state_map(
        &full_trace.body_dfa,
        &synthesized_trace.body_dfa,
    );
    let suffix_state_map = deterministic_component_homomorphism_state_map(
        &full_trace.suffix_dfa,
        &synthesized_trace.suffix_dfa,
    );
    if full_trace.prefix_len != synthesized_trace.prefix_len
        || full_trace.max < synthesized_trace.max
        || !full_layout
        || !synthesized_layout
        || body_state_map.is_none()
        || suffix_state_map.is_none()
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=shape prefix={}/{} max={}/{} trace_states={}/{} actual_states={}/{} full_layout={} synthesized_layout={} body_map={} suffix_map={}",
                full_trace.prefix_len,
                synthesized_trace.prefix_len,
                full_trace.max,
                synthesized_trace.max,
                full_trace.dfa.num_states(),
                synthesized_trace.dfa.num_states(),
                full.num_states(),
                synthesized.num_states(),
                full_layout,
                synthesized_layout,
                body_state_map.is_some(),
                suffix_state_map.is_some(),
            );
        }
        return None;
    }
    let body_state_map = body_state_map.unwrap();
    let suffix_state_map = suffix_state_map.unwrap();

    let minimum_body_width = full_trace.body_dfa.min_match_byte_len()?.max(1);
    let fallback_crossed_boundaries = max_token_len
        .div_ceil(minimum_body_width)
        .saturating_add(1);
    let crossed_boundaries = vocab
        .zip(repeat_horizons)
        .and_then(|(vocab, horizons)| horizons.horizon_for_dfa(&full_trace.body_dfa, vocab))
        .unwrap_or(fallback_crossed_boundaries);
    if synthesized_trace.max <= crossed_boundaries {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=horizon full_max={} synthesized_max={} body_width={} crossed_boundaries={}",
                full_trace.max,
                synthesized_trace.max,
                minimum_body_width,
                crossed_boundaries,
            );
        }
        return None;
    }
    // A dominance state can keep several body residuals alive at different
    // completed-copy counts.  Translating only the minimum count preserves the
    // current state, but a following token can expose the upper-bound
    // difference through the largest live count.  Reserve room for the widest
    // live count spread in addition to the vocabulary displacement horizon.
    // A span of zero still needs one layer for the endpoint future observation;
    // this is the `max(1)` below and matches the historical `+ 1` interior
    // representative formula without double-counting it in the horizon.
    let max_live_count_span = full_trace
        .tail_states
        .iter()
        .filter_map(|state| {
            let mut counts = state
                .body_min_counts
                .iter()
                .copied()
                .filter(|&count| count != u32::MAX);
            let first = counts.next()?;
            let (minimum, maximum) =
                counts.fold((first, first), |(minimum, maximum), count| {
                    (minimum.min(count), maximum.max(count))
                });
            Some(maximum.saturating_sub(minimum) as usize)
        })
        .max()
        .unwrap_or(0)
        .max(1);
    let interior_headroom = crossed_boundaries.checked_add(max_live_count_span)?;
    if synthesized_trace.max < interior_headroom {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=count_span full_max={} synthesized_max={} crossed_boundaries={} max_live_count_span={}",
                full_trace.max,
                synthesized_trace.max,
                crossed_boundaries,
                max_live_count_span,
            );
        }
        return None;
    }
    let interior_representative = synthesized_trace
        .max
        .checked_sub(interior_headroom)? as u32;
    let full_max = full_trace.max as u32;
    let synthesized_max = synthesized_trace.max as u32;

    let mut mapping = Vec::with_capacity(full.num_states());
    for state in 0..full_trace.prefix_len {
        mapping.push(state as u32);
    }
    for state in &full_trace.tail_states {
        let minimum_completed = state
            .body_min_counts
            .iter()
            .copied()
            .filter(|&count| count != u32::MAX)
            .min();
        let mapped_minimum = if let Some(minimum) = minimum_completed {
            let distance_to_upper = full_max.checked_sub(minimum)?;
            if distance_to_upper <= crossed_boundaries as u32 {
                Some(synthesized_max.checked_sub(distance_to_upper)?)
            } else {
                Some(interior_representative)
            }
        } else {
            None
        };
        let Some(mapped) = zero_min_repeat_suffix_candidate(
            state,
            minimum_completed,
            mapped_minimum,
            &body_state_map,
            &suffix_state_map,
            &synthesized_trace,
            synthesized_max,
        ) else {
            if profile {
                let full_counts = state
                    .body_min_counts
                    .iter()
                    .enumerate()
                    .filter_map(|(body_state, &count)| {
                        (count != u32::MAX).then_some((body_state, count))
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=missing_key full_state={} full_max={} synthesized_max={} body_width={} crossed_boundaries={} full_counts={:?} mapped_minimum={:?} suffix_states={:?}",
                    mapping.len(),
                    full_trace.max,
                    synthesized_trace.max,
                    minimum_body_width,
                    crossed_boundaries,
                    full_counts,
                    mapped_minimum,
                    state.suffix_states,
                );
            }
            return None;
        };
        let full_state = mapping.len() as u32;
        if full.finalizers(full_state) != synthesized.finalizers(mapped)
            || full.possible_future_group_ids(full_state)
                != synthesized.possible_future_group_ids(mapped)
        {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] dominance_state_map_rejected stage=metadata full_state={} synthesized_state={}",
                    full_state, mapped,
                );
            }
            return None;
        }
        mapping.push(mapped);
    }
    if mapping.len() != full.num_states() {
        return None;
    }
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] dominance_state_map_accepted full_states={} synthesized_states={} full_max={} synthesized_max={} body_width={} crossed_boundaries={}",
            full.num_states(),
            synthesized.num_states(),
            full_trace.max,
            synthesized_trace.max,
            minimum_body_width,
            crossed_boundaries,
        );
    }
    Some(ZeroMinRepeatSuffixStateMap {
        primary: mapping,
        full_trace,
        synthesized_trace,
        body_state_map: Arc::from(body_state_map.into_boxed_slice()),
        suffix_state_map: Arc::from(suffix_state_map.into_boxed_slice()),
        crossed_boundaries: crossed_boundaries as u32,
        interior_representative,
        full_max,
        synthesized_max,
    })
}

pub(super) struct DirectBoundedSuffixStateMap {
    primary: Vec<u32>,
    prefix_states: usize,
    full_base_states: usize,
    synthesized_base_states: usize,
    full_max: usize,
    synthesized_max: usize,
    min: usize,
    crossed_boundaries: usize,
    base_state_map: Vec<u32>,
    full_suffix_start: usize,
    synthesized_suffix_start: usize,
}

impl DirectBoundedSuffixStateMap {
    pub(super) fn primary(&self) -> &[u32] {
        &self.primary
    }

    fn visit_candidates(&self, full_state: u32, mut visit: impl FnMut(u32) -> bool) -> bool {
        let full_state = full_state as usize;
        let primary = self.primary[full_state];
        if visit(primary) {
            return true;
        }
        if full_state < self.prefix_states || full_state >= self.full_suffix_start {
            return false;
        }

        let local = full_state - self.prefix_states;
        let completed = local / self.full_base_states;
        let body_state = local % self.full_base_states;
        let distance_to_upper = self.full_max - completed;
        if completed < self.min || distance_to_upper <= self.crossed_boundaries {
            return false;
        }

        let Some(last_interior) = self
            .synthesized_max
            .checked_sub(self.crossed_boundaries.saturating_add(1))
        else {
            return false;
        };
        let mapped_body_state = self.base_state_map[body_state] as usize;
        for mapped_completed in self.min..=last_interior {
            let candidate = (self.prefix_states
                + mapped_completed * self.synthesized_base_states
                + mapped_body_state) as u32;
            if candidate != primary && visit(candidate) {
                return true;
            }
        }
        false
    }
}

pub(super) enum ProductComponentStateMap {
    Fixed(Vec<u32>),
    Layered(DirectBoundedSuffixStateMap),
    Dominance(ZeroMinRepeatSuffixStateMap),
    Vocabulary(CertifiedVocabularyExactStateCandidates),
}

impl ProductComponentStateMap {
    pub(super) fn primary(&self) -> &[u32] {
        match self {
            Self::Fixed(mapping) => mapping,
            Self::Layered(mapping) => mapping.primary(),
            Self::Dominance(mapping) => mapping.primary(),
            Self::Vocabulary(mapping) => mapping.primary(),
        }
    }

    fn visit_candidates(&self, full_state: u32, visit: impl FnMut(u32) -> bool) -> bool {
        match self {
            Self::Fixed(mapping) => {
                let mut visit = visit;
                visit(mapping[full_state as usize])
            }
            Self::Layered(mapping) => mapping.visit_candidates(full_state, visit),
            Self::Dominance(mapping) => mapping.visit_candidates(full_state, visit),
            Self::Vocabulary(mapping) => mapping.visit_candidates(full_state, visit),
        }
    }

    fn is_flexible(&self) -> bool {
        matches!(
            self,
            Self::Layered(_) | Self::Dominance(_) | Self::Vocabulary(_)
        )
    }
}

fn direct_bounded_suffix_shape(expr: &Expr) -> Option<DirectBoundedSuffixShape<'_>> {
    let expr = match expr {
        Expr::Shared(inner) => inner.as_ref(),
        expr => expr,
    };
    let Expr::Seq(parts) = expr else {
        return None;
    };
    let mut flat_parts = Vec::new();
    for part in parts {
        match part {
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter()),
                _ => flat_parts.push(part),
            },
            Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter()),
            _ => flat_parts.push(part),
        }
    }
    for repeat_index in 0..flat_parts.len() {
        let repeat = match flat_parts[repeat_index] {
            Expr::Shared(inner) => inner.as_ref(),
            expr => expr,
        };
        let Expr::Repeat {
            expr: body,
            min,
            max: Some(max),
        } = repeat
        else {
            continue;
        };
        let prefix = if repeat_index == 0 {
            Vec::new()
        } else {
            collect_suffix_bytes(
                &flat_parts[..repeat_index]
                    .iter()
                    .map(|expr| (*expr).clone())
                    .collect::<Vec<_>>(),
            )?
        };
        let suffix = collect_suffix_bytes(
            &flat_parts[repeat_index + 1..]
                .iter()
                .map(|expr| (*expr).clone())
                .collect::<Vec<_>>(),
        )?;
        return Some(DirectBoundedSuffixShape {
            prefix,
            body: body.as_ref(),
            min: *min,
            max: *max,
            suffix,
        });
    }
    None
}

/// Cheap exact state-count estimate for the direct layered bounded-suffix DFA.
///
/// This deliberately compiles only one copy of the repeat body. It is used by
/// higher-level compile-route selection to avoid materializing a multi-million
/// state full repeat merely to discover that a finite-horizon/synthetic route
/// was a bad certification choice.
pub fn direct_bounded_suffix_state_count_estimate(expr: &Expr) -> Option<usize> {
    let shape = direct_bounded_suffix_shape(expr)?;
    if shape.suffix.is_empty() {
        return None;
    }
    let base = compile_direct_bounded_repeat_base_dfa_unconditionally(shape.body)?;
    shape
        .prefix
        .len()
        .checked_add((shape.max + 1).checked_mul(base.num_states())?)?
        .checked_add(shape.suffix.len())
}

/// Largest exact layered bounded-suffix state-count estimate found anywhere
/// inside an expression tree. Intersections/exclusions can hide the expensive
/// materialized component one level below the terminal root, so compile-route
/// preflight must inspect those children before deciding to certify a
/// synthetic tokenizer by constructing the full product.
pub fn max_direct_bounded_suffix_state_count_estimate(expr: &Expr) -> Option<usize> {
    let own = direct_bounded_suffix_state_count_estimate(expr);
    let child = match expr {
        Expr::Intersect { expr, intersect } => [expr.as_ref(), intersect.as_ref()]
            .into_iter()
            .filter_map(max_direct_bounded_suffix_state_count_estimate)
            .max(),
        Expr::Exclude { expr, exclude } => [expr.as_ref(), exclude.as_ref()]
            .into_iter()
            .filter_map(max_direct_bounded_suffix_state_count_estimate)
            .max(),
        Expr::Seq(parts) | Expr::Choice(parts) => parts
            .iter()
            .filter_map(max_direct_bounded_suffix_state_count_estimate)
            .max(),
        Expr::Repeat { expr, .. } => max_direct_bounded_suffix_state_count_estimate(expr),
        Expr::Shared(expr) => max_direct_bounded_suffix_state_count_estimate(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => None,
    };
    own.into_iter().chain(child).max()
}

/// Exact finite-token-horizon transport for the direct DFA emitted for
/// `Repeat(body, min..=max) + literal_suffix`.
///
/// The direct compiler numbers repeat states by `(completed_copies,
/// body_state)`, followed by the literal suffix chain. Once the minimum has
/// been crossed, completed-copy counts differ observably only through their
/// remaining distance to the upper bound. A token of `K` bytes can cross at
/// most `ceil(K / minimum_body_width) + 1` repetition boundaries. We therefore
/// retain low counts exactly, retain that many upper-bound layers exactly, and
/// map every deeper interior layer to one synthesized interior layer.
pub(super) fn direct_bounded_suffix_state_map(
    full_expr: &Expr,
    synthesized_expr: &Expr,
    full: &DFA,
    synthesized: &DFA,
    max_token_len: usize,
    relevant_bytes: &[u8],
    vocab: Option<&Vocab>,
    repeat_horizons: Option<&VocabularyRepeatHorizonCache>,
) -> Option<DirectBoundedSuffixStateMap> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let Some(full_shape) = direct_bounded_suffix_shape(full_expr)
    else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=full_shape expr={:?}",
                expr_profile_summary(full_expr),
            );
        }
        return None;
    };
    let Some(synthesized_shape) = direct_bounded_suffix_shape(synthesized_expr)
    else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=synthesized_shape expr={:?}",
                expr_profile_summary(synthesized_expr),
            );
        }
        return None;
    };
    if full_shape.prefix != synthesized_shape.prefix
        || full_shape.min != synthesized_shape.min
        || full_shape.suffix != synthesized_shape.suffix
        || full_shape.max < synthesized_shape.max
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=shape_mismatch full_prefix={} synthesized_prefix={} full_min={} synthesized_min={} full_max={} synthesized_max={} full_suffix={} synthesized_suffix={}",
                full_shape.prefix.len(),
                synthesized_shape.prefix.len(),
                full_shape.min,
                synthesized_shape.min,
                full_shape.max,
                synthesized_shape.max,
                full_shape.suffix.len(),
                synthesized_shape.suffix.len(),
            );
        }
        return None;
    }

    let base = compile_direct_bounded_repeat_base_dfa_unconditionally(full_shape.body)?;
    let synthesized_base =
        compile_direct_bounded_repeat_base_dfa_unconditionally(synthesized_shape.body)?;
    let base_state_map = exact_kbounded_single_group_state_map(
        &base,
        &synthesized_base,
        max_token_len,
        relevant_bytes,
    );
    let Some(base_state_map) = base_state_map else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected stage=base_dfa_map full_base_states={} synthesized_base_states={}",
                base.num_states(),
                synthesized_base.num_states(),
            );
        }
        return None;
    };
    let full_base_states = base.num_states();
    let synthesized_base_states = synthesized_base.num_states();
    let prefix_states = full_shape.prefix.len();
    let suffix_states = full_shape.suffix.len();
    let expected_full_states =
        prefix_states + (full_shape.max + 1).checked_mul(full_base_states)? + suffix_states;
    let expected_synthesized_states =
        prefix_states
            + (synthesized_shape.max + 1).checked_mul(synthesized_base_states)?
            + suffix_states;
    let suffix_overlaps = base.states()[0]
        .transitions
        .get(full_shape.suffix[0])
        .is_some();
    if suffix_states == 0
        || suffix_overlaps
        || full.num_states() != expected_full_states
        || synthesized.num_states() != expected_synthesized_states
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] layered_bounded_suffix_rejected prefix_states={} base_states={} suffix_states={} full_max={} synthesized_max={} expected_full_states={} actual_full_states={} expected_synthesized_states={} actual_synthesized_states={} suffix_overlaps={}",
                prefix_states,
                full_base_states,
                suffix_states,
                full_shape.max,
                synthesized_shape.max,
                expected_full_states,
                full.num_states(),
                expected_synthesized_states,
                synthesized.num_states(),
                suffix_overlaps,
            );
        }
        return None;
    }

    let minimum_body_width = base.min_match_byte_len()?.max(1);
    let fallback_crossed_boundaries = max_token_len
        .div_ceil(minimum_body_width)
        .saturating_add(1);
    let crossed_boundaries = vocab
        .zip(repeat_horizons)
        .and_then(|(vocab, horizons)| horizons.horizon_for_dfa(&base, vocab))
        .unwrap_or(fallback_crossed_boundaries);
    // Translation-invariant interior counts must retain enough room for the
    // current token and a later completing token. Mapping them immediately
    // below the synthetic upper-bound stencil consumes that headroom after one
    // token. The first accepting count is the stable interior representative;
    // the upper-bound stencil still preserves states close to the true maximum.
    if synthesized_shape.max.saturating_sub(synthesized_shape.min) <= crossed_boundaries {
        return None;
    }
    let interior_representative = synthesized_shape.min;

    let full_suffix_start = prefix_states + (full_shape.max + 1) * full_base_states;
    let synthesized_suffix_start =
        prefix_states + (synthesized_shape.max + 1) * synthesized_base_states;
    let mut mapping = Vec::with_capacity(full.num_states());
    mapping.extend((0..prefix_states).map(|state| state as u32));
    for completed in 0..=full_shape.max {
        let distance_to_upper = full_shape.max - completed;
        let mapped_completed = if completed < full_shape.min {
            completed
        } else if distance_to_upper <= crossed_boundaries {
            synthesized_shape.max - distance_to_upper
        } else {
            interior_representative
        };
        for &mapped_body_state in &base_state_map {
            mapping.push(
                (prefix_states
                    + mapped_completed * synthesized_base_states
                    + mapped_body_state as usize) as u32,
            );
        }
    }
    for suffix_state in 0..suffix_states {
        debug_assert_eq!(mapping.len(), full_suffix_start + suffix_state);
        mapping.push((synthesized_suffix_start + suffix_state) as u32);
    }
    Some(DirectBoundedSuffixStateMap {
        primary: mapping,
        prefix_states,
        full_base_states,
        synthesized_base_states,
        full_max: full_shape.max,
        synthesized_max: synthesized_shape.max,
        min: full_shape.min,
        crossed_boundaries,
        base_state_map,
        full_suffix_start,
        synthesized_suffix_start,
    })
}

fn add_product_tuple_state(
    dfa: &mut DFA,
    trace: &mut ProductBuildTrace,
    tuple: ProductStateTuple,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> (u32, bool) {
    if let Some(state) = trace.state_lookup.get(&tuple) {
        return (state, false);
    }
    let state = dfa.add_state();
    let (finalizers, futures) = if trace.direct_single_visible_group {
        (
            product_state_single_visible_finalizer(
                &trace.components,
                &tuple,
                exclusions,
                intersections,
            ),
            BitSet::new(1),
        )
    } else {
        product_state_metadata(&trace.components, &tuple)
    };
    dfa.overwrite_state_metadata(state, finalizers, futures);
    trace.state_lookup.insert(tuple.clone(), state);
    trace.state_tuples.push(tuple);
    (state, true)
}

pub(super) fn augment_product_dfa_from_seed_tuples(
    dfa: &mut DFA,
    trace: &mut ProductBuildTrace,
    seeds: &[ProductStateTuple],
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) {
    let (class_map, class_members) = compute_product_equivalence_classes(&trace.components);
    let component_class_transitions =
        build_product_class_transitions(&trace.components, &class_map);
    let component_dead_states = trace
        .components
        .iter()
        .map(ProductComponent::dead_state)
        .collect::<Vec<_>>();
    let mut worklist = VecDeque::<(u32, ProductStateTuple)>::new();
    for tuple in seeds {
        let (state, inserted) = add_product_tuple_state(
            dfa,
            trace,
            tuple.clone(),
            exclusions,
            intersections,
        );
        if inserted {
            worklist.push_back((state, tuple.clone()));
        }
    }

    let mut class_buffers = (0..class_members.len())
        .map(|_| ProductStateTuple::new())
        .collect::<Vec<_>>();
    let mut class_active = vec![false; class_members.len()];
    let mut used_classes = Vec::<usize>::new();
    while let Some((state, tuple)) = worklist.pop_front() {
        for &(component_id, component_state) in &tuple {
            let component = component_id as usize;
            match (
                &trace.components[component],
                &component_class_transitions[component],
            ) {
                (
                    ProductComponent::Materialized(_)
                    | ProductComponent::MaterializedZeroMinRepeatSuffix { .. },
                    ProductComponentClassTransitions::Materialized(transitions),
                ) => {
                    for &(class, target) in &transitions[component_state as usize] {
                        let class = class as usize;
                        if !class_active[class] {
                            class_active[class] = true;
                            used_classes.push(class);
                        }
                        let (accepting, future) =
                            product_component_state_flags(&trace.components[component], target);
                        if component_dead_states[component] != Some(target)
                            && (accepting || future)
                        {
                            class_buffers[class].push((component_id, target));
                        }
                    }
                }
                (
                    ProductComponent::VirtualFixedSequence { .. },
                    ProductComponentClassTransitions::VirtualFixedSequence(transitions),
                ) => {
                    for &(class, target) in &transitions[component_state as usize] {
                        let class = class as usize;
                        if !class_active[class] {
                            class_active[class] = true;
                            used_classes.push(class);
                        }
                        class_buffers[class].push((component_id, target));
                    }
                }
                (
                    ProductComponent::VirtualBoundedRepeat { base_dfa, max, .. },
                    ProductComponentClassTransitions::VirtualBoundedRepeat(transitions),
                ) => {
                    let base_states = base_dfa.num_states() as u32;
                    let copies = component_state / base_states;
                    if copies >= *max {
                        continue;
                    }
                    let base_state = component_state % base_states;
                    if base_dfa.finalizers(base_state).contains(0) {
                        continue;
                    }
                    for &(class, target_base) in &transitions[base_state as usize] {
                        let class = class as usize;
                        if !class_active[class] {
                            class_active[class] = true;
                            used_classes.push(class);
                        }
                        if component_dead_states[component] == Some(target_base) {
                            continue;
                        }
                        let target = if base_dfa.finalizers(target_base).contains(0) {
                            (copies + 1) * base_states
                        } else {
                            copies * base_states + target_base
                        };
                        class_buffers[class].push((component_id, target));
                    }
                }
                _ => unreachable!("component and transition representations must align"),
            }
        }

        let mut byte_transitions = Vec::new();
        for &class in &used_classes {
            let next_tuple = class_buffers[class].clone();
            // In a pure binary intersection, a byte transition is viable only
            // when both component coordinates survive.  A materialized target
            // with neither acceptance nor future is language-dead even if it
            // is not the canonical full-byte sink, so dropping that coordinate
            // must kill the whole product transition rather than leave the
            // giant partner running alone through arbitrarily many layers.
            if trace.components.len() == 2
                && pure_binary_intersection(exclusions, intersections)
                && next_tuple.len() != 2
            {
                class_buffers[class].clear();
                class_active[class] = false;
                continue;
            }
            let (target, inserted) = add_product_tuple_state(
                dfa,
                trace,
                next_tuple.clone(),
                exclusions,
                intersections,
            );
            if inserted {
                worklist.push_back((target, next_tuple));
            }
            byte_transitions.extend(
                class_members[class]
                    .iter()
                    .copied()
                    .map(|byte| (byte, target)),
            );
            class_buffers[class].clear();
            class_active[class] = false;
        }
        used_classes.clear();
        byte_transitions.sort_unstable_by_key(|entry| entry.0);
        dfa.set_transitions_from_sorted_entries(state, byte_transitions);
    }
    if trace.direct_single_visible_group {
        dfa.recompute_possible_futures();
    }
}

pub struct CompiledTerminalExpressionPair {
    pub synthesized: Regex,
    pub full: Regex,
    pub full_to_synthesized: Vec<u32>,
    pub synthesized_expression: Expr,
}

pub(super) struct PreparedTerminalExpressionPair {
    pub(super) synthesized: Regex,
    pub(super) full: DeferredDfa,
    pub(super) full_to_synthesized: Vec<u32>,
    pub(super) synthesized_expression: Expr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StructuralPairProof {
    RawHomomorphism,
    VocabularyTokenQuotient,
}

/// Return the number of structurally aligned product components available to
/// the paired terminal compiler without constructing either DFA.
///
/// Vocabulary-only repeat reductions currently rely on the explicit
/// component maps below. A single-component pair would instead require the
/// generic finite-horizon equivalence search, which is exact but can create a
/// large and input-sensitive planning cliff. Callers may therefore use this as
/// a cheap fail-closed capability check before selecting a new optimization
/// candidate.
pub fn structural_pair_component_count(
    full_expression: &Expr,
    synthesized_expression: &Expr,
) -> Option<usize> {
    let normalized_full = lift_single_nested_intersection(full_expression);
    let normalized_synthesized = lift_single_nested_intersection(synthesized_expression);
    let full_expression = normalized_full.as_ref().unwrap_or(full_expression);
    let synthesized_expression = normalized_synthesized
        .as_ref()
        .unwrap_or(synthesized_expression);
    let full_plan = build_exclusion_compile_plan(std::slice::from_ref(full_expression));
    let synthesized_plan =
        build_exclusion_compile_plan(std::slice::from_ref(synthesized_expression));
    (full_plan.visible_groups == 1
        && synthesized_plan.visible_groups == 1
        && full_plan.compiled_exprs.len() == synthesized_plan.compiled_exprs.len()
        && full_plan.exclusions == synthesized_plan.exclusions
        && full_plan.intersections == synthesized_plan.intersections
        && !full_plan.compiled_exprs.is_empty())
    .then_some(full_plan.compiled_exprs.len())
}



pub(super) fn prepare_terminal_expression_pair_with_structural_map_inner(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    allow_component_identity_fallback: bool,
    proof: StructuralPairProof,
) -> Option<PreparedTerminalExpressionPair> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile.then(Instant::now);
    let normalized_full = lift_single_nested_intersection(full_expression);
    let normalized_synthesized = lift_single_nested_intersection(synthesized_expression);
    let full_expression = normalized_full.as_ref().unwrap_or(full_expression);
    let synthesized_expression = normalized_synthesized
        .as_ref()
        .unwrap_or(synthesized_expression);
    let full_plan = build_exclusion_compile_plan(std::slice::from_ref(full_expression));
    let synthesized_plan =
        build_exclusion_compile_plan(std::slice::from_ref(synthesized_expression));
    if full_plan.visible_groups != 1
        || synthesized_plan.visible_groups != 1
        || full_plan.compiled_exprs.len() != synthesized_plan.compiled_exprs.len()
        || full_plan.exclusions != synthesized_plan.exclusions
        || full_plan.intersections != synthesized_plan.intersections
        || full_plan.compiled_exprs.is_empty()
    {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=plan_shape full_visible={} synthesized_visible={} full_components={} synthesized_components={} exclusions_equal={} intersections_equal={}",
                full_plan.visible_groups,
                synthesized_plan.visible_groups,
                full_plan.compiled_exprs.len(),
                synthesized_plan.compiled_exprs.len(),
                full_plan.exclusions == synthesized_plan.exclusions,
                full_plan.intersections == synthesized_plan.intersections,
            );
        }
        return None;
    }
    let exclusions = synthesized_plan.exclusions.clone();
    let intersections = synthesized_plan.intersections.clone();
    let full_component_expressions = full_plan.compiled_exprs.clone();
    let synthesized_component_expressions = synthesized_plan.compiled_exprs.clone();

    // A nested intersection/exclusion can already have been resolved into one
    // visible expression by the grammar lowerer. The previous implementation
    // unnecessarily required at least two hidden product components, even
    // though one deterministic component admits the same exact finite-horizon
    // refinement directly.
    if full_component_expressions.len() == 1 {
        let product_build_started_at = profile.then(Instant::now);
        let (full_dfa, synthesized_dfa) = rayon::join(
            || compile_with_plan(full_plan),
            || compile_with_plan(synthesized_plan),
        );
        let product_build_ms = product_build_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let map_started_at = profile.then(Instant::now);
        let homomorphism = deterministic_component_homomorphism_state_map(
            &full_dfa,
            &synthesized_dfa,
        );
        let used_homomorphism = homomorphism.is_some();
        let full_to_synthesized = homomorphism.or_else(|| {
            exact_kbounded_single_group_state_map(
                &full_dfa,
                &synthesized_dfa,
                max_token_len,
                relevant_bytes,
            )
        });
        let Some(full_to_synthesized) = full_to_synthesized else {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] structural_pair_rejected stage=single_component_map full_states={} synthesized_states={} depth={}",
                    full_dfa.num_states(),
                    synthesized_dfa.num_states(),
                    max_token_len,
                );
            }
            return None;
        };
        if !used_homomorphism && proof == StructuralPairProof::RawHomomorphism {
            if profile {
                eprintln!(
                    "[glrmask/profile][tokenizer] structural_pair_rejected reason=single_component_map_not_raw_homomorphism full_states={} synthesized_states={} depth={}",
                    full_dfa.num_states(),
                    synthesized_dfa.num_states(),
                    max_token_len,
                );
            }
            return None;
        }
        let map_ms = map_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_single_component_pair full_states={} synthesized_states={} depth={} path={} build_ms={:.3} map_ms={:.3}",
                full_dfa.num_states(),
                synthesized_dfa.num_states(),
                max_token_len,
                if used_homomorphism { "deterministic_homomorphism" } else { "moore" },
                product_build_ms,
                map_ms,
            );
        }
        return Some(PreparedTerminalExpressionPair {
            synthesized: Regex {
                dfa: synthesized_dfa,
            },
            full: DeferredDfa::Ready(full_dfa),
            full_to_synthesized,
            synthesized_expression: synthesized_expression.clone(),
        });
    }

    let product_build_started_at = profile.then(Instant::now);
    let ((mut full, full_trace, full_build_ms), (mut synthesized_dfa, synthesized_trace, synthesized_build_ms)) = rayon::join(
        || {
            let started_at = profile.then(Instant::now);
            let (full, trace) = match try_compile_with_plan_deferred_dense(full_plan) {
                Ok(prepared) => prepared,
                Err(full_plan) => {
                    let (dfa, trace) = compile_with_plan_internal(full_plan, true);
                    (DeferredDfa::Ready(dfa), trace)
                }
            };
            (
                full,
                trace,
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            )
        },
        || {
            let started_at = profile.then(Instant::now);
            let (dfa, trace) = compile_with_plan_internal(synthesized_plan, true);
            (
                dfa,
                trace,
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            )
        },
    );
    let product_build_ms = product_build_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair_product_build full_ms={:.3} synthesized_ms={:.3} wall_ms={:.3}",
            full_build_ms,
            synthesized_build_ms,
            product_build_ms,
        );
    }
    let full_state_count = full.num_states();
    let Some(full_trace) = full_trace else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=missing_full_trace full_states={} synthesized_states={}",
                full_state_count,
                synthesized_dfa.num_states(),
            );
        }
        return None;
    };
    let Some(mut synthesized_trace) = synthesized_trace else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=missing_synthesized_trace full_states={} synthesized_states={}",
                full_state_count,
                synthesized_dfa.num_states(),
            );
        }
        return None;
    };
    if full_trace.components.len() != synthesized_trace.components.len() {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=component_count full_components={} synthesized_components={}",
                full_trace.components.len(),
                synthesized_trace.components.len(),
            );
        }
        return None;
    }

    let component_maps_started_at = profile.then(Instant::now);
    let component_maps = full_trace
        .components
        .par_iter()
        .zip(&synthesized_trace.components)
        .zip(
            full_component_expressions
                .par_iter()
                .zip(&synthesized_component_expressions),
        )
        .enumerate()
        .map(
            |(
                component_index,
                ((full_component, synthesized_component), (full_expr, synthesized_expr)),
            )| {
            let started_at = profile.then(Instant::now);
            let full = product_component_mapping_dfa(full_component)?;
            let synthesized = product_component_mapping_dfa(synthesized_component)?;
            let identical_mapping = (full == synthesized)
                .then(|| (0..full.num_states() as u32).collect::<Vec<_>>());
            let used_identical_mapping = identical_mapping.is_some();
            let homomorphism_mapping = if used_identical_mapping {
                None
            } else {
                deterministic_component_homomorphism_state_map(&full, &synthesized)
            };
            let used_homomorphism_mapping = homomorphism_mapping.is_some();
            if profile && !used_homomorphism_mapping && full.num_states() == synthesized.num_states() {
                let reachable_count = |dfa: &DFA| {
                    let mut seen = vec![false; dfa.num_states()];
                    let mut stack = vec![0u32];
                    while let Some(state) = stack.pop() {
                        if seen[state as usize] {
                            continue;
                        }
                        seen[state as usize] = true;
                        stack.extend(dfa.states()[state as usize].transitions.iter().map(|(_, &target)| target));
                    }
                    seen.into_iter().filter(|seen| *seen).count()
                };
                eprintln!(
                    "[glrmask/profile][tokenizer] same_size_component_homomorphism_failed expr_equal={} full_dfa_equal={} full_states={} synthesized_states={} full_reachable={} synthesized_reachable={} full_expr={:?} synthesized_expr={:?}",
                    full_expr == synthesized_expr,
                    full == synthesized,
                    full.num_states(),
                    synthesized.num_states(),
                    reachable_count(&full),
                    reachable_count(&synthesized),
                    expr_profile_summary(full_expr),
                    expr_profile_summary(synthesized_expr),
                );
            }
            let layered_mapping = if used_identical_mapping || used_homomorphism_mapping {
                None
            } else {
                direct_bounded_suffix_state_map(
                    full_expr,
                    synthesized_expr,
                    &full,
                    &synthesized,
                    max_token_len,
                    relevant_bytes,
                    Some(vocab),
                    Some(repeat_horizons),
                )
            };
            let dominance_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
            {
                None
            } else {
                zero_min_repeat_suffix_state_map(
                    full_expr,
                    synthesized_expr,
                    &full,
                    &synthesized,
                    full_component.zero_min_repeat_suffix_trace(),
                    synthesized_component.zero_min_repeat_suffix_trace(),
                    max_token_len,
                    Some(vocab),
                    Some(repeat_horizons),
                )
            };
            let unsafe_override_horizon = std::env::var(
                "GLRMASK_UNSAFE_STRUCTURAL_MAP_HORIZON",
            )
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&horizon| horizon < max_token_len);
            let override_layered_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
                || dominance_mapping.is_some()
            {
                None
            } else {
                unsafe_override_horizon.and_then(|horizon| {
                    direct_bounded_suffix_state_map(
                        full_expr,
                        synthesized_expr,
                        &full,
                        &synthesized,
                        horizon,
                        relevant_bytes,
                        None,
                        None,
                    )
                })
            };
            let override_dominance_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
                || dominance_mapping.is_some()
                || override_layered_mapping.is_some()
            {
                None
            } else {
                unsafe_override_horizon.and_then(|horizon| {
                    zero_min_repeat_suffix_state_map(
                        full_expr,
                        synthesized_expr,
                        &full,
                        &synthesized,
                        full_component.zero_min_repeat_suffix_trace(),
                        synthesized_component.zero_min_repeat_suffix_trace(),
                        horizon,
                        None,
                        None,
                    )
                })
            };
            let structural_component_vocab_map_enabled = std::env::var("GLRMASK_DISABLE_STRUCTURAL_COMPONENT_VOCABULARY_MAP")
                .ok()
                .is_none_or(|value| {
                    let value = value.trim();
                    value.is_empty() || value == "0" || value.eq_ignore_ascii_case("false")
                });
            let vocabulary_mapping = if used_identical_mapping
                || used_homomorphism_mapping
                || layered_mapping.is_some()
                || dominance_mapping.is_some()
                || override_layered_mapping.is_some()
                || override_dominance_mapping.is_some()
                || !structural_component_vocab_map_enabled
            {
                None
            } else {
                const MAX_LOCAL_FULL_STATES: usize = 4_096;
                const MAX_LOCAL_SYNTHESIZED_STATES: usize = 2_048;
                const MAX_LOCAL_PAIR_CELLS: usize = 2_000_000;
                (full.num_states() <= MAX_LOCAL_FULL_STATES
                    && synthesized.num_states() <= MAX_LOCAL_SYNTHESIZED_STATES
                    && full
                        .num_states()
                        .checked_mul(synthesized.num_states())
                        .is_some_and(|cells| cells <= MAX_LOCAL_PAIR_CELLS))
                .then(|| {
                    let full_tokenizer = Regex { dfa: full.clone() }.into_tokenizer(1, None);
                    let synthesized_tokenizer =
                        Regex { dfa: synthesized.clone() }.into_tokenizer(1, None);
                    VOCABULARY_EXACT_STATE_CERTIFIER.get().and_then(|certify| {
                        certify(
                            &full_tokenizer,
                            &synthesized_tokenizer,
                            vocab,
                            Some(&[true]),
                        )
                    })
                })
                .flatten()
            };
            let used_layered_mapping = layered_mapping.is_some();
            let used_dominance_mapping = dominance_mapping.is_some();
            let used_override_layered_mapping = override_layered_mapping.is_some();
            let used_override_dominance_mapping = override_dominance_mapping.is_some();
            let used_vocabulary_mapping = vocabulary_mapping.is_some();
            let mapping = if let Some(mapping) = identical_mapping {
                Some(ProductComponentStateMap::Fixed(mapping))
            } else if let Some(mapping) = homomorphism_mapping {
                Some(ProductComponentStateMap::Fixed(mapping))
            } else if let Some(mapping) = layered_mapping {
                Some(ProductComponentStateMap::Layered(mapping))
            } else if let Some(mapping) = dominance_mapping {
                Some(ProductComponentStateMap::Dominance(mapping))
            } else if let Some(mapping) = override_layered_mapping {
                Some(ProductComponentStateMap::Layered(mapping))
            } else if let Some(mapping) = override_dominance_mapping {
                Some(ProductComponentStateMap::Dominance(mapping))
            } else if let Some(mapping) = vocabulary_mapping {
                Some(ProductComponentStateMap::Vocabulary(mapping))
            } else {
                exact_kbounded_single_group_state_map(
                    &full,
                    &synthesized,
                    max_token_len,
                    relevant_bytes,
                )
                .map(ProductComponentStateMap::Fixed)
            };
            if profile {
                let kind = |component: &ProductComponent| match component {
                    ProductComponent::Materialized(_)
                    | ProductComponent::MaterializedZeroMinRepeatSuffix { .. } => "materialized",
                    ProductComponent::VirtualFixedSequence { .. } => "virtual_fixed_sequence",
                    ProductComponent::VirtualBoundedRepeat { .. } => "virtual_bounded_repeat",
                };
                eprintln!(
                    "[glrmask/profile][tokenizer] structural_component_map component={} full_kind={} synthesized_kind={} full_states={} synthesized_states={} depth={} path={} success={} elapsed_ms={:.3} expr={:?}",
                    component_index,
                    kind(full_component),
                    kind(synthesized_component),
                    full.num_states(),
                    synthesized.num_states(),
                    max_token_len,
                    if used_identical_mapping {
                        "identical_dfa"
                    } else if used_homomorphism_mapping {
                        "deterministic_homomorphism"
                    } else if used_layered_mapping {
                        "layered_bounded_suffix"
                    } else if used_dominance_mapping {
                        "zero_min_repeat_suffix_dominance"
                    } else if used_override_layered_mapping {
                        "UNSAFE_layered_bounded_suffix_override"
                    } else if used_override_dominance_mapping {
                        "UNSAFE_zero_min_repeat_suffix_dominance_override"
                    } else if used_vocabulary_mapping {
                        "bounded_component_vocabulary_exact"
                    } else {
                        "moore"
                    },
                    mapping.is_some(),
                    started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    expr_profile_summary(full_expr),
                );
            }
            mapping
        },
        )
        .collect::<Vec<_>>();
    let failed_components = component_maps
        .iter()
        .enumerate()
        .filter_map(|(component, mapping)| mapping.is_none().then_some(component))
        .collect::<Vec<_>>();
    if !failed_components.is_empty() && allow_component_identity_fallback {
        let mut effective_components = synthesized_component_expressions.clone();
        for &component in &failed_components {
            effective_components[component] = full_component_expressions[component].clone();
        }
        let effective_expression = rebuild_single_visible_group_expression(
            &effective_components,
            &exclusions,
            &intersections,
        )?;
        if effective_expression == *synthesized_expression {
            return None;
        }
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_component_identity_fallback components={:?} full_states={} synthesized_states={} total_components={}",
                failed_components,
                full_state_count,
                synthesized_dfa.num_states(),
                full_trace.components.len(),
            );
            if std::env::var_os("GLRMASK_PROFILE_FAILED_COMPONENT_EXPR").is_some() {
                for &component in &failed_components {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_component_identity_fallback_expr component={} full={:#?} synthesized={:#?}",
                        component,
                        full_component_expressions[component],
                        synthesized_component_expressions[component],
                    );
                }
            }
        }
        return prepare_terminal_expression_pair_with_structural_map_inner(
            full_expression,
            &effective_expression,
            vocab,
            repeat_horizons,
            max_token_len,
            relevant_bytes,
            false,
            proof,
        );
    }
    let Some(component_maps) = component_maps.into_iter().collect::<Option<Vec<_>>>() else {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected stage=component_map full_states={} synthesized_states={} components={}",
                full_state_count,
                synthesized_dfa.num_states(),
                full_trace.components.len(),
            );
        }
        return None;
    };
    let component_maps_ms = component_maps_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let synthesized_component_dead_states = synthesized_trace
        .components
        .iter()
        .map(ProductComponent::dead_state)
        .collect::<Vec<_>>();

    let tuple_map_started_at = profile.then(Instant::now);
    const DENSE_PRODUCT_LOOKUP_MAX_CELLS: usize = 16 * 1024 * 1024;
    let component_extents = synthesized_trace
        .components
        .iter()
        .map(|component| component.partition_dfa().num_states().saturating_add(1))
        .collect::<Vec<_>>();
    let dense_two_component_cells = (component_maps.len() == 2)
        .then(|| component_extents[0].checked_mul(component_extents[1]))
        .flatten()
        .filter(|&cells| cells <= DENSE_PRODUCT_LOOKUP_MAX_CELLS);

    let mut full_to_synthesized = if let Some(cells) = dense_two_component_cells {
        let right_extent = component_extents[1];
        let mut state_by_key = vec![u32::MAX; cells];
        match &synthesized_trace.state_tuples {
            ProductStateTuples::Generic(tuples) => {
                for (state, tuple) in tuples.iter().enumerate() {
                    let mut coordinates = [0usize; 2];
                    for &(component, component_state) in tuple {
                        coordinates[component as usize] = component_state as usize + 1;
                    }
                    state_by_key[coordinates[0] * right_extent + coordinates[1]] = state as u32;
                }
            }
            ProductStateTuples::DenseBinary(pairs) => {
                for (state, &(left, right)) in pairs.iter().enumerate() {
                    state_by_key[(left as usize + 1) * right_extent + right as usize + 1] =
                        state as u32;
                }
            }
        }
        let map_full_states = |full_states: [u32; 2]| {
                let mut coordinates = [0usize; 2];
                for component in 0..2 {
                    let full_state = full_states[component];
                    if full_state == u32::MAX {
                        continue;
                    }
                    let synthesized_state =
                        component_maps[component].primary()[full_state as usize];
                    if synthesized_component_dead_states[component] != Some(synthesized_state) {
                        coordinates[component] = synthesized_state as usize + 1;
                    }
                }
                let primary = state_by_key[coordinates[0] * right_extent + coordinates[1]];
                if primary != u32::MAX {
                    return primary;
                }

                for component in 0..2 {
                    let full_state = full_states[component];
                    if full_state == u32::MAX || !component_maps[component].is_flexible() {
                        continue;
                    }
                    let original_coordinate = coordinates[component];
                    let mut found = u32::MAX;
                    component_maps[component].visit_candidates(full_state, |candidate| {
                        coordinates[component] = if synthesized_component_dead_states[component]
                            == Some(candidate)
                        {
                            0
                        } else {
                            candidate as usize + 1
                        };
                        let state = state_by_key[coordinates[0] * right_extent + coordinates[1]];
                        if state != u32::MAX {
                            found = state;
                            true
                        } else {
                            false
                        }
                    });
                    coordinates[component] = original_coordinate;
                    if found != u32::MAX {
                        return found;
                    }
                }

                if full_states.iter().all(|&state| state != u32::MAX)
                    && component_maps.iter().all(ProductComponentStateMap::is_flexible)
                {
                    let mut left_candidates = SmallVec::<[u32; 8]>::new();
                    let mut right_candidates = SmallVec::<[u32; 8]>::new();
                    component_maps[0].visit_candidates(full_states[0], |candidate| {
                        left_candidates.push(candidate);
                        false
                    });
                    component_maps[1].visit_candidates(full_states[1], |candidate| {
                        right_candidates.push(candidate);
                        false
                    });
                    for &left in &left_candidates {
                        coordinates[0] = if synthesized_component_dead_states[0] == Some(left) {
                            0
                        } else {
                            left as usize + 1
                        };
                        for &right in &right_candidates {
                            coordinates[1] =
                                if synthesized_component_dead_states[1] == Some(right) {
                                    0
                                } else {
                                    right as usize + 1
                                };
                            let state = state_by_key
                                [coordinates[0] * right_extent + coordinates[1]];
                            if state != u32::MAX {
                                return state;
                            }
                        }
                    }
                }
                u32::MAX
            };
        match &full_trace.state_tuples {
            ProductStateTuples::Generic(tuples) => tuples
                .par_iter()
                .map(|tuple| {
                    let mut full_states = [u32::MAX; 2];
                    for &(component_id, full_state) in tuple {
                        full_states[component_id as usize] = full_state;
                    }
                    map_full_states(full_states)
                })
                .collect::<Vec<_>>(),
            ProductStateTuples::DenseBinary(pairs) => pairs
                .par_iter()
                .map(|&(left, right)| map_full_states([left, right]))
                .collect::<Vec<_>>(),
        }
    } else {
        let map_tuple = |tuple: &[(u32, u32)]| {
            let mut mapped = ProductStateTuple::new();
            for &(component_id, full_state) in tuple {
                let component = component_id as usize;
                let synthesized_state = component_maps[component].primary()[full_state as usize];
                if synthesized_component_dead_states[component] != Some(synthesized_state) {
                    mapped.push((component_id, synthesized_state));
                }
            }
            synthesized_trace.state_lookup.get(&mapped).unwrap_or(u32::MAX)
        };
        match &full_trace.state_tuples {
            ProductStateTuples::Generic(tuples) => {
                tuples.par_iter().map(|tuple| map_tuple(tuple)).collect()
            }
            ProductStateTuples::DenseBinary(pairs) => pairs
                .par_iter()
                .map(|&(left, right)| map_tuple(&[(0, left), (1, right)]))
                .collect(),
        }
    };
    let tuple_map_ms = tuple_map_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let missing_positions = full_to_synthesized
        .iter()
        .enumerate()
        .filter_map(|(position, &state)| (state == u32::MAX).then_some(position))
        .collect::<Vec<_>>();
    let missing_before = missing_positions.len();
    let missing_tuples = missing_positions
        .par_iter()
        .map(|&position| {
            let mut mapped = ProductStateTuple::new();
            for &(component_id, full_state) in &full_trace.state_tuples.tuple(position) {
                let component = component_id as usize;
                let synthesized_state = component_maps[component].primary()[full_state as usize];
                if synthesized_component_dead_states[component] != Some(synthesized_state) {
                    mapped.push((component_id, synthesized_state));
                }
            }
            mapped
        })
        .collect::<Vec<_>>();
    let states_before_augment = synthesized_dfa.num_states();
    let augment_started_at = profile.then(Instant::now);
    augment_product_dfa_from_seed_tuples(
        &mut synthesized_dfa,
        &mut synthesized_trace,
        &missing_tuples,
        &exclusions,
        &intersections,
    );
    let augment_ms = augment_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let lookup_started_at = profile.then(Instant::now);
    for (&position, tuple) in missing_positions.iter().zip(&missing_tuples) {
        full_to_synthesized[position] = synthesized_trace.state_lookup.get(tuple)?;
    }
    full.attach_dense_runtime_trace(full_trace)?;
    let (full_dfa, full_compressed) = full.finish_runtime();
    let raw_homomorphism = certify_supplied_dfa_state_homomorphism(
        &full_dfa,
        full_compressed.as_ref(),
        &synthesized_dfa,
        &full_to_synthesized,
    );
    if !raw_homomorphism && proof == StructuralPairProof::RawHomomorphism {
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_rejected reason=augmented_map_not_raw_homomorphism missing_tuples={}",
                missing_before,
            );
        }
        if std::env::var_os("GLRMASK_PROFILE_SYNTH_CERT").is_some() {
            let mut materialized = full_dfa.clone();
            if let Some(segment) = full_compressed.as_ref() {
                segment.materialize_into_dfa(&mut materialized);
            }
            let started_at = Instant::now();
            let minimized = materialized.minimize();
            eprintln!(
                "[glrmask/profile][synth_cert] exact_minimized_full_states={} minimize_ms={:.3}",
                minimized.num_states(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return None;
    }
    if !raw_homomorphism && profile {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair_accepted proof=vocabulary_token_quotient raw_homomorphism=false missing_tuples={}",
            missing_before,
        );
    }
    let lookup_ms = lookup_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let augmented_state_count = synthesized_dfa
        .num_states()
        .saturating_sub(states_before_augment);
    if std::env::var_os("GLRMASK_MINIMIZE_SYNTHETIC_PRODUCT").is_some()
        && augmented_state_count == 0
    {
        let minimize_started_at = Instant::now();
        let states_before = synthesized_dfa.num_states();
        let (minimized, old_to_new) = synthesized_dfa.minimize_with_state_mapping();
        for state in &mut full_to_synthesized {
            let mapped = old_to_new[*state as usize];
            if mapped == u32::MAX {
                return None;
            }
            *state = mapped;
        }
        synthesized_dfa = minimized;
        if profile {
            eprintln!(
                "[glrmask/profile][tokenizer] structural_pair_minimize states_before={} states_after={} elapsed_ms={:.3}",
                states_before,
                synthesized_dfa.num_states(),
                minimize_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    } else if profile
        && std::env::var_os("GLRMASK_MINIMIZE_SYNTHETIC_PRODUCT").is_some()
    {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair_minimize_skipped reason=augmented_residual_roots augmented_states={}",
            augmented_state_count,
        );
    }

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] structural_pair full_states={} synthesized_states_before={} synthesized_states_after={} mapped_tuples={} missing_before={} product_build_ms={:.3} component_maps_ms={:.3} tuple_map_ms={:.3} augment_ms={:.3} lookup_ms={:.3} total_ms={:.3}",
            full_state_count,
            states_before_augment,
            synthesized_dfa.num_states(),
            full_to_synthesized.len(),
            missing_before,
            product_build_ms,
            component_maps_ms,
            tuple_map_ms,
            augment_ms,
            lookup_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let full = match full_compressed {
        Some(segment) => DeferredDfa::ReadyCompressed {
            dfa: full_dfa,
            segment,
        },
        None => DeferredDfa::Ready(full_dfa),
    };

    Some(PreparedTerminalExpressionPair {
        synthesized: Regex {
            dfa: synthesized_dfa,
        },
        full,
        full_to_synthesized,
        synthesized_expression: synthesized_expression.clone(),
    })
}

pub(super) fn compile_terminal_expression_pair_with_structural_map_and_proof(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    proof: StructuralPairProof,
) -> Option<CompiledTerminalExpressionPair> {
    let prepared = prepare_terminal_expression_pair_with_structural_map_inner(
        full_expression,
        synthesized_expression,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        true,
        proof,
    )?;
    Some(CompiledTerminalExpressionPair {
        synthesized: prepared.synthesized,
        full: Regex {
            dfa: prepared.full.finish(),
        },
        full_to_synthesized: prepared.full_to_synthesized,
        synthesized_expression: prepared.synthesized_expression,
    })
}

pub fn compile_terminal_expression_pair_with_structural_map(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledTerminalExpressionPair> {
    compile_terminal_expression_pair_with_structural_map_and_proof(
        full_expression,
        synthesized_expression,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        StructuralPairProof::RawHomomorphism,
    )
}

pub fn compile_terminal_expression_pair_with_vocabulary_token_quotient(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledTerminalExpressionPair> {
    compile_terminal_expression_pair_with_structural_map_and_proof(
        full_expression,
        synthesized_expression,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        StructuralPairProof::VocabularyTokenQuotient,
    )
}
