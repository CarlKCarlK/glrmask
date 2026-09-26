use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ds::{bitset::BitSet, u8set::U8Set};
use crate::Vocab;

use super::ast::Expr;
use super::runtime_repeat_product::{
    VirtualBinaryRepeatIntersectionDescriptor, VirtualBoundedRepeatSpec,
};
use super::runtime_unit_repeat::virtual_unit_repeat_state_ids_fit;
use super::tokenizer::{
    CompressedTransitionEntries, CompressedTransitionSegment, Lexer,
    TerminalExclusionCertificate, TerminalExclusionResidualState,
    TerminalResidualCoordinates, Tokenizer,
};
use super::dfa::DFA;
use super::nfa::NFA;

type ProductStateTuple = SmallVec<[(u32, u32); 12]>;

const PRODUCT_STATE_FINGERPRINT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline]
fn extend_product_state_fingerprint(fingerprint: u64, group: u32, state: u32) -> u64 {
    let pair = ((group as u64) << 32) | state as u64;
    fingerprint
        .rotate_left(27)
        .wrapping_add(pair.wrapping_mul(0x9e37_79b1_85eb_ca87))
        .wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
}

fn product_state_fingerprint(tuple: &ProductStateTuple) -> u64 {
    tuple.iter().fold(
        PRODUCT_STATE_FINGERPRINT_SEED,
        |fingerprint, &(group, state)| extend_product_state_fingerprint(fingerprint, group, state),
    )
}

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

const DEAD_REPEAT_TRANSLATION_STATE: u32 = u32::MAX;

#[derive(Clone, PartialEq, Eq, Hash)]
struct RepeatTranslationState(Box<[u32]>);

#[derive(Clone, Copy)]
struct RepeatTranslationEdge {
    target: u32,
    completed: u16,
}

impl RepeatTranslationEdge {
    const DEAD: Self = Self {
        target: DEAD_REPEAT_TRANSLATION_STATE,
        completed: 0,
    };
}

struct RepeatTranslationAutomaton {
    transitions: Vec<Box<[RepeatTranslationEdge; 256]>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct RepeatBodyLanguageState {
    accepting: bool,
    transitions: Box<[(u8, u32)]>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct RepeatBodyLanguageKey(Box<[RepeatBodyLanguageState]>);

impl RepeatBodyLanguageKey {
    fn from_dfa(body: &DFA) -> Self {
        if body.num_states() == 0 {
            return Self(Box::new([]));
        }

        // Minimized equivalent DFAs may still carry different raw state ids
        // depending on how their expressions were factored. Canonical BFS
        // numbering from the start state makes the cache language-shaped rather
        // than construction-shaped. Byte transitions are visited in sorted
        // order, so the resulting key is deterministic.
        let mut canonical = vec![u32::MAX; body.num_states()];
        canonical[0] = 0;
        let mut next_id = 1u32;
        let mut queue = VecDeque::from([0u32]);
        let mut states = Vec::with_capacity(body.num_states());
        while let Some(state) = queue.pop_front() {
            let mut transitions = Vec::new();
            for (byte, &target) in body.states()[state as usize].transitions.iter() {
                let mapped = if canonical[target as usize] == u32::MAX {
                    let mapped = next_id;
                    next_id += 1;
                    canonical[target as usize] = mapped;
                    queue.push_back(target);
                    mapped
                } else {
                    canonical[target as usize]
                };
                transitions.push((byte, mapped));
            }
            states.push(RepeatBodyLanguageState {
                accepting: body.finalizers(state).contains(0),
                transitions: transitions.into_boxed_slice(),
            });
        }
        Self(states.into_boxed_slice())
    }
}

#[derive(Default)]
pub struct VocabularyRepeatHorizonCache {
    horizons: Mutex<FxHashMap<RepeatBodyLanguageKey, Option<usize>>>,
}

#[doc(hidden)]
pub fn build_bounded_code_mask_component_for_vocab(
    expr: &Expr,
    vocab: &Vocab,
    max_token_len: usize,
    repeat_horizons: &VocabularyRepeatHorizonCache,
) -> Option<(DFA, u32)> {
    super::runtime_residual::build_bounded_code_mask_component_for_vocab(
        expr,
        vocab,
        max_token_len,
        repeat_horizons,
    )
}

#[doc(hidden)]
pub struct PreparedBoundedCodeMaskComponent(
    super::runtime_residual::PreparedBoundedCodeMaskComponent,
);

impl PreparedBoundedCodeMaskComponent {
    pub fn finish_for_vocab(
        self,
        vocab: &Vocab,
        max_token_len: usize,
        repeat_horizons: &VocabularyRepeatHorizonCache,
    ) -> Option<(DFA, u32)> {
        self.0
            .finish_for_vocab(vocab, max_token_len, repeat_horizons)
    }

    pub fn finish_for_vocab_conservative(
        self,
        max_token_len: usize,
    ) -> Option<(DFA, u32)> {
        self.0.finish_for_vocab_conservative(max_token_len)
    }
}

#[doc(hidden)]
pub fn prepare_bounded_code_mask_component(
    expr: &Expr,
) -> Option<PreparedBoundedCodeMaskComponent> {
    super::runtime_residual::prepare_bounded_code_mask_component(expr)
        .map(PreparedBoundedCodeMaskComponent)
}

impl VocabularyRepeatHorizonCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn horizon_for_dfa(&self, body: &DFA, vocab: &Vocab) -> Option<usize> {
        let key = RepeatBodyLanguageKey::from_dfa(body);
        if let Some(cached) = self
            .horizons
            .lock()
            .ok()
            .and_then(|horizons| horizons.get(&key).copied())
        {
            return cached;
        }

        // Compute outside the cache lock and never block a Rayon worker on an
        // in-progress value. The horizon computation itself uses nested Rayon
        // work; an OnceLock per key lets sibling terminal jobs occupy every
        // worker while waiting for the initializer, starving that initializer's
        // children and deadlocking a cold build. Concurrent misses may duplicate
        // this bounded computation, then race to publish the same deterministic
        // result.
        let computed = vocabulary_repeat_boundary_horizon_for_dfa_uncached(body, vocab);
        let mut horizons = self.horizons.lock().ok()?;
        *horizons.entry(key).or_insert(computed)
    }

    pub fn horizon_for_expr(&self, body: &Expr, vocab: &Vocab) -> Option<usize> {
        // The ordinary expression-to-DFA helper requires nested language
        // operations to be lowered before NFA compilation. Repeat synthesis is
        // an optimization, so unsupported body shapes must fail closed rather
        // than reaching that internal invariant.
        if expr_contains_group_op(body) {
            return None;
        }
        self.horizon_for_dfa(&compile_expr_to_dfa(body), vocab)
    }

    /// Warm this cache once per distinct repeat-body language before callers
    /// launch sibling component work. Deduplicating first avoids the deliberate
    /// duplicate cold-miss policy in `horizon_for_dfa` without ever blocking a
    /// Rayon worker on another worker's nested vocabulary scan.
    pub fn prewarm_dfas<'a>(
        &self,
        bodies: impl IntoIterator<Item = &'a DFA>,
        vocab: &Vocab,
    ) {
        let mut unique = FxHashMap::<RepeatBodyLanguageKey, &'a DFA>::default();
        for body in bodies {
            unique
                .entry(RepeatBodyLanguageKey::from_dfa(body))
                .or_insert(body);
        }
        unique
            .into_values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .for_each(|body| {
                let _ = self.horizon_for_dfa(body, vocab);
            });
    }
}

fn normalize_repeat_translation_counts(counts: &mut [u32]) -> Option<u32> {
    let shift = counts
        .iter()
        .copied()
        .filter(|&count| count != u32::MAX)
        .min()?;
    for count in counts {
        if *count != u32::MAX {
            *count -= shift;
        }
    }
    Some(shift)
}

/// Advance the translation-invariant interior control state of `body*` by one
/// byte. Counts are minimum completed-copy counts for each live body residual.
/// A uniform count translation is factored out and returned as the edge weight.
///
/// This is the same dominance law used by the bounded-repeat/suffix compiler,
/// but with no finite upper boundary. It therefore captures exactly how many
/// repeat layers a continuation can move while remaining independent of the
/// absolute bounded-repeat count.
fn step_repeat_translation_state(
    body: &DFA,
    state: &RepeatTranslationState,
    byte: u8,
) -> Option<(RepeatTranslationState, u32)> {
    let mut next = vec![u32::MAX; body.num_states()];
    for (body_state, &completed) in state.0.iter().enumerate() {
        if completed == u32::MAX {
            continue;
        }
        let Some(target) = body.step(body_state as u32, byte) else {
            continue;
        };

        if body.finalizers(target).contains(0) {
            next[0] = next[0].min(completed.saturating_add(1));
        }
        if body.possible_future_group_ids(target).contains(0) {
            next[target as usize] = next[target as usize].min(completed);
        }
    }
    let shift = normalize_repeat_translation_counts(&mut next)?;
    Some((
        RepeatTranslationState(next.into_boxed_slice()),
        shift,
    ))
}

fn build_repeat_translation_automaton(
    body: &DFA,
    relevant_bytes: &[u8],
) -> Option<RepeatTranslationAutomaton> {
    const MAX_TRANSLATION_STATES: usize = 8_192;
    const MAX_TRANSLATION_EDGES: usize = 1_000_000;

    if body.num_states() == 0
        || body.num_groups() != 1
        || body.finalizers(0).contains(0)
    {
        return None;
    }

    let mut start = vec![u32::MAX; body.num_states()];
    start[0] = 0;
    let start = RepeatTranslationState(start.into_boxed_slice());
    let mut state_ids = FxHashMap::<RepeatTranslationState, u32>::default();
    state_ids.insert(start.clone(), 0);
    let mut states = vec![start.clone()];
    let mut worklist = VecDeque::from([start]);
    let mut transitions = Vec::<Box<[RepeatTranslationEdge; 256]>>::new();
    let mut edge_count = 0usize;

    while let Some(state) = worklist.pop_front() {
        let mut row = Box::new([RepeatTranslationEdge::DEAD; 256]);
        for &byte in relevant_bytes {
            let Some((target_state, completed)) =
                step_repeat_translation_state(body, &state, byte)
            else {
                continue;
            };
            let target = if let Some(&target) = state_ids.get(&target_state) {
                target
            } else {
                if states.len() >= MAX_TRANSLATION_STATES {
                    return None;
                }
                let target = states.len() as u32;
                state_ids.insert(target_state.clone(), target);
                states.push(target_state.clone());
                worklist.push_back(target_state);
                target
            };
            let completed = u16::try_from(completed).ok()?;
            row[byte as usize] = RepeatTranslationEdge { target, completed };
            edge_count += 1;
            if edge_count > MAX_TRANSLATION_EDGES {
                return None;
            }
        }
        transitions.push(row);
    }
    debug_assert_eq!(transitions.len(), states.len());
    Some(RepeatTranslationAutomaton { transitions })
}

/// Exact maximum uniform repeat-count displacement observable within the
/// suffix of one vocabulary token, starting from any translation-invariant
/// repeat-body residual.
///
/// Injecting every control state before each byte makes the analyzed language
/// suffix-closed. This is required because a token may consume a literal or
/// another product coordinate before entering the bounded repeat. The result
/// is vocabulary-relative but grammar-independent: it applies to every bounded
/// repeat whose body compiles to the supplied deterministic automaton.
fn max_repeat_translation_over_vocab_suffixes(
    automaton: &RepeatTranslationAutomaton,
    vocab: &Vocab,
) -> usize {
    let state_count = automaton.transitions.len();
    if state_count == 0 {
        return 0;
    }

    vocab
        .entries_map()
        .values()
        .collect::<Vec<_>>()
        .par_iter()
        .map_init(
            || {
                (
                    vec![i32::MIN; state_count],
                    vec![i32::MIN; state_count],
                )
            },
            |(current, next), token| {
                current.fill(i32::MIN);
                let mut best = 0i32;
                for &byte in token.iter() {
                    next.fill(i32::MIN);
                    for (state, row) in automaton.transitions.iter().enumerate() {
                        // Start a new suffix at this byte from any reachable
                        // translation control state, or continue an earlier
                        // suffix when that has accumulated more completions.
                        let score = current[state].max(0);
                        let edge = row[byte as usize];
                        if edge.target == DEAD_REPEAT_TRANSLATION_STATE {
                            continue;
                        }
                        let candidate = score.saturating_add(i32::from(edge.completed));
                        best = best.max(candidate);
                        let slot = &mut next[edge.target as usize];
                        *slot = (*slot).max(candidate);
                    }
                    std::mem::swap(current, next);
                }
                best.max(0) as usize
            },
        )
        .max()
        .unwrap_or(0)
}

/// Compute a vocabulary-exact repeat-boundary horizon for one repeat body.
/// Returns `None` only when the body's translation control automaton exceeds a
/// conservative proof budget, in which case callers must retain their existing
/// byte-length upper bound or reject the synthesis candidate.
fn vocabulary_repeat_boundary_horizon_for_dfa_uncached(
    body: &DFA,
    vocab: &Vocab,
) -> Option<usize> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let relevant_bytes = vocab.relevant_bytes();
    let automaton = build_repeat_translation_automaton(body, &relevant_bytes)?;
    let horizon = max_repeat_translation_over_vocab_suffixes(&automaton, vocab);
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] vocabulary_repeat_horizon body_states={} translation_states={} relevant_bytes={} horizon={} elapsed_ms={:.3}",
            body.num_states(),
            automaton.transitions.len(),
            relevant_bytes.len(),
            horizon,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(horizon)
}

pub fn vocabulary_repeat_boundary_horizon(
    body_expr: &Expr,
    vocab: &Vocab,
) -> Option<usize> {
    VocabularyRepeatHorizonCache::new().horizon_for_expr(body_expr, vocab)
}

fn unwrap_shared(expr: &Expr) -> &Expr {
    match expr {
        Expr::Shared(inner) => unwrap_shared(inner),
        other => other,
    }
}

fn seq_from_parts(mut parts: Vec<Expr>) -> Expr {
    match parts.len() {
        0 => Expr::Epsilon,
        1 => parts.pop().unwrap(),
        _ => Expr::Seq(parts),
    }
}

fn choice_first_part(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Shared(inner) => choice_first_part(inner),
        Expr::Seq(parts) => parts.first(),
        Expr::Epsilon => None,
        other => Some(other),
    }
}

fn choice_without_first_part(expr: &Expr) -> Expr {
    match expr {
        Expr::Shared(inner) => choice_without_first_part(inner),
        Expr::Seq(parts) => seq_from_parts(parts[1..].to_vec()),
        Expr::Epsilon => Expr::Epsilon,
        _ => Expr::Epsilon,
    }
}

fn choice_last_part(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Shared(inner) => choice_last_part(inner),
        Expr::Seq(parts) => parts.last(),
        Expr::Epsilon => None,
        other => Some(other),
    }
}

fn choice_without_last_part(expr: &Expr) -> Expr {
    match expr {
        Expr::Shared(inner) => choice_without_last_part(inner),
        Expr::Seq(parts) => seq_from_parts(parts[..parts.len() - 1].to_vec()),
        Expr::Epsilon => Expr::Epsilon,
        _ => Expr::Epsilon,
    }
}

fn factor_choice_common_prefix(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let prefix = choice_first_part(options.first()?)?.clone();
    if !options
        .iter()
        .all(|option| choice_first_part(option) == Some(&prefix))
    {
        return None;
    }

    let remainders = options
        .iter()
        .map(choice_without_first_part)
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        prefix,
        factor_choice_of_factored(remainders),
    ]))
}

fn factor_choice_common_suffix(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let suffix = choice_last_part(options.first()?)?.clone();
    if !options
        .iter()
        .all(|option| choice_last_part(option) == Some(&suffix))
    {
        return None;
    }

    let prefixes = options
        .iter()
        .map(choice_without_last_part)
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        factor_choice_of_factored(prefixes),
        suffix,
    ]))
}

/// Factor one repeated leading atom even when it is shared by only a subset
/// of the choice arms. This is the exact identity
///
/// ```text
/// A B | A C | D  ==  A (B | C) | D
/// ```
///
/// and is particularly important for mixed anchored/unanchored JSON-schema
/// string patterns: the unanchored arms often share a `JSON_STRING_CHAR*`
/// prefix while an anchored arm does not. Requiring every arm to share the
/// prefix leaves subset construction to rediscover that factoring through a
/// potentially enormous intermediate DFA.
fn factor_choice_repeated_prefix_subset(options: &[Expr]) -> Option<Expr> {
    // Group once instead of rescanning the whole choice for every candidate.
    // Large finite literal languages routinely have thousands of distinct
    // arms; the old nested scan made a no-op subset-factor probe quadratic.
    // Iterating `options` again below preserves the historical deterministic
    // choice of the first repeated leading atom.
    let mut groups = FxHashMap::<&Expr, Vec<usize>>::default();
    for (index, option) in options.iter().enumerate() {
        if let Some(prefix) = choice_first_part(option) {
            groups.entry(prefix).or_default().push(index);
        }
    }

    for (first_index, option) in options.iter().enumerate() {
        let Some(prefix) = choice_first_part(option) else {
            continue;
        };
        let Some(matching) = groups.get(prefix) else {
            continue;
        };
        if matching.len() < 2 || matching.len() == options.len() {
            continue;
        }
        let prefix = prefix.clone();

        let remainders = matching
            .iter()
            .map(|&index| choice_without_first_part(&options[index]))
            .collect::<Vec<_>>();
        let factored_group = seq_from_parts(vec![
            prefix,
            factor_choice_of_factored(remainders),
        ]);
        let mut is_matching = vec![false; options.len()];
        for &index in matching {
            is_matching[index] = true;
        }
        let mut rewritten = Vec::with_capacity(options.len() - matching.len() + 1);
        for (index, option) in options.iter().enumerate() {
            if index == first_index {
                rewritten.push(factored_group.clone());
            } else if !is_matching[index] {
                rewritten.push(option.clone());
            }
        }
        return Some(factor_choice_of_factored(rewritten));
    }
    None
}

/// Suffix counterpart of `factor_choice_repeated_prefix_subset`:
///
/// ```text
/// B A | C A | D  ==  (B | C) A | D
/// ```
fn factor_choice_repeated_suffix_subset(options: &[Expr]) -> Option<Expr> {
    let mut groups = FxHashMap::<&Expr, Vec<usize>>::default();
    for (index, option) in options.iter().enumerate() {
        if let Some(suffix) = choice_last_part(option) {
            groups.entry(suffix).or_default().push(index);
        }
    }

    for (first_index, option) in options.iter().enumerate() {
        let Some(suffix) = choice_last_part(option) else {
            continue;
        };
        let Some(matching) = groups.get(suffix) else {
            continue;
        };
        if matching.len() < 2 || matching.len() == options.len() {
            continue;
        }
        let suffix = suffix.clone();

        let prefixes = matching
            .iter()
            .map(|&index| choice_without_last_part(&options[index]))
            .collect::<Vec<_>>();
        let factored_group = seq_from_parts(vec![
            factor_choice_of_factored(prefixes),
            suffix,
        ]);
        let mut is_matching = vec![false; options.len()];
        for &index in matching {
            is_matching[index] = true;
        }
        let mut rewritten = Vec::with_capacity(options.len() - matching.len() + 1);
        for (index, option) in options.iter().enumerate() {
            if index == first_index {
                rewritten.push(factored_group.clone());
            } else if !is_matching[index] {
                rewritten.push(option.clone());
            }
        }
        return Some(factor_choice_of_factored(rewritten));
    }
    None
}

/// Factor sibling exclusions that subtract the same language:
///
/// ```text
/// (A \\ B) | (C \\ B) | D  ==  ((A | C) \\ B) | D
/// ```
///
/// This is ordinary set algebra on languages. Keeping the subtraction outside
/// the union matters for schema-generated key languages: otherwise every arm
/// is materialized as an independent DFA before the union is determinized.
fn factor_choice_repeated_exclusion_rhs(options: &[Expr]) -> Option<Expr> {
    let mut groups = FxHashMap::<&Expr, Vec<usize>>::default();
    for (index, option) in options.iter().enumerate() {
        if let Expr::Exclude { exclude, .. } = unwrap_shared(option) {
            groups.entry(unwrap_shared(exclude)).or_default().push(index);
        }
    }

    let mut replacement_by_first = FxHashMap::<usize, Expr>::default();
    let mut remove = vec![false; options.len()];
    for matching in groups.values() {
        if matching.len() < 2 {
            continue;
        }
        let first_index = matching[0];
        let Expr::Exclude { exclude, .. } = unwrap_shared(&options[first_index]) else {
            unreachable!("repeated exclusion group contains a non-exclusion arm");
        };
        let lefts = matching
            .iter()
            .map(|&index| match unwrap_shared(&options[index]) {
                Expr::Exclude { expr, .. } => (**expr).clone(),
                _ => unreachable!("repeated exclusion group contains a non-exclusion arm"),
            })
            .collect::<Vec<_>>();
        let left = if lefts.len() == 1 {
            lefts.into_iter().next().unwrap()
        } else {
            // `options` were recursively factored before this helper runs, so
            // every left subtree is already normalized. Do not walk all of
            // them again merely because the set identity introduced a union.
            Expr::Choice(lefts)
        };
        replacement_by_first.insert(
            first_index,
            Expr::Exclude {
                expr: Box::new(left),
                exclude: Box::new((**exclude).clone()),
            },
        );
        for &index in matching.iter().skip(1) {
            remove[index] = true;
        }
    }

    if replacement_by_first.is_empty() {
        return None;
    }
    let mut rewritten = Vec::with_capacity(options.len());
    for (index, option) in options.iter().enumerate() {
        if let Some(replacement) = replacement_by_first.remove(&index) {
            rewritten.push(replacement);
        } else if !remove[index] {
            rewritten.push(option.clone());
        }
    }
    if rewritten.len() == 1 {
        rewritten.pop()
    } else {
        Some(Expr::Choice(rewritten))
    }
}

fn factor_choice_literals(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let first_byte = match unwrap_shared(options.first()?) {
        Expr::U8Seq(bytes) if !bytes.is_empty() => bytes[0],
        _ => return None,
    };
    for option in options {
        match unwrap_shared(option) {
            Expr::U8Seq(bytes) if !bytes.is_empty() && bytes[0] == first_byte => {}
            _ => return None,
        }
    }

    let remainders = options
        .iter()
        .map(|option| match unwrap_shared(option) {
            Expr::U8Seq(bytes) => {
            if bytes.len() == 1 {
                Expr::Epsilon
            } else {
                Expr::U8Seq(bytes[1..].to_vec())
            }
            }
            _ => unreachable!("literal choice was validated above"),
        })
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        Expr::U8Seq(vec![first_byte]),
        factor_choice_of_factored(remainders),
    ]))
}

/// Return the exact byte language of an expression which consumes precisely
/// one byte.  Keeping this proof deliberately syntactic makes the aligned
/// repeat-intersection rewrite fail closed for variable-width bodies.
fn exact_unit_byte_language(expr: &Expr) -> Option<U8Set> {
    match unwrap_shared(expr) {
        Expr::U8Seq(bytes) if bytes.len() == 1 => Some(U8Set::single(bytes[0])),
        Expr::U8Class(bytes) => Some(*bytes),
        Expr::Choice(options) => {
            let mut bytes = U8Set::empty();
            for option in options {
                bytes |= exact_unit_byte_language(option)?;
            }
            Some(bytes)
        }
        Expr::Intersect { expr, intersect } => Some(
            exact_unit_byte_language(expr)?.intersection(&exact_unit_byte_language(intersect)?),
        ),
        _ => None,
    }
}

#[doc(hidden)]
pub fn virtual_zero_min_unit_repeat_fits_state_ids(
    max: usize,
    physical_state_count: u32,
) -> bool {
    virtual_unit_repeat_state_ids_fit(max, physical_state_count)
}

/// Recognize the exact arithmetic runtime lane after factoring. The body proof
/// is syntactic and therefore fails closed for any variable-width repetition.
#[doc(hidden)]
pub fn virtual_unit_repeat_descriptor(expr: &Expr) -> Option<(U8Set, usize, usize)> {
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared(expr)
    else {
        return None;
    };
    // The standalone arithmetic runtime has one physical reset state and
    // reserves the high raw-state bit. Keep descriptor eligibility identical
    // to that state-ID contract. Larger exact repeats can still use the
    // binary repeat-product lane, where the repetition count is not encoded as
    // a contiguous virtual-state interval.
    if !virtual_zero_min_unit_repeat_fits_state_ids(*max, 1) {
        return None;
    }
    let bytes = exact_unit_byte_language(body)?;
    (!bytes.is_empty() && *max > 0 && min <= max).then_some((bytes, *min, *max))
}

#[doc(hidden)]
pub fn virtual_zero_min_unit_repeat_descriptor(expr: &Expr) -> Option<(U8Set, usize)> {
    let (body, min, max) = virtual_unit_repeat_descriptor(expr)?;
    (min == 0).then_some((body, max))
}

/// Build the O(1)-storage physical tokenizer for the supported standalone
/// repeat. Positive-length residuals are supplied by `Tokenizer`'s arithmetic
/// virtual-state sidecar.
pub fn build_virtual_unit_repeat_tokenizer(expressions: &[Expr]) -> Option<Tokenizer> {
    let [expression] = expressions else {
        return None;
    };
    let (body, min, max) = virtual_unit_repeat_descriptor(expression)?;
    let mut tokenizer = Tokenizer::from_parts(
        DFA::new(1),
        1,
        Some(Arc::from(expressions.to_vec().into_boxed_slice())),
    );
    tokenizer.install_virtual_unit_repeat(body, min, max)?;
    Some(tokenizer)
}


pub fn build_virtual_zero_min_unit_repeat_tokenizer(expressions: &[Expr]) -> Option<Tokenizer> {
    let [expression] = expressions else {
        return None;
    };
    virtual_zero_min_unit_repeat_descriptor(expression)?;
    build_virtual_unit_repeat_tokenizer(expressions)
}

const VIRTUAL_BINARY_REPEAT_MIN_BOUND: usize = 4_096;

fn virtual_bounded_repeat_spec(expr: &Expr) -> Option<VirtualBoundedRepeatSpec> {
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared(expr)
    else {
        return None;
    };
    if *max < VIRTUAL_BINARY_REPEAT_MIN_BOUND || min > max {
        return None;
    }
    // The body is compiled below in order to prove the deterministic,
    // non-nullable, prefix-free residual model. Do not perform that semantic
    // probe if the body itself contains another giant bounded repeat: doing so
    // would eagerly materialize exactly the state space this virtual lane is
    // intended to avoid.
    if expression_contains_large_bounded_repeat(body) {
        return None;
    }
    let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(body, None)?;
    Some(VirtualBoundedRepeatSpec {
        base_dfa,
        min: u32::try_from(*min).ok()?,
        max: u32::try_from(*max).ok()?,
    })
}

fn virtual_zero_min_bounded_repeat_spec(expr: &Expr) -> Option<VirtualBoundedRepeatSpec> {
    let spec = virtual_bounded_repeat_spec(expr)?;
    (spec.min == 0).then_some(spec)
}

/// Exactly factor an intersection of two bounded repetitions over the same
/// certified prefix-free body. Prefix-free body languages are codes, so every
/// word in `L*` has a unique factor count; a common word therefore has one
/// count which must lie in both intervals.
///
/// Keep this rewrite focused on the giant nonzero-minimum case which otherwise
/// needs the special synchronized virtual runtime. Small ordinary products and
/// established zero-minimum paths remain unchanged.
fn factor_same_body_nonzero_repeat_intersection(left: &Expr, right: &Expr) -> Option<Expr> {
    let Expr::Repeat {
        expr: left_body,
        min: left_min,
        max: Some(left_max),
    } = unwrap_shared(left)
    else {
        return None;
    };
    let Expr::Repeat {
        expr: right_body,
        min: right_min,
        max: Some(right_max),
    } = unwrap_shared(right)
    else {
        return None;
    };
    if (*left_min == 0 && *right_min == 0)
        || (*left_max < VIRTUAL_BINARY_REPEAT_MIN_BOUND
            && *right_max < VIRTUAL_BINARY_REPEAT_MIN_BOUND)
        || unwrap_shared(left_body) != unwrap_shared(right_body)
        || expression_contains_large_bounded_repeat(left_body)
    {
        return None;
    }

    // Establish unique factor counts before reasoning about interval overlap.
    // Without this proof, a non-code body such as {"a", "aa"} can represent
    // the same byte string with different repetition counts on the two sides.
    cached_direct_bounded_repeat_base_dfa_unconditionally(left_body, None)?;

    let min = (*left_min).max(*right_min);
    let max = (*left_max).min(*right_max);
    if min > max {
        return Some(Expr::U8Class(U8Set::empty()));
    }
    Some(Expr::Repeat {
        expr: Box::new((**left_body).clone()),
        min,
        max: Some(max),
    })
}

struct DelimitedLiteralRepeatShape<'a> {
    prefix: Vec<u8>,
    body: &'a Expr,
    min: usize,
    max: usize,
    suffix: Vec<u8>,
}

const DELIMITED_REPEAT_SUFFIX_MERGED_STATE_BUDGET: usize = 100_000;
const DELIMITED_REPEAT_SUFFIX_MERGED_TRANSITION_BUDGET: usize = 1_000_000;

fn delimited_literal_repeat_shape(expr: &Expr) -> Option<DelimitedLiteralRepeatShape<'_>> {
    fn literal_bytes(parts: &[Expr]) -> Option<Vec<u8>> {
        let mut bytes = Vec::new();
        for part in parts {
            match unwrap_shared(part) {
                Expr::U8Seq(chunk) => bytes.extend_from_slice(chunk),
                Expr::Epsilon => {}
                _ => return None,
            }
        }
        Some(bytes)
    }

    let Expr::Seq(parts) = unwrap_shared(expr) else {
        return None;
    };
    let mut repeat_index = None;
    for (index, part) in parts.iter().enumerate() {
        if matches!(unwrap_shared(part), Expr::Repeat { max: Some(_), .. }) {
            if repeat_index.replace(index).is_some() {
                return None;
            }
        }
    }
    let repeat_index = repeat_index?;
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared(&parts[repeat_index])
    else {
        return None;
    };
    let prefix = literal_bytes(&parts[..repeat_index])?;
    let suffix = literal_bytes(&parts[repeat_index + 1..])?;
    if suffix.is_empty() {
        return None;
    }
    Some(DelimitedLiteralRepeatShape {
        prefix,
        body,
        min: *min,
        max: *max,
        suffix,
    })
}

/// Exactly factor two bounded repeats with a shared literal
/// prefix/body and an unambiguous literal suffix boundary. A certified
/// prefix-free body is an instantaneous code. If the first suffix byte cannot
/// begin any productive body word, both parses must leave the repeated body at
/// the same code-word boundary and therefore at the same repetition count.
/// The remaining exact literal suffixes can then be compared directly.
///
/// Keep this focused on giant repeat+suffix shapes. The rewrite deliberately
/// refuses suffixes whose first byte can begin another body word: e.g.
/// `a*ab ∩ a*b` overlaps through a one-copy count shift and must remain a real
/// intersection.
fn factor_same_body_delimited_literal_repeat_suffix_intersection(
    left: &Expr,
    right: &Expr,
) -> Option<Expr> {
    let left = delimited_literal_repeat_shape(left)?;
    let right = delimited_literal_repeat_shape(right)?;
    if (left.max < VIRTUAL_BINARY_REPEAT_MIN_BOUND
        && right.max < VIRTUAL_BINARY_REPEAT_MIN_BOUND)
        || left.prefix != right.prefix
        || unwrap_shared(left.body) != unwrap_shared(right.body)
        || expression_contains_large_bounded_repeat(left.body)
    {
        return None;
    }

    let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(left.body, None)?;
    let productive = productive_dfa_states(&base_dfa);
    let suffix_can_start_body = |suffix: &[u8]| {
        base_dfa
            .step(0, suffix[0])
            .is_some_and(|target| productive[target as usize])
    };
    if suffix_can_start_body(&left.suffix) || suffix_can_start_body(&right.suffix) {
        return None;
    }

    let min = left.min.max(right.min);
    let max = left.max.min(right.max);
    if min > max {
        return Some(Expr::U8Class(U8Set::empty()));
    }
    if left.suffix != right.suffix {
        return Some(Expr::U8Class(U8Set::empty()));
    }

    // The theorem is exact for arbitrary minima, but a giant positive-minimum
    // repeat followed by a suffix does not yet have its own symbolic runtime
    // lane. Keep that still-giant result as the original intersection instead
    // of turning a safely rejected shape into a later eager compile. If the
    // merged upper bound is ordinary, explicitly bound the layered DFA that
    // the rewrite can cause rather than treating the giant threshold itself as
    // a memory budget.
    if max >= VIRTUAL_BINARY_REPEAT_MIN_BOUND {
        if min != 0 {
            return None;
        }
    } else if max != 0 {
        let layers = max.checked_add(1)?;
        let states = layers
            .checked_mul(base_dfa.num_states())?
            .checked_add(left.prefix.len())?
            .checked_add(left.suffix.len())?;
        let transitions = max
            .checked_mul(dfa_transition_count(&base_dfa))?
            .checked_add(layers)?
            .checked_add(left.prefix.len())?
            .checked_add(left.suffix.len())?;
        if states > DELIMITED_REPEAT_SUFFIX_MERGED_STATE_BUDGET
            || transitions > DELIMITED_REPEAT_SUFFIX_MERGED_TRANSITION_BUDGET
        {
            return None;
        }
    }

    let mut parts = Vec::with_capacity(3);
    if !left.prefix.is_empty() {
        parts.push(Expr::U8Seq(left.prefix));
    }
    if max != 0 {
        parts.push(Expr::Repeat {
            expr: Box::new((*left.body).clone()),
            min,
            max: Some(max),
        });
    }
    parts.push(Expr::U8Seq(left.suffix));
    Some(seq_from_parts(parts))
}

/// Recognize an exact pure intersection whose two large bounded-repeat
/// coordinates can remain symbolic at runtime. Each body must be deterministic,
/// non-nullable and prefix-free; that makes `(completed copies, body state)` a
/// complete residual coordinate for each side.
#[doc(hidden)]
pub fn virtual_binary_bounded_repeat_intersection_descriptor(
    expr: &Expr,
) -> Option<VirtualBinaryRepeatIntersectionDescriptor> {
    let Expr::Intersect { expr, intersect } = unwrap_shared(expr) else {
        return None;
    };
    let left = virtual_zero_min_bounded_repeat_spec(expr)?;
    let right = virtual_zero_min_bounded_repeat_spec(intersect)?;
    let byte_support = expr_u8set(expr).intersection(&expr_u8set(intersect));
    (!byte_support.is_empty()).then_some(VirtualBinaryRepeatIntersectionDescriptor {
        left,
        right,
        byte_support,
    })
}

/// Recognize one large bounded repetition whose body admits the
/// exact symbolic repeat coordinate used by the lazy product runtime.
///
/// The runtime already implements the exact language intersection of two
/// bounded-repeat coordinates.  Supplying the same coordinate on both sides
/// therefore represents the original language exactly by `L = L ∩ L`, while
/// avoiding `(max + 1) * body_states` materialization.
#[doc(hidden)]
pub fn virtual_large_bounded_repeat_descriptor(
    expr: &Expr,
) -> Option<VirtualBinaryRepeatIntersectionDescriptor> {
    let spec = virtual_bounded_repeat_spec(expr)?;
    let byte_support = expr_u8set(expr);
    (!byte_support.is_empty()).then_some(VirtualBinaryRepeatIntersectionDescriptor {
        left: spec.clone(),
        right: spec,
        byte_support,
    })
}

/// Return the declared upper bound of a top-level bounded repetition large
/// enough to require symbolic treatment. This deliberately ignores whether
/// the body is currently supported by the exact symbolic backend: callers use
/// it to fail closed instead of falling through to the eager repeat compiler.
#[doc(hidden)]
pub fn large_top_level_bounded_repeat_bound(expr: &Expr) -> Option<usize> {
    let Expr::Repeat { max: Some(max), .. } = unwrap_shared(expr) else {
        return None;
    };
    (*max >= VIRTUAL_BINARY_REPEAT_MIN_BOUND).then_some(*max)
}

/// Exactly normalize two aligned unit-byte repeats. Iteration boundaries
/// coincide after every consumed byte, so the byte language and repetition
/// count intervals can be intersected independently.
fn factor_aligned_unit_repeat_intersection(left: &Expr, right: &Expr) -> Option<Expr> {
    let Expr::Repeat {
        expr: left_body,
        min: left_min,
        max: Some(left_max),
    } = unwrap_shared(left)
    else {
        return None;
    };
    let Expr::Repeat {
        expr: right_body,
        min: right_min,
        max: Some(right_max),
    } = unwrap_shared(right)
    else {
        return None;
    };

    let body = exact_unit_byte_language(left_body)?
        .intersection(&exact_unit_byte_language(right_body)?);
    let min = (*left_min).max(*right_min);
    let max = (*left_max).min(*right_max);
    if min > max {
        return Some(Expr::U8Class(U8Set::empty()));
    }
    if body.is_empty() {
        return Some(if min == 0 {
            Expr::Epsilon
        } else {
            Expr::U8Class(U8Set::empty())
        });
    }
    Some(Expr::Repeat {
        expr: Box::new(Expr::U8Class(body)),
        min,
        max: Some(max),
    })
}

/// Factor a newly introduced choice whose children were already recursively
/// factored. Prefix/suffix rewrites change the union/concatenation spine, not
/// their child languages. Restarting the entire recursive normalizer here
/// repeats work and expands cached Shared children under a fresh empty cache.
fn factor_choice_of_factored(mut options: Vec<Expr>) -> Expr {
    const LARGE_PURE_LITERAL_CHOICE_NO_FACTOR: usize = 64;
    if options.len() >= LARGE_PURE_LITERAL_CHOICE_NO_FACTOR
        && options.iter().all(|option| matches!(unwrap_shared(option), Expr::U8Seq(_)))
    {
        return Expr::Choice(options);
    }
    if options.len() == 1 {
        return options.pop().unwrap();
    }
    if let Some(factored) = factor_choice_literals(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_common_prefix(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_common_suffix(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_repeated_exclusion_rhs(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_repeated_prefix_subset(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_repeated_suffix_subset(&options) {
        return factored;
    }
    Expr::Choice(options)
}

fn factor_regex_expr_impl(
    expr: Expr,
    shared_cache: Option<&FxHashMap<usize, Arc<Expr>>>,
) -> Expr {
    match expr {
        Expr::Seq(parts) => {
            let mut out = Vec::new();
            for part in parts {
                match factor_regex_expr_impl(part, shared_cache) {
                    Expr::Seq(inner) => out.extend(inner),
                    Expr::Epsilon => {}
                    other => out.push(other),
                }
            }
            seq_from_parts(out)
        }
        Expr::Choice(options) => {
            // A large finite choice of exact byte strings is already an ideal
            // input to ordinary subset construction: the resulting DFA is the
            // shared-prefix trie of those strings. Recursively peeling common
            // bytes and probing subset prefix/suffix factors only rebuilds and
            // hashes the entire literal set repeatedly, while providing no
            // protection against subset-state explosion (there is none for a
            // finite literal trie). Keep small choices on the historical path,
            // where syntactic factoring is cheap and can reduce setup overhead.
            const LARGE_PURE_LITERAL_CHOICE_NO_FACTOR: usize = 64;
            if options.len() >= LARGE_PURE_LITERAL_CHOICE_NO_FACTOR
                && options
                    .iter()
                    .all(|option| matches!(unwrap_shared(option), Expr::U8Seq(_)))
            {
                return Expr::Choice(options);
            }
            let factored_options = options
                .into_iter()
                .map(|expr| factor_regex_expr_impl(expr, shared_cache))
                .collect::<Vec<_>>();
            factor_choice_of_factored(factored_options)
        }
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(factor_regex_expr_impl(*expr, shared_cache)),
            min,
            max,
        },
        Expr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(factor_regex_expr_impl(*expr, shared_cache)),
            exclude: Box::new(factor_regex_expr_impl(*exclude, shared_cache)),
        },
        Expr::Intersect { expr, intersect } => {
            let expr = factor_regex_expr_impl(*expr, shared_cache);
            let intersect = factor_regex_expr_impl(*intersect, shared_cache);
            factor_same_body_delimited_literal_repeat_suffix_intersection(&expr, &intersect)
                .or_else(|| factor_aligned_unit_repeat_intersection(&expr, &intersect))
                .or_else(|| factor_same_body_nonzero_repeat_intersection(&expr, &intersect))
                .unwrap_or_else(|| Expr::Intersect {
                    expr: Box::new(expr),
                    intersect: Box::new(intersect),
                })
        }
        Expr::Shared(inner) => {
            if let Some(cached) = shared_cache
                .and_then(|cache| cache.get(&(Arc::as_ptr(&inner) as usize)))
            {
                // Preserve the reference boundary after factoring. Expanding a
                // cached result back into each caller recreates the expression
                // tree and makes downstream Boolean materialization repeat the
                // same structural work that this cache was meant to avoid.
                return Expr::Shared(Arc::clone(cached));
            }
            factor_regex_expr_impl((*inner).clone(), shared_cache)
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => expr,
    }
}

pub fn factor_regex_expr(expr: Expr) -> Expr {
    factor_regex_expr_impl(expr, None)
}

pub fn factor_regex_expr_with_shared_cache(
    expr: Expr,
    shared_cache: &FxHashMap<usize, Arc<Expr>>,
) -> Expr {
    factor_regex_expr_impl(expr, Some(shared_cache))
}

fn common_prefix_factor(exprs: &[Expr]) -> Option<(Expr, Vec<Expr>)> {
    fn candidate_prefix(expr: &Expr) -> Option<&Expr> {
        match expr {
            Expr::Seq(parts) if !parts.is_empty() => Some(&parts[0]),
            Expr::Shared(inner) => candidate_prefix(inner),
            _ => None,
        }
    }

    let prefix = candidate_prefix(exprs.first()?)?.clone();
    let mut remainders = Vec::with_capacity(exprs.len());
    for expr in exprs {
        remainders.push(expr.strip_prefix(&prefix)?);
    }
    Some((prefix, remainders))
}

fn expr_contains_group_op(expr: &Expr) -> bool {
    match expr {
        Expr::Exclude { .. } | Expr::Intersect { .. } => true,
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_contains_group_op),
        Expr::Repeat { expr, .. } => expr_contains_group_op(expr),
        Expr::Shared(inner) => expr_contains_group_op(inner),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
    }
}

fn group_op_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Exclude { expr, exclude } => {
            1 + group_op_node_count(expr) + group_op_node_count(exclude)
        }
        Expr::Intersect { expr, intersect } => {
            1 + group_op_node_count(expr) + group_op_node_count(intersect)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().map(group_op_node_count).sum(),
        Expr::Repeat { expr, .. } => group_op_node_count(expr),
        Expr::Shared(expr) => group_op_node_count(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => 0,
    }
}

fn split_top_level_group_ops(expr: &Expr) -> (Expr, Vec<Expr>, Vec<Expr>) {
    match expr {
        Expr::Exclude { expr, exclude } => {
            let (base, mut excluded, intersections) = split_top_level_group_ops(expr);
            excluded.push((**exclude).clone());
            (base, excluded, intersections)
        }
        Expr::Intersect { expr, intersect } => {
            let (base, excluded, mut intersections) = split_top_level_group_ops(expr);
            intersections.push((**intersect).clone());
            (base, excluded, intersections)
        }
        Expr::Shared(inner)
            if matches!(inner.as_ref(), Expr::Exclude { .. } | Expr::Intersect { .. }) => {
            split_top_level_group_ops(inner.as_ref())
        }
        _ => (expr.clone(), Vec::new(), Vec::new()),
    }
}

fn rebuild_single_visible_group_expression(
    compiled_exprs: &[Expr],
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> Option<Expr> {
    let mut used = vec![false; compiled_exprs.len()];
    let mut expression = compiled_exprs.first()?.clone();
    used[0] = true;

    let mut apply_hidden = |hidden: u32, intersect: bool| -> Option<()> {
        let hidden = hidden as usize;
        if hidden == 0 || hidden >= compiled_exprs.len() || used[hidden] {
            return None;
        }
        used[hidden] = true;
        let hidden_expression = compiled_exprs[hidden].clone();
        expression = if intersect {
            Expr::Intersect {
                expr: Box::new(expression.clone()),
                intersect: Box::new(hidden_expression),
            }
        } else {
            Expr::Exclude {
                expr: Box::new(expression.clone()),
                exclude: Box::new(hidden_expression),
            }
        };
        Some(())
    };

    if exclusions.keys().any(|&visible| visible != 0)
        || intersections.keys().any(|&visible| visible != 0)
    {
        return None;
    }
    for &hidden in exclusions.get(&0).into_iter().flatten() {
        apply_hidden(hidden, false)?;
    }
    for &hidden in intersections.get(&0).into_iter().flatten() {
        apply_hidden(hidden, true)?;
    }
    used.into_iter().all(|used| used).then(|| expression.optimize())
}

/// Expose one intersection nested under a common sequence shell. This is an
/// exact distributive rewrite:
///
/// ```text
/// prefix · (left ∩ right) · suffix
///   = (prefix · left · suffix) ∩ (prefix · right · suffix)
/// ```
///
/// Keeping the operands visible lets structural full/stencil compilation map
/// their residual coordinates independently and materialize missing correlated
/// product tuples. We deliberately handle exactly one nested intersection to
/// avoid an exponential distribution over multiple group operations.
fn lift_single_nested_intersection(expr: &Expr) -> Option<Expr> {
    let Expr::Seq(parts) = expr else {
        return None;
    };
    let mut intersection = None;
    for (index, part) in parts.iter().enumerate() {
        let part = match part {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        if let Expr::Intersect { expr, intersect } = part {
            if intersection.is_some() {
                return None;
            }
            intersection = Some((index, expr.as_ref(), intersect.as_ref()));
        }
    }
    let (index, left, right) = intersection?;
    let branch = |replacement: &Expr| {
        let mut branch = parts.to_vec();
        branch[index] = replacement.clone();
        Expr::Seq(branch).optimize()
    };
    Some(Expr::Intersect {
        expr: Box::new(branch(left)),
        intersect: Box::new(branch(right)),
    })
}

#[derive(Default)]
struct NestedGroupOpCache {
    compiled: FxHashMap<Expr, Arc<DFA>>,
    shared_duplicates: Option<Arc<SharedDuplicateNestedGroupOpCache>>,
    allow_shared_initialization: bool,
    cache_hits: usize,
    cache_misses: usize,
    compiled_ms: f64,
    max_compile_ms: f64,
}

struct SharedDuplicateNestedGroupOpCache {
    duplicated: FxHashSet<Expr>,
    compiled: Mutex<FxHashMap<Expr, Arc<OnceLock<Arc<DFA>>>>>,
}

fn unique_dense_exclusion_rhs_is_compound(expr: &Expr) -> bool {
    !matches!(expr, Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon)
}

impl SharedDuplicateNestedGroupOpCache {
    fn cell_if_duplicated(&self, expr: &Expr) -> Option<Arc<OnceLock<Arc<DFA>>>> {
        if !self.duplicated.contains(expr) {
            return None;
        }
        let mut compiled = self
            .compiled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Some(Arc::clone(
            compiled
                .entry(expr.clone())
                .or_insert_with(|| Arc::new(OnceLock::new())),
        ))
    }

    fn all_entries_initialized(&self) -> bool {
        let compiled = self
            .compiled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        compiled.len() == self.duplicated.len()
            && compiled.values().all(|cell| cell.get().is_some())
    }
}

fn materialize_nested_group_ops(expr: Expr, cache: &mut NestedGroupOpCache) -> Expr {
    match expr {
        expr @ (Expr::Exclude { .. } | Expr::Intersect { .. }) => {
            if let Some(compiled) = cache.compiled.get(&expr) {
                cache.cache_hits += 1;
                return Expr::Dfa(compiled.clone());
            }

            if let Some(shared) = cache.shared_duplicates.clone()
                && let Some(cell) = shared.cell_if_duplicated(&expr)
            {
                let started_at = Instant::now();
                let was_ready = cell.get().is_some();
                let compiled = if cache.allow_shared_initialization {
                    Arc::clone(cell.get_or_init(|| {
                        let mut nested_cache = NestedGroupOpCache {
                            shared_duplicates: Some(Arc::clone(&shared)),
                            allow_shared_initialization: true,
                            ..NestedGroupOpCache::default()
                        };
                        Arc::new(compile_with_plan(
                            build_exclusion_compile_plan_with_labels_and_cache(
                                std::slice::from_ref(&expr),
                                None,
                                &mut nested_cache,
                            ),
                        ))
                    }))
                } else {
                    Arc::clone(cell.get().expect(
                        "shared nested group-op cache must be prewarmed before parallel compilation",
                    ))
                };
                if was_ready {
                    cache.cache_hits += 1;
                } else {
                    cache.cache_misses += 1;
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    cache.compiled_ms += elapsed_ms;
                    cache.max_compile_ms = cache.max_compile_ms.max(elapsed_ms);
                }
                cache.compiled.insert(expr, Arc::clone(&compiled));
                return Expr::Dfa(compiled);
            }

            cache.cache_misses += 1;
            let started_at = Instant::now();
            let compiled = if let Expr::Exclude { expr: left, exclude: right } = &expr
                && unique_dense_exclusion_rhs_is_compound(right)
            {
                // Nested exclusions used to take the generic group-product path unless
                // the exact same subtree appeared often enough to enter the shared
                // prewarm cache.  Unique exclusions are just as amenable to the exact
                // dense binary subtraction kernel: materialize each operand once, then
                // keep a dead-RHS sentinel instead of hashing partial product tuples.
                // Atomic RHS exclusions are intentionally left on the generic planner:
                // recursively densifying long chains of exact literals repeatedly
                // rebuilds a growing left DFA and can be several times slower.
                // Fall back to the generic planner if either operand cannot use that
                // representation.
                let left = materialize_nested_group_ops((**left).clone(), cache);
                let right = materialize_nested_group_ops((**right).clone(), cache);
                let left = match left {
                    Expr::Dfa(dfa) => dfa,
                    other => Arc::new(compile_single_expr_dfa(&other)),
                };
                let right = match right {
                    Expr::Dfa(dfa) => dfa,
                    other => Arc::new(compile_single_expr_dfa(&other)),
                };
                build_dense_binary_exclusion_dfa(
                    left.as_ref(),
                    right.as_ref(),
                    expr_u8set(&expr),
                )
                .map(Arc::new)
                .unwrap_or_else(|| {
                    Arc::new(compile_with_plan(
                        build_exclusion_compile_plan_with_labels_and_cache(
                            std::slice::from_ref(&expr),
                            None,
                            cache,
                        ),
                    ))
                })
            } else {
                Arc::new(compile_with_plan(
                    build_exclusion_compile_plan_with_labels_and_cache(
                        std::slice::from_ref(&expr),
                        None,
                        cache,
                    ),
                ))
            };
            let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            cache.compiled_ms += elapsed_ms;
            cache.max_compile_ms = cache.max_compile_ms.max(elapsed_ms);
            cache.compiled.insert(expr, compiled.clone());
            Expr::Dfa(compiled)
        }
        Expr::Seq(parts) => Expr::Seq(
            parts
                .into_iter()
                .map(|part| materialize_nested_group_ops(part, cache))
                .collect(),
        ),
        Expr::Choice(options) => Expr::Choice(
            options
                .into_iter()
                .map(|option| materialize_nested_group_ops(option, cache))
                .collect(),
        ),
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(materialize_nested_group_ops(*expr, cache)),
            min,
            max,
        },
        Expr::Shared(inner) => {
            let rewritten = materialize_nested_group_ops((*inner).clone(), cache);
            if rewritten == *inner {
                Expr::Shared(inner)
            } else {
                rewritten
            }
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => expr,
    }
}

fn count_nested_group_ops(expr: &Expr, counts: &mut FxHashMap<Expr, usize>) {
    match expr {
        Expr::Exclude { expr: inner, exclude } => {
            *counts.entry(expr.clone()).or_default() += 1;
            count_nested_group_ops(inner, counts);
            count_nested_group_ops(exclude, counts);
        }
        Expr::Intersect { expr: inner, intersect } => {
            *counts.entry(expr.clone()).or_default() += 1;
            count_nested_group_ops(inner, counts);
            count_nested_group_ops(intersect, counts);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            for part in parts {
                count_nested_group_ops(part, counts);
            }
        }
        Expr::Repeat { expr, .. } => count_nested_group_ops(expr, counts),
        Expr::Shared(expr) => count_nested_group_ops(expr, counts),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
    }
}

#[derive(Clone, Copy)]
struct SubexpressionScanInfo {
    fingerprint: u64,
    size: usize,
}

struct SubexpressionCount<'a> {
    expr: &'a Expr,
    count: usize,
    size: usize,
}

fn scan_all_subexpressions<'a>(
    expr: &'a Expr,
    infos: &mut Vec<SubexpressionScanInfo>,
    counts_by_fingerprint: &mut FxHashMap<u64, Vec<SubexpressionCount<'a>>>,
    min_size: usize,
) -> SubexpressionScanInfo {
    let info_index = infos.len();
    infos.push(SubexpressionScanInfo { fingerprint: 0, size: 0 });
    let mut hasher = FxHasher::default();
    let size = match expr {
        Expr::U8Seq(bytes) => { 0u8.hash(&mut hasher); bytes.hash(&mut hasher); 1 }
        Expr::U8Class(class) => { 1u8.hash(&mut hasher); class.hash(&mut hasher); 1 }
        Expr::Dfa(dfa) => {
            2u8.hash(&mut hasher);
            (Arc::as_ptr(dfa) as usize).hash(&mut hasher);
            1
        }
        Expr::Intersect { expr, intersect } => {
            3u8.hash(&mut hasher);
            let left = scan_all_subexpressions(expr, infos, counts_by_fingerprint, min_size);
            let right = scan_all_subexpressions(intersect, infos, counts_by_fingerprint, min_size);
            left.fingerprint.hash(&mut hasher); right.fingerprint.hash(&mut hasher);
            1 + left.size + right.size
        }
        Expr::Seq(parts) => {
            4u8.hash(&mut hasher); parts.len().hash(&mut hasher);
            let mut size = 1usize;
            for part in parts {
                let child = scan_all_subexpressions(part, infos, counts_by_fingerprint, min_size);
                child.fingerprint.hash(&mut hasher); size = size.saturating_add(child.size);
            }
            size
        }
        Expr::Choice(parts) => {
            5u8.hash(&mut hasher); parts.len().hash(&mut hasher);
            let mut size = 1usize;
            for part in parts {
                let child = scan_all_subexpressions(part, infos, counts_by_fingerprint, min_size);
                child.fingerprint.hash(&mut hasher); size = size.saturating_add(child.size);
            }
            size
        }
        Expr::Exclude { expr, exclude } => {
            6u8.hash(&mut hasher);
            let left = scan_all_subexpressions(expr, infos, counts_by_fingerprint, min_size);
            let right = scan_all_subexpressions(exclude, infos, counts_by_fingerprint, min_size);
            left.fingerprint.hash(&mut hasher); right.fingerprint.hash(&mut hasher);
            1 + left.size + right.size
        }
        Expr::Repeat { expr, min, max } => {
            7u8.hash(&mut hasher); min.hash(&mut hasher); max.hash(&mut hasher);
            let child = scan_all_subexpressions(expr, infos, counts_by_fingerprint, min_size);
            child.fingerprint.hash(&mut hasher); 1 + child.size
        }
        Expr::Shared(inner) => {
            8u8.hash(&mut hasher);
            let child = scan_all_subexpressions(inner, infos, counts_by_fingerprint, min_size);
            child.fingerprint.hash(&mut hasher); 1 + child.size
        }
        Expr::Epsilon => { 9u8.hash(&mut hasher); 1 }
    };
    let info = SubexpressionScanInfo { fingerprint: hasher.finish(), size };
    infos[info_index] = info;
    if size >= min_size {
        let bucket = counts_by_fingerprint.entry(info.fingerprint).or_default();
        if let Some(existing) = bucket.iter_mut().find(|entry| entry.expr == expr) {
            existing.count += 1;
        } else {
            bucket.push(SubexpressionCount { expr, count: 1, size });
        }
    }
    info
}

fn collect_maximal_repeated_subexpressions_fast(
    expr: &Expr,
    infos: &[SubexpressionScanInfo],
    cursor: &mut usize,
    candidate_fingerprints: &FxHashSet<u64>,
    candidates: &FxHashSet<Expr>,
    selected: &mut FxHashSet<Expr>,
) {
    let info = infos[*cursor];
    *cursor += 1;
    if candidate_fingerprints.contains(&info.fingerprint) && candidates.contains(expr) {
        selected.insert(expr.clone());
        *cursor += info.size.saturating_sub(1);
        return;
    }
    match expr {
        Expr::Exclude { expr, exclude } => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
            collect_maximal_repeated_subexpressions_fast(exclude, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::Intersect { expr, intersect } => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
            collect_maximal_repeated_subexpressions_fast(intersect, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => for part in parts {
            collect_maximal_repeated_subexpressions_fast(part, infos, cursor, candidate_fingerprints, candidates, selected);
        },
        Expr::Repeat { expr, .. } => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::Shared(expr) => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
    }
}

fn replace_compiled_subexpressions(
    expr: Expr,
    compiled: &FxHashMap<Expr, Arc<DFA>>,
    replace_root: bool,
) -> Expr {
    if replace_root && let Some(dfa) = compiled.get(&expr) {
        return Expr::Dfa(Arc::clone(dfa));
    }
    match expr {
        Expr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            exclude: Box::new(replace_compiled_subexpressions(*exclude, compiled, true)),
        },
        Expr::Intersect { expr, intersect } => Expr::Intersect {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            intersect: Box::new(replace_compiled_subexpressions(*intersect, compiled, true)),
        },
        Expr::Seq(parts) => Expr::Seq(
            parts
                .into_iter()
                .map(|part| replace_compiled_subexpressions(part, compiled, true))
                .collect(),
        ),
        Expr::Choice(parts) => Expr::Choice(
            parts
                .into_iter()
                .map(|part| replace_compiled_subexpressions(part, compiled, true))
                .collect(),
        ),
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            min,
            max,
        },
        Expr::Shared(inner) => {
            let rewritten = replace_compiled_subexpressions((*inner).clone(), compiled, true);
            if rewritten == *inner {
                Expr::Shared(inner)
            } else {
                rewritten
            }
        }
        leaf @ (Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon) => leaf,
    }
}

fn materialize_repeated_subexpression_dfas_with_limits(
    exprs: &[Expr],
    min_size: usize,
    min_occurrences: usize,
) -> Option<Vec<Expr>> {
    debug_assert!(min_size > 1);
    debug_assert!(min_occurrences > 1);
    let started_at = Instant::now();
    let mut infos = Vec::<SubexpressionScanInfo>::new();
    let mut counts_by_fingerprint = FxHashMap::<u64, Vec<SubexpressionCount<'_>>>::default();
    for expr in exprs {
        scan_all_subexpressions(expr, &mut infos, &mut counts_by_fingerprint, min_size);
    }
    let mut occurrences = FxHashMap::<Expr, usize>::default();
    let mut candidate_fingerprints = FxHashSet::<u64>::default();
    for (&fingerprint, bucket) in &counts_by_fingerprint {
        for entry in bucket {
            if entry.count >= min_occurrences
                && entry.size >= min_size
                && !expr_contains_group_op(entry.expr)
                && !expression_contains_large_bounded_repeat(entry.expr)
                && !matches!(entry.expr, Expr::Dfa(_))
            {
                candidate_fingerprints.insert(fingerprint);
                occurrences.insert(entry.expr.clone(), entry.count);
            }
        }
    }
    let candidates = occurrences.keys().cloned().collect::<FxHashSet<_>>();
    let mut selected = FxHashSet::default();
    let mut cursor = 0usize;
    for expr in exprs {
        collect_maximal_repeated_subexpressions_fast(
            expr,
            &infos,
            &mut cursor,
            &candidate_fingerprints,
            &candidates,
            &mut selected,
        );
    }
    debug_assert_eq!(cursor, infos.len());
    if selected.is_empty() {
        return None;
    }

    let selected = selected.into_iter().collect::<Vec<_>>();
    let compile_started_at = Instant::now();
    let compiled_entries = selected
        .into_par_iter()
        .map(|expr| {
            let expr_size = expr_structural_size(&expr);
            let occurrence_count = occurrences.get(&expr).copied().unwrap_or(0);
            let entry_started_at = Instant::now();
            let dfa = Arc::new(compile_expr_to_dfa(&expr));
            let entry_compile_ms = entry_started_at.elapsed().as_secs_f64() * 1000.0;
            (expr, dfa, expr_size, occurrence_count, entry_compile_ms)
        })
        .collect::<Vec<_>>();
    let compile_ms = compile_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut compiled = FxHashMap::<Expr, Arc<DFA>>::default();
    for (expr, dfa, expr_size, occurrence_count, entry_compile_ms) in compiled_entries {
        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][tokenizer] shared_subexpr_dfa_entry size={} occurrences={} states={} transitions={} compile_ms={:.3}",
                expr_size,
                occurrence_count,
                dfa.num_states(),
                dfa_transition_count(&dfa),
                entry_compile_ms,
            );
        }
        compiled.insert(expr, dfa);
    }
    let rewritten = exprs
        .iter()
        .cloned()
        .map(|expr| replace_compiled_subexpressions(expr, &compiled, true))
        .collect::<Vec<_>>();
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_subexpr_dfas entries={} compile_ms={:.3} total_ms={:.3}",
            compiled.len(),
            compile_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(rewritten)
}

fn materialize_repeated_subexpression_dfas(exprs: &[Expr]) -> Option<Vec<Expr>> {
    if std::env::var_os("GLRMASK_DISABLE_SHARED_SUBEXPR_DFA_CACHE").is_some() {
        return None;
    }
    let forced = std::env::var_os("GLRMASK_FORCE_SHARED_SUBEXPR_DFA_CACHE").is_some();
    let min_total_size = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_TOTAL_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(40_000);
    let total_size = exprs.iter().map(expr_structural_size).sum::<usize>();
    if !forced && total_size < min_total_size {
        return None;
    }
    let min_size = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(80);
    let min_occurrences = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_OCCURRENCES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(12);
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_subexpr_scan forced={} total_size={} min_total_size={} min_size={} min_occurrences={}",
            forced, total_size, min_total_size, min_size, min_occurrences,
        );
    }
    materialize_repeated_subexpression_dfas_with_limits(exprs, min_size, min_occurrences)
}

fn shared_duplicate_nested_group_op_cache(
    exprs: &[Expr],
    grouped: &BTreeMap<u32, Vec<usize>>,
) -> Option<Arc<SharedDuplicateNestedGroupOpCache>> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let singleton_partitions = grouped
        .values()
        .filter(|terminal_ids| terminal_ids.len() == 1)
        .count();
    // The original policy targets many singleton partitions.  Product-trace
    // builds also have a distinct expensive shape: one large partition whose
    // nested group-op plan is itself parallelized.  Private per-expression
    // caches make duplicated nested exclusions/intersections in that partition
    // compile repeatedly.  Detect exactly the same minimum work threshold used
    // by the parallel planner before paying for the cross-partition duplicate
    // scan, so small/ordinary partitions keep the old zero-overhead path.
    const PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS: usize = 16;
    const PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS: usize = 128;
    let has_large_nested_partition = grouped.values().any(|terminal_ids| {
        terminal_ids.len() >= PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS
            && terminal_ids
                .iter()
                .map(|&terminal_id| group_op_node_count(&exprs[terminal_id]))
                .sum::<usize>()
                >= PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS
    });
    let force = std::env::var_os(
        "GLRMASK_EXPERIMENT_PRODUCT_TRACE_SHARED_NESTED_GROUP_OPS",
    )
    .is_some();
    let many_singleton_partitions =
        grouped.len() >= 64 && singleton_partitions * 4 >= grouped.len() * 3;
    if !force && !many_singleton_partitions && !has_large_nested_partition {
        return None;
    }

    let mut counts = FxHashMap::<Expr, usize>::default();
    for terminal_ids in grouped.values() {
        for &terminal_id in terminal_ids {
            let (base, excluded, intersections) = split_top_level_group_ops(&exprs[terminal_id]);
            count_nested_group_ops(&base, &mut counts);
            for expr in &excluded {
                count_nested_group_ops(expr, &mut counts);
            }
            for expr in &intersections {
                count_nested_group_ops(expr, &mut counts);
            }
        }
    }
    let duplicated = counts
        .into_iter()
        .filter_map(|(expr, count)| (count > 1).then_some(expr))
        .collect::<FxHashSet<_>>();
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_duplicate_nested_ops partitions={} singleton_partitions={} duplicated={} scan_ms={:.3}",
            grouped.len(),
            singleton_partitions,
            duplicated.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (!duplicated.is_empty()).then(|| {
        Arc::new(SharedDuplicateNestedGroupOpCache {
            duplicated,
            compiled: Mutex::new(FxHashMap::default()),
        })
    })
}

fn expr_structural_size(expr: &Expr) -> usize {
    match expr {
        Expr::Exclude { expr, exclude } => {
            1 + expr_structural_size(expr) + expr_structural_size(exclude)
        }
        Expr::Intersect { expr, intersect } => {
            1 + expr_structural_size(expr) + expr_structural_size(intersect)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            1 + parts.iter().map(expr_structural_size).sum::<usize>()
        }
        Expr::Repeat { expr, .. } => 1 + expr_structural_size(expr),
        Expr::Shared(expr) => 1 + expr_structural_size(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => 1,
    }
}

/// Populate every cross-partition nested-group-op cache entry before the
/// partition compilation itself fans out over Rayon.
///
/// Lazy `OnceLock` initialization inside the parallel partition loop can
/// deadlock: several workers may block on one cache entry while the worker
/// initializing it launches nested Rayon work that requires those same
/// workers. Compile proper subexpressions first, then their parents, so nested
/// shared entries are already available while a parent is materialized.
fn prewarm_shared_duplicate_nested_group_ops(
    shared: &Arc<SharedDuplicateNestedGroupOpCache>,
) {
    fn compile_materialized_operand(
        expr: &Expr,
        shared: &Arc<SharedDuplicateNestedGroupOpCache>,
    ) -> Arc<DFA> {
        let mut cache = NestedGroupOpCache {
            shared_duplicates: Some(Arc::clone(shared)),
            allow_shared_initialization: false,
            ..NestedGroupOpCache::default()
        };
        let materialized = materialize_nested_group_ops(expr.clone(), &mut cache);
        match materialized {
            Expr::Dfa(dfa) => dfa,
            other => Arc::new(compile_single_expr_dfa(&other)),
        }
    }

    fn try_compile_direct_exclusion(
        expr: &Expr,
        shared: &Arc<SharedDuplicateNestedGroupOpCache>,
    ) -> Option<Arc<DFA>> {
        let Expr::Exclude { expr: left, exclude: right } = expr else {
            return None;
        };
        let (left, right) = rayon::join(
            || compile_materialized_operand(left, shared),
            || compile_materialized_operand(right, shared),
        );
        build_dense_binary_exclusion_dfa(
            left.as_ref(),
            right.as_ref(),
            expr_u8set(expr),
        )
        .map(Arc::new)
    }

    fn collect_proper_shared_dependencies(
        expr: &Expr,
        duplicated: &FxHashSet<Expr>,
        out: &mut FxHashSet<Expr>,
        is_root: bool,
    ) {
        if !is_root
            && matches!(expr, Expr::Exclude { .. } | Expr::Intersect { .. })
            && duplicated.contains(expr)
        {
            out.insert(expr.clone());
        }
        match expr {
            Expr::Exclude { expr, exclude } => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
                collect_proper_shared_dependencies(exclude, duplicated, out, false);
            }
            Expr::Intersect { expr, intersect } => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
                collect_proper_shared_dependencies(intersect, duplicated, out, false);
            }
            Expr::Seq(parts) | Expr::Choice(parts) => {
                for part in parts {
                    collect_proper_shared_dependencies(part, duplicated, out, false);
                }
            }
            Expr::Repeat { expr, .. } => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
            }
            Expr::Shared(expr) => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
            }
            Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
        }
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let mut pending = shared
        .duplicated
        .iter()
        .cloned()
        .map(|expr| {
            let mut dependencies = FxHashSet::default();
            collect_proper_shared_dependencies(
                &expr,
                &shared.duplicated,
                &mut dependencies,
                true,
            );
            (expr, dependencies.into_iter().collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    let mut finished = FxHashSet::<Expr>::default();
    let mut layer_count = 0usize;

    while !pending.is_empty() {
        let mut ready = Vec::<Expr>::new();
        let mut blocked = Vec::new();
        for (expr, dependencies) in pending {
            if dependencies.iter().all(|dependency| finished.contains(dependency)) {
                ready.push(expr);
            } else {
                blocked.push((expr, dependencies));
            }
        }
        assert!(
            !ready.is_empty(),
            "duplicated nested group-op dependency graph must be acyclic",
        );
        layer_count += 1;

        ready.par_iter().for_each(|expr| {
            let cell = shared
                .cell_if_duplicated(expr)
                .expect("prewarm root must be registered as duplicated");
            if cell.get().is_some() {
                return;
            }
            let mut cache = NestedGroupOpCache {
                shared_duplicates: Some(Arc::clone(shared)),
                allow_shared_initialization: false,
                ..NestedGroupOpCache::default()
            };
            let compiled = try_compile_direct_exclusion(expr, shared).unwrap_or_else(|| {
                Arc::new(compile_with_plan(
                    build_exclusion_compile_plan_with_labels_and_cache(
                        std::slice::from_ref(expr),
                        None,
                        &mut cache,
                    ),
                ))
            });
            let _ = cell.set(compiled);
        });
        finished.extend(ready);
        pending = blocked;
    }

    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_nested_prewarm entries={} layers={} elapsed_ms={:.3}",
            finished.len(),
            layer_count,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

struct ExclusionCompilePlan {
    compiled_exprs: Vec<Expr>,
    exclusions: BTreeMap<u32, BTreeSet<u32>>,
    intersections: BTreeMap<u32, BTreeSet<u32>>,
    visible_groups: usize,
    profile_labels: Option<Vec<ProductComponentProfileLabel>>,
    /// Keep small product-construction kernels on the calling worker instead of
    /// entering nested Rayon regions. Used by request-local auxiliary lexers
    /// that are themselves built inside a parallel compiler DAG.
    local_small_product: bool,
}

struct ProductComponentProfileLabel {
    name: String,
    origin: &'static str,
    shared: bool,
}

struct ProductGrowthTrieNode {
    children: HashMap<u32, usize>,
}

impl ProductGrowthTrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
        }
    }
}

struct ProductGrowthRecorder {
    nodes: Vec<ProductGrowthTrieNode>,
    prefix_counts: Vec<usize>,
    dense_states: Vec<u32>,
}

impl ProductGrowthRecorder {
    fn new(num_groups: usize) -> Self {
        Self {
            nodes: vec![ProductGrowthTrieNode::new()],
            prefix_counts: vec![0; num_groups],
            dense_states: vec![0; num_groups],
        }
    }

    fn record(&mut self, num_groups: usize, state_tuple: &ProductStateTuple) {
        self.dense_states.fill(0);
        for &(group_id, state) in state_tuple {
            let group_index = group_id as usize;
            if group_index < num_groups {
                self.dense_states[group_index] = state.saturating_add(1);
            }
        }

        let mut node_index = 0usize;
        for (depth, &state) in self.dense_states.iter().enumerate() {
            let next_index = if let Some(&existing) = self.nodes[node_index].children.get(&state) {
                existing
            } else {
                let new_index = self.nodes.len();
                self.nodes.push(ProductGrowthTrieNode::new());
                self.nodes[node_index].children.insert(state, new_index);
                self.prefix_counts[depth] += 1;
                new_index
            };
            node_index = next_index;
        }
    }

    fn prefix_counts(&self) -> &[usize] {
        &self.prefix_counts
    }
}

fn expr_is_shared(expr: &Expr) -> bool {
    match expr {
        Expr::Shared(_) => true,
        Expr::Exclude { expr, exclude } => expr_is_shared(expr) || expr_is_shared(exclude),
        Expr::Intersect { expr, intersect } => expr_is_shared(expr) || expr_is_shared(intersect),
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_is_shared),
        Expr::Repeat { expr, .. } => expr_is_shared(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
    }
}

fn expr_profile_summary(expr: &Expr) -> String {
    const MAX_LEN: usize = 80;
    let mut summary = format!("{:?}", expr);
    if summary.len() > MAX_LEN {
        summary.truncate(MAX_LEN - 3);
        summary.push_str("...");
    }
    summary
}

fn build_exclusion_compile_plan_with_labels(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
) -> ExclusionCompilePlan {
    let mut nested_group_op_cache = NestedGroupOpCache::default();
    let plan = build_exclusion_compile_plan_with_labels_and_cache(
        exprs,
        visible_labels,
        &mut nested_group_op_cache,
    );
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        && nested_group_op_cache.cache_misses > 0
    {
        eprintln!(
            "[glrmask/profile][tokenizer] nested_group_ops cache_entries={} cache_hits={} cache_misses={} compiled_ms={:.3} max_compile_ms={:.3}",
            nested_group_op_cache.compiled.len(),
            nested_group_op_cache.cache_hits,
            nested_group_op_cache.cache_misses,
            nested_group_op_cache.compiled_ms,
            nested_group_op_cache.max_compile_ms,
        );
    }
    plan
}

fn build_exclusion_compile_plan_with_labels_and_cache(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    nested_group_op_cache: &mut NestedGroupOpCache,
) -> ExclusionCompilePlan {
    let visible_groups = exprs.len();
    let mut compiled_exprs = Vec::with_capacity(visible_groups);
    let mut deferred_exclusions = Vec::<Vec<Expr>>::with_capacity(visible_groups);
    let mut deferred_intersections = Vec::<Vec<Expr>>::with_capacity(visible_groups);
    let mut profile_labels = visible_labels.map(|_| Vec::with_capacity(visible_groups));

    if let Some(labels) = visible_labels {
        assert_eq!(
            labels.len(),
            visible_groups,
            "visible profile labels must match expression count"
        );
    }

    // Parallelize nested group-op materialization only when there is enough
    // actual set-operation work to amortize per-expression caches and Rayon
    // scheduling. `visible_groups` alone is a poor proxy: a medium-sized
    // partition can contain hundreds of nested exclusions, while a much larger
    // literal partition may contain none.
    const PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS: usize = 16;
    const PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS: usize = 128;
    let nested_group_ops = (visible_groups >= PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS)
        .then(|| exprs.iter().map(group_op_node_count).sum::<usize>())
        .unwrap_or(0);
    let shared_is_prewarmed = nested_group_op_cache
        .shared_duplicates
        .as_ref()
        .is_none_or(|shared| shared.all_entries_initialized());
    let parallel_materialize = visible_groups >= PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS
        && nested_group_ops >= PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS
        && shared_is_prewarmed
        && std::env::var_os("GLRMASK_DISABLE_PARALLEL_NESTED_GROUP_PLAN").is_none();
    if parallel_materialize {
        let shared_duplicates = nested_group_op_cache.shared_duplicates.clone();
        let materialized = exprs
            .par_iter()
            .map(|expr| {
                let mut local_cache = NestedGroupOpCache {
                    shared_duplicates: shared_duplicates.clone(),
                    allow_shared_initialization: false,
                    ..NestedGroupOpCache::default()
                };
                let (base, excluded, intersections) = split_top_level_group_ops(expr);
                let base = materialize_nested_group_ops(base, &mut local_cache);
                let excluded = excluded
                    .into_iter()
                    .map(|expr| materialize_nested_group_ops(expr, &mut local_cache))
                    .collect::<Vec<_>>();
                let intersections = intersections
                    .into_iter()
                    .map(|expr| materialize_nested_group_ops(expr, &mut local_cache))
                    .collect::<Vec<_>>();
                (base, excluded, intersections, local_cache)
            })
            .collect::<Vec<_>>();
        for (index, (base, excluded, intersections, local_cache)) in
            materialized.into_iter().enumerate()
        {
            nested_group_op_cache.cache_hits += local_cache.cache_hits;
            nested_group_op_cache.cache_misses += local_cache.cache_misses;
            nested_group_op_cache.compiled_ms += local_cache.compiled_ms;
            nested_group_op_cache.max_compile_ms = nested_group_op_cache
                .max_compile_ms
                .max(local_cache.max_compile_ms);
            assert!(
                !expr_contains_group_op(&base),
                "Expr::Exclude and Expr::Intersect are currently only supported at the top level of a terminal expression"
            );
            for excluded_expr in &excluded {
                assert!(
                    !expr_contains_group_op(excluded_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an exclusion branch is not supported"
                );
            }
            for intersection_expr in &intersections {
                assert!(
                    !expr_contains_group_op(intersection_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an intersection branch is not supported"
                );
            }
            compiled_exprs.push(base);
            if let (Some(labels), Some(profile_labels)) = (visible_labels, profile_labels.as_mut()) {
                profile_labels.push(ProductComponentProfileLabel {
                    name: labels[index].clone(),
                    origin: "visible",
                    shared: expr_is_shared(&exprs[index]),
                });
            }
            deferred_exclusions.push(excluded);
            deferred_intersections.push(intersections);
        }
    } else {
        for (index, expr) in exprs.iter().enumerate() {
            let (base, excluded, intersections) = split_top_level_group_ops(expr);
            let base = materialize_nested_group_ops(base, nested_group_op_cache);
            let excluded = excluded
                .into_iter()
                .map(|expr| materialize_nested_group_ops(expr, nested_group_op_cache))
                .collect::<Vec<_>>();
            let intersections = intersections
                .into_iter()
                .map(|expr| materialize_nested_group_ops(expr, nested_group_op_cache))
                .collect::<Vec<_>>();
            assert!(
                !expr_contains_group_op(&base),
                "Expr::Exclude and Expr::Intersect are currently only supported at the top level of a terminal expression"
            );
            for excluded_expr in &excluded {
                assert!(
                    !expr_contains_group_op(excluded_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an exclusion branch is not supported"
                );
            }
            for intersection_expr in &intersections {
                assert!(
                    !expr_contains_group_op(intersection_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an intersection branch is not supported"
                );
            }
            compiled_exprs.push(base);
            if let (Some(labels), Some(profile_labels)) = (visible_labels, profile_labels.as_mut()) {
                profile_labels.push(ProductComponentProfileLabel {
                    name: labels[index].clone(),
                    origin: "visible",
                    shared: expr_is_shared(expr),
                });
            }
            deferred_exclusions.push(excluded);
            deferred_intersections.push(intersections);
        }
    }

    let mut exclusions = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut intersections = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut next_group = visible_groups as u32;
    for (group_id, (excluded_exprs, intersection_exprs)) in deferred_exclusions
        .into_iter()
        .zip(deferred_intersections.into_iter())
        .enumerate()
    {
        let exclusion_entry = exclusions.entry(group_id as u32).or_default();
        for (excluded_index, excluded_expr) in excluded_exprs.into_iter().enumerate() {
            let is_shared = expr_is_shared(&excluded_expr);
            compiled_exprs.push(excluded_expr);
            exclusion_entry.insert(next_group);
            if let Some(profile_labels) = profile_labels.as_mut() {
                let base_name = profile_labels[group_id].name.clone();
                profile_labels.push(ProductComponentProfileLabel {
                    name: format!("{}::exclude#{}", base_name, excluded_index),
                    origin: "internal_exclusion",
                    shared: is_shared,
                });
            }
            next_group += 1;
        }

        let intersection_entry = intersections.entry(group_id as u32).or_default();
        for (intersection_index, intersection_expr) in intersection_exprs.into_iter().enumerate() {
            let is_shared = expr_is_shared(&intersection_expr);
            compiled_exprs.push(intersection_expr);
            intersection_entry.insert(next_group);
            if let Some(profile_labels) = profile_labels.as_mut() {
                let base_name = profile_labels[group_id].name.clone();
                profile_labels.push(ProductComponentProfileLabel {
                    name: format!("{}::intersect#{}", base_name, intersection_index),
                    origin: "internal_intersection",
                    shared: is_shared,
                });
            }
            next_group += 1;
        }
    }

    exclusions.retain(|_, v| !v.is_empty());
    intersections.retain(|_, v| !v.is_empty());

    ExclusionCompilePlan {
        compiled_exprs,
        exclusions,
        intersections,
        visible_groups,
        profile_labels,
        local_small_product: false,
    }
}

fn build_exclusion_compile_plan(exprs: &[Expr]) -> ExclusionCompilePlan {
    build_exclusion_compile_plan_with_labels(exprs, None)
}

fn expr_accepts_empty(expr: &Expr) -> bool {
    match expr {
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::U8Class(_) => false,
        Expr::Dfa(dfa) => !dfa.finalizers(0).is_empty(),
        Expr::Intersect { expr, intersect } => {
            expr_accepts_empty(expr) && expr_accepts_empty(intersect)
        }
        Expr::Seq(parts) => parts.iter().all(expr_accepts_empty),
        Expr::Choice(options) => options.iter().any(expr_accepts_empty),
        Expr::Exclude { expr, exclude } => expr_accepts_empty(expr) && !expr_accepts_empty(exclude),
        Expr::Repeat { expr: _, min, .. } => *min == 0,
        Expr::Shared(inner) => expr_accepts_empty(inner),
        Expr::Epsilon => true,
    }
}

pub(crate) fn expr_u8set(expr: &Expr) -> U8Set {
    match expr {
        Expr::U8Seq(bytes) => U8Set::from_bytes(bytes),
        Expr::U8Class(set) => *set,
        Expr::Dfa(dfa) => {
            let mut set = U8Set::empty();
            for state in dfa.states() {
                for (byte, _) in state.transitions.iter() {
                    set.insert(byte);
                }
            }
            set
        }
        Expr::Seq(parts) | Expr::Choice(parts) => parts
            .iter()
            .fold(U8Set::empty(), |acc, part| acc | expr_u8set(part)),
        Expr::Intersect { expr, intersect } => expr_u8set(expr).intersection(&expr_u8set(intersect)),
        Expr::Exclude { expr, .. } => expr_u8set(expr),
        Expr::Repeat { expr, .. } => expr_u8set(expr),
        Expr::Shared(inner) => expr_u8set(inner),
        Expr::Epsilon => U8Set::empty(),
    }
}

fn highest_power_of_two_leq(value: usize) -> usize {
    debug_assert!(value > 0);
    1usize << (usize::BITS - value.leading_zeros() - 1)
}

struct RepeatCompiler<'expr, 'nfa> {
    expr: &'expr Expr,
    nfa: &'nfa mut NFA,
    power_cache: HashMap<(usize, u32), u32>,
    upto_cache: HashMap<(usize, u32), u32>,
}

impl<'expr, 'nfa> RepeatCompiler<'expr, 'nfa> {
    fn new(expr: &'expr Expr, nfa: &'nfa mut NFA) -> Self {
        Self {
            expr,
            nfa,
            power_cache: HashMap::new(),
            upto_cache: HashMap::new(),
        }
    }

    fn compile_power(&mut self, copies: usize, end: u32) -> u32 {
        debug_assert!(copies.is_power_of_two());

        if let Some(&start) = self.power_cache.get(&(copies, end)) {
            return start;
        }

        let start = if copies == 1 {
            let start = self.nfa.add_state();
            append_compiled_expr(self.expr, self.nfa, start, end);
            start
        } else {
            let half = copies / 2;
            let suffix_start = self.compile_power(half, end);
            self.compile_power(half, suffix_start)
        };

        self.power_cache.insert((copies, end), start);
        start
    }

    fn compile_exact(&mut self, copies: usize, end: u32) -> u32 {
        if copies == 0 {
            return end;
        }

        let largest_power = highest_power_of_two_leq(copies);
        let suffix_start = self.compile_exact(copies - largest_power, end);
        self.compile_power(largest_power, suffix_start)
    }

    fn compile_upto(&mut self, copies: usize, end: u32) -> u32 {
        if copies == 0 {
            return end;
        }

        if let Some(&start) = self.upto_cache.get(&(copies, end)) {
            return start;
        }

        let largest_power = highest_power_of_two_leq(copies);
        let split = self.nfa.add_state();

        let smaller_start = self.compile_upto(largest_power - 1, end);
        self.nfa.add_epsilon(split, smaller_start);

        let suffix_start = self.compile_upto(copies - largest_power, end);
        let power_start = self.compile_power(largest_power, suffix_start);
        self.nfa.add_epsilon(split, power_start);

        self.upto_cache.insert((copies, end), split);
        split
    }
}

fn append_byte_sequence_expr(bytes: &[u8], nfa: &mut NFA, start: u32, end: u32) {
    let mut state = start;
    for (index, &byte) in bytes.iter().enumerate() {
        let next = if index + 1 == bytes.len() {
            end
        } else {
            nfa.add_state()
        };
        nfa.add_transition(state, byte, next);
        state = next;
    }

    if bytes.is_empty() {
        nfa.add_epsilon(start, end);
    }
}

fn append_dfa_expr(dfa: &DFA, nfa: &mut NFA, start: u32, end: u32) {
    let mut state_map = Vec::with_capacity(dfa.num_states());
    for _ in 0..dfa.num_states() {
        state_map.push(nfa.add_state());
    }
    nfa.add_epsilon(start, state_map[0]);

    for (state_id, state) in dfa.states().iter().enumerate() {
        let mapped_state = state_map[state_id];
        for (byte, &target) in state.transitions.iter() {
            nfa.add_transition(mapped_state, byte, state_map[target as usize]);
        }
        if !state.finalizers.is_empty() {
            nfa.add_epsilon(mapped_state, end);
        }
    }
}

fn append_sequence_expr(parts: &[Expr], nfa: &mut NFA, start: u32, end: u32) {
    let mut state = start;
    for (index, part) in parts.iter().enumerate() {
        let next = if index + 1 == parts.len() {
            end
        } else {
            nfa.add_state()
        };
        append_compiled_expr(part, nfa, state, next);
        state = next;
    }

    if parts.is_empty() {
        nfa.add_epsilon(start, end);
    }
}

fn append_choice_expr(options: &[Expr], nfa: &mut NFA, start: u32, end: u32) {
    if options.is_empty() {
        nfa.add_epsilon(start, end);
        return;
    }

    for option in options {
        append_compiled_expr(option, nfa, start, end);
    }
}

const DIRECT_BOUNDED_REPEAT_THRESHOLD: usize = 32;

fn compile_expr_to_dfa(expr: &Expr) -> DFA {
    let mut nfa = build_regex_nfa_impl(std::slice::from_ref(expr));
    nfa.condense_epsilon_sccs();
    nfa.to_minimized_dfa()
}

fn direct_expression_graph_byte_classes(local_dfas: &[DFA]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut classes = vec![0u8; 256];
    let mut num_classes = 1usize;
    for dfa in local_dfas {
        for state in 0..dfa.num_states() as u32 {
            if num_classes == 256 {
                break;
            }
            let mut keys = FxHashMap::<(u8, u32), u8>::default();
            let mut refined = vec![0u8; 256];
            let mut next_class = 0u16;
            for byte in 0u8..=255 {
                let key = (classes[byte as usize], dfa.step(state, byte).unwrap_or(u32::MAX));
                let class = *keys.entry(key).or_insert_with(|| {
                    let class = next_class as u8;
                    next_class += 1;
                    class
                });
                refined[byte as usize] = class;
            }
            if refined != classes {
                classes = refined;
                num_classes = next_class as usize;
            }
        }
    }
    let mut members = vec![Vec::<u8>::new(); num_classes];
    for byte in 0u8..=255 {
        members[classes[byte as usize] as usize].push(byte);
    }
    (classes, members)
}

fn direct_expression_graph_class_transitions(
    local_dfas: &[DFA],
    class_members: &[Vec<u8>],
) -> Vec<Vec<Box<[(u8, u32)]>>> {
    local_dfas
        .iter()
        .map(|dfa| {
            (0..dfa.num_states() as u32)
                .map(|state| {
                    let mut transitions = Vec::new();
                    for (class, members) in class_members.iter().enumerate() {
                        let target = dfa.step(state, members[0]);
                        debug_assert!(members
                            .iter()
                            .all(|&byte| dfa.step(state, byte) == target));
                        if let Some(target) = target {
                            transitions.push((class as u8, target));
                        }
                    }
                    transitions.into_boxed_slice()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn expand_direct_expression_graph_classes(dfa: &mut DFA, class_members: &[Vec<u8>]) {
    let expanded = dfa
        .states()
        .iter()
        .map(|state| {
            let capacity = state
                .transitions
                .iter()
                .map(|(class, _)| class_members[class as usize].len())
                .sum();
            let mut entries = Vec::with_capacity(capacity);
            for (class, &target) in state.transitions.iter() {
                entries.extend(
                    class_members[class as usize]
                        .iter()
                        .map(|&byte| (byte, target)),
                );
            }
            entries.sort_unstable_by_key(|entry| entry.0);
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(entries)
        })
        .collect::<Vec<_>>();
    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded) {
        state.transitions = transitions;
    }
}

struct DirectExpressionGraphEdge {
    symbol: usize,
    targets: Box<[u32]>,
    config_base: u32,
}

struct DirectReadyExpansion {
    configs: Box<[u32]>,
    accepting: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DirectGroupedConfig {
    symbol: u32,
    local_state: u32,
    edge_set: u32,
}

#[derive(Clone, Debug)]
struct DirectGroupedState {
    accepting: bool,
    groups: Box<[DirectGroupedConfig]>,
    hash: u64,
}

impl DirectGroupedState {
    fn new(accepting: bool, groups: Box<[DirectGroupedConfig]>) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        std::hash::Hash::hash(&accepting, &mut hasher);
        std::hash::Hash::hash(&groups, &mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);
        Self {
            accepting,
            groups,
            hash,
        }
    }
}

impl PartialEq for DirectGroupedState {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.accepting == other.accepting
            && self.groups == other.groups
    }
}

impl Eq for DirectGroupedState {}

impl std::hash::Hash for DirectGroupedState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[derive(Clone)]
struct DirectGroupedContinuation {
    accepting: bool,
    groups: Box<[DirectGroupedConfig]>,
}

struct DirectEdgeSetInterner {
    sets: Vec<Arc<[u32]>>,
    map: FxHashMap<Arc<[u32]>, u32>,
    union_cache: FxHashMap<Arc<[u32]>, u32>,
    merge_buffer: Vec<u32>,
    union_cache_hits: usize,
    union_cache_misses: usize,
}

impl DirectEdgeSetInterner {
    fn new() -> Self {
        Self {
            sets: Vec::new(),
            map: FxHashMap::default(),
            union_cache: FxHashMap::default(),
            merge_buffer: Vec::new(),
            union_cache_hits: 0,
            union_cache_misses: 0,
        }
    }

    fn intern_sorted(&mut self, edges: &[u32]) -> u32 {
        debug_assert!(!edges.is_empty());
        debug_assert!(edges.windows(2).all(|pair| pair[0] < pair[1]));
        if let Some(&existing) = self.map.get(edges) {
            return existing;
        }
        let id = self.sets.len() as u32;
        let edges: Arc<[u32]> = Arc::from(edges);
        self.map.insert(Arc::clone(&edges), id);
        self.sets.push(edges);
        id
    }

    fn get(&self, id: u32) -> &[u32] {
        &self.sets[id as usize]
    }

    fn union(&mut self, ids: &[u32]) -> u32 {
        debug_assert!(!ids.is_empty());
        debug_assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        if ids.len() == 1 {
            return ids[0];
        }
        if let Some(&existing) = self.union_cache.get(ids) {
            self.union_cache_hits += 1;
            return existing;
        }
        self.union_cache_misses += 1;
        self.merge_buffer.clear();
        for &id in ids {
            self.merge_buffer
                .extend_from_slice(&self.sets[id as usize]);
        }
        self.merge_buffer.sort_unstable();
        self.merge_buffer.dedup();
        let result = if let Some(&existing) = self.map.get(self.merge_buffer.as_slice()) {
            existing
        } else {
            let id = self.sets.len() as u32;
            let edges: Arc<[u32]> = Arc::from(self.merge_buffer.as_slice());
            self.map.insert(Arc::clone(&edges), id);
            self.sets.push(edges);
            id
        };
        self.union_cache.insert(Arc::from(ids), result);
        result
    }
}

fn group_direct_configs(
    configs: &[u32],
    config_edge: &[u32],
    config_local_state: &[u32],
    edges: &[DirectExpressionGraphEdge],
    edge_sets: &mut DirectEdgeSetInterner,
) -> Box<[DirectGroupedConfig]> {
    let mut entries = configs
        .iter()
        .map(|&config| {
            let edge = config_edge[config as usize];
            let edge_data = &edges[edge as usize];
            (
                edge_data.symbol as u32,
                config_local_state[config as usize],
                edge,
            )
        })
        .collect::<Vec<_>>();
    entries.sort_unstable();
    entries.dedup();
    let mut groups = Vec::new();
    let mut position = 0usize;
    while position < entries.len() {
        let (symbol, local_state, _) = entries[position];
        let start = position;
        position += 1;
        while position < entries.len()
            && entries[position].0 == symbol
            && entries[position].1 == local_state
        {
            position += 1;
        }
        let edge_ids = entries[start..position]
            .iter()
            .map(|entry| entry.2)
            .collect::<Vec<_>>();
        let edge_set = edge_sets.intern_sorted(&edge_ids);
        groups.push(DirectGroupedConfig {
            symbol,
            local_state,
            edge_set,
        });
    }
    groups.into_boxed_slice()
}

fn canonicalize_direct_group_fragments(
    fragments: &mut Vec<DirectGroupedConfig>,
    edge_sets: &mut DirectEdgeSetInterner,
    edge_set_ids: &mut Vec<u32>,
    output: &mut Vec<DirectGroupedConfig>,
) {
    fragments.sort_unstable();
    output.clear();
    let mut position = 0usize;
    while position < fragments.len() {
        let symbol = fragments[position].symbol;
        let local_state = fragments[position].local_state;
        edge_set_ids.clear();
        while position < fragments.len()
            && fragments[position].symbol == symbol
            && fragments[position].local_state == local_state
        {
            edge_set_ids.push(fragments[position].edge_set);
            position += 1;
        }
        edge_set_ids.sort_unstable();
        edge_set_ids.dedup();
        output.push(DirectGroupedConfig {
            symbol,
            local_state,
            edge_set: edge_sets.union(edge_set_ids),
        });
    }
}

fn compile_direct_expression_graph_grouped(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    edges: &[DirectExpressionGraphEdge],
    config_edge: &[u32],
    config_local_state: &[u32],
    ready_expansions: &[DirectReadyExpansion],
    initial_accepting: bool,
    initial_configs: &[u32],
    local_dfas: &[DFA],
    local_class_transitions: &[Vec<Box<[(u8, u32)]>>],
    class_members: &[Vec<u8>],
    profile: bool,
) -> DFA {
    let setup_started = profile.then(Instant::now);
    let mut edge_sets = DirectEdgeSetInterner::new();
    let ready_groups = ready_expansions
        .iter()
        .map(|expansion| {
            group_direct_configs(
                &expansion.configs,
                config_edge,
                config_local_state,
                edges,
                &mut edge_sets,
            )
        })
        .collect::<Vec<_>>();
    let mut fragments = Vec::<DirectGroupedConfig>::new();
    let mut edge_set_ids = Vec::<u32>::new();
    let mut canonical_groups = Vec::<DirectGroupedConfig>::new();
    let grouped_continuations = edges
        .iter()
        .map(|edge| {
            fragments.clear();
            let mut accepting = false;
            for &target in edge.targets.iter() {
                accepting |= ready_expansions[target as usize].accepting;
                fragments.extend_from_slice(&ready_groups[target as usize]);
            }
            canonicalize_direct_group_fragments(
                &mut fragments,
                &mut edge_sets,
                &mut edge_set_ids,
                &mut canonical_groups,
            );
            DirectGroupedContinuation {
                accepting,
                groups: canonical_groups.clone().into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>();
    let initial_groups = group_direct_configs(
        initial_configs,
        config_edge,
        config_local_state,
        edges,
        &mut edge_sets,
    );
    let setup_ms = setup_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let determinize_started = profile.then(Instant::now);
    let initial = Arc::new(DirectGroupedState::new(initial_accepting, initial_groups));
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    let state_capacity = config_edge.len().max(edges.len() * 8);
    let mut state_map = FxHashMap::<Arc<DirectGroupedState>, u32>::default();
    state_map.reserve(state_capacity);
    state_map.insert(Arc::clone(&initial), 0);
    let mut states = Vec::<Arc<DirectGroupedState>>::with_capacity(state_capacity);
    states.push(initial);
    let mut accepting_states = Vec::with_capacity(state_capacity);
    accepting_states.push(initial_accepting);
    let mut worklist = Vec::with_capacity(state_capacity);
    worklist.push(0u32);

    let num_classes = class_members.len();
    let mut class_fragments = (0..num_classes)
        .map(|_| Vec::<DirectGroupedConfig>::new())
        .collect::<Vec<_>>();
    let mut class_accepting = vec![false; num_classes];
    let mut class_used = vec![false; num_classes];
    let mut used_classes = Vec::<u8>::with_capacity(num_classes);
    let mut target_groups = Vec::<DirectGroupedConfig>::new();
    let mut state_hits = 0usize;
    let mut state_misses = 0usize;
    let mut source_groups = 0usize;
    let mut direct_group_contributions = 0usize;
    let mut continuation_group_contributions = 0usize;
    let mut scan_duration = std::time::Duration::ZERO;
    let mut canonicalize_duration = std::time::Duration::ZERO;
    let mut lookup_duration = std::time::Duration::ZERO;

    while let Some(source_state) = worklist.pop() {
        let source = Arc::clone(&states[source_state as usize]);
        if profile {
            source_groups += source.groups.len();
        }
        let scan_started = profile.then(Instant::now);
        for group in source.groups.iter().copied() {
            let symbol = group.symbol as usize;
            let local_dfa = &local_dfas[symbol];
            for &(class, next_local) in
                local_class_transitions[symbol][group.local_state as usize].iter()
            {
                let class_index = class as usize;
                if !class_used[class_index] {
                    class_used[class_index] = true;
                    used_classes.push(class);
                }
                if !local_class_transitions[symbol][next_local as usize].is_empty() {
                    class_fragments[class_index].push(DirectGroupedConfig {
                        symbol: group.symbol,
                        local_state: next_local,
                        edge_set: group.edge_set,
                    });
                    if profile {
                        direct_group_contributions += 1;
                    }
                }
                if !local_dfa.finalizers(next_local).is_empty() {
                    for &edge in edge_sets.get(group.edge_set) {
                        let continuation = &grouped_continuations[edge as usize];
                        class_accepting[class_index] |= continuation.accepting;
                        class_fragments[class_index].extend_from_slice(&continuation.groups);
                        if profile {
                            continuation_group_contributions += continuation.groups.len();
                        }
                    }
                }
            }
        }
        if let Some(started) = scan_started {
            scan_duration += started.elapsed();
        }

        used_classes.sort_unstable();
        let mut transitions = Vec::with_capacity(used_classes.len());
        for &class in &used_classes {
            let class_index = class as usize;
            let canonicalize_started = profile.then(Instant::now);
            canonicalize_direct_group_fragments(
                &mut class_fragments[class_index],
                &mut edge_sets,
                &mut edge_set_ids,
                &mut target_groups,
            );
            if let Some(started) = canonicalize_started {
                canonicalize_duration += started.elapsed();
            }
            let accepting = class_accepting[class_index];
            class_fragments[class_index].clear();
            class_accepting[class_index] = false;
            class_used[class_index] = false;
            if target_groups.is_empty() && !accepting {
                continue;
            }

            let lookup_started = profile.then(Instant::now);
            let key = DirectGroupedState::new(
                accepting,
                std::mem::take(&mut target_groups).into_boxed_slice(),
            );
            let target = if let Some(&existing) = state_map.get(&key) {
                if profile {
                    state_hits += 1;
                }
                target_groups = key.groups.into_vec();
                existing
            } else {
                if profile {
                    state_misses += 1;
                }
                let target = dfa.add_state();
                let key = Arc::new(key);
                state_map.insert(Arc::clone(&key), target);
                states.push(key);
                accepting_states.push(accepting);
                worklist.push(target);
                target_groups = Vec::new();
                target
            };
            if let Some(started) = lookup_started {
                lookup_duration += started.elapsed();
            }
            transitions.push((class, target));
        }
        used_classes.clear();
        dfa.set_transitions_from_sorted_entries(source_state, transitions);
    }
    set_direct_expression_graph_finalizers(&mut dfa, &accepting_states);
    let determinize_ms = determinize_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let pre_states = dfa.num_states();
    let pre_transitions = dfa.transition_count();
    let minimize_started = profile.then(Instant::now);
    let mut minimized = dfa.minimize_owned_reachable();
    let minimize_ms = minimize_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let post_class_transitions = minimized.transition_count();
    let expand_started = profile.then(Instant::now);
    expand_direct_expression_graph_classes(&mut minimized, class_members);
    let expand_ms = expand_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] grouped_expression_graph edge_sets={} union_cache_hits={} union_cache_misses={} pre_states={} pre_class_transitions={} post_states={} post_class_transitions={} post_byte_transitions={} avg_source_groups={:.2} direct_group_contributions={} continuation_group_contributions={} state_hits={} state_misses={} setup_ms={:.3} scan_ms={:.3} canonicalize_ms={:.3} lookup_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} expand_ms={:.3}",
            edge_sets.sets.len(),
            edge_sets.union_cache_hits,
            edge_sets.union_cache_misses,
            pre_states,
            pre_transitions,
            minimized.num_states(),
            post_class_transitions,
            minimized.transition_count(),
            source_groups as f64 / pre_states.max(1) as f64,
            direct_group_contributions,
            continuation_group_contributions,
            state_hits,
            state_misses,
            setup_ms,
            scan_duration.as_secs_f64() * 1000.0,
            canonicalize_duration.as_secs_f64() * 1000.0,
            lookup_duration.as_secs_f64() * 1000.0,
            determinize_ms,
            minimize_ms,
            expand_ms,
        );
    }
    minimized
}

fn direct_expression_graph_ready_expansions(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    outgoing_edges: &[Vec<u32>],
    edges: &[DirectExpressionGraphEdge],
    local_dfas: &[DFA],
    local_class_transitions: &[Vec<Box<[(u8, u32)]>>],
) -> Vec<DirectReadyExpansion> {
    let num_states = graph.states.len();
    let mut marks = vec![0u32; num_states];
    let mut generation = 0u32;
    let mut stack = Vec::<u32>::new();
    let mut configs = Vec::<u32>::new();
    let mut result = Vec::with_capacity(num_states);

    for root in 0..num_states {
        generation = generation.wrapping_add(1);
        if generation == 0 {
            marks.fill(0);
            generation = 1;
        }
        stack.clear();
        configs.clear();
        stack.push(root as u32);
        marks[root] = generation;
        let mut accepting = false;
        while let Some(state) = stack.pop() {
            let graph_state = &graph.states[state as usize];
            accepting |= graph_state.is_accepting;
            for &edge_id in &outgoing_edges[state as usize] {
                let edge = &edges[edge_id as usize];
                let local_dfa = &local_dfas[edge.symbol];
                if !local_class_transitions[edge.symbol][0].is_empty() {
                    configs.push(edge.config_base);
                }
                if !local_dfa.finalizers(0).is_empty() {
                    for &target in edge.targets.iter() {
                        let mark = &mut marks[target as usize];
                        if *mark != generation {
                            *mark = generation;
                            stack.push(target);
                        }
                    }
                }
            }
            for &target in &graph_state.epsilons {
                let mark = &mut marks[target as usize];
                if *mark != generation {
                    *mark = generation;
                    stack.push(target);
                }
            }
        }
        configs.sort_unstable();
        configs.dedup();
        result.push(DirectReadyExpansion {
            configs: configs.clone().into_boxed_slice(),
            accepting,
        });
    }
    result
}

fn union_direct_ready_expansions(
    targets: impl IntoIterator<Item = u32>,
    expansions: &[DirectReadyExpansion],
    config_marks: &mut [u32],
    generation: &mut u32,
    configs: &mut Vec<u32>,
) -> bool {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        config_marks.fill(0);
        *generation = 1;
    }
    configs.clear();
    let mut accepting = false;
    for target in targets {
        let expansion = &expansions[target as usize];
        accepting |= expansion.accepting;
        for &config in expansion.configs.iter() {
            let mark = &mut config_marks[config as usize];
            if *mark != *generation {
                *mark = *generation;
                configs.push(config);
            }
        }
    }
    configs.sort_unstable();
    accepting
}

fn set_direct_expression_graph_finalizers(dfa: &mut DFA, accepting: &[bool]) {
    for (state, &is_accepting) in accepting.iter().enumerate() {
        let mut finalizers = BitSet::new(1);
        if is_accepting {
            finalizers.set(0);
        }
        // The owned minimizer recomputes strict possible-future metadata after
        // quotienting. Computing it on the pre-minimized graph would be a full
        // redundant graph traversal.
        dfa.overwrite_state_metadata(state as u32, finalizers, BitSet::new(1));
    }
}

fn try_compile_expression_labeled_nfa_direct(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    symbols: &[Expr],
    force: bool,
) -> Result<Option<DFA>, String> {
    let edge_group_count = graph
        .states
        .iter()
        .map(|state| state.transitions.len())
        .sum::<usize>();
    if !force && (graph.states.len() < 128 || edge_group_count < 64) {
        return Ok(None);
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started = profile.then(Instant::now);
    let local_started = profile.then(Instant::now);
    let local_dfas = symbols
        .par_iter()
        .map(compile_terminal_expr_dfa)
        .collect::<Vec<_>>();
    let local_ms = local_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if local_dfas
        .iter()
        .any(|dfa| dfa.num_states() == 0 || dfa.epsilon_transition_count() != 0)
    {
        return Ok(None);
    }

    let (_byte_class_map, class_members) = direct_expression_graph_byte_classes(&local_dfas);
    let local_class_transitions =
        direct_expression_graph_class_transitions(&local_dfas, &class_members);

    let setup_started = profile.then(Instant::now);
    let mut outgoing_edges = vec![Vec::<u32>::new(); graph.states.len()];
    let mut edges = Vec::<DirectExpressionGraphEdge>::with_capacity(edge_group_count);
    let mut config_edge = Vec::<u32>::new();
    let mut config_local_state = Vec::<u32>::new();
    for (source, state) in graph.states.iter().enumerate() {
        for (&label, targets) in &state.transitions {
            let symbol = usize::try_from(label)
                .map_err(|_| format!("NFA has invalid negative symbol label {label}"))?;
            let local_dfa = local_dfas.get(symbol).ok_or_else(|| {
                format!(
                    "NFA symbol label {label} is out of range for {} symbols",
                    symbols.len()
                )
            })?;
            let config_base = u32::try_from(config_edge.len())
                .map_err(|_| "direct expression graph configuration overflow".to_string())?;
            let edge_id = u32::try_from(edges.len())
                .map_err(|_| "direct expression graph edge overflow".to_string())?;
            let mut checked_targets = Vec::with_capacity(targets.len());
            for &target in targets {
                if target as usize >= graph.states.len() {
                    return Err(format!("NFA target state {target} is out of range"));
                }
                checked_targets.push(target);
            }
            checked_targets.sort_unstable();
            checked_targets.dedup();
            for local_state in 0..local_dfa.num_states() {
                config_edge.push(edge_id);
                config_local_state.push(local_state as u32);
            }
            edges.push(DirectExpressionGraphEdge {
                symbol,
                targets: checked_targets.into_boxed_slice(),
                config_base,
            });
            outgoing_edges[source].push(edge_id);
        }
    }
    if config_edge.is_empty() {
        return Ok(None);
    }

    for &start in &graph.start_states {
        if start as usize >= graph.states.len() {
            return Err(format!("NFA start state {start} is out of range"));
        }
    }

    let ready_expansions = direct_expression_graph_ready_expansions(
        graph,
        &outgoing_edges,
        &edges,
        &local_dfas,
        &local_class_transitions,
    );
    let mut config_marks = vec![0u32; config_edge.len()];
    let mut generation = 0u32;
    let mut initial_configs = Vec::<u32>::new();
    let initial_accepting = union_direct_ready_expansions(
        graph.start_states.iter().copied(),
        &ready_expansions,
        &mut config_marks,
        &mut generation,
        &mut initial_configs,
    );
    let setup_ms = setup_started.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let result = compile_direct_expression_graph_grouped(
        graph,
        &edges,
        &config_edge,
        &config_local_state,
        &ready_expansions,
        initial_accepting,
        &initial_configs,
        &local_dfas,
        &local_class_transitions,
        &class_members,
        profile,
    );
    if let Some(started) = total_started {
        eprintln!(
            "[glrmask/profile][tokenizer] grouped_expression_graph_total graph_states={} graph_edges={} symbols={} classes={} local_states={} configs={} local_ms={:.3} outer_setup_ms={:.3} total_ms={:.3}",
            graph.states.len(),
            edge_group_count,
            symbols.len(),
            class_members.len(),
            local_dfas.iter().map(DFA::num_states).sum::<usize>(),
            config_edge.len(),
            local_ms,
            setup_ms,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(Some(result))
}

fn compile_expression_labeled_nfa_via_byte_nfa(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    symbols: &[Expr],
) -> Result<DFA, String> {
    // State zero is a synthetic root so multiple starts remain representable.
    let mut nfa = NFA::new(1);
    let state_map = (0..graph.states.len())
        .map(|_| nfa.add_state())
        .collect::<Vec<_>>();

    for &start in &graph.start_states {
        let Some(&mapped) = state_map.get(start as usize) else {
            return Err(format!("NFA start state {start} is out of range"));
        };
        nfa.add_epsilon(0, mapped);
    }

    for (state_id, state) in graph.states.iter().enumerate() {
        let mapped_from = state_map[state_id];
        if state.is_accepting {
            nfa.add_finalizer(mapped_from, 0);
        }

        for &target in &state.epsilons {
            let Some(&mapped_target) = state_map.get(target as usize) else {
                return Err(format!("NFA epsilon target {target} is out of range"));
            };
            nfa.add_epsilon(mapped_from, mapped_target);
        }

        for (&label, targets) in &state.transitions {
            let symbol_index = usize::try_from(label)
                .map_err(|_| format!("NFA has invalid negative symbol label {label}"))?;
            let symbol = symbols.get(symbol_index).ok_or_else(|| {
                format!(
                    "NFA symbol label {label} is out of range for {} symbols",
                    symbols.len()
                )
            })?;
            for &target in targets {
                let Some(&mapped_target) = state_map.get(target as usize) else {
                    return Err(format!("NFA target state {target} is out of range"));
                };
                append_compiled_expr(symbol, &mut nfa, mapped_from, mapped_target);
            }
        }
    }

    nfa.condense_epsilon_sccs();
    Ok(nfa.to_minimized_dfa())
}

/// Compile an explicit expression-labelled graph into one byte-level lexer DFA.
///
/// Large graphs are composed directly from independently compiled edge DFAs,
/// including nullable edge languages. This preserves the graph's factorization
/// and avoids constructing
/// a global epsilon NFA whose powerset immediately has to rediscover it.
pub fn compile_expression_labeled_nfa(
    graph: &crate::automata::unweighted_u32::nfa::NFA,
    symbols: &[Expr],
) -> Result<DFA, String> {
    if graph.start_states.is_empty() {
        return Err("expression-labelled NFA has no start state".to_string());
    }
    if let Some(dfa) = try_compile_expression_labeled_nfa_direct(graph, symbols, false)? {
        return Ok(dfa);
    }
    compile_expression_labeled_nfa_via_byte_nfa(graph, symbols)
}

fn productive_dfa_states(dfa: &DFA) -> Vec<bool> {
    let mut reverse_edges = vec![Vec::new(); dfa.num_states()];
    for (state_id, state) in dfa.states().iter().enumerate() {
        for (_, &target) in state.transitions.iter() {
            reverse_edges[target as usize].push(state_id as u32);
        }
    }

    let mut productive = vec![false; dfa.num_states()];
    let mut stack = Vec::new();
    for state_id in 0..dfa.num_states() as u32 {
        if !dfa.finalizers(state_id).is_empty() {
            productive[state_id as usize] = true;
            stack.push(state_id);
        }
    }

    while let Some(state_id) = stack.pop() {
        for &pred in &reverse_edges[state_id as usize] {
            if !productive[pred as usize] {
                productive[pred as usize] = true;
                stack.push(pred);
            }
        }
    }

    productive
}

fn dfa_is_nonnullable_and_prefix_free(dfa: &DFA) -> bool {
    if !dfa.finalizers(0).is_empty() {
        return false;
    }

    let productive = productive_dfa_states(dfa);
    for state in dfa.states() {
        if state.finalizers.is_empty() {
            continue;
        }
        for (_, &target) in state.transitions.iter() {
            if productive[target as usize] {
                return false;
            }
        }
    }

    true
}

type RepeatBaseDfaCache = FxHashMap<Expr, Arc<DFA>>;

fn cached_direct_bounded_repeat_base_dfa_unconditionally(
    expr: &Expr,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<Arc<DFA>> {
    let expr = unwrap_shared(expr);
    if let Some(cached) = cache.and_then(|cache| cache.get(expr)) {
        return Some(Arc::clone(cached));
    }
    compile_direct_bounded_repeat_base_dfa_unconditionally(expr).map(Arc::new)
}

fn cached_direct_bounded_repeat_base_dfa(
    expr: &Expr,
    max: usize,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<Arc<DFA>> {
    if max < DIRECT_BOUNDED_REPEAT_THRESHOLD {
        return None;
    }
    cached_direct_bounded_repeat_base_dfa_unconditionally(expr, cache)
}

fn compile_direct_bounded_repeat_base_dfa_unconditionally(expr: &Expr) -> Option<DFA> {
    let base_dfa = compile_expr_to_dfa(expr);
    if base_dfa.num_states() == 0 || !dfa_is_nonnullable_and_prefix_free(&base_dfa) {
        return None;
    }

    Some(base_dfa)
}

fn compile_direct_bounded_repeat_base_dfa(expr: &Expr, max: usize) -> Option<DFA> {
    if max < DIRECT_BOUNDED_REPEAT_THRESHOLD {
        return None;
    }
    compile_direct_bounded_repeat_base_dfa_unconditionally(expr)
}

fn build_bounded_repeat_dfa_with_cache(
    expr: &Expr,
    min: usize,
    max: usize,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<DFA> {
    let base_dfa = cached_direct_bounded_repeat_base_dfa(expr, max, cache)?;
    build_bounded_repeat_dfa_from_base(base_dfa.as_ref(), min, max)
}

fn build_bounded_repeat_dfa(expr: &Expr, min: usize, max: usize) -> Option<DFA> {
    build_bounded_repeat_dfa_with_cache(expr, min, max, None)
}

fn build_bounded_repeat_dfa_from_base(base_dfa: &DFA, min: usize, max: usize) -> Option<DFA> {
    if min > max {
        return None;
    }

    let base_states = base_dfa.states();
    let base_state_count = base_states.len();
    let layers = max.checked_add(1)?;
    let total_states = layers.checked_mul(base_state_count)?;
    u32::try_from(total_states).ok()?;
    let productive = productive_dfa_states(base_dfa);
    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(1);

    for copies_done in 0..=max {
        for (state_id, state) in base_states.iter().enumerate() {
            let mapped_state = (copies_done * base_state_count + state_id) as u32;
            let mut finalizers = crate::ds::bitset::BitSet::new(1);
            let mut future = crate::ds::bitset::BitSet::new(1);
            if state_id == 0 && copies_done >= min {
                finalizers.set(0);
            }
            if copies_done < max && productive[state_id] {
                future.set(0);
            }
            dfa.overwrite_state_metadata(mapped_state, finalizers, future);

            if copies_done == max || !base_dfa.finalizers(state_id as u32).is_empty() {
                continue;
            }

            let mut transitions = Vec::with_capacity(state.transitions.len());
            for (byte, &target) in state.transitions.iter() {
                let mapped_target = if !base_dfa.finalizers(target).is_empty() {
                    ((copies_done + 1) * base_state_count) as u32
                } else {
                    (copies_done * base_state_count + target as usize) as u32
                };
                transitions.push((byte, mapped_target));
            }
            dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
        }
    }

    Some(dfa)
}

/// Collects all bytes from a slice of suffix expressions that are all U8Seq.
/// Returns None if any expression is not a simple byte sequence.
fn collect_suffix_bytes(exprs: &[Expr]) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    for expr in exprs {
        match expr {
            Expr::U8Seq(b) => bytes.extend_from_slice(b),
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::U8Seq(b) => bytes.extend_from_slice(b),
                _ => return None,
            },
            _ => return None,
        }
    }
    if bytes.is_empty() { None } else { Some(bytes) }
}

/// Builds a DFA for `Seq([Repeat{expr, min, max}, suffix_bytes...])` directly,
/// avoiding NFA→DFA determinization. Works when the first suffix byte does not
/// overlap with the repeat expression's start-state transitions (e.g., closing
/// quote `"` after JSON string chars that exclude `"`).
fn build_bounded_repeat_with_suffix_dfa_with_cache(
    parts: &[Expr],
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    if parts.len() < 2 {
        return None;
    }

    // Extract repeat parameters, unwrapping Shared if needed.
    let first = match &parts[0] {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    let (repeat_expr, min, max) = match first {
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => (expr.as_ref(), *min, *max),
        _ => return None,
    };

    let suffix_bytes = collect_suffix_bytes(&parts[1..])?;
    let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(repeat_expr, cache)?;

    let base_states = base_dfa.states();
    let base_state_count = base_states.len();
    let layers = max.checked_add(1)?;
    let repeat_state_count = layers.checked_mul(base_state_count)?;
    let suffix_len = suffix_bytes.len();
    let total_states = repeat_state_count.checked_add(suffix_len)?;
    u32::try_from(total_states).ok()?;
    let productive = productive_dfa_states(&base_dfa);

    // The suffix boundary is ambiguous only when its first byte can begin a
    // body word. A syntactic start transition into an unproductive residual is
    // irrelevant to the body language and may be discarded below.
    if base_states[0]
        .transitions
        .get(suffix_bytes[0])
        .is_some_and(|&target| productive[target as usize])
    {
        return None;
    }

    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(1);
    let first_suffix_state = repeat_state_count as u32;

    for copies_done in 0..=max {
        for (state_id, state) in base_states.iter().enumerate() {
            let mapped_state = (copies_done * base_state_count + state_id) as u32;
            // No finalizers on repeat states — only the suffix chain end finalizes.
            let finalizers = crate::ds::bitset::BitSet::new(1);
            let mut future = crate::ds::bitset::BitSet::new(1);

            let is_accepting_pos = state_id == 0 && copies_done >= min;
            if (copies_done < max && productive[state_id]) || is_accepting_pos {
                future.set(0);
            }
            dfa.overwrite_state_metadata(mapped_state, finalizers, future);

            // At max copies or at a base-DFA finalizer state: no repeat transitions,
            // but accepting positions still get the suffix entry transition.
            if copies_done == max || !base_dfa.finalizers(state_id as u32).is_empty() {
                if is_accepting_pos {
                    dfa.set_transitions_from_sorted_entries(
                        mapped_state,
                        vec![(suffix_bytes[0], first_suffix_state)],
                    );
                }
                continue;
            }

            // Build transitions: repeat transitions + optional suffix entry.
            let extra = if is_accepting_pos { 1 } else { 0 };
            let mut transitions = Vec::with_capacity(state.transitions.len() + extra);
            for (byte, &target) in state.transitions.iter() {
                if !productive[target as usize] {
                    continue;
                }
                let mapped_target = if !base_dfa.finalizers(target).is_empty() {
                    ((copies_done + 1) * base_state_count) as u32
                } else {
                    (copies_done * base_state_count + target as usize) as u32
                };
                transitions.push((byte, mapped_target));
            }
            if is_accepting_pos {
                let pos = transitions.partition_point(|&(b, _)| b < suffix_bytes[0]);
                transitions.insert(pos, (suffix_bytes[0], first_suffix_state));
            }
            dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
        }
    }

    // Build suffix chain: each state transitions on the NEXT suffix byte.
    for i in 0..suffix_len {
        let suffix_state = (repeat_state_count + i) as u32;
        if i + 1 < suffix_len {
            let next_suffix = (repeat_state_count + i + 1) as u32;
            let mut future = crate::ds::bitset::BitSet::new(1);
            future.set(0);
            dfa.overwrite_state_metadata(
                suffix_state,
                crate::ds::bitset::BitSet::new(1),
                future,
            );
            dfa.set_transitions_from_sorted_entries(
                suffix_state,
                vec![(suffix_bytes[i + 1], next_suffix)],
            );
        } else {
            // Last suffix state: finalizer, no transitions, no future.
            let mut finalizers = crate::ds::bitset::BitSet::new(1);
            finalizers.set(0);
            dfa.overwrite_state_metadata(
                suffix_state,
                finalizers,
                crate::ds::bitset::BitSet::new(1),
            );
        }
    }

    Some((dfa, false))
}

fn build_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {
    build_bounded_repeat_with_suffix_dfa_with_cache(parts, None)
}

/// TODO: replace this compact product with an exact subset construction if we
/// need to support ambiguous body/suffix boundaries or nullable suffixes in the
/// fast path. Until then, this function must return None for any case requiring
/// multiple simultaneous boundary choices.
///
/// Builds a DFA for `Seq([Repeat{body, min, max}, suffix_exprs...])` using a
/// product construction of body_DFA × suffix_DFA × completion_counter.
///
/// Handles cases where the suffix is a regex (not just bytes) and/or the body
/// is not prefix-free, which `build_bounded_repeat_with_suffix_dfa` cannot handle.
/// Avoids the exponential NFA→DFA blowup that occurs with unrolled bounded repeats.
///
/// The product state is `(body_state, suffix_state, counter)`:
///   - body tracks progress through the repeat body expression
///   - suffix tracks the suffix match (started at body boundaries when counter >= min)
///   - counter tracks completed body repetitions (0..max)
///
/// This is a fast path, not a general NFA subset construction. It must return
/// `None` whenever the compact state would need to represent multiple live
/// body/suffix boundary choices.
///
/// At body completion, the counter increments and the suffix may start. If two
/// live paths cannot be represented by one `(body_state, suffix_state, counter)`
/// tuple, this function falls back to the general compiler.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ZeroMinRepeatSuffixState {
    /// Minimum completed-copy count reaching each body DFA residual. A smaller
    /// count dominates every larger count at the same residual.
    body_min_counts: Box<[u32]>,
    /// Exact live suffix residuals. Count is irrelevant once the suffix starts.
    suffix_states: Box<[u32]>,
}

fn close_zero_min_repeat_suffix_state(
    body_min_counts: &mut [u32],
    suffix_states: &mut Vec<u32>,
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
) {
    let mut completed_boundary = u32::MAX;
    for (body_state, &completed) in body_min_counts.iter().enumerate() {
        if completed == u32::MAX {
            continue;
        }
        if body_dfa.finalizers(body_state as u32).contains(0) {
            completed_boundary = completed_boundary.min(completed.saturating_add(1));
        }
    }

    if completed_boundary != u32::MAX {
        suffix_states.push(0);
        if completed_boundary < max as u32 {
            body_min_counts[0] = body_min_counts[0].min(completed_boundary);
        }
    }

    // Keep only residuals that can consume at least one more byte toward a
    // body match. Acceptance at the current body state has already spawned the
    // repeat/suffix boundary above.
    for (body_state, completed) in body_min_counts.iter_mut().enumerate() {
        if *completed != u32::MAX
            && !body_dfa
                .possible_future_group_ids(body_state as u32)
                .contains(0)
        {
            *completed = u32::MAX;
        }
    }

    suffix_states.retain(|&state| {
        suffix_dfa.finalizers(state).contains(0)
            || suffix_dfa.possible_future_group_ids(state).contains(0)
    });
    suffix_states.sort_unstable();
    suffix_states.dedup();
}

fn set_zero_min_repeat_suffix_metadata(
    dfa: &mut DFA,
    state_id: u32,
    state: &ZeroMinRepeatSuffixState,
    suffix_dfa: &DFA,
) {
    let mut finalizers = BitSet::new(1);
    if state
        .suffix_states
        .iter()
        .any(|&suffix_state| suffix_dfa.finalizers(suffix_state).contains(0))
    {
        finalizers.set(0);
    }
    let mut future = BitSet::new(1);
    if state.body_min_counts.iter().any(|&count| count != u32::MAX)
        || state
            .suffix_states
            .iter()
            .any(|&suffix_state| {
                suffix_dfa
                    .possible_future_group_ids(suffix_state)
                    .contains(0)
            })
    {
        future.set(0);
    }
    dfa.overwrite_state_metadata(state_id, finalizers, future);
}

struct ZeroMinRepeatSuffixBuild {
    dfa: DFA,
    states: Vec<ZeroMinRepeatSuffixState>,
    state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
}

/// Lazily interns the exact dominance states for a zero-minimum bounded
/// repeat followed by a non-nullable suffix. Product construction can then
/// visit only component residuals that survive the other intersection
/// coordinate instead of materializing every repeat-count layer first.
struct LazyZeroMinRepeatSuffixComponent {
    prefix: Vec<u8>,
    body_dfa: DFA,
    suffix_dfa: DFA,
    max: usize,
    group_u8set: U8Set,
    tail_states: Vec<ZeroMinRepeatSuffixState>,
    tail_state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
    class_count: usize,
    class_targets: Vec<u32>,
}

const LAZY_ZERO_MIN_REPEAT_SUFFIX_MIN_BOUND: usize = 1_024;

impl LazyZeroMinRepeatSuffixComponent {
    fn from_expr(expr: &Expr) -> Option<Self> {
        let unwrapped = unwrap_shared(expr);
        let Expr::Seq(parts) = unwrapped else {
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
            // The lazy product is intended for very large bounded residuals
            // whose eager constituent DFA would dominate compilation. Smaller
            // repeats are already cheap to materialize and preserve better
            // downstream state locality through the ordinary product path.
            if *max < LAZY_ZERO_MIN_REPEAT_SUFFIX_MIN_BOUND {
                continue;
            }
            // `u32::MAX` is the unreachable sentinel in body_min_counts, so it
            // cannot also be a real completed-copy count. Reject it (and any
            // larger usize bound) before compiling either body or suffix.
            if *max >= u32::MAX as usize {
                return None;
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

            let mut start_body = vec![u32::MAX; body_dfa.num_states()];
            start_body[0] = 0;
            let mut start_suffix = vec![0u32];
            close_zero_min_repeat_suffix_state(
                &mut start_body,
                &mut start_suffix,
                &body_dfa,
                &suffix_dfa,
                *max,
            );
            let start = ZeroMinRepeatSuffixState {
                body_min_counts: start_body.into_boxed_slice(),
                suffix_states: start_suffix.into_boxed_slice(),
            };
            let mut tail_state_by_key = FxHashMap::default();
            tail_state_by_key.insert(start.clone(), 0);
            return Some(Self {
                prefix,
                body_dfa,
                suffix_dfa,
                max: *max,
                group_u8set: expr_u8set(expr),
                tail_states: vec![start],
                tail_state_by_key,
                class_count: 0,
                class_targets: Vec::new(),
            });
        }
        None
    }

    fn prefix_len(&self) -> usize {
        self.prefix.len()
    }

    fn num_states(&self) -> usize {
        self.prefix.len() + self.tail_states.len()
    }

    fn start_state(&self) -> u32 {
        0
    }

    fn prepare_classes(&mut self, class_count: usize) {
        debug_assert_eq!(self.class_count, 0);
        debug_assert!(self.class_targets.is_empty());
        self.class_count = class_count;
        self.class_targets = vec![u32::MAX; self.num_states().saturating_mul(class_count)];
    }

    fn is_accepting(&self, state: u32) -> bool {
        let Some(tail) = (state as usize)
            .checked_sub(self.prefix.len())
            .and_then(|index| self.tail_states.get(index))
        else {
            return false;
        };
        tail.suffix_states
            .iter()
            .any(|&suffix_state| self.suffix_dfa.finalizers(suffix_state).contains(0))
    }

    fn has_future(&self, state: u32) -> bool {
        if (state as usize) < self.prefix.len() {
            return true;
        }
        let tail = &self.tail_states[state as usize - self.prefix.len()];
        tail.body_min_counts.iter().any(|&count| count != u32::MAX)
            || tail.suffix_states.iter().any(|&suffix_state| {
                self.suffix_dfa
                    .possible_future_group_ids(suffix_state)
                    .contains(0)
            })
    }

    fn step_class(&mut self, state: u32, class: u8, byte: u8) -> Option<u32> {
        let cache_index = (state as usize)
            .checked_mul(self.class_count)?
            .checked_add(class as usize)?;
        let cached = *self.class_targets.get(cache_index)?;
        // `u32::MAX - 1` is the cached dead-transition sentinel. The real DFA
        // state space is bounded far below it by the surrounding product.
        const DEAD: u32 = u32::MAX - 1;
        if cached == DEAD {
            return None;
        }
        if cached != u32::MAX {
            return Some(cached);
        }
        let target = self.step_uncached(state, byte).unwrap_or(DEAD);
        self.class_targets[cache_index] = target;
        if target == DEAD {
            None
        } else {
            Some(target)
        }
    }

    fn step_uncached(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state = state as usize;
        if state < self.prefix.len() {
            if self.prefix[state] != byte {
                return None;
            }
            return Some((state + 1) as u32);
        }

        let tail_index = state - self.prefix.len();
        let next = {
            let current = self.tail_states.get(tail_index)?;
            let mut next_body = vec![u32::MAX; self.body_dfa.num_states()];
            for (body_state, &completed) in current.body_min_counts.iter().enumerate() {
                if completed == u32::MAX {
                    continue;
                }
                if let Some(target) = self.body_dfa.step(body_state as u32, byte) {
                    let target_count = &mut next_body[target as usize];
                    *target_count = (*target_count).min(completed);
                }
            }
            let mut next_suffix = Vec::with_capacity(current.suffix_states.len());
            for &suffix_state in current.suffix_states.iter() {
                if let Some(target) = self.suffix_dfa.step(suffix_state, byte) {
                    next_suffix.push(target);
                }
            }
            close_zero_min_repeat_suffix_state(
                &mut next_body,
                &mut next_suffix,
                &self.body_dfa,
                &self.suffix_dfa,
                self.max,
            );
            if next_body.iter().all(|&count| count == u32::MAX) && next_suffix.is_empty() {
                return None;
            }
            ZeroMinRepeatSuffixState {
                body_min_counts: next_body.into_boxed_slice(),
                suffix_states: next_suffix.into_boxed_slice(),
            }
        };
        let local = if let Some(&local) = self.tail_state_by_key.get(&next) {
            local
        } else {
            let local = u32::try_from(self.tail_states.len()).ok()?;
            self.tail_state_by_key.insert(next.clone(), local);
            self.tail_states.push(next);
            self.class_targets
                .extend(std::iter::repeat_n(u32::MAX, self.class_count));
            local
        };
        Some(self.prefix.len() as u32 + local)
    }

    fn into_product_component(
        self,
        class_members: &[Vec<u8>],
    ) -> ProductComponent {
        let mut dfa = DFA::new(self.num_states());
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, self.group_u8set);
        for state in 0..self.num_states() {
            let mut finalizers = BitSet::new(1);
            if self.is_accepting(state as u32) {
                finalizers.set(0);
            }
            let mut future = BitSet::new(1);
            if self.has_future(state as u32) {
                future.set(0);
            }
            dfa.overwrite_state_metadata(state as u32, finalizers, future);
            let mut transitions = Vec::new();
            let row_start = state * self.class_count;
            for class in 0..self.class_count {
                let target = self.class_targets[row_start + class];
                if target >= u32::MAX - 1 {
                    continue;
                }
                transitions.extend(
                    class_members[class]
                        .iter()
                        .copied()
                        .map(|byte| (byte, target)),
                );
            }
            transitions.sort_unstable_by_key(|entry| entry.0);
            dfa.set_transitions_from_sorted_entries(state as u32, transitions);
        }
        let dfa = Arc::new(dfa);
        let trace = ZeroMinRepeatSuffixComponentTrace {
            dfa: Arc::clone(&dfa),
            prefix_len: self.prefix.len(),
            body_dfa: self.body_dfa,
            suffix_dfa: self.suffix_dfa,
            max: self.max,
            tail_states: self.tail_states,
            tail_state_by_key: self.tail_state_by_key,
        };
        ProductComponent::MaterializedZeroMinRepeatSuffix {
            dfa,
            trace: Arc::new(trace),
        }
    }
}

fn compute_dfa_byte_equivalence_classes(dfas: &[&DFA]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for dfa in dfas {
        for state in dfa.states() {
            let mut bytes_by_target = FxHashMap::<u32, U8Set>::default();
            for (byte, &target) in state.transitions.iter() {
                bytes_by_target
                    .entry(target)
                    .and_modify(|set| {
                        set.insert(byte);
                    })
                    .or_insert_with(|| U8Set::single(byte));
            }
            for byte_set in bytes_by_target.into_values() {
                if seen_sets.insert(byte_set) {
                    partitions = refine_u8_partitions(partitions, byte_set);
                }
            }
        }
    }

    let mut class_map = vec![0u8; 256];
    let mut class_members = vec![Vec::new(); partitions.len()];
    for (class_id, partition) in partitions.iter().enumerate() {
        for byte in partition.iter() {
            class_map[byte as usize] = class_id as u8;
            class_members[class_id].push(byte);
        }
    }
    (class_map, class_members)
}

fn compute_lazy_zero_min_repeat_product_equivalence_classes(
    lazy: &LazyZeroMinRepeatSuffixComponent,
    other: &DFA,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let (_, class_members) = compute_dfa_byte_equivalence_classes(&[
        &lazy.body_dfa,
        &lazy.suffix_dfa,
        other,
    ]);
    let mut partitions = class_members
        .into_iter()
        .map(|bytes| {
            let mut set = U8Set::empty();
            for byte in bytes {
                set.insert(byte);
            }
            set
        })
        .collect::<Vec<_>>();
    for &byte in &lazy.prefix {
        partitions = refine_u8_partitions(partitions, U8Set::single(byte));
    }

    let mut class_map = vec![0u8; 256];
    let mut class_members = vec![Vec::new(); partitions.len()];
    for (class_id, partition) in partitions.iter().enumerate() {
        for byte in partition.iter() {
            class_map[byte as usize] = class_id as u8;
            class_members[class_id].push(byte);
        }
    }
    (class_map, class_members)
}

fn build_zero_min_repeat_suffix_dominance_dfa_internal(
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
    preserve_coordinates: bool,
) -> Option<ZeroMinRepeatSuffixBuild> {
    if max == 0
        || body_dfa.num_states() == 0
        || suffix_dfa.num_states() == 0
        || body_dfa.finalizers(0).contains(0)
        || suffix_dfa.finalizers(0).contains(0)
    {
        return None;
    }

    let mut start_body = vec![u32::MAX; body_dfa.num_states()];
    start_body[0] = 0;
    let mut start_suffix = vec![0u32];
    close_zero_min_repeat_suffix_state(
        &mut start_body,
        &mut start_suffix,
        body_dfa,
        suffix_dfa,
        max,
    );
    let start = ZeroMinRepeatSuffixState {
        body_min_counts: start_body.into_boxed_slice(),
        suffix_states: start_suffix.into_boxed_slice(),
    };

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    set_zero_min_repeat_suffix_metadata(&mut dfa, 0, &start, suffix_dfa);
    let mut state_map = FxHashMap::<ZeroMinRepeatSuffixState, u32>::default();
    state_map.insert(start.clone(), 0);
    let mut states = vec![start.clone()];
    let mut worklist = VecDeque::from([(0u32, start)]);
    let (byte_to_class, class_members) =
        compute_dfa_byte_equivalence_classes(&[body_dfa, suffix_dfa]);
    // Dominance states are dense only as an externally visible coordinate.
    // In practice their live body frontier is tiny (typically one or two
    // residuals), while the body DFA can have dozens or hundreds of states.
    // Reuse one dense scratch row and touch only live residuals while stepping;
    // materialize a boxed dense row only for a live class edge that must be
    // interned. This preserves the historical state key exactly while avoiding
    // one allocation + full body scan for every state/class attempt.
    let mut next_body_scratch = vec![u32::MAX; body_dfa.num_states()];

    while let Some((state_id, state)) = worklist.pop_front() {
        let live_body = state
            .body_min_counts
            .iter()
            .enumerate()
            .filter_map(|(body_state, &completed)| {
                (completed != u32::MAX).then_some((body_state, completed))
            })
            .collect::<SmallVec<[(usize, u32); 8]>>();
        let mut target_by_class = vec![u32::MAX; class_members.len()];
        for (class, members) in class_members.iter().enumerate() {
            let byte = members[0];
            let mut touched_body = SmallVec::<[usize; 8]>::new();
            for &(body_state, completed) in &live_body {
                if let Some(target) = body_dfa.step(body_state as u32, byte) {
                    let target = target as usize;
                    let target_count = &mut next_body_scratch[target];
                    if *target_count == u32::MAX {
                        *target_count = completed;
                        touched_body.push(target);
                    } else {
                        *target_count = (*target_count).min(completed);
                    }
                }
            }

            let mut next_suffix = SmallVec::<[u32; 4]>::new();
            for &suffix_state in state.suffix_states.iter() {
                if let Some(target) = suffix_dfa.step(suffix_state, byte) {
                    next_suffix.push(target);
                }
            }

            let mut completed_boundary = u32::MAX;
            for &body_state in &touched_body {
                if body_dfa.finalizers(body_state as u32).contains(0) {
                    completed_boundary = completed_boundary.min(
                        next_body_scratch[body_state].saturating_add(1),
                    );
                }
            }
            if completed_boundary != u32::MAX {
                next_suffix.push(0);
                if completed_boundary < max as u32 {
                    if next_body_scratch[0] == u32::MAX {
                        next_body_scratch[0] = completed_boundary;
                        touched_body.push(0);
                    } else {
                        next_body_scratch[0] =
                            next_body_scratch[0].min(completed_boundary);
                    }
                }
            }

            touched_body.retain(|body_state| {
                let body_state = *body_state;
                if body_dfa
                    .possible_future_group_ids(body_state as u32)
                    .contains(0)
                {
                    true
                } else {
                    next_body_scratch[body_state] = u32::MAX;
                    false
                }
            });
            next_suffix.retain(|suffix_state| {
                let suffix_state = *suffix_state;
                suffix_dfa.finalizers(suffix_state).contains(0)
                    || suffix_dfa
                        .possible_future_group_ids(suffix_state)
                        .contains(0)
            });
            next_suffix.sort_unstable();
            next_suffix.dedup();

            if touched_body.is_empty() && next_suffix.is_empty() {
                continue;
            }

            let next = ZeroMinRepeatSuffixState {
                body_min_counts: next_body_scratch.clone().into_boxed_slice(),
                suffix_states: next_suffix.into_vec().into_boxed_slice(),
            };
            for &body_state in &touched_body {
                next_body_scratch[body_state] = u32::MAX;
            }
            let target = if let Some(&target) = state_map.get(&next) {
                target
            } else {
                let target = dfa.add_state();
                set_zero_min_repeat_suffix_metadata(&mut dfa, target, &next, suffix_dfa);
                state_map.insert(next.clone(), target);
                states.push(next.clone());
                worklist.push_back((target, next));
                target
            };
            target_by_class[class] = target;
        }
        let transitions = byte_to_class
            .iter()
            .enumerate()
            .filter_map(|(byte, &class)| {
                let target = target_by_class[class as usize];
                (target != u32::MAX).then_some((byte as u8, target))
            })
            .collect();
        dfa.set_transitions_from_sorted_entries(state_id, transitions);
    }

    // Hopcroft-style minimization is counterproductive for broad direct
    // residual DFAs: a few thousand states with dense byte rows can cost
    // seconds even when almost no states merge. Downstream product/DWA
    // construction is already designed to consume unminimized deterministic
    // components. Keep minimization for compact results where it is cheap and
    // useful, but preserve the exact direct DFA as-is above that threshold.
    let transitions = dfa_transition_count(&dfa);
    if !preserve_coordinates && dfa.num_states() <= 2_048 && transitions <= 100_000 {
        dfa = dfa.minimize();
        states.clear();
        state_map.clear();
    }
    Some(ZeroMinRepeatSuffixBuild {
        dfa,
        states,
        state_by_key: state_map,
    })
}

fn build_zero_min_repeat_suffix_dominance_dfa(
    body_dfa: &DFA,
    suffix_dfa: &DFA,
    max: usize,
) -> Option<DFA> {
    build_zero_min_repeat_suffix_dominance_dfa_internal(
        body_dfa,
        suffix_dfa,
        max,
        false,
    )
    .map(|built| built.dfa)
}

fn build_bounded_repeat_with_regex_suffix_with_options_and_cache(
    parts: &[Expr],
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile_timing.then(Instant::now);
    if parts.len() < 2 {
        return None;
    }

    // Flatten one level of nested Seq: Seq([Seq([a, b]), c]) → [a, b, c]
    let flat_parts: Vec<&Expr>;
    let parts_ref: &[&Expr] = {
        let first_unwrapped = match &parts[0] {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        if let Expr::Seq(inner_parts) = first_unwrapped {
            flat_parts = inner_parts.iter().chain(parts[1..].iter()).collect();
            &flat_parts
        } else {
            flat_parts = parts.iter().collect();
            &flat_parts
        }
    };

    if parts_ref.len() < 2 {
        return None;
    }

    let first = match parts_ref[0] {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    let (repeat_expr, min, max) = match first {
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => (expr.as_ref(), *min, *max),
        _ => return None,
    };

    // With max == 0 the body is not allowed to consume anything. The compact
    // product construction starts with a live body state, so it would otherwise
    // permit one body occurrence before the suffix. Let the general path handle
    // this exact zero-repeat case.
    if max == 0 {
        return None;
    }

    let body_started_at = profile_timing.then(Instant::now);
    let body_dfa = cache
        .and_then(|cache| cache.get(unwrap_shared(repeat_expr)))
        .map(|dfa| dfa.as_ref().clone())
        .unwrap_or_else(|| compile_expr_to_dfa(repeat_expr));
    let body_ms = body_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if body_dfa.num_states() == 0 || !body_dfa.finalizers(0).is_empty() {
        return None;
    }

    let suffix_expr = if parts_ref.len() == 2 {
        parts_ref[1].clone()
    } else {
        Expr::Seq(parts_ref[1..].iter().map(|e| (*e).clone()).collect())
    };
    let suffix_started_at = profile_timing.then(Instant::now);
    let suffix_dfa = compile_expr_to_dfa(&suffix_expr);
    let suffix_ms = suffix_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if suffix_dfa.num_states() == 0 {
        return None;
    }

    // Nullable suffixes require considering a body/suffix boundary without
    // consuming another byte. This compact construction only starts fresh
    // suffix paths while processing a byte, so it can miss finalization at
    // end-of-input. Fall back to the general construction.
    if !suffix_dfa.finalizers(0).is_empty() {
        return None;
    }

    // For a zero-minimum repeat, paths that reach the same body residual with
    // different completed-copy counts are ordered by language inclusion. The
    // smallest count has at least as much remaining repeat budget and may take
    // every suffix boundary available to a larger count, so larger counts are
    // redundant. Track only that minimum count per body state, plus the exact
    // set of live suffix states. This is an exact quotient of the ordinary NFA
    // subset construction and handles ambiguous body/suffix boundaries without
    // materializing the enormous unrolled-repeat powerset.
    // The dense minimum-count vector makes this quotient exceptionally fast
    // for the small repeat bodies that cause large counter products, but its
    // per-state work is quadratic in a very large body DFA. Large-body,
    // low-repeat wrappers are better served by the existing compact product
    // below (or the generic fallback when genuinely ambiguous).
    const DOMINANCE_MAX_BODY_STATES: usize = 256;
    if min == 0
        && body_dfa.num_states() <= DOMINANCE_MAX_BODY_STATES
        && let Some(built) = build_zero_min_repeat_suffix_dominance_dfa_internal(
            &body_dfa,
            &suffix_dfa,
            max,
            preserve_coordinates,
        )
    {
        let dfa = built.dfa;
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] bounded_repeat_regex_suffix_dominance body_states={} suffix_states={} max={} final_states={} final_transitions={} body_ms={:.3} suffix_ms={:.3} total_ms={:.3}",
                body_dfa.num_states(),
                suffix_dfa.num_states(),
                max,
                dfa.num_states(),
                dfa_transition_count(&dfa),
                body_ms,
                suffix_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some((dfa, false));
    }

    let max_product =
        (body_dfa.num_states() + 1) * (suffix_dfa.num_states() + 1) * (max + 1);
    if max_product > 500_000 {
        return None;
    }

    let body_dead = body_dfa.num_states() as u32;
    let suffix_dead = suffix_dfa.num_states() as u32;

    let mut state_map: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, (u32, u32, u32))> = VecDeque::new();
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);

    let start_suffix = if min == 0 { 0u32 } else { suffix_dead };
    let start_key = (0u32, start_suffix, 0u32);
    state_map.insert(start_key, 0);
    worklist.push_back((0, start_key));

    {
        let is_accept = start_suffix < suffix_dead
            && !suffix_dfa.finalizers(start_suffix).is_empty();
        let mut finalizers = BitSet::new(1);
        let mut future = BitSet::new(1);
        if is_accept {
            finalizers.set(0);
        }
        future.set(0);
        dfa.overwrite_state_metadata(0, finalizers, future);
    }

    let construct_started_at = profile_timing.then(Instant::now);
    while let Some((dfa_state, (b, s, c))) = worklist.pop_front() {
        let body_is_accept = b < body_dead && !body_dfa.finalizers(b).is_empty();

        let mut transitions = Vec::new();
        for byte_val in 0u16..=255 {
            let x = byte_val as u8;

            let b_next = if b < body_dead {
                body_dfa.step(b, x).map_or(body_dead, |t| t)
            } else {
                body_dead
            };

            // If the body is accepting before this byte, there is an implicit
            // boundary here: either continue the repeated body with `x`, or
            // finish the body and let the suffix consume `x`.
            //
            // This product state tracks only one body state and one suffix
            // state. If both paths are live, keeping only the body path is a
            // lossy greedy approximation.
            //
            // Minimal counterexample:
            //
            //   ("a"+)? "a"
            //
            // on input "aa". At the second "a", the body can continue, but the
            // suffix can also start. Dropping the suffix path falsely rejects.
            if body_is_accept && b_next != body_dead {
                let new_c = c + 1;
                if new_c >= min as u32 && new_c <= max as u32 {
                    let fresh_s = suffix_dfa
                        .step(0, x)
                        .map_or(suffix_dead, |t| t);
                    if fresh_s != suffix_dead {
                        return None;
                    }
                }
            }

            let (final_b, final_s, final_c) =
                if body_is_accept && b_next == body_dead {
                    let new_c = c + 1;
                    let new_b = if new_c < max as u32 {
                        body_dfa.step(0, x).map_or(body_dead, |t| t)
                    } else {
                        body_dead
                    };
                    let old_s_next = if s < suffix_dead {
                        suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    let fresh_s = if new_c >= min as u32 {
                        suffix_dfa.step(0, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    let new_s = match (old_s_next < suffix_dead, fresh_s < suffix_dead) {
                        (true, true) if old_s_next != fresh_s => return None,
                        (true, _) => old_s_next,
                        (_, true) => fresh_s,
                        _ => suffix_dead,
                    };
                    (new_b, new_s, new_c)
                } else {
                    let s_next = if s < suffix_dead {
                        suffix_dfa.step(s, x).map_or(suffix_dead, |t| t)
                    } else {
                        suffix_dead
                    };
                    (b_next, s_next, c)
                };

            if final_b == body_dead && final_s == suffix_dead {
                continue;
            }

            let target_key = (final_b, final_s, final_c);
            let target_dfa_state =
                if let Some(&existing) = state_map.get(&target_key) {
                    existing
                } else {
                    let new_state = dfa.add_state();
                    let accept = final_s < suffix_dead
                        && !suffix_dfa.finalizers(final_s).is_empty()
                        && final_c >= min as u32;
                    let has_future = final_b < body_dead || final_s < suffix_dead;
                    let mut finalizers = BitSet::new(1);
                    let mut future = BitSet::new(1);
                    if accept {
                        finalizers.set(0);
                    }
                    if has_future {
                        future.set(0);
                    }
                    dfa.overwrite_state_metadata(new_state, finalizers, future);
                    state_map.insert(target_key, new_state);
                    worklist.push_back((new_state, target_key));
                    new_state
                };

            transitions.push((x, target_dfa_state));
        }

        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|e| e.0);
        }
        dfa.set_transitions_from_sorted_entries(dfa_state, transitions);
    }

    let construct_ms = construct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let pre_minimize_states = dfa.num_states();
    let pre_minimize_transitions = dfa_transition_count(&dfa);
    let minimize_started_at = profile_timing.then(Instant::now);
    let dfa = dfa.minimize();
    let minimize_ms = minimize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] bounded_repeat_regex_suffix body_states={} suffix_states={} max={} pre_minimize_states={} pre_minimize_transitions={} final_states={} final_transitions={} body_ms={:.3} suffix_ms={:.3} construct_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            body_dfa.num_states(),
            suffix_dfa.num_states(),
            max,
            pre_minimize_states,
            pre_minimize_transitions,
            dfa.num_states(),
            dfa_transition_count(&dfa),
            body_ms,
            suffix_ms,
            construct_ms,
            minimize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some((dfa, false))
}

fn build_bounded_repeat_with_regex_suffix(parts: &[Expr]) -> Option<(DFA, bool)> {
    build_bounded_repeat_with_regex_suffix_with_options(parts, false)
}

fn prepend_literal_prefix_to_dfa(prefix_bytes: &[u8], tail_dfa: DFA) -> Option<DFA> {
    if prefix_bytes.is_empty() {
        return Some(tail_dfa);
    }

    let total_states = prefix_bytes.len().checked_add(tail_dfa.num_states())?;
    let tail_offset = prefix_bytes.len() as u32;
    let mut dfa = DFA::new(total_states);
    dfa.ensure_group_capacity(tail_dfa.num_groups());

    for (i, &byte) in prefix_bytes.iter().enumerate() {
        let mut future = BitSet::new(tail_dfa.num_groups());
        if tail_dfa.num_groups() > 0 {
            future.set(0);
        }
        dfa.overwrite_state_metadata(i as u32, BitSet::new(tail_dfa.num_groups()), future);
        let target = if i + 1 == prefix_bytes.len() {
            tail_offset
        } else {
            (i + 1) as u32
        };
        dfa.set_transitions_from_sorted_entries(i as u32, vec![(byte, target)]);
    }

    for state_id in 0..tail_dfa.num_states() {
        let mapped_state = tail_offset + state_id as u32;
        dfa.overwrite_state_metadata(
            mapped_state,
            tail_dfa.finalizers(state_id as u32).clone(),
            tail_dfa.possible_future_group_ids(state_id as u32).clone(),
        );
        let transitions = tail_dfa.states()[state_id]
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, tail_offset + target))
            .collect();
        dfa.set_transitions_from_sorted_entries(mapped_state, transitions);
    }

    Some(dfa)
}

fn build_bounded_repeat_with_regex_suffix_with_options(
    parts: &[Expr],
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
    build_bounded_repeat_with_regex_suffix_with_options_and_cache(
        parts,
        preserve_coordinates,
        None,
    )
}

fn add_disjoint_literal_alternative_from_start(
    dfa: &mut DFA,
    literal: &[u8],
) -> Option<()> {
    if literal.is_empty() {
        mark_state_accepting(dfa, 0);
        return Some(());
    }
    // This helper deliberately handles only a branch whose first byte is not
    // already live from the existing start state.  Under that proof the new
    // literal path is disjoint after its first byte, so grafting a private
    // chain is exactly a DFA union and needs no determinization.
    if dfa.step(0, literal[0]).is_some() {
        return None;
    }
    dfa.ensure_group_capacity(1);
    let mut source = 0u32;
    for (index, &byte) in literal.iter().enumerate() {
        let target = dfa.add_state();
        let is_last = index + 1 == literal.len();
        let mut finalizers = BitSet::new(1);
        let mut futures = BitSet::new(1);
        if is_last {
            finalizers.set(0);
        } else {
            futures.set(0);
        }
        dfa.overwrite_state_metadata(target, finalizers, futures);
        dfa.add_transition(source, byte, target);
        source = target;
    }
    Some(())
}

fn build_prefixed_bounded_repeat_with_suffix_dfa_with_options_and_cache(
    parts: &[Expr],
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    let mut flat_parts = Vec::new();
    for part in parts {
        match part {
            Expr::Shared(inner) => match inner.as_ref() {
                Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter().cloned()),
                _ => flat_parts.push(part.clone()),
            },
            Expr::Seq(inner_parts) => flat_parts.extend(inner_parts.iter().cloned()),
            _ => flat_parts.push(part.clone()),
        }
    }

    let parts = flat_parts.as_slice();
    if parts.len() < 2 {
        return None;
    }

    // A literal prefix followed directly by a bounded repeat is the same
    // structural family as the suffix-bearing forms below, but it has no
    // boundary ambiguity to resolve. Compile the repeat directly and prepend
    // the fixed bytes instead of falling back through an unrolled NFA.
    let final_part = match parts.last()? {
        Expr::Shared(inner) => inner.as_ref(),
        other => other,
    };
    if let Expr::Repeat {
        expr,
        min,
        max: Some(max),
    } = final_part
    {
        let prefix_bytes = collect_suffix_bytes(&parts[..parts.len() - 1])?;
        let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(expr, cache)?;
        let tail_dfa = build_bounded_repeat_dfa_from_base(base_dfa.as_ref(), *min, *max)?;
        return prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)
            .map(|dfa| (dfa, false));
    }

    for repeat_index in 1..parts.len() - 1 {
        let repeat_expr = match &parts[repeat_index] {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        let Expr::Repeat { .. } = repeat_expr else {
            continue;
        };

        let prefix_bytes = collect_suffix_bytes(&parts[..repeat_index])?;
        let tail_parts: Vec<Expr> = parts[repeat_index..].to_vec();
        let (tail_dfa, needs_future_recompute) =
            build_bounded_repeat_with_suffix_dfa_with_cache(&tail_parts, cache)
                .or_else(|| {
                    build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                        &tail_parts,
                        preserve_coordinates,
                        cache,
                    )
                })?;
        let dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
        return Some((dfa, needs_future_recompute));
    }

    // Fixed literal prefix + optional repeat tail + fixed literal suffix.
    // JSON property terminals frequently lower to this shape.  Compile the
    // non-empty arm with the existing bounded-repeat fast path, then graft the
    // epsilon arm's literal suffix directly when its first byte is disjoint
    // from the non-empty arm at the tail start.  The byte-disjointness check is
    // a complete determinism proof for this representation rewrite; otherwise
    // fail closed to the generic compiler.
    for optional_index in 1..parts.len().saturating_sub(1) {
        let Some(mut tail_parts) = optional_tail_parts(&parts[optional_index]) else {
            continue;
        };
        if tail_parts.len() < 2 {
            continue;
        }
        let Some(prefix_bytes) = collect_suffix_bytes(&parts[..optional_index]) else {
            continue;
        };
        let Some(suffix_bytes) = collect_suffix_bytes(&parts[optional_index + 1..]) else {
            continue;
        };
        tail_parts.extend_from_slice(&parts[optional_index + 1..]);
        let Some((mut tail_dfa, needs_future_recompute)) =
            build_bounded_repeat_with_suffix_dfa_with_cache(&tail_parts, cache)
                .or_else(|| {
                    build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                        &tail_parts,
                        preserve_coordinates,
                        cache,
                    )
                })
        else {
            continue;
        };
        if add_disjoint_literal_alternative_from_start(&mut tail_dfa, &suffix_bytes).is_none() {
            continue;
        }
        let dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
        return Some((dfa, needs_future_recompute));
    }

    if parts.len() == 2 {
        let prefix_bytes = collect_suffix_bytes(&parts[..1])?;
        let tail_parts = optional_tail_parts(&parts[1])?;
        if tail_parts.len() >= 2 {
            let (tail_dfa, needs_future_recompute) =
                build_bounded_repeat_with_suffix_dfa_with_cache(&tail_parts, cache)
                    .or_else(|| {
                        build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                            &tail_parts,
                            preserve_coordinates,
                            cache,
                        )
                    })?;
            let mut dfa = prepend_literal_prefix_to_dfa(&prefix_bytes, tail_dfa)?;
            mark_state_accepting(&mut dfa, prefix_bytes.len() as u32);
            return Some((dfa, needs_future_recompute));
        }
    }

    None
}

fn build_prefixed_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {
    build_prefixed_bounded_repeat_with_suffix_dfa_with_options(parts, false)
}

fn append_bounded_repeat_expr(expr: &Expr, min: usize, max: usize, nfa: &mut NFA, start: u32, end: u32) {
    if max < min {
        return;
    }

    if let Some(dfa) = build_bounded_repeat_dfa(expr, min, max) {
        append_dfa_expr(&dfa, nfa, start, end);
        return;
    }

    let mut repeat_compiler = RepeatCompiler::new(expr, nfa);
    let optional = max - min;
    let tail_start = repeat_compiler.compile_upto(optional, end);
    let repeat_start = repeat_compiler.compile_exact(min, tail_start);
    repeat_compiler.nfa.add_epsilon(start, repeat_start);
}

fn append_unbounded_repeat_expr(
    expr: &Expr,
    min: usize,
    nfa: &mut NFA,
    start: u32,
    end: u32,
) {
    let mut current = start;
    for _ in 0..min {
        let next = nfa.add_state();
        append_compiled_expr(expr, nfa, current, next);
        current = next;
    }

    if current == start {
        let fresh = nfa.add_state();
        nfa.add_epsilon(start, fresh);
        current = fresh;
    }

    nfa.add_epsilon(current, end);
    let loop_state = nfa.add_state();
    append_compiled_expr(expr, nfa, current, loop_state);
    nfa.add_epsilon(loop_state, current);
    if expr_accepts_empty(expr) {
        nfa.add_epsilon(loop_state, end);
    }
}

fn append_compiled_expr(expr: &Expr, nfa: &mut NFA, start: u32, end: u32) {
    match expr {
        Expr::U8Seq(bytes) => append_byte_sequence_expr(bytes, nfa, start, end),
        Expr::U8Class(set) => {
            nfa.add_u8set_transition(start, *set, end);
        }
        Expr::Dfa(dfa) => append_dfa_expr(dfa, nfa, start, end),
        Expr::Intersect { .. } => {
            unreachable!("nested Expr::Intersect must be lowered before NFA compilation")
        }
        Expr::Seq(parts) => append_sequence_expr(parts, nfa, start, end),
        Expr::Choice(options) => append_choice_expr(options, nfa, start, end),
        Expr::Exclude { .. } => {
            unreachable!("nested Expr::Exclude must be lowered before NFA compilation")
        }
        Expr::Repeat { expr, min, max } => match max {
            Some(max) => append_bounded_repeat_expr(expr, *min, *max, nfa, start, end),
            None => append_unbounded_repeat_expr(expr, *min, nfa, start, end),
        },
        Expr::Shared(inner) => append_compiled_expr(inner, nfa, start, end),
        Expr::Epsilon => nfa.add_epsilon(start, end),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regex {
    pub(super) dfa: DFA,
}

impl Regex {
    pub fn into_tokenizer(self, num_terminals: u32, exprs: Option<std::sync::Arc<[Expr]>>) -> Tokenizer {
        Tokenizer {
            dfa: self.dfa,
            num_terminals,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs,
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        }
    }

    pub fn num_states(&self) -> usize {
        self.dfa.num_states()
    }

    pub fn num_transitions(&self) -> usize {
        dfa_transition_count(&self.dfa)
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    pub fn get_u8set(&self, state: u32) -> U8Set {
        self.dfa.get_u8set(state)
    }
}

fn dfa_transition_count(dfa: &DFA) -> usize {
    dfa.transition_count()
}

impl Expr {
    pub fn build(self) -> Regex {
        build_regex(&[self])
    }
}

/// Compile multiple expressions into a single multi-group [`Regex`].
///
/// Each expression's index becomes its group ID in the resulting DFA.
fn compile_single_expr_dfa(expr: &Expr) -> DFA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let direct_started_at = profile_timing.then(Instant::now);
    if let Some((mut dfa, needs_future_recompute)) = compile_product_component_dfa_direct(expr) {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        if needs_future_recompute {
            dfa.recompute_possible_futures();
        }
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] single_expr_path path=direct states={} transitions={} direct_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
            );
        }
        return dfa;
    }

    let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let nfa_started_at = profile_timing.then(Instant::now);
    let mut nfa = build_regex_nfa(std::slice::from_ref(expr));
    let nfa_ms = nfa_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let condense_started_at = profile_timing.then(Instant::now);
    nfa.condense_epsilon_sccs();
    let condense_ms =
        condense_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let determinize_started_at = profile_timing.then(Instant::now);
    // Minimize over the NFA's byte-equivalence classes before expanding the
    // surviving transitions back to bytes.  Large JSON string/property regexes
    // can determinize to thousands of states with ~150k byte transitions only
    // to collapse to a few hundred states immediately afterward.  The class
    // alphabet path is language-equivalent and avoids materializing that large
    // transient byte graph.
    let dfa = nfa.to_minimized_dfa();
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] single_expr_path path=nfa_dfa states={} transitions={} direct_attempt_ms={:.3} nfa_ms={:.3} condense_ms={:.3} determinize_ms={:.3}",
            dfa.num_states(),
            dfa_transition_count(&dfa),
            direct_ms,
            nfa_ms,
            condense_ms,
            determinize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    dfa
}

fn compile_with_plan(plan: ExclusionCompilePlan) -> DFA {
    compile_with_plan_internal(plan, false).0
}

fn compile_with_plan_internal(
    plan: ExclusionCompilePlan,
    capture_product_trace: bool,
) -> (DFA, Option<ProductBuildTrace>) {
    compile_with_plan_internal_options(plan, capture_product_trace, true, false)
}

fn compile_with_plan_internal_options(
    plan: ExclusionCompilePlan,
    capture_product_trace: bool,
    virtual_fixed_sequences: bool,
    collapse_traced_duplicate_coordinates: bool,
) -> (DFA, Option<ProductBuildTrace>) {
    let profile_trace = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some();
    let profile_detail = profile_trace
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    // Product construction compiles many one-group component DFAs internally.
    // Timing/detail mode should describe the enclosing lexer build, not emit a
    // line (and take timestamps) for every nested leaf compile. Exhaustive trace
    // mode remains available when that low-level view is explicitly requested.
    let profile_plan = profile_trace || (profile_timing && plan.compiled_exprs.len() > 1);
    let profile_started_at = Instant::now();
    let group_set_started_at = Instant::now();
    let group_sets: Vec<U8Set> = plan
        .compiled_exprs
        .iter()
        .map(expr_u8set)
        .collect();
    let group_set_ms = profile_plan
        .then(|| group_set_started_at.elapsed().as_secs_f64() * 1000.0);
    let used_product_dfa = plan.compiled_exprs.len() > 1;
    let dfa_build_started_at = Instant::now();
    let (mut dfa, product_group_ops_applied, mut product_trace) = if plan.compiled_exprs.is_empty() {
        // A grammar can lower to the empty language, for example when a const
        // literal conflicts with sibling assertions. Keep a single non-final
        // start state so tokenizer users can still query/step the DFA safely.
        (DFA::new(1), false, None)
    } else if used_product_dfa {
        build_product_dfa(
            &plan.compiled_exprs,
            plan.profile_labels.as_deref(),
            plan.visible_groups,
            &plan.exclusions,
            &plan.intersections,
            capture_product_trace,
            virtual_fixed_sequences,
            plan.local_small_product,
            collapse_traced_duplicate_coordinates,
        )
    } else {
        (compile_single_expr_dfa(&plan.compiled_exprs[0]), false, None)
    };
    let dfa_build_ms = profile_plan
        .then(|| dfa_build_started_at.elapsed().as_secs_f64() * 1000.0);

    let metadata_started_at = Instant::now();
    let output_group_count = if product_group_ops_applied {
        plan.visible_groups
    } else {
        group_sets.len()
    };
    dfa.ensure_group_capacity(output_group_count);
    for (group_id, set) in group_sets.into_iter().take(output_group_count).enumerate() {
        dfa.set_group_u8set(group_id as u32, set);
    }
    let metadata_ms = profile_plan
        .then(|| metadata_started_at.elapsed().as_secs_f64() * 1000.0);

    let group_ops_started_at = Instant::now();
    let mut group_ops_changed = false;
    if !product_group_ops_applied && !plan.exclusions.is_empty() {
        group_ops_changed |= dfa.apply_group_exclusions(&plan.exclusions);
    }
    if !product_group_ops_applied && !plan.intersections.is_empty() {
        group_ops_changed |= dfa.apply_group_intersections(&plan.intersections);
    }
    let project_in_place = !product_group_ops_applied
        && plan.visible_groups < plan.compiled_exprs.len()
        && std::env::var_os("GLRMASK_DISABLE_IN_PLACE_GROUP_PROJECTION").is_none();
    if group_ops_changed && !project_in_place {
        dfa.recompute_possible_futures();
    }
    let group_ops_ms = profile_plan
        .then(|| group_ops_started_at.elapsed().as_secs_f64() * 1000.0);

    let project_started_at = Instant::now();
    let dfa = if project_in_place {
        dfa.project_groups_in_place(plan.visible_groups);
        if group_ops_changed {
            // Reachability commutes with projection: after applying hidden
            // exclusion/intersection predicates to finalizers, recomputing the
            // visible futures directly is equivalent to recomputing all groups
            // and then projecting, while using narrower bitsets throughout.
            dfa.recompute_possible_futures();
        }
        dfa
    } else if !product_group_ops_applied && plan.visible_groups < plan.compiled_exprs.len() {
        dfa.project_groups(plan.visible_groups)
    } else {
        dfa
    };
    let project_ms = profile_plan
        .then(|| project_started_at.elapsed().as_secs_f64() * 1000.0);

    let pre_minimize_states = profile_plan.then(|| dfa.num_states());
    let pre_minimize_transitions = profile_plan.then(|| dfa_transition_count(&dfa));
    let force_tokenizer_minimize = std::env::var_os("GLRMASK_FORCE_TOKENIZER_MINIMIZE").is_some();
    let minimize_started_at = Instant::now();
    let final_dfa = if used_product_dfa && !force_tokenizer_minimize {
        dfa
    } else {
        product_trace = None;
        dfa.minimize()
    };
    let minimize_ms = profile_plan
        .then(|| minimize_started_at.elapsed().as_secs_f64() * 1000.0);
    if profile_detail && profile_plan {
        let minimized_states = if used_product_dfa && !force_tokenizer_minimize {
            "not_run".to_string()
        } else {
            final_dfa.num_states().to_string()
        };
        eprintln!(
            "[glrmask/profile][tokenizer] combined groups={} visible_groups={} product_dfa={} pre_minimize_states={} pre_minimize_transitions={} final_states={} final_transitions={} minimized_states={}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            pre_minimize_states.unwrap_or_default(),
            pre_minimize_transitions.unwrap_or_default(),
            final_dfa.num_states(),
            dfa_transition_count(&final_dfa),
            minimized_states,
        );
    }
    if profile_plan {
        eprintln!(
            "[glrmask/profile][tokenizer] compile_plan groups={} visible_groups={} product_dfa={} group_set_ms={:.3} dfa_build_ms={:.3} metadata_ms={:.3} group_ops_ms={:.3} project_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            plan.compiled_exprs.len(),
            plan.visible_groups,
            used_product_dfa,
            group_set_ms.unwrap_or_default(),
            dfa_build_ms.unwrap_or_default(),
            metadata_ms.unwrap_or_default(),
            group_ops_ms.unwrap_or_default(),
            project_ms.unwrap_or_default(),
            minimize_ms.unwrap_or_default(),
            profile_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (final_dfa, product_trace)
}

pub fn build_regex(exprs: &[Expr]) -> Regex {
    build_regex_monolithic(exprs)
}

/// Compile a small auxiliary product lexer while keeping its fine-grained
/// construction work on the calling worker. This is intended for callers that
/// already execute inside a parallel compiler DAG, where nested Rayon fan-out
/// can otherwise turn a few milliseconds of local work into scheduler tail.
pub fn build_regex_local_small_product(exprs: &[Expr]) -> Regex {
    let mut plan = build_exclusion_compile_plan(exprs);
    plan.local_small_product = true;
    Regex {
        dfa: compile_with_plan(plan),
    }
}

/// Compile all expressions into one traditional deterministic lexer.
pub fn build_regex_monolithic(exprs: &[Expr]) -> Regex {
    Regex {
        dfa: compile_with_plan(build_exclusion_compile_plan(exprs)),
    }
}

pub fn build_regex_with_profile_labels(exprs: &[Expr], visible_labels: &[String]) -> Regex {
    Regex {
        dfa: compile_with_plan(build_exclusion_compile_plan_with_labels(
            exprs,
            Some(visible_labels),
        )),
    }
}

/// Compile terminals in caller-selected deterministic partitions, then join
/// those partitions with epsilon edges from one global start state. Terminals
/// sharing a partition may be jointly determinized; terminals in different
/// partitions can never cause a cross-partition subset/product blow-up.
pub fn build_regex_partitioned(exprs: &[Expr], partitions: &[u32]) -> Regex {
    build_regex_partitioned_with_adaptive(exprs, partitions, adaptive_lexer_enabled())
}

pub fn build_regex_partitioned_with_residual_isolation(
    exprs: &[Expr],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
) -> Regex {
    build_regex_partitioned_with_adaptive_and_residual_isolation(
        exprs,
        partitions,
        residual_isolation_classes,
        adaptive_lexer_enabled(),
    )
}

pub fn build_regex_partitioned_with_adaptive(
    exprs: &[Expr],
    partitions: &[u32],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(exprs, None, partitions, None, adaptive),
    }
}

pub fn build_regex_partitioned_with_adaptive_and_residual_isolation(
    exprs: &[Expr],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            None,
            partitions,
            Some(residual_isolation_classes),
            adaptive,
        ),
    }
}

pub fn build_regex_partitioned_with_profile_labels(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
) -> Regex {
    build_regex_partitioned_with_profile_labels_and_adaptive(
        exprs,
        visible_labels,
        partitions,
        adaptive_lexer_enabled(),
    )
}


pub fn build_regex_partitioned_with_profile_labels_and_residual_isolation(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
) -> Regex {
    build_regex_partitioned_with_profile_labels_and_adaptive_and_residual_isolation(
        exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive_lexer_enabled(),
    )
}

pub fn build_regex_partitioned_with_profile_labels_and_adaptive(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            Some(visible_labels),
            partitions,
            None,
            adaptive,
        ),
    }
}
pub fn build_regex_partitioned_with_profile_labels_and_adaptive_and_residual_isolation(
    exprs: &[Expr],
    visible_labels: &[String],
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
) -> Regex {
    Regex {
        dfa: compile_terminal_partitions(
            exprs,
            Some(visible_labels),
            partitions,
            Some(residual_isolation_classes),
            adaptive,
        ),
    }
}

/// Compile-time tokenizer construction from already-compiled single-terminal
/// DFAs. Besides the exact tokenizer, retain for every raw
/// tokenizer state the product coordinates `(terminal, terminal_dfa_state)`.
///
/// This deliberately bypasses adaptive component determinization: the retained
/// coordinate is the point of this path, and adaptive products would need to
/// carry the same trace explicitly rather than reconstructing it afterward.
pub fn build_partitioned_tokenizer_from_precompiled_terminal_dfas(
    exprs: &[Expr],
    partitions: &[u32],
    retained_exprs: Arc<[Expr]>,
) -> Option<Tokenizer> {
    if exprs.len() != partitions.len() || exprs.len() != retained_exprs.len() {
        return None;
    }
    let terminal_dfas = exprs
        .iter()
        .map(|expr| match expr {
            Expr::Dfa(dfa) => Some(Arc::clone(dfa)),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }

    struct ComponentWithCoordinates {
        terminal_ids: Vec<usize>,
        dfa: DFA,
        rows: Vec<Vec<(u32, u32)>>,
    }

    let components = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_, terminal_ids)| -> Option<ComponentWithCoordinates> {
            if terminal_ids.len() == 1 {
                let terminal = terminal_ids[0];
                let dfa = terminal_dfas.get(terminal)?.as_ref().clone();
                let rows = (0..dfa.num_states())
                    .map(|state| vec![(terminal as u32, state as u32)])
                    .collect();
                return Some(ComponentWithCoordinates {
                    terminal_ids,
                    dfa,
                    rows,
                });
            }

            // The traced builder historically placed every terminal in the
            // product independently. Large schema imports contain many exactly
            // identical single-terminal DFAs, so that needlessly multiplies the
            // product coordinate even though those terminals can never diverge.
            // DFA equality is exact structural equality (including group
            // metadata), making it safe to share one product coordinate and
            // fan its observation/residual state back out to every alias.
            let mut unique_dfas = Vec::<Arc<DFA>>::new();
            let mut unique_members = Vec::<Vec<(usize, usize)>>::new();
            let mut unique_by_dfa = FxHashMap::<Arc<DFA>, usize>::default();
            for (local_group, &terminal) in terminal_ids.iter().enumerate() {
                let terminal_dfa = Arc::clone(terminal_dfas.get(terminal)?);
                let unique = if let Some(&unique) = unique_by_dfa.get(&terminal_dfa) {
                    unique
                } else {
                    let unique = unique_dfas.len();
                    unique_by_dfa.insert(Arc::clone(&terminal_dfa), unique);
                    unique_dfas.push(Arc::clone(terminal_dfas.get(terminal)?));
                    unique_members.push(Vec::new());
                    unique
                };
                unique_members[unique].push((local_group, terminal));
            }
            if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                eprintln!(
                    "[glrmask/profile][precompiled_terminal_dfa_dedup] terminals={} unique={} saved={}",
                    terminal_ids.len(),
                    unique_dfas.len(),
                    terminal_ids.len().saturating_sub(unique_dfas.len()),
                );
            }
            let local_exprs = unique_dfas
                .iter()
                .map(|dfa| Expr::Dfa(Arc::clone(dfa)))
                .collect::<Vec<_>>();
            let empty_ops = BTreeMap::<u32, BTreeSet<u32>>::new();
            let (mut dfa, group_ops_applied, trace) = build_product_dfa(
                &local_exprs,
                None,
                local_exprs.len(),
                &empty_ops,
                &empty_ops,
                true,
                true,
                false,
                false,
            );
            if group_ops_applied {
                return None;
            }
            // `build_product_dfa` labels states by unique-DFA coordinate. Expand
            // those exact labels back to the original terminal-local groups.
            // This preserves the tokenizer's public terminal observations even
            // though identical terminals share transition topology.
            let compact_metadata = (0..dfa.num_states() as u32)
                .map(|state| {
                    (
                        dfa.finalizers(state).clone(),
                        dfa.possible_future_group_ids(state).clone(),
                    )
                })
                .collect::<Vec<_>>();
            dfa.ensure_group_capacity(terminal_ids.len());
            for (state, (compact_finalizers, compact_futures)) in
                compact_metadata.into_iter().enumerate()
            {
                let mut finalizers = BitSet::new(terminal_ids.len());
                for unique in compact_finalizers.iter() {
                    for &(local_group, _) in unique_members.get(unique)? {
                        finalizers.set(local_group);
                    }
                }
                let mut futures = BitSet::new(terminal_ids.len());
                for unique in compact_futures.iter() {
                    for &(local_group, _) in unique_members.get(unique)? {
                        futures.set(local_group);
                    }
                }
                dfa.overwrite_state_metadata(state as u32, finalizers, futures);
            }
            for (local_group, &terminal) in terminal_ids.iter().enumerate() {
                dfa.set_group_u8set(
                    local_group as u32,
                    *terminal_dfas[terminal].group_id_to_u8set(0),
                );
            }
            let trace = trace?;
            if trace.state_tuples.len() != dfa.num_states() {
                return None;
            }
            let mut rows = Vec::with_capacity(dfa.num_states());
            for state in 0..dfa.num_states() {
                let tuple = trace.state_tuples.tuple(state);
                let mut row = Vec::with_capacity(tuple.len());
                for (coordinate, residual_state) in tuple {
                    for &(_, terminal) in unique_members.get(coordinate as usize)? {
                        row.push((terminal as u32, residual_state));
                    }
                }
                row.sort_unstable();
                rows.push(row);
            }
            Some(ComponentWithCoordinates {
                terminal_ids,
                dfa,
                rows,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());
    let mut rows = vec![Vec::<(u32, u32)>::new()];
    for component in components {
        for local_group in component.dfa.possible_future_group_ids(0).iter() {
            root_futures.set(component.terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(component.dfa, &component.terminal_ids);
        if offset as usize != rows.len() {
            return None;
        }
        combined.add_epsilon_transition(0, offset);
        rows.extend(component.rows);
    }
    combined.set_possible_future_group_ids(0, root_futures);

    let mut tokenizer = Regex { dfa: combined }.into_tokenizer(
        exprs.len() as u32,
        Some(retained_exprs),
    );
    tokenizer.set_terminal_residual_coordinates(
        TerminalResidualCoordinates::from_rows_and_dfas(rows, terminal_dfas),
    );
    Some(tokenizer)
}

/// Build the direct-L1 sidecar from the same product components used by normal
/// partition compilation. This avoids both recompiling every terminal DFA and
/// projecting terminal residual coordinates back out of the finished tokenizer
/// after the fact.
pub fn build_partitioned_tokenizer_with_product_trace_terminal_residuals(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
    retained_exprs: Arc<[Expr]>,
    adaptive: Option<bool>,
    collapse_traced_duplicate_coordinates: bool,
) -> Option<Tokenizer> {
    if exprs.len() != partitions.len() || exprs.len() != retained_exprs.len() || exprs.is_empty() {
        return None;
    }
    if adaptive.unwrap_or_else(adaptive_lexer_enabled) {
        // Cross-partition adaptive products need their own trace composition.
        // The default lexer path is non-adaptive; fail closed for explicit
        // adaptive diagnostics rather than silently changing that topology.
        return None;
    }
    if let Some(labels) = visible_labels
        && labels.len() != exprs.len()
    {
        return None;
    }
    if let Some(classes) = residual_isolation_classes
        && classes.len() != exprs.len()
    {
        return None;
    }

    let started_at = Instant::now();
    let rewritten_exprs = materialize_repeated_subexpression_dfas(exprs);
    let exprs = rewritten_exprs.as_deref().unwrap_or(exprs);
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }
    if let Some(classes) = residual_isolation_classes {
        for terminal_ids in grouped.values() {
            let mut class = None;
            let mut has_unprotected = false;
            for &terminal in terminal_ids {
                match classes[terminal] {
                    Some(current) => match class {
                        Some(previous) if previous != current => return None,
                        Some(_) => {}
                        None => class = Some(current),
                    },
                    None => has_unprotected = true,
                }
            }
            if class.is_some() && has_unprotected {
                return None;
            }
        }
    }
    let shared_duplicates = shared_duplicate_nested_group_op_cache(exprs, &grouped);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }
    struct TracedComponent {
        terminal_ids: Vec<usize>,
        dfa: Arc<DFA>,
        rows: Vec<Vec<(u32, u32)>>,
        sources: Vec<(usize, Arc<DFA>, u32, Option<Arc<TerminalExclusionCertificate>>)>,
    }

    struct ComplexTerminalSource {
        source: Arc<DFA>,
        mapping: Vec<u32>,
        exclusion: Option<Arc<TerminalExclusionCertificate>>,
    }

    let components = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_, terminal_ids)| -> Option<TracedComponent> {
            let local_exprs = terminal_ids
                .iter()
                .map(|&terminal| exprs[terminal].clone())
                .collect::<Vec<_>>();
            let local_labels = visible_labels.map(|labels| {
                terminal_ids
                    .iter()
                    .map(|&terminal| labels[terminal].clone())
                    .collect::<Vec<_>>()
            });
            let mut nested_group_op_cache = NestedGroupOpCache {
                shared_duplicates: shared_duplicates.clone(),
                ..NestedGroupOpCache::default()
            };
            let plan = build_exclusion_compile_plan_with_labels_and_cache(
                &local_exprs,
                local_labels.as_deref(),
                &mut nested_group_op_cache,
            );
            let visible_groups = plan.visible_groups;
            if visible_groups != terminal_ids.len() {
                return None;
            }
            let mut complex = vec![false; visible_groups];
            for &group in plan.exclusions.keys().chain(plan.intersections.keys()) {
                if let Some(slot) = complex.get_mut(group as usize) {
                    *slot = true;
                }
            }
            let complex_group_count = complex.iter().filter(|&&is_complex| is_complex).count();

            let compiled_group_count = plan.compiled_exprs.len();
            // Preserve only the logical group-op wiring for the rare complex
            // visible groups. After product compilation we rebuild their
            // standalone expression from the trace's already-compiled DFA
            // coordinates, avoiding any recursive recompilation of the source
            // expressions.
            let mut complex_ops = vec![None::<(Vec<u32>, Vec<u32>)>; visible_groups];
            for local_group in 0..visible_groups {
                if !complex[local_group] {
                    continue;
                }
                let exclusions = plan
                    .exclusions
                    .get(&(local_group as u32))
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                let intersections = plan
                    .intersections
                    .get(&(local_group as u32))
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                complex_ops[local_group] = Some((exclusions, intersections));
            }

            let (dfa, trace) = compile_with_plan_internal_options(
                plan,
                true,
                true,
                collapse_traced_duplicate_coordinates,
            );
            let dfa = Arc::new(dfa);
            if visible_groups == 1 {
                let terminal = terminal_ids[0];
                let rows = (0..dfa.num_states() as u32)
                    .map(|state| {
                        (dfa.finalizers(state).contains(0)
                            || dfa.possible_future_group_ids(state).contains(0))
                            .then_some(vec![(terminal as u32, state)])
                            .unwrap_or_default()
                    })
                    .collect();
                return Some(TracedComponent {
                    terminal_ids,
                    dfa: Arc::clone(&dfa),
                    rows,
                    sources: vec![(terminal, dfa, 0, None)],
                });
            }

            let trace = trace?;
            if trace.state_tuples.len() != dfa.num_states() {
                return None;
            }
            let mut coordinate_for_logical_group = vec![usize::MAX; compiled_group_count];
            for (coordinate, groups) in trace.coordinate_groups.iter().enumerate() {
                for &group in groups {
                    if group >= compiled_group_count {
                        continue;
                    }
                    if coordinate_for_logical_group[group] != usize::MAX {
                        return None;
                    }
                    coordinate_for_logical_group[group] = coordinate;
                }
            }
            if coordinate_for_logical_group
                .iter()
                .any(|&coordinate| coordinate == usize::MAX)
            {
                return None;
            }
            // Materialize the sparse product trace once as a dense
            // state×physical-coordinate table. Complex visible terminals below
            // are themselves small products of these same coordinates, so this
            // lets us translate a partition state directly into the standalone
            // product tuple without replaying every byte edge for every terminal.
            let partition_coordinate_states = if complex_group_count == 0 {
                None
            } else {
                let mut states = vec![vec![u32::MAX; trace.components.len()]; dfa.num_states()];
                for state in 0..dfa.num_states() {
                    for (coordinate, residual_state) in trace.state_tuples.tuple(state) {
                        let slot = states.get_mut(state)?.get_mut(coordinate as usize)?;
                        *slot = residual_state;
                    }
                }
                Some(states)
            };
            let mut simple_sources = Vec::<Option<Arc<DFA>>>::with_capacity(visible_groups);
            for local_group in 0..visible_groups {
                let coordinate = coordinate_for_logical_group[local_group];
                let source = (!complex[local_group])
                    .then(|| trace.components[coordinate].terminal_residual_dfa_arc())
                    .flatten();
                simple_sources.push(source);
            }

            // A visible terminal affected by an exclusion/intersection cannot
            // use one raw product coordinate directly: its language is the
            // group operation over several coordinates. Compile only those rare
            // exceptional terminals independently, then propagate their exact
            // residual state through the already-built partition DFA. This is
            // deliberately strict: every reachable partition state must agree
            // on the standalone residual coordinate and on terminal liveness.
            // Any disagreement fails closed to the established projected path.
            let mut complex_sources = (0..visible_groups)
                .into_par_iter()
                .map(|local_group| -> Option<Option<ComplexTerminalSource>> {
                if simple_sources[local_group].is_some() {
                    return Some(None);
                }
                let (exclusions, intersections) = complex_ops[local_group].as_ref()?;
                let source_compile_started = Instant::now();
                // The already-built partition product contains every component
                // coordinate needed by this exclusion/intersection terminal.
                // Project only those coordinates, deduplicate the resulting
                // tuples, then minimize that much smaller residual DFA. This
                // avoids recompiling the group expression and avoids running
                // Hopcroft once per terminal over the whole partition DFA.
                let mut relevant_coordinates = SmallVec::<[usize; 4]>::new();
                for logical_group in std::iter::once(local_group)
                    .chain(exclusions.iter().map(|&group| group as usize))
                    .chain(intersections.iter().map(|&group| group as usize))
                {
                    let coordinate = *coordinate_for_logical_group.get(logical_group)?;
                    if !relevant_coordinates.contains(&coordinate) {
                        relevant_coordinates.push(coordinate);
                    }
                }
                relevant_coordinates.sort_unstable();

                let partition_live = |state: u32| {
                    dfa.finalizers(state).contains(local_group)
                        || dfa.possible_future_group_ids(state).contains(local_group)
                };
                if !partition_live(0) {
                    // The group operation is the empty language. Keep an
                    // exact canonical dead residual instead of discarding the
                    // successful traced partition and rebuilding via the
                    // fallback lexer.
                    let mut source = DFA::new(1);
                    source.ensure_group_capacity(1);
                    source.set_group_u8set(0, *dfa.group_id_to_u8set(local_group as u32));
                    return Some(Some(ComplexTerminalSource {
                        source: Arc::new(source),
                        mapping: vec![u32::MAX; dfa.num_states()],
                        exclusion: None,
                    }));
                }

                let mut raw_state_by_key = FxHashMap::<SmallVec<[u32; 4]>, u32>::default();
                let mut raw_representatives = Vec::<u32>::new();
                let mut partition_to_raw = vec![u32::MAX; dfa.num_states()];
                let coordinate_states = partition_coordinate_states.as_ref()?;
                for partition_state in 0..dfa.num_states() {
                    if !partition_live(partition_state as u32) {
                        continue;
                    }
                    let key = relevant_coordinates
                        .iter()
                        .map(|&coordinate| coordinate_states[partition_state][coordinate])
                        .collect::<SmallVec<[u32; 4]>>();
                    let raw_state = if let Some(&existing) = raw_state_by_key.get(&key) {
                        existing
                    } else {
                        let next = raw_representatives.len() as u32;
                        raw_state_by_key.insert(key, next);
                        raw_representatives.push(partition_state as u32);
                        next
                    };
                    partition_to_raw[partition_state] = raw_state;
                }
                if partition_to_raw[0] != 0 {
                    return None;
                }

                let mut raw = DFA::new(raw_representatives.len());
                raw.ensure_group_capacity(1);
                raw.set_group_u8set(0, *dfa.group_id_to_u8set(local_group as u32));
                for (raw_state, &representative) in raw_representatives.iter().enumerate() {
                    let mut transitions = Vec::<(u8, u32)>::new();
                    for (byte, target) in dfa.transitions(representative) {
                        let mapped = partition_to_raw[target as usize];
                        if mapped != u32::MAX {
                            transitions.push((byte, mapped));
                        }
                    }
                    raw.set_transitions_from_sorted_entries(raw_state as u32, transitions);
                    let mut finalizers = BitSet::new(1);
                    if dfa.finalizers(representative).contains(local_group) {
                        finalizers.set(0);
                    }
                    raw.overwrite_state_metadata(raw_state as u32, finalizers, BitSet::new(1));
                }
                raw.recompute_possible_futures();
                // The projection is already an exact deterministic residual DFA.
                // Minimizing it here is unnecessary for the proof sidecar and, on
                // the large exclusion families that dominate the dynamic-build
                // tail, does not remove any states. Keep the exact projection and
                // its direct partition-state mapping instead of paying another
                // Hopcroft pass for every exceptional terminal.
                let source = Arc::new(raw);
                let mapping = partition_to_raw;
                let source_compile_ms = source_compile_started.elapsed().as_secs_f64() * 1000.0;
                if source.has_epsilon_transitions() {
                    return None;
                }

                let source_live = |state: u32| {
                    source.finalizers(state).contains(0)
                        || source.possible_future_group_ids(state).contains(0)
                };
                if mapping.len() != dfa.num_states()
                    || partition_live(0) != source_live(*mapping.first()?)
                {
                    return None;
                }

                let mapping_started = Instant::now();
                for partition_state in 0..dfa.num_states() {
                    let source_state = *mapping.get(partition_state)?;
                    let partition_is_live = partition_live(partition_state as u32);
                    let source_is_live = source_state != u32::MAX && source_live(source_state);
                    if partition_is_live != source_is_live {
                        return None;
                    }
                }

                // Optional Layer-2 certificate for the narrow, common shape
                // `Exclude(left, finite_right)`. The exact product trace gives
                // us left/right residual coordinates for every representative
                // of the standalone terminal residual DFA. Failure here drops
                // only the optimization sidecar; tokenizer correctness does not
                // depend on this certificate.
                let exclusion = (|| -> Option<Arc<TerminalExclusionCertificate>> {
                    if exclusions.len() != 1 || !intersections.is_empty() {
                        return None;
                    }
                    let terminal = *terminal_ids.get(local_group)?;
                    if !matches!(retained_exprs.get(terminal)?, Expr::Exclude { .. }) {
                        return None;
                    }
                    let left_coordinate = *coordinate_for_logical_group.get(local_group)?;
                    let right_group = *exclusions.first()? as usize;
                    let right_coordinate = *coordinate_for_logical_group.get(right_group)?;
                    if left_coordinate == right_coordinate {
                        return None;
                    }
                    let left_dfa = trace.components.get(left_coordinate)?.terminal_residual_dfa_arc()?;
                    let right_dfa = trace.components.get(right_coordinate)?.terminal_residual_dfa_arc()?;
                    let right_max_remaining = single_group_dfa_max_remaining(right_dfa.as_ref())?;
                    let coordinate_states = partition_coordinate_states.as_ref()?;
                    let mut residual_states = Vec::with_capacity(raw_representatives.len());
                    for &representative in &raw_representatives {
                        let coordinate_row = coordinate_states.get(representative as usize)?;
                        let left_state = *coordinate_row.get(left_coordinate)?;
                        if left_state == u32::MAX
                            || !single_group_dfa_state_is_live(left_dfa.as_ref(), left_state)
                        {
                            return None;
                        }
                        let raw_right_state = *coordinate_row.get(right_coordinate)?;
                        let (right_state, max_remaining) = if raw_right_state == u32::MAX
                            || !single_group_dfa_state_is_live(right_dfa.as_ref(), raw_right_state)
                        {
                            (u32::MAX, u32::MAX)
                        } else {
                            let remaining = *right_max_remaining
                                .get(raw_right_state as usize)?
                                .as_ref()?;
                            (raw_right_state, remaining)
                        };
                        residual_states.push(TerminalExclusionResidualState {
                            left_state,
                            right_state,
                            right_max_remaining: max_remaining,
                        });
                    }
                    Some(Arc::new(TerminalExclusionCertificate::new(
                        left_dfa,
                        residual_states,
                    )?))
                })();

                if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
                    eprintln!(
                        "[glrmask/profile][product_trace_exceptional_terminal] local_group={} exclusions={} intersections={} source_states={} source_transitions={} source_compile_ms={:.3} mapping_ms={:.3}",
                        local_group,
                        exclusions.len(),
                        intersections.len(),
                        source.num_states(),
                        source.transition_count(),
                        source_compile_ms,
                        mapping_started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                Some(Some(ComplexTerminalSource {
                    source,
                    mapping,
                    exclusion,
                }))
            })
            .collect::<Option<Vec<_>>>()?;

            let mut rows = Vec::with_capacity(dfa.num_states());
            for state in 0..dfa.num_states() {
                let tuple = trace.state_tuples.tuple(state);
                let mut row = Vec::<(u32, u32)>::with_capacity(
                    tuple.len().saturating_add(complex_group_count),
                );
                if complex_group_count != visible_groups {
                    for &(coordinate, residual_state) in &tuple {
                        for &local_group in trace.coordinate_groups.get(coordinate as usize)? {
                            if local_group >= visible_groups || simple_sources[local_group].is_none() {
                                continue;
                            }
                            row.push((terminal_ids[local_group] as u32, residual_state));
                        }
                    }
                }
                if complex_group_count != 0 {
                    for local_group in 0..visible_groups {
                        if simple_sources[local_group].is_some() {
                            continue;
                        }
                        let complex_source = complex_sources[local_group].as_ref()?;
                        let residual_state = complex_source.mapping[state];
                        if residual_state != u32::MAX {
                            row.push((terminal_ids[local_group] as u32, residual_state));
                        }
                    }
                }
                if complex_group_count != visible_groups {
                    row.sort_unstable_by_key(|&(terminal, _)| terminal);
                }
                rows.push(row);
            }

            let mut sources = Vec::with_capacity(visible_groups);
            for local_group in 0..visible_groups {
                let terminal = terminal_ids[local_group];
                if let Some(source) = simple_sources[local_group].take() {
                    sources.push((terminal, source, 0, None));
                } else {
                    let complex_source = complex_sources[local_group].take()?;
                    sources.push((
                        terminal,
                        complex_source.source,
                        0,
                        complex_source.exclusion,
                    ));
                }
            }
            Some(TracedComponent {
                terminal_ids,
                dfa,
                rows,
                sources,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let total_states = 1usize
        + components
            .iter()
            .map(|component| component.dfa.num_states())
            .sum::<usize>();
    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());
    let mut rows = Vec::<Vec<(u32, u32)>>::with_capacity(total_states);
    rows.push(Vec::new());
    let mut terminal_dfas = vec![None::<Arc<DFA>>; exprs.len()];
    let mut terminal_groups = vec![u32::MAX; exprs.len()];
    let mut exclusion_certificates = vec![None::<Arc<TerminalExclusionCertificate>>; exprs.len()];
    let mut offset = 1u32;
    for component in components {
        for (terminal, source, group, exclusion) in component.sources {
            if terminal >= exprs.len() || terminal_dfas[terminal].is_some() {
                return None;
            }
            terminal_dfas[terminal] = Some(source);
            terminal_groups[terminal] = group;
            exclusion_certificates[terminal] = exclusion;
        }
        for local_group in component.dfa.possible_future_group_ids(0).iter() {
            root_futures.set(component.terminal_ids[local_group]);
        }
        combined.add_epsilon_transition(0, offset);
        let actual_offset = combined.append_rebased_component_ref(
            component.dfa.as_ref(),
            &component.terminal_ids,
        );
        if actual_offset != offset {
            return None;
        }
        rows.extend(component.rows);
        offset += component.dfa.num_states() as u32;
    }
    combined.set_possible_future_group_ids(0, root_futures);
    let terminal_dfas = terminal_dfas.into_iter().collect::<Option<Vec<_>>>()?;
    if terminal_groups.iter().any(|&group| group == u32::MAX) || rows.len() != combined.num_states() {
        return None;
    }
    let mut tokenizer = Regex { dfa: combined }.into_tokenizer(exprs.len() as u32, Some(retained_exprs));
    let coordinates = TerminalResidualCoordinates::from_rows_and_dfa_groups(
        rows,
        terminal_dfas,
        terminal_groups,
    )
    .with_exclusion_certificates(exclusion_certificates)?;
    tokenizer.set_terminal_residual_coordinates(coordinates);
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][product_trace_terminal_residual_coordinates] terminals={} states={} elapsed_ms={:.3}",
            exprs.len(),
            tokenizer.num_states(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(tokenizer)
}

/// Build an exact partitioned tokenizer while keeping very large pure binary
/// intersection transition tables in their compressed runtime form.
///
/// Unlike the structural compile/runtime pair, this path does not construct a
/// synthesized tokenizer or a full-to-synthesized state map. It is intended
/// for consumers, such as direct dynamic masking, that need only the exact
/// runtime tokenizer.
pub fn build_exact_partitioned_runtime_tokenizer(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
) -> Tokenizer {
    // Below this size, the ordinary materialized product is already cheap and
    // has better locality. Large products keep class transitions compressed.
    const MIN_COMPRESSED_RUNTIME_PAIR_CELLS: usize = 1_000_000;

    assert_eq!(exprs.len(), partitions.len());
    assert_eq!(exprs.len(), residual_isolation_classes.len());
    if let Some(labels) = visible_labels {
        assert_eq!(exprs.len(), labels.len());
    }

    let rewritten_exprs = materialize_repeated_subexpression_dfas(exprs);
    let compile_exprs = rewritten_exprs.as_deref().unwrap_or(exprs);
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }

    for terminal_ids in grouped.values() {
        let mut class = None;
        let mut has_unprotected = false;
        for &terminal in terminal_ids {
            match residual_isolation_classes[terminal] {
                Some(current) => match class {
                    Some(previous) => assert_eq!(
                        previous, current,
                        "one lexer partition cannot contain distinct protected residual classes",
                    ),
                    None => class = Some(current),
                },
                None => has_unprotected = true,
            }
        }
        assert!(
            class.is_none() || !has_unprotected,
            "one lexer partition cannot mix protected and unprotected residual coordinates",
        );
    }

    let shared_duplicates = shared_duplicate_nested_group_op_cache(compile_exprs, &grouped);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }

    let components = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_partition, terminal_ids)| {
            let local_exprs = terminal_ids
                .iter()
                .map(|&terminal| compile_exprs[terminal].clone())
                .collect::<Vec<_>>();
            let local_labels = visible_labels.map(|labels| {
                terminal_ids
                    .iter()
                    .map(|&terminal| labels[terminal].clone())
                    .collect::<Vec<_>>()
            });
            let mut nested_group_op_cache = NestedGroupOpCache {
                shared_duplicates: shared_duplicates.clone(),
                ..NestedGroupOpCache::default()
            };
            let plan = build_exclusion_compile_plan_with_labels_and_cache(
                &local_exprs,
                local_labels.as_deref(),
                &mut nested_group_op_cache,
            );
            let full = match try_compile_with_plan_deferred_dense_min_pair_cells(
                plan,
                MIN_COMPRESSED_RUNTIME_PAIR_CELLS,
                true,
            ) {
                Ok((mut full, trace)) => {
                    if let Some(trace) = trace {
                        full.attach_dense_runtime_trace(trace)
                            .expect("deferred exact runtime tokenizer must accept its state trace");
                    }
                    full
                }
                Err(plan) => DeferredDfa::Ready(compile_with_plan(plan)),
            };
            (terminal_ids, full)
        })
        .collect::<Vec<_>>();

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());
    let mut compressed_segments = Vec::new();
    for (terminal_ids, full) in components {
        let (mut component_dfa, compressed) = full.finish_runtime();
        if compressed.is_none() {
            // Isolate nullable roots while this component is still small and
            // independent. The final tokenizer is a disjoint epsilon union of
            // these components, so this is equivalent to isolating the full
            // union but avoids rescanning every state in unrelated huge
            // compressed components.
            component_dfa = isolate_component_nullable_start(
                component_dfa,
                terminal_ids.len(),
            )
            .0;
        }
        debug_assert_eq!(component_dfa.num_groups(), terminal_ids.len());
        for local_group in component_dfa.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(component_dfa, &terminal_ids);
        combined.add_epsilon_transition(0, offset);
        if let Some(mut segment) = compressed {
            segment.state_offset = offset;
            compressed_segments.push(segment);
        }
    }
    combined.set_possible_future_group_ids(0, root_futures);
    Tokenizer::from_parts_with_compressed_transitions(
        combined,
        exprs.len() as u32,
        Some(Arc::from(exprs.to_vec().into_boxed_slice())),
        compressed_segments,
    )
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "invalid {name}={other:?}; expected one of 1/0, true/false, yes/no, or on/off"
            ),
        },
        Err(_) => default,
    }
}

fn adaptive_lexer_enabled() -> bool {
    match std::env::var("GLRMASK_LEXER_ADAPTIVE") {
        Ok(_) => env_flag("GLRMASK_LEXER_ADAPTIVE", false),
        // Keep the long-standing depth override useful as a one-variable
        // diagnostic opt-in even though adaptive determinization is no longer
        // enabled by default.
        Err(_) => std::env::var_os("GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH").is_some(),
    }
}

fn adaptive_lexer_state_limit() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(32_768)
}

fn adaptive_lexer_max_depth() -> Option<usize> {
    let Ok(value) = std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH") else {
        return Some(1);
    };
    let value = value.trim();
    if matches!(value.to_ascii_lowercase().as_str(), "full" | "unbounded") {
        return None;
    }
    Some(value.parse::<usize>().unwrap_or_else(|_| {
        panic!(
            "invalid GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH={value:?}; expected a byte depth or full"
        )
    }))
}

/// Extra product-prefix states permitted for a bounded adaptive lexer.
///
/// A bounded product retains exact copies of the independently compiled
/// component DFAs after its cutoff, so percentage growth relative to those
/// components is the wrong resource model.  At depth one the prefix can add at
/// most one state per byte (plus the shared root already present in the
/// untouched epsilon union), making 256 a tight representation-independent
/// default overhead cap. Deeper explicit experiments can raise this budget.
fn adaptive_lexer_bounded_overhead_states() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_BOUNDED_OVERHEAD_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(256)
}

fn adaptive_lexer_growth_percent() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_GROWTH_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(100)
}

fn adaptive_lexer_transition_growth_percent() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_TRANSITION_GROWTH_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(600)
}

fn adaptive_transition_growth_is_acceptable(
    input_transitions: usize,
    output_transitions: usize,
    growth_percent: usize,
) -> bool {
    output_transitions.saturating_mul(100)
        <= input_transitions.saturating_mul(growth_percent)
}

fn compile_terminal_ids(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: &[usize],
) -> DFA {
    compile_terminal_ids_with_shared_duplicate_cache(exprs, visible_labels, terminal_ids, None)
}

fn compile_terminal_ids_with_shared_duplicate_cache(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    terminal_ids: &[usize],
    shared_duplicates: Option<&Arc<SharedDuplicateNestedGroupOpCache>>,
) -> DFA {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile.then(Instant::now);
    let clone_started_at = profile.then(Instant::now);
    let local_exprs = terminal_ids
        .iter()
        .map(|&terminal| exprs[terminal].clone())
        .collect::<Vec<_>>();
    let local_labels = visible_labels.map(|labels| {
        terminal_ids
            .iter()
            .map(|&terminal| labels[terminal].clone())
            .collect::<Vec<_>>()
    });
    let clone_ms = clone_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let mut nested_group_op_cache = NestedGroupOpCache {
        shared_duplicates: shared_duplicates.cloned(),
        ..NestedGroupOpCache::default()
    };
    let plan_started_at = profile.then(Instant::now);
    let plan = build_exclusion_compile_plan_with_labels_and_cache(
        &local_exprs,
        local_labels.as_deref(),
        &mut nested_group_op_cache,
    );
    let plan_ms = plan_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let compile_started_at = profile.then(Instant::now);
    let dfa = compile_with_plan(plan);
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] partition_plan_detail terminals={} clone_ms={:.3} plan_ms={:.3} nested_entries={} nested_hits={} nested_misses={} nested_compiled_ms={:.3} nested_max_compile_ms={:.3} compile_ms={:.3} total_ms={:.3}",
            terminal_ids.len(),
            clone_ms,
            plan_ms,
            nested_group_op_cache.compiled.len(),
            nested_group_op_cache.cache_hits,
            nested_group_op_cache.cache_misses,
            nested_group_op_cache.compiled_ms,
            nested_group_op_cache.max_compile_ms,
            compile_started_at
                .map(|started| started.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    dfa
}

struct LexerComponent {
    terminal_ids: Vec<usize>,
    dfa: DFA,
    protected_residual: bool,
}

struct LexerComponentPair {
    terminal_ids: Vec<usize>,
    synthesized: DFA,
    full: DeferredDfa,
    full_to_synthesized: Vec<u32>,
    protected_residual: bool,
}

enum DeferredDfa {
    Ready(DFA),
    ReadyCompressed {
        dfa: DFA,
        segment: CompressedTransitionSegment,
    },
    DenseBinary(DeferredDenseBinaryIntersectionProduct),
}

impl DeferredDfa {
    fn num_states(&self) -> usize {
        match self {
            Self::Ready(dfa) => dfa.num_states(),
            Self::ReadyCompressed { dfa, .. } => dfa.num_states(),
            Self::DenseBinary(product) => product.num_states(),
        }
    }

    fn initial_is_nonnullable_without_epsilon(&self) -> bool {
        match self {
            Self::Ready(dfa) => {
                dfa.finalizers(0).is_empty() && !dfa.has_epsilon_transitions()
            }
            Self::ReadyCompressed { dfa, .. } => {
                dfa.finalizers(0).is_empty() && !dfa.has_epsilon_transitions()
            }
            Self::DenseBinary(product) => !product.initial_is_accepting(),
        }
    }

    fn ensure_group_capacity(&mut self, num_groups: usize) {
        match self {
            Self::Ready(dfa) => dfa.ensure_group_capacity(num_groups),
            Self::ReadyCompressed { dfa, .. } => dfa.ensure_group_capacity(num_groups),
            Self::DenseBinary(product) => product.ensure_group_capacity(num_groups),
        }
    }

    fn set_group_u8set(&mut self, group_id: u32, set: U8Set) {
        match self {
            Self::Ready(dfa) => dfa.set_group_u8set(group_id, set),
            Self::ReadyCompressed { dfa, .. } => dfa.set_group_u8set(group_id, set),
            Self::DenseBinary(product) => product.set_group_u8set(group_id, set),
        }
    }

    fn attach_dense_runtime_trace(&mut self, trace: ProductBuildTrace) -> Option<()> {
        match self {
            Self::Ready(_) | Self::ReadyCompressed { .. } => Some(()),
            Self::DenseBinary(product) => product.attach_runtime_trace(trace),
        }
    }

    fn has_deferred_runtime_materialization(&self) -> bool {
        match self {
            Self::Ready(_) => false,
            Self::ReadyCompressed { .. } => true,
            Self::DenseBinary(product) => product.deferred_transition_rows.is_some(),
        }
    }

    fn finish(self) -> DFA {
        match self {
            Self::Ready(dfa) => dfa,
            Self::ReadyCompressed { mut dfa, segment } => {
                segment.materialize_into_dfa(&mut dfa);
                dfa
            }
            Self::DenseBinary(product) => product.finish(),
        }
    }

    fn finish_runtime(self) -> (DFA, Option<CompressedTransitionSegment>) {
        match self {
            Self::Ready(dfa) => (dfa, None),
            Self::ReadyCompressed { dfa, segment } => (dfa, Some(segment)),
            Self::DenseBinary(product) => {
                let (dfa, segment) = product.finish_compressed();
                (dfa, Some(segment))
            }
        }
    }
}

fn isolate_component_nullable_start(dfa: DFA, num_terminals: usize) -> (DFA, BTreeSet<u32>) {
    let mut tokenizer = Regex { dfa }.into_tokenizer(num_terminals as u32, None);
    let nullable = tokenizer.isolate_start_state_and_drain_nullable_terminals();
    (tokenizer.dfa, nullable)
}

pub struct CompiledPartitionedExpressionPair {
    pub synthesized: Regex,
    pub full: Regex,
    pub full_to_synthesized: Vec<u32>,
}

pub struct PreparedPartitionedExpressionPair {
    pub synthesized: Regex,
    full: DeferredPartitionedRegex,
    pub full_to_synthesized: Vec<u32>,
    pub synthesized_expressions: Vec<Expr>,
}

impl PreparedPartitionedExpressionPair {
    pub fn full_num_states(&self) -> usize {
        self.full.num_states()
    }

    pub fn finish_full(self) -> CompiledPartitionedExpressionPair {
        CompiledPartitionedExpressionPair {
            synthesized: self.synthesized,
            full: self.full.finish(),
            full_to_synthesized: self.full_to_synthesized,
        }
    }

    pub fn into_parts(
        self,
    ) -> (Regex, DeferredPartitionedRegex, Vec<u32>, Vec<Expr>) {
        (
            self.synthesized,
            self.full,
            self.full_to_synthesized,
            self.synthesized_expressions,
        )
    }
}

pub struct DeferredPartitionedRegex {
    components: Vec<LexerComponentPair>,
    total_groups: usize,
}

impl DeferredPartitionedRegex {
    pub fn num_states(&self) -> usize {
        1 + self
            .components
            .iter()
            .map(|component| component.full.num_states())
            .sum::<usize>()
    }

    pub fn has_deferred_runtime_materialization(&self) -> bool {
        self.components
            .iter()
            .any(|component| component.full.has_deferred_runtime_materialization())
    }

    pub fn finish(self) -> Regex {
        let components = self
            .components
            .into_iter()
            .map(|component| LexerComponent {
                terminal_ids: component.terminal_ids,
                dfa: component.full.finish(),
                protected_residual: component.protected_residual,
            })
            .collect::<Vec<_>>();
        Regex {
            dfa: combine_lexer_components_under_epsilon_root(components, self.total_groups),
        }
    }

    pub fn finish_runtime_tokenizer(
        self,
        num_terminals: u32,
        expressions: Arc<[Expr]>,
    ) -> Tokenizer {
        let mut combined = DFA::new(1);
        combined.ensure_group_capacity(self.total_groups);
        let mut root_futures = BitSet::new(self.total_groups);
        let mut compressed_segments = Vec::new();

        for component in self.components {
            let terminal_ids = component.terminal_ids;
            let (component_dfa, compressed) = component.full.finish_runtime();
            debug_assert_eq!(component_dfa.num_groups(), terminal_ids.len());
            for local_group in component_dfa.possible_future_group_ids(0).iter() {
                root_futures.set(terminal_ids[local_group]);
            }
            let offset = combined.append_rebased_component(component_dfa, &terminal_ids);
            combined.add_epsilon_transition(0, offset);
            if let Some(mut segment) = compressed {
                segment.state_offset = offset;
                compressed_segments.push(segment);
            }
        }
        combined.set_possible_future_group_ids(0, root_futures);
        Tokenizer::from_parts_with_compressed_transitions(
            combined,
            num_terminals,
            Some(expressions),
            compressed_segments,
        )
    }
}

fn compile_partition_components(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
) -> Vec<LexerComponent> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_top = profile
        || std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOKENIZER_TOP").is_some();
    let rewrite_started_at = Instant::now();
    let rewritten_exprs = materialize_repeated_subexpression_dfas(exprs);
    let rewrite_ms = rewrite_started_at.elapsed().as_secs_f64() * 1000.0;
    let exprs = rewritten_exprs.as_deref().unwrap_or(exprs);
    let setup_started_at = Instant::now();
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }
    if let Some(classes) = residual_isolation_classes {
        assert_eq!(
            classes.len(),
            exprs.len(),
            "one residual isolation class entry is required per terminal",
        );
        for terminal_ids in grouped.values() {
            let mut class = None;
            let mut has_unprotected = false;
            for &terminal in terminal_ids {
                match classes[terminal] {
                    Some(current) => match class {
                        Some(previous) => assert_eq!(
                            previous, current,
                            "one lexer partition cannot contain distinct protected residual classes",
                        ),
                        None => class = Some(current),
                    },
                    None => has_unprotected = true,
                }
            }
            assert!(
                class.is_none() || !has_unprotected,
                "one lexer partition cannot mix protected and unprotected residual coordinates",
            );
        }
    }
    let shared_duplicates = shared_duplicate_nested_group_op_cache(exprs, &grouped);
    let prewarm_started_at = Instant::now();
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }
    let prewarm_ms = prewarm_started_at.elapsed().as_secs_f64() * 1000.0;
    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0 - prewarm_ms;

    let compile_started_at = Instant::now();
    let components: Vec<LexerComponent> = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(partition, terminal_ids)| {
            let started_at = Instant::now();
            let dfa = compile_terminal_ids_with_shared_duplicate_cache(
                exprs,
                visible_labels,
                &terminal_ids,
                shared_duplicates.as_ref(),
            );
            if profile {
                let single_label = (terminal_ids.len() == 1)
                    .then(|| {
                        visible_labels
                            .and_then(|labels| labels.get(terminal_ids[0]))
                            .map(String::as_str)
                    })
                    .flatten();
                let single_expr = (terminal_ids.len() == 1)
                    .then(|| expr_profile_summary(&exprs[terminal_ids[0]]));
                eprintln!(
                    "[glrmask/profile][tokenizer] partition_compile partition={} terminals={} terminal_ids={:?} single_label={:?} single_expr={:?} states={} transitions={} total_ms={:.3}",
                    partition,
                    terminal_ids.len(),
                    terminal_ids,
                    single_label,
                    single_expr,
                    dfa.num_states(),
                    dfa_transition_count(&dfa),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let protected_residual = residual_isolation_classes.is_some_and(|classes| {
                terminal_ids
                    .first()
                    .is_some_and(|&terminal| classes[terminal].is_some())
            });
            LexerComponent {
                terminal_ids,
                dfa,
                protected_residual,
            }
        })
        .collect();
    if profile_top {
        eprintln!(
            "[glrmask/profile][tokenizer_partition_components_top] terminals={} components={} rewrite_ms={:.3} setup_ms={:.3} prewarm_ms={:.3} compile_wall_ms={:.3} total_ms={:.3}",
            exprs.len(),
            components.len(),
            rewrite_ms,
            setup_ms,
            prewarm_ms,
            compile_started_at.elapsed().as_secs_f64() * 1000.0,
            rewrite_ms + setup_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    components
}

fn lexer_component_product_metadata(
    components: &[LexerComponent],
    group_offsets: &[usize],
    state_tuple: &ProductStateTuple,
    total_groups: usize,
) -> (BitSet, BitSet) {
    let mut finalizers = BitSet::new(total_groups);
    let mut futures = BitSet::new(total_groups);

    for &(component_id, component_state) in state_tuple {
        let component_index = component_id as usize;
        let component = &components[component_index].dfa;
        let offset = group_offsets[component_index];
        for group in component.finalizers(component_state).iter() {
            finalizers.set(offset + group);
        }
        for group in component
            .possible_future_group_ids(component_state)
            .iter()
        {
            futures.set(offset + group);
        }
    }

    (finalizers, futures)
}

fn compute_lexer_component_equivalence_classes(
    components: &[LexerComponent],
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for component in components {
        for state in component.dfa.states() {
            let mut bytes_by_target = FxHashMap::<u32, U8Set>::default();
            for (byte, &target) in state.transitions.iter() {
                bytes_by_target
                    .entry(target)
                    .and_modify(|set| {
                        set.insert(byte);
                    })
                    .or_insert_with(|| U8Set::single(byte));
            }
            for byte_set in bytes_by_target.into_values() {
                if seen_sets.insert(byte_set) {
                    partitions = refine_u8_partitions(partitions, byte_set);
                }
            }
        }
    }

    let mut class_map = vec![0u8; 256];
    let mut class_members = vec![Vec::new(); partitions.len()];
    for (class_id, partition) in partitions.iter().enumerate() {
        for byte in partition.iter() {
            class_map[byte as usize] = class_id as u8;
            class_members[class_id].push(byte);
        }
    }
    (class_map, class_members)
}

/// Attempt one exact prefix determinization of the final union of independently
/// compiled lexer partitions. The sparse product tuple contains only component
/// states still live after the consumed bytes. Product construction stops at
/// `max_depth` consumed bytes and reconnects each frontier tuple to exact copies
/// of its live component states with epsilon edges. Thus adaptive
/// determinization can coalesce a bounded prefix without forcing unrelated
/// long-running terminals into the product for their entire lifetime.
///
/// Construction also stops before allocating the first product state beyond
/// `state_limit`; callers can then preserve the original epsilon-NFA unchanged.
/// `None` retains the historical full-product behavior.
fn try_product_union_components(
    components: &[LexerComponent],
    state_limit: usize,
    transition_limit: usize,
    max_depth: Option<usize>,
) -> Option<DFA> {
    assert!(state_limit > 0, "adaptive lexer state limit must be positive");
    assert!(transition_limit > 0, "adaptive lexer transition limit must be positive");
    debug_assert!(components
        .iter()
        .all(|component| !component.dfa.has_epsilon_transitions()));

    // A bounded product is not just its determinized prefix: after the cutoff
    // it appends exact copies of the component DFAs and reconnects frontier
    // tuples to them.  Account for those copies *before* constructing the
    // prefix so `state_limit` / `transition_limit` bound the final DFA rather
    // than only the speculative prefix.  The previous implementation applied
    // the state cap only to `combined` and then appended every component,
    // allowing a nominal 100% growth limit to produce a larger lexer.
    let copied_states = if max_depth.is_some() {
        components
            .iter()
            .map(|component| component.dfa.num_states())
            .sum::<usize>()
    } else {
        0
    };
    let copied_transitions = if max_depth.is_some() {
        components
            .iter()
            .map(|component| dfa_transition_count(&component.dfa))
            .sum::<usize>()
    } else {
        0
    };
    let product_state_limit = state_limit.checked_sub(copied_states)?;
    if product_state_limit == 0 {
        return None;
    }
    let product_transition_limit = transition_limit.checked_sub(copied_transitions)?;

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let setup_started_at = Instant::now();
    let total_groups = components
        .iter()
        .map(|component| component.dfa.num_groups())
        .sum::<usize>();
    let mut group_offsets = Vec::with_capacity(components.len());
    let mut group_offset = 0usize;
    for component in components {
        group_offsets.push(group_offset);
        group_offset += component.dfa.num_groups();
    }
    let component_dead_states = components
        .iter()
        .map(|component| explicit_dead_sink_state(&component.dfa))
        .collect::<Vec<_>>();
    let class_started_at = Instant::now();
    let (class_map, class_members) = compute_lexer_component_equivalence_classes(components);
    let class_ms = class_started_at.elapsed().as_secs_f64() * 1000.0;
    let class_transitions_started_at = Instant::now();
    let component_class_transitions = components
        .iter()
        .map(|component| build_product_class_transitions_for_dfa(&component.dfa, &class_map))
        .collect::<Vec<_>>();
    let class_transitions_ms = class_transitions_started_at.elapsed().as_secs_f64() * 1000.0;
    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_classes = class_members.len();

    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(total_groups);
    for (component_index, component) in components.iter().enumerate() {
        let offset = group_offsets[component_index];
        for local_group in 0..component.dfa.num_groups() {
            combined.set_group_u8set(
                (offset + local_group) as u32,
                *component.dfa.group_id_to_u8set(local_group as u32),
            );
        }
    }

    let mut start = ProductStateTuple::with_capacity(components.len());
    for component_id in 0..components.len() {
        start.push((component_id as u32, 0));
    }
    let (finalizers, futures) =
        lexer_component_product_metadata(components, &group_offsets, &start, total_groups);
    combined.overwrite_state_metadata(0, finalizers, futures);

    #[derive(Clone, PartialEq, Eq, Hash)]
    enum ProductStateKey {
        Full(ProductStateTuple),
        Bounded(usize, ProductStateTuple),
    }
    let state_key = |depth: usize, tuple: ProductStateTuple| match max_depth {
        Some(_) => ProductStateKey::Bounded(depth, tuple),
        None => ProductStateKey::Full(tuple),
    };

    let mut state_map = FxHashMap::<ProductStateKey, u32>::default();
    state_map.insert(state_key(0, start.clone()), 0);
    let mut worklist = VecDeque::from([(0u32, start, 0usize)]);
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    let mut frontier_states = Vec::<(u32, ProductStateTuple)>::new();
    let mut class_buffers = (0..num_classes)
        .map(|_| ProductStateTuple::new())
        .collect::<Vec<_>>();
    let mut class_active = vec![false; num_classes];
    let mut used_classes = Vec::<usize>::new();
    let mut projected_byte_transitions = 0usize;

    let state_expand_started_at = Instant::now();
    while let Some((combined_state, state_tuple, depth)) = worklist.pop_front() {
        if max_depth.is_some_and(|max_depth| depth >= max_depth) {
            frontier_states.push((combined_state, state_tuple));
            continue;
        }
        for &(component_id, component_state) in &state_tuple {
            let component_index = component_id as usize;
            for &(class_id, target) in
                &component_class_transitions[component_index][component_state as usize]
            {
                let class_index = class_id as usize;
                if !class_active[class_index] {
                    class_active[class_index] = true;
                    used_classes.push(class_index);
                }
                if component_dead_states[component_index] == Some(target) {
                    continue;
                }
                class_buffers[class_index].push((component_id, target));
            }
        }

        let mut transitions = Vec::with_capacity(used_classes.len());
        for &class_index in &used_classes {
            projected_byte_transitions = projected_byte_transitions
                .saturating_add(class_members[class_index].len());
            if projected_byte_transitions > product_transition_limit {
                return None;
            }
            let next_tuple = &class_buffers[class_index];
            let next_depth = depth + 1;
            let next_key = state_key(next_depth, next_tuple.clone());
            let target = if let Some(&existing) = state_map.get(&next_key) {
                existing
            } else {
                if combined.num_states() >= product_state_limit {
                    return None;
                }
                let new_state = combined.add_state();
                let (finalizers, futures) = lexer_component_product_metadata(
                    components,
                    &group_offsets,
                    next_tuple,
                    total_groups,
                );
                combined.overwrite_state_metadata(new_state, finalizers, futures);
                state_map.insert(next_key, new_state);
                pending_class_transitions.push(Vec::new());
                worklist.push_back((new_state, next_tuple.clone(), next_depth));
                new_state
            };
            transitions.push((class_index as u8, target));
            class_buffers[class_index].clear();
            class_active[class_index] = false;
        }
        used_classes.clear();
        pending_class_transitions[combined_state as usize] = transitions;
    }
    let state_expand_ms = state_expand_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_expand_started_at = Instant::now();
    let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
        pending_class_transitions
            .into_par_iter()
            .map(|class_transitions| {
                let byte_capacity = class_transitions
                    .iter()
                    .map(|(class_id, _)| class_members[*class_id as usize].len())
                    .sum::<usize>();
                const DENSE_BYTE_EXPANSION_THRESHOLD: usize = 96;
                let transitions = if byte_capacity >= DENSE_BYTE_EXPANSION_THRESHOLD {
                    let mut target_by_byte = [u32::MAX; 256];
                    for (class_id, target) in class_transitions {
                        for &byte in &class_members[class_id as usize] {
                            target_by_byte[byte as usize] = target;
                        }
                    }
                    target_by_byte
                        .into_iter()
                        .enumerate()
                        .filter_map(|(byte, target)| {
                            (target != u32::MAX).then_some((byte as u8, target))
                        })
                        .collect()
                } else {
                    let mut transitions = Vec::with_capacity(byte_capacity);
                    for (class_id, target) in class_transitions {
                        for &byte in &class_members[class_id as usize] {
                            transitions.push((byte, target));
                        }
                    }
                    if transitions.len() > 1 {
                        transitions.sort_unstable_by_key(|entry| entry.0);
                    }
                    transitions
                };
                crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
            })
            .collect();
    for (state, transitions) in combined.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }

    if max_depth.is_some() {
        // Append one exact copy of every independently compiled component. A
        // product frontier can then resume from the precise component states in
        // its sparse tuple instead of continuing cross-component
        // determinization.
        let mut component_offsets = Vec::with_capacity(components.len());
        for (component_index, component) in components.iter().enumerate() {
            let component_offset = combined.num_states() as u32;
            component_offsets.push(component_offset);
            for _ in 0..component.dfa.num_states() {
                combined.add_state();
            }

            let group_offset = group_offsets[component_index];
            for (state_index, state) in component.dfa.states().iter().enumerate() {
                let mapped_state = component_offset + state_index as u32;
                combined.set_transitions_from_sorted_entries(
                    mapped_state,
                    state
                        .transitions
                        .iter()
                        .map(|(byte, &target)| (byte, component_offset + target))
                        .collect(),
                );
                for &target in &state.epsilon_transitions {
                    combined.add_epsilon_transition(mapped_state, component_offset + target);
                }

                let mut finalizers = BitSet::new(total_groups);
                let mut futures = BitSet::new(total_groups);
                for local_group in state.finalizers.iter() {
                    finalizers.set(group_offset + local_group);
                }
                for local_group in component
                    .dfa
                    .possible_future_group_ids(state_index as u32)
                    .iter()
                {
                    futures.set(group_offset + local_group);
                }
                combined.overwrite_state_metadata(mapped_state, finalizers, futures);
            }
        }

        for (frontier_state, state_tuple) in frontier_states {
            // The frontier is an epsilon branching state, like the root of the
            // ordinary partition union. Its exact component children carry
            // acceptance. Duplicating finalizers on the branch state creates a
            // second accepting path for the same terminal and is observably
            // different to the original epsilon-NFA for downstream state-set
            // analyses. Strict futures may remain cached on the branch state,
            // just as they are on the ordinary epsilon-union root.
            let futures = combined.possible_future_group_ids(frontier_state).clone();
            combined.overwrite_state_metadata(
                frontier_state,
                BitSet::new(total_groups),
                futures,
            );
            for (component_id, component_state) in state_tuple {
                combined.add_epsilon_transition(
                    frontier_state,
                    component_offsets[component_id as usize] + component_state,
                );
            }
        }

        debug_assert!(combined.num_states() <= state_limit);
        debug_assert!(dfa_transition_count(&combined) <= transition_limit);
    }
    let byte_expand_ms = byte_expand_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_product states={} classes={} max_depth={:?} setup_ms={:.3} class_ms={:.3} class_transitions_ms={:.3} state_expand_ms={:.3} byte_expand_ms={:.3}",
            combined.num_states(),
            num_classes,
            max_depth,
            setup_ms,
            class_ms,
            class_transitions_ms,
            state_expand_ms,
            byte_expand_ms,
        );
    }

    Some(combined)
}

fn adaptively_determinize_components(
    inputs: Vec<LexerComponent>,
    state_limit: usize,
) -> Vec<LexerComponent> {
    adaptively_determinize_components_with_limits(
        inputs,
        state_limit,
        adaptive_lexer_growth_percent(),
        adaptive_lexer_transition_growth_percent(),
        adaptive_lexer_max_depth(),
    )
}

fn adaptively_determinize_components_with_limits(
    inputs: Vec<LexerComponent>,
    state_limit: usize,
    growth_percent: usize,
    transition_growth_percent: usize,
    max_depth: Option<usize>,
) -> Vec<LexerComponent> {
    if inputs.iter().any(|component| component.protected_residual) {
        let mut protected = Vec::new();
        let mut ordinary = Vec::new();
        for component in inputs {
            if component.protected_residual {
                protected.push(component);
            } else {
                ordinary.push(component);
            }
        }

        let mut output = if ordinary.len() >= 2 {
            adaptively_determinize_components_with_limits(
                ordinary,
                state_limit,
                growth_percent,
                transition_growth_percent,
                max_depth,
            )
        } else {
            ordinary
        };
        output.append(&mut protected);
        output.sort_unstable_by_key(|component| {
            component.terminal_ids.first().copied().unwrap_or(usize::MAX)
        });
        return output;
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = Instant::now();
    let input_states = inputs.iter().map(|batch| batch.dfa.num_states()).sum::<usize>();
    let input_transitions = inputs
        .iter()
        .map(|batch| dfa_transition_count(&batch.dfa))
        .sum::<usize>();
    let input_batches = inputs.len();
    let terminals = inputs
        .iter()
        .map(|batch| batch.terminal_ids.len())
        .sum::<usize>();

    // Compare against the representation we would actually return if adaptive
    // determinization is rejected: the disjoint component DFAs plus their one
    // epsilon-union root.  Using only `input_states` made a 100% growth cap one
    // state stricter for full products while, paradoxically, bounded products
    // could exceed it after appending their component copies.
    let baseline_states = input_states.saturating_add(usize::from(input_batches > 1));
    let percentage_growth_limit = baseline_states
        .saturating_mul(growth_percent)
        .saturating_add(99)
        / 100;
    let effective_state_limit = match max_depth {
        None => state_limit.min(percentage_growth_limit).max(1),
        Some(_) => state_limit
            .min(
                baseline_states.saturating_add(adaptive_lexer_bounded_overhead_states()),
            )
            .max(1),
    };
    let transition_limit = input_transitions
        .saturating_mul(transition_growth_percent)
        / 100;
    let attempt_started_at = Instant::now();
    let candidate = try_product_union_components(
        &inputs,
        effective_state_limit,
        transition_limit.max(1),
        max_depth,
    );
    let attempt_ms = attempt_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_transitions = candidate.as_ref().map(dfa_transition_count);
    let accepted = candidate.as_ref().is_some_and(|dfa| {
        dfa.num_states() <= effective_state_limit
            && adaptive_transition_growth_is_acceptable(
                input_transitions,
                dfa_transition_count(dfa),
                transition_growth_percent,
            )
    });
    let terminal_ids = inputs
        .iter()
        .flat_map(|component| component.terminal_ids.iter().copied())
        .collect::<Vec<_>>();
    let output_states = candidate.as_ref().map_or(input_states, DFA::num_states);
    let output_transitions = candidate_transitions.unwrap_or(input_transitions);

    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] adaptive_determinize partitions={} terminals={} output_components={} input_states={} baseline_states={} output_states={} input_transitions={} output_transitions={} accepted={} max_states={} effective_state_limit={} max_depth={:?} bounded_overhead_states={} max_growth_percent={} max_transition_growth_percent={} attempt_ms={:.3} total_ms={:.3}",
            input_batches,
            terminals,
            if accepted { 1 } else { input_batches },
            input_states,
            baseline_states,
            output_states,
            input_transitions,
            output_transitions,
            accepted,
            state_limit,
            effective_state_limit,
            max_depth,
            adaptive_lexer_bounded_overhead_states(),
            growth_percent,
            transition_growth_percent,
            attempt_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    match candidate {
        Some(dfa) if accepted => vec![LexerComponent {
            terminal_ids,
            dfa,
            protected_residual: false,
        }],
        _ => inputs,
    }
}

fn remap_component_groups(
    component: &DFA,
    terminal_ids: &[usize],
    total_groups: usize,
) -> DFA {
    debug_assert_eq!(component.num_groups(), terminal_ids.len());
    let mut remapped = DFA::new(component.num_states());
    remapped.ensure_group_capacity(total_groups);

    for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
        remapped.set_group_u8set(
            terminal_id as u32,
            *component.group_id_to_u8set(local_group as u32),
        );
    }

    for (state_index, state) in component.states().iter().enumerate() {
        remapped.set_transitions_from_sorted_entries(
            state_index as u32,
            state.transitions.iter().map(|(byte, &target)| (byte, target)).collect(),
        );
        for &target in &state.epsilon_transitions {
            remapped.add_epsilon_transition(state_index as u32, target);
        }

        let mut finalizers = BitSet::new(total_groups);
        let mut futures = BitSet::new(total_groups);
        for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
            if state.finalizers.contains(local_group) {
                finalizers.set(terminal_id);
            }
            if component
                .possible_future_group_ids(state_index as u32)
                .contains(local_group)
            {
                futures.set(terminal_id);
            }
        }
        remapped.overwrite_state_metadata(state_index as u32, finalizers, futures);
    }

    remapped
}

fn compile_terminal_partitions(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
    adaptive: bool,
) -> DFA {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        || std::env::var_os("GLRMASK_PROFILE_DYNAMIC_TOKENIZER_TOP").is_some();
    let total_started_at = Instant::now();
    assert_eq!(exprs.len(), partitions.len(), "one lexer partition id is required per terminal");
    if let Some(classes) = residual_isolation_classes {
        assert_eq!(
            exprs.len(),
            classes.len(),
            "one residual isolation class entry is required per terminal",
        );
    }
    if let Some(labels) = visible_labels {
        assert_eq!(exprs.len(), labels.len(), "one profile label is required per terminal");
    }
    if exprs.is_empty() {
        return compile_with_plan(build_exclusion_compile_plan(exprs));
    }

    let num_partitions = partitions.iter().copied().collect::<BTreeSet<_>>().len();
    if num_partitions == 1 {
        return compile_with_plan(build_exclusion_compile_plan_with_labels(exprs, visible_labels));
    }

    // Every partition is compiled exactly as declared, independently of the
    // adaptive policy. Adaptive determinization is a second, generic step over
    // the disjoint deterministic components of the combined epsilon-NFA.
    let partition_compile_started_at = Instant::now();
    let mut components = compile_partition_components(
        exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
    );
    let partition_compile_ms = partition_compile_started_at.elapsed().as_secs_f64() * 1000.0;

    let adaptive_started_at = Instant::now();
    if adaptive {
        components =
            adaptively_determinize_components(components, adaptive_lexer_state_limit());
    }
    let adaptive_ms = adaptive_started_at.elapsed().as_secs_f64() * 1000.0;

    if let [component] = components.as_slice() {
        return remap_component_groups(&component.dfa, &component.terminal_ids, exprs.len());
    }

    let combine_started_at = Instant::now();
    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(exprs.len());
    let mut root_futures = BitSet::new(exprs.len());

    for batch in components {
        for local_group in batch.dfa.possible_future_group_ids(0).iter() {
            root_futures.set(batch.terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(batch.dfa, &batch.terminal_ids);
        combined.add_epsilon_transition(0, offset);
    }
    let combine_ms = combine_started_at.elapsed().as_secs_f64() * 1000.0;

    let futures_started_at = Instant::now();
    // Components are disjoint below a new epsilon-only root. Their strict
    // possible-future sets remain exact after local->global terminal remapping.
    // The root's strict futures are exactly the union of the component start
    // states' strict futures; no generic epsilon fixpoint is needed.
    combined.set_possible_future_group_ids(0, root_futures);
    let futures_ms = futures_started_at.elapsed().as_secs_f64() * 1000.0;
    if profile {
        eprintln!(
            "[glrmask/profile][tokenizer] partitioned_build terminals={} partitions={} partition_compile_ms={:.3} adaptive_ms={:.3} combine_ms={:.3} futures_ms={:.3} total_ms={:.3}",
            exprs.len(),
            num_partitions,
            partition_compile_ms,
            adaptive_ms,
            combine_ms,
            futures_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    combined
}

fn combine_synthesized_component_pairs_under_epsilon_root(
    components: &[LexerComponentPair],
    total_groups: usize,
) -> (DFA, Vec<u32>) {
    let total_states = 1usize
        + components
            .iter()
            .map(|component| component.synthesized.num_states())
            .sum::<usize>();
    let mut combined = DFA::new(total_states);
    combined.ensure_group_capacity(total_groups);
    let mut root_futures = BitSet::new(total_groups);
    let mut offsets = Vec::with_capacity(components.len());

    let mut offset = 1u32;
    for component_pair in components {
        offsets.push(offset);
        let terminal_ids = &component_pair.terminal_ids;
        let component = &component_pair.synthesized;
        debug_assert_eq!(component.num_groups(), terminal_ids.len());
        combined.add_epsilon_transition(0, offset);

        for (local_group, &terminal_id) in terminal_ids.iter().enumerate() {
            combined.set_group_u8set(
                terminal_id as u32,
                *component.group_id_to_u8set(local_group as u32),
            );
        }
        for local_group in component.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }

        for (state_index, state) in component.states().iter().enumerate() {
            let mapped_state = offset + state_index as u32;
            combined.set_transitions_from_sorted_entries(
                mapped_state,
                state
                    .transitions
                    .iter()
                    .map(|(byte, &target)| (byte, offset + target))
                    .collect(),
            );
            for &target in &state.epsilon_transitions {
                combined.add_epsilon_transition(mapped_state, offset + target);
            }

            let mut finalizers = BitSet::new(total_groups);
            let mut futures = BitSet::new(total_groups);
            for local_group in state.finalizers.iter() {
                finalizers.set(terminal_ids[local_group]);
            }
            for local_group in component.possible_future_group_ids(state_index as u32).iter() {
                futures.set(terminal_ids[local_group]);
            }
            combined.overwrite_state_metadata(mapped_state, finalizers, futures);
        }
        offset += component.num_states() as u32;
    }
    combined.set_possible_future_group_ids(0, root_futures);
    (combined, offsets)
}

fn compose_partitioned_component_state_maps(
    components: &[LexerComponentPair],
    synthesized_offsets: &[u32],
) -> Option<Vec<u32>> {
    if components.len() != synthesized_offsets.len() {
        return None;
    }
    let full_state_count = 1usize.checked_add(
        components
            .iter()
            .map(|component| component.full.num_states())
            .sum::<usize>(),
    )?;
    let mut full_to_synthesized = Vec::with_capacity(full_state_count);
    full_to_synthesized.push(0);
    let mut expected_synthesized_offset = 1u32;

    for (component, &synthesized_offset) in components.iter().zip(synthesized_offsets) {
        if synthesized_offset != expected_synthesized_offset
            || component.full_to_synthesized.len() != component.full.num_states()
            || component
                .full_to_synthesized
                .iter()
                .any(|&state| state as usize >= component.synthesized.num_states())
        {
            return None;
        }
        full_to_synthesized.extend(
            component
                .full_to_synthesized
                .iter()
                .map(|&state| synthesized_offset.checked_add(state))
                .collect::<Option<Vec<_>>>()?,
        );
        expected_synthesized_offset = expected_synthesized_offset
            .checked_add(component.synthesized.num_states() as u32)?;
    }

    (full_to_synthesized.len() == full_state_count).then_some(full_to_synthesized)
}

fn combine_lexer_components_under_epsilon_root(
    components: Vec<LexerComponent>,
    total_groups: usize,
) -> DFA {
    let mut combined = DFA::new(1);
    combined.ensure_group_capacity(total_groups);
    let mut root_futures = BitSet::new(total_groups);

    for lexer_component in components {
        let terminal_ids = lexer_component.terminal_ids;
        let component = lexer_component.dfa;
        debug_assert_eq!(component.num_groups(), terminal_ids.len());
        for local_group in component.possible_future_group_ids(0).iter() {
            root_futures.set(terminal_ids[local_group]);
        }
        let offset = combined.append_rebased_component(component, &terminal_ids);
        combined.add_epsilon_transition(0, offset);
    }
    combined.set_possible_future_group_ids(0, root_futures);
    combined
}

pub struct ExtractedDispatchComponent {
    pub terminal_ids: Vec<usize>,
    pub source_states: Vec<u32>,
    pub dfa: DFA,
}

pub struct PrecompiledFurtherSynthesisPairs {
    pairs: Mutex<BTreeMap<usize, CompiledTerminalExpressionPair>>,
    pub build_ms: f64,
}

impl PrecompiledFurtherSynthesisPairs {
    fn take(&self, terminal: usize) -> Option<CompiledTerminalExpressionPair> {
        self.pairs
            .lock()
            .expect("precompiled synthesis-pair cache poisoned")
            .remove(&terminal)
    }
}

pub fn precompile_further_synthesis_pairs(
    source_expressions: &[Expr],
    synthesized_expressions: &[Expr],
    protected_terminal_ids: &[u32],
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<Arc<PrecompiledFurtherSynthesisPairs>> {
    if source_expressions.len() != synthesized_expressions.len() {
        return None;
    }
    let protected = protected_terminal_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let changed = source_expressions
        .iter()
        .zip(synthesized_expressions)
        .enumerate()
        .filter_map(|(terminal, (source, synthesized))| {
            (source != synthesized).then_some(terminal)
        })
        .collect::<Vec<_>>();
    if changed.is_empty()
        || changed
            .iter()
            .any(|&terminal| !protected.contains(&(terminal as u32)))
    {
        return None;
    }
    let started_at = Instant::now();
    let pairs = changed
        .par_iter()
        .map(|&terminal| {
            compile_terminal_expression_pair_with_structural_map_and_proof(
                &source_expressions[terminal],
                &synthesized_expressions[terminal],
                vocab,
                repeat_horizons,
                max_token_len,
                relevant_bytes,
                StructuralPairProof::VocabularyTokenQuotient,
            )
            .map(|pair| (terminal, pair))
        })
        .collect::<Option<BTreeMap<_, _>>>()?;
    Some(Arc::new(PrecompiledFurtherSynthesisPairs {
        pairs: Mutex::new(pairs),
        build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    }))
}

fn extract_dispatch_component_from_states(
    tokenizer: &Tokenizer,
    root: u32,
    mut source_states: Vec<u32>,
) -> Option<ExtractedDispatchComponent> {
    source_states.sort_unstable();
    let root_position = source_states.iter().position(|&state| state == root)?;
    source_states.swap(0, root_position);

    let terminal_ids = tokenizer
        .dfa
        .possible_future_group_ids(root)
        .iter()
        .collect::<Vec<_>>();
    if terminal_ids.is_empty() {
        return None;
    }
    let mut local_group_by_terminal = vec![usize::MAX; tokenizer.num_terminals as usize];
    for (local_group, &terminal) in terminal_ids.iter().enumerate() {
        *local_group_by_terminal.get_mut(terminal)? = local_group;
    }
    let mut local_state_by_source = vec![u32::MAX; tokenizer.dfa.num_states()];
    for (local_state, &source_state) in source_states.iter().enumerate() {
        local_state_by_source[source_state as usize] = local_state as u32;
    }

    let mut dfa = DFA::new(source_states.len());
    dfa.ensure_group_capacity(terminal_ids.len());
    for (local_group, &terminal) in terminal_ids.iter().enumerate() {
        dfa.set_group_u8set(
            local_group as u32,
            *tokenizer.dfa.group_id_to_u8set(terminal as u32),
        );
    }
    for (local_state, &source_state) in source_states.iter().enumerate() {
        let source_dfa_state = tokenizer.dfa.states().get(source_state as usize)?;
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| {
                let target = *local_state_by_source.get(target as usize)?;
                (target != u32::MAX).then_some((byte, target))
            })
            .collect::<Option<Vec<_>>>()?;
        dfa.set_transitions_from_sorted_entries(local_state as u32, transitions);
        for &target in &source_dfa_state.epsilon_transitions {
            let target = *local_state_by_source.get(target as usize)?;
            if target == u32::MAX {
                return None;
            }
            dfa.add_epsilon_transition(local_state as u32, target);
        }

        let mut finalizers = BitSet::new(terminal_ids.len());
        let mut futures = BitSet::new(terminal_ids.len());
        for terminal in tokenizer.dfa.finalizers(source_state).iter() {
            let local_group = *local_group_by_terminal.get(terminal)?;
            if local_group == usize::MAX {
                return None;
            }
            finalizers.set(local_group);
        }
        for terminal in tokenizer
            .dfa
            .possible_future_group_ids(source_state)
            .iter()
        {
            let local_group = *local_group_by_terminal.get(terminal)?;
            if local_group == usize::MAX {
                return None;
            }
            futures.set(local_group);
        }
        dfa.overwrite_state_metadata(local_state as u32, finalizers, futures);
    }
    Some(ExtractedDispatchComponent {
        terminal_ids,
        source_states,
        dfa,
    })
}

pub fn extract_dispatch_components(
    tokenizer: &Tokenizer,
) -> Option<Vec<ExtractedDispatchComponent>> {
    let roots = tokenizer.deterministic_dispatch_roots()?;
    let components = tokenizer.disjoint_dispatch_components()?;
    if roots.len() != components.len() {
        return None;
    }
    roots
        .iter()
        .copied()
        .zip(components)
        .map(|(root, source_states)| {
            extract_dispatch_component_from_states(tokenizer, root, source_states)
        })
        .collect()
}

/// Extract singleton dispatch components together with externally entered
/// residual states that were cloned outside the live root-reachable component
/// graph by protected residual synthesis.
pub fn extract_augmented_singleton_dispatch_components(
    tokenizer: &Tokenizer,
) -> Option<Vec<ExtractedDispatchComponent>> {
    let profile = std::env::var_os("GLRMASK_DEFINITION_PHYSICAL_REPORT").is_some();
    let roots = tokenizer.deterministic_dispatch_roots()?.to_vec();
    let components = tokenizer.disjoint_dispatch_components()?;
    if roots.len() != components.len() {
        return None;
    }
    let component_count = components.len();
    let state_count = tokenizer.dfa.num_states();
    let mut owner = vec![usize::MAX; state_count];
    owner[tokenizer.initial_state_id() as usize] = roots.len();
    let mut terminal_to_singleton_component = vec![usize::MAX; tokenizer.num_terminals as usize];
    let mut singleton_terminal_by_component = vec![usize::MAX; roots.len()];
    let mut singleton_components = 0usize;
    for (component, (&root, states)) in roots.iter().zip(&components).enumerate() {
        for &state in states {
            owner[state as usize] = component;
        }
        let terminals = tokenizer
            .dfa
            .possible_future_group_ids(root)
            .iter()
            .collect::<Vec<_>>();
        if terminals.len() == 1 {
            terminal_to_singleton_component[terminals[0]] = component;
            singleton_terminal_by_component[component] = terminals[0];
            singleton_components += 1;
        }
    }

    // Appended protected residuals retain exact final/future terminal metadata.
    // Use that label as the initial owner, then admit only states whose complete
    // outgoing graph remains inside the same singleton component.
    let initially_unowned = owner.iter().filter(|&&owner| owner == usize::MAX).count();
    let mut label_assigned = 0usize;
    for state in 0..state_count {
        if owner[state] != usize::MAX {
            continue;
        }
        let mut terminals = tokenizer.dfa.finalizers(state as u32).iter().collect::<Vec<_>>();
        terminals.extend(tokenizer.dfa.possible_future_group_ids(state as u32).iter());
        terminals.sort_unstable();
        terminals.dedup();
        if terminals.len() == 1 {
            let component = terminal_to_singleton_component[terminals[0]];
            if component != usize::MAX {
                owner[state] = component;
                label_assigned += 1;
            }
        }
    }
    let mut forward_propagated = 0usize;
    let mut queue = VecDeque::<u32>::new();
    for state in 0..state_count {
        if owner[state] < roots.len()
            && singleton_terminal_by_component[owner[state]] != usize::MAX
            && !components[owner[state]].contains(&(state as u32))
        {
            queue.push_back(state as u32);
        }
    }
    while let Some(state) = queue.pop_front() {
        let component = owner[state as usize];
        let terminal = singleton_terminal_by_component[component];
        for target in tokenizer.dfa.states()[state as usize]
            .transitions
            .iter()
            .map(|(_, &target)| target)
            .chain(
                tokenizer.dfa.states()[state as usize]
                    .epsilon_transitions
                    .iter()
                    .copied(),
            )
        {
            let target_slot = &mut owner[target as usize];
            if *target_slot == component {
                continue;
            }
            if *target_slot != usize::MAX {
                continue;
            }
            let mut target_terminals = tokenizer
                .dfa
                .finalizers(target)
                .iter()
                .collect::<Vec<_>>();
            target_terminals.extend(tokenizer.dfa.possible_future_group_ids(target).iter());
            target_terminals.sort_unstable();
            target_terminals.dedup();
            if target_terminals.is_empty()
                || (target_terminals.len() == 1 && target_terminals[0] == terminal)
            {
                *target_slot = component;
                forward_propagated += 1;
                queue.push_back(target);
            }
        }
    }
    let mut propagated = 0usize;
    let mut changed = true;
    while changed {
        changed = false;
        for state in 0..state_count {
            if owner[state] != usize::MAX {
                continue;
            }
            let dfa_state = &tokenizer.dfa.states()[state];
            let mut candidate = usize::MAX;
            let mut valid = true;
            for target in dfa_state
                .transitions
                .iter()
                .map(|(_, &target)| target)
                .chain(dfa_state.epsilon_transitions.iter().copied())
            {
                let target_owner = owner[target as usize];
                if target_owner == usize::MAX || target_owner == roots.len() {
                    valid = false;
                    break;
                }
                if candidate == usize::MAX {
                    candidate = target_owner;
                } else if candidate != target_owner {
                    valid = false;
                    break;
                }
            }
            if valid
                && candidate != usize::MAX
                && terminal_to_singleton_component.contains(&candidate)
            {
                owner[state] = candidate;
                changed = true;
                propagated += 1;
            }
        }
    }

    let mut augmented = components;
    for state in 0..state_count {
        let component = owner[state];
        if component < roots.len() && !augmented[component].contains(&(state as u32)) {
            augmented[component].push(state as u32);
        }
    }
    let remaining_unowned = owner.iter().filter(|&&owner| owner == usize::MAX).count();
    let mut build_failures = 0usize;
    let extracted = roots
        .into_iter()
        .zip(augmented)
        .filter_map(|(root, states)| {
            let terminals = tokenizer
                .dfa
                .possible_future_group_ids(root)
                .iter()
                .collect::<Vec<_>>();
            if terminals.len() != 1 {
                return None;
            }
            match extract_dispatch_component_from_states(tokenizer, root, states) {
                Some(component) => Some(component),
                None => {
                    build_failures += 1;
                    None
                }
            }
        })
        .collect::<Vec<_>>();
    if profile {
        eprintln!(
            "[glrmask/profile][definition_dispatch_augmented] states={} roots={} singleton_components={} initially_unowned={} label_assigned={} forward_propagated={} propagated={} remaining_unowned={} extracted_singletons={} build_failures={} augmented_states={}",
            state_count,
            component_count,
            singleton_components,
            initially_unowned,
            label_assigned,
            forward_propagated,
            propagated,
            remaining_unowned,
            extracted.len(),
            build_failures,
            extracted.iter().map(|component| component.source_states.len()).sum::<usize>(),
        );
    }
    Some(extracted)
}

fn augment_component_from_verified_prefix(
    source: &DFA,
    rebuilt: &DFA,
    synthesized: &mut DFA,
    rebuilt_to_synthesized: &[u32],
) -> Option<Vec<u32>> {
    if source.num_groups() != rebuilt.num_groups()
        || source.num_groups() != synthesized.num_groups()
        || rebuilt_to_synthesized.len() != rebuilt.num_states()
        || rebuilt.num_states() > source.num_states()
    {
        return None;
    }
    for state in 0..rebuilt.num_states() {
        let state = state as u32;
        if source.finalizers(state) != rebuilt.finalizers(state)
            || source.possible_future_group_ids(state)
                != rebuilt.possible_future_group_ids(state)
            || source.states()[state as usize].epsilon_transitions
                != rebuilt.states()[state as usize].epsilon_transitions
            || source.states()[state as usize].transitions
                != rebuilt.states()[state as usize].transitions
        {
            return None;
        }
    }

    let mut source_to_synthesized = vec![u32::MAX; source.num_states()];
    source_to_synthesized[..rebuilt.num_states()].copy_from_slice(rebuilt_to_synthesized);
    for source_state in rebuilt.num_states()..source.num_states() {
        source_to_synthesized[source_state] = synthesized.add_state();
    }
    for source_state in rebuilt.num_states()..source.num_states() {
        let source_state_u32 = source_state as u32;
        let target_state = source_to_synthesized[source_state];
        let source_dfa_state = &source.states()[source_state];
        if !source_dfa_state.epsilon_transitions.is_empty() {
            return None;
        }
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| {
                Some((byte, *source_to_synthesized.get(target as usize)?))
            })
            .collect::<Option<Vec<_>>>()?;
        if transitions.iter().any(|&(_, target)| target == u32::MAX) {
            return None;
        }
        synthesized.set_transitions_from_sorted_entries(target_state, transitions);
        synthesized.overwrite_state_metadata(
            target_state,
            source.finalizers(source_state_u32).clone(),
            source.possible_future_group_ids(source_state_u32).clone(),
        );
    }
    Some(source_to_synthesized)
}

/// Further synthesize protected singleton components of an already-built
/// partitioned tokenizer without recompiling its ordinary components.
///
/// Every unchanged dispatch component is cloned exactly. Changed protected
/// components use the same structural terminal-pair proof as the global
/// synthesis, then clone any externally-entered residual states appended to
/// the actual source component. The returned map covers the actual source
/// tokenizer's raw-state domain.
pub fn compile_further_synthesized_tokenizer_with_structural_map(
    source: &Tokenizer,
    source_expressions: &[Expr],
    synthesized_expressions: &[Expr],
    protected_terminal_ids: &[u32],
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    precompiled_pairs: Option<&PrecompiledFurtherSynthesisPairs>,
) -> Option<(Tokenizer, Vec<u32>)> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let started_at = profile.then(Instant::now);
    let reject = |stage: &str| {
        if profile {
            eprintln!(
                "[glrmask/profile][partition_local_component_reuse] selected=false stage={} elapsed_ms={:.3}",
                stage,
                started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        None
    };
    if source_expressions.len() != synthesized_expressions.len()
        || source_expressions.len() != source.num_terminals as usize
    {
        return reject("input_shape");
    }
    let protected = protected_terminal_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let changed = source_expressions
        .iter()
        .zip(synthesized_expressions)
        .map(|(source, synthesized)| source != synthesized)
        .collect::<Vec<_>>();
    if !changed.iter().any(|&changed| changed) {
        return reject("unchanged");
    }

    let Some(extracted) = extract_dispatch_components(source) else {
        return reject("extract_dispatch");
    };
    let mut output_components = Vec::<LexerComponent>::with_capacity(extracted.len());
    let mut component_maps = Vec::<(Vec<u32>, Vec<u32>)>::with_capacity(extracted.len());
    let mut handled_changed = vec![false; changed.len()];
    let mut effective_synthesized_expressions = synthesized_expressions.to_vec();

    for component in extracted {
        let changed_terminals = component
            .terminal_ids
            .iter()
            .copied()
            .filter(|&terminal| changed.get(terminal).copied().unwrap_or(false))
            .collect::<Vec<_>>();
        if changed_terminals.is_empty() {
            let state_count = component.dfa.num_states() as u32;
            component_maps.push((
                component.source_states,
                (0..state_count).collect::<Vec<_>>(),
            ));
            output_components.push(LexerComponent {
                terminal_ids: component.terminal_ids,
                dfa: component.dfa,
                protected_residual: false,
            });
            continue;
        }
        if changed_terminals.len() != 1
            || component.terminal_ids.len() != 1
            || !protected.contains(&(changed_terminals[0] as u32))
        {
            return reject("changed_component_shape");
        }
        let terminal = changed_terminals[0];
        handled_changed[terminal] = true;
        let Some(pair) = precompiled_pairs
            .and_then(|pairs| pairs.take(terminal))
            .or_else(|| {
                compile_terminal_expression_pair_with_structural_map_and_proof(
                    &source_expressions[terminal],
                    &synthesized_expressions[terminal],
                    vocab,
                    repeat_horizons,
                    max_token_len,
                    relevant_bytes,
                    StructuralPairProof::VocabularyTokenQuotient,
                )
            })
        else {
            return reject("protected_pair");
        };
        effective_synthesized_expressions[terminal] = pair.synthesized_expression.clone();
        let (mut synthesized, synthesized_nullable) =
            isolate_component_nullable_start(pair.synthesized.dfa, 1);
        let (rebuilt, rebuilt_nullable) = isolate_component_nullable_start(pair.full.dfa, 1);
        if !synthesized_nullable.is_empty() || !rebuilt_nullable.is_empty() {
            return reject("protected_nullable");
        }
        let Some(source_to_synthesized) = augment_component_from_verified_prefix(
            &component.dfa,
            &rebuilt,
            &mut synthesized,
            &pair.full_to_synthesized,
        ) else {
            if profile {
                eprintln!(
                    "[glrmask/profile][partition_local_component_reuse_detail] terminal={} source_states={} rebuilt_states={} synthesized_states={}",
                    terminal,
                    component.dfa.num_states(),
                    rebuilt.num_states(),
                    synthesized.num_states(),
                );
            }
            return reject("protected_prefix");
        };
        // This lane consumes complete entries from `vocab`, not arbitrary
        // one-byte continuations from every raw source state. The structural
        // pair above is therefore certified in the vocabulary-token quotient
        // coordinate. Requiring a raw byte homomorphism here is strictly
        // stronger and rejects valid bounded-repeat quotients.
        component_maps.push((component.source_states, source_to_synthesized));
        output_components.push(LexerComponent {
            terminal_ids: component.terminal_ids,
            dfa: synthesized,
            protected_residual: true,
        });
    }
    if changed
        .iter()
        .enumerate()
        .any(|(terminal, &changed)| changed && !handled_changed[terminal])
    {
        return reject("unhandled_changed");
    }

    let mut source_to_synthesized = vec![u32::MAX; source.dfa.num_states()];
    source_to_synthesized[0] = 0;
    let mut offset = 1u32;
    for (component, (source_states, local_map)) in output_components.iter().zip(&component_maps) {
        if source_states.len() != local_map.len() {
            return reject("component_map_length");
        }
        for (&source_state, &local_state) in source_states.iter().zip(local_map) {
            source_to_synthesized[source_state as usize] = offset + local_state;
        }
        offset = offset.checked_add(component.dfa.num_states() as u32)?;
    }
    let mut dfa = combine_lexer_components_under_epsilon_root(
        output_components,
        source_expressions.len(),
    );
    // Nullable-start isolation and prior structural augmentation may retain
    // unreachable raw states outside the live dispatch components. They remain
    // part of the compiler's raw-state coordinate, so clone them after the live
    // component layout is fixed and redirect every edge through the completed
    // source-state map.
    let extra_source_states = source_to_synthesized
        .iter()
        .enumerate()
        .filter_map(|(state, &mapped)| (mapped == u32::MAX).then_some(state))
        .collect::<Vec<_>>();
    for &source_state in &extra_source_states {
        source_to_synthesized[source_state] = dfa.add_state();
    }
    for source_state in extra_source_states {
        let target_state = source_to_synthesized[source_state];
        let source_dfa_state = &source.dfa.states()[source_state];
        let transitions = source_dfa_state
            .transitions
            .iter()
            .map(|(byte, &target)| (byte, source_to_synthesized[target as usize]))
            .collect::<Vec<_>>();
        dfa.set_transitions_from_sorted_entries(target_state, transitions);
        for &target in &source_dfa_state.epsilon_transitions {
            dfa.add_epsilon_transition(target_state, source_to_synthesized[target as usize]);
        }
        dfa.overwrite_state_metadata(
            target_state,
            source.dfa.finalizers(source_state as u32).clone(),
            source
                .dfa
                .possible_future_group_ids(source_state as u32)
                .clone(),
        );
    }
    let tokenizer = Regex { dfa }.into_tokenizer(
        source_expressions.len() as u32,
        Some(Arc::from(
            effective_synthesized_expressions.into_boxed_slice(),
        )),
    );
    if profile {
        eprintln!(
            "[glrmask/profile][partition_local_component_reuse] selected=true source_states={} synthesized_states={} elapsed_ms={:.3}",
            source.dfa.num_states(),
            tokenizer.dfa.num_states(),
            started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some((tokenizer, source_to_synthesized))
}

/// Compile independently protected terminal partitions as exact/synthesized
/// pairs while compiling every unchanged partition only once. The returned
/// state map is structural: the global epsilon root maps to the global root,
/// unchanged components map identically, and protected components use their
/// certified local product maps. Adaptive prefix determinization remains
/// available for the ordinary components but never crosses a protected
/// residual coordinate.
fn prepare_partitioned_expression_pair_with_proof(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
    proof: StructuralPairProof,
) -> Option<PreparedPartitionedExpressionPair> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    if full_exprs.len() != synthesized_exprs.len()
        || full_exprs.len() != partitions.len()
        || full_exprs.len() != residual_isolation_classes.len()
        || visible_labels.is_some_and(|labels| labels.len() != full_exprs.len())
        || full_exprs.is_empty()
    {
        return None;
    }

    let changed = full_exprs
        .iter()
        .zip(synthesized_exprs)
        .zip(residual_isolation_classes)
        .map(|((full, synthesized), isolation_class)| {
            isolation_class.is_some() && full != synthesized
        })
        .collect::<Vec<_>>();
    if !changed.iter().any(|&changed| changed) {
        return None;
    }

    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for (terminal, &partition) in partitions.iter().enumerate() {
        grouped.entry(partition).or_default().push(terminal);
    }

    // Only ordinary terminals participate in repeated-subexpression sharing.
    // Replacing changed expressions with epsilon avoids compiling their large
    // exact bodies merely to prepare a cache they cannot use.
    let mut ordinary_exprs = synthesized_exprs.to_vec();
    for (terminal, &is_changed) in changed.iter().enumerate() {
        if is_changed {
            ordinary_exprs[terminal] = Expr::Epsilon;
        }
    }
    let rewritten_ordinary = materialize_repeated_subexpression_dfas(&ordinary_exprs);
    let ordinary_exprs = rewritten_ordinary.as_deref().unwrap_or(&ordinary_exprs);
    let ordinary_groups = grouped
        .iter()
        .filter_map(|(&partition, terminals)| {
            terminals
                .iter()
                .all(|&terminal| !changed[terminal])
                .then_some((partition, terminals.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let shared_duplicates = shared_duplicate_nested_group_op_cache(ordinary_exprs, &ordinary_groups);
    if let Some(shared_duplicates) = &shared_duplicates {
        prewarm_shared_duplicate_nested_group_ops(shared_duplicates);
    }

    let compiled = grouped
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_partition, terminal_ids)| {
            let changed_terminals = terminal_ids
                .iter()
                .copied()
                .filter(|&terminal| changed[terminal])
                .collect::<Vec<_>>();
            if changed_terminals.is_empty() {
                let dfa = compile_terminal_ids_with_shared_duplicate_cache(
                    ordinary_exprs,
                    visible_labels,
                    &terminal_ids,
                    shared_duplicates.as_ref(),
                );
                let minimize_started_at = profile.then(Instant::now);
                let before_states = dfa.num_states();
                let dfa = dfa.minimize();
                if profile {
                    let labels = visible_labels
                        .map(|labels| {
                            terminal_ids
                                .iter()
                                .filter_map(|&terminal| labels.get(terminal))
                                .take(4)
                                .cloned()
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    eprintln!(
                        "[glrmask/profile][tokenizer] paired_ordinary_minimize terminals={} terminal_ids={:?} labels={:?} states_before={} states_after={} elapsed_ms={:.3}",
                        terminal_ids.len(),
                        terminal_ids,
                        labels,
                        before_states,
                        dfa.num_states(),
                        minimize_started_at
                            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    );
                }
                let state_count = dfa.num_states() as u32;
                return Some((
                    LexerComponentPair {
                        terminal_ids,
                        synthesized: dfa.clone(),
                        full: DeferredDfa::Ready(dfa),
                        full_to_synthesized: (0..state_count).collect(),
                        protected_residual: false,
                    },
                    None,
                ));
            }

            if changed_terminals.len() != 1 || terminal_ids.len() != 1 {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=non_singleton_partition terminals={:?} changed={:?}",
                        terminal_ids,
                        changed_terminals,
                    );
                }
                return None;
            }
            let terminal = changed_terminals[0];
            residual_isolation_classes[terminal]?;
            let pair = prepare_terminal_expression_pair_with_structural_map_inner(
                &full_exprs[terminal],
                &synthesized_exprs[terminal],
                vocab,
                repeat_horizons,
                max_token_len,
                relevant_bytes,
                true,
                proof,
            );
            let Some(pair) = pair else {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=local_pair_failed terminal={} full_expr={:?} synthesized_expr={:?}",
                        terminal,
                        expr_profile_summary(&full_exprs[terminal]),
                        expr_profile_summary(&synthesized_exprs[terminal]),
                    );
                }
                return None;
            };
            let (synthesized, synthesized_nullable) =
                isolate_component_nullable_start(pair.synthesized.dfa, terminal_ids.len());
            let mut full = pair.full;
            let full_nullable = if full.initial_is_nonnullable_without_epsilon() {
                BTreeSet::new()
            } else {
                let (dfa, nullable) =
                    isolate_component_nullable_start(full.finish(), terminal_ids.len());
                full = DeferredDfa::Ready(dfa);
                nullable
            };
            if !synthesized_nullable.is_empty() || !full_nullable.is_empty() {
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] structural_partition_pair_rejected reason=nullable_protected_terminal terminal={} synthesized_nullable={:?} full_nullable={:?}",
                        terminal, synthesized_nullable, full_nullable,
                    );
                }
                return None;
            }
            Some((
                LexerComponentPair {
                    terminal_ids,
                    synthesized,
                    full,
                    full_to_synthesized: pair.full_to_synthesized,
                    protected_residual: true,
                },
                Some((terminal, pair.synthesized_expression)),
            ))
        })
        .collect::<Option<Vec<_>>>()?;

    let mut protected = Vec::new();
    let mut ordinary = Vec::new();
    let mut effective_synthesized_expressions = synthesized_exprs.to_vec();
    for (pair, effective_expression) in compiled {
        if let Some((terminal, expression)) = effective_expression {
            effective_synthesized_expressions[terminal] = expression;
        }
        if pair.protected_residual {
            protected.push(pair);
        } else {
            let dfa = match pair.full {
                DeferredDfa::Ready(dfa) => dfa,
                DeferredDfa::ReadyCompressed { .. } | DeferredDfa::DenseBinary(_) => {
                    unreachable!("ordinary lexer components are never deferred")
                }
            };
            ordinary.push(LexerComponent {
                terminal_ids: pair.terminal_ids,
                dfa,
                protected_residual: false,
            });
        }
    }

    let ordinary = if adaptive && ordinary.len() >= 2 {
        adaptively_determinize_components(ordinary, adaptive_lexer_state_limit())
    } else {
        ordinary
    };
    let mut pairs = ordinary
        .into_iter()
        .map(|component| {
            let terminal_ids = component.terminal_ids;
            let (dfa, _) = isolate_component_nullable_start(component.dfa, terminal_ids.len());
            let state_count = dfa.num_states() as u32;
            LexerComponentPair {
                terminal_ids,
                synthesized: dfa.clone(),
                full: DeferredDfa::Ready(dfa),
                full_to_synthesized: (0..state_count).collect(),
                protected_residual: false,
            }
        })
        .collect::<Vec<_>>();
    pairs.append(&mut protected);
    pairs.sort_unstable_by_key(|pair| {
        pair.terminal_ids.first().copied().unwrap_or(usize::MAX)
    });

    let (synthesized, synthesized_offsets) =
        combine_synthesized_component_pairs_under_epsilon_root(&pairs, full_exprs.len());
    let full_to_synthesized =
        compose_partitioned_component_state_maps(&pairs, &synthesized_offsets)?;

    Some(PreparedPartitionedExpressionPair {
        synthesized: Regex { dfa: synthesized },
        full: DeferredPartitionedRegex {
            components: pairs,
            total_groups: full_exprs.len(),
        },
        full_to_synthesized,
        synthesized_expressions: effective_synthesized_expressions,
    })
}

pub fn prepare_partitioned_expression_pair_with_structural_map(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<PreparedPartitionedExpressionPair> {
    prepare_partitioned_expression_pair_with_proof(
        full_exprs,
        synthesized_exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        StructuralPairProof::RawHomomorphism,
    )
}

pub fn prepare_partitioned_expression_pair_with_vocabulary_token_quotient(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<PreparedPartitionedExpressionPair> {
    prepare_partitioned_expression_pair_with_proof(
        full_exprs,
        synthesized_exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        StructuralPairProof::VocabularyTokenQuotient,
    )
}

pub fn compile_partitioned_expression_pair_with_structural_map(
    full_exprs: &[Expr],
    synthesized_exprs: &[Expr],
    visible_labels: Option<&[String]>,
    partitions: &[u32],
    residual_isolation_classes: &[Option<u32>],
    adaptive: bool,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<CompiledPartitionedExpressionPair> {
    prepare_partitioned_expression_pair_with_structural_map(
        full_exprs,
        synthesized_exprs,
        visible_labels,
        partitions,
        residual_isolation_classes,
        adaptive,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
    )
    .map(PreparedPartitionedExpressionPair::finish_full)
}

fn product_component_state_flags(component: &ProductComponent, state: u32) -> (bool, bool) {
    match component {
        ProductComponent::Materialized(dfa)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => (
            dfa.finalizers(state).contains(0),
            dfa.possible_future_group_ids(state).contains(0),
        ),
        ProductComponent::VirtualFixedSequence {
            byte_sets,
            suffix_live,
        } => {
            let position = state as usize;
            (
                position == byte_sets.len(),
                position < byte_sets.len()
                    && suffix_live.get(position).copied().unwrap_or(false),
            )
        }
        ProductComponent::VirtualBoundedRepeat { base_dfa, min, max } => {
            let base_state_count = base_dfa.num_states() as u32;
            let copy_count = state / base_state_count;
            let base_state = state % base_state_count;
            (base_state == 0 && copy_count >= *min, copy_count < *max)
        }
    }
}

fn product_state_metadata_with_layout(
    components: &[ProductComponent],
    coordinate_groups: &[Vec<usize>],
    num_groups: usize,
    state_tuple: &ProductStateTuple,
) -> (BitSet, BitSet) {
    let mut finalizers = BitSet::new(num_groups);
    let mut future = BitSet::new(num_groups);

    for &(coordinate_id, state) in state_tuple {
        let coordinate = coordinate_id as usize;
        let (accepting, can_continue) =
            product_component_state_flags(&components[coordinate], state);
        for &group in &coordinate_groups[coordinate] {
            if accepting {
                finalizers.set(group);
            }
            if can_continue {
                future.set(group);
            }
        }
    }

    (finalizers, future)
}

fn product_state_single_visible_finalizer_with_layout(
    components: &[ProductComponent],
    coordinate_groups: &[Vec<usize>],
    num_groups: usize,
    state_tuple: &ProductStateTuple,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> BitSet {
    let mut accepting = vec![false; num_groups];
    for &(coordinate_id, state) in state_tuple {
        let coordinate = coordinate_id as usize;
        let is_accepting = product_component_state_flags(&components[coordinate], state).0;
        for &group in &coordinate_groups[coordinate] {
            accepting[group] = is_accepting;
        }
    }

    let mut visible_accepting = accepting.first().copied().unwrap_or(false);
    if visible_accepting
        && exclusions
            .get(&0)
            .is_some_and(|blocked| blocked.iter().any(|&group| accepting[group as usize]))
    {
        visible_accepting = false;
    }
    if visible_accepting
        && intersections
            .get(&0)
            .is_some_and(|required| required.iter().any(|&group| !accepting[group as usize]))
    {
        visible_accepting = false;
    }

    let mut finalizers = BitSet::new(1);
    if visible_accepting {
        finalizers.set(0);
    }
    finalizers
}

fn identity_product_coordinate_groups(num_groups: usize) -> Vec<Vec<usize>> {
    (0..num_groups).map(|group| vec![group]).collect()
}

fn product_state_metadata(
    components: &[ProductComponent],
    state_tuple: &ProductStateTuple,
) -> (BitSet, BitSet) {
    let coordinate_groups = identity_product_coordinate_groups(components.len());
    product_state_metadata_with_layout(
        components,
        &coordinate_groups,
        components.len(),
        state_tuple,
    )
}

fn product_state_single_visible_finalizer(
    components: &[ProductComponent],
    state_tuple: &ProductStateTuple,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> BitSet {
    let coordinate_groups = identity_product_coordinate_groups(components.len());
    product_state_single_visible_finalizer_with_layout(
        components,
        &coordinate_groups,
        components.len(),
        state_tuple,
        exclusions,
        intersections,
    )
}

fn set_single_group_futures_from_class_graph(
    dfa: &mut DFA,
    class_transitions: &[Vec<(u8, u32)>],
) {
    let states = dfa.num_states();
    debug_assert_eq!(states, class_transitions.len());
    let mut predecessor_counts = vec![0u32; states];
    let mut seen_target_epoch = vec![0u32; states];
    let mut epoch = 1u32;
    let mut edge_count = 0usize;
    for transitions in class_transitions {
        for &(_, target) in transitions {
            if seen_target_epoch[target as usize] == epoch {
                continue;
            }
            seen_target_epoch[target as usize] = epoch;
            predecessor_counts[target as usize] += 1;
            edge_count += 1;
        }
        epoch = epoch.checked_add(1).expect("product state epoch overflow");
    }

    let mut offsets = vec![0usize; states + 1];
    for state in 0..states {
        offsets[state + 1] = offsets[state] + predecessor_counts[state] as usize;
    }
    debug_assert_eq!(offsets[states], edge_count);
    let mut write_offsets = offsets[..states].to_vec();
    let mut predecessors = vec![0u32; edge_count];
    seen_target_epoch.fill(0);
    epoch = 1;
    for (source, transitions) in class_transitions.iter().enumerate() {
        for &(_, target) in transitions {
            if seen_target_epoch[target as usize] == epoch {
                continue;
            }
            seen_target_epoch[target as usize] = epoch;
            let slot = &mut write_offsets[target as usize];
            predecessors[*slot] = source as u32;
            *slot += 1;
        }
        epoch = epoch.checked_add(1).expect("product state epoch overflow");
    }

    let mut can_reach_final = vec![false; states];
    let mut queue = VecDeque::<u32>::new();
    for state in 0..states {
        if dfa.finalizers(state as u32).contains(0) {
            can_reach_final[state] = true;
            queue.push_back(state as u32);
        }
    }
    while let Some(target) = queue.pop_front() {
        let target = target as usize;
        for &source in &predecessors[offsets[target]..offsets[target + 1]] {
            if !can_reach_final[source as usize] {
                can_reach_final[source as usize] = true;
                queue.push_back(source);
            }
        }
    }

    for (source, transitions) in class_transitions.iter().enumerate() {
        let mut future = BitSet::new(1);
        if transitions
            .iter()
            .any(|&(_, target)| can_reach_final[target as usize])
        {
            future.set(0);
        }
        dfa.set_possible_future_group_ids(source as u32, future);
    }
}

/// Compute single-group futures from a class graph without first rebuilding a
/// CSR predecessor graph. The deferred dense intersection already stores one
/// compact transition row per state. A linked reverse graph can be assembled
/// in one pass, and the reverse reachability walk itself identifies exactly
/// the states with an outgoing edge to a final-reachable state.
fn compute_single_group_futures_from_class_graph_csr(
    accepting: &[bool],
    transition_offsets: &[u32],
    transition_targets: &[u32],
    cooperative_yield_interval: usize,
) -> Vec<bool> {
    let states = accepting.len();
    debug_assert_eq!(transition_offsets.len(), states + 1);

    let mut predecessor_heads = vec![u32::MAX; states];
    let mut predecessor_sources = Vec::<u32>::new();
    let mut predecessor_next = Vec::<u32>::new();
    let mut seen_target_source = vec![u32::MAX; states];

    for source in 0..states {
        if cooperative_yield_interval != 0 && source % cooperative_yield_interval == 0 {
            let _ = rayon::yield_now();
        }
        let source_u32 = source as u32;
        let row_start = transition_offsets[source] as usize;
        let row_end = transition_offsets[source + 1] as usize;
        for &target in &transition_targets[row_start..row_end] {
            let target = target as usize;
            if seen_target_source[target] == source_u32 {
                continue;
            }
            seen_target_source[target] = source_u32;
            let edge = predecessor_sources.len() as u32;
            predecessor_sources.push(source_u32);
            predecessor_next.push(predecessor_heads[target]);
            predecessor_heads[target] = edge;
        }
    }

    let mut can_reach_final = vec![false; states];
    let mut has_future = vec![false; states];
    let mut queue = VecDeque::<u32>::new();
    for (state, &is_accepting) in accepting.iter().enumerate() {
        if is_accepting {
            can_reach_final[state] = true;
            queue.push_back(state as u32);
        }
    }

    let mut processed_targets = 0usize;
    while let Some(target) = queue.pop_front() {
        if cooperative_yield_interval != 0
            && processed_targets % cooperative_yield_interval == 0
        {
            let _ = rayon::yield_now();
        }
        processed_targets += 1;
        let mut edge = predecessor_heads[target as usize];
        while edge != u32::MAX {
            let source = predecessor_sources[edge as usize] as usize;
            has_future[source] = true;
            if !can_reach_final[source] {
                can_reach_final[source] = true;
                queue.push_back(source as u32);
            }
            edge = predecessor_next[edge as usize];
        }
    }
    has_future
}

fn set_single_group_futures_from_class_graph_csr(
    dfa: &mut DFA,
    transition_offsets: &[u32],
    transition_targets: &[u32],
    cooperative_yield_interval: usize,
) {
    let accepting = (0..dfa.num_states())
        .map(|state| dfa.finalizers(state as u32).contains(0))
        .collect::<Vec<_>>();
    let has_future = compute_single_group_futures_from_class_graph_csr(
        &accepting,
        transition_offsets,
        transition_targets,
        cooperative_yield_interval,
    );
    for (state, &future) in has_future.iter().enumerate() {
        let mut future_groups = BitSet::new(1);
        if future {
            future_groups.set(0);
        }
        dfa.set_possible_future_group_ids(state as u32, future_groups);
    }
}

fn explicit_dead_sink_state(dfa: &DFA) -> Option<u32> {
    for (state_id, state) in dfa.states().iter().enumerate() {
        if !state.finalizers.is_empty() {
            continue;
        }

        let mut transition_count = 0usize;
        let mut loops_to_self = true;
        for (_, &target) in state.transitions.iter() {
            transition_count += 1;
            if target != state_id as u32 {
                loops_to_self = false;
                break;
            }
        }

        if loops_to_self && transition_count == 256 {
            return Some(state_id as u32);
        }
    }

    None
}

fn build_prefixed_bounded_repeat_with_suffix_dfa_with_options(
    parts: &[Expr],
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
    build_prefixed_bounded_repeat_with_suffix_dfa_with_options_and_cache(
        parts,
        preserve_coordinates,
        None,
    )
}

fn expr_is_epsilon_only(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::Seq(parts) => parts.iter().all(expr_is_epsilon_only),
        Expr::Shared(inner) => expr_is_epsilon_only(inner),
        Expr::U8Class(_)
        | Expr::Dfa(_)
        | Expr::Choice(_)
        | Expr::Exclude { .. }
        | Expr::Intersect { .. }
        | Expr::Repeat { .. } => false,
    }
}

fn optional_choice_non_epsilon(expr: &Expr) -> Option<&Expr> {
    let options = match expr {
        Expr::Shared(inner) => return optional_choice_non_epsilon(inner),
        Expr::Choice(options) if options.len() == 2 => options,
        _ => return None,
    };

    if expr_is_epsilon_only(&options[0]) {
        Some(&options[1])
    } else if expr_is_epsilon_only(&options[1]) {
        Some(&options[0])
    } else {
        None
    }
}

fn optional_tail_parts(expr: &Expr) -> Option<Vec<Expr>> {
    let non_epsilon = optional_choice_non_epsilon(expr)?;
    match non_epsilon {
        Expr::Shared(inner) => optional_tail_parts(inner).or_else(|| Some(vec![inner.as_ref().clone()])),
        Expr::Seq(parts) => Some(parts.clone()),
        other => Some(vec![other.clone()]),
    }
}

fn mark_state_accepting(dfa: &mut DFA, state_id: u32) {
    dfa.ensure_group_capacity(1);

    let mut finalizers = dfa.finalizers(state_id).clone();
    finalizers.set(0);
    let mut future = dfa.possible_future_group_ids(state_id).clone();
    future.set(0);
    dfa.overwrite_state_metadata(state_id, finalizers, future);
}

fn collect_fixed_sequence_byte_sets(expr: &Expr, byte_sets: &mut Vec<U8Set>) -> bool {
    match expr {
        Expr::U8Seq(bytes) => {
            byte_sets.extend(bytes.iter().copied().map(U8Set::from_byte));
            true
        }
        Expr::U8Class(bytes) => {
            byte_sets.push(*bytes);
            true
        }
        Expr::Seq(parts) => parts
            .iter()
            .all(|part| collect_fixed_sequence_byte_sets(part, byte_sets)),
        Expr::Shared(inner) => collect_fixed_sequence_byte_sets(inner, byte_sets),
        Expr::Epsilon => true,
        Expr::Dfa(_)
        | Expr::Choice(_)
        | Expr::Exclude { .. }
        | Expr::Intersect { .. }
        | Expr::Repeat { .. } => false,
    }
}

fn build_fixed_sequence_dfa(expr: &Expr) -> Option<DFA> {
    let mut byte_sets = Vec::new();
    collect_fixed_sequence_byte_sets(expr, &mut byte_sets).then(|| {
        let mut dfa = DFA::new(byte_sets.len() + 1);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        for (state, bytes) in byte_sets.into_iter().enumerate() {
            dfa.set_transitions_from_sorted_entries(
                state as u32,
                bytes
                    .iter()
                    .map(|byte| (byte, state as u32 + 1))
                    .collect(),
            );
        }
        let final_state = (dfa.num_states() - 1) as u32;
        mark_state_accepting(&mut dfa, final_state);
        dfa.recompute_possible_futures();
        dfa
    })
}

const MAX_FINITE_LITERAL_ALTERNATIVES: usize = 4096;
const MAX_FINITE_LITERAL_BYTES: usize = 1 << 20;

fn collect_finite_literal_language(expr: &Expr) -> Option<Vec<Vec<u8>>> {
    match unwrap_shared(expr) {
        Expr::U8Seq(bytes) => Some(vec![bytes.clone()]),
        Expr::Epsilon => Some(vec![Vec::new()]),
        Expr::Choice(options) => {
            let mut out = Vec::new();
            let mut total_bytes = 0usize;
            for option in options {
                let alternatives = collect_finite_literal_language(option)?;
                if out.len().saturating_add(alternatives.len()) > MAX_FINITE_LITERAL_ALTERNATIVES {
                    return None;
                }
                for literal in alternatives {
                    total_bytes = total_bytes.saturating_add(literal.len());
                    if total_bytes > MAX_FINITE_LITERAL_BYTES {
                        return None;
                    }
                    out.push(literal);
                }
            }
            Some(out)
        }
        Expr::Seq(parts) => {
            let mut prefixes = vec![Vec::<u8>::new()];
            for part in parts {
                let suffixes = collect_finite_literal_language(part)?;
                let next_count = prefixes.len().checked_mul(suffixes.len())?;
                if next_count > MAX_FINITE_LITERAL_ALTERNATIVES {
                    return None;
                }
                let mut next = Vec::with_capacity(next_count);
                let mut total_bytes = 0usize;
                for prefix in &prefixes {
                    for suffix in &suffixes {
                        let len = prefix.len().checked_add(suffix.len())?;
                        total_bytes = total_bytes.checked_add(len)?;
                        if total_bytes > MAX_FINITE_LITERAL_BYTES {
                            return None;
                        }
                        let mut literal = Vec::with_capacity(len);
                        literal.extend_from_slice(prefix);
                        literal.extend_from_slice(suffix);
                        next.push(literal);
                    }
                }
                prefixes = next;
            }
            Some(prefixes)
        }
        _ => None,
    }
}

fn build_finite_literal_language_dfa(expr: &Expr) -> Option<DFA> {
    if std::env::var_os("GLRMASK_DISABLE_FINITE_LITERAL_TRIE_DIRECT").is_some() {
        return None;
    }
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile_timing.then(Instant::now);
    let literals = collect_finite_literal_language(expr)?;
    if literals.len() <= 1 {
        return None;
    }

    #[derive(Default)]
    struct Node {
        children: BTreeMap<u8, usize>,
        accepting: bool,
    }

    let mut trie = vec![Node::default()];
    for literal in &literals {
        let mut node = 0usize;
        for &byte in literal {
            let next = if let Some(&next) = trie[node].children.get(&byte) {
                next
            } else {
                let next = trie.len();
                trie.push(Node::default());
                trie[node].children.insert(byte, next);
                next
            };
            node = next;
        }
        trie[node].accepting = true;
    }

    let trie_nodes = trie.len();
    let mut dfa = DFA::new(trie_nodes);
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, expr_u8set(expr));
    for (node, trie_node) in trie.iter().enumerate() {
        dfa.set_transitions_from_sorted_entries(
            node as u32,
            trie_node
                .children
                .iter()
                .map(|(&byte, &target)| (byte, target as u32))
                .collect(),
        );
        if trie_node.accepting {
            mark_state_accepting(&mut dfa, node as u32);
        }
    }
    dfa.recompute_possible_futures();
    let minimized = dfa.minimize_owned_reachable();
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] finite_literal_trie_direct alternatives={} trie_nodes={} final_states={} final_transitions={} total_ms={:.3}",
            literals.len(),
            trie_nodes,
            minimized.num_states(),
            dfa_transition_count(&minimized),
            started_at.unwrap().elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(minimized)
}


/// Exact union of already-deterministic single-group DFAs and fixed byte
/// strings. General Choice compilation lowers every DFA arm back into an NFA
/// and redeterminizes the union. Here the determinized state is simply the
/// sparse tuple of still-live DFA states plus one literal-trie DFA state.
/// The product stays on the global byte-class alphabet through minimization and
/// expands classes back to bytes only for the final compact automaton.
fn build_dfas_with_literal_choice(expr: &Expr) -> Option<DFA> {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let total_started_at = profile_timing.then(Instant::now);
    if std::env::var_os("GLRMASK_DISABLE_DFA_LITERAL_CHOICE_DIRECT").is_some() {
        return None;
    }
    let Expr::Choice(options) = unwrap_shared(expr) else {
        return None;
    };
    let mut dfas = Vec::<Arc<DFA>>::new();
    let mut literals = Vec::<&[u8]>::new();
    for option in options {
        match unwrap_shared(option) {
            Expr::Dfa(dfa) => dfas.push(Arc::clone(dfa)),
            Expr::U8Seq(bytes) => literals.push(bytes.as_slice()),
            Expr::Epsilon => literals.push(&[]),
            _ => return None,
        }
    }
    if dfas.is_empty() || literals.is_empty() {
        return None;
    }
    if dfas
        .iter()
        .any(|dfa| dfa.num_groups() != 1 || dfa.has_epsilon_transitions())
    {
        return None;
    }

    #[derive(Default)]
    struct LiteralTrieNode {
        children: BTreeMap<u8, usize>,
        accepting: bool,
    }
    let mut trie = vec![LiteralTrieNode::default()];
    for literal in literals {
        let mut node = 0usize;
        for &byte in literal {
            let next = if let Some(&next) = trie[node].children.get(&byte) {
                next
            } else {
                let next = trie.len();
                trie.push(LiteralTrieNode::default());
                trie[node].children.insert(byte, next);
                next
            };
            node = next;
        }
        trie[node].accepting = true;
    }
    let trie_nodes = trie.len();

    let mut literal_dfa = DFA::new(trie.len());
    literal_dfa.ensure_group_capacity(1);
    literal_dfa.set_group_u8set(0, expr_u8set(expr));
    for (node, trie_node) in trie.iter().enumerate() {
        literal_dfa.set_transitions_from_sorted_entries(
            node as u32,
            trie_node
                .children
                .iter()
                .map(|(&byte, &target)| (byte, target as u32))
                .collect(),
        );
        let mut finalizers = BitSet::new(1);
        if trie_node.accepting {
            finalizers.set(0);
        }
        literal_dfa.overwrite_state_metadata(node as u32, finalizers, BitSet::new(1));
    }

    let mut components = dfas.iter().map(Arc::as_ref).collect::<Vec<_>>();
    components.push(&literal_dfa);
    let class_started_at = profile_timing.then(Instant::now);
    let (class_map, class_members) = compute_dfa_byte_equivalence_classes(&components);
    let class_ms = class_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if class_members.len() > u8::MAX as usize + 1 {
        return None;
    }
    let class_count = class_members.len();
    let component_class_transitions = components
        .iter()
        .map(|dfa| {
            build_product_class_transitions_for_dfa(dfa, &class_map)
                .into_iter()
                .map(|row| {
                    let mut dense = vec![u32::MAX; class_count];
                    for (class, target) in row {
                        dense[class as usize] = target;
                    }
                    dense
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dead_states = components
        .iter()
        .map(|dfa| explicit_dead_sink_state(dfa))
        .collect::<Vec<_>>();
    let start = std::iter::repeat_n(0u32, components.len())
        .collect::<SmallVec<[u32; 4]>>();
    let mut result = DFA::new(1);
    result.ensure_group_capacity(1);
    result.set_group_u8set(0, expr_u8set(expr));
    let mut state_by_tuple = FxHashMap::<SmallVec<[u32; 4]>, u32>::default();
    state_by_tuple.insert(start.clone(), 0);
    let mut queue = VecDeque::from([(0u32, start)]);
    const MAX_DIRECT_CHOICE_STATES: usize = 32_768;
    const MAX_DIRECT_CHOICE_CLASS_TRANSITIONS: usize = 2_000_000;
    let mut transition_count = 0usize;
    let construct_started_at = profile_timing.then(Instant::now);

    while let Some((result_state, tuple)) = queue.pop_front() {
        let accepting = components.iter().enumerate().any(|(index, dfa)| {
            let state = tuple[index];
            state != u32::MAX && dfa.finalizers(state).contains(0)
        });
        let mut finalizers = BitSet::new(1);
        if accepting {
            finalizers.set(0);
        }
        result.overwrite_state_metadata(result_state, finalizers, BitSet::new(1));

        let mut transitions = Vec::<(u8, u32)>::new();
        for class in 0..class_count {
            let mut next = SmallVec::<[u32; 4]>::with_capacity(components.len());
            let mut any_live = false;
            for (index, dfa) in components.iter().enumerate() {
                let state = tuple[index];
                if state == u32::MAX {
                    next.push(u32::MAX);
                    continue;
                }
                let target = component_class_transitions[index][state as usize][class];
                if target != u32::MAX && dead_states[index] != Some(target) {
                    next.push(target);
                    any_live = true;
                } else {
                    next.push(u32::MAX);
                }
            }
            if !any_live {
                continue;
            }
            let target = if let Some(&target) = state_by_tuple.get(&next) {
                target
            } else {
                if result.num_states() >= MAX_DIRECT_CHOICE_STATES {
                    return None;
                }
                let target = result.add_state();
                state_by_tuple.insert(next.clone(), target);
                queue.push_back((target, next));
                target
            };
            transitions.push((class as u8, target));
        }
        transition_count = transition_count.saturating_add(transitions.len());
        if transition_count > MAX_DIRECT_CHOICE_CLASS_TRANSITIONS {
            return None;
        }
        result.set_transitions_from_sorted_entries(result_state, transitions);
    }
    let construct_ms = construct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let raw_states = result.num_states();
    let raw_class_transitions = dfa_transition_count(&result);
    let futures_started_at = profile_timing.then(Instant::now);
    result.recompute_possible_futures();
    let futures_ms = futures_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let minimize_started_at = profile_timing.then(Instant::now);
    let mut minimized = result.minimize_owned_reachable();
    let minimize_ms = minimize_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let final_class_transitions = dfa_transition_count(&minimized);
    let expand_started_at = profile_timing.then(Instant::now);
    expand_direct_expression_graph_classes(&mut minimized, &class_members);
    let expand_ms = expand_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] dfa_literal_choice_direct dfa_arms={} trie_nodes={} classes={} raw_states={} raw_class_transitions={} final_states={} final_class_transitions={} final_byte_transitions={} class_ms={:.3} construct_ms={:.3} futures_ms={:.3} minimize_ms={:.3} expand_ms={:.3} total_ms={:.3}",
            dfas.len(),
            trie_nodes,
            class_members.len(),
            raw_states,
            raw_class_transitions,
            minimized.num_states(),
            final_class_transitions,
            dfa_transition_count(&minimized),
            class_ms,
            construct_ms,
            futures_ms,
            minimize_ms,
            expand_ms,
            total_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Some(minimized)
}

fn compile_product_component_dfa_direct_with_options_and_cache(
    expr: &Expr,
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> Option<(DFA, bool)> {
    match expr {
        Expr::Shared(inner) => {
            compile_product_component_dfa_direct_with_options_and_cache(
                inner,
                preserve_coordinates,
                cache,
            )
        }
        Expr::U8Seq(bytes) => {
            let mut dfa = DFA::new(bytes.len() + 1);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, U8Set::from_bytes(bytes));
            for (index, &byte) in bytes.iter().enumerate() {
                dfa.add_transition(index as u32, byte, index as u32 + 1);
            }
            mark_state_accepting(&mut dfa, bytes.len() as u32);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::U8Class(bytes) => {
            let mut dfa = DFA::new(2);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, *bytes);
            dfa.set_transitions_from_sorted_entries(
                0,
                bytes.iter().map(|byte| (byte, 1)).collect(),
            );
            mark_state_accepting(&mut dfa, 1);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::Epsilon => {
            let mut dfa = DFA::new(1);
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, U8Set::empty());
            mark_state_accepting(&mut dfa, 0);
            dfa.recompute_possible_futures();
            Some((dfa, false))
        }
        Expr::Dfa(dfa) => Some((dfa.as_ref().clone(), true)),
        Expr::Choice(_) => {
            if let Some(dfa) = build_dfas_with_literal_choice(expr) {
                return Some((dfa, false));
            }
            let non_epsilon = optional_choice_non_epsilon(expr)?;
            let (mut dfa, needs_future_recompute) =
                compile_product_component_dfa_direct_with_options_and_cache(
                    non_epsilon,
                    preserve_coordinates,
                    cache,
                )?;
            mark_state_accepting(&mut dfa, 0);
            Some((dfa, needs_future_recompute))
        }
        Expr::Repeat {
            expr,
            min,
            max: Some(max),
        } => build_bounded_repeat_dfa_with_cache(expr, *min, *max, cache)
            .map(|dfa| (dfa, false)),
        Expr::Seq(parts) => build_fixed_sequence_dfa(expr)
            .map(|dfa| (dfa, false))
            .or_else(|| build_finite_literal_language_dfa(expr).map(|dfa| (dfa, false)))
            .or_else(|| build_bounded_repeat_with_suffix_dfa_with_cache(parts, cache))
            .or_else(|| {
                build_bounded_repeat_with_regex_suffix_with_options_and_cache(
                    parts,
                    preserve_coordinates,
                    cache,
                )
            })
            .or_else(|| {
                build_prefixed_bounded_repeat_with_suffix_dfa_with_options_and_cache(
                    parts,
                    preserve_coordinates,
                    cache,
                )
            }),
        _ => None,
    }
}

fn compile_product_component_dfa_direct_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
) -> Option<(DFA, bool)> {
    compile_product_component_dfa_direct_with_options_and_cache(
        expr,
        preserve_coordinates,
        None,
    )
}

fn compile_product_component_dfa_direct(expr: &Expr) -> Option<(DFA, bool)> {
    compile_product_component_dfa_direct_with_options(expr, false)
}

fn compile_product_component_dfa(expr: &Expr) -> DFA {
    compile_with_plan(build_exclusion_compile_plan(std::slice::from_ref(expr)))
}

/// Compile one logical terminal definition through the full expression plan.
///
/// Unlike `compile_single_expr_dfa`, this lowers nested exclusions and
/// intersections before NFA/DFA construction. Definition-level analyses must
/// use this entry point because terminal expressions are not guaranteed to be
/// primitive product leaves.
pub fn compile_terminal_expr_dfa(expr: &Expr) -> DFA {
    compile_product_component_dfa(expr)
}

fn compile_product_component_materialized_dfa_with_options_and_cache(
    expr: &Expr,
    preserve_coordinates: bool,
    cache: Option<&RepeatBaseDfaCache>,
) -> DFA {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let direct_started_at = profile_timing.then(Instant::now);
    let dfa = if let Some((mut dfa, needs_future_recompute)) =
        compile_product_component_dfa_direct_with_options_and_cache(
            expr,
            preserve_coordinates,
            cache,
        )
    {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, expr_u8set(expr));
        if needs_future_recompute {
            dfa.recompute_possible_futures();
        }
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] product_component_path path=direct states={} transitions={} direct_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
            );
        }
        dfa
    } else {
        let direct_ms = direct_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        // The generic NFA->DFA path may still carry conservative future-group
        // metadata until the ordinary single-expression compile/minimize pass
        // recomputes it. Product construction consumes that metadata directly,
        // so retain the exact old fallback rather than using the raw DFA here.
        let fallback_started_at = profile_timing.then(Instant::now);
        let dfa = compile_product_component_dfa(expr);
        if profile_timing {
            eprintln!(
                "[glrmask/profile][tokenizer] product_component_path path=fallback states={} transitions={} direct_attempt_ms={:.3} fallback_ms={:.3}",
                dfa.num_states(),
                dfa_transition_count(&dfa),
                direct_ms,
                fallback_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        dfa
    };
    dfa
}

fn compile_product_component_materialized_dfa_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
) -> DFA {
    compile_product_component_materialized_dfa_with_options_and_cache(
        expr,
        preserve_coordinates,
        None,
    )
}

fn compile_product_component_materialized_dfa(expr: &Expr) -> DFA {
    compile_product_component_materialized_dfa_with_options(expr, false)
}

#[derive(Clone)]
enum ProductComponent {
    Materialized(Arc<DFA>),
    MaterializedZeroMinRepeatSuffix {
        dfa: Arc<DFA>,
        trace: Arc<ZeroMinRepeatSuffixComponentTrace>,
    },
    VirtualFixedSequence {
        byte_sets: Arc<[U8Set]>,
        suffix_live: Arc<[bool]>,
    },
    VirtualBoundedRepeat {
        base_dfa: Arc<DFA>,
        min: u32,
        max: u32,
    },
}

impl ProductComponent {
    fn materialized_dfa(&self) -> Option<&DFA> {
        match self {
            Self::Materialized(dfa) | Self::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                Some(dfa)
            }
            Self::VirtualFixedSequence { .. } | Self::VirtualBoundedRepeat { .. } => None,
        }
    }

    fn materialized_dfa_arc(&self) -> Option<Arc<DFA>> {
        match self {
            Self::Materialized(dfa) | Self::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                Some(Arc::clone(dfa))
            }
            Self::VirtualFixedSequence { .. } | Self::VirtualBoundedRepeat { .. } => None,
        }
    }

    /// Exact one-terminal DFA carrying the same residual state numbering used
    /// by this product coordinate.  This lets compile-time residual analyses
    /// consume the normal fast virtual fixed-sequence representation without
    /// forcing tokenizer construction to materialize those components.
    fn terminal_residual_dfa_arc(&self) -> Option<Arc<DFA>> {
        match self {
            Self::Materialized(dfa) | Self::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                Some(Arc::clone(dfa))
            }
            Self::VirtualFixedSequence { byte_sets, .. } => {
                let mut dfa = DFA::new(byte_sets.len() + 1);
                dfa.ensure_group_capacity(1);
                let mut support = U8Set::empty();
                for (position, bytes) in byte_sets.iter().enumerate() {
                    support = support.union(bytes);
                    dfa.set_transitions_from_sorted_entries(
                        position as u32,
                        bytes
                            .iter()
                            .map(|byte| (byte, position as u32 + 1))
                            .collect(),
                    );
                }
                dfa.set_group_u8set(0, support);
                mark_state_accepting(&mut dfa, byte_sets.len() as u32);
                dfa.recompute_possible_futures();
                Some(Arc::new(dfa))
            }
            Self::VirtualBoundedRepeat { .. } => None,
        }
    }

    fn zero_min_repeat_suffix_trace(&self) -> Option<Arc<ZeroMinRepeatSuffixComponentTrace>> {
        match self {
            Self::MaterializedZeroMinRepeatSuffix { trace, .. } => Some(Arc::clone(trace)),
            _ => None,
        }
    }
}

struct ProductBuildTrace {
    components: Vec<ProductComponent>,
    /// Logical product groups represented by each physical product coordinate.
    /// Normally this is the identity layout. Trace-enabled vocab partition
    /// builds may collapse duplicate coordinates exactly as the ordinary lexer
    /// does, while retaining this fan-out map for terminal residual recovery.
    coordinate_groups: Vec<Vec<usize>>,
    state_tuples: ProductStateTuples,
    state_lookup: ProductStateLookup,
    direct_single_visible_group: bool,
}

enum ProductStateTuples {
    Generic(Vec<ProductStateTuple>),
    DenseBinary(Vec<(u32, u32)>),
}

impl ProductStateTuples {
    fn len(&self) -> usize {
        match self {
            Self::Generic(tuples) => tuples.len(),
            Self::DenseBinary(pairs) => pairs.len(),
        }
    }

    fn tuple(&self, state: usize) -> ProductStateTuple {
        match self {
            Self::Generic(tuples) => tuples[state].clone(),
            Self::DenseBinary(pairs) => {
                let (left, right) = pairs[state];
                let mut tuple = ProductStateTuple::new();
                tuple.push((0, left));
                tuple.push((1, right));
                tuple
            }
        }
    }

    fn push(&mut self, tuple: ProductStateTuple) {
        match self {
            Self::Generic(tuples) => tuples.push(tuple),
            Self::DenseBinary(pairs)
                if tuple.len() == 2 && tuple[0].0 == 0 && tuple[1].0 == 1 =>
            {
                pairs.push((tuple[0].1, tuple[1].1));
            }
            Self::DenseBinary(pairs) => {
                let mut tuples = Vec::with_capacity(pairs.len() + 1);
                tuples.extend(pairs.drain(..).map(|(left, right)| {
                    let mut tuple = ProductStateTuple::new();
                    tuple.push((0, left));
                    tuple.push((1, right));
                    tuple
                }));
                tuples.push(tuple);
                *self = Self::Generic(tuples);
            }
        }
    }
}

enum ProductStateLookup {
    Hash(FxHashMap<ProductStateTuple, u32>),
    Fingerprint {
        state_by_fingerprint: FxHashMap<u64, u32>,
        canonical_tuples: Vec<ProductStateTuple>,
        overflow: FxHashMap<ProductStateTuple, u32>,
    },
    DenseBinary {
        right_states: usize,
        state_by_pair: Vec<u32>,
        overflow: FxHashMap<ProductStateTuple, u32>,
    },
}

impl ProductStateLookup {
    fn dense_pair_index(tuple: &ProductStateTuple, right_states: usize) -> Option<usize> {
        if tuple.len() != 2 || tuple[0].0 != 0 || tuple[1].0 != 1 {
            return None;
        }
        (tuple[0].1 as usize)
            .checked_mul(right_states)?
            .checked_add(tuple[1].1 as usize)
    }

    fn get(&self, tuple: &ProductStateTuple) -> Option<u32> {
        match self {
            Self::Hash(states) => states.get(tuple).copied(),
            Self::Fingerprint {
                state_by_fingerprint,
                canonical_tuples,
                overflow,
            } => {
                let fingerprint = product_state_fingerprint(tuple);
                state_by_fingerprint
                    .get(&fingerprint)
                    .copied()
                    .filter(|&state| canonical_tuples[state as usize] == *tuple)
                    .or_else(|| overflow.get(tuple).copied())
            }
            Self::DenseBinary {
                right_states,
                state_by_pair,
                overflow,
            } => Self::dense_pair_index(tuple, *right_states)
                .and_then(|index| state_by_pair.get(index).copied())
                .filter(|&state| state != u32::MAX)
                .or_else(|| overflow.get(tuple).copied()),
        }
    }

    fn insert(&mut self, tuple: ProductStateTuple, state: u32) {
        match self {
            Self::Hash(states) => {
                states.insert(tuple, state);
            }
            Self::Fingerprint {
                state_by_fingerprint,
                canonical_tuples,
                overflow,
            } => {
                debug_assert_eq!(canonical_tuples.len(), state as usize);
                let fingerprint = product_state_fingerprint(&tuple);
                if state_by_fingerprint.contains_key(&fingerprint) {
                    overflow.insert(tuple.clone(), state);
                } else {
                    state_by_fingerprint.insert(fingerprint, state);
                }
                canonical_tuples.push(tuple);
            }
            Self::DenseBinary {
                right_states,
                state_by_pair,
                overflow,
            } => {
                if let Some(index) = Self::dense_pair_index(&tuple, *right_states)
                    && let Some(slot) = state_by_pair.get_mut(index)
                {
                    *slot = state;
                } else {
                    overflow.insert(tuple, state);
                }
            }
        }
    }
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

struct ZeroMinRepeatSuffixComponentTrace {
    dfa: Arc<DFA>,
    prefix_len: usize,
    body_dfa: DFA,
    suffix_dfa: DFA,
    max: usize,
    tail_states: Vec<ZeroMinRepeatSuffixState>,
    tail_state_by_key: FxHashMap<ZeroMinRepeatSuffixState, u32>,
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

fn zero_min_repeat_suffix_component_trace(
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

struct ZeroMinRepeatSuffixStateMap {
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

fn zero_min_repeat_suffix_state_map(
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

struct DirectBoundedSuffixStateMap {
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
    fn primary(&self) -> &[u32] {
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

enum ProductComponentStateMap {
    Fixed(Vec<u32>),
    Layered(DirectBoundedSuffixStateMap),
    Dominance(ZeroMinRepeatSuffixStateMap),
    Vocabulary(CertifiedVocabularyExactStateCandidates),
}

impl ProductComponentStateMap {
    fn primary(&self) -> &[u32] {
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
fn direct_bounded_suffix_state_map(
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

fn augment_product_dfa_from_seed_tuples(
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

struct PreparedTerminalExpressionPair {
    synthesized: Regex,
    full: DeferredDfa,
    full_to_synthesized: Vec<u32>,
    synthesized_expression: Expr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructuralPairProof {
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

fn prepare_terminal_expression_pair_with_structural_map(
    full_expression: &Expr,
    synthesized_expression: &Expr,
    vocab: &Vocab,
    repeat_horizons: &VocabularyRepeatHorizonCache,
    max_token_len: usize,
    relevant_bytes: &[u8],
) -> Option<PreparedTerminalExpressionPair> {
    prepare_terminal_expression_pair_with_structural_map_inner(
        full_expression,
        synthesized_expression,
        vocab,
        repeat_horizons,
        max_token_len,
        relevant_bytes,
        true,
        StructuralPairProof::RawHomomorphism,
    )
}

fn prepare_terminal_expression_pair_with_structural_map_inner(
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

fn compile_terminal_expression_pair_with_structural_map_and_proof(
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

enum ProductComponentClassTransitions {
    Materialized(Vec<Vec<(u8, u32)>>),
    VirtualFixedSequence(Vec<Vec<(u8, u32)>>),
    VirtualBoundedRepeat(Vec<Vec<(u8, u32)>>),
}

fn product_class_transition_entry_count(
    transitions: &[ProductComponentClassTransitions],
) -> usize {
    transitions
        .iter()
        .map(|component| match component {
            ProductComponentClassTransitions::Materialized(rows)
            | ProductComponentClassTransitions::VirtualFixedSequence(rows)
            | ProductComponentClassTransitions::VirtualBoundedRepeat(rows) => {
                rows.iter().map(Vec::len).sum::<usize>()
            }
        })
        .sum()
}

impl ProductComponent {
    fn partition_dfa(&self) -> &DFA {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => dfa,
            ProductComponent::VirtualFixedSequence { .. } => {
                panic!("virtual fixed sequences do not materialize a partition DFA")
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => base_dfa,
        }
    }

    fn profile_state_count(&self) -> usize {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => dfa.num_states(),
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => byte_sets.len() + 1,
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => base_dfa.num_states(),
        }
    }

    fn profile_transition_count(&self) -> usize {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                dfa_transition_count(dfa)
            }
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => {
                byte_sets.iter().map(U8Set::len).sum()
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => {
                dfa_transition_count(base_dfa)
            }
        }
    }

    fn dead_state(&self) -> Option<u32> {
        match self {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                explicit_dead_sink_state(dfa)
            }
            ProductComponent::VirtualFixedSequence { .. } => None,
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => explicit_dead_sink_state(base_dfa),
        }
    }
}

fn compile_product_component_with_options(
    expr: &Expr,
    preserve_coordinates: bool,
    virtual_fixed_sequences: bool,
    repeat_base_cache: Option<&RepeatBaseDfaCache>,
) -> ProductComponent {
    if preserve_coordinates
        && let Some(trace) = zero_min_repeat_suffix_component_trace(expr)
    {
        let dfa = Arc::clone(&trace.dfa);
        return ProductComponent::MaterializedZeroMinRepeatSuffix {
            dfa,
            trace: Arc::new(trace),
        };
    }
    if !preserve_coordinates && virtual_fixed_sequences {
        let mut byte_sets = Vec::new();
        if collect_fixed_sequence_byte_sets(expr, &mut byte_sets) {
            let mut suffix_live = vec![false; byte_sets.len() + 1];
            suffix_live[byte_sets.len()] = true;
            for position in (0..byte_sets.len()).rev() {
                suffix_live[position] = !byte_sets[position].is_empty() && suffix_live[position + 1];
            }
            return ProductComponent::VirtualFixedSequence {
                byte_sets: Arc::from(byte_sets.into_boxed_slice()),
                suffix_live: Arc::from(suffix_live.into_boxed_slice()),
            };
        }
    }
    match expr {
        Expr::Shared(inner) => compile_product_component_with_options(
            inner,
            preserve_coordinates,
            virtual_fixed_sequences,
            repeat_base_cache,
        ),
        Expr::Repeat {
            expr: repeat_expr,
            min,
            max: Some(max),
        } => {
            if let Some(base_dfa) = cached_direct_bounded_repeat_base_dfa(
                repeat_expr,
                *max,
                repeat_base_cache,
            ) {
                return ProductComponent::VirtualBoundedRepeat {
                    base_dfa,
                    min: *min as u32,
                    max: *max as u32,
                };
            }

            ProductComponent::Materialized(Arc::new(
                compile_product_component_materialized_dfa_with_options_and_cache(
                    expr,
                    preserve_coordinates,
                    repeat_base_cache,
                ),
            ))
        }
        _ => ProductComponent::Materialized(Arc::new(
            compile_product_component_materialized_dfa_with_options_and_cache(
                expr,
                preserve_coordinates,
                repeat_base_cache,
            ),
        )),
    }
}

fn compile_product_component(expr: &Expr) -> ProductComponent {
    compile_product_component_with_options(expr, false, true, None)
}

fn certify_supplied_dfa_state_homomorphism(
    full: &DFA,
    full_compressed: Option<&CompressedTransitionSegment>,
    synthesized: &DFA,
    full_to_synthesized: &[u32],
) -> bool {
    if full.num_groups() != synthesized.num_groups()
        || full_to_synthesized.len() != full.num_states()
        || full_to_synthesized
            .iter()
            .any(|&state| state as usize >= synthesized.num_states())
    {
        return false;
    }

    let diagnostics = std::env::var_os("GLRMASK_PROFILE_SYNTH_CERT").is_some();
    let mut metadata_mismatch_states = 0usize;
    let mut epsilon_mismatch_states = 0usize;
    let mut transition_mismatch_states = 0usize;
    let mut mismatched_synthesized_states = FxHashSet::<u32>::default();
    let synthesized_dense_len = match synthesized.num_states().checked_mul(256) {
        Some(len) => len,
        None => return false,
    };
    let mut synthesized_dense = vec![u32::MAX; synthesized_dense_len];
    let mut synthesized_transition_counts = vec![0usize; synthesized.num_states()];
    for (state, dfa_state) in synthesized.states().iter().enumerate() {
        synthesized_transition_counts[state] = dfa_state.transitions.len();
        let row = &mut synthesized_dense[state * 256..(state + 1) * 256];
        for (byte, &target) in dfa_state.transitions.iter() {
            row[byte as usize] = target;
        }
    }

    let mut mapped_epsilon = Vec::<u32>::new();
    for full_state in 0..full.num_states() as u32 {
        let synthesized_state = full_to_synthesized[full_state as usize];
        if full.finalizers(full_state) != synthesized.finalizers(synthesized_state)
            || full.possible_future_group_ids(full_state)
                != synthesized.possible_future_group_ids(synthesized_state)
        {
            if !diagnostics {
                return false;
            }
            metadata_mismatch_states += 1;
            mismatched_synthesized_states.insert(synthesized_state);
        }

        mapped_epsilon.clear();
        mapped_epsilon.extend(
            full.states()[full_state as usize]
                .epsilon_transitions
                .iter()
                .map(|&target| full_to_synthesized[target as usize]),
        );
        mapped_epsilon.sort_unstable();
        mapped_epsilon.dedup();
        if mapped_epsilon
            != synthesized.states()[synthesized_state as usize].epsilon_transitions
        {
            if !diagnostics {
                return false;
            }
            epsilon_mismatch_states += 1;
            mismatched_synthesized_states.insert(synthesized_state);
        }

        let synthesized_row = &synthesized_dense
            [synthesized_state as usize * 256..(synthesized_state as usize + 1) * 256];
        let full_transition_count = match full_compressed {
            Some(segment) if segment.contains_state(full_state) => {
                segment.transition_count(full_state)
            }
            _ => full.states()[full_state as usize].transitions.len(),
        };
        let mut transition_mismatch = full_transition_count
            != synthesized_transition_counts[synthesized_state as usize];
        let transitions_match = match full_compressed {
            Some(segment) if segment.contains_state(full_state) => segment.transitions_satisfy(
                full_state,
                |byte, target| {
                    full_to_synthesized[target as usize] == synthesized_row[byte as usize]
                },
            ),
            _ => full.states()[full_state as usize]
                .transitions
                .iter()
                .all(|(byte, &target)| {
                    full_to_synthesized[target as usize] == synthesized_row[byte as usize]
                }),
        };
        transition_mismatch |= !transitions_match;
        if transition_mismatch && !diagnostics {
            return false;
        }
        if transition_mismatch {
            transition_mismatch_states += 1;
            mismatched_synthesized_states.insert(synthesized_state);
        }
    }
    if diagnostics {
        eprintln!(
            "[glrmask/profile][synth_cert] full_states={} synthesized_states={} metadata_mismatch_states={} epsilon_mismatch_states={} transition_mismatch_states={} mismatched_synthesized_states={}",
            full.num_states(),
            synthesized.num_states(),
            metadata_mismatch_states,
            epsilon_mismatch_states,
            transition_mismatch_states,
            mismatched_synthesized_states.len(),
        );
    }
    metadata_mismatch_states == 0
        && epsilon_mismatch_states == 0
        && transition_mismatch_states == 0
}

fn deterministic_component_homomorphism_state_map(
    full: &DFA,
    synthesized: &DFA,
) -> Option<Vec<u32>> {
    if full.num_groups() != synthesized.num_groups() {
        return None;
    }
    let mut mapping = vec![u32::MAX; full.num_states()];
    let mut worklist = VecDeque::from([(0u32, 0u32)]);
    mapping[0] = 0;

    while let Some((full_state, synthesized_state)) = worklist.pop_front() {
        if full.finalizers(full_state) != synthesized.finalizers(synthesized_state)
            || full.possible_future_group_ids(full_state)
                != synthesized.possible_future_group_ids(synthesized_state)
        {
            return None;
        }
        for byte in 0u16..=255 {
            let full_target = full.step(full_state, byte as u8);
            let synthesized_target = synthesized.step(synthesized_state, byte as u8);
            match (full_target, synthesized_target) {
                (None, None) => {}
                (Some(full_target), Some(synthesized_target)) => {
                    let slot = &mut mapping[full_target as usize];
                    if *slot == u32::MAX {
                        *slot = synthesized_target;
                        worklist.push_back((full_target, synthesized_target));
                    } else if *slot != synthesized_target {
                        return None;
                    }
                }
                _ => return None,
            }
        }
    }
    mapping.iter().all(|&state| state != u32::MAX).then_some(mapping)
}

fn count_bounded_repeat_base_uses(expr: &Expr, counts: &mut FxHashMap<Expr, usize>) {
    match unwrap_shared(expr) {
        Expr::Repeat {
            expr,
            max: Some(_),
            ..
        } => {
            let body = unwrap_shared(expr);
            *counts.entry(body.clone()).or_default() += 1;
            count_bounded_repeat_base_uses(body, counts);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            for part in parts {
                count_bounded_repeat_base_uses(part, counts);
            }
        }
        Expr::Exclude { expr, exclude } => {
            count_bounded_repeat_base_uses(expr, counts);
            count_bounded_repeat_base_uses(exclude, counts);
        }
        Expr::Intersect { expr, intersect } => {
            count_bounded_repeat_base_uses(expr, counts);
            count_bounded_repeat_base_uses(intersect, counts);
        }
        Expr::Shared(_) => unreachable!("unwrap_shared removes Shared"),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
        Expr::Repeat { max: None, expr, .. } => {
            count_bounded_repeat_base_uses(expr, counts);
        }
    }
}

const LOCAL_SMALL_PRODUCT_MAX_COORDINATES: usize = 32;

fn local_small_product_work_enabled(component_count: usize, local_small_product: bool) -> bool {
    local_small_product && component_count <= LOCAL_SMALL_PRODUCT_MAX_COORDINATES
}

fn build_repeat_base_dfa_cache(
    exprs: &[&Expr],
    local_small_product: bool,
) -> RepeatBaseDfaCache {
    let mut counts = FxHashMap::<Expr, usize>::default();
    for expr in exprs {
        count_bounded_repeat_base_uses(expr, &mut counts);
    }
    let repeated = counts
        .into_iter()
        .filter_map(|(expr, uses)| (uses >= 2).then_some(expr))
        .collect::<Vec<_>>();
    let compile = |expr: Expr| {
        compile_direct_bounded_repeat_base_dfa_unconditionally(&expr)
            .map(|dfa| (expr, Arc::new(dfa)))
    };
    if local_small_product_work_enabled(exprs.len(), local_small_product) {
        repeated.into_iter().filter_map(compile).collect()
    } else {
        repeated.into_par_iter().filter_map(compile).collect()
    }
}

#[derive(Debug)]
struct ProductComponentCompileProfile {
    first_group_index: usize,
    uses: usize,
    compile_ms: f64,
    states: usize,
    transitions: usize,
}

fn compile_product_components_profiled(
    exprs: &[Expr],
    profile_detail: bool,
    preserve_coordinates: bool,
    virtual_fixed_sequences: bool,
    local_small_product: bool,
) -> (
    Vec<ProductComponent>,
    usize,
    Option<Vec<ProductComponentCompileProfile>>,
    Vec<usize>,
) {
    let mut unique_exprs = Vec::<&Expr>::new();
    let mut unique_first_group_indices = Vec::<usize>::new();
    let mut component_indices = Vec::with_capacity(exprs.len());
    let mut index_by_expr = FxHashMap::<&Expr, usize>::default();

    for (group_index, expr) in exprs.iter().enumerate() {
        let expr = unwrap_shared(expr);
        let index = if let Some(&index) = index_by_expr.get(expr) {
            index
        } else {
            let index = unique_exprs.len();
            unique_exprs.push(expr);
            unique_first_group_indices.push(group_index);
            index_by_expr.insert(expr, index);
            index
        };
        component_indices.push(index);
    }

    let repeat_base_cache = build_repeat_base_dfa_cache(&unique_exprs, local_small_product);
    let repeat_base_cache = (!repeat_base_cache.is_empty()).then_some(&repeat_base_cache);

    let compile_component = |(_index, expr): (usize, &&Expr)| {
        if profile_detail {
            let started_at = Instant::now();
            let component = compile_product_component_with_options(
                expr,
                preserve_coordinates,
                virtual_fixed_sequences,
                repeat_base_cache,
            );
            (
                component,
                Some(started_at.elapsed().as_secs_f64() * 1000.0),
            )
        } else {
            (
                compile_product_component_with_options(
                    expr,
                    preserve_coordinates,
                    virtual_fixed_sequences,
                    repeat_base_cache,
                ),
                None,
            )
        }
    };
    let compiled: Vec<(ProductComponent, Option<f64>)> =
        if local_small_product_work_enabled(unique_exprs.len(), local_small_product) {
            unique_exprs.iter().enumerate().map(compile_component).collect()
        } else {
            unique_exprs.par_iter().enumerate().map(compile_component).collect()
        };
    let (unique_components, compile_times): (Vec<_>, Vec<_>) = compiled.into_iter().unzip();
    let cache_hits = exprs.len() - unique_components.len();
    let profiles = profile_detail.then(|| {
        let mut uses = vec![0usize; unique_components.len()];
        for &index in &component_indices {
            uses[index] += 1;
        }
        unique_components
            .iter()
            .zip(&compile_times)
            .enumerate()
            .map(|(index, (component, compile_ms))| ProductComponentCompileProfile {
                first_group_index: unique_first_group_indices[index],
                uses: uses[index],
                compile_ms: compile_ms.unwrap_or_default(),
                states: component.profile_state_count(),
                transitions: component.profile_transition_count(),
            })
            .collect::<Vec<_>>()
    });
    let components = component_indices
        .iter()
        .copied()
        .map(|index| unique_components[index].clone())
        .collect();
    (components, cache_hits, profiles, component_indices)
}

fn compile_product_components(exprs: &[Expr]) -> (Vec<ProductComponent>, usize) {
    let (components, cache_hits, _, _) =
        compile_product_components_profiled(exprs, false, false, true, false);
    (components, cache_hits)
}

fn product_coordinate_layout(
    logical_components: Vec<ProductComponent>,
    component_indices: &[usize],
    collapse_duplicates: bool,
) -> (Vec<ProductComponent>, Vec<Vec<usize>>) {
    if !collapse_duplicates {
        let coordinate_groups = (0..logical_components.len())
            .map(|group| vec![group])
            .collect();
        return (logical_components, coordinate_groups);
    }

    let coordinate_count = component_indices
        .iter()
        .copied()
        .max()
        .map_or(0, |max_index| max_index + 1);
    let mut components = vec![None; coordinate_count];
    let mut coordinate_groups = vec![Vec::new(); coordinate_count];
    for (group, (component, &coordinate)) in logical_components
        .into_iter()
        .zip(component_indices)
        .enumerate()
    {
        coordinate_groups[coordinate].push(group);
        if components[coordinate].is_none() {
            components[coordinate] = Some(component);
        }
    }
    let components = components
        .into_iter()
        .map(|component| component.expect("every product coordinate has a component"))
        .collect();
    (components, coordinate_groups)
}

fn build_product_dfa(
    exprs: &[Expr],
    profile_labels: Option<&[ProductComponentProfileLabel]>,
    visible_groups: usize,
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
    capture_trace: bool,
    virtual_fixed_sequences: bool,
    local_small_product: bool,
    collapse_traced_duplicate_coordinates: bool,
) -> (DFA, bool, Option<ProductBuildTrace>) {
    let profile_trace = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some();
    let profile_detail = profile_trace
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_started_at = Instant::now();
    let component_compile_started_at = Instant::now();
    let preserve_component_coordinates = capture_trace && !collapse_traced_duplicate_coordinates;
    let (logical_components, component_cache_hits, component_profiles, component_indices) =
        compile_product_components_profiled(
            exprs,
            profile_detail,
            preserve_component_coordinates,
            virtual_fixed_sequences,
            local_small_product,
        );
    let component_compile_ms = profile_timing
        .then(|| component_compile_started_at.elapsed().as_secs_f64() * 1000.0);
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] product_component_cache groups={} unique_components={} cache_hits={}",
            exprs.len(),
            exprs.len() - component_cache_hits,
            component_cache_hits,
        );
    }
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] product_components groups={} unique_components={} cache_hits={} compile_components_ms={:.3}",
            logical_components.len(),
            logical_components.len() - component_cache_hits,
            component_cache_hits,
            profile_started_at.elapsed().as_secs_f64() * 1000.0
        );
        let mut ranked = component_profiles
            .as_ref()
            .expect("detail profiling records unique component profiles")
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|(_, left), (_, right)| {
            right.compile_ms.total_cmp(&left.compile_ms)
        });
        let report_count = if profile_trace {
            ranked.len()
        } else {
            ranked.len().min(20)
        };
        for (rank, (unique_index, component)) in ranked.into_iter().take(report_count).enumerate() {
            let index = component.first_group_index;
            let label = profile_labels
                .and_then(|labels| labels.get(index))
                .map(|label| {
                    format!(
                        " name={:?} origin={} shared={} expr={:?}",
                        label.name,
                        label.origin,
                        label.shared,
                        expr_profile_summary(&exprs[index]),
                    )
                })
                .unwrap_or_else(|| format!(" expr={:?}", expr_profile_summary(&exprs[index])));
            eprintln!(
                "[glrmask/profile][tokenizer/component-rank] rank={} unique_index={} first_group_index={} uses={} states={} transitions={} compile_ms={:.3}{}",
                rank + 1,
                unique_index,
                index,
                component.uses,
                component.states,
                component.transitions,
                component.compile_ms,
                label,
            );
        }
        let omitted = component_profiles
            .as_ref()
            .map_or(0, |profiles| profiles.len().saturating_sub(report_count));
        if omitted > 0 {
            eprintln!(
                "[glrmask/profile][tokenizer/component-rank] omitted={} set GLRMASK_PROFILE_TOKENIZER_TRACE=1 for exhaustive component output",
                omitted,
            );
        }
    }
    let num_groups = logical_components.len();
    let collapse_duplicates = component_cache_hits > 0
        && visible_groups > 1
        && (!capture_trace || collapse_traced_duplicate_coordinates)
        && !profile_trace;
    let (components, coordinate_groups) = product_coordinate_layout(
        logical_components,
        &component_indices,
        collapse_duplicates,
    );
    let num_coordinates = components.len();
    let direct_single_visible_group = visible_groups == 1
        && num_groups > 1
        && (!exclusions.is_empty() || !intersections.is_empty());
    let component_dead_states: Vec<Option<u32>> = components
        .iter()
        .map(ProductComponent::dead_state)
        .collect();
    let equivalence_classes_started_at = Instant::now();
    let (class_map, class_members) = compute_product_equivalence_classes(&components);
    let equivalence_classes_ms = profile_timing
        .then(|| equivalence_classes_started_at.elapsed().as_secs_f64() * 1000.0);
    let num_classes = class_members.len();
    let class_transition_started_at = Instant::now();
    let component_class_transitions = build_product_class_transitions(&components, &class_map);
    let class_transition_ms = profile_timing
        .then(|| class_transition_started_at.elapsed().as_secs_f64() * 1000.0);
    if direct_single_visible_group
        && pure_binary_intersection(exclusions, intersections)
        && let Some(result) = try_build_dense_binary_intersection_product(
            &components,
            &class_members,
            &component_class_transitions,
            capture_trace,
            profile_timing,
        )
    {
        return result;
    }
    if direct_single_visible_group
        && pure_binary_intersection(exclusions, intersections)
        && let Some(result) = try_build_sparse_virtual_repeat_finite_intersection_product(
            &components,
            &class_members,
            &component_class_transitions,
            capture_trace,
            profile_timing,
        )
    {
        return result;
    }
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(if direct_single_visible_group {
        1
    } else {
        num_groups
    });

    assert!(num_groups <= u32::MAX as usize, "too many product DFA groups");
    let mut start_tuple = ProductStateTuple::with_capacity(num_coordinates);
    for coordinate_id in 0..num_coordinates {
        start_tuple.push((coordinate_id as u32, 0u32));
    }
    let (start_finalizers, start_future) = if direct_single_visible_group {
        (
            product_state_single_visible_finalizer_with_layout(
                &components,
                &coordinate_groups,
                num_groups,
                &start_tuple,
                exclusions,
                intersections,
            ),
            BitSet::new(1),
        )
    } else {
        product_state_metadata_with_layout(&components, &coordinate_groups, num_groups, &start_tuple)
    };
    dfa.overwrite_state_metadata(0, start_finalizers, start_future);

    const PRODUCT_STATE_FINGERPRINT_MIN_CLASS_TRANSITIONS: usize = 20_000;
    let fingerprint_state_lookup = capture_trace
        && std::env::var_os("GLRMASK_DISABLE_PRODUCT_STATE_FINGERPRINT").is_none()
        && (std::env::var_os("GLRMASK_EXPERIMENT_PRODUCT_STATE_FINGERPRINT").is_some()
            || product_class_transition_entry_count(&component_class_transitions)
                >= PRODUCT_STATE_FINGERPRINT_MIN_CLASS_TRANSITIONS);
    let mut state_map = FxHashMap::<ProductStateTuple, u32>::default();
    let mut state_by_fingerprint = FxHashMap::<u64, u32>::default();
    let mut fingerprint_collisions = FxHashMap::<ProductStateTuple, u32>::default();
    let mut fingerprint_state_tuples = fingerprint_state_lookup.then(|| vec![start_tuple.clone()]);
    let mut worklist = VecDeque::new();
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    // Pre-allocated buffers for class transition tuples (reused across states)
    let mut class_buffers: Vec<ProductStateTuple> = (0..num_classes)
        .map(|_| ProductStateTuple::new())
        .collect();
    let mut class_fingerprints = vec![PRODUCT_STATE_FINGERPRINT_SEED; num_classes];
    let mut class_active = vec![false; num_classes];
    let mut used_classes = Vec::<usize>::new();
    let mut growth_recorder = profile_trace.then(|| ProductGrowthRecorder::new(num_coordinates));
    let mut state_tuples = capture_trace.then(|| vec![start_tuple.clone()]);
    if fingerprint_state_lookup {
        state_by_fingerprint.insert(product_state_fingerprint(&start_tuple), 0);
    } else {
        state_map.insert(start_tuple.clone(), 0);
    }
    if let Some(recorder) = growth_recorder.as_mut() {
        recorder.record(num_coordinates, &start_tuple);
    }
    worklist.push_back((0, start_tuple));

    let product_state_expand_started_at = Instant::now();
    while let Some((current_state, state_tuple)) = worklist.pop_front() {
        for &(group_id, component_state) in &state_tuple {
            let group_index = group_id as usize;

            match (&components[group_index], &component_class_transitions[group_index]) {
                (
                    ProductComponent::Materialized(_)
                    | ProductComponent::MaterializedZeroMinRepeatSuffix { .. },
                    ProductComponentClassTransitions::Materialized(class_transitions),
                ) => {
                    for &(class_id, target) in &class_transitions[component_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
                        }
                        if component_dead_states[group_index] == Some(target) {
                            continue;
                        }

                        class_buffers[class_index].push((group_id, target));
                        if fingerprint_state_lookup {
                            class_fingerprints[class_index] = extend_product_state_fingerprint(
                                class_fingerprints[class_index],
                                group_id,
                                target,
                            );
                        }
                    }
                }
                (
                    ProductComponent::VirtualFixedSequence { .. },
                    ProductComponentClassTransitions::VirtualFixedSequence(class_transitions),
                ) => {
                    for &(class_id, target) in &class_transitions[component_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
                        }
                        class_buffers[class_index].push((group_id, target));
                        if fingerprint_state_lookup {
                            class_fingerprints[class_index] = extend_product_state_fingerprint(
                                class_fingerprints[class_index],
                                group_id,
                                target,
                            );
                        }
                    }
                }
                (
                    ProductComponent::VirtualBoundedRepeat { base_dfa, max, .. },
                    ProductComponentClassTransitions::VirtualBoundedRepeat(base_class_transitions),
                ) => {
                    let base_state_count = base_dfa.num_states() as u32;
                    let copy_count = component_state / base_state_count;
                    if copy_count >= *max {
                        continue;
                    }

                    let base_state = component_state % base_state_count;
                    if base_dfa.finalizers(base_state).contains(0) {
                        continue;
                    }
                    for &(class_id, target_base) in &base_class_transitions[base_state as usize] {
                        let class_index = class_id as usize;
                        if !class_active[class_index] {
                            class_active[class_index] = true;
                            used_classes.push(class_index);
                            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
                        }
                        if component_dead_states[group_index] == Some(target_base) {
                            continue;
                        }

                        let target = if base_dfa.finalizers(target_base).contains(0) {
                            (copy_count + 1) * base_state_count
                        } else {
                            copy_count * base_state_count + target_base
                        };

                        class_buffers[class_index].push((group_id, target));
                        if fingerprint_state_lookup {
                            class_fingerprints[class_index] = extend_product_state_fingerprint(
                                class_fingerprints[class_index],
                                group_id,
                                target,
                            );
                        }
                    }
                }
                _ => unreachable!("component and class-transition kinds must match"),
            }
        }

        let mut class_transitions = Vec::with_capacity(used_classes.len());
        for &class_index in &used_classes {
            let next_tuple = &class_buffers[class_index];
            let existing = if fingerprint_state_lookup {
                let fingerprint = class_fingerprints[class_index];
                state_by_fingerprint
                    .get(&fingerprint)
                    .copied()
                    .filter(|&state| {
                        fingerprint_state_tuples
                            .as_ref()
                            .expect("fingerprint lookup retains canonical tuples")[state as usize]
                            == *next_tuple
                    })
                    .or_else(|| fingerprint_collisions.get(next_tuple).copied())
            } else {
                state_map.get(next_tuple).copied()
            };
            let next_state = if let Some(existing) = existing {
                existing
            } else {
                let new_state = dfa.add_state();
                let (finalizers, future) = if direct_single_visible_group {
                    (
                        product_state_single_visible_finalizer_with_layout(
                            &components,
                            &coordinate_groups,
                            num_groups,
                            next_tuple,
                            exclusions,
                            intersections,
                        ),
                        BitSet::new(1),
                    )
                } else {
                    product_state_metadata_with_layout(&components, &coordinate_groups, num_groups, next_tuple)
                };
                dfa.overwrite_state_metadata(new_state, finalizers, future);
                if fingerprint_state_lookup {
                    let fingerprint = class_fingerprints[class_index];
                    if state_by_fingerprint.contains_key(&fingerprint) {
                        fingerprint_collisions.insert(next_tuple.clone(), new_state);
                    } else {
                        state_by_fingerprint.insert(fingerprint, new_state);
                    }
                    let tuples = fingerprint_state_tuples
                        .as_mut()
                        .expect("fingerprint lookup retains canonical tuples");
                    debug_assert_eq!(tuples.len(), new_state as usize);
                    tuples.push(next_tuple.clone());
                } else {
                    state_map.insert(next_tuple.clone(), new_state);
                }
                if let Some(state_tuples) = state_tuples.as_mut() {
                    debug_assert_eq!(state_tuples.len(), new_state as usize);
                    state_tuples.push(next_tuple.clone());
                }
                if let Some(recorder) = growth_recorder.as_mut() {
                    recorder.record(num_coordinates, next_tuple);
                }
                pending_class_transitions.push(Vec::new());
                worklist.push_back((new_state, next_tuple.clone()));
                new_state
            };
            class_transitions.push((class_index as u8, next_state));
            class_buffers[class_index].clear();
            class_fingerprints[class_index] = PRODUCT_STATE_FINGERPRINT_SEED;
            class_active[class_index] = false;
        }
        used_classes.clear();
        pending_class_transitions[current_state as usize] = class_transitions;
    }
    let product_state_expand_ms = profile_timing
        .then(|| product_state_expand_started_at.elapsed().as_secs_f64() * 1000.0);

    let direct_future_started_at = Instant::now();
    if direct_single_visible_group {
        set_single_group_futures_from_class_graph(&mut dfa, &pending_class_transitions);
    }
    let direct_future_ms = profile_timing
        .then(|| direct_future_started_at.elapsed().as_secs_f64() * 1000.0);

    if profile_trace {
        if let Some(recorder) = growth_recorder.as_ref() {
            let mut states_before = 0usize;
            for (index, states_after) in recorder.prefix_counts().iter().copied().enumerate() {
                let label = profile_labels
                    .and_then(|labels| labels.get(index))
                    .map(|label| {
                        format!(
                            " name={:?} origin={} shared={}",
                            label.name,
                            label.origin,
                            label.shared
                        )
                    })
                    .unwrap_or_else(|| format!(" expr={:?}", expr_profile_summary(&exprs[index])));
                eprintln!(
                    "[glrmask/profile][tokenizer/product-growth] component_index={} states_before={} states_after={} delta_states={}{}",
                    index,
                    states_before,
                    states_after,
                    states_after.saturating_sub(states_before),
                    label
                );
                states_before = states_after;
            }
        }
        eprintln!(
            "[glrmask/profile][tokenizer] product_reachable states={} classes={} construct_ms={:.3}",
            dfa.num_states(),
            num_classes,
            profile_started_at.elapsed().as_secs_f64() * 1000.0
        );
    }

    let byte_expand_started_at = Instant::now();
    let expand_byte_row = |class_transitions: Vec<(u8, u32)>| {
        let byte_capacity: usize = class_transitions
            .iter()
            .map(|(class_id, _)| class_members[*class_id as usize].len())
            .sum();
        const DENSE_BYTE_EXPANSION_THRESHOLD: usize = 96;

        let transitions = if byte_capacity >= DENSE_BYTE_EXPANSION_THRESHOLD {
            // Byte-equivalence classes are disjoint but need not be
            // contiguous, so expanding classes in class-ID order does
            // not generally produce byte-sorted output.  For dense rows,
            // scatter targets into the fixed byte alphabet and scan it
            // once. This avoids a large per-state comparison sort.
            let mut target_by_byte = [u32::MAX; 256];
            for (class_id, target) in class_transitions {
                for &byte in &class_members[class_id as usize] {
                    target_by_byte[byte as usize] = target;
                }
            }
            target_by_byte
                .into_iter()
                .enumerate()
                .filter_map(|(byte, target)| {
                    (target != u32::MAX).then_some((byte as u8, target))
                })
                .collect()
        } else {
            let mut transitions = Vec::with_capacity(byte_capacity);
            for (class_id, target) in class_transitions {
                for &byte in &class_members[class_id as usize] {
                    transitions.push((byte, target));
                }
            }
            if transitions.len() > 1 {
                transitions.sort_unstable_by_key(|entry| entry.0);
            }
            transitions
        };
        crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
    };
    let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
        if local_small_product_work_enabled(num_coordinates, local_small_product) {
            pending_class_transitions
                .into_iter()
                .map(expand_byte_row)
                .collect()
        } else {
            pending_class_transitions
                .into_par_iter()
                .map(expand_byte_row)
                .collect()
        };
    let byte_expand_ms = profile_timing
        .then(|| byte_expand_started_at.elapsed().as_secs_f64() * 1000.0);

    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }

    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] product_phases groups={} coordinates={} classes={} direct_single_visible_group={} component_compile_ms={:.3} equivalence_classes_ms={:.3} class_transition_ms={:.3} product_state_expand_ms={:.3} direct_future_ms={:.3} byte_expand_ms={:.3}",
            num_groups,
            num_coordinates,
            num_classes,
            direct_single_visible_group,
            component_compile_ms.unwrap_or_default(),
            equivalence_classes_ms.unwrap_or_default(),
            class_transition_ms.unwrap_or_default(),
            product_state_expand_ms.unwrap_or_default(),
            direct_future_ms.unwrap_or_default(),
            byte_expand_ms.unwrap_or_default(),
        );
    }

    let state_lookup = if fingerprint_state_lookup {
        ProductStateLookup::Fingerprint {
            state_by_fingerprint,
            canonical_tuples: fingerprint_state_tuples
                .expect("fingerprint lookup retains canonical tuples"),
            overflow: fingerprint_collisions,
        }
    } else {
        ProductStateLookup::Hash(state_map)
    };
    let trace = state_tuples.map(|state_tuples| ProductBuildTrace {
        components,
        coordinate_groups,
        state_tuples: ProductStateTuples::Generic(state_tuples),
        state_lookup,
        direct_single_visible_group,
    });
    (dfa, direct_single_visible_group, trace)
}

fn compute_product_equivalence_classes(components: &[ProductComponent]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut partitions = vec![U8Set::all()];
    let mut seen_sets = FxHashSet::default();

    for component in components {
        match component {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. }
            | ProductComponent::VirtualBoundedRepeat { base_dfa: dfa, .. } => {
                for state in dfa.states() {
                    let mut bytes_by_target = FxHashMap::<u32, U8Set>::default();
                    for (byte, &target) in state.transitions.iter() {
                        bytes_by_target
                            .entry(target)
                            .and_modify(|set| {
                                set.insert(byte);
                            })
                            .or_insert_with(|| U8Set::single(byte));
                    }

                    for byte_set in bytes_by_target.into_values() {
                        if seen_sets.insert(byte_set) {
                            partitions = refine_u8_partitions(partitions, byte_set);
                        }
                    }
                }
            }
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => {
                for &byte_set in byte_sets.iter() {
                    if seen_sets.insert(byte_set) {
                        partitions = refine_u8_partitions(partitions, byte_set);
                    }
                }
            }
        }
    }

    let mut class_map = vec![0u8; 256];
    let mut class_members = vec![Vec::new(); partitions.len()];
    for (class_id, partition) in partitions.iter().enumerate() {
        for byte in partition.iter() {
            class_map[byte as usize] = class_id as u8;
            class_members[class_id].push(byte);
        }
    }

    (class_map, class_members)
}

fn build_product_class_transitions_for_dfa(dfa: &DFA, class_map: &[u8]) -> Vec<Vec<(u8, u32)>> {
    let class_count = class_map
        .iter()
        .copied()
        .max()
        .map_or(0usize, |class| class as usize + 1);
    dfa.states()
        .par_iter()
        .map(|state| {
            let mut target_by_class = [u32::MAX; 256];
            for (byte, &target) in state.transitions.iter() {
                let class = class_map[byte as usize] as usize;
                let slot = &mut target_by_class[class];
                if *slot == u32::MAX {
                    *slot = target;
                } else {
                    debug_assert_eq!(
                        *slot, target,
                        "lexer byte-equivalence class must have one target per DFA state",
                    );
                }
            }
            (0..class_count)
                .filter_map(|class| {
                    let target = target_by_class[class];
                    (target != u32::MAX).then_some((class as u8, target))
                })
                .collect()
        })
        .collect()
}

fn build_product_class_transitions(
    components: &[ProductComponent],
    class_map: &[u8],
) -> Vec<ProductComponentClassTransitions> {
    components
        .iter()
        .map(|component| match component {
            ProductComponent::Materialized(dfa)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
                ProductComponentClassTransitions::Materialized(build_product_class_transitions_for_dfa(
                    dfa, class_map,
                ))
            }
            ProductComponent::VirtualFixedSequence { byte_sets, .. } => {
                let mut transitions = Vec::with_capacity(byte_sets.len() + 1);
                for (position, bytes) in byte_sets.iter().enumerate() {
                    let mut classes = bytes
                        .iter()
                        .map(|byte| class_map[byte as usize])
                        .collect::<Vec<_>>();
                    classes.sort_unstable();
                    classes.dedup();
                    transitions.push(
                        classes
                            .into_iter()
                            .map(|class| (class, position as u32 + 1))
                            .collect(),
                    );
                }
                transitions.push(Vec::new());
                ProductComponentClassTransitions::VirtualFixedSequence(transitions)
            }
            ProductComponent::VirtualBoundedRepeat { base_dfa, .. } => {
                ProductComponentClassTransitions::VirtualBoundedRepeat(
                    build_product_class_transitions_for_dfa(base_dfa, class_map),
                )
            }
        })
        .collect()
}

fn finite_product_component_class_row(
    component: &ProductComponent,
    class_transitions: &ProductComponentClassTransitions,
    state: u32,
    out: &mut Vec<(u8, u32)>,
) -> Option<()> {
    out.clear();
    match (component, class_transitions) {
        (
            ProductComponent::Materialized(_)
            | ProductComponent::MaterializedZeroMinRepeatSuffix { .. },
            ProductComponentClassTransitions::Materialized(rows),
        ) => out.extend_from_slice(rows.get(state as usize)?),
        (
            ProductComponent::VirtualFixedSequence { .. },
            ProductComponentClassTransitions::VirtualFixedSequence(rows),
        ) => out.extend_from_slice(rows.get(state as usize)?),
        _ => return None,
    }
    Some(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SparseVirtualRepeatResidual {
    completed: u32,
    base_state: u32,
}

fn sparse_virtual_repeat_class_row(
    component: &ProductComponent,
    class_transitions: &ProductComponentClassTransitions,
    state: SparseVirtualRepeatResidual,
    out: &mut Vec<(u8, SparseVirtualRepeatResidual)>,
) -> Option<()> {
    let ProductComponent::VirtualBoundedRepeat { base_dfa, max, .. } = component else {
        return None;
    };
    let ProductComponentClassTransitions::VirtualBoundedRepeat(base_rows) = class_transitions else {
        return None;
    };
    out.clear();
    if state.completed >= *max {
        return Some(());
    }
    if base_dfa.finalizers(state.base_state).contains(0) {
        return Some(());
    }
    let dead = explicit_dead_sink_state(base_dfa);
    for &(class, target_base) in base_rows.get(state.base_state as usize)? {
        if dead == Some(target_base) {
            continue;
        }
        let target = if base_dfa.finalizers(target_base).contains(0) {
            SparseVirtualRepeatResidual {
                completed: state.completed.checked_add(1)?,
                base_state: 0,
            }
        } else {
            SparseVirtualRepeatResidual {
                completed: state.completed,
                base_state: target_base,
            }
        };
        out.push((class, target));
    }
    Some(())
}

fn sparse_virtual_repeat_is_accepting(
    component: &ProductComponent,
    state: SparseVirtualRepeatResidual,
) -> Option<bool> {
    let ProductComponent::VirtualBoundedRepeat { min, .. } = component else {
        return None;
    };
    Some(state.base_state == 0 && state.completed >= *min)
}

/// Exact sparse product for a pure binary intersection containing one virtual
/// bounded-repeat coordinate and one finite-language coordinate.  The generic
/// product builder intentionally keeps partial tuples so it can represent
/// unions/exclusions as well as intersections; for a giant repeat that can
/// leave a repeat-only tuple alive after the other coordinate dies and expand
/// once per declared repetition.  Here intersection semantics let us require
/// both coordinates to transition on every byte and discard targets that can
/// no longer accept, so the finite coordinate's live DAG bounds discovery
/// independently of the repeat maximum.
fn try_build_sparse_virtual_repeat_finite_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    profile_timing: bool,
) -> Option<(DFA, bool, Option<ProductBuildTrace>)> {
    if components.len() != 2 || component_class_transitions.len() != 2 {
        return None;
    }
    let repeat_index = match (&components[0], &components[1]) {
        (ProductComponent::VirtualBoundedRepeat { .. }, ProductComponent::VirtualBoundedRepeat { .. }) => {
            return None;
        }
        (ProductComponent::VirtualBoundedRepeat { .. }, _) => 0usize,
        (_, ProductComponent::VirtualBoundedRepeat { .. }) => 1usize,
        _ => return None,
    };
    let finite_index = 1 - repeat_index;
    if !product_component_has_finite_language(&components[finite_index]) {
        return None;
    }

    let started_at = profile_timing.then(Instant::now);
    let start_repeat = SparseVirtualRepeatResidual {
        completed: 0,
        base_state: 0,
    };
    let start = (start_repeat, 0u32);
    let mut pair_to_state = FxHashMap::<(SparseVirtualRepeatResidual, u32), u32>::default();
    pair_to_state.insert(start, 0);
    let mut pairs = vec![start];
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    let start_accepting = sparse_virtual_repeat_is_accepting(
        &components[repeat_index],
        start_repeat,
    )
    .expect("identified virtual repeat component must expose repeat acceptance")
        && product_component_state_flags(&components[finite_index], 0).0;
    let mut accepting = vec![start_accepting];
    let mut repeat_row = Vec::<(u8, SparseVirtualRepeatResidual)>::new();
    let mut finite_row = Vec::<(u8, u32)>::new();

    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (repeat_state, finite_state) = pairs[cursor];
        sparse_virtual_repeat_class_row(
            &components[repeat_index],
            &component_class_transitions[repeat_index],
            repeat_state,
            &mut repeat_row,
        )
        .expect("identified virtual repeat component must expose an exact class row");
        finite_product_component_class_row(
            &components[finite_index],
            &component_class_transitions[finite_index],
            finite_state,
            &mut finite_row,
        )
        .expect("finite intersection component must expose an exact class row");

        let mut repeat_position = 0usize;
        let mut finite_position = 0usize;
        let mut row = Vec::<(u8, u32)>::new();
        while repeat_position < repeat_row.len() && finite_position < finite_row.len() {
            let (repeat_class, repeat_target) = repeat_row[repeat_position];
            let (finite_class, finite_target) = finite_row[finite_position];
            if repeat_class < finite_class {
                repeat_position += 1;
                continue;
            }
            if finite_class < repeat_class {
                finite_position += 1;
                continue;
            }
            repeat_position += 1;
            finite_position += 1;

            let finite_flags =
                product_component_state_flags(&components[finite_index], finite_target);
            if !finite_flags.0 && !finite_flags.1 {
                continue;
            }
            let repeat_accepting = sparse_virtual_repeat_is_accepting(
                &components[repeat_index],
                repeat_target,
            )
            .expect("identified virtual repeat component must expose repeat acceptance");
            let pair = (repeat_target, finite_target);
            let target = if let Some(&target) = pair_to_state.get(&pair) {
                target
            } else {
                let target = u32::try_from(pairs.len())
                    .expect("sparse exact product exceeded the DFA u32 state-id domain");
                pair_to_state.insert(pair, target);
                pairs.push(pair);
                pending_class_transitions.push(Vec::new());
                accepting.push(repeat_accepting && finite_flags.0);
                target
            };
            row.push((repeat_class, target));
        }
        pending_class_transitions[cursor] = row;
        cursor += 1;
    }

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    while dfa.num_states() < pairs.len() {
        dfa.add_state();
    }
    for (state, &is_accepting) in accepting.iter().enumerate() {
        let mut finalizers = BitSet::new(1);
        if is_accepting {
            finalizers.set(0);
        }
        dfa.overwrite_state_metadata(state as u32, finalizers, BitSet::new(1));
    }
    set_single_group_futures_from_class_graph(&mut dfa, &pending_class_transitions);

    let expanded = pending_class_transitions
        .into_iter()
        .map(|row| {
            let mut transitions = Vec::new();
            for (class, target) in row {
                transitions.extend(
                    class_members[class as usize]
                        .iter()
                        .copied()
                        .map(|byte| (byte, target)),
                );
            }
            if transitions.len() > 1 {
                transitions.sort_unstable_by_key(|entry| entry.0);
            }
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
        })
        .collect::<Vec<_>>();
    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded) {
        state.transitions = transitions;
    }
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] sparse_virtual_repeat_finite_intersection repeat_component={} states={} classes={} total_ms={:.3}",
            repeat_index,
            dfa.num_states(),
            class_members.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let _ = capture_trace;
    Some((dfa, true, None))
}

fn pure_binary_intersection(
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> bool {
    exclusions.is_empty()
        && intersections.len() == 1
        && intersections
            .get(&0)
            .is_some_and(|required| required.len() == 1 && required.contains(&1))
}

/// Exact single-group DFA for `left \\ right`.
///
/// Unlike the ordinary two-component product, the RHS is allowed to die while
/// the LHS remains live.  A dedicated sentinel represents that permanently-dead
/// RHS coordinate, so reachable states fit in one dense
/// `(left_state, right_state_or_dead)` table with no tuple hashing.
fn build_dense_binary_exclusion_dfa(left: &DFA, right: &DFA, support: U8Set) -> Option<DFA> {
    const MAX_DENSE_PAIR_CELLS: usize = 40_000_000;

    if left.num_states() == 0
        || right.num_states() == 0
        || left.num_groups() != 1
        || right.num_groups() != 1
        || left.has_epsilon_transitions()
        || right.has_epsilon_transitions()
    {
        return None;
    }

    let left_states = left.num_states();
    let right_states = right.num_states();
    let right_dead_sentinel = u32::try_from(right_states).ok()?;
    let right_stride = right_states.checked_add(1)?;
    let pair_cells = left_states.checked_mul(right_stride)?;
    if pair_cells == 0 || pair_cells > MAX_DENSE_PAIR_CELLS {
        return None;
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let class_started_at = profile.then(Instant::now);
    let (class_map, class_members) = compute_dfa_byte_equivalence_classes(&[left, right]);
    let class_ms = class_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let left_transitions = build_product_class_transitions_for_dfa(left, &class_map);
    let right_transitions = build_product_class_transitions_for_dfa(right, &class_map);
    let left_dead = explicit_dead_sink_state(left);
    let right_dead = explicit_dead_sink_state(right);
    if left_dead == Some(0) {
        let mut dfa = DFA::new(1);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, support);
        return Some(dfa);
    }
    let right_start = if right_dead == Some(0) {
        right_dead_sentinel
    } else {
        0
    };

    let pair_index = |left_state: u32, right_state: u32| -> Option<usize> {
        (left_state as usize)
            .checked_mul(right_stride)?
            .checked_add(right_state as usize)
    };

    let mut state_by_pair = vec![u32::MAX; pair_cells];
    state_by_pair[pair_index(0, right_start)?] = 0;
    let mut pairs = vec![(0u32, right_start)];
    let mut accepting = Vec::<bool>::new();
    let mut row_offsets = Vec::<u32>::with_capacity(256);
    let mut row_classes = Vec::<u8>::new();
    let mut row_targets = Vec::<u32>::new();
    row_offsets.push(0);

    let is_accepting = |left_state: u32, right_state: u32| {
        if !left.finalizers(left_state).contains(0) {
            return false;
        }
        right_state == right_dead_sentinel || !right.finalizers(right_state).contains(0)
    };

    let construct_started_at = profile.then(Instant::now);
    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (left_state, right_state) = pairs[cursor];
        accepting.push(is_accepting(left_state, right_state));
        let left_row = left_transitions.get(left_state as usize)?;
        let right_row = if right_state == right_dead_sentinel {
            None
        } else {
            Some(right_transitions.get(right_state as usize)?)
        };
        let mut right_index = 0usize;
        for &(class, left_target) in left_row {
            if left_target == u32::MAX || left_dead == Some(left_target) {
                continue;
            }

            let right_target = if let Some(right_row) = right_row {
                while right_index < right_row.len() && right_row[right_index].0 < class {
                    right_index += 1;
                }
                if right_index < right_row.len() && right_row[right_index].0 == class {
                    let target = right_row[right_index].1;
                    if target == u32::MAX || right_dead == Some(target) {
                        right_dead_sentinel
                    } else {
                        target
                    }
                } else {
                    right_dead_sentinel
                }
            } else {
                right_dead_sentinel
            };

            let index = pair_index(left_target, right_target)?;
            let target = if state_by_pair[index] != u32::MAX {
                state_by_pair[index]
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_pair[index] = target;
                pairs.push((left_target, right_target));
                target
            };
            row_classes.push(class);
            row_targets.push(target);
        }
        row_offsets.push(u32::try_from(row_targets.len()).ok()?);
        cursor += 1;
    }
    let construct_ms = construct_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let future_started_at = profile.then(Instant::now);
    let has_future = compute_single_group_futures_from_class_graph_csr(
        &accepting,
        &row_offsets,
        &row_targets,
        0,
    );
    let future_ms = future_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let mut dfa = DFA::new(accepting.len());
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, support);
    for state in 0..dfa.num_states() {
        let mut finalizers = BitSet::new(1);
        if accepting[state] {
            finalizers.set(0);
        }
        let mut future = BitSet::new(1);
        if has_future[state] {
            future.set(0);
        }
        dfa.overwrite_state_metadata(state as u32, finalizers, future);
    }

    let expand_started_at = profile.then(Instant::now);
    for state in 0..dfa.num_states() {
        let start = row_offsets[state] as usize;
        let end = row_offsets[state + 1] as usize;
        let byte_capacity = row_classes[start..end]
            .iter()
            .map(|class| class_members[*class as usize].len())
            .sum();
        let mut transitions = Vec::<(u8, u32)>::with_capacity(byte_capacity);
        for (&class, &target) in row_classes[start..end]
            .iter()
            .zip(&row_targets[start..end])
        {
            transitions.extend(
                class_members[class as usize]
                    .iter()
                    .copied()
                    .map(|byte| (byte, target)),
            );
        }
        if transitions.len() > 1 {
            transitions.sort_unstable_by_key(|entry| entry.0);
        }
        dfa.set_transitions_from_sorted_entries(state as u32, transitions);
    }
    let expand_ms = expand_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] dense_binary_exclusion left_states={} right_states={} pair_cells={} reachable_states={} classes={} class_ms={:.3} construct_ms={:.3} future_ms={:.3} expand_ms={:.3} total_ms={:.3}",
            left_states,
            right_states,
            pair_cells,
            dfa.num_states(),
            class_members.len(),
            class_ms,
            construct_ms,
            future_ms,
            expand_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(dfa)
}

struct DeferredDenseBinaryIntersectionProduct {
    metadata: DFA,
    accepting: Vec<bool>,
    pending_class_transition_offsets: Vec<u32>,
    pending_class_transition_classes: Vec<u8>,
    pending_class_transition_targets: Vec<u32>,
    deferred_transition_rows: Option<DeferredDenseBinaryTransitionRows>,
    class_members: Vec<Vec<u8>>,
    left_states: usize,
    right_states: usize,
    pair_cells: usize,
    num_classes: usize,
    cooperative_yield_interval: usize,
    discovery_ms: f64,
    started_at: Option<Instant>,
}

struct DeferredDenseBinaryTransitionRows {
    pairs: Vec<(u32, u32)>,
    state_by_pair: Vec<u32>,
    left_transitions: Vec<Vec<(u8, u32)>>,
    right_transitions: Vec<Vec<(u8, u32)>>,
    left_dead: Option<u32>,
    right_dead: Option<u32>,
}

impl DeferredDenseBinaryIntersectionProduct {
    fn num_states(&self) -> usize {
        self.accepting.len()
    }

    fn initial_is_accepting(&self) -> bool {
        self.accepting.first().copied().unwrap_or(false)
    }

    fn ensure_group_capacity(&mut self, num_groups: usize) {
        self.metadata.ensure_group_capacity(num_groups);
    }

    fn set_group_u8set(&mut self, group_id: u32, set: U8Set) {
        self.metadata.set_group_u8set(group_id, set);
    }

    fn materialize_metadata_parts(
        metadata: DFA,
        accepting: Vec<bool>,
        cooperative_yield_interval: usize,
    ) -> DFA {
        if metadata.num_states() == accepting.len() {
            return metadata;
        }
        let num_groups = metadata.num_groups();
        let mut dfa = DFA::new(accepting.len());
        dfa.ensure_group_capacity(num_groups);
        for group in 0..num_groups {
            dfa.set_group_u8set(
                group as u32,
                *metadata.group_id_to_u8set(group as u32),
            );
        }
        if num_groups != 0 {
            for (state, &is_accepting) in accepting.iter().enumerate() {
                if cooperative_yield_interval != 0
                    && state % cooperative_yield_interval == 0
                {
                    let _ = rayon::yield_now();
                }
                if is_accepting {
                    let mut finalizers = BitSet::new(num_groups);
                    finalizers.set(0);
                    dfa.overwrite_state_metadata(
                        state as u32,
                        finalizers,
                        BitSet::new(num_groups),
                    );
                }
            }
        }
        dfa
    }

    fn attach_runtime_trace(&mut self, trace: ProductBuildTrace) -> Option<()> {
        let ProductStateTuples::DenseBinary(pairs) = trace.state_tuples else {
            return None;
        };
        let ProductStateLookup::DenseBinary {
            right_states,
            state_by_pair,
            overflow,
        } = trace.state_lookup
        else {
            return None;
        };
        if right_states != self.right_states || !overflow.is_empty() || pairs.len() != self.num_states()
        {
            return None;
        }
        let Some(rows) = self.deferred_transition_rows.as_mut() else {
            return Some(());
        };
        rows.pairs = pairs;
        rows.state_by_pair = state_by_pair;
        Some(())
    }

    fn materialize_pending_class_transitions(&mut self) -> Option<()> {
        let Some(rows) = self.deferred_transition_rows.take() else {
            return Some(());
        };
        let state_count = self.pending_class_transition_offsets.len().checked_sub(1)?;
        if rows.pairs.len() != state_count
            || rows.state_by_pair.len() != self.pair_cells
        {
            return None;
        }
        if self.pending_class_transition_offsets.len() != rows.pairs.len() + 1 {
            return None;
        }
        let total_entries = *self.pending_class_transition_offsets.last()? as usize;
        self.pending_class_transition_classes = Vec::with_capacity(total_entries);
        self.pending_class_transition_targets = Vec::with_capacity(total_entries);
        for (state, &(left_state, right_state)) in rows.pairs.iter().enumerate() {
            if self.cooperative_yield_interval != 0
                && state % self.cooperative_yield_interval == 0
            {
                let _ = rayon::yield_now();
            }
            let expected_start = self.pending_class_transition_offsets[state] as usize;
            if self.pending_class_transition_classes.len() != expected_start
                || self.pending_class_transition_targets.len() != expected_start
            {
                return None;
            }
            let left_row = &rows.left_transitions[left_state as usize];
            let right_row = &rows.right_transitions[right_state as usize];
            let mut left_index = 0usize;
            let mut right_index = 0usize;
            while left_index < left_row.len() && right_index < right_row.len() {
                let (left_class, left_target) = left_row[left_index];
                let (right_class, right_target) = right_row[right_index];
                if left_class < right_class {
                    left_index += 1;
                    continue;
                }
                if right_class < left_class {
                    right_index += 1;
                    continue;
                }
                left_index += 1;
                right_index += 1;
                if left_target == u32::MAX
                    || right_target == u32::MAX
                    || rows.left_dead == Some(left_target)
                    || rows.right_dead == Some(right_target)
                {
                    continue;
                }
                let pair_index = (left_target as usize)
                    .checked_mul(self.right_states)?
                    .checked_add(right_target as usize)?;
                let target = *rows.state_by_pair.get(pair_index)?;
                if target == u32::MAX {
                    return None;
                }
                self.pending_class_transition_classes.push(left_class);
                self.pending_class_transition_targets.push(target);
            }
            if self.pending_class_transition_classes.len()
                != self.pending_class_transition_offsets[state + 1] as usize
                || self.pending_class_transition_targets.len()
                    != self.pending_class_transition_offsets[state + 1] as usize
            {
                return None;
            }
        }
        debug_assert_eq!(self.pending_class_transition_classes.len(), total_entries);
        debug_assert_eq!(self.pending_class_transition_targets.len(), total_entries);
        Some(())
    }

    fn finish(mut self) -> DFA {
        let profile_timing = self.started_at.is_some();
        let metadata = std::mem::take(&mut self.metadata);
        let accepting = std::mem::take(&mut self.accepting);
        let cooperative_yield_interval = self.cooperative_yield_interval;
        let ((mut dfa, metadata_ms), (transition_ok, transition_rows_ms)) = rayon::join(
            || {
                let started_at = profile_timing.then(Instant::now);
                let dfa = Self::materialize_metadata_parts(
                    metadata,
                    accepting,
                    cooperative_yield_interval,
                );
                let ms = started_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                (dfa, ms)
            },
            || {
                let started_at = profile_timing.then(Instant::now);
                let result = self.materialize_pending_class_transitions();
                let ms = started_at
                    .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
                (result, ms)
            },
        );
        transition_ok.expect("deferred dense transition rows must materialize");
        let future_started_at = profile_timing.then(Instant::now);
        set_single_group_futures_from_class_graph_csr(
            &mut dfa,
            &self.pending_class_transition_offsets,
            &self.pending_class_transition_targets,
            self.cooperative_yield_interval,
        );
        let future_ms = future_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let expansion_started_at = profile_timing.then(Instant::now);
        let parallel_expansion = std::env::var("GLRMASK_PARALLEL_DEFERRED_BYTE_EXPANSION")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false);
        let dense_expansion_threshold = std::env::var(
            "GLRMASK_DEFERRED_DENSE_BYTE_EXPANSION_THRESHOLD",
        )
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
        let mut byte_to_class = [0u8; 256];
        for (class, bytes) in self.class_members.iter().enumerate() {
            for &byte in bytes {
                byte_to_class[byte as usize] = class as u8;
            }
        }
        let expand = |classes: &[u8], targets: &[u32], target_by_class: &mut [u32]| {
            debug_assert_eq!(classes.len(), targets.len());
            let byte_capacity: usize = classes
                .iter()
                .map(|class| self.class_members[*class as usize].len())
                .sum();
            let entries = if byte_capacity >= dense_expansion_threshold {
                target_by_class.fill(u32::MAX);
                for (&class, &target) in classes.iter().zip(targets) {
                    target_by_class[class as usize] = target;
                }
                let mut entries = Vec::with_capacity(byte_capacity);
                for (byte, &class) in byte_to_class.iter().enumerate() {
                    let target = target_by_class[class as usize];
                    if target != u32::MAX {
                        entries.push((byte as u8, target));
                    }
                }
                entries
            } else {
                let mut entries = Vec::with_capacity(byte_capacity);
                for (&class, &target) in classes.iter().zip(targets) {
                    for &byte in &self.class_members[class as usize] {
                        entries.push((byte, target));
                    }
                }
                if entries.len() > 1 {
                    entries.sort_unstable_by_key(|entry| entry.0);
                }
                entries
            };
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(entries)
        };
        let expanded_transitions: Vec<crate::ds::char_transitions::CharTransitions<u32>> =
            if parallel_expansion {
                (0..dfa.num_states())
                    .into_par_iter()
                    .map_init(
                        || vec![u32::MAX; self.num_classes],
                        |target_by_class, state| {
                            let row_start = self.pending_class_transition_offsets[state] as usize;
                            let row_end =
                                self.pending_class_transition_offsets[state + 1] as usize;
                            expand(
                                &self.pending_class_transition_classes[row_start..row_end],
                                &self.pending_class_transition_targets[row_start..row_end],
                                target_by_class,
                            )
                        },
                    )
                    .collect()
            } else {
                let mut target_by_class = vec![u32::MAX; self.num_classes];
                (0..dfa.num_states())
                    .map(|state| {
                        let row_start = self.pending_class_transition_offsets[state] as usize;
                        let row_end = self.pending_class_transition_offsets[state + 1] as usize;
                        expand(
                            &self.pending_class_transition_classes[row_start..row_end],
                            &self.pending_class_transition_targets[row_start..row_end],
                            &mut target_by_class,
                        )
                    })
                    .collect()
            };
        for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
            state.transitions = transitions;
        }
        let expansion_ms = expansion_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(started_at) = self.started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] dense_binary_intersection left_states={} right_states={} pair_cells={} reachable_states={} class_entries={} classes={} discovery_ms={:.3} transition_rows_ms={:.3} metadata_ms={:.3} future_ms={:.3} expansion_ms={:.3} total_ms={:.3}",
                self.left_states,
                self.right_states,
                self.pair_cells,
                dfa.num_states(),
                self.pending_class_transition_classes.len(),
                self.num_classes,
                self.discovery_ms,
                transition_rows_ms,
                metadata_ms,
                future_ms,
                expansion_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        dfa
    }

    fn finish_compressed(mut self) -> (DFA, CompressedTransitionSegment) {
        let profile_timing = self.started_at.is_some();
        let transition_rows_started_at = profile_timing.then(Instant::now);
        self.materialize_pending_class_transitions()
            .expect("deferred dense transition rows must materialize");
        let transition_rows_ms = transition_rows_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let future_started_at = profile_timing.then(Instant::now);
        let has_future = compute_single_group_futures_from_class_graph_csr(
            &self.accepting,
            &self.pending_class_transition_offsets,
            &self.pending_class_transition_targets,
            self.cooperative_yield_interval,
        );
        let future_ms = future_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let metadata_started_at = profile_timing.then(Instant::now);
        debug_assert_eq!(self.metadata.num_groups(), 1);
        let group_u8set = *self.metadata.group_id_to_u8set(0);
        let dfa = DFA::new_with_single_group_metadata(
            &self.accepting,
            &has_future,
            group_u8set,
        );
        let metadata_ms = metadata_started_at
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let mut byte_to_class = [0u8; 256];
        let class_members = self
            .class_members
            .into_iter()
            .enumerate()
            .map(|(class, bytes)| {
                for &byte in &bytes {
                    byte_to_class[byte as usize] = class as u8;
                }
                bytes.into_boxed_slice()
            })
            .collect::<Vec<_>>();
        let expanded_transition_count = self
            .pending_class_transition_classes
            .iter()
            .map(|class| class_members[*class as usize].len())
            .sum();
        if let Some(started_at) = self.started_at {
            eprintln!(
                "[glrmask/profile][tokenizer] dense_binary_intersection left_states={} right_states={} pair_cells={} reachable_states={} class_entries={} classes={} discovery_ms={:.3} transition_rows_ms={:.3} metadata_ms={:.3} future_ms={:.3} expansion_ms=0.000 compressed=true total_ms={:.3}",
                self.left_states,
                self.right_states,
                self.pair_cells,
                dfa.num_states(),
                self.pending_class_transition_classes.len(),
                self.num_classes,
                self.discovery_ms,
                transition_rows_ms,
                metadata_ms,
                future_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let segment = CompressedTransitionSegment {
            state_offset: 0,
            state_count: dfa.num_states() as u32,
            byte_to_class: Arc::from(byte_to_class.to_vec().into_boxed_slice()),
            class_members: Arc::from(class_members),
            row_offsets: Arc::from(self.pending_class_transition_offsets),
            entries: CompressedTransitionEntries::from_parts(
                self.pending_class_transition_classes,
                self.pending_class_transition_targets,
            ),
            expanded_transition_count,
        };
        (dfa, segment)
    }
}

fn try_discover_dense_binary_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    defer_transition_rows: bool,
    defer_metadata: bool,
    profile_timing: bool,
) -> Option<(
    DeferredDenseBinaryIntersectionProduct,
    Option<ProductBuildTrace>,
)> {
    const MAX_DENSE_PAIR_CELLS: usize = 40_000_000;

    if components.len() != 2 || component_class_transitions.len() != 2 {
        return None;
    }
    let left = components[0].materialized_dfa()?;
    let right = components[1].materialized_dfa()?;
    let left_states = left.num_states();
    let right_states = right.num_states();
    let pair_cells = left_states.checked_mul(right_states)?;
    if pair_cells == 0 || pair_cells > MAX_DENSE_PAIR_CELLS {
        return None;
    }
    let num_classes = class_members.len();
    let cooperative_yield_interval = (defer_transition_rows || defer_metadata)
        .then(|| {
            std::env::var("GLRMASK_DEFERRED_DENSE_YIELD_INTERVAL")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let ProductComponentClassTransitions::Materialized(left_transitions) =
        &component_class_transitions[0]
    else {
        return None;
    };
    let ProductComponentClassTransitions::Materialized(right_transitions) =
        &component_class_transitions[1]
    else {
        return None;
    };
    let left_dead = components[0].dead_state();
    let right_dead = components[1].dead_state();

    let started_at = profile_timing.then(Instant::now);
    let discovery_started_at = profile_timing.then(Instant::now);
    let mut state_by_pair = vec![u32::MAX; pair_cells];
    state_by_pair[0] = 0;
    let mut pairs = vec![(0u32, 0u32)];
    let mut pending_class_transition_offsets = Vec::<u32>::new();
    pending_class_transition_offsets.push(0);
    let mut pending_class_transition_classes = Vec::<u8>::new();
    let mut pending_class_transition_targets = Vec::<u32>::new();
    let mut deferred_transition_count = 0usize;
    let mut metadata = DFA::new(1);
    metadata.ensure_group_capacity(1);
    let is_accepting = |left_state: u32, right_state: u32| {
        left.finalizers(left_state).contains(0) && right.finalizers(right_state).contains(0)
    };
    let mut accepting = vec![is_accepting(0, 0)];
    if accepting[0] && !defer_metadata {
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        metadata.overwrite_state_metadata(0, finalizers, BitSet::new(1));
    }

    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (left_state, right_state) = pairs[cursor];
        let left_row = &left_transitions[left_state as usize];
        let right_row = &right_transitions[right_state as usize];
        let mut left_index = 0usize;
        let mut right_index = 0usize;
        while left_index < left_row.len() && right_index < right_row.len() {
            let (left_class, left_target) = left_row[left_index];
            let (right_class, right_target) = right_row[right_index];
            if left_class < right_class {
                left_index += 1;
                continue;
            }
            if right_class < left_class {
                right_index += 1;
                continue;
            }
            left_index += 1;
            right_index += 1;
            if left_target == u32::MAX
                || right_target == u32::MAX
                || left_dead == Some(left_target)
                || right_dead == Some(right_target)
            {
                continue;
            }
            let pair_index = (left_target as usize)
                .checked_mul(right_states)?
                .checked_add(right_target as usize)?;
            let target = if state_by_pair[pair_index] != u32::MAX {
                state_by_pair[pair_index]
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_pair[pair_index] = target;
                pairs.push((left_target, right_target));
                let target_accepting = is_accepting(left_target, right_target);
                accepting.push(target_accepting);
                if !defer_metadata {
                    let added = metadata.add_state();
                    debug_assert_eq!(added, target);
                    if target_accepting {
                        let mut finalizers = BitSet::new(1);
                        finalizers.set(0);
                        metadata.overwrite_state_metadata(
                            target,
                            finalizers,
                            BitSet::new(1),
                        );
                    }
                }
                debug_assert_eq!(accepting.len() - 1, target as usize);
                target
            };
            if defer_transition_rows {
                deferred_transition_count += 1;
            } else {
                pending_class_transition_classes.push(left_class);
                pending_class_transition_targets.push(target);
            }
        }
        pending_class_transition_offsets.push(u32::try_from(if defer_transition_rows {
            deferred_transition_count
        } else {
            pending_class_transition_classes.len()
        }).ok()?);
        cursor += 1;
    }
    let discovery_ms = discovery_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let trace = capture_trace.then(|| ProductBuildTrace {
            components: components.to_vec(),
            coordinate_groups: identity_product_coordinate_groups(components.len()),
            state_tuples: ProductStateTuples::DenseBinary(pairs),
            state_lookup: ProductStateLookup::DenseBinary {
                right_states,
                state_by_pair,
                overflow: FxHashMap::default(),
            },
            direct_single_visible_group: true,
    });

    Some((
        DeferredDenseBinaryIntersectionProduct {
            metadata,
            accepting,
            pending_class_transition_offsets,
            pending_class_transition_classes,
            pending_class_transition_targets,
            deferred_transition_rows: defer_transition_rows.then(|| {
                DeferredDenseBinaryTransitionRows {
                    pairs: Vec::new(),
                    state_by_pair: Vec::new(),
                    left_transitions: left_transitions.clone(),
                    right_transitions: right_transitions.clone(),
                    left_dead,
                    right_dead,
                }
            }),
            class_members: class_members.to_vec(),
            left_states,
            right_states,
            pair_cells,
            num_classes,
            cooperative_yield_interval,
            discovery_ms,
            started_at,
        },
        trace,
    ))
}

/// Exact dense compiler for a single visible terminal defined as the
/// intersection of two materialized deterministic components.
///
/// The ordinary product builder stores each reachable tuple in a hash table.
/// For a binary intersection every live product state is exactly one pair and
/// any transition with a dead component is dead for the logical terminal. A
/// dense `(left_state, right_state) -> product_state` table therefore provides
/// the same construction with constant-time array lookup and no tuple hashing.
fn try_build_dense_binary_intersection_product(
    components: &[ProductComponent],
    class_members: &[Vec<u8>],
    component_class_transitions: &[ProductComponentClassTransitions],
    capture_trace: bool,
    profile_timing: bool,
) -> Option<(DFA, bool, Option<ProductBuildTrace>)> {
    let (deferred, trace) = try_discover_dense_binary_intersection_product(
        components,
        class_members,
        component_class_transitions,
        capture_trace,
        false,
        false,
        profile_timing,
    )?;
    Some((deferred.finish(), true, trace))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CompactZeroMinRepeatTailState {
    body: Option<(u32, u32)>,
    suffix_state: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CompactZeroMinRepeatState {
    Prefix(u32),
    Tail(CompactZeroMinRepeatTailState),
}

fn compact_zero_min_repeat_tail(
    state: &ZeroMinRepeatSuffixState,
) -> Option<CompactZeroMinRepeatTailState> {
    let mut body_iter = state
        .body_min_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count != u32::MAX);
    let body = body_iter
        .next()
        .map(|(body_state, &completed)| (body_state as u32, completed));
    if body_iter.next().is_some() || state.suffix_states.len() > 1 {
        return None;
    }
    let suffix_state = state.suffix_states.first().copied();
    (body.is_some() || suffix_state.is_some()).then_some(CompactZeroMinRepeatTailState {
        body,
        suffix_state,
    })
}

fn compact_zero_min_repeat_start(
    lazy: &LazyZeroMinRepeatSuffixComponent,
) -> Option<CompactZeroMinRepeatState> {
    if lazy.prefix.is_empty() {
        compact_zero_min_repeat_tail(lazy.tail_states.first()?)
            .map(CompactZeroMinRepeatState::Tail)
    } else {
        Some(CompactZeroMinRepeatState::Prefix(0))
    }
}

fn compact_zero_min_repeat_step(
    lazy: &LazyZeroMinRepeatSuffixComponent,
    state: CompactZeroMinRepeatState,
    byte: u8,
) -> Result<Option<CompactZeroMinRepeatState>, ()> {
    let tail = match state {
        CompactZeroMinRepeatState::Prefix(position) => {
            let position = position as usize;
            if lazy.prefix.get(position).copied() != Some(byte) {
                return Ok(None);
            }
            let next = position + 1;
            if next < lazy.prefix.len() {
                return Ok(Some(CompactZeroMinRepeatState::Prefix(next as u32)));
            }
            let tail = compact_zero_min_repeat_tail(lazy.tail_states.first().ok_or(())?)
                .map(CompactZeroMinRepeatState::Tail)
                .map(Some)
                .ok_or(())?;
            return Ok(tail);
        }
        CompactZeroMinRepeatState::Tail(tail) => tail,
    };

    let mut body_candidates = SmallVec::<[(u32, u32); 2]>::new();
    let mut suffix_candidates = SmallVec::<[u32; 2]>::new();

    if let Some((body_state, completed)) = tail.body
        && let Some(target) = lazy.body_dfa.step(body_state, byte)
    {
        if lazy.body_dfa.finalizers(target).contains(0) {
            let completed = completed.saturating_add(1);
            suffix_candidates.push(0);
            if completed < lazy.max as u32 {
                body_candidates.push((0, completed));
            }
        }
        if lazy.body_dfa.possible_future_group_ids(target).contains(0) {
            body_candidates.push((target, completed));
        }
    }
    if let Some(suffix_state) = tail.suffix_state
        && let Some(target) = lazy.suffix_dfa.step(suffix_state, byte)
    {
        if lazy.suffix_dfa.finalizers(target).contains(0)
            || lazy
                .suffix_dfa
                .possible_future_group_ids(target)
                .contains(0)
        {
            suffix_candidates.push(target);
        }
    }

    body_candidates.sort_unstable();
    body_candidates.dedup();
    suffix_candidates.sort_unstable();
    suffix_candidates.dedup();
    if body_candidates.is_empty() && suffix_candidates.is_empty() {
        return Ok(None);
    }
    if body_candidates.len() > 1 || suffix_candidates.len() > 1 {
        return Err(());
    }
    Ok(Some(CompactZeroMinRepeatState::Tail(
        CompactZeroMinRepeatTailState {
            body: body_candidates.first().copied(),
            suffix_state: suffix_candidates.first().copied(),
        },
    )))
}

fn compact_zero_min_repeat_is_accepting(
    lazy: &LazyZeroMinRepeatSuffixComponent,
    state: CompactZeroMinRepeatState,
) -> bool {
    match state {
        CompactZeroMinRepeatState::Prefix(_) => false,
        CompactZeroMinRepeatState::Tail(tail) => tail
            .suffix_state
            .is_some_and(|suffix_state| lazy.suffix_dfa.finalizers(suffix_state).contains(0)),
    }
}

/// Build the exact runtime-only form of a large zero-minimum bounded-repeat
/// intersection when every reachable lazy residual has at most one body
/// state/count and at most one suffix state. This is the common maxLength string
/// shape. The helper is deliberately fail-closed: any transition requiring a
/// genuine residual set returns `None`, and the general lazy product remains
/// authoritative.
fn select_lazy_zero_min_repeat_suffix_component(
    expressions: &[Expr],
) -> Option<(usize, LazyZeroMinRepeatSuffixComponent)> {
    let giant_indices = expressions
        .iter()
        .enumerate()
        .filter_map(|(index, expr)| {
            expression_contains_large_bounded_repeat(expr).then_some(index)
        })
        .collect::<SmallVec<[usize; 2]>>();
    match giant_indices.as_slice() {
        [index] => LazyZeroMinRepeatSuffixComponent::from_expr(&expressions[*index])
            .map(|lazy| (*index, lazy)),
        [] => expressions.iter().enumerate().find_map(|(index, expr)| {
            LazyZeroMinRepeatSuffixComponent::from_expr(expr).map(|lazy| (index, lazy))
        }),
        _ => None,
    }
}

fn try_compile_compact_zero_min_repeat_intersection_runtime(
    plan: &ExclusionCompilePlan,
    profile_timing: bool,
) -> Option<DeferredDfa> {
    if plan.visible_groups != 1
        || plan.compiled_exprs.len() != 2
        || !pure_binary_intersection(&plan.exclusions, &plan.intersections)
    {
        return None;
    }

    let started_at = profile_timing.then(Instant::now);
    let (lazy_index, lazy) =
        select_lazy_zero_min_repeat_suffix_component(&plan.compiled_exprs)?;
    let other_index = 1 - lazy_index;
    let other_component = compile_lazy_intersection_materialized_other_component(
        &plan.compiled_exprs[other_index],
        false,
    )?;
    let other_dfa = other_component.materialized_dfa()?;
    let other_dead = other_component.dead_state();
    let (class_map, class_members) =
        compute_lazy_zero_min_repeat_product_equivalence_classes(&lazy, other_dfa);
    let other_transitions = build_product_class_transitions_for_dfa(other_dfa, &class_map);

    let start_lazy = compact_zero_min_repeat_start(&lazy)?;
    let start = (start_lazy, 0u32);
    let mut state_by_pair = FxHashMap::<(CompactZeroMinRepeatState, u32), u32>::default();
    state_by_pair.insert(start, 0);
    let mut pairs = vec![start];
    let mut accepting = vec![compact_zero_min_repeat_is_accepting(&lazy, start_lazy)
        && other_dfa.finalizers(0).contains(0)];
    // The whole point of this runtime form is that storage is proportional to
    // reachable product residuals, not to the syntactic repetition bound.
    // Reserving `max` here made a billion-bound repeat try to reserve billions
    // of offsets even when only a handful of residuals survive the intersecting
    // automaton.
    let mut row_offsets = Vec::<u32>::new();
    row_offsets.push(0);
    let mut row_classes = Vec::<u8>::new();
    let mut row_targets = Vec::<u32>::new();

    let discovery_started_at = profile_timing.then(Instant::now);
    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let (lazy_state, other_state) = pairs[cursor];
        for &(class, other_target) in &other_transitions[other_state as usize] {
            if other_dead == Some(other_target)
                || !single_group_dfa_state_is_live(other_dfa, other_target)
            {
                continue;
            }
            let representative = *class_members[class as usize].first()?;
            let lazy_target = match compact_zero_min_repeat_step(
                &lazy,
                lazy_state,
                representative,
            ) {
                Ok(Some(target)) => target,
                Ok(None) => continue,
                Err(()) => return None,
            };
            let key = (lazy_target, other_target);
            let target = if let Some(&target) = state_by_pair.get(&key) {
                target
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_pair.insert(key, target);
                pairs.push(key);
                accepting.push(
                    compact_zero_min_repeat_is_accepting(&lazy, lazy_target)
                        && other_dfa.finalizers(other_target).contains(0),
                );
                target
            };
            row_classes.push(class);
            row_targets.push(target);
        }
        row_offsets.push(u32::try_from(row_classes.len()).ok()?);
        cursor += 1;
    }
    let discovery_ms = discovery_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    let future_started_at = profile_timing.then(Instant::now);
    let has_future = compute_single_group_futures_from_class_graph_csr(
        &accepting,
        &row_offsets,
        &row_targets,
        0,
    );
    let future_ms = future_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let dfa = DFA::new_with_single_group_metadata(
        &accepting,
        &has_future,
        expr_u8set(&plan.compiled_exprs[0]),
    );

    let mut byte_to_class = [0u8; 256];
    let class_members = class_members
        .into_iter()
        .enumerate()
        .map(|(class, bytes)| {
            for &byte in &bytes {
                byte_to_class[byte as usize] = class as u8;
            }
            bytes.into_boxed_slice()
        })
        .collect::<Vec<_>>();
    let expanded_transition_count = row_classes
        .iter()
        .map(|class| class_members[*class as usize].len())
        .sum();
    let segment = CompressedTransitionSegment {
        state_offset: 0,
        state_count: dfa.num_states() as u32,
        byte_to_class: Arc::from(byte_to_class.to_vec().into_boxed_slice()),
        class_members: Arc::from(class_members),
        row_offsets: Arc::from(row_offsets),
        entries: CompressedTransitionEntries::from_parts(row_classes, row_targets),
        expanded_transition_count,
    };
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] compact_zero_min_repeat_runtime lazy_component={} product_states={} classes={} body_states={} suffix_states={} repeat_max={} discovery_ms={:.3} future_ms={:.3} total_ms={:.3}",
            lazy_index,
            dfa.num_states(),
            segment.class_members.len(),
            lazy.body_dfa.num_states(),
            lazy.suffix_dfa.num_states(),
            lazy.max,
            discovery_ms,
            future_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(DeferredDfa::ReadyCompressed { dfa, segment })
}

fn try_compile_lazy_zero_min_repeat_intersection(
    plan: &ExclusionCompilePlan,
    profile_timing: bool,
) -> Option<(DFA, ProductBuildTrace)> {
    if plan.visible_groups != 1
        || plan.compiled_exprs.len() != 2
        || !pure_binary_intersection(&plan.exclusions, &plan.intersections)
    {
        return None;
    }

    let started_at = profile_timing.then(Instant::now);
    let (lazy_index, mut lazy) =
        select_lazy_zero_min_repeat_suffix_component(&plan.compiled_exprs)?;
    let other_index = 1 - lazy_index;
    let other_component = compile_lazy_intersection_materialized_other_component(
        &plan.compiled_exprs[other_index],
        true,
    )?;
    let other_dfa = other_component.materialized_dfa()?;
    let other_dead = other_component.dead_state();

    let (class_map, class_members) =
        compute_lazy_zero_min_repeat_product_equivalence_classes(&lazy, other_dfa);
    lazy.prepare_classes(class_members.len());
    let other_transitions = build_product_class_transitions_for_dfa(other_dfa, &class_map);

    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, expr_u8set(&plan.compiled_exprs[0]));
    let start_accepting = lazy.is_accepting(0) && other_dfa.finalizers(0).contains(0);
    if start_accepting {
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(0, finalizers, BitSet::new(1));
    }

    let mut pairs = vec![(0u32, 0u32)];
    let mut state_by_lazy_other = FxHashMap::<(u32, u32), u32>::default();
    state_by_lazy_other.insert((0, 0), 0);
    let mut pending_class_transitions = vec![Vec::<(u8, u32)>::new()];
    let discovery_started_at = profile_timing.then(Instant::now);
    let mut cursor = 0usize;
    while cursor < pairs.len() {
        let pair = pairs[cursor];
        let (lazy_state, other_state) = if lazy_index == 0 {
            (pair.0, pair.1)
        } else {
            (pair.1, pair.0)
        };
        let mut row = Vec::with_capacity(other_transitions[other_state as usize].len());
        for &(class, other_target) in &other_transitions[other_state as usize] {
            if other_dead == Some(other_target)
                || !single_group_dfa_state_is_live(other_dfa, other_target)
            {
                continue;
            }
            let representative = *class_members[class as usize].first()?;
            let Some(lazy_target) = lazy.step_class(lazy_state, class, representative) else {
                continue;
            };
            let target_pair = if lazy_index == 0 {
                (lazy_target, other_target)
            } else {
                (other_target, lazy_target)
            };
            let lookup_key = (lazy_target, other_target);
            let target = if let Some(&existing) = state_by_lazy_other.get(&lookup_key) {
                existing
            } else {
                let target = u32::try_from(pairs.len()).ok()?;
                state_by_lazy_other.insert(lookup_key, target);
                pairs.push(target_pair);
                pending_class_transitions.push(Vec::new());
                let added = dfa.add_state();
                debug_assert_eq!(added, target);
                let accepting = lazy.is_accepting(lazy_target)
                    && other_dfa.finalizers(other_target).contains(0);
                if accepting {
                    let mut finalizers = BitSet::new(1);
                    finalizers.set(0);
                    dfa.overwrite_state_metadata(target, finalizers, BitSet::new(1));
                }
                target
            };
            row.push((class, target));
        }
        pending_class_transitions[cursor] = row;
        cursor += 1;
    }
    let discovery_ms = discovery_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let future_started_at = profile_timing.then(Instant::now);
    set_single_group_futures_from_class_graph(&mut dfa, &pending_class_transitions);
    let future_ms = future_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let expansion_started_at = profile_timing.then(Instant::now);
    let expanded_transitions = pending_class_transitions
        .into_par_iter()
        .map(|class_transitions| {
            let byte_capacity = class_transitions
                .iter()
                .map(|(class, _)| class_members[*class as usize].len())
                .sum();
            let mut transitions = Vec::with_capacity(byte_capacity);
            for (class, target) in class_transitions {
                for &byte in &class_members[class as usize] {
                    transitions.push((byte, target));
                }
            }
            if transitions.len() > 1 {
                transitions.sort_unstable_by_key(|entry| entry.0);
            }
            crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions)
        })
        .collect::<Vec<_>>();
    for (state, transitions) in dfa.states_mut().iter_mut().zip(expanded_transitions) {
        state.transitions = transitions;
    }
    let expansion_ms = expansion_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let component_started_at = profile_timing.then(Instant::now);
    let body_states = lazy.body_dfa.num_states();
    let suffix_states = lazy.suffix_dfa.num_states();
    let repeat_max = lazy.max;
    let lazy_component = lazy.into_product_component(&class_members);
    let lazy_states = lazy_component
        .materialized_dfa()
        .map(DFA::num_states)?;
    let other_states = other_dfa.num_states();
    let components = if lazy_index == 0 {
        vec![lazy_component, other_component]
    } else {
        vec![other_component, lazy_component]
    };
    let component_ms = component_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let lookup_started_at = profile_timing.then(Instant::now);
    let mut sparse_lookup = FxHashMap::<ProductStateTuple, u32>::default();
    sparse_lookup.reserve(pairs.len());
    for (state, &(left, right)) in pairs.iter().enumerate() {
        let mut tuple = ProductStateTuple::new();
        tuple.push((0, left));
        tuple.push((1, right));
        sparse_lookup.insert(tuple, state as u32);
    }
    let trace = ProductBuildTrace {
        components,
        coordinate_groups: identity_product_coordinate_groups(2),
        state_tuples: ProductStateTuples::DenseBinary(pairs),
        state_lookup: ProductStateLookup::Hash(sparse_lookup),
        direct_single_visible_group: true,
    };
    let lookup_ms = lookup_started_at
        .map(|started| started.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] lazy_zero_min_repeat_intersection lazy_component={} lazy_states={} other_states={} product_states={} classes={} body_states={} suffix_states={} repeat_max={} discovery_ms={:.3} future_ms={:.3} expansion_ms={:.3} component_ms={:.3} lookup_ms={:.3} total_ms={:.3}",
            lazy_index,
            lazy_states,
            other_states,
            dfa.num_states(),
            class_members.len(),
            body_states,
            suffix_states,
            repeat_max,
            discovery_ms,
            future_ms,
            expansion_ms,
            component_ms,
            lookup_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some((dfa, trace))
}

fn try_compile_with_plan_deferred_dense(
    plan: ExclusionCompilePlan,
) -> Result<(DeferredDfa, Option<ProductBuildTrace>), ExclusionCompilePlan> {
    try_compile_with_plan_deferred_dense_min_pair_cells(plan, 0, false)
}

fn try_compile_with_plan_deferred_dense_min_pair_cells(
    plan: ExclusionCompilePlan,
    min_pair_cells: usize,
    retain_transition_rows_during_discovery: bool,
) -> Result<(DeferredDfa, Option<ProductBuildTrace>), ExclusionCompilePlan> {
    if plan.visible_groups != 1
        || plan.compiled_exprs.len() != 2
        || !pure_binary_intersection(&plan.exclusions, &plan.intersections)
    {
        return Err(plan);
    }

    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let profile_timing = profile_detail
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    // For giant nested repeats this is a correctness/safety lane, not merely a
    // performance optimization. A debug override must not route an expression
    // that validation accepted back into eager repeat materialization.
    let requires_lazy_giant = plan
        .compiled_exprs
        .iter()
        .any(expression_contains_large_bounded_repeat);
    let lazy_zero_min_repeat_product = requires_lazy_giant
        || std::env::var_os("GLRMASK_DISABLE_LAZY_ZERO_MIN_REPEAT_PRODUCT").is_none();
    if retain_transition_rows_during_discovery
        && lazy_zero_min_repeat_product
        && let Some(runtime) =
            try_compile_compact_zero_min_repeat_intersection_runtime(&plan, profile_timing)
    {
        return Ok((runtime, None));
    }
    if lazy_zero_min_repeat_product
        && let Some((dfa, trace)) =
            try_compile_lazy_zero_min_repeat_intersection(&plan, profile_timing)
    {
        return Ok((DeferredDfa::Ready(dfa), Some(trace)));
    }
    let (components, component_cache_hits, _, _) =
        compile_product_components_profiled(
            &plan.compiled_exprs,
            profile_detail,
            true,
            true,
            false,
        );
    let pair_cells = components
        .first()
        .and_then(ProductComponent::materialized_dfa)
        .and_then(|left| {
            components
                .get(1)
                .and_then(ProductComponent::materialized_dfa)
                .and_then(|right| left.num_states().checked_mul(right.num_states()))
        });
    if pair_cells.is_none_or(|pair_cells| pair_cells < min_pair_cells) {
        return Err(plan);
    }
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] deferred_product_component_cache groups={} unique_components={} cache_hits={}",
            plan.compiled_exprs.len(),
            plan.compiled_exprs.len() - component_cache_hits,
            component_cache_hits,
        );
    }
    let (class_map, class_members) = compute_product_equivalence_classes(&components);
    let component_class_transitions = build_product_class_transitions(&components, &class_map);
    let defer_runtime_materialization =
        std::env::var("GLRMASK_DEFER_DENSE_RUNTIME_MATERIALIZATION")
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or_else(|_| rayon::current_num_threads() > 1);
    let retain_transition_rows_during_discovery =
        retain_transition_rows_during_discovery && defer_runtime_materialization;
    let Some((mut deferred, trace)) = try_discover_dense_binary_intersection_product(
        &components,
        &class_members,
        &component_class_transitions,
        !retain_transition_rows_during_discovery,
        defer_runtime_materialization && !retain_transition_rows_during_discovery,
        defer_runtime_materialization,
        profile_timing,
    ) else {
        return Err(plan);
    };

    deferred.ensure_group_capacity(1);
    deferred.set_group_u8set(0, expr_u8set(&plan.compiled_exprs[0]));
    Ok((DeferredDfa::DenseBinary(deferred), trace))
}

/// Return whether one terminal expression has the exact pure binary-product
/// shape supported by the compressed runtime tokenizer builder.
pub fn expression_supports_deferred_dense_runtime(expr: &Expr) -> bool {
    // `build_exclusion_compile_plan([expr])` can be expensive because it
    // materializes nested group operations before returning its outer plan.
    // A pure binary intersection is possible only when the top-level group-op
    // chain contributes exactly one intersection and no exclusions. Nested
    // group operations inside either operand are materialized into that
    // operand and cannot create another outer plan component. Reject every
    // other shape before doing the expensive exact plan construction; all
    // plausible candidates still go through the historical proof below.
    fn outer_group_op_counts(expr: &Expr) -> (usize, usize) {
        match expr {
            Expr::Exclude { expr, .. } => {
                let (exclusions, intersections) = outer_group_op_counts(expr);
                (exclusions + 1, intersections)
            }
            Expr::Intersect { expr, .. } => {
                let (exclusions, intersections) = outer_group_op_counts(expr);
                (exclusions, intersections + 1)
            }
            Expr::Shared(inner)
                if matches!(inner.as_ref(), Expr::Exclude { .. } | Expr::Intersect { .. }) =>
            {
                outer_group_op_counts(inner)
            }
            _ => (0, 0),
        }
    }
    if outer_group_op_counts(expr) != (0, 1) {
        return false;
    }
    let plan = build_exclusion_compile_plan(std::slice::from_ref(expr));
    plan.visible_groups == 1
        && plan.compiled_exprs.len() == 2
        && pure_binary_intersection(&plan.exclusions, &plan.intersections)
}

/// Whether the exact general residual runtime has the bounded-code liveness
/// certificate needed to make this expression a safe protected dynamic
/// component. This is an internal representation-policy predicate; it does not
/// change the accepted language.
#[doc(hidden)]
pub fn expression_supports_bounded_code_residual_runtime(expr: &Expr) -> bool {
    super::runtime_residual::expression_supports_bounded_code_liveness_oracle(expr)
}

/// Cheap shape-only prefilter used by Static compilation before the preserving
/// residual constructor performs the full exact oracle certification.
#[doc(hidden)]
pub fn expression_may_support_bounded_code_residual_runtime(expr: &Expr) -> bool {
    super::runtime_residual::expression_may_support_bounded_code_liveness_oracle(expr)
}

/// Whether this expression contains any bounded repetition large enough that
/// falling through to the ordinary repeat compiler could allocate in direct
/// proportion to the declared bound.
#[doc(hidden)]
pub fn expression_contains_large_bounded_repeat(expr: &Expr) -> bool {
    match unwrap_shared(expr) {
        Expr::Repeat { expr, max, .. } => {
            max.is_some_and(|max| max >= VIRTUAL_BINARY_REPEAT_MIN_BOUND)
                || expression_contains_large_bounded_repeat(expr)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            parts.iter().any(expression_contains_large_bounded_repeat)
        }
        Expr::Intersect { expr, intersect } => {
            expression_contains_large_bounded_repeat(expr)
                || expression_contains_large_bounded_repeat(intersect)
        }
        Expr::Exclude { expr, exclude } => {
            expression_contains_large_bounded_repeat(expr)
                || expression_contains_large_bounded_repeat(exclude)
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
        Expr::Shared(_) => unreachable!("unwrap_shared removes Shared"),
    }
}

const LAZY_INTERSECTION_OTHER_MATERIALIZED_STATE_BUDGET: usize = 100_000;
const LAZY_INTERSECTION_OTHER_MATERIALIZED_TRANSITION_BUDGET: usize = 1_000_000;

/// Compile the ordinary side of the lazy zero-min-repeat/suffix intersection
/// lane without ever expanding a giant bounded repeat. The generic product
/// compiler represents deterministic bounded roots with `VirtualBoundedRepeat`
/// once their bound reaches the direct-repeat threshold, even when the exact
/// materialized DFA would still be tiny. That representation choice should not
/// force the lazy intersection lane to reject an otherwise cheap component.
///
/// Keep the ordinary coordinate independent of the giant repeat bound. Its
/// accepted language must be finite, so its live DFA is acyclic and bounds the
/// length of every synchronized product walk. For a non-giant virtual repeat,
/// the new layered expansion is additionally limited by its exact
/// `(max + 1) * base_states` footprint and a conservative
/// `max * base_transitions` bound. Validation calls this same helper, so an
/// expression cannot be admitted under a weaker proof than compilation uses.
fn compile_lazy_intersection_materialized_other_component(
    expr: &Expr,
    preserve_coordinates: bool,
) -> Option<ProductComponent> {
    if expression_contains_large_bounded_repeat(expr) {
        return None;
    }
    let component =
        compile_product_component_with_options(expr, preserve_coordinates, true, None);
    let component = match component {
        component @ (ProductComponent::Materialized(_)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { .. }) => Some(component),
        ProductComponent::VirtualBoundedRepeat {
            base_dfa,
            min,
            max,
        } => {
            let total_states = (max as usize + 1).checked_mul(base_dfa.num_states())?;
            let total_transitions = (max as usize).checked_mul(dfa_transition_count(&base_dfa))?;
            if total_states > LAZY_INTERSECTION_OTHER_MATERIALIZED_STATE_BUDGET
                || total_transitions > LAZY_INTERSECTION_OTHER_MATERIALIZED_TRANSITION_BUDGET
            {
                return None;
            }
            let mut dfa = build_bounded_repeat_dfa_from_base(
                base_dfa.as_ref(),
                min as usize,
                max as usize,
            )?;
            dfa.ensure_group_capacity(1);
            dfa.set_group_u8set(0, expr_u8set(expr));
            Some(ProductComponent::Materialized(Arc::new(dfa)))
        }
        ProductComponent::VirtualFixedSequence { .. } => None,
    }?;
    product_component_has_finite_language(&component).then_some(component)
}

/// Whether the language accepted by one already-materializable product
/// component is finite.  We only care about states that are both reachable
/// from the component root and can still participate in an accepting word;
/// that live subgraph accepts an infinite language iff it contains a cycle.
///
/// This is a resource-safety proof for pairing an exact virtual bounded-repeat
/// coordinate with an ordinary product coordinate.  If the ordinary language
/// is finite, its live DAG bounds every product walk independently of the
/// repeat's declared (possibly billion-scale) upper bound.
fn single_group_dfa_state_is_live(dfa: &DFA, state: u32) -> bool {
    dfa.finalizers(state).contains(0) || dfa.possible_future_group_ids(state).contains(0)
}

fn single_group_dfa_has_finite_language(dfa: &DFA) -> bool {
    if dfa.num_states() == 0 {
        return true;
    }

    let is_live = |state: u32| single_group_dfa_state_is_live(dfa, state);
    if !is_live(0) {
        return true;
    }

    let mut reachable = vec![false; dfa.num_states()];
    let mut stack = vec![0u32];
    reachable[0] = true;
    while let Some(state) = stack.pop() {
        for (_, &target) in dfa.states()[state as usize].transitions.iter() {
            if is_live(target) && !reachable[target as usize] {
                reachable[target as usize] = true;
                stack.push(target);
            }
        }
    }

    let mut indegree = vec![0u32; dfa.num_states()];
    let mut live_count = 0usize;
    for state in 0..dfa.num_states() as u32 {
        if !reachable[state as usize] {
            continue;
        }
        live_count += 1;
        for (_, &target) in dfa.states()[state as usize].transitions.iter() {
            if reachable[target as usize] {
                indegree[target as usize] = indegree[target as usize].saturating_add(1);
            }
        }
    }
    let mut queue = VecDeque::new();
    for state in 0..dfa.num_states() as u32 {
        if reachable[state as usize] && indegree[state as usize] == 0 {
            queue.push_back(state);
        }
    }
    let mut removed = 0usize;
    while let Some(state) = queue.pop_front() {
        removed += 1;
        for (_, &target) in dfa.states()[state as usize].transitions.iter() {
            if !reachable[target as usize] {
                continue;
            }
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    removed == live_count
}

/// Exact maximum accepted suffix length from every live state of a
/// deterministic one-group DFA. Returns `None` if the live subgraph contains a
/// cycle or inconsistent future metadata, proving that no finite bound is
/// available. Dead states are represented by `None` entries.
fn single_group_dfa_max_remaining(dfa: &DFA) -> Option<Vec<Option<u32>>> {
    fn visit(
        dfa: &DFA,
        state: u32,
        colors: &mut [u8],
        memo: &mut [Option<u32>],
    ) -> Option<u32> {
        let index = state as usize;
        match *colors.get(index)? {
            1 => return None,
            2 => return *memo.get(index)?,
            _ => {}
        }
        if !single_group_dfa_state_is_live(dfa, state) {
            colors[index] = 2;
            return None;
        }
        colors[index] = 1;
        let mut best = dfa.finalizers(state).contains(0).then_some(0u32);
        for (_, target) in dfa.transitions(state) {
            if !single_group_dfa_state_is_live(dfa, target) {
                continue;
            }
            let remaining = visit(dfa, target, colors, memo)?.checked_add(1)?;
            best = Some(best.map_or(remaining, |current| current.max(remaining)));
        }
        let best = best?;
        memo[index] = Some(best);
        colors[index] = 2;
        Some(best)
    }

    let mut colors = vec![0u8; dfa.num_states()];
    let mut memo = vec![None; dfa.num_states()];
    for state in 0..dfa.num_states() as u32 {
        if single_group_dfa_state_is_live(dfa, state) && colors[state as usize] == 0 {
            visit(dfa, state, &mut colors, &mut memo)?;
        }
    }
    Some(memo)
}

fn product_component_has_finite_language(component: &ProductComponent) -> bool {
    match component {
        ProductComponent::Materialized(dfa)
        | ProductComponent::MaterializedZeroMinRepeatSuffix { dfa, .. } => {
            single_group_dfa_has_finite_language(dfa)
        }
        ProductComponent::VirtualFixedSequence { .. } => true,
        ProductComponent::VirtualBoundedRepeat { .. } => false,
    }
}

fn refine_u8_partitions(partitions: Vec<U8Set>, split: U8Set) -> Vec<U8Set> {
    let mut next_partitions = Vec::with_capacity(partitions.len() * 2);
    for partition in partitions {
        let intersection = partition.intersection(&split);
        let difference = partition.difference(&split);
        if !intersection.is_empty() {
            next_partitions.push(intersection);
        }
        if !difference.is_empty() {
            next_partitions.push(difference);
        }
    }
    next_partitions
}

/// Compile multiple expressions into a single NFA (without determinization).
///
/// Each expression's index becomes its group ID. Equal literal/class prefixes
/// are shared recursively. The previous implementation factored at most one
/// prefix common to *every* terminal, which left large families of line-like
/// terminals as thousands of duplicated suffix chains after their first byte.
fn build_regex_nfa(exprs: &[Expr]) -> NFA {
    build_regex_nfa_impl(exprs)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DeterministicPrefixAtom {
    Byte(u8),
    Class(U8Set),
}

fn deterministic_prefix(expr: &Expr) -> (Vec<DeterministicPrefixAtom>, Expr) {
    fn collect(expr: &Expr, atoms: &mut Vec<DeterministicPrefixAtom>) -> Expr {
        match expr {
            Expr::Shared(inner) => collect(inner, atoms),
            Expr::U8Seq(bytes) => {
                atoms.extend(bytes.iter().copied().map(DeterministicPrefixAtom::Byte));
                Expr::Epsilon
            }
            Expr::U8Class(bytes) => {
                atoms.push(DeterministicPrefixAtom::Class(*bytes));
                Expr::Epsilon
            }
            Expr::Seq(parts) => {
                for (index, part) in parts.iter().enumerate() {
                    let remainder = collect(part, atoms);
                    if !matches!(remainder, Expr::Epsilon) {
                        let mut tail = Vec::with_capacity(parts.len() - index);
                        tail.push(remainder);
                        tail.extend_from_slice(&parts[index + 1..]);
                        return seq_from_parts(tail);
                    }
                }
                Expr::Epsilon
            }
            Expr::Dfa(_)
            | Expr::Intersect { .. }
            | Expr::Choice(_)
            | Expr::Exclude { .. }
            | Expr::Repeat { .. }
            | Expr::Epsilon => expr.clone(),
        }
    }

    let mut atoms = Vec::new();
    let remainder = collect(expr, &mut atoms);
    (atoms, remainder)
}

fn append_prefix_shared_expressions(
    nfa: &mut NFA,
    expressions: Vec<(u32, Expr)>,
) {
    let mut transitions = FxHashMap::<(u32, DeterministicPrefixAtom), u32>::default();
    let mut finals = Vec::<(u32, u32)>::new();
    let mut opaque = Vec::<(u32, u32, Expr)>::new();

    for (group_id, expr) in expressions {
        let (prefix, remainder) = deterministic_prefix(&expr);
        let mut state = 0u32;
        for atom in prefix {
            let key = (state, atom.clone());
            state = if let Some(&target) = transitions.get(&key) {
                target
            } else {
                let target = nfa.add_state();
                match &atom {
                    DeterministicPrefixAtom::Byte(byte) => {
                        append_compiled_expr(&Expr::U8Seq(vec![*byte]), nfa, state, target);
                    }
                    DeterministicPrefixAtom::Class(bytes) => {
                        append_compiled_expr(&Expr::U8Class(*bytes), nfa, state, target);
                    }
                }
                transitions.insert(key, target);
                target
            };
        }
        if matches!(remainder, Expr::Epsilon) {
            finals.push((state, group_id));
        } else {
            opaque.push((state, group_id, remainder));
        }
    }

    for (state, group_id) in finals {
        nfa.add_finalizer(state, group_id);
    }
    for (state, group_id, expr) in opaque {
        let accept = nfa.add_state();
        append_compiled_expr(&expr, nfa, state, accept);
        nfa.add_finalizer(accept, group_id);
    }
}

fn build_regex_nfa_impl(exprs: &[Expr]) -> NFA {
    let optimized_exprs = exprs
        .iter()
        .cloned()
        .map(Expr::optimize)
        .enumerate()
        .map(|(group_id, expr)| (group_id as u32, expr))
        .collect::<Vec<_>>();

    let mut nfa = NFA::new(1);
    append_prefix_shared_expressions(&mut nfa, optimized_exprs);
    nfa
}

#[cfg(test)]
mod tests {
    use super::super::dfa::DFA;
    use super::super::Lexer;
    use super::{
        build_regex,
        build_regex_local_small_product,
        build_regex_monolithic,
        build_regex_partitioned_with_adaptive,
        try_product_union_components,
    };
    use super::{compile_product_component_dfa, compile_product_component_dfa_direct};
    use super::factor_regex_expr;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::regex::parse_regex;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::ds::bitset::BitSet;
    use crate::ds::u8set::U8Set;
    use crate::Vocab;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
    use std::sync::Arc;

    fn byte_expr(byte: u8) -> Expr {
        Expr::U8Seq(vec![byte])
    }

    fn byte_choice(bytes: &[u8]) -> Expr {
        Expr::Choice(bytes.iter().copied().map(byte_expr).collect())
    }

    #[test]
    fn bounded_suffix_preflight_finds_expensive_nested_component() {
        let inner = Expr::Seq(vec![
            byte_expr(b'<'),
            Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(20_000),
            },
            byte_expr(b'>'),
        ]);
        let own = super::direct_bounded_suffix_state_count_estimate(&inner)
            .expect("direct bounded suffix should have an exact state estimate");
        assert!(own > 10_000);

        let wrapped = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![byte_expr(b'x'), inner.clone()])),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: None,
            }),
        };
        assert_eq!(super::direct_bounded_suffix_state_count_estimate(&wrapped), None);
        assert_eq!(super::max_direct_bounded_suffix_state_count_estimate(&wrapped), Some(own));
    }

    #[test]
    fn lazy_zero_min_repeat_suffix_accepts_last_non_sentinel_bound() {
        let expr_for = |max| {
            Expr::Seq(vec![
                byte_expr(b'['),
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 0,
                    max: Some(max),
                },
                byte_expr(b'z'),
            ])
        };

        assert!(super::LazyZeroMinRepeatSuffixComponent::from_expr(
            &expr_for((u32::MAX - 1) as usize),
        )
        .is_some());
        assert!(super::LazyZeroMinRepeatSuffixComponent::from_expr(
            &expr_for(u32::MAX as usize),
        )
        .is_none());
    }

    #[test]
    fn lazy_zero_min_repeat_suffix_accepts_empty_literal_prefix() {
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            },
            Expr::U8Class(U8Set::from_bytes(b"bc")),
        ]);
        let mut component = super::LazyZeroMinRepeatSuffixComponent::from_expr(&expr)
            .expect("zero-prefix lazy repeat+suffix component");
        assert_eq!(component.prefix_len(), 0);
        let start = component.start_state();
        assert!(!component.is_accepting(start));
        assert!(component.has_future(start));
        let suffix_target = component
            .step_uncached(start, b'b')
            .expect("zero copies may enter the suffix immediately");
        assert!(component.is_accepting(suffix_target));
        let trace = super::zero_min_repeat_suffix_component_trace(&expr)
            .expect("zero-prefix structural trace must rebuild from the expression");
        assert_eq!(trace.prefix_len, 0);
    }

    fn terminal_matches(expr: Expr, input: &[u8]) -> bool {
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };
        let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
        exec.matches
            .iter()
            .any(|matched| matched.id == 0 && matched.width == input.len())
    }

    fn dfa_accepts(dfa: &DFA, input: &[u8]) -> bool {
        let mut state = 0u32;
        for &byte in input {
            let Some(next) = dfa.step(state, byte) else {
                return false;
            };
            state = next;
        }
        dfa.finalizers(state).contains(0)
    }

    fn enumerate_inputs(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        fn extend(out: &mut Vec<Vec<u8>>, prefix: &mut Vec<u8>, alphabet: &[u8], max_len: usize) {
            out.push(prefix.clone());
            if prefix.len() == max_len {
                return;
            }
            for &byte in alphabet {
                prefix.push(byte);
                extend(out, prefix, alphabet, max_len);
                prefix.pop();
            }
        }

        let mut out = Vec::new();
        extend(&mut out, &mut Vec::new(), alphabet, max_len);
        out
    }

    fn assert_dfa_observation_equivalent(left: &DFA, right: &DFA) {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::from([(Some(0u32), Some(0u32))]);
        while let Some((left_state, right_state)) = queue.pop_front() {
            if !seen.insert((left_state, right_state)) {
                continue;
            }
            let left_accepting = left_state
                .is_some_and(|state| !left.finalizers(state).is_empty());
            let right_accepting = right_state
                .is_some_and(|state| !right.finalizers(state).is_empty());
            assert_eq!(left_accepting, right_accepting, "acceptance mismatch at {left_state:?}/{right_state:?}");

            let left_future = left_state
                .is_some_and(|state| left.possible_future_group_ids(state).contains(0));
            let right_future = right_state
                .is_some_and(|state| right.possible_future_group_ids(state).contains(0));
            assert_eq!(left_future, right_future, "future mismatch at {left_state:?}/{right_state:?}");

            for byte in 0u8..=255 {
                let next = (
                    left_state.and_then(|state| left.step(state, byte)),
                    right_state.and_then(|state| right.step(state, byte)),
                );
                if next != (None, None) && !seen.contains(&next) {
                    queue.push_back(next);
                }
            }
        }
    }

    #[test]
    fn unique_dense_exclusion_skips_atomic_rhs_languages() {
        assert!(!super::unique_dense_exclusion_rhs_is_compound(&Expr::U8Seq(
            b"literal".to_vec()
        )));
        assert!(!super::unique_dense_exclusion_rhs_is_compound(&Expr::U8Class(
            U8Set::from_bytes(b"ab")
        )));
        assert!(!super::unique_dense_exclusion_rhs_is_compound(&Expr::Epsilon));
        let dfa = super::compile_expr_to_dfa(&Expr::U8Seq(b"dfa".to_vec()));
        assert!(!super::unique_dense_exclusion_rhs_is_compound(&Expr::Dfa(
            Arc::new(dfa)
        )));

        assert!(super::unique_dense_exclusion_rhs_is_compound(&Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ])));
        assert!(super::unique_dense_exclusion_rhs_is_compound(&Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ])));
    }

    #[test]
    fn dense_binary_exclusion_matches_general_exclusion() {
        let left_expr = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
            Expr::Seq(vec![
                Expr::U8Seq(b"c".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                    min: 0,
                    max: Some(2),
                },
            ]),
        ]);
        let right_expr = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
            Expr::U8Seq(b"caa".to_vec()),
        ]);
        let expr = Expr::Exclude {
            expr: Box::new(left_expr.clone()),
            exclude: Box::new(right_expr.clone()),
        };
        let expected = compile_product_component_dfa(&expr);
        let left = super::compile_expr_to_dfa(&left_expr);
        let right = super::compile_expr_to_dfa(&right_expr);
        let direct = super::build_dense_binary_exclusion_dfa(
            &left,
            &right,
            super::expr_u8set(&expr),
        )
        .expect("dense binary exclusion should apply");

        assert_dfa_observation_equivalent(&expected, &direct);
        for input in enumerate_inputs(b"abc", 4) {
            assert_eq!(
                dfa_accepts(&expected, &input),
                dfa_accepts(&direct, &input),
                "language mismatch for {input:?}",
            );
        }
    }

    fn expression_graph_symbols() -> Vec<Expr> {
        vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::Choice(vec![Expr::Epsilon, Expr::U8Seq(b"c".to_vec())]),
            Expr::Choice(vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"ab".to_vec())]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"d".to_vec())),
                min: 0,
                max: Some(2),
            },
        ]
    }

    #[test]
    fn local_small_product_compilation_matches_default_product() {
        let expressions = vec![
            Expr::U8Seq(b"alpha".to_vec()),
            Expr::Seq(vec![
                Expr::U8Seq(b"b".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"cd"))),
                    min: 1,
                    max: Some(4),
                },
            ]),
            Expr::Choice(vec![
                Expr::U8Seq(b"cat".to_vec()),
                Expr::U8Seq(b"car".to_vec()),
            ]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"xy"))),
                min: 0,
                max: Some(3),
            },
        ];
        let ordinary = build_regex(&expressions).dfa;
        let local = build_regex_local_small_product(&expressions).dfa;
        assert_dfa_observation_equivalent(&ordinary, &local);
    }

    #[test]
    fn grouped_expression_graph_matches_byte_nfa_with_nullable_overlap_and_cycles() {
        use crate::automata::unweighted_u32::nfa::NFA as GraphNfa;

        let mut graph = GraphNfa::new_empty();
        for _ in 0..7 {
            graph.add_state();
        }
        graph.start_states = vec![0, 1];
        graph.add_epsilon(0, 2);
        graph.add_epsilon(2, 0);
        graph.add_transition(0, 0, 3);
        graph.add_transition(1, 3, 3);
        graph.add_transition(2, 2, 4);
        graph.add_transition(3, 4, 3);
        graph.add_transition(3, 2, 5);
        graph.add_transition(3, 2, 6);
        graph.add_epsilon(4, 3);
        graph.add_transition(5, 1, 6);
        graph.set_accepting(6);

        let symbols = expression_graph_symbols();
        let grouped = super::try_compile_expression_labeled_nfa_direct(
            &graph,
            &symbols,
            true,
        )
        .unwrap()
        .expect("forced direct compiler should select the graph");
        let reference = super::compile_expression_labeled_nfa_via_byte_nfa(&graph, &symbols)
            .expect("byte-NFA reference should compile");
        assert_dfa_observation_equivalent(&grouped, &reference);
    }

    #[test]
    fn grouped_expression_graph_seeded_differential() {
        use crate::automata::unweighted_u32::nfa::NFA as GraphNfa;

        let symbols = expression_graph_symbols();
        let mut rng = StdRng::seed_from_u64(0xD1EC_7E67_2026_0731);
        for case in 0..48 {
            let state_count = rng.gen_range(2..=8);
            let mut graph = GraphNfa::new_empty();
            for _ in 0..state_count {
                graph.add_state();
            }
            graph.start_states.push(rng.gen_range(0..state_count) as u32);
            if rng.gen_bool(0.35) {
                let second = rng.gen_range(0..state_count) as u32;
                if !graph.start_states.contains(&second) {
                    graph.start_states.push(second);
                }
            }
            graph.set_accepting(rng.gen_range(0..state_count) as u32);
            for source in 0..state_count as u32 {
                if rng.gen_bool(0.45) {
                    graph.add_epsilon(source, rng.gen_range(0..state_count) as u32);
                }
                for _ in 0..rng.gen_range(1..=3) {
                    graph.add_transition(
                        source,
                        rng.gen_range(0..symbols.len()) as i32,
                        rng.gen_range(0..state_count) as u32,
                    );
                }
            }

            let grouped = super::try_compile_expression_labeled_nfa_direct(
                &graph,
                &symbols,
                true,
            )
            .unwrap()
            .expect("forced direct compiler should select the graph");
            let reference = super::compile_expression_labeled_nfa_via_byte_nfa(&graph, &symbols)
                .expect("byte-NFA reference should compile");
            assert_dfa_observation_equivalent(&grouped, &reference);
            assert_eq!(
                grouped.num_states(),
                reference.num_states(),
                "minimal state count differs in case {case}",
            );
        }
    }

    fn component_pair_for_map_test(
        full_states: usize,
        synthesized_states: usize,
        full_to_synthesized: Vec<u32>,
        protected_residual: bool,
    ) -> super::LexerComponentPair {
        super::LexerComponentPair {
            terminal_ids: vec![0],
            synthesized: DFA::new(synthesized_states),
            full: super::DeferredDfa::Ready(DFA::new(full_states)),
            full_to_synthesized,
            protected_residual,
        }
    }

    #[test]
    fn partitioned_component_maps_compose_identity_and_protected_offsets() {
        let components = vec![
            component_pair_for_map_test(2, 2, vec![0, 1], false),
            component_pair_for_map_test(3, 2, vec![0, 1, 1], true),
        ];
        let map = super::compose_partitioned_component_state_maps(&components, &[1, 3])
            .expect("complete component maps compose");

        assert_eq!(map, vec![0, 1, 2, 3, 4, 4]);
        assert_eq!(map.len(), 1 + 2 + 3);
    }

    #[test]
    fn partitioned_component_map_composition_rejects_incomplete_or_invalid_maps() {
        let incomplete = vec![component_pair_for_map_test(3, 2, vec![0, 1], true)];
        assert!(super::compose_partitioned_component_state_maps(&incomplete, &[1]).is_none());

        let invalid_target = vec![component_pair_for_map_test(2, 2, vec![0, 2], true)];
        assert!(
            super::compose_partitioned_component_state_maps(&invalid_target, &[1]).is_none()
        );

        let wrong_offset = vec![component_pair_for_map_test(2, 2, vec![0, 1], false)];
        assert!(super::compose_partitioned_component_state_maps(&wrong_offset, &[2]).is_none());
    }

    #[test]
    fn rebuilt_mixed_product_expression_preserves_group_operation_semantics() {
        let components = vec![byte_choice(b"ab"), byte_expr(b'b'), byte_expr(b'a')];
        let exclusions = BTreeMap::from([(0, BTreeSet::from([1]))]);
        let intersections = BTreeMap::from([(0, BTreeSet::from([2]))]);
        let rebuilt = super::rebuild_single_visible_group_expression(
            &components,
            &exclusions,
            &intersections,
        )
        .expect("single visible product reconstruction");

        for input in enumerate_inputs(b"ab", 2) {
            assert_eq!(terminal_matches(rebuilt.clone(), &input), input == b"a");
        }
    }

    #[test]
    fn vocabulary_repeat_horizon_cache_allows_nested_parallel_misses() {
        use rayon::prelude::*;

        let body = Arc::new(super::compile_expr_to_dfa(&Expr::Choice(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"aba".to_vec()),
        ])));
        let vocab = Arc::new(Vocab::new(
            (0..256)
                .map(|token| {
                    let length = token as usize % 32 + 1;
                    (token, b"ab".repeat(length))
                })
                .collect(),
        ));
        let cache = Arc::new(super::VocabularyRepeatHorizonCache::new());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool");
        let results = pool.install(|| {
            (0..16)
                .into_par_iter()
                .map(|_| cache.horizon_for_dfa(&body, &vocab))
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(|result| *result == results[0]));
    }

    #[test]
    fn vocabulary_repeat_horizon_is_suffix_closed_and_residual_aware() {
        let body = Expr::U8Seq(b"ab".to_vec());
        let vocab = Vocab::new(vec![(0, b"xabab".to_vec())]);

        // The token suffix "abab" crosses two body boundaries from the body
        // start state. The analysis must consider that suffix even though it is
        // not itself a vocabulary entry.
        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(2),
        );

        let vocab = Vocab::new(vec![(0, b"babab".to_vec())]);
        // Starting from the reachable residual after an initial 'a', this token
        // completes one pending copy and two further copies.
        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(3),
        );
    }

    #[test]
    fn vocabulary_repeat_horizon_counts_boundaries_before_token_exit() {
        let body = Expr::U8Seq(b"a".to_vec());
        let vocab = Vocab::new(vec![(0, b"aaaaX".to_vec())]);

        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(4),
        );
    }

    #[test]
    fn vocabulary_repeat_horizon_respects_dominance_not_path_count() {
        let body = parse_regex("a+", false);
        let vocab = Vocab::new(vec![(0, b"aaaaaaaa".to_vec())]);

        // Although there are paths that choose a repeat boundary after every
        // byte, the zero-boundary path at the same residual dominates all of
        // them. Consequently an upper repetition counter is unobservable for
        // this body language and the translation displacement is exactly zero.
        assert_eq!(
            super::vocabulary_repeat_boundary_horizon(&body, &vocab),
            Some(0),
        );
    }

    #[test]
    fn zero_min_repeat_transport_reserves_live_count_spread() {
        let ambiguous_body = Expr::Choice(vec![
            byte_expr(b'a'),
            Expr::U8Seq(b"aaaaaaaa".to_vec()),
        ]);
        let body_dfa = super::compile_expr_to_dfa(&ambiguous_body);
        let suffix_dfa = super::compile_expr_to_dfa(&byte_expr(b'b'));
        let trace = |max| {
            let built = super::build_zero_min_repeat_suffix_dominance_dfa_internal(
                &body_dfa,
                &suffix_dfa,
                max,
                true,
            )
            .expect("dominance trace");
            Arc::new(super::ZeroMinRepeatSuffixComponentTrace {
                dfa: Arc::new(built.dfa),
                prefix_len: 0,
                body_dfa: body_dfa.clone(),
                suffix_dfa: suffix_dfa.clone(),
                max,
                tail_states: built.states,
                tail_state_by_key: built.state_by_key,
            })
        };
        let full_trace = trace(40);
        let synthesized_trace = trace(10);
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
                Some(maximum - minimum)
            })
            .max()
            .unwrap_or(0);
        assert!(max_live_count_span > 1);

        let vocab = Vocab::new(vec![(0, b"aaaaaaaa".to_vec())]);
        let horizons = super::VocabularyRepeatHorizonCache::new();
        assert!(
            super::zero_min_repeat_suffix_state_map(
                &Expr::Epsilon,
                &Expr::Epsilon,
                full_trace.dfa.as_ref(),
                synthesized_trace.dfa.as_ref(),
                Some(Arc::clone(&full_trace)),
                Some(Arc::clone(&synthesized_trace)),
                vocab.max_token_byte_len(),
                Some(&vocab),
                Some(&horizons),
            )
            .is_none(),
            "the shortened repeat must fail closed when live count spread and one-token displacement exceed its upper-bound headroom",
        );
    }

    #[test]
    fn dense_binary_intersection_matches_component_conjunction_exhaustively() {
        let cases = vec![
            (
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(byte_choice(b"ab")),
                        min: 1,
                        max: Some(4),
                    },
                    Expr::U8Seq(b"b".to_vec()),
                ]),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                    min: 0,
                    max: Some(3),
                },
            ),
            (
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab "))),
                    min: 0,
                    max: Some(5),
                },
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(byte_choice(b"ab")),
                        min: 0,
                        max: Some(3),
                    },
                    Expr::U8Seq(b" ".to_vec()),
                ]),
            ),
            (
                Expr::Choice(vec![
                    Expr::U8Seq(b"abba".to_vec()),
                    Expr::U8Seq(b"baab".to_vec()),
                    Expr::U8Seq(b" ".to_vec()),
                ]),
                Expr::Repeat {
                    expr: Box::new(byte_choice(b"ab ")),
                    min: 1,
                    max: Some(4),
                },
            ),
        ];
        let inputs = enumerate_inputs(b"ab ", 6);

        for (left_expr, right_expr) in cases {
            let components = vec![
                super::compile_product_component(&left_expr),
                super::compile_product_component(&right_expr),
            ];
            let left = components[0].partition_dfa().clone();
            let right = components[1].partition_dfa().clone();
            let (class_map, class_members) =
                super::compute_product_equivalence_classes(&components);
            let class_transitions =
                super::build_product_class_transitions(&components, &class_map);
            let (product, _, trace) = super::try_build_dense_binary_intersection_product(
                &components,
                &class_members,
                &class_transitions,
                true,
                false,
            )
            .expect("dense binary intersection");
            let trace = trace.expect("captured dense trace");
            assert_eq!(trace.state_tuples.len(), product.num_states());

            for input in &inputs {
                assert_eq!(
                    dfa_accepts(&product, input),
                    dfa_accepts(&left, input) && dfa_accepts(&right, input),
                    "left={left_expr:?} right={right_expr:?} input={input:?}",
                );
            }
        }
    }

    #[test]
    fn lazy_zero_min_repeat_intersection_matches_materialized_product() {
        let repeat_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::Choice(vec![
                    Expr::U8Seq(b"a".to_vec()),
                    Expr::U8Seq(b"ab".to_vec()),
                ])),
                min: 0,
                max: Some(1_024),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let filter_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(7),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(repeat_component),
            intersect: Box::new(filter_component),
        };

        let eager = super::compile_with_plan(super::build_exclusion_compile_plan(
            std::slice::from_ref(&expression),
        ));
        let plan = super::build_exclusion_compile_plan(std::slice::from_ref(&expression));
        let (lazy, trace) = super::try_compile_lazy_zero_min_repeat_intersection(&plan, false)
            .expect("eligible repeat intersection must compile lazily");
        assert_eq!(trace.state_tuples.len(), lazy.num_states());
        assert!(matches!(
            &trace.state_lookup,
            super::ProductStateLookup::Hash(_)
        ));

        for input in enumerate_inputs(b"[]abx", 9) {
            assert_eq!(
                dfa_state_observation(&lazy, 0, &input),
                dfa_state_observation(&eager, 0, &input),
                "lazy product differed for input {input:?}",
            );
        }
    }

    #[test]
    fn lazy_repeat_selector_prefers_the_unique_giant_operand() {
        let component = |max| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 0,
                    max: Some(max),
                },
                byte_expr(b'b'),
            ])
        };

        let non_giant = component(super::LAZY_ZERO_MIN_REPEAT_SUFFIX_MIN_BOUND);
        let giant = component(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND);
        let (index, _) = super::select_lazy_zero_min_repeat_suffix_component(&[
            non_giant.clone(),
            giant.clone(),
        ])
        .expect("the unique giant lazy component must be selected");
        assert_eq!(index, 1);

        let (index, _) = super::select_lazy_zero_min_repeat_suffix_component(&[
            giant.clone(),
            non_giant,
        ])
        .expect("operand reversal must still select the unique giant component");
        assert_eq!(index, 0);

        assert!(
            super::select_lazy_zero_min_repeat_suffix_component(&[
                giant,
                component(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 1),
            ])
            .is_none(),
            "the one-lazy-component lane must not claim two giant coordinates",
        );
    }

    fn finite_dfa_with_partial_nonlive_x_cycle() -> DFA {
        // Language is exactly {"b"}. The x-edge enters a partial dead cycle
        // rather than the canonical 256-byte sink, so product construction
        // must use finalizer/future liveness rather than sink identity alone.
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, U8Set::from_bytes(b"bx"));
        dfa.set_transitions_from_sorted_entries(0, vec![(b'b', 1), (b'x', 2)]);
        dfa.set_transitions_from_sorted_entries(2, vec![(b'x', 2)]);

        let mut start_future = crate::ds::bitset::BitSet::new(1);
        start_future.set(0);
        dfa.overwrite_state_metadata(
            0,
            crate::ds::bitset::BitSet::new(1),
            start_future,
        );
        let mut accepting = crate::ds::bitset::BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(
            1,
            accepting,
            crate::ds::bitset::BitSet::new(1),
        );
        dfa.overwrite_state_metadata(
            2,
            crate::ds::bitset::BitSet::new(1),
            crate::ds::bitset::BitSet::new(1),
        );
        dfa
    }

    #[test]
    fn lazy_giant_intersection_prunes_partial_nonlive_other_cycles() {
        let giant = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(byte_expr(b'x')),
                min: 0,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            },
            byte_expr(b'b'),
        ]);
        let ordinary = Expr::Dfa(Arc::new(finite_dfa_with_partial_nonlive_x_cycle()));

        for reversed in [false, true] {
            let expression = if reversed {
                Expr::Intersect {
                    expr: Box::new(ordinary.clone()),
                    intersect: Box::new(giant.clone()),
                }
            } else {
                Expr::Intersect {
                    expr: Box::new(giant.clone()),
                    intersect: Box::new(ordinary.clone()),
                }
            };
            let eager = super::compile_with_plan(super::build_exclusion_compile_plan(
                std::slice::from_ref(&expression),
            ));
            let plan = super::build_exclusion_compile_plan(std::slice::from_ref(&expression));
            let (mut lazy, mut trace) =
                super::try_compile_lazy_zero_min_repeat_intersection(&plan, false)
                    .expect("partial dead cycle must not reject the exact lazy product");
            assert_eq!(lazy.num_states(), 2, "only start and accepted 'b' survive");
            assert!(lazy.step(0, b'x').is_none());
            assert!(matches!(
                &trace.state_lookup,
                super::ProductStateLookup::Hash(_)
            ));

            for input in [b"".as_slice(), b"b", b"x", b"xx", b"xb"] {
                let semantic_observation = |dfa: &DFA| {
                    let (_, accepting, future) = dfa_state_observation(dfa, 0, input);
                    (accepting, future)
                };
                assert_eq!(
                    semantic_observation(&lazy),
                    semantic_observation(&eager),
                    "lazy product differed for reversed={reversed} input={input:?}",
                );
            }

            // Retained structural traces can be augmented from seed tuples.
            // Seed an impossible residual containing the ordinary dead-cycle
            // state and verify augmentation cannot turn it into a giant-only
            // x-walk after the ordinary coordinate is pruned.
            let ordinary_index = if reversed { 0usize } else { 1usize };
            let mut seed = super::ProductStateTuple::new();
            seed.push((0, if ordinary_index == 0 { 2 } else { 0 }));
            seed.push((1, if ordinary_index == 1 { 2 } else { 0 }));
            let seed_state = lazy.num_states() as u32;
            super::augment_product_dfa_from_seed_tuples(
                &mut lazy,
                &mut trace,
                &[seed],
                &plan.exclusions,
                &plan.intersections,
            );
            assert!(lazy.num_states() > seed_state as usize);
            assert!(lazy.step(seed_state, b'x').is_none());

            let compact = super::try_compile_compact_zero_min_repeat_intersection_runtime(
                &plan,
                false,
            )
            .expect("compact lazy lane should also prune the partial dead cycle");
            let (compact_dfa, compact_segment) = compact.finish_runtime();
            let compact = Tokenizer::from_parts_with_compressed_transitions(
                compact_dfa,
                1,
                None,
                compact_segment.into_iter().collect(),
            );
            let eager = Tokenizer::from_parts(eager, 1, None);
            for input in [b"".as_slice(), b"b", b"x", b"xx", b"xb"] {
                let semantic_observation = |tokenizer: &Tokenizer| {
                    let (matches, futures, _) = tokenizer_observation(tokenizer, input);
                    (matches, futures)
                };
                assert_eq!(
                    semantic_observation(&compact),
                    semantic_observation(&eager),
                    "compact product differed for reversed={reversed} input={input:?}",
                );
            }
        }
    }

    #[test]
    fn compact_zero_min_repeat_runtime_matches_materialized_product() {
        let repeat_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(1_024),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let filter_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(7),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(repeat_component),
            intersect: Box::new(filter_component),
        };

        let eager = super::compile_with_plan(super::build_exclusion_compile_plan(
            std::slice::from_ref(&expression),
        ));
        let plan = super::build_exclusion_compile_plan(std::slice::from_ref(&expression));
        let compact = super::try_compile_compact_zero_min_repeat_intersection_runtime(
            &plan,
            false,
        )
        .expect("eligible deterministic repeat intersection must compile compactly");
        let (compact_dfa, compact_segment) = compact.finish_runtime();
        let compact = Tokenizer::from_parts_with_compressed_transitions(
            compact_dfa,
            1,
            None,
            compact_segment.into_iter().collect(),
        );
        let eager = Tokenizer::from_parts(eager, 1, None);

        for input in enumerate_inputs(b"[]abx", 9) {
            assert_eq!(
                tokenizer_observation(&compact, &input),
                tokenizer_observation(&eager, &input),
                "compact runtime product differed for input {input:?}",
            );
        }
    }

    #[test]
    fn compact_zero_min_repeat_runtime_does_not_allocate_from_billion_bound() {
        let repeat_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(1_000_000_000),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        // The intersecting coordinate bounds the reachable residual product to
        // a tiny set. Construction must therefore scale with those reachable
        // residuals, not with the syntactic billion-count upper bound.
        let filter_component = Expr::Seq(vec![
            Expr::U8Seq(b"[".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: Some(7),
            },
            Expr::U8Seq(b"]".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(repeat_component),
            intersect: Box::new(filter_component),
        };
        let plan = super::build_exclusion_compile_plan(std::slice::from_ref(&expression));
        let compact = super::try_compile_compact_zero_min_repeat_intersection_runtime(
            &plan,
            false,
        )
        .expect("eligible billion-bound repeat intersection must stay compact");
        assert!(compact.num_states() < 128);
    }

    #[test]
    fn exact_runtime_union_preisolates_nullable_components() {
        let expressions = vec![
            Expr::Epsilon,
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),
        ];
        let partitions = [0, 1, 2];
        let residual_classes = [None, None, None];

        let baseline_regex =
            super::build_regex_partitioned_with_adaptive_and_residual_isolation(
                &expressions,
                &partitions,
                &residual_classes,
                false,
            );
        let mut baseline = baseline_regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        assert_eq!(
            baseline.isolate_start_state_and_drain_nullable_terminals(),
            BTreeSet::from([0, 1]),
        );

        let mut preisolated = super::build_exact_partitioned_runtime_tokenizer(
            &expressions,
            None,
            &partitions,
            &residual_classes,
        );
        assert!(
            preisolated
                .isolate_start_state_and_drain_nullable_terminals()
                .is_empty(),
            "component-local isolation must leave no nullable finalizer at the union root",
        );

        for input in enumerate_inputs(b"abx", 5) {
            assert_eq!(
                tokenizer_observation(&preisolated, &input),
                tokenizer_observation(&baseline, &input),
                "component-local nullable isolation differed for input {input:?}",
            );
        }
    }

    fn dfa_state_observation(
        dfa: &super::DFA,
        mut state: u32,
        input: &[u8],
    ) -> (bool, bool, bool) {
        for &byte in input {
            let Some(next) = dfa.step(state, byte) else {
                return (true, false, false);
            };
            state = next;
        }
        (
            false,
            dfa.finalizers(state).contains(0),
            dfa.possible_future_group_ids(state).contains(0),
        )
    }

    #[test]
    fn layered_bounded_suffix_transport_matches_all_short_observations() {
        const HORIZON: usize = 4;
        let alphabet = b"[]abx";
        let inputs = enumerate_inputs(alphabet, HORIZON);
        let bodies = [
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::U8Seq(b"ab".to_vec()),
        ];

        for body in bodies {
            let make_expr = |max| {
                factor_regex_expr(Expr::Seq(vec![
                    Expr::U8Seq(b"[".to_vec()),
                    Expr::Repeat {
                        expr: Box::new(body.clone()),
                        min: 2,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"]".to_vec()),
                ]))
            };
            let full_expr = make_expr(24);
            let synthesized_expr = make_expr(16);
            let full = super::compile_product_component_materialized_dfa(&full_expr);
            let synthesized =
                super::compile_product_component_materialized_dfa(&synthesized_expr);
            let mapping = super::direct_bounded_suffix_state_map(
                &full_expr,
                &synthesized_expr,
                &full,
                &synthesized,
                HORIZON,
                alphabet,
                None,
                None,
            )
            .expect("direct layered transport");
            assert_eq!(mapping.primary().len(), full.num_states());

            for full_state in 0..full.num_states() as u32 {
                let synthesized_state = mapping.primary()[full_state as usize];
                for input in &inputs {
                    assert_eq!(
                        dfa_state_observation(&full, full_state, input),
                        dfa_state_observation(&synthesized, synthesized_state, input),
                        "body={body:?} full_state={full_state} synthesized_state={synthesized_state} input={input:?}",
                    );
                }
            }
        }
    }

    fn tokenizer_observation(
        tokenizer: &Tokenizer,
        input: &[u8],
    ) -> (Vec<(u32, usize)>, Vec<u32>, bool) {
        let execution = tokenizer.execute_from_state_all_widths(input, tokenizer.initial_state());
        let mut matches = execution
            .matches
            .iter()
            .map(|matched| (matched.id, matched.width))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        matches.dedup();

        let mut futures = execution
            .end_state
            .iter()
            .flat_map(|&state| tokenizer.possible_future_terminals_iter(state))
            .collect::<Vec<_>>();
        futures.sort_unstable();
        futures.dedup();
        (matches, futures, execution.end_state.is_empty())
    }

    fn execute_state_set_observation(
        tokenizer: &Tokenizer,
        roots: &[u32],
        input: &[u8],
    ) -> (Vec<(u32, usize)>, Vec<u32>) {
        let mut matches = Vec::new();
        let mut end_states = Vec::new();
        for &root in roots {
            let execution = tokenizer.execute_from_state_all_widths(input, root);
            matches.extend(
                execution
                    .matches
                    .into_iter()
                    .map(|matched| (matched.id, matched.width)),
            );
            end_states.extend(execution.end_state);
        }
        matches.sort_unstable();
        matches.dedup();
        end_states.sort_unstable();
        end_states.dedup();

        let mut futures = end_states
            .iter()
            .flat_map(|&state| tokenizer.possible_future_terminals_iter(state))
            .collect::<Vec<_>>();
        futures.sort_unstable();
        futures.dedup();
        (matches, futures)
    }

    fn random_small_expr(rng: &mut StdRng, depth: usize) -> Expr {
        let atom = |rng: &mut StdRng| match rng.gen_range(0..4) {
            0 => Expr::U8Seq(vec![b'a' + rng.gen_range(0..3)]),
            1 => {
                let len = rng.gen_range(1..=3);
                Expr::U8Seq(
                    (0..len)
                        .map(|_| b'a' + rng.gen_range(0..3))
                        .collect(),
                )
            }
            2 => {
                let mut bytes = Vec::new();
                for byte in b'a'..=b'c' {
                    if rng.gen_bool(0.5) {
                        bytes.push(byte);
                    }
                }
                if bytes.is_empty() {
                    bytes.push(b'a' + rng.gen_range(0..3));
                }
                Expr::U8Class(U8Set::from_bytes(&bytes))
            }
            _ => Expr::Epsilon,
        };

        if depth == 0 {
            return atom(rng);
        }
        match rng.gen_range(0..9) {
            0..=2 => atom(rng),
            3 => Expr::Choice(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            4 => Expr::Seq(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            5 => Expr::Repeat {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                min: rng.gen_range(0..=1),
                max: Some(rng.gen_range(1..=3)),
            },
            6 => Expr::Repeat {
                expr: Box::new(atom(rng)),
                min: rng.gen_range(0..=1),
                max: None,
            },
            7 => Expr::Exclude {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                exclude: Box::new(random_small_expr(rng, depth - 1)),
            },
            _ => Expr::Intersect {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                intersect: Box::new(random_small_expr(rng, depth - 1)),
            },
        }
    }

    fn random_group_free_expr(rng: &mut StdRng, depth: usize) -> Expr {
        let atom = |rng: &mut StdRng| match rng.gen_range(0..4) {
            0 => Expr::U8Seq(vec![b'a' + rng.gen_range(0..3)]),
            1 => Expr::U8Seq(
                (0..rng.gen_range(1..=3))
                    .map(|_| b'a' + rng.gen_range(0..3))
                    .collect(),
            ),
            2 => Expr::U8Class(U8Set::from_bytes(match rng.gen_range(0..3) {
                0 => b"ab",
                1 => b"bc",
                _ => b"abc",
            })),
            _ => Expr::Epsilon,
        };

        if depth == 0 {
            return atom(rng);
        }
        match rng.gen_range(0..7) {
            0..=2 => atom(rng),
            3 => Expr::Choice(vec![
                random_group_free_expr(rng, depth - 1),
                random_group_free_expr(rng, depth - 1),
            ]),
            4 => Expr::Seq(vec![
                random_group_free_expr(rng, depth - 1),
                random_group_free_expr(rng, depth - 1),
            ]),
            5 => Expr::Repeat {
                expr: Box::new(random_group_free_expr(rng, depth - 1)),
                min: rng.gen_range(0..=1),
                max: Some(rng.gen_range(1..=3)),
            },
            _ => Expr::Shared(Arc::new(random_group_free_expr(rng, depth - 1))),
        }
    }

    fn expr_contains_dfa(expr: &Expr) -> bool {
        match expr {
            Expr::Dfa(_) => true,
            Expr::Exclude { expr, exclude } => {
                expr_contains_dfa(expr) || expr_contains_dfa(exclude)
            }
            Expr::Intersect { expr, intersect } => {
                expr_contains_dfa(expr) || expr_contains_dfa(intersect)
            }
            Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_contains_dfa),
            Expr::Repeat { expr, .. } => expr_contains_dfa(expr),
            Expr::Shared(expr) => expr_contains_dfa(expr),
            Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
        }
    }

    fn tokenizer_from_partitioned_exprs(exprs: &[Expr]) -> Tokenizer {
        let partitions = (0..exprs.len() as u32).collect::<Vec<_>>();
        build_regex_partitioned_with_adaptive(exprs, &partitions, false).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.to_vec().into_boxed_slice())),
        )
    }

    #[test]
    fn partitioned_lexer_matches_monolithic_semantics_exhaustively() {
        let shared_tail = Arc::new(Expr::Choice(vec![
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ]));
        let exprs = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"ac".to_vec()),
            ]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                    min: 0,
                    max: Some(2),
                },
                Expr::Shared(Arc::clone(&shared_tail)),
            ]),
            Expr::Exclude {
                expr: Box::new(byte_choice(b"abc")),
                exclude: Box::new(byte_expr(b'b')),
            },
            Expr::Intersect {
                expr: Box::new(byte_choice(b"ab")),
                intersect: Box::new(byte_choice(b"bc")),
            },
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Shared(shared_tail),
            ]),
        ];
        let monolithic = build_regex_monolithic(&exprs).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.clone().into_boxed_slice())),
        );
        let partitionings = [
            (0..exprs.len() as u32).collect::<Vec<_>>(),
            vec![0, 0, 1, 1, 2, 2, 1],
        ];
        let inputs = enumerate_inputs(b"abc ", 5);

        for partitions in partitionings {
            let partitioned = build_regex_partitioned_with_adaptive(
                &exprs,
                &partitions,
                false,
            )
            .into_tokenizer(
                exprs.len() as u32,
                Some(Arc::from(exprs.clone().into_boxed_slice())),
            );
            for input in &inputs {
                assert_eq!(
                    tokenizer_observation(&partitioned, input),
                    tokenizer_observation(&monolithic, input),
                    "partitioned lexer differed for partitions={partitions:?} input={input:?}",
                );
            }
        }
    }

    #[test]
    fn seeded_partitioned_lexer_differential_fuzz() {
        let mut rng = StdRng::seed_from_u64(0xE051_10FA_2026_0710);
        let prefixes = enumerate_inputs(b"abc", 3);
        let suffixes = enumerate_inputs(b"abc", 3);

        for case in 0..48 {
            let expr_count = rng.gen_range(2..=6);
            let exprs = (0..expr_count)
                .map(|_| random_small_expr(&mut rng, 2))
                .collect::<Vec<_>>();
            let monolithic = build_regex_monolithic(&exprs).into_tokenizer(
                exprs.len() as u32,
                Some(Arc::from(exprs.clone().into_boxed_slice())),
            );

            for partition_case in 0..3 {
                let partitions = match partition_case {
                    0 => (0..expr_count as u32).collect::<Vec<_>>(),
                    1 => (0..expr_count)
                        .map(|_| rng.gen_range(0..3))
                        .collect::<Vec<_>>(),
                    _ => vec![0; expr_count],
                };
                let partitioned = build_regex_partitioned_with_adaptive(
                    &exprs,
                    &partitions,
                    false,
                )
                .into_tokenizer(
                    exprs.len() as u32,
                    Some(Arc::from(exprs.clone().into_boxed_slice())),
                );
                let adaptive = build_regex_partitioned_with_adaptive(
                    &exprs,
                    &partitions,
                    true,
                )
                .into_tokenizer(
                    exprs.len() as u32,
                    Some(Arc::from(exprs.clone().into_boxed_slice())),
                );

                for prefix in &prefixes {
                    assert_eq!(
                        tokenizer_observation(&partitioned, prefix),
                        tokenizer_observation(&monolithic, prefix),
                        "top-level mismatch case={case} partitions={partitions:?} prefix={prefix:?} exprs={exprs:?}",
                    );
                    assert_eq!(
                        tokenizer_observation(&adaptive, prefix),
                        tokenizer_observation(&monolithic, prefix),
                        "adaptive top-level mismatch case={case} partitions={partitions:?} prefix={prefix:?} exprs={exprs:?}",
                    );

                    let partitioned_roots = partitioned.execute_from_state_end_only(
                        prefix,
                        partitioned.initial_state(),
                    );
                    let adaptive_roots = adaptive.execute_from_state_end_only(
                        prefix,
                        adaptive.initial_state(),
                    );
                    let monolithic_roots = monolithic.execute_from_state_end_only(
                        prefix,
                        monolithic.initial_state(),
                    );
                    for suffix in &suffixes {
                        assert_eq!(
                            execute_state_set_observation(
                                &partitioned,
                                &partitioned_roots,
                                suffix,
                            ),
                            execute_state_set_observation(
                                &monolithic,
                                &monolithic_roots,
                                suffix,
                            ),
                            "residual mismatch case={case} partitions={partitions:?} prefix={prefix:?} suffix={suffix:?} exprs={exprs:?}",
                        );
                        assert_eq!(
                            execute_state_set_observation(
                                &adaptive,
                                &adaptive_roots,
                                suffix,
                            ),
                            execute_state_set_observation(
                                &monolithic,
                                &monolithic_roots,
                                suffix,
                            ),
                            "adaptive residual mismatch case={case} partitions={partitions:?} prefix={prefix:?} suffix={suffix:?} exprs={exprs:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nested_group_op_materialization_reuses_structurally_equal_dfas() {
        let nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
            exclude: Box::new(Expr::U8Seq(vec![b'b'])),
        };
        let mut cache = super::NestedGroupOpCache::default();
        let first = super::materialize_nested_group_ops(nested.clone(), &mut cache);
        let second = super::materialize_nested_group_ops(nested, &mut cache);

        assert_eq!(cache.cache_misses, 1);
        assert_eq!(cache.cache_hits, 1);
        match (first, second) {
            (Expr::Dfa(first), Expr::Dfa(second)) => assert!(Arc::ptr_eq(&first, &second)),
            _ => panic!("nested group operation was not materialized to a DFA"),
        }
    }

    #[test]
    fn repeated_subexpression_dfa_materialization_preserves_observations_exhaustively() {
        let repeated = Expr::Seq(vec![
            Expr::Choice(vec![byte_expr(b'a'), byte_expr(b'b')]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 1,
                max: Some(2),
            },
        ]);
        let exprs = vec![
            Expr::Seq(vec![byte_expr(b'a'), repeated.clone(), byte_expr(b'c')]),
            Expr::Choice(vec![repeated.clone(), Expr::U8Seq(b"cc".to_vec())]),
            Expr::Repeat {
                expr: Box::new(repeated.clone()),
                min: 0,
                max: Some(2),
            },
            Expr::Repeat {
                expr: Box::new(repeated.clone()),
                min: 1,
                max: None,
            },
            Expr::Exclude {
                expr: Box::new(Expr::Choice(vec![repeated.clone(), byte_expr(b'c')])),
                exclude: Box::new(byte_expr(b'c')),
            },
            Expr::Intersect {
                expr: Box::new(Expr::Choice(vec![repeated.clone(), Expr::U8Seq(b"ab".to_vec())])),
                intersect: Box::new(Expr::Choice(vec![repeated, byte_expr(b'b')])),
            },
        ];
        let rewritten = super::materialize_repeated_subexpression_dfas_with_limits(&exprs, 2, 2)
            .expect("the repeated subtree should be materialized");
        assert!(rewritten.iter().any(expr_contains_dfa));

        let original = tokenizer_from_partitioned_exprs(&exprs);
        let materialized = tokenizer_from_partitioned_exprs(&rewritten);
        for input in enumerate_inputs(b"abc", 6) {
            assert_eq!(
                tokenizer_observation(&materialized, &input),
                tokenizer_observation(&original, &input),
                "materialized repeated subtree changed tokenizer observation for input={input:?}",
            );
        }
    }

    #[test]
    fn repeated_subexpression_dfa_materialization_seeded_differential() {
        let mut rng = StdRng::seed_from_u64(0xC5E0_2026_0718);
        let inputs = enumerate_inputs(b"abc", 4);

        for case in 0..32 {
            let repeated = Expr::Seq(vec![
                random_group_free_expr(&mut rng, 2),
                random_group_free_expr(&mut rng, 2),
            ]);
            let exprs = vec![
                Expr::Seq(vec![byte_expr(b'a'), repeated.clone(), byte_expr(b'b')]),
                Expr::Choice(vec![repeated.clone(), random_group_free_expr(&mut rng, 1)]),
                Expr::Repeat {
                    expr: Box::new(repeated.clone()),
                    min: rng.gen_range(0..=1),
                    max: Some(rng.gen_range(1..=3)),
                },
                Expr::Shared(Arc::new(repeated)),
            ];
            let rewritten =
                super::materialize_repeated_subexpression_dfas_with_limits(&exprs, 2, 2)
                    .expect("the seeded repeated subtree should be materialized");
            let original = tokenizer_from_partitioned_exprs(&exprs);
            let materialized = tokenizer_from_partitioned_exprs(&rewritten);

            for input in &inputs {
                assert_eq!(
                    tokenizer_observation(&materialized, input),
                    tokenizer_observation(&original, input),
                    "seeded CSE mismatch case={case} input={input:?} exprs={exprs:?} rewritten={rewritten:?}",
                );
            }
        }
    }

    #[test]
    fn duplicate_product_coordinates_collapse_without_changing_observations() {
        let repeated = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 1,
                max: Some(3),
            },
        ]);
        let expressions = vec![
            repeated.clone(),
            repeated,
            Expr::U8Seq(b"ac".to_vec()),
        ];
        let exclusions = BTreeMap::new();
        let intersections = BTreeMap::new();
        let collapsed = super::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            false,
            true,
            false,
            false,
        )
        .0;
        let uncollapsed = super::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            true,
            true,
            false,
            false,
        )
        .0;
        assert_eq!(collapsed, uncollapsed);

        let (traced_collapsed, _, trace) = super::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            true,
            true,
            false,
            true,
        );
        assert_eq!(collapsed, traced_collapsed);
        let trace = trace.expect("trace-enabled collapsed product must retain its state tuples");
        assert_eq!(trace.state_tuples.len(), traced_collapsed.num_states());
        assert!(
            trace.coordinate_groups.iter().any(|groups| groups.len() > 1),
            "duplicate logical groups should fan out from one physical product coordinate"
        );
    }

    #[test]
    fn duplicate_product_coordinates_preserve_group_operations() {
        let repeated = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 1,
                max: Some(3),
            },
        ]);
        let expressions = vec![
            Expr::Exclude {
                expr: Box::new(repeated.clone()),
                exclude: Box::new(repeated.clone()),
            },
            Expr::Choice(vec![repeated, Expr::U8Seq(b"ac".to_vec())]),
        ];
        let collapsed = super::compile_with_plan_internal(
            super::build_exclusion_compile_plan(&expressions),
            false,
        )
        .0;
        let uncollapsed = super::compile_with_plan_internal(
            super::build_exclusion_compile_plan(&expressions),
            true,
        )
        .0;
        assert_eq!(collapsed, uncollapsed);
    }

    #[test]
    fn duplicate_product_components_share_the_same_dfa() {
        let expr = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
                min: 0,
                max: None,
            },
        ]);
        let shared_expr = Expr::Shared(Arc::new(expr.clone()));
        let (components, cache_hits) = super::compile_product_components(&[expr, shared_expr]);

        assert_eq!(cache_hits, 1);
        match (&components[0], &components[1]) {
            (
                super::ProductComponent::Materialized(first),
                super::ProductComponent::Materialized(second),
            ) => assert!(Arc::ptr_eq(first, second)),
            _ => panic!("expected materialized product components"),
        }
    }

    #[test]
    fn factors_contains_literal_choice_with_common_quoted_string_shell() {
        let char = Expr::U8Class(U8Set::from_bytes(br#"ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"#));
        let mk = |s: &[u8]| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(char.clone()),
                    min: 0,
                    max: None,
                },
                Expr::U8Seq(s.to_vec()),
                Expr::Repeat {
                    expr: Box::new(char.clone()),
                    min: 0,
                    max: None,
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let expr = Expr::Seq(vec![
            Expr::U8Seq(b"\"interval\": ".to_vec()),
            Expr::Choice(vec![
                mk(b"INTERVAL_TICK"),
                mk(b"INTERVAL_M1"),
                mk(b"INTERVAL_M2"),
                mk(b"INTERVAL_M3"),
                mk(b"INTERVAL_M4"),
                mk(b"INTERVAL_M5"),
                mk(b"INTERVAL_M6"),
                mk(b"INTERVAL_M10"),
                mk(b"INTERVAL_M15"),
                mk(b"INTERVAL_M20"),
                mk(b"INTERVAL_M30"),
                mk(b"INTERVAL_H1"),
                mk(b"INTERVAL_H2"),
                mk(b"INTERVAL_H4"),
                mk(b"INTERVAL_D1"),
                mk(b"INTERVAL_W1"),
                mk(b"INTERVAL_MN1"),
            ]),
        ]);
        let factored = factor_regex_expr(expr);
        let regex = build_regex(&[factored]);
        assert!(
            regex.num_states() < 500,
            "factored regex should not construct a huge terminal DFA; states={}",
            regex.num_states(),
        );
        let accept = |bytes: &[u8]| {
            let mut state = 0;
            for &b in bytes {
                let Some(next) = regex.step(state, b) else {
                    return false;
                };
                state = next;
            }
            !regex.dfa.finalizers(state).is_empty()
        };
        assert!(accept(br#""interval": "INTERVAL_M1""#));
        assert!(accept(br#""interval": "XXXINTERVAL_M1YYY""#));
        assert!(accept(br#""interval": "INTERVAL_TICK""#));
        assert!(!accept(br#""interval": "NOPE""#));
    }

    #[test]
    fn factors_repeated_choice_edges_shared_by_only_some_arms() {
        let char = Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"));
        let star = Expr::Repeat {
            expr: Box::new(char),
            min: 0,
            max: None,
        };
        let original = Expr::Choice(vec![
            Expr::Seq(vec![Expr::U8Seq(b"list".to_vec()), star.clone()]),
            Expr::Seq(vec![
                star.clone(),
                Expr::U8Seq(b"date".to_vec()),
                star.clone(),
            ]),
            Expr::Seq(vec![
                star.clone(),
                Expr::U8Seq(b"time".to_vec()),
                star.clone(),
            ]),
            Expr::Seq(vec![star.clone(), Expr::U8Seq(b"number".to_vec())]),
        ]);
        let factored = factor_regex_expr(original.clone());

        for input in [
            b"list".as_slice(),
            b"listanything".as_slice(),
            b"xxdateyy".as_slice(),
            b"time".as_slice(),
            b"prefixnumbersuffix".as_slice(),
            b"prefixnumber".as_slice(),
            b"nothing".as_slice(),
        ] {
            assert_eq!(
                terminal_matches(original.clone(), input),
                terminal_matches(factored.clone(), input),
                "subset choice-edge factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn factors_repeated_exclusion_rhs_across_choice_arms_exactly() {
        let literal_choice = |a: &[u8], b: &[u8]| {
            Expr::Choice(vec![Expr::U8Seq(a.to_vec()), Expr::U8Seq(b.to_vec())])
        };
        let original = Expr::Choice(vec![
            Expr::Exclude {
                expr: Box::new(literal_choice(b"bad", b"cat")),
                exclude: Box::new(Expr::U8Seq(b"bad".to_vec())),
            },
            Expr::Exclude {
                expr: Box::new(literal_choice(b"bad", b"dog")),
                exclude: Box::new(Expr::U8Seq(b"bad".to_vec())),
            },
            Expr::Exclude {
                expr: Box::new(literal_choice(b"fox", b"yak")),
                exclude: Box::new(Expr::U8Seq(b"yak".to_vec())),
            },
            Expr::U8Seq(b"eel".to_vec()),
        ]);
        let factored = factor_regex_expr(original.clone());
        assert!(
            super::group_op_node_count(&factored) < super::group_op_node_count(&original),
            "repeated exclusion RHS should collapse sibling set-difference nodes",
        );

        for input in [
            b"".as_slice(),
            b"bad".as_slice(),
            b"cat".as_slice(),
            b"dog".as_slice(),
            b"fox".as_slice(),
            b"yak".as_slice(),
            b"eel".as_slice(),
            b"other".as_slice(),
        ] {
            assert_eq!(
                terminal_matches(original.clone(), input),
                terminal_matches(factored.clone(), input),
                "repeated exclusion-RHS factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn shared_factoring_cache_preserves_factored_arc_identity() {
        let shared = Arc::new(Expr::Choice(vec![
            Expr::Seq(vec![Expr::U8Seq(b"pre".to_vec()), Expr::U8Seq(b"a".to_vec())]),
            Expr::Seq(vec![Expr::U8Seq(b"pre".to_vec()), Expr::U8Seq(b"b".to_vec())]),
        ]));
        let factored_shared = Arc::new(factor_regex_expr((*shared).clone()));
        let mut cache = rustc_hash::FxHashMap::default();
        cache.insert(Arc::as_ptr(&shared) as usize, Arc::clone(&factored_shared));
        let expression = Expr::Choice(vec![
            Expr::Shared(Arc::clone(&shared)),
            Expr::Seq(vec![Expr::U8Seq(b"x".to_vec()), Expr::Shared(shared)]),
        ]);

        let factored = super::factor_regex_expr_with_shared_cache(expression, &cache);
        let Expr::Choice(arms) = factored else {
            panic!("factoring should retain the outer choice");
        };
        let Expr::Shared(first) = &arms[0] else {
            panic!("cached shared root should remain shared");
        };
        let Expr::Seq(second_parts) = &arms[1] else {
            panic!("second arm should remain a sequence");
        };
        let Expr::Shared(second) = &second_parts[1] else {
            panic!("nested cached root should remain shared");
        };
        assert!(Arc::ptr_eq(first, &factored_shared));
        assert!(Arc::ptr_eq(second, &factored_shared));
        assert!(Arc::ptr_eq(first, second));
    }

    #[test]
    fn factoring_choice_spines_does_not_rewalk_cached_children() {
        fn contains(expr: &Expr, wanted: &Arc<Expr>) -> bool {
            match expr {
                Expr::Shared(inner) => Arc::ptr_eq(inner, wanted) || contains(inner, wanted),
                Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(|part| contains(part, wanted)),
                Expr::Repeat { expr, .. } => contains(expr, wanted),
                Expr::Exclude { expr, exclude } => contains(expr, wanted) || contains(exclude, wanted),
                Expr::Intersect { expr, intersect } => contains(expr, wanted) || contains(intersect, wanted),
                _ => false,
            }
        }
        let shared = Arc::new(Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))), min: 1, max: None,
        });
        let cached = Arc::new(factor_regex_expr((*shared).clone()));
        let cache = rustc_hash::FxHashMap::from_iter([(Arc::as_ptr(&shared) as usize, Arc::clone(&cached))]);
        let literal = |s: &str| Expr::U8Seq(s.as_bytes().to_vec());
        for suffix in [false, true] {
            for subset in [false, true] {
                let mut options = ["a", "b"].into_iter().map(|different| {
                    if suffix {
                        Expr::Seq(vec![literal(different), Expr::Shared(Arc::clone(&shared)), literal("x")])
                    } else {
                        Expr::Seq(vec![literal("x"), Expr::Shared(Arc::clone(&shared)), literal(different)])
                    }
                }).collect::<Vec<_>>();
                if subset { options.push(literal("z")); }
                let original = Expr::Choice(options);
                let factored = super::factor_regex_expr_with_shared_cache(original.clone(), &cache);
                assert!(contains(&factored, &cached), "choice rewrite must retain the already factored child");
                for input in ["", "z", "x", "xaa", "xab", "xaba", "xbab", "aax", "abx", "bax", "bbbx", "xc", "cax"] {
                    assert_eq!(terminal_matches(original.clone(), input.as_bytes()),
                        terminal_matches(factored.clone(), input.as_bytes()),
                        "suffix={suffix} subset={subset} input={input}");
                }
            }
        }
    }

    #[test]
    fn factors_aligned_zero_min_unit_repeat_intersection_exactly() {
        let word = U8Set::from_bytes(
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_",
        );
        let letters =
            U8Set::from_bytes(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(word)),
                min: 0,
                max: Some(1_000_000_000),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(letters)),
                min: 0,
                max: Some(700_000_000),
            }),
        };

        assert_eq!(
            factor_regex_expr(expression),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(letters)),
                min: 0,
                max: Some(700_000_000),
            },
        );
    }

    #[test]
    fn aligned_unit_repeat_intersection_factoring_preserves_small_language() {
        let left = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 0,
            max: Some(3),
        };
        let right = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
            min: 0,
            max: Some(2),
        };
        let original = Expr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        };
        let factored = factor_regex_expr(original.clone());

        for input in enumerate_inputs(b"abc", 4) {
            assert_eq!(
                terminal_matches(original.clone(), &input),
                terminal_matches(factored.clone(), &input),
                "factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn factors_aligned_nonzero_unit_repeat_intersection_exactly() {
        let left = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 2,
            max: Some(5),
        };
        let right = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
            min: 3,
            max: Some(4),
        };
        let original = Expr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        };
        let factored = factor_regex_expr(original.clone());
        assert_eq!(
            factored,
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::single(b'b'))),
                min: 3,
                max: Some(4),
            },
        );
        for input in enumerate_inputs(b"abc", 6) {
            assert_eq!(
                terminal_matches(original.clone(), &input),
                terminal_matches(factored.clone(), &input),
                "nonzero aligned-unit factoring changed acceptance for {input:?}",
            );
        }

        let disjoint = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: Some(2),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                min: 3,
                max: Some(4),
            }),
        };
        assert_eq!(factor_regex_expr(disjoint), Expr::U8Class(U8Set::empty()));
    }

    #[test]
    fn aligned_unit_empty_byte_intersection_normalizes_fully() {
        let expression_for = |min| Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
        };
        assert_eq!(factor_regex_expr(expression_for(0)), Expr::Epsilon);
        assert_eq!(
            factor_regex_expr(expression_for(1)),
            Expr::U8Class(U8Set::empty()),
        );
    }

    #[test]
    fn aligned_repeat_intersection_rewrite_fails_closed_for_variable_width_body() {
        let variable_width = Expr::Choice(vec![byte_expr(b'a'), Expr::U8Seq(b"aa".to_vec())]);
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(variable_width),
                min: 0,
                max: Some(100),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(100),
            }),
        };

        assert!(matches!(factor_regex_expr(expression), Expr::Intersect { .. }));
    }

    #[test]
    fn sparse_virtual_repeat_finite_product_does_not_flatten_repeat_state_ids() {
        const LONG: usize = 65_536;
        let mut base_dfa = DFA::new(LONG + 1);
        base_dfa.ensure_group_capacity(1);
        base_dfa.set_transitions_from_sorted_entries(0, vec![(b'a', LONG as u32)]);
        let mut finalizers = crate::ds::bitset::BitSet::new(1);
        finalizers.set(0);
        base_dfa.overwrite_state_metadata(
            LONG as u32,
            finalizers,
            crate::ds::bitset::BitSet::new(1),
        );
        let components = vec![
            super::ProductComponent::VirtualBoundedRepeat {
                base_dfa: Arc::new(base_dfa),
                min: 1,
                max: 1_000_000_000,
            },
            super::ProductComponent::VirtualFixedSequence {
                byte_sets: Arc::from(
                    vec![U8Set::single(b'a'); LONG].into_boxed_slice(),
                ),
                suffix_live: Arc::from(vec![true; LONG + 1].into_boxed_slice()),
            },
        ];
        let (class_map, class_members) = super::compute_product_equivalence_classes(&components);
        let class_transitions = super::build_product_class_transitions(&components, &class_map);
        let (dfa, direct, trace) =
            super::try_build_sparse_virtual_repeat_finite_intersection_product(
                &components,
                &class_members,
                &class_transitions,
                false,
                false,
            )
            .expect("virtual repeat with finite coordinate should use sparse exact product");

        // The synthetic base DFA has 65,537 states while the finite side
        // drives 65,536 completed copies. The old flattened coordinate would
        // need 65,536 * 65,537 = 4,295,032,832, beyond u32::MAX. The structural
        // coordinate keeps only the actually reachable O(LONG) pair states.
        assert!(direct);
        assert!(trace.is_none());
        assert_eq!(dfa.num_states(), LONG + 1);
        let mut state = 0u32;
        for _ in 0..LONG {
            state = dfa.step(state, b'a').expect("accepted finite-prefix step");
        }
        assert!(dfa.finalizers(state).contains(0));
    }

    #[test]
    fn lazy_giant_intersection_materializes_only_budgeted_non_giant_repeat_side() {
        // This root repeat is above the generic product's direct-repeat
        // threshold, so it would normally be represented as a
        // VirtualBoundedRepeat. Its exact DFA is still tiny, however, and the
        // lazy intersection lane may materialize it under the shared budget.
        let small_other = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
            min: 0,
            max: Some(100),
        };
        assert!(super::compile_lazy_intersection_materialized_other_component(
            &small_other,
            true,
        )
        .is_some());

        // Stay one below the giant-repeat threshold so the right side remains
        // nominally "non-giant", but make its exact bounded-repeat DFA exceed
        // the 100k materialization budget. The specialized fast path must
        // decline instead of turning this optimization into a large eager
        // allocation; the general residual runtime remains the correctness
        // fallback at the dynamic-constraint layer.
        let oversized_other = Expr::Repeat {
            expr: Box::new(Expr::U8Seq({
                let mut bytes = vec![b'a'; 31];
                bytes.push(b'b');
                bytes
            })),
            min: 0,
            max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND - 1),
        };
        assert!(super::compile_lazy_intersection_materialized_other_component(
            &oversized_other,
            true,
        )
        .is_none());

        let cyclic_other = Expr::Repeat {
            expr: Box::new(byte_expr(b'a')),
            min: 0,
            max: None,
        };
        assert!(super::compile_lazy_intersection_materialized_other_component(
            &cyclic_other,
            true,
        )
        .is_none());
    }

    #[test]
    fn bounded_repeat_from_base_rejects_layer_and_state_id_overflow() {
        let base = super::compile_expr_to_dfa(&byte_expr(b'a'));
        assert!(
            super::build_bounded_repeat_dfa_from_base(&base, 2, 1).is_none(),
            "an invalid repeat interval must fail closed",
        );
        assert!(
            super::build_bounded_repeat_dfa_from_base(&base, 0, usize::MAX).is_none(),
            "max + 1 must fail closed instead of overflowing",
        );
        assert!(
            super::build_bounded_repeat_dfa_from_base(&base, 0, u32::MAX as usize).is_none(),
            "layered state IDs must fit in u32",
        );
    }

    #[test]
    fn bounded_repeat_from_base_marks_only_productive_residuals_future_live() {
        let mut base = DFA::new(3);
        base.ensure_group_capacity(1);
        base.set_transitions_from_sorted_entries(0, vec![(b'a', 1), (b'x', 2)]);
        base.set_transitions_from_sorted_entries(2, vec![(b'x', 2)]);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        base.overwrite_state_metadata(1, finalizers, BitSet::new(1));

        let repeated = super::build_bounded_repeat_dfa_from_base(&base, 0, 2)
            .expect("small repeat must materialize");
        let dead = repeated.step(0, b'x').expect("dead residual remains represented");
        assert!(
            !repeated.possible_future_group_ids(dead).contains(0),
            "a reachable base residual with no path to a finalizer must not advertise a repeat future",
        );
        let live = repeated.step(0, b'a').expect("accepted copy transition");
        assert!(repeated.finalizers(live).contains(0));
        assert!(repeated.possible_future_group_ids(live).contains(0));
    }

    #[test]
    fn bounded_repeat_suffix_ignores_unproductive_delimiter_transition() {
        let body = byte_expr(b'a');
        let mut supplied_base = DFA::new(3);
        supplied_base.ensure_group_capacity(1);
        supplied_base.set_transitions_from_sorted_entries(0, vec![(b'a', 1), (b'x', 2)]);
        supplied_base.set_transitions_from_sorted_entries(2, vec![(b'x', 2)]);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        supplied_base.overwrite_state_metadata(1, finalizers, BitSet::new(1));

        let mut cache = super::RepeatBaseDfaCache::default();
        cache.insert(body.clone(), Arc::new(supplied_base));
        let parts = vec![
            Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 0,
                max: Some(2),
            },
            byte_expr(b'x'),
        ];
        let (with_dead_edge, _) = super::build_bounded_repeat_with_suffix_dfa_with_cache(
            &parts,
            Some(&cache),
        )
        .expect("an unproductive body edge on the suffix byte is not a boundary ambiguity");

        let clean_parts = vec![
            Expr::Repeat {
                expr: Box::new(body),
                min: 0,
                max: Some(2),
            },
            byte_expr(b'x'),
        ];
        let clean = super::build_bounded_repeat_with_suffix_dfa(&clean_parts)
            .expect("clean prefix-free repeat+suffix must use the direct builder")
            .0;
        for input in enumerate_inputs(b"ax", 4) {
            assert_eq!(
                dfa_accepts(&with_dead_edge, &input),
                dfa_accepts(&clean, &input),
                "dropping an unproductive body transition changed the repeat+suffix language for {input:?}",
            );
        }
        let suffix_target = with_dead_edge
            .step(0, b'x')
            .expect("the delimiter must enter the suffix, not the dead body residual");
        assert!(with_dead_edge.finalizers(suffix_target).contains(0));

        let overflowing = vec![
            Expr::Repeat {
                expr: Box::new(byte_expr(b'a')),
                min: 0,
                max: Some(usize::MAX),
            },
            byte_expr(b'x'),
        ];
        assert!(
            super::build_bounded_repeat_with_suffix_dfa(&overflowing).is_none(),
            "repeat+suffix layer arithmetic must fail closed on max + 1 overflow",
        );
    }

    #[test]
    fn factors_same_body_giant_repeats_with_unique_literal_suffix_boundary() {
        let body = factor_regex_expr(Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]));
        let component = |max, suffix: &[u8]| {
            Expr::Seq(vec![
                Expr::U8Seq(b"[".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body.clone()),
                    min: 0,
                    max: Some(max),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };

        let shared_suffix = Expr::Intersect {
            expr: Box::new(component(
                super::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 7,
                b"]abca",
            )),
            intersect: Box::new(component(
                super::VIRTUAL_BINARY_REPEAT_MIN_BOUND,
                b"]abca",
            )),
        };
        assert_eq!(
            factor_regex_expr(shared_suffix),
            Expr::Seq(vec![
                Expr::U8Seq(b"[".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body.clone()),
                    min: 0,
                    max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
                },
                Expr::U8Seq(b"]abca".to_vec()),
            ]),
        );

        let disjoint_suffixes = Expr::Intersect {
            expr: Box::new(component(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND, b"]")),
            intersect: Box::new(component(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 3, b"}")),
        };
        assert_eq!(
            factor_regex_expr(disjoint_suffixes),
            Expr::U8Class(U8Set::empty()),
        );

        // A small materialized analogue is a cheap semantic oracle for the
        // variable-width prefix-free body: no byte string can satisfy the two
        // different uniquely-delimited suffixes.
        let small_left = component(3, b"]");
        let small_right = component(4, b"}");
        let small_left = compile_product_component_dfa(&small_left);
        let small_right = compile_product_component_dfa(&small_right);
        for input in enumerate_inputs(b"[abc]}", 6) {
            assert!(
                !(dfa_accepts(&small_left, &input) && dfa_accepts(&small_right, &input)),
                "unexpected common word for uniquely delimited suffixes: {input:?}",
            );
        }
    }

    #[test]
    fn delimited_repeat_suffix_factoring_handles_only_safe_nonzero_min_results() {
        let giant = super::VIRTUAL_BINARY_REPEAT_MIN_BOUND;
        let component = |min, max, suffix: &[u8]| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min,
                    max: Some(max),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };

        let disjoint_counts = Expr::Intersect {
            expr: Box::new(component(5, giant, b"b")),
            intersect: Box::new(component(1, 3, b"b")),
        };
        assert_eq!(
            factor_regex_expr(disjoint_counts),
            Expr::U8Class(U8Set::empty()),
            "unique body/suffix boundaries make disjoint count intervals exactly empty",
        );

        let different_suffixes = Expr::Intersect {
            expr: Box::new(component(2, giant, b"b")),
            intersect: Box::new(component(3, giant + 1, b"c")),
        };
        assert_eq!(
            factor_regex_expr(different_suffixes),
            Expr::U8Class(U8Set::empty()),
            "different uniquely-delimited literals are disjoint even with positive minima",
        );

        let bounded_merge = Expr::Intersect {
            expr: Box::new(component(2, giant, b"b")),
            intersect: Box::new(component(3, 7, b"b")),
        };
        assert_eq!(
            factor_regex_expr(bounded_merge),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 3,
                    max: Some(7),
                },
                Expr::U8Seq(b"b".to_vec()),
            ]),
            "a positive-minimum merged interval is safe when its upper bound is ordinary",
        );

        let unsupported_giant_result = Expr::Intersect {
            expr: Box::new(component(2, giant + 1, b"b")),
            intersect: Box::new(component(3, giant, b"b")),
        };
        assert!(
            matches!(
                factor_regex_expr(unsupported_giant_result),
                Expr::Intersect { .. }
            ),
            "do not rewrite to a positive-minimum giant repeat+suffix before that shape has an exact runtime lane",
        );

        let long_body = Expr::U8Seq(vec![b'a'; 32]);
        let large_ordinary = |min, max| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(long_body.clone()),
                    min,
                    max: Some(max),
                },
                Expr::U8Seq(b"b".to_vec()),
            ])
        };
        let over_budget_ordinary_result = Expr::Intersect {
            expr: Box::new(large_ordinary(0, giant)),
            intersect: Box::new(large_ordinary(0, giant - 1)),
        };
        assert!(
            matches!(
                factor_regex_expr(over_budget_ordinary_result),
                Expr::Intersect { .. }
            ),
            "a sub-threshold merged repeat must still remain unfactored when its exact layered DFA exceeds the explicit budget",
        );
    }

    #[test]
    fn delimited_repeat_suffix_factoring_rejects_count_shift_and_non_code_bodies() {
        let giant = super::VIRTUAL_BINARY_REPEAT_MIN_BOUND;
        let repeat = |body, suffix: &[u8]| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(body),
                    min: 0,
                    max: Some(giant),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };

        // The left suffix begins with 'a', which can start another body word.
        // Indeed a*ab and a*b overlap by shifting one repetition across the
        // suffix boundary, so the algebraic rewrite must not fire.
        let shifted = Expr::Intersect {
            expr: Box::new(repeat(byte_expr(b'a'), b"ab")),
            intersect: Box::new(repeat(byte_expr(b'a'), b"b")),
        };
        assert!(matches!(factor_regex_expr(shifted), Expr::Intersect { .. }));

        // Structural body equality is not enough: {a, aa} is not prefix-free,
        // so a byte string can carry different factor counts before the same
        // delimiter. Keep the original intersection when the code proof fails.
        let ambiguous = Expr::Choice(vec![byte_expr(b'a'), Expr::U8Seq(b"aa".to_vec())]);
        let non_code = Expr::Intersect {
            expr: Box::new(repeat(ambiguous.clone(), b"b")),
            intersect: Box::new(repeat(ambiguous, b"b")),
        };
        assert!(matches!(factor_regex_expr(non_code), Expr::Intersect { .. }));

        // Even identical arbitrary prefix languages cannot in general be
        // cancelled from an intersection. This rewrite only accepts one fixed
        // literal prefix byte string.
        let nonliteral_prefix = Expr::Choice(vec![Expr::Epsilon, byte_expr(b'p')]);
        let prefixed = |suffix: &[u8]| {
            Expr::Seq(vec![
                nonliteral_prefix.clone(),
                Expr::Repeat {
                    expr: Box::new(byte_expr(b'a')),
                    min: 0,
                    max: Some(giant),
                },
                Expr::U8Seq(suffix.to_vec()),
            ])
        };
        let nonliteral = Expr::Intersect {
            expr: Box::new(prefixed(b"b")),
            intersect: Box::new(prefixed(b"c")),
        };
        assert!(matches!(factor_regex_expr(nonliteral), Expr::Intersect { .. }));
    }

    #[test]
    fn factors_same_prefix_free_body_nonzero_repeat_intersection_exactly() {
        let body = factor_regex_expr(Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]));
        let original = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 2,
                max: Some(3),
            }),
        };
        let factored = factor_regex_expr(original.clone());
        assert_eq!(
            factored,
            Expr::Repeat {
                expr: Box::new(body),
                min: 2,
                max: Some(3),
            },
        );

        // Keep the giant side only barely above the virtualization threshold
        // so the unfactored expression is still a cheap materialized oracle.
        let original_dfa = build_regex(std::slice::from_ref(&original)).dfa;
        let factored_dfa = build_regex(std::slice::from_ref(&factored)).dfa;
        for input in enumerate_inputs(b"abc", 6) {
            assert_eq!(
                dfa_accepts(&original_dfa, &input),
                dfa_accepts(&factored_dfa, &input),
                "same-body factoring changed acceptance for {input:?}",
            );
        }
    }

    #[test]
    fn same_prefix_free_body_disjoint_repeat_intervals_factor_to_empty() {
        let body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bc".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 4,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body),
                min: 1,
                max: Some(3),
            }),
        };
        assert_eq!(factor_regex_expr(expression), Expr::U8Class(U8Set::empty()));
    }

    #[test]
    fn same_body_nonzero_repeat_factoring_requires_prefix_free_body() {
        let body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"aa".to_vec()),
        ]);
        let expression = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body),
                min: 2,
                max: Some(3),
            }),
        };
        assert!(matches!(factor_regex_expr(expression), Expr::Intersect { .. }));

        // Disjoint factor-count intervals are not enough without unique
        // decipherability: a^(4097) has both a 4097-copy decomposition and a
        // 4096-copy decomposition when the body language is {"a", "aa"}.
        let body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"aa".to_vec()),
        ]);
        let disjoint = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(body),
                min: super::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 1,
                max: Some(super::VIRTUAL_BINARY_REPEAT_MIN_BOUND + 1),
            }),
        };
        assert!(matches!(factor_regex_expr(disjoint), Expr::Intersect { .. }));
    }

    #[test]
    fn nested_exclude_in_exclusion_branch_compiles() {
        let nested_residual = Expr::Exclude {
            expr: Box::new(byte_choice(b"ab")),
            exclude: Box::new(byte_expr(b'a')),
        };
        assert!(!terminal_matches(nested_residual.clone(), b"a"));
        assert!(terminal_matches(nested_residual.clone(), b"b"));
        assert!(!terminal_matches(nested_residual.clone(), b"c"));

        let expr = Expr::Exclude {
            expr: Box::new(byte_choice(b"bc")),
            exclude: Box::new(nested_residual),
        };

        assert!(!terminal_matches(expr.clone(), b"a"));
        assert!(!terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"c"));
    }

    #[test]
    fn nested_intersect_in_exclusion_branch_compiles() {
        let nested_intersection = Expr::Intersect {
            expr: Box::new(byte_choice(b"ab")),
            intersect: Box::new(byte_expr(b'b')),
        };
        assert!(!terminal_matches(nested_intersection.clone(), b"a"));
        assert!(terminal_matches(nested_intersection.clone(), b"b"));
        assert!(!terminal_matches(nested_intersection.clone(), b"c"));

        let expr = Expr::Exclude {
            expr: Box::new(byte_choice(b"bc")),
            exclude: Box::new(nested_intersection),
        };

        assert!(!terminal_matches(expr.clone(), b"a"));
        assert!(!terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"c"));
    }

    #[test]
    fn product_trace_residuals_keep_empty_complex_terminal_as_dead_coordinate() {
        let live = byte_choice(b"ab");
        let dead = Expr::Exclude {
            expr: Box::new(byte_choice(b"ab")),
            exclude: Box::new(byte_choice(b"ab")),
        };
        let exprs = vec![live, dead];
        let partitions = vec![0u32, 0u32];
        let retained = Arc::from(exprs.clone().into_boxed_slice());
        let tokenizer = super::build_partitioned_tokenizer_with_product_trace_terminal_residuals(
            &exprs,
            None,
            &partitions,
            None,
            retained,
            Some(false),
            true,
        )
        .expect("empty complex terminal should not force product-trace fallback");

        for input in [b"a".as_slice(), b"b".as_slice()] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(exec.matches.iter().any(|matched| matched.id == 0));
            assert!(!exec.matches.iter().any(|matched| matched.id == 1));
        }

        let coordinates = tokenizer
            .terminal_residual_coordinates()
            .expect("product-trace tokenizer should retain terminal residual coordinates");
        let dead_dfa = coordinates
            .terminal_dfa(1)
            .expect("dead terminal should retain a residual DFA");
        assert_eq!(dead_dfa.num_states(), 1);
        assert!(dead_dfa.finalizers(0).is_empty());
        assert!(dead_dfa.possible_future_group_ids(0).is_empty());
        for state in 0..tokenizer.num_states() {
            assert!(
                coordinates
                    .row(state)
                    .expect("coordinate row must exist")
                    .iter()
                    .all(|&(terminal, _)| terminal != 1),
                "dead terminal unexpectedly appears in residual row {state}",
            );
        }
    }

    #[test]
    fn product_trace_retains_finite_exclusion_continuation_certificate() {
        let open = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"\": ".to_vec()),
        ]);
        let excluded = Expr::U8Seq(b"\"ab\": ".to_vec());
        let exprs = vec![
            Expr::Exclude {
                expr: Box::new(open),
                exclude: Box::new(excluded),
            },
            Expr::U8Seq(b"x".to_vec()),
        ];
        let partitions = vec![0u32, 0u32];
        let tokenizer = super::build_partitioned_tokenizer_with_product_trace_terminal_residuals(
            &exprs,
            None,
            &partitions,
            None,
            Arc::from(exprs.clone().into_boxed_slice()),
            Some(false),
            false,
        )
        .expect("finite exclusion should retain product-trace coordinates");

        let certificates = (0..tokenizer.num_states())
            .filter_map(|state| tokenizer.terminal_exclusion_continuation(state, 0))
            .collect::<Vec<_>>();
        assert!(!certificates.is_empty(), "expected exclusion continuation metadata");
        assert!(
            certificates
                .iter()
                .any(|certificate| certificate.right_max_remaining == Some(6)),
            "the initial finite exclusion should have an exact six-byte remaining bound: {certificates:?}",
        );
        assert!(
            certificates.iter().any(|certificate| certificate.right_state.is_none()),
            "a non-matching open-name prefix should kill the finite exclusion while LEFT stays live: {certificates:?}",
        );
        assert!(certificates.iter().all(|certificate| {
            certificate
                .right_max_remaining
                .is_none_or(|remaining| remaining <= 6)
        }));
        assert!(tokenizer.terminal_exclusion_left_dfa(0).is_some());
    }

    #[test]
    fn standalone_exact_repeat_matches_only_at_full_length() {
        let expr = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 0),
                "exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 0 && matched.width == 16),
            "exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn product_exact_repeat_matches_only_at_full_length() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let exact_repeat = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };

        let regex = build_regex(&[space.clone(), exact_repeat.clone()]);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 2,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![space, exact_repeat].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "product exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 16),
            "product exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn product_vbr_exact_repeat_matches_only_at_full_length() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let exact_repeat = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 32,
            max: Some(32),
        };

        let regex = build_regex(&[space.clone(), exact_repeat.clone()]);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 2,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![space, exact_repeat].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 31] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "product VBR exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 32];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 32),
            "product VBR exact repeat did not match at len 32: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn glrm_chunk16_terminal_family_keeps_exact_repeat_nonfinal_until_16() {
        let space = Expr::U8Class(U8Set::single(b' '));
        let quote = Expr::U8Seq(vec![b'"']);
        let exact_16 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 16,
            max: Some(16),
        };
        let upto_16 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(16),
        };
        let upto_close_16 = Expr::Seq(vec![upto_16.clone(), quote.clone()]);
        let upto_3 = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(3),
        };
        let upto_close_3 = Expr::Seq(vec![upto_3.clone(), quote.clone()]);

        let exprs = vec![
            space.clone(),
            exact_16.clone(),
            upto_16,
            upto_close_16,
            upto_3,
            upto_close_3,
            quote,
        ];
        let regex = build_regex(&exprs);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: exprs.len() as u32,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(exprs.into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [1usize, 2, 15] {
            let input = vec![b' '; len];
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                !exec.matches.iter().any(|matched| matched.id == 1),
                "GLRM family exact repeat matched too early at len {len}: {:?}",
                exec.matches,
            );
        }

        let input = vec![b' '; 16];
        let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
        assert!(
            exec.matches.iter().any(|matched| matched.id == 1 && matched.width == 16),
            "GLRM family exact repeat did not match at len 16: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn multi_dfa_literal_choice_direct_matches_generic_language() {
        let arm_a = Arc::new(super::compile_expr_to_dfa(&Expr::Choice(vec![
            Expr::U8Seq(b"alpha".to_vec()),
            Expr::U8Seq(b"alpine".to_vec()),
            Expr::U8Seq(b"amber".to_vec()),
        ])));
        let arm_b = Arc::new(super::compile_expr_to_dfa(&Expr::Choice(vec![
            Expr::U8Seq(b"beta".to_vec()),
            Expr::U8Seq(b"better".to_vec()),
            Expr::U8Seq(b"binary".to_vec()),
        ])));
        let arm_c = Arc::new(super::compile_expr_to_dfa(&Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"xyz"))),
            min: 2,
            max: Some(5),
        }));
        let expression = Expr::Choice(vec![
            Expr::Dfa(arm_a),
            Expr::U8Seq(b"anchor".to_vec()),
            Expr::U8Seq(b"alphabet".to_vec()),
            Expr::Dfa(arm_b),
            Expr::U8Seq(b"zebra".to_vec()),
            Expr::Dfa(arm_c),
        ]);
        let direct = super::build_dfas_with_literal_choice(&expression)
            .expect("multi-DFA plus literal choice should compile directly");
        let generic = super::compile_expr_to_dfa(&expression);
        assert_dfa_observation_equivalent(&direct, &generic);
    }

    #[test]
    fn product_vbr_with_literal_prefix_and_no_suffix_uses_direct_path() {
        let expression = Expr::Seq(vec![
            Expr::U8Seq(b"pre".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 2,
                max: Some(5),
            },
        ]);

        let direct = super::compile_product_component_dfa_direct(&expression)
            .expect("literal prefix followed by bounded repeat should compile directly")
            .0;
        let generic = super::compile_product_component_dfa(&expression);
        assert_dfa_observation_equivalent(&direct, &generic);
        assert_eq!(direct.num_states(), 3 + (5 + 1) * 2);
    }

    #[test]
    fn repeated_bounded_repeat_bodies_share_one_local_base_dfa() {
        let body = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"ab")),
            Expr::U8Class(U8Set::from_bytes(b"cd")),
        ]);
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(body.clone()),
                min: 1,
                max: Some(4),
            },
            Expr::Seq(vec![
                Expr::U8Seq(b"pre".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body.clone()),
                    min: 2,
                    max: Some(5),
                },
            ]),
        ];
        let expression_refs = expressions.iter().collect::<Vec<_>>();
        let cache = super::build_repeat_base_dfa_cache(&expression_refs, false);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&body));

        for expression in &expressions {
            let cached = super::compile_product_component_materialized_dfa_with_options_and_cache(
                expression,
                false,
                Some(&cache),
            );
            let uncached =
                super::compile_product_component_materialized_dfa_with_options(expression, false);
            assert_dfa_observation_equivalent(&cached, &uncached);
        }
    }

    #[test]
    fn product_vbr_with_literal_prefix_uses_direct_bounded_repeat_tail() {
        let quote = Expr::U8Seq(vec![b'"']);
        let spaces = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 0,
            max: Some(32),
        };
        let expr = Expr::Seq(vec![quote.clone(), spaces, quote]);

        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat did not use direct product component path");
        };
        assert!(
            dfa.num_states() <= 80,
            "direct prefixed bounded repeat DFA unexpectedly large: {} states",
            dfa.num_states(),
        );

        let tokenizer = Tokenizer {
            dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for len in [0usize, 1, 31, 32] {
            let mut input = Vec::with_capacity(len + 2);
            input.push(b'"');
            input.extend(std::iter::repeat(b' ').take(len));
            input.push(b'"');
            let exec = tokenizer.execute_from_state(&input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed bounded repeat did not match length {len}: {:?}",
                exec.matches,
            );
        }
    }

    #[test]
    fn product_vbr_with_literal_prefix_and_regex_suffix_matches() {
        let quote = Expr::U8Seq(vec![b'"']);
        let word = Expr::U8Class(U8Set::single(b'a'));
        let space = Expr::U8Class(U8Set::single(b' '));
        let word_run = Expr::Repeat {
            expr: Box::new(word.clone()),
            min: 1,
            max: None,
        };
        let space_run = Expr::Repeat {
            expr: Box::new(space),
            min: 1,
            max: None,
        };
        let pair = Expr::Seq(vec![word_run.clone(), space_run]);
        let repeated_pairs = Expr::Repeat {
            expr: Box::new(pair),
            min: 0,
            max: Some(49),
        };
        let expr = Expr::Seq(vec![quote, repeated_pairs, word_run]);

        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 400,
            "direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );

        let tokenizer = Tokenizer {
            dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for input in [b"\"a".as_slice(), b"\"aa", b"\"a a", b"\"aa  aaa"] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed bounded repeat with suffix did not match {:?}: {:?}",
                std::str::from_utf8(input).unwrap(),
                exec.matches,
            );
        }

        let exec = tokenizer.execute_from_state(b"\"a ", tokenizer.initial_state());
        assert!(
            !exec
                .matches
                .iter()
                .any(|matched| matched.id == 0 && matched.width == 3),
            "prefixed bounded repeat with suffix matched trailing space: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn prefixed_bounded_repeat_with_regex_suffix_uses_direct_path_without_repeat_cutoff() {
        let quote = Expr::U8Seq(vec![b'"']);
        let word = Expr::U8Class(U8Set::single(b'a'));
        let space = Expr::U8Class(U8Set::single(b' '));
        let word_run = Expr::Repeat {
            expr: Box::new(word.clone()),
            min: 1,
            max: None,
        };
        let space_run = Expr::Repeat {
            expr: Box::new(space),
            min: 1,
            max: None,
        };
        let pair = Expr::Seq(vec![word_run.clone(), space_run]);
        let repeated_pairs = Expr::Repeat {
            expr: Box::new(pair),
            min: 0,
            max: Some(29),
        };
        let expr = Expr::Seq(vec![quote, repeated_pairs, word_run]);

        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 300,
            "direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );

        let repeated_pairs = Expr::Repeat {
            expr: Box::new(Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
                    min: 1,
                    max: None,
                },
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
                    min: 1,
                    max: None,
                },
            ])),
            min: 0,
            max: Some(2),
        };
        let expr = Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            repeated_pairs,
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
                min: 1,
                max: None,
            },
        ]);
        let Some((dfa, _)) = super::compile_product_component_dfa_direct(&expr) else {
            panic!("small prefixed bounded repeat with regex suffix did not use direct path");
        };
        assert!(
            dfa.num_states() <= 40,
            "small direct prefixed bounded repeat with regex suffix unexpectedly large: {} states",
            dfa.num_states(),
        );
    }

    fn prefixed_optional_word_list_expr(max_pairs: usize) -> Expr {
        let nonspace_plus = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b'a'))),
            min: 1,
            max: None,
        };
        let space_plus = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 1,
            max: None,
        };
        let body = Expr::Seq(vec![nonspace_plus.clone(), space_plus]);
        let repeated = Expr::Repeat {
            expr: Box::new(body),
            min: 0,
            max: Some(max_pairs),
        };

        Expr::Seq(vec![
            Expr::U8Seq(vec![b'"']),
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::Seq(vec![repeated, nonspace_plus]),
            ]),
        ])
    }

    #[test]
    fn prefixed_optional_choice_uses_direct_component_path_for_bounded_repeat_suffix() {
        let expr = prefixed_optional_word_list_expr(199);

        let Some((dfa, _)) = compile_product_component_dfa_direct(&expr) else {
            panic!("prefixed optional wrapper did not use direct product component path");
        };

        assert!(
            dfa.num_states() < 10_000,
            "prefixed optional direct-path DFA unexpectedly large: {} states",
            dfa.num_states(),
        );
        assert!(dfa.finalizers(1).contains(0));
    }

    #[test]
    fn prefixed_optional_word_list_semantics() {
        let expr = prefixed_optional_word_list_expr(2);
        let regex = build_regex(std::slice::from_ref(&expr));
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 1,
            packed_runtime_transitions: None,
            packed_runtime_transition_segments: Arc::from([]),
            compressed_transition_segments: Arc::from([]),
            packed_runtime_metadata: None,
            packed_runtime_metadata_segments: Arc::from([]),
            packed_compressed_transition_segments: Arc::from([]),
            virtual_unit_repeat: None,
            virtual_repeat_intersections: Vec::new(),
            virtual_residuals: Vec::new(),
            exprs: Some(Arc::from(vec![expr].into_boxed_slice())),
            terminal_residual_coordinates: None,
            singleton_epsilon_closures: std::sync::OnceLock::new(),
            matched_terminals_cache: std::sync::OnceLock::new(),
            initial_byte_frontiers: std::sync::OnceLock::new(),
            all_self_loop_bytes_cache: std::sync::OnceLock::new(),
            transition_count_cache: std::sync::OnceLock::new(),
            forced_minimized_state_count_cache: std::sync::OnceLock::new(),
            scalar_deterministic_dispatch_cache: std::sync::OnceLock::new(),
            sorted_dispatch_roots_cache: std::sync::OnceLock::new(),
            state_first_bytes_cache: std::sync::OnceLock::new(),
        };

        for input in [b"\"".as_slice(), b"\"a", b"\"a a", b"\"a  a"] {
            let exec = tokenizer.execute_from_state(input, tokenizer.initial_state());
            assert!(
                exec.matches
                    .iter()
                    .any(|matched| matched.id == 0 && matched.width == input.len()),
                "prefixed optional word-list did not match {:?}: {:?}",
                std::str::from_utf8(input).unwrap(),
                exec.matches,
            );
        }

        let exec = tokenizer.execute_from_state(b"\" a", tokenizer.initial_state());
        assert!(
            !exec
                .matches
                .iter()
                .any(|matched| matched.id == 0 && matched.width == 3),
            "prefixed optional word-list matched leading space unexpectedly: {:?}",
            exec.matches,
        );
    }

    #[test]
    fn prefixed_optional_word_list_with_literal_suffix_uses_direct_exact_path() {
        let base = prefixed_optional_word_list_expr(2);
        let Expr::Seq(mut parts) = base else { unreachable!() };
        parts.push(Expr::U8Seq(vec![b'"']));
        let expr = Expr::Seq(parts);

        let Some((direct, _)) = compile_product_component_dfa_direct(&expr) else {
            panic!("optional-middle bounded repeat did not use direct component path");
        };
        let generic = super::compile_single_expr_dfa(&expr);
        let samples: &[&[u8]] = &[
            br#""""#,
            br#""a""#,
            br#""a a""#,
            br#""a  a""#,
            br#"" a""#,
            br#""a a a""#,
            br#""a a a a""#,
        ];
        for input in samples {
            assert_eq!(
                dfa_accepts(&direct, input),
                dfa_accepts(&generic, input),
                "direct optional-middle path changed language for {input:?}",
            );
        }
        assert!(dfa_accepts(&direct, br#""""#));
        assert!(dfa_accepts(&direct, br#""a""#));
        assert!(!dfa_accepts(&direct, br#"" a""#));
    }

    #[test]
    fn finite_literal_sequence_choice_trie_matches_generic_compilation() {
        let expr = Expr::Seq(vec![
            Expr::U8Seq(b"<".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())]),
                Expr::U8Seq(b"b".to_vec()),
            ]),
            Expr::Choice(vec![Expr::Epsilon, Expr::U8Seq(b">".to_vec())]),
        ]);

        let direct = super::build_finite_literal_language_dfa(&expr)
            .expect("finite concatenated literal language should compile as a trie");
        let generic = super::compile_single_expr_dfa(&expr);

        for input in enumerate_inputs(b"<>abx", 4) {
            assert_eq!(
                dfa_state_observation(&direct, 0, &input),
                dfa_state_observation(&generic, 0, &input),
                "finite-literal trie changed DFA semantics for input {input:?}",
            );
        }
    }

    #[test]
    fn direct_tokenizer_states_for_o35155_regex_groups() {
        let expr_1 = parse_regex(r"(\w+\.)+\d+", false);
        let expr_2 = parse_regex(r"\w+_(\w_)?\d+", false);
        let expr_3 = parse_regex(r"(\w|-){12}", false);
        let expr_5 = parse_regex(r"\d{7,9}", false);

        let regex_1235 = build_regex(&[
            expr_1.clone(),
            expr_2.clone(),
            expr_3.clone(),
            expr_5.clone(),
        ]);
        let regex_125 = build_regex(&[expr_1, expr_2, expr_5]);

        eprintln!(
            "o35155 direct tokenizer states for regex groups 1,2,3,5: states={} transitions={}",
            regex_1235.num_states(),
            regex_1235.num_transitions()
        );
        eprintln!(
            "o35155 direct tokenizer states for regex groups 1,2,5: states={} transitions={}",
            regex_125.num_states(),
            regex_125.num_transitions()
        );
    }

    #[test]
    fn bounded_repeat_regex_suffix_must_fork_at_ambiguous_boundary() {
        // ("a"+)? "a" matches "aa":
        //
        //   optional body "a"+ consumes the first "a"
        //   suffix "a" consumes the second "a"
        //
        // The regex-suffix fast path used to greedily continue the body on the
        // second "a" and drop the valid suffix path.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(vec![b'a'])),
                    min: 1,
                    max: None,
                }),
                min: 0,
                max: Some(1),
            },
            Expr::U8Seq(vec![b'a']),
        ]);
        assert!(terminal_matches(expr, b"aa"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_nullable_class_suffix_finalizes_after_body() {
        // "a" [b]? matches both "a" and "ab".
        //
        // The regex-suffix fast path used to miss the body/suffix boundary at
        // end-of-input when the suffix was nullable.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 1,
                max: Some(1),
            },
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::U8Class(U8Set::single(b'b')),
            ]),
        ]);
        assert!(terminal_matches(expr.clone(), b"a"));
        assert!(terminal_matches(expr, b"ab"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_nullable_suffix_after_optional_body() {
        // ("a")? [b]? matches "a", "b", and "ab".
        //
        // Do not assert the empty string here: zero-length terminals are not a
        // useful lexer regression target and may be intentionally unsupported by
        // terminal_matches / terminal DFA metadata. This test is about preserving
        // non-empty matches when both the repeated body and regex suffix are
        // nullable.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: Some(1),
            },
            Expr::Choice(vec![
                Expr::Epsilon,
                Expr::U8Class(U8Set::single(b'b')),
            ]),
        ]);
        assert!(terminal_matches(expr.clone(), b"a"));
        assert!(terminal_matches(expr.clone(), b"b"));
        assert!(terminal_matches(expr, b"ab"));
    }
    #[test]
    fn bounded_repeat_regex_suffix_zero_max_must_not_consume_body() {
        // "a"{0,0} [b] is just [b]. It must not match "ab".
        //
        // The regex-suffix fast path used to start with a live body state even when
        // max == 0.
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: Some(0),
            },
            Expr::U8Class(U8Set::single(b'b')),
        ]);
        assert!(terminal_matches(expr.clone(), b"b"));
        assert!(!terminal_matches(expr, b"ab"));
    }

    #[test]
    fn build_regex_defaults_to_one_monolithic_dfa() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        assert!(!tokenizer.has_epsilon_transitions());
    }

    #[test]
    fn separate_terminal_partitions_preserve_multiple_live_end_states() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
            .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        assert!(tokenizer.has_epsilon_transitions());
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert_eq!(result.end_state.len(), 2, "both terminal components must remain live");
        assert_eq!(
            result.matches.iter().map(|matched| matched.id).collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1]),
        );

        let continued = tokenizer.execute_from_state(b"a", result.end_state[1]);
        assert!(continued.matches.iter().any(|matched| matched.id == 1));
    }

    #[test]
    fn explicit_partition_ids_control_joint_determinization() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"z".to_vec()),
        ];
        let regex = super::build_regex_partitioned_with_adaptive(
            &expressions,
            &[7, 7, 9],
            false,
        );
        assert_eq!(
            regex.dfa.states()[0].epsilon_transitions.len(),
            2,
            "two declared partitions must produce two epsilon branches",
        );
        let tokenizer = regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert!(result.matches.iter().any(|matched| matched.id == 0));
        assert!(!result.end_state.is_empty());
    }

    #[test]
    fn protected_residual_components_survive_adaptive_determinization() {
        let expressions = vec![
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"same".to_vec()),
            Expr::U8Seq(b"ordinary-a".to_vec()),
            Expr::U8Seq(b"ordinary-b".to_vec()),
        ];
        let isolation = [Some(100), Some(101), None, None];
        let components = super::compile_partition_components(
            &expressions,
            None,
            &[0, 1, 2, 3],
            Some(&isolation),
        );

        let output = super::adaptively_determinize_components_with_limits(
            components,
            32_768,
            1_000,
            1_000,
            Some(1),
        );

        for protected_terminal in [0usize, 1] {
            let component = output
                .iter()
                .find(|component| component.terminal_ids.contains(&protected_terminal))
                .expect("protected terminal must remain present");
            assert!(component.protected_residual);
            assert_eq!(component.terminal_ids, vec![protected_terminal]);
        }
    }

    #[test]
    fn protected_identical_terminals_keep_independent_live_states() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(4),
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(4),
            },
        ];
        let regex = super::build_regex_partitioned_with_adaptive_and_residual_isolation(
            &expressions,
            &[0, 1],
            &[Some(200), Some(201)],
            true,
        );
        assert_eq!(
            regex.dfa.states()[0].epsilon_transitions.len(),
            2,
            "protected coordinates must not collapse into one adaptive product",
        );
        let tokenizer = regex.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let result = tokenizer.execute_from_state(b"a", tokenizer.initial_state());
        assert_eq!(result.end_state.len(), 2);
        assert_eq!(
            result
                .matches
                .iter()
                .map(|matched| matched.id)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1]),
        );
    }

    #[test]
    fn shared_nested_group_ops_are_initialized_before_parallel_partition_compile() {
        let shared_nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"))),
            exclude: Box::new(Expr::U8Seq(b"x".to_vec())),
        };
        let expressions = (0..64u8)
            .map(|suffix| {
                Expr::Seq(vec![
                    shared_nested.clone(),
                    Expr::U8Seq(vec![b'0' + suffix % 10]),
                ])
            })
            .collect::<Vec<_>>();
        let partitions = (0..expressions.len() as u32).collect::<Vec<_>>();
        let mut grouped = std::collections::BTreeMap::<u32, Vec<usize>>::new();
        for (terminal, &partition) in partitions.iter().enumerate() {
            grouped.entry(partition).or_default().push(terminal);
        }

        let shared = super::shared_duplicate_nested_group_op_cache(&expressions, &grouped)
            .expect("the repeated nested exclusion should use the shared cache");
        super::prewarm_shared_duplicate_nested_group_ops(&shared);
        assert!(shared.all_entries_initialized());

        let components =
            super::compile_partition_components(&expressions, None, &partitions, None);
        assert_eq!(components.len(), expressions.len());
    }

    #[test]
    #[should_panic(
        expected = "shared nested group-op cache must be prewarmed before parallel compilation"
    )]
    fn shared_nested_group_ops_cannot_initialize_from_partition_workers() {
        let shared_nested = Expr::Exclude {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz"))),
            exclude: Box::new(Expr::U8Seq(b"x".to_vec())),
        };
        let expressions = (0..64u8)
            .map(|suffix| {
                Expr::Seq(vec![
                    shared_nested.clone(),
                    Expr::U8Seq(vec![b'0' + suffix % 10]),
                ])
            })
            .collect::<Vec<_>>();
        let partitions = (0..expressions.len() as u32).collect::<Vec<_>>();
        let mut grouped = std::collections::BTreeMap::<u32, Vec<usize>>::new();
        for (terminal, &partition) in partitions.iter().enumerate() {
            grouped.entry(partition).or_default().push(terminal);
        }
        let shared = super::shared_duplicate_nested_group_op_cache(&expressions, &grouped)
            .expect("the repeated nested exclusion should use the shared cache");

        let _ = super::compile_terminal_ids_with_shared_duplicate_cache(
            &expressions,
            None,
            &[0],
            Some(&shared),
        );
    }

    #[test]
    fn partitioned_union_transports_exact_possible_futures_without_recompute() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Epsilon,
        ];
        let regex = build_regex_partitioned_with_adaptive(
            &expressions,
            &[7, 7, 9, 11],
            false,
        );
        let exact = regex.dfa;
        let mut recomputed = exact.clone();
        recomputed.recompute_possible_futures();

        assert_eq!(
            exact, recomputed,
            "transported component futures and epsilon-root union must match the generic fixpoint",
        );
    }

    #[test]
    fn bounded_product_trial_stops_before_cross_pattern_blowup() {
        let expressions = vec![
            parse_regex(r"\w+_(\w_)?\d+", false),
            parse_regex(r"(\w|-){12}", false),
        ];
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1], None);
        assert!(
            try_product_union_components(&components, 32, usize::MAX, None).is_none(),
            "the bounded trial unexpectedly completed within 32 product states",
        );
    }

    #[test]
    fn adaptive_transition_growth_limit_is_inclusive() {
        assert!(super::adaptive_transition_growth_is_acceptable(100, 600, 600));
        assert!(!super::adaptive_transition_growth_is_acceptable(100, 601, 600));
    }

    #[test]
    fn adaptive_transition_growth_rejection_keeps_partition_components() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1, 2], None);
        let original_terminal_ids = components
            .iter()
            .map(|component| component.terminal_ids.clone())
            .collect::<Vec<_>>();

        let retained = super::adaptively_determinize_components_with_limits(
            components,
            32_768,
            100,
            1,
            None,
        );

        assert_eq!(retained.len(), original_terminal_ids.len());
        assert_eq!(
            retained
                .iter()
                .map(|component| component.terminal_ids.clone())
                .collect::<Vec<_>>(),
            original_terminal_ids,
        );
    }

    #[test]
    fn bounded_adaptive_product_accounts_for_copied_component_states() {
        let expressions = vec![
            Expr::U8Seq(b"abcd".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(8),
            },
        ];
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1], None);
        let untouched_states = 1usize
            + components
                .iter()
                .map(|component| component.dfa.num_states())
                .sum::<usize>();

        assert!(
            try_product_union_components(
                &components,
                untouched_states,
                usize::MAX,
                Some(1),
            )
            .is_none(),
            "a depth-one prefix plus copied suffix DFAs must not fit inside the untouched-union state count",
        );

        let expanded = try_product_union_components(
            &components,
            untouched_states.saturating_mul(2),
            usize::MAX,
            Some(1),
        )
        .expect("the bounded product should fit once explicit growth is allowed");
        assert!(
            expanded.num_states() > untouched_states,
            "this fixture should expose the bounded prefix + copied-component overhead",
        );
    }

    #[test]
    fn adaptive_bounded_depth_uses_additive_prefix_budget() {
        let expressions = vec![
            Expr::U8Seq(b"abc".to_vec()),
            Expr::U8Seq(b"abd".to_vec()),
        ];
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1], None);
        let untouched_states = 1usize
            + components
                .iter()
                .map(|component| component.dfa.num_states())
                .sum::<usize>();

        let bounded = super::adaptively_determinize_components_with_limits(
            super::compile_partition_components(&expressions, None, &[0, 1], None),
            32_768,
            100,
            600,
            Some(1),
        );
        assert_eq!(
            bounded.len(),
            1,
            "bounded depth should use its explicit additive prefix-state budget rather than the full-product percentage budget",
        );
        assert!(
            bounded[0].dfa.num_states()
                <= untouched_states + super::adaptive_lexer_bounded_overhead_states(),
        );

        let full = super::adaptively_determinize_components_with_limits(
            components,
            32_768,
            100,
            600,
            None,
        );
        let [combined] = full.as_slice() else {
            panic!("full mode should accept the compressing exact product");
        };
        assert!(
            combined.dfa.num_states() < untouched_states,
            "the exact product should be smaller than the untouched epsilon union for this shared-prefix fixture",
        );
        assert!(
            !combined.dfa.has_epsilon_transitions(),
            "the accepted exact product should remain deterministic",
        );
    }

    #[test]
    fn adaptive_policy_does_not_change_per_partition_compilation() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"c".to_vec()),
        ];
        let partitions = [7, 7, 9, 11];
        let components =
            super::compile_partition_components(&expressions, None, &partitions, None);
        let expected_terminal_ids = [vec![0, 1], vec![2], vec![3]];

        assert_eq!(components.len(), expected_terminal_ids.len());
        for (component, expected_ids) in components.iter().zip(expected_terminal_ids) {
            assert_eq!(component.terminal_ids, expected_ids);
            let expected = super::compile_terminal_ids(&expressions, None, &expected_ids);
            assert_eq!(component.dfa, expected);
        }
    }

    #[test]
    fn adaptive_prefix_depth_preserves_partitioned_semantics() {
        let expressions = vec![
            Expr::U8Seq(b"abcd".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: Some(8),
            },
        ];
        let baseline = build_regex_partitioned_with_adaptive(&expressions, &[0, 1], false)
            .into_tokenizer(
                expressions.len() as u32,
                Some(Arc::from(expressions.clone().into_boxed_slice())),
            );
        let components =
            super::compile_partition_components(&expressions, None, &[0, 1], None);
        let prefix_dfa = try_product_union_components(&components, 32_768, usize::MAX, Some(1))
            .expect("depth-one adaptive product should fit");
        assert!(prefix_dfa.states()[0].epsilon_transitions.is_empty());
        assert!(
            prefix_dfa.states()[0]
                .transitions
                .iter()
                .any(|(_, &target)| !prefix_dfa.states()[target as usize].epsilon_transitions.is_empty()),
            "the depth-one product should resume exact component states at its frontier",
        );
        let adaptive = super::Regex { dfa: prefix_dfa }.into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b"abcdx", 8) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&baseline, &input),
                "depth-one adaptive prefix differed for input {input:?}",
            );
        }
    }

    #[test]
    fn adaptive_final_nfa_determinization_preserves_partitioned_semantics() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
        ];
        let singleton = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        let adaptive = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2],
            true,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b"ab", 4) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&singleton, &input),
                "adaptive partitioning differed for input {input:?}",
            );
        }
    }

    #[test]
    fn direct_trivial_product_components_match_generic_compilation() {
        let expressions = [
            Expr::U8Seq(b"literal".to_vec()),
            Expr::U8Seq(Vec::new()),
            Expr::U8Class(U8Set::from_bytes(b"abc")),
            Expr::Epsilon,
        ];

        for expr in expressions {
            let direct = compile_product_component_dfa_direct(&expr)
                .expect("trivial expression must have a direct DFA")
                .0;
            let generic = compile_product_component_dfa(&expr);
            assert_eq!(
                direct.group_id_to_u8set(0),
                generic.group_id_to_u8set(0),
                "group byte set differed for {expr:?}",
            );

            let mut inputs = vec![Vec::new()];
            match &expr {
                Expr::U8Seq(bytes) => {
                    for prefix_len in 0..=bytes.len() {
                        let prefix = bytes[..prefix_len].to_vec();
                        inputs.push(prefix.clone());
                        for byte in 0..=u8::MAX {
                            let mut deviated = prefix.clone();
                            deviated.push(byte);
                            inputs.push(deviated);
                        }
                    }
                }
                Expr::U8Class(_) => {
                    inputs.extend((0..=u8::MAX).map(|byte| vec![byte]));
                    inputs.extend((0..=u8::MAX).map(|byte| vec![byte, byte]));
                }
                Expr::Epsilon => inputs.push(vec![0]),
                _ => unreachable!(),
            }

            let observe = |dfa: DFA, input: &[u8]| {
                let tokenizer = Tokenizer::from_parts(dfa, 1, None);
                let states = tokenizer.run(input);
                let mut matched = BTreeSet::new();
                let mut future = BTreeSet::new();
                for state in states {
                    matched.extend(tokenizer.matched_terminals_iter(state));
                    future.extend(tokenizer.possible_future_terminals_iter(state));
                }
                (matched, future)
            };
            for input in inputs {
                assert_eq!(
                    observe(direct.clone(), &input),
                    observe(generic.clone(), &input),
                    "direct DFA behavior differed for {expr:?} on {input:?}",
                );
            }
        }
    }

    #[test]
    fn direct_fixed_sequence_component_matches_generic_compilation() {
        let expression = Expr::Seq(vec![
            Expr::U8Class(U8Set::from_bytes(b"+-")),
            Expr::Shared(Arc::new(Expr::U8Seq(b"ab".to_vec()))),
            Expr::Epsilon,
            Expr::U8Class(U8Set::from_bytes(b"xy")),
        ]);
        let direct = super::compile_product_component_dfa_direct(&expression)
            .expect("fixed sequence should compile directly")
            .0;
        let generic = super::compile_product_component_dfa(&expression);
        let direct = super::Regex { dfa: direct }.into_tokenizer(1, None);
        let generic = super::Regex { dfa: generic }.into_tokenizer(1, None);

        for input in enumerate_inputs(b"+-abxyz", 4) {
            assert_eq!(
                tokenizer_observation(&direct, &input),
                tokenizer_observation(&generic, &input),
                "direct fixed sequence differed for input {input:?}",
            );
        }
    }

    #[test]
    fn virtual_fixed_sequence_components_preserve_exact_product_layout() {
        let expressions = vec![
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_bytes(b"+-")),
                Expr::U8Seq(b"alpha".to_vec()),
            ]),
            Expr::U8Seq(b" beta".to_vec()),
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_bytes(b"xy")),
                Expr::U8Class(U8Set::from_bytes(b"12")),
            ]),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: Some(3),
            },
        ];
        let exclusions = BTreeMap::new();
        let intersections = BTreeMap::new();

        let baseline = super::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            false,
            false,
            false,
            false,
        )
        .0;
        let virtualized = super::build_product_dfa(
            &expressions,
            None,
            expressions.len(),
            &exclusions,
            &intersections,
            false,
            true,
            false,
            false,
        )
        .0;

        assert_eq!(virtualized, baseline);
    }

    #[test]
    fn zero_min_repeat_suffix_dominance_matches_generic_ambiguous_boundaries() {
        let expressions = [
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("a+", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("a+", false),
            ]),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("a+b+", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("a+", false),
            ]),
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(parse_regex("(?:ab|a)", false)),
                    min: 0,
                    max: Some(4),
                },
                parse_regex("b+", false),
            ]),
        ];

        for expression in expressions {
            let direct_dfa = compile_product_component_dfa_direct(&expression)
                .expect("zero-minimum bounded repeat with non-nullable suffix must compile directly")
                .0;
            let direct = super::Regex { dfa: direct_dfa }.into_tokenizer(
                1,
                Some(Arc::from(vec![expression.clone()].into_boxed_slice())),
            );

            let mut nfa = super::build_regex_nfa(std::slice::from_ref(&expression));
            nfa.condense_epsilon_sccs();
            let generic = super::Regex {
                dfa: nfa.to_minimized_dfa(),
            }
            .into_tokenizer(
                1,
                Some(Arc::from(vec![expression.clone()].into_boxed_slice())),
            );

            for input in enumerate_inputs(b"abx", 8) {
                assert_eq!(
                    tokenizer_observation(&direct, &input),
                    tokenizer_observation(&generic, &input),
                    "dominance quotient differed from generic determinization for expression {expression:?}, input {input:?}",
                );
            }
        }
    }

    #[test]
    fn generic_product_component_fallback_preserves_strict_future_metadata() {
        let expr = Expr::Repeat {
            expr: Box::new(Expr::Epsilon),
            min: 0,
            max: None,
        };
        let product = super::compile_product_component(&expr);
        let product = product.partition_dfa();
        let generic = compile_product_component_dfa(&expr);

        assert_eq!(product.finalizers(0), generic.finalizers(0));
        assert_eq!(
            product.possible_future_group_ids(0),
            generic.possible_future_group_ids(0),
        );
        assert!(product.finalizers(0).contains(0));
        assert!(
            !product.possible_future_group_ids(0).contains(0),
            "an epsilon-only terminal is final now but is not reachable after another byte",
        );
    }

    #[test]
    fn deferred_dense_binary_intersection_matches_eager_product() {
        let bounded_string = |max| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                    // Keep this regression on the ordinary deferred-product
                    // implementation; zero-minimum repeat intersections have
                    // their own exact lazy-product regression above.
                    min: 1,
                    max: Some(max),
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let expression = Expr::Intersect {
            expr: Box::new(bounded_string(31)),
            intersect: Box::new(bounded_string(17)),
        };

        let eager = super::compile_with_plan(super::build_exclusion_compile_plan(
            std::slice::from_ref(&expression),
        ));
        let (mut deferred, trace) = match super::try_compile_with_plan_deferred_dense(
            super::build_exclusion_compile_plan(std::slice::from_ref(&expression)),
        ) {
            Ok(prepared) => prepared,
            Err(_) => panic!("binary intersection should admit deferred dense construction"),
        };
        let trace = trace.expect("deferred construction must retain its state tuples");
        deferred
            .attach_dense_runtime_trace(trace)
            .expect("deferred construction must accept its runtime trace");
        let finished = deferred.finish();

        assert_eq!(finished, eager);
    }

    #[test]
    fn retained_dense_runtime_matches_eager_product() {
        let bounded_string = |body: Expr, max| {
            Expr::Seq(vec![
                Expr::U8Seq(b"\"".to_vec()),
                Expr::Repeat {
                    expr: Box::new(body),
                    min: 1,
                    max: Some(max),
                },
                Expr::U8Seq(b"\"".to_vec()),
            ])
        };
        let cases = [
            Expr::Intersect {
                expr: Box::new(bounded_string(
                    Expr::U8Class(U8Set::from_bytes(b"ab")),
                    31,
                )),
                intersect: Box::new(bounded_string(
                    Expr::U8Class(U8Set::from_bytes(b"ab")),
                    17,
                )),
            },
            Expr::Intersect {
                expr: Box::new(bounded_string(
                    Expr::Choice(vec![
                        Expr::U8Seq(b"a".to_vec()),
                        Expr::U8Seq(b"ab".to_vec()),
                    ]),
                    24,
                )),
                intersect: Box::new(bounded_string(
                    Expr::U8Class(U8Set::from_bytes(b"ab")),
                    19,
                )),
            },
        ];

        for expression in cases {
            let eager = Tokenizer::from_parts(
                super::compile_with_plan(super::build_exclusion_compile_plan(
                    std::slice::from_ref(&expression),
                )),
                1,
                None,
            );
            let (retained, trace) = match
                super::try_compile_with_plan_deferred_dense_min_pair_cells(
                    super::build_exclusion_compile_plan(std::slice::from_ref(&expression)),
                    0,
                    true,
                ) {
                    Ok(prepared) => prepared,
                    Err(_) => panic!("binary intersection must admit retained dense construction"),
                };
            assert!(trace.is_none(), "retained rows must not require a state trace");
            let (dfa, segment) = retained.finish_runtime();
            let runtime = Tokenizer::from_parts_with_compressed_transitions(
                dfa,
                1,
                None,
                segment.into_iter().collect(),
            );

            for input in enumerate_inputs(b"\"abx", 8) {
                assert_eq!(
                    tokenizer_observation(&runtime, &input),
                    tokenizer_observation(&eager, &input),
                    "retained compressed product differed for expression {expression:?}, input {input:?}",
                );
            }
        }
    }

    #[test]
    fn adaptive_final_representation_matches_monolithic_for_ignore_and_repeated_terminals() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b" ".to_vec())),
                min: 1,
                max: None,
            },
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ];
        let monolithic = build_regex_monolithic(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.clone().into_boxed_slice())),
        );
        let adaptive = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1, 2, 3],
            true,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        for input in enumerate_inputs(b" abc", 6) {
            assert_eq!(
                tokenizer_observation(&adaptive, &input),
                tokenizer_observation(&monolithic, &input),
                "adaptive tokenizer differed for input {input:?}",
            );
        }
    }

    #[test]
    fn epsilon_partitioned_tokenizer_round_trips_through_serde() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = build_regex_partitioned_with_adaptive(
            &expressions,
            &[0, 1],
            false,
        )
        .into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let encoded = bincode::serialize(&tokenizer).unwrap();
        let decoded: Tokenizer = bincode::deserialize(&encoded).unwrap();
        assert!(decoded.has_epsilon_transitions());
        assert_eq!(
            tokenizer.execute_from_state(b"a", tokenizer.initial_state()),
            decoded.execute_from_state(b"a", decoded.initial_state()),
        );
    }

    #[test]
    fn fingerprint_product_state_lookup_verifies_canonical_tuple_and_overflow() {
        let mut first = super::ProductStateTuple::new();
        first.push((0, 7));
        first.push((3, 11));
        let mut overflow_tuple = super::ProductStateTuple::new();
        overflow_tuple.push((1, 5));
        overflow_tuple.push((4, 13));

        let mut state_by_fingerprint = rustc_hash::FxHashMap::default();
        state_by_fingerprint.insert(super::product_state_fingerprint(&first), 0);
        let mut overflow = rustc_hash::FxHashMap::default();
        overflow.insert(overflow_tuple.clone(), 1);
        let mut lookup = super::ProductStateLookup::Fingerprint {
            state_by_fingerprint,
            canonical_tuples: vec![first.clone(), overflow_tuple.clone()],
            overflow,
        };

        assert_eq!(lookup.get(&first), Some(0));
        assert_eq!(lookup.get(&overflow_tuple), Some(1));

        let mut inserted = super::ProductStateTuple::new();
        inserted.push((2, 17));
        lookup.insert(inserted.clone(), 2);
        assert_eq!(lookup.get(&inserted), Some(2));
    }

}
