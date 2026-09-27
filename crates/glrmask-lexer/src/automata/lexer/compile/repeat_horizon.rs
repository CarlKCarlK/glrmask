//! Vocabulary-relative repeat horizons and their language-keyed cache.
//! A horizon bounds how many body completions one token can cross; it is not a grammar bound.

use crate::Vocab;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;

use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;
use super::factor::expr_contains_group_op;
use super::nfa::compile_expr_to_dfa;
use rayon::prelude::*;

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
