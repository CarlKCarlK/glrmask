//! General lazy regex residuals for dynamic tokenization.
//!
//! The key invariant is that bounded repetition remains a symbolic
//! `(body, min, max)` node. Byte derivatives decrement those integers only
//! when a body copy is actually consumed; construction never allocates in
//! proportion to `max`.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use rustc_hash::{FxHashMap, FxHashSet};
use rayon::prelude::*;

use super::ast::Expr;
use super::compile::{compile_terminal_expr_dfa, expression_contains_large_bounded_repeat, VocabularyRepeatHorizonCache};
use super::dfa::DFA;
use super::runtime_repeat_product::{VirtualRuntimeStateOwners, VirtualStateAllocator};
use super::tokenizer::{CompressedTransitionEntries, CompressedTransitionSegment};
use crate::ds::bitset::BitSet;
use crate::ds::char_transitions::CharTransitions;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

pub(crate) type ResidualId = u32;

#[derive(Debug, Clone)]
struct ResidualDfa(Arc<DFA>);

impl PartialEq for ResidualDfa {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ResidualDfa {}

impl Hash for ResidualDfa {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

impl Deref for ResidualDfa {
    type Target = DFA;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ResidualNode {
    Empty,
    Epsilon,
    SigmaStar,
    Literal { bytes: Arc<[u8]>, offset: u32 },
    Class(U8Set),
    Dfa { dfa: ResidualDfa, states: Box<[u32]> },
    Seq(Box<[ResidualId]>),
    Choice(Box<[ResidualId]>),
    Intersect(ResidualId, ResidualId),
    Exclude(ResidualId, ResidualId),
    Repeat {
        body: ResidualId,
        min: usize,
        max: Option<usize>,
    },
}

#[derive(Debug)]
pub(crate) struct ResidualArena {
    nodes: Vec<ResidualNode>,
    ids: FxHashMap<ResidualNode, ResidualId>,
    nullable: Vec<bool>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    first_bytes_cache: Vec<Option<U8Set>>,
    nonempty_cache: Vec<Option<bool>>,
    empty: ResidualId,
    epsilon: ResidualId,
    sigma_star: ResidualId,
}

const TRANSITION_UNKNOWN: u32 = u32::MAX;
const DEFAULT_LIVENESS_STATE_BUDGET: usize = 262_144;
const MAX_FINITE_MASK_DENSE_STATES: usize = 8 * 1024 * 1024;
const DEFAULT_LIVENESS_TRANSITION_BUDGET: usize = 4_194_304;

struct ResidualLivenessBudget {
    state_limit: usize,
    transition_limit: usize,
    states_used: usize,
    transitions_used: usize,
}

impl ResidualLivenessBudget {
    fn new(state_limit: usize, transition_limit: usize) -> Self {
        Self {
            state_limit,
            transition_limit,
            states_used: 0,
            transitions_used: 0,
        }
    }

    fn consume_state(&mut self) -> Result<(), String> {
        self.states_used = self
            .states_used
            .checked_add(1)
            .ok_or_else(|| "dynamic residual liveness state count overflow".to_owned())?;
        if self.states_used > self.state_limit {
            return Err(format!(
                "dynamic residual liveness exceeded state budget ({})",
                self.state_limit
            ));
        }
        Ok(())
    }

    fn consume_transition(&mut self) -> Result<(), String> {
        self.transitions_used = self
            .transitions_used
            .checked_add(1)
            .ok_or_else(|| "dynamic residual liveness transition count overflow".to_owned())?;
        if self.transitions_used > self.transition_limit {
            return Err(format!(
                "dynamic residual liveness exceeded transition budget ({})",
                self.transition_limit
            ));
        }
        Ok(())
    }
}

impl ResidualArena {
    pub(crate) fn from_expr(expr: &Expr) -> Option<(Self, ResidualId)> {
        let mut arena = Self {
            nodes: Vec::new(),
            ids: FxHashMap::default(),
            nullable: Vec::new(),
            transitions: Vec::new(),
            first_bytes_cache: Vec::new(),
            nonempty_cache: Vec::new(),
            empty: 0,
            epsilon: 0,
            sigma_star: 0,
        };
        arena.empty = arena.intern_raw(ResidualNode::Empty, false)?;
        arena.epsilon = arena.intern_raw(ResidualNode::Epsilon, true)?;
        arena.sigma_star = arena.intern_raw(ResidualNode::SigmaStar, true)?;
        let root = arena.compile_expr(expr)?;
        Some((arena, root))
    }

    pub(crate) fn state_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn is_nullable(&self, id: ResidualId) -> bool {
        self.nullable[id as usize]
    }

    pub(crate) fn is_empty(&self, id: ResidualId) -> bool {
        id == self.empty
    }

    fn intern_raw(&mut self, node: ResidualNode, nullable: bool) -> Option<ResidualId> {
        if let Some(&id) = self.ids.get(&node) {
            return Some(id);
        }
        let id = u32::try_from(self.nodes.len()).ok()?;
        self.ids.insert(node.clone(), id);
        self.nodes.push(node);
        self.nullable.push(nullable);
        self.transitions.push(None);
        self.first_bytes_cache.push(None);
        self.nonempty_cache.push(None);
        Some(id)
    }

    fn compile_expr(&mut self, expr: &Expr) -> Option<ResidualId> {
        match expr {
            Expr::U8Seq(bytes) => self.literal(bytes),
            Expr::U8Class(bytes) => self.class(*bytes),
            Expr::Dfa(dfa) => {
                if dfa.num_states() == 0 {
                    return None;
                }
                self.dfa(ResidualDfa(Arc::clone(dfa)), &[0])
            }
            Expr::Intersect { expr, intersect } => {
                let left = self.compile_expr(expr)?;
                let right = self.compile_expr(intersect)?;
                self.intersect(left, right)
            }
            Expr::Seq(parts) => {
                let parts = parts
                    .iter()
                    .map(|part| self.compile_expr(part))
                    .collect::<Option<Vec<_>>>()?;
                self.seq(parts)
            }
            Expr::Choice(parts) => {
                let parts = parts
                    .iter()
                    .map(|part| self.compile_expr(part))
                    .collect::<Option<Vec<_>>>()?;
                self.choice(parts)
            }
            Expr::Exclude { expr, exclude } => {
                let left = self.compile_expr(expr)?;
                let right = self.compile_expr(exclude)?;
                self.exclude(left, right)
            }
            Expr::Repeat { expr, min, max } => {
                let body = self.compile_expr(expr)?;
                self.repeat(body, *min, *max)
            }
            Expr::Shared(inner) => self.compile_expr(inner),
            Expr::Epsilon => Some(self.epsilon),
        }
    }

    fn literal(&mut self, bytes: &[u8]) -> Option<ResidualId> {
        if bytes.is_empty() {
            return Some(self.epsilon);
        }
        self.intern_raw(
            ResidualNode::Literal {
                bytes: Arc::from(bytes),
                offset: 0,
            },
            false,
        )
    }

    fn literal_at(&mut self, bytes: Arc<[u8]>, offset: u32) -> Option<ResidualId> {
        if offset as usize >= bytes.len() {
            return Some(self.epsilon);
        }
        self.intern_raw(ResidualNode::Literal { bytes, offset }, false)
    }

    fn class(&mut self, bytes: U8Set) -> Option<ResidualId> {
        if bytes.is_empty() {
            Some(self.empty)
        } else {
            self.intern_raw(ResidualNode::Class(bytes), false)
        }
    }

    fn dfa(&mut self, dfa: ResidualDfa, roots: &[u32]) -> Option<ResidualId> {
        if roots.iter().any(|&state| state as usize >= dfa.num_states()) {
            return Some(self.empty);
        }
        let mut states = dfa.epsilon_closure(roots);
        states.sort_unstable();
        states.dedup();
        if states.is_empty() {
            return Some(self.empty);
        }
        let nullable = states.iter().any(|&state| !dfa.finalizers(state).is_empty());
        self.intern_raw(
            ResidualNode::Dfa {
                dfa,
                states: states.into_vec().into_boxed_slice(),
            },
            nullable,
        )
    }

    fn seq(&mut self, parts: Vec<ResidualId>) -> Option<ResidualId> {
        let mut flat = Vec::new();
        for part in parts {
            if part == self.empty {
                return Some(self.empty);
            }
            if part == self.epsilon {
                continue;
            }
            match self.nodes[part as usize].clone() {
                ResidualNode::Seq(children) => flat.extend(children.iter().copied()),
                _ => flat.push(part),
            }
        }
        match flat.len() {
            0 => Some(self.epsilon),
            1 => Some(flat[0]),
            _ => {
                let nullable = flat.iter().all(|&id| self.is_nullable(id));
                self.intern_raw(ResidualNode::Seq(flat.into_boxed_slice()), nullable)
            }
        }
    }

    fn choice(&mut self, parts: Vec<ResidualId>) -> Option<ResidualId> {
        let mut flat = Vec::new();
        for part in parts {
            if part == self.sigma_star {
                return Some(self.sigma_star);
            }
            if part == self.empty {
                continue;
            }
            match self.nodes[part as usize].clone() {
                ResidualNode::Choice(children) => flat.extend(children.iter().copied()),
                _ => flat.push(part),
            }
        }
        flat.sort_unstable();
        flat.dedup();
        match flat.len() {
            0 => Some(self.empty),
            1 => Some(flat[0]),
            _ => {
                let nullable = flat.iter().any(|&id| self.is_nullable(id));
                self.intern_raw(ResidualNode::Choice(flat.into_boxed_slice()), nullable)
            }
        }
    }

    fn intersect(&mut self, mut left: ResidualId, mut right: ResidualId) -> Option<ResidualId> {
        if left == self.empty || right == self.empty {
            return Some(self.empty);
        }
        if left == self.sigma_star {
            return Some(right);
        }
        if right == self.sigma_star {
            return Some(left);
        }
        if left == right {
            return Some(left);
        }
        if left == self.epsilon {
            return Some(if self.is_nullable(right) { self.epsilon } else { self.empty });
        }
        if right == self.epsilon {
            return Some(if self.is_nullable(left) { self.epsilon } else { self.empty });
        }
        if right < left {
            std::mem::swap(&mut left, &mut right);
        }
        self.intern_raw(
            ResidualNode::Intersect(left, right),
            self.is_nullable(left) && self.is_nullable(right),
        )
    }

    fn exclude(&mut self, left: ResidualId, right: ResidualId) -> Option<ResidualId> {
        if left == self.empty || left == right {
            return Some(self.empty);
        }
        if right == self.sigma_star {
            return Some(self.empty);
        }
        if right == self.empty {
            return Some(left);
        }
        if left == self.epsilon {
            return Some(if self.is_nullable(right) { self.empty } else { self.epsilon });
        }
        if right == self.epsilon && !self.is_nullable(left) {
            return Some(left);
        }
        self.intern_raw(
            ResidualNode::Exclude(left, right),
            self.is_nullable(left) && !self.is_nullable(right),
        )
    }

    fn repeat(
        &mut self,
        mut body: ResidualId,
        mut min: usize,
        max: Option<usize>,
    ) -> Option<ResidualId> {
        if max.is_some_and(|max| min > max) {
            return Some(self.empty);
        }
        if max == Some(0) {
            return Some(if min == 0 { self.epsilon } else { self.empty });
        }
        if body == self.empty {
            return Some(if min == 0 { self.epsilon } else { self.empty });
        }
        if body == self.epsilon {
            return Some(self.epsilon);
        }
        if body == self.sigma_star {
            return Some(self.sigma_star);
        }

        if min == 0
            && max.is_none()
            && matches!(self.nodes[body as usize], ResidualNode::Class(bytes) if bytes.is_full())
        {
            return Some(self.sigma_star);
        }

        // If B is nullable, B^[m,n] equals (B\\{epsilon})^[0,n]. Empty
        // copies can satisfy the lower bound without consuming input. This is
        // the crucial normalization that prevents a derivative from skipping
        // O(n) nullable copies when n is enormous.
        if self.is_nullable(body) {
            body = self.exclude(body, self.epsilon)?;
            min = 0;
            if body == self.empty {
                return Some(self.epsilon);
            }
        }

        if min == 1 && max == Some(1) {
            return Some(body);
        }
        self.intern_raw(
            ResidualNode::Repeat { body, min, max },
            min == 0,
        )
    }

    pub(crate) fn step(&mut self, id: ResidualId, byte: u8) -> Option<ResidualId> {
        if let Some(row) = self.transitions[id as usize].as_ref() {
            let cached = row[byte as usize];
            if cached != TRANSITION_UNKNOWN {
                return Some(cached);
            }
        }
        let target = self.derive_uncached(id, byte)?;
        let row = self.transitions[id as usize]
            .get_or_insert_with(|| Box::new([TRANSITION_UNKNOWN; 256]));
        row[byte as usize] = target;
        Some(target)
    }

    fn sparse_step(
        &mut self,
        id: ResidualId,
        byte: u8,
        budget: &mut ResidualLivenessBudget,
        cache: &mut FxHashMap<u64, ResidualId>,
    ) -> Result<ResidualId, String> {
        let key = (u64::from(id) << 8) | u64::from(byte);
        if let Some(&target) = cache.get(&key) {
            return Ok(target);
        }
        // Charge every actual derivative computation, including recursive
        // child derivatives. This makes the transition budget a real bound on
        // the temporary sparse cache/work rather than only on outer BFS edges.
        budget.consume_transition()?;
        let target = self.derive_uncached_sparse(id, byte, budget, cache)?;
        cache.insert(key, target);
        Ok(target)
    }

    /// Derivative used only by exact liveness reachability. Unlike `step`, it
    /// must not populate the persistent dense 256-entry transition row for
    /// every explored residual: a hard Boolean liveness proof can visit many
    /// states that normal token traversal will never touch. A temporary sparse
    /// cache keeps repeated sub-derivatives cheap without retaining
    /// `O(256 * visited_states)` memory after the query completes.
    fn derive_uncached_sparse(
        &mut self,
        id: ResidualId,
        byte: u8,
        budget: &mut ResidualLivenessBudget,
        cache: &mut FxHashMap<u64, ResidualId>,
    ) -> Result<ResidualId, String> {
        let overflow = || "dynamic residual state-id overflow".to_owned();
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Ok(self.empty),
            ResidualNode::SigmaStar => Ok(self.sigma_star),
            ResidualNode::Literal { bytes, offset } => {
                if bytes[offset as usize] != byte {
                    Ok(self.empty)
                } else {
                    self.literal_at(bytes, offset + 1).ok_or_else(overflow)
                }
            }
            ResidualNode::Class(bytes) => Ok(if bytes.contains(byte) {
                self.epsilon
            } else {
                self.empty
            }),
            ResidualNode::Dfa { dfa, states } => {
                let mut targets = Vec::new();
                for &state in states.iter() {
                    if let Some(target) = dfa.step(state, byte) {
                        targets.push(target);
                    }
                }
                self.dfa(dfa, &targets).ok_or_else(overflow)
            }
            ResidualNode::Choice(parts) => {
                let mut derivatives = Vec::with_capacity(parts.len());
                for &part in parts.iter() {
                    derivatives.push(self.sparse_step(part, byte, budget, cache)?);
                }
                self.choice(derivatives).ok_or_else(overflow)
            }
            ResidualNode::Seq(parts) => {
                let mut alternatives = Vec::new();
                for index in 0..parts.len() {
                    let head = parts[index];
                    let derivative = self.sparse_step(head, byte, budget, cache)?;
                    if derivative != self.empty {
                        let mut sequence = Vec::with_capacity(parts.len() - index);
                        sequence.push(derivative);
                        sequence.extend_from_slice(&parts[index + 1..]);
                        alternatives.push(self.seq(sequence).ok_or_else(overflow)?);
                    }
                    if !self.is_nullable(head) {
                        break;
                    }
                }
                self.choice(alternatives).ok_or_else(overflow)
            }
            ResidualNode::Intersect(left, right) => {
                let left = self.sparse_step(left, byte, budget, cache)?;
                let right = self.sparse_step(right, byte, budget, cache)?;
                self.intersect(left, right).ok_or_else(overflow)
            }
            ResidualNode::Exclude(left, right) => {
                let left = self.sparse_step(left, byte, budget, cache)?;
                let right = self.sparse_step(right, byte, budget, cache)?;
                self.exclude(left, right).ok_or_else(overflow)
            }
            ResidualNode::Repeat { body, min, max } => {
                let derivative = self.sparse_step(body, byte, budget, cache)?;
                if derivative == self.empty {
                    return Ok(self.empty);
                }
                let next_max = max.map(|max| max - 1);
                let tail = self
                    .repeat(body, min.saturating_sub(1), next_max)
                    .ok_or_else(overflow)?;
                self.seq(vec![derivative, tail]).ok_or_else(overflow)
            }
        }
    }

    fn derive_uncached(&mut self, id: ResidualId, byte: u8) -> Option<ResidualId> {
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Some(self.empty),
            ResidualNode::SigmaStar => Some(self.sigma_star),
            ResidualNode::Literal { bytes, offset } => {
                if bytes[offset as usize] != byte {
                    Some(self.empty)
                } else {
                    self.literal_at(bytes, offset + 1)
                }
            }
            ResidualNode::Class(bytes) => {
                Some(if bytes.contains(byte) { self.epsilon } else { self.empty })
            }
            ResidualNode::Dfa { dfa, states } => {
                let mut targets = Vec::new();
                for &state in states.iter() {
                    if let Some(target) = dfa.step(state, byte) {
                        targets.push(target);
                    }
                }
                self.dfa(dfa, &targets)
            }
            ResidualNode::Choice(parts) => {
                let derivatives = parts
                    .iter()
                    .map(|&part| self.step(part, byte))
                    .collect::<Option<Vec<_>>>()?;
                self.choice(derivatives)
            }
            ResidualNode::Seq(parts) => {
                let mut alternatives = Vec::new();
                for index in 0..parts.len() {
                    let head = parts[index];
                    let derivative = self.step(head, byte)?;
                    if derivative != self.empty {
                        let mut sequence = Vec::with_capacity(parts.len() - index);
                        sequence.push(derivative);
                        sequence.extend_from_slice(&parts[index + 1..]);
                        alternatives.push(self.seq(sequence)?);
                    }
                    if !self.is_nullable(head) {
                        break;
                    }
                }
                self.choice(alternatives)
            }
            ResidualNode::Intersect(left, right) => {
                let left = self.step(left, byte)?;
                let right = self.step(right, byte)?;
                self.intersect(left, right)
            }
            ResidualNode::Exclude(left, right) => {
                let left = self.step(left, byte)?;
                let right = self.step(right, byte)?;
                self.exclude(left, right)
            }
            ResidualNode::Repeat { body, min, max } => {
                let derivative = self.step(body, byte)?;
                if derivative == self.empty {
                    return Some(self.empty);
                }
                let next_max = max.map(|max| max - 1);
                let tail = self.repeat(body, min.saturating_sub(1), next_max)?;
                self.seq(vec![derivative, tail])
            }
        }
    }

    fn has_nonempty_fast(&mut self, id: ResidualId) -> Option<bool> {
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Some(false),
            ResidualNode::SigmaStar => Some(true),
            ResidualNode::Literal { .. } | ResidualNode::Class(_) => Some(true),
            // `Expr::Dfa` is allowed to carry stale derived future metadata,
            // so there is no metadata-only fast proof here. Let the ordinary
            // bounded derivative reachability below answer this exactly; that
            // keeps arbitrary embedded-DFA graphs under the same hard work
            // ceilings as every other non-structural residual.
            ResidualNode::Dfa { .. } => None,
            ResidualNode::Choice(parts) => {
                let mut unknown = false;
                for &part in parts.iter() {
                    match self.has_nonempty_fast(part) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
                (!unknown).then_some(false)
            }
            ResidualNode::Seq(parts) => {
                let mut any_nonempty = false;
                let mut unknown = false;
                for &part in parts.iter() {
                    let any_word = if self.is_nullable(part) {
                        Some(true)
                    } else {
                        self.has_nonempty_fast(part)
                    };
                    match any_word {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => unknown = true,
                    }
                    match self.has_nonempty_fast(part) {
                        Some(true) => any_nonempty = true,
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
                if !unknown {
                    Some(any_nonempty)
                } else if !any_nonempty {
                    None
                } else {
                    None
                }
            }
            ResidualNode::Repeat { body, max, .. } => {
                if max == Some(0) {
                    Some(false)
                } else {
                    self.has_nonempty_fast(body)
                }
            }
            ResidualNode::Intersect(_, _) | ResidualNode::Exclude(_, _) => None,
        }
    }

    /// Byte alphabet that can possibly begin a nonempty word of this
    /// residual. For exclusion this is deliberately an over-approximation
    /// (`FIRST(left)`), which is sufficient for exact reachability because the
    /// derivative itself still decides whether the byte is actually live.
    fn first_bytes(&mut self, id: ResidualId) -> Option<U8Set> {
        if let Some(bytes) = self.first_bytes_cache[id as usize] {
            return Some(bytes);
        }
        let bytes = match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => U8Set::empty(),
            ResidualNode::SigmaStar => U8Set::all(),
            ResidualNode::Literal { bytes, offset } => {
                U8Set::single(*bytes.get(offset as usize)?)
            }
            ResidualNode::Class(bytes) => bytes,
            ResidualNode::Dfa { dfa, states } => {
                let mut bytes = U8Set::empty();
                for &state in states.iter() {
                    for (byte, _) in dfa.states().get(state as usize)?.transitions.iter() {
                        bytes.insert(byte);
                    }
                }
                bytes
            }
            ResidualNode::Choice(parts) => {
                let mut bytes = U8Set::empty();
                for &part in parts.iter() {
                    bytes |= self.first_bytes(part)?;
                }
                bytes
            }
            ResidualNode::Seq(parts) => {
                let mut bytes = U8Set::empty();
                for &part in parts.iter() {
                    bytes |= self.first_bytes(part)?;
                    if !self.is_nullable(part) {
                        break;
                    }
                }
                bytes
            }
            ResidualNode::Intersect(left, right) => {
                self.first_bytes(left)?.intersection(&self.first_bytes(right)?)
            }
            ResidualNode::Exclude(left, _) => self.first_bytes(left)?,
            ResidualNode::Repeat { body, max, .. } => {
                if max == Some(0) {
                    U8Set::empty()
                } else {
                    self.first_bytes(body)?
                }
            }
        };
        self.first_bytes_cache[id as usize] = Some(bytes);
        Some(bytes)
    }

    /// Cheap exact answer when structural recursion is sufficient, otherwise
    /// a conservative `true`. This is suitable for the infallible tokenizer
    /// future bitset: dynamic mask/commit perform the fallible exact check at
    /// token boundaries before retaining a symbolic residual.
    fn conservative_has_future(&mut self, id: ResidualId) -> bool {
        if let Some(value) = self.nonempty_cache[id as usize] {
            return value;
        }
        self.has_nonempty_fast(id).unwrap_or(true)
    }

    /// Exact existence of a nonempty accepted continuation. Most expressions,
    /// including giant bounded repeats, resolve structurally in O(AST) time.
    /// Boolean combinations fall back to lazy derivative-graph reachability.
    /// The search has a hard resource ceiling; exceeding it is an error, never
    /// a semantic "dead" result.
    pub(crate) fn has_future(&mut self, id: ResidualId) -> Result<bool, String> {
        self.has_future_with_budget(
            id,
            DEFAULT_LIVENESS_STATE_BUDGET,
            DEFAULT_LIVENESS_TRANSITION_BUDGET,
        )
    }

    fn has_future_with_budget(
        &mut self,
        id: ResidualId,
        state_budget: usize,
        transition_budget: usize,
    ) -> Result<bool, String> {
        let mut budget = ResidualLivenessBudget::new(state_budget, transition_budget);
        self.has_future_with_work_budget(id, &mut budget)
    }

    fn has_future_with_work_budget(
        &mut self,
        id: ResidualId,
        budget: &mut ResidualLivenessBudget,
    ) -> Result<bool, String> {
        if let Some(value) = self.nonempty_cache[id as usize] {
            return Ok(value);
        }
        if let Some(value) = self.has_nonempty_fast(id) {
            self.nonempty_cache[id as usize] = Some(value);
            return Ok(value);
        }

        // Positive-word existence composes exactly through these regular
        // operators. Resolve their children independently so a giant repeat
        // never turns a hard *body* liveness question into a search over its
        // repetition counter. Boolean language relations remain on the general
        // derivative-graph fallback below.
        let structural = match self.nodes[id as usize].clone() {
            ResidualNode::Choice(parts) => {
                let mut live = false;
                for &part in parts.iter() {
                    if self.has_future_with_work_budget(part, budget)? {
                        live = true;
                        break;
                    }
                }
                Some(live)
            }
            ResidualNode::Seq(parts) => {
                let has_nonnullable = parts.iter().any(|&part| !self.is_nullable(part));
                if has_nonnullable {
                    let mut live = true;
                    for &part in parts.iter() {
                        if self.is_nullable(part) {
                            continue;
                        }
                        if !self.has_future_with_work_budget(part, budget)? {
                            live = false;
                            break;
                        }
                    }
                    // Every live nonnullable component contributes at least one
                    // byte, while nullable siblings can always contribute
                    // epsilon. Their positive-word languages are irrelevant.
                    Some(live)
                } else {
                    let mut live = false;
                    for &part in parts.iter() {
                        if self.has_future_with_work_budget(part, budget)? {
                            live = true;
                            break;
                        }
                    }
                    Some(live)
                }
            }
            ResidualNode::Repeat { body, max, .. } => Some(
                max != Some(0) && self.has_future_with_work_budget(body, budget)?,
            ),
            _ => None,
        };
        if let Some(value) = structural {
            self.nonempty_cache[id as usize] = Some(value);
            return Ok(value);
        }

        let mut seen = FxHashSet::<ResidualId>::default();
        let mut queue = VecDeque::from([id]);
        let mut sparse_transitions = FxHashMap::<u64, ResidualId>::default();
        seen.insert(id);
        budget.consume_state()?;
        while let Some(state) = queue.pop_front() {
            let first_bytes = self
                .first_bytes(state)
                .ok_or_else(|| "dynamic residual FIRST-set construction overflow".to_owned())?;
            for byte in first_bytes.iter() {
                let target =
                    self.sparse_step(state, byte, budget, &mut sparse_transitions)?;
                if target == self.empty {
                    continue;
                }
                if self.is_nullable(target) {
                    self.nonempty_cache[id as usize] = Some(true);
                    return Ok(true);
                }
                if seen.insert(target) {
                    budget.consume_state()?;
                    queue.push_back(target);
                }
            }
        }
        self.nonempty_cache[id as usize] = Some(false);
        Ok(false)
    }
}

// Exact liveness oracle for the important bounded-code intersection shape
// emitted by the JSON Schema string importer:
//
//     pattern_language intersect prefix + C^[min,max] + suffix
//
// `C` must be a deterministic, non-nullable prefix code and the first suffix
// byte must not begin a productive C word.  JSON_STRING_CHAR satisfies these
// conditions.  The exact byte residual remains authoritative for transitions;
// this sidecar proves only the Boolean observation "some nonempty accepted
// continuation exists".
//
// At a C boundary, consuming one complete C word induces a finite relation on
// states of the independently compiled pattern DFA.  Therefore future
// liveness is exactly existence of a path whose number of relation edges lies
// in the remaining repetition interval.  Binary relation doubling answers
// that interval query in O(log max) relation applications without expanding
// the repeat counter.

const MAX_BOUNDED_CODE_ORACLE_PATTERN_STATES: usize = 4_096;
const MAX_BOUNDED_CODE_ORACLE_BODY_PRODUCT_CELLS: usize = 2_000_000;
const MAX_BOUNDED_CODE_ORACLE_RELATION_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BoundedCodeEnvelopeState {
    Prefix { next: usize },
    Body { completed: usize, body_state: u32 },
    Suffix { next: usize },
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoundedCodeOracleCoordinate {
    pattern_state: u32,
    envelope: BoundedCodeEnvelopeState,
}

/// Opaque exact bounded-code residual coordinate for the dynamic masker's
/// parser-transparent first-match lane. The runtime index is part of the key
/// so coordinates from distinct terminals cannot be mixed accidentally.
///
/// This is lexer state only. It contains no parser state, token result, or DWA
/// effect.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualResidualDirectCoordinate {
    pub(super) runtime_index: u32,
    coordinate: BoundedCodeOracleCoordinate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundedCodeOracleSlot {
    Unknown,
    Exact(BoundedCodeOracleCoordinate),
    Ambiguous,
}

#[derive(Debug, Clone)]
struct SparseBoolRelation {
    state_count: usize,
    row_offsets: Arc<[u32]>,
    targets: Arc<[u16]>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BoolRelation {
    // `sparse` is an in-memory representation for current compact artifacts.
    // Skipping it preserves the historical bincode shape (`rows` only), so old
    // oracle payloads continue to deserialize unchanged.
    rows: Vec<BitSet>,
    #[serde(skip)]
    sparse: Option<Arc<SparseBoolRelation>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SparseBoolRelationWire {
    state_count: u32,
    row_offsets: Vec<u32>,
    targets: Vec<u16>,
}

impl SparseBoolRelationWire {
    fn from_relation(relation: &BoolRelation) -> Option<Self> {
        let state_count = relation.state_count();
        if state_count > u16::MAX as usize + 1 {
            return None;
        }
        let mut row_offsets = Vec::with_capacity(state_count + 1);
        let mut targets = Vec::new();
        row_offsets.push(0);
        for state in 0..state_count {
            relation.for_each_target(state, |target| targets.push(target as u16))?;
            row_offsets.push(u32::try_from(targets.len()).ok()?);
        }
        Some(Self {
            state_count: u32::try_from(state_count).ok()?,
            row_offsets,
            targets,
        })
    }

    fn into_relation(self, expected_states: usize) -> Option<BoolRelation> {
        let state_count = self.state_count as usize;
        if state_count != expected_states
            || self.row_offsets.len() != state_count + 1
            || self.row_offsets.first().copied() != Some(0)
            || self.row_offsets.last().copied()? as usize != self.targets.len()
            || self.row_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self.targets.iter().any(|&target| target as usize >= state_count)
        {
            return None;
        }
        Some(BoolRelation {
            rows: Vec::new(),
            sparse: Some(Arc::new(SparseBoolRelation {
                state_count,
                row_offsets: Arc::from(self.row_offsets.into_boxed_slice()),
                targets: Arc::from(self.targets.into_boxed_slice()),
            })),
        })
    }
}

impl BoolRelation {
    fn identity(states: usize) -> Self {
        let mut rows = Vec::with_capacity(states);
        for state in 0..states {
            let mut row = BitSet::new(states);
            row.set(state);
            rows.push(row);
        }
        Self { rows, sparse: None }
    }

    #[inline]
    fn state_count(&self) -> usize {
        self.sparse.as_ref().map_or(self.rows.len(), |sparse| sparse.state_count)
    }

    fn for_each_target(&self, state: usize, mut visit: impl FnMut(usize)) -> Option<()> {
        if let Some(sparse) = &self.sparse {
            let start = *sparse.row_offsets.get(state)? as usize;
            let end = *sparse.row_offsets.get(state + 1)? as usize;
            for &target in sparse.targets.get(start..end)? {
                visit(target as usize);
            }
            return Some(());
        }
        for target in self.rows.get(state)?.iter() {
            visit(target);
        }
        Some(())
    }

    #[inline]
    fn any_target_in(&self, state: usize, targets: &BitSet) -> Option<bool> {
        if let Some(sparse) = &self.sparse {
            let start = *sparse.row_offsets.get(state)? as usize;
            let end = *sparse.row_offsets.get(state + 1)? as usize;
            return Some(
                sparse.targets.get(start..end)?.iter().any(|&target| targets.contains(target as usize)),
            );
        }
        Some(!self.rows.get(state)?.is_disjoint(targets))
    }

    fn row_bitset(&self, state: usize) -> Option<BitSet> {
        if self.sparse.is_none() {
            return self.rows.get(state).cloned();
        }
        let mut row = BitSet::new(self.state_count());
        self.for_each_target(state, |target| row.set(target))?;
        Some(row)
    }

    fn valid_for(&self, states: usize) -> bool {
        if self.state_count() != states {
            return false;
        }
        if let Some(sparse) = &self.sparse {
            return sparse.row_offsets.len() == states + 1
                && sparse.row_offsets.first().copied() == Some(0)
                && sparse.row_offsets.last().copied().map(|value| value as usize)
                    == Some(sparse.targets.len())
                && sparse.row_offsets.windows(2).all(|pair| pair[0] <= pair[1])
                && sparse.targets.iter().all(|&state| (state as usize) < states);
        }
        self.rows
            .iter()
            .all(|row| row.iter().all(|state| state < states))
    }

    fn apply(&self, states: &BitSet) -> BitSet {
        let mut out = BitSet::new(self.state_count());
        if let Some(sparse) = &self.sparse {
            for state in states.iter() {
                let Some(&start) = sparse.row_offsets.get(state) else { continue };
                let Some(&end) = sparse.row_offsets.get(state + 1) else { continue };
                for &target in &sparse.targets[start as usize..end as usize] {
                    out.set(target as usize);
                }
            }
            return out;
        }
        for state in states.iter() {
            out.union_with(&self.rows[state]);
        }
        out
    }

    /// Relation composition in execution order: first `self`, then `next`.
    fn then(&self, next: &Self) -> Self {
        debug_assert_eq!(self.state_count(), next.state_count());
        let rows = (0..self.state_count())
            .map(|state| next.apply(&self.row_bitset(state).expect("valid relation row")))
            .collect::<Vec<_>>();
        Self { rows, sparse: None }
    }

    fn union(&self, other: &Self) -> Self {
        debug_assert_eq!(self.state_count(), other.state_count());
        let rows = (0..self.state_count())
            .map(|state| {
                let mut row = self.row_bitset(state).expect("valid relation row");
                row.union_with(&other.row_bitset(state).expect("valid relation row"));
                row
            })
            .collect::<Vec<_>>();
        Self { rows, sparse: None }
    }

    fn transpose(&self) -> Option<Self> {
        let states = self.state_count();
        let mut rows = (0..states)
            .map(|_| BitSet::new(states))
            .collect::<Vec<_>>();
        for source in 0..states {
            self.for_each_target(source, |target| rows[target].set(source))?;
        }
        Some(Self { rows, sparse: None })
    }
}

/// Cheap structural prefilter for Static residual selection. This deliberately
/// does not claim that the full oracle is constructible: state/product/resource
/// certification remains in `BoundedCodeIntersectionOracle::from_expr`. Static
/// construction uses the preserving runtime constructor as that exact final
/// proof and falls back if it fails, avoiding repeated expensive proof builds.
pub(crate) fn expression_may_support_bounded_code_liveness_oracle(expr: &Expr) -> bool {
    let mut operands = Vec::new();
    flatten_intersection_operands(expr, &mut operands);
    if operands.len() < 2 {
        return false;
    }
    let mut envelope: Option<(Vec<u8>, Expr, usize, usize, Vec<u8>)> = None;
    let mut pattern_operands = Vec::new();
    for operand in operands {
        if let Some((prefix, body, min, max, suffix)) = bounded_code_envelope(operand) {
            match &mut envelope {
                None => {
                    envelope = Some((prefix, body, min, max, suffix));
                    continue;
                }
                Some((existing_prefix, existing_body, existing_min, existing_max, existing_suffix))
                    if *existing_prefix == prefix
                        && *existing_body == body
                        && *existing_suffix == suffix =>
                {
                    *existing_min = (*existing_min).max(min);
                    *existing_max = (*existing_max).min(max);
                    continue;
                }
                Some(_) => {}
            }
        }
        pattern_operands.push(operand.clone());
    }
    let Some((_, body_expr, _, max, _)) = envelope else {
        return false;
    };
    if pattern_operands.is_empty() || max == usize::MAX {
        return false;
    }
    let Some(pattern_expr) = pattern_operands.into_iter().reduce(|expr, intersect| Expr::Intersect {
        expr: Box::new(expr),
        intersect: Box::new(intersect),
    }) else {
        return false;
    };
    !expression_contains_large_bounded_repeat(&pattern_expr)
        && !expression_contains_large_bounded_repeat(&body_expr)
}

/// Return whether the exact bounded-code liveness oracle can be constructed
/// for this expression without changing its language. Dynamic compilation uses
/// this as a proof-backed representation selector for bounded intersections
/// that would otherwise be eagerly materialized below the generic giant-repeat
/// threshold.
pub(crate) fn expression_supports_bounded_code_liveness_oracle(expr: &Expr) -> bool {
    build_bounded_code_liveness_oracle(expr).is_some()
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(super) struct BoundedCodeIntersectionOracle {
    pattern: Arc<DFA>,
    body: Arc<DFA>,
    body_productive: Box<[bool]>,
    prefix: Arc<[u8]>,
    suffix: Arc<[u8]>,
    min: usize,
    max: usize,
    suffix_accepting: BitSet,
    completion_relations: Vec<Option<BoolRelation>>,
    /// Runtime-only memo for singleton completion rows. Exact liveness queries
    /// usually start from one pattern state; building the complete relation for
    /// every pattern state on the first such query can dominate mask latency.
    #[serde(skip)]
    completion_row_cache: FxHashMap<(u32, u32), BitSet>,
    /// Mask-result-only proof cache; never changes recognition or persisted
    /// finite-mask DFA coordinates. Each bound uses a body-copy-reachable
    /// closed subset, not observed mask equality.
    #[serde(skip)]
    mask_result_future_headroom: FxHashMap<(u32, u32), usize>,
    exact_powers: Vec<BoolRelation>,
    prefix_sums: Vec<BoolRelation>,
}

pub(super) fn build_bounded_code_liveness_oracle(
    expr: &Expr,
) -> Option<BoundedCodeIntersectionOracle> {
    BoundedCodeIntersectionOracle::from_dynamic_expr(expr)
}

const SPARSE_BOUNDED_CODE_ORACLE_MAGIC: [u8; 4] = *b"BCO2";

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SparseBoundedCodeOracleWire {
    pattern: Arc<DFA>,
    body: Arc<DFA>,
    body_productive: Box<[bool]>,
    prefix: Arc<[u8]>,
    suffix: Arc<[u8]>,
    min: usize,
    max: usize,
    suffix_accepting: BitSet,
    completion_relations: Vec<Option<SparseBoolRelationWire>>,
    exact_powers: Vec<SparseBoolRelationWire>,
    prefix_sums: Vec<SparseBoolRelationWire>,
}

impl SparseBoundedCodeOracleWire {
    fn from_oracle(oracle: &BoundedCodeIntersectionOracle) -> Option<Self> {
        Some(Self {
            pattern: Arc::clone(&oracle.pattern),
            body: Arc::clone(&oracle.body),
            body_productive: oracle.body_productive.clone(),
            prefix: Arc::clone(&oracle.prefix),
            suffix: Arc::clone(&oracle.suffix),
            min: oracle.min,
            max: oracle.max,
            suffix_accepting: oracle.suffix_accepting.clone(),
            completion_relations: oracle
                .completion_relations
                .iter()
                .map(|relation| match relation {
                    Some(relation) => Some(Some(SparseBoolRelationWire::from_relation(relation)?)),
                    None => Some(None),
                })
                .collect::<Option<Vec<_>>>()?,
            exact_powers: oracle
                .exact_powers
                .iter()
                .map(SparseBoolRelationWire::from_relation)
                .collect::<Option<Vec<_>>>()?,
            prefix_sums: oracle
                .prefix_sums
                .iter()
                .map(SparseBoolRelationWire::from_relation)
                .collect::<Option<Vec<_>>>()?,
        })
    }

    fn into_oracle(self) -> Option<BoundedCodeIntersectionOracle> {
        let pattern_states = self.pattern.num_states();
        let completion_relations = self
            .completion_relations
            .into_iter()
            .map(|relation| match relation {
                Some(relation) => Some(Some(relation.into_relation(pattern_states)?)),
                None => Some(None),
            })
            .collect::<Option<Vec<_>>>()?;
        let exact_powers = self
            .exact_powers
            .into_iter()
            .map(|relation| relation.into_relation(pattern_states))
            .collect::<Option<Vec<_>>>()?;
        let prefix_sums = self
            .prefix_sums
            .into_iter()
            .map(|relation| relation.into_relation(pattern_states))
            .collect::<Option<Vec<_>>>()?;
        Some(BoundedCodeIntersectionOracle {
            pattern: self.pattern,
            body: self.body,
            body_productive: self.body_productive,
            prefix: self.prefix,
            suffix: self.suffix,
            min: self.min,
            max: self.max,
            suffix_accepting: self.suffix_accepting,
            completion_relations,
            completion_row_cache: FxHashMap::default(),
            mask_result_future_headroom: FxHashMap::default(),
            exact_powers,
            prefix_sums,
        })
    }
}

fn canonicalize_bounded_code_oracle_dfa(dfa: DFA) -> DFA {
    // Oracle coordinates are persisted indirectly through Static TSID tables.
    // Make their component state IDs depend only on DFA language/topology, not
    // hash-map insertion order during an independent compile/load rebuild.
    let (dfa, _) = dfa.minimize_with_state_mapping();
    let n = dfa.num_states();
    if n <= 1 {
        return dfa;
    }

    let mut old_to_new = vec![u32::MAX; n];
    let mut order = Vec::with_capacity(n);
    let mut queue = VecDeque::new();
    old_to_new[0] = 0;
    order.push(0u32);
    queue.push_back(0u32);
    while let Some(source) = queue.pop_front() {
        for byte in 0u16..=255 {
            let Some(target) = dfa.step(source, byte as u8) else { continue; };
            if old_to_new[target as usize] == u32::MAX {
                let next = order.len() as u32;
                old_to_new[target as usize] = next;
                order.push(target);
                queue.push_back(target);
            }
        }
    }
    // `compile_terminal_expr_dfa` should already be root-reachable, but retain
    // any unexpected disconnected states deterministically rather than making
    // this canonicalizer lossy.
    for old in 0..n as u32 {
        if old_to_new[old as usize] == u32::MAX {
            old_to_new[old as usize] = order.len() as u32;
            order.push(old);
        }
    }

    let mut out = DFA::new(n);
    out.ensure_group_capacity(dfa.num_groups());
    for group in 0..dfa.num_groups() as u32 {
        out.set_group_u8set(group, *dfa.group_id_to_u8set(group));
    }
    for (new_state, &old_state) in order.iter().enumerate() {
        out.overwrite_state_metadata(
            new_state as u32,
            dfa.finalizers(old_state).clone(),
            BitSet::new(dfa.num_groups()),
        );
        for byte in 0u16..=255 {
            if let Some(old_target) = dfa.step(old_state, byte as u8) {
                out.add_transition(
                    new_state as u32,
                    byte as u8,
                    old_to_new[old_target as usize],
                );
            }
        }
    }
    out.recompute_possible_futures();
    out
}

fn dfa_global_byte_classes(dfa: &DFA) -> Vec<Vec<u8>> {
    let mut signatures: [Vec<(u32, u32)>; 256] = std::array::from_fn(|_| Vec::new());
    for (state_index, state) in dfa.states().iter().enumerate() {
        let state_index = state_index as u32;
        for (byte, &target) in state.transitions.iter() {
            signatures[byte as usize].push((state_index, target));
        }
    }
    let mut grouped = FxHashMap::<Vec<(u32, u32)>, Vec<u8>>::default();
    for byte in 0u16..=255 {
        grouped
            .entry(std::mem::take(&mut signatures[byte as usize]))
            .or_default()
            .push(byte as u8);
    }
    let mut classes = grouped.into_values().collect::<Vec<_>>();
    classes.sort_unstable_by_key(|members| members[0]);
    classes
}

fn bounded_code_byte_classes(
    pattern: &DFA,
    body: &DFA,
    prefix: &[u8],
    suffix: &[u8],
) -> Vec<Vec<u8>> {
    let pattern_classes = dfa_global_byte_classes(pattern);
    let body_classes = dfa_global_byte_classes(body);
    let mut pattern_class = [0u8; 256];
    let mut body_class = [0u8; 256];
    for (class, members) in pattern_classes.iter().enumerate() {
        for &byte in members {
            pattern_class[byte as usize] = class as u8;
        }
    }
    for (class, members) in body_classes.iter().enumerate() {
        for &byte in members {
            body_class[byte as usize] = class as u8;
        }
    }
    let mut literal = [false; 256];
    for &byte in prefix.iter().chain(suffix.iter()) {
        literal[byte as usize] = true;
    }
    let mut grouped = FxHashMap::<(u8, u8, u16), Vec<u8>>::default();
    for byte in 0u16..=255 {
        let b = byte as u8;
        // Literal bytes must remain individually distinguishable because a
        // prefix/suffix coordinate tests equality with that exact byte.
        let literal_tag = literal[byte as usize].then_some(byte + 1).unwrap_or(0);
        grouped
            .entry((
                pattern_class[byte as usize],
                body_class[byte as usize],
                literal_tag,
            ))
            .or_default()
            .push(b);
    }
    let mut classes = grouped.into_values().collect::<Vec<_>>();
    classes.sort_unstable_by_key(|members| members[0]);
    classes
}

fn expand_exact_byte_classes(dfa: &mut DFA, class_members: &[Vec<u8>]) {
    dfa.states_mut().par_iter_mut().for_each(|state| {
        let class_transitions = std::mem::take(&mut state.transitions);
        let capacity = class_transitions
            .iter()
            .map(|(class, _)| class_members[class as usize].len())
            .sum();
        let mut entries = Vec::with_capacity(capacity);
        for (class, &target) in class_transitions.iter() {
            entries.extend(
                class_members[class as usize]
                    .iter()
                    .copied()
                    .map(|byte| (byte, target)),
            );
        }
        // A single class expands to its already-sorted member bytes. Multiple
        // classes can interleave, so only those states need sorting.
        if class_transitions.len() > 1 {
            entries.sort_unstable_by_key(|entry| entry.0);
        }
        state.transitions = CharTransitions::from_sorted_entries(entries);
    });
    // Expanding a class-labelled edge into one edge per member byte changes
    // only labels/multiplicity, not the directed state graph or finalizers.
    // The minimizer has just recomputed strict possible-future groups on this
    // graph, so recomputing the same fixpoint over the denser byte-expanded
    // transitions would be redundant.
}

impl BoundedCodeIntersectionOracle {
    /// A shortest suffix-reaching path through complete body copies is a
    /// simple path in the relation graph on the P pattern states. It uses at
    /// most P - 1 copies; finishing an already-started copy costs at most one
    /// more. P + 1 is therefore conservative for every productive coordinate.
    /// Retain that witness after the candidate token even when the witness
    /// itself spans several future model tokens.
    fn finite_mask_future_repeat_headroom(&self) -> Option<usize> {
        self.pattern.num_states().checked_add(1)
    }

    fn finite_mask_stencil_crossings(&self, token_crossings: usize) -> Option<usize> {
        token_crossings.checked_add(self.finite_mask_future_repeat_headroom()?)
    }

    /// Close all completions of the current partial body under whole body-code
    /// copies. Every later body boundary belongs to this closed relation graph.
    /// Compute shortest distances to suffix-admitting boundaries in this closed
    /// graph, and retain the maximum finite distance plus two. A shortest path
    /// has at most R-1 copies among R boundary states, so this never exceeds the
    /// global P+1 bound. The extra copies cover partial-body/strict-future cases.
    /// Keep the global P+1 stencil for compiler/recognition/transition caches.
    fn mask_result_future_repeat_headroom(&mut self, pattern_state: u32, body_state: u32) -> Option<usize> {
        let key = (pattern_state, body_state);
        if let Some(&headroom) = self.mask_result_future_headroom.get(&key) {
            return Some(headroom);
        }
        let states = self.pattern.num_states();
        if pattern_state as usize >= states || body_state as usize >= self.body.num_states()
            || self.pattern.has_epsilon_transitions() || self.body.has_epsilon_transitions() {
            return None;
        }
        let mut seen = vec![false; states];
        let mut incoming = vec![Vec::<u32>::new(); states];
        let mut pending = vec![pattern_state];
        seen[pattern_state as usize] = true;
        // Include the source pattern state conservatively even for a partial
        // body. Completing that partial body seeds every real next boundary.
        for target in self.completion_row(body_state, pattern_state).iter_ones() {
            if !seen[target] { seen[target] = true; pending.push(target as u32); }
        }
        while let Some(state) = pending.pop() {
            for target in self.completion_row(0, state).iter_ones() {
                if target >= states { return None; }
                incoming[target].push(state);
                if !seen[target] {
                    seen[target] = true;
                    pending.push(target as u32);
                }
            }
        }
        let mut distances = vec![None::<usize>; states];
        let mut queue = VecDeque::new();
        for state in 0..states {
            if seen[state] && self.suffix_accepting.contains(state) {
                distances[state] = Some(0);
                queue.push_back(state);
            }
        }
        while let Some(state) = queue.pop_front() {
            let distance = distances[state]?.checked_add(1)?;
            for &predecessor in &incoming[state] {
                let predecessor = predecessor as usize;
                if distances[predecessor].is_none() {
                    distances[predecessor] = Some(distance);
                    queue.push_back(predecessor);
                }
            }
        }
        let headroom = distances.into_iter().flatten().max().unwrap_or(0).checked_add(2)?;
        self.mask_result_future_headroom.insert(key, headroom);
        Some(headroom)
    }

    fn from_expr(expr: &Expr) -> Option<Self> {
        Self::from_expr_with_coordinate_policy(expr, true)
    }

    fn from_dynamic_expr(expr: &Expr) -> Option<Self> {
        Self::from_expr_with_coordinate_policy(expr, false)
    }

    fn from_expr_with_coordinate_policy(
        expr: &Expr,
        canonicalize_coordinates: bool,
    ) -> Option<Self> {
        let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_RESIDUAL").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let mut operands = Vec::new();
        flatten_intersection_operands(expr, &mut operands);
        if operands.len() < 2 {
            return None;
        }

        let mut envelope: Option<(Vec<u8>, Expr, usize, usize, Vec<u8>)> = None;
        let mut pattern_operands = Vec::new();
        for operand in operands {
            if let Some((prefix, body, min, max, suffix)) = bounded_code_envelope(operand) {
                match &mut envelope {
                    None => {
                        envelope = Some((prefix, body, min, max, suffix));
                        continue;
                    }
                    Some((existing_prefix, existing_body, existing_min, existing_max, existing_suffix))
                        if *existing_prefix == prefix
                            && *existing_body == body
                            && *existing_suffix == suffix =>
                    {
                        // Intersecting identical code envelopes is exactly an
                        // intersection of their copy-count intervals. This
                        // occurs naturally when JSON Schema `allOf` contains
                        // multiple differently-patterned bounded strings.
                        *existing_min = (*existing_min).max(min);
                        *existing_max = (*existing_max).min(max);
                        continue;
                    }
                    Some(_) => {}
                }
            }
            pattern_operands.push(operand.clone());
        }
        let (prefix, body_expr, min, max, suffix) = envelope?;
        if max == usize::MAX {
            return None;
        }
        // An unbounded copy of the envelope language contributes no additional
        // constraint to the intersection:
        //
        //     prefix body* suffix  &  prefix body{min,max} suffix
        //       == prefix body{min,max} suffix
        //
        // This shape is useful to the exact dynamic representation because it
        // exposes the bounded-code envelope explicitly, but retaining the
        // redundant operand as the oracle's pattern coordinate multiplies every
        // finite mask state by an otherwise unnecessary pattern DFA. Drop only
        // operands whose prefix, body language, and suffix are structurally
        // identical to the selected envelope; any independent pattern/format
        // operand remains untouched.
        pattern_operands.retain(|operand| {
            unbounded_code_envelope(operand).is_none_or(
                |(candidate_prefix, candidate_body, candidate_suffix)| {
                    candidate_prefix != prefix
                        || candidate_body != body_expr
                        || candidate_suffix != suffix
                },
            )
        });
        let pattern_expr = pattern_operands.into_iter().reduce(|expr, intersect| Expr::Intersect {
            expr: Box::new(expr),
            intersect: Box::new(intersect),
        });
        // The oracle is a sidecar for avoiding giant-repeat materialization, so
        // its own proof construction must never eagerly materialize a giant
        // bounded repeat hidden inside either finite coordinate. The outer
        // envelope repeat is represented by the relation-doubling counter and
        // is intentionally not part of this check.
        if pattern_expr
            .as_ref()
            .is_some_and(expression_contains_large_bounded_repeat)
            || expression_contains_large_bounded_repeat(&body_expr)
        {
            return None;
        }
        let pattern_started = profile.then(std::time::Instant::now);
        let pattern_raw = match pattern_expr {
            Some(pattern_expr) => compile_terminal_expr_dfa(&pattern_expr),
            None => universal_pattern_dfa(),
        };
        let pattern_precanonical_states = pattern_raw.num_states();
        let pattern = Arc::new(if canonicalize_coordinates {
            canonicalize_bounded_code_oracle_dfa(pattern_raw)
        } else {
            pattern_raw
        });
        let pattern_ms = pattern_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let body_started = profile.then(std::time::Instant::now);
        let body_raw = compile_terminal_expr_dfa(&body_expr);
        let body_precanonical_states = body_raw.num_states();
        let body = Arc::new(if canonicalize_coordinates {
            canonicalize_bounded_code_oracle_dfa(body_raw)
        } else {
            body_raw
        });
        let body_ms = body_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if pattern.num_states() == 0
            || pattern.num_states() > MAX_BOUNDED_CODE_ORACLE_PATTERN_STATES
            || pattern
                .states()
                .iter()
                .any(|state| !state.epsilon_transitions.is_empty())
            || body.num_states() == 0
            || body
                .states()
                .iter()
                .any(|state| !state.epsilon_transitions.is_empty())
        {
            return None;
        }

        let body_productive = exact_productive_states(&body);
        if !body.finalizers(0).is_empty() || !dfa_language_is_prefix_free(&body, &body_productive) {
            return None;
        }
        // At a repetition boundary a suffix byte must choose exactly one of
        // "start another code word" and "start the suffix".  A transition
        // into a semantically dead body state does not create ambiguity.
        if body
            .step(0, suffix[0])
            .is_some_and(|target| body_productive[target as usize])
        {
            return None;
        }

        let pattern_states = pattern.num_states();
        let body_states = body.num_states();
        if pattern_states.checked_mul(body_states)?
            > MAX_BOUNDED_CODE_ORACLE_BODY_PRODUCT_CELLS
        {
            return None;
        }
        // `apply_up_to(k)` represents the inclusive range 0..=k as `k + 1`
        // binary blocks.  Size the doubling table for `max + 1`, not `max`:
        // when max = 2^n - 1 the inclusive range needs the R^(2^n) block even
        // though an exact count <= max does not. `usize::MAX` was rejected
        // above, so the addition is exact.
        let range_block_count = max + 1;
        let bits = usize::BITS as usize - range_block_count.leading_zeros() as usize;
        let words_per_row = pattern_states.div_ceil(64);
        let relation_bytes = pattern_states
            .checked_mul(words_per_row)?
            .checked_mul(std::mem::size_of::<u64>())?;
        let estimated_relation_bytes = relation_bytes
            .checked_mul(body_states.checked_add(bits.checked_mul(2)?)?)?;
        if estimated_relation_bytes > MAX_BOUNDED_CODE_ORACLE_RELATION_BYTES {
            return None;
        }

        let mut suffix_accepting = BitSet::new(pattern_states);
        for state in 0..pattern_states as u32 {
            if let Some(end) = step_fixed_bytes(&pattern, state, &suffix)
                && !pattern.finalizers(end).is_empty()
            {
                suffix_accepting.set(state as usize);
            }
        }

        let mut oracle = Self {
            pattern,
            body,
            body_productive: body_productive.into_boxed_slice(),
            prefix: Arc::from(prefix.into_boxed_slice()),
            suffix: Arc::from(suffix.into_boxed_slice()),
            min,
            max,
            suffix_accepting,
            completion_relations: vec![None; body_states],
            completion_row_cache: FxHashMap::default(),
            mask_result_future_headroom: FxHashMap::default(),
            exact_powers: Vec::new(),
            prefix_sums: Vec::new(),
        };
        let one_code_started = profile.then(std::time::Instant::now);
        let one_code = oracle.completion_relation(0).clone();
        let one_code_ms = one_code_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        oracle.exact_powers.push(one_code);
        oracle
            .prefix_sums
            .push(BoolRelation::identity(pattern_states));
        let powers_started = profile.then(std::time::Instant::now);
        oracle.ensure_power(bits.saturating_sub(1));
        let powers_ms = powers_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][bounded_code_oracle] canonical={} pattern_states={} pattern_precanonical_states={} body_states={} body_precanonical_states={} min={} max={} bits={} relation_bytes={} pattern_ms={:.3} body_ms={:.3} one_code_ms={:.3} powers_ms={:.3} total_ms={:.3}",
                canonicalize_coordinates,
                pattern_states,
                pattern_precanonical_states,
                body_states,
                body_precanonical_states,
                min,
                max,
                bits,
                relation_bytes,
                pattern_ms,
                body_ms,
                one_code_ms,
                powers_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some(oracle)
    }

    fn root_coordinate(&self) -> BoundedCodeOracleCoordinate {
        BoundedCodeOracleCoordinate {
            pattern_state: 0,
            envelope: BoundedCodeEnvelopeState::Prefix { next: 0 },
        }
    }

    #[inline]
    fn coordinate_accepting(&self, coordinate: BoundedCodeOracleCoordinate) -> bool {
        matches!(coordinate.envelope, BoundedCodeEnvelopeState::Done)
            && !self.pattern.finalizers(coordinate.pattern_state).is_empty()
    }

    fn compute_completion_row(&self, body_state: u32, pattern_start: u32) -> BitSet {
        let pattern_states = self.pattern.num_states();
        let body_states = self.body.num_states();
        let mut targets = BitSet::new(pattern_states);
        let mut seen = FxHashSet::<u64>::default();
        let mut queue = VecDeque::from([(pattern_start, body_state)]);
        seen.insert((u64::from(pattern_start) << 32) | u64::from(body_state));
        while let Some((pattern_state, code_state)) = queue.pop_front() {
            for (byte, &code_target) in self.body.states()[code_state as usize].transitions.iter() {
                let Some(pattern_target) = self.pattern.step(pattern_state, byte) else {
                    continue;
                };
                if !self.body.finalizers(code_target).is_empty() {
                    targets.set(pattern_target as usize);
                    continue;
                }
                if !self.body_productive[code_target as usize] {
                    continue;
                }
                debug_assert!((code_target as usize) < body_states);
                let key = (u64::from(pattern_target) << 32) | u64::from(code_target);
                if seen.insert(key) {
                    queue.push_back((pattern_target, code_target));
                }
            }
        }
        targets
    }

    fn completion_row(&mut self, body_state: u32, pattern_start: u32) -> BitSet {
        if let Some(relation) = self
            .completion_relations
            .get(body_state as usize)
            .and_then(Option::as_ref)
        {
            return relation
                .row_bitset(pattern_start as usize)
                .expect("bounded-code completion relation row must exist");
        }
        let key = (body_state, pattern_start);
        if let Some(row) = self.completion_row_cache.get(&key) {
            return row.clone();
        }
        let row = self.compute_completion_row(body_state, pattern_start);
        self.completion_row_cache.insert(key, row.clone());
        row
    }

    fn completion_relation(&mut self, body_state: u32) -> &BoolRelation {
        let index = body_state as usize;
        if self.completion_relations[index].is_none() {
            let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_RESIDUAL").is_some();
            let started = profile.then(std::time::Instant::now);
            let pattern_states = self.pattern.num_states();
            let rows = (0..pattern_states as u32)
                .map(|pattern_start| self.compute_completion_row(body_state, pattern_start))
                .collect::<Vec<_>>();
            self.completion_relations[index] = Some(BoolRelation { rows, sparse: None });
            if let Some(started) = started {
                eprintln!(
                    "[glrmask/profile][bounded_code_completion_relation] body_state={} pattern_states={} elapsed_ms={:.3}",
                    body_state,
                    pattern_states,
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
        }
        self.completion_relations[index].as_ref().unwrap()
    }

    fn ensure_power(&mut self, bit: usize) {
        while self.exact_powers.len() <= bit {
            let previous_power = self.exact_powers.last().unwrap().clone();
            let previous_sum = self.prefix_sums.last().unwrap().clone();
            let next_power = previous_power.then(&previous_power);
            let shifted_sum = previous_power.then(&previous_sum);
            self.exact_powers.push(next_power);
            self.prefix_sums.push(previous_sum.union(&shifted_sum));
        }
    }

    fn apply_exact_count(&self, mut states: BitSet, count: usize) -> BitSet {
        let mut remaining = count;
        let mut bit = 0usize;
        while remaining != 0 {
            if remaining & 1 != 0 {
                states = self.exact_powers[bit].apply(&states);
                if states.is_empty() {
                    break;
                }
            }
            remaining >>= 1;
            bit += 1;
        }
        states
    }

    /// Union states reachable after any number of whole code words in
    /// `[0, max_extra]`.
    fn apply_up_to(&self, states: BitSet, max_extra: usize) -> BitSet {
        let mut exact_offset = states;
        let mut union = BitSet::new(self.pattern.num_states());
        let mut block_count = max_extra.checked_add(1).unwrap();
        let mut bit = 0usize;
        while block_count != 0 {
            if block_count & 1 != 0 {
                union.union_with(&self.prefix_sums[bit].apply(&exact_offset));
                exact_offset = self.exact_powers[bit].apply(&exact_offset);
            }
            block_count >>= 1;
            bit += 1;
        }
        union
    }

    fn range_reaches_suffix(
        &self,
        starts: BitSet,
        completed: usize,
    ) -> bool {
        if completed > self.max {
            return false;
        }
        let low = self.min.saturating_sub(completed);
        let high = self.max - completed;
        if low > high {
            return false;
        }
        let after_low = self.apply_exact_count(starts, low);
        if after_low.is_empty() {
            return false;
        }
        let reachable = self.apply_up_to(after_low, high - low);
        !reachable.is_disjoint(&self.suffix_accepting)
    }

    /// For each whole-body repetition count `c`, return the pattern states from
    /// which some legal continuation can finish the bounded-code lexeme. This
    /// is the backwards dynamic-programming form of `range_reaches_suffix` and
    /// is much cheaper when a universal slice proof needs the same query for
    /// many pattern states and adjacent counts.
    fn body_boundary_future_sets(&self) -> Option<Vec<BitSet>> {
        // Resource guard: this is a runtime accelerator, not part of language
        // semantics. Giant bounds continue to use the logarithmic relation-
        // powers fallback rather than allocating one bitset per repetition.
        const MAX_CACHED_BOUNDARIES: usize = 4096;
        if self.max > MAX_CACHED_BOUNDARIES {
            return None;
        }
        let relation = self.completion_relations.first()?.as_ref()?;
        let reverse_relation = relation.transpose()?;
        let states = self.pattern.num_states();
        let mut future = (0..=self.max)
            .map(|_| BitSet::new(states))
            .collect::<Vec<_>>();
        for completed in (0..=self.max).rev() {
            if completed >= self.min {
                future[completed].union_with(&self.suffix_accepting);
            }
            if completed == self.max {
                continue;
            }
            let predecessors = reverse_relation.apply(&future[completed + 1]);
            future[completed].union_with(&predecessors);
        }
        Some(future)
    }

    fn step_coordinate(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        byte: u8,
    ) -> Option<BoundedCodeOracleCoordinate> {
        self.step_coordinate_with_max(coordinate, byte, self.max)
    }

    fn step_coordinate_with_max(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        byte: u8,
        max: usize,
    ) -> Option<BoundedCodeOracleCoordinate> {
        let pattern_state = self.pattern.step(coordinate.pattern_state, byte)?;
        let envelope = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if self.prefix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.prefix.len() {
                    BoundedCodeEnvelopeState::Body {
                        completed: 0,
                        body_state: 0,
                    }
                } else {
                    BoundedCodeEnvelopeState::Prefix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                if body_state == 0
                    && completed >= self.min
                    && self.suffix[0] == byte
                {
                    if self.suffix.len() == 1 {
                        BoundedCodeEnvelopeState::Done
                    } else {
                        BoundedCodeEnvelopeState::Suffix { next: 1 }
                    }
                } else {
                    if completed >= max {
                        return None;
                    }
                    let target = self.body.step(body_state, byte)?;
                    if !self.body.finalizers(target).is_empty() {
                        BoundedCodeEnvelopeState::Body {
                            completed: completed.checked_add(1)?,
                            body_state: 0,
                        }
                    } else if self.body_productive[target as usize] {
                        BoundedCodeEnvelopeState::Body {
                            completed,
                            body_state: target,
                        }
                    } else {
                        return None;
                    }
                }
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if self.suffix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.suffix.len() {
                    BoundedCodeEnvelopeState::Done
                } else {
                    BoundedCodeEnvelopeState::Suffix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Done => return None,
        };
        Some(BoundedCodeOracleCoordinate {
            pattern_state,
            envelope,
        })
    }


    fn slice_atom_is_exact_body_code(
        &self,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
    ) -> bool {
        let slice_state_count = slice_accepting.len();
        if slice_state_count == 0
            || slice_can_reach_accepting.len() != slice_state_count
            || slice_start as usize >= slice_state_count
            || slice_class_count == 0
            || slice_transitions.len() != slice_state_count.saturating_mul(slice_class_count)
        {
            return false;
        }
        let mut seen = FxHashSet::<(u32, u32)>::default();
        let mut queue = VecDeque::from([(slice_start, 0u32)]);
        seen.insert((slice_start, 0));
        while let Some((slice_state, body_state)) = queue.pop_front() {
            let row = slice_state as usize * slice_class_count;
            for byte in 0u16..=255 {
                let byte = byte as u8;
                let class = slice_byte_to_class[byte as usize] as usize;
                if class >= slice_class_count {
                    return false;
                }
                let slice_target = slice_transitions[row + class];
                if slice_target as usize >= slice_state_count
                    || !slice_can_reach_accepting[slice_target as usize]
                {
                    continue;
                }
                let Some(body_target) = self.body.step(body_state, byte) else {
                    return false;
                };
                let slice_done = slice_accepting[slice_target as usize];
                let body_done = !self.body.finalizers(body_target).is_empty();
                if slice_done {
                    // One complete slice atom must end exactly at one complete
                    // body code word. We deliberately stop at that boundary;
                    // the caller's slice is a repetition of these atoms.
                    if !body_done {
                        return false;
                    }
                    continue;
                }
                // The body must not finish early inside one slice atom, and the
                // partial code state must still have a completion.
                if body_done || !self.body_productive[body_target as usize] {
                    return false;
                }
                if seen.insert((slice_target, body_target)) {
                    queue.push_back((slice_target, body_target));
                }
            }
        }
        true
    }

    fn invariant_body_repeat_radius(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        atom_is_exact_body_code: bool,
        first_atom_is_total: bool,
        max_repetitions: u32,
    ) -> Option<u32> {
        let BoundedCodeEnvelopeState::Body {
            completed,
            body_state: 0,
        } = coordinate.envelope
        else {
            return None;
        };
        if completed > self.max {
            return Some(0);
        }
        // The completion relation is existential: missing pattern transitions
        // are omitted by compute_completion_row. A surviving self-loop does
        // not imply that every slice atom is defined. Require a separately
        // completed universal first-atom proof before extrapolating the loop.
        if !atom_is_exact_body_code || !first_atom_is_total {
            return None;
        }
        let relation = self.completion_relations.first()?.as_ref()?;
        let mut saw_target = false;
        let mut invariant = true;
        relation.for_each_target(coordinate.pattern_state as usize, |target| {
            saw_target = true;
            if target != coordinate.pattern_state as usize {
                invariant = false;
            }
        })?;
        if !saw_target || !invariant {
            return None;
        }
        let mut starts = BitSet::new(self.pattern.num_states());
        starts.set(coordinate.pattern_state as usize);
        if !self.range_reaches_suffix(starts, completed) {
            return Some(0);
        }
        Some(
            u32::try_from(self.max.saturating_sub(completed))
                .unwrap_or(u32::MAX)
                .min(max_repetitions),
        )
    }

    /// Fast exact bounded-repeat proof for a slice whose first accepting
    /// boundary is exactly one body code word. Unlike
    /// `invariant_body_repeat_radius`, the pattern state may advance after each
    /// slice atom (for example through a bounded-length counter). The proof only
    /// succeeds when *every* word in one slice atom converges to the same next
    /// pattern state, so chaining that transition preserves universal
    /// containment without exploring the full byte-level oracle product.
    #[allow(clippy::too_many_arguments)]
    fn uniform_slice_repeat_radius(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        atom_is_exact_body_code: bool,
        slice_atom_fingerprint: u64,
        pattern_targets_cache: &mut FxHashMap<(u64, u32), Option<BitSet>>,
        body_boundary_future_by_completed: Option<&[BitSet]>,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
        max_repetitions: u32,
    ) -> Option<u32> {
        let BoundedCodeEnvelopeState::Body {
            completed,
            body_state: 0,
        } = coordinate.envelope
        else {
            return None;
        };
        if completed > self.max {
            return Some(0);
        }
        if !atom_is_exact_body_code {
            return None;
        }
        let slice_state_count = slice_accepting.len();

        let slice_targets = |pattern_start: u32| -> Option<BitSet> {
            let mut seen = FxHashSet::<(u32, u32, u32)>::default();
            let mut queue = VecDeque::from([(slice_start, pattern_start, 0u32)]);
            seen.insert((slice_start, pattern_start, 0));
            let mut final_targets = BitSet::new(self.pattern.num_states());
            while let Some((slice_state, pattern_state, body_state)) = queue.pop_front() {
                let row = (slice_state as usize).checked_mul(slice_class_count)?;
                for byte in 0u16..=255 {
                    let byte = byte as u8;
                    let class = slice_byte_to_class[byte as usize] as usize;
                    let slice_target = *slice_transitions.get(row + class)?;
                    if slice_target as usize >= slice_state_count
                        || !slice_can_reach_accepting[slice_target as usize]
                    {
                        continue;
                    }
                    let body_target = self.body.step(body_state, byte)?;
                    let body_accepting = !self.body.finalizers(body_target).is_empty();
                    let pattern_target = self.pattern.step(pattern_state, byte)?;
                    if slice_accepting[slice_target as usize] {
                        if !body_accepting {
                            return None;
                        }
                        final_targets.set(pattern_target as usize);
                        // First slice-accepting boundary is one atom. Do not
                        // follow the `slice+` DFA into a second atom here.
                        continue;
                    }
                    if body_accepting || !self.body_productive[body_target as usize] {
                        return None;
                    }
                    let key = (slice_target, pattern_target, body_target);
                    if seen.insert(key) {
                        queue.push_back(key);
                    }
                }
            }
            (!final_targets.is_empty()).then_some(final_targets)
        };

        let max_radius = u32::try_from(self.max.saturating_sub(completed))
            .unwrap_or(u32::MAX)
            .min(max_repetitions);
        let mut radius = 0u32;
        let mut pattern_states = BitSet::new(self.pattern.num_states());
        pattern_states.set(coordinate.pattern_state as usize);
        while radius < max_radius {
            let mut next_states = BitSet::new(self.pattern.num_states());
            let mut complete = true;
            for pattern_state in pattern_states.iter_ones() {
                let cached = pattern_targets_cache
                    .entry((slice_atom_fingerprint, pattern_state as u32))
                    .or_insert_with(|| slice_targets(pattern_state as u32));
                let Some(targets) = cached.as_ref() else {
                    complete = false;
                    break;
                };
                next_states.union_with(targets);
            }
            if !complete || next_states.is_empty() {
                break;
            }
            let completed_after = completed.saturating_add(radius as usize + 1);
            let all_have_future = body_boundary_future_by_completed
                .and_then(|future| future.get(completed_after))
                .map_or_else(
                    || {
                        next_states.iter_ones().all(|pattern_state| {
                            let mut starts = BitSet::new(self.pattern.num_states());
                            starts.set(pattern_state);
                            self.range_reaches_suffix(starts, completed_after)
                        })
                    },
                    |future| next_states.is_subset(future),
                );
            if !all_have_future {
                break;
            }
            pattern_states = next_states;
            radius += 1;
        }
        Some(radius)
    }

    #[inline]
    fn step_coordinate_class(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        byte: u8,
        class: usize,
        max: usize,
        transitions: &FiniteMaskClassTransitions,
    ) -> Option<BoundedCodeOracleCoordinate> {
        let pattern_state = transitions.pattern_step(coordinate.pattern_state, class)?;
        let envelope = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if self.prefix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.prefix.len() {
                    BoundedCodeEnvelopeState::Body {
                        completed: 0,
                        body_state: 0,
                    }
                } else {
                    BoundedCodeEnvelopeState::Prefix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                if body_state == 0 && completed >= self.min && self.suffix[0] == byte {
                    if self.suffix.len() == 1 {
                        BoundedCodeEnvelopeState::Done
                    } else {
                        BoundedCodeEnvelopeState::Suffix { next: 1 }
                    }
                } else {
                    if completed >= max {
                        return None;
                    }
                    let target = transitions.body_step(&self.body, body_state, byte, class)?;
                    if !self.body.finalizers(target).is_empty() {
                        BoundedCodeEnvelopeState::Body {
                            completed: completed.checked_add(1)?,
                            body_state: 0,
                        }
                    } else if self.body_productive[target as usize] {
                        BoundedCodeEnvelopeState::Body {
                            completed,
                            body_state: target,
                        }
                    } else {
                        return None;
                    }
                }
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if self.suffix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.suffix.len() {
                    BoundedCodeEnvelopeState::Done
                } else {
                    BoundedCodeEnvelopeState::Suffix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Done => return None,
        };
        Some(BoundedCodeOracleCoordinate {
            pattern_state,
            envelope,
        })

    }

    fn has_future(&mut self, coordinate: BoundedCodeOracleCoordinate) -> bool {
        match coordinate.envelope {
            BoundedCodeEnvelopeState::Done => false,
            BoundedCodeEnvelopeState::Prefix { next } => {
                let Some(pattern_state) =
                    step_fixed_bytes(&self.pattern, coordinate.pattern_state, &self.prefix[next..])
                else {
                    return false;
                };
                let mut starts = BitSet::new(self.pattern.num_states());
                starts.set(pattern_state as usize);
                self.range_reaches_suffix(starts, 0)
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } if body_state == 0 => {
                let mut starts = BitSet::new(self.pattern.num_states());
                starts.set(coordinate.pattern_state as usize);
                self.range_reaches_suffix(starts, completed)
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                if completed >= self.max {
                    return false;
                }
                let mut starts = BitSet::new(self.pattern.num_states());
                starts.set(coordinate.pattern_state as usize);
                let after_current = self.completion_row(body_state, coordinate.pattern_state);
                if after_current.is_empty() {
                    return false;
                }
                self.range_reaches_suffix(after_current, completed + 1)
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                step_fixed_bytes(&self.pattern, coordinate.pattern_state, &self.suffix[next..])
                    .is_some_and(|state| !self.pattern.finalizers(state).is_empty())
            }
        }
    }

    /// Exact future query using the build-time backwards-DP table instead of
    /// relation powers.  The table contains, for every completed body count,
    /// exactly the pattern states from which some legal continuation reaches
    /// the fixed suffix.  A partial body word is completed once, then tested
    /// against the next boundary row.
    fn has_future_with_boundary_table(
        &mut self,
        coordinate: BoundedCodeOracleCoordinate,
        future: &[BitSet],
    ) -> Option<bool> {
        if future.len() != self.max.checked_add(1)?
            || future.iter().any(|row| row.len() != self.pattern.num_states())
        {
            return None;
        }
        let boundary_live = |completed: usize, pattern_state: u32| {
            future
                .get(completed)
                .is_some_and(|row| row.contains(pattern_state as usize))
        };
        Some(match coordinate.envelope {
            BoundedCodeEnvelopeState::Done => false,
            BoundedCodeEnvelopeState::Prefix { next } => {
                let Some(pattern_state) =
                    step_fixed_bytes(&self.pattern, coordinate.pattern_state, &self.prefix[next..])
                else {
                    return Some(false);
                };
                boundary_live(0, pattern_state)
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state: 0,
            } => boundary_live(completed, coordinate.pattern_state),
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                if completed >= self.max {
                    false
                } else {
                    let after_current = self.completion_row(body_state, coordinate.pattern_state);
                    let Some(boundary) = future.get(completed + 1) else {
                        return Some(false);
                    };
                    !after_current.is_disjoint(boundary)
                }
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                step_fixed_bytes(&self.pattern, coordinate.pattern_state, &self.suffix[next..])
                    .is_some_and(|state| !self.pattern.finalizers(state).is_empty())
            }
        })
    }

    /// Slice proofs also run on compact loaded oracles. Those intentionally
    /// omit relation powers because the transferred backwards-DP table is an
    /// exact replacement. Do not accidentally route such a proof through the
    /// missing powers; unavailable proof data means decline the accelerator,
    /// never interpret an unsupported query as a dead language.
    fn has_future_for_slice_proof(
        &mut self,
        coordinate: BoundedCodeOracleCoordinate,
        future: Option<&[BitSet]>,
    ) -> Option<bool> {
        if let Some(future) = future {
            return self.has_future_with_boundary_table(coordinate, future);
        }
        let count = self.max.checked_add(1)?;
        let bits = (usize::BITS - count.leading_zeros()) as usize;
        if self.exact_powers.len() < bits || self.prefix_sums.len() < bits {
            return None;
        }
        Some(self.has_future(coordinate))
    }
}

/// Build only the finite one-token observation component for a certified
/// bounded-code expression. Unlike `VirtualResidualRuntime`, this does not
/// allocate or retain any exact/runtime residual state and does not construct
/// an exact->mask projection: callers that only need the mask coordinate (for
/// example vocabulary-equivalence analysis) can compile that coordinate
/// directly.
pub(crate) fn build_bounded_code_mask_component_for_vocab(
    expr: &Expr,
    vocab: &Vocab,
    max_token_len: usize,
    repeat_horizons: &VocabularyRepeatHorizonCache,
) -> Option<(DFA, u32)> {
    prepare_bounded_code_mask_component(expr)?.finish_for_vocab(
        vocab,
        max_token_len,
        repeat_horizons,
    )
}

pub(crate) struct PreparedBoundedCodeMaskComponent {
    oracle: BoundedCodeIntersectionOracle,
}

impl PreparedBoundedCodeMaskComponent {
    pub(crate) fn finish_for_vocab(
        self,
        vocab: &Vocab,
        max_token_len: usize,
        repeat_horizons: &VocabularyRepeatHorizonCache,
    ) -> Option<(DFA, u32)> {
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let total_started = std::time::Instant::now();
        let oracle = self.oracle;
        let horizon_started = std::time::Instant::now();
        let crossed_boundaries = repeat_horizons
            .horizon_for_dfa(oracle.body.as_ref(), vocab)
            .unwrap_or_else(|| {
                let minimum_body_width = oracle.body.min_match_byte_len().unwrap_or(1).max(1);
                max_token_len
                    .div_ceil(minimum_body_width)
                    .saturating_add(1)
            });
        let horizon_ms = horizon_started.elapsed().as_secs_f64() * 1000.0;
        Self::finish_with_crossed_boundaries(oracle, crossed_boundaries, horizon_ms, total_started)
    }

    /// Finish the finite one-token component using the byte-length upper bound
    /// rather than scanning the complete vocabulary for a tighter repeat
    /// horizon. This is exact: it is the same conservative fallback used by
    /// `finish_for_vocab` when the exact horizon proof is unavailable. It is
    /// useful for vocabulary-partition compilation, where a slightly larger
    /// finite observation DFA is cheaper than an additional full-vocabulary
    /// scan on the critical path.
    pub(crate) fn finish_for_vocab_conservative(
        self,
        max_token_len: usize,
    ) -> Option<(DFA, u32)> {
        let total_started = std::time::Instant::now();
        let oracle = self.oracle;
        let minimum_body_width = oracle.body.min_match_byte_len().unwrap_or(1).max(1);
        let crossed_boundaries = max_token_len
            .div_ceil(minimum_body_width)
            .saturating_add(1);
        Self::finish_with_crossed_boundaries(oracle, crossed_boundaries, 0.0, total_started)
    }

    fn finish_with_crossed_boundaries(
        oracle: BoundedCodeIntersectionOracle,
        crossed_boundaries: usize,
        horizon_ms: f64,
        total_started: std::time::Instant,
    ) -> Option<(DFA, u32)> {
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let crossed_boundaries = oracle.finite_mask_stencil_crossings(crossed_boundaries)?;
        // Preserve the complete lower-bound prefix when it is still small enough
        // for the finite mask coordinate. `finite_mask_dfa` enforces the global
        // dense-state cap, so a moderate minLength need not fall back to the
        // full declared-max DFA merely because one token cannot cross the lower
        // bound from the language root.
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)?
            .checked_add(1)?;
        let mask_max = oracle.max.min(desired_mask_max);
        let finite_started = std::time::Instant::now();
        let (mut dfa, segment, root, _dense_to_mask) = oracle.finite_mask_dfa(mask_max)?;
        // `finite_mask_dfa` keeps its minimized transition graph in a compressed
        // sidecar and clears the ordinary DFA rows. Runtime residual projections
        // retain that sidecar, but this API deliberately returns a standalone
        // DFA for vocabulary-equivalence compilation. Materialize the sidecar
        // back into the DFA here; dropping it would leave a state-only shell
        // with no byte transitions and make terminal classification incorrectly
        // treat the bounded language as unreachable.
        debug_assert_eq!(segment.state_offset, 0);
        for source in 0..segment.state_count {
            let row_start = segment.row_offsets[source as usize] as usize;
            let row_end = segment.row_offsets[source as usize + 1] as usize;
            for (class, target) in segment.entries.iter_range(row_start, row_end) {
                for &byte in segment.class_members[class as usize].iter() {
                    dfa.add_transition(source, byte, target);
                }
            }
        }
        debug_assert_eq!(dfa.transition_count(), segment.expanded_transition_count);
        let finite_ms = finite_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][bounded_code_mask_component] oracle_ms=0.000 horizon_ms={horizon_ms:.3} finite_ms={finite_ms:.3} body_states={} pattern_states={} min={} max={} mask_max={} total_ms={:.3}",
                oracle.body.num_states(),
                oracle.pattern.num_states(),
                oracle.min,
                oracle.max,
                mask_max,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some((dfa, root))
    }
}

pub(crate) fn prepare_bounded_code_mask_component(
    expr: &Expr,
) -> Option<PreparedBoundedCodeMaskComponent> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let oracle_started = std::time::Instant::now();
    let oracle = BoundedCodeIntersectionOracle::from_expr(expr)?;
    let oracle_ms = oracle_started.elapsed().as_secs_f64() * 1000.0;
    if profile {
        eprintln!(
            "[glrmask/profile][bounded_code_mask_prepare] oracle_ms={oracle_ms:.3} body_states={} pattern_states={} min={} max={}",
            oracle.body.num_states(),
            oracle.pattern.num_states(),
            oracle.min,
            oracle.max,
        );
    }
    Some(PreparedBoundedCodeMaskComponent { oracle })
}


#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualResidualMaskProjection {
    runtime: Arc<VirtualResidualRuntime>,
    state_offset: u32,
    pattern_states: usize,
    body_states: usize,
    prefix_len: usize,
    suffix_len: usize,
    min: usize,
    full_max: usize,
    mask_max: usize,
    crossed_boundaries: usize,
    local_to_mask_state: Arc<[u32]>,
}

/// Serialized half of a [`VirtualResidualMaskProjection`].
///
/// The exact residual runtime is already reconstructed from the terminal
/// expression + virtual-runtime metadata. Persist only the compiled finite
/// observation transport and reattach it to that exact runtime after load.
#[derive(Debug, Clone, serde::Deserialize)]
#[doc(hidden)]
pub struct VirtualResidualMaskProjectionArtifact {
    terminal: TerminalID,
    state_offset: u32,
    local_to_mask_state: Vec<u32>,
    oracle_bytes: Vec<u8>,
    #[serde(skip)]
    runtime_expr_bytes: Vec<u8>,
    #[serde(skip)]
    compiled_mask_max: usize,
    #[serde(skip)]
    compiled_crossed_boundaries: usize,
    #[serde(skip)]
    validated_component_state_count: Option<u32>,
}

#[derive(serde::Serialize)]
#[doc(hidden)]
pub struct VirtualResidualMaskProjectionArtifactRef<'a> {
    terminal: TerminalID,
    state_offset: u32,
    local_to_mask_state: &'a [u32],
    oracle_bytes: Vec<u8>,
}

/// Compact immutable preparation carried across DynamicConstraint transfer so
/// loading an already-built constraint does not rebuild the bounded-code
/// backward-future table. This is a deterministic derivative of the residual
/// oracle, not runtime history from a mask call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[doc(hidden)]
pub(crate) struct CompactResidualDfaArtifact {
    state_count: u32,
    transition_offsets: Vec<u32>,
    transition_bytes: Vec<u8>,
    transition_targets: Vec<u16>,
    accepting_words: Vec<u64>,
    future_words: Vec<u64>,
    group_bytes: U8Set,
}

impl CompactResidualDfaArtifact {
    fn from_dfa(dfa: &DFA) -> Option<Self> {
        if dfa.has_epsilon_transitions() || dfa.num_states() > u16::MAX as usize {
            return None;
        }
        let state_count = dfa.num_states();
        let word_count = state_count.div_ceil(64);
        let mut transition_offsets = Vec::with_capacity(state_count + 1);
        let mut transition_bytes = Vec::new();
        let mut transition_targets = Vec::new();
        let mut accepting_words = vec![0u64; word_count];
        let mut future_words = vec![0u64; word_count];
        let mut group_bytes = U8Set::empty();
        for group in 0..dfa.num_groups() {
            group_bytes |= *dfa.group_id_to_u8set(group as u32);
        }
        for state in 0..state_count {
            transition_offsets.push(u32::try_from(transition_bytes.len()).ok()?);
            if !dfa.finalizers(state as u32).is_empty() {
                accepting_words[state / 64] |= 1u64 << (state % 64);
            }
            if !dfa.possible_future_group_ids(state as u32).is_empty() {
                future_words[state / 64] |= 1u64 << (state % 64);
            }
            for (byte, target) in dfa.transitions(state as u32) {
                transition_bytes.push(byte);
                transition_targets.push(u16::try_from(target).ok()?);
            }
        }
        transition_offsets.push(u32::try_from(transition_bytes.len()).ok()?);
        Some(Self {
            state_count: u32::try_from(state_count).ok()?,
            transition_offsets,
            transition_bytes,
            transition_targets,
            accepting_words,
            future_words,
            group_bytes,
        })
    }

    fn into_dfa(self) -> Option<DFA> {
        let state_count = self.state_count as usize;
        DFA::new_from_compact_single_group(
            state_count,
            &self.transition_offsets,
            &self.transition_bytes,
            &self.transition_targets,
            &self.accepting_words,
            &self.future_words,
            self.group_bytes,
        )
    }

    #[inline]
    fn state_count(&self) -> usize {
        self.state_count as usize
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[doc(hidden)]
pub struct VirtualResidualMasterSliceArtifact {
    pub(crate) terminal: TerminalID,
    pub(crate) root_state: u32,
    /// Already-compiled bounded-code liveness program.  Keeping these DFAs in
    /// the build artifact avoids recompiling the pattern language during load;
    /// `words` below replaces the much larger relation-power tables for exact
    /// future queries at whole-body boundaries.
    pub(crate) pattern: CompactResidualDfaArtifact,
    pub(crate) body: CompactResidualDfaArtifact,
    pub(crate) body_productive: Vec<bool>,
    pub(crate) prefix: Vec<u8>,
    pub(crate) suffix: Vec<u8>,
    pub(crate) min: usize,
    pub(crate) max: usize,
    pub(crate) suffix_accepting: BitSet,
    pub(crate) pattern_states: u32,
    pub(crate) rows: u32,
    pub(crate) words: Vec<u64>,
    #[serde(default)]
    pub(crate) body_exact: Vec<(u64, bool)>,
    #[serde(default)]
    pub(crate) pattern_targets: Vec<(u64, u32, Option<Vec<u64>>)>,
}

impl VirtualResidualMaskProjectionArtifact {
    #[doc(hidden)]
    pub fn state_offset(&self) -> u32 {
        self.state_offset
    }

    #[doc(hidden)]
    pub fn terminal(&self) -> TerminalID {
        self.terminal
    }

    #[doc(hidden)]
    pub fn oracle_bytes(&self) -> &[u8] {
        &self.oracle_bytes
    }

    #[doc(hidden)]
    pub fn runtime_expr_bytes(&self) -> &[u8] {
        &self.runtime_expr_bytes
    }

    #[doc(hidden)]
    pub fn compiled_mask_max(&self) -> usize { self.compiled_mask_max }

    #[doc(hidden)]
    pub fn compiled_crossed_boundaries(&self) -> usize { self.compiled_crossed_boundaries }

    #[doc(hidden)]
    pub fn from_wire(
        terminal: TerminalID,
        state_offset: u32,
        local_to_mask_state: Vec<u32>,
        oracle_bytes: Vec<u8>,
        runtime_expr_bytes: Vec<u8>,
        compiled_mask_max: usize,
        compiled_crossed_boundaries: usize,
    ) -> Self {
        Self {
            terminal, state_offset, local_to_mask_state, oracle_bytes, runtime_expr_bytes,
            compiled_mask_max, compiled_crossed_boundaries,
            validated_component_state_count: None,
        }
    }

    #[doc(hidden)]
    pub fn from_validated_wire(
        terminal: TerminalID,
        state_offset: u32,
        local_to_mask_state: Vec<u32>,
        oracle_bytes: Vec<u8>,
        runtime_expr_bytes: Vec<u8>,
        compiled_mask_max: usize,
        compiled_crossed_boundaries: usize,
        component_state_count: u32,
    ) -> Self {
        Self {
            terminal, state_offset, local_to_mask_state, oracle_bytes, runtime_expr_bytes,
            compiled_mask_max, compiled_crossed_boundaries,
            validated_component_state_count: Some(component_state_count),
        }
    }
}

impl VirtualResidualMaskProjection {
    pub(super) fn clone_for_equivalent_runtime(
        &self,
        runtime: Arc<VirtualResidualRuntime>,
    ) -> Option<Self> {
        if self.runtime.finite_mask_projection_language_key()?
            != runtime.finite_mask_projection_language_key()?
        {
            return None;
        }
        Some(Self {
            runtime,
            state_offset: self.state_offset,
            pattern_states: self.pattern_states,
            body_states: self.body_states,
            prefix_len: self.prefix_len,
            suffix_len: self.suffix_len,
            min: self.min,
            full_max: self.full_max,
            mask_max: self.mask_max,
            crossed_boundaries: self.crossed_boundaries,
            local_to_mask_state: Arc::clone(&self.local_to_mask_state),
        })
    }

    #[doc(hidden)]
    pub fn artifact_ref(&self) -> VirtualResidualMaskProjectionArtifactRef<'_> {
        VirtualResidualMaskProjectionArtifactRef {
            terminal: self.runtime.terminal(),
            state_offset: self.state_offset,
            local_to_mask_state: self.local_to_mask_state.as_ref(),
            oracle_bytes: self.runtime.serialized_bounded_code_oracle(),
        }
    }

    /// Canonical transport parameters for a checked finite observation view.
    /// Mutable reached-state stores and pointer identities are not serialized.
    #[doc(hidden)]
    pub fn observation_descriptor(&self) -> Vec<u8> {
        bincode::serialize(&(self.runtime.terminal(), self.state_offset,
            self.pattern_states, self.body_states, self.prefix_len, self.suffix_len,
            self.min, self.full_max, self.mask_max, self.crossed_boundaries,
            self.local_to_mask_state.as_ref()))
            .expect("finite residual descriptor serialization")
    }

    #[doc(hidden)]
    pub fn artifact_wire_parts(&self) -> (TerminalID, u32, &[u32], Vec<u8>, usize, usize) {
        (
            self.runtime.terminal(),
            self.state_offset,
            self.local_to_mask_state.as_ref(),
            self.runtime.serialized_bounded_code_oracle(),
            self.mask_max,
            self.crossed_boundaries,
        )
    }

    fn local_state_for_coordinate(&self, coordinate: BoundedCodeOracleCoordinate) -> Option<u32> {
        let pattern = coordinate.pattern_state as usize;
        if pattern >= self.pattern_states {
            return None;
        }
        let prefix_block = self.prefix_len.checked_mul(self.pattern_states)?;
        let body_layer = self.body_states.checked_mul(self.pattern_states)?;
        let body_block = self.mask_max.checked_add(1)?.checked_mul(body_layer)?;
        let suffix_slots = self.suffix_len.saturating_sub(1);
        let suffix_block = suffix_slots.checked_mul(self.pattern_states)?;
        let local = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if next >= self.prefix_len { return None; }
                next.checked_mul(self.pattern_states)?.checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Body { completed, body_state } => {
                let body_state = body_state as usize;
                if body_state >= self.body_states || completed > self.full_max {
                    return None;
                }
                let distance_to_upper = self.full_max - completed;
                let mapped_completed = if completed < self.min {
                    completed
                } else if distance_to_upper <= self.crossed_boundaries {
                    self.mask_max.checked_sub(distance_to_upper)?
                } else {
                    self.min
                };
                if mapped_completed > self.mask_max { return None; }
                prefix_block
                    .checked_add(mapped_completed.checked_mul(body_layer)?)?
                    .checked_add(body_state.checked_mul(self.pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if next == 0 || next >= self.suffix_len { return None; }
                prefix_block
                    .checked_add(body_block)?
                    .checked_add((next - 1).checked_mul(self.pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Done => prefix_block
                .checked_add(body_block)?
                .checked_add(suffix_block)?
                .checked_add(pattern)?,
        };
        u32::try_from(local).ok()
    }

    #[inline]
    pub fn project(&self, full_state: u32) -> Option<u32> {
        if !self.runtime.handles_state(full_state) {
            return None;
        }
        let coordinate = self.runtime.oracle_coordinate(full_state)?;
        let local = self.local_state_for_coordinate(coordinate)?;
        let mapped = *self.local_to_mask_state.get(local as usize)?;
        if mapped == u32::MAX {
            return None;
        }
        mapped.checked_add(self.state_offset)
    }

    pub(super) fn set_state_offset(&mut self, state_offset: u32) {
        self.state_offset = state_offset;
    }

    pub fn physical_state_count(&self) -> u32 {
        self.runtime.physical_state_count()
    }
}


fn dfa_language_is_finite(dfa: &DFA) -> bool {
    let n = dfa.num_states();
    if n == 0 {
        return true;
    }
    let live = |state: u32| {
        !dfa.finalizers(state).is_empty() || !dfa.possible_future_group_ids(state).is_empty()
    };
    if !live(0) {
        return true;
    }
    let mut reachable = vec![false; n];
    let mut stack = vec![0u32];
    reachable[0] = true;
    while let Some(state) = stack.pop() {
        for (_, &target) in dfa.states()[state as usize].transitions.iter() {
            if live(target) && !reachable[target as usize] {
                reachable[target as usize] = true;
                stack.push(target);
            }
        }
    }
    let mut indegree = vec![0u32; n];
    let mut live_count = 0usize;
    for state in 0..n as u32 {
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
    for state in 0..n as u32 {
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

impl BoundedCodeIntersectionOracle {
    fn byte_to_class_map(&self) -> Box<[u8; 256]> {
        let classes = bounded_code_byte_classes(
            &self.pattern,
            &self.body,
            &self.prefix,
            &self.suffix,
        );
        let mut map = [0u8; 256];
        for (class, members) in classes.iter().enumerate() {
            for &byte in members {
                map[byte as usize] = class as u8;
            }
        }
        Box::new(map)
    }

}


#[derive(Debug, Clone, Copy)]
struct FiniteMaskLayout {
    pattern_states: usize,
    body_states: usize,
    prefix_len: usize,
    suffix_len: usize,
    mask_max: usize,
    prefix_block: usize,
    body_layer: usize,
    body_block: usize,
    suffix_block: usize,
    dense_state_count: usize,
}

impl FiniteMaskLayout {
    fn new(oracle: &BoundedCodeIntersectionOracle, mask_max: usize) -> Option<Self> {
        if mask_max < oracle.min || mask_max > oracle.max {

            return None;
        }
        let pattern_states = oracle.pattern.num_states();
        let body_states = oracle.body.num_states();
        let prefix_len = oracle.prefix.len();
        let suffix_len = oracle.suffix.len();
        let prefix_block = prefix_len.checked_mul(pattern_states)?;
        let body_layer = body_states.checked_mul(pattern_states)?;
        let body_block = mask_max.checked_add(1)?.checked_mul(body_layer)?;
        let suffix_block = suffix_len
            .saturating_sub(1)
            .checked_mul(pattern_states)?;
        let dense_state_count = prefix_block
            .checked_add(body_block)?
            .checked_add(suffix_block)?
            .checked_add(pattern_states)?;
        Some(Self {
            pattern_states,
            body_states,
            prefix_len,
            suffix_len,
            mask_max,
            prefix_block,
            body_layer,
            body_block,
            suffix_block,
            dense_state_count,
        })
    }

    #[inline]
    fn coordinate_local_state(&self, coordinate: BoundedCodeOracleCoordinate) -> Option<u32> {
        let pattern = coordinate.pattern_state as usize;
        if pattern >= self.pattern_states {
            return None;
        }
        // The constructor checked every block product and total. The bounds
        // below keep each offset inside those prechecked blocks, so the hot
        // transition path can use direct arithmetic without changing overflow
        // or invalid-coordinate behavior.
        let local = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if next >= self.prefix_len {
                    return None;
                }
                next * self.pattern_states + pattern
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                let body_state = body_state as usize;
                if completed > self.mask_max || body_state >= self.body_states {
                    return None;
                }
                self.prefix_block
                    + completed * self.body_layer
                    + body_state * self.pattern_states
                    + pattern
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if next == 0 || next >= self.suffix_len {
                    return None;
                }
                self.prefix_block
                    + self.body_block
                    + (next - 1) * self.pattern_states
                    + pattern
            }
            BoundedCodeEnvelopeState::Done => {
                self.prefix_block + self.body_block + self.suffix_block + pattern
            }
        };
        u32::try_from(local).ok()
    }

}

#[derive(Debug, Clone)]
struct FiniteMaskClassTransitions {
    class_count: usize,
    pattern: Box<[u32]>,
    body: Option<Box<[u32]>>,
}

impl FiniteMaskClassTransitions {
    fn new(oracle: &BoundedCodeIntersectionOracle, byte_classes: &[Vec<u8>]) -> Option<Self> {
        let class_count = byte_classes.len();
        if class_count == 0 {
            return None;
        }
        let mut byte_to_class = [0u8; 256];
        for (class, members) in byte_classes.iter().enumerate() {
            let class = u8::try_from(class).ok()?;
            for &byte in members {
                byte_to_class[byte as usize] = class;
            }
        }

        let build_table = |dfa: &DFA| -> Option<Box<[u32]>> {
            let cells = dfa.num_states().checked_mul(class_count)?;
            let mut table = vec![u32::MAX; cells];
            for (state, row) in dfa.states().iter().zip(table.chunks_mut(class_count)) {
                for (byte, &target) in state.transitions.iter() {
                    let slot = &mut row[byte_to_class[byte as usize] as usize];
                    debug_assert!(*slot == u32::MAX || *slot == target);
                    *slot = target;
                }
            }
            Some(table.into_boxed_slice())
        };

        // Pattern DFAs are capped at 4096 states by oracle construction, so
        // their class table is at most 4096 * 256 u32s (4 MiB). A body DFA can
        // be much wider when the pattern is tiny; cap that optional accelerator
        // at the existing finite-mask dense-state budget and fall back to the
        // ordinary sparse transition lookup if it would exceed it.
        let pattern = build_table(&oracle.pattern)?;
        let body_cells = oracle.body.num_states().checked_mul(class_count)?;
        let body = if body_cells <= MAX_FINITE_MASK_DENSE_STATES {
            build_table(&oracle.body)
        } else {
            None
        };
        Some(Self {
            class_count,
            pattern,
            body,
        })
    }

    #[inline]
    fn pattern_step(&self, state: u32, class: usize) -> Option<u32> {
        debug_assert!(class < self.class_count);
        let target = self.pattern[state as usize * self.class_count + class];
        (target != u32::MAX).then_some(target)
    }

    #[inline]
    fn body_step(&self, dfa: &DFA, state: u32, byte: u8, class: usize) -> Option<u32> {
        debug_assert!(class < self.class_count);
        if let Some(table) = &self.body {
            let target = table[state as usize * self.class_count + class];
            (target != u32::MAX).then_some(target)
        } else {
            dfa.step(state, byte)
        }
    }
}

/// Count-independent part of one exact pattern/body byte-class transition.
/// A suffix candidate and a body candidate may coexist; the original suffix
/// priority and min/max checks are applied at every concrete counter layer.
#[derive(Clone, Copy, Debug)]
struct FiniteMaskBodyTransition {
    class: u8,
    pattern_state: u32,
    body_state: u32,
    completes_body: bool,
    starts_suffix: bool,
}

impl FiniteMaskBodyTransition {
    #[inline]
    fn next_coordinate(
        &self,
        completed: usize,
        min: usize,
        max: usize,
        suffix_len: usize,
    ) -> Option<BoundedCodeOracleCoordinate> {
        let envelope = if self.starts_suffix && completed >= min {
            if suffix_len == 1 {
                BoundedCodeEnvelopeState::Done
            } else {
                BoundedCodeEnvelopeState::Suffix { next: 1 }
            }
        } else {
            if completed >= max || self.body_state == u32::MAX {
                return None;
            }
            BoundedCodeEnvelopeState::Body {
                completed: completed.checked_add(usize::from(self.completes_body))?,
                body_state: self.body_state,
            }
        };
        Some(BoundedCodeOracleCoordinate { pattern_state: self.pattern_state, envelope })
    }
}

impl BoundedCodeIntersectionOracle {
    fn finite_mask_body_row(
        &self,
        pattern_state: u32,
        body_state: u32,
        byte_classes: &[Vec<u8>],
        transitions: &FiniteMaskClassTransitions,
    ) -> Vec<FiniteMaskBodyTransition> {
        let mut row = Vec::new();
        for (class, members) in byte_classes.iter().enumerate() {
            let byte = members[0];
            let Some(next_pattern) = transitions.pattern_step(pattern_state, class) else {
                continue;
            };
            let starts_suffix = body_state == 0 && self.suffix[0] == byte;
            let body_target = transitions.body_step(&self.body, body_state, byte, class);
            let completes_body = body_target.is_some_and(|target| !self.body.finalizers(target).is_empty());
            let next_body = if completes_body {
                0
            } else {
                body_target.filter(|&target| self.body_productive[target as usize]).unwrap_or(u32::MAX)
            };
            if starts_suffix || next_body != u32::MAX {
                row.push(FiniteMaskBodyTransition {
                    class: class as u8,
                    pattern_state: next_pattern,
                    body_state: next_body,
                    completes_body,
                    starts_suffix,
                });
            }
        }
        row
    }

    fn finite_mask_dense_state_count(&self, mask_max: usize) -> Option<usize> {
        FiniteMaskLayout::new(self, mask_max).map(|layout| layout.dense_state_count)
    }

    fn finite_mask_dfa(
        &self,
        mask_max: usize,
    ) -> Option<(DFA, CompressedTransitionSegment, u32, Vec<u32>)> {
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let total_started = std::time::Instant::now();
        let pattern_states = self.pattern.num_states();
        let layout = FiniteMaskLayout::new(self, mask_max)?;
        let dense_state_count = layout.dense_state_count;
        if dense_state_count == 0 || dense_state_count > MAX_FINITE_MASK_DENSE_STATES {
            return None;
        }


        let byte_classes = bounded_code_byte_classes(
            &self.pattern,
            &self.body,
            &self.prefix,
            &self.suffix,
        );

        // The finite envelope deliberately identifies the huge exact count
        // interval with a small one-token stencil. A state that is a valid
        // projection source after a long exact input need not be reachable from
        // the finite root at the corresponding small count, because the format
        // DFA can be in a state only reachable after many code words. Seed the
        // finite graph with every *actually possible* projected code-word
        // boundary, then retain the transition closure of those roots. This
        // avoids the full count x body x pattern Cartesian product while still
        // covering every exact token-boundary state.
        // `mask_max` can equal `min` when the declared interval is fixed
        // (or otherwise narrower than the one-token stencil). In that case
        // there is no collapsed interior layer, but the exact lower/upper
        // boundary is still a perfectly valid finite projection root.
        let crossed_boundaries = mask_max.saturating_sub(self.min).saturating_sub(1);
        let after_prefix = step_fixed_bytes(&self.pattern, 0, &self.prefix)?;
        let mut pattern_start = BitSet::new(pattern_states);
        pattern_start.set(after_prefix as usize);
        let mut boundary_classes = Vec::<(usize, BitSet)>::new();

        // Counts below min are observed exactly.
        for completed in 0..self.min {
            let states = self.apply_exact_count(pattern_start.clone(), completed);
            if !states.is_empty() {
                boundary_classes.push((completed, states));
            }
        }

        // The deep interior collapses to the first accepting count. Compute the
        // exact union of format states reachable at any full count represented
        // by that interior layer using the existing relation-doubling oracle.
        let interior_high = self
            .max
            .checked_sub(crossed_boundaries.saturating_add(1));
        if let Some(interior_high) = interior_high.filter(|&high| high >= self.min) {
            let after_low = self.apply_exact_count(pattern_start.clone(), self.min);
            if !after_low.is_empty() {
                let states = self.apply_up_to(after_low, interior_high - self.min);
                if !states.is_empty() {
                    boundary_classes.push((self.min, states));
                }
            }
        }

        // Near the true upper bound, preserve distance-to-upper exactly.
        let upper_start = self.max.saturating_sub(crossed_boundaries).max(self.min);
        for completed in upper_start..=self.max {
            let distance_to_upper = self.max - completed;
            let mapped_completed = mask_max.checked_sub(distance_to_upper)?;
            let states = self.apply_exact_count(pattern_start.clone(), completed);
            if !states.is_empty() {
                boundary_classes.push((mapped_completed, states));
            }
        }

        let seeds_started = std::time::Instant::now();
        let mut dense_to_sparse = vec![u32::MAX; dense_state_count];
        let mut coordinates = Vec::<BoundedCodeOracleCoordinate>::new();
        let mut seed_count = 0usize;
        let mut add_seed = |coordinate: BoundedCodeOracleCoordinate| -> Option<()> {
            let dense = layout.coordinate_local_state(coordinate)? as usize;
            if dense_to_sparse[dense] == u32::MAX {
                dense_to_sparse[dense] = u32::try_from(coordinates.len()).ok()?;
                coordinates.push(coordinate);
                seed_count += 1;
            }
            Some(())
        };
        add_seed(self.root_coordinate())?;
        for (mapped_completed, states) in boundary_classes {
            for pattern_state in states.iter_ones() {
                add_seed(BoundedCodeOracleCoordinate {
                    pattern_state: u32::try_from(pattern_state).ok()?,
                    envelope: BoundedCodeEnvelopeState::Body {
                        completed: mapped_completed,
                        body_state: 0,
                    },
                })?;
            }
        }
        drop(add_seed);
        let seeds_ms = seeds_started.elapsed().as_secs_f64() * 1000.0;

        let expand_started = std::time::Instant::now();
        let class_transitions = FiniteMaskClassTransitions::new(self, &byte_classes)?;
        let mut dfa = DFA::new(coordinates.len());
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, crate::ds::u8set::U8Set::all());
        let mut source = 0usize;
        const EXPANSION_BATCH: usize = 1_024;
        // A collapsed interval or a one-code pattern loop can visit the same
        // pattern/body pair at many exact count layers. Reuse those byte-class
        // rows. Acyclic full-bound patterns retain their original expansion,
        // avoiding cache allocations when code count already fixes the pattern
        // state. Admission affects construction work only, never recognition.
        let repeats_pattern_state = self.exact_powers.first().is_some_and(|one_code| {
            (0..one_code.state_count()).any(|state| {
                let mut self_loop = false;
                let _ = one_code.for_each_target(state, |target| self_loop |= target == state);
                self_loop
            })
        });
        let mut cached_body_pairs = 0;
        if self.max > mask_max || repeats_pattern_state {
            let mut accepting_rows = Vec::<u8>::new();
            // Retain the exact next coordinate (no dense-coordinate decoding), but
            // only store present transitions. Most body-interior states admit very
            // few classes, so clearing/scanning a dense Option matrix is dead work.
            let mut transition_rows = Vec::<Vec<(u8, BoundedCodeOracleCoordinate, usize)>>::new();
            // The layout already bounds body_states * pattern_states inside the
            // finite dense-state budget. Cache only rows actually encountered; the
            // dense index costs four bytes per possible pattern/body pair, not per
            // count layer, and never contains parser state or model-token results.
            let mut body_row_by_pair = vec![u32::MAX; layout.body_layer];
            let mut body_rows = Vec::<Vec<FiniteMaskBodyTransition>>::new();
            while source < coordinates.len() {
                while dfa.num_states() < coordinates.len() {
                    dfa.add_state();
                }
                let batch_end = coordinates.len().min(source.saturating_add(EXPANSION_BATCH));
                let batch_len = batch_end - source;
                accepting_rows.clear();
                accepting_rows.resize(batch_len, 0);
                for coordinate in &coordinates[source..batch_end] {
                    let BoundedCodeEnvelopeState::Body { body_state, .. } = coordinate.envelope else {
                        continue;
                    };
                    let pair = body_state as usize * pattern_states + coordinate.pattern_state as usize;
                    if body_row_by_pair[pair] == u32::MAX {
                        body_row_by_pair[pair] = u32::try_from(body_rows.len()).ok()?;
                        body_rows.push(self.finite_mask_body_row(
                            coordinate.pattern_state, body_state, &byte_classes, &class_transitions,
                        ));
                    }
                }
                if transition_rows.len() < batch_len {
                    transition_rows.resize_with(batch_len, Vec::new);
                }
                transition_rows[..batch_len]
                    .par_iter_mut()
                    .zip(accepting_rows.par_iter_mut())
                    .enumerate()
                    .for_each(|(row_offset, (row, accepting_slot))| {
                        let coordinate = coordinates[source + row_offset];
                        let accepting = matches!(coordinate.envelope, BoundedCodeEnvelopeState::Done)
                            && !self.pattern.finalizers(coordinate.pattern_state).is_empty();
                        *accepting_slot = u8::from(accepting);
                        row.clear();
                        if let BoundedCodeEnvelopeState::Body { completed, body_state } = coordinate.envelope {
                            let pair = body_state as usize * pattern_states + coordinate.pattern_state as usize;
                            for edge in &body_rows[body_row_by_pair[pair] as usize] {
                                let Some(next) = edge.next_coordinate(completed, self.min, mask_max, self.suffix.len()) else {
                                    continue;
                                };
                                let Some(dense) = layout.coordinate_local_state(next) else { continue; };
                                row.push((edge.class, next, dense as usize));
                            }
                        } else {
                            // Prefix, suffix and done coordinates occur only once
                            // per pattern state. Keep their original exact step.
                            for (class, members) in byte_classes.iter().enumerate() {
                                let Some(next) = self.step_coordinate_class(
                                    coordinate, members[0], class, mask_max, &class_transitions,
                                ) else { continue; };
                                let Some(dense) = layout.coordinate_local_state(next) else { continue; };
                                row.push((class as u8, next, dense as usize));
                            }
                        }
                    });

                for row_offset in 0..batch_len {
                    let source_state = (source + row_offset) as u32;
                    let mut finalizers = BitSet::new(1);
                    if accepting_rows[row_offset] != 0 {
                        finalizers.set(0);
                    }
                    dfa.overwrite_state_metadata(source_state, finalizers, BitSet::new(1));
                    // Both cached and ordinary rows retain ascending class order,
                    // so sparse-state discovery is identical to the reference BFS.
                    for &(class, next, dense) in &transition_rows[row_offset] {
                        let target = if dense_to_sparse[dense] == u32::MAX {
                            let target = u32::try_from(coordinates.len()).ok()?;
                            dense_to_sparse[dense] = target;
                            coordinates.push(next);
                            target
                        } else {
                            dense_to_sparse[dense]
                        };
                        while dfa.num_states() <= target as usize {
                            dfa.add_state();
                        }
                        dfa.add_transition(source_state, class, target);
                    }
                }
                source = batch_end;
            }
            cached_body_pairs = body_rows.len();
        } else {
            let class_count = byte_classes.len();
            let slot_width = class_count.max(1);
            let mut accepting_rows = Vec::<u8>::new();
            let mut transition_slots =
                Vec::<Option<(BoundedCodeOracleCoordinate, usize)>>::new();
            while source < coordinates.len() {
                while dfa.num_states() < coordinates.len() {
                    dfa.add_state();
                }
                let batch_end = coordinates.len().min(source.saturating_add(EXPANSION_BATCH));
                let batch_len = batch_end - source;
                accepting_rows.clear();
                accepting_rows.resize(batch_len, 0);
                transition_slots.clear();
                transition_slots.resize(batch_len.saturating_mul(slot_width), None);
                transition_slots
                    .par_chunks_mut(slot_width)
                    .zip(accepting_rows.par_iter_mut())
                    .enumerate()
                    .for_each(|(row_offset, (row, accepting_slot))| {
                        let coordinate = coordinates[source + row_offset];
                        let accepting = matches!(coordinate.envelope, BoundedCodeEnvelopeState::Done)
                            && !self.pattern.finalizers(coordinate.pattern_state).is_empty();
                        *accepting_slot = u8::from(accepting);
                        for (class, members) in byte_classes.iter().enumerate() {
                            let Some(next) = self.step_coordinate_class(
                                coordinate,
                                members[0],
                                class,
                                mask_max,
                                &class_transitions,
                            ) else {
                                continue;
                            };
                            let Some(dense) = layout.coordinate_local_state(next) else {
                                continue;
                            };
                            row[class] = Some((next, dense as usize));
                        }
                    });

                for row_offset in 0..batch_len {
                    let source_state = (source + row_offset) as u32;
                    let mut finalizers = BitSet::new(1);
                    if accepting_rows[row_offset] != 0 {
                        finalizers.set(0);
                    }
                    dfa.overwrite_state_metadata(source_state, finalizers, BitSet::new(1));
                    let row_start = row_offset * slot_width;
                    for (class, slot) in transition_slots[row_start..row_start + class_count]
                        .iter()
                        .enumerate()
                    {
                        let Some((next, dense)) = *slot else {
                            continue;
                        };
                        let target = if dense_to_sparse[dense] == u32::MAX {
                            let target = u32::try_from(coordinates.len()).ok()?;
                            dense_to_sparse[dense] = target;
                            coordinates.push(next);
                            target
                        } else {
                            dense_to_sparse[dense]
                        };
                        while dfa.num_states() <= target as usize {
                            dfa.add_state();
                        }
                        dfa.add_transition(source_state, class as u8, target);
                    }
                }
                source = batch_end;
            }
        }
        // Minimization immediately clears possible-future metadata before
        // partitioning and recomputes it on the minimized DFA. Computing the
        // full sparse DFA's futures here would therefore be pure dead work.
        let expand_ms = expand_started.elapsed().as_secs_f64() * 1000.0;

        let sparse_state_count = dfa.num_states();
        let root_dense = layout.coordinate_local_state(self.root_coordinate())? as usize;
        let root_sparse = dense_to_sparse[root_dense];
        if root_sparse == u32::MAX {
            return None;
        }
        // Every sparse state is reachable from at least one exact projection
        // seed, but many seeds are intentionally disconnected from state 0.
        // Preserve those roots while quotienting language-equivalent states.
        let minimize_started = std::time::Instant::now();
        let (mut dfa, sparse_to_minimized) =
            dfa.minimize_with_state_mapping_preserve_unreachable();
        let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
        let root = *sparse_to_minimized.get(root_sparse as usize)?;
        if root == u32::MAX {
            return None;
        }
        // Keep the minimized transition graph in its native byte-class
        // alphabet instead of eagerly expanding every class back to its member
        // bytes. The projection tokenizer already supports exact compressed
        // transition segments, and the compact TKS3 wire can persist them
        // directly. State IDs and metadata are unchanged; only transition
        // storage differs from the expanded representation.
        let mut byte_to_class = [0u8; 256];
        let class_members = byte_classes
            .iter()
            .enumerate()
            .map(|(class, members)| {
                let class = u8::try_from(class).ok()?;
                for &byte in members {
                    byte_to_class[byte as usize] = class;
                }
                Some(members.clone().into_boxed_slice())
            })
            .collect::<Option<Vec<_>>>()?;
        let mut row_offsets = Vec::<u32>::with_capacity(dfa.num_states() + 1);
        let mut classes = Vec::<u8>::new();
        let mut targets = Vec::<u32>::new();
        let mut expanded_transition_count = 0usize;
        row_offsets.push(0);
        for state in dfa.states() {
            for (class, &target) in state.transitions.iter() {
                let members = class_members.get(class as usize)?;
                classes.push(class);
                targets.push(target);
                expanded_transition_count = expanded_transition_count.checked_add(members.len())?;
            }
            row_offsets.push(u32::try_from(classes.len()).ok()?);
        }
        for state in dfa.states_mut() {
            state.transitions.clear();
        }
        let segment = CompressedTransitionSegment {
            state_offset: 0,
            state_count: u32::try_from(dfa.num_states()).ok()?,
            byte_to_class: Arc::from(byte_to_class.to_vec().into_boxed_slice()),
            class_members: Arc::from(class_members.into_boxed_slice()),
            row_offsets: Arc::from(row_offsets.into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(classes, targets),
            expanded_transition_count,
        };
        let remap_started = std::time::Instant::now();
        let dense_to_minimized = dense_to_sparse
            .into_iter()
            .map(|sparse| {
                if sparse == u32::MAX {
                    u32::MAX
                } else {
                    sparse_to_minimized[sparse as usize]
                }
            })
            .collect::<Vec<_>>();
        let remap_ms = remap_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][residual_mask_symbolic_sources] dense_states={} seeds={} sparse_states={} minimized_states={} byte_classes={} full_max={} mask_max={} cached_body_pairs={} seeds_ms={:.3} expand_ms={:.3} minimize_ms={:.3} remap_ms={:.3} total_ms={:.3}",
                dense_state_count, seed_count, sparse_state_count, dfa.num_states(), byte_classes.len(),
                self.max, mask_max, cached_body_pairs, seeds_ms, expand_ms, minimize_ms, remap_ms, total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some((dfa, segment, root, dense_to_minimized))
    }

    #[cfg(test)]
    fn finite_mask_dfa_reference(
        &self,
        mask_max: usize,
    ) -> Option<(DFA, CompressedTransitionSegment, u32, Vec<u32>)> {
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let total_started = std::time::Instant::now();
        let pattern_states = self.pattern.num_states();
        let layout = FiniteMaskLayout::new(self, mask_max)?;
        let dense_state_count = layout.dense_state_count;
        if dense_state_count == 0 || dense_state_count > MAX_FINITE_MASK_DENSE_STATES {
            return None;
        }


        let byte_classes = bounded_code_byte_classes(
            &self.pattern,
            &self.body,
            &self.prefix,
            &self.suffix,
        );

        // The finite envelope deliberately identifies the huge exact count
        // interval with a small one-token stencil. A state that is a valid
        // projection source after a long exact input need not be reachable from
        // the finite root at the corresponding small count, because the format
        // DFA can be in a state only reachable after many code words. Seed the
        // finite graph with every *actually possible* projected code-word
        // boundary, then retain the transition closure of those roots. This
        // avoids the full count x body x pattern Cartesian product while still
        // covering every exact token-boundary state.
        // `mask_max` can equal `min` when the declared interval is fixed
        // (or otherwise narrower than the one-token stencil). In that case
        // there is no collapsed interior layer, but the exact lower/upper
        // boundary is still a perfectly valid finite projection root.
        let crossed_boundaries = mask_max.saturating_sub(self.min).saturating_sub(1);
        let after_prefix = step_fixed_bytes(&self.pattern, 0, &self.prefix)?;
        let mut pattern_start = BitSet::new(pattern_states);
        pattern_start.set(after_prefix as usize);
        let mut boundary_classes = Vec::<(usize, BitSet)>::new();

        // Counts below min are observed exactly.
        for completed in 0..self.min {
            let states = self.apply_exact_count(pattern_start.clone(), completed);
            if !states.is_empty() {
                boundary_classes.push((completed, states));
            }
        }

        // The deep interior collapses to the first accepting count. Compute the
        // exact union of format states reachable at any full count represented
        // by that interior layer using the existing relation-doubling oracle.
        let interior_high = self
            .max
            .checked_sub(crossed_boundaries.saturating_add(1));
        if let Some(interior_high) = interior_high.filter(|&high| high >= self.min) {
            let after_low = self.apply_exact_count(pattern_start.clone(), self.min);
            if !after_low.is_empty() {
                let states = self.apply_up_to(after_low, interior_high - self.min);
                if !states.is_empty() {
                    boundary_classes.push((self.min, states));
                }
            }
        }

        // Near the true upper bound, preserve distance-to-upper exactly.
        let upper_start = self.max.saturating_sub(crossed_boundaries).max(self.min);
        for completed in upper_start..=self.max {
            let distance_to_upper = self.max - completed;
            let mapped_completed = mask_max.checked_sub(distance_to_upper)?;
            let states = self.apply_exact_count(pattern_start.clone(), completed);
            if !states.is_empty() {
                boundary_classes.push((mapped_completed, states));
            }
        }

        let seeds_started = std::time::Instant::now();
        let mut dense_to_sparse = vec![u32::MAX; dense_state_count];
        let mut coordinates = Vec::<BoundedCodeOracleCoordinate>::new();
        let mut seed_count = 0usize;
        let mut add_seed = |coordinate: BoundedCodeOracleCoordinate| -> Option<()> {
            let dense = layout.coordinate_local_state(coordinate)? as usize;
            if dense_to_sparse[dense] == u32::MAX {
                dense_to_sparse[dense] = u32::try_from(coordinates.len()).ok()?;
                coordinates.push(coordinate);
                seed_count += 1;
            }
            Some(())
        };
        add_seed(self.root_coordinate())?;
        for (mapped_completed, states) in boundary_classes {
            for pattern_state in states.iter_ones() {
                add_seed(BoundedCodeOracleCoordinate {
                    pattern_state: u32::try_from(pattern_state).ok()?,
                    envelope: BoundedCodeEnvelopeState::Body {
                        completed: mapped_completed,
                        body_state: 0,
                    },
                })?;
            }
        }
        drop(add_seed);
        let seeds_ms = seeds_started.elapsed().as_secs_f64() * 1000.0;

        let expand_started = std::time::Instant::now();
        let class_transitions = FiniteMaskClassTransitions::new(self, &byte_classes)?;
        let mut dfa = DFA::new(coordinates.len());
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, crate::ds::u8set::U8Set::all());
        let mut source = 0usize;
        const EXPANSION_BATCH: usize = 1_024;
        let class_count = byte_classes.len();
        let slot_width = class_count.max(1);
        let mut accepting_rows = Vec::<u8>::new();
        let mut transition_slots =
            Vec::<Option<(BoundedCodeOracleCoordinate, usize)>>::new();
        while source < coordinates.len() {
            while dfa.num_states() < coordinates.len() {
                dfa.add_state();
            }
            let batch_end = coordinates.len().min(source.saturating_add(EXPANSION_BATCH));
            let batch_len = batch_end - source;
            accepting_rows.clear();
            accepting_rows.resize(batch_len, 0);
            transition_slots.clear();
            transition_slots.resize(batch_len.saturating_mul(slot_width), None);
            transition_slots
                .par_chunks_mut(slot_width)
                .zip(accepting_rows.par_iter_mut())
                .enumerate()
                .for_each(|(row_offset, (row, accepting_slot))| {
                    let coordinate = coordinates[source + row_offset];
                    let accepting = matches!(coordinate.envelope, BoundedCodeEnvelopeState::Done)
                        && !self.pattern.finalizers(coordinate.pattern_state).is_empty();
                    *accepting_slot = u8::from(accepting);
                    for (class, members) in byte_classes.iter().enumerate() {
                        let Some(next) = self.step_coordinate_class(
                            coordinate,
                            members[0],
                            class,
                            mask_max,
                            &class_transitions,
                        ) else {
                            continue;
                        };
                        let Some(dense) = layout.coordinate_local_state(next) else {
                            continue;
                        };
                        row[class] = Some((next, dense as usize));
                    }
                });

            for row_offset in 0..batch_len {
                let source_state = (source + row_offset) as u32;
                let mut finalizers = BitSet::new(1);
                if accepting_rows[row_offset] != 0 {
                    finalizers.set(0);
                }
                dfa.overwrite_state_metadata(source_state, finalizers, BitSet::new(1));
                let row_start = row_offset * slot_width;
                for (class, slot) in transition_slots[row_start..row_start + class_count]
                    .iter()
                    .enumerate()
                {
                    let Some((next, dense)) = *slot else {
                        continue;
                    };
                    let target = if dense_to_sparse[dense] == u32::MAX {
                        let target = u32::try_from(coordinates.len()).ok()?;
                        dense_to_sparse[dense] = target;
                        coordinates.push(next);
                        target
                    } else {
                        dense_to_sparse[dense]
                    };
                    while dfa.num_states() <= target as usize {
                        dfa.add_state();
                    }
                    dfa.add_transition(source_state, class as u8, target);
                }
            }
            source = batch_end;
        }
        // Minimization immediately clears possible-future metadata before
        // partitioning and recomputes it on the minimized DFA. Computing the
        // full sparse DFA's futures here would therefore be pure dead work.
        let expand_ms = expand_started.elapsed().as_secs_f64() * 1000.0;

        let sparse_state_count = dfa.num_states();
        let root_dense = layout.coordinate_local_state(self.root_coordinate())? as usize;
        let root_sparse = dense_to_sparse[root_dense];
        if root_sparse == u32::MAX {
            return None;
        }
        // Every sparse state is reachable from at least one exact projection
        // seed, but many seeds are intentionally disconnected from state 0.
        // Preserve those roots while quotienting language-equivalent states.
        let minimize_started = std::time::Instant::now();
        let (mut dfa, sparse_to_minimized) =
            dfa.minimize_with_state_mapping_preserve_unreachable();
        let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
        let root = *sparse_to_minimized.get(root_sparse as usize)?;
        if root == u32::MAX {
            return None;
        }
        // Keep the minimized transition graph in its native byte-class
        // alphabet instead of eagerly expanding every class back to its member
        // bytes. The projection tokenizer already supports exact compressed
        // transition segments, and the compact TKS3 wire can persist them
        // directly. State IDs and metadata are unchanged; only transition
        // storage differs from the expanded representation.
        let mut byte_to_class = [0u8; 256];
        let class_members = byte_classes
            .iter()
            .enumerate()
            .map(|(class, members)| {
                let class = u8::try_from(class).ok()?;
                for &byte in members {
                    byte_to_class[byte as usize] = class;
                }
                Some(members.clone().into_boxed_slice())
            })
            .collect::<Option<Vec<_>>>()?;
        let mut row_offsets = Vec::<u32>::with_capacity(dfa.num_states() + 1);
        let mut classes = Vec::<u8>::new();
        let mut targets = Vec::<u32>::new();
        let mut expanded_transition_count = 0usize;
        row_offsets.push(0);
        for state in dfa.states() {
            for (class, &target) in state.transitions.iter() {
                let members = class_members.get(class as usize)?;
                classes.push(class);
                targets.push(target);
                expanded_transition_count = expanded_transition_count.checked_add(members.len())?;
            }
            row_offsets.push(u32::try_from(classes.len()).ok()?);
        }
        for state in dfa.states_mut() {
            state.transitions.clear();
        }
        let segment = CompressedTransitionSegment {
            state_offset: 0,
            state_count: u32::try_from(dfa.num_states()).ok()?,
            byte_to_class: Arc::from(byte_to_class.to_vec().into_boxed_slice()),
            class_members: Arc::from(class_members.into_boxed_slice()),
            row_offsets: Arc::from(row_offsets.into_boxed_slice()),
            entries: CompressedTransitionEntries::from_parts(classes, targets),
            expanded_transition_count,
        };
        let remap_started = std::time::Instant::now();
        let dense_to_minimized = dense_to_sparse
            .into_iter()
            .map(|sparse| {
                if sparse == u32::MAX {
                    u32::MAX
                } else {
                    sparse_to_minimized[sparse as usize]
                }
            })
            .collect::<Vec<_>>();
        let remap_ms = remap_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][residual_mask_symbolic_sources] dense_states={} seeds={} sparse_states={} minimized_states={} byte_classes={} seeds_ms={:.3} expand_ms={:.3} minimize_ms={:.3} remap_ms={:.3} total_ms={:.3}",
                dense_state_count, seed_count, sparse_state_count, dfa.num_states(), byte_classes.len(),
                seeds_ms, expand_ms, minimize_ms, remap_ms, total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some((dfa, segment, root, dense_to_minimized))
    }
}

fn unwrap_shared_expr(mut expr: &Expr) -> &Expr {
    while let Expr::Shared(inner) = expr {
        expr = inner;
    }
    expr
}

fn flatten_intersection_operands<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match unwrap_shared_expr(expr) {
        Expr::Intersect { expr, intersect } => {
            flatten_intersection_operands(expr, out);
            flatten_intersection_operands(intersect, out);
        }
        other => out.push(other),
    }
}

fn flatten_sequence_operands<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match unwrap_shared_expr(expr) {
        Expr::Seq(parts) => {
            for part in parts {
                flatten_sequence_operands(part, out);
            }
        }
        other => out.push(other),
    }
}

fn universal_pattern_dfa() -> DFA {
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, crate::ds::u8set::U8Set::all());
    dfa.set_transitions_from_sorted_entries(0, (u8::MIN..=u8::MAX).map(|byte| (byte, 0)).collect());
    let mut accepting = BitSet::new(1);
    accepting.set(0);
    let mut future = BitSet::new(1);
    future.set(0);
    dfa.overwrite_state_metadata(0, accepting, future);
    dfa
}

/// Recognize the language skeleton `prefix body* suffix`. This is intentionally
/// stricter than general regex equivalence: it is used only to prove that an
/// intersection operand is structurally the unbounded version of a selected
/// bounded-code envelope.
fn unbounded_code_envelope(expr: &Expr) -> Option<(Vec<u8>, Expr, Vec<u8>)> {
    let mut parts = Vec::new();
    flatten_sequence_operands(expr, &mut parts);
    let repeat_index = parts.iter().position(|part| {
        matches!(
            unwrap_shared_expr(part),
            Expr::Repeat {
                min: 0,
                max: None,
                ..
            }
        )
    })?;
    if parts.iter().enumerate().any(|(index, part)| {
        index != repeat_index
            && !matches!(unwrap_shared_expr(part), Expr::U8Seq(bytes) if !bytes.is_empty())
    }) {
        return None;
    }
    if parts[repeat_index + 1..]
        .iter()
        .any(|part| matches!(unwrap_shared_expr(part), Expr::Repeat { .. }))
    {
        return None;
    }
    let mut prefix = Vec::new();
    for part in &parts[..repeat_index] {
        let Expr::U8Seq(bytes) = unwrap_shared_expr(part) else {
            return None;
        };
        prefix.extend_from_slice(bytes);
    }
    let mut suffix = Vec::new();
    for part in &parts[repeat_index + 1..] {
        let Expr::U8Seq(bytes) = unwrap_shared_expr(part) else {
            return None;
        };
        suffix.extend_from_slice(bytes);
    }
    if prefix.is_empty() || suffix.is_empty() {
        return None;
    }
    let Expr::Repeat {
        expr: body,
        min: 0,
        max: None,
    } = unwrap_shared_expr(parts[repeat_index])
    else {
        return None;
    };
    Some((
        prefix,
        unwrap_shared_expr(body).clone(),
        suffix,
    ))
}

fn bounded_code_envelope(expr: &Expr) -> Option<(Vec<u8>, Expr, usize, usize, Vec<u8>)> {
    let mut parts = Vec::new();
    flatten_sequence_operands(expr, &mut parts);
    let repeat_index = parts
        .iter()
        .position(|part| matches!(unwrap_shared_expr(part), Expr::Repeat { max: Some(_), .. }))?;
    if parts
        .iter()
        .enumerate()
        .any(|(index, part)| {
            index != repeat_index
                && !matches!(unwrap_shared_expr(part), Expr::U8Seq(bytes) if !bytes.is_empty())
        })
    {
        return None;
    }
    if parts[repeat_index + 1..]
        .iter()
        .any(|part| matches!(unwrap_shared_expr(part), Expr::Repeat { .. }))
    {
        return None;
    }
    let mut prefix = Vec::new();
    for part in &parts[..repeat_index] {
        let Expr::U8Seq(bytes) = unwrap_shared_expr(part) else {
            return None;
        };
        prefix.extend_from_slice(bytes);
    }
    let mut suffix = Vec::new();
    for part in &parts[repeat_index + 1..] {
        let Expr::U8Seq(bytes) = unwrap_shared_expr(part) else {
            return None;
        };
        suffix.extend_from_slice(bytes);
    }
    if prefix.is_empty() || suffix.is_empty() {
        return None;
    }
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared_expr(parts[repeat_index])
    else {
        return None;
    };
    (*min <= *max).then(|| (prefix, unwrap_shared_expr(body).clone(), *min, *max, suffix))
}

fn exact_productive_states(dfa: &DFA) -> Vec<bool> {
    let mut reverse = vec![Vec::<u32>::new(); dfa.num_states()];
    for (source, state) in dfa.states().iter().enumerate() {
        for (_, &target) in state.transitions.iter() {
            reverse[target as usize].push(source as u32);
        }
    }
    let mut productive = vec![false; dfa.num_states()];
    let mut stack = Vec::new();
    for state in 0..dfa.num_states() as u32 {
        if !dfa.finalizers(state).is_empty() {
            productive[state as usize] = true;
            stack.push(state);
        }
    }
    while let Some(state) = stack.pop() {
        for &predecessor in &reverse[state as usize] {
            if !productive[predecessor as usize] {
                productive[predecessor as usize] = true;
                stack.push(predecessor);
            }
        }
    }
    productive
}

fn dfa_language_is_prefix_free(dfa: &DFA, productive: &[bool]) -> bool {
    for state in dfa.states() {
        if state.finalizers.is_empty() {
            continue;
        }
        if state
            .transitions
            .iter()
            .any(|(_, &target)| productive[target as usize])
        {
            return false;
        }
    }
    true
}

fn step_fixed_bytes(dfa: &DFA, mut state: u32, bytes: &[u8]) -> Option<u32> {
    for &byte in bytes {
        state = dfa.step(state, byte)?;
    }
    Some(state)
}

#[derive(Debug)]
struct ResidualRuntimeStore {
    arena: ResidualArena,
    root: ResidualId,
    state_by_residual: Vec<u32>,
    residual_by_state: FxHashMap<u32, ResidualId>,
    state_by_oracle_coordinate: FxHashMap<BoundedCodeOracleCoordinate, u32>,
    state_by_residual_coordinate:
        FxHashMap<(ResidualId, BoundedCodeOracleCoordinate), u32>,
    coordinate_by_state: FxHashMap<u32, BoundedCodeOracleCoordinate>,
    oracle_future_by_state: FxHashMap<u32, bool>,
    liveness_oracle: Option<BoundedCodeIntersectionOracle>,
    oracle_byte_to_class: Option<Box<[u8; 256]>>,
    oracle_language_finite: Option<bool>,
    oracle_coordinates: Vec<BoundedCodeOracleSlot>,
    oracle_futures: Vec<Option<bool>>,
    parser_transparent_byte_family_cache: FxHashSet<(u32, U8Set, u32)>,
    slice_atom_body_exact_cache: FxHashMap<u64, bool>,
    slice_atom_pattern_targets_cache: FxHashMap<(u64, u32), Option<BitSet>>,
    body_boundary_future_by_completed: Option<Arc<Vec<BitSet>>>,
    transition_rows_by_state: FxHashMap<u32, Box<[u32; 256]>>,
}

/// Immutable byte-step view of a certified bounded-code residual. This is the
/// same transition semantics as `BoundedCodeIntersectionOracle::step_coordinate`
/// but keeps only immutable data outside the residual store mutex so a
/// parser-transparent trie walk can advance coordinates without allocating or
/// interning virtual state ids.
#[derive(Debug, Clone)]
struct DirectCoordinateStepper {
    pattern: Arc<DFA>,
    body: Arc<DFA>,
    byte_to_class: Box<[u8; 256]>,
    class_transitions: FiniteMaskClassTransitions,
    body_productive: Arc<[bool]>,
    prefix: Arc<[u8]>,
    suffix: Arc<[u8]>,
    min: usize,
    max: usize,
}

impl DirectCoordinateStepper {
    fn new(oracle: &BoundedCodeIntersectionOracle) -> Option<Self> {
        let byte_classes = bounded_code_byte_classes(
            &oracle.pattern,
            &oracle.body,
            &oracle.prefix,
            &oracle.suffix,
        );
        let mut byte_to_class = Box::new([0u8; 256]);
        for (class, members) in byte_classes.iter().enumerate() {
            let class = u8::try_from(class).ok()?;
            for &byte in members {
                byte_to_class[byte as usize] = class;
            }
        }
        let class_transitions = FiniteMaskClassTransitions::new(oracle, &byte_classes)?;
        Some(Self {
            pattern: Arc::clone(&oracle.pattern),
            body: Arc::clone(&oracle.body),
            byte_to_class,
            class_transitions,
            body_productive: Arc::from(oracle.body_productive.clone()),
            prefix: Arc::clone(&oracle.prefix),
            suffix: Arc::clone(&oracle.suffix),
            min: oracle.min,
            max: oracle.max,
        })
    }

    #[inline(always)]
    fn step(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        byte: u8,
    ) -> Option<BoundedCodeOracleCoordinate> {
        let class = self.byte_to_class[byte as usize] as usize;
        let pattern_state = self
            .class_transitions
            .pattern_step(coordinate.pattern_state, class)?;
        let envelope = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if self.prefix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.prefix.len() {
                    BoundedCodeEnvelopeState::Body {
                        completed: 0,
                        body_state: 0,
                    }
                } else {
                    BoundedCodeEnvelopeState::Prefix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                if body_state == 0
                    && completed >= self.min
                    && self.suffix.first().copied()? == byte
                {
                    if self.suffix.len() == 1 {
                        BoundedCodeEnvelopeState::Done
                    } else {
                        BoundedCodeEnvelopeState::Suffix { next: 1 }
                    }
                } else {
                    if completed >= self.max {
                        return None;
                    }
                    let target = self
                        .class_transitions
                        .body_step(&self.body, body_state, byte, class)?;
                    if !self.body.finalizers(target).is_empty() {
                        BoundedCodeEnvelopeState::Body {
                            completed: completed.checked_add(1)?,
                            body_state: 0,
                        }
                    } else if self.body_productive[target as usize] {
                        BoundedCodeEnvelopeState::Body {
                            completed,
                            body_state: target,
                        }
                    } else {
                        return None;
                    }
                }
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if self.suffix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.suffix.len() {
                    BoundedCodeEnvelopeState::Done
                } else {
                    BoundedCodeEnvelopeState::Suffix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Done => return None,
        };
        Some(BoundedCodeOracleCoordinate {
            pattern_state,
            envelope,
        })
    }

    #[inline(always)]
    fn accepting(&self, coordinate: BoundedCodeOracleCoordinate) -> bool {
        matches!(coordinate.envelope, BoundedCodeEnvelopeState::Done)
            && !self.pattern.finalizers(coordinate.pattern_state).is_empty()
    }
}

/// Exact general symbolic tokenizer component. The regex upper bounds live in
/// `ResidualNode::Repeat`; this store grows only when runtime bytes discover a
/// new canonical language residual.
#[derive(Debug)]
pub(super) struct VirtualResidualRuntime {
    runtime_index: u32,
    terminal: TerminalID,
    physical_state_count: u32,
    root_state: u32,
    root_has_future: bool,
    preserve_oracle_coordinate: bool,
    coordinate_runtime: bool,
    direct_coordinate_stepper: Option<DirectCoordinateStepper>,
    state_allocator: Arc<VirtualStateAllocator>,
    state_owners: Arc<VirtualRuntimeStateOwners>,
    accepting: BitSet,
    live: BitSet,
    dead: BitSet,
    accepting_list: Box<[TerminalID]>,
    store: Mutex<ResidualRuntimeStore>,
}

/// Exact semantic key for the finite one-token projection language of a
/// bounded-code residual.  The relation tables used by `finite_mask_dfa` are
/// deterministic derivatives of these fields, so equality here is sufficient
/// to share one finite DFA between residual runtimes that differ only in their
/// outer terminal identity/runtime coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct FiniteMaskProjectionLanguageKey {
    pattern: Arc<DFA>,
    body: Arc<DFA>,
    prefix: Arc<[u8]>,
    suffix: Arc<[u8]>,
    min: usize,
    max: usize,
}

impl VirtualResidualRuntime {
    fn intern_oracle_coordinate_locked(
        &self,
        store: &mut ResidualRuntimeStore,
        coordinate: BoundedCodeOracleCoordinate,
    ) -> Option<u32> {
        if let Some(&state) = store.state_by_oracle_coordinate.get(&coordinate) {
            return Some(state);
        }
        let state = self.state_allocator.allocate().expect(
            "exact residual tokenizer state-id space exhausted below the dynamic-NFA high-bit tag",
        );
        self.state_owners
            .register_virtual(state, self.runtime_index)
            .expect("residual virtual state owner index must follow shared allocator");
        store.state_by_oracle_coordinate.insert(coordinate, state);
        store.coordinate_by_state.insert(state, coordinate);
        Some(state)
    }

    fn coordinate_future_locked(
        store: &mut ResidualRuntimeStore,
        state: u32,
        coordinate: BoundedCodeOracleCoordinate,
    ) -> Option<bool> {
        if let Some(&future) = store.oracle_future_by_state.get(&state) {
            return Some(future);
        }
        let boundary_future = store
            .body_boundary_future_by_completed
            .as_ref()
            .map(Arc::clone);
        let oracle = store.liveness_oracle.as_mut()?;
        let future = boundary_future
            .as_deref()
            .and_then(|rows| oracle.has_future_with_boundary_table(coordinate, rows))
            .unwrap_or_else(|| oracle.has_future(coordinate));
        store.oracle_future_by_state.insert(state, future);
        Some(future)
    }

    pub(super) fn master_slice_artifact(&self) -> Option<VirtualResidualMasterSliceArtifact> {
        let store = self.store.lock().unwrap();
        let oracle = store.liveness_oracle.as_ref()?;
        let rows = store.body_boundary_future_by_completed.as_ref()?;
        let pattern_states = oracle.pattern.num_states();
        let row_words = pattern_states.div_ceil(64);
        if rows.len() != oracle.max.checked_add(1)?
            || rows.iter().any(|row| row.len() != pattern_states || row.words().len() != row_words)
        {
            return None;
        }
        let mut words = Vec::with_capacity(rows.len().saturating_mul(row_words));
        for row in rows.iter() {
            words.extend_from_slice(row.words());
        }
        Some(VirtualResidualMasterSliceArtifact {
            terminal: self.terminal,
            root_state: self.root_state,
            pattern: CompactResidualDfaArtifact::from_dfa(oracle.pattern.as_ref())?,
            body: CompactResidualDfaArtifact::from_dfa(oracle.body.as_ref())?,
            body_productive: oracle.body_productive.to_vec(),
            prefix: oracle.prefix.to_vec(),
            suffix: oracle.suffix.to_vec(),
            min: oracle.min,
            max: oracle.max,
            suffix_accepting: oracle.suffix_accepting.clone(),
            pattern_states: u32::try_from(pattern_states).ok()?,
            rows: u32::try_from(rows.len()).ok()?,
            words,
            body_exact: store
                .slice_atom_body_exact_cache
                .iter()
                .map(|(&fingerprint, &exact)| (fingerprint, exact))
                .collect(),
            pattern_targets: store
                .slice_atom_pattern_targets_cache
                .iter()
                .map(|(&(fingerprint, pattern_state), targets)| {
                    (
                        fingerprint,
                        pattern_state,
                        targets.as_ref().map(|targets| targets.words().to_vec()),
                    )
                })
                .collect(),
        })
    }

    pub(super) fn restore_master_slice_artifact(
        &self,
        artifact: VirtualResidualMasterSliceArtifact,
    ) -> Result<(), String> {
        if artifact.terminal != self.terminal || artifact.root_state != self.root_state {
            return Err("virtual residual master-slice artifact runtime mismatch".to_owned());
        }
        let mut store = self.store.lock().unwrap();
        let oracle = store
            .liveness_oracle
            .as_ref()
            .ok_or_else(|| "virtual residual master-slice artifact has no liveness oracle".to_owned())?;
        let pattern_states = oracle.pattern.num_states();
        let expected_rows = oracle.max.saturating_add(1);
        if artifact.pattern_states as usize != pattern_states || artifact.rows as usize != expected_rows {
            return Err("virtual residual master-slice artifact shape mismatch".to_owned());
        }
        let row_words = pattern_states.div_ceil(64);
        if artifact.words.len() != expected_rows.saturating_mul(row_words) {
            return Err("virtual residual master-slice artifact word count mismatch".to_owned());
        }
        let mut rows = Vec::with_capacity(expected_rows);
        for words in artifact.words.chunks_exact(row_words) {
            rows.push(
                BitSet::from_dense_words(pattern_states, words)
                    .ok_or_else(|| "virtual residual master-slice artifact has invalid dense row".to_owned())?,
            );
        }
        let mut body_exact = FxHashMap::default();
        for (fingerprint, exact) in artifact.body_exact {
            body_exact.insert(fingerprint, exact);
        }
        let mut pattern_targets = FxHashMap::default();
        for (fingerprint, pattern_state, words) in artifact.pattern_targets {
            if pattern_state as usize >= pattern_states {
                return Err(
                    "virtual residual master-slice artifact pattern state is out of range"
                        .to_owned(),
                );
            }
            let targets = words
                .map(|words| {
                    BitSet::from_dense_words(pattern_states, &words).ok_or_else(|| {
                        "virtual residual master-slice artifact has invalid target row".to_owned()
                    })
                })
                .transpose()?;
            pattern_targets.insert((fingerprint, pattern_state), targets);
        }
        store.body_boundary_future_by_completed = Some(Arc::new(rows));
        store.slice_atom_body_exact_cache = body_exact;
        store.slice_atom_pattern_targets_cache = pattern_targets;
        Ok(())
    }

    pub(super) fn compact_liveness_oracle_from_master_slice_artifact(
        artifact: &VirtualResidualMasterSliceArtifact,
    ) -> Option<BoundedCodeIntersectionOracle> {
        let pattern_states = artifact.pattern.state_count();
        let body_states = artifact.body.state_count();
        if pattern_states == 0
            || pattern_states != artifact.pattern_states as usize
            || body_states == 0
            || artifact.body_productive.len() != body_states
            || artifact.min > artifact.max
            || artifact.max != artifact.rows.checked_sub(1)? as usize
            || artifact.prefix.is_empty()
            || artifact.suffix.is_empty()
            || artifact.suffix_accepting.len() != pattern_states
        {
            return None;
        }
        let pattern = Arc::new(artifact.pattern.clone().into_dfa()?);
        let body = Arc::new(artifact.body.clone().into_dfa()?);
        Some(BoundedCodeIntersectionOracle {
            pattern,
            body,
            body_productive: artifact.body_productive.clone().into_boxed_slice(),
            prefix: Arc::from(artifact.prefix.clone().into_boxed_slice()),
            suffix: Arc::from(artifact.suffix.clone().into_boxed_slice()),
            min: artifact.min,
            max: artifact.max,
            suffix_accepting: artifact.suffix_accepting.clone(),
            completion_relations: vec![None; body_states],
            completion_row_cache: FxHashMap::default(),
            mask_result_future_headroom: FxHashMap::default(),
            // The transferred backward-future table is the exact dynamic-
            // programming replacement for these relation powers. Loaded
            // runtimes route future queries through that table below.
            exact_powers: Vec::new(),
            prefix_sums: Vec::new(),
        })
    }

    pub(super) fn finite_mask_projection_language_key(
        &self,
    ) -> Option<FiniteMaskProjectionLanguageKey> {
        let store = self.store.lock().unwrap();
        let oracle = store.liveness_oracle.as_ref()?;
        Some(FiniteMaskProjectionLanguageKey {
            pattern: Arc::clone(&oracle.pattern),
            body: Arc::clone(&oracle.body),
            prefix: Arc::clone(&oracle.prefix),
            suffix: Arc::clone(&oracle.suffix),
            min: oracle.min,
            max: oracle.max,
        })
    }
    pub(super) fn finite_mask_projection_dense_state_count(
        &self,
        max_token_len: usize,
    ) -> Option<usize> {
        let store = self.store.lock().unwrap();
        let oracle = store.liveness_oracle.as_ref()?;
        let minimum_body_width = oracle.body.min_match_byte_len()?.max(1);
        let crossed_boundaries = max_token_len
            .div_ceil(minimum_body_width)
            .saturating_add(1);
        let crossed_boundaries = oracle.finite_mask_stencil_crossings(crossed_boundaries)?;
        if oracle.min > crossed_boundaries.saturating_add(1) {
            return None;
        }
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)?
            .checked_add(1)?;
        let mask_max = oracle.max.min(desired_mask_max);
        oracle.finite_mask_dense_state_count(mask_max)
    }

    /// Exact one-model-token RESULT key. Unlike the reusable finite-mask DFA
    /// coordinate, this may use a smaller closed-subgraph future-witness bound.
    /// Do not use it for lexer byte-transition rows or compiled projection maps.
    ///
    /// Let R count body-boundary pattern states reachable by completing the
    /// current body and then taking whole body-code copies. Every boundary
    /// reached during any candidate token remains in this closed set. A
    /// shortest accepting body-copy path has no repeated pattern vertex. Use
    /// the maximum finite shortest distance to a suffix-admitting boundary,
    /// plus two conservative copies; this is at most R+1. Add the maximum
    /// copies crossed by a token of length K before collapsing only counts
    /// already above the lower bound and farther than that sum from the upper
    /// bound. Acceptance itself, prefix/suffix positions and body state remain
    /// exact. This is not a claim of equal unrestricted residual languages.
    ///
    /// The dense encoding retains pattern_state as its residue modulo the
    /// pattern-state count. Thus state-dependent stencil sizes cannot alias
    /// different pattern states; within one pattern state phase ranges remain
    /// disjoint. Full parser paths, correlated exclusions, initial-state status
    /// and vocabulary identity are retained by the caller's result key.
    pub(super) fn direct_coordinate_mask_result_dense_key(
        &self,
        source: VirtualResidualDirectCoordinate,
        max_token_len: usize,
    ) -> Option<(u32, u32)> {
        if source.runtime_index != self.runtime_index || max_token_len == 0 {
            return None;
        }
        let mut store = self.store.lock().ok()?;
        let oracle = store.liveness_oracle.as_mut()?;
        let minimum_body_width = oracle.body.min_match_byte_len()?.max(1);
        let token_crossings = max_token_len.div_ceil(minimum_body_width).checked_add(1)?;
        let BoundedCodeEnvelopeState::Body { body_state, .. } = source.coordinate.envelope else {
            let crossings = oracle.finite_mask_stencil_crossings(token_crossings)?;
            return self.direct_coordinate_dense_key_with_stencil(source, oracle, crossings);
        };
        let headroom = oracle.mask_result_future_repeat_headroom(source.coordinate.pattern_state, body_state)?;
        let mut crossed_boundaries = token_crossings.checked_add(headroom)?;
        // Retain the already-supported global lane for large lower minima.
        if oracle.min > crossed_boundaries.checked_add(1)? {
            crossed_boundaries = oracle.finite_mask_stencil_crossings(token_crossings)?;
        }
        self.direct_coordinate_dense_key_with_stencil(source, oracle, crossed_boundaries)
    }

    pub(super) fn new(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        Self::new_impl(
            expr,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            false,
            false,
            false,
            None,
        )
    }

    /// Construct a residual runtime for direct dynamic masking. When the
    /// expression admits a certified bounded-code oracle, keep that exact
    /// coordinate as the runtime state representation; otherwise this is
    /// identical to `new`. Keeping this choice out of `new` preserves the
    /// legacy/general residual semantics used by callers that do not opt into
    /// the dynamic-mask representation.
    pub(super) fn new_dynamic(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        Self::new_impl(
            expr,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            false,
            false,
            true,
            None,
        )
    }

    /// Construct the exact derivative runtime without eagerly building the
    /// optional bounded-code liveness oracle. This is used by serialized
    /// dynamic compilation: the producer needs the physical proxy/root
    /// metadata for the transfer artifact, but the execution process will
    /// reconstruct the runtime (and its liveness oracle) after load.
    pub(super) fn new_without_liveness_oracle(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        Self::new_impl(
            expr,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            false,
            true,
            false,
            None,
        )
    }

    pub(super) fn new_with_liveness_oracle(
        expr: &Expr,
        liveness_oracle: BoundedCodeIntersectionOracle,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        Self::new_impl(
            expr,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            false,
            false,
            true,
            Some(liveness_oracle),
        )
    }

    pub(super) fn new_preserving_oracle_coordinate(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        Self::new_impl(
            expr,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            true,
            false,
            false,
            None,
        )
    }

    fn new_impl(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
        preserve_oracle_coordinate: bool,
        suppress_dynamic_liveness_oracle: bool,
        coordinate_runtime: bool,
        prebuilt_liveness_oracle: Option<BoundedCodeIntersectionOracle>,
    ) -> Option<Self> {
        if terminal >= num_terminals
            || physical_state_count == 0
            || root_state >= physical_state_count
            || state_owners.owner_index(root_state) != Some(runtime_index as usize)
        {
            return None;
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_RESIDUAL").is_some();
        let arena_started = profile.then(std::time::Instant::now);
        let (mut arena, root) = ResidualArena::from_expr(expr)?;
        let arena_ms = arena_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let arena_states = arena.state_count();
        let build_dynamic_liveness_oracle = !suppress_dynamic_liveness_oracle
            && std::env::var("GLRMASK_DYNAMIC_RESIDUAL_LIVENESS_ORACLE")
                .ok()
                .is_none_or(|value| !matches!(value.trim(), "0" | "false" | "no" | "off"));
        let oracle_started = profile.then(std::time::Instant::now);
        let liveness_oracle = if prebuilt_liveness_oracle.is_some() {
            prebuilt_liveness_oracle
        } else if preserve_oracle_coordinate || build_dynamic_liveness_oracle {
            if preserve_oracle_coordinate {
                BoundedCodeIntersectionOracle::from_expr(expr)
            } else {
                BoundedCodeIntersectionOracle::from_dynamic_expr(expr)
            }
        } else {
            None
        };
        let oracle_ms = oracle_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_residual_new] terminal={} preserve_coordinate={} arena_states={} arena_ms={:.3} oracle={} oracle_ms={:.3}",
                terminal,
                preserve_oracle_coordinate,
                arena_states,
                arena_ms,
                liveness_oracle.is_some(),
                oracle_ms,
            );
        }
        let root_oracle_coordinate = liveness_oracle
            .as_ref()
            .map(BoundedCodeIntersectionOracle::root_coordinate);
        let coordinate_runtime = coordinate_runtime
            && !preserve_oracle_coordinate
            && liveness_oracle.is_some();
        let direct_coordinate_stepper = coordinate_runtime
            .then(|| liveness_oracle.as_ref().and_then(DirectCoordinateStepper::new))
            .flatten();
        // Keep the physical proxy root's serialized future bit conservative.
        // Old artifacts and load-time validation use that physical metadata, so
        // installing an exact liveness sidecar must not silently change the wire
        // contract. At runtime `observation()` overlays an exact future bit for
        // residuals carrying a certified bounded-code coordinate; uncertified
        // Boolean residuals retain the conservative bit and are resolved through
        // the fallible `exact_has_future` boundary. Construction therefore does
        // not run the generic potentially expensive emptiness solver merely to
        // populate serialized proxy metadata.
        let root_live = arena.conservative_has_future(root);
        let mut state_by_residual = vec![u32::MAX; root as usize + 1];
        state_by_residual[root as usize] = root_state;
        let mut oracle_coordinates = vec![BoundedCodeOracleSlot::Unknown; arena.state_count()];
        let oracle_futures = vec![None; arena.state_count()];
        if let Some(coordinate) = root_oracle_coordinate {
            oracle_coordinates[root as usize] = BoundedCodeOracleSlot::Exact(coordinate);
        }
        let mut state_by_residual_coordinate = FxHashMap::default();
        let mut coordinate_by_state = FxHashMap::default();
        let mut state_by_oracle_coordinate = FxHashMap::default();
        if preserve_oracle_coordinate {
            let coordinate = root_oracle_coordinate?;
            state_by_residual_coordinate.insert((root, coordinate), root_state);
            coordinate_by_state.insert(root_state, coordinate);
        } else if coordinate_runtime {
            let coordinate = root_oracle_coordinate?;
            state_by_oracle_coordinate.insert(coordinate, root_state);
            coordinate_by_state.insert(root_state, coordinate);
        }
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        let oracle_byte_to_class = liveness_oracle
            .as_ref()
            .map(BoundedCodeIntersectionOracle::byte_to_class_map);
        let oracle_language_finite = liveness_oracle
            .as_ref()
            .map(|oracle| dfa_language_is_finite(&oracle.body));
        Some(Self {
            runtime_index,
            terminal,
            physical_state_count,
            root_state,
            root_has_future: root_live,
            preserve_oracle_coordinate,
            coordinate_runtime,
            direct_coordinate_stepper,
            state_allocator,
            state_owners,
            accepting,
            live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
            store: Mutex::new(ResidualRuntimeStore {
                arena,
                root,
                state_by_residual,
                residual_by_state: FxHashMap::default(),
                state_by_oracle_coordinate,
                state_by_residual_coordinate,
                coordinate_by_state,
                oracle_future_by_state: FxHashMap::default(),
                liveness_oracle,
                oracle_byte_to_class,
                oracle_language_finite,
                oracle_coordinates,
                oracle_futures,
                parser_transparent_byte_family_cache: FxHashSet::default(),
                slice_atom_body_exact_cache: FxHashMap::default(),
                slice_atom_pattern_targets_cache: FxHashMap::default(),
                body_boundary_future_by_completed: None,
                transition_rows_by_state: FxHashMap::default(),
            }),
        })
    }

    pub(super) fn terminal(&self) -> TerminalID {
        self.terminal
    }

    pub(super) fn root_state(&self) -> u32 {
        self.root_state
    }

    pub(super) fn physical_state_count(&self) -> u32 {
        self.physical_state_count
    }

    fn residual_for_state(store: &ResidualRuntimeStore, root_state: u32, state: u32) -> Option<ResidualId> {
        if state == root_state {
            Some(store.root)
        } else {
            store.residual_by_state.get(&state).copied()
        }
    }

    #[inline]
    fn oracle_coordinate_for_state_locked(
        &self,
        store: &ResidualRuntimeStore,
        state: u32,
    ) -> Option<BoundedCodeOracleCoordinate> {
        if self.coordinate_runtime || self.preserve_oracle_coordinate {
            return store.coordinate_by_state.get(&state).copied();
        }
        let residual = Self::residual_for_state(store, self.root_state, state)?;
        match store.oracle_coordinates.get(residual as usize).copied()? {
            BoundedCodeOracleSlot::Exact(coordinate) => Some(coordinate),
            BoundedCodeOracleSlot::Unknown | BoundedCodeOracleSlot::Ambiguous => None,
        }
    }

    fn intern_locked(
        &self,
        store: &mut ResidualRuntimeStore,
        residual: ResidualId,
        coordinate: Option<BoundedCodeOracleCoordinate>,
    ) -> Option<u32> {
        if self.preserve_oracle_coordinate {
            let coordinate = coordinate?;
            if let Some(&state) = store
                .state_by_residual_coordinate
                .get(&(residual, coordinate))
            {
                return Some(state);
            }
            let state = self.state_allocator.allocate().expect(
                "exact residual tokenizer state-id space exhausted below the dynamic-NFA high-bit tag",
            );
            self.state_owners
                .register_virtual(state, self.runtime_index)
                .expect("residual virtual state owner index must follow shared allocator");
            store
                .state_by_residual_coordinate
                .insert((residual, coordinate), state);
            store.coordinate_by_state.insert(state, coordinate);
            store.residual_by_state.insert(state, residual);
            return Some(state);
        }

        let residual_index = residual as usize;
        if store.state_by_residual.len() <= residual_index {
            store.state_by_residual.resize(residual_index + 1, u32::MAX);
        }
        let state = store.state_by_residual[residual_index];
        if state != u32::MAX {
            return Some(state);
        }
        let state = self.state_allocator.allocate().expect(
            "exact residual tokenizer state-id space exhausted below the dynamic-NFA high-bit tag",
        );
        self.state_owners
            .register_virtual(state, self.runtime_index)
            .expect("residual virtual state owner index must follow shared allocator");
        store.state_by_residual[residual_index] = state;
        store.residual_by_state.insert(state, residual);
        Some(state)
    }

    pub(super) fn handles_state(&self, state: u32) -> bool {
        self.state_owners.owner_index(state) == Some(self.runtime_index as usize)
    }

    pub(super) fn owner_index(&self, state: u32) -> Option<usize> {
        self.state_owners.owner_index(state)
    }

    fn step_residual_locked(
        &self,
        store: &mut ResidualRuntimeStore,
        state: u32,
        residual: ResidualId,
        byte: u8,
    ) -> Option<u32> {
        let source_coordinate = if self.preserve_oracle_coordinate {
            store
                .coordinate_by_state
                .get(&state)
                .copied()
                .map(BoundedCodeOracleSlot::Exact)
                .unwrap_or(BoundedCodeOracleSlot::Unknown)
        } else {
            store
                .oracle_coordinates
                .get(residual as usize)
                .copied()
                .unwrap_or(BoundedCodeOracleSlot::Unknown)
        };
        let target = store.arena.step(residual, byte)?;
        if store.arena.is_empty(target) {
            return None;
        }
        if store.oracle_coordinates.len() < store.arena.state_count() {
            store.oracle_coordinates.resize(
                store.arena.state_count(),
                BoundedCodeOracleSlot::Unknown,
            );
            store.oracle_futures.resize(store.arena.state_count(), None);
        }
        if let Some(oracle) = store.liveness_oracle.as_ref() {
            let target_slot = match source_coordinate {
                BoundedCodeOracleSlot::Exact(source_coordinate) => {
                    if let Some(target_coordinate) =
                        oracle.step_coordinate(source_coordinate, byte)
                    {
                        BoundedCodeOracleSlot::Exact(target_coordinate)
                    } else {
                        // A structurally non-empty residual can still denote
                        // the empty language. The generic dynamic lane falls
                        // back to the exact residual solver; the Static
                        // coordinate-preserving lane treats this as a violated
                        // construction invariant because it cannot project an
                        // unmodelled state into a precompiled TSID coordinate.
                        if self.preserve_oracle_coordinate {
                            panic!(
                                "coordinate-preserving bounded-code runtime lost its exact coordinate on byte {byte}"
                            );
                        }
                        BoundedCodeOracleSlot::Ambiguous
                    }
                }
                BoundedCodeOracleSlot::Ambiguous | BoundedCodeOracleSlot::Unknown => {
                    if self.preserve_oracle_coordinate {
                        panic!(
                            "coordinate-preserving bounded-code runtime reached a state without an exact source coordinate"
                        );
                    }
                    BoundedCodeOracleSlot::Ambiguous
                }
            };

            if self.preserve_oracle_coordinate {
                let BoundedCodeOracleSlot::Exact(target_coordinate) = target_slot else {
                    unreachable!("coordinate-preserving target is exact or panics above");
                };
                return self.intern_locked(store, target, Some(target_coordinate));
            }

            let target_index = target as usize;
            let slot = &mut store.oracle_coordinates[target_index];
            let previous = *slot;
            *slot = match (previous, target_slot) {
                (BoundedCodeOracleSlot::Unknown, next) => next,
                (BoundedCodeOracleSlot::Exact(existing), BoundedCodeOracleSlot::Exact(next))
                    if existing == next =>
                {
                    BoundedCodeOracleSlot::Exact(existing)
                }
                (BoundedCodeOracleSlot::Ambiguous, _)
                | (_, BoundedCodeOracleSlot::Ambiguous)
                | (BoundedCodeOracleSlot::Exact(_), BoundedCodeOracleSlot::Exact(_)) => {
                    BoundedCodeOracleSlot::Ambiguous
                }
                (existing, BoundedCodeOracleSlot::Unknown) => existing,
            };
            if *slot != previous {
                store.oracle_futures[target_index] = None;
            }
        }
        self.intern_locked(store, target, None)
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        if state == self.root_state && !self.root_has_future {
            return None;
        }
        let mut store = self.store.lock().unwrap();
        if self.coordinate_runtime {
            let source_coordinate = store.coordinate_by_state.get(&state).copied()?;
            let target_coordinate = store
                .liveness_oracle
                .as_ref()?
                .step_coordinate(source_coordinate, byte)?;
            return self.intern_oracle_coordinate_locked(&mut store, target_coordinate);
        }
        static PERSIST_TRANSITIONS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let persist_transitions = *PERSIST_TRANSITIONS.get_or_init(|| {
            std::env::var_os("GLRMASK_EXPERIMENT_PERSIST_VIRTUAL_RESIDUAL_TRANSITIONS")
                .is_some()
        });
        const TRANSITION_UNKNOWN: u32 = u32::MAX;
        const TRANSITION_DEAD: u32 = u32::MAX - 1;
        if persist_transitions
            && let Some(row) = store.transition_rows_by_state.get(&state)
        {
            let cached = row[byte as usize];
            if cached != TRANSITION_UNKNOWN {
                return (cached != TRANSITION_DEAD).then_some(cached);
            }
        }
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        let target = self.step_residual_locked(&mut store, state, residual, byte);
        if persist_transitions {
            let row = store
                .transition_rows_by_state
                .entry(state)
                .or_insert_with(|| Box::new([TRANSITION_UNKNOWN; 256]));
            row[byte as usize] = target.unwrap_or(TRANSITION_DEAD);
        }
        target
    }

    pub(super) fn direct_coordinate_for_state(
        &self,
        state: u32,
    ) -> Option<VirtualResidualDirectCoordinate> {
        if !self.coordinate_runtime || self.direct_coordinate_stepper.is_none() {
            return None;
        }
        let store = self.store.lock().ok()?;
        let coordinate = store.coordinate_by_state.get(&state).copied()?;
        Some(VirtualResidualDirectCoordinate {
            runtime_index: self.runtime_index,
            coordinate,
        })
    }

    #[inline(always)]
    pub(super) fn direct_coordinate_step(
        &self,
        source: VirtualResidualDirectCoordinate,
        byte: u8,
    ) -> Option<VirtualResidualDirectCoordinate> {
        if source.runtime_index != self.runtime_index {
            return None;
        }
        let coordinate = self
            .direct_coordinate_stepper
            .as_ref()?
            .step(source.coordinate, byte)?;
        Some(VirtualResidualDirectCoordinate {
            runtime_index: self.runtime_index,
            coordinate,
        })
    }

    #[inline(always)]
    pub(super) fn direct_coordinate_accepting(
        &self,
        source: VirtualResidualDirectCoordinate,
    ) -> Option<bool> {
        (source.runtime_index == self.runtime_index).then(|| {
            self.direct_coordinate_stepper
                .as_ref()
                .is_some_and(|stepper| stepper.accepting(source.coordinate))
        })
    }

    pub(super) fn direct_coordinate_has_future(
        &self,
        source: VirtualResidualDirectCoordinate,
    ) -> Option<bool> {
        if source.runtime_index != self.runtime_index {
            return None;
        }
        let mut store = self.store.lock().ok()?;
        let boundary_future = store
            .body_boundary_future_by_completed
            .as_ref()
            .map(Arc::clone);
        let oracle = store.liveness_oracle.as_mut()?;
        Some(
            boundary_future
                .as_deref()
                .and_then(|rows| {
                    oracle.has_future_with_boundary_table(source.coordinate, rows)
                })
                .unwrap_or_else(|| oracle.has_future(source.coordinate)),
        )
    }

    /// Return the sound pre-minimization finite-mask coordinate for one exact
    /// bounded-code residual. This is the same count-collapse stencil used by
    /// `build_finite_mask_projection`, but it does not materialize the DFA or
    /// its minimized state mapping. Equal returned coordinates therefore imply
    /// equal one-model-token behavior, although unequal coordinates may still
    /// be equivalent after minimization.
    pub(super) fn direct_coordinate_finite_mask_dense_key(
        &self,
        source: VirtualResidualDirectCoordinate,
        max_token_len: usize,
    ) -> Option<(u32, u32)> {
        if source.runtime_index != self.runtime_index || max_token_len == 0 {
            return None;
        }
        let store = self.store.lock().ok()?;
        let oracle = store.liveness_oracle.as_ref()?;
        let minimum_body_width = oracle.body.min_match_byte_len()?.max(1);
        let crossed_boundaries = max_token_len
            .div_ceil(minimum_body_width)
            .saturating_add(1);
        let crossed_boundaries = oracle.finite_mask_stencil_crossings(crossed_boundaries)?;
        self.direct_coordinate_dense_key_with_stencil(source, oracle, crossed_boundaries)
    }

    fn direct_coordinate_dense_key_with_stencil(
        &self,
        source: VirtualResidualDirectCoordinate,
        oracle: &BoundedCodeIntersectionOracle,
        crossed_boundaries: usize,
    ) -> Option<(u32, u32)> {
        if oracle.min > crossed_boundaries.saturating_add(1) {
            return None;
        }
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)?
            .checked_add(1)?;
        let mask_max = oracle.max.min(desired_mask_max);

        let pattern_states = oracle.pattern.num_states();
        let body_states = oracle.body.num_states();
        let prefix_len = oracle.prefix.len();
        let suffix_len = oracle.suffix.len();
        let pattern = source.coordinate.pattern_state as usize;
        if pattern >= pattern_states {
            return None;
        }
        let prefix_block = prefix_len.checked_mul(pattern_states)?;
        let body_layer = body_states.checked_mul(pattern_states)?;
        let body_block = mask_max.checked_add(1)?.checked_mul(body_layer)?;
        let suffix_slots = suffix_len.saturating_sub(1);
        let suffix_block = suffix_slots.checked_mul(pattern_states)?;
        let local = match source.coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if next >= prefix_len {
                    return None;
                }
                next.checked_mul(pattern_states)?.checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                let body_state = body_state as usize;
                if body_state >= body_states || completed > oracle.max {
                    return None;
                }
                let distance_to_upper = oracle.max - completed;
                let mapped_completed = if completed < oracle.min {
                    completed
                } else if distance_to_upper <= crossed_boundaries {
                    mask_max.checked_sub(distance_to_upper)?
                } else {
                    oracle.min
                };
                if mapped_completed > mask_max {
                    return None;
                }
                prefix_block
                    .checked_add(mapped_completed.checked_mul(body_layer)?)?
                    .checked_add(body_state.checked_mul(pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if next == 0 || next >= suffix_len {
                    return None;
                }
                prefix_block
                    .checked_add(body_block)?
                    .checked_add((next - 1).checked_mul(pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Done => prefix_block
                .checked_add(body_block)?
                .checked_add(suffix_block)?
                .checked_add(pattern)?,
        };
        Some((self.runtime_index, u32::try_from(local).ok()?))
    }

    /// Prove that every live string in a finite byte-language slice can be
    /// consumed from this exact residual without losing the current terminal.
    /// The slice is supplied in byte-class form so the product explores only
    /// distinct `(slice class, residual class)` transitions.
    pub(super) fn direct_coordinate_parser_transparent_byte_dfa(
        &self,
        source: VirtualResidualDirectCoordinate,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_can_reach_accepting: &[bool],
        slice_language_finite: bool,
        work_limit: usize,
    ) -> Option<bool> {
        let slice_state_count = slice_can_reach_accepting.len();
        if source.runtime_index != self.runtime_index
            || slice_state_count == 0
            || slice_class_count == 0
            || slice_class_count > 256
            || slice_start as usize >= slice_state_count
            || slice_transitions.len() != slice_state_count.checked_mul(slice_class_count)?
            || slice_byte_to_class
                .iter()
                .any(|&class| class as usize >= slice_class_count)
        {
            return None;
        }

        let mut store = self.store.lock().ok()?;
        if store.oracle_language_finite == Some(true) && !slice_language_finite {
            return Some(false);
        }
        let oracle_byte_to_class = store.oracle_byte_to_class.as_ref()?;
        let oracle_class_count = oracle_byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |class| class as usize + 1);
        if oracle_class_count == 0 {
            return None;
        }
        let mut pair_seen = vec![false; slice_class_count * oracle_class_count];
        let mut representatives = Vec::<u8>::new();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let pair = slice_byte_to_class[byte as usize] as usize * oracle_class_count
                + oracle_byte_to_class[byte as usize] as usize;
            if !pair_seen[pair] {
                pair_seen[pair] = true;
                representatives.push(byte);
            }
        }

        // Compact loaded oracles transfer this exact backwards-liveness table
        // instead of their relation powers. Direct-coordinate proofs must use
        // the same representation-aware query as state-indexed slice proofs.
        let body_boundary_future = store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let oracle = store.liveness_oracle.as_mut()?;
        let mut future_cache = FxHashMap::<BoundedCodeOracleCoordinate, bool>::default();
        let mut seen = FxHashSet::<(u32, BoundedCodeOracleCoordinate)>::default();
        let mut queue = VecDeque::from([(slice_start, source.coordinate)]);
        let mut work = 0usize;
        while let Some((slice_state, coordinate)) = queue.pop_front() {
            if !seen.insert((slice_state, coordinate)) {
                continue;
            }
            let row = (slice_state as usize).checked_mul(slice_class_count)?;
            for &byte in &representatives {
                let class = slice_byte_to_class[byte as usize] as usize;
                let slice_target = *slice_transitions.get(row + class)?;
                if slice_target as usize >= slice_state_count
                    || !slice_can_reach_accepting[slice_target as usize]
                {
                    continue;
                }
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let Some(target) = oracle.step_coordinate(coordinate, byte) else {
                    return Some(false);
                };
                let target_future = if oracle.coordinate_accepting(target) {
                    true
                } else if let Some(&future) = future_cache.get(&target) {
                    future
                } else {
                    let future = oracle.has_future_for_slice_proof(
                        target, body_boundary_future.as_deref().map(Vec::as_slice),
                    )?;
                    future_cache.insert(target, future);
                    future
                };
                if !target_future {
                    return Some(false);
                }
                if !seen.contains(&(slice_target, target)) {
                    queue.push_back((slice_target, target));
                }
            }
        }
        Some(true)
    }

    pub(super) fn state_for_direct_coordinate(
        &self,
        source: VirtualResidualDirectCoordinate,
    ) -> Option<u32> {
        if source.runtime_index != self.runtime_index || !self.coordinate_runtime {
            return None;
        }
        let mut store = self.store.lock().ok()?;
        self.intern_oracle_coordinate_locked(&mut store, source.coordinate)
    }

    fn certified_oracle_future(
        store: &mut ResidualRuntimeStore,
        residual: ResidualId,
    ) -> Option<bool> {
        let index = residual as usize;
        let BoundedCodeOracleSlot::Exact(coordinate) = store
            .oracle_coordinates
            .get(index)
            .copied()
            .unwrap_or(BoundedCodeOracleSlot::Unknown)
        else {
            return None;
        };
        if let Some(cached) = store.oracle_futures.get(index).copied().flatten() {
            return Some(cached);
        }
        let boundary_future = store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let oracle = store.liveness_oracle.as_mut()?;
        let future = boundary_future
            .as_deref()
            .and_then(|rows| oracle.has_future_with_boundary_table(coordinate, rows))
            .unwrap_or_else(|| oracle.has_future(coordinate));
        store.oracle_futures[index] = Some(future);
        Some(future)
    }

    fn certified_oracle_future_for_state(
        &self,
        store: &mut ResidualRuntimeStore,
        state: u32,
        residual: ResidualId,
    ) -> Option<bool> {
        if !self.preserve_oracle_coordinate {
            return Self::certified_oracle_future(store, residual);
        }
        if let Some(&future) = store.oracle_future_by_state.get(&state) {
            return Some(future);
        }
        let coordinate = *store.coordinate_by_state.get(&state)?;
        let boundary_future = store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let oracle = store.liveness_oracle.as_mut()?;
        let future = boundary_future
            .as_deref()
            .and_then(|rows| oracle.has_future_with_boundary_table(coordinate, rows))
            .unwrap_or_else(|| oracle.has_future(coordinate));
        store.oracle_future_by_state.insert(state, future);
        Some(future)
    }

    fn observation(&self, state: u32) -> Option<(bool, bool)> {
        let mut store = self.store.lock().unwrap();
        if self.coordinate_runtime {
            let coordinate = *store.coordinate_by_state.get(&state)?;
            let accepting = state != self.root_state
                && store
                    .liveness_oracle
                    .as_ref()?
                    .coordinate_accepting(coordinate);
            let future = Self::coordinate_future_locked(&mut store, state, coordinate)?;
            return Some((accepting, future));
        }
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        // Match the existing virtual-runtime convention: the physical proxy
        // root is the drained zero-byte configuration and must not emit a
        // terminal match before any input is consumed.
        let accepting = state != self.root_state && store.arena.is_nullable(residual);
        // A certified bounded-code coordinate has an infallible exact future
        // query once its oracle has been constructed. Surface that answer
        // through the ordinary tokenizer metadata path as well as through
        // `exact_has_future`: dynamic mask/commit contain several generic
        // consumers of `possible_future_terminals`, and those consumers should
        // observe the same exact liveness bit for this certified family.
        //
        // Unknown/ambiguous coordinates deliberately retain the old
        // conservative contract. Their exact query remains fallible and is
        // resolved only at the explicit dynamic residual boundary.
        let future = if let Some(future) =
            self.certified_oracle_future_for_state(&mut store, state, residual)
        {
            future
        } else if state == self.root_state {
            self.root_has_future
        } else {
            store.arena.conservative_has_future(residual)
        };
        Some((accepting, future))
    }

    /// Return only the accepting-now bit for a virtual residual state.
    ///
    /// Callers that only need terminal-finalizer metadata must not pay for the
    /// independent future-liveness observation.  In bounded-code residuals the
    /// latter may invoke an exact oracle query and populate its cache, which is
    /// materially more expensive than the nullable test used for acceptance.
    #[inline]
    fn accepting_now(&self, state: u32) -> Option<bool> {
        let store = self.store.lock().unwrap();
        if self.coordinate_runtime {
            let coordinate = *store.coordinate_by_state.get(&state)?;
            return Some(
                state != self.root_state
                    && store
                        .liveness_oracle
                        .as_ref()?
                        .coordinate_accepting(coordinate),
            );
        }
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        Some(state != self.root_state && store.arena.is_nullable(residual))
    }

    pub(super) fn vocabulary_repeat_boundary_horizon(
        &self,
        vocab: &Vocab,
        cache: &VocabularyRepeatHorizonCache,
    ) -> Option<usize> {
        let store = self.store.lock().ok()?;
        let oracle = store.liveness_oracle.as_ref()?;
        cache.horizon_for_dfa(oracle.body.as_ref(), vocab)
    }

    pub(super) fn serialized_bounded_code_oracle(&self) -> Vec<u8> {
        let store = self.store.lock().unwrap();
        let oracle = store
            .liveness_oracle
            .as_ref()
            .expect("Static residual projection requires bounded-code oracle");
        let sparse = SparseBoundedCodeOracleWire::from_oracle(oracle)
            .expect("bounded-code oracle is within sparse wire limits");
        let mut out = Vec::new();
        out.extend_from_slice(&SPARSE_BOUNDED_CODE_ORACLE_MAGIC);
        bincode::serialize_into(&mut out, &sparse)
            .expect("bounded-code sparse oracle serialization should succeed");
        out
    }

    pub(super) fn new_from_oracle_bytes(
        expr: &Expr,
        oracle_bytes: &[u8],
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        let mut runtime = Self::new_preserving_oracle_coordinate_from_oracle_bytes(
            expr,
            oracle_bytes,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
        )?;
        runtime.preserve_oracle_coordinate = false;
        let store = runtime.store.get_mut().ok()?;
        store.state_by_residual_coordinate.clear();
        store.coordinate_by_state.clear();
        store.oracle_future_by_state.clear();
        Some(runtime)
    }

    pub(super) fn new_preserving_oracle_coordinate_from_oracle_bytes(
        expr: &Expr,
        oracle_bytes: &[u8],
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        if terminal >= num_terminals
            || physical_state_count == 0
            || root_state >= physical_state_count
            || state_owners.owner_index(root_state) != Some(runtime_index as usize)
        {
            return None;
        }
        let (mut arena, root) = ResidualArena::from_expr(expr)?;
        let liveness_oracle: BoundedCodeIntersectionOracle = if let Some(body) =
            oracle_bytes.strip_prefix(&SPARSE_BOUNDED_CODE_ORACLE_MAGIC)
        {
            bincode::deserialize::<SparseBoundedCodeOracleWire>(body)
                .ok()?
                .into_oracle()?
        } else {
            bincode::deserialize(oracle_bytes).ok()?
        };
        let root_oracle_coordinate = liveness_oracle.root_coordinate();
        // Structural checks sufficient for compiled-artifact restoration. The
        // exact residual derivative remains authoritative for transitions and
        // will fail closed if a coordinate ceases to track it.
        if liveness_oracle.pattern.num_states() == 0
            || liveness_oracle.pattern.num_states() > MAX_BOUNDED_CODE_ORACLE_PATTERN_STATES
            || liveness_oracle.body.num_states() == 0
            || liveness_oracle.body_productive.len() != liveness_oracle.body.num_states()
            || liveness_oracle.completion_relations.len() != liveness_oracle.body.num_states()
            || liveness_oracle.min > liveness_oracle.max
            || liveness_oracle.prefix.is_empty()
            || liveness_oracle.suffix.is_empty()
        {
            return None;
        }
        let pattern_states = liveness_oracle.pattern.num_states();
        let relation_valid = |relation: &BoolRelation| relation.valid_for(pattern_states);
        if liveness_oracle
            .completion_relations
            .iter()
            .flatten()
            .any(|relation| !relation_valid(relation))
            || liveness_oracle.exact_powers.iter().any(|relation| !relation_valid(relation))
            || liveness_oracle.prefix_sums.iter().any(|relation| !relation_valid(relation))
        {
            return None;
        }
        let root_live = arena.conservative_has_future(root);
        let mut state_by_residual = vec![u32::MAX; root as usize + 1];
        state_by_residual[root as usize] = root_state;
        let mut oracle_coordinates = vec![BoundedCodeOracleSlot::Unknown; arena.state_count()];
        let oracle_futures = vec![None; arena.state_count()];
        oracle_coordinates[root as usize] = BoundedCodeOracleSlot::Exact(root_oracle_coordinate);
        let mut state_by_residual_coordinate = FxHashMap::default();
        let mut coordinate_by_state = FxHashMap::default();
        state_by_residual_coordinate.insert((root, root_oracle_coordinate), root_state);
        coordinate_by_state.insert(root_state, root_oracle_coordinate);
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        let oracle_byte_to_class = Some(liveness_oracle.byte_to_class_map());
        let oracle_language_finite = Some(dfa_language_is_finite(&liveness_oracle.body));
        Some(Self {
            runtime_index, terminal, physical_state_count, root_state, root_has_future: root_live,
            preserve_oracle_coordinate: true, coordinate_runtime: false,
            direct_coordinate_stepper: None,
            state_allocator, state_owners, accepting, live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
            store: Mutex::new(ResidualRuntimeStore {
                arena, root, state_by_residual, residual_by_state: FxHashMap::default(),
                state_by_oracle_coordinate: FxHashMap::default(),
                state_by_residual_coordinate, coordinate_by_state,
                oracle_future_by_state: FxHashMap::default(),
                liveness_oracle: Some(liveness_oracle), oracle_byte_to_class,
                oracle_language_finite, oracle_coordinates, oracle_futures,
                parser_transparent_byte_family_cache: FxHashSet::default(),
                slice_atom_body_exact_cache: FxHashMap::default(),
                slice_atom_pattern_targets_cache: FxHashMap::default(),
                body_boundary_future_by_completed: None,
                transition_rows_by_state: FxHashMap::default(),
            }),
        })
    }

    pub(super) fn root_has_future(&self) -> bool {
        self.root_has_future
    }

    /// Exact nonempty-continuation query for a state owned by this runtime.
    /// Resource exhaustion is propagated to dynamic mask/commit instead of
    /// being collapsed into a dead transition.
    pub(super) fn exact_has_future(&self, state: u32) -> Result<Option<bool>, String> {
        let mut store = self.store.lock().unwrap();
        if self.coordinate_runtime {
            let Some(&coordinate) = store.coordinate_by_state.get(&state) else {
                return Ok(None);
            };
            return Ok(Self::coordinate_future_locked(&mut store, state, coordinate));
        }
        let Some(residual) = Self::residual_for_state(&store, self.root_state, state) else {
            return Ok(None);
        };
        if let Some(future) =
            self.certified_oracle_future_for_state(&mut store, state, residual)
        {
            return Ok(Some(future));
        }
        store.arena.has_future(residual).map(Some)
    }

    pub(super) fn has_bounded_code_liveness_oracle(&self) -> bool {
        self.store.lock().unwrap().liveness_oracle.is_some()
    }

    /// Prove that every string over `bytes` of length `1..=max_horizon`
    /// remains inside this residual terminal without finalizing and still has
    /// a nonempty continuation. This is intentionally stronger than proving
    /// only the concrete vocabulary strings: callers can certify an entire
    /// vocabulary partition from its byte alphabet and maximum token length.
    ///
    /// The proof runs directly on the bounded-code oracle coordinate and does
    /// not materialize the finite one-token mask DFA. `None` means the state
    /// has no exact oracle coordinate or the bounded work budget was exceeded;
    /// callers must fall back to the exact vocabulary walk in that case.
    /// Exact containment of a byte-DFA slice in this bounded-code residual.
    /// The supplied DFA is a language view owned by the caller; only strings
    /// whose target state can still reach acceptance participate in the proof.
    /// This mirrors the finite tokenizer-product containment check, but advances
    /// the symbolic bounded-code oracle coordinate directly instead of walking a
    /// finite mask projection.
    pub(super) fn parser_transparent_byte_dfa(
        &self,
        state: u32,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_can_reach_accepting: &[bool],
        slice_language_finite: bool,
        work_limit: usize,
    ) -> Option<bool> {
        let slice_state_count = slice_can_reach_accepting.len();
        if slice_state_count == 0
            || slice_class_count == 0
            || slice_class_count > 256
            || slice_start as usize >= slice_state_count
            || slice_transitions.len() != slice_state_count.checked_mul(slice_class_count)?
            || slice_byte_to_class
                .iter()
                .any(|&class| class as usize >= slice_class_count)
            || !self.handles_state(state)
        {
            return None;
        }

        let mut store = self.store.lock().unwrap();
        let coordinate = self.oracle_coordinate_for_state_locked(&store, state)?;
        if store.oracle_language_finite == Some(true) && !slice_language_finite {
            return Some(false);
        }
        let oracle_byte_to_class = store.oracle_byte_to_class.as_ref()?;
        let oracle_class_count = oracle_byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |class| class as usize + 1);
        if oracle_class_count == 0 {
            return None;
        }
        let mut pair_seen = vec![false; slice_class_count * oracle_class_count];
        let mut representatives = Vec::<u8>::new();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let pair = slice_byte_to_class[byte as usize] as usize * oracle_class_count
                + oracle_byte_to_class[byte as usize] as usize;
            if !pair_seen[pair] {
                pair_seen[pair] = true;
                representatives.push(byte);
            }
        }
        let body_boundary_future = store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let oracle = store.liveness_oracle.as_mut()?;
        let mut future_cache = FxHashMap::<BoundedCodeOracleCoordinate, bool>::default();
        let mut seen = FxHashSet::<(u32, BoundedCodeOracleCoordinate)>::default();
        let mut queue = VecDeque::from([(slice_start, coordinate)]);
        let mut work = 0usize;
        while let Some((slice_state, coordinate)) = queue.pop_front() {
            if !seen.insert((slice_state, coordinate)) {
                continue;
            }
            let row = (slice_state as usize).checked_mul(slice_class_count)?;
            for &byte in &representatives {
                let class = slice_byte_to_class[byte as usize] as usize;
                let slice_target = *slice_transitions.get(row + class)?;
                if slice_target as usize >= slice_state_count
                    || !slice_can_reach_accepting[slice_target as usize]
                {
                    continue;
                }
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }
                let Some(target) = oracle.step_coordinate(coordinate, byte) else {
                    return Some(false);
                };
                let target_future = if oracle.coordinate_accepting(target) {
                    true
                } else if let Some(&future) = future_cache.get(&target) {
                    future
                } else {
                    let future = oracle.has_future_for_slice_proof(
                        target, body_boundary_future.as_deref().map(Vec::as_slice),
                    )?;
                    future_cache.insert(target, future);
                    future
                };
                if !target_future {
                    return Some(false);
                }
                if !seen.contains(&(slice_target, target)) {
                    queue.push_back((slice_target, target));
                }
            }
        }
        Some(true)
    }

    /// Return the largest repetition bound `r <= max_repetitions` for
    /// which every word in the caller's `slice+` language with at most `r`
    /// completed slice atoms remains a valid prefix of this exact bounded-code
    /// residual. The slice DFA is expected to accept after each complete atom
    /// (as the llguidance safe+ DFA does); UTF-8 continuation states are
    /// non-accepting and therefore contribute zero to the repetition count.
    ///
    /// This is the same exact product as `parser_transparent_byte_dfa`, but it
    /// finds the first counterexample repetition count in one 0/1 BFS instead
    /// of separately proving several `{1,n}` DFAs.
    pub(super) fn prepare_master_slice_artifacts(
        &self,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
        max_repetitions: u32,
    ) {
        let slice_state_count = slice_can_reach_accepting.len();
        if max_repetitions == 0
            || slice_state_count == 0
            || slice_accepting.len() != slice_state_count
            || slice_class_count == 0
            || slice_class_count > 256
            || slice_start as usize >= slice_state_count
            || slice_transitions.len() != slice_state_count.checked_mul(slice_class_count).unwrap_or(0)
            || slice_byte_to_class.iter().any(|&class| class as usize >= slice_class_count)
            || slice_accepting.get(slice_start as usize).copied().unwrap_or(false)
        {
            return;
        }

        let mut store = self.store.lock().unwrap();
        if store.liveness_oracle.is_none() {
            return;
        }

        if store.body_boundary_future_by_completed.is_none() {
            let started = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES")
                .is_some()
                .then(std::time::Instant::now);
            let built = store
                .liveness_oracle
                .as_ref()
                .and_then(BoundedCodeIntersectionOracle::body_boundary_future_sets)
                .map(Arc::new);
            if let Some(started) = started {
                let oracle = store.liveness_oracle.as_ref();
                let (rows, words, set_bits) = built.as_ref().map_or((0, 0, 0), |rows| {
                    (
                        rows.len(),
                        rows.iter().map(|row| row.words().len()).sum::<usize>(),
                        rows.iter().map(BitSet::count_ones).sum::<usize>(),
                    )
                });
                eprintln!(
                    "[glrmask/profile][virtual_radius_future_table_prep] terminal={} pattern_states={} max={} built={} rows={} words={} raw_word_bytes={} set_bits={} ms={:.3}",
                    self.terminal,
                    oracle.map_or(0, |oracle| oracle.pattern.num_states()),
                    oracle.map_or(0, |oracle| oracle.max),
                    built.is_some(),
                    rows,
                    words,
                    words * std::mem::size_of::<u64>(),
                    set_bits,
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            if built.is_some() {
                store.body_boundary_future_by_completed = built;
            }
        }

        let slice_atom_fingerprint = {
            let mut hasher = rustc_hash::FxHasher::default();
            slice_start.hash(&mut hasher);
            slice_class_count.hash(&mut hasher);
            slice_byte_to_class.hash(&mut hasher);
            slice_transitions.hash(&mut hasher);
            slice_accepting.hash(&mut hasher);
            slice_can_reach_accepting.hash(&mut hasher);
            hasher.finish()
        };

        let atom_is_exact_body_code = if let Some(&cached) =
            store.slice_atom_body_exact_cache.get(&slice_atom_fingerprint)
        {
            cached
        } else {
            let certified = store
                .liveness_oracle
                .as_ref()
                .is_some_and(|oracle| {
                    oracle.slice_atom_is_exact_body_code(
                        slice_start,
                        slice_class_count,
                        slice_byte_to_class,
                        slice_transitions,
                        slice_accepting,
                        slice_can_reach_accepting,
                    )
                });
            store
                .slice_atom_body_exact_cache
                .insert(slice_atom_fingerprint, certified);
            certified
        };

        if !atom_is_exact_body_code {
            return;
        }

        let body_boundary_future =
            store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let ResidualRuntimeStore {
            liveness_oracle,
            slice_atom_pattern_targets_cache,
            ..
        } = &mut *store;
        let Some(oracle) = liveness_oracle.as_ref() else {
            return;
        };

        let mut coord = oracle.root_coordinate();
        let mut prefix_valid = true;
        for &byte in oracle.prefix.iter() {
            if let Some(next) = oracle.step_coordinate(coord, byte) {
                coord = next;
            } else {
                prefix_valid = false;
                break;
            }
        }

        if prefix_valid
            && matches!(
                coord.envelope,
                BoundedCodeEnvelopeState::Body { body_state: 0, .. }
            )
        {
            let _ = oracle.uniform_slice_repeat_radius(
                coord,
                atom_is_exact_body_code,
                slice_atom_fingerprint,
                slice_atom_pattern_targets_cache,
                body_boundary_future.as_deref().map(Vec::as_slice),
                slice_start,
                slice_class_count,
                slice_byte_to_class,
                slice_transitions,
                slice_accepting,
                slice_can_reach_accepting,
                max_repetitions,
            );

            for byte in 0u16..=255 {
                let byte = byte as u8;
                let class = slice_byte_to_class[byte as usize] as usize;
                let row = slice_start as usize * slice_class_count;
                if let Some(&slice_target) = slice_transitions.get(row + class)
                    && slice_accepting.get(slice_target as usize).copied().unwrap_or(false)
                    && let Some(stepped) = oracle.step_coordinate(coord, byte)
                    && matches!(
                        stepped.envelope,
                        BoundedCodeEnvelopeState::Body { body_state: 0, .. }
                    )
                {
                    let _ = oracle.uniform_slice_repeat_radius(
                        stepped,
                        atom_is_exact_body_code,
                        slice_atom_fingerprint,
                        slice_atom_pattern_targets_cache,
                        body_boundary_future.as_deref().map(Vec::as_slice),
                        slice_start,
                        slice_class_count,
                        slice_byte_to_class,
                        slice_transitions,
                        slice_accepting,
                        slice_can_reach_accepting,
                        max_repetitions,
                    );
                    break;
                }
            }
        }

    }

    /// Upper bound on complete atoms consumed before this residual must use
    /// its literal closing suffix. The atom language must coincide with one
    /// body code word, and must be unable to start that suffix. Unlike a
    /// containment radius, this is a negative certificate: longer whole-atom
    /// words cannot remain in the terminal or finalize it and change parser.
    pub(super) fn safe_atom_length_upper_bound(
        &self,
        state: u32,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
    ) -> Option<u32> {
        if slice_class_count == 0 || slice_class_count > 256
            || slice_start as usize >= slice_accepting.len()
            || slice_accepting.len() != slice_can_reach_accepting.len()
            || slice_transitions.len() != slice_accepting.len().checked_mul(slice_class_count)?
            || slice_accepting[slice_start as usize]
            || slice_byte_to_class.iter().any(|&x| x as usize >= slice_class_count)
        { return None; }
        let mut store = self.store.lock().unwrap();
        let coordinate = self.oracle_coordinate_for_state_locked(&store, state)?;
        let BoundedCodeEnvelopeState::Body { completed, body_state: 0 } = coordinate.envelope
        else { return None; };
        let oracle = store.liveness_oracle.as_ref()?;
        let closing_first = *oracle.suffix.first()?;
        let next = slice_transitions[(slice_start as usize)*slice_class_count
            + slice_byte_to_class[closing_first as usize] as usize];
        if slice_can_reach_accepting.get(next as usize).copied().unwrap_or(false) {
            return None;
        }
        let remaining = oracle.max.checked_sub(completed)?;
        let fingerprint = {
            let mut hash = rustc_hash::FxHasher::default();
            slice_start.hash(&mut hash);
            slice_class_count.hash(&mut hash);
            slice_byte_to_class.hash(&mut hash);
            slice_transitions.hash(&mut hash);
            slice_accepting.hash(&mut hash);
            slice_can_reach_accepting.hash(&mut hash);
            hash.finish()
        };
        let exact_atom = if let Some(&v) = store.slice_atom_body_exact_cache.get(&fingerprint) {
            v
        } else {
            let v = store.liveness_oracle.as_ref()?.slice_atom_is_exact_body_code(
                slice_start, slice_class_count, slice_byte_to_class, slice_transitions,
                slice_accepting, slice_can_reach_accepting,
            );
            store.slice_atom_body_exact_cache.insert(fingerprint, v);
            v
        };
        exact_atom.then_some(u32::try_from(remaining).unwrap_or(u32::MAX))
    }

    pub(super) fn parser_transparent_byte_dfa_repeat_radius(
        &self,
        state: u32,
        slice_start: u32,
        slice_class_count: usize,
        slice_byte_to_class: &[u8; 256],
        slice_transitions: &[u32],
        slice_accepting: &[bool],
        slice_can_reach_accepting: &[bool],
        max_repetitions: u32,
        work_limit: usize,
    ) -> Option<u32> {
        let slice_state_count = slice_can_reach_accepting.len();
        if max_repetitions == 0
            || slice_state_count == 0
            || slice_accepting.len() != slice_state_count
            || slice_class_count == 0
            || slice_class_count > 256
            || slice_start as usize >= slice_state_count
            || slice_transitions.len() != slice_state_count.checked_mul(slice_class_count)?
            || slice_byte_to_class
                .iter()
                .any(|&class| class as usize >= slice_class_count)
            || slice_accepting.get(slice_start as usize).copied().unwrap_or(false)
            || !self.handles_state(state)
        {
            return None;
        }

        let mut store = self.store.lock().unwrap();
        let coordinate = self.oracle_coordinate_for_state_locked(&store, state)?;

        let slice_atom_fingerprint = {
            let mut hasher = rustc_hash::FxHasher::default();
            slice_start.hash(&mut hasher);
            slice_class_count.hash(&mut hasher);
            slice_byte_to_class.hash(&mut hasher);
            slice_transitions.hash(&mut hasher);
            slice_accepting.hash(&mut hasher);
            slice_can_reach_accepting.hash(&mut hasher);
            hasher.finish()
        };
        let atom_is_exact_body_code = if let Some(&cached) =
            store.slice_atom_body_exact_cache.get(&slice_atom_fingerprint)
        {
            cached
        } else {
            let certified = store
                .liveness_oracle
                .as_ref()?
                .slice_atom_is_exact_body_code(
                    slice_start,
                    slice_class_count,
                    slice_byte_to_class,
                    slice_transitions,
                    slice_accepting,
                    slice_can_reach_accepting,
                );
            store
                .slice_atom_body_exact_cache
                .insert(slice_atom_fingerprint, certified);
            certified
        };
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
            eprintln!(
                "[glrmask/profile][virtual_radius_fast] state={} terminal={} envelope={:?} atom_body={} max_repeat={}",
                state,
                self.terminal,
                coordinate.envelope,
                atom_is_exact_body_code,
                max_repetitions,
            );
        }
        let at_body_boundary = atom_is_exact_body_code
            && matches!(
                coordinate.envelope,
                BoundedCodeEnvelopeState::Body { body_state: 0, .. }
            );
        // Before constructing `max+1` backwards-DP bitsets, cheaply ask only
        // whether the first complete slice atom already fails.  This detects a
        // zero radius without paying for the large future table, while avoiding
        // the expensive no-table search for genuinely large positive radii.
        let existing_body_boundary_future =
            store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let zero_probe_limit = max_repetitions.min(1);
        let zero_probe = {
            let ResidualRuntimeStore {
                liveness_oracle,
                slice_atom_pattern_targets_cache,
                ..
            } = &mut *store;
            liveness_oracle.as_ref()?.uniform_slice_repeat_radius(
                coordinate,
                atom_is_exact_body_code,
                slice_atom_fingerprint,
                slice_atom_pattern_targets_cache,
                existing_body_boundary_future.as_deref().map(Vec::as_slice),
                slice_start,
                slice_class_count,
                slice_byte_to_class,
                slice_transitions,
                slice_accepting,
                slice_can_reach_accepting,
                zero_probe_limit,
            )
        };
        if zero_probe == Some(0) || (max_repetitions <= 1 && zero_probe.is_some()) {
            let radius = zero_probe.expect("checked Some above");
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                eprintln!(
                    "[glrmask/profile][virtual_radius_fast] state={} terminal={} zero_probe_radius={}",
                    state,
                    self.terminal,
                    radius,
                );
            }
            return Some(radius);
        }

        // First establish that EVERY atom in the actual slice can be read.
        // If all successful body completions also return to the same pattern
        // coordinate, this total one-atom transition can then be repeated.
        // In particular, a last-word nonspace loop must not certify a space
        // that has no pattern transition at all.
        if let Some(radius) = store
            .liveness_oracle
            .as_ref()?
            .invariant_body_repeat_radius(
                coordinate,
                atom_is_exact_body_code,
                zero_probe == Some(1),
                max_repetitions,
            )
        {
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                eprintln!(
                    "[glrmask/profile][virtual_radius_fast] state={} terminal={} fast_radius={}",
                    state,
                    self.terminal,
                    radius,
                );
            }
            return Some(radius);
        }

        if at_body_boundary && store.body_boundary_future_by_completed.is_none() {
            let started = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES")
                .is_some()
                .then(std::time::Instant::now);
            let built = store
                .liveness_oracle
                .as_ref()
                .and_then(BoundedCodeIntersectionOracle::body_boundary_future_sets)
                .map(Arc::new);
            if let Some(started) = started {
                let oracle = store.liveness_oracle.as_ref();
                eprintln!(
                    "[glrmask/profile][virtual_radius_future_table] state={} terminal={} pattern_states={} max={} built={} ms={:.3}",
                    state,
                    self.terminal,
                    oracle.map_or(0, |oracle| oracle.pattern.num_states()),
                    oracle.map_or(0, |oracle| oracle.max),
                    built.is_some(),
                    started.elapsed().as_secs_f64() * 1e3,
                );
            }
            if built.is_some() {
                store.body_boundary_future_by_completed = built;
            }
        }
        let body_boundary_future_by_completed =
            store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let uniform_radius = {
            let ResidualRuntimeStore {
                liveness_oracle,
                slice_atom_pattern_targets_cache,
                ..
            } = &mut *store;
            liveness_oracle.as_ref()?.uniform_slice_repeat_radius(
                coordinate,
                atom_is_exact_body_code,
                slice_atom_fingerprint,
                slice_atom_pattern_targets_cache,
                body_boundary_future_by_completed.as_deref().map(Vec::as_slice),
                slice_start,
                slice_class_count,
                slice_byte_to_class,
                slice_transitions,
                slice_accepting,
                slice_can_reach_accepting,
                max_repetitions,
            )
        };
        if let Some(radius) = uniform_radius {
            if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_PROOF_PHASES").is_some() {
                eprintln!(
                    "[glrmask/profile][virtual_radius_fast] state={} terminal={} uniform_slice_radius={}",
                    state,
                    self.terminal,
                    radius,
                );
            }
            return Some(radius);
        }

        let oracle_byte_to_class = store.oracle_byte_to_class.as_ref()?;
        let oracle_class_count = oracle_byte_to_class
            .iter()
            .copied()
            .max()
            .map_or(0usize, |class| class as usize + 1);
        if oracle_class_count == 0 {
            return None;
        }
        let mut pair_seen = vec![false; slice_class_count * oracle_class_count];
        let mut representatives = Vec::<u8>::new();
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let pair = slice_byte_to_class[byte as usize] as usize * oracle_class_count
                + oracle_byte_to_class[byte as usize] as usize;
            if !pair_seen[pair] {
                pair_seen[pair] = true;
                representatives.push(byte);
            }
        }

        // Minimum additional completed slice atoms needed to reach acceptance
        // from every slice state. Edge cost is one exactly when the target DFA
        // state is accepting (a complete safe Unicode scalar for safe+).
        let mut reverse = vec![Vec::<(u32, u8)>::new(); slice_state_count];
        for source in 0..slice_state_count as u32 {
            let row = source as usize * slice_class_count;
            let mut seen_targets = FxHashSet::<u32>::default();
            for class in 0..slice_class_count {
                let target = *slice_transitions.get(row + class)?;
                if target as usize >= slice_state_count || !seen_targets.insert(target) {
                    continue;
                }
                reverse[target as usize].push((
                    source,
                    u8::from(slice_accepting[target as usize]),
                ));
            }
        }
        let mut min_to_accept = vec![u32::MAX; slice_state_count];
        let mut distance_queue = VecDeque::<u32>::new();
        for (slice_state, &accepting) in slice_accepting.iter().enumerate() {
            if accepting {
                min_to_accept[slice_state] = 0;
                distance_queue.push_back(slice_state as u32);
            }
        }
        while let Some(target) = distance_queue.pop_front() {
            let target_distance = min_to_accept[target as usize];
            for &(source, cost) in &reverse[target as usize] {
                let candidate = target_distance.saturating_add(u32::from(cost));
                if candidate < min_to_accept[source as usize] {
                    min_to_accept[source as usize] = candidate;
                    if cost == 0 {
                        distance_queue.push_front(source);
                    } else {
                        distance_queue.push_back(source);
                    }
                }
            }
        }

        let oracle = store.liveness_oracle.as_mut()?;
        let mut future_cache = FxHashMap::<BoundedCodeOracleCoordinate, bool>::default();
        let mut best = FxHashMap::<(u32, BoundedCodeOracleCoordinate), u32>::default();
        let mut queue = VecDeque::<(u32, BoundedCodeOracleCoordinate, u32)>::new();
        best.insert((slice_start, coordinate), 0);
        queue.push_back((slice_start, coordinate, 0));
        let mut work = 0usize;
        let mut first_counterexample = max_repetitions.saturating_add(1);

        while let Some((slice_state, coordinate, completed)) = queue.pop_front() {
            if best.get(&(slice_state, coordinate)).copied() != Some(completed) {
                continue;
            }
            if completed >= first_counterexample || completed > max_repetitions {
                continue;
            }
            let row = (slice_state as usize).checked_mul(slice_class_count)?;
            for &byte in &representatives {
                let class = slice_byte_to_class[byte as usize] as usize;
                let slice_target = *slice_transitions.get(row + class)?;
                if slice_target as usize >= slice_state_count
                    || !slice_can_reach_accepting[slice_target as usize]
                {
                    continue;
                }
                let completed_target = completed
                    .saturating_add(u32::from(slice_accepting[slice_target as usize]));
                let completion_cost = min_to_accept[slice_target as usize];
                if completion_cost == u32::MAX {
                    continue;
                }
                let shortest_complete_word = completed_target.saturating_add(completion_cost);
                if shortest_complete_word > max_repetitions {
                    continue;
                }
                work = work.saturating_add(1);
                if work > work_limit {
                    return None;
                }

                let target = oracle.step_coordinate(coordinate, byte);
                let target_live = if let Some(target) = target {
                    if oracle.coordinate_accepting(target) {
                        true
                    } else if let Some(&future) = future_cache.get(&target) {
                        future
                    } else {
                        let future = oracle.has_future_for_slice_proof(
                            target,
                            body_boundary_future_by_completed.as_deref().map(Vec::as_slice),
                        )?;
                        future_cache.insert(target, future);
                        future
                    }
                } else {
                    false
                };
                if !target_live {
                    first_counterexample = first_counterexample.min(shortest_complete_word);
                    continue;
                }
                let target = target.expect("live target must exist");
                if completed_target >= first_counterexample
                    || completed_target > max_repetitions
                {
                    continue;
                }
                let key = (slice_target, target);
                if completed_target < best.get(&key).copied().unwrap_or(u32::MAX) {
                    best.insert(key, completed_target);
                    if slice_accepting[slice_target as usize] {
                        queue.push_back((slice_target, target, completed_target));
                    } else {
                        queue.push_front((slice_target, target, completed_target));
                    }
                }
            }
        }

        let radius = first_counterexample
            .saturating_sub(1)
            .min(max_repetitions);
        Some(radius)
    }

    pub(super) fn parser_transparent_byte_family(
        &self,
        state: u32,
        bytes: U8Set,
        max_horizon: u32,
    ) -> Option<bool> {
        if bytes.is_empty() || max_horizon == 0 || !self.handles_state(state) {
            return Some(false);
        }

        const MAX_COORDINATE_BYTE_STEPS: usize = 128 * 1024;
        let key = (state, bytes, max_horizon);
        let mut store = self.store.lock().unwrap();
        if store.parser_transparent_byte_family_cache.contains(&key) {
            return Some(true);
        }
        let coordinate = self.oracle_coordinate_for_state_locked(&store, state)?;
        let body_boundary_future = store.body_boundary_future_by_completed.as_ref().map(Arc::clone);
        let result = 'proof: {
            let oracle = store.liveness_oracle.as_mut()?;
            if oracle.coordinate_accepting(coordinate)
                || !oracle.has_future_for_slice_proof(
                    coordinate, body_boundary_future.as_deref().map(Vec::as_slice),
                )?
            {
                break 'proof Some(false);
            }

            let mut frontier = FxHashSet::default();
            frontier.insert(coordinate);
            let mut work = 0usize;
            for _ in 0..max_horizon {
                let mut next = FxHashSet::default();
                for coordinate in frontier.drain() {
                    for byte in bytes.iter() {
                        work = work.saturating_add(1);
                        if work > MAX_COORDINATE_BYTE_STEPS {
                            break 'proof None;
                        }
                        let Some(target) = oracle.step_coordinate(coordinate, byte) else {
                            break 'proof Some(false);
                        };
                        next.insert(target);
                    }
                }
                if next.is_empty() {
                    break 'proof Some(false);
                }
                for &target in &next {
                    if oracle.coordinate_accepting(target)
                        || !oracle.has_future_for_slice_proof(
                            target, body_boundary_future.as_deref().map(Vec::as_slice),
                        )?
                    {
                        break 'proof Some(false);
                    }
                }
                frontier = next;
            }
            Some(true)
        };
        if result == Some(true) {
            // Virtual residual states may advance throughout a very long
            // bounded string. Keep successful proofs only (failed partitions
            // usually fail after one or a few byte steps) and bound retained
            // state history instead of growing with generation length.
            const MAX_TRANSPARENT_BYTE_FAMILY_CACHE_ENTRIES: usize = 8 * 1024;
            if store.parser_transparent_byte_family_cache.len()
                >= MAX_TRANSPARENT_BYTE_FAMILY_CACHE_ENTRIES
            {
                store.parser_transparent_byte_family_cache.clear();
            }
            store.parser_transparent_byte_family_cache.insert(key);
        }
        result
    }

    pub(super) fn finalizers(&self, state: u32) -> Option<&BitSet> {
        let accepting = self.accepting_now(state)?;
        Some(if accepting { &self.accepting } else { &self.dead })
    }

    pub(super) fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        let accepting = self.accepting_now(state)?;
        Some(if accepting { self.accepting_list.as_ref() } else { &[] })
    }

    pub(super) fn futures(&self, state: u32) -> Option<&BitSet> {
        let (_, future) = self.observation(state)?;
        Some(if future { &self.live } else { &self.dead })
    }

    pub(super) fn transitions(&self, state: u32) -> Option<Vec<(u8, u32)>> {
        if !self.handles_state(state) {
            return None;
        }
        if state == self.root_state && !self.root_has_future {
            return Some(Vec::new());
        }
        let mut store = self.store.lock().unwrap();
        if self.coordinate_runtime {
            let coordinate = *store.coordinate_by_state.get(&state)?;
            let mut out = Vec::new();
            for byte in 0u16..=255 {
                let Some(target_coordinate) = store
                    .liveness_oracle
                    .as_ref()?
                    .step_coordinate(coordinate, byte as u8)
                else {
                    continue;
                };
                let target = self.intern_oracle_coordinate_locked(&mut store, target_coordinate)?;
                out.push((byte as u8, target));
            }
            return Some(out);
        }
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        let bytes = store.arena.first_bytes(residual)?;
        let mut out = Vec::new();
        for byte in bytes.iter() {
            if let Some(target) = self.step_residual_locked(&mut store, state, residual, byte) {
                out.push((byte, target));
            }
        }
        Some(out)
    }

    fn oracle_coordinate(&self, state: u32) -> Option<BoundedCodeOracleCoordinate> {
        let store = self.store.lock().unwrap();
        self.oracle_coordinate_for_state_locked(&store, state)
    }

    pub(super) fn restore_compiled_finite_mask_projection(
        self: &Arc<Self>,
        component_state_count: u32,
        artifact: VirtualResidualMaskProjectionArtifact,
    ) -> Result<VirtualResidualMaskProjection, String> {
        if artifact.terminal != self.terminal {
            return Err(format!(
                "virtual residual projection terminal mismatch: artifact={} runtime={}",
                artifact.terminal, self.terminal,
            ));
        }
        if !self.preserve_oracle_coordinate {
            return Err("compiled virtual residual projection requires coordinate-preserving runtime".to_owned());
        }
        let store = self.store.lock().map_err(|_| "virtual residual runtime lock poisoned".to_owned())?;
        let oracle = store.liveness_oracle.as_ref().ok_or_else(|| "compiled virtual residual projection has no bounded-code oracle".to_owned())?;
        let mask_max = artifact.compiled_mask_max;
        let crossed_boundaries = artifact.compiled_crossed_boundaries;
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "compiled virtual residual projection stencil overflow".to_owned())?;
        let future_headroom = oracle.finite_mask_future_repeat_headroom()
            .ok_or_else(|| "compiled virtual residual projection future stencil overflow".to_owned())?;
        if crossed_boundaries < future_headroom || mask_max != oracle.max.min(desired_mask_max) {
            return Err("compiled virtual residual projection stencil is inconsistent".to_owned());
        }
        let expected_dense_states = oracle
            .finite_mask_dense_state_count(mask_max)
            .filter(|&states| states <= MAX_FINITE_MASK_DENSE_STATES)
            .ok_or_else(|| "compiled virtual residual projection dense coordinate is invalid".to_owned())?;
        if artifact.local_to_mask_state.len() != expected_dense_states {
            return Err(format!(
                "compiled virtual residual projection map has {} entries, expected {}",
                artifact.local_to_mask_state.len(), expected_dense_states,
            ));
        }
        if artifact.validated_component_state_count != Some(component_state_count)
            && artifact.local_to_mask_state.iter().any(|&state| state != u32::MAX && state >= component_state_count)
        {
            return Err(format!(
                "compiled virtual residual projection references state outside component width {}",
                component_state_count,
            ));
        }
        Ok(VirtualResidualMaskProjection {
            runtime: Arc::clone(self),
            state_offset: artifact.state_offset,
            pattern_states: oracle.pattern.num_states(),
            body_states: oracle.body.num_states(),
            prefix_len: oracle.prefix.len(),
            suffix_len: oracle.suffix.len(),
            min: oracle.min,
            full_max: oracle.max,
            mask_max,
            crossed_boundaries,
            local_to_mask_state: Arc::from(artifact.local_to_mask_state.into_boxed_slice()),
        })
    }

    pub(super) fn restore_finite_mask_projection(
        self: &Arc<Self>,
        max_token_len: usize,
        component_state_count: u32,
        artifact: VirtualResidualMaskProjectionArtifact,
    ) -> Result<VirtualResidualMaskProjection, String> {
        if artifact.terminal != self.terminal {
            return Err(format!(
                "virtual residual projection terminal mismatch: artifact={} runtime={}",
                artifact.terminal, self.terminal,
            ));
        }
        if !self.preserve_oracle_coordinate {
            return Err(
                "virtual residual projection requires coordinate-preserving residual runtime"
                    .to_owned(),
            );
        }
        let store = self
            .store
            .lock()
            .map_err(|_| "virtual residual runtime lock poisoned".to_owned())?;
        let oracle = store
            .liveness_oracle
            .as_ref()
            .ok_or_else(|| "virtual residual projection has no bounded-code oracle".to_owned())?;
        let minimum_body_width = oracle
            .body
            .min_match_byte_len()
            .ok_or_else(|| "virtual residual projection body has no minimum byte width".to_owned())?
            .max(1);
        let crossed_boundaries = max_token_len
            .div_ceil(minimum_body_width)
            .saturating_add(1);
        let crossed_boundaries = oracle.finite_mask_stencil_crossings(crossed_boundaries)
            .ok_or_else(|| "virtual residual projection future stencil overflow".to_owned())?;
        if oracle.min > crossed_boundaries.saturating_add(1) {
            return Err("virtual residual projection lower bound exceeds finite stencil".to_owned());
        }
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "virtual residual projection stencil overflow".to_owned())?;
        let mask_max = oracle.max.min(desired_mask_max);
        let expected_dense_states = oracle
            .finite_mask_dense_state_count(mask_max)
            .filter(|&states| states <= MAX_FINITE_MASK_DENSE_STATES)
            .ok_or_else(|| "virtual residual projection dense coordinate is invalid".to_owned())?;
        if artifact.local_to_mask_state.len() != expected_dense_states {
            return Err(format!(
                "virtual residual projection map has {} entries, expected {}",
                artifact.local_to_mask_state.len(), expected_dense_states,
            ));
        }
        if artifact.local_to_mask_state.iter().any(|&state| {
            state != u32::MAX && state >= component_state_count
        }) {
            return Err(format!(
                "virtual residual projection references state outside component width {}",
                component_state_count,
            ));
        }
        Ok(VirtualResidualMaskProjection {
            runtime: Arc::clone(self),
            state_offset: artifact.state_offset,
            pattern_states: oracle.pattern.num_states(),
            body_states: oracle.body.num_states(),
            prefix_len: oracle.prefix.len(),
            suffix_len: oracle.suffix.len(),
            min: oracle.min,
            full_max: oracle.max,
            mask_max,
            crossed_boundaries,
            local_to_mask_state: Arc::from(artifact.local_to_mask_state.into_boxed_slice()),
        })
    }

    pub(super) fn build_finite_mask_projection(
        self: &Arc<Self>,
        max_token_len: usize,
        state_offset: u32,
    ) -> Option<(
        DFA,
        CompressedTransitionSegment,
        u32,
        VirtualResidualMaskProjection,
    )> {
        let minimum_body_width = {
            let store = self.store.lock().unwrap();
            store.liveness_oracle.as_ref()?.body.min_match_byte_len()?.max(1)
        };
        // A token that begins in the middle of one body copy can complete at
        // most ceil(token_bytes / minimum_body_width) copies.
        let crossed_boundaries = max_token_len
            .div_ceil(minimum_body_width)
            .saturating_add(1);
        self.build_finite_mask_projection_for_crossed_boundaries(crossed_boundaries, state_offset)
    }

    pub(super) fn build_finite_mask_projection_for_crossed_boundaries(
        self: &Arc<Self>,
        crossed_boundaries: usize,
        state_offset: u32,
    ) -> Option<(
        DFA,
        CompressedTransitionSegment,
        u32,
        VirtualResidualMaskProjection,
    )> {
        let store = self.store.lock().unwrap();
        let oracle = store.liveness_oracle.as_ref()?;
        // Preserve token crossings and a complete future witness. Distinguish
        // distance to the real upper bound with the same extended stencil.
        let crossed_boundaries = oracle.finite_mask_stencil_crossings(crossed_boundaries)?;
        // Large lower minima need their own lower-bound abstraction; decline
        // rather than making this first exact lane scale with minLength.
        if oracle.min > crossed_boundaries.saturating_add(1) {
            return None;
        }
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)?
            .checked_add(1)?;
        let mask_max = oracle.max.min(desired_mask_max);
        // Even when the declared upper bound already fits inside one model-token
        // stencil, keep using the finite oracle coordinate. The absence of a
        // truncating stencil does not imply that eagerly materializing the

        // original pattern × length product is cheap.
        let (dfa, segment, root, local_to_mask_state) = oracle.finite_mask_dfa(mask_max)?;

        let projection = VirtualResidualMaskProjection {
            runtime: Arc::clone(self),
            state_offset,
            pattern_states: oracle.pattern.num_states(),
            body_states: oracle.body.num_states(),
            prefix_len: oracle.prefix.len(),
            suffix_len: oracle.suffix.len(),
            min: oracle.min,
            full_max: oracle.max,
            mask_max,
            crossed_boundaries,
            local_to_mask_state: Arc::from(local_to_mask_state.into_boxed_slice()),
        };
        Some((dfa, segment, root, projection))
    }

    pub(super) fn interned_state_count(&self) -> usize {
        self.store.lock().unwrap().residual_by_state.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn exact_byte_class_expansion_preserves_possible_futures() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, 0, 1);
        dfa.add_transition(1, 1, 2);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(2, finalizers, BitSet::new(1));
        dfa.recompute_possible_futures();

        let futures_before = (0..dfa.num_states() as u32)
            .map(|state| dfa.possible_future_group_ids(state).clone())
            .collect::<Vec<_>>();

        expand_exact_byte_classes(&mut dfa, &[vec![b'a', b'b'], vec![b'c', b'd']]);

        for (state, expected) in futures_before.iter().enumerate() {
            assert_eq!(dfa.possible_future_group_ids(state as u32), expected);
        }
        assert_eq!(dfa.step(0, b'a'), Some(1));
        assert_eq!(dfa.step(0, b'b'), Some(1));
        assert_eq!(dfa.step(1, b'c'), Some(2));
        assert_eq!(dfa.step(1, b'd'), Some(2));
    }

    fn bytes(value: &[u8]) -> Expr {
        Expr::U8Seq(value.to_vec())
    }

    fn accepts(arena: &mut ResidualArena, mut state: ResidualId, input: &[u8]) -> bool {
        for &byte in input {
            state = arena.step(state, byte).unwrap();
        }
        arena.is_nullable(state)
    }

    fn materialized_accepts(dfa: &DFA, input: &[u8]) -> bool {
        let mut state = 0;
        for &byte in input {
            let Some(target) = dfa.step(state, byte) else {
                return false;
            };
            state = target;
        }
        dfa.finalizers(state).contains(0)
    }

    fn random_small_expr(rng: &mut StdRng, depth: usize) -> Expr {
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
                min: rng.gen_range(0..=2),
                max: Some(rng.gen_range(2..=4)),
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

    fn all_words(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        let mut words = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in frontier {
                for &byte in alphabet {
                    let mut word = prefix.clone();
                    word.push(byte);
                    words.push(word.clone());
                    next.push(word);
                }
            }
            frontier = next;
        }
        words
    }

    #[test]
    fn giant_repeat_bound_stays_symbolic() {
        let expr = Expr::Repeat {
            expr: Box::new(bytes(b"ab")),
            min: 3,
            max: Some(1_000_000_000),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        let initial_states = arena.state_count();
        assert!(arena.has_future(root).unwrap());
        assert!(arena.state_count() <= initial_states + 2);

        let mut state = root;
        for _ in 0..3 {
            state = arena.step(state, b'a').unwrap();
            state = arena.step(state, b'b').unwrap();
        }
        assert!(arena.is_nullable(state));
        assert!(arena.has_future(state).unwrap());
        assert!(arena.state_count() < 32);
    }

    #[test]
    fn repeat_suffix_boundary_is_general_derivative_nondeterminism() {
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: Some(1_000_000_000),
            },
            bytes(b"ab"),
        ]);
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(accepts(&mut arena, root, b"ab"));
        assert!(accepts(&mut arena, root, b"aab"));
        assert!(accepts(&mut arena, root, b"aaaaab"));
        assert!(!accepts(&mut arena, root, b"b"));
        assert!(arena.state_count() < 64);
    }

    #[test]
    fn nullable_repeat_body_does_not_walk_the_bound() {
        let body = Expr::Choice(vec![Expr::Epsilon, bytes(b"a")]);
        let expr = Expr::Repeat {
            expr: Box::new(body),
            min: 500_000_000,
            max: Some(1_000_000_000),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(arena.is_nullable(root));
        assert!(arena.has_future(root).unwrap());
        assert!(accepts(&mut arena, root, b"aaa"));
        assert!(arena.state_count() < 32);
    }

    #[test]
    fn boolean_residuals_derive_compositionally() {
        let left = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: Some(32),
            },
            bytes(b"b"),
        ]);
        let right = Expr::Choice(vec![bytes(b"b"), bytes(b"aab"), bytes(b"c")]);
        let expr = Expr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(arena.has_future(root).unwrap());
        assert!(accepts(&mut arena, root, b"b"));
        assert!(accepts(&mut arena, root, b"aab"));
        assert!(!accepts(&mut arena, root, b"ab"));
        assert!(!accepts(&mut arena, root, b"c"));
    }

    #[test]
    fn embedded_dfa_epsilon_closure_is_a_compositional_residual() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_transition(1, b'a', 2);
        dfa.add_epsilon_transition(2, 3);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(3, accepting, BitSet::new(1));

        let expr = Expr::Dfa(Arc::new(dfa));
        let (mut arena, root) = ResidualArena::from_expr(&expr)
            .expect("epsilon-bearing embedded DFA must stay in the general residual algebra");
        assert!(!arena.is_nullable(root));
        assert!(arena.has_future(root).unwrap());
        assert!(accepts(&mut arena, root, b"a"));
        assert!(!accepts(&mut arena, root, b""));
        assert!(!accepts(&mut arena, root, b"aa"));
    }

    #[test]
    fn boolean_liveness_ceiling_is_error_not_dead() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 100,
                max: Some(100),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"aa")),
                min: 50,
                max: Some(50),
            }),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        let error = arena
            .has_future_with_budget(root, 8, 64)
            .expect_err("a deliberately tiny resource ceiling must not become a false dead result");
        assert!(error.contains("budget"), "unexpected liveness error: {error}");
        assert!(arena.has_future_with_budget(root, 256, 512).unwrap());
    }

    #[test]
    fn boolean_liveness_does_not_retain_dense_transition_rows() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 2,
                max: Some(4),
            }),
            intersect: Box::new(bytes(b"aa")),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(arena.transitions.iter().all(Option::is_none));
        assert!(arena.has_future(root).unwrap());
        assert!(
            arena.transitions.iter().all(Option::is_none),
            "exact liveness must keep derivative caching query-local rather than retaining dense rows",
        );

        // Normal runtime stepping deliberately keeps the dense hot-path cache.
        assert_ne!(arena.step(root, b'a').unwrap(), arena.empty);
        assert!(arena.transitions[root as usize].is_some());
    }

    #[test]
    fn boolean_liveness_budget_charges_recursive_sparse_derivatives() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"b"), bytes(b"c")])),
            intersect: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"b")])),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        let error = arena
            .has_future_with_budget(root, 16, 1)
            .expect_err("recursive sparse derivative work must consume the transition budget");
        assert!(error.contains("transition budget"), "unexpected error: {error}");
        assert!(arena.has_future_with_budget(root, 16, 32).unwrap());
    }

    #[test]
    fn giant_repeat_liveness_solves_embedded_body_once() {
        let mut body = DFA::new(3);
        body.ensure_group_capacity(1);
        body.add_transition(0, b'a', 1);
        body.add_transition(1, b'b', 2);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        body.overwrite_state_metadata(2, accepting, BitSet::new(1));
        // Deliberately leave derived future metadata stale: the residual
        // engine must reason from the DFA graph, not from precomputed labels.

        let expr = Expr::Repeat {
            expr: Box::new(Expr::Dfa(Arc::new(body))),
            min: 1_000_000_000,
            max: Some(1_000_000_000),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(
            arena.has_future_with_budget(root, 8, 16).unwrap(),
            "repeat liveness must solve the body language once rather than walk the billion-copy counter",
        );
    }

    #[test]
    fn sigma_star_identity_eliminates_trivial_boolean_counter_search() {
        let mut body = DFA::new(2);
        body.ensure_group_capacity(1);
        body.add_transition(0, b'a', 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        body.overwrite_state_metadata(1, accepting, BitSet::new(1));

        let counted = Expr::Repeat {
            expr: Box::new(Expr::Dfa(Arc::new(body))),
            min: 1_000_000_000,
            max: Some(1_000_000_000),
        };
        let sigma_star = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::all())),
            min: 0,
            max: None,
        };
        let expr = Expr::Intersect {
            expr: Box::new(counted),
            intersect: Box::new(sigma_star.clone()),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(
            arena.has_future_with_budget(root, 4, 4).unwrap(),
            "intersection with sigma-star must simplify before walking the billion-copy counter",
        );

        let choice = Expr::Choice(vec![bytes(b"literal"), sigma_star.clone()]);
        let (mut arena, root) = ResidualArena::from_expr(&choice).unwrap();
        assert_eq!(root, arena.sigma_star);
        assert!(accepts(&mut arena, root, b"anything\0goes"));

        let excluded = Expr::Exclude {
            expr: Box::new(bytes(b"literal")),
            exclude: Box::new(sigma_star),
        };
        let (arena, root) = ResidualArena::from_expr(&excluded).unwrap();
        assert!(arena.is_empty(root));
    }

    #[test]
    fn seeded_residual_algebra_matches_materialized_dfa() {
        let mut rng = StdRng::seed_from_u64(0x5E51_DA1A_2026_0826);
        let words = all_words(b"abc", 4);
        for case in 0..256 {
            let expr = random_small_expr(&mut rng, 3);
            let dfa = super::super::compile::compile_terminal_expr_dfa(&expr);
            let (mut arena, root) = ResidualArena::from_expr(&expr)
                .unwrap_or_else(|| panic!("residual compilation failed for case {case}: {expr:?}"));

            for word in &words {
                assert_eq!(
                    accepts(&mut arena, root, word),
                    materialized_accepts(&dfa, word),
                    "residual/materialized language mismatch in case {case}, expr={expr:?}, word={word:?}",
                );
            }

            assert_eq!(
                arena.has_future(root).unwrap(),
                dfa.possible_future_group_ids(0).contains(0),
                "residual/materialized root liveness mismatch in case {case}, expr={expr:?}",
            );
        }
    }

    #[test]
    fn sequence_liveness_skips_hard_nullable_siblings() {
        let hard_nullable = Expr::Repeat {
            expr: Box::new(Expr::Intersect {
                expr: Box::new(Expr::Repeat {
                    expr: Box::new(bytes(b"a")),
                    min: 100,
                    max: Some(100),
                }),
                intersect: Box::new(Expr::Repeat {
                    expr: Box::new(bytes(b"aa")),
                    min: 50,
                    max: Some(50),
                }),
            }),
            min: 0,
            max: Some(1_000_000_000),
        };
        let expr = Expr::Seq(vec![hard_nullable, bytes(b"z")]);
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(
            arena.has_future_with_budget(root, 0, 0).unwrap(),
            "the nonnullable literal proves a positive sequence word without solving the nullable sibling",
        );
    }

    #[test]
    fn runtime_future_bit_is_conservative_until_exact_boundary_check() {
        // The whole intersection accepts "c", but after consuming 'a' its
        // residual is exactly b intersect c: syntactically nonempty, semantically dead.
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![bytes(b"ab"), bytes(b"c")])),
            intersect: Box::new(Expr::Choice(vec![bytes(b"ac"), bytes(b"c")])),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        let dead_prefix = runtime
            .step(1, b'a')
            .expect("syntactic derivative is retained until the exact boundary check");
        assert!(runtime.futures(dead_prefix).unwrap().contains(0));
        assert_eq!(runtime.exact_has_future(dead_prefix).unwrap(), Some(false));

        let accepting = runtime.step(1, b'c').unwrap();
        assert!(runtime.finalizers(accepting).unwrap().contains(0));
        assert_eq!(runtime.exact_has_future(accepting).unwrap(), Some(false));
    }

    #[test]
    fn empty_boolean_root_is_conservative_until_exact_boundary_check() {
        let a_star = || Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 0,
            max: None,
        };
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Seq(vec![a_star(), bytes(b"b")])),
            intersect: Box::new(Expr::Seq(vec![a_star(), bytes(b"c")])),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.root_has_future());
        assert!(runtime.futures(1).unwrap().contains(0));
        assert_eq!(runtime.exact_has_future(1).unwrap(), Some(false));
        assert_eq!(
            runtime.step(1, b'a'),
            Some(1),
            "a syntactically continuing dead Boolean residual may remain as a conservative proxy until exact boundary pruning",
        );
    }

    #[test]
    fn runtime_construction_does_not_force_boolean_liveness_search() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 100,
                max: Some(100),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"aa")),
                min: 50,
                max: Some(50),
            }),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.root_has_future());

        let store = runtime.store.lock().unwrap();
        assert_eq!(
            store.arena.nonempty_cache[store.root as usize],
            None,
            "constructing the runtime must not eagerly solve a hard Boolean liveness problem",
        );
    }

    fn bounded_code_body() -> Expr {
        Expr::Choice(vec![bytes(b"a"), bytes(b"bc")])
    }

    fn bounded_code_envelope_expr(min: usize, max: usize) -> Expr {
        Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min,
                max: Some(max),
            },
            bytes(b">"),
        ])
    }

    fn bounded_code_envelope_with_body(body: Expr, min: usize, max: usize) -> Expr {
        Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(body),
                min,
                max: Some(max),
            },
            bytes(b">"),
        ])
    }

    fn exact_code_count_pattern(count: usize) -> Expr {
        let mut parts = Vec::with_capacity(count + 2);
        parts.push(bytes(b"<"));
        parts.extend((0..count).map(|_| bounded_code_body()));
        parts.push(bytes(b">"));
        Expr::Seq(parts)
    }

    fn finite_mask_row_fixtures() -> Vec<BoundedCodeIntersectionOracle> {
        let bodies = [
            bounded_code_body(),
            // Infinite but prefix-free code: every word ends at its first b.
            // This retains cycles and exercises the non-DAG minimizer too.
            Expr::Seq(vec![Expr::Repeat { expr: Box::new(bytes(b"a")), min: 0, max: None }, bytes(b"b")]),
            Expr::Choice(vec![bytes(b"a"), bytes("β".as_bytes()), bytes("猫".as_bytes()),
                bytes("🙂".as_bytes()), bytes(b"\\n")]),
        ];
        let mut fixtures = Vec::new();
        for body in bodies {
            for (min, max) in [(0, 17), (2, 17), (3, 3)] {
                for (prefix, suffix) in [(b"<".as_slice(), b">".as_slice()), (b"<<", b">>")] {
                    let pattern = Expr::Seq(vec![
                        bytes(prefix),
                        Expr::Repeat { expr: Box::new(Expr::Seq(vec![body.clone(); 3])), min: 0, max: None },
                        Expr::Repeat { expr: Box::new(body.clone()), min: 0, max: Some(1) },
                        bytes(suffix),
                    ]);
                    let envelope = Expr::Seq(vec![bytes(prefix),
                        Expr::Repeat { expr: Box::new(body.clone()), min, max: Some(max) }, bytes(suffix)]);
                    fixtures.push(BoundedCodeIntersectionOracle::from_expr(&Expr::Intersect {
                        expr: Box::new(pattern), intersect: Box::new(envelope),
                    }).expect("prefix-free finite-mask row fixture"));
                }
            }
            let unbounded = Expr::Seq(vec![bytes(b"<"),
                Expr::Repeat { expr: Box::new(body.clone()), min: 0, max: None }, bytes(b">")]);
            fixtures.push(BoundedCodeIntersectionOracle::from_expr(&Expr::Intersect {
                expr: Box::new(unbounded),
                intersect: Box::new(bounded_code_envelope_with_body(body, 1, 17)),
            }).expect("one-code loop with a full finite bound"));
        }
        fixtures
    }

    #[test]
    fn finite_mask_cached_body_rows_match_every_old_coordinate_class_step() {
        for (case, mut oracle) in finite_mask_row_fixtures().into_iter().enumerate() {
            // Also stress the original suffix-priority rule when both a suffix
            // and a body transition exist. This mutated test-only oracle does
            // not claim the frontend would certify such a language skeleton.
            for overlapping_suffix in [false, true] {
                if overlapping_suffix { oracle.suffix = Arc::from(b"a".as_slice()); }
                let classes = bounded_code_byte_classes(&oracle.pattern, &oracle.body, &oracle.prefix, &oracle.suffix);
                let transitions = FiniteMaskClassTransitions::new(&oracle, &classes).unwrap();
                for p in 0..oracle.pattern.num_states() as u32 {
                    for b in 0..oracle.body.num_states() as u32 {
                        let row = oracle.finite_mask_body_row(p, b, &classes, &transitions);
                        assert!(row.windows(2).all(|pair| pair[0].class < pair[1].class));
                        for max in [oracle.min, oracle.max, usize::MAX] {
                            for completed in [0, oracle.min.saturating_sub(1), oracle.min,
                                max.saturating_sub(1), max, max.saturating_add(1)] {
                                let source = BoundedCodeOracleCoordinate { pattern_state: p,
                                    envelope: BoundedCodeEnvelopeState::Body { completed, body_state: b } };
                                for (class, members) in classes.iter().enumerate() {
                                    let expected = oracle.step_coordinate_class(source, members[0], class, max, &transitions);
                                    let cached = row.iter().find(|edge| edge.class as usize == class)
                                        .and_then(|edge| edge.next_coordinate(completed, oracle.min, max, oracle.suffix.len()));
                                    assert_eq!(cached, expected,
                                        "case={case} overlap={overlapping_suffix} p={p} b={b} completed={completed} max={max} class={class}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn finite_mask_cached_rows_preserve_complete_dfa_wire_and_every_coordinate_mapping() {
        for (case, oracle) in finite_mask_row_fixtures().into_iter().enumerate() {
            for mask_max in [oracle.min, oracle.min.saturating_add(1).min(oracle.max), oracle.max] {
                let reference = oracle.finite_mask_dfa_reference(mask_max).expect("reference finite graph");
                let cached = oracle.finite_mask_dfa(mask_max).expect("cached finite graph");
                assert_eq!(cached.2, reference.2, "root case={case} max={mask_max}");
                assert_eq!(cached.3, reference.3, "all dense mappings case={case} max={mask_max}");
                assert_eq!(bincode::serialize(&cached.0).unwrap(), bincode::serialize(&reference.0).unwrap(),
                    "all DFA metadata case={case} max={mask_max}");
                assert_eq!(bincode::serialize(&cached.1).unwrap(), bincode::serialize(&reference.1).unwrap(),
                    "all compressed transitions case={case} max={mask_max}");
            }
        }
    }

    #[test]
    #[ignore = "manual paired finite-mask construction diagnostic; not whole-engine BUILD acceptance"]
    fn finite_mask_body_row_microbenchmark() {
        use std::time::Instant;
        let body = Expr::Choice(vec![
            Expr::U8Class(U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:/?&%=-")),
            bytes("β".as_bytes()), bytes("猫".as_bytes()), bytes("🙂".as_bytes()), bytes(b"\\n"),
        ]);
        let mut fixtures = [
            ("wide_count_format", Expr::Seq(vec![bytes(b"<"),
                Expr::Repeat { expr: Box::new(body.clone()), min: 0, max: Some(96) }, bytes(b"bcdef>")]), 0, 128, 128),
            ("minimum_and_long_suffix", Expr::Seq(vec![bytes(b"<"),
                Expr::Repeat { expr: Box::new(body.clone()), min: 0, max: Some(40) }, bytes(b"abcabc>")]), 3, 128, 128),
            ("tiny", Expr::Seq(vec![bytes(b"<"),
                Expr::Repeat { expr: Box::new(body.clone()), min: 0, max: None }, bytes(b">")]), 0, 4, 4),
            ("collapsed_count_format", Expr::Seq(vec![bytes(b"<"),
                Expr::Repeat { expr: Box::new(body.clone()), min: 0, max: Some(96) }, bytes(b"bcdef>")]), 0, 1_000_000, 128),
            ("collapsed_minimum_and_suffix", Expr::Seq(vec![bytes(b"<"),
                Expr::Repeat { expr: Box::new(body.clone()), min: 0, max: Some(40) }, bytes(b"abcabc>")]), 3, 1_000_000, 128),
        ].into_iter().map(|(name, pattern, min, max, mask_max)| {
            // The first bounded envelope selects the oracle coordinate. The
            // independent format also contains a bounded repeat, but its tail
            // begins with a body character and is not a prefix-code envelope.
            let oracle = BoundedCodeIntersectionOracle::from_expr(&Expr::Intersect {
                expr: Box::new(bounded_code_envelope_with_body(body.clone(), min, max)),
                intersect: Box::new(pattern),
            }).unwrap_or_else(|| panic!("microbenchmark oracle admission: {name}"));
            (name, oracle, mask_max)
        }).collect::<Vec<_>>();
        // A real slow schema has a cyclic address pattern and a full254-code
        // stencil, not a collapsed bound. This is a shape diagnostic rather
        // than a claim to reproduce the frontend's JSON-escape transduction.
        let address_body = Expr::Choice(vec![body.clone(), bytes(b"@")]);
        let address_pattern = Expr::Seq(vec![bytes(b"<"),
            crate::parse_regex(r"[^@^\s]+@[^@^\.^\s]+(\.[^@^\.^\s]*)*.gov.uk", true), bytes(b">")]);
        let address = BoundedCodeIntersectionOracle::from_expr(&Expr::Intersect {
            expr: Box::new(bounded_code_envelope_with_body(address_body, 5, 254)),
            intersect: Box::new(address_pattern),
        }).expect("cyclic full-bound address shape");
        fixtures.push(("cyclic_address_full_bound", address, 254));
        for threads in [2, 10] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                for (name, oracle, mask_max) in &fixtures {
                    let a = oracle.finite_mask_dfa_reference(*mask_max).unwrap();
                    let b = oracle.finite_mask_dfa(*mask_max).unwrap();
                    assert_eq!(bincode::serialize(&a).unwrap(), bincode::serialize(&b).unwrap());
                    for repeat in 0..8 {
                        for cached in if repeat % 2 == 0 { [false, true] } else { [true, false] } {
                            let started = Instant::now();
                            let result = if cached { oracle.finite_mask_dfa(*mask_max) }
                                else { oracle.finite_mask_dfa_reference(*mask_max) }.unwrap();
                            std::hint::black_box(&result);
                            let ns = started.elapsed().as_nanos();
                            eprintln!("[finite-body-row-benchmark] fixture={name} threads={threads} repeat={repeat} cached={cached} ns={ns} states={}", result.0.num_states());
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn dynamic_direct_coordinate_matches_owned_runtime_state_semantics() {
        let unbounded = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min: 0,
                max: None,
            },
            bytes(b">")
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(unbounded),
            intersect: Box::new(bounded_code_envelope_expr(0, 4)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime = VirtualResidualRuntime::new_dynamic(
            &expr, 0, 0, 1, 2, 1, allocator, owners,
        )
        .expect("bounded-code dynamic runtime should certify a direct coordinate");

        for input in [
            &b"<>"[..],
            &b"<a>"[..],
            &b"<bc>"[..],
            &b"<abc>"[..],
            &b"<aaaa>"[..],
            &b"<aaaaa>"[..],
            &b"<b>"[..],
        ] {
            let mut direct = runtime
                .direct_coordinate_for_state(1)
                .expect("dynamic runtime root should expose a direct coordinate");
            let mut state = Some(1u32);
            for &byte in input {
                let direct_next = runtime.direct_coordinate_step(direct, byte);
                let state_next = state.and_then(|source| runtime.step(source, byte));
                assert_eq!(direct_next.is_some(), state_next.is_some(), "step liveness mismatch for {input:?} at byte {byte:?}");
                let (Some(next_direct), Some(next_state)) = (direct_next, state_next) else {
                    break;
                };
                direct = next_direct;
                state = Some(next_state);
                let direct_accepting = runtime
                    .direct_coordinate_accepting(direct)
                    .expect("coordinate belongs to this runtime");
                assert_eq!(direct_accepting, runtime.accepting_now(next_state).unwrap());
                let direct_future = runtime
                    .direct_coordinate_has_future(direct)
                    .expect("coordinate belongs to this runtime");
                assert_eq!(direct_future, runtime.exact_has_future(next_state).unwrap().unwrap());
                assert_eq!(runtime.state_for_direct_coordinate(direct), Some(next_state));
            }
        }
    }

    #[test]
    fn coordinate_runtime_retains_bounded_slice_radius_after_prefix() {
        let unbounded = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min: 0,
                max: None,
            },
            bytes(b">")
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(unbounded),
            intersect: Box::new(bounded_code_envelope_expr(0, 4)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime = VirtualResidualRuntime::new_dynamic(
            &expr, 0, 0, 1, 2, 1, allocator, owners,
        )
        .expect("bounded-code dynamic runtime should certify a direct coordinate");

        let mut byte_to_class = [1u8; 256];
        byte_to_class[b'a' as usize] = 0;
        let class_count = 2usize;
        let transitions = [1u32, 2, 1, 2, 2, 2];
        let accepting = [false, true, false];
        let productive = [true, true, false];

        let after_open = runtime.step(1, b'<').expect("opening delimiter should be live");
        assert_eq!(
            runtime.parser_transparent_byte_dfa_repeat_radius(
                after_open, 0, class_count, &byte_to_class, &transitions,
                &accepting, &productive, 8, 16 * 1024,
            ),
            Some(4),
        );
        let after_one = runtime.step(after_open, b'a').expect("first body atom should be live");
        assert_eq!(
            runtime.parser_transparent_byte_dfa_repeat_radius(
                after_one, 0, class_count, &byte_to_class, &transitions,
                &accepting, &productive, 8, 16 * 1024,
            ),
            Some(3),
        );
    }

    #[test]
    fn slice_radius_totality_rejects_missing_multibyte_atom() {
        // The existential completion relation contains only the successful
        // a-loop; it omits the body word bc, which the pattern cannot read.
        let pattern = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat { expr: Box::new(bytes(b"a")), min: 0, max: None },
            bytes(b">"),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_expr(0, 4)),
        };
        let runtime = VirtualResidualRuntime::new_dynamic(
            &expr, 0, 0, 1, 2, 1,
            Arc::new(VirtualStateAllocator::new(2).unwrap()),
            Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap()),
        ).expect("bounded-code test runtime");
        let source = runtime.step(1, b'<').expect("live prefix");
        assert!(runtime.step(source, b'b').is_none());

        // Exact atom language a|bc, repeated by the slice DFA. First accepting
        // boundaries are complete body code words, including the missing bc.
        let mut classes = [3u8; 256];
        classes[b'a' as usize] = 0;
        classes[b'b' as usize] = 1;
        classes[b'c' as usize] = 2;
        let transitions = [1, 2, 3, 3, 1, 2, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3];
        let accepting = [false, true, false, false];
        let productive = [true, true, true, false];
        assert_eq!(
            runtime.parser_transparent_byte_dfa_repeat_radius(
                source, 0, 4, &classes, &transitions, &accepting, &productive, 8, 16 * 1024,
            ),
            Some(0),
            "an existing self-loop cannot certify an absent body-code transition",
        );
    }

    #[test]
    fn slice_radius_totality_rejects_space_after_last_allowed_word() {
        for max_words in 1usize..=4 {
            let word = || Expr::Repeat {
                expr: Box::new(bytes(b"a")), min: 1, max: None,
            };
            let pattern = Expr::Seq(vec![
                bytes(b"<"), word(),
                Expr::Repeat {
                    expr: Box::new(Expr::Seq(vec![bytes(b" "), word()])),
                    min: 0, max: Some(max_words - 1),
                },
                bytes(b">"),
            ]);
            let expr = Expr::Intersect {
                expr: Box::new(pattern),
                intersect: Box::new(bounded_code_envelope_with_body(
                    Expr::Choice(vec![bytes(b"a"), bytes(b" ")]), 0, 32,
                )),
            };
            let runtime = VirtualResidualRuntime::new_dynamic(
                &expr, 0, 0, 1, 2, 1,
                Arc::new(VirtualStateAllocator::new(2).unwrap()),
                Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap()),
            ).expect("word-count bounded-code runtime");
            let prefix = format!("<{}", vec!["a"; max_words].join(" "));
            let mut source = 1u32;
            for byte in prefix.bytes() {
                source = runtime.step(source, byte).expect("valid last-word prefix");
            }
            assert!(runtime.step(source, b' ').is_none(), "word bound must reject another separator");
            let mut broad_classes = [1u8; 256];
            broad_classes[b'a' as usize] = 0;
            broad_classes[b' ' as usize] = 0;
            let transitions = [1, 2, 1, 2, 2, 2];
            let accepting = [false, true, false];
            let productive = [true, true, false];
            assert_eq!(
                runtime.parser_transparent_byte_dfa_repeat_radius(
                    source, 0, 2, &broad_classes, &transitions,
                    &accepting, &productive, 32, 16 * 1024,
                ),
                Some(0), "broad atom set includes the forbidden separator; max_words={max_words}",
            );
            // A narrower alphabet really is total and invariant, and should
            // retain the fast positive radius up to the character envelope.
            let mut narrow_classes = [1u8; 256];
            narrow_classes[b'a' as usize] = 0;
            assert_eq!(
                runtime.parser_transparent_byte_dfa_repeat_radius(
                    source, 0, 2, &narrow_classes, &transitions,
                    &accepting, &productive, 32, 16 * 1024,
                ),
                Some((32 - (2 * max_words - 1)) as u32),
            );
        }
    }

    #[test]
    fn finite_mask_projection_retains_future_witness_beyond_token_horizon() {
        // Independent finite literal language: <a{0,75}bcdef>. One-byte
        // tokens need a five-copy future witness before the closing suffix.
        let body = Expr::Choice(b"abcdef".iter().map(|&byte| bytes(&[byte])).collect());
        let pattern = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat { expr: Box::new(bytes(b"a")), min: 0, max: None },
            bytes(b"bcdef>"),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_with_body(body, 0, 80)),
        };
        let runtime = Arc::new(VirtualResidualRuntime::new_dynamic(
            &expr, 0, 0, 1, 2, 1,
            Arc::new(VirtualStateAllocator::new(2).unwrap()),
            Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap()),
        ).expect("bounded-code tail fixture"));
        let (dfa, segment, _, projection) = runtime
            .build_finite_mask_projection_for_crossed_boundaries(1, 0)
            .expect("future-preserving one-byte vocabulary projection");
        let words = (0..=75).map(|count| {
            let mut word = vec![b'<'];
            word.extend(std::iter::repeat_n(b'a', count));
            word.extend_from_slice(b"bcdef>");
            word
        }).collect::<Vec<_>>();
        for count in [0, 4, 74, 75] {
            for tail in [b"".as_slice(), b"b", b"bcde", b"bcdef"] {
                let mut prefix = vec![b'<'];
                prefix.extend(std::iter::repeat_n(b'a', count));
                prefix.extend_from_slice(tail);
                let mut source = 1;
                for &byte in &prefix {
                    source = runtime.step(source, byte).expect("independently live prefix");
                }
                let projected = projection.project(source).expect("live prefix projects");
                for &byte in b"abcdefz>" {
                    let mut candidate = prefix.clone();
                    candidate.push(byte);
                    let expected = (
                        words.iter().any(|word| word == &candidate),
                        words.iter().any(|word| word.len() > candidate.len()
                            && word.starts_with(&candidate)),
                    );
                    let exact = runtime.step(source, byte)
                        .and_then(|target| runtime.observation(target))
                        .unwrap_or((false, false));
                    assert_eq!(exact, expected, "exact prefix={candidate:?}");
                    let row_start = segment.row_offsets[projected as usize] as usize;
                    let row_end = segment.row_offsets[projected as usize + 1] as usize;
                    let class = segment.byte_to_class[byte as usize];
                    let projected_target = segment.entries.iter_range(row_start, row_end)
                        .find_map(|(edge_class, target)| (edge_class == class).then_some(target));
                    let observed = projected_target.map(|target| (
                        dfa.finalizers(target).contains(0),
                        dfa.possible_future_group_ids(target).contains(0),
                    )).unwrap_or((false, false));
                    assert_eq!(observed, expected, "projected prefix={candidate:?}");
                }
            }
        }
    }

    #[test]
    fn mask_result_dense_key_retains_closed_subgraph_future_witnesses() {
        // Long unreachable pattern prefixes must not inflate this query's
        // witness. The required bcdef tail still spans more than one token.
        fn check(prefix_len: usize, min: usize, max: usize) {
            let mut required = vec![b'<'];
            required.extend(std::iter::repeat_n(b'p', prefix_len));
            let pattern = Expr::Seq(vec![
                bytes(&required),
                Expr::Repeat { expr: Box::new(bytes(b"a")), min: 0, max: None },
                bytes(b"bcdef>"),
            ]);
            let body = Expr::Choice(b"pabcdef".iter().map(|&b| bytes(&[b])).collect());
            let expr = Expr::Intersect {
                expr: Box::new(pattern),
                intersect: Box::new(bounded_code_envelope_with_body(body, min, max)),
            };
            let runtime = Arc::new(VirtualResidualRuntime::new_dynamic(
                &expr, 0, 0, 1, 2, 1,
                Arc::new(VirtualStateAllocator::new(2).unwrap()),
                Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap()),
            ).expect("certified prefix-free nonnullable body"));
            let words = (0..=max-prefix_len-5).filter_map(|count| {
                if prefix_len+count+5 < min { return None; }
                let mut word = required.clone();
                word.extend(std::iter::repeat_n(b'a', count));
                word.extend_from_slice(b"bcdef>");
                Some(word)
            }).collect::<Vec<_>>();
            let mut candidates = vec![Vec::new()];
            let mut layer = vec![Vec::new()];
            for _ in 0..3 {
                let mut next = Vec::new();
                for prefix in &layer {
                    for &byte in b"abcdef>z" {
                        let mut candidate = prefix.clone();
                        candidate.push(byte);
                        next.push(candidate);
                    }
                }
                candidates.extend(next.iter().cloned());
                layer = next;
            }
            let mut by_key = FxHashMap::<(u32,u32), Vec<(bool,bool)>>::default();
            let mut aliases = 0;
            let mut focused = Vec::new();
            for count in 0..=max-prefix_len-5 {
                for tail in [b"".as_slice(), b"b", b"bcde", b"bcdef", b"bcdef>"] {
                    let mut prefix = required.clone();
                    prefix.extend(std::iter::repeat_n(b'a', count));
                    prefix.extend_from_slice(tail);
                    let root = runtime.direct_coordinate_for_state(1).unwrap();
                    let Some(coordinate) = prefix.iter().try_fold(root, |state,&byte|
                        runtime.direct_coordinate_step(state,byte)) else { continue; };
                    let Some(key) = runtime.direct_coordinate_mask_result_dense_key(coordinate,3) else { continue; };
                    let observations = candidates.iter().map(|candidate| {
                        let target = candidate.iter().try_fold(coordinate, |state,&byte|
                            runtime.direct_coordinate_step(state,byte));
                        let observed = target.map(|target| (
                            runtime.direct_coordinate_accepting(target).unwrap(),
                            runtime.direct_coordinate_has_future(target).unwrap(),
                        )).unwrap_or((false,false));
                        let mut full = prefix.clone();
                        full.extend(candidate);
                        let expected = (
                            words.iter().any(|word| word == &full),
                            words.iter().any(|word| word.len()>full.len() && word.starts_with(&full)),
                        );
                        assert_eq!(observed,expected,"prefix={prefix:?} candidate={candidate:?}");
                        observed
                    }).collect::<Vec<_>>();
                    if let Some(previous) = by_key.insert(key,observations.clone()) {
                        aliases += 1;
                        assert_eq!(previous,observations,"equal result keys must preserve every <=3-byte observation");
                    }
                    if prefix_len == 100 && tail.is_empty() && matches!(count,10|15) {
                        focused.push((key,runtime.direct_coordinate_finite_mask_dense_key(coordinate,3).unwrap()));
                    }
                }
            }
            assert!(aliases>0);
            if prefix_len == 100 {
                assert_eq!(focused.len(),2);
                assert_eq!(focused[0].0,focused[1].0,"closed-subgraph result cache must coalesce the interior pair");
                assert_ne!(focused[0].1,focused[1].1,"compiler/global finite projection must keep its original distinction");
            }
            let mut store = runtime.store.lock().unwrap();
            let oracle = store.liveness_oracle.as_mut().unwrap();
            let before = bincode::serialize(oracle).unwrap();
            for p in 0..oracle.pattern.num_states() as u32 {
                let bound = oracle.mask_result_future_repeat_headroom(p, 0).unwrap();
                assert!(bound<=oracle.finite_mask_future_repeat_headroom().unwrap());
                let targets = oracle.completion_row(0, p).iter_ones().map(|target| target as u32).collect::<Vec<_>>();
                for target in targets {
                    // Closure can only shrink after a complete body code.
                    assert!(oracle.mask_result_future_repeat_headroom(target, 0).unwrap() <= bound);
                }
            }
            assert_eq!(before,bincode::serialize(oracle).unwrap());
        }
        check(100,0,160);
        check(3,9,25);
    }

    #[test]
    fn direct_coordinate_dense_mask_key_refines_full_finite_projection() {
        let unbounded = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min: 0,
                max: None,
            },
            bytes(b">")
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(unbounded),
            intersect: Box::new(bounded_code_envelope_expr(0, 80)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime = Arc::new(
            VirtualResidualRuntime::new_dynamic(
                &expr, 0, 0, 1, 2, 1, allocator, owners,
            )
            .expect("bounded-code dynamic runtime should certify a direct coordinate"),
        );
        let max_token_len = 8usize;
        let (_, _, _, projection) = Arc::clone(&runtime)
            .build_finite_mask_projection(max_token_len, 10_000)
            .expect("fixture should admit a finite mask projection");

        let mut coordinate = runtime
            .direct_coordinate_for_state(1)
            .expect("dynamic runtime root should expose a direct coordinate");
        let mut by_dense = FxHashMap::<(u32, u32), u32>::default();
        let mut exact_states = 0usize;
        let mut alias_count = 0usize;
        for &byte in std::iter::once(&b'<').chain(std::iter::repeat_n(&b'a', 70)) {
            coordinate = runtime
                .direct_coordinate_step(coordinate, byte)
                .expect("long valid body prefix should remain live");
            let state = runtime
                .state_for_direct_coordinate(coordinate)
                .expect("reachable direct coordinate should materialize");
            let Some(key) = runtime
                .direct_coordinate_finite_mask_dense_key(coordinate, max_token_len)
            else {
                continue;
            };
            exact_states += 1;
            let projected = projection
                .project(state)
                .expect("reachable exact state must project");
            if let Some(previous) = by_dense.insert(key, projected) {
                alias_count += 1;
                assert_eq!(
                    previous, projected,
                    "equal dense one-token coordinates must map to one minimized projection state",
                );
            }
        }
        assert!(exact_states > 40, "fixture should traverse the deep exact count interval");
        assert!(alias_count > 20, "dense key should collapse many deep-interior exact counts");
    }

    #[test]
    fn bounded_code_oracle_drops_redundant_unbounded_envelope_pattern() {
        let unbounded = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(unbounded),
            intersect: Box::new(bounded_code_envelope_expr(2, 4)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .expect("redundant unbounded envelope pattern should certify");
        assert_eq!(oracle.pattern.num_states(), 1);
        assert!(oracle.has_future(oracle.root_coordinate()));

        let materialized = compile_terminal_expr_dfa(&expr);
        assert_eq!(
            oracle.has_future(oracle.root_coordinate()),
            materialized.possible_future_group_ids(0).contains(0),
        );
    }

    #[test]
    fn standalone_bounded_code_mask_component_retains_byte_transitions() {
        let unbounded = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(unbounded),
            intersect: Box::new(bounded_code_envelope_expr(0, 100)),
        };

        let (dfa, root) = prepare_bounded_code_mask_component(&expr)
            .expect("bounded-code expression should prepare")
            .finish_for_vocab_conservative(8)
            .expect("bounded-code mask component should compile");

        assert!(
            dfa.transition_count() > 0,
            "standalone mask component must materialize its compressed transition sidecar"
        );
        assert!(dfa.possible_future_group_ids(root).contains(0));
        let after_prefix = dfa
            .step(root, b'<')
            .expect("standalone mask component must execute its prefix byte");
        assert!(dfa.possible_future_group_ids(after_prefix).contains(0));
    }

    #[test]
    fn standalone_bounded_code_mask_component_accepts_moderate_large_minimum() {
        let unbounded = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(unbounded),
            intersect: Box::new(bounded_code_envelope_expr(500, 10_000)),
        };

        let (dfa, root) = prepare_bounded_code_mask_component(&expr)
            .expect("bounded-code expression should prepare")
            .finish_for_vocab_conservative(128)
            .expect("moderate lower bound should stay on the finite mask path");

        assert!(dfa.possible_future_group_ids(root).contains(0));
        assert!(
            dfa.num_states() < 20_000,
            "finite mask coordinate should scale with the lower bound plus one-token stencil, not maxLength: {} states",
            dfa.num_states(),
        );
    }

    #[test]
    fn bounded_code_intersection_oracle_respects_gapped_copy_counts() {
        // The pattern admits exactly two or four code words.  An envelope of
        // exactly three words is therefore dead even though each operand is
        // individually live.  This is the counterexample that rules out a
        // simple min/max-distance liveness approximation.
        let pattern = Expr::Choice(vec![exact_code_count_pattern(2), exact_code_count_pattern(4)]);
        let dead = Expr::Intersect {
            expr: Box::new(pattern.clone()),
            intersect: Box::new(bounded_code_envelope_expr(3, 3)),
        };
        let mut dead_oracle = BoundedCodeIntersectionOracle::from_expr(&dead)
            .expect("prefix-code bounded intersection should certify");
        assert!(!dead_oracle.has_future(dead_oracle.root_coordinate()));

        let live = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_expr(3, 4)),
        };
        let mut live_oracle = BoundedCodeIntersectionOracle::from_expr(&live)
            .expect("prefix-code bounded intersection should certify");
        assert!(live_oracle.has_future(live_oracle.root_coordinate()));
    }

    #[test]
    fn bounded_code_oracle_coalesces_identical_envelope_intervals_exactly() {
        let pattern = exact_code_count_pattern(3);
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Intersect {
                expr: Box::new(pattern),
                intersect: Box::new(bounded_code_envelope_expr(1, 4)),
            }),
            intersect: Box::new(bounded_code_envelope_expr(3, 6)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .expect("identical bounded-code envelopes should coalesce");
        assert_eq!((oracle.min, oracle.max), (3, 4));
        assert!(oracle.has_future(oracle.root_coordinate()));

        let materialized = compile_terminal_expr_dfa(&expr);
        assert!(
            materialized
                .possible_future_group_ids(0)
                .contains(0),
            "materialized intersection must agree that the root has a future",
        );
    }

    #[test]
    fn bounded_code_oracle_coalesces_disjoint_identical_envelopes_to_dead() {
        let pattern = Expr::Choice(vec![exact_code_count_pattern(2), exact_code_count_pattern(4)]);
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Intersect {
                expr: Box::new(pattern),
                intersect: Box::new(bounded_code_envelope_expr(1, 2)),
            }),
            intersect: Box::new(bounded_code_envelope_expr(4, 5)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .expect("disjoint identical envelopes should still admit an exact dead certificate");
        assert_eq!((oracle.min, oracle.max), (4, 2));
        assert!(!oracle.has_future(oracle.root_coordinate()));

        let materialized = compile_terminal_expr_dfa(&expr);
        assert!(
            !materialized
                .possible_future_group_ids(0)
                .contains(0),
            "materialized disjoint intersection must also be dead",
        );
    }

    #[test]
    fn bounded_code_oracle_rejects_ambiguous_code_boundaries() {
        let pattern = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::all())),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ]);

        // "a" is a prefix of "ab", so greedily treating the first accepting
        // body state as one completed code word would not be exact.
        let non_prefix_free = Expr::Intersect {
            expr: Box::new(pattern.clone()),
            intersect: Box::new(bounded_code_envelope_with_body(
                Expr::Choice(vec![bytes(b"a"), bytes(b"ab")]),
                0,
                4,
            )),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&non_prefix_free).is_none());

        // At a code boundary, '>' could either begin another productive body
        // word or begin the fixed suffix. The sidecar deliberately refuses
        // such an envelope rather than choosing one interpretation.
        let suffix_ambiguous = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_with_body(
                Expr::Choice(vec![bytes(b"a"), bytes(b">x")]),
                0,
                4,
            )),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&suffix_ambiguous).is_none());
    }

    #[test]
    fn bounded_code_oracle_does_not_materialize_nested_giant_repeats() {
        let pattern = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::all())),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ]);
        let giant_body = Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 4_096,
            max: Some(4_096),
        };
        let body_giant = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_with_body(giant_body, 0, 5_000)),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&body_giant).is_none());

        // A giant bounded repeat can also hide inside the independently
        // compiled pattern operand. Even when simplification could make a
        // particular example cheap (epsilon repeated many times), the oracle
        // must not rely on eagerly discovering that after materialization.
        let giant_pattern = Expr::Choice(vec![
            Expr::Seq(vec![
                bytes(b"<"),
                Expr::Repeat {
                    expr: Box::new(Expr::Epsilon),
                    min: 0,
                    max: Some(4_096),
                },
                bytes(b">"),
            ]),
            bytes(b"x"),
        ]);
        let pattern_giant = Expr::Intersect {
            expr: Box::new(giant_pattern),
            intersect: Box::new(bounded_code_envelope_expr(0, 5_000)),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&pattern_giant).is_none());
    }

    #[test]
    fn bounded_code_oracle_matches_materialized_future_at_every_small_prefix() {
        let pattern = Expr::Choice(vec![
            exact_code_count_pattern(1),
            exact_code_count_pattern(3),
            exact_code_count_pattern(4),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_expr(1, 4)),
        };
        let materialized = compile_terminal_expr_dfa(&expr);
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.has_bounded_code_liveness_oracle());

        let alphabet = [b'<', b'>', b'a', b'b', b'c'];
        let mut queue = VecDeque::from([(Vec::<u8>::new(), 1u32, 0u32)]);
        let mut seen = FxHashSet::<(u32, u32)>::default();
        seen.insert((1, 0));
        while let Some((prefix, residual_state, materialized_state)) = queue.pop_front() {
            assert_eq!(
                runtime.exact_has_future(residual_state).unwrap(),
                Some(
                    materialized
                        .possible_future_group_ids(materialized_state)
                        .contains(0)
                ),
                "future mismatch after prefix {:?}",
                String::from_utf8_lossy(&prefix),
            );
            assert_eq!(
                runtime
                    .futures(residual_state)
                    .expect("reached residual state must have metadata")
                    .contains(0),
                materialized
                    .possible_future_group_ids(materialized_state)
                    .contains(0),
                "ordinary future metadata mismatch after prefix {:?}",
                String::from_utf8_lossy(&prefix),
            );
            if prefix.len() >= 12 {
                continue;
            }
            for &byte in &alphabet {
                let Some(materialized_target) = materialized.step(materialized_state, byte) else {
                    continue;
                };
                let Some(residual_target) = runtime.step(residual_state, byte) else {
                    continue;
                };
                if seen.insert((residual_target, materialized_target)) {
                    let mut next_prefix = prefix.clone();
                    next_prefix.push(byte);
                    queue.push_back((next_prefix, residual_target, materialized_target));
                }
            }
        }
    }

    #[test]
    fn bounded_code_sparse_oracle_wire_preserves_runtime_liveness() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![
                exact_code_count_pattern(1),
                exact_code_count_pattern(3),
                exact_code_count_pattern(4),
            ])),
            intersect: Box::new(bounded_code_envelope_expr(1, 4)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let original = VirtualResidualRuntime::new(
            &expr,
            0,
            0,
            1,
            2,
            1,
            Arc::clone(&allocator),
            Arc::clone(&owners),
        )
        .unwrap();
        assert!(original.has_bounded_code_liveness_oracle());
        let wire = original.serialized_bounded_code_oracle();
        assert!(wire.starts_with(&SPARSE_BOUNDED_CODE_ORACLE_MAGIC));

        let loaded = VirtualResidualRuntime::new_preserving_oracle_coordinate_from_oracle_bytes(
            &expr,
            &wire,
            0,
            0,
            1,
            2,
            1,
            allocator,
            owners,
        )
        .expect("BCO2 oracle should restore without rebuilding dense relations");
        assert!(loaded.has_bounded_code_liveness_oracle());

        let alphabet = [b'<', b'>', b'a', b'b', b'c'];
        let mut queue = VecDeque::from([(Vec::<u8>::new(), 1u32, 1u32)]);
        let mut seen = FxHashSet::<(u32, u32)>::default();
        seen.insert((1, 1));
        while let Some((prefix, original_state, loaded_state)) = queue.pop_front() {
            assert_eq!(
                loaded.exact_has_future(loaded_state).unwrap(),
                original.exact_has_future(original_state).unwrap(),
                "BCO2 future mismatch after {:?}",
                String::from_utf8_lossy(&prefix),
            );
            if prefix.len() >= 10 {
                continue;
            }
            for &byte in &alphabet {
                let original_next = original.step(original_state, byte);
                let loaded_next = loaded.step(loaded_state, byte);
                assert_eq!(
                    loaded_next.is_some(),
                    original_next.is_some(),
                    "BCO2 transition mismatch after {:?} + {:?}",
                    String::from_utf8_lossy(&prefix),
                    byte as char,
                );
                let (Some(original_next), Some(loaded_next)) = (original_next, loaded_next) else {
                    continue;
                };
                if seen.insert((original_next, loaded_next)) {
                    let mut next_prefix = prefix.clone();
                    next_prefix.push(byte);
                    queue.push_back((next_prefix, original_next, loaded_next));
                }
            }
        }
    }

    #[test]
    fn master_slice_artifact_round_trips_prepared_runtime_and_rejects_bad_shape() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Seq(vec![
                bytes(b"<"),
                Expr::Repeat {
                    expr: Box::new(bounded_code_body()),
                    min: 0,
                    max: None,
                },
                bytes(b">"),
            ])),
            intersect: Box::new(bounded_code_envelope_expr(0, 8)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let original = VirtualResidualRuntime::new(
            &expr,
            0,
            0,
            1,
            2,
            1,
            Arc::clone(&allocator),
            Arc::clone(&owners),
        )
        .unwrap();

        // A compact DFA for (a|bc)+. State 1 marks an atom boundary and also
        // accepts the next atom, matching the master-slice contract.
        let mut byte_to_class = [0u8; 256];
        for (byte, class) in byte_to_class.iter_mut().enumerate() {
            *class = byte as u8;
        }
        let state_count = 3usize;
        let class_count = 256usize;
        let dead = state_count as u32;
        let mut transitions = vec![dead; state_count * class_count];
        transitions[b'a' as usize] = 1;
        transitions[b'b' as usize] = 2;
        transitions[class_count + b'a' as usize] = 1;
        transitions[class_count + b'b' as usize] = 2;
        transitions[2 * class_count + b'c' as usize] = 1;
        let accepting = [false, true, false];
        let productive = [true, true, true];
        original.prepare_master_slice_artifacts(
            0,
            class_count,
            &byte_to_class,
            &transitions,
            &accepting,
            &productive,
            8,
        );
        let artifact = original
            .master_slice_artifact()
            .expect("master-slice preparation should produce a transfer artifact");
        assert!(!artifact.words.is_empty());
        assert!(!artifact.body_exact.is_empty());
        assert!(!artifact.pattern_targets.is_empty());

        let oracle_bytes = original.serialized_bounded_code_oracle();
        let loaded_allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let loaded_owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let loaded = VirtualResidualRuntime::new_preserving_oracle_coordinate_from_oracle_bytes(
            &expr,
            &oracle_bytes,
            0,
            0,
            1,
            2,
            1,
            loaded_allocator,
            loaded_owners,
        )
        .unwrap();
        loaded.restore_master_slice_artifact(artifact.clone()).unwrap();

        let original_store = original.store.lock().unwrap();
        let loaded_store = loaded.store.lock().unwrap();
        assert_eq!(
            loaded_store.body_boundary_future_by_completed,
            original_store.body_boundary_future_by_completed,
        );
        assert_eq!(
            loaded_store.slice_atom_body_exact_cache,
            original_store.slice_atom_body_exact_cache,
        );
        assert_eq!(
            loaded_store.slice_atom_pattern_targets_cache,
            original_store.slice_atom_pattern_targets_cache,
        );
        drop(loaded_store);
        drop(original_store);

        // The compact load path deliberately omits relation powers. Its
        // backwards-DP artifact must also serve the generic (non-body-equal)
        // slice proof and the repeat-radius BFS, not only ordinary liveness.
        let mut compact = VirtualResidualRuntime::compact_liveness_oracle_from_master_slice_artifact(&artifact)
            .expect("valid transferred oracle");
        assert!(compact.exact_powers.is_empty() && compact.prefix_sums.is_empty());
        let coordinate = compact.root_coordinate();
        assert_eq!(compact.has_future_for_slice_proof(coordinate, None), None,
            "missing proof data must decline, not certify a dead language");
        loaded.store.lock().unwrap().liveness_oracle = Some(compact);

        for prefix in [b"<".as_slice(), b"<a", b"<bc", b"<b"] {
            let mut old_state = 1;
            let mut loaded_state = 1;
            for &byte in prefix {
                old_state = original.step(old_state, byte).expect("valid original prefix");
                loaded_state = loaded.step(loaded_state, byte).expect("valid loaded prefix");
            }
            // This alphabet is intentionally not the oracle body (a|bc), so
            // the specialized exact-body proof must fall through to the BFS.
            let first = if prefix == b"<b" { b'c' } else { b'a' };
            let mut repeating = vec![2_u32; 2 * class_count];
            repeating[first as usize] = 1;
            repeating[class_count + b'a' as usize] = 1;
            let repeating_accepting = [false, true];
            let repeating_productive = [true, true];
            let expected_radius = original.parser_transparent_byte_dfa_repeat_radius(
                old_state, 0, class_count, &byte_to_class, &repeating,
                &repeating_accepting, &repeating_productive, 8, 10_000,
            );
            let loaded_radius = loaded.parser_transparent_byte_dfa_repeat_radius(
                loaded_state, 0, class_count, &byte_to_class, &repeating,
                &repeating_accepting, &repeating_productive, 8, 10_000,
            );
            assert!(expected_radius.is_some());
            assert_eq!(loaded_radius, expected_radius, "prefix={prefix:?}");

            let finite_states = 5usize;
            let mut finite = vec![finite_states as u32; finite_states * class_count];
            finite[first as usize] = 1;
            for state in 1..finite_states - 1 {
                finite[state * class_count + b'a' as usize] = (state + 1) as u32;
            }
            let finite_productive = vec![true; finite_states];
            let expected = original.parser_transparent_byte_dfa(
                old_state, 0, class_count, &byte_to_class, &finite,
                &finite_productive, true, 10_000,
            );
            let actual = loaded.parser_transparent_byte_dfa(
                loaded_state, 0, class_count, &byte_to_class, &finite,
                &finite_productive, true, 10_000,
            );
            assert_eq!(actual, expected, "finite slice prefix={prefix:?}");
            assert_eq!(actual, Some(true));

            // Dynamic joint-root and uniform-byte fast paths use different
            // proof entry points. Exercise them against the genuinely compact
            // oracle too, rather than accidentally retaining relation powers.
            let old_coordinate = {
                let store = original.store.lock().unwrap();
                original.oracle_coordinate_for_state_locked(&store, old_state).unwrap()
            };
            let loaded_coordinate = {
                let store = loaded.store.lock().unwrap();
                loaded.oracle_coordinate_for_state_locked(&store, loaded_state).unwrap()
            };
            let direct_expected = original.direct_coordinate_parser_transparent_byte_dfa(
                VirtualResidualDirectCoordinate { runtime_index: original.runtime_index, coordinate: old_coordinate },
                0, class_count, &byte_to_class, &finite, &finite_productive, true, 10_000,
            );
            let direct_actual = loaded.direct_coordinate_parser_transparent_byte_dfa(
                VirtualResidualDirectCoordinate { runtime_index: loaded.runtime_index, coordinate: loaded_coordinate },
                0, class_count, &byte_to_class, &finite, &finite_productive, true, 10_000,
            );
            assert_eq!(direct_actual, direct_expected, "direct finite slice prefix={prefix:?}");
            assert_eq!(direct_actual, Some(true));
            for byte in [b'a', b'c'] {
                let family = U8Set::single(byte);
                for horizon in [1, 3] {
                    let expected = original.parser_transparent_byte_family(old_state, family, horizon);
                    let actual = loaded.parser_transparent_byte_family(loaded_state, family, horizon);
                    assert!(expected.is_some());
                    assert_eq!(actual, expected, "byte family prefix={prefix:?} byte={byte} horizon={horizon}");
                }
            }
        }

        let mut malformed = artifact;
        malformed.pattern_states += 1;
        assert!(loaded.restore_master_slice_artifact(malformed).is_err());
    }

    #[test]
    fn bounded_code_oracle_keeps_billion_bound_logarithmic() {
        let expr = Expr::Intersect {
            expr: Box::new(exact_code_count_pattern(2)),
            intersect: Box::new(bounded_code_envelope_expr(0, 1_000_000_000)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .expect("billion-copy prefix-code envelope should certify");
        assert!(oracle.has_future(oracle.root_coordinate()));
        assert!(
            oracle.exact_powers.len() <= 31,
            "doubling table must scale with log2(max), got {} layers",
            oracle.exact_powers.len(),
        );
    }

    #[test]
    fn bounded_code_oracle_sizes_doubling_for_inclusive_power_of_two_ranges() {
        for max in [0usize, 1, 3, 7, 15] {
            let expr = Expr::Intersect {
                expr: Box::new(exact_code_count_pattern(max)),
                intersect: Box::new(bounded_code_envelope_expr(0, max)),
            };
            let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
                .unwrap_or_else(|| panic!("bounded-code oracle should certify max={max}"));
            assert!(
                oracle.has_future(oracle.root_coordinate()),
                "exactly {max} code words must be reachable inside 0..={max}",
            );
        }
    }

    #[test]
    fn bounded_code_oracle_ambiguity_propagates_to_existing_successors() {
        let expr = Expr::Intersect {
            expr: Box::new(exact_code_count_pattern(1)),
            intersect: Box::new(bounded_code_envelope_expr(0, 4)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();

        let body_boundary = runtime.step(1, b'<').unwrap();
        let after_one = runtime.step(body_boundary, b'a').unwrap();
        assert!(runtime.futures(after_one).unwrap().contains(0));

        {
            let mut store = runtime.store.lock().unwrap();
            let source = VirtualResidualRuntime::residual_for_state(
                &store,
                runtime.root_state,
                body_boundary,
            )
            .unwrap() as usize;
            let target = VirtualResidualRuntime::residual_for_state(
                &store,
                runtime.root_state,
                after_one,
            )
            .unwrap() as usize;
            assert!(matches!(
                store.oracle_coordinates[target],
                BoundedCodeOracleSlot::Exact(_)
            ));
            assert_eq!(store.oracle_futures[target], Some(true));
            store.oracle_coordinates[source] = BoundedCodeOracleSlot::Ambiguous;
        }

        assert_eq!(runtime.step(body_boundary, b'a'), Some(after_one));
        let store = runtime.store.lock().unwrap();
        let target = VirtualResidualRuntime::residual_for_state(
            &store,
            runtime.root_state,
            after_one,
        )
        .unwrap() as usize;
        assert_eq!(
            store.oracle_coordinates[target],
            BoundedCodeOracleSlot::Ambiguous
        );
        assert_eq!(store.oracle_futures[target], None);
    }

    #[test]
    fn bounded_code_runtime_liveness_does_not_fall_back_to_boolean_search() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![
                exact_code_count_pattern(2),
                exact_code_count_pattern(4),
            ])),
            intersect: Box::new(bounded_code_envelope_expr(3, 3)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.has_bounded_code_liveness_oracle());
        assert_eq!(runtime.exact_has_future(1).unwrap(), Some(false));
        assert!(
            runtime.futures(1).unwrap().is_empty(),
            "certified exact liveness must be visible through ordinary tokenizer future metadata",
        );
        let store = runtime.store.lock().unwrap();
        assert_eq!(
            store.arena.nonempty_cache[store.root as usize],
            None,
            "certified bounded-code liveness must not invoke generic Boolean reachability",
        );
    }
}
