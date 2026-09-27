//! Exact direct-regular terminal support and frontier representations.

use super::DenseAcceptanceRows;
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::weight::Weight;
use crate::grammar::flat::DirectRegularAutomaton;
use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;
#[derive(Debug, Clone)]
pub(crate) struct DirectRegularWideFrontierAcceptance {
    /// Pointer identities of immutable replace-target or StackShifts slices in the live table
    /// that all produce this exact frontier. Runtime-only and rebuilt after
    /// deserialization.
    pub(crate) action_origins: Vec<usize>,
    pub(crate) state_count: usize,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) frontier_states: Arc<[u32]>,
    pub(crate) empty_acc_frontier: ParserGSS,
    pub(crate) acceptance_parts: Arc<[Weight]>,
    pub(crate) dense_by_tsid: Arc<DenseAcceptanceRows>,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularDynamicHotFrontier {
    pub(crate) frontier_states: Arc<[u32]>,
    pub(crate) empty_acc_frontier: ParserGSS,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularParserStateAcceptance {
    pub(crate) parser_state: u32,
    pub(crate) acceptance_parts: Arc<[Weight]>,
    pub(crate) dense_by_tsid: Arc<DenseAcceptanceRows>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DirectRegularSupportNode {
    Leaf(u64),
    Branch(u32, u32),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DirectRegularSmallSupport {
    pub(super) len: u8,
    pub(super) terminals: [u16; 4],
}

impl DirectRegularSmallSupport {
    pub(super) const UNAVAILABLE: u8 = u8::MAX;

    pub(super) fn unavailable() -> Self {
        Self {
            len: Self::UNAVAILABLE,
            terminals: [0; 4],
        }
    }

    pub(super) fn from_leaf(mut value: u64) -> Self {
        if value.count_ones() > 4 {
            return Self::unavailable();
        }
        let mut result = Self {
            len: 0,
            terminals: [0; 4],
        };
        while value != 0 {
            result.terminals[result.len as usize] = value.trailing_zeros() as u16;
            result.len += 1;
            value &= value - 1;
        }
        result
    }

    pub(super) fn combine(left: Self, right: Self, right_offset: usize) -> Self {
        if left.len == Self::UNAVAILABLE
            || right.len == Self::UNAVAILABLE
            || usize::from(left.len) + usize::from(right.len) > 4
            || right_offset > u16::MAX as usize
        {
            return Self::unavailable();
        }
        let mut result = Self {
            len: left.len + right.len,
            terminals: [0; 4],
        };
        result.terminals[..left.len as usize].copy_from_slice(&left.terminals[..left.len as usize]);
        for (index, &terminal) in right.terminals[..right.len as usize].iter().enumerate() {
            let Some(terminal) = usize::from(terminal).checked_add(right_offset) else {
                return Self::unavailable();
            };
            let Ok(terminal) = u16::try_from(terminal) else {
                return Self::unavailable();
            };
            result.terminals[left.len as usize + index] = terminal;
        }
        result
    }

    pub(super) fn terminals(&self) -> Option<&[u16]> {
        (self.len != Self::UNAVAILABLE).then(|| &self.terminals[..self.len as usize])
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DirectRegularTerminalSupport {
    pub(super) roots: Vec<u32>,
    pub(super) nodes: Vec<DirectRegularSupportNode>,
    pub(super) node_counts: Vec<u16>,
    pub(super) node_small_support: Vec<DirectRegularSmallSupport>,
    pub(super) dense_state_rows: FxHashMap<u32, Arc<[u64]>>,
    pub(super) zero: Vec<u32>,
    pub(super) levels: u8,
    pub(super) num_terminals: usize,
}

pub(super) struct DirectRegularTerminalSupportBuilder {
    pub(super) nodes: Vec<DirectRegularSupportNode>,
    pub(super) node_counts: Vec<u16>,
    pub(super) node_small_support: Vec<DirectRegularSmallSupport>,
    pub(super) leaf_intern: FxHashMap<u64, u32>,
    pub(super) branch_intern: Vec<FxHashMap<(u32, u32), u32>>,
    pub(super) union_memo: Vec<FxHashMap<(u32, u32), u32>>,
    pub(super) zero: Vec<u32>,
}

impl DirectRegularTerminalSupportBuilder {
    pub(super) fn new(levels: usize) -> Self {
        let mut builder = Self {
            nodes: Vec::new(),
            node_counts: Vec::new(),
            node_small_support: Vec::new(),
            leaf_intern: FxHashMap::default(),
            branch_intern: (0..=levels).map(|_| FxHashMap::default()).collect(),
            union_memo: (0..=levels).map(|_| FxHashMap::default()).collect(),
            zero: Vec::with_capacity(levels + 1),
        };
        let leaf = builder.intern_leaf(0);
        builder.zero.push(leaf);
        for level in 1..=levels {
            let child = builder.zero[level - 1];
            let root = builder.intern_branch(level, child, child);
            builder.zero.push(root);
        }
        builder
    }

    pub(super) fn intern_leaf(&mut self, value: u64) -> u32 {
        if let Some(&id) = self.leaf_intern.get(&value) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(DirectRegularSupportNode::Leaf(value));
        self.node_counts.push(value.count_ones() as u16);
        self.node_small_support
            .push(DirectRegularSmallSupport::from_leaf(value));
        self.leaf_intern.insert(value, id);
        id
    }

    pub(super) fn intern_branch(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if let Some(&id) = self.branch_intern[level].get(&(left, right)) {
            return id;
        }
        let id = self.nodes.len() as u32;
        self.nodes
            .push(DirectRegularSupportNode::Branch(left, right));
        self.node_counts
            .push(self.node_counts[left as usize].saturating_add(self.node_counts[right as usize]));
        let right_offset = 64usize << (level - 1);
        self.node_small_support
            .push(DirectRegularSmallSupport::combine(
                self.node_small_support[left as usize],
                self.node_small_support[right as usize],
                right_offset,
            ));
        self.branch_intern[level].insert((left, right), id);
        id
    }

    pub(super) fn union(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if left == right {
            return left;
        }
        if left == self.zero[level] {
            return right;
        }
        if right == self.zero[level] {
            return left;
        }
        let key = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        if let Some(&id) = self.union_memo[level].get(&key) {
            return id;
        }
        let result = if level == 0 {
            let DirectRegularSupportNode::Leaf(left_value) = self.nodes[left as usize] else {
                unreachable!()
            };
            let DirectRegularSupportNode::Leaf(right_value) = self.nodes[right as usize] else {
                unreachable!()
            };
            self.intern_leaf(left_value | right_value)
        } else {
            let DirectRegularSupportNode::Branch(left_a, left_b) = self.nodes[left as usize] else {
                unreachable!()
            };
            let DirectRegularSupportNode::Branch(right_a, right_b) = self.nodes[right as usize]
            else {
                unreachable!()
            };
            let a = self.union(level - 1, left_a, right_a);
            let b = self.union(level - 1, left_b, right_b);
            self.intern_branch(level, a, b)
        };
        self.union_memo[level].insert(key, result);
        result
    }

    pub(super) fn singleton(&mut self, levels: usize, terminal: usize) -> u32 {
        let word = terminal / 64;
        let mut node = self.intern_leaf(1u64 << (terminal % 64));
        for level in 1..=levels {
            let zero = self.zero[level - 1];
            node = if ((word >> (level - 1)) & 1) == 0 {
                self.intern_branch(level, node, zero)
            } else {
                self.intern_branch(level, zero, node)
            };
        }
        node
    }
}

impl DirectRegularTerminalSupport {
    pub(crate) fn build(automaton: &DirectRegularAutomaton, num_terminals: usize) -> Self {
        if automaton.states.is_empty() || num_terminals == 0 {
            return Self::default();
        }
        let word_count = num_terminals.div_ceil(64).next_power_of_two();
        let levels = word_count.trailing_zeros() as usize;
        let mut builder = DirectRegularTerminalSupportBuilder::new(levels);
        let singletons = (0..num_terminals)
            .map(|terminal| builder.singleton(levels, terminal))
            .collect::<Vec<_>>();

        let mut parents = vec![Vec::<u32>::new(); automaton.states.len()];
        let mut remaining_children = Vec::<u32>::with_capacity(automaton.states.len());
        let mut queue = VecDeque::<u32>::new();
        for (source, state) in automaton.states.iter().enumerate() {
            remaining_children.push(state.epsilons.len() as u32);
            if state.epsilons.is_empty() {
                queue.push_back(source as u32);
            }
            for &child in &state.epsilons {
                parents[child as usize].push(source as u32);
            }
        }

        let mut roots = vec![builder.zero[levels]; automaton.states.len()];
        let mut processed = 0usize;
        while let Some(raw) = queue.pop_front() {
            let state = &automaton.states[raw as usize];
            let mut root = builder.zero[levels];
            for &terminal in state.transitions.keys() {
                if (terminal as usize) < num_terminals {
                    root = builder.union(levels, root, singletons[terminal as usize]);
                }
            }
            for &child in &state.epsilons {
                root = builder.union(levels, root, roots[child as usize]);
            }
            roots[raw as usize] = root;
            processed += 1;
            for &parent in &parents[raw as usize] {
                let remaining = &mut remaining_children[parent as usize];
                *remaining -= 1;
                if *remaining == 0 {
                    queue.push_back(parent);
                }
            }
        }
        if processed != automaton.states.len() {
            return Self::default();
        }
        let mut support = Self {
            roots,
            nodes: builder.nodes,
            node_counts: builder.node_counts,
            node_small_support: builder.node_small_support,
            dense_state_rows: FxHashMap::default(),
            zero: builder.zero,
            levels: levels as u8,
            num_terminals,
        };
        let dense_word_count = num_terminals.div_ceil(64);
        for &raw_state in &automaton.start_states {
            let mut words = vec![0u64; dense_word_count];
            support.or_state_into(raw_state, &mut words);
            support.dense_state_rows.insert(raw_state, Arc::from(words));
        }
        support
    }

    pub(crate) fn is_initialized(&self) -> bool {
        !self.roots.is_empty()
    }

    pub(crate) fn for_each_small_state_terminal(
        &self,
        raw_state: u32,
        mut visit: impl FnMut(TerminalID),
    ) -> bool {
        let Some(root) = self.root_id(raw_state) else {
            return false;
        };
        let Some(terminals) = self.node_small_support[root as usize].terminals() else {
            return false;
        };
        for &terminal in terminals {
            let terminal = TerminalID::from(terminal);
            if (terminal as usize) < self.num_terminals {
                visit(terminal);
            }
        }
        true
    }

    #[inline]
    pub(crate) fn contains(&self, raw_state: u32, terminal: TerminalID) -> bool {
        let terminal = terminal as usize;
        if terminal >= self.num_terminals {
            return false;
        }
        let Some(&mut_node) = self.roots.get(raw_state as usize) else {
            return false;
        };
        let mut node = mut_node;
        let mut level = self.levels as usize;
        let word = terminal / 64;
        while level != 0 {
            let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
                return false;
            };
            node = if ((word >> (level - 1)) & 1) == 0 {
                left
            } else {
                right
            };
            level -= 1;
        }
        let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
            return false;
        };
        value & (1u64 << (terminal % 64)) != 0
    }

    pub(super) fn or_node(&self, node: u32, level: usize, word_base: usize, output: &mut [u64]) {
        if node == self.zero[level] {
            return;
        }
        if level == 0 {
            let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
                return;
            };
            if let Some(word) = output.get_mut(word_base) {
                *word |= value;
            }
            return;
        }
        let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
            return;
        };
        let half = 1usize << (level - 1);
        self.or_node(left, level - 1, word_base, output);
        self.or_node(right, level - 1, word_base + half, output);
    }

    pub(crate) fn or_state_into(&self, raw_state: u32, output: &mut [u64]) {
        if let Some(words) = self.dense_state_rows.get(&raw_state) {
            for (target, source) in output.iter_mut().zip(words.iter()) {
                *target |= *source;
            }
            return;
        }
        if let Some(&root) = self.roots.get(raw_state as usize) {
            self.or_node(root, self.levels as usize, 0, output);
        }
    }

    #[inline]
    pub(crate) fn root_id(&self, raw_state: u32) -> Option<u32> {
        self.roots.get(raw_state as usize).copied()
    }

    #[inline]
    pub(crate) fn state_terminal_count(&self, raw_state: u32) -> Option<u16> {
        let root = *self.roots.get(raw_state as usize)?;
        self.node_counts.get(root as usize).copied()
    }

    pub(crate) fn singleton_terminal(&self, raw_state: u32) -> Option<TerminalID> {
        let root = self.root_id(raw_state)?;
        let terminals = self.node_small_support[root as usize].terminals()?;
        let [terminal] = terminals else {
            return None;
        };
        Some(TerminalID::from(*terminal))
    }

    pub(super) fn intersects_node(
        &self,
        node: u32,
        level: usize,
        word_base: usize,
        terminals: &[u64],
    ) -> bool {
        if node == self.zero[level] {
            return false;
        }
        if level == 0 {
            let DirectRegularSupportNode::Leaf(value) = self.nodes[node as usize] else {
                return false;
            };
            return terminals
                .get(word_base)
                .is_some_and(|word| (*word & value) != 0);
        }
        let DirectRegularSupportNode::Branch(left, right) = self.nodes[node as usize] else {
            return false;
        };
        let half = 1usize << (level - 1);
        self.intersects_node(left, level - 1, word_base, terminals)
            || self.intersects_node(right, level - 1, word_base + half, terminals)
    }

    pub(crate) fn intersects(&self, raw_state: u32, terminals: &[u64]) -> bool {
        if let Some(words) = self.dense_state_rows.get(&raw_state) {
            return words
                .iter()
                .zip(terminals)
                .any(|(left, right)| (*left & *right) != 0);
        }
        self.roots
            .get(raw_state as usize)
            .is_some_and(|&root| self.intersects_node(root, self.levels as usize, 0, terminals))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DirectRegularDynamicFrontierCacheEntry {
    /// Retain the source interface so its pointer-derived key cannot be reused
    /// while this cache entry exists.
    pub(crate) source: ParserGSS,
    pub(crate) actionable_terminals: crate::ds::bitset::BitSet,
    pub(crate) advance_by_terminal: Arc<[(TerminalID, Arc<[u32]>)]>,
}
