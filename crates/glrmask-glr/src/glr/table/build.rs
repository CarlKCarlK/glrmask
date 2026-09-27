use super::*;
use super::row::SparseRow;
use std::sync::Arc;

use crate::ds::bitset::BitSet;
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

const DISABLE_UNIT_REDUCTION_INLINING_ENV: &str = "GLRMASK_DISABLE_UNIT_REDUCTION_INLINING";
const GLR_TABLE_CONSTRUCTION_ENV: &str = "GLRMASK_GLR_TABLE_CONSTRUCTION";
const UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES_ENV: &str =
    "GLRMASK_UNIT_REDUCTION_INLINE_MAX_PRE_MERGE_STATES";
const DEFAULT_UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES: u32 = 8_192;
const ROW_BISIM_MAX_PRE_MERGE_STATES_ENV: &str =
    "GLRMASK_ROW_BISIM_MAX_PRE_MERGE_STATES";
const SAME_CORE_MAX_PRE_MERGE_STATES_ENV: &str =
    "GLRMASK_SAME_CORE_MAX_PRE_MERGE_STATES";
const INCREMENTAL_ROW_MERGE_ENV: &str = "GLRMASK_INCREMENTAL_ROW_MERGE";
const DEFAULT_SAME_CORE_MAX_PRE_MERGE_STATES: u32 = 4_096;

fn glr_table_construction_override() -> Option<GlrTableConstruction> {
    match std::env::var(GLR_TABLE_CONSTRUCTION_ENV) {
        Ok(value) if value.trim().eq_ignore_ascii_case("legacy")
            || value.trim().eq_ignore_ascii_case("legacy-row-bisim")
            || value.trim().eq_ignore_ascii_case("row-bisim") =>
        {
            Some(GlrTableConstruction::LegacyRowBisim)
        }
        Ok(value) if value.trim().eq_ignore_ascii_case("lalr") => {
            Some(GlrTableConstruction::Lalr)
        }
        Ok(value) if value.trim().eq_ignore_ascii_case("core-lac") => {
            Some(GlrTableConstruction::ExperimentalCoreMerged)
        }
        _ => None,
    }
}

fn is_large_left_linear_grammar(grammar: &AnalyzedGrammar) -> bool {
    const MIN_RULES: usize = 64;
    if grammar.rules.len() < MIN_RULES {
        return false;
    }

    let mut left_recursive_rules = 0usize;
    for rule in grammar.rules.iter().skip(1) {
        let mut nonterminal_positions = rule
            .rhs
            .iter()
            .enumerate()
            .filter_map(|(index, symbol)| {
                matches!(symbol, Symbol::Nonterminal(_)).then_some(index)
            });
        match (nonterminal_positions.next(), nonterminal_positions.next()) {
            (None, None) => {}
            (Some(0), None) => left_recursive_rules += 1,
            _ => return false,
        }
    }

    left_recursive_rules >= MIN_RULES / 2
}

fn selected_glr_table_construction(
    grammar: &AnalyzedGrammar,
    default: GlrTableConstruction,
) -> GlrTableConstruction {
    const LEGACY_LALR_RULE_THRESHOLD: usize = 40_000;
    const LEGACY_LALR_MAX_TERMINALS: u32 = 512;
    if let Some(explicit) = glr_table_construction_override() {
        return explicit;
    }
    if default == GlrTableConstruction::LegacyRowBisim
        && grammar.rules.len() >= LEGACY_LALR_RULE_THRESHOLD
        && grammar.num_terminals <= LEGACY_LALR_MAX_TERMINALS
    {
        // Static JSON imports historically request LegacyRowBisim, while the
        // reduced-latency/dynamic JSON path already uses LALR. At very large
        // lowered rule counts with a small terminal alphabet, constructing the
        // canonical LR(1) graph only to quotient it back down dominates compile
        // time. Use the same exact, conflict-preserving LALR construction as
        // the dynamic path for this rare tail shape. Large-terminal grammars
        // stay on LegacyRowBisim: their canonical LR(1) graph can already be
        // cheap, while LALR may enlarge the downstream parser-DWA workload.
        GlrTableConstruction::Lalr
    } else if default == GlrTableConstruction::ExperimentalCoreMerged
        && is_large_left_linear_grammar(grammar)
    {
        GlrTableConstruction::LegacyRowBisim
    } else {
        default
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn unit_reduction_inlining_enabled() -> bool {
    !env_flag_enabled(DISABLE_UNIT_REDUCTION_INLINING_ENV)
}

fn incremental_row_merge_enabled() -> bool {
    // The incremental path is exact after the preceding full quotient and
    // avoids re-refining unaffected rows. Keep it on by default; callers can
    // explicitly set 0/false/no/off to force the full second quotient.
    std::env::var(INCREMENTAL_ROW_MERGE_ENV)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true)
}


fn unit_reduction_inlining_max_pre_merge_states() -> Option<u32> {
    match std::env::var(UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES_ENV) {
        Ok(value) => match value.trim().parse::<u32>() {
            Ok(0) => None,
            Ok(parsed) => Some(parsed),
            Err(_) => Some(DEFAULT_UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES),
        },
        Err(_) => Some(DEFAULT_UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES),
    }
}

fn row_bisim_max_pre_merge_states() -> Option<u32> {
    match std::env::var(ROW_BISIM_MAX_PRE_MERGE_STATES_ENV) {
        Ok(value) => match value.trim().parse::<u32>() {
            Ok(0) => None,
            Ok(parsed) => Some(parsed),
            Err(_) => None,
        },
        Err(_) => None,
    }
}

fn row_bisim_quotient_enabled(pre_merge_states: u32) -> bool {
    row_bisim_max_pre_merge_states()
        .is_none_or(|max_pre_merge_states| pre_merge_states <= max_pre_merge_states)
}


fn same_core_max_pre_merge_states() -> Option<u32> {
    match std::env::var(SAME_CORE_MAX_PRE_MERGE_STATES_ENV) {
        Ok(value) => match value.trim().parse::<u32>() {
            Ok(0) => None,
            Ok(parsed) => Some(parsed),
            Err(_) => Some(DEFAULT_SAME_CORE_MAX_PRE_MERGE_STATES),
        },
        Err(_) => Some(DEFAULT_SAME_CORE_MAX_PRE_MERGE_STATES),
    }
}

fn same_core_quotient_enabled(pre_merge_states: u32) -> bool {
    same_core_max_pre_merge_states()
        .is_none_or(|max_pre_merge_states| pre_merge_states <= max_pre_merge_states)
}


#[derive(Default)]
struct DirectRegularActionInterner {
    replace_target_scratch: Vec<u32>,
    replace_targets: FxHashSet<Arc<[u32]>>,
}

impl DirectRegularActionInterner {
    fn intern_targets(&mut self, targets: impl IntoIterator<Item = u32>) -> Arc<[u32]> {
        self.replace_target_scratch.clear();
        self.replace_target_scratch.extend(targets);
        if let Some(existing) = self.replace_targets.get(self.replace_target_scratch.as_slice()) {
            return Arc::clone(existing);
        }
        let targets: Arc<[u32]> = Arc::from(self.replace_target_scratch.clone());
        self.replace_targets.insert(Arc::clone(&targets));
        targets
    }

    fn action_from_sorted_scratch(&mut self) -> Option<Action> {
        match self.replace_target_scratch.len() {
            0 => None,
            1 => Some(Action::Shift(self.replace_target_scratch[0], true)),
            _ => {
                let targets = if let Some(existing) = self
                    .replace_targets
                    .get(self.replace_target_scratch.as_slice())
                {
                    Arc::clone(existing)
                } else {
                    let targets: Arc<[u32]> = Arc::from(self.replace_target_scratch.clone());
                    self.replace_targets.insert(Arc::clone(&targets));
                    targets
                };
                Some(Action::ReplaceShifts(targets))
            }
        }
    }

}


#[derive(Clone, Copy)]
enum DirectRegularPersistentTargetNode {
    Leaf { bits: u64, count: u32 },
    Branch { left: u32, right: u32, count: u32 },
}

struct DirectRegularPersistentTargetSets {
    nodes: Vec<DirectRegularPersistentTargetNode>,
    leaf_intern: FxHashMap<u64, u32>,
    branch_intern: Vec<FxHashMap<(u32, u32), u32>>,
    union_memo: Vec<FxHashMap<(u32, u32), u32>>,
    zero: Vec<u32>,
    levels: usize,
}

impl DirectRegularPersistentTargetSets {
    fn new(target_count: usize) -> Self {
        let word_count = target_count.div_ceil(64).max(1).next_power_of_two();
        let levels = word_count.trailing_zeros() as usize;
        let mut result = Self {
            nodes: Vec::new(),
            leaf_intern: FxHashMap::default(),
            branch_intern: (0..=levels).map(|_| FxHashMap::default()).collect(),
            union_memo: (0..=levels).map(|_| FxHashMap::default()).collect(),
            zero: Vec::with_capacity(levels + 1),
            levels,
        };
        let zero_leaf = result.intern_leaf(0);
        result.zero.push(zero_leaf);
        for level in 1..=levels {
            let child = result.zero[level - 1];
            let root = result.intern_branch(level, child, child);
            result.zero.push(root);
        }
        result
    }

    #[inline]
    fn empty(&self) -> u32 {
        self.zero[self.levels]
    }

    #[inline]
    fn count(&self, node: u32) -> usize {
        match self.nodes[node as usize] {
            DirectRegularPersistentTargetNode::Leaf { count, .. }
            | DirectRegularPersistentTargetNode::Branch { count, .. } => count as usize,
        }
    }

    fn intern_leaf(&mut self, bits: u64) -> u32 {
        if let Some(&existing) = self.leaf_intern.get(&bits) {
            return existing;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(DirectRegularPersistentTargetNode::Leaf {
            bits,
            count: bits.count_ones(),
        });
        self.leaf_intern.insert(bits, id);
        id
    }

    fn intern_branch(&mut self, level: usize, left: u32, right: u32) -> u32 {
        if let Some(&existing) = self.branch_intern[level].get(&(left, right)) {
            return existing;
        }
        let count = self.count(left).saturating_add(self.count(right)) as u32;
        let id = self.nodes.len() as u32;
        self.nodes
            .push(DirectRegularPersistentTargetNode::Branch { left, right, count });
        self.branch_intern[level].insert((left, right), id);
        id
    }

    fn singleton(&mut self, target: usize) -> u32 {
        let leaf_word = target / 64;
        let mut node = self.intern_leaf(1u64 << (target % 64));
        for level in 1..=self.levels {
            let zero = self.zero[level - 1];
            node = if ((leaf_word >> (level - 1)) & 1) == 0 {
                self.intern_branch(level, node, zero)
            } else {
                self.intern_branch(level, zero, node)
            };
        }
        node
    }

    fn union(&mut self, level: usize, left: u32, right: u32) -> u32 {
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
        if let Some(&existing) = self.union_memo[level].get(&key) {
            return existing;
        }
        let result = if level == 0 {
            let DirectRegularPersistentTargetNode::Leaf { bits: left, .. } =
                self.nodes[left as usize]
            else {
                unreachable!("level-zero persistent target node must be a leaf")
            };
            let DirectRegularPersistentTargetNode::Leaf { bits: right, .. } =
                self.nodes[right as usize]
            else {
                unreachable!("level-zero persistent target node must be a leaf")
            };
            self.intern_leaf(left | right)
        } else {
            let DirectRegularPersistentTargetNode::Branch {
                left: left_a,
                right: left_b,
                ..
            } = self.nodes[left as usize]
            else {
                unreachable!("persistent target node level mismatch")
            };
            let DirectRegularPersistentTargetNode::Branch {
                left: right_a,
                right: right_b,
                ..
            } = self.nodes[right as usize]
            else {
                unreachable!("persistent target node level mismatch")
            };
            let a = self.union(level - 1, left_a, right_a);
            let b = self.union(level - 1, left_b, right_b);
            self.intern_branch(level, a, b)
        };
        self.union_memo[level].insert(key, result);
        result
    }

    #[inline]
    fn union_roots(&mut self, left: u32, right: u32) -> u32 {
        self.union(self.levels, left, right)
    }

    fn materialize(&self, root: u32) -> Vec<u32> {
        fn visit(
            sets: &DirectRegularPersistentTargetSets,
            node: u32,
            level: usize,
            base_word: usize,
            output: &mut Vec<u32>,
        ) {
            match sets.nodes[node as usize] {
                DirectRegularPersistentTargetNode::Leaf { mut bits, .. } => {
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        output.push((base_word * 64 + bit) as u32);
                        bits &= bits - 1;
                    }
                }
                DirectRegularPersistentTargetNode::Branch { left, right, .. } => {
                    debug_assert_ne!(level, 0);
                    visit(sets, left, level - 1, base_word, output);
                    visit(
                        sets,
                        right,
                        level - 1,
                        base_word + (1usize << (level - 1)),
                        output,
                    );
                }
            }
        }

        let mut output = Vec::with_capacity(self.count(root));
        visit(self, root, self.levels, 0, &mut output);
        output
    }
}

struct DirectRegularClosureWorkspace {
    seen_epoch: Vec<u32>,
    epoch: u32,
    stack: Vec<u32>,
    terminal_targets: Vec<(TerminalID, u32)>,
    actions: DirectRegularActionInterner,
}

impl DirectRegularClosureWorkspace {
    fn new(state_count: usize) -> Self {
        Self {
            seen_epoch: vec![0; state_count],
            epoch: 0,
            stack: Vec::new(),
            terminal_targets: Vec::new(),
            actions: DirectRegularActionInterner::default(),
        }
    }

    fn begin(&mut self, roots: impl IntoIterator<Item = u32>) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.seen_epoch.fill(0);
            self.epoch = 1;
        }
        self.stack.clear();
        self.stack.extend(roots);
        self.terminal_targets.clear();
    }

    fn mark_new(&mut self, state: u32) -> bool {
        let entry = &mut self.seen_epoch[state as usize];
        if *entry == self.epoch {
            false
        } else {
            *entry = self.epoch;
            true
        }
    }

    fn intern_replace_targets(&mut self, start: usize, end: usize) -> Arc<[u32]> {
        self.actions.intern_targets(
            self.terminal_targets[start..end]
                .iter()
                .map(|&(_, target)| target),
        )
    }
}

fn direct_regular_action_row_for_roots_with_widest(
    grammar: &AnalyzedGrammar,
    roots: impl IntoIterator<Item = u32>,
    workspace: &mut DirectRegularClosureWorkspace,
) -> Option<(ActionRow, BitSet, Option<(TerminalID, usize)>)> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    workspace.begin(roots);
    let mut accepting = false;

    while let Some(state_id) = workspace.stack.pop() {
        if state_id as usize >= automaton.states.len() {
            return None;
        }
        if !workspace.mark_new(state_id) {
            continue;
        }
        let state = &automaton.states[state_id as usize];
        accepting |= state.is_accepting;
        workspace.stack.extend(state.epsilons.iter().copied());
        for (&terminal, targets) in &state.transitions {
            if terminal >= grammar.num_terminals {
                return None;
            }
            for &target in targets {
                if target as usize >= automaton.states.len() {
                    return None;
                }
                workspace.terminal_targets.push((terminal, target + 1));
            }
        }
    }

    workspace.terminal_targets.sort_unstable();
    workspace.terminal_targets.dedup();
    let mut row = Vec::with_capacity(workspace.terminal_targets.len() + usize::from(accepting));
    let mut advance = BitSet::new(grammar.num_terminals as usize + 1);
    let mut widest: Option<(TerminalID, usize)> = None;
    let mut index = 0usize;
    while index < workspace.terminal_targets.len() {
        let terminal = workspace.terminal_targets[index].0;
        let mut end = index + 1;
        while end < workspace.terminal_targets.len()
            && workspace.terminal_targets[end].0 == terminal
        {
            end += 1;
        }
        if widest
            .as_ref()
            .is_none_or(|(_, width)| end - index > *width)
        {
            widest = Some((terminal, end - index));
        }
        let action = if end == index + 1 {
            Action::Shift(workspace.terminal_targets[index].1, true)
        } else {
            Action::ReplaceShifts(workspace.intern_replace_targets(index, end))
        };
        advance.set(terminal as usize);
        row.push((terminal, action));
        index = end;
    }
    if accepting {
        advance.set(grammar.num_terminals as usize);
        row.push((EOF, Action::Accept));
    }
    Some((
        ActionRow::Sparse(SparseRow::from_sorted_unique(row)),
        advance,
        widest,
    ))
}

fn direct_regular_action_targets<'a>(
    action: &'a Action,
    singleton: &'a mut [u32; 1],
) -> Option<&'a [u32]> {
    match action {
        Action::Shift(target, true) => {
            singleton[0] = *target;
            Some(singleton)
        }
        Action::ReplaceShifts(targets) => Some(targets.as_ref()),
        _ => None,
    }
}

fn merge_direct_regular_actions(
    left: &Action,
    right: &Action,
    interner: &mut DirectRegularActionInterner,
) -> Option<Action> {
    if left == right {
        return Some(left.clone());
    }

    let mut left_singleton = [0u32; 1];
    let mut right_singleton = [0u32; 1];
    let left_targets = direct_regular_action_targets(left, &mut left_singleton)?;
    let right_targets = direct_regular_action_targets(right, &mut right_singleton)?;
    interner.replace_target_scratch.clear();
    interner
        .replace_target_scratch
        .reserve(left_targets.len().saturating_add(right_targets.len()));
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left_targets.len() || right_index < right_targets.len() {
        let next = match (
            left_targets.get(left_index).copied(),
            right_targets.get(right_index).copied(),
        ) {
            (Some(left), Some(right)) if left < right => {
                left_index += 1;
                left
            }
            (Some(left), Some(right)) if right < left => {
                right_index += 1;
                right
            }
            (Some(left), Some(_)) => {
                left_index += 1;
                right_index += 1;
                left
            }
            (Some(left), None) => {
                left_index += 1;
                left
            }
            (None, Some(right)) => {
                right_index += 1;
                right
            }
            (None, None) => break,
        };
        interner.replace_target_scratch.push(next);
    }
    interner.action_from_sorted_scratch()
}

fn push_direct_regular_row_entry(
    row: &mut Vec<(TerminalID, Action)>,
    advance: &mut BitSet,
    widest: &mut Option<(TerminalID, usize)>,
    num_terminals: u32,
    terminal: TerminalID,
    action: Action,
) {
    if terminal == EOF {
        advance.set(num_terminals as usize);
    } else {
        advance.set(terminal as usize);
        let width = match &action {
            Action::Shift(_, true) => 1,
            Action::ReplaceShifts(targets) => targets.len(),
            _ => 0,
        };
        if width != 0 && widest.as_ref().is_none_or(|(_, current)| width > *current) {
            *widest = Some((terminal, width));
        }
    }
    row.push((terminal, action));
}

fn merge_direct_regular_rows(
    left: &ActionRow,
    right: &ActionRow,
    num_terminals: u32,
    interner: &mut DirectRegularActionInterner,
) -> Option<(ActionRow, BitSet, Option<(TerminalID, usize)>)> {
    let mut output = Vec::with_capacity(left.len().saturating_add(right.len()));
    let mut advance = BitSet::new(num_terminals as usize + 1);
    let mut widest = None;
    let mut left = left.iter().peekable();
    let mut right = right.iter().peekable();
    while left.peek().is_some() || right.peek().is_some() {
        match (left.peek().copied(), right.peek().copied()) {
            (Some((left_terminal, left_action)), Some((right_terminal, right_action))) => {
                match left_terminal.cmp(&right_terminal) {
                    std::cmp::Ordering::Less => {
                        push_direct_regular_row_entry(
                            &mut output,
                            &mut advance,
                            &mut widest,
                            num_terminals,
                            left_terminal,
                            left_action.clone(),
                        );
                        left.next();
                    }
                    std::cmp::Ordering::Greater => {
                        push_direct_regular_row_entry(
                            &mut output,
                            &mut advance,
                            &mut widest,
                            num_terminals,
                            right_terminal,
                            right_action.clone(),
                        );
                        right.next();
                    }
                    std::cmp::Ordering::Equal => {
                        let action = merge_direct_regular_actions(
                            left_action,
                            right_action,
                            interner,
                        )?;
                        push_direct_regular_row_entry(
                            &mut output,
                            &mut advance,
                            &mut widest,
                            num_terminals,
                            left_terminal,
                            action,
                        );
                        left.next();
                        right.next();
                    }
                }
            }
            (Some((terminal, action)), None) => {
                push_direct_regular_row_entry(
                    &mut output,
                    &mut advance,
                    &mut widest,
                    num_terminals,
                    terminal,
                    action.clone(),
                );
                left.next();
            }
            (None, Some((terminal, action))) => {
                push_direct_regular_row_entry(
                    &mut output,
                    &mut advance,
                    &mut widest,
                    num_terminals,
                    terminal,
                    action.clone(),
                );
                right.next();
            }
            (None, None) => break,
        }
    }
    Some((
        ActionRow::Sparse(SparseRow::from_sorted_unique(output)),
        advance,
        widest,
    ))
}

fn direct_regular_local_row(
    grammar: &AnalyzedGrammar,
    state_id: u32,
    interner: &mut DirectRegularActionInterner,
) -> Option<(ActionRow, BitSet, Option<(TerminalID, usize)>)> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    let state = automaton.states.get(state_id as usize)?;
    let mut row = Vec::with_capacity(state.transitions.len() + usize::from(state.is_accepting));
    let mut advance = BitSet::new(grammar.num_terminals as usize + 1);
    let mut widest = None;
    for (&terminal, targets) in &state.transitions {
        if terminal >= grammar.num_terminals {
            return None;
        }
        interner.replace_target_scratch.clear();
        for &target in targets {
            interner
                .replace_target_scratch
                .push(target.checked_add(1)?);
        }
        interner.replace_target_scratch.sort_unstable();
        interner.replace_target_scratch.dedup();
        let Some(action) = interner.action_from_sorted_scratch() else {
            continue;
        };
        push_direct_regular_row_entry(
            &mut row,
            &mut advance,
            &mut widest,
            grammar.num_terminals,
            terminal,
            action,
        );
    }
    if state.is_accepting {
        push_direct_regular_row_entry(
            &mut row,
            &mut advance,
            &mut widest,
            grammar.num_terminals,
            EOF,
            Action::Accept,
        );
    }
    Some((
        ActionRow::Sparse(SparseRow::from_sorted_unique(row)),
        advance,
        widest,
    ))
}

fn direct_regular_dag_rows(
    grammar: &AnalyzedGrammar,
) -> Option<(
    (ActionRow, BitSet, Option<(TerminalID, usize)>),
    Vec<(ActionRow, BitSet, Option<(TerminalID, usize)>)>,
)> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    let mut indegree = vec![0u32; automaton.states.len()];
    for state in &automaton.states {
        for &target in &state.epsilons {
            *indegree.get_mut(target as usize)? += 1;
        }
    }
    let mut ready = VecDeque::new();
    for (state, &indegree) in indegree.iter().enumerate() {
        if indegree == 0 {
            ready.push_back(state as u32);
        }
    }
    let mut order = Vec::with_capacity(automaton.states.len());
    while let Some(state_id) = ready.pop_front() {
        order.push(state_id);
        for &target in &automaton.states[state_id as usize].epsilons {
            let target_indegree = &mut indegree[target as usize];
            *target_indegree -= 1;
            if *target_indegree == 0 {
                ready.push_back(target);
            }
        }
    }
    if order.len() != automaton.states.len() {
        return None;
    }

    let mut interner = DirectRegularActionInterner::default();
    let mut rows: Vec<Option<(ActionRow, BitSet, Option<(TerminalID, usize)>)>> =
        (0..automaton.states.len()).map(|_| None).collect();
    for state_id in order.into_iter().rev() {
        let state = &automaton.states[state_id as usize];
        let mut built = direct_regular_local_row(grammar, state_id, &mut interner)?;
        for &target in &state.epsilons {
            let child = rows[target as usize].as_ref()?;
            if built.0.is_empty() {
                built = child.clone();
            } else {
                built = merge_direct_regular_rows(
                    &built.0,
                    &child.0,
                    grammar.num_terminals,
                    &mut interner,
                )?;
            }
        }
        rows[state_id as usize] = Some(built);
    }
    let rows = rows.into_iter().collect::<Option<Vec<_>>>()?;
    let mut initial: Option<(ActionRow, BitSet, Option<(TerminalID, usize)>)> = None;
    for &root in &automaton.start_states {
        let root_row = rows.get(root as usize)?.clone();
        initial = Some(match initial {
            None => root_row,
            Some(current) => merge_direct_regular_rows(
                &current.0,
                &root_row.0,
                grammar.num_terminals,
                &mut interner,
            )?,
        });
    }
    Some((initial?, rows))
}

fn direct_regular_dag_rows_enabled() -> bool {
    std::env::var_os("GLRMASK_DISABLE_DIRECT_REGULAR_DAG_ROWS").is_none()
}

fn direct_regular_wide_frontier_descriptor(
    source_state: u32,
    terminal: TerminalID,
    row: &ActionRow,
) -> Option<DirectRegularWideFrontierDescriptor> {
    let mut target_states = match row.get(&terminal)? {
        Action::Shift(target, _) => vec![*target],
        Action::ReplaceShifts(targets) => targets.to_vec(),
        Action::StackShifts(shifts)
            if shifts
                .iter()
                .all(|shift| shift.pop == 1 && shift.pushes.len() == 1) =>
        {
            shifts.iter().map(|shift| shift.pushes[0]).collect()
        }
        _ => return None,
    };
    target_states.sort_unstable();
    target_states.dedup();
    Some(DirectRegularWideFrontierDescriptor {
        source_state,
        terminal,
        target_states,
    })
}

fn direct_regular_action_row_for_roots(
    grammar: &AnalyzedGrammar,
    roots: impl IntoIterator<Item = u32>,
    workspace: &mut DirectRegularClosureWorkspace,
) -> Option<ActionRow> {
    direct_regular_action_row_for_roots_with_widest(grammar, roots, workspace)
        .map(|(row, _, _)| row)
}


fn direct_regular_compact_table_enabled() -> bool {
    std::env::var("GLRMASK_DISABLE_STATIC_DIRECT_REGULAR_COMPACT_TABLE")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

/// Build the exact direct-regular admission relation without materializing the
/// epsilon-closed action rows. The retained parser automaton executes commits;
/// static masks need only the terminal-support bitset for each parser top.
fn try_build_compact_direct_regular_table(grammar: &AnalyzedGrammar) -> Option<GLRTable> {
    const WIDE_FRONTIER_MIN_TARGETS: usize = 64;

    let automaton = grammar.direct_regular_automaton.as_ref()?;
    if automaton.states.is_empty() || automaton.start_states.is_empty() {
        return None;
    }
    let started_at = std::time::Instant::now();
    let mut parents = vec![Vec::<u32>::new(); automaton.states.len()];
    let mut remaining_children = Vec::<u32>::with_capacity(automaton.states.len());
    let mut ready = VecDeque::<u32>::new();
    let mut target_refs_by_terminal = vec![0usize; grammar.num_terminals as usize];
    for (source, state) in automaton.states.iter().enumerate() {
        remaining_children.push(state.epsilons.len() as u32);
        if state.epsilons.is_empty() {
            ready.push_back(source as u32);
        }
        for &target in &state.epsilons {
            parents.get_mut(target as usize)?.push(source as u32);
        }
        for (&terminal, targets) in &state.transitions {
            if terminal >= grammar.num_terminals
                || targets
                    .iter()
                    .any(|&target| target as usize >= automaton.states.len())
            {
                return None;
            }
            target_refs_by_terminal[terminal as usize] = target_refs_by_terminal
                [terminal as usize]
                .saturating_add(targets.len());
        }
    }

    let mut support = (0..automaton.states.len())
        .map(|_| BitSet::new(grammar.num_terminals as usize + 1))
        .collect::<Vec<_>>();
    let mut bottom_up_order = Vec::with_capacity(automaton.states.len());
    while let Some(raw) = ready.pop_front() {
        let state = &automaton.states[raw as usize];
        let mut row = BitSet::new(grammar.num_terminals as usize + 1);
        for &terminal in state.transitions.keys() {
            row.set(terminal as usize);
        }
        if state.is_accepting {
            row.set(grammar.num_terminals as usize);
        }
        for &child in &state.epsilons {
            row.union_with(&support[child as usize]);
        }
        support[raw as usize] = row;
        bottom_up_order.push(raw);
        for &parent in &parents[raw as usize] {
            let remaining = &mut remaining_children[parent as usize];
            *remaining -= 1;
            if *remaining == 0 {
                ready.push_back(parent);
            }
        }
    }
    if bottom_up_order.len() != automaton.states.len() {
        return None;
    }

    let mut initial = BitSet::new(grammar.num_terminals as usize + 1);
    for &start in &automaton.start_states {
        initial.union_with(support.get(start as usize)?);
    }
    let mut advance = Vec::with_capacity(support.len() + 1);
    advance.push(initial);
    advance.extend(support);

    // A terminal whose complete automaton contains fewer than 64 target
    // references cannot produce a 64-state epsilon-closed frontier. Restrict
    // exact target-set propagation to the few terminals that can matter to the
    // wide-frontier runtime fast path.
    let wide_candidate_terminals = target_refs_by_terminal
        .iter()
        .enumerate()
        .filter_map(|(terminal, &target_refs)| {
            (target_refs >= WIDE_FRONTIER_MIN_TARGETS).then_some(terminal as TerminalID)
        })
        .collect::<Vec<_>>();
    let mut target_sets =
        DirectRegularPersistentTargetSets::new(automaton.states.len().saturating_add(1));
    let mut singleton_roots = vec![None::<u32>; automaton.states.len().saturating_add(1)];
    let mut maximum_frontier = 0usize;
    let mut maximum_frontiers = FxHashMap::<u32, (u32, TerminalID)>::default();
    for terminal in wide_candidate_terminals.iter().copied() {
        let empty = target_sets.empty();
        let mut roots = vec![empty; automaton.states.len()];
        for &raw in &bottom_up_order {
            let state = &automaton.states[raw as usize];
            let mut root = empty;
            if let Some(targets) = state.transitions.get(&terminal) {
                for &target in targets {
                    let parser_target = target as usize + 1;
                    let singleton = if let Some(root) = singleton_roots[parser_target] {
                        root
                    } else {
                        let root = target_sets.singleton(parser_target);
                        singleton_roots[parser_target] = Some(root);
                        root
                    };
                    root = target_sets.union_roots(root, singleton);
                }
            }
            for &child in &state.epsilons {
                root = target_sets.union_roots(root, roots[child as usize]);
            }
            roots[raw as usize] = root;
            let width = target_sets.count(root);
            if width > maximum_frontier {
                maximum_frontier = width;
                maximum_frontiers.clear();
            }
            if width == maximum_frontier && width >= WIDE_FRONTIER_MIN_TARGETS {
                maximum_frontiers
                    .entry(root)
                    .or_insert((raw + 1, terminal));
            }
        }
        let mut initial_root = empty;
        for &start in &automaton.start_states {
            initial_root = target_sets.union_roots(initial_root, roots[start as usize]);
        }
        let width = target_sets.count(initial_root);
        if width > maximum_frontier {
            maximum_frontier = width;
            maximum_frontiers.clear();
        }
        if width == maximum_frontier && width >= WIDE_FRONTIER_MIN_TARGETS {
            maximum_frontiers
                .entry(initial_root)
                .or_insert((0, terminal));
        }
    }
    let mut direct_regular_wide_frontiers = maximum_frontiers
        .into_iter()
        .map(|(root, (source_state, terminal))| DirectRegularWideFrontierDescriptor {
            source_state,
            terminal,
            target_states: target_sets.materialize(root),
        })
        .collect::<Vec<_>>();
    direct_regular_wide_frontiers.sort_unstable_by_key(|descriptor| {
        (descriptor.source_state, descriptor.terminal)
    });

    let num_states = u32::try_from(advance.len()).ok()?;
    let table = GLRTable {
        action: Vec::new(),
        goto: vec![SparseRow::default(); advance.len()],
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance,
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers,
    };
    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][direct_regular_compact_table] states={} support_words={} wide_candidate_terminals={} maximum_frontier={} wide_frontiers={} persistent_nodes={} union_memo={} total_ms={:.3}",
            table.num_states,
            table.advance.iter().map(|row| row.words().len()).sum::<usize>(),
            wide_candidate_terminals.len(),
            maximum_frontier,
            table.direct_regular_wide_frontiers.len(),
            target_sets.nodes.len(),
            target_sets.union_memo.iter().map(FxHashMap::len).sum::<usize>(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(table)
}

fn try_build_direct_regular_table(grammar: &AnalyzedGrammar) -> Option<GLRTable> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    if automaton.states.is_empty() || automaton.start_states.is_empty() {
        return None;
    }

    // Runtime parser state 0 is fixed as the initial stack state. NFA state N
    // maps to parser state N+1. Each row is built directly from the epsilon
    // closure of its corresponding root, reusing one generation-marked DFS
    // workspace rather than allocating and retaining every closure.
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let total_started_at = profile.then(std::time::Instant::now);
    if profile {
        let mut epsilon_edges = 0usize;
        let mut epsilon_forward = 0usize;
        let mut epsilon_backward = 0usize;
        let mut epsilon_self = 0usize;
        let mut epsilon_outdegree_zero = 0usize;
        let mut epsilon_outdegree_one = 0usize;
        let mut epsilon_outdegree_many = 0usize;
        let mut max_epsilon_outdegree = 0usize;
        let mut terminal_edges = 0usize;
        let mut terminal_targets = 0usize;
        let mut epsilon_indegree = vec![0u32; automaton.states.len()];
        for (source, state) in automaton.states.iter().enumerate() {
            epsilon_edges += state.epsilons.len();
            max_epsilon_outdegree = max_epsilon_outdegree.max(state.epsilons.len());
            match state.epsilons.len() {
                0 => epsilon_outdegree_zero += 1,
                1 => epsilon_outdegree_one += 1,
                _ => epsilon_outdegree_many += 1,
            }
            for &target in &state.epsilons {
                epsilon_indegree[target as usize] += 1;
                match (target as usize).cmp(&source) {
                    std::cmp::Ordering::Greater => epsilon_forward += 1,
                    std::cmp::Ordering::Less => epsilon_backward += 1,
                    std::cmp::Ordering::Equal => epsilon_self += 1,
                }
            }
            terminal_edges += state.transitions.len();
            terminal_targets += state.transitions.values().map(Vec::len).sum::<usize>();
        }
        let mut ready = std::collections::VecDeque::new();
        for (state, &indegree) in epsilon_indegree.iter().enumerate() {
            if indegree == 0 {
                ready.push_back(state as u32);
            }
        }
        let mut epsilon_topological_states = 0usize;
        while let Some(state) = ready.pop_front() {
            epsilon_topological_states += 1;
            for &target in &automaton.states[state as usize].epsilons {
                let indegree = &mut epsilon_indegree[target as usize];
                *indegree -= 1;
                if *indegree == 0 {
                    ready.push_back(target);
                }
            }
        }
        eprintln!(
            "[glrmask/profile][direct_regular_topology] states={} start_states={} accepting={} epsilon_edges={} epsilon_forward={} epsilon_backward={} epsilon_self={} epsilon_outdegree_zero={} epsilon_outdegree_one={} epsilon_outdegree_many={} max_epsilon_outdegree={} terminal_edges={} terminal_targets={} epsilon_topological_states={} epsilon_dag={}",
            automaton.states.len(),
            automaton.start_states.len(),
            automaton.states.iter().filter(|state| state.is_accepting).count(),
            epsilon_edges,
            epsilon_forward,
            epsilon_backward,
            epsilon_self,
            epsilon_outdegree_zero,
            epsilon_outdegree_one,
            epsilon_outdegree_many,
            max_epsilon_outdegree,
            terminal_edges,
            terminal_targets,
            epsilon_topological_states,
            epsilon_topological_states == automaton.states.len(),
        );
    }
    let row_started_at = profile.then(std::time::Instant::now);
    let dag_rows = direct_regular_dag_rows_enabled()
        .then(|| direct_regular_dag_rows(grammar))
        .flatten();
    let ((initial, initial_advance, initial_widest), rows_with_widest) =
        if let Some(rows) = dag_rows {
            rows
        } else {
            let mut initial_workspace = DirectRegularClosureWorkspace::new(automaton.states.len());
            let initial = direct_regular_action_row_for_roots_with_widest(
                grammar,
                automaton.start_states.iter().copied(),
                &mut initial_workspace,
            )?;
            let rows = (0..automaton.states.len() as u32)
                .into_par_iter()
                .map_init(
                    || DirectRegularClosureWorkspace::new(automaton.states.len()),
                    |workspace, root| {
                        direct_regular_action_row_for_roots_with_widest(
                            grammar,
                            std::iter::once(root),
                            workspace,
                        )
                    },
                )
                .collect::<Vec<_>>()
                .into_iter()
                .collect::<Option<Vec<_>>>()?;
            (initial, rows)
        };
    let row_ms = row_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let descriptor_started_at = profile.then(std::time::Instant::now);
    let mut widest_candidates = Vec::<(u32, TerminalID, usize)>::new();
    if let Some((terminal, width)) = initial_widest {
        widest_candidates.push((0, terminal, width));
    }
    let mut rows = Vec::with_capacity(rows_with_widest.len());
    let mut advance = Vec::with_capacity(rows_with_widest.len() + 1);
    advance.push(initial_advance);
    for (root, (row, advance_row, widest)) in rows_with_widest.into_iter().enumerate() {
        rows.push(row);
        advance.push(advance_row);
        if let Some((terminal, width)) = widest {
            widest_candidates.push((root as u32 + 1, terminal, width));
        }
    }
    let max_frontier = widest_candidates
        .iter()
        .map(|(_, _, width)| *width)
        .max()
        .unwrap_or(0);
    let direct_regular_wide_frontiers = if max_frontier >= 64 {
        widest_candidates
            .into_iter()
            .filter(|(_, _, width)| *width == max_frontier)
            .filter_map(|(source_state, terminal, _)| {
                let row = if source_state == 0 {
                    &initial
                } else {
                    rows.get(source_state as usize - 1)?
                };
                direct_regular_wide_frontier_descriptor(source_state, terminal, row)
            })
            .collect()
    } else {
        Vec::new()
    };
    let descriptor_ms = descriptor_started_at
        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let mut action = Vec::with_capacity(rows.len() + 1);
    action.push(initial);
    action.extend(rows);

    if profile {
        let mut action_entries = 0usize;
        let mut shift_targets = 0usize;
        let mut stack_shift_actions = 0usize;
        let mut accept_actions = 0usize;
        for row in &action {
            action_entries += row.len();
            for (_, action) in row {
                match action {
                    Action::Shift(_, _) => shift_targets += 1,
                    Action::StackShifts(shifts) => {
                        stack_shift_actions += 1;
                        shift_targets += shifts.len();
                    }
                    Action::ReplaceShifts(targets) => shift_targets += targets.len(),
                    Action::Accept => accept_actions += 1,
                    _ => {}
                }
            }
        }
        eprintln!(
            "[glrmask/profile][direct_regular_table_shape] rows={} action_entries={} shift_targets={} stack_shift_actions={} accept_actions={} max_frontier={}",
            action.len(),
            action_entries,
            shift_targets,
            stack_shift_actions,
            accept_actions,
            max_frontier,
        );
    }

    let num_states = u32::try_from(action.len()).ok()?;
    let table = GLRTable {
        goto: vec![SparseRow::default(); action.len()],
        action,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance,
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers,
    };
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][direct_regular_table_detail] automaton_states={} table_states={} rows_ms={:.3} descriptor_ms={:.3} total_ms={:.3}",
            automaton.states.len(),
            table.num_states,
            row_ms,
            descriptor_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(table)
}

#[cfg(test)]
fn try_build_direct_regular_table_reference(grammar: &AnalyzedGrammar) -> Option<GLRTable> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    if automaton.states.is_empty() || automaton.start_states.is_empty() {
        return None;
    }
    let mut action = Vec::with_capacity(automaton.states.len() + 1);
    let roots = std::iter::once(automaton.start_states.clone())
        .chain((0..automaton.states.len() as u32).map(|root| vec![root]));
    for roots in roots {
        let mut seen = vec![false; automaton.states.len()];
        let mut stack = roots;
        let mut closure = Vec::new();
        while let Some(state) = stack.pop() {
            if state as usize >= automaton.states.len() {
                return None;
            }
            if std::mem::replace(&mut seen[state as usize], true) {
                continue;
            }
            closure.push(state);
            stack.extend(automaton.states[state as usize].epsilons.iter().copied());
        }
        let accepting = closure
            .iter()
            .any(|&state| automaton.states[state as usize].is_accepting);
        let mut targets_by_terminal = BTreeMap::<TerminalID, BTreeSet<u32>>::new();
        for state in closure {
            for (&terminal, targets) in &automaton.states[state as usize].transitions {
                if terminal >= grammar.num_terminals {
                    return None;
                }
                let output = targets_by_terminal.entry(terminal).or_default();
                for &target in targets {
                    if target as usize >= automaton.states.len() {
                        return None;
                    }
                    output.insert(target + 1);
                }
            }
        }
        let mut row = Vec::with_capacity(targets_by_terminal.len() + usize::from(accepting));
        for (terminal, targets) in targets_by_terminal {
            let targets = targets.into_iter().collect::<Vec<_>>();
            let action = if targets.len() == 1 {
                Action::Shift(targets[0], true)
            } else {
                Action::ReplaceShifts(targets.into())
            };
            row.push((terminal, action));
        }
        if accepting {
            row.push((EOF, Action::Accept));
        }
        row.sort_unstable_by_key(|(terminal, _)| *terminal);
        action.push(ActionRow::Sparse(SparseRow::from_sorted_unique(row)));
    }
    let num_states = action.len() as u32;
    let mut table = GLRTable {
        goto: vec![SparseRow::default(); action.len()],
        action,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        embedded_start: Default::default(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.rebuild_advance_rows_from_actions();
    table.rebuild_guarded_shift_index();
    table.compress_default_action_rows();
    Some(table)
}

pub(super) fn build_table(grammar: &AnalyzedGrammar) -> GLRTable {
    build_table_with_default_construction(grammar, GlrTableConstruction::ExperimentalCoreMerged)
}

pub(super) fn build_table_with_default_construction(
    grammar: &AnalyzedGrammar,
    default_construction: GlrTableConstruction,
) -> GLRTable {
    let t1 = std::time::Instant::now();
    let construction_override = glr_table_construction_override();
    if construction_override.is_none()
        && direct_regular_compact_table_enabled()
        && let Some(table) = try_build_compact_direct_regular_table(grammar)
    {
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][glr_table] construction=DirectRegularCompact construction_ms={:.3} pre_merge_states={} post_merge_states={} direct_regular=true",
                t1.elapsed().as_secs_f64() * 1000.0,
                table.num_states,
                table.num_states,
            );
        }
        return table;
    }
    if construction_override.is_none()
        && let Some(table) = try_build_direct_regular_table(grammar)
    {
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][glr_table] construction=DirectRegular construction_ms={:.3} pre_merge_states={} post_merge_states={} direct_regular=true",
                t1.elapsed().as_secs_f64() * 1000.0,
                table.num_states,
                table.num_states,
            );
        }
        return table;
    }
    assert!(
        !grammar.rules.is_empty(),
        "{GLR_TABLE_CONSTRUCTION_ENV} requested an LR table builder, but this grammar has only a direct-regular representation and no CFG rules"
    );
    let construction = selected_glr_table_construction(grammar, default_construction);
    let mut lr1_ms = 0.0;
    let mut table = match construction {
        GlrTableConstruction::LegacyRowBisim => {
            let t0 = std::time::Instant::now();
            let (item_sets, transitions) = build_lr1_item_sets(grammar);
            lr1_ms = t0.elapsed().as_secs_f64() * 1000.0;
            build_legacy_row_bisim_table(grammar, &item_sets, &transitions)
        }
        GlrTableConstruction::Lalr => build_lalr_table(grammar),
        GlrTableConstruction::ExperimentalCoreMerged => {
            let t0 = std::time::Instant::now();
            let (item_sets, transitions) = build_lr1_item_sets(grammar);
            lr1_ms = t0.elapsed().as_secs_f64() * 1000.0;
            build_experimental_core_merged_table(grammar, &item_sets, &transitions)
                .unwrap_or_else(|| build_legacy_row_bisim_table(grammar, &item_sets, &transitions))
        }
    };
    let construction = table.construction;
    let construction_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let pre_merge_states = table.num_states;
    let row_bisim_quotient_skip_reason = if construction == GlrTableConstruction::ExperimentalCoreMerged {
        "construction"
    } else if !row_bisim_quotient_enabled(pre_merge_states) {
        "pre_merge_states"
    } else {
        "none"
    };
    let row_bisim_quotient_enabled = row_bisim_quotient_skip_reason == "none";
    let t2 = std::time::Instant::now();
    let merge_identical1_ms = if row_bisim_quotient_enabled {
        let merge1_started_at = std::time::Instant::now();
        table.merge_identical_rows();
        merge1_started_at.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    // From here on, `action` is allowed to become an optimized execution table
    // containing guarded stack effects. Capture the exact recognizer/admission
    // row support before that lowering so runtime `may_advance` stays a pure
    // row-presence query.
    table.rebuild_advance_rows_from_actions();
    let unit_collapse_skip_reason = if construction == GlrTableConstruction::Lalr {
        "construction"
    } else if !unit_reduction_inlining_enabled() {
        "disabled"
    } else if unit_reduction_inlining_max_pre_merge_states()
        .is_some_and(|max_pre_merge_states| pre_merge_states > max_pre_merge_states)
    {
        "pre_merge_states"
    } else {
        "none"
    };
    let unit_collapse_enabled = unit_collapse_skip_reason == "none";
    let collapse_started_at = std::time::Instant::now();
    let unit_collapse_report = if unit_collapse_enabled {
        Some(
            table.collapse_sr_unit_reductions_with_compatible_gotos_except(
                &grammar.protected_shift_terminals,
            ),
        )
    } else {
        None
    };
    if unit_collapse_enabled {
        // Unit-collapse may append synthetic merged states. Preserve the
        // captured admission semantics for existing rows while backfilling the
        // new synthetic rows from their current action support.
        table.extend_advance_rows_from_actions();
        if !table.advance.is_empty() {
            debug_assert_eq!(table.advance.len(), table.num_states as usize);
        }
    }
    let unit_collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;
    let prune_started_at = std::time::Instant::now();
    let states_before_prune = table.num_states;
    table.prune_unreachable_states();
    let prune_ms = prune_started_at.elapsed().as_secs_f64() * 1000.0;
    // Without unit inlining, the first quotient is already a fixed point and
    // rebuilding `advance` is a pure function of each action row. A second
    // quotient can only reveal something if pruning actually removed states.
    let merge_identical2_needed = row_bisim_quotient_enabled
        && (unit_collapse_enabled || table.num_states != states_before_prune);
    let merge_identical2_ms = if merge_identical2_needed {
        let merge2_started_at = std::time::Instant::now();
        let use_incremental = incremental_row_merge_enabled()
            && table.num_states == states_before_prune
            && unit_collapse_report.as_ref().is_some_and(|report| {
                !report.aborted
                    && report.synthetic_states == 0
                    && !report.changed_original_states.is_empty()
            });
        if use_incremental {
            let report = unit_collapse_report
                .as_ref()
                .expect("incremental post-unit merge requires its report");
            table.merge_identical_rows_from_dirty(&report.changed_original_states);
        } else {
            table.merge_identical_rows();
        }
        merge2_started_at.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let merge_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = std::time::Instant::now();
    let stack_shift_canon_started_at = std::time::Instant::now();
    // The downstream parser and template builders already merge equivalent
    // artifacts. Running the recognizer-only equivalence pass here costs more
    // on large schemas than it saves in later phases.
    if construction == GlrTableConstruction::LegacyRowBisim {
        table.canonicalize_stack_shift_predecessors_except(&grammar.protected_shift_terminals);
    }
    let stack_shift_canon_ms = stack_shift_canon_started_at.elapsed().as_secs_f64() * 1000.0;
    let suffix_quotient_started_at = std::time::Instant::now();
    if construction == GlrTableConstruction::LegacyRowBisim {
        table.quotient_recognizer_stack_suffixes_except(&grammar.protected_shift_terminals);
    }
    let suffix_quotient_ms = suffix_quotient_started_at.elapsed().as_secs_f64() * 1000.0;
    let recog_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let _ = (
        lr1_ms,
        construction_ms,
        pre_merge_states,
        merge_ms,
        merge_identical1_ms,
        unit_collapse_ms,
        prune_ms,
        merge_identical2_ms,
        recog_ms,
    );

    if construction == GlrTableConstruction::LegacyRowBisim && default_action_rows_enabled() {
        table.compress_default_action_rows();
    }

    table.rebuild_guarded_shift_index();

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][glr_table] construction={:?} lr1_item_sets_ms={:.3} construction_ms={:.3} pre_merge_states={} post_merge_states={} row_bisim_quotient={} row_bisim_quotient_skip_reason={} unit_collapse={} unit_collapse_aborted={} unit_collapse_reason={} unit_collapse_skip_reason={} merge_ms={:.3} merge_identical1_ms={:.3} unit_collapse_ms={:.3} prune_ms={:.3} merge_identical2_needed={} merge_identical2_ms={:.3} stack_shift_canon_ms={:.3} suffix_quotient_ms={:.3}",
            construction,
            lr1_ms,
            construction_ms,
            pre_merge_states,
            table.num_states,
            row_bisim_quotient_enabled,
            row_bisim_quotient_skip_reason,
            unit_collapse_enabled,
            unit_collapse_report
                .as_ref()
                .is_some_and(|report| report.aborted),
            unit_collapse_report
                .as_ref()
                .and_then(|report| report.reason)
                .unwrap_or("none"),
            unit_collapse_skip_reason,
            merge_ms,
            merge_identical1_ms,
            unit_collapse_ms,
            prune_ms,
            merge_identical2_needed,
            merge_identical2_ms,
            stack_shift_canon_ms,
            suffix_quotient_ms,
        );
    }

    table
}

fn replace_shifts_enabled() -> bool {
    true
}

fn replace_gotos_enabled() -> bool {
    true
}

std::thread_local! {
    pub static LOCAL_FORWARD_REPLACE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn local_forward_replace_enabled() -> bool {
    LOCAL_FORWARD_REPLACE_OVERRIDE.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        false
    })
}



#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Item {
    pub(super) rule: u32,
    pub(super) dot: u32,
    pub(super) stack_depth: u32,
}

impl Item {
    pub(super) fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {
        Self { rule, dot, stack_depth }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub(super) struct PendingAction {
    pub(super) shift: Option<(u32, bool)>,
    pub(super) reduces: Vec<(NonterminalID, u32)>,
    pub(super) accept: bool,
}

fn pending_identity_row_fingerprint(
    pending: &FxHashMap<TerminalID, PendingAction>,
    goto: &FxHashMap<NonterminalID, (u32, bool)>,
) -> u64 {
    // Hash-map iteration order is deliberately irrelevant. Combine strong
    // per-entry Fx hashes commutatively, then include row cardinalities. Any
    // fingerprint collision is verified by exact map equality before merging.
    let mut action_sum = 0u64;
    let mut action_xor = 0u64;
    for (&terminal, action) in pending {
        let mut hasher = FxHasher::default();
        0xA1u8.hash(&mut hasher);
        terminal.hash(&mut hasher);
        action.hash(&mut hasher);
        let hash = hasher.finish();
        action_sum = action_sum.wrapping_add(hash);
        action_xor ^= hash.rotate_left((hash >> 58) as u32);
    }
    let mut goto_sum = 0u64;
    let mut goto_xor = 0u64;
    for (&nonterminal, &(target, replace)) in goto {
        let mut hasher = FxHasher::default();
        0xB2u8.hash(&mut hasher);
        nonterminal.hash(&mut hasher);
        target.hash(&mut hasher);
        replace.hash(&mut hasher);
        let hash = hasher.finish();
        goto_sum = goto_sum.wrapping_add(hash);
        goto_xor ^= hash.rotate_left((hash >> 58) as u32);
    }
    let mut hasher = FxHasher::default();
    pending.len().hash(&mut hasher);
    action_sum.hash(&mut hasher);
    action_xor.hash(&mut hasher);
    goto.len().hash(&mut hasher);
    goto_sum.hash(&mut hasher);
    goto_xor.hash(&mut hasher);
    hasher.finish()
}

fn finish_table_with_early_identity_quotient(
    grammar: &AnalyzedGrammar,
    pending: Vec<FxHashMap<TerminalID, PendingAction>>,
    goto: Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    forwarded_shifts: FxHashSet<(u32, TerminalID)>,
    construction: GlrTableConstruction,
    admission_policy: AdmissionPolicy,
    fixed_point: bool,
) -> GLRTable {
    let mut pending = pending;
    let mut goto = goto;
    let mut forwarded_shifts = forwarded_shifts;
    for row in &mut pending {
        for action in row.values_mut() {
            action.reduces.sort_unstable();
            action.reduces.dedup();
        }
    }
    loop {
        let old_state_count = pending.len();
        let mut first_by_fingerprint = FxHashMap::<u64, Vec<u32>>::default();
        let mut representative = (0..old_state_count as u32).collect::<Vec<_>>();
        for state in 0..old_state_count {
            let fingerprint = pending_identity_row_fingerprint(&pending[state], &goto[state]);
            let candidates = first_by_fingerprint.entry(fingerprint).or_default();
            if let Some(&existing) = candidates.iter().find(|&&candidate| {
                pending[candidate as usize] == pending[state]
                    && goto[candidate as usize] == goto[state]
            }) {
                representative[state] = existing;
            } else {
                candidates.push(state as u32);
            }
        }

        if representative
            .iter()
            .enumerate()
            .all(|(state, &rep)| rep == state as u32)
        {
            break;
        }

        let mut rep_to_new = vec![u32::MAX; old_state_count];
        let mut kept = Vec::new();
        for state in 0..old_state_count {
            if representative[state] == state as u32 {
                rep_to_new[state] = kept.len() as u32;
                kept.push(state);
            }
        }
        let mapping = representative
            .iter()
            .map(|&rep| rep_to_new[rep as usize])
            .collect::<Vec<_>>();

        let mut next_pending = Vec::with_capacity(kept.len());
        let mut next_goto = Vec::with_capacity(kept.len());
        for (state, (mut by_terminal, mut by_nonterminal)) in
            pending.into_iter().zip(goto).enumerate()
        {
            if representative[state] != state as u32 {
                continue;
            }
            for pending in by_terminal.values_mut() {
                if let Some((target, replace)) = pending.shift {
                    pending.shift = Some((mapping[target as usize], replace));
                }
            }
            for (target, _) in by_nonterminal.values_mut() {
                *target = mapping[*target as usize];
            }
            next_pending.push(by_terminal);
            next_goto.push(by_nonterminal);
        }
        forwarded_shifts = forwarded_shifts
            .into_iter()
            .map(|(state, terminal)| (mapping[state as usize], terminal))
            .collect();
        pending = next_pending;
        goto = next_goto;
        if !fixed_point {
            break;
        }
    }

    let mut action = Vec::with_capacity(pending.len());
    let mut new_goto = Vec::with_capacity(goto.len());
    for (by_terminal, by_nonterminal) in pending.into_iter().zip(goto) {
        let mut action_entries = by_terminal.into_iter().collect::<Vec<_>>();
        action_entries.sort_unstable_by_key(|(terminal, _)| *terminal);
        action.push(ActionRow::Sparse(SparseRow::from_sorted_unique(
            action_entries
                .into_iter()
                .map(|(terminal, pending)| (terminal, pending.finish()))
                .collect(),
        )));
        new_goto.push(SparseRow::from_hash_map(by_nonterminal));
    }

    let num_states = action.len() as u32;
    GLRTable {
        action,
        goto: new_goto,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: grammar.rules.len() as u32,
        rules: grammar.rules.clone(),
        nonterminal_display_names: grammar.nonterminal_display_names.clone(),
        embedded_start: Default::default(),
        construction,
        admission_policy,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts,
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    }
}

impl PendingAction {
    pub(super) fn push_shift(&mut self, target: u32, is_replace: bool) {
        match self.shift {
            Some((existing, _)) => debug_assert_eq!(existing, target),
            None => self.shift = Some((target, is_replace)),
        }
    }

    pub(super) fn push_reduce(&mut self, nt: NonterminalID, len: u32) {
        self.reduces.push((nt, len));
    }

    pub(super) fn push_accept(&mut self) {
        self.accept = true;
    }

    pub(super) fn maybe_finish(mut self) -> Option<Action> {
        self.reduces.sort_unstable();
        self.reduces.dedup();
        match (self.shift, self.reduces.len(), self.accept) {
            (None, 0, false) => None,
            (Some((target, replace)), 0, false) => Some(Action::Shift(target, replace)),
            (None, 1, false) => Some(Action::Reduce(self.reduces[0].0, self.reduces[0].1)),
            (None, 0, true) => Some(Action::Accept),
            (shift, _, accept) => Some(Action::Split {
                shift,
                reduces: self.reduces,
                accept,
            }),
        }
    }

    pub(super) fn finish(self) -> Action {
        self.maybe_finish()
            .expect("PendingAction::finish called on an empty action")
    }
}

fn initialize_pending_and_goto(
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> (
    Vec<FxHashMap<TerminalID, PendingAction>>,
    Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    FxHashSet<(u32, TerminalID)>,
) {
    let mut pending = std::iter::repeat_with(FxHashMap::<TerminalID, PendingAction>::default)
        .take(transitions.len())
        .collect::<Vec<_>>();
    let mut goto: Vec<FxHashMap<NonterminalID, (u32, bool)>> = (0..transitions.len()).map(|_| FxHashMap::default()).collect();
    let mut forwarded_shifts = FxHashSet::default();

    for (state_id, by_symbol) in transitions.iter().enumerate() {
        for (symbol, &(target, is_replace, is_forwarded)) in by_symbol {
            match symbol {
                Symbol::Terminal(terminal) => {
                    pending[state_id]
                        .entry(*terminal)
                        .or_default()
                        .push_shift(target, is_replace);
                    if is_forwarded {
                        forwarded_shifts.insert((state_id as u32, *terminal));
                    }
                }
                Symbol::Nonterminal(nonterminal) => {
                    goto[state_id].insert(*nonterminal, (target, is_replace));
                }
            }
        }
    }

    (pending, goto, forwarded_shifts)
}

fn finish_table(
    grammar: &AnalyzedGrammar,
    pending: Vec<FxHashMap<TerminalID, PendingAction>>,
    goto: Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    forwarded_shifts: FxHashSet<(u32, TerminalID)>,
    construction: GlrTableConstruction,
    admission_policy: AdmissionPolicy,
) -> GLRTable {
    let action: Vec<ActionRow> = pending
        .into_iter()
        .map(|by_terminal| {
            let mut entries = by_terminal
                .into_iter()
                .collect::<Vec<_>>();
            // Preserve canonical order for stable construction and artifacts.
            entries.sort_unstable_by_key(|(terminal, _)| *terminal);
            ActionRow::Sparse(SparseRow::from_sorted_unique(
                entries
                    .into_iter()
                    .map(|(terminal, pending)| (terminal, pending.finish()))
                    .collect(),
            ))
        })
        .collect();
    let goto: Vec<GotoRow> = goto
        .into_iter()
        .map(SparseRow::from_hash_map)
        .collect();
    let num_states = action.len() as u32;

    GLRTable {
        action,
        goto,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: grammar.rules.len() as u32,
        rules: grammar.rules.clone(),
        nonterminal_display_names: grammar.nonterminal_display_names.clone(),
        embedded_start: Default::default(),
        construction,
        admission_policy,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts,
        control_terminals: Default::default(),
        skip_terminals: Default::default(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    }
}

#[derive(Debug, Clone)]
struct Lr0State {
    // Canonical LR(0) kernels are already sorted/deduplicated before state
    // interning. Retain that exact sequence in one shared contiguous
    // allocation rather than rebuilding a BTreeSet for every accepted state.
    // The kernel index holds an Arc clone of the same allocation while the
    // automaton is being constructed.
    kernel: Arc<[Item]>,
    closure: Vec<Item>,
}

fn item_next_symbol<'a>(item: &Item, rules: &'a [Rule]) -> Option<&'a Symbol> {
    rules[item.rule as usize].rhs.get(item.dot as usize)
}

fn lr0_closure(grammar: &AnalyzedGrammar, kernel: &[Item]) -> Vec<Item> {
    let mut result = kernel.iter().copied().collect::<BTreeSet<_>>();
    let mut queue: VecDeque<Item> = kernel.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        let Some(Symbol::Nonterminal(nonterminal)) = item_next_symbol(&item, &grammar.rules) else {
            continue;
        };
        for &rule_id in &grammar.rules_by_lhs[*nonterminal as usize] {
            let stack_depth = grammar.rules[rule_id as usize].rhs.len() as u32;
            let next = Item::new(rule_id, 0, stack_depth);
            if result.insert(next) {
                queue.push_back(next);
            }
        }
    }

    result.into_iter().collect()
}

struct Lr0ClosureScratch {
    nonterminal_marks: Vec<u32>,
    generation: u32,
    queue: VecDeque<NonterminalID>,
}

impl Lr0ClosureScratch {
    fn new(grammar: &AnalyzedGrammar) -> Self {
        Self {
            nonterminal_marks: vec![0; grammar.rules_by_lhs.len()],
            generation: 0,
            queue: VecDeque::new(),
        }
    }

    fn begin(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.nonterminal_marks.fill(0);
            self.generation = 1;
        }
        self.queue.clear();
    }

    #[inline]
    fn enqueue_once(&mut self, nonterminal: NonterminalID) {
        let index = nonterminal as usize;
        let Some(mark) = self.nonterminal_marks.get_mut(index) else {
            return;
        };
        if *mark == self.generation {
            return;
        }
        *mark = self.generation;
        self.queue.push_back(nonterminal);
    }
}

fn lr0_closure_fast(
    grammar: &AnalyzedGrammar,
    kernel: &[Item],
    scratch: &mut Lr0ClosureScratch,
) -> Vec<Item> {
    scratch.begin();
    let mut result = Vec::with_capacity(kernel.len().saturating_add(8));
    result.extend(kernel.iter().copied());

    for item in kernel {
        if let Some(Symbol::Nonterminal(nonterminal)) = item_next_symbol(item, &grammar.rules) {
            scratch.enqueue_once(*nonterminal);
        }
    }

    while let Some(nonterminal) = scratch.queue.pop_front() {
        for &rule_id in &grammar.rules_by_lhs[nonterminal as usize] {
            let rule = &grammar.rules[rule_id as usize];
            let item = Item::new(rule_id, 0, rule.rhs.len() as u32);
            result.push(item);
            if let Some(Symbol::Nonterminal(next_nonterminal)) = rule.rhs.first() {
                scratch.enqueue_once(*next_nonterminal);
            }
        }
    }

    result.sort_unstable();
    result.dedup();
    #[cfg(debug_assertions)]
    debug_assert_eq!(result, lr0_closure(grammar, kernel));
    result
}

fn build_lr0_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<Lr0State>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    if std::env::var_os("GLRMASK_DISABLE_FLAT_LR0_KERNELS").is_none() {
        let result = build_lr0_item_sets_flat(grammar);
        if std::env::var_os("GLRMASK_ASSERT_FLAT_LR0_KERNELS").is_some() {
            let reference = build_lr0_item_sets_tree(grammar);
            assert_eq!(result.1, reference.1);
            assert_eq!(result.0.len(), reference.0.len());
            for (left, right) in result.0.iter().zip(&reference.0) {
                assert_eq!(left.kernel, right.kernel);
                assert_eq!(left.closure, right.closure);
            }
        }
        return result;
    }
    build_lr0_item_sets_tree(grammar)
}

// Sorting the same (symbol, advanced-item) pairs produces exactly the order
// of the original BTreeMap<Symbol, BTreeSet<Item>>. Only novel kernels need
// a shared contiguous allocation; hits avoid temporary trees.
fn build_lr0_item_sets_flat(
    grammar: &AnalyzedGrammar,
) -> (Vec<Lr0State>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let mut scratch = Lr0ClosureScratch::new(grammar);
    let start_kernel: Arc<[Item]> =
        Arc::from([Item::new(0, 0, grammar.rules[0].rhs.len() as u32)]);
    let start_closure = lr0_closure_fast(grammar, start_kernel.as_ref(), &mut scratch);
    let mut state_by_kernel = FxHashMap::<Arc<[Item]>, u32>::default();
    state_by_kernel.insert(Arc::clone(&start_kernel), 0);
    let mut states = vec![Lr0State { kernel: start_kernel, closure: start_closure }];
    let mut transitions = vec![BTreeMap::new()];
    let mut queue = VecDeque::from([0u32]);
    let mut advanced = Vec::<(Symbol, Item)>::new();
    let mut key = Vec::<Item>::new();
    while let Some(source) = queue.pop_front() {
        advanced.clear();
        for item in &states[source as usize].closure {
            if let Some(symbol) = item_next_symbol(item, &grammar.rules) {
                advanced.push((symbol.clone(), Item::new(item.rule, item.dot + 1, item.stack_depth)));
            }
        }
        advanced.sort_unstable();
        advanced.dedup();
        for group in advanced.chunk_by(|left, right| left.0 == right.0) {
            let symbol = &group[0].0;
            let has_dot_1 = group.iter().any(|(_, item)| item.dot == 1);
            let is_replace = !has_dot_1 && match symbol {
                Symbol::Terminal(_) => replace_shifts_enabled(),
                Symbol::Nonterminal(_) => replace_gotos_enabled(),
            };
            key.clear();
            key.extend(group.iter().map(|(_, item)| {
                if is_replace {
                    Item::new(item.rule, item.dot, item.stack_depth.saturating_sub(1))
                } else { *item }
            }));
            // Saturating depth adjustment may merge distinct input items.
            key.sort_unstable();
            key.dedup();
            let target = if let Some(&target) = state_by_kernel.get(key.as_slice()) {
                target
            } else {
                let target = states.len() as u32;
                let kernel: Arc<[Item]> = Arc::from(key.as_slice());
                let closure = lr0_closure_fast(grammar, kernel.as_ref(), &mut scratch);
                state_by_kernel.insert(Arc::clone(&kernel), target);
                states.push(Lr0State { kernel, closure });
                transitions.push(BTreeMap::new());
                queue.push_back(target);
                target
            };
            transitions[source as usize].insert(symbol.clone(), (target, is_replace, false));
        }
    }
    (states, transitions)
}

fn build_lr0_item_sets_tree(
    grammar: &AnalyzedGrammar,
) -> (Vec<Lr0State>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let mut closure_scratch = Lr0ClosureScratch::new(grammar);
    let mut start_kernel = BTreeSet::new();
    start_kernel.insert(Item::new(0, 0, grammar.rules[0].rhs.len() as u32));
    let start_kernel_vec = start_kernel.iter().copied().collect::<Vec<_>>();
    let start_closure = lr0_closure_fast(grammar, &start_kernel_vec, &mut closure_scratch);
    let start_kernel: Arc<[Item]> = Arc::from(start_kernel_vec);

    let mut states = vec![Lr0State {
        kernel: Arc::clone(&start_kernel),
        closure: start_closure,
    }];
    let mut transitions = vec![BTreeMap::new()];
    let mut state_by_kernel: FxHashMap<Arc<[Item]>, u32> = FxHashMap::default();
    state_by_kernel.insert(start_kernel, 0);

    let mut queue = VecDeque::from([0u32]);
    while let Some(source) = queue.pop_front() {
        let mut kernels: BTreeMap<Symbol, BTreeSet<Item>> = BTreeMap::new();
        for item in &states[source as usize].closure {
            let Some(symbol) = item_next_symbol(item, &grammar.rules) else {
                continue;
            };
            kernels
                .entry(symbol.clone())
                .or_default()
                .insert(Item::new(item.rule, item.dot + 1, item.stack_depth));
        }

        for (symbol, kernel) in kernels {
            let has_dot_1 = kernel.iter().any(|item| item.dot == 1);
            let is_replace = match &symbol {
                Symbol::Terminal(_) => !has_dot_1 && replace_shifts_enabled(),
                Symbol::Nonterminal(_) => !has_dot_1 && replace_gotos_enabled(),
            };

            let adjusted_kernel: BTreeSet<Item> = if is_replace {
                kernel
                    .iter()
                    .map(|item| Item::new(item.rule, item.dot, item.stack_depth.saturating_sub(1)))
                    .collect()
            } else {
                kernel
            };
            if adjusted_kernel.is_empty() {
                continue;
            }

            let key = adjusted_kernel.iter().copied().collect::<Vec<_>>();
            let target = if let Some(&target) = state_by_kernel.get(key.as_slice()) {
                target
            } else {
                let target = states.len() as u32;
                let kernel: Arc<[Item]> = Arc::from(key);
                let closure = lr0_closure_fast(grammar, kernel.as_ref(), &mut closure_scratch);
                state_by_kernel.insert(Arc::clone(&kernel), target);
                states.push(Lr0State {
                    kernel,
                    closure,
                });
                transitions.push(BTreeMap::new());
                queue.push_back(target);
                target
            };

            transitions[source as usize].insert(symbol, (target, is_replace, false));
        }
    }

    (states, transitions)
}

fn sequence_nullable(symbols: &[Symbol], nullable: &BTreeSet<NonterminalID>) -> bool {
    symbols.iter().all(|symbol| match symbol {
        Symbol::Terminal(_) => false,
        Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
    })
}

fn lalr_global_node_id(offsets: &[usize], state: usize, item: usize) -> usize {
    offsets[state] + item
}

#[inline]
fn union_lalr_lookahead_nodes(
    lookaheads: &mut [BitSet],
    source_node: usize,
    target_node: usize,
) -> bool {
    if source_node == target_node {
        return false;
    }
    if source_node < target_node {
        let (before_target, target_and_after) = lookaheads.split_at_mut(target_node);
        target_and_after[0].union_with_changed(&before_target[source_node])
    } else {
        let (before_source, source_and_after) = lookaheads.split_at_mut(source_node);
        before_source[target_node].union_with_changed(&source_and_after[0])
    }
}

fn compute_lalr_item_lookaheads(
    grammar: &AnalyzedGrammar,
    states: &[Lr0State],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> Vec<Vec<BitSet>> {
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let total_started = profile.then(std::time::Instant::now);
    let lookahead_len = grammar.num_terminals as usize + 1;
    let suffix_started = profile.then(std::time::Instant::now);
    let suffix_first = rule_suffix_first_sets(grammar);
    let suffix_ms = suffix_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

    let index_started = profile.then(std::time::Instant::now);
    let mut offsets = Vec::with_capacity(states.len() + 1);
    offsets.push(0usize);
    for state in states {
        offsets.push(offsets.last().copied().unwrap() + state.closure.len());
    }
    let total_nodes = *offsets.last().unwrap();

    let mut item_index_by_state = Vec::with_capacity(states.len());
    for state in states {
        let mut index = FxHashMap::default();
        index.reserve(state.closure.len());
        for (item_index, item) in state.closure.iter().enumerate() {
            index.insert(*item, item_index);
        }
        item_index_by_state.push(index);
    }
    let index_ms = index_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

    let alloc_started = profile.then(std::time::Instant::now);
    let mut lookaheads = vec![BitSet::new(lookahead_len); total_nodes];
    let mut edges = vec![Vec::<usize>::new(); total_nodes];
    let mut worklist = VecDeque::<usize>::new();
    let mut queued = vec![false; total_nodes];
    let alloc_ms = alloc_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

    let start = Item::new(0, 0, grammar.rules[0].rhs.len() as u32);
    if let Some(&start_index) = item_index_by_state[0].get(&start) {
        let start_node = lalr_global_node_id(&offsets, 0, start_index);
        lookaheads[start_node].set(lookahead_bit(EOF, grammar.num_terminals));
        worklist.push_back(start_node);
        queued[start_node] = true;
    }

    let graph_started = profile.then(std::time::Instant::now);
    let mut spontaneous_updates = 0usize;
    for (state_id, state) in states.iter().enumerate() {
        for (item_index, item) in state.closure.iter().enumerate() {
            let source_node = lalr_global_node_id(&offsets, state_id, item_index);

            if let Some(symbol) = item_next_symbol(item, &grammar.rules) {
                let Some(&(target_state, is_replace, _)) = transitions[state_id].get(symbol) else {
                    continue;
                };
                let mut advanced = Item::new(item.rule, item.dot + 1, item.stack_depth);
                if is_replace {
                    advanced.stack_depth = advanced.stack_depth.saturating_sub(1);
                }
                if let Some(&target_item_index) = item_index_by_state[target_state as usize].get(&advanced) {
                    edges[source_node].push(lalr_global_node_id(
                        &offsets,
                        target_state as usize,
                        target_item_index,
                    ));
                }
            }

            let Some(Symbol::Nonterminal(nonterminal)) = item_next_symbol(item, &grammar.rules) else {
                continue;
            };
            let suffix = &suffix_first[item.rule as usize];
            let suffix_index = item.dot as usize + 1;
            let spontaneous = &suffix.first[suffix_index];
            let propagates = suffix.nullable[suffix_index];

            for &rule_id in &grammar.rules_by_lhs[*nonterminal as usize] {
                let closure_item = Item::new(rule_id, 0, grammar.rules[rule_id as usize].rhs.len() as u32);
                let Some(&target_item_index) = item_index_by_state[state_id].get(&closure_item) else {
                    continue;
                };
                let target_node = lalr_global_node_id(&offsets, state_id, target_item_index);
                if !spontaneous.is_empty() {
                    if lookaheads[target_node].union_with_changed(spontaneous) {
                        spontaneous_updates += 1;
                        if !queued[target_node] {
                            queued[target_node] = true;
                            worklist.push_back(target_node);
                        }
                    }
                }
                if propagates {
                    edges[source_node].push(target_node);
                }
            }
        }
    }
    let graph_ms = graph_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

    let dedup_started = profile.then(std::time::Instant::now);
    for targets in &mut edges {
        targets.sort_unstable();
        targets.dedup();
    }
    let dedup_ms = dedup_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
    let edge_count = profile
        .then(|| edges.iter().map(Vec::len).sum::<usize>())
        .unwrap_or(0);

    let propagation_started = profile.then(std::time::Instant::now);
    let mut propagation_pops = 0usize;
    let mut propagation_updates = 0usize;
    while let Some(source_node) = worklist.pop_front() {
        propagation_pops += 1;
        queued[source_node] = false;
        for &target_node in &edges[source_node] {
            if !union_lalr_lookahead_nodes(&mut lookaheads, source_node, target_node) {
                continue;
            }
            propagation_updates += 1;
            if !queued[target_node] {
                queued[target_node] = true;
                worklist.push_back(target_node);
            }
        }
    }
    let propagation_ms = propagation_started
        .map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

    let collect_started = profile.then(std::time::Instant::now);
    let mut flat_lookaheads = lookaheads.into_iter();
    let result = states
        .iter()
        .map(|state| flat_lookaheads.by_ref().take(state.closure.len()).collect())
        .collect();
    if let Some(total_started) = total_started {
        eprintln!(
            "[glrmask/profile][lalr_lookahead] states={} total_nodes={} edges={} spontaneous_updates={} propagation_pops={} propagation_updates={} suffix_ms={:.3} index_ms={:.3} alloc_ms={:.3} graph_ms={:.3} dedup_ms={:.3} propagation_ms={:.3} collect_ms={:.3} total_ms={:.3}",
            states.len(),
            total_nodes,
            edge_count,
            spontaneous_updates,
            propagation_pops,
            propagation_updates,
            suffix_ms,
            index_ms,
            alloc_ms,
            graph_ms,
            dedup_ms,
            propagation_ms,
            collect_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0),
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

fn pending_action_has_conflict(action: &PendingAction) -> bool {
    let mut branches = usize::from(action.shift.is_some()) + usize::from(action.accept);
    let mut unique_reduces = 0usize;
    for (index, reduce) in action.reduces.iter().enumerate() {
        if !action.reduces[..index].contains(reduce) {
            unique_reduces += 1;
            if branches + unique_reduces > 1 {
                return true;
            }
        }
    }
    branches += unique_reduces;
    branches > 1
}

fn pending_table_has_conflict(pending: &[FxHashMap<TerminalID, PendingAction>]) -> bool {
    pending
        .iter()
        .flat_map(FxHashMap::values)
        .any(pending_action_has_conflict)
}

/// Check whether FOLLOW-set (SLR) reductions would introduce any parse-table
/// conflict without actually materializing them. The adaptive SLR path used to
/// insert all broad FOLLOW reductions into `PendingAction`, scan the completed
/// table, then throw that work away on the grammars which need LALR. Tail JSON
/// schemas can have thousands of LR(0) states, so that failed speculation is a
/// material fraction of compile time.
///
/// This mirrors `pending_action_has_conflict` exactly: one existing terminal
/// shift is one branch, accept is one branch, and duplicate identical reduces
/// count only once.
fn slr_reductions_would_conflict(
    grammar: &AnalyzedGrammar,
    states: &[Lr0State],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> bool {
    let lookahead_len = grammar.num_terminals as usize + 1;
    // Per-lookahead state for one LR(0) state. A generation stamp avoids
    // clearing these small arrays for every parser state.
    let mut stamp = vec![0u32; lookahead_len];
    let mut first_reduce = vec![(0u32, 0u32); lookahead_len];
    let mut saw_accept = vec![false; lookahead_len];
    let mut generation = 0u32;

    for (state_id, state) in states.iter().enumerate() {
        generation = generation.wrapping_add(1);
        if generation == 0 {
            stamp.fill(0);
            generation = 1;
        }
        let by_symbol = &transitions[state_id];
        for item in &state.closure {
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }
            let reduce = (rule.lhs, item.stack_depth);
            for bit in grammar.follow[rule.lhs as usize].iter_ones() {
                let lookahead = bit_lookahead(bit, grammar.num_terminals);
                let slot = bit;
                let has_shift = by_symbol.contains_key(&Symbol::Terminal(lookahead));
                if stamp[slot] != generation {
                    stamp[slot] = generation;
                    saw_accept[slot] = item.rule == 0;
                    first_reduce[slot] = reduce;
                    if has_shift {
                        return true;
                    }
                    continue;
                }

                if has_shift {
                    return true;
                }
                if item.rule == 0 {
                    // Repeated accepts are one branch; accept plus any reduce
                    // is a conflict.
                    if !saw_accept[slot] {
                        return true;
                    }
                } else {
                    if saw_accept[slot] || first_reduce[slot] != reduce {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn add_completed_lr0_reductions(
    grammar: &AnalyzedGrammar,
    states: &[Lr0State],
    lookaheads: Option<&[Vec<BitSet>]>,
    pending: &mut [FxHashMap<TerminalID, PendingAction>],
) {
    for (state_id, state) in states.iter().enumerate() {
        for (item_index, item) in state.closure.iter().enumerate() {
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            let reduction_lookaheads = lookaheads.map_or_else(
                || &grammar.follow[rule.lhs as usize],
                |lookaheads| &lookaheads[state_id][item_index],
            );
            for bit in reduction_lookaheads.iter_ones() {
                let lookahead = bit_lookahead(bit, grammar.num_terminals);
                if item.rule == 0 {
                    pending[state_id]
                        .entry(lookahead)
                        .or_default()
                        .push_accept();
                } else {
                    pending[state_id]
                        .entry(lookahead)
                        .or_default()
                        .push_reduce(rule.lhs, item.stack_depth);
                }
            }
        }
    }
}

fn build_lalr_table(grammar: &AnalyzedGrammar) -> GLRTable {
    build_lalr_table_impl(grammar, lalr_early_identity_quotient_enabled(), false)
}

fn lalr_early_identity_quotient_enabled() -> bool {
    std::env::var("GLRMASK_LALR_EARLY_IDENTITY_QUOTIENT")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

fn build_lalr_table_impl(
    grammar: &AnalyzedGrammar,
    early_identity_quotient: bool,
    early_identity_fixed_point: bool,
) -> GLRTable {
    let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let total_started = profile.then(std::time::Instant::now);
    let started = profile.then(std::time::Instant::now);
    let (states, transitions) = build_lr0_item_sets(grammar);
    let lr0_ms = started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

    let try_slr_fast_path = std::env::var("GLRMASK_LALR_TRY_SLR_FAST_PATH")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);

    let started = profile.then(std::time::Instant::now);
    let (mut pending, goto, forwarded_shifts) = initialize_pending_and_goto(&transitions);
    let init_ms = started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

    let reduce_started = profile.then(std::time::Instant::now);
    let mut selected_slr = false;
    let mut slr_conflict = false;
    let mut lookahead_ms = 0.0;
    if try_slr_fast_path {
        slr_conflict = slr_reductions_would_conflict(grammar, &states, &transitions);
        if !slr_conflict {
            add_completed_lr0_reductions(grammar, &states, None, &mut pending);
            selected_slr = true;
        }
    }
    if !selected_slr {
        let started = profile.then(std::time::Instant::now);
        let lookaheads = compute_lalr_item_lookaheads(grammar, &states, &transitions);
        lookahead_ms = started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
        add_completed_lr0_reductions(grammar, &states, Some(&lookaheads), &mut pending);
    }

    let reduce_ms = reduce_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
    let finish_started = profile.then(std::time::Instant::now);
    let table = if early_identity_quotient {
        finish_table_with_early_identity_quotient(
            grammar,
            pending,
            goto,
            forwarded_shifts,
            GlrTableConstruction::Lalr,
            AdmissionPolicy::ExactSimulation,
            early_identity_fixed_point,
        )
    } else {
        finish_table(
            grammar,
            pending,
            goto,
            forwarded_shifts,
            GlrTableConstruction::Lalr,
            AdmissionPolicy::ExactSimulation,
        )
    };
    if let Some(total_started) = total_started {
        eprintln!(
            "[glrmask/profile][lalr_detail] try_slr={} selected_slr={} slr_conflict={} lr0_ms={:.3} lookahead_ms={:.3} init_ms={:.3} reduce_ms={:.3} finish_ms={:.3} total_ms={:.3} states={} transitions={}",
            try_slr_fast_path,
            selected_slr,
            slr_conflict,
            lr0_ms,
            lookahead_ms,
            init_ms,
            reduce_ms,
            finish_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0),
            total_started.elapsed().as_secs_f64() * 1000.0,
            states.len(),
            transitions.len(),
        );
    }
    table
}

// LR(1) item set construction.

fn lookahead_bit(term: TerminalID, num_terminals: u32) -> usize {
    if term == EOF {
        num_terminals as usize
    } else {
        term as usize
    }
}

fn bit_lookahead(bit: usize, num_terminals: u32) -> TerminalID {
    if bit == num_terminals as usize {
        EOF
    } else {
        bit as TerminalID
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1ItemCore {
    rule: u32,
    dot: u32,
    stack_depth: u32,
    transferred: bool,
}

impl LR1ItemCore {
    fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {
        Self {
            rule,
            dot,
            stack_depth,
            transferred: false,
        }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

type LR1ItemSet = BTreeMap<LR1ItemCore, BitSet>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1Item {
    rule: u32,
    dot: u32,
    lookahead: TerminalID,
    stack_depth: u32,
    /// When true, this item was "transferred" from a parent state to provide
    /// goto information.  Transferred items do NOT participate in closure,
    /// shift actions, or reduce actions — only goto transitions.
    transferred: bool,
}

impl LR1Item {
    fn new(rule: u32, dot: u32, lookahead: TerminalID, stack_depth: u32) -> Self {
        Self { rule, dot, lookahead, stack_depth, transferred: false }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

/// Compute FIRST set for a sequence of symbols followed by a lookahead terminal.
fn first_of_sequence_bits(
    symbols: &[Symbol],
    lookaheads: &BitSet,
    first: &[BitSet],
    nullable: &BTreeSet<NonterminalID>,
    num_terminals: u32,
) -> BitSet {
    let mut result = BitSet::new(num_terminals as usize + 1);
    let mut all_nullable = true;
    for sym in symbols {
        match sym {
            Symbol::Terminal(t) => {
                result.set(*t as usize);
                all_nullable = false;
                break;
            }
            Symbol::Nonterminal(nt) => {
                result.union_with(&first[*nt as usize]);
                if !nullable.contains(nt) {
                    all_nullable = false;
                    break;
                }
            }
        }
    }
    if all_nullable {
        result.union_with(lookaheads);
    }
    result
}

fn first_bitsets(grammar: &AnalyzedGrammar) -> Vec<BitSet> {
    grammar.first.clone()
}

struct RuleSuffixFirst {
    first: Vec<BitSet>,
    nullable: Vec<bool>,
}

/// FIRST and nullability for every suffix of every RHS. LR(1) closure visits
/// the same `(rule, dot + 1)` suffixes many times across successor kernels;
/// compute those grammar-only facts once rather than rescanning each suffix.
fn rule_suffix_first_sets(grammar: &AnalyzedGrammar) -> Vec<RuleSuffixFirst> {
    grammar
        .rules
        .iter()
        .map(|rule| {
            let len = rule.rhs.len();
            let mut first = (0..=len)
                .map(|_| BitSet::new(grammar.num_terminals as usize + 1))
                .collect::<Vec<_>>();
            let mut nullable = vec![false; len + 1];
            nullable[len] = true;
            for index in (0..len).rev() {
                match rule.rhs[index] {
                    Symbol::Terminal(terminal) => {
                        first[index].set(terminal as usize);
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        first[index] = grammar.first[nonterminal as usize].clone();
                        let nonterminal_nullable = grammar.nullable.contains(&nonterminal);
                        if nonterminal_nullable {
                            let following_first = first[index + 1].clone();
                            first[index].union_with(&following_first);
                        }
                        nullable[index] = nonterminal_nullable && nullable[index + 1];
                    }
                }
            }
            RuleSuffixFirst { first, nullable }
        })
        .collect()
}

fn union_lookaheads(item_set: &mut LR1ItemSet, core: LR1ItemCore, lookaheads: &BitSet) -> BitSet {
    let entry = item_set
        .entry(core)
        .or_insert_with(|| BitSet::new(lookaheads.len()));
    entry.union_with_delta(lookaheads)
}

fn lr1_closure(
    mut result: LR1ItemSet,
    grammar: &AnalyzedGrammar,
    suffix_first: &[RuleSuffixFirst],
) -> LR1ItemSet {
    let rules = &grammar.rules;
    // Every caller constructs its kernel solely to close it. Consume that
    // kernel directly instead of cloning its ordered map and lookahead bitsets
    // before the fixed point starts.
    let mut queue: VecDeque<(LR1ItemCore, BitSet)> = result
        .iter()
        .map(|(core, lookaheads)| (*core, lookaheads.clone()))
        .collect();

    while let Some((item, lookahead_delta)) = queue.pop_front() {
        // Transferred items do not participate in closure.
        if item.transferred {
            continue;
        }
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            let suffix = &suffix_first[item.rule as usize];
            let suffix_index = item.dot as usize + 1;
            let base_lookaheads = &suffix.first[suffix_index];
            if suffix.nullable[suffix_index] {
                let mut propagated_lookaheads = base_lookaheads.clone();
                propagated_lookaheads.union_with(&lookahead_delta);
                for &i in &grammar.rules_by_lhs[*nt as usize] {
                    let sd = grammar.rules[i as usize].rhs.len() as u32;
                    let new_item = LR1ItemCore::new(i, 0, sd);
                    let delta = union_lookaheads(
                        &mut result,
                        new_item,
                        &propagated_lookaheads,
                    );
                    if !delta.is_empty() {
                        queue.push_back((new_item, delta));
                    }
                }
            } else {
                for &i in &grammar.rules_by_lhs[*nt as usize] {
                    let sd = grammar.rules[i as usize].rhs.len() as u32;
                    let new_item = LR1ItemCore::new(i, 0, sd);
                    let delta = union_lookaheads(&mut result, new_item, base_lookaheads);
                    if !delta.is_empty() {
                        queue.push_back((new_item, delta));
                    }
                }
            }
        }
    }
    result
}

/// Compute transferred items for the local-forward replace optimisation.
///
/// For each dot-1 item `[A → X . rest, la]` in `kernel`, find "foo items" in
/// `source_items` — items whose symbol-after-dot is `Nonterminal(A)`.  These
/// are the items that generate gotos for `A` in the source state.  Transferring
/// them into the target kernel provides the same gotos at the target so the
/// transition can be marked replace.
///
/// Returns `None` if:
/// - any dot-1 item belongs to `rule == 0` (augmented start), or
/// - any dot-1 item is NOT completed (i.e., not a single-symbol production), or
/// - any dot-1 item's LHS nonterminal has NO foo items in the source.
///
/// Recursively follows single-symbol production chains: when a foo item is
/// itself a single-symbol production at dot=0, its LHS nonterminal also needs
/// foo items in the source.
///
/// Returns `Some(transferred)` with the set of transferred items otherwise.

/// Eagerly advance transferred items past completed nonterminals in the
fn compute_transfer_items(
    kernel: &BTreeSet<LR1Item>,
    source_items: &BTreeSet<LR1Item>,
    rules: &[Rule],
) -> Option<Vec<LR1Item>> {
    let mut transferred = Vec::new();

    // Collect the LHS nonterminals of all dot-1 items.
    // Only completed dot-1 items (single-symbol productions) are eligible.
    let mut needed_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    for item in kernel.iter().filter(|it| it.dot == 1) {
        if item.rule == 0 {
            return None;
        }
        let rule = &rules[item.rule as usize];
        if (item.dot as usize) != rule.rhs.len() {
            return None;
        }
        needed_nts.insert(rule.lhs);
    }

    if needed_nts.is_empty() {
        return None;
    }

    // Iteratively find foo items, following the nonterminal chain.
    // The chain extends through ALL foo items' LHS nonterminals so that
    // gotos for every nonterminal in the reduce chain are available.
    let mut all_needed = needed_nts.clone();
    let mut found_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    loop {
        let mut new_needed: BTreeSet<NonterminalID> = BTreeSet::new();
        for item in source_items {
            if item.transferred {
                continue;
            }
            if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                if all_needed.contains(nt) && !found_nts.contains(nt) {
                    transferred.push(LR1Item {
                        transferred: true,
                        ..*item
                    });
                    found_nts.insert(*nt);
                    // Add this foo item's LHS to the chain so that gotos
                    // for it are also generated in the target state.
                    if item.dot == 0 {
                        let foo_rule = &rules[item.rule as usize];
                        let chain_nt = foo_rule.lhs;
                        if !all_needed.contains(&chain_nt) {
                            new_needed.insert(chain_nt);
                        }
                    }
                }
            }
        }
        if new_needed.is_empty() {
            break;
        }
        all_needed.extend(&new_needed);
    }

    // ALL initially needed nonterminals must have at least one foo item.
    // Chain-extended nonterminals may not have foo items (e.g. the
    // augmented start nonterminal) which is fine.
    if !needed_nts.is_subset(&found_nts) {
        return None;
    }

    if transferred.is_empty() {
        return None;
    }

    Some(transferred)
}

struct LR1Successor {
    symbol: Symbol,
    is_replace: bool,
    is_forwarded: bool,
    target_items: Option<Arc<LR1ItemSet>>,
    target_fingerprint: u64,
    kernel_fingerprint: u64,
    preclosed_target: Option<u32>,
}

/// Structural fingerprint of an LR(1) item set. Equal item sets always hash to
/// the same value; the interner resolves the (rare) collisions with a full
/// equality check, so this only needs to be a good hash, not perfect.
fn lr1_item_set_fingerprint(set: &LR1ItemSet) -> u64 {
    let mut hasher = FxHasher::default();
    set.len().hash(&mut hasher);
    for (core, lookaheads) in set {
        core.hash(&mut hasher);
        lookaheads.hash(&mut hasher);
    }
    hasher.finish()
}

/// Successor kernels contain only advanced (`dot > 0`) items. LR(1) closure
/// only adds rule-entry (`dot == 0`) items, so the original kernel of a closed
/// state is recoverable exactly without retaining a second copy of the map.
/// This lets later transitions recognize an already-known raw kernel before
/// paying to close it again.
fn lr1_kernel_matches_closed_state(kernel: &LR1ItemSet, closed: &LR1ItemSet) -> bool {
    let mut closed_kernel = closed.iter().filter(|(core, _)| core.dot > 0);
    for (core, lookaheads) in kernel {
        let Some((closed_core, closed_lookaheads)) = closed_kernel.next() else {
            return false;
        };
        if core != closed_core || lookaheads != closed_lookaheads {
            return false;
        }
    }
    closed_kernel.next().is_none()
}

fn expand_lr1_state(
    source_items: &LR1ItemSet,
    grammar: &AnalyzedGrammar,
    suffix_first: &[RuleSuffixFirst],
    item_sets: &[Arc<LR1ItemSet>],
    kernel_fingerprint_to_ids: &FxHashMap<u64, Vec<u32>>,
    preclosure_reuse_enabled: bool,
) -> Vec<LR1Successor> {
    let rules = &grammar.rules;
    // Accumulate by hash, then restore the existing canonical symbol order
    // before closure/interner traversal. This avoids tree maintenance in the
    // hot expansion path without changing successor or state numbering.
    let mut kernels: FxHashMap<Symbol, LR1ItemSet> = FxHashMap::default();
    for (item, lookaheads) in source_items {
        if item.transferred {
            if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                let advanced = LR1ItemCore {
                    rule: item.rule,
                    dot: item.dot + 1,
                    stack_depth: item.stack_depth,
                    transferred: false,
                };
                union_lookaheads(
                    kernels.entry(Symbol::Nonterminal(*nt)).or_default(),
                    advanced,
                    lookaheads,
                );
            }
            continue;
        }
        if let Some(symbol) = item.next_symbol(rules) {
            let advanced = LR1ItemCore::new(item.rule, item.dot + 1, item.stack_depth);
            union_lookaheads(kernels.entry(symbol.clone()).or_default(), advanced, lookaheads);
        }
    }

    let mut kernels = kernels.into_iter().collect::<Vec<_>>();
    kernels.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    kernels
        .into_iter()
        .filter_map(|(symbol, kernel)| {
            let has_dot_1 = kernel.keys().any(|item| item.dot == 1);
            let is_replace = match &symbol {
                Symbol::Terminal(_) => !has_dot_1 && replace_shifts_enabled(),
                Symbol::Nonterminal(_) => !has_dot_1 && replace_gotos_enabled(),
            };
            let adjusted_kernel: LR1ItemSet = if is_replace {
                kernel
                    .iter()
                    .map(|(item, lookaheads)| {
                        let adjusted = if item.transferred {
                            *item
                        } else {
                            LR1ItemCore {
                                rule: item.rule,
                                dot: item.dot,
                                stack_depth: item.stack_depth.saturating_sub(1),
                                ..*item
                            }
                        };
                        (adjusted, lookaheads.clone())
                    })
                    .collect()
            } else {
                kernel
            };
            let kernel_fingerprint = preclosure_reuse_enabled
                .then(|| lr1_item_set_fingerprint(&adjusted_kernel));
            if let Some(kernel_fingerprint) = kernel_fingerprint
                && let Some(candidates) = kernel_fingerprint_to_ids.get(&kernel_fingerprint)
                && let Some(target_id) = candidates.iter().copied().find(|&candidate| {
                    lr1_kernel_matches_closed_state(
                        &adjusted_kernel,
                        &item_sets[candidate as usize],
                    )
                })
            {
                return Some(LR1Successor {
                    symbol,
                    is_replace,
                    is_forwarded: false,
                    target_items: None,
                    target_fingerprint: 0,
                    kernel_fingerprint,
                    preclosed_target: Some(target_id),
                });
            }
            let target_items = Arc::new(lr1_closure(adjusted_kernel, grammar, suffix_first));
            if target_items.is_empty() {
                None
            } else {
                let fingerprint = lr1_item_set_fingerprint(&target_items);
                Some(LR1Successor {
                    symbol,
                    is_replace,
                    is_forwarded: false,
                    target_items: Some(target_items),
                    target_fingerprint: fingerprint,
                    kernel_fingerprint: kernel_fingerprint.unwrap_or(0),
                    preclosed_target: None,
                })
            }
        })
        .collect()
}

fn build_lr1_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<LR1ItemSet>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let preclosure_reuse_enabled =
        std::env::var_os("GLRMASK_DISABLE_LR1_PRECLOSURE_REUSE").is_none();
    build_lr1_item_sets_with_preclosure_reuse(grammar, preclosure_reuse_enabled)
}

fn build_lr1_item_sets_with_preclosure_reuse(
    grammar: &AnalyzedGrammar,
    preclosure_reuse_enabled: bool,
) -> (Vec<LR1ItemSet>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let rules = &grammar.rules;
    let lookahead_len = grammar.num_terminals as usize + 1;
    let suffix_first = rule_suffix_first_sets(grammar);

    let initial = Arc::new({
        let mut s = LR1ItemSet::new();
        let sd = rules[0].rhs.len() as u32;
        let mut lookaheads = BitSet::new(lookahead_len);
        lookaheads.set(lookahead_bit(EOF, grammar.num_terminals));
        s.insert(LR1ItemCore::new(0, 0, sd), lookaheads);
        lr1_closure(s, grammar, &suffix_first)
    });

    let mut item_sets = vec![initial.clone()];
    let mut transitions: Vec<BTreeMap<Symbol, (u32, bool, bool)>> = vec![BTreeMap::new()];
    // Intern item sets by structural fingerprint. The fingerprint is computed
    // in the parallel expansion phase, so the serial interning loop below only
    // performs cheap `u64` hashing plus a full equality check on the rare
    // fingerprint collision. Canonical state numbering is unchanged because the
    // serial loop still visits successors in the old source/symbol order.
    let mut fingerprint_to_ids: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
    fingerprint_to_ids
        .entry(lr1_item_set_fingerprint(&initial))
        .or_default()
        .push(0);
    let mut kernel_fingerprint_to_ids: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
    drop(initial);

    let mut frontier = vec![0u32];
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let mut expand_ms = 0.0f64;
    let mut intern_ms = 0.0f64;
    let mut successor_count = 0usize;
    let mut existing_successor_count = 0usize;
    let mut new_successor_count = 0usize;
    let mut preclosure_reuse_count = 0usize;
    while !frontier.is_empty() {
        // State expansion is independent within a BFS frontier. Interning
        // remains serial below in the old source/symbol order, preserving
        // canonical state numbering and artifact layout.
        let expand_started_at = profile_enabled.then(std::time::Instant::now);
        let expanded = frontier
            .par_iter()
            .map(|&state_id| {
                let successors = expand_lr1_state(
                    &item_sets[state_id as usize],
                    grammar,
                    &suffix_first,
                    &item_sets,
                    &kernel_fingerprint_to_ids,
                    preclosure_reuse_enabled,
                );
                (state_id, successors)
            })
            .collect::<Vec<_>>();
        if let Some(started_at) = expand_started_at {
            expand_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let intern_started_at = profile_enabled.then(std::time::Instant::now);
        let mut next_frontier = Vec::new();
        for (state_id, successors) in expanded {
            for successor in successors {
                successor_count += 1;
                let target_id = if let Some(target_id) = successor.preclosed_target {
                    preclosure_reuse_count += 1;
                    existing_successor_count += 1;
                    target_id
                } else {
                    let target_items = successor
                        .target_items
                        .expect("non-preclosed LR1 successor must retain closed items");
                    let candidates = fingerprint_to_ids
                        .entry(successor.target_fingerprint)
                        .or_default();
                    let existing = candidates
                        .iter()
                        .copied()
                        .find(|&cand| *item_sets[cand as usize] == *target_items);
                    if let Some(existing_id) = existing {
                        existing_successor_count += 1;
                        existing_id
                    } else {
                        new_successor_count += 1;
                        let new_id = item_sets.len() as u32;
                        candidates.push(new_id);
                        item_sets.push(target_items);
                        transitions.push(BTreeMap::new());
                        next_frontier.push(new_id);
                        new_id
                    }
                };
                if preclosure_reuse_enabled {
                    let kernel_candidates = kernel_fingerprint_to_ids
                        .entry(successor.kernel_fingerprint)
                        .or_default();
                    if !kernel_candidates.contains(&target_id) {
                        kernel_candidates.push(target_id);
                    }
                }
                transitions[state_id as usize].insert(
                    successor.symbol,
                    (target_id, successor.is_replace, successor.is_forwarded),
                );
            }
        }
        if let Some(started_at) = intern_started_at {
            intern_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        frontier = next_frontier;
    }
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][lr1_item_sets] states={} successors={} preclosure_reuses={} existing_successors={} new_successors={} expand_ms={:.3} intern_ms={:.3}",
            item_sets.len(),
            successor_count,
            preclosure_reuse_count,
            existing_successor_count,
            new_successor_count,
            expand_ms,
            intern_ms,
        );
    }

    drop(fingerprint_to_ids);
    let item_sets = item_sets
        .into_iter()
        .map(|items| Arc::try_unwrap(items).expect("canonical item set still shared"))
        .collect();
    (item_sets, transitions)
}

fn current_unique_reduce_len(
    pending: &[BTreeMap<TerminalID, PendingAction>],
    state: u32,
    lookahead: TerminalID,
    nonterminal: NonterminalID,
) -> Option<u32> {
    let pending_action = pending.get(state as usize)?.get(&lookahead)?;
    let mut unique_len = None;
    for &(reduce_nt, reduce_len) in &pending_action.reduces {
        if reduce_nt != nonterminal {
            continue;
        }
        match unique_len {
            None => unique_len = Some(reduce_len),
            Some(existing) if existing == reduce_len => {}
            Some(_) => return None,
        }
    }
    unique_len
}

/// Check if any nonterminal can transitively reach itself through the
/// grammar's production rules. Returns true if any recursion exists.
fn grammar_has_recursion(rules: &[Rule]) -> bool {
    let max_nt = rules.iter().map(|r| r.lhs).max().unwrap_or(0) as usize + 1;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); max_nt];
    for rule in rules {
        let lhs = rule.lhs as usize;
        for sym in &rule.rhs {
            if let Symbol::Nonterminal(nt) = sym {
                adj[lhs].push(*nt as usize);
            }
        }
    }

    // 0 = unvisited, 1 = visiting, 2 = done.
    let mut color = vec![0u8; max_nt];
    fn dfs(node: usize, adj: &[Vec<usize>], color: &mut [u8]) -> bool {
        color[node] = 1;
        for &next in &adj[node] {
            match color[next] {
                1 => return true,
                0 => {
                    if dfs(next, adj, color) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        color[node] = 2;
        false
    }

    for nt in 0..max_nt {
        if color[nt] == 0 && dfs(nt, &adj, &mut color) {
            return true;
        }
    }
    false
}

fn apply_local_forward_replace(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
    rules: &[Rule],
) {
    // Build incoming-transition count per state so we can check whether
    // a reduce-state is uniquely reachable.  If multiple transitions
    // lead into the reduce state, rewriting its reduce length would
    // corrupt other paths (e.g. recursive grammars).
    let mut in_count = vec![0u32; item_sets.len()];
    for t in transitions {
        for &(target, _, _) in t.values() {
            in_count[target as usize] += 1;
        }
    }

    loop {
        let mut changed = false;

        for source in 0..item_sets.len() {
            for (symbol, &(target, _, _)) in &transitions[source] {
                let currently_non_replace = match symbol {
                    Symbol::Terminal(terminal) => pending[source]
                        .get(terminal)
                        .and_then(|pending_action| pending_action.shift)
                        .is_some_and(|(_, is_replace)| !is_replace),
                    Symbol::Nonterminal(nonterminal) => goto[source]
                        .get(nonterminal)
                        .is_some_and(|&(_, is_replace)| !is_replace),
                };
                if !currently_non_replace {
                    continue;
                }

                let dot1_items: Vec<_> = item_sets[target as usize]
                    .iter()
                    .copied()
                    .filter(|item| item.dot == 1)
                    .collect();
                if dot1_items.is_empty() {
                    continue;
                }

                let mut forwarded: BTreeMap<(u32, NonterminalID), u32> = BTreeMap::new();
                let mut rewrites = Vec::new();
                let mut valid = true;

                for item in dot1_items {
                    if item.rule == 0 {
                        valid = false;
                        break;
                    }

                    let rule = &rules[item.rule as usize];
                    let reduce_nt = rule.lhs;
                    let Some(&(forward_target, _)) = goto[source].get(&reduce_nt) else {
                        valid = false;
                        break;
                    };

                    let mut reduce_state = target;
                    let mut chain_unique = in_count[target as usize] <= 1;
                    for next_symbol in &rule.rhs[item.dot as usize..] {
                        let Some(&(next_state, _, _)) = transitions[reduce_state as usize].get(next_symbol) else {
                            valid = false;
                            break;
                        };
                        reduce_state = next_state;
                        // If a state in the chain is reachable from
                        // multiple predecessors, we can't safely rewrite
                        // its reduce because other paths share it.
                        if in_count[reduce_state as usize] > 1 {
                            chain_unique = false;
                        }
                    }
                    if !valid {
                        break;
                    }
                    if !chain_unique {
                        valid = false;
                        break;
                    }

                    let Some(current_len) = current_unique_reduce_len(
                        pending,
                        reduce_state,
                        item.lookahead,
                        reduce_nt,
                    ) else {
                        valid = false;
                        break;
                    };
                    if current_len != 1 {
                        valid = false;
                        break;
                    }

                    match forwarded.get(&(reduce_state, reduce_nt)) {
                        Some(&existing_target) if existing_target != forward_target => {
                            valid = false;
                            break;
                        }
                        Some(_) => {}
                        None => {
                            if let Some(&(existing_target, existing_replace)) =
                                goto[reduce_state as usize].get(&reduce_nt)
                            {
                                if existing_target != forward_target || !existing_replace {
                                    valid = false;
                                    break;
                                }
                            }
                            forwarded.insert((reduce_state, reduce_nt), forward_target);
                        }
                    }

                    rewrites.push((reduce_state, item.lookahead, reduce_nt));
                }

                if !valid || rewrites.is_empty() {
                    continue;
                }

                changed = true;

                match symbol {
                    Symbol::Terminal(terminal) => {
                        if let Some(pending_action) = pending[source].get_mut(terminal) {
                            if let Some((_, is_replace)) = pending_action.shift.as_mut() {
                                *is_replace = true;
                            }
                        }
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some((_, is_replace)) = goto[source].get_mut(nonterminal) {
                            *is_replace = true;
                        }
                    }
                }

                for &(reduce_state, lookahead, reduce_nt) in &rewrites {
                    if let Some(pending_action) = pending[reduce_state as usize].get_mut(&lookahead) {
                        for (existing_nt, reduce_len) in pending_action.reduces.iter_mut() {
                            if *existing_nt == reduce_nt && *reduce_len == 1 {
                                *reduce_len = 0;
                            }
                        }
                    }
                }

                for ((reduce_state, reduce_nt), forward_target) in forwarded {
                    goto[reduce_state as usize].insert(reduce_nt, (forward_target, true));
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Inline zero-pop reduces: when a state has Reduce(nt, 0) on some terminal,
/// follow the goto to the target state and copy the target's action for that
/// terminal into the current state. This eliminates the zero-pop reduce
/// entirely, replacing it with a direct shift or accept.
///
/// Iterates until no more inlining is possible (handles chains of zero-pop
/// reduces).
fn inline_zero_pop_reduces(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
) {
    loop {
        let mut changed = false;

        for state in 0..pending.len() {
            // Collect (terminal, nt, target_state) triples for zero-pop reduces.
            let mut to_inline: Vec<(TerminalID, NonterminalID, u32)> = Vec::new();
            if let Some(by_terminal) = pending.get(state) {
                for (&terminal, pa) in by_terminal {
                    for &(reduce_nt, reduce_len) in &pa.reduces {
                        if reduce_len == 0 {
                            if let Some(&(target, _)) = goto[state].get(&reduce_nt) {
                                to_inline.push((terminal, reduce_nt, target));
                            }
                        }
                    }
                }
            }

            for (terminal, reduce_nt, target) in to_inline {
                if target as usize == state {
                    continue; // avoid self-reference
                }

                // Read action at target state for the same terminal.
                let target_pa = pending[target as usize].get(&terminal).cloned();

                if let Some(pa) = pending[state].get_mut(&terminal) {
                    // Remove the zero-pop reduce.
                    pa.reduces.retain(|&(nt, len)| !(nt == reduce_nt && len == 0));

                    // Inline target's actions.
                    if let Some(tpa) = target_pa {
                        if let Some((shift_target, _)) = tpa.shift {
                            pa.push_shift(shift_target, true);
                        }
                        for (nt, len) in &tpa.reduces {
                            pa.push_reduce(*nt, *len);
                        }
                        if tpa.accept {
                            pa.push_accept();
                        }

                        // Propagate gotos needed for any zero-pop reduces
                        // we just inlined from the target state.
                        for &(nt, len) in &tpa.reduces {
                            if len == 0 {
                                if let Some(&goto_entry) = goto[target as usize].get(&nt) {
                                    goto[state].entry(nt).or_insert(goto_entry);
                                }
                            }
                        }
                    }

                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn build_lr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[LR1ItemSet],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let init_started_at = profile_enabled.then(std::time::Instant::now);
    let (pending, goto, forwarded_shifts) = initialize_pending_and_goto(transitions);
    let init_ms = init_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // Replace flags are now carried in the transitions map from
    // build_lr1_item_sets, so we don't need to recompute them here.

    let mut pending = pending;
    let goto = goto;
    let reductions_started_at = profile_enabled.then(std::time::Instant::now);
    // Each state's pending reduce/accept actions depend only on that state's
    // items, so populate the rows in parallel. Shifts were already filled in by
    // `initialize_pending_and_goto`; we only append reduces/accepts here.
    pending
        .par_iter_mut()
        .zip(item_sets.par_iter())
        .for_each(|(pending_row, items)| {
            for (item, lookaheads) in items {
                // Transferred items do not generate reduces.
                if item.transferred {
                    continue;
                }
                let rule = &grammar.rules[item.rule as usize];
                if item.dot as usize != rule.rhs.len() {
                    continue;
                }

                for bit in lookaheads.iter_ones() {
                    let lookahead = bit_lookahead(bit, grammar.num_terminals);
                    if item.rule == 0 {
                        pending_row.entry(lookahead).or_default().push_accept();
                        continue;
                    }

                    pending_row
                        .entry(lookahead)
                        .or_default()
                        .push_reduce(rule.lhs, item.stack_depth);
                }
            }
        });
    let reductions_ms = reductions_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    // Grouped LR(1) lookahead sets delay scalar fanout until pending-action
    // emission. The old local-forward path is written for scalar LR1 items,
    // so keep replace handling conservative here rather than approximating.
    let finish_started_at = profile_enabled.then(std::time::Instant::now);
    let table = finish_table(
        grammar,
        pending,
        goto,
        forwarded_shifts,
        GlrTableConstruction::LegacyRowBisim,
        AdmissionPolicy::RowPresenceExact,
    );
    if let Some(finish_started_at) = finish_started_at {
        eprintln!(
            "[glrmask/profile][lr1_table] init_ms={:.3} reductions_ms={:.3} finish_ms={:.3} states={}",
            init_ms,
            reductions_ms,
            finish_started_at.elapsed().as_secs_f64() * 1000.0,
            table.num_states,
        );
    }
    table
}

// Legacy row-bisimulation merge over canonical LR(1) item sets.

fn lr1_core_key(items: &LR1ItemSet) -> Vec<Item> {
    // `LR1ItemSet` is already ordered by (rule, dot, stack_depth,
    // transferred). The table-core key deliberately ignores `transferred`, so
    // adjacent entries can only duplicate after that projection. Avoid the
    // second BTreeSet allocation used by the old implementation.
    let mut core = Vec::with_capacity(items.len());
    for item in items.keys() {
        let projected = Item::new(item.rule, item.dot, item.stack_depth);
        if core.last().copied() != Some(projected) {
            core.push(projected);
        }
    }
    core
}

fn build_experimental_core_merged_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[LR1ItemSet],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> Option<GLRTable> {
    let canonical = build_lr1_table(grammar, item_sets, transitions);
    let core_keys = item_sets.iter().map(lr1_core_key).collect::<Vec<_>>();
    let partition = refine_experimental_core_partition(&canonical, &core_keys);
    let mut table = union_experimental_core_rows(canonical, &partition)?;
    table.construction = GlrTableConstruction::ExperimentalCoreMerged;
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    table.rebuild_advance_rows_from_actions();
    Some(table)
}

fn refine_experimental_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {
    let mut class_by_core: BTreeMap<Vec<Item>, u32> = BTreeMap::new();
    let mut partition = Vec::with_capacity(core_keys.len());
    for key in core_keys {
        let next = class_by_core.len() as u32;
        partition.push(*class_by_core.entry(key.clone()).or_insert(next));
    }

    loop {
        let mut sig_to_class: BTreeMap<ExperimentalCoreCompatibilitySig, u32> = BTreeMap::new();
        let mut next_partition = Vec::with_capacity(partition.len());
        for state in 0..table.num_states as usize {
            let sig = ExperimentalCoreCompatibilitySig::new(table, state, partition[state], &partition);
            let next = sig_to_class.len() as u32;
            next_partition.push(*sig_to_class.entry(sig).or_insert(next));
        }
        if next_partition == partition {
            return partition;
        }
        partition = next_partition;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExperimentalCoreCompatibilitySig {
    core_class: u32,
    shifts: Vec<(TerminalID, u32, bool, bool)>,
    gotos: Vec<(NonterminalID, u32, bool)>,
}

impl ExperimentalCoreCompatibilitySig {
    fn new(table: &GLRTable, state: usize, core_class: u32, partition: &[u32]) -> Self {
        let mut shifts = Vec::new();
        for (terminal, action) in &table.action[state] {
            if let Some((target, replace)) = action_shift(action) {
                shifts.push((
                    terminal,
                    partition[target as usize],
                    replace,
                    table.forwarded_shifts.contains(&(state as u32, terminal)),
                ));
            }
        }
        shifts.sort_unstable();

        let mut gotos = table.goto[state]
            .iter()
            .map(|(&nt, &(target, replace))| (nt, partition[target as usize], replace))
            .collect::<Vec<_>>();
        gotos.sort_unstable();

        Self {
            core_class,
            shifts,
            gotos,
        }
    }
}

fn action_shift(action: &Action) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, replace) => Some((*target, *replace)),
        Action::Split {
            shift: Some((target, replace)),
            ..
        } => Some((*target, *replace)),
        _ => None,
    }
}

fn union_experimental_core_rows(table: GLRTable, partition: &[u32]) -> Option<GLRTable> {
    let nstates = table.num_states as usize;
    let ngroups = partition.iter().copied().max().map(|x| x + 1).unwrap_or(0) as usize;

    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(ngroups)
        .collect::<Vec<_>>();
    let mut goto = (0..ngroups).map(|_| FxHashMap::default()).collect::<Vec<_>>();
    let mut forwarded_shifts = FxHashSet::default();

    for state in 0..nstates {
        let group = partition[state] as usize;
        for (terminal, action) in &table.action[state] {
            add_remapped_action_to_pending(
                action,
                &mut pending[group].entry(terminal).or_default(),
                partition,
            )?;
            if action_shift(action).is_some()
                && table.forwarded_shifts.contains(&(state as u32, terminal))
            {
                forwarded_shifts.insert((group as u32, terminal));
            }
        }
        for (&nt, &(target, replace)) in &table.goto[state] {
            let remapped = (partition[target as usize], replace);
            match goto[group].get(&nt).copied() {
                Some(existing) if existing != remapped => return None,
                Some(_) => {}
                None => {
                    goto[group].insert(nt, remapped);
                }
            }
        }
    }

    let action = pending
        .into_iter()
        .map(|by_terminal| {
            by_terminal
                .into_iter()
                .filter_map(|(terminal, pending)| pending.maybe_finish().map(|action| (terminal, action)))
                .collect::<ActionRow>()
        })
        .collect();
    let goto = goto
        .into_iter()
        .map(|row| row.into_iter().collect::<GotoRow>())
        .collect();

    let direct_regular_wide_frontiers = table
        .direct_regular_wide_frontiers
        .iter()
        .map(|descriptor| {
            let mut target_states = descriptor
                .target_states
                .iter()
                .map(|&state| partition[state as usize])
                .collect::<Vec<_>>();
            target_states.sort_unstable();
            target_states.dedup();
            DirectRegularWideFrontierDescriptor {
                source_state: partition[descriptor.source_state as usize],
                terminal: descriptor.terminal,
                target_states,
            }
        })
        .collect();

    Some(GLRTable {
        action,
        goto,
        num_states: ngroups as u32,
        num_terminals: table.num_terminals,
        num_rules: table.num_rules,
        rules: table.rules,
        nonterminal_display_names: table.nonterminal_display_names,
        embedded_start: table.embedded_start,
        construction: table.construction,
        admission_policy: table.admission_policy,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts,
        control_terminals: table.control_terminals,
        skip_terminals: table.skip_terminals,
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers,
    })
}

fn add_remapped_action_to_pending(
    action: &Action,
    pending: &mut PendingAction,
    partition: &[u32],
) -> Option<()> {
    match action {
        Action::Shift(target, replace) => pending.push_shift(partition[*target as usize], *replace),
        Action::Reduce(nt, len) => pending.push_reduce(*nt, *len),
        Action::Accept => pending.push_accept(),
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                pending.push_shift(partition[*target as usize], *replace);
            }
            for &(nt, len) in reduces {
                pending.push_reduce(nt, len);
            }
            if *accept {
                pending.push_accept();
            }
        }
        Action::StackShifts(_)
        | Action::ReplaceShifts(_)
        | Action::GuardedStackShifts(_)
        | Action::Skip => return None,
    }
    Some(())
}

fn build_legacy_row_bisim_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[LR1ItemSet],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let canonical_started_at = profile_enabled.then(std::time::Instant::now);
    let canonical = build_lr1_table(grammar, item_sets, transitions);
    let canonical_ms = canonical_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    if !same_core_quotient_enabled(canonical.num_states) {
        if let Some(canonical_ms) = canonical_ms {
            eprintln!(
                "[glrmask/profile][glr_legacy_build] canonical_table_ms={:.3} core_keys_ms=0.000 same_core_merge_ms=0.000 pre_merge_states={} post_core_states={} same_core_skip_reason=pre_merge_states",
                canonical_ms,
                item_sets.len(),
                canonical.num_states,
            );
        }
        return canonical;
    }

    let core_keys_started_at = profile_enabled.then(std::time::Instant::now);
    let core_keys = item_sets.par_iter().map(lr1_core_key).collect::<Vec<_>>();
    let core_keys_ms = core_keys_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
    let merge_started_at = profile_enabled.then(std::time::Instant::now);
    let table = merge_same_core_lr1_states(canonical, &core_keys);
    if let (Some(canonical_ms), Some(core_keys_ms), Some(merge_started_at)) =
        (canonical_ms, core_keys_ms, merge_started_at)
    {
        eprintln!(
            "[glrmask/profile][glr_legacy_build] canonical_table_ms={:.3} core_keys_ms={:.3} same_core_merge_ms={:.3} pre_merge_states={} post_core_states={} same_core_skip_reason=none",
            canonical_ms,
            core_keys_ms,
            merge_started_at.elapsed().as_secs_f64() * 1000.0,
            item_sets.len(),
            table.num_states,
        );
    }
    table
}

#[cfg(test)]
fn grouped_item_lookahead_counts(grammar: &AnalyzedGrammar) -> Vec<Vec<(u32, u32, u32, usize)>> {
    let (item_sets, _) = build_lr1_item_sets(grammar);
    item_sets
        .into_iter()
        .map(|items| {
            items
                .into_iter()
                .map(|(core, lookaheads)| (core.rule, core.dot, core.stack_depth, lookaheads.count_ones()))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests ;
