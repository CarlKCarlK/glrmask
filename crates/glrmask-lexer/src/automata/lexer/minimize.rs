//! Hopcroft-based DFA minimization for the lexer DFA.
//!
//! DFA minimization for byte-oriented lexer automata.
//! Uses topology-aware pre-refinement before the final Hopcroft pass.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

use rustc_hash::FxHashMap;

use crate::ds::bitset::BitSet;
use crate::ds::char_transitions::CharTransitions;

use super::dfa::DFA;

/// Exact reverse-topological partition for finite, one-terminal components.
///
/// Every state is a root for this operation, including states disconnected from
/// state zero: they can be reached through the symbolic-to-finite projection.
/// A signature includes the finalizer and every byte-labelled successor class.
/// Hash-map equality checks the complete signature; a hash collision is never
/// an equivalence certificate. Cycles decline before any source mutation.
struct AcyclicSingleGroupPartition {
    // Canonical IDs are ordered by each class's smallest original state. This
    // is the same order as sorted Hopcroft blocks, with old state zero first.
    state_mapping: Vec<u32>,
    representatives: Vec<u32>,
    can_reach_accepting: Vec<bool>,
}

#[cfg(test)]
fn acyclic_single_group_partition(dfa: &DFA) -> Option<(Vec<Vec<u32>>, Vec<bool>)> {
    acyclic_single_group_partition_impl::<{ u64::MAX }>(dfa)
}

// Keep the block-shaped test API so the independent original partition and
// Hopcroft rebuild remain useful oracles. Production never materializes these
// per-class member vectors.
#[cfg(test)]
fn acyclic_single_group_partition_impl<const HASH_MASK: u64>(dfa: &DFA) -> Option<(Vec<Vec<u32>>, Vec<bool>)> {
    let flat = acyclic_single_group_flat_partition_impl::<HASH_MASK>(dfa)?;
    let mut blocks = vec![Vec::new(); flat.representatives.len()];
    for (state, &class) in flat.state_mapping.iter().enumerate() {
        blocks[class as usize].push(u32::try_from(state).ok()?);
    }
    Some((blocks, flat.can_reach_accepting))
}

// A zero hash mask in tests forces every distinct signature into one bucket;
// production equality must still be exact even under total hash collision.
fn acyclic_single_group_flat_partition_impl<const HASH_MASK: u64>(dfa: &DFA) -> Option<AcyclicSingleGroupPartition> {
    if dfa.num_groups() != 1 || dfa.has_epsilon_transitions() || dfa.has_self_loops() {
        return None;
    }
    let n = dfa.num_states();
    let mut visited = vec![0u8; n];
    let mut order = Vec::with_capacity(n);
    // Iterative DFS avoids a call-stack dependency on the repeat bound.
    let mut stack = Vec::new();
    for root in 0..n {
        if visited[root] != 0 { continue; }
        visited[root] = 1;
        stack.push((root, dfa.states()[root].transitions.iter()));
        while let Some((state, edges)) = stack.last_mut() {
            if let Some((_, &target)) = edges.next() {
                let target = target as usize;
                match *visited.get(target)? {
                    0 => {
                        visited[target] = 1;
                        stack.push((target, dfa.states()[target].transitions.iter()));
                    }
                    1 => return None,
                    _ => {}
                }
            } else {
                visited[*state] = 2;
                order.push(*state);
                stack.pop();
            }
        }
    }

    let mut classes = vec![u32::MAX; n];
    let mut bucket_heads = FxHashMap::<u64, u32>::default();
    bucket_heads.reserve(n);
    let mut representatives = Vec::<u32>::with_capacity(n);
    let mut collision_next = Vec::<u32>::with_capacity(n);
    let mut can_reach_accepting = vec![false; n];
    for state in order {
        let accepting = !dfa.finalizers(state as u32).is_empty();
        let transitions = &dfa.states()[state].transitions;
        let mut signature = rustc_hash::FxHasher::default();
        accepting.hash(&mut signature);
        transitions.len().hash(&mut signature);
        let mut reaches_accepting = accepting;
        for (byte, &target) in transitions.iter() {
            debug_assert_ne!(classes[target as usize], u32::MAX);
            byte.hash(&mut signature);
            classes[target as usize].hash(&mut signature);
            reaches_accepting |= can_reach_accepting[target as usize];
        }
        can_reach_accepting[state] = reaches_accepting;
        let hash = signature.finish() & HASH_MASK;
        let bucket = bucket_heads.entry(hash).or_insert(u32::MAX);
        let mut class = *bucket;
        while class != u32::MAX {
            let representative = representatives[class as usize];
            let other = &dfa.states()[representative as usize];
            // The immutable representative graph IS the exact signature. The
            // hash is never a certificate, even when every hash collides.
            if accepting == !other.finalizers.is_empty()
                && transitions.len() == other.transitions.len()
                && transitions.iter().zip(other.transitions.iter()).all(
                    |((byte, &target), (other_byte, &other_target))| {
                        byte == other_byte && classes[target as usize] == classes[other_target as usize]
                    },
                )
            {
                break;
            }
            class = collision_next[class as usize];
        }
        if class == u32::MAX {
            class = u32::try_from(representatives.len()).ok()?;
            representatives.push(u32::try_from(state).ok()?);
            collision_next.push(*bucket);
            *bucket = class;
        }
        classes[state] = class;
    }
    let mut canonical_classes = vec![u32::MAX; representatives.len()];
    representatives.clear();
    // A class's first occurrence is its smallest source state, so this one
    // ordered scan exactly replaces allocating and sorting member blocks.
    for (state, class) in classes.iter_mut().enumerate() {
        let canonical = &mut canonical_classes[*class as usize];
        if *canonical == u32::MAX {
            *canonical = u32::try_from(representatives.len()).ok()?;
            representatives.push(u32::try_from(state).ok()?);
        }
        *class = *canonical;
    }
    Some(AcyclicSingleGroupPartition {
        state_mapping: classes,
        representatives,
        can_reach_accepting,
    })
}

#[cfg(test)]
fn acyclic_single_group_partition_reference(dfa: &DFA) -> Option<(Vec<Vec<u32>>, Vec<bool>)> {
    if dfa.num_groups() != 1 || dfa.has_epsilon_transitions() || dfa.has_self_loops() {
        return None;
    }
    let n = dfa.num_states();
    let mut visited = vec![0u8; n];
    let mut order = Vec::with_capacity(n);
    // Iterative DFS avoids a call-stack dependency on the repeat bound.
    let mut stack = Vec::new();
    for root in 0..n {
        if visited[root] != 0 { continue; }
        visited[root] = 1;
        stack.push((root, dfa.states()[root].transitions.iter()));
        while let Some((state, edges)) = stack.last_mut() {
            if let Some((_, &target)) = edges.next() {
                let target = target as usize;
                match *visited.get(target)? {
                    0 => {
                        visited[target] = 1;
                        stack.push((target, dfa.states()[target].transitions.iter()));
                    }
                    1 => return None,
                    _ => {}
                }
            } else {
                visited[*state] = 2;
                order.push(*state);
                stack.pop();
            }
        }
    }

    let mut classes = vec![u32::MAX; n];
    let mut signatures = FxHashMap::<(bool, Vec<(u8, u32)>), u32>::default();
    let mut can_reach_accepting = vec![false; n];
    for state in order {
        let accepting = !dfa.finalizers(state as u32).is_empty();
        let edges = dfa.states()[state].transitions.iter()
            .map(|(byte, &target)| {
                debug_assert_ne!(classes[target as usize], u32::MAX);
                (byte, classes[target as usize])
            }).collect::<Vec<_>>();
        // Inclusive reachability is passed to the existing rebuilder, which
        // computes STRICT futures via at least one outgoing byte transition.
        can_reach_accepting[state] = accepting || dfa.states()[state].transitions.iter()
            .any(|(_, &target)| can_reach_accepting[target as usize]);
        let next_class = u32::try_from(signatures.len()).ok()?;
        classes[state] = *signatures.entry((accepting, edges)).or_insert(next_class);
    }
    let mut blocks = vec![Vec::new(); signatures.len()];
    for (state, &class) in classes.iter().enumerate() {
        blocks[class as usize].push(u32::try_from(state).ok()?);
    }
    // Match the original preserve-unreachable Hopcroft numbering exactly.
    // Members were inserted in ascending source-state order already.
    blocks.sort_unstable_by_key(|block| block[0]);
    Some((blocks, can_reach_accepting))
}

enum TopologyPrerefine {
    AlreadyMinimal(Vec<Vec<u32>>),
    Refined {
        partition: Vec<u32>,
        blocks: Vec<Vec<u32>>,
    },
    Skip,
}

fn partition_by_finalizers(dfa: &DFA) -> (Vec<u32>, Vec<Vec<u32>>) {
    let num_states = dfa.states().len();
    let mut partition = vec![0u32; num_states];
    let mut blocks: Vec<Vec<u32>> = Vec::new();
    let mut finalizer_to_block: FxHashMap<BitSet, u32> = FxHashMap::default();

    for (state_idx, state) in dfa.states().iter().enumerate() {
        let key = state.finalizers.clone();
        let block_idx = *finalizer_to_block.entry(key).or_insert_with(|| {
            let idx = blocks.len() as u32;
            blocks.push(Vec::new());
            idx
        });
        partition[state_idx] = block_idx;
        blocks[block_idx as usize].push(state_idx as u32);
    }

    (partition, blocks)
}

/// Refine finalizer classes by frozen future-terminal observations.
fn clear_possible_futures_for_minimization(dfa: &mut DFA) {
    let empty = BitSet::new(dfa.num_groups());
    dfa.mask_possible_futures(&empty);
}

fn dedup_adjacency(dfa: &DFA) -> Vec<Vec<usize>> {
    dfa.states()
        .iter()
        .map(|state| {
            let mut targets: Vec<usize> =
                state.transitions.iter().map(|(_, &target)| target as usize).collect();
            targets.sort_unstable();
            targets.dedup();
            targets
        })
        .collect()
}

fn compute_post_order(adj: &[Vec<usize>]) -> Vec<usize> {
    let num_states = adj.len();
    let mut post_order = Vec::with_capacity(num_states);
    let mut visited = vec![0u8; num_states];
    let mut dfs_stack: Vec<(usize, usize)> = Vec::new();

    for root in 0..num_states {
        if visited[root] != 0 {
            continue;
        }
        dfs_stack.push((root, 0));
        visited[root] = 1;

        while let Some((state, edge_index)) = dfs_stack.last_mut() {
            let state = *state;
            if *edge_index < adj[state].len() {
                let target = adj[state][*edge_index];
                *edge_index += 1;
                if visited[target] == 0 {
                    visited[target] = 1;
                    dfs_stack.push((target, 0));
                }
            } else {
                visited[state] = 2;
                post_order.push(state);
                dfs_stack.pop();
            }
        }
    }

    post_order
}

fn reverse_adjacency(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut reverse_adj = vec![Vec::new(); adj.len()];
    for (state, targets) in adj.iter().enumerate() {
        for &target in targets {
            reverse_adj[target].push(state);
        }
    }
    reverse_adj
}

fn compute_kosaraju_scc_ids(adj: &[Vec<usize>], post_order: &[usize]) -> Vec<u32> {
    let reverse_adj = reverse_adjacency(adj);
    let mut scc_id = vec![u32::MAX; adj.len()];
    let mut current_scc = 0u32;

    for &state in post_order.iter().rev() {
        if scc_id[state] != u32::MAX {
            continue;
        }

        let mut scc_stack = vec![state];
        scc_id[state] = current_scc;
        while let Some(node) = scc_stack.pop() {
            for &pred in &reverse_adj[node] {
                if scc_id[pred] == u32::MAX {
                    scc_id[pred] = current_scc;
                    scc_stack.push(pred);
                }
            }
        }

        current_scc += 1;
    }

    scc_id
}

fn build_blocks_from_labels(labels: &[u32], num_labels: u32) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut partition = vec![0u32; labels.len()];
    let mut blocks = vec![Vec::new(); num_labels as usize];

    for (state, &label) in labels.iter().enumerate() {
        partition[state] = label;
        blocks[label as usize].push(state as u32);
    }

    (partition, blocks)
}

/// Fast iterative partition refinement.
///
/// Starting from an initial partition (e.g. by finalizers), iteratively refine
/// by computing a signature for each state = (block_id, [(byte, target_block_id)])
/// and re-partitioning by signature. Converges in O(depth) iterations where
/// depth is the DFA's longest shortest path. Each iteration is O(n * avg_degree).
///
/// This is semantically equivalent to Hopcroft but much faster when the DFA
/// is large and the result has few equivalence classes (common after clearing
/// many terminal groups).
fn iterative_signature_refine(dfa: &DFA, initial_blocks: Vec<Vec<u32>>, max_iterations: u32) -> Option<Vec<Vec<u32>>> {
    let n = dfa.states().len();
    let mut partition = vec![0u32; n];
    for (block_id, block) in initial_blocks.iter().enumerate() {
        for &state in block {
            partition[state as usize] = block_id as u32;
        }
    }
    let mut num_blocks = initial_blocks.len() as u32;

    let mut label_map: FxHashMap<u64, u32> = FxHashMap::default();
    let mut iterations = 0u32;

    loop {
        if iterations >= max_iterations {
            return None;
        }

        label_map.clear();
        let mut next_label = 0u32;
        let mut new_partition = vec![0u32; n];

        for (state_idx, state) in dfa.states().iter().enumerate() {
            // Build signature: (current_block, [(byte, target_block)])
            // Use FxHash for speed instead of full signature comparison.
            let mut hasher = rustc_hash::FxHasher::default();
            partition[state_idx].hash(&mut hasher);
            state.transitions.len().hash(&mut hasher);
            for (byte, &target) in state.transitions.iter() {
                byte.hash(&mut hasher);
                partition[target as usize].hash(&mut hasher);
            }
            let sig_hash = hasher.finish();

            let label = *label_map.entry(sig_hash).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            new_partition[state_idx] = label;
        }

        if next_label == num_blocks {
            break;
        }
        num_blocks = next_label;
        partition = new_partition;
        iterations += 1;
    }

    // Build final blocks
    let mut blocks = vec![Vec::new(); num_blocks as usize];
    for (state_idx, &block_id) in partition.iter().enumerate() {
        blocks[block_id as usize].push(state_idx as u32);
    }
    Some(blocks)
}

fn topology_prerefine_partition(dfa: &DFA, partition: &[u32]) -> TopologyPrerefine {
    let adj = dedup_adjacency(dfa);
    let post_order = compute_post_order(&adj);
    let scc_id = compute_kosaraju_scc_ids(&adj, &post_order);

    let mut labels = vec![u32::MAX; dfa.states().len()];
    let mut label_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    label_map.reserve(dfa.states().len().min(200_000));
    let mut next_label = 0u32;
    let mut signature = Vec::with_capacity(32);

    for &state in &post_order {
        signature.clear();
        signature.push(partition[state]);

        for (byte, &target) in dfa.states()[state].transitions.iter() {
            let target = target as usize;
            signature.push(byte as u32);
            if scc_id[state] == scc_id[target] {
                signature.push(partition[target]);
            } else if labels[target] != u32::MAX {
                signature.push(labels[target] | 0x8000_0000);
            } else {
                signature.push(partition[target]);
            }
        }

        let label = *label_map.entry(signature.clone()).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        labels[state] = label;
    }

    let (partition, blocks) = build_blocks_from_labels(&labels, next_label);
    if next_label as usize == dfa.states().len() {
        if dfa.has_self_loops() {
            TopologyPrerefine::Skip
        } else {
            TopologyPrerefine::AlreadyMinimal(blocks)
        }
    } else {
        TopologyPrerefine::Refined { partition, blocks }
    }
}

enum InverseEdges {
    // Finite-mask DFAs are capped well below 2^24 states. Packing the source
    // into 24 bits and the byte label into the high byte halves inverse-edge
    // storage and reduces memory traffic in Hopcroft's predecessor scans.
    Packed(Vec<u32>),
    Wide(Vec<(u8, u32)>),
}

struct InverseTransitions {
    offsets: Vec<u32>,
    edges: InverseEdges,
}

impl InverseTransitions {
    fn build(dfa: &DFA) -> Self {
        const PACKED_SOURCE_LIMIT: usize = 1 << 24;
        let num_states = dfa.states().len();
        let mut counts = vec![0u32; num_states];
        for state in dfa.states() {
            for (_, &target) in state.transitions.iter() {
                counts[target as usize] += 1;
            }
        }

        let mut offsets = vec![0u32; num_states + 1];
        for state in 0..num_states {
            offsets[state + 1] = offsets[state] + counts[state];
        }

        let edge_count = offsets[num_states] as usize;
        let mut cursor = offsets[..num_states].to_vec();
        // Preserve predecessor order exactly: source states are visited in
        // ascending order and each state's byte transitions are already ordered.
        let edges = if num_states < PACKED_SOURCE_LIMIT {
            let mut edges = vec![0u32; edge_count];
            for (src, state) in dfa.states().iter().enumerate() {
                debug_assert!(src < PACKED_SOURCE_LIMIT);
                for (input, &target) in state.transitions.iter() {
                    let target = target as usize;
                    let index = cursor[target] as usize;
                    edges[index] = ((input as u32) << 24) | src as u32;
                    cursor[target] += 1;
                }
            }
            InverseEdges::Packed(edges)
        } else {
            let mut edges = vec![(0u8, 0u32); edge_count];
            for (src, state) in dfa.states().iter().enumerate() {
                for (input, &target) in state.transitions.iter() {
                    let target = target as usize;
                    let index = cursor[target] as usize;
                    edges[index] = (input, src as u32);
                    cursor[target] += 1;
                }
            }
            InverseEdges::Wide(edges)
        };

        Self { offsets, edges }
    }

    fn build_from_projected_rows(rows: &[Box<[(u8, u32)]>]) -> Self {
        const PACKED_SOURCE_LIMIT: usize = 1 << 24;
        let num_states = rows.len();
        let mut counts = vec![0u32; num_states];
        for row in rows {
            for &(_, target) in row.iter() {
                counts[target as usize] += 1;
            }
        }
        let mut offsets = vec![0u32; num_states + 1];
        for state in 0..num_states {
            offsets[state + 1] = offsets[state] + counts[state];
        }
        let edge_count = offsets[num_states] as usize;
        let mut cursor = offsets[..num_states].to_vec();
        let edges = if num_states < PACKED_SOURCE_LIMIT {
            let mut edges = vec![0u32; edge_count];
            for (src, row) in rows.iter().enumerate() {
                for &(input, target) in row.iter() {
                    let target = target as usize;
                    let index = cursor[target] as usize;
                    edges[index] = ((input as u32) << 24) | src as u32;
                    cursor[target] += 1;
                }
            }
            InverseEdges::Packed(edges)
        } else {
            let mut edges = vec![(0u8, 0u32); edge_count];
            for (src, row) in rows.iter().enumerate() {
                for &(input, target) in row.iter() {
                    let target = target as usize;
                    let index = cursor[target] as usize;
                    edges[index] = (input, src as u32);
                    cursor[target] += 1;
                }
            }
            InverseEdges::Wide(edges)
        };
        Self { offsets, edges }
    }

    #[inline]
    fn for_each_predecessor(&self, target: usize, mut f: impl FnMut(u8, u32)) {
        let start = self.offsets[target] as usize;
        let end = self.offsets[target + 1] as usize;
        match &self.edges {
            InverseEdges::Packed(edges) => {
                for &edge in &edges[start..end] {
                    f((edge >> 24) as u8, edge & 0x00ff_ffff);
                }
            }
            InverseEdges::Wide(edges) => {
                for &(input, source) in &edges[start..end] {
                    f(input, source);
                }
            }
        }
    }

    #[inline]
    fn for_each_predecessor_source(&self, target: usize, mut f: impl FnMut(u32)) {
        let start = self.offsets[target] as usize;
        let end = self.offsets[target + 1] as usize;
        match &self.edges {
            InverseEdges::Packed(edges) => {
                for &edge in &edges[start..end] {
                    f(edge & 0x00ff_ffff);
                }
            }
            InverseEdges::Wide(edges) => {
                for &(_, source) in &edges[start..end] {
                    f(source);
                }
            }
        }
    }
}

fn hopcroft_refine_partition_impl<const CANONICAL_OUTPUT: bool>(
    dfa: &DFA,
    partition: Vec<u32>,
    blocks: Vec<Vec<u32>>,
    inverse: &InverseTransitions,
) -> Vec<Vec<u32>> {
    hopcroft_refine_partition_impl_for_state_count::<CANONICAL_OUTPUT>(
        dfa.states().len(),
        partition,
        blocks,
        inverse,
    )
}

fn hopcroft_refine_partition_impl_for_state_count<const CANONICAL_OUTPUT: bool>(
    num_states: usize,
    mut partition: Vec<u32>,
    mut blocks: Vec<Vec<u32>>,
    inverse: &InverseTransitions,
) -> Vec<Vec<u32>> {

    let mut worklist: VecDeque<u32> = (0..blocks.len() as u32).collect();
    let mut in_worklist = vec![true; blocks.len()];
    let mut position_in_block = if CANONICAL_OUTPUT {
        let mut positions = vec![0u32; num_states];
        for block in &blocks {
            for (position, &state) in block.iter().enumerate() {
                positions[state as usize] = position as u32;
            }
        }
        positions
    } else {
        Vec::new()
    };

    let mut source_set = vec![false; num_states];
    let mut sources_to_clear: Vec<u32> = Vec::with_capacity(num_states.min(10_000));
    let mut touched_blocks: Vec<u32> = Vec::with_capacity(1024);
    let mut block_touched = vec![false; blocks.len()];
    let mut block_sources: Vec<Vec<u32>> = vec![Vec::new(); blocks.len()];
    let mut input_sources: Vec<Vec<u32>> = vec![Vec::new(); 256];
    let mut touched_inputs: Vec<u8> = Vec::with_capacity(64);

    while let Some(splitter_block) = worklist.pop_front() {
        let splitter_idx = splitter_block as usize;
        if splitter_idx >= in_worklist.len() {
            continue;
        }
        in_worklist[splitter_idx] = false;

        if splitter_idx >= blocks.len() || blocks[splitter_idx].is_empty() {
            continue;
        }
        touched_inputs.clear();
        for &target in &blocks[splitter_idx] {
            inverse.for_each_predecessor(target as usize, |input, src| {
                let bucket = &mut input_sources[input as usize];
                if bucket.is_empty() {
                    touched_inputs.push(input);
                }
                bucket.push(src);
            });
        }

        if touched_inputs.is_empty() {
            continue;
        }

        for &input in &touched_inputs {
            sources_to_clear.clear();
            let bucket = &mut input_sources[input as usize];
            for &src in bucket.iter() {
                if CANONICAL_OUTPUT {
                    // For a deterministic DFA, a source has at most one
                    // transition on a fixed input byte, so this bucket has no
                    // duplicate sources. The canonical fast path only needs
                    // `source_set` when the predecessor side is the larger
                    // half and we must scan/materialize its complement.
                    let block_id = partition[src as usize] as usize;
                    if block_id < block_touched.len() && !block_touched[block_id] {
                        block_touched[block_id] = true;
                        touched_blocks.push(block_id as u32);
                    }
                    block_sources[block_id].push(src);
                } else if !source_set[src as usize] {
                    source_set[src as usize] = true;
                    sources_to_clear.push(src);

                    let block_id = partition[src as usize] as usize;
                    if block_id < block_touched.len() && !block_touched[block_id] {
                        block_touched[block_id] = true;
                        touched_blocks.push(block_id as u32);
                    }
                    block_sources[block_id].push(src);
                }
            }
            bucket.clear();

            for &block_id in &touched_blocks {
                let block_idx = block_id as usize;
                if block_idx >= blocks.len() {
                    continue;
                }
                let block_len = blocks[block_idx].len();
                if block_len <= 1 {
                    continue;
                }

                let source_count = block_sources[block_idx].len();
                if source_count == 0 || source_count == block_len {
                    continue;
                }

                let new_block_idx = blocks.len();
                let move_sources = source_count <= block_len - source_count;
                let new_block = if CANONICAL_OUTPUT && move_sources {
                    // This variant is used only when all resulting blocks are
                    // canonicalized before rebuilding. Temporary within-block
                    // order therefore does not affect representatives or state
                    // numbering. Remove just the already-materialized smaller
                    // predecessor side instead of rescanning the entire block.
                    let moved = std::mem::take(&mut block_sources[block_idx]);
                    for &state in &moved {
                        let state_idx = state as usize;
                        let position = position_in_block[state_idx] as usize;
                        debug_assert_eq!(blocks[block_idx][position], state);
                        let removed = blocks[block_idx].swap_remove(position);
                        debug_assert_eq!(removed, state);
                        if position < blocks[block_idx].len() {
                            let swapped = blocks[block_idx][position] as usize;
                            position_in_block[swapped] = position as u32;
                        }
                    }
                    moved
                } else {
                    if CANONICAL_OUTPUT {
                        for &state in &block_sources[block_idx] {
                            source_set[state as usize] = true;
                        }
                    }
                    let old_block = std::mem::take(&mut blocks[block_idx]);
                    let (remaining, new_block) = if move_sources {
                        let mut remaining = Vec::with_capacity(block_len - source_count);
                        for state in old_block {
                            if !source_set[state as usize] {
                                remaining.push(state);
                            }
                        }
                        (remaining, std::mem::take(&mut block_sources[block_idx]))
                    } else {
                        let mut new_block = Vec::with_capacity(block_len - source_count);
                        for state in old_block {
                            if !source_set[state as usize] {
                                new_block.push(state);
                            }
                        }
                        (std::mem::take(&mut block_sources[block_idx]), new_block)
                    };
                    if CANONICAL_OUTPUT {
                        // On the canonical path this branch is reached only
                        // when the predecessor side is the larger half, so
                        // `remaining` is exactly the marked predecessor side
                        // that was taken out of `block_sources` above.
                        for &state in &remaining {
                            source_set[state as usize] = false;
                        }
                    }
                    blocks[block_idx] = remaining;
                    if CANONICAL_OUTPUT {
                        for (position, &state) in blocks[block_idx].iter().enumerate() {
                            position_in_block[state as usize] = position as u32;
                        }
                    }
                    new_block
                };

                for (position, &state) in new_block.iter().enumerate() {
                    partition[state as usize] = new_block_idx as u32;
                    if CANONICAL_OUTPUT {
                        position_in_block[state as usize] = position as u32;
                    }
                }

                blocks.push(new_block);

                in_worklist.push(false);
                block_touched.push(false);
                block_sources.push(Vec::new());

                if in_worklist[block_idx] {
                    in_worklist[new_block_idx] = true;
                    worklist.push_back(new_block_idx as u32);
                } else if blocks[block_idx].len() <= blocks[new_block_idx].len() {
                    in_worklist[block_idx] = true;
                    worklist.push_back(block_idx as u32);
                } else {
                    in_worklist[new_block_idx] = true;
                    worklist.push_back(new_block_idx as u32);
                }
            }

            if !CANONICAL_OUTPUT {
                for &src in &sources_to_clear {
                    source_set[src as usize] = false;
                }
            }

            for &block_id in &touched_blocks {
                if (block_id as usize) < block_touched.len() {
                    block_touched[block_id as usize] = false;
                    block_sources[block_id as usize].clear();
                }
            }
            touched_blocks.clear();
        }
    }

    blocks
}

/// Exact Moore-machine equivalence classes for an already-projected scalar
/// byte automaton. Missing byte edges have the common implicit dead behavior.
/// Class zero is reserved by the caller, so returned live classes start at 1.
pub(super) fn hopcroft_projected_observation_partition(
    rows: &[Box<[(u8, u32)]>],
    observations: &[u8],
) -> Vec<u32> {
    debug_assert_eq!(rows.len(), observations.len());
    let num_states = rows.len();
    let mut partition = vec![0u32; num_states];
    let mut blocks = Vec::<Vec<u32>>::new();
    let mut by_observation = FxHashMap::<u8, u32>::default();
    for (state, &observation) in observations.iter().enumerate() {
        let block = *by_observation.entry(observation).or_insert_with(|| {
            let id = blocks.len() as u32;
            blocks.push(Vec::new());
            id
        });
        partition[state] = block;
        blocks[block as usize].push(state as u32);
    }
    let inverse = InverseTransitions::build_from_projected_rows(rows);
    let blocks = hopcroft_refine_partition_impl_for_state_count::<false>(
        num_states,
        partition,
        blocks,
        &inverse,
    );
    let mut classes = vec![0u32; num_states];
    for (block, states) in blocks.into_iter().enumerate() {
        let class = block as u32 + 1;
        for state in states {
            classes[state as usize] = class;
        }
    }
    classes
}

fn hopcroft_refine_partition(
    dfa: &DFA,
    partition: Vec<u32>,
    blocks: Vec<Vec<u32>>,
) -> Vec<Vec<u32>> {
    let inverse = InverseTransitions::build(dfa);
    hopcroft_refine_partition_impl::<false>(dfa, partition, blocks, &inverse)
}

fn hopcroft_refine_partition_canonical(
    dfa: &DFA,
    partition: Vec<u32>,
    blocks: Vec<Vec<u32>>,
) -> Vec<Vec<u32>> {
    let inverse = InverseTransitions::build(dfa);
    hopcroft_refine_partition_impl::<true>(dfa, partition, blocks, &inverse)
}

fn can_reach_accepting_from_inverse(dfa: &DFA, inverse: &InverseTransitions) -> Vec<bool> {
    let n = dfa.states().len();
    let mut can_reach_accepting = vec![false; n];
    let mut queue = VecDeque::new();
    for state in 0..n {
        if !dfa.finalizers(state as u32).is_empty() {
            can_reach_accepting[state] = true;
            queue.push_back(state as u32);
        }
    }
    while let Some(target) = queue.pop_front() {
        inverse.for_each_predecessor_source(target as usize, |source| {
            let source = source as usize;
            if !can_reach_accepting[source] {
                can_reach_accepting[source] = true;
                queue.push_back(source as u32);
            }
        });
    }
    can_reach_accepting
}

fn compute_tarjan_scc_ids(adj: &[Vec<usize>]) -> (Vec<u32>, u32) {
    let num_states = adj.len();
    let mut scc_id = vec![u32::MAX; num_states];
    let mut scc_count: u32 = 0;
    let mut index_counter: u32 = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; num_states];
    let mut lowlink = vec![0u32; num_states];
    let mut disc = vec![u32::MAX; num_states];
    let mut dfs_stack: Vec<(usize, usize)> = Vec::new();

    for root in 0..num_states {
        if disc[root] != u32::MAX {
            continue;
        }
        dfs_stack.push((root, 0));
        disc[root] = index_counter;
        lowlink[root] = index_counter;
        index_counter += 1;
        stack.push(root);
        on_stack[root] = true;

        while let Some(&mut (state, ref mut edge_index)) = dfs_stack.last_mut() {
            if *edge_index < adj[state].len() {
                let target = adj[state][*edge_index];
                *edge_index += 1;
                if disc[target] == u32::MAX {
                    disc[target] = index_counter;
                    lowlink[target] = index_counter;
                    index_counter += 1;
                    stack.push(target);
                    on_stack[target] = true;
                    dfs_stack.push((target, 0));
                } else if on_stack[target] {
                    lowlink[state] = lowlink[state].min(disc[target]);
                }
            } else {
                if lowlink[state] == disc[state] {
                    while let Some(member) = stack.pop() {
                        on_stack[member] = false;
                        scc_id[member] = scc_count;
                        if member == state {
                            break;
                        }
                    }
                    scc_count += 1;
                }
                dfs_stack.pop();
                if let Some(&mut (parent, _)) = dfs_stack.last_mut() {
                    lowlink[parent] = lowlink[parent].min(lowlink[state]);
                }
            }
        }
    }

    (scc_id, scc_count)
}

impl DFA {
    /// Minimize this DFA using Hopcroft's algorithm.
    /// Returns a new, minimized DFA.  State 0 remains the start state.
    pub(super) fn minimize(&self) -> DFA {
        // Hopcroft below assumes one deterministic byte target and no epsilon
        // edges. Epsilon-NFAs are already assembled from independently
        // minimized DFA components, so preserving them is both correct and
        // avoids reintroducing the cross-terminal subset blow-up this mode is
        // intended to prevent.
        if self.has_epsilon_transitions() {
            return self.clone();
        }
        self.minimize_impl(true, false).0
    }

    /// Minimize a freshly constructed, fully reachable DFA in place.
    ///
    /// This avoids cloning the source and moves representative states out of
    /// their partition blocks, rewriting transition targets in place. Callers
    /// must guarantee that every state is reachable from state zero.
    pub(super) fn minimize_owned_reachable(mut self) -> DFA {
        if self.has_epsilon_transitions() || self.states().len() <= 1 {
            self.recompute_possible_futures();
            return self;
        }

        clear_possible_futures_for_minimization(&mut self);
        let (partition, blocks) = partition_by_finalizers(&self);
        let blocks = if self.has_self_loops() {
            hopcroft_refine_partition(&self, partition, blocks)
        } else {
            match topology_prerefine_partition(&self, &partition) {
                TopologyPrerefine::AlreadyMinimal(blocks) => blocks,
                TopologyPrerefine::Refined { blocks: refined, .. }
                    if refined.iter().all(|block| block.len() <= 1) => refined,
                TopologyPrerefine::Refined { .. } | TopologyPrerefine::Skip => {
                    hopcroft_refine_partition(&self, partition, blocks)
                }
            }
        };
        self.rebuild_owned_from_blocks(blocks)
    }

    /// Minimize this DFA and return the mapping from original states to
    /// minimized states.  `mapping[old_state] = new_state`.
    /// Unreachable original states map to `u32::MAX`.
    pub(super) fn minimize_with_state_mapping(&self) -> (DFA, Vec<u32>) {
        if self.has_epsilon_transitions() {
            return (self.clone(), (0..self.states().len() as u32).collect());
        }
        self.minimize_impl(true, false)
    }

    pub(super) fn minimize_with_state_mapping_preserve_unreachable(mut self) -> (DFA, Vec<u32>) {
        let orig_n = self.states().len();
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let total_started = std::time::Instant::now();
        if self.has_epsilon_transitions() {
            return (self, (0..orig_n as u32).collect());
        }
        if orig_n == 0 {
            return (self, Vec::new());
        }

        // This path is used by the finite vocabulary-mask component builder,
        // which already owns the sparse DFA. Minimize that allocation in place
        // rather than cloning every state/transition before refinement.
        clear_possible_futures_for_minimization(&mut self);
        if orig_n <= 1 {
            self.recompute_possible_futures();
            return (self, (0..orig_n as u32).collect());
        }

        // Finite repeat projections are commonly DAGs. Their exact partition
        // follows in one reverse-topological pass; avoid constructing inverse
        // edges and running repeated Hopcroft splitters in that case. General
        // cyclic or multi-terminal automata retain the established path.
        let acyclic_started = std::time::Instant::now();
        if std::env::var_os("GLRMASK_DISABLE_FINITE_MASK_ACYCLIC_MINIMIZE").is_none()
            && let Some(partition) = acyclic_single_group_flat_partition_impl::<{ u64::MAX }>(&self)
        {
            let partition_ms = acyclic_started.elapsed().as_secs_f64() * 1000.0;
            let result = self.rebuild_owned_from_flat_acyclic_partition(partition);
            if profile {
                eprintln!("[glrmask/profile][lexer_minimize_preserve_acyclic] states={} partition_ms={partition_ms:.3} total_ms={:.3}",
                    orig_n, total_started.elapsed().as_secs_f64() * 1000.0);
            }
            return result;
        }

        let partition_started = std::time::Instant::now();
        let (partition, blocks) = partition_by_finalizers(&self);
        let partition_ms = partition_started.elapsed().as_secs_f64() * 1000.0;

        let hopcroft_started = std::time::Instant::now();
        let inverse = InverseTransitions::build(&self);
        let mut blocks =
            hopcroft_refine_partition_impl::<true>(&self, partition, blocks, &inverse);
        let hopcroft_ms = hopcroft_started.elapsed().as_secs_f64() * 1000.0;
        let canonicalize_started = std::time::Instant::now();
        canonicalize_partition_blocks(&mut blocks);
        let canonicalize_ms = canonicalize_started.elapsed().as_secs_f64() * 1000.0;
        let future_started = std::time::Instant::now();
        let single_group_reachability =
            (self.num_groups() == 1).then(|| can_reach_accepting_from_inverse(&self, &inverse));
        let future_ms = future_started.elapsed().as_secs_f64() * 1000.0;
        let rebuild_started = std::time::Instant::now();
        let result = self.rebuild_owned_from_blocks_with_mapping_impl(
            blocks,
            single_group_reachability.as_deref(),
        );
        let rebuild_ms = rebuild_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][lexer_minimize_preserve] states={} partition_ms={partition_ms:.3} hopcroft_ms={hopcroft_ms:.3} canonicalize_ms={canonicalize_ms:.3} future_ms={future_ms:.3} rebuild_ms={rebuild_ms:.3} total_ms={:.3}",
                orig_n,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    fn minimize_impl(&self, drop_unreachable: bool, canonicalize_blocks: bool) -> (DFA, Vec<u32>) {
        let orig_n = self.states().len();
        if orig_n == 0 {
            return (self.clone(), Vec::new());
        }

        let mut working = self.clone();
        let old_to_working = if drop_unreachable {
            working.remove_unreachable_states_with_mapping()
        } else {
            (0..orig_n as u32).collect()
        };
        clear_possible_futures_for_minimization(&mut working);
        let n = working.states().len();

        if n <= 1 {
            working.recompute_possible_futures();
            return (working, old_to_working);
        }

        let (partition, blocks) = partition_by_finalizers(&working);
        let mut minimality_check_blocks = blocks.clone();

        // Topology pre-refinement is only an early minimality certificate.
        // Its classes are not fed to Hopcroft because the one-pass topology
        // signature can over-split cyclic/product states. A DFA with a self-loop
        // can never be certified by this pass, so scanning it is dead work.
        let prerefine = if working.has_self_loops() {
            TopologyPrerefine::Skip
        } else {
            topology_prerefine_partition(&working, &partition)
        };
        match prerefine {
            TopologyPrerefine::AlreadyMinimal(blocks) => {
                let mut blocks = blocks;
                if canonicalize_blocks {
                    canonicalize_partition_blocks(&mut blocks);
                }
                let (result, block_map) = working.rebuild_from_blocks_with_mapping(blocks);
                let composed = compose_mappings(&old_to_working, &block_map);
                return (result, composed);
            }
            TopologyPrerefine::Refined { blocks: refined_blocks, .. } => {
                // Previously bailed out when refined_blocks.len() > n*9/10,
                // assuming the DFA was near-minimal. That heuristic was
                // unsound: topology_prerefine uses a one-pass signature that
                // over-splits when blocks depend on each other's refinement
                // (product-DFA states whose active language is equal but
                // whose one-pass signatures differ due to inactive-dimension
                // coupling). Always fall through to Hopcroft.
                minimality_check_blocks = refined_blocks;
            }
            TopologyPrerefine::Skip => {}
        }

        if minimality_check_blocks.iter().all(|block| block.len() <= 1) {
            if canonicalize_blocks {
                canonicalize_partition_blocks(&mut minimality_check_blocks);
            }
            let (result, block_map) = working.rebuild_from_blocks_with_mapping(minimality_check_blocks);
            let composed = compose_mappings(&old_to_working, &block_map);
            return (result, composed);
        }

        let mut blocks = hopcroft_refine_partition(&working, partition, blocks);
        if canonicalize_blocks {
            canonicalize_partition_blocks(&mut blocks);
        }

        let (result, block_map) = working.rebuild_from_blocks_with_mapping(blocks);
        let composed = compose_mappings(&old_to_working, &block_map);
        (result, composed)
    }

    /// Remove unreachable states, returning old→new mapping.
    /// Unreachable states map to `u32::MAX`.
    fn remove_unreachable_states_with_mapping(&mut self) -> Vec<u32> {
        self.remove_unreachable_states_with_roots_with_mapping(&[])
    }

    /// Remove states unreachable from state 0 or any provided extra roots,
    /// returning old→new mapping. Unreachable states map to `u32::MAX`.
    fn remove_unreachable_states_with_roots_with_mapping(&mut self, extra_roots: &[u32]) -> Vec<u32> {
        let n = self.states().len();
        let mut reachable = vec![false; n];
        let mut queue = vec![0usize];
        reachable[0] = true;
        for &root in extra_roots {
            let root = root as usize;
            if root < n && !reachable[root] {
                reachable[root] = true;
                queue.push(root);
            }
        }

        while let Some(state) = queue.pop() {
            for (_, &next) in self.states()[state].transitions.iter() {
                let next = next as usize;
                if !reachable[next] {
                    reachable[next] = true;
                    queue.push(next);
                }
            }
        }

        let mut state_mapping = vec![u32::MAX; n];
        if reachable.iter().all(|&is_reachable| is_reachable) {
            // All reachable — identity mapping.
            for i in 0..n {
                state_mapping[i] = i as u32;
            }
            return state_mapping;
        }

        let mut new_index: u32 = 0;
        for (old_index, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                state_mapping[old_index] = new_index;
                new_index += 1;
            }
        }

        let old_states = std::mem::take(self.states_mut());
        let mut new_states = Vec::with_capacity(new_index as usize);
        for (old_index, state) in old_states.into_iter().enumerate() {
            if reachable[old_index] {
                let mut new_state = state;
                let entries: Vec<(u8, u32)> = new_state
                    .transitions
                    .iter()
                    .map(|(byte, &next)| (byte, state_mapping[next as usize]))
                    .collect();
                new_state.transitions = CharTransitions::from_sorted_entries(entries);
                new_state.epsilon_transitions = new_state
                    .epsilon_transitions
                    .iter()
                    .filter_map(|&next| {
                        let mapped = state_mapping[next as usize];
                        (mapped != u32::MAX).then_some(mapped)
                    })
                    .collect();
                new_states.push(new_state);
            }
        }

        *self.states_mut() = new_states;
        state_mapping
    }

    /// Recompute the strict one-group future predicate in linear time.
    ///
    /// With one terminal group, the future bit is set exactly when one byte
    /// transition can reach a state from which acceptance remains possible.
    fn recompute_single_group_possible_futures(&mut self) {
        let n = self.states().len();
        let mut predecessor_counts = vec![0usize; n];
        for state in self.states() {
            for (_, &target) in state.transitions.iter() {
                predecessor_counts[target as usize] += 1;
            }
        }

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0usize);
        for &count in &predecessor_counts {
            offsets.push(offsets.last().copied().unwrap() + count);
        }
        let mut cursors = offsets[..n].to_vec();
        let mut predecessors = vec![0u32; offsets[n]];
        for (source, state) in self.states().iter().enumerate() {
            for (_, &target) in state.transitions.iter() {
                let target = target as usize;
                let cursor = &mut cursors[target];
                predecessors[*cursor] = source as u32;
                *cursor += 1;
            }
        }

        let mut can_reach_accepting = vec![false; n];
        let mut queue = VecDeque::new();
        for state in 0..n {
            if !self.finalizers(state as u32).is_empty() {
                can_reach_accepting[state] = true;
                queue.push_back(state as u32);
            }
        }
        while let Some(target) = queue.pop_front() {
            let target = target as usize;
            for &source in &predecessors[offsets[target]..offsets[target + 1]] {
                let source = source as usize;
                if !can_reach_accepting[source] {
                    can_reach_accepting[source] = true;
                    queue.push_back(source as u32);
                }
            }
        }

        let strict_future = self
            .states()
            .iter()
            .map(|state| {
                state
                    .transitions
                    .iter()
                    .any(|(_, &target)| can_reach_accepting[target as usize])
            })
            .collect::<Vec<_>>();
        for (state, has_future) in strict_future.into_iter().enumerate() {
            let mut future = BitSet::new(1);
            if has_future {
                future.set(0);
            }
            self.set_possible_future_group_ids(state as u32, future);
        }
    }

    /// Recompute `possible_future_group_ids` for all states via fixpoint.
    pub(super) fn recompute_possible_futures(&mut self) {
        let n = self.states().len();
        let num_groups = self.num_groups();
        if n == 0 {
            return;
        }

        if self.has_epsilon_transitions() {
            self.recompute_possible_futures_with_epsilon();
            return;
        }
        if num_groups == 1 {
            self.recompute_single_group_possible_futures();
            return;
        }

        let adj = dedup_adjacency(self);
        let (scc_id, scc_count) = compute_tarjan_scc_ids(&adj);

        // `possible_future_group_ids` is strict: it should include only
        // groups reachable after consuming at least one more byte. That means
        // an accepting sink state has no possible futures, while a cyclic SCC
        // can include its own finalizers because they are reachable again via
        // a non-empty path through the cycle.
        let mut scc_finalizers: Vec<BitSet> = (0..scc_count as usize)
            .map(|_| BitSet::new(num_groups))
            .collect();
        let mut scc_sizes = vec![0usize; scc_count as usize];
        let mut scc_has_self_loop = vec![false; scc_count as usize];
        for (state_idx, state) in self.states().iter().enumerate() {
            let sid = scc_id[state_idx] as usize;
            scc_sizes[sid] += 1;
            for bit in state.finalizers.iter() {
                scc_finalizers[sid].set(bit);
            }
            if state
                .transitions
                .iter()
                .any(|(_, &target)| target as usize == state_idx)
            {
                scc_has_self_loop[sid] = true;
            }
        }

        let mut scc_futures: Vec<BitSet> = (0..scc_count as usize)
            .map(|_| BitSet::new(num_groups))
            .collect();
        for sid in 0..scc_count as usize {
            let is_cyclic = scc_sizes[sid] > 1 || scc_has_self_loop[sid];
            if is_cyclic {
                scc_futures[sid].union_with(&scc_finalizers[sid]);
            }
        }

        // Build SCC adjacency (successor SCCs for each SCC)
        let mut scc_successors: Vec<Vec<u32>> = vec![vec![]; scc_count as usize];
        for (state_idx, targets) in adj.iter().enumerate() {
            let src_scc = scc_id[state_idx];
            for &target in targets {
                let dst_scc = scc_id[target];
                if src_scc != dst_scc {
                    scc_successors[src_scc as usize].push(dst_scc);
                }
            }
        }
        // Dedup successors
        for succs in &mut scc_successors {
            succs.sort_unstable();
            succs.dedup();
        }

        let mut scc_predecessors: Vec<Vec<u32>> = vec![vec![]; scc_count as usize];
        let mut remaining_successors = vec![0usize; scc_count as usize];
        for (sid, successors) in scc_successors.iter().enumerate() {
            remaining_successors[sid] = successors.len();
            for &succ in successors {
                scc_predecessors[succ as usize].push(sid as u32);
            }
        }

        // Process SCCs from sinks upward so every predecessor sees fully
        // computed successor futures.
        let mut queue: VecDeque<u32> = remaining_successors
            .iter()
            .enumerate()
            .filter_map(|(sid, &count)| (count == 0).then_some(sid as u32))
            .collect();

        while let Some(sid) = queue.pop_front() {
            let sid = sid as usize;
            let sid_finalizers = scc_finalizers[sid].clone();
            let sid_futures = scc_futures[sid].clone();

            for &pred in &scc_predecessors[sid] {
                let pred = pred as usize;
                scc_futures[pred].union_with(&sid_finalizers);
                scc_futures[pred].union_with(&sid_futures);
                remaining_successors[pred] -= 1;
                if remaining_successors[pred] == 0 {
                    queue.push_back(pred as u32);
                }
            }
        }

        // Assign futures to states
        for state_idx in 0..n {
            let sid = scc_id[state_idx] as usize;
            self.set_possible_future_group_ids(state_idx as u32, scc_futures[sid].clone());
        }
    }

    fn recompute_possible_futures_with_epsilon(&mut self) {
        let n = self.states().len();
        let num_groups = self.num_groups();
        let mut successors: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut immediate: Vec<BitSet> = Vec::with_capacity(n);

        for state in 0..n as u32 {
            let source_closure = self.epsilon_closure(&[state]);
            let mut targets = Vec::new();
            let mut finals = BitSet::new(num_groups);
            for source in source_closure {
                for (_, &target) in self.states()[source as usize].transitions.iter() {
                    for closed_target in self.epsilon_closure(&[target]) {
                        if !targets.contains(&closed_target) {
                            targets.push(closed_target);
                        }
                        finals.union_with(&self.states()[closed_target as usize].finalizers);
                    }
                }
            }
            targets.sort_unstable();
            targets.dedup();
            successors.push(targets);
            immediate.push(finals);
        }

        let mut futures = immediate.clone();
        loop {
            let mut changed = false;
            for state in 0..n {
                let mut next = immediate[state].clone();
                for &target in &successors[state] {
                    next.union_with(&futures[target as usize]);
                }
                if next != futures[state] {
                    futures[state] = next;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for (state, future) in futures.into_iter().enumerate() {
            self.set_possible_future_group_ids(state as u32, future);
        }
    }

    fn rebuild_owned_from_flat_acyclic_partition(
        mut self,
        partition: AcyclicSingleGroupPartition,
    ) -> (DFA, Vec<u32>) {
        let AcyclicSingleGroupPartition {
            state_mapping, representatives, can_reach_accepting,
        } = partition;
        debug_assert_eq!(self.num_groups(), 1);
        debug_assert_eq!(self.num_states(), state_mapping.len());
        let mut old_index = 0usize;
        let mut next_representative = 0usize;
        // Canonical representatives are already ordered in the owned source
        // vector. Retain/move them in place rather than allocating a second
        // state vector and a separate old-representative-to-new map.
        self.states_mut().retain_mut(|state| {
            let source = old_index;
            old_index += 1;
            if representatives.get(next_representative).map(|&state| state as usize) != Some(source) {
                return false;
            }
            debug_assert_eq!(state_mapping[source] as usize, next_representative);
            next_representative += 1;
            // Inclusive successor reachability means STRICT nonempty futures:
            // an accepting sink does not become its own future witness.
            let has_future = state.transitions.iter()
                .any(|(_, &target)| can_reach_accepting[target as usize]);
            state.possible_future_group_ids = BitSet::new(1);
            if has_future {
                state.possible_future_group_ids.set(0);
            }
            for (_, target) in state.transitions.iter_mut() {
                *target = state_mapping[*target as usize];
            }
            debug_assert!(state.epsilon_transitions.is_empty());
            true
        });
        debug_assert_eq!(next_representative, representatives.len());
        (self, state_mapping)
    }

    fn rebuild_owned_from_blocks(self, partition_blocks: Vec<Vec<u32>>) -> DFA {
        self.rebuild_owned_from_blocks_with_mapping(partition_blocks).0
    }

    fn rebuild_owned_from_blocks_with_mapping(
        self,
        partition_blocks: Vec<Vec<u32>>,
    ) -> (DFA, Vec<u32>) {
        self.rebuild_owned_from_blocks_with_mapping_impl(partition_blocks, None)
    }

    fn rebuild_owned_from_blocks_with_mapping_impl(
        mut self,
        mut partition_blocks: Vec<Vec<u32>>,
        single_group_reachability: Option<&[bool]>,
    ) -> (DFA, Vec<u32>) {
        debug_assert!(single_group_reachability.is_none() || self.num_groups() == 1);
        let n = self.states().len();
        partition_blocks.retain(|block| !block.is_empty());
        if let Some(start_part_idx) = partition_blocks
            .iter()
            .position(|block| block.iter().any(|&state| state == 0))
        {
            partition_blocks.swap(0, start_part_idx);
        }

        let mut state_mapping = vec![0u32; n];
        let mut representative_to_new = vec![u32::MAX; n];
        for (new_idx, block) in partition_blocks.iter().enumerate() {
            representative_to_new[block[0] as usize] = new_idx as u32;
            for &old_idx in block {
                state_mapping[old_idx as usize] = new_idx as u32;
            }
        }

        let old_states = std::mem::take(self.states_mut());
        let mut representatives = std::iter::repeat_with(|| None)
            .take(partition_blocks.len())
            .collect::<Vec<_>>();
        for (old_idx, mut state) in old_states.into_iter().enumerate() {
            let new_idx = representative_to_new[old_idx];
            if new_idx == u32::MAX {
                continue;
            }
            if let Some(can_reach_accepting) = single_group_reachability {
                let has_future = state
                    .transitions
                    .iter()
                    .any(|(_, &target)| can_reach_accepting[target as usize]);
                state.possible_future_group_ids = BitSet::new(1);
                if has_future {
                    state.possible_future_group_ids.set(0);
                }
            }
            for (_, target) in state.transitions.iter_mut() {
                *target = state_mapping[*target as usize];
            }
            for target in &mut state.epsilon_transitions {
                *target = state_mapping[*target as usize];
            }
            representatives[new_idx as usize] = Some(state);
        }
        *self.states_mut() = representatives
            .into_iter()
            .map(|state| state.expect("partition representative missing"))
            .collect();
        if single_group_reachability.is_none() {
            self.recompute_possible_futures();
        }
        (self, state_mapping)
    }

    /// Rebuild DFA from partition blocks.
    /// Ensures state 0 in the new DFA corresponds to the block
    /// containing old state 0.
    fn rebuild_from_blocks(&self, partition_blocks: Vec<Vec<u32>>) -> DFA {
        self.rebuild_from_blocks_with_mapping(partition_blocks).0
    }

    /// Like `rebuild_from_blocks` but also returns old→new state mapping.
    fn rebuild_from_blocks_with_mapping(&self, mut partition_blocks: Vec<Vec<u32>>) -> (DFA, Vec<u32>) {
        let n = self.states().len();
        let mut state_mapping = vec![0u32; n];

        partition_blocks.retain(|block| !block.is_empty());

        // Ensure the block containing start state (0) is first.
        if let Some(start_part_idx) = partition_blocks
            .iter()
            .position(|block| block.iter().any(|&state| state == 0))
        {
            partition_blocks.swap(0, start_part_idx);
        }

        for (new_idx, block) in partition_blocks.iter().enumerate() {
            for &old_idx in block {
                state_mapping[old_idx as usize] = new_idx as u32;
            }
        }

        let num_groups = self.num_groups();
        let mut result = DFA::new(0);
        result.ensure_group_capacity(num_groups);
        for gid in 0..num_groups {
            result.set_group_u8set(gid as u32, self.group_id_to_u8set(gid as u32).clone());
        }

        for block in &partition_blocks {
            let representative = block[0] as usize;
            let old_state = &self.states()[representative];

            let new_id = result.add_state();
            let new_state = &mut result.states_mut()[new_id as usize];
            new_state.finalizers = old_state.finalizers.clone();
            let entries: Vec<(u8, u32)> = old_state
                .transitions
                .iter()
                .map(|(byte, &old_next)| (byte, state_mapping[old_next as usize]))
                .collect();
            new_state.transitions = CharTransitions::from_sorted_entries(entries);
            new_state.epsilon_transitions = old_state
                .epsilon_transitions
                .iter()
                .map(|&old_next| state_mapping[old_next as usize])
                .collect();
        }

        result.recompute_possible_futures();
        (result, state_mapping)
    }

}

fn canonicalize_partition_blocks(blocks: &mut Vec<Vec<u32>>) {
    for block in blocks.iter_mut() {
        block.sort_unstable();
    }
    blocks.sort_unstable_by_key(|block| block.first().copied().unwrap_or(u32::MAX));
}

/// Compose two state mappings: first[i] → second[first[i]].
/// Entries with `u32::MAX` in `first` stay as `u32::MAX`.
fn compose_mappings(first: &[u32], second: &[u32]) -> Vec<u32> {
    first
        .iter()
        .map(|&f| {
            if f == u32::MAX {
                u32::MAX
            } else {
                second[f as usize]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deliberately bypass the new path: this is the original canonical
    // preserve-unreachable Hopcroft implementation, not a self-comparison.
    fn original_preserve_hopcroft(mut dfa: DFA) -> (DFA, Vec<u32>) {
        let n = dfa.states().len();
        if dfa.has_epsilon_transitions() { return (dfa, (0..n as u32).collect()); }
        clear_possible_futures_for_minimization(&mut dfa);
        if n <= 1 { dfa.recompute_possible_futures(); return (dfa, (0..n as u32).collect()); }
        let (partition, blocks) = partition_by_finalizers(&dfa);
        let inverse = InverseTransitions::build(&dfa);
        let mut blocks = hopcroft_refine_partition_impl::<true>(&dfa, partition, blocks, &inverse);
        canonicalize_partition_blocks(&mut blocks);
        let reach = (dfa.num_groups() == 1).then(|| can_reach_accepting_from_inverse(&dfa, &inverse));
        dfa.rebuild_owned_from_blocks_with_mapping_impl(blocks, reach.as_deref())
    }

    fn assert_acyclic_matches_original(source: DFA) {
        let (blocks, reach) = acyclic_single_group_partition(&source).expect("generated DAG");
        let expected_partition = acyclic_single_group_partition_reference(&source).unwrap();
        assert_eq!((&blocks, &reach), (&expected_partition.0, &expected_partition.1));
        // Deliberately put ALL signatures into one collision chain, including
        // unequal accepting bits, edge labels and successor classes.
        assert_eq!(acyclic_single_group_partition_impl::<0>(&source).unwrap(), expected_partition);
        let actual = source.clone().rebuild_owned_from_blocks_with_mapping_impl(blocks, Some(&reach));
        let expected = original_preserve_hopcroft(source.clone());
        assert_eq!(actual.1, expected.1, "source-coordinate map changed");
        assert!(actual.0 == expected.0, "canonical DFA/strict future metadata changed");
        let flat = acyclic_single_group_flat_partition_impl::<0>(&source).unwrap();
        let flat_expected_representatives = expected_partition.0.iter()
            .map(|block| block[0]).collect::<Vec<_>>();
        assert_eq!(flat.representatives, flat_expected_representatives);
        assert_eq!(flat.state_mapping, expected.1);
        let flat_actual = source.clone().rebuild_owned_from_flat_acyclic_partition(flat);
        assert_eq!(flat_actual.1, expected.1);
        assert!(flat_actual.0 == expected.0, "flat rebuild under total hash collision changed DFA");
        let integrated = source.clone().minimize_with_state_mapping_preserve_unreachable();
        assert_eq!(integrated.1, expected.1);
        assert!(integrated.0 == expected.0);
        let mut recomputed = actual.0.clone();
        recomputed.recompute_possible_futures();
        assert!(actual.0 == recomputed, "strict futures must equal independent reachability");
        for old in 0..source.num_states() as u32 {
            let new = actual.1[old as usize];
            assert_ne!(new, u32::MAX, "unreachable projection roots must remain mapped");
            assert_eq!(source.finalizers(old), actual.0.finalizers(new));
            for byte in 0..=255u8 {
                assert_eq!(source.step(old, byte).map(|q| actual.1[q as usize]), actual.0.step(new, byte));
            }
        }
    }

    #[test]
    fn acyclic_preserve_matches_original_on_shuffled_generated_dags() {
        use rand::{Rng, SeedableRng};
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x109174_da6);
        for _case in 0..256 {
            let n = rng.gen_range(2..85);
            let mut order = (0..n as u32).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            let mut dfa = DFA::new(n);
            dfa.ensure_group_capacity(1);
            for (position, &state) in order.iter().enumerate() {
                let mut finalizers = BitSet::new(1);
                if rng.gen_bool(0.3) { finalizers.set(0); }
                // Stale future bits are intentionally ignored by minimization.
                let mut future = BitSet::new(1);
                if rng.gen_bool(0.5) { future.set(0); }
                dfa.overwrite_state_metadata(state, finalizers, future);
                if position + 1 < n {
                    for byte in 0..8u8 {
                        if rng.gen_bool(0.35) {
                            dfa.add_transition(state, byte, order[rng.gen_range(position+1..n)]);
                        }
                    }
                }
            }
            assert_acyclic_matches_original(dfa);
        }
    }

    #[test]
    fn acyclic_preserve_retains_disconnected_roots_and_strict_future() {
        let mut dfa = DFA::new(8);
        dfa.ensure_group_capacity(1);
        for (source, byte, target) in [(0,b'a',1),(1,b'b',2),(3,b'a',4),(4,b'b',5),(6,b'x',7)] {
            dfa.add_transition(source, byte, target);
        }
        let mut accepting = BitSet::new(1); accepting.set(0);
        for state in [2,5,7] { dfa.overwrite_state_metadata(state, accepting.clone(), accepting.clone()); }
        assert_acyclic_matches_original(dfa);
        for n in 0..2 {
            let mut tiny = DFA::new(n); tiny.ensure_group_capacity(1);
            assert_acyclic_matches_original(tiny);
        }
    }

    #[test]
    fn flat_acyclic_rebuild_retains_owned_storage_and_canonical_representatives() {
        // DFS discovery is deliberately different from source-state order, and
        // equivalent states are interleaved with disconnected/dead components.
        let mut dfa = DFA::new(12);
        dfa.ensure_group_capacity(1);
        for (source, byte, target) in [
            (0,b'a',8),(1,b'b',10),(2,b'a',9),(3,b'b',11),
            (4,b'c',1),(5,b'c',3),(6,b'd',7),
        ] {
            dfa.add_transition(source, byte, target);
        }
        let mut accepting = BitSet::new(1); accepting.set(0);
        for state in [8,9,10,11] {
            dfa.overwrite_state_metadata(state, accepting.clone(), accepting.clone());
        }
        let expected = original_preserve_hopcroft(dfa.clone());
        let flat = acyclic_single_group_flat_partition_impl::<{ u64::MAX }>(&dfa).unwrap();
        assert!(flat.representatives.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(flat.representatives[0], 0);
        assert!(flat.representatives.len() < dfa.num_states());
        let allocation = dfa.states().as_ptr();
        let (actual, mapping) = dfa.rebuild_owned_from_flat_acyclic_partition(flat);
        assert_eq!(actual.states().as_ptr(), allocation, "owned state storage was reallocated");
        assert_eq!(mapping, expected.1);
        assert!(actual == expected.0);
        assert!(actual.possible_future_group_ids(mapping[8]).is_empty(), "accepting sink is not a future");
        assert!(actual.possible_future_group_ids(mapping[6]).is_empty(), "dead continuation is not accepting");
    }

    #[test]
    fn acyclic_preserve_declines_cycles_epsilon_and_multiple_groups() {
        let mut cycle = DFA::new(5); cycle.ensure_group_capacity(1);
        cycle.add_transition(0,b'a',1);
        // Unreachable cycles must also be detected before classifying states.
        cycle.add_transition(3,b'b',4); cycle.add_transition(4,b'c',3);
        assert!(acyclic_single_group_partition(&cycle).is_none());
        let mut epsilon = DFA::new(2); epsilon.ensure_group_capacity(1);
        epsilon.add_epsilon_transition(0,1);
        let mut multigroup = DFA::new(2); multigroup.ensure_group_capacity(2);
        multigroup.add_transition(0,b'x',1);
        let mut self_loop = DFA::new(1); self_loop.ensure_group_capacity(1);
        self_loop.add_transition(0,b'x',0);
        for dfa in [cycle,epsilon,multigroup,self_loop] {
            assert!(acyclic_single_group_partition(&dfa).is_none());
            assert!(acyclic_single_group_partition_reference(&dfa).is_none());
            assert!(acyclic_single_group_partition_impl::<0>(&dfa).is_none());
            let actual = dfa.clone().minimize_with_state_mapping_preserve_unreachable();
            let expected = original_preserve_hopcroft(dfa);
            assert_eq!(actual.1,expected.1); assert!(actual.0==expected.0);
        }
    }

    #[test]
    #[ignore = "manual exact partition diagnostic; not whole-engine BUILD qualification"]
    fn acyclic_representative_signature_microbenchmark() {
        use std::time::Instant;
        let fixtures = [("tiny", 12, 4, false), ("chain", 20_000, 1, false),
            ("layered_sparse", 392, 96, false), ("layered_wide", 256, 96, true)];
        for (name, layers, width, wide) in fixtures {
            let n = layers * width;
            let mut dfa = DFA::new(n); dfa.ensure_group_capacity(1);
            for layer in 0..layers {
                for local in 0..width {
                    let state = (layer * width + local) as u32;
                    let mut accepting = BitSet::new(1);
                    if layer == layers-1 && local % 7 == 0 { accepting.set(0); }
                    dfa.overwrite_state_metadata(state, accepting, BitSet::new(1));
                    if layer+1 == layers { continue; }
                    let classes = if width == 1 { 1 } else if wide || local % 7 == 0 { 54 } else { 2 };
                    for class in 0..classes {
                        let target = ((layer+1)*width + (local*31 + class*17 + 3)%width) as u32;
                        dfa.add_transition(state, class as u8, target);
                    }
                }
            }
            assert_eq!(acyclic_single_group_partition(&dfa), acyclic_single_group_partition_reference(&dfa));
            for repeat in 0..8 {
                for optimized in if repeat%2 == 0 { [false,true] } else { [true,false] } {
                    let started = Instant::now();
                    let result = if optimized { acyclic_single_group_partition(&dfa) }
                        else { acyclic_single_group_partition_reference(&dfa) }.unwrap();
                    std::hint::black_box(&result);
                    let ns = started.elapsed().as_nanos();
                    eprintln!("[acyclic-signature-benchmark] fixture={name} states={n} repeat={repeat} optimized={optimized} ns={ns} classes={}",result.0.len());
                }
            }
        }
    }

    #[test]
    fn acyclic_preserve_deep_chain_uses_no_recursive_walk() {
        let n = 20_000;
        let mut dfa = DFA::new(n); dfa.ensure_group_capacity(1);
        for state in 0..n as u32-1 { dfa.add_transition(state,b'a',state+1); }
        let mut accepting = BitSet::new(1); accepting.set(0);
        dfa.overwrite_state_metadata(n as u32-1,accepting,BitSet::new(1));
        let (blocks, reach) = acyclic_single_group_partition(&dfa).unwrap();
        assert_eq!(blocks.len(),n); assert!(reach.iter().all(|&v|v));
        let expected = dfa.clone().rebuild_owned_from_blocks_with_mapping_impl(blocks,Some(&reach));
        let (result,map) = dfa.minimize_with_state_mapping_preserve_unreachable();
        assert_eq!(map, expected.1);
        assert!(result == expected.0);
        assert_eq!(map,(0..n as u32).collect::<Vec<_>>());
        assert!(result.possible_future_group_ids(n as u32-1).is_empty());
        assert!(!result.possible_future_group_ids(0).is_empty());
    }

    #[test]
    fn preserve_unreachable_single_group_future_carry_matches_recompute() {
        let mut dfa = DFA::new(8);
        dfa.ensure_group_capacity(1);
        for (source, byte, target) in [
            (0, b'a', 1),
            (1, b'b', 2),
            (2, b'c', 2),
            (3, b'd', 4),
            (4, b'e', 5),
            (6, b'f', 7),
        ] {
            dfa.add_transition(source, byte, target);
        }
        for accepting in [2u32, 5] {
            let mut finalizers = BitSet::new(1);
            finalizers.set(0);
            dfa.overwrite_state_metadata(accepting, finalizers, BitSet::new(1));
        }

        let (fast, _) = dfa.minimize_with_state_mapping_preserve_unreachable();
        let mut recomputed = fast.clone();
        recomputed.recompute_possible_futures();

        assert_eq!(fast.num_states(), recomputed.num_states());
        for state in 0..fast.num_states() as u32 {
            assert_eq!(
                fast.possible_future_group_ids(state),
                recomputed.possible_future_group_ids(state),
                "future mismatch at minimized state {state}",
            );
        }
    }

    #[test]
    fn canonical_hopcroft_swap_remove_matches_order_preserving_partition() {
        const STATES: usize = 128;
        let mut dfa = DFA::new(STATES);
        dfa.ensure_group_capacity(3);
        for state in 0..STATES {
            let mut finalizers = BitSet::new(3);
            if state % 3 == 0 {
                finalizers.set(0);
            }
            if state % 5 == 0 {
                finalizers.set(1);
            }
            if state % 11 == 0 {
                finalizers.set(2);
            }
            dfa.overwrite_state_metadata(state as u32, finalizers, BitSet::new(3));
            for input in 0..12u8 {
                let target = (state * 37 + input as usize * 17 + state / 4 + 3) % STATES;
                dfa.add_transition(state as u32, input, target as u32);
            }
        }

        let (partition, blocks) = partition_by_finalizers(&dfa);
        let initial_blocks = blocks.len();
        let mut expected = hopcroft_refine_partition(&dfa, partition.clone(), blocks.clone());
        let mut actual = hopcroft_refine_partition_canonical(&dfa, partition, blocks);
        canonicalize_partition_blocks(&mut expected);
        canonicalize_partition_blocks(&mut actual);

        assert!(actual.len() > initial_blocks, "test DFA must exercise block splitting");
        assert_eq!(actual, expected);
    }
}
