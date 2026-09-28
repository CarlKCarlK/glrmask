//! Vocabulary radix tries, packed walk programs, and slice-trie metadata.

use crate::compiler::stages::id_map_and_terminal_dwa::classify::VocabPartitionDfa;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::OnceLock;
/// Compact runtime-only vocabulary trie. It deliberately stores only the
/// information dynamic mask traversal consumes: compressed byte edges, child
/// ranges, and canonical token leaves.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieNode {
    pub(crate) token_id: Option<u32>,
    pub(crate) first_child: u32,
    pub(crate) child_len: u32,
    /// Canonical token ids below this node occupy one contiguous range in
    /// `DynamicMaskTrie::subtree_tokens`.
    pub(crate) subtree_token_start: u32,
    pub(crate) subtree_token_end: u32,
    /// Union of every byte on every edge strictly below this node.
    pub(crate) subtree_bytes: [u64; 4],
    /// Bytes that may occur next on some non-empty token suffix below this
    /// node. Unlike `subtree_bytes`, this records only the first consumed byte;
    /// zero-byte structural layout edges transparently inherit their child's
    /// first-byte set. Dynamic masking uses it to reject whole vocabulary
    /// layout classes before entering the radix walk when the current lexer
    /// configuration cannot consume any of their first bytes.
    pub(crate) subtree_first_bytes: [u64; 4],
    /// Number of vocabulary bytes consumed from the global trie root to reach
    /// this node. Structural partition edges have length zero and therefore do
    /// not affect it. This lets finite-horizon root-state certificates be
    /// reused only for subtrees whose *complete token strings* fit the proof
    /// horizon.
    pub(crate) prefix_byte_len: u32,
    /// Maximum number of token bytes still reachable strictly below this node.
    /// This is a runtime-only certificate aid: dynamic masking can prove that
    /// a lexer configuration stays safely live for this bounded horizon and
    /// accept the whole subtree without walking every token edge.
    pub(crate) subtree_max_byte_len: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieEdge {
    pub(crate) byte_start: u32,
    pub(crate) byte_len: u32,
    pub(crate) child: u32,
}

/// One radix edge in depth-first preorder. `subtree_end` is the first walk
/// entry after the child subtree, so a failed edge or accepted whole subtree
/// can be skipped with one index assignment.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieWalkEdge {
    pub(crate) byte_start: u32,
    pub(crate) child: u32,
    pub(crate) subtree_end: u32,
    pub(crate) byte_len: u16,
    pub(crate) parent_depth: u16,
}

/// One sequential operation in the strict full-vocabulary byte walk. Every
/// radix edge contributes at least one op; non-empty edges contribute one op
/// per consumed byte. This is purely a flatter view of the vocabulary trie: it
/// does not omit or summarize any edge.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrieFullWalkOp {
    pub(super) meta: u32,
}

const _: () = assert!(std::mem::size_of::<DynamicMaskTrieFullWalkOp>() == 4);

impl DynamicMaskTrieFullWalkOp {
    pub(super) const CONSUME: u32 = 1 << 24;
    pub(super) const START: u32 = 1 << 25;
    pub(super) const END: u32 = 1 << 26;
    pub(super) const TOKEN: u32 = 1 << 27;

    #[inline(always)]
    pub(crate) fn byte(&self) -> u8 {
        self.meta as u8
    }
    #[inline(always)]
    pub(crate) fn parent_depth(&self) -> u16 {
        ((self.meta >> 8) & 0xffff) as u16
    }
    #[inline(always)]
    pub(crate) fn consumes_byte(&self) -> bool {
        self.meta & Self::CONSUME != 0
    }
    #[inline(always)]
    pub(crate) fn starts_edge(&self) -> bool {
        self.meta & Self::START != 0
    }
    #[inline(always)]
    pub(crate) fn ends_edge(&self) -> bool {
        self.meta & Self::END != 0
    }
    #[inline(always)]
    pub(crate) fn child_is_token(&self) -> bool {
        self.meta & Self::TOKEN != 0
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskTrie {
    pub(crate) nodes: Vec<DynamicMaskTrieNode>,
    pub(crate) edges: Vec<DynamicMaskTrieEdge>,
    pub(super) edge_bytes: Vec<u8>,
    pub(super) subtree_tokens: Vec<u32>,
    pub(super) walk_edges: Vec<DynamicMaskTrieWalkEdge>,
    pub(super) full_walk_ops: Vec<DynamicMaskTrieFullWalkOp>,
    /// Edge index owning each strict full-walk op. Read only after death.
    pub(super) full_walk_op_edges: Vec<u32>,
    /// Starting full-walk op index for each DFS-preorder walk edge, plus one
    /// sentinel at the end. Combined with `DynamicMaskTrieWalkEdge::subtree_end`
    /// this gives an O(1) exact jump past a dead child subtree.
    pub(super) full_walk_edge_op_starts: Vec<u32>,
    /// For an ordinary, unpartitioned radix root, map each possible first byte
    /// directly to the first strict-walk op for its root child. `u32::MAX`
    /// means the vocabulary has no token beginning with that byte. Structural
    /// zero-byte roots deliberately leave this unavailable.
    #[serde(with = "optional_u32_256_serde")]
    pub(super) full_walk_root_byte_op_starts: Option<Box<[u32; 256]>>,
    pub(super) full_walk_token_nodes: Vec<u32>,
    /// Maximum structural radix-edge parent depth encoded by `full_walk_ops`.
    /// This is deliberately distinct from token byte length: one long
    /// compressed edge consumes many bytes without requiring additional DFS
    /// stack slots in the strict full walker.
    pub(super) full_walk_max_parent_depth: u16,
    /// Declared regular-language partition id for each zero-byte structural
    /// child of the true root. An empty table disables partition certificates.
    pub(super) root_layout_classes: Vec<u16>,
    /// True iff every complete token in the corresponding structural root
    /// class is valid UTF-8. Logical-scalar subtree proofs require this exact
    /// vocabulary property before treating non-ASCII bytes as UTF-8 scalars.
    pub(super) root_layout_all_valid_utf8: Vec<bool>,
}

mod optional_u32_256_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Option<Box<[u32; 256]>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .as_deref()
            .map(|row| row.as_slice())
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Box<[u32; 256]>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<Vec<u32>>::deserialize(deserializer)?;
        value
            .map(|row| {
                row.try_into().map(Box::new).map_err(|row: Vec<u32>| {
                    serde::de::Error::custom(format!(
                        "expected 256 root-byte entries, got {}",
                        row.len()
                    ))
                })
            })
            .transpose()
    }
}

/// Stable structural class used by the dynamic-mask radix trie.
///
/// This is deliberately *only* the declared regular-language partition id.
/// Older revisions refined it with arbitrary first-byte/character flags; that
/// made one structural root cease to correspond to a first-class language and
/// therefore prevented exact language-containment proofs. Any further runtime
/// acceleration must be represented as an explicit regular language instead.
pub(crate) fn dynamic_mask_vocab_layout_class(base_partition: u8, _bytes: &[u8]) -> u16 {
    u16::from(base_partition)
}

pub(crate) const DYNAMIC_MASK_LLG_MASTER_CACHE_ID: u32 = 0x200;

pub(super) const DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT: u16 = 1 << 15;

pub(super) const DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK: u16 =
    DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT - 1;

/// Structural class for the dynamic-radius LLG vocabulary trie. The low bits
/// are the exact number of Unicode scalar values in a whole token matching the
/// regex-defined safe-string language (`0` means not in that language); the high
/// bit records whole-token whitespace-regex membership. These are language
/// properties only. The trie deliberately does not mark them as compiler
/// partition languages, so generic partition certificates cannot reinterpret
/// the encoding.
pub(crate) fn dynamic_mask_llg_master_layout_class(safe_chars: u16, whitespace: bool) -> u16 {
    debug_assert!(safe_chars <= DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK);
    safe_chars
        | if whitespace {
            DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT
        } else {
            0
        }
}

#[inline(always)]
pub(crate) fn dynamic_mask_llg_master_safe_chars(class: u16) -> u16 {
    class & DYNAMIC_MASK_LLG_MASTER_SAFE_LEN_MASK
}

#[inline(always)]
pub(crate) fn dynamic_mask_llg_master_is_whitespace(class: u16) -> bool {
    class & DYNAMIC_MASK_LLG_MASTER_WHITESPACE_BIT != 0
}

impl DynamicMaskTrie {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![DynamicMaskTrieNode::default()],
            edges: Vec::new(),
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
            walk_edges: Vec::new(),
            full_walk_ops: Vec::new(),
            full_walk_op_edges: Vec::new(),
            full_walk_edge_op_starts: Vec::new(),
            full_walk_root_byte_op_starts: None,
            full_walk_token_nodes: Vec::new(),
            full_walk_max_parent_depth: 0,
            root_layout_classes: Vec::new(),
            root_layout_all_valid_utf8: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn root_layout_class(&self, root_slot: usize) -> Option<u16> {
        self.root_layout_classes.get(root_slot).copied()
    }

    #[inline]
    pub(crate) fn root_layout_all_valid_utf8(&self, root_slot: usize) -> bool {
        self.root_layout_all_valid_utf8
            .get(root_slot)
            .copied()
            .unwrap_or(false)
    }
    #[inline]
    pub(crate) fn node(&self, node: u32) -> &DynamicMaskTrieNode {
        &self.nodes[node as usize]
    }

    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub(crate) fn children(&self, node: u32) -> &[DynamicMaskTrieEdge] {
        let node = self.node(node);
        let start = node.first_child as usize;
        let end = start + node.child_len as usize;
        &self.edges[start..end]
    }

    #[inline]
    pub(crate) fn edge_bytes(&self, edge: &DynamicMaskTrieEdge) -> &[u8] {
        let start = edge.byte_start as usize;
        let end = start + edge.byte_len as usize;
        &self.edge_bytes[start..end]
    }

    #[inline]
    pub(crate) fn walk_edges(&self) -> &[DynamicMaskTrieWalkEdge] {
        &self.walk_edges
    }

    #[inline]
    pub(crate) fn walk_edge_bytes(&self, edge: &DynamicMaskTrieWalkEdge) -> &[u8] {
        let start = edge.byte_start as usize;
        let end = start + edge.byte_len as usize;
        &self.edge_bytes[start..end]
    }

    #[inline]
    pub(crate) fn full_walk_ops(&self) -> &[DynamicMaskTrieFullWalkOp] {
        &self.full_walk_ops
    }

    /// Exact strict-walk operation count after skipping structural root classes
    /// accepted by `admitted`.  Each admitted class still costs its one
    /// zero-byte structural root op; the remainder of that root subtree is
    /// skipped in O(1) by the runtime walker.
    pub(crate) fn full_walk_ops_after_root_class_skips(
        &self,
        mut admitted: impl FnMut(u16) -> bool,
    ) -> Option<usize> {
        if self.root_layout_classes.is_empty() {
            return None;
        }
        let mut root_slot = 0usize;
        let mut work = 0usize;
        for (edge_index, edge) in self.walk_edges.iter().copied().enumerate() {
            if edge.parent_depth != 0 {
                continue;
            }
            let class = *self.root_layout_classes.get(root_slot)?;
            root_slot += 1;
            let start = *self.full_walk_edge_op_starts.get(edge_index)? as usize;
            let end = *self
                .full_walk_edge_op_starts
                .get(edge.subtree_end as usize)? as usize;
            work = work.saturating_add(if admitted(class) {
                1
            } else {
                end.saturating_sub(start)
            });
        }
        (root_slot == self.root_layout_classes.len()).then_some(work)
    }

    /// Every full-walk op consumes one byte iff there are no synthetic
    /// empty-edge ops. Non-empty radix edges contribute exactly one op per
    /// byte; empty edges contribute one op and zero bytes.
    #[inline(always)]
    pub(crate) fn full_walk_all_consume(&self) -> bool {
        self.full_walk_ops.len() == self.edge_bytes.len()
    }

    #[inline(always)]
    pub(crate) fn full_walk_dead_subtree(&self, op_index: usize) -> (u32, u32) {
        debug_assert!(op_index < self.full_walk_op_edges.len());
        let edge_index = unsafe { *self.full_walk_op_edges.get_unchecked(op_index) } as usize;
        let edge = unsafe { *self.walk_edges.get_unchecked(edge_index) };
        let subtree_end_edge = edge.subtree_end as usize;
        debug_assert!(subtree_end_edge < self.full_walk_edge_op_starts.len());
        let subtree_end_op = unsafe {
            *self
                .full_walk_edge_op_starts
                .get_unchecked(subtree_end_edge)
        };
        (edge.child, subtree_end_op)
    }

    #[inline]
    pub(crate) fn full_walk_token_nodes(&self) -> &[u32] {
        &self.full_walk_token_nodes
    }

    #[inline]
    pub(crate) fn full_walk_max_parent_depth(&self) -> u16 {
        self.full_walk_max_parent_depth
    }

    #[inline(always)]
    pub(crate) fn has_full_walk_root_byte_index(&self) -> bool {
        self.full_walk_root_byte_op_starts.is_some()
    }

    #[inline(always)]
    pub(crate) fn full_walk_root_byte_range(&self, byte: u8) -> Option<(u32, u32, usize)> {
        let starts = self.full_walk_root_byte_op_starts.as_ref()?;
        let start_op = starts[byte as usize];
        if start_op == u32::MAX {
            return None;
        }
        let (child, end_op) = self.full_walk_dead_subtree(start_op as usize);
        let root_token_offset = usize::from(self.node(0).token_id.is_some());
        let marker_start = self
            .subtree_token_index_range(child)
            .start
            .saturating_sub(root_token_offset);
        Some((start_op, end_op, marker_start))
    }

    #[inline]
    pub(crate) fn subtree_tokens(&self, node: u32) -> &[u32] {
        let node = self.node(node);
        &self.subtree_tokens[node.subtree_token_start as usize..node.subtree_token_end as usize]
    }

    #[inline]
    pub(crate) fn subtree_token_index_range(&self, node: u32) -> std::ops::Range<usize> {
        let node = self.node(node);
        node.subtree_token_start as usize..node.subtree_token_end as usize
    }

    #[inline]
    pub(crate) fn all_subtree_tokens(&self) -> &[u32] {
        &self.subtree_tokens
    }

    #[inline]
    pub(crate) fn subtree_bytes(&self, node: u32) -> [u64; 4] {
        self.node(node).subtree_bytes
    }

    #[inline]
    pub(crate) fn subtree_first_bytes(&self, node: u32) -> [u64; 4] {
        self.node(node).subtree_first_bytes
    }

    #[inline]
    pub(crate) fn subtree_max_byte_len(&self, node: u32) -> u32 {
        self.node(node).subtree_max_byte_len
    }

    #[inline]
    pub(crate) fn subtree_max_total_byte_len(&self, node: u32) -> u32 {
        let node = self.node(node);
        node.prefix_byte_len
            .saturating_add(node.subtree_max_byte_len)
    }

    pub(crate) fn push_edge_bytes(&mut self, bytes: &[u8]) -> (u32, u32) {
        let start = self.edge_bytes.len() as u32;
        self.edge_bytes.extend_from_slice(bytes);
        (start, bytes.len() as u32)
    }

    #[inline]
    pub(crate) fn edge_bytes_len(&self) -> usize {
        self.edge_bytes.len()
    }

    pub(super) fn collect_subtree_metadata(
        &mut self,
        node_id: u32,
        prefix_byte_len: u32,
    ) -> ([u64; 4], [u64; 4], u32) {
        self.nodes[node_id as usize].prefix_byte_len = prefix_byte_len;
        let start = self.subtree_tokens.len() as u32;
        if let Some(token_id) = self.nodes[node_id as usize].token_id {
            self.subtree_tokens.push(token_id);
        }

        let first_child = self.nodes[node_id as usize].first_child as usize;
        let child_len = self.nodes[node_id as usize].child_len as usize;
        let mut subtree_bytes = [0u64; 4];
        let mut subtree_first_bytes = [0u64; 4];
        let mut subtree_max_byte_len = 0u32;
        for edge_index in first_child..first_child + child_len {
            // Copy the compact edge fields before recursing so no borrow of
            // `self.edges` remains live across the mutable recursive call.
            let edge = self.edges[edge_index].clone();
            let byte_start = edge.byte_start as usize;
            let byte_end = byte_start + edge.byte_len as usize;
            for &byte in &self.edge_bytes[byte_start..byte_end] {
                subtree_bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
            }
            let child_prefix_byte_len = prefix_byte_len
                .checked_add(edge.byte_len)
                .expect("dynamic mask trie token byte length exceeds u32");
            let (child_bytes, child_first_bytes, child_max_byte_len) =
                self.collect_subtree_metadata(edge.child, child_prefix_byte_len);
            for (target, child) in subtree_bytes.iter_mut().zip(child_bytes) {
                *target |= child;
            }
            if edge.byte_len == 0 {
                for (target, child) in subtree_first_bytes.iter_mut().zip(child_first_bytes) {
                    *target |= child;
                }
            } else {
                let first = self.edge_bytes[byte_start];
                subtree_first_bytes[first as usize >> 6] |= 1u64 << (first & 63);
            }
            subtree_max_byte_len = subtree_max_byte_len.max(
                edge.byte_len
                    .checked_add(child_max_byte_len)
                    .expect("dynamic mask trie token byte length exceeds u32"),
            );
        }

        let end = self.subtree_tokens.len() as u32;
        let node = &mut self.nodes[node_id as usize];
        node.subtree_token_start = start;
        node.subtree_token_end = end;
        node.subtree_bytes = subtree_bytes;
        node.subtree_first_bytes = subtree_first_bytes;
        node.subtree_max_byte_len = subtree_max_byte_len;
        (subtree_bytes, subtree_first_bytes, subtree_max_byte_len)
    }

    pub(crate) fn finalize_subtree_metadata(&mut self) {
        self.subtree_tokens.clear();
        self.subtree_tokens.reserve(self.nodes.len());
        if !self.nodes.is_empty() {
            self.collect_subtree_metadata(0, 0);
        }
        self.finalize_walk_edges();
    }

    pub(super) fn append_walk_edges(&mut self, node_id: u32, parent_depth: u16) {
        let first_child = self.nodes[node_id as usize].first_child as usize;
        let child_len = self.nodes[node_id as usize].child_len as usize;
        for edge_index in first_child..first_child + child_len {
            let edge = self.edges[edge_index].clone();
            let byte_len = u16::try_from(edge.byte_len)
                .expect("dynamic mask trie radix edge exceeds u16 length");
            let entry_index = self.walk_edges.len();
            self.walk_edges.push(DynamicMaskTrieWalkEdge {
                byte_start: edge.byte_start,
                child: edge.child,
                subtree_end: 0,
                byte_len,
                parent_depth,
            });
            self.append_walk_edges(
                edge.child,
                parent_depth
                    .checked_add(1)
                    .expect("dynamic mask trie depth exceeds u16"),
            );
            self.walk_edges[entry_index].subtree_end = self.walk_edges.len() as u32;
        }
    }

    pub(super) fn finalize_walk_edges(&mut self) {
        self.walk_edges.clear();
        self.walk_edges.reserve(self.edges.len());
        if !self.nodes.is_empty() {
            self.append_walk_edges(0, 0);
        }
        debug_assert_eq!(self.walk_edges.len(), self.edges.len());

        self.full_walk_ops.clear();
        self.full_walk_op_edges.clear();
        self.full_walk_edge_op_starts.clear();
        self.full_walk_root_byte_op_starts = None;
        self.full_walk_token_nodes.clear();
        self.full_walk_max_parent_depth = self
            .walk_edges
            .iter()
            .map(|edge| edge.parent_depth)
            .max()
            .unwrap_or(0);
        self.full_walk_ops
            .reserve(self.edge_bytes.len().max(self.walk_edges.len()));
        self.full_walk_op_edges
            .reserve(self.edge_bytes.len().max(self.walk_edges.len()));
        self.full_walk_edge_op_starts
            .reserve(self.walk_edges.len() + 1);
        self.full_walk_token_nodes.reserve(self.walk_edges.len());
        for (edge_index, edge) in self.walk_edges.iter().copied().enumerate() {
            self.full_walk_edge_op_starts
                .push(self.full_walk_ops.len() as u32);
            let start = edge.byte_start as usize;
            let end = start + edge.byte_len as usize;
            let bytes = &self.edge_bytes[start..end];
            let depth = u32::from(edge.parent_depth) << 8;
            let child_is_token = self.nodes[edge.child as usize].token_id.is_some();
            if bytes.is_empty() {
                let mut meta =
                    depth | DynamicMaskTrieFullWalkOp::START | DynamicMaskTrieFullWalkOp::END;
                if child_is_token {
                    meta |= DynamicMaskTrieFullWalkOp::TOKEN;
                }
                self.full_walk_ops.push(DynamicMaskTrieFullWalkOp { meta });
                self.full_walk_op_edges.push(edge_index as u32);
                if child_is_token {
                    self.full_walk_token_nodes.push(edge.child);
                }
                continue;
            }
            for (index, &byte) in bytes.iter().enumerate() {
                let mut meta = u32::from(byte) | depth | DynamicMaskTrieFullWalkOp::CONSUME;
                if index == 0 {
                    meta |= DynamicMaskTrieFullWalkOp::START;
                }
                let last = index + 1 == bytes.len();
                if last {
                    meta |= DynamicMaskTrieFullWalkOp::END;
                    if child_is_token {
                        meta |= DynamicMaskTrieFullWalkOp::TOKEN;
                    }
                }
                self.full_walk_ops.push(DynamicMaskTrieFullWalkOp { meta });
                self.full_walk_op_edges.push(edge_index as u32);
                if last && child_is_token {
                    self.full_walk_token_nodes.push(edge.child);
                }
            }
        }
        self.full_walk_edge_op_starts
            .push(self.full_walk_ops.len() as u32);
        debug_assert_eq!(self.full_walk_op_edges.len(), self.full_walk_ops.len());
        debug_assert_eq!(
            self.full_walk_edge_op_starts.len(),
            self.walk_edges.len() + 1
        );

        // Ordinary radix tries have one non-empty edge per distinct first byte
        // below the true root. Index those DFS ranges once so sparse lexer
        // roots can enter only byte-live vocabulary subtrees. Partitioned
        // tries have zero-byte structural root edges and intentionally decline
        // this optimization rather than conflating layout with byte language.
        let mut root_byte_starts = Box::new([u32::MAX; 256]);
        let mut root_byte_index_valid = true;
        for (edge_index, edge) in self.walk_edges.iter().copied().enumerate() {
            if edge.parent_depth != 0 {
                continue;
            }
            let bytes = self.walk_edge_bytes(&edge);
            let Some(&first) = bytes.first() else {
                root_byte_index_valid = false;
                break;
            };
            let slot = &mut root_byte_starts[first as usize];
            if *slot != u32::MAX {
                root_byte_index_valid = false;
                break;
            }
            *slot = self.full_walk_edge_op_starts[edge_index];
        }
        if root_byte_index_valid {
            self.full_walk_root_byte_op_starts = Some(root_byte_starts);
        }
    }

    pub(super) fn flatten_vocab_node(node: &VocabPrefixTreeNode, output: &mut Self) -> u32 {
        let node_id = output.nodes.len() as u32;
        output.nodes.push(DynamicMaskTrieNode {
            token_id: node.has_token().then_some(node.token_id() as u32),
            first_child: 0,
            child_len: 0,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });

        let children = node.children();
        if children.is_empty() {
            return node_id;
        }

        let first_child = output.edges.len() as u32;
        output.edges.resize_with(
            output.edges.len() + children.len(),
            DynamicMaskTrieEdge::default,
        );
        output.nodes[node_id as usize].first_child = first_child;
        output.nodes[node_id as usize].child_len = children.len() as u32;

        for (offset, (segment, child)) in node.iter_children().enumerate() {
            let child_id = Self::flatten_vocab_node(child, output);
            let (byte_start, byte_len) = output.push_edge_bytes(segment);
            output.edges[first_child as usize + offset] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: child_id,
            };
        }

        node_id
    }

    pub(super) fn from_vocab_prefix_tree_node(node: &VocabPrefixTreeNode) -> Self {
        let mut output = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_bytes: Vec::new(),
            subtree_tokens: Vec::new(),
            walk_edges: Vec::new(),
            full_walk_ops: Vec::new(),
            full_walk_op_edges: Vec::new(),
            full_walk_edge_op_starts: Vec::new(),
            full_walk_root_byte_op_starts: None,
            full_walk_token_nodes: Vec::new(),
            full_walk_max_parent_depth: 0,
            root_layout_classes: Vec::new(),
            root_layout_all_valid_utf8: Vec::new(),
        };
        let root = Self::flatten_vocab_node(node, &mut output);
        debug_assert_eq!(root, 0);
        output.finalize_subtree_metadata();
        output
    }

    pub(crate) fn from_vocab_prefix_tree(tree: &VocabPrefixTree) -> Self {
        // Root children are disjoint lexical subtrees. Flattening them in
        // parallel is safe, then the compact fragments are stitched with fixed
        // index offsets. This keeps the runtime representation lean without
        // making finalization wait on a single 140k-node recursive walk.
        let root = &tree.root;
        let root_children = root.children();
        if rayon::current_num_threads() == 1 || root_children.len() < 8 {
            return Self::from_vocab_prefix_tree_node(root);
        }

        let root_prefix_len = root.prefix().len();
        let mut fragments: Vec<(Box<[u8]>, Self)> = root_children
            .par_iter()
            .map(|child| {
                let edge = child.prefix()[root_prefix_len..]
                    .to_vec()
                    .into_boxed_slice();
                (edge, Self::from_vocab_prefix_tree_node(child))
            })
            .collect();
        let node_capacity = 1 + fragments
            .iter()
            .map(|(_, fragment)| fragment.nodes.len())
            .sum::<usize>();
        let edge_capacity = root_children.len()
            + fragments
                .iter()
                .map(|(_, fragment)| fragment.edges.len())
                .sum::<usize>();
        let byte_capacity = fragments
            .iter()
            .map(|(edge, fragment)| edge.len() + fragment.edge_bytes.len())
            .sum::<usize>();
        let mut output = Self {
            nodes: Vec::with_capacity(node_capacity),
            edges: Vec::with_capacity(edge_capacity),
            edge_bytes: Vec::with_capacity(byte_capacity),
            subtree_tokens: Vec::with_capacity(node_capacity),
            walk_edges: Vec::with_capacity(edge_capacity),
            full_walk_ops: Vec::with_capacity(byte_capacity.max(edge_capacity)),
            full_walk_op_edges: Vec::with_capacity(byte_capacity.max(edge_capacity)),
            full_walk_edge_op_starts: Vec::with_capacity(edge_capacity + 1),
            full_walk_root_byte_op_starts: None,
            full_walk_token_nodes: Vec::with_capacity(edge_capacity),
            full_walk_max_parent_depth: 0,
            root_layout_classes: Vec::new(),
            root_layout_all_valid_utf8: Vec::new(),
        };
        output.nodes.push(DynamicMaskTrieNode {
            token_id: root.has_token().then_some(root.token_id() as u32),
            first_child: 0,
            child_len: root_children.len() as u32,
            subtree_token_start: 0,
            subtree_token_end: 0,
            subtree_bytes: [0; 4],
            subtree_first_bytes: [0; 4],
            prefix_byte_len: 0,
            subtree_max_byte_len: 0,
        });
        output
            .edges
            .resize_with(root_children.len(), DynamicMaskTrieEdge::default);

        for (root_slot, (root_edge, mut fragment)) in fragments.drain(..).enumerate() {
            let node_base = output.nodes.len() as u32;
            let edge_base = output.edges.len() as u32;
            let byte_base = output.edge_bytes.len() as u32;
            output.edge_bytes.extend_from_slice(&fragment.edge_bytes);
            for node in &mut fragment.nodes {
                if node.child_len != 0 {
                    node.first_child += edge_base;
                }
            }
            for edge in &mut fragment.edges {
                edge.byte_start += byte_base;
                edge.child += node_base;
            }
            output.nodes.append(&mut fragment.nodes);
            output.edges.append(&mut fragment.edges);
            let (byte_start, byte_len) = output.push_edge_bytes(&root_edge);
            output.edges[root_slot] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: node_base,
            };
        }

        output.finalize_subtree_metadata();
        output
    }

    /// Build the same flat runtime radix-trie representation, but place one
    /// zero-byte structural node above each caller-supplied vocabulary layout
    /// class. `entries` must be ordered by `(class, token_bytes)` and token byte
    /// strings must already be canonical/deduplicated.
    ///
    /// The structural edges are a layout device only: they consume no input
    /// and are invisible to lexer semantics. Their purpose is to keep token
    /// families with different byte behaviour from contaminating one another's
    /// subtree metadata, so the generic runtime subtree certificates can skip
    /// large groups without any partition-specific masking logic.
    pub(crate) fn from_partitioned_token_refs(entries: &[(u16, usize, &[u8])]) -> Self {
        static DIRECT_FLAT: OnceLock<bool> = OnceLock::new();
        if *DIRECT_FLAT.get_or_init(|| std::env::var("GLRMASK_DIRECT_FLAT_MASK_TRIE")
            .map_or(true, |value| matches!(value.as_str(), "1" | "true")))
        {
            return Self::from_partitioned_sorted_direct(entries);
        }
        Self::from_partitioned_token_refs_reference(entries)
    }
}

impl Default for DynamicMaskTrie {
    fn default() -> Self {
        Self::new()
    }
}

/// One overlapping runtime slice language and the residual vocabulary trie
/// to walk after that language has been proved contained. The slice language
/// itself is not a compiler partition and may overlap/nest with other slices.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicMaskSliceTrie {
    pub(super) cache_id: u32,
    pub(super) dfa: Arc<VocabPartitionDfa>,
    pub(super) trie: Arc<DynamicMaskTrie>,
    pub(super) full_walk_token_markers: Arc<Vec<u64>>,
    pub(super) subtree_original_token_offsets: Arc<Vec<u32>>,
    pub(super) subtree_original_tokens: Arc<Vec<u32>>,
    pub(super) slice_original_token_words: Arc<Vec<u32>>,
    /// Conservative byte family containing every byte of every current-vocab
    /// token in this slice. Proving this whole family parser-transparent through
    /// `slice_max_token_byte_len` is sufficient to skip every slice token.
    pub(super) slice_token_bytes: U8Set,
    pub(super) slice_max_token_byte_len: u32,
    /// First-byte set of the slice DFA start state. Immutable per slice;
    /// cached because the dynamic pre-collapse gate needs it on every mask.
    #[serde(default, skip)]
    pub(super) first_bytes: OnceLock<U8Set>,
}

impl DynamicMaskSliceTrie {
    #[inline(always)]
    pub(crate) fn cache_id(&self) -> u32 {
        self.cache_id
    }

    #[inline(always)]
    pub(crate) fn dfa(&self) -> &VocabPartitionDfa {
        self.dfa.as_ref()
    }

    /// First-byte set of the slice DFA start state: the exact first-byte
    /// language of this proof slice. Only bytes whose one-byte derivative
    /// can still reach an accepting slice word are relevant here. This is
    /// deliberately computed from the DFA rather than from the bytes that
    /// occur anywhere inside whole slice tokens (UTF-8 continuation bytes,
    /// for example, are not valid first bytes). Immutable per slice;
    /// computed once because the dynamic pre-collapse gate needs it on
    /// every mask.
    pub(crate) fn first_bytes(&self) -> U8Set {
        *self.first_bytes.get_or_init(|| {
            let dfa = self.dfa.as_ref();
            let start = dfa.start_state();
            let mut result = U8Set::empty();
            for byte in 0u16..=255 {
                let byte = byte as u8;
                if dfa.can_reach_accepting(dfa.step(start, byte)) {
                    result.insert(byte);
                }
            }
            result
        })
    }

    #[inline(always)]
    pub(crate) fn trie(&self) -> &DynamicMaskTrie {
        self.trie.as_ref()
    }

    #[inline(always)]
    pub(super) fn full_walk_token_markers(&self) -> &[u64] {
        self.full_walk_token_markers.as_ref()
    }

    #[inline(always)]
    pub(super) fn subtree_original_tokens(&self, node: u32) -> &[u32] {
        let canonical_range = self.trie.subtree_token_index_range(node);
        let start = self.subtree_original_token_offsets[canonical_range.start] as usize;
        let end = self.subtree_original_token_offsets[canonical_range.end] as usize;
        &self.subtree_original_tokens[start..end]
    }

    #[inline(always)]
    pub(crate) fn slice_original_token_words(&self) -> &[u32] {
        self.slice_original_token_words.as_ref()
    }

    #[inline(always)]
    pub(crate) fn slice_token_bytes(&self) -> U8Set {
        self.slice_token_bytes
    }

    #[inline(always)]
    pub(crate) fn slice_max_token_byte_len(&self) -> u32 {
        self.slice_max_token_byte_len
    }
}

#[cfg(test)]
#[test]
fn direct_flat_mask_trie_preserves_all_reference_runtime_fields() {
    fn check(entries: &[(u16, usize, &[u8])]) {
        let reference = DynamicMaskTrie::from_partitioned_token_refs_reference(entries);
        let direct = DynamicMaskTrie::from_partitioned_sorted_direct(entries);
        assert_eq!(bincode::serialize(&reference).unwrap(), bincode::serialize(&direct).unwrap(),
            "all nodes, edges, byte layout, metadata, walk ops and language classes must agree");
    }
    check(&[]);
    let mut random = 7u64;
    for case in 0..512 {
        let mut next = || {random ^= random << 13; random ^= random >> 7; random ^= random << 17; random};
        let mut words = vec![(0u16, Vec::<u8>::new())];
        for _ in 0..case % 150 {
            let class = (next() % 5) as u16;
            let len = (next() % 32) as usize;
            words.push((class, (0..len).map(|_|(next() % 256) as u8).collect()));
        }
        words.sort(); words.dedup();
        let entries = words.iter().enumerate().map(|(i,(class,word))|
            (*class,i*17,word.as_slice())).collect::<Vec<_>>();
        check(&entries);
    }
    for depth in [1,16,64,256,512] {
        let words = (0..depth).map(|n|vec![b'a';n]).collect::<Vec<_>>();
        let entries = words.iter().enumerate().map(|(i,word)|(0u16,i,word.as_slice())).collect::<Vec<_>>();
        check(&entries);
    }
}


impl DynamicMaskTrie {
    /// Direct flat construction from sorted canonical entries. The iterative
    /// worklist preserves the reference's preorder nodes/postorder byte ranges
    /// without transient prefix trees, copied whole prefixes or range-set
    /// metadata. Existing subtree/walk metadata finalization is unchanged.
    fn from_partitioned_sorted_direct(entries: &[(u16, usize, &[u8])]) -> Self {
        enum Task {
            Node { lo: usize, hi: usize, depth: usize, edge_from: usize, parent_edge: usize },
            FinishEdge { entry: usize, from: usize, to: usize, edge: usize, child: u32 },
        }
        let mut output = Self::new();
        if entries.is_empty() { return output; }
        let mut index = 0;
        if entries[0].2.is_empty() {
            output.nodes[0].token_id = Some(entries[0].1 as u32);
            index = 1;
        }
        let mut groups = Vec::new();
        while index < entries.len() {
            let start = index; let class = entries[index].0; index += 1;
            while index < entries.len() && entries[index].0 == class { index += 1; }
            groups.push((start,index));
        }
        output.nodes.reserve(entries.len().saturating_mul(2));
        output.edges.reserve(entries.len().saturating_mul(2));
        output.edge_bytes.reserve(entries.iter().map(|e|e.2.len()).sum());
        output.edges.resize_with(groups.len(), DynamicMaskTrieEdge::default);
        output.nodes[0].child_len = groups.len() as u32;
        for &(lo,hi) in &groups {
            output.root_layout_classes.push(entries[lo].0);
            output.root_layout_all_valid_utf8.push(entries[lo..hi].iter().all(|e|std::str::from_utf8(e.2).is_ok()));
        }
        let mut work = Vec::with_capacity(64);
        for (parent_edge,&(lo,hi)) in groups.iter().enumerate().rev() {
            work.push(Task::Node {lo,hi,depth:0,edge_from:0,parent_edge});
        }
        while let Some(task) = work.pop() {
            match task {
                Task::FinishEdge {entry,from,to,edge,child} => {
                    let (byte_start,byte_len)=output.push_edge_bytes(&entries[entry].2[from..to]);
                    output.edges[edge]=DynamicMaskTrieEdge {byte_start,byte_len,child};
                }
                Task::Node {mut lo,hi,depth,edge_from,parent_edge} => {
                    let node=output.nodes.len() as u32;
                    output.nodes.push(DynamicMaskTrieNode::default());
                    work.push(Task::FinishEdge {entry:lo,from:edge_from,to:depth,edge:parent_edge,child:node});
                    if entries[lo].2.len()==depth {
                        output.nodes[node as usize].token_id=Some(entries[lo].1 as u32);
                        lo+=1;
                    }
                    if lo==hi {continue;}
                    let child_count=1+entries[lo..hi].windows(2)
                        .filter(|p|p[0].2[depth]!=p[1].2[depth]).count();
                    let first=output.edges.len();
                    output.edges.resize_with(first+child_count,DynamicMaskTrieEdge::default);
                    output.nodes[node as usize].first_child=first as u32;
                    output.nodes[node as usize].child_len=child_count as u32;
                    let mut end=hi;let mut slot=child_count;
                    while end>lo {
                        let byte=entries[end-1].2[depth];
                        let mut start=end-1;
                        while start>lo && entries[start-1].2[depth]==byte {start-=1;}
                        let a=entries[start].2;let b=entries[end-1].2;
                        let mut common=depth+1;
                        while common<a.len().min(b.len()) && a[common]==b[common] {common+=1;}
                        slot-=1;
                        work.push(Task::Node {lo:start,hi:end,depth:common,edge_from:depth,parent_edge:first+slot});
                        end=start;
                    }
                }
            }
        }
        output.finalize_subtree_metadata();
        output
    }
}


impl DynamicMaskTrie {

    fn from_partitioned_token_refs_reference(entries: &[(u16, usize, &[u8])]) -> Self {
        let mut output = Self::new();
        if entries.is_empty() {
            return output;
        }

        // Empty-token aliases are canonicalized before this stage, so at most
        // one canonical empty byte string may exist. Keep it on the true root.
        let mut start = 0usize;
        if entries[0].2.is_empty() {
            output.nodes[0].token_id = Some(entries[0].1 as u32);
            start = 1;
        }

        let mut groups = Vec::<Self>::new();
        let mut group_classes = Vec::<u16>::new();
        let mut group_all_valid_utf8 = Vec::<bool>::new();
        let mut index = start;
        while index < entries.len() {
            let class = entries[index].0;
            let group_start = index;
            index += 1;
            while index < entries.len() && entries[index].0 == class {
                index += 1;
            }
            let refs = entries[group_start..index]
                .iter()
                .map(|(_, token_id, bytes)| (*token_id, *bytes))
                .collect::<Vec<_>>();
            debug_assert!(refs.windows(2).all(|pair| pair[0].1 <= pair[1].1));
            group_classes.push(class);
            group_all_valid_utf8.push(
                entries[group_start..index]
                    .iter()
                    .all(|(_, _, bytes)| std::str::from_utf8(bytes).is_ok()),
            );
            let tree = VocabPrefixTree::build_presorted(&refs);
            groups.push(Self::from_vocab_prefix_tree_node(&tree.root));
        }

        let root_child_count = groups.len();
        output.edges.resize_with(root_child_count, DynamicMaskTrieEdge::default);
        output.nodes[0].first_child = 0;
        output.nodes[0].child_len = root_child_count as u32;

        for (root_slot, mut fragment) in groups.into_iter().enumerate() {
            let node_base = output.nodes.len() as u32;
            let edge_base = output.edges.len() as u32;
            let byte_base = output.edge_bytes.len() as u32;

            output.edge_bytes.extend_from_slice(&fragment.edge_bytes);
            for node in &mut fragment.nodes {
                if node.child_len != 0 {
                    node.first_child += edge_base;
                }
            }
            for edge in &mut fragment.edges {
                edge.byte_start += byte_base;
                edge.child += node_base;
            }
            output.nodes.append(&mut fragment.nodes);
            output.edges.append(&mut fragment.edges);

            // Structural class edge: no lexer byte is consumed here.
            let (byte_start, byte_len) = output.push_edge_bytes(&[]);
            output.edges[root_slot] = DynamicMaskTrieEdge {
                byte_start,
                byte_len,
                child: node_base,
            };
        }

        output.finalize_subtree_metadata();
        output.root_layout_classes = group_classes;
        output.root_layout_all_valid_utf8 = group_all_valid_utf8;
        output
    }
}
