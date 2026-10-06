use std::fmt;

use range_set_blaze::RangeSetBlaze;

#[derive(PartialEq)]
pub struct VocabPrefixTreeNode {
    token_id: usize,
    has_token: bool,
    prefix: Box<[u8]>,

    // Packed child storage.
    // Edge labels are implicit:
    // child_edge_label = &child.prefix[self.prefix.len()..]
    children: Box<[VocabPrefixTreeNode]>,

    reachable_token_ids: RangeSetBlaze<usize>,
    subtree_bytes: [u64; 4],
}

#[derive(Clone)]
pub struct VocabPrefixTreeChildIter<'a> {
    parent_prefix_len: usize,
    inner: std::slice::Iter<'a, VocabPrefixTreeNode>,
}

impl<'a> Iterator for VocabPrefixTreeChildIter<'a> {
    type Item = (&'a [u8], &'a VocabPrefixTreeNode);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let child = self.inner.next()?;
        Some((&child.prefix[self.parent_prefix_len..], child))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for VocabPrefixTreeChildIter<'_> {}

impl VocabPrefixTreeNode {
    #[inline]
    fn new(token_id: usize, prefix: Box<[u8]>, has_token: bool) -> Self {
        Self {
            token_id,
            has_token,
            prefix,
            children: Box::new([]),
            reachable_token_ids: RangeSetBlaze::new(),
            subtree_bytes: [0u64; 4],
        }
    }

    #[inline]
    pub fn token_id(&self) -> usize {
        self.token_id
    }

    #[inline]
    pub fn has_token(&self) -> bool {
        self.has_token
    }

    #[inline]
    pub fn prefix_length(&self) -> usize {
        self.prefix.len()
    }

    #[inline]
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    #[inline]
    pub fn children(&self) -> &[VocabPrefixTreeNode] {
        &self.children
    }

    #[inline]
    pub fn iter_children(&self) -> VocabPrefixTreeChildIter<'_> {
        VocabPrefixTreeChildIter {
            parent_prefix_len: self.prefix.len(),
            inner: self.children.iter(),
        }
    }

    #[inline]
    pub fn reachable_token_ids(&self) -> &RangeSetBlaze<usize> {
        &self.reachable_token_ids
    }

    #[inline]
    pub fn subtree_bytes(&self) -> &[u64; 4] {
        &self.subtree_bytes
    }

    #[inline]
    fn child_edge_label<'a>(&'a self, child: &'a VocabPrefixTreeNode) -> &'a [u8] {
        &child.prefix[self.prefix.len()..]
    }

    #[inline]
    fn child_key_byte(&self, child: &VocabPrefixTreeNode) -> u8 {
        child.prefix[self.prefix.len()]
    }

    #[inline]
    fn find_child(&self, next_byte: u8) -> Option<&VocabPrefixTreeNode> {
        let idx = self
            .children
            .binary_search_by_key(&next_byte, |child| self.child_key_byte(child))
            .ok()?;
        Some(&self.children[idx])
    }
}

fn insert_bytes_into_mask(mask: &mut [u64; 4], bytes: &[u8]) {
    for &byte in bytes {
        mask[byte as usize >> 6] |= 1u64 << (byte & 63);
    }
}

fn merge_reachable_token_ids(
    reachable_token_ids: &mut RangeSetBlaze<usize>,
    child: &VocabPrefixTreeNode,
) {
    *reachable_token_ids |= &child.reachable_token_ids;
}

fn merge_child_metadata(
    reachable_token_ids: &mut RangeSetBlaze<usize>,
    subtree_bytes: &mut [u64; 4],
    parent_prefix_len: usize,
    child: &VocabPrefixTreeNode,
) {
    insert_bytes_into_mask(subtree_bytes, &child.prefix[parent_prefix_len..]);
    for (target_word, child_word) in subtree_bytes.iter_mut().zip(child.subtree_bytes.iter()) {
        *target_word |= *child_word;
    }
    merge_reachable_token_ids(reachable_token_ids, child);
}

fn next_matching_child<'tree, 'bytes>(
    current: &'tree VocabPrefixTreeNode,
    remaining: &'bytes [u8],
) -> Option<(&'tree VocabPrefixTreeNode, &'bytes [u8])> {
    let child = current.find_child(*remaining.first()?)?;
    let edge = current.child_edge_label(child);
    remaining
        .starts_with(edge)
        .then_some((child, &remaining[edge.len()..]))
}

impl fmt::Debug for VocabPrefixTreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn format_bytes(bytes: &[u8]) -> String {
            const MAX_BYTES_DISPLAY: usize = 10;
            let display_str =
                String::from_utf8_lossy(bytes.get(..MAX_BYTES_DISPLAY).unwrap_or(bytes));
            if bytes.len() > MAX_BYTES_DISPLAY {
                format!("{}...({} bytes)", display_str, bytes.len())
            } else {
                display_str.to_string()
            }
        }

        let children_summary: Vec<String> = self
            .iter_children()
            .map(|(edge, _)| format_bytes(edge))
            .collect();

        let mut debug_struct = f.debug_struct("VocabPrefixTreeNode");
        debug_struct.field("token_id", &self.token_id);
        debug_struct.field("has_token", &self.has_token);
        debug_struct.field("prefix_length", &self.prefix.len());
        debug_struct.field("prefix", &format_bytes(&self.prefix));
        debug_struct.field(
            "reachable_token_ids",
            &format!("{} items", self.reachable_token_ids.len()),
        );
        debug_struct.field("children", &children_summary);
        debug_struct.finish()
    }
}

#[derive(Debug, PartialEq)]
pub struct VocabPrefixTree {
    pub root: VocabPrefixTreeNode,
    max_token_id: usize,
    has_empty_string_token: bool,
}

impl VocabPrefixTree {
    #[inline]
    pub fn new() -> Self {
        Self {
            root: VocabPrefixTreeNode::new(0, Vec::new().into_boxed_slice(), false),
            max_token_id: 0,
            has_empty_string_token: false,
        }
    }

    pub fn build(tokens: &[(usize, Vec<u8>)]) -> Self {
        Self::build_owned(tokens.iter().map(|(id, bytes)| (*id, bytes.clone())).collect())
    }

    fn sort_and_dedup_tokens(tokens: &mut Vec<(usize, Vec<u8>)>) {
        tokens.sort_unstable_by(|left, right| left.1.as_slice().cmp(right.1.as_slice()));

        let mut write_index = 0usize;
        for read_index in 0..tokens.len() {
            if write_index > 0 && tokens[write_index - 1].1 == tokens[read_index].1 {
                tokens[write_index - 1].0 = tokens[read_index].0;
                continue;
            }
            if write_index != read_index {
                tokens.swap(write_index, read_index);
            }
            write_index += 1;
        }
        tokens.truncate(write_index);
    }

    fn build_children(
        entries: &[(usize, &[u8])],
        parent_prefix_len: usize,
    ) -> Vec<VocabPrefixTreeNode> {
        let mut children = Vec::new();
        let mut index = 0usize;
        while index < entries.len() {
            let next_byte = entries[index].1[parent_prefix_len];
            let group_start = index;
            index += 1;
            while index < entries.len() && entries[index].1[parent_prefix_len] == next_byte {
                index += 1;
            }
            children.push(Self::build_subtree(&entries[group_start..index], parent_prefix_len));
        }
        children
    }

    pub fn build_owned(mut tokens: Vec<(usize, Vec<u8>)>) -> Self {
        Self::sort_and_dedup_tokens(&mut tokens);
        let refs: Vec<(usize, &[u8])> = tokens.iter().map(|(id, bytes)| (*id, bytes.as_slice())).collect();
        Self::build_presorted(&refs)
    }

    /// Build from tokens that are already sorted by byte content and deduplicated.
    pub fn build_presorted(tokens: &[(usize, &[u8])]) -> Self {
        let mut tree = Self::new();
        tree.max_token_id = tokens.iter().map(|(id, _)| *id).max().unwrap_or(0);

        if tokens.is_empty() {
            return tree;
        }

        let mut start = 0usize;
        if !tokens.is_empty() && tokens[0].1.is_empty() {
            tree.root.token_id = tokens[0].0;
            tree.root.has_token = true;
            tree.has_empty_string_token = true;
            tree.root.reachable_token_ids.insert(tokens[0].0);
            start = 1;
        }

        if start == tokens.len() {
            return tree;
        }

        let entries = &tokens[start..];
        let children = Self::build_children(entries, 0);
        let mut root_subtree_bytes = [0u64; 4];
        for child in &children {
            merge_child_metadata(
                &mut tree.root.reachable_token_ids,
                &mut root_subtree_bytes,
                0,
                child,
            );
        }

        tree.root.children = children.into_boxed_slice();
        tree.root.subtree_bytes = root_subtree_bytes;

        tree
    }

    #[inline]
    fn lcp_len(a: &[u8], b: &[u8], from: usize) -> usize {
        let max_len = a.len().min(b.len());
        let mut i = from;
        while i < max_len && a[i] == b[i] {
            i += 1;
        }
        i
    }

    fn build_subtree(entries: &[(usize, &[u8])], parent_prefix_len: usize) -> VocabPrefixTreeNode {
        debug_assert!(!entries.is_empty());

        let first = entries.first().unwrap().1;
        let last = entries.last().unwrap().1;

        // For lexicographically sorted entries, the common prefix of first and last
        // is the common prefix of the entire group.
        let prefix_len = Self::lcp_len(first, last, parent_prefix_len);
        let has_token = entries[0].1.len() == prefix_len;
        let token_id = if has_token { entries[0].0 } else { 0 };
        let prefix = first[..prefix_len].to_vec().into_boxed_slice();

        let child_entries = if has_token { &entries[1..] } else { entries };
        let children = Self::build_children(child_entries, prefix_len);

        let mut reachable_token_ids = RangeSetBlaze::new();
        if has_token {
            reachable_token_ids.insert(token_id);
        }

        let mut subtree_bytes = [0u64; 4];
        for child in &children {
            merge_child_metadata(&mut reachable_token_ids, &mut subtree_bytes, prefix_len, child);
        }

        VocabPrefixTreeNode {
            token_id,
            has_token,
            prefix,
            children: children.into_boxed_slice(),
            reachable_token_ids,
            subtree_bytes,
        }
    }

    /// Return the tree's token IDs whose positive proper prefix contains any
    /// byte in `bytes`. The last byte alone is never a witness. Result ordering
    /// is unspecified; byte-spelling aliases belong to the caller's ID map.
    ///
    /// This reads existing subtree metadata only. `work_limit` bounds examined
    /// nodes and emitted IDs; exhaustion returns None, NEVER a partial result.
    /// This permits callers to retain a direct-scan fallback for dense queries.
    pub fn token_ids_with_proper_prefix_byte(
        &self, bytes: &[u64; 4], mut work_limit: usize,
    ) -> Option<Vec<usize>> {
        if bytes.iter().all(|&word| word == 0) { return Some(Vec::new()); }
        let intersects = |observed: &[u64; 4]| {
            observed.iter().zip(bytes).any(|(&left, &right)| left & right != 0)
        };
        let contains = |byte: u8| bytes[byte as usize >> 6] & (1u64 << (byte & 63)) != 0;
        // Plan disjoint matching subtrees using their cached cardinalities
        // before materializing ANY IDs. A broad query can then decline without
        // allocating/enumerating a large partial answer that the caller discards.
        let mut subtrees = Vec::new();
        let mut token_count = 0usize;
        let mut pending = vec![&self.root];
        while let Some(node) = pending.pop() {
            work_limit = work_limit.checked_sub(1)?;
            // No earlier edge had a matching byte, otherwise that complete
            // subtree was emitted already. Prefix bytes are not double-scanned.
            if !intersects(node.subtree_bytes()) { continue; }
            for (edge, child) in node.iter_children() {
                work_limit = work_limit.checked_sub(1)?;
                if let Some(offset) = edge.iter().position(|&byte| contains(byte)) {
                    let exclude_endpoint = offset + 1 == edge.len() && child.has_token();
                    // An edge-end witness is a proper prefix for STRICT
                    // descendants, but not the token ending at that node.
                    let count = usize::try_from(child.reachable_token_ids().len()).ok()?
                        .checked_sub(usize::from(exclude_endpoint))?;
                    work_limit = work_limit.checked_sub(count)?;
                    token_count = token_count.checked_add(count)?;
                    if count != 0 {
                        subtrees.push((child, exclude_endpoint));
                    }
                } else if intersects(child.subtree_bytes()) {
                    pending.push(child);
                }
            }
        }
        let mut ids = Vec::with_capacity(token_count);
        for (child, exclude_endpoint) in subtrees {
            ids.extend(child.reachable_token_ids().iter().filter(|&id|
                !exclude_endpoint || id != child.token_id()));
        }
        debug_assert_eq!(ids.len(), token_count);
        Some(ids)
    }

    #[inline]
    pub fn find_token(&self, bytes: &[u8]) -> Option<usize> {
        if bytes.is_empty() {
            return self.has_empty_string_token.then_some(self.root.token_id);
        }

        let mut current = &self.root;
        let mut remaining = bytes;

        loop {
            let (child, next_remaining) = next_matching_child(current, remaining)?;
            remaining = next_remaining;
            current = child;

            if remaining.is_empty() {
                return current.has_token.then_some(current.token_id);
            }
        }
    }

    #[inline]
    pub fn find_longest_prefix_token<'s>(&'s self, bytes: &[u8]) -> Option<(usize, &'s [u8])> {
        let mut best = None;

        let mut current = &self.root;
        let mut remaining = bytes;

        if self.has_empty_string_token {
            best = Some((self.root.token_id, self.root.prefix()));
        }

        while !remaining.is_empty() {
            let Some((child, next_remaining)) = next_matching_child(current, remaining) else {
                break;
            };

            remaining = next_remaining;
            current = child;

            if current.has_token {
                best = Some((current.token_id, current.prefix()));
            }
        }

        best
    }

    #[inline]
    pub fn has_empty_string_token(&self) -> bool {
        self.has_empty_string_token
    }

    #[inline]
    pub fn root_children(&self) -> VocabPrefixTreeChildIter<'_> {
        self.root.iter_children()
    }

    #[inline]
    pub fn max_token_id(&self) -> usize {
        self.max_token_id
    }
}

impl Default for VocabPrefixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl Eq for VocabPrefixTreeNode {}

#[cfg(test)]
mod proper_prefix_byte_tests {
    use super::*;
    use std::collections::BTreeSet;

    fn mask(bytes: impl IntoIterator<Item=u8>) -> [u64; 4] {
        let mut mask = [0;4];
        for byte in bytes { mask[byte as usize >> 6] |= 1u64 << (byte & 63); }
        mask
    }

    fn reference(entries: &[(usize,Vec<u8>)], query: &[u64;4]) -> Vec<usize> {
        let mut ids = entries.iter().filter_map(|(id, bytes)| {
            bytes.get(..bytes.len().saturating_sub(1)).unwrap().iter()
                .any(|&byte| query[byte as usize >> 6] & (1u64 << (byte & 63)) != 0)
                .then_some(*id)
        }).collect::<Vec<_>>();
        ids.sort_unstable(); ids
    }

    #[test]
    fn proper_prefix_byte_query_respects_endpoints_compressed_edges_and_budget() {
        let entries = [b"".as_slice(),b"b",b"ab",b"abc",b"abcd",b"abef",b"bc",
            b"zzzzzzzzzzb",b"zzzzzzzzzzbx",b"\xff",b"\xff\x00",b"x\xff",b"x\xffy"]
            .into_iter().enumerate().map(|(i,b)| (i*19+7,b.to_vec())).collect::<Vec<_>>();
        let tree = VocabPrefixTree::build_owned(entries.clone());
        for query in [mask([b'a']),mask([b'b']),mask([b'c']),mask([255]),mask([0]),mask([255,b'b']),[u64::MAX;4],[0;4]] {
            let expected=reference(&entries,&query);
            let mut actual=tree.token_ids_with_proper_prefix_byte(&query,usize::MAX).unwrap();
            actual.sort_unstable(); assert_eq!(actual,expected);
            for budget in 0..=64 {
                if let Some(mut actual)=tree.token_ids_with_proper_prefix_byte(&query,budget) {
                    actual.sort_unstable(); assert_eq!(actual,expected,"budget must never truncate the answer");
                }
            }
        }
        assert!(tree.token_ids_with_proper_prefix_byte(&mask([b'b']),0).is_none());
        assert_eq!(tree.token_ids_with_proper_prefix_byte(&[0;4],0),Some(Vec::new()));
    }

    #[test]
    fn proper_prefix_byte_cardinality_budget_excludes_only_the_matching_endpoint() {
        let entries = vec![(7,b"a".to_vec()),(19,b"aa".to_vec()),(41,b"ab".to_vec())];
        let tree = VocabPrefixTree::build_owned(entries.clone());
        // One root visit, one matching compressed edge and two strict
        // descendants. The endpoint "a" is not an extra emitted ID.
        assert_eq!(tree.token_ids_with_proper_prefix_byte(&mask([b'a']),3),None);
        let mut ids = tree.token_ids_with_proper_prefix_byte(&mask([b'a']),4).unwrap();
        ids.sort_unstable(); assert_eq!(ids,vec![19,41]);
        let leaf = VocabPrefixTree::build_owned(vec![(99,b"za".to_vec())]);
        assert_eq!(leaf.token_ids_with_proper_prefix_byte(&mask([b'a']),2),Some(Vec::new()));
        assert_eq!(leaf.token_ids_with_proper_prefix_byte(&mask([b'z']),2),None);
        assert_eq!(leaf.token_ids_with_proper_prefix_byte(&mask([b'z']),3),Some(vec![99]));
    }

    #[test]
    fn proper_prefix_byte_dense_query_declines_without_returning_a_partial_subtree() {
        let entries = (0..8192usize).map(|id| {
            let mut bytes=vec![b'a'+u8::from(id>=4096)];
            bytes.extend_from_slice(&(id as u32).to_le_bytes());
            (id*3+11,bytes)
        }).collect::<Vec<_>>();
        let tree=VocabPrefixTree::build_owned(entries.clone());
        let query=mask([b'a',b'b']);
        // Either first subtree fits alone, but their union does not. The
        // complete plan must succeed before any partial answer is returned.
        assert!(tree.token_ids_with_proper_prefix_byte(&query,5000).is_none());
        let mut ids=tree.token_ids_with_proper_prefix_byte(&query,8195).unwrap();
        ids.sort_unstable(); assert_eq!(ids,reference(&entries,&query));
        assert!(tree.token_ids_with_proper_prefix_byte(&query,8194).is_none());
    }

    #[test]
    fn proper_prefix_byte_query_matches_independent_all_byte_and_random_set_scans() {
        fn next(state: &mut u64)->u64 {
            *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17; *state
        }
        let mut random=0x109174_b0_0d_u64;
        for _case in 0..96 {
            let mut words=BTreeSet::new(); words.insert(Vec::new());
            for _ in 0..64 {
                let len=(next(&mut random)%13) as usize;
                words.insert((0..len).map(|_| next(&mut random) as u8).collect::<Vec<_>>());
            }
            let entries=words.into_iter().enumerate().map(|(i,b)|(i*31+11,b)).collect::<Vec<_>>();
            let tree=VocabPrefixTree::build_owned(entries.clone());
            for byte in 0..=255u8 {
                let query=mask([byte]); let expected=reference(&entries,&query);
                let mut actual=tree.token_ids_with_proper_prefix_byte(&query,usize::MAX).unwrap();
                actual.sort_unstable(); assert_eq!(actual,expected);
            }
            for _ in 0..128 {
                let query=[next(&mut random),next(&mut random),next(&mut random),next(&mut random)];
                let expected=reference(&entries,&query);
                let mut actual=tree.token_ids_with_proper_prefix_byte(&query,usize::MAX).unwrap();
                actual.sort_unstable(); assert_eq!(actual,expected);
            }
        }
    }
}
