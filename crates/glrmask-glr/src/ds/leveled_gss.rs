use im::{HashMap as IHashMap, OrdMap};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use smallvec::{SmallVec, smallvec};
use super::stack_vecs::dispatch::DynStackVec;
use std::collections::{HashMap as StdHashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

/// Type alias for segment values. Set `STACKVEC` before process startup:
///   normal builds: `vec` (default), `arc`
type SV<T> = DynStackVec<T>;

pub trait Merge: Clone {
    fn merge(&self, other: &Self) -> Self;

    fn subsumes(&self, _other: &Self) -> bool {
        false
    }
}

impl Merge for () {
    fn merge(&self, _other: &Self) -> Self {}

    fn subsumes(&self, _other: &Self) -> bool {
        true
    }
}

impl Merge for glrmask_weight::Weight {
    fn merge(&self, other: &Self) -> Self {
        self.union(other)
    }
}

/// A map optimized for small sizes (≤4 entries). Uses inline SmallVec storage
/// for small maps and falls back to im::HashMap for larger ones.
/// Drop-in replacement for im::HashMap in GSS children maps.
#[derive(Clone, PartialEq, Eq)]
enum CompactMap<K: Clone + Eq + Hash, V: Clone> {
    Inline(SmallVec<[(K, V); 4]>),
    Large(IHashMap<K, V>),
}

impl<K: Clone + Eq + Hash, V: Clone> CompactMap<K, V> {
    #[inline(always)]
    fn new() -> Self {
        CompactMap::Inline(SmallVec::new())
    }

    #[inline(always)]
    fn unit(key: K, value: V) -> Self {
        let mut sv = SmallVec::new();
        sv.push((key, value));
        CompactMap::Inline(sv)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            CompactMap::Inline(sv) => sv.len(),
            CompactMap::Large(m) => m.len(),
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        match self {
            CompactMap::Inline(sv) => sv.is_empty(),
            CompactMap::Large(m) => m.is_empty(),
        }
    }

    #[inline(always)]
    fn get(&self, key: &K) -> Option<&V> {
        match self {
            CompactMap::Inline(sv) => sv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            CompactMap::Large(m) => m.get(key),
        }
    }

    #[inline(always)]
    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self {
            CompactMap::Inline(sv) => sv.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            CompactMap::Large(m) => m.get_mut(key),
        }
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self {
            CompactMap::Inline(sv) => {
                for entry in sv.iter_mut() {
                    if entry.0 == key {
                        let old = std::mem::replace(&mut entry.1, value);
                        return Some(old);
                    }
                }
                if sv.len() < 4 {
                    sv.push((key, value));
                    None
                } else {
                    // Promote to Large
                    let mut m = IHashMap::new();
                    for (k, v) in sv.drain(..) {
                        m.insert(k, v);
                    }
                    let result = m.insert(key, value);
                    *self = CompactMap::Large(m);
                    result
                }
            }
            CompactMap::Large(m) => m.insert(key, value),
        }
    }

    #[inline(always)]
    fn contains_key(&self, key: &K) -> bool {
        match self {
            CompactMap::Inline(sv) => sv.iter().any(|(k, _)| k == key),
            CompactMap::Large(m) => m.contains_key(key),
        }
    }

    fn keys(&self) -> CompactMapKeys<'_, K, V> {
        match self {
            CompactMap::Inline(sv) => CompactMapKeys::Inline(sv.iter()),
            CompactMap::Large(m) => CompactMapKeys::Large(m.keys()),
        }
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CompactMap::Large(a), CompactMap::Large(b)) => a.ptr_eq(b),
            _ => false,
        }
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        match self {
            CompactMap::Inline(sv) => {
                if let Some(pos) = sv.iter().position(|(k, _)| k == key) {
                    Some(sv.swap_remove(pos).1)
                } else {
                    None
                }
            }
            CompactMap::Large(m) => m.remove(key),
        }
    }

    fn values(&self) -> CompactMapValues<'_, K, V> {
        match self {
            CompactMap::Inline(sv) => CompactMapValues::Inline(sv.iter()),
            CompactMap::Large(m) => CompactMapValues::Large(m.values()),
        }
    }

    fn iter(&self) -> CompactMapIter<'_, K, V> {
        match self {
            CompactMap::Inline(sv) => CompactMapIter::Inline(sv.iter()),
            CompactMap::Large(m) => CompactMapIter::Large(m.iter()),
        }
    }
}

enum CompactMapKeys<'a, K, V> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(im::hashmap::Keys<'a, K, V>),
}

impl<'a, K: Clone, V> Iterator for CompactMapKeys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactMapKeys::Inline(it) => it.next().map(|(k, _)| k),
            CompactMapKeys::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactMapKeys::Inline(it) => it.size_hint(),
            CompactMapKeys::Large(it) => it.size_hint(),
        }
    }
}

enum CompactMapValues<'a, K, V> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(im::hashmap::Values<'a, K, V>),
}

impl<'a, K, V> Iterator for CompactMapValues<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactMapValues::Inline(it) => it.next().map(|(_, v)| v),
            CompactMapValues::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactMapValues::Inline(it) => it.size_hint(),
            CompactMapValues::Large(it) => it.size_hint(),
        }
    }
}

enum CompactMapIter<'a, K, V> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(im::hashmap::Iter<'a, K, V>),
}

impl<'a, K: Clone, V: Clone> Iterator for CompactMapIter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactMapIter::Inline(it) => it.next().map(|(k, v)| (k, v)),
            CompactMapIter::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactMapIter::Inline(it) => it.size_hint(),
            CompactMapIter::Large(it) => it.size_hint(),
        }
    }
}

impl<'a, K: Clone + Eq + Hash, V: Clone> IntoIterator for &'a CompactMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = CompactMapIter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A map optimized for small sizes (≤2 entries) keyed by u32 (depth).
/// Replaces `OrdMap<u32, Arc<N>>` in the GSS children maps.
/// Typical GSS paths have 1 entry; this avoids B-tree overhead.
#[derive(Clone, PartialEq, Eq)]
enum CompactOrdMap<V: Clone> {
    Inline(SmallVec<[(u32, V); 2]>),
    Large(OrdMap<u32, V>),
}

impl<V: Clone> CompactOrdMap<V> {
    #[inline(always)]
    fn new() -> Self {
        CompactOrdMap::Inline(SmallVec::new())
    }

    #[inline(always)]
    fn unit(key: u32, value: V) -> Self {
        let mut sv = SmallVec::new();
        sv.push((key, value));
        CompactOrdMap::Inline(sv)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            CompactOrdMap::Inline(sv) => sv.len(),
            CompactOrdMap::Large(m) => m.len(),
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        match self {
            CompactOrdMap::Inline(sv) => sv.is_empty(),
            CompactOrdMap::Large(m) => m.is_empty(),
        }
    }

    #[inline(always)]
    fn get(&self, key: &u32) -> Option<&V> {
        match self {
            CompactOrdMap::Inline(sv) => sv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            CompactOrdMap::Large(m) => m.get(key),
        }
    }

    fn insert(&mut self, key: u32, value: V) -> Option<V> {
        match self {
            CompactOrdMap::Inline(sv) => {
                for entry in sv.iter_mut() {
                    if entry.0 == key {
                        let old = std::mem::replace(&mut entry.1, value);
                        return Some(old);
                    }
                }
                if sv.len() < 2 {
                    sv.push((key, value));
                    None
                } else {
                    // Promote to Large
                    let mut m = OrdMap::new();
                    for (k, v) in sv.drain(..) {
                        m.insert(k, v);
                    }
                    let result = m.insert(key, value);
                    *self = CompactOrdMap::Large(m);
                    result
                }
            }
            CompactOrdMap::Large(m) => m.insert(key, value),
        }
    }

    fn keys(&self) -> CompactOrdMapKeys<'_, V> {
        match self {
            CompactOrdMap::Inline(sv) => CompactOrdMapKeys::Inline(sv.iter()),
            CompactOrdMap::Large(m) => CompactOrdMapKeys::Large(m.keys()),
        }
    }

    fn get_max(&self) -> Option<(&u32, &V)> {
        match self {
            CompactOrdMap::Inline(sv) => {
                sv.iter().max_by_key(|(k, _)| *k).map(|(k, v)| (k, v))
            }
            CompactOrdMap::Large(m) => m.get_max().map(|(k, v)| (k, v)),
        }
    }

    fn iter(&self) -> CompactOrdMapIter<'_, V> {
        match self {
            CompactOrdMap::Inline(sv) => CompactOrdMapIter::Inline(sv.iter()),
            CompactOrdMap::Large(m) => CompactOrdMapIter::Large(m.iter()),
        }
    }

    fn values(&self) -> CompactOrdMapValues<'_, V> {
        match self {
            CompactOrdMap::Inline(sv) => CompactOrdMapValues::Inline(sv.iter()),
            CompactOrdMap::Large(m) => CompactOrdMapValues::Large(m.values()),
        }
    }
}

enum CompactOrdMapIter<'a, V> {
    Inline(std::slice::Iter<'a, (u32, V)>),
    Large(im::ordmap::Iter<'a, u32, V>),
}

impl<'a, V: Clone> Iterator for CompactOrdMapIter<'a, V> {
    type Item = (&'a u32, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactOrdMapIter::Inline(it) => it.next().map(|(k, v)| (k, v)),
            CompactOrdMapIter::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactOrdMapIter::Inline(it) => it.size_hint(),
            CompactOrdMapIter::Large(it) => it.size_hint(),
        }
    }
}

enum CompactOrdMapValues<'a, V> {
    Inline(std::slice::Iter<'a, (u32, V)>),
    Large(im::ordmap::Values<'a, u32, V>),
}

impl<'a, V: Clone> Iterator for CompactOrdMapValues<'a, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactOrdMapValues::Inline(it) => it.next().map(|(_, v)| v),
            CompactOrdMapValues::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactOrdMapValues::Inline(it) => it.size_hint(),
            CompactOrdMapValues::Large(it) => it.size_hint(),
        }
    }
}

impl<V: Clone> std::iter::FromIterator<(u32, V)> for CompactOrdMap<V> {
    fn from_iter<I: IntoIterator<Item = (u32, V)>>(iter: I) -> Self {
        let mut map = CompactOrdMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

enum CompactOrdMapKeys<'a, V> {
    Inline(std::slice::Iter<'a, (u32, V)>),
    Large(im::ordmap::Keys<'a, u32, V>),
}

impl<'a, V> Iterator for CompactOrdMapKeys<'a, V> {
    type Item = &'a u32;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompactOrdMapKeys::Inline(it) => it.next().map(|(k, _)| k),
            CompactOrdMapKeys::Large(it) => it.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            CompactOrdMapKeys::Inline(it) => it.size_hint(),
            CompactOrdMapKeys::Large(it) => it.size_hint(),
        }
    }
}

impl<'a, V: Clone> IntoIterator for &'a CompactOrdMap<V> {
    type Item = (&'a u32, &'a V);
    type IntoIter = CompactOrdMapIter<'a, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

type Children<T, N> = CompactMap<T, CompactOrdMap<Arc<N>>>;

/// Linear segment of the stack: multiple values packed into one node.
/// `values[0]` is the deepest value (closest to `next`),
/// `values[last]` is the shallowest (top of stack).
/// Intermediate levels (all except the top) are guaranteed to have empty=false.
/// Values are stored in SV<T> (type-aliased segment vector).
/// Segments are always non-accepting (empty is implicitly false).
struct Segment<T: Clone + Eq + Hash> {
    values: SV<T>,
    next: Arc<Lower<T>>,
    max_depth: u32,
    segments_len: usize,
    rest: OnceLock<Arc<Lower<T>>>,
}

impl<T: Clone + Eq + Hash> Clone for Segment<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            next: self.next.clone(),
            max_depth: self.max_depth,
            segments_len: self.segments_len,
            rest: OnceLock::new(),
        }
    }
}

impl<T: Clone + Eq + Hash> PartialEq for Segment<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
            && self.next == other.next
            && self.max_depth == other.max_depth
            && self.segments_len == other.segments_len
    }
}

impl<T: Clone + Eq + Hash> Eq for Segment<T> {}

#[derive(Clone, PartialEq, Eq)]
enum Lower<T: Clone + Eq + Hash> {
    General {
        children: Children<T, Lower<T>>,
        empty: bool,
        max_depth: u32,
    },
    Segment(Arc<Segment<T>>),
}

/// Get a stable identity for a Lower node wrapped in Arc.
/// For Segment nodes, uses the inner Arc<Segment> pointer (since the outer Arc<Lower>
/// may be ephemeral when constructed from segment_rest_arc or children()).
/// For General nodes, uses the outer Arc<Lower> pointer directly.
#[inline]
fn lower_node_id<T: Clone + Eq + Hash>(node: &Arc<Lower<T>>) -> usize {
    match &**node {
        Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
        _ => Arc::as_ptr(node) as usize,
    }
}

impl<T: Clone + Eq + Hash> Lower<T> {
    #[inline(always)]
    fn empty(&self) -> bool {
        match self {
            Lower::General { empty, .. } => *empty,
            Lower::Segment(_) => false,
        }
    }

    #[inline(always)]
    fn max_depth(&self) -> u32 {
        match self {
            Lower::General { max_depth, .. } => *max_depth,
            Lower::Segment(seg) => seg.max_depth,
        }
    }

    #[inline(always)]
    fn segments_len(&self) -> usize {
        match self {
            Lower::Segment(seg) => seg.segments_len,
            Lower::General { .. } => 0,
        }
    }

    /// If this node is a deterministic chain link (Segment or single-child
    /// General with one edge), return the next node below it plus the number
    /// of stack values represented at this node.
    #[inline]
    fn chain_step(&self) -> Option<(&Arc<Lower<T>>, usize)> {
        match self {
            Lower::Segment(seg) => Some((&seg.next, seg.values.len())),
            Lower::General { children, .. } if children.len() == 1 => {
                let ordmap = children.values().next().unwrap();
                if ordmap.len() == 1 {
                    Some((ordmap.iter().next().unwrap().1, 1))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Append the values represented by this deterministic chain node,
    /// top-first, into `out`.
    #[inline]
    fn append_chain_values_top_first(&self, out: &mut SmallVec<[T; 16]>) {
        match self {
            Lower::Segment(seg) => {
                for value in seg.values.iter().rev() {
                    out.push(value.clone());
                }
            }
            Lower::General { children, .. } => {
                out.push(children.keys().next().unwrap().clone());
            }
        }
    }

    /// Get children as a general Children map.
    /// For Segment, constructs a map with the top value → rest-of-segment.
    fn children(&self) -> Children<T, Lower<T>> {
        match self {
            Lower::General { children, .. } => children.clone(),
            Lower::Segment(seg) => {
                let top_value = seg.values.last().unwrap().clone();
                let child = self.segment_rest_arc();
                CompactMap::unit(top_value, CompactOrdMap::unit(child.max_depth(), child))
            }
        }
    }

    /// Consume self and return (children, empty, max_depth).
    fn into_parts(self) -> (Children<T, Lower<T>>, bool, u32) {
        match self {
            Lower::General { children, empty, max_depth } => (children, empty, max_depth),
            Lower::Segment(seg) => {
                let top_value = seg.values.last().unwrap().clone();
                let seg = Arc::try_unwrap(seg).unwrap_or_else(|arc| (*arc).clone());
                let max_depth = seg.max_depth;
                let child = if seg.values.len() == 1 {
                    seg.next
                } else {
                    // Pop the top value by taking all-but-last. O(1) for view-based types.
                    // Don't call new_segment() — no need to merge, just shrink the segment.
                    let rest_values = seg.values.take(seg.values.len() - 1);
                    let child_max_depth = seg.max_depth - 1;
                    let segments_len = seg.segments_len - 1;
                    Arc::new(Lower::Segment(Arc::new(Segment {
                        values: rest_values,
                        next: seg.next,
                        max_depth: child_max_depth,
                        segments_len,
                        rest: OnceLock::new(),
                    })))
                };
                let children = CompactMap::unit(top_value, CompactOrdMap::unit(child.max_depth(), child));
                (children, false, max_depth)
            }
        }
    }

    /// Number of distinct child keys (at the top level).
    #[inline(always)]
    fn children_len(&self) -> usize {
        match self {
            Lower::General { children, .. } => children.len(),
            Lower::Segment(_) => 1,
        }
    }

    /// Whether there are no children.
    #[inline(always)]
    fn children_is_empty(&self) -> bool {
        match self {
            Lower::General { children, .. } => children.is_empty(),
            Lower::Segment(_) => false,
        }
    }

    /// Check if the top-level children contains a key.
    fn children_contains_key(&self, key: &T) -> bool {
        match self {
            Lower::General { children, .. } => children.contains_key(key),
            Lower::Segment(seg) => seg.values.last().unwrap() == key,
        }
    }

    /// Ensure this Lower is in General form, converting from Segment if necessary.
    fn ensure_general(&mut self) {
        if let Lower::Segment(_) = self {
            let old = std::mem::replace(self, Lower::General {
                children: CompactMap::new(),
                empty: false,
                max_depth: 0,
            });
            let (children, empty, max_depth) = old.into_parts();
            *self = Lower::General { children, empty, max_depth };
        }
    }

    /// Returns true if this is a Segment variant.
    #[inline(always)]
    fn is_segment(&self) -> bool {
        matches!(self, Lower::Segment(_))
    }

    /// For Segment variant, get the shallowest (top) value by reference.
    /// Panics if called on General.
    #[inline(always)]
    fn segment_top_value(&self) -> &T {
        match self {
            Lower::Segment(seg) => seg.values.last().unwrap(),
            Lower::General { .. } => panic!("segment_top_value called on General"),
        }
    }

    /// For Segment variant, get the deep-end next pointer.
    /// Panics if called on General.
    #[inline(always)]
    fn segment_next(&self) -> &Arc<Lower<T>> {
        match self {
            Lower::Segment(seg) => &seg.next,
            Lower::General { .. } => panic!("segment_next called on General"),
        }
    }

    /// For Segment variant, get the values vector.
    /// Panics if called on General.
    #[inline(always)]
    fn segment_values(&self) -> &SV<T> {
        match self {
            Lower::Segment(seg) => &seg.values,
            Lower::General { .. } => panic!("segment_values called on General"),
        }
    }

    /// For Segment variant, return an Arc to the "rest" (everything below the top value).
    /// If len==1, wraps next in Arc. Otherwise creates a new shorter Segment.
    fn segment_rest_arc(&self) -> Arc<Lower<T>> {
        match self {
            Lower::Segment(seg) if seg.values.len() == 1 => seg.next.clone(),
            Lower::Segment(seg) => seg.rest.get_or_init(|| {
                let rest_values = seg.values.take(seg.values.len() - 1);
                // Don't merge with child; just create a shorter segment.
                let max_depth = seg.max_depth - 1;
                let segments_len = seg.segments_len - 1;
                Arc::new(Lower::Segment(Arc::new(Segment {
                    values: rest_values,
                    next: seg.next.clone(),
                    max_depth,
                    segments_len,
                    rest: OnceLock::new(),
                })))
            }).clone(),
            Lower::General { .. } => panic!("segment_rest_arc called on General"),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Interface<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Lower<T>>,
    acc: A,
}

#[derive(Clone, PartialEq, Eq)]
struct UpperBranch<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    children: Children<T, Upper<T, A>>,
    empty: Option<A>,
    max_depth: u32,
}

#[derive(Clone, PartialEq, Eq)]
enum Upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    Branch(Arc<UpperBranch<T, A>>),
    Interface(Arc<Interface<T, A>>),
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Upper<T, A> {
    fn max_depth(&self) -> u32 {
        match self {
            Upper::Branch(branch) => branch.max_depth,
            Upper::Interface(interface) => interface.inner.max_depth(),
        }
    }

    #[inline(always)]
    fn children_len(&self) -> usize {
        match self {
            Upper::Branch(branch) => branch.children.len(),
            Upper::Interface(interface) => interface.inner.children_len(),
        }
    }

    fn children_keys(&self) -> SmallVec<[T; 8]> {
        match self {
            Upper::Branch(branch) => branch.children.keys().cloned().collect(),
            Upper::Interface(interface) => match &*interface.inner {
                Lower::Segment(seg) => smallvec![seg.values.last().unwrap().clone()],
                Lower::General { children, .. } => children.keys().cloned().collect(),
            },
        }
    }

    fn single_child_key(&self) -> Option<T> {
        match self {
            Upper::Branch(branch) => {
                if branch.children.len() == 1 {
                    branch.children.keys().next().cloned()
                } else {
                    None
                }
            }
            Upper::Interface(interface) => {
                if interface.inner.children_len() == 1 {
                    match &*interface.inner {
                        Lower::Segment(seg) => Some(seg.values.last().unwrap().clone()),
                        Lower::General { children, .. } => children.keys().next().cloned(),
                    }
                } else {
                    None
                }
            }
        }
    }

    fn single_child_key_without_empty(&self) -> Option<T> {
        match self {
            Upper::Branch(branch) => {
                if branch.empty.is_none() && branch.children.len() == 1 {
                    branch.children.keys().next().cloned()
                } else {
                    None
                }
            }
            Upper::Interface(interface) => {
                if !interface.inner.empty() && interface.inner.children_len() == 1 {
                    match &*interface.inner {
                        Lower::Segment(seg) => Some(seg.values.last().unwrap().clone()),
                        Lower::General { children, .. } => children.keys().next().cloned(),
                    }
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LeveledGSSSummary {
    pub top_values_count: usize,
    pub upperbranch_nodes: usize,
    pub interface_nodes: usize,
    pub lower_nodes: usize,
    pub lower_general_nodes: usize,
    pub lower_segment_nodes: usize,
    pub total_unique_nodes: usize,
    pub total_edges: usize,
    pub accumulator_instances: usize,
    pub max_depth: u32,
}




fn merge_optional_acc<A: Merge + Clone>(a: &Option<A>, b: &Option<A>) -> Option<A> {
    match (a, b) {
        (None, Some(bv)) => Some(bv.clone()),
        (Some(av), None) => Some(av.clone()),
        (Some(av), Some(bv)) => Some(av.merge(bv)),
        (None, None) => None,
    }
}

fn max_depth_from_children<T, N, F>(children: &Children<T, N>, depth_of: F) -> u32
where
    T: Clone + Eq + Hash,
    F: Fn(&Arc<N>) -> u32,
{
    children
        .values()
        .flat_map(|kids| kids.values())
        .map(|c| depth_of(c))
        .max()
        .map_or(0, |d| d + 1)
}

fn merge_children<T, N, F>(c1: &Children<T, N>, c2: &Children<T, N>, merge_fn: F) -> Children<T, N>
where
    T: Clone + Eq + Hash,
    F: Fn(&Arc<N>, &Arc<N>) -> Arc<N>,
{
    if c1.ptr_eq(c2) {
        return c1.clone();
    }
    let mut merged = c1.clone();
    for (k, v2_map) in c2.iter() {
        if let Some(v1_map) = merged.get(k) {
            let mut new_map = v1_map.clone();
            for (depth, child2) in v2_map.iter() {
                if let Some(child1) = new_map.get(depth) {
                    let merged_child = merge_fn(child1, child2);
                    new_map.insert(*depth, merged_child);
                } else {
                    new_map.insert(*depth, child2.clone());
                }
            }
            merged.insert(k.clone(), new_map);
        } else {
            merged.insert(k.clone(), v2_map.clone());
        }
    }
    merged
}

fn find_equal_lower_child<T: Clone + Eq + Hash>(
    children: &Children<T, Lower<T>>,
    depth: u32,
    child: &Arc<Lower<T>>,
) -> Option<Arc<Lower<T>>> {
    for kids in children.values() {
        let Some(existing) = kids.get(&depth) else {
            continue;
        };
        if Arc::ptr_eq(existing, child) || **existing == **child {
            return Some(existing.clone());
        }
    }
    None
}

fn canonicalize_lower_children<T: Clone + Eq + Hash>(children: &mut Children<T, Lower<T>>) {
    if children.len() <= 1 {
        return;
    }
    let keys: Vec<T> = children.keys().cloned().collect();
    let mut seen: Vec<(u32, Arc<Lower<T>>)> = Vec::new();
    for key in keys {
        let entries: Vec<(u32, Arc<Lower<T>>)> = children
            .get(&key)
            .map(|kids| kids.iter().map(|(depth, child)| (*depth, child.clone())).collect())
            .unwrap_or_default();
        let Some(kids) = children.get_mut(&key) else {
            continue;
        };
        for (depth, child) in entries {
            let canonical = seen
                .iter()
                .find(|(seen_depth, seen_child)| {
                    *seen_depth == depth && (Arc::ptr_eq(seen_child, &child) || **seen_child == *child)
                })
                .map(|(_, seen_child)| seen_child.clone());
            match canonical {
                Some(existing) => {
                    if !Arc::ptr_eq(&existing, &child) {
                        kids.insert(depth, existing);
                    }
                }
                None => seen.push((depth, child)),
            }
        }
    }
}

fn insert_lower_child_shared<T: Clone + Eq + Hash>(
    children: &mut Children<T, Lower<T>>,
    key: T,
    depth: u32,
    child: Arc<Lower<T>>,
) {
    let child = find_equal_lower_child(children, depth, &child).unwrap_or(child);
    let merged = {
        let Some(ord_map) = children.get_mut(&key) else {
            children.insert(key, CompactOrdMap::unit(depth, child));
            return;
        };
        let Some(existing) = ord_map.get(&depth).cloned() else {
            ord_map.insert(depth, child);
            return;
        };
        merge_lower(&existing, &child)
    };
    let merged = find_equal_lower_child(children, merged.max_depth(), &merged).unwrap_or(merged);
    let Some(ord_map) = children.get_mut(&key) else {
        return;
    };
    ord_map.insert(depth, merged);
}

fn insert_upper_child_shared<T, A>(
    children: &mut Children<T, Upper<T, A>>,
    key: T,
    child: Arc<Upper<T, A>>,
) where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let depth = child.max_depth();
    let merged = {
        let Some(ord_map) = children.get_mut(&key) else {
            children.insert(key, CompactOrdMap::unit(depth, child));
            return;
        };
        let Some(existing) = ord_map.get(&depth).cloned() else {
            ord_map.insert(depth, child);
            return;
        };
        merge_upper(&existing, &child)
    };
    let Some(ord_map) = children.get_mut(&key) else {
        return;
    };
    ord_map.insert(depth, merged);
}

fn new_lower_precanonicalized<T: Clone + Eq + Hash>(
    children: Children<T, Lower<T>>,
    empty: bool,
) -> Arc<Lower<T>> {
    // Only compress to Segment when empty is false — Segments are non-accepting.
    if !empty && children.len() == 1 {
        let (key, ord_map) = children.iter().next().unwrap();
        if ord_map.len() == 1 {
            let (_, next) = ord_map.iter().next().unwrap();
            return new_segment(SV::unit(key.clone()), next.clone());
        }
    }
    let max_depth = max_depth_from_children(&children, |n: &Arc<Lower<T>>| n.max_depth());
    Arc::new(Lower::General {
        children,
        empty,
        max_depth,
    })
}

fn new_lower<T: Clone + Eq + Hash>(mut children: Children<T, Lower<T>>, empty: bool) -> Arc<Lower<T>> {
    // Use Segment variant when there's exactly one key with one depth entry.
    // NOTE: We do NOT pack into existing child Segments here. Packing only happens
    // in batch-construction paths (into_gss) so that incremental push/pop
    // preserves Arc sharing (the child Arc is reused on pop).
    // Share structurally equal child suffixes across different top values.
    // This keeps common-bottom, different-top frontiers compact even when they
    // are assembled incrementally rather than via `from_stacks`.
    canonicalize_lower_children(&mut children);

    // Only compress to Segment when empty is false — Segments are non-accepting.
    if !empty && children.len() == 1 {
        let (key, ord_map) = children.iter().next().unwrap();
        if ord_map.len() == 1 {
            let (_, next) = ord_map.iter().next().unwrap();
            let values = SV::unit(key.clone());
            return new_segment(values, next.clone());
        }
    }
    let max_depth = max_depth_from_children(&children, |n: &Arc<Lower<T>>| n.max_depth());
    Arc::new(Lower::General {
        children,
        empty,
        max_depth,
    })
}

fn new_segment<T: Clone + Eq + Hash>(values: SV<T>, next: Arc<Lower<T>>) -> Arc<Lower<T>> {
    // Merge with child segment if possible and fits.
    if let Lower::Segment(child_seg) = &*next {
        if let Some(merged) = child_seg.values.try_append(&values) {
            let max_depth = child_seg.next.max_depth() + merged.len() as u32;
            let segments_len = merged.len() + child_seg.next.segments_len();
            return Arc::new(Lower::Segment(Arc::new(Segment {
                values: merged,
                next: child_seg.next.clone(),
                max_depth,
                segments_len,
                rest: OnceLock::new(),
            })));
        }
    }
    let max_depth = next.max_depth() + values.len() as u32;
    let segments_len = values.len() + next.segments_len();
    Arc::new(Lower::Segment(Arc::new(Segment {
        values,
        next,
        max_depth,
        segments_len,
        rest: OnceLock::new(),
    })))
}

fn new_interface<T, A>(inner: Arc<Lower<T>>, acc: A) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    Arc::new(Upper::Interface(Arc::new(Interface { inner, acc })))
}

fn new_branch<T, A>(
    children: Children<T, Upper<T, A>>,
    empty: Option<A>,
) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let max_depth = max_depth_from_children(&children, |n: &Arc<Upper<T, A>>| n.max_depth());
    Arc::new(Upper::Branch(Arc::new(UpperBranch {
        children,
        empty,
        max_depth,
    })))
}

fn empty_upper_inner<T, A>() -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    new_branch(CompactMap::new(), None)
}



fn truncate_lower<T: Clone + Eq + Hash>(
    node: &Arc<Lower<T>>,
    current_depth: isize,
    max_len: isize,
    memo: &mut StdHashMap<usize, Option<Arc<Lower<T>>>>,
) -> Option<Arc<Lower<T>>> {
    let ptr = lower_node_id(node);
    if let Some(cached) = memo.get(&ptr) {
        return cached.clone();
    }

    if current_depth == max_len {
        let res = if node.empty() || !node.children_is_empty() {
            Some(new_lower(CompactMap::new(), true))
        } else {
            None
        };
        memo.insert(ptr, res.clone());
        return res;
    }

    let mut new_children: Children<T, Lower<T>> = CompactMap::new();
    let mut children_identical = true;

    match &**node {
        Lower::Segment(seg) => {
            let value = seg.values.last().unwrap();
            let rest = node.segment_rest_arc();
            if let Some(new_child) = truncate_lower(&rest, current_depth + 1, max_len, memo) {
                if !Arc::ptr_eq(&rest, &new_child) || rest.max_depth() != new_child.max_depth() {
                    children_identical = false;
                }
                new_children.insert(value.clone(), CompactOrdMap::unit(new_child.max_depth(), new_child));
            } else {
                children_identical = false;
            }
        }
        Lower::General { children: node_children, .. } => {
            for (v, kids) in node_children.iter() {
                let mut new_kids_map = CompactOrdMap::new();
                let mut kids_identical = true;
                for (depth, child) in kids.iter() {
                    if let Some(new_child) = truncate_lower(child, current_depth + 1, max_len, memo) {
                        if !Arc::ptr_eq(child, &new_child) || *depth != new_child.max_depth() {
                            kids_identical = false;
                        }
                        new_kids_map.insert(new_child.max_depth(), new_child);
                    } else {
                        kids_identical = false;
                    }
                }
                if !new_kids_map.is_empty() {
                    new_children.insert(v.clone(), new_kids_map);
                } else {
                    children_identical = false;
                }
                children_identical &= kids_identical;
            }
        }
    }

    if node.empty() && children_identical {
        memo.insert(ptr, Some(node.clone()));
        return Some(node.clone());
    }

    let res = if !node.empty() && new_children.is_empty() {
        None
    } else {
        Some(new_lower(new_children, node.empty()))
    };
    memo.insert(ptr, res.clone());
    res
}

fn truncate_upper<T, A>(
    node: &Arc<Upper<T, A>>,
    current_depth: isize,
    max_len: isize,
    memo_upper: &mut StdHashMap<usize, Option<Arc<Upper<T, A>>>>,
    memo_lower: &mut StdHashMap<usize, Option<Arc<Lower<T>>>>,
) -> Option<Arc<Upper<T, A>>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let ptr = Arc::as_ptr(node) as usize;
    if let Some(cached) = memo_upper.get(&ptr) {
        return cached.clone();
    }

    if current_depth == max_len {
        let sub_gss = LeveledGSS {
            inner: node.clone(),
        };
        let res = if let Some(acc) = sub_gss.reduce_acc() {
            let terminal_lower = new_lower(CompactMap::new(), true);
            Some(new_interface(terminal_lower, acc))
        } else {
            None
        };
        memo_upper.insert(ptr, res.clone());
        return res;
    }

    let res = match &**node {
        Upper::Branch(b) => {
            let new_empty = b.empty.clone();
            let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
            let mut children_identical = true;

            for (v, kids) in b.children.iter() {
                let mut new_kids_map = CompactOrdMap::new();
                let mut kids_identical = true;
                for (depth, child) in kids.iter() {
                    if let Some(new_child) =
                        truncate_upper(child, current_depth + 1, max_len, memo_upper, memo_lower)
                    {
                        if !Arc::ptr_eq(child, &new_child) || *depth != new_child.max_depth() {
                            kids_identical = false;
                        }
                        new_kids_map.insert(new_child.max_depth(), new_child);
                    } else {
                        kids_identical = false;
                    }
                }
                if !new_kids_map.is_empty() {
                    new_children.insert(v.clone(), new_kids_map);
                } else {
                    children_identical = false;
                }
                children_identical &= kids_identical;
            }

            if new_empty == b.empty && children_identical {
                memo_upper.insert(ptr, Some(node.clone()));
                return Some(node.clone());
            }

            if new_children.is_empty() && new_empty.is_none() {
                None
            } else {
                Some(try_promote(&new_branch(new_children, new_empty)))
            }
        }
        Upper::Interface(i) => {
            if let Some(new_inner) = truncate_lower(&i.inner, current_depth, max_len, memo_lower) {
                if Arc::ptr_eq(&i.inner, &new_inner) {
                    Some(node.clone())
                } else {
                    Some(new_interface(new_inner, i.acc.clone()))
                }
            } else {
                None
            }
        }
    };

    memo_upper.insert(ptr, res.clone());
    res
}




fn merge_lower<T: Clone + Eq + Hash>(l1: &Arc<Lower<T>>, l2: &Arc<Lower<T>>) -> Arc<Lower<T>> {
    if Arc::ptr_eq(l1, l2) {
        return l1.clone();
    }

    let new_empty = l1.empty() || l2.empty();
    let merged_children = match (&**l1, &**l2) {
        (Lower::Segment(s1), Lower::Segment(s2)) => {
            // Fast path: if both segments share the same tail and have identical
            // values, they are structurally identical — skip deep recursion.
            if Arc::ptr_eq(&s1.next, &s2.next) && s1.values == s2.values {
                return l1.clone();
            }

            let v1 = l1.segment_top_value();
            let v2 = l2.segment_top_value();
            let r1 = l1.segment_rest_arc();
            let r2 = l2.segment_rest_arc();
            if v1 == v2 {
                let merged_next = merge_lower(&r1, &r2);
                CompactMap::unit(v1.clone(), CompactOrdMap::unit(merged_next.max_depth(), merged_next))
            } else {
                let mut c = CompactMap::unit(v1.clone(), CompactOrdMap::unit(r1.max_depth(), r1));
                c.insert(v2.clone(), CompactOrdMap::unit(r2.max_depth(), r2));
                c
            }
        }
        (Lower::Segment(_), Lower::General { children, .. }) |
        (Lower::General { children, .. }, Lower::Segment(_)) => {
            let seg = if l1.is_segment() { l1 } else { l2 };
            let value = seg.segment_top_value();
            let rest = seg.segment_rest_arc();
            let mut merged = children.clone();
            let seg_kids = CompactOrdMap::unit(rest.max_depth(), rest.clone());
            if let Some(existing) = merged.get(value) {
                let mut new_map = existing.clone();
                let depth = rest.max_depth();
                if let Some(existing_child) = new_map.get(&depth) {
                    new_map.insert(depth, merge_lower(existing_child, &rest));
                } else {
                    new_map.insert(depth, rest);
                }
                merged.insert(value.clone(), new_map);
            } else {
                merged.insert(value.clone(), seg_kids);
            }
            merged
        }
        (Lower::General { children: c1, .. }, Lower::General { children: c2, .. }) => {
            merge_children(c1, c2, |a, b| merge_lower(a, b))
        }
    };
    new_lower(merged_children, new_empty)
}

fn interface_to_upperbranch<T, A>(it: &Arc<Interface<T, A>>) -> Arc<UpperBranch<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let mut children: Children<T, Upper<T, A>> = CompactMap::new();
    match &*it.inner {
        Lower::Segment(seg) => {
            let value = seg.values.last().unwrap();
            let rest = it.inner.segment_rest_arc();
            let ci = new_interface(rest, it.acc.clone());
            let v_map = CompactOrdMap::unit(ci.max_depth(), ci);
            children.insert(value.clone(), v_map);
        }
        Lower::General { children: inner_children, .. } => {
            for (v, kids) in inner_children.iter() {
                let mut v_map: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
                for lchild in kids.values() {
                    let ci = new_interface(lchild.clone(), it.acc.clone());
                    v_map.insert(ci.max_depth(), ci);
                }
                if !v_map.is_empty() {
                    children.insert(v.clone(), v_map);
                }
            }
        }
    }

    let new_empty = if it.inner.empty() {
        Some(it.acc.clone())
    } else {
        None
    };

    let max_depth = max_depth_from_children(&children, |n: &Arc<Upper<T, A>>| n.max_depth());
    Arc::new(UpperBranch {
        children,
        empty: new_empty,
        max_depth,
    })
}

fn nonempty_deterministic_top_step<T>(lower: &Arc<Lower<T>>) -> Option<(T, Arc<Lower<T>>)> 
where
    T: Clone + Eq + Hash,
{
    match &**lower {
        Lower::Segment(seg) => {
            let value = seg.values.last()?.clone();
            Some((value, lower.segment_rest_arc()))
        }
        Lower::General { children, empty, .. } if !*empty && children.len() == 1 => {
            let (value, kids) = children.iter().next()?;
            if kids.len() != 1 {
                return None;
            }
            let child = kids.values().next()?.clone();
            Some((value.clone(), child))
        }
        _ => None,
    }
}

fn shared_nonempty_deterministic_prefix<T>(
    left: &Arc<Lower<T>>,
    right: &Arc<Lower<T>>,
) -> (SmallVec<[T; 8]>, Arc<Lower<T>>, Arc<Lower<T>>)
where
    T: Clone + Eq + Hash,
{
    let mut prefix = SmallVec::<[T; 8]>::new();
    let mut left_rest = left.clone();
    let mut right_rest = right.clone();

    loop {
        let Some((left_value, next_left)) = nonempty_deterministic_top_step(&left_rest) else {
            break;
        };
        let Some((right_value, next_right)) = nonempty_deterministic_top_step(&right_rest) else {
            break;
        };

        if left_value != right_value {
            break;
        }

        prefix.push(left_value);
        left_rest = next_left;
        right_rest = next_right;
    }

    (prefix, left_rest, right_rest)
}

fn merge_upperbranches<T, A>(
    a: &Arc<UpperBranch<T, A>>,
    b: &Arc<UpperBranch<T, A>>,
) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    if Arc::ptr_eq(a, b) {
        return Arc::new(Upper::Branch(a.clone()));
    }
    let new_empty = merge_optional_acc(&a.empty, &b.empty);
    let merged_children = merge_children(&a.children, &b.children, |x, y| merge_upper(x, y));
    let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
        children: merged_children,
        empty: new_empty,
        max_depth: a.max_depth.max(b.max_depth),
    })));
    try_promote(&new_b)
}

fn merge_interfaces<T, A>(a: &Arc<Interface<T, A>>, b: &Arc<Interface<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let acc_equal = a.acc == b.acc;
    let inner_ptr_eq = Arc::ptr_eq(&a.inner, &b.inner);
    if inner_ptr_eq {
        if b.acc.subsumes(&a.acc) {
            return Arc::new(Upper::Interface(b.clone()));
        }
        if a.acc.subsumes(&b.acc) {
            return Arc::new(Upper::Interface(a.clone()));
        }
        return new_interface(a.inner.clone(), a.acc.merge(&b.acc));
    }
    if acc_equal {
        let merged_lower = merge_lower(&a.inner, &b.inner);
        let new_acc = a.acc.merge(&b.acc);
        new_interface(merged_lower, new_acc)
    } else {
        let (shared_prefix, left_rest, right_rest) =
            shared_nonempty_deterministic_prefix(&a.inner, &b.inner);
        if !shared_prefix.is_empty() {
            let left_remainder = Arc::new(Interface {
                inner: left_rest,
                acc: a.acc.clone(),
            });
            let right_remainder = Arc::new(Interface {
                inner: right_rest,
                acc: b.acc.clone(),
            });

            let mut merged = merge_interfaces(&left_remainder, &right_remainder);
            for value in shared_prefix.into_iter().rev() {
                let children = CompactMap::unit(value, CompactOrdMap::unit(merged.max_depth(), merged));
                merged = try_promote(&new_branch(children, None));
            }
            merged
        } else {
            let ub1 = interface_to_upperbranch(a);
            let ub2 = interface_to_upperbranch(b);
            merge_upperbranches(&ub1, &ub2)
        }
    }
}

fn merge_upper<T, A>(u1: &Arc<Upper<T, A>>, u2: &Arc<Upper<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    if Arc::ptr_eq(u1, u2) {
        return u1.clone();
    }
    match (&**u1, &**u2) {
        (Upper::Interface(i1), Upper::Interface(i2)) => merge_interfaces(i1, i2),
        (Upper::Branch(b1), Upper::Branch(b2)) => merge_upperbranches(b1, b2),
        (Upper::Branch(b), Upper::Interface(i)) | (Upper::Interface(i), Upper::Branch(b)) => {
            let ub = interface_to_upperbranch(i);
            merge_upperbranches(b, &ub)
        }
    }
}

fn try_promote<T, A>(node: &Arc<Upper<T, A>>) -> Arc<Upper<T, A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    if let Upper::Branch(b) = &**node {
        // Check all children are Interface (early exit without allocation).
        let mut has_children = false;
        for c in b.children.values().flat_map(|kids| kids.values()) {
            has_children = true;
            if !matches!(&**c, Upper::Interface(_)) {
                return node.clone();
            }
        }

        if !has_children {
            if let Some(empty) = &b.empty {
                let lower_root = new_lower(CompactMap::new(), true);
                return new_interface(lower_root, empty.clone());
            }
            return node.clone();
        }

        // All children are Interface. Collect accumulators (re-iterate).
        let mut accs: HashSet<A> = HashSet::new();
        if let Some(empty) = &b.empty {
            accs.insert(empty.clone());
        }
        for c in b.children.values().flat_map(|kids| kids.values()) {
            if let Upper::Interface(ic) = &**c {
                accs.insert(ic.acc.clone());
            }
        }

        if accs.len() <= 1 {
            if let Some(the_acc) = accs.into_iter().next() {

                let mut l_children: Children<T, Lower<T>> = CompactMap::new();
                for (v, kids) in b.children.iter() {
                    let mut v_map: CompactOrdMap<Arc<Lower<T>>> = CompactOrdMap::new();
                    for child in kids.values() {
                        if let Upper::Interface(ci) = &**child {
                            let lower = ci.inner.clone();
                            v_map.insert(lower.max_depth(), lower);
                        }
                    }
                    if !v_map.is_empty() {
                        l_children.insert(v.clone(), v_map);
                    }
                }
                let lower_root = new_lower(l_children, b.empty.is_some());
                return new_interface(lower_root, the_acc);
            } else {
                return empty_upper_inner();
            }
        }
    }
    node.clone()
}

fn empty_upper<T, A>() -> LeveledGSS<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    LeveledGSS {
        inner: empty_upper_inner(),
    }
}














#[derive(Clone, PartialEq, Eq, Hash)]
struct SemanticTrieNode<T> {
    empty: bool,
    children: Vec<(T, u32)>,
    max_depth: u32,
}

enum SemanticPendingNode<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    Lower(Arc<Lower<T>>),
    Upper(Arc<Upper<T, A>>),
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> SemanticPendingNode<T, A> {
    fn sort_key(&self) -> (u32, u8) {
        match self {
            Self::Lower(node) => (node.max_depth(), 0),
            Self::Upper(node) => (node.max_depth(), 1),
        }
    }
}

/// Canonicalizes the finite stack language represented by one or more GSSes.
///
/// Keys are exact within this interner: Segment boundaries, one-child General
/// nodes, depth-slot layout, accumulator values, and DAG sharing do not affect
/// the result. The canonical representation is a deterministic trie whose
/// nodes are interned and shared. Work follows reachable GSS/trie nodes rather
/// than enumerating concrete stack paths.
pub struct GssSemanticKeyInterner<
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash,
> {
    nodes: Vec<SemanticTrieNode<T>>,
    interned: FxHashMap<u64, SmallVec<[u32; 2]>>,
    lower_memo: FxHashMap<usize, (Arc<Lower<T>>, u32)>,
    upper_memo: FxHashMap<usize, (Arc<Upper<T, A>>, u32)>,
    union_memo: FxHashMap<(u32, u32), u32>,
    reconstructed: FxHashMap<u32, Arc<Lower<T>>>,
    union_pending: Vec<(u32, u32)>,
    union_unseen: FxHashSet<(u32, u32)>,
    union_order: Vec<(u32, u32)>,
    max_nodes: usize,
    max_source_keys: usize,
    max_union_entries: usize,
    exhausted: bool,
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash> std::fmt::Debug
    for GssSemanticKeyInterner<T, A>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GssSemanticKeyInterner")
            .field("nodes", &self.nodes.len())
            .field("lower_memo", &self.lower_memo.len())
            .field("upper_memo", &self.upper_memo.len())
            .field("union_memo", &self.union_memo.len())
            .field("exhausted", &self.exhausted)
            .finish()
    }
}

impl<T: Clone + Eq + Hash + Ord, A: Merge + Clone + Eq + Hash>
    GssSemanticKeyInterner<T, A>
{
    pub fn new() -> Self {
        Self::with_budget(usize::MAX, usize::MAX, usize::MAX)
    }

    /// Compatibility constructor for reusable, effectively-unbounded
    /// interners used by the established runtime paths.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut interner = Self::with_budget(usize::MAX, usize::MAX, usize::MAX);
        interner.nodes.reserve(capacity);
        interner.interned.reserve(capacity);
        interner.lower_memo.reserve(capacity);
        interner.upper_memo.reserve(capacity);
        interner.union_memo.reserve(capacity);
        interner.reconstructed.reserve(capacity);
        interner
    }

    pub fn with_budget(
        max_nodes: usize,
        max_source_keys: usize,
        max_union_entries: usize,
    ) -> Self {
        let empty_language = SemanticTrieNode {
            empty: false,
            children: Vec::new(),
            max_depth: 0,
        };
        let empty_hash = Self::semantic_node_hash(&empty_language);
        let mut interned = FxHashMap::default();
        interned.insert(empty_hash, smallvec![0]);
        Self {
            nodes: vec![empty_language],
            interned,
            lower_memo: FxHashMap::default(),
            upper_memo: FxHashMap::default(),
            union_memo: FxHashMap::default(),
            reconstructed: FxHashMap::default(),
            union_pending: Vec::new(),
            union_unseen: FxHashSet::default(),
            union_order: Vec::new(),
            max_nodes: max_nodes.max(1),
            max_source_keys,
            max_union_entries,
            exhausted: false,
        }
    }

    #[inline]
    fn semantic_node_hash(node: &SemanticTrieNode<T>) -> u64 {
        let mut hasher = FxHasher::default();
        node.hash(&mut hasher);
        hasher.finish()
    }

    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Reset semantic contents while retaining scratch allocations and the
    /// configured work budgets.
    pub fn clear(&mut self) {
        let empty_language = SemanticTrieNode {
            empty: false,
            children: Vec::new(),
            max_depth: 0,
        };
        let empty_hash = Self::semantic_node_hash(&empty_language);
        self.nodes.clear();
        self.nodes.push(empty_language);
        self.interned.clear();
        self.interned.insert(empty_hash, smallvec![0]);
        self.lower_memo.clear();
        self.upper_memo.clear();
        self.union_memo.clear();
        self.reconstructed.clear();
        self.union_pending.clear();
        self.union_unseen.clear();
        self.union_order.clear();
        self.exhausted = false;
    }

    #[inline]
    fn source_budget_available(&mut self) -> bool {
        if self.lower_memo.len().saturating_add(self.upper_memo.len())
            >= self.max_source_keys
        {
            self.exhausted = true;
            false
        } else {
            true
        }
    }

    #[inline]
    fn lower_id(&self, node: &Arc<Lower<T>>) -> Option<u32> {
        self.lower_memo
            .get(&lower_node_id(node))
            .map(|(_, id)| *id)
    }

    #[inline]
    fn upper_id(&self, node: &Arc<Upper<T, A>>) -> Option<u32> {
        self.upper_memo
            .get(&(Arc::as_ptr(node) as usize))
            .map(|(_, id)| *id)
    }

    fn intern_node(&mut self, empty: bool, mut children: Vec<(T, u32)>) -> u32 {
        children.retain(|(_, child)| *child != 0);
        children.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        debug_assert!(children.windows(2).all(|pair| pair[0].0 != pair[1].0));
        let max_depth = children
            .iter()
            .map(|(_, child)| self.nodes[*child as usize].max_depth.saturating_add(1))
            .max()
            .unwrap_or(0);
        let node = SemanticTrieNode {
            empty,
            children,
            max_depth,
        };
        let hash = Self::semantic_node_hash(&node);
        if let Some(candidates) = self.interned.get(&hash)
            && let Some(&id) = candidates
                .iter()
                .find(|&&id| self.nodes[id as usize] == node)
        {
            return id;
        }
        if self.nodes.len() >= self.max_nodes {
            self.exhausted = true;
            return 0;
        }
        let id = u32::try_from(self.nodes.len()).expect("semantic GSS trie exceeded u32 node IDs");
        self.nodes.push(node);
        self.interned.entry(hash).or_default().push(id);
        id
    }

    #[inline]
    fn union_pair(left: u32, right: u32) -> (u32, u32) {
        if left <= right {
            (left, right)
        } else {
            (right, left)
        }
    }

    fn union_ids(&mut self, left: u32, right: u32) -> u32 {
        if self.exhausted {
            return 0;
        }
        if left == right || right == 0 {
            return left;
        }
        if left == 0 {
            return right;
        }
        let root = Self::union_pair(left, right);
        if let Some(id) = self.union_memo.get(&root) {
            return *id;
        }

        let mut pending = std::mem::take(&mut self.union_pending);
        let mut unseen = std::mem::take(&mut self.union_unseen);
        let mut order = std::mem::take(&mut self.union_order);
        pending.clear();
        unseen.clear();
        order.clear();
        pending.push(root);

        let result = self.union_ids_uncached(root, &mut pending, &mut unseen, &mut order);

        pending.clear();
        unseen.clear();
        order.clear();
        self.union_pending = pending;
        self.union_unseen = unseen;
        self.union_order = order;
        result
    }

    fn union_ids_uncached(
        &mut self,
        root: (u32, u32),
        pending: &mut Vec<(u32, u32)>,
        unseen: &mut FxHashSet<(u32, u32)>,
        order: &mut Vec<(u32, u32)>,
    ) -> u32 {
        while let Some(pair @ (left, right)) = pending.pop() {
            if left == right || left == 0 || right == 0 || self.union_memo.contains_key(&pair) {
                continue;
            }
            if !unseen.insert(pair) {
                continue;
            }
            if unseen.len() > self.max_union_entries {
                self.exhausted = true;
                return 0;
            }
            order.push(pair);

            let left_children = &self.nodes[left as usize].children;
            let right_children = &self.nodes[right as usize].children;
            let mut li = 0;
            let mut ri = 0;
            while li < left_children.len() && ri < right_children.len() {
                match left_children[li].0.cmp(&right_children[ri].0) {
                    std::cmp::Ordering::Less => li += 1,
                    std::cmp::Ordering::Greater => ri += 1,
                    std::cmp::Ordering::Equal => {
                        let child_pair = Self::union_pair(left_children[li].1, right_children[ri].1);
                        if child_pair.0 != child_pair.1
                            && child_pair.0 != 0
                            && !self.union_memo.contains_key(&child_pair)
                        {
                            pending.push(child_pair);
                        }
                        li += 1;
                        ri += 1;
                    }
                }
            }
        }

        order.sort_unstable_by_key(|(left, right)| {
            self.nodes[*left as usize]
                .max_depth
                .max(self.nodes[*right as usize].max_depth)
        });

        for pair @ (left, right) in order.drain(..) {
            if self.union_memo.contains_key(&pair) {
                continue;
            }
            let (empty, children) = {
                let left_node = &self.nodes[left as usize];
                let right_node = &self.nodes[right as usize];
                let mut children =
                    Vec::with_capacity(left_node.children.len() + right_node.children.len());
                let mut li = 0;
                let mut ri = 0;
                while li < left_node.children.len() || ri < right_node.children.len() {
                    if ri == right_node.children.len()
                        || (li < left_node.children.len()
                            && left_node.children[li].0 < right_node.children[ri].0)
                    {
                        children.push(left_node.children[li].clone());
                        li += 1;
                    } else if li == left_node.children.len()
                        || right_node.children[ri].0 < left_node.children[li].0
                    {
                        children.push(right_node.children[ri].clone());
                        ri += 1;
                    } else {
                        let left_child = left_node.children[li].1;
                        let right_child = right_node.children[ri].1;
                        let child = if left_child == right_child {
                            left_child
                        } else if left_child == 0 {
                            right_child
                        } else if right_child == 0 {
                            left_child
                        } else {
                            *self
                                .union_memo
                                .get(&Self::union_pair(left_child, right_child))
                                .expect(
                                    "semantic trie child union must be processed before its parent",
                                )
                        };
                        children.push((left_node.children[li].0.clone(), child));
                        li += 1;
                        ri += 1;
                    }
                }
                (left_node.empty || right_node.empty, children)
            };
            let id = self.intern_node(empty, children);
            if self.exhausted {
                return 0;
            }
            if self.union_memo.len() >= self.max_union_entries {
                self.exhausted = true;
                return 0;
            }
            self.union_memo.insert(pair, id);
        }

        self.union_memo.get(&root).copied().unwrap_or(0)
    }

    pub fn union_keys(&mut self, left: u32, right: u32) -> u32 {
        self.union_ids(left, right)
    }

    pub fn top_branches(&self, root: u32) -> &[(T, u32)] {
        &self.nodes[root as usize].children
    }

    pub fn push_key(&mut self, root: u32, value: T) -> u32 {
        if root == 0 {
            return 0;
        }
        self.intern_node(false, vec![(value, root)])
    }

    pub fn gss_from_key(&mut self, root: u32, accumulator: A) -> LeveledGSS<T, A> {
        if root == 0 {
            return LeveledGSS::empty();
        }

        fn build<T: Clone + Eq + Hash + Ord>(
            nodes: &[SemanticTrieNode<T>],
            id: u32,
            memo: &mut FxHashMap<u32, Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            if let Some(cached) = memo.get(&id) {
                return cached.clone();
            }
            let node = &nodes[id as usize];
            let mut children: Children<T, Lower<T>> = CompactMap::new();
            for (value, child_id) in &node.children {
                let child = build(nodes, *child_id, memo);
                insert_lower_child_shared(
                    &mut children,
                    value.clone(),
                    child.max_depth(),
                    child,
                );
            }
            let lower = new_lower(children, node.empty);
            memo.insert(id, lower.clone());
            lower
        }

        let lower = build(&self.nodes, root, &mut self.reconstructed);
        let gss = LeveledGSS {
            inner: new_interface(lower, accumulator),
        };
        self.upper_memo.insert(
            Arc::as_ptr(&gss.inner) as usize,
            (gss.inner.clone(), root),
        );
        gss
    }

    pub fn key(&mut self, gss: &LeveledGSS<T, A>) -> u32 {
        if self.exhausted {
            return 0;
        }
        if let Some(id) = self.upper_id(&gss.inner) {
            return id;
        }

        // Deterministic shallow stacks dominate exact-admission queries. Build
        // their canonical unary trie directly instead of allocating traversal
        // worklists and pointer memo tables. The cached max-depth check makes
        // rejection O(1), and the traversal is strictly capped.
        const SINGLE_STACK_KEY_MAX_DEPTH: usize = 64;
        if gss.max_depth() as usize <= SINGLE_STACK_KEY_MAX_DEPTH
            && let Some((stack, _)) = gss.try_single_stack_bounded(SINGLE_STACK_KEY_MAX_DEPTH)
        {
            let mut id = self.intern_node(true, Vec::new());
            for value in stack {
                id = self.intern_node(false, vec![(value, id)]);
            }
            if self.exhausted || !self.source_budget_available() {
                return 0;
            }
            self.upper_memo.insert(
                Arc::as_ptr(&gss.inner) as usize,
                (gss.inner.clone(), id),
            );
            return id;
        }

        // Shallow ambiguous DAGs are common in token-local admission. Walk
        // them in bounded postorder using the exact pointer memo tables rather
        // than allocating two visited sets, a pending list, and a sortable
        // node list. The iterative path below remains the fallback for deeper
        // structures, so recursion depth is strictly bounded.
        const RECURSIVE_SEMANTIC_KEY_MAX_DEPTH: usize = 64;
        if gss.max_depth() as usize <= RECURSIVE_SEMANTIC_KEY_MAX_DEPTH
            && let Some(id) = self.key_upper_recursive(
                &gss.inner,
                0,
                RECURSIVE_SEMANTIC_KEY_MAX_DEPTH,
            )
        {
            return id;
        }

        let mut pending = vec![SemanticPendingNode::Upper(gss.inner.clone())];
        let mut seen_upper = FxHashSet::default();
        let mut seen_lower = FxHashSet::default();
        let mut nodes = Vec::new();

        while let Some(node) = pending.pop() {
            match &node {
                SemanticPendingNode::Lower(lower) => {
                    let ptr = lower_node_id(lower);
                    if self.lower_memo.contains_key(&ptr) || !seen_lower.insert(ptr) {
                        continue;
                    }
                    if self
                        .lower_memo
                        .len()
                        .saturating_add(self.upper_memo.len())
                        .saturating_add(seen_lower.len())
                        .saturating_add(seen_upper.len())
                        > self.max_source_keys
                    {
                        self.exhausted = true;
                        return 0;
                    }
                    match &**lower {
                        Lower::Segment(segment) => {
                            pending.push(SemanticPendingNode::Lower(segment.next.clone()));
                        }
                        Lower::General { children, .. } => {
                            pending.extend(
                                children
                                    .values()
                                    .flat_map(|kids| kids.values())
                                    .cloned()
                                    .map(SemanticPendingNode::Lower),
                            );
                        }
                    }
                    nodes.push(node);
                }
                SemanticPendingNode::Upper(upper) => {
                    let ptr = Arc::as_ptr(upper) as usize;
                    if self.upper_memo.contains_key(&ptr) || !seen_upper.insert(ptr) {
                        continue;
                    }
                    if self
                        .lower_memo
                        .len()
                        .saturating_add(self.upper_memo.len())
                        .saturating_add(seen_lower.len())
                        .saturating_add(seen_upper.len())
                        > self.max_source_keys
                    {
                        self.exhausted = true;
                        return 0;
                    }
                    match &**upper {
                        Upper::Interface(interface) => {
                            pending.push(SemanticPendingNode::Lower(interface.inner.clone()));
                        }
                        Upper::Branch(branch) => {
                            pending.extend(
                                branch
                                    .children
                                    .values()
                                    .flat_map(|kids| kids.values())
                                    .cloned()
                                    .map(SemanticPendingNode::Upper),
                            );
                        }
                    }
                    nodes.push(node);
                }
            }
        }

        nodes.sort_unstable_by_key(SemanticPendingNode::sort_key);
        for node in nodes {
            if self.exhausted {
                return 0;
            }
            match node {
                SemanticPendingNode::Lower(lower) => {
                    let ptr = lower_node_id(&lower);
                    if self.lower_memo.contains_key(&ptr) {
                        continue;
                    }
                    let id = match &*lower {
                        Lower::Segment(segment) => {
                            let mut id = self
                                .lower_id(&segment.next)
                                .expect("semantic trie Segment child must precede parent");
                            for value in segment.values.iter() {
                                id = self.intern_node(false, vec![(value.clone(), id)]);
                            }
                            id
                        }
                        Lower::General {
                            children, empty, ..
                        } => {
                            let mut canonical_children = Vec::with_capacity(children.len());
                            for (value, kids) in children.iter() {
                                let mut child_union = 0;
                                for child in kids.values() {
                                    let child_id = self
                                        .lower_id(child)
                                        .expect("semantic trie General child must precede parent");
                                    child_union = self.union_ids(child_union, child_id);
                                }
                                canonical_children.push((value.clone(), child_union));
                            }
                            self.intern_node(*empty, canonical_children)
                        }
                    };
                    if self.exhausted || !self.source_budget_available() {
                        return 0;
                    }
                    self.lower_memo.insert(ptr, (lower, id));
                }
                SemanticPendingNode::Upper(upper) => {
                    let ptr = Arc::as_ptr(&upper) as usize;
                    if self.upper_memo.contains_key(&ptr) {
                        continue;
                    }
                    let id = match &*upper {
                        Upper::Interface(interface) => self
                            .lower_id(&interface.inner)
                            .expect("semantic trie Interface lower node must precede parent"),
                        Upper::Branch(branch) => {
                            let mut canonical_children = Vec::with_capacity(branch.children.len());
                            for (value, kids) in branch.children.iter() {
                                let mut child_union = 0;
                                for child in kids.values() {
                                    let child_id = self
                                        .upper_id(child)
                                        .expect("semantic trie Branch child must precede parent");
                                    child_union = self.union_ids(child_union, child_id);
                                }
                                canonical_children.push((value.clone(), child_union));
                            }
                            self.intern_node(branch.empty.is_some(), canonical_children)
                        }
                    };
                    if self.exhausted || !self.source_budget_available() {
                        return 0;
                    }
                    self.upper_memo.insert(ptr, (upper, id));
                }
            }
        }

        self.upper_id(&gss.inner).unwrap_or(0)
    }

    fn key_lower_recursive(
        &mut self,
        lower: &Arc<Lower<T>>,
        depth: usize,
        max_depth: usize,
    ) -> Option<u32> {
        if let Some(id) = self.lower_id(lower) {
            return Some(id);
        }
        if depth > max_depth {
            return None;
        }
        let id = match &**lower {
            Lower::Segment(segment) => {
                let mut id = self.key_lower_recursive(&segment.next, depth + 1, max_depth)?;
                for value in segment.values.iter() {
                    id = self.intern_node(false, vec![(value.clone(), id)]);
                    if self.exhausted {
                        return Some(0);
                    }
                }
                id
            }
            Lower::General {
                children, empty, ..
            } => {
                let mut canonical_children = Vec::with_capacity(children.len());
                for (value, kids) in children.iter() {
                    let mut child_union = 0;
                    for child in kids.values() {
                        let child_id = self.key_lower_recursive(child, depth + 1, max_depth)?;
                        child_union = self.union_ids(child_union, child_id);
                        if self.exhausted {
                            return Some(0);
                        }
                    }
                    canonical_children.push((value.clone(), child_union));
                }
                self.intern_node(*empty, canonical_children)
            }
        };
        if self.exhausted || !self.source_budget_available() {
            return Some(0);
        }
        let ptr = lower_node_id(lower);
        self.lower_memo.insert(ptr, (lower.clone(), id));
        Some(id)
    }

    fn key_upper_recursive(
        &mut self,
        upper: &Arc<Upper<T, A>>,
        depth: usize,
        max_depth: usize,
    ) -> Option<u32> {
        if let Some(id) = self.upper_id(upper) {
            return Some(id);
        }
        if depth > max_depth {
            return None;
        }
        let id = match &**upper {
            Upper::Interface(interface) => {
                self.key_lower_recursive(&interface.inner, depth + 1, max_depth)?
            }
            Upper::Branch(branch) => {
                let mut canonical_children = Vec::with_capacity(branch.children.len());
                for (value, kids) in branch.children.iter() {
                    let mut child_union = 0;
                    for child in kids.values() {
                        let child_id = self.key_upper_recursive(child, depth + 1, max_depth)?;
                        child_union = self.union_ids(child_union, child_id);
                        if self.exhausted {
                            return Some(0);
                        }
                    }
                    canonical_children.push((value.clone(), child_union));
                }
                self.intern_node(branch.empty.is_some(), canonical_children)
            }
        };
        if self.exhausted || !self.source_budget_available() {
            return Some(0);
        }
        let ptr = Arc::as_ptr(upper) as usize;
        self.upper_memo.insert(ptr, (upper.clone(), id));
        Some(id)
    }

    pub fn work_summary(&self) -> (usize, usize, usize, usize) {
        (
            self.nodes.len(),
            self.lower_memo.len(),
            self.upper_memo.len(),
            self.union_memo.len(),
        )
    }

    #[cfg(test)]
    fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

pub struct IndexedLowerIdentity<T: Clone + Eq + Hash> {
    ptr: usize,
    _keepalive: Arc<Lower<T>>,
}

impl<T: Clone + Eq + Hash> IndexedLowerIdentity<T> {
    fn new(lower: &Arc<Lower<T>>) -> Self {
        Self {
            ptr: lower_node_id(lower),
            _keepalive: lower.clone(),
        }
    }

    #[inline]
    pub fn ptr_key(&self) -> usize {
        self.ptr
    }
}

impl<T: Clone + Eq + Hash> Clone for IndexedLowerIdentity<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _keepalive: self._keepalive.clone(),
        }
    }
}

impl<T: Clone + Eq + Hash> PartialEq for IndexedLowerIdentity<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}
impl<T: Clone + Eq + Hash> Eq for IndexedLowerIdentity<T> {}
impl<T: Clone + Eq + Hash> std::hash::Hash for IndexedLowerIdentity<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

#[derive(Clone)]
pub enum IndexedLeveledGssNode<T: Clone + Eq + Hash, A> {
    UpperBranch {
        empty: Option<A>,
        children: Vec<(T, u32)>,
    },
    Interface {
        accumulator: A,
        lower: u32,
    },
    LowerGeneral {
        source: IndexedLowerIdentity<T>,
        empty: bool,
        children: Vec<(T, u32)>,
    },
    LowerSegment {
        source: IndexedLowerIdentity<T>,
        values: Vec<T>,
        next: u32,
    },
}

pub struct IndexedLeveledGss<T: Clone + Eq + Hash, A> {
    pub root: u32,
    pub nodes: Vec<IndexedLeveledGssNode<T, A>>,
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
    /// Build a compact indexed view of the existing leveled-GSS DAG.
    ///
    /// Node sharing is preserved by pointer identity. This is intended for
    /// allocation-free dynamic programs that need random access to the DAG but
    /// must not rebuild, flatten, or enumerate the represented stack language.
    pub fn indexed_dag(&self) -> IndexedLeveledGss<T, A> {
        fn index_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            lower: &Arc<Lower<T>>,
            nodes: &mut Vec<IndexedLeveledGssNode<T, A>>,
            lower_ids: &mut FxHashMap<usize, u32>,
            upper_ids: &mut FxHashMap<usize, u32>,
        ) -> u32 {
            let ptr = lower_node_id(lower);
            if let Some(id) = lower_ids.get(&ptr) {
                return *id;
            }
            let node = match &**lower {
                Lower::General { children, empty, .. } => {
                    let mut indexed_children = Vec::new();
                    for (value, kids) in children.iter() {
                        for child in kids.values() {
                            indexed_children.push((
                                value.clone(),
                                index_lower(child, nodes, lower_ids, upper_ids),
                            ));
                        }
                    }
                    IndexedLeveledGssNode::LowerGeneral {
                        source: IndexedLowerIdentity::new(lower),
                        empty: *empty,
                        children: indexed_children,
                    }
                }
                Lower::Segment(segment) => IndexedLeveledGssNode::LowerSegment {
                    source: IndexedLowerIdentity::new(lower),
                    values: segment.values.iter().cloned().collect::<Vec<_>>(),
                    next: index_lower(&segment.next, nodes, lower_ids, upper_ids),
                },
            };
            let id = u32::try_from(nodes.len()).expect("indexed GSS exceeded u32 node IDs");
            nodes.push(node);
            lower_ids.insert(ptr, id);
            id
        }

        fn index_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            upper: &Arc<Upper<T, A>>,
            nodes: &mut Vec<IndexedLeveledGssNode<T, A>>,
            lower_ids: &mut FxHashMap<usize, u32>,
            upper_ids: &mut FxHashMap<usize, u32>,
        ) -> u32 {
            let ptr = Arc::as_ptr(upper) as usize;
            if let Some(id) = upper_ids.get(&ptr) {
                return *id;
            }
            let node = match &**upper {
                Upper::Branch(branch) => {
                    let mut indexed_children = Vec::new();
                    for (value, kids) in branch.children.iter() {
                        for child in kids.values() {
                            indexed_children.push((
                                value.clone(),
                                index_upper(child, nodes, lower_ids, upper_ids),
                            ));
                        }
                    }
                    IndexedLeveledGssNode::UpperBranch {
                        empty: branch.empty.clone(),
                        children: indexed_children,
                    }
                }
                Upper::Interface(interface) => IndexedLeveledGssNode::Interface {
                    accumulator: interface.acc.clone(),
                    lower: index_lower(&interface.inner, nodes, lower_ids, upper_ids),
                },
            };
            let id = u32::try_from(nodes.len()).expect("indexed GSS exceeded u32 node IDs");
            nodes.push(node);
            upper_ids.insert(ptr, id);
            id
        }

        let mut nodes = Vec::new();
        let mut lower_ids = FxHashMap::default();
        let mut upper_ids = FxHashMap::default();
        let root = index_upper(
            &self.inner,
            &mut nodes,
            &mut lower_ids,
            &mut upper_ids,
        );
        IndexedLeveledGss { root, nodes }
    }

    pub fn indexed_dag_many(
        roots: &[Self],
    ) -> (IndexedLeveledGss<T, A>, Vec<u32>) {
        let (dag, root_ids, _, _) = Self::indexed_dag_many_reusing(
            roots,
            Vec::new(),
            Vec::new(),
            FxHashMap::default(),
            FxHashMap::default(),
        );
        (dag, root_ids)
    }

    pub fn indexed_dag_many_reusing(
        roots: &[Self],
        mut nodes: Vec<IndexedLeveledGssNode<T, A>>,
        mut root_ids: Vec<u32>,
        mut lower_ids: FxHashMap<usize, u32>,
        mut upper_ids: FxHashMap<usize, u32>,
    ) -> (
        IndexedLeveledGss<T, A>,
        Vec<u32>,
        FxHashMap<usize, u32>,
        FxHashMap<usize, u32>,
    ) {
        fn index_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            lower: &Arc<Lower<T>>,
            nodes: &mut Vec<IndexedLeveledGssNode<T, A>>,
            lower_ids: &mut FxHashMap<usize, u32>,
            upper_ids: &mut FxHashMap<usize, u32>,
        ) -> u32 {
            let ptr = lower_node_id(lower);
            if let Some(id) = lower_ids.get(&ptr) {
                return *id;
            }
            let node = match &**lower {
                Lower::General { children, empty, .. } => {
                    let mut indexed_children = Vec::new();
                    for (value, kids) in children.iter() {
                        for child in kids.values() {
                            indexed_children.push((
                                value.clone(),
                                index_lower(child, nodes, lower_ids, upper_ids),
                            ));
                        }
                    }
                    IndexedLeveledGssNode::LowerGeneral {
                        source: IndexedLowerIdentity::new(lower),
                        empty: *empty,
                        children: indexed_children,
                    }
                }
                Lower::Segment(segment) => IndexedLeveledGssNode::LowerSegment {
                    source: IndexedLowerIdentity::new(lower),
                    values: segment.values.iter().cloned().collect::<Vec<_>>(),
                    next: index_lower(&segment.next, nodes, lower_ids, upper_ids),
                },
            };
            let id = u32::try_from(nodes.len()).expect("indexed GSS exceeded u32 node IDs");
            nodes.push(node);
            lower_ids.insert(ptr, id);
            id
        }

        fn index_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            upper: &Arc<Upper<T, A>>,
            nodes: &mut Vec<IndexedLeveledGssNode<T, A>>,
            lower_ids: &mut FxHashMap<usize, u32>,
            upper_ids: &mut FxHashMap<usize, u32>,
        ) -> u32 {
            let ptr = Arc::as_ptr(upper) as usize;
            if let Some(id) = upper_ids.get(&ptr) {
                return *id;
            }
            let node = match &**upper {
                Upper::Branch(branch) => {
                    let mut indexed_children = Vec::new();
                    for (value, kids) in branch.children.iter() {
                        for child in kids.values() {
                            indexed_children.push((
                                value.clone(),
                                index_upper(child, nodes, lower_ids, upper_ids),
                            ));
                        }
                    }
                    IndexedLeveledGssNode::UpperBranch {
                        empty: branch.empty.clone(),
                        children: indexed_children,
                    }
                }
                Upper::Interface(interface) => IndexedLeveledGssNode::Interface {
                    accumulator: interface.acc.clone(),
                    lower: index_lower(&interface.inner, nodes, lower_ids, upper_ids),
                },
            };
            let id = u32::try_from(nodes.len()).expect("indexed GSS exceeded u32 node IDs");
            nodes.push(node);
            upper_ids.insert(ptr, id);
            id
        }

        nodes.clear();
        root_ids.clear();
        lower_ids.clear();
        upper_ids.clear();
        root_ids.extend(roots.iter().map(|root| {
            index_upper(
                &root.inner,
                &mut nodes,
                &mut lower_ids,
                &mut upper_ids,
            )
        }));
        let root = root_ids.first().copied().unwrap_or(0);
        (
            IndexedLeveledGss { root, nodes },
            root_ids,
            lower_ids,
            upper_ids,
        )
    }
}

#[derive(Clone)]
pub struct LeveledGSS<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    inner: Arc<Upper<T, A>>,
}

/// A mutable view of the top Segment prefix of a GSS as a flat stack of values.
///
/// This deliberately does not walk a single-child General chain during
/// construction: without cached deterministic-prefix length, doing so could make
/// `try_virtual_stack` an unbounded linear operation. Short single-path General
/// chains are handled by the separately depth-bounded flat-stack fast path.
///
/// Instead of extracting all states upfront, this keeps a reference to the
/// original chain and only tracks pushed states (from goto operations).
/// Pops walk through the original chain via Arc references.
/// On commit, only the pushed portion needs new Segment nodes.
///
/// The stack has a "floor" — the Lower node below the deterministic chain.
/// When a pop would cross the floor, the caller can materialize the current
/// chain and continue with GSS-level operations for the leftover pop depth.
///
/// `pending_top` is a lazy optimization: pushes set `pending_top` instead of
/// immediately modifying the backing values. If a pop immediately follows,
/// we consume `pending_top` first, avoiding touching the segment chain at all.
/// This is a common pattern during deterministic reduce chains:
///   pop(n) → push(goto) → pop(m) → push(goto2) → ...
#[derive(Clone)]
pub struct VirtualStack<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    values: SV<T>,
    next: Arc<Lower<T>>,
    acc: A,
    pending_top: Option<T>,
}

/// Controls how eagerly VirtualStack creates new segments vs reusing existing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PushMode {
    /// Use try_push only. On failure, create a new segment. (default)
    Lazy,
    /// Use try_harder_push first (clone shared data if needed).
    /// Only create a new segment if that also fails.
    Eager,
}

static PUSH_MODE: OnceLock<PushMode> = OnceLock::new();

fn push_mode() -> PushMode {
    *PUSH_MODE.get_or_init(|| {
        match std::env::var("PUSH_MODE").as_deref() {
            Ok("eager") | Ok("harder") => PushMode::Eager,
            _ => PushMode::Lazy,
        }
    })
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> VirtualStack<T, A> {
    /// The current top-of-stack value, or None if the stack is empty.
    #[inline]
    pub fn top(&self) -> Option<&T> {
        self.pending_top.as_ref().or_else(|| self.values.last())
    }

    /// If `self` is exactly `base` with one extra top value pushed, return that
    /// extra value.  This is deliberately structural: it compares the full
    /// visible stack prefix, not just the immediate parent, so callers may use
    /// it to batch many independently-advanced virtual stacks back into one GSS.
    pub fn single_top_extension_of(&self, base: &Self) -> Option<T> {
        if self.len() != base.len() + 1 {
            return None;
        }
        for depth in 1..=base.len() {
            if self.top_after_popping(depth)? != base.top_after_popping(depth - 1)? {
                return None;
            }
        }
        self.top().cloned()
    }

    /// Return the top value that would be visible after popping `remaining`
    /// values, without mutating or cloning the virtual stack.
    #[inline]
    pub fn top_after_popping(&self, mut remaining: usize) -> Option<&T> {
        if let Some(top) = self.pending_top.as_ref() {
            if remaining == 0 {
                return Some(top);
            }
            remaining -= 1;
        }

        let mut values = &self.values;
        let mut next = &self.next;
        loop {
            let len = values.len();
            if remaining < len {
                return values.iter().rev().nth(remaining);
            }
            remaining -= len;
            match &**next {
                Lower::Segment(seg) => {
                    values = &seg.values;
                    next = &seg.next;
                }
                _ => return None,
            }
        }
    }

    /// Flush pending_top into the backing values.
    #[inline]
    fn flush_pending(&mut self) {
        if let Some(val) = self.pending_top.take() {
            self.realize_push(val);
        }
    }

    /// Actually push a value into the backing storage.
    #[inline]
    fn realize_push(&mut self, value: T) {
        let pushed = match push_mode() {
            PushMode::Lazy => self.values.try_push(value.clone()),
            PushMode::Eager => {
                if self.values.try_push(value.clone()) {
                    true
                } else {
                    self.values.try_harder_push(value.clone())
                }
            }
        };
        if !pushed {
            let seg = new_segment(self.values.clone(), self.next.clone());
            self.next = seg;
            self.values = SV::unit(value);
        }
    }

    /// Pop `n` values from the top.
    /// Returns the number of values that could not be popped because the
    /// segment chain ended at a non-Segment lower node.
    #[inline]
    pub fn pop(&mut self, mut remaining: usize) -> usize {
        // Consume pending_top first (free).
        if remaining > 0 && self.pending_top.is_some() {
            self.pending_top = None;
            remaining -= 1;
        }
        while remaining > 0 {
            let take = remaining.min(self.values.len());
            let keep = self.values.len() - take;
            self.values.truncate(keep);
            remaining -= take;
            if remaining == 0 {
                break;
            }
            match &*self.next {
                Lower::Segment(seg) => {
                    self.values = seg.values.clone();
                    self.next = seg.next.clone();
                }
                _ => break,
            }
        }
        // If values was exactly drained, advance to next Segment so top() works.
        if self.values.is_empty() && self.pending_top.is_none() {
            if let Lower::Segment(seg) = &*self.next {
                self.values = seg.values.clone();
                self.next = seg.next.clone();
            }
        }
        remaining
    }

    /// Push a value onto the top of the stack.
    /// Defers the actual push — stores in pending_top. If there's already a
    /// pending value, flushes it to the backing storage first.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.flush_pending();
        self.pending_top = Some(value);
    }

    /// Return the value immediately below the current top, if any.
    #[inline]
    pub fn parent_of_top(&self) -> Option<T> {
        self.top_after_popping(1).cloned()
    }

    /// Replace the current top-of-stack value in place.
    #[inline]
    pub fn replace_top(&mut self, value: T) -> bool {
        if self.top().is_none() {
            return false;
        }
        if self.pending_top.is_some() {
            self.pending_top = Some(value);
            return true;
        }

        let len = self.values.len();
        if len > 0 {
            self.values.truncate(len - 1);
            self.pending_top = Some(value);
            return true;
        }

        true
    }

    /// The total number of values available across the current segment chain.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len() + self.next.segments_len() + if self.pending_top.is_some() { 1 } else { 0 }
    }

    /// Whether the non-segment floor below this virtual stack still contains
    /// one or more stack values. A virtual stack only materializes the linear
    /// Segment prefix above that floor; guarded operations that need to inspect
    /// deeper values must fall back to the branch-aware GSS representation.
    #[inline]
    pub fn has_hidden_floor_values(&self) -> bool {
        let mut next = &self.next;
        loop {
            match &**next {
                Lower::Segment(segment) => next = &segment.next,
                Lower::General { children, .. } => return !children.is_empty(),
            }
        }
    }

    /// Materialize the virtual stack back into a GSS.
    pub fn into_gss(mut self) -> LeveledGSS<T, A> {
        self.flush_pending();
        if self.values.is_empty() {
            return LeveledGSS {
                inner: new_interface(self.next, self.acc),
            };
        }
        LeveledGSS {
            inner: new_interface(new_segment(self.values, self.next), self.acc),
        }
    }

    pub fn into_gss_after_popping(mut self, n: usize) -> LeveledGSS<T, A> {
        self.flush_pending();
        let remaining = self.pop(n);
        let gss = self.into_gss();
        if remaining == 0 {
            gss
        } else {
            gss.popn(remaining as isize)
        }
    }

    pub fn into_gss_after_popping_and_pushing_branches<'a, I>(
        mut self,
        n: usize,
        pushes: I,
    ) -> Option<LeveledGSS<T, A>>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        self.flush_pending();
        if self.pop(n) != 0 {
            return None;
        }

        let base = if self.values.is_empty() {
            self.next
        } else {
            new_segment(self.values, self.next)
        };

        let mut children: Children<T, Lower<T>> = CompactMap::new();
        for pushes in pushes {
            let (top, prefix) = pushes.split_last()?;
            let mut child = base.clone();
            for value in prefix {
                child = new_segment(SV::unit(value.clone()), child);
            }

            let depth = child.max_depth();
            if let Some(existing) = children.get_mut(top) {
                if let Some(existing_child) = existing.get(&depth).cloned() {
                    existing.insert(depth, merge_lower(&existing_child, &child));
                } else {
                    existing.insert(depth, child);
                }
            } else {
                children.insert(top.clone(), CompactOrdMap::unit(depth, child));
            }
        }

        if children.is_empty() {
            return Some(LeveledGSS {
                inner: new_interface(base, self.acc),
            });
        }

        Some(LeveledGSS {
            inner: new_interface(new_lower(children, false), self.acc),
        })
    }

    /// Replace one concrete stack suffix with many already-unique single-value
    /// branches. The caller guarantees unique targets, so construction is
    /// linear and the shared child is already canonical.
    pub fn into_gss_after_popping_and_pushing_unique_single_branches<'a, I>(
        mut self,
        n: usize,
        targets: I,
    ) -> Option<LeveledGSS<T, A>>
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        self.flush_pending();
        if self.pop(n) != 0 {
            return None;
        }

        let base = if self.values.is_empty() {
            self.next
        } else {
            new_segment(self.values, self.next)
        };
        let base_depth = base.max_depth();
        let child = CompactOrdMap::unit(base_depth, base.clone());
        let entries: SmallVec<[(T, CompactOrdMap<Arc<Lower<T>>>); 4]> = targets
            .into_iter()
            .map(|target| (target.clone(), child.clone()))
            .collect();

        if entries.is_empty() {
            return Some(LeveledGSS {
                inner: new_interface(base, self.acc),
            });
        }

        Some(LeveledGSS {
            inner: new_interface(
                new_lower_precanonicalized(CompactMap::Inline(entries), false),
                self.acc,
            ),
        })
    }

    pub fn into_gss_after_popping_and_pushing_single_branches<'a, I>(
        mut self,
        n: usize,
        targets: I,
    ) -> Option<LeveledGSS<T, A>>
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        self.flush_pending();
        if self.pop(n) != 0 {
            return None;
        }

        let base = if self.values.is_empty() {
            self.next
        } else {
            new_segment(self.values, self.next)
        };
        let base_depth = base.max_depth();

        let mut entries: SmallVec<[(T, CompactOrdMap<Arc<Lower<T>>>); 4]> = SmallVec::new();
        let child = CompactOrdMap::unit(base_depth, base.clone());
        for target in targets {
            if entries.iter().any(|(existing, _)| existing == target) {
                continue;
            }
            entries.push((target.clone(), child.clone()));
        }

        if entries.is_empty() {
            return Some(LeveledGSS {
                inner: new_interface(base, self.acc),
            });
        }

        Some(LeveledGSS {
            inner: new_interface(new_lower(CompactMap::Inline(entries), false), self.acc),
        })
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> PartialEq for LeveledGSS<T, A> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || *self.inner == *other.inner
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Eq for LeveledGSS<T, A> {}

impl<T: Clone + Eq + Hash + std::fmt::Debug, A: Merge + Clone + Eq + Hash + std::fmt::Debug> std::fmt::Debug for LeveledGSS<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeveledGSS")
            .field("top_values", &self.peek_values().len())
            .field("max_depth", &self.max_depth())
            .finish_non_exhaustive()
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
    /// Remap every stack value while pruning any path containing a value for
    /// which `map` returns `None`.
    ///
    /// Unlike `remap_top_values`, this transforms the full shared stack DAG.
    /// Upper/lower node memoization preserves sharing, so callers can project a
    /// composed LR-state GSS into one component's local LR coordinate without
    /// enumerating represented stacks.  When several source values map to the
    /// same destination, their sublanguages are merged exactly.
    pub fn filter_map_values<U, F>(&self, mut map: F) -> LeveledGSS<U, A>
    where
        U: Clone + Eq + Hash,
        F: FnMut(&T) -> Option<U>,
    {
        fn map_value<T, U, F>(
            value: &T,
            value_memo: &mut StdHashMap<T, Option<U>>,
            map: &mut F,
        ) -> Option<U>
        where
            T: Clone + Eq + Hash,
            U: Clone,
            F: FnMut(&T) -> Option<U>,
        {
            if let Some(mapped) = value_memo.get(value) {
                return mapped.clone();
            }
            let mapped = map(value);
            value_memo.insert(value.clone(), mapped.clone());
            mapped
        }

        fn transform_lower<T, U, F>(
            node: &Arc<Lower<T>>,
            value_memo: &mut StdHashMap<T, Option<U>>,
            lower_memo: &mut StdHashMap<usize, Option<Arc<Lower<U>>>>,
            map: &mut F,
        ) -> Option<Arc<Lower<U>>>
        where
            T: Clone + Eq + Hash,
            U: Clone + Eq + Hash,
            F: FnMut(&T) -> Option<U>,
        {
            let key = lower_node_id(node);
            if let Some(cached) = lower_memo.get(&key) {
                return cached.clone();
            }

            let result = match &**node {
                Lower::Segment(segment) => {
                    let mut mapped_values = Vec::with_capacity(segment.values.len());
                    for value in segment.values.iter() {
                        let Some(mapped) = map_value(value, value_memo, map) else {
                            lower_memo.insert(key, None);
                            return None;
                        };
                        mapped_values.push(mapped);
                    }
                    let Some(next) = transform_lower(
                        &segment.next,
                        value_memo,
                        lower_memo,
                        map,
                    ) else {
                        lower_memo.insert(key, None);
                        return None;
                    };
                    Some(new_segment(SV::from_vec(mapped_values), next))
                }
                Lower::General {
                    children, empty, ..
                } => {
                    let mut mapped_children = Children::<U, Lower<U>>::new();
                    for (value, kids) in children.iter() {
                        let Some(mapped_value) = map_value(value, value_memo, map) else {
                            continue;
                        };
                        for child in kids.values() {
                            let Some(mapped_child) = transform_lower(
                                child,
                                value_memo,
                                lower_memo,
                                map,
                            ) else {
                                continue;
                            };
                            insert_lower_child_shared(
                                &mut mapped_children,
                                mapped_value.clone(),
                                mapped_child.max_depth(),
                                mapped_child,
                            );
                        }
                    }
                    if mapped_children.is_empty() && !*empty {
                        None
                    } else {
                        Some(new_lower_precanonicalized(mapped_children, *empty))
                    }
                }
            };
            lower_memo.insert(key, result.clone());
            result
        }

        fn transform_upper<T, U, A, F>(
            node: &Arc<Upper<T, A>>,
            value_memo: &mut StdHashMap<T, Option<U>>,
            lower_memo: &mut StdHashMap<usize, Option<Arc<Lower<U>>>>,
            upper_memo: &mut StdHashMap<usize, Option<Arc<Upper<U, A>>>>,
            map: &mut F,
        ) -> Option<Arc<Upper<U, A>>>
        where
            T: Clone + Eq + Hash,
            U: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            F: FnMut(&T) -> Option<U>,
        {
            let key = Arc::as_ptr(node) as usize;
            if let Some(cached) = upper_memo.get(&key) {
                return cached.clone();
            }
            let result = match &**node {
                Upper::Interface(interface) => transform_lower(
                    &interface.inner,
                    value_memo,
                    lower_memo,
                    map,
                )
                .map(|lower| try_promote(&new_interface(lower, interface.acc.clone()))),
                Upper::Branch(branch) => {
                    let mut mapped_children = Children::<U, Upper<U, A>>::new();
                    for (value, kids) in branch.children.iter() {
                        let Some(mapped_value) = map_value(value, value_memo, map) else {
                            continue;
                        };
                        for child in kids.values() {
                            let Some(mapped_child) = transform_upper(
                                child,
                                value_memo,
                                lower_memo,
                                upper_memo,
                                map,
                            ) else {
                                continue;
                            };
                            insert_upper_child_shared(
                                &mut mapped_children,
                                mapped_value.clone(),
                                mapped_child,
                            );
                        }
                    }
                    if mapped_children.is_empty() && branch.empty.is_none() {
                        None
                    } else {
                        Some(try_promote(&new_branch(
                            mapped_children,
                            branch.empty.clone(),
                        )))
                    }
                }
            };
            upper_memo.insert(key, result.clone());
            result
        }

        let mut value_memo = StdHashMap::new();
        let mut lower_memo = StdHashMap::new();
        let mut upper_memo = StdHashMap::new();
        transform_upper(
            &self.inner,
            &mut value_memo,
            &mut lower_memo,
            &mut upper_memo,
            &mut map,
        )
        .map_or_else(LeveledGSS::empty, |inner| LeveledGSS { inner })
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn ptr_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    #[inline(always)]
    pub fn top_value_count(&self) -> usize {
        self.inner.children_len()
    }

    pub fn single_interface_lower_id(&self) -> Option<usize> {
        match &*self.inner {
            Upper::Interface(i) => Some(lower_node_id(&i.inner)),
            Upper::Branch(_) => None,
        }
    }

    pub fn empty() -> Self {
        empty_upper()
    }

    pub fn from_stacks(stacks: &[(Vec<T>, A)]) -> Self {

        let mut canon: StdHashMap<Vec<T>, A> = StdHashMap::new();
        for (vals, acc) in stacks {
            if let Some(existing) = canon.get_mut(vals) {
                let merged = existing.merge(acc);
                *existing = merged;
            } else {
                canon.insert(vals.clone(), acc.clone());
            }
        }

        struct Entry<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
            end: Option<A>,
            sub: StdHashMap<T, Entry<T, A>>,
        }

        impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Default for Entry<T, A> {
            fn default() -> Self {
                Self {
                    end: None,
                    sub: StdHashMap::new(),
                }
            }
        }

        let mut trie: StdHashMap<T, Entry<T, A>> = StdHashMap::new();
        let mut empty_acc: Option<A> = None;

        for (mut vals, acc) in canon.into_iter() {
            if vals.is_empty() {
                empty_acc = match empty_acc.take() {
                    None => Some(acc),
                    Some(prev) => Some(prev.merge(&acc)),
                };
                continue;
            }

            vals.reverse();
            if let Some(last_val) = vals.pop() {
                let mut node = &mut trie;
                for v in vals {
                    node = &mut node.entry(v).or_default().sub;
                }
                let final_entry = node.entry(last_val).or_default();
                final_entry.end = Some(acc);
            }
        }

        fn intern_lower<T: Clone + Eq + Hash>(
            node: Arc<Lower<T>>,
            pool: &mut Vec<Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            if let Some(existing) = pool.iter().find(|existing| ***existing == *node) {
                return existing.clone();
            }
            pool.push(node.clone());
            node
        }

        fn build_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            d: &StdHashMap<T, Entry<T, A>>,
            pool: &mut Vec<Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            let mut l_children: Children<T, Lower<T>> = CompactMap::new();
            for (v, e) in d.iter() {
                let sub_children = if e.sub.is_empty() {
                    CompactMap::new()
                } else {
                    build_lower(&e.sub, pool).children()
                };
                let node_for_v = intern_lower(new_lower(sub_children, e.end.is_some()), pool);
                l_children.insert(v.clone(), CompactOrdMap::unit(node_for_v.max_depth(), node_for_v));
            }
            intern_lower(new_lower(l_children, false), pool)
        }

        fn build_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            d: &StdHashMap<T, Entry<T, A>>,
            root_empty: Option<A>,
        ) -> Arc<Upper<T, A>> {
            let mut children: Children<T, Upper<T, A>> = CompactMap::new();
            let mut all_child_nodes: Vec<Arc<Upper<T, A>>> = Vec::new();

            for (v, e) in d.iter() {
                let mut nodes_for_v: Vec<Arc<Upper<T, A>>> = Vec::new();
                if let Some(end_acc) = &e.end {
                    let leaf = new_branch(CompactMap::new(), Some(end_acc.clone()));
                    nodes_for_v.push(try_promote(&leaf));
                }
                if !e.sub.is_empty() {
                    nodes_for_v.push(build_upper(&e.sub, None));
                }
                if !nodes_for_v.is_empty() {
                    let mut kids_map: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
                    for n in nodes_for_v.iter() {
                        kids_map.insert(n.max_depth(), n.clone());
                    }
                    children.insert(v.clone(), kids_map);
                    all_child_nodes.extend(nodes_for_v);
                }
            }

            let all_interfaces = all_child_nodes
                .iter()
                .all(|c| matches!(&**c, Upper::Interface(_)));

            if all_interfaces {
                let mut accs: HashSet<A> = HashSet::new();
                for node in &all_child_nodes {
                    if let Upper::Interface(i) = &**node {
                        accs.insert(i.acc.clone());
                    }
                }
                if let Some(e) = &root_empty {
                    accs.insert(e.clone());
                }

                if accs.len() <= 1 {
                    if let Some(the_acc) = accs.into_iter().next() {
                        let mut lower_pool = Vec::new();
                        let lower_tree = build_lower(d, &mut lower_pool);
                        let lower_root = intern_lower(
                            new_lower(lower_tree.children(), root_empty.is_some()),
                            &mut lower_pool,
                        );
                        return new_interface(lower_root, the_acc);
                    } else {
                        return empty_upper_inner();
                    }
                }
            }

            new_branch(children, root_empty)
        }

        LeveledGSS {
            inner: build_upper(&trie, empty_acc),
        }
    }

    pub fn from_single_stack(values: Vec<T>, acc: A) -> Self {
        let floor = new_lower(CompactMap::new(), true);
        let inner = if values.is_empty() {
            new_interface(floor, acc)
        } else {
            new_interface(new_segment(SV::from_vec(values), floor), acc)
        };
        LeveledGSS { inner }
    }

    /// Construct a set of already-sorted, unique one-value stacks sharing one
    /// accumulator. This is the canonical depth-one representation used by
    /// cached direct-regular frontiers.
    pub fn from_sorted_unique_single_value_stacks(values: &[T], acc: A) -> Self
    where
        T: Ord,
    {
        if values.is_empty() {
            return Self::empty();
        }
        debug_assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        if let [value] = values {
            return Self::from_single_stack(vec![value.clone()], acc);
        }

        let floor = new_lower(CompactMap::new(), true);
        let depth = floor.max_depth();
        let child = CompactOrdMap::unit(depth, floor);
        let entries: SmallVec<[(T, CompactOrdMap<Arc<Lower<T>>>); 4]> = values
            .iter()
            .map(|value| (value.clone(), child.clone()))
            .collect();
        Self {
            inner: new_interface(
                new_lower_precanonicalized(CompactMap::Inline(entries), false),
                acc,
            ),
        }
    }

    /// Materialize at most `max_stacks` concrete stacks.
    ///
    /// Returns `None` rather than silently truncating when the represented path
    /// count exceeds the caller-selected limit. This is a diagnostic and test
    /// API; production hot paths should operate on the shared GSS directly or
    /// use a purpose-built bounded traversal.
    pub fn to_stacks(&self, max_stacks: usize) -> Option<Vec<(Vec<T>, A)>> {
        let mut stacks = Vec::new();
        let complete = self.for_each_stack_top_first_bounded(
            max_stacks,
            |top_first, acc| {
                let mut stack = top_first.to_vec();
                stack.reverse();
                stacks.push((stack, acc.clone()));
            },
        );
        complete.then_some(stacks)
    }

    /// Visit concrete stack lengths without materializing stack values.
    ///
    /// Returns `false` as soon as more than `limit` paths are represented. This
    /// is intended for cheap admission checks before an exact bounded stack
    /// traversal. Unlike `for_each_stack_top_first_bounded`, it does not clone
    /// stack values or maintain a path buffer.
    pub fn for_each_stack_len_bounded(
        &self,
        limit: usize,
        mut f: impl FnMut(usize, &A),
    ) -> bool {
        fn emit<A, F>(
            depth: usize,
            acc: &A,
            limit: usize,
            emitted: &mut usize,
            f: &mut F,
        ) -> bool
        where
            F: FnMut(usize, &A),
        {
            if *emitted >= limit {
                return false;
            }
            *emitted += 1;
            f(depth, acc);
            true
        }

        fn dfs_lower<T, A, F>(
            lower: &Lower<T>,
            depth: usize,
            acc: &A,
            limit: usize,
            emitted: &mut usize,
            f: &mut F,
        ) -> bool
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            F: FnMut(usize, &A),
        {
            if lower.empty() && !emit(depth, acc, limit, emitted, f) {
                return false;
            }
            match lower {
                Lower::Segment(seg) => dfs_lower(
                    &seg.next,
                    depth + seg.values.len(),
                    acc,
                    limit,
                    emitted,
                    f,
                ),
                Lower::General { children, .. } => {
                    for kids in children.values() {
                        for child in kids.values() {
                            if !dfs_lower(child, depth + 1, acc, limit, emitted, f) {
                                return false;
                            }
                        }
                    }
                    true
                }
            }
        }

        fn dfs_upper<T, A, F>(
            upper: &Upper<T, A>,
            depth: usize,
            limit: usize,
            emitted: &mut usize,
            f: &mut F,
        ) -> bool
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            F: FnMut(usize, &A),
        {
            match upper {
                Upper::Branch(branch) => {
                    if let Some(acc) = &branch.empty
                        && !emit(depth, acc, limit, emitted, f)
                    {
                        return false;
                    }
                    for kids in branch.children.values() {
                        for child in kids.values() {
                            if !dfs_upper(child, depth + 1, limit, emitted, f) {
                                return false;
                            }
                        }
                    }
                    true
                }
                Upper::Interface(interface) => {
                    if interface.inner.empty()
                        && !emit(depth, &interface.acc, limit, emitted, f)
                    {
                        return false;
                    }
                    match &*interface.inner {
                        Lower::Segment(seg) => dfs_lower(
                            &seg.next,
                            depth + seg.values.len(),
                            &interface.acc,
                            limit,
                            emitted,
                            f,
                        ),
                        Lower::General { children, .. } => {
                            for kids in children.values() {
                                for child in kids.values() {
                                    if !dfs_lower(
                                        child,
                                        depth + 1,
                                        &interface.acc,
                                        limit,
                                        emitted,
                                        f,
                                    ) {
                                        return false;
                                    }
                                }
                            }
                            true
                        }
                    }
                }
            }
        }

        let mut emitted = 0usize;
        dfs_upper(&self.inner, 0, limit, &mut emitted, &mut f)
    }

    /// Visit concrete stacks in top-first order without materializing the full
    /// stack set. Returns `false` as soon as more than `limit` paths are found.
    ///
    /// This is intended for bounded runtime fast paths which already reject
    /// highly ambiguous GSSes. The traversal buffer stays inline for stack
    /// depths up to 64.
    pub fn for_each_stack_top_first_bounded(
        &self,
        limit: usize,
        mut f: impl FnMut(&[T], &A),
    ) -> bool {
        fn emit<T, A, F>(
            pref: &[T],
            acc: &A,
            limit: usize,
            emitted: &mut usize,
            f: &mut F,
        ) -> bool
        where
            F: FnMut(&[T], &A),
        {
            if *emitted >= limit {
                return false;
            }
            *emitted += 1;
            f(pref, acc);
            true
        }

        fn dfs_lower<T, A, F>(
            lower: &Lower<T>,
            pref: &mut SmallVec<[T; 64]>,
            acc: &A,
            limit: usize,
            emitted: &mut usize,
            f: &mut F,
        ) -> bool
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            F: FnMut(&[T], &A),
        {
            if lower.empty() && !emit(pref, acc, limit, emitted, f) {
                return false;
            }
            match lower {
                Lower::Segment(seg) => {
                    for value in seg.values.iter().rev() {
                        pref.push(value.clone());
                    }
                    let complete =
                        dfs_lower(&seg.next, pref, acc, limit, emitted, f);
                    pref.truncate(pref.len() - seg.values.len());
                    complete
                }
                Lower::General { children, .. } => {
                    for (value, kids) in children.iter() {
                        for child in kids.values() {
                            pref.push(value.clone());
                            let complete =
                                dfs_lower(child, pref, acc, limit, emitted, f);
                            pref.pop();
                            if !complete {
                                return false;
                            }
                        }
                    }
                    true
                }
            }
        }

        fn dfs_upper<T, A, F>(
            upper: &Upper<T, A>,
            pref: &mut SmallVec<[T; 64]>,
            limit: usize,
            emitted: &mut usize,
            f: &mut F,
        ) -> bool
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            F: FnMut(&[T], &A),
        {
            match upper {
                Upper::Branch(branch) => {
                    if let Some(acc) = &branch.empty
                        && !emit(pref, acc, limit, emitted, f)
                    {
                        return false;
                    }
                    for (value, kids) in branch.children.iter() {
                        for child in kids.values() {
                            pref.push(value.clone());
                            let complete = dfs_upper(child, pref, limit, emitted, f);
                            pref.pop();
                            if !complete {
                                return false;
                            }
                        }
                    }
                    true
                }
                Upper::Interface(interface) => {
                    if interface.inner.empty()
                        && !emit(pref, &interface.acc, limit, emitted, f)
                    {
                        return false;
                    }
                    match &*interface.inner {
                        Lower::Segment(seg) => {
                            for value in seg.values.iter().rev() {
                                pref.push(value.clone());
                            }
                            let complete = dfs_lower(
                                &seg.next,
                                pref,
                                &interface.acc,
                                limit,
                                emitted,
                                f,
                            );
                            pref.truncate(pref.len() - seg.values.len());
                            complete
                        }
                        Lower::General { children, .. } => {
                            for (value, kids) in children.iter() {
                                for child in kids.values() {
                                    pref.push(value.clone());
                                    let complete = dfs_lower(
                                        child,
                                        pref,
                                        &interface.acc,
                                        limit,
                                        emitted,
                                        f,
                                    );
                                    pref.pop();
                                    if !complete {
                                        return false;
                                    }
                                }
                            }
                            true
                        }
                    }
                }
            }
        }

        let mut pref = SmallVec::<[T; 64]>::new();
        let mut emitted = 0usize;
        dfs_upper(&self.inner, &mut pref, limit, &mut emitted, &mut f)
    }

    /// Compare the concrete stack/accumulator set represented by two GSSes,
    /// independent of their internal sharing or node layout.
    ///
    /// This intentionally materializes stacks and is meant for validation and
    /// diagnostics, not production hot paths. `PartialEq` remains structural.
    pub fn semantically_eq(&self, other: &Self, max_stacks: usize) -> Option<bool> {
        if self == other {
            return Some(true);
        }
        let left = self.to_stacks(max_stacks)?;
        let right = other.to_stacks(max_stacks)?;
        Some(
            left.len() == right.len()
                && left.iter().all(|entry| right.contains(entry))
                && right.iter().all(|entry| left.contains(entry)),
        )
    }

    /// Apply a set of stack effects by materializing the single concrete stack.
    ///
    /// This is only a win for already-deterministic parser states that have a
    /// large `StackShifts` action. In that shape, the generic GSS branch builder
    /// can spend most of its time constructing and merging branches that collapse
    /// back to one or two concrete stacks.
    pub fn apply_stack_effects_to_single_concrete_path<'a, I>(
        &self,
        effects: I,
        max_materialized_depth: usize,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = (usize, &'a [T])>,
        T: 'a,
    {
        let effects: SmallVec<[(usize, &'a [T]); 8]> = effects.into_iter().collect();

        if let Some(stack) = self.try_virtual_stack()
            && (!stack.has_hidden_floor_values()
                || effects.iter().all(|(pop, _)| *pop <= stack.len()))
        {
            let mut out: Option<Self> = None;
            for &(pop, pushes) in &effects {
                let mut branch = stack.clone();
                if branch.pop(pop) != 0 {
                    continue;
                }
                for value in pushes {
                    branch.push(value.clone());
                }
                let branch = branch.into_gss();
                out = Some(match out {
                    Some(existing) => existing.merge(&branch),
                    None => branch,
                });
            }
            if let Some(out) = out {
                return Some(out);
            }
            let empty: Vec<(Vec<T>, A)> = Vec::new();
            return Some(Self::from_stacks(&empty));
        }

        let (stack, acc) = self.try_single_stack_bounded(max_materialized_depth)?;

        let mut out: Vec<(Vec<T>, A)> = Vec::new();
        for (pop, pushes) in effects {
            if pop > stack.len() {
                continue;
            }

            let keep = stack.len() - pop;
            let mut next = Vec::with_capacity(keep + pushes.len());
            next.extend_from_slice(&stack[..keep]);
            next.extend_from_slice(pushes);

            if let Some((_, existing_acc)) =
                out.iter_mut().find(|(existing_stack, _)| *existing_stack == next)
            {
                *existing_acc = existing_acc.merge(&acc);
            } else {
                out.push((next, acc.clone()));
            }
        }

        match out.len() {
            1 => {
                let (stack, acc) = out.pop().unwrap();
                Some(Self::from_single_stack(stack, acc))
            }
            _ => Some(Self::from_stacks(&out)),
        }
    }

    pub fn apply_shared_pop_push_branches<'a, I>(
        &self,
        pop: usize,
        pushes: I,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        self.try_virtual_stack()?
            .into_gss_after_popping_and_pushing_branches(pop, pushes)
    }

    pub fn apply_shared_pop_push_single_branches<'a, I>(
        &self,
        pop: usize,
        targets: I,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        self.try_virtual_stack()?
            .into_gss_after_popping_and_pushing_single_branches(pop, targets)
    }

    pub fn apply_guarded_stack_effects_to_single_concrete_path<'a, I, G>(
        &self,
        effects: I,
        max_materialized_depth: usize,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = (G, usize, &'a [T])>,
        G: IntoIterator<Item = (usize, &'a [T])>,
        T: 'a,
    {
        let (stack, acc) = self.try_single_stack_bounded(max_materialized_depth)?;

        let mut out: Vec<(Vec<T>, A)> = Vec::new();
        for (guards, pop, pushes) in effects {
            if pop > stack.len() {
                continue;
            }

            let mut allowed = true;
            for (guard_pop, guard_states) in guards {
                if guard_pop >= stack.len() {
                    allowed = false;
                    break;
                }
                let state = &stack[stack.len() - 1 - guard_pop];
                if !guard_states.iter().any(|candidate| candidate == state) {
                    allowed = false;
                    break;
                }
            }
            if !allowed {
                continue;
            }

            let keep = stack.len() - pop;
            let mut next = Vec::with_capacity(keep + pushes.len());
            next.extend_from_slice(&stack[..keep]);
            next.extend_from_slice(pushes);

            if let Some((_, existing_acc)) =
                out.iter_mut().find(|(existing_stack, _)| *existing_stack == next)
            {
                *existing_acc = existing_acc.merge(&acc);
            } else {
                out.push((next, acc.clone()));
            }
        }

        match out.len() {
            1 => {
                let (stack, acc) = out.pop().unwrap();
                Some(Self::from_single_stack(stack, acc))
            }
            _ => Some(Self::from_stacks(&out)),
        }
    }

    pub fn push(&self, value: T) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        let new_inner = match &*self.inner {
            Upper::Interface(i) => {
                let new_lower_root = new_segment(
                    SV::unit(value),
                    i.inner.clone(),
                );
                new_interface(new_lower_root, i.acc.clone())
            }
            Upper::Branch(_) => {
                let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
                new_children.insert(
                    value,
                    CompactOrdMap::unit(self.inner.max_depth(), self.inner.clone()),
                );
                new_branch(new_children, None)
            }
        };
        LeveledGSS { inner: new_inner }
    }

    /// Equivalent to merging `self.isolate(Some(from)).push(to)` for each
    /// `(from, to)` pair, but avoids repeated isolate/push/merge churn by
    /// rebuilding the shifted top layer in one pass.
    /// Apply a bulk remapping to the current frontier values.
    pub fn remap_top_values<I>(&self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let pairs: SmallVec<[(T, T); 8]> = shifts.into_iter().collect();
        if pairs.is_empty() {
            return Self::empty();
        }
        if pairs.len() == 1 {
            let (ref from, ref to) = pairs[0];
            if self.single_exclusive_top_value().as_ref() == Some(from) {
                return self.push(to.clone());
            }
        }

        match &*self.inner {
            Upper::Interface(i) => {
                // Use SmallVec instead of HashMap for grouping by target.
                // Linear scan is faster than HashMap for typical ≤8 shift pairs.
                let mut children_by_target: SmallVec<[(T, Children<T, Lower<T>>); 4]> = SmallVec::new();

                // Build a Segment-aware lookup closure
                let inner_children;
                let seg_entry: Option<(&T, CompactOrdMap<Arc<Lower<T>>>)>;
                match &*i.inner {
                    Lower::Segment(seg) => {
                        let top_val = seg.values.last().unwrap();
                        let rest = i.inner.segment_rest_arc();
                        inner_children = None;
                        seg_entry = Some((top_val, CompactOrdMap::unit(rest.max_depth(), rest)));
                    }
                    Lower::General { children, .. } => {
                        inner_children = Some(children);
                        seg_entry = None;
                    }
                }

                for (from, to) in pairs.iter().cloned() {
                    let kids_opt = if let Some((cv, ref ck)) = seg_entry {
                        if *cv == from { Some(ck) } else { None }
                    } else {
                        inner_children.unwrap().get(&from)
                    };
                    let Some(kids) = kids_opt else {
                        continue;
                    };
                    // Find or create the target entry
                    let target_children = if let Some(pos) = children_by_target.iter().position(|(t, _)| *t == to) {
                        &mut children_by_target[pos].1
                    } else {
                        children_by_target.push((to, CompactMap::new()));
                        &mut children_by_target.last_mut().unwrap().1
                    };
                    match target_children.get(&from) {
                        Some(existing_kids) => {
                            let mut merged_kids = existing_kids.clone();
                            for (depth, child) in kids.iter() {
                                if let Some(existing_child) = merged_kids.get(depth) {
                                    merged_kids.insert(*depth, merge_lower(existing_child, child));
                                } else {
                                    merged_kids.insert(*depth, child.clone());
                                }
                            }
                            target_children.insert(from, merged_kids);
                        }
                        None => {
                            target_children.insert(from, kids.clone());
                        }
                    }
                }

                if children_by_target.is_empty() {
                    return Self::empty();
                }

                let mut shifted_children: Children<T, Lower<T>> = CompactMap::new();
                for (to, lower_children) in children_by_target {
                    let lower = new_lower(lower_children, false);
                    shifted_children.insert(to, CompactOrdMap::unit(lower.max_depth(), lower));
                }

                let shifted_root = new_lower(shifted_children, false);
                LeveledGSS {
                    inner: new_interface(shifted_root, i.acc.clone()),
                }
            }
            Upper::Branch(_) => {
                let shifted = pairs
                    .into_iter()
                    .map(|(from, to)| self.isolate(Some(from)).push(to));
                Self::merge_many(shifted)
            }
        }
    }

    /// Like `remap_top_values` but takes ownership, allowing extraction of
    /// children by move instead of clone when the Arcs are uniquely owned.
    pub fn remap_top_values_owned<I>(self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let pairs: SmallVec<[(T, T); 8]> = shifts.into_iter().collect();
        if pairs.is_empty() {
            return Self::empty();
        }
        if pairs.len() == 1 {
            let (ref from, ref to) = pairs[0];
            if self.single_exclusive_top_value().as_ref() == Some(from) {
                return self.push(to.clone());
            }
        }

        // Try to extract children by move if we have unique ownership
        let (acc, mut children) = match Arc::try_unwrap(self.inner) {
            Ok(Upper::Interface(iface_arc)) => {
                match Arc::try_unwrap(iface_arc) {
                    Ok(Interface { inner: lower_arc, acc }) => {
                        match Arc::try_unwrap(lower_arc) {
                            Ok(lower) => {
                                let (c, _empty, _md) = lower.into_parts();
                                (acc, c)
                            }
                            Err(lower_arc) => {
                                // Can't unwrap lower, fall back to clone path
                                let i = Interface { inner: lower_arc, acc: acc.clone() };
                                let gss = LeveledGSS { inner: Arc::new(Upper::Interface(Arc::new(i))) };
                                return gss.remap_top_values(pairs);
                            }
                        }
                    }
                    Err(iface_arc) => {
                        let gss = LeveledGSS { inner: Arc::new(Upper::Interface(iface_arc)) };
                        return gss.remap_top_values(pairs);
                    }
                }
            }
            Ok(upper @ Upper::Branch(_)) => {
                let gss = LeveledGSS { inner: Arc::new(upper) };
                return gss.remap_top_values(pairs);
            }
            Err(arc) => {
                let gss = LeveledGSS { inner: arc };
                return gss.remap_top_values(pairs);
            }
        };

        // We own `children` by value — extract entries without cloning
        let mut children_by_target: SmallVec<[(T, Children<T, Lower<T>>); 4]> = SmallVec::new();

        for (from, to) in pairs {
            let Some(kids) = children.remove(&from) else {
                continue;
            };
            let target_children = if let Some(pos) = children_by_target.iter().position(|(t, _)| *t == to) {
                &mut children_by_target[pos].1
            } else {
                children_by_target.push((to, CompactMap::new()));
                &mut children_by_target.last_mut().unwrap().1
            };
            match target_children.get(&from) {
                Some(existing_kids) => {
                    let mut merged_kids = existing_kids.clone();
                    for (depth, child) in kids.iter() {
                        if let Some(existing_child) = merged_kids.get(depth) {
                            merged_kids.insert(*depth, merge_lower(existing_child, child));
                        } else {
                            merged_kids.insert(*depth, child.clone());
                        }
                    }
                    target_children.insert(from, merged_kids);
                }
                None => {
                    // Move kids directly — no clone needed
                    target_children.insert(from, kids);
                }
            }
        }

        if children_by_target.is_empty() {
            return Self::empty();
        }

        let mut shifted_children: Children<T, Lower<T>> = CompactMap::new();
        for (to, lower_children) in children_by_target {
            let lower = new_lower(lower_children, false);
            shifted_children.insert(to, CompactOrdMap::unit(lower.max_depth(), lower));
        }

        let shifted_root = new_lower(shifted_children, false);
        LeveledGSS {
            inner: new_interface(shifted_root, acc),
        }
    }

    /// Apply pure top-frontier shifts in one pass. Each tuple is
    /// `(current_top, target_top, replace_top)`.
    pub fn try_apply_selective_top_pure_shifts<I>(&self, shifts: I) -> Option<Self>
    where
        I: IntoIterator<Item = (T, T, bool)>,
    {
        let shifts: SmallVec<[(T, T, bool); 2]> = shifts.into_iter().collect();
        let [(from, to, replace_top)] = shifts.as_slice() else {
            return None;
        };

        let Upper::Interface(i) = &*self.inner else {
            return None;
        };
        let Lower::General { children, .. } = &*i.inner else {
            return None;
        };
        let kids = children.get(from)?;

        fn lower_with_top<T: Clone + Eq + Hash>(
            top: T,
            kids: &CompactOrdMap<Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            if kids.len() == 1 {
                let (_, child) = kids.iter().next().unwrap();
                new_segment(SV::unit(top), child.clone())
            } else {
                new_lower(CompactMap::unit(top, kids.clone()), false)
            }
        }

        let shifted_root = if *replace_top {
            lower_with_top(to.clone(), kids)
        } else {
            new_segment(SV::unit(to.clone()), lower_with_top(from.clone(), kids))
        };

        Some(LeveledGSS {
            inner: new_interface(shifted_root, i.acc.clone()),
        })
    }

    pub fn apply_top_pure_shifts<I>(&self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T, bool)>,
    {
        let shifts: SmallVec<[(T, T, bool); 8]> = shifts.into_iter().collect();
        if shifts.is_empty() {
            return Self::empty();
        }

        match &*self.inner {
            Upper::Interface(i) => {
                let inner_children;
                let seg_entry: Option<(&T, CompactOrdMap<Arc<Lower<T>>>)>;
                match &*i.inner {
                    Lower::Segment(seg) => {
                        let top_val = seg.values.last().unwrap();
                        let rest = i.inner.segment_rest_arc();
                        inner_children = None;
                        seg_entry = Some((top_val, CompactOrdMap::unit(rest.max_depth(), rest)));
                    }
                    Lower::General { children, .. } => {
                        inner_children = Some(children);
                        seg_entry = None;
                    }
                }

                let mut shifted_children: Children<T, Lower<T>> = CompactMap::new();
                for (from, to, replace_top) in shifts {
                    let kids_opt = if let Some((seg_top, ref seg_kids)) = seg_entry {
                        if *seg_top == from { Some(seg_kids) } else { None }
                    } else {
                        inner_children.unwrap().get(&from)
                    };
                    let Some(kids) = kids_opt else {
                        continue;
                    };

                    if replace_top {
                        for (depth, child) in kids.iter() {
                            insert_lower_child_shared(&mut shifted_children, to.clone(), *depth, child.clone());
                        }
                    } else {
                        let mut pushed_children: Children<T, Lower<T>> = CompactMap::new();
                        pushed_children.insert(from, kids.clone());
                        let pushed_child = new_lower(pushed_children, false);
                        insert_lower_child_shared(
                            &mut shifted_children,
                            to,
                            pushed_child.max_depth(),
                            pushed_child,
                        );
                    }
                }

                if shifted_children.is_empty() {
                    return Self::empty();
                }
                let shifted_root = new_lower(shifted_children, false);
                LeveledGSS {
                    inner: new_interface(shifted_root, i.acc.clone()),
                }
            }
            Upper::Branch(_) => {
                let shifted = shifts.into_iter().map(|(from, to, replace_top)| {
                    let base = self.isolate(Some(from));
                    if replace_top {
                        base.popn(1).push(to)
                    } else {
                        base.push(to)
                    }
                });
                Self::merge_many(shifted)
            }
        }
    }

    /// Absorb `base.push(value)` into `self`, using an in-place Interface merge
    /// when both sides carry the same accumulator annotation.
    ///
    /// The accumulator check is semantically required: inserting a lower child
    /// into an Interface labels that child with the Interface's accumulator.
    /// Different annotations must therefore use the ordinary GSS merge path to
    /// preserve stack/accumulator correlation.
    pub fn absorb_push_same_acc(self, value: T, base: &Self) -> Self {
        if base.is_empty() {
            return self;
        }
        if self.is_empty() {
            return base.push(value);
        }

        let compatible_base = match (&*self.inner, &*base.inner) {
            (Upper::Interface(self_iface), Upper::Interface(base_iface))
                if self_iface.acc == base_iface.acc =>
            {
                Some(base_iface.clone())
            }
            _ => None,
        };
        if let Some(base_iface) = compatible_base {
            return self.absorb_push_interface_inplace(value, &base_iface);
        }

        self.merge(&base.push(value))
    }

    pub fn absorb_vstack_same_acc(mut self, stack: &VirtualStack<T, A>) -> Self {
        let mut stack = stack.clone();
        stack.flush_pending();

        if stack.values.is_empty() {
            return self;
        }
        if self.is_empty() {
            return stack.into_gss();
        }

        let top = stack.values.last().unwrap().clone();
        let child_node = if stack.values.len() == 1 {
            stack.next.clone()
        } else {
            new_segment(stack.values.take(stack.values.len() - 1), stack.next.clone())
        };
        let child_depth = child_node.max_depth();

        let inner_mut = Arc::make_mut(&mut self.inner);
        if let Upper::Interface(self_iface_arc) = inner_mut {
            let iface_mut = Arc::make_mut(self_iface_arc);
            if iface_mut.acc == stack.acc {
                let lower_mut = Arc::make_mut(&mut iface_mut.inner);
                lower_mut.ensure_general();
                match lower_mut {
                    Lower::General { children, max_depth, .. } => {
                        insert_lower_child_shared(children, top, child_depth, child_node);

                        if child_depth + 1 > *max_depth {
                            *max_depth = child_depth + 1;
                        }
                    }
                    Lower::Segment(_) => unreachable!(),
                }
                return self;
            }
        }

        self.merge(&stack.into_gss())
    }

    pub fn absorb_vstack_same_acc_owned(mut self, mut stack: VirtualStack<T, A>) -> Self {
        stack.flush_pending();

        if stack.values.is_empty() {
            return self;
        }
        if self.is_empty() {
            return stack.into_gss();
        }

        let top = stack.values.last().unwrap().clone();
        let child_node = if stack.values.len() == 1 {
            stack.next.clone()
        } else {
            new_segment(stack.values.take(stack.values.len() - 1), stack.next.clone())
        };
        let child_depth = child_node.max_depth();

        let inner_mut = Arc::make_mut(&mut self.inner);
        if let Upper::Interface(self_iface_arc) = inner_mut {
            let iface_mut = Arc::make_mut(self_iface_arc);
            if iface_mut.acc == stack.acc {
                let lower_mut = Arc::make_mut(&mut iface_mut.inner);
                lower_mut.ensure_general();
                match lower_mut {
                    Lower::General { children, max_depth, .. } => {
                        insert_lower_child_shared(children, top, child_depth, child_node);

                        if child_depth + 1 > *max_depth {
                            *max_depth = child_depth + 1;
                        }
                    }
                    Lower::Segment(_) => unreachable!(),
                }
                return self;
            }
        }

        self.merge(&stack.into_gss())
    }

    fn absorb_push_interface_inplace(
        mut self,
        value: T,
        base_iface: &Arc<Interface<T, A>>,
    ) -> Self {
        let child_depth = base_iface.inner.max_depth();
        let child_node = base_iface.inner.clone();

        let inner_mut = Arc::make_mut(&mut self.inner);
        let Upper::Interface(self_iface_arc) = inner_mut else {
            unreachable!("absorb_push_interface_inplace requires an Interface receiver");
        };
        let iface_mut = Arc::make_mut(self_iface_arc);
        debug_assert!(iface_mut.acc == base_iface.acc);
        let lower_mut = Arc::make_mut(&mut iface_mut.inner);
        // Always convert to General for in-place mutation
        lower_mut.ensure_general();
        match lower_mut {
            Lower::General { children, max_depth, .. } => {
                insert_lower_child_shared(children, value, child_depth, child_node);

                if child_depth + 1 > *max_depth {
                    *max_depth = child_depth + 1;
                }
            }
            Lower::Segment(_) => unreachable!(),
        }

        self
    }

    pub fn popn(&self, n: isize) -> Self {
        if n <= 0 {
            return self.clone();
        }
        if self.is_empty() {
            return self.clone();
        }
        if let Some(fast) = self.popn_single_interface_path(n) {
            return fast;
        }

        let mut memo_upper: StdHashMap<(usize, isize), Arc<Upper<T, A>>> = StdHashMap::new();
        let mut memo_lower: StdHashMap<(usize, isize), Arc<Lower<T>>> = StdHashMap::new();

        fn popn_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Lower<T>>,
            k: isize,
            memo_lower: &mut StdHashMap<(usize, isize), Arc<Lower<T>>>,
        ) -> Arc<Lower<T>> {
            if k == 0 {
                return node.clone();
            }
            let node_id = match &**node {
                Lower::Segment(seg) => Arc::as_ptr(seg) as usize,
                _ => Arc::as_ptr(node) as usize,
            };
            let key = (node_id, k);
            if let Some(cached) = memo_lower.get(&key) {
                return cached.clone();
            }

            // Segment fast path: pop through the whole segment at once
            let merged: Option<Arc<Lower<T>>> = if node.is_segment() {
                let values = node.segment_values();
                let seg_len = values.len() as isize;
                if k >= seg_len {
                    // Pop past entire segment
                    let next_arc = node.segment_next().clone();
                    let popped = popn_lower::<T, A>(&next_arc, k - seg_len, memo_lower);
                    Some(popped)
                } else {
                    // Pop within segment: create shorter segment with remaining values
                    let keep = (seg_len - k) as usize;
                    let new_values = values.take(keep);
                    let next = node.segment_next();
                    Some(new_segment(new_values, next.clone()))
                }
            } else {
                let mut m: Option<Arc<Lower<T>>> = None;
                if let Lower::General { children, .. } = &**node {
                    for child in children.values().flat_map(|kids| kids.values()) {
                        let popped_child = popn_lower::<T, A>(child, k - 1, memo_lower);
                        m = Some(match m {
                            Some(acc) => merge_lower(&acc, &popped_child),
                            None => popped_child,
                        });
                    }
                }
                m
            };

            let mut res = merged.unwrap_or_else(|| new_lower(CompactMap::new(), false));

            if node.empty() && k == 1 {
                let terminal_node = new_lower(CompactMap::new(), true);
                res = merge_lower(&res, &terminal_node);
            }

            memo_lower.insert(key, res.clone());
            res
        }

        fn popn_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Upper<T, A>>,
            k: isize,
            memo_upper: &mut StdHashMap<(usize, isize), Arc<Upper<T, A>>>,
            memo_lower: &mut StdHashMap<(usize, isize), Arc<Lower<T>>>,
        ) -> Arc<Upper<T, A>> {
            if k == 0 {
                return node.clone();
            }
            let key = (Arc::as_ptr(node) as usize, k);
            if let Some(cached) = memo_upper.get(&key) {
                return cached.clone();
            }

            let res = match &**node {
                Upper::Branch(b) => {
                    let mut merged: Option<Arc<Upper<T, A>>> = None;
                    for kids in b.children.values() {
                        for child in kids.values() {
                            let popped_child = popn_upper(child, k - 1, memo_upper, memo_lower);
                            merged = Some(match merged {
                                Some(acc) => merge_upper(&acc, &popped_child),
                                None => popped_child,
                            });
                        }
                    }

                    if let Some(acc) = &b.empty {
                        if k == 1 {
                            let terminal_lower = new_lower(CompactMap::new(), true);
                            let terminal_upper = new_interface(terminal_lower, acc.clone());
                            merged = Some(match merged {
                                Some(current) => merge_upper(&current, &terminal_upper),
                                None => terminal_upper,
                            });
                        }
                    }

                    if let Some(merged) = merged {
                        try_promote(&merged)
                    } else {
                        empty_upper_inner()
                    }
                }
                Upper::Interface(i) => {
                    let popped_lower = popn_lower::<T, A>(&i.inner, k, memo_lower);
                    if popped_lower.children_is_empty() && !popped_lower.empty() {
                        empty_upper_inner()
                    } else {
                        new_interface(popped_lower, i.acc.clone())
                    }
                }
            };

            memo_upper.insert(key, res.clone());
            res
        }

        let new_inner = popn_upper::<T, A>(&self.inner, n, &mut memo_upper, &mut memo_lower);
        LeveledGSS { inner: new_inner }
    }

    fn popn_single_interface_path(&self, n: isize) -> Option<Self> {
        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };

        let mut current = interface.inner.clone();
        let mut remaining = n;
        while remaining > 0 {
            let next_child: Option<Arc<Lower<T>>> = if current.is_segment() {
                let values = current.segment_values();
                if values.len() as isize <= remaining {
                    // Decrement by (len-1) here; the loop bottom adds the final -1
                    remaining -= values.len() as isize - 1;
                    Some(current.segment_next().clone())
                } else {
                    // Would land inside segment — can't use this fast path
                    return None;
                }
            } else {
                match current.children_len() {
                    0 => None,
                    1 => {
                        if let Lower::General { children, .. } = &*current {
                            let kids = children.values().next().expect("single child entry");
                            if kids.len() != 1 {
                                return None;
                            }
                            Some(kids.values().next().expect("single child node").clone())
                        } else {
                            None
                        }
                    }
                    _ => return None,
                }
            };

            if remaining == 1 {
                let mut result = next_child
                    .unwrap_or_else(|| new_lower(CompactMap::new(), false));
                if current.empty() {
                    let terminal = new_lower(CompactMap::new(), true);
                    result = merge_lower(&result, &terminal);
                }

                if result.children_is_empty() && !result.empty() {
                    return Some(Self::empty());
                }

                return Some(Self {
                    inner: new_interface(result, interface.acc.clone()),
                });
            }

            let Some(child) = next_child else {
                return Some(Self::empty());
            };
            current = child;
            remaining -= 1;
        }

        if current.children_is_empty() && !current.empty() {
            Some(Self::empty())
        } else {
            Some(Self {
                inner: new_interface(current, interface.acc.clone()),
            })
        }
    }

    pub fn pop(&self) -> Self {
        self.popn(1)
    }

    /// Return the stack obtained by keeping only top-level paths whose top value
    /// is `value`, then popping that top value.
    ///
    /// This is equivalent to `self.isolate(Some(value.clone())).popn(1)`, but it
    /// avoids rebuilding the isolated top layer for the common Interface/General
    /// frontier shape produced by small GLR waves.
    pub fn pop_top_value(&self, value: &T) -> Self {
        match &*self.inner {
            Upper::Interface(interface) => {
                let popped = match &*interface.inner {
                    Lower::Segment(_) => {
                        if interface.inner.segment_top_value() == value {
                            interface.inner.segment_rest_arc()
                        } else {
                            return Self::empty();
                        }
                    }
                    Lower::General { children, .. } => {
                        let Some(kids) = children.get(value) else {
                            return Self::empty();
                        };
                        let mut iter = kids.values();
                        let Some(first) = iter.next() else {
                            return Self::empty();
                        };
                        iter.fold(first.clone(), |acc, child| merge_lower(&acc, child))
                    }
                };
                if popped.children_is_empty() && !popped.empty() {
                    Self::empty()
                } else {
                    Self {
                        inner: new_interface(popped, interface.acc.clone()),
                    }
                }
            }
            Upper::Branch(_) => self.isolate(Some(value.clone())).popn(1),
        }
    }

    /// Fast path for a top-level interface whose alternatives all share the
    /// same base after popping one value.
    pub fn pop1_common_interface_base(&self) -> Option<Self> {
        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };
        let Lower::General {
            children,
            empty: false,
            ..
        } = &*interface.inner
        else {
            return None;
        };
        if children.len() < 2 {
            return None;
        }

        let mut common_child: Option<Arc<Lower<T>>> = None;
        let mut common_child_id: Option<usize> = None;
        for kids in children.values() {
            if kids.len() != 1 {
                return None;
            }
            let child = kids.values().next().expect("single child");
            let child_id = lower_node_id(child);
            match common_child_id {
                None => {
                    common_child = Some(child.clone());
                    common_child_id = Some(child_id);
                }
                Some(id) if id == child_id => {}
                Some(_) => return None,
            }
        }

        Some(Self {
            inner: new_interface(common_child?, interface.acc.clone()),
        })
    }

    /// Like `decompose_and_pop` but invokes a callback for each (value, popped_gss) pair
    /// instead of allocating a Vec. Avoids heap allocation for the common single-element case.
    pub fn for_each_decomposed(&self, mut f: impl FnMut(T, Self)) {
        match &*self.inner {
            Upper::Branch(b) => {
                for (val, kids) in b.children.iter() {
                    let m = if kids.len() == 1 {
                        kids.values().next().unwrap().clone()
                    } else {
                        let mut it = kids.values();
                        let mut acc = it.next().unwrap().clone();
                        for child in it {
                            acc = merge_upper(&acc, child);
                        }
                        acc
                    };
                    let inner = if matches!(&*m, Upper::Interface(_)) {
                        m
                    } else {
                        try_promote(&m)
                    };
                    let is_empty = matches!(&*inner,
                        Upper::Branch(b) if b.children.is_empty() && b.empty.is_none());
                    if !is_empty {
                        f(val.clone(), LeveledGSS { inner });
                    }
                }
            }
            Upper::Interface(i) => {
                if i.inner.is_segment() {
                    let val = i.inner.segment_top_value().clone();
                    let lower = i.inner.segment_rest_arc();
                    if !lower.children_is_empty() || lower.empty() {
                        let upper = new_interface(lower, i.acc.clone());
                        f(val, LeveledGSS { inner: upper });
                    }
                    return;
                }
                if let Lower::General { children, .. } = &*i.inner {
                    for (val, kids) in children.iter() {
                        let lower = if kids.len() == 1 {
                            kids.values().next().unwrap().clone()
                        } else {
                            let mut it = kids.values();
                            let mut acc = it.next().unwrap().clone();
                            for child in it {
                                acc = merge_lower(&acc, child);
                            }
                            acc
                        };
                        if !lower.children_is_empty() || lower.empty() {
                            let upper = new_interface(lower, i.acc.clone());
                            f(val.clone(), LeveledGSS { inner: upper });
                        }
                    }
                }
            }
        }
    }

    /// Try to view the top of this GSS as a flat virtual stack.
    /// Succeeds when the GSS is an Interface whose top is a chain of Segment nodes.
    /// The chain is extracted until a non-Segment node is hit — that node becomes
    /// the "floor". The floor can be a General with splits, an empty terminal, etc.
    ///
    /// Returns `None` if the GSS is not an Interface whose top node is a Segment.
    pub fn try_virtual_stack(&self) -> Option<VirtualStack<T, A>> {
        let interface = match &*self.inner {
            Upper::Interface(iface) => iface,
            _ => return None,
        };
        let (values, next) = match &*interface.inner {
            Lower::Segment(seg) => (seg.values.clone(), seg.next.clone()),
            _ => return None,
        };
        Some(VirtualStack { values, next, acc: interface.acc.clone(), pending_top: None })
    }

    /// Copy the complete sole stack into preallocated scratch. Unary General
    /// nodes are traversed exactly; branching or an optional early end declines.
    pub fn single_path_acc(&self) -> Option<A> {
        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };
        self.single_path_terminal_floor()?;
        Some(interface.acc.clone())
    }

    pub fn copy_single_path_stack_into(&self, out: &mut Vec<T>) -> bool {
        let Upper::Interface(interface) = &*self.inner else {
            return false;
        };
        out.clear();
        let mut node = &interface.inner;
        loop {
            match &**node {
                Lower::Segment(segment) => {
                    if out.len() + segment.values.len() > out.capacity() {
                        out.clear();
                        return false;
                    }
                    out.extend(segment.values.iter().rev().cloned());
                    node = &segment.next;
                }
                Lower::General {
                    children, empty, ..
                } => {
                    if *empty {
                        if !children.is_empty() {
                            out.clear();
                            return false;
                        }
                        out.reverse();
                        return !out.is_empty();
                    }
                    if children.len() != 1 {
                        out.clear();
                        return false;
                    }
                    let (value, kids) = children.iter().next().unwrap();
                    if kids.len() != 1 || out.len() == out.capacity() {
                        out.clear();
                        return false;
                    }
                    out.push(value.clone());
                    node = kids.values().next().unwrap();
                }
            }
        }
    }

    fn single_path_terminal_floor(&self) -> Option<Arc<Lower<T>>> {
        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };
        let mut node = &interface.inner;
        loop {
            match &**node {
                Lower::Segment(segment) => node = &segment.next,
                Lower::General {
                    children, empty, ..
                } => {
                    if *empty {
                        return children.is_empty().then(|| node.clone());
                    }
                    if children.len() != 1 {
                        return None;
                    }
                    let kids = children.values().next().unwrap();
                    if kids.len() != 1 {
                        return None;
                    }
                    node = kids.values().next().unwrap();
                }
            }
        }
    }

    pub fn can_replace_single_path_state_in_place(&self, values: &[T]) -> bool {
        if values.is_empty() || self.single_path_terminal_floor().is_none() {
            return false;
        }
        if Arc::strong_count(&self.inner) != 1 {
            return false;
        }
        let Upper::Interface(interface) = &*self.inner else {
            return false;
        };
        if Arc::strong_count(interface) != 1 || Arc::strong_count(&interface.inner) != 1 {
            return false;
        }
        let Lower::Segment(segment) = &*interface.inner else {
            return false;
        };
        Arc::strong_count(segment) == 1
            && segment.rest.get().is_none()
            && segment.values.can_push_without_alloc(values.len())
    }

    /// Replace a uniquely-owned sole stack and accumulator, reusing the reserved
    /// top Segment and existing terminal floor.
    pub fn try_replace_single_path_state_in_place(
        &mut self,
        values: &[T],
        acc: A,
    ) -> bool {
        if !self.can_replace_single_path_state_in_place(values) {
            return false;
        }
        let floor = self
            .single_path_terminal_floor()
            .expect("replacement eligibility checked terminal floor");
        let upper = Arc::get_mut(&mut self.inner).expect("replacement eligibility checked upper");
        let Upper::Interface(interface) = upper else {
            unreachable!();
        };
        let interface = Arc::get_mut(interface).expect("replacement eligibility checked interface");
        let lower = Arc::get_mut(&mut interface.inner).expect("replacement eligibility checked lower");
        let Lower::Segment(segment) = lower else {
            unreachable!();
        };
        let segment = Arc::get_mut(segment).expect("replacement eligibility checked segment");

        segment.values.truncate(0);
        for value in values {
            if !segment.values.try_push(value.clone()) {
                unreachable!("replacement capacity checked before mutation");
            }
        }
        segment.next = floor;
        segment.max_depth = segment.next.max_depth() + values.len() as u32;
        segment.segments_len = segment.next.segments_len() + values.len();
        interface.acc = acc;
        true
    }

    pub fn try_replace_single_path_stack_in_place(&mut self, values: &[T]) -> bool {
        let Upper::Interface(interface) = &*self.inner else {
            return false;
        };
        let acc = interface.acc.clone();
        self.try_replace_single_path_state_in_place(values, acc)
    }

    /// Reserve the uniquely-owned linear Segment used by the deterministic
    /// runtime fast path. This is called before timed decoding operations.
    pub fn reserve_single_segment_capacity(&mut self, min_capacity: usize) -> bool {
        // Startup is outside the timed hot path: deliberately detach the small
        // linear representation from any compile-time/shared Arcs so subsequent
        // deterministic commits can mutate it without allocating.
        let upper = Arc::make_mut(&mut self.inner);
        let Upper::Interface(interface) = upper else {
            return false;
        };
        let interface = Arc::make_mut(interface);
        let lower = Arc::make_mut(&mut interface.inner);
        let Lower::Segment(segment) = lower else {
            return false;
        };
        let segment = Arc::make_mut(segment);
        // Cloning a Segment deliberately clears this derived suffix cache.
        segment.rest = OnceLock::new();
        segment.values.reserve_total(min_capacity)
    }

    /// Apply one deterministic stack effect directly to a uniquely-owned
    /// single Segment. Returns false without mutation when the operation would
    /// cross the Segment floor, grow capacity, or invalidate a cached suffix.
    pub fn try_apply_single_segment_stack_effect_in_place(
        &mut self,
        pop: usize,
        pushes: &[T],
    ) -> bool {
        let Some(upper) = Arc::get_mut(&mut self.inner) else {
            return false;
        };
        let Upper::Interface(interface) = upper else {
            return false;
        };
        let Some(interface) = Arc::get_mut(interface) else {
            return false;
        };
        let Some(lower) = Arc::get_mut(&mut interface.inner) else {
            return false;
        };
        let Lower::Segment(segment) = lower else {
            return false;
        };
        let Some(segment) = Arc::get_mut(segment) else {
            return false;
        };
        if segment.rest.get().is_some() || pop > segment.values.len() {
            return false;
        }
        let new_len = segment.values.len() - pop + pushes.len();
        if new_len == 0 || !segment.values.can_push_without_alloc(new_len) {
            return false;
        }

        segment.values.truncate(segment.values.len() - pop);
        for value in pushes {
            if !segment.values.try_push(value.clone()) {
                return false;
            }
        }
        segment.max_depth = segment.next.max_depth() + new_len as u32;
        segment.segments_len = segment.next.segments_len() + new_len;
        true
    }

    pub fn is_empty(&self) -> bool {
        match &*self.inner {
            Upper::Branch(b) => b.children.is_empty() && b.empty.is_none(),
            Upper::Interface(_) => false,
        }
    }

    pub fn max_depth(&self) -> u32 {
        self.inner.max_depth()
    }

    pub fn summary(&self) -> LeveledGSSSummary {
        let mut visited_upperbranch: HashSet<usize> = HashSet::new();
        let mut visited_interface: HashSet<usize> = HashSet::new();
        let mut visited_lower: HashSet<usize> = HashSet::new();

        let mut upperbranch_nodes = 0usize;
        let mut interface_nodes = 0usize;
        let mut lower_nodes = 0usize;
        let mut lower_general_nodes = 0usize;
        let mut lower_segment_nodes = 0usize;
        let mut total_edges = 0usize;
        let mut accumulator_instances = 0usize;

        let mut upper_queue: VecDeque<Arc<Upper<T, A>>> = VecDeque::new();
        upper_queue.push_back(self.inner.clone());
        let mut lower_queue: VecDeque<Arc<Lower<T>>> = VecDeque::new();

        while let Some(node) = upper_queue.pop_front() {
            match &*node {
                Upper::Branch(branch) => {
                    let node_id = Arc::as_ptr(branch) as usize;
                    if !visited_upperbranch.insert(node_id) {
                        continue;
                    }
                    upperbranch_nodes += 1;
                    if branch.empty.is_some() {
                        accumulator_instances += 1;
                    }
                    for children in branch.children.values() {
                        total_edges += children.len();
                        for child in children.values() {
                            upper_queue.push_back(child.clone());
                        }
                    }
                }
                Upper::Interface(interface) => {
                    let node_id = Arc::as_ptr(interface) as usize;
                    if !visited_interface.insert(node_id) {
                        continue;
                    }
                    interface_nodes += 1;
                    accumulator_instances += 1;
                    total_edges += 1;
                    lower_queue.push_back(interface.inner.clone());
                }
            }
        }

        while let Some(node) = lower_queue.pop_front() {
            let node_id = lower_node_id(&node);
            if !visited_lower.insert(node_id) {
                continue;
            }
            lower_nodes += 1;
            match &*node {
                Lower::Segment(_) => lower_segment_nodes += 1,
                Lower::General { .. } => lower_general_nodes += 1,
            }
            // Walk through this node and any owned segment chain below it.
            let mut current: &Lower<T> = &*node;
            loop {
                match current {
                    Lower::Segment(seg) => {
                        total_edges += 1; // One edge from this Segment to its next
                        match &*seg.next {
                            Lower::Segment(inner_seg) => {
                                let inner_id = Arc::as_ptr(inner_seg) as usize;
                                if !visited_lower.insert(inner_id) { break; }
                                lower_nodes += 1;
                                lower_segment_nodes += 1;
                                current = &*seg.next;
                            }
                            Lower::General { children, .. } => {
                                lower_nodes += 1;
                                lower_general_nodes += 1;
                                for kids in children.values() {
                                    total_edges += kids.len();
                                    for child in kids.values() {
                                        lower_queue.push_back(child.clone());
                                    }
                                }
                                break;
                            }
                        }
                    }
                    Lower::General { children, .. } => {
                        for kids in children.values() {
                            total_edges += kids.len();
                            for child in kids.values() {
                                lower_queue.push_back(child.clone());
                            }
                        }
                        break;
                    }
                }
            }
        }

        LeveledGSSSummary {
            top_values_count: self.inner.children_keys().len(),
            upperbranch_nodes,
            interface_nodes,
            lower_nodes,
            lower_general_nodes,
            lower_segment_nodes,
            total_unique_nodes: upperbranch_nodes + interface_nodes + lower_nodes,
            total_edges,
            accumulator_instances,
            max_depth: self.max_depth(),
        }
    }

    pub fn isolate(&self, value: Option<T>) -> Self {

        if let Some(ref v) = value {
            match &*self.inner {
                Upper::Branch(b) => {
                    if b.empty.is_none() && b.children.len() == 1 && b.children.contains_key(v) {
                        return self.clone();
                    }
                }
                Upper::Interface(i) => {
                    if !i.inner.empty() && i.inner.children_len() == 1 && i.inner.children_contains_key(v) {
                        return self.clone();
                    }
                }
            }
        } else {
            match &*self.inner {
                Upper::Branch(b) => {
                    if b.children.is_empty() {
                        return self.clone();
                    }
                }
                Upper::Interface(i) => {
                    if i.inner.children_is_empty() && i.inner.empty() {
                        return self.clone();
                    }
                }
            }
        }

        let new_inner = if let Some(val) = value {
            match &*self.inner {
                Upper::Branch(b) => {
                    let filtered_children = b
                        .children
                        .get(&val)
                        .map(|kids| CompactMap::unit(val.clone(), kids.clone()))
                        .unwrap_or_else(CompactMap::new);
                    let max_depth = b
                        .children
                        .get(&val)
                        .and_then(|kids| kids.get_max().map(|(depth, _)| *depth + 1))
                        .unwrap_or(0);
                    let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: filtered_children,
                        empty: None,
                        max_depth,
                    })));
                    try_promote(&new_b)
                }
                Upper::Interface(i) => {
                    // Fast path for Segment: if isolating the top value, reconstruct
                    if i.inner.is_segment() {
                        if i.inner.segment_top_value() == &val {
                            let rest = i.inner.segment_rest_arc();
                            let new_lower_root = new_lower(
                                CompactMap::unit(val.clone(), CompactOrdMap::unit(rest.max_depth(), rest)),
                                false,
                            );
                            new_interface(new_lower_root, i.acc.clone())
                        } else {
                            empty_upper_inner()
                        }
                    } else if let Lower::General { children, .. } = &*i.inner {
                        if let Some(kids) = children.get(&val) {
                            let filtered_children = CompactMap::unit(val.clone(), kids.clone());
                            let new_lower_root = new_lower(filtered_children, false);
                            new_interface(new_lower_root, i.acc.clone())
                        } else {
                            empty_upper_inner()
                        }
                    } else {
                        empty_upper_inner()
                    }
                }
            }
        } else {
            let empty_acc = match &*self.inner {
                Upper::Branch(b) => b.empty.clone(),
                Upper::Interface(i) => {
                    if i.inner.empty() {
                        Some(i.acc.clone())
                    } else {
                        None
                    }
                }
            };
            let new_b = new_branch(CompactMap::new(), empty_acc);
            try_promote(&new_b)
        };
        LeveledGSS { inner: new_inner }
    }

    pub fn apply<B, F>(&self, mut func: F) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        F: FnMut(&A) -> B,
    {
        let mut acc_memo: StdHashMap<A, B> = StdHashMap::new();

        fn map_acc<A, B, F>(a: &A, memo: &mut StdHashMap<A, B>, f: &mut F) -> B
        where
            A: Clone + Eq + Hash,
            B: Clone,
            F: FnMut(&A) -> B,
        {
            if let Some(v) = memo.get(a) {
                return v.clone();
            }
            let r = f(a);
            memo.insert(a.clone(), r.clone());
            r
        }

        fn transform<T, A, B, F>(
            node: &Arc<Upper<T, A>>,
            memo_acc: &mut StdHashMap<A, B>,
            f: &mut F,
        ) -> Arc<Upper<T, B>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            B: Merge + Clone + Eq + Hash,
            F: FnMut(&A) -> B,
        {
            match &**node {
                Upper::Interface(i) => {
                    let new_acc = map_acc(&i.acc, memo_acc, f);
                    let res = new_interface(i.inner.clone(), new_acc);
                    try_promote(&res)
                }
                Upper::Branch(b) => {
                    let new_empty = b.empty.as_ref().map(|e| map_acc(e, memo_acc, f));
                    let mut new_children: Children<T, Upper<T, B>> = CompactMap::new();
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: CompactOrdMap<Arc<Upper<T, B>>> = CompactOrdMap::new();
                        for child in kids.values() {
                            let new_child = transform::<T, A, B, F>(child, memo_acc, f);
                            new_kids.insert(new_child.max_depth(), new_child);
                        }
                        new_children.insert(v.clone(), new_kids);
                    }
                    let res = new_branch(new_children, new_empty);
                    try_promote(&res)
                }
            }
        }

        LeveledGSS {
            inner: transform::<T, A, B, F>(&self.inner, &mut acc_memo, &mut func),
        }
    }

    /// Partition this GSS by accumulator value.
    ///
    /// Each result retains precisely the original stack paths labelled with the
    /// paired accumulator. The returned GSS erases that label to `()`, so later
    /// users can keep the accumulator as branch-local state without losing its
    /// correlation with parser paths.
    pub fn partition_by_accumulator(&self) -> Vec<(LeveledGSS<T, ()>, A)> {
        self.partition_by_accumulator_at_most(usize::MAX, usize::MAX)
            .expect("unbounded accumulator partition cannot exhaust its budget")
    }

    /// Partition by accumulator while bounding the accumulator-labelled upper
    /// structure inspected before any transformed GSSes are built.
    ///
    /// Lower GSS nodes carry no accumulator and are therefore irrelevant to
    /// discovering the partition labels. Returning `None` is transactional:
    /// `self` is immutable and no partial partition escapes.
    pub fn partition_by_accumulator_at_most(
        &self,
        max_accumulators: usize,
        max_upper_nodes: usize,
    ) -> Option<Vec<(LeveledGSS<T, ()>, A)>> {
        let mut accumulators = Vec::new();
        let mut pending = vec![self.inner.clone()];
        let mut visited = HashSet::new();

        while let Some(node) = pending.pop() {
            let ptr = Arc::as_ptr(&node) as usize;
            if !visited.insert(ptr) {
                continue;
            }
            if visited.len() > max_upper_nodes {
                return None;
            }

            let mut record = |accumulator: &A| -> Option<()> {
                if !accumulators.contains(accumulator) {
                    if accumulators.len() == max_accumulators {
                        return None;
                    }
                    accumulators.push(accumulator.clone());
                }
                Some(())
            };

            match &*node {
                Upper::Interface(interface) => record(&interface.acc)?,
                Upper::Branch(branch) => {
                    if let Some(accumulator) = &branch.empty {
                        record(accumulator)?;
                    }
                    pending.extend(
                        branch
                            .children
                            .values()
                            .flat_map(|kids| kids.values())
                            .cloned(),
                    );
                }
            }
        }

        Some(
            accumulators
                .into_iter()
                .map(|accumulator| {
                    let paths = self
                        .apply_and_prune(|candidate| (candidate == &accumulator).then_some(()));
                    (paths, accumulator)
                })
                .collect(),
        )
    }

    pub fn apply_and_prune<B, M>(&self, mut mutator: M) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        M: FnMut(&A) -> Option<B>,
    {
        // Fast path: single Interface at root — no memo or tree traversal needed.
        if let Upper::Interface(i) = &*self.inner {
            return match mutator(&i.acc) {
                Some(new_acc) => LeveledGSS { inner: new_interface(i.inner.clone(), new_acc) },
                None => LeveledGSS::empty(),
            };
        }

        // Use a flat Vec for memo instead of HashMap — avoids hashing cost
        // for the typical case of 2-4 unique accumulators.
        let mut acc_memo: Vec<(A, Option<B>)> = Vec::with_capacity(4);

        fn mutate_acc<A, B, M>(
            a: &A,
            memo: &mut Vec<(A, Option<B>)>,
            m: &mut M,
        ) -> Option<B>
        where
            A: Clone + Eq,
            B: Clone,
            M: FnMut(&A) -> Option<B>,
        {
            for (k, v) in memo.iter() {
                if k == a {
                    return v.clone();
                }
            }
            let r = m(a);
            memo.push((a.clone(), r.clone()));
            r
        }

        fn transform<T, A, B, M>(
            node: &Arc<Upper<T, A>>,
            memo: &mut Vec<(A, Option<B>)>,
            m: &mut M,
        ) -> Option<Arc<Upper<T, B>>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            B: Merge + Clone + Eq + Hash,
            M: FnMut(&A) -> Option<B>,
        {
            match &**node {
                Upper::Interface(i) => {
                    let new_acc_opt = mutate_acc(&i.acc, memo, m);
                    if let Some(new_acc) = new_acc_opt {
                        let new_i = new_interface(i.inner.clone(), new_acc);
                        Some(try_promote(&new_i))
                    } else {
                        None
                    }
                }
                Upper::Branch(b) => {
                    let new_empty_opt = b.empty.as_ref().and_then(|e| mutate_acc(e, memo, m));
                    let mut new_children: Children<T, Upper<T, B>> = CompactMap::new();
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: CompactOrdMap<Arc<Upper<T, B>>> = CompactOrdMap::new();
                        for child in kids.values() {
                            if let Some(nc) = transform::<T, A, B, M>(child, memo, m) {
                                new_kids.insert(nc.max_depth(), nc);
                            }
                        }
                        if !new_kids.is_empty() {
                            new_children.insert(v.clone(), new_kids);
                        }
                    }

                    if new_children.is_empty() && new_empty_opt.is_none() {
                        None
                    } else {
                        let new_b = new_branch(new_children, new_empty_opt);
                        Some(try_promote(&new_b))
                    }
                }
            }
        }

        let res_inner_opt = transform::<T, A, B, M>(&self.inner, &mut acc_memo, &mut mutator);
        res_inner_opt.map_or_else(LeveledGSS::<T, B>::empty, |inner| LeveledGSS::<T, B> { inner })
    }

    /// Like a cross-type no-promote transform followed by decompose_and_pop, but avoids
    /// building the root-level Branch node. Returns (value, sub_gss) pairs directly,
    /// plus a Vec of "root accumulators" (transformed empty values at the root Branch)
    /// to be checked for final_weight separately.
    pub fn apply_transform_and_decompose<B, M>(
        &self,
        mut mutator: M,
    ) -> (Vec<(T, LeveledGSS<T, B>)>, Vec<B>)
    where
        B: Merge + Clone + Eq + Hash,
        M: FnMut(&A) -> Option<B>,
    {
        let mut acc_memo: Vec<(A, Option<B>)> = Vec::with_capacity(4);

        fn mutate_acc_td<A, B, M>(
            a: &A,
            memo: &mut Vec<(A, Option<B>)>,
            m: &mut M,
        ) -> Option<B>
        where
            A: Clone + Eq,
            B: Clone,
            M: FnMut(&A) -> Option<B>,
        {
            for (k, v) in memo.iter() {
                if k == a {
                    return v.clone();
                }
            }
            let r = m(a);
            memo.push((a.clone(), r.clone()));
            r
        }

        fn transform_td<T, A, B, M>(
            node: &Arc<Upper<T, A>>,
            memo: &mut Vec<(A, Option<B>)>,
            m: &mut M,
        ) -> Option<Arc<Upper<T, B>>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            B: Merge + Clone + Eq + Hash,
            M: FnMut(&A) -> Option<B>,
        {
            match &**node {
                Upper::Interface(i) => {
                    mutate_acc_td(&i.acc, memo, m)
                        .map(|new_acc| new_interface(i.inner.clone(), new_acc))
                }
                Upper::Branch(b) => {
                    let new_empty_opt = b.empty.as_ref().and_then(|e| mutate_acc_td(e, memo, m));
                    // Fast path: single child entry with single child.
                    if b.children.len() == 1 && new_empty_opt.is_none() {
                        let (v, kids) = b.children.iter().next().unwrap();
                        if kids.len() == 1 {
                            let child = kids.values().next().unwrap();
                            if let Some(nc) = transform_td::<T, A, B, M>(child, memo, m) {
                                let new_kids = CompactOrdMap::unit(nc.max_depth(), nc);
                                let new_children = CompactMap::unit(v.clone(), new_kids);
                                return Some(new_branch(new_children, None));
                            } else {
                                return None;
                            }
                        }
                    }
                    let mut new_children: Children<T, Upper<T, B>> = CompactMap::new();
                    for (v, kids) in b.children.iter() {
                        let new_kids: CompactOrdMap<Arc<Upper<T, B>>> = kids.values()
                            .filter_map(|child| transform_td::<T, A, B, M>(child, memo, m))
                            .map(|nc| (nc.max_depth(), nc))
                            .collect();
                        if !new_kids.is_empty() {
                            new_children.insert(v.clone(), new_kids);
                        }
                    }
                    if new_children.is_empty() && new_empty_opt.is_none() {
                        None
                    } else {
                        Some(new_branch(new_children, new_empty_opt))
                    }
                }
            }
        }

        match &*self.inner {
            Upper::Interface(i) => {
                // Interface root: transform acc, then decompose inner Lower's children.
                let new_acc = match mutate_acc_td(&i.acc, &mut acc_memo, &mut mutator) {
                    Some(a) => a,
                    None => return (Vec::new(), Vec::new()),
                };
                let mut result = Vec::with_capacity(i.inner.children_len());
                match &*i.inner {
                    Lower::Segment(seg) => {
                        let value = seg.values.last().unwrap();
                        let rest = i.inner.segment_rest_arc();
                        if !rest.children_is_empty() || rest.empty() {
                            let upper = new_interface(rest, new_acc.clone());
                            result.push((value.clone(), LeveledGSS { inner: upper }));
                        }
                    }
                    Lower::General { children, .. } => {
                        for (val, kids) in children.iter() {
                            let lower = if kids.len() == 1 {
                                kids.values().next().unwrap().clone()
                            } else {
                                let mut it = kids.values();
                                let mut acc = it.next().unwrap().clone();
                                for child in it {
                                    acc = merge_lower(&acc, child);
                                }
                                acc
                            };
                            if !lower.children_is_empty() || lower.empty() {
                                let upper = new_interface(lower, new_acc.clone());
                                result.push((val.clone(), LeveledGSS { inner: upper }));
                            }
                        }
                    }
                }
                (result, Vec::new())
            }
            Upper::Branch(b) => {
                // Branch root: transform each child subtree, decompose into (value, sub_gss) pairs.
                let root_accs: Vec<B> = b.empty.iter()
                    .filter_map(|e| mutate_acc_td(e, &mut acc_memo, &mut mutator))
                    .collect();
                let mut result = Vec::with_capacity(b.children.len());
                for (val, kids) in b.children.iter() {
                    // Transform each child, collect into new_kids.
                    let mut new_kids: Vec<Arc<Upper<T, B>>> = Vec::new();
                    for child in kids.values() {
                        if let Some(nc) = transform_td::<T, A, B, M>(child, &mut acc_memo, &mut mutator) {
                            new_kids.push(nc);
                        }
                    }
                    if new_kids.is_empty() {
                        continue;
                    }
                    // Merge children (like decompose_and_pop does).
                    let merged = if new_kids.len() == 1 {
                        new_kids.into_iter().next().unwrap()
                    } else {
                        let mut it = new_kids.into_iter();
                        let mut acc = it.next().unwrap();
                        for child in it {
                            acc = merge_upper(&acc, &child);
                        }
                        acc
                    };
                    let is_empty = matches!(&*merged,
                        Upper::Branch(b) if b.children.is_empty() && b.empty.is_none());
                    if !is_empty {
                        result.push((val.clone(), LeveledGSS { inner: merged }));
                    }
                }
                (result, root_accs)
            }
        }
    }

    /// Like apply_and_prune but skips try_promote. Use when the tree is already
    /// canonical and the transformation preserves structure (e.g., DenseMaskAcc → DenseMaskAcc).
    pub fn apply_and_prune_no_promote(&self, mut mutator: impl FnMut(&A) -> Option<A>) -> Self {
        // Fast path: single Interface at root.
        if let Upper::Interface(i) = &*self.inner {
            return match mutator(&i.acc) {
                Some(new_acc) if new_acc == i.acc => self.clone(),
                Some(new_acc) => LeveledGSS { inner: new_interface(i.inner.clone(), new_acc) },
                None => LeveledGSS::empty(),
            };
        }

        let mut acc_memo: Vec<(A, Option<A>)> = Vec::with_capacity(4);

        fn mutate_acc_np<A, M>(
            a: &A,
            memo: &mut Vec<(A, Option<A>)>,
            m: &mut M,
        ) -> Option<A>
        where
            A: Clone + Eq,
            M: FnMut(&A) -> Option<A>,
        {
            for (k, v) in memo.iter() {
                if k == a {
                    return v.clone();
                }
            }
            let r = m(a);
            memo.push((a.clone(), r.clone()));
            r
        }

        fn transform_np<T, A, M>(
            node: &Arc<Upper<T, A>>,
            memo: &mut Vec<(A, Option<A>)>,
            m: &mut M,
        ) -> Option<Arc<Upper<T, A>>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            M: FnMut(&A) -> Option<A>,
        {
            match &**node {
                Upper::Interface(i) => {
                    let new_acc_opt = mutate_acc_np(&i.acc, memo, m);
                    new_acc_opt.map(|new_acc| {
                        if new_acc == i.acc {
                            node.clone()
                        } else {
                            new_interface(i.inner.clone(), new_acc)
                        }
                    })
                }
                Upper::Branch(b) => {
                    let new_empty_opt = b.empty.as_ref().and_then(|e| mutate_acc_np(e, memo, m));
                    let empty_unchanged = match (&b.empty, &new_empty_opt) {
                        (None, None) => true,
                        (Some(old), Some(new)) => old == new,
                        _ => false,
                    };
                    // Fast path: single child entry with single child.
                    if b.children.len() == 1 && new_empty_opt.is_none() {
                        let (v, kids) = b.children.iter().next().unwrap();
                        if kids.len() == 1 {
                            let child = kids.values().next().unwrap();
                            if let Some(nc) = transform_np::<T, A, M>(child, memo, m) {
                                if empty_unchanged && Arc::ptr_eq(&nc, child) {
                                    return Some(node.clone());
                                }
                                let new_kids = CompactOrdMap::unit(nc.max_depth(), nc);
                                let new_children = CompactMap::unit(v.clone(), new_kids);
                                return Some(new_branch(new_children, None));
                            } else {
                                return None;
                            }
                        }
                    }
                    let mut new_children: Children<T, Upper<T, A>> = CompactMap::new();
                    let mut unchanged = empty_unchanged;
                    for (v, kids) in b.children.iter() {
                        let mut new_kids: CompactOrdMap<Arc<Upper<T, A>>> = CompactOrdMap::new();
                        for child in kids.values() {
                            let Some(nc) = transform_np::<T, A, M>(child, memo, m) else {
                                unchanged = false;
                                continue;
                            };
                            unchanged &= Arc::ptr_eq(&nc, child);
                            new_kids.insert(nc.max_depth(), nc);
                        }
                        unchanged &= new_kids.len() == kids.len();
                        if !new_kids.is_empty() {
                            new_children.insert(v.clone(), new_kids);
                        }
                    }
                    if new_children.is_empty() && new_empty_opt.is_none() {
                        None
                    } else if unchanged && new_children.len() == b.children.len() {
                        Some(node.clone())
                    } else {
                        Some(new_branch(new_children, new_empty_opt))
                    }
                }
            }
        }

        let res_inner_opt = transform_np::<T, A, _>(&self.inner, &mut acc_memo, &mut mutator);
        res_inner_opt.map_or_else(Self::empty, |inner| Self { inner })
    }

    pub fn merge(&self, other: &Self) -> Self {
        if self.is_empty() { return other.clone(); }
        if other.is_empty() { return self.clone(); }
        let merged_inner = merge_upper(&self.inner, &other.inner);
        LeveledGSS {
            inner: merged_inner,
        }
    }

    pub fn merge_many(gsses: impl IntoIterator<Item = Self>) -> Self {
        let mut items: Vec<Self> = gsses.into_iter().collect();
        if items.is_empty() {
            return LeveledGSS::empty();
        }
        if items.len() == 1 {
            return items.into_iter().next().unwrap();
        }
        while items.len() > 1 {
            let mut next = Vec::with_capacity((items.len() + 1) / 2);
            let mut iter = items.into_iter();
            while let Some(a) = iter.next() {
                if let Some(b) = iter.next() {
                    next.push(a.merge(&b));
                } else {
                    next.push(a);
                }
            }
            items = next;
        }
        items.into_iter().next().unwrap()
    }

    pub fn fuse(&self, levels: Option<isize>) -> Self {
        if let Some(l) = levels {
            if l <= 0 {
                return self.clone();
            }
        }

        // Fast path for fuse(Some(1)): children see remain=0 → identity.
        // So fuse is a no-op iff the top node has no multi-depth slots.
        if levels == Some(1) {
            let no_multi_depth = match &*self.inner {
                Upper::Interface(i) => {
                    match &*i.inner {
                        Lower::Segment(_) => true,
                        Lower::General { children, .. } => !children.values().any(|kids| kids.len() > 1),
                    }
                }
                Upper::Branch(b) => {
                    !b.children.values().any(|kids| kids.len() > 1)
                }
            };
            if no_multi_depth {
                return self.clone();
            }
        }

        let mut memo_upper: StdHashMap<(usize, Option<isize>), Arc<Upper<T, A>>> = StdHashMap::new();
        let mut memo_lower: StdHashMap<(usize, Option<isize>), Arc<Lower<T>>> = StdHashMap::new();

        fn fuse_lower<T, A>(
            node: &Arc<Lower<T>>,
            remain: Option<isize>,
            memo: &mut StdHashMap<(usize, Option<isize>), Arc<Lower<T>>>,
        ) -> Arc<Lower<T>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            if let Some(r) = remain {
                if r == 0 {
                    return node.clone();
                }
            }
            let key = (lower_node_id(node), remain);
            if let Some(cached) = memo.get(&key) {
                return cached.clone();
            }

            let next_remain = remain.map(|r| r - 1);

            // Fast path for Segment: no multi-depth slots, recurse on next only
            if let Lower::Segment(seg) = &**node {
                let next_remain_seg = remain.map(|r| r - seg.values.len() as isize);
                let next_arc = seg.next.clone();
                let fused_next = fuse_lower::<T, A>(&next_arc, next_remain_seg, memo);
                if Arc::ptr_eq(&fused_next, &next_arc) {
                    memo.insert(key, node.clone());
                    return node.clone();
                }
                let res = new_segment(seg.values.clone(), fused_next.clone());
                memo.insert(key, res.clone());
                return res;
            }

            // General path
            let Lower::General { children, .. } = &**node else { unreachable!() };
            let has_multi_depth_slots = children.values().any(|kids| kids.len() > 1);

            let mut new_children_by_value: StdHashMap<T, Vec<Arc<Lower<T>>>> = StdHashMap::new();
            let mut children_changed = false;

            for (v, kids) in children.iter() {
                for child in kids.values() {
                    let fused_child = fuse_lower::<T, A>(child, next_remain, memo);
                    if !Arc::ptr_eq(&fused_child, child) {
                        children_changed = true;
                    }
                    new_children_by_value
                        .entry(v.clone())
                        .or_default()
                        .push(fused_child);
                }
            }

            if !has_multi_depth_slots && !children_changed {
                memo.insert(key, node.clone());
                return node.clone();
            }

            let mut final_children: Children<T, Lower<T>> = CompactMap::new();
            for (v, fused_kids) in new_children_by_value {
                if fused_kids.is_empty() {
                    continue;
                }
                let mut it = fused_kids.into_iter();
                let first = it.next().unwrap();
                let merged_child = it.fold(first, |acc, next| merge_lower(&acc, &next));
                final_children.insert(v, CompactOrdMap::unit(merged_child.max_depth(), merged_child));
            }

            let res = new_lower(final_children, node.empty());
            memo.insert(key, res.clone());
            res
        }

        fn fuse_upper<T, A>(
            node: &Arc<Upper<T, A>>,
            remain: Option<isize>,
            memo_upper: &mut StdHashMap<(usize, Option<isize>), Arc<Upper<T, A>>>,
            memo_lower: &mut StdHashMap<(usize, Option<isize>), Arc<Lower<T>>>,
        ) -> Arc<Upper<T, A>>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            if let Some(r) = remain {
                if r == 0 {
                    return node.clone();
                }
            }
            let key = (Arc::as_ptr(node) as usize, remain);
            if let Some(cached) = memo_upper.get(&key) {
                return cached.clone();
            }

            let next_remain = remain.map(|r| r - 1);

            let res = match &**node {
                Upper::Interface(i) => {
                    let has_multi_depth_slots = match &*i.inner {
                        Lower::Segment(_) => false,
                        Lower::General { children, .. } => children.values().any(|kids| kids.len() > 1),
                    };
                    let fused_lower = fuse_lower::<T, A>(&i.inner, next_remain, memo_lower);
                    if !has_multi_depth_slots && Arc::ptr_eq(&fused_lower, &i.inner) {
                        memo_upper.insert(key, node.clone());
                        return node.clone();
                    }
                    new_interface(fused_lower, i.acc.clone())
                }
                Upper::Branch(b) => {
                    let has_multi_depth_slots = b.children.values().any(|kids| kids.len() > 1);
                    let mut new_children_by_value: StdHashMap<T, Vec<Arc<Upper<T, A>>>> =
                        StdHashMap::new();
                    let mut children_changed = false;

                    for (v, kids) in b.children.iter() {
                        for child in kids.values() {
                            let fused_child =
                                fuse_upper(child, next_remain, memo_upper, memo_lower);
                            if !Arc::ptr_eq(&fused_child, child) {
                                children_changed = true;
                            }
                            new_children_by_value
                                .entry(v.clone())
                                .or_default()
                                .push(fused_child);
                        }
                    }

                    if !has_multi_depth_slots && !children_changed {
                        memo_upper.insert(key, node.clone());
                        return node.clone();
                    }

                    let mut final_children: Children<T, Upper<T, A>> = CompactMap::new();
                    for (v, fused_kids) in new_children_by_value {
                        if fused_kids.is_empty() {
                            continue;
                        }
                        let mut it = fused_kids.into_iter();
                        let first = it.next().unwrap();
                        let merged_child = it.fold(first, |acc, next| merge_upper(&acc, &next));
                        final_children
                            .insert(v, CompactOrdMap::unit(merged_child.max_depth(), merged_child));
                    }
                    let new_b = new_branch(final_children, b.empty.clone());
                    try_promote(&new_b)
                }
            };

            memo_upper.insert(key, res.clone());
            res
        }

        let new_inner = fuse_upper::<T, A>(&self.inner, levels, &mut memo_upper, &mut memo_lower);
        if Arc::ptr_eq(&new_inner, &self.inner) {
            self.clone()
        } else {
            LeveledGSS { inner: new_inner }
        }
    }

    pub fn peek(&self) -> HashSet<T> {
        self.inner.children_keys().into_iter().collect()
    }

    pub fn peek_values(&self) -> SmallVec<[T; 8]> {
        self.inner.children_keys()
    }

    /// Iterate over top values without allocating a Vec.
    /// Calls `f` for each top-level value in the GSS.
    pub fn for_each_top_value<F: FnMut(T)>(&self, mut f: F) {
        match &*self.inner {
            Upper::Branch(branch) => {
                for k in branch.children.keys() {
                    f(k.clone());
                }
            }
            Upper::Interface(interface) => {
                match &*interface.inner {
                    Lower::Segment(seg) => f(seg.values.last().unwrap().clone()),
                    Lower::General { children, .. } => {
                        for k in children.keys() {
                            f(k.clone());
                        }
                    }
                }
            }
        }
    }

    pub fn single_top_value(&self) -> Option<T> {
        self.inner.single_child_key()
    }

    pub fn single_exclusive_top_value(&self) -> Option<T> {
        self.inner.single_child_key_without_empty()
    }

    /// Count unique reachable upper and lower GSS nodes, stopping at `limit`.
    ///
    /// This measures compact representation work rather than represented stack
    /// count. It retains no node identities after returning and never expands
    /// concrete paths.
    pub fn node_count_at_most(&self, limit: usize) -> usize {
        if limit == 0 || self.is_empty() {
            return 0;
        }

        let mut upper_pending = vec![self.inner.clone()];
        let mut lower_pending = Vec::new();
        let mut upper_seen = HashSet::new();
        let mut lower_seen = HashSet::new();
        let mut count = 0usize;

        while let Some(node) = upper_pending.pop() {
            let ptr = Arc::as_ptr(&node) as usize;
            if !upper_seen.insert(ptr) {
                continue;
            }
            count += 1;
            if count == limit {
                return limit;
            }
            match &*node {
                Upper::Branch(branch) => upper_pending.extend(
                    branch
                        .children
                        .values()
                        .flat_map(|kids| kids.values())
                        .cloned(),
                ),
                Upper::Interface(interface) => lower_pending.push(interface.inner.clone()),
            }
        }

        while let Some(node) = lower_pending.pop() {
            let ptr = lower_node_id(&node);
            if !lower_seen.insert(ptr) {
                continue;
            }
            count += 1;
            if count == limit {
                return limit;
            }
            match &*node {
                Lower::Segment(segment) => lower_pending.push(segment.next.clone()),
                Lower::General { children, .. } => lower_pending.extend(
                    children
                        .values()
                        .flat_map(|kids| kids.values())
                        .cloned(),
                ),
            }
        }

        count
    }

    pub fn path_count_at_most(&self, limit: usize) -> usize {
        if limit == 0 || self.is_empty() {
            return 0;
        }

        fn capped_add(acc: usize, value: usize, limit: usize) -> usize {
            acc.saturating_add(value).min(limit)
        }

        fn count_lower<T>(
            node: &Arc<Lower<T>>,
            limit: usize,
            memo: &mut StdHashMap<usize, usize>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
        {
            let ptr = lower_node_id(node);
            if let Some(&cached) = memo.get(&ptr) {
                return cached;
            }
            let count = count_lower_inner(&**node, limit, memo);
            memo.insert(ptr, count);
            count
        }

        fn count_lower_inner<T>(
            node: &Lower<T>,
            limit: usize,
            memo: &mut StdHashMap<usize, usize>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
        {
            let mut count = usize::from(node.empty());
            match node {
                Lower::Segment(seg) => {
                    count = capped_add(count, count_lower_inner(&seg.next, limit, memo), limit);
                }
                Lower::General { children, .. } => {
                    for kids in children.values() {
                        for child in kids.values() {
                            count = capped_add(count, count_lower(child, limit, memo), limit);
                            if count == limit {
                                return count;
                            }
                        }
                    }
                }
            }

            count
        }

        fn count_upper<T, A>(
            node: &Arc<Upper<T, A>>,
            limit: usize,
            memo_upper: &mut StdHashMap<usize, usize>,
            memo_lower: &mut StdHashMap<usize, usize>,
        ) -> usize
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            let ptr = Arc::as_ptr(node) as usize;
            if let Some(&cached) = memo_upper.get(&ptr) {
                return cached;
            }

            let mut count = match &**node {
                Upper::Branch(b) => usize::from(b.empty.is_some()),
                Upper::Interface(i) => usize::from(i.inner.empty()),
            };

            match &**node {
                Upper::Branch(b) => {
                    for children in b.children.values() {
                        for child in children.values() {
                            count = capped_add(
                                count,
                                count_upper(child, limit, memo_upper, memo_lower),
                                limit,
                            );
                            if count == limit {
                                memo_upper.insert(ptr, count);
                                return count;
                            }
                        }
                    }
                }
                Upper::Interface(i) => {
                    match &*i.inner {
                        Lower::Segment(seg) => {
                            count = capped_add(count, count_lower_inner(&seg.next, limit, memo_lower), limit);
                        }
                        Lower::General { children, .. } => {
                            for kids in children.values() {
                                for child in kids.values() {
                                    count = capped_add(count, count_lower(child, limit, memo_lower), limit);
                                    if count == limit {
                                        memo_upper.insert(ptr, count);
                                        return count;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            memo_upper.insert(ptr, count);
            count
        }

        let mut memo_upper = StdHashMap::new();
        let mut memo_lower = StdHashMap::new();
        count_upper(&self.inner, limit, &mut memo_upper, &mut memo_lower)
    }

    pub fn is_single_path(&self) -> bool {
        self.path_count_at_most(2) <= 1
    }

    /// Materialize the sole concrete stack only when its cached maximum depth
    /// is within `max_depth`. The O(1) depth rejection happens before walking
    /// any General-node chain, and the subsequent traversal rejects branching
    /// immediately rather than enumerating alternative paths.
    pub fn try_single_stack_bounded(&self, max_depth: usize) -> Option<(Vec<T>, A)> {
        if self.max_depth() as usize > max_depth {
            return None;
        }
        let mut top_first = SmallVec::<[T; 16]>::new();
        let acc = self.single_path_top_first_and_acc(&mut top_first)?;
        let mut stack = top_first.into_vec();
        stack.reverse();
        Some((stack, acc))
    }

    pub fn single_path_top_first_and_acc<S>(&self, out: &mut SmallVec<S>) -> Option<A>
    where
        S: smallvec::Array<Item = T>,
    {
        fn push_lower_path<T, S>(node: &Arc<Lower<T>>, out: &mut SmallVec<S>) -> bool
        where
            T: Clone + Eq + Hash,
            S: smallvec::Array<Item = T>,
        {
            match &**node {
                Lower::Segment(seg) => {
                    for value in seg.values.iter().rev() {
                        out.push(value.clone());
                    }
                    push_lower_path(&seg.next, out)
                }
                Lower::General { children, empty, .. } => {
                    if *empty {
                        return children.is_empty();
                    }
                    if children.len() != 1 {
                        return false;
                    }
                    let (value, kids) = children.iter().next().unwrap();
                    if kids.len() != 1 {
                        return false;
                    }
                    out.push(value.clone());
                    push_lower_path(kids.values().next().unwrap(), out)
                }
            }
        }

        fn push_upper_path<T, A, S>(
            node: &Arc<Upper<T, A>>,
            out: &mut SmallVec<S>,
        ) -> Option<A>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
            S: smallvec::Array<Item = T>,
        {
            match &**node {
                Upper::Interface(interface) => {
                    if push_lower_path(&interface.inner, out) {
                        Some(interface.acc.clone())
                    } else {
                        None
                    }
                }
                Upper::Branch(branch) => {
                    if branch.empty.is_some() || branch.children.len() != 1 {
                        return None;
                    }
                    let (value, kids) = branch.children.iter().next().unwrap();
                    if kids.len() != 1 {
                        return None;
                    }
                    out.push(value.clone());
                    push_upper_path(kids.values().next().unwrap(), out)
                }
            }
        }

        out.clear();
        let start_len = out.len();
        match push_upper_path(&self.inner, out) {
            Some(acc) => Some(acc),
            None => {
                out.truncate(start_len);
                None
            }
        }
    }

    pub fn reduce_acc(&self) -> Option<A> {

        let mut unique: HashSet<A> = HashSet::new();
        let mut queue: VecDeque<Arc<Upper<T, A>>> = VecDeque::new();
        let mut visited: HashSet<usize> = HashSet::new();

        queue.push_back(self.inner.clone());
        while let Some(node) = queue.pop_front() {
            let ptr = Arc::as_ptr(&node) as usize;
            if !visited.insert(ptr) {
                continue;
            }
            match &*node {
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        unique.insert(acc.clone());
                    }
                    for kids in b.children.values() {
                        for child in kids.values() {
                            queue.push_back(child.clone());
                        }
                    }
                }
                Upper::Interface(i) => {
                    unique.insert(i.acc.clone());
                }
            }
        }

        let mut it = unique.into_iter();
        let first = it.next()?;
        let reduced = it.fold(first, |acc, next| acc.merge(&next));
        Some(reduced)
    }

    /// Visit each accumulator in the GSS without collecting or merging.
    /// Uses pointer-based visited set to avoid hashing accumulators.
    pub fn for_each_acc(&self, mut f: impl FnMut(&A)) {
        const INLINE_VISITED_PTRS: usize = 32;

        enum VisitedPtrs {
            Small(SmallVec<[usize; INLINE_VISITED_PTRS]>),
            Large(HashSet<usize>),
        }

        impl VisitedPtrs {
            fn new() -> Self {
                Self::Small(SmallVec::new())
            }

            fn insert(&mut self, ptr: usize) -> bool {
                match self {
                    Self::Small(seen) => {
                        if seen.contains(&ptr) {
                            return false;
                        }
                        if seen.len() < INLINE_VISITED_PTRS {
                            seen.push(ptr);
                            return true;
                        }
                        let mut upgraded = HashSet::with_capacity(seen.len() * 2);
                        for &existing in seen.iter() {
                            upgraded.insert(existing);
                        }
                        let inserted = upgraded.insert(ptr);
                        *self = Self::Large(upgraded);
                        inserted
                    }
                    Self::Large(seen) => seen.insert(ptr),
                }
            }
        }

        fn walk<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Upper<T, A>>,
            visited: &mut VisitedPtrs,
            f: &mut impl FnMut(&A),
        ) {
            let ptr = Arc::as_ptr(node) as usize;
            if !visited.insert(ptr) {
                return;
            }
            match &**node {
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        f(acc);
                    }
                    for kids in b.children.values() {
                        for child in kids.values() {
                            walk(child, visited, f);
                        }
                    }
                }
                Upper::Interface(i) => {
                    f(&i.acc);
                }
            }
        }
        let mut visited = VisitedPtrs::new();
        walk(&self.inner, &mut visited, &mut f);
    }

    pub fn uniform_accumulator(&self) -> Option<A> {
        let mut value: Option<A> = None;
        let mut uniform = true;
        self.for_each_acc(|candidate| match &value {
            None => value = Some(candidate.clone()),
            Some(existing) if existing == candidate => {}
            Some(_) => uniform = false,
        });
        uniform.then_some(value).flatten()
    }

    /// Reuse an Interface-backed stack-language topology with a different
    /// uniform accumulator. The lower graph is immutable and shared.
    pub fn with_uniform_accumulator(&self, acc: A) -> Option<Self> {
        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };
        Some(Self {
            inner: new_interface(interface.inner.clone(), acc),
        })
    }

    /// Returns true if all accumulators in the upper tree satisfy the predicate.
    /// Short-circuits on the first accumulator that doesn't.
    /// For a single Interface node (common case), this is O(1).
    pub fn all_accs_satisfy(&self, pred: impl Fn(&A) -> bool) -> bool {
        fn check<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(
            node: &Arc<Upper<T, A>>,
            pred: &impl Fn(&A) -> bool,
        ) -> bool {
            match &**node {
                Upper::Interface(iface) => pred(&iface.acc),
                Upper::Branch(b) => {
                    if let Some(acc) = &b.empty {
                        if !pred(acc) {
                            return false;
                        }
                    }
                    for kids in b.children.values() {
                        for child in kids.values() {
                            if !check(child, pred) {
                                return false;
                            }
                        }
                    }
                    true
                }
            }
        }
        check(&self.inner, &pred)
    }

    pub fn truncate(&self, max_len: isize) -> Self {
        if max_len < 0 {
            return Self::empty();
        }

        let mut memo_upper = StdHashMap::new();
        let mut memo_lower = StdHashMap::new();

        let new_inner = truncate_upper(
            &self.inner,
            0,
            max_len,
            &mut memo_upper,
            &mut memo_lower,
        );

        new_inner.map_or_else(Self::empty, |inner| Self { inner })
    }

    fn get_node_info_lower(node: &Arc<Lower<T>>) -> String
    where
        T: std::fmt::Debug,
    {
        let mut info = format!(
            "Lower @ {:#x} (MaxDepth: {})",
            Arc::as_ptr(node) as usize,
            node.max_depth()
        );
        if node.empty() {
            info.push_str(" [TERMINAL]");
        }
        info
    }

    fn get_node_info_upper(node: &Arc<Upper<T, A>>) -> String
    where
        T: std::fmt::Debug,
        A: std::fmt::Debug,
    {
        match &**node {
            Upper::Branch(b) => {
                let mut info = format!(
                    "UpperBranch @ {:#x} (MaxDepth: {})",
                    Arc::as_ptr(b) as usize,
                    b.max_depth
                );
                if let Some(e) = &b.empty {
                    info.push_str(&format!(" [TERMINAL empty: {:?}]", e));
                }
                info
            }
            Upper::Interface(i) => {
                let mut info = format!(
                    "Interface @ {:#x} -> Lower @ {:#x} (MaxDepth: {}) | acc: {:?}",
                    Arc::as_ptr(i) as usize,
                    Arc::as_ptr(&i.inner) as usize,
                    node.max_depth(),
                    i.acc
                );
                if i.inner.empty() {
                    info.push_str(&format!(" [TERMINAL via acc: {:?}]", i.acc));
                }
                info
            }
        }
    }

    fn format_recursive_lower(
        node: &Arc<Lower<T>>,
        current_prefix: &str,
        printed_nodes: &mut HashSet<usize>,
        output_lines: &mut Vec<String>,
    ) where
        T: std::fmt::Debug,
    {
        printed_nodes.insert(Arc::as_ptr(node) as usize);

        let mut children_to_print = Vec::new();
        let node_children = node.children();
        let mut sorted_values: Vec<_> = node_children.keys().collect();
        sorted_values.sort_by_key(|v| format!("{:?}", v));

        for v in sorted_values {
            if let Some(kids_at_depths) = node_children.get(v) {
                for (depth, child) in kids_at_depths.iter() {
                    let label = format!("Edge {:?} (d={})", v, depth);
                    children_to_print.push((label, child.clone()));
                }
            }
        }

        let num_children = children_to_print.len();
        for (i, (label, child)) in children_to_print.into_iter().enumerate() {
            let is_last = i == num_children - 1;
            let prefix_char = if is_last { "└── " } else { "├── " };
            let child_id = Arc::as_ptr(&child) as usize;

            if printed_nodes.contains(&child_id) {
                let line = format!(
                    "{}{}{} -> Ref to Node @ {:#x}",
                    current_prefix, prefix_char, label, child_id
                );
                output_lines.push(line);
            } else {
                let line = format!(
                    "{}{}{} -> {}",
                    current_prefix,
                    prefix_char,
                    label,
                    Self::get_node_info_lower(&child)
                );
                output_lines.push(line);

                let child_prefix = format!("{}{}", current_prefix, if is_last { "    " } else { "│   " });
                Self::format_recursive_lower(&child, &child_prefix, printed_nodes, output_lines);
            }
        }
    }

    fn format_recursive_upper(
        node: &Arc<Upper<T, A>>,
        current_prefix: &str,
        printed_nodes: &mut HashSet<usize>,
        output_lines: &mut Vec<String>,
        upper_only: bool,
    ) where
        T: std::fmt::Debug,
        A: std::fmt::Debug,
    {
        printed_nodes.insert(Arc::as_ptr(node) as usize);

        match &**node {
            Upper::Branch(b) => {
                let mut children_to_print = Vec::new();
                let mut sorted_values: Vec<_> = b.children.keys().collect();
                sorted_values.sort_by_key(|v| format!("{:?}", v));

                for v in sorted_values {
                    if let Some(kids_at_depths) = b.children.get(v) {
                        for (depth, child) in kids_at_depths.iter() {
                            let label = format!("Edge {:?} (d={})", v, depth);
                            children_to_print.push((label, child.clone()));
                        }
                    }
                }

                let num_children = children_to_print.len();
                for (i, (label, child)) in children_to_print.into_iter().enumerate() {
                    let is_last = i == num_children - 1;
                    let prefix_char = if is_last { "└── " } else { "├── " };
                    let child_id = Arc::as_ptr(&child) as usize;

                    if printed_nodes.contains(&child_id) {
                        let line = format!(
                            "{}{}{} -> Ref to Node @ {:#x}",
                            current_prefix, prefix_char, label, child_id
                        );
                        output_lines.push(line);
                    } else {
                        let line = format!(
                            "{}{}{} -> {}",
                            current_prefix,
                            prefix_char,
                            label,
                            Self::get_node_info_upper(&child)
                        );
                        output_lines.push(line);

                        let child_prefix =
                            format!("{}{}", current_prefix, if is_last { "    " } else { "│   " });
                        Self::format_recursive_upper(
                            &child,
                            &child_prefix,
                            printed_nodes,
                            output_lines,
                            upper_only,
                        );
                    }
                }
            }
            Upper::Interface(i) => {
                if upper_only && !i.inner.children().is_empty() {
                    let prefix_char = "└── ";
                    let num_lower_edges: usize = i.inner.children().values().map(|kids| kids.len()).sum();
                    let line = format!("{}[{} lower edges omitted]", prefix_char, num_lower_edges);
                    output_lines.push(format!("{}{}", current_prefix, line));
                    return;
                }

                let mut children_to_print = Vec::new();
                let iface_children = i.inner.children();
                let mut sorted_values: Vec<_> = iface_children.keys().collect();
                sorted_values.sort_by_key(|v| format!("{:?}", v));

                for v in sorted_values {
                    if let Some(kids_at_depths) = iface_children.get(v) {
                        for (depth, child) in kids_at_depths.iter() {
                            let label = format!("Edge {:?} (d={})", v, depth);
                            children_to_print.push((label, child.clone()));
                        }
                    }
                }

                let num_children = children_to_print.len();
                for (i_idx, (label, child)) in children_to_print.into_iter().enumerate() {
                    let is_last = i_idx == num_children - 1;
                    let prefix_char = if is_last { "└── " } else { "├── " };
                    let child_id = Arc::as_ptr(&child) as usize;

                    if printed_nodes.contains(&child_id) {
                        let line = format!(
                            "{}{}{} -> Ref to Node @ {:#x}",
                            current_prefix, prefix_char, label, child_id
                        );
                        output_lines.push(line);
                    } else {
                        let line = format!(
                            "{}{}{} -> {}",
                            current_prefix,
                            prefix_char,
                            label,
                            Self::get_node_info_lower(&child)
                        );
                        output_lines.push(line);

                        let child_prefix =
                            format!("{}{}", current_prefix, if is_last { "    " } else { "│   " });
                        Self::format_recursive_lower(
                            &child,
                            &child_prefix,
                            printed_nodes,
                            output_lines,
                        );
                    }
                }
            }
        }
    }

}


impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
    pub fn single_predecessor_below_top(&self, top: &T) -> Option<T> {
        fn single_top<T: Clone + Eq + Hash>(node: &Arc<Lower<T>>) -> Option<T> {
            match &**node {
                Lower::Segment(_) => Some(node.segment_top_value().clone()),
                Lower::General {
                    children,
                    empty: false,
                    ..
                } if children.len() == 1 => children.keys().next().cloned(),
                _ => None,
            }
        }

        let Upper::Interface(interface) = &*self.inner else {
            return None;
        };
        match &*interface.inner {
            Lower::Segment(_) => {
                if interface.inner.segment_top_value() != top {
                    return None;
                }
                single_top(&interface.inner.segment_rest_arc())
            }
            Lower::General { children, .. } => {
                let kids = children.get(top)?;
                let mut predecessor: Option<T> = None;
                for child in kids.values() {
                    let candidate = single_top(child)?;
                    if predecessor
                        .as_ref()
                        .is_some_and(|existing| existing != &candidate)
                    {
                        return None;
                    }
                    predecessor = Some(candidate);
                }
                predecessor
            }
        }
    }

    pub fn from_small_stack_slices_shared_prefix<'a>(
        stacks: &[&'a [T]],
        acc: A,
    ) -> Option<Self>
    where
        T: 'a,
    {
        const MAX_STACKS: usize = 16;
        const MAX_INTERNED_NODES: usize = 96;
        if stacks.is_empty() || stacks.len() > MAX_STACKS {
            return None;
        }

        fn intern<T: Clone + Eq + Hash>(
            node: Arc<Lower<T>>,
            nodes: &mut SmallVec<[Arc<Lower<T>>; MAX_INTERNED_NODES]>,
        ) -> Option<Arc<Lower<T>>> {
            if let Some(existing) = nodes.iter().find(|existing| ***existing == *node) {
                return Some(existing.clone());
            }
            if nodes.len() == nodes.capacity() {
                return None;
            }
            nodes.push(node.clone());
            Some(node)
        }

        fn build<'a, T>(
            stacks: &[&'a [T]],
            mut base: Arc<Lower<T>>,
            interned: &mut SmallVec<[Arc<Lower<T>>; MAX_INTERNED_NODES]>,
        ) -> Option<Arc<Lower<T>>>
        where
            T: Clone + Eq + Hash + 'a,
        {
            let min_len = stacks.iter().map(|stack| stack.len()).min()?;
            let mut common = 0usize;
            while common < min_len
                && stacks
                    .iter()
                    .skip(1)
                    .all(|stack| stack[common] == stacks[0][common])
            {
                common += 1;
            }
            if common != 0 {
                base = intern(
                    new_segment(SV::from_vec(stacks[0][..common].to_vec()), base),
                    interned,
                )?;
                let mut trimmed = SmallVec::<[&'a [T]; MAX_STACKS]>::new();
                for &stack in stacks {
                    trimmed.push(&stack[common..]);
                }
                return build(&trimmed, base, interned);
            }

            if stacks.iter().all(|stack| stack.is_empty()) {
                return Some(base);
            }
            let includes_base = stacks.iter().any(|stack| stack.is_empty());
            let mut groups = SmallVec::<
                [(T, SmallVec<[&'a [T]; MAX_STACKS]>); MAX_STACKS],
            >::new();
            for &stack in stacks {
                let Some((top, rest)) = stack.split_last() else {
                    continue;
                };
                if let Some((_, grouped)) = groups.iter_mut().find(|(key, _)| key == top) {
                    grouped.push(rest);
                } else {
                    if groups.len() == groups.capacity() {
                        return None;
                    }
                    let mut grouped = SmallVec::new();
                    grouped.push(rest);
                    groups.push((top.clone(), grouped));
                }
            }

            let mut children: Children<T, Lower<T>> = CompactMap::new();
            for (top, grouped) in groups {
                let child = build(&grouped, base.clone(), interned)?;
                insert_lower_child_shared(
                    &mut children,
                    top,
                    child.max_depth(),
                    child,
                );
            }
            // `insert_lower_child_shared` has already canonicalised equal
            // suffixes, and every child came from this builder's interner. The
            // generic `new_lower` canonicalisation would only rebuild temporary
            // key/entry vectors here.
            let extensions = intern(new_lower_precanonicalized(children, false), interned)?;
            if includes_base {
                intern(merge_lower(&base, &extensions), interned)
            } else {
                Some(extensions)
            }
        }

        let floor = new_lower_precanonicalized(CompactMap::new(), true);
        let mut interned = SmallVec::<[Arc<Lower<T>>; MAX_INTERNED_NODES]>::new();
        interned.push(floor.clone());
        let lower = build(stacks, floor, &mut interned)?;
        Some(Self {
            inner: new_interface(lower, acc),
        })
    }

    pub fn try_replace_predecessors_below_top(
        &self,
        top: &T,
        mut replacement: impl FnMut(&T) -> Result<Option<T>, ()>,
    ) -> Result<Self, ()> {
        fn append_replacements<T, A>(
            node: &Arc<Lower<T>>,
            replacement: &mut impl FnMut(&T) -> Result<Option<T>, ()>,
            shifted: &mut Children<T, Lower<T>>,
        ) -> Result<(), ()>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            match &**node {
                Lower::Segment(_) => {
                    let predecessor = node.segment_top_value();
                    let Some(target) = replacement(predecessor)? else {
                        return Ok(());
                    };
                    let below = node.segment_rest_arc();
                    insert_lower_child_shared(
                        shifted,
                        target,
                        below.max_depth(),
                        below,
                    );
                }
                Lower::General { children, .. } => {
                    for (predecessor, kids) in children.iter() {
                        let Some(target) = replacement(predecessor)? else {
                            continue;
                        };
                        for (depth, below) in kids.iter() {
                            insert_lower_child_shared(
                                shifted,
                                target.clone(),
                                *depth,
                                below.clone(),
                            );
                        }
                    }
                }
            }
            Ok(())
        }

        let Upper::Interface(interface) = &*self.inner else {
            return Err(());
        };
        let mut shifted = Children::<T, Lower<T>>::new();
        match &*interface.inner {
            Lower::Segment(_) => {
                if interface.inner.segment_top_value() != top {
                    return Ok(Self::empty());
                }
                let below_top = interface.inner.segment_rest_arc();
                append_replacements::<T, A>(&below_top, &mut replacement, &mut shifted)?;
            }
            Lower::General { children, .. } => {
                let Some(top_children) = children.get(top) else {
                    return Ok(Self::empty());
                };
                for below_top in top_children.values() {
                    append_replacements::<T, A>(
                        below_top,
                        &mut replacement,
                        &mut shifted,
                    )?;
                }
            }
        }

        if shifted.is_empty() {
            return Ok(Self::empty());
        }
        let lower = new_lower_precanonicalized(shifted, false);
        Ok(Self {
            inner: new_interface(lower, interface.acc.clone()),
        })
    }

    pub fn try_replace_top_predecessors_with_sequences(
        &self,
        max_lower_prefix: usize,
        mut replacement: impl FnMut(
            &T,
            &T,
            &[T],
        ) -> Result<Option<SmallVec<[T; 4]>>, ()>,
    ) -> Result<Self, ()> {
        fn unique_lower_prefix<T: Clone + Eq + Hash>(
            node: &Arc<Lower<T>>,
            max_len: usize,
        ) -> SmallVec<[T; 8]> {
            let mut prefix = SmallVec::<[T; 8]>::new();
            let mut current = node.clone();
            while prefix.len() < max_len {
                match &*current {
                    Lower::Segment(_) => {
                        prefix.push(current.segment_top_value().clone());
                        current = current.segment_rest_arc();
                    }
                    Lower::General {
                        children,
                        empty: false,
                        ..
                    } if children.len() == 1 => {
                        let (value, kids) = children.iter().next().unwrap();
                        prefix.push(value.clone());
                        if kids.len() != 1 {
                            break;
                        }
                        current = kids.values().next().unwrap().clone();
                    }
                    _ => break,
                }
            }
            prefix
        }

        fn append_replacements<T, A>(
            top: &T,
            node: &Arc<Lower<T>>,
            max_lower_prefix: usize,
            replacement: &mut impl FnMut(
                &T,
                &T,
                &[T],
            ) -> Result<Option<SmallVec<[T; 4]>>, ()>,
            shifted: &mut Children<T, Lower<T>>,
        ) -> Result<(), ()>
        where
            T: Clone + Eq + Hash,
            A: Merge + Clone + Eq + Hash,
        {
            let mut append = |predecessor: &T, below: &Arc<Lower<T>>| -> Result<(), ()> {
                let lower_prefix = unique_lower_prefix(below, max_lower_prefix);
                let Some(pushes) = replacement(top, predecessor, &lower_prefix)? else {
                    return Ok(());
                };
                let (target, prefix) = pushes.split_last().ok_or(())?;
                let below_target = if prefix.is_empty() {
                    below.clone()
                } else {
                    new_segment(SV::from_vec(prefix.to_vec()), below.clone())
                };
                insert_lower_child_shared(
                    shifted,
                    target.clone(),
                    below_target.max_depth(),
                    below_target,
                );
                Ok(())
            };

            match &**node {
                Lower::Segment(_) => {
                    let predecessor = node.segment_top_value();
                    let below = node.segment_rest_arc();
                    append(predecessor, &below)?;
                }
                Lower::General { children, .. } => {
                    for (predecessor, kids) in children.iter() {
                        for below in kids.values() {
                            append(predecessor, below)?;
                        }
                    }
                }
            }
            Ok(())
        }

        let Upper::Interface(interface) = &*self.inner else {
            return Err(());
        };
        let mut shifted = Children::<T, Lower<T>>::new();
        match &*interface.inner {
            Lower::Segment(_) => {
                let top = interface.inner.segment_top_value();
                let below_top = interface.inner.segment_rest_arc();
                append_replacements::<T, A>(
                    top,
                    &below_top,
                    max_lower_prefix,
                    &mut replacement,
                    &mut shifted,
                )?;
            }
            Lower::General { children, .. } => {
                for (top, top_children) in children.iter() {
                    for below_top in top_children.values() {
                        append_replacements::<T, A>(
                            top,
                            below_top,
                            max_lower_prefix,
                            &mut replacement,
                            &mut shifted,
                        )?;
                    }
                }
            }
        }

        if shifted.is_empty() {
            return Ok(Self::empty());
        }
        let lower = new_lower_precanonicalized(shifted, false);
        Ok(Self {
            inner: new_interface(lower, interface.acc.clone()),
        })
    }

    pub fn try_apply_top_stack_effects<'a, I>(&self, effects: I) -> Option<Self>
    where
        I: IntoIterator<Item = (T, usize, &'a [T])>,
        T: 'a,
    {
        const MAX_EFFECTS: usize = 16;
        let effects = SmallVec::<[(T, usize, &'a [T]); MAX_EFFECTS]>::from_iter(effects);
        if effects.is_empty()
            || effects.len() > MAX_EFFECTS
            || effects
                .iter()
                .any(|(_, pop, pushes)| *pop > 1 || pushes.is_empty())
        {
            return None;
        }

        match &*self.inner {
            Upper::Interface(interface) => {
                let Lower::General { children, .. } = &*interface.inner else {
                    return None;
                };

                let mut shifted_children: Children<T, Lower<T>> = CompactMap::new();
                for (from, pop, pushes) in &effects {
                    let Some(kids) = children.get(from) else {
                        continue;
                    };
                    let (target, pushed_prefix) = pushes.split_last()?;
                    let mut prefix = SmallVec::<[T; 8]>::new();
                    if *pop == 0 {
                        prefix.push(from.clone());
                    }
                    prefix.extend(pushed_prefix.iter().cloned());

                    for child in kids.values() {
                        let below_target = if prefix.is_empty() {
                            child.clone()
                        } else {
                            new_segment(SV::from_vec(prefix.to_vec()), child.clone())
                        };
                        insert_lower_child_shared(
                            &mut shifted_children,
                            target.clone(),
                            below_target.max_depth(),
                            below_target,
                        );
                    }
                }

                if shifted_children.is_empty() {
                    return Some(Self::empty());
                }
                let shifted_root = new_lower_precanonicalized(shifted_children, false);
                Some(Self {
                    inner: new_interface(shifted_root, interface.acc.clone()),
                })
            }
            Upper::Branch(branch) => {
                fn wrap_upper<T, A>(value: T, child: Arc<Upper<T, A>>) -> Arc<Upper<T, A>>
                where
                    T: Clone + Eq + Hash,
                    A: Merge + Clone + Eq + Hash,
                {
                    let depth = child.max_depth();
                    new_branch(
                        CompactMap::unit(value, CompactOrdMap::unit(depth, child)),
                        None,
                    )
                }

                let mut shifted_children: Children<T, Upper<T, A>> = CompactMap::new();
                for (from, pop, pushes) in &effects {
                    let Some(kids) = branch.children.get(from) else {
                        continue;
                    };
                    let (target, pushed_prefix) = pushes.split_last()?;
                    for child in kids.values() {
                        let mut below_target = child.clone();
                        if *pop == 0 {
                            below_target = wrap_upper(from.clone(), below_target);
                        }
                        for value in pushed_prefix.iter().cloned() {
                            below_target = wrap_upper(value, below_target);
                        }
                        insert_upper_child_shared(
                            &mut shifted_children,
                            target.clone(),
                            below_target,
                        );
                    }
                }

                if shifted_children.is_empty() {
                    return Some(Self::empty());
                }
                Some(Self {
                    inner: new_branch(shifted_children, None),
                })
            }
        }
    }

    pub fn apply_guarded_stack_effects_to_bounded_paths<'a, I, G>(
        &self,
        effects: I,
        max_paths: usize,
        max_depth: usize,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = (G, usize, &'a [T])>,
        G: IntoIterator<Item = (usize, &'a [T])>,
        T: 'a,
    {
        const MAX_PATHS: usize = 16;
        const MAX_EFFECTS: usize = 32;
        const MAX_GUARDS: usize = 8;
        const MAX_DEPTH: usize = 64;

        if max_paths == 0
            || max_paths > MAX_PATHS
            || max_depth == 0
            || max_depth > MAX_DEPTH
            || self.max_depth() as usize > max_depth
        {
            return None;
        }

        let mut bounded_effects = SmallVec::<
            [(SmallVec<[(usize, &'a [T]); MAX_GUARDS]>, usize, &'a [T]); MAX_EFFECTS],
        >::new();
        for (guards, pop, pushes) in effects {
            if bounded_effects.len() == bounded_effects.capacity() {
                return None;
            }
            let mut bounded_guards = SmallVec::<[(usize, &'a [T]); MAX_GUARDS]>::new();
            for guard in guards {
                if bounded_guards.len() == bounded_guards.capacity() {
                    return None;
                }
                bounded_guards.push(guard);
            }
            bounded_effects.push((bounded_guards, pop, pushes));
        }
        if bounded_effects.is_empty() {
            return Some(Self::empty());
        }

        let mut outputs = SmallVec::<[(SmallVec<[T; MAX_DEPTH]>, A); MAX_PATHS]>::new();
        let mut overflowed = false;
        let complete = self.for_each_stack_top_first_bounded(max_paths, |top_first, acc| {
            if overflowed {
                return;
            }
            for (guards, pop, pushes) in &bounded_effects {
                if *pop > top_first.len() {
                    continue;
                }
                let allowed = guards.iter().all(|(guard_pop, guard_states)| {
                    *guard_pop < top_first.len()
                        && guard_states
                            .iter()
                            .any(|candidate| candidate == &top_first[*guard_pop])
                });
                if !allowed {
                    continue;
                }

                let keep = top_first.len() - *pop;
                if keep + pushes.len() > max_depth {
                    overflowed = true;
                    return;
                }
                let mut next = SmallVec::<[T; MAX_DEPTH]>::new();
                for value in top_first[*pop..].iter().rev() {
                    next.push(value.clone());
                }
                next.extend(pushes.iter().cloned());

                if let Some((_, existing_acc)) =
                    outputs.iter_mut().find(|(existing, _)| existing == &next)
                {
                    *existing_acc = existing_acc.merge(acc);
                    continue;
                }
                if outputs.len() == max_paths || outputs.len() == outputs.capacity() {
                    overflowed = true;
                    return;
                }
                outputs.push((next, acc.clone()));
            }
        });
        if !complete || overflowed {
            return None;
        }
        if outputs.is_empty() {
            return Some(Self::empty());
        }

        let common_acc = outputs[0].1.clone();
        if outputs.iter().any(|(_, acc)| acc != &common_acc) {
            return None;
        }
        let mut slices = SmallVec::<[&[T]; MAX_PATHS]>::new();
        for (stack, _) in &outputs {
            slices.push(stack.as_slice());
        }
        Self::from_small_stack_slices_shared_prefix(&slices, common_acc)
    }
}

#[cfg(test)]
mod tests ;
