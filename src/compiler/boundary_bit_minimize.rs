//! Exact finite-coordinate, guard-preserving weighted DAG quotient.
//!
//! The declared universe is `rows * 64` coordinates. Values outside it are
//! not represented; callers must decode coordinates before publication.
//! Backward pushing restricts incoming edges to productive target domains.
//! States can then share a representation if they agree on their overlapping
//! domains, with identical productive targets on shared labels. A DEFAULT row
//! additionally retains its complete explicit-label guard: an empty explicit
//! edge must never become an absent edge and accidentally enable DEFAULT.
//!
//! This is deliberately a sufficient quotient, not a minimum-size guarantee.
//! No graph or weight is approximated. Unsupported shapes and resource budgets
//! return None; the caller must use its unchanged exact implementation.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;
use rustc_hash::{FxHashMap,FxHasher};
use std::hash::{Hash,Hasher};
use crate::automata::weighted::dwa::{DWA, DWAState};
use crate::ds::weight::Weight;

#[derive(Debug, Default)]
pub struct BitMinimizeProfile {
    pub input_states: usize,
    pub output_states: usize,
    pub input_edges: usize,
    pub output_edges: usize,
    pub weights: usize,
    pub compatibility_attempts: usize,
    pub exact_hits: usize,
    pub atom_count: usize,
    pub quotient_ms: f64,
    pub convert_ms: f64,
    pub push_ms: f64,
    pub merge_ms: f64,
    pub reconstruct_ms: f64,
    pub observation_weights_before: usize,
    pub observation_weights_used: usize,
}

const MAX_POINTS: usize = 4096;
const MAX_STATES: usize = 200_000;
const MAX_EDGES: usize = 4_000_000;
const MAX_MASKS: usize = 500_000;
// Both the intern table and indexed pool retain bit rows. This caps their
// principal word storage, not allocator overhead or total process RSS.
const MAX_MASK_WORDS: usize = 8 * 1024 * 1024;
type Mask = u32;
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Edge { label: i32, target: u32, mask: Mask }
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Signature { final_mask: Mask, edges: Vec<Edge> }
struct State { final_mask: Mask, edges: Vec<Edge>, guarded: bool }
struct Group { domain: Mask, signature: Signature, guarded: bool }

enum MaskBucket { One(Mask), Many(Vec<Mask>) }

#[cfg(test)]
mod mask_storage_tests {
    use super::*;

    #[test]
    fn indexed_and_sparse_masks_preserve_ids_under_forced_collisions() {
        let mut seed = 93u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        for rows in [1, 9, 44, 64] {
            for (indexed, sparse, force_collisions) in [
                (true, false, false), (false, true, false),
                (true, true, false), (true, true, true),
            ] {
                let mut reference = Masks::with_storage(rows, false, false);
                let mut candidate = Masks::with_storage(rows, indexed, sparse);
                if force_collisions {
                    candidate.fingerprint_mask = 0;
                    candidate.indexed.clear();
                    candidate.indexed.insert(0, MaskBucket::Many(vec![0, 1]));
                }
                let mut inputs = vec![vec![0; rows], vec![u64::MAX; rows]];
                for _ in 0..96 {
                    inputs.push((0..rows).map(|_| {
                        let v = next();
                        if v % 5 == 0 { v } else { 0 }
                    }).collect());
                }
                for _ in 0..3 {
                    for value in &inputs {
                        assert_eq!(reference.intern(value.clone()), candidate.intern(value.clone()));
                    }
                }
                for _ in 0..512 {
                    let n = reference.values.len() as u64;
                    let a = (next() % n) as Mask;
                    let b = (next() % n) as Mask;
                    let d1 = (next() % n) as Mask;
                    let d2 = (next() % n) as Mask;
                    assert_eq!(reference.equal_on_overlap(a, b, d1, d2),
                               candidate.equal_on_overlap(a, b, d1, d2));
                    assert_eq!(reference.and(a, b), candidate.and(a, b));
                    assert_eq!(reference.or(a, b), candidate.or(a, b));
                    assert_eq!(reference.popcount(a), candidate.popcount(a));
                }
                assert_eq!(reference.values, candidate.values);
                assert_eq!(reference.cardinalities, candidate.cardinalities);
                let before = candidate.values.len();
                assert!(candidate.intern(vec![0; rows + 1]).is_none());
                assert_eq!(candidate.values.len(), before, "invalid input cannot publish a row");
                if indexed { assert!(candidate.ids.is_empty(), "indexed mode must not duplicate bit rows"); }
                if sparse { assert_eq!(candidate.values.len(), candidate.nonzero_words.len()); }
                if force_collisions { assert_eq!(candidate.indexed.len(), 1); }
            }
        }
    }
}

struct Masks {
    rows: usize,
    points: Option<Vec<(u32,u32)>>,
    point_rows: Vec<(u32,usize,usize)>,
    decoded_atoms: Option<Vec<Weight>>,
    decoded_point_rows: Option<Vec<(u32, Vec<(u32, usize)>)>>,
    values: Vec<Box<[u64]>>,
    cardinalities: Vec<u32>,
    ids: FxHashMap<Box<[u64]>, Mask>,
    indexed: FxHashMap<u64, MaskBucket>,
    nonzero_words: Vec<Box<[u16]>>,
    use_index: bool,
    borrow_rows: bool,
    use_sparse_overlap: bool,
    #[cfg(test)]
    fingerprint_mask: u64,
    and_cache: FxHashMap<(Mask, Mask), Mask>,
    source_cache: FxHashMap<usize, (Weight, Mask)>,
}

impl Masks {
    fn new(rows: usize) -> Self {
        Self::with_storage(rows,
            crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_MIN_MASK_INDEX"),
            std::env::var_os("GLRMASK_BOUNDARY_MIN_SPARSE_OVERLAP").is_some())
    }

    fn with_storage(rows: usize, use_index: bool, use_sparse_overlap: bool) -> Self {
        let mut result = Self {
            rows, points: None, point_rows: Vec::new(), decoded_atoms: None, decoded_point_rows: None, values: Vec::new(), cardinalities: Vec::new(), ids: FxHashMap::default(),
            indexed: FxHashMap::default(), nonzero_words: Vec::new(), use_index,
            borrow_rows: crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_MIN_BORROWED_ROWS"),
            // Decline only this storage shortcut, not an otherwise valid
            // minimization, when word offsets do not fit the compact index.
            use_sparse_overlap: use_sparse_overlap && rows <= u16::MAX as usize,
            #[cfg(test)]
            fingerprint_mask: u64::MAX,
            and_cache: FxHashMap::default(), source_cache: FxHashMap::default(),
        };
        assert_eq!(result.intern(vec![0; rows]).expect("initial empty mask fits"), 0);
        assert_eq!(result.intern(vec![u64::MAX; rows]).expect("initial full mask fits"), 1);
        result
    }
    fn from_finite_points(domain:&Weight)->Option<Self> {
        if domain.is_empty() || domain.is_full() {return None;}
        let mut points=Vec::new();
        for (lo,hi,tokens) in domain.range_entries() {
            let volume=(u128::from(hi)-u128::from(lo)+1)*(tokens.len() as u128);
            if volume>(MAX_POINTS-points.len()) as u128 {return None;}
            for row in lo..=hi {for token in tokens.iter(){points.push((row,token));}}
        }
        let mut masks=Self::new(points.len().div_ceil(64));
        let mut start=0usize;
        while start<points.len() {
            let row=points[start].0;
            let end=start+points[start..].partition_point(|point|point.0==row);
            masks.point_rows.push((row,start,end));start=end;
        }
        masks.points=Some(points);Some(masks)
    }

    fn fingerprint(&self, value: &[u64]) -> u64 {
        let mut hasher = FxHasher::default();
        value.hash(&mut hasher);
        let hash = hasher.finish();
        #[cfg(test)]
        { return hash & self.fingerprint_mask; }
        #[cfg(not(test))]
        { hash }
    }

    fn lookup(&self, value: &[u64], fingerprint: u64) -> Option<Mask> {
        if value.len() != self.rows { return None; }
        if self.use_index {
            if let Some(bucket) = self.indexed.get(&fingerprint) {
                // A fingerprint is never an equality certificate. The owned
                // immutable row in `values` is compared in full on every hit.
                match bucket {
                    MaskBucket::One(id) => {
                        if self.values[*id as usize].as_ref() == value {
                            return Some(*id);
                        }
                    }
                    MaskBucket::Many(ids) => {
                        for &id in ids {
                            if self.values[id as usize].as_ref() == value {
                                return Some(id);
                            }
                        }
                    }
                }
            }
        } else if let Some(&id) = self.ids.get(value) { return Some(id); }
        None
    }
    /// Interner membership is exact even under a fingerprint collision.
    /// Only a new row is allocated; the policy is fixed once per invocation.
    fn intern_slice(&mut self, value: &[u64]) -> Option<Mask> {
        if value.len()!=self.rows{return None;}
        let fingerprint=if self.use_index{self.fingerprint(value)}else{0};
        if let Some(id)=self.lookup(value,fingerprint){return Some(id);}
        self.insert_new(value.to_vec(),fingerprint)
    }
    fn intern(&mut self, value: Vec<u64>) -> Option<Mask> {
        if value.len()!=self.rows{return None;}
        let fingerprint=if self.use_index{self.fingerprint(&value)}else{0};
        if let Some(id)=self.lookup(&value,fingerprint){return Some(id);}
        self.insert_new(value,fingerprint)
    }
    // The caller has checked width and exact non-membership. Existing resource
    // bounds and insertion/ID order are the same for borrowed and owned paths.
    fn insert_new(&mut self, value: Vec<u64>, fingerprint: u64) -> Option<Mask> {
        if self.values.len() >= MAX_MASKS
            || (self.values.len() + 1).checked_mul(self.rows)? > MAX_MASK_WORDS
        { return None; }
        let value = value.into_boxed_slice();
        let id = self.values.len() as Mask;
        self.cardinalities.push(value.iter().map(|word|word.count_ones()).sum());
        if self.use_sparse_overlap {
            self.nonzero_words.push(value.iter().enumerate()
                .filter_map(|(i, &word)| (word != 0).then_some(i as u16)).collect());
        }
        if self.use_index {
            self.values.push(value);
            match self.indexed.entry(fingerprint) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(MaskBucket::One(id));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    match entry.get_mut() {
                        MaskBucket::One(previous) => {
                            let previous = *previous;
                            *entry.get_mut() = MaskBucket::Many(vec![previous, id]);
                        }
                        MaskBucket::Many(ids) => ids.push(id),
                    }
                }
            }
        } else {
            self.values.push(value.clone()); self.ids.insert(value, id);
        }
        Some(id)
    }
    fn boolean_row(&mut self,a:Mask,b:Mask,intersection:bool)->Option<Mask> {
        if self.borrow_rows && self.rows<=64 {
            let mut temporary=[0u64;64];
            for (i,dst) in temporary[..self.rows].iter_mut().enumerate() {
                *dst=if intersection {self.values[a as usize][i]&self.values[b as usize][i]}
                    else {self.values[a as usize][i]|self.values[b as usize][i]};
            }
            self.intern_slice(&temporary[..self.rows])
        }else{
            let bits=self.values[a as usize].iter().zip(&self.values[b as usize])
                .map(|(&a,&b)|if intersection{a&b}else{a|b}).collect();
            self.intern(bits)
        }
    }
    fn import(&mut self, weight: &Weight) -> Option<Mask> {
        if weight.is_empty() { return Some(0); }
        if weight.is_full() { return Some(1); }
        if let Some((_, id)) = self.source_cache.get(&weight.ptr_key()) { return Some(*id); }
        let mut bits = vec![0u64; self.rows];
        if let Some(points)=&self.points {
            if points.len().checked_mul(self.source_cache.len()+1)?>64_000_000 {return None;}
            // Intersect sorted coordinate rows and token intervals directly.
            // Every interval maps to a contiguous span of point indices even
            // when the original token IDs are sparse. This computes exactly
            // the same membership matrix without scanning every point for a
            // weight that only addresses one small row/range.
            for (lo,hi,tokens) in weight.range_entries() {
                let first=self.point_rows.partition_point(|(row,_,_)|*row<lo);
                for &(row,start,end) in &self.point_rows[first..] {
                    if row>hi {break;}
                    let slice=&points[start..end];
                    for range in tokens.ranges() {
                        let a=start+slice.partition_point(|p|p.1<*range.start());
                        let b=start+slice.partition_point(|p|p.1<=*range.end());
                        if a==b {continue;}
                        let left=u64::MAX<<(a%64);
                        let right=u64::MAX>>(63-(b-1)%64);
                        if a/64==(b-1)/64 {bits[a/64]|=left&right;} else {
                            bits[a/64]|=left;
                            bits[a/64+1..(b-1)/64].fill(u64::MAX);
                            bits[(b-1)/64]|=right;
                        }
                    }
                }
            }
            let id=self.intern(bits)?;
            self.source_cache.insert(weight.ptr_key(),(weight.clone(),id));return Some(id);
        }
        for (lo, hi, tokens) in weight.range_entries() {
            if hi as usize >= self.rows { return None; }
            let mut word = 0u64;
            for range in tokens.ranges() {
                let (a, b) = (*range.start(), *range.end());
                if b >= 64 { return None; }
                word |= (u64::MAX << a) & (u64::MAX >> (63-b));
            }
            for row in lo..=hi { bits[row as usize] |= word; }
        }
        let id = self.intern(bits)?;
        self.source_cache.insert(weight.ptr_key(), (weight.clone(), id));
        Some(id)
    }
    fn and(&mut self, a: Mask, b: Mask) -> Option<Mask> {
        if a == b || b == 1 || a == 0 { return Some(a); }
        if a == 1 || b == 0 { return Some(b); }
        let key = if a < b { (a,b) } else { (b,a) };
        if let Some(&id) = self.and_cache.get(&key) { return Some(id); }
        let id = self.boolean_row(a,b,true)?;
        if self.and_cache.len() < 262_144 { self.and_cache.insert(key,id); }
        Some(id)
    }
    fn or(&mut self, a: Mask, b: Mask) -> Option<Mask> {
        if a == b || b == 0 || a == 1 { return Some(a); }
        if a == 0 || b == 1 { return Some(b); }
        self.boolean_row(a,b,false)
    }
    fn equal_on_overlap(&self, a: Mask, b: Mask, d1: Mask, d2: Mask) -> bool {
        if a == b { return true; }
        if self.use_sparse_overlap {
            let left = &self.nonzero_words[d1 as usize];
            let right = &self.nonzero_words[d2 as usize];
            let selected = if left.len() <= right.len() { left } else { right };
            // Outside either domain's nonzero-word support, d1 & d2 is zero.
            // Thus the omitted terms in (a XOR b) & d1 & d2 are exactly zero.
            return selected.iter().all(|&word| {
                let i = word as usize;
                ((self.values[a as usize][i] ^ self.values[b as usize][i])
                    & self.values[d1 as usize][i] & self.values[d2 as usize][i]) == 0
            });
        }
        let (a,b,d1,d2) = (&self.values[a as usize], &self.values[b as usize],
            &self.values[d1 as usize], &self.values[d2 as usize]);
        (0..self.rows).all(|i| ((a[i] ^ b[i]) & d1[i] & d2[i]) == 0)
    }
    fn popcount(&self, id: Mask) -> u32 {
        self.cardinalities[id as usize]
    }
    fn export(&self, id: Mask) -> Weight {
        if id == 0 { return Weight::empty(); }
        if let Some(rows)=&self.decoded_point_rows {
            let mask=&self.values[id as usize];
            return Weight::from_per_tsid_token_sets(rows.iter().filter_map(|(row, points)| {
                let tokens=points.iter().filter_map(|&(token,atom)|
                    (mask[atom/64]&(1u64<<(atom%64))!=0).then_some(token))
                    .collect::<range_set_blaze::RangeSetBlaze<u32>>();
                (!tokens.is_empty()).then_some((*row,tokens))
            }));
        }
        if let Some(atoms)=&self.decoded_atoms {
            let mut selected=Vec::new();
            for (word_index,&word) in self.values[id as usize].iter().enumerate() {
                let mut bits=word;
                while bits!=0 {
                    let bit=bits.trailing_zeros() as usize;bits&=bits-1;
                    if let Some(atom)=atoms.get(word_index*64+bit){selected.push(atom);}
                }
            }
            return Weight::union_all(selected);
        }
        if let Some(points)=&self.points {
            let mut rows=BTreeMap::<u32,Vec<u32>>::new();
            for (word_index,&word) in self.values[id as usize].iter().enumerate() {
                let mut bits=word;
                while bits!=0 {
                    let bit=bits.trailing_zeros() as usize;bits&=bits-1;
                    if let Some(&(row,token))=points.get(word_index*64+bit){rows.entry(row).or_default().push(token);}
                }
            }
            return Weight::from_per_tsid_token_sets(rows.into_iter().map(|(row,tokens)|(row,tokens.into_iter().collect())));
        }
        // Export even ALL as an explicit finite universe. Its caller's decoder
        // determines the meaning of padding, never the Weight::all sentinel.
        Weight::from_per_tsid_token_sets(self.values[id as usize].iter().enumerate()
            .filter(|(_, word)| **word != 0).map(|(row,&word)| {
                let tokens = (0..64u32).filter(|bit| word & (1u64 << bit) != 0).collect();
                (row as u32,tokens)
            }))
    }

    /// A finite disjoint atom decoder is a sparse Boolean incidence map:
    /// each original point has exactly one owning atom. Decode by that bit
    /// instead of constructing unions of intermediate range-weight objects.
    /// Unsupported/non-disjoint inputs keep the unchanged union decoder.
    fn prepare_point_decoder(&mut self) -> Option<()> {
        self.decoded_point_rows=None;
        let atoms=self.decoded_atoms.as_ref()?;
        let mut points=Vec::<(u32,u32,usize)>::new();
        for (atom,weight) in atoms.iter().enumerate() {
            if atom>=self.rows*64 || weight.is_full(){return None;}
            for (lo,hi,tokens) in weight.range_entries() {
                let count=(u128::from(hi)-u128::from(lo)+1)*(tokens.len() as u128);
                if count>(32768-points.len()) as u128{return None;}
                for row in lo..=hi {for token in tokens.iter(){points.push((row,token,atom));}}
            }
        }
        points.sort_unstable();
        if points.windows(2).any(|p|p[0].0==p[1].0 && p[0].1==p[1].1){return None;}
        let mut rows=Vec::<(u32,Vec<(u32,usize)>)>::new();
        for (row,token,atom) in points {
            if rows.last().is_none_or(|(last,_)|*last!=row){rows.push((row,Vec::new()));}
            rows.last_mut().unwrap().1.push((token,atom));
        }
        self.decoded_point_rows=Some(rows);
        Some(())
    }

    /// Partition the finite coordinate universe by membership in every input
    /// weight. Exact signature equality, not a hash value, decides equality.
    /// Boolean operations on input weights cannot distinguish these atoms.
    /// Only the deterministic post-normalization graph will use this quotient.
    fn compress_input_observations(&mut self,domain:Mask)->Option<(Vec<Mask>,usize)> {
        self.compress_selected_observations(domain,None)
    }

    /// Every later predicate is a Boolean expression in the referenced masks.
    /// Extra, unused interner values must not refine their observation classes.
    /// Omitted IDs receive an invalid mapping; callers must map referenced IDs only.
    fn compress_selected_observations(&mut self,domain:Mask,observed:Option<&[Mask]>)
        ->Option<(Vec<Mask>,usize)>
    {
        if observed.is_some_and(|ids| ids.iter().any(|&id|id as usize>=self.values.len())) {
            return None;
        }
        let original_count = self.decoded_atoms.as_ref().map_or_else(
            || self.points.as_ref().map_or(self.rows*64, Vec::len), Vec::len,
        );
        let selected=(0..original_count).filter(|&p|
            self.values[domain as usize][p/64]&(1u64<<(p%64))!=0).collect::<Vec<_>>();
        if selected.is_empty(){return None;}
        let count=selected.len();
        let mut selected_id=vec![usize::MAX;original_count];
        for (id,&point) in selected.iter().enumerate(){selected_id[point]=id;}
        let predicate_count=observed.map_or(self.values.len(),<[Mask]>::len);
        let cols=predicate_count.div_ceil(64);
        if count.checked_mul(cols)?.checked_mul(8)?>64*1024*1024{return None;}
        let mut signatures=vec![vec![0u64;cols];count];
        for column in 0..predicate_count {
            let old_id=observed.map_or(column,|ids|ids[column]as usize);
            let mask=&self.values[old_id];
            for (word_index,&word) in mask.iter().enumerate() {
                let mut bits=word;
                while bits!=0 {
                    let bit=bits.trailing_zeros() as usize;bits&=bits-1;
                    let point=word_index*64+bit;
                    if point<original_count && selected_id[point]!=usize::MAX {
                        signatures[selected_id[point]][column/64]|=1u64<<(column%64);
                    }
                }
            }
        }
        let mut ids=FxHashMap::<Vec<u64>,usize>::default();
        let mut members=Vec::<Vec<usize>>::new();
        for (point,signature) in signatures.into_iter().enumerate() {
            let atom=if let Some(&id)=ids.get(&signature){id}else{
                let id=members.len();ids.insert(signature,id);members.push(Vec::new());id
            };
            members[atom].push(selected[point]);
        }
        if members.len().div_ceil(64)>=self.rows{return None;}
        let atom_count=members.len();
        let mut compressed=Self::new(atom_count.div_ceil(64));
        let mut mapping=Vec::with_capacity(self.values.len());
        if observed.is_some(){mapping.resize(self.values.len(),Mask::MAX);}
        for column in 0..predicate_count {
            let old_id=observed.map_or(column,|ids|ids[column]as usize);
            let old=&self.values[old_id];
            let mut bits=vec![0u64;compressed.rows];
            for (atom,points) in members.iter().enumerate() {
                let representative=points[0];
                if old[representative/64]&(1u64<<(representative%64))!=0 {
                    bits[atom/64]|=1u64<<(atom%64);
                }
            }
            let new_id=compressed.intern(bits)?;
            if observed.is_some(){mapping[old_id]=new_id;}else{mapping.push(new_id);}
        }
        let decoded = if let Some(atoms) = self.decoded_atoms.as_ref() {
            // Composition of two exact disjoint partitions is exact: the
            // new observation atom decodes to the union of its old atoms.
            members.iter().map(|members| Weight::union_all(
                members.iter().map(|&point| &atoms[point]),
            )).collect()
        } else {
            members.iter().map(|members| {
            let mut rows=BTreeMap::<u32,Vec<u32>>::new();
            for &point in members {
                let (row,token)=self.points.as_ref().map_or(((point/64) as u32,(point%64) as u32),|points|points[point]);
                rows.entry(row).or_default().push(token);
            }
            Weight::from_per_tsid_token_sets(rows.into_iter().map(|(row,tokens)|(row,tokens.into_iter().collect())))
        }).collect()
        };
        compressed.decoded_atoms=Some(decoded);
        *self=compressed;
        Some((mapping,atom_count))
    }
}

fn compatible(group: &Group, state: &Signature, domain: Mask, guarded: bool, masks: &Masks) -> bool {
    if group.guarded != guarded { return false; }
    if guarded && (group.signature.edges.len() != state.edges.len()
        || !group.signature.edges.iter().zip(&state.edges).all(|(a,b)| a.label == b.label)) {
        return false;
    }
    if !masks.equal_on_overlap(group.signature.final_mask, state.final_mask, group.domain, domain) {
        return false;
    }
    let (a,b) = (&group.signature.edges,&state.edges);
    let (mut i,mut j) = (0,0);
    while i<a.len() || j<b.len() {
        let (left,right) = match (a.get(i),b.get(j)) {
            (Some(x),Some(y)) if x.label==y.label => { i+=1;j+=1;(Some(x),Some(y)) },
            (Some(x),Some(y)) if x.label<y.label => { i+=1;(Some(x),None) },
            (_,Some(y)) => { j+=1;(None,Some(y)) },
            (Some(x),None) => { i+=1;(Some(x),None) },
            _=> unreachable!(),
        };
        let l=left.map_or(0,|e|e.mask); let r=right.map_or(0,|e|e.mask);
        if l!=0 && r!=0 && left.unwrap().target!=right.unwrap().target { return false; }
        if !masks.equal_on_overlap(l,r,group.domain,domain) { return false; }
    }
    true
}

fn merge(group: &mut Group, state: &Signature, domain: Mask, masks: &mut Masks) -> Option<()> {
    group.domain=masks.or(group.domain,domain)?;
    group.signature.final_mask=masks.or(group.signature.final_mask,state.final_mask)?;
    let old=std::mem::take(&mut group.signature.edges);
    let mut joined=Vec::with_capacity(old.len().max(state.edges.len()));
    let (mut i,mut j)=(0,0);
    while i<old.len() || j<state.edges.len() {
        match (old.get(i),state.edges.get(j)) {
            (Some(a),Some(b)) if a.label==b.label => {
                let mask=masks.or(a.mask,b.mask)?;
                joined.push(Edge{label:a.label,target:if a.mask!=0 {a.target} else {b.target},mask});
                i+=1;j+=1;
            },
            (Some(a),Some(b)) if a.label<b.label => {joined.push(a.clone());i+=1;},
            (_,Some(b))=>{joined.push(b.clone());j+=1;},
            (Some(a),None)=>{joined.push(a.clone());i+=1;},
            _=>unreachable!(),
        }
    }
    group.signature.edges=joined;
    Some(())
}

pub fn minimize_finite_bits(
    input: &DWA, rows: usize, default_label: i32,
) -> Option<(DWA, BitMinimizeProfile)> {
    if !(1..=64).contains(&rows){return None;}
    minimize_core(input,Masks::new(rows),default_label,false)
}

/// Equivalent from the root whenever `domain` contains every possible prefix
/// mask coordinate. Unlike an atom quotient this uses one bit per actual point,
/// so no source-predicate partition construction is necessary. The caller may
/// obtain a conservative domain using ordinary backward accepting support.
pub fn minimize_finite_points(
    input:&DWA, domain:&Weight, default_label:i32,
)->Option<(DWA,BitMinimizeProfile)> {
    minimize_core(input,Masks::from_finite_points(domain)?,default_label,false)
}

/// A cheap sufficient coordinate universe: every prefix contribution is an
/// incoming path weight intersected with a final weight, hence is contained
/// in the union of ALL final weights. No reachability or DEFAULT assumption
/// is required. Extra, globally unreachable points merely reduce compression.
/// Decline if that conservative universe exceeds the finite-point budget;
/// callers may then derive a tighter domain or use their unchanged compiler.
pub fn minimize_finite_final_union(
    input:&DWA, default_label:i32,
)->Option<(DWA,BitMinimizeProfile)> {
    minimize_final_union_impl(input,default_label,false)
}

/// As above, but replace the input-weight Boolean algebra by its exact atoms
/// before performing the same guard-preserving minimization.
pub fn minimize_finite_final_atoms(
    input:&DWA, default_label:i32,
)->Option<(DWA,BitMinimizeProfile)> {
    minimize_final_union_impl(input,default_label,true)
}

// Cardinality can be checked from disjoint compressed ranges without
// enumerating a potentially huge token/state product. Union is monotone, so
// exceeding the budget at any batch proves the final universe is too large.
fn fits_finite_point_budget(weight: &Weight, limit: usize) -> bool {
    if weight.is_full() { return false; }
    let mut count = 0u128;
    for (lo, hi, tokens) in weight.range_entries() {
        count += (u128::from(hi) - u128::from(lo) + 1) * tokens.len() as u128;
        if count > limit as u128 { return false; }
    }
    true
}

fn minimize_final_union_impl(
    input:&DWA, default_label:i32, compress:bool,
)->Option<(DWA,BitMinimizeProfile)> {
    if input.states().is_empty() || input.states().len() > MAX_STATES { return None; }
    let mut finals=FxHashMap::<usize,&Weight>::default();
    let mut batch = Vec::<&Weight>::with_capacity(65);
    let mut domain = Weight::empty();
    let mut edges = 0usize;
    for state in input.states() {
        edges = edges.checked_add(state.transitions.len())?;
        if edges > MAX_EDGES { return None; }
        if let Some(weight)=&state.final_weight {
            if !finals.contains_key(&weight.ptr_key()) {
                if !fits_finite_point_budget(weight, MAX_POINTS) { return None; }
                finals.insert(weight.ptr_key(), weight);
                batch.push(weight);
                if batch.len() == 64 {
                    let next = Weight::union_all(batch.iter().copied().chain(std::iter::once(&domain)));
                    if !fits_finite_point_budget(&next, MAX_POINTS) { return None; }
                    batch.clear();
                    domain = next;
                }
            }
        }
    }
    batch.push(&domain);
    let domain = Weight::union_all(batch);
    if !fits_finite_point_budget(&domain, MAX_POINTS) { return None; }
    minimize_core(input,Masks::from_finite_points(&domain)?,default_label,compress)
}

fn minimize_core(input:&DWA,mut masks:Masks,default_label:i32,compress:bool)->Option<(DWA,BitMinimizeProfile)> {
    let started=Instant::now();
    let n=input.states().len();
    if n==0 || n>MAX_STATES { return None; }
    if input.start_state() as usize>=n { return None; }
    let mut states=Vec::with_capacity(n);
    let mut indegree=vec![0usize;n];
    let mut edge_count=0usize;
    for state in input.states() {
        let final_mask=match state.final_weight.as_ref() {Some(w)=>masks.import(w)?,None=>0};
        let mut edges=Vec::with_capacity(state.transitions.len());
        let mut guarded=false;
        for (label,target,w) in state.transitions.entries() {
            if target as usize>=n { return None; }
            guarded|=label==default_label;
            indegree[target as usize]+=1;
            edges.push(Edge{label,target,mask:masks.import(w)?});
        }
        edge_count+=edges.len(); if edge_count>MAX_EDGES {return None;}
        states.push(State{final_mask,edges,guarded});
    }
    minimize_prepared(states, masks, indegree, edge_count, input.start_state(),
        started.elapsed().as_secs_f64() * 1000.0, compress)
}

/// Use the smaller exact observation algebra by default. Explicit zero keeps
/// the historical all-interned-weights implementation for diagnostic A/B runs.
fn referenced_observations_enabled() -> bool {
    std::env::var("GLRMASK_BOUNDARY_MIN_REFERENCED_OBSERVATIONS")
        .map_or(true, |value| !matches!(value.trim(), "0" | "false" | "off"))
}

fn minimize_prepared(
    states: Vec<State>, masks: Masks, indegree: Vec<usize>,
    edge_count: usize, start_state: u32, convert_ms: f64, compress: bool,
) -> Option<(DWA, BitMinimizeProfile)> {
    minimize_prepared_with_observations(states, masks, indegree, edge_count,
        start_state, convert_ms, compress, referenced_observations_enabled())
}

fn minimize_prepared_with_observations(
    mut states: Vec<State>, mut masks: Masks, mut indegree: Vec<usize>,
    edge_count: usize, start_state: u32, convert_ms: f64, compress: bool,
    referenced_only: bool,
) -> Option<(DWA, BitMinimizeProfile)> {
    const MAX_ATTEMPTS: usize = 20_000_000;
    let n = states.len();
    let mut todo:VecDeque<usize>=(0..n).filter(|&s|indegree[s]==0).collect();
    let mut topo=Vec::with_capacity(n);
    while let Some(s)=todo.pop_front() {
        topo.push(s);
        for edge in &states[s].edges {
            let degree=&mut indegree[edge.target as usize]; *degree-=1;
            if *degree==0 {todo.push_back(edge.target as usize);}
        }
    }
    if topo.len()!=n {return None;}
    let mut profile=BitMinimizeProfile{input_states:n,input_edges:edge_count,convert_ms,..Default::default()};
    let started=Instant::now();
    let mut needed=vec![0;n];
    let mut heights=vec![0usize;n];
    let dedup_support=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_MIN_DEDUP_SUPPORT");
    let mut seen_contribution=Vec::<u32>::new();
    let mut duplicate_contributions=0usize;
    for &s in topo.iter().rev() {
        // Repeated identical contributions are common in substituted parser
        // templates. Union is idempotent: retain an existing ID until a second
        // distinct nonzero set appears, then allocate one accumulator. This
        // avoids both per-edge and per-singleton-state intermediate allocation.
        let mut support=states[s].final_mask;
        let mut accumulator:Option<Vec<u64>>=None;
        let dedup_this_row=dedup_support && states[s].edges.len()>=8;
        let epoch=s as u32+1;
        if dedup_this_row {
            seen_contribution.resize(masks.values.len(),0);
            seen_contribution[support as usize]=epoch;
        }
        for edge in &mut states[s].edges {
            edge.mask=masks.and(edge.mask,needed[edge.target as usize])?;
            if edge.mask!=0 {
                // Every edge still contributes its own target height and keeps
                // its exact clipped coefficient. Only the repeated OR into
                // this source's support is omitted: X union X equals X.
                heights[s]=heights[s].max(heights[edge.target as usize]+1);
                if dedup_this_row {
                    if edge.mask as usize>=seen_contribution.len() {
                        seen_contribution.resize(masks.values.len(),0);
                    }
                    if seen_contribution[edge.mask as usize]==epoch {
                        duplicate_contributions+=1;
                        continue;
                    }
                    seen_contribution[edge.mask as usize]=epoch;
                }
                if let Some(accumulator)=accumulator.as_mut() {
                    for (dst,&src) in accumulator.iter_mut().zip(&masks.values[edge.mask as usize]) {*dst|=src;}
                } else if support==0 {
                    support=edge.mask;
                } else if support!=edge.mask && support!=1 {
                    if edge.mask==1 {support=1;} else {
                        let mut union=masks.values[support as usize].to_vec();
                        for (dst,&src) in union.iter_mut().zip(&masks.values[edge.mask as usize]) {*dst|=src;}
                        accumulator=Some(union);
                    }
                }
            }
        }
        needed[s]=match accumulator { Some(bits) => masks.intern(bits)?, None => support };
        if !states[s].guarded {states[s].edges.retain(|e|e.mask!=0);}
    }
    profile.push_ms=started.elapsed().as_secs_f64()*1000.0;
    if std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some() {
        eprintln!("[glrmask/profile][min_support_dedup] enabled={dedup_support} duplicates={duplicate_contributions} indexed_masks={} push_ms={:.3}",
            seen_contribution.len(),profile.push_ms);
    }
    if compress {
        let start=Instant::now();
        profile.observation_weights_before=masks.values.len();
        let observed=referenced_only.then(||{
            let mut used=vec![false;masks.values.len()];
            used[0]=true;used[1]=true;
            for (state,&support) in states.iter().zip(&needed) {
                used[state.final_mask as usize]=true;
                used[support as usize]=true;
                for edge in &state.edges{used[edge.mask as usize]=true;}
            }
            used.iter().enumerate().filter_map(|(id,&used)|used.then_some(id as Mask)).collect::<Vec<_>>()
        });
        profile.observation_weights_used=observed.as_ref().map_or(masks.values.len(),Vec::len);
        // Root support is already available from the necessary backward pass.
        // Projecting every weight to that constant set commutes with the
        // Boolean operations and removes no root-prefix acceptance. Reusing
        // it here avoids a separate expensive generic range-weight prepass.
        let compressed=if let Some(observed)=&observed {
            masks.compress_selected_observations(needed[start_state as usize],Some(observed))
        } else {masks.compress_input_observations(needed[start_state as usize])};
        if let Some((mapping,atoms))=compressed {
            for state in &mut states {
                state.final_mask=mapping[state.final_mask as usize];
                for edge in &mut state.edges{edge.mask=mapping[edge.mask as usize];}
                if !state.guarded{state.edges.retain(|e|e.mask!=0);}
            }
            for id in &mut needed{*id=mapping[*id as usize];}
            heights.fill(0);
            for &s in topo.iter().rev(){for edge in &states[s].edges{
                if edge.mask!=0{heights[s]=heights[s].max(heights[edge.target as usize]+1);}
            }}
            profile.atom_count=atoms;
        }
        profile.quotient_ms=start.elapsed().as_secs_f64()*1000.0;
    }
    let started=Instant::now();
    let mut live=vec![false;n];
    let mut stack=vec![start_state as usize];
    while let Some(s)=stack.pop() {
        if live[s] || needed[s]==0 {continue;}
        live[s]=true;
        stack.extend(states[s].edges.iter().filter(|e|e.mask!=0).map(|e|e.target as usize));
    }
    profile.push_ms+=started.elapsed().as_secs_f64()*1000.0;
    let started=Instant::now();
    let height=heights.iter().copied().max().unwrap_or(0);
    let mut buckets=vec![Vec::new();height+1];
    for s in 0..n {if live[s] {buckets[heights[s]].push(s);}}
    let mut mapped=vec![0u32;n];
    let mut groups=vec![Group{domain:0,signature:Signature{final_mask:0,edges:Vec::new()},guarded:false}];
    for mut bucket in buckets {
        bucket.sort_unstable_by_key(|&s|std::cmp::Reverse((masks.popcount(needed[s]),states[s].edges.len(),s)));
        let base=groups.len();
        let mut exact=FxHashMap::<Signature,u32>::default();
        let reuse_rows=crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_MIN_ROW_REUSE");
        let mut row_keys=FxHashMap::<u64,smallvec::SmallVec<[(usize,u32);1]>>::default();
        for s in bucket {
            let signature=if reuse_rows {
                let mut edges=std::mem::take(&mut states[s].edges);
                for edge in &mut edges {edge.target=if edge.mask==0{0}else{mapped[edge.target as usize]};}
                Signature{final_mask:states[s].final_mask,edges}
            }else{Signature {
                final_mask:states[s].final_mask,
                edges:states[s].edges.iter().map(|edge|Edge{label:edge.label,
                    target:if edge.mask==0 {0} else {mapped[edge.target as usize]},mask:edge.mask}).collect(),
            }};
            let fingerprint=if reuse_rows {
                let mut hasher=FxHasher::default();signature.hash(&mut hasher);let h=hasher.finish();
                // All test-mode keys deliberately collide. Full row comparison,
                // not this fingerprint, is the equality proof.
                if cfg!(test){0}else{h}
            }else{0};
            let existing=if reuse_rows {
                row_keys.get(&fingerprint).and_then(|entries| entries.iter().find_map(|&(source,id)| {
                    (states[source].final_mask==signature.final_mask && states[source].edges==signature.edges).then_some(id)
                }))
            }else{exact.get(&signature).copied()};
            if let Some(id)=existing {
                mapped[s]=id;profile.exact_hits+=1;
                if reuse_rows{states[s].edges=signature.edges;}
                continue;
            }
            let mut found=None;
            for id in base..groups.len() {
                profile.compatibility_attempts+=1;
                if profile.compatibility_attempts>MAX_ATTEMPTS || masks.values.len()>500_000 {return None;}
                if compatible(&groups[id],&signature,needed[s],states[s].guarded,&masks) {found=Some(id);break;}
            }
            let id=match found {
                Some(id)=>{merge(&mut groups[id],&signature,needed[s],&mut masks)?;id},
                None=>{let id=groups.len();groups.push(Group{domain:needed[s],signature:signature.clone(),guarded:states[s].guarded});id},
            };
            mapped[s]=id as u32;
            if reuse_rows {
                states[s].edges=signature.edges;
                row_keys.entry(fingerprint).or_default().push((s,id as u32));
            }else{exact.insert(signature,id as u32);}
        }
    }
    profile.merge_ms=started.elapsed().as_secs_f64()*1000.0;
    let started=Instant::now();
    if crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_MIN_POINT_DECODE") {
        let selected=masks.prepare_point_decoder().is_some();
        if std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some(){
            eprintln!("[glrmask/profile][min_point_decoder] selected={selected} rows={} points={} prepare_ms={:.3}",
                masks.decoded_point_rows.as_ref().map_or(0,Vec::len),
                masks.decoded_point_rows.as_ref().map_or(0,|rows|rows.iter().map(|(_,p)|p.len()).sum::<usize>()),
                started.elapsed().as_secs_f64()*1000.0);
        }
    }
    let mut exported=FxHashMap::<Mask,Weight>::default();
    if crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_PARALLEL_WEIGHT_EXPORT")
        && rayon::current_num_threads()>1 && groups.len()>=256 {
        use rayon::prelude::*;
        let mut seen=vec![false;masks.values.len()];let mut ids=Vec::new();
        for group in &groups {
            for id in std::iter::once(group.signature.final_mask).chain(group.signature.edges.iter().map(|e|e.mask)) {
                if !std::mem::replace(&mut seen[id as usize],true){ids.push(id);}
            }
        }
        // Independent exact decoding, once per used mask. Group and edge order
        // stay fixed; publication still uses the ordinary Weight constructor.
        exported=ids.par_iter().map(|&id|(id,masks.export(id))).collect();
    }
    let mut export=|id:Mask|exported.entry(id).or_insert_with(||masks.export(id)).clone();
    let output_states=groups.into_iter().map(|g| {
        let mut s=DWAState::default();
        if g.signature.final_mask!=0 {s.final_weight=Some(export(g.signature.final_mask));}
        for e in g.signature.edges {s.transitions.insert(e.label,(e.target,export(e.mask)));}
        s
    }).collect::<Vec<_>>();
    profile.output_states=output_states.len();
    profile.output_edges=output_states.iter().map(|s|s.transitions.len()).sum();
    profile.weights=masks.values.len();
    profile.reconstruct_ms=started.elapsed().as_secs_f64()*1000.0;
    Some((DWA::from_parts(output_states,mapped[start_state as usize]),profile))
}

/// Direct adapter for the ordinary native normalizer. Re-intern once per
/// distinct input bit vector, not once per expanded generic edge weight.
/// The existing guard-preserving proof/minimization body is shared unchanged.
/// Checked, finite disjoint atom decoding. Pair enumeration is budgeted and
/// performed once; unchecked overlapping images cannot enter this API.
pub struct FiniteAtomDecoder {
    atoms: Vec<Weight>,
}

impl FiniteAtomDecoder {
    pub fn new_checked(atoms: Vec<Weight>) -> Option<Self> {
        if atoms.is_empty() || atoms.len() > 4096 { return None; }
        let mut points = Vec::<(u32, u32)>::new();
        for atom in &atoms {
            if atom.is_full() { return None; }
            for (low, high, tokens) in atom.range_entries() {
                let count = (u128::from(high) - u128::from(low) + 1) * tokens.len() as u128;
                if count > (32_768 - points.len()) as u128 { return None; }
                for row in low..=high { for token in tokens.iter() { points.push((row, token)); } }
            }
        }
        points.sort_unstable();
        if points.windows(2).any(|pair| pair[0] == pair[1]) { return None; }
        Some(Self { atoms })
    }
}

pub fn minimize_native_finite(
    input: &glrmask_parser_dwa::__private::parser_dwa::FiniteBoundaryDwa, default_label: i32,
) -> Option<(DWA, BitMinimizeProfile)> {
    minimize_native_impl(input, default_label, None)
}

/// Minimize in the private bit algebra, then decode the final graph directly
/// to original coordinates. No generic intermediate atom-weight graph exists.
pub fn minimize_native_decoded(
    input: &glrmask_parser_dwa::__private::parser_dwa::FiniteBoundaryDwa, decoder: &FiniteAtomDecoder,
    default_label: i32,
) -> Option<(DWA, BitMinimizeProfile)> {
    minimize_native_impl(input, default_label, Some(decoder))
}

fn minimize_native_impl(
    input: &glrmask_parser_dwa::__private::parser_dwa::FiniteBoundaryDwa, default_label: i32,
    decoder: Option<&FiniteAtomDecoder>,
) -> Option<(DWA, BitMinimizeProfile)> {
    let selected = referenced_observations_enabled();
    let candidate = minimize_native_with_observations(input, default_label, decoder, selected)?;
    #[cfg(feature = "internal-api")]
    if selected && std::env::var_os("GLRMASK_VALIDATE_MIN_REFERENCED_OBSERVATIONS").is_some() {
        let reference = minimize_native_with_observations(input, default_label, decoder, false)
            .expect("reference observation minimizer declined");
        // All explicit labels plus a representative unnamed label cover the
        // exact parser-prefix alphabet, including DEFAULT and zero guards.
        let max_label = input.states.iter().flat_map(|row| row.transitions.iter())
            .filter_map(|&(label, _, _)| (label >= 0 && label != default_label).then_some(label as u32))
            .max().unwrap_or(0);
        let alphabet = max_label.checked_add(2)?.min(default_label as u32);
        let comparison = glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &reference.0, &candidate.0, alphabet, 2_000_000)
            .expect("observation prefix comparison invalid or budget exhausted");
        assert!(comparison.difference.is_none(), "observation mismatch: {:?}", comparison.difference);
        eprintln!("[glrmask/validate][min_referenced_observations] exact=true pairs={} branches={}",
            comparison.product_states, comparison.compared_branches);
    }
    Some(candidate)
}

fn minimize_native_with_observations(
    input: &glrmask_parser_dwa::__private::parser_dwa::FiniteBoundaryDwa, default_label: i32,
    decoder: Option<&FiniteAtomDecoder>, referenced_only: bool,
) -> Option<(DWA, BitMinimizeProfile)> {
    let started = Instant::now();
    let n = input.states.len();
    if n == 0 || n > MAX_STATES || !(1..=64).contains(&input.rows)
        || !(1..=64).contains(&input.token_count) || input.weights.len() > MAX_MASKS { return None; }
    let mut masks = Masks::new(input.rows);
    if let Some(decoder) = decoder {
        if decoder.atoms.len() > input.rows * 64 { return None; }
        masks.decoded_atoms = Some(decoder.atoms.clone());
    }
    let mut ids = Vec::with_capacity(input.weights.len());
    let all = if input.token_count == 64 { u64::MAX } else { (1u64 << input.token_count) - 1 };
    for weight in &input.weights {
        if weight.len() != input.rows || weight.iter().any(|bits| bits & !all != 0) { return None; }
        ids.push(if masks.borrow_rows {masks.intern_slice(weight)?}
            else{masks.intern(weight.to_vec())?});
    }
    if ids.first().copied() != Some(0) { return None; }
    let mut indegree = vec![0usize; n];
    let mut states = Vec::with_capacity(n);
    let mut edge_count = 0usize;
    for state in &input.states {
        let final_mask = *ids.get(state.final_weight as usize)?;
        let mut edges = Vec::with_capacity(state.transitions.len());
        let mut guarded = false;
        let mut previous = None;
        for &(label, target, weight) in &state.transitions {
            if target as usize >= n || previous.is_some_and(|old| old >= label) { return None; }
            previous = Some(label);
            guarded |= label == default_label;
            indegree[target as usize] += 1;
            edges.push(Edge { label, target, mask: *ids.get(weight as usize)? });
        }
        edge_count = edge_count.checked_add(edges.len())?;
        if edge_count > MAX_EDGES { return None; }
        states.push(State { final_mask, edges, guarded });
    }
    minimize_prepared_with_observations(states, masks, indegree, edge_count, 0,
        started.elapsed().as_secs_f64() * 1000.0, true, referenced_only)
}


#[cfg(test)]
mod tests {
    use super::*;
    const DEFAULT:i32=2147483646;
    #[test]
    fn referenced_observations_ignore_unused_predicates_and_preserve_prefix_masks() {
        use glrmask_parser_dwa::__private::parser_dwa::{FiniteBoundaryDwa,SmallBoundaryDwaState};
        let mut weights=vec![vec![0u64;4].into_boxed_slice(),vec![u64::MAX;4].into_boxed_slice()];
        // These historical temporary masks distinguish all128 selected points,
        // but no final, edge, or needed-domain below observes them separately.
        for point in 0..128 {
            let mut value=vec![0u64;4];
            value[point/32]=1u64<<(point%32);
            weights.push(value.into_boxed_slice());
        }
        let a=weights.len()as u32;weights.push(vec![0xffffu64;4].into_boxed_slice());
        let b=weights.len()as u32;weights.push(vec![0xffff_0000u64;4].into_boxed_slice());
        let input=FiniteBoundaryDwa{rows:4,token_count:64,weights,states:vec![
            SmallBoundaryDwaState{final_weight:0,transitions:vec![(0,1,a),(1,2,b),(DEFAULT,3,0)]},
            SmallBoundaryDwaState{final_weight:a,transitions:vec![(1,3,0)]},
            SmallBoundaryDwaState{final_weight:b,transitions:vec![]},
            SmallBoundaryDwaState{final_weight:0,transitions:vec![]},
        ]};
        let(reference,rp)=minimize_native_with_observations(&input,DEFAULT,None,false).unwrap();
        let(candidate,cp)=minimize_native_with_observations(&input,DEFAULT,None,true).unwrap();
        assert_eq!(rp.atom_count,128);
        assert_eq!(cp.atom_count,2);
        assert!(cp.observation_weights_used<cp.observation_weights_before/4);
        let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &reference,&candidate,3,100_000).unwrap();
        assert!(check.difference.is_none(),"{:?}",check.difference);
    }

    #[test]
    fn referenced_observation_native_dags_match_all_finalized_prefixes() {
        use glrmask_parser_dwa::__private::parser_dwa::{FiniteBoundaryDwa,SmallBoundaryDwaState};
        let mut seed=471959u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
        for case in 0..256 {
            let mut weights=vec![vec![0u64;4].into_boxed_slice(),vec![u64::MAX;4].into_boxed_slice()];
            for point in 0..128 {
                let mut value=vec![0u64;4];value[point/32]=1u64<<(point%32);
                weights.push(value.into_boxed_slice());
            }
            let masks=(0..16u64).map(|bits|{
                let id=weights.len()as u32;
                weights.push(vec![bits*0x1111_1111_1111_1111;4].into_boxed_slice());id
            }).collect::<Vec<_>>();
            let n=3+next()%8;
            let mut states=Vec::new();
            for q in 0..n {
                let mut row=SmallBoundaryDwaState{final_weight:masks[next()%16],transitions:vec![]};
                if q+1<n {
                    for label in [0,1,2,DEFAULT] {
                        if next()%3!=0 {
                            row.transitions.push((label,(q+1+next()%(n-q-1))as u32,masks[next()%16]));
                        }
                    }
                }
                states.push(row);
            }
            let input=FiniteBoundaryDwa{rows:4,token_count:64,weights,states};
            let(reference,_)=minimize_native_with_observations(&input,DEFAULT,None,false).unwrap();
            for observed in [false, true] {
                let(candidate,_)=minimize_native_with_observations(&input,DEFAULT,None,observed).unwrap();
                let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                    &reference,&candidate,4,100_000).unwrap();
                assert!(check.difference.is_none(),"case={case} observed={observed} {:?}",check.difference);
            }
        }
    }

    #[test]
    fn resource_decline_does_not_publish_or_mutate_partial_masks() {
        let exact = Weight::from_token_set_for_tsid(7,(0..4096).collect());
        let oversized = Weight::from_token_set_for_tsid(7,(0..4097).collect());
        assert!(fits_finite_point_budget(&exact,4096));
        assert!(!fits_finite_point_budget(&oversized,4096));
        assert!(!fits_finite_point_budget(&Weight::all(),4096));
        assert!(!fits_finite_point_budget(&Weight::from_uniform(0..=u32::MAX,[0].into_iter().collect()),4096));
        let mut states=vec![DWAState::default();2];
        states[0].final_weight=Some(exact.clone());
        states[1].final_weight=Some(Weight::from_token_set_for_tsid(7,[4096].into_iter().collect()));
        let original=DWA::from_parts(states,0);
        assert!(minimize_finite_final_atoms(&original,DEFAULT).is_none());
        assert_eq!(original.states()[0].final_weight.as_ref(),Some(&exact));
        // Exhaustion of a private intern allocation is an explicit decline.
        // Altering the private row contract exercises the real fallible path
        // without creating a deliberately huge allocation in a unit test.
        let mut masks=Masks::new(1);
        assert!(masks.intern(vec![3,0]).is_none());
        assert_eq!(masks.values.len(),2);
        assert_eq!(masks.intern(vec![3]),Some(2));
    }
    #[test]
    fn batched_final_union_matches_one_shot_domain() {
        let mut states=Vec::new();
        for i in 0..200u32 {
            let mut state=DWAState::default();
            state.final_weight=Some(Weight::from_token_set_for_tsid(11,[i].into_iter().collect()));
            states.push(state);
        }
        let original=DWA::from_parts(states,0);
        let (output,_)=minimize_finite_final_atoms(&original,DEFAULT).unwrap();
        let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(&original,&output,4,10000).unwrap();
        assert!(comparison.difference.is_none());
    }
    fn w(bits:u64)->Weight {Weight::from_token_set_for_tsid(0,(0..64u32).filter(|b|bits&(1u64<<b)!=0).collect())}
    fn mask(dwa:&DWA,word:&[i32])->Weight {
        let mut state=dwa.start_state() as usize;let mut path=w(u64::MAX);let mut accepted=Weight::empty();
        for position in 0..=word.len() {
            if let Some(final_weight)=&dwa.states()[state].final_weight {accepted=accepted.union(&path.intersection(final_weight));}
            if position==word.len(){break;}
            let row=&dwa.states()[state];
            let Some((next,weight))=row.transitions.get(&word[position]).or_else(||row.transitions.get(&DEFAULT)) else {break;};
            path=path.intersection(weight);if path.is_empty(){break;}state=*next as usize;
        }
        accepted
    }
    #[test]
    fn generated_finite_dags_preserve_all_prefix_masks() {
        let mut seed=149u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        for case in 0..512 {
            let n=3+next()%9;
            let mut states=vec![DWAState::default();n];
            for s in 0..n {
                let f=next()%8;if f>0{states[s].final_weight=Some(w(f as u64));}
                if s+1==n{continue;}
                for label in [0,1,2,DEFAULT] {if next()%3!=0 {
                    let target=(s+1+next()%(n-s-1)) as u32;
                    let bits=next()%8;states[s].transitions.insert(label,(target,w(bits as u64)));
                }}
            }
            let original=DWA::from_parts(states,0);
            let (reduced,_)=minimize_finite_bits(&original,1,DEFAULT).unwrap();
            let mut words=vec![Vec::new()];
            for _ in 0..=n {
                let mut following=Vec::new();
                for word in &words {
                    assert_eq!(mask(&original,word),mask(&reduced,word),"case={case} word={word:?}");
                    if word.len()<4 {for label in [0,1,2,3] {let mut w=word.clone();w.push(label);following.push(w);}}
                }
                if following.is_empty(){break;}words=following;
            }
        }
    }
    #[test]
    fn explicit_empty_edge_keeps_default_blocked() {
        let mut states=vec![DWAState::default();3];
        states[0].transitions.insert(DEFAULT,(2,w(1)));
        states[0].transitions.insert(0,(1,Weight::empty()));
        states[2].final_weight=Some(w(1));
        let input=DWA::from_parts(states,0);
        let (output,_)=minimize_finite_bits(&input,1,DEFAULT).unwrap();
        assert_eq!(mask(&output,&[0]),Weight::empty());assert_eq!(mask(&output,&[1]),w(1));
        assert!(output.states()[output.start_state() as usize].transitions.contains_key(&0));
    }
    #[test]
    fn unsupported_coordinates_and_cycles_decline() {
        let mut states=vec![DWAState::default()];
        states[0].final_weight=Some(Weight::from_token_set_for_tsid(1,[3].into_iter().collect()));
        assert!(minimize_finite_bits(&DWA::from_parts(states.clone(),0),1,DEFAULT).is_none());
        states[0].final_weight=Some(w(1));states[0].transitions.insert(0,(0,w(1)));
        assert!(minimize_finite_bits(&DWA::from_parts(states,0),1,DEFAULT).is_none());
    }

    #[test]
    fn multirow_dags_match_exhaustive_symbolic_prefix_language() {
        let mut seed=472u64;
        let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);seed>>32};
        for case in 0..128 {
            let rows=[1,2,17,32,64][case%5];
            let n=4+(next()%9) as usize;
            let mut states=vec![DWAState::default();n];
            for s in 0..n {
                let mut random_weight=|| {
                    Weight::from_per_tsid_token_sets((0..rows).filter_map(|row| {
                        let pattern=next();
                        (pattern&3!=0).then(||(row as u32,(0..8u32).filter(|bit|pattern&(1u64<<(bit+4))!=0).map(|b|b*9).collect()))
                    }))
                };
                states[s].final_weight=Some(random_weight());
                for label in [0,1,2,DEFAULT] {
                    if s+1==n {break;}
                    let weight=random_weight();
                    let target=(s+1+(label as usize%3)%(n-s-1)) as u32;
                    // Retain selected explicit zero edges to test guards.
                    states[s].transitions.insert(label,(target,weight));
                }
            }
            let original=DWA::from_parts(states,0);
            let (candidate,_)=minimize_finite_bits(&original,rows,DEFAULT).unwrap();
            let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                &original,&candidate,4,50_000).unwrap();
            assert!(check.difference.is_none(),"case={case} difference={:?}",check.difference);
            let (compressed,_)=minimize_finite_final_atoms(&original,DEFAULT).unwrap();
            let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                &original,&compressed,4,50_000).unwrap();
            assert!(check.difference.is_none(),"atom case={case} difference={:?}",check.difference);
        }
    }

    #[test]
    fn disjoint_domains_merge_without_cross_path_admission() {
        let mut states=vec![DWAState::default();5];
        states[0].transitions.insert(0,(1,w(1)));
        states[0].transitions.insert(1,(2,w(2)));
        states[1].transitions.insert(2,(3,w(1)));
        states[2].transitions.insert(2,(4,w(2)));
        states[3].final_weight=Some(w(1));
        states[4].final_weight=Some(w(2));
        let original=DWA::from_parts(states,0);
        let (candidate,p)=minimize_finite_bits(&original,1,DEFAULT).unwrap();
        assert!(p.output_states<p.input_states);
        assert_eq!(mask(&candidate,&[0,2]),w(1));
        assert_eq!(mask(&candidate,&[1,2]),w(2));
        let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &original,&candidate,4,50_000).unwrap();
        assert!(check.difference.is_none());
    }

    #[test]
    fn finite_point_coordinates_preserve_sparse_ids_and_identity_edges() {
        let mut states=vec![DWAState::default();4];
        let a=Weight::from_token_set_for_tsid(99,[13,500_000].into_iter().collect());
        let b=Weight::from_token_set_for_tsid(1_000_000,[7,1234].into_iter().collect());
        states[0].transitions.insert(0,(1,Weight::all()));
        states[0].transitions.insert(DEFAULT,(2,Weight::all()));
        states[0].transitions.insert(1,(3,Weight::empty()));
        states[1].final_weight=Some(a.clone());states[2].final_weight=Some(b.clone());
        let original=DWA::from_parts(states,0);
        let domain=a.union(&b);
        let (candidate,_)=minimize_finite_points(&original,&domain,DEFAULT).unwrap();
        let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &original,&candidate,4,50_000).unwrap();
        assert!(check.difference.is_none());
    }

    #[test]
    fn final_union_domain_preserves_masks_despite_unreachable_final_points() {
        let mut states=vec![DWAState::default();4];
        states[0].transitions.insert(0,(1,w(1)));
        states[0].transitions.insert(DEFAULT,(2,w(2)));
        states[1].final_weight=Some(w(3));
        states[2].final_weight=Some(w(6));
        states[3].final_weight=Some(Weight::from_token_set_for_tsid(999_999,[1000].into_iter().collect()));
        let original=DWA::from_parts(states,0);
        let (candidate,_)=minimize_finite_final_union(&original,DEFAULT).unwrap();
        let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &original,&candidate,4,50_000).unwrap();
        assert!(check.difference.is_none());
    }

    #[test]
    fn post_normalized_observation_atoms_are_exact_and_selected() {
        let a=Weight::from_per_tsid_token_sets((0..64).map(|row|(row*1000,[7,100,5000].into_iter().collect())));
        let b=Weight::from_per_tsid_token_sets((0..64).map(|row|(row*1000,[8,101,5001].into_iter().collect())));
        let mut states=vec![DWAState::default();5];
        states[0].transitions.insert(0,(1,Weight::all()));
        states[0].transitions.insert(1,(2,a.union(&b)));
        states[0].transitions.insert(DEFAULT,(3,Weight::all()));
        states[1].final_weight=Some(a.clone());
        states[2].transitions.insert(2,(4,b.clone()));
        states[3].final_weight=Some(b.clone());
        states[4].final_weight=Some(a.union(&b));
        let original=DWA::from_parts(states,0);
        let (candidate,profile)=minimize_finite_final_atoms(&original,DEFAULT).unwrap();
        assert_eq!(profile.atom_count,2,"equivalent point columns must collapse");
        let check=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &original,&candidate,4,50_000).unwrap();
        assert!(check.difference.is_none());
    }

    #[test]
    fn indexed_point_import_matches_exact_range_intersection() {
        let domain=Weight::from_per_tsid_token_sets([5,7,100].into_iter().map(|row|
            (row,(0..260u32).step_by(3).chain([10000]).collect())));
        let mut masks=Masks::from_finite_points(&domain).unwrap();
        for start in [0,3,5,6,7,8,99,100,101] {
            for lo in [0,1,61,63,64,65,100,128,191,192,10000] {
                for width in [0,1,7,65,128] {
                    let weight=Weight::from_per_tsid_token_sets((start..=start+3).map(|row|
                        (row,(lo..=lo+width).collect())));
                    let id=masks.import(&weight).unwrap();
                    assert_eq!(masks.export(id),domain.intersection(&weight),"row={start} lo={lo} width={width}");
                }
            }
        }
    }
}

#[cfg(test)]
#[test]
fn finite_point_decoder_equals_union_for_every_generated_selection() {
    let mut seed=703u64;
    let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);seed>>32};
    for count in [1usize,7,64,65,130] {
        let atoms=(0..count).map(|i| {
            // Distinct original tokens keep atoms disjoint, but different
            // atoms overlap in TSID extent and require an actual set union.
            Weight::from_uniform((i%11)as u32..=(i%11+3)as u32,
                [i as u32*3+1,i as u32*3+2].into_iter().collect())
        }).collect::<Vec<_>>();
        let mut reference=Masks::new(count.div_ceil(64));reference.decoded_atoms=Some(atoms.clone());
        let mut candidate=Masks::new(count.div_ceil(64));candidate.decoded_atoms=Some(atoms);
        candidate.prepare_point_decoder().unwrap();
        for _ in 0..256 {
            let bits=(0..count.div_ceil(64)).map(|_|next()).collect::<Vec<_>>();
            let a=reference.intern(bits.clone()).unwrap();let b=candidate.intern(bits).unwrap();
            assert_eq!(reference.export(a),candidate.export(b));
        }
        assert_eq!(reference.export(1),candidate.export(1),"padding does not decode as ALL");
    }
    let mut bad=Masks::new(1);
    bad.decoded_atoms=Some(vec![Weight::all()]);assert!(bad.prepare_point_decoder().is_none());
    let atom=Weight::from_token_set_for_tsid(2,[3].into_iter().collect());
    bad.decoded_atoms=Some(vec![atom.clone(),atom]);assert!(bad.prepare_point_decoder().is_none());
    assert!(bad.decoded_point_rows.is_none());
}

#[cfg(test)]
#[test]
fn decoder_rejects_overlap_and_unbounded_universe() {
    let a = Weight::from_token_set_for_tsid(11, [17, 19].into_iter().collect());
    let b = Weight::from_token_set_for_tsid(11, [19, 23].into_iter().collect());
    assert!(FiniteAtomDecoder::new_checked(vec![a.clone(), b]).is_none());
    assert!(FiniteAtomDecoder::new_checked(vec![a, Weight::all()]).is_none());
    assert!(FiniteAtomDecoder::new_checked(Vec::new()).is_none());
    let a = Weight::from_token_set_for_tsid(2, [17, 19].into_iter().collect());
    let b = Weight::from_token_set_for_tsid(5, [17, 19].into_iter().collect());
    assert!(FiniteAtomDecoder::new_checked(vec![a,b]).is_some());
}

#[cfg(test)]
#[test]
fn native_decoder_retains_bounded_large_output_and_declines_above_cap() {
    let first=Weight::from_token_set_for_tsid(11,(0..16384).collect());
    let second=Weight::from_token_set_for_tsid(29,(0..16384).collect());
    assert!(FiniteAtomDecoder::new_checked(vec![first.clone(),second.clone()]).is_some());
    let extra=Weight::from_token_set_for_tsid(37,[0].into_iter().collect());
    assert!(FiniteAtomDecoder::new_checked(vec![first,second,extra]).is_none());
}


#[cfg(test)]
#[test]
fn large_final_decode_preserves_full_prefix_masks() {
    // More than 256 distinct residual depths exercises the optional parallel
    // unique-mask decoder when that feature is enabled in the test process.
    let mut rows = vec![DWAState::default(); 301];
    for q in 0..300 { rows[q].transitions.insert(0, ((q + 1) as u32, Weight::all())); }
    rows[300].final_weight = Some(Weight::from_per_tsid_token_sets([
        (0, range_set_blaze::RangeSetBlaze::from_iter([7..=7])),
    ]));
    let input = DWA::from_parts(rows, 0);
    let (output, _) = minimize_finite_final_atoms(&input, 2147483646)
        .expect("small one-coordinate depth fixture fits resource bounds");
    assert!(output.num_states() >= 256);
    let comparison = glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
        &input, &output, 2, 2000,
    ).unwrap();
    assert!(comparison.difference.is_none(), "parallel weight export changed prefix masks: {:?}", comparison.difference);
}

#[cfg(test)]
#[test]
fn borrowed_rows_match_owned_ids_and_algebra_with_collisions() {
    let mut seed=7u64;
    let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);seed};
    for rows in [1,9,44,64]{for indexed in [false,true]{for sparse in [false,true]{
        let mut old=Masks::with_storage(rows,indexed,sparse);old.borrow_rows=false;
        let mut new=Masks::with_storage(rows,indexed,sparse);new.borrow_rows=true;
        if indexed {
            for pool in [&mut old,&mut new]{pool.fingerprint_mask=0;pool.indexed.clear();pool.indexed.insert(0,MaskBucket::Many(vec![0,1]));}
        }
        let mut inputs=vec![vec![0;rows],vec![u64::MAX;rows]];
        for _ in 0..64{inputs.push((0..rows).map(|_|next()).collect());}
        for _ in 0..3{for v in &inputs{assert_eq!(old.intern(v.clone()),new.intern_slice(v));}}
        for _ in 0..256 {
            let n=old.values.len();let a=(next()as usize%n)as u32;let b=(next()as usize%n)as u32;
            assert_eq!(old.and(a,b),new.and(a,b));assert_eq!(old.or(a,b),new.or(a,b));
        }
        assert_eq!(old.values,new.values);assert_eq!(old.cardinalities,new.cardinalities);assert_eq!(old.nonzero_words,new.nonzero_words);
        let before=new.values.len();assert!(new.intern_slice(&vec![0;rows+1]).is_none());assert_eq!(before,new.values.len());
    }}}
}

#[cfg(test)]
#[test]
fn repeated_support_rows_preserve_original_prefix_masks() {
    let mut rng=981u64;
    let mut next=|| {rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);rng>>32};
    for case in 0..64 {
        let n=8;
        let mut states=vec![DWAState::default();n];
        let weights=(0..8u32).map(|x|Weight::from_token_set_for_tsid(0,
            (0..8u32).filter(|&bit|(x+1)&(1<<bit)!=0).collect())).collect::<Vec<_>>();
        for source in 0..n {
            states[source].final_weight=Some(weights[(next()%8)as usize].clone());
            if source+1==n {continue;}
            for label in 0..64 {
                let target=source+1+(next()as usize%(n-source-1));
                let weight=&weights[((label/8+case)%8)as usize];
                states[source].transitions.insert(label as i32,(target as u32,weight.clone()));
            }
            if source%2==0 {
                states[source].transitions.insert(2147483646,((n-1)as u32,Weight::empty()));
            }
        }
        let input=DWA::from_parts(states,0);
        let new=minimize_finite_bits(&input,1,2147483646).unwrap().0;
        let cmp=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &input,&new,65,100000).unwrap();
        assert!(cmp.difference.is_none(),"case {case}: {:?}",cmp.difference);
    }
}
