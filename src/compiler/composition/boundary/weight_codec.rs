//! Global source-predicate atoms, including classes outside the output domain.
//!
//! The ordinary normalizer observes supports/target identity, not only final
//! accepted weights. Projecting to final support before normalization can
//! therefore change DEFAULT behavior. This codec preserves every nonempty
//! source-membership signature globally. Unobservable atoms decode to empty
//! only AFTER normalization. A symbolic row/token interval sweep avoids a
//! Cartesian enumeration of tokenizer-state and token-ID coordinates.

use std::{collections::BTreeMap, sync::Arc};
use rustc_hash::{FxHashMap,FxHashSet};
use crate::ds::weight::{Weight,SharedTokenSet};

const MAX_SOURCES:usize=2048;
const MAX_POINTS:usize=32_768;
const MAX_ATOMS:usize=4096;
const MAX_ROW_EVENTS:usize=1_000_000;
const MAX_TOKEN_EVENTS:usize=16_000_000;
const MAX_ROW_KEY_PAIRS:usize=2_000_000;

#[derive(Default,Debug)]
pub struct FaithfulQuotientStats {
    pub row_events:usize,
    pub unique_row_signatures:usize,
    pub token_events:usize,
    pub observed_atoms:usize,
    pub ghost_atoms:usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn membership(weight:&Weight,row:u32,token:u32)->bool {
        weight.is_full()||weight.token_set_for_tsid_ref(row).is_some_and(|set|set.contains(token))
    }

    #[test]
    fn symbolic_partition_matches_all_small_grid_membership_classes() {
        let mut random=51143u64;
        let mut next=||{random=random.wrapping_mul(6364136223846793005).wrapping_add(1);(random>>32) as usize};
        for case in 0..256 {
            let mut sources=vec![Weight::empty(),Weight::all()];
            for _ in 0..2+next()%8 {
                sources.push(Weight::from_per_tsid_token_sets((0..6u32).map(|row|
                    (row,(0..10u32).filter(|_|next()%4==0).collect()))));
            }
            let domain=Weight::from_per_tsid_token_sets((0..3).map(|row|(row,[0,1,2,3].into_iter().collect())));
            let quotient=FaithfulWeightQuotient::new(&domain,&sources).unwrap();
            let mut expected=FxHashSet::default();
            for row in 0..8 {for token in 0..12 {
                expected.insert(sources.iter().map(|source|membership(source,row,token)).collect::<Vec<_>>());
            }}
            let encoded=sources.iter().map(|source|quotient.encode(source).unwrap()).collect::<Vec<_>>();
            let actual=(0..quotient.atom_count()).map(|atom|encoded.iter().map(|weight|
                membership(weight,(atom/64) as u32,(atom%64) as u32)).collect::<Vec<_>>()).collect::<FxHashSet<_>>();
            assert_eq!(expected,actual,"case={case}");
            for (source,encoded) in sources.iter().zip(&encoded) {
                assert_eq!(quotient.decode(encoded),source.intersection(&domain),"case={case}");
            }
            for (i,left) in encoded.iter().enumerate(){for (j,right) in encoded.iter().enumerate(){
                assert_eq!(quotient.decode(&left.intersection(right)),sources[i].intersection(&sources[j]).intersection(&domain));
                assert_eq!(quotient.decode(&left.union(right)),sources[i].union(&sources[j]).intersection(&domain));
            }}
        }
    }

    #[test]
    fn symbolic_ranges_handle_large_sparse_ids_and_endpoint_overflow() {
        let x=u32::MAX-3;
        let sources=vec![
            Weight::all(), Weight::empty(),
            Weight::from_uniform(4..=x,[2..=7, x-2..=x].into_iter().collect()),
            Weight::from_uniform(9..=x-1,[5..=11].into_iter().collect()),
            Weight::from_uniform(x..=x,[x..=x].into_iter().collect()),
        ];
        let domain=Weight::from_per_tsid_token_sets([(4,[2,5].into_iter().collect()),(x,[x].into_iter().collect())]);
        let quotient=FaithfulWeightQuotient::new(&domain,&sources).unwrap();
        let mut rows=vec![0,1,u32::MAX];let mut tokens=rows.clone();
        for source in &sources {if !source.is_full(){for (lo,hi,set) in source.range_entries(){
            rows.extend([lo,hi,lo.saturating_sub(1),hi.saturating_add(1)]);
            for range in set.ranges(){tokens.extend([*range.start(),*range.end(),range.start().saturating_sub(1),range.end().saturating_add(1)]);}
        }}}
        rows.sort();rows.dedup();tokens.sort();tokens.dedup();
        let expected=rows.iter().flat_map(|&row|tokens.iter().map(move|&token|(row,token)))
            .map(|(row,token)|sources.iter().map(|source|membership(source,row,token)).collect::<Vec<_>>()).collect::<FxHashSet<_>>();
        let encoded=sources.iter().map(|source|quotient.encode(source).unwrap()).collect::<Vec<_>>();
        let actual=(0..quotient.atom_count()).map(|atom|encoded.iter().map(|weight|
            membership(weight,(atom/64) as u32,(atom%64) as u32)).collect::<Vec<_>>()).collect::<FxHashSet<_>>();
        assert_eq!(expected,actual);
        for (source,encoded) in sources.iter().zip(&encoded){assert_eq!(quotient.decode(encoded),source.intersection(&domain));}
    }

    #[test]
    fn outside_final_domain_classes_are_not_erased() {
        let a=Weight::from_token_set_for_tsid(0,[1].into_iter().collect());
        let b=Weight::from_token_set_for_tsid(0,[2].into_iter().collect());
        let q=FaithfulWeightQuotient::new(&a,&[a.clone(),b.clone(),Weight::all()]).unwrap();
        let encoded=q.encode(&b).unwrap();
        assert!(!encoded.is_empty(),"ghost support must survive compilation");
        assert!(q.decode(&encoded).is_empty(),"ghost may not publish a token mask");
        assert!(q.stats.ghost_atoms>=2);
    }
}

pub struct FaithfulWeightQuotient {
    domain:Weight,
    atoms:Vec<Weight>,
    source_encodings:FxHashMap<usize,(Weight,Weight)>,
    points:usize,
    pub stats:FaithfulQuotientStats,
}

struct AtomSignatures {
    ids:FxHashMap<Vec<u64>,usize>,
    signatures:Vec<Vec<u64>>,
}
impl AtomSignatures {
    fn intern(&mut self,signature:&[u64])->Option<usize> {
        if let Some(&id)=self.ids.get(signature){return Some(id);}
        if self.signatures.len()>=MAX_ATOMS{return None;}
        let id=self.signatures.len();
        self.signatures.push(signature.to_vec());self.ids.insert(signature.to_vec(),id);Some(id)
    }
}

impl FaithfulWeightQuotient {
    pub fn new(domain:&Weight,sources:&[Weight])->Option<Self> {
        if domain.is_empty()||domain.is_full(){return None;}
        let mut unique=Vec::new();let mut seen=FxHashSet::default();
        for source in sources {
            if seen.insert(source.ptr_key()){unique.push(source.clone());}
        }
        if unique.len()>MAX_SOURCES{return None;}
        let finite=unique.iter().filter(|w|!w.is_empty()&&!w.is_full()).collect::<Vec<_>>();
        let words=finite.len().div_ceil(64);
        let mut points=Vec::new();
        for (lo,hi,tokens) in domain.range_entries() {
            let count=(u128::from(hi)-u128::from(lo)+1)*tokens.len() as u128;
            if count>(MAX_POINTS-points.len()) as u128{return None;}
            for row in lo..=hi{for token in tokens.iter(){points.push((row,token));}}
        }
        if points.len().checked_mul(finite.len())?>32_000_000{return None;}
        // Classify only output-observable points explicitly; all other global
        // membership classes are found symbolically below and remain ghosts.
        let mut point_signatures=vec![vec![0u64;words];points.len()];
        for (column,weight) in finite.iter().enumerate() {
            let mut ranges=weight.range_entries().peekable();
            for (point,&(row,token)) in points.iter().enumerate() {
                while ranges.peek().is_some_and(|(_,hi,_)|*hi<row){ranges.next();}
                if ranges.peek().is_some_and(|(lo,hi,tokens)|*lo<=row&&row<=*hi&&tokens.contains(token)) {
                    point_signatures[point][column/64]|=1u64<<(column%64);
                }
            }
        }
        let mut atoms=AtomSignatures{ids:FxHashMap::default(),signatures:Vec::new()};
        let mut point_atoms=Vec::with_capacity(points.len());
        for signature in point_signatures {point_atoms.push(atoms.intern(&signature)?);}
        let observed_atoms=atoms.signatures.len();
        // Preserve the region outside all finite source predicates. ALL is
        // the algebra's symbolic identity, not a finite rectangle, and maps
        // to the full native universe including this zero-signature class.
        atoms.intern(&vec![0u64;words])?;

        let mut events=Vec::<(u64,usize,Option<&SharedTokenSet>)>::new();
        for (column,weight) in finite.iter().enumerate() {
            for (lo,hi,tokens) in weight.range_entries() {
                if events.len()+2>MAX_ROW_EVENTS{return None;}
                events.push((u64::from(lo),column,Some(tokens)));
                events.push((u64::from(hi)+1,column,None));
            }
        }
        events.sort_unstable_by_key(|event|(event.0,event.1,event.2.is_some()));
        let mut stats=FaithfulQuotientStats{row_events:events.len(),observed_atoms,..Default::default()};
        let mut active=vec![None::<&SharedTokenSet>;finite.len()];
        let mut seen_rows=FxHashSet::<Vec<(usize,usize)>>::default();
        let mut key_pairs=0usize;
        let mut i=0usize;
        while i<events.len() {
            let row=events[i].0;
            while i<events.len()&&events[i].0==row {
                let (_,column,set)=events[i];active[column]=set;i+=1;
            }
            if row>u64::from(u32::MAX){break;}
            let key=active.iter().enumerate().filter_map(|(column,set)|set.map(|set|
                (column,Arc::as_ptr(set) as usize))).collect::<Vec<_>>();
            if key.is_empty()||seen_rows.contains(&key){continue;}
            key_pairs=key_pairs.checked_add(key.len())?;
            if key_pairs>MAX_ROW_KEY_PAIRS{return None;}
            seen_rows.insert(key);
            let mut token_events=Vec::<(u64,usize,bool)>::new();
            for (column,set) in active.iter().enumerate() {
                if let Some(set)=set {
                    for range in set.ranges() {
                        token_events.push((u64::from(*range.start()),column,true));
                        token_events.push((u64::from(*range.end())+1,column,false));
                    }
                }
            }
            stats.token_events=stats.token_events.checked_add(token_events.len())?;
            if stats.token_events>MAX_TOKEN_EVENTS{return None;}
            token_events.sort_unstable_by_key(|event|(event.0,event.1,event.2));
            let mut signature=vec![0u64;words];let mut j=0usize;
            while j<token_events.len() {
                let token=token_events[j].0;
                while j<token_events.len()&&token_events[j].0==token {
                    let (_,column,present)=token_events[j];
                    if present{signature[column/64]|=1u64<<(column%64);}else{signature[column/64]&=!(1u64<<(column%64));}
                    j+=1;
                }
                if token<=u64::from(u32::MAX){atoms.intern(&signature)?;}
            }
        }
        stats.unique_row_signatures=seen_rows.len();
        let mut images=vec![BTreeMap::<u32,Vec<u32>>::new();atoms.signatures.len()];
        for (&(row,token),&atom) in points.iter().zip(&point_atoms){images[atom].entry(row).or_default().push(token);}
        let images=images.into_iter().map(|rows|Weight::from_per_tsid_token_sets(
            rows.into_iter().map(|(row,tokens)|(row,tokens.into_iter().collect())))).collect::<Vec<_>>();
        stats.ghost_atoms=images.iter().filter(|image|image.is_empty()).count();
        let mut encoded=FxHashMap::default();
        for (column,source) in finite.into_iter().enumerate() {
            let mut rows=BTreeMap::<u32,Vec<u32>>::new();
            for (atom,signature) in atoms.signatures.iter().enumerate() {
                if signature[column/64]&(1u64<<(column%64))!=0 {
                    rows.entry((atom/64) as u32).or_default().push((atom%64) as u32);
                }
            }
            let result=Weight::from_per_tsid_token_sets(rows.into_iter().map(|(row,tokens)|(row,tokens.into_iter().collect())));
            encoded.insert(source.ptr_key(),(source.clone(),result));
        }
        for source in unique {
            if source.is_empty()||source.is_full(){encoded.insert(source.ptr_key(),(source.clone(),source));}
        }
        Some(Self{domain:domain.clone(),atoms:images,source_encodings:encoded,points:points.len(),stats})
    }
    pub fn atoms(&self)->&[Weight]{&self.atoms}
    pub fn atom_count(&self)->usize{self.atoms.len()}
    pub fn point_count(&self)->usize{self.points}
    pub fn rows(&self)->usize{self.atoms.len().div_ceil(64)}
    pub fn source_count(&self)->usize{self.source_encodings.len()}
    pub fn encode(&self,source:&Weight)->Option<Weight>{self.source_encodings.get(&source.ptr_key()).map(|(_,w)|w.clone())}
    pub fn decode(&self,encoded:&Weight)->Weight {
        if encoded.is_empty(){return Weight::empty();}
        if encoded.is_full(){return self.domain.clone();}
        let mut selected=Vec::new();
        for (lo,hi,tokens) in encoded.range_entries() {
            if lo as usize>=self.rows(){continue;}
            for row in lo..=hi.min(self.rows() as u32-1){for token in tokens.iter().take_while(|token|*token<64){
                if let Some(atom)=self.atoms.get(row as usize*64+token as usize){selected.push(atom);}
            }}
        }
        Weight::union_all(selected)
    }
}
