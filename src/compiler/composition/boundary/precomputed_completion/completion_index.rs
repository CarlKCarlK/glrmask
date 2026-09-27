//! Component-only completion observations for the FIRST segment of a model token.
//!
//! A certified prefix transformer proves T(xb) = delta_b o T(x), retaining
//! original-state provenance. This index records match(q, x) for every original
//! q and transformer state reached by the component's reusable whitelist. A
//! later query only selects columns and intersects classes with its seed set.
//! It never quotients reset/continuation states or changes parser labels.
use super::prefix_observer::{CertifiedPrefixObserver, PrefixObserver};
use glrmask_lexer::__private::automata::lexer::{Lexer, tokenizer::Tokenizer};
use rustc_hash::{FxHashMap, FxHasher};
use std::{hash::{Hash, Hasher}, sync::Arc};

const MAX_RAW: usize = 500_000;
const MAX_COLUMNS: usize = 32_768;
const MAX_CELLS: usize = 16_000_000;
const MAX_LABELS: usize = 8_000_000;
const MAX_WIRE: usize = 128 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"GCCI0001";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error { ResourceLimit, Malformed, CertificateMismatch, QueryOutsideCoverage }
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Data {
    raw_to_class: Vec<u32>,
    columns: usize,
    signatures: Vec<Vec<u32>>,
    observations: Vec<Vec<u32>>,
}

struct ObservationPool {
    values: Vec<Vec<u32>>,
    ids: FxHashMap<Vec<u32>, u32>,
    joins: FxHashMap<(u32, u32), u32>,
    labels: usize,
}
impl ObservationPool {
    fn new() -> Self {
        let mut ids = FxHashMap::default(); ids.insert(Vec::new(), 0);
        Self { values: vec![Vec::new()], ids, joins: FxHashMap::default(), labels: 0 }
    }
    fn intern(&mut self, labels: Vec<u32>) -> Result<u32> {
        if let Some(&id) = self.ids.get(&labels) { return Ok(id); }
        self.labels = self.labels.checked_add(labels.len()).ok_or(Error::ResourceLimit)?;
        if self.labels > MAX_LABELS || self.values.len() >= 500_000 { return Err(Error::ResourceLimit); }
        let id = self.values.len() as u32;
        self.ids.insert(labels.clone(), id); self.values.push(labels); Ok(id)
    }
    fn join(&mut self, a: u32, b: u32) -> Result<u32> {
        if a == 0 || a == b { return Ok(b); }
        if b == 0 { return Ok(a); }
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&id) = self.joins.get(&key) { return Ok(id); }
        let left = &self.values[a as usize]; let right = &self.values[b as usize];
        let mut out = Vec::with_capacity(left.len() + right.len());
        let (mut i, mut j) = (0, 0);
        while i < left.len() && j < right.len() {
            match left[i].cmp(&right[j]) {
                std::cmp::Ordering::Less => { out.push(left[i]); i += 1; }
                std::cmp::Ordering::Greater => { out.push(right[j]); j += 1; }
                std::cmp::Ordering::Equal => { out.push(left[i]); i += 1; j += 1; }
            }
        }
        out.extend_from_slice(&left[i..]); out.extend_from_slice(&right[j..]);
        let id = self.intern(out)?;
        if self.joins.len() < 262_144 { self.joins.insert(key, id); }
        Ok(id)
    }
}
impl Data {
    fn derive(cert: &CertifiedPrefixObserver<'_>) -> Result<Self> {
        let tok = cert.source(); let observer = cert.observer();
        let n = tok.num_states() as usize; let columns = observer.states.len();
        let cells = n.checked_mul(columns).ok_or(Error::ResourceLimit)?;
        if n == 0 || n > MAX_RAW || columns == 0 || columns > MAX_COLUMNS || cells > MAX_CELLS {
            return Err(Error::ResourceLimit);
        }
        let mut pool = ObservationPool::new();
        let mut raw_outputs = Vec::with_capacity(n);
        for q in 0..n {
            let mut labels = tok.matched_terminals_iter(q as u32).collect::<Vec<_>>();
            labels.sort_unstable(); labels.dedup(); raw_outputs.push(pool.intern(labels)?);
        }
        let mut dense = vec![0u32; cells]; let mut accum = vec![0u32; n];
        let mut touched = Vec::new(); let mut work = 0usize;
        for (column, state) in observer.states.iter().enumerate() {
            for run in &state.runs {
                work = work.checked_add(run.len as usize).ok_or(Error::ResourceLimit)?;
                if work > 16_000_000 { return Err(Error::ResourceLimit); }
                for i in 0..run.len {
                    // T4's finite certificate has already checked indices and arithmetic.
                    let target = (run.target + i * u32::from(run.target_step)) as usize;
                    let origin = (run.origin + i * u32::from(run.origin_step)) as usize;
                    let labels = raw_outputs[target]; if labels == 0 { continue; }
                    if accum[origin] == 0 { touched.push(origin); }
                    accum[origin] = pool.join(accum[origin], labels)?;
                }
            }
            for origin in touched.drain(..) { dense[origin * columns + column] = std::mem::take(&mut accum[origin]); }
        }
        let mut classes = FxHashMap::<Vec<u32>, u32>::default();
        let mut raw_to_class = Vec::with_capacity(n); let mut signatures = Vec::new();
        for row in dense.chunks_exact(columns) {
            let id = match classes.get(row) {
                Some(&id) => id,
                None => { let id = signatures.len() as u32; classes.insert(row.to_vec(), id); signatures.push(row.to_vec()); id }
            };
            raw_to_class.push(id);
        }
        Ok(Self { raw_to_class, columns, signatures, observations: pool.values })
    }
    fn validate(&self) -> Result<()> {
        let n = self.raw_to_class.len(); let k = self.columns; let classes = self.signatures.len();
        if n == 0 || n > MAX_RAW || k == 0 || k > MAX_COLUMNS || classes == 0 || classes > n
            || classes.checked_mul(k).ok_or(Error::ResourceLimit)? > MAX_CELLS
            || self.observations.is_empty() || self.observations.len() > 500_000
            || !self.observations[0].is_empty() { return Err(Error::Malformed); }
        let mut seen = vec![false; classes];
        for &c in &self.raw_to_class { let yes = seen.get_mut(c as usize).ok_or(Error::Malformed)?; *yes = true; }
        if seen.contains(&false) { return Err(Error::Malformed); }
        for row in &self.signatures {
            if row.len() != k || row.iter().any(|&v| v as usize >= self.observations.len()) { return Err(Error::Malformed); }
        }
        let mut labels = 0usize;
        for row in &self.observations {
            labels = labels.checked_add(row.len()).ok_or(Error::ResourceLimit)?;
            if labels > MAX_LABELS || !row.windows(2).all(|w| w[0] < w[1]) { return Err(Error::Malformed); }
        }
        Ok(())
    }
}

/// Decoded bytes are not queryable. Certification must bind both the complete
/// origin transformer and its derived observations to the immutable source.
#[derive(Clone)]
pub struct UntrustedCompletionIndex { prefix: PrefixObserver, data: Data }

/// The source borrow prevents a certificate from outliving or silently switching
/// its lexer. Production may place source and derived data in one immutable Arc
/// context; it must not forge a self-referential lifetime.
pub struct CertifiedCompletionIndex<'a> {
    source: &'a Tokenizer,
    transitions: Vec<Vec<(u8, u32)>>,
    covered_words: Vec<Vec<u8>>,
    data: Data,
    members: Vec<Vec<u32>>,
}

impl UntrustedCompletionIndex {
    pub fn from_certified_prefix(prefix: &CertifiedPrefixObserver<'_>) -> Result<Self> {
        let data = Data::derive(prefix)?;
        Ok(Self { prefix: prefix.observer().clone(), data })
    }
    pub fn certify(self, source: &Tokenizer) -> Result<CertifiedCompletionIndex<'_>> {
        self.data.validate()?;
        let certified = self.prefix.certify(source).ok_or(Error::CertificateMismatch)?;
        // Do not trust serialized class numbers, observation IDs, or a hash.
        let expected = Data::derive(&certified)?;
        if self.data != expected { return Err(Error::CertificateMismatch); }
        let mut members = vec![Vec::new(); self.data.signatures.len()];
        for (q, &c) in self.data.raw_to_class.iter().enumerate() { members[c as usize].push(q as u32); }
        Ok(CertifiedCompletionIndex {
            source,
            transitions: certified.observer().states.iter().map(|s| s.transitions.clone()).collect(),
            covered_words: certified.observer().covered_words.clone(),
            data: self.data, members,
        })
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.data.validate()?;
        let prefix = self.prefix.to_bounded_bytes().ok_or(Error::Malformed)?;
        let mut out = MAGIC.to_vec();
        fn put(out: &mut Vec<u8>, n: usize) -> Result<()> {
            out.extend_from_slice(&u32::try_from(n).map_err(|_| Error::ResourceLimit)?.to_le_bytes()); Ok(())
        }
        put(&mut out, prefix.len())?; put(&mut out, self.data.raw_to_class.len())?;
        put(&mut out, self.data.columns)?; put(&mut out, self.data.signatures.len())?;
        put(&mut out, self.data.observations.len())?;
        out.extend_from_slice(&prefix);
        for &x in &self.data.raw_to_class { put(&mut out, x as usize)?; }
        for row in &self.data.signatures { for &x in row { put(&mut out, x as usize)?; } }
        for row in &self.data.observations { put(&mut out, row.len())?; for &x in row { put(&mut out, x as usize)?; } }
        if out.len() > MAX_WIRE { return Err(Error::ResourceLimit); } Ok(out)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_WIRE { return Err(Error::ResourceLimit); }
        if !bytes.starts_with(MAGIC) { return Err(Error::Malformed); }
        struct Reader<'a> { bytes: &'a [u8], offset: usize }
        impl Reader<'_> {
            fn take(&mut self, count: usize) -> Result<&[u8]> {
                let end = self.offset.checked_add(count).ok_or(Error::ResourceLimit)?;
                let result = self.bytes.get(self.offset..end).ok_or(Error::Malformed)?; self.offset = end; Ok(result)
            }
            fn word(&mut self) -> Result<usize> { Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| Error::Malformed)?) as usize) }
            fn words(&mut self, count: usize) -> Result<Vec<u32>> {
                let source = self.take(count.checked_mul(4).ok_or(Error::ResourceLimit)?)?;
                Ok(source.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().expect("four-byte chunk"))).collect())
            }
        }
        let mut r = Reader { bytes, offset: 8 };
        let prefix_len = r.word()?; let n = r.word()?; let k = r.word()?; let classes = r.word()?; let observations = r.word()?;
        let cells = classes.checked_mul(k).ok_or(Error::ResourceLimit)?;
        if n == 0 || n > MAX_RAW || k == 0 || k > MAX_COLUMNS || classes == 0 || classes > n
            || cells > MAX_CELLS || observations == 0 || observations > 500_000 || prefix_len > 64 * 1024 * 1024 { return Err(Error::ResourceLimit); }
        let minimum = n.checked_add(cells).and_then(|x| x.checked_add(observations)).and_then(|x| x.checked_mul(4))
            .and_then(|x| x.checked_add(prefix_len)).ok_or(Error::ResourceLimit)?;
        if minimum > bytes.len().saturating_sub(r.offset) { return Err(Error::Malformed); }
        let prefix = PrefixObserver::from_bounded_bytes(r.take(prefix_len)?).ok_or(Error::Malformed)?;
        if prefix.raw_states != n || prefix.states.len() != k { return Err(Error::Malformed); }
        let raw_to_class = r.words(n)?;
        let mut signatures = Vec::with_capacity(classes);
        for _ in 0..classes { signatures.push(r.words(k)?); }
        let mut labels_left = MAX_LABELS; let mut output_rows = Vec::with_capacity(observations);
        for _ in 0..observations {
            let count = r.word()?; if count > labels_left { return Err(Error::ResourceLimit); }
            labels_left -= count; output_rows.push(r.words(count)?);
        }
        if r.offset != bytes.len() { return Err(Error::Malformed); }
        let data = Data { raw_to_class, columns: k, signatures, observations: output_rows }; data.validate()?;
        Ok(Self { prefix, data })
    }
}

impl CertifiedCompletionIndex<'_> {
    pub fn raw_states(&self) -> usize { self.data.raw_to_class.len() }
    pub fn class_count(&self) -> usize { self.members.len() }
    pub fn column_count(&self) -> usize { self.data.columns }
    pub fn source(&self) -> &Tokenizer { self.source }
    /// This is an observation equivalence, not a mask admission decision.
    /// Caller must preserve crossing-only/owner-local first-role eligibility
    /// and lift every result to original IDs while leaving resets unchanged.
    pub fn query(&self, seeds: &[bool], words: &[Vec<u8>]) -> Result<Vec<(u32, Vec<u32>)>> {
        query_data(&self.data, &self.members, &self.transitions, &self.covered_words, seeds, words)
    }
    /// Exact positive-byte prefix-completion support, not token admission.
    /// Entry-only nullable matches are excluded. Resets are never traversed.
    pub fn prefix_support(&self, seeds: &[bool], words: &[Vec<u8>]) -> Result<Vec<bool>> {
        prefix_support_data(&self.data,&self.members,&self.transitions,&self.covered_words,seeds,words)
    }


}

/// Immutable owning integration boundary. Keeping a strong Arc to the exact
/// certified physical lexer prevents a caller from mutating that object while
/// these observations are in use. No self-reference or unsafe lifetime exists.
pub struct CompletionContext {
    source: Arc<Tokenizer>,
    transitions: Vec<Vec<(u8, u32)>>,
    covered_words: Vec<Vec<u8>>,
    data: Data,
    members: Vec<Vec<u32>>,
}
impl CompletionContext {
    pub fn load(source: Arc<Tokenizer>, bytes: &[u8]) -> Result<Self> {
        let decoded = UntrustedCompletionIndex::from_bytes(bytes)?;
        let certified = decoded.certify(source.as_ref())?;
        let CertifiedCompletionIndex { transitions, covered_words, data, members, .. } = certified;
        Ok(Self { source, transitions, covered_words, data, members })
    }
    pub fn source(&self) -> &Tokenizer { self.source.as_ref() }
    pub fn raw_states(&self) -> usize { self.data.raw_to_class.len() }
    pub fn class_count(&self) -> usize { self.members.len() }
    pub fn query(&self, seeds: &[bool], words: &[Vec<u8>]) -> Result<Vec<(u32, Vec<u32>)>> {
        query_data(&self.data, &self.members, &self.transitions, &self.covered_words, seeds, words)
    }
    /// Exact positive-byte prefix-completion support, not token admission.
    /// Entry-only nullable matches are excluded. Resets are never traversed.
    pub fn prefix_support(&self, seeds: &[bool], words: &[Vec<u8>]) -> Result<Vec<bool>> {
        prefix_support_data(&self.data,&self.members,&self.transitions,&self.covered_words,seeds,words)
    }

}

fn prefix_support_data(data:&Data,members:&[Vec<u32>],transitions:&[Vec<(u8,u32)>],covered_words:&[Vec<u8>],seeds:&[bool],words:&[Vec<u8>])->Result<Vec<bool>> {
    if seeds.len()!=data.raw_to_class.len(){return Err(Error::Malformed);}
    if words.len()>131_072||words.iter().try_fold(0usize,|n,w|n.checked_add(w.len())).ok_or(Error::ResourceLimit)?>262_144{return Err(Error::ResourceLimit);}
    let mut selected=vec![false;data.columns];
    for word in words {
        if covered_words.binary_search(word).is_err(){return Err(Error::QueryOutsideCoverage);}
        let mut state=0;
        for &byte in word {
            let row=&transitions[state];
            state=row[row.binary_search_by_key(&byte,|&(b,_)|b).map_err(|_|Error::QueryOutsideCoverage)?].1 as usize;
            // Include state0 if a NONEMPTY prefix has this exact transformation.
            selected[state]=true;
        }
    }
    let columns=selected.iter().enumerate().filter_map(|(i,x)|x.then_some(i)).collect::<Vec<_>>();
    let mut result=vec![false;seeds.len()];let mut work=0usize;
    for (class,group)in members.iter().enumerate(){
        if !group.iter().any(|q|seeds[*q as usize]){continue;}
        work=work.checked_add(columns.len()).ok_or(Error::ResourceLimit)?;if work>MAX_CELLS{return Err(Error::ResourceLimit);}
        let observations=&data.signatures[class];
        if columns.iter().any(|c|observations[*c]!=0){for &q in group{result[q as usize]=seeds[q as usize];}}
    }
    Ok(result)
}

fn query_data(data: &Data, members: &[Vec<u32>], transitions: &[Vec<(u8,u32)>], covered_words: &[Vec<u8>],
    seeds: &[bool], words: &[Vec<u8>]) -> Result<Vec<(u32, Vec<u32>)>> {
        if seeds.len() != data.raw_to_class.len() { return Err(Error::Malformed); }
        if words.len() > 131_072 || words.iter().try_fold(0usize, |n, w| n.checked_add(w.len())).ok_or(Error::ResourceLimit)? > 262_144 {
            return Err(Error::ResourceLimit);
        }
        let mut selected = vec![false; data.columns]; selected[0] = true;
        for word in words {
            if covered_words.binary_search(word).is_err() { return Err(Error::QueryOutsideCoverage); }
            let mut state = 0;
            for &byte in word {
                let row = &transitions[state];
                state = row[row.binary_search_by_key(&byte, |&(b, _)| b).map_err(|_| Error::QueryOutsideCoverage)?].1 as usize;
                selected[state] = true;
            }
        }
        let columns = selected.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect::<Vec<_>>();
        let mut groups: Vec<Vec<u32>> = Vec::new();
        let mut source_classes = Vec::<usize>::new();
        let mut buckets = FxHashMap::<u64, Vec<usize>>::default();
        let mut work = 0usize;
        for (class, members) in members.iter().enumerate() {
            if !members.iter().any(|&q| seeds[q as usize]) { continue; }
            work = work.checked_add(columns.len()).ok_or(Error::ResourceLimit)?;
            if work > MAX_CELLS { return Err(Error::ResourceLimit); }
            let row = &data.signatures[class];
            let mut hash = FxHasher::default();
            for &column in &columns { row[column].hash(&mut hash); }
            let fingerprint = hash.finish();
            let mut found = None;
            if let Some(candidates) = buckets.get(&fingerprint) {
                for &candidate in candidates {
                    // Fingerprints only choose a bucket; every equality decision
                    // checks the complete selected observation columns.
                    work = work.checked_add(columns.len()).ok_or(Error::ResourceLimit)?;
                    if work > MAX_CELLS { return Err(Error::ResourceLimit); }
                    let reference = &data.signatures[source_classes[candidate]];
                    if columns.iter().all(|&j| row[j] == reference[j]) { found = Some(candidate); break; }
                }
            }
            let id = found.unwrap_or_else(|| {
                let id = groups.len(); groups.push(Vec::new()); source_classes.push(class);
                buckets.entry(fingerprint).or_default().push(id); id
            });
            groups[id].extend(members.iter().copied().filter(|&q| seeds[q as usize]));
        }
        for row in &mut groups { row.sort_unstable(); }
        groups.sort_unstable_by_key(|row| row[0]);
        Ok(groups.into_iter().map(|row| (row[0], row)).collect())
    }

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::prefix_observer::{Limits, Profile};
    use glrmask_lexer::__private::automata::lexer::{ast::{bytes, choice, plus, Expr}, compile::{build_regex_monolithic, build_regex_partitioned}};
    fn prepare(tok: &Tokenizer, words: &[Vec<u8>]) -> UntrustedCompletionIndex {
        let prefix = PrefixObserver::prepare(tok, words, Limits::default(), &mut Profile::default()).unwrap().certify(tok).unwrap();
        UntrustedCompletionIndex::from_certified_prefix(&prefix).unwrap()
    }
    fn observed(tok: &Tokenizer, q: u32, word: &[u8]) -> Vec<Vec<u32>> {
        let mut states = tok.singleton_epsilon_closure(q).into_vec(); let mut result = Vec::new();
        for byte in std::iter::once(None).chain(word.iter().map(Some)) {
            if let Some(&byte) = byte { states = tok.step_all(&states, byte).into_vec(); }
            let mut labels = states.iter().flat_map(|&q| tok.matched_terminals_iter(q)).collect::<Vec<_>>();
            labels.sort_unstable(); labels.dedup(); result.push(labels);
        } result
    }
    // Independent oracle: execute each selected source over every word,
    // rather than consulting the persisted transformer, columns, or classes.
    fn reference_classes(tok: &Tokenizer, seeds: &[bool], words: &[Vec<u8>]) -> Vec<(u32, Vec<u32>)> {
        let mut groups = std::collections::BTreeMap::<Vec<Vec<Vec<u32>>>, Vec<u32>>::new();
        for (q, &selected) in seeds.iter().enumerate() {
            if !selected { continue; }
            // The initial observation is relevant even for an empty query.
            let mut signature = vec![observed(tok, q as u32, &[])];
            signature.extend(words.iter().map(|word| observed(tok, q as u32, word)));
            groups.entry(signature).or_default().push(q as u32);
        }
        let mut result = groups.into_values().map(|members| (members[0], members)).collect::<Vec<_>>();
        result.sort_unstable_by_key(|group| group.0);
        result
    }
    #[test]
    fn exact_projected_classes_roundtrip_and_match_independent_scanner() {
        let expressions = vec![choice(vec![bytes(b"xab"), bytes(b"yab"), bytes(b"xabc")]), plus(choice(vec![bytes(b"a"), bytes(b"ba")])), Expr::Epsilon];
        let a = build_regex_monolithic(&expressions).into_tokenizer(3, None);
        let b = build_regex_partitioned(&expressions, &[0,1,2]).into_tokenizer(3, None);
        let (c, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&a,0),(&b,3)]);
        let mut random = 682304u64; let mut next = || { random = random.wrapping_mul(6364136223846793005).wrapping_add(1); (random >> 32) as usize };
        let mut removed = 0usize; let mut checks = 0usize;
        for tok in [a, b, c] { for _ in 0..48 {
            let mut words = vec![b"ab".to_vec(), b"abc".to_vec(), b"ba".to_vec(), vec![]];
            for _ in 0..10 { let len=next()%6; words.push((0..len).map(|_|b"abcxy !"[next()%7]).collect()); }
            let p = PrefixObserver::prepare(&tok,&words,Limits::default(),&mut Profile::default()).unwrap().certify(&tok).unwrap();

            let data=UntrustedCompletionIndex::from_certified_prefix(&p).unwrap(); let wire=data.to_bytes().unwrap();
            let index=UntrustedCompletionIndex::from_bytes(&wire).unwrap().certify(&tok).unwrap();
            for _ in 0..4 {
                let seeds=(0..tok.num_states()).map(|_|next()%3!=0).collect::<Vec<_>>();
                let query=words.iter().filter(|_|next()%2==0).cloned().collect::<Vec<_>>();
                let actual=index.query(&seeds,&query).unwrap(); let expected=reference_classes(&tok,&seeds,&query);
                assert_eq!(actual,expected);let mut lifted=vec![false;seeds.len()];
                removed+=seeds.iter().filter(|&&x|x).count()-actual.len();
                for(rep,members)in actual{assert!(seeds[rep as usize]);for q in members{assert!(seeds[q as usize]);assert!(!lifted[q as usize]);lifted[q as usize]=true;for word in &query{assert_eq!(observed(&tok,q,word),observed(&tok,rep,word));}}}
                assert_eq!(lifted,seeds);checks+=1;
            }
        }}assert_eq!(checks,576);assert!(removed>0);println!("CERTIFIED_COMPLETION_CHECKS={checks} removed={removed}");
    }
    #[test]
    fn bounded_wire_and_valid_index_corruption_are_rejected() {
        let tok=build_regex_partitioned(&[bytes(b"xa"),bytes(b"yab"),plus(bytes(b"a"))],&[0,1,2]).into_tokenizer(3,None);
        let words=vec![b"a".to_vec(),b"ab".to_vec(),b"aaa".to_vec()];let data=prepare(&tok,&words);let wire=data.to_bytes().unwrap();
        for end in 0..wire.len(){assert!(UntrustedCompletionIndex::from_bytes(&wire[..end]).is_err(),"truncation{end}");}
        let mut trailing=wire.clone();trailing.push(0);assert!(UntrustedCompletionIndex::from_bytes(&trailing).is_err());
        for field in [8,12,16,20,24]{let mut bad=wire.clone();bad[field..field+4].copy_from_slice(&u32::MAX.to_le_bytes());assert!(UntrustedCompletionIndex::from_bytes(&bad).is_err());}
        let mut changed=data.clone();let value=changed.data.signatures.iter_mut().flatten().find(|v|**v!=0).unwrap();*value=0;
        let changed=UntrustedCompletionIndex::from_bytes(&changed.to_bytes().unwrap()).unwrap();assert!(matches!(changed.certify(&tok),Err(Error::CertificateMismatch)));
        let mut changed=data.clone();let row=changed.data.observations.iter_mut().find(|row|row.len()==1).unwrap();row[0]=(row[0]+1)%3;
        let changed=UntrustedCompletionIndex::from_bytes(&changed.to_bytes().unwrap()).unwrap();assert!(matches!(changed.certify(&tok),Err(Error::CertificateMismatch)));
        let mut changed=data.clone();let edge=changed.prefix.states.iter_mut().flat_map(|s|s.transitions.iter_mut()).find(|(_,q)|*q!=0).unwrap();edge.1=0;
        let changed=UntrustedCompletionIndex::from_bytes(&changed.to_bytes().unwrap()).unwrap();assert!(matches!(changed.certify(&tok),Err(Error::CertificateMismatch)));
        let index=data.certify(&tok).unwrap();assert_eq!(index.query(&vec![true;tok.num_states()as usize],&[b"uncovered".to_vec()]),Err(Error::QueryOutsideCoverage));
        assert_eq!(index.query(&[],&words),Err(Error::Malformed));
    }
    #[test]
    fn owning_context_pins_source_and_rejects_same_shape_different_lexer() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<CompletionContext>();
        let mut source=Arc::new(build_regex_partitioned(&[bytes(b"xa"),bytes(b"yab")], &[0,1]).into_tokenizer(2,None));
        let words=vec![b"a".to_vec(),b"ab".to_vec()];let wire=prepare(source.as_ref(),&words).to_bytes().unwrap();
        let context=CompletionContext::load(Arc::clone(&source),&wire).unwrap();
        assert!(Arc::get_mut(&mut source).is_none(),"immutable source must remain pinned");
        let seeds=vec![true;context.raw_states()];let before=context.query(&seeds,&words).unwrap();
        drop(source);assert_eq!(context.query(&seeds,&words).unwrap(),before);
        let different=Arc::new(build_regex_partitioned(&[bytes(b"xb"),bytes(b"yac")], &[0,1]).into_tokenizer(2,None));
        assert_eq!(different.num_states()as usize,context.raw_states(),"must not reject only by shape");
        assert!(CompletionContext::load(different,&wire).is_err());
    }

}

#[cfg(test)]mod prefix_support_tests{
 use super::*;
 use glrmask_lexer::__private::automata::lexer::{ast::{bytes,choice,plus,Expr},compile::{build_regex_monolithic,build_regex_partitioned}};
 use super::super::prefix_observer::{Limits,Profile};
 fn direct(tok:&Tokenizer,seeds:&[bool],words:&[Vec<u8>])->Vec<bool>{
  seeds.iter().enumerate().map(|(q,keep)|*keep&&words.iter().any(|word|{
    let mut states=tok.singleton_epsilon_closure(q as u32).into_vec();
    for &b in word{states=tok.step_all(&states,b).into_vec();if states.iter().any(|q|tok.matched_terminals_iter(*q).next().is_some()){return true;}}false
  })).collect()
 }
 #[test]fn prefix_support_matches_raw_scanner_for_all_source_roles_and_subsets(){
  let exprs=vec![choice(vec![bytes(b"aab"),bytes(b"bac"),bytes(b"aba")]),plus(choice(vec![bytes(b"a"),bytes(b"bc")])),Expr::Epsilon];
  let mut seed=541u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};
  for tokenizer in [build_regex_monolithic(&exprs).into_tokenizer(3,None),build_regex_partitioned(&exprs,&[0,1,2]).into_tokenizer(3,None)]{
   for _ in 0..64{
    let mut words=vec![vec![],b"a".to_vec(),b"bbb".to_vec()];for _ in 0..16{words.push((0..next()%7).map(|_|b'a'+(next()%4)as u8).collect());}words.sort();words.dedup();
    let observer=PrefixObserver::prepare(&tokenizer,&words,Limits::default(),&mut Profile::default()).unwrap().certify(&tokenizer).unwrap();
    let index=UntrustedCompletionIndex::from_certified_prefix(&observer).unwrap();let wire=index.to_bytes().unwrap();let context=UntrustedCompletionIndex::from_bytes(&wire).unwrap().certify(&tokenizer).unwrap();
    for _ in 0..4{
     let seeds=(0..tokenizer.num_states()).map(|_|next()%3!=0).collect::<Vec<_>>();let query=words.iter().filter(|_|next()%3!=0).cloned().collect::<Vec<_>>();
     assert_eq!(context.prefix_support(&seeds,&query).unwrap(),direct(&tokenizer,&seeds,&query));
    }
   }
  }
 }
 #[test]fn entry_finalizer_alone_does_not_count_as_a_positive_byte_event(){
  let tokenizer=build_regex_monolithic(&[bytes(b"a"),Expr::Epsilon]).into_tokenizer(2,None);let words=vec![vec![],b"a".to_vec(),b"z".to_vec()];
  let observer=PrefixObserver::prepare(&tokenizer,&words,Limits::default(),&mut Profile::default()).unwrap().certify(&tokenizer).unwrap();
  let index=UntrustedCompletionIndex::from_certified_prefix(&observer).unwrap().certify(&tokenizer).unwrap();let seeds=vec![true;tokenizer.num_states()as usize];
  assert!(index.prefix_support(&seeds,&[vec![]]).unwrap().iter().all(|x|!*x));
  assert_eq!(index.prefix_support(&seeds,&[b"z".to_vec()]).unwrap(),direct(&tokenizer,&seeds,&[b"z".to_vec()]));
  assert_eq!(index.prefix_support(&seeds,&[b"unknown".to_vec()]),Err(Error::QueryOutsideCoverage));
  assert_eq!(index.prefix_support(&[true],&words),Err(Error::Malformed));
 }
}

// Read-only queries on the existing certified component index. These return
// projected OBSERVATIONS, never token admission. A caller must separately
// prove that erased terminal labels cannot occur on accepted boundary paths.
#[derive(Debug)]
pub(crate) struct PrefixUnions {
    pub columns: Vec<Vec<u32>>,
    pub word_columns: Vec<Vec<usize>>,
}
fn query_columns(transitions:&[Vec<(u8,u32)>],covered:&[Vec<u8>],words:&[Vec<u8>])
 ->Result<(Vec<Vec<usize>>,Vec<usize>)>{
    if words.len()>131_072||words.iter().try_fold(0usize,|n,w|n.checked_add(w.len())).ok_or(Error::ResourceLimit)?>262_144{return Err(Error::ResourceLimit)}
    let mut selected=vec![false;transitions.len()];selected[0]=true;
    let mut paths=Vec::with_capacity(words.len());
    for word in words{
        if covered.binary_search(word).is_err(){return Err(Error::QueryOutsideCoverage)}
        let mut q=0;let mut path=vec![q];
        for &b in word{let row=&transitions[q];q=row[row.binary_search_by_key(&b,|&(b,_)|b).map_err(|_|Error::QueryOutsideCoverage)?].1 as usize;selected[q]=true;path.push(q);}
        paths.push(path);
    }
    Ok((paths,selected.iter().enumerate().filter_map(|(i,&yes)|yes.then_some(i)).collect()))
}

/// Transpose immutable component observations once. Each group is the exact
/// class support of one nonempty completion observation at a prefix column.
/// This is independent of future seeds, vocabulary subset and composition.
pub(crate) struct FirstUnionIndex{
    context:Arc<CompletionContext>,
    groups:Vec<Vec<(usize,Vec<u64>)>>,
    paths:Vec<Vec<usize>>,
    pub groups_count:usize,
    pub cells:usize,
}
impl FirstUnionIndex{
    pub fn prepare(context:Arc<CompletionContext>)->Result<Self>{
        let words=context.members.len().div_ceil(64);let mut groups=Vec::with_capacity(context.data.columns);let mut cells=0usize;let mut groups_count=0;
        for col in 0..context.data.columns{
            let mut index=FxHashMap::<usize,usize>::default();let mut entries=Vec::<(usize,Vec<u64>)>::new();
            for(class,sig)in context.data.signatures.iter().enumerate(){let observation=sig[col]as usize;if context.data.observations[observation].is_empty(){continue}
                let id=if let Some(&id)=index.get(&observation){id}else{
                    cells=cells.checked_add(words).ok_or(Error::ResourceLimit)?;if cells>4_000_000{return Err(Error::ResourceLimit)}
                    let id=entries.len();index.insert(observation,id);entries.push((observation,vec![0u64;words]));id
                };entries[id].1[class/64]|=1u64<<(class%64);
            }
            groups_count+=entries.len();groups.push(entries);
        }
        let mut paths=Vec::with_capacity(context.covered_words.len());let mut path_cells=0usize;
        for word in &context.covered_words{
            path_cells=path_cells.checked_add(word.len()).and_then(|n|n.checked_add(1)).ok_or(Error::ResourceLimit)?;
            if path_cells>8_000_000{return Err(Error::ResourceLimit)}
            let mut q=0usize;let mut path=vec![q];for &b in word{let row=&context.transitions[q];let i=row.binary_search_by_key(&b,|&(b,_)|b).map_err(|_|Error::CertificateMismatch)?;q=row[i].1 as usize;path.push(q);}paths.push(path);
        }
        Ok(Self{context,groups,paths,groups_count,cells})
    }
    pub fn source(&self)->&Tokenizer{self.context.source()}
    pub fn query(&self,seeds:&[bool],words:&[Vec<u8>])->Result<PrefixUnions>{
        let result=self.query_borrowed(seeds,&words.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        Ok(PrefixUnions{columns:result.columns,word_columns:result.word_columns.into_iter().map(|p|p.to_vec()).collect()})
    }
    pub fn query_borrowed<'s>(&'s self,seeds:&[bool],words:&[&[u8]])->Result<PrefixUnionViews<'s>>{
        let ctx=self.context.as_ref();if seeds.len()!=ctx.data.raw_to_class.len(){return Err(Error::Malformed)}
        if words.len()>131072||words.iter().try_fold(0usize,|n,w|n.checked_add(w.len())).ok_or(Error::ResourceLimit)?>262144{return Err(Error::ResourceLimit)}
        let mut selected=vec![false;ctx.data.columns];selected[0]=true;let mut word_columns=Vec::with_capacity(words.len());
        for &word in words{let index=ctx.covered_words.binary_search_by(|x|x.as_slice().cmp(word)).map_err(|_|Error::QueryOutsideCoverage)?;let path=self.paths[index].as_slice();for &c in path{selected[c]=true}word_columns.push(path);}
        let selected=selected.iter().enumerate().filter_map(|(i,&yes)|yes.then_some(i)).collect::<Vec<_>>();
        let mut active=vec![0u64;ctx.members.len().div_ceil(64)];
        for(raw,&yes)in seeds.iter().enumerate(){if yes{let c=ctx.data.raw_to_class[raw]as usize;active[c/64]|=1u64<<(c%64);}}
        let mut seen=vec![0u32;ctx.source.num_terminals()as usize];let mut columns=vec![Vec::new();ctx.data.columns];
        for(generation,&col)in selected.iter().enumerate(){let epoch=generation as u32+1;
            for (observation,classes)in &self.groups[col]{if !active.iter().zip(classes).any(|(&a,&b)|a&b!=0){continue}
                for &terminal in &ctx.data.observations[*observation]{let old=seen.get_mut(terminal as usize).ok_or(Error::Malformed)?;if *old!=epoch{*old=epoch;columns[col].push(terminal);}}
            }
            columns[col].sort_unstable();
        }
        Ok(PrefixUnionViews{columns,word_columns})
    }
}

pub(crate) struct PrefixUnionViews<'a>{pub columns:Vec<Vec<u32>>,pub word_columns:Vec<&'a[usize]>}
