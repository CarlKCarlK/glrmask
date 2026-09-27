//! Exact component-only prefix transformations with origin correlation.
//!
//! A relation state stores (current raw state, original raw state). Reading
//! byte b composes with the ordinary epsilon-NFA byte relation. No origin is
//! existentially forgotten. Therefore querying any later start subset S is
//! exactly relational image T_x[S], not a superset prepared for another S.
use glrmask_lexer::__private::automata::lexer::{Lexer,tokenizer::Tokenizer};
use rustc_hash::FxHashMap;
use serde::{Serialize,Deserialize};

#[derive(Clone,Copy,Debug)]
pub struct Limits{pub states:usize,pub pairs:usize,pub work:usize,pub edges:usize}
impl Default for Limits{fn default()->Self{Self{states:32768,pairs:8_000_000,work:250_000_000,edges:1_000_000}}}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq,Hash)]
pub struct Run { pub target:u32, pub origin:u32, pub len:u32, pub target_step:u8, pub origin_step:u8 }
impl Run {
    fn pairs(&self)->impl Iterator<Item=(u32,u32)>+'_ {
        (0..self.len).map(|i|(self.target+i*u32::from(self.target_step),self.origin+i*u32::from(self.origin_step)))
    }
}
fn compress(pairs:&[(u32,u32)])->Vec<Run>{
    let mut runs=Vec::new();let mut i=0;
    while i<pairs.len(){
        let (target,origin)=pairs[i];let mut end=i+1;let mut delta=(0,0);
        if let Some(&(t,o))=pairs.get(end){
            if let (Some(dt),Some(do_))=(t.checked_sub(target),o.checked_sub(origin)){
                if dt<=1&&do_<=1&&(dt!=0||do_!=0){
                    delta=(dt,do_);end+=1;
                    while let Some(&(t,o))=pairs.get(end){let k=(end-i)as u32;if t==target+k*dt&&o==origin+k*do_{end+=1;}else{break;}}
                }
            }
        }
        runs.push(Run{target,origin,len:(end-i)as u32,target_step:delta.0 as u8,origin_step:delta.1 as u8});i=end;
    }
    runs
}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
pub struct State{pub runs:Vec<Run>,pub transitions:Vec<(u8,u32)>}
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq,Eq)]
pub struct PrefixObserver{pub raw_states:usize,pub states:Vec<State>,pub covered_words:Vec<Vec<u8>>}
#[derive(Default,Debug)]
pub struct Profile{pub words:usize,pub prefixes:usize,pub queries:usize,pub hits:usize,pub pairs:usize,pub work:usize,pub maximum_pairs:usize,pub encoded_runs:usize}

pub struct CertifiedPrefixObserver<'a>{source:&'a Tokenizer,observer:PrefixObserver}
impl CertifiedPrefixObserver<'_>{
    pub fn query(&self,words:&[Vec<u8>],seeds:&[bool])->Option<Vec<bool>>{
        debug_assert_eq!(self.source.num_states()as usize,self.observer.raw_states);
        self.observer.query(words,seeds)
    }
    pub fn observer(&self)->&PrefixObserver{&self.observer}
    pub fn source(&self)->&Tokenizer{self.source}
}

impl PrefixObserver{
    pub fn validate(&self)->Option<()>{
        let limits=Limits::default();
        if self.raw_states==0||self.raw_states>500_000||self.states.is_empty()||self.states.len()>limits.states
            ||self.covered_words.len()>1_000_000||!self.covered_words.windows(2).all(|p|p[0]<p[1]){return None;}
        if self.covered_words.iter().try_fold(0usize,|n,w|n.checked_add(w.len()))?>50_000_000{return None;}
        let mut run_count=0usize;let mut edge_count=0usize;
        for state in &self.states{
            run_count=run_count.checked_add(state.runs.len())?;edge_count=edge_count.checked_add(state.transitions.len())?;
            if run_count.checked_mul(3)?>limits.pairs||edge_count>limits.edges{return None;}
            if !state.transitions.windows(2).all(|p|p[0].0<p[1].0)||state.transitions.iter().any(|&(_,q)|q as usize>=self.states.len()){return None;}
            let mut previous=None;let mut expanded=0usize;
            for run in &state.runs{
                if run.len==0||run.target_step>1||run.origin_step>1
                    ||(run.len>1&&run.target_step==0&&run.origin_step==0){return None;}
                expanded=expanded.checked_add(run.len as usize)?;if expanded>2_000_000{return None;}
                let last=(run.target.checked_add((run.len-1).checked_mul(u32::from(run.target_step))?)?,
                    run.origin.checked_add((run.len-1).checked_mul(u32::from(run.origin_step))?)?);
                if last.0 as usize>=self.raw_states||last.1 as usize>=self.raw_states
                    ||previous.is_some_and(|p|p>=(run.target,run.origin)){return None;}
                previous=Some(last);
            }
        }
        Some(())
    }

    /// Check every equation T_(xb) = delta_b o T_x and the exact epsilon
    /// identity relation. This certifies ALL later origin subsets, not only
    /// the particular subset used during one benchmark.
    pub fn certify(self,tok:&Tokenizer)->Option<CertifiedPrefixObserver<'_>>{
        self.validate()?;
        if self.raw_states!=tok.num_states()as usize||tok.has_virtual_residual_runtime(){return None;}
        let closures=tok.all_singleton_epsilon_closures();let mut initial=Vec::new();
        for origin in 0..self.raw_states{
            if initial.len().checked_add(closures[origin].len())?>2_000_000{return None;}
            for &target in &closures[origin]{initial.push((target,origin as u32));}
        }
        initial.sort_unstable();initial.dedup();
        if !self.states[0].runs.iter().flat_map(Run::pairs).eq(initial){return None;}
        let mut work=0usize;
        for state in &self.states{
            let source=state.runs.iter().flat_map(Run::pairs).collect::<Vec<_>>();
            for &(byte,target_state)in &state.transitions{
                work=work.checked_add(source.len())?;if work>Limits::default().work{return None;}
                let mut expected=Vec::new();let mut index=0;
                while index<source.len(){
                    let raw=source[index].0;let mut end=index+1;while end<source.len()&&source[end].0==raw{end+=1;}
                    let targets=tok.step_all(&[raw],byte);
                    if expected.len().checked_add(targets.len().checked_mul(end-index)?)?>2_000_000{return None;}
                    for target in targets{for &(_,origin)in &source[index..end]{expected.push((target,origin));}}
                    index=end;
                }
                expected.sort_unstable();expected.dedup();
                if !self.states[target_state as usize].runs.iter().flat_map(Run::pairs).eq(expected){return None;}
            }
        }
        Some(CertifiedPrefixObserver{source:tok,observer:self})
    }

    pub fn from_bounded_bytes(bytes:&[u8])->Option<Self>{
        if bytes.len()>64*1024*1024||!bytes.starts_with(b"GPCI0001"){return None;}
        struct Reader<'a>{bytes:&'a[u8],offset:usize}
        impl Reader<'_>{
            fn take(&mut self,n:usize)->Option<&[u8]>{let end=self.offset.checked_add(n)?;let out=self.bytes.get(self.offset..end)?;self.offset=end;Some(out)}
            fn u32(&mut self)->Option<u32>{Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))}
        }
        let mut reader=Reader{bytes,offset:8};let raw_states=reader.u32()?as usize;let n=reader.u32()?as usize;
        let words=reader.u32()?as usize;let limits=Limits::default();
        if raw_states==0||raw_states>500_000||n==0||n>limits.states||words>1_000_000{return None;}
        if words.checked_mul(4)?.checked_add(n.checked_mul(8)?)?>bytes.len().saturating_sub(reader.offset){return None;}
        let mut word_bytes=0usize;let mut covered_words=Vec::with_capacity(words);
        for _ in 0..words{let len=reader.u32()?as usize;word_bytes=word_bytes.checked_add(len)?;if word_bytes>50_000_000{return None;}covered_words.push(reader.take(len)?.to_vec());}
        let mut states=Vec::with_capacity(n);let mut runs_left=limits.pairs/3;let mut edges_left=limits.edges;
        for _ in 0..n{
            let count=reader.u32()?as usize;if count>runs_left{return None;}runs_left-=count;
            let payload=reader.take(count.checked_mul(14)?)?;
            let runs=payload.chunks_exact(14).map(|b|Run{target:u32::from_le_bytes(b[..4].try_into().expect("exact chunk")),origin:u32::from_le_bytes(b[4..8].try_into().expect("exact chunk")),len:u32::from_le_bytes(b[8..12].try_into().expect("exact chunk")),target_step:b[12],origin_step:b[13]}).collect();
            let count=reader.u32()?as usize;if count>256||count>edges_left{return None;}edges_left-=count;
            let payload=reader.take(count.checked_mul(5)?)?;
            let transitions=payload.chunks_exact(5).map(|b|(b[0],u32::from_le_bytes(b[1..].try_into().expect("exact chunk")))).collect();
            states.push(State{runs,transitions});
        }
        if reader.offset!=bytes.len(){return None;}
        let result=Self{raw_states,states,covered_words};result.validate()?;Some(result)
    }

    pub fn to_bounded_bytes(&self)->Option<Vec<u8>>{
        self.validate()?;let mut out=b"GPCI0001".to_vec();
        let put=|out:&mut Vec<u8>,value:usize|->Option<()>{out.extend_from_slice(&u32::try_from(value).ok()?.to_le_bytes());Some(())};
        put(&mut out,self.raw_states)?;put(&mut out,self.states.len())?;put(&mut out,self.covered_words.len())?;
        for word in &self.covered_words{put(&mut out,word.len())?;out.extend_from_slice(word);}
        for state in &self.states{
            put(&mut out,state.runs.len())?;
            for run in &state.runs{out.extend_from_slice(&run.target.to_le_bytes());out.extend_from_slice(&run.origin.to_le_bytes());out.extend_from_slice(&run.len.to_le_bytes());out.push(run.target_step);out.push(run.origin_step);}
            put(&mut out,state.transitions.len())?;for &(byte,target)in &state.transitions{out.push(byte);out.extend_from_slice(&target.to_le_bytes());}
        }
        if out.len()>64*1024*1024{return None;}Some(out)
    }

    pub fn prepare(tok:&Tokenizer,vocab:&[Vec<u8>],limits:Limits,profile:&mut Profile)->Option<Self>{
        let n=tok.num_states()as usize;
        if n==0||n>500_000||tok.has_virtual_residual_runtime(){return None;}
        let closures=tok.all_singleton_epsilon_closures();
        let mut initial=Vec::new();
        for origin in 0..n{
            if initial.len().checked_add(closures[origin].len())?>limits.pairs{return None;}
            for &target in &closures[origin]{initial.push((target,origin as u32));}
        }
        initial.sort_unstable();initial.dedup();
        profile.pairs=initial.len();profile.maximum_pairs=initial.len();
        let initial=compress(&initial);profile.encoded_runs=initial.len();
        let mut states=vec![State{runs:initial.clone(),transitions:Vec::new()}];
        let mut ids=FxHashMap::default();ids.insert(initial,0u32);
        let mut transitions=FxHashMap::<(u32,u8),u32>::default();
        let mut words=vocab.iter().map(Vec::as_slice).collect::<Vec<_>>();words.sort_unstable();words.dedup();profile.words=words.len();
        let mut previous:&[u8]=&[];let mut path=vec![0u32];
        for word in words{
            let lcp=word.iter().zip(previous).take_while(|(a,b)|a==b).count();path.truncate(lcp+1);
            for &byte in &word[lcp..]{
                profile.prefixes+=1;let current=*path.last()?;
                let next=if let Some(&known)=transitions.get(&(current,byte)){profile.hits+=1;known}else{
                    if transitions.len()>=limits.edges{return None;}
                    profile.queries+=1;
                    let row=states[current as usize].runs.iter().flat_map(Run::pairs).collect::<Vec<_>>();
                    profile.work=profile.work.checked_add(row.len())?;if profile.work>limits.work{return None;}
                    let mut result=Vec::new();let mut index=0;
                    while index<row.len(){
                        let raw=row[index].0;let mut end=index+1;while end<row.len()&&row[end].0==raw{end+=1;}
                        let targets=tok.step_all(&[raw],byte);
                        if result.len().checked_add(targets.len().checked_mul(end-index)?)?>limits.pairs.min(2_000_000){return None;}
                        for target in targets{
                            if target as usize>=n{return None;}
                            for &(_,origin)in &row[index..end]{result.push((target,origin));}
                        }
                        index=end;
                    }
                    result.sort_unstable();result.dedup();
                    let pair_count=result.len();let runs=compress(&result);
                    let next=if let Some(&id)=ids.get(&runs){id}else{
                        if states.len()>=limits.states{return None;}
                        profile.pairs=profile.pairs.checked_add(pair_count)?;
                        profile.encoded_runs=profile.encoded_runs.checked_add(runs.len())?;
                        // Each run has more fields than one pair. Bound the
                        // actual retained representation, not decoded size.
                        if profile.encoded_runs.checked_mul(3)?>limits.pairs{return None;}
                        profile.maximum_pairs=profile.maximum_pairs.max(pair_count);
                        let id=states.len()as u32;ids.insert(runs.clone(),id);
                        states.push(State{runs,transitions:Vec::new()});id
                    };
                    transitions.insert((current,byte),next);states[current as usize].transitions.push((byte,next));next
                };
                path.push(next);
            }
            previous=word;
        }
        for state in &mut states{state.transitions.sort_unstable();}
        let mut covered_words=vocab.to_vec();covered_words.sort_unstable();covered_words.dedup();
        Some(Self{raw_states:n,states,covered_words})
    }
    fn query(&self,words:&[Vec<u8>],seeds:&[bool])->Option<Vec<bool>>{
        if seeds.len()!=self.raw_states||self.states.is_empty(){return None;}
        if words.iter().any(|word|self.covered_words.binary_search(word).is_err()){return None;}
        let mut output=vec![false;self.raw_states];let mut seen=vec![false;self.states.len()];
        let mut counts=vec![0usize;seeds.len()+1];
        for (i,&yes)in seeds.iter().enumerate(){counts[i+1]=counts[i]+usize::from(yes);}
        let mut visit=|id:usize|{
            if std::mem::replace(&mut seen[id],true){return;}
            for run in &self.states[id].runs{
                let start=run.origin as usize;let length=run.len as usize;
                let count=if run.origin_step==0{usize::from(seeds[start])*length}else{counts[start+length]-counts[start]};
                if count==0{continue;}
                if run.target_step==0{output[run.target as usize]=true;}
                else if count==length{output[run.target as usize..run.target as usize+length].fill(true);}
                else{for (target,origin)in run.pairs(){if seeds[origin as usize]{output[target as usize]=true;}}}
            }
        };
        visit(0);
        for word in words{
            let mut state=0usize;
            for &byte in word{
                let row=&self.states[state].transitions;let index=row.binary_search_by_key(&byte,|&(b,_)|b).ok()?;
                state=row[index].1 as usize;visit(state);
            }
        }
        Some(output)
    }
}

#[cfg(test)]
#[test]
fn coordinate_run_codec_is_lossless_on_sparse_nondeterministic_relations(){
    let mut seed=397u64;
    let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as u32};
    for _ in 0..1024{
        let n=next()%1000;let mut pairs=Vec::new();
        for _ in 0..n{pairs.push((next()%128,next()%128));}
        for i in 0..96{pairs.push((i,i+4));pairs.push((7,i));pairs.push((i,121));}
        pairs.sort_unstable();pairs.dedup();
        let restored=compress(&pairs).iter().flat_map(Run::pairs).collect::<Vec<_>>();
        assert_eq!(pairs,restored);
    }
}
