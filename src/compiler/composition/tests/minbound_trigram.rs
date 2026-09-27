use crate::grammar::flat::Symbol;

// Test-only helpers for the minimal-boundary terminal-sequence experiment.
// This file is include!()'d inside constraint_compose::tests.

const MINBOUND_TRI_BITS: u32 = 21;
const MINBOUND_TRI_MASK: u64 = (1u64 << MINBOUND_TRI_BITS) - 1;

fn mb_pair(a: u32, b: u32) -> u64 { ((a as u64) << 32) | b as u64 }
fn mb_pair_a(x: u64) -> u32 { (x >> 32) as u32 }
fn mb_pair_b(x: u64) -> u32 { x as u32 }
fn mb_tri(a: u32, b: u32, c: u32) -> u64 {
    assert!(a as u64 <= MINBOUND_TRI_MASK && b as u64 <= MINBOUND_TRI_MASK && c as u64 <= MINBOUND_TRI_MASK);
    ((a as u64) << (2 * MINBOUND_TRI_BITS)) | ((b as u64) << MINBOUND_TRI_BITS) | c as u64
}
fn mb_tri_parts(x: u64) -> (u32, u32, u32) {
    ((x >> (2 * MINBOUND_TRI_BITS)) as u32,
     ((x >> MINBOUND_TRI_BITS) & MINBOUND_TRI_MASK) as u32,
     (x & MINBOUND_TRI_MASK) as u32)
}

#[derive(Clone, Default)]
struct MbTriSummary {
    nullable: bool,
    complete1: std::collections::HashSet<u32>,
    complete2: std::collections::HashSet<u64>,
    prefix1: std::collections::HashSet<u32>,
    prefix2: std::collections::HashSet<u64>,
    suffix1: std::collections::HashSet<u32>,
    suffix2: std::collections::HashSet<u64>,
}

fn mb_eps() -> MbTriSummary { MbTriSummary { nullable: true, ..Default::default() } }
fn mb_term(t: u32) -> MbTriSummary {
    let mut one = std::collections::HashSet::new(); one.insert(t);
    MbTriSummary {
        nullable: false,
        complete1: one.clone(), complete2: Default::default(),
        prefix1: one.clone(), prefix2: Default::default(),
        suffix1: one, suffix2: Default::default(),
    }
}
fn mb_concat(a: &MbTriSummary, b: &MbTriSummary) -> MbTriSummary {
    let mut o = MbTriSummary::default();
    o.nullable = a.nullable && b.nullable;
    if b.nullable { o.complete1.extend(a.complete1.iter().copied()); o.complete2.extend(a.complete2.iter().copied()); }
    if a.nullable { o.complete1.extend(b.complete1.iter().copied()); o.complete2.extend(b.complete2.iter().copied()); }
    for &x in &a.complete1 { for &y in &b.complete1 { o.complete2.insert(mb_pair(x,y)); } }

    o.prefix1.extend(a.prefix1.iter().copied());
    if a.nullable { o.prefix1.extend(b.prefix1.iter().copied()); }
    o.prefix2.extend(a.prefix2.iter().copied());
    if a.nullable { o.prefix2.extend(b.prefix2.iter().copied()); }
    for &x in &a.complete1 { for &y in &b.prefix1 { o.prefix2.insert(mb_pair(x,y)); } }

    o.suffix1.extend(b.suffix1.iter().copied());
    if b.nullable { o.suffix1.extend(a.suffix1.iter().copied()); }
    o.suffix2.extend(b.suffix2.iter().copied());
    if b.nullable { o.suffix2.extend(a.suffix2.iter().copied()); }
    for &x in &a.suffix1 { for &y in &b.complete1 { o.suffix2.insert(mb_pair(x,y)); } }
    o
}
fn mb_union(dst: &mut MbTriSummary, src: MbTriSummary) -> bool {
    let before=(dst.nullable,dst.complete1.len(),dst.complete2.len(),dst.prefix1.len(),dst.prefix2.len(),dst.suffix1.len(),dst.suffix2.len());
    dst.nullable |= src.nullable;
    dst.complete1.extend(src.complete1); dst.complete2.extend(src.complete2);
    dst.prefix1.extend(src.prefix1); dst.prefix2.extend(src.prefix2);
    dst.suffix1.extend(src.suffix1); dst.suffix2.extend(src.suffix2);
    before != (dst.nullable,dst.complete1.len(),dst.complete2.len(),dst.prefix1.len(),dst.prefix2.len(),dst.suffix1.len(),dst.suffix2.len())
}

fn mb_candidate_trigrams(dwa: &DWA) -> std::collections::HashSet<u64> {
    let started=Instant::now();
    fn live(s:u32,d:&DWA,m:&mut[Option<bool>])->bool{
        if let Some(v)=m[s as usize]{return v}
        let r=&d.states()[s as usize];
        let mut v=r.final_weight.as_ref().is_some_and(|w|!w.is_empty());
        for (_,(t,_)) in &r.transitions { v |= live(*t,d,m); }
        m[s as usize]=Some(v); v
    }
    let mut memo=vec![None;dwa.num_states() as usize];
    let mut coreach=vec![false;dwa.num_states() as usize];
    for s in 0..dwa.num_states(){coreach[s as usize]=live(s,dwa,&mut memo);}
    let mut out=std::collections::HashSet::new();
    for s in 0..dwa.num_states(){
        for (&a,(s1,_)) in &dwa.states()[s as usize].transitions { if a<0{continue}
            for (&b,(s2,_)) in &dwa.states()[*s1 as usize].transitions { if b<0{continue}
                for (&c,(s3,_)) in &dwa.states()[*s2 as usize].transitions {
                    if c>=0 && coreach[*s3 as usize] { out.insert(mb_tri(a as u32,b as u32,c as u32)); }
                }
            }
        }
    }
    eprintln!("MINBOUND trigram_candidates count={} ms={:.3}",out.len(),started.elapsed().as_secs_f64()*1000.0);
    out
}

fn mb_reachable_nt(grammar:&AnalyzedGrammar,start:u32)->Vec<bool>{
    let mut seen=vec![false;grammar.num_nonterminals as usize];
    let mut q=VecDeque::from([start]);
    while let Some(n)=q.pop_front(){
        if n as usize>=seen.len() || std::mem::replace(&mut seen[n as usize],true){continue}
        for &ri in grammar.rules_by_lhs.get(n as usize).into_iter().flatten(){
            for sym in &grammar.rules[ri as usize].rhs { if let Symbol::Nonterminal(c)=sym { if !seen[*c as usize]{q.push_back(*c);} } }
        }
    }
    seen
}
fn mb_sym_summary(sym:&Symbol,sums:&[MbTriSummary],grammar:&AnalyzedGrammar,controls:&BTreeSet<u32>)->MbTriSummary{
    match sym {
        Symbol::Terminal(t) if controls.contains(t)=>mb_eps(),
        Symbol::Terminal(t) if *t<grammar.num_terminals=>mb_term(*t),
        Symbol::Terminal(_)=>MbTriSummary::default(),
        Symbol::Nonterminal(n)=>sums[*n as usize].clone(),
    }
}

fn mb_valid_candidate_trigrams(grammar:&AnalyzedGrammar,controls:&BTreeSet<u32>,candidates:&std::collections::HashSet<u64>)->std::collections::HashSet<u64>{
    let started=Instant::now();
    let start=grammar.rules.first().expect("augmented start").lhs;
    let reachable=mb_reachable_nt(grammar,start);
    let mut sums=vec![MbTriSummary::default();grammar.num_nonterminals as usize];
    let mut iters=0usize;
    loop{
        iters+=1; let mut changed=false;
        for rule in &grammar.rules {
            if !reachable.get(rule.lhs as usize).copied().unwrap_or(false){continue}
            let mut s=mb_eps();
            for sym in &rule.rhs { let x=mb_sym_summary(sym,&sums,grammar,controls); s=mb_concat(&s,&x); }
            changed |= mb_union(&mut sums[rule.lhs as usize],s);
        }
        if !changed{break} assert!(iters<256,"trigram summary did not converge");
    }
    let mut by_ab:std::collections::HashMap<u64,Vec<u32>>=Default::default();
    let mut by_a:std::collections::HashMap<u32,Vec<u64>>=Default::default();
    for &x in candidates { let(a,b,c)=mb_tri_parts(x); by_ab.entry(mb_pair(a,b)).or_default().push(c); by_a.entry(a).or_default().push(mb_pair(b,c)); }
    for v in by_ab.values_mut(){v.sort_unstable();v.dedup();} for v in by_a.values_mut(){v.sort_unstable();v.dedup();}
    let mut valid=std::collections::HashSet::new();
    for rule in &grammar.rules {
        if !reachable.get(rule.lhs as usize).copied().unwrap_or(false){continue}
        let mut left=mb_eps();
        for sym in &rule.rhs {
            let right=mb_sym_summary(sym,&sums,grammar,controls);
            for &ab in &left.suffix2 { if let Some(cs)=by_ab.get(&ab){ for &c in cs { if right.prefix1.contains(&c){valid.insert(mb_tri(mb_pair_a(ab),mb_pair_b(ab),c));} } } }
            for &a in &left.suffix1 { if let Some(bcs)=by_a.get(&a){ for &bc in bcs { if right.prefix2.contains(&bc){valid.insert(mb_tri(a,mb_pair_a(bc),mb_pair_b(bc)));} } } }
            left=mb_concat(&left,&right);
        }
    }
    eprintln!("MINBOUND trigram_cfg iterations={} reachable_nonterminals={} candidates={} valid={} invalid={} prefix2={} suffix2={} ms={:.3}",
        iters,reachable.iter().filter(|&&x|x).count(),candidates.len(),valid.len(),candidates.len().saturating_sub(valid.len()),
        sums.iter().map(|s|s.prefix2.len()).sum::<usize>(),sums.iter().map(|s|s.suffix2.len()).sum::<usize>(),started.elapsed().as_secs_f64()*1000.0);
    valid
}

fn mb_filter_valid_trigrams(residual:&DWA,valid:&std::collections::HashSet<u64>)->DWA{
    const NONE:u32=u32::MAX;
    let started=Instant::now();
    let mut states=vec![DWAState::default()];
    let mut payload=vec![(residual.start_state(),NONE,NONE)];
    let mut ids:std::collections::HashMap<(u32,u32,u32),u32>=Default::default(); ids.insert(payload[0],0);
    let mut q=VecDeque::from([0u32]); let mut rejected=0usize;
    while let Some(o)=q.pop_front(){
        let(s,p2,p1)=payload[o as usize]; let row=&residual.states()[s as usize]; states[o as usize].final_weight=row.final_weight.clone();
        for(&lab,(target,w)) in &row.transitions { assert!(lab>=0 && lab!=DEFAULT_LABEL); let cur=lab as u32;
            if p2!=NONE && !valid.contains(&mb_tri(p2,p1,cur)){rejected+=1;continue}
            let ctx=if p1==NONE{(NONE,cur)}else{(p1,cur)}; let key=(*target,ctx.0,ctx.1);
            let n=if let Some(&n)=ids.get(&key){n}else{let n=states.len() as u32;ids.insert(key,n);states.push(DWAState::default());payload.push(key);q.push_back(n);n};
            states[o as usize].transitions.insert(lab,(n,w.clone()));
        }
    }
    let raw=DWA::from_parts(states,0); let rs=raw.num_states(); let rt=raw.num_transitions();
    let out=crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
    eprintln!("MINBOUND trigram_filter rejected_edges={} raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",rejected,rs,rt,out.num_states(),out.num_transitions(),started.elapsed().as_secs_f64()*1000.0);
    out
}
