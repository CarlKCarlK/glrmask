//! Exact ignore-incident FOLLOW of the ordinary component-padded grammar.
//! Reuses original FIRST/FOLLOW/nullable, solving only added ignore columns.
use std::collections::{BTreeMap,VecDeque};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::grammar::flat::Symbol;
use crate::ds::bitset::BitSet;
const MAX_WORDS:usize=4*1024*1024;
const MAX_GRAPH_EDGES:usize=4*1024*1024;
struct Delta { first:Vec<u64>,follow:Vec<u64>,width:usize,ids:Vec<u32>,owner:Vec<usize>,local:Vec<u64> }
fn propagate(rows:&mut[u64],width:usize,dependents:&[Vec<usize>]){
 let n=dependents.len();let mut queued=vec![false;n];let mut queue=VecDeque::new();
 for v in 0..n{if rows[v*width..(v+1)*width].iter().any(|x|*x!=0){queue.push_back(v);queued[v]=true;}}
 while let Some(v)=queue.pop_front(){queued[v]=false;for &to in &dependents[v]{let mut changed=false;for w in 0..width{let old=rows[to*width+w];rows[to*width+w]|=rows[v*width+w];changed|=rows[to*width+w]!=old;}if changed&&!queued[to]{queued[to]=true;queue.push_back(to);}}}
}
fn deltas(g:&AnalyzedGrammar,counts:&[usize],ignores:&[Vec<u32>])->Option<Delta>{
 let n=g.num_nonterminals as usize;let nt=g.num_terminals as usize;
 if counts.len()!=ignores.len()||counts.iter().try_fold(0usize,|a,b|a.checked_add(*b))?!=n||g.rules.is_empty()
  ||n>1_000_000||nt>1_000_000||g.first.len()<n||g.follow.len()<n
  ||ignores.iter().flatten().any(|t|*t>=g.num_terminals){return None;}
 let mut ids=ignores.iter().flatten().copied().collect::<Vec<_>>();ids.sort_unstable();ids.dedup();let w=ids.len().div_ceil(64);
 if n.checked_mul(w)?>MAX_WORDS||nt.checked_mul(nt.div_ceil(64))?>MAX_WORDS||counts.len().checked_mul(nt.div_ceil(64))?>MAX_WORDS{return None;}
 let mut owner=Vec::with_capacity(n);for (c,count)in counts.iter().enumerate(){owner.extend(std::iter::repeat_n(c,*count));}
 let mut local=vec![0u64;counts.len().checked_mul(w)?];for (c,ii)in ignores.iter().enumerate(){for i in ii{let j=ids.binary_search(i).unwrap();local[c*w+j/64]|=1u64<<(j%64);}}
 let mut first=vec![0u64;n*w];let mut deps=vec![Vec::new();n];let mut edgecount=0usize;
 for r in &g.rules{
  let a=r.lhs as usize;let c=*owner.get(a)?;
  for x in 0..w{first[a*w+x]|=local[c*w+x];}
  for s in &r.rhs{match s{
   Symbol::Terminal(_)=>break,
   Symbol::Nonterminal(b)=>{let b=*b as usize;if b>=n{return None;}edgecount+=1;if edgecount>MAX_GRAPH_EDGES{return None;}deps[b].push(a);if !g.nullable.contains(&(b as u32)){break;}}
  }}
 }
 propagate(&mut first,w,&deps);
 let mut follow=vec![0u64;n*w];for row in &mut deps{row.clear();}edgecount=0;
 let mut tail=vec![0u64;w];
 for r in &g.rules{let a=r.lhs as usize;let c=owner[a];tail.copy_from_slice(&local[c*w..(c+1)*w]);let mut nullable=true;
  for s in r.rhs.iter().rev(){match s{
   Symbol::Terminal(_)=>{tail.copy_from_slice(&local[c*w..(c+1)*w]);nullable=false;}
   Symbol::Nonterminal(b)=>{let b=*b as usize;if b>=n{return None;}
    for x in 0..w{follow[b*w+x]|=tail[x];}
    if nullable{edgecount+=1;if edgecount>MAX_GRAPH_EDGES{return None;}deps[a].push(b);}
    let bn=g.nullable.contains(&(b as u32));if !bn{tail.fill(0);}
    for x in 0..w{tail[x]|=first[b*w+x]|local[c*w+x];}nullable&=bn;
   }
  }}
 }
 propagate(&mut follow,w,&deps);
 Some(Delta{first,follow,width:w,ids,owner,local})
}
fn add_small(row:&mut[u64],small:&[u64],ids:&[u32]){
 for (block,&mask)in small.iter().enumerate(){let mut bits=mask;while bits!=0{let j=block*64+bits.trailing_zeros()as usize;let t=ids[j]as usize;row[t/64]|=1u64<<(t%64);bits&=bits-1;}}
}
pub fn scoped_ignore_follow_relation(g:&AnalyzedGrammar,counts:&[usize],ignores:&[Vec<u32>])->Option<BTreeMap<u32,BitSet>>{
 let d=deltas(g,counts,ignores)?;let nt=g.num_terminals as usize;let words=nt.div_ceil(64);let w=d.width;
 #[cfg(test)] verify_delta_reference(g,counts,ignores,&d);
 if d.ids.is_empty(){return Some(BTreeMap::new());}
 let mut ignored=BitSet::new(nt);for t in &d.ids{ignored.set(*t as usize);}
 let mut blocked=(0..nt).map(|t|if ignored.get(t){BitSet::all(nt)}else{ignored.clone()}).collect::<Vec<_>>();
 let mut stars=vec![0u64;counts.len()*words];let mut local=vec![0u64;counts.len()*words];
 for c in 0..counts.len(){add_small(&mut local[c*words..(c+1)*words],&d.local[c*w..(c+1)*w],&d.ids);stars[c*words..(c+1)*words].copy_from_slice(&local[c*words..(c+1)*words]);}
 let mut suffix=vec![0u64;words];
 for r in &g.rules{let a=r.lhs as usize;let c=d.owner[a];let ignored_here=&local[c*words..(c+1)*words];let star=&mut stars[c*words..(c+1)*words];
  suffix.fill(0);for (x,y)in suffix.iter_mut().zip(g.follow[a].words()){*x=*y;}add_small(&mut suffix,&d.follow[a*w..(a+1)*w],&d.ids);
  for x in 0..words{suffix[x]|=ignored_here[x];star[x]|=suffix[x];}
  for s in r.rhs.iter().rev(){match s{
   Symbol::Terminal(t)=>{if let Some(row)=blocked.get_mut(*t as usize){for (x,y)in row.words_mut().iter_mut().zip(&suffix){*x&=!*y;}}
    suffix.copy_from_slice(ignored_here);if (*t as usize)<nt{suffix[*t as usize/64]|=1u64<<(*t%64);}
   }
   Symbol::Nonterminal(b)=>{let b=*b as usize;if !g.nullable.contains(&(b as u32)){suffix.fill(0);}
    for (x,y)in suffix.iter_mut().zip(g.first[b].words()){*x|=*y;}add_small(&mut suffix,&d.first[b*w..(b+1)*w],&d.ids);
    for x in 0..words{suffix[x]|=ignored_here[x];}
   }
  }for x in 0..words{star[x]|=suffix[x];}}
 }
 for (c,ii)in ignores.iter().enumerate(){for &t in ii{for (x,y)in blocked[t as usize].words_mut().iter_mut().zip(&stars[c*words..(c+1)*words]){*x&=!*y;}}}
 Some(blocked.into_iter().enumerate().filter_map(|(t,b)|(!b.is_zero()).then_some((t as u32,b))).collect())
}

#[cfg(test)]
use crate::grammar::flat::Rule;
#[cfg(test)]
fn reference_analyzed(
    grammar: &AnalyzedGrammar,
    component_nonterminals: &[usize],
    ignores: &[Vec<u32>],
) -> Option<AnalyzedGrammar> {
    if component_nonterminals.len() != ignores.len()
        || component_nonterminals.iter().try_fold(0usize, |a,b| a.checked_add(*b))?
            != grammar.num_nonterminals as usize
        || grammar.rules.is_empty()
    { return None; }
    if ignores.iter().flatten().any(|&t| t >= grammar.num_terminals) { return None; }
    let mut owner = Vec::with_capacity(grammar.num_nonterminals as usize);
    for (component,&count) in component_nonterminals.iter().enumerate() {
        owner.extend(std::iter::repeat_n(component,count));
    }
    let mut names = grammar.nonterminal_display_names.clone();
    let mut stars = Vec::with_capacity(ignores.len());
    for (component,labels) in ignores.iter().enumerate() {
        if labels.is_empty() { stars.push(None); }
        else {
            let id = u32::try_from(names.len()).ok()?;
            names.push(format!("<boundary-ignore-star:{component}>")); stars.push(Some(id));
        }
    }
    let mut rules = Vec::with_capacity(grammar.rules.len()+ignores.iter().map(Vec::len).sum::<usize>()+ignores.len());
    for rule in &grammar.rules {
        let component = *owner.get(rule.lhs as usize)?;
        let Some(star) = stars[component] else { rules.push(rule.clone()); continue; };
        let mut rhs = Vec::with_capacity(rule.rhs.len()*2+1);
        rhs.push(Symbol::Nonterminal(star));
        for symbol in &rule.rhs { rhs.push(symbol.clone()); rhs.push(Symbol::Nonterminal(star)); }
        rules.push(Rule { lhs:rule.lhs, rhs });
    }
    for (component,&star) in stars.iter().enumerate() {
        let Some(star) = star else { continue; };
        rules.push(Rule { lhs:star, rhs:Vec::new() });
        for &ignore in &ignores[component] {
            rules.push(Rule { lhs:star, rhs:vec![Symbol::Terminal(ignore),Symbol::Nonterminal(star)] });
        }
    }
    let analyzed = AnalyzedGrammar::from_composed_rules(rules,grammar.num_terminals,
        grammar.terminal_display_names.clone(),names,grammar.rules[0].lhs);
    Some(analyzed)
}

#[cfg(test)]
fn verify_delta_reference(g:&AnalyzedGrammar,counts:&[usize],ignores:&[Vec<u32>],d:&Delta){
 let refg=reference_analyzed(g,counts,ignores).unwrap();
 for n in 0..g.num_nonterminals as usize{
  assert_eq!(g.nullable.contains(&(n as u32)),refg.nullable.contains(&(n as u32)),"nullable {n}");
  let mut f=g.first[n].clone();add_small(f.words_mut(),&d.first[n*d.width..(n+1)*d.width],&d.ids);
  assert_eq!(f,refg.first[n],"FIRST {n}");
  let mut follow=g.follow[n].clone();add_small(follow.words_mut(),&d.follow[n*d.width..(n+1)*d.width],&d.ids);
  assert_eq!(follow,refg.follow[n],"FOLLOW {n}");
 }
}

#[cfg(test)]
fn reference_projected(grammar:&AnalyzedGrammar,counts:&[usize],ignores:&[Vec<u32>])->Option<BTreeMap<u32,BitSet>> {
    let mut relation=crate::compiler::composition::boundary::scoped_follow::scoped_follow_relation(grammar,counts,ignores)?;
    let ignored=ignores.iter().flatten().copied().collect::<std::collections::BTreeSet<_>>();
    for (&previous,blocked) in relation.iter_mut(){if !ignored.contains(&previous){for next in 0..grammar.num_terminals{if !ignored.contains(&next){blocked.clear(next as usize);}}}}
    relation.retain(|_,blocked|!blocked.is_zero());Some(relation)
}

#[cfg(test)]mod tests{
 use super::*;
 #[test]fn exact_projected_follow_on_generated_nullable_recursive_grammars(){
  let mut random=84563u64;let mut next=||{random=random.wrapping_mul(6364136223846793005).wrapping_add(1);(random>>32)as usize};
  for case in 0..1024{let nt=2+next()%35;let nc=1+next()%4;let counts=(0..nc).map(|_|1+next()%5).collect::<Vec<_>>();let nn=counts.iter().sum::<usize>();
   let ignores=(0..nc).map(|_|(0..nt as u32).filter(|_|next()%12==0).collect()).collect::<Vec<Vec<_>>>();
   let mut rules=vec![Rule{lhs:0,rhs:vec![Symbol::Nonterminal((nn-1)as u32)]}];
   for n in 0..nn{for _ in 0..(1+next()%3){let len=next()%7;let rhs=(0..len).map(|_|if next()%2==0{Symbol::Nonterminal((next()%nn)as u32)}else{Symbol::Terminal((next()%nt)as u32)}).collect();rules.push(Rule{lhs:n as u32,rhs});}}
   let g=AnalyzedGrammar::from_composed_rules(rules,nt as u32,(0..nt).map(|i|format!("T{i}")).collect(),(0..nn).map(|i|format!("N{i}")).collect(),0);
   assert_eq!(scoped_ignore_follow_relation(&g,&counts,&ignores),reference_projected(&g,&counts,&ignores),"case{case}");
  }
 }
 #[test]fn malformed_scopes_decline_like_reference_projected(){
  let g=AnalyzedGrammar::from_composed_rules(vec![Rule{lhs:0,rhs:vec![Symbol::Terminal(1)]}],2,vec!["0".into(),"1".into()],vec!["S".into()],0);
  for (counts,ignores)in[(vec![2],vec![vec![1]]),(vec![1],vec![]),(vec![1],vec![vec![2]])]{assert!(reference_projected(&g,&counts,&ignores).is_none());assert!(scoped_ignore_follow_relation(&g,&counts,&ignores).is_none());}
 }
}
