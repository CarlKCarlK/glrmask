//! Optional owner/control-phase product for the bounded signed program.
//!
//! A valid ordinary terminal transfer stays in its component. Entry changes
//! parent to child; Finish changes child to parent. The existing nonnullable
//! closure certificate admits only R*E* within a zero-byte gap. This product
//! removes structurally impossible words before template substitution. It is
//! an execution-domain proof, not a claim about malformed arbitrary stacks.
use super::{SignedLinkContext, FragmentLibrary, append_weighted_fragment, NWA, DWA, Weight};
use std::collections::{BTreeMap,BTreeSet,VecDeque};
use super::template_top;
use crate::compiler::glr::table::Action;

fn after_control(owner: u32, entered: bool, parent: u32, child: u32, entry: bool)
    -> Option<(u32,bool)>
{
    if entry { (owner==parent).then_some((child,true)) }
    else { (owner==child && !entered).then_some((parent,false)) }
}

pub(super) fn assemble(
    context: &SignedLinkContext, library: &FragmentLibrary, lexical: &DWA, top_flow:bool,
) -> Result<Option<(NWA,usize,usize)>,String> {
    if context.global_ignores && context.ignore_terminals.iter().any(Option::is_some) {
        return Ok(None);
    }
    if context.links.iter().any(|link|link.child_start_nullable) { return Ok(None); }
    // A consuming pop-only effect can expose a different component's frame.
    // Such tables need a richer owner abstraction; retain the exact old path.
    for owner in 0..context.state_offsets.len() {
        for row in &context.component_table(owner as u32)?.action {for (_,action) in row.iter(){
            let pop_only=match action {
                Action::StackShifts(shifts)=>shifts.iter().any(|s|s.pop>0&&s.pushes.is_empty()),
                Action::GuardedStackShifts(shifts)=>shifts.iter().any(|s|s.pop>0&&s.pushes.is_empty()),
                _=>false,
            };
            if pop_only{return Ok(None);}
        }}
    }
    let started=std::time::Instant::now();
    let owners=context.state_offsets.len();
    let depths=context.closure.max_controls_per_gap as usize+1;
    let vertices=lexical.states().len();
    let Some(count)=vertices.checked_mul(depths).and_then(|n|n.checked_mul(owners)).and_then(|n|n.checked_mul(2)) else {return Ok(None)};
    if owners==0 || count>250_000 { return Ok(None); }
    let id=|v:usize,d:usize,o:usize,e:bool|(((v*depths+d)*owners+o)*2)+usize::from(e);
    let mut graph=vec![Vec::<usize>::new();count];
    let mut reverse=vec![Vec::<usize>::new();count];
    let mut edges=0usize;
    for (v,row) in lexical.states().iter().enumerate() {
        for depth in 0..depths { for owner in 0..owners { for entered in [false,true] {
            let source=id(v,depth,owner,entered);
            for (label,target,weight) in row.transitions.entries() {
                if label<0 { return Err("negative lexical label in owner product".into()); }
                if weight.is_empty() {continue;}
                if target as usize>=vertices {return Err("invalid owner-product lexical target".into());}
                let (terminal_owner,_)=context.terminal_owner(label as u32)?;
                if terminal_owner==owner as u32 {graph[source].push(id(target as usize,0,owner,false));}
            }
            if depth+1<depths { for link in &context.links { for entry in [true,false] {
                if let Some((next,phase))=after_control(owner as u32,entered,link.parent_component,link.child_component,entry) {
                    if next as usize>=owners {return Err("invalid owner-product control component".into());}
                    graph[source].push(id(v,depth+1,next as usize,phase));
                }
            }}}
            graph[source].sort_unstable();graph[source].dedup();
            edges+=graph[source].len();if edges>2_000_000{return Ok(None);}
            for &target in &graph[source]{reverse[target].push(source);}
        }}}
    }
    let starts=(0..owners).map(|o|id(lexical.start_state() as usize,0,o,false)).collect::<Vec<_>>();
    let mut reachable=vec![false;count];let mut queue=VecDeque::from(starts.clone());
    while let Some(q)=queue.pop_front(){if reachable[q]{continue;}reachable[q]=true;queue.extend(graph[q].iter().copied());}
    let mut productive=vec![false;count];let mut queue=VecDeque::new();
    for (v,row) in lexical.states().iter().enumerate(){
        if row.final_weight.as_ref().is_some_and(|w|!w.is_empty()) {
            for owner in 0..owners{queue.push_back(id(v,0,owner,false));}
        }
    }
    while let Some(q)=queue.pop_front(){if productive[q]{continue;}productive[q]=true;queue.extend(reverse[q].iter().copied());}
    let mut live=reachable.iter().zip(&productive).map(|(a,b)|*a&&*b).collect::<Vec<_>>();
    let flow_started=std::time::Instant::now();
    let mut summaries=BTreeMap::new();
    let mut structural_templates=BTreeMap::new();
    let owner_tops=(0..owners).map(|owner| {
        let begin=context.state_offsets[owner];
        context.component_table(owner as u32).map(|table|(begin..begin+table.num_states).collect::<BTreeSet<_>>())
    }).collect::<Result<Vec<_>,_>>()?;
    let mut top_sets=vec![BTreeSet::<u32>::new();count];
    let mut unknown_summaries=0usize;
    if top_flow {
        for (&terminal,template) in &library.templates.by_terminal_nwa {
            // append_weighted_fragment deliberately treats templates as
            // unweighted structure: every existing edge is stamped with the
            // caller's weight and every Some(final) is redirected. Analyze
            // precisely that unit-relabeled view, not placeholder annotations.
            let mut structural=template.clone();
            for row in structural.states_mut() {
                if row.final_weight.is_some(){row.final_weight=Some(Weight::all());}
                for (_,w) in &mut row.epsilons{*w=Weight::all();}
                for (_,w) in row.transitions.values_mut().flatten(){*w=Weight::all();}
            }
            let summary=template_top::summarize(&structural,context.total_scoped_states);
            structural_templates.insert(terminal,structural);
            if summary.is_none(){unknown_summaries+=1;}
            summaries.insert(terminal,summary);
        }
        for owner in 0..owners{top_sets[id(lexical.start_state() as usize,0,owner,false)]=owner_tops[owner].clone();}
        let mut degree=reverse.iter().map(Vec::len).collect::<Vec<_>>();
        let mut queue=(0..count).filter(|&q|degree[q]==0).collect::<VecDeque<_>>();let mut visited=0usize;
        while let Some(q)=queue.pop_front(){visited+=1;
            if live[q]&&!top_sets[q].is_empty(){
                let entered=q%2!=0;let owner=q/2%owners;let depth=q/(2*owners)%depths;let v=q/(2*owners*depths);
                let current=top_sets[q].clone();
                for (label,target,weight) in lexical.states()[v].transitions.entries(){
                    if weight.is_empty(){continue;}let (terminal_owner,_)=context.terminal_owner(label as u32)?;
                    if terminal_owner as usize!=owner{continue;}
                    let output=summaries.get(&(label as u32)).and_then(Option::as_ref)
                        .and_then(|summary|summary.output(&current)).unwrap_or_else(||owner_tops[owner].clone());
                    if !output.is_subset(&owner_tops[owner]){return Ok(None);}
                    top_sets[id(target as usize,0,owner,false)].extend(output);
                }
                if depth+1<depths{for link in &context.links{for entry in [true,false]{
                    if let Some((next,phase))=after_control(owner as u32,entered,link.parent_component,link.child_component,entry){
                        let destination=id(v,depth+1,next as usize,phase);
                        if entry{top_sets[destination].insert(context.state_offsets[next as usize]+link.child_start);}
                        else{top_sets[destination].extend(owner_tops[next as usize].iter().copied());}
                    }
                }}}
            }
            for &target in &graph[q]{degree[target]-=1;if degree[target]==0{queue.push_back(target);}}
        }
        if visited!=count{return Ok(None);}
        for q in 0..count{live[q]&=!top_sets[q].is_empty();}
    }
    let flow_ms=flow_started.elapsed().as_secs_f64()*1000.0;
    let mut arena=NWA::new(0,0);let root=arena.add_state();arena.set_start_states(vec![root]);
    let mut ports=vec![u32::MAX;count];
    for (q,&yes) in live.iter().enumerate(){if yes{ports[q]=arena.add_state();}}
    for q in starts{if live[q]{arena.add_epsilon(root,ports[q],Weight::all());}}
    for (v,row) in lexical.states().iter().enumerate(){
        if let Some(weight)=row.final_weight.as_ref().filter(|w|!w.is_empty()){
            for owner in 0..owners{let q=id(v,0,owner,false);if live[q]{arena.set_final_weight(ports[q],weight.clone());}}
        }
    }
    let mut ordinary_states=0usize;let mut control_states=0usize;
    let mut restricted_cache=BTreeMap::<(u32,Vec<u32>),NWA>::new();
    let mut reduced_fragments=0usize;let mut before_states=0usize;
    for (v,row) in lexical.states().iter().enumerate(){
        for (label,target,weight) in row.transitions.entries(){
            if weight.is_empty(){continue;}
            let (owner,_)=context.terminal_owner(label as u32)?;let owner=owner as usize;
            let destination=id(target as usize,0,owner,false);
            if !live[destination]{continue;}
            let mut source_ports=Vec::new();let mut input_tops=BTreeSet::new();
            for depth in 0..depths{for entered in [false,true]{let q=id(v,depth,owner,entered);if live[q]{source_ports.push(ports[q]);if top_flow{input_tops.extend(top_sets[q].iter().copied());}}}}
            if source_ports.is_empty(){continue;}
            let fragment=library.templates.by_terminal_nwa.get(&(label as u32))
                .ok_or_else(||format!("missing owner-product terminal {label}"))?;
            let original_states=fragment.num_states() as usize;before_states+=original_states;
            let key=(label as u32,input_tops.iter().copied().collect::<Vec<_>>());
            if top_flow&&!restricted_cache.contains_key(&key){
                let structural=structural_templates.get(&(label as u32)).expect("structural template prepared");
                let candidate=template_top::restrict_first(structural,&input_tops,context.total_scoped_states);
                let result=candidate.filter(|x|x.num_states()<fragment.num_states()||x.num_transitions()<fragment.num_transitions())
                    .unwrap_or_else(||fragment.clone());
                restricted_cache.insert(key.clone(),result);
            }
            let fragment=restricted_cache.get(&key).unwrap_or(fragment);
            reduced_fragments+=usize::from((fragment.num_states() as usize)<original_states);
            let body=append_weighted_fragment(&mut arena,fragment,weight,ports[destination])?;
            ordinary_states+=fragment.states().len();
            for source in source_ports{for &start in &body.start_states{arena.add_epsilon(source,start,Weight::all());}}
        }
        for depth in 0..depths.saturating_sub(1){for owner in 0..owners{for entered in [false,true]{
            let source=id(v,depth,owner,entered);if !live[source]{continue;}
            for (link_index,link) in context.links.iter().enumerate(){for entry in [true,false]{
                let Some((next,phase))=after_control(owner as u32,entered,link.parent_component,link.child_component,entry) else{continue};
                let target=id(v,depth+1,next as usize,phase);if !live[target]{continue;}
                let key=if entry{library.entry_keys[link_index]}else{library.finish_keys[link_index]};
                let fragment=library.templates.by_terminal_nwa.get(&key)
                    .ok_or_else(||"missing owner-product control fragment".to_owned())?;
                let body=append_weighted_fragment(&mut arena,fragment,&Weight::all(),ports[target])?;
                control_states+=fragment.states().len();
                for start in body.start_states{arena.add_epsilon(ports[source],start,Weight::all());}
            }}
        }}}
    }
    if std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some(){
        eprintln!("[glrmask/profile][boundary_owner_program] top_flow={top_flow} abstract={count} live={} ordinary={ordinary_states} controls={control_states} states={} edges={} ms={:.3}",
            live.iter().filter(|&&b|b).count(),arena.num_states(),arena.num_transitions(),started.elapsed().as_secs_f64()*1000.0);
        if top_flow{eprintln!("[glrmask/profile][boundary_top_flow] summaries={} unknown={unknown_summaries} top_states={} cache={} reduced_fragments={reduced_fragments} input_fragment_states={before_states} retained_states={ordinary_states} flow_ms={flow_ms:.3}",
            summaries.len(),top_sets.iter().map(BTreeSet::len).sum::<usize>(),restricted_cache.len());}
    }
    Ok(Some((arena,ordinary_states,control_states)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_preserve_owners_and_return_before_entry_phase(){
        assert_eq!(after_control(1,false,0,1,false),Some((0,false)));
        assert_eq!(after_control(0,false,0,2,true),Some((2,true)));
        assert_eq!(after_control(2,true,0,2,false),None);
        assert_eq!(after_control(2,true,2,3,true),Some((3,true)));
        assert_eq!(after_control(0,false,1,2,true),None);
        assert_eq!(after_control(3,false,1,2,false),None);
    }
}
