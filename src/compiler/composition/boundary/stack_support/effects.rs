//! Conservative action-to-stack-effect projection. Reduction pops expose
//! suffixes; all local GOTOs are included, so forgetting reduction/guard
//! feasibility only enlarges the domain. CALLs retain their saved parent
//! frame and child start; supported slot shapes were checked by the linker.
use std::collections::BTreeSet;
use crate::compiler::glr::table::{Action,GLRTable};
use crate::compiler::glr::analysis::EOF;
#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord)]
pub(super) struct Effect { pub source:u32, pub pop:usize, pub pushes:Vec<u32> }
pub(super) type Link=(u32,u32,u32,u32,u32,bool);
fn action_effects(action:&Action,source:u32,offset:u32,extra_nonreplace:bool,out:&mut BTreeSet<Effect>){
    let mut add=|pop:u32,pushes:&[u32]|{out.insert(Effect{source,pop:pop as usize,pushes:pushes.iter().map(|q|offset+q).collect()});};
    match action{
        Action::Shift(q,replace)=>{add(u32::from(*replace),&[*q]);if *replace&&extra_nonreplace{add(0,&[*q]);}},
        Action::ReplaceShifts(targets)=>for q in targets.iter(){add(1,&[*q]);},
        Action::StackShifts(shifts)=>for s in shifts{add(s.pop,&s.pushes);},
        Action::GuardedStackShifts(shifts)=>for s in shifts{add(s.pop,&s.pushes);},
        Action::Split{shift, ..}=>{if let Some((q,replace))=shift{add(u32::from(*replace),&[*q]);if *replace&&extra_nonreplace{add(0,&[*q]);}}},
        Action::Reduce(..)|Action::Accept|Action::Skip=>{},
    }
}

pub(super) fn read_effects(tables:&[&GLRTable],offsets:&[u32],links:&[Link],alphabet:u32)->Result<Vec<Effect>,String>{
    if tables.is_empty()||offsets.len()!=tables.len()||offsets[0]!=0{return Err("invalid scoped table layout".into());}
    for (i,table) in tables.iter().enumerate(){
        let end=offsets[i].checked_add(table.num_states).ok_or("table state overflow")?;
        if end!=offsets.get(i+1).copied().unwrap_or(alphabet){return Err("table offsets are not complete disjoint layout".into());}
        if !table.control_terminals.is_empty(){return Err("unmapped in-table control terminals".into());}
    }
    let mut effects=BTreeSet::new();
    for (owner,table) in tables.iter().enumerate(){
        let offset=offsets[owner];
        for (q,row) in table.action.iter().enumerate(){for (terminal,action) in row.iter(){
            action_effects(action,offset+q as u32,offset,table.forwarded_shifts.contains(&(q as u32,terminal)),&mut effects);
        }}
        // A reduction first exposes an existing nonempty suffix of a stack;
        // every such suffix is in the adjacency language. Its actual GOTO is
        // then one of these effects. Include all GOTOs even if not reached by
        // a real reduction: this is an intentional overapproximation.
        for (q,row) in table.goto.iter().enumerate(){for (_, &(target,replace)) in row.iter(){
            effects.insert(Effect{source:offset+q as u32,pop:usize::from(replace),pushes:vec![offset+target]});
        }}
    }
    for &(parent,slot,child,child_start,return_pop,nullable) in links{
        let p=tables.get(parent as usize).ok_or("invalid parent")?;
        let c=tables.get(child as usize).ok_or("invalid child")?;
        if child_start>=c.num_states{return Err("invalid child start".into());}
        for q in 0..p.num_states{
            match p.action(q,slot){
                Some(Action::Shift(target,replace))|Some(Action::Split{shift:Some((target,replace)),..})=>{
                    effects.insert(Effect{source:offsets[parent as usize]+q,pop:usize::from(*replace),
                        pushes:vec![offsets[parent as usize]+target,offsets[child as usize]+child_start]});
                },
                None|Some(Action::Reduce(..))|Some(Action::Split{shift:None,accept:false,..})=>{},
                other=>return Err(format!("unsupported CALL effect shape: {other:?}")),
            }
        }
        for q in 0..c.num_states{
            if matches!(c.action(q,EOF),Some(Action::Accept)|Some(Action::Split{accept:true,..})){
                effects.insert(Effect{source:offsets[child as usize]+q,pop:return_pop as usize,pushes:vec![]});
            }
        }
        if nullable{effects.insert(Effect{source:offsets[child as usize]+child_start,pop:1,pushes:vec![]});}
    }
    if effects.len() > 200_000 || effects.iter().any(|e| e.source >= alphabet
        || e.pushes.iter().any(|&q| q >= alphabet) || e.pop > 1024 || e.pushes.len() > 1024) {
        return Err("stack effect coordinate or resource bound".into());
    }
    Ok(effects.into_iter().collect())
}
#[cfg(test)]
mod mapper_tests{
    use super::*;
    use glrmask_glr::__private::glr::table::action::{StackShift,GuardedStackShift,StackShiftGuard};
    #[test]
    fn maps_every_consuming_action_and_forwards_conservatively(){
        let cases=vec![Action::Shift(3,false),Action::Shift(3,true),Action::ReplaceShifts(vec![2,3].into()),
            Action::StackShifts(vec![StackShift{pop:2,pushes:vec![1,3]}]),
            Action::GuardedStackShifts(vec![GuardedStackShift{guards:vec![StackShiftGuard{pop:3,states:vec![2]}],pop:1,pushes:vec![2,3]}]),
            Action::Split{shift:Some((3,true)),reduces:vec![(0,2)],accept:false}];
        for action in cases{
            let mut out=BTreeSet::new();action_effects(&action,11,10,true,&mut out);assert!(!out.is_empty());
            for e in out{assert_eq!(e.source,11);assert!(e.pushes.iter().all(|q|(10..14).contains(q)));}
        }
    }
}
