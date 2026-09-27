//! Link-local, effect-certified predecessor support for boundary compilation.
//!
//! This is a necessary-context rejection proof, not a token-admission rule.
//! A parser input stack remains in the suffix-closed abstract language by
//! induction over all mapped consuming actions, reduction GOTOs and controls.
//! Unsupported coordinates or exhausted budgets return an error to an exact
//! fallback. Complete-domain validation uses the independent ordinary parser
//! normalizer, not the same optimized backend on both sides.
mod domain;
mod effects;
mod coarse;
mod positive;
mod template;
pub(super) mod trim;
mod compare;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::glr::parser::ScopedSubgrammarLink;
use crate::automata::weighted_u32::{nwa::NWA,dwa::DWA};
pub(super) use coarse::Certificate;

pub(super) fn build(tables:&[&GLRTable],offsets:&[u32],links:&[ScopedSubgrammarLink],alphabet:u32)
    -> Result<Certificate,String>
{
    let links=links.iter().map(|l|(l.parent_component,l.slot_terminal,l.child_component,
        l.child_start,l.return_pop,l.child_start_nullable)).collect::<Vec<_>>();
    let effects=effects::read_effects(tables,offsets,&links,alphabet)?;
    coarse::certify(&[0],&effects,alphabet)
}

impl Certificate {
    pub(super) fn template_support(&self)->Option<template::TemplateReadSupport>{
        template::TemplateReadSupport::new(self)
    }
    pub(super) fn restrict(&self,nwa:&mut NWA)->Result<positive::Stats,String>{
        positive::restrict(nwa,self)
    }
    pub(super) fn compare(&self,reference:&DWA,candidate:&DWA)->Result<(),String>{
        let result=compare::compare(reference,candidate,&self.domain,Default::default())?;
        if let Some(difference)=result.difference {
            return Err(format!("predecessor support changed complete-domain mask: {difference:?}"));
        }
        eprintln!("[glrmask/validate][boundary_predecessor_support] exact=true products={} branches={} regions={} domain_subsets={}",
            result.products,result.branches,result.region_visits,result.domain_subsets);
        Ok(())
    }
    pub(super) fn native_context(&self)->Option<crate::compiler::stages::parser_dwa::FiniteParserReadSupport>{
        let d=&self.domain;
        if d.nodes.iter().any(|n|!n.epsilons.is_empty()){return None;}
        let rows=d.nodes.iter().map(|node|node.transitions.iter().map(|(&label,targets)|{
            (targets.len()==1).then(||(label,*targets.first().unwrap()))
        }).collect::<Option<Vec<_>>>()).collect::<Option<Vec<_>>>()?;
        crate::compiler::stages::parser_dwa::FiniteParserReadSupport::new_checked(
            d.alphabet as usize,d.start as usize,&rows,&d.coaccessible(),d.nodes[d.start as usize].accepting)
    }

    pub(super) fn state_count(&self)->usize { self.domain.nodes.len() }
    pub(super) fn effect_count(&self)->usize { self.stats.effects }
}

/// An embedded artifact sees arbitrary frames below its former local root.
/// All such labels are observationally OTHER: this DWA tests only local IDs
/// and DEFAULT. Close a local-root/OTHER initial stack under every original
/// local effect and OTHER-tail extension, then compare on complete words.
pub(super) fn compare_external(
    tables:&[&GLRTable],offsets:&[u32],links:&[ScopedSubgrammarLink],alphabet:u32,
    reference:&DWA,candidate:&DWA,
)->Result<(),String>{
    let tuples=links.iter().map(|l|(l.parent_component,l.slot_terminal,l.child_component,
        l.child_start,l.return_pop,l.child_start_nullable)).collect::<Vec<_>>();
    let mut effects=effects::read_effects(tables,offsets,&tuples,alphabet)?;
    effects.push(effects::Effect{source:alphabet,pop:0,pushes:vec![alphabet]});
    let extended=coarse::certify(&[0,alphabet],&effects,alphabet.checked_add(1).ok_or("external label overflow")?)?;
    let result=compare::compare(reference,candidate,&extended.domain,Default::default())?;
    if let Some(difference)=result.difference{
        return Err(format!("embedded predecessor support changed external-tail mask: {difference:?}"));
    }
    eprintln!("[glrmask/validate][boundary_predecessor_external] exact=true products={} branches={} domain_subsets={}",
        result.products,result.branches,result.domain_subsets);
    Ok(())
}
