//! Lossless compact encoding of the same split programs used by the runtime.
//! This changes neither graph numbering nor transition choice: explicit dead
//! edges which shadow DEFAULT remain present, and omitted trailing link arrays
//! retain their exact length. No runtime representation is specialized here.
use super::{CommitTemplateDfas, DFA, Input, ParserSeed, TemplateParser, validate_dimensions};
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label, negative_to_positive_label};
use crate::runtime::artifact::TemplateDfasByTerminal;
use std::collections::BTreeSet;
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"TPR2";
const MAX_DECODED_STATES: usize = 2_000_000;
const MAX_DECODED_EDGES: usize = 8_000_000;

#[derive(Clone, Copy)]
enum Phase { Pop, Read, Push }

fn put(out: &mut Vec<u8>, mut value: u32) {
    while value >= 128 {
        out.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    out.push(value as u8);
}
fn count(out: &mut Vec<u8>, value: usize) {
    put(out, u32::try_from(value).expect("validated template count fits u32"));
}
fn dfa(out: &mut Vec<u8>, graph: &DFA, phase: Phase, alphabet: u32) {
    count(out, graph.states.len());
    if graph.states.is_empty() { assert_eq!(graph.start_state, 0); return; }
    put(out, graph.start_state);
    for state in &graph.states {
        let size=u32::try_from(state.transitions.len()).expect("template row fits u32");
        put(out, size.checked_mul(2).expect("template row count fits packed flag") | u32::from(state.is_accepting));
        let mut previous = None;
        for (&label, &target) in &state.transitions {
            let symbol = match phase {
                Phase::Pop if label == DEFAULT_LABEL => alphabet,
                Phase::Push => negative_to_positive_label(label) as u32,
                _ => label as u32,
            };
            let delta = previous.map_or(symbol, |old| symbol.checked_sub(old).expect("normalized labels preserve order"));
            debug_assert!(previous.is_none() || delta > 0);
            put(out,delta); put(out,target);
            previous = Some(symbol);
        }
    }
}
fn links(out: &mut Vec<u8>, links: &[Option<u32>]) {
    count(out,links.len());
    for link in links { put(out,link.map_or(0, |target| target.checked_add(1).expect("validated target is not the sentinel"))); }
}
fn program(out: &mut Vec<u8>, template: &CommitTemplateDfas, alphabet: u32) {
    dfa(out,&template.pop,Phase::Pop,alphabet);
    dfa(out,&template.read,Phase::Read,alphabet);
    dfa(out,&template.push,Phase::Push,alphabet);
    links(out,&template.pop_to_read); links(out,&template.pop_to_push); links(out,&template.read_to_push);
}

pub(super) fn encode(parser: &TemplateParser, templates: &TemplateDfasByTerminal) -> Vec<u8> {
    assert_eq!(templates.len(),parser.terminal_count as usize);
    let mut out=Vec::new();
    out.extend_from_slice(if parser.composition.is_some() { b"TPR3" } else { MAGIC });
    put(&mut out,parser.state_count); put(&mut out,parser.terminal_count);
    count(&mut out,parser.skip_terminals.len());
    let mut previous=None;
    for &terminal in &parser.skip_terminals {
        put(&mut out,previous.map_or(terminal,|old|terminal-old)); previous=Some(terminal);
    }
    for template in templates {
        program(&mut out,template.as_ref().expect("every terminal has an explicit template"),parser.state_count);
    }
    program(&mut out,&parser.completion_template,parser.state_count);
    if let Some(composition) = &parser.composition {
        put(&mut out, composition.control_start);
        count(&mut out, composition.programs.len());
        for template in &composition.programs {
            program(&mut out, template.as_ref().expect("validated scoped relation"), parser.state_count);
        }
    }
    out
}

impl Input<'_> {
    fn var(&mut self) -> Result<u32,String> {
        let mut value=0;
        for shift in (0..=28).step_by(7) {
            let byte=self.take::<1>()?[0];
            if shift==28 && byte & 0xf0 != 0 { return Err("template varint overflows u32".into()); }
            value |= u32::from(byte & 127) << shift;
            if byte & 128 == 0 {
                if shift>0 && byte==0 { return Err("noncanonical template varint".into()); }
                return Ok(value);
            }
        }
        Err("overlong template varint".into())
    }
    fn variable_count(&mut self, minimum_item_bytes: usize) -> Result<usize,String> {
        let count=self.var()? as usize;
        if count > self.bytes.len().saturating_sub(self.offset)/minimum_item_bytes {
            return Err("compact template count exceeds remaining section".into());
        }
        Ok(count)
    }
    fn compact_dfa(&mut self, phase:Phase, alphabet:u32, budget:&mut Budget) -> Result<DFA,String> {
        let count=self.variable_count(1)?;
        budget.states=budget.states.checked_add(count).ok_or("template state count overflow")?;
        if budget.states>MAX_DECODED_STATES { return Err("template graph exceeds decoded-state budget".into()); }
        let mut graph=DFA::default();
        if count==0 { return Ok(graph); }
        graph.start_state=self.var()?;
        if graph.start_state as usize >= count { return Err("compact template start outside graph".into()); }
        for _ in 0..count {
            let flags=self.var()?; let edges=(flags>>1) as usize;
            if edges>self.bytes.len().saturating_sub(self.offset)/2 {
                return Err("compact template edge count exceeds remaining section".into());
            }
            budget.edges=budget.edges.checked_add(edges).ok_or("template edge count overflow")?;
            if budget.edges>MAX_DECODED_EDGES { return Err("template graph exceeds decoded-edge budget".into()); }
            let id=graph.add_state(); graph.states[id as usize].is_accepting=flags&1!=0;
            let mut previous:Option<u32>=None;
            for _ in 0..edges {
                let delta=self.var()?;
                let symbol=if let Some(old)=previous {
                    if delta==0 { return Err("duplicate compact template edge label".into()); }
                    old.checked_add(delta).ok_or("template edge delta overflow")?
                } else { delta };
                let label=if symbol<alphabet {
                    match phase { Phase::Push=>encode_negative_label(symbol), _=>symbol as i32 }
                } else if symbol==alphabet && matches!(phase,Phase::Pop) {
                    DEFAULT_LABEL
                } else { return Err("compact template edge outside its phase alphabet".into()); };
                let target=self.var()?;
                if target as usize>=count { return Err("compact template edge targets a missing state".into()); }
                graph.states[id as usize].transitions.insert(label,target);
                previous=Some(symbol);
            }
        }
        Ok(graph)
    }
    fn compact_links(&mut self, source:usize, target:usize) -> Result<Vec<Option<u32>>,String> {
        let count=self.variable_count(1)?;
        if count>source { return Err("compact phase links exceed source states".into()); }
        let mut links=Vec::with_capacity(count);
        for _ in 0..count {
            let value=self.var()?;
            if value==0 { links.push(None); }
            else {
                let id=value-1;
                if id as usize>=target { return Err("compact phase link targets missing state".into()); }
                links.push(Some(id));
            }
        }
        Ok(links)
    }
    fn compact_program(&mut self, alphabet:u32,budget:&mut Budget)->Result<CommitTemplateDfas,String> {
        let pop=self.compact_dfa(Phase::Pop,alphabet,budget)?;
        let read=self.compact_dfa(Phase::Read,alphabet,budget)?;
        let push=self.compact_dfa(Phase::Push,alphabet,budget)?;
        let pop_to_read=self.compact_links(pop.states.len(),read.states.len())?;
        let pop_to_push=self.compact_links(pop.states.len(),push.states.len())?;
        let read_to_push=self.compact_links(read.states.len(),push.states.len())?;
        Ok(CommitTemplateDfas {pop,read,push,pop_to_read,pop_to_push,read_to_push})
    }
}
#[derive(Default)]
struct Budget {states:usize,edges:usize}

pub(super) fn decode(bytes:&[u8])->Result<ParserSeed,String> {
    let mut input=Input{bytes,offset:0};
    let magic = input.take::<4>()?;
    let composed = &magic == b"TPR3";
    if &magic != MAGIC && !composed {return Err("invalid compact parser section".into());}
    let state_count=input.var()?; let terminal_count=input.var()?;
    validate_dimensions(state_count,terminal_count)?;
    let count=input.variable_count(1)?;
    if count>terminal_count as usize {return Err("too many compact skip terminals".into());}
    let mut skip_terminals=BTreeSet::new(); let mut previous:Option<u32>=None;
    for _ in 0..count {
        let delta=input.var()?;
        let terminal=if let Some(old)=previous {
            if delta==0 {return Err("duplicate compact skip terminal".into());}
            old.checked_add(delta).ok_or("compact skip delta overflow")?
        } else {delta};
        if terminal>=terminal_count {return Err("compact skip terminal outside alphabet".into());}
        skip_terminals.insert(terminal); previous=Some(terminal);
    }
    let program_count=(terminal_count as usize).checked_add(1).ok_or("compact program count overflow")?;
    // Even an empty program has three empty graphs and three empty link arrays.
    if program_count>bytes.len().saturating_sub(input.offset)/6 {
        return Err("compact program inventory exceeds remaining section".into());
    }
    let mut budget=Budget::default(); let mut templates=Vec::with_capacity(terminal_count as usize);
    for _ in 0..terminal_count {
        templates.push(Some(Arc::new(input.compact_program(state_count,&mut budget)?)));
    }
    let completion=input.compact_program(state_count,&mut budget)?;
    let composition = if composed {
        let control_start = input.var()?;
        let extra_count = input.variable_count(6)?;
        if control_start as usize > extra_count {
            return Err("composition control offset exceeds its scoped relation inventory".into());
        }
        let total = terminal_count.checked_add(u32::try_from(extra_count)
            .map_err(|_| "too many scoped parser relations")?).ok_or("scoped terminal coordinate overflow")?;
        if total == u32::MAX { return Err("scoped parser labels collide with EOF".into()); }
        validate_dimensions(state_count, total)?;
        let controls = extra_count - control_start as usize;
        if state_count as u64 * (controls as u64 * 4 + 24) > super::MAX_CERTIFICATE_BYTES {
            return Err("composition control certificates exceed the load budget".into());
        }
        let mut programs = Vec::with_capacity(extra_count);
        for _ in 0..extra_count {
            programs.push(Some(Arc::new(input.compact_program(state_count, &mut budget)?)));
        }
        Some((control_start, programs))
    } else { None };
    if input.offset!=bytes.len() {return Err("trailing compact parser bytes".into());}
    // Cycle/phase validation is shared with domain derivation during install.
    // No materialization path can expose these graphs before it succeeds.
    Ok(ParserSeed {state_count,terminal_count,skip_terminals,completion,programs:Some(templates),composition})
}
