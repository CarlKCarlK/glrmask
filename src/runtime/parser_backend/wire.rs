//! A standalone parser section. The ordinary terminal templates live once in
//! the shared constraint core; this section adds only their coordinate metadata
//! and the exact EOF relation. Neither save nor load constructs an LR table.
use std::collections::BTreeSet;
use std::sync::Arc;

use super::{CommitTemplateDfas, Constraint, ParserTableStorage, TemplateParser};
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::labels::{DEFAULT_LABEL, negative_to_positive_label};

const MAGIC: &[u8; 4] = b"TPR1";
const NONE: u32 = u32::MAX;
// The row certificates are derived, not trusted wire data. Reject forged
// dimensions before allocating their Cartesian product. This is a load-time
// resource limit, not a relaxation of any accepted template's semantics.
const MAX_CERTIFICATE_BYTES: u64 = 256 * 1024 * 1024;

pub(crate) struct ParserSeed {
    state_count: u32,
    terminal_count: u32,
    skip_terminals: BTreeSet<u32>,
    completion: CommitTemplateDfas,
}

fn put_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&u32::try_from(value).expect("validated template wire count fits u32").to_le_bytes());
}
fn encode_dfa(out: &mut Vec<u8>, dfa: &DFA) {
    out.extend_from_slice(&dfa.start_state.to_le_bytes());
    put_u32(out, dfa.states.len());
    for state in &dfa.states {
        out.push(u8::from(state.is_accepting));
        put_u32(out, state.transitions.len());
        for (&label, &target) in &state.transitions {
            out.extend_from_slice(&label.to_le_bytes());
            out.extend_from_slice(&target.to_le_bytes());
        }
    }
}
fn encode_links(out: &mut Vec<u8>, links: &[Option<u32>]) {
    put_u32(out, links.len());
    for link in links { out.extend_from_slice(&link.unwrap_or(NONE).to_le_bytes()); }
}

pub(crate) fn encode(parser: &TemplateParser) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&parser.state_count.to_le_bytes());
    bytes.extend_from_slice(&parser.terminal_count.to_le_bytes());
    put_u32(&mut bytes, parser.skip_terminals.len());
    for terminal in &parser.skip_terminals { bytes.extend_from_slice(&terminal.to_le_bytes()); }
    let completion = &parser.completion_template;
    for dfa in [&completion.pop, &completion.read, &completion.push] { encode_dfa(&mut bytes, dfa); }
    for links in [&completion.pop_to_read, &completion.pop_to_push, &completion.read_to_push] {
        encode_links(&mut bytes, links);
    }
    bytes
}

struct Input<'a> { bytes: &'a [u8], offset: usize }
impl Input<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let end = self.offset.checked_add(N).ok_or("template parser offset overflow")?;
        let bytes = self.bytes.get(self.offset..end).ok_or("truncated template parser section")?;
        self.offset = end;
        Ok(bytes.try_into().expect("checked fixed-width field"))
    }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take()?)) }
    fn bounded_count(&mut self, minimum_item_bytes: usize) -> Result<usize, String> {
        let count = self.u32()? as usize;
        if count > self.bytes.len().saturating_sub(self.offset) / minimum_item_bytes {
            return Err("template parser count exceeds remaining section".into());
        }
        Ok(count)
    }
    fn dfa(&mut self) -> Result<DFA, String> {
        let start = self.u32()?;
        let count = self.bounded_count(5)?;
        let mut dfa = DFA::default();
        dfa.start_state = start;
        for _ in 0..count {
            let flag = self.take::<1>()?[0];
            if flag > 1 { return Err("invalid template acceptance flag".into()); }
            let edges = self.bounded_count(8)?;
            let id = dfa.add_state();
            dfa.states[id as usize].is_accepting = flag == 1;
            let mut previous = None;
            for _ in 0..edges {
                let label = i32::from_le_bytes(self.take()?);
                let target = self.u32()?;
                if previous.is_some_and(|old| old >= label) {
                    return Err("template edges are not strictly ordered".into());
                }
                if target as usize >= count { return Err("template edge targets a missing state".into()); }
                previous = Some(label);
                dfa.states[id as usize].transitions.insert(label, target);
            }
        }
        Ok(dfa)
    }
    fn links(&mut self) -> Result<Vec<Option<u32>>, String> {
        let count = self.bounded_count(4)?;
        (0..count).map(|_| self.u32().map(|value| (value != NONE).then_some(value))).collect()
    }
}

fn validate_dimensions(state_count: u32, terminal_count: u32) -> Result<(), String> {
    if state_count == 0 || state_count >= DEFAULT_LABEL as u32 {
        return Err("invalid template parser stack alphabet size".into());
    }
    let words = (terminal_count as u64 + 1).div_ceil(64);
    let bytes = state_count as u64 * (words * 16 + 64);
    if bytes > MAX_CERTIFICATE_BYTES {
        return Err("template parser derived certificates exceed the artifact load budget".into());
    }
    Ok(())
}

pub(crate) fn decode(bytes: &[u8]) -> Result<ParserSeed, String> {
    let mut input = Input { bytes, offset: 0 };
    if &input.take::<4>()? != MAGIC { return Err("invalid template parser section tag".into()); }
    let state_count = input.u32()?;
    let terminal_count = input.u32()?;
    validate_dimensions(state_count, terminal_count)?;
    let count = input.bounded_count(4)?;
    if count > terminal_count as usize { return Err("too many template skip terminals".into()); }
    let mut skip_terminals = BTreeSet::new();
    let mut previous = None;
    for _ in 0..count {
        let terminal = input.u32()?;
        if terminal >= terminal_count || previous.is_some_and(|old| old >= terminal) {
            return Err("invalid or unordered template skip terminals".into());
        }
        skip_terminals.insert(terminal);
        previous = Some(terminal);
    }
    let pop = input.dfa()?;
    let read = input.dfa()?;
    let push = input.dfa()?;
    let completion = CommitTemplateDfas { pop, read, push,
        pop_to_read: input.links()?, pop_to_push: input.links()?, read_to_push: input.links()?,
    };
    if input.offset != bytes.len() { return Err("trailing bytes in template parser section".into()); }
    validate_alphabet(&completion, state_count)?;
    super::compile_domain(&completion).map_err(|error| error.to_string())?;
    Ok(ParserSeed { state_count, terminal_count, skip_terminals, completion })
}

fn validate_alphabet(template: &CommitTemplateDfas, state_count: u32) -> Result<(), String> {
    for dfa in [&template.pop, &template.read, &template.push] {
        for state in &dfa.states {
            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL { continue; }
                let symbol = if label < 0 { negative_to_positive_label(label) as u32 } else { label as u32 };
                if symbol >= state_count { return Err(format!("template stack symbol {symbol} outside alphabet {state_count}")); }
            }
        }
    }
    Ok(())
}

impl ParserSeed {
    pub(crate) fn install(self, constraint: &mut Constraint) -> Result<(), String> {
        if constraint.template_dfas_by_terminal.len() != self.terminal_count as usize {
            return Err("template parser terminal count does not match its relation inventory".into());
        }
        if constraint.uses_compact_segmented_parser_runtime() || !constraint.late_grammar_slots.is_empty()
            || constraint.static_dynamic_overlay.is_some()
        {
            return Err("template parser artifact cannot contain unsupported composition machinery".into());
        }
        for template in &constraint.template_dfas_by_terminal {
            let template = template.as_deref().ok_or("missing template parser terminal relation")?;
            validate_alphabet(template, self.state_count)?;
        }
        let parser = TemplateParser::compile(self.state_count, self.terminal_count, self.skip_terminals,
            &constraint.template_dfas_by_terminal, self.completion).map_err(|error| error.to_string())?;
        constraint.template_parser = Some(Arc::new(parser));
        constraint.table = ParserTableStorage::absent();
        constraint.deferred_table_rules_blob = None;
        constraint.deferred_table_rules = Default::default();
        constraint.fast_template_dfas_by_terminal = constraint.compute_fast_template_dfas();
        Ok(())
    }
}
