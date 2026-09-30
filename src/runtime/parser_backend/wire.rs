//! A standalone parser section. The ordinary terminal templates live once in
//! the shared constraint core; this section adds only their coordinate metadata
//! and the exact EOF relation. Neither save nor load constructs an LR table.
mod compact;

use std::collections::BTreeSet;
use crate::runtime::artifact::TemplateDfasByTerminal;
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
    programs: Option<TemplateDfasByTerminal>,
}

pub(crate) fn encode(parser: &TemplateParser, templates: &TemplateDfasByTerminal) -> Vec<u8> {
    let bytes = compact::encode(parser, templates);
    if std::env::var_os("GLRMASK_VALIDATE_TEMPLATE_WIRE").is_some() {
        let decoded = compact::decode(&bytes).expect("newly encoded template programs must decode");
        assert_eq!(decoded.state_count, parser.state_count);
        assert_eq!(decoded.terminal_count, parser.terminal_count);
        assert_eq!(decoded.skip_terminals, parser.skip_terminals);
        assert!(bincode::serialize(decoded.programs.as_ref().unwrap()).unwrap() == bincode::serialize(templates).unwrap(),
            "compact template wire changed raw graph numbering, edges, or links");
        assert!(bincode::serialize(&decoded.completion).unwrap() == bincode::serialize(&*parser.completion_template).unwrap(),
            "compact template wire changed raw EOF completion program");
        eprintln!("[glrmask/validate][template_parser_wire] exact_graphs=true templates={} bytes={}", templates.len(), bytes.len());
    }
    bytes
}

/// The vocabulary digest authenticates the binding coordinate, not the
/// authorship of this artifact. The parser program still comes only from the
/// saved acyclic templates, never from an LR reconstruction.
pub(crate) fn encode_external(parser: &TemplateParser, templates: &TemplateDfasByTerminal, digest: [u8; 32]) -> Vec<u8> {
    let body = encode(parser, templates);
    let mut bytes = Vec::with_capacity(36 + body.len());
    bytes.extend_from_slice(b"TPX1");
    bytes.extend_from_slice(&digest);
    bytes.extend_from_slice(&body);
    bytes
}

pub(crate) fn decode_external(bytes: &[u8], vocab: &crate::Vocab) -> Result<ParserSeed, String> {
    if bytes.len() < 36 || !bytes.starts_with(b"TPX1") {
        return Err("invalid external-vocabulary template parser section".into());
    }
    if bytes[4..36] != crate::compiler::compile::vocab_content_digest(vocab) {
        return Err("template parser artifact does not match the supplied vocabulary".into());
    }
    decode(&bytes[36..])
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
    if bytes.starts_with(b"TPR2") { return compact::decode(bytes); }
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
    Ok(ParserSeed { state_count, terminal_count, skip_terminals, completion, programs: None })
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
    pub(crate) fn install(mut self, constraint: &mut Constraint) -> Result<(), String> {
        if let Some(programs) = self.programs.take() {
            if !constraint.template_dfas_by_terminal.is_empty() {
                return Err("compact template artifact has duplicate core parser programs".into());
            }
            constraint.template_dfas_by_terminal = programs;
        }
        if !constraint.terminal_display_names.is_empty()
            && constraint.terminal_display_names.len() != self.terminal_count as usize
        {
            return Err("template parser terminal coordinate disagrees with lexical metadata".into());
        }
        if constraint.template_dfas_by_terminal.len() != self.terminal_count as usize {
            return Err("template parser terminal count does not match its relation inventory".into());
        }
        if constraint.uses_compact_segmented_parser_runtime() || !constraint.late_grammar_slots.is_empty()
            || constraint.static_dynamic_overlay.is_some()
        {
            return Err("template parser artifact cannot contain unsupported composition machinery".into());
        }
        // Complete graph and alphabet validation is shared with input-domain
        // and fast-view preparation. This also checks legacy core programs;
        // no persisted flag is trusted and no validation is deferred to runtime.
        let (parser, runtime) = TemplateParser::compile_with_runtime(self.state_count,
            self.terminal_count, self.skip_terminals, &constraint.template_dfas_by_terminal,
            self.completion).map_err(|error| error.to_string())?;
        constraint.template_parser = Some(Arc::new(parser));
        constraint.table = ParserTableStorage::absent();
        constraint.deferred_table_rules_blob = None;
        constraint.deferred_table_rules = Default::default();
        constraint.fast_template_dfas_by_terminal = runtime;
        Ok(())
    }
}

/// The shared core keeps its established field shape. New parser sections own
/// the programs once; legacy/ordinary encoders continue writing the old vector.
/// A scoped guard restores the thread-local policy even if serialization fails.
pub(crate) mod core_programs {
    use std::cell::Cell;
    use serde::{Serialize, Deserialize, Serializer, Deserializer};
    use super::TemplateDfasByTerminal;
    thread_local! { static EXTERNAL: Cell<bool> = const { Cell::new(false) }; }
    pub(crate) struct Guard(bool);
    pub(crate) fn externalize(enabled: bool) -> Guard {
        Guard(EXTERNAL.with(|value| value.replace(enabled)))
    }
    impl Drop for Guard {
        fn drop(&mut self) { EXTERNAL.with(|value| value.set(self.0)); }
    }
    pub(crate) fn serialize<S: Serializer>(templates: &TemplateDfasByTerminal, serializer: S) -> Result<S::Ok, S::Error> {
        if EXTERNAL.with(Cell::get) { TemplateDfasByTerminal::new().serialize(serializer) }
        else { templates.serialize(serializer) }
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<TemplateDfasByTerminal, D::Error> {
        TemplateDfasByTerminal::deserialize(deserializer)
    }
}
