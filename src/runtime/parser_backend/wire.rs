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
    composition: Option<CompositionSeed>,
    embedding: Option<super::embedding::TemplateEmbedding>,
    link_grammar: Option<Arc<super::link_grammar::LinkGrammar>>,
}

pub(super) struct CompositionSeed {
    control_start: u32,
    outer: Vec<super::scoped_program::ScopedProgram>,
    scoped: Vec<super::scoped_program::ScopedProgram>,
    completion: super::scoped_program::ScopedProgram,
}

pub(crate) fn encode(parser: &TemplateParser, templates: &TemplateDfasByTerminal) -> Vec<u8> {
    let bytes = compact::encode(parser, templates);
    if std::env::var_os("GLRMASK_VALIDATE_TEMPLATE_WIRE").is_some() {
        let decoded = compact::decode(&bytes).expect("newly encoded template programs must decode");
        assert_eq!(decoded.state_count, parser.state_count);
        assert_eq!(decoded.terminal_count, parser.terminal_count);
        assert_eq!(decoded.skip_terminals, parser.skip_terminals);
        assert_eq!(decoded.link_grammar, parser.link_grammar);
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
    compact::decode(bytes)
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
        // Component bodies are restored from the runtime section after this
        // parser section. Coordinate/layout validation runs after that restore,
        // before any derived caches or public runtime state can be published.
        // Core graph/alphabet validation and runtime-view preparation share the
        // standalone backend's complete validated preparation path.
        let (mut parser, runtime) = if let Some(composition) = self.composition {
            // Descriptors are authoritative aliases into the source dictionary.
            // Drop redundant raw outer records after validation rather than
            // retaining a second graph allocation for a reused component.
            constraint.template_dfas_by_terminal = composition.outer.iter()
                .map(|view| Some(Arc::clone(&view.source))).collect();
            let composition = super::composition::TemplateComposition::from_views(
                self.state_count, composition.control_start, composition.outer,
                composition.scoped, Some(composition.completion)).map_err(|error| error.to_string())?;
            (TemplateParser::from_composition(self.state_count, self.terminal_count, composition)
                .map_err(|error| error.to_string())?, Vec::new())
        } else { TemplateParser::compile_with_runtime(
            self.state_count, self.terminal_count, self.skip_terminals,
            &constraint.template_dfas_by_terminal, self.completion,
        ).map_err(|error| error.to_string())? };
        if let Some(embedding) = self.embedding {
            validate_alphabet(&embedding.finish, self.state_count)?;
            super::compile_domain(&embedding.finish).map_err(|error| error.to_string())?;
            parser.embedding = Some(Arc::new(embedding));
        }
        if let Some(grammar) = self.link_grammar {
            grammar.validate()?;
            if grammar.terminal_count != self.terminal_count {
                return Err("template grammar terminal coordinate disagrees with parser".into());
            }
            parser.link_grammar = Some(grammar);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_parser_section_rejects_duplicate_core_program_inventory() {
        let vocab = crate::Vocab::new(vec![(0, b"a".to_vec())]);
        let original = Constraint::from_glrm_grammar("start root; nt root ::= \"a\";", &vocab).unwrap();
        let wire = encode(original.template_parser.as_ref().unwrap(), &original.template_dfas_by_terminal);
        let mut duplicate = original.clone();
        let error = decode(&wire).unwrap().install(&mut duplicate).unwrap_err();
        assert!(error.contains("duplicate core parser programs"), "{error}");
        assert!(!duplicate.table.is_present());

        let mut section_owned = original.clone();
        section_owned.template_dfas_by_terminal.clear();
        decode(&wire).unwrap().install(&mut section_owned).unwrap();
        assert_eq!(section_owned.start().mask(), original.start().mask());
        let mut state = section_owned.start();
        state.commit_token(0).unwrap();
        assert!(state.is_accepting());
    }
}
