//! Lossless compact encoding of the same split programs used by the runtime.
//! This changes neither graph numbering nor transition choice: explicit dead
//! edges which shadow DEFAULT remain present, and omitted trailing link arrays
//! retain their exact length. No runtime representation is specialized here.
use super::{CommitTemplateDfas, DFA, Input, ParserSeed, TemplateParser, validate_dimensions};
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label, negative_to_positive_label};
use crate::runtime::artifact::TemplateDfasByTerminal;
use std::collections::{BTreeSet, BTreeMap};
use super::super::scoped_program::ScopedProgram;
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"TPR7";
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
    out.extend_from_slice(MAGIC);
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
    put(&mut out, u32::from(parser.composition.is_some()));
    if let Some(composition) = &parser.composition {
        put(&mut out, composition.control_start);
        // All outer and exact aliases reference the same immutable graph.
        // Only FINISH/new tiny controls absent from the outer inventory add
        // graph records; offsets and CALL outputs are independent descriptors.
        let mut dictionary = BTreeMap::<usize, u32>::new();
        for (id, source) in templates.iter().map(|program| program.as_ref().unwrap())
            .chain(std::iter::once(&parser.completion_template)).enumerate() {
            dictionary.entry(Arc::as_ptr(source) as usize).or_insert(id as u32);
        }
        let mut extras = Vec::new();
        for view in composition.outer_views.iter().chain(&composition.views)
            .chain(composition.completion_view.iter()) {
            let key = Arc::as_ptr(&view.source) as usize;
            if !dictionary.contains_key(&key) {
                let id = templates.len() as u32 + 1 + extras.len() as u32;
                dictionary.insert(key, id); extras.push(Arc::clone(&view.source));
            }
        }
        count(&mut out, extras.len());
        for source in &extras { program(&mut out, source, parser.state_count); }
        let write_view = |out: &mut Vec<u8>, view: &ScopedProgram| {
            put(out, dictionary[&(Arc::as_ptr(&view.source) as usize)]);
            put(out, view.offset); put(out, view.symbols);
            put(out, u32::from(view.guard_owner));
            put(out, view.append_push.map_or(0, |symbol| symbol + 1));
        };
        for view in &composition.outer_views { write_view(&mut out, view); }
        count(&mut out, composition.views.len());
        for view in &composition.views { write_view(&mut out, view); }
        write_view(&mut out, composition.completion_view.as_ref().expect("scoped completion"));
    }
    put(&mut out, u32::from(parser.embedding.is_some()));
    if let Some(embedding) = &parser.embedding {
        put(&mut out, u32::from(embedding.nullable));
        put(&mut out, embedding.return_pop);
        count(&mut out, embedding.entries.len());
        for &terminal in &embedding.entries { put(&mut out, terminal); }
        put(&mut out, embedding.finish_view.symbols);
        program(&mut out, &embedding.finish, embedding.finish_view.symbols);
    }
    put(&mut out, u32::from(parser.link_grammar.is_some()));
    if let Some(grammar) = &parser.link_grammar {
        let bytes = bincode::serialize(grammar.as_ref()).expect("composition grammar metadata serializes");
        count(&mut out, bytes.len());
        out.extend_from_slice(&bytes);
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
        graph.states.reserve_exact(count);
        for _ in 0..count {
            let flags=self.var()?; let edges=(flags>>1) as usize;
            if edges>self.bytes.len().saturating_sub(self.offset)/2 {
                return Err("compact template edge count exceeds remaining section".into());
            }
            budget.edges=budget.edges.checked_add(edges).ok_or("template edge count overflow")?;
            if budget.edges>MAX_DECODED_EDGES { return Err("template graph exceeds decoded-edge budget".into()); }
            let id=graph.add_state(); graph.states[id as usize].is_accepting=flags&1!=0;
            let mut previous:Option<u32>=None;
            // Small rows retain their one-leaf insertion path. Larger rows
            // have already-ordered unique labels; collect validated pairs
            // once and let the ordered-map bulk constructor build the tree.
            let mut ordered = (edges > 8).then(|| Vec::with_capacity(edges));
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
                if let Some(entries) = &mut ordered { entries.push((label, target)); }
                else { graph.states[id as usize].transitions.insert(label,target); }
                previous=Some(symbol);
            }
            if let Some(entries) = ordered {
                graph.states[id as usize].transitions = entries.into_iter().collect();
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
    fn scoped_view(&mut self, alphabet: u32, sources: &[Arc<CommitTemplateDfas>],
        prepared: &mut BTreeMap<(u32, u32), ScopedProgram>) -> Result<ScopedProgram, String> {
        let source = self.var()?; let offset = self.var()?; let symbols = self.var()?;
        let guard_owner = match self.var()? { 0 => false, 1 => true, _ => return Err("invalid scope owner guard".into()) };
        let appended = self.var()?;
        let append_push = appended.checked_sub(1);
        if symbols == 0 || offset.checked_add(symbols).is_none_or(|end| end > alphabet)
            || append_push.is_some_and(|symbol| symbol >= alphabet) {
            return Err("scoped program coordinate outside parser alphabet".into());
        }
        let source_program = sources.get(source as usize).ok_or("scoped source reference outside dictionary")?;
        let key = (source, symbols);
        if !prepared.contains_key(&key) {
            prepared.insert(key, ScopedProgram::prepare(Arc::clone(source_program), symbols)
                .map_err(|error| error.to_string())?);
        }
        let mut view = prepared[&key].clone();
        view.offset = offset; view.guard_owner = guard_owner; view.append_push = append_push;
        view.validate_coordinate(alphabet).map_err(|error| error.to_string())?;
        Ok(view)
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

#[cfg(test)]
mod bulk_decode_tests {
    use super::*;

    fn scoped_source() -> Arc<CommitTemplateDfas> {
        let mut pop = DFA::new();
        pop.set_accepting(0, true);
        Arc::new(CommitTemplateDfas { pop, read: DFA::new(), push: DFA::new(),
            pop_to_read: vec![], pop_to_push: vec![], read_to_push: vec![] })
    }

    #[test]
    fn scoped_descriptors_reject_foreign_references_ranges_and_call_symbols() {
        let sources = [scoped_source()];
        for fields in [
            [1, 3, 2, 1, 0], // missing dictionary entry
            [0, 4, 2, 1, 0], // range exceeds global alphabet
            [0, u32::MAX, 2, 1, 0], // overflowing range
            [0, 0, 0, 1, 0], // empty source alphabet
            [0, 3, 2, 2, 0], // invalid owner flag
            [0, 3, 2, 1, 6], // appended CALL symbol outside alphabet
        ] {
            let mut bytes = Vec::new();
            for field in fields { put(&mut bytes, field); }
            let mut input = Input { bytes: &bytes, offset: 0 };
            assert!(input.scoped_view(5, &sources, &mut BTreeMap::new()).is_err(),
                "accepted malformed scoped descriptor {fields:?}");
        }
    }

    #[test]
    fn scoped_descriptor_aliases_share_source_and_prepared_domain() {
        let sources = [scoped_source()];
        let mut bytes = Vec::new();
        for fields in [[0, 0, 2, 1, 0], [0, 3, 2, 1, 5]] {
            for field in fields { put(&mut bytes, field); }
        }
        let mut input = Input { bytes: &bytes, offset: 0 };
        let mut prepared = BTreeMap::new();
        let first = input.scoped_view(5, &sources, &mut prepared).unwrap();
        let second = input.scoped_view(5, &sources, &mut prepared).unwrap();
        assert!(Arc::ptr_eq(&first.source, &second.source));
        assert!(Arc::ptr_eq(&first.domain, &second.domain));
        assert_eq!(prepared.len(), 1);
        assert_eq!(second.append_push, Some(4));
        assert_eq!(second.offset, 3);
        assert_eq!(bincode::serialize(&*sources[0]).unwrap(),
            bincode::serialize(&*scoped_source()).unwrap());
    }

    #[test]
    fn compact_rows_preserve_exact_graphs_across_bulk_boundary() {
        for phase in [Phase::Pop, Phase::Read, Phase::Push] {
            for width in [0, 1, 4, 8, 9, 11, 16, 64, 129] {
                let alphabet = 256;
                let mut graph = DFA::new();
                let yes = graph.add_state();
                let dead = graph.add_state();
                graph.set_accepting(yes, true);
                for symbol in 0..width {
                    let label = match phase {
                        Phase::Push => encode_negative_label(symbol),
                        _ => symbol as i32,
                    };
                    graph.add_transition(0, label, if symbol % 3 == 0 { dead } else { yes });
                }
                if matches!(phase, Phase::Pop) { graph.add_transition(0, DEFAULT_LABEL, yes); }
                let mut bytes = Vec::new();
                dfa(&mut bytes, &graph, phase, alphabet);
                let mut input = Input { bytes: &bytes, offset: 0 };
                let decoded = input.compact_dfa(phase, alphabet, &mut Budget::default()).unwrap();
                assert_eq!(decoded, graph);
                assert_eq!(input.offset, bytes.len());
                let mut roundtrip = Vec::new();
                dfa(&mut roundtrip, &decoded, phase, alphabet);
                assert_eq!(roundtrip, bytes);
                for end in 0..bytes.len() {
                    let mut truncated = Input { bytes: &bytes[..end], offset: 0 };
                    assert!(truncated.compact_dfa(phase, alphabet, &mut Budget::default()).is_err());
                }
            }
        }
    }
}
#[derive(Default)]
struct Budget {states:usize,edges:usize}

pub(super) fn decode(bytes:&[u8])->Result<ParserSeed,String> {
    let mut input=Input{bytes,offset:0};
    let magic = input.take::<4>()?;
    if &magic != MAGIC {return Err("unsupported template parser format; recompile the pre-release artifact".into());}
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
    let composed = match input.var()? { 0 => false, 1 => true,
        _ => return Err("invalid composition flag".into()) };
    let composition = if composed {
        let control_start = input.var()?;
        let source_count = input.variable_count(6)?;
        let mut sources = templates.iter().map(|p| Arc::clone(p.as_ref().unwrap())).collect::<Vec<_>>();
        sources.push(Arc::new(completion.clone()));
        for _ in 0..source_count { sources.push(Arc::new(input.compact_program(state_count, &mut budget)?)); }
        let mut prepared = BTreeMap::<(u32, u32), ScopedProgram>::new();
        let mut outer = Vec::with_capacity(terminal_count as usize);
        for _ in 0..terminal_count { outer.push(input.scoped_view(state_count, &sources, &mut prepared)?); }
        let count = input.variable_count(5)?;
        if control_start as usize > count { return Err("composition control offset exceeds inventory".into()); }
        let total = terminal_count.checked_add(u32::try_from(count).map_err(|_| "too many scoped relations")?)
            .filter(|&total| total != u32::MAX).ok_or("scoped terminal coordinate overflow")?;
        validate_dimensions(state_count, total)?;
        if state_count as u64 * ((count - control_start as usize) as u64 * 4 + 24) > super::MAX_CERTIFICATE_BYTES {
            return Err("composition control certificates exceed load budget".into());
        }
        let mut scoped = Vec::with_capacity(count);
        for _ in 0..count { scoped.push(input.scoped_view(state_count, &sources, &mut prepared)?); }
        let completion = input.scoped_view(state_count, &sources, &mut prepared)?;
        Some(super::CompositionSeed { control_start, outer, scoped, completion })
    } else { None };
    let has_embedding = match input.var()? { 0 => false, 1 => true,
        _ => return Err("invalid embedding flag".into()) };
    let embedding = if has_embedding {
        let nullable = match input.var()? { 0 => false, 1 => true,
            _ => return Err("invalid embedding nullability flag".into()) };
        let return_pop = input.var()?;
        if !matches!(return_pop, 1 | 2) { return Err("invalid embedding return convention".into()); }
        let count = input.variable_count(1)?;
        if count > terminal_count as usize { return Err("embedding slot inventory exceeds terminals".into()); }
        let mut entries = BTreeSet::new();
        let mut previous = None;
        for _ in 0..count {
            let terminal = input.var()?;
            if terminal >= terminal_count || previous.is_some_and(|old| old >= terminal) {
                return Err("invalid or unordered embedding slot".into());
            }
            entries.insert(terminal); previous = Some(terminal);
        }
        let symbols = input.var()?;
        if symbols == 0 || symbols > state_count { return Err("invalid FINISH component alphabet".into()); }
        let finish = Arc::new(input.compact_program(symbols, &mut budget)?);
        Some(super::super::embedding::TemplateEmbedding::new(nullable, return_pop, entries, finish, symbols)?)
    } else { None };
    let link_grammar = match input.var()? {
        0 => None,
        1 => {
            use bincode::Options;
            let count = input.variable_count(1)?;
            if count > 32 * 1024 * 1024 { return Err("template grammar metadata exceeds load budget".into()); }
            let end = input.offset.checked_add(count).ok_or("grammar metadata offset overflow")?;
            let grammar: super::super::link_grammar::LinkGrammar = bincode::DefaultOptions::new()
                .with_fixint_encoding().with_limit(count as u64).reject_trailing_bytes()
                .deserialize(&input.bytes[input.offset..end]).map_err(|e| e.to_string())?;
            input.offset = end;
            grammar.validate()?;
            if grammar.terminal_count != terminal_count { return Err("template grammar terminal coordinate mismatch".into()); }
            if grammar.stack_effects().is_some_and(|effects| effects.states != state_count) {
                return Err("compiler stack-effect parser coordinate mismatch".into());
            }
            Some(Arc::new(grammar))
        }
        _ => return Err("invalid template grammar metadata flag".into()),
    };
    if input.offset!=bytes.len() {return Err("trailing compact parser bytes".into());}
    // Cycle/phase validation is shared with domain derivation during install.
    // No materialization path can expose these graphs before it succeeds.
    Ok(ParserSeed {state_count,terminal_count,skip_terminals,completion,programs:Some(templates),composition,embedding,link_grammar})
}
