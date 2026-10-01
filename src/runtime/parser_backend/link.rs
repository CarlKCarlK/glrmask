//! Link already table-free components by relocating their exact programs.
//! Lexer execution, vocabulary walking and CALL/RETURN closure stay in the
//! common recursive runtime. No LR action/goto representation is constructed.
use std::{collections::{BTreeMap, BTreeSet}, sync::Arc};
use crate::{Error, Result, Vocab};
use crate::automata::lexer::Lexer;
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::{accumulator::TerminalsDisallowed, parser::ParserGSS};
use crate::runtime::{CommitTemplateDfas, Constraint, SegmentedBoundaryShard,
    SegmentedBoundaryShardBackend, SegmentedParserComponent, SegmentedParserLink,
    SpecialTokenTerminal, StaticDynamicOverlayMetadata};
use super::{TemplateParser, composition::TemplateComposition, embedding::TemplateEmbedding, link_program};

fn fail(message: impl Into<String>) -> Error { Error::Compilation(message.into()) }

fn identity() -> CommitTemplateDfas {
    let mut pop = DFA::new(); pop.set_accepting(0, true);
    CommitTemplateDfas { pop, read: DFA::new(), push: DFA::new(),
        pop_to_read: vec![], pop_to_push: vec![], read_to_push: vec![] }
}

// These shared bounded transformations preserve per-row DEFAULT priority,
// including explicit dead edges, before returning to the ordinary executor.
fn scope(program: &CommitTemplateDfas, alphabet: u32, offset: u32) -> Result<CommitTemplateDfas> {
    link_program::scoped_template(program, offset, alphabet).map_err(fail)
}

fn append_child_start(program: &CommitTemplateDfas, child_start: u32) -> Result<CommitTemplateDfas> {
    let mut nfa = link_program::action_nfa(program).map_err(fail)?;
    link_program::append_push(&mut nfa, child_start);
    link_program::compile(&[nfa]).map_err(fail)
}

fn with_nullable_return(program: &CommitTemplateDfas) -> Result<CommitTemplateDfas> {
    let nfa = link_program::action_nfa(program).map_err(fail)?;
    link_program::compile(&[nfa, link_program::nullable_return(0)]).map_err(fail)
}

/// Caller has chosen a dynamic boundary explicitly, or Auto. All supplied
/// components are already compiled; this function never calls a grammar/LR
/// compiler, including for previously saved components.
pub(crate) fn compose(parent: Constraint, children: &[(String, Arc<Constraint>)], vocab: &Vocab) -> Result<Constraint> {
    let parent_parser = parent.template_parser.as_ref().ok_or_else(|| fail("parent is not table-free"))?;
    let parent_embedding = parent_parser.embedding.as_ref()
        .ok_or_else(|| fail("parent artifact lacks a finite embedding relation"))?.clone();
    let mut components = vec![Arc::new(parent)];
    let mut slots = Vec::<Vec<u32>>::new();
    for (name, child) in children {
        let matching = components[0].late_grammar_slots.iter().filter(|slot| slot.name == *name)
            .map(|slot| slot.terminal_id).collect::<Vec<_>>();
        if matching.is_empty() { continue; }
        for slot in &matching {
            if !parent_embedding.entries.contains(slot) {
                return Err(fail(format!("slot {name:?} has no validated template CALL relation")));
            }
        }
        components.push(Arc::clone(child)); slots.push(matching);
    }
    if components.len() == 1 { return Ok((*components.remove(0)).clone()); }
    let bound_slots = slots.iter().flatten().copied().collect::<BTreeSet<_>>();
    if components[0].late_grammar_slots.iter().any(|slot| !bound_slots.contains(&slot.terminal_id))
        || components.iter().skip(1).any(|child| !child.late_grammar_slots.is_empty())
    { return Err(fail("template composition requires all external slots to be bound")); }
    let mut state_offsets = Vec::new(); let mut terminal_offsets = Vec::new();
    let mut tokenizer_offsets = Vec::new(); let mut names = Vec::new();
    let mut state_count = 0u32; let mut terminal_count = 0u32; let mut tokenizer_count = 0u32;
    for component in &components {
        let parser = component.template_parser.as_ref().ok_or_else(|| fail("composition contains an LR component"))?;
        if component.table.is_present() || parser.embedding.is_none() {
            return Err(fail("component lacks a finite table-free embedding relation; no LR reconstruction is permitted"));
        }
        if !component.token_bytes_match_vocab(vocab) { return Err(fail("component vocabulary differs from its parent")); }
        if component.terminal_display_names.len() != parser.terminal_count as usize {
            return Err(fail("component terminal names disagree with its parser coordinate"));
        }
        state_offsets.push(state_count); terminal_offsets.push(terminal_count); tokenizer_offsets.push(tokenizer_count);
        state_count = state_count.checked_add(parser.state_count).ok_or_else(|| fail("parser coordinate overflow"))?;
        terminal_count = terminal_count.checked_add(parser.terminal_count).ok_or_else(|| fail("terminal coordinate overflow"))?;
        let span = component.recursive_parser_layout().map_err(fail)?
            .map_or(component.tokenizer.num_states(), |layout| layout.total_tokenizer_states);
        tokenizer_count = tokenizer_count.checked_add(span).ok_or_else(|| fail("tokenizer coordinate overflow"))?;
        names.extend(component.terminal_display_names.iter().cloned());
    }
    if state_count as u64 * (terminal_count as u64 + 1) > 16_000_000 {
        return Err(fail("linked template certificates exceed 16 million symbol/terminal pairs"));
    }
    let mut outer = Vec::new(); let mut exact = Vec::new(); let mut controls = Vec::new();
    let mut cache = BTreeMap::<(usize, u32, u32), Arc<CommitTemplateDfas>>::new();
    for (index, component) in components.iter().enumerate() {
        let parser = component.template_parser.as_ref().unwrap(); let offset = state_offsets[index];
        let mut relocate = |program: &Arc<CommitTemplateDfas>| -> Result<Arc<CommitTemplateDfas>> {
            let key = (Arc::as_ptr(program) as usize, parser.state_count, offset);
            if let Some(program) = cache.get(&key) { return Ok(Arc::clone(program)); }
            let result = Arc::new(scope(program, parser.state_count, offset)?);
            cache.insert(key, Arc::clone(&result)); Ok(result)
        };
        for (terminal, program) in component.template_dfas_by_terminal.iter().enumerate() {
            let program = program.as_ref().ok_or_else(|| fail("missing component terminal program"))?;
            let local = if parser.composition.is_none() && (component.ignore_terminal == Some(terminal as u32)
                || parser.skip_terminals.contains(&(terminal as u32))) {
                Arc::new(scope(&identity(), parser.state_count, offset)?)
            } else { relocate(program)? };
            outer.push(Some(Arc::clone(&local)));
            if parser.composition.is_none() { exact.push(Some(local)); }
        }
        if let Some(composition) = &parser.composition {
            for (terminal, program) in composition.programs.iter().enumerate() {
                let program = relocate(program.as_ref().ok_or_else(|| fail("missing nested scoped program"))?)?;
                if terminal < composition.control_start as usize { exact.push(Some(program)); }
                else { controls.push(Some(program)); }
            }
        }
    }
    let mut links = Vec::new();
    for (child_index, slots) in slots.iter().enumerate() {
        let component = child_index + 1;
        let child_parser = components[component].template_parser.as_ref().unwrap();
        let embedding = child_parser.embedding.as_ref().unwrap();
        for &slot in slots {
            controls.push(Some(Arc::new(append_child_start(outer[slot as usize].as_ref().unwrap(), state_offsets[component])?)));
            links.push(SegmentedParserLink { parent_component: 0, slot_terminal: slot,
                child_component: component as u32, child_start: 0, return_pop: embedding.return_pop,
                child_start_nullable: embedding.nullable });
        }
        controls.push(Some(Arc::new(scope(&embedding.finish, child_parser.state_count, state_offsets[component])?)));
    }
    let control_start = u32::try_from(exact.len()).map_err(|_| fail("too many scoped terminals"))?;
    exact.extend(controls);
    let completion = scope(&components[0].template_parser.as_ref().unwrap().completion_template,
        components[0].parser_symbol_count(), 0)?;
    let mut parser = TemplateParser::compile(state_count, terminal_count, BTreeSet::new(), &outer, completion)?;
    parser.composition = Some(Arc::new(TemplateComposition::compile(state_count, control_start, &outer, exact)?));
    let initial = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let nullable = parent_embedding.nullable || parser.finished(&initial);
    let finish = if nullable && !parent_embedding.nullable {
        Arc::new(with_nullable_return(&parent_embedding.finish)?)
    } else { Arc::clone(&parent_embedding.finish) };
    parser.embedding = Some(Arc::new(TemplateEmbedding { nullable, return_pop: parent_embedding.return_pop,
        entries: BTreeSet::new(), finish }));
    let dynamic_vocab = crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab);
    let mut constraint = crate::dynamic_constraint::DynamicConstraint::from_template_runtime_parts_unfinalized(
        (*components[0].tokenizer).clone(), names, None, outer, Arc::new(parser), vocab, dynamic_vocab);
    // DynamicDirect never consumes a coordinator TSID quotient. Preserve the
    // ordinary recursive linker's one-class wire compatibility image, while
    // leaving all actual component lexer and parser coordinates unchanged.
    constraint.state_to_internal_tsid = vec![0];
    let mut specials = Vec::new();
    let mut wrappers = Vec::new(); let mut shards = Vec::new();
    for (index, component) in components.into_iter().enumerate() {
        for special in &component.special_token_terminals {
            if index == 0 && bound_slots.contains(&special.terminal_id) { continue; }
            specials.push(SpecialTokenTerminal { terminal_id: terminal_offsets[index] + special.terminal_id,
                token_id: special.token_id });
        }
        let shard = SegmentedBoundaryShard { start_component: index as u32,
            start_parser_states: crate::ds::bitset::BitSet::new(0), accepts_empty_stack: index == 0,
            candidate_tokens: None, mask_vocabulary: Default::default(), backend: SegmentedBoundaryShardBackend::DynamicDirect };
        shards.push(shard.clone());
        wrappers.push(SegmentedParserComponent { constraint: component, boundary: Some(shard),
            tokenizer_state_offset: tokenizer_offsets[index], terminal_offset: terminal_offsets[index],
            global_terminal_aliases: vec![], local_tsid_to_global_tsids: vec![],
            root_disallowed_terminal: None, global_to_local_parser_state: vec![] });
    }
    specials.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    specials.dedup(); constraint.special_token_terminals = specials;
    constraint.static_dynamic_overlay = Some(StaticDynamicOverlayMetadata {
        terminal_offsets, tokenizer_state_offsets: tokenizer_offsets, segmented_parser_components: wrappers,
        segmented_parser_links: links, segmented_mask_authoritative: true,
        segmented_boundary_shards: shards, ..Default::default()
    });
    constraint.validate_template_composition_layout().map_err(fail)?;
    constraint.rebuild_dynamic_runtime_caches();
    Ok(constraint)
}
