//! Link table-free components through shared programs and offset descriptors.
//! Lexer execution, vocabulary walking and CALL/RETURN closure stay in the
//! common recursive runtime. No LR action/goto representation is constructed.
use std::{collections::BTreeSet, sync::Arc};
use crate::{Error, Result, Vocab};
use crate::automata::lexer::Lexer;
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::{accumulator::TerminalsDisallowed, parser::ParserGSS};
use crate::runtime::{CommitTemplateDfas, Constraint, SegmentedBoundaryShard,
    SegmentedBoundaryShardBackend, SegmentedParserComponent, SegmentedParserLink,
    SpecialTokenTerminal, StaticDynamicOverlayMetadata};
use super::{TemplateParser, composition::TemplateComposition, scoped_program::ScopedProgram, embedding::TemplateEmbedding, link_program};

fn fail(message: impl Into<String>) -> Error { Error::Compilation(message.into()) }

fn identity() -> CommitTemplateDfas {
    let mut pop = DFA::new(); pop.set_accepting(0, true);
    CommitTemplateDfas { pop, read: DFA::new(), push: DFA::new(),
        pop_to_read: vec![], pop_to_push: vec![], read_to_push: vec![] }
}

fn with_nullable_return(program: &CommitTemplateDfas) -> Result<CommitTemplateDfas> {
    let nfa = link_program::action_nfa(program).map_err(fail)?;
    link_program::compile(&[nfa, link_program::nullable_return(0)]).map_err(fail)
}

/// Caller has chosen a dynamic boundary explicitly, or Auto. All supplied
/// components are already compiled; this function never calls a grammar/LR
/// compiler, including for previously saved components.
pub(crate) fn compose(mut parent: Constraint, children: &[(String, Arc<Constraint>)], vocab: &Vocab) -> Result<Constraint> {
    parent.materialize_composition_link_metadata_for_compilation().map_err(fail)?;
    // Internal callers can register slots after ordinary source compilation.
    // Remove their non-vocabulary sentinels before retaining the component:
    // its output domain and its cached token fragments must agree.
    if parent.sanitize_late_grammar_placeholder_token_domain() {
        parent.rebuild_runtime_caches();
    }
    let parent_parser = parent.template_parser.as_ref().ok_or_else(|| fail("parent is not table-free"))?;
    let parent_embedding = parent_parser.embedding.as_ref()
        .ok_or_else(|| fail("parent artifact lacks a finite embedding relation"))?.clone();
    let mut components = vec![Arc::new(parent)];
    let mut slots = Vec::<Vec<u32>>::new();
    let mut binding_names = Vec::new();
    for (name, child) in children {
        let matching = components[0].late_grammar_slots.iter().filter(|slot| slot.name == *name)
            .map(|slot| slot.terminal_id).collect::<Vec<_>>();
        if matching.is_empty() { continue; }
        for slot in &matching {
            if !parent_embedding.entries.contains(slot) {
                return Err(fail(format!("slot {name:?} has no validated template CALL relation")));
            }
        }
        let mut child = Arc::clone(child);
        if child.deferred_composition_metadata_blob.is_some() && !child.composition_link_metadata_materialized {
            Arc::make_mut(&mut child).materialize_composition_link_metadata_for_compilation().map_err(fail)?;
        }
        components.push(child); slots.push(matching); binding_names.push(name.clone());
    }
    if components.len() == 1 { return Ok((*components.remove(0)).clone()); }
    let bound_slots = slots.iter().flatten().copied().collect::<BTreeSet<_>>();
    // Preserve the ordinary linker's reusable outgoing interface-tail proof
    // while the semantic parent/child bindings are still explicit. A failed
    // bounded proof remains unavailable and later candidate queries widen.
    let tail_components = &components;
    let tail_bindings = slots.iter().enumerate().flat_map(|(index, slots)|
        slots.iter().map(move |&slot| (slot, tail_components[index + 1].as_ref())))
        .collect::<Vec<_>>();
    let boundary_tail = crate::compiler::boundary_tail::build_composition_boundary_tail_r2(
        &components[0], &tail_bindings, vocab,
    ).map(|proof| (proof.candidate_ids, proof.fixed_point_widened))
        .or_else(|_| crate::compiler::boundary_tail::build_composition_boundary_tail_r1(
            &components[0], &tail_bindings, vocab,
        ).map(|proof| (proof.candidate_ids, proof.fixed_point_widened))).ok();
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
    let mut outer = Vec::new(); let mut outer_views = Vec::new();
    let mut exact_views = Vec::new(); let mut control_views = Vec::new();
    for (index, component) in components.iter().enumerate() {
        let parser = component.template_parser.as_ref().unwrap(); let offset = state_offsets[index];
        let ignore_view = if parser.composition.is_none()
            && (component.ignore_terminal.is_some() || !parser.skip_terminals.is_empty()) {
            Some(ScopedProgram::prepare(Arc::new(identity()), parser.state_count)?)
        } else { None };
        for terminal in 0..component.template_dfas_by_terminal.len() {
            let view = if let Some(composition) = &parser.composition {
                composition.outer_views[terminal].relocated(offset)?
            } else if component.ignore_terminal == Some(terminal as u32)
                || parser.skip_terminals.contains(&(terminal as u32)) {
                ignore_view.as_ref().unwrap().relocated(offset)?
            } else { ScopedProgram::terminal(component, terminal).relocated(offset)? };
            outer.push(Some(Arc::clone(&view.source)));
            if parser.composition.is_none() { exact_views.push(view.clone()); }
            outer_views.push(view);
        }
        if let Some(composition) = &parser.composition {
            for (terminal, view) in composition.views.iter().enumerate() {
                let view = view.relocated(offset)?;
                if terminal < composition.control_start as usize { exact_views.push(view); }
                else { control_views.push(view); }
            }
        }
    }
    let mut links = Vec::new();
    for (child_index, slots) in slots.iter().enumerate() {
        let component = child_index + 1;
        let child_parser = components[component].template_parser.as_ref().unwrap();
        let embedding = child_parser.embedding.as_ref().unwrap();
        for &slot in slots {
            let mut call = outer_views[slot as usize].clone();
            call.append_push = Some(state_offsets[component]);
            control_views.push(call);
            links.push(SegmentedParserLink { parent_component: 0, slot_terminal: slot,
                child_component: component as u32, child_start: 0, return_pop: embedding.return_pop,
                child_start_nullable: embedding.nullable });
        }
        control_views.push(embedding.finish_view.relocated(state_offsets[component])?);
    }
    let mut remaining_slots = components[0].late_grammar_slots.iter()
        .filter(|slot| !bound_slots.contains(&slot.terminal_id)).cloned().collect::<Vec<_>>();
    for (component, child) in components.iter().enumerate().skip(1) {
        for slot in &child.late_grammar_slots {
            let mut slot = slot.clone();
            slot.terminal_id += terminal_offsets[component];
            slot.name = format!("{}.{}", binding_names[component - 1], slot.name);
            remaining_slots.push(slot);
        }
    }
    let control_start = u32::try_from(exact_views.len()).map_err(|_| fail("too many scoped terminals"))?;
    exact_views.extend(control_views);
    let parent_parser = components[0].template_parser.as_ref().unwrap();
    let completion = if let Some(composition) = &parent_parser.composition {
        composition.completion_view.as_ref().expect("native scoped completion").clone()
    } else { ScopedProgram { source: Arc::clone(&parent_parser.completion_template),
        domain: Arc::clone(&parent_parser.completion), fast: None, offset: 0,
        symbols: parent_parser.state_count, guard_owner: true, append_push: None } };
    let mut parser = TemplateParser::from_composition(state_count, terminal_count,
        TemplateComposition::from_views(state_count, control_start, outer_views, exact_views, Some(completion))?)?;
    let initial = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let nullable = parent_embedding.nullable || parser.finished(&initial);
    // Internal compiled composition can designate a validated entry as a
    // slot in a later binding. Preserve those proofs even when it has not
    // acquired a public slot name yet.
    let mut remaining_entries = parent_embedding.entries.iter().copied()
        .filter(|terminal| !bound_slots.contains(terminal)).collect::<BTreeSet<_>>();
    for (component, child) in components.iter().enumerate().skip(1) {
        remaining_entries.extend(child.template_parser.as_ref().unwrap().embedding.as_ref().unwrap()
            .entries.iter().map(|&terminal| terminal_offsets[component] + terminal));
    }
    let mut embedding = (*parent_embedding).clone();
    embedding.entries = remaining_entries.clone();
    if nullable && !parent_embedding.nullable {
        embedding = TemplateEmbedding::new(nullable, parent_embedding.return_pop,
            remaining_entries,
            Arc::new(with_nullable_return(&parent_embedding.finish)?), parent_embedding.finish_view.symbols)
            .map_err(fail)?;
    }
    parser.embedding = Some(Arc::new(embedding));
    parser.link_grammar = super::link_grammar::LinkGrammar::compose(&components, &terminal_offsets, &slots)
        .map_err(fail)?;
    if let Some(grammar) = parser.link_grammar.as_mut() {
        Arc::make_mut(grammar).root_nullable = nullable;
    }
    let dynamic_vocab = crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab);
    let mut constraint = crate::dynamic_constraint::DynamicConstraint::from_template_runtime_parts_unfinalized(
        (*components[0].tokenizer).clone(), names, None, outer, Arc::new(parser), vocab, dynamic_vocab);
    constraint.late_grammar_slots = remaining_slots;
    constraint.end_tokens = Arc::clone(&components[0].end_tokens);
    // DynamicDirect never consumes a coordinator TSID quotient. Preserve the
    // ordinary recursive linker's one-class wire compatibility image, while
    // leaving all actual component lexer and parser coordinates unchanged.
    constraint.state_to_internal_tsid = vec![0];
    let candidate_tokens = components.iter().map(|component| {
        crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(component, vocab)
            .map(|ids| ids.map(Arc::<[u32]>::from))
    }).collect::<std::result::Result<Vec<_>, _>>().map_err(fail)?;
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
            candidate_tokens: candidate_tokens[index].clone(), mask_vocabulary: Default::default(), backend: SegmentedBoundaryShardBackend::DynamicDirect };
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
    constraint.sanitize_late_grammar_placeholder_token_domain();
    constraint.validate_template_composition_layout().map_err(fail)?;
    constraint.rebuild_dynamic_runtime_caches();
    // Sparse/data-only components can have an interface-tail proof without
    // source grammar rules. Its reusable fingerprint is then unavailable;
    // retain the ordinary conservative candidate query for that component.
    if constraint.template_parser.as_ref().is_some_and(|parser| parser.link_grammar.is_some())
        && let Some((ids, widened)) = boundary_tail {
        crate::compiler::boundary_candidates::install_precomputed_boundary_candidate_ids(
            &mut constraint, vocab, &ids, widened,
        ).map_err(fail)?;
    }
    Ok(constraint)
}

#[cfg(test)]
mod candidate_tests {
    use super::*;
    use crate::{BuildOptions, Grammar, Optimization, ParserBackend};

    #[test]
    fn direct_table_free_link_carries_persisted_candidate_sets_through_reload() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ax".to_vec()),
            (3, b"x".to_vec()),
            (4, b"y".to_vec()),
            (5, b"xay".to_vec()),
            (6, b"xaby".to_vec()),
        ]);
        let child = Grammar::from_glrm(r#"glrm 1; start value; nt value = "a";"#)
            .compile_with(
                &vocab,
                BuildOptions::default().parser_backend(ParserBackend::TemplateDfa),
            )
            .unwrap();
        let expected = crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(
            &child,
            &vocab,
        )
        .unwrap()
        .expect("compiled table-free child should retain a candidate certificate");
        assert!(!expected.is_empty());
        assert!(expected.len() < vocab.len());

        let parent = Grammar::from_glrm(
            r#"glrm 1; extern grammar child; start root; nt root = "x" child "y";"#,
        )
        .compile_unlinked(&vocab)
        .unwrap();
        let linked = parent
            .bind("child", &child)
            .unwrap()
            .link_with(
                BuildOptions::default()
                    .optimization(Optimization::FastBuild)
                    .parser_backend(ParserBackend::TemplateDfa),
            )
            .unwrap();

        let check = |constraint: &Constraint| {
            let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
            let boundary = overlay.segmented_parser_components[1]
                .boundary
                .as_ref()
                .unwrap();
            assert!(matches!(boundary.backend, SegmentedBoundaryShardBackend::DynamicDirect));
            assert_eq!(boundary.candidate_tokens.as_deref(), Some(expected.as_slice()));
        };
        check(&linked);
        check(&Constraint::load(&linked.save()).unwrap());
    }
}
