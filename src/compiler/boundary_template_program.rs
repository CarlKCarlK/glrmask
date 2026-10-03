//! Shared finite control-gap assembly for static parser queries.
//!
//! The lexical graph chooses ordinary terminals and their correlated token
//! weights. A bounded control certificate supplies the zero-width gap depth.
//! No LR table, parser action row, or grammar production is consulted here.
use super::*;

fn minimize_symbolic_class_boundary(graph: DWA) -> DWA {
    if crate::compiler::boundary_env::enabled("GLRMASK_BOUNDARY_FINITE_ATOM_MINIMIZE")
        && let Some((candidate, profile)) = crate::compiler::boundary_bit_minimize::minimize_finite_final_atoms(
            &graph, crate::compiler::glr::labels::DEFAULT_LABEL) {
        if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
            eprintln!("[glrmask/profile][symbolic_class_finite_minimize] selected=true profile={profile:?}");
        }
        candidate
    } else {
        if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
            eprintln!("[glrmask/profile][symbolic_class_finite_minimize] selected=false");
        }
        crate::automata::weighted::minimize_acyclic::minimize_acyclic_owned(graph)
    }
}

// A normalized DEFAULT covers the finite construction alphabet: concrete
// stack symbols plus opaque class identities. Preserve that exact substitution
// with an existing finite complement and the missing class identities, rather
// than allocating one literal edge per concrete symbol. Source row scope and
// dead guards are unchanged. A descriptor collision retains literal expansion.
fn restore_finite_pop_defaults(
    graph:&mut DWA,classes:&glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
)->Option<glrmask_parser_dwa::__private::pop_classes::PopLabelClasses> {
    let concrete=classes.symbol_count();let alphabet=concrete.checked_add(classes.len() as u32)?;
    let default_label=crate::compiler::glr::labels::DEFAULT_LABEL;
    let mut published=classes.clone();let mut expanded_edges=0usize;
    for row in graph.states_mut() {
        let mut dense=row.transitions.entries().map(|(label,target,weight)|(label,(target,weight.clone())))
            .collect::<BTreeMap<_,_>>();
        let default=dense.remove(&default_label).filter(|edge|!edge.1.is_empty());
        if dense.keys().any(|&label|label<0 || label as u32>=alphabet) {return None;}
        // Keep the previous hypothetical literal-expansion resource contract.
        expanded_edges=expanded_edges.checked_add(if default.is_some(){alphabet as usize}else{dense.len()})?;
        if expanded_edges>4_000_000 {return None;}
        let restore=|label:i32|if label>=concrete as i32 {default_label-1-(label-concrete as i32)}else{label};
        let mut mapped=dense.iter().map(|(&label,edge)|(restore(label),edge.clone())).collect::<BTreeMap<_,_>>();
        if let Some(default)=default {
            let complement=published.intern_complement(dense.keys().copied()
                .filter(|&label|label<concrete as i32).map(|label|label as u32)).ok()?;
            if complement.is_some_and(|label|mapped.contains_key(&label)) {
                // Two opaque branches can substitute to the same descriptor
                // with different targets. DWA rows cannot store both branches.
                for label in 0..alphabet as i32 {dense.entry(label).or_insert_with(||default.clone());}
                mapped=dense.into_iter().map(|(label,edge)|(restore(label),edge)).collect();
            } else {
                if let Some(label)=complement {mapped.insert(label,default.clone());}
                for index in 0..classes.len() {
                    if !dense.contains_key(&(concrete as i32+index as i32)) {
                        mapped.insert(default_label-1-index as i32,default.clone());
                    }
                }
            }
        }
        row.transitions=mapped.into_iter().collect();
    }
    Some(published)
}

#[cfg(test)]
fn compile_compact_classed(
    templates:&BTreeMap<u32,NWA>,admissions:Option<&BTreeMap<u32,NWA>>,controls:&[u32],
    max_controls:u32,lexical:&DWA,classes:&glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
    context:Option<&crate::compiler::stages::parser_dwa::FiniteParserReadSupport>,
)->Option<SignedShardOutput> {
    compile_shared_classed(templates,admissions,controls,Some(max_controls),lexical,classes,context)
}

/// Immutable construction-only alphabet conversion shared by the lexical
/// shards of one link. It contains no coefficients or assembled parser query.
pub(crate) struct PreparedClassedPrograms {
    ids: BTreeMap<(bool,u32),usize>,
    templates: Vec<NWA>,
}

pub(crate) fn prepare_classed_programs(
    templates:&BTreeMap<u32,NWA>,admissions:Option<&BTreeMap<u32,NWA>>,
    classes:&glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
)->Option<PreparedClassedPrograms> {
    use crate::automata::weighted_u32::nwa::NWAState;
    let mut ids=BTreeMap::new(); let mut dense_templates=Vec::new();
    for (admission,inventory) in std::iter::once((false,templates)).chain(admissions.map(|a|(true,a))) {
        for (&terminal,source) in inventory {
            let mut states=Vec::with_capacity(source.states().len());
            for row in source.states() {
                let mut mapped=BTreeMap::new();
                for (&label,edges) in &row.transitions {
                    let label=if label>=classes.symbol_count() as i32 {
                        let index=crate::compiler::glr::labels::DEFAULT_LABEL.checked_sub(1)?.checked_sub(label)?;
                        if index<0 || index as usize>=classes.len() {return None;}
                        classes.symbol_count() as i32+index
                    } else {label};
                    mapped.insert(label,edges.clone());
                }
                states.push(NWAState { final_weight:row.final_weight.clone(),
                    transitions:mapped,epsilons:row.epsilons.clone() });
            }
            let dense=NWA::from_parts(states,source.start_states().to_vec());
            ids.insert((admission,terminal),dense_templates.len()); dense_templates.push(dense);
        }
    }
    Some(PreparedClassedPrograms { ids,templates:dense_templates })
}

fn compile_shared_classed(
    templates:&BTreeMap<u32,NWA>,admissions:Option<&BTreeMap<u32,NWA>>,controls:&[u32],
    max_controls:Option<u32>,lexical:&DWA,classes:&glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
    context:Option<&crate::compiler::stages::parser_dwa::FiniteParserReadSupport>,
)->Option<SignedShardOutput> {
    compile_shared_classed_prepared(templates,admissions,controls,max_controls,lexical,classes,context,None)
}

fn compile_shared_classed_prepared(
    templates:&BTreeMap<u32,NWA>,admissions:Option<&BTreeMap<u32,NWA>>,controls:&[u32],
    max_controls:Option<u32>,lexical:&DWA,classes:&glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
    context:Option<&crate::compiler::stages::parser_dwa::FiniteParserReadSupport>,
    prepared:Option<&PreparedClassedPrograms>,
)->Option<SignedShardOutput> {
    use crate::compiler::stages::parser_dwa::{FiniteTemplateInstance,FiniteTemplateProgram,
        normalize_finite_template_program_with_pop_classes};
    use crate::compiler::boundary_weight_codec::FaithfulWeightQuotient;
    use crate::compiler::boundary_bit_minimize::{FiniteAtomDecoder,minimize_native_decoded};
    let started=Instant::now();let depths=match max_controls {Some(depth)=>depth.checked_add(1)? as usize,None=>2};
    if max_controls.is_some() && depths>16 {return None;}
    let ports=lexical.states().len().checked_mul(depths)?;
    let alphabet=classes.symbol_count().checked_add(classes.len() as u32)?;
    let owned;
    let prepared=if let Some(prepared)=prepared {prepared} else {
        owned=prepare_classed_programs(templates,admissions,classes)?; &owned
    };
    let ids=&prepared.ids; let dense_templates=&prepared.templates;
    let mut coefficients=vec![Weight::all()];let mut indices=rustc_hash::FxHashMap::default();
    indices.insert(coefficients[0].ptr_key(),0usize);
    let mut weight_id=|weight:&Weight| {
        *indices.entry(weight.ptr_key()).or_insert_with(|| {
            let id=coefficients.len();coefficients.push(weight.clone());id
        })
    };
    let mut finals=vec![None;ports+usize::from(admissions.is_some())];
    // Lexical finals may be the identity ALL even though every path to them
    // carries a finite token mask. Use the existing exact path-support proof
    // for publication; all source coefficients retain their global ghosts.
    let domain=crate::compiler::constraint_compose::accepted_weight_support(lexical);
    if admissions.is_some() {finals[ports]=Some(weight_id(&domain));}
    let mut instances=Vec::new();let mut estimated_states=finals.len();let mut estimated_edges=0usize;
    let mut add=|template:usize,coefficient:usize,continuation:u32,entries:std::ops::Range<u32>|->Option<()> {
        estimated_states=estimated_states.checked_add(dense_templates[template].states().len())?;
        estimated_edges=estimated_edges.checked_add(dense_templates[template].num_transitions())?;
        instances.push(FiniteTemplateInstance{template,coefficient,continuation,entries});Some(())
    };
    for (q,row) in lexical.states().iter().enumerate() {
        if let Some(weight)=row.final_weight.as_ref().filter(|w|!w.is_empty()) {finals[q*depths]=Some(weight_id(weight));}
        for (label,target,weight) in row.transitions.entries() {
            if label<0 || target as usize>=lexical.states().len() {return None;}
            if weight.is_empty() {continue;}
            let entries=(q*depths) as u32..((q+1)*depths) as u32;
            if admissions.is_some() && lexical.states()[target as usize].final_weight.as_ref()
                .is_some_and(|final_weight|weight.is_subset(final_weight)) {
                add(*ids.get(&(true,label as u32))?,weight_id(weight),ports as u32,entries)?;
            } else {
                add(*ids.get(&(false,label as u32))?,weight_id(weight),target*depths as u32,entries)?;
            }
        }
    }
    drop(weight_id);
    for q in 0..lexical.states().len() {
        if max_controls.is_none() {
            for &control in controls {add(*ids.get(&(false,control))?,0,(q*depths+1) as u32,
                (q*depths) as u32..((q+1)*depths) as u32)?;}
        } else {for depth in 0..depths-1 {for &control in controls {
            let source=(q*depths+depth) as u32;
            add(*ids.get(&(false,control))?,0,source+1,source..source+1)?;
        }}}
    }
    drop(add);
    let profiling=std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    if profiling {eprintln!("[glrmask/profile][native_compact_template_program_contract] templates={} instances={} logical_states={estimated_states} logical_edges={estimated_edges} coefficients={} lexical_states={} depth={max_controls:?} domain_full={} domain_points={}",dense_templates.len(),instances.len(),coefficients.len(),lexical.states().len(),domain.is_full(),domain.range_entries().map(|(lo,hi,tokens)|(u128::from(hi)-u128::from(lo)+1)*tokens.len() as u128).sum::<u128>());}
    if !(1..=200_000).contains(&estimated_states) || coefficients.len()>2048 {
        if profiling {eprintln!("[glrmask/profile][native_compact_template_program] selected=false stage=instance_budget");} return None;
    }
    if max_controls.is_none() {
        let refs=dense_templates.iter().collect::<Vec<_>>();let starts=[lexical.start_state()*depths as u32];
        let program=FiniteTemplateProgram{templates:&refs,coefficients:&coefficients,port_finals:&finals,
            starts:&starts,instances:&instances};
        let prepare_ms=started.elapsed().as_secs_f64()*1000.0;
        let (parser_dwa,profile)=crate::compiler::stages::parser_dwa::normalize_control_star_template_program_with_pop_classes(
            &program,classes,minimize_symbolic_class_boundary).map_err(|error| {
                if profiling {eprintln!("[glrmask/profile][native_shared_template_program] mode=control_star selected=false error={error}");}
            }).ok()?;
        if profiling {eprintln!("[glrmask/profile][native_shared_template_program] mode=control_star selected=true context=false templates={} instances={} profile={profile:?}",refs.len(),instances.len());}
        return Some(SignedShardOutput{parser_dwa,templates_ms:0.0,compose_ms:prepare_ms+profile.assembly_ms,
            resolve_ms:profile.resolve_ms,normalize_ms:profile.normalize_ms,signed_states:profile.input_states,
            signed_transitions:profile.input_edges,terms:templates.len().saturating_sub(controls.len())});
    }
    let declined=|stage| {if profiling {eprintln!("[glrmask/profile][native_compact_template_program] selected=false stage={stage}");}None};
    let quotient=FaithfulWeightQuotient::new(&domain,&coefficients).or_else(||declined("faithful_weight_codec"))?;
    let decoder=FiniteAtomDecoder::new_checked(quotient.atoms().to_vec()).or_else(||{
        if profiling {eprintln!("[glrmask/profile][native_compact_template_program] selected=false stage=atom_decoder");}None
    })?;
    let encoded=coefficients.iter().map(|weight|quotient.encode(weight)).collect::<Option<Vec<_>>>()?;
    let refs=dense_templates.iter().collect::<Vec<_>>();let starts=[lexical.start_state()*depths as u32];
    let program=FiniteTemplateProgram{templates:&refs,coefficients:&encoded,port_finals:&finals,
        starts:&starts,instances:&instances};
    let prepare_ms=started.elapsed().as_secs_f64()*1000.0;
    let (native,profile)=normalize_finite_template_program_with_pop_classes(&program,classes,quotient.rows(),context,true).or_else(|| {
        if profiling {eprintln!("[glrmask/profile][native_compact_template_program] selected=false stage=shared_normalization");}None
    })?;
    let (mut symbolic,min_profile)=minimize_native_decoded(&native,&decoder,crate::compiler::glr::labels::DEFAULT_LABEL).or_else(|| {
        if profiling {eprintln!("[glrmask/profile][native_compact_template_program] selected=false stage=decoded_minimization");}None
    })?;
    let publication_classes=restore_finite_pop_defaults(&mut symbolic,classes)?;
    let phase=Instant::now();
    let parser_dwa=publication_classes.compile_positive_dwa(symbolic,8_000_000).ok()?;
    if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
        eprintln!("[glrmask/profile][native_compact_template_program] selected=true context={} templates={} instances={} coefficients={} prepare_ms={prepare_ms:.3} class_ms={:.3} total_ms={:.3} native={profile:?} minimize={min_profile:?}",
            context.is_some(),refs.len(),instances.len(),coefficients.len(),phase.elapsed().as_secs_f64()*1000.0,started.elapsed().as_secs_f64()*1000.0);
    }
    Some(SignedShardOutput{parser_dwa,templates_ms:0.0,compose_ms:prepare_ms,
        resolve_ms:profile.resolve_ms,normalize_ms:started.elapsed().as_secs_f64()*1000.0-prepare_ms-profile.resolve_ms,
        signed_states:estimated_states,signed_transitions:estimated_edges,terms:templates.len().saturating_sub(controls.len())})
}

pub(super) fn assemble(
    templates: &Templates,
    controls: &[u32],
    max_controls_per_gap: u32,
    lexical: &DWA,
    state_budget: Option<usize>,
) -> Result<(NWA, usize, usize), String> {
    assemble_impl(&templates.by_terminal_nwa, controls, Some(max_controls_per_gap), lexical, state_budget, None)
}

fn assemble_impl(
    templates: &BTreeMap<u32, NWA>,
    controls: &[u32],
    bounded_depth: Option<u32>,
    lexical: &DWA,
    state_budget: Option<usize>,
    admissions: Option<&BTreeMap<u32,NWA>>,
) -> Result<(NWA, usize, usize), String> {
    let depths = match bounded_depth {
        Some(depth) => usize::try_from(depth).ok().and_then(|n| n.checked_add(1))
            .ok_or("static template control depth overflow")?,
        // Exact C*: only the after-consuming port 0 may publish a lexical
        // final. The after-control port 1 loops through arbitrary controls.
        // Two ports do not bound the number of zero-width parser advances.
        None => 2,
    };
    let port_count = lexical.states().len().checked_mul(depths)
        .filter(|&n| n <= u32::MAX as usize).ok_or("static template port count overflow")?;
    let mut arena = NWA::new(0, 0);
    let mut ordinary_states = 0usize;
    let mut control_states = 0usize;
    let check_growth = |current: usize, extra: usize| -> Result<(), String> {
        let next = current.checked_add(extra).filter(|&n| n <= u32::MAX as usize)
            .ok_or("static template graph coordinate overflow")?;
        if state_budget.is_some_and(|limit| next > limit) {
            return Err("static template graph exceeds representation budget; no relation was truncated".into());
        }
        Ok(())
    };
    check_growth(0, port_count)?;
    let mut ports = Vec::with_capacity(port_count);
    for (index, row) in lexical.states().iter().enumerate() {
        for _ in 0..depths { ports.push(arena.add_state()); }
        // Token admission observes no trailing control closure: a lexical
        // final can publish only immediately after its consuming transition.
        if let Some(weight) = &row.final_weight {
            if !weight.is_empty() { arena.set_final_weight(ports[index * depths], weight.clone()); }
        }
    }
    let start = *ports.get(lexical.start_state() as usize * depths)
        .ok_or("static template query has no start port")?;
    arena.set_start_states(vec![start]);
    let sink = if admissions.is_some() {
        check_growth(arena.states().len(),1)?;
        let sink=arena.add_state();
        arena.set_final_weight(sink,Weight::union_all(lexical.states().iter().filter_map(|row|row.final_weight.as_ref())));
        Some(sink)
    } else {None};
    for (index, row) in lexical.states().iter().enumerate() {
        for (terminal, target, weight) in row.transitions.entries() {
            if terminal < 0 { return Err("static lexical query contains a negative terminal label".into()); }
            if weight.is_empty() { continue; }
            if let (Some(admissions),Some(sink))=(admissions,sink) {
                if lexical.states().get(target as usize).and_then(|row|row.final_weight.as_ref())
                    .is_some_and(|final_weight|weight.is_subset(final_weight)) {
                    let template=admissions.get(&(terminal as u32))
                        .ok_or_else(||format!("static query has no admission for terminal {terminal}"))?;
                    check_growth(arena.states().len(),template.states().len())?;
                    let body=append_weighted_fragment(&mut arena,template,weight,sink)?;
                    ordinary_states+=template.states().len();
                    for depth in 0..depths {for &entry in &body.start_states {
                        arena.add_epsilon(ports[index*depths+depth],entry,Weight::all());
                    }}
                    continue;
                }
            }
            let template = templates.get(&(terminal as u32))
                .ok_or_else(|| format!("static query has no template for terminal {terminal}"))?;
            let continuation = *ports.get(target as usize * depths)
                .ok_or("static lexical query edge leaves its graph")?;
            check_growth(arena.states().len(), template.states().len())?;
            let body = append_weighted_fragment(&mut arena, template, weight, continuation)?;
            ordinary_states += template.states().len();
            for depth in 0..depths {
                for &entry in &body.start_states {
                    arena.add_epsilon(ports[index * depths + depth], entry, Weight::all());
                }
            }
        }
    }
    for index in 0..lexical.states().len() {
        let control_layers = if bounded_depth.is_some() { depths - 1 } else { 1 };
        for depth in 0..control_layers {
            for &control in controls {
                let template = templates.get(&control)
                    .ok_or_else(|| format!("static query has no template for control {control}"))?;
                check_growth(arena.states().len(), template.states().len())?;
                let body = append_weighted_fragment(&mut arena, template, &Weight::all(),
                    ports[index * depths + depth + 1])?;
                control_states += template.states().len();
                for &entry in &body.start_states {
                    arena.add_epsilon(ports[index * depths + depth], entry, Weight::all());
                    if bounded_depth.is_none() {
                        arena.add_epsilon(ports[index * depths + 1], entry, Weight::all());
                    }
                }
            }
        }
    }
    Ok((arena, ordinary_states, control_states))
}

/// Retired native expanded-builder entry. Every production native caller
/// must use the shared template-program constructor below.
pub(crate) fn compile(
    _templates: &Templates,
    _controls: &[u32],
    _certificate: &ClosureCertificate,
    _lexical: &DWA,
    _symbol_count: u32,
) -> Result<SignedShardOutput, String> {
    panic!("obsolete expanded native parser-DWA builder; use the shared template-program constructor");
}

/// Retired native control-star builder entry. Cyclic controls use the exact
/// fixed-point mode of the shared template-program constructor below.
pub(crate) fn compile_saturated(
    _templates: &Templates,
    _controls: &[u32],
    _lexical: &DWA,
    _symbol_count: u32,
) -> Result<SignedShardOutput, String> {
    panic!("obsolete expanded native control-star builder; use the shared template-program constructor");
}

pub(crate) fn compile_classed(
    templates: &BTreeMap<u32, NWA>,
    controls: &[u32],
    certificate: Option<&ClosureCertificate>,
    lexical: &DWA,
    classes: &glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
) -> Result<SignedShardOutput, String> {
    assert!(lexical.is_acyclic(),"shared template constructor requires an acyclic lexical query");
    let depth=certificate.map(|c|c.max_controls_per_gap).or_else(||controls.is_empty().then_some(0));
    Ok(compile_shared_classed(templates,None,controls,depth,lexical,classes,None)
        .expect("shared template constructor refused its checked contract; redundant expanded-builder fallback is disabled"))
}

pub(crate) fn compile_classed_with_admissions(
    templates:&BTreeMap<u32,NWA>,admissions:&BTreeMap<u32,NWA>,controls:&[u32],
    certificate:Option<&ClosureCertificate>,lexical:&DWA,
    classes:&glrmask_parser_dwa::__private::pop_classes::PopLabelClasses,
    read_context:Option<&crate::compiler::stages::parser_dwa::FiniteParserReadSupport>,
    prepared:Option<&PreparedClassedPrograms>,
)->Result<SignedShardOutput,String> {
    assert!(lexical.is_acyclic(),"shared template constructor requires an acyclic lexical query");
    let depth=certificate.map(|c|c.max_controls_per_gap).or_else(||controls.is_empty().then_some(0));
    Ok(compile_shared_classed_prepared(templates,Some(admissions),controls,depth,lexical,classes,read_context,prepared)
        .expect("shared template constructor refused its checked contract; redundant expanded-builder fallback is disabled"))
}

#[cfg(test)]
mod symbolic_class_tests {
    use super::*;

    fn token_weight(tokens: &[u32]) -> Weight {
        Weight::from_uniform(0..=0,tokens.iter().copied().collect())
    }

    #[test]
    fn finite_default_remapping_matches_full_literal_expansion_and_descriptor_collisions() {
        use glrmask_parser_dwa::__private::pop_classes::PopLabelClasses;
        let default=crate::compiler::glr::labels::DEFAULT_LABEL;
        for collision in [false,true] {
            let mut classes=PopLabelClasses::new(4).unwrap();
            let first=if collision {classes.intern_complement([0]).unwrap().unwrap()}
                else {classes.intern_scoped_complement(1..4,[2]).unwrap().unwrap()};
            classes.intern_scoped_complement(0..4,[0,1,3]).unwrap().unwrap();
            let alphabet=classes.symbol_count()+classes.len() as u32;
            let mut source=DWA::from_parts(vec![Default::default();4],0);
            source.add_transition(0,0,3,Weight::empty());
            source.add_transition(0,4,2,token_weight(&[0]));
            source.add_transition(0,default,1,token_weight(&[1]));
            source.set_final_weight(1,token_weight(&[1]));
            source.set_final_weight(2,token_weight(&[0]));
            source.set_final_weight(3,token_weight(&[0,1]));
            source.add_transition(1,1,3,Weight::empty());
            source.add_transition(1,default,0,Weight::all());
            let mut reference=source.clone();
            for row in reference.states_mut() {
                let mut dense=row.transitions.entries().map(|(label,target,w)|(label,(target,w.clone())))
                    .collect::<BTreeMap<_,_>>();
                if let Some(edge)=dense.remove(&default) {
                    for label in 0..alphabet as i32 {dense.entry(label).or_insert_with(||edge.clone());}
                }
                row.transitions=dense.into_iter().map(|(label,edge)|
                    (if label>=4{default-1-(label-4)}else{label},edge)).collect();
            }
            let reference=classes.compile_positive_dwa(reference,10000).unwrap();
            let published=restore_finite_pop_defaults(&mut source,&classes).unwrap();
            if collision {assert!(source.states()[0].transitions.contains_key(&first));}
            let candidate=published.compile_positive_dwa(source,10000).unwrap();
            let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                &reference,&candidate,4,10000).unwrap();
            assert!(comparison.difference.is_none(),"collision={collision}: {:?}",comparison.difference);
        }
    }

    #[test]
    fn whole_edge_admission_keeps_partial_overlap_as_one_complete_transfer() {
        let weight=token_weight(&[0,1]);
        let mut lexical=DWA::new(1,1);
        let end=lexical.add_state();
        lexical.add_transition(0,0,end,weight.clone());
        lexical.set_final_weight(end,token_weight(&[0]));
        let mut full=NWA::from_parts(vec![Default::default();3],vec![0]);
        full.add_transition(0,0,1,Weight::all());
        full.add_transition(1,crate::compiler::glr::labels::encode_negative_label(1),2,Weight::all());
        full.set_final_weight(2,Weight::all());
        let mut checker=NWA::from_parts(vec![Default::default();2],vec![0]);
        checker.add_transition(0,2,1,Weight::all());checker.set_final_weight(1,Weight::all());
        let admissions=BTreeMap::from([(0,checker)]);
        let templates=BTreeMap::from([(0,full)]);
        let (mixed,ordinary,_) = assemble_impl(&templates,&[],Some(0),&lexical,None,Some(&admissions)).unwrap();
        assert_eq!(ordinary,3,"partial overlap must instantiate exactly one full transfer");
        assert!(mixed.states().iter().any(|row|row.transitions.keys().any(|&label|label<0)));
        assert!(!mixed.states().iter().any(|row|row.transitions.contains_key(&2)),"no partial checker branch");
        assert!(mixed.states().iter().flat_map(|row|row.transitions.values().flatten()).any(|(_,w)|w==&weight),"whole coefficient is retained");
        let classes=glrmask_parser_dwa::__private::pop_classes::PopLabelClasses::new(4).unwrap();
        let candidate=compile_compact_classed(&templates,Some(&admissions),&[],0,&lexical,&classes,None).unwrap();
        let mut reference=mixed;
        glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(&mut reference,&classes).unwrap();
        let reference=classes.compile_positive(reference,10000).unwrap();
        let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &reference,&candidate.parser_dwa,4,10000).unwrap();
        assert!(comparison.difference.is_none(),"compact partial-overlap branch changed masks: {:?}",comparison.difference);
        lexical.set_final_weight(end,weight);
        // There is deliberately no full template: a whole ending edge needs
        // only its input checker, even when its destination has other paths.
        lexical.add_transition(end,0,end,token_weight(&[0]));
        let (ending,ordinary,_) = assemble_impl(&BTreeMap::new(),&[],Some(0),&lexical,None,Some(&admissions)).unwrap();
        assert_eq!(ordinary,4);
        assert!(!ending.states().iter().any(|row|row.transitions.keys().any(|&label|label<0)));
    }

    #[test]
    fn native_compact_constructor_handles_small_whole_ending_edges_without_expanded_fallback() {
        let classes=glrmask_parser_dwa::__private::pop_classes::PopLabelClasses::new(4).unwrap();
        let mut checker=NWA::from_parts(vec![Default::default();2],vec![0]);
        checker.add_transition(0,1,1,Weight::all());checker.set_final_weight(1,Weight::all());
        let weight=token_weight(&[0]);let mut lexical=DWA::new(1,0);let end=lexical.add_state();
        lexical.add_transition(0,0,end,weight);lexical.set_final_weight(end,Weight::all());
        let admissions=BTreeMap::from([(0,checker)]);
        let output=compile_compact_classed(&BTreeMap::new(),Some(&admissions),&[],0,&lexical,&classes,None)
            .expect("small class-aware programs use the same shared constructor");
        let (raw,_,_)=assemble_impl(&BTreeMap::new(),&[],Some(0),&lexical,None,Some(&admissions)).unwrap();
        let reference=classes.compile_positive(raw,10000).unwrap();
        let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &reference,&output.parser_dwa,4,10000).unwrap();
        assert!(comparison.difference.is_none(),"ALL-final path support changed masks: {:?}",comparison.difference);
        assert!(output.signed_states<4096);
        assert_eq!(output.parser_dwa.states()[output.parser_dwa.start_state() as usize].final_weight,None);
    }

    #[test]
    fn native_compact_constructor_is_selected_and_matches_expanded_general_class_solver() {
        let mut classes=glrmask_parser_dwa::__private::pop_classes::PopLabelClasses::new(4).unwrap();
        let class=classes.intern_scoped_complement(1..3,[2]).unwrap().unwrap();
        let mut template=NWA::from_parts(vec![Default::default();4098],vec![0]);
        for q in 0..4096 {template.add_transition(q,class,q+1,Weight::all());}
        template.add_transition(4096,crate::compiler::glr::labels::encode_negative_label(3),4097,Weight::all());
        template.set_final_weight(4097,Weight::all());
        let templates=BTreeMap::from([(0,template)]);
        let weight=token_weight(&[0]);let mut lexical=DWA::new(1,0);let end=lexical.add_state();
        lexical.add_transition(0,0,end,weight.clone());lexical.set_final_weight(end,weight);
        let candidate=compile_compact_classed(&templates,None,&[],0,&lexical,&classes,None)
            .expect("the existing compact finite template constructor must actually select");
        let (mut raw,_,_)=assemble_impl(&templates,&[],Some(0),&lexical,None,None).unwrap();
        glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(&mut raw,&classes).unwrap();
        let reference=classes.compile_positive(raw,100000).unwrap();
        let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &reference,&candidate.parser_dwa,4,100000).unwrap();
        assert!(comparison.difference.is_none(),"{:?}",comparison.difference);
        assert!(comparison.product_states>=4097);
    }

    #[test]
    fn shared_control_star_program_matches_raw_full_transfer_without_depth_truncation() {
        let mut classes=glrmask_parser_dwa::__private::pop_classes::PopLabelClasses::new(4).unwrap();
        let local=classes.intern_scoped_complement(1..3,[]).unwrap().unwrap();
        let mut control=NWA::from_parts(vec![Default::default();3],vec![0]);
        control.add_transition(0,local,1,Weight::all());
        control.add_transition(1,crate::compiler::glr::labels::encode_negative_label(1),2,Weight::all());
        control.set_final_weight(2,Weight::all());
        let mut terminal=NWA::from_parts(vec![Default::default();3],vec![0]);
        terminal.add_transition(0,1,1,Weight::all());
        terminal.add_transition(1,crate::compiler::glr::labels::encode_negative_label(3),2,Weight::all());
        terminal.set_final_weight(2,Weight::all());
        let mut checker=NWA::from_parts(vec![Default::default();2],vec![0]);
        checker.add_transition(0,1,1,Weight::all());checker.set_final_weight(1,Weight::all());
        let templates=BTreeMap::from([(0,terminal),(1,control)]);let admissions=BTreeMap::from([(0,checker)]);
        let mut lexical=DWA::new(1,0);let end=lexical.add_state();
        lexical.add_transition(0,0,end,token_weight(&[0]));lexical.set_final_weight(end,Weight::all());
        let candidate=compile_shared_classed(&templates,Some(&admissions),&[1],None,&lexical,&classes,None).unwrap();
        let (mut reference,_,_)=assemble_impl(&templates,&[1],None,&lexical,None,None).unwrap();
        assert!(!reference.is_acyclic(),"this is a genuine control cycle, not a finite unfolding");
        glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(&mut reference,&classes).unwrap();
        let reference=classes.compile_positive(reference,10000).unwrap();
        let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &reference,&candidate.parser_dwa,4,10000).unwrap();
        assert!(comparison.difference.is_none(),"{:?}",comparison.difference);
        assert!(comparison.product_states>1);
    }

    #[test]
    fn finite_prefix_quotient_preserves_overlapping_scoped_class_masks_for_every_stack_word() {
        let mut classes = glrmask_parser_dwa::__private::pop_classes::PopLabelClasses::new(7).unwrap();
        let local = classes.intern_scoped_complement(3..6, [4]).unwrap().unwrap();
        let mut graph = NWA::from_parts(vec![Default::default(); 3], vec![0]);
        let a = Weight::from_uniform(0..=0, std::iter::once(0u32).collect());
        let b = Weight::from_uniform(0..=0, std::iter::once(1u32).collect());
        graph.add_transition(0, local, 1, a.clone());
        graph.add_transition(0, 5, 2, b.clone());
        graph.set_final_weight(1, a); graph.set_final_weight(2, b);
        let reference = classes.compile_positive(graph.clone(), 1000).unwrap();
        let candidate = classes.compile_positive_with_minimizer(graph, 1000, minimize_symbolic_class_boundary).unwrap();
        let comparison = glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
            &reference, &candidate, 7, 10000).unwrap();
        assert!(comparison.difference.is_none(), "{:?}", comparison.difference);
        assert!(comparison.product_states > 0);
        assert!(candidate.states()[candidate.start_state() as usize].final_weight.is_none());
    }
    #[test]
    fn shared_program_inventory_preserves_independent_shard_coefficients() {
        let mut classes=glrmask_parser_dwa::__private::pop_classes::PopLabelClasses::new(4).unwrap();
        let class=classes.intern_scoped_complement(0..4,[2]).unwrap().unwrap();
        let mut template=NWA::from_parts(vec![Default::default();3],vec![0]);
        template.add_transition(0,class,1,Weight::all());
        template.add_transition(1,crate::compiler::glr::labels::encode_negative_label(1),2,Weight::all());
        template.set_final_weight(2,Weight::all());
        let mut admission=NWA::from_parts(vec![Default::default();2],vec![0]);
        admission.add_transition(0,class,1,Weight::all()); admission.set_final_weight(1,Weight::all());
        let templates=BTreeMap::from([(0,template)]); let admissions=BTreeMap::from([(0,admission)]);
        let prepared=prepare_classed_programs(&templates,Some(&admissions),&classes).unwrap();
        let snapshot=||prepared.templates.iter().map(|graph|
            (graph.start_states().to_vec(),graph.states().to_vec())).collect::<Vec<_>>();
        let inventory_before=snapshot();
        for tokens in [&[0][..],&[1][..],&[0,1][..]] {
            let weight=token_weight(tokens); let mut lexical=DWA::new(1,0);
            let middle=lexical.add_state(); let end=lexical.add_state();
            lexical.add_transition(0,0,middle,weight.clone());
            lexical.add_transition(middle,0,end,weight); lexical.set_final_weight(end,Weight::all());
            let expected=compile_shared_classed(&templates,Some(&admissions),&[],Some(0),&lexical,&classes,None).unwrap();
            let actual=compile_shared_classed_prepared(&templates,Some(&admissions),&[],Some(0),&lexical,&classes,None,Some(&prepared)).unwrap();
            assert_eq!(bincode::serialize(&expected.parser_dwa).unwrap(),bincode::serialize(&actual.parser_dwa).unwrap());
            let (mut expanded,_,_)=assemble_impl(&templates,&[],Some(0),&lexical,None,Some(&admissions)).unwrap();
            glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(&mut expanded,&classes).unwrap();
            let reference=classes.compile_positive(expanded,10000).unwrap();
            let comparison=glrmask_parser_dwa::__private::parser_equivalence::compare_parser_mask_prefix_languages(
                &reference,&actual.parser_dwa,4,10000).unwrap();
            assert!(comparison.difference.is_none(),"{tokens:?}: {:?}",comparison.difference);
        }
        assert_eq!(inventory_before,snapshot());
    }

}
