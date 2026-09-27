use super::*;
use glrmask_terminal_dwa::__private::terminal_dwa::{l1,scope::{BoundaryOwnership,ImmediateComponentId}};
use std::collections::BTreeMap;

fn with_prepared_cut(mut c:Constraint,v:&Vocab)->Constraint{
    let source=Arc::clone(&c.tokenizer);
    let mut words=v.iter().map(|(_,b)|b.to_vec()).collect::<Vec<_>>();words.sort();words.dedup();
    let prefix=PrefixObserver::prepare(&source,&words,Limits::default(),&mut Profile::default()).unwrap().certify(&source).unwrap();
    let index=UntrustedCompletionIndex::from_certified_prefix(&prefix).unwrap();
    c.boundary_completion_index=Some(Arc::new(PreparedCompletion::load(source,Arc::from(index.to_bytes().unwrap())).unwrap()));
    assert!(prepare_cut_component(&mut c,v).unwrap());c
}

#[test]
fn prepared_cut_roundtrip_pins_source_and_matches_actual_union_filter(){
    let v=Vocab::new(vec![(1,b"abc!".to_vec()),(2,b"bc!".to_vec()),(3,b"c!".to_vec()),(4,b"abc !".to_vec()),(5,b"c!".to_vec()),(6,b"!a".to_vec()),(7,b"a!".to_vec()),(8,b"ab".to_vec()),(9,Vec::new())]);
    let a=with_prepared_cut(Constraint::from_glrm_grammar(r#"start x; nt x ::= "abc" | "ab" | "a";"#,&v).unwrap(),&v);
    let b=with_prepared_cut(Constraint::from_glrm_grammar(r#"start x; nt x ::= "!" | " a" | "abc";"#,&v).unwrap(),&v);
    let mut components=Vec::new();
    for original in [a,b]{
        let wire=original.save();let loaded=Constraint::load_with_vocab(&wire,&v).unwrap();assert_eq!(loaded.save(),wire);
        let prepared=loaded.boundary_completion_index.as_ref().unwrap();assert!(prepared.cut.is_some());assert!(prepared.matches(loaded.composition_tokenizer()));
        assert!(saved_wire(&loaded).unwrap().starts_with(CUT_MAGIC));
        let mut clone=loaded.clone();assert!(Arc::get_mut(&mut clone.tokenizer).is_none(),"source must remain pinned");
        components.push(loaded);
    }
    let nt0=components[0].composition_tokenizer().num_terminals();let nt=nt0+components[1].composition_tokenizer().num_terminals();
    let(merged,offsets)=Tokenizer::disjoint_union_with_terminal_offsets(&[(components[0].composition_tokenizer(),0),(components[1].composition_tokenizer(),nt0)]);
    let spans=vec![PreparedSourceSpan::for_component(&components[0],offsets[0],0),PreparedSourceSpan::for_component(&components[1],offsets[1],nt0)];
    let ownership=BoundaryOwnership::flat(&[0,nt0],nt).unwrap();let flat=l1::build_flat_transition_table(&merged);
    let trans=crate::compiler::composition::boundary::token_support::PreparedSupportTransitions::new(&merged,&flat).unwrap();
    let rows=BTreeMap::new();
    for owner in 0..2{
        for seed_mode in 0..3{
            let mut initial=vec![false;merged.num_states()as usize];
            for q in 0..components[owner].tokenizer.num_states(){initial[(offsets[owner]+q)as usize]=seed_mode==0||(seed_mode==1&&q%2==0);}
            let expected=crate::compiler::composition::boundary::cut_support::crossing_support_with_transitions(&merged,&v,&initial,&ownership,ImmediateComponentId(owner as u32),&rows,None,None,None,Some(&trans)).unwrap();
            let actual=cut_support(&spans,&merged,&v,&initial,&ownership,ImmediateComponentId(owner as u32),&rows,None,None,None).unwrap();assert_eq!(actual.tokens,expected.tokens);
        }
    }
    let mut mixed=vec![false;merged.num_states()as usize];mixed[offsets[0]as usize]=true;mixed[offsets[1]as usize]=true;
    assert!(cut_support(&spans,&merged,&v,&mixed,&ownership,ImmediateComponentId(0),&rows,None,None,None).is_none());
    assert!(cut_support(&spans,&merged,&v,&[],&ownership,ImmediateComponentId(0),&rows,None,None,None).is_none());
    let mut wrong=components[0].clone();wrong.tokenizer=Arc::new(components[1].tokenizer.as_ref().clone());assert!(saved_wire(&wrong).is_none());assert!(PreparedSourceSpan::for_component(&wrong,1,0).is_none());
}

#[test]
fn prepared_cut_wire_is_bounded_and_rejects_corruption_and_other_source(){
    let v=Vocab::new(vec![(1,b"ab!".to_vec()),(2,b"b!".to_vec()),(3,b"xy!".to_vec())]);
    let c=with_prepared_cut(Constraint::from_glrm_grammar(r#"start x; nt x ::= "ab" | "xy";"#,&v).unwrap(),&v);
    let bytes=saved_wire(&c).unwrap();let(a,b)=split_cut_wire(bytes).unwrap();let reset=b.unwrap();assert!(!a.is_empty()&&!reset.is_empty());
    for end in 8..bytes.len(){assert!(split_cut_wire(&bytes[..end]).is_err(),"truncation{end}");}
    let mut trailing=bytes.to_vec();trailing.push(0);assert!(split_cut_wire(&trailing).is_err());
    let mut huge=bytes.to_vec();huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());assert!(split_cut_wire(&huge).is_err());
    let mut corrupt=bytes.to_vec();*corrupt.last_mut().unwrap()^=1;assert!(PreparedCompletion::load(Arc::clone(&c.tokenizer),Arc::from(corrupt)).is_err());
    let other=Constraint::from_glrm_grammar(r#"start x; nt x ::= "az" | "xy";"#,&v).unwrap();assert!(PreparedCompletion::load(Arc::clone(&other.tokenizer),Arc::from(bytes)).is_err());
    let legacy=PreparedCompletion::load(Arc::clone(&c.tokenizer),Arc::from(a)).unwrap();assert!(legacy.cut.is_none());assert_eq!(legacy.wire.as_ref(),a);
    fn send_sync<T:Send+Sync>(){}send_sync::<PreparedCompletion>();
}
