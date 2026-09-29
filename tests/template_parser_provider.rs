//! The parser is supplied as data; no LR grammar or callback implementation is
//! involved in constructing or replaying the parenthesis recognizer.
use std::cell::Cell;
use glrmask::{BuildOptions, Constraint, Grammar, ParserBackend, Vocab};
use glrmask::template_parser::{LexerDefinition, ParserDefinition, ParserProgram, ParserProvider,
    StackDfa, StackLabel, StackState, StackTemplate, StackTransition, TemplateBuildOptions, TerminalPattern};

#[test]
fn reconverging_push_and_pop_languages_remain_shared_across_reload() {
    for depth in [12usize,24,128] {
        let mut push=StackDfa{start:0,states:vec![StackState::default();depth+1]};
        let mut pop=StackDfa{start:0,states:vec![StackState::default();depth+2]};
        for i in 0..depth {
            for symbol in [1,2] {
                push.states[i].transitions.push(StackTransition{label:StackLabel::Symbol(symbol),target:i as u32+1});
            }
            pop.states[i].transitions.push(StackTransition{label:StackLabel::Symbol(0),target:depth as u32+1});
            pop.states[i].transitions.push(StackTransition{label:StackLabel::Default,target:i as u32+1});
        }
        push.states[depth].accepting=true;pop.states[depth].accepting=true;
        let parser=ParserProgram::new(ParserDefinition{stack_symbol_count:3,terminals:vec![
            StackTemplate{push,pop_to_push:vec![Some(0)],..StackTemplate::reject()},
            StackTemplate{pop,..StackTemplate::reject()},
        ],completion:StackTemplate::read_top_and_push([0],[])}).unwrap();
        let vocabulary=Vocab::new(["a","b","ab","ba","aa","bb"].iter().enumerate()
            .map(|(id,piece)|(id as u32,piece.as_bytes().to_vec())).collect());
        let lex=LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec()),TerminalPattern::literal(b"b".to_vec())]);
        let fresh=parser.compile(&lex,&vocabulary).unwrap();
        let bytes=fresh.save();
        assert!(bytes.len()<64_000+depth*64,"a linear graph must not serialize its exponentially many paths");
        let restored=Constraint::load(&bytes).unwrap();
        for constraint in [&fresh,&restored] {
            assert_eq!(constraint.parser_backend(),ParserBackend::TemplateDfa);
            let mut state=constraint.start();let mut mask=vec![0;constraint.mask_len()];
            state.fill_mask(&mut mask);assert_eq!(mask[0]&63,21);assert!(state.is_accepting());
            for (token,expected,complete) in [(0,31,false),(0,63,false),(1,31,false),(1,21,true)] {
                state.commit_token(token).unwrap();state.fill_mask(&mut mask);
                assert_eq!(mask[0]&63,expected,"depth={depth} token={token}");
                assert_eq!(state.is_accepting(),complete);
            }
            state.commit_token(2).unwrap();state.fill_mask(&mut mask);
            assert!(state.is_accepting());assert_eq!(mask[0]&63,21);
        }
    }
}

fn definition() -> ParserDefinition {
    ParserDefinition { stack_symbol_count: 2,
        terminals: vec![
            StackTemplate::read_top_and_push([0,1], [1]),
            StackTemplate::rewrite([StackLabel::Symbol(1)], []),
            StackTemplate::identity(),
        ],
        completion: StackTemplate::read_top_and_push([0], []),
    }
}
fn lexer() -> LexerDefinition {
    LexerDefinition::new(vec![TerminalPattern::literal(b"(".to_vec()),
        TerminalPattern::literal(b")".to_vec()), TerminalPattern::regex(r"[ \t\n]+")]).ignoring(2)
}
fn words() -> Vec<&'static str> {
    vec!["(",")","()","((","))","(()","())","()()"," "," \t\n","a","(a)"]
}
fn vocab() -> Vocab {
    let mut tokens=words().into_iter().enumerate().map(|(i,s)|(i as u32,s.as_bytes().to_vec())).collect::<Vec<_>>();
    tokens.push((64,Vec::new())); Vocab::new(tokens)
}
fn depth(bytes:&[u8], mut d:usize)->Option<usize> {
    for &b in bytes {match b {
        b'('=>d+=1,b')'=>d=d.checked_sub(1)?,b' '|b'\t'|b'\n'=>{},_=>return None,
    }} Some(d)
}
fn check_all_prefixes(c:&Constraint) {
    assert_eq!(c.parser_backend(),ParserBackend::TemplateDfa);
    for length in 0..=8 {
        for bits in 0..(1<<length) {
            let prefix=(0..length).map(|i|if bits & (1<<i)==0 {b'('} else {b')'}).collect::<Vec<_>>();
            let Some(d)=depth(&prefix,0) else {continue;};
            let mut state=c.start(); state.commit_bytes(&prefix).unwrap();
            assert_eq!(state.is_accepting(),d==0,"completion at {prefix:?}");
            let mut mask=vec![0;c.mask_len()]; state.fill_mask(&mut mask);
            for (i,word) in words().iter().enumerate() {
                let allowed=mask[i/32] & (1 << (i%32)) != 0;
                assert_eq!(allowed,depth(word.as_bytes(),d).is_some(),"prefix={prefix:?} token={word:?}");
            }
            assert_eq!(mask[2] & 1 != 0,d==0,"end token requires exact completion");
        }
    }
}

#[test]
fn data_only_recursive_parser_matches_independent_prefix_oracle_and_roundtrips() {
    let program=ParserProgram::new(definition()).unwrap();
    let v=vocab();
    let c=program.compile_with(&lexer(),&v,TemplateBuildOptions::default().end_tokens([64])).unwrap();
    check_all_prefixes(&c);
    let bytes=c.save();let restored=Constraint::load(bytes.clone()).unwrap();
    check_all_prefixes(&restored);assert_eq!(restored.save(),bytes);
    let external=c.save_with_external_vocab().unwrap();
    assert!(Constraint::load(external.clone()).is_err());
    let wrong=Vocab::new(vec![(0,b"different".to_vec()),(64,Vec::new())]);
    assert!(Constraint::load_with_vocab(external.clone(),&wrong).is_err());
    let restored=Constraint::load_with_vocab(external,&v).unwrap();check_all_prefixes(&restored);
    let mut s=c.start();assert!(s.commit_bytes(b")").is_err());assert!(s.is_rejected());
    let mut s=c.start();s.commit_token(64).unwrap();assert!(s.is_terminated());
}

#[test]
fn provider_is_called_only_when_validating_the_program() {
    struct Provider(Cell<usize>);
    impl ParserProvider for Provider {
        fn parser_definition(&self)->glrmask::Result<ParserDefinition> {
            self.0.set(self.0.get()+1);Ok(definition())
        }
    }
    let provider=Provider(Cell::new(0));
    let program=ParserProgram::from_provider(&provider).unwrap();
    let c=program.compile_with(&lexer(),&vocab(),TemplateBuildOptions::default().end_tokens([64])).unwrap();
    check_all_prefixes(&c);assert_eq!(provider.0.get(),1);
    fn send_sync<T:Send+Sync>(){}send_sync::<ParserProgram>();
    let data=serde_json::to_vec(&definition()).unwrap();
    let decoded:ParserDefinition=serde_json::from_slice(&data).unwrap();assert_eq!(decoded,definition());
    assert_eq!(ParserProgram::new(decoded).unwrap().terminal_count(),3);
}

#[test]
fn explicit_dead_edge_shadows_default_in_a_user_supplied_program() {
    let mut a=StackTemplate::rewrite([StackLabel::Default],[1]);
    let dead=a.pop.states.len() as u32;a.pop.states.push(StackState::default());
    a.pop.states[0].transitions.push(StackTransition{label:StackLabel::Symbol(0),target:dead});
    let b=StackTemplate::rewrite([StackLabel::Symbol(0)],[1]);
    let p=ParserProgram::new(ParserDefinition{stack_symbol_count:2,terminals:vec![a,b],completion:StackTemplate::identity()}).unwrap();
    let l=LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec()),TerminalPattern::literal(b"b".to_vec())]);
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ba".to_vec()),(3,b"ab".to_vec())]);
    let c=p.compile(&l,&v).unwrap();let mut s=c.start();let mut m=vec![0;c.mask_len()];s.fill_mask(&mut m);
    assert_eq!(m[0]&15,0b0110);s.commit_token(1).unwrap();s.fill_mask(&mut m);assert_eq!(m[0]&15,1);
}

#[test]
fn malformed_provider_graphs_fail_before_runtime_construction() {
    let mut d=definition();d.stack_symbol_count=0;assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.stack_symbol_count=u32::MAX;assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.stack_symbol_count=100_000_000;assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals.clear();assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].pop.states.clear();assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].read.start=99;assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].read.states[0].transitions[0].target=99;assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].read.states[0].transitions[0].label=StackLabel::Symbol(2);assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].read.states[0].transitions[0].label=StackLabel::Default;assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].push.states[0].transitions[0].label=StackLabel::Default;assert!(ParserProgram::new(d).is_err());
    let mut d=definition();let edge=d.terminals[0].read.states[0].transitions[0].clone();d.terminals[0].read.states[0].transitions.push(edge);assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].pop_to_read=vec![Some(99)];assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].pop_to_read=vec![None,None];assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.terminals[0].push.states[1].transitions.push(StackTransition{label:StackLabel::Symbol(0),target:0});assert!(ParserProgram::new(d).is_err());
    let mut d=definition();d.completion.read.states.push(StackState{accepting:false,transitions:vec![StackTransition{label:StackLabel::Symbol(0),target:2}]});assert!(ParserProgram::new(d).is_err(),"unreachable cycles also rejected");
}

#[test]
fn invalid_lexer_or_nonidentity_ignore_never_changes_the_requested_language() {
    let p=ParserProgram::new(definition()).unwrap();let v=vocab();
    let mut l=lexer();l.terminals.pop();assert!(p.compile(&l,&v).is_err());
    for pattern in [TerminalPattern::literal(Vec::new()),TerminalPattern::regex("a*"),TerminalPattern::regex("(")] {
        let mut l=lexer();l.terminals[0]=pattern;assert!(p.compile(&l,&v).is_err());
    }
    for ignore in [0,1,99] {let mut l=lexer();l.ignore_terminal=Some(ignore);assert!(p.compile(&l,&v).is_err());}
    let mut l=lexer();l.ignore_terminal=None;
    let c=p.compile_with(&l,&v,TemplateBuildOptions::default().end_tokens([64])).unwrap();check_all_prefixes(&c);
}

#[test]
fn builtin_parser_selection_is_per_constraint_and_survives_loading() {
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec())]);
    let grammar=Grammar::from_glrm(r#"start root; nt root ::= "a" "b"?;"#);
    let lr=grammar.compile(&v).unwrap();assert_eq!(lr.parser_backend(),ParserBackend::LrTable);
    let tp=grammar.compile_with(&v,BuildOptions::default().parser_backend(ParserBackend::TemplateDfa)).unwrap();
    assert_eq!(tp.parser_backend(),ParserBackend::TemplateDfa);
    let loaded=Constraint::load(tp.save()).unwrap();assert_eq!(loaded.parser_backend(),ParserBackend::TemplateDfa);
    for c in [&lr,&tp,&loaded] {let mut s=c.start();s.commit_bytes(b"ab").unwrap();assert!(s.is_accepting());}
    assert_eq!(lr.parser_backend(),ParserBackend::LrTable,"selecting another constraint must not change an existing backend");
}

#[test]
fn bounded_fast_build_selection_supports_all_builtin_source_frontends() {
    use glrmask::Optimization;
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),
        (3,b"\"".to_vec()),(4,b"\"ab\"".to_vec()),(5,b"x".to_vec())]);
    let schema=r#"{"enum":["a","ab"]}"#;
    let sources=[
        (Grammar::from_glrm(r#"start root; nt root ::= "a" "b"?;"#),"ab"),
        (Grammar::from_ebnf(r#"start ::= "a" "b"?"#),"ab"),
        (Grammar::from_lark(r#"start: "a" "b"?"#),"ab"),
        (Grammar::from_json_schema(schema),"\"ab\""),
    ];
    for (source,word) in sources {
        let c=source.compile_with(&v,BuildOptions::default().optimization(Optimization::FastBuild)
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_eq!(c.parser_backend(),ParserBackend::TemplateDfa);
        let lr=source.compile_with(&v,BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
        for parser in [&c,&Constraint::load(c.save()).unwrap()] {
            let mut candidate=parser.start();let mut reference=lr.start();
            for &byte in word.as_bytes() {
                assert_eq!(candidate.mask(),reference.mask());
                assert_eq!(candidate.is_accepting(),reference.is_accepting());
                candidate.commit_bytes(&[byte]).unwrap();reference.commit_bytes(&[byte]).unwrap();
            }
            assert!(candidate.is_accepting());assert_eq!(candidate.mask(),reference.mask());
        }
    }
}

#[test]
fn unsupported_template_composition_fails_explicitly_without_changing_default_lr() {
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"xa".to_vec()),(2,b"x".to_vec())]);
    let child=Grammar::from_glrm(r#"start root; nt root ::= "a";"#);
    let parent=Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = "x" child;"#);
    let bound=parent.clone().bind("child",&child).unwrap();
    let error=bound.compile_with(&v,BuildOptions::default().parser_backend(ParserBackend::TemplateDfa)).unwrap_err();
    assert!(error.to_string().contains("composition"));
    assert_eq!(bound.compile(&v).unwrap().parser_backend(),ParserBackend::LrTable);
    let module=parent.compile_unlinked(&v).unwrap().bind("child",&child.compile(&v).unwrap()).unwrap();
    assert!(module.link_with(BuildOptions::default().parser_backend(ParserBackend::TemplateDfa)).is_err());
    assert_eq!(module.link().unwrap().parser_backend(),ParserBackend::LrTable);
}

#[test]
fn ordinary_lr_parents_reject_table_free_children_without_panicking() {
    let v = Vocab::new(vec![(0, b"a".to_vec()), (1, b"xa".to_vec()), (2, b"x".to_vec())]);
    let child = Grammar::from_glrm(r#"start root; nt root ::= "a";"#)
        .compile_with(&v, BuildOptions::default().parser_backend(ParserBackend::TemplateDfa))
        .unwrap();
    let module = Grammar::from_glrm(
        r#"glrm 1; start root; extern grammar child; nt root = "x" child;"#,
    ).compile_unlinked(&v).unwrap();
    for child in [&child, &Constraint::load(child.save()).unwrap()] {
        for optimization in [glrmask::Optimization::Auto, glrmask::Optimization::FastRuntime, glrmask::Optimization::FastBuild] {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                module.bind("child", child).and_then(|bound| bound.link_with(
                    BuildOptions::default().optimization(optimization)))
            }));
            assert!(outcome.is_ok(), "unsupported composition must return an error, not touch an absent LR table");
            let error = outcome.unwrap().unwrap_err();
            assert!(error.to_string().contains("template") && error.to_string().contains("composition"), "{error}");
            assert_eq!(child.parser_backend(), ParserBackend::TemplateDfa);
        }
    }
    let lr_child = Grammar::from_glrm(r#"start root; nt root ::= "a";"#).compile(&v).unwrap();
    let linked = module.bind("child", &lr_child).unwrap().link().unwrap();
    let mut state = linked.start(); state.commit_bytes(b"xa").unwrap(); assert!(state.is_accepting());
}

#[test]
fn exponentially_many_custom_push_outputs_remain_runnable_after_roundtrip() {
    use glrmask::template_parser::StackDfa;
    // 25 states encode 16,777,216 distinct output stacks. This must be a graph
    // computation, not an enumeration of those words during commit or masking.
    let depth = 24;
    let mut push = StackDfa { start: 0, states: vec![StackState::default(); depth + 1] };
    for i in 0..depth {
        for symbol in [1, 2] {
            push.states[i].transitions.push(StackTransition {
                label: StackLabel::Symbol(symbol), target: i as u32 + 1,
            });
        }
    }
    push.states[depth].accepting = true;
    let program = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 3,
        terminals: vec![StackTemplate { push, pop_to_push: vec![Some(0)], ..StackTemplate::reject() }],
        completion: StackTemplate::identity(),
    }).unwrap();
    let lexer = LexerDefinition::new(vec![TerminalPattern::literal(b"a".to_vec())]);
    let v = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec()), (2, b"b".to_vec())]);
    let c = program.compile(&lexer, &v).unwrap();
    for parser in [&c, &Constraint::load(c.save()).unwrap()] {
        assert_eq!(parser.parser_backend(), ParserBackend::TemplateDfa);
        let mut state = parser.start(); let mut mask = vec![0; parser.mask_len()];
        for _ in 0..3 {
            state.fill_mask(&mut mask); assert_eq!(mask[0] & 7, 3);
            state.commit_token(0).unwrap(); assert!(state.is_accepting());
        }
    }
}
