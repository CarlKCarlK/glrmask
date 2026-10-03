//! Persistence must not reintroduce an executable LR table into a mandatory
//! template backend. These tests use the same shared mask/commit engines before
//! and after loading, including invalid prefixes and exact EOF completion.
use glrmask::{Constraint, DynamicConstraint, Grammar, Vocab};
use glrmask::__private::{DynamicConstraintExt, parser_backend_report, dynamic_parser_backend_report};

fn vocab() -> Vocab {
    Vocab::new(["a", "b", ",", "(", ")", "[", "]", "{", "}", "\"", ":", " ", "1", "2", "true", "null", "\"a\"", "\"b\"", "\"x\":", "\\", "u", "\\u", "ab", "aa", "[1", "]}", "\n"]
        .into_iter().enumerate().map(|(id, token)| (id as u32, token.as_bytes().to_vec())).collect())
}
fn sequences() -> Vec<Vec<u32>> {
    let mut all = vec![vec![], vec![0], vec![1], vec![0,2,0], vec![3,0,4],
        vec![3,3,0,4,4], vec![5,12,2,13,6], vec![7,18,12,8], vec![7,16,10,17,8],
        vec![9,0,9], vec![9,19,20,12,9], vec![14], vec![15], vec![7,9,0,9,10,5,12,6,8]];
    for a in 0..18 { for b in 0..18 { all.push(vec![a,b]); } }
    all
}
fn compare_static(left: &Constraint, right: &Constraint) {
    assert_eq!(left.mask_len(), right.mask_len());
    for tokens in sequences() {
        let mut a=left.start(); let mut b=right.start();
        let mut am=vec![0;left.mask_len()]; let mut bm=am.clone();
        for token in tokens.into_iter().map(Some).chain([None]) {
            a.fill_mask(&mut am); b.fill_mask(&mut bm);
            assert_eq!(am,bm,"static mask at next token {token:?}");
            assert_eq!(a.is_accepting(), b.is_accepting());
            let Some(token)=token else { break; };
            if am[token as usize/32] & (1 << (token%32)) == 0 { break; }
            a.commit_token(token).unwrap(); b.commit_token(token).unwrap();
        }
    }
}
fn compare_dynamic(left: &DynamicConstraint, right: &DynamicConstraint) {
    assert_eq!(left.mask_len(), right.mask_len());
    for tokens in sequences() {
        let mut a=left.start(); let mut b=right.start();
        let mut am=vec![0;left.mask_len()]; let mut bm=am.clone();
        for token in tokens.into_iter().map(Some).chain([None]) {
            a.fill_mask(&mut am); b.fill_mask(&mut bm);
            assert_eq!(am,bm,"dynamic mask at next token {token:?}");
            assert_eq!(a.is_accepting(), b.is_accepting());
            let Some(token)=token else { break; };
            if am[token as usize/32] & (1 << (token%32)) == 0 { break; }
            a.commit_token(token).unwrap(); b.commit_token(token).unwrap();
        }
    }
}

const GRAMMARS: &[&str] = &[
    r#"start start; t A ::= "a"; nt start ::= A ("," A)*;"#,
    r#"start start; t A ::= "a"; nt start ::= A | "(" start ")";"#,
    r#"start start; ignore WS; t WS ::= " "+; t A ::= "a"+; t B ::= "a"+ "b"?; nt start ::= A | B;"#,
];

#[test]
fn static_template_artifacts_roundtrip_without_lr_storage() {
    let v=vocab();
    for source in GRAMMARS {
        let reference=Constraint::compile(Grammar::glrm(source),&v).unwrap();
        let native_bytes=reference.save();
        assert_eq!(u16::from_le_bytes(native_bytes[8..10].try_into().unwrap()),37);
        compare_static(&reference,&Constraint::load(native_bytes).unwrap());
        let template=reference.clone();
        let bytes=template.save();
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()),37);
        let loaded=Constraint::load(bytes.clone()).unwrap();
        assert_eq!(parser_backend_report(&loaded)["lr_table_present"],false);
        compare_static(&reference,&loaded);
        compare_static(&template,&loaded);
        assert_eq!(loaded.save(),bytes,"unchanged loaded template artifact must re-save verbatim");
    }
}

#[test]
fn o2_template_artifacts_roundtrip_without_lr_storage() {
    let v=vocab();
    for source in GRAMMARS {
        let reference=DynamicConstraint::compile_with_vocab_partition(Grammar::glrm(source),&v).unwrap();
        let template=reference.clone();
        let bytes=template.save();
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()),21);
        let body=&bytes[30..]; // first native alternative after its length descriptor
        assert_eq!(u16::from_le_bytes(body[8..10].try_into().unwrap()),37);
        assert_eq!(&body[parser_range(body)][..4],b"TPR7");
        let loaded=DynamicConstraint::load(&bytes).unwrap();
        for report in dynamic_parser_backend_report(&loaded).as_array().unwrap() {
            assert_eq!(report["lr_table_present"],false);
        }
        compare_dynamic(&reference,&loaded);
        compare_dynamic(&template,&loaded);
        assert_eq!(loaded.save(),bytes);
        let transfer=template.save_with_external_vocab();
        assert!(DynamicConstraint::load(&transfer).is_err());
        let transfer=<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&transfer,&v).unwrap();
        compare_dynamic(&template,&transfer);
    }
}

fn parser_range(bytes:&[u8])->std::ops::Range<usize> {
    assert_eq!(&bytes[18..22],b"S30\0");
    let sizes:Vec<usize>=(0..11).map(|i|u64::from_le_bytes(bytes[22+i*8..30+i*8].try_into().unwrap()) as usize).collect();
    let start=18+4+11*8+sizes[0]+sizes[1]; start..start+sizes[2]
}
fn replace_parser(bytes:&[u8], parser:&[u8])->Vec<u8> {
    let range=parser_range(bytes);
    let mut out=bytes[..range.start].to_vec(); out.extend_from_slice(parser); out.extend_from_slice(&bytes[range.end..]);
    out[38..46].copy_from_slice(&(parser.len() as u64).to_le_bytes());
    let size=(out.len()-18) as u64; out[10..18].copy_from_slice(&size.to_le_bytes()); out
}
#[test]
fn malformed_template_parser_metadata_is_rejected_without_a_table_fallback() {
    let original=compact_fixture(); let parser=&original[parser_range(&original)];
    assert_eq!(&parser[..4],b"TPR7");
    for count in [0usize,1,3,4,7,15,parser.len()-1] {
        assert!(Constraint::load(replace_parser(&original,&parser[..count])).is_err(),"accepted truncation {count}");
    }
    let mut bad=parser.to_vec(); bad[0]=b'X';
    assert!(Constraint::load(replace_parser(&original,&bad)).is_err());
    let mut cursor=4;
    for field in ["stack alphabet", "terminal count", "skip count"] {
        let start=cursor; read_var(parser,&mut cursor);
        let bad=mutate_range(parser,start..cursor,&var_bytes(u32::MAX));
        assert!(Constraint::load(replace_parser(&original,&bad)).is_err(),"accepted forged {field}");
    }
    let alphabet_end={let mut end=4; read_var(parser,&mut end); end};
    let bad=mutate_range(parser,4..alphabet_end,&var_bytes(0));
    assert!(Constraint::load(replace_parser(&original,&bad)).is_err(),"accepted empty stack alphabet");
    let mut bad=parser.to_vec(); bad.extend_from_slice(&[0]);
    assert!(Constraint::load(replace_parser(&original,&bad)).is_err());
    let mut bad=original.clone(); bad[8..10].copy_from_slice(&31u16.to_le_bytes());
    assert!(Constraint::load(bad).is_err(),"current parser section accepted under an obsolete envelope");
}

#[test]
fn cyclic_or_out_of_domain_completion_program_is_rejected() {
    let original=compact_fixture(); let parser=&original[parser_range(&original)];
    let mut cursor=4;
    let alphabet=read_var(parser,&mut cursor); let terminals=read_var(parser,&mut cursor);
    let skips=read_var(parser,&mut cursor); for _ in 0..skips {read_var(parser,&mut cursor);}
    // EOF completion follows every terminal program. Mutate that relation
    // specifically, so terminal-graph validation cannot satisfy this test.
    for _ in 0..terminals {skip_program(parser,&mut cursor);}
    let mut exercised=false;
    'phases: for _ in 0..3 {
        let states=read_var(parser,&mut cursor); if states==0 {continue;}
        read_var(parser,&mut cursor);
        for source in 0..states {
            let edges=read_var(parser,&mut cursor)>>1;
            for _ in 0..edges {
                let label=cursor; read_var(parser,&mut cursor); let label_end=cursor;
                let target=cursor; read_var(parser,&mut cursor); let target_end=cursor;
                let cycle=mutate_range(parser,target..target_end,&var_bytes(source));
                let error=Constraint::load(replace_parser(&original,&cycle)).unwrap_err();
                assert!(error.to_string().contains("cyclic"),"wrong completion-cycle rejection: {error}");
                let missing=mutate_range(parser,target..target_end,&var_bytes(states));
                let error=Constraint::load(replace_parser(&original,&missing)).unwrap_err();
                assert!(error.to_string().contains("missing state"),"wrong completion-target rejection: {error}");
                let outside=mutate_range(parser,label..label_end,&var_bytes(alphabet+1));
                let error=Constraint::load(replace_parser(&original,&outside)).unwrap_err();
                assert!(error.to_string().contains("alphabet"),"wrong completion-label rejection: {error}");
                exercised=true; break 'phases;
            }
        }
    }
    assert!(exercised,"fixture must have a completion edge");
}

#[test]
fn external_template_artifacts_omit_vocab_require_exact_binding_and_roundtrip() {
    let v = vocab();
    let reference = DynamicConstraint::compile_with_vocab_partition(Grammar::glrm(GRAMMARS[1]), &v).unwrap();
    let template = reference.clone();
    let external = template.save_with_external_vocab();
    assert_eq!(&external[..8], b"GLRDXF\0\0");
    assert_eq!(u16::from_le_bytes(external[8..10].try_into().unwrap()), 14);
    assert!(DynamicConstraint::load(&external).is_err(), "external artifact accepted without a vocabulary");
    // First dynamic alternative: outer18 + count4 + descriptor8.
    let body = &external[30..];
    assert_eq!(u16::from_le_bytes(body[8..10].try_into().unwrap()),38);
    let token_section_len = u64::from_le_bytes(body[22 + 5*8..30 + 5*8].try_into().unwrap());
    assert_eq!(token_section_len,0,"model-token bytes must really be absent");
    let parser = &body[parser_range(body)];
    assert_eq!(&parser[..4],b"TPX1");
    assert_eq!(&parser[36..40],b"TPR7");
    let loaded = <DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&external, &v).unwrap();
    compare_dynamic(&reference, &loaded);
    assert_eq!(loaded.save_with_external_vocab(), external,"external re-save must use same-mode backing bytes");
    for report in dynamic_parser_backend_report(&loaded).as_array().unwrap() {
        assert_eq!(report["lr_table_present"], false);
    }
    // A mode switch must encode token bytes, not return cached external bytes
    // under a self-contained API. The inverse conversion must remain exact too.
    let self_contained = loaded.save();
    assert_eq!(&self_contained[..8],b"GLRDYN\0\0");
    let restored = DynamicConstraint::load(&self_contained).unwrap();
    compare_dynamic(&template, &restored);
    let external_again = restored.save_with_external_vocab();
    compare_dynamic(&template, &<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&external_again,&v).unwrap());
    let mut mapping = v.iter().map(|(id,bytes)| (id,bytes.to_vec())).collect::<std::collections::BTreeMap<_,_>>();
    mapping.insert(0,b"different token bytes".to_vec());
    let incompatible = Vocab::new(mapping.into_iter().collect());
    let error = <DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&external,&incompatible).unwrap_err();
    assert!(error.to_string().contains("vocabulary"),"wrong-vocab failure: {error}");
}

#[test]
fn malformed_external_template_binding_is_rejected() {
    let v=vocab();
    let template=DynamicConstraint::compile_with_vocab_partition(Grammar::glrm(GRAMMARS[0]),&v).unwrap();
    let external=template.save_with_external_vocab();
    let baseline=<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&external,&v)
        .expect("malformed binding cases must start from a valid current external artifact");
    compare_dynamic(&template,&baseline);
    assert_eq!(baseline.save_with_external_vocab(),external);
    for length in [0,7,8,17,18,21,25,external.len()-1] {
        assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&external[..length],&v).is_err(),"accepted dynamic truncation{length}");
    }
    let mut bad=external.clone(); bad.push(0);
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&bad,&v).is_err());
    let mut bad=external.clone(); bad[18..22].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&bad,&v).is_err());
    let range=parser_range(&external[30..]);
    let mut bad=external.clone(); bad[30+range.start+4]^=1;
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&bad,&v).is_err(),"accepted forged vocabulary digest");
    let mut bad=external.clone(); bad[38..40].copy_from_slice(&37u16.to_le_bytes());
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&bad,&v).is_err(),"accepted external parser as self-contained");
}

#[test]
fn obsolete_lr_and_template_artifacts_are_rejected() {
    let v = vocab();
    let lr = include_bytes!("fixtures/template_parser_v1/static-v30-lr.bin");
    let rejected = std::panic::catch_unwind(|| Constraint::load(lr));
    assert!(rejected.is_err(), "LR-backed artifact materialization must panic loudly");
    assert!(Constraint::load(include_bytes!("fixtures/template_parser_v1/static-v31-tpr1.bin")).is_err());
    assert!(DynamicConstraint::load(include_bytes!("fixtures/template_parser_v1/o2-v21-tpr1.bin")).is_err());
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(
        include_bytes!("fixtures/template_parser_v1/o2-transfer-v14-tpx1.bin"), &v).is_err());
}

#[test]
fn artifacts_with_pre_nullable_fix_boundary_programs_are_rejected() {
    let v = vocab();
    let static_constraint = Constraint::compile(Grammar::glrm(GRAMMARS[0]), &v).unwrap();
    let current = static_constraint.save();
    for version in [35u16, 36u16] {
        let mut old = current.clone();
        old[8..10].copy_from_slice(&version.to_le_bytes());
        let error = Constraint::load(&old).unwrap_err();
        assert!(error.to_string().contains("unsupported constraint artifact version"),
            "pre-fix boundary artifacts must fail before executing a stored exclusion: {error}");
        assert!(Constraint::load_with_vocab(&old, &v).is_err());
    }
    let mut external = static_constraint.save_with_external_vocab().unwrap();
    external[8..10].copy_from_slice(&36u16.to_le_bytes());
    assert!(Constraint::load_with_vocab(&external, &v).is_err());

    let dynamic = DynamicConstraint::compile_with_vocab_partition(
        Grammar::glrm(GRAMMARS[0]), &v,
    ).unwrap();
    let mut old_dynamic = dynamic.save();
    old_dynamic[38..40].copy_from_slice(&35u16.to_le_bytes());
    assert!(DynamicConstraint::load(&old_dynamic).is_err());
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&old_dynamic, &v).is_err());
    let mut old_external_dynamic = dynamic.save_with_external_vocab();
    old_external_dynamic[38..40].copy_from_slice(&36u16.to_le_bytes());
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&old_external_dynamic, &v).is_err());

    let mut old_module = Grammar::glrm(GRAMMARS[0]).compile_unlinked(&v).unwrap().save();
    let body_start = 16 + u64::from_le_bytes(old_module[8..16].try_into().unwrap()) as usize;
    old_module[body_start + 8..body_start + 10].copy_from_slice(&35u16.to_le_bytes());
    assert!(glrmask::UnlinkedConstraint::load(&old_module).is_err());
    assert!(glrmask::UnlinkedConstraint::load_with_vocab(&old_module, &v).is_err());
}

fn compact_fixture()->Vec<u8> {
    let source=Constraint::compile(Grammar::glrm(GRAMMARS[1]),&vocab()).unwrap();
    let bytes=source.save();
    let loaded=Constraint::load(&bytes).expect("malformed cases must start from a valid current artifact");
    assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()),37);
    assert_eq!(parser_backend_report(&loaded)["lr_table_present"],false);
    compare_static(&source,&loaded);
    assert_eq!(loaded.save(),bytes);
    bytes
}
fn read_var(bytes:&[u8],offset:&mut usize)->u32 {
    let mut value=0;
    for shift in (0..=28).step_by(7) {
        let byte=bytes[*offset];*offset+=1;value|=u32::from(byte&127)<<shift;
        if byte&128==0{return value;}
    }
    panic!("test fixture has malformed varint")
}
fn skip_program(bytes: &[u8], cursor: &mut usize) {
    for _ in 0..3 {
        let states=read_var(bytes,cursor);
        if states==0 {continue;}
        read_var(bytes,cursor); // start state
        for _ in 0..states {
            let edges=read_var(bytes,cursor)>>1;
            for _ in 0..edges {read_var(bytes,cursor); read_var(bytes,cursor);}
        }
    }
    for _ in 0..3 {
        let count=read_var(bytes,cursor);
        for _ in 0..count {read_var(bytes,cursor);}
    }
}
fn var_bytes(mut value:u32)->Vec<u8> {
    let mut bytes=Vec::new();
    while value>=128 {bytes.push((value as u8&127)|128);value>>=7;}
    bytes.push(value as u8);bytes
}
fn mutate_range(bytes:&[u8],range:std::ops::Range<usize>,value:&[u8])->Vec<u8> {
    let mut result=bytes[..range.start].to_vec();result.extend_from_slice(value);result.extend_from_slice(&bytes[range.end..]);result
}

#[test]
fn compact_template_programs_reject_noncanonical_counts_and_truncation() {
    let original=compact_fixture();let parser=&original[parser_range(&original)];
    assert_eq!(&parser[..4],b"TPR7");
    let mut cursor=4;let alphabet=read_var(parser,&mut cursor);let first_end=cursor;
    for bad in [vec![128,0],vec![255,255,255,255,31],vec![128;6]] {
        let bad=mutate_range(parser,4..first_end,&bad);
        let error=Constraint::load(replace_parser(&original,&bad)).unwrap_err();
        assert!(error.to_string().contains("varint"),"wrong rejection for malformed varint: {error}");
    }
    assert!(alphabet>0);
    for length in 0..parser.len() {
        assert!(Constraint::load(replace_parser(&original,&parser[..length])).is_err(),"accepted compact truncation{length}");
    }
    let mut trailing=parser.to_vec();trailing.push(0);
    assert!(Constraint::load(replace_parser(&original,&trailing)).is_err());
}

#[test]
fn compact_template_edges_reject_cycles_missing_targets_and_bad_labels() {
    let original=compact_fixture();let parser=&original[parser_range(&original)];
    let mut cursor=4;
    let alphabet=read_var(parser,&mut cursor);let terminals=read_var(parser,&mut cursor);
    let skips=read_var(parser,&mut cursor);for _ in 0..skips{read_var(parser,&mut cursor);}
    let mut exercised=false;
    'programs: for _ in 0..=terminals {
        for _phase in 0..3 {
            let states=read_var(parser,&mut cursor);
            if states==0{continue;}
            read_var(parser,&mut cursor); // start state
            for source in 0..states {
                let flags=read_var(parser,&mut cursor);
                for _ in 0..flags>>1 {
                    let label_start=cursor;read_var(parser,&mut cursor);let label_end=cursor;
                    let target_start=cursor;read_var(parser,&mut cursor);let target_end=cursor;
                    let bad=mutate_range(parser,target_start..target_end,&var_bytes(source));
                    let error=Constraint::load(replace_parser(&original,&bad)).unwrap_err();
                    assert!(error.to_string().contains("cyclic"),"wrong self-cycle rejection: {error}");
                    let bad=mutate_range(parser,target_start..target_end,&var_bytes(states));
                    let error=Constraint::load(replace_parser(&original,&bad)).unwrap_err();
                    assert!(error.to_string().contains("missing state"),"wrong missing-target rejection: {error}");
                    let bad=mutate_range(parser,label_start..label_end,&var_bytes(alphabet+1));
                    let error=Constraint::load(replace_parser(&original,&bad)).unwrap_err();
                    assert!(error.to_string().contains("alphabet"),"wrong label rejection: {error}");
                    exercised=true;break 'programs;
                }
            }
        }
        for _ in 0..3 {let links=read_var(parser,&mut cursor);for _ in 0..links{read_var(parser,&mut cursor);}}
    }
    assert!(exercised,"fixture needs a nonempty transition graph");
}

#[test]
fn table_free_root_end_and_exact_only_token_policies_survive_roundtrip() {
    use glrmask::BuildOptions;
    let v=Vocab::new_with_exact_token_ids(vec![(0,b"a".to_vec()),(1,b"b".to_vec())],[31,77]);
    let source=Grammar::from_glrm(r#"start start; t A ::= "a"; nt start ::= A;"#);
    let reference=source.compile_with(&v,BuildOptions::default().end_tokens([77])).unwrap();
    let template=reference.clone();
    let saved=template.save();
    assert_eq!(&saved[..8],b"GLRROOT2");
    let restored=Constraint::load_with_vocab(saved.clone(),&v).unwrap();
    assert_eq!(parser_backend_report(&restored)["lr_table_present"],false);
    let mut a=reference.start();let mut b=restored.start();
    for token in [0,77] {
        assert_eq!(a.mask(),b.mask());
        a.commit_token(token).unwrap();b.commit_token(token).unwrap();
        assert_eq!(a.is_accepting(),b.is_accepting());
        assert_eq!(a.is_terminated(),b.is_terminated());
    }
    assert!(b.is_terminated());
    let missing=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec())]);
    assert!(Constraint::load_with_vocab(saved,&missing).is_err());
}

#[test]
fn malformed_embedding_flags_slots_and_finish_graphs_are_rejected() {
    let original = compact_fixture(); let parser = &original[parser_range(&original)];
    assert_eq!(&parser[..4], b"TPR7");
    let mut cursor = 4; let alphabet = read_var(parser, &mut cursor); let terminals = read_var(parser, &mut cursor);
    let skips = read_var(parser, &mut cursor); for _ in 0..skips { read_var(parser, &mut cursor); }
    for _ in 0..=terminals { skip_program(parser, &mut cursor); }
    let composed = cursor; assert_eq!(read_var(parser, &mut cursor), 0);
    let embedding = cursor; assert_eq!(read_var(parser, &mut cursor), 1);
    let nullable = cursor; read_var(parser, &mut cursor);
    let return_pop = cursor; read_var(parser, &mut cursor);
    let count_start = cursor; let slots = read_var(parser, &mut cursor);
    for _ in 0..slots { read_var(parser, &mut cursor); }
    let finish_start = cursor;
    let finish_alphabet = cursor; read_var(parser, &mut cursor);
    for (position, value) in [(composed, 2), (embedding, 2), (nullable, 2), (return_pop, 0), (return_pop, 3), (finish_alphabet, 0), (finish_alphabet, alphabet + 1)] {
        let mut end = position; read_var(parser, &mut end);
        let bad = mutate_range(parser, position..end, &var_bytes(value));
        assert!(Constraint::load(replace_parser(&original, &bad)).is_err(), "accepted malformed embedding field {position}");
    }
    let mut invalid_slot = var_bytes(1); invalid_slot.extend(var_bytes(terminals));
    let bad = mutate_range(parser, count_start..finish_start, &invalid_slot);
    assert!(Constraint::load(replace_parser(&original, &bad)).is_err());
    let mut exercised = false;
    'phases: for _ in 0..3 {
        let states = read_var(parser, &mut cursor); if states == 0 { continue; }
        read_var(parser, &mut cursor);
        for source in 0..states {
            let edges = read_var(parser, &mut cursor) >> 1;
            for _ in 0..edges {
                let label = cursor; read_var(parser, &mut cursor); let label_end = cursor;
                let target = cursor; read_var(parser, &mut cursor); let target_end = cursor;
                let cycle = mutate_range(parser, target..target_end, &var_bytes(source));
                assert!(Constraint::load(replace_parser(&original, &cycle)).is_err());
                let outside = mutate_range(parser, label..label_end, &var_bytes(alphabet + 1));
                assert!(Constraint::load(replace_parser(&original, &outside)).is_err());
                exercised = true; break 'phases;
            }
        }
    }
    assert!(exercised, "embedding fixture must contain a nontrivial Finish relation");
}
