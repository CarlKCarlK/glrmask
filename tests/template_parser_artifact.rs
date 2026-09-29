//! Persistence must not reintroduce an executable LR table into a mandatory
//! template backend. These tests use the same shared mask/commit engines before
//! and after loading, including invalid prefixes and exact EOF completion.
use glrmask::{Constraint, DynamicConstraint, Grammar, Vocab};
use glrmask::__private::{DynamicConstraintExt,into_template_parser, into_dynamic_template_parser,
    parser_backend_report, dynamic_parser_backend_report};

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
        let lr_bytes=reference.save();
        assert_eq!(u16::from_le_bytes(lr_bytes[8..10].try_into().unwrap()),30);
        compare_static(&reference,&Constraint::load(lr_bytes).unwrap());
        let template=into_template_parser(reference.clone()).unwrap();
        let bytes=template.save();
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()),31);
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
        let template=into_dynamic_template_parser(reference.clone()).unwrap();
        let bytes=template.save();
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()),21);
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
fn fixture()->Vec<u8> {
    let c=Constraint::compile(Grammar::glrm(GRAMMARS[1]),&vocab()).unwrap();
    into_template_parser(c).unwrap().save()
}

#[test]
fn malformed_template_parser_metadata_is_rejected_without_a_table_fallback() {
    let original=fixture(); let range=parser_range(&original); let parser=&original[range];
    assert_eq!(&parser[..4],b"TPR1");
    for count in [0usize,1,3,4,7,15,parser.len()-1] {
        assert!(Constraint::load(replace_parser(&original,&parser[..count])).is_err(),"accepted truncation {count}");
    }
    let mut bad=parser.to_vec(); bad[0]=b'X';
    assert!(Constraint::load(replace_parser(&original,&bad)).is_err());
    for offset in [4usize,8,12] {
        let mut bad=parser.to_vec(); bad[offset..offset+4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Constraint::load(replace_parser(&original,&bad)).is_err(),"accepted forged count at{offset}");
    }
    let mut bad=parser.to_vec(); bad.extend_from_slice(&[0]);
    assert!(Constraint::load(replace_parser(&original,&bad)).is_err());
    let mut bad=original.clone(); bad[8..10].copy_from_slice(&30u16.to_le_bytes());
    assert!(Constraint::load(bad).is_err(),"a template section cannot masquerade as an LR artifact");
}

#[test]
fn cyclic_or_out_of_domain_completion_program_is_rejected() {
    let original=fixture(); let parser=&original[parser_range(&original)];
    let skips=u32::from_le_bytes(parser[12..16].try_into().unwrap()) as usize;
    let mut pos=16+skips*4;
    let mut exercised=false;
    for _ in 0..3 {
        let states=u32::from_le_bytes(parser[pos+4..pos+8].try_into().unwrap()) as usize; pos+=8;
        for state in 0..states {
            let edges=u32::from_le_bytes(parser[pos+1..pos+5].try_into().unwrap()) as usize; pos+=5;
            if edges>0 {
                let mut bad=parser.to_vec(); bad[pos+4..pos+8].copy_from_slice(&(state as u32).to_le_bytes());
                assert!(Constraint::load(replace_parser(&original,&bad)).is_err(),"accepted completion self-cycle");
                let mut bad=parser.to_vec(); bad[pos+4..pos+8].copy_from_slice(&u32::MAX.to_le_bytes());
                assert!(Constraint::load(replace_parser(&original,&bad)).is_err(),"accepted missing target");
                exercised=true; break;
            }
            pos+=8*edges;
        }
        if exercised { break; }
    }
    assert!(exercised,"fixture must have a completion edge");
}

#[test]
fn external_template_artifacts_omit_vocab_require_exact_binding_and_roundtrip() {
    let v = vocab();
    let reference = DynamicConstraint::compile_with_vocab_partition(Grammar::glrm(GRAMMARS[1]), &v).unwrap();
    let template = into_dynamic_template_parser(reference.clone()).unwrap();
    let external = template.save_with_external_vocab();
    assert_eq!(&external[..8], b"GLRDXF\0\0");
    assert_eq!(u16::from_le_bytes(external[8..10].try_into().unwrap()), 14);
    assert!(DynamicConstraint::load(&external).is_err(), "external artifact accepted without a vocabulary");
    // First dynamic alternative: outer18 + count4 + descriptor8.
    let body = &external[30..];
    assert_eq!(u16::from_le_bytes(body[8..10].try_into().unwrap()),32);
    let token_section_len = u64::from_le_bytes(body[22 + 5*8..30 + 5*8].try_into().unwrap());
    assert_eq!(token_section_len,0,"model-token bytes must really be absent");
    let parser = &body[parser_range(body)];
    assert_eq!(&parser[..4],b"TPX1");
    assert_eq!(&parser[36..40],b"TPR1");
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
    let template=into_dynamic_template_parser(DynamicConstraint::compile_with_vocab_partition(Grammar::glrm(GRAMMARS[0]),&v).unwrap()).unwrap();
    let external=template.save_with_external_vocab();
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
    let mut bad=external.clone(); bad[38..40].copy_from_slice(&31u16.to_le_bytes());
    assert!(<DynamicConstraint as DynamicConstraintExt>::load_with_vocab(&bad,&v).is_err(),"accepted external parser as self-contained");
}
