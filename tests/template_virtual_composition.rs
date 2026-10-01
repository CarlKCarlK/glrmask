use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

const URI: &str = r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#;

fn isolated(name: &str) -> bool {
    const MARKER: &str = "GLRMASK_VIRTUAL_COMPOSITION_ISOLATED";
    if std::env::var(MARKER).is_ok_and(|active| active == name) { return false; }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact").arg(name).arg("--nocapture").arg("--test-threads=1")
        .env(MARKER, name).env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1").output().unwrap();
    assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    true
}

fn verify_forms(reference: &Constraint, candidate: &Constraint, vocab: &Vocab, tokens: &[(u32, Vec<u8>)], prefixes: &[Vec<u8>]) {
    let saved = candidate.save();
    let loaded = Constraint::load(&saved).unwrap();
    assert_eq!(saved, loaded.save());
    let external = Constraint::load_with_vocab(candidate.save_with_external_vocab().unwrap(), vocab).unwrap();
    for c in [candidate, &loaded, &external] { compare(reference, c, tokens, prefixes); }
}

fn options(optimization: Optimization) -> BuildOptions {
    BuildOptions::default().optimization(optimization).parser_backend(ParserBackend::TemplateDfa)
}

fn check_static(constraint: &Constraint) {
    fn check(report: &serde_json::Value) {
        assert_eq!(report["lr_table_present"], false);
        if let Some(children) = report["component_parsers"].as_array() {
            assert_eq!(report["packed_lr_compiler_table_present"], false);
            assert_eq!(report["dynamic_boundary_shards"], 0);
            for child in children { check(child); }
        }
    }
    let report = glrmask::__private::parser_backend_report(constraint);
    check(&report);
    assert!(report["finite_observation_leaves"].as_u64().unwrap_or(0) > 0, "{report}");
}

fn compare(reference: &Constraint, candidate: &Constraint, tokens: &[(u32, Vec<u8>)], prefixes: &[Vec<u8>]) {
    check_static(candidate);
    for prefix in prefixes {
        let mut left = reference.start(); let mut right = candidate.start();
        assert_eq!(left.commit_bytes(prefix).is_ok(), right.commit_bytes(prefix).is_ok(), "prefix len={}", prefix.len());
        let expected = left.mask();
        assert_eq!(expected, right.mask(), "mask prefix len={}", prefix.len());
        assert_eq!(left.is_accepting(), right.is_accepting(), "completion len={}", prefix.len());
        for (id, spelling) in tokens {
            if !expected.get(*id as usize / 32).is_some_and(|word| word & (1 << (*id % 32)) != 0) { continue; }
            let mut a = left.clone(); let mut b = right.clone();
            a.commit_token(*id).unwrap_or_else(|error| panic!("dynamic reference rejected admitted token={id} spelling={spelling:?} prefix_len={} prefix_head={:?}: {error}", prefix.len(), &prefix[..prefix.len().min(24)]));
            b.commit_token(*id).unwrap_or_else(|error| panic!("token={id} prefix_len={}: {error}", prefix.len()));
            assert_eq!(a.is_accepting(), b.is_accepting(), "token={id} prefix_len={}", prefix.len());
            assert_eq!(a.mask(), b.mask(), "successor token={id} prefix_len={}", prefix.len());
        }
    }
}

#[test]
fn strict_static_virtual_uri_child_crossings_and_reloads() {
    const CHILD: &str = "GLRMASK_VIRTUAL_COMPOSITION_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("strict_static_virtual_uri_child_crossings_and_reloads")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        return;
    }
    let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    tokens.extend([(300, b"p\"x:a\"q".to_vec()), (303, b"a\"q".to_vec()),
        (307, b"\"q".to_vec()), (315, b"x:".to_vec()), (317, b"aaa".to_vec()),
        (400, b"p\"x:a\"q".to_vec()), (500, b"q\"".to_vec())]);
    let vocab = Vocab::new(tokens.clone());
    let child = Grammar::from_json_schema(r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#)
        .compile_with(&vocab, options(Optimization::FastRuntime)).unwrap();
    assert_eq!(glrmask::__private::parser_backend_report(&child)["virtual_lexer"], true, "test must exercise a virtual lexer");
    let child = Constraint::load(child.save()).unwrap();
    let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = "p" child "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("child", &child).unwrap();
    let reference = parent.link_with(options(Optimization::FastBuild)).unwrap();
    let candidate = parent.link_with(options(Optimization::FastRuntime)).unwrap();
    let mut prefixes = vec![vec![], b"p".to_vec(), b"p\"".to_vec(), b"p\"x:".to_vec(), b"p\"x:a".to_vec(), b"p\"x:a\"q".to_vec()];
    for count in [31, 4996, 4997, 4998] {
        let mut prefix = b"p\"x:".to_vec(); prefix.extend(std::iter::repeat_n(b'a', count)); prefixes.push(prefix);
    }
    let saved = candidate.save();
    let loaded = Constraint::load(&saved).unwrap();
    assert_eq!(saved, loaded.save());
    let external = Constraint::load_with_vocab(candidate.save_with_external_vocab().unwrap(), &vocab).unwrap();
    for c in [&candidate, &loaded, &external] { compare(&reference, c, &tokens, &prefixes); }
    let mut too_long = b"p\"x:".to_vec(); too_long.extend(std::iter::repeat_n(b'a', 4999));
    assert!(candidate.start().commit_bytes(&too_long).is_err());
}

#[test]
fn virtual_children_remain_exact_when_nested_repeated_and_terminated() {
    if isolated("virtual_children_remain_exact_when_nested_repeated_and_terminated") { return; }
    let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    tokens.extend([(300, b"p\"x:a\"q".to_vec()), (301, b"q.p\"x:".to_vec()),
        (303, b"a\"q!".to_vec()), (310, b"xp\"x:a\"q.p\"x:b\"q!".to_vec()),
        (320, b"aaa".to_vec()), (1000, vec![])]);
    let vocab = Vocab::new(tokens.clone());
    let leaf = Grammar::from_json_schema(URI).compile_with(&vocab, options(Optimization::FastRuntime)).unwrap();
    assert_eq!(glrmask::__private::parser_backend_report(&leaf)["virtual_lexer"], true);
    let middle = Grammar::from_glrm(r#"glrm 1; start root; extern grammar leaf; nt root = "p" leaf "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap()
        .link_with(options(Optimization::FastRuntime)).unwrap();
    let middle = Constraint::load(middle.save()).unwrap();
    let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle "." middle "!";"#)
        .compile_unlinked(&vocab).unwrap().bind("middle", &middle).unwrap();
    let reference = parent.link_with(options(Optimization::FastBuild).end_tokens([1000])).unwrap();
    let candidate = parent.link_with(options(Optimization::FastRuntime).end_tokens([1000])).unwrap();
    let word = b"xp\"x:a\"q.p\"x:b\"q!";
    let mut prefixes = (0..=word.len()).map(|len| word[..len].to_vec()).collect::<Vec<_>>();
    let mut near_end = b"xp\"x:a\"q.p\"x:".to_vec(); near_end.extend(std::iter::repeat_n(b'a', 4998)); prefixes.push(near_end);
    verify_forms(&reference, &candidate, &vocab, &tokens, &prefixes);
    let mut state = candidate.start(); state.commit_token(310).unwrap();
    assert!(state.is_accepting()); state.commit_token(1000).unwrap(); assert!(state.is_terminated());
}

#[test]
fn virtual_parent_observation_is_not_its_component_mask_quotient() {
    if isolated("virtual_parent_observation_is_not_its_component_mask_quotient") { return; }
    use glrmask::__private::ConstraintExt;
    let mut source = Constraint::dump_json_schema_grammar_glrm(URI).unwrap();
    let start_line = source.lines().find(|line| line.trim_start().starts_with("start ")).unwrap().to_owned();
    let old_start = start_line.trim().strip_prefix("start ").unwrap().trim_end_matches(';');
    let wrapper = format!("\nextern grammar SUFFIX;\nnt wrapped_root ::= {old_start} SUFFIX;\n");
    source = source.replacen(&start_line, "start wrapped_root;", 1); source.push_str(&wrapper);
    let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    tokens.extend([(300, b"\"x:a\"!".to_vec()), (301, b"a\"!".to_vec()), (302, b"aaa".to_vec())]);
    let vocab = Vocab::new(tokens.clone());
    let leaf = Grammar::from_ebnf(r#"start ::= "!""#).compile_with(&vocab, options(Optimization::FastRuntime)).unwrap();
    let parent = Grammar::from_glrm(&source).compile_unlinked(&vocab).unwrap().bind("SUFFIX", &leaf).unwrap();
    let reference = parent.link_with(options(Optimization::FastBuild)).unwrap();
    let candidate = parent.link_with(options(Optimization::FastRuntime)).unwrap();
    let report = glrmask::__private::parser_backend_report(&candidate);
    assert_eq!(report["component_parsers"][0]["virtual_lexer"], true, "{report}");
    let word = b"\"x:a\"!";
    let mut prefixes = (0..=word.len()).map(|len| word[..len].to_vec()).collect::<Vec<_>>();
    let mut near_end = b"\"x:".to_vec(); near_end.extend(std::iter::repeat_n(b'a', 4998)); prefixes.push(near_end);
    verify_forms(&reference, &candidate, &vocab, &tokens, &prefixes);
}

#[test]
fn malformed_projected_observation_offsets_are_rejected_on_load() {
    let vocab = Vocab::new(vec![(0, b"p".to_vec()), (1, b"\"x:a\"".to_vec()),
        (2, b"q".to_vec()), (3, b"p\"x:a\"q".to_vec())]);
    let leaf = Grammar::from_json_schema(URI).compile_with(&vocab, options(Optimization::FastRuntime)).unwrap();
    let candidate = Grammar::from_glrm(r#"glrm 1; start root; extern grammar leaf; nt root = "p" leaf "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap()
        .link_with(options(Optimization::FastRuntime)).unwrap();
    let report = glrmask::__private::parser_backend_report(&candidate);
    let offsets = report["finite_observation_offsets"].as_array().unwrap().iter()
        .map(|value| value.as_u64().unwrap() as u32).collect::<Vec<_>>();
    let saved = candidate.save();
    assert_eq!(&saved[18..22], b"S30\0");
    let sizes = (0..11).map(|index| u64::from_le_bytes(saved[22 + index * 8..30 + index * 8].try_into().unwrap()) as usize).collect::<Vec<_>>();
    let start = 18 + 4 + 11 * 8 + sizes[..4].iter().sum::<usize>();
    let runtime = &saved[start..start + sizes[4]];
    assert_eq!(&runtime[..4], b"R29\0");
    let metadata_len = u64::from_le_bytes(runtime[4..12].try_into().unwrap()) as usize;
    let metadata = &runtime[20..20 + metadata_len];
    let mut needle = (offsets.len() as u64).to_le_bytes().to_vec();
    for offset in &offsets { needle.extend(offset.to_le_bytes()); }
    let positions = metadata.windows(needle.len()).enumerate().filter_map(|(i, value)| (value == needle).then_some(i)).collect::<Vec<_>>();
    assert_eq!(positions.len(), 1, "the exact encoded observation must be unique in this fixture");
    let base = start + 20 + positions[0];
    for (index, replacement) in [(0, 0), (1, 1), (offsets.len() - 1, u32::MAX)] {
        let mut corrupted = saved.clone();
        corrupted[base + 8 + index * 4..base + 12 + index * 4].copy_from_slice(&u32::to_le_bytes(replacement));
        let error = Constraint::load(corrupted).unwrap_err().to_string();
        assert!(error.contains("observation"), "{error}");
    }
    let loaded = Constraint::load(saved).unwrap();
    let mut state = loaded.start(); state.commit_token(3).unwrap(); assert!(state.is_accepting());
}
