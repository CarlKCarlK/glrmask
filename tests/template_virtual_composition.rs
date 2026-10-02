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

    // An already static composition whose root lexer is virtual is not itself
    // an intact lexer leaf. Reusing it must retain its leaf observations rather
    // than trying to prepare a projection over the coordinator's tokenizer.
    let saved = candidate.save();
    let middle = Constraint::load(&saved).unwrap();
    let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "p" middle "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("middle", &middle).unwrap();
    let outer_reference = outer.link_with(options(Optimization::FastBuild)).unwrap();
    let outer_candidate = outer.link_with(options(Optimization::FastRuntime)).unwrap();
    let mut outer_prefixes = vec![vec![], b"p".to_vec()];
    outer_prefixes.extend(prefixes.iter().map(|prefix| {
        let mut word = vec![b'p']; word.extend(prefix); word
    }));
    outer_prefixes.push(b"p\"x:a\"!q".to_vec());
    verify_forms(&outer_reference, &outer_candidate, &vocab, &tokens, &outer_prefixes);
    assert_eq!(middle.save(), saved, "linking must not mutate the reusable virtual-root component");
}

#[test]
fn malformed_projected_observation_offsets_are_rejected_on_load() {
    let tokens = vec![(0, b"p".to_vec()), (1, b"\"x:a\"".to_vec()),
        (2, b"q".to_vec()), (3, b"p\"x:a\"q".to_vec())];
    let vocab = Vocab::new(tokens.clone());
    let leaf = Grammar::from_json_schema(URI).compile_with(&vocab, options(Optimization::FastRuntime)).unwrap();
    let candidate = Grammar::from_glrm(r#"glrm 1; start root; extern grammar leaf; nt root = "p" leaf "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("leaf", &leaf).unwrap()
        .link_with(options(Optimization::FastRuntime)).unwrap();
    let report = glrmask::__private::parser_backend_report(&candidate);
    let offsets = report["finite_observation_offsets"].as_array().unwrap().iter()
        .map(|value| value.as_u64().unwrap() as u32).collect::<Vec<_>>();
    let saved = candidate.save();
    let loaded = Constraint::load(&saved).expect("corruption fixture must have a valid native baseline");
    assert_eq!(loaded.save(), saved);
    let prefixes = vec![vec![], b"p".to_vec(), b"p\"".to_vec(), b"p\"x:".to_vec(),
        b"p\"x:a".to_vec(), b"p\"x:a\"".to_vec(), b"p\"x:a\"q".to_vec()];
    compare(&candidate, &loaded, &tokens, &prefixes);
    assert_eq!(u16::from_le_bytes(saved[8..10].try_into().unwrap()), 33);
    assert_eq!(&saved[18..22], b"S30\0");
    let sizes = (0..11).map(|index| u64::from_le_bytes(saved[22 + index * 8..30 + index * 8].try_into().unwrap()) as usize).collect::<Vec<_>>();
    let start = 18 + 4 + 11 * 8 + sizes[..4].iter().sum::<usize>();
    let runtime = &saved[start..start + sizes[4]];
    // R33 retains the 20-byte header (magic, metadata length, residual length).
    // Its metadata now starts with variable-size native dynamic proofs, so
    // locate the observation inventory inside that bounded metadata section.
    const RUNTIME_HEADER_LEN: usize = 20;
    assert_eq!(&runtime[..4], b"R33\0");
    let metadata_len = u64::from_le_bytes(runtime[4..12].try_into().unwrap()) as usize;
    let residual_len = u64::from_le_bytes(runtime[12..20].try_into().unwrap()) as usize;
    assert_eq!(runtime.len(), RUNTIME_HEADER_LEN + metadata_len + residual_len);
    let metadata = &runtime[RUNTIME_HEADER_LEN..RUNTIME_HEADER_LEN + metadata_len];
    assert!(offsets.len() >= 3, "fixture must retain parent and virtual child observations");
    let mut needle = (offsets.len() as u64).to_le_bytes().to_vec();
    for offset in &offsets { needle.extend(offset.to_le_bytes()); }
    let positions = metadata.windows(needle.len()).enumerate().filter_map(|(i, value)| (value == needle).then_some(i)).collect::<Vec<_>>();
    assert_eq!(positions.len(), 1, "the exact encoded observation must be unique in this fixture");
    let base = start + RUNTIME_HEADER_LEN + positions[0];
    for (index, replacement) in [(0, 0), (1, 1), (offsets.len() - 1, u32::MAX)] {
        let mut corrupted = saved.clone();
        corrupted[base + 8 + index * 4..base + 12 + index * 4].copy_from_slice(&u32::to_le_bytes(replacement));
        let error = Constraint::load(corrupted).unwrap_err().to_string();
        assert!(error.contains("observation"), "{error}");
    }
    let descriptor = base + needle.len();
    for (offset, replacement) in [(descriptor, 0u32), (descriptor + 4, 1u32)] {
        let mut corrupted = saved.clone();
        corrupted[offset..offset + 4].copy_from_slice(&replacement.to_le_bytes());
        let error = Constraint::load(corrupted).unwrap_err().to_string();
        assert!(error.contains("observation"), "{error}");
    }
    let count = u64::from_le_bytes(saved[descriptor + 8..descriptor + 16].try_into().unwrap()) as usize;
    assert_eq!(count, offsets.len() - 1);
    let mut digest = descriptor + 16;
    let mut corrupted_fingerprint = false;
    for _ in 0..count {
        let present = saved[digest]; digest += 1;
        if present != 0 {
            assert_eq!(present, 1);
            let mut corrupted = saved.clone(); corrupted[digest] ^= 1;
            let error = Constraint::load(corrupted).unwrap_err().to_string();
            assert!(error.contains("observation"), "{error}");
            corrupted_fingerprint = true;
            break;
        }
    }
    assert!(corrupted_fingerprint, "virtual child must supply an observation fingerprint to corrupt");
    let loaded = Constraint::load(saved).unwrap();
    let mut state = loaded.start(); state.commit_token(3).unwrap(); assert!(state.is_accepting());
}

#[test]
fn virtual_repeat_observations_preserve_literal_lower_and_upper_bounds() {
    if isolated("virtual_repeat_observations_preserve_literal_lower_and_upper_bounds") { return; }
    let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    tokens.extend([(300, b"paaq".to_vec()), (303, b"aaq".to_vec()), (305, b"aaa".to_vec()),
        (400, b"paaq".to_vec()), (501, b"aq".to_vec()), (502, b"pq".to_vec())]);
    let mut long_crossing = vec![b'a'; 64]; long_crossing.push(b'q');
    tokens.push((4097, long_crossing));
    let vocab = Vocab::new(tokens.clone());
    for (regex, minimum, maximum) in [("a{2,5000}", 2usize, 5000usize),
        ("a{4096,8192}", 4096, 8192)] {
        let source = format!("start root; t A ::= /{regex}/; nt root ::= A;");
        // FastRuntime may materialize this simple finite repeat. Preserve a
        // genuinely virtual child, then request static boundary compilation;
        // its explicitly dynamic local mask engine remains independent.
        let child = Grammar::from_glrm(&source).compile_with(&vocab, options(Optimization::FastBuild)).unwrap();
        assert_eq!(glrmask::__private::parser_backend_report(&child)["virtual_lexer"], true,
            "the fixture must exercise a virtual repeat: {regex}");
        let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "p" C "q";"#)
            .compile_unlinked(&vocab).unwrap().bind("C", &child).unwrap();
        let reference = parent.link_with(options(Optimization::FastBuild)).unwrap();
        let candidate = parent.link_with(options(Optimization::FastRuntime)).unwrap();
        let mut prefixes = vec![vec![]];
        for count in [0, 1, 2, 31, minimum.saturating_sub(65), minimum.saturating_sub(64),
            minimum.saturating_sub(63), maximum - 65, maximum - 64, maximum - 63,
            maximum - 40, maximum - 2, maximum - 1, maximum] {
            let mut word = vec![b'p']; word.extend(std::iter::repeat_n(b'a', count)); prefixes.push(word);
        }
        verify_forms(&reference, &candidate, &vocab, &tokens, &prefixes);
        for prefix in &prefixes {
            let mut state = candidate.start(); state.commit_bytes(prefix).unwrap();
            let mask = state.mask();
            for (id, bytes) in &tokens {
                let mut word = prefix.clone(); word.extend(bytes);
                let expected = if let Some(rest) = word.strip_prefix(b"p") {
                    let (body, complete) = rest.strip_suffix(b"q").map_or((rest, false), |body| (body, true));
                    body.iter().all(|&byte| byte == b'a') && body.len() <= maximum
                        && (!complete || body.len() >= minimum)
                } else { word.is_empty() };
                let actual = mask.get(*id as usize / 32).is_some_and(|bits| bits & (1 << (*id % 32)) != 0);
                assert_eq!(actual, expected, "regex={regex} prefix_len={} token={bytes:?}", prefix.len());
            }
        }
        let mut too_long = vec![b'p']; too_long.extend(std::iter::repeat_n(b'a', maximum + 1));
        assert!(candidate.start().commit_bytes(&too_long).is_err(), "regex={regex}");
    }
}

#[test]
fn unsupported_general_virtual_projection_fails_without_changing_dynamic_language() {
    if isolated("unsupported_general_virtual_projection_fails_without_changing_dynamic_language") { return; }
    // This non-prefix-free general residual has no certified finite projector.
    // Static construction must report that limitation, not silently use the
    // dynamic B path or reuse an unrelated component-mask projection.
    let vocab = Vocab::new(vec![(0, b"p".to_vec()), (1, b"a".to_vec()),
        (2, b"aaa".to_vec()), (3, b"q".to_vec()), (4, b"aq".to_vec())]);
    let child = Grammar::from_glrm("start root; t A ::= /(a|aa){1,5000}/; nt root ::= A;")
        .compile_with(&vocab, options(Optimization::FastBuild)).unwrap();
    assert_eq!(glrmask::__private::parser_backend_report(&child)["virtual_lexer"], true);
    let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "p" C "q";"#)
        .compile_unlinked(&vocab).unwrap().bind("C", &child).unwrap();
    let error = parent.link_with(options(Optimization::FastRuntime)).unwrap_err().to_string();
    assert!(error.contains("no supported finite boundary observation"), "{error}");
    let dynamic = parent.link_with(options(Optimization::FastBuild)).unwrap();
    let loaded = Constraint::load(dynamic.save()).unwrap();
    for c in [&dynamic, &loaded] {
        for count in [0usize, 1, 2, 31, 65] {
            let mut prefix = vec![b'p']; prefix.extend(std::iter::repeat_n(b'a', count));
            let mut state = c.start(); state.commit_bytes(&prefix).unwrap();
            let mask = state.mask();
            assert_eq!(mask[0] & (1 << 3) != 0, count >= 1);
            assert_ne!(mask[0] & (1 << 4), 0);
            state.commit_token(4).unwrap(); assert!(state.is_accepting());
        }
    }
}


mod product_review {
    use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};

    fn options(mode: Optimization) -> BuildOptions {
        BuildOptions::default().optimization(mode).parser_backend(ParserBackend::TemplateDfa)
    }

    // (ab|c)* intersect (a|bc)* is (abc)*. Both repeated bodies consume
    // two copies per abc, so bounds 5000 and 4000 permit at most 2000 abc.
    fn viable(word: &[u8]) -> bool {
        if word.is_empty() { return true; }
        if word[0] != b'p' { return false; }
        let body = &word[1..];
        let (body, finished) = body.strip_suffix(b"q").map_or((body, false), |b| (b, true));
        body.len() <= 6000 && (!finished || body.len() % 3 == 0)
            && body.iter().enumerate().all(|(i, byte)| *byte == b"abc"[i % 3])
    }

    #[test]
    fn product_intersection_child_has_exact_full_horizon_crossings() -> Result<(), Box<dyn std::error::Error>> {
        if super::isolated("product_review::product_intersection_child_has_exact_full_horizon_crossings") { return Ok(()); }
        let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
        tokens.extend([(300, b"pq".to_vec()), (301, b"pabcq".to_vec()),
            (302, b"abc".to_vec()), (303, b"bcq".to_vec()), (304, b"cq".to_vec()),
            (305, format!("{}q", "abc".repeat(20)).into_bytes()),
            (306, "abc".repeat(20).into_bytes())]);
        let vocab = Vocab::new(tokens.clone());
        // Lexical epsilon is lifted to a parser alternative before execution.
        // Preserve that source nullability in the reusable child's embedding;
        // requiring A? here would conceal a lost source-language alternative.
        let source = "start root; t A ::= /(ab|c){0,5000}/ & /(a|bc){0,4000}/; nt root ::= A;";
        let child = Grammar::from_glrm(source).compile_with(&vocab, options(Optimization::FastBuild))?;
        let report = glrmask::__private::parser_backend_report(&child);
        println!("CHILD {report}");
        assert_eq!(report["virtual_lexer"], true, "review must use an actual virtual lexer");
        let original = child.save();
        let parent = Grammar::from_glrm("glrm 1; start root; extern grammar child; nt root = \"p\" child \"q\";")
            .compile_unlinked(&vocab)?.bind("child", &child)?;
        let dynamic = parent.link_with(options(Optimization::FastBuild))?;
        println!("DYNAMIC LINKED");
        let candidate = parent.link_with(options(Optimization::FastRuntime))?;
        println!("STATIC {}", glrmask::__private::parser_backend_report(&candidate));
        assert_eq!(child.save(), original, "link changed reusable child");
        let saved = candidate.save();
        let loaded = Constraint::load(&saved)?;
        assert_eq!(loaded.save(), saved);
        let external = Constraint::load_with_vocab(candidate.save_with_external_vocab()?, &vocab)?;
        let mut prefixes = vec![vec![], b"p".to_vec(), b"pq".to_vec(), b"pabcq".to_vec()];
        for count in [1, 7, 1979, 1980, 1998, 1999, 2000] {
            let prefix = format!("p{}", "abc".repeat(count)).into_bytes();
            prefixes.push(prefix.clone());
            if count < 2000 {
                for suffix in [b"a".as_slice(), b"ab"] {
                    let mut p = prefix.clone(); p.extend(suffix); prefixes.push(p);
                }
            }
        }
        let mut comparisons = 0;
        for (representation, c) in [("dynamic", &dynamic), ("static", &candidate),
            ("self-contained", &loaded), ("external-vocabulary", &external)] {
            for prefix in &prefixes {
                let mut state = c.start(); state.commit_bytes(prefix)?;
                assert_eq!(state.is_accepting(), prefix.ends_with(b"q"));
                let mask = state.mask();
                for (id, token) in &tokens {
                    let mut word = prefix.clone(); word.extend(token);
                    let expected = viable(&word);
                    let actual = mask.get(*id as usize / 32).is_some_and(|w| w & (1 << (*id % 32)) != 0);
                    assert_eq!(actual, expected, "representation={representation} prefix_len={} token={token:?}", prefix.len());
                    if actual {
                        let mut branch = state.clone(); branch.commit_token(*id)?;
                        assert_eq!(branch.is_accepting(), word.ends_with(b"q"));
                    }
                    comparisons += 1;
                }
            }
        }
        println!("PASS independent product language checks={comparisons}, all artifact forms and long crossing horizon");
        Ok(())
    }
}
