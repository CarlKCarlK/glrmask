//! Native selected10 composition qualification against a grammar-inlined oracle.
//! Usage: template_precompiled_composition FIXTURE_DIR OUTPUT_DIR static|dynamic JS_GRAMMAR [--measure-runtime]
//! FIXTURE_DIR contains schema-00.json through schema-09.json, vocab_dump.bin, traces.json.
//! All masks are compared in memory; only measurements and hashes are saved.
use std::{fs, path::Path, io::{BufWriter, Write}, time::Instant};
use glrmask::{BuildOptions,Grammar,Optimization,Constraint,ParserBackend,Vocab};
use glrmask::__private::parser_backend_report;
use serde::Deserialize;
use glrmask_grammar::__private::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};
use std::collections::{BTreeMap, BTreeSet};

// Schema ASTs may contain inline regex atoms which AST lowering accepts but
// GLRM's nonterminal syntax deliberately rejects. Give those same atoms named
// emitting terminal aliases before serializing. The selected10 inputs use
// default placement for these four atoms; the independent schema oracle checks
// their language against original JSON import. Existing partitions, scopes and
// parser-FA edges are retained.
fn source_for_reparse(mut grammar: NamedGrammar) -> String {
    fn hoist(expr: &mut GrammarExpr, names: &mut BTreeSet<String>,
        aliases: &mut BTreeMap<String, String>, rules: &mut Vec<NamedRule>) {
        match expr {
            GrammarExpr::RawRegex(pattern) => {
                let name = aliases.entry(pattern.clone()).or_insert_with(|| {
                    let mut index = rules.len();
                    let name = loop {
                        let name = format!("__canonical_regex_{index}");
                        if names.insert(name.clone()) { break name; }
                        index += 1;
                    };
                    rules.push(NamedRule { name: name.clone(),
                        expr: GrammarExpr::RawRegex(pattern.clone()),
                        is_terminal: true, is_internal: false });
                    name
                }).clone();
                *expr = GrammarExpr::Ref(name);
            }
            GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) =>
                hoist(inner, names, aliases, rules),
            GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) =>
                for item in items { hoist(item, names, aliases, rules); },
            GrammarExpr::Exclude { expr, exclude } => {
                hoist(expr, names, aliases, rules); hoist(exclude, names, aliases, rules);
            }
            GrammarExpr::Intersect { expr, intersect } => {
                hoist(expr, names, aliases, rules); hoist(intersect, names, aliases, rules);
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (item, _) in items { hoist(item, names, aliases, rules); }
                hoist(separator, names, aliases, rules);
            }
            GrammarExpr::ExprNFA(nfa) =>
                for symbol in &mut nfa.symbols { hoist(symbol, names, aliases, rules); },
            _ => {}
        }
    }
    let mut names = grammar.rules.iter().map(|rule| rule.name.clone()).collect();
    let mut aliases = BTreeMap::new(); let mut rules = Vec::new();
    for rule in &mut grammar.rules {
        if !rule.is_terminal { hoist(&mut rule.expr, &mut names, &mut aliases, &mut rules); }
    }
    grammar.rules.extend(rules);
    glrmask_grammar::__private::grammar::glrm::to_glrm(&grammar)
}

#[derive(Deserialize)]
struct Trace { label: String, text: String, token_ids: Vec<u32> }

fn mask_hash(mask: &[u32]) -> u64 {
    mask.iter().fold(0xcbf29ce484222325u64, |hash, word|
        (hash ^ *word as u64).wrapping_mul(0x100000001b3))
}

// One arm runs alone from fresh states. The timer spans commitment of the
// previous token and the following full-vocabulary mask; checking and writing
// evidence are outside it. Prefix zero measures only the initial mask.
fn measure_arm(constraint: &Constraint, arm: &str, repeat: Option<usize>,
    traces: &[Trace], expected: &[Vec<(u64, bool)>], csv: &mut BufWriter<fs::File>)
    -> Result<(), Box<dyn std::error::Error>> {
    let mut mask = vec![0; constraint.mask_len()];
    for (index, trace) in traces.iter().enumerate() {
        let mut state = constraint.start();
        for step in 0..=trace.token_ids.len() {
            let previous = step.checked_sub(1).map(|step| trace.token_ids[step]);
            let started = Instant::now();
            let commit_started = Instant::now();
            let commit_result = previous.map(|token| state.commit_token(token));
            let commit_ns = if previous.is_some() { commit_started.elapsed().as_nanos() } else { 0 };
            let mask_started = Instant::now();
            state.fill_mask(&mut mask);
            let mask_ns = mask_started.elapsed().as_nanos();
            let tbm_ns = started.elapsed().as_nanos();
            if let Some(result) = commit_result { result?; }
            let hash = mask_hash(&mask);
            assert_eq!((hash, state.is_accepting()), expected[index][step],
                "isolated arm={arm} trace={index} step={step}");
            assert!(!state.is_rejected());
            if let Some(repeat) = repeat {
                writeln!(csv, "{arm},{repeat},{index},{step},{},{commit_ns},{mask_ns},{tbm_ns},{hash:016x}",
                    previous.map(|id| id.to_string()).unwrap_or_default())?;
            }
        }
        let eof = [b"<|end".as_slice(), b"oftext|>"].concat();
        state.commit_bytes(&eof)?; assert!(state.is_accepting());
    }
    csv.flush()?;
    eprintln!("ISOLATED arm={arm} repeat={repeat:?} positions=11767");
    Ok(())
}

fn read_vocab(path: &Path) -> Result<(Vocab, Vec<Vec<u8>>), Box<dyn std::error::Error>> {
    let data = fs::read(path)?; let mut offset = 0;
    let number = |offset: &mut usize| -> Result<u32, Box<dyn std::error::Error>> {
        let bytes = data.get(*offset..*offset + 4).ok_or("truncated vocabulary")?;
        *offset += 4; Ok(u32::from_le_bytes(bytes.try_into()?))
    };
    let count = number(&mut offset)?; if count != 128256 { return Err("selected10 requires the complete 128256-entry vocabulary".into()); }
    let mut entries = Vec::new(); let mut by_id = vec![Vec::new(); count as usize];
    for _ in 0..count {
        let id = number(&mut offset)?; let length = number(&mut offset)? as usize;
        let bytes = data.get(offset..offset + length).ok_or("truncated vocabulary bytes")?.to_vec(); offset += length;
        if id >= count { return Err("unexpected selected10 vocabulary coordinate".into()); }
        by_id[id as usize] = bytes.clone(); entries.push((id, bytes));
    }
    if offset != data.len() { return Err("vocabulary has trailing bytes".into()); }
    Ok((Vocab::new(entries), by_id))
}

fn assert_table_free(report: &serde_json::Value) {
    assert_eq!(report["lr_table_present"], false);
    if let Some(children) = report["component_parsers"].as_array() {
        assert_eq!(report["packed_lr_compiler_table_present"], false);
        for child in children { assert_table_free(child); }
    }
}

fn assert_static(report: &serde_json::Value) {
    if let Some(children) = report["component_parsers"].as_array() {
        assert_eq!(report["dynamic_boundary_shards"], 0, "a static request retained a dynamic boundary: {report}");
        assert!(report["static_boundary_shards"].as_u64().unwrap_or(0) > 0, "{report}");
        for child in children { assert_static(child); }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(args.len() == 5 || (args.len() == 6 && args[5] == "--measure-runtime"))
        || !matches!(args[3].as_str(), "static" | "dynamic") {
        return Err("usage: template_precompiled_composition FIXTURE_DIR OUTPUT_DIR static|dynamic JS_GRAMMAR [--measure-runtime]".into());
    }
    let fixture = Path::new(&args[1]); let output = Path::new(&args[2]);
    if output.exists() { return Err("output directory exists; preserve earlier evidence and use a new directory".into()); }
    fs::create_dir_all(output)?;
    let (vocab, token_bytes) = read_vocab(&fixture.join("vocab_dump.bin"))?;
    let traces: Vec<Trace> = serde_json::from_slice(&fs::read(fixture.join("traces.json"))?)?;
    assert_eq!(traces.iter().map(|trace| trace.token_ids.len() + 1).sum::<usize>(), 11767);
    for trace in &traces {
        let decoded = trace.token_ids.iter().flat_map(|&id| token_bytes[id as usize].iter().copied()).collect::<Vec<_>>();
        assert_eq!(decoded, trace.text.as_bytes(), "canonical trace byte identity");
    }
    eprintln!("LOAD canonical selected10, traces={} positions=11767", traces.len());
    let options = BuildOptions::default()
        .optimization(if args[3] == "static" { Optimization::FastRuntime } else { Optimization::FastBuild })
        .parser_backend(ParserBackend::TemplateDfa);
    // Prepare the complete native child before starting the link timer. Schema
    // lowering and child compilation are reported separately, never hidden.
    let started = Instant::now();
    let mut dispatch = String::from("start suffix;\n");
    let mut schemas = Vec::new();
    for index in 0..10 {
        dispatch.push_str(&format!("extern grammar ARGS_{index};\n"));
        let schema = serde_json::from_slice(&fs::read(fixture.join(format!("schema-{index:02}.json")))?)?;
        let named = glrmask_json_schema::schema_to_named_grammar(&schema)?;
        let mut factored = glrmask_grammar::__private::grammar::factoring::factor_named_grammar(named);
        glrmask_json_schema::prepare_named_grammar_for_dump(&mut factored)?;
        let schema_source = source_for_reparse(factored);
        fs::write(output.join(format!("schema-{index:02}-source.glrm")), &schema_source)?;
        schemas.push((format!("ARGS_{index}"), schema_source));
    }
    dispatch.push_str("nt suffix ::=\n");
    for index in 0..10 {
        if index != 0 { dispatch.push_str("  | "); }
        dispatch.push_str(&format!("\".tool_{index}(\" ARGS_{index} \")\"\n"));
    }
    dispatch.push_str(";\n");
    let bindings = schemas.iter().map(|(name, source)| (name.as_str(), source.as_str())).collect::<Vec<_>>();
    let named = glrmask_grammar::__private::grammar::glrm::from_glrm_with_inline_subgrammars(&dispatch, &bindings)?;
    let child_source = source_for_reparse(named);
    fs::write(output.join("child-source.glrm"), &child_source)?;
    let child = Grammar::from_glrm(&child_source).compile_with(&vocab, options.clone())?;
    let child_compile_ns = started.elapsed().as_nanos();
    let child_report = parser_backend_report(&child); assert_table_free(&child_report);
    fs::write(output.join("child-backend.json"),serde_json::to_vec_pretty(&child_report)?)?;
    eprintln!("CHILD source_prepare_compile_ns={child_compile_ns}");
    let source_path = Path::new(&args[4]);
    let mut source = fs::read_to_string(source_path)?.replace("\r\n", "\n");
    let needle = "nt member_expression_with_suffixes ::=\n    primary_expression";
    assert!(source.contains(needle), "canonical JS source shape changed");
    source = source.replacen(needle, "nt member_expression_with_suffixes ::=\n    'tools' PROGRAMMATIC_TOOL_SUFFIX\n  | primary_expression", 1);
    source.push_str("\nextern grammar PROGRAMMATIC_TOOL_SUFFIX;\n");
    fs::write(output.join("parent-source.glrm"), &source)?;
    let started = Instant::now(); let parent = Grammar::from_glrm(&source).compile_unlinked(&vocab)?;
    let parent_compile_ns = started.elapsed().as_nanos();
    eprintln!("PARENT source_compile_ns={parent_compile_ns}");
    fs::write(output.join("prepared-child-backend.json"),serde_json::to_vec_pretty(&child_report)?)?;
    let child_artifact=child.save();
    eprintln!("PREPARED_CHILD_ARTIFACT bytes={}",child_artifact.len());
    fs::write(output.join("prepared-child-artifact.bin"),child_artifact)?;
    fs::write(output.join("prepared-parent-artifact.bin"),parent.save())?;
    eprintln!("PREPARED_METADATA child={}",child_report["compiler_effect_metadata"]);
    // The oracle compiles one fully resolved grammar. It never uses the native
    // component linker under test or historical LR-backed artifacts.
    let started = Instant::now();
    let mut inline_bindings = vec![("PROGRAMMATIC_TOOL_SUFFIX".to_owned(), dispatch)];
    inline_bindings.extend(schemas.iter().map(|(name, source)| (format!("PROGRAMMATIC_TOOL_SUFFIX::{name}"), source.clone())));
    let bindings = inline_bindings.iter().map(|(name, source)| (name.as_str(), source.as_str())).collect::<Vec<_>>();
    let named = glrmask_grammar::__private::grammar::glrm::from_glrm_with_inline_subgrammars(&source, &bindings)?;
    let inline_source = source_for_reparse(named);
    fs::write(output.join("inline-reference-source.glrm"), &inline_source)?;
    let reference = Grammar::from_glrm(&inline_source).compile_with(&vocab, options.clone())?;
    let reference_compile_ns = started.elapsed().as_nanos();
    assert_table_free(&parser_backend_report(&reference));
    eprintln!("INLINE_REFERENCE source_prepare_compile_ns={reference_compile_ns}");
    let start = Instant::now();
    let candidate = parent.bind("PROGRAMMATIC_TOOL_SUFFIX", &child)?.link_with(options.clone())?;
    let link_ns = start.elapsed().as_nanos(); eprintln!("DIRECT_LINK ns={link_ns}");
    assert_eq!(candidate.parser_backend(), ParserBackend::TemplateDfa);
    let report = parser_backend_report(&candidate); assert_table_free(&report); if args[3] == "static" { assert_static(&report); }
    let sharing = &report["component_sharing"];
    assert_eq!(sharing["shared_retained_sources"], sharing["checked_retained_terminal_relations"]);
    assert_eq!(sharing["shared_retained_domains"], sharing["checked_retained_terminal_relations"]);
    assert_eq!(sharing["dense_global_certificate_rows"], 0);
    fs::write(output.join("backend.json"), serde_json::to_vec_pretty(&report)?)?;
    let start = Instant::now(); let saved = candidate.save(); let save_ns = start.elapsed().as_nanos();
    if std::env::var_os("GLRMASK_PROFILE_METADATA_STORAGE").is_some() {
        let isolated=glrmask::__private::save_without_effect_metadata_for_diagnostic(&candidate);
        let checked=Constraint::load(&isolated)?;assert_table_free(&parser_backend_report(&checked));
        let footprint=serde_json::json!({"with_metadata_bytes":saved.len(),"without_optional_effect_metadata_bytes":isolated.len(),
            "actual_artifact_difference_bytes":saved.len() as i64-isolated.len() as i64,
            "isolated_clone_only":true,"original_artifact_and_runtime_unchanged":true});
        fs::write(output.join("effect-metadata-storage.json"),serde_json::to_vec_pretty(&footprint)?)?;
        eprintln!("EFFECT_METADATA_STORAGE {footprint}");
    }
    let start = Instant::now(); let loaded = Constraint::load(&saved)?; let load_ns = start.elapsed().as_nanos();
    assert_table_free(&parser_backend_report(&loaded)); if args[3] == "static" { assert_static(&parser_backend_report(&loaded)); } assert_eq!(saved, loaded.save());
    let external = candidate.save_with_external_vocab()?; let external_loaded = Constraint::load_with_vocab(&external, &vocab)?;
    assert_table_free(&parser_backend_report(&external_loaded)); if args[3] == "static" { assert_static(&parser_backend_report(&external_loaded)); }
    let mut csv = BufWriter::new(fs::File::create(output.join("exact-replay.csv"))?);
    writeln!(csv, "trace,step,token,inline_mask_ns,template_mask_ns,self_load_mask_ns,external_load_mask_ns,hash")?;
    let mut reference_mask = vec![0; reference.mask_len()]; let mut mask = vec![0; candidate.mask_len()];
    assert_eq!(reference_mask.len(), mask.len()); let mut samples = 0usize;
    let mut expected = Vec::with_capacity(traces.len());
    for (index, trace) in traces.iter().enumerate() {
        let mut trace_expected = Vec::with_capacity(trace.token_ids.len() + 1);
        let mut oracle = reference.start(); let mut fresh = candidate.start();
        let mut reloaded = loaded.start(); let mut ext = external_loaded.start();
        for step in 0..=trace.token_ids.len() {
            let started = Instant::now(); oracle.fill_mask(&mut reference_mask); let inline_ns = started.elapsed().as_nanos();
            let mut times = [0u128; 3];
            for (representation, state) in [&mut fresh, &mut reloaded, &mut ext].into_iter().enumerate() {
                let started = Instant::now(); state.fill_mask(&mut mask); times[representation] = started.elapsed().as_nanos();
                if mask != reference_mask || state.is_accepting() != oracle.is_accepting() {
                    let differences = mask.iter().zip(&reference_mask).enumerate().flat_map(|(word, (&a, &b))|
                        (0..32).filter_map(move |bit| (((a ^ b) & (1 << bit)) != 0).then_some(word * 32 + bit))).take(16).collect::<Vec<_>>();
                    csv.flush()?; return Err(format!("mismatch trace={} step={step} representation={representation} tokens={differences:?}", trace.label).into());
                }
            }
            let hash = mask_hash(&reference_mask);
            trace_expected.push((hash, oracle.is_accepting()));
            let token = trace.token_ids.get(step).copied();
            writeln!(csv, "{index},{step},{},{inline_ns},{},{},{},{hash:016x}", token.map(|id| id.to_string()).unwrap_or_default(), times[0], times[1], times[2])?;
            if let Some(token) = token {
                assert_ne!(mask[token as usize / 32] & (1 << (token % 32)), 0, "canonical token not admitted");
                oracle.commit_token(token)?; fresh.commit_token(token)?; reloaded.commit_token(token)?; ext.commit_token(token)?;
            }
            samples += 1;
            if step % 512 == 0 { csv.flush()?; eprintln!("TRACE {index} {}/{}", step, trace.token_ids.len()); }
        }
        // This historical JS fixture is `statement_list? EOF`, with a literal
        // legacy EOF terminal. The saved trajectories intentionally stop just
        // before that terminal. Qualify both the prefix and explicit completion
        // rather than silently dropping a failing end-of-program assertion.
        let eof = [b"<|end".as_slice(), b"oftext|>"].concat();
        assert!(!oracle.is_accepting(), "the canonical trajectory omits its literal EOF");
        oracle.commit_bytes(&eof)?;
        oracle.fill_mask(&mut reference_mask);
        assert!(oracle.is_accepting(), "the canonical program plus its explicit EOF must accept");
        for state in [&mut fresh, &mut reloaded, &mut ext] {
            state.commit_bytes(&eof)?; state.fill_mask(&mut mask);
            assert!(state.is_accepting()); assert_eq!(mask, reference_mask);
        }
        expected.push(trace_expected);
    }
    csv.flush()?;
    let mut link_samples = vec![link_ns];
    let runtime_measurement = if args.len() == 6 {
        for repeat in 0..2 {
            let started = Instant::now();
            let linked = parent.bind("PROGRAMMATIC_TOOL_SUFFIX", &child)?.link_with(options.clone())?;
            link_samples.push(started.elapsed().as_nanos());
            assert_eq!(linked.save(), saved, "repeated prepared-component link must preserve exact artifact bytes");
            eprintln!("ISOLATED_LINK repeat={repeat} ns={}", link_samples.last().unwrap());
        }
        let mut measurements = BufWriter::new(fs::File::create(output.join("isolated-runtime.csv"))?);
        writeln!(measurements, "arm,repeat,trace,step,token,commit_ns,mask_ns,tbm_ns,hash")?;
        for (name, constraint) in [("inline", &reference), ("composed", &candidate)] {
            measure_arm(constraint, name, None, &traces, &expected, &mut measurements)?;
        }
        for repeat in 0..2 {
            let arms = if repeat == 0 { [("inline", &reference), ("composed", &candidate)] }
                else { [("composed", &candidate), ("inline", &reference)] };
            for (name, constraint) in arms {
                measure_arm(constraint, name, Some(repeat), &traces, &expected, &mut measurements)?;
            }
        }
        Some(serde_json::json!({"complete":true,"arms":["inline","composed"],
            "warmup_replays_per_arm":1,"measured_replays_per_arm":2,"positions_per_replay":samples,
            "measured_rows":samples * 4,"clock":"Instant wall time",
            "tbm":"previous token commit followed immediately by full128256-vocabulary mask; prefix0 mask only",
            "order":"warmup inline/composed; repeat0 inline/composed; repeat1 composed/inline",
            "note":"quiet same-process arms, one heavy lane; includes timer overhead; distinct from historical boundary-only thread-CPU B"}))
    } else { None };
    let result = serde_json::json!({"mode":args[3],"positions":samples,"candidate_representations":3,
        "complete":true,"masks_commits_completion_equal":true,"explicit_eof_checks":traces.len(),"link_ns":link_ns,
        "child_compile_ns":child_compile_ns,"parent_compile_ns":parent_compile_ns,
        "reference_compile_ns":reference_compile_ns,"reference":"independent grammar-inlined native compiler",
        "save_ns":save_ns,"load_ns":load_ns,"prepared_link_samples_ns":link_samples,
        "boundary_candidate_tokens":report["boundary_candidate_tokens"],
        "component_sharing":report["component_sharing"],"runtime_measurement":runtime_measurement,
        "self_bytes":saved.len(),"external_bytes":external.len(),
        "timing_note":"Correctness-run observations, not an isolated performance qualification"});
    fs::write(output.join("summary.json"), serde_json::to_vec_pretty(&result)?)?;
    println!("{result}"); Ok(())
}
