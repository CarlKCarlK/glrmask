//! Real selected10 composition qualification using the existing canonical cache.
//! Usage: template_composition_control FIXTURE_DIR OUTPUT_DIR static|dynamic
//! FIXTURE_DIR contains core.bin, dispatch-literal.bin, vocab_dump.bin, traces.json.
//! All masks are compared in memory; only measurements and hashes are saved.
use std::{fs, path::Path, io::{BufWriter, Write}, time::Instant};
use glrmask::{Constraint, ParserBackend, Vocab};
use glrmask::__private::{ConstraintExt, into_template_parser, parser_backend_report};
use serde::Deserialize;

#[derive(Deserialize)]
struct Trace { label: String, text: String, token_ids: Vec<u32> }

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 || !matches!(args[3].as_str(), "static" | "dynamic") {
        return Err("usage: template_composition_control FIXTURE_DIR OUTPUT_DIR static|dynamic".into());
    }
    let fixture = Path::new(&args[1]); let output = Path::new(&args[2]); fs::create_dir_all(output)?;
    let (vocab, token_bytes) = read_vocab(&fixture.join("vocab_dump.bin"))?;
    let traces: Vec<Trace> = serde_json::from_slice(&fs::read(fixture.join("traces.json"))?)?;
    assert_eq!(traces.iter().map(|trace| trace.token_ids.len() + 1).sum::<usize>(), 11767);
    for trace in &traces {
        let decoded = trace.token_ids.iter().flat_map(|&id| token_bytes[id as usize].iter().copied()).collect::<Vec<_>>();
        assert_eq!(decoded, trace.text.as_bytes(), "canonical trace byte identity");
    }
    eprintln!("LOAD canonical selected10, traces={} positions=11767", traces.len());
    let core = Constraint::load_with_vocab(fs::read(fixture.join("core.bin"))?, &vocab)?;
    let dispatch = Constraint::load_with_vocab(fs::read(fixture.join("dispatch-literal.bin"))?, &vocab)?;
    let start = Instant::now();
    let reference = if args[3] == "static" {
        core.compose_compiled_subgrammars(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], &vocab)?
    } else {
        core.compose_compiled_subgrammars_dynamic(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], &vocab)?
    };
    let link_ns = start.elapsed().as_nanos(); eprintln!("LINK mode={} ns={link_ns}", args[3]);
    let start = Instant::now(); let candidate = into_template_parser(reference.clone())?;
    let conversion_ns = start.elapsed().as_nanos(); eprintln!("CONVERT ns={conversion_ns}");
    assert_eq!(candidate.parser_backend(), ParserBackend::TemplateDfa);
    let report = parser_backend_report(&candidate); assert_table_free(&report);
    fs::write(output.join("backend.json"), serde_json::to_vec_pretty(&report)?)?;
    let start = Instant::now(); let saved = candidate.save(); let save_ns = start.elapsed().as_nanos();
    let start = Instant::now(); let loaded = Constraint::load(&saved)?; let load_ns = start.elapsed().as_nanos();
    assert_table_free(&parser_backend_report(&loaded)); assert_eq!(saved, loaded.save());
    let external = candidate.save_with_external_vocab()?; let external_loaded = Constraint::load_with_vocab(&external, &vocab)?;
    assert_table_free(&parser_backend_report(&external_loaded));
    let mut csv = BufWriter::new(fs::File::create(output.join("exact-replay.csv"))?);
    writeln!(csv, "trace,step,token,lr_mask_ns,template_mask_ns,self_load_mask_ns,external_load_mask_ns,hash")?;
    let mut reference_mask = vec![0; reference.mask_len()]; let mut mask = vec![0; candidate.mask_len()];
    assert_eq!(reference_mask.len(), mask.len()); let mut samples = 0usize;
    for (index, trace) in traces.iter().enumerate() {
        let mut oracle = reference.start(); let mut fresh = candidate.start();
        let mut reloaded = loaded.start(); let mut ext = external_loaded.start();
        for step in 0..=trace.token_ids.len() {
            let started = Instant::now(); oracle.fill_mask(&mut reference_mask); let lr_ns = started.elapsed().as_nanos();
            let mut times = [0u128; 3];
            for (representation, state) in [&mut fresh, &mut reloaded, &mut ext].into_iter().enumerate() {
                let started = Instant::now(); state.fill_mask(&mut mask); times[representation] = started.elapsed().as_nanos();
                if mask != reference_mask || state.is_accepting() != oracle.is_accepting() {
                    let differences = mask.iter().zip(&reference_mask).enumerate().flat_map(|(word, (&a, &b))|
                        (0..32).filter_map(move |bit| (((a ^ b) & (1 << bit)) != 0).then_some(word * 32 + bit))).take(16).collect::<Vec<_>>();
                    csv.flush()?; return Err(format!("mismatch trace={} step={step} representation={representation} tokens={differences:?}", trace.label).into());
                }
            }
            let hash = mask.iter().fold(0xcbf29ce484222325u64, |hash, word| (hash ^ *word as u64).wrapping_mul(0x100000001b3));
            let token = trace.token_ids.get(step).copied();
            writeln!(csv, "{index},{step},{},{lr_ns},{},{},{},{hash:016x}", token.map(|id| id.to_string()).unwrap_or_default(), times[0], times[1], times[2])?;
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
    }
    csv.flush()?;
    let result = serde_json::json!({"mode":args[3],"positions":samples,"candidate_representations":3,
        "complete":true,"masks_commits_completion_equal":true,"explicit_eof_checks":traces.len(),"link_ns":link_ns,
        "conversion_ns":conversion_ns,"save_ns":save_ns,"load_ns":load_ns,
        "self_bytes":saved.len(),"external_bytes":external.len(),
        "timing_note":"Correctness-run observations, not an isolated performance qualification"});
    fs::write(output.join("summary.json"), serde_json::to_vec_pretty(&result)?)?;
    println!("{result}"); Ok(())
}
