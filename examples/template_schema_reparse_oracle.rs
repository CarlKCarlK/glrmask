//! Compare schema-04's original JSON import with its separately emitted GLRM.
//! Usage: template_schema_reparse_oracle SCHEMA_JSON REPARSED_GLRM VOCAB_DUMP OUTPUT_JSON
use std::{fs, io::Write, path::Path, time::Instant};

use glrmask::{BuildOptions, Constraint, Optimization, ParserBackend, Vocab};
use glrmask::Grammar;
use glrmask::__private::parser_backend_report;
use serde_json::json;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn read_vocab(path: &Path) -> Result<(Vocab, usize)> {
    let data = fs::read(path)?;
    let mut offset = 0;
    let number = |offset: &mut usize| -> Result<u32> {
        let bytes = data.get(*offset..*offset + 4).ok_or("truncated vocabulary")?;
        *offset += 4;
        Ok(u32::from_le_bytes(bytes.try_into()?))
    };
    let count = number(&mut offset)? as usize;
    if count != 128256 {
        return Err("oracle requires the complete 128256-entry canonical vocabulary".into());
    }
    let mut entries = Vec::with_capacity(count);
    let mut seen = vec![false; count];
    for _ in 0..count {
        let id = number(&mut offset)?;
        let length = number(&mut offset)? as usize;
        let bytes = data.get(offset..offset + length).ok_or("truncated vocabulary bytes")?;
        offset += length;
        if id as usize >= count || seen[id as usize] {
            return Err("duplicate or out-of-range canonical vocabulary coordinate".into());
        }
        seen[id as usize] = true;
        entries.push((id, bytes.to_vec()));
    }
    if offset != data.len() || seen.iter().any(|present| !present) {
        return Err("incomplete vocabulary or trailing bytes".into());
    }
    Ok((Vocab::new(entries), count))
}

fn native_report(constraint: &Constraint) -> Result<serde_json::Value> {
    let report = parser_backend_report(constraint);
    if constraint.parser_backend() != ParserBackend::TemplateDfa
        || report["lr_table_present"] != false
    {
        return Err(format!("oracle materialized an unexpected backend: {report}").into());
    }
    Ok(report)
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 5 {
        return Err("usage: template_schema_reparse_oracle SCHEMA_JSON REPARSED_GLRM VOCAB_DUMP OUTPUT_JSON".into());
    }
    let output = Path::new(&args[4]);
    if output.exists() {
        return Err("output already exists; preserve earlier evidence".into());
    }
    let schema = fs::read_to_string(&args[1])?;
    let source = fs::read_to_string(&args[2])?;
    let (vocab, vocab_count) = read_vocab(Path::new(&args[3]))?;
    let options = BuildOptions::default()
        .optimization(Optimization::FastRuntime)
        .parser_backend(ParserBackend::TemplateDfa);
    let started = Instant::now();
    let original = Grammar::from_json_schema(&schema).compile_with(&vocab, options.clone())?;
    let original_compile_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let reparsed = Grammar::from_glrm(&source).compile_with(&vocab, options)?;
    let reparsed_compile_ns = started.elapsed().as_nanos();
    let original_report = native_report(&original)?;
    let reparsed_report = native_report(&reparsed)?;
    if original.mask_len() != reparsed.mask_len() {
        return Err("mask coordinates differ".into());
    }

    // Independent literal documents use schema-04's property order and exact
    // canonical separators. Both root and comma-prefixed regex atoms occur.
    let documents = [
        ("empty_object", r#"{}"#, true),
        ("root_layout_null", r#"{"layoutId": null}"#, true),
        ("root_layout_hex", r#"{"layoutId": "0123456789abcdefABCDEF01"}"#, true),
        ("root_layout_short_hex", r#"{"layoutId": "0123456789abcdefABCDEF0"}"#, false),
        ("root_layout_non_hex", r#"{"layoutId": "0123456789abcdefABCDEG01"}"#, false),
        ("root_tags_empty", r#"{"viewTags": {}}"#, true),
        ("root_tags_value", r#"{"viewTags": {"tag": "x"}}"#, true),
        ("root_tags_invalid_key", r#"{"viewTags": {"tag!": "x"}}"#, false),
        ("root_tags_empty_value", r#"{"viewTags": {"tag": ""}}"#, false),
        ("name_layout_null", r#"{"name": "x", "layoutId": null}"#, true),
        ("description_layout_null", r#"{"description": "x", "layoutId": null}"#, true),
        ("name_layout_hex", r#"{"name": "x", "layoutId": "0123456789abcdefABCDEF01"}"#, true),
        ("name_layout_short_hex", r#"{"name": "x", "layoutId": "0123456789abcdefABCDEF0"}"#, false),
        ("name_tags_empty", r#"{"name": "x", "viewTags": {}}"#, true),
        ("description_tags_empty", r#"{"description": "x", "viewTags": {}}"#, true),
        ("body_tags_value", r#"{"body": "x", "viewTags": {"tag": "x"}}"#, true),
        ("name_tags_invalid_key", r#"{"name": "x", "viewTags": {"tag!": "x"}}"#, false),
        ("combined_null_empty", r#"{"name": "x", "layoutId": null, "viewTags": {}}"#, true),
        ("combined_hex_value", r#"{"name": "x", "layoutId": "0123456789abcdefABCDEF01", "viewTags": {"tag": "x"}}"#, true),
        ("combined_null_invalid_key", r#"{"name": "x", "layoutId": null, "viewTags": {"tag!": "x"}}"#, false),
    ];
    let mut original_mask = vec![0u32; original.mask_len()];
    let mut reparsed_mask = vec![0u32; reparsed.mask_len()];
    let mut prefix_checks = 0usize;
    let mut byte_commit_checks = 0usize;
    let mut document_records = Vec::new();
    let checks_started = Instant::now();
    for (label, document, expected_accepting) in documents {
        let mut a = original.start();
        let mut b = reparsed.start();
        let mut first_rejected_prefix = None;
        for prefix_len in 0..=document.len() {
            if prefix_len > 0 {
                let byte = &document.as_bytes()[prefix_len - 1..prefix_len];
                let a_result = a.commit_bytes(byte);
                let b_result = b.commit_bytes(byte);
                if a_result.is_ok() != b_result.is_ok() {
                    return Err(format!("byte commit result differs: {label} prefix={prefix_len} original={a_result:?} reparsed={b_result:?}").into());
                }
                byte_commit_checks += 1;
            }
            // An invalid known token/byte can return Ok and enter a fail state.
            // Compare rejection explicitly, including every dead suffix prefix.
            if a.is_rejected() != b.is_rejected() || a.is_accepting() != b.is_accepting() {
                return Err(format!("byte prefix state differs: {label} prefix={prefix_len}").into());
            }
            if a.is_rejected() && first_rejected_prefix.is_none() {
                first_rejected_prefix = Some(prefix_len);
            }
            a.fill_mask(&mut original_mask);
            b.fill_mask(&mut reparsed_mask);
            if original_mask != reparsed_mask {
                let differences = original_mask.iter().zip(&reparsed_mask).enumerate()
                    .flat_map(|(word, (&x, &y))| (0..32).filter_map(move |bit|
                        (((x ^ y) & (1 << bit)) != 0).then_some(word * 32 + bit)))
                    .take(16).collect::<Vec<_>>();
                return Err(format!("mask differs: {label} prefix={prefix_len} token_ids={differences:?}").into());
            }
            prefix_checks += 1;
        }
        if a.is_accepting() != expected_accepting || a.is_rejected() == expected_accepting {
            return Err(format!("independent document expectation failed: {label} expected_accepting={expected_accepting} accepting={} rejected={}", a.is_accepting(), a.is_rejected()).into());
        }
        document_records.push(json!({
            "label": label, "text": document, "expected_accepting": expected_accepting,
            "byte_prefixes": document.len() + 1,
            "first_rejected_prefix": first_rejected_prefix,
        }));
        eprintln!("DOCUMENT {label}: {} exact byte prefixes", document.len() + 1);
    }

    // Bound the expensive all-admitted-token branches to lexical frontiers,
    // including both comma-prefixed atoms. No token is sampled or truncated at
    // these frontiers; every one of the 128256 coordinates is inspected.
    let frontiers = [
        r#"{"#,
        r#"{"layoutId": "#,
        r#"{"layoutId": n"#,
        r#"{"layoutId": "0123456789abcdefABCDEF01"#,
        r#"{"viewTags": "#,
        r#"{"viewTags": {"#,
        r#"{"viewTags": {"tag"#,
        r#"{"name": "x", "#,
        r#"{"name": "x", "layoutId": "#,
        r#"{"name": "x", "viewTags": "#,
        r#"{"name": "x", "layoutId": null, "viewTags": "#,
    ];
    let mut token_commit_checks = 0usize;
    let mut frontier_records = Vec::new();
    for (index, prefix) in frontiers.into_iter().enumerate() {
        let mut a = original.start();
        let mut b = reparsed.start();
        a.commit_bytes(prefix.as_bytes())?;
        b.commit_bytes(prefix.as_bytes())?;
        if a.is_rejected() || b.is_rejected() || a.is_accepting() != b.is_accepting() {
            return Err(format!("independent branch frontier is invalid: {prefix:?}").into());
        }
        a.fill_mask(&mut original_mask);
        b.fill_mask(&mut reparsed_mask);
        if original_mask != reparsed_mask {
            return Err(format!("branch frontier mask differs: {prefix:?}").into());
        }
        prefix_checks += 1;
        let mut admitted = 0usize;
        for token_id in 0..vocab_count {
            if original_mask[token_id / 32] & (1 << (token_id % 32)) == 0 {
                continue;
            }
            // Reconstruct independent fresh states, rather than carrying any
            // scratch caches or prior token branch into the next commitment.
            let mut fresh_a = original.start();
            let mut fresh_b = reparsed.start();
            fresh_a.commit_bytes(prefix.as_bytes())?;
            fresh_b.commit_bytes(prefix.as_bytes())?;
            let a_result = fresh_a.commit_token(token_id as u32);
            let b_result = fresh_b.commit_token(token_id as u32);
            if a_result.is_ok() != b_result.is_ok()
                || fresh_a.is_rejected() != fresh_b.is_rejected()
                || fresh_a.is_accepting() != fresh_b.is_accepting()
                || fresh_a.is_terminated() != fresh_b.is_terminated()
            {
                return Err(format!("token branch differs: frontier={index} token={token_id} original={a_result:?} reparsed={b_result:?}").into());
            }
            if a_result.is_err() || fresh_a.is_rejected() {
                return Err(format!("admitted token rejected: frontier={index} token={token_id} result={a_result:?}").into());
            }
            admitted += 1;
            token_commit_checks += 1;
        }
        frontier_records.push(json!({"prefix": prefix, "admitted_tokens_checked": admitted}));
        eprintln!("FRONTIER {index}: {admitted} exact admitted-token branches");
    }
    let evidence = json!({
        "complete": true,
        "schema_json": fs::canonicalize(&args[1])?,
        "reparsed_glrm": fs::canonicalize(&args[2])?,
        "vocab_dump": fs::canonicalize(&args[3])?,
        "optimization": "FastRuntime", "parser_backend": "TemplateDfa",
        "vocab_entries": vocab_count, "mask_words": original.mask_len(),
        "full_mask_comparisons": prefix_checks,
        "mask_words_compared": prefix_checks * original.mask_len(),
        "byte_commit_comparisons": byte_commit_checks,
        "admitted_token_commit_comparisons": token_commit_checks,
        "original_compile_ns": original_compile_ns,
        "reparsed_compile_ns": reparsed_compile_ns,
        "oracle_checks_ns": checks_started.elapsed().as_nanos(),
        "original_backend": original_report, "reparsed_backend": reparsed_report,
        "documents": document_records, "token_frontiers": frontier_records,
    });
    // Create evidence only after every comparison and independent expectation
    // passes; create_new also rejects a concurrently created output path.
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&evidence)?)?;
    file.write_all(b"\n")?;
    eprintln!("PASS: {prefix_checks} full masks, {byte_commit_checks} byte commits, {token_commit_checks} admitted-token commits");
    Ok(())
}
