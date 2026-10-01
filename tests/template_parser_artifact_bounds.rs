//! Field-aware malformed-program regressions for both template runtimes.
//! These tests check structural rejection, not authentication of artifact data.
use glrmask::template_parser::{
    LexerDefinition, ParserDefinition, ParserProgram, StackLabel, StackTemplate,
    TemplateBuildOptions, TerminalPattern,
};
use glrmask::{Constraint, Optimization, ParserBackend, Vocab};
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"(".to_vec()),
        (1, b")".to_vec()),
        (2, b"()".to_vec()),
        (3, b"((".to_vec()),
        (7, b" ".to_vec()),
        (11, b"()".to_vec()),
    ])
}

fn fixture(optimization: Optimization, external: bool) -> Vec<u8> {
    let program = ParserProgram::new(ParserDefinition {
        stack_symbol_count: 2,
        terminals: vec![
            StackTemplate::read_top_and_push([0, 1], [1]),
            StackTemplate::rewrite([StackLabel::Symbol(1)], []),
            StackTemplate::identity(),
        ],
        completion: StackTemplate::read_top_and_push([0], []),
    })
    .unwrap();
    let lexer = LexerDefinition::new(vec![
        TerminalPattern::literal(b"(".to_vec()),
        TerminalPattern::literal(b")".to_vec()),
        TerminalPattern::regex("[ ]+"),
    ])
    .ignoring(2);
    let constraint = program
        .compile_with(
            &lexer,
            &vocab(),
            TemplateBuildOptions::default().optimization(optimization),
        )
        .unwrap();
    if external {
        constraint.save_with_external_vocab().unwrap()
    } else {
        constraint.save()
    }
}

fn parser_range(bytes: &[u8]) -> Range<usize> {
    assert!(bytes.starts_with(b"GLRCONS\0"));
    assert_eq!(&bytes[18..22], b"S30\0");
    let length =
        |i: usize| u64::from_le_bytes(bytes[22 + i * 8..30 + i * 8].try_into().unwrap()) as usize;
    let start = 110 + length(0) + length(1);
    start..start + length(2)
}

fn replace_parser(bytes: &[u8], parser: &[u8]) -> Vec<u8> {
    let range = parser_range(bytes);
    let mut result = bytes[..range.start].to_vec();
    result.extend_from_slice(parser);
    result.extend_from_slice(&bytes[range.end..]);
    result[38..46].copy_from_slice(&(parser.len() as u64).to_le_bytes());
    let payload_len = (result.len() - 18) as u64;
    result[10..18].copy_from_slice(&payload_len.to_le_bytes());
    result
}

fn read_var(bytes: &[u8], offset: &mut usize) -> (u32, Range<usize>) {
    let start = *offset;
    let mut value = 0;
    for shift in (0..=28).step_by(7) {
        let byte = bytes[*offset];
        *offset += 1;
        value |= u32::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return (value, start..*offset);
        }
    }
    panic!("the generated fixture must use canonical u32 varints");
}

fn replace_var(bytes: &[u8], range: Range<usize>, mut value: u32) -> Vec<u8> {
    let mut result = bytes[..range.start].to_vec();
    while value >= 128 {
        result.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    result.push(value as u8);
    result.extend_from_slice(&bytes[range.end..]);
    result
}

fn load(bytes: Vec<u8>, external: bool) -> glrmask::Result<Constraint> {
    if external {
        Constraint::load_with_vocab(bytes, &vocab())
    } else {
        Constraint::load(bytes)
    }
}

fn assert_rejected(original: &[u8], parser: &[u8], external: bool, label: &str) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        load(replace_parser(original, parser), external)
    }));
    assert!(result.is_ok(), "loader panicked for {label}");
    assert!(
        result.unwrap().is_err(),
        "accepted malformed field: {label}"
    );
}

#[test]
fn every_compact_graph_phase_and_link_rejects_out_of_bounds_fields() {
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        for external in [false, true] {
            let original = fixture(optimization, external);
            let parser = &original[parser_range(&original)];
            let header = if external { 36 } else { 0 };
            assert_eq!(&parser[header..header + 4], b"TPR2");
            let mut offset = header + 4;
            let (alphabet, _) = read_var(parser, &mut offset);
            let (terminals, _) = read_var(parser, &mut offset);
            let (skips, _) = read_var(parser, &mut offset);
            for _ in 0..skips {
                read_var(parser, &mut offset);
            }
            let mut mutations = Vec::new();
            let mut phases_with_edges = [false; 3];
            let mut links_checked = 0;
            for program in 0..=terminals {
                let mut sizes = [0u32; 3];
                for phase in 0..3 {
                    let (states, _) = read_var(parser, &mut offset);
                    sizes[phase] = states;
                    if states == 0 {
                        continue;
                    }
                    let (_, start_range) = read_var(parser, &mut offset);
                    mutations.push((
                        start_range,
                        states,
                        format!("program{program}/phase{phase}/start"),
                    ));
                    for state in 0..states {
                        let (flags, flag_range) = read_var(parser, &mut offset);
                        mutations.push((
                            flag_range,
                            u32::MAX,
                            format!("program{program}/phase{phase}/row-count"),
                        ));
                        for _ in 0..flags >> 1 {
                            phases_with_edges[phase] = true;
                            let (_, label_range) = read_var(parser, &mut offset);
                            let (_, target_range) = read_var(parser, &mut offset);
                            let label = format!("program{program}/phase{phase}/state{state}");
                            mutations.push((label_range, alphabet + 1, format!("{label}/label")));
                            mutations.push((
                                target_range.clone(),
                                states,
                                format!("{label}/missing-target"),
                            ));
                            mutations.push((target_range, state, format!("{label}/self-cycle")));
                        }
                    }
                }
                for (source, target) in [(0, 1), (0, 2), (1, 2)] {
                    let (count, range) = read_var(parser, &mut offset);
                    mutations.push((
                        range,
                        sizes[source] + 1,
                        format!("program{program}/link-count"),
                    ));
                    for _ in 0..count {
                        let (_, range) = read_var(parser, &mut offset);
                        mutations.push((
                            range,
                            sizes[target] + 1,
                            format!("program{program}/link-target"),
                        ));
                        links_checked += 1;
                    }
                }
            }
            assert_eq!(offset, parser.len());
            assert!(phases_with_edges.into_iter().all(|present| present));
            assert!(links_checked > 0 && mutations.len() > 40);
            for (range, value, label) in mutations {
                assert_rejected(
                    &original,
                    &replace_var(parser, range, value),
                    external,
                    &label,
                );
            }
            // Failed loads must not poison thread-local state for later loads.
            let good = load(original, external).unwrap();
            assert_eq!(good.parser_backend(), ParserBackend::TemplateDfa);
            assert_eq!(
                glrmask::__private::parser_backend_report(&good)["lr_table_present"],
                false
            );
            let mut state = good.start();
            state.commit_token(2).unwrap();
            assert!(state.is_accepting());
        }
    }
}

#[test]
fn derived_certificate_limit_rejects_small_frames_before_large_allocation() {
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        for external in [false, true] {
            let original = fixture(optimization, external);
            let parser = &original[parser_range(&original)];
            let mut offset = if external { 40 } else { 4 };
            let (_, alphabet_range) = read_var(parser, &mut offset);
            let (terminals, _) = read_var(parser, &mut offset);
            let row_bytes = ((u64::from(terminals) + 1).div_ceil(64)) * 16 + 64;
            let first_too_large = (256 * 1024 * 1024 / row_bytes + 1) as u32;
            let invalid = replace_parser(
                &original,
                &replace_var(parser, alphabet_range, first_too_large),
            );
            let result = catch_unwind(AssertUnwindSafe(|| load(invalid, external)));
            assert!(
                result.is_ok(),
                "oversized certificates must be an error, not a panic"
            );
            let error = result.unwrap().unwrap_err();
            assert!(
                error.to_string().contains("budget"),
                "wrong early rejection: {error}"
            );
        }
    }
}

#[test]
fn external_binding_accepts_reordered_identical_mapping_but_not_id_changes() {
    let mapping = vocab()
        .iter()
        .map(|(id, bytes)| (id, bytes.to_vec()))
        .collect::<Vec<_>>();
    let reversed = Vocab::new(mapping.iter().cloned().rev().collect());
    let mut remapped = mapping.clone();
    remapped[0].0 = 23;
    let remapped = Vocab::new(remapped);
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        let bytes = fixture(optimization, true);
        let good = Constraint::load_with_vocab(bytes.clone(), &reversed).unwrap();
        assert_eq!(good.parser_backend(), ParserBackend::TemplateDfa);
        let mut state = good.start();
        state.commit_token(11).unwrap(); // duplicate token bytes, distinct token ID
        assert!(state.is_accepting());
        assert_eq!(good.save_with_external_vocab().unwrap(), bytes);
        assert!(Constraint::load_with_vocab(bytes, &remapped).is_err());
    }
}
