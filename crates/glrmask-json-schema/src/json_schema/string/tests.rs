use regex::Regex;

use super::{
    clamp_absolute_single_fixed_width_repetition, preprocess_ascii_shorthand,
    plain_fully_anchored_ascii_literal, quoted_string_body_regex,
    string_pattern_as_body_regex, use_lazy_ordinary_bounded_string, FixedWidthRepeatClamp,
    JsonStringCompatMode, JsonStringContext, TEST_COMPAT_MODE,
};
use crate::json_schema::config::JsonSchemaConfig;

#[test]
fn publication_runner_name_is_not_misclassified_as_test_binary() {
    assert!(!super::executable_name_looks_like_test("final-jsb-glrmask"));
    assert!(super::executable_name_looks_like_test("integration-1234"));
    assert!(super::executable_name_looks_like_test("json_schema_test-1234"));
}

#[test]
fn explicit_llguidance_compat_overrides_test_binary_default() {
    // Cargo test binaries intentionally default to full JSON semantics, but
    // an explicit process-level compatibility policy must remain authoritative.
    // Test the parsing logic directly to avoid mutating process env in parallel.
    assert_eq!(
        super::explicit_compat_mode_from_value(Some("1")),
        Some(JsonStringCompatMode::LlGuidanceNative),
    );
    assert_eq!(
        super::explicit_compat_mode_from_value(Some("0")),
        Some(JsonStringCompatMode::JsonSchema),
    );
    assert_eq!(super::explicit_compat_mode_from_value(None), None);
}

#[test]
fn lazy_ordinary_bounded_strings_are_enabled_independent_of_bound() {
    let mut config = JsonSchemaConfig::default();
    assert!(!use_lazy_ordinary_bounded_string(&config));
    config.lazy_ordinary_bounded_strings = true;
    assert!(use_lazy_ordinary_bounded_string(&config));
}

#[test]
fn recognizes_only_plain_fully_anchored_ascii_literals_for_decoded_reasoning() {
    assert_eq!(
        plain_fully_anchored_ascii_literal("^PowerShell@1$"),
        Some("PowerShell@1")
    );
    assert_eq!(plain_fully_anchored_ascii_literal("^a-b_c/1$"), Some("a-b_c/1"));
    assert_eq!(plain_fully_anchored_ascii_literal("PowerShell@1"), None);
    assert_eq!(plain_fully_anchored_ascii_literal("^PowerShell.*$"), None);
    assert_eq!(plain_fully_anchored_ascii_literal(r"^PowerShell\@1$"), None);
    assert_eq!(plain_fully_anchored_ascii_literal("^a b$"), None);
    assert_eq!(plain_fully_anchored_ascii_literal("^é$"), None);
}

#[test]
fn fixed_width_repeat_clamp_matches_literal_language_intersection_exhaustively() {
    for (body, width) in [("a", 1usize), ("(?:ab)", 2), ("(?:ab|cd)", 2)] {
        for original_min in 0usize..=4 {
            for original_max in ((original_min + 1)..=6)
                .map(Some)
                .chain(std::iter::once(None))
            {
                // Exact-count repetitions can be canonicalized out of the
                // HIR and, more importantly, never need clamping: their one
                // decoded length is already classified as redundant or
                // disjoint by the preceding length-relation proof.
                let quantifier = match original_max {
                    Some(max) if max == original_min => format!("{{{original_min}}}"),
                    Some(max) => format!("{{{original_min},{max}}}"),
                    None => format!("{{{original_min},}}"),
                };
                let pattern = format!("^{body}{quantifier}$");
                let original = Regex::new(&pattern).unwrap();

                for min_length in 0usize..=12 {
                    for max_length in min_length..=12 {
                        let clamped = clamp_absolute_single_fixed_width_repetition(
                            &pattern,
                            min_length,
                            max_length,
                        )
                        .unwrap_or_else(|| panic!("single fixed-width variable repeat should clamp: pattern={pattern:?} min_length={min_length} max_length={max_length}"));
                        let rewritten = match &clamped {
                            FixedWidthRepeatClamp::Empty => None,
                            FixedWidthRepeatClamp::Pattern(pattern) => {
                                Some(Regex::new(pattern).unwrap())
                            }
                        };

                        for count in 0usize..=8 {
                            let candidate = if width == 1 {
                                "a".repeat(count)
                            } else {
                                "ab".repeat(count)
                            };
                            let expected = original.is_match(&candidate)
                                && (min_length..=max_length).contains(&candidate.chars().count());
                            let actual = rewritten
                                .as_ref()
                                .is_some_and(|regex| regex.is_match(&candidate));
                            assert_eq!(
                                actual, expected,
                                "pattern={pattern:?} min_length={min_length} max_length={max_length} count={count}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fixed_width_repeat_clamp_with_fixed_prefix_suffix_matches_intersection() {
    for (prefix, body, suffix, width) in [
        ("pre", "a", "post", 1usize),
        ("x", "(?:ab)", "yz", 2usize),
        ("é", "(?:ab|cd)", "Z", 2usize),
    ] {
        for original_min in 0usize..=3 {
            for original_max in ((original_min + 1)..=5)
                .map(Some)
                .chain(std::iter::once(None))
            {
                let quantifier = match original_max {
                    Some(max) => format!("{{{original_min},{max}}}"),
                    None => format!("{{{original_min},}}"),
                };
                let pattern = format!("^{prefix}{body}{quantifier}{suffix}$");
                let original = Regex::new(&pattern).unwrap();

                for min_length in 0usize..=18 {
                    for max_length in min_length..=18 {
                        let clamped = clamp_absolute_single_fixed_width_repetition(
                            &pattern,
                            min_length,
                            max_length,
                        )
                        .unwrap_or_else(|| panic!(
                            "single fixed-width repeat with fixed context should clamp: pattern={pattern:?} min_length={min_length} max_length={max_length}"
                        ));
                        let rewritten = match &clamped {
                            FixedWidthRepeatClamp::Empty => None,
                            FixedWidthRepeatClamp::Pattern(pattern) => {
                                Some(Regex::new(pattern).unwrap())
                            }
                        };

                        for count in 0usize..=7 {
                            let repeated = if width == 1 {
                                "a".repeat(count)
                            } else {
                                "ab".repeat(count)
                            };
                            let candidate = format!("{prefix}{repeated}{suffix}");
                            let decoded_len = candidate.chars().count();
                            let expected = original.is_match(&candidate)
                                && (min_length..=max_length).contains(&decoded_len);
                            let actual = rewritten
                                .as_ref()
                                .is_some_and(|regex| regex.is_match(&candidate));
                            assert_eq!(
                                actual, expected,
                                "pattern={pattern:?} min_length={min_length} max_length={max_length} count={count}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn preprocess_ascii_shorthand_rewrites_generic_word_shorthand() {
    assert_eq!(preprocess_ascii_shorthand(r"^\w+$"), r"^[A-Za-z0-9_]+$");
    assert_eq!(preprocess_ascii_shorthand(r"^[\w.-]+$"), r"^[A-Za-z0-9_.-]+$");
}

#[test]
fn preprocess_ascii_shorthand_preserves_escaped_word_shorthand() {
    assert_eq!(preprocess_ascii_shorthand(r"^\\w+$"), r"^\\w+$");
}

#[test]
fn lowered_bounded_free_text_pattern_rejects_leading_space_slash() {
    let body = string_pattern_as_body_regex(r"^$|(^(?:\S+\s+){0,19}\S+$)", JsonStringContext::Value).unwrap();
    let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

    assert!(regex.is_match(r#""REST API""#));
    assert!(!regex.is_match(r#"" /""#));
}

#[test]
fn lowered_optional_decimal_pattern_rejects_backslash_digit_string() {
    let body = string_pattern_as_body_regex(r"^$|^\d{1,15}(?:\.\d{1,5})?$", JsonStringContext::Value).unwrap();
    let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

    assert!(regex.is_match(r#""""#));
    assert!(regex.is_match(r#""123.45""#));
    assert!(!regex.is_match(r#""\\1""#));
}

#[test]
fn llguidance_value_pattern_terminal_regex_classes_reject_ascii_unicode_escape_spellings() {
    TEST_COMPAT_MODE.with(|cell| cell.set(JsonStringCompatMode::LlGuidanceNative));

    let body = string_pattern_as_body_regex(r"^[0-9a-f]{8}$", JsonStringContext::Value).unwrap();
    assert!(!body.contains(r"\u00"), "{body}");
    let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

    assert!(regex.is_match(r#""1234abcd""#));
    assert!(!regex.is_match(r#""\u0031234abcd""#));
    assert!(!regex.is_match(r#""\uC1234abcd""#));
}

#[test]
fn pattern_whitespace_table_matches_regex_syntax_unicode_space_class() {
    use regex_syntax::hir::{Class, HirKind};
    use regex_syntax::Parser;

    let hir = Parser::new().parse(r"\s").unwrap();
    let HirKind::Class(Class::Unicode(class)) = hir.kind() else {
        panic!(r"expected Unicode class for \s");
    };
    let mut actual = std::collections::BTreeSet::new();
    for range in class.ranges() {
        for codepoint in u32::from(range.start())..=u32::from(range.end()) {
            if let Some(ch) = char::from_u32(codepoint) {
                actual.insert(ch);
            }
        }
    }
    let expected = super::PATTERN_WHITESPACE_CHARS.iter().copied().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn json_schema_pattern_whitespace_classes_include_vertical_tab() {
    TEST_COMPAT_MODE.with(|cell| cell.set(JsonStringCompatMode::JsonSchema));

    let whitespace = string_pattern_as_body_regex(r"^\s$", JsonStringContext::Value).unwrap();
    let whitespace = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&whitespace))).unwrap();
    assert!(whitespace.is_match(r#""\u000b""#));

    let non_whitespace = string_pattern_as_body_regex(r"^\S$", JsonStringContext::Value).unwrap();
    let non_whitespace = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&non_whitespace))).unwrap();
    assert!(!non_whitespace.is_match(r#""\u000b""#));
}

#[test]
fn llguidance_value_pattern_whitespace_class_accepts_unicode_escape_spelling() {
    TEST_COMPAT_MODE.with(|cell| cell.set(JsonStringCompatMode::LlGuidanceNative));

    let body = string_pattern_as_body_regex(r"^\s$", JsonStringContext::Value).unwrap();
    let regex = Regex::new(&format!(r"^(?:{})$", quoted_string_body_regex(&body))).unwrap();

    assert!(regex.is_match(r#""\t""#));
    assert!(regex.is_match(r#""\u0009""#));
}
