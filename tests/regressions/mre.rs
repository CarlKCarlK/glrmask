use std::{env, ffi::OsString, sync::{Mutex, MutexGuard}};

use glrmask::{Constraint as Constraint, Vocab};
use glrmask::__private::{ConstraintExt as _, ConstraintStateExt as _, parser_backend_report};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var_os(key);
        unsafe {
            env::set_var(key, value);
        }
        if key == "GLRMASK_LLGUIDANCE_COMPAT" {
            let enabled = value != "0" && !value.is_empty();
            glrmask::Constraint::set_test_compat_mode(enabled);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        let original_enabled = match &self.original {
            Some(value) => unsafe {
                env::set_var(self.key, value);
                let val = value.to_string_lossy();
                val != "0" && !val.is_empty()
            },
            None => unsafe {
                env::remove_var(self.key);
                false
            },
        };
        if self.key == "GLRMASK_LLGUIDANCE_COMPAT" {
            glrmask::Constraint::set_test_compat_mode(original_enabled);
        }
    }
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn unique_parser_stack_count(state: &glrmask::ConstraintState<'_>) -> usize {
    let mut stacks = state
        .debug_parser_stacks()
        .into_iter()
        .flat_map(|(_, stacks)| stacks.into_iter().map(|(stack, _)| stack))
        .collect::<Vec<_>>();
    stacks.sort_unstable();
    stacks.dedup();
    stacks.len()
}

fn total_final_stack_count(stacks: &[(u32, Vec<Vec<u32>>)]) -> usize {
    stacks.iter().map(|(_, stacks)| stacks.len()).sum()
}


#[test]
fn glrm_ignore_prefix_token_is_mask_commit_equivalent() {
    let vocab = Vocab::new(
        vec![
            (0, b"if".to_vec()),
            (1, b"(".to_vec()),
            (2, b" (".to_vec()),
            (3, b"true".to_vec()),
            (4, b")".to_vec()),
            (5, b" ".to_vec()),
        ]);
    let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+ ;
nt start ::= 'if' '(' 'true' ')' ;
"#;

    let constraint = Constraint::compile(glrmask::Grammar::glrm(grammar), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_token(0).unwrap();

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 2),
        "token b\" (\" should be admitted after b\"if\" because it is WS followed by '('"
    );

    state.commit_token(2).unwrap();
}

#[test]
fn glrm_initial_ignore_is_epsiloned_after_ti() {
    let vocab = Vocab::new(
        vec![
            (0, b" ".to_vec()),
            (1, b"a".to_vec()),
            (2, b" a".to_vec()),
        ]);
    let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+ ;
nt start ::= 'a' ;
"#;

    let constraint = Constraint::compile(glrmask::Grammar::glrm(grammar), &vocab).unwrap();

    let mut separate_token_state = constraint.start();
    assert!(
        token_allowed(&separate_token_state.mask(), 0),
        "an initial ignore token must be admitted by the parser DWA"
    );
    separate_token_state.commit_token(0).unwrap();
    assert!(
        token_allowed(&separate_token_state.mask(), 1),
        "committing the initial ignore token must leave the parser ready for 'a'"
    );
    separate_token_state.commit_token(1).unwrap();

    let mut combined_token_state = constraint.start();
    assert!(
        token_allowed(&combined_token_state.mask(), 2),
        "a token beginning with ignore then 'a' must survive post-TI ignore epsilon conversion"
    );
    combined_token_state.commit_token(2).unwrap();
}

#[test]
fn certified_terminal_run_collapse_preserves_bounded_item_count() {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"aba".to_vec()),
            (4, b"abab".to_vec()),
        ]);
    let grammar = r#"
start start;
t A ::= "a";
t B ::= "b";
nt item ::= A | B;
nt start ::= item item? item?;
"#;

    let constraint = Constraint::compile(glrmask::Grammar::glrm(grammar), &vocab).unwrap();
    let mut state = constraint.start();
    let initial = state.mask();
    assert!(token_allowed(&initial, 2), "two items in one token must be admitted");
    assert!(token_allowed(&initial, 3), "three items in one token must be admitted");
    assert!(
        !token_allowed(&initial, 4),
        "four items must not be admitted by a grammar capped at three"
    );

    state.commit_token(2).unwrap();
    let after_two = state.mask();
    assert!(token_allowed(&after_two, 0));
    assert!(token_allowed(&after_two, 1));
    state.commit_token(0).unwrap();
}

fn byte_vocab_with_separator_token() -> (Vocab, u32) {
    let mut entries: Vec<(u32, Vec<u8>)> = (0u32..=255).map(|byte| (byte, vec![byte as u8])).collect();
    let separator_token_id = 256;
    entries.push((separator_token_id, b" \"".to_vec()));
    (Vocab::new(entries), separator_token_id)
}

#[test]
fn glrm_subtraction_space_escaped_quote_control_accepts_token() {
    let grammar = r#"
start s;

nt s ::= "\"" "a" ws+ non_ws;
t ws_char ::= " ";
nt ws ::= ws_char;
nt non_ws ::= json_char - ws_char;
t json_char ::= /(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfnrt]|\\u00(?:[01][0-9A-Fa-f]|7[Ff]))/;
"#;

    let token_id = 0u32;
    let end_of_text = 1u32;
    let vocab = Vocab::new(
        vec![
            (token_id, b" \\\"".to_vec()),
            (end_of_text, b"<|endoftext|>".to_vec()),
        ]);

    let constraint = Constraint::compile(glrmask::Grammar::glrm(grammar), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"\"a").unwrap();

    // Control: this handcrafted GLRM still admits the token.
    assert!(token_allowed(&state.mask(), token_id as usize));
}

#[test]
fn glrm_dumped_constrained_terminal_space_escaped_quote_gap_one_token_vocab() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _compat = EnvVarGuard::set("GLRMASK_LLGUIDANCE_COMPAT", "1");

    let token_id = 0u32;
    let vocab = Vocab::new(
        vec![
            (token_id, b"a \\\"".to_vec()),
        ]);

    let grammar = r####"
        start s;
        nt s ::= X "$";
        t X ::= NON_WS WS NON_WS;
        t NON_WS ::= CHAR - WS;
        t CHAR ::= "a" | "\\\"";
        t WS ::= "\\n" | " ";
    "####;
    let constraint = Constraint::compile(glrmask::Grammar::glrm(&grammar), &vocab).unwrap();
    let state = constraint.start();
    assert!(token_allowed(&state.mask(), token_id as usize));


    let grammar = r####"
        start s;
        nt s ::= X "$";
        t X ::= NON_WS+ WS+ NON_WS;
        t NON_WS ::= CHAR - WS;
        t CHAR ::= "a" | "\\\"";
        t WS ::= "\\n" | " ";
    "####;
    let constraint = Constraint::compile(glrmask::Grammar::glrm(&grammar), &vocab).unwrap();
    let state = constraint.start();
    assert!(token_allowed(&state.mask(), token_id as usize));


    let grammar = r####"
        start s;
        nt s ::= X "$";
        t X ::= (NON_WS+ WS+)? NON_WS;
        t NON_WS ::= CHAR - WS;
        t CHAR ::= "a" | "\\\"";
        t WS ::= "\\n" | " ";
    "####;
    let constraint = Constraint::compile(glrmask::Grammar::glrm(&grammar), &vocab).unwrap();
    let state = constraint.start();
    // Regression: after `a `, the optional body may finish before `\`, allowing the
    // suffix NON_WS to consume `\"`. The regex-suffix optimizer used to greedily
    // continue WS as a possible "\\n" and drop the suffix path.
    assert!(token_allowed(&state.mask(), token_id as usize));


    let grammar = r####"
        start s;
        nt s ::= X "$";
        t X ::= (NON_WS+ WS+)? NON_WS;
        t NON_WS ::= CHAR - WS;
        t CHAR ::= "a" | "\\\"";
        t WS ::= " ";
    "####;
    let constraint = Constraint::compile(glrmask::Grammar::glrm(&grammar), &vocab).unwrap();
    let state = constraint.start();
    assert!(token_allowed(&state.mask(), token_id as usize));
}

#[test]
fn bounded_repeat_suffix_must_not_greedily_drop_suffix_path() {
    let token_id = 0u32;
    let vocab = Vocab::new(vec![(token_id, b"aa".to_vec())]);

    let grammar = r####"
        start s;
        nt s ::= X "$";
        t X ::= ("a"+)? "a";
    "####;

    let constraint = Constraint::compile(glrmask::Grammar::glrm(grammar), &vocab).unwrap();
    let state = constraint.start();

    // Semantically valid:
    //
    //   ("a"+)?  consumes the first "a"
    //   "a"      consumes the second "a"
    //
    // The buggy bounded-repeat-with-suffix optimizer instead keeps extending
    // the optional "a"+ body on the second "a" and drops the valid suffix path.
    assert!(token_allowed(&state.mask(), token_id as usize));
}

#[test]
fn optional_choice_allows_viable_prefix_before_required_suffix() {
    let token_a = 0u32;
    let token_b = 1u32;
    let token_ab = 2u32;
    let vocab = Vocab::new(
        vec![
            (token_a, b"a".to_vec()),
            (token_b, b"b".to_vec()),
            (token_ab, b"ab".to_vec()),
        ]);

    let grammar = r####"
        start s;
        nt s ::= X "$";
        t X ::= "a"* "b" | "";
    "####;

    let constraint = Constraint::compile(glrmask::Grammar::glrm(grammar), &vocab).unwrap();
    let mut state = constraint.start();
    let mask = state.mask();

    // "a" is not a complete value for `a*b`, but it is a viable token prefix:
    // after committing it, a later "b" can complete the terminal.
    assert!(token_allowed(&mask, token_a as usize));
    assert!(token_allowed(&mask, token_b as usize));
    assert!(token_allowed(&mask, token_ab as usize));

    state.commit_token(token_a).unwrap();
    let mask = state.mask();
    assert!(token_allowed(&mask, token_b as usize));
    state.commit_token(token_b).unwrap();
}

#[test]
fn chunk16_bounded_service_name_allows_spaces_token_after_open_quote() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _chunk = EnvVarGuard::set("GLRMASK_STRING_REPEAT_CHUNK", "16");

    let schema = r#"{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "serviceName": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100
            }
        },
        "required": ["serviceName"]
    }"#;
    let prefix = br#"{"serviceName": ""#;
    let vocab = Vocab::new(vec![(0, vec![b' '; 24])]);

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let parser = parser_backend_report(&constraint);
    assert_eq!(parser["backend"], "acyclic-template-dfa");
    assert_eq!(parser["lr_table_present"], false, "native runtime retained LR storage: {parser}");
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();

    assert!(token_allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
}

#[test]
fn minimized_sp343_separator_wave_matches_profile_oracle() {
    let _lock = env_lock();
    let _trace = EnvVarGuard::set("GLRMASK_PROFILE_ADVANCE_TRACE", "1");

    let schema = r#"{
        "properties": {
            "failure": {
                "properties": {
                    "messages": {
                        "items": {
                            "anyOf": [
                                {
                                    "additionalProperties": false,
                                    "properties": {
                                        "error": { "type": "string" },
                                        "field": {}
                                    },
                                    "required": ["field"]
                                },
                                {
                                    "additionalProperties": false,
                                    "properties": {
                                        "error": {},
                                        "schemaKey": {}
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    }"#;
    let prefix = b"{\"failure\": {\"messages\": [{\"error\": \"Json parsing issue\",";
    let (vocab, separator_token_id) = byte_vocab_with_separator_token();

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();

    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();

    assert_eq!(
        unique_parser_stack_count(&state),
        1,
        "recognizer-equivalent lexer residuals should carry one parser stack: {:?}",
        state.debug_parser_stacks(),
    );

    let mask = state.mask();
    assert!(token_allowed(&mask, separator_token_id as usize));

    assert_eq!(unique_parser_stack_count(&state), 1);

    let (advances, final_stacks, commit_profile) = state.commit_token_per_advance(separator_token_id).unwrap();

    // The separator token may complete the relevant terminal and produce a
    // final stack without requiring a parser advance. The oracle here is not
    // "exactly one advance"; it is "no parser ambiguity / nondeterministic
    // wave, and one final stack".
    assert_eq!(advances.len(), 0, "advances={advances:#?}");
    assert_eq!(total_final_stack_count(&final_stacks), 1);
    assert_eq!(
        commit_profile.adv_n_nondet_waves, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );
    assert_eq!(
        commit_profile.adv_n_nondet_reduce_ops, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );
    assert_eq!(
        commit_profile.adv_n_nondet_merges, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );
    assert_eq!(
        commit_profile.adv_n_nondet_isolates, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );

    for (i, advance) in advances.iter().enumerate() {
        assert_eq!(advance.profile.n_nondet_waves, 0);
        assert_eq!(advance.profile.n_nondet_reduce_ops, 0);
        assert_eq!(advance.profile.n_nondet_merges, 0);
        assert_eq!(advance.profile.n_nondet_isolates, 0);
        assert_eq!(i, 0, "unexpected multiple advances: {advances:#?}");
        assert_eq!(advance.gss_stacks_before.len(), 1);
        assert_eq!(advance.gss_stacks_after.len(), 1);
    }

    assert_eq!(total_final_stack_count(&final_stacks), 1);
}

#[test]
fn sp343_delete_only_subset_separator_wave_matches_cfa_oracle() {
    let _lock = env_lock();

    // Deletion-only subset of CFA `jsb/data/Snowplow---sp_343_Normalized`.
    // This keeps the same `failure.messages[*]` branches that produce the
    // full-case separator ambiguity, and deletes unrelated root properties,
    // descriptions, and the unused first-branch `value` property. No branch is
    // invented from scratch.
    let schema = r#"{"additionalProperties":false,"properties":{"failure":{"additionalProperties":false,"properties":{"messages":{"items":{"anyOf":[{"additionalProperties":false,"properties":{"error":{"type":"string"},"field":{"maxLength":64,"type":"string"}},"required":["field","error"],"type":"object"},{"additionalProperties":false,"properties":{"error":{"anyOf":[{"additionalProperties":false,"properties":{"error":{"enum":["ResolutionError"]},"lookupHistory":{"items":{"properties":{"attempts":{"minimum":0,"type":"integer"},"errors":{"items":{"properties":{"error":{"enum":["RepoFailure","ClientFailure","NotFound"]},"message":{"maxLength":256,"type":"string"}},"required":["error"],"type":"object"},"minItems":1,"type":"array"},"lastAttempt":{"_format":"date-time","type":"string"},"repostitory":{"type":"string"}},"required":["repository","errors","attempts","lastAttempt"],"type":"object"},"minItems":1,"type":"array"}},"required":["error","lookupHistory"]},{"additionalProperties":false,"properties":{"dataReports":{"items":{"additionalProperties":false,"properties":{"keyword":{"type":["string","null"]},"message":{"type":"string"},"path":{"type":["string","null"]},"targets":{"type":["array","null"]}},"required":["path","message","keyword","targets"]},"minItems":1,"type":"array"},"error":{"enum":["ValidationError"]}},"required":["dataReports"]},{"additionalProperties":false,"properties":{"error":{"enum":["ValidationError"]},"schemaIssues":{"items":{"additionalProperties":false,"properties":{"message":{"type":"string"},"path":{"type":"string"}},"required":["path","message"],"type":"object"},"minItems":1,"type":"array"}},"required":["error"]}]},"schemaKey":{"type":"string"}},"type":"object"}]},"type":"array"}},"required":["messages"],"type":"object"}},"required":["failure"],"type":"object"}"#;
    let prefix = b"{\"failure\": {\"messages\": [{\"error\": \"Json parsing issue\",";
    let (vocab, separator_token_id) = byte_vocab_with_separator_token();

    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();

    assert!(token_allowed(&state.mask(), separator_token_id as usize));
    assert_eq!(
        unique_parser_stack_count(&state),
        1,
        "delete-only Snowplow subset should keep one parser stack across lexer residuals: {:?}",
        state.debug_parser_stacks(),
    );

    let (advances, final_stacks, commit_profile) =
        state.commit_token_per_advance(separator_token_id).unwrap();

    // The separator token may complete the relevant terminal and produce a
    // final stack without requiring a parser advance. The oracle here is not
    // "exactly one advance"; it is "no parser ambiguity / nondeterministic
    // wave, and one final stack".
    assert_eq!(advances.len(), 0, "advances={advances:#?}");
    assert_eq!(total_final_stack_count(&final_stacks), 1);

    assert_eq!(
        commit_profile.adv_n_nondet_waves, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );
    assert_eq!(
        commit_profile.adv_n_nondet_reduce_ops, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );
    assert_eq!(
        commit_profile.adv_n_nondet_merges, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );
    assert_eq!(
        commit_profile.adv_n_nondet_isolates, 0,
        "commit_profile={commit_profile:#?} advances={advances:#?} final_stacks={final_stacks:#?}",
    );

    for advance in &advances {
        assert_eq!(advance.profile.n_nondet_waves, 0);
        assert_eq!(advance.profile.n_nondet_reduce_ops, 0);
        assert_eq!(advance.profile.n_nondet_merges, 0);
    }
}

#[test]
fn kb304_nullable_enum_bare_quote_requires_canonical_separator_space() {
    // CFA `jsb/data/Kubernetes---kb_304_Normalized` reports this exact frontier:
    // prefix `{"apiVersion":`, token id 22 / bytes `"`.
    //
    // The current JSON-schema lowering uses canonical separators with exactly
    // one trailing space after `:` and `,`, so the quote is valid after
    // `{"apiVersion": `, not immediately after `{"apiVersion":`.
    let schema = r#"{
        "type": "object",
        "properties": {
            "apiVersion": {
                "type": ["string", "null"],
                "enum": ["v1"]
            }
        }
    }"#;
    let token_id = 22u32;
    let token_bytes = b"\"";
    let vocab = Vocab::new(vec![(token_id, token_bytes.to_vec())]);
    let constraint = Constraint::compile(glrmask::Grammar::json_schema(schema), &vocab).unwrap();

    let invalid_prefix = br#"{"apiVersion":"#;
    let canonical_prefix = br#"{"apiVersion": "#;

    let mut no_space_quote_state = constraint.start();
    no_space_quote_state.commit_bytes(invalid_prefix).unwrap();
    let no_space_quote_accepts = no_space_quote_state.commit_bytes(token_bytes).is_ok();

    let mut no_space_full_state = constraint.start();
    no_space_full_state.commit_bytes(invalid_prefix).unwrap();
    let no_space_full_accepts = no_space_full_state.commit_bytes(br#""v1""#).is_ok();

    assert!(
        !no_space_quote_accepts,
        "without the canonical separator space, bare quote should not be accepted",
    );
    assert!(
        !no_space_full_accepts,
        "without the canonical separator space, full enum value should not be accepted",
    );

    let mut mask_state = constraint.start();
    mask_state.commit_bytes(canonical_prefix).unwrap();
    let mask_contains = token_allowed(&mask_state.mask(), token_id as usize);

    let mut bytes_state = constraint.start();
    bytes_state.commit_bytes(canonical_prefix).unwrap();
    let commit_bytes_accepts = bytes_state.commit_bytes(token_bytes).is_ok();

    let mut token_state = constraint.start();
    token_state.commit_bytes(canonical_prefix).unwrap();
    let commit_token_accepts = token_state.commit_token(token_id).is_ok();

    let mut full_value_state = constraint.start();
    full_value_state.commit_bytes(canonical_prefix).unwrap();
    let commit_full_accepts = full_value_state.commit_bytes(br#""v1""#).is_ok();

    let mut null_state = constraint.start();
    null_state.commit_bytes(canonical_prefix).unwrap();
    let commit_null_accepts = null_state.commit_bytes(b"null").is_ok();

    eprintln!(
        "kb304 canonical separator truth table: \
         no_space_quote_accepts={no_space_quote_accepts} \
         no_space_full_accepts={no_space_full_accepts} \
         mask_contains={mask_contains} \
         commit_token_accepts={commit_token_accepts} \
         commit_bytes_accepts={commit_bytes_accepts} \
         commit_full_accepts={commit_full_accepts} \
         commit_null_accepts={commit_null_accepts}",
    );

    assert!(commit_bytes_accepts);
    assert!(commit_full_accepts);
    assert!(mask_contains);
    assert!(commit_token_accepts);

    // JSON Schema `enum` restricts the accepted values. Even though the schema
    // says type ["string", "null"], enum ["v1"] excludes null.
    assert!(!commit_null_accepts);
}

#[test]
fn bounded_analysis_byte_classes_must_not_merge_distinct_vocab_tokens() {
    const INVALID: u32 = 1;

    fn spelling(value: usize) -> Vec<u8> {
        const ALPHABET: &[u8] =
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        vec![
            ALPHABET[value % ALPHABET.len()],
            ALPHABET[(value / ALPHABET.len()) % ALPHABET.len()],
            ALPHABET[(value / ALPHABET.len().pow(2)) % ALPHABET.len()],
        ]
    }

    // The faulty prequotient activates at 50,000 distinct vocabulary
    // spellings. All generated fillers are three-byte alphanumeric strings, so
    // active-language canonicalization reduces the local vocabulary to exactly
    // the two relevant spellings: two bytes (`aa`) and three bytes (`aaa`).
    let mut entries = vec![(0, b"aa".to_vec()), (INVALID, b"aaa".to_vec())];
    let mut value = 0usize;
    while entries.len() < 50_000 {
        let filler = spelling(value);
        value += 1;
        if filler == b"aaa" {
            continue;
        }
        entries.push((entries.len() as u32, filler));
    }
    let vocab = Vocab::new(entries);
    let grammar = r#"
start S;
lexer group x ::= X;
t X ::= /./;
nt S ::= X X "!";
"#;

    let constraint = Constraint::compile(glrmask::Grammar::glrm(grammar), &vocab).unwrap();
    let mask = constraint.start().mask();
    assert!(token_allowed(&mask, 0));
    assert!(!token_allowed(&mask, INVALID as usize));

    let mut byte_state = constraint.start();
    assert!(byte_state.commit_bytes(b"aaa").is_err());
}
