//! First-class programmatic JavaScript tool calling.
//!
//! [`ProgrammaticJsCompiler`] separates the reusable JavaScript work from the
//! per-tool-set work:
//!
//! 1. compile the full JavaScript parent and a conservative dynamic-value
//!    expression grammar once per vocabulary;
//! 2. compile each tool's JSON Schema with the shared dynamic-value grammar at
//!    nested property/array value positions;
//! 3. compose those compiled schemas into a tool dispatcher, then link that
//!    dispatcher into `tools.<name>(...)` call sites in the full JS parent.
//!
//! Literal values still take the ordinary schema branch. Dynamic expressions
//! are deliberately excluded at the schema root so a computed expression
//! cannot replace the entire arguments object and bypass its shape.

use std::collections::BTreeSet;

use crate::compiler::composition::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend, compose_constraints_owned_parent_segmented,
};
use crate::{Constraint, GlrMaskError, Vocab};

const JAVASCRIPT_GLRM: &str = include_str!("programmatic_js/javascript.glrm");
const PARENT_PLACEHOLDER_NAME: &str = "PROGRAMMATIC_TOOL_SUFFIX";

const DYNAMIC_VALUE_RULES: &str = r#"
// Opaque runtime values only. These forms retrieve/produce a runtime value
// without embedding a statically visible literal as the value itself. In
// particular there are deliberately no arithmetic/logical/conditional tails:
// schema-aware constructions such as `cond ? "open" : "closed"` are lowered
// by the JSON-Schema layer, where each result arm can be checked.
nt dynamic_value_expression ::=
    dynamic_reference_expression
  | 'await' dynamic_reference_expression
  ;

nt dynamic_reference_expression ::=
    IDENTIFIER dynamic_reference_suffix*
  | 'this' dynamic_reference_suffix*
  | 'new' '.' 'target' dynamic_reference_suffix*
  | 'import' '(' assignment_expression ')' dynamic_reference_suffix*
  | 'import' '.' 'meta' dynamic_reference_suffix*
  ;

nt dynamic_reference_suffix ::=
    '[' expression ']'
  | '?.' '[' expression ']'
  | '.' IDENTIFIER
  | '?.' IDENTIFIER
  | '.' private_identifier
  | '?.' private_identifier
  | arguments
  | '?.' arguments
  | TEMPLATE_LITERAL
  ;
"#;



/// Reusable compiler for programmatic JavaScript tool calling.
#[derive(Debug)]
pub(crate) struct ProgrammaticJsCompiler {
    parent: Constraint,
    dynamic_value: Constraint,
    condition: Constraint,
    parent_placeholder_terminal: u32,
}

impl ProgrammaticJsCompiler {
    /// Compile all reusable programmatic-JavaScript components for `vocab`.
    pub fn new(vocab: &Vocab) -> crate::Result<Self> {
        let parent = Self::compile_parent(vocab)?;
        let dynamic_value = Self::compile_dynamic_value(vocab)?;
        let condition = Self::compile_condition(vocab)?;
        Self::from_components(parent, dynamic_value, condition)
    }

    /// Compile the reusable full-JavaScript parent containing the reserved
    /// `tools` dispatcher boundary. This is independent of any concrete tool
    /// schemas and may be built once per vocabulary.
    pub fn compile_parent(vocab: &Vocab) -> crate::Result<Constraint> {
        let placeholder_token_id =
            crate::import::external_placeholder_token_id_avoiding(vocab, std::iter::empty())?;
        Constraint::from_glrm_grammar(&programmatic_parent_source(placeholder_token_id)?, vocab)
    }

    /// Compile the reusable opaque-runtime-value expression subgrammar.
    pub fn compile_dynamic_value(vocab: &Vocab) -> crate::Result<Constraint> {
        Constraint::from_glrm_grammar(&dynamic_value_source()?, vocab)
    }

    /// Compile the reusable unrestricted JavaScript condition subgrammar used
    /// only as the test of schema-aware conditional expressions.
    pub fn compile_condition(vocab: &Vocab) -> crate::Result<Constraint> {
        Constraint::from_glrm_grammar(&condition_source()?, vocab)
    }

    /// Assemble a reusable compiler from independently compiled shared parts.
    /// This exists so build systems and benchmarks can time/cache each shared
    /// component separately without changing programmatic-tool semantics.
    pub fn from_components(
        parent: Constraint,
        dynamic_value: Constraint,
        condition: Constraint,
    ) -> crate::Result<Self> {
        let parent_placeholder_terminal = parent
            .terminal_display_names
            .iter()
            .position(|name| name == PARENT_PLACEHOLDER_NAME)
            .map(|index| index as u32)
            .ok_or_else(|| {
                GlrMaskError::Compilation(
                    "programmatic JavaScript parent has no dispatcher linker terminal".into(),
                )
            })?;
        Ok(Self {
            parent,
            dynamic_value,
            condition,
            parent_placeholder_terminal,
        })
    }

    /// The reusable dynamic-value child used by every schema compiled through
    /// this compiler.
    pub fn dynamic_value_constraint(&self) -> &Constraint {
        &self.dynamic_value
    }

    /// Compile one tool arguments schema. The schema root stays static; nested
    /// object-property and array-item values may be dynamic JS expressions.
    #[allow(deprecated)]
    pub fn compile_schema(&self, schema: &str, vocab: &Vocab) -> crate::Result<Constraint> {
        Constraint::from_json_schema_with_programmatic_values(
            schema,
            &self.dynamic_value,
            &self.condition,
            vocab,
        )
    }

    /// Compile a named tool dispatcher from already-compiled tool schemas.
    /// This is separate from the outer JavaScript link so callers can time or
    /// configure the two composition stages independently.
    pub fn compile_dispatcher(
        &self,
        tools: &[(&str, &Constraint)],
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        validate_tool_names(tools.iter().map(|(name, _)| *name))?;
        if tools.is_empty() {
            return Err(GlrMaskError::Compilation(
                "programmatic JavaScript requires at least one tool".into(),
            ));
        }

        let dispatcher_source = dispatcher_source(tools.iter().map(|(name, _)| *name));
        let bindings = tools
            .iter()
            .enumerate()
            .map(|(index, (_, schema))| (format!("args_{index}"), *schema))
            .collect::<Vec<_>>();
        let borrowed = bindings
            .iter()
            .map(|(name, schema)| (name.as_str(), *schema))
            .collect::<Vec<_>>();
        Constraint::from_glrm_grammar_with_subgrammars(&dispatcher_source, &borrowed, vocab)
    }

    /// Link a compiled tool dispatcher into the reusable full-JavaScript parent.
    pub fn compose_dispatcher(
        &self,
        dispatcher: &Constraint,
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        compose_constraints_owned_parent_segmented(
            self.parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: self.parent_placeholder_terminal,
                additional_placeholder_terminals: &[],
                constraint: dispatcher,
            }],
            vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .map(|composition| composition.constraint)
        .map_err(GlrMaskError::Compilation)
    }

    /// Compose already-compiled tool schemas into a named dispatcher and link
    /// it into the reusable full-JavaScript parent.
    pub fn compose_tools(
        &self,
        tools: &[(&str, &Constraint)],
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        let dispatcher = self.compile_dispatcher(tools, vocab)?;
        self.compose_dispatcher(&dispatcher, vocab)
    }

    /// Convenience path that compiles every schema and then composes the full
    /// tool-calling constraint. Use [`Self::compile_schema`] and
    /// [`Self::compose_tools`] separately when build-phase timings matter.
    pub fn compile_tools(
        &self,
        tools: &[(&str, &str)],
        vocab: &Vocab,
    ) -> crate::Result<Constraint> {
        validate_tool_names(tools.iter().map(|(name, _)| *name))?;
        let compiled = tools
            .iter()
            .map(|(name, schema)| {
                self.compile_schema(schema, vocab)
                    .map(|constraint| (*name, constraint))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let borrowed = compiled
            .iter()
            .map(|(name, constraint)| (*name, constraint))
            .collect::<Vec<_>>();
        self.compose_tools(&borrowed, vocab)
    }
}

/// Canonical full JavaScript grammar bundled with GLRMask's programmatic tool
/// calling support. It omits CFA's textual EOF sentinel; model end-token IDs
/// remain a separate runtime concern.
pub(crate) fn javascript_glrm() -> &'static str {
    JAVASCRIPT_GLRM
}

fn reserve_tools_identifier(grammar: &mut crate::grammar::ast::NamedGrammar) -> crate::Result<()> {
    use crate::grammar::ast::GrammarExpr;

    let identifier = grammar
        .rules
        .iter_mut()
        .find(|rule| rule.name == "IDENTIFIER")
        .ok_or_else(|| {
            GlrMaskError::Compilation("bundled JavaScript grammar has no IDENTIFIER terminal".into())
        })?;
    identifier.expr = GrammarExpr::Exclude {
        expr: Box::new(identifier.expr.clone()),
        exclude: Box::new(GrammarExpr::Literal(b"tools".to_vec())),
    };
    Ok(())
}

fn pruned_javascript_source(start_rule: &str, extra_rules: &str) -> crate::Result<String> {
    let mut source = JAVASCRIPT_GLRM.to_owned();
    let start = "start program;";
    if !source.starts_with(start) {
        return Err(GlrMaskError::Compilation(
            "bundled JavaScript grammar start rule changed unexpectedly".into(),
        ));
    }
    source.replace_range(..start.len(), &format!("start {start_rule};"));
    source.push_str(extra_rules);

    let mut grammar = crate::grammar::glrm::from_glrm(&source)?;
    reserve_tools_identifier(&mut grammar)?;
    // The generic reachability pass follows grammar references from `start`,
    // while `ignore IGNORE` is metadata. Temporarily root IGNORE through a
    // synthetic rule so its lexical dependency closure survives pruning.
    let original_start = grammar.start.clone();
    let synthetic = "__ptc_prune_root".to_string();
    let mut roots = vec![crate::grammar::ast::GrammarExpr::Ref(original_start.clone())];
    if let Some(ignore) = grammar.ignore.clone() {
        roots.push(crate::grammar::ast::GrammarExpr::Ref(ignore));
    }
    grammar.rules.push(crate::grammar::ast::NamedRule {
        name: synthetic.clone(),
        expr: crate::grammar::ast::GrammarExpr::Choice(roots),
        is_terminal: false,
        is_internal: true,
    });
    grammar.start = synthetic.clone();
    crate::grammar::right_linear::retain_reachable_rules(&mut grammar);
    grammar.rules.retain(|rule| rule.name != synthetic);
    grammar.start = original_start;
    Ok(crate::grammar::glrm::to_glrm(&grammar))
}

fn dynamic_value_source() -> crate::Result<String> {
    pruned_javascript_source("dynamic_value_expression", DYNAMIC_VALUE_RULES)
}

fn condition_source() -> crate::Result<String> {
    pruned_javascript_source("coalesce_expression", "")
}

fn programmatic_parent_source(placeholder_token_id: u32) -> crate::Result<String> {
    use crate::grammar::ast::{GrammarExpr, NamedRule};

    let mut grammar = crate::grammar::glrm::from_glrm(JAVASCRIPT_GLRM)?;
    // `tools` is a reserved namespace in programmatic-tool mode. Without this
    // subtraction, `tools.foo(...)` can take the ordinary IDENTIFIER/call path
    // and bypass the schema dispatcher entirely.
    reserve_tools_identifier(&mut grammar)?;

    let member = grammar
        .rules
        .iter_mut()
        .find(|rule| rule.name == "member_expression_with_suffixes")
        .ok_or_else(|| GlrMaskError::Compilation(
            "bundled JavaScript grammar has no member_expression_with_suffixes rule".into(),
        ))?;
    let tool_expr = GrammarExpr::Sequence(vec![
        GrammarExpr::Literal(b"tools".to_vec()),
        GrammarExpr::Ref(PARENT_PLACEHOLDER_NAME.to_string()),
    ]);
    member.expr = match std::mem::replace(&mut member.expr, GrammarExpr::Epsilon) {
        GrammarExpr::Choice(mut alternatives) => {
            alternatives.insert(0, tool_expr);
            GrammarExpr::Choice(alternatives)
        }
        other => GrammarExpr::Choice(vec![tool_expr, other]),
    };
    grammar.rules.push(NamedRule {
        name: PARENT_PLACEHOLDER_NAME.to_string(),
        expr: GrammarExpr::SpecialToken(placeholder_token_id),
        is_terminal: true,
        is_internal: false,
    });
    Ok(crate::grammar::glrm::to_glrm(&grammar))
}

fn dispatcher_source<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names = names.into_iter().collect::<Vec<_>>();
    let mut source = String::new();
    for index in 0..names.len() {
        source.push_str(&format!("extern grammar args_{index};\n"));
    }
    source.push_str("start suffix;\nnt suffix ::=\n");
    for (index, name) in names.iter().enumerate() {
        let prefix = if index == 0 { "    " } else { "  | " };
        let head = serde_json::to_string(&format!(".{name}(")).expect("tool name is UTF-8");
        source.push_str(&format!("{prefix}{head} args_{index} ')'\n"));
    }
    source.push_str("  ;\n");
    source
}

fn validate_tool_names<'a>(names: impl IntoIterator<Item = &'a str>) -> crate::Result<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !is_ascii_identifier_name(name) {
            return Err(GlrMaskError::Compilation(format!(
                "programmatic tool name {name:?} is not an ASCII JavaScript identifier name"
            )));
        }
        if !seen.insert(name) {
            return Err(GlrMaskError::Compilation(format!(
                "programmatic tool name {name:?} was supplied more than once"
            )));
        }
    }
    Ok(())
}

fn is_ascii_identifier_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || matches!(first, b'_' | b'$'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        let pieces = [
            "tools", ".lookup(", ")", "{", "}", "customer_id", "\"customer_id\"",
            ": ", "customer", ".id", "\"abc\"", "123", "[", "]", ", ", "x", "+", "1",
        ];
        Vocab::new(
            pieces
                .iter()
                .enumerate()
                .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
                .collect(),
        )
    }

    fn accepts_bytes(constraint: &Constraint, bytes: &[u8]) -> bool {
        let mut state = constraint.start();
        state.commit_bytes(bytes).is_ok() && state.is_accepting()
    }

    #[test]
    fn dynamic_expression_grammar_excludes_bare_literals() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let dynamic = compiler.dynamic_value_constraint();
        assert!(accepts_bytes(dynamic, b"customer"));
        assert!(accepts_bytes(dynamic, b"customer.id"));
        assert!(!accepts_bytes(dynamic, b"x + 1"));
        assert!(!accepts_bytes(dynamic, b"123"));
        assert!(!accepts_bytes(dynamic, br#""abc""#));
        assert!(!accepts_bytes(dynamic, br#"{"wrong": 123}"#));
        assert!(!accepts_bytes(dynamic, br#"[1, 2]"#));
        assert!(!accepts_bytes(dynamic, b"tools.lookup()"));
    }

    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    fn programmatic_tool_schema_accepts_dynamic_property_value_and_unquoted_key() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"customer_id":{"type":"string"}},
          "required":["customer_id"],
          "additionalProperties":false
        }"#;
        let compiled_schema = compiler.compile_schema(schema, &vocab).unwrap();
        let constraint = compiler
            .compose_tools(&[("lookup", &compiled_schema)], &vocab)
            .unwrap();
        assert!(accepts_bytes(
            &constraint,
            b"tools.lookup({customer_id: customer.id});"
        ));
        assert!(accepts_bytes(
            &constraint,
            b"tools.lookup({customer_id:customer.id});"
        ));
        assert!(accepts_bytes(
            &constraint,
            b"tools.lookup({customer_id:\ncustomer.id});"
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({"customer_id": "abc"});"#
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({"customer_id":customer.id});"#
        ));
        assert!(!accepts_bytes(&constraint, b"tools.lookup(customer);"));
        assert!(!accepts_bytes(
            &constraint,
            b"tools.lookup({wrong: customer.id});"
        ));
    }

    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    fn programmatic_enum_allows_opaque_and_schema_checked_conditional() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"status":{"enum":["open","closed"]}},
          "required":["status"],
          "additionalProperties":false
        }"#;
        let constraint = compiler
            .compile_tools(&[("lookup", schema)], &vocab)
            .unwrap();

        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: customer.id});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? "open" : "closed"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? "open" : "bogus"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: "open" + x});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: "bogus"});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: customer.ready && other.flag ? "open" : "closed"});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? (customer.ready ? "open" : "closed") : "open"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: x ? (customer.ready ? "open" : "bogus") : "closed"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.unknown({status: "open"});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({status: tools.unknown({}) ? "open" : "closed"});"#,
        ));
    }

    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    fn programmatic_nested_object_and_array_values_stay_schema_aware() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{
            "meta":{
              "type":"object",
              "properties":{"status":{"enum":["open","closed"]}},
              "required":["status"],
              "additionalProperties":false
            },
            "ids":{"type":"array","items":{"type":"string"}}
          },
          "required":["meta","ids"],
          "additionalProperties":false
        }"#;
        let constraint = compiler
            .compile_tools(&[("lookup", schema)], &vocab)
            .unwrap();

        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: customer.status}, ids: [customer.id, other.id]});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: x ? "open" : "closed"}, ids: [customer.id]});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: "bogus"}, ids: [customer.id]});"#,
        ));
        assert!(accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: customer, ids: [customer.id]});"#,
        ));
        assert!(!accepts_bytes(
            &constraint,
            br#"tools.lookup({meta: {status: customer.status, extra: customer.id}, ids: [customer.id]});"#,
        ));
    }

    /// End-to-end regression for the explicit-static programmatic chain.
    ///
    /// Production `compile_schema`, `compile_dispatcher` and `compose_dispatcher`
    /// now compose explicit `StaticParserDwa` boundary links. The resulting
    /// constraint must be mask-for-mask equal to the unfiltered `Dynamic`
    /// oracle built over the SAME leaves at every accepted token boundary, and
    /// must reject the invalid controls.
    #[test]
    #[ignore = "Programmatic JSON schema API is intentionally unsupported"]
    fn programmatic_static_chain_matches_unfiltered_dynamic_regression() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"customer_id":{"type":"string"}},
          "required":["customer_id"],
          "additionalProperties":false
        }"#;
        let compiled_schema = compiler.compile_schema(schema, &vocab).unwrap();
        let dispatcher = compiler
            .compile_dispatcher(&[("lookup", &compiled_schema)], &vocab)
            .unwrap();

        // System under test: the production end-to-end static chain.
        let production_static = compiler.compose_dispatcher(&dispatcher, &vocab).unwrap();

        // Oracle: unfiltered Dynamic over the SAME (statically composed) leaves.
        // Deliberately not `compiler.compose_dispatcher`, which is now the static
        // system under test.
        let dynamic = compose_constraints_owned_parent_segmented(
            compiler.parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: compiler.parent_placeholder_terminal,
                additional_placeholder_terminals: &[],
                constraint: &dispatcher,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;

        // The all-static chain must not hide a DynamicDirect fallback.
        assert!(
            production_static.has_recursive_segmented_parser_tree(),
            "production static chain must install the recursive segmented runtime",
        );
        let overlay = production_static
            .static_dynamic_overlay
            .as_ref()
            .expect("production static chain must carry an overlay");
        assert!(
            !overlay.segmented_parser_components.is_empty(),
            "production static chain must have segmented components",
        );
        for component in &overlay.segmented_parser_components {
            if let Some(shard) = component.boundary.as_ref() {
                assert!(
                    !matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                    ),
                    "production static chain must not install a DynamicDirect fallback shard",
                );
            }
        }

        // While both paths are alive, masks must match exactly at every token
        // boundary. Returns the first chunk index at which either path stops
        // committing, plus both final acceptance verdicts.
        fn drive(
            static_c: &Constraint,
            dynamic_c: &Constraint,
            steps: &[&[u8]],
        ) -> (Option<usize>, bool, bool) {
            let mut a = static_c.start();
            let mut b = dynamic_c.start();
            let mut stopped = None;
            for (index, step) in steps.iter().enumerate() {
                assert_eq!(
                    a.mask(),
                    b.mask(),
                    "static vs dynamic mask mismatch before step {index} ({:?})",
                    std::str::from_utf8(step).unwrap_or("<binary>"),
                );
                let a_ok = a.commit_bytes(step).is_ok();
                let b_ok = b.commit_bytes(step).is_ok();
                if !a_ok || !b_ok {
                    stopped = Some(index);
                    break;
                }
            }
            (stopped, a.is_accepting(), b.is_accepting())
        }

        let accepted_quoted_continuation: &[&[u8]] = &[
            b"tools",
            b".lookup(",
            b"{",
            b"\"customer_id\"",
            b": ",
            b"\"abc\"",
            b"}",
            b")",
            b".id",
            b";",
        ];
        assert_eq!(
            drive(&production_static, &dynamic, accepted_quoted_continuation),
            (None, true, true),
            "quoted string argument plus post-call member suffix must commit and accept",
        );

        let accepted_customer_id: &[&[u8]] = &[
            b"tools",
            b".lookup(",
            b"{",
            b"\"customer_id\"",
            b": ",
            b"customer",
            b".id",
            b"}",
            b")",
            b";",
        ];
        assert_eq!(
            drive(&production_static, &dynamic, accepted_customer_id),
            (None, true, true),
            "dynamic property value must commit and accept",
        );

        let accepted_toolsx: &[&[u8]] = &[
            b"tools",
            b".lookup(",
            b"{",
            b"\"customer_id\"",
            b": ",
            b"tools",
            b"x",
            b"}",
            b")",
            b";",
        ];
        assert_eq!(
            drive(&production_static, &dynamic, accepted_toolsx),
            (None, true, true),
            "maximal-munch identifier split `tools` + `x` must commit and accept",
        );

        // Token 0 (`tools`) must be allowed at the value slot itself: it is the
        // first token of the accepted identifier `toolsx`.
        let value_slot_prefix = b"tools.lookup({\"customer_id\": ";
        for (label, constraint) in [("static", &production_static), ("dynamic", &dynamic)] {
            let mut state = constraint.start();
            state
                .commit_bytes(value_slot_prefix)
                .expect("value-slot prefix must commit");
            assert!(
                state.mask().first().is_some_and(|word| word & 1 != 0),
                "{label}: token 0 (`tools`) must be allowed at the value slot",
            );
        }

        // Invalid controls: masks are compared while both paths are alive and
        // rejection is required at some point (not necessarily the same chunk).
        let rejected: &[(&str, &[&[u8]])] = &[
            (
                "tools.id argument",
                &[
                    b"tools",
                    b".lookup(",
                    b"{",
                    b"\"customer_id\"",
                    b": ",
                    b"tools",
                    b".id",
                    b"}",
                    b")",
                    b";",
                ],
            ),
            (
                "unquoted tools.id",
                &[
                    b"tools",
                    b".lookup(",
                    b"{",
                    b"customer_id",
                    b": ",
                    b"tools",
                    b".id",
                    b"}",
                    b")",
                    b";",
                ],
            ),
            (
                "dispatcher call as value",
                &[
                    b"tools",
                    b".lookup(",
                    b"{",
                    b"\"customer_id\"",
                    b": ",
                    b"tools",
                    b".lookup(",
                    b"{",
                    b"}",
                    b")",
                    b"}",
                    b")",
                    b";",
                ],
            ),
            (
                "non-object argument",
                &[b"tools", b".lookup(", b"customer", b")", b";"],
            ),
            (
                "unknown property",
                &[
                    b"tools",
                    b".lookup(",
                    b"{",
                    b"wrong",
                    b": ",
                    b"customer",
                    b".id",
                    b"}",
                    b")",
                    b";",
                ],
            ),
        ];
        for &(label, steps) in rejected {
            let (stopped, static_acc, dynamic_acc) = drive(&production_static, &dynamic, steps);
            assert!(
                stopped.is_some(),
                "{label}: input must stop committing before completing",
            );
            assert!(
                !static_acc && !dynamic_acc,
                "{label}: input must not be accepted (static={static_acc}, dynamic={dynamic_acc})",
            );
        }

        // Serde roundtrip of the production composed constraint: identical
        // masks, acceptance and rejection after save/load.
        let saved = production_static.save();
        let reloaded = Constraint::load(&saved).expect("production static constraint must reload");
        assert_eq!(
            reloaded.start().mask(),
            production_static.start().mask(),
            "serde roundtrip must preserve the start mask",
        );
        let mut before = production_static.start();
        let mut after = reloaded.start();
        for step in accepted_quoted_continuation {
            assert_eq!(
                before.mask(),
                after.mask(),
                "serde roundtrip mask mismatch while committing accepted input",
            );
            assert_eq!(
                before.commit_bytes(step).is_ok(),
                after.commit_bytes(step).is_ok(),
                "serde roundtrip commit divergence on accepted input",
            );
        }
        assert!(before.is_accepting() && after.is_accepting());
        let mut rejected_before = production_static.start();
        let mut rejected_after = reloaded.start();
        for &step in rejected[0].1 {
            assert_eq!(
                rejected_before.mask(),
                rejected_after.mask(),
                "serde roundtrip mask mismatch on rejected input",
            );
            let a = rejected_before.commit_bytes(step).is_ok();
            let b = rejected_after.commit_bytes(step).is_ok();
            assert_eq!(a, b, "serde roundtrip commit divergence on rejected input");
            if !a {
                break;
            }
        }
        assert!(!rejected_before.is_accepting() && !rejected_after.is_accepting());
    }

    /// Asserting backend isolation on identical prepared schema leaves: the
    /// explicit `StaticParserDwa` and `Dynamic` schema compositions both retain
    /// the reserved-prefix token and agree; `None` is diagnostic only.
    #[test]
    fn programmatic_schema_explicit_backends_retain_reserved_prefix() {
        let vocab = vocab();
        let compiler = ProgrammaticJsCompiler::new(&vocab).unwrap();
        let schema = r#"{
          "type":"object",
          "properties":{"customer_id":{"type":"string"}},
          "required":["customer_id"],
          "additionalProperties":false
        }"#;
        let value_slot_prefix = b"{\"customer_id\": ";

        let mut value_slot_masks = Vec::new();
        for (label, backend) in [
            ("Dynamic", SegmentedBoundaryBackend::Dynamic),
            ("StaticParserDwa", SegmentedBoundaryBackend::StaticParserDwa),
        ] {
            let composed = Constraint::from_json_schema_with_programmatic_values_backend(
                schema,
                &compiler.dynamic_value,
                &compiler.condition,
                &vocab,
                Some(backend),
            )
            .expect("explicit backend schema composition must succeed");

            let mut state = composed.start();
            state
                .commit_bytes(value_slot_prefix)
                .expect("value-slot prefix must commit");
            assert!(
                state.mask().first().is_some_and(|word| word & 1 != 0),
                "{label}: token 0 (`tools`) must be allowed at the value slot",
            );
            value_slot_masks.push(state.mask());

            let mut toolsx = composed.start();
            assert!(toolsx.commit_bytes(value_slot_prefix).is_ok());
            assert!(toolsx.commit_bytes(b"tools").is_ok());
            assert!(toolsx.commit_bytes(b"x").is_ok());
            assert!(toolsx.commit_bytes(b"}").is_ok());
            assert!(toolsx.is_accepting(), "{label}: `toolsx` value must be accepted");

            let mut tools_id = composed.start();
            tools_id
                .commit_bytes(value_slot_prefix)
                .expect("value-slot prefix must commit");
            let reached_accepting = tools_id.commit_bytes(b"tools").is_ok()
                && tools_id.commit_bytes(b".id").is_ok()
                && tools_id.commit_bytes(b"}").is_ok()
                && tools_id.is_accepting();
            assert!(
                !reached_accepting,
                "{label}: bare reserved `tools.id` must be rejected",
            );
        }
        assert_eq!(
            value_slot_masks[0], value_slot_masks[1],
            "Dynamic and StaticParserDwa value-slot masks must agree",
        );

        // Legacy `None`: the disabled owned-parent path must now be rejected.
        let legacy_error = Constraint::from_json_schema_with_programmatic_values_backend(
            schema,
            &compiler.dynamic_value,
            &compiler.condition,
            &vocab,
            None,
        )
        .err()
        .expect("legacy None schema composition must be unsupported");
        match legacy_error {
            crate::GlrMaskError::Compilation(message) => assert!(
                message.contains("legacy owned-parent composition is unsupported"),
                "unexpected legacy None error: {message}",
            ),
            other => panic!("expected a Compilation error, got {other:?}"),
        }
    }
}
