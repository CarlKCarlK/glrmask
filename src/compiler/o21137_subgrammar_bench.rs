use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

use crate::automata::lexer::tokenizer::Lexer;
use crate::compiler::glr::table::GlrTableConstruction;
use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};
use crate::{Constraint, Vocab};

const SCHEMA_PATH: &str = "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/maskbench/data/Github_ultra---o21137.json";
const VOCAB_PATH: &str = "/Users/isaacbreen/Projects2/constraint-framework-analysis/.cache/vocab_cache/llama3_vocab.json";

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn load_wrapper() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(SCHEMA_PATH).expect("read o21137 wrapper"))
        .expect("parse o21137 wrapper")
}

fn import_o21137_grammar(wrapper: &serde_json::Value) -> GrammarDef {
    let schema = wrapper.get("schema").expect("schema field");
    let named = crate::import::json_schema::schema_to_named_grammar(schema).expect("schema import");
    let mut factored = crate::grammar::factoring::factor_named_grammar(named);
    crate::import::json_schema::prepare_named_grammar(&mut factored).expect("schema prepare");
    crate::grammar::ast::lower(&factored).expect("grammar lower")
}

fn decode_hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0, "hex token has odd length");
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid hex digit {byte:?}"),
            };
            (digit(pair[0]) << 4) | digit(pair[1])
        })
        .collect()
}

fn load_llama3_vocab() -> Vocab {
    let encoded: BTreeMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(VOCAB_PATH).expect("read vocab"))
            .expect("parse vocab");
    Vocab::new(
        encoded
            .into_iter()
            .map(|(id, bytes)| (id.parse::<u32>().expect("numeric token id"), decode_hex(&bytes)))
            .collect(),
    )
}

#[derive(Debug, Clone)]
struct Candidate {
    root: u32,
    closure: BTreeSet<u32>,
    external_calls: usize,
    rules: usize,
    terminals: usize,
}

fn rules_by_lhs(grammar: &GrammarDef) -> BTreeMap<u32, Vec<&Rule>> {
    let mut by_lhs = BTreeMap::<u32, Vec<&Rule>>::new();
    for rule in &grammar.rules {
        by_lhs.entry(rule.lhs).or_default().push(rule);
    }
    by_lhs
}

fn closed_candidate(grammar: &GrammarDef, by_lhs: &BTreeMap<u32, Vec<&Rule>>, root: u32) -> Option<Candidate> {
    if root == grammar.start {
        return None;
    }
    let mut closure = BTreeSet::from([root]);
    let mut queue = VecDeque::from([root]);
    while let Some(nt) = queue.pop_front() {
        for rule in by_lhs.get(&nt).into_iter().flatten() {
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(target) = symbol
                    && closure.insert(*target)
                {
                    queue.push_back(*target);
                }
            }
        }
    }
    let mut external_calls = 0usize;
    for rule in &grammar.rules {
        if closure.contains(&rule.lhs) {
            continue;
        }
        for symbol in &rule.rhs {
            let Symbol::Nonterminal(target) = symbol else { continue };
            if !closure.contains(target) {
                continue;
            }
            if *target != root {
                return None;
            }
            external_calls += 1;
        }
    }
    if external_calls == 0 {
        return None;
    }
    let module_rules = grammar
        .rules
        .iter()
        .filter(|rule| closure.contains(&rule.lhs))
        .collect::<Vec<_>>();
    let terminals = module_rules
        .iter()
        .flat_map(|rule| &rule.rhs)
        .filter_map(|symbol| match symbol {
            Symbol::Terminal(terminal) => Some(*terminal),
            Symbol::Nonterminal(_) => None,
        })
        .collect::<BTreeSet<_>>()
        .len();
    Some(Candidate {
        root,
        closure,
        external_calls,
        rules: module_rules.len(),
        terminals,
    })
}

fn all_closed_candidates(grammar: &GrammarDef) -> Vec<Candidate> {
    let by_lhs = rules_by_lhs(grammar);
    (0..grammar.num_nonterminals())
        .filter_map(|root| closed_candidate(grammar, &by_lhs, root))
        .collect()
}

fn selected_modules(grammar: &GrammarDef) -> Vec<Candidate> {
    let max_closure = std::env::var("O21137_SUBGRAMMAR_MAX_CLOSURE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);
    let min_rules = std::env::var("O21137_SUBGRAMMAR_MIN_RULES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let min_calls = std::env::var("O21137_SUBGRAMMAR_MIN_CALLS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let max_modules = std::env::var("O21137_SUBGRAMMAR_MAX_MODULES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    let mut candidates = all_closed_candidates(grammar)
        .into_iter()
        .filter(|candidate| {
            candidate.closure.len() <= max_closure
                && candidate.rules >= min_rules
                && candidate.external_calls >= min_calls
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| {
        std::cmp::Reverse((
            candidate.closure.len(),
            candidate.rules,
            candidate.terminals,
            candidate.external_calls,
        ))
    });
    let mut occupied = BTreeSet::<u32>::new();
    let mut selected = Vec::new();
    for candidate in candidates {
        if candidate.closure.iter().any(|nt| occupied.contains(nt)) {
            continue;
        }
        occupied.extend(candidate.closure.iter().copied());
        selected.push(candidate);
        if selected.len() == max_modules {
            break;
        }
    }
    selected
}

fn clone_terminal_with_id(terminal: &Terminal, id: u32) -> Terminal {
    match terminal {
        Terminal::Literal { bytes, .. } => Terminal::Literal {
            id,
            bytes: bytes.clone(),
        },
        Terminal::Pattern { pattern, utf8, .. } => Terminal::Pattern {
            id,
            pattern: pattern.clone(),
            utf8: *utf8,
        },
        Terminal::Expr { expr, .. } => Terminal::Expr {
            id,
            expr: expr.clone(),
        },
        Terminal::SpecialToken { token_id, .. } => Terminal::SpecialToken {
            id,
            token_id: *token_id,
        },
    }
}

fn remap_grammar(
    source: &GrammarDef,
    start: u32,
    rules: Vec<Rule>,
    extra_terminals: &BTreeMap<u32, Terminal>,
    extra_terminal_names: &BTreeMap<u32, String>,
) -> GrammarDef {
    let mut by_lhs = BTreeMap::<u32, Vec<Rule>>::new();
    for rule in rules {
        by_lhs.entry(rule.lhs).or_default().push(rule);
    }
    let mut nt_order = Vec::new();
    let mut seen_nts = BTreeSet::from([start]);
    let mut queue = VecDeque::from([start]);
    while let Some(nt) = queue.pop_front() {
        nt_order.push(nt);
        for rule in by_lhs.get(&nt).into_iter().flatten() {
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(target) = symbol
                    && seen_nts.insert(*target)
                {
                    queue.push_back(*target);
                }
            }
        }
    }
    let nt_map = nt_order
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, new as u32))
        .collect::<BTreeMap<_, _>>();

    let mut terminal_order = Vec::<u32>::new();
    let mut seen_terminals = BTreeSet::<u32>::new();
    for old_lhs in &nt_order {
        for rule in by_lhs.get(old_lhs).into_iter().flatten() {
            for symbol in &rule.rhs {
                if let Symbol::Terminal(terminal) = symbol
                    && seen_terminals.insert(*terminal)
                {
                    terminal_order.push(*terminal);
                }
            }
        }
    }
    if let Some(ignore) = source.ignore_terminal
        && seen_terminals.insert(ignore)
    {
        terminal_order.push(ignore);
    }
    let terminal_map = terminal_order
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, new as u32))
        .collect::<BTreeMap<_, _>>();

    let mut remapped_rules = Vec::new();
    for old_lhs in &nt_order {
        for rule in by_lhs.get(old_lhs).into_iter().flatten() {
            remapped_rules.push(Rule {
                lhs: nt_map[&rule.lhs],
                rhs: rule
                    .rhs
                    .iter()
                    .map(|symbol| match symbol {
                        Symbol::Terminal(terminal) => Symbol::Terminal(terminal_map[terminal]),
                        Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(nt_map[nonterminal]),
                    })
                    .collect(),
            });
        }
    }
    let terminals = terminal_order
        .iter()
        .enumerate()
        .map(|(new, old)| {
            let terminal = extra_terminals
                .get(old)
                .unwrap_or_else(|| &source.terminals[*old as usize]);
            clone_terminal_with_id(terminal, new as u32)
        })
        .collect::<Vec<_>>();
    let terminal_names = terminal_order
        .iter()
        .enumerate()
        .filter_map(|(new, old)| {
            extra_terminal_names
                .get(old)
                .or_else(|| source.terminal_names.get(old))
                .map(|name| (new as u32, name.clone()))
        })
        .collect();
    let nonterminal_names = nt_order
        .iter()
        .enumerate()
        .filter_map(|(new, old)| {
            source
                .nonterminal_names
                .get(old)
                .map(|name| (new as u32, name.clone()))
        })
        .collect();
    let lexer_partitions = terminal_order
        .iter()
        .enumerate()
        .filter_map(|(new, old)| {
            source
                .lexer_partitions
                .get(old)
                .map(|partition| (new as u32, partition.clone()))
        })
        .collect();
    let residual_isolation_classes = terminal_order
        .iter()
        .enumerate()
        .filter_map(|(new, old)| {
            source
                .residual_isolation_classes
                .get(old)
                .map(|class| (new as u32, *class))
        })
        .collect();

    GrammarDef {
        rules: remapped_rules,
        start: nt_map[&start],
        terminals,
        nonterminal_names,
        terminal_names,
        ignore_terminal: source.ignore_terminal.map(|ignore| terminal_map[&ignore]),
        lexer_partitions,
        residual_isolation_classes,
        requires_global_terminal_observation: source.requires_global_terminal_observation,
        direct_regular_automaton: None,
    }
}

fn extract_child(source: &GrammarDef, module: &Candidate) -> GrammarDef {
    let rules = source
        .rules
        .iter()
        .filter(|rule| module.closure.contains(&rule.lhs))
        .cloned()
        .collect();
    remap_grammar(source, module.root, rules, &BTreeMap::new(), &BTreeMap::new())
}

fn build_parent(
    source: &GrammarDef,
    modules: &[Candidate],
    module_to_unique: &[usize],
    unique_module_count: usize,
    first_sentinel_token: u32,
) -> (GrammarDef, Vec<String>) {
    let removed = modules
        .iter()
        .flat_map(|module| module.closure.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut root_to_placeholder = BTreeMap::<u32, u32>::new();
    let mut extra_terminals = BTreeMap::<u32, Terminal>::new();
    let mut extra_terminal_names = BTreeMap::<u32, String>::new();
    assert_eq!(modules.len(), module_to_unique.len());
    let mut names = Vec::with_capacity(unique_module_count);
    for unique in 0..unique_module_count {
        let terminal = source.terminals.len() as u32 + unique as u32;
        let name = format!("__O21137_SUBGRAMMAR_{unique}");
        extra_terminals.insert(
            terminal,
            Terminal::SpecialToken {
                id: terminal,
                token_id: first_sentinel_token + unique as u32,
            },
        );
        extra_terminal_names.insert(terminal, name.clone());
        names.push(name);
    }
    for (module, &unique) in modules.iter().zip(module_to_unique) {
        root_to_placeholder.insert(
            module.root,
            source.terminals.len() as u32 + unique as u32,
        );
    }
    let rules = source
        .rules
        .iter()
        .filter(|rule| !removed.contains(&rule.lhs))
        .map(|rule| Rule {
            lhs: rule.lhs,
            rhs: rule
                .rhs
                .iter()
                .map(|symbol| match symbol {
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some(placeholder) = root_to_placeholder.get(nonterminal) {
                            Symbol::Terminal(*placeholder)
                        } else {
                            assert!(
                                !removed.contains(nonterminal),
                                "external reference enters module below its root"
                            );
                            Symbol::Nonterminal(*nonterminal)
                        }
                    }
                    Symbol::Terminal(terminal) => Symbol::Terminal(*terminal),
                })
                .collect(),
        })
        .collect();
    (
        remap_grammar(
            source,
            source.start,
            rules,
            &extra_terminals,
            &extra_terminal_names,
        ),
        names,
    )
}

// The legacy owned-parent path is intentionally unsupported; this benchmark
// now returns that error rather than silently switching backends.
#[allow(deprecated)]
fn compose_named_parent_owned(
    parent: Constraint,
    children: &[(&str, &Constraint)],
    vocab: &Vocab,
) -> Constraint {
    let inputs = children
        .iter()
        .map(|(name, child)| {
            let placeholder_terminal = parent
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap_or_else(|| panic!("benchmark placeholder terminal {name:?}"))
                as u32;
            crate::compiler::composition::CompiledSubgrammarInput {
                placeholder_terminal,
                additional_placeholder_terminals: &[],
                constraint: child,
            }
        })
        .collect::<Vec<_>>();
    crate::compiler::composition::compose_constraints_owned_parent(parent, &inputs, vocab)
        .expect("compose benchmark subgrammars")
        .constraint
}

fn terminal_child(source: &GrammarDef, terminal: u32) -> GrammarDef {
    let source_terminal = &source.terminals[terminal as usize];
    let terminals = vec![clone_terminal_with_id(source_terminal, 0)];
    let mut terminal_names = BTreeMap::new();
    if let Some(name) = source.terminal_names.get(&terminal) {
        terminal_names.insert(0, name.clone());
    }
    let mut nonterminal_names = BTreeMap::new();
    nonterminal_names.insert(0, format!("terminal_{terminal}_root"));
    let mut lexer_partitions = BTreeMap::new();
    if let Some(partition) = source.lexer_partitions.get(&terminal) {
        lexer_partitions.insert(0, partition.clone());
    }
    let mut residual_isolation_classes = BTreeMap::new();
    if let Some(class) = source.residual_isolation_classes.get(&terminal) {
        residual_isolation_classes.insert(0, *class);
    }
    GrammarDef {
        rules: vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }],
        start: 0,
        terminals,
        nonterminal_names,
        terminal_names,
        ignore_terminal: None,
        lexer_partitions,
        residual_isolation_classes,
        requires_global_terminal_observation: source.requires_global_terminal_observation,
        direct_regular_automaton: None,
    }
}

fn selected_terminals(grammar: &GrammarDef) -> Vec<u32> {
    let filter = std::env::var("O21137_TERMINAL_FILTER").unwrap_or_else(|_| "bounded".into());
    grammar
        .terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, definition)| {
            if matches!(definition, Terminal::SpecialToken { .. }) {
                return None;
            }
            let name = grammar
                .terminal_names
                .get(&(terminal as u32))
                .map(String::as_str)
                .unwrap_or("");
            let selected = match filter.as_str() {
                "bounded" => name.contains("bounded"),
                "expr" => matches!(definition, Terminal::Expr { .. }),
                "nonliteral" => !matches!(definition, Terminal::Literal { .. }),
                "all" => true,
                other => panic!("unknown O21137_TERMINAL_FILTER={other}"),
            };
            selected.then_some(terminal as u32)
        })
        .collect()
}

fn build_terminal_parent(
    source: &GrammarDef,
    selected: &[u32],
    terminal_to_unique: &BTreeMap<u32, usize>,
    unique_count: usize,
    first_sentinel_token: u32,
) -> (GrammarDef, Vec<String>) {
    let mut extra_terminals = BTreeMap::<u32, Terminal>::new();
    let mut extra_terminal_names = BTreeMap::<u32, String>::new();
    let mut names = Vec::with_capacity(unique_count);
    for unique in 0..unique_count {
        let terminal = source.terminals.len() as u32 + unique as u32;
        let name = format!("__O21137_TERMINAL_SUBGRAMMAR_{unique}");
        extra_terminals.insert(
            terminal,
            Terminal::SpecialToken {
                id: terminal,
                token_id: first_sentinel_token + unique as u32,
            },
        );
        extra_terminal_names.insert(terminal, name.clone());
        names.push(name);
    }
    let selected = selected.iter().copied().collect::<BTreeSet<_>>();
    let rules = source
        .rules
        .iter()
        .map(|rule| Rule {
            lhs: rule.lhs,
            rhs: rule
                .rhs
                .iter()
                .map(|symbol| match symbol {
                    Symbol::Terminal(terminal) if selected.contains(terminal) => {
                        let unique = terminal_to_unique[terminal];
                        Symbol::Terminal(source.terminals.len() as u32 + unique as u32)
                    }
                    other => other.clone(),
                })
                .collect(),
        })
        .collect();
    (
        remap_grammar(
            source,
            source.start,
            rules,
            &extra_terminals,
            &extra_terminal_names,
        ),
        names,
    )
}

fn build_terminal_decomposed(grammar: &GrammarDef, vocab: &Vocab) -> DecomposedBuild {
    let planning_started = Instant::now();
    let selected = selected_terminals(grammar);
    assert!(!selected.is_empty(), "terminal decomposition selected no terminals");
    let mut key_to_unique = BTreeMap::<Vec<u8>, usize>::new();
    let mut unique_grammars = Vec::<GrammarDef>::new();
    let mut terminal_to_unique = BTreeMap::<u32, usize>::new();
    for terminal in &selected {
        let child = terminal_child(grammar, *terminal);
        let key = semantic_key(&child);
        let unique = if let Some(&unique) = key_to_unique.get(&key) {
            unique
        } else {
            let unique = unique_grammars.len();
            key_to_unique.insert(key, unique);
            unique_grammars.push(child);
            unique
        };
        terminal_to_unique.insert(*terminal, unique);
    }
    let first_sentinel = vocab
        .entries_map()
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(20_000);
    let (parent_grammar, placeholder_names) = build_terminal_parent(
        grammar,
        &selected,
        &terminal_to_unique,
        unique_grammars.len(),
        first_sentinel,
    );
    let planning_ms = elapsed_ms(planning_started);

    let mut constraints = Vec::with_capacity(unique_grammars.len());
    let mut child_ms = Vec::with_capacity(unique_grammars.len());
    for child in unique_grammars {
        let started = Instant::now();
        constraints.push(compile_grammar(child, vocab));
        child_ms.push(elapsed_ms(started));
    }
    let parent_started = Instant::now();
    let parent = compile_grammar(parent_grammar, vocab);
    let parent_ms = elapsed_ms(parent_started);
    let child_bindings = placeholder_names
        .iter()
        .zip(constraints.iter())
        .map(|(name, constraint)| (name.as_str(), constraint))
        .collect::<Vec<_>>();
    profile_owned_parent_tokenizer(&parent, &child_bindings);
    let composition_started = Instant::now();
    let constraint = compose_named_parent_owned(parent, &child_bindings, vocab);
    let composition_ms = elapsed_ms(composition_started);
    DecomposedBuild {
        constraint,
        modules: selected.len(),
        unique_modules: constraints.len(),
        planning_ms,
        child_ms,
        parent_ms,
        composition_ms,
    }
}

fn profile_owned_parent_tokenizer(
    parent: &Constraint,
    child_bindings: &[(&str, &Constraint)],
) {
    if std::env::var_os("O21137_PROFILE_OWNED_PARENT_TOKENIZER").is_none() {
        return;
    }
    let table_inputs = child_bindings
        .iter()
        .map(|(name, child)| {
            let placeholder_terminal = parent
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .expect("benchmark placeholder terminal") as u32;
            crate::compiler::glr::table::SubgrammarTableInput {
                placeholder_terminal,
                additional_placeholder_terminals: &[],
                table: &child.table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.table.embedded_start_nullable(),
            }
        })
        .collect::<Vec<_>>();
    let composed_table = crate::compiler::glr::table::compose_subgrammar_tables(
        &parent.table,
        None,
        &table_inputs,
    )
    .expect("benchmark table composition");
    // Clone outside the measured interval. The production fast path consumes
    // the freshly compiled parent and therefore does not pay this clone.
    let owned_parent = parent.tokenizer.as_ref().clone();
    let child_tokenizers = child_bindings
        .iter()
        .enumerate()
        .map(|(index, (_, child))| {
            (child.tokenizer.as_ref(), composed_table.terminal_offsets[index + 1])
        })
        .collect::<Vec<_>>();
    let started_at = Instant::now();
    let (merged, offsets) =
        crate::automata::lexer::tokenizer::Tokenizer::disjoint_union_with_owned_parent(
            owned_parent,
            composed_table.terminal_offsets[0],
            &child_tokenizers,
        );
    eprintln!(
        "O21137_OWNED_PARENT_TOKENIZER ms={:.3} states={} offsets={:?}",
        elapsed_ms(started_at),
        merged.num_states(),
        offsets,
    );
}

fn semantic_key(grammar: &GrammarDef) -> Vec<u8> {
    let mut semantic = grammar.clone();
    semantic.terminal_names.clear();
    semantic.nonterminal_names.clear();
    serde_json::to_vec(&semantic).expect("serialize canonical child grammar")
}

fn compile_grammar(grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    crate::compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
        crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar,
            vocab,
            GlrTableConstruction::LegacyRowBisim,
        )
    })
}

struct DecomposedBuild {
    constraint: Constraint,
    modules: usize,
    unique_modules: usize,
    planning_ms: f64,
    child_ms: Vec<f64>,
    parent_ms: f64,
    composition_ms: f64,
}

struct NonterminalDecomposedParts {
    parent: Constraint,
    children: Vec<Constraint>,
    placeholder_names: Vec<String>,
    modules: usize,
    planning_ms: f64,
    child_ms: Vec<f64>,
    parent_ms: f64,
}

fn prepare_nonterminal_decomposed_parts(
    grammar: &GrammarDef,
    vocab: &Vocab,
) -> NonterminalDecomposedParts {
    let planning_started = Instant::now();
    let modules = selected_modules(grammar);
    let extracted = modules
        .iter()
        .map(|module| extract_child(grammar, module))
        .collect::<Vec<_>>();
    let mut key_to_unique = BTreeMap::<Vec<u8>, usize>::new();
    let mut unique_grammars = Vec::<GrammarDef>::new();
    let mut module_to_unique = Vec::<usize>::new();
    for child in extracted {
        let key = semantic_key(&child);
        let unique = if let Some(&unique) = key_to_unique.get(&key) {
            unique
        } else {
            let unique = unique_grammars.len();
            key_to_unique.insert(key, unique);
            unique_grammars.push(child);
            unique
        };
        module_to_unique.push(unique);
    }
    let first_sentinel = vocab
        .entries_map()
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(10_000);
    let (parent_grammar, placeholder_names) = build_parent(
        grammar,
        &modules,
        &module_to_unique,
        unique_grammars.len(),
        first_sentinel,
    );
    let planning_ms = elapsed_ms(planning_started);

    let profile_setup = std::env::var_os("O21137_PROFILE_SETUP").is_some();
    let mut children = Vec::<Constraint>::with_capacity(unique_grammars.len());
    let mut child_ms = Vec::with_capacity(unique_grammars.len());
    for (child_index, child) in unique_grammars.into_iter().enumerate() {
        if profile_setup {
            eprintln!(
                "O21137_SETUP_CHILD_START index={child_index} rules={} terminals={} nonterminals={}",
                child.rules.len(),
                child.terminals.len(),
                child.num_nonterminals(),
            );
        }
        let started = Instant::now();
        children.push(compile_grammar(child, vocab));
        let child_elapsed = elapsed_ms(started);
        if profile_setup {
            eprintln!("O21137_SETUP_CHILD_DONE index={child_index} ms={child_elapsed:.3}");
        }
        child_ms.push(child_elapsed);
    }
    if profile_setup {
        eprintln!(
            "O21137_SETUP_PARENT_START rules={} terminals={} nonterminals={}",
            parent_grammar.rules.len(),
            parent_grammar.terminals.len(),
            parent_grammar.num_nonterminals(),
        );
    }
    let parent_started = Instant::now();
    let parent = compile_grammar(parent_grammar, vocab);
    let parent_ms = elapsed_ms(parent_started);
    if profile_setup {
        eprintln!("O21137_SETUP_PARENT_DONE ms={parent_ms:.3}");
    }
    NonterminalDecomposedParts {
        parent,
        children,
        placeholder_names,
        modules: modules.len(),
        planning_ms,
        child_ms,
        parent_ms,
    }
}

fn child_bindings<'a>(
    placeholder_names: &'a [String],
    children: &'a [Constraint],
) -> Vec<(&'a str, &'a Constraint)> {
    placeholder_names
        .iter()
        .zip(children)
        .map(|(name, constraint)| (name.as_str(), constraint))
        .collect()
}

fn build_nonterminal_decomposed(grammar: &GrammarDef, vocab: &Vocab) -> DecomposedBuild {
    let parts = prepare_nonterminal_decomposed_parts(grammar, vocab);
    let bindings = child_bindings(&parts.placeholder_names, &parts.children);
    profile_owned_parent_tokenizer(&parts.parent, &bindings);
    let composition_started = Instant::now();
    let constraint = compose_named_parent_owned(parts.parent, &bindings, vocab);
    let composition_ms = elapsed_ms(composition_started);
    DecomposedBuild {
        constraint,
        modules: parts.modules,
        unique_modules: parts.children.len(),
        planning_ms: parts.planning_ms,
        child_ms: parts.child_ms,
        parent_ms: parts.parent_ms,
        composition_ms,
    }
}

fn build_decomposed(grammar: &GrammarDef, vocab: &Vocab) -> DecomposedBuild {
    match std::env::var("O21137_SUBGRAMMAR_KIND")
        .unwrap_or_else(|_| "nonterminal".into())
        .as_str()
    {
        "nonterminal" => build_nonterminal_decomposed(grammar, vocab),
        "terminal" => build_terminal_decomposed(grammar, vocab),
        other => panic!("unknown O21137_SUBGRAMMAR_KIND={other}"),
    }
}

fn accepts_bytes(constraint: &Constraint, bytes: &[u8]) -> bool {
    let mut state = constraint.start();
    state.commit_bytes(bytes).is_ok() && state.is_accepting()
}

fn token_allowed(mask: &[u32], token: u32) -> bool {
    mask.get(token as usize / 32)
        .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
}

fn sampled_tokens(allowed: &[u32], path: &[u32], limit: usize) -> Vec<u32> {
    if allowed.len() <= limit {
        return allowed.to_vec();
    }
    let mut indices = BTreeSet::<usize>::new();
    for index in 0..4.min(allowed.len()) {
        indices.insert(index);
        indices.insert(allowed.len() - 1 - index);
    }
    for slot in 1..=limit.saturating_sub(indices.len()) {
        indices.insert(slot * (allowed.len() - 1) / (limit + 1));
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for &token in path {
        seed ^= token as u64;
        seed = seed
            .wrapping_mul(0xbf58_476d_1ce4_e5b9)
            .rotate_left(17);
    }
    while indices.len() < limit {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        indices.insert((seed as usize) % allowed.len());
    }
    indices.into_iter().map(|index| allowed[index]).collect()
}

fn verify_sampled_reachable_masks(
    monolithic: &Constraint,
    decomposed: &Constraint,
    vocab: &Vocab,
    max_depth: usize,
    max_frontier: usize,
    tokens_per_state: usize,
) -> (usize, usize) {
    let vocab_tokens = vocab.entries_map().keys().copied().collect::<Vec<_>>();
    let mut frontier = BTreeSet::<Vec<u32>>::from([Vec::new()]);
    let mut checked_prefixes = 0usize;
    let mut checked_commits = 0usize;
    for depth in 0..=max_depth {
        let mut next = BTreeSet::<Vec<u32>>::new();
        for path in &frontier {
            let mut monolithic_state = monolithic.start();
            let mut decomposed_state = decomposed.start();
            for &token in path {
                monolithic_state
                    .commit_token(token)
                    .unwrap_or_else(|error| panic!("monolithic rejected sampled path {path:?}: {error}"));
                decomposed_state
                    .commit_token(token)
                    .unwrap_or_else(|error| panic!("decomposed rejected sampled path {path:?}: {error}"));
            }
            let monolithic_mask = monolithic_state.mask();
            let decomposed_mask = decomposed_state.mask();
            assert_eq!(
                decomposed_mask, monolithic_mask,
                "mask mismatch after sampled o21137 token path {path:?}",
            );
            assert_eq!(
                decomposed_state.is_accepting(),
                monolithic_state.is_accepting(),
                "completion mismatch after sampled o21137 token path {path:?}",
            );
            checked_prefixes += 1;
            if depth == max_depth {
                continue;
            }
            let allowed = vocab_tokens
                .iter()
                .copied()
                .filter(|&token| token_allowed(&monolithic_mask, token))
                .collect::<Vec<_>>();
            for token in sampled_tokens(&allowed, path, tokens_per_state) {
                let mut next_monolithic = monolithic_state.clone();
                let mut next_decomposed = decomposed_state.clone();
                let monolithic_result = next_monolithic.commit_token(token);
                let decomposed_result = next_decomposed.commit_token(token);
                assert_eq!(
                    decomposed_result.is_ok(),
                    monolithic_result.is_ok(),
                    "commit result mismatch for token {token} after sampled path {path:?}",
                );
                checked_commits += 1;
                if monolithic_result.is_ok() {
                    let mut extended = path.clone();
                    extended.push(token);
                    next.insert(extended);
                }
            }
        }
        frontier = next.into_iter().take(max_frontier).collect();
        if frontier.is_empty() && depth < max_depth {
            break;
        }
    }
    (checked_prefixes, checked_commits)
}

pub(crate) fn run(mode: &str) {
    let mode = mode.to_string();
    let total_started = Instant::now();
    let wrapper = load_wrapper();
    let vocab_started = Instant::now();
    let vocab = load_llama3_vocab();
    let vocab_ms = elapsed_ms(vocab_started);
    let vocab_prepare_started = Instant::now();
    crate::compiler::compile::prepare_vocab_for_compile(&vocab);
    let vocab_prepare_ms = elapsed_ms(vocab_prepare_started);
    let import_started = Instant::now();
    let grammar = import_o21137_grammar(&wrapper);
    let import_ms = elapsed_ms(import_started);

    match mode.as_str() {
        "plan" => {
            let mut candidates = all_closed_candidates(&grammar);
            candidates.sort_by_key(|candidate| {
                std::cmp::Reverse((candidate.closure.len(), candidate.rules, candidate.terminals))
            });
            eprintln!(
                "O21137_PLAN grammar_nonterminals={} grammar_rules={} grammar_terminals={} candidates={}",
                grammar.num_nonterminals(),
                grammar.rules.len(),
                grammar.terminals.len(),
                candidates.len(),
            );
            for candidate in candidates.iter().take(200) {
                let name = grammar
                    .nonterminal_names
                    .get(&candidate.root)
                    .map(String::as_str)
                    .unwrap_or("<unnamed>");
                eprintln!(
                    "O21137_CANDIDATE root={} name={:?} closure={} rules={} terminals={} calls={}",
                    candidate.root,
                    name,
                    candidate.closure.len(),
                    candidate.rules,
                    candidate.terminals,
                    candidate.external_calls,
                );
            }
        }
        "standard" => {
            let schema_json = serde_json::to_string(wrapper.get("schema").expect("schema field"))
                .expect("serialize schema");
            let compile_started = Instant::now();
            let constraint = Constraint::from_json_schema(&schema_json, &vocab)
                .expect("standard o21137 compile");
            let compile_ms = elapsed_ms(compile_started);
            eprintln!(
                "O21137_STANDARD vocab_ms={vocab_ms:.3} vocab_prepare_ms={vocab_prepare_ms:.3} import_ms={import_ms:.3} compile_ms={compile_ms:.3} tokenizer_states={} table_states={} total_ms={:.3}",
                constraint.tokenizer.num_states(),
                constraint.table.num_states,
                elapsed_ms(total_started),
            );
            std::hint::black_box(constraint);
        }
        "monolithic" => {
            let compile_started = Instant::now();
            let constraint = compile_grammar(grammar, &vocab);
            let compile_ms = elapsed_ms(compile_started);
            std::hint::black_box(constraint);
            eprintln!(
                "O21137_MONOLITHIC vocab_ms={vocab_ms:.3} vocab_prepare_ms={vocab_prepare_ms:.3} import_ms={import_ms:.3} compile_ms={compile_ms:.3} total_ms={:.3}",
                elapsed_ms(total_started),
            );
        }
        "compose-only" => {
            let parts = prepare_nonterminal_decomposed_parts(&grammar, &vocab);
            let repeats = std::env::var("O21137_COMPOSE_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(5)
                .max(1);
            let consume_original = std::env::var_os("O21137_COMPOSE_CONSUME_ORIGINAL").is_some();
            assert!(
                !consume_original || repeats == 1,
                "O21137_COMPOSE_CONSUME_ORIGINAL requires O21137_COMPOSE_REPEATS=1",
            );
            let mut parent_once = Some(parts.parent);
            let mut clone_samples = Vec::with_capacity(repeats);
            let mut compose_samples = Vec::with_capacity(repeats);
            for sample in 0..repeats {
                let clone_started = Instant::now();
                let parent = if consume_original {
                    parent_once.take().expect("original parent already consumed")
                } else {
                    parent_once
                        .as_ref()
                        .expect("benchmark parent missing")
                        .clone()
                };
                let clone_ms = if consume_original {
                    0.0
                } else {
                    elapsed_ms(clone_started)
                };
                let bindings = child_bindings(&parts.placeholder_names, &parts.children);
                let compose_started = Instant::now();
                let composed = compose_named_parent_owned(parent, &bindings, &vocab);
                let compose_ms = elapsed_ms(compose_started);
                eprintln!(
                    "O21137_COMPOSE_SAMPLE sample={sample} clone_ms={clone_ms:.3} compose_ms={compose_ms:.3} tokenizer_states={} parser_dwa_states={}",
                    composed.tokenizer.num_states(),
                    composed.parser_dwa.num_states(),
                );
                clone_samples.push(clone_ms);
                compose_samples.push(compose_ms);
                std::hint::black_box(composed);
            }
            clone_samples.sort_by(f64::total_cmp);
            compose_samples.sort_by(f64::total_cmp);
            let child_total_ms = parts.child_ms.iter().sum::<f64>();
            eprintln!(
                "O21137_COMPOSE_ONLY modules={} unique_modules={} repeats={} consume_original={} planning_ms={:.3} child_total_ms={child_total_ms:.3} child_ms={:?} parent_ms={:.3} clone_median_ms={:.3} compose_min_ms={:.3} compose_median_ms={:.3} compose_max_ms={:.3} total_ms={:.3}",
                parts.modules,
                parts.children.len(),
                repeats,
                consume_original,
                parts.planning_ms,
                parts.child_ms,
                parts.parent_ms,
                clone_samples[repeats / 2],
                compose_samples[0],
                compose_samples[repeats / 2],
                compose_samples[repeats - 1],
                elapsed_ms(total_started),
            );
        }
        "decomposed" => {
            let build = build_decomposed(&grammar, &vocab);
            let child_total_ms = build.child_ms.iter().sum::<f64>();
            let compile_compose_ms =
                build.planning_ms + child_total_ms + build.parent_ms + build.composition_ms;
            eprintln!(
                "O21137_DECOMPOSED modules={} unique_modules={} vocab_ms={vocab_ms:.3} vocab_prepare_ms={vocab_prepare_ms:.3} import_ms={import_ms:.3} planning_ms={:.3} child_total_ms={child_total_ms:.3} child_ms={:?} parent_ms={:.3} composition_ms={:.3} compile_compose_ms={compile_compose_ms:.3} total_ms={:.3}",
                build.modules,
                build.unique_modules,
                build.planning_ms,
                build.child_ms,
                build.parent_ms,
                build.composition_ms,
                elapsed_ms(total_started),
            );
            std::hint::black_box(build.constraint);
        }
        "verify" => {
            let monolithic = compile_grammar(grammar.clone(), &vocab);
            let decomposed = build_decomposed(&grammar, &vocab);
            let tests = wrapper
                .get("tests")
                .and_then(serde_json::Value::as_array)
                .expect("test array");
            for (index, test) in tests.iter().enumerate() {
                let bytes = serde_json::to_vec(test.get("data").expect("test data"))
                    .expect("serialize test data");
                let monolithic_accepts = accepts_bytes(&monolithic, &bytes);
                let decomposed_accepts = accepts_bytes(&decomposed.constraint, &bytes);
                assert_eq!(
                    decomposed_accepts, monolithic_accepts,
                    "monolithic/composed language mismatch on o21137 test {index}"
                );
                eprintln!(
                    "O21137_VERIFY test={} bytes={} monolithic={} decomposed={}",
                    index,
                    bytes.len(),
                    monolithic_accepts,
                    decomposed_accepts,
                );
            }
            let (checked_prefixes, checked_commits) = verify_sampled_reachable_masks(
                &monolithic,
                &decomposed.constraint,
                &vocab,
                12,
                48,
                16,
            );
            eprintln!(
                "O21137_VERIFY_PREFIXES prefixes={} commits={} depth=12 frontier=48 tokens_per_state=16",
                checked_prefixes,
                checked_commits,
            );
            eprintln!(
                "O21137_VERIFY_OK modules={} unique_modules={} total_ms={:.3}",
                decomposed.modules,
                decomposed.unique_modules,
                elapsed_ms(total_started),
            );
        }
        other => panic!("unknown O21137_SUBGRAMMAR_MODE={other}"),
    }
}
