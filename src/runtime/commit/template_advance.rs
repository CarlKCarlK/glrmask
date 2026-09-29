use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{
    DEFAULT_LABEL,
    is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::leveled_gss::{GssSemanticKeyInterner, VirtualStack};
use crate::grammar::flat::TerminalID;
use crate::runtime::{CommitTemplateDfas, FastCommitTemplateDfas};
use crate::runtime::constraint::Constraint;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

// The token-local language evaluator is an optimization with an exact table
// fallback. Keep every internal representation explicitly bounded so a compact
// but adversarial GSS/template product cannot turn the fast path into a runtime
// cliff before fallback.
const LANGUAGE_QUEUE_MAX_SEMANTIC_NODES: usize = 1_024;
const LANGUAGE_QUEUE_MAX_SOURCE_KEYS: usize = 1_024;
const LANGUAGE_QUEUE_MAX_UNION_ENTRIES: usize = 2_048;
const LANGUAGE_QUEUE_MAX_TEMPLATE_PRODUCTS: usize = 1_024;
const LANGUAGE_QUEUE_MAX_ACCUMULATOR_COMPONENTS: usize = 32;
const LANGUAGE_QUEUE_MAX_ACCUMULATOR_UPPER_NODES: usize = 512;

pub(crate) fn advance_stacks_template_dfa(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    let dfa = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    let output = advance_with_prepared_template(dfa, stack.clone(),
        constraint.fast_template_dfas_by_terminal.get(terminal as usize).and_then(|t| t.as_deref()));
    debug_validate_template_output(dfa, stack, &output, terminal);
    Some(output)
}

pub(super) fn advance_stacks_template_dfa_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    let dfa = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    let input = debug_template_literal_enabled().then(|| stack.clone());
    let output = advance_with_prepared_template(dfa, stack,
        constraint.fast_template_dfas_by_terminal.get(terminal as usize).and_then(|t| t.as_deref()));
    if let Some(input) = input { debug_validate_template_output(dfa, &input, &output, terminal); }
    Some(output)
}

// Opt-in independent interpreter used only to diagnose relation/GSS errors.
// It never references an LR table and is excluded from timed runs.
fn debug_template_literal_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("GLRMASK_DIAG_TEMPLATE_LITERAL").is_some())
}

fn debug_validate_template_output(
    template: &CommitTemplateDfas, input: &ParserGSS, output: &ParserGSS, terminal: u32,
) {
    if !debug_template_literal_enabled() { return; }
    let Some(inputs) = input.to_stacks(4_096) else { return; };
    let mut outputs = Vec::new();
    let mut operations = 0usize;
    for (stack, accumulator) in &inputs {
        let mut work = vec![(Phase::Pop, template.pop.start_state, stack.clone())];
        let mut visited = rustc_hash::FxHashSet::default();
        while let Some((phase, id, stack)) = work.pop() {
            operations += 1;
            if operations > 1_000_000 { return; }
            if !visited.insert((phase, id, stack.clone())) { continue; }
            let state = match phase {
                Phase::Pop => &template.pop.states[id as usize],
                Phase::Read => &template.read.states[id as usize],
                Phase::Push => &template.push.states[id as usize],
            };
            if state.is_accepting { outputs.push((stack.clone(), accumulator.clone())); }
            match phase {
                Phase::Pop => {
                    if let Some(&top) = stack.last() {
                        if let Some(&next) = state.transitions.get(&(top as i32))
                            .or_else(|| state.transitions.get(&DEFAULT_LABEL)) {
                            let mut popped = stack.clone(); popped.pop();
                            work.push((Phase::Pop, next, popped));
                        }
                    }
                    if let Some(next) = template.pop_to_read[id as usize] {
                        work.push((Phase::Read, next, stack.clone()));
                    }
                    if let Some(next) = template.pop_to_push[id as usize] {
                        work.push((Phase::Push, next, stack));
                    }
                }
                Phase::Read => {
                    if let Some(top) = stack.last() {
                        if let Some(&next) = state.transitions.get(&(*top as i32)) {
                            work.push((Phase::Read, next, stack.clone()));
                        }
                    }
                    if let Some(next) = template.read_to_push[id as usize] {
                        work.push((Phase::Push, next, stack));
                    }
                }
                Phase::Push => {
                    for (&label, &next) in &state.transitions {
                        let mut pushed = stack.clone();
                        pushed.push(negative_to_positive_label(label) as u32);
                        work.push((Phase::Push, next, pushed));
                    }
                }
            }
        }
    }
    if terminal == 1 && std::env::var_os("GLRMASK_DIAG_TEMPLATE_TRACE").is_some() {
        use std::io::Write;
        let path = std::env::var("GLRMASK_DIAG_TEMPLATE_TRACE").unwrap();
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        let record = serde_json::json!({
            "terminal":terminal,
            "input":inputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "output":outputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "accumulators":format!("{inputs:?}"),
        });
        writeln!(file, "{record}").unwrap();
    }
    let expected = ParserGSS::from_stacks(&outputs);
    if output.semantically_eq(&expected, 65_536) != Some(true) {
        let fixture = serde_json::json!({
            "terminal":terminal,"template":template,
            "input_stacks":inputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "literal_stacks":outputs.iter().map(|(stack,_)|stack).collect::<Vec<_>>(),
            "actual_stacks":output.to_stacks(4096).map(|stacks|stacks.into_iter().map(|(stack,_)|stack).collect::<Vec<_>>()),
            "input_debug":format!("{input:?}"),
        });
        let path = std::env::var("GLRMASK_DIAG_TEMPLATE_LITERAL_PATH")
            .unwrap_or_else(|_| "template-literal-difference.json".into());
        std::fs::write(path, serde_json::to_vec(&fixture).unwrap()).unwrap();
        panic!("template GSS output differs from independent literal interpretation for terminal {terminal}");
    }
}

pub(crate) struct TemplateAdvanceRuntime {
    interner: GssSemanticKeyInterner<u32, TerminalsDisallowed>,
    memo_rows: Vec<SmallVec<[TemplateMemoEntry; 2]>>,
    memo_entries: usize,
    component_cache:
        FxHashMap<usize, (ParserGSS, Vec<(u32, TerminalsDisallowed)>)>,
    calls: u64,
    memo_hits: u64,
    products_started: usize,
    max_template_products: usize,
    exhausted: bool,
}

impl Default for TemplateAdvanceRuntime {
    fn default() -> Self {
        Self::with_budget(
            LANGUAGE_QUEUE_MAX_SEMANTIC_NODES,
            LANGUAGE_QUEUE_MAX_SOURCE_KEYS,
            LANGUAGE_QUEUE_MAX_UNION_ENTRIES,
            LANGUAGE_QUEUE_MAX_TEMPLATE_PRODUCTS,
        )
    }
}

impl std::fmt::Debug for TemplateAdvanceRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemplateAdvanceRuntime")
            .field("memo_entries", &self.memo_entries)
            .field("component_cache", &self.component_cache.len())
            .field("calls", &self.calls)
            .field("memo_hits", &self.memo_hits)
            .field("products_started", &self.products_started)
            .field("exhausted", &self.exhausted)
            .finish_non_exhaustive()
    }
}

impl TemplateAdvanceRuntime {
    /// Read-only mask shadows do not execute template commits. Keep a valid
    /// empty runtime without eagerly allocating the unused memo row reserve.
    pub(crate) fn for_mask_only_shadow() -> Self {
        Self::with_budget_and_capacity(
            LANGUAGE_QUEUE_MAX_SEMANTIC_NODES,
            LANGUAGE_QUEUE_MAX_SOURCE_KEYS,
            LANGUAGE_QUEUE_MAX_UNION_ENTRIES,
            LANGUAGE_QUEUE_MAX_TEMPLATE_PRODUCTS,
            0,
        )
    }

    fn with_budget(
        max_semantic_nodes: usize,
        max_source_keys: usize,
        max_union_entries: usize,
        max_template_products: usize,
    ) -> Self {
        Self::with_budget_and_capacity(max_semantic_nodes, max_source_keys,
            max_union_entries, max_template_products, 256)
    }

    fn with_budget_and_capacity(
        max_semantic_nodes: usize,
        max_source_keys: usize,
        max_union_entries: usize,
        max_template_products: usize,
        memo_capacity: usize,
    ) -> Self {
        Self {
            interner: GssSemanticKeyInterner::with_budget(
                max_semantic_nodes,
                max_source_keys,
                max_union_entries,
            ),
            memo_rows: Vec::with_capacity(memo_capacity),
            memo_entries: 0,
            component_cache: FxHashMap::default(),
            calls: 0,
            memo_hits: 0,
            products_started: 0,
            max_template_products,
            exhausted: false,
        }
    }

    #[inline]
    pub(crate) fn is_exhausted(&self) -> bool {
        self.exhausted || self.interner.is_exhausted()
    }

    #[inline]
    fn sync_interner_budget(&mut self) -> bool {
        if self.interner.is_exhausted() {
            self.exhausted = true;
        }
        !self.exhausted
    }

    #[inline]
    fn push_language(&mut self, language: u32, state: u32) -> u32 {
        if self.is_exhausted() {
            return 0;
        }
        let pushed = self.interner.push_key(language, state);
        self.sync_interner_budget();
        pushed
    }

    pub(crate) fn begin_commit(&mut self) {
        // The canonical language representation is intentionally token-local.
        // Rebuild it from the authoritative compact GSS for each selected token
        // rather than carrying a second semantic view across parser states.
        *self = Self::default();
    }

    pub(crate) fn reset_all(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn memo_summary(&self) -> (u64, u64, usize) {
        (self.calls, self.memo_hits, self.memo_entries)
    }

    pub(crate) fn work_summary(
        &self,
    ) -> (u64, u64, usize, usize, usize, usize, usize, usize) {
        let (semantic_nodes, lower_keys, upper_keys, union_entries) =
            self.interner.work_summary();
        (
            self.calls,
            self.memo_hits,
            self.memo_entries,
            self.products_started,
            semantic_nodes,
            lower_keys,
            upper_keys,
            union_entries,
        )
    }

    pub(crate) fn language_from_uniform_gss(
        &mut self,
        stack: &ParserGSS,
    ) -> Option<(u32, TerminalsDisallowed)> {
        if self.is_exhausted() {
            return None;
        }
        let accumulator = stack.uniform_accumulator()?;
        let language = self.interner.key(stack);
        self.sync_interner_budget().then_some((language, accumulator))
    }

    pub(crate) fn language_components_from_gss(
        &mut self,
        stack: &ParserGSS,
    ) -> Option<Vec<(u32, TerminalsDisallowed)>> {
        let ptr = stack.ptr_key();
        if let Some((_, cached)) = self.component_cache.get(&ptr) {
            return Some(cached.clone());
        }

        let components = if let Some(component) = self.language_from_uniform_gss(stack) {
            vec![component]
        } else {
            stack
                .partition_by_accumulator_at_most(
                    LANGUAGE_QUEUE_MAX_ACCUMULATOR_COMPONENTS,
                    LANGUAGE_QUEUE_MAX_ACCUMULATOR_UPPER_NODES,
                )?
                .into_iter()
                .map(|(paths, accumulator)| {
                    let restored = paths.apply(|_| accumulator.clone());
                    (self.interner.key(&restored), accumulator)
                })
                .collect()
        };
        if self.is_exhausted() {
            return None;
        }
        self.register_components(stack, components.clone());
        Some(components)
    }

    pub(crate) fn register_components(
        &mut self,
        stack: &ParserGSS,
        components: Vec<(u32, TerminalsDisallowed)>,
    ) {
        if !self.is_exhausted() {
            self.component_cache
                .insert(stack.ptr_key(), (stack.clone(), components));
        }
    }

    pub(crate) fn gss_from_language(
        &mut self,
        language: u32,
        accumulator: TerminalsDisallowed,
    ) -> ParserGSS {
        self.interner.gss_from_key(language, accumulator)
    }

    pub(crate) fn language_top_states(&self, language: u32) -> SmallVec<[u32; 8]> {
        self.interner
            .top_branches(language)
            .iter()
            .map(|(state, _)| *state)
            .collect()
    }

    pub(crate) fn union_languages(&mut self, left: u32, right: u32) -> u32 {
        if self.is_exhausted() {
            return 0;
        }
        let union = self.interner.union_keys(left, right);
        self.sync_interner_budget();
        union
    }

    #[inline]
    fn memo_get(
        &mut self,
        terminal: u32,
        phase: Phase,
        state_id: u32,
        language: u32,
    ) -> Option<u32> {
        let row = self.memo_rows.get(language as usize)?;
        let output = row
            .iter()
            .find(|entry| {
                entry.terminal == terminal
                    && entry.phase == phase
                    && entry.state_id == state_id
            })?
            .output;
        self.memo_hits += 1;
        Some(output)
    }

    #[inline]
    fn memo_insert(
        &mut self,
        terminal: u32,
        phase: Phase,
        state_id: u32,
        language: u32,
        output: u32,
    ) {
        let language = language as usize;
        if self.memo_rows.len() <= language {
            self.memo_rows.resize_with(language + 1, SmallVec::new);
        }
        debug_assert!(!self.memo_rows[language].iter().any(|entry| {
            entry.terminal == terminal && entry.phase == phase && entry.state_id == state_id
        }));
        self.memo_rows[language].push(TemplateMemoEntry {
            terminal,
            phase,
            state_id,
            output,
        });
        self.memo_entries += 1;
    }

    pub(crate) fn advance_language(
        &mut self,
        constraint: &Constraint,
        terminal: TerminalID,
        language: u32,
    ) -> Option<u32> {
        if self.is_exhausted() {
            return None;
        }
        let serialized_template = constraint
            .template_dfas_by_terminal
            .get(terminal as usize)?
            .as_ref()?;
        let fallback_template;
        let template = if let Some(template) = constraint
            .fast_template_dfas_by_terminal
            .get(terminal as usize)
            .and_then(Option::as_deref)
        {
            template
        } else {
            fallback_template = FastCommitTemplateDfas::from_template(serialized_template);
            &fallback_template
        };
        let advanced = evaluate_template_language(
            template,
            terminal,
            Phase::Pop,
            template.pop.start_state,
            language,
            self,
        );
        (!self.is_exhausted()).then_some(advanced)
    }
}

#[derive(Debug, Clone, Copy)]
struct TemplateMemoEntry {
    terminal: u32,
    phase: Phase,
    state_id: u32,
    output: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Phase {
    Pop,
    Read,
    Push,
}

/// Homomorphic extension of one terminal's compiled stack transducer from
/// individual stacks to finite stack languages.
///
/// If `T_t(s)` is the table-equivalent template result for stack `s`, this
/// function computes `union_{s in L} T_t(s)` for the language ID `L`. Trie
/// branches are language union; target grouping applies the same DFA state to
/// the union of equal-target suffixes. Exactness follows by induction on the
/// acyclic product of template phase/state and canonical stack-trie node.
fn evaluate_template_language(
    template: &FastCommitTemplateDfas,
    terminal: TerminalID,
    phase: Phase,
    state_id: u32,
    language: u32,
    runtime: &mut TemplateAdvanceRuntime,
) -> u32 {
    if language == 0 || runtime.is_exhausted() {
        return 0;
    }
    runtime.calls += 1;
    if let Some(cached) = runtime.memo_get(terminal, phase, state_id, language) {
        return cached;
    }
    if runtime.products_started >= runtime.max_template_products {
        runtime.exhausted = true;
        return 0;
    }
    runtime.products_started += 1;

    let mut output = 0;

    fn merge_target(
        groups: &mut SmallVec<[(u32, u32); 8]>,
        target: u32,
        language: u32,
        runtime: &mut TemplateAdvanceRuntime,
    ) {
        if let Some((_, existing)) = groups
            .iter_mut()
            .find(|(candidate, _)| *candidate == target)
        {
            *existing = runtime.union_languages(*existing, language);
        } else {
            groups.push((target, language));
        }
    }

    match phase {
        Phase::Pop => {
            let Some(dfa_state) = template.pop.states.get(state_id as usize) else {
                return 0;
            };
            if dfa_state.is_accepting {
                output = language;
            }

            let branches = runtime
                .interner
                .top_branches(language)
                .iter()
                .copied()
                .collect::<SmallVec<[(u32, u32); 8]>>();
            let default_target = dfa_state.default_target;
            let mut by_target = SmallVec::<[(u32, u32); 8]>::new();
            for (top, suffix) in branches {
                let target = dfa_state.transitions.get(top as i32).or(default_target);
                if let Some(target) = target {
                    merge_target(&mut by_target, target, suffix, runtime);
                }
            }
            for (target, suffixes) in by_target {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Pop,
                    target,
                    suffixes,
                    runtime,
                );
                output = runtime.union_languages(output, branch);
            }

            if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize) {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Read,
                    *read_state,
                    language,
                    runtime,
                );
                output = runtime.union_languages(output, branch);
            }
            if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Push,
                    *push_state,
                    language,
                    runtime,
                );
                output = runtime.union_languages(output, branch);
            }
        }
        Phase::Read => {
            let Some(dfa_state) = template.read.states.get(state_id as usize) else {
                return 0;
            };
            if dfa_state.is_accepting {
                output = language;
            }

            let branches = runtime
                .interner
                .top_branches(language)
                .iter()
                .copied()
                .collect::<SmallVec<[(u32, u32); 8]>>();
            let mut by_target = SmallVec::<[(u32, u32); 8]>::new();
            for (top, suffix) in branches {
                if let Some(target) = dfa_state.transitions.get(top as i32) {
                    let selected = runtime.interner.push_key(suffix, top);
                    merge_target(&mut by_target, target, selected, runtime);
                }
            }
            for (target, selected) in by_target {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Read,
                    target,
                    selected,
                    runtime,
                );
                output = runtime.union_languages(output, branch);
            }

            if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Push,
                    *push_state,
                    language,
                    runtime,
                );
                output = runtime.union_languages(output, branch);
            }
        }
        Phase::Push => {
            let Some(dfa_state) = template.push.states.get(state_id as usize) else {
                return 0;
            };
            if dfa_state.is_accepting {
                output = language;
            }
            dfa_state.transitions.for_each(|label, target| {
                if !is_negative_label(label) {
                    panic!(
                        "commit template push DFA contains non-push label {label} at state {state_id}"
                    );
                }
                let pushed = runtime.push_language(
                    language,
                    negative_to_positive_label(label) as u32,
                );
                let branch = evaluate_template_language(
                    template,
                    terminal,
                    Phase::Push,
                    target,
                    pushed,
                    runtime,
                );
                output = runtime.union_languages(output, branch);
            });
        }
    }

    if runtime.is_exhausted() {
        return 0;
    }
    runtime.memo_insert(terminal, phase, state_id, language, output);
    output
}

#[cfg(test)]
fn advance_with_template(template: &CommitTemplateDfas, stack: ParserGSS) -> ParserGSS {
    advance_with_prepared_template(template, stack, None)
}

fn advance_with_prepared_template(template: &CommitTemplateDfas, stack: ParserGSS,
    prepared: Option<&FastCommitTemplateDfas>) -> ParserGSS {

    let mut output = ParserGSS::empty();
    // Uniform annotations make branch-order changes immaterial. Keep the
    // established merge order for an input with correlated annotations; this
    // optimization does not redefine weighted-path normalization.
    let sparse_input = stack.single_interface_lower_id().is_some();
    let mut worklist = SmallVec::<[(Phase, u32, ParserGSS); 8]>::new();
    worklist.push((Phase::Pop, template.pop.start_state, stack));
    // Retain every visited source GSS. Raw pointer keys alone are not safe:
    // temporary isolate/push results can be dropped and their addresses reused
    // later in the same evaluation.
    let mut visited = FxHashMap::<(Phase, u32, usize), ParserGSS>::default();

    while let Some((phase, state_id, gss)) = worklist.pop() {
        if gss.is_empty() {
            continue;
        }
        let visit_key = (phase, state_id, gss.ptr_key());
        if let Some(source) = visited.get(&visit_key) {
            debug_assert!(source.ptr_eq(&gss));
            continue;
        }
        visited.insert(visit_key, gss.clone());

        match phase {
            Phase::Pop => {
                let Some(dfa_state) = template.pop.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                if sparse_input {
                    // Intersect the automaton row with the actual GSS frontier,
                    // not with every stack symbol in the grammar. A JSON/JS
                    // template row can contain hundreds of labels while the live
                    // GSS typically exposes only one or two. DEFAULT still has
                    // lower priority than *every* explicit edge, including a
                    // rejecting edge retained as a wildcard exception.
                    if let Some(top) = gss.single_exclusive_top_value() {
                        if let Some(&target) = dfa_state.transitions.get(&(top as i32))
                            .or_else(|| dfa_state.transitions.get(&DEFAULT_LABEL))
                        {
                            worklist.push((Phase::Pop, target, gss.popn(1)));
                        }
                    } else {
                        for top in gss.peek_values() {
                            if let Some(&target) = dfa_state.transitions.get(&(top as i32))
                                .or_else(|| dfa_state.transitions.get(&DEFAULT_LABEL))
                            {
                                let branch = gss.pop_top_value(&top);
                                if !branch.is_empty() {
                                    worklist.push((Phase::Pop, target, branch));
                                }
                            }
                        }
                    }
                } else {
                    // Preserve the original BTreeMap row order for correlated
                    // annotations. Only live top labels can produce a branch;
                    // sorting them preserves the exact old merge order while
                    // avoiding scans of all absent grammar-state labels.
                    let tops = gss.peek_values();
                    let mut explicit_tops = tops.clone();
                    explicit_tops.sort_unstable();
                    for &top in &explicit_tops {
                        let label = top as i32;
                        if label == DEFAULT_LABEL || is_negative_label(label) { continue; }
                        if let Some(&target) = dfa_state.transitions.get(&label) {
                            let isolated = gss.isolate(Some(top));
                            if !isolated.is_empty() {
                                worklist.push((Phase::Pop, target, isolated.popn(1)));
                            }
                        }
                    }
                    if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                        // The old fallback visits remaining GSS top values in
                        // their native order, separately from explicit edges.
                        for top in tops {
                            if dfa_state.transitions.contains_key(&(top as i32)) { continue; }
                            let branch = gss.isolate(Some(top)).popn(1);
                            if !branch.is_empty() {
                                worklist.push((Phase::Pop, target, branch));
                            }
                        }
                    }
                }

                if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize) {
                    worklist.push((Phase::Read, *read_state, gss.clone()));
                }
                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Read => {
                let Some(dfa_state) = template.read.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                if sparse_input {
                    if let Some(top) = gss.single_exclusive_top_value() {
                        if let Some(&target) = dfa_state.transitions.get(&(top as i32)) {
                            worklist.push((Phase::Read, target, gss.clone()));
                        }
                    } else {
                        for top in gss.peek_values() {
                            if let Some(&target) = dfa_state.transitions.get(&(top as i32)) {
                                let branch = gss.isolate(Some(top));
                                if !branch.is_empty() {
                                    worklist.push((Phase::Read, target, branch));
                                }
                            }
                        }
                    }
                } else {
                    let mut tops = gss.peek_values();
                    tops.sort_unstable();
                    for top in tops {
                        let label = top as i32;
                        if is_negative_label(label) { continue; }
                        if let Some(&target) = dfa_state.transitions.get(&label) {
                            let isolated = gss.isolate(Some(top));
                            if !isolated.is_empty() {
                                worklist.push((Phase::Read, target, isolated));
                            }
                        }
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Push => {
                if sparse_input
                    && let Some(plan) = prepared.and_then(|p| p.push_suffixes.get(state_id as usize))
                    .and_then(Option::as_ref)
                    && let Some(outputs) = plan.apply(&gss)
                {
                    output = output.merge(&outputs);
                    continue;
                }
                if sparse_input
                    && let Some(plan) = prepared.and_then(|p| p.push_dag.as_ref())
                    && let Some(outputs) = plan.apply(state_id, &gss)
                {
                    output = output.merge(&outputs);
                    continue;
                }
                let Some(dfa_state) = template.push.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    worklist.push((
                        Phase::Push,
                        target,
                        gss.push(negative_to_positive_label(label) as u32),
                    ));
                }
            }
        }
    }

    output
}

#[cfg(test)]
fn advance_with_template_reference(template: &CommitTemplateDfas, stack: ParserGSS) -> ParserGSS {

    let mut output = ParserGSS::empty();
    let mut worklist = vec![(Phase::Pop, template.pop.start_state, stack)];
    // Retain every visited source GSS. Raw pointer keys alone are not safe:
    // temporary isolate/push results can be dropped and their addresses reused
    // later in the same evaluation.
    let mut visited = FxHashMap::<(Phase, u32, usize), ParserGSS>::default();

    while let Some((phase, state_id, gss)) = worklist.pop() {
        if gss.is_empty() {
            continue;
        }
        let visit_key = (phase, state_id, gss.ptr_key());
        if let Some(source) = visited.get(&visit_key) {
            debug_assert!(source.ptr_eq(&gss));
            continue;
        }
        visited.insert(visit_key, gss.clone());

        match phase {
            Phase::Pop => {
                let Some(dfa_state) = template.pop.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if is_negative_label(label) {
                        panic!(
                            "commit template pop DFA contains push label {label} at state {state_id}"
                        );
                    }
                    if label != DEFAULT_LABEL && label >= 0 {
                        let state = label as u32;
                        let branch = gss.isolate(Some(state)).popn(1);
                        if !branch.is_empty() {
                            worklist.push((Phase::Pop, target, branch));
                        }
                    }
                }
                if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                    for top in gss.peek_values() {
                        if dfa_state.transitions.contains_key(&(top as i32)) {
                            continue;
                        }
                        let branch = gss.isolate(Some(top)).popn(1);
                        if !branch.is_empty() {
                            worklist.push((Phase::Pop, target, branch));
                        }
                    }
                }

                if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize) {
                    worklist.push((Phase::Read, *read_state, gss.clone()));
                }
                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Read => {
                let Some(dfa_state) = template.read.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if label == DEFAULT_LABEL || is_negative_label(label) {
                        panic!(
                            "commit template read DFA contains non-read label {label} at state {state_id}"
                        );
                    }
                    let branch = gss.isolate(Some(label as u32));
                    if !branch.is_empty() {
                        worklist.push((Phase::Read, target, branch));
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Push => {
                let Some(dfa_state) = template.push.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    worklist.push((
                        Phase::Push,
                        target,
                        gss.push(negative_to_positive_label(label) as u32),
                    ));
                }
            }
        }
    }

    output
}

/// Apply a deterministic single-stack commit template to preallocated flat
/// stack scratch. `Some(true)` is an accepting result, `Some(false)` is an
/// empty result, and `None` means the template branches or scratch would grow.
pub(super) fn advance_flat_stack_single_path(
    constraint: &Constraint,
    terminal: TerminalID,
    stack: &mut Vec<u32>,
) -> Option<bool> {
    let template = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    let mut phase = Phase::Pop;
    let mut state_id = template.pop.start_state;
    let total_states = template
        .pop
        .states
        .len()
        .saturating_add(template.read.states.len())
        .saturating_add(template.push.states.len());
    let max_steps = total_states.saturating_mul(2).saturating_add(8);
    let mut steps = 0usize;

    loop {
        let mut choice = None;
        let mut choices = 0usize;
        let accepting;
        match phase {
            Phase::Pop => {
                let dfa_state = template.pop.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;
                if let Some(&top) = stack.last() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    } else if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    }
                    if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize)
                        && template
                            .read
                            .states
                            .get(*read_state as usize)
                            .is_some_and(|state| state.transitions.contains_key(&label))
                    {
                        choice = Some(SingleChoice::Read(*read_state));
                        choices += 1;
                    }
                }
                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Read => {
                let dfa_state = template.read.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;
                if let Some(&top) = stack.last() {
                    if let Some(&target) = dfa_state.transitions.get(&(top as i32)) {
                        choice = Some(SingleChoice::Read(target));
                        choices += 1;
                    }
                }
                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Push => {
                let dfa_state = template.push.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;
                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        return None;
                    }
                    choice = Some(SingleChoice::Push(
                        target,
                        Some(negative_to_positive_label(label) as u32),
                    ));
                    choices += 1;
                }
            }
        }

        if choices == 0 {
            return Some(accepting);
        }
        if accepting || choices > 1 {
            return None;
        }
        match choice? {
            SingleChoice::Pop(target) => {
                stack.pop()?;
                phase = Phase::Pop;
                state_id = target;
            }
            SingleChoice::Read(target) => {
                phase = Phase::Read;
                state_id = target;
            }
            SingleChoice::Push(target, pushed) => {
                if let Some(pushed) = pushed {
                    if stack.len() == stack.capacity() {
                        return None;
                    }
                    stack.push(pushed);
                }
                phase = Phase::Push;
                state_id = target;
            }
        }
        steps += 1;
        if steps > max_steps {
            return None;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SingleChoice {
    Pop(u32),
    Read(u32),
    Push(u32, Option<u32>),
}

fn advance_virtual_stack_single_path(
    template: &CommitTemplateDfas,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
) -> Option<ParserGSS> {
    let mut phase = Phase::Pop;
    let mut state_id = template.pop.start_state;
    let total_states = template
        .pop
        .states
        .len()
        .saturating_add(template.read.states.len())
        .saturating_add(template.push.states.len());
    let max_steps = total_states.saturating_mul(2).saturating_add(8);
    let mut steps = 0usize;

    loop {
        if matches!(phase, Phase::Pop | Phase::Read)
            && stack.top().is_none()
            && stack.has_hidden_floor_values()
        {
            // The visible Segment prefix has been exhausted, but the GSS floor
            // still contains branch-specific parser states. Pop/read decisions
            // depend on those states, so the single-prefix fast path is no
            // longer exact. Return to the branch-aware template worklist.
            return None;
        }

        let mut choice = None;
        let mut choices = 0usize;
        let accepting;

        match phase {
            Phase::Pop => {
                let dfa_state = template.pop.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                debug_assert!(
                    dfa_state
                        .transitions
                        .keys()
                        .all(|&label| !is_negative_label(label)),
                    "commit template pop DFA contains push label at state {state_id}"
                );

                if let Some(top) = stack.top().copied() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    } else if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    }

                    if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize)
                        && template
                            .read
                            .states
                            .get(*read_state as usize)
                            .is_some_and(|state| state.transitions.contains_key(&label))
                    {
                        choice = Some(SingleChoice::Read(*read_state));
                        choices += 1;
                    }
                }

                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Read => {
                let dfa_state = template.read.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                debug_assert!(
                    dfa_state
                        .transitions
                        .keys()
                        .all(|&label| label != DEFAULT_LABEL && !is_negative_label(label)),
                    "commit template read DFA contains non-read label at state {state_id}"
                );

                if let Some(top) = stack.top().copied() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Read(target));
                        choices += 1;
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Push => {
                let dfa_state = template.push.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    choice = Some(SingleChoice::Push(
                        target,
                        Some(negative_to_positive_label(label) as u32),
                    ));
                    choices += 1;
                }
            }
        }

        if choices == 0 {
            return Some(if accepting {
                stack.into_gss()
            } else {
                ParserGSS::empty()
            });
        }
        if accepting || choices > 1 {
            return None;
        }

        match choice.expect("single applicable split template transition") {
            SingleChoice::Pop(target) => {
                if stack.pop(1) != 0 {
                    return Some(ParserGSS::empty());
                }
                phase = Phase::Pop;
                state_id = target;
            }
            SingleChoice::Read(target) => {
                phase = Phase::Read;
                state_id = target;
            }
            SingleChoice::Push(target, pushed) => {
                if let Some(pushed) = pushed {
                    stack.push(pushed);
                }
                phase = Phase::Push;
                state_id = target;
            }
        }

        steps += 1;
        if steps > max_steps {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Phase, TemplateAdvanceRuntime, advance_with_template, evaluate_template_language};
    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::parser::ParserGSS;
    use crate::runtime::{CommitTemplateDfas, FastCommitTemplateDfas};

    #[test]
    fn sparse_frontier_walk_matches_row_scanning_on_generated_dag_programs() {
        use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label};
        let mut seed = 608894u64;
        let mut next = || {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed
        };
        for case in 0..128 {
            let mut pop = UnweightedDfa::new();
            let mut read = UnweightedDfa::new();
            let mut push = UnweightedDfa::new();
            for _ in 1..5 { pop.add_state(); read.add_state(); push.add_state(); }
            for from in 0..5u32 {
                pop.set_accepting(from, next() % 5 == 0);
                read.set_accepting(from, next() % 5 == 0);
                push.set_accepting(from, next() % 3 == 0);
                if from == 4 { continue; }
                for label in 0..7u32 {
                    if next() % 3 == 0 { pop.add_transition(from, label as i32, from + 1 + (next() % (4-from) as u64) as u32); }
                    if next() % 3 == 0 { read.add_transition(from, label as i32, from + 1 + (next() % (4-from) as u64) as u32); }
                    if next() % 5 == 0 { push.add_transition(from, encode_negative_label(label), from + 1 + (next() % (4-from) as u64) as u32); }
                }
                if next() % 2 == 0 { pop.add_transition(from, DEFAULT_LABEL, from + 1); }
            }
            let mut link = || (0..5).map(|_| if next()%3==0 { Some((next()%5) as u32) } else { None }).collect();
            let template = CommitTemplateDfas { pop, read, push,
                pop_to_read:link(), pop_to_push:link(), read_to_push:link() };
            let clean = TerminalsDisallowed::new();
            let guarded = clean.with_insert(0, 5);
            let paths = vec![
                (vec![], clean.clone()), (vec![0], clean.clone()),
                (vec![0,1,4], clean.clone()), (vec![0,2,4], guarded.clone()),
                (vec![0,6,5], guarded.clone()), (vec![3,1], clean.clone()),
            ];
            for group in [&paths[..1], &paths[1..2], &paths[2..4], &paths[1..], &paths[..]] {
                let source = ParserGSS::from_stacks(group);
                let expected = super::advance_with_template_reference(&template, source.clone());
                let fast = crate::runtime::FastCommitTemplateDfas::from_template(&template);
                let accelerated = super::advance_with_prepared_template(&template, source.clone(), Some(&fast));
                let actual = advance_with_template(&template, source);
                assert_eq!(accelerated.semantically_eq(&expected, 16_384), Some(true),
                    "prepared PUSH differs, case={case}, paths={group:?}");
                assert_eq!(actual.semantically_eq(&expected, 16_384), Some(true),
                    "sparse frontier relation differs, case={case}, paths={group:?}");
            }
        }
    }

    #[test]
    fn template_advance_distributes_over_merged_branched_floor() {
        let mut pop = UnweightedDfa::new();
        let after_common = pop.add_state();
        let after_left = pop.add_state();
        let after_right = pop.add_state();
        pop.add_transition(pop.start_state, 10, after_common);
        pop.add_transition(after_common, 1, after_left);
        pop.add_transition(after_common, 2, after_right);
        pop.set_accepting(after_left, true);
        pop.set_accepting(after_right, true);

        let template = CommitTemplateDfas {
            pop,
            read: UnweightedDfa::default(),
            push: UnweightedDfa::default(),
            pop_to_read: vec![None; 4],
            pop_to_push: vec![None; 4],
            read_to_push: Vec::new(),
        };
        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 1, 10], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 2, 10], acc);
        let merged = left.merge(&right);

        let expected = advance_with_template(&template, left)
            .merge(&advance_with_template(&template, right));
        let actual = advance_with_template(&template, merged);

        let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks);
    }


    #[test]
    fn canonical_language_evaluation_matches_gss_template_walker() {
        let mut pop = UnweightedDfa::new();
        let after_common = pop.add_state();
        let after_left = pop.add_state();
        let after_right = pop.add_state();
        pop.add_transition(pop.start_state, 10, after_common);
        pop.add_transition(after_common, 1, after_left);
        pop.add_transition(after_common, 2, after_right);
        pop.set_accepting(after_left, true);
        pop.set_accepting(after_right, true);

        let template = CommitTemplateDfas {
            pop,
            read: UnweightedDfa::default(),
            push: UnweightedDfa::default(),
            pop_to_read: vec![None; 4],
            pop_to_push: vec![None; 4],
            read_to_push: Vec::new(),
        };
        let acc = TerminalsDisallowed::new();
        let merged = ParserGSS::from_single_stack(vec![0, 1, 10], acc.clone()).merge(
            &ParserGSS::from_single_stack(vec![0, 2, 10], acc.clone()),
        );
        let expected = advance_with_template(&template, merged.clone());

        let mut runtime = TemplateAdvanceRuntime::default();
        let (language, accumulator) = runtime
            .language_from_uniform_gss(&merged)
            .expect("test GSS has one uniform accumulator");
        let fast_template = FastCommitTemplateDfas::from_template(&template);
        let output = evaluate_template_language(
            &fast_template,
            0,
            Phase::Pop,
            fast_template.pop.start_state,
            language,
            &mut runtime,
        );
        let actual = runtime.gss_from_language(output, accumulator);
        assert!(
            actual
                .semantically_eq(&expected, 4_096)
                .expect("test languages should remain explicitly bounded")
        );
    }

    #[test]
    fn template_product_budget_declines_transactionally() {
        let mut pop = UnweightedDfa::new();
        let after_top = pop.add_state();
        let accepting = pop.add_state();
        pop.add_transition(pop.start_state, 2, after_top);
        pop.add_transition(after_top, 1, accepting);
        pop.set_accepting(accepting, true);
        let template = CommitTemplateDfas {
            pop,
            read: UnweightedDfa::default(),
            push: UnweightedDfa::default(),
            pop_to_read: vec![None; 3],
            pop_to_push: vec![None; 3],
            read_to_push: Vec::new(),
        };
        let input = ParserGSS::from_single_stack(
            vec![0, 1, 2],
            TerminalsDisallowed::new(),
        );
        let mut runtime = TemplateAdvanceRuntime::with_budget(64, 64, 64, 1);
        let (language, _) = runtime
            .language_from_uniform_gss(&input)
            .expect("input canonicalization fits its independent budget");
        let fast_template = FastCommitTemplateDfas::from_template(&template);
        let output = evaluate_template_language(
            &fast_template,
            0,
            Phase::Pop,
            fast_template.pop.start_state,
            language,
            &mut runtime,
        );
        assert_eq!(output, 0);
        assert!(runtime.is_exhausted());
        assert_eq!(
            runtime.memo_entries, 0,
            "partial products must not be published"
        );
    }

    #[test]
    fn template_walker_retains_temporary_gss_identity_across_pop_read_and_push() {
        let mut pop = UnweightedDfa::new();
        let popped = pop.add_state();
        pop.add_transition(pop.start_state, 9, popped);

        let mut read = UnweightedDfa::new();
        let read_left = read.add_state();
        let read_right = read.add_state();
        read.add_transition(read.start_state, 1, read_left);
        read.add_transition(read.start_state, 2, read_right);

        let mut push = UnweightedDfa::new();
        let pushed_left = push.add_state();
        let pushed_right = push.add_state();
        push.add_transition(
            push.start_state,
            crate::compiler::glr::labels::encode_negative_label(20),
            pushed_left,
        );
        push.add_transition(
            push.start_state,
            crate::compiler::glr::labels::encode_negative_label(30),
            pushed_right,
        );
        push.set_accepting(pushed_left, true);
        push.set_accepting(pushed_right, true);

        let mut pop_to_read = vec![None; pop.states.len()];
        pop_to_read[popped as usize] = Some(read.start_state);
        let mut read_to_push = vec![None; read.states.len()];
        read_to_push[read_left as usize] = Some(push.start_state);
        read_to_push[read_right as usize] = Some(push.start_state);
        let template = CommitTemplateDfas {
            pop,
            read,
            push,
            pop_to_read,
            pop_to_push: vec![None; 2],
            read_to_push,
        };

        let acc_a = TerminalsDisallowed::new();
        let acc_b = TerminalsDisallowed::new().with_insert(0, 7);
        let input = ParserGSS::from_stacks(&[
            (vec![0, 1, 9], acc_a.clone()),
            (vec![0, 2, 9], acc_b.clone()),
        ]);
        let actual = advance_with_template(&template, input);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 1, 20], acc_a.clone()),
            (vec![0, 1, 30], acc_a),
            (vec![0, 2, 20], acc_b.clone()),
            (vec![0, 2, 30], acc_b),
        ]);
        assert!(
            actual
                .semantically_eq(&expected, 64)
                .expect("test output is explicitly bounded")
        );
    }

}
