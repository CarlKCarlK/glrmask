//! Scoped adapter for the existing ordinary epsilon/virtual lexer executor.
//!
//! Composition supplies namespaces and CALL/RETURN reset routing. Vocabulary
//! traversal, exact lexer execution, guard advancement and product caching are
//! the ordinary masking implementations, not another composition mask walk.
use super::*;

#[derive(Clone, Copy)]
struct ScopedConfig {
    leaf: usize,
    local: u32,
}

struct RecursiveConfigTransitions<'scan, 'constraint> {
    pre_admitted_original_mask: Option<&'scan [u32]>,
    routing: RecursiveFullWalkTransitions<'constraint>,
    tables: Vec<FullWalkConfigTransitions<'scan, 'constraint>>,
    states: Vec<ScopedConfig>,
    ids: FxHashMap<(usize, u32), u32>,
    roots: FxHashMap<u32, u32>,
    resets: Vec<Option<u32>>,
    rows: Vec<Option<Box<[u64; 256]>>>,
    candidate_futures: Vec<Option<BitSet>>,
    boundary: FxHashMap<(u32, u32), bool>,
    projected_resets: FxHashMap<(usize, u32), u32>,
    project_resets: bool,
    error: Option<String>,
}

impl<'scan, 'constraint> RecursiveConfigTransitions<'scan, 'constraint> {
    fn intern(&mut self, leaf: usize, local: u32) -> Result<u32, String> {
        if let Some(&id) = self.ids.get(&(leaf, local)) { return Ok(id); }
        let id = u32::try_from(self.states.len())
            .ok().filter(|&id| id < u32::MAX - 4)
            .ok_or_else(|| "recursive config namespace exhausted".to_owned())?;
        self.states.push(ScopedConfig { leaf, local });
        self.ids.insert((leaf, local), id);
        self.rows.push(None);
        self.candidate_futures.push(None);
        Ok(id)
    }

    fn fail(&mut self, error: String) {
        if self.error.is_none() { self.error = Some(error); }
    }

    fn projected_reset(&mut self, lexer: u32, parser: u32, cache: &FullWalkParserCache) -> u32 {
        let ScopedConfig { leaf, local } = self.states[lexer as usize];
        if !self.project_resets || !self.tables[leaf].parser_projection_enabled() { return lexer; }
        if let Some(&cached) = self.projected_resets.get(&(leaf, parser)) { return cached; }
        let source = self.routing.leaves[leaf].constraint;
        // A row-based upper bound is allowed only for a control-free leaf;
        // otherwise retain the ordinary unprojected exact configuration.
        if !source.table.control_terminals.is_empty() { return lexer; }
        let mut admitted = BitSet::new(source.table.num_terminals as usize);
        for top in cache.nodes[parser as usize].gss.peek_values() {
            let Some((owner, state)) = self.routing.constraint.recursive_parser_leaf_state(top)
                else { return lexer; };
            if owner != leaf { return lexer; }
            let Some(row) = source.table.advance_row(state) else { return lexer; };
            for terminal in row.iter_ones().take_while(|&t| t < source.table.num_terminals as usize) {
                admitted.set(terminal);
            }
        }
        if let Some(ignore) = source.ignore_terminal { admitted.set(ignore as usize); }
        for &skip in &source.table.skip_terminals { admitted.set(skip as usize); }
        if admitted.count_ones() > 16 || admitted.is_empty() { return lexer; }
        let projected = self.tables[leaf].parser_initial_state(local, &admitted);
        let result = match self.intern(leaf, projected) {
            Ok(id) => id,
            Err(error) => { self.fail(error); lexer }
        };
        self.projected_resets.insert((leaf, parser), result);
        result
    }

    fn finish(self) -> Result<(), String> {
        if let Some(error) = self.error { return Err(error); }
        for table in self.tables { table.finish()?; }
        Ok(())
    }

    /// Cheap support only. A virtual residual may not actually extend every
    /// terminal listed here; consumers must still call exact future_contains.
    fn candidate_scoped_future(&mut self, state: u32) -> &BitSet {
        if self.candidate_futures[state as usize].is_none() {
            let ScopedConfig { leaf, local } = self.states[state as usize];
            let descriptor = self.routing.leaves[leaf];
            let table = &mut self.tables[leaf];
            let config = table.generic_config_for_state(local);
            let mut candidates = BitSet::new(descriptor.terminal_count as usize);
            if let Some(index) = table.cache.config_index(config).filter(|_| !table.cache.deterministic) {
                // The ordinary config executor already maintains this union,
                // including any admitted-output filter. Do not rebuild it by
                // scanning every physical member at each scoped boundary.
                candidates.union_with(&table.cache.config_futures[index]);
            } else {
                for index in 0..table.cache.config_len(config) {
                    let raw = table.cache.config_state(config, index);
                    candidates.union_with(table.cache.tokenizer().possible_future_terminals(raw));
                }
            }
            let terminal_count = self.routing.leaves.last().map_or(0, |last| {
                last.terminal_offset as usize + last.terminal_count as usize
            });
            let mut scoped_candidates = BitSet::new(terminal_count);
            for terminal in candidates.iter_ones() {
                scoped_candidates.set(descriptor.terminal_offset as usize + terminal);
            }
            self.candidate_futures[state as usize] = Some(scoped_candidates);
        }
        self.candidate_futures[state as usize].as_ref().expect("candidates initialized above")
    }
}

impl FullWalkTransitionTable for RecursiveConfigTransitions<'_, '_> {
    type Cell = FullWalkConfigCell;

    fn pre_admitted_original_mask(&self) -> Option<&[u32]> { self.pre_admitted_original_mask }

    #[inline]
    fn cell(&mut self, state: u32, byte: u8) -> Self::Cell {
        const UNKNOWN: u64 = u64::MAX;
        if let Some(row) = self.rows[state as usize].as_ref() {
            let packed = row[byte as usize];
            if packed != UNKNOWN {
                return FullWalkConfigCell {
                    target: packed as u32,
                    has_finalizer: (packed >> 32) != 0,
                };
            }
        }
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let cell = self.tables[leaf].cell(local, byte);
        let target = if cell.target == u32::MAX {
            u32::MAX
        } else {
            match self.intern(leaf, cell.target) {
                Ok(target) => target,
                Err(error) => { self.fail(error); u32::MAX }
            }
        };
        let row = self.rows[state as usize].get_or_insert_with(|| Box::new([UNKNOWN; 256]));
        row[byte as usize] = u64::from(target) | (u64::from(cell.has_finalizer) << 32);
        FullWalkConfigCell { target, has_finalizer: cell.has_finalizer }
    }

    #[inline(always)]
    fn cell_is_dead(cell: Self::Cell) -> bool { cell.target == u32::MAX }
    #[inline(always)]
    fn cell_has_finalizer(cell: Self::Cell) -> bool { cell.has_finalizer }
    #[inline(always)]
    fn cell_target(cell: Self::Cell) -> u32 { cell.target }

    fn root_state(&mut self, scoped_raw: u32) -> Result<u32, String> {
        if let Some(&id) = self.roots.get(&scoped_raw) { return Ok(id); }
        let (leaf, raw) = self.routing.constraint.recursive_tokenizer_leaf_state(scoped_raw)
            .ok_or_else(|| format!("recursive lexer root {scoped_raw} is not scoped"))?;
        let local = self.tables[leaf].root_state(raw)?;
        let id = self.intern(leaf, local)?;
        self.roots.insert(scoped_raw, id);
        if self.routing.leaves[leaf].reset == scoped_raw { self.resets[leaf] = Some(id); }
        Ok(id)
    }

    fn validate_mask_result(&mut self) -> Result<(), String> {
        if let Some(error) = self.error.as_ref() { return Err(error.clone()); }
        for table in &mut self.tables { table.validate_mask_result()?; }
        Ok(())
    }

    fn root_state_for_parser(&mut self, scoped_raw: u32, parser: &ParserStacks) -> Result<u32, String> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| env_flag("GLRMASK_PROVIDER_INITIAL_ROOT_PROJECTION", true)) {
            return self.root_state(scoped_raw);
        }
        static DIRECT: OnceLock<bool> = OnceLock::new();
        let direct = *DIRECT.get_or_init(|| env_flag("GLRMASK_PROVIDER_DIRECT_ROOT_PROJECTION", true));
        // A projected root need not first build the much larger unprojected
        // epsilon closure. Keep that work in the reference branch for A/B.
        let normal_root = if direct { None } else { Some(self.root_state(scoped_raw)?) };
        let fallback = |this: &mut Self| {
            if let Some(root) = normal_root { Ok(root) } else { this.root_state(scoped_raw) }
        };
        let (leaf, raw) = self.routing.constraint.recursive_tokenizer_leaf_state(scoped_raw)
            .ok_or_else(|| format!("recursive lexer root {scoped_raw} is not scoped"))?;
        let is_reset = self.routing.leaves[leaf].reset == scoped_raw;
        static PARTIAL: OnceLock<bool> = OnceLock::new();
        let allow_partial = direct && *PARTIAL.get_or_init(|| {
            env_flag("GLRMASK_PROVIDER_PARTIAL_ROOT_PROJECTION", true)
        });
        let source = self.routing.leaves[leaf].constraint;
        if (!is_reset && !allow_partial)
            || !self.tables[leaf].parser_projection_enabled()
            || !parser.isolate(None).is_empty()
            || (!is_reset && (source.tokenizer.has_any_virtual_runtime()
                || raw >= source.tokenizer.num_states()))
        { return fallback(self); }
        if !source.table.control_terminals.is_empty() { return fallback(self); }
        let layout = self.routing.constraint.recursive_parser_layout()?
            .ok_or("missing recursive layout for initial projection")?;
        let mut admitted = BitSet::new(source.table.num_terminals as usize);
        for top in parser.peek_values() {
            let Some((owner, state)) = self.routing.constraint.recursive_parser_leaf_state(top)
                else { return fallback(self); };
            if owner != leaf { return fallback(self); }
            // Synthetic provider calls/returns need not occur in the local
            // table's control-terminal list. Preserve the full interpreter
            // whenever a zero-width owner change cannot be ruled out cheaply.
            if source.table.action(state, u32::MAX).is_some()
                || layout.links.iter().any(|link| {
                    (link.parent_component as usize == leaf
                        && source.table.action(state, link.slot_terminal).is_some())
                    || (link.child_component as usize == leaf
                        && link.child_start_nullable && state == link.child_start)
                })
            { return fallback(self); }
            let Some(row) = source.table.advance_row(state) else { return fallback(self); };
            for terminal in row.iter_ones().take_while(|&t| t < source.table.num_terminals as usize) {
                admitted.set(terminal);
            }
        }
        if let Some(ignore) = source.ignore_terminal { admitted.set(ignore as usize); }
        for &skip in &source.table.skip_terminals { admitted.set(skip as usize); }
        let count = admitted.count_ones();
        if count == 0 || count > 16 { return fallback(self); }
        if !is_reset {
            // If every output colour already belongs to A, adding an admitted
            // filter only introduces work. Matched and future colours both
            // matter at a partial root; pruning memories remain independent.
            let future = source.tokenizer.possible_future_terminals(raw);
            let subset = future.words().iter().enumerate().all(|(i, &word)| {
                word & !admitted.words().get(i).copied().unwrap_or(0) == 0
            }) && source.tokenizer.matched_terminals_iter(raw)
                .all(|terminal| admitted.contains(terminal as usize));
            if subset { return fallback(self); }
        }
        let projected = if direct {
            // These are exactly the ordinary lexer projection and fresh-reset
            // operations used by parser_initial_state. No scoped/raw ID is
            // injected into the local configuration namespace.
            let cache = &mut self.tables[leaf].cache;
            let Some(config) = cache.config_for_parser_admitted(raw, &admitted)?
                else { return fallback(self); };
            // A partial root must NOT gain fresh-reset/epsilon token-end
            // semantics. Only actual reset roots receive that wrapper.
            if is_reset { cache.fresh_reset_config(config)? } else { config }
        } else {
            let root = normal_root.expect("reference root was prepared");
            let local = self.states[root as usize].local;
            self.tables[leaf].parser_initial_state(local, &admitted)
        };
        static PROFILE: OnceLock<bool> = OnceLock::new();
        if *PROFILE.get_or_init(||std::env::var_os("GLRMASK_PROFILE_INITIAL_ROOT_PROJECTION").is_some()) {
            eprintln!("[glrmask/profile][initial_root_projection] raw={} leaf={} admitted={} direct={} reset={} configs={}",
                scoped_raw, leaf, count, direct, is_reset, self.tables[leaf].cache.configs.len());
        }
        // Do not cache this under roots[scoped_raw]: another correlated parser
        // frontier may admit a different terminal set for that same lexer root.
        self.intern(leaf, projected)
    }

    fn walk_initial_state(&mut self, constraint: &Constraint, _: &DynamicMaskVocab) -> Result<u32, String> {
        debug_assert!(std::ptr::eq(constraint, self.routing.constraint));
        self.root_state(self.routing.leaves[0].reset)
    }

    fn finalizer_code(&self, state: u32) -> u32 {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let code = self.tables[leaf].finalizer_code(local);
        if code >= u32::MAX - 1 { code } else { self.routing.leaves[leaf].terminal_offset + code }
    }

    fn single_finalizer_continues(&mut self, state: u32) -> bool {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        self.tables[leaf].single_finalizer_continues(local)
    }

    fn matched_terminals(&self, state: u32) -> SmallVec<[TerminalID; 4]> {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let base = self.routing.leaves[leaf].terminal_offset;
        self.tables[leaf].matched_terminals(local).into_iter().map(|t| base + t).collect()
    }

    fn future_contains(&mut self, state: u32, terminal: TerminalID) -> bool {
        let ScopedConfig { leaf, local } = self.states[state as usize];
        let descriptor = self.routing.leaves[leaf];
        let Some(local_terminal) = terminal.checked_sub(descriptor.terminal_offset) else { return false; };
        local_terminal < descriptor.terminal_count && self.tables[leaf].future_contains(local, local_terminal)
    }

    fn future_intersects(&mut self, state: u32, terminals: &BitSet) -> bool {
        let candidates = self.candidate_scoped_future(state).clone();
        candidates.iter_ones().any(|terminal| {
            terminals.contains(terminal) && self.future_contains(state, terminal as u32)
        })
    }

    fn merge_states(&mut self, states: &[u32]) -> Option<u32> {
        let first = *states.first()?;
        if states.iter().all(|&state| state == first) { return Some(first); }
        // The driver proves equal parser languages and discharged guards.
        // Within one scoped leaf, use the ordinary exact lazy-subset union
        // instead of carrying equivalent lexical lanes as separate branches.
        let leaf = self.states.get(first as usize)?.leaf;
        let mut locals = SmallVec::<[u32; 4]>::new();
        for &state in states {
            let state = self.states.get(state as usize)?;
            if state.leaf != leaf { return None; }
            locals.push(state.local);
        }
        let local = self.tables[leaf].merge_states(&locals)?;
        match self.intern(leaf, local) {
            Ok(state) => Some(state),
            Err(error) => { self.fail(error); None }
        }
    }

    #[inline(always)]
    fn dense_state_count(&self) -> Option<usize> { None }
    #[inline(always)]
    fn exact_raw_state(&self, _: u32) -> Option<u32> { None }
    #[inline(always)]
    fn uses_scoped_reset_routing(&self) -> bool { true }
    #[inline(always)]
    fn parser_conditioned_dead_skip_default(&self) -> bool { true }
    #[inline(always)]
    fn prefer_adaptive_output(&self) -> bool { true }
    #[inline(always)]
    fn product_transition_cache_capacity(&self) -> usize {
        // General scoped lexer subsets need more distinct product frontiers
        // than scalar finite leaves. Grow lazily, with a fixed upper bound;
        // retain the exact test override for disabled/tiny/exhausted caches.
        #[cfg(test)]
        { self.routing.product_transition_cache_capacity() }
        #[cfg(not(test))]
        { 2048 }
    }
    #[inline(always)]
    fn cache_two_branch_products(&self) -> bool { true }
    fn cache_single_branch_products(&self) -> bool { true }

    #[inline]
    fn terminal_is_ignore(&self, constraint: &Constraint, terminal: TerminalID) -> bool {
        self.routing.terminal_is_ignore(constraint, terminal)
    }

    fn scoped_reset_branches(
        &mut self, parser_cache: &mut FullWalkParserCache, constraint: &Constraint,
        parser_node: u32, terminal: TerminalID,
    ) -> SmallVec<[(u32, u32); 4]> {
        let raw_resets = self.routing.scoped_reset_branches(parser_cache, constraint, parser_node, terminal);
        let mut result = SmallVec::new();
        for (raw_reset, parser) in raw_resets {
            match self.root_state(raw_reset) {
                Ok(lexer) => result.push((self.projected_reset(lexer, parser, parser_cache), parser)),
                Err(error) => self.fail(error),
            }
        }
        result
    }

    fn token_boundary_allowed(
        &mut self, parser_cache: &mut FullWalkParserCache, constraint: &Constraint,
        _: u32, lexer_state: u32, parser_node: u32,
    ) -> bool {
        let ScopedConfig { leaf, local } = self.states[lexer_state as usize];
        if self.resets[leaf] == Some(lexer_state) { return true; }
        let config = self.tables[leaf].generic_config_for_state(local);
        if !self.tables[leaf].cache.deterministic
            && self.tables[leaf].cache.config_index(config).is_some_and(|i|
                self.tables[leaf].cache.config_is_fresh_reset[i]) { return true; }
        if let Some(&value) = self.boundary.get(&(lexer_state, parser_node)) { return value; }
        let ignored = self.routing.leaves[leaf].constraint.ignore_terminal
            .map(|terminal| self.routing.leaves[leaf].terminal_offset + terminal);
        let allowed = ignored.is_some_and(|terminal| self.future_contains(lexer_state, terminal)) || {
            let candidates = self.candidate_scoped_future(lexer_state).clone();
            let gss = self.routing.parser_gss(parser_node, parser_cache);
            constraint.compact_segmented_parser_may_advance_on_any_matching(
                &gss,
                candidates.iter_ones().map(|terminal| terminal as u32),
                |terminal| self.future_contains(lexer_state, terminal),
            ).unwrap_or(false)
        };
        self.boundary.insert((lexer_state, parser_node), allowed);
        allowed
    }
}

pub(super) fn fill(state: &ConstraintState<'_>, buf: &mut [u32]) -> Result<bool, String> {
    let vocab = state.constraint.dynamic_mask_vocab_for_runtime();
    fill_with_vocab(state, buf, vocab)
}

pub(super) fn fill_with_vocab(
    state: &ConstraintState<'_>, buf: &mut [u32], vocab: &DynamicMaskVocab,
) -> Result<bool, String> {
    with_provider(state, vocab.max_token_byte_len(), |provider| {
        fill_recursive_mask_using_vocab(state, buf, provider, vocab)
    })
}

pub(super) fn fill_with_vocab_factory<V, F>(
    state: &ConstraintState<'_>, buf: &mut [u32], domain: PureMaskByteDomain, factory: F,
) -> Result<bool, String>
where
    V: std::ops::Deref<Target = DynamicMaskVocab>,
    F: FnOnce() -> Result<V, String>,
{
    with_provider(state, domain.max_token_len, |provider| {
        fill_pure_byte_mask_using_factory(state, buf, provider, domain, factory)
    })
}

fn with_provider(
    state: &ConstraintState<'_>, max_token_len: usize,
    evaluate: impl FnOnce(&mut RecursiveConfigTransitions<'_, '_>) -> Result<bool, String>,
) -> Result<bool, String> {
    with_provider_and_baseline(state, max_token_len, None, evaluate)
}

pub(super) fn fill_with_vocab_additive(
    state: &ConstraintState<'_>, buf: &mut [u32], vocab: &DynamicMaskVocab, already: &[u32],
) -> Result<bool, String> {
    with_provider_and_baseline(state, vocab.max_token_byte_len(), Some(already), |provider| {
        fill_recursive_mask_using_vocab(state, buf, provider, vocab)
    })
}

pub(super) fn fill_with_vocab_factory_additive<V, F>(
    state: &ConstraintState<'_>, buf: &mut [u32], domain: PureMaskByteDomain,
    already: &[u32], factory: F,
) -> Result<bool, String>
where V: std::ops::Deref<Target = DynamicMaskVocab>, F: FnOnce() -> Result<V, String> {
    with_provider_and_baseline(state, domain.max_token_len, Some(already), |provider| {
        fill_pure_byte_mask_using_factory(state, buf, provider, domain, factory)
    })
}

fn with_provider_and_baseline(
    state: &ConstraintState<'_>, max_token_len: usize, already: Option<&[u32]>,
    evaluate: impl FnOnce(&mut RecursiveConfigTransitions<'_, '_>) -> Result<bool, String>,
) -> Result<bool, String> {
    let profile = std::env::var_os("GLRMASK_PROFILE_RECURSIVE_PHASES").is_some();
    let setup_start = profile.then(std::time::Instant::now);
    let routing = RecursiveFullWalkTransitions::new_for_parser_routing(state.constraint)
        .ok_or_else(|| "recursive composition has no valid scoped lexer/parser layout".to_owned())?;
    let mut scans = routing.leaves.iter()
        .map(|leaf| DynamicNfaScanCache::new(leaf.constraint, None)).collect::<Vec<_>>();
    let tables = scans.iter_mut().map(|scan| {
        FullWalkConfigTransitions::new(scan, max_token_len, state.generation)
    }).collect();
    let reset_count = routing.leaves.len();
    let mut provider = RecursiveConfigTransitions {
        pre_admitted_original_mask: already,
        routing, tables, states: Vec::new(), ids: FxHashMap::default(),
        roots: FxHashMap::default(), resets: vec![None; reset_count], rows: Vec::new(),
        candidate_futures: Vec::new(), boundary: FxHashMap::default(),
        projected_resets: FxHashMap::default(),
        project_resets: std::env::var_os("GLRMASK_PROFILE_SCOPED_RESET_PROJECTION").is_some(),
        error: None,
    };
    if let Some(start) = setup_start {
        eprintln!("[glrmask/profile][recursive_phases] setup_ns={}", start.elapsed().as_nanos());
    }
    let walk_start = profile.then(std::time::Instant::now);
    let result = evaluate(&mut provider);
    if let Some(start) = walk_start {
        eprintln!("[glrmask/profile][recursive_phases] walk_ns={}", start.elapsed().as_nanos());
    }
    let finish_start = profile.then(std::time::Instant::now);
    provider.finish()?;
    drop(scans);
    if let Some(start) = finish_start {
        eprintln!("[glrmask/profile][recursive_phases] finish_ns={}", start.elapsed().as_nanos());
    }
    result
}
