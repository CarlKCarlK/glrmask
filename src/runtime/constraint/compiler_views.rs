//! Retained compiler views and lazy reconstruction for compiled composition.

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::tokenizer::TokenizerStateSet;
use crate::automata::regex::Expr;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::glr::table::subgrammar_child_return_pop;
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::runtime::artifact::Constraint;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::OnceLock;

impl Constraint {

    /// Build the parser-state-independent trigger level used by dynamic
    /// composition. This is deliberately optional: ordinary Constraint and
    /// DynamicConstraint compilation never calls it, preserving zero trigger
    /// build cost by default.
    ///
    /// The scan starts from every tokenizer state, so the resulting original
    /// model-token set is conservative for any runtime TSID. A token is kept
    /// when a proper prefix can complete a terminal that may either finish the
    /// component or immediately precede one of its unresolved grammar slots.
    pub(crate) fn build_boundary_token_trigger(&mut self) -> Result<(), String> {
        self.materialize_composition_link_metadata_for_compilation()?;

        let mut relevant = BitSet::new(self.table.num_terminals as usize);
        if let Some(summary) = self.composition_grammar_summary.as_ref() {
            relevant.union_with(&summary.root_last);
            let placeholders = self
                .unbound_grammar_placeholders
                .values()
                .copied()
                .chain(self.late_grammar_slots.iter().map(|slot| slot.terminal_id));
            for placeholder in placeholders {
                for (terminal, follows) in summary.allowed_follows.iter().enumerate() {
                    if follows.contains(placeholder as usize) {
                        relevant.set(terminal);
                    }
                }
            }
            // Scoped ignore can occur immediately before an entry/finish point
            // without appearing in grammar adjacency. A model token may
            // therefore consume one or more ignore lexemes and then cross the
            // component boundary internally. Tokens is intentionally an
            // overapproximation, so treating any matched ignore as a possible
            // boundary precursor is the sound parser-state-independent choice.
            if let Some(ignore) = self.ignore_terminal {
                relevant.set(ignore as usize);
            }
        } else {
            // Older/stripped artifacts may lack the grammar summary. The
            // Tokens trigger is only a pruning accelerator, so falling back to
            // every terminal preserves exactness.
            relevant = BitSet::all(self.table.num_terminals as usize);
        }

        if relevant.is_empty() {
            self.boundary_trigger =
                crate::runtime::BoundaryTrigger::Tokens(Arc::from([]));
            self.serialized_artifact_cache = None;
            return Ok(());
        }

        let all_states = (0..self.tokenizer.num_states()).collect::<Vec<_>>();
        let tokens = self.token_bytes_iter().collect::<Vec<_>>();
        let mut candidates = tokens
            .par_iter()
            .filter_map(|(token_id, bytes)| {
                if bytes.len() < 2 {
                    return None;
                }
                let mut states = TokenizerStateSet::from_iter(all_states.iter().copied());
                for &byte in &bytes[..bytes.len() - 1] {
                    states = self.tokenizer.step_all(states.as_slice(), byte);
                    if states.is_empty() {
                        return None;
                    }
                    let mut matched_any = false;
                    let mut matched_relevant = false;
                    for state in states.iter().copied() {
                        for terminal in self.tokenizer.matched_terminals_iter(state) {
                            matched_any = true;
                            matched_relevant |= relevant.contains(terminal as usize);
                        }
                    }
                    if matched_relevant {
                        return Some(*token_id);
                    }
                    if matched_any {
                        // A real lexer commits only according to its exact
                        // longest-match policy. Forking a reset continuation at
                        // every match is deliberately broader: it preserves all
                        // real multi-lexeme paths (including ignore -> reset ->
                        // terminal -> boundary) while admitting only harmless
                        // false-positive trigger tokens.
                        let reset = self.runtime_commit_initial_state();
                        if !states.contains(&reset) {
                            states.push(reset);
                            states.sort_unstable();
                        }
                    }
                }
                None
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        candidates.dedup();
        self.boundary_trigger = crate::runtime::BoundaryTrigger::Tokens(Arc::from(
            candidates.into_boxed_slice(),
        ));
        self.serialized_artifact_cache = None;
        Ok(())
    }

    /// Build reusable boundary-trigger metadata at the requested detail level.
    ///
    /// `None` is a no-op and preserves the zero-cost default. `Tokens` builds
    /// only the conservative parser-state-independent token set. `Exact` first
    /// builds that set as a candidate prefilter, then upgrades to a full local
    /// Parser DWA when this component supports exact trigger compilation.
    pub fn build_boundary_trigger(
        &mut self,
        detail: crate::runtime::BoundaryTriggerDetail,
    ) -> Result<(), String> {
        match detail {
            crate::runtime::BoundaryTriggerDetail::None => Ok(()),
            crate::runtime::BoundaryTriggerDetail::Tokens => self.build_boundary_token_trigger(),
            crate::runtime::BoundaryTriggerDetail::Exact => self.build_exact_boundary_trigger(),
        }
    }

    /// Build the full GSS-sensitive proper-prefix boundary trigger Parser DWA.
    ///
    /// This is an optional accelerator for dynamic composition and is
    /// deliberately separate from ordinary constraint compilation. The trigger
    /// uses raw local tokenizer-state IDs and original model-token IDs rather
    /// than the component's whole-token TSID/token quotient. Recursive
    /// coordinators build it from private compiler-materialized table/tokenizer
    /// views; those flattened views are never reattached to live runtime state.
    pub fn build_exact_boundary_trigger(&mut self) -> Result<(), String> {
        if matches!(self.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_)) {
            return Ok(());
        }

        // Recursive coordinators deliberately keep only leaf-native runtime
        // parser/tokenizer views. Exact-trigger construction is compiler work:
        // it is still defined over the exact flattened component coordinate,
        // so materialize those compiler-only views in a private clone rather
        // than reattaching them to the live constraint. The resulting trigger
        // remains an optional accelerator; recursive outer runtimes decline its
        // materialized parser coordinate and fall back to exact scoped commits.
        let dwa = if self.uses_compact_segmented_parser_runtime() {
            let mut compiler_view = self.clone();
            compiler_view.prepare_recursive_compiler_table_for_composition()?;
            compiler_view.prepare_recursive_compiler_tokenizer_for_composition()?;
            compiler_view.build_boundary_token_trigger()?;
            let candidates = compiler_view
                .boundary_trigger
                .token_summary()
                .map(|tokens| tokens.to_vec())
                .unwrap_or_default();
            crate::compiler::composition::build_exact_component_boundary_trigger(
                &compiler_view,
                &candidates,
            )?
        } else {
            self.build_boundary_token_trigger()?;
            let candidates = self
                .boundary_trigger
                .token_summary()
                .map(|tokens| tokens.to_vec())
                .unwrap_or_default();
            crate::compiler::composition::build_exact_component_boundary_trigger(
                self,
                &candidates,
            )?
        };
        let Some(dwa) = dwa else {
            return Err(
                "exact boundary trigger construction could not characterize the component parser"
                    .to_owned(),
            );
        };
        self.boundary_trigger = crate::runtime::BoundaryTrigger::Exact(Arc::new(dwa));
        self.serialized_artifact_cache = None;
        Ok(())
    }

    /// Terminal source expressions retained for later compiled-constraint
    /// composition. Fresh constraints keep them directly on the tokenizer;
    /// current loaded artifacts may keep only the canonical serialized blob so
    /// ordinary static load/mask/commit does not rebuild expression trees.
    pub(crate) fn retained_terminal_exprs(&self) -> Option<&[Expr]> {
        if let Some(exprs) = self.tokenizer.terminal_exprs() {
            return Some(exprs);
        }
        if let Some(exprs) = self.deferred_terminal_exprs.get() {
            return Some(exprs.as_ref());
        }
        let decoded = self.deferred_terminal_exprs_blob.as_ref()?.decode_exprs().ok()?;
        if decoded.len() != self.tokenizer.num_terminals() as usize {
            return None;
        }
        let decoded = Arc::<[Expr]>::from(decoded.into_boxed_slice());
        let _ = self.deferred_terminal_exprs.set(decoded);
        self.deferred_terminal_exprs.get().map(Arc::as_ref)
    }

    #[inline]
    pub(crate) fn retained_terminal_expr(&self, terminal: TerminalID) -> Option<&Expr> {
        self.retained_terminal_exprs()?.get(terminal as usize)
    }

    /// Return the complete source grammar rules for composition. Ordinary
    /// runtime execution never consults these rules, so large current-format
    /// artifacts may retain their canonical bincode payload and decode it only
    /// when a later composition actually needs grammar structure.
    pub(crate) fn retained_table_rules(&self) -> Result<&[crate::grammar::flat::Rule], String> {
        if self.deferred_table_rules_blob.is_none() {
            return Ok(&self.table.rules);
        }
        if let Some(rules) = self.deferred_table_rules.get() {
            return Ok(rules.as_ref());
        }
        let blob = self
            .deferred_table_rules_blob
            .as_ref()
            .ok_or_else(|| "missing deferred GLR rules payload".to_owned())?;
        let decoded = bincode::deserialize::<Vec<crate::grammar::flat::Rule>>(blob.as_slice())
            .map_err(|err| err.to_string())?;
        if decoded.len() != self.table.num_rules as usize {
            return Err("deferred GLR rule count does not match table num_rules".to_owned());
        }
        if decoded.first() != self.table.rules.first() {
            return Err("deferred GLR augmented-start rule mismatch".to_owned());
        }
        let decoded = Arc::<[crate::grammar::flat::Rule]>::from(decoded.into_boxed_slice());
        let _ = self.deferred_table_rules.set(decoded);
        self.deferred_table_rules
            .get()
            .map(Arc::as_ref)
            .ok_or_else(|| "failed to install deferred GLR rules".to_owned())
    }

    /// Exact root nullability as seen by a later linker.
    ///
    /// A composed child's descendants can change whether its root language is
    /// nullable, so the intact root leaf table is not sufficient here. The
    /// retained composition grammar summary is the exact wrapper-language
    /// fact when available. Older/transitional artifacts fall back to the
    /// materialized compiler-oracle table; ordinary constraints use that table
    /// directly as before.
    pub(crate) fn composition_start_nullable(&self) -> Result<bool, String> {
        if self.uses_compact_segmented_parser_runtime() {
            if let Some(summary) = self.composition_grammar_summary.as_ref() {
                return Ok(summary.root_nullable);
            }
        }
        Ok(self.table.embedded_start_nullable())
    }

    /// Reconstruct the compiler-only flattened tokenizer of a recursive
    /// composition directly from its intact leaf tokenizers.
    ///
    /// Live recursive execution never needs the old outer union tokenizer, but
    /// late composition still has compiler analyses which consume one flat raw
    /// tokenizer-state coordinate. The recursive layout deliberately uses the
    /// same owned-parent state ordering: root leaf states begin at zero and all
    /// remaining leaves are appended contiguously. Rebuilding in that order
    /// therefore preserves `recursive_tokenizer_internal_tsids` exactly; no
    /// state relation is transported or approximated here.
    fn rebuild_recursive_compiler_tokenizer(&self) -> Result<Tokenizer, String> {
        let layout = self
            .recursive_parser_layout_ref()
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let root_leaf = layout
            .leaves
            .first()
            .ok_or_else(|| "recursive composition has no tokenizer leaves".to_owned())?;
        let root = self
            .constraint_at_recursive_component_path(&root_leaf.component_path)
            .ok_or_else(|| "recursive tokenizer root leaf does not resolve".to_owned())?;

        fn terminal_base_for_path(
            root: &Constraint,
            path: &[u32],
        ) -> Result<u32, String> {
            let mut constraint = root;
            let mut base = 0u32;
            for &component_index in path {
                let overlay = constraint.static_dynamic_overlay.as_ref().ok_or_else(|| {
                    format!(
                        "recursive tokenizer path {path:?} enters a non-composed constraint"
                    )
                })?;
                let component = overlay
                    .segmented_parser_components
                    .get(component_index as usize)
                    .ok_or_else(|| {
                        format!(
                            "recursive tokenizer path {path:?} references missing component {component_index}"
                        )
                    })?;
                base = base
                    .checked_add(component.terminal_offset)
                    .ok_or_else(|| "recursive tokenizer terminal offset overflow".to_owned())?;
                constraint = component.constraint.as_ref();
            }
            Ok(base)
        }

        let root_terminal_base = terminal_base_for_path(self, &root_leaf.component_path)?;
        if root_terminal_base != 0 {
            return Err(format!(
                "recursive tokenizer root leaf terminal base is {root_terminal_base}, expected zero"
            ));
        }

        let mut child_inputs = Vec::<(&Tokenizer, u32)>::with_capacity(layout.leaves.len().saturating_sub(1));
        for leaf in layout.leaves.iter().skip(1) {
            let constraint = self
                .constraint_at_recursive_component_path(&leaf.component_path)
                .ok_or_else(|| {
                    format!(
                        "recursive tokenizer leaf path {:?} does not resolve",
                        leaf.component_path,
                    )
                })?;
            child_inputs.push((
                &constraint.tokenizer,
                terminal_base_for_path(self, &leaf.component_path)?,
            ));
        }

        let (mut tokenizer, state_offsets) = Tokenizer::disjoint_union_with_owned_parent(
            root.tokenizer.as_ref().clone(),
            root_terminal_base,
            &child_inputs,
        );
        if state_offsets != layout.leaf_tokenizer_state_offsets {
            return Err(format!(
                "rebuilt recursive compiler tokenizer state offsets {state_offsets:?} disagree with live layout {:?}",
                layout.leaf_tokenizer_state_offsets,
            ));
        }

        // Preserve the exact canonical terminal chosen at every wrapper level.
        // `global_terminal_aliases` is directed metadata: `(canonical, local)`
        // means the component's direct `terminal_offset + local` terminal was
        // folded into `canonical`. Nested wrappers therefore form alias chains.
        // Collapse those chains once in the root terminal coordinate before
        // mutating the rebuilt tokenizer.
        fn collect_alias_edges(
            constraint: &Constraint,
            base: u32,
            edges: &mut Vec<(u32, u32)>,
        ) -> Result<(), String> {
            let Some(overlay) = constraint.static_dynamic_overlay.as_ref() else {
                return Ok(());
            };
            for component in &overlay.segmented_parser_components {
                let component_base = base
                    .checked_add(component.terminal_offset)
                    .ok_or_else(|| "recursive tokenizer terminal alias offset overflow".to_owned())?;
                for &(canonical, local) in &component.global_terminal_aliases {
                    let canonical = base.checked_add(canonical).ok_or_else(|| {
                        "recursive tokenizer canonical terminal overflow".to_owned()
                    })?;
                    let alias = component_base.checked_add(local).ok_or_else(|| {
                        "recursive tokenizer alias terminal overflow".to_owned()
                    })?;
                    if canonical != alias {
                        edges.push((canonical, alias));
                    }
                }
                collect_alias_edges(component.constraint.as_ref(), component_base, edges)?;
            }
            Ok(())
        }

        let mut edges = Vec::<(u32, u32)>::new();
        collect_alias_edges(self, 0, &mut edges)?;
        let mut parent = BTreeMap::<u32, u32>::new();
        for (canonical, alias) in edges {
            if canonical >= tokenizer.num_terminals() || alias >= tokenizer.num_terminals() {
                return Err(format!(
                    "recursive tokenizer terminal alias {alias}->{canonical} lies outside rebuilt domain {}",
                    tokenizer.num_terminals(),
                ));
            }
            if let Some(previous) = parent.insert(alias, canonical)
                && previous != canonical
            {
                return Err(format!(
                    "recursive tokenizer terminal alias {alias} has conflicting canonicals {previous} and {canonical}"
                ));
            }
        }
        let resolve = |start: u32, parent: &BTreeMap<u32, u32>| -> Result<u32, String> {
            let mut current = start;
            let mut steps = 0usize;
            while let Some(&next) = parent.get(&current) {
                current = next;
                steps += 1;
                if steps > parent.len() {
                    return Err("recursive tokenizer terminal alias cycle".to_owned());
                }
            }
            Ok(current)
        };
        let mut aliases_by_root = BTreeMap::<u32, Vec<u32>>::new();
        for &alias in parent.keys() {
            let root = resolve(alias, &parent)?;
            if alias != root {
                aliases_by_root.entry(root).or_default().push(alias);
            }
        }
        for (canonical, aliases) in aliases_by_root {
            tokenizer.canonicalize_terminal_aliases(canonical, &aliases);
        }
        Ok(tokenizer)
    }

    /// Ensure a compiler-owned recursive component has the flat tokenizer
    /// coordinate required by current late-composition analyses. Current v25
    /// artifacts still carry that tokenizer eagerly, so this is normally a
    /// no-op. It becomes the exact lazy reconstruction path once the outer
    /// compiler tokenizer is omitted from a future wire artifact.
    pub(crate) fn prepare_recursive_compiler_tokenizer_for_composition(
        &mut self,
    ) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        let layout = self
            .recursive_parser_layout_for_pending_root()?
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let expected_states = layout.total_tokenizer_states as usize;
        if self.tokenizer.num_states() as usize == expected_states
            && self.state_to_internal_tsid.len() == expected_states
        {
            return Ok(false);
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .ok_or_else(|| "recursive compiler tokenizer reconstruction requires overlay".to_owned())?;
        let provider_native = overlay
            .recursive_compiler_table
            .get()
            .is_some_and(|blob| blob.is_empty());
        let relation = overlay.recursive_tokenizer_internal_tsids.get().cloned();
        if !provider_native {
            let relation = relation.as_ref().ok_or_else(|| {
                "recursive compiler tokenizer reconstruction has no persisted state/TSID relation"
                    .to_owned()
            })?;
            if relation.len() != expected_states {
                return Err(format!(
                    "recursive compiler tokenizer TSID relation has {} rows for {expected_states} states",
                    relation.len(),
                ));
            }
        }
        let tokenizer = self.rebuild_recursive_compiler_tokenizer()?;
        if tokenizer.num_states() as usize != expected_states {
            return Err(format!(
                "rebuilt recursive compiler tokenizer has {} states, expected {expected_states}",
                tokenizer.num_states(),
            ));
        }

        let (state_to_internal_tsid, internal_tsid_to_states, state_internal_tsid_offsets, state_internal_tsids) =
            if provider_native {
                // The fast DynamicDirect coordinator intentionally stores no
                // outer quotient because live recursive execution never uses
                // it. A later static/compiler analysis can safely over-refine
                // to the exact raw-state identity coordinate on demand.
                (
                    (0..expected_states as u32).collect::<Vec<_>>(),
                    Vec::new(),
                    vec![u32::MAX],
                    Vec::new(),
                )
            } else {
                let relation = relation
                    .as_ref()
                    .expect("non-provider-native recursive compiler tokenizer requires relation");
                let tsid_count = self.internal_tsid_count();
                let mut state_to_internal_tsid = Vec::with_capacity(expected_states);
                let mut internal_tsid_to_states = vec![Vec::<u32>::new(); tsid_count];
                let mut state_internal_tsid_offsets = Vec::with_capacity(expected_states + 1);
                let mut state_internal_tsids = Vec::<u32>::new();
                state_internal_tsid_offsets.push(0);
                for (state, row) in relation.iter().enumerate() {
                    let Some(&primary) = row.first() else {
                        return Err(format!(
                            "recursive compiler tokenizer state {state} has no internal TSID"
                        ));
                    };
                    state_to_internal_tsid.push(primary);
                    for &tsid in row {
                        if tsid as usize >= tsid_count {
                            return Err(format!(
                                "recursive compiler tokenizer state {state} references TSID {tsid}/{tsid_count}"
                            ));
                        }
                        internal_tsid_to_states[tsid as usize].push(state as u32);
                        state_internal_tsids.push(tsid);
                    }
                    state_internal_tsid_offsets.push(state_internal_tsids.len() as u32);
                }
                (
                    state_to_internal_tsid,
                    internal_tsid_to_states,
                    state_internal_tsid_offsets,
                    state_internal_tsids,
                )
            };

        self.tokenizer = tokenizer.into();
        self.state_to_internal_tsid = state_to_internal_tsid;
        self.internal_tsid_to_states = internal_tsid_to_states;
        self.deferred_internal_tsid_to_states = OnceLock::new();
        self.state_internal_tsid_offsets = state_internal_tsid_offsets;
        self.state_internal_tsids = state_internal_tsids;
        // The rebuilt tokenizer is a direct union of exact leaf states, not a
        // runtime subset/product expansion of the old outer tokenizer.
        self.runtime_source_state_offset = None;
        self.runtime_product_source_offsets.clear();
        self.runtime_product_source_states.clear();
        self.runtime_product_exact_source_states.clear();
        self.runtime_product_state_by_source_subset.clear();
        self.tokenizer_has_epsilon_transitions = self.tokenizer.has_epsilon_transitions();
        self.terminal_live_states = self.compute_terminal_live_states();
        self.tokenizer_fast_transitions = Self::compute_tokenizer_fast_transitions_for(&self.tokenizer);
        Ok(true)
    }


    /// Reconstruct the exact compiler-only flattened table from the retained
    /// recursive component tree. Provider-native DynamicDirect coordinators
    /// intentionally omit their historical flattened compiler blob at build
    /// time; a later static/legacy composition is allowed to pay this cost on
    /// demand without changing the live recursive runtime representation.
    fn rebuild_recursive_compiler_table_from_components(&self) -> Result<GLRTable, String> {
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .ok_or_else(|| "recursive compiler-table reconstruction requires overlay".to_owned())?;
        if overlay.segmented_parser_components.is_empty() {
            return Err("recursive compiler-table reconstruction has no components".to_owned());
        }

        let mut prepared = Vec::<Constraint>::with_capacity(overlay.segmented_parser_components.len());
        for component in &overlay.segmented_parser_components {
            let mut constraint = component.constraint.as_ref().clone();
            constraint.prepare_recursive_compiler_table_for_composition()?;
            prepared.push(constraint);
        }

        let mut slots_by_child = vec![Vec::<u32>::new(); prepared.len()];
        for (link_index, link) in overlay.segmented_parser_links.iter().enumerate() {
            if link.parent_component != 0 {
                return Err(format!(
                    "recursive compiler-table reconstruction link {link_index} has non-root parent component {}",
                    link.parent_component,
                ));
            }
            let child = link.child_component as usize;
            if child == 0 || child >= prepared.len() {
                return Err(format!(
                    "recursive compiler-table reconstruction link {link_index} references invalid child {child}",
                ));
            }
            slots_by_child[child].push(link.slot_terminal);
        }
        for slots in &mut slots_by_child {
            slots.sort_unstable();
            slots.dedup();
        }

        let mut child_rules = Vec::with_capacity(prepared.len().saturating_sub(1));
        for child in prepared.iter().skip(1) {
            child_rules.push(child.retained_table_rules()?);
        }
        let parent_rules = prepared[0].retained_table_rules()?;
        let mut table_inputs = Vec::with_capacity(prepared.len().saturating_sub(1));
        for component_index in 1..prepared.len() {
            let slots = &slots_by_child[component_index];
            let Some((&placeholder_terminal, additional_placeholder_terminals)) = slots.split_first()
            else {
                return Err(format!(
                    "recursive compiler-table reconstruction component {component_index} has no linker slot",
                ));
            };
            let child = &prepared[component_index];
            table_inputs.push(crate::compiler::glr::table::SubgrammarTableInput {
                placeholder_terminal,
                additional_placeholder_terminals,
                table: &child.table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.composition_start_nullable()?,
            });
        }

        let composed = crate::compiler::glr::table::compose_subgrammar_tables_explicit_with_rules(
            &prepared[0].table,
            parent_rules,
            prepared[0].ignore_terminal,
            &table_inputs,
            &child_rules,
        )?;
        let mut table = composed.table;
        if table.num_terminals != self.table.num_terminals {
            return Err(format!(
                "rebuilt recursive compiler table has {} terminals, grammar shell has {}",
                table.num_terminals, self.table.num_terminals,
            ));
        }
        table.set_embedded_end_token_ids(&self.table.embedded_end_token_ids());
        Ok(table)
    }

    /// Materialize the exact flattened parser table used only by later
    /// composition. Recursive runtime execution never consults this table: its
    /// parser coordinate is the disjoint union of intact leaf tables plus
    /// CALL/RETURN provider controls. We nevertheless retain the exact compiled
    /// table in packed form because grammar rules alone do not encode all LR
    /// conflict/precedence decisions.
    pub(crate) fn prepare_recursive_compiler_table_for_composition(
        &mut self,
    ) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        if self.table.num_states != 0 && !self.table.action.is_empty() {
            return Ok(false);
        }
        let blob = self
            .static_dynamic_overlay
            .as_ref()
            .and_then(|overlay| overlay.recursive_compiler_table.get())
            .cloned()
            .ok_or_else(|| {
                "recursive composition has no packed compiler table".to_owned()
            })?;
        let provider_native = blob.is_empty();
        let table = if provider_native {
            self.rebuild_recursive_compiler_table_from_components()?
        } else {
            crate::compiler::glr::table::artifact_serde::from_compact_bytes(blob.as_ref())?
        };
        if table.num_terminals != self.table.num_terminals {
            return Err(format!(
                "recursive compiler table has {} terminals, grammar shell has {}",
                table.num_terminals, self.table.num_terminals,
            ));
        }
        if !provider_native && table.num_rules != self.table.num_rules {
            return Err(format!(
                "recursive compiler table has {} rules, grammar shell has {}",
                table.num_rules, self.table.num_rules,
            ));
        }
        self.table = table;
        self.deferred_table_rules_blob = None;
        self.deferred_table_rules = OnceLock::new();
        Ok(true)
    }

    /// Replace the live recursive coordinator's flattened LR machine with a
    /// grammar shell while retaining an exact packed compiler copy for future
    /// rebinding. The shell deliberately keeps terminal/rule/nonterminal
    /// metadata because those are semantic grammar facts; only executable
    /// materialized parser-state machinery is discarded.
    pub(crate) fn detach_recursive_outer_table(&mut self) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .ok_or_else(|| "recursive runtime is missing overlay metadata".to_owned())?;
        if overlay.recursive_compiler_table.get().is_none() {
            if self.table.num_states == 0 || self.table.action.is_empty() {
                return Err(
                    "recursive grammar shell has no packed compiler table".to_owned(),
                );
            }
            let packed = Arc::<[u8]>::from(
                crate::compiler::glr::table::artifact_serde::to_compact_bytes(&self.table),
            );
            overlay
                .recursive_compiler_table
                .set(packed)
                .map_err(|_| "recursive compiler table initialized twice".to_owned())?;
        }
        if self.table.num_states == 0 && self.table.action.is_empty() && self.table.goto.is_empty() {
            return Ok(false);
        }
        self.table.action.clear();
        self.table.goto.clear();
        self.table.advance.clear();
        self.table.unconditional_advance.clear();
        self.table.forwarded_shifts.clear();
        self.table.control_terminals.clear();
        self.table.skip_terminals.clear();
        self.table.guarded_shift_index.clear();
        self.table.direct_regular_wide_frontiers.clear();
        self.table.num_states = 0;
        Ok(true)
    }

    /// Drop the redundant flattened union tokenizer from a live recursive
    /// coordinator after compilation has finished. Recursive execution scans
    /// the intact leaf tokenizers directly; the outer `Constraint::tokenizer`
    /// field is therefore only a structural placeholder for this runtime kind.
    /// Keep the root leaf tokenizer there so generic/debug code still sees a
    /// valid tokenizer object. If this constraint is composed again later,
    /// `prepare_recursive_compiler_tokenizer_for_composition` reconstructs the
    /// exact temporary flat compiler view from the leaf tree and the persisted
    /// recursive state/TSID relation.
    pub(crate) fn detach_recursive_outer_tokenizer(&mut self) -> Result<bool, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(false);
        }
        let layout = self
            .recursive_parser_layout_for_pending_root()?
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let root_leaf = layout
            .leaves
            .first()
            .ok_or_else(|| "recursive composition has no tokenizer leaves".to_owned())?;
        let root = self
            .constraint_at_recursive_component_path(&root_leaf.component_path)
            .ok_or_else(|| "recursive tokenizer root leaf does not resolve".to_owned())?;
        if self.tokenizer.num_states() == root.tokenizer.num_states()
            && self.tokenizer.num_terminals() == root.tokenizer.num_terminals()
        {
            return Ok(false);
        }
        let tokenizer = root.tokenizer.clone();
        let tokenizer_fast_transitions = root.tokenizer_fast_transitions.clone();
        let tokenizer_has_epsilon_transitions = root.tokenizer_has_epsilon_transitions;
        self.tokenizer = tokenizer.into();
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        self.tokenizer_has_epsilon_transitions = tokenizer_has_epsilon_transitions;
        self.terminal_live_states.clear();
        Ok(true)
    }

    /// Exact zero-width pop depth for returning from this constraint when it is
    /// used as an opaque subgrammar by a later composition.
    ///
    /// Descendant bindings do not change the augmented-root goto of the
    /// outermost intact grammar frame. Recursive runtimes therefore derive the
    /// invariant from the first/root leaf instead of the materialized composed
    /// table; ordinary constraints keep the historical direct derivation.
    pub(crate) fn composition_child_return_pop(&self) -> Result<u32, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return subgrammar_child_return_pop(&self.table, self.retained_table_rules()?);
        }
        let layout = self
            .recursive_parser_layout_ref()
            .ok_or_else(|| "recursive composition layout is unavailable".to_owned())?;
        let root_leaf = layout
            .leaves
            .first()
            .ok_or_else(|| "recursive composition has no root leaf".to_owned())?;
        let root = self
            .constraint_at_recursive_component_path(&root_leaf.component_path)
            .ok_or_else(|| "recursive composition root leaf does not resolve".to_owned())?;
        subgrammar_child_return_pop(&root.table, root.retained_table_rules()?)
    }

    /// Finite tokenizer coordinate used by Static composition/link analysis.
    ///
    /// Static constraints with an exact virtual residual lexer commit in the
    /// exact symbolic coordinate, but their parser DWA / TSID tables were
    /// compiled against the finite vocabulary-horizon observation tokenizer.
    /// Composition is likewise a whole-model-token analysis, so it must use
    /// that same finite coordinate. The retained component constraint remains
    /// exact and is used unchanged by the segmented runtime.
    pub(crate) fn composition_tokenizer(&self) -> &Tokenizer {
        if !self.uses_dynamic_runtime() && self.tokenizer.has_virtual_residual_runtime() {
            self.dynamic_mask_vocab
                .mask_projection_tokenizer()
                .expect("Static virtual-residual constraint requires its finite observation tokenizer")
        } else {
            &self.tokenizer
        }
    }

    pub(crate) fn compute_composition_reset_tokens_by_terminal(&self) -> Vec<Vec<u32>> {
        let terminal_count = self.tokenizer.num_terminals() as usize;
        let empty = || (0..terminal_count).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
        if terminal_count == 0 || self.token_bytes.is_empty() {
            return empty();
        }
        let start = self.tokenizer.start_state();
        let mut rows = self
            .token_bytes
            .par_iter()
            .fold(empty, |mut rows, (&token_id, bytes)| {
                if bytes.is_empty() {
                    return rows;
                }
                let (_, matches) = self.tokenizer.execute_summary_from_state(bytes, start);
                for (terminal, width) in matches {
                    if width == bytes.len() {
                        if let Some(row) = rows.get_mut(terminal as usize) {
                            row.push(token_id);
                        }
                    }
                }
                rows
            })
            .reduce(empty, |mut left, mut right| {
                for (left_row, right_row) in left.iter_mut().zip(&mut right) {
                    left_row.append(right_row);
                }
                left
            });
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
        }
        rows
    }

    pub(crate) fn ensure_composition_reset_tokens_by_terminal(&mut self) {
        if self.composition_reset_tokens_by_terminal.len()
            != self.tokenizer.num_terminals() as usize
        {
            self.composition_reset_tokens_by_terminal =
                self.compute_composition_reset_tokens_by_terminal();
        }
    }

    fn compute_scoped_ignore_runtime_tokens(
        &self,
    ) -> (
        Vec<(TerminalID, Box<[u32]>)>,
        Vec<(TerminalID, Box<[(u32, u32)]>)>,
    ) {
        if self.static_dynamic_overlay.is_none() || self.table.skip_terminals.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let special_tokens = self
            .special_token_terminals
            .iter()
            .map(|special| special.token_id)
            .collect::<std::collections::BTreeSet<_>>();
        let mut tokens_by_bytes = FxHashMap::<&[u8], SmallVec<[u32; 2]>>::default();
        for (&token, bytes) in self.token_bytes.iter() {
            // A suffix is replayed as ordinary bytes inside a larger model
            // token. Do not use an exact-special-token-only identity as the
            // witness for that byte suffix.
            if !special_tokens.contains(&token) {
                tokens_by_bytes.entry(bytes.as_slice()).or_default().push(token);
            }
        }

        let rows = self
            .table
            .skip_terminals
            .par_iter()
            .filter_map(|&terminal| {
                let expr = self.tokenizer.terminal_expr(terminal)?;
                let dfa = crate::automata::lexer::compile::compile_terminal_expr_dfa(expr);
                if dfa.has_epsilon_transitions() || dfa.finalizers(0).contains(0) {
                    return None;
                }

                let mut tokens = Vec::<u32>::new();
                let mut fusions = Vec::<(u32, u32)>::new();
                for (&token, bytes) in self.token_bytes.iter() {
                    if bytes.is_empty() {
                        continue;
                    }
                    let mut states = SmallVec::<[u32; 8]>::new();
                    states.push(0);
                    let mut valid_end = false;
                    for (index, &byte) in bytes.iter().enumerate() {
                        let mut next = SmallVec::<[u32; 8]>::new();
                        for &state in &states {
                            if let Some(target) = dfa.step(state, byte)
                                && !next.contains(&target)
                            {
                                next.push(target);
                            }
                        }
                        if next.is_empty() {
                            valid_end = false;
                            states.clear();
                            break;
                        }
                        let completed_here = next
                            .iter()
                            .any(|&state| dfa.finalizers(state).contains(0));
                        valid_end = completed_here
                            || next.iter().any(|&state| {
                                dfa.possible_future_group_ids(state).contains(0)
                            });

                        let prefix_end = index + 1;
                        if completed_here && prefix_end < bytes.len() {
                            let suffix_bytes = &bytes[prefix_end..];
                            if let Some(suffix_tokens) = tokens_by_bytes.get(suffix_bytes) {
                                // This is deliberately a permissive candidate
                                // relation, not an acceptance proof. Runtime
                                // correlates the suffix with the exact parser
                                // branch and then sends the fused token through
                                // the exact dynamic recognizer before admitting
                                // it. Keeping unfinished suffixes here is what
                                // recovers model tokens such as " (" or " &&\n".
                                fusions.extend(
                                    suffix_tokens
                                        .iter()
                                        .copied()
                                        .map(|suffix| (token, suffix)),
                                );
                            }
                            if !next.contains(&0) {
                                // The completed Skip may reset before the next
                                // byte, while the current accepting DFA state
                                // remains live for a possible longer match.
                                next.push(0);
                            }
                        }
                        states = next;
                    }
                    if valid_end && !states.is_empty() {
                        tokens.push(token);
                    }
                }
                tokens.sort_unstable();
                tokens.dedup();
                fusions.sort_unstable_by_key(|&(fused, suffix)| {
                    (
                        self.token_bytes.get(&fused).map_or(usize::MAX, Vec::len),
                        fused,
                        suffix,
                    )
                });
                fusions.dedup();
                Some((
                    terminal,
                    tokens.into_boxed_slice(),
                    fusions.into_boxed_slice(),
                ))
            })
            .collect::<Vec<_>>();

        let mut tokens = Vec::with_capacity(rows.len());
        let mut fusions = Vec::with_capacity(rows.len());
        for (terminal, token_row, fusion_row) in rows {
            if !token_row.is_empty() {
                tokens.push((terminal, token_row));
            }
            if !fusion_row.is_empty() {
                fusions.push((terminal, fusion_row));
            }
        }
        (tokens, fusions)
    }

    /// Materialize non-DWA weights retained in the compact current-artifact
    /// pool. Compiler transformations still have a few mutable/map-oriented
    /// consumers, so one-time composition preparation reconstructs those maps
    /// rather than mixing packed and materialized sources of truth.
    pub(crate) fn materialize_non_dwa_weights_for_compilation(&mut self) -> Result<(), String> {
        let Some(packed) = self.packed_non_dwa_weights.take() else {
            return Ok(());
        };
        let weights = crate::ds::weight::unpack_pooled_weights(packed.pool.packed_bytes())?;
        let weight = |id: u32| -> Result<Weight, String> {
            weights
                .get(id as usize)
                .cloned()
                .ok_or_else(|| format!("packed non-DWA Weight id {id} is out of range"))
        };

        self.parser_top_accept = packed
            .parser_top_accept
            .iter()
            .map(|(&label, &id)| Ok((label, weight(id)?)))
            .collect::<Result<_, String>>()?;
        self.parser_top_accept_parts = packed
            .parser_top_accept_parts
            .iter()
            .map(|(&label, ids)| {
                let parts = ids
                    .iter()
                    .map(|&id| weight(id))
                    .collect::<Result<Vec<_>, String>>()?;
                Ok((label, parts))
            })
            .collect::<Result<_, String>>()?;
        self.direct_regular_l1_complete_by_terminal = packed
            .direct_regular_l1_complete_by_terminal
            .iter()
            .map(|(&terminal, &id)| Ok((terminal, weight(id)?)))
            .collect::<Result<_, String>>()?;
        self.possible_matches = packed
            .possible_matches
            .iter()
            .map(|(&terminal, &id)| Ok((terminal, weight(id)?)))
            .collect::<Result<_, String>>()?;
        self.serialized_artifact_cache = None;
        Ok(())
    }

    /// Compiler-side escape hatch for transformations that genuinely require
    /// a mutable ordinary parser DWA. Ordinary load/mask/commit and the
    /// segmented late-binding composition path deliberately keep
    /// `packed_parser_dwa` zero-copy.
    pub(crate) fn materialize_parser_dwa_for_compilation(&mut self) -> Result<(), String> {
        if let Some(packed) = self.packed_parser_dwa.take() {
            self.parser_dwa = packed.to_dwa()?;
            self.serialized_artifact_cache = None;
            self.parser_runtime_caches_prebuilt = false;
            self.packed_dwa_token_dense_masks.clear();
            self.dwa_fast_transitions = Default::default();
            self.indexed_dag_dense_transitions.clear();
            self.indexed_dag_dense_finals.clear();
        }
        if let Some(override_weight) = self.parser_start_final_override.take() {
            let start = self.parser_dwa.start_state() as usize;
            self.parser_dwa.states_mut()[start].final_weight =
                (!override_weight.is_empty()).then_some(override_weight);
            self.serialized_artifact_cache = None;
            self.parser_runtime_caches_prebuilt = false;
            self.dwa_fast_transitions = Default::default();
            self.indexed_dag_dense_transitions.clear();
            self.indexed_dag_dense_finals.clear();
        }
        Ok(())
    }

    pub(crate) fn rebuild_scoped_ignore_runtime_tokens(&mut self) {
        // This cache is consumed only by the opt-in exact scoped-ignore mask
        // overlay. Building it eagerly can require a vocabulary-wide fusion
        // scan (tens of milliseconds on selected10) even when that runtime
        // path is disabled. Keep ordinary/static constraints at zero cost.
        if self.static_dynamic_overlay.is_none()
            || std::env::var_os("GLRMASK_EXPERIMENT_SCOPED_IGNORE_EXACT_OVERLAY").is_none()
        {
            self.scoped_ignore_only_tokens.clear();
            self.scoped_ignore_prefix_fusions.clear();
            return;
        }
        let (tokens, fusions) = self.compute_scoped_ignore_runtime_tokens();
        self.scoped_ignore_only_tokens = tokens;
        self.scoped_ignore_prefix_fusions = fusions;
    }
}
