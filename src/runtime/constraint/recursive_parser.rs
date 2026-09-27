//! Recursive component layouts and exact scoped parser execution.

use crate::automata::lexer::Lexer;
use crate::compiler::glr::parser::DisjointComponentActionProvider;
use crate::compiler::glr::parser::ParserComponentTableSource;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::parser::ScopedParserSymbol;
use crate::compiler::glr::parser::advance_provider_control_closed_stacks;
use crate::compiler::glr::parser::close_provider_control_stacks;
use crate::compiler::glr::parser::find_admitted_symbol_with_provider;
use crate::compiler::glr::parser::for_each_admitted_symbol_with_provider;
use crate::compiler::glr::parser::materialize_control_eliminated_scoped_provider_table;
use crate::compiler::glr::parser::stack_may_advance_on_any_with_provider;
use crate::compiler::glr::parser::stack_may_advance_on_with_provider;
use crate::compiler::glr::parser::stacks_finished_with_provider;
use crate::compiler::glr::table::GLRTable;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::runtime::artifact::Constraint;
use crate::runtime::artifact::RecursiveParserLayout;
use crate::runtime::artifact::RecursiveParserLeafLayout;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::OnceLock;

struct RecursiveSegmentedParserTables<'a> {
    root: &'a Constraint,
    layout: &'a RecursiveParserLayout,
}


impl RecursiveSegmentedParserTables<'_> {
    #[inline]
    fn leaf_constraint(&self, component: u32) -> Option<&Constraint> {
        let leaf = self.layout.leaves.get(component as usize)?;
        self.root
            .constraint_at_recursive_component_path(&leaf.component_path)
    }
}


impl ParserComponentTableSource for RecursiveSegmentedParserTables<'_> {
    #[inline]
    fn component_count(&self) -> usize {
        self.layout.leaves.len()
    }

    #[inline]
    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.leaf_constraint(component).map(|constraint| &constraint.table)
    }

    #[inline]
    fn component_ignore_terminal(&self, component: u32) -> Option<TerminalID> {
        let constraint = self.leaf_constraint(component)?;
        // Same idempotence contract: composed leaf tables own their scoped
        // ignores in-row; only raw leaves need provider-level Identity.
        let ignore = constraint.ignore_terminal?;
        (!constraint.table.skip_terminals.contains(&ignore)).then_some(ignore)
    }
}

impl Constraint {


    #[inline]
    pub(crate) fn has_recursive_segmented_parser_tree(&self) -> bool {
        self.static_dynamic_overlay.as_ref().is_some_and(|overlay| {
            !overlay.segmented_parser_components.is_empty()
                && !overlay.segmented_parser_links.is_empty()
        })
    }

    /// Width of the endpoint parser-state coordinate when this constraint is
    /// embedded as one component. Nested compositions contribute the disjoint
    /// union of their intact descendants, not the size of their transitional
    /// materialized composed table.
    pub(crate) fn recursive_parser_state_span(&self) -> Result<u32, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(self.table.num_states);
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        let mut total = 0u32;
        for component in &overlay.segmented_parser_components {
            total = total
                .checked_add(component.constraint.recursive_parser_state_span()?)
                .ok_or_else(|| "recursive parser-state coordinate overflow".to_owned())?;
        }
        Ok(total)
    }

    pub(super) fn constraint_at_recursive_component_path(&self, path: &[u32]) -> Option<&Constraint> {
        let mut current = self;
        for &component_index in path {
            let overlay = current.static_dynamic_overlay.as_ref()?;
            current = overlay
                .segmented_parser_components
                .get(component_index as usize)?
                .constraint
                .as_ref();
        }
        Some(current)
    }

    fn append_recursive_terminal_targets(
        &self,
        terminal: TerminalID,
        expand_this: bool,
        component_path: &mut Vec<u32>,
        leaves: &[RecursiveParserLeafLayout],
        out: &mut SmallVec<[(u32, TerminalID); 4]>,
    ) -> Result<(), String> {
        if !expand_this {
            if terminal >= self.table.num_terminals {
                return Ok(());
            }
            let leaf_index = leaves
                .iter()
                .position(|leaf| leaf.component_path == *component_path)
                .ok_or_else(|| "recursive parser terminal target has no leaf layout".to_owned())?;
            let target = (leaf_index as u32, terminal);
            if !out.contains(&target) {
                out.push(target);
            }
            return Ok(());
        }

        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
            let offset = component.terminal_offset;
            let end = offset.saturating_add(component.constraint.table.num_terminals);
            if terminal >= offset && terminal < end {
                component_path.push(component_index as u32);
                component.constraint.append_recursive_terminal_targets(
                    terminal - offset,
                    component.constraint.uses_compact_segmented_parser_runtime(),
                    component_path,
                    leaves,
                    out,
                )?;
                component_path.pop();
            }
            for &(alias, local_terminal) in &component.global_terminal_aliases {
                if alias == terminal {
                    component_path.push(component_index as u32);
                    component.constraint.append_recursive_terminal_targets(
                        local_terminal,
                        component.constraint.uses_compact_segmented_parser_runtime(),
                        component_path,
                        leaves,
                        out,
                    )?;
                    component_path.pop();
                }
            }
        }
        Ok(())
    }

    fn append_recursive_parser_layout(
        &self,
        base: u32,
        top_component: u32,
        expand_this: bool,
        component_path: &mut Vec<u32>,
        leaves: &mut Vec<RecursiveParserLeafLayout>,
        links: &mut Vec<crate::runtime::artifact::SegmentedParserLink>,
    ) -> Result<(u32, u32), String> {
        if !expand_this {
            let leaf_index = u32::try_from(leaves.len())
                .map_err(|_| "recursive parser leaf index overflow".to_owned())?;
            leaves.push(RecursiveParserLeafLayout {
                state_offset: base,
                state_count: self.table.num_states,
                top_component,
                component_path: component_path.clone(),
            });
            let next = base
                .checked_add(self.table.num_states)
                .ok_or_else(|| "recursive parser-state coordinate overflow".to_owned())?;
            return Ok((next, leaf_index));
        }
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        let mut next = base;
        let mut component_root_leaves = Vec::with_capacity(overlay.segmented_parser_components.len());
        for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
            component_path.push(component_index as u32);
            let (component_next, component_root_leaf) = component.constraint.append_recursive_parser_layout(
                next,
                top_component,
                component.constraint.uses_compact_segmented_parser_runtime(),
                component_path,
                leaves,
                links,
            )?;
            component_path.pop();
            next = component_next;
            component_root_leaves.push(component_root_leaf);
        }

        self.append_recursive_node_links(
            component_path,
            leaves,
            &component_root_leaves,
            links,
        )?;

        let root_leaf = *component_root_leaves
            .first()
            .ok_or_else(|| "recursive parser composition has no root component".to_owned())?;
        Ok((next, root_leaf))
    }

    fn append_recursive_node_links(
        &self,
        component_path: &mut Vec<u32>,
        leaves: &[RecursiveParserLeafLayout],
        component_root_leaves: &[u32],
        links: &mut Vec<crate::runtime::artifact::SegmentedParserLink>,
    ) -> Result<(), String> {
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        for (link_index, link) in overlay.segmented_parser_links.iter().enumerate() {
            let parent_component = overlay
                .segmented_parser_components
                .get(link.parent_component as usize)
                .ok_or_else(|| format!("recursive parser link {link_index} has missing parent component"))?;
            let child_root_leaf = *component_root_leaves
                .get(link.child_component as usize)
                .ok_or_else(|| format!("recursive parser link {link_index} has missing child component"))?;
            if link.child_start != 0 {
                return Err(format!(
                    "recursive parser link {link_index} has unsupported non-root child start {}",
                    link.child_start,
                ));
            }

            component_path.push(link.parent_component);
            let mut parent_targets = SmallVec::<[(u32, TerminalID); 4]>::new();
            parent_component.constraint.append_recursive_terminal_targets(
                link.slot_terminal,
                parent_component
                    .constraint
                    .uses_compact_segmented_parser_runtime(),
                component_path,
                leaves,
                &mut parent_targets,
            )?;
            component_path.pop();
            let [(parent_leaf, local_slot)] = parent_targets.as_slice() else {
                return Err(format!(
                    "recursive parser link {link_index} slot terminal {} resolves to {} leaf targets, expected exactly one",
                    link.slot_terminal,
                    parent_targets.len(),
                ));
            };
            links.push(crate::runtime::artifact::SegmentedParserLink {
                parent_component: *parent_leaf,
                slot_terminal: *local_slot,
                child_component: child_root_leaf,
                child_start: 0,
                return_pop: link.return_pop,
                child_start_nullable: link.child_start_nullable,
            });
        }
        Ok(())
    }

    fn build_recursive_parser_layout_root_expanded(
        &self,
    ) -> Result<Arc<RecursiveParserLayout>, String> {
        let overlay = self
            .static_dynamic_overlay
            .as_ref()
            .expect("recursive segmented parser tree requires overlay");
        if let Some(layout) = overlay.recursive_parser_layout.get() {
            return Ok(Arc::clone(layout));
        }
        let mut component_offsets = Vec::with_capacity(overlay.segmented_parser_components.len());
        let mut leaves = Vec::new();
        let mut links = Vec::new();
        let mut component_path = Vec::new();
        let mut component_root_leaves = Vec::with_capacity(overlay.segmented_parser_components.len());
        let mut next = 0u32;
        for (component_index, component) in overlay.segmented_parser_components.iter().enumerate() {
            component_offsets.push(next);
            component_path.push(component_index as u32);
            let (component_next, component_root_leaf) = component.constraint.append_recursive_parser_layout(
                next,
                component_index as u32,
                component.constraint.uses_compact_segmented_parser_runtime(),
                &mut component_path,
                &mut leaves,
                &mut links,
            )?;
            component_path.pop();
            next = component_next;
            component_root_leaves.push(component_root_leaf);
        }
        self.append_recursive_node_links(
            &mut component_path,
            &leaves,
            &component_root_leaves,
            &mut links,
        )?;
        let leaf_state_offsets = leaves.iter().map(|leaf| leaf.state_offset).collect();
        let mut leaf_tokenizer_state_offsets = Vec::with_capacity(leaves.len());
        let mut leaf_terminal_offsets = Vec::with_capacity(leaves.len());
        let mut next_tokenizer_state = 0u32;
        let mut next_leaf_terminal = 0u32;
        for leaf in &leaves {
            let constraint = self
                .constraint_at_recursive_component_path(&leaf.component_path)
                .ok_or_else(|| {
                    format!(
                        "recursive tokenizer leaf path {:?} does not resolve to a constraint",
                        leaf.component_path,
                    )
                })?;
            leaf_tokenizer_state_offsets.push(next_tokenizer_state);
            next_tokenizer_state = next_tokenizer_state
                .checked_add(constraint.tokenizer.num_states())
                .ok_or_else(|| "recursive tokenizer-state coordinate overflow".to_owned())?;
            leaf_terminal_offsets.push(next_leaf_terminal);
            next_leaf_terminal = next_leaf_terminal
                .checked_add(constraint.table.num_terminals)
                .ok_or_else(|| "recursive terminal coordinate overflow".to_owned())?;
        }
        let mut terminal_targets = Vec::with_capacity(self.table.num_terminals as usize);
        for terminal in 0..self.table.num_terminals {
            let mut targets = SmallVec::<[(u32, TerminalID); 4]>::new();
            self.append_recursive_terminal_targets(
                terminal,
                true,
                &mut Vec::new(),
                &leaves,
                &mut targets,
            )?;
            terminal_targets.push(targets);
        }
        let layout = Arc::new(RecursiveParserLayout {
            component_offsets,
            leaves,
            leaf_state_offsets,
            leaf_tokenizer_state_offsets,
            total_tokenizer_states: next_tokenizer_state,
            leaf_terminal_offsets,
            total_leaf_terminals: next_leaf_terminal,
            outer_terminal_count: self.table.num_terminals,
            tokenizer_future_scoped: (0..next_tokenizer_state)
                .map(|_| OnceLock::new())
                .collect(),
            links,
            terminal_targets,
            total_states: next,
        });
        let _ = overlay.recursive_parser_layout.set(Arc::clone(&layout));
        Ok(overlay
            .recursive_parser_layout
            .get()
            .cloned()
            .unwrap_or(layout))
    }

    /// Derived recursive layout for the live endpoint parser coordinate.
    /// Immediate wrappers occupy contiguous intervals. Descendant leaves
    /// inside the same wrapper retain the same `top_component`.
    pub(crate) fn recursive_parser_layout(
        &self,
    ) -> Result<Option<Arc<RecursiveParserLayout>>, String> {
        if !self.uses_compact_segmented_parser_runtime() {
            return Ok(None);
        }
        self.build_recursive_parser_layout_root_expanded().map(Some)
    }

    pub(super) fn recursive_parser_layout_ref(&self) -> Option<&RecursiveParserLayout> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let initialized = self
            .static_dynamic_overlay
            .as_ref()?
            .recursive_parser_layout
            .get()
            .is_some();
        if !initialized {
            self.build_recursive_parser_layout_root_expanded().ok()?;
        }
        self.static_dynamic_overlay
            .as_ref()?
            .recursive_parser_layout
            .get()
            .map(Arc::as_ref)
    }

    /// Compiler/load bridge used while a static boundary is still stored in
    /// the materialized composed-table coordinate. The root is expanded
    /// unconditionally; descendants are expanded only when their own runtime
    /// has already migrated to the recursive coordinate.
    pub(crate) fn recursive_parser_layout_for_pending_root(
        &self,
    ) -> Result<Option<Arc<RecursiveParserLayout>>, String> {
        if !self.has_recursive_segmented_parser_tree() {
            return Ok(None);
        }
        self.build_recursive_parser_layout_root_expanded().map(Some)
    }

    #[inline]
    pub(crate) fn recursive_tokenizer_leaf_state(
        &self,
        scoped_state: u32,
    ) -> Option<(usize, u32)> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        if scoped_state < layout.total_tokenizer_states {
            let leaf_index = layout
                .leaf_tokenizer_state_offsets
                .partition_point(|&offset| offset <= scoped_state)
                .checked_sub(1)?;
            let offset = *layout.leaf_tokenizer_state_offsets.get(leaf_index)?;
            let local_state = scoped_state.checked_sub(offset)?;
            let leaf = layout.leaves.get(leaf_index)?;
            let constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
            return (local_state < constraint.tokenizer.num_states())
                .then_some((leaf_index, local_state));
        }
        self.static_dynamic_overlay
            .as_ref()?
            .recursive_virtual_tokenizer_states
            .local_state(scoped_state)
    }

    #[inline]
    pub(crate) fn recursive_tokenizer_scoped_state(
        &self,
        leaf_index: usize,
        local_state: u32,
    ) -> Option<u32> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        if local_state < constraint.tokenizer.num_states() {
            return layout
                .leaf_tokenizer_state_offsets
                .get(leaf_index)?
                .checked_add(local_state);
        }
        if !constraint.tokenizer.contains_runtime_state(local_state) {
            return None;
        }
        self.static_dynamic_overlay
            .as_ref()?
            .recursive_virtual_tokenizer_states
            .scoped_state(layout.total_tokenizer_states, leaf_index, local_state)
    }

    /// Project one tokenizer state from this composition's recursive leaf
    /// coordinate into an immediate component's own live tokenizer coordinate.
    /// Intact components receive a raw local state; recursively composed
    /// components receive the corresponding state in their own leaf union.
    pub(crate) fn recursive_tokenizer_state_for_component(
        &self,
        component_index: usize,
        scoped_state: u32,
    ) -> Option<u32> {
        let (leaf_index, local_state) = self.recursive_tokenizer_leaf_state(scoped_state)?;
        let layout = self.recursive_parser_layout_ref()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let (&owner, descendant_path) = leaf.component_path.split_first()?;
        if owner as usize != component_index {
            return None;
        }

        let overlay = self.static_dynamic_overlay.as_ref()?;
        let component = overlay.segmented_parser_components.get(component_index)?;
        let component_constraint = component.constraint.as_ref();
        if !component_constraint.uses_compact_segmented_parser_runtime() {
            if !descendant_path.is_empty()
                || !component_constraint.tokenizer.contains_runtime_state(local_state)
            {
                return None;
            }
            return Some(local_state);
        }

        let component_layout = component_constraint.recursive_parser_layout_ref()?;
        let component_leaf_index = component_layout
            .leaves
            .iter()
            .position(|candidate| candidate.component_path.as_slice() == descendant_path)?;
        component_constraint.recursive_tokenizer_scoped_state(component_leaf_index, local_state)
    }

    /// Exact internal-TSID image of one live tokenizer state. Recursive
    /// compositions use their persisted/derived leaf-state relation; intact
    /// constraints use the ordinary tokenizer-state relation. Dynamic intact
    /// constraints without a TSID quotient use raw tokenizer state as TSID.
    pub(crate) fn runtime_internal_tsids_for_tokenizer_state(
        &self,
        tokenizer_state: u32,
    ) -> Option<SmallVec<[u32; 4]>> {
        if self.uses_compact_segmented_parser_runtime() {
            let layout = self.recursive_parser_layout_ref()?;
            if tokenizer_state < layout.total_tokenizer_states {
                let overlay = self.static_dynamic_overlay.as_ref()?;
                let provider_native_compiler_view = overlay
                    .recursive_compiler_table
                    .get()
                    .is_some_and(|blob| blob.is_empty())
                    && self.state_to_internal_tsid.len() == layout.total_tokenizer_states as usize;
                if provider_native_compiler_view {
                    return Some(
                        self.internal_tsids_for_state(tokenizer_state)
                            .iter()
                            .copied()
                            .collect(),
                    );
                }
                return overlay
                    .recursive_tokenizer_internal_tsids
                    .get()?
                    .get(tokenizer_state as usize)
                    .map(|row| row.iter().copied().collect());
            }
            let (leaf_index, _) = self.recursive_tokenizer_leaf_state(tokenizer_state)?;
            let leaf = layout.leaves.get(leaf_index)?;
            let owner = *leaf.component_path.first()? as usize;
            let overlay = self.static_dynamic_overlay.as_ref()?;
            let component = overlay.segmented_parser_components.get(owner)?;
            let component_state =
                self.recursive_tokenizer_state_for_component(owner, tokenizer_state)?;
            let local_tsids = component
                .constraint
                .runtime_internal_tsids_for_tokenizer_state(component_state)?;
            let mut global = SmallVec::<[u32; 4]>::new();
            for local_tsid in local_tsids {
                global.extend(
                    component
                        .local_tsid_to_global_tsids
                        .get(local_tsid as usize)?
                        .iter()
                        .copied(),
                );
            }
            global.sort_unstable();
            global.dedup();
            return (!global.is_empty()).then_some(global);
        }
        if self.state_to_internal_tsid.is_empty() && self.internal_tsid_to_states.is_empty() {
            return (tokenizer_state < self.tokenizer.num_states())
                .then(|| smallvec::smallvec![tokenizer_state]);
        }
        if self.tokenizer.has_virtual_residual_runtime() {
            return self
                .tokenizer
                .contains_runtime_state(tokenizer_state)
                .then(|| self.internal_tsids_for_state(tokenizer_state).iter().copied().collect());
        }
        (tokenizer_state < self.tokenizer.num_states())
            .then(|| self.internal_tsids_for_state(tokenizer_state).iter().copied().collect())
    }

    pub(crate) fn install_recursive_tokenizer_internal_tsids(
        &mut self,
        mut relation: Vec<Vec<u32>>,
    ) -> Result<(), String> {
        let layout = self
            .recursive_parser_layout_for_pending_root()?
            .ok_or_else(|| "recursive tokenizer TSID relation requires recursive runtime".to_owned())?;
        if relation.len() != layout.total_tokenizer_states as usize {
            return Err(format!(
                "recursive tokenizer TSID relation has {} rows for {} scoped tokenizer states",
                relation.len(),
                layout.total_tokenizer_states,
            ));
        }
        let tsid_count = self.internal_tsid_count();
        for (state, row) in relation.iter_mut().enumerate() {
            row.sort_unstable();
            row.dedup();
            if row.is_empty() {
                return Err(format!(
                    "recursive tokenizer state {state} has no internal TSID image"
                ));
            }
            if let Some(&bad) = row.iter().find(|&&tsid| tsid as usize >= tsid_count) {
                return Err(format!(
                    "recursive tokenizer state {state} references out-of-range internal TSID {bad}/{tsid_count}"
                ));
            }
        }
        let overlay = self
            .static_dynamic_overlay
            .as_mut()
            .ok_or_else(|| "recursive tokenizer TSID relation requires overlay".to_owned())?;
        if let Some(existing) = overlay.recursive_tokenizer_internal_tsids.get() {
            if existing.as_ref() != &relation {
                return Err("recursive tokenizer TSID relation disagrees with existing view".to_owned());
            }
            return Ok(());
        }
        overlay
            .recursive_tokenizer_internal_tsids
            .set(Arc::new(relation))
            .map_err(|_| "recursive tokenizer TSID relation initialized twice".to_owned())
    }

    #[inline]
    pub(crate) fn recursive_tokenizer_reset_state(&self, leaf_index: usize) -> Option<u32> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        self.recursive_tokenizer_scoped_state(
            leaf_index,
            constraint.runtime_commit_initial_state(),
        )
    }

    pub(crate) fn recursive_tokenizer_is_reset_state(&self, scoped_state: u32) -> bool {
        let Some(layout) = self.recursive_parser_layout_ref() else {
            return false;
        };
        if scoped_state >= layout.total_tokenizer_states {
            return false;
        }
        let Some(leaf_index) = layout
            .leaf_tokenizer_state_offsets
            .partition_point(|&offset| offset <= scoped_state)
            .checked_sub(1)
        else {
            return false;
        };
        let offset = layout.leaf_tokenizer_state_offsets[leaf_index];
        let local_state = scoped_state - offset;
        let Some(leaf) = layout.leaves.get(leaf_index) else {
            return false;
        };
        let Some(constraint) = self.constraint_at_recursive_component_path(&leaf.component_path)
        else {
            return false;
        };
        local_state == constraint.runtime_commit_initial_state()
    }

    #[inline]
    pub(crate) fn recursive_runtime_terminal_count(&self) -> Option<usize> {
        let layout = self.recursive_parser_layout_ref()?;
        usize::try_from(
            layout
                .outer_terminal_count
                .checked_add(layout.total_leaf_terminals)?,
        )
        .ok()
    }

    #[inline]
    pub(crate) fn recursive_terminal_scoped_id(
        &self,
        leaf_index: usize,
        local_terminal: TerminalID,
    ) -> Option<u32> {
        let layout = self.recursive_parser_layout_ref()?;
        let leaf = layout.leaves.get(leaf_index)?;
        let leaf_constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        if local_terminal >= leaf_constraint.table.num_terminals {
            return None;
        }
        layout
            .outer_terminal_count
            .checked_add(*layout.leaf_terminal_offsets.get(leaf_index)?)?
            .checked_add(local_terminal)
    }

    #[inline]
    pub(crate) fn recursive_terminal_leaf_local(
        &self,
        runtime_terminal: u32,
    ) -> Option<(usize, TerminalID)> {
        let layout = self.recursive_parser_layout_ref()?;
        let scoped = runtime_terminal.checked_sub(layout.outer_terminal_count)?;
        if scoped >= layout.total_leaf_terminals {
            return None;
        }
        let leaf_index = layout
            .leaf_terminal_offsets
            .partition_point(|&offset| offset <= scoped)
            .checked_sub(1)?;
        let local_terminal = scoped.checked_sub(layout.leaf_terminal_offsets[leaf_index])?;
        let leaf = layout.leaves.get(leaf_index)?;
        let leaf_constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        (local_terminal < leaf_constraint.table.num_terminals)
            .then_some((leaf_index, local_terminal))
    }

    pub(crate) fn recursive_terminal_for_component(
        &self,
        component_index: usize,
        runtime_terminal: u32,
    ) -> Option<u32> {
        let layout = self.recursive_parser_layout_ref()?;
        let (leaf_index, local_terminal) = self.recursive_terminal_leaf_local(runtime_terminal)?;
        let leaf = layout.leaves.get(leaf_index)?;
        let (&owner, descendant_path) = leaf.component_path.split_first()?;
        if owner as usize != component_index {
            return None;
        }
        let component = self
            .static_dynamic_overlay
            .as_ref()?
            .segmented_parser_components
            .get(component_index)?;
        let component_constraint = component.constraint.as_ref();
        if !component_constraint.uses_compact_segmented_parser_runtime() {
            if !descendant_path.is_empty()
                || local_terminal >= component_constraint.table.num_terminals
            {
                return None;
            }
            return Some(local_terminal);
        }
        let component_layout = component_constraint.recursive_parser_layout_ref()?;
        let component_leaf_index = component_layout
            .leaves
            .iter()
            .position(|candidate| candidate.component_path.as_slice() == descendant_path)?;
        component_constraint.recursive_terminal_scoped_id(component_leaf_index, local_terminal)
    }

    pub(crate) fn recursive_terminal_is_ignore(&self, runtime_terminal: u32) -> bool {
        let Some((leaf_index, local_terminal)) =
            self.recursive_terminal_leaf_local(runtime_terminal)
        else {
            return false;
        };
        self.recursive_leaf_constraint(leaf_index)
            .is_some_and(|leaf| leaf.ignore_terminal == Some(local_terminal))
    }

    pub(crate) fn recursive_tokenizer_future_scoped_terminals(
        &self,
        scoped_state: u32,
    ) -> Option<Cow<'_, BitSet>> {
        let layout = self.recursive_parser_layout_ref()?;
        let (leaf_index, local_state) = self.recursive_tokenizer_leaf_state(scoped_state)?;
        let leaf = layout.leaves.get(leaf_index)?;
        let leaf_constraint = self.constraint_at_recursive_component_path(&leaf.component_path)?;
        let build_scoped = || {
            let local_future = leaf_constraint
                .tokenizer
                .possible_future_terminals(local_state);
            let mut scoped = BitSet::new(
                self.recursive_runtime_terminal_count()
                    .expect("recursive terminal layout must fit usize"),
            );
            for local_terminal in local_future.iter_ones() {
                if let Some(runtime_terminal) =
                    self.recursive_terminal_scoped_id(leaf_index, local_terminal as u32)
                {
                    scoped.set(runtime_terminal as usize);
                }
            }
            scoped
        };
        if scoped_state < layout.total_tokenizer_states {
            let cache = layout.tokenizer_future_scoped.get(scoped_state as usize)?;
            return Some(Cow::Borrowed(cache.get_or_init(build_scoped)));
        }
        Some(Cow::Owned(build_scoped()))
    }

    pub(crate) fn recursive_leaf_constraint(&self, leaf_index: usize) -> Option<&Constraint> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        let leaf = layout.leaves.get(leaf_index)?;
        self.constraint_at_recursive_component_path(&leaf.component_path)
    }

    #[inline]
    pub(crate) fn recursive_parser_leaf_state(
        &self,
        scoped_state: u32,
    ) -> Option<(usize, u32)> {
        let layout = self.recursive_parser_layout().ok().flatten()?;
        if scoped_state >= layout.total_states {
            return None;
        }
        let leaf_index = layout
            .leaf_state_offsets
            .partition_point(|&offset| offset <= scoped_state)
            .checked_sub(1)?;
        let leaf = layout.leaves.get(leaf_index)?;
        let local_state = scoped_state.checked_sub(leaf.state_offset)?;
        (local_state < leaf.state_count).then_some((leaf_index, local_state))
    }

    /// Split a recursive parser language by the leaf owning each live stack
    /// top. `LeveledGSS::isolate` prunes only the selected top branch, so the
    /// original lower-stack and accumulator correlations remain intact. This is
    /// the exact bridge needed when a zero-width CALL/RETURN closure changes the
    /// active lexer scope after one terminal commits.
    pub(crate) fn partition_recursive_parser_gss_by_active_leaf(
        &self,
        gss: &ParserGSS,
    ) -> Option<SmallVec<[(usize, ParserGSS); 4]>> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        if gss.is_empty() {
            return Some(SmallVec::new());
        }
        let tops = gss.peek_values();
        if tops.is_empty() {
            // A non-empty parser language with an empty stack has no active
            // component to own a tokenizer continuation. Parser runtime stacks
            // are rooted, so treat this as an invalid recursive frontier rather
            // than guessing a scope.
            return None;
        }
        let mut partitions = SmallVec::<[(usize, ParserGSS); 4]>::new();
        for top in tops {
            let (leaf_index, _) = self.recursive_parser_leaf_state(top)?;
            let branch = gss.isolate(Some(top));
            if branch.is_empty() {
                continue;
            }
            if let Some((_, existing)) = partitions
                .iter_mut()
                .find(|(candidate, _)| *candidate == leaf_index)
            {
                *existing = existing.merge(&branch);
            } else {
                partitions.push((leaf_index, branch));
            }
        }
        partitions.sort_unstable_by_key(|(leaf_index, _)| *leaf_index);
        Some(partitions)
    }

    /// Exact ordinary-terminal GLR table for compiler-side analyses that still
    /// consume a table rather than a `ParserActionProvider`. Its parser-state
    /// alphabet is the live recursive leaf coordinate; CALL/RETURN are first
    /// encoded as private controls and then eliminated exactly.
    pub(crate) fn recursive_control_eliminated_parser_table(
        &self,
    ) -> Result<Option<Arc<GLRTable>>, String> {
        let Some(layout) = self.recursive_parser_layout_for_pending_root()? else {
            return Ok(None);
        };
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        let terminal_symbols = layout
            .terminal_targets
            .iter()
            .map(|targets| {
                targets
                    .iter()
                    .map(|&(component, terminal)| ScopedParserSymbol::Terminal {
                        component,
                        terminal,
                    })
                    .collect::<SmallVec<[ScopedParserSymbol; 4]>>()
            })
            .collect::<Vec<_>>();
        let table = materialize_control_eliminated_scoped_provider_table(
            &provider,
            &terminal_symbols,
        )?;
        Ok(Some(Arc::new(table)))
    }

    /// Drop the materialized-table start-state ownership sets once recursive
    /// wrapper ownership is authoritative. v23 requires those bitsets on the
    /// wire; v24 recursive runtimes do not need them after load/build.
    pub(crate) fn clear_recursive_legacy_boundary_start_states(&mut self) {
        if !self.uses_compact_segmented_parser_runtime() {
            return;
        }
        let Some(overlay) = self.static_dynamic_overlay.as_mut() else {
            return;
        };
        for component in &mut overlay.segmented_parser_components {
            if let Some(shard) = component.boundary.as_mut() {
                shard.start_parser_states = BitSet::new(0);
            }
        }
        overlay.segmented_boundary_shards = overlay
            .segmented_parser_components
            .iter()
            .filter_map(|component| component.boundary.clone())
            .collect();
    }

    /// Drop the materialized composed-table -> immediate-component parser
    /// projections after every runtime consumer has moved to recursive wrapper
    /// ownership. v25 stores recursive B directly; v24 legacy static B remains
    /// on the materialized runtime and therefore retains these projections.
    pub(crate) fn clear_recursive_legacy_parser_state_projections(&mut self) {
        if !self.uses_compact_segmented_parser_runtime() {
            return;
        }
        let Some(overlay) = self.static_dynamic_overlay.as_mut() else {
            return;
        };
        for component in &mut overlay.segmented_parser_components {
            component.global_to_local_parser_state.clear();
        }
    }

    fn recursive_parser_symbols_for_global_terminal(
        &self,
        layout: &RecursiveParserLayout,
        global_terminal: TerminalID,
        out: &mut SmallVec<[ScopedParserSymbol; 8]>,
    ) -> bool {
        out.clear();
        let Some(targets) = layout.terminal_targets.get(global_terminal as usize) else {
            return false;
        };
        for &(component, terminal) in targets {
            let symbol = ScopedParserSymbol::Terminal {
                component,
                terminal,
            };
            if !out.contains(&symbol) {
                out.push(symbol);
            }
        }
        !out.is_empty()
    }

    fn recursive_parser_symbols_for_runtime_terminal(
        &self,
        layout: &RecursiveParserLayout,
        runtime_terminal: TerminalID,
        out: &mut SmallVec<[ScopedParserSymbol; 8]>,
    ) -> bool {
        out.clear();
        if let Some((leaf_index, local_terminal)) =
            self.recursive_terminal_leaf_local(runtime_terminal)
        {
            out.push(ScopedParserSymbol::Terminal {
                component: leaf_index as u32,
                terminal: local_terminal,
            });
            return true;
        }
        self.recursive_parser_symbols_for_global_terminal(layout, runtime_terminal, out)
    }

    /// Correctness/reference entry point for the recursive endpoint parser
    /// coordinate. This deliberately does not switch the live masking/commit
    /// path yet: component A projection and dynamic-trigger routing must move
    /// to the same recursive ownership model atomically.
    pub(crate) fn close_recursive_segmented_parser_reference(
        &self,
        stack: &ParserGSS,
    ) -> Result<Option<ParserGSS>, String> {
        let Some(layout) = self.recursive_parser_layout()? else {
            return Ok(None);
        };
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        for (component, leaf) in layout.leaves.iter().enumerate() {
            if provider.scoped_state(component as u32, 0) != Some(leaf.state_offset) {
                return Err(format!(
                    "recursive parser leaf {component} offset disagrees with provider coordinate",
                ));
            }
        }
        Ok(Some(close_provider_control_stacks(&provider, stack)))
    }

    /// Advance one composed/global terminal through the recursive leaf view.
    /// Global terminal IDs remain only an outer migration coordinate here;
    /// each provider action itself sees a leaf-local terminal.
    pub(crate) fn advance_recursive_segmented_parser_reference(
        &self,
        stack: &ParserGSS,
        global_terminal: TerminalID,
    ) -> Result<Option<ParserGSS>, String> {
        let Some(layout) = self.recursive_parser_layout()? else {
            return Ok(None);
        };
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
        if !self.recursive_parser_symbols_for_global_terminal(&layout, global_terminal, &mut symbols)
        {
            return Ok(Some(ParserGSS::empty()));
        }
        let mut advanced = ParserGSS::empty();
        for symbol in symbols {
            let branch = advance_provider_control_closed_stacks(&provider, stack, symbol);
            if advanced.is_empty() {
                advanced = branch;
            } else if !branch.is_empty() {
                advanced = advanced.merge(&branch);
            }
        }
        Ok(Some(advanced))
    }

    pub(crate) fn recursive_segmented_parser_is_finished_reference(
        &self,
        stack: &ParserGSS,
    ) -> Result<Option<bool>, String> {
        let Some(layout) = self.recursive_parser_layout()? else {
            return Ok(None);
        };
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )?;
        Ok(Some(stacks_finished_with_provider(
            &provider,
            stack,
            ScopedParserSymbol::Terminal {
                component: 0,
                terminal: crate::compiler::glr::analysis::EOF,
            },
        )))
    }

    /// Whether this live, source-built composition can use the compact parser
    /// coordinate directly in the ordinary `ParserGSS<u32>`. Serialized legacy
    /// overlays intentionally omit links/offsets and therefore stay on the old
    /// composed-table coordinate until the wire format is versioned.
    #[inline]
    pub(crate) fn uses_compact_segmented_parser_runtime(&self) -> bool {
        self.static_dynamic_overlay.as_ref().is_some_and(|overlay| {
            let boundary_is_provider_native = overlay.segmented_parser_components.iter().all(|component| {
                match component.boundary.as_ref().map(|shard| &shard.backend) {
                    Some(crate::runtime::artifact::SegmentedBoundaryShardBackend::DynamicDirect) => true,
                    Some(crate::runtime::artifact::SegmentedBoundaryShardBackend::StaticParser(boundary)) => {
                        boundary.recursive_parser_dwa.is_some()
                    }
                    None => true,
                    _ => false,
                }
            });
            overlay.segmented_mask_authoritative
                && !overlay.segmented_parser_components.is_empty()
                && !overlay.segmented_parser_links.is_empty()
                && boundary_is_provider_native
        })
    }

    pub(crate) fn close_compact_segmented_parser(
        &self,
        stack: &ParserGSS,
    ) -> Option<ParserGSS> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        Some(close_provider_control_stacks(&provider, stack))
    }

    pub(crate) fn advance_compact_segmented_parser(
        &self,
        stack: &ParserGSS,
        global_terminal: u32,
    ) -> Option<ParserGSS> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
        if !self.recursive_parser_symbols_for_runtime_terminal(
            &layout,
            global_terminal,
            &mut symbols,
        ) {
            return Some(ParserGSS::empty());
        }
        let mut advanced = ParserGSS::empty();
        for symbol in symbols {
            let branch = advance_provider_control_closed_stacks(&provider, stack, symbol);
            if advanced.is_empty() {
                advanced = branch;
            } else if !branch.is_empty() {
                advanced = advanced.merge(&branch);
            }
        }
        Some(advanced)
    }

    pub(crate) fn compact_segmented_parser_may_advance_on(
        &self,
        stack: &ParserGSS,
        global_terminal: u32,
    ) -> Option<bool> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
        if !self.recursive_parser_symbols_for_runtime_terminal(
            &layout,
            global_terminal,
            &mut symbols,
        ) {
            return Some(false);
        }
        Some(
            symbols
                .into_iter()
                .any(|symbol| stack_may_advance_on_with_provider(&provider, stack, symbol)),
        )
    }

    pub(crate) fn compact_segmented_parser_may_advance_on_any(
        &self,
        stack: &ParserGSS,
        terminals: &BitSet,
    ) -> Option<bool> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self.recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables { root: self, layout: &layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        ).expect("validated compact segmented parser metadata");
        let symbols = terminals.iter_ones().flat_map(|terminal| {
            let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
            self.recursive_parser_symbols_for_runtime_terminal(
                &layout, terminal as u32, &mut symbols,
            );
            symbols
        });
        Some(stack_may_advance_on_any_with_provider(&provider, stack, symbols))
    }

    pub(crate) fn compact_segmented_parser_admitted_terminals(
        &self,
        stack: &ParserGSS,
        candidates: &BitSet,
    ) -> Option<BitSet> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self.recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables { root: self, layout: &layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        ).expect("validated compact segmented parser metadata");
        let symbols = candidates.iter_ones().flat_map(|terminal| {
            let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
            self.recursive_parser_symbols_for_runtime_terminal(
                &layout, terminal as u32, &mut symbols,
            );
            symbols.into_iter().map(move |symbol| (terminal, symbol))
        });
        let mut admitted = BitSet::new(candidates.len());
        for_each_admitted_symbol_with_provider(&provider, stack, symbols,
            |terminal| admitted.set(terminal));
        Some(admitted)
    }

    /// Exact existential intersection of parser admission and a caller's
    /// terminal predicate. Candidate supports may over-approximate lexical
    /// liveness: only a terminal passing BOTH checks can establish success.
    pub(crate) fn compact_segmented_parser_may_advance_on_any_matching(
        &self,
        stack: &ParserGSS,
        candidates: impl IntoIterator<Item = u32>,
        mut matches: impl FnMut(u32) -> bool,
    ) -> Option<bool> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self.recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables { root: self, layout: &layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        ).expect("validated compact segmented parser metadata");
        let symbols = candidates.into_iter().flat_map(|terminal| {
            let mut symbols = SmallVec::<[ScopedParserSymbol; 8]>::new();
            self.recursive_parser_symbols_for_runtime_terminal(
                &layout, terminal, &mut symbols,
            );
            symbols.into_iter().map(move |symbol| (terminal, symbol))
        });
        Some(find_admitted_symbol_with_provider(
            &provider, stack, symbols, |&terminal| matches(terminal),
        ).is_some())
    }

    pub(crate) fn compact_segmented_parser_is_finished(
        &self,
        stack: &ParserGSS,
    ) -> Option<bool> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let tables = RecursiveSegmentedParserTables {
            root: self,
            layout: &layout,
        };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &layout.links,
            &layout.leaf_state_offsets,
        )
        .expect("validated compact segmented parser metadata");
        Some(stacks_finished_with_provider(
            &provider,
            stack,
            ScopedParserSymbol::Terminal {
                component: 0,
                terminal: crate::compiler::glr::analysis::EOF,
            },
        ))
    }

    #[inline]
    pub(crate) fn compact_segmented_parser_local_state(
        &self,
        component_index: usize,
        scoped_state: u32,
    ) -> Option<u32> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        let offset = *layout.component_offsets.get(component_index)?;
        let end = layout
            .component_offsets
            .get(component_index + 1)
            .copied()
            .unwrap_or(layout.total_states);
        if scoped_state < offset || scoped_state >= end {
            return None;
        }
        let local = scoped_state.checked_sub(offset)?;
        Some(local)
    }

    #[inline]
    pub(crate) fn compact_segmented_parser_component(
        &self,
        scoped_state: u32,
    ) -> Option<(usize, u32)> {
        if !self.uses_compact_segmented_parser_runtime() {
            return None;
        }
        let layout = self
            .recursive_parser_layout()
            .expect("validated recursive compact parser metadata")?;
        if scoped_state >= layout.total_states {
            return None;
        }
        let leaf_index = layout
            .leaf_state_offsets
            .partition_point(|&offset| offset <= scoped_state)
            .checked_sub(1)?;
        let component_index = layout.leaves.get(leaf_index)?.top_component as usize;
        let local = scoped_state.checked_sub(*layout.component_offsets.get(component_index)?)?;
        Some((component_index, local))
    }
}
