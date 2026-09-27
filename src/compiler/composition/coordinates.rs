//! Translate component-local parser, tokenizer, and token coordinates.

use crate::automata::lexer::Lexer;
use super::{
    Action, Arc, BTreeMap, BTreeSet, BitSet, Constraint, DEFAULT_LABEL, DWAState, EOF, Instant,
    InternalIdMap, ManyToOneIdMap, NWA, PublishedStaticBoundaryShard, SmallVec, Weight,
    compose_profile_enabled, encode_negative_label, encode_positive_label, is_negative_label,
    negative_to_positive_label, reverse_hashcons_owned,
};

#[derive(Clone, Copy)]
pub(crate) struct ParserDwaComponent<'a> {
    pub(crate) constraint: &'a Constraint,
    /// Local parser state -> merged parser states.
    pub(crate) parser_state_relation: &'a [Vec<u32>],
    /// Local raw tokenizer state `s` is merged state `tokenizer_state_offset+s`.
    pub(crate) tokenizer_state_offset: u32,
    /// Global terminal offset of this component in the composed table.
    pub(crate) terminal_offset: u32,
    /// Composed execution table used to scope a standalone global-ignore
    /// identity to parser tops where that terminal is actually `Skip`.
    pub(crate) composed_table: Option<&'a crate::compiler::glr::table::GLRTable>,
}

pub(super) const NO_PARSER_DOMAIN_LABEL: i32 = i32::MAX;

#[derive(Debug, Clone)]
pub(super) struct ParserDefaultDomain {
    /// Fallback label for exact component-owned states that had no pre-existing
    /// symbolic parser-domain label in the cached component.
    pub(super) label: i32,
    pub(super) base_has_states: bool,
    /// Cached/nested symbolic fallback label -> refined outer fallback label.
    ///
    /// A nested constraint already uses lookup order
    /// `concrete -> nested-domain -> DEFAULT`.  When it is linked again we do
    /// not need to expand that nested domain back to thousands of concrete LR
    /// labels.  Instead, split this component's outer DEFAULT domain by the
    /// cached nested-domain partition.  Every composed parser state still has
    /// exactly one runtime fallback label; row construction resolves
    /// nested-domain-over-DEFAULT precedence onto that refined label.
    pub(super) nested_labels: BTreeMap<i32, i32>,
    pub(super) states: BitSet,
    pub(super) predicted_saved_edges: usize,
}

impl ParserDefaultDomain {
    pub(super) fn output_labels(&self) -> impl Iterator<Item = i32> + '_ {
        self.base_has_states
            .then_some(self.label)
            .into_iter()
            .chain(self.nested_labels.values().copied())
    }
}

#[derive(Debug, Clone)]
pub(super) struct ParserDefaultDomainPlan {
    /// Composed parser state -> synthetic fallback label. The sentinel means
    /// that state has no component-local wildcard domain.
    pub(super) parser_state_labels: Vec<i32>,
    /// One optional exact wildcard domain per component. Component zero is the
    /// parent and is deliberately never assigned a domain.
    pub(super) component_domains: Vec<Option<ParserDefaultDomain>>,
    pub(super) predicted_saved_edges: usize,
}

pub(super) fn symbolic_child_defaults_env_override() -> Option<bool> {
    std::env::var("GLRMASK_COMPOSE_SYMBOLIC_CHILD_DEFAULTS")
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
}

pub(super) fn symbolic_child_default_min_saved_edges() -> usize {
    std::env::var("GLRMASK_COMPOSE_SYMBOLIC_CHILD_DEFAULT_MIN_SAVED_EDGES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4_096)
}

/// Build an exact quotient of the positive parser-state input alphabet for
/// component-local DEFAULT transitions.
///
/// A composed LR state is eligible for child `i` iff it has exactly one
/// preimage `(i, local_state)` across *all* component state relations. This
/// excludes caller/return aliases, same-child many-to-one quotients, and
/// cross-child structural sharing. Eligible domains are therefore pairwise
/// disjoint, and every state in a domain has one unambiguous local-state
/// interpretation. A child DEFAULT can then be represented once by a synthetic
/// label: runtime lookup tries the concrete LR-state label first and the domain
/// label only on a miss. States outside the domain retain ordinary explicit
/// DEFAULT materialization, making this representation exactly equivalent to
/// full materialization on every parser stack, not merely reachable stacks.
pub(super) fn build_parser_default_domain_plan(
    components: &[ParserDwaComponent<'_>],
    num_parser_states: u32,
) -> ParserDefaultDomainPlan {
    build_parser_default_domain_plan_with_policy(
        components,
        num_parser_states,
        symbolic_child_defaults_env_override(),
        std::env::var_os("GLRMASK_EXPERIMENT_SYMBOLIC_PARENT_DEFAULTS").is_some(),
    )
}

pub(super) fn build_parser_default_domain_plan_with_policy(
    components: &[ParserDwaComponent<'_>],
    num_parser_states: u32,
    force: Option<bool>,
    force_parent_defaults: bool,
) -> ParserDefaultDomainPlan {
    let n = num_parser_states as usize;
    let mut preimage_count = vec![0u32; n];
    let mut owner_component = vec![u32::MAX; n];
    let mut owner_local = vec![u32::MAX; n];

    for (component_index, component) in components.iter().enumerate() {
        for (local_state, targets) in component.parser_state_relation.iter().enumerate() {
            for &target in targets {
                let Some(count) = preimage_count.get_mut(target as usize) else {
                    continue;
                };
                *count = count.saturating_add(1);
                if *count == 1 {
                    owner_component[target as usize] = component_index as u32;
                    owner_local[target as usize] = local_state as u32;
                } else {
                    owner_component[target as usize] = u32::MAX;
                    owner_local[target as usize] = u32::MAX;
                }
            }
        }
    }

    let mut local_multiplicities = components
        .iter()
        .map(|component| vec![0usize; component.parser_state_relation.len()])
        .collect::<Vec<_>>();
    for state in 0..n {
        if preimage_count[state] != 1 {
            continue;
        }
        let component = owner_component[state] as usize;
        let local = owner_local[state] as usize;
        if component >= components.len() {
            continue;
        }
        if let Some(slot) = local_multiplicities
            .get_mut(component)
            .and_then(|rows| rows.get_mut(local))
        {
            *slot += 1;
        }
    }

    let min_saved = symbolic_child_default_min_saved_edges();
    let mut component_predicted = vec![0usize; components.len()];
    for component_index in 0..components.len() {
        let component = components[component_index];
        let domain_total = local_multiplicities[component_index].iter().sum::<usize>();
        if domain_total == 0 {
            continue;
        }
        for state in component.constraint.parser_dwa.states() {
            if !state.transitions.contains_key(&DEFAULT_LABEL) {
                continue;
            }
            let explicit_domain = state
                .transitions
                .keys()
                .filter_map(|&label| {
                    (label >= 0 && label != DEFAULT_LABEL).then_some(label as usize)
                })
                .filter_map(|local| local_multiplicities[component_index].get(local))
                .copied()
                .sum::<usize>();
            component_predicted[component_index] = component_predicted[component_index]
                .saturating_add(domain_total.saturating_sub(explicit_domain));
        }
    }

    // A symbolic child domain already pays the one runtime state->domain map
    // lookup. Once that map is selected, representing the parent's exact
    // unambiguous domain as one additional label is purely a graph-size win:
    // it replaces concrete DEFAULT expansion without adding lookup depth.
    // Keep parent-only/small compositions on the historical zero-overhead
    // path unless explicitly forced.
    let child_feature_selected = match force {
        Some(false) => false,
        Some(true) => true,
        None => component_predicted
            .iter()
            .skip(1)
            .any(|&predicted| predicted >= min_saved),
    };
    let symbolic_parent_defaults = force_parent_defaults || child_feature_selected;
    let first_domain_component = usize::from(!symbolic_parent_defaults);
    let required_domain_labels = (first_domain_component..components.len())
        .map(|component_index| {
            let component = components[component_index];
            let mut nested = BTreeSet::<i32>::new();
            let mut has_base = false;
            for (local, &multiplicity) in local_multiplicities[component_index].iter().enumerate() {
                if multiplicity == 0 {
                    continue;
                }
                let source_domain = component
                    .constraint
                    .parser_state_domain_labels
                    .get(local)
                    .copied()
                    .unwrap_or(NO_PARSER_DOMAIN_LABEL);
                if source_domain == NO_PARSER_DOMAIN_LABEL {
                    has_base = true;
                } else {
                    nested.insert(source_domain);
                }
            }
            nested.len() + usize::from(has_base)
        })
        .sum::<usize>();
    let labels_fit = (num_parser_states as i64)
        .saturating_add(required_domain_labels as i64)
        < DEFAULT_LABEL as i64;
    let mut next_label = num_parser_states as i32;
    let mut component_domains = vec![None; components.len()];
    let mut parser_state_labels = vec![NO_PARSER_DOMAIN_LABEL; n];
    let mut predicted_saved_edges = 0usize;

    let feature_selected = labels_fit && (child_feature_selected || force_parent_defaults);
    if feature_selected {
        // Once one child amortizes the table-sized runtime map, every further
        // exact child domain is a marginal win: one synthetic label replaces a
        // positive number of concrete edges without increasing lookup depth.
        for component_index in first_domain_component..components.len() {
            let predicted = component_predicted[component_index];
            if predicted == 0 {
                continue;
            }
            let mut states = BitSet::new(n);
            let mut nested_source_labels = BTreeSet::<i32>::new();
            let mut base_has_states = false;
            for (local, &multiplicity) in local_multiplicities[component_index].iter().enumerate() {
                if multiplicity == 0 {
                    continue;
                }
                let source_domain = components[component_index]
                    .constraint
                    .parser_state_domain_labels
                    .get(local)
                    .copied()
                    .unwrap_or(NO_PARSER_DOMAIN_LABEL);
                if source_domain == NO_PARSER_DOMAIN_LABEL {
                    base_has_states = true;
                } else {
                    nested_source_labels.insert(source_domain);
                }
            }
            // Keep one stable base label even when every uniquely-owned state
            // belongs to a nested domain; it simplifies the transport plan and
            // is never emitted/installed when `base_has_states` is false.
            let label = next_label;
            if base_has_states {
                next_label += 1;
            }
            let mut nested_labels = BTreeMap::<i32, i32>::new();
            for source_label in nested_source_labels {
                let output_label = next_label;
                next_label += 1;
                nested_labels.insert(source_label, output_label);
            }
            for state in 0..n {
                if preimage_count[state] == 1
                    && owner_component[state] == component_index as u32
                {
                    states.set(state);
                    let local = owner_local[state] as usize;
                    let source_domain = components[component_index]
                        .constraint
                        .parser_state_domain_labels
                        .get(local)
                        .copied()
                        .unwrap_or(NO_PARSER_DOMAIN_LABEL);
                    parser_state_labels[state] = if source_domain == NO_PARSER_DOMAIN_LABEL {
                        label
                    } else {
                        *nested_labels
                            .get(&source_domain)
                            .expect("nested source domain was inventoried above")
                    };
                }
            }
            if states.is_empty() {
                continue;
            }
            predicted_saved_edges = predicted_saved_edges.saturating_add(predicted);
            component_domains[component_index] = Some(ParserDefaultDomain {
                label,
                base_has_states,
                nested_labels,
                states,
                predicted_saved_edges: predicted,
            });
        }
    }

    if predicted_saved_edges == 0 {
        // Keep ordinary/small compositions on the zero-overhead runtime path:
        // an empty map makes the exact-label -> DEFAULT lookup identical to
        // pre-v12 constraints.
        parser_state_labels.clear();
    }
    ParserDefaultDomainPlan {
        parser_state_labels,
        component_domains,
        predicted_saved_edges,
    }
}

pub(crate) struct CompiledSubgrammarInput<'a> {
    pub(crate) placeholder_terminal: u32,
    /// Additional parent terminals that call the same compiled child. The
    /// child parser/tokenizer artifacts are incorporated once and shared by
    /// every listed call site.
    pub(crate) additional_placeholder_terminals: &'a [u32],
    pub(crate) constraint: &'a Constraint,
}

impl<'a> CompiledSubgrammarInput<'a> {
    #[inline]
    pub(super) fn placeholder_terminals(&self) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(self.placeholder_terminal)
            .chain(self.additional_placeholder_terminals.iter().copied())
    }
}

pub(crate) fn build_segmented_parser_links(
    children: &[CompiledSubgrammarInput<'_>],
) -> Result<Vec<crate::runtime::SegmentedParserLink>, String> {
    let link_count = children
        .iter()
        .map(|child| 1 + child.additional_placeholder_terminals.len())
        .sum();
    let mut links = Vec::with_capacity(link_count);
    for (child_index, child) in children.iter().enumerate() {
        let child_component = u32::try_from(child_index + 1)
            .map_err(|_| "segmented parser component index overflow".to_owned())?;
        let return_pop = child.constraint.composition_child_return_pop()?;
        let child_start_nullable = child.constraint.composition_start_nullable()?;
        for slot_terminal in child.placeholder_terminals() {
            links.push(crate::runtime::SegmentedParserLink {
                parent_component: 0,
                slot_terminal,
                child_component,
                child_start: 0,
                return_pop,
                child_start_nullable,
            });
        }
    }
    Ok(links)
}

pub(crate) struct ConstraintComposition {
    pub(crate) constraint: Constraint,
    pub(crate) terminal_offsets: Vec<u32>,
    pub(crate) tokenizer_state_offsets: Vec<u32>,
    pub(crate) parser_state_relations: Vec<Vec<Vec<u32>>>,
}

pub(super) struct DirectComponentCoordinateMaps {
    pub(super) local_to_global_tsids: Vec<Vec<u32>>,
    pub(super) local_to_global_tokens: Vec<Vec<u32>>,
}

pub(super) struct DirectComponentStateCoordinates {
    pub(super) tokenizer_states: ManyToOneIdMap,
    pub(super) local_to_global_tsids: Vec<Vec<Vec<u32>>>,
}

pub(super) fn build_recursive_tokenizer_internal_tsid_relation(
    constraint: &Constraint,
    component_maps: &[DirectComponentCoordinateMaps],
) -> Result<Vec<Vec<u32>>, String> {
    let layout = constraint
        .recursive_parser_layout_for_pending_root()?
        .ok_or_else(|| "recursive tokenizer TSID relation requires recursive layout".to_owned())?;
    let overlay = constraint
        .static_dynamic_overlay
        .as_ref()
        .ok_or_else(|| "recursive tokenizer TSID relation requires segmented overlay".to_owned())?;
    if component_maps.len() != overlay.segmented_parser_components.len() {
        return Err(format!(
            "recursive tokenizer TSID component-map count mismatch: maps={} components={}",
            component_maps.len(),
            overlay.segmented_parser_components.len(),
        ));
    }

    let mut relation = Vec::with_capacity(layout.total_tokenizer_states as usize);
    for scoped_state in 0..layout.total_tokenizer_states {
        let (leaf_index, _) = constraint
            .recursive_tokenizer_leaf_state(scoped_state)
            .ok_or_else(|| format!("recursive tokenizer state {scoped_state} has no leaf"))?;
        let leaf = layout
            .leaves
            .get(leaf_index)
            .ok_or_else(|| format!("recursive tokenizer leaf {leaf_index} is missing"))?;
        let owner = *leaf
            .component_path
            .first()
            .ok_or_else(|| format!("recursive tokenizer leaf {leaf_index} has empty path"))?
            as usize;
        let component = overlay
            .segmented_parser_components
            .get(owner)
            .ok_or_else(|| format!("recursive tokenizer leaf {leaf_index} has invalid owner {owner}"))?;
        let component_state = constraint
            .recursive_tokenizer_state_for_component(owner, scoped_state)
            .ok_or_else(|| {
                format!(
                    "recursive tokenizer state {scoped_state} does not project to owner component {owner}"
                )
            })?;
        let local_tsids = component
            .constraint
            .runtime_internal_tsids_for_tokenizer_state(component_state)
            .ok_or_else(|| {
                format!(
                    "recursive tokenizer state {scoped_state} projects to component {owner} state {component_state} without a TSID image"
                )
            })?;
        let map = &component_maps[owner].local_to_global_tsids;
        let mut global_tsids = Vec::new();
        for local_tsid in local_tsids {
            let targets = map.get(local_tsid as usize).ok_or_else(|| {
                format!(
                    "component {owner} TSID {local_tsid} lies outside its composition map"
                )
            })?;
            global_tsids.extend_from_slice(targets);
        }
        global_tsids.sort_unstable();
        global_tsids.dedup();
        if global_tsids.is_empty() {
            return Err(format!(
                "recursive tokenizer state {scoped_state} has empty composed TSID image"
            ));
        }
        relation.push(global_tsids);
    }
    Ok(relation)
}

/// Invert one component's local->global LR-state relation for runtime A
/// projection. A local state may appear at several composed states (for
/// example distinct call/ignore phases); that is harmless because each such
/// global state still projects back to the same local state. The representation
/// is unavailable only when one global state would project to two different
/// local states of this component.
pub(super) fn build_segmented_parser_state_offsets(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> Result<Vec<u32>, String> {
    let mut offsets = Vec::with_capacity(children.len() + 1);
    let mut next = 0u32;
    for constraint in std::iter::once(parent).chain(children.iter().map(|child| child.constraint)) {
        offsets.push(next);
        next = next
            .checked_add(constraint.table.num_states)
            .ok_or_else(|| "segmented parser-state coordinate overflow".to_owned())?;
    }
    Ok(offsets)
}


pub(super) fn invert_functional_parser_state_relation(
    relation: &[Vec<u32>],
    global_state_count: usize,
    excluded_local_state: Option<u32>,
) -> Option<Vec<u32>> {
    let mut inverse = vec![u32::MAX; global_state_count];
    for (local_state, targets) in relation.iter().enumerate() {
        if excluded_local_state == Some(local_state as u32) {
            continue;
        }
        for &global_state in targets {
            let slot = inverse.get_mut(global_state as usize)?;
            if *slot != u32::MAX && *slot != local_state as u32 {
                return None;
            }
            *slot = local_state as u32;
        }
    }
    Some(inverse)
}

pub(super) fn standalone_eof_accept_state(constraint: &Constraint) -> Option<u32> {
    let mut states = constraint
        .table
        .action
        .iter()
        .enumerate()
        .filter_map(|(state, row)| {
            matches!(row.get(&EOF), Some(Action::Accept)).then_some(state as u32)
        });
    let state = states.next()?;
    states.next().is_none().then_some(state)
}

#[inline]
pub(super) fn runtime_weight_survives_component_coordinate_maps(
    weight: crate::runtime::RuntimeWeightRef<'_>,
    maps: &DirectComponentCoordinateMaps,
) -> bool {
    if weight.is_empty() {
        return false;
    }
    if weight.is_full() {
        return maps.local_to_global_tsids.iter().any(|targets| !targets.is_empty())
            && maps.local_to_global_tokens.iter().any(|targets| !targets.is_empty());
    }
    let mut survives = false;
    weight.for_each_entry(|start_tsid, end_tsid, tokens| {
        if survives {
            return;
        }
        let has_tsid = (start_tsid..=end_tsid).any(|tsid| {
            maps.local_to_global_tsids
                .get(tsid as usize)
                .is_some_and(|targets| !targets.is_empty())
        });
        if !has_tsid {
            return;
        }
        tokens.for_each_range(|start, end| {
            if !survives
                && (start..=end).any(|token| {
                    maps.local_to_global_tokens
                        .get(token as usize)
                        .is_some_and(|targets| !targets.is_empty())
                })
            {
                survives = true;
            }
        });
    });
    survives
}

#[inline]
pub(super) fn weight_survives_component_coordinate_maps(
    weight: &Weight,
    maps: &DirectComponentCoordinateMaps,
) -> bool {
    runtime_weight_survives_component_coordinate_maps(weight.into(), maps)
}

pub(super) fn deterministic_component_union_root_dispatch_direct(
    components: &[crate::runtime::SegmentedParserComponent],
    component_maps: &[DirectComponentCoordinateMaps],
    global_state_count: usize,
) -> Option<Vec<u32>> {
    let started_at = Instant::now();
    if components.len() != component_maps.len()
        || components
            .iter()
            .any(|component| component.global_to_local_parser_state.len() < global_state_count)
    {
        return None;
    }
    let mut dispatch = vec![u32::MAX; global_state_count];
    let mut selected_counts = vec![0usize; components.len()];
    let mut dead = 0usize;
    let mut syntactic_overlaps = 0usize;
    for global_state in 0..global_state_count {
        let mut candidates = SmallVec::<[(u32, crate::runtime::RuntimeWeightRef<'_>); 4]>::new();
        for (component_index, component) in components.iter().enumerate() {
            let local_state = component.global_to_local_parser_state[global_state];
            if local_state == u32::MAX {
                continue;
            }
            let source = component.constraint.as_ref();
            let root = source.runtime_parser_dwa_start_state();
            let Some((_, weight)) = source.runtime_parser_dwa_transition(root, local_state) else {
                continue;
            };
            if !weight.is_empty() {
                candidates.push((component_index as u32, weight));
            }
        }
        let selected = match candidates.len() {
            0 => None,
            1 => Some(candidates[0].0),
            _ => {
                syntactic_overlaps += 1;
                let mut live = SmallVec::<[u32; 4]>::new();
                for (component_index, weight) in candidates {
                    if runtime_weight_survives_component_coordinate_maps(
                        weight,
                        &component_maps[component_index as usize],
                    ) {
                        live.push(component_index);
                    }
                }
                match live.as_slice() {
                    [] => None,
                    [component] => Some(*component),
                    _ => return None,
                }
            }
        };
        if let Some(component) = selected {
            dispatch[global_state] = component;
            selected_counts[component as usize] += 1;
        } else {
            dead += 1;
        }
    }
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_component_union_root_dispatch_direct] deterministic=true global_states={} dead={} selected_counts={selected_counts:?} syntactic_overlaps={syntactic_overlaps} total_ms={:.3}",
            global_state_count,
            dead,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(dispatch)
}

/// Certify the compressed deterministic union of cached component parser DWAs.
///
/// The materialized selected10 component union has exactly one synthetic state:
/// its start subset.  The equivalent zero-copy representation is therefore a
/// synthetic deterministic root whose first parser-state read selects one
/// cached component DWA body.  This helper proves the required property rather
/// than assuming it: for every composed LR state, at most one component start
/// row may have a live transition after projection to that component's local LR
/// coordinate.  Any overlap declines the fast representation and lets the
/// ordinary materialized deterministic union remain the fallback.
pub(super) fn build_segmented_runtime_metadata(
    source_constraints: Vec<(Arc<Constraint>, Option<u32>)>,
    parser_state_relations: &[Vec<Vec<u32>>],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    automata_maps: &[DirectComponentCoordinateMaps],
    global_state_count: usize,
    two_dwa_runtime_requested: bool,
    parser_default_domains: &ParserDefaultDomainPlan,
    id_num_tsids: u32,
    canonical_terminal: Option<u32>,
    terminal_aliases: &[u32],
) -> Result<(Vec<crate::runtime::SegmentedParserComponent>, Option<Vec<u32>>, f64), String> {
    let started_at = Instant::now();
    let global_terminal_count = source_constraints
        .iter()
        .enumerate()
        .map(|(component_index, (source, _))| {
            terminal_offsets[component_index] as usize + source.table.num_terminals as usize
        })
        .max()
        .unwrap_or(0);
    let mut segmented_components = Vec::with_capacity(source_constraints.len());
    for (component_index, (source, root_disallowed_terminal)) in source_constraints.into_iter().enumerate() {
        let excluded_local_state = if component_index == 0 {
            None
        } else {
            Some(standalone_eof_accept_state(source.as_ref()).ok_or_else(|| {
                format!(
                    "segmented parser child component {component_index} has no unique EOF accept state"
                )
            })?)
        };
        let global_to_local_parser_state = invert_functional_parser_state_relation(
            &parser_state_relations[component_index],
            global_state_count,
            excluded_local_state,
        ).ok_or_else(|| format!("segmented parser component {component_index} has a non-functional LR-state relation"))?;
        let terminal_offset = terminal_offsets[component_index];
        let mut global_terminal_aliases = Vec::new();
        if let Some(canonical) = canonical_terminal {
            let component_end = terminal_offset + source.table.num_terminals;
            for &alias in terminal_aliases {
                if alias >= terminal_offset && alias < component_end {
                    global_terminal_aliases.push((canonical, alias - terminal_offset));
                }
            }
        }
        segmented_components.push(crate::runtime::SegmentedParserComponent {
            constraint: source,
            boundary: None,
            tokenizer_state_offset: tokenizer_state_offsets[component_index],
            terminal_offset,
            global_terminal_aliases,
            local_tsid_to_global_tsids: automata_maps[component_index]
                .local_to_global_tsids
                .clone(),
            root_disallowed_terminal,
            global_to_local_parser_state,
        });
    }
    let deterministic_root_dispatch = if two_dwa_runtime_requested {
        Some({
            let direct = deterministic_component_union_root_dispatch_direct(
                &segmented_components, automata_maps, global_state_count,
            ).ok_or_else(|| "two-DWA runtime requires a deterministic root-only component parser union".to_string())?;
            if std::env::var_os("GLRMASK_VALIDATE_DIRECT_COMPONENT_ROOT_DISPATCH").is_some() {
                let reference = deterministic_component_union_root_dispatch(
                    &segmented_components, parser_state_relations,
                    &parser_default_domains.component_domains,
                    &parser_default_domains.parser_state_labels,
                    automata_maps, id_num_tsids as usize, global_state_count,
                ).ok_or_else(|| "reference component root dispatch failed during direct validation".to_string())?;
                assert_eq!(direct, reference, "direct component root dispatch differs from transported-root reference");
                eprintln!("[glrmask/validate][direct_component_root_dispatch] exact=true states={global_state_count}");
            }
            direct
        })
    } else { None };
    Ok((segmented_components, deterministic_root_dispatch, started_at.elapsed().as_secs_f64() * 1000.0))
}

pub(super) fn segmented_boundary_start_parser_states(
    component: &crate::runtime::SegmentedParserComponent,
) -> BitSet {
    let mut start_parser_states = BitSet::new(component.global_to_local_parser_state.len());
    for (global_state, &local_state) in component
        .global_to_local_parser_state
        .iter()
        .enumerate()
    {
        if local_state != u32::MAX {
            start_parser_states.set(global_state);
        }
    }
    start_parser_states
}

pub(super) fn install_segmented_boundary_shards(
    overlay: &mut crate::runtime::StaticDynamicOverlayMetadata,
    backend: crate::runtime::SegmentedBoundaryShardBackend,
    boundary_tokens_by_start_component: Option<&[Vec<u32>]>,
) {
    overlay.segmented_boundary_shards.clear();
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for component_index in 0..overlay.segmented_parser_components.len() {
        let start_parser_states = segmented_boundary_start_parser_states(
            &overlay.segmented_parser_components[component_index],
        );
        let candidate_tokens = boundary_tokens_by_start_component
            .and_then(|rows| rows.get(component_index))
            .map(|tokens| Arc::<[u32]>::from(tokens.clone()));
        let shard = crate::runtime::SegmentedBoundaryShard {
            start_component: component_index as u32,
            start_parser_states,
            accepts_empty_stack: component_index == 0,
            candidate_tokens,
            backend: backend.clone(),
        };
        overlay.segmented_parser_components[component_index].boundary = Some(shard.clone());
        // Temporary wire-format compatibility: v23 still serializes shards as
        // a parallel list. Live runtime dispatch uses the wrapper field above.
        overlay.segmented_boundary_shards.push(shard);
    }
}

pub(crate) fn install_published_static_boundary_shards(
    overlay: &mut crate::runtime::StaticDynamicOverlayMetadata,
    shards: Vec<PublishedStaticBoundaryShard>,
) -> Result<(), String> {
    overlay.segmented_boundary_shards.clear();
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for shard in shards {
        let component_index = shard.start_component as usize;
        let component = overlay
            .segmented_parser_components
            .get(component_index)
            .ok_or_else(|| {
                format!(
                    "static boundary shard references unknown start component {}",
                    shard.start_component,
                )
            })?;
        let runtime_shard = crate::runtime::SegmentedBoundaryShard {
            start_component: shard.start_component,
            start_parser_states: segmented_boundary_start_parser_states(component),
            accepts_empty_stack: component_index == 0,
            candidate_tokens: Some(shard.candidate_tokens),
            backend: crate::runtime::SegmentedBoundaryShardBackend::StaticParser(shard.boundary),
        };
        overlay.segmented_parser_components[component_index].boundary = Some(runtime_shard.clone());
        overlay.segmented_boundary_shards.push(runtime_shard);
    }
    Ok(())
}

pub(super) fn install_dynamic_direct_boundary_shards(
    overlay: &mut crate::runtime::StaticDynamicOverlayMetadata,
    boundary_tokens_by_start_component: Option<&[Vec<u32>]>,
) {
    overlay.segmented_boundary_shards.clear();
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for component_index in 0..overlay.segmented_parser_components.len() {
        let shard = crate::runtime::SegmentedBoundaryShard {
            start_component: component_index as u32,
            start_parser_states: segmented_boundary_start_parser_states(
                &overlay.segmented_parser_components[component_index],
            ),
            accepts_empty_stack: component_index == 0,
            candidate_tokens: boundary_tokens_by_start_component
                .and_then(|rows| rows.get(component_index))
                .map(|tokens| Arc::<[u32]>::from(tokens.clone())),
            backend: crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect,
        };
        overlay.segmented_parser_components[component_index].boundary = Some(shard.clone());
        overlay.segmented_boundary_shards.push(shard);
    }
}

pub(super) fn append_dynamic_direct_boundary_shards_for_unselected(
    overlay: &mut crate::runtime::StaticDynamicOverlayMetadata,
    static_components: &BitSet,
    boundary_tokens_by_start_component: Option<&[Vec<u32>]>,
) {
    for component_index in 0..overlay.segmented_parser_components.len() {
        if static_components.contains(component_index) {
            continue;
        }
        let shard = crate::runtime::SegmentedBoundaryShard {
            start_component: component_index as u32,
            start_parser_states: segmented_boundary_start_parser_states(
                &overlay.segmented_parser_components[component_index],
            ),
            accepts_empty_stack: component_index == 0,
            candidate_tokens: boundary_tokens_by_start_component
                .and_then(|rows| rows.get(component_index))
                .map(|tokens| Arc::<[u32]>::from(tokens.clone())),
            backend: crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect,
        };
        overlay.segmented_parser_components[component_index].boundary = Some(shard.clone());
        overlay.segmented_boundary_shards.push(shard);
    }
    overlay
        .segmented_boundary_shards
        .sort_unstable_by_key(|shard| shard.start_component);
}

pub(super) fn deterministic_component_union_root_dispatch(
    components: &[crate::runtime::SegmentedParserComponent],
    parser_state_relations: &[Vec<Vec<u32>>],
    default_domains: &[Option<ParserDefaultDomain>],
    global_domain_labels: &[i32],
    component_maps: &[DirectComponentCoordinateMaps],
    global_tsid_count: usize,
    global_state_count: usize,
) -> Option<Vec<u32>> {
    let total_started_at = Instant::now();
    if components.len() != parser_state_relations.len()
        || components.len() != default_domains.len()
        || components.len() != component_maps.len()
        || (!global_domain_labels.is_empty() && global_domain_labels.len() < global_state_count)
    {
        return None;
    }
    // Transport only each component's start row using the exact same linker
    // semantics as `component_parser_nwa`. Targets may refer to body state IDs
    // not present in this one-state scratch NWA; they are opaque here because
    // certification needs only the root label relation.
    let transport_started_at = Instant::now();
    let mut transported_roots = Vec::with_capacity(components.len());
    for (component_index, component) in components.iter().enumerate() {
        let source = component.constraint.as_ref();
        let source_root = source
            .parser_dwa
            .states()
            .get(source.parser_dwa.start_state() as usize)?;
        let mut transported = NWA::new(0, 0);
        transported.add_state();
        add_component_parser_state_transitions(
            &mut transported,
            0,
            source_root,
            &parser_state_relations[component_index],
            &source.parser_state_domain_labels,
            source.table.num_states,
            default_domains[component_index].as_ref(),
        )
        .ok()?;
        transported_roots.push(transported);
    }
    let transport_ms = transport_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut dispatch = vec![u32::MAX; global_state_count];
    let mut selected_counts = vec![0usize; components.len()];
    let mut dead = 0usize;
    let mut syntactic_overlaps = 0usize;
    let mut overlap_weight_remap_ms = 0.0f64;
    for global_state in 0..global_state_count {
        let mut candidates = SmallVec::<[(u32, Weight); 4]>::new();
        for (component_index, root) in transported_roots.iter().enumerate() {
            let row = &root.states().first()?.transitions;
            let positive = encode_positive_label(global_state as u32);
            let targets = row
                .get(&positive)
                .or_else(|| {
                    let domain = global_domain_labels
                        .get(global_state)
                        .copied()
                        .unwrap_or(NO_PARSER_DOMAIN_LABEL);
                    (domain != NO_PARSER_DOMAIN_LABEL)
                        .then(|| row.get(&domain))
                        .flatten()
                })
                .or_else(|| row.get(&DEFAULT_LABEL));
            let Some(targets) = targets else {
                continue;
            };
            let live_targets = targets
                .iter()
                .filter(|(_, weight)| !weight.is_empty())
                .count();
            if live_targets == 0 {
                continue;
            }
            if live_targets != 1 {
                if compose_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][constraint_component_union_root_dispatch] deterministic=false component={} global_state={} component_targets={live_targets}",
                        component_index,
                        global_state,
                    );
                }
                return None;
            }
            let weight = targets
                .iter()
                .find_map(|(_, weight)| (!weight.is_empty()).then(|| weight.clone()))?;
            candidates.push((component_index as u32, weight));
        }
        let selected = match candidates.len() {
            0 => None,
            1 => Some(candidates[0].0),
            _ => {
                syntactic_overlaps += 1;
                // Only syntactically overlapping root labels need common
                // weight coordinates. Unique labels are deterministic already;
                // remapping all ~8k root edges was pure certification overhead.
                let started_at = Instant::now();
                let mut live = SmallVec::<[u32; 4]>::new();
                for (component_index, weight) in candidates {
                    let maps = &component_maps[component_index as usize];
                    if weight_survives_component_coordinate_maps(&weight, maps) {
                        live.push(component_index);
                    }
                }
                overlap_weight_remap_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                match live.as_slice() {
                    [] => None,
                    [component] => Some(*component),
                    _ => {
                        if compose_profile_enabled() {
                            eprintln!(
                                "[glrmask/profile][constraint_component_union_root_dispatch] deterministic=false overlap_global_state={global_state} live_components={live:?}"
                            );
                        }
                        return None;
                    }
                }
            }
        };
        if let Some(component) = selected {
            dispatch[global_state] = component;
            selected_counts[component as usize] += 1;
        } else {
            dead += 1;
        }
    }
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_component_union_root_dispatch] deterministic=true global_states={} dead={} selected_counts={selected_counts:?} syntactic_overlaps={syntactic_overlaps} transport_ms={transport_ms:.3} overlap_weight_remap_ms={overlap_weight_remap_ms:.3} total_ms={:.3}",
            global_state_count,
            dead,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(dispatch)
}

#[inline]
pub(super) fn tokenizer_tsid_relation_is_singleton(constraint: &Constraint) -> bool {
    constraint.state_internal_tsid_offsets.as_slice() == [u32::MAX]
}

pub(super) fn build_direct_component_state_coordinates_from_constraints(
    components: &[(&Constraint, u32)],
    merged_tokenizer_state_count: usize,
) -> Result<DirectComponentStateCoordinates, String> {
    let mut state_to_global = vec![u32::MAX; merged_tokenizer_state_count];
    let mut global_to_states = vec![vec![0u32]];
    let mut state_representatives = vec![0u32];
    if let Some(reset) = state_to_global.first_mut() {
        *reset = 0;
    }

    let mut local_to_global_tsids = Vec::with_capacity(components.len());
    for (component_index, &(constraint, tokenizer_state_offset)) in components.iter().enumerate() {
        let component_tokenizer = constraint.composition_tokenizer();
        // Dynamic constraints deliberately omit the static parser-DWA TSID
        // quotient. Preserve that backend and use the exact raw tokenizer
        // coordinate as this component's private TSID coordinate instead of
        // compiling/materializing a quotient solely for linking.
        if constraint.state_to_internal_tsid.is_empty()
            && constraint.internal_tsid_to_states.is_empty()
        {
            let local_tsid_count = component_tokenizer.num_states() as usize;
            if local_tsid_count == 0 {
                return Err("component tokenizer contains no states".into());
            }
            let mut local_map = vec![Vec::<u32>::new(); local_tsid_count];
            for local_state in 0..component_tokenizer.num_states() {
                let merged_state = tokenizer_state_offset
                    .checked_add(local_state)
                    .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
                if merged_state == 0 {
                    local_map[local_state as usize].push(0);
                    continue;
                }
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to more than one component class"
                    ));
                }
                let global_tsid = global_to_states.len() as u32;
                *slot = global_tsid;
                state_representatives.push(merged_state);
                global_to_states.push(vec![merged_state]);
                local_map[local_state as usize].push(global_tsid);
            }
            let local_start = component_tokenizer.initial_state() as usize;
            let Some(start_targets) = local_map.get_mut(local_start) else {
                return Err("component start tokenizer state lies outside its raw-state domain".into());
            };
            start_targets.push(0);
            start_targets.sort_unstable();
            start_targets.dedup();
            local_to_global_tsids.push(local_map);
            continue;
        }
        if constraint.state_to_internal_tsid.len() != component_tokenizer.num_states() as usize {
            return Err(format!(
                "component {component_index} tokenizer-state map does not cover its runtime tokenizer: mapped_states={} tokenizer_states={} dynamic={}",
                constraint.state_to_internal_tsid.len(),
                component_tokenizer.num_states(),
                constraint.uses_dynamic_runtime(),
            ));
        }
        let internal_tsid_to_states = constraint.internal_tsid_groups();
        let local_tsid_count = internal_tsid_to_states.len();
        if local_tsid_count == 0 {
            return Err("component tokenizer-state map contains no internal TSIDs".into());
        }
        let mut local_map = vec![Vec::<u32>::new(); local_tsid_count];

        if tokenizer_tsid_relation_is_singleton(constraint) {
            for (local_tsid, local_states) in internal_tsid_to_states.iter().enumerate() {
                if local_states.is_empty() {
                    continue;
                }
                let mut merged_states = Vec::with_capacity(local_states.len());
                for &local_state in local_states {
                    let merged_state = tokenizer_state_offset.checked_add(local_state)
                        .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
                    if merged_state == 0 {
                        local_map[local_tsid].push(0);
                        continue;
                    }
                    let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                        return Err(format!(
                            "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
                        ));
                    };
                    if *slot != u32::MAX {
                        return Err(format!(
                            "merged tokenizer state {merged_state} belongs to more than one component class"
                        ));
                    }
                    merged_states.push(merged_state);
                }
                if !merged_states.is_empty() {
                    let global_tsid = global_to_states.len() as u32;
                    for &merged_state in &merged_states {
                        state_to_global[merged_state as usize] = global_tsid;
                    }
                    state_representatives.push(merged_states[0]);
                    global_to_states.push(merged_states);
                    local_map[local_tsid].push(global_tsid);
                }
            }
            let local_start = constraint.tokenizer.initial_state() as usize;
            let start_tsid = constraint
                .state_to_internal_tsid
                .get(local_start)
                .copied()
                .ok_or_else(|| "component start tokenizer state has no internal TSID".to_string())?;
            let Some(start_targets) = local_map.get_mut(start_tsid as usize) else {
                return Err("component start TSID maps outside its internal TSID domain".into());
            };
            start_targets.push(0);
            start_targets.sort_unstable();
            start_targets.dedup();
            local_to_global_tsids.push(local_map);
            continue;
        }

        // A runtime-product tokenizer state can intentionally represent several
        // pre-product TSIDs.  A many-to-one global coordinate cannot identify
        // such a state with *each* constituent TSID independently.  Instead,
        // quotient raw states by their exact local-TSID membership signature.
        // A local TSID then maps to every global signature-class containing it.
        // This is the exact powerset lifting of the local TSID relation and
        // degenerates to the historical one-class-per-local-TSID mapping when
        // every raw state has singleton membership.
        let mut states_by_signature = BTreeMap::<Vec<u32>, Vec<u32>>::new();
        for local_state in 0..component_tokenizer.num_states() {
            let mut signature = constraint.internal_tsids_for_state(local_state).to_vec();
            signature.sort_unstable();
            signature.dedup();
            if signature.is_empty() {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} has no internal TSID"
                ));
            }
            if let Some(&bad) = signature
                .iter()
                .find(|&&tsid| tsid as usize >= local_tsid_count)
            {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} references out-of-range internal TSID {bad}"
                ));
            }
            let merged_state = tokenizer_state_offset.checked_add(local_state)
                .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
            if merged_state as usize >= merged_tokenizer_state_count {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
                ));
            }
            // State zero is the merged reset coordinate. In the owned-parent
            // layout the parent's local reset can physically be state zero;
            // its complete TSID signature is attached to global reset below.
            if merged_state != 0 {
                states_by_signature
                    .entry(signature)
                    .or_default()
                    .push(merged_state);
            }
        }

        for (signature, mut merged_states) in states_by_signature {
            merged_states.sort_unstable();
            merged_states.dedup();
            let global_tsid = global_to_states.len() as u32;
            for &merged_state in &merged_states {
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {merged_state} lies outside merged tokenizer"
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to more than one exact membership class"
                    ));
                }
                *slot = global_tsid;
            }
            state_representatives.push(merged_states[0]);
            global_to_states.push(merged_states);
            for local_tsid in signature {
                local_map[local_tsid as usize].push(global_tsid);
            }
        }

        let local_start = constraint.tokenizer.initial_state();
        let mut start_signature = constraint.internal_tsids_for_state(local_start).to_vec();
        start_signature.sort_unstable();
        start_signature.dedup();
        if start_signature.is_empty() {
            return Err("component start tokenizer state has no internal TSID".into());
        }
        for start_tsid in start_signature {
            let Some(start_targets) = local_map.get_mut(start_tsid as usize) else {
                return Err("component start TSID maps outside its internal TSID domain".into());
            };
            start_targets.push(0);
        }
        for targets in &mut local_map {
            targets.sort_unstable();
            targets.dedup();
        }
        local_to_global_tsids.push(local_map);
    }
    if state_to_global.iter().any(|&tsid| tsid == u32::MAX) {
        return Err("direct component state map does not cover the merged tokenizer".into());
    }
    Ok(DirectComponentStateCoordinates {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_to_global,
            internal_to_originals: global_to_states,
            representative_original_ids: state_representatives,
        },
        local_to_global_tsids,
    })

}

pub(super) fn build_direct_component_state_coordinates_from_precomputed_map(
    components: &[ParserDwaComponent<'_>],
    tokenizer_states: &ManyToOneIdMap,
    merged_tokenizer_state_count: usize,
) -> Result<DirectComponentStateCoordinates, String> {
    if tokenizer_states.original_to_internal.len() != merged_tokenizer_state_count {
        return Err("precomputed component state map does not cover merged tokenizer".into());
    }
    let mut local_to_global_tsids = Vec::with_capacity(components.len());
    for (component_index, component) in components.iter().enumerate() {
        let constraint = component.constraint;
        let local_tsid_count = constraint.internal_tsid_groups().len();
        if local_tsid_count == 0 {
            return Err(format!("component {component_index} has no internal TSIDs"));
        }
        let mut local_map = vec![Vec::<u32>::new(); local_tsid_count];
        for local_state in 0..constraint.tokenizer.num_states() {
            let merged_state = component
                .tokenizer_state_offset
                .checked_add(local_state)
                .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
            let global_tsid = tokenizer_states
                .original_to_internal
                .get(merged_state as usize)
                .copied()
                .ok_or_else(|| {
                    format!(
                        "component {component_index} tokenizer state {local_state} maps outside precomputed state coordinate"
                    )
                })?;
            if global_tsid == u32::MAX {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} is unmapped in precomputed state coordinate"
                ));
            }
            let local_tsids = constraint.internal_tsids_for_state(local_state);
            if local_tsids.is_empty() {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} has no local TSID"
                ));
            }
            for &local_tsid in local_tsids {
                let Some(targets) = local_map.get_mut(local_tsid as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {local_state} references out-of-range local TSID {local_tsid}"
                    ));
                };
                targets.push(global_tsid);
            }
        }
        // Global raw state zero epsilon-dispatches to every component start.
        // Preserve the same reset membership added by the full coordinate builder.
        let local_start = constraint.tokenizer.initial_state();
        for &local_tsid in constraint.internal_tsids_for_state(local_start) {
            let Some(targets) = local_map.get_mut(local_tsid as usize) else {
                return Err(format!(
                    "component {component_index} start state references out-of-range local TSID {local_tsid}"
                ));
            };
            targets.push(0);
        }
        for targets in &mut local_map {
            targets.sort_unstable();
            targets.dedup();
        }
        local_to_global_tsids.push(local_map);
    }
    Ok(DirectComponentStateCoordinates {
        tokenizer_states: tokenizer_states.clone(),
        local_to_global_tsids,
    })
}

pub(super) fn build_direct_component_state_coordinates(
    components: &[ParserDwaComponent<'_>],
    merged_tokenizer_state_count: usize,
) -> Result<DirectComponentStateCoordinates, String> {
    let inputs = components
        .iter()
        .map(|component| (component.constraint, component.tokenizer_state_offset))
        .collect::<Vec<_>>();
    build_direct_component_state_coordinates_from_constraints(&inputs, merged_tokenizer_state_count)
}

pub(super) fn build_direct_component_token_coordinates(
    components: &[ParserDwaComponent<'_>],
    original_token_ids: &[u32],
) -> Result<(ManyToOneIdMap, Vec<Vec<Vec<u32>>>, bool), String> {
    let original_token_count = original_token_ids
        .last()
        .map_or(0, |token| *token as usize + 1);
    let mut selected_original = vec![false; original_token_count];
    for &original in original_token_ids {
        selected_original[original as usize] = true;
    }
    let mut token_to_global = vec![u32::MAX; original_token_count];
    let mut global_to_tokens = Vec::<Vec<u32>>::new();
    let mut token_representatives = Vec::<u32>::new();
    let mut local_to_global_tokens = components
        .iter()
        .map(|component| {
            let constraint = component.constraint;
            let local_count = if constraint.has_original_token_map() {
                constraint.internal_token_count()
            } else {
                original_token_count
            };
            vec![Vec::<u32>::new(); local_count]
        })
        .collect::<Vec<_>>();

    // A retained dynamic component has no static internal-token quotient: its
    // exact private coordinate is the original model-token ID. That coordinate
    // is injective, so once it participates in a product refinement no two
    // distinct original tokens can ever share a global class. Construct that
    // singleton refinement directly instead of inserting one unique Vec-key per
    // token into a BTreeMap merely to rediscover the identity partition.
    if components
        .iter()
        .any(|component| !component.constraint.has_original_token_map())
    {
        for &original in original_token_ids {
            let global = global_to_tokens.len() as u32;
            token_to_global[original as usize] = global;
            for (component_index, component) in components.iter().enumerate() {
                let constraint = component.constraint;
                let local = if constraint.has_original_token_map() {
                    constraint
                        .original_token_internal_at(original)
                        .unwrap_or(u32::MAX)
                } else {
                    // `original_token_ids` is already the composed public token
                    // domain: supplied vocabulary IDs plus live merged exact
                    // specials. Compiler-only late-grammar sentinels are removed
                    // before this point and are therefore not identity members.
                    original
                };
                if local == u32::MAX {
                    continue;
                }
                let Some(destinations) = local_to_global_tokens
                    .get_mut(component_index)
                    .and_then(|classes| classes.get_mut(local as usize))
                else {
                    return Err(format!(
                        "component {component_index} token class {local} lies outside its internal token domain"
                    ));
                };
                destinations.push(global);
            }
            token_representatives.push(original);
            global_to_tokens.push(vec![original]);
        }
        return Ok((
            ManyToOneIdMap {
                original_to_internal: token_to_global,
                internal_to_originals: global_to_tokens,
                representative_original_ids: token_representatives,
            },
            local_to_global_tokens,
            true,
        ));
    }

    // The overwhelmingly common subgrammar link is parent + one composed
    // child. Its exact common vocabulary partition is keyed by one child class
    // inside each parent class; avoid allocating a temporary Vec tuple for
    // every original token and using that Vec as a tree key.
    if components.len() == 2 {
        let parent = components[0].constraint;
        let child = components[1].constraint;
        let mut add_pair_class =
            |originals: Vec<u32>, parent_local: u32, child_local: u32| -> Result<(), String> {
                if originals.is_empty()
                    || (parent_local == u32::MAX && child_local == u32::MAX)
                {
                    return Ok(());
                }
                let global = global_to_tokens.len() as u32;
                for &original in &originals {
                    token_to_global[original as usize] = global;
                }
                for (component_index, local) in [parent_local, child_local].into_iter().enumerate() {
                    if local == u32::MAX {
                        continue;
                    }
                    let Some(destinations) = local_to_global_tokens
                        .get_mut(component_index)
                        .and_then(|classes| classes.get_mut(local as usize))
                    else {
                        return Err(format!(
                            "component {component_index} token class {local} lies outside its internal token domain"
                        ));
                    };
                    destinations.push(global);
                }
                token_representatives.push(originals[0]);
                global_to_tokens.push(originals);
                Ok(())
            };

        let singleton_fast =
            std::env::var_os("GLRMASK_DISABLE_TOKEN_COORD_SINGLETON_FAST").is_none();
        let dense_child_buckets =
            std::env::var_os("GLRMASK_DISABLE_TOKEN_COORD_DENSE_CHILD_BUCKETS").is_none();
        if dense_child_buckets {
            let child_class_count = child.internal_token_count();
            let mut buckets = (0..child_class_count)
                .map(|_| Vec::<u32>::new())
                .collect::<Vec<_>>();
            let mut touched = Vec::<usize>::new();
            let mut touched_flags = vec![false; child_class_count];
            let mut unmapped = Vec::<u32>::new();

            if let Some(parent_token_groups) = parent.internal_token_groups() {
                for (parent_local, originals) in parent_token_groups.iter().enumerate() {
                    if singleton_fast && originals.len() == 1 {
                        let original = originals[0];
                        if selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            let child_local = child
                                .original_token_internal_at(original)
                                .unwrap_or(u32::MAX);
                            add_pair_class(vec![original], parent_local as u32, child_local)?;
                        }
                        continue;
                    }
                    for &original in originals {
                        if !selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let child_local = child
                            .original_token_internal_at(original)
                            .unwrap_or(u32::MAX);
                        if child_local == u32::MAX {
                            unmapped.push(original);
                            continue;
                        }
                        let child_index = child_local as usize;
                        if child_index >= child_class_count {
                            return Err(format!(
                                "component 1 token class {child_local} lies outside its internal token domain"
                            ));
                        }
                        if !touched_flags[child_index] {
                            touched_flags[child_index] = true;
                            touched.push(child_index);
                        }
                        buckets[child_index].push(original);
                    }
                    touched.sort_unstable();
                    for child_index in touched.drain(..) {
                        touched_flags[child_index] = false;
                        let originals = std::mem::take(&mut buckets[child_index]);
                        add_pair_class(originals, parent_local as u32, child_index as u32)?;
                    }
                    if !unmapped.is_empty() {
                        add_pair_class(
                            std::mem::take(&mut unmapped),
                            parent_local as u32,
                            u32::MAX,
                        )?;
                    }
                }
            }

            for &original in original_token_ids {
                let parent_local = parent
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                if parent_local != u32::MAX {
                    continue;
                }
                let child_local = child
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                if child_local == u32::MAX {
                    unmapped.push(original);
                    continue;
                }
                let child_index = child_local as usize;
                if child_index >= child_class_count {
                    return Err(format!(
                        "component 1 token class {child_local} lies outside its internal token domain"
                    ));
                }
                if !touched_flags[child_index] {
                    touched_flags[child_index] = true;
                    touched.push(child_index);
                }
                buckets[child_index].push(original);
            }
            touched.sort_unstable();
            for child_index in touched.drain(..) {
                touched_flags[child_index] = false;
                let originals = std::mem::take(&mut buckets[child_index]);
                add_pair_class(originals, u32::MAX, child_index as u32)?;
            }
            if !unmapped.is_empty() {
                add_pair_class(std::mem::take(&mut unmapped), u32::MAX, u32::MAX)?;
            }
        } else {
            if let Some(parent_token_groups) = parent.internal_token_groups() {
                for (parent_local, originals) in parent_token_groups.iter().enumerate() {
                    if singleton_fast && originals.len() == 1 {
                        let original = originals[0];
                        if selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            let child_local = child
                                .original_token_internal_at(original)
                                .unwrap_or(u32::MAX);
                            add_pair_class(vec![original], parent_local as u32, child_local)?;
                        }
                        continue;
                    }
                    let mut groups = BTreeMap::<u32, Vec<u32>>::new();
                    for &original in originals {
                        if !selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let child_local = child
                            .original_token_internal_at(original)
                            .unwrap_or(u32::MAX);
                        groups.entry(child_local).or_default().push(original);
                    }
                    for (child_local, originals) in groups {
                        add_pair_class(originals, parent_local as u32, child_local)?;
                    }
                }
            }

            let mut parent_unmapped = BTreeMap::<u32, Vec<u32>>::new();
            for &original in original_token_ids {
                let parent_local = parent
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                if parent_local != u32::MAX {
                    continue;
                }
                let child_local = child
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                parent_unmapped.entry(child_local).or_default().push(original);
            }
            for (child_local, originals) in parent_unmapped {
                add_pair_class(originals, u32::MAX, child_local)?;
            }
        }

        return Ok((
            ManyToOneIdMap {
                original_to_internal: token_to_global,
                internal_to_originals: global_to_tokens,
                representative_original_ids: token_representatives,
            },
            local_to_global_tokens,
            false,
        ));
    }
    let mut add_token_class = |originals: Vec<u32>, tuple: Vec<u32>| -> Result<(), String> {
        if originals.is_empty() || tuple.iter().all(|&local| local == u32::MAX) {
            return Ok(());
        }
        let global = global_to_tokens.len() as u32;
        for &original in &originals {
            token_to_global[original as usize] = global;
        }
        for (component_index, &local) in tuple.iter().enumerate() {
            if local == u32::MAX {
                continue;
            }
            let Some(destinations) = local_to_global_tokens
                .get_mut(component_index)
                .and_then(|classes| classes.get_mut(local as usize))
            else {
                return Err(format!(
                    "component {component_index} token class {local} lies outside its internal token domain"
                ));
            };
            destinations.push(global);
        }
        token_representatives.push(originals[0]);
        global_to_tokens.push(originals);
        Ok(())
    };

    let parent = components[0].constraint;
    let parent_token_groups = parent
        .internal_token_groups()
        .ok_or_else(|| "composed parent has no explicit internal-token partition".to_owned())?;
    for (parent_local, originals) in parent_token_groups.iter().enumerate() {
        let mut groups = BTreeMap::<Vec<u32>, Vec<u32>>::new();
        for &original in originals {
            if !selected_original
                .get(original as usize)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let child_tuple = components[1..]
                .iter()
                .map(|component| {
                    component
                        .constraint
                        .original_token_internal_at(original)
                        .unwrap_or(u32::MAX)
                })
                .collect::<Vec<_>>();
            groups.entry(child_tuple).or_default().push(original);
        }
        for (child_tuple, originals) in groups {
            let mut tuple = Vec::with_capacity(components.len());
            tuple.push(parent_local as u32);
            tuple.extend(child_tuple);
            add_token_class(originals, tuple)?;
        }
    }

    let mut parent_unmapped = BTreeMap::<Vec<u32>, Vec<u32>>::new();
    for &original in original_token_ids {
        let parent_local = parent.original_token_internal_at(original).unwrap_or(u32::MAX);
        if parent_local != u32::MAX {
            continue;
        }
        let child_tuple = components[1..]
            .iter()
            .map(|component| {
                component
                    .constraint
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX)
            })
            .collect::<Vec<_>>();
        parent_unmapped.entry(child_tuple).or_default().push(original);
    }
    for (child_tuple, originals) in parent_unmapped {
        let mut tuple = Vec::with_capacity(components.len());
        tuple.push(u32::MAX);
        tuple.extend(child_tuple);
        add_token_class(originals, tuple)?;
    }

    Ok((
        ManyToOneIdMap {
            original_to_internal: token_to_global,
            internal_to_originals: global_to_tokens,
            representative_original_ids: token_representatives,
        },
        local_to_global_tokens,
        false,
    ))
}

pub(super) fn build_direct_component_coordinate_maps(
    components: &[ParserDwaComponent<'_>],
    merged_tokenizer_state_count: usize,
    original_token_ids: &[u32],
) -> Result<(InternalIdMap, Vec<DirectComponentCoordinateMaps>), String> {
    let total_started_at = Instant::now();
    let state_started_at = Instant::now();
    let state_coordinates =
        build_direct_component_state_coordinates(components, merged_tokenizer_state_count)?;
    let state_ms = state_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_started_at = Instant::now();
    let (vocab_tokens, local_to_global_tokens, _token_coordinate_is_singleton) =
        build_direct_component_token_coordinates(components, original_token_ids)?;
    let token_ms = token_started_at.elapsed().as_secs_f64() * 1000.0;
    let component_maps = state_coordinates
        .local_to_global_tsids
        .into_iter()
        .zip(local_to_global_tokens)
        .map(|(local_to_global_tsids, local_to_global_tokens)| {
            DirectComponentCoordinateMaps {
                local_to_global_tsids,
                local_to_global_tokens,
            }
        })
        .collect::<Vec<_>>();
    let id_map = InternalIdMap {
        tokenizer_states: state_coordinates.tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_component_coordinates] components={} states={} tsids={} tokens={} classes={} state_ms={state_ms:.3} token_ms={token_ms:.3} total_ms={:.3}",
            components.len(),
            merged_tokenizer_state_count,
            id_map.num_tsids(),
            original_token_ids.len(),
            id_map.num_internal_tokens(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((id_map, component_maps))
}

pub(super) fn concrete_local_parser_states_for_label(
    local_state: u32,
    relation: &[Vec<u32>],
    parser_state_domain_labels: &[i32],
) -> Result<Vec<u32>, String> {
    if (local_state as usize) < relation.len() {
        return Ok(vec![local_state]);
    }
    if parser_state_domain_labels.is_empty() {
        return Err(format!("parser-state relation omits local state {local_state}"));
    }
    let synthetic = i32::try_from(local_state)
        .map_err(|_| format!("parser-state label {local_state} exceeds i32 range"))?;
    let states = parser_state_domain_labels
        .iter()
        .enumerate()
        .filter_map(|(state, &label)| (label == synthetic).then_some(state as u32))
        .collect::<Vec<_>>();
    if states.is_empty() {
        return Err(format!(
            "parser-state relation omits local state {local_state} and no stored domain expands it"
        ));
    }
    Ok(states)
}

pub(super) fn mapped_labels(
    label: i32,
    relation: &[Vec<u32>],
    parser_state_domain_labels: &[i32],
) -> Result<Vec<i32>, String> {
    if label == DEFAULT_LABEL {
        return Err("default labels must be materialized before parser-state transport".into());
    }
    let (local_state, negative) = if is_negative_label(label) {
        (negative_to_positive_label(label) as u32, true)
    } else if label >= 0 {
        (label as u32, false)
    } else {
        return Err(format!("unsupported parser-DWA label {label}"));
    };
    let local_states = concrete_local_parser_states_for_label(
        local_state,
        relation,
        parser_state_domain_labels,
    )?;
    let mut mapped = Vec::new();
    for local_state in local_states {
        let targets = relation
            .get(local_state as usize)
            .ok_or_else(|| format!("parser-state relation omits local state {local_state}"))?;
        if targets.is_empty() {
            return Err(format!("parser-state relation maps local state {local_state} nowhere"));
        }
        mapped.extend(targets.iter().copied());
    }
    mapped.sort_unstable();
    mapped.dedup();
    Ok(mapped
        .into_iter()
        .map(|state| {
            if negative {
                encode_negative_label(state)
            } else {
                encode_positive_label(state)
            }
        })
        .collect())
}

pub(super) fn add_transition_for_mapped_label(
    nwa: &mut NWA,
    from: u32,
    local_label: i32,
    target: u32,
    weight: &Weight,
    relation: &[Vec<u32>],
    parser_state_domain_labels: &[i32],
) -> Result<(), String> {
    for label in mapped_labels(local_label, relation, parser_state_domain_labels)? {
        nwa.add_transition(from, label, target, weight.clone());
    }
    Ok(())
}

pub(super) fn materialized_top_acceptance(constraint: &Constraint) -> BTreeMap<i32, Weight> {
    let mut result = BTreeMap::new();
    let default_combined = constraint.parser_top_accept.get(&DEFAULT_LABEL);
    let default_parts = constraint.parser_top_accept_parts.get(&DEFAULT_LABEL);
    for parser_state in 0..constraint.table.num_states {
        let label = encode_positive_label(parser_state);
        let mut parts = Vec::<Weight>::new();
        if let Some(weight) = constraint
            .parser_top_accept
            .get(&label)
            .or_else(|| {
                constraint
                    .parser_state_domain_label(parser_state)
                    .and_then(|domain| constraint.parser_top_accept.get(&domain))
            })
            .or(default_combined)
        {
            parts.push(weight.clone());
        }
        if let Some(weights) = constraint
            .parser_top_accept_parts
            .get(&label)
            .or_else(|| {
                constraint
                    .parser_state_domain_label(parser_state)
                    .and_then(|domain| constraint.parser_top_accept_parts.get(&domain))
            })
            .or(default_parts)
        {
            parts.extend(weights.iter().cloned());
        }
        if let Some(row) = constraint.table.advance.get(parser_state as usize) {
            for terminal in row.iter_ones() {
                if let Some(weight) = constraint
                    .direct_regular_l1_complete_by_terminal
                    .get(&(terminal as u32))
                {
                    parts.push(weight.clone());
                }
            }
        }
        if !parts.is_empty() {
            let weight = Weight::union_all(parts.iter());
            if !weight.is_empty() {
                result.insert(label, weight);
            }
        }
    }
    result
}

pub(super) fn add_component_parser_state_transitions(
    nwa: &mut NWA,
    source_state: u32,
    source: &DWAState,
    parser_state_relation: &[Vec<u32>],
    parser_state_domain_labels: &[i32],
    num_local_parser_states: u32,
    default_domain: Option<&ParserDefaultDomain>,
) -> Result<(), String> {
    let mut explicit_positive = BTreeSet::new();
    for &label in source.transitions.keys() {
        if label < 0 || label == DEFAULT_LABEL {
            continue;
        }
        for local_state in concrete_local_parser_states_for_label(
            label as u32,
            parser_state_relation,
            parser_state_domain_labels,
        )? {
            explicit_positive.insert(local_state);
        }
    }

    for (&label, (target, weight)) in &source.transitions {
        if label == DEFAULT_LABEL {
            continue;
        }
        if let Some(domain) = default_domain
            && let Some(&refined_label) = domain.nested_labels.get(&label)
        {
            // Preserve the cached inner symbolic domain for all local states
            // that remain uniquely owned after this outer composition.
            nwa.add_transition(
                source_state,
                refined_label,
                *target,
                weight.clone(),
            );
            // Any local member of the inner domain that is not in the exact
            // outer symbolic domain still needs its ordinary concrete edge.
            for local_state in concrete_local_parser_states_for_label(
                label as u32,
                parser_state_relation,
                parser_state_domain_labels,
            )? {
                for &mapped_state in parser_state_relation
                    .get(local_state as usize)
                    .ok_or_else(|| {
                        format!("parser-state relation omits local state {local_state}")
                    })?
                {
                    if !domain.states.contains(mapped_state as usize) {
                        nwa.add_transition(
                            source_state,
                            encode_positive_label(mapped_state),
                            *target,
                            weight.clone(),
                        );
                    }
                }
            }
            continue;
        }
        add_transition_for_mapped_label(
            nwa,
            source_state,
            label,
            *target,
            weight,
            parser_state_relation,
            parser_state_domain_labels,
        )?;
    }

    let Some((target, weight)) = source.transitions.get(&DEFAULT_LABEL) else {
        return Ok(());
    };
    if let Some(domain) = default_domain {
        // One refined symbolic label denotes each exact nested-domain class,
        // plus an optional base class for states with no cached inner domain.
        // If the source row already has an explicit nested-domain transition,
        // that transition wins; otherwise the source DEFAULT is installed on
        // the refined nested label. This compiles the lookup chain
        // `concrete -> inner domain -> DEFAULT` into a single outer fallback.
        if domain.base_has_states {
            nwa.add_transition(source_state, domain.label, *target, weight.clone());
        }
        for (&source_domain, &refined_label) in &domain.nested_labels {
            if !source.transitions.contains_key(&source_domain) {
                nwa.add_transition(
                    source_state,
                    refined_label,
                    *target,
                    weight.clone(),
                );
            }
        }
    }
    for local_state in 0..num_local_parser_states {
        if explicit_positive.contains(&local_state) {
            continue;
        }
        let targets = parser_state_relation
            .get(local_state as usize)
            .ok_or_else(|| format!("parser-state relation omits local state {local_state}"))?;
        for &mapped_state in targets {
            if default_domain
                .is_some_and(|domain| domain.states.contains(mapped_state as usize))
            {
                continue;
            }
            nwa.add_transition(
                source_state,
                encode_positive_label(mapped_state),
                *target,
                weight.clone(),
            );
        }
    }
    Ok(())
}

pub(super) fn component_parser_nwa(
    component: &ParserDwaComponent<'_>,
    default_domain: Option<&ParserDefaultDomain>,
) -> Result<NWA, String> {
    let constraint = component.constraint;
    if component.parser_state_relation.len() != constraint.table.num_states as usize {
        return Err(format!(
            "parser-state relation has {} rows for a {}-state component table",
            component.parser_state_relation.len(),
            constraint.table.num_states,
        ));
    }

    let prepared_source;
    let source = if std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_HASHCONS_PREP").is_some()
        && constraint.parser_dwa.num_states() >= 10_000
    {
        let started_at = Instant::now();
        let before_states = constraint.parser_dwa.num_states();
        let before_transitions = constraint.parser_dwa.num_transitions();
        prepared_source = reverse_hashcons_owned(constraint.parser_dwa.clone());
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_component_hashcons_prep] before_states={} before_transitions={} after_states={} after_transitions={} ms={:.3}",
                before_states,
                before_transitions,
                prepared_source.num_states(),
                prepared_source.num_transitions(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        &prepared_source
    } else {
        &constraint.parser_dwa
    };
    let mut nwa = NWA::new(0, 0);
    for _ in source.states() {
        nwa.add_state();
    }
    nwa.set_start_states(vec![source.start_state()]);

    for (state_id, state) in source.states().iter().enumerate() {
        if let Some(final_weight) = &state.final_weight {
            nwa.set_final_weight(state_id as u32, final_weight.clone());
        }
        add_component_parser_state_transitions(
            &mut nwa,
            state_id as u32,
            state,
            component.parser_state_relation,
            &constraint.parser_state_domain_labels,
            constraint.table.num_states,
            default_domain,
        )?;
    }

    let top_accept = materialized_top_acceptance(constraint);
    if std::env::var_os("GLRMASK_EXPERIMENT_EXTERNALIZE_COMPONENT_TOP_ACCEPT").is_none()
        && !top_accept.is_empty()
    {
        let start = nwa.add_state();
        let final_state = nwa.add_state();
        nwa.set_final_weight(final_state, Weight::all());
        for (label, weight) in top_accept {
            add_transition_for_mapped_label(
                &mut nwa,
                start,
                label,
                final_state,
                &weight,
                component.parser_state_relation,
                &constraint.parser_state_domain_labels,
            )?;
        }
        nwa.start_states_mut().push(start);
    }
    if std::env::var_os("GLRMASK_EXPERIMENT_EXTERNALIZE_COMPONENT_TOP_ACCEPT").is_none()
        && std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_SCOPED_IGNORE_TOP_ACCEPT").is_some()
        && let (Some(table), Some(local_ignore)) =
            (component.composed_table, constraint.ignore_terminal)
        && let Some(ignore_possible) = constraint.possible_matches.get(&local_ignore)
        && let Some(parser_empty) = constraint
            .parser_dwa
            .states()
            .get(constraint.parser_dwa.start_state() as usize)
            .and_then(|state| state.final_weight.as_ref())
    {
        let identity_weight = parser_empty.intersection(ignore_possible);
        let global_ignore = component.terminal_offset + local_ignore;
        let identity_states = table
            .action
            .iter()
            .enumerate()
            .filter_map(|(state, row)| {
                let action = row.get(&global_ignore)?;
                let stack_neutral = matches!(action, Action::Skip | Action::Shift(_, true))
                    || matches!(action, Action::ReplaceShifts(_))
                    || matches!(action, Action::StackShifts(shifts) if shifts.iter().all(|shift| shift.pop == 1 && shift.pushes.len() == 1))
                    || matches!(action, Action::Split { shift: Some((_, true)), reduces, accept: false } if reduces.is_empty());
                stack_neutral.then_some(state as u32)
            })
            .collect::<Vec<_>>();
        if !identity_states.is_empty() && !identity_weight.is_empty() {
            let start = nwa.add_state();
            let final_state = nwa.add_state();
            nwa.set_final_weight(final_state, Weight::all());
            for state in identity_states {
                nwa.add_transition(
                    start,
                    encode_positive_label(state),
                    final_state,
                    identity_weight.clone(),
                );
            }
            nwa.start_states_mut().push(start);
        }
    }
    if compose_profile_enabled() {
        let stored_domain_counts = constraint
            .parser_state_domain_labels
            .iter()
            .copied()
            .filter(|&label| label != NO_PARSER_DOMAIN_LABEL)
            .fold(BTreeMap::<i32, usize>::new(), |mut counts, label| {
                *counts.entry(label).or_default() += 1;
                counts
            });
        eprintln!(
            "[glrmask/profile][constraint_component_parser_transport_shape] source_states={} source_transitions={} source_default_states={} stored_domain_states={} stored_domains={:?} mapped_states={} mapped_transitions={} symbolic_default={} relation_singleton={}",
            source.num_states(),
            source.num_transitions(),
            source
                .states()
                .iter()
                .filter(|state| state.transitions.contains_key(&DEFAULT_LABEL))
                .count(),
            stored_domain_counts.values().sum::<usize>(),
            stored_domain_counts,
            nwa.num_states(),
            nwa.num_transitions(),
            default_domain.is_some(),
            component.parser_state_relation.iter().all(|targets| targets.len() == 1),
        );
    }
    Ok(nwa)
}

/// Standalone component parser DWAs erase their one global ignore terminal
/// before parser-state interpretation. Consequently an ignore-only model token
/// appears as an unqualified empty-word final weight at the parser-DWA start.
/// That is correct for a standalone/global-ignore constraint, but would leak a
/// child ignore token into parent states after scoped linking. Remove only that
/// identity branch; boundary repair reintroduces it through the scoped `Skip`
/// actions in the explicit composed table. Other terminalizations of the same
/// model token remain intact.
pub(super) fn strip_unscoped_ignore_identity(
    automaton: &mut NWA,
    ignore_possible_matches: Option<&Weight>,
) {
    let Some(ignore_weight) = ignore_possible_matches else {
        return;
    };
    let starts = automaton.start_states().to_vec();
    for start in starts {
        let Some(state) = automaton.states_mut().get_mut(start as usize) else {
            continue;
        };
        let Some(final_weight) = state.final_weight.take() else {
            continue;
        };
        let retained = final_weight.difference(ignore_weight);
        if !retained.is_empty() {
            state.final_weight = Some(retained);
        }
    }
}
