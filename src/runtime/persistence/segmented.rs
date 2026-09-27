//! Exact materialized and recursive component-runtime artifacts.

use crate::automata::lexer::Lexer;
use rayon::prelude::*;
use super::{Arc, Constraint, Cow, Deserialize, Serialize, Weight};

#[derive(Serialize)]
pub(super) struct BoundaryTrieNodeArtifactRef<'a> {
    pub(super) children: &'a [(u32, u32)],
    pub(super) outputs: &'a [u32],
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) struct BoundaryTrieNodeArtifact {
    pub(super) children: Vec<(u32, u32)>,
    pub(super) outputs: Vec<u32>,
}

#[derive(Serialize)]
pub(super) struct BoundaryTrieArtifactRef<'a> {
    pub(super) nodes: Vec<BoundaryTrieNodeArtifactRef<'a>>,
    pub(super) root_by_tsid: &'a [u32],
    pub(super) tokenizer_state_to_tsid: &'a [u32],
    pub(super) internal_token_to_originals: &'a [Vec<u32>],
    pub(super) symbolic_nwa: Option<&'a crate::runtime::artifact::BoundaryTerminalNwa>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) struct BoundaryTrieArtifact {
    pub(super) nodes: Vec<BoundaryTrieNodeArtifact>,
    pub(super) root_by_tsid: Vec<u32>,
    pub(super) tokenizer_state_to_tsid: Vec<u32>,
    pub(super) internal_token_to_originals: Vec<Vec<u32>>,
    pub(super) symbolic_nwa: Option<crate::runtime::artifact::BoundaryTerminalNwa>,
}

#[derive(Serialize)]
pub(super) struct BoundaryParserArtifactRef<'a> {
    pub(super) parser_dwa: Cow<'a, crate::automata::weighted_u32::dwa::DWA>,
    pub(super) uses_composed_tsid_coordinate: bool,
    pub(super) tokenizer_state_to_tsid: &'a [u32],
    pub(super) internal_token_to_originals: &'a [Vec<u32>],
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) struct BoundaryParserArtifact {
    pub(super) parser_dwa: crate::automata::weighted_u32::dwa::DWA,
    pub(super) uses_composed_tsid_coordinate: bool,
    pub(super) tokenizer_state_to_tsid: Vec<u32>,
    pub(super) internal_token_to_originals: Vec<Vec<u32>>,
}

#[derive(Serialize)]
pub(super) enum BoundaryScopeArtifactRef<'a> {
    Global,
    Component {
        start_component: u32,
        start_parser_states: &'a crate::ds::bitset::BitSet,
        accepts_empty_stack: bool,
    },
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) enum BoundaryScopeArtifact {
    Global,
    Component {
        start_component: u32,
        start_parser_states: crate::ds::bitset::BitSet,
        accepts_empty_stack: bool,
    },
}

#[derive(Serialize)]
pub(super) enum BoundaryBackendArtifactRef<'a> {
    StaticParser(BoundaryParserArtifactRef<'a>),
    DynamicTerminalTrie(BoundaryTrieArtifactRef<'a>),
    DynamicDirect,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) enum BoundaryBackendArtifact {
    StaticParser(BoundaryParserArtifact),
    DynamicTerminalTrie(BoundaryTrieArtifact),
    DynamicDirect,
}

#[derive(Serialize)]
pub(super) struct BoundaryShardArtifactRef<'a> {
    pub(super) scope: BoundaryScopeArtifactRef<'a>,
    pub(super) candidate_tokens: Option<&'a [u32]>,
    pub(super) backend: BoundaryBackendArtifactRef<'a>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) struct BoundaryShardArtifact {
    pub(super) scope: BoundaryScopeArtifact,
    pub(super) candidate_tokens: Option<Vec<u32>>,
    pub(super) backend: BoundaryBackendArtifact,
}

#[derive(Serialize)]
pub(super) struct MaterializedComponentArtifactRef<'a> {
    pub(super) constraint_artifact: Vec<u8>,
    pub(super) tokenizer_state_offset: u32,
    pub(super) terminal_offset: u32,
    pub(super) global_terminal_aliases: &'a [(u32, u32)],
    pub(super) local_tsid_to_global_tsids: &'a [Vec<u32>],
    pub(super) root_entry_terminals: crate::ds::bitset::BitSet,
    pub(super) root_disallowed_terminal: Option<u32>,
    pub(super) global_to_local_parser_state: &'a [u32],
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) struct MaterializedComponentArtifact {
    pub(super) constraint_artifact: Vec<u8>,
    pub(super) tokenizer_state_offset: u32,
    pub(super) terminal_offset: u32,
    pub(super) global_terminal_aliases: Vec<(u32, u32)>,
    pub(super) local_tsid_to_global_tsids: Vec<Vec<u32>>,
    pub(super) root_entry_terminals: crate::ds::bitset::BitSet,
    pub(super) root_disallowed_terminal: Option<u32>,
    pub(super) global_to_local_parser_state: Vec<u32>,
}

#[derive(Serialize)]
pub(super) struct ParserLinkArtifactRef {
    pub(super) parent_component: u32,
    pub(super) slot_terminal: u32,
    pub(super) child_component: u32,
    pub(super) child_start: u32,
    pub(super) return_pop: u32,
    pub(super) child_start_nullable: bool,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) struct ParserLinkArtifact {
    pub(super) parent_component: u32,
    pub(super) slot_terminal: u32,
    pub(super) child_component: u32,
    pub(super) child_start: u32,
    pub(super) return_pop: u32,
    pub(super) child_start_nullable: bool,
}

#[derive(Serialize)]
pub(super) struct MaterializedRuntimeArtifactRef<'a> {
    pub(super) materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    pub(super) materialized_static_parser_state_domain_labels: Vec<i32>,
    pub(super) components: Vec<MaterializedComponentArtifactRef<'a>>,
    pub(super) segmented_parser_links: Vec<ParserLinkArtifactRef>,
    pub(super) segmented_parser_state_offsets: &'a [u32],
    pub(super) segmented_mask_authoritative: bool,
    pub(super) segmented_component_union_root_dispatch: &'a [u32],
    pub(super) boundary_shards: Vec<BoundaryShardArtifactRef<'a>>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub(super) struct MaterializedRuntimeArtifact {
    pub(super) materialized_static_component_parser: Option<crate::automata::weighted_u32::dwa::DWA>,
    pub(super) materialized_static_parser_state_domain_labels: Vec<i32>,
    pub(super) components: Vec<MaterializedComponentArtifact>,
    pub(super) segmented_parser_links: Vec<ParserLinkArtifact>,
    pub(super) segmented_parser_state_offsets: Vec<u32>,
    pub(super) segmented_mask_authoritative: bool,
    pub(super) segmented_component_union_root_dispatch: Vec<u32>,
    pub(super) boundary_shards: Vec<BoundaryShardArtifact>,
}

#[derive(Serialize)]
pub(super) struct RecursiveComponentArtifactRef<'a> {
    pub(super) constraint_artifact: Vec<u8>,
    pub(super) tokenizer_state_offset: u32,
    pub(super) terminal_offset: u32,
    pub(super) global_terminal_aliases: &'a [(u32, u32)],
    pub(super) local_tsid_to_global_tsids: &'a [Vec<u32>],
    pub(super) root_entry_terminals: crate::ds::bitset::BitSet,
    pub(super) root_disallowed_terminal: Option<u32>,
}

#[derive(Deserialize)]
pub(super) struct RecursiveComponentArtifact {
    pub(super) constraint_artifact: Vec<u8>,
    pub(super) tokenizer_state_offset: u32,
    pub(super) terminal_offset: u32,
    pub(super) global_terminal_aliases: Vec<(u32, u32)>,
    pub(super) local_tsid_to_global_tsids: Vec<Vec<u32>>,
    pub(super) root_entry_terminals: crate::ds::bitset::BitSet,
    pub(super) root_disallowed_terminal: Option<u32>,
}

#[derive(Serialize)]
pub(super) struct RecursiveBoundaryParserArtifactRef<'a> {
    pub(super) parser_dwa: &'a crate::automata::weighted_u32::dwa::DWA,
    pub(super) uses_composed_tsid_coordinate: bool,
    pub(super) tokenizer_state_to_tsid: &'a [u32],
    pub(super) internal_token_to_originals: &'a [Vec<u32>],
}

#[derive(Deserialize)]
pub(super) struct RecursiveBoundaryParserArtifact {
    pub(super) parser_dwa: crate::automata::weighted_u32::dwa::DWA,
    pub(super) uses_composed_tsid_coordinate: bool,
    pub(super) tokenizer_state_to_tsid: Vec<u32>,
    pub(super) internal_token_to_originals: Vec<Vec<u32>>,
}

#[derive(Serialize)]
pub(super) enum RecursiveBoundaryBackendArtifactRef<'a> {
    StaticParser(RecursiveBoundaryParserArtifactRef<'a>),
    DynamicDirect,
}

#[derive(Deserialize)]
pub(super) enum RecursiveBoundaryBackendArtifact {
    StaticParser(RecursiveBoundaryParserArtifact),
    DynamicDirect,
}

#[derive(Serialize)]
pub(super) struct RecursiveBoundaryShardArtifactRef<'a> {
    pub(super) start_component: u32,
    pub(super) accepts_empty_stack: bool,
    pub(super) candidate_tokens: Option<&'a [u32]>,
    pub(super) backend: RecursiveBoundaryBackendArtifactRef<'a>,
}

#[derive(Deserialize)]
pub(super) struct RecursiveBoundaryShardArtifact {
    pub(super) start_component: u32,
    pub(super) accepts_empty_stack: bool,
    pub(super) candidate_tokens: Option<Vec<u32>>,
    pub(super) backend: RecursiveBoundaryBackendArtifact,
}

#[derive(Serialize)]
pub(super) struct RecursiveRuntimeArtifactRef<'a> {
    pub(super) components: Vec<RecursiveComponentArtifactRef<'a>>,
    pub(super) segmented_parser_links: Vec<ParserLinkArtifactRef>,
    pub(super) recursive_compiler_table: &'a [u8],
    pub(super) recursive_tokenizer_internal_tsids: &'a [Vec<u32>],
    pub(super) segmented_mask_authoritative: bool,
    pub(super) boundary_shards: Vec<RecursiveBoundaryShardArtifactRef<'a>>,
}

#[derive(Deserialize)]
pub(super) struct RecursiveRuntimeArtifact {
    pub(super) components: Vec<RecursiveComponentArtifact>,
    pub(super) segmented_parser_links: Vec<ParserLinkArtifact>,
    pub(super) recursive_compiler_table: Vec<u8>,
    pub(super) recursive_tokenizer_internal_tsids: Vec<Vec<u32>>,
    pub(super) segmented_mask_authoritative: bool,
    pub(super) boundary_shards: Vec<RecursiveBoundaryShardArtifact>,
}

#[derive(Serialize)]
pub(super) enum SegmentedRuntimeArtifactRef<'a> {
    Recursive(RecursiveRuntimeArtifactRef<'a>),
    Materialized(MaterializedRuntimeArtifactRef<'a>),
}

#[derive(Deserialize)]
pub(super) enum SegmentedRuntimeArtifact {
    Recursive(RecursiveRuntimeArtifact),
    Materialized(MaterializedRuntimeArtifact),
}

pub(super) fn boundary_parser_artifact_ref(
    boundary: &crate::runtime::artifact::SegmentedBoundaryParser,
) -> BoundaryParserArtifactRef<'_> {
    let parser_dwa = if let Some(compact) = boundary.compact_parser_dwa.as_ref() {
        Cow::Owned(compact.to_generic_dwa())
    } else {
        Cow::Borrowed(&boundary.parser_dwa)
    };
    BoundaryParserArtifactRef {
        parser_dwa,
        uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
        tokenizer_state_to_tsid: &boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: &boundary.internal_token_to_originals,
    }
}

pub(super) fn boundary_trie_artifact_ref(
    boundary: &crate::runtime::artifact::SegmentedBoundaryTerminalTrie,
) -> BoundaryTrieArtifactRef<'_> {
    BoundaryTrieArtifactRef {
        nodes: boundary
            .nodes
            .iter()
            .map(|node| BoundaryTrieNodeArtifactRef {
                children: &node.children,
                outputs: &node.outputs,
            })
            .collect(),
        root_by_tsid: &boundary.root_by_tsid,
        tokenizer_state_to_tsid: &boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: &boundary.internal_token_to_originals,
        symbolic_nwa: boundary.symbolic_nwa.as_ref(),
    }
}

pub(super) fn serialized_component_root_entry_terminals(
    component: &crate::runtime::SegmentedParserComponent,
    global_terminal_count: usize,
) -> crate::ds::bitset::BitSet {
    let mut terminals = crate::ds::bitset::BitSet::new(global_terminal_count);
    let end = component
        .terminal_offset
        .saturating_add(component.constraint.table.num_terminals)
        .min(global_terminal_count as u32);
    for terminal in component.terminal_offset..end {
        terminals.set(terminal as usize);
    }
    terminals
}

pub(super) fn materialized_runtime_artifact_ref(
    constraint: &Constraint,
) -> Option<MaterializedRuntimeArtifactRef<'_>> {
    let overlay = constraint.static_dynamic_overlay.as_ref()?;
    if overlay.segmented_parser_components.is_empty()
        && overlay.segmented_boundary_shards.is_empty()
        && overlay.segmented_boundary_parser.is_none()
        && overlay.segmented_boundary_terminal_trie.is_none()
    {
        return None;
    }
    let components = overlay
        .segmented_parser_components
        .iter()
        .map(|component| MaterializedComponentArtifactRef {
            // Segmented components are embeddable bodies, not generation
            // roots. They may intentionally retain linker slots, so never
            // wrap them in the closed-root policy/vocabulary envelope.
            constraint_artifact: component.constraint.save_body(),
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: &component.global_terminal_aliases,
            local_tsid_to_global_tsids: &component.local_tsid_to_global_tsids,
            root_entry_terminals: serialized_component_root_entry_terminals(
                component,
                constraint.table.num_terminals as usize,
            ),
            root_disallowed_terminal: component.root_disallowed_terminal,
            global_to_local_parser_state: &component.global_to_local_parser_state,
        })
        .collect();

    let boundary_shards = if !overlay.segmented_boundary_shards.is_empty() {
        overlay
            .segmented_boundary_shards
            .iter()
            .map(|shard| BoundaryShardArtifactRef {
                scope: BoundaryScopeArtifactRef::Component {
                    start_component: shard.start_component,
                    start_parser_states: &shard.start_parser_states,
                    accepts_empty_stack: shard.accepts_empty_stack,
                },
                candidate_tokens: shard.candidate_tokens.as_deref(),
                backend: match &shard.backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                        BoundaryBackendArtifactRef::StaticParser(
                            boundary_parser_artifact_ref(boundary),
                        )
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(boundary) => {
                        BoundaryBackendArtifactRef::DynamicTerminalTrie(
                            boundary_trie_artifact_ref(boundary),
                        )
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        BoundaryBackendArtifactRef::DynamicDirect
                    }
                },
            })
            .collect()
    } else {
        let mut shards = Vec::new();
        if let Some(boundary) = overlay.segmented_boundary_parser.as_deref() {
            shards.push(BoundaryShardArtifactRef {
                scope: BoundaryScopeArtifactRef::Global,
                candidate_tokens: None,
                backend: BoundaryBackendArtifactRef::StaticParser(
                    boundary_parser_artifact_ref(boundary),
                ),
            });
        }
        if let Some(boundary) = overlay.segmented_boundary_terminal_trie.as_deref() {
            shards.push(BoundaryShardArtifactRef {
                scope: BoundaryScopeArtifactRef::Global,
                candidate_tokens: None,
                backend: BoundaryBackendArtifactRef::DynamicTerminalTrie(
                    boundary_trie_artifact_ref(boundary),
                ),
            });
        }
        shards
    };

    let segmented_parser_links = overlay
        .segmented_parser_links
        .iter()
        .map(|link| ParserLinkArtifactRef {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect();

    Some(MaterializedRuntimeArtifactRef {
        materialized_static_component_parser: None,
        materialized_static_parser_state_domain_labels: Vec::new(),
        components,
        segmented_parser_links,
        segmented_parser_state_offsets: &overlay.segmented_parser_state_offsets,
        segmented_mask_authoritative: overlay.segmented_mask_authoritative,
        segmented_component_union_root_dispatch: &overlay.segmented_component_union_root_dispatch,
        boundary_shards,
    })
}

pub(super) fn segmented_runtime_artifact_ref(
    constraint: &Constraint,
) -> Option<SegmentedRuntimeArtifactRef<'_>> {
    if !constraint.uses_compact_segmented_parser_runtime() {
        return materialized_runtime_artifact_ref(constraint)
            .map(SegmentedRuntimeArtifactRef::Materialized);
    }
    let layout = constraint
        .recursive_parser_layout()
        .expect("validated recursive runtime must derive its parser layout before serialization")
        .expect("provider-native segmented runtime must have a recursive parser layout");
    let overlay = constraint.static_dynamic_overlay.as_ref()?;
    let recursive_compiler_table = overlay
        .recursive_compiler_table
        .get()
        .expect("recursive runtime must retain its compiler table blob");
    if overlay.recursive_tokenizer_internal_tsids.get().is_none() {
        let compatibility_omission = recursive_compiler_table.is_empty()
            && !overlay.segmented_parser_components.is_empty()
            && overlay.segmented_parser_components.iter().all(|component| {
                component.boundary.as_ref().is_some_and(|shard| {
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                    )
                })
            });
        if !compatibility_omission {
            panic!("recursive runtime must retain its scoped tokenizer TSID relation");
        }
        // Wire compatibility only: old v27 readers require one row per scoped
        // tokenizer state. The live DynamicDirect coordinator never consumes
        // this quotient, so materialize the historical all-zero relation only
        // when serialization is explicitly requested.
        overlay
            .recursive_tokenizer_internal_tsids
            .set(Arc::new(vec![vec![0u32]; layout.total_tokenizer_states as usize]))
            .expect("recursive serializer compatibility relation initialized twice");
    }
    let recursive_tokenizer_internal_tsids = overlay
        .recursive_tokenizer_internal_tsids
        .get()
        .expect("recursive runtime must retain its scoped tokenizer TSID relation");
    let components = overlay
        .segmented_parser_components
        .iter()
        .map(|component| RecursiveComponentArtifactRef {
            // Recursive segmented components are likewise internal bodies.
            // The outer root/module artifact carries exact-vocabulary identity
            // and rebinds it recursively after load.
            constraint_artifact: component.constraint.save_body(),
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: &component.global_terminal_aliases,
            local_tsid_to_global_tsids: &component.local_tsid_to_global_tsids,
            root_entry_terminals: serialized_component_root_entry_terminals(
                component,
                constraint.table.num_terminals as usize,
            ),
            root_disallowed_terminal: component.root_disallowed_terminal,
        })
        .collect();
    let segmented_parser_links = overlay
        .segmented_parser_links
        .iter()
        .map(|link| ParserLinkArtifactRef {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect();
    let boundary_shards = overlay
        .segmented_boundary_shards
        .iter()
        .map(|shard| {
            let backend = match &shard.backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                    let parser_dwa = boundary.recursive_parser_dwa.as_ref().expect(
                        "recursive static boundary must carry its recursive parser-coordinate DWA",
                    );
                    RecursiveBoundaryBackendArtifactRef::StaticParser(
                        RecursiveBoundaryParserArtifactRef {
                            parser_dwa,
                            uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
                            tokenizer_state_to_tsid: &boundary.tokenizer_state_to_tsid,
                            internal_token_to_originals: &boundary.internal_token_to_originals,
                        },
                    )
                }
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                    RecursiveBoundaryBackendArtifactRef::DynamicDirect
                }
                crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(_) => {
                    unreachable!(
                        "recursive provider-native runtime cannot carry a terminal-trie boundary"
                    )
                }
            };
            RecursiveBoundaryShardArtifactRef {
                start_component: shard.start_component,
                accepts_empty_stack: shard.accepts_empty_stack,
                candidate_tokens: shard.candidate_tokens.as_deref(),
                backend,
            }
        })
        .collect();
    Some(SegmentedRuntimeArtifactRef::Recursive(
        RecursiveRuntimeArtifactRef {
            components,
            segmented_parser_links,
            recursive_compiler_table: recursive_compiler_table.as_ref(),
            recursive_tokenizer_internal_tsids:
                recursive_tokenizer_internal_tsids.as_slice(),
            segmented_mask_authoritative: overlay.segmented_mask_authoritative,
            boundary_shards,
        },
    ))
}

pub(super) fn restore_boundary_trie(
    boundary: BoundaryTrieArtifact,
    global_terminal_count: usize,
) -> crate::Result<crate::runtime::artifact::SegmentedBoundaryTerminalTrie> {
    fn weight_within_boundary_domain(weight: &Weight, tsids: usize, tokens: usize) -> bool {
        if weight.is_empty() || weight.is_full() {
            return true;
        }
        weight.range_entries().all(|(_, end_tsid, token_set)| {
            end_tsid < tsids as u32
                && token_set
                    .ranges()
                    .all(|range| *range.end() < tokens as u32)
        })
    }

    let node_count = boundary.nodes.len();
    if boundary
        .root_by_tsid
        .iter()
        .any(|&root| root != u32::MAX && root as usize >= node_count)
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary terminal trie references an invalid root node".to_owned(),
        ));
    }
    if boundary
        .tokenizer_state_to_tsid
        .iter()
        .any(|&tsid| tsid != u32::MAX && tsid as usize >= boundary.root_by_tsid.len())
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary terminal trie references an invalid TSID".to_owned(),
        ));
    }
    let internal_token_count = boundary.internal_token_to_originals.len();
    for (node_index, node) in boundary.nodes.iter().enumerate() {
        if node
            .children
            .iter()
            .any(|&(_, child)| child as usize >= node_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized boundary terminal trie node {node_index} references an invalid child"
            )));
        }
        if node
            .outputs
            .iter()
            .any(|&token| token as usize >= internal_token_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized boundary terminal trie node {node_index} references an invalid token class"
            )));
        }
    }

    if let Some(symbolic_nwa) = boundary.symbolic_nwa.as_ref() {
        let symbolic_node_count = symbolic_nwa.nodes.len();
        if symbolic_nwa.topological_order.len() != symbolic_node_count {
            return Err(crate::GlrMaskError::Serialization(
                "serialized boundary terminal NWA has an incomplete topological order".to_owned(),
            ));
        }
        let mut position = vec![usize::MAX; symbolic_node_count];
        for (index, &state) in symbolic_nwa.topological_order.iter().enumerate() {
            let Some(slot) = position.get_mut(state as usize) else {
                return Err(crate::GlrMaskError::Serialization(
                    "serialized boundary terminal NWA topological order references an invalid state"
                        .to_owned(),
                ));
            };
            if *slot != usize::MAX {
                return Err(crate::GlrMaskError::Serialization(
                    "serialized boundary terminal NWA topological order contains a duplicate state"
                        .to_owned(),
                ));
            }
            *slot = index;
        }
        if symbolic_nwa
            .start_states
            .iter()
            .any(|&state| state as usize >= symbolic_node_count)
        {
            return Err(crate::GlrMaskError::Serialization(
                "serialized boundary terminal NWA references an invalid start state".to_owned(),
            ));
        }
        for (source, node) in symbolic_nwa.nodes.iter().enumerate() {
            if node.final_weight.as_ref().is_some_and(|weight| {
                !weight_within_boundary_domain(
                    weight,
                    boundary.root_by_tsid.len(),
                    internal_token_count,
                )
            }) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized boundary terminal NWA state {source} has an out-of-domain final weight"
                )));
            }
            for transition in &node.transitions {
                if transition.terminal as usize >= global_terminal_count
                    || transition.target as usize >= symbolic_node_count
                    || position[source] >= position[transition.target as usize]
                    || !weight_within_boundary_domain(
                        &transition.weight,
                        boundary.root_by_tsid.len(),
                        internal_token_count,
                    )
                {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "serialized boundary terminal NWA state {source} has an invalid labeled transition"
                    )));
                }
            }
            for (target, weight) in &node.epsilons {
                if *target as usize >= symbolic_node_count
                    || position[source] >= position[*target as usize]
                    || !weight_within_boundary_domain(
                        weight,
                        boundary.root_by_tsid.len(),
                        internal_token_count,
                    )
                {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "serialized boundary terminal NWA state {source} has an invalid epsilon transition"
                    )));
                }
            }
        }
    }

    Ok(crate::runtime::artifact::SegmentedBoundaryTerminalTrie {
        nodes: boundary
            .nodes
            .into_iter()
            .map(|node| crate::runtime::artifact::BoundaryTerminalTrieNode {
                children: node.children,
                outputs: node.outputs,
            })
            .collect(),
        root_by_tsid: boundary.root_by_tsid,
        tokenizer_state_to_tsid: boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: boundary.internal_token_to_originals,
        symbolic_nwa: boundary.symbolic_nwa,
    })
}

pub(super) fn restore_boundary_parser(
    constraint: &Constraint,
    boundary: BoundaryParserArtifact,
) -> crate::Result<crate::runtime::artifact::SegmentedBoundaryParser> {
    let global_tokenizer_states = constraint.tokenizer.num_states();
    let tsid_count = if boundary.uses_composed_tsid_coordinate {
        if !boundary.tokenizer_state_to_tsid.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "composed-coordinate boundary shard redundantly stores a private tokenizer-state map"
                    .to_owned(),
            ));
        }
        constraint.internal_tsid_count()
    } else {
        if boundary.tokenizer_state_to_tsid.len() != global_tokenizer_states as usize {
            return Err(crate::GlrMaskError::Serialization(format!(
                "private-coordinate boundary shard has {} tokenizer-state entries for {global_tokenizer_states} outer states",
                boundary.tokenizer_state_to_tsid.len(),
            )));
        }
        boundary
            .tokenizer_state_to_tsid
            .iter()
            .copied()
            .filter(|&tsid| tsid != u32::MAX)
            .max()
            .map_or(0, |tsid| tsid as usize + 1)
    };
    let token_count = boundary.internal_token_to_originals.len();
    let state_count = boundary.parser_dwa.num_states() as usize;
    if boundary.parser_dwa.start_state() as usize >= state_count {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary parser has an invalid start state".to_owned(),
        ));
    }
    let weight_in_domain = |weight: &Weight| {
        weight.is_empty()
            || weight.is_full()
            || weight.raw_range_values().all(|(range, tokens)| {
                *range.end() < tsid_count as u32
                    && tokens
                        .ranges()
                        .all(|token_range| *token_range.end() < token_count as u32)
            })
    };
    for (state_index, state) in boundary.parser_dwa.states().iter().enumerate() {
        if state
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight_in_domain(weight))
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized boundary parser state {state_index} has an out-of-domain final weight"
            )));
        }
        for (target, weight) in state.transitions.values() {
            if *target as usize >= state_count || !weight_in_domain(weight) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized boundary parser state {state_index} has an invalid transition"
                )));
            }
        }
    }
    if !boundary.uses_composed_tsid_coordinate
        && boundary
            .tokenizer_state_to_tsid
            .iter()
            .any(|&tsid| tsid != u32::MAX && tsid as usize >= tsid_count)
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized boundary parser tokenizer-state map references an invalid TSID".to_owned(),
        ));
    }
    Ok(crate::runtime::artifact::SegmentedBoundaryParser {
        parser_dwa: boundary.parser_dwa,
        compact_parser_dwa: None,
        recursive_parser_dwa: None,
        uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
        tokenizer_state_to_tsid: boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: boundary.internal_token_to_originals,
    })
}

pub(super) fn restore_recursive_boundary_parser(
    constraint: &Constraint,
    boundary: RecursiveBoundaryParserArtifact,
    recursive_parser_state_count: u32,
    recursive_tokenizer_state_count: u32,
) -> crate::Result<crate::runtime::artifact::SegmentedBoundaryParser> {
    let tsid_count = if boundary.uses_composed_tsid_coordinate {
        if !boundary.tokenizer_state_to_tsid.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "recursive composed-coordinate boundary shard redundantly stores a private tokenizer-state map"
                    .to_owned(),
            ));
        }
        constraint.internal_tsid_count()
    } else {
        // v27 restores compact segmented runtimes only, whose shard queries
        // index live leaf states directly. The private map therefore covers
        // the recursive leaf total — not the outer composed tokenizer, which
        // may canonicalize states the leaves keep distinct.
        if boundary.tokenizer_state_to_tsid.len() != recursive_tokenizer_state_count as usize {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive private-coordinate boundary shard has {} tokenizer-state entries for {recursive_tokenizer_state_count} leaf states",
                boundary.tokenizer_state_to_tsid.len(),
            )));
        }
        boundary
            .tokenizer_state_to_tsid
            .iter()
            .copied()
            .filter(|&tsid| tsid != u32::MAX)
            .max()
            .map_or(0, |tsid| tsid as usize + 1)
    };
    let token_count = boundary.internal_token_to_originals.len();
    let state_count = boundary.parser_dwa.num_states() as usize;
    if boundary.parser_dwa.start_state() as usize >= state_count {
        return Err(crate::GlrMaskError::Serialization(
            "serialized recursive boundary parser has an invalid start state".to_owned(),
        ));
    }
    let weight_in_domain = |weight: &Weight| {
        weight.is_empty()
            || weight.is_full()
            || weight.raw_range_values().all(|(range, tokens)| {
                *range.end() < tsid_count as u32
                    && tokens
                        .ranges()
                        .all(|token_range| *token_range.end() < token_count as u32)
            })
    };
    for (state_index, state) in boundary.parser_dwa.states().iter().enumerate() {
        if state
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight_in_domain(weight))
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized recursive boundary parser state {state_index} has an out-of-domain final weight"
            )));
        }
        for (&label, (target, weight)) in &state.transitions {
            if *target as usize >= state_count || !weight_in_domain(weight) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized recursive boundary parser state {state_index} has an invalid transition"
                )));
            }
            if label != crate::compiler::glr::labels::DEFAULT_LABEL
                && (label < 0 || label as u32 >= recursive_parser_state_count)
            {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "serialized recursive boundary parser state {state_index} references parser state {label} outside the recursive domain"
                )));
            }
        }
    }
    if !boundary.uses_composed_tsid_coordinate
        && boundary
            .tokenizer_state_to_tsid
            .iter()
            .any(|&tsid| tsid != u32::MAX && tsid as usize >= tsid_count)
    {
        return Err(crate::GlrMaskError::Serialization(
            "serialized recursive boundary parser tokenizer-state map references an invalid TSID"
                .to_owned(),
        ));
    }
    Ok(crate::runtime::artifact::SegmentedBoundaryParser {
        parser_dwa: crate::automata::weighted_u32::dwa::DWA::new(0, 0),
        compact_parser_dwa: None,
        recursive_parser_dwa: Some(boundary.parser_dwa),
        uses_composed_tsid_coordinate: boundary.uses_composed_tsid_coordinate,
        tokenizer_state_to_tsid: boundary.tokenizer_state_to_tsid,
        internal_token_to_originals: boundary.internal_token_to_originals,
    })
}

pub(super) fn restore_boundary_shards(
    constraint: &mut Constraint,
    boundary_shards: Vec<BoundaryShardArtifact>,
    allow_empty_component_start_states: bool,
) -> crate::Result<()> {
    let global_state_count = constraint.table.num_states as usize;
    let global_terminal_count = constraint.table.num_terminals as usize;
    let component_count = constraint
        .static_dynamic_overlay
        .as_ref()
        .map_or(0, |overlay| overlay.segmented_parser_components.len());
    let mut seen_components = vec![false; component_count];
    let mut global_static = None;
    let mut global_dynamic = None;
    let mut restored_shards = Vec::new();

    for (index, shard) in boundary_shards.into_iter().enumerate() {
        let backend = match shard.backend {
            BoundaryBackendArtifact::StaticParser(boundary) => {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(Arc::new(
                    restore_boundary_parser(constraint, boundary)?,
                ))
            }
            BoundaryBackendArtifact::DynamicTerminalTrie(boundary) => {
                crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(Arc::new(
                    restore_boundary_trie(boundary, global_terminal_count)?,
                ))
            }
            BoundaryBackendArtifact::DynamicDirect => {
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
            }
        };
        match shard.scope {
            BoundaryScopeArtifact::Global => {
                if shard.candidate_tokens.is_some() {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "global boundary shard {index} unexpectedly carries component trigger tokens"
                    )));
                }
                match backend {
                    crate::runtime::SegmentedBoundaryShardBackend::StaticParser(boundary) => {
                        if global_static.replace(boundary).is_some() {
                            return Err(crate::GlrMaskError::Serialization(
                                "serialized v23 runtime contains multiple global static boundary shards"
                                    .to_owned(),
                            ));
                        }
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicTerminalTrie(boundary) => {
                        if global_dynamic.replace(boundary).is_some() {
                            return Err(crate::GlrMaskError::Serialization(
                                "serialized v23 runtime contains multiple global dynamic boundary shards"
                                    .to_owned(),
                            ));
                        }
                    }
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect => {
                        return Err(crate::GlrMaskError::Serialization(
                            "serialized v23 runtime cannot use an unscoped direct-dynamic boundary shard"
                                .to_owned(),
                        ));
                    }
                }
            }
            BoundaryScopeArtifact::Component {
                start_component,
                start_parser_states,
                accepts_empty_stack,
            } => {
                let component_index = start_component as usize;
                if component_index >= component_count {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "boundary shard {index} references unknown component {start_component}"
                    )));
                }
                if seen_components[component_index] {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "serialized v23 runtime contains multiple boundary shards for component {start_component}"
                    )));
                }
                seen_components[component_index] = true;
                if start_parser_states.len() != global_state_count
                    && !(allow_empty_component_start_states && start_parser_states.len() == 0)
                {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "boundary shard {index} start-state set has length {} for {global_state_count} outer states",
                        start_parser_states.len(),
                    )));
                }
                if accepts_empty_stack && start_component != 0 {
                    return Err(crate::GlrMaskError::Serialization(format!(
                        "boundary shard {index} gives empty-stack ownership to non-root component {start_component}"
                    )));
                }
                let candidate_tokens = shard.candidate_tokens.map(|mut tokens| {
                    tokens.sort_unstable();
                    tokens.dedup();
                    Arc::<[u32]>::from(tokens)
                });
                restored_shards.push(crate::runtime::SegmentedBoundaryShard {
                    start_component,
                    start_parser_states,
                    accepts_empty_stack,
                    candidate_tokens,
                    backend,
                });
            }
        }
    }
    if !restored_shards.is_empty() && (global_static.is_some() || global_dynamic.is_some()) {
        return Err(crate::GlrMaskError::Serialization(
            "serialized v23 runtime mixes global and component-scoped boundary shards".to_owned(),
        ));
    }
    let overlay = constraint
        .static_dynamic_overlay
        .get_or_insert_with(Default::default);
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for shard in &restored_shards {
        if let Some(component) = overlay
            .segmented_parser_components
            .get_mut(shard.start_component as usize)
        {
            component.boundary = Some(shard.clone());
        }
    }
    overlay.segmented_boundary_shards = restored_shards;
    overlay.segmented_boundary_parser = global_static;
    overlay.segmented_boundary_terminal_trie = global_dynamic;
    Ok(())
}

pub(super) fn restore_materialized_runtime(
    constraint: &mut Constraint,
    runtime: MaterializedRuntimeArtifact,
) -> crate::Result<()> {
    let global_state_count = constraint.table.num_states as usize;
    let global_terminal_count = constraint.table.num_terminals as usize;
    let global_tokenizer_states = constraint.tokenizer.num_states();
    let has_static_baseline = runtime.materialized_static_component_parser.is_some();
    if let Some(parser_dwa) = runtime.materialized_static_component_parser {
        if !runtime.materialized_static_parser_state_domain_labels.is_empty()
            && runtime.materialized_static_parser_state_domain_labels.len() != global_state_count
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized static component parser has {} domain labels for {global_state_count} outer states",
                runtime.materialized_static_parser_state_domain_labels.len(),
            )));
        }
        constraint.parser_dwa = parser_dwa;
        constraint.packed_parser_dwa = None;
        constraint.parser_state_domain_labels =
            runtime.materialized_static_parser_state_domain_labels;
    } else if !runtime.materialized_static_parser_state_domain_labels.is_empty() {
        return Err(crate::GlrMaskError::Serialization(
            "serialized static component parser labels exist without a parser".to_owned(),
        ));
    }
    let mut components = Vec::with_capacity(runtime.components.len());
    for (index, component) in runtime.components.into_iter().enumerate() {
        if component.global_to_local_parser_state.len() != global_state_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} parser-state projection has {} entries for {global_state_count} outer states",
                component.global_to_local_parser_state.len(),
            )));
        }
        if component.root_entry_terminals.len() != global_terminal_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} root-entry terminal set has length {} for {global_terminal_count} outer terminals",
                component.root_entry_terminals.len(),
            )));
        }
        let child = Constraint::load_body_artifact(component.constraint_artifact)?;
        if component
            .terminal_offset
            .checked_add(child.table.num_terminals)
            .is_none_or(|end| end as usize > global_terminal_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} terminal range lies outside outer terminal domain"
            )));
        }
        if component
            .tokenizer_state_offset
            .checked_add(child.tokenizer.num_states().saturating_sub(1))
            .is_none_or(|last| last >= global_tokenizer_states)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} tokenizer-state range lies outside outer tokenizer"
            )));
        }
        if component
            .root_disallowed_terminal
            .is_some_and(|terminal| terminal >= child.table.num_terminals)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} root-disallowed terminal lies outside the component"
            )));
        }
        if component.global_to_local_parser_state.iter().any(|&local| {
            local != u32::MAX && local >= child.table.num_states
        }) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "segmented component {index} parser-state projection references an invalid local state"
            )));
        }
        if component.global_terminal_aliases.iter().any(|&(global, local)| {
            global as usize >= global_terminal_count || local >= child.table.num_terminals
        }) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "serialized component {index} contains an invalid terminal alias"
            )));
        }
        components.push(crate::runtime::SegmentedParserComponent {
            constraint: std::sync::Arc::new(child),
            boundary: None,
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: component.global_terminal_aliases,
            local_tsid_to_global_tsids: Vec::new(),
            root_disallowed_terminal: component.root_disallowed_terminal,
            global_to_local_parser_state: component.global_to_local_parser_state,
        });
    }
    if !runtime.segmented_component_union_root_dispatch.is_empty()
        && runtime.segmented_component_union_root_dispatch.len() != global_state_count
    {
        return Err(crate::GlrMaskError::Serialization(
            "segmented component root dispatch has the wrong outer parser-state domain".to_owned(),
        ));
    }
    if runtime
        .segmented_component_union_root_dispatch
        .iter()
        .any(|&component| component != u32::MAX && component as usize >= components.len())
    {
        return Err(crate::GlrMaskError::Serialization(
            "segmented component root dispatch references an unknown component".to_owned(),
        ));
    }
    let overlay = constraint
        .static_dynamic_overlay
        .get_or_insert_with(Default::default);
    overlay.segmented_parser_components = components;
    overlay.segmented_mask_authoritative = runtime.segmented_mask_authoritative;
    overlay.segmented_static_baseline = has_static_baseline;
    overlay.segmented_component_union_root_dispatch =
        runtime.segmented_component_union_root_dispatch;

    restore_boundary_shards(
        constraint, runtime.boundary_shards, !runtime.segmented_parser_links.is_empty(),
    )?;
    let overlay = constraint.static_dynamic_overlay.as_mut()
        .expect("component restoration initializes its runtime overlay");
    let links = runtime.segmented_parser_links
        .into_iter()
        .map(|link| crate::runtime::SegmentedParserLink {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect::<Vec<_>>();
    overlay.segmented_parser_links = links;
    overlay.segmented_parser_state_offsets = runtime.segmented_parser_state_offsets;

    if !overlay.segmented_parser_state_offsets.is_empty() {
        let tables = crate::runtime::SegmentedParserComponentTables::new(
            &overlay.segmented_parser_components,
        );
        crate::compiler::glr::parser::DisjointComponentActionProvider::with_state_offsets(
            &tables,
            &overlay.segmented_parser_links,
            &overlay.segmented_parser_state_offsets,
        )
        .map_err(crate::GlrMaskError::Serialization)?;
    }

    // v24 static B was compiled in the materialized composed-table parser
    // coordinate. There is no exact state-symbol homomorphism which can, in
    // general, transport that stack language to the recursive leaf coordinate:
    // independently mapping each stack symbol loses correlations between stack
    // positions. Keep such artifacts on their exact legacy runtime instead.
    //
    // Transitional v24 artifacts produced while the recursive runtime was being
    // developed may already have cleared their component start-state bitsets.
    // Reconstruct those exactly from the still-serialized materialized -> local
    // component projections. Historical v24 artifacts already carry the same
    // full-length sets, so this is a no-op for them.
    let has_legacy_boundary = overlay.segmented_boundary_parser.is_some()
        || overlay.segmented_boundary_terminal_trie.is_some()
        || overlay.segmented_parser_components.iter().any(|component| {
            component.boundary.as_ref().is_some_and(|shard| {
                !matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                )
            })
        });
    if has_legacy_boundary {
        let global_state_count = constraint.table.num_states as usize;
        for component in &mut overlay.segmented_parser_components {
            let Some(shard) = component.boundary.as_mut() else {
                continue;
            };
            if shard.start_parser_states.len() != 0 {
                continue;
            }
            let mut starts = crate::ds::bitset::BitSet::new(global_state_count);
            for (global_state, &local_state) in
                component.global_to_local_parser_state.iter().enumerate()
            {
                if local_state != u32::MAX {
                    starts.set(global_state);
                }
            }
            shard.start_parser_states = starts;
        }
        overlay.segmented_boundary_shards = overlay
            .segmented_parser_components
            .iter()
            .filter_map(|component| component.boundary.clone())
            .collect();
        // Prevent a global/static legacy B from accidentally satisfying the
        // recursive-runtime shape test merely because v24 also serialized
        // linker metadata. As a loaded legacy constraint this node remains an
        // opaque materialized parser if it is later embedded in v27.
        overlay.segmented_parser_links.clear();
        overlay.segmented_parser_state_offsets.clear();
        return Ok(());
    }

    if constraint.uses_compact_segmented_parser_runtime() {
        constraint
            .static_dynamic_overlay
            .as_mut()
            .expect("compact v24 runtime requires overlay")
            .segmented_parser_state_offsets
            .clear();
        constraint.clear_recursive_legacy_boundary_start_states();
        constraint.clear_recursive_legacy_parser_state_projections();
    }
    Ok(())
}

pub(super) fn restore_recursive_runtime(
    constraint: &mut Constraint,
    runtime: RecursiveRuntimeArtifact,
) -> crate::Result<()> {
    if !runtime.segmented_mask_authoritative {
        return Err(crate::GlrMaskError::Serialization(
            "recursive v27 segmented runtime must be mask-authoritative".to_owned(),
        ));
    }
    let recursive_compiler_table = runtime.recursive_compiler_table;
    let recursive_tokenizer_internal_tsids = runtime.recursive_tokenizer_internal_tsids;
    let global_terminal_count = constraint.table.num_terminals as usize;
    let global_tokenizer_states = constraint.tokenizer.num_states();
    // Immediate component artifacts are independent. Decode them in parallel
    // before validating their placement in the outer compiler-oracle
    // coordinates. This is especially important for nested recursive
    // compositions: a sequential loop turns load latency into the sum of every
    // descendant artifact load, while Rayon lets the component tree collapse
    // to roughly its critical path without changing the wire or runtime tree.
    let decoded_components = runtime
        .components
        .into_par_iter()
        .map(|component| {
            let child = Constraint::load_body_artifact(&component.constraint_artifact)?;
            Ok::<_, crate::GlrMaskError>((component, child))
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let mut components = Vec::with_capacity(decoded_components.len());
    let mut expected_tokenizer_state_offset = 0u32;
    for (index, (component, child)) in decoded_components.into_iter().enumerate() {
        if component.root_entry_terminals.len() != global_terminal_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} root-entry terminal set has length {} for {global_terminal_count} outer terminals",
                component.root_entry_terminals.len(),
            )));
        }
        if component
            .terminal_offset
            .checked_add(child.table.num_terminals)
            .is_none_or(|end| end as usize > global_terminal_count)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} terminal range lies outside outer terminal domain"
            )));
        }
        if component.tokenizer_state_offset != expected_tokenizer_state_offset {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} tokenizer-state offset is {}, expected {expected_tokenizer_state_offset}",
                component.tokenizer_state_offset,
            )));
        }
        let child_tokenizer_span = child
            .recursive_parser_layout()
            .map_err(crate::GlrMaskError::Serialization)?
            .map_or(child.tokenizer.num_states(), |layout| layout.total_tokenizer_states);
        expected_tokenizer_state_offset = expected_tokenizer_state_offset
            .checked_add(child_tokenizer_span)
            .ok_or_else(|| {
                crate::GlrMaskError::Serialization(
                    "recursive v27 tokenizer-state coordinate overflow".to_owned(),
                )
            })?;
        if component
            .root_disallowed_terminal
            .is_some_and(|terminal| terminal >= child.table.num_terminals)
        {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} root-disallowed terminal lies outside the component"
            )));
        }
        if component.global_terminal_aliases.iter().any(|&(global, local)| {
            global >= constraint.table.num_terminals || local >= child.table.num_terminals
        }) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 component {index} contains an invalid terminal alias"
            )));
        }
        components.push(crate::runtime::SegmentedParserComponent {
            constraint: Arc::new(child),
            boundary: None,
            tokenizer_state_offset: component.tokenizer_state_offset,
            terminal_offset: component.terminal_offset,
            global_terminal_aliases: component.global_terminal_aliases,
            local_tsid_to_global_tsids: component.local_tsid_to_global_tsids,
            root_disallowed_terminal: component.root_disallowed_terminal,
            global_to_local_parser_state: Vec::new(),
        });
    }
    if expected_tokenizer_state_offset as usize != recursive_tokenizer_internal_tsids.len() {
        return Err(crate::GlrMaskError::Serialization(format!(
            "recursive v27 tokenizer-state relation has {} rows for {expected_tokenizer_state_offset} recursive states",
            recursive_tokenizer_internal_tsids.len(),
        )));
    }
    let links = runtime
        .segmented_parser_links
        .into_iter()
        .map(|link| crate::runtime::SegmentedParserLink {
            parent_component: link.parent_component,
            slot_terminal: link.slot_terminal,
            child_component: link.child_component,
            child_start: link.child_start,
            return_pop: link.return_pop,
            child_start_nullable: link.child_start_nullable,
        })
        .collect::<Vec<_>>();
    if components.is_empty() || links.is_empty() {
        return Err(crate::GlrMaskError::Serialization(
            "recursive v27 segmented runtime requires components and linker controls".to_owned(),
        ));
    }
    {
        let overlay = constraint
            .static_dynamic_overlay
            .get_or_insert_with(Default::default);
        overlay.segmented_parser_components = components;
        overlay.segmented_parser_links = links;
        overlay.segmented_parser_state_offsets.clear();
        overlay.segmented_mask_authoritative = true;
        overlay.segmented_static_baseline = false;
        overlay.segmented_component_union_root_dispatch.clear();
        overlay.segmented_boundary_shards.clear();
        overlay.segmented_boundary_parser = None;
        overlay.segmented_boundary_terminal_trie = None;
        overlay
            .recursive_compiler_table
            .set(Arc::from(recursive_compiler_table.into_boxed_slice()))
            .map_err(|_| {
                crate::GlrMaskError::Serialization(
                    "recursive compiler table was initialized twice".to_owned(),
                )
            })?;
    }
    let layout = constraint
        .recursive_parser_layout_for_pending_root()
        .map_err(crate::GlrMaskError::Serialization)?
        .ok_or_else(|| {
            crate::GlrMaskError::Serialization(
                "recursive v27 runtime failed to derive its parser layout".to_owned(),
            )
        })?;
    constraint
        .install_recursive_tokenizer_internal_tsids(recursive_tokenizer_internal_tsids)
        .map_err(crate::GlrMaskError::Serialization)?;
    let component_count = constraint
        .static_dynamic_overlay
        .as_ref()
        .expect("recursive v27 overlay exists")
        .segmented_parser_components
        .len();
    let mut seen = vec![false; component_count];
    let mut restored_shards = Vec::new();
    for (index, shard) in runtime.boundary_shards.into_iter().enumerate() {
        let component_index = shard.start_component as usize;
        if component_index >= component_count {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 boundary shard {index} references unknown component {}",
                shard.start_component,
            )));
        }
        if seen[component_index] {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 runtime contains multiple boundary shards for component {}",
                shard.start_component,
            )));
        }
        seen[component_index] = true;
        if shard.accepts_empty_stack && shard.start_component != 0 {
            return Err(crate::GlrMaskError::Serialization(format!(
                "recursive v27 boundary shard {index} gives empty-stack ownership to non-root component {}",
                shard.start_component,
            )));
        }
        let backend = match shard.backend {
            RecursiveBoundaryBackendArtifact::StaticParser(boundary) => {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(Arc::new(
                    restore_recursive_boundary_parser(
                        constraint,
                        boundary,
                        layout.total_states,
                        layout.total_tokenizer_states,
                    )?,
                ))
            }
            RecursiveBoundaryBackendArtifact::DynamicDirect => {
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
            }
        };
        let candidate_tokens = shard.candidate_tokens.map(|mut tokens| {
            tokens.sort_unstable();
            tokens.dedup();
            Arc::<[u32]>::from(tokens)
        });
        restored_shards.push(crate::runtime::SegmentedBoundaryShard {
            start_component: shard.start_component,
            start_parser_states: crate::ds::bitset::BitSet::new(0),
            accepts_empty_stack: shard.accepts_empty_stack,
            candidate_tokens,
            backend,
        });
    }
    let overlay = constraint
        .static_dynamic_overlay
        .as_mut()
        .expect("recursive v27 overlay exists");
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for shard in &restored_shards {
        overlay.segmented_parser_components[shard.start_component as usize].boundary =
            Some(shard.clone());
    }
    overlay.segmented_boundary_shards = restored_shards;
    if !constraint.uses_compact_segmented_parser_runtime() {
        return Err(crate::GlrMaskError::Serialization(
            "recursive v27 segmented runtime did not restore a provider-native parser"
                .to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn restore_segmented_runtime(
    constraint: &mut Constraint,
    runtime: SegmentedRuntimeArtifact,
) -> crate::Result<()> {
    match runtime {
        SegmentedRuntimeArtifact::Recursive(runtime) => {
            restore_recursive_runtime(constraint, runtime)
        }
        SegmentedRuntimeArtifact::Materialized(runtime) => {
            restore_materialized_runtime(constraint, runtime)
        }
    }
}
