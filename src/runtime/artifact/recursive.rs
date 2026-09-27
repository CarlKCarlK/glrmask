//! Scoped recursive parser layouts and segmented boundary representations.

use super::Constraint;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::parser::ParserComponentTableSource;
use crate::compiler::glr::parser::ScopedSubgrammarLink;
use crate::compiler::glr::table::GLRTable;
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
pub(crate) type SegmentedParserLink = ScopedSubgrammarLink;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecursiveParserLeafLayout {
    pub(crate) state_offset: u32,
    pub(crate) state_count: u32,
    /// Immediate wrapper of the composition whose layout was requested. Inner
    /// leaf changes within one nested component deliberately keep this owner.
    pub(crate) top_component: u32,
    /// Immediate-component path from the requested composition root to this
    /// intact LR table.
    pub(crate) component_path: Vec<u32>,
}

pub(super) const RECURSIVE_VIRTUAL_TOKENIZER_STATE_LIMIT: u32 = 1 << 31;

#[derive(Debug, Default)]
pub(super) struct RecursiveVirtualTokenizerStateStore {
    pub(super) next_state: u32,
    pub(super) scoped_by_local: FxHashMap<(u32, u32), u32>,
    pub(super) local_by_scoped: FxHashMap<u32, (u32, u32)>,
}

#[derive(Debug, Default)]
pub(crate) struct RecursiveVirtualTokenizerStates {
    pub(super) store: Mutex<RecursiveVirtualTokenizerStateStore>,
}

impl RecursiveVirtualTokenizerStates {
    pub(crate) fn scoped_state(
        &self,
        physical_state_count: u32,
        leaf_index: usize,
        local_state: u32,
    ) -> Option<u32> {
        let leaf_index = u32::try_from(leaf_index).ok()?;
        let mut store = self.store.lock().ok()?;
        if let Some(&scoped) = store.scoped_by_local.get(&(leaf_index, local_state)) {
            return Some(scoped);
        }
        if store.next_state < physical_state_count {
            store.next_state = physical_state_count;
        }
        if store.next_state >= RECURSIVE_VIRTUAL_TOKENIZER_STATE_LIMIT {
            return None;
        }
        let scoped = store.next_state;
        store.next_state = store.next_state.checked_add(1)?;
        store
            .scoped_by_local
            .insert((leaf_index, local_state), scoped);
        store
            .local_by_scoped
            .insert(scoped, (leaf_index, local_state));
        Some(scoped)
    }

    pub(crate) fn local_state(&self, scoped_state: u32) -> Option<(usize, u32)> {
        let store = self.store.lock().ok()?;
        let &(leaf_index, local_state) = store.local_by_scoped.get(&scoped_state)?;
        Some((leaf_index as usize, local_state))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecursiveParserLayout {
    pub(crate) component_offsets: Vec<u32>,
    pub(crate) leaves: Vec<RecursiveParserLeafLayout>,
    pub(crate) leaf_state_offsets: Vec<u32>,
    /// Disjoint-union tokenizer-state coordinate over the same intact leaves.
    /// Parser and tokenizer leaves have identical ordering/component paths, but
    /// independent offsets because their local state counts differ.
    pub(crate) leaf_tokenizer_state_offsets: Vec<u32>,
    pub(crate) total_tokenizer_states: u32,
    /// Disjoint-union terminal coordinate over the same intact leaves. Runtime
    /// byte terminals are encoded after the outer/materialized terminal range,
    /// so the `u32` is self-describing while byte scanning no longer needs a
    /// leaf-local -> outer -> leaf-local round trip.
    pub(crate) leaf_terminal_offsets: Vec<u32>,
    pub(crate) total_leaf_terminals: u32,
    /// Size of the transitional outer/global terminal prefix at layout-build
    /// time. Live recursive terminal IDs use this immutable tag boundary and
    /// therefore do not need to consult the materialized outer table again.
    pub(crate) outer_terminal_count: u32,
    /// Lazily mapped future-terminal support in the live runtime terminal
    /// coordinate for each scoped leaf tokenizer state.
    pub(crate) tokenizer_future_scoped: Vec<OnceLock<BitSet>>,
    /// Linker controls rewritten only into the leaf-component coordinate.
    pub(crate) links: Vec<SegmentedParserLink>,
    /// Outer/global terminal -> intact leaf-local terminals. Ordinary byte
    /// commit no longer consumes this relation; it remains for compiler-side
    /// static-B materialization, compatibility/reference evaluation, and outer
    /// special-token routing.
    pub(crate) terminal_targets: Vec<SmallVec<[(u32, TerminalID); 4]>>,
    pub(crate) total_states: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct StaticDynamicOverlayMetadata {
    /// Global terminal-id offsets for the transported composition components.
    pub(crate) terminal_offsets: Vec<u32>,
    /// Global raw-tokenizer-state offsets for those same components. State zero
    /// is the merged reset dispatcher and deliberately belongs to no component.
    pub(crate) tokenizer_state_offsets: Vec<u32>,
    /// Terminals whose composed parser template has behavior absent from the
    /// transported component parser artifacts (including scoped-ignore repair
    /// and conservative unsafe terminals).
    pub(crate) repair_terminals: Vec<bool>,
    /// Composed LR states which belong to one or more child components but not
    /// to the parent component. Runtime lookahead-return factoring is useful
    /// only while the concrete top state is still inside such a child-owned
    /// region; ordinary parent reductions must not pay for that machinery.
    #[serde(default)]
    pub(crate) non_parent_only_parser_states: Vec<bool>,
    /// Exact segmented parser backend. Each retained source constraint keeps
    /// its own parser/token coordinate and is projected from the composed
    /// tokenizer/LR coordinates at mask time. Current artifacts flatten static
    /// components into one parser DWA but retain dynamic/hybrid components.
    #[serde(skip, default)]
    pub(crate) segmented_parser_components: Vec<SegmentedParserComponent>,
    /// Explicit local-coordinate calls between the wrappers above. This is the
    /// semantic relation consumed by the compact parser-action provider; it is
    /// intentionally scoped to this composition level only.
    #[serde(skip, default)]
    pub(crate) segmented_parser_links: Vec<SegmentedParserLink>,
    /// Legacy v24 prefix offsets for the old immediate-component compact parser
    /// coordinate. New recursive runtimes derive wrapper/leaf intervals from
    /// `segmented_parser_components` and leave this empty; retained only for
    /// loading older current-version artifacts and legacy experiment paths.
    #[serde(skip, default)]
    pub(crate) segmented_parser_state_offsets: Vec<u32>,
    /// Runtime-derived endpoint parser view. The semantic component tree stays
    /// literal; this cache contains only the flat leaf coordinate required by
    /// the existing `LeveledGSS<u32, ...>` action provider.
    #[serde(skip, default)]
    pub(crate) recursive_parser_layout: OnceLock<Arc<RecursiveParserLayout>>,
    /// Exact packed flattened parser table retained only for future compiler
    /// rebinding. Live recursive execution never reads it. The coordinator's
    /// ordinary `table` field is reduced to a grammar shell after compilation;
    /// late composition materializes this blob into a temporary compiler clone
    /// and detaches it again before publishing runtime components.
    #[serde(skip, default)]
    pub(crate) recursive_compiler_table: OnceLock<Arc<[u8]>>,
    /// Exact endpoint lexical relation: recursive leaf tokenizer state -> this
    /// constraint's internal TSID set. Runtime-product components can make the
    /// image genuinely set-valued, so this must not be collapsed to one TSID.
    #[serde(skip, default)]
    pub(crate) recursive_tokenizer_internal_tsids: OnceLock<Arc<Vec<Vec<u32>>>>,
    /// Lazily allocated outer scoped IDs for exact virtual tokenizer states in
    /// retained recursive leaves. Physical states keep their contiguous layout
    /// IDs; only actually reached virtual states enter this runtime-only map.
    #[serde(skip, default)]
    pub(crate) recursive_virtual_tokenizer_states: Arc<RecursiveVirtualTokenizerStates>,
    /// The segmented A/B factorization is the masking implementation for this
    /// constraint, rather than an optional validation view of a flattened
    /// parser DWA. Current serialization preserves this split explicitly.
    #[serde(skip, default)]
    pub(crate) segmented_mask_authoritative: bool,
    /// A serialized hybrid may flatten only its static component contribution
    /// into `Constraint::parser_dwa` while retaining dynamic components below.
    /// When set, segmented masking starts from that static parser-DWA mask and
    /// ORs the retained dynamic components plus boundary B.
    #[serde(skip, default)]
    pub(crate) segmented_static_baseline: bool,
    /// Compressed deterministic union root for `segmented_parser_components`.
    /// Entry `g` is the unique component selected by composed LR state `g`, or
    /// `u32::MAX` when no component has a root transition on that state. When
    /// non-empty, the component collection is one deterministic parser DWA in
    /// segmented storage: a synthetic root followed by one cached component
    /// body. No runtime parser-NWA branching is involved.
    #[serde(skip, default)]
    pub(crate) segmented_component_union_root_dispatch: Vec<u32>,
    /// Authoritative cross-component accelerators, optionally restricted to
    /// model tokens whose parser execution starts in one component.  This is
    /// the new composition model: boundary acceleration is selected per
    /// starting component, while the retained component constraints remain
    /// the ordinary A contribution.  `start_component == None` is the exact
    /// legacy/global fallback used while older artifacts and compiler paths
    /// are migrated.
    #[serde(skip, default)]
    pub(crate) segmented_boundary_shards: Vec<SegmentedBoundaryShard>,
    /// Legacy/global v22 storage.  New live compositions also publish an Arc
    /// to the same backend through `segmented_boundary_shards`; these fields
    /// remain until the partitioned boundary wire format is versioned.
    #[serde(skip, default)]
    pub(crate) segmented_boundary_parser: Option<Arc<SegmentedBoundaryParser>>,
    #[serde(skip, default)]
    pub(crate) segmented_boundary_terminal_trie: Option<Arc<SegmentedBoundaryTerminalTrie>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedBoundaryShard {
    /// Component in which the model token starts.
    pub(crate) start_component: u32,
    /// Exact composed LR top states that can represent this starting
    /// component. A state may intentionally occur in more than one shard when
    /// the composed relation is non-functional; evaluating both shards is then
    /// conservative and exact under OR.
    pub(crate) start_parser_states: BitSet,
    /// Whether an empty composed parser stack belongs to this shard. Only the
    /// outer/root component normally owns that coordinate.
    pub(crate) accepts_empty_stack: bool,
    /// Optional conservative model-token summary for the first internal
    /// component crossing. `None` means no trigger information is available;
    /// it must never be interpreted as "no crossings".
    pub(crate) candidate_tokens: Option<Arc<[u32]>>,
    pub(crate) backend: SegmentedBoundaryShardBackend,
}

#[derive(Debug, Clone)]
pub(crate) enum SegmentedBoundaryShardBackend {
    StaticParser(Arc<SegmentedBoundaryParser>),
    /// Preserved terminal-language/NWA crossing accelerator used by historical
    /// artifacts and the explicit experimental runtime path. Normal current
    /// dynamic composition publishes `DynamicDirect` instead.
    DynamicTerminalTrie(Arc<SegmentedBoundaryTerminalTrie>),
    /// Exact dynamic crossing with no required composition-specific B artifact.
    /// Materialized runtimes use the strict dynamic full-vocabulary walker;
    /// recursive runtimes use the scoped shared-prefix vocabulary walker.
    DynamicDirect,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentedParserComponent {
    pub(crate) constraint: Arc<Constraint>,
    /// Composition-specific crossing backend for model tokens that start in
    /// this component. This lives on the wrapper, not on the reusable
    /// underlying constraint.
    pub(crate) boundary: Option<SegmentedBoundaryShard>,
    pub(crate) tokenizer_state_offset: u32,
    pub(crate) terminal_offset: u32,
    /// Outer composed terminals which are canonical aliases for a local
    /// terminal of this component. Direct interval mapping remains implicit;
    /// this stores only exceptional many-scope aliases (currently globally
    /// equivalent ignore terminals).
    pub(crate) global_terminal_aliases: Vec<(u32, u32)>,
    /// Component-local Static TSID -> composed Static TSID relation retained
    /// for lazy exact tokenizer states that are allocated after link time.
    pub(crate) local_tsid_to_global_tsids: Vec<Vec<u32>>,
    /// Legacy v22 compatibility metadata. New authoritative A+B compositions
    /// leave this `None`: component parser DWAs keep their standalone semantics
    /// unchanged, and scope/link behavior lives in the composed parser view/B.
    pub(crate) root_disallowed_terminal: Option<u32>,
    pub(crate) global_to_local_parser_state: Vec<u32>,
}

#[derive(Clone, Copy)]
pub(crate) struct SegmentedParserComponentTables<'a> {
    pub(super) components: &'a [SegmentedParserComponent],
}

impl<'a> SegmentedParserComponentTables<'a> {
    #[inline]
    pub(crate) fn new(components: &'a [SegmentedParserComponent]) -> Self {
        Self { components }
    }
}

impl ParserComponentTableSource for SegmentedParserComponentTables<'_> {
    #[inline]
    fn component_count(&self) -> usize {
        self.components.len()
    }

    #[inline]
    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.components
            .get(component as usize)
            .map(|component| &component.constraint.table)
    }

    #[inline]
    fn component_ignore_terminal(&self, component: u32) -> Option<u32> {
        let component = self.components.get(component as usize)?;
        // A composed component table already owns its scoped ignores in its
        // rows (`skip_terminals` provenance); provider-level Identity would
        // shadow those rows. Only raw component tables need it here.
        let ignore = component.constraint.ignore_terminal?;
        (!component.constraint.table.skip_terminals.contains(&ignore)).then_some(ignore)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalTrieNode {
    pub(crate) children: Vec<(u32, u32)>,
    /// Legacy v21 representation: private boundary token classes accepted at
    /// this node after expanding the TSID dimension during construction.
    pub(crate) outputs: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalNwaTransition {
    pub(crate) terminal: u32,
    pub(crate) target: u32,
    pub(crate) weight: Weight,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalNwaNode {
    pub(crate) final_weight: Option<Weight>,
    pub(crate) transitions: Vec<BoundaryTerminalNwaTransition>,
    pub(crate) epsilons: Vec<(u32, Weight)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct BoundaryTerminalNwa {
    pub(crate) nodes: Vec<BoundaryTerminalNwaNode>,
    pub(crate) start_states: Vec<u32>,
    /// A topological order of `nodes`, validated when the artifact is loaded.
    /// Runtime evaluation uses it to coalesce equal token domains at converged
    /// NWA states without materializing the exponentially large prefix trie.
    pub(crate) topological_order: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentedBoundaryTerminalTrie {
    pub(crate) nodes: Vec<BoundaryTerminalTrieNode>,
    pub(crate) root_by_tsid: Vec<u32>,
    pub(crate) tokenizer_state_to_tsid: Vec<u32>,
    pub(crate) internal_token_to_originals: Vec<Vec<u32>>,
    /// Current representation. The legacy v21 serde shape above is retained so
    /// old artifacts still decode; v22 persists this DAG explicitly.
    #[serde(skip, default)]
    pub(crate) symbolic_nwa: Option<BoundaryTerminalNwa>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct SegmentedBoundaryParser {
    /// Generic wire/reference representation. Compact in-memory boundary
    /// parsers leave this as the empty one-state DWA and use
    /// `compact_parser_dwa`; V17 serialization materializes the compact machine
    /// back into this exact generic wire shape.
    pub(crate) parser_dwa: DWA,
    #[serde(skip, default)]
    pub(crate) compact_parser_dwa: Option<crate::compiler::stages::parser_dwa::SmallBoundaryDwa>,
    /// Provider-native static boundary parser in the recursive leaf parser-state
    /// coordinate. New v25 compositions compile and serialize this coordinate
    /// directly. Legacy v24 static boundaries retain only `parser_dwa` and stay
    /// on their materialized runtime when loaded.
    #[serde(skip, default)]
    pub(crate) recursive_parser_dwa: Option<DWA>,
    /// New static boundary shards are compiled directly in the authoritative
    /// composed constraint TSID coordinate. They therefore need no private
    /// raw-tokenizer-state map; runtime reads `Constraint::state_to_internal_tsid`
    /// instead. Legacy/global boundary artifacts keep this false and retain the
    /// explicit private map below.
    #[serde(skip, default)]
    pub(crate) uses_composed_tsid_coordinate: bool,
    pub(crate) tokenizer_state_to_tsid: Vec<u32>,
    pub(crate) internal_token_to_originals: Vec<Vec<u32>>,
}
