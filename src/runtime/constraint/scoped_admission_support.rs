//! Query-local native-terminal support for an already control-closed frontier.
//! This is a rejection filter, not a replacement parser or admission oracle.
use super::recursive_parser::RecursiveSegmentedParserTables;
use crate::compiler::glr::parser::{
    DisjointComponentActionProvider, ParserActionProvider, ParserComponentTableSource,
    ParserGSS, ProvidedAction, ScopedParserSymbol, close_provider_control_stacks,
    find_admitted_symbol_with_provider,
};
use crate::ds::bitset::BitSet;
use crate::runtime::artifact::{Constraint, RecursiveParserLayout};

const MAX_TERMINALS: usize = 4096;
const MAX_TOPS: usize = 16;
const MAX_ROW_KEYS: usize = 4096;
const MAX_DEPTH: u32 = 256;

pub(crate) struct ScopedAdmissionSupport<'a> {
    root: &'a Constraint,
    layout: &'a RecursiveParserLayout,
    closed: ParserGSS,
    leaf: u32,
    offset: u32,
    support: BitSet,
}

// Only input control closure is suppressed. All visible actions, reductions,
// guards, scopes and goto behavior are delegated unchanged. In particular we
// never assume the bounded reference closure is mathematically idempotent.
struct AlreadyClosed<'a, P>(&'a P);
impl<P: ParserActionProvider> ParserActionProvider for AlreadyClosed<'_, P> {
    type Symbol = P::Symbol;
    fn action(&self, state: u32, symbol: Self::Symbol) -> Option<ProvidedAction<'_>> {
        self.0.action(state, symbol)
    }
    fn scope_state(&self, scope: u32, state: u32) -> Option<u32> {
        self.0.scope_state(scope, state)
    }
    fn goto_target(&self, scope: u32, from: u32, nonterminal: u32) -> Option<(u32, bool)> {
        self.0.goto_target(scope, from, nonterminal)
    }
    fn state_count_hint(&self) -> usize { self.0.state_count_hint() }
}

impl ScopedAdmissionSupport<'_> {
    pub(crate) fn retained_count(&self, candidates: &BitSet) -> usize {
        self.support.iter_ones().filter(|&local| {
            candidates.contains(self.offset as usize + local)
        }).count()
    }

    /// Candidates come from exactly this immutable lexer's native leaf domain.
    /// Iterating ascending support bits preserves the original candidate and
    /// predicate order; removed candidates have no first provider action.
    pub(crate) fn matches(
        &self, candidates: &BitSet, mut lexical: impl FnMut(u32) -> bool,
    ) -> bool {
        if self.support.is_empty() { return false; }
        let tables = RecursiveSegmentedParserTables { root: self.root, layout: self.layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &self.layout.links, &self.layout.leaf_state_offsets,
        ).expect("validated scoped support layout");
        let symbols = self.support.iter_ones().filter_map(|local| {
            let terminal = self.offset + local as u32;
            candidates.contains(terminal as usize).then_some((terminal, ScopedParserSymbol::Terminal {
                component: self.leaf, terminal: local as u32,
            }))
        });
        find_admitted_symbol_with_provider(
            &AlreadyClosed(&provider), &self.closed, symbols, |&terminal| lexical(terminal),
        ).is_some()
    }
}

impl Constraint {
    pub(crate) fn prepare_scoped_admission_support(
        &self, input: &ParserGSS, leaf: usize,
    ) -> Option<ScopedAdmissionSupport<'_>> {
        if input.max_depth() > MAX_DEPTH { return None; }
        let layout = self.recursive_parser_layout_ref()?;
        let descriptor = layout.leaves.get(leaf)?;
        let source = self.constraint_at_recursive_component_path(&descriptor.component_path)?;
        let count = source.table.num_terminals;
        if count as usize > MAX_TERMINALS { return None; }
        let offset = layout.outer_terminal_count.checked_add(*layout.leaf_terminal_offsets.get(leaf)?)?;
        offset.checked_add(count)?;
        let tables = RecursiveSegmentedParserTables { root: self, layout };
        let provider = DisjointComponentActionProvider::with_state_offsets(
            &tables, &layout.links, &layout.leaf_state_offsets,
        ).ok()?;
        let closed = close_provider_control_stacks(&provider, input);
        if closed.max_depth() > MAX_DEPTH { return None; }
        let tops = closed.peek_values();
        if tops.len() > MAX_TOPS { return None; }
        let mut support = BitSet::new(count as usize);
        let mut remaining_keys = MAX_ROW_KEYS;
        for top in tops {
            let (owner, local) = provider.decode_state(top)?;
            if owner as usize != leaf { continue; }
            let row = source.table.action.get(local as usize)?;
            // Use CURRENT execution-row keys, not a row-presence approximation
            // captured before guard-producing optimizations. Default rows'
            // iterator enumerates their actual non-hole terminal cells.
            for terminal in row.keys() {
                remaining_keys = remaining_keys.checked_sub(1)?;
                if terminal < count { support.set(terminal as usize); }
            }
            // Provider-owned ignores can supply Identity without a table cell.
            if let Some(ignore) = tables.component_ignore_terminal(owner) {
                if ignore < count { support.set(ignore as usize); }
            }
        }
        Some(ScopedAdmissionSupport { root: self, layout, closed, leaf: leaf as u32, offset, support })
    }
}
