use super::accumulator::TerminalsDisallowed;
use super::analysis::EOF;
use super::table::{
    Action,
    AdmissionPolicy,
    GLRTable,
    GlrTableConstruction,
    GuardedShiftCellIndex,
    GuardedStackShift,
    StackShift,
    StackShiftGuard,
};
use super::table::row::{ActionRow, GotoRow};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{GssSemanticKeyInterner, LeveledGSS, Merge, VirtualStack};
use crate::grammar::flat::TerminalID;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::OnceLock;

mod profile;

pub use profile::{
    AdvanceTrace,
    AdvanceTraceGoto,
    AdvanceProfile,
    AdvanceTraceReduce,
    AdvanceTraceStep,
    AdvanceTraceWave,
};

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopedParserSymbol {
    Terminal { component: u32, terminal: TerminalID },
    Entry { link: u32 },
    Finish { component: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopedSubgrammarLink {
    pub parent_component: u32,
    pub slot_terminal: TerminalID,
    pub child_component: u32,
    pub child_start: u32,
    pub return_pop: u32,
    /// Standalone child compilation may normalize the epsilon-root reduction
    /// out of the local action rows. In that case FINISH at child start still
    /// has an immediate pop-one return alternative.
    pub child_start_nullable: bool,
}

/// Borrowed action view for a compact disjoint-union parser-state coordinate.
/// Component tables keep local state IDs unchanged; the provider scopes them
/// lazily only when a state is actually pushed onto the existing `u32` GSS.
pub enum ProvidedActionRef<'a> {
    Local { scope: u32, action: &'a Action },
    /// Composition-scoped identity action, used primarily to expose an intact
    /// component's standalone ignore terminal without rewriting every LR row.
    Identity,
    Call {
        parent_target: u32,
        child_start: u32,
        replace: bool,
    },
    Return { pop: u32 },
}

pub struct ProvidedAction<'a> {
    pub action: ProvidedActionRef<'a>,
    /// Component whose local nonterminal coordinate must be used by reductions.
    pub reduction_scope: u32,
    /// Additional already-scoped stack effects which coexist with `action`.
    /// Used for nullable child FINISH when local EOF work also exists.
    pub extra_stack_shifts: SmallVec<[StackShift; 1]>,
}

pub trait ParserActionProvider {
    type Symbol: Copy + Eq;

    fn action(&self, state: u32, symbol: Self::Symbol) -> Option<ProvidedAction<'_>>;

    /// Inject one component-local LR state into the compact runtime disjoint
    /// union. Ordinary one-table parsing is the identity injection.
    fn scope_state(&self, scope: u32, local_state: u32) -> Option<u32>;

    fn goto_target(
        &self,
        scope: u32,
        goto_from: u32,
        nonterminal: u32,
    ) -> Option<(u32, bool)>;

    fn control_symbols(&self, _state: u32, _out: &mut SmallVec<[Self::Symbol; 4]>) {}

    fn state_count_hint(&self) -> usize;
}

fn scoped_provider_nonterminal(
    ids: &mut BTreeMap<(u32, u32), u32>,
    scope: u32,
    local: u32,
) -> Result<u32, String> {
    if let Some(&id) = ids.get(&(scope, local)) {
        return Ok(id);
    }
    let id = u32::try_from(ids.len())
        .map_err(|_| "scoped provider nonterminal coordinate overflow".to_owned())?;
    ids.insert((scope, local), id);
    Ok(id)
}

fn materialize_scoped_local_action<P: ParserActionProvider>(
    provider: &P,
    scope: u32,
    action: &Action,
    nonterminals: &mut BTreeMap<(u32, u32), u32>,
) -> Result<Action, String> {
    let scoped = |state| {
        provider
            .scope_state(scope, state)
            .ok_or_else(|| format!("provider cannot scope local state {state} in scope {scope}"))
    };
    Ok(match action {
        Action::Shift(target, replace) => Action::Shift(scoped(*target)?, *replace),
        Action::StackShifts(shifts) => Action::StackShifts(
            shifts
                .iter()
                .map(|shift| {
                    Ok(StackShift {
                        pop: shift.pop,
                        pushes: shift
                            .pushes
                            .iter()
                            .map(|&state| scoped(state))
                            .collect::<Result<Vec<_>, String>>()?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        Action::GuardedStackShifts(shifts) => Action::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| {
                    Ok(GuardedStackShift {
                        guards: shift
                            .guards
                            .iter()
                            .map(|guard| {
                                Ok(StackShiftGuard {
                                    pop: guard.pop,
                                    states: guard
                                        .states
                                        .iter()
                                        .map(|&state| scoped(state))
                                        .collect::<Result<Vec<_>, String>>()?,
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                        pop: shift.pop,
                        pushes: shift
                            .pushes
                            .iter()
                            .map(|&state| scoped(state))
                            .collect::<Result<Vec<_>, String>>()?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        Action::Reduce(nonterminal, len) => Action::Reduce(
            scoped_provider_nonterminal(nonterminals, scope, *nonterminal)?,
            *len,
        ),
        Action::Split {
            shift,
            reduces,
            accept,
        } => Action::Split {
            shift: shift
                .map(|(target, replace)| -> Result<(u32, bool), String> {
                    Ok((scoped(target)?, replace))
                })
                .transpose()?,
            reduces: reduces
                .iter()
                .map(|&(nonterminal, len)| {
                    Ok((
                        scoped_provider_nonterminal(nonterminals, scope, nonterminal)?,
                        len,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?,
            accept: *accept,
        },
        Action::Accept => Action::Accept,
        Action::ReplaceShifts(targets) => Action::ReplaceShifts(
            targets
                .iter()
                .map(|&target| scoped(target))
                .collect::<Result<Vec<_>, String>>()?
                .into(),
        ),
        Action::Skip => Action::Skip,
    })
}

fn materialize_scoped_provided_action<P: ParserActionProvider>(
    provider: &P,
    provided: &ProvidedAction<'_>,
    nonterminals: &mut BTreeMap<(u32, u32), u32>,
) -> Result<Action, String> {
    match &provided.action {
        ProvidedActionRef::Local { scope, action } => {
            materialize_scoped_local_action(provider, *scope, action, nonterminals)
        }
        ProvidedActionRef::Identity => Ok(Action::Skip),
        ProvidedActionRef::Call {
            parent_target,
            child_start,
            replace,
        } => Ok(Action::StackShifts(vec![StackShift {
            pop: u32::from(*replace),
            pushes: vec![*parent_target, *child_start],
        }])),
        ProvidedActionRef::Return { pop } => Ok(Action::StackShifts(vec![StackShift {
            pop: *pop,
            pushes: Vec::new(),
        }])),
    }
}

/// Materialize the exact stack semantics of a scoped action provider into a
/// temporary ordinary GLR table, then eliminate the provider's zero-width
/// controls. The resulting state alphabet is exactly the provider's `u32`
/// coordinate; it does not reconstruct or depend on a composed/materialized LR
/// state space.
///
/// `terminal_symbols[t]` lists the provider symbols represented by ordinary
/// terminal `t`. Multiple entries are permitted only when they induce the same
/// action from a state (the normal alias case). Each semantic provider control
/// symbol receives one temporary terminal shared across all states so reductions
/// can continue through goto on that same control. Additional nullable FINISH
/// alternatives use private branch terminals, allowing them to coexist exactly
/// with local EOF work without requiring a new `Action` variant.
pub fn materialize_control_eliminated_scoped_provider_table<P>(
    provider: &P,
    terminal_symbols: &[SmallVec<[ScopedParserSymbol; 4]>],
) -> Result<GLRTable, String>
where
    P: ParserActionProvider<Symbol = ScopedParserSymbol>,
{
    let state_count = provider.state_count_hint();
    let num_states = u32::try_from(state_count)
        .map_err(|_| "scoped provider state coordinate overflow".to_owned())?;
    let ordinary_terminals = u32::try_from(terminal_symbols.len())
        .map_err(|_| "scoped provider terminal coordinate overflow".to_owned())?;
    let mut rows = (0..state_count)
        .map(|_| Vec::<(TerminalID, Action)>::new())
        .collect::<Vec<_>>();
    let mut nonterminals = BTreeMap::<(u32, u32), u32>::new();
    let mut skip_terminals = BTreeSet::<TerminalID>::new();

    for state in 0..num_states {
        for (terminal, symbols) in terminal_symbols.iter().enumerate() {
            let terminal = terminal as u32;
            let mut materialized = None::<Action>;
            for &symbol in symbols {
                let Some(provided) = provider.action(state, symbol) else {
                    continue;
                };
                if !provided.extra_stack_shifts.is_empty() {
                    return Err(format!(
                        "ordinary provider symbol {symbol:?} unexpectedly carries extra control stack shifts"
                    ));
                }
                let action =
                    materialize_scoped_provided_action(provider, &provided, &mut nonterminals)?;
                if let Some(existing) = materialized.as_ref() {
                    if existing != &action {
                        return Err(format!(
                            "ordinary terminal {terminal} has distinct provider actions in state {state}"
                        ));
                    }
                } else {
                    materialized = Some(action);
                }
            }
            if let Some(action) = materialized {
                if matches!(action, Action::Skip) {
                    skip_terminals.insert(terminal);
                }
                rows[state as usize].push((terminal, action));
            }
        }
    }

    // A semantic control symbol must keep one terminal ID across every state.
    // In particular, a FINISH/ENTRY reduction continues by looking up the same
    // terminal after its goto; assigning IDs per cell would cut that reduction
    // chain before the eventual RETURN/CALL action.
    let mut provider_controls = Vec::<ScopedParserSymbol>::new();
    for state in 0..num_states {
        let mut symbols = SmallVec::<[ScopedParserSymbol; 4]>::new();
        provider.control_symbols(state, &mut symbols);
        for symbol in symbols {
            if !provider_controls.contains(&symbol) {
                provider_controls.push(symbol);
            }
        }
    }
    let provider_control_count = u32::try_from(provider_controls.len())
        .map_err(|_| "scoped provider control-terminal coordinate overflow".to_owned())?;
    let mut next_control = ordinary_terminals
        .checked_add(provider_control_count)
        .ok_or_else(|| "scoped provider control-terminal coordinate overflow".to_owned())?;
    let mut control_terminals = (ordinary_terminals..next_control).collect::<BTreeSet<_>>();

    for state in 0..num_states {
        for (control_index, &symbol) in provider_controls.iter().enumerate() {
            let Some(provided) = provider.action(state, symbol) else {
                continue;
            };
            let primary =
                materialize_scoped_provided_action(provider, &provided, &mut nonterminals)?;
            let terminal = ordinary_terminals + control_index as u32;
            rows[state as usize].push((terminal, primary));
            for shift in provided.extra_stack_shifts {
                let terminal = next_control;
                next_control = next_control.checked_add(1).ok_or_else(|| {
                    "scoped provider control-terminal coordinate overflow".to_owned()
                })?;
                control_terminals.insert(terminal);
                rows[state as usize].push((terminal, Action::StackShifts(vec![shift])));
            }
        }
    }

    let mut goto_pairs = nonterminals
        .iter()
        .map(|(&(scope, local), &global)| (global, scope, local))
        .collect::<Vec<_>>();
    goto_pairs.sort_unstable();
    let goto = (0..num_states)
        .map(|state| {
            GotoRow::from_iter(goto_pairs.iter().filter_map(|&(global, scope, local)| {
                provider
                    .goto_target(scope, state, local)
                    .map(|target| (global, target))
            }))
        })
        .collect::<Vec<_>>();
    let action = rows
        .into_iter()
        .map(ActionRow::from_iter)
        .collect::<Vec<_>>();
    let mut table = GLRTable {
        action,
        goto,
        num_states,
        num_terminals: next_control,
        num_rules: 0,
        rules: Vec::new(),
        nonterminal_display_names: Vec::new(),
        construction: GlrTableConstruction::LegacyRowBisim,
        admission_policy: AdmissionPolicy::RowPresenceExact,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals,
        skip_terminals,
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    table.rebuild_advance_rows_from_actions();
    table.rebuild_unconditional_advance_rows();
    table.rebuild_guarded_shift_index();
    table.eliminate_control_terminals_exact()?;
    // Private controls are gone from every action row. Shrink the externally
    // visible terminal domain back to the real composed terminal coordinate so
    // downstream parser-DWA compilation sees exactly its existing alphabet.
    table.num_terminals = ordinary_terminals;
    table.rebuild_advance_rows_from_actions();
    table.rebuild_unconditional_advance_rows();
    table.rebuild_guarded_shift_index();
    Ok(table)
}

/// Zero-remap adapter for an ordinary standalone table.
pub struct GLRTableActionProvider<'a> {
    table: &'a GLRTable,
}

impl<'a> GLRTableActionProvider<'a> {
    #[inline]
    pub fn new(table: &'a GLRTable) -> Self {
        Self { table }
    }
}

impl ParserActionProvider for GLRTableActionProvider<'_> {
    type Symbol = TerminalID;

    #[inline]
    fn action(&self, state: u32, symbol: TerminalID) -> Option<ProvidedAction<'_>> {
        self.table.action(state, symbol).map(|action| ProvidedAction {
            action: ProvidedActionRef::Local { scope: 0, action },
            reduction_scope: 0,
            extra_stack_shifts: SmallVec::new(),
        })
    }

    #[inline]
    fn scope_state(&self, _scope: u32, local_state: u32) -> Option<u32> {
        (local_state < self.table.num_states).then_some(local_state)
    }

    #[inline]
    fn goto_target(
        &self,
        _scope: u32,
        goto_from: u32,
        nonterminal: u32,
    ) -> Option<(u32, bool)> {
        self.table.goto_target(goto_from, nonterminal)
    }

    #[inline]
    fn state_count_hint(&self) -> usize {
        self.table.num_states as usize
    }
}

/// Read-only source of intact component LR tables for compact composed parsing.
/// The runtime can implement this over its component wrappers without building
/// a temporary `Vec<&GLRTable>`.
pub trait ParserComponentTableSource {
    fn component_count(&self) -> usize;
    fn component_table(&self, component: u32) -> Option<&GLRTable>;

    /// Optional terminal whose parser effect is identity throughout this
    /// component scope. Standalone constraints normally implement `ignore`
    /// outside their LR rows; composition makes that scope explicit here.
    #[inline]
    fn component_ignore_terminal(&self, _component: u32) -> Option<TerminalID> {
        None
    }
}

impl<const N: usize> ParserComponentTableSource for [&GLRTable; N] {
    #[inline]
    fn component_count(&self) -> usize {
        N
    }

    #[inline]
    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.get(component as usize).copied()
    }
}

impl ParserComponentTableSource for [&GLRTable] {
    #[inline]
    fn component_count(&self) -> usize {
        self.len()
    }

    #[inline]
    fn component_table(&self, component: u32) -> Option<&GLRTable> {
        self.get(component as usize).copied()
    }
}

enum ProviderStateOffsets<'a> {
    Owned(Vec<u32>),
    Borrowed(&'a [u32]),
}

impl ProviderStateOffsets<'_> {
    #[inline]
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Owned(offsets) => offsets,
            Self::Borrowed(offsets) => offsets,
        }
    }
}

/// Compact `u32` action provider over the disjoint union of intact component
/// LR-state sets. Offsets encode scope but do not form or require a merged LR
/// table. Ordinary actions remain borrowed and local.
pub struct DisjointComponentActionProvider<'a, S: ParserComponentTableSource + ?Sized> {
    components: &'a S,
    links: &'a [ScopedSubgrammarLink],
    state_offsets: ProviderStateOffsets<'a>,
    total_states: u32,
}

impl<'a, S: ParserComponentTableSource + ?Sized> DisjointComponentActionProvider<'a, S> {
    pub fn new(components: &'a S, links: &'a [ScopedSubgrammarLink]) -> Result<Self, String> {
        let mut offsets = Vec::with_capacity(components.component_count());
        let mut total_states = 0u32;
        for component in 0..components.component_count() {
            let table = components
                .component_table(component as u32)
                .ok_or_else(|| format!("missing parser component {component}"))?;
            offsets.push(total_states);
            total_states = total_states
                .checked_add(table.num_states)
                .ok_or_else(|| "scoped parser-state coordinate overflow".to_owned())?;
        }
        Self::from_offsets(components, links, ProviderStateOffsets::Owned(offsets), total_states)
    }

    /// Zero-allocation runtime constructor using offsets precomputed with the
    /// component wrapper metadata.
    pub fn with_state_offsets(
        components: &'a S,
        links: &'a [ScopedSubgrammarLink],
        state_offsets: &'a [u32],
    ) -> Result<Self, String> {
        if state_offsets.len() != components.component_count() {
            return Err("segmented parser-state offset/component count mismatch".to_owned());
        }
        let mut expected = 0u32;
        for component in 0..components.component_count() {
            if state_offsets[component] != expected {
                return Err(format!(
                    "segmented parser-state offset {component} is {}, expected {expected}",
                    state_offsets[component],
                ));
            }
            let table = components
                .component_table(component as u32)
                .ok_or_else(|| format!("missing parser component {component}"))?;
            expected = expected
                .checked_add(table.num_states)
                .ok_or_else(|| "scoped parser-state coordinate overflow".to_owned())?;
        }
        Self::from_offsets(
            components,
            links,
            ProviderStateOffsets::Borrowed(state_offsets),
            expected,
        )
    }

    fn from_offsets(
        components: &'a S,
        links: &'a [ScopedSubgrammarLink],
        state_offsets: ProviderStateOffsets<'a>,
        total_states: u32,
    ) -> Result<Self, String> {
        for (index, link) in links.iter().enumerate() {
            let parent = components
                .component_table(link.parent_component)
                .ok_or_else(|| format!("link {index} references missing parent component"))?;
            let child = components
                .component_table(link.child_component)
                .ok_or_else(|| format!("link {index} references missing child component"))?;
            if link.slot_terminal >= parent.num_terminals || link.child_start >= child.num_states {
                return Err(format!("link {index} references an invalid local coordinate"));
            }
        }
        Ok(Self {
            components,
            links,
            state_offsets,
            total_states,
        })
    }

    #[inline]
    pub fn scoped_state(&self, component: u32, local_state: u32) -> Option<u32> {
        let table = self.components.component_table(component)?;
        if local_state >= table.num_states {
            return None;
        }
        self.state_offsets
            .as_slice()
            .get(component as usize)?
            .checked_add(local_state)
    }

    #[inline]
    pub fn decode_state(&self, state: u32) -> Option<(u32, u32)> {
        if state >= self.total_states {
            return None;
        }
        let offsets = self.state_offsets.as_slice();
        let component = offsets
            .partition_point(|&offset| offset <= state)
            .checked_sub(1)?;
        Some((component as u32, state - offsets[component]))
    }

    #[inline]
    fn table(&self, component: u32) -> Option<&GLRTable> {
        self.components.component_table(component)
    }
}

impl<S: ParserComponentTableSource + ?Sized> ParserActionProvider
    for DisjointComponentActionProvider<'_, S>
{
    type Symbol = ScopedParserSymbol;

    fn action(&self, state: u32, symbol: Self::Symbol) -> Option<ProvidedAction<'_>> {
        let (component, local_state) = self.decode_state(state)?;
        let table = self.table(component)?;
        match symbol {
            ScopedParserSymbol::Terminal {
                component: symbol_component,
                terminal,
            } if symbol_component == component => {
                if self.components.component_ignore_terminal(component) == Some(terminal) {
                    return Some(ProvidedAction {
                        action: ProvidedActionRef::Identity,
                        reduction_scope: component,
                        extra_stack_shifts: SmallVec::new(),
                    });
                }
                let action = table.action(local_state, terminal)?;
                Some(ProvidedAction {
                    action: ProvidedActionRef::Local {
                        scope: component,
                        action,
                    },
                    reduction_scope: component,
                    extra_stack_shifts: SmallVec::new(),
                })
            }
            ScopedParserSymbol::Entry { link } => {
                let link = *self.links.get(link as usize)?;
                if link.parent_component != component {
                    return None;
                }
                let action = table.action(local_state, link.slot_terminal)?;
                match action {
                    Action::Shift(parent_target, replace)
                    | Action::Split {
                        shift: Some((parent_target, replace)),
                        reduces: _,
                        accept: false,
                    } if action.reduce_count() == 0 => Some(ProvidedAction {
                        action: ProvidedActionRef::Call {
                            parent_target: self.scoped_state(component, *parent_target)?,
                            child_start: self.scoped_state(link.child_component, link.child_start)?,
                            replace: *replace,
                        },
                        reduction_scope: component,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    Action::Reduce(..)
                    | Action::Split {
                        shift: None,
                        reduces: _,
                        accept: false,
                    } if action.reduce_count() != 0 => Some(ProvidedAction {
                        action: ProvidedActionRef::Local {
                            scope: component,
                            action,
                        },
                        reduction_scope: component,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    _ => None,
                }
            }
            ScopedParserSymbol::Finish {
                component: symbol_component,
            } if symbol_component == component => {
                let link = *self.links.iter().find(|link| link.child_component == component)?;
                let nullable_start = link.child_start_nullable && local_state == link.child_start;
                let eof_action = table.action(local_state, EOF);
                let mut provided = match eof_action {
                    Some(Action::Accept) => ProvidedAction {
                        action: ProvidedActionRef::Return { pop: link.return_pop },
                        reduction_scope: component,
                        extra_stack_shifts: SmallVec::new(),
                    },
                    Some(action) => ProvidedAction {
                        action: ProvidedActionRef::Local {
                            scope: component,
                            action,
                        },
                        reduction_scope: component,
                        extra_stack_shifts: SmallVec::new(),
                    },
                    None if nullable_start => ProvidedAction {
                        action: ProvidedActionRef::Return { pop: 1 },
                        reduction_scope: component,
                        extra_stack_shifts: SmallVec::new(),
                    },
                    None => return None,
                };
                if nullable_start && eof_action.is_some() {
                    provided.extra_stack_shifts.push(StackShift {
                        pop: 1,
                        pushes: Vec::new(),
                    });
                }
                Some(provided)
            }
            _ => None,
        }
    }

    #[inline]
    fn scope_state(&self, scope: u32, local_state: u32) -> Option<u32> {
        self.scoped_state(scope, local_state)
    }

    fn goto_target(
        &self,
        scope: u32,
        goto_from: u32,
        nonterminal: u32,
    ) -> Option<(u32, bool)> {
        let (component, local_goto_from) = self.decode_state(goto_from)?;
        if component != scope {
            return None;
        }
        let (target, replace) = self.table(component)?.goto_target(local_goto_from, nonterminal)?;
        Some((self.scoped_state(component, target)?, replace))
    }

    fn control_symbols(&self, state: u32, out: &mut SmallVec<[Self::Symbol; 4]>) {
        let Some((component, local_state)) = self.decode_state(state) else {
            return;
        };
        let Some(table) = self.table(component) else {
            return;
        };
        // A retained component may itself already be a composed constraint.
        // Its local linker controls remain ordinary zero-width actions in that
        // component table during the migration to recursively literal wrappers.
        // Scope them exactly like any other local terminal rather than flattening
        // or rewriting the component.
        for &terminal in &table.control_terminals {
            out.push(ScopedParserSymbol::Terminal { component, terminal });
        }
        for (index, link) in self.links.iter().enumerate() {
            if link.parent_component == component
                && table.action(local_state, link.slot_terminal).is_some()
            {
                out.push(ScopedParserSymbol::Entry { link: index as u32 });
            }
        }
        if self.links.iter().any(|link| {
            link.child_component == component
                && (table.action(local_state, EOF).is_some()
                    || (link.child_start_nullable && local_state == link.child_start))
        }) {
            out.push(ScopedParserSymbol::Finish { component });
        }
    }

    #[inline]
    fn state_count_hint(&self) -> usize {
        self.total_states as usize
    }
}

type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;
type ReduceBranches = SmallVec<[(ParserGSS, u32, bool); 4]>;
type FloorCrossShift = (u32, u32, bool);

const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 64;
const GUARDED_STACK_TO_STACKS_MAX_DEPTH: usize = 64;
const SMALL_REDUCE_FANOUT_COLLAPSE_MAX_BRANCHES: usize = 8;
const ADVANCE_ASSERT_DISTRIBUTIVITY: u8 = 1 << 0;
const ADVANCE_ASSERT_FAST_PATH_EQUIVALENCE: u8 = 1 << 1;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn assert_row_presence_exact_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("GLRMASK_ASSERT_ROW_PRESENCE_EXACT"))
}

fn advance_assertion_flags() -> u8 {
    static FLAGS: OnceLock<u8> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        let mut flags = 0;
        if env_flag_enabled("GLRMASK_ASSERT_ADVANCE_DISTRIBUTIVITY") {
            flags |= ADVANCE_ASSERT_DISTRIBUTIVITY;
        }
        if env_flag_enabled("GLRMASK_ASSERT_ADVANCE_FAST_PATH_EQUIVALENCE") {
            flags |= ADVANCE_ASSERT_FAST_PATH_EQUIVALENCE;
        }
        flags
    })
}

fn guarded_stack_to_stacks_fallback_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| env_flag_enabled("GLRMASK_DISABLE_GUARDED_STACK_TO_STACKS_FALLBACK"))
}

fn stack_effect_to_stacks_fallback_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| env_flag_enabled("GLRMASK_DISABLE_STACK_EFFECT_TO_STACKS_FALLBACK"))
}

fn advance_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("GLRMASK_PROFILE_ADVANCE_TRACE"))
}

fn nondet_wave_specializations_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("GLRMASK_ENABLE_NONDET_WAVE_SPECIALIZATIONS"))
}

fn trace_action_kind(action: Option<&Action>) -> &'static str {
    match action {
        Some(Action::Shift(..)) => "shift",
        Some(Action::StackShifts(..)) => "stack-shifts",
        Some(Action::ReplaceShifts(..)) => "replace-shifts",
        Some(Action::GuardedStackShifts(..)) => "guarded-stack-shifts",
        Some(Action::Reduce(..)) => "reduce",
        Some(Action::Split { accept: true, .. }) => "split-accept",
        Some(Action::Split { .. }) => "split",
        Some(Action::Accept) => "accept",
        Some(Action::Skip) => "skip",
        None => "none",
    }
}

fn trace_reduce_summary(
    table: &GLRTable,
    gss: &ParserGSS,
    lhs_nt: u32,
    pop_len: usize,
) -> AdvanceTraceReduce {
    let mut goto_sources = Vec::new();
    let mut goto_targets = Vec::new();
    for (goto_from, _) in reduce_sources_from_isolated(gss, pop_len) {
        goto_sources.push(goto_from);
        if let Some((target_state, replace)) = table.goto_target(goto_from, lhs_nt) {
            goto_targets.push(AdvanceTraceGoto {
                source_state: goto_from,
                target_state,
                replace,
            });
        }
    }
    goto_sources.sort_unstable();
    goto_sources.dedup();
    goto_targets.sort_by_key(|entry| (entry.source_state, entry.target_state, entry.replace));
    goto_targets.dedup_by(|left, right| {
        left.source_state == right.source_state
            && left.target_state == right.target_state
            && left.replace == right.replace
    });
    AdvanceTraceReduce {
        lhs_nt,
        lhs_name: table.nonterminal_display_name(lhs_nt).map(str::to_owned),
        pop_len: pop_len as u32,
        goto_sources,
        goto_targets,
    }
}

fn trace_action_summary(
    table: &GLRTable,
    source_state: u32,
    gss: &ParserGSS,
    action: Option<&Action>,
) -> AdvanceTraceStep {
    match action {
        Some(Action::Shift(target, replace)) => AdvanceTraceStep {
            source_state,
            action_kind: trace_action_kind(action).to_string(),
            shift_target: Some(*target),
            shift_replace: Some(*replace),
            reduces: Vec::new(),
        },
        Some(Action::StackShifts(..))
        | Some(Action::ReplaceShifts(..))
        | Some(Action::GuardedStackShifts(..))
        | Some(Action::Skip)
        | Some(Action::Accept)
        | None => {
            AdvanceTraceStep {
                source_state,
                action_kind: trace_action_kind(action).to_string(),
                shift_target: None,
                shift_replace: None,
                reduces: Vec::new(),
            }
        }
        Some(Action::Reduce(lhs_nt, pop_len)) => AdvanceTraceStep {
            source_state,
            action_kind: trace_action_kind(action).to_string(),
            shift_target: None,
            shift_replace: None,
            reduces: vec![trace_reduce_summary(table, gss, *lhs_nt, *pop_len as usize)],
        },
        Some(Action::Split { shift, reduces, accept }) => AdvanceTraceStep {
            source_state,
            action_kind: if *accept { "split-accept" } else { "split" }.to_string(),
            shift_target: shift.map(|(target, _)| target),
            shift_replace: shift.map(|(_, replace)| replace),
            reduces: reduces
                .iter()
                .map(|&(lhs_nt, pop_len)| trace_reduce_summary(table, gss, lhs_nt, pop_len as usize))
                .collect(),
        },
    }
}

enum AdvancedBranch {
    Stack(VirtualStack<u32, TerminalsDisallowed>),
    Gss(ParserGSS),
}

impl AdvancedBranch {
    fn into_gss(self) -> ParserGSS {
        match self {
            AdvancedBranch::Stack(stack) => stack.into_gss(),
            AdvancedBranch::Gss(gss) => gss,
        }
    }
}

pub fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    let advanced = advance_stacks_with_control_closure(table, stack.clone(), token);
    if !table.control_terminals.is_empty() {
        return advanced;
    }
    assert_advance_oracles(
        advance_assertion_flags(),
        table,
        stack,
        token,
        &advanced,
    );
    advanced
}

/// Like `advance_stacks` but takes ownership of the GSS, avoiding an
/// unnecessary Arc clone when the caller doesn't need the original.
pub fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {
    if !table.control_terminals.is_empty() {
        return advance_stacks_with_control_closure(table, stack, token);
    }
    let assertion_flags = advance_assertion_flags();
    if assertion_flags != 0 {
        let before = stack.clone();
        let advanced = advance_stacks_core(table, stack, token);
        assert_advance_oracles(assertion_flags, table, &before, token, &advanced);
        advanced
    } else {
        advance_stacks_core(table, stack, token)
    }
}

/// Compute the reflexive-transitive closure of linker-internal zero-width
/// terminals. The original frontier remains live: at a subgrammar call site a
/// parser may consume parent trivia before entering, or enter first and consume
/// child trivia. Control actions are compiler generated and must reach a finite
/// fixed point; a positive stack-growth cycle is an invalid composed table.
fn control_closure(table: &GLRTable, stack: &ParserGSS) -> ParserGSS {
    if table.control_terminals.is_empty() || stack.is_empty() {
        return stack.clone();
    }

    let mut closure = stack.clone();
    let pass_limit = (table.num_states as usize)
        .saturating_mul(table.control_terminals.len().max(1))
        .saturating_add(2);
    for _ in 0..pass_limit {
        let mut additions = ParserGSS::empty();
        for &control in &table.control_terminals {
            merge_into(
                &mut additions,
                advance_stacks_core(table, closure.clone(), control),
            );
        }
        if additions.is_empty() {
            return closure;
        }
        let next = closure.merge(&additions);
        if next == closure {
            return closure;
        }
        closure = next;
    }
    panic!(
        "linker control closure did not converge: states={} controls={:?}",
        table.num_states, table.control_terminals,
    );
}

fn advance_stacks_with_control_closure(
    table: &GLRTable,
    stack: ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    if table.control_terminals.is_empty() {
        return advance_stacks_core(table, stack, token);
    }
    let before = control_closure(table, &stack);
    let advanced = advance_stacks_core(table, before, token);
    control_closure(table, &advanced)
}

/// Close a runtime parser frontier under linker-internal zero-width controls.
/// Ordinary callers should use [`advance_stacks`]; this is exposed for runtimes
/// which maintain control-closed frontiers as a persistent invariant.
pub fn close_control_stacks(table: &GLRTable, stack: &ParserGSS) -> ParserGSS {
    control_closure(table, stack)
}

/// Advance a frontier which is already closed under linker controls, then
/// restore the invariant. This avoids recomputing the pre-terminal closure on
/// every lexeme in composed-constraint runtime commit.
pub fn advance_control_closed_stacks(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    if table.control_terminals.is_empty() {
        return advance_stacks_core(table, stack.clone(), token);
    }
    let advanced = advance_stacks_core(table, stack.clone(), token);
    control_closure(table, &advanced)
}

/// Owned variant of [`advance_control_closed_stacks`].
pub fn advance_control_closed_stacks_owned(
    table: &GLRTable,
    stack: ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    if table.control_terminals.is_empty() {
        return advance_stacks_core(table, stack, token);
    }
    let advanced = advance_stacks_core(table, stack, token);
    control_closure(table, &advanced)
}

fn normalized_concrete_stacks(
    gss: &ParserGSS,
) -> Vec<(Vec<u32>, TerminalsDisallowed)> {
    let mut normalized: Vec<(Vec<u32>, TerminalsDisallowed)> = Vec::new();
    for (stack, acc) in gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit") {
        if let Some((_, existing_acc)) = normalized
            .iter_mut()
            .find(|(existing_stack, _)| *existing_stack == stack)
        {
            *existing_acc = existing_acc.merge(&acc);
        } else {
            normalized.push((stack, acc));
        }
    }
    normalized.sort_by(|left, right| left.0.cmp(&right.0));
    normalized
}

fn merge_concrete_path_accumulator(
    paths: &mut BTreeMap<Vec<u32>, TerminalsDisallowed>,
    stack: Vec<u32>,
    acc: TerminalsDisallowed,
) -> bool {
    if let Some(existing) = paths.get_mut(&stack) {
        let merged = existing.merge(&acc);
        if *existing == merged {
            return false;
        }
        *existing = merged;
        true
    } else {
        paths.insert(stack, acc);
        true
    }
}

fn apply_concrete_stack_effect(
    stack: &[u32],
    pop: usize,
    pushes: &[u32],
) -> Option<Vec<u32>> {
    // An empty concrete parser stack is still a live GSS path. Stack effects
    // are atomic and may therefore pop the final state before pushing a new
    // sequence. Only an actual underflow kills the effect.
    if pop > stack.len() {
        return None;
    }
    let mut next = stack[..stack.len() - pop].to_vec();
    next.extend_from_slice(pushes);
    Some(next)
}

fn concrete_stack_satisfies_guards(stack: &[u32], guards: &[StackShiftGuard]) -> bool {
    guards.iter().all(|guard| {
        let pop = guard.pop as usize;
        pop < stack.len()
            && guard
                .states
                .binary_search(&stack[stack.len() - 1 - pop])
                .is_ok()
    })
}

fn enqueue_concrete_reduction(
    table: &GLRTable,
    stack: &[u32],
    acc: &TerminalsDisallowed,
    nt: u32,
    rhs_len: usize,
    closure: &mut BTreeMap<Vec<u32>, TerminalsDisallowed>,
    queue: &mut VecDeque<Vec<u32>>,
) {
    if rhs_len >= stack.len() {
        return;
    }
    let mut base = stack[..stack.len() - rhs_len].to_vec();
    let goto_from = *base.last().expect("reduction preserves a parser state");
    let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
        return;
    };
    if is_replace {
        let Some(next) = apply_concrete_stack_effect(&base, 1, &[target]) else {
            return;
        };
        base = next;
    } else {
        base.push(target);
    }
    if merge_concrete_path_accumulator(closure, base.clone(), acc.clone()) {
        queue.push_back(base);
    }
}

fn advance_concrete_stacks_reference(
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    let mut closure = BTreeMap::<Vec<u32>, TerminalsDisallowed>::new();
    let mut queue = VecDeque::<Vec<u32>>::new();
    for (stack, acc) in before.to_stacks(4_096).expect("stack enumeration exceeded explicit limit") {
        if merge_concrete_path_accumulator(&mut closure, stack.clone(), acc) {
            queue.push_back(stack);
        }
    }

    let mut shifted = BTreeMap::<Vec<u32>, TerminalsDisallowed>::new();
    while let Some(stack) = queue.pop_front() {
        let Some(acc) = closure.get(&stack).cloned() else {
            continue;
        };
        let Some(&state) = stack.last() else {
            continue;
        };
        let Some(action) = table.action(state, token) else {
            continue;
        };

        match action {
            Action::Skip => {
                merge_concrete_path_accumulator(&mut shifted, stack, acc);
            }
            Action::Shift(target, is_replace) => {
                let pop = usize::from(*is_replace);
                if let Some(next) = apply_concrete_stack_effect(&stack, pop, &[*target]) {
                    merge_concrete_path_accumulator(&mut shifted, next, acc);
                }
            }
            Action::ReplaceShifts(targets) => {
                for target in targets.iter() {
                    if let Some(next) = apply_concrete_stack_effect(&stack, 1, &[*target]) {
                        merge_concrete_path_accumulator(&mut shifted, next, acc.clone());
                    }
                }
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    if let Some(next) = apply_concrete_stack_effect(
                        &stack,
                        shift.pop as usize,
                        &shift.pushes,
                    ) {
                        merge_concrete_path_accumulator(&mut shifted, next, acc.clone());
                    }
                }
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    if concrete_stack_satisfies_guards(&stack, &shift.guards)
                        && let Some(next) = apply_concrete_stack_effect(
                            &stack,
                            shift.pop as usize,
                            &shift.pushes,
                        )
                    {
                        merge_concrete_path_accumulator(&mut shifted, next, acc.clone());
                    }
                }
            }
            Action::Reduce(nt, rhs_len) => enqueue_concrete_reduction(
                table,
                &stack,
                &acc,
                *nt,
                *rhs_len as usize,
                &mut closure,
                &mut queue,
            ),
            Action::Split {
                shift,
                reduces,
                accept: _,
            } => {
                if let Some((target, is_replace)) = shift {
                    let is_replace = *is_replace
                        && !table.forwarded_shifts.contains(&(state, token));
                    if let Some(next) = apply_concrete_stack_effect(
                        &stack,
                        usize::from(is_replace),
                        &[*target],
                    ) {
                        merge_concrete_path_accumulator(&mut shifted, next, acc.clone());
                    }
                }
                for &(nt, rhs_len) in reduces {
                    enqueue_concrete_reduction(
                        table,
                        &stack,
                        &acc,
                        nt,
                        rhs_len as usize,
                        &mut closure,
                        &mut queue,
                    );
                }
            }
            Action::Accept => {}
        }
    }

    if shifted.is_empty() {
        ParserGSS::empty()
    } else {
        let stacks = shifted.into_iter().collect::<Vec<_>>();
        ParserGSS::from_stacks(&stacks)
    }
}

#[cold]
fn assert_advance_matches_concrete_reference(
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
    actual: &ParserGSS,
) {
    let expected = advance_concrete_stacks_reference(table, before, token);
    assert_eq!(
        normalized_concrete_stacks(actual),
        normalized_concrete_stacks(&expected),
        "parser fast-path advance mismatch for terminal {token}: construction={:?} before={:?}",
        table.construction,
        normalized_concrete_stacks(before),
    );
}

#[cold]
fn assert_advance_distributes_over_concrete_paths(
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
    actual: &ParserGSS,
) {
    let concrete_paths = before.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
    if concrete_paths.len() <= 1 {
        return;
    }

    let mut expected = ParserGSS::empty();
    for (stack, acc) in concrete_paths {
        merge_into(
            &mut expected,
            advance_stacks_core(table, ParserGSS::from_single_stack(stack, acc), token),
        );
    }

    assert_eq!(
        normalized_concrete_stacks(actual),
        normalized_concrete_stacks(&expected),
        "parser advance is not distributive over GSS path union for terminal {token}: construction={:?} before={:?}",
        table.construction,
        normalized_concrete_stacks(before),
    );
}

#[inline]
fn assert_advance_oracles(
    assertion_flags: u8,
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
    actual: &ParserGSS,
) {
    if assertion_flags & ADVANCE_ASSERT_FAST_PATH_EQUIVALENCE != 0 {
        assert_advance_matches_concrete_reference(table, before, token, actual);
    }
    if assertion_flags & ADVANCE_ASSERT_DISTRIBUTIVITY != 0 {
        assert_advance_distributes_over_concrete_paths(table, before, token, actual);
    }
}

pub fn advance_stacks_profiled(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> (ParserGSS, AdvanceProfile) {
    if !table.control_terminals.is_empty() {
        let before = control_closure(table, stack);
        let (advanced, profile) = advance_stacks_profiled_core(table, &before, token);
        return (control_closure(table, &advanced), profile);
    }
    let (advanced, profile) = advance_stacks_profiled_core(table, stack, token);
    assert_advance_oracles(
        advance_assertion_flags(),
        table,
        stack,
        token,
        &advanced,
    );
    (advanced, profile)
}

fn advance_stacks_profiled_core(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> (ParserGSS, AdvanceProfile) {
    use std::time::Instant;

    let total_start = Instant::now();
    let clone_start = Instant::now();
    let mut gss = stack.clone();
    let mut profile = AdvanceProfile {
        clone_ns: clone_start.elapsed().as_nanos() as u64,
        top_states: stack.peek_values().len() as u32,
        gss_depth: stack.max_depth(),
        vstack_len: stack.try_virtual_stack().map_or(0, |vstack| vstack.len() as u32),
        trace: advance_trace_enabled().then(AdvanceTrace::default),
        ..AdvanceProfile::default()
    };

    let fast_path_start = Instant::now();
    if let Some(state) = gss.single_exclusive_top_value() {
        match table.action(state, token) {
            Some(Action::Shift(target, is_replace)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(
                        table,
                        state,
                        &gss,
                        Some(&Action::Shift(*target, *is_replace)),
                    ));
                }
                profile.pure_shift = true;
                profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                let apply_start = Instant::now();
                let shifted = if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                };
                profile.stack_shift_apply_ns = apply_start.elapsed().as_nanos() as u64;
                profile.total_ns = total_start.elapsed().as_nanos() as u64;
                return (shifted, profile);
            }
            Some(Action::StackShifts(shifts)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(table, state, &gss, Some(&Action::StackShifts(shifts.clone()))));
                }
                profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                let apply_start = Instant::now();
                let shifted = apply_stack_shifts(gss, shifts);
                profile.stack_shift_apply_ns = apply_start.elapsed().as_nanos() as u64;
                profile.total_ns = total_start.elapsed().as_nanos() as u64;
                return (shifted, profile);
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(
                        table,
                        state,
                        &gss,
                        Some(&Action::GuardedStackShifts(shifts.clone())),
                    ));
                }
                profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                let apply_start = Instant::now();
                let shifted = apply_guarded_stack_shifts(
                    gss,
                    shifts,
                    table.guarded_shift_index(state, token),
                );
                profile.stack_shift_apply_ns = apply_start.elapsed().as_nanos() as u64;
                profile.total_ns = total_start.elapsed().as_nanos() as u64;
                return (shifted, profile);
            }
            Some(Action::Reduce(nt, len)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(
                        table,
                        state,
                        &gss,
                        Some(&Action::Reduce(*nt, *len)),
                    ));
                }
                if let Some(shifted) =
                    try_advance_uniform_deterministic_frontier(table, &gss, token)
                {
                    profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                    profile.stack_shift_apply_ns = profile.fast_path_ns;
                    profile.total_ns = total_start.elapsed().as_nanos() as u64;
                    return (shifted, profile);
                }
                if gss.single_predecessor_below_top(&state).is_none()
                    && let Some(shifted) =
                        try_advance_bounded_concrete_paths(table, &gss, token)
                {
                    profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                    profile.stack_shift_apply_ns = profile.fast_path_ns;
                    profile.total_ns = total_start.elapsed().as_nanos() as u64;
                    return (shifted, profile);
                }
                if *len == 1
                    && let Some(shifted) =
                        try_advance_single_top_pop1_reduce_frontier(table, &gss, token)
                {
                    profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                    profile.stack_shift_apply_ns = profile.fast_path_ns;
                    profile.total_ns = total_start.elapsed().as_nanos() as u64;
                    return (shifted, profile);
                }
                if let Some(shifted) = try_collapse_small_reduce_fanout(table, &gss, token) {
                    profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                    profile.stack_shift_apply_ns = profile.fast_path_ns;
                    profile.total_ns = total_start.elapsed().as_nanos() as u64;
                    return (shifted, profile);
                }
            }
            _ => {}
        }
    }
    profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;

    let frontier_start = Instant::now();
    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        profile.pure_shift = true;
        profile.stack_shift_apply_ns = frontier_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let mixed_top_replace_start = Instant::now();
    if let Some(shifted) = try_advance_mixed_top_replace_wave(table, &gss, token) {
        profile.stack_shift_apply_ns += mixed_top_replace_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let mixed_reduce_guarded_start = Instant::now();
    if let Some(shifted) =
        try_advance_pop1_reduce_guarded_stackshift_wave(table, &gss, token)
    {
        profile.stack_shift_apply_ns +=
            mixed_reduce_guarded_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let bounded_reduce_start = Instant::now();
    if let Some(shifted) =
        try_advance_bounded_concrete_paths(table, &gss, token)
    {
        profile.stack_shift_apply_ns += bounded_reduce_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let single_active_start = Instant::now();
    if let Some(shifted) =
        try_advance_single_active_pop1_stackshift_wave(table, &gss, token)
    {
        profile.pure_shift = true;
        profile.stack_shift_apply_ns += single_active_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let det_start = Instant::now();
    let det_ok = advance_deterministically_profiled(table, &mut gss, token, &mut profile);
    profile.det_ns = det_start.elapsed().as_nanos() as u64;
    if det_ok {
        profile.deterministic_finished = true;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (gss, profile);
    }

    let post_det_frontier_start = Instant::now();
    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        profile.pure_shift = true;
        profile.stack_shift_apply_ns += post_det_frontier_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let nondet_start = Instant::now();
    profile.nondeterministic_entered = true;
    let gss = advance_nondeterministically_profiled(table, gss, token, &mut profile);
    profile.nondet_ns = nondet_start.elapsed().as_nanos() as u64;
    profile.total_ns = total_start.elapsed().as_nanos() as u64;
    (gss, profile)
}

/// Advance the GSS by one token.
///
/// First try the deterministic single-chain path: repeatedly reduce a flat LR
/// stack, and finish immediately if that path ends in a pure shift.
///
/// If the frontier is ambiguous, or the deterministic path stops without a
/// pure shift, fall back to the GLR path: build the reduce closure to a
/// fixpoint and return the shifted next frontier.
fn advance_stacks_core(table: &GLRTable, mut gss: ParserGSS, token: TerminalID) -> ParserGSS {
    if let Some(state) = gss.single_exclusive_top_value() {
        if let Some(Action::Shift(target, is_replace)) = table.action(state, token) {
            return if *is_replace {
                gss.popn(1).push(*target)
            } else {
                gss.push(*target)
            };
        }
        if let Some(Action::ReplaceShifts(targets)) = table.action(state, token) {
            return apply_replace_shifts(gss, targets);
        }
        if let Some(Action::StackShifts(shifts)) = table.action(state, token) {
            return apply_stack_shifts(gss, shifts);
        }
        if let Some(Action::GuardedStackShifts(shifts)) = table.action(state, token) {
            return apply_guarded_stack_shifts(gss, shifts, table.guarded_shift_index(state, token));
        }
        if matches!(table.action(state, token), Some(Action::Reduce(..)))
            && let Some(shifted) =
                try_advance_uniform_deterministic_frontier(table, &gss, token)
        {
            return shifted;
        }
        if matches!(table.action(state, token), Some(Action::Reduce(..)))
            && gss.single_predecessor_below_top(&state).is_none()
            && let Some(shifted) =
                try_advance_bounded_concrete_paths(table, &gss, token)
        {
            return shifted;
        }
        if matches!(table.action(state, token), Some(Action::Reduce(..)))
            && let Some(shifted) =
                try_advance_single_top_pop1_reduce_frontier(table, &gss, token)
        {
            return shifted;
        }
        if matches!(table.action(state, token), Some(Action::Reduce(..)))
            && let Some(shifted) = try_collapse_small_reduce_fanout(table, &gss, token)
        {
            return shifted;
        }
    }

    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        return shifted;
    }

    if let Some(shifted) = try_advance_mixed_top_replace_wave(table, &gss, token) {
        return shifted;
    }

    if let Some(shifted) =
        try_advance_pop1_reduce_guarded_stackshift_wave(table, &gss, token)
    {
        return shifted;
    }

    if let Some(shifted) =
        try_advance_bounded_concrete_paths(table, &gss, token)
    {
        return shifted;
    }

    if let Some(shifted) =
        try_advance_single_active_pop1_stackshift_wave(table, &gss, token)
    {
        return shifted;
    }

    if advance_deterministically(table, &mut gss, token) {
        return gss;
    }

    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        return shifted;
    }

    advance_nondeterministically(table, gss, token)
}

fn try_collapse_small_reduce_fanout(
    table: &GLRTable,
    gss: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    // This optimization only applies when reducing exposes multiple isolated
    // branches. A virtual stack is a single concrete path, so the existing
    // branch construction would produce at most one branch and return `None`.
    if gss.try_virtual_stack().is_some() {
        return None;
    }

    let state = gss.single_exclusive_top_value()?;
    let Action::Reduce(nt, len) = table.action(state, token)? else {
        return None;
    };

    let branches = reduce_branches_from_isolated(table, gss, *nt, *len as usize);
    if branches.len() <= 1 || branches.len() > SMALL_REDUCE_FANOUT_COLLAPSE_MAX_BRANCHES {
        return None;
    }

    let mut collapsed: Option<ParserGSS> = None;
    for (base, target, is_replace) in branches {
        let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
        if !det_ok {
            return None;
        }

        let branch = branch.into_gss();
        if branch.is_empty() {
            return None;
        }

        if let Some(existing) = collapsed.as_ref() {
            if &branch != existing {
                return None;
            }
        } else {
            collapsed = Some(branch);
        }
    }

    collapsed
}

#[inline]
fn pure_frontier_shift(action: &Action) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, is_replace) => Some((*target, *is_replace)),
        Action::ReplaceShifts(targets) if targets.len() == 1 => Some((targets[0], true)),
        Action::StackShifts(shifts)
            if shifts.len() == 1 && shifts[0].pushes.len() == 1 && shifts[0].pop <= 1 =>
        {
            Some((shifts[0].pushes[0], shifts[0].pop == 1))
        }
        _ => None,
    }
}

fn effective_pure_frontier_shift(
    table: &GLRTable,
    state: u32,
    token: TerminalID,
    action: &Action,
) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, is_replace) => Some((
            *target,
            *is_replace && !table.forwarded_shifts.contains(&(state, token)),
        )),
        Action::ReplaceShifts(targets) if targets.len() == 1 => Some((targets[0], true)),
        Action::StackShifts(shifts)
            if shifts.len() == 1 && shifts[0].pushes.len() == 1 && shifts[0].pop <= 1 =>
        {
            Some((shifts[0].pushes[0], shifts[0].pop == 1))
        }
        _ => None,
    }
}

fn advance_pure_frontier_shifts(
    table: &GLRTable,
    gss: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = gss.peek_values();
    if states.len() <= 1 {
        return None;
    }

    let mut shifts: SmallVec<[(u32, u32, bool); 8]> = SmallVec::new();
    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        let (target, replace_top) = pure_frontier_shift(action)?;
        shifts.push((state, target, replace_top));
    }
    if shifts.is_empty() {
        return None;
    }
    if let Some(shifted) = gss.try_apply_selective_top_pure_shifts(shifts.iter().copied()) {
        return Some(shifted);
    }
    Some(gss.apply_top_pure_shifts(shifts))
}

/// Collapse a mixed frontier of pure shifts and pop-one reductions when every
/// reduction is immediately followed by a replace shift. In that shape the
/// whole wave is just a top-layer remap: the lower GSS children can be retained
/// directly instead of materialising reduction branches and merging them back.
fn try_advance_mixed_top_replace_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut remaps = SmallVec::<[(u32, u32, bool); 16]>::new();
    let mut saw_reduce = false;
    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        if let Some((target, replace_top)) =
            effective_pure_frontier_shift(table, state, token, action)
        {
            if remaps.len() == remaps.capacity() {
                return None;
            }
            remaps.push((state, target, replace_top));
            continue;
        }

        let nt = match action {
            Action::Reduce(nt, 1) => *nt,
            _ => return None,
        };
        saw_reduce = true;
        let Some(predecessor) = closure.single_predecessor_below_top(&state) else {
            return None;
        };
        let Some((goto_target, false)) = table.goto_target(predecessor, nt) else {
            // A missing goto kills this reduction branch. A replacing goto is
            // not a top-only remap and must use the general path.
            if table.goto_target(predecessor, nt).is_none() {
                continue;
            }
            return None;
        };
        let Some(next_action) = table.action(goto_target, token) else {
            continue;
        };
        let Some((target, true)) =
            effective_pure_frontier_shift(table, goto_target, token, next_action)
        else {
            return None;
        };
        if remaps.len() == remaps.capacity() {
            return None;
        }
        remaps.push((state, target, true));
    }

    if !saw_reduce {
        return None;
    }
    if remaps.is_empty() {
        return Some(ParserGSS::empty());
    }
    Some(closure.apply_top_pure_shifts(remaps))
}

/// Advance a small frontier whose live actions are deterministic reduction
/// chains (with dead alternatives allowed). Concrete paths are traversed into
/// inline buffers, advanced exactly, and rebuilt once with shared prefixes.
/// Larger, branching, or accumulator-heterogeneous frontiers decline to the
/// ordinary GSS interpreter.
/// Advance a deterministic reduction chain directly on the whole GSS while
/// every live path exposes the same top state. This preserves divergent lower
/// prefixes and avoids materializing each concrete stack separately.
fn try_advance_uniform_deterministic_frontier(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    const MAX_STEPS: usize = 16;

    let initial_state = closure.single_exclusive_top_value()?;
    if !matches!(table.action(initial_state, token), Some(Action::Reduce(..))) {
        return None;
    }

    let mut frontier = closure.clone();
    for _ in 0..MAX_STEPS {
        let state = frontier.single_exclusive_top_value()?;
        match table.action(state, token) {
            Some(Action::Reduce(nonterminal, len)) => {
                let base = frontier.popn(*len as isize);
                if base.is_empty() {
                    return Some(ParserGSS::empty());
                }
                let goto_from = base.single_exclusive_top_value()?;
                let Some((target, replace_top)) = table.goto_target(goto_from, *nonterminal)
                else {
                    return Some(ParserGSS::empty());
                };
                frontier = if replace_top {
                    base.popn(1).push(target)
                } else {
                    base.push(target)
                };
            }
            Some(Action::Shift(target, replace_top)) => {
                return Some(if *replace_top {
                    frontier.popn(1).push(*target)
                } else {
                    frontier.push(*target)
                });
            }
            Some(Action::ReplaceShifts(targets)) if targets.len() == 1 => {
                return Some(apply_replace_shifts(frontier, targets));
            }
            Some(Action::StackShifts(shifts)) if shifts.len() == 1 => {
                return Some(apply_stack_shifts(frontier, shifts));
            }
            None => return Some(ParserGSS::empty()),
            _ => return None,
        }
    }
    None
}

fn try_advance_bounded_concrete_paths(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    const MAX_PATHS: usize = 32;
    const MAX_DEPTH: usize = 64;
    const MAX_STEPS: usize = 16;

    let states = closure.peek_values();
    if states.is_empty() || states.len() > 4 || closure.max_depth() as usize > MAX_DEPTH {
        return None;
    }
    let mut saw_complex_action = false;
    for &state in &states {
        match table.action(state, token) {
            None | Some(Action::Shift(..)) => {}
            Some(Action::Reduce(..)) | Some(Action::GuardedStackShifts(..)) => {
                saw_complex_action = true;
            }
            Some(Action::ReplaceShifts(targets)) if targets.len() == 1 => {}
            Some(Action::StackShifts(shifts)) if shifts.len() == 1 => {}
            _ => return None,
        }
    }
    if !saw_complex_action {
        return None;
    }

    let mut outputs =
        SmallVec::<[(SmallVec<[u32; MAX_DEPTH]>, TerminalsDisallowed); MAX_PATHS]>::new();
    let mut unsupported = false;
    let complete = closure.for_each_stack_top_first_bounded(MAX_PATHS, |top_first, acc| {
        if unsupported || top_first.len() > MAX_DEPTH {
            unsupported = true;
            return;
        }

        let mut stack = SmallVec::<[u32; MAX_DEPTH]>::new();
        stack.extend(top_first.iter().rev().copied());
        let mut finished = false;
        let mut dead = false;

        for _ in 0..MAX_STEPS {
            let Some(&state) = stack.last() else {
                dead = true;
                finished = true;
                break;
            };
            match table.action(state, token) {
                Some(Action::Reduce(nt, len)) => {
                    let len = *len as usize;
                    if len >= stack.len() {
                        unsupported = true;
                        return;
                    }
                    stack.truncate(stack.len() - len);
                    let goto_from = *stack.last().unwrap();
                    let Some((target, replace_top)) = table.goto_target(goto_from, *nt) else {
                        dead = true;
                        finished = true;
                        break;
                    };
                    if replace_top {
                        *stack.last_mut().unwrap() = target;
                    } else {
                        if stack.len() == MAX_DEPTH {
                            unsupported = true;
                            return;
                        }
                        stack.push(target);
                    }
                }
                Some(Action::Shift(target, replace_top)) => {
                    if *replace_top {
                        *stack.last_mut().unwrap() = *target;
                    } else {
                        if stack.len() == MAX_DEPTH {
                            unsupported = true;
                            return;
                        }
                        stack.push(*target);
                    }
                    finished = true;
                    break;
                }
                Some(Action::StackShifts(shifts)) if shifts.len() == 1 => {
                    let shift = &shifts[0];
                    let pop = shift.pop as usize;
                    if pop > stack.len()
                        || stack.len() - pop + shift.pushes.len() > MAX_DEPTH
                    {
                        unsupported = true;
                        return;
                    }
                    stack.truncate(stack.len() - pop);
                    stack.extend(shift.pushes.iter().copied());
                    finished = true;
                    break;
                }
                Some(Action::GuardedStackShifts(shifts)) => {
                    let mut matched: Option<&GuardedStackShift> = None;
                    for shift in shifts {
                        let guards_match = shift.guards.iter().all(|guard| {
                            let pop = guard.pop as usize;
                            pop < stack.len()
                                && guard
                                    .states
                                    .binary_search(&stack[stack.len() - 1 - pop])
                                    .is_ok()
                        });
                        if !guards_match {
                            continue;
                        }
                        if matched.is_some() {
                            unsupported = true;
                            return;
                        }
                        matched = Some(shift);
                    }

                    let Some(shift) = matched else {
                        dead = true;
                        finished = true;
                        break;
                    };
                    let pop = shift.pop as usize;
                    if pop > stack.len()
                        || stack.len() - pop + shift.pushes.len() > MAX_DEPTH
                    {
                        unsupported = true;
                        return;
                    }
                    stack.truncate(stack.len() - pop);
                    stack.extend(shift.pushes.iter().copied());
                    finished = true;
                    break;
                }
                None | Some(Action::Accept) => {
                    dead = true;
                    finished = true;
                    break;
                }
                _ => {
                    unsupported = true;
                    return;
                }
            }
        }

        if !finished {
            unsupported = true;
            return;
        }
        if dead {
            return;
        }

        if let Some((_, existing_acc)) = outputs
            .iter_mut()
            .find(|(existing_stack, _)| existing_stack == &stack)
        {
            *existing_acc = existing_acc.merge(acc);
            return;
        }
        if outputs.len() == outputs.capacity() {
            unsupported = true;
            return;
        }
        outputs.push((stack, acc.clone()));
    });

    if !complete || unsupported {
        return None;
    }
    if outputs.is_empty() {
        return Some(ParserGSS::empty());
    }
    let mut groups = SmallVec::<
        [(TerminalsDisallowed, SmallVec<[&[u32]; MAX_PATHS]>); 4],
    >::new();
    for (stack, acc) in &outputs {
        if let Some((_, slices)) = groups
            .iter_mut()
            .find(|(existing_acc, _)| existing_acc == acc)
        {
            slices.push(stack.as_slice());
        } else {
            let mut slices = SmallVec::<[&[u32]; MAX_PATHS]>::new();
            slices.push(stack.as_slice());
            groups.push((acc.clone(), slices));
        }
    }

    let mut out = ParserGSS::empty();
    for (acc, slices) in groups {
        let group = ParserGSS::from_small_stack_slices_shared_prefix(&slices, acc)?;
        merge_into(&mut out, group);
    }
    Some(out)
}

/// Exact bounded admission for a small frontier where each current top is
/// assigned at most one terminal.  Returns `None` on unsupported branching so
/// callers can fall back to the general exact simulator.
pub fn stack_may_advance_disjoint_top_terminals_bounded(
    table: &GLRTable,
    closure: &ParserGSS,
    terminal_by_top: &[(u32, TerminalID)],
) -> Option<bool> {
    const MAX_PATHS: usize = 32;
    const MAX_DEPTH: usize = 64;
    const MAX_STEPS: usize = 16;
    if terminal_by_top.is_empty()
        || terminal_by_top.len() > 8
        || closure.max_depth() as usize > MAX_DEPTH
    {
        return None;
    }
    let terminal_for_top = |top: u32| {
        terminal_by_top
            .iter()
            .find(|(candidate, _)| *candidate == top)
            .map(|(_, terminal)| *terminal)
    };
    let mut unsupported = false;
    let mut admitted = false;
    let complete = closure.for_each_stack_top_first_bounded(MAX_PATHS, |top_first, _acc| {
        if admitted || unsupported || top_first.len() > MAX_DEPTH {
            if top_first.len() > MAX_DEPTH {
                unsupported = true;
            }
            return;
        }
        let Some(&initial_top) = top_first.first() else {
            return;
        };
        let Some(token) = terminal_for_top(initial_top) else {
            return;
        };
        let mut stack = SmallVec::<[u32; MAX_DEPTH]>::new();
        stack.extend(top_first.iter().rev().copied());

        for _ in 0..MAX_STEPS {
            let Some(&state) = stack.last() else {
                return;
            };
            match table.action(state, token) {
                Some(Action::Reduce(nt, len)) => {
                    let len = *len as usize;
                    if len >= stack.len() {
                        unsupported = true;
                        return;
                    }
                    stack.truncate(stack.len() - len);
                    let goto_from = *stack.last().unwrap();
                    let Some((target, replace_top)) = table.goto_target(goto_from, *nt) else {
                        return;
                    };
                    if replace_top {
                        *stack.last_mut().unwrap() = target;
                    } else {
                        if stack.len() == MAX_DEPTH {
                            unsupported = true;
                            return;
                        }
                        stack.push(target);
                    }
                }
                Some(Action::Shift(..)) | Some(Action::ReplaceShifts(_)) | Some(Action::Skip) => {
                    admitted = true;
                    return;
                }
                Some(Action::StackShifts(shifts)) => {
                    admitted |= shifts.iter().any(|shift| {
                        let pop = shift.pop as usize;
                        pop <= stack.len() && stack.len() - pop + shift.pushes.len() > 0
                    });
                    return;
                }
                Some(Action::GuardedStackShifts(shifts)) => {
                    admitted |= shifts.iter().any(|shift| {
                        let guards_match = shift.guards.iter().all(|guard| {
                            let pop = guard.pop as usize;
                            pop < stack.len()
                                && guard
                                    .states
                                    .binary_search(&stack[stack.len() - 1 - pop])
                                    .is_ok()
                        });
                        let pop = shift.pop as usize;
                        guards_match
                            && pop <= stack.len()
                            && stack.len() - pop + shift.pushes.len() > 0
                    });
                    return;
                }
                Some(Action::Split {
                    shift,
                    reduces,
                    accept,
                }) => {
                    if shift.is_some() || (*accept && token == EOF) {
                        admitted = true;
                        return;
                    }
                    // Multiple reduction alternatives need the general exact
                    // closure to preserve branching.
                    if reduces.is_empty() {
                        return;
                    }
                    unsupported = true;
                    return;
                }
                Some(Action::Accept) => {
                    admitted |= token == EOF;
                    return;
                }
                None => return,
            }
        }
        unsupported = true;
    });
    if !complete || unsupported {
        None
    } else {
        Some(admitted)
    }
}

/// Advance a small GSS frontier where each live top state is assigned one
/// terminal.  This is the exact union of advancing each disjoint top partition
/// by its assigned terminal, but it traverses concrete paths once and rebuilds
/// one shared-prefix GSS at the end.
///
/// The helper is deliberately bounded and fail-closed.  It is intended for
/// lexically equivalent/duplicate labels produced by composition, where the
/// current parser top makes the applicable logical label unique.  Callers must
/// provide at most one terminal per top state.
pub fn advance_stacks_disjoint_top_terminals_bounded(
    table: &GLRTable,
    closure: &ParserGSS,
    terminal_by_top: &[(u32, TerminalID)],
) -> Option<ParserGSS> {
    const MAX_PATHS: usize = 32;
    const MAX_DEPTH: usize = 64;
    const MAX_STEPS: usize = 16;

    if terminal_by_top.len() < 2
        || terminal_by_top.len() > 8
        || terminal_by_top
            .iter()
            .enumerate()
            .any(|(index, (top, _))| {
                terminal_by_top[..index]
                    .iter()
                    .any(|(other_top, _)| other_top == top)
            })
        || closure.max_depth() as usize > MAX_DEPTH
    {
        return None;
    }

    let terminal_for_top = |top: u32| {
        terminal_by_top
            .iter()
            .find(|(candidate, _)| *candidate == top)
            .map(|(_, terminal)| *terminal)
    };

    let states = closure.peek_values();
    if states.is_empty() || states.len() > 8 {
        return None;
    }
    let mut saw_complex_action = false;
    for &state in &states {
        let Some(token) = terminal_for_top(state) else {
            continue;
        };
        match table.action(state, token) {
            None | Some(Action::Shift(..)) | Some(Action::Skip) => {}
            Some(Action::Reduce(..)) | Some(Action::GuardedStackShifts(..)) => {
                saw_complex_action = true;
            }
            Some(Action::ReplaceShifts(targets)) if targets.len() == 1 => {}
            Some(Action::StackShifts(shifts)) if shifts.len() == 1 => {}
            _ => return None,
        }
    }
    if !saw_complex_action {
        return None;
    }

    let mut outputs =
        SmallVec::<[(SmallVec<[u32; MAX_DEPTH]>, TerminalsDisallowed); MAX_PATHS]>::new();
    let mut unsupported = false;
    let complete = closure.for_each_stack_top_first_bounded(MAX_PATHS, |top_first, acc| {
        if unsupported || top_first.len() > MAX_DEPTH {
            unsupported = true;
            return;
        }
        let Some(&initial_top) = top_first.first() else {
            return;
        };
        let Some(token) = terminal_for_top(initial_top) else {
            // This parser path has no matched logical terminal in the alias
            // group, so it contributes no successor to the union.
            return;
        };

        let mut stack = SmallVec::<[u32; MAX_DEPTH]>::new();
        stack.extend(top_first.iter().rev().copied());
        let mut finished = false;
        let mut dead = false;

        for _ in 0..MAX_STEPS {
            let Some(&state) = stack.last() else {
                dead = true;
                finished = true;
                break;
            };
            match table.action(state, token) {
                Some(Action::Reduce(nt, len)) => {
                    let len = *len as usize;
                    if len >= stack.len() {
                        unsupported = true;
                        return;
                    }
                    stack.truncate(stack.len() - len);
                    let goto_from = *stack.last().unwrap();
                    let Some((target, replace_top)) = table.goto_target(goto_from, *nt) else {
                        dead = true;
                        finished = true;
                        break;
                    };
                    if replace_top {
                        *stack.last_mut().unwrap() = target;
                    } else {
                        if stack.len() == MAX_DEPTH {
                            unsupported = true;
                            return;
                        }
                        stack.push(target);
                    }
                }
                Some(Action::Shift(target, replace_top)) => {
                    if *replace_top {
                        *stack.last_mut().unwrap() = *target;
                    } else {
                        if stack.len() == MAX_DEPTH {
                            unsupported = true;
                            return;
                        }
                        stack.push(*target);
                    }
                    finished = true;
                    break;
                }
                Some(Action::ReplaceShifts(targets)) if targets.len() == 1 => {
                    *stack.last_mut().unwrap() = targets[0];
                    finished = true;
                    break;
                }
                Some(Action::StackShifts(shifts)) if shifts.len() == 1 => {
                    let shift = &shifts[0];
                    let pop = shift.pop as usize;
                    if pop > stack.len()
                        || stack.len() - pop + shift.pushes.len() > MAX_DEPTH
                    {
                        unsupported = true;
                        return;
                    }
                    stack.truncate(stack.len() - pop);
                    stack.extend(shift.pushes.iter().copied());
                    finished = true;
                    break;
                }
                Some(Action::GuardedStackShifts(shifts)) => {
                    let mut matched: Option<&GuardedStackShift> = None;
                    for shift in shifts {
                        let guards_match = shift.guards.iter().all(|guard| {
                            let pop = guard.pop as usize;
                            pop < stack.len()
                                && guard
                                    .states
                                    .binary_search(&stack[stack.len() - 1 - pop])
                                    .is_ok()
                        });
                        if !guards_match {
                            continue;
                        }
                        if matched.is_some() {
                            unsupported = true;
                            return;
                        }
                        matched = Some(shift);
                    }
                    let Some(shift) = matched else {
                        dead = true;
                        finished = true;
                        break;
                    };
                    let pop = shift.pop as usize;
                    if pop > stack.len()
                        || stack.len() - pop + shift.pushes.len() > MAX_DEPTH
                    {
                        unsupported = true;
                        return;
                    }
                    stack.truncate(stack.len() - pop);
                    stack.extend(shift.pushes.iter().copied());
                    finished = true;
                    break;
                }
                Some(Action::Skip) => {
                    finished = true;
                    break;
                }
                None | Some(Action::Accept) => {
                    dead = true;
                    finished = true;
                    break;
                }
                _ => {
                    unsupported = true;
                    return;
                }
            }
        }

        if !finished {
            unsupported = true;
            return;
        }
        if dead {
            return;
        }
        if let Some((_, existing_acc)) = outputs
            .iter_mut()
            .find(|(existing_stack, _)| existing_stack == &stack)
        {
            *existing_acc = existing_acc.merge(acc);
            return;
        }
        if outputs.len() == outputs.capacity() {
            unsupported = true;
            return;
        }
        outputs.push((stack, acc.clone()));
    });

    if !complete || unsupported {
        return None;
    }
    if outputs.is_empty() {
        return Some(ParserGSS::empty());
    }
    let mut groups = SmallVec::<
        [(TerminalsDisallowed, SmallVec<[&[u32]; MAX_PATHS]>); 4],
    >::new();
    for (stack, acc) in &outputs {
        if let Some((_, slices)) = groups
            .iter_mut()
            .find(|(existing_acc, _)| existing_acc == acc)
        {
            slices.push(stack.as_slice());
        } else {
            let mut slices = SmallVec::<[&[u32]; MAX_PATHS]>::new();
            slices.push(stack.as_slice());
            groups.push((acc.clone(), slices));
        }
    }
    let mut out = ParserGSS::empty();
    for (acc, slices) in groups {
        let group = ParserGSS::from_small_stack_slices_shared_prefix(&slices, acc)?;
        merge_into(&mut out, group);
    }
    Some(out)
}

/// Apply one pop-one reduction across all of its live predecessor states, then
/// finish the same terminal with a pure shift frontier. This preserves the
/// shared lower GSS instead of isolating one predecessor branch at a time.
fn advance_pop1_reduce_predecessor_frontier(
    table: &GLRTable,
    closure: &ParserGSS,
    state: u32,
    nt: u32,
    token: TerminalID,
) -> Option<ParserGSS> {
    if let Ok(remapped) = closure.try_replace_predecessors_below_top(&state, |predecessor| {
        let Some((goto_target, goto_replace)) = table.goto_target(*predecessor, nt) else {
            return Ok(None);
        };
        if !goto_replace {
            return Err(());
        }
        let Some(action) = table.action(goto_target, token) else {
            return Ok(None);
        };
        let Some((target, replace_top)) =
            effective_pure_frontier_shift(table, goto_target, token, action)
        else {
            return Err(());
        };
        if !replace_top {
            return Err(());
        }
        Ok(Some(target))
    }) {
        return Some(remapped);
    }

    let popped = closure.pop_top_value(&state);
    if popped.is_empty() {
        return Some(ParserGSS::empty());
    }

    let mut gotos = SmallVec::<[(u32, u32, bool); 16]>::new();
    for goto_from in popped.peek_values() {
        let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
            continue;
        };
        if gotos.len() == gotos.capacity() {
            return None;
        }
        gotos.push((goto_from, target, is_replace));
    }
    if gotos.is_empty() {
        return Some(ParserGSS::empty());
    }

    let reduced = popped
        .try_apply_selective_top_pure_shifts(gotos.iter().copied())
        .unwrap_or_else(|| popped.apply_top_pure_shifts(gotos));
    if reduced.is_empty() {
        return Some(ParserGSS::empty());
    }

    let mut shifts = SmallVec::<[(u32, u32, bool); 16]>::new();
    for next_state in reduced.peek_values() {
        let Some(action) = table.action(next_state, token) else {
            continue;
        };
        let (target, replace_top) =
            effective_pure_frontier_shift(table, next_state, token, action)?;
        if shifts.len() == shifts.capacity() {
            return None;
        }
        shifts.push((next_state, target, replace_top));
    }
    if shifts.is_empty() {
        return Some(ParserGSS::empty());
    }

    Some(
        reduced
            .try_apply_selective_top_pure_shifts(shifts.iter().copied())
            .unwrap_or_else(|| reduced.apply_top_pure_shifts(shifts)),
    )
}

fn try_advance_single_top_pop1_reduce_frontier(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let state = closure.single_exclusive_top_value()?;
    let nt = match table.action(state, token)? {
        Action::Reduce(nt, 1) => *nt,
        Action::Split {
            shift: None,
            reduces,
            accept: false,
        } if reduces.len() == 1 && reduces[0].1 == 1 => reduces[0].0,
        _ => return None,
    };
    advance_pop1_reduce_predecessor_frontier(table, closure, state, nt, token)
}

/// Handle a one-wave frontier containing pop-one reductions and guarded stack
/// shifts. Each active top is transformed structurally from its popped lower
/// interface; unsupported action shapes decline to the authoritative GLR path.
fn try_advance_pop1_reduce_guarded_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut combined_saw_reduce = false;
    let mut combined_saw_guarded = false;
    let mut max_guard_pop = 0usize;
    let mut combined_shape_supported = true;
    for state in &states {
        match table.action(*state, token) {
            None => {}
            Some(Action::Reduce(_, 1)) => combined_saw_reduce = true,
            Some(Action::Split {
                shift: None,
                reduces,
                accept: false,
            }) if reduces.len() == 1 && reduces[0].1 == 1 => {
                combined_saw_reduce = true;
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                combined_saw_guarded = true;
                for shift in shifts {
                    for guard in &shift.guards {
                        max_guard_pop = max_guard_pop.max(guard.pop as usize);
                    }
                }
            }
            _ => combined_shape_supported = false,
        }
    }

    if combined_shape_supported
        && combined_saw_reduce
        && combined_saw_guarded
        && max_guard_pop <= 9
    {
        let lower_prefix_len = max_guard_pop.saturating_sub(1);
        if let Ok(combined) = closure.try_replace_top_predecessors_with_sequences(
            lower_prefix_len,
            |top, predecessor, lower_prefix| {
                let action = table.action(*top, token).ok_or(())?;
                let reduction_nt = match action {
                    Action::Reduce(nt, 1) => Some(*nt),
                    Action::Split {
                        shift: None,
                        reduces,
                        accept: false,
                    } if reduces.len() == 1 && reduces[0].1 == 1 => Some(reduces[0].0),
                    _ => None,
                };
                if let Some(nt) = reduction_nt {
                    let Some((goto_target, goto_replace)) = table.goto_target(*predecessor, nt) else {
                        return Ok(None);
                    };
                    // This structural remap replaces both the old top and its
                    // predecessor. A non-replacing goto needs to keep that
                    // predecessor, so decline the optimization rather than
                    // incorrectly treating the live reduction branch as dead.
                    if !goto_replace {
                        return Err(());
                    }
                    let Some(next_action) = table.action(goto_target, token) else {
                        return Ok(None);
                    };
                    let Some((target, true)) =
                        effective_pure_frontier_shift(table, goto_target, token, next_action)
                    else {
                        return Err(());
                    };
                    let mut pushes = SmallVec::<[u32; 4]>::new();
                    pushes.push(target);
                    return Ok(Some(pushes));
                }

                let Action::GuardedStackShifts(shifts) = action else {
                    return Ok(None);
                };
                let mut selected = SmallVec::<[u32; 4]>::new();
                let mut matched = false;
                for shift in shifts {
                    let guards_match = shift.guards.iter().try_fold(true, |matches, guard| {
                        if !matches {
                            return Ok(false);
                        }
                        let value = match guard.pop {
                            0 => top,
                            1 => predecessor,
                            pop => lower_prefix.get(pop as usize - 2).ok_or(())?,
                        };
                        Ok::<bool, ()>(guard.states.binary_search(value).is_ok())
                    })?;
                    if !guards_match {
                        continue;
                    }
                    if shift.pop != 2 || shift.pushes.is_empty() {
                        return Err(());
                    }
                    if matched && selected.as_slice() != shift.pushes.as_slice() {
                        return Err(());
                    }
                    if !matched {
                        selected.extend(shift.pushes.iter().copied());
                        matched = true;
                    }
                }
                Ok(matched.then_some(selected))
            },
        ) {
            return Some(combined);
        }
    }

    let mut out = ParserGSS::empty();
    let mut saw_reduce = false;
    let mut saw_guarded_shift = false;
    for state in states {
        match table.action(state, token) {
            None => {}
            Some(Action::Reduce(nt, 1)) => {
                saw_reduce = true;
                merge_into(
                    &mut out,
                    advance_pop1_reduce_predecessor_frontier(
                        table, closure, state, *nt, token,
                    )?,
                );
            }
            Some(Action::Split {
                shift: None,
                reduces,
                accept: false,
            }) if reduces.len() == 1 && reduces[0].1 == 1 => {
                saw_reduce = true;
                merge_into(
                    &mut out,
                    advance_pop1_reduce_predecessor_frontier(
                        table,
                        closure,
                        state,
                        reduces[0].0,
                        token,
                    )?,
                );
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                saw_guarded_shift = true;
                let popped = closure.pop_top_value(&state);
                if popped.is_empty() {
                    continue;
                }
                let branch = popped.push(state);
                merge_into(
                    &mut out,
                    apply_guarded_stack_shifts_fast(
                        &branch,
                        shifts,
                        table.guarded_shift_index(state, token),
                    )?,
                );
            }
            _ => return None,
        }
    }

    (saw_reduce && saw_guarded_shift && !out.is_empty()).then_some(out)
}

/// Advance a frontier where exactly one top state is live and its action is
/// a set of pop-one stack shifts. Dead top alternatives are discarded by
/// taking the active state's already-popped lower frontier directly, avoiding
/// isolate/pop/merge reconstruction of the complete GSS.
fn try_advance_single_active_pop1_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut active: Option<(u32, &[StackShift])> = None;
    for state in states {
        match table.action(state, token) {
            None => {}
            Some(Action::StackShifts(shifts))
                if shifts.iter().all(|shift| shift.pop == 1) =>
            {
                if active.is_some() {
                    return None;
                }
                active = Some((state, shifts));
            }
            _ => return None,
        }
    }

    let (state, shifts) = active?;
    let base = closure.pop_top_value(&state);
    if base.is_empty() {
        return Some(ParserGSS::empty());
    }
    let pushes = shifts
        .iter()
        .map(|shift| shift.pushes.as_slice())
        .collect::<SmallVec<[&[u32]; 4]>>();
    Some(apply_push_sequences(base, &pushes))
}

fn try_advance_single_alt_pop1_common_suffix_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut grouped_prefixes: SmallVec<[(u32, SmallVec<[(u32, u32, bool); 8]>); 4]> = SmallVec::new();
    let mut pure_shifts: SmallVec<[(u32, u32, bool); 4]> = SmallVec::new();
    let mut saw_stack_shift = false;

    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        match action {
            Action::StackShifts(shifts) => {
                let [shift] = shifts.as_slice() else {
                    return None;
                };
                if shift.pop != 1 || shift.pushes.len() != 2 {
                    return None;
                }
                saw_stack_shift = true;
                let prefix = shift.pushes[0];
                let suffix = shift.pushes[1];
                if let Some((_, prefixes)) = grouped_prefixes
                    .iter_mut()
                    .find(|(existing_suffix, _)| *existing_suffix == suffix)
                {
                    prefixes.push((state, prefix, true));
                } else {
                    let mut prefixes = SmallVec::new();
                    prefixes.push((state, prefix, true));
                    grouped_prefixes.push((suffix, prefixes));
                }
            }
            Action::Shift(target, is_replace) => {
                pure_shifts.push((
                    state,
                    *target,
                    *is_replace && !table.forwarded_shifts.contains(&(state, token)),
                ));
            }
            _ => return None,
        }
    }

    if !saw_stack_shift {
        return None;
    }

    let mut out = ParserGSS::empty();
    if !pure_shifts.is_empty() {
        merge_into(&mut out, closure.apply_top_pure_shifts(pure_shifts));
    }
    for (suffix, prefixes) in grouped_prefixes {
        merge_into(&mut out, closure.apply_top_pure_shifts(prefixes).push(suffix));
    }

    (!out.is_empty()).then_some(out)
}

fn try_advance_single_active_pop1_reduce_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut reduction: Option<(u32, u32)> = None;
    for state in states {
        match table.action(state, token) {
            None => {}
            Some(Action::Reduce(nt, 1)) => {
                if reduction.replace((state, *nt)).is_some() {
                    return None;
                }
            }
            Some(Action::Split { shift: None, reduces, accept: false })
                if reduces.len() == 1 && reduces[0].1 == 1 =>
            {
                if reduction.replace((state, reduces[0].0)).is_some() {
                    return None;
                }
            }
            _ => return None,
        }
    }

    let (state, nt) = reduction?;
    let popped = closure.pop_top_value(&state);
    if popped.is_empty() {
        return None;
    }

    if let Some(mut stack) = popped.try_virtual_stack() {
        let Some(&goto_from) = stack.top() else {
            return None;
        };
        let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
            return None;
        };
        if is_replace {
            stack.replace_top(target);
        } else {
            stack.push(target);
        }
        let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, stack, token);
        return det_ok.then(|| branch.into_gss());
    }

    let goto_sources = popped.peek_values();
    let [goto_from] = goto_sources.as_slice() else {
        return None;
    };
    let (target, is_replace) = table.goto_target(*goto_from, nt)?;
    let (branch, det_ok) = advance_reduce_branch(table, popped, target, is_replace, token);
    det_ok.then(|| branch.into_gss())
}

fn try_advance_pop1_common_base_reduce_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let base = closure.pop1_common_interface_base()?;
    let base_stack = base.try_virtual_stack()?;
    let Some(&goto_from) = base_stack.top() else {
        return None;
    };

    let mut targets = Vec::with_capacity(states.len());
    let mut fallback = ParserGSS::empty();
    let mut saw_reduce = false;

    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        let nt = match action {
            Action::Reduce(nt, 1) => *nt,
            Action::Split {
                shift: None,
                reduces,
                accept: false,
            } if reduces.len() == 1 && reduces[0].1 == 1 => reduces[0].0,
            _ => return None,
        };
        saw_reduce = true;

        let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
            continue;
        };

        if !is_replace
            && let Some(next_action) = table.action(target, token)
            && let Some((next_target, true)) =
                effective_pure_frontier_shift(table, target, token, next_action)
        {
            targets.push(next_target);
            continue;
        }

        let mut stack = base_stack.clone();
        if is_replace {
            if !stack.replace_top(target) {
                return None;
            }
        } else {
            stack.push(target);
        }

        let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, stack, token);
        if !det_ok {
            return None;
        }
        match branch {
            AdvancedBranch::Stack(stack) => {
                if let Some(top) = stack.single_top_extension_of(&base_stack) {
                    targets.push(top);
                } else {
                    merge_into(&mut fallback, stack.into_gss());
                }
            }
            AdvancedBranch::Gss(branch) => merge_into(&mut fallback, branch),
        }
    }

    if !saw_reduce {
        return None;
    }

    let mut out = ParserGSS::empty();
    if !targets.is_empty() {
        if let Some(grouped) = base_stack
            .clone()
            .into_gss_after_popping_and_pushing_single_branches(0, targets.iter())
        {
            merge_into(&mut out, grouped);
        }
    }
    merge_into(&mut out, fallback);
    (!out.is_empty()).then_some(out)
}

fn try_advance_mixed_top_pop1_shift_reduce_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let mut closure = closure.clone();
    let mut shifted = ParserGSS::empty();

    let mut seen_frontiers: FxHashSet<Vec<u32>> = FxHashSet::default();
    loop {
        let states = closure.peek_values();
        if states.len() < 2 {
            return None;
        }
        let mut frontier_signature = states.to_vec();
        frontier_signature.sort_unstable();
        if !seen_frontiers.insert(frontier_signature) {
            return None;
        }

        let mut pure_shifts: SmallVec<[(u32, u32, bool); 16]> = SmallVec::new();
        let mut stack_shifts: SmallVec<[(u32, &[StackShift]); 16]> = SmallVec::new();
        let mut reductions: SmallVec<[(u32, u32); 16]> = SmallVec::new();

        for state in states {
            let Some(action) = table.action(state, token) else {
                continue;
            };
            match action {
                Action::Shift(target, is_replace) => {
                    pure_shifts.push((state, *target, *is_replace));
                }
                Action::StackShifts(shifts) => {
                    if let Some((target, is_replace)) = pure_frontier_shift(action) {
                        pure_shifts.push((state, target, is_replace));
                    } else {
                        stack_shifts.push((state, shifts));
                    }
                }
                Action::Reduce(nt, 1) => {
                    reductions.push((state, *nt));
                }
                Action::Split { shift, reduces, accept: false }
                    if reduces.iter().all(|(_, len)| *len == 1) =>
                {
                    if let Some((target, is_replace)) = shift {
                        pure_shifts.push((
                            state,
                            *target,
                            *is_replace && !table.forwarded_shifts.contains(&(state, token)),
                        ));
                    }
                    for &(nt, _) in reduces {
                        reductions.push((state, nt));
                    }
                }
                _ => return None,
            }
        }

        if pure_shifts.is_empty() && stack_shifts.is_empty() && reductions.is_empty() {
            return None;
        }
        if reductions.is_empty() && !stack_shifts.is_empty() {
            return None;
        }

        if !pure_shifts.is_empty() {
            let shifted_wave = match pure_shifts.as_slice() {
                [shift] => closure
                    .try_apply_selective_top_pure_shifts([*shift])
                    .unwrap_or_else(|| closure.apply_top_pure_shifts([*shift])),
                _ => closure.apply_top_pure_shifts(pure_shifts),
            };
            merge_into(&mut shifted, shifted_wave);
        }

        let common_pop1_base = closure.pop1_common_interface_base();

        for (state, shifts) in stack_shifts {
            let branch_base = common_pop1_base
                .as_ref()
                .map(|base| base.clone().push(state))
                .unwrap_or_else(|| closure.isolate(Some(state)));
            merge_into(&mut shifted, apply_stack_shifts(branch_base, shifts));
        }

        if reductions.is_empty() {
            return (!shifted.is_empty()).then_some(shifted);
        }

        let common_reduce_source = common_pop1_base.as_ref().and_then(|base| {
            let values = base.peek_values();
            let [source] = values.as_slice() else {
                return None;
            };
            Some((base.clone(), *source))
        });

        let mut next = ParserGSS::empty();
        for (state, nt) in reductions {
            if let Some((base, goto_from)) = common_reduce_source.as_ref() {
                let Some((target, is_replace)) = table.goto_target(*goto_from, nt) else { continue; };
                let (branch, det_ok) = advance_reduce_branch(table, base.clone(), target, is_replace, token);
                if det_ok {
                    match branch {
                        AdvancedBranch::Stack(stack) => {
                            shifted = shifted.absorb_vstack_same_acc_owned(stack);
                        }
                        AdvancedBranch::Gss(branch) => {
                            merge_into(&mut shifted, branch);
                        }
                    }
                } else {
                    merge_into(&mut next, branch.into_gss());
                }
                continue;
            }

            let popped = closure.pop_top_value(&state);
            if popped.is_empty() {
                continue;
            }

            if let Some(mut stack) = popped.try_virtual_stack() {
                let Some(&goto_from) = stack.top() else { continue; };
                let Some((target, is_replace)) = table.goto_target(goto_from, nt) else { continue; };
                if is_replace {
                    stack.replace_top(target);
                } else {
                    stack.push(target);
                }
                let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, stack, token);
                if det_ok {
                    match branch {
                        AdvancedBranch::Stack(stack) => {
                            shifted = shifted.absorb_vstack_same_acc_owned(stack);
                        }
                        AdvancedBranch::Gss(branch) => {
                            merge_into(&mut shifted, branch);
                        }
                    }
                } else {
                    merge_into(&mut next, branch.into_gss());
                }
                continue;
            }

            let goto_sources = popped.peek_values();
            for goto_from in goto_sources.iter().copied() {
                let Some((target, is_replace)) = table.goto_target(goto_from, nt) else { continue; };
                let base = if goto_sources.len() == 1 {
                    popped.clone()
                } else {
                    popped.isolate(Some(goto_from))
                };
                let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
                if det_ok {
                    match branch {
                        AdvancedBranch::Stack(stack) => {
                            shifted = shifted.absorb_vstack_same_acc_owned(stack);
                        }
                        AdvancedBranch::Gss(branch) => {
                            merge_into(&mut shifted, branch);
                        }
                    }
                } else {
                    merge_into(&mut next, branch.into_gss());
                }
            }
        }

        if next.is_empty() {
            return (!shifted.is_empty()).then_some(shifted);
        }
        closure = next;
    }
}

fn try_advance_pop1_stackshift_shift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut effects = SmallVec::<[(u32, usize, &[u32]); 16]>::new();
    let mut saw_stack_shift = false;
    let mut saw_action = false;
    let mut top_local = true;
    for &state in &states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        saw_action = true;
        match action {
            Action::StackShifts(shifts) => {
                saw_stack_shift = true;
                for shift in shifts {
                    if shift.pop > 1
                        || shift.pushes.is_empty()
                        || effects.len() == effects.capacity()
                    {
                        top_local = false;
                        break;
                    }
                    effects.push((state, shift.pop as usize, shift.pushes.as_slice()));
                }
            }
            Action::Shift(target, is_replace) => {
                let pop = usize::from(
                    *is_replace && !table.forwarded_shifts.contains(&(state, token)),
                );
                if effects.len() == effects.capacity() {
                    top_local = false;
                    break;
                }
                effects.push((state, pop, std::slice::from_ref(target)));
            }
            _ => {
                top_local = false;
                break;
            }
        }
        if !top_local {
            break;
        }
    }
    if top_local && saw_action && saw_stack_shift
        && let Some(shifted) = closure.try_apply_top_stack_effects(effects)
    {
        return Some(shifted);
    }

    if let Some(shifted) = try_advance_single_alt_pop1_common_suffix_stackshift_wave(table, closure, token) {
        return Some(shifted);
    }

    let base = closure.pop1_common_interface_base()?;

    let mut out = ParserGSS::empty();
    let mut saw_stack_shift = false;
    let mut saw_action = false;
    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        saw_action = true;
        match action {
            Action::StackShifts(shifts) => {
                if let Some((target, is_replace)) = pure_frontier_shift(action) {
                    let branch = if is_replace {
                        base.clone().push(target)
                    } else {
                        base.clone().push(state).push(target)
                    };
                    merge_into(&mut out, branch);
                } else {
                    saw_stack_shift = true;
                    merge_into(&mut out, apply_stack_shifts(base.clone().push(state), shifts));
                }
            }
            Action::Shift(target, is_replace) => {
                let is_replace = *is_replace && !table.forwarded_shifts.contains(&(state, token));
                let branch = if is_replace {
                    base.clone().push(*target)
                } else {
                    base.clone().push(state).push(*target)
                };
                merge_into(&mut out, branch);
            }
            _ => return None,
        }
    }

    (saw_action && saw_stack_shift && !out.is_empty()).then_some(out)
}

fn try_advance_pop1_reduce_plus_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let base = if let Some(base) = closure.pop1_common_interface_base() {
        base
    } else {
        let base = closure.popn(1);
        let mut reconstructed = ParserGSS::empty();
        for &state in &states {
            merge_into(&mut reconstructed, base.clone().push(state));
        }
        if &reconstructed != closure {
            return None;
        }
        base
    };
    let base_values = base.peek_values();
    let [base_top] = base_values.as_slice() else {
        return None;
    };
    let base_top = *base_top;

    let mut reduce_nt: Option<u32> = None;
    let mut shifted = ParserGSS::empty();
    for state in states {
        let action = table.action(state, token)?;
        match action {
            Action::Reduce(nt, 1) => {
                if reduce_nt.replace(*nt).is_some() {
                    return None;
                }
            }
            Action::StackShifts(shifts) => {
                merge_into(&mut shifted, apply_stack_shifts(base.push(state), shifts));
            }
            Action::Shift(target, is_replace) => {
                let branch = base.push(state);
                let is_replace = *is_replace && !table.forwarded_shifts.contains(&(state, token));
                if is_replace {
                    merge_into(&mut shifted, branch.popn(1).push(*target));
                } else {
                    merge_into(&mut shifted, branch.push(*target));
                }
            }
            _ => return None,
        }
    }

    let reduce_nt = reduce_nt?;
    if shifted.is_empty() {
        return None;
    }

    let (target, is_replace) = table.goto_target(base_top, reduce_nt)?;
    let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
    if !det_ok {
        return None;
    }
    merge_into(&mut shifted, branch.into_gss());
    Some(shifted)
}

fn rebuild_floor_cross_from_shifts(
    popped: ParserGSS,
    shifts: SmallVec<[FloorCrossShift; 8]>,
) -> ParserGSS {
    if shifts.iter().any(|(_, _, is_replace)| *is_replace) {
        popped.apply_top_pure_shifts(shifts)
    } else {
        popped.remap_top_values_owned(
            shifts
                .into_iter()
                .map(|(goto_from, target, _)| (goto_from, target)),
        )
    }
}

fn push_states(mut gss: ParserGSS, states: &[u32]) -> ParserGSS {
    for &state in states {
        gss = gss.push(state);
    }
    gss
}

fn common_stack_shift_suffix_len(pushes: &[&[u32]]) -> usize {
    let Some(first) = pushes.first() else {
        return 0;
    };
    let mut suffix_len = 0;
    'suffix: while suffix_len < first.len() {
        let state = first[first.len() - 1 - suffix_len];
        for pushes in &pushes[1..] {
            if suffix_len >= pushes.len() || pushes[pushes.len() - 1 - suffix_len] != state {
                break 'suffix;
            }
        }
        suffix_len += 1;
    }
    suffix_len
}

fn apply_push_sequences(base: ParserGSS, pushes: &[&[u32]]) -> ParserGSS {
    match pushes {
        [] => ParserGSS::empty(),
        [pushes] => push_states(base, pushes),
        _ => {
            let common_suffix_len = common_stack_shift_suffix_len(pushes);
            if common_suffix_len > 0 {
                let mut prefixes = ParserGSS::empty();
                for pushes in pushes {
                    let prefix_len = pushes.len() - common_suffix_len;
                    merge_into(&mut prefixes, push_states(base.clone(), &pushes[..prefix_len]));
                }
                let suffix = &pushes[0][pushes[0].len() - common_suffix_len..];
                push_states(prefixes, suffix)
            } else {
                let mut out = ParserGSS::empty();
                for pushes in pushes {
                    merge_into(&mut out, push_states(base.clone(), pushes));
                }
                out
            }
        }
    }
}

fn apply_replace_shifts(gss: ParserGSS, targets: &[u32]) -> ParserGSS {
    if targets.is_empty() {
        return ParserGSS::empty();
    }
    if let Some(stack) = gss.try_virtual_stack() {
        let sorted_unique = targets.windows(2).all(|pair| pair[0] < pair[1]);
        let shifted = if sorted_unique {
            stack.into_gss_after_popping_and_pushing_unique_single_branches(1, targets.iter())
        } else {
            stack.into_gss_after_popping_and_pushing_single_branches(1, targets.iter())
        };
        if let Some(shifted) = shifted {
            return shifted;
        }
    }

    let base = gss.popn(1);
    let mut out = ParserGSS::empty();
    for &target in targets {
        merge_into(&mut out, base.clone().push(target));
    }
    out
}

fn apply_stack_shifts(gss: ParserGSS, shifts: &[StackShift]) -> ParserGSS {
    if let Some(stack) = gss.try_virtual_stack()
        && let Some(first) = shifts.first()
        && !first.pushes.is_empty()
        && shifts
            .iter()
            .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
    {
        if shifts.iter().all(|shift| shift.pushes.len() == 1) {
            let targets_are_sorted_unique = shifts
                .windows(2)
                .all(|pair| pair[0].pushes[0] < pair[1].pushes[0]);
            let shifted = if targets_are_sorted_unique {
                stack.clone().into_gss_after_popping_and_pushing_unique_single_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| &shift.pushes[0]),
                )
            } else {
                stack.clone().into_gss_after_popping_and_pushing_single_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| &shift.pushes[0]),
                )
            };
            if let Some(shifted) = shifted {
                return shifted;
            }
        }
        if let Some(shifted) = stack.into_gss_after_popping_and_pushing_branches(
            first.pop as usize,
            shifts.iter().map(|shift| shift.pushes.as_slice()),
        ) {
            return shifted;
        }
    }

    if let Some(shifted) = gss.apply_stack_effects_to_single_concrete_path(
        shifts
            .iter()
            .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
        if stack_effect_to_stacks_fallback_disabled() {
            0
        } else {
            SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH
        },
    ) {
        return shifted;
    }
    let mut out = ParserGSS::empty();
    let mut groups: SmallVec<[(u32, SmallVec<[&[u32]; 4]>); 4]> = SmallVec::new();

    for shift in shifts {
        if let Some((_, group_pushes)) = groups.iter_mut().find(|(pop, _)| *pop == shift.pop) {
            group_pushes.push(shift.pushes.as_slice());
        } else {
            let mut group_pushes = SmallVec::new();
            group_pushes.push(shift.pushes.as_slice());
            groups.push((shift.pop, group_pushes));
        }
    }

    for (pop, pushes) in groups {
        let base = gss.popn(pop as isize);
        merge_into(&mut out, apply_push_sequences(base, &pushes));
    }
    out
}

fn guarded_stack_shifts_are_decidable_from_vstack(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    shifts: &[GuardedStackShift],
) -> bool {
    if !stack.has_hidden_floor_values() {
        return true;
    }

    let visible_len = stack.len();
    for shift in shifts {
        let mut ruled_out = false;
        for guard in &shift.guards {
            let pop = guard.pop as usize;
            if pop >= visible_len {
                // This still-live shift needs a branch-specific value from the
                // hidden floor. The virtual prefix cannot decide the guard.
                return false;
            }
            let state = stack
                .top_after_popping(pop)
                .expect("guard depth lies within visible virtual-stack prefix");
            if guard.states.binary_search(state).is_err() {
                ruled_out = true;
                break;
            }
        }

        if !ruled_out && shift.pop as usize > visible_len {
            // The guards are satisfied by the visible prefix, but applying the
            // effect would pop through the branched floor.
            return false;
        }
    }

    true
}

fn apply_guarded_stack_shifts_from_vstack(
    stack: VirtualStack<u32, TerminalsDisallowed>,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> ParserGSS {
    if guarded_stack_shifts_are_decidable_from_vstack(&stack, shifts) {
        apply_guarded_stack_shifts_to_vstack(&stack, shifts, index)
    } else {
        apply_guarded_stack_shifts(stack.into_gss(), shifts, index)
    }
}

fn apply_guarded_stack_shifts_as_predecessor_remap(
    gss: &ParserGSS,
    shifts: &[GuardedStackShift],
) -> Option<ParserGSS> {
    let virtual_stack = gss.try_virtual_stack()?;
    if !virtual_stack.has_hidden_floor_values() {
        return None;
    }
    let predecessor_pop = u32::try_from(virtual_stack.len()).ok()?;
    if predecessor_pop == 0 {
        return None;
    }
    let predecessor_frontier = gss.popn(predecessor_pop as isize);
    if predecessor_frontier.is_empty() {
        return Some(ParserGSS::empty());
    }
    let predecessor_values = predecessor_frontier.peek_values();
    let mut remaps = SmallVec::<[(u32, u32, bool); 16]>::new();

    for shift in shifts {
        let predecessor_guards = shift
            .guards
            .iter()
            .filter(|guard| guard.pop == predecessor_pop)
            .collect::<SmallVec<[_; 2]>>();
        if predecessor_guards.is_empty() {
            // Without a guard at the first hidden floor we cannot prove this
            // shift irrelevant without branch-specific traversal.
            return None;
        }
        let matching_predecessors = predecessor_values
            .iter()
            .copied()
            .filter(|predecessor| {
                predecessor_guards
                    .iter()
                    .all(|guard| guard.states.binary_search(predecessor).is_ok())
            })
            .collect::<SmallVec<[u32; 8]>>();
        if matching_predecessors.is_empty() {
            // This shift is dead for the represented GSS, even if its stack
            // effect has a shape the structural remap does not support.
            continue;
        }

        if shift.pop != predecessor_pop + 1
            || shift.pushes.len() != 1
            || shift
                .guards
                .iter()
                .any(|guard| guard.pop != predecessor_pop && guard.pop != shift.pop)
        {
            return None;
        }
        let base_guards = shift
            .guards
            .iter()
            .filter(|guard| guard.pop == shift.pop)
            .collect::<SmallVec<[_; 2]>>();

        for predecessor in matching_predecessors {
            let below = predecessor_frontier.pop_top_value(&predecessor);
            if below.is_empty() {
                continue;
            }
            let below_values = below.peek_values();
            let matching = below_values
                .iter()
                .filter(|state| {
                    base_guards
                        .iter()
                        .all(|guard| guard.states.binary_search(state).is_ok())
                })
                .count();
            if matching == 0 {
                continue;
            }
            if matching != below_values.len() {
                // The predecessor branch contains both matching and
                // non-matching deeper contexts. A top-only remap would lose
                // that path correlation, so leave it to the general path.
                return None;
            }
            if remaps.len() == remaps.capacity() {
                return None;
            }
            remaps.push((predecessor, shift.pushes[0], true));
        }
    }

    if remaps.is_empty() {
        return Some(ParserGSS::empty());
    }
    Some(predecessor_frontier.apply_top_pure_shifts(remaps))
}

pub fn apply_guarded_stack_shifts_fast(
    gss: &ParserGSS,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> Option<ParserGSS> {
    if let Some(stack) = gss.try_virtual_stack()
        && guarded_stack_shifts_are_decidable_from_vstack(&stack, shifts)
    {
        return Some(apply_guarded_stack_shifts_to_vstack(&stack, shifts, index));
    }

    if let Some(shifted) = apply_guarded_stack_shifts_as_predecessor_remap(gss, shifts) {
        return Some(shifted);
    }

    if let Some(shifted) = gss.apply_guarded_stack_effects_to_bounded_paths(
        shifts.iter().map(|shift| {
            (
                shift
                    .guards
                    .iter()
                    .map(|guard| (guard.pop as usize, guard.states.as_slice())),
                shift.pop as usize,
                shift.pushes.as_slice(),
            )
        }),
        16,
        64,
    ) {
        return Some(shifted);
    }

    if !guarded_stack_to_stacks_fallback_disabled()
        && let Some(shifted) = gss.apply_guarded_stack_effects_to_single_concrete_path(
            shifts.iter().map(|shift| {
                (
                    shift
                        .guards
                        .iter()
                        .map(|guard| (guard.pop as usize, guard.states.as_slice())),
                    shift.pop as usize,
                    shift.pushes.as_slice(),
                )
            }),
            GUARDED_STACK_TO_STACKS_MAX_DEPTH,
        )
    {
        return Some(shifted);
    }

    None
}

fn apply_guarded_stack_shifts(
    gss: ParserGSS,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> ParserGSS {
    if let Some(shifted) = apply_guarded_stack_shifts_fast(&gss, shifts, index) {
        return shifted;
    }

    let mut out = ParserGSS::empty();

    for shift in shifts {
        debug_assert!(shift.guards.windows(2).all(|w| w[0].pop <= w[1].pop));
        debug_assert!(shift.guards.iter().all(|guard| guard.pop <= shift.pop));

        let mut base = gss.clone();
        let mut depth = 0u32;
        let mut dead = false;

        for guard in &shift.guards {
            if guard.pop < depth {
                dead = true;
                break;
            }

            base = base.popn((guard.pop - depth) as isize);
            if base.is_empty() {
                dead = true;
                break;
            }

            let mut filtered = ParserGSS::empty();
            for &state in &guard.states {
                merge_into(&mut filtered, base.isolate(Some(state)));
            }

            base = filtered;
            if base.is_empty() {
                dead = true;
                break;
            }

            depth = guard.pop;
        }

        if dead || shift.pop < depth {
            continue;
        }

        let branch = push_states(base.popn((shift.pop - depth) as isize), &shift.pushes);
        merge_into(&mut out, branch);
    }

    out
}

fn indexed_guarded_shift_candidates(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    index: &GuardedShiftCellIndex,
) -> SmallVec<[u32; 8]> {
    let mut counts: FxHashMap<u32, u16> = FxHashMap::default();

    for &pop in &index.guard_pops {
        let Some(state) = stack.top_after_popping(pop as usize).copied() else {
            continue;
        };
        if let Some(shift_indices) = index.by_guard_key.get(&(pop, state)) {
            for &shift_index in shift_indices.iter() {
                *counts.entry(shift_index).or_insert(0) += 1;
            }
        }
    }

    let mut candidates = SmallVec::<[u32; 8]>::new();
    for (shift_index, count) in counts {
        if index
            .guard_counts
            .get(shift_index as usize)
            .is_some_and(|required| *required == count)
        {
            candidates.push(shift_index);
        }
    }
    candidates.extend(index.unguarded_indices.iter().copied());
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn apply_guarded_stack_shifts_to_vstack(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> ParserGSS {
    let mut groups: SmallVec<[(u32, SmallVec<[&[u32]; 4]>); 4]> = SmallVec::new();
    let mut empty_pushes: SmallVec<[u32; 4]> = SmallVec::new();
    let stack_len = stack.len();
    let mut state_after_pop_cache: SmallVec<[(u32, Option<u32>); 8]> = SmallVec::new();

    #[inline]
    fn state_after_popping(
        stack: &VirtualStack<u32, TerminalsDisallowed>,
        cache: &mut SmallVec<[(u32, Option<u32>); 8]>,
        pop: u32,
    ) -> Option<u32> {
        if let Some((_, cached)) = cache.iter().find(|(cached_pop, _)| *cached_pop == pop) {
            return *cached;
        }
        let value = stack.top_after_popping(pop as usize).copied();
        cache.push((pop, value));
        value
    }

    fn consider_guarded_shift<'a>(
        stack: &VirtualStack<u32, TerminalsDisallowed>,
        stack_len: usize,
        state_after_pop_cache: &mut SmallVec<[(u32, Option<u32>); 8]>,
        groups: &mut SmallVec<[(u32, SmallVec<[&'a [u32]; 4]>); 4]>,
        empty_pushes: &mut SmallVec<[u32; 4]>,
        shift: &'a GuardedStackShift,
    ) {
        debug_assert!(shift.guards.windows(2).all(|w| w[0].pop <= w[1].pop));
        debug_assert!(shift.guards.iter().all(|guard| guard.pop <= shift.pop));

        let mut dead = false;
        for guard in &shift.guards {
            let Some(state) = state_after_popping(stack, state_after_pop_cache, guard.pop) else {
                dead = true;
                break;
            };
            if guard.states.binary_search(&state).is_err() {
                dead = true;
                break;
            }
        }

        if dead || shift.pop as usize > stack_len {
            return;
        }

        if shift.pushes.is_empty() {
            empty_pushes.push(shift.pop);
        } else if let Some((_, pushes)) = groups.iter_mut().find(|(pop, _)| *pop == shift.pop) {
            pushes.push(shift.pushes.as_slice());
        } else {
            let mut pushes = SmallVec::new();
            pushes.push(shift.pushes.as_slice());
            groups.push((shift.pop, pushes));
        }
    }

    if let Some(index) = index {
        for shift_index in indexed_guarded_shift_candidates(stack, index) {
            if let Some(shift) = shifts.get(shift_index as usize) {
                consider_guarded_shift(
                    stack,
                    stack_len,
                    &mut state_after_pop_cache,
                    &mut groups,
                    &mut empty_pushes,
                    shift,
                );
            }
        }
    } else {
        for shift in shifts {
            consider_guarded_shift(
                stack,
                stack_len,
                &mut state_after_pop_cache,
                &mut groups,
                &mut empty_pushes,
                shift,
            );
        }
    }

    let mut out = ParserGSS::empty();
    for (pop, pushes) in groups {
        if let Some(branch) =
            stack.clone().into_gss_after_popping_and_pushing_branches(pop as usize, pushes)
        {
            merge_into(&mut out, branch);
        }
    }
    for pop in empty_pushes {
        let mut branch = stack.clone();
        if branch.pop(pop as usize) == 0 {
            merge_into(&mut out, branch.into_gss());
        }
    }
    out
}

#[inline]
fn virtual_stack_satisfies_guards(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    guards: &[StackShiftGuard],
) -> bool {
    let mut cursor = stack.clone();
    let mut depth = 0u32;

    for guard in guards {
        if guard.pop < depth {
            return false;
        }

        if cursor.pop((guard.pop - depth) as usize) != 0 {
            return false;
        }

        let Some(&state) = cursor.top() else {
            return false;
        };
        if guard.states.binary_search(&state).is_err() {
            return false;
        }

        depth = guard.pop;
    }

    true
}

#[inline]
fn virtual_stack_may_apply_guarded_shift(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    shift: &GuardedStackShift,
) -> bool {
    if !virtual_stack_satisfies_guards(stack, &shift.guards) {
        return false;
    }

    let mut cursor = stack.clone();
    cursor.pop(shift.pop as usize) == 0
}

fn reduce_sources_from_isolated(gss: &ParserGSS, rhs_len: usize) -> ReduceSources {
    let popped = gss.popn(rhs_len as isize);
    if popped.is_empty() {
        return SmallVec::new();
    }
    if let Some(v) = popped.single_top_value() {
        let mut result = SmallVec::new();
        // A sole *visible* predecessor does not exclude an epsilon stack in
        // the same GSS. Goto(v, nt) applies only to paths actually ending in v;
        // retaining the epsilon alternative invents a path and can leak its
        // accumulator into a valid path. The ordinary sole-top/no-epsilon
        // case preserves its shared pointer through isolate.
        result.push((v, popped.isolate(Some(v))));
        return result;
    }
    let top_vals = popped.peek_values();
    let mut result = SmallVec::new();
    for v in top_vals {
        result.push((v, popped.isolate(Some(v))));
    }
    result
}

fn reduce_branches_from_isolated(
    table: &GLRTable,
    gss: &ParserGSS,
    nt: u32,
    rhs_len: usize,
) -> ReduceBranches {
    if let Some(mut stack) = gss.try_virtual_stack() {
        if stack.pop(rhs_len) == 0 {
            if let Some(&goto_from) = stack.top() {
                if let Some((target, is_replace)) = table.goto_target(goto_from, nt) {
                    let mut branches = SmallVec::new();
                    branches.push((stack.into_gss(), target, is_replace));
                    return branches;
                }
            }
        }
    }

    let mut branches = SmallVec::new();
    for (goto_from, base) in reduce_sources_from_isolated(gss, rhs_len) {
        if let Some((target, is_replace)) = table.goto_target(goto_from, nt) {
            branches.push((base, target, is_replace));
        }
    }
    branches
}

fn merge_into(dst: &mut ParserGSS, branch: ParserGSS) {
    if branch.is_empty() {
        return;
    }
    if dst.is_empty() {
        *dst = branch;
    } else {
        *dst = dst.merge(&branch);
    }
}

fn advance_deterministically_from_vstack_raw(
    table: &GLRTable,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
    token: TerminalID,
) -> (AdvancedBranch, bool) {
    loop {
        let Some(&state) = stack.top() else {
            break;
        };

        match table.action(state, token) {
            Some(Action::Skip) => {
                return (AdvancedBranch::Stack(stack), true);
            }
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    if rhs_len == 1 {
                        if let Some(goto_from) = stack.parent_of_top() {
                            match table.goto_target(goto_from, *nt) {
                                Some((target, false)) if stack.replace_top(target) => continue,
                                Some((target, true)) => {
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    return (AdvancedBranch::Gss(ParserGSS::empty()), false);
                                }
                            }
                        }
                    }

                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                              return (AdvancedBranch::Gss(ParserGSS::empty()), false);
                        }
                    }
                } else {
                    let popped = stack.into_gss_after_popping(rhs_len);
                    let mut shifts = SmallVec::<[FloorCrossShift; 8]>::new();
                    for goto_from in popped.peek_values() {
                        if let Some((target, is_replace)) = table.goto_target(goto_from, *nt) {
                            shifts.push((goto_from, target, is_replace));
                        }
                    }
                    let rebuilt = rebuild_floor_cross_from_shifts(popped, shifts);
                      let Some(next_stack) = rebuilt.try_virtual_stack() else {
                          return (AdvancedBranch::Gss(rebuilt), false);
                      };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                } else {
                    stack.push(*target);
                }
                return (AdvancedBranch::Stack(stack), true);
            }
            Some(Action::ReplaceShifts(targets)) => {
                return (AdvancedBranch::Gss(apply_replace_shifts(stack.into_gss(), targets)), true);
            }
            Some(Action::StackShifts(shifts)) => {
                return (AdvancedBranch::Gss(apply_stack_shifts(stack.into_gss(), shifts)), true);
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                return (
                    AdvancedBranch::Gss(apply_guarded_stack_shifts_from_vstack(
                        stack,
                        shifts,
                        table.guarded_shift_index(state, token),
                    )),
                    true,
                );
            }
            Some(Action::Split { .. }) | Some(Action::Accept) | None => break,
        }
    }

    (AdvancedBranch::Stack(stack), false)
}

fn advance_deterministically_from_vstack(
    table: &GLRTable,
    stack: VirtualStack<u32, TerminalsDisallowed>,
    token: TerminalID,
) -> (ParserGSS, bool) {
    let (advanced, ok) = advance_deterministically_from_vstack_raw(table, stack, token);
    (advanced.into_gss(), ok)
}

fn advance_reduce_branch(
    table: &GLRTable,
    base: ParserGSS,
    target: u32,
    is_replace: bool,
    token: TerminalID,
) -> (AdvancedBranch, bool) {
    if let Some(mut stack) = base.try_virtual_stack() {
        if is_replace {
            stack.replace_top(target);
        } else {
            stack.push(target);
        }
        advance_deterministically_from_vstack_raw(table, stack, token)
    } else {
        let mut branch = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        let det_ok = advance_deterministically(table, &mut branch, token);
        (AdvancedBranch::Gss(branch), det_ok)
    }
}

fn single_concrete_path_as_vstack(
    gss: &ParserGSS,
) -> Option<VirtualStack<u32, TerminalsDisallowed>> {
    let (stack, acc) = gss.try_single_stack_bounded(SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH)?;
    ParserGSS::from_single_stack(stack, acc).try_virtual_stack()
}

fn advance_split_from_vstack(
    table: &GLRTable,
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    token: TerminalID,
    shift: Option<(u32, bool)>,
    reduces: &[(u32, u32)],
) -> Option<ParserGSS> {
    let mut shifted = ParserGSS::empty();

    if let Some((target, is_replace)) = shift {
        let mut branch = stack.clone();
        if is_replace {
            branch.replace_top(target);
        } else {
            branch.push(target);
        }
        shifted = branch.into_gss();
    }

    for &(nt, len) in reduces {
        let mut branch = stack.clone();
        if branch.pop(len as usize) != 0 {
            return None;
        }
        let goto_from = *branch.top()?;
        let (target, is_replace) = table.goto_target(goto_from, nt)?;
        if is_replace {
            branch.replace_top(target);
        } else {
            branch.push(target);
        }

        let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, branch, token);
        if !det_ok {
            return None;
        }
        match branch {
            AdvancedBranch::Stack(stack) => {
                shifted = shifted.absorb_vstack_same_acc_owned(stack);
            }
            AdvancedBranch::Gss(gss) => {
                merge_into(&mut shifted, gss);
            }
        }
    }

    Some(shifted)
}

fn advance_deterministically_profiled(
    table: &GLRTable,
    gss: &mut ParserGSS,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> bool {
    let Some(mut stack) = gss
        .try_virtual_stack()
        .or_else(|| single_concrete_path_as_vstack(gss))
    else {
        profile.det_exit_reason = 6;
        return false;
    };

    profile.deterministic_entered = true;
    profile.vstack_len = stack.len() as u32;

    loop {
        let Some(&state) = stack.top() else {
            profile.det_exit_reason = 5;
            break;
        };

        profile.n_det_action_lookups += 1;
        let action = table.action(state, token);
        if let Some(trace) = profile.trace.as_mut() {
            let trace_gss = stack.clone().into_gss();
            trace.det_steps
                .push(trace_action_summary(table, state, &trace_gss, action));
        }
        match action {
            Some(Action::Skip) => {
                *gss = stack.into_gss();
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    profile.n_reduces_above_floor += 1;
                    if rhs_len == 1 {
                        if let Some(goto_from) = stack.parent_of_top() {
                            profile.n_det_goto_lookups += 1;
                            match table.goto_target(goto_from, *nt) {
                                Some((target, false)) if stack.replace_top(target) => continue,
                                Some((target, true)) => {
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    *gss = ParserGSS::empty();
                                    profile.det_exit_reason = 4;
                                    return false;
                                }
                            }
                        }
                    }

                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    profile.n_det_goto_lookups += 1;
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                            *gss = ParserGSS::empty();
                            profile.det_exit_reason = 4;
                            return false;
                        }
                    }
                } else {
                    profile.n_floor_crossings += 1;
                    profile.n_det_popn_ops += 1;
                    let floor_cross_start = std::time::Instant::now();
                    let popped = stack.into_gss_after_popping(rhs_len);
                    let mut shifts = SmallVec::<[FloorCrossShift; 8]>::new();
                    for goto_from in popped.peek_values() {
                        profile.n_det_goto_lookups += 1;
                        if let Some((target, is_replace)) = table.goto_target(goto_from, *nt) {
                            shifts.push((goto_from, target, is_replace));
                        }
                    }
                    let rebuilt = rebuild_floor_cross_from_shifts(popped, shifts);
                    profile.det_floor_cross_ns += floor_cross_start.elapsed().as_nanos() as u64;
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        *gss = rebuilt;
                        profile.det_exit_reason = 7;
                        return false;
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                } else {
                    stack.push(*target);
                }
                *gss = stack.into_gss();
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::ReplaceShifts(targets)) => {
                *gss = apply_replace_shifts(stack.into_gss(), targets);
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::StackShifts(shifts)) => {
                *gss = apply_stack_shifts(stack.into_gss(), shifts);
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                *gss = apply_guarded_stack_shifts_from_vstack(
                    stack,
                    shifts,
                    table.guarded_shift_index(state, token),
                );
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::Split { shift, reduces, accept: false }) => {
                if let Some(shifted) =
                    advance_split_from_vstack(table, &stack, token, *shift, reduces)
                {
                    *gss = shifted;
                    profile.det_exit_reason = 1;
                    return true;
                }
                profile.det_exit_reason = 2;
                profile.det_exit_state = state;
                break;
            }
            Some(Action::Split { .. }) => {
                profile.det_exit_reason = 2;
                profile.det_exit_state = state;
                break;
            }
            Some(Action::Accept) => {
                profile.det_exit_reason = 3;
                profile.det_exit_state = state;
                break;
            }
            None => {
                profile.det_exit_reason = 4;
                profile.det_exit_state = state;
                break;
            }
        }
    }

    *gss = stack.into_gss();
    false
}

fn advance_nondeterministically_profiled(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> ParserGSS {
    use std::time::Instant;

    let mut shifted = ParserGSS::empty();
    loop {
        profile.n_nondet_waves += 1;
        let frontier_states = closure.peek_values();
        if let Some(trace) = profile.trace.as_mut() {
            trace.nondet_waves.push(AdvanceTraceWave {
                wave_index: profile.n_nondet_waves,
                frontier_states: frontier_states.to_vec(),
                branches: Vec::new(),
            });
        }

        if let Some(shifted_wave) =
            try_advance_single_active_pop1_stackshift_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_single_active_pop1_reduce_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += 1;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_pop1_common_base_reduce_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += frontier_states.len() as u32;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_mixed_top_pop1_shift_reduce_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += frontier_states
                .iter()
                .filter(|&&state| matches!(table.action(state, token), Some(Action::Reduce(..))))
                .count() as u32;
            profile.n_nondet_merges += frontier_states.len() as u32;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_pop1_stackshift_shift_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_pop1_reduce_plus_stackshift_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += 1;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_single_alt_pop1_common_suffix_stackshift_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        let mut next = ParserGSS::empty();

        for state in frontier_states {
            profile.n_nondet_branches += 1;
            let action = table.action(state, token);
            if let Some(trace) = profile.trace.as_mut() {
                let branch_gss = closure.isolate(Some(state));
                if let Some(wave) = trace.nondet_waves.last_mut() {
                    wave.branches
                        .push(trace_action_summary(table, state, &branch_gss, action));
                }
            }
            let Some(action) = action else {
                continue;
            };
            profile.n_nondet_isolates += 1;
            let pure_reduce = matches!(action, Action::Reduce(..));
            let mut isolated = closure.isolate(Some(state));
            match action {
                Action::Shift(target, is_replace) => {
                    profile.n_nondet_merges += 1;
                    if *is_replace {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated.popn(1));
                    } else {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated);
                    }
                    continue;
                }
                Action::ReplaceShifts(targets) => {
                    profile.n_nondet_merges += 1;
                    merge_into(&mut shifted, apply_replace_shifts(isolated, targets));
                    continue;
                }
                Action::StackShifts(shifts) => {
                    profile.n_nondet_merges += 1;
                    merge_into(&mut shifted, apply_stack_shifts(isolated, shifts));
                    continue;
                }
                Action::GuardedStackShifts(shifts) => {
                    profile.n_nondet_merges += 1;
                    let branch = if let Some(stack) = isolated.try_virtual_stack() {
                        apply_guarded_stack_shifts_from_vstack(
                            stack,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    } else {
                        apply_guarded_stack_shifts(
                            isolated,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    };
                    merge_into(&mut shifted, branch);
                    continue;
                }
                _ => {}
            }
            let reduce_base = if pure_reduce { None } else { Some(isolated.clone()) };
            if !pure_reduce {
                let det_start = Instant::now();
                if advance_deterministically(table, &mut isolated, token) {
                    profile.nondet_det_ns += det_start.elapsed().as_nanos() as u64;
                    profile.n_nondet_merges += 1;
                    merge_into(&mut shifted, isolated);
                    continue;
                }
                profile.nondet_det_ns += det_start.elapsed().as_nanos() as u64;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace() && !table.forwarded_shifts.contains(&(state, token));
                profile.n_nondet_merges += 1;
                if is_replace {
                    shifted = shifted.absorb_push_same_acc(target, &isolated.popn(1));
                } else {
                    shifted = shifted.absorb_push_same_acc(target, &isolated);
                }
            }

            if let Action::ReplaceShifts(targets) = action {
                profile.n_nondet_merges += 1;
                merge_into(&mut shifted, apply_replace_shifts(isolated.clone(), targets));
            }
            if let Action::StackShifts(shifts) = action {
                profile.n_nondet_merges += 1;
                merge_into(&mut shifted, apply_stack_shifts(isolated.clone(), shifts));
            }

            action.for_each_reduce(|nt, len| {
                let reduce_base = reduce_base.as_ref().unwrap_or(&isolated);
                for (base, target, is_replace) in
                    reduce_branches_from_isolated(table, reduce_base, nt, len as usize)
                {
                    profile.n_nondet_reduce_ops += 1;
                    let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
                    if det_ok {
                        profile.n_nondet_merges += 1;
                        match branch {
                            AdvancedBranch::Stack(stack) => {
                                let current = std::mem::replace(&mut shifted, ParserGSS::empty());
                                shifted = current.absorb_vstack_same_acc_owned(stack);
                            }
                            AdvancedBranch::Gss(branch) => {
                                merge_into(&mut shifted, branch);
                            }
                        }
                    } else {
                        profile.n_nondet_merges += 1;
                        merge_into(&mut next, branch.into_gss());
                    }
                }
            });
        }

        if next.is_empty() {
            return shifted;
        }
        closure = next;
    }
}



fn provider_push_local_states<P: ParserActionProvider>(
    provider: &P,
    scope: u32,
    mut gss: ParserGSS,
    states: &[u32],
) -> ParserGSS {
    for &local_state in states {
        let Some(state) = provider.scope_state(scope, local_state) else {
            return ParserGSS::empty();
        };
        gss = gss.push(state);
    }
    gss
}

fn apply_provider_guarded_stack_shifts<P: ParserActionProvider>(
    provider: &P,
    scope: u32,
    gss: ParserGSS,
    shifts: &[GuardedStackShift],
) -> ParserGSS {
    let mut out = ParserGSS::empty();
    for shift in shifts {
        debug_assert!(shift.guards.windows(2).all(|w| w[0].pop <= w[1].pop));
        debug_assert!(shift.guards.iter().all(|guard| guard.pop <= shift.pop));
        let mut base = gss.clone();
        let mut depth = 0u32;
        let mut dead = false;
        for guard in &shift.guards {
            if guard.pop < depth {
                dead = true;
                break;
            }
            base = base.popn((guard.pop - depth) as isize);
            if base.is_empty() {
                dead = true;
                break;
            }
            let mut filtered = ParserGSS::empty();
            for &local_state in &guard.states {
                let Some(state) = provider.scope_state(scope, local_state) else {
                    continue;
                };
                merge_into(&mut filtered, base.isolate(Some(state)));
            }
            base = filtered;
            if base.is_empty() {
                dead = true;
                break;
            }
            depth = guard.pop;
        }
        if dead || shift.pop < depth {
            continue;
        }
        let branch = provider_push_local_states(
            provider,
            scope,
            base.popn((shift.pop - depth) as isize),
            &shift.pushes,
        );
        merge_into(&mut out, branch);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProviderAdvanceMode {
    Advance,
    Completion,
}

struct ProviderAdvanceResult {
    shifted: ParserGSS,
    accepted: bool,
}

/// Bounded deterministic reductions on the existing virtual stack.
/// Only an unambiguous initial reduction enters this lane. After a proved
/// prefix, branching, guards, acceptance, extra effects or a hidden lower floor
/// continue in the original traversal with the lookahead still unconsumed.
/// Primitive failures remain transactional. The budget belongs to the complete
/// enclosing advance, so a resumed branch cannot obtain a new allowance.
enum ProviderReductionPrefix {
    Shifted(ParserGSS),
    /// No lookahead consumed yet; continue the ordinary reduction frontier.
    Pending(ParserGSS),
}

#[inline]
fn suspend_provider_reductions<const RESUME: bool>(
    stack: VirtualStack<u32, TerminalsDisallowed>, completed_reductions: usize,
) -> Option<ProviderReductionPrefix> {
    // A zero-progress resume would create a loop at the same unsupported action.
    (RESUME && completed_reductions != 0)
        .then(|| ProviderReductionPrefix::Pending(stack.into_gss()))
}

#[cfg(test)]
fn try_provider_reduction_prefix<P: ParserActionProvider>(
    provider: &P, stack: VirtualStack<u32, TerminalsDisallowed>,
    symbol: P::Symbol, first: &ProvidedAction<'_>, remaining_steps: &mut usize,
) -> Option<ParserGSS> {
    match try_provider_reduction_prefix_impl::<P, false>(provider, stack, symbol, first, remaining_steps)? {
        ProviderReductionPrefix::Shifted(result) => Some(result),
        ProviderReductionPrefix::Pending(_) => unreachable!("transactional policy never resumes"),
    }
}

fn try_provider_reduction_prefix_impl<P: ParserActionProvider, const RESUME: bool>(
    provider: &P,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
    symbol: P::Symbol,
    first: &ProvidedAction<'_>,
    remaining_steps: &mut usize,
) -> Option<ProviderReductionPrefix> {
    let allowance = *remaining_steps;
    let mut completed_reductions = 0usize;
    for step in 0..allowance {
        *remaining_steps -= 1;
        let top = *stack.top()?;
        let later;
        let provided = if step == 0 { first } else {
            later = provider.action(top, symbol);
            match later.as_ref() {
                Some(action) => action,
                None => return Some(ProviderReductionPrefix::Shifted(ParserGSS::empty())),
            }
        };
        if !provided.extra_stack_shifts.is_empty() {
            return suspend_provider_reductions::<RESUME>(stack, completed_reductions);
        }
        match &provided.action {
            ProvidedActionRef::Identity => return Some(ProviderReductionPrefix::Shifted(stack.into_gss())),
            ProvidedActionRef::Call { parent_target, child_start, replace } => {
                if *replace && stack.pop(1) != 0 { return None; }
                stack.push(*parent_target);
                stack.push(*child_start);
                return Some(ProviderReductionPrefix::Shifted(stack.into_gss()));
            }
            ProvidedActionRef::Return { pop } => {
                if *pop as usize > stack.len() {
                    return suspend_provider_reductions::<RESUME>(stack, completed_reductions);
                }
                stack.pop(*pop as usize);
                return Some(ProviderReductionPrefix::Shifted(stack.into_gss()));
            }
            ProvidedActionRef::Local { scope, action } => match action {
                Action::Skip => return Some(ProviderReductionPrefix::Shifted(stack.into_gss())),
                Action::Shift(target, replace) => {
                    let Some(target) = provider.scope_state(*scope, *target) else {
                        return Some(ProviderReductionPrefix::Shifted(ParserGSS::empty()));
                    };
                    if *replace {
                        if !stack.replace_top(target) { return None; }
                    } else { stack.push(target); }
                    return Some(ProviderReductionPrefix::Shifted(stack.into_gss()));
                }
                Action::Reduce(nonterminal, count) => {
                    // A reduction needs the concrete predecessor for its goto.
                    // Do not speculate across a hidden or ambiguous GSS floor.
                    if *count as usize >= stack.len() {
                        return suspend_provider_reductions::<RESUME>(stack, completed_reductions);
                    }
                    stack.pop(*count as usize);
                    let predecessor = *stack.top()?;
                    let Some((target, replace)) = provider.goto_target(
                        provided.reduction_scope, predecessor, *nonterminal,
                    ) else { return Some(ProviderReductionPrefix::Shifted(ParserGSS::empty())); };
                    if replace {
                        if !stack.replace_top(target) { return None; }
                    } else { stack.push(target); }
                    completed_reductions += 1;
                }
                _ => return suspend_provider_reductions::<RESUME>(stack, completed_reductions),
            },
        }
    }
    suspend_provider_reductions::<RESUME>(stack, completed_reductions)
}

/// Enabled unless an explicit override declines the bounded optimization.
/// Keeping parsing independent of the process environment permits deterministic
/// policy tests without changing flags underneath concurrently running tests.
fn provider_reduction_policy_value(value: Option<&str>) -> bool {
    value.is_none_or(|value| matches!(value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"))
}

fn advance_provider_traversal<P: ParserActionProvider>(
    provider: &P,
    closure: ParserGSS,
    symbol: P::Symbol,
    mode: ProviderAdvanceMode,
) -> ProviderAdvanceResult {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if mode == ProviderAdvanceMode::Advance
        && *ENABLED.get_or_init(|| provider_reduction_policy_value(std::env::var("GLRMASK_PROVIDER_REDUCTION_PREFIX").ok().as_deref()))
    {
        static RESUME: OnceLock<bool> = OnceLock::new();
        if *RESUME.get_or_init(|| provider_reduction_policy_value(std::env::var("GLRMASK_PROVIDER_RESUME_REDUCTIONS").ok().as_deref())) {
            advance_provider_traversal_with_policy::<P, true, true>(provider, closure, symbol, mode)
        } else {
            advance_provider_traversal_impl::<P, true>(provider, closure, symbol, mode)
        }
    } else {
        advance_provider_traversal_impl::<P, false>(provider, closure, symbol, mode)
    }
}

fn advance_provider_traversal_impl<P: ParserActionProvider, const REDUCTION_PREFIX: bool>(
    provider: &P, closure: ParserGSS, symbol: P::Symbol, mode: ProviderAdvanceMode,
) -> ProviderAdvanceResult {
    advance_provider_traversal_with_policy::<P, REDUCTION_PREFIX, false>(provider, closure, symbol, mode)
}

/// Changing reduction-wave grouping is harmless only under a fixed label.
/// The current GSS accumulator merge is observable for mixed labels, so do not
/// reorder those frontiers. An Interface certifies one shared empty label over
/// its complete lower graph in O(1); unknown representations use the reference.
#[inline]
fn provider_reduction_input_is_uniform_empty(stack: &ParserGSS) -> bool {
    stack.single_interface_lower_id().is_some()
        && stack.all_accs_satisfy(TerminalsDisallowed::is_empty)
}

// Keep the eager initializer available as an explicit diagnostic fallback.
// The default avoids work only when the exact first action is absent, or when
// this traversal never needs a Completion semantic key.
fn provider_light_start_policy(value: Option<&str>) -> bool {
    value.is_none_or(|value| matches!(value, "1" | "true"))
}

fn advance_provider_traversal_with_policy<P: ParserActionProvider, const REDUCTION_PREFIX: bool, const RESUME: bool>(
    provider: &P,
    closure: ParserGSS,
    symbol: P::Symbol,
    mode: ProviderAdvanceMode,
) -> ProviderAdvanceResult {
    static LIGHT_START: OnceLock<bool> = OnceLock::new();
    if *LIGHT_START.get_or_init(|| {
        provider_light_start_policy(std::env::var("GLRMASK_PROVIDER_LIGHT_START").ok().as_deref())
    })
    {
        advance_provider_traversal_with_initialization::<P, REDUCTION_PREFIX, RESUME, true>(
            provider, closure, symbol, mode,
        )
    } else {
        advance_provider_traversal_with_initialization::<P, REDUCTION_PREFIX, RESUME, false>(
            provider, closure, symbol, mode,
        )
    }
}

// Reuse the first queried action so live frontiers do not pay for a second
// probe. A missing action for the sole visible top is exactly the reference
// first wave with no shifted/reduced/accepted output. An epsilon alternative
// has no visible top and cannot add a transition for this symbol either.
fn advance_provider_traversal_with_initialization<
    P: ParserActionProvider, const REDUCTION_PREFIX: bool, const RESUME: bool,
    const LIGHT_START: bool,
>(
    provider: &P, mut closure: ParserGSS, symbol: P::Symbol, mode: ProviderAdvanceMode,
) -> ProviderAdvanceResult {
    let mut first_action = if LIGHT_START {
        if let Some(state) = closure.single_top_value() {
            match provider.action(state, symbol) {
                Some(provided) => Some((state, provided)),
                None => return ProviderAdvanceResult { shifted: ParserGSS::empty(), accepted: false },
            }
        } else { None }
    } else { None };
    let reduction_prefix_eligible = REDUCTION_PREFIX
        && mode == ProviderAdvanceMode::Advance
        && provider_reduction_input_is_uniform_empty(&closure);
    let mut reduction_steps_remaining = 64usize;
    let mut shifted = ParserGSS::empty();
    let mut accepted = false;
    let mut visited = FxHashSet::<u32>::default();
    // Only Completion reduction cycles use semantic keys. Advance and
    // immediately resolved Completion queries must not initialize that table.
    let mut key_interner = (!LIGHT_START)
        .then(GssSemanticKeyInterner::<u32, TerminalsDisallowed>::new);

    loop {
        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            let provided = if first_action.as_ref().is_some_and(|(top, _)| *top == state) {
                first_action.take().map(|(_, provided)| provided)
            } else {
                provider.action(state, symbol)
            };
            let Some(provided) = provided else { continue; };
            let isolated = closure.isolate(Some(state));
            if reduction_prefix_eligible
                && reduction_steps_remaining != 0 && provided.extra_stack_shifts.is_empty()
                && matches!(&provided.action, ProvidedActionRef::Local { action: Action::Reduce(..), .. })
                && let Some(stack) = isolated.try_virtual_stack()
                && let Some(advanced) = try_provider_reduction_prefix_impl::<P, RESUME>(
                    provider, stack, symbol, &provided, &mut reduction_steps_remaining,
                )
            {
                match advanced {
                    ProviderReductionPrefix::Shifted(stack) => merge_into(&mut shifted, stack),
                    ProviderReductionPrefix::Pending(stack) => merge_into(&mut next, stack),
                }
                continue;
            }

            if mode == ProviderAdvanceMode::Advance {
                for shift in &provided.extra_stack_shifts {
                    let branch = push_states(isolated.clone().popn(shift.pop as isize), &shift.pushes);
                    merge_into(&mut shifted, branch);
                }
            }

            match provided.action {
                ProvidedActionRef::Local { scope, action } => {
                    let is_accept =
                        matches!(action, Action::Accept | Action::Split { accept: true, .. });
                    if is_accept {
                        accepted = true;
                        if mode == ProviderAdvanceMode::Completion {
                            return ProviderAdvanceResult { shifted, accepted: true };
                        }
                    }

                    if mode == ProviderAdvanceMode::Advance {
                        match action {
                            Action::GuardedStackShifts(shifts) => {
                                merge_into(
                                    &mut shifted,
                                    apply_provider_guarded_stack_shifts(
                                        provider,
                                        scope,
                                        isolated.clone(),
                                        shifts,
                                    ),
                                );
                            }
                            _ => {
                                action.for_each_stack_shift(|pop, pushes| {
                                    let branch = provider_push_local_states(
                                        provider,
                                        scope,
                                        isolated.clone().popn(pop as isize),
                                        pushes,
                                    );
                                    merge_into(&mut shifted, branch);
                                });
                            }
                        }
                    }

                    action.for_each_reduce(|nt, len| {
                        for (goto_from, base) in
                            reduce_sources_from_isolated(&isolated, len as usize)
                        {
                            let Some((target, is_replace)) = provider.goto_target(
                                provided.reduction_scope,
                                goto_from,
                                nt,
                            ) else {
                                continue;
                            };
                            let branch = if is_replace {
                                base.popn(1).push(target)
                            } else {
                                base.push(target)
                            };
                            if mode == ProviderAdvanceMode::Completion {
                                if !branch.is_empty() {
                                    let key = key_interner.get_or_insert_with(GssSemanticKeyInterner::new).key(&branch);
                                    if visited.insert(key) {
                                        merge_into(&mut next, branch);
                                    }
                                }
                            } else {
                                merge_into(&mut next, branch);
                            }
                        }
                    });
                }
                ProvidedActionRef::Identity => {
                    if mode == ProviderAdvanceMode::Advance {
                        merge_into(&mut shifted, isolated.clone());
                    }
                }
                ProvidedActionRef::Call {
                    parent_target,
                    child_start,
                    replace,
                } => {
                    if mode == ProviderAdvanceMode::Advance {
                        let branch = isolated
                            .clone()
                            .popn(isize::from(replace))
                            .push(parent_target)
                            .push(child_start);
                        merge_into(&mut shifted, branch);
                    }
                }
                ProvidedActionRef::Return { pop } => {
                    if mode == ProviderAdvanceMode::Advance {
                        merge_into(&mut shifted, isolated.clone().popn(pop as isize));
                    }
                }
            }
        }

        if next.is_empty() {
            return ProviderAdvanceResult { shifted, accepted };
        }
        closure = if mode == ProviderAdvanceMode::Completion {
            close_provider_control_stacks(provider, &next)
        } else {
            next
        };
    }
}

/// Canonical provider-driven GLR advance over the existing `u32` parser GSS.
/// Local component actions remain borrowed; state IDs are injected into the
/// compact disjoint-union coordinate only when pushed.
pub fn advance_stacks_with_provider<P: ParserActionProvider>(
    provider: &P,
    closure: ParserGSS,
    symbol: P::Symbol,
) -> ParserGSS {
    advance_provider_traversal(provider, closure, symbol, ProviderAdvanceMode::Advance).shifted
}

/// Close a provider-driven frontier under provider-owned zero-width controls.
pub fn close_provider_control_stacks<P: ParserActionProvider>(
    provider: &P,
    stack: &ParserGSS,
) -> ParserGSS {
    if stack.is_empty() {
        return stack.clone();
    }
    let mut closure = stack.clone();
    let pass_limit = provider.state_count_hint().saturating_mul(4).saturating_add(2);
    let mut symbols = SmallVec::<[P::Symbol; 4]>::new();
    for _ in 0..pass_limit {
        symbols.clear();
        for state in closure.peek_values() {
            provider.control_symbols(state, &mut symbols);
        }
        let mut unique = SmallVec::<[P::Symbol; 4]>::new();
        for symbol in symbols.drain(..) {
            if !unique.contains(&symbol) {
                unique.push(symbol);
            }
        }
        if unique.is_empty() {
            return closure;
        }
        let mut additions = ParserGSS::empty();
        for symbol in unique {
            merge_into(
                &mut additions,
                advance_stacks_with_provider(provider, closure.clone(), symbol),
            );
        }
        if additions.is_empty() {
            return closure;
        }
        let merged = closure.merge(&additions);
        if merged == closure {
            return closure;
        }
        closure = merged;
    }
    closure
}

/// Advance one visible/provider symbol from a control-closed frontier and
/// restore zero-width closure afterwards.
pub fn advance_provider_control_closed_stacks<P: ParserActionProvider>(
    provider: &P,
    stack: &ParserGSS,
    symbol: P::Symbol,
) -> ParserGSS {
    let advanced = advance_stacks_with_provider(provider, stack.clone(), symbol);
    close_provider_control_stacks(provider, &advanced)
}

/// Exact semantic admission through the provider. This is deliberately the
/// reference implementation; provider-specific row/guard fast paths can replace
/// it later without changing the contract.
/// Fast path for [`stack_may_advance_on_with_provider`]: after control
/// closure, return true for any top with `Identity` or a zero-pop shift whose
/// nonempty pushes all map via `scope_state` (pop-0 on a live branch provably
/// yields a shifted branch). Everything else stays on the exact reference
/// path (`advance_stacks_with_provider(...).is_empty()`).
pub fn stack_may_advance_on_with_provider<P: ParserActionProvider>(
    provider: &P,
    stack: &ParserGSS,
    symbol: P::Symbol,
) -> bool {
    if stack.is_empty() {
        return false;
    }
    let closed = close_provider_control_stacks(provider, stack);
    let tops = closed.peek_values();
    provider_closed_symbol_may_advance(provider, &closed, &tops, symbol)
}

/// Exact set admission using the same provider action/advance machinery as
/// scalar admission. The frontier is closed once for the whole candidate set,
/// not once per lexer-future terminal. No grammar flattening or new tables are
/// required, and ordinary table-based admission remains unchanged.
pub fn stack_may_advance_on_any_with_provider<P: ParserActionProvider>(
    provider: &P,
    stack: &ParserGSS,
    symbols: impl IntoIterator<Item = P::Symbol>,
) -> bool {
    if stack.is_empty() {
        return false;
    }
    let closed = close_provider_control_stacks(provider, stack);
    let tops = closed.peek_values();
    symbols.into_iter().any(|symbol| {
        provider_closed_symbol_may_advance(provider, &closed, &tops, symbol)
    })
}

/// Visit candidate keys whose provider symbols admit an exact continuation.
/// Multiple symbols may share a key (for a scoped alias); callers can union
/// the reported keys. Closure and stack-top inspection are reused for the
/// complete batch, while predecessor-feasible reductions use the unchanged
/// canonical provider traversal.
pub fn for_each_admitted_symbol_with_provider<P: ParserActionProvider, K>(
    provider: &P,
    stack: &ParserGSS,
    symbols: impl IntoIterator<Item = (K, P::Symbol)>,
    mut admit: impl FnMut(K),
) {
    if stack.is_empty() {
        return;
    }
    let closed = close_provider_control_stacks(provider, stack);
    let tops = closed.peek_values();
    for (key, symbol) in symbols {
        if provider_closed_symbol_may_advance(provider, &closed, &tops, symbol) {
            admit(key);
        }
    }
}

/// Find an exactly admitted candidate satisfying an additional predicate.
/// Control closure and stack-top inspection happen once; the predicate is
/// evaluated only for parser-admissible candidates and stops at the first
/// witness. This lets a lexer defer expensive exact residual checks without
/// admitting any token from an over-approximate support set alone.
pub fn find_admitted_symbol_with_provider<P: ParserActionProvider, K>(
    provider: &P,
    stack: &ParserGSS,
    symbols: impl IntoIterator<Item = (K, P::Symbol)>,
    mut matches: impl FnMut(&K) -> bool,
) -> Option<K> {
    if stack.is_empty() {
        return None;
    }
    let closed = close_provider_control_stacks(provider, stack);
    let tops = closed.peek_values();
    for (key, symbol) in symbols {
        if provider_closed_symbol_may_advance(provider, &closed, &tops, symbol)
            && matches(&key)
        {
            return Some(key);
        }
    }
    None
}

fn provider_closed_symbol_may_advance<P: ParserActionProvider>(
    provider: &P,
    closed: &ParserGSS,
    tops: &[u32],
    symbol: P::Symbol,
) -> bool {
    let mut has_action = false;
    for &top in tops {
        let Some(provided) = provider.action(top, symbol) else {
            continue;
        };
        has_action = true;
        if !provided.extra_stack_shifts.is_empty() {
            continue;
        }
        match provided.action {
            ProvidedActionRef::Identity => return true,
            ProvidedActionRef::Local { scope, action } => {
                let mut sufficient = false;
                action.for_each_stack_shift(|pop, pushes| {
                    if sufficient || pop != 0 || pushes.is_empty() {
                        return;
                    }
                    if pushes
                        .iter()
                        .all(|&local| provider.scope_state(scope, local).is_some())
                    {
                        sufficient = true;
                    }
                });
                if sufficient {
                    return true;
                }
            }
            ProvidedActionRef::Call { .. } | ProvidedActionRef::Return { .. } => {}
        }
    }
    has_action && !advance_stacks_with_provider(provider, closed.clone(), symbol).is_empty()
}

/// Provider equivalent of `stacks_finished`: after zero-width closure, report
/// whether any top can reach a genuine accepting action on the caller-supplied
/// outer EOF symbol via predecessor-feasible reductions.
pub fn stacks_finished_with_provider<P: ParserActionProvider>(
    provider: &P,
    stack: &ParserGSS,
    eof_symbol: P::Symbol,
) -> bool {
    if stack.is_empty() {
        return false;
    }
    let closed = close_provider_control_stacks(provider, stack);
    advance_provider_traversal(provider, closed, eof_symbol, ProviderAdvanceMode::Completion).accepted
}

/// Provider equivalent of linker-control closure. The caller supplies the
/// zero-width symbols explicitly; composition providers therefore do not need
/// to manufacture a global terminal coordinate merely for control actions.
pub fn close_control_stacks_with_provider<P: ParserActionProvider>(
    provider: &P,
    stack: &ParserGSS,
    controls: &[P::Symbol],
) -> ParserGSS
where
    P::Symbol: std::fmt::Debug,
{
    if controls.is_empty() || stack.is_empty() {
        return stack.clone();
    }

    let mut closure = stack.clone();
    let pass_limit = provider
        .state_count_hint()
        .saturating_mul(controls.len().max(1))
        .saturating_add(2);
    for _ in 0..pass_limit {
        let mut additions = ParserGSS::empty();
        for &control in controls {
            merge_into(
                &mut additions,
                advance_stacks_with_provider(provider, closure.clone(), control),
            );
        }
        if additions.is_empty() {
            return closure;
        }
        let next = closure.merge(&additions);
        if next == closure {
            return closure;
        }
        closure = next;
    }
    panic!(
        "provider control closure did not converge: states={} controls={controls:?}",
        provider.state_count_hint(),
    );
}

fn advance_nondeterministically(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    let mut shifted = ParserGSS::empty();

    loop {
        if let Some(shifted_wave) =
            try_advance_single_active_pop1_stackshift_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_single_active_pop1_reduce_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_pop1_common_base_reduce_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_mixed_top_pop1_shift_reduce_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_pop1_stackshift_shift_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_pop1_reduce_plus_stackshift_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if nondet_wave_specializations_enabled()
            && let Some(shifted_wave) = try_advance_single_alt_pop1_common_suffix_stackshift_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            let Some(action) = table.action(state, token) else {
                continue;
            };
            let pure_reduce = matches!(action, Action::Reduce(..));
            let mut isolated = closure.isolate(Some(state));
            match action {
                Action::Shift(target, is_replace) => {
                    if *is_replace {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated.popn(1));
                    } else {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated);
                    }
                    continue;
                }
                Action::ReplaceShifts(targets) => {
                    merge_into(&mut shifted, apply_replace_shifts(isolated, targets));
                    continue;
                }
                Action::StackShifts(shifts) => {
                    merge_into(&mut shifted, apply_stack_shifts(isolated, shifts));
                    continue;
                }
                Action::GuardedStackShifts(shifts) => {
                    let branch = if let Some(stack) = isolated.try_virtual_stack() {
                        apply_guarded_stack_shifts_from_vstack(
                            stack,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    } else {
                        apply_guarded_stack_shifts(
                            isolated,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    };
                    merge_into(&mut shifted, branch);
                    continue;
                }
                _ => {}
            }
            let reduce_base = if pure_reduce { None } else { Some(isolated.clone()) };
            if !pure_reduce && advance_deterministically(table, &mut isolated, token) {
                merge_into(&mut shifted, isolated);
                continue;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace() && !table.forwarded_shifts.contains(&(state, token));
                if is_replace {
                    shifted = shifted.absorb_push_same_acc(target, &isolated.popn(1));
                } else {
                    shifted = shifted.absorb_push_same_acc(target, &isolated);
                }
            }

            if let Action::ReplaceShifts(targets) = action {
                merge_into(&mut shifted, apply_replace_shifts(isolated.clone(), targets));
            }
            if let Action::StackShifts(shifts) = action {
                merge_into(&mut shifted, apply_stack_shifts(isolated.clone(), shifts));
            }

            action.for_each_reduce(|nt, len| {
                let reduce_base = reduce_base.as_ref().unwrap_or(&isolated);
                for (base, target, is_replace) in
                    reduce_branches_from_isolated(table, reduce_base, nt, len as usize)
                {
                    let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
                    if det_ok {
                        match branch {
                            AdvancedBranch::Stack(stack) => {
                                let current = std::mem::replace(&mut shifted, ParserGSS::empty());
                                shifted = current.absorb_vstack_same_acc_owned(stack);
                            }
                            AdvancedBranch::Gss(branch) => {
                                merge_into(&mut shifted, branch);
                            }
                        }
                    } else {
                        merge_into(&mut next, branch.into_gss());
                    }
                }
            });
        }

        if next.is_empty() {
            return shifted;
        }
        closure = next;
    }
}

/// Standard LR reduce loop for the deterministic case.
///
/// When the GSS frontier is a single linear chain (no ambiguity), the GSS
/// degenerates to an ordinary flat parse stack. This applies the textbook
/// LR reduce loop directly: inspect the top state's action, pop |rhs|
/// symbols, push the goto target, repeat - until a non-reduce action is
/// reached or the chain becomes ambiguous.
///
/// If this deterministic pass ends in a pure shift, it performs that shift
/// itself and returns true to signal that the parser step is finished.
/// Otherwise it mutates `gss` and returns false so the caller can continue
/// with the nondeterministic reduce closure.
fn advance_deterministically(
    table: &GLRTable,
    gss: &mut ParserGSS,
    token: TerminalID,
) -> bool {
    let Some(mut stack) = gss
        .try_virtual_stack()
        .or_else(|| single_concrete_path_as_vstack(gss))
    else {
        return false;
    };

    loop {
        let Some(&state) = stack.top() else {
            break;
        };

        match table.action(state, token) {
            Some(Action::Skip) => {
                *gss = stack.into_gss();
                return true;
            }
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    if rhs_len == 1 {
                        if let Some(goto_from) = stack.parent_of_top() {
                            match table.goto_target(goto_from, *nt) {
                                Some((target, false)) if stack.replace_top(target) => continue,
                                Some((target, true)) => {
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    *gss = ParserGSS::empty();
                                    return false;
                                }
                            }
                        }
                    }

                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                            *gss = ParserGSS::empty();
                            return false;
                        }
                    }
                } else {
                    let popped = stack.into_gss_after_popping(rhs_len);
                    let mut shifts = SmallVec::<[FloorCrossShift; 8]>::new();
                    for goto_from in popped.peek_values() {
                        if let Some((target, is_replace)) = table.goto_target(goto_from, *nt) {
                            shifts.push((goto_from, target, is_replace));
                        }
                    }
                    let rebuilt = rebuild_floor_cross_from_shifts(popped, shifts);
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        *gss = rebuilt;
                        return false;
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                    *gss = stack.into_gss();
                } else {
                    stack.push(*target);
                    *gss = stack.into_gss();
                }
                return true;
            }
            Some(Action::ReplaceShifts(targets)) => {
                *gss = apply_replace_shifts(stack.into_gss(), targets);
                return true;
            }
            Some(Action::StackShifts(shifts)) => {
                *gss = apply_stack_shifts(stack.into_gss(), shifts);
                return true;
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                *gss = apply_guarded_stack_shifts_from_vstack(
                    stack,
                    shifts,
                    table.guarded_shift_index(state, token),
                );
                return true;
            }
            Some(Action::Split { shift, reduces, accept: false }) => {
                if let Some(shifted) =
                    advance_split_from_vstack(table, &stack, token, *shift, reduces)
                {
                    *gss = shifted;
                    return true;
                }
                break;
            }
            Some(Action::Split { .. }) => break,
            Some(Action::Accept) => break,
            None => break,
        }
    }

    *gss = stack.into_gss();
    false
}

/// Precise predicate for whether this parser stack can advance on `token`.
///
/// Returns `true` if and only if at least one current parser path can definitely
/// advance on the given terminal. Returns `false` if no current parser path can
/// advance.
///
/// For `RowPresenceExact`, `advance` was captured from the precise parser row
/// before stack-effect lowering. Guarded stack shifts therefore select the exact
/// execution effect; they do not weaken admission. `ExactSimulation` tables use
/// reduction/guard simulation instead.
///
/// TODO: Rename this eventually, e.g. to `stack_can_advance_on`. The current
/// `may_advance` name sounds like a speculative approximation, but this is an
/// exact applicability predicate.
pub fn stack_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    let control_closed = (!table.control_terminals.is_empty())
        .then(|| control_closure(table, stack));
    let stack = control_closed.as_ref().unwrap_or(stack);
    stack_may_advance_on_control_closed(table, stack, token)
}

/// Exact admission for a frontier already closed under linker controls.
pub fn stack_may_advance_on_control_closed(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> bool {
    if table.admission_policy == AdmissionPolicy::ExactSimulation {
        return exact_admission_may_advance_on(table, stack, token);
    }

    let admitted = stack
        .peek_values()
        .into_iter()
        .any(|state| table.advance_row_allows(state, token));
    if assert_row_presence_exact_enabled() {
        let simulated = exact_admission_may_advance_on(table, stack, token);
        assert_eq!(
            admitted,
            simulated,
            "RowPresenceExact mismatch for terminal {token}: construction={:?} stacks={:?}",
            table.construction,
            stack.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        );
    }
    admitted
}

type ExactAdmissionSemanticKey = u32;

type ExactAdmissionKeyInterner = GssSemanticKeyInterner<u32, TerminalsDisallowed>;

/// What `exact_admission_may_advance_on` is asked to prove.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExactAdmissionMode {
    /// Ordinary admission: any action that can advance on the token counts,
    /// including shifts, skips, replace-shifts and stack shifts.
    MayAdvance,
    /// Completion: only a genuine accepting action counts. A shift, skip,
    /// replace-shift or stack shift is progress, not completion; a feasible
    /// reduction chain that reaches an accept is still completion.
    Completion,
}

fn exact_admission_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    exact_admission_may_advance_on_mode(table, stack, token, ExactAdmissionMode::MayAdvance)
}

/// Completion on EOF: a genuine accepting action or a GSS-predecessor-feasible
/// reduction chain to one. Never a bare shift/skip/replace.
fn exact_admission_completes_on_eof(table: &GLRTable, stack: &ParserGSS) -> bool {
    exact_admission_may_advance_on_mode(table, stack, EOF, ExactAdmissionMode::Completion)
}

fn exact_admission_may_advance_on_mode(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
    mode: ExactAdmissionMode,
) -> bool {
    let may_advance = mode == ExactAdmissionMode::MayAdvance;
    let mut queue = VecDeque::<ParserGSS>::new();
    let mut visited = FxHashSet::<ExactAdmissionSemanticKey>::default();
    let mut key_interner = ExactAdmissionKeyInterner::new();

    for state in stack.peek_values() {
        exact_admission_enqueue_frontier(
            stack.isolate(Some(state)),
            &mut queue,
            &mut visited,
            &mut key_interner,
        );
    }

    while let Some(frontier) = queue.pop_front() {
        for source_state in frontier.peek_values() {
            if !table.advance_row_allows(source_state, token) {
                continue;
            }
            let isolated = frontier.isolate(Some(source_state));
            let Some(action) = table.action(source_state, token) else {
                continue;
            };
            match action {
                Action::Shift(..) | Action::ReplaceShifts(_) | Action::Skip => {
                    if may_advance {
                        return true;
                    }
                }
                Action::StackShifts(shifts) => {
                    if may_advance && !apply_stack_shifts(isolated.clone(), shifts).is_empty() {
                        return true;
                    }
                }
                Action::GuardedStackShifts(shifts) => {
                    if may_advance && stack_may_apply_guarded_shifts(&isolated, shifts) {
                        return true;
                    }
                }
                Action::Reduce(nt, len) => {
                    exact_admission_enqueue_reduce(
                        table,
                        &isolated,
                        *nt,
                        *len as usize,
                        &mut queue,
                        &mut visited,
                        &mut key_interner,
                    );
                }
                Action::Split {
                    shift,
                    reduces,
                    accept,
                } => {
                    if *accept && token == EOF {
                        return true;
                    }
                    if may_advance && shift.is_some() {
                        return true;
                    }
                    for &(nt, len) in reduces {
                        exact_admission_enqueue_reduce(
                            table,
                            &isolated,
                            nt,
                            len as usize,
                            &mut queue,
                            &mut visited,
                            &mut key_interner,
                        );
                    }
                }
                Action::Accept => {
                    if token == EOF {
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn exact_admission_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    if terminals.is_empty() {
        return false;
    }

    let mut queue = VecDeque::<(ParserGSS, BitSet)>::new();
    let mut visited = FxHashMap::<ExactAdmissionSemanticKey, BitSet>::default();
    let mut key_interner = ExactAdmissionKeyInterner::new();

    for state in stack.peek_values() {
        exact_admission_enqueue_frontier_any(
            stack.isolate(Some(state)),
            terminals,
            &mut queue,
            &mut visited,
            &mut key_interner,
        );
    }

    while let Some((frontier, frontier_terminals)) = queue.pop_front() {
        for source_state in frontier.peek_values() {
            if !table.advance_row_intersects(source_state, &frontier_terminals) {
                continue;
            }

            let isolated = frontier.isolate(Some(source_state));
            let mut pending_reduces = SmallVec::<[(u32, u32, BitSet); 8]>::new();

            if exact_admission_for_each_matching_action(
                table,
                source_state,
                &frontier_terminals,
                |terminal, terminal_bit, action| {
                    exact_admission_process_action_any(
                        &isolated,
                        terminal,
                        terminal_bit,
                        action,
                        frontier_terminals.len(),
                        &mut pending_reduces,
                    )
                },
            ) {
                return true;
            }

            for (nt, len, reduce_terminals) in pending_reduces {
                exact_admission_enqueue_reduce_any(
                    table,
                    &isolated,
                    nt,
                    len as usize,
                    &reduce_terminals,
                    &mut queue,
                    &mut visited,
                    &mut key_interner,
                );
            }
        }
    }

    false
}

fn exact_admission_admissible_terminals(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> BitSet {
    let mut admitted = BitSet::new(terminals.len());
    if terminals.is_empty() {
        return admitted;
    }

    let mut queue = VecDeque::<(ParserGSS, BitSet)>::new();
    let mut visited = FxHashMap::<ExactAdmissionSemanticKey, BitSet>::default();
    let mut key_interner = ExactAdmissionKeyInterner::new();

    for state in stack.peek_values() {
        exact_admission_enqueue_frontier_any(
            stack.isolate(Some(state)),
            terminals,
            &mut queue,
            &mut visited,
            &mut key_interner,
        );
    }

    while let Some((frontier, frontier_terminals)) = queue.pop_front() {
        let remaining = frontier_terminals.difference(&admitted);
        if remaining.is_empty() {
            continue;
        }
        for source_state in frontier.peek_values() {
            if !table.advance_row_intersects(source_state, &remaining) {
                continue;
            }

            let isolated = frontier.isolate(Some(source_state));
            let mut pending_reduces = SmallVec::<[(u32, u32, BitSet); 8]>::new();
            exact_admission_for_each_matching_action(
                table,
                source_state,
                &remaining,
                |terminal, terminal_bit, action| {
                    if exact_admission_process_action_any(
                        &isolated,
                        terminal,
                        terminal_bit,
                        action,
                        remaining.len(),
                        &mut pending_reduces,
                    ) {
                        admitted.set(terminal_bit);
                    }
                    false
                },
            );

            for (nt, len, reduce_terminals) in pending_reduces {
                let reduce_terminals = reduce_terminals.difference(&admitted);
                if reduce_terminals.is_empty() {
                    continue;
                }
                exact_admission_enqueue_reduce_any(
                    table,
                    &isolated,
                    nt,
                    len as usize,
                    &reduce_terminals,
                    &mut queue,
                    &mut visited,
                    &mut key_interner,
                );
            }
        }
    }

    admitted
}

fn exact_admission_for_each_matching_action(
    table: &GLRTable,
    state: u32,
    terminals: &BitSet,
    mut f: impl FnMut(TerminalID, usize, &Action) -> bool,
) -> bool {
    let Some(row) = table.action.get(state as usize) else {
        return false;
    };

    let active_count = terminals.count_ones();
    if active_count == 0 {
        return false;
    }

    if row.len() <= active_count {
        for (terminal, action) in row {
            let Some(bit) = exact_admission_terminal_bit(table, terminal, terminals.len()) else {
                continue;
            };
            if !terminals.contains(bit) || !table.advance_row_allows(state, terminal) {
                continue;
            }
            if f(terminal, bit, action) {
                return true;
            }
        }
        return false;
    }

    for bit in terminals.iter_ones() {
        let Some(terminal) = exact_admission_terminal_from_bit(table, bit) else {
            continue;
        };
        if !table.advance_row_allows(state, terminal) {
            continue;
        }
        if let Some(action) = table.action(state, terminal)
            && f(terminal, bit, action)
        {
            return true;
        }
    }

    false
}

fn exact_admission_process_action_any(
    isolated: &ParserGSS,
    terminal: TerminalID,
    terminal_bit: usize,
    action: &Action,
    terminals_len: usize,
    pending_reduces: &mut SmallVec<[(u32, u32, BitSet); 8]>,
) -> bool {
    match action {
        Action::Shift(..) | Action::ReplaceShifts(_) | Action::Skip => true,
        Action::StackShifts(shifts) => {
            !apply_stack_shifts(isolated.clone(), shifts).is_empty()
        }
        Action::GuardedStackShifts(shifts) => stack_may_apply_guarded_shifts(isolated, shifts),
        Action::Reduce(nt, len) => {
            exact_admission_add_pending_reduce(
                pending_reduces,
                *nt,
                *len,
                terminal_bit,
                terminals_len,
            );
            false
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if *accept && terminal == EOF {
                return true;
            }
            if shift.is_some() {
                return true;
            }
            for &(nt, len) in reduces {
                exact_admission_add_pending_reduce(
                    pending_reduces,
                    nt,
                    len,
                    terminal_bit,
                    terminals_len,
                );
            }
            false
        }
        Action::Accept => terminal == EOF,
    }
}

fn exact_admission_add_pending_reduce(
    pending_reduces: &mut SmallVec<[(u32, u32, BitSet); 8]>,
    nt: u32,
    rhs_len: u32,
    terminal_bit: usize,
    terminals_len: usize,
) {
    if let Some((_, _, terminals)) = pending_reduces
        .iter_mut()
        .find(|(pending_nt, pending_len, _)| *pending_nt == nt && *pending_len == rhs_len)
    {
        terminals.set(terminal_bit);
        return;
    }

    let mut terminals = BitSet::new(terminals_len);
    terminals.set(terminal_bit);
    pending_reduces.push((nt, rhs_len, terminals));
}

fn exact_admission_enqueue_reduce_any(
    table: &GLRTable,
    isolated: &ParserGSS,
    nt: u32,
    rhs_len: usize,
    terminals: &BitSet,
    queue: &mut VecDeque<(ParserGSS, BitSet)>,
    visited: &mut FxHashMap<ExactAdmissionSemanticKey, BitSet>,
    key_interner: &mut ExactAdmissionKeyInterner,
) {
    for (base, target, is_replace) in
        reduce_branches_from_isolated(table, isolated, nt, rhs_len)
    {
        let next = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        exact_admission_enqueue_frontier_any(
            next,
            terminals,
            queue,
            visited,
            key_interner,
        );
    }
}

fn exact_admission_enqueue_frontier_any(
    frontier: ParserGSS,
    terminals: &BitSet,
    queue: &mut VecDeque<(ParserGSS, BitSet)>,
    visited: &mut FxHashMap<ExactAdmissionSemanticKey, BitSet>,
    key_interner: &mut ExactAdmissionKeyInterner,
) {
    if frontier.is_empty() || terminals.is_empty() {
        return;
    }

    let key = key_interner.key(&frontier);
    let new_terminals = if let Some(seen_terminals) = visited.get_mut(&key) {
        let delta = terminals.difference(seen_terminals);
        if delta.is_empty() {
            return;
        }
        seen_terminals.union_with(&delta);
        delta
    } else {
        visited.insert(key, terminals.clone());
        terminals.clone()
    };

    queue.push_back((frontier, new_terminals));
}

#[inline]
fn exact_admission_terminal_bit(
    table: &GLRTable,
    terminal: TerminalID,
    terminals_len: usize,
) -> Option<usize> {
    let bit = if terminal == EOF {
        table.num_terminals as usize
    } else if terminal < table.num_terminals {
        terminal as usize
    } else {
        return None;
    };
    (bit < terminals_len).then_some(bit)
}

#[inline]
fn exact_admission_terminal_from_bit(table: &GLRTable, bit: usize) -> Option<TerminalID> {
    if bit == table.num_terminals as usize {
        Some(EOF)
    } else if bit < table.num_terminals as usize {
        Some(bit as TerminalID)
    } else {
        None
    }
}

fn exact_admission_enqueue_reduce(
    table: &GLRTable,
    isolated: &ParserGSS,
    nt: u32,
    rhs_len: usize,
    queue: &mut VecDeque<ParserGSS>,
    visited: &mut FxHashSet<ExactAdmissionSemanticKey>,
    key_interner: &mut ExactAdmissionKeyInterner,
) {
    for (base, target, is_replace) in
        reduce_branches_from_isolated(table, isolated, nt, rhs_len)
    {
        let next = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        exact_admission_enqueue_frontier(next, queue, visited, key_interner);
    }
}

fn exact_admission_enqueue_frontier(
    frontier: ParserGSS,
    queue: &mut VecDeque<ParserGSS>,
    visited: &mut FxHashSet<ExactAdmissionSemanticKey>,
    key_interner: &mut ExactAdmissionKeyInterner,
) {
    if frontier.is_empty() {
        return;
    }
    let key = key_interner.key(&frontier);
    if !visited.insert(key) {
        return;
    }
    queue.push_back(frontier);
}

fn stack_may_apply_guarded_shifts(stack: &ParserGSS, shifts: &[GuardedStackShift]) -> bool {
    if let Some(virtual_stack) = stack.try_virtual_stack()
        && !virtual_stack.has_hidden_floor_values()
    {
        return shifts
            .iter()
            .any(|shift| virtual_stack_may_apply_guarded_shift(&virtual_stack, shift));
    }

    !apply_guarded_stack_shifts(stack.clone(), shifts, None).is_empty()
}

#[cfg(test)]
mod tests {
    use super::{
        DisjointComponentActionProvider, ParserComponentTableSource, ParserGSS,
        ScopedParserSymbol, ScopedSubgrammarLink, close_provider_control_stacks,
        advance_provider_control_closed_stacks,
        materialize_control_eliminated_scoped_provider_table,
        ParserActionProvider,
        advance_concrete_stacks_reference,
        advance_stacks_with_provider,
        advance_stacks_disjoint_top_terminals_bounded,
        normalized_concrete_stacks,
        advance_stacks,
        apply_guarded_stack_shifts,
        apply_guarded_stack_shifts_as_predecessor_remap,
        apply_guarded_stack_shifts_fast,
        apply_guarded_stack_shifts_to_vstack,
        stack_admissible_terminals,
        stack_may_advance_on,
        stack_may_advance_on_any,
        stacks_finished,
        stacks_finished_with_provider,
        GLRTableActionProvider,
        stack_may_advance_disjoint_top_terminals_bounded,
        try_advance_bounded_concrete_paths,
        try_advance_mixed_top_replace_wave,
        try_advance_uniform_deterministic_frontier,
        try_advance_pop1_reduce_guarded_stackshift_wave,
        try_advance_pop1_reduce_plus_stackshift_wave,
        try_advance_pop1_stackshift_shift_wave,
        try_advance_single_active_pop1_stackshift_wave,
        GssSemanticKeyInterner,
        ProvidedAction,
        ProvidedActionRef,
        merge_into,
        reduce_sources_from_isolated,
    };
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::analysis::EOF;
    use crate::compiler::glr::table::testing::build_test_table;
    use crate::compiler::glr::table::{
        Action, AdmissionPolicy, GLRTable, GuardedStackShift, StackShift, StackShiftGuard,
    };
    use crate::ds::bitset::BitSet;
    use crate::ds::leveled_gss::Merge;
    use crate::grammar::flat::TerminalID;
    use rustc_hash::FxHashSet;
    use smallvec::SmallVec;

    #[test]
    fn provider_light_start_defaults_enabled_and_keeps_explicit_eager_fallback() {
        assert!(super::provider_light_start_policy(None));
        for value in ["1", "true"] {
            assert!(super::provider_light_start_policy(Some(value)));
        }
        for value in ["0", "false", "", "invalid"] {
            assert!(!super::provider_light_start_policy(Some(value)));
        }
    }

    #[test]
    fn provider_light_start_matches_eager_on_scopes_branches_guards_and_labels() {
        fn compare<P: ParserActionProvider>(provider: &P, stack: &ParserGSS, symbol: P::Symbol) {
            fn policy<P: ParserActionProvider, const R: bool, const S: bool>(
                provider: &P, stack: &ParserGSS, symbol: P::Symbol, mode: super::ProviderAdvanceMode,
            ) {
                let expected = super::advance_provider_traversal_with_initialization::<P,R,S,false>(provider,stack.clone(),symbol,mode);
                let actual = super::advance_provider_traversal_with_initialization::<P,R,S,true>(provider,stack.clone(),symbol,mode);
                let a=actual.shifted.to_stacks(4096).expect("bounded fixture");
                let b=expected.shifted.to_stacks(4096).expect("bounded fixture");
                assert_eq!(a.len(),b.len());
                assert!(a.iter().all(|x|b.contains(x)),"stack paths or accumulator labels changed");
                assert_eq!(actual.accepted,expected.accepted);
            }
            for mode in [super::ProviderAdvanceMode::Advance,super::ProviderAdvanceMode::Completion] {
                policy::<P,false,false>(provider,stack,symbol,mode);
                policy::<P,true,false>(provider,stack,symbol,mode);
                policy::<P,true,true>(provider,stack,symbol,mode);
            }
        }
        struct Components<'a>(&'a GLRTable);
        impl ParserComponentTableSource for Components<'_> {
            fn component_count(&self) -> usize { 2 }
            fn component_table(&self, component: u32) -> Option<&GLRTable> {
                (component < 2).then_some(self.0)
            }
        }
        for pop in [0, 1, 2, 5, 40] {
            for replace in [false, true] {
                let rows = [
                    vec![(0, Action::Shift(7, replace))],
                    vec![(0, Action::Reduce(0, pop))],
                    vec![(0, Action::ReplaceShifts(vec![4, 5].into()))],
                    vec![(0, Action::GuardedStackShifts(vec![GuardedStackShift {
                        pop: 1,
                        pushes: vec![7],
                        guards: vec![StackShiftGuard { pop: 0, states: vec![3].into() }],
                    }]))],
                    vec![(0, Action::StackShifts(vec![
                        StackShift { pop: 0, pushes: vec![7] },
                        StackShift { pop: 1, pushes: vec![5, 7] },
                        StackShift { pop: 20, pushes: vec![7] },
                    ]))],
                    vec![(0, Action::Split {
                        shift: Some((7, replace)), reduces: vec![(0, pop)], accept: true,
                    })],
                    vec![(0, Action::Skip)],
                    vec![(0, Action::Shift(7, false))],
                ];
                let gotos = (0..8).map(|_| vec![(0, (7, replace))]).collect::<Vec<_>>();
                let table = build_test_table(8, 2,
                    &rows.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
                let ordinary = GLRTableActionProvider::new(&table);
                let components = Components(&table);
                let scoped = DisjointComponentActionProvider::with_state_offsets(
                    &components, &[], &[0, 8],
                ).unwrap();
                for depth in [0, 1, 2, 6, 48] {
                    for top in 0..8 {
                        let mut values = vec![0; depth];
                        if let Some(last) = values.last_mut() { *last = top; }
                        let plain = TerminalsDisallowed::new();
                        let guarded = plain.with_insert(23, 11);
                        let first = ParserGSS::from_single_stack(values.clone(), plain.clone());
                        let second = ParserGSS::from_single_stack(values.iter().map(|x| (x + 1) % 8).collect(), guarded.clone());
                        compare(&ordinary, &first, 0);
                        compare(&ordinary, &first.merge(&second), 0);
                        compare(&ordinary, &first, 1);
                        let scoped_values = values.iter().map(|x| x + 8).collect();
                        let scoped_stack = ParserGSS::from_single_stack(scoped_values, guarded);
                        compare(&scoped, &scoped_stack, ScopedParserSymbol::Terminal { component: 1, terminal: 0 });
                        compare(&scoped, &scoped_stack, ScopedParserSymbol::Terminal { component: 0, terminal: 0 });
                    }
                }
            }
        }
    }

    #[test]
    fn provider_light_start_reuses_positive_probe_and_preserves_query_order() {
        use std::cell::RefCell;
        struct Counting { shift: Action, reduce: Action, queries: RefCell<Vec<(u32,u32)>> }
        impl ParserActionProvider for Counting {
            type Symbol=u32;
            fn action(&self,state:u32,symbol:u32)->Option<ProvidedAction<'_>> {
                self.queries.borrow_mut().push((state,symbol));
                let action=match (state,symbol) {
                    (0,0)|(1,1)=>&self.shift,
                    (0,1)=>&self.reduce,
                    _=>return None,
                };
                Some(ProvidedAction {action:ProvidedActionRef::Local {scope:0,action},
                    reduction_scope:0,extra_stack_shifts:SmallVec::new()})
            }
            fn scope_state(&self,scope:u32,state:u32)->Option<u32>{(scope==0&&state<4).then_some(state)}
            fn goto_target(&self,scope:u32,from:u32,nt:u32)->Option<(u32,bool)>{
                (scope==0&&from==0&&nt==0).then_some((1,false))
            }
            fn state_count_hint(&self)->usize{4}
        }
        let p=Counting {shift:Action::Shift(2,false),reduce:Action::Reduce(0,0),queries:RefCell::new(Vec::new())};
        let a=TerminalsDisallowed::new();let guarded=a.with_insert(17,9);
        let inputs=[
            ParserGSS::empty(),
            ParserGSS::from_single_stack(vec![],a.clone()),
            ParserGSS::from_single_stack(vec![0],a.clone()),
            ParserGSS::from_stacks(&[(vec![3,0],a.clone()),(vec![2,0],guarded.clone())]),
            ParserGSS::from_stacks(&[(vec![],a.clone()),(vec![0],guarded.clone())]),
            ParserGSS::from_stacks(&[(vec![0],a.clone()),(vec![1],guarded.clone())]),
        ];
        for stack in inputs {
            for mode in [super::ProviderAdvanceMode::Advance,super::ProviderAdvanceMode::Completion] {
                for symbol in 0..3 {
                    p.queries.borrow_mut().clear();
                    let expected=super::advance_provider_traversal_with_initialization::<_,false,false,false>(&p,stack.clone(),symbol,mode);
                    let sequence=p.queries.borrow().clone();p.queries.borrow_mut().clear();
                    let actual=super::advance_provider_traversal_with_initialization::<_,false,false,true>(&p,stack.clone(),symbol,mode);
                    assert_eq!(*p.queries.borrow(),sequence,"positive probe duplicated or action order changed");
                    let x=actual.shifted.to_stacks(64).unwrap();let y=expected.shifted.to_stacks(64).unwrap();
                    assert_eq!(x.len(),y.len());assert!(x.iter().all(|v|y.contains(v)));
                    assert_eq!(actual.accepted,expected.accepted);
                }
            }
        }
    }

    #[test]
    fn provider_reduction_prefix_preserves_scopes_branches_guards_and_accumulators() {
        fn compare<P: ParserActionProvider>(provider: &P, stack: &ParserGSS, symbol: P::Symbol) {
            for mode in [super::ProviderAdvanceMode::Advance, super::ProviderAdvanceMode::Completion] {
                let reference = super::advance_provider_traversal_impl::<P, false>(
                    provider, stack.clone(), symbol, mode,
                );
                let actual = super::advance_provider_traversal_impl::<P, true>(
                    provider, stack.clone(), symbol, mode,
                );
                let mut keys = GssSemanticKeyInterner::<u32, TerminalsDisallowed>::new();
                assert_eq!(keys.key(&actual.shifted), keys.key(&reference.shifted),
                    "virtual prefix changed stack language or accumulator correlation");
                assert_eq!(actual.accepted, reference.accepted);
                let resumed = super::advance_provider_traversal_with_policy::<P, true, true>(
                    provider, stack.clone(), symbol, mode,
                );
                assert_eq!(keys.key(&resumed.shifted), keys.key(&reference.shifted),
                    "resumption changed stack language or accumulator correlation");
                assert_eq!(resumed.accepted, reference.accepted);
            }
        }
        struct Components<'a>(&'a GLRTable);
        impl ParserComponentTableSource for Components<'_> {
            fn component_count(&self) -> usize { 2 }
            fn component_table(&self, component: u32) -> Option<&GLRTable> {
                (component < 2).then_some(self.0)
            }
        }
        for pop in [0, 1, 2, 5, 40] {
            for replace in [false, true] {
                let rows = [
                    vec![(0, Action::Shift(7, replace))],
                    vec![(0, Action::Reduce(0, pop))],
                    vec![(0, Action::ReplaceShifts(vec![4, 5].into()))],
                    vec![(0, Action::GuardedStackShifts(vec![GuardedStackShift {
                        pop: 1,
                        pushes: vec![7],
                        guards: vec![StackShiftGuard { pop: 0, states: vec![3].into() }],
                    }]))],
                    vec![(0, Action::StackShifts(vec![
                        StackShift { pop: 0, pushes: vec![7] },
                        StackShift { pop: 1, pushes: vec![5, 7] },
                        StackShift { pop: 20, pushes: vec![7] },
                    ]))],
                    vec![(0, Action::Split {
                        shift: Some((7, replace)), reduces: vec![(0, pop)], accept: true,
                    })],
                    vec![(0, Action::Skip)],
                    vec![(0, Action::Shift(7, false))],
                ];
                let gotos = (0..8).map(|_| vec![(0, (7, replace))]).collect::<Vec<_>>();
                let table = build_test_table(8, 2,
                    &rows.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
                let ordinary = GLRTableActionProvider::new(&table);
                let components = Components(&table);
                let scoped = DisjointComponentActionProvider::with_state_offsets(
                    &components, &[], &[0, 8],
                ).unwrap();
                for depth in [0, 1, 2, 6, 48] {
                    for top in 0..8 {
                        let mut values = vec![0; depth];
                        if let Some(last) = values.last_mut() { *last = top; }
                        let plain = TerminalsDisallowed::new();
                        let guarded = plain.with_insert(23, 11);
                        let first = ParserGSS::from_single_stack(values.clone(), plain.clone());
                        let second = ParserGSS::from_single_stack(values.iter().map(|x| (x + 1) % 8).collect(), guarded.clone());
                        compare(&ordinary, &first, 0);
                        compare(&ordinary, &first.merge(&second), 0);
                        compare(&ordinary, &first, 1);
                        let scoped_values = values.iter().map(|x| x + 8).collect();
                        let scoped_stack = ParserGSS::from_single_stack(scoped_values, guarded);
                        compare(&scoped, &scoped_stack, ScopedParserSymbol::Terminal { component: 1, terminal: 0 });
                        compare(&scoped, &scoped_stack, ScopedParserSymbol::Terminal { component: 0, terminal: 0 });
                    }
                }
            }
        }
    }

    #[test]
    fn provider_light_start_controls_and_extra_effects_match_eager() {
        struct Controls { ordinary: Action }
        impl ParserActionProvider for Controls {
            type Symbol = u32;
            fn action(&self, _: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
                let action = match symbol {
                    0 => ProvidedActionRef::Identity,
                    1 => ProvidedActionRef::Call { parent_target: 10, child_start: 20, replace: false },
                    2 => ProvidedActionRef::Call { parent_target: 10, child_start: 20, replace: true },
                    3 => ProvidedActionRef::Return { pop: 1 },
                    4 => ProvidedActionRef::Return { pop: 5 },
                    5 => ProvidedActionRef::Local { scope: 1, action: &self.ordinary },
                    6 => ProvidedActionRef::Local { scope: 99, action: &self.ordinary },
                    _ => return None,
                };
                Some(ProvidedAction {
                    action, reduction_scope: 1,
                    extra_stack_shifts: if symbol == 5 {
                        smallvec::smallvec![StackShift { pop: 0, pushes: vec![31] }]
                    } else { SmallVec::new() },
                })
            }
            fn scope_state(&self, scope: u32, state: u32) -> Option<u32> {
                (scope == 1).then_some(state + 100)
            }
            fn goto_target(&self, _: u32, _: u32, _: u32) -> Option<(u32, bool)> { None }
            fn state_count_hint(&self) -> usize { 128 }
        }
        let provider = Controls { ordinary: Action::Shift(5, false) };
        for n in 0..12 {
            let a = ParserGSS::from_single_stack((0..n).collect(), TerminalsDisallowed::new());
            let b = ParserGSS::from_single_stack((5..n+5).collect(), TerminalsDisallowed::new().with_insert(5,7));
            for stack in [a.clone(), a.merge(&b)] {
                for symbol in 0..8 {
                    let expected = super::advance_provider_traversal_with_initialization::<_, true, true, false>(
                        &provider, stack.clone(), symbol, super::ProviderAdvanceMode::Advance,
                    );
                    let actual = super::advance_provider_traversal_with_initialization::<_, true, true, true>(
                        &provider, stack.clone(), symbol, super::ProviderAdvanceMode::Advance,
                    );
                    let mut keys = GssSemanticKeyInterner::<u32, TerminalsDisallowed>::new();
                    assert_eq!(keys.key(&actual.shifted), keys.key(&expected.shifted), "n={n} symbol={symbol}");
                }
            }
        }
    }


    #[test]
    fn provider_light_start_completion_cycle_uses_the_same_semantic_visited_set() {
        let rows=[vec![],vec![(0,Action::Reduce(0,1))]];
        let gotos=[vec![(0,(1,false))],vec![]];
        let table=build_test_table(2,1,&rows.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let provider=GLRTableActionProvider::new(&table);
        let input=ParserGSS::from_single_stack(vec![0,1],TerminalsDisallowed::new().with_insert(3,5));
        let expected=super::advance_provider_traversal_with_initialization::<_,false,false,false>(
            &provider,input.clone(),0,super::ProviderAdvanceMode::Completion);
        let actual=super::advance_provider_traversal_with_initialization::<_,false,false,true>(
            &provider,input,0,super::ProviderAdvanceMode::Completion);
        assert!(!expected.accepted && !actual.accepted);
        assert!(expected.shifted.is_empty() && actual.shifted.is_empty());
    }

    #[test]
    fn provider_reduction_prefix_control_and_extra_effects_match_reference() {
        struct Controls { ordinary: Action }
        impl ParserActionProvider for Controls {
            type Symbol = u32;
            fn action(&self, _: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
                let action = match symbol {
                    0 => ProvidedActionRef::Identity,
                    1 => ProvidedActionRef::Call { parent_target: 10, child_start: 20, replace: false },
                    2 => ProvidedActionRef::Call { parent_target: 10, child_start: 20, replace: true },
                    3 => ProvidedActionRef::Return { pop: 1 },
                    4 => ProvidedActionRef::Return { pop: 5 },
                    5 => ProvidedActionRef::Local { scope: 1, action: &self.ordinary },
                    6 => ProvidedActionRef::Local { scope: 99, action: &self.ordinary },
                    _ => return None,
                };
                Some(ProvidedAction {
                    action, reduction_scope: 1,
                    extra_stack_shifts: if symbol == 5 {
                        smallvec::smallvec![StackShift { pop: 0, pushes: vec![31] }]
                    } else { SmallVec::new() },
                })
            }
            fn scope_state(&self, scope: u32, state: u32) -> Option<u32> {
                (scope == 1).then_some(state + 100)
            }
            fn goto_target(&self, _: u32, _: u32, _: u32) -> Option<(u32, bool)> { None }
            fn state_count_hint(&self) -> usize { 128 }
        }
        let provider = Controls { ordinary: Action::Shift(5, false) };
        for n in 0..12 {
            let a = ParserGSS::from_single_stack((0..n).collect(), TerminalsDisallowed::new());
            let b = ParserGSS::from_single_stack((5..n+5).collect(), TerminalsDisallowed::new().with_insert(5,7));
            for stack in [a.clone(), a.merge(&b)] {
                for symbol in 0..8 {
                    let expected = super::advance_provider_traversal_impl::<_, false>(
                        &provider, stack.clone(), symbol, super::ProviderAdvanceMode::Advance,
                    );
                    let actual = super::advance_provider_traversal_impl::<_, true>(
                        &provider, stack.clone(), symbol, super::ProviderAdvanceMode::Advance,
                    );
                    let mut keys = GssSemanticKeyInterner::<u32, TerminalsDisallowed>::new();
                    assert_eq!(keys.key(&actual.shifted), keys.key(&expected.shifted), "n={n} symbol={symbol}");
                }
            }
        }
    }


    #[test]
    fn provider_reduction_prefix_is_bounded_transactional_and_actually_runs() {
        use std::cell::Cell;
        struct Chain { reduce: Action, shift: Action, calls: Cell<usize> }
        impl ParserActionProvider for Chain {
            type Symbol = u32;
            fn action(&self, state: u32, _: u32) -> Option<ProvidedAction<'_>> {
                self.calls.set(self.calls.get()+1);
                Some(ProvidedAction {
                    action: ProvidedActionRef::Local { scope: 0,
                        action: if state < 5 { &self.reduce } else { &self.shift } },
                    reduction_scope: 0, extra_stack_shifts: SmallVec::new(),
                })
            }
            fn scope_state(&self, scope: u32, state: u32) -> Option<u32> {
                (scope == 0).then_some(state)
            }
            fn goto_target(&self, scope: u32, from: u32, _: u32) -> Option<(u32,bool)> {
                (scope == 0).then_some((from+1,false))
            }
            fn state_count_hint(&self) -> usize { 32 }
        }
        let provider=Chain {reduce:Action::Reduce(0,0),shift:Action::Shift(9,false),calls:Cell::new(0)};
        let original=ParserGSS::from_single_stack(vec![0],TerminalsDisallowed::new().with_insert(7,11));
        let preserved=original.clone();
        let first=provider.action(0,0).unwrap();
        let mut budget=2;
        assert!(super::try_provider_reduction_prefix(&provider,original.try_virtual_stack().unwrap(),0,&first,&mut budget).is_none());
        assert_eq!(budget,0);
        assert!(original.ptr_eq(&preserved),"declined prefix must not mutate its input");
        let calls=provider.calls.get();
        assert!(super::try_provider_reduction_prefix(&provider,original.try_virtual_stack().unwrap(),0,&first,&mut budget).is_none());
        assert_eq!(provider.calls.get(),calls,"an exhausted budget must not restart speculation");
        let mut enough=64;
        let actual=super::try_provider_reduction_prefix(&provider,original.try_virtual_stack().unwrap(),0,&first,&mut enough)
            .expect("the fixture must exercise a successful deterministic reduction prefix");
        assert!(enough<64);
        let reference=super::advance_provider_traversal_impl::<_,false>(
            &provider,original,0,super::ProviderAdvanceMode::Advance);
        let mut keys=GssSemanticKeyInterner::<u32,TerminalsDisallowed>::new();
        assert_eq!(keys.key(&actual),keys.key(&reference.shifted));
    }


    #[test]
    fn reduction_sources_exclude_epsilon_alternative_from_a_sole_predecessor() {
        for uniform_empty in [false, true] {
            let a = if uniform_empty { TerminalsDisallowed::new() }
                else { TerminalsDisallowed::new().with_insert(100, 7) };
            let b = if uniform_empty { a.clone() } else { a.with_insert(808, 909) };
            let input = ParserGSS::from_stacks(&[
                (vec![11, 22], a.clone()), (vec![22], b),
            ]);
            let popped = input.popn(1);
            assert_eq!(popped.single_top_value(), Some(11));
            assert!(!popped.isolate(None).is_empty(), "fixture must include epsilon");
            let sources = super::reduce_sources_from_isolated(&input, 1);
            assert_eq!(sources.len(), 1);
            assert_eq!(sources[0].0, 11);
            assert_eq!(sources[0].1.to_stacks(8).unwrap(), vec![(vec![11], a)]);
        }
    }

    #[test]
    fn provider_reduction_goto_never_uses_an_empty_predecessor_path() {
        struct Machine { reduce: Action, shift: Action, replace: bool }
        impl ParserActionProvider for Machine {
            type Symbol = u32;
            fn action(&self, state: u32, _: u32) -> Option<ProvidedAction<'_>> {
                let action = match state { 22 => &self.reduce, 30 => &self.shift, _ => return None };
                Some(ProvidedAction { action: ProvidedActionRef::Local { scope: 0, action },
                    reduction_scope: 0, extra_stack_shifts: SmallVec::new() })
            }
            fn scope_state(&self, scope: u32, state: u32) -> Option<u32> {
                (scope == 0).then_some(state)
            }
            fn goto_target(&self, scope: u32, from: u32, nt: u32) -> Option<(u32, bool)> {
                (scope == 0 && from == 11 && nt == 0).then_some((30, self.replace))
            }
            fn state_count_hint(&self) -> usize { 41 }
        }
        for uniform_empty in [false, true] { for replace in [false, true] {
            let a = if uniform_empty { TerminalsDisallowed::new() }
                else { TerminalsDisallowed::new().with_insert(100, 7) };
            let b = if uniform_empty { a.clone() } else { a.with_insert(808, 909) };
            let input = ParserGSS::from_stacks(&[(vec![11, 22], a.clone()), (vec![22], b)]);
            let machine = Machine { reduce: Action::Reduce(0, 1), shift: Action::Shift(40, false), replace };
            let expected = vec![(if replace { vec![30, 40] } else { vec![11, 30, 40] }, a)];
            for actual in [
                super::advance_provider_traversal_with_policy::<_, false, false>(&machine, input.clone(), 0, super::ProviderAdvanceMode::Advance),
                super::advance_provider_traversal_with_policy::<_, true, false>(&machine, input.clone(), 0, super::ProviderAdvanceMode::Advance),
                super::advance_provider_traversal_with_policy::<_, true, true>(&machine, input.clone(), 0, super::ProviderAdvanceMode::Advance),
            ] {
                assert_eq!(actual.shifted.to_stacks(8).unwrap(), expected,
                    "uniform_empty={uniform_empty} replace={replace}");
                assert!(!actual.accepted);
            }
        }}
    }

    #[test]
    fn provider_reduction_eligibility_requires_complete_uniform_empty_input() {
        let empty_label = TerminalsDisallowed::new();
        let simple = ParserGSS::from_single_stack(vec![1, 2], empty_label.clone());
        assert!(super::provider_reduction_input_is_uniform_empty(&simple));
        let hidden = ParserGSS::from_stacks(&[
            (vec![3], empty_label.clone()), (vec![4], empty_label),
        ]).push(2);
        assert!(hidden.try_virtual_stack().unwrap().has_hidden_floor_values());
        assert!(super::provider_reduction_input_is_uniform_empty(&hidden));
        let a = TerminalsDisallowed::new().with_insert(100, 7);
        let b = a.with_insert(808, 909);
        let nonempty = ParserGSS::from_single_stack(vec![1, 2], a.clone());
        assert!(!super::provider_reduction_input_is_uniform_empty(&nonempty));
        let mixed = ParserGSS::from_stacks(&[
            (vec![], a.clone()), (vec![1008, 1014], a), (vec![1009, 18], b),
        ]);
        assert!(!super::provider_reduction_input_is_uniform_empty(&mixed),
            "a single isolated branch must not certify the whole input");
    }

    #[test]
    fn bounded_provider_reductions_default_on_and_honor_reference_overrides() {
        assert!(super::provider_reduction_policy_value(None));
        for value in ["1", "true", "yes", "on", " TRUE "] {
            assert!(super::provider_reduction_policy_value(Some(value)), "{value}");
        }
        for value in ["0", "false", "no", "off", "", "unknown"] {
            assert!(!super::provider_reduction_policy_value(Some(value)), "{value}");
        }
    }

    #[test]
    fn provider_reduction_resumption_preserves_branch_floor_and_budget_boundaries() {
        use super::{ProviderReductionPrefix, ProviderAdvanceMode};
        struct Machine { first: Action, middle: Action, terminal: Action, floor: bool }
        impl ParserActionProvider for Machine {
            type Symbol = u32;
            fn action(&self, state:u32, _:u32)->Option<ProvidedAction<'_>> {
                let action=if state==0 { &self.first }
                    else if state==1 { &self.middle } else { &self.terminal };
                Some(ProvidedAction{ action:ProvidedActionRef::Local{scope:0,action},
                    reduction_scope:0, extra_stack_shifts:SmallVec::new() })
            }
            fn scope_state(&self, scope:u32, local:u32)->Option<u32> { (scope==0).then_some(local) }
            fn goto_target(&self,scope:u32,from:u32,nt:u32)->Option<(u32,bool)> {
                if scope!=0{return None;}
                if self.floor && nt==1 {Some((from+4,false))} else {Some((from+1,false))}
            }
            fn state_count_hint(&self)->usize{32}
        }
        let label=TerminalsDisallowed::new().with_insert(17,23);
        let ordinary=ParserGSS::from_single_stack(vec![0],label.clone());
        let floor=ParserGSS::from_stacks(&[(vec![3],label.clone()),(vec![4],label)]).push(0);
        assert!(floor.try_virtual_stack().unwrap().has_hidden_floor_values());
        let cases=[
            (Machine{first:Action::Reduce(0,0), middle:Action::ReplaceShifts(vec![7,8].into()), terminal:Action::Shift(9,false),floor:false},ordinary.clone(),64),
            (Machine{first:Action::Reduce(0,0), middle:Action::Reduce(1,2), terminal:Action::Shift(9,false),floor:true},floor,64),
            (Machine{first:Action::Reduce(0,0), middle:Action::Reduce(0,0), terminal:Action::Shift(9,false),floor:false},ordinary,1),
        ];
        for (index,(provider,input,mut budget)) in cases.into_iter().enumerate() {
            let saved=input.clone(); let first=provider.action(0,0).unwrap();
            let outcome=super::try_provider_reduction_prefix_impl::<_,true>(
                &provider,input.try_virtual_stack().unwrap(),0,&first,&mut budget)
                .expect("fixture must really suspend after a proved reduction");
            let ProviderReductionPrefix::Pending(pending)=outcome else {panic!("expected pending reduction frontier");};
            assert!(input.ptr_eq(&saved));
            let expected=super::advance_provider_traversal_impl::<_,false>(
                &provider,input.clone(),0,ProviderAdvanceMode::Advance);
            let remainder=super::advance_provider_traversal_impl::<_,false>(
                &provider,pending,0,ProviderAdvanceMode::Advance);
            let integrated=super::advance_provider_traversal_with_policy::<_,true,true>(
                &provider,input,0,ProviderAdvanceMode::Advance);
            let mut keys=GssSemanticKeyInterner::<u32,TerminalsDisallowed>::new();
            assert_eq!(keys.key(&remainder.shifted),keys.key(&expected.shifted),"case={index}");
            assert_eq!(keys.key(&integrated.shifted),keys.key(&expected.shifted),"integrated case={index}");
            assert_eq!(remainder.accepted,expected.accepted);
            assert_eq!(integrated.accepted,expected.accepted);
            if index==2 {assert_eq!(budget,0);}
        }
    }

    #[test]
    fn provider_may_advance_fastpath_matches_exact_reference() {
        use super::stack_may_advance_on_with_provider;
        // Exact old expression, kept as the reference (unchanged semantics).
        fn exact<P: ParserActionProvider>(
            provider: &P,
            stack: &ParserGSS,
            symbol: P::Symbol,
        ) -> bool {
            if stack.is_empty() {
                return false;
            }
            !advance_stacks_with_provider(
                provider,
                close_provider_control_stacks(provider, stack),
                symbol,
            )
            .is_empty()
        }

        fn check<P: ParserActionProvider>(
            provider: &P,
            stack: &ParserGSS,
            symbol: P::Symbol,
            expect: bool,
            case: &str,
        ) {
            assert_eq!(
                stack_may_advance_on_with_provider(provider, stack, symbol),
                expect,
                "optimized predicate disagrees with expected ({case})"
            );
            assert_eq!(
                exact(provider, stack, symbol),
                expect,
                "reference disagrees with expected (fixture wrong? {case})"
            );
            assert_eq!(
                super::stack_may_advance_on_any_with_provider(provider, stack, [symbol]),
                expect,
                "batched singleton differs ({case})"
            );
            assert_eq!(
                super::stack_may_advance_on_any_with_provider(provider, stack, [symbol, symbol]),
                expect,
                "batched duplicate differs ({case})"
            );
            assert!(!super::stack_may_advance_on_any_with_provider(
                provider, stack, std::iter::empty::<P::Symbol>(),
            ));
            let mut admitted = Vec::new();
            super::for_each_admitted_symbol_with_provider(
                provider, stack, [(0u32, symbol),(1u32, symbol)],
                |key| admitted.push(key),
            );
            assert_eq!(admitted, if expect { vec![0,1] } else { vec![] },
                "batched exact admission differs ({case})");
            let mut predicate_calls = 0;
            let found = super::find_admitted_symbol_with_provider(
                provider, stack, [(0u32, symbol), (1u32, symbol)],
                |&key| { predicate_calls += 1; key == 0 },
            );
            assert_eq!(found, expect.then_some(0), "lazy witness differs ({case})");
            assert_eq!(predicate_calls, usize::from(expect),
                "predicate must run only for admitted symbols and stop at a witness ({case})");
            let found = super::find_admitted_symbol_with_provider(
                provider, stack, [(0u32, symbol), (1u32, symbol)], |&key| key == 1,
            );
            assert_eq!(found, expect.then_some(1), "later lazy witness differs ({case})");
            assert!(super::find_admitted_symbol_with_provider(
                provider, stack, [(0u32, symbol)], |_| false,
            ).is_none(), "parser admission alone must not satisfy the predicate ({case})");
        }

        // Plain zero-pop shift: sufficient fast path, true.
        let plain = build_test_table(
            2,
            2,
            &[&[(0, Action::Shift(1, false))], &[]],
            &[&[], &[]],
        );
        let provider = GLRTableActionProvider::new(&plain);
        let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        check(&provider, &stack, 0, true, "plain-true");
        check(&provider, &stack, 1, false, "plain-false");

        // Split accept+shift: shift alternative suffices, true.
        let split = build_test_table(
            3,
            2,
            &[&[(
                0,
                Action::Split {
                    shift: Some((1, false)),
                    reduces: vec![(0, 1)],
                    accept: true,
                },
            )], &[], &[]],
            &[&[(0, (2, false))], &[], &[]],
        );
        let provider = GLRTableActionProvider::new(&split);
        check(&provider, &stack, 0, true, "split");

        // Accept only: no shift, false.
        let accept_only = build_test_table(2, 2, &[&[(0, Action::Accept)], &[]], &[&[], &[]]);
        let provider = GLRTableActionProvider::new(&accept_only);
        check(&provider, &stack, 0, false, "accept-only");

        // Empty input: false without touching the provider.
        let empty = ParserGSS::empty();
        check(&provider, &empty, 0, false, "empty");

        // Skip stays on the reference path (fast path requires nonempty
        // pushes); both must agree it advances.
        let skip = build_test_table(2, 2, &[&[(0, Action::Skip)], &[]], &[&[], &[]]);
        let provider = GLRTableActionProvider::new(&skip);
        check(&provider, &stack, 0, true, "skip");

        // Zero-pop StackShifts with mapped pushes: true.
        let shifts = build_test_table(
            3,
            2,
            &[&[(
                0,
                Action::StackShifts(vec![
                    StackShift { pop: 1, pushes: vec![2] },
                    StackShift { pop: 0, pushes: vec![1] },
                ]),
            )], &[], &[]],
            &[&[], &[], &[]],
        );
        let provider = GLRTableActionProvider::new(&shifts);
        check(&provider, &stack, 0, true, "zeropop-shift");

        // Nonzero-pop shift: the fallback decides. NOTE: this fixture's lone
        // single-stack CANNOT pop (pop(1) on [0] saturates at the root, so
        // the reference admits it) — the fast path must not claim pop>0.
        let pop1 = build_test_table(
            2,
            2,
            &[&[(0, Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![1] }]))], &[]],
            &[&[], &[]],
        );
        let provider = GLRTableActionProvider::new(&pop1);
        let lone = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        check(&provider, &lone, 0, true, "pop1-saturated");

        // Reduce-to-shift via feasible goto: fallback computes true.
        let red = build_test_table(
            3,
            2,
            &[&[(0, Action::Reduce(0, 1))], &[(0, Action::Shift(2, false))], &[]],
            &[&[(0, (1, false))], &[], &[]],
        );
        let provider = GLRTableActionProvider::new(&red);
        let deep = ParserGSS::from_single_stack(vec![0, 0], TerminalsDisallowed::new());
        check(&provider, &deep, 0, true, "reduce-shift");

        // Infeasible reduction (no goto): false.
        let nogoto = build_test_table(
            2,
            2,
            &[&[(0, Action::Reduce(0, 1))], &[]],
            &[&[], &[]],
        );
        let provider = GLRTableActionProvider::new(&nogoto);
        check(&provider, &deep, 0, false, "infeasible-reduce");

        // Guarded shift, guard matches: fallback true; guard fails: false.
        let guarded = build_test_table(
            3,
            2,
            &[&[(
                0,
                Action::GuardedStackShifts(vec![GuardedStackShift {
                    guards: vec![StackShiftGuard { pop: 1, states: vec![0] }],
                    pop: 1,
                    pushes: vec![2],
                }]),
            )], &[], &[]],
            &[&[], &[], &[]],
        );
        let provider = GLRTableActionProvider::new(&guarded);
        check(&provider, &deep, 0, true, "guarded-pass");
        let bad = ParserGSS::from_single_stack(vec![1, 0], TerminalsDisallowed::new());
        check(&provider, &bad, 0, false, "guarded-fail");

        // Identity via disjoint-provider ignore terminal: fast path true.
        let ws_parent = build_test_table(2, 2, &[&[(0, Action::Shift(1, false))], &[]], &[&[], &[]]);
        struct WsSource<'a> {
            table: &'a GLRTable,
        }
        impl ParserComponentTableSource for WsSource<'_> {
            fn component_count(&self) -> usize {
                1
            }
            fn component_table(&self, component: u32) -> Option<&GLRTable> {
                (component == 0).then_some(self.table)
            }
            fn component_ignore_terminal(&self, component: u32) -> Option<TerminalID> {
                (component == 0).then_some(1)
            }
        }
        let ws_source = WsSource { table: &ws_parent };
        let ws_links: [ScopedSubgrammarLink; 0] = [];
        let ws_provider =
            DisjointComponentActionProvider::new(&ws_source, &ws_links).unwrap();
        let ws_top = ws_provider.scoped_state(0, 0).unwrap();
        let ws_stack =
            ParserGSS::from_single_stack(vec![ws_top], TerminalsDisallowed::new());
        check(
            &ws_provider,
            &ws_stack,
            ScopedParserSymbol::Terminal { component: 0, terminal: 1 },
            true,
            "identity-ignore",
        );

        // Unmapped PUSH target on a valid top: provider denies the mapping,
        // so the alternative is infeasible — both must agree false.
        struct DenyTargetProvider<'a> {
            inner: GLRTableActionProvider<'a>,
        }
        impl ParserActionProvider for DenyTargetProvider<'_> {
            type Symbol = TerminalID;
            fn action(&self, state: u32, symbol: TerminalID) -> Option<ProvidedAction<'_>> {
                self.inner.action(state, symbol)
            }
            fn scope_state(&self, _scope: u32, _local: u32) -> Option<u32> {
                None
            }
            fn goto_target(
                &self,
                scope: u32,
                goto_from: u32,
                nt: u32,
            ) -> Option<(u32, bool)> {
                self.inner.goto_target(scope, goto_from, nt)
            }
            fn state_count_hint(&self) -> usize {
                self.inner.state_count_hint()
            }
        }
        let deny_table = build_test_table(
            2,
            2,
            &[&[(0, Action::Shift(1, false))], &[]],
            &[&[], &[]],
        );
        let deny_inner = GLRTableActionProvider::new(&deny_table);
        let deny_provider = DenyTargetProvider { inner: deny_inner };
        check(&deny_provider, &stack, 0, false, "unmapped-push");

        // Unknown-scope top (provider.action None): false, no wrong short-circuit.
        let parent = build_test_table(2, 2, &[&[(0, Action::Shift(1, false))], &[]], &[&[], &[]]);
        let child = build_test_table(2, 2, &[&[(0, Action::Shift(1, false))], &[]], &[&[], &[]]);
        let components = [&parent, &child];
        let links: [ScopedSubgrammarLink; 0] = [];
        let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
        let foreign = ParserGSS::from_single_stack(vec![u32::MAX - 1], TerminalsDisallowed::new());
        check(
            &provider,
            &foreign,
            ScopedParserSymbol::Terminal { component: 0, terminal: 0 },
            false,
            "unknown-scope",
        );
        // Same disjoint provider, valid mapped shift: true.
        let p0 = provider.scoped_state(0, 0).unwrap();
        let ok = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
        check(
            &provider,
            &ok,
            ScopedParserSymbol::Terminal { component: 0, terminal: 0 },
            true,
            "disjoint-mapped-shift",
        );
    }

    #[test]
    fn disjoint_component_provider_parses_across_link_without_composed_table() {
        let slot = 0;
        let parent_tail = 1;
        let child_token = 0;
        let parent = build_test_table(
            3,
            2,
            &[
                &[(slot, Action::Shift(1, false))],
                &[(parent_tail, Action::Shift(2, false))],
                &[],
            ],
            &[&[], &[], &[]],
        );
        let child = build_test_table(
            2,
            1,
            &[
                &[(child_token, Action::Shift(1, false))],
                &[(EOF, Action::Accept)],
            ],
            &[&[], &[]],
        );
        let components = [&parent, &child];
        let links = [ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: slot,
            child_component: 1,
            child_start: 0,
            return_pop: 2,
            child_start_nullable: false,
        }];
        let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
        let p0 = provider.scoped_state(0, 0).unwrap();
        let p1 = provider.scoped_state(0, 1).unwrap();
        let p2 = provider.scoped_state(0, 2).unwrap();
        let c0 = provider.scoped_state(1, 0).unwrap();
        let c1 = provider.scoped_state(1, 1).unwrap();
        let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());

        let called = advance_stacks_with_provider(
            &provider,
            start,
            ScopedParserSymbol::Entry { link: 0 },
        );
        assert_eq!(called.single_top_value(), Some(c0));
        assert!(called
            .to_stacks(8)
            .unwrap()
            .iter()
            .any(|(stack, _)| stack == &vec![p0, p1, c0]));

        let child_advanced = advance_stacks_with_provider(
            &provider,
            called,
            ScopedParserSymbol::Terminal {
                component: 1,
                terminal: child_token,
            },
        );
        assert_eq!(child_advanced.single_top_value(), Some(c1));

        let returned = advance_stacks_with_provider(
            &provider,
            child_advanced,
            ScopedParserSymbol::Finish { component: 1 },
        );
        assert_eq!(returned.single_top_value(), Some(p1));

        let finished = advance_stacks_with_provider(
            &provider,
            returned,
            ScopedParserSymbol::Terminal {
                component: 0,
                terminal: parent_tail,
            },
        );
        assert_eq!(finished.single_top_value(), Some(p2));
    }

    #[test]
    fn materialized_scoped_provider_table_preserves_call_return_stack_language() {
        let slot = 0;
        let parent_tail = 1;
        let child_token = 0;
        let parent = build_test_table(
            3,
            2,
            &[
                &[(slot, Action::Shift(1, false))],
                &[(parent_tail, Action::Shift(2, false))],
                &[],
            ],
            &[&[], &[], &[]],
        );
        let child = build_test_table(
            2,
            1,
            &[
                &[(child_token, Action::Shift(1, false))],
                &[(EOF, Action::Accept)],
            ],
            &[&[], &[]],
        );
        let components = [&parent, &child];
        let links = [ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: slot,
            child_component: 1,
            child_start: 0,
            return_pop: 2,
            child_start_nullable: false,
        }];
        let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
        let p0 = provider.scoped_state(0, 0).unwrap();
        let p2 = provider.scoped_state(0, 2).unwrap();
        let terminal_symbols = vec![
            SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
                component: 1,
                terminal: child_token,
            }]),
            SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
                component: 0,
                terminal: parent_tail,
            }]),
        ];
        let table = materialize_control_eliminated_scoped_provider_table(
            &provider,
            &terminal_symbols,
        )
        .unwrap();
        assert!(table.control_terminals.is_empty());
        assert_eq!(table.num_states, provider.state_count_hint() as u32);
        assert_eq!(table.num_terminals, terminal_symbols.len() as u32);

        let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
        let provider_after_child = advance_provider_control_closed_stacks(
            &provider,
            &close_provider_control_stacks(&provider, &start),
            terminal_symbols[0][0],
        );
        let table_after_child = advance_stacks(&table, &start, 0);
        assert!(!provider_after_child.is_empty());
        assert!(!table_after_child.is_empty());

        let provider_finished = advance_provider_control_closed_stacks(
            &provider,
            &provider_after_child,
            terminal_symbols[1][0],
        );
        let table_finished = advance_stacks(&table, &table_after_child, 1);
        assert!(!provider_finished.is_empty());
        assert!(!table_finished.is_empty());
        assert_eq!(table_finished.single_top_value(), Some(p2));
    }

    #[test]
    fn materialized_scoped_provider_table_preserves_midword_call_and_return() {
        let x = 0;
        let slot = 1;
        let z = 2;
        let y = 0;
        let child_nt = 0;
        let parent = build_test_table(
            4,
            3,
            &[
                &[(x, Action::Shift(1, false))],
                &[(slot, Action::Shift(2, false))],
                &[(z, Action::Shift(3, false))],
                &[],
            ],
            &[&[], &[], &[], &[]],
        );
        let child = build_test_table(
            3,
            1,
            &[
                &[(y, Action::Shift(1, false))],
                &[(EOF, Action::Reduce(child_nt, 1))],
                &[(EOF, Action::Accept)],
            ],
            &[&[(child_nt, (2, false))], &[], &[]],
        );
        let components = [&parent, &child];
        let links = [ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: slot,
            child_component: 1,
            child_start: 0,
            return_pop: 2,
            child_start_nullable: false,
        }];
        let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
        let p0 = provider.scoped_state(0, 0).unwrap();
        let p3 = provider.scoped_state(0, 3).unwrap();
        let terminal_symbols = vec![
            SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
                component: 0,
                terminal: x,
            }]),
            SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
                component: 1,
                terminal: y,
            }]),
            SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
                component: 0,
                terminal: z,
            }]),
        ];
        let table = materialize_control_eliminated_scoped_provider_table(
            &provider,
            &terminal_symbols,
        )
        .unwrap();

        let mut provider_stack =
            ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
        let mut table_stack = provider_stack.clone();
        for (terminal, symbol) in terminal_symbols.iter().enumerate() {
            provider_stack = advance_provider_control_closed_stacks(
                &provider,
                &provider_stack,
                symbol[0],
            );
            table_stack = advance_stacks(&table, &table_stack, terminal as u32);
            assert!(!provider_stack.is_empty(), "provider rejected terminal {terminal}");
            assert!(!table_stack.is_empty(), "table rejected terminal {terminal}");
        }
        assert_eq!(
            normalized_concrete_stacks(&provider_stack),
            normalized_concrete_stacks(&table_stack),
            "provider/table visible-word result diverged",
        );
        assert_eq!(table_stack.single_top_value(), Some(p3));
    }

    #[test]
    fn materialized_scoped_provider_table_preserves_nullable_finish_branch() {
        let slot = 0;
        let tail = 1;
        let parent = build_test_table(
            3,
            2,
            &[
                &[(slot, Action::Shift(1, false))],
                &[(tail, Action::Shift(2, false))],
                &[],
            ],
            &[&[], &[], &[]],
        );
        let child = build_test_table(1, 0, &[&[]], &[&[]]);
        let components = [&parent, &child];
        let links = [ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: slot,
            child_component: 1,
            child_start: 0,
            return_pop: 2,
            child_start_nullable: true,
        }];
        let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
        let p0 = provider.scoped_state(0, 0).unwrap();
        let p2 = provider.scoped_state(0, 2).unwrap();
        let terminal_symbols = vec![SmallVec::from_slice(&[ScopedParserSymbol::Terminal {
            component: 0,
            terminal: tail,
        }])];
        let table = materialize_control_eliminated_scoped_provider_table(
            &provider,
            &terminal_symbols,
        )
        .unwrap();
        let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
        let provider_finished = advance_provider_control_closed_stacks(
            &provider,
            &close_provider_control_stacks(&provider, &start),
            terminal_symbols[0][0],
        );
        let table_finished = advance_stacks(&table, &start, 0);
        assert_eq!(
            normalized_concrete_stacks(&provider_finished),
            normalized_concrete_stacks(&table_finished),
        );
        assert_eq!(table_finished.single_top_value(), Some(p2));
    }

    #[test]
    fn disjoint_local_guarded_shift_respects_component_scope() {
        let token = 0;
        let parent = build_test_table(1, 0, &[&[]], &[&[]]);
        let child = build_test_table(
            3,
            1,
            &[
                &[],
                &[(token, Action::GuardedStackShifts(vec![GuardedStackShift {
                    guards: vec![StackShiftGuard { pop: 1, states: vec![0] }],
                    pop: 1,
                    pushes: vec![2],
                }]))],
                &[],
            ],
            &[&[], &[], &[]],
        );
        let components = [&parent, &child];
        let provider = DisjointComponentActionProvider::new(&components, &[]).unwrap();
        let p0 = provider.scoped_state(0, 0).unwrap();
        let c0 = provider.scoped_state(1, 0).unwrap();
        let c1 = provider.scoped_state(1, 1).unwrap();
        let c2 = provider.scoped_state(1, 2).unwrap();

        let matching = ParserGSS::from_single_stack(vec![c0, c1], TerminalsDisallowed::new());
        let shifted = advance_stacks_with_provider(
            &provider,
            matching,
            ScopedParserSymbol::Terminal {
                component: 1,
                terminal: token,
            },
        );
        assert_eq!(shifted.single_top_value(), Some(c2));

        let cross_scope =
            ParserGSS::from_single_stack(vec![p0, c1], TerminalsDisallowed::new());
        let rejected = advance_stacks_with_provider(
            &provider,
            cross_scope,
            ScopedParserSymbol::Terminal {
                component: 1,
                terminal: token,
            },
        );
        assert!(rejected.is_empty());
    }

    #[test]
    fn disjoint_provider_preserves_reductions_under_entry_and_finish() {
        let slot = 0;
        let child_token = 0;
        let nt = 0;
        let parent = build_test_table(
            4,
            1,
            &[
                &[],
                &[(slot, Action::Reduce(nt, 1))],
                &[(slot, Action::Shift(3, false))],
                &[],
            ],
            &[&[(nt, (2, false))], &[], &[], &[]],
        );
        let child = build_test_table(
            3,
            1,
            &[
                &[(child_token, Action::Shift(1, false))],
                &[(EOF, Action::Reduce(nt, 1))],
                &[(EOF, Action::Accept)],
            ],
            &[&[(nt, (2, false))], &[], &[]],
        );
        let components = [&parent, &child];
        let links = [ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: slot,
            child_component: 1,
            child_start: 0,
            return_pop: 2,
            child_start_nullable: false,
        }];
        let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
        let p0 = provider.scoped_state(0, 0).unwrap();
        let p1 = provider.scoped_state(0, 1).unwrap();
        let p2 = provider.scoped_state(0, 2).unwrap();
        let p3 = provider.scoped_state(0, 3).unwrap();
        let c0 = provider.scoped_state(1, 0).unwrap();
        let start = ParserGSS::from_single_stack(vec![p0, p1], TerminalsDisallowed::new());

        let called = advance_stacks_with_provider(
            &provider,
            start,
            ScopedParserSymbol::Entry { link: 0 },
        );
        assert_eq!(called.single_top_value(), Some(c0));
        assert!(called
            .to_stacks(8)
            .unwrap()
            .iter()
            .any(|(stack, _)| stack == &vec![p0, p2, p3, c0]));

        let child_advanced = advance_stacks_with_provider(
            &provider,
            called,
            ScopedParserSymbol::Terminal {
                component: 1,
                terminal: child_token,
            },
        );
        let returned = advance_stacks_with_provider(
            &provider,
            child_advanced,
            ScopedParserSymbol::Finish { component: 1 },
        );
        assert_eq!(returned.single_top_value(), Some(p3));
    }

    #[test]
    fn disjoint_provider_nullable_child_can_return_immediately() {
        let slot = 0;
        let parent = build_test_table(
            2,
            1,
            &[&[(slot, Action::Shift(1, false))], &[]],
            &[&[], &[]],
        );
        let child = build_test_table(1, 0, &[&[]], &[&[]]);
        let components = [&parent, &child];
        let links = [ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: slot,
            child_component: 1,
            child_start: 0,
            return_pop: 2,
            child_start_nullable: true,
        }];
        let provider = DisjointComponentActionProvider::new(&components, &links).unwrap();
        let p0 = provider.scoped_state(0, 0).unwrap();
        let p1 = provider.scoped_state(0, 1).unwrap();
        let c0 = provider.scoped_state(1, 0).unwrap();
        let start = ParserGSS::from_single_stack(vec![p0], TerminalsDisallowed::new());
        let closed = close_provider_control_stacks(&provider, &start);
        let stacks = closed.to_stacks(16).unwrap();
        assert!(stacks.iter().any(|(stack, _)| stack == &vec![p0, p1]));
        assert!(stacks
            .iter()
            .any(|(stack, _)| stack == &vec![p0, p1, c0]));
    }

    #[test]
    fn uniform_deterministic_frontier_preserves_divergent_lower_prefixes() {
        let token = 0;
        let nt0 = 0;
        let nt1 = 1;
        let mut action_rows = vec![Vec::<(u32, Action)>::new(); 10];
        action_rows[9].push((token, Action::Reduce(nt0, 1)));
        action_rows[6].push((token, Action::Reduce(nt1, 1)));
        action_rows[7].push((token, Action::Shift(8, false)));
        let action_refs = action_rows
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();

        let mut goto_rows = vec![Vec::<(u32, (u32, bool))>::new(); 10];
        goto_rows[4].push((nt0, (6, true)));
        goto_rows[1].push((nt1, (7, true)));
        let goto_refs = goto_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let table = build_test_table(10, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 2, 1, 4, 9], acc.clone()),
            (vec![0, 3, 1, 4, 9], acc.clone()),
            (vec![0, 5, 1, 4, 9], acc),
        ]);
        let fast = try_advance_uniform_deterministic_frontier(&table, &before, token)
            .expect("uniform reduction chain should be structural");
        let expected = advance_concrete_stacks_reference(&table, &before, token);

        assert_eq!(
            normalized_concrete_stacks(&fast),
            normalized_concrete_stacks(&expected),
        );
        assert_eq!(
            normalized_concrete_stacks(&fast),
            vec![
                (vec![0, 2, 7, 8], TerminalsDisallowed::new()),
                (vec![0, 3, 7, 8], TerminalsDisallowed::new()),
                (vec![0, 5, 7, 8], TerminalsDisallowed::new()),
            ],
        );
    }

    #[test]
    fn concrete_advance_reference_applies_whole_stack_effect_atomically() {
        let token = 0;
        let table = build_test_table(
            2,
            1,
            &[
                &[],
                &[(
                    token,
                    Action::StackShifts(vec![StackShift {
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[]],
        );
        let before = ParserGSS::from_single_stack(
            vec![0, 1],
            TerminalsDisallowed::new(),
        );
        let expected = ParserGSS::from_single_stack(
            vec![7],
            TerminalsDisallowed::new(),
        );

        assert_eq!(
            advance_concrete_stacks_reference(&table, &before, token),
            expected,
        );
    }

    #[test]
    fn concrete_advance_reference_merges_accumulators_at_reduce_closure_join() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            6,
            1,
            &[
                &[],
                &[],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Shift(5, false))],
                &[],
            ],
            &[&[(nt, (4, false))], &[], &[], &[], &[], &[]],
        );
        let left_acc = TerminalsDisallowed::new().with_insert(10, 20);
        let right_acc = TerminalsDisallowed::new().with_insert(11, 21);
        let before = ParserGSS::from_stacks(&[
            (vec![0, 2], left_acc.clone()),
            (vec![0, 3], right_acc.clone()),
        ]);
        let expected = ParserGSS::from_single_stack(
            vec![0, 4, 5],
            left_acc.merge(&right_acc),
        );

        assert_eq!(
            advance_concrete_stacks_reference(&table, &before, token),
            expected,
        );
    }

    #[test]
    fn advance_stacks_matches_reduce_fanout_collapse_fast_path() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            5,
            1,
            &[
                &[],
                &[],
                &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
                &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
                &[(token, Action::Reduce(nt, 1))],
            ],
            &[
                &[(nt, (2, false))],
                &[(nt, (3, false))],
                &[],
                &[],
                &[],
            ],
        );

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 4], acc.clone()),
            (vec![1, 4], acc),
        ]);
        let expected = ParserGSS::from_single_stack(vec![7], TerminalsDisallowed::new());

        assert_eq!(advance_stacks(&table, &before, token), expected);
    }

    #[test]
    fn advance_stacks_selective_pure_frontier_shift_keeps_only_actionable_top() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 134];
        action_rows[131] = vec![(
            token,
            Action::StackShifts(vec![StackShift {
                pop: 0,
                pushes: vec![96],
            }]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();
        let goto_rows = vec![Vec::new(); 134];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();
        let table = build_test_table(134, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 133], acc),
        ]);
        let expected = ParserGSS::from_single_stack(
            vec![0_u32, 1, 17, 47, 74, 131, 96],
            TerminalsDisallowed::new(),
        );

        assert_eq!(advance_stacks(&table, &before, token), expected);
    }

    #[test]
    fn single_active_pop1_stackshift_wave_discards_dead_top_without_cross_product() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 500];
        action_rows[452] = vec![(
            token,
            Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![384, 411],
            }]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();
        let goto_rows = vec![Vec::new(); 500];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();
        let table = build_test_table(500, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1, 186, 40, 85, 123, 452], acc.clone()),
            (vec![0, 1, 202, 40, 85, 123, 452], acc.clone()),
            (vec![0, 1, 322, 92, 150, 426], acc),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (
                vec![0, 1, 186, 40, 85, 123, 384, 411],
                TerminalsDisallowed::new(),
            ),
            (
                vec![0, 1, 202, 40, 85, 123, 384, 411],
                TerminalsDisallowed::new(),
            ),
        ]);

        let fast = try_advance_single_active_pop1_stackshift_wave(&table, &before, token)
            .expect("single live stack-shift branch should be structural");
        let mut fast_stacks = fast
            .to_stacks(16)
            .expect("bounded structural result should enumerate");
        let mut expected_stacks = expected
            .to_stacks(16)
            .expect("bounded expected result should enumerate");
        fast_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(fast_stacks, expected_stacks);

        let mut actual_stacks = advance_stacks(&table, &before, token)
            .to_stacks(16)
            .expect("bounded advanced result should enumerate");
        actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(actual_stacks, expected_stacks);
    }

    #[test]
    fn pop1_reduce_and_guarded_shift_wave_remaps_predecessors_structurally() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 400];
        action_rows[133] = vec![(
            token,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![322],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![10],
                    },
                ],
                pop: 2,
                pushes: vec![327],
            }]),
        )];
        action_rows[137] = vec![(token, Action::Reduce(nt, 1))];
        action_rows[266] = vec![(token, Action::Shift(351, true))];
        action_rows[284] = vec![(token, Action::Shift(360, true))];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();

        let mut goto_rows = vec![Vec::new(); 400];
        goto_rows[168] = vec![(nt, (266, true))];
        goto_rows[181] = vec![(nt, (284, true))];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();
        let table = build_test_table(400, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 10, 168, 137], acc.clone()),
            (vec![0, 10, 181, 137], acc.clone()),
            (vec![0, 10, 322, 133], acc),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 10, 351], TerminalsDisallowed::new()),
            (vec![0, 10, 360], TerminalsDisallowed::new()),
            (vec![0, 10, 327], TerminalsDisallowed::new()),
        ]);

        let bounded = try_advance_bounded_concrete_paths(&table, &before, token)
            .expect("small mixed paths should be interpreted concretely");
        assert!(bounded.semantically_eq(&expected, 16).unwrap());

        let fast = try_advance_pop1_reduce_guarded_stackshift_wave(&table, &before, token)
            .expect("mixed wave should be handled structurally");
        assert!(fast.semantically_eq(&expected, 16).unwrap());
        assert!(advance_stacks(&table, &before, token)
            .semantically_eq(&expected, 16)
            .unwrap());
    }

    #[test]
    fn pop1_reduce_plus_stackshift_wave_fast_path_matches_snowplow_shape() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 989];
        action_rows[655] = vec![(
            token,
            Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
        )];
        action_rows[659] = vec![(
            token,
            Action::StackShifts(vec![
                StackShift { pop: 1, pushes: vec![654] },
                StackShift { pop: 1, pushes: vec![988] },
            ]),
        )];
        action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();

        let mut goto_rows = vec![Vec::new(); 989];
        goto_rows[87] = vec![(nt, (659, true))];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();

        let table = build_test_table(989, 1, &action_refs, &goto_refs);
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 987], acc.clone()),
            (vec![0_u32, 87, 655], acc),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 975], TerminalsDisallowed::new()),
            (vec![0_u32, 654], TerminalsDisallowed::new()),
            (vec![0_u32, 988], TerminalsDisallowed::new()),
        ]);

        let mut fast_stacks = try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token)
            .expect("fast path should match this wave")
            .to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        fast_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(fast_stacks, expected_stacks);

        let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(actual_stacks, expected_stacks);
    }

    #[test]
    fn pop1_reduce_plus_stackshift_wave_rejects_cross_product_base() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 989];
        action_rows[655] = vec![(
            token,
            Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
        )];
        action_rows[659] = vec![(
            token,
            Action::StackShifts(vec![
                StackShift { pop: 1, pushes: vec![654] },
                StackShift { pop: 1, pushes: vec![988] },
            ]),
        )];
        action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();

        let mut goto_rows = vec![Vec::new(); 989];
        goto_rows[87] = vec![(nt, (659, true))];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();

        let table = build_test_table(989, 1, &action_refs, &goto_refs);
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 987], acc.clone()),
            (vec![1_u32, 87, 655], acc),
        ]);

        assert_eq!(
            try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token),
            None
        );
    }

    #[test]
    fn may_advance_consults_admission_rows_not_execution_actions() {
        let token = 0;
        let mut table = build_test_table(
            2,
            1,
            &[&[], &[(token, Action::Shift(1, false))]],
            &[&[], &[]],
        );
        table.advance[1].clear(token as usize);

        let stack = ParserGSS::from_single_stack(vec![1], TerminalsDisallowed::new());
        assert!(table.action(1, token).is_some());
        assert!(!stack_may_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn may_advance_rechecks_guarded_stack_shifts_against_concrete_stack() {
        let token = 0;
        let mut table = build_test_table(
            3,
            1,
            &[
                &[],
                &[],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![0],
                        }],
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());

        assert!(table.advance_row_allows(2, token));
        assert!(advance_stacks(&table, &stack, token).is_empty());
        assert!(!stack_may_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn row_presence_admission_does_not_recheck_lowered_guarded_effects() {
        let token = 0;
        let table = build_test_table(
            3,
            1,
            &[
                &[],
                &[],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![0],
                        }],
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[], &[]],
        );
        let stack = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());

        assert_eq!(table.admission_policy, AdmissionPolicy::RowPresenceExact);
        assert!(table.advance_row_allows(2, token));
        assert!(stack_may_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn guarded_predecessor_remap_preserves_branched_floor_correlation() {
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 353, 669, 800], acc.clone()),
            (vec![0_u32, 353, 711, 800], acc.clone()),
            (vec![0_u32, 353, 753, 800], acc),
        ]);
        let shifts = vec![
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![669],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![353, 1503],
                    },
                ],
                pop: 2,
                pushes: vec![1115],
            },
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![711],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![353, 1504],
                    },
                ],
                pop: 2,
                pushes: vec![1168],
            },
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![753],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![353, 1505],
                    },
                ],
                pop: 2,
                pushes: vec![1221],
            },
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![1348],
                }],
                pop: 2,
                pushes: vec![1457, 1497],
            },
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![1547],
                    },
                    StackShiftGuard {
                        pop: 3,
                        states: vec![1457],
                    },
                ],
                pop: 3,
                pushes: vec![1498, 1497],
            },
        ];

        let fast = apply_guarded_stack_shifts_as_predecessor_remap(&before, &shifts)
            .expect("predecessor remap shape should be recognized");
        let general = apply_guarded_stack_shifts(before, &shifts, None);
        assert!(fast.semantically_eq(&general, 16).unwrap());

        let mut stacks = fast
            .to_stacks(16)
            .expect("three-path result should remain bounded");
        stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(
            stacks
                .into_iter()
                .map(|(stack, _)| stack)
                .collect::<Vec<_>>(),
            vec![
                vec![0, 353, 1115],
                vec![0, 353, 1168],
                vec![0, 353, 1221],
            ],
        );
    }

    #[test]
    fn bounded_guarded_effects_handle_machine_schema_branched_floor() {
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (
                vec![0, 1, 10, 40, 87, 124, 85, 123, 186, 40, 85, 123, 322, 133],
                acc.clone(),
            ),
            (
                vec![0, 1, 10, 40, 87, 124, 85, 123, 202, 40, 85, 123, 322, 133],
                acc.clone(),
            ),
            (
                vec![
                    0, 1, 10, 40, 87, 124, 85, 123, 322, 92, 150, 92, 250, 343, 426, 133,
                ],
                acc,
            ),
        ]);
        let shifts = vec![
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![50],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![1],
                    },
                ],
                pop: 2,
                pushes: vec![74],
            },
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![322],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![123],
                    },
                ],
                pop: 2,
                pushes: vec![327],
            },
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![426],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![250],
                    },
                ],
                pop: 2,
                pushes: vec![343, 342],
            },
            GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![426],
                    },
                    StackShiftGuard {
                        pop: 2,
                        states: vec![343],
                    },
                    StackShiftGuard {
                        pop: 3,
                        states: vec![250],
                    },
                ],
                pop: 3,
                pushes: vec![343, 342],
            },
        ];
        let expected = ParserGSS::from_stacks(&[
            (
                vec![0, 1, 10, 40, 87, 124, 85, 123, 186, 40, 85, 123, 327],
                TerminalsDisallowed::new(),
            ),
            (
                vec![0, 1, 10, 40, 87, 124, 85, 123, 202, 40, 85, 123, 327],
                TerminalsDisallowed::new(),
            ),
            (
                vec![
                    0, 1, 10, 40, 87, 124, 85, 123, 322, 92, 150, 92, 250, 343, 342,
                ],
                TerminalsDisallowed::new(),
            ),
        ]);

        let fast = apply_guarded_stack_shifts_fast(&before, &shifts, None)
            .expect("small branched floor should use bounded guarded effects");
        assert!(fast.semantically_eq(&expected, 16).unwrap());
    }

    #[test]
    fn guarded_stack_shift_advance_distributes_over_merged_branched_floor() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 329];
        action_rows[74] = vec![(
            token,
            Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![171],
                    }],
                    pop: 2,
                    pushes: vec![213, 265],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![323],
                    }],
                    pop: 2,
                    pushes: vec![328, 370],
                },
            ]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();
        let goto_rows = vec![Vec::new(); 329];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(329, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 171, 74], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 323, 74], acc);
        let merged = left.merge(&right);

        let expected = advance_stacks(&table, &left, token)
            .merge(&advance_stacks(&table, &right, token));
        let actual = advance_stacks(&table, &merged, token);

        let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks);
        assert!(stack_may_advance_on(&table, &merged, token));
    }

    #[test]
    fn stack_shift_advance_distributes_over_merged_branched_floor() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 75];
        action_rows[74] = vec![(
            token,
            Action::StackShifts(vec![StackShift {
                pop: 2,
                pushes: vec![265],
            }]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();
        let goto_rows = vec![Vec::new(); 75];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(75, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 171, 74], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 323, 74], acc);
        let merged = left.merge(&right);

        let expected = advance_stacks(&table, &left, token)
            .merge(&advance_stacks(&table, &right, token));
        let actual = advance_stacks(&table, &merged, token);

        let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks);
    }

    fn assert_advance_distributes_over_merge(
        table: &GLRTable,
        left: &ParserGSS,
        right: &ParserGSS,
        token: u32,
        case: &str,
    ) {
        let merged = left.merge(right);
        let expected = advance_stacks(table, left, token)
            .merge(&advance_stacks(table, right, token));
        let actual = advance_stacks(table, &merged, token);
        let concrete_reference = advance_concrete_stacks_reference(table, &merged, token);

        let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut actual_stacks = actual.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut reference_stacks = concrete_reference.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        reference_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks, "{case}");
        assert_eq!(actual_stacks, reference_stacks, "{case}");
    }

    fn branched_floor_pair(common_suffix_len: usize) -> (ParserGSS, ParserGSS) {
        let suffix = [10_u32, 11, 12];
        let mut left = vec![0, 1];
        left.extend_from_slice(&suffix[..common_suffix_len]);
        let mut right = vec![0, 2];
        right.extend_from_slice(&suffix[..common_suffix_len]);
        let acc = TerminalsDisallowed::new();
        (
            ParserGSS::from_single_stack(left, acc.clone()),
            ParserGSS::from_single_stack(right, acc),
        )
    }

    fn assert_advance_matches_concrete_reference_case(
        table: &GLRTable,
        before: &ParserGSS,
        token: u32,
        case: &str,
    ) {
        let actual = advance_stacks(table, before, token);
        let expected = advance_concrete_stacks_reference(table, before, token);
        assert_eq!(
            super::normalized_concrete_stacks(&actual),
            super::normalized_concrete_stacks(&expected),
            "{case}: before={:?}",
            before.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        );
    }

    #[test]
    fn generated_mixed_pop1_frontier_actions_match_concrete_reference() {
        let token = 0;
        let nt0 = 0;
        let nt1 = 1;
        let actions = vec![
            None,
            Some(Action::Shift(40, false)),
            Some(Action::Shift(41, true)),
            Some(Action::StackShifts(vec![StackShift {
                pop: 0,
                pushes: vec![42],
            }])),
            Some(Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![43],
            }])),
            Some(Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![44, 45],
            }])),
            Some(Action::StackShifts(vec![StackShift {
                pop: 2,
                pushes: vec![46],
            }])),
            Some(Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![47],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![48],
                },
            ])),
            Some(Action::Reduce(nt0, 1)),
            Some(Action::Reduce(nt1, 1)),
            Some(Action::Split {
                shift: Some((49, false)),
                reduces: vec![(nt0, 1)],
                accept: false,
            }),
            Some(Action::Split {
                shift: None,
                reduces: vec![(nt0, 1), (nt1, 1)],
                accept: false,
            }),
            Some(Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![10],
                }],
                pop: 2,
                pushes: vec![52],
            }])),
        ];

        for (left_index, left_action) in actions.iter().enumerate() {
            for (right_index, right_action) in actions.iter().enumerate() {
                let mut action_rows = vec![Vec::new(); 64];
                if let Some(action) = left_action {
                    action_rows[20].push((token, action.clone()));
                }
                if let Some(action) = right_action {
                    action_rows[21].push((token, action.clone()));
                }
                action_rows[30].push((token, Action::Shift(50, false)));
                action_rows[31].push((
                    token,
                    Action::StackShifts(vec![StackShift {
                        pop: 1,
                        pushes: vec![51],
                    }]),
                ));
                let action_refs: Vec<&[(u32, Action)]> =
                    action_rows.iter().map(Vec::as_slice).collect();

                let mut goto_rows = vec![Vec::new(); 64];
                goto_rows[10].push((nt0, (30, false)));
                goto_rows[10].push((nt1, (31, false)));
                let goto_refs: Vec<&[(u32, (u32, bool))]> =
                    goto_rows.iter().map(Vec::as_slice).collect();
                let table = build_test_table(64, 1, &action_refs, &goto_refs);

                let left_acc = TerminalsDisallowed::new().with_insert(60, 0);
                let right_acc = TerminalsDisallowed::new().with_insert(61, 0);
                let before = ParserGSS::from_stacks(&[
                    (vec![0, 10, 20], left_acc),
                    (vec![0, 10, 21], right_acc),
                ]);
                let case = format!(
                    "left_index={left_index} left={left_action:?} right_index={right_index} right={right_action:?}",
                );
                assert_advance_matches_concrete_reference_case(&table, &before, token, &case);
            }
        }
    }

    #[test]
    fn mixed_top_replace_wave_handles_machine_schema_step333_shape() {
        let token = 0;
        let nt0 = 0;
        let nt1 = 1;
        let mut action_rows = vec![Vec::new(); 400];
        action_rows[355].push((token, Action::Shift(264, true)));
        action_rows[363].push((token, Action::Reduce(nt0, 1)));
        action_rows[373].push((token, Action::Reduce(nt1, 1)));
        action_rows[193].push((token, Action::Shift(280, true)));
        action_rows[208].push((token, Action::Shift(299, true)));
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();

        let mut goto_rows = vec![Vec::new(); 400];
        goto_rows[10].push((nt0, (193, false)));
        goto_rows[10].push((nt1, (208, false)));
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(400, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1, 10, 327], acc.clone()),
            (vec![0, 2, 10, 327], acc.clone()),
            (vec![0, 1, 10, 342], acc.clone()),
            (vec![0, 1, 10, 355], acc.clone()),
            (vec![0, 2, 10, 355], acc.clone()),
            (vec![0, 1, 10, 363], acc.clone()),
            (vec![0, 2, 10, 363], acc.clone()),
            (vec![0, 1, 10, 373], acc.clone()),
            (vec![0, 2, 10, 373], acc),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 1, 10, 264], TerminalsDisallowed::new()),
            (vec![0, 2, 10, 264], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 280], TerminalsDisallowed::new()),
            (vec![0, 2, 10, 280], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 299], TerminalsDisallowed::new()),
            (vec![0, 2, 10, 299], TerminalsDisallowed::new()),
        ]);

        let remapped = try_advance_mixed_top_replace_wave(&table, &before, token)
            .expect("machine schema step333 shape should be a direct top remap");
        assert!(remapped.semantically_eq(&expected, 16).unwrap());
        assert!(advance_stacks(&table, &before, token)
            .semantically_eq(&expected, 16)
            .unwrap());
    }

    #[test]
    fn mixed_top_replace_wave_handles_reduce_plus_dead_frontier() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 200];
        action_rows[99].push((token, Action::Reduce(nt, 1)));
        action_rows[112].push((token, Action::Shift(170, true)));
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();
        let mut goto_rows = vec![Vec::new(); 200];
        goto_rows[56].push((nt, (112, false)));
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(200, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1, 40, 56, 99], acc.clone()),
            (vec![0, 1, 40, 56, 83], acc),
        ]);
        let expected = ParserGSS::from_single_stack(
            vec![0, 1, 40, 56, 170],
            TerminalsDisallowed::new(),
        );

        let fast = try_advance_mixed_top_replace_wave(&table, &before, token)
            .expect("reduce plus dead frontier should be a direct top remap");
        assert!(fast.semantically_eq(&expected, 8).unwrap());
        assert!(advance_stacks(&table, &before, token)
            .semantically_eq(&expected, 8)
            .unwrap());
    }

    #[test]
    fn bounded_deterministic_reduce_paths_handle_predecessor_dependent_chains() {
        let token = 0;
        let nt_value = 0;
        let nt_regular = 1;
        let nt_deep = 2;
        let mut action_rows = vec![Vec::new(); 500];
        action_rows[90].push((token, Action::Reduce(nt_value, 1)));
        action_rows[335].push((token, Action::Reduce(nt_regular, 1)));
        action_rows[336].push((token, Action::Shift(337, true)));
        action_rows[446].push((token, Action::Reduce(nt_deep, 1)));
        action_rows[343].push((
            token,
            Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 2,
                        states: vec![59],
                    }],
                    pop: 3,
                    pushes: vec![107],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 2,
                        states: vec![80],
                    }],
                    pop: 3,
                    pushes: vec![119],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 2,
                        states: vec![92],
                    }],
                    pop: 3,
                    pushes: vec![135],
                },
            ]),
        ));
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();

        let mut goto_rows = vec![Vec::new(); 500];
        goto_rows[389].push((nt_value, (335, true)));
        goto_rows[426].push((nt_value, (446, true)));
        goto_rows[138].push((nt_regular, (336, false)));
        goto_rows[250].push((nt_deep, (343, false)));
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(500, 1, &action_refs, &goto_refs);

        let acc_a = TerminalsDisallowed::new().with_insert(60, 0);
        let acc_b = TerminalsDisallowed::new().with_insert(61, 0);
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1, 168, 138, 389, 90], acc_a.clone()),
            (vec![0, 1, 181, 138, 389, 90], acc_a.clone()),
            (vec![0, 1, 198, 138, 389, 90], acc_a.clone()),
            (vec![0, 1, 322, 91, 59, 250, 426, 90], acc_b.clone()),
            (vec![0, 1, 322, 91, 80, 250, 426, 90], acc_b.clone()),
            (vec![0, 1, 322, 91, 92, 250, 426, 90], acc_b.clone()),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 1, 168, 138, 337], acc_a.clone()),
            (vec![0, 1, 181, 138, 337], acc_a.clone()),
            (vec![0, 1, 198, 138, 337], acc_a),
            (vec![0, 1, 322, 91, 107], acc_b.clone()),
            (vec![0, 1, 322, 91, 119], acc_b.clone()),
            (vec![0, 1, 322, 91, 135], acc_b),
        ]);

        let fast = try_advance_bounded_concrete_paths(&table, &before, token)
            .expect("small predecessor-dependent reduce chains should remain bounded");
        assert!(fast.semantically_eq(&expected, 16).unwrap());
        assert!(advance_stacks(&table, &before, token)
            .semantically_eq(&expected, 16)
            .unwrap());
    }

    #[test]
    fn bounded_concrete_paths_handle_guarded_shift_and_pure_shift() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 64];
        action_rows[10].push((
            token,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![1],
                }],
                pop: 2,
                pushes: vec![20],
            }]),
        ));
        action_rows[11].push((token, Action::Shift(30, true)));
        let action_refs = action_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let goto_rows = vec![Vec::new(); 64];
        let goto_refs = goto_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let table = build_test_table(64, 1, &action_refs, &goto_refs);

        let before = ParserGSS::from_stacks(&[
            (vec![0, 1, 10], TerminalsDisallowed::new()),
            (vec![0, 2, 11], TerminalsDisallowed::new()),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 20], TerminalsDisallowed::new()),
            (vec![0, 2, 30], TerminalsDisallowed::new()),
        ]);

        let fast = try_advance_bounded_concrete_paths(&table, &before, token)
            .expect("small guarded/shift frontier should stay bounded");
        assert!(fast.semantically_eq(&expected, 8).unwrap());
        assert!(advance_stacks(&table, &before, token)
            .semantically_eq(&expected, 8)
            .unwrap());
    }

    #[test]
    fn top_local_stack_effects_preserve_upper_branch_accumulators() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 64];
        action_rows[10].push((
            token,
            Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![20, 30],
            }]),
        ));
        action_rows[11].push((token, Action::Shift(40, false)));
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();
        let goto_rows = vec![Vec::new(); 64];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(64, 1, &action_refs, &goto_refs);

        let acc_a = TerminalsDisallowed::new().with_insert(60, 0);
        let acc_b = TerminalsDisallowed::new().with_insert(61, 0);
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1, 10], acc_a.clone()),
            (vec![0, 2, 11], acc_b.clone()),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 1, 20, 30], acc_a),
            (vec![0, 2, 11, 40], acc_b),
        ]);

        let fast = try_advance_pop1_stackshift_shift_wave(&table, &before, token)
            .expect("top-local effects should traverse upper accumulator branches");
        assert!(fast.semantically_eq(&expected, 8).unwrap());
        assert!(advance_stacks(&table, &before, token)
            .semantically_eq(&expected, 8)
            .unwrap());
    }

    #[test]
    fn top_local_stack_effect_wave_handles_machine_schema_step333_followup() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 500];
        action_rows[264].push((
            token,
            Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![171, 88],
            }]),
        ));
        action_rows[280].push((
            token,
            Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![185, 288],
            }]),
        ));
        action_rows[299].push((
            token,
            Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![201, 288],
            }]),
        ));
        action_rows[426].push((token, Action::Shift(92, false)));
        action_rows[322].push((token, Action::Shift(92, false)));
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();
        let goto_rows = vec![Vec::new(); 500];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(500, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1, 10, 186, 40, 85, 123, 264], acc.clone()),
            (vec![0, 1, 10, 202, 40, 85, 123, 264], acc.clone()),
            (vec![0, 1, 10, 186, 40, 85, 123, 280], acc.clone()),
            (vec![0, 1, 10, 202, 40, 85, 123, 280], acc.clone()),
            (vec![0, 1, 10, 186, 40, 85, 123, 299], acc.clone()),
            (vec![0, 1, 10, 202, 40, 85, 123, 299], acc.clone()),
            (vec![0, 1, 10, 322, 92, 150, 250, 343, 426], acc.clone()),
            (vec![0, 1, 10, 186, 40, 85, 123, 322], acc.clone()),
            (vec![0, 1, 10, 202, 40, 85, 123, 322], acc),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 1, 10, 322, 92, 150, 250, 343, 426, 92], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 186, 40, 85, 123, 322, 92], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 202, 40, 85, 123, 322, 92], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 186, 40, 85, 123, 171, 88], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 202, 40, 85, 123, 171, 88], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 186, 40, 85, 123, 185, 288], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 202, 40, 85, 123, 185, 288], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 186, 40, 85, 123, 201, 288], TerminalsDisallowed::new()),
            (vec![0, 1, 10, 202, 40, 85, 123, 201, 288], TerminalsDisallowed::new()),
        ]);

        let fast = try_advance_pop1_stackshift_shift_wave(&table, &before, token)
            .expect("machine schema follow-up should be top-local");
        assert!(fast.semantically_eq(&expected, 32).unwrap());
        assert!(advance_stacks(&table, &before, token)
            .semantically_eq(&expected, 32)
            .unwrap());
    }

    #[test]
    fn generated_top_action_advance_distributes_over_branched_floor_merge() {
        let token = 0;
        for common_suffix_len in 1..=3 {
            let (left, right) = branched_floor_pair(common_suffix_len);
            let top = 9 + common_suffix_len as u32;
            let mut actions = vec![
                Action::Shift(40, false),
                Action::Shift(40, true),
            ];

            for pop in 0..=common_suffix_len + 1 {
                actions.push(Action::StackShifts(vec![StackShift {
                    pop: pop as u32,
                    pushes: vec![40],
                }]));
                actions.push(Action::StackShifts(vec![StackShift {
                    pop: pop as u32,
                    pushes: Vec::new(),
                }]));
            }
            actions.push(Action::StackShifts(vec![
                StackShift {
                    pop: 0,
                    pushes: vec![40],
                },
                StackShift {
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![41],
                },
            ]));
            actions.push(Action::StackShifts(vec![
                StackShift {
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![40],
                },
                StackShift {
                    pop: 0,
                    pushes: vec![41],
                },
            ]));
            actions.push(Action::StackShifts(vec![
                StackShift {
                    pop: common_suffix_len as u32,
                    pushes: vec![40, 42],
                },
                StackShift {
                    pop: common_suffix_len as u32,
                    pushes: vec![41, 42],
                },
            ]));

            actions.push(Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: common_suffix_len as u32,
                        states: vec![1],
                    }],
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![40],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: common_suffix_len as u32,
                        states: vec![2],
                    }],
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![41],
                },
            ]));

            for (action_index, action) in actions.into_iter().enumerate() {
                let mut action_rows = vec![Vec::new(); 64];
                action_rows[top as usize] = vec![(token, action)];
                let action_refs: Vec<&[(u32, Action)]> =
                    action_rows.iter().map(Vec::as_slice).collect();
                let goto_rows = vec![Vec::new(); 64];
                let goto_refs: Vec<&[(u32, (u32, bool))]> =
                    goto_rows.iter().map(Vec::as_slice).collect();
                let table = build_test_table(64, 1, &action_refs, &goto_refs);
                let case = format!(
                    "common_suffix_len={common_suffix_len} action_index={action_index} action={:?}",
                    table.action(top, token),
                );
                assert_advance_distributes_over_merge(&table, &left, &right, token, &case);
            }
        }
    }

    #[test]
    fn generated_reduce_chain_advance_distributes_over_branched_floor_merge() {
        let token = 0;
        let nt = 0;
        for common_suffix_len in 1..=3 {
            let (left, right) = branched_floor_pair(common_suffix_len);
            let top = 9 + common_suffix_len as u32;
            for reduce_len in 1..=common_suffix_len {
                for is_replace in [false, true] {
                    let mut action_rows = vec![Vec::new(); 64];
                    action_rows[top as usize] =
                        vec![(token, Action::Reduce(nt, reduce_len as u32))];
                    action_rows[50] = vec![(token, Action::Shift(60, false))];
                    action_rows[51] = vec![(token, Action::Shift(60, false))];

                    let left_stacks = left.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
                    let right_stacks = right.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
                    let left_values = &left_stacks[0].0;
                    let right_values = &right_stacks[0].0;
                    let left_goto_from = left_values[left_values.len() - reduce_len - 1];
                    let right_goto_from = right_values[right_values.len() - reduce_len - 1];

                    let mut goto_rows = vec![Vec::new(); 64];
                    goto_rows[left_goto_from as usize].push((nt, (50, is_replace)));
                    if right_goto_from != left_goto_from {
                        goto_rows[right_goto_from as usize].push((nt, (51, is_replace)));
                    }

                    let action_refs: Vec<&[(u32, Action)]> =
                        action_rows.iter().map(Vec::as_slice).collect();
                    let goto_refs: Vec<&[(u32, (u32, bool))]> =
                        goto_rows.iter().map(Vec::as_slice).collect();
                    let table = build_test_table(64, 1, &action_refs, &goto_refs);
                    let case = format!(
                        "reduce common_suffix_len={common_suffix_len} reduce_len={reduce_len} is_replace={is_replace} left_goto={left_goto_from} right_goto={right_goto_from}",
                    );
                    assert_advance_distributes_over_merge(&table, &left, &right, token, &case);
                }
            }
        }
    }

    #[test]
    fn exact_admission_rejects_union_reduce_with_no_real_goto() {
        let token = 0;
        let nt = 0;
        let mut table = build_test_table(
            3,
            1,
            &[&[], &[], &[(token, Action::Reduce(nt, 1))]],
            &[&[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

        assert!(!stack_may_advance_on(&table, &stack, token));
    }

    #[test]
    fn exact_admission_accepts_reduce_goto_then_shift_path() {
        let token = 0;
        let nt = 0;
        let mut table = build_test_table(
            5,
            1,
            &[
                &[],
                &[],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

        assert!(stack_may_advance_on(&table, &stack, token));
    }

    #[test]
    fn exact_admission_any_uses_same_exactness_as_single_terminal() {
        let token = 0;
        let nt = 0;
        let mut table = build_test_table(
            5,
            2,
            &[
                &[],
                &[],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

        let mut terminals = BitSet::new(3);
        terminals.set(token as usize);
        assert_eq!(
            stack_may_advance_on(&table, &stack, token),
            stack_may_advance_on_any(&table, &stack, &terminals)
        );
    }

    fn assert_exact_any_matches_disjunction(
        table: &GLRTable,
        stack: &ParserGSS,
        terminals: &BitSet,
    ) {
        let disjunction = terminals.iter_ones().any(|bit| match bit {
            bit if bit == table.num_terminals as usize => stack_may_advance_on(table, stack, EOF),
            bit if bit < table.num_terminals as usize => {
                stack_may_advance_on(table, stack, bit as u32)
            }
            _ => false,
        });

        assert_eq!(
            stack_may_advance_on_any(table, stack, terminals),
            disjunction
        );
    }

    fn assert_admissible_set_matches_individual(
        table: &GLRTable,
        stack: &ParserGSS,
        terminals: &BitSet,
    ) {
        let admitted = stack_admissible_terminals(table, stack, terminals);
        let mut expected = BitSet::new(terminals.len());
        for bit in terminals.iter_ones() {
            let terminal = if bit == table.num_terminals as usize {
                EOF
            } else if bit < table.num_terminals as usize {
                bit as u32
            } else {
                continue;
            };
            if stack_may_advance_on(table, stack, terminal) {
                expected.set(bit);
            }
        }
        assert_eq!(admitted, expected);
    }

    #[test]
    fn exact_admission_any_does_not_mix_lookahead_reductions() {
        let reduce_token = 0;
        let shift_token = 1;
        let nt = 0;
        let mut table = build_test_table(
            5,
            2,
            &[
                &[],
                &[],
                &[(reduce_token, Action::Reduce(nt, 1))],
                &[(shift_token, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
        let mut terminals = BitSet::new(3);
        terminals.set(reduce_token as usize);
        terminals.set(shift_token as usize);

        assert!(!stack_may_advance_on(&table, &stack, reduce_token));
        assert!(!stack_may_advance_on(&table, &stack, shift_token));
        assert_exact_any_matches_disjunction(&table, &stack, &terminals);
        assert_admissible_set_matches_individual(&table, &stack, &terminals);
        assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn exact_admission_any_batches_reduce_frontier_by_terminal_set() {
        let token_a = 0;
        let token_b = 1;
        let nt = 0;
        let mut table = build_test_table(
            5,
            2,
            &[
                &[],
                &[],
                &[
                    (token_a, Action::Reduce(nt, 1)),
                    (token_b, Action::Reduce(nt, 1)),
                ],
                &[(token_b, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
        let mut terminals = BitSet::new(3);
        terminals.set(token_a as usize);
        terminals.set(token_b as usize);

        assert!(!stack_may_advance_on(&table, &stack, token_a));
        assert!(stack_may_advance_on(&table, &stack, token_b));
        assert_exact_any_matches_disjunction(&table, &stack, &terminals);
        assert_admissible_set_matches_individual(&table, &stack, &terminals);
        assert!(stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn exact_admission_any_preserves_guarded_shift_checks() {
        let token = 0;
        let other_token = 1;
        let mut table = build_test_table(
            3,
            2,
            &[
                &[],
                &[],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![0],
                        }],
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let mut terminals = BitSet::new(3);
        terminals.set(token as usize);
        terminals.set(other_token as usize);

        let rejected = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());
        assert_exact_any_matches_disjunction(&table, &rejected, &terminals);
        assert_admissible_set_matches_individual(&table, &rejected, &terminals);
        assert!(!stack_may_advance_on_any(&table, &rejected, &terminals));

        let accepted = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
        assert_exact_any_matches_disjunction(&table, &accepted, &terminals);
        assert_admissible_set_matches_individual(&table, &accepted, &terminals);
        assert!(stack_may_advance_on_any(&table, &accepted, &terminals));
    }

    #[test]
    fn exact_admission_any_handles_eof_acceptance() {
        let token = 0;
        let mut table = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());

        let mut non_eof = BitSet::new(2);
        non_eof.set(token as usize);
        assert_exact_any_matches_disjunction(&table, &stack, &non_eof);
        assert_admissible_set_matches_individual(&table, &stack, &non_eof);
        assert!(!stack_may_advance_on_any(&table, &stack, &non_eof));

        let mut eof = BitSet::new(2);
        eof.set(table.num_terminals as usize);
        assert_exact_any_matches_disjunction(&table, &stack, &eof);
        assert_admissible_set_matches_individual(&table, &stack, &eof);
        assert!(stack_may_advance_on_any(&table, &stack, &eof));
    }

    /// A nullable-skip / trivia reduction leaves an EOF `Reduce` on the top
    /// state. That is NOT completion when the reduction has no feasible goto
    /// continuation to an accepting state.
    #[test]
    fn eof_reduce_without_root_accept_is_incomplete() {
        let table = build_test_table(
            3,
            1,
            &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
            &[&[], &[], &[]],
        );
        let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        assert!(
            !stacks_finished(&table, &stack),
            "an EOF reduce with no goto continuation must not count as completion",
        );
    }

    /// The same EOF `Reduce` DOES complete when its reduction chain has a
    /// feasible goto to an accepting state.
    #[test]
    fn eof_reduction_chain_reaches_root_accept() {
        let table = build_test_table(
            3,
            1,
            &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
            &[&[(0, (2, false))], &[], &[]],
        );
        let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        assert!(
            stacks_finished(&table, &stack),
            "an EOF reduce chain that reaches an accepting state must complete",
        );
    }

    /// A nullable/empty root accepts EOF directly from the start state.
    #[test]
    fn nullable_empty_root_eof_accepts() {
        let table = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
        let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(stacks_finished(&table, &stack));
    }

    /// A valid reduce branch to Accept completes even when an ambiguous sibling
    /// reduce branch is dead.
    #[test]
    fn ambiguous_eof_split_with_dead_branch_still_completes() {
        let table = build_test_table(
            4,
            1,
            &[
                &[],
                &[(EOF, Action::Split {
                    shift: None,
                    reduces: vec![(0, 1), (1, 1)],
                    accept: false,
                })],
                &[(EOF, Action::Accept)],
                &[],
            ],
            &[&[(0, (2, false))], &[], &[], &[]],
        );
        let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        assert!(
            stacks_finished(&table, &stack),
            "an ambiguous EOF split completes when one reduce branch reaches Accept",
        );
    }

    /// An EOF shift is ordinary progress, not completion: `may_advance` must
    /// stay true while the completion predicate rejects it.
    #[test]
    fn eof_shift_may_advance_but_is_not_completion() {
        let mut table =
            build_test_table(2, 1, &[&[(EOF, Action::Shift(1, false))], &[]], &[&[], &[]]);
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(
            stack_may_advance_on(&table, &stack, EOF),
            "an EOF shift is an ordinary advance",
        );
        assert!(
            !stacks_finished(&table, &stack),
            "an EOF shift alone must not count as completion",
        );
    }

    /// An EOF skip (scoped-ignore shape) is ordinary progress, not completion.
    #[test]
    fn eof_skip_may_advance_but_is_not_completion() {
        let mut table = build_test_table(1, 1, &[&[(EOF, Action::Skip)]], &[&[]]);
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(
            stack_may_advance_on(&table, &stack, EOF),
            "an EOF skip is an ordinary advance",
        );
        assert!(
            !stacks_finished(&table, &stack),
            "an EOF skip alone must not count as completion",
        );
    }

    /// A split with only a shift branch (no accepting branch) is not
    /// completion, even though the shift is an ordinary advance.
    #[test]
    fn eof_split_shift_only_is_not_completion() {
        let mut table = build_test_table(
            2,
            1,
            &[
                &[],
                &[(EOF, Action::Split {
                    shift: Some((1, false)),
                    reduces: Vec::new(),
                    accept: false,
                })],
            ],
            &[&[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        assert!(
            stack_may_advance_on(&table, &stack, EOF),
            "an EOF split shift is an ordinary advance",
        );
        assert!(
            !stacks_finished(&table, &stack),
            "an EOF split with only a shift branch must not count as completion",
        );
    }

    #[test]
    fn provider_eof_suite_mirrors_concrete_semantics() {
        // 1. Orphan reduce false: reduce with no goto continuation does not complete
        let table1 = build_test_table(
            3,
            1,
            &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
            &[&[], &[], &[]],
        );
        let provider1 = GLRTableActionProvider::new(&table1);
        let stack1 = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        assert!(!stacks_finished_with_provider(&provider1, &stack1, EOF));

        // 2. Reduce chain Accept true: reduce reaches Accept via goto
        let table2 = build_test_table(
            3,
            1,
            &[&[], &[(EOF, Action::Reduce(0, 1))], &[(EOF, Action::Accept)]],
            &[&[(0, (2, false))], &[], &[]],
        );
        let provider2 = GLRTableActionProvider::new(&table2);
        let stack2 = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        assert!(stacks_finished_with_provider(&provider2, &stack2, EOF));

        // 3. Nullable empty root true
        let table3 = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
        let provider3 = GLRTableActionProvider::new(&table3);
        let stack3 = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(stacks_finished_with_provider(&provider3, &stack3, EOF));

        // 4. Ambiguous dead branch + Accept true
        let table4 = build_test_table(
            4,
            1,
            &[
                &[],
                &[(EOF, Action::Split {
                    shift: None,
                    reduces: vec![(0, 1), (1, 1)],
                    accept: false,
                })],
                &[(EOF, Action::Accept)],
                &[],
            ],
            &[&[(0, (2, false))], &[], &[], &[]],
        );
        let provider4 = GLRTableActionProvider::new(&table4);
        let stack4 = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        assert!(stacks_finished_with_provider(&provider4, &stack4, EOF));

        // 5. EOF shift/skip/stackshift false
        let table_shift = build_test_table(2, 1, &[&[(EOF, Action::Shift(1, false))], &[]], &[&[], &[]]);
        let provider_shift = GLRTableActionProvider::new(&table_shift);
        let stack_shift = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(!stacks_finished_with_provider(&provider_shift, &stack_shift, EOF));

        let table_skip = build_test_table(1, 1, &[&[(EOF, Action::Skip)]], &[&[]]);
        let provider_skip = GLRTableActionProvider::new(&table_skip);
        let stack_skip = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(!stacks_finished_with_provider(&provider_skip, &stack_skip, EOF));

        let table_stackshift = build_test_table(
            2,
            1,
            &[&[(EOF, Action::StackShifts(vec![StackShift { pop: 0, pushes: vec![1] }]))], &[]],
            &[&[], &[]],
        );
        let provider_stackshift = GLRTableActionProvider::new(&table_stackshift);
        let stack_stackshift = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(!stacks_finished_with_provider(&provider_stackshift, &stack_stackshift, EOF));

        // 6. No action / empty stack false
        let table_no_action = build_test_table(1, 1, &[&[]], &[&[]]);
        let provider_no_action = GLRTableActionProvider::new(&table_no_action);
        let stack_no_action = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(!stacks_finished_with_provider(&provider_no_action, &stack_no_action, EOF));
        assert!(!stacks_finished_with_provider(&provider_no_action, &ParserGSS::empty(), EOF));

        // 7. Infeasible pop false (reduce length > stack depth)
        let table_infeasible = build_test_table(
            2,
            1,
            &[&[(EOF, Action::Reduce(0, 5))], &[(EOF, Action::Accept)]],
            &[&[(0, (1, false))], &[]],
        );
        let provider_infeasible = GLRTableActionProvider::new(&table_infeasible);
        let stack_infeasible = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(!stacks_finished_with_provider(&provider_infeasible, &stack_infeasible, EOF));

        // 8. Reduction cycle terminates false
        let table_cycle = build_test_table(
            2,
            1,
            &[&[(EOF, Action::Reduce(0, 1))], &[]],
            &[&[(0, (0, false))], &[]],
        );
        let provider_cycle = GLRTableActionProvider::new(&table_cycle);
        let stack_cycle = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(!stacks_finished_with_provider(&provider_cycle, &stack_cycle, EOF));
    }

    #[test]
    fn provider_split_with_accept_shift_and_reduce_preserves_shifts_and_accepts() {
        let token = 0;
        let nt = 0;
        // States:
        // State 0 (base): goto on nt -> State 2
        // State 1 (top): Split with accept + shift(3) + reduce(nt, 1)
        // State 2: reduction goto target, shifts 4 on token
        // State 3: direct shift target
        // State 4: reduction-derived shift target
        let table = build_test_table(
            5,
            1,
            &[
                &[],
                &[(
                    token,
                    Action::Split {
                        shift: Some((3, false)),
                        reduces: vec![(nt, 1)],
                        accept: true,
                    },
                )],
                &[(token, Action::Shift(4, false))],
                &[],
                &[],
            ],
            &[
                &[(nt, (2, false))],
                &[],
                &[],
                &[],
                &[],
            ],
        );
        let provider = GLRTableActionProvider::new(&table);
        let stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());

        // 1. Advance mode: verifies both direct shift (state 3) and reduction-derived shift (state 4) are preserved
        let advanced = advance_stacks_with_provider(&provider, stack.clone(), token);
        let paths = advanced.to_stacks(32).expect("should enumerate paths");
        let top_states: FxHashSet<u32> = paths.iter().map(|(s, _)| *s.last().unwrap()).collect();
        assert!(
            top_states.contains(&3),
            "Advance must preserve direct shift from Action::Split; top_states={:?}",
            top_states
        );
        assert!(
            top_states.contains(&4),
            "Advance must preserve reduction-derived shift from Action::Split; top_states={:?}",
            top_states
        );
        assert_eq!(top_states.len(), 2);

        // 2. Completion mode: verifies accept: true causes stacks_finished_with_provider to accept
        assert!(
            stacks_finished_with_provider(&provider, &stack, token),
            "Completion must accept when Action::Split has accept: true"
        );
    }

    #[test]
    fn provider_eof_non_acceptance_variants() {
        // EOF extra_stack_shifts alone does not count as acceptance
        struct ExtraStackShiftsProvider;
        impl ParserActionProvider for ExtraStackShiftsProvider {
            type Symbol = TerminalID;
            fn action(&self, _state: u32, _symbol: TerminalID) -> Option<ProvidedAction<'_>> {
                Some(ProvidedAction {
                    action: ProvidedActionRef::Identity,
                    reduction_scope: 0,
                    extra_stack_shifts: smallvec::smallvec![StackShift {
                        pop: 0,
                        pushes: vec![1],
                    }],
                })
            }
            fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
                Some(local)
            }
            fn goto_target(&self, _scope: u32, _from: u32, _nt: u32) -> Option<(u32, bool)> {
                None
            }
            fn state_count_hint(&self) -> usize {
                2
            }
        }

        let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(
            !stacks_finished_with_provider(&ExtraStackShiftsProvider, &stack, EOF),
            "extra_stack_shifts must not count as acceptance in Completion mode"
        );

        // EOF Identity alone does not count as acceptance
        struct IdentityProvider;
        impl ParserActionProvider for IdentityProvider {
            type Symbol = TerminalID;
            fn action(&self, _state: u32, _symbol: TerminalID) -> Option<ProvidedAction<'_>> {
                Some(ProvidedAction {
                    action: ProvidedActionRef::Identity,
                    reduction_scope: 0,
                    extra_stack_shifts: SmallVec::new(),
                })
            }
            fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
                Some(local)
            }
            fn goto_target(&self, _scope: u32, _from: u32, _nt: u32) -> Option<(u32, bool)> {
                None
            }
            fn state_count_hint(&self) -> usize {
                1
            }
        }

        assert!(
            !stacks_finished_with_provider(&IdentityProvider, &stack, EOF),
            "Identity must not count as acceptance in Completion mode"
        );
    }

    #[test]
    fn provider_nullable_growing_stack_cycle_bounded_witness() {
        let nt = 0;
        let table = build_test_table(
            2,
            1,
            &[
                &[(EOF, Action::Reduce(nt, 0))],
                &[(EOF, Action::Reduce(nt, 0))],
            ],
            &[
                &[(nt, (1, false))],
                &[(nt, (1, false))],
            ],
        );
        let provider = GLRTableActionProvider::new(&table);
        let mut key_interner = GssSemanticKeyInterner::<u32, TerminalsDisallowed>::new();
        let mut visited = FxHashSet::<u32>::default();

        let mut current = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        let mut keys = Vec::new();

        for _ in 0..10 {
            let mut next = ParserGSS::empty();
            for state in current.peek_values() {
                let provided = provider.action(state, EOF).unwrap();
                let isolated = current.isolate(Some(state));
                if let ProvidedActionRef::Local { action, .. } = provided.action {
                    action.for_each_reduce(|nt, len| {
                        for (goto_from, base) in
                            reduce_sources_from_isolated(&isolated, len as usize)
                        {
                            if let Some((target, is_replace)) =
                                provider.goto_target(provided.reduction_scope, goto_from, nt)
                            {
                                let branch = if is_replace {
                                    base.popn(1).push(target)
                                } else {
                                    base.push(target)
                                };
                                let key = key_interner.key(&branch);
                                assert!(
                                    visited.insert(key),
                                    "key must be fresh because stack depth increased"
                                );
                                keys.push(key);
                                merge_into(&mut next, branch);
                            }
                        }
                    });
                }
            }
            assert!(!next.is_empty());
            current = next;
        }
        // NOTE: This test is explicitly diagnostic. Passing this probe confirms that whole-stack
        // deduplication alone does not terminate nullable growing-stack cycles; it does NOT imply
        // fixed correctness of unbounded cycle termination.
        assert_eq!(keys.len(), 10);
        let unique_keys: FxHashSet<_> = keys.iter().copied().collect();
        assert_eq!(
            unique_keys.len(),
            10,
            "whole-stack dedup generated unique keys for all growing stacks; dedup does not terminate growing-stack cycles"
        );
    }

    #[test]
    fn provider_eof_reduction_exposing_control_transition_accepts() {
        // Test synthetic chain: reduce -> control -> Accept on EOF.
        // Stack starts at [0, 10].
        // State 10 on EOF: Reduce(nt=0, len=1).
        // Goto from 0 on nt=0: State 1.
        // State 1 has NO EOF action, but exposes control symbol CTRL (999).
        // On CTRL (999): State 1 shifts to State 2.
        // State 2 on EOF: Action::Accept.
        const CTRL_SYM: u32 = 999;
        static ACTION_REDUCE: Action = Action::Reduce(0, 1);
        static ACTION_SHIFT_CTRL: Action = Action::Shift(2, false);
        static ACTION_ACCEPT: Action = Action::Accept;

        struct ReduceControlAcceptProvider;
        impl ParserActionProvider for ReduceControlAcceptProvider {
            type Symbol = u32;
            fn action(&self, state: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
                match (state, symbol) {
                    (10, EOF) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &ACTION_REDUCE },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    (1, CTRL_SYM) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &ACTION_SHIFT_CTRL },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    (2, EOF) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &ACTION_ACCEPT },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    _ => None,
                }
            }
            fn control_symbols(&self, state: u32, out: &mut SmallVec<[Self::Symbol; 4]>) {
                if state == 1 {
                    out.push(CTRL_SYM);
                }
            }
            fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
                Some(local)
            }
            fn goto_target(&self, _scope: u32, goto_from: u32, nt: u32) -> Option<(u32, bool)> {
                if goto_from == 0 && nt == 0 {
                    Some((1, false))
                } else {
                    None
                }
            }
            fn state_count_hint(&self) -> usize {
                16
            }
        }

        let provider = ReduceControlAcceptProvider;
        let stack = ParserGSS::from_single_stack(vec![0, 10], TerminalsDisallowed::new());

        // Completion-mode traversal closes controls on the next reduction frontier,
        // so post-reduction control transitions are explored and reach Accept.
        let finished = stacks_finished_with_provider(&provider, &stack, EOF);
        assert!(
            finished,
            "Completion-mode traversal closes controls on the next reduction frontier, exposing post-reduction control transitions to Accept"
        );

        // Retain separate post-reduction frontier direct verification:
        let post_reduce_stack = ParserGSS::from_single_stack(vec![0, 1], TerminalsDisallowed::new());
        let closed_post = close_provider_control_stacks(&provider, &post_reduce_stack);
        assert!(
            stacks_finished_with_provider(&provider, &closed_post, EOF),
            "When post-reduction frontier is control-closed, it reaches State 2 which accepts on EOF"
        );
    }

    #[test]
    fn provider_eof_reduction_two_stage_control_accepts() {
        // Chain: reduce1 -> control1 -> reduce2 -> control2 -> Accept
        // [0, 10] -> Reduce(nt0=0, len=1) -> [0, 1]
        // [0, 1] on CTRL1 (991) -> shifts 2 -> [0, 1, 2]
        // [0, 1, 2] on EOF -> Reduce(nt1=1, len=2) -> base [0], goto nt1=1 -> [0, 3]
        // [0, 3] on CTRL2 (992) -> shifts 4 -> [0, 3, 4]
        // [0, 3, 4] on EOF -> Accept
        const CTRL1: u32 = 991;
        const CTRL2: u32 = 992;
        static REDUCE_1: Action = Action::Reduce(0, 1);
        static SHIFT_CTRL1: Action = Action::Shift(2, false);
        static REDUCE_2: Action = Action::Reduce(1, 2);
        static SHIFT_CTRL2: Action = Action::Shift(4, false);
        static ACCEPT_ACT: Action = Action::Accept;

        struct TwoStageProvider;
        impl ParserActionProvider for TwoStageProvider {
            type Symbol = u32;
            fn action(&self, state: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
                match (state, symbol) {
                    (10, EOF) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &REDUCE_1 },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    (1, CTRL1) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &SHIFT_CTRL1 },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    (2, EOF) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &REDUCE_2 },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    (3, CTRL2) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &SHIFT_CTRL2 },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    (4, EOF) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &ACCEPT_ACT },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    _ => None,
                }
            }
            fn control_symbols(&self, state: u32, out: &mut SmallVec<[Self::Symbol; 4]>) {
                match state {
                    1 => out.push(CTRL1),
                    3 => out.push(CTRL2),
                    _ => {}
                }
            }
            fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
                Some(local)
            }
            fn goto_target(&self, _scope: u32, goto_from: u32, nt: u32) -> Option<(u32, bool)> {
                match (goto_from, nt) {
                    (0, 0) => Some((1, false)),
                    (0, 1) => Some((3, false)),
                    _ => None,
                }
            }
            fn state_count_hint(&self) -> usize {
                16
            }
        }

        let provider = TwoStageProvider;
        let stack = ParserGSS::from_single_stack(vec![0, 10], TerminalsDisallowed::new());
        assert!(
            stacks_finished_with_provider(&provider, &stack, EOF),
            "Two consecutive reduce/control stages should complete to Accept"
        );
    }

    #[test]
    fn provider_eof_reduction_exposing_control_transition_no_accept_rejects() {
        // Negative case: reduce -> control -> non-accepting state
        const CTRL_SYM: u32 = 999;
        static ACTION_REDUCE: Action = Action::Reduce(0, 1);
        static ACTION_SHIFT_CTRL: Action = Action::Shift(2, false);

        struct NoAcceptProvider;
        impl ParserActionProvider for NoAcceptProvider {
            type Symbol = u32;
            fn action(&self, state: u32, symbol: u32) -> Option<ProvidedAction<'_>> {
                match (state, symbol) {
                    (10, EOF) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &ACTION_REDUCE },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    (1, CTRL_SYM) => Some(ProvidedAction {
                        action: ProvidedActionRef::Local { scope: 0, action: &ACTION_SHIFT_CTRL },
                        reduction_scope: 0,
                        extra_stack_shifts: SmallVec::new(),
                    }),
                    // State 2 has no EOF action
                    _ => None,
                }
            }
            fn control_symbols(&self, state: u32, out: &mut SmallVec<[Self::Symbol; 4]>) {
                if state == 1 {
                    out.push(CTRL_SYM);
                }
            }
            fn scope_state(&self, _scope: u32, local: u32) -> Option<u32> {
                Some(local)
            }
            fn goto_target(&self, _scope: u32, goto_from: u32, nt: u32) -> Option<(u32, bool)> {
                if goto_from == 0 && nt == 0 {
                    Some((1, false))
                } else {
                    None
                }
            }
            fn state_count_hint(&self) -> usize {
                16
            }
        }

        let provider = NoAcceptProvider;
        let stack = ParserGSS::from_single_stack(vec![0, 10], TerminalsDisallowed::new());
        assert!(
            !stacks_finished_with_provider(&provider, &stack, EOF),
            "reduce -> control -> non-accepting state must reject"
        );
    }

    #[test]
    fn provider_scoped_composition_eof_control_closure_accepts() {
        let parent_table = build_test_table(
            2,
            1,
            &[&[(0, Action::Shift(1, false))], &[(EOF, Action::Accept)]],
            &[&[], &[]],
        );
        let child_table = build_test_table(
            1,
            1,
            &[&[(EOF, Action::Accept)]],
            &[&[]],
        );
        let tables: [&GLRTable; 2] = [&parent_table, &child_table];
        let links = [ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: 0,
            child_component: 1,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: true,
        }];
        let provider = DisjointComponentActionProvider::new(&tables, &links).unwrap();

        let child_start_scoped = provider.scoped_state(1, 0).unwrap();
        let stack = ParserGSS::from_single_stack(
            vec![provider.scoped_state(0, 0).unwrap(), child_start_scoped],
            TerminalsDisallowed::new(),
        );
        let parent_eof = ScopedParserSymbol::Terminal { component: 0, terminal: EOF };
        assert!(stacks_finished_with_provider(&provider, &stack, parent_eof));
    }

    #[test]
    fn advance_stacks_materializes_single_concrete_path_for_split() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            6,
            1,
            &[
                &[],
                &[],
                &[(token, Action::Split {
                    shift: Some((3, false)),
                    reduces: vec![(nt, 1)],
                    accept: false,
                })],
                &[],
                &[(token, Action::Shift(5, false))],
                &[],
            ],
            &[
                &[(nt, (4, false))],
                &[],
                &[],
                &[],
                &[],
                &[],
            ],
        );

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1], acc.clone()),
            (vec![0, 2], acc.clone()),
        ])
        .popn(1)
        .push(2);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 2, 3], acc.clone()),
            (vec![0, 4, 5], acc),
        ]);

        let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        let mut expected_stacks = expected.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
        actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(actual_stacks, expected_stacks);
    }

    #[test]
    fn disjoint_top_terminal_admission_preserves_earlier_success() {
        let token_a = 0;
        let token_b = 1;
        let nt = 0;
        let mut table = build_test_table(
            6,
            2,
            &[
                &[],
                &[],
                &[(token_a, Action::Reduce(nt, 1))],
                &[(token_a, Action::Shift(5, false))],
                &[],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_stacks(&[
            (vec![0, 2], TerminalsDisallowed::new()),
            (vec![1, 4], TerminalsDisallowed::new()),
        ]);

        assert_eq!(
            stack_may_advance_disjoint_top_terminals_bounded(
                &table,
                &stack,
                &[(2, token_a), (4, token_b)],
            ),
            Some(true),
        );
    }

    #[test]
    fn disjoint_top_terminal_advance_matches_pointwise_union() {
        let token_a = 0;
        let token_b = 1;
        let nt_a = 0;
        let nt_b = 1;
        let table = build_test_table(
            8,
            2,
            &[
                &[],
                &[],
                &[(token_a, Action::Reduce(nt_a, 1))],
                &[(token_a, Action::Shift(6, false))],
                &[(token_b, Action::Reduce(nt_b, 1))],
                &[(token_b, Action::Shift(7, false))],
                &[],
                &[],
            ],
            &[
                &[(nt_a, (3, false))],
                &[(nt_b, (5, false))],
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
            ],
        );
        let acc = TerminalsDisallowed::new();
        let stack = ParserGSS::from_stacks(&[
            (vec![0, 2], acc.clone()),
            (vec![1, 4], acc),
        ]);

        let actual = advance_stacks_disjoint_top_terminals_bounded(
            &table,
            &stack,
            &[(2, token_a), (4, token_b)],
        )
        .expect("bounded disjoint advance should apply");
        let left = advance_stacks(&table, &stack.isolate(Some(2)), token_a);
        let right = advance_stacks(&table, &stack.isolate(Some(4)), token_b);
        let expected = left.merge(&right);
        assert!(
            actual
                .semantically_eq(&expected, 32)
                .expect("small exact stack languages should fit"),
            "actual={:?} expected={:?}",
            actual.to_stacks(32),
            expected.to_stacks(32),
        );
    }

    #[test]
    fn indexed_guarded_vstack_matches_linear_guarded_vstack() {
        let token = 0;
        let mut table = build_test_table(
            1,
            1,
            &[&[(
                token,
                Action::GuardedStackShifts(vec![
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![10, 20],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![1],
                            },
                        ],
                        pop: 3,
                        pushes: vec![50],
                    },
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![10],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![2],
                            },
                        ],
                        pop: 3,
                        pushes: vec![51],
                    },
                    GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![10, 20],
                        }],
                        pop: 2,
                        pushes: vec![52],
                    },
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![30],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![1],
                            },
                        ],
                        pop: 3,
                        pushes: vec![53],
                    },
                ]),
            )]],
            &[&[]],
        );
        table.rebuild_guarded_shift_index();

        let shifts = match table.action(0, token) {
            Some(Action::GuardedStackShifts(shifts)) => shifts,
            other => panic!("expected guarded stack shifts, got {other:?}"),
        };
        let index = table
            .guarded_shift_index(0, token)
            .expect("expected guarded shift index");

        let stack_a = ParserGSS::from_single_stack(vec![0, 1, 10, 99], TerminalsDisallowed::new());
        let stack_b = ParserGSS::from_single_stack(vec![0, 2, 10, 99], TerminalsDisallowed::new());

        for stack in [&stack_a, &stack_b] {
            let vstack = stack.try_virtual_stack().expect("expected virtual stack");
            let mut indexed = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, Some(index)).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
            let mut linear = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, None).to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
            indexed.sort_by(|left, right| left.0.cmp(&right.0));
            linear.sort_by(|left, right| left.0.cmp(&right.0));
            assert_eq!(indexed, linear);
        }
    }
}

/// Return exactly the candidate terminals on which this parser stack can
/// advance. Unlike repeated `stack_may_advance_on_any` calls, ExactSimulation
/// explores the reduction closure once while propagating all candidate bits.
pub fn stack_admissible_terminals(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> BitSet {
    let control_closed = (!table.control_terminals.is_empty())
        .then(|| control_closure(table, stack));
    let stack = control_closed.as_ref().unwrap_or(stack);
    if table.admission_policy == AdmissionPolicy::ExactSimulation {
        return exact_admission_admissible_terminals(table, stack, terminals);
    }

    let mut admitted = BitSet::new(terminals.len());
    if table.advance.len() == table.num_states as usize {
        // RowPresenceExact admission is a union of the live top-state rows.
        // Iterate those sparse rows once instead of rescanning every top state
        // for every candidate terminal.
        for state in stack.peek_values() {
            let Some(row) = table.advance.get(state as usize) else {
                continue;
            };
            for bit in row.iter_ones() {
                if bit < terminals.len() && terminals.contains(bit) {
                    admitted.set(bit);
                }
            }
        }
    } else {
        // Compatibility fallback for hand-built tables without admission rows.
        let top_states = stack.peek_values();
        for bit in terminals.iter_ones() {
            let Some(terminal) = exact_admission_terminal_from_bit(table, bit) else {
                continue;
            };
            if top_states
                .iter()
                .any(|&state| table.advance_row_allows(state, terminal))
            {
                admitted.set(bit);
            }
        }
    }
    if assert_row_presence_exact_enabled() {
        let simulated = exact_admission_admissible_terminals(table, stack, terminals);
        assert_eq!(admitted, simulated, "RowPresenceExact admissible-set mismatch");
    }
    admitted
}

/// Precise predicate for whether this parser stack can advance on any terminal in
/// `terminals`.
///
/// Returns `true` if and only if at least one current parser path can definitely
/// advance on one of the given terminals. Returns `false` if no current parser
/// path can advance on any of them.
///
/// For `RowPresenceExact`, `advance` was captured from the precise parser row
/// before stack-effect lowering. Guarded stack shifts therefore select the exact
/// execution effect; they do not weaken admission. `ExactSimulation` tables use
/// reduction/guard simulation instead.
///
/// TODO: Rename this eventually, e.g. to `stack_can_advance_on_any`. The current
/// `may_advance` name sounds like a speculative approximation, but this is an
/// exact applicability predicate.


/// Return one exact mask-only reduction factor for a concrete parser frontier.
///
/// Let `R` be the set of exactly-admissible lookaheads whose current top-state
/// action is one common `Reduce(nt, len)`. We return the post-reduction frontier
/// only when its exact admissible-lookahead set is *exactly* `R`. Therefore the
/// new frontier denotes precisely a subset of the original future language:
/// every lookahead it can consume was forced to take this reduction first in
/// the original frontier, while lookaheads belonging to another action class
/// (notably a scoped `Skip`) remain exclusively on the original branch.
pub fn lookahead_reduction_factor(
    table: &GLRTable,
    stack: &ParserGSS,
) -> Option<(ParserGSS, SmallVec<[TerminalID; 4]>)> {
    stack.try_virtual_stack()?;
    let top = stack.single_exclusive_top_value()?;
    let all_terminals = BitSet::all(table.num_terminals as usize + 1);
    let before = stack_admissible_terminals(table, stack, &all_terminals);
    if before.is_empty() {
        return None;
    }

    let mut reduction = None::<(u32, u32)>;
    let mut reduction_terminals = BitSet::new(all_terminals.len());
    for bit in before.iter_ones() {
        let terminal = exact_admission_terminal_from_bit(table, bit)?;
        let Some(Action::Reduce(nt, len)) = table.action(top, terminal) else {
            continue;
        };
        match reduction {
            None => reduction = Some((*nt, *len)),
            Some(expected) if expected == (*nt, *len) => {}
            Some(_) => return None,
        }
        reduction_terminals.set(bit);
    }
    let (nt, rhs_len) = reduction?;
    if reduction_terminals.is_empty() || reduction_terminals == before {
        return None;
    }

    let mut next = ParserGSS::empty();
    for (base, target, is_replace) in
        reduce_branches_from_isolated(table, stack, nt, rhs_len as usize)
    {
        let branch = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        merge_into(&mut next, branch);
    }
    if next.is_empty() {
        return None;
    }

    let after = stack_admissible_terminals(table, &next, &all_terminals);
    // Every terminal in the reduction class must remain admissible after the
    // reduction. Any additional post-reduction terminals are guarded away on
    // the mask-only factor with exact delayed terminal exclusions.
    if reduction_terminals
        .iter_ones()
        .any(|bit| !after.contains(bit))
    {
        return None;
    }
    let mut blocked = SmallVec::<[TerminalID; 4]>::new();
    for bit in after.iter_ones() {
        if reduction_terminals.contains(bit) {
            continue;
        }
        // EOF cannot be represented as a lexer-terminal exclusion. Decline
        // rather than weakening the certificate.
        if bit == table.num_terminals as usize || blocked.len() == blocked.capacity() {
            return None;
        }
        blocked.push(bit as TerminalID);
    }

    if std::env::var_os("GLRMASK_DEBUG_LOOKAHEAD_FACTOR").is_some() {
        eprintln!(
            "[glrmask/debug][lookahead_factor] policy={:?} top={} reduce=({},{}) reduce_terms={:?} after={:?} blocked={:?} next={:?}",
            table.admission_policy,
            top,
            nt,
            rhs_len,
            reduction_terminals.iter_ones().collect::<Vec<_>>(),
            after.iter_ones().collect::<Vec<_>>(),
            blocked,
            next.to_stacks(64),
        );
    }
    Some((next, blocked))
}

/// Cheap, exact-under-approximate form of [`lookahead_reduction_factor`].
///
/// This deliberately avoids `ExactSimulation` admission closure.  If every
/// terminal in one current-row class has the same first action
/// `Reduce(nt, len)`, then applying that reduction without consuming input is
/// semantics-preserving *for that class*.  The returned frontier is guarded
/// against every terminal appearing in a post-reduction top row outside the
/// reduction class.  Thus any terminal that the added frontier can actually
/// consume is forced to belong to the class for which the original parser's
/// first action was exactly this reduction.
///
/// `advance` rows may over-approximate exact admission on composed tables. That
/// only makes the guard stronger: we can block a terminal which was impossible
/// anyway without adding language.  We decline when the conservative guard
/// would be too large for the inline accumulator or would need to block EOF.
pub fn lookahead_reduction_factor_row_subset(
    table: &GLRTable,
    stack: &ParserGSS,
) -> Option<(ParserGSS, BitSet, BitSet)> {
    stack.try_virtual_stack()?;
    let top = stack.single_exclusive_top_value()?;
    let top_row = table.advance_row(top)?;
    let mut reduction = None::<(u32, u32)>;
    let mut reduction_terminals = BitSet::new(table.num_terminals as usize + 1);
    let mut saw_exception = false;

    for bit in top_row.iter_ones() {
        let terminal = exact_admission_terminal_from_bit(table, bit)?;
        match table.action(top, terminal) {
            Some(Action::Reduce(nt, len)) => match reduction {
                None => {
                    reduction = Some((*nt, *len));
                    reduction_terminals.set(bit);
                }
                Some(expected) if expected == (*nt, *len) => {
                    reduction_terminals.set(bit);
                }
                Some(_) => return None,
            },
            Some(_) => saw_exception = true,
            None => return None,
        }
    }
    let (nt, rhs_len) = reduction?;
    if reduction_terminals.is_empty() || !saw_exception {
        return None;
    }

    let mut next = ParserGSS::empty();
    for (base, target, is_replace) in
        reduce_branches_from_isolated(table, stack, nt, rhs_len as usize)
    {
        let branch = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        merge_into(&mut next, branch);
    }
    if next.is_empty() {
        return None;
    }

    let mut after_terminals = BitSet::new(table.num_terminals as usize + 1);
    for state in next.peek_values() {
        let row = table.advance_row(state)?;
        for bit in row.iter_ones() {
            after_terminals.set(bit);
        }
    }

    if std::env::var_os("GLRMASK_VALIDATE_FAST_LOOKAHEAD_FACTOR").is_some() {
        if let Some((exact_next, exact_blocked)) = lookahead_reduction_factor(table, stack) {
            let exact_blocked = exact_blocked.into_iter().collect::<BTreeSet<_>>();
            let fast_blocked = after_terminals
                .difference(&reduction_terminals)
                .iter_ones()
                .filter_map(|bit| (bit < table.num_terminals as usize).then_some(bit as TerminalID))
                .collect::<BTreeSet<_>>();
            assert_eq!(next, exact_next, "fast lookahead factor changed reduction frontier");
            assert!(
                exact_blocked.is_subset(&fast_blocked),
                "fast lookahead factor failed to conservatively guard exact blockers: exact={exact_blocked:?} fast={fast_blocked:?}",
            );
        }
    }

    Some((next, reduction_terminals, after_terminals))
}

/// Eagerly apply parser reductions that are provably independent of the next
/// lookahead terminal for this concrete parser frontier.
///
/// A reduction is applied only when:
/// 1. the frontier is one concrete stack;
/// 2. every exactly-admissible next terminal (including EOF) has the same
///    `Reduce(nt, len)` as its first parser action; and
/// 3. after applying that one reduction without consuming input, the exact set
///    of admissible next terminals is unchanged.
///
/// Under those conditions the reduction commutes with every possible next
/// terminal: the original parser must take that same reduction before it can
/// consume any valid lookahead, and the post-reduction frontier introduces no
/// new lookaheads. Repeating the certificate therefore preserves the complete
/// future terminal language while exposing lookahead-independent reductions
/// earlier to runtimes that reuse precompiled parser summaries.
pub fn normalize_lookahead_invariant_reductions(
    table: &GLRTable,
    stack: &ParserGSS,
) -> ParserGSS {
    const MAX_EAGER_REDUCTIONS: usize = 64;

    let mut current = stack.clone();
    let all_terminals = BitSet::all(table.num_terminals as usize + 1);

    for _ in 0..MAX_EAGER_REDUCTIONS {
        // Keep this optimization deliberately narrow. A single concrete stack
        // covers the subgrammar-return cliff without having to prove that a
        // reduction commutes independently across ambiguous GSS branches.
        if current.try_virtual_stack().is_none() {
            break;
        }
        let Some(top) = current.single_exclusive_top_value() else {
            break;
        };

        let before = stack_admissible_terminals(table, &current, &all_terminals);
        if before.is_empty() {
            break;
        }

        let mut common_reduce = None::<(u32, u32)>;
        let mut certified = true;
        for bit in before.iter_ones() {
            let Some(terminal) = exact_admission_terminal_from_bit(table, bit) else {
                certified = false;
                break;
            };
            let Some(Action::Reduce(nt, len)) = table.action(top, terminal) else {
                certified = false;
                break;
            };
            match common_reduce {
                None => common_reduce = Some((*nt, *len)),
                Some(expected) if expected == (*nt, *len) => {}
                Some(_) => {
                    certified = false;
                    break;
                }
            }
        }
        if !certified {
            break;
        }
        let Some((nt, rhs_len)) = common_reduce else {
            break;
        };

        let mut next = ParserGSS::empty();
        for (base, target, is_replace) in
            reduce_branches_from_isolated(table, &current, nt, rhs_len as usize)
        {
            let branch = if is_replace {
                base.popn(1).push(target)
            } else {
                base.push(target)
            };
            merge_into(&mut next, branch);
        }
        if next.is_empty() || next == current {
            break;
        }

        let after = stack_admissible_terminals(table, &next, &all_terminals);
        if after != before {
            break;
        }
        current = next;
    }

    current
}

pub fn stack_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    let control_closed = (!table.control_terminals.is_empty())
        .then(|| control_closure(table, stack));
    let stack = control_closed.as_ref().unwrap_or(stack);
    stack_may_advance_on_any_control_closed(table, stack, terminals)
}

/// Exact set admission for a frontier already closed under linker controls.
pub fn stack_may_advance_on_any_control_closed(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    if table.admission_policy == AdmissionPolicy::ExactSimulation {
        return exact_admission_may_advance_on_any(table, stack, terminals);
    }

    let admitted = stack
        .peek_values()
        .into_iter()
        .any(|state| table.advance_row_intersects(state, terminals));
    if assert_row_presence_exact_enabled() {
        let simulated = exact_admission_may_advance_on_any(table, stack, terminals);
        assert_eq!(
            admitted,
            simulated,
            "RowPresenceExact mismatch for terminal set {:?}: construction={:?} stacks={:?}",
            terminals.iter_ones().collect::<Vec<_>>(),
            table.construction,
            stack.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
        );
    }
    admitted
}

pub fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    let control_closed = (!table.control_terminals.is_empty())
        .then(|| control_closure(table, stack));
    let stack = control_closed.as_ref().unwrap_or(stack);

    stacks_finished_control_closed(table, stack)
}

/// Completion predicate for a frontier already closed under linker controls.
///
/// Reuses the exact admission simulation on `EOF` (the same
/// reduction-feasibility closure `stack_may_advance_on_control_closed` uses) in
/// accept-only `Completion` mode instead of treating *any* EOF action as
/// completion. A nullable skip reduction leaves an EOF `Reduce` on a trivia
/// state; that reduce is only a genuine completion when its
/// GSS-predecessor-feasible reduction chain reaches `Action::Accept` (or an
/// accepting `Split`). Requiring an immediate `Accept` would reject valid
/// reducible completions, so the full closure is followed. A bare EOF
/// shift/skip/replace is never completion.
pub fn stacks_finished_control_closed(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    exact_admission_completes_on_eof(table, stack)
}
