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
        embedded_start: Default::default(),
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
                    let Some((goto_target, true)) = table.goto_target(*predecessor, nt) else {
                        return Ok(None);
                    };
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
        result.push((v, popped));
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

fn advance_provider_traversal<P: ParserActionProvider>(
    provider: &P,
    mut closure: ParserGSS,
    symbol: P::Symbol,
    mode: ProviderAdvanceMode,
) -> ProviderAdvanceResult {
    let mut shifted = ParserGSS::empty();
    let mut accepted = false;
    let mut visited = FxHashSet::<u32>::default();
    let mut key_interner = GssSemanticKeyInterner::<u32, TerminalsDisallowed>::new();

    loop {
        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            let Some(provided) = provider.action(state, symbol) else {
                continue;
            };
            let isolated = closure.isolate(Some(state));

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
                                    let key = key_interner.key(&branch);
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
mod tests ;

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
