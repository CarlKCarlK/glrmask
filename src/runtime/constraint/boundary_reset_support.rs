//! Bounded, rejection-only proof for token-end-only reset configurations.
//!
//! The proof explores zero-width controls without changing the live parser.
//! It uses the existing VirtualStack operations and declines on unsupported
//! actions, hidden-floor reads, cycles, or resource exhaustion. This module
//! does not execute vocabulary bytes and does not cache parser answers.
use super::*;
use crate::compiler::glr::parser::{ParserActionProvider, ProvidedActionRef};
use crate::ds::leveled_gss::VirtualStack;

type ControlCursor = VirtualStack<u32, ()>;
enum ControlAdvance {
    Unknown,
    Dead,
    Unchanged,
    Advanced(ControlCursor),
}

fn no_controls<P: ParserActionProvider>(provider: &P, cursor: &ControlCursor) -> bool {
    let Some(&top) = cursor.top() else { return false; };
    let mut controls = SmallVec::<[P::Symbol; 4]>::new();
    provider.control_symbols(top, &mut controls);
    controls.is_empty()
}

fn execute_impl<P: ParserActionProvider, const REQUIRE_CLOSED_OUTPUT: bool>(
    provider: &P, input: &ControlCursor, symbol: P::Symbol,
) -> ControlAdvance {
    // Most wide lexical-future candidates have NO action. Do not clone even
    // the small cursor for those negative lookups, and reuse the first lookup.
    let Some(&top) = input.top() else { return ControlAdvance::Unknown; };
    let Some(provided) = provider.action(top, symbol) else { return ControlAdvance::Dead; };
    if !provided.extra_stack_shifts.is_empty() { return ControlAdvance::Unknown; }
    let mut first = Some(provided);
    let mut cursor = input.clone();
    let mut changed = false;
    for _ in 0..64 {
        let Some(&top) = cursor.top() else { return ControlAdvance::Unknown; };
        let Some(provided) = first.take().or_else(|| provider.action(top, symbol)) else { return ControlAdvance::Dead; };
        if !provided.extra_stack_shifts.is_empty() { return ControlAdvance::Unknown; }
        match provided.action {
            ProvidedActionRef::Identity => {},
            ProvidedActionRef::Call { parent_target, child_start, replace } => {
                if replace && cursor.pop(1) != 0 { return ControlAdvance::Unknown; }
                cursor.push(parent_target); cursor.push(child_start); changed = true;
            }
            ProvidedActionRef::Return { pop } => {
                if pop as usize > cursor.len() { return ControlAdvance::Unknown; }
                if cursor.pop(pop as usize) != 0 { return ControlAdvance::Unknown; }
                changed |= pop != 0;
            }
            ProvidedActionRef::Local { scope, action } => match action {
                Action::Skip => {},
                Action::Shift(target, replace) => {
                    let Some(target) = provider.scope_state(scope, *target) else { return ControlAdvance::Dead; };
                    if *replace {
                        if !cursor.replace_top(target) { return ControlAdvance::Unknown; }
                    } else { cursor.push(target); }
                    changed = true;
                }
                Action::Reduce(nonterminal, count) => {
                    if *count as usize >= cursor.len() { return ControlAdvance::Unknown; }
                    if cursor.pop(*count as usize) != 0 { return ControlAdvance::Unknown; }
                    let Some(&predecessor) = cursor.top() else { return ControlAdvance::Unknown; };
                    let Some((target, replace)) = provider.goto_target(provided.reduction_scope, predecessor, *nonterminal)
                        else { return ControlAdvance::Dead; };
                    if replace {
                        if !cursor.replace_top(target) { return ControlAdvance::Unknown; }
                    } else { cursor.push(target); }
                    changed = true;
                    continue; // reduction has NOT consumed the lookahead
                }
                _ => return ControlAdvance::Unknown,
            },
        }
        // The reference unions all zero-width control successors after the
        // consuming action. A lone cursor may stand for the result only if
        // that closure adds nothing, without assuming bounded C is idempotent.
        if REQUIRE_CLOSED_OUTPUT && !no_controls(provider, &cursor) { return ControlAdvance::Unknown; }
        return if changed { ControlAdvance::Advanced(cursor) } else { ControlAdvance::Unchanged };
    }
    ControlAdvance::Unknown
}


/// A rejection-only proof must never use a partial control search. Budget
/// exhaustion is recorded separately from the interpreter's dead result.
struct ControlQueryBudget<'a, P> {
    inner: &'a P,
    remaining: std::cell::Cell<usize>,
    exhausted: std::cell::Cell<bool>,
}
impl<P> ControlQueryBudget<'_, P> {
    fn spend(&self) -> bool {
        let n=self.remaining.get();
        if n==0 {self.exhausted.set(true);false} else {self.remaining.set(n-1);true}
    }
}
impl<P: ParserActionProvider> ParserActionProvider for ControlQueryBudget<'_, P> {
    type Symbol=P::Symbol;
    fn action(&self,state:u32,symbol:P::Symbol)->Option<crate::compiler::glr::parser::ProvidedAction<'_>> {
        self.spend().then(||self.inner.action(state,symbol)).flatten()
    }
    fn scope_state(&self,scope:u32,state:u32)->Option<u32> {
        self.spend().then(||self.inner.scope_state(scope,state)).flatten()
    }
    fn goto_target(&self,scope:u32,from:u32,nt:u32)->Option<(u32,bool)> {
        self.spend().then(||self.inner.goto_target(scope,from,nt)).flatten()
    }
    fn control_symbols(&self,state:u32,out:&mut SmallVec<[P::Symbol;4]>) {
        if self.spend(){self.inner.control_symbols(state,out);}
    }
    fn state_count_hint(&self)->usize {self.inner.state_count_hint()}
}

/// Enumerate ONLY a small zero-width control closure, through the existing
/// exact cursor interpreter. No visible terminal is consumed and the caller's
/// parser is never changed. Success requires draining every reachable branch;
/// cycles, hidden-floor reads, ambiguity or any exhausted bound decline.
/// The resulting tops conservatively cover the reference's operationally
/// bounded closure. They are used solely as a necessary lexical support set.
fn bounded_control_top_states<P: ParserActionProvider>(
    provider:&P,input:&ControlCursor,query_budget:usize,
)->Option<SmallVec<[u32;8]>> {
    const MAX_FRONTIERS:usize=16;
    const MAX_PREFIX:usize=256;
    if input.len()>MAX_PREFIX{return None;}
    let provider=ControlQueryBudget {inner:provider,remaining:std::cell::Cell::new(query_budget),exhausted:std::cell::Cell::new(false)};
    let mut work=SmallVec::<[ControlCursor;4]>::new();
    work.push(input.clone());
    let mut tops=SmallVec::<[u32;8]>::new();
    let mut expanded=0;
    while let Some(cursor)=work.pop() {
        if expanded>=MAX_FRONTIERS || cursor.len()>MAX_PREFIX{return None;}
        expanded+=1;
        let top=*cursor.top()?;
        if !tops.contains(&top){tops.push(top);}
        let mut symbols=SmallVec::<[P::Symbol;4]>::new();
        provider.control_symbols(top,&mut symbols);
        if provider.exhausted.get() || symbols.len()>8{return None;}
        let mut seen=SmallVec::<[P::Symbol;4]>::new();
        for symbol in symbols {
            if seen.contains(&symbol){continue;}
            seen.push(symbol);
            let result=execute_impl::<_,false>(&provider,&cursor,symbol);
            if provider.exhausted.get(){return None;}
            match result {
                ControlAdvance::Unknown=>return None,
                ControlAdvance::Dead|ControlAdvance::Unchanged=>{},
                ControlAdvance::Advanced(next)=>{
                    if work.len()>=MAX_FRONTIERS{return None;}
                    work.push(next);
                }
            }
        }
    }
    Some(tops)
}

impl Constraint {
    fn bounded_mask_control_tops_cursor(
        &self,input:&ControlCursor,
    )->Option<SmallVec<[u32;8]>> {
        // Decline before cloning a long frontier or scanning many links.
        if input.len()>256{return None;}
        let layout=self.recursive_parser_layout_ref()?;
        if layout.leaves.len()>8 || layout.links.len()>8{return None;}
        for leaf in &layout.leaves {
            if leaf.component_path.len()>16{return None;}
            if !self.constraint_at_recursive_component_path(&leaf.component_path)?.table.control_terminals.is_empty(){return None;}
        }
        let tables=RecursiveSegmentedParserTables{root:self,layout};
        let provider=DisjointComponentActionProvider::with_state_offsets(&tables,&layout.links,&layout.leaf_state_offsets).ok()?;
        bounded_control_top_states(&provider,input,128)
    }

    /// Prove that after the complete bounded zero-width control closure, this
    /// leaf has no ordinary terminal it can consume. Foreign-owner tops are
    /// intentionally ignored: canonical reset routing partitions those into
    /// their own lexer reset branches. Any incomplete proof declines.
    pub(crate) fn bounded_mask_same_leaf_support_empty_cursor(
        &self,input:&ControlCursor,leaf_index:usize,
    )->Option<bool> {
        let leaf=self.recursive_leaf_constraint(leaf_index)?;
        // Bound necessary-support bitset scans as well as control traversal.
        if leaf.table.num_terminals > 16_384 { return None; }
        let tops=self.bounded_mask_control_tops_cursor(input)?;
        let mut saw_same_leaf=false;
        for top in tops {
            let (owner,state)=self.recursive_parser_leaf_state(top)?;
            if owner!=leaf_index {continue;}
            saw_same_leaf=true;
            let row=leaf.table.advance_row(state)?;
            if row.iter_ones().any(|terminal|terminal<leaf.table.num_terminals as usize){
                return Some(false);
            }
        }
        if !saw_same_leaf{return None;}
        if leaf.ignore_terminal.is_some() || !leaf.table.skip_terminals.is_empty(){
            return Some(false);
        }
        Some(true)
    }

}

#[cfg(test)]
mod bounded_control_support_tests {
    use super::*;
    use crate::compiler::glr::parser::ProvidedAction;
    use crate::ds::leveled_gss::LeveledGSS;
    struct Controls {mode:u8,branch:Action}
    impl ParserActionProvider for Controls {
        type Symbol=u32;
        fn action(&self,state:u32,symbol:u32)->Option<ProvidedAction<'_>> {
            let action=match (self.mode,state,symbol) {
                (0,3,10)=>ProvidedActionRef::Call{parent_target:8,child_start:20,replace:false},
                (0,3,12)=>ProvidedActionRef::Call{parent_target:9,child_start:21,replace:false},
                (0,20,11)|(0,21,11)=>ProvidedActionRef::Return{pop:1},
                (1,3,10)=>ProvidedActionRef::Call{parent_target:3,child_start:3,replace:false},
                (2,3,10)=>ProvidedActionRef::Return{pop:3},
                (3,3,10)=>ProvidedActionRef::Local{scope:0,action:&self.branch},
                (4,3,10)=>ProvidedActionRef::Identity,
                _=>return None,
            };
            Some(ProvidedAction{action,reduction_scope:0,extra_stack_shifts:SmallVec::new()})
        }
        fn scope_state(&self,_:u32,state:u32)->Option<u32>{Some(state)}
        fn goto_target(&self,_:u32,_:u32,_:u32)->Option<(u32,bool)>{None}
        fn state_count_hint(&self)->usize{32}
        fn control_symbols(&self,state:u32,out:&mut SmallVec<[u32;4]>){
            if state==3{out.push(10);if self.mode==0{out.push(12);out.push(10);}}
            if self.mode==0 && (state==20 || state==21){out.push(11);}
        }
    }
    #[test]
    fn complete_control_top_support_matches_full_gss_closure_with_foreign_floors() {
        let p=Controls{mode:0,branch:Action::ReplaceShifts(vec![8,9].into())};
        for paths in [vec![vec![1,3]],vec![vec![4,1,3],vec![5,1,3]]] {
            let unit=LeveledGSS::from_stacks(&paths.iter().map(|s|(s.clone(),())).collect::<Vec<_>>());
            let old=unit.to_stacks(64).unwrap();
            let mut got=bounded_control_top_states(&p,&unit.try_virtual_stack().unwrap(),128).unwrap().to_vec();got.sort_unstable();
            let mut expected=close_provider_control_stacks(&p,&unit.apply(|_|TerminalsDisallowed::new())).peek_values().to_vec();expected.sort_unstable();
            assert_eq!(got,expected);assert_eq!(got,vec![3,8,9,20,21]);
            assert_eq!(unit.to_stacks(64).unwrap(),old);
        }
    }
    #[test]
    fn exhausted_budget_never_returns_partial_support() {
        let p=Controls{mode:0,branch:Action::Skip};
        let c=LeveledGSS::from_single_stack(vec![1,3],()).try_virtual_stack().unwrap();
        for budget in [0,1,2,3,4]{assert!(bounded_control_top_states(&p,&c,budget).is_none());}
        assert!(bounded_control_top_states(&p,&c,128).is_some());
    }
    #[test]
    fn growing_controls_ambiguous_actions_and_hidden_floors_decline() {
        let c=LeveledGSS::from_single_stack(vec![1,3],()).try_virtual_stack().unwrap();
        for mode in [1,2,3]{let p=Controls{mode,branch:Action::ReplaceShifts(vec![8,9].into())};assert!(bounded_control_top_states(&p,&c,128).is_none());}
        let hidden=LeveledGSS::from_stacks(&[(vec![4,1,3],()),(vec![5,1,3],())]).try_virtual_stack().unwrap();
        let p=Controls{mode:2,branch:Action::Skip};assert!(bounded_control_top_states(&p,&hidden,128).is_none());
    }
    #[test]
    fn identity_controls_and_empty_control_sets_terminate_without_expansion() {
        let p=Controls{mode:4,branch:Action::Skip};
        for top in [3,7]{let c=LeveledGSS::from_single_stack(vec![1,top],()).try_virtual_stack().unwrap();assert_eq!(bounded_control_top_states(&p,&c,128).unwrap().as_slice(),&[top]);}
        let deep=LeveledGSS::from_single_stack((0..257).collect(),()).try_virtual_stack().unwrap();
        assert!(bounded_control_top_states(&p,&deep,128).is_none());
    }
}
