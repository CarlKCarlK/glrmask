use super::*;

enum Effect { PopOne, PushZero, Toggle }
struct Relations(Effect);

impl ParserActionProvider for Relations {
    type Symbol = u32;
    fn action(&self, _state: u32, _symbol: u32) -> Option<ProvidedAction<'_>> { unreachable!() }
    fn scope_state(&self, _scope: u32, _local: u32) -> Option<u32> { unreachable!() }
    fn goto_target(&self, _scope: u32, _from: u32, _nt: u32) -> Option<(u32, bool)> { unreachable!() }
    fn state_count_hint(&self) -> usize { 2 }
    fn control_symbols(&self, top: u32, out: &mut SmallVec<[u32; 4]>) {
        if !matches!(self.0, Effect::PopOne) || top == 1 { out.push(0); }
    }
    fn advance_relation(&self, stack: &ParserGSS, _terminal: u32) -> Option<ParserGSS> {
        Some(match self.0 {
            Effect::PopOne => stack.isolate(Some(1)).popn(1),
            Effect::PushZero => stack.clone().push(0),
            Effect::Toggle => stack.isolate(Some(0)).popn(1).push(1)
                .merge(&stack.isolate(Some(1)).popn(1).push(0)),
        })
    }
}

#[test]
fn finite_closure_can_require_more_rounds_than_four_times_the_state_count() {
    let mut word = vec![0]; word.extend(std::iter::repeat_n(1, 128));
    let input = ParserGSS::from_single_stack(word, TerminalsDisallowed::new());
    let closed = close_provider_control_stacks(&Relations(Effect::PopOne), &input);
    let stacks = closed.to_stacks(130).unwrap();
    assert_eq!(stacks.len(), 129);
    for length in 1..=129 {
        assert!(stacks.iter().any(|(stack, _)| stack.len() == length
            && stack[0] == 0 && stack[1..].iter().all(|&state| state == 1)));
    }
}

#[test]
fn finite_control_cycle_stops_at_its_exact_union_fixed_point() {
    let input = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let closed = close_provider_control_stacks(&Relations(Effect::Toggle), &input);
    let mut stacks = closed.to_stacks(3).unwrap().into_iter().map(|(stack, _)| stack).collect::<Vec<_>>();
    stacks.sort(); assert_eq!(stacks, vec![vec![0], vec![1]]);
}

#[test]
fn exhausted_closure_budget_is_an_error_not_a_partial_success() {
    let input = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
    let error = glrmask_invariant::__private::catch_compilation_resource_limit(|| {
        close_provider_control_stacks_budgeted(&Relations(Effect::PushZero), &input, 64)
    }).unwrap_err();
    assert!(error.contains("work budget") && error.contains("no partial closure"), "{error}");
}
