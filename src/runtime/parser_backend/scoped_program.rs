//! Borrowed stack-coordinate views. Component graphs and their derived local
//! domain/runtime data remain immutable and shared across every link.
use std::{ops::ControlFlow, sync::Arc};
use super::{CommitTemplateDfas, Constraint, TemplateDomain, TopAdmission, DomainProbe};
use crate::compiler::glr::{labels::DEFAULT_LABEL, parser::ParserGSS};
use crate::runtime::FastCommitTemplateDfas;

#[derive(Debug, Clone)]
pub(crate) struct ScopedProgram {
    pub(crate) source: Arc<CommitTemplateDfas>,
    pub(crate) domain: Arc<TemplateDomain>,
    pub(crate) fast: Option<Arc<FastCommitTemplateDfas>>,
    pub(crate) offset: u32,
    pub(crate) symbols: u32,
    pub(crate) guard_owner: bool,
    /// Explicit CALL crossing, applied after the local relation succeeds.
    pub(crate) append_push: Option<u32>,
}

impl ScopedProgram {
    pub(crate) fn prepare(source: Arc<CommitTemplateDfas>, symbols: u32) -> crate::Result<Self> {
        let validated = crate::runtime::commit::template_prepare::TemplatePreparation::new(&source)
            .map_err(crate::Error::Compilation)?;
        validated.validate_alphabet(symbols).map_err(crate::Error::Compilation)?;
        let domain = Arc::new(TemplateDomain::from_validated(&validated)
            .map_err(crate::Error::Compilation)?);
        let fast = Some(Arc::new(FastCommitTemplateDfas::from_prepared(&validated)));
        Ok(Self { source, domain, fast, offset: 0, symbols, guard_owner: true, append_push: None })
    }

    pub(crate) fn terminal(component: &Constraint, terminal: usize) -> Self {
        let parser = component.template_parser.as_ref().expect("native component");
        Self { source: Arc::clone(component.template_dfas_by_terminal[terminal].as_ref().unwrap()),
            domain: Arc::clone(&parser.domains[terminal]),
            fast: component.fast_template_dfas_by_terminal.get(terminal).cloned().flatten(),
            offset: 0, symbols: parser.state_count, guard_owner: true, append_push: None }
    }

    pub(crate) fn completion(component: &Constraint) -> Self {
        let parser = component.template_parser.as_ref().expect("native component");
        if let Some(composition) = &parser.composition {
            return composition.completion_view.as_ref().unwrap().clone();
        }
        Self { source: Arc::clone(&parser.completion_template),
            domain: Arc::clone(&parser.completion), fast: None, offset: 0,
            symbols: parser.state_count, guard_owner: false, append_push: None }
    }

    pub(crate) fn relocated(&self, offset: u32) -> crate::Result<Self> {
        let mut view = self.clone();
        view.offset = view.offset.checked_add(offset).ok_or_else(||
            crate::Error::Compilation("scoped program offset overflow".into()))?;
        view.append_push = view.append_push.map(|symbol| symbol.checked_add(offset)
            .ok_or_else(|| crate::Error::Compilation("CALL output offset overflow".into()))).transpose()?;
        view.validate_coordinate(DEFAULT_LABEL as u32 - 1)?;
        Ok(view)
    }

    pub(crate) fn validate_coordinate(&self, alphabet: u32) -> crate::Result<()> {
        if self.symbols == 0 || self.offset.checked_add(self.symbols).is_none_or(|end|
            end > alphabet || end >= DEFAULT_LABEL as u32)
            || self.append_push.is_some_and(|symbol| symbol >= alphabet) {
            return Err(crate::Error::Compilation("scoped program lies outside parser coordinate".into()));
        }
        Ok(())
    }

    pub(crate) fn local(&self, top: u32) -> Option<u32> {
        top.checked_sub(self.offset).filter(|&symbol| symbol < self.symbols)
    }

    pub(crate) fn classify_top(&self, top: u32) -> TopAdmission {
        self.local(top).map_or(TopAdmission::Never, |symbol| self.domain.classify_top(symbol))
    }

    pub(crate) fn admits(&self, stack: &ParserGSS) -> bool {
        if stack.is_empty() { return false; }
        let start = self.domain.start();
        if !self.guard_owner {
            match start { DomainProbe::Accept => return true, DomainProbe::Reject => return false, _ => {} }
        }
        let cursor = match start {
            DomainProbe::Reject => return false,
            DomainProbe::Accept => u32::MAX,
            DomainProbe::NeedMore(cursor) => cursor,
        };
        stack.any_prefix_matching(cursor, |cursor, top| {
            let Some(local) = self.local(*top) else { return ControlFlow::Break(false); };
            if cursor == u32::MAX { return ControlFlow::Break(true); }
            match self.domain.step(cursor, local) {
                DomainProbe::Accept => ControlFlow::Break(true),
                DomainProbe::Reject => ControlFlow::Break(false),
                DomainProbe::NeedMore(next) => ControlFlow::Continue(next),
            }
        })
    }

    pub(crate) fn advance(&self, stack: &ParserGSS) -> ParserGSS {
        if let Some(top) = stack.single_exclusive_top_value()
            && let Some(local) = self.local(top)
            && let Some(symbol) = self.fast.as_ref().and_then(|fast| fast.read_shift.as_ref())
                .and_then(|plan| plan.symbol_for_top(local)) {
            #[cfg(test)]
            crate::runtime::commit::template_advance::PREPARED_RELATION_SHORTCUTS
                .with(|count| count.set(count.get() + 1));
            let output = stack.push(self.offset + symbol);
            return self.append_push.map_or(output.clone(), |symbol| output.push(symbol));
        }
        let input = if self.guard_owner {
            let mut owned = ParserGSS::empty();
            for top in stack.peek_values() {
                if self.local(top).is_some() { owned = owned.merge(&stack.isolate(Some(top))); }
            }
            owned
        } else { stack.clone() };
        let output = crate::runtime::commit::template_advance::advance_with_template_coordinate(
            &self.source, input, Some((self.offset, self.symbols)));
        self.append_push.map_or(output.clone(), |symbol| output.push(symbol))
    }
}

impl crate::compiler::template_follow_support::ScopedFollowProgram for ScopedProgram {
    fn source(&self) -> &CommitTemplateDfas {
        &self.source
    }
    fn offset(&self) -> u32 {
        self.offset
    }
    fn append_push(&self) -> Option<u32> {
        self.append_push
    }
    fn classify_top(&self, top: u32) -> TopAdmission {
        ScopedProgram::classify_top(self, top)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::unweighted_u32::dfa::DFA;
    use crate::compiler::glr::{accumulator::TerminalsDisallowed, labels::encode_negative_label};

    fn view() -> ScopedProgram {
        let mut pop = DFA::new();
        let first = pop.add_state(); let second = pop.add_state(); let dead = pop.add_state();
        pop.add_transition(0, 1, first);
        pop.add_transition(first, DEFAULT_LABEL, second);
        // An explicit dead edge must shadow the productive local DEFAULT.
        pop.add_transition(first, 3, dead);
        let mut push = DFA::new(); let end = push.add_state(); push.set_accepting(end, true);
        push.add_transition(0, encode_negative_label(2), end);
        ScopedProgram::prepare(Arc::new(CommitTemplateDfas { pop, read: DFA::new(), push,
            pop_to_read: vec![], pop_to_push: vec![None, None, Some(0)], read_to_push: vec![] }), 4)
            .unwrap().relocated(100).unwrap()
    }

    #[test]
    fn deeper_default_rejects_foreign_symbols_and_preserves_explicit_dead_priority() {
        let view = view();
        for (word, accepted) in [(vec![5, 101], false), (vec![100, 101], true),
            (vec![103, 101], false), (vec![101], false), (vec![100, 5], false)] {
            let input = ParserGSS::from_single_stack(word, TerminalsDisallowed::new().with_insert(2, 3));
            assert_eq!(view.admits(&input), accepted);
            let actual = view.advance(&input);
            if accepted {
                let expected = input.popn(2).push(102);
                assert_eq!(actual.semantically_eq(&expected, 65536), Some(true));
            } else { assert!(actual.is_empty()); }
        }
    }

    #[test]
    fn nested_offsets_share_graphs_domains_and_fast_views() {
        let a = view(); let b = a.relocated(200).unwrap();
        assert!(Arc::ptr_eq(&a.source, &b.source));
        assert!(Arc::ptr_eq(&a.domain, &b.domain));
        assert!(Arc::ptr_eq(a.fast.as_ref().unwrap(), b.fast.as_ref().unwrap()));
        assert_eq!(b.offset, 300);
    }

    #[test]
    fn live_frontier_with_large_rows_keeps_deeper_scope_and_dead_priority_on_merged_stacks() {
        let mut source = (*view().source).clone();
        let dead = 3;
        // These absent concrete labels must have no effect on the live GSS.
        for label in 4..2048 { source.pop.states[0].transitions.insert(label, dead); }
        let scoped = ScopedProgram::prepare(Arc::new(source), 2048).unwrap().relocated(100).unwrap();
        let annotation = TerminalsDisallowed::new().with_insert(2, 3);
        for words in [vec![vec![100,101],vec![103,101],vec![5,101]],
            vec![vec![103,101],vec![5,101]],vec![vec![100,101],vec![100,102]]] {
            let inputs = words.iter().map(|word| (word.clone(),annotation.clone())).collect::<Vec<_>>();
            let stack = ParserGSS::from_stacks(&inputs);
            let accepted = words.iter().any(|word| word == &[100,101]);
            let expected = if accepted { ParserGSS::from_single_stack(vec![102],annotation.clone()) }
                else { ParserGSS::empty() };
            assert_eq!(scoped.admits(&stack), accepted);
            assert_eq!(scoped.advance(&stack).semantically_eq(&expected,65536),Some(true));
        }
    }

    #[test]
    fn static_mask_query_obeys_the_same_deeper_default_scope_and_empty_stack_rule() {
        let view = view();
        let (mut queries, classes) = crate::template_parser::static_compile::prepare_scoped_boundary_programs(
            &[view.clone()], 104, &std::collections::BTreeSet::from([0])).unwrap();
        let mut query = queries.remove(&0).unwrap();
        glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa_with_pop_classes(
            &mut query, &classes).unwrap();
        let predicate = classes.compile_positive(query, 65536).unwrap();
        for word in [vec![], vec![5, 101], vec![100, 101], vec![103, 101], vec![101]] {
            let input = ParserGSS::from_single_stack(word.clone(), TerminalsDisallowed::new());
            let mut state = predicate.start_state();
            let mut accepted = predicate.states()[state as usize].final_weight.is_some();
            for &top in word.iter().rev() {
                let row = &predicate.states()[state as usize];
                let Some((next, weight)) = row.transitions.get(&(top as i32))
                    .or_else(|| row.transitions.get(&DEFAULT_LABEL)) else { break; };
                if weight.is_empty() { break; }
                state = *next;
                accepted |= predicate.states()[state as usize].final_weight.as_ref()
                    .is_some_and(|weight| !weight.is_empty());
            }
            assert_eq!(accepted, view.admits(&input), "static query differs on {word:?}");
        }
    }
}
