//! Parser advancement, template selection, and concrete stack actions.

use super::admission::parser_may_advance_on;
use super::engine::AdvanceResultCache;
use super::lexical::apply_future_terminal_disallow;
use super::template_advance::advance_stacks_template_dfa;
use super::template_advance::advance_stacks_template_dfa_owned;
use crate::automata::lexer::tokenizer::TokenizerExecResult;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::AdvanceProfile;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::parser::advance_control_closed_stacks;
use crate::compiler::glr::parser::advance_control_closed_stacks_owned;
use crate::compiler::glr::parser::advance_stacks;
use crate::compiler::glr::parser::advance_stacks_owned;
use crate::compiler::glr::parser::advance_stacks_profiled;
use crate::compiler::glr::parser::apply_guarded_stack_shifts_fast;
use crate::compiler::glr::table::Action;
use crate::compiler::glr::table::AdmissionPolicy;
use crate::compiler::glr::table::GLRTable;
use crate::runtime::constraint::Constraint;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

pub(super) const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 256;

pub(super) static TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();

pub(super) static VALIDATE_TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();

pub(super) fn template_advance_enabled() -> bool {
    *TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_ENABLE_TEMPLATE_DFA_ADVANCE").is_some())
}

pub(super) fn validate_template_advance_enabled() -> bool {
    *VALIDATE_TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_VALIDATE_TEMPLATE_DFA_ADVANCE").is_some())
}

pub(super) fn advance_parser_stacks(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        return advanced;
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(stack, terminal) {
        return cached;
    }
    if template_advance_enabled()
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks(&constraint.table, stack, terminal);
            assert!(
                template_advanced
                    .semantically_eq(&table_advanced, 4_096)
                    .expect("template validation exceeded explicit stack limit"),
                "template-DFA advance mismatch for terminal {terminal}; template={:?} table={:?}",
                template_advanced
                    .to_stacks(4_096)
                    .expect("stack enumeration exceeded explicit limit"),
                table_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
            );
        }
        return template_advanced;
    }

    if constraint.table.control_terminals.is_empty() {
        advance_stacks(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks(&constraint.table, stack, terminal)
    }
}

pub(super) fn advance_parser_stacks_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if let Some(advanced) = constraint.advance_compact_segmented_parser(&stack, terminal) {
        return advanced;
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(&stack, terminal) {
        return cached;
    }
    if template_advance_enabled()
        && let Some(template_advanced) =
            advance_stacks_template_dfa_owned(constraint, stack.clone(), terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks_owned(&constraint.table, stack, terminal);
            assert!(
                template_advanced
                    .semantically_eq(&table_advanced, 4_096)
                    .expect("template validation exceeded explicit stack limit"),
                "template-DFA advance mismatch for terminal {terminal}; template={:?} table={:?}",
                template_advanced
                    .to_stacks(4_096)
                    .expect("stack enumeration exceeded explicit limit"),
                table_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
            );
        }
        return template_advanced;
    }

    if constraint.table.control_terminals.is_empty() {
        advance_stacks_owned(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks_owned(&constraint.table, stack, terminal)
    }
}

pub(super) fn advance_parser_stacks_profiled(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> (ParserGSS, AdvanceProfile) {
    let template_start = std::time::Instant::now();
    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        let elapsed = template_start.elapsed().as_nanos() as u64;
        return (
            advanced,
            AdvanceProfile {
                total_ns: elapsed,
                fast_path_ns: elapsed,
                top_states: stack.top_value_count() as u32,
                gss_depth: stack.max_depth(),
                ..AdvanceProfile::default()
            },
        );
    }
    if let Some(cached) = constraint.direct_regular_cached_advance(stack, terminal) {
        let elapsed = template_start.elapsed().as_nanos() as u64;
        return (
            cached,
            AdvanceProfile {
                total_ns: elapsed,
                fast_path_ns: elapsed,
                top_states: stack.top_value_count() as u32,
                gss_depth: stack.max_depth(),
                ..AdvanceProfile::default()
            },
        );
    }
    if template_advance_enabled()
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        let template_elapsed = template_start.elapsed().as_nanos() as u64;
        if validate_template_advance_enabled() {
            let (table_advanced, table_profile) =
                advance_stacks_profiled(&constraint.table, stack, terminal);
            assert!(
                template_advanced
                    .semantically_eq(&table_advanced, 4_096)
                    .expect("template validation exceeded explicit stack limit"),
                "template-DFA advance mismatch for terminal {terminal}; template={:?} table={:?}",
                template_advanced
                    .to_stacks(4_096)
                    .expect("stack enumeration exceeded explicit limit"),
                table_advanced.to_stacks(4_096).expect("stack enumeration exceeded explicit limit"),
            );
            return (template_advanced, table_profile);
        }
        return (
            template_advanced,
            AdvanceProfile {
                total_ns: template_elapsed,
                fast_path_ns: template_elapsed,
                top_states: stack.peek_values().len() as u32,
                gss_depth: stack.max_depth(),
                vstack_len: stack.try_virtual_stack().map_or(0, |vstack| vstack.len() as u32),
                ..AdvanceProfile::default()
            },
        );
    }

    advance_stacks_profiled(&constraint.table, stack, terminal)
}

pub(crate) fn advance_parser_stacks_if_possible(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if constraint.table.admission_policy == AdmissionPolicy::RowPresenceExact
        && !parser_may_advance_on(constraint, stack, terminal)
    {
        return None;
    }
    let advanced = advance_parser_stacks(constraint, stack, terminal);
    (!advanced.is_empty()).then_some(advanced)
}

/// Advance against the authoritative composed GLR table without consulting
/// parser-DWA/direct-regular admission caches.
///
/// Segmented boundary B deliberately represents language that is absent from
/// retained component/static A. After loading a hybrid, the constraint's
/// parser DWA may therefore be exactly A; using the ordinary admission
/// prefilter here would circularly reject B-only cross-component terminals.
pub(crate) fn advance_parser_stacks_table_exact(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    if let Some(advanced) = constraint.advance_compact_segmented_parser(stack, terminal) {
        return (!advanced.is_empty()).then_some(advanced);
    }
    let advanced = if constraint.table.control_terminals.is_empty() {
        advance_stacks(&constraint.table, stack, terminal)
    } else {
        advance_control_closed_stacks(&constraint.table, stack, terminal)
    };
    (!advanced.is_empty()).then_some(advanced)
}

pub(super) struct ProfiledAdvanceAttempt {
    pub(super) advanced: ParserGSS,
    pub(super) profile: AdvanceProfile,
    pub(super) may_ns: u64,
    pub(super) core_ns: u64,
}

pub(super) fn advance_parser_stacks_profiled_if_possible(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ProfiledAdvanceAttempt {
    use std::time::Instant;

    let mut may_ns = 0;
    if constraint.table.admission_policy == AdmissionPolicy::RowPresenceExact {
        let may_started_at = Instant::now();
        let admitted = parser_may_advance_on(constraint, stack, terminal);
        may_ns = may_started_at.elapsed().as_nanos() as u64;
        if !admitted {
            return ProfiledAdvanceAttempt {
                advanced: ParserGSS::empty(),
                profile: AdvanceProfile::default(),
                may_ns,
                core_ns: 0,
            };
        }
    }

    let core_started_at = Instant::now();
    let (advanced, profile) = advance_parser_stacks_profiled(constraint, stack, terminal);
    let core_ns = core_started_at.elapsed().as_nanos() as u64;
    ProfiledAdvanceAttempt { advanced, profile, may_ns, core_ns }
}

#[inline]
pub(super) fn try_apply_single_top_action_in_place(gss: &mut ParserGSS, action: &Action) -> bool {
    match action {
        Action::Skip => true,
        Action::Shift(target, replace) => {
            let pushes = [*target];
            gss.try_apply_single_segment_stack_effect_in_place(usize::from(*replace), &pushes)
        }
        Action::ReplaceShifts(targets) if targets.len() == 1 => {
            gss.try_apply_single_segment_stack_effect_in_place(1, targets)
        }
        Action::StackShifts(shifts) => {
            let [shift] = shifts.as_slice() else {
                return false;
            };
            gss.try_apply_single_segment_stack_effect_in_place(shift.pop as usize, &shift.pushes)
        }
        _ => false,
    }
}

#[inline]
pub(super) fn apply_single_top_action_fast(
    constraint: &Constraint,
    gss: &ParserGSS,
    state: u32,
    terminal: u32,
    action: &Action,
) -> Option<ParserGSS> {
    if let Some(cached) = constraint.direct_regular_cached_advance(gss, terminal) {
        return Some(cached);
    }
    let table = &constraint.table;
    match action {
        Action::Skip => Some(gss.clone()),
        Action::Shift(target, is_replace) => {
            if let Some(mut stack) = gss.try_virtual_stack() {
                if *is_replace && stack.pop(1) != 0 {
                    return Some(gss.popn(1).push(*target));
                }
                stack.push(*target);
                return Some(stack.into_gss());
            } else {
                Some(if *is_replace { gss.popn(1).push(*target) } else { gss.push(*target) })
            }
        }
        Action::ReplaceShifts(targets) => {
            let stack = gss.try_virtual_stack()?;
            stack.into_gss_after_popping_and_pushing_unique_single_branches(1, targets.iter())
        }
        Action::StackShifts(shifts) => {
            if let [shift] = shifts.as_slice() {
                let mut branch = gss.try_virtual_stack()?;
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                return Some(branch.into_gss());
            }
            if let Some(stack) = gss.try_virtual_stack()
                && let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts.iter().all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = stack.into_gss_after_popping_and_pushing_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }
            if let Some(shifted) = gss.apply_stack_effects_to_single_concrete_path(
                shifts.iter().map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
            ) {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && shifts.iter().all(|shift| shift.pop == first.pop && shift.pushes.len() == 1)
                && let Some(shifted) = gss.apply_shared_pop_push_single_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| &shift.pushes[0]),
                )
            {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts.iter().all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = gss.apply_shared_pop_push_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }

            let stack = gss.try_virtual_stack()?;
            let mut shifted = ParserGSS::empty();
            for shift in shifts {
                let mut branch = stack.clone();
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                let branch = branch.into_gss();
                shifted = if shifted.is_empty() { branch } else { shifted.merge(&branch) };
            }
            Some(shifted)
        }
        Action::GuardedStackShifts(shifts) => {
            apply_guarded_stack_shifts_fast(gss, shifts, table.guarded_shift_index(state, terminal))
        }
        Action::Reduce(..) => apply_single_path_reduce_chain_fast(table, gss, terminal),
        _ => None,
    }
}

pub(super) fn try_apply_action_to_carried_virtual_stack(
    stack: &mut crate::ds::leveled_gss::VirtualStack<u32, TerminalsDisallowed>,
    action: &Action,
) -> bool {
    match action {
        Action::Skip => true,
        Action::Shift(target, is_replace) => {
            if *is_replace {
                stack.replace_top(*target)
            } else {
                stack.push(*target);
                true
            }
        }
        Action::StackShifts(shifts) => {
            let [shift] = shifts.as_slice() else {
                return false;
            };
            let mut candidate = stack.clone();
            if candidate.pop(shift.pop as usize) != 0 {
                return false;
            }
            for &target in &shift.pushes {
                candidate.push(target);
            }
            if candidate.top().is_none() {
                // The next parser/top-row decision requires a concrete top
                // state. If the visible prefix has been exhausted, continuing
                // to carry this virtual stack would make the stale materialized
                // GSS an invalid proxy for the current parser frontier.
                return false;
            }
            *stack = candidate;
            true
        }
        _ => false,
    }
}

pub(super) fn apply_single_path_reduce_chain_fast(
    table: &GLRTable,
    gss: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    let (mut stack, acc) = gss.try_single_stack_bounded(SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH)?;

    loop {
        let state = *stack.last()?;
        match table.action(state, terminal)? {
            Action::Skip => {
                return Some(ParserGSS::from_single_stack(stack, acc));
            }
            Action::Reduce(nt, len) => {
                let rhs_len = *len as usize;
                if rhs_len >= stack.len() {
                    return None;
                }
                stack.truncate(stack.len() - rhs_len);
                let goto_from = *stack.last()?;
                let (target, is_replace) = table.goto_target(goto_from, *nt)?;
                if is_replace {
                    *stack.last_mut()? = target;
                } else {
                    stack.push(target);
                }
            }
            Action::Shift(target, is_replace) => {
                if *is_replace {
                    *stack.last_mut()? = *target;
                } else {
                    stack.push(*target);
                }
                return Some(ParserGSS::from_single_stack(stack, acc));
            }
            Action::StackShifts(shifts) => {
                return ParserGSS::from_single_stack(stack, acc)
                    .apply_stack_effects_to_single_concrete_path(
                        shifts.iter().map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                        SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                    );
            }
            Action::Split { shift, reduces, accept: false } => {
                let mut out: Vec<(Vec<u32>, TerminalsDisallowed)> = Vec::new();

                if let Some((target, is_replace)) = shift {
                    let mut branch = stack.clone();
                    if *is_replace {
                        *branch.last_mut()? = *target;
                    } else {
                        branch.push(*target);
                    }
                    out.push((branch, acc.clone()));
                }

                for &(nt, len) in reduces {
                    let mut branch = stack.clone();
                    let rhs_len = len as usize;
                    if rhs_len >= branch.len() {
                        return None;
                    }
                    branch.truncate(branch.len() - rhs_len);
                    let goto_from = *branch.last()?;
                    let (target, is_replace) = table.goto_target(goto_from, nt)?;
                    if is_replace {
                        *branch.last_mut()? = target;
                    } else {
                        branch.push(target);
                    }

                    let follow_state = *branch.last()?;
                    match table.action(follow_state, terminal)? {
                        Action::Skip => {
                            out.push((branch, acc.clone()));
                        }
                        Action::Shift(target, is_replace) => {
                            if *is_replace {
                                *branch.last_mut()? = *target;
                            } else {
                                branch.push(*target);
                            }
                            out.push((branch, acc.clone()));
                        }
                        Action::StackShifts(shifts) => {
                            let shifted = ParserGSS::from_single_stack(branch, acc.clone())
                                .apply_stack_effects_to_single_concrete_path(
                                    shifts
                                        .iter()
                                        .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                                    SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                                )?;
                            let shifted_stacks = shifted
                                .to_stacks(shifts.len())
                                .expect("stack-shift result exceeded its effect count");
                            out.extend(shifted_stacks);
                        }
                        _ => return None,
                    }
                }

                return (!out.is_empty()).then(|| ParserGSS::from_stacks(&out));
            }
            _ => return None,
        }
    }
}

pub(super) fn advance_terminal_match(
    constraint: &Constraint,
    gss_at_offset: &ParserGSS,
    terminal: u32,
    exec_result: &TokenizerExecResult,
    advance_result_cache: &mut AdvanceResultCache,
    terminal_result_cache: &mut FxHashMap<u32, ParserGSS>,
) -> Option<ParserGSS> {
    if let Some(cached) = terminal_result_cache.get(&terminal) {
        return (!cached.is_empty()).then(|| cached.clone());
    }

    let advance_cache_key = (gss_at_offset.ptr_key(), terminal);
    let advanced = if let Some((_, cached)) = advance_result_cache.get(&advance_cache_key) {
        cached.clone()
    } else {
        let advanced = advance_parser_stacks(constraint, gss_at_offset, terminal);
        advance_result_cache.insert(advance_cache_key, (gss_at_offset.clone(), advanced.clone()));
        advanced
    };

    let advanced = apply_future_terminal_disallow(constraint, exec_result, terminal, advanced);
    terminal_result_cache.insert(terminal, advanced.clone());
    (!advanced.is_empty()).then_some(advanced)
}
