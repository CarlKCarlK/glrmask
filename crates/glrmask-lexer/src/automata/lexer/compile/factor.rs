//! Language-preserving expression normalization and common-factor extraction.

use crate::automata::lexer::ast::Expr;

use crate::ds::u8set::U8Set;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use super::bounded_repeat::cached_direct_bounded_repeat_base_dfa_unconditionally;
use super::deferred::expression_contains_large_bounded_repeat;
use super::dfa_analysis::{dfa_transition_count, productive_dfa_states};
use super::virtual_repeat::{VIRTUAL_BINARY_REPEAT_MIN_BOUND, exact_unit_byte_language};

pub(super) fn unwrap_shared(expr: &Expr) -> &Expr {
    match expr {
        Expr::Shared(inner) => unwrap_shared(inner),
        other => other,
    }
}

pub(super) fn seq_from_parts(mut parts: Vec<Expr>) -> Expr {
    match parts.len() {
        0 => Expr::Epsilon,
        1 => parts.pop().unwrap(),
        _ => Expr::Seq(parts),
    }
}

fn choice_first_part(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Shared(inner) => choice_first_part(inner),
        Expr::Seq(parts) => parts.first(),
        Expr::Epsilon => None,
        other => Some(other),
    }
}

fn choice_without_first_part(expr: &Expr) -> Expr {
    match expr {
        Expr::Shared(inner) => choice_without_first_part(inner),
        Expr::Seq(parts) => seq_from_parts(parts[1..].to_vec()),
        Expr::Epsilon => Expr::Epsilon,
        _ => Expr::Epsilon,
    }
}

fn choice_last_part(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Shared(inner) => choice_last_part(inner),
        Expr::Seq(parts) => parts.last(),
        Expr::Epsilon => None,
        other => Some(other),
    }
}

fn choice_without_last_part(expr: &Expr) -> Expr {
    match expr {
        Expr::Shared(inner) => choice_without_last_part(inner),
        Expr::Seq(parts) => seq_from_parts(parts[..parts.len() - 1].to_vec()),
        Expr::Epsilon => Expr::Epsilon,
        _ => Expr::Epsilon,
    }
}

fn factor_choice_common_prefix(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let prefix = choice_first_part(options.first()?)?.clone();
    if !options
        .iter()
        .all(|option| choice_first_part(option) == Some(&prefix))
    {
        return None;
    }

    let remainders = options
        .iter()
        .map(choice_without_first_part)
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        prefix,
        factor_choice_of_factored(remainders),
    ]))
}

fn factor_choice_common_suffix(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let suffix = choice_last_part(options.first()?)?.clone();
    if !options
        .iter()
        .all(|option| choice_last_part(option) == Some(&suffix))
    {
        return None;
    }

    let prefixes = options
        .iter()
        .map(choice_without_last_part)
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        factor_choice_of_factored(prefixes),
        suffix,
    ]))
}

/// Factor one repeated leading atom even when it is shared by only a subset
/// of the choice arms. This is the exact identity
///
/// ```text
/// A B | A C | D  ==  A (B | C) | D
/// ```
///
/// and is particularly important for mixed anchored/unanchored JSON-schema
/// string patterns: the unanchored arms often share a `JSON_STRING_CHAR*`
/// prefix while an anchored arm does not. Requiring every arm to share the
/// prefix leaves subset construction to rediscover that factoring through a
/// potentially enormous intermediate DFA.
fn factor_choice_repeated_prefix_subset(options: &[Expr]) -> Option<Expr> {
    // Group once instead of rescanning the whole choice for every candidate.
    // Large finite literal languages routinely have thousands of distinct
    // arms; the old nested scan made a no-op subset-factor probe quadratic.
    // Iterating `options` again below preserves the historical deterministic
    // choice of the first repeated leading atom.
    let mut groups = FxHashMap::<&Expr, Vec<usize>>::default();
    for (index, option) in options.iter().enumerate() {
        if let Some(prefix) = choice_first_part(option) {
            groups.entry(prefix).or_default().push(index);
        }
    }

    for (first_index, option) in options.iter().enumerate() {
        let Some(prefix) = choice_first_part(option) else {
            continue;
        };
        let Some(matching) = groups.get(prefix) else {
            continue;
        };
        if matching.len() < 2 || matching.len() == options.len() {
            continue;
        }
        let prefix = prefix.clone();

        let remainders = matching
            .iter()
            .map(|&index| choice_without_first_part(&options[index]))
            .collect::<Vec<_>>();
        let factored_group = seq_from_parts(vec![
            prefix,
            factor_choice_of_factored(remainders),
        ]);
        let mut is_matching = vec![false; options.len()];
        for &index in matching {
            is_matching[index] = true;
        }
        let mut rewritten = Vec::with_capacity(options.len() - matching.len() + 1);
        for (index, option) in options.iter().enumerate() {
            if index == first_index {
                rewritten.push(factored_group.clone());
            } else if !is_matching[index] {
                rewritten.push(option.clone());
            }
        }
        return Some(factor_choice_of_factored(rewritten));
    }
    None
}

/// Suffix counterpart of `factor_choice_repeated_prefix_subset`:
///
/// ```text
/// B A | C A | D  ==  (B | C) A | D
/// ```
fn factor_choice_repeated_suffix_subset(options: &[Expr]) -> Option<Expr> {
    let mut groups = FxHashMap::<&Expr, Vec<usize>>::default();
    for (index, option) in options.iter().enumerate() {
        if let Some(suffix) = choice_last_part(option) {
            groups.entry(suffix).or_default().push(index);
        }
    }

    for (first_index, option) in options.iter().enumerate() {
        let Some(suffix) = choice_last_part(option) else {
            continue;
        };
        let Some(matching) = groups.get(suffix) else {
            continue;
        };
        if matching.len() < 2 || matching.len() == options.len() {
            continue;
        }
        let suffix = suffix.clone();

        let prefixes = matching
            .iter()
            .map(|&index| choice_without_last_part(&options[index]))
            .collect::<Vec<_>>();
        let factored_group = seq_from_parts(vec![
            factor_choice_of_factored(prefixes),
            suffix,
        ]);
        let mut is_matching = vec![false; options.len()];
        for &index in matching {
            is_matching[index] = true;
        }
        let mut rewritten = Vec::with_capacity(options.len() - matching.len() + 1);
        for (index, option) in options.iter().enumerate() {
            if index == first_index {
                rewritten.push(factored_group.clone());
            } else if !is_matching[index] {
                rewritten.push(option.clone());
            }
        }
        return Some(factor_choice_of_factored(rewritten));
    }
    None
}

/// Factor sibling exclusions that subtract the same language:
///
/// ```text
/// (A \\ B) | (C \\ B) | D  ==  ((A | C) \\ B) | D
/// ```
///
/// This is ordinary set algebra on languages. Keeping the subtraction outside
/// the union matters for schema-generated key languages: otherwise every arm
/// is materialized as an independent DFA before the union is determinized.
fn factor_choice_repeated_exclusion_rhs(options: &[Expr]) -> Option<Expr> {
    let mut groups = FxHashMap::<&Expr, Vec<usize>>::default();
    for (index, option) in options.iter().enumerate() {
        if let Expr::Exclude { exclude, .. } = unwrap_shared(option) {
            groups.entry(unwrap_shared(exclude)).or_default().push(index);
        }
    }

    let mut replacement_by_first = FxHashMap::<usize, Expr>::default();
    let mut remove = vec![false; options.len()];
    for matching in groups.values() {
        if matching.len() < 2 {
            continue;
        }
        let first_index = matching[0];
        let Expr::Exclude { exclude, .. } = unwrap_shared(&options[first_index]) else {
            unreachable!("repeated exclusion group contains a non-exclusion arm");
        };
        let lefts = matching
            .iter()
            .map(|&index| match unwrap_shared(&options[index]) {
                Expr::Exclude { expr, .. } => (**expr).clone(),
                _ => unreachable!("repeated exclusion group contains a non-exclusion arm"),
            })
            .collect::<Vec<_>>();
        let left = if lefts.len() == 1 {
            lefts.into_iter().next().unwrap()
        } else {
            // `options` were recursively factored before this helper runs, so
            // every left subtree is already normalized. Do not walk all of
            // them again merely because the set identity introduced a union.
            Expr::Choice(lefts)
        };
        replacement_by_first.insert(
            first_index,
            Expr::Exclude {
                expr: Box::new(left),
                exclude: Box::new((**exclude).clone()),
            },
        );
        for &index in matching.iter().skip(1) {
            remove[index] = true;
        }
    }

    if replacement_by_first.is_empty() {
        return None;
    }
    let mut rewritten = Vec::with_capacity(options.len());
    for (index, option) in options.iter().enumerate() {
        if let Some(replacement) = replacement_by_first.remove(&index) {
            rewritten.push(replacement);
        } else if !remove[index] {
            rewritten.push(option.clone());
        }
    }
    if rewritten.len() == 1 {
        rewritten.pop()
    } else {
        Some(Expr::Choice(rewritten))
    }
}

fn factor_choice_literals(options: &[Expr]) -> Option<Expr> {
    if options.len() < 2 {
        return None;
    }

    let first_byte = match unwrap_shared(options.first()?) {
        Expr::U8Seq(bytes) if !bytes.is_empty() => bytes[0],
        _ => return None,
    };
    for option in options {
        match unwrap_shared(option) {
            Expr::U8Seq(bytes) if !bytes.is_empty() && bytes[0] == first_byte => {}
            _ => return None,
        }
    }

    let remainders = options
        .iter()
        .map(|option| match unwrap_shared(option) {
            Expr::U8Seq(bytes) => {
            if bytes.len() == 1 {
                Expr::Epsilon
            } else {
                Expr::U8Seq(bytes[1..].to_vec())
            }
            }
            _ => unreachable!("literal choice was validated above"),
        })
        .collect::<Vec<_>>();

    Some(seq_from_parts(vec![
        Expr::U8Seq(vec![first_byte]),
        factor_choice_of_factored(remainders),
    ]))
}

/// Exactly factor an intersection of two bounded repetitions over the same
/// certified prefix-free body. Prefix-free body languages are codes, so every
/// word in `L*` has a unique factor count; a common word therefore has one
/// count which must lie in both intervals.
///
/// Keep this rewrite focused on the giant nonzero-minimum case which otherwise
/// needs the special synchronized virtual runtime. Small ordinary products and
/// established zero-minimum paths remain unchanged.
fn factor_same_body_nonzero_repeat_intersection(left: &Expr, right: &Expr) -> Option<Expr> {
    let Expr::Repeat {
        expr: left_body,
        min: left_min,
        max: Some(left_max),
    } = unwrap_shared(left)
    else {
        return None;
    };
    let Expr::Repeat {
        expr: right_body,
        min: right_min,
        max: Some(right_max),
    } = unwrap_shared(right)
    else {
        return None;
    };
    if (*left_min == 0 && *right_min == 0)
        || (*left_max < VIRTUAL_BINARY_REPEAT_MIN_BOUND
            && *right_max < VIRTUAL_BINARY_REPEAT_MIN_BOUND)
        || unwrap_shared(left_body) != unwrap_shared(right_body)
        || expression_contains_large_bounded_repeat(left_body)
    {
        return None;
    }

    // Establish unique factor counts before reasoning about interval overlap.
    // Without this proof, a non-code body such as {"a", "aa"} can represent
    // the same byte string with different repetition counts on the two sides.
    cached_direct_bounded_repeat_base_dfa_unconditionally(left_body, None)?;

    let min = (*left_min).max(*right_min);
    let max = (*left_max).min(*right_max);
    if min > max {
        return Some(Expr::U8Class(U8Set::empty()));
    }
    Some(Expr::Repeat {
        expr: Box::new((**left_body).clone()),
        min,
        max: Some(max),
    })
}

struct DelimitedLiteralRepeatShape<'a> {
    prefix: Vec<u8>,
    body: &'a Expr,
    min: usize,
    max: usize,
    suffix: Vec<u8>,
}

const DELIMITED_REPEAT_SUFFIX_MERGED_STATE_BUDGET: usize = 100_000;

const DELIMITED_REPEAT_SUFFIX_MERGED_TRANSITION_BUDGET: usize = 1_000_000;

fn delimited_literal_repeat_shape(expr: &Expr) -> Option<DelimitedLiteralRepeatShape<'_>> {
    fn literal_bytes(parts: &[Expr]) -> Option<Vec<u8>> {
        let mut bytes = Vec::new();
        for part in parts {
            match unwrap_shared(part) {
                Expr::U8Seq(chunk) => bytes.extend_from_slice(chunk),
                Expr::Epsilon => {}
                _ => return None,
            }
        }
        Some(bytes)
    }

    let Expr::Seq(parts) = unwrap_shared(expr) else {
        return None;
    };
    let mut repeat_index = None;
    for (index, part) in parts.iter().enumerate() {
        if matches!(unwrap_shared(part), Expr::Repeat { max: Some(_), .. }) {
            if repeat_index.replace(index).is_some() {
                return None;
            }
        }
    }
    let repeat_index = repeat_index?;
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared(&parts[repeat_index])
    else {
        return None;
    };
    let prefix = literal_bytes(&parts[..repeat_index])?;
    let suffix = literal_bytes(&parts[repeat_index + 1..])?;
    if suffix.is_empty() {
        return None;
    }
    Some(DelimitedLiteralRepeatShape {
        prefix,
        body,
        min: *min,
        max: *max,
        suffix,
    })
}

/// Exactly factor two bounded repeats with a shared literal
/// prefix/body and an unambiguous literal suffix boundary. A certified
/// prefix-free body is an instantaneous code. If the first suffix byte cannot
/// begin any productive body word, both parses must leave the repeated body at
/// the same code-word boundary and therefore at the same repetition count.
/// The remaining exact literal suffixes can then be compared directly.
///
/// Keep this focused on giant repeat+suffix shapes. The rewrite deliberately
/// refuses suffixes whose first byte can begin another body word: e.g.
/// `a*ab ∩ a*b` overlaps through a one-copy count shift and must remain a real
/// intersection.
fn factor_same_body_delimited_literal_repeat_suffix_intersection(
    left: &Expr,
    right: &Expr,
) -> Option<Expr> {
    let left = delimited_literal_repeat_shape(left)?;
    let right = delimited_literal_repeat_shape(right)?;
    if (left.max < VIRTUAL_BINARY_REPEAT_MIN_BOUND
        && right.max < VIRTUAL_BINARY_REPEAT_MIN_BOUND)
        || left.prefix != right.prefix
        || unwrap_shared(left.body) != unwrap_shared(right.body)
        || expression_contains_large_bounded_repeat(left.body)
    {
        return None;
    }

    let base_dfa = cached_direct_bounded_repeat_base_dfa_unconditionally(left.body, None)?;
    let productive = productive_dfa_states(&base_dfa);
    let suffix_can_start_body = |suffix: &[u8]| {
        base_dfa
            .step(0, suffix[0])
            .is_some_and(|target| productive[target as usize])
    };
    if suffix_can_start_body(&left.suffix) || suffix_can_start_body(&right.suffix) {
        return None;
    }

    let min = left.min.max(right.min);
    let max = left.max.min(right.max);
    if min > max {
        return Some(Expr::U8Class(U8Set::empty()));
    }
    if left.suffix != right.suffix {
        return Some(Expr::U8Class(U8Set::empty()));
    }

    // The theorem is exact for arbitrary minima, but a giant positive-minimum
    // repeat followed by a suffix does not yet have its own symbolic runtime
    // lane. Keep that still-giant result as the original intersection instead
    // of turning a safely rejected shape into a later eager compile. If the
    // merged upper bound is ordinary, explicitly bound the layered DFA that
    // the rewrite can cause rather than treating the giant threshold itself as
    // a memory budget.
    if max >= VIRTUAL_BINARY_REPEAT_MIN_BOUND {
        if min != 0 {
            return None;
        }
    } else if max != 0 {
        let layers = max.checked_add(1)?;
        let states = layers
            .checked_mul(base_dfa.num_states())?
            .checked_add(left.prefix.len())?
            .checked_add(left.suffix.len())?;
        let transitions = max
            .checked_mul(dfa_transition_count(&base_dfa))?
            .checked_add(layers)?
            .checked_add(left.prefix.len())?
            .checked_add(left.suffix.len())?;
        if states > DELIMITED_REPEAT_SUFFIX_MERGED_STATE_BUDGET
            || transitions > DELIMITED_REPEAT_SUFFIX_MERGED_TRANSITION_BUDGET
        {
            return None;
        }
    }

    let mut parts = Vec::with_capacity(3);
    if !left.prefix.is_empty() {
        parts.push(Expr::U8Seq(left.prefix));
    }
    if max != 0 {
        parts.push(Expr::Repeat {
            expr: Box::new((*left.body).clone()),
            min,
            max: Some(max),
        });
    }
    parts.push(Expr::U8Seq(left.suffix));
    Some(seq_from_parts(parts))
}

/// Exactly normalize two aligned unit-byte repeats. Iteration boundaries
/// coincide after every consumed byte, so the byte language and repetition
/// count intervals can be intersected independently.
fn factor_aligned_unit_repeat_intersection(left: &Expr, right: &Expr) -> Option<Expr> {
    let Expr::Repeat {
        expr: left_body,
        min: left_min,
        max: Some(left_max),
    } = unwrap_shared(left)
    else {
        return None;
    };
    let Expr::Repeat {
        expr: right_body,
        min: right_min,
        max: Some(right_max),
    } = unwrap_shared(right)
    else {
        return None;
    };

    let body = exact_unit_byte_language(left_body)?
        .intersection(&exact_unit_byte_language(right_body)?);
    let min = (*left_min).max(*right_min);
    let max = (*left_max).min(*right_max);
    if min > max {
        return Some(Expr::U8Class(U8Set::empty()));
    }
    if body.is_empty() {
        return Some(if min == 0 {
            Expr::Epsilon
        } else {
            Expr::U8Class(U8Set::empty())
        });
    }
    Some(Expr::Repeat {
        expr: Box::new(Expr::U8Class(body)),
        min,
        max: Some(max),
    })
}

fn factor_regex_expr_impl(
    expr: Expr,
    shared_cache: Option<&FxHashMap<usize, Arc<Expr>>>,
) -> Expr {
    match expr {
        Expr::Seq(parts) => {
            let mut out = Vec::new();
            for part in parts {
                match factor_regex_expr_impl(part, shared_cache) {
                    Expr::Seq(inner) => out.extend(inner),
                    Expr::Epsilon => {}
                    other => out.push(other),
                }
            }
            seq_from_parts(out)
        }
        Expr::Choice(options) => {
            // A large finite choice of exact byte strings is already an ideal
            // input to ordinary subset construction: the resulting DFA is the
            // shared-prefix trie of those strings. Recursively peeling common
            // bytes and probing subset prefix/suffix factors only rebuilds and
            // hashes the entire literal set repeatedly, while providing no
            // protection against subset-state explosion (there is none for a
            // finite literal trie). Keep small choices on the historical path,
            // where syntactic factoring is cheap and can reduce setup overhead.
            const LARGE_PURE_LITERAL_CHOICE_NO_FACTOR: usize = 64;
            if options.len() >= LARGE_PURE_LITERAL_CHOICE_NO_FACTOR
                && options
                    .iter()
                    .all(|option| matches!(unwrap_shared(option), Expr::U8Seq(_)))
            {
                return Expr::Choice(options);
            }
            let factored_options = options
                .into_iter()
                .map(|expr| factor_regex_expr_impl(expr, shared_cache))
                .collect::<Vec<_>>();
            factor_choice_of_factored(factored_options)
        }
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(factor_regex_expr_impl(*expr, shared_cache)),
            min,
            max,
        },
        Expr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(factor_regex_expr_impl(*expr, shared_cache)),
            exclude: Box::new(factor_regex_expr_impl(*exclude, shared_cache)),
        },
        Expr::Intersect { expr, intersect } => {
            let expr = factor_regex_expr_impl(*expr, shared_cache);
            let intersect = factor_regex_expr_impl(*intersect, shared_cache);
            factor_same_body_delimited_literal_repeat_suffix_intersection(&expr, &intersect)
                .or_else(|| factor_aligned_unit_repeat_intersection(&expr, &intersect))
                .or_else(|| factor_same_body_nonzero_repeat_intersection(&expr, &intersect))
                .unwrap_or_else(|| Expr::Intersect {
                    expr: Box::new(expr),
                    intersect: Box::new(intersect),
                })
        }
        Expr::Shared(inner) => {
            if let Some(cached) = shared_cache
                .and_then(|cache| cache.get(&(Arc::as_ptr(&inner) as usize)))
            {
                // Preserve the reference boundary after factoring. Expanding a
                // cached result back into each caller recreates the expression
                // tree and makes downstream Boolean materialization repeat the
                // same structural work that this cache was meant to avoid.
                return Expr::Shared(Arc::clone(cached));
            }
            factor_regex_expr_impl((*inner).clone(), shared_cache)
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => expr,
    }
}

pub fn factor_regex_expr(expr: Expr) -> Expr {
    factor_regex_expr_impl(expr, None)
}

pub fn factor_regex_expr_with_shared_cache(
    expr: Expr,
    shared_cache: &FxHashMap<usize, Arc<Expr>>,
) -> Expr {
    factor_regex_expr_impl(expr, Some(shared_cache))
}



pub(super) fn expr_contains_group_op(expr: &Expr) -> bool {
    match expr {
        Expr::Exclude { .. } | Expr::Intersect { .. } => true,
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_contains_group_op),
        Expr::Repeat { expr, .. } => expr_contains_group_op(expr),
        Expr::Shared(inner) => expr_contains_group_op(inner),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
    }
}

pub(super) fn group_op_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Exclude { expr, exclude } => {
            1 + group_op_node_count(expr) + group_op_node_count(exclude)
        }
        Expr::Intersect { expr, intersect } => {
            1 + group_op_node_count(expr) + group_op_node_count(intersect)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().map(group_op_node_count).sum(),
        Expr::Repeat { expr, .. } => group_op_node_count(expr),
        Expr::Shared(expr) => group_op_node_count(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => 0,
    }
}

/// Factor a newly introduced choice whose children were already recursively
/// factored. Prefix/suffix rewrites change the union/concatenation spine, not
/// their child languages. Restarting the entire recursive normalizer here
/// repeats work and expands cached Shared children under a fresh empty cache.
fn factor_choice_of_factored(mut options: Vec<Expr>) -> Expr {
    const LARGE_PURE_LITERAL_CHOICE_NO_FACTOR: usize = 64;
    if options.len() >= LARGE_PURE_LITERAL_CHOICE_NO_FACTOR
        && options.iter().all(|option| matches!(unwrap_shared(option), Expr::U8Seq(_)))
    {
        return Expr::Choice(options);
    }
    if options.len() == 1 {
        return options.pop().unwrap();
    }
    if let Some(factored) = factor_choice_literals(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_common_prefix(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_common_suffix(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_repeated_exclusion_rhs(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_repeated_prefix_subset(&options) {
        return factored;
    }
    if let Some(factored) = factor_choice_repeated_suffix_subset(&options) {
        return factored;
    }
    Expr::Choice(options)
}
