//! Lower nested intersections/exclusions into explicit product compilation plans.
//! Shared subexpressions are compiled once without holding locks across Rayon work.

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::dfa::DFA;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use super::{compile_single_expr_dfa, compile_with_plan};
use super::deferred::expression_contains_large_bounded_repeat;
use super::dfa_analysis::dfa_transition_count;
use super::factor::{expr_contains_group_op, group_op_node_count};
use super::nfa::{compile_expr_to_dfa, expr_u8set};
use super::product::{ProductComponentProfileLabel, build_dense_binary_exclusion_dfa};
use rayon::prelude::*;

fn split_top_level_group_ops(expr: &Expr) -> (Expr, Vec<Expr>, Vec<Expr>) {
    match expr {
        Expr::Exclude { expr, exclude } => {
            let (base, mut excluded, intersections) = split_top_level_group_ops(expr);
            excluded.push((**exclude).clone());
            (base, excluded, intersections)
        }
        Expr::Intersect { expr, intersect } => {
            let (base, excluded, mut intersections) = split_top_level_group_ops(expr);
            intersections.push((**intersect).clone());
            (base, excluded, intersections)
        }
        Expr::Shared(inner)
            if matches!(inner.as_ref(), Expr::Exclude { .. } | Expr::Intersect { .. }) => {
            split_top_level_group_ops(inner.as_ref())
        }
        _ => (expr.clone(), Vec::new(), Vec::new()),
    }
}

pub(super) fn rebuild_single_visible_group_expression(
    compiled_exprs: &[Expr],
    exclusions: &BTreeMap<u32, BTreeSet<u32>>,
    intersections: &BTreeMap<u32, BTreeSet<u32>>,
) -> Option<Expr> {
    let mut used = vec![false; compiled_exprs.len()];
    let mut expression = compiled_exprs.first()?.clone();
    used[0] = true;

    let mut apply_hidden = |hidden: u32, intersect: bool| -> Option<()> {
        let hidden = hidden as usize;
        if hidden == 0 || hidden >= compiled_exprs.len() || used[hidden] {
            return None;
        }
        used[hidden] = true;
        let hidden_expression = compiled_exprs[hidden].clone();
        expression = if intersect {
            Expr::Intersect {
                expr: Box::new(expression.clone()),
                intersect: Box::new(hidden_expression),
            }
        } else {
            Expr::Exclude {
                expr: Box::new(expression.clone()),
                exclude: Box::new(hidden_expression),
            }
        };
        Some(())
    };

    if exclusions.keys().any(|&visible| visible != 0)
        || intersections.keys().any(|&visible| visible != 0)
    {
        return None;
    }
    for &hidden in exclusions.get(&0).into_iter().flatten() {
        apply_hidden(hidden, false)?;
    }
    for &hidden in intersections.get(&0).into_iter().flatten() {
        apply_hidden(hidden, true)?;
    }
    used.into_iter().all(|used| used).then(|| expression.optimize())
}

/// Expose one intersection nested under a common sequence shell. This is an
/// exact distributive rewrite:
///
/// ```text
/// prefix · (left ∩ right) · suffix
///   = (prefix · left · suffix) ∩ (prefix · right · suffix)
/// ```
///
/// Keeping the operands visible lets structural full/stencil compilation map
/// their residual coordinates independently and materialize missing correlated
/// product tuples. We deliberately handle exactly one nested intersection to
/// avoid an exponential distribution over multiple group operations.
pub(super) fn lift_single_nested_intersection(expr: &Expr) -> Option<Expr> {
    let Expr::Seq(parts) = expr else {
        return None;
    };
    let mut intersection = None;
    for (index, part) in parts.iter().enumerate() {
        let part = match part {
            Expr::Shared(inner) => inner.as_ref(),
            other => other,
        };
        if let Expr::Intersect { expr, intersect } = part {
            if intersection.is_some() {
                return None;
            }
            intersection = Some((index, expr.as_ref(), intersect.as_ref()));
        }
    }
    let (index, left, right) = intersection?;
    let branch = |replacement: &Expr| {
        let mut branch = parts.to_vec();
        branch[index] = replacement.clone();
        Expr::Seq(branch).optimize()
    };
    Some(Expr::Intersect {
        expr: Box::new(branch(left)),
        intersect: Box::new(branch(right)),
    })
}

#[derive(Default)]
pub(super) struct NestedGroupOpCache {
    pub(super) compiled: FxHashMap<Expr, Arc<DFA>>,
    pub(super) shared_duplicates: Option<Arc<SharedDuplicateNestedGroupOpCache>>,
    pub(super) allow_shared_initialization: bool,
    pub(super) cache_hits: usize,
    pub(super) cache_misses: usize,
    pub(super) compiled_ms: f64,
    pub(super) max_compile_ms: f64,
}

pub(super) struct SharedDuplicateNestedGroupOpCache {
    pub(super) duplicated: FxHashSet<Expr>,
    pub(super) compiled: Mutex<FxHashMap<Expr, Arc<OnceLock<Arc<DFA>>>>>,
}

pub(super) fn unique_dense_exclusion_rhs_is_compound(expr: &Expr) -> bool {
    !matches!(expr, Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon)
}

impl SharedDuplicateNestedGroupOpCache {
    pub(super) fn cell_if_duplicated(&self, expr: &Expr) -> Option<Arc<OnceLock<Arc<DFA>>>> {
        if !self.duplicated.contains(expr) {
            return None;
        }
        let mut compiled = self
            .compiled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Some(Arc::clone(
            compiled
                .entry(expr.clone())
                .or_insert_with(|| Arc::new(OnceLock::new())),
        ))
    }

    pub(super) fn all_entries_initialized(&self) -> bool {
        let compiled = self
            .compiled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        compiled.len() == self.duplicated.len()
            && compiled.values().all(|cell| cell.get().is_some())
    }
}

pub(super) fn materialize_nested_group_ops(expr: Expr, cache: &mut NestedGroupOpCache) -> Expr {
    match expr {
        expr @ (Expr::Exclude { .. } | Expr::Intersect { .. }) => {
            if let Some(compiled) = cache.compiled.get(&expr) {
                cache.cache_hits += 1;
                return Expr::Dfa(compiled.clone());
            }

            if let Some(shared) = cache.shared_duplicates.clone()
                && let Some(cell) = shared.cell_if_duplicated(&expr)
            {
                let started_at = Instant::now();
                let was_ready = cell.get().is_some();
                let compiled = if cache.allow_shared_initialization {
                    Arc::clone(cell.get_or_init(|| {
                        let mut nested_cache = NestedGroupOpCache {
                            shared_duplicates: Some(Arc::clone(&shared)),
                            allow_shared_initialization: true,
                            ..NestedGroupOpCache::default()
                        };
                        Arc::new(compile_with_plan(
                            build_exclusion_compile_plan_with_labels_and_cache(
                                std::slice::from_ref(&expr),
                                None,
                                &mut nested_cache,
                            ),
                        ))
                    }))
                } else {
                    Arc::clone(cell.get().expect(
                        "shared nested group-op cache must be prewarmed before parallel compilation",
                    ))
                };
                if was_ready {
                    cache.cache_hits += 1;
                } else {
                    cache.cache_misses += 1;
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    cache.compiled_ms += elapsed_ms;
                    cache.max_compile_ms = cache.max_compile_ms.max(elapsed_ms);
                }
                cache.compiled.insert(expr, Arc::clone(&compiled));
                return Expr::Dfa(compiled);
            }

            cache.cache_misses += 1;
            let started_at = Instant::now();
            let compiled = if let Expr::Exclude { expr: left, exclude: right } = &expr
                && unique_dense_exclusion_rhs_is_compound(right)
            {
                // Nested exclusions used to take the generic group-product path unless
                // the exact same subtree appeared often enough to enter the shared
                // prewarm cache.  Unique exclusions are just as amenable to the exact
                // dense binary subtraction kernel: materialize each operand once, then
                // keep a dead-RHS sentinel instead of hashing partial product tuples.
                // Atomic RHS exclusions are intentionally left on the generic planner:
                // recursively densifying long chains of exact literals repeatedly
                // rebuilds a growing left DFA and can be several times slower.
                // Fall back to the generic planner if either operand cannot use that
                // representation.
                let left = materialize_nested_group_ops((**left).clone(), cache);
                let right = materialize_nested_group_ops((**right).clone(), cache);
                let left = match left {
                    Expr::Dfa(dfa) => dfa,
                    other => Arc::new(compile_single_expr_dfa(&other)),
                };
                let right = match right {
                    Expr::Dfa(dfa) => dfa,
                    other => Arc::new(compile_single_expr_dfa(&other)),
                };
                build_dense_binary_exclusion_dfa(
                    left.as_ref(),
                    right.as_ref(),
                    expr_u8set(&expr),
                )
                .map(Arc::new)
                .unwrap_or_else(|| {
                    Arc::new(compile_with_plan(
                        build_exclusion_compile_plan_with_labels_and_cache(
                            std::slice::from_ref(&expr),
                            None,
                            cache,
                        ),
                    ))
                })
            } else {
                Arc::new(compile_with_plan(
                    build_exclusion_compile_plan_with_labels_and_cache(
                        std::slice::from_ref(&expr),
                        None,
                        cache,
                    ),
                ))
            };
            let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            cache.compiled_ms += elapsed_ms;
            cache.max_compile_ms = cache.max_compile_ms.max(elapsed_ms);
            cache.compiled.insert(expr, compiled.clone());
            Expr::Dfa(compiled)
        }
        Expr::Seq(parts) => Expr::Seq(
            parts
                .into_iter()
                .map(|part| materialize_nested_group_ops(part, cache))
                .collect(),
        ),
        Expr::Choice(options) => Expr::Choice(
            options
                .into_iter()
                .map(|option| materialize_nested_group_ops(option, cache))
                .collect(),
        ),
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(materialize_nested_group_ops(*expr, cache)),
            min,
            max,
        },
        Expr::Shared(inner) => {
            let rewritten = materialize_nested_group_ops((*inner).clone(), cache);
            if rewritten == *inner {
                Expr::Shared(inner)
            } else {
                rewritten
            }
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => expr,
    }
}

fn count_nested_group_ops(expr: &Expr, counts: &mut FxHashMap<Expr, usize>) {
    match expr {
        Expr::Exclude { expr: inner, exclude } => {
            *counts.entry(expr.clone()).or_default() += 1;
            count_nested_group_ops(inner, counts);
            count_nested_group_ops(exclude, counts);
        }
        Expr::Intersect { expr: inner, intersect } => {
            *counts.entry(expr.clone()).or_default() += 1;
            count_nested_group_ops(inner, counts);
            count_nested_group_ops(intersect, counts);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            for part in parts {
                count_nested_group_ops(part, counts);
            }
        }
        Expr::Repeat { expr, .. } => count_nested_group_ops(expr, counts),
        Expr::Shared(expr) => count_nested_group_ops(expr, counts),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
    }
}

#[derive(Clone, Copy)]
struct SubexpressionScanInfo {
    fingerprint: u64,
    size: usize,
}

struct SubexpressionCount<'a> {
    expr: &'a Expr,
    count: usize,
    size: usize,
}

fn scan_all_subexpressions<'a>(
    expr: &'a Expr,
    infos: &mut Vec<SubexpressionScanInfo>,
    counts_by_fingerprint: &mut FxHashMap<u64, Vec<SubexpressionCount<'a>>>,
    min_size: usize,
) -> SubexpressionScanInfo {
    let info_index = infos.len();
    infos.push(SubexpressionScanInfo { fingerprint: 0, size: 0 });
    let mut hasher = FxHasher::default();
    let size = match expr {
        Expr::U8Seq(bytes) => { 0u8.hash(&mut hasher); bytes.hash(&mut hasher); 1 }
        Expr::U8Class(class) => { 1u8.hash(&mut hasher); class.hash(&mut hasher); 1 }
        Expr::Dfa(dfa) => {
            2u8.hash(&mut hasher);
            (Arc::as_ptr(dfa) as usize).hash(&mut hasher);
            1
        }
        Expr::Intersect { expr, intersect } => {
            3u8.hash(&mut hasher);
            let left = scan_all_subexpressions(expr, infos, counts_by_fingerprint, min_size);
            let right = scan_all_subexpressions(intersect, infos, counts_by_fingerprint, min_size);
            left.fingerprint.hash(&mut hasher); right.fingerprint.hash(&mut hasher);
            1 + left.size + right.size
        }
        Expr::Seq(parts) => {
            4u8.hash(&mut hasher); parts.len().hash(&mut hasher);
            let mut size = 1usize;
            for part in parts {
                let child = scan_all_subexpressions(part, infos, counts_by_fingerprint, min_size);
                child.fingerprint.hash(&mut hasher); size = size.saturating_add(child.size);
            }
            size
        }
        Expr::Choice(parts) => {
            5u8.hash(&mut hasher); parts.len().hash(&mut hasher);
            let mut size = 1usize;
            for part in parts {
                let child = scan_all_subexpressions(part, infos, counts_by_fingerprint, min_size);
                child.fingerprint.hash(&mut hasher); size = size.saturating_add(child.size);
            }
            size
        }
        Expr::Exclude { expr, exclude } => {
            6u8.hash(&mut hasher);
            let left = scan_all_subexpressions(expr, infos, counts_by_fingerprint, min_size);
            let right = scan_all_subexpressions(exclude, infos, counts_by_fingerprint, min_size);
            left.fingerprint.hash(&mut hasher); right.fingerprint.hash(&mut hasher);
            1 + left.size + right.size
        }
        Expr::Repeat { expr, min, max } => {
            7u8.hash(&mut hasher); min.hash(&mut hasher); max.hash(&mut hasher);
            let child = scan_all_subexpressions(expr, infos, counts_by_fingerprint, min_size);
            child.fingerprint.hash(&mut hasher); 1 + child.size
        }
        Expr::Shared(inner) => {
            8u8.hash(&mut hasher);
            let child = scan_all_subexpressions(inner, infos, counts_by_fingerprint, min_size);
            child.fingerprint.hash(&mut hasher); 1 + child.size
        }
        Expr::Epsilon => { 9u8.hash(&mut hasher); 1 }
    };
    let info = SubexpressionScanInfo { fingerprint: hasher.finish(), size };
    infos[info_index] = info;
    if size >= min_size {
        let bucket = counts_by_fingerprint.entry(info.fingerprint).or_default();
        if let Some(existing) = bucket.iter_mut().find(|entry| entry.expr == expr) {
            existing.count += 1;
        } else {
            bucket.push(SubexpressionCount { expr, count: 1, size });
        }
    }
    info
}

fn collect_maximal_repeated_subexpressions_fast(
    expr: &Expr,
    infos: &[SubexpressionScanInfo],
    cursor: &mut usize,
    candidate_fingerprints: &FxHashSet<u64>,
    candidates: &FxHashSet<Expr>,
    selected: &mut FxHashSet<Expr>,
) {
    let info = infos[*cursor];
    *cursor += 1;
    if candidate_fingerprints.contains(&info.fingerprint) && candidates.contains(expr) {
        selected.insert(expr.clone());
        *cursor += info.size.saturating_sub(1);
        return;
    }
    match expr {
        Expr::Exclude { expr, exclude } => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
            collect_maximal_repeated_subexpressions_fast(exclude, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::Intersect { expr, intersect } => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
            collect_maximal_repeated_subexpressions_fast(intersect, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::Seq(parts) | Expr::Choice(parts) => for part in parts {
            collect_maximal_repeated_subexpressions_fast(part, infos, cursor, candidate_fingerprints, candidates, selected);
        },
        Expr::Repeat { expr, .. } => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::Shared(expr) => {
            collect_maximal_repeated_subexpressions_fast(expr, infos, cursor, candidate_fingerprints, candidates, selected);
        }
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
    }
}

fn replace_compiled_subexpressions(
    expr: Expr,
    compiled: &FxHashMap<Expr, Arc<DFA>>,
    replace_root: bool,
) -> Expr {
    if replace_root && let Some(dfa) = compiled.get(&expr) {
        return Expr::Dfa(Arc::clone(dfa));
    }
    match expr {
        Expr::Exclude { expr, exclude } => Expr::Exclude {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            exclude: Box::new(replace_compiled_subexpressions(*exclude, compiled, true)),
        },
        Expr::Intersect { expr, intersect } => Expr::Intersect {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            intersect: Box::new(replace_compiled_subexpressions(*intersect, compiled, true)),
        },
        Expr::Seq(parts) => Expr::Seq(
            parts
                .into_iter()
                .map(|part| replace_compiled_subexpressions(part, compiled, true))
                .collect(),
        ),
        Expr::Choice(parts) => Expr::Choice(
            parts
                .into_iter()
                .map(|part| replace_compiled_subexpressions(part, compiled, true))
                .collect(),
        ),
        Expr::Repeat { expr, min, max } => Expr::Repeat {
            expr: Box::new(replace_compiled_subexpressions(*expr, compiled, true)),
            min,
            max,
        },
        Expr::Shared(inner) => {
            let rewritten = replace_compiled_subexpressions((*inner).clone(), compiled, true);
            if rewritten == *inner {
                Expr::Shared(inner)
            } else {
                rewritten
            }
        }
        leaf @ (Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon) => leaf,
    }
}

pub(super) fn materialize_repeated_subexpression_dfas_with_limits(
    exprs: &[Expr],
    min_size: usize,
    min_occurrences: usize,
) -> Option<Vec<Expr>> {
    debug_assert!(min_size > 1);
    debug_assert!(min_occurrences > 1);
    let started_at = Instant::now();
    let mut infos = Vec::<SubexpressionScanInfo>::new();
    let mut counts_by_fingerprint = FxHashMap::<u64, Vec<SubexpressionCount<'_>>>::default();
    for expr in exprs {
        scan_all_subexpressions(expr, &mut infos, &mut counts_by_fingerprint, min_size);
    }
    let mut occurrences = FxHashMap::<Expr, usize>::default();
    let mut candidate_fingerprints = FxHashSet::<u64>::default();
    for (&fingerprint, bucket) in &counts_by_fingerprint {
        for entry in bucket {
            if entry.count >= min_occurrences
                && entry.size >= min_size
                && !expr_contains_group_op(entry.expr)
                && !expression_contains_large_bounded_repeat(entry.expr)
                && !matches!(entry.expr, Expr::Dfa(_))
            {
                candidate_fingerprints.insert(fingerprint);
                occurrences.insert(entry.expr.clone(), entry.count);
            }
        }
    }
    let candidates = occurrences.keys().cloned().collect::<FxHashSet<_>>();
    let mut selected = FxHashSet::default();
    let mut cursor = 0usize;
    for expr in exprs {
        collect_maximal_repeated_subexpressions_fast(
            expr,
            &infos,
            &mut cursor,
            &candidate_fingerprints,
            &candidates,
            &mut selected,
        );
    }
    debug_assert_eq!(cursor, infos.len());
    if selected.is_empty() {
        return None;
    }

    let selected = selected.into_iter().collect::<Vec<_>>();
    let compile_started_at = Instant::now();
    let compiled_entries = selected
        .into_par_iter()
        .map(|expr| {
            let expr_size = expr_structural_size(&expr);
            let occurrence_count = occurrences.get(&expr).copied().unwrap_or(0);
            let entry_started_at = Instant::now();
            let dfa = Arc::new(compile_expr_to_dfa(&expr));
            let entry_compile_ms = entry_started_at.elapsed().as_secs_f64() * 1000.0;
            (expr, dfa, expr_size, occurrence_count, entry_compile_ms)
        })
        .collect::<Vec<_>>();
    let compile_ms = compile_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut compiled = FxHashMap::<Expr, Arc<DFA>>::default();
    for (expr, dfa, expr_size, occurrence_count, entry_compile_ms) in compiled_entries {
        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][tokenizer] shared_subexpr_dfa_entry size={} occurrences={} states={} transitions={} compile_ms={:.3}",
                expr_size,
                occurrence_count,
                dfa.num_states(),
                dfa_transition_count(&dfa),
                entry_compile_ms,
            );
        }
        compiled.insert(expr, dfa);
    }
    let rewritten = exprs
        .iter()
        .cloned()
        .map(|expr| replace_compiled_subexpressions(expr, &compiled, true))
        .collect::<Vec<_>>();
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_subexpr_dfas entries={} compile_ms={:.3} total_ms={:.3}",
            compiled.len(),
            compile_ms,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(rewritten)
}

pub(super) fn materialize_repeated_subexpression_dfas(exprs: &[Expr]) -> Option<Vec<Expr>> {
    if std::env::var_os("GLRMASK_DISABLE_SHARED_SUBEXPR_DFA_CACHE").is_some() {
        return None;
    }
    let forced = std::env::var_os("GLRMASK_FORCE_SHARED_SUBEXPR_DFA_CACHE").is_some();
    let min_total_size = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_TOTAL_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(40_000);
    let total_size = exprs.iter().map(expr_structural_size).sum::<usize>();
    if !forced && total_size < min_total_size {
        return None;
    }
    let min_size = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(80);
    let min_occurrences = std::env::var("GLRMASK_SHARED_SUBEXPR_MIN_OCCURRENCES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 1)
        .unwrap_or(12);
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_subexpr_scan forced={} total_size={} min_total_size={} min_size={} min_occurrences={}",
            forced, total_size, min_total_size, min_size, min_occurrences,
        );
    }
    materialize_repeated_subexpression_dfas_with_limits(exprs, min_size, min_occurrences)
}

pub(super) fn shared_duplicate_nested_group_op_cache(
    exprs: &[Expr],
    grouped: &BTreeMap<u32, Vec<usize>>,
) -> Option<Arc<SharedDuplicateNestedGroupOpCache>> {
    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let singleton_partitions = grouped
        .values()
        .filter(|terminal_ids| terminal_ids.len() == 1)
        .count();
    // The original policy targets many singleton partitions.  Product-trace
    // builds also have a distinct expensive shape: one large partition whose
    // nested group-op plan is itself parallelized.  Private per-expression
    // caches make duplicated nested exclusions/intersections in that partition
    // compile repeatedly.  Detect exactly the same minimum work threshold used
    // by the parallel planner before paying for the cross-partition duplicate
    // scan, so small/ordinary partitions keep the old zero-overhead path.
    const PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS: usize = 16;
    const PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS: usize = 128;
    let has_large_nested_partition = grouped.values().any(|terminal_ids| {
        terminal_ids.len() >= PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS
            && terminal_ids
                .iter()
                .map(|&terminal_id| group_op_node_count(&exprs[terminal_id]))
                .sum::<usize>()
                >= PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS
    });
    let force = std::env::var_os(
        "GLRMASK_EXPERIMENT_PRODUCT_TRACE_SHARED_NESTED_GROUP_OPS",
    )
    .is_some();
    let many_singleton_partitions =
        grouped.len() >= 64 && singleton_partitions * 4 >= grouped.len() * 3;
    if !force && !many_singleton_partitions && !has_large_nested_partition {
        return None;
    }

    let mut counts = FxHashMap::<Expr, usize>::default();
    for terminal_ids in grouped.values() {
        for &terminal_id in terminal_ids {
            let (base, excluded, intersections) = split_top_level_group_ops(&exprs[terminal_id]);
            count_nested_group_ops(&base, &mut counts);
            for expr in &excluded {
                count_nested_group_ops(expr, &mut counts);
            }
            for expr in &intersections {
                count_nested_group_ops(expr, &mut counts);
            }
        }
    }
    let duplicated = counts
        .into_iter()
        .filter_map(|(expr, count)| (count > 1).then_some(expr))
        .collect::<FxHashSet<_>>();
    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_duplicate_nested_ops partitions={} singleton_partitions={} duplicated={} scan_ms={:.3}",
            grouped.len(),
            singleton_partitions,
            duplicated.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (!duplicated.is_empty()).then(|| {
        Arc::new(SharedDuplicateNestedGroupOpCache {
            duplicated,
            compiled: Mutex::new(FxHashMap::default()),
        })
    })
}

fn expr_structural_size(expr: &Expr) -> usize {
    match expr {
        Expr::Exclude { expr, exclude } => {
            1 + expr_structural_size(expr) + expr_structural_size(exclude)
        }
        Expr::Intersect { expr, intersect } => {
            1 + expr_structural_size(expr) + expr_structural_size(intersect)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => {
            1 + parts.iter().map(expr_structural_size).sum::<usize>()
        }
        Expr::Repeat { expr, .. } => 1 + expr_structural_size(expr),
        Expr::Shared(expr) => 1 + expr_structural_size(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => 1,
    }
}

/// Populate every cross-partition nested-group-op cache entry before the
/// partition compilation itself fans out over Rayon.
///
/// Lazy `OnceLock` initialization inside the parallel partition loop can
/// deadlock: several workers may block on one cache entry while the worker
/// initializing it launches nested Rayon work that requires those same
/// workers. Compile proper subexpressions first, then their parents, so nested
/// shared entries are already available while a parent is materialized.
pub(super) fn prewarm_shared_duplicate_nested_group_ops(
    shared: &Arc<SharedDuplicateNestedGroupOpCache>,
) {
    fn compile_materialized_operand(
        expr: &Expr,
        shared: &Arc<SharedDuplicateNestedGroupOpCache>,
    ) -> Arc<DFA> {
        let mut cache = NestedGroupOpCache {
            shared_duplicates: Some(Arc::clone(shared)),
            allow_shared_initialization: false,
            ..NestedGroupOpCache::default()
        };
        let materialized = materialize_nested_group_ops(expr.clone(), &mut cache);
        match materialized {
            Expr::Dfa(dfa) => dfa,
            other => Arc::new(compile_single_expr_dfa(&other)),
        }
    }

    fn try_compile_direct_exclusion(
        expr: &Expr,
        shared: &Arc<SharedDuplicateNestedGroupOpCache>,
    ) -> Option<Arc<DFA>> {
        let Expr::Exclude { expr: left, exclude: right } = expr else {
            return None;
        };
        let (left, right) = rayon::join(
            || compile_materialized_operand(left, shared),
            || compile_materialized_operand(right, shared),
        );
        build_dense_binary_exclusion_dfa(
            left.as_ref(),
            right.as_ref(),
            expr_u8set(expr),
        )
        .map(Arc::new)
    }

    fn collect_proper_shared_dependencies(
        expr: &Expr,
        duplicated: &FxHashSet<Expr>,
        out: &mut FxHashSet<Expr>,
        is_root: bool,
    ) {
        if !is_root
            && matches!(expr, Expr::Exclude { .. } | Expr::Intersect { .. })
            && duplicated.contains(expr)
        {
            out.insert(expr.clone());
        }
        match expr {
            Expr::Exclude { expr, exclude } => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
                collect_proper_shared_dependencies(exclude, duplicated, out, false);
            }
            Expr::Intersect { expr, intersect } => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
                collect_proper_shared_dependencies(intersect, duplicated, out, false);
            }
            Expr::Seq(parts) | Expr::Choice(parts) => {
                for part in parts {
                    collect_proper_shared_dependencies(part, duplicated, out, false);
                }
            }
            Expr::Repeat { expr, .. } => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
            }
            Expr::Shared(expr) => {
                collect_proper_shared_dependencies(expr, duplicated, out, false);
            }
            Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => {}
        }
    }

    let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let started_at = profile.then(Instant::now);
    let mut pending = shared
        .duplicated
        .iter()
        .cloned()
        .map(|expr| {
            let mut dependencies = FxHashSet::default();
            collect_proper_shared_dependencies(
                &expr,
                &shared.duplicated,
                &mut dependencies,
                true,
            );
            (expr, dependencies.into_iter().collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    let mut finished = FxHashSet::<Expr>::default();
    let mut layer_count = 0usize;

    while !pending.is_empty() {
        let mut ready = Vec::<Expr>::new();
        let mut blocked = Vec::new();
        for (expr, dependencies) in pending {
            if dependencies.iter().all(|dependency| finished.contains(dependency)) {
                ready.push(expr);
            } else {
                blocked.push((expr, dependencies));
            }
        }
        assert!(
            !ready.is_empty(),
            "duplicated nested group-op dependency graph must be acyclic",
        );
        layer_count += 1;

        ready.par_iter().for_each(|expr| {
            let cell = shared
                .cell_if_duplicated(expr)
                .expect("prewarm root must be registered as duplicated");
            if cell.get().is_some() {
                return;
            }
            let mut cache = NestedGroupOpCache {
                shared_duplicates: Some(Arc::clone(shared)),
                allow_shared_initialization: false,
                ..NestedGroupOpCache::default()
            };
            let compiled = try_compile_direct_exclusion(expr, shared).unwrap_or_else(|| {
                Arc::new(compile_with_plan(
                    build_exclusion_compile_plan_with_labels_and_cache(
                        std::slice::from_ref(expr),
                        None,
                        &mut cache,
                    ),
                ))
            });
            let _ = cell.set(compiled);
        });
        finished.extend(ready);
        pending = blocked;
    }

    if let Some(started_at) = started_at {
        eprintln!(
            "[glrmask/profile][tokenizer] shared_nested_prewarm entries={} layers={} elapsed_ms={:.3}",
            finished.len(),
            layer_count,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

pub(super) struct ExclusionCompilePlan {
    pub(super) compiled_exprs: Vec<Expr>,
    pub(super) exclusions: BTreeMap<u32, BTreeSet<u32>>,
    pub(super) intersections: BTreeMap<u32, BTreeSet<u32>>,
    pub(super) visible_groups: usize,
    pub(super) profile_labels: Option<Vec<ProductComponentProfileLabel>>,
    /// Keep small product-construction kernels on the calling worker instead of
    /// entering nested Rayon regions. Used by request-local auxiliary lexers
    /// that are themselves built inside a parallel compiler DAG.
    pub(super) local_small_product: bool,
}

fn expr_is_shared(expr: &Expr) -> bool {
    match expr {
        Expr::Shared(_) => true,
        Expr::Exclude { expr, exclude } => expr_is_shared(expr) || expr_is_shared(exclude),
        Expr::Intersect { expr, intersect } => expr_is_shared(expr) || expr_is_shared(intersect),
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(expr_is_shared),
        Expr::Repeat { expr, .. } => expr_is_shared(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
    }
}

pub(super) fn expr_profile_summary(expr: &Expr) -> String {
    const MAX_LEN: usize = 80;
    let mut summary = format!("{:?}", expr);
    if summary.len() > MAX_LEN {
        summary.truncate(MAX_LEN - 3);
        summary.push_str("...");
    }
    summary
}

pub(super) fn build_exclusion_compile_plan_with_labels(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
) -> ExclusionCompilePlan {
    let mut nested_group_op_cache = NestedGroupOpCache::default();
    let plan = build_exclusion_compile_plan_with_labels_and_cache(
        exprs,
        visible_labels,
        &mut nested_group_op_cache,
    );
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
        && nested_group_op_cache.cache_misses > 0
    {
        eprintln!(
            "[glrmask/profile][tokenizer] nested_group_ops cache_entries={} cache_hits={} cache_misses={} compiled_ms={:.3} max_compile_ms={:.3}",
            nested_group_op_cache.compiled.len(),
            nested_group_op_cache.cache_hits,
            nested_group_op_cache.cache_misses,
            nested_group_op_cache.compiled_ms,
            nested_group_op_cache.max_compile_ms,
        );
    }
    plan
}

pub(super) fn build_exclusion_compile_plan_with_labels_and_cache(
    exprs: &[Expr],
    visible_labels: Option<&[String]>,
    nested_group_op_cache: &mut NestedGroupOpCache,
) -> ExclusionCompilePlan {
    let visible_groups = exprs.len();
    let mut compiled_exprs = Vec::with_capacity(visible_groups);
    let mut deferred_exclusions = Vec::<Vec<Expr>>::with_capacity(visible_groups);
    let mut deferred_intersections = Vec::<Vec<Expr>>::with_capacity(visible_groups);
    let mut profile_labels = visible_labels.map(|_| Vec::with_capacity(visible_groups));

    if let Some(labels) = visible_labels {
        assert_eq!(
            labels.len(),
            visible_groups,
            "visible profile labels must match expression count"
        );
    }

    // Parallelize nested group-op materialization only when there is enough
    // actual set-operation work to amortize per-expression caches and Rayon
    // scheduling. `visible_groups` alone is a poor proxy: a medium-sized
    // partition can contain hundreds of nested exclusions, while a much larger
    // literal partition may contain none.
    const PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS: usize = 16;
    const PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS: usize = 128;
    let nested_group_ops = (visible_groups >= PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS)
        .then(|| exprs.iter().map(group_op_node_count).sum::<usize>())
        .unwrap_or(0);
    let shared_is_prewarmed = nested_group_op_cache
        .shared_duplicates
        .as_ref()
        .is_none_or(|shared| shared.all_entries_initialized());
    let parallel_materialize = visible_groups >= PARALLEL_NESTED_GROUP_PLAN_MIN_VISIBLE_GROUPS
        && nested_group_ops >= PARALLEL_NESTED_GROUP_PLAN_MIN_GROUP_OPS
        && shared_is_prewarmed
        && std::env::var_os("GLRMASK_DISABLE_PARALLEL_NESTED_GROUP_PLAN").is_none();
    if parallel_materialize {
        let shared_duplicates = nested_group_op_cache.shared_duplicates.clone();
        let materialized = exprs
            .par_iter()
            .map(|expr| {
                let mut local_cache = NestedGroupOpCache {
                    shared_duplicates: shared_duplicates.clone(),
                    allow_shared_initialization: false,
                    ..NestedGroupOpCache::default()
                };
                let (base, excluded, intersections) = split_top_level_group_ops(expr);
                let base = materialize_nested_group_ops(base, &mut local_cache);
                let excluded = excluded
                    .into_iter()
                    .map(|expr| materialize_nested_group_ops(expr, &mut local_cache))
                    .collect::<Vec<_>>();
                let intersections = intersections
                    .into_iter()
                    .map(|expr| materialize_nested_group_ops(expr, &mut local_cache))
                    .collect::<Vec<_>>();
                (base, excluded, intersections, local_cache)
            })
            .collect::<Vec<_>>();
        for (index, (base, excluded, intersections, local_cache)) in
            materialized.into_iter().enumerate()
        {
            nested_group_op_cache.cache_hits += local_cache.cache_hits;
            nested_group_op_cache.cache_misses += local_cache.cache_misses;
            nested_group_op_cache.compiled_ms += local_cache.compiled_ms;
            nested_group_op_cache.max_compile_ms = nested_group_op_cache
                .max_compile_ms
                .max(local_cache.max_compile_ms);
            assert!(
                !expr_contains_group_op(&base),
                "Expr::Exclude and Expr::Intersect are currently only supported at the top level of a terminal expression"
            );
            for excluded_expr in &excluded {
                assert!(
                    !expr_contains_group_op(excluded_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an exclusion branch is not supported"
                );
            }
            for intersection_expr in &intersections {
                assert!(
                    !expr_contains_group_op(intersection_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an intersection branch is not supported"
                );
            }
            compiled_exprs.push(base);
            if let (Some(labels), Some(profile_labels)) = (visible_labels, profile_labels.as_mut()) {
                profile_labels.push(ProductComponentProfileLabel {
                    name: labels[index].clone(),
                    origin: "visible",
                    shared: expr_is_shared(&exprs[index]),
                });
            }
            deferred_exclusions.push(excluded);
            deferred_intersections.push(intersections);
        }
    } else {
        for (index, expr) in exprs.iter().enumerate() {
            let (base, excluded, intersections) = split_top_level_group_ops(expr);
            let base = materialize_nested_group_ops(base, nested_group_op_cache);
            let excluded = excluded
                .into_iter()
                .map(|expr| materialize_nested_group_ops(expr, nested_group_op_cache))
                .collect::<Vec<_>>();
            let intersections = intersections
                .into_iter()
                .map(|expr| materialize_nested_group_ops(expr, nested_group_op_cache))
                .collect::<Vec<_>>();
            assert!(
                !expr_contains_group_op(&base),
                "Expr::Exclude and Expr::Intersect are currently only supported at the top level of a terminal expression"
            );
            for excluded_expr in &excluded {
                assert!(
                    !expr_contains_group_op(excluded_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an exclusion branch is not supported"
                );
            }
            for intersection_expr in &intersections {
                assert!(
                    !expr_contains_group_op(intersection_expr),
                    "nested Expr::Exclude/Expr::Intersect inside an intersection branch is not supported"
                );
            }
            compiled_exprs.push(base);
            if let (Some(labels), Some(profile_labels)) = (visible_labels, profile_labels.as_mut()) {
                profile_labels.push(ProductComponentProfileLabel {
                    name: labels[index].clone(),
                    origin: "visible",
                    shared: expr_is_shared(expr),
                });
            }
            deferred_exclusions.push(excluded);
            deferred_intersections.push(intersections);
        }
    }

    let mut exclusions = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut intersections = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut next_group = visible_groups as u32;
    for (group_id, (excluded_exprs, intersection_exprs)) in deferred_exclusions
        .into_iter()
        .zip(deferred_intersections.into_iter())
        .enumerate()
    {
        let exclusion_entry = exclusions.entry(group_id as u32).or_default();
        for (excluded_index, excluded_expr) in excluded_exprs.into_iter().enumerate() {
            let is_shared = expr_is_shared(&excluded_expr);
            compiled_exprs.push(excluded_expr);
            exclusion_entry.insert(next_group);
            if let Some(profile_labels) = profile_labels.as_mut() {
                let base_name = profile_labels[group_id].name.clone();
                profile_labels.push(ProductComponentProfileLabel {
                    name: format!("{}::exclude#{}", base_name, excluded_index),
                    origin: "internal_exclusion",
                    shared: is_shared,
                });
            }
            next_group += 1;
        }

        let intersection_entry = intersections.entry(group_id as u32).or_default();
        for (intersection_index, intersection_expr) in intersection_exprs.into_iter().enumerate() {
            let is_shared = expr_is_shared(&intersection_expr);
            compiled_exprs.push(intersection_expr);
            intersection_entry.insert(next_group);
            if let Some(profile_labels) = profile_labels.as_mut() {
                let base_name = profile_labels[group_id].name.clone();
                profile_labels.push(ProductComponentProfileLabel {
                    name: format!("{}::intersect#{}", base_name, intersection_index),
                    origin: "internal_intersection",
                    shared: is_shared,
                });
            }
            next_group += 1;
        }
    }

    exclusions.retain(|_, v| !v.is_empty());
    intersections.retain(|_, v| !v.is_empty());

    ExclusionCompilePlan {
        compiled_exprs,
        exclusions,
        intersections,
        visible_groups,
        profile_labels,
        local_small_product: false,
    }
}

pub(super) fn build_exclusion_compile_plan(exprs: &[Expr]) -> ExclusionCompilePlan {
    build_exclusion_compile_plan_with_labels(exprs, None)
}
