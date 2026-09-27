pub mod compile;
pub(crate) mod weight_observation_quotient;
pub(crate) mod composition;
#[cfg(feature = "internal-api")]
pub(crate) mod o21137_subgrammar_bench;
pub(crate) mod constraint_possible_matches;
pub(crate) use glrmask_glr::__private::glr;
pub mod grammar;
pub(crate) mod pipeline;
pub(crate) mod vocab_partition;
pub(crate) mod pm_profile;
pub(crate) use glrmask_lexer::__private::possible_matches;
pub mod stages;

pub(crate) use compile::compile_owned;

pub(crate) fn macro_parallelism_disabled() -> bool {
    std::env::var("GLRMASK_DISABLE_MACRO_PARALLELISM")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

pub(crate) fn macro_profile_enabled() -> bool {
    macro_parallelism_disabled()
        && (std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some())
}

pub(crate) fn report_macro_item_timings(label: &str, timings_ms: &[f64]) {
    if !macro_profile_enabled() || timings_ms.is_empty() {
        return;
    }
    let mut sorted = timings_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let total_work_ms = sorted.iter().sum::<f64>();
    let max_item_ms = sorted[count - 1];
    let mean_ms = total_work_ms / count as f64;
    let p50_ms = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) * 0.5
    } else {
        sorted[count / 2]
    };
    let p90_ms = sorted[((count as f64 * 0.90).ceil() as usize).clamp(1, count) - 1];
    let ideal_parallelism = if max_item_ms > 0.0 {
        total_work_ms / max_item_ms
    } else {
        0.0
    };
    eprintln!(
        "[glrmask/profile][macro_fanout] label={label} count={count} total_work_ms={total_work_ms:.3} max_item_ms={max_item_ms:.3} mean_ms={mean_ms:.3} p50_ms={p50_ms:.3} p90_ms={p90_ms:.3} ideal_parallelism={ideal_parallelism:.2}",
    );
}

pub(crate) fn macro_join_if<A, B, Left, Right>(
    parallel: bool,
    label: &'static str,
    left: Left,
    right: Right,
) -> (A, B)
where
    A: Send,
    B: Send,
    Left: FnOnce() -> A + Send,
    Right: FnOnce() -> B + Send,
{
    if !parallel || macro_parallelism_disabled() {
        let left_started = std::time::Instant::now();
        let left = left();
        let left_ms = left_started.elapsed().as_secs_f64() * 1000.0;
        let right_started = std::time::Instant::now();
        let right = right();
        let right_ms = right_started.elapsed().as_secs_f64() * 1000.0;
        report_macro_item_timings(label, &[left_ms, right_ms]);
        (left, right)
    } else {
        rayon::join(left, right)
    }
}

pub(crate) fn macro_join<A, B, Left, Right>(
    label: &'static str,
    left: Left,
    right: Right,
) -> (A, B)
where
    A: Send,
    B: Send,
    Left: FnOnce() -> A + Send,
    Right: FnOnce() -> B + Send,
{
    macro_join_if(true, label, left, right)
}

/// Exact bounded-terminal synthesis is enabled by default. Runtime always keeps
/// the full exact lexer, while terminal/parser DWA construction may use a
/// certified smaller representative lexer. Retain an explicit opt-out only for
/// diagnostics and performance comparisons; it must not change schema
/// semantics.
pub(crate) fn synthetic_bounded_terminals_enabled() -> bool {
    match std::env::var("GLRMASK_SYNTHETIC_BOUNDED_TERMINALS") {
        Err(_) => true,
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "invalid GLRMASK_SYNTHETIC_BOUNDED_TERMINALS={other:?}; expected one of 1/0, true/false, yes/no, or on/off"
            ),
        },
    }
}
