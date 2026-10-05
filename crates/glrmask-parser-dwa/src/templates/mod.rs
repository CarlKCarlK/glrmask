fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        let normalized = value.trim().to_ascii_lowercase();
        !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
    })
}

pub(crate) fn macro_parallelism_disabled() -> bool {
    env_flag("GLRMASK_DISABLE_MACRO_PARALLELISM").unwrap_or(false)
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

/// Commit-template automata remain an explicit experimental opt-in. They must
/// not change ordinary static compilation unless the caller requests them.
/// The disable flag wins so an operator always has an unambiguous rollback
/// switch when testing the feature.
pub fn commit_template_dfas_enabled() -> bool {
    if env_flag("GLRMASK_DISABLE_COMMIT_TEMPLATE_DFAS") == Some(true) {
        return false;
    }
    env_flag("GLRMASK_ENABLE_COMMIT_TEMPLATE_DFAS").unwrap_or(false)
}

/// Expensive compiler algorithm equivalence checks are validation work, not
/// template construction. Unit tests always run them; integration tests/CI
/// opt in explicitly. Runtime admission and input validation are independent.
pub(crate) fn compiler_template_validation_enabled() -> bool {
    compiler_template_validation_policy(cfg!(test), env_flag("GLRMASK_VALIDATE_COMPILER_TEMPLATES"))
}

fn compiler_template_validation_policy(unit_test: bool, requested: Option<bool>) -> bool {
    unit_test || requested.unwrap_or(false)
}

#[cfg(test)]
mod validation_policy_tests {
    #[test]
    fn production_default_off_explicit_on_tests_cannot_disable() {
        use super::compiler_template_validation_policy as enabled;
        assert!(!enabled(false, None));
        assert!(!enabled(false, Some(false)));
        assert!(enabled(false, Some(true)));
        for flag in [None, Some(false), Some(true)] { assert!(enabled(true, flag)); }
    }
}

pub(crate) mod admissibility;
pub(crate) mod characterize;
pub(crate) mod compile_bundle;
pub(crate) mod completion;
pub(crate) mod compile_dfa;

pub use compile_dfa::Templates;
