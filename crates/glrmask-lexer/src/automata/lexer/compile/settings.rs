//! Existing lexer build policy and environment overrides. Defaults are resolved at use time.

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "invalid {name}={other:?}; expected one of 1/0, true/false, yes/no, or on/off"
            ),
        },
        Err(_) => default,
    }
}

pub(super) fn adaptive_lexer_enabled() -> bool {
    match std::env::var("GLRMASK_LEXER_ADAPTIVE") {
        Ok(_) => env_flag("GLRMASK_LEXER_ADAPTIVE", false),
        // Keep the long-standing depth override useful as a one-variable
        // diagnostic opt-in even though adaptive determinization is no longer
        // enabled by default.
        Err(_) => std::env::var_os("GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH").is_some(),
    }
}

pub(super) fn adaptive_lexer_state_limit() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(32_768)
}

pub(super) fn adaptive_lexer_max_depth() -> Option<usize> {
    let Ok(value) = std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH") else {
        return Some(1);
    };
    let value = value.trim();
    if matches!(value.to_ascii_lowercase().as_str(), "full" | "unbounded") {
        return None;
    }
    Some(value.parse::<usize>().unwrap_or_else(|_| {
        panic!(
            "invalid GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH={value:?}; expected a byte depth or full"
        )
    }))
}

/// Extra product-prefix states permitted for a bounded adaptive lexer.
///
/// A bounded product retains exact copies of the independently compiled
/// component DFAs after its cutoff, so percentage growth relative to those
/// components is the wrong resource model.  At depth one the prefix can add at
/// most one state per byte (plus the shared root already present in the
/// untouched epsilon union), making 256 a tight representation-independent
/// default overhead cap. Deeper explicit experiments can raise this budget.
pub(super) fn adaptive_lexer_bounded_overhead_states() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_BOUNDED_OVERHEAD_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(256)
}

pub(super) fn adaptive_lexer_growth_percent() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_GROWTH_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(100)
}

pub(super) fn adaptive_lexer_transition_growth_percent() -> usize {
    std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_TRANSITION_GROWTH_PERCENT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(600)
}

pub(super) fn adaptive_transition_growth_is_acceptable(
    input_transitions: usize,
    output_transitions: usize,
    growth_percent: usize,
) -> bool {
    output_transitions.saturating_mul(100)
        <= input_transitions.saturating_mul(growth_percent)
}
