//! Environment overrides for the validated boundary compiler defaults.
//!
//! The production default is the validated fast path. Environment variables
//! remain rollback/debug controls: an absent variable keeps the optimization
//! enabled, while "", 0, false, no, or off disable it explicitly.

fn value_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

pub(crate) fn enabled(name: &str) -> bool {
    std::env::var_os(name)
        .map(|value| value_enabled(&value.to_string_lossy()))
        .unwrap_or(true)
}

/// Return a bounded integer override. Absence uses the validated default;
/// an explicit zero, invalid value, or out-of-range value disables the option.
fn bounded_usize_value(
    value: Option<&str>,
    default: usize,
    min: usize,
    max: usize,
) -> Option<usize> {
    match value {
        None => Some(default),
        Some(value) => value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&value| (min..=max).contains(&value)),
    }
}

pub(crate) fn bounded_usize(
    name: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Option<usize> {
    let value = std::env::var_os(name);
    bounded_usize_value(value.as_deref().and_then(std::ffi::OsStr::to_str), default, min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_boolean_override_values_are_unambiguous() {
        for value in ["", "0", "false", "FALSE", " no ", "Off"] {
            assert!(!value_enabled(value), "{value:?}");
        }
        for value in ["1", "true", "yes", "on", "garbage"] {
            assert!(value_enabled(value), "{value:?}");
        }
    }

    #[test]
    fn validated_query_tile_default_has_explicit_disable_and_override() {
        assert_eq!(bounded_usize_value(None, 32, 1, 4096), Some(32));
        assert_eq!(bounded_usize_value(Some("0"), 32, 1, 4096), None);
        assert_eq!(bounded_usize_value(Some(""), 32, 1, 4096), None);
        assert_eq!(bounded_usize_value(Some("nope"), 32, 1, 4096), None);
        assert_eq!(bounded_usize_value(Some("4097"), 32, 1, 4096), None);
        assert_eq!(bounded_usize_value(Some(" 64 "), 32, 1, 4096), Some(64));
    }
}
