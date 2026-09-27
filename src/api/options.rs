//! Final compile/link policy; no engine implementation is exposed.



/// Preferred build/runtime trade-off.
///
/// These variants express caller intent rather than exposing GLRMask's
/// internal static/dynamic implementation choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Optimization {
    /// Let GLRMask choose automatically.
    #[default]
    Auto,
    /// Prefer low compile/link latency.
    FastBuild,
    /// Prefer lower per-token masking latency, accepting more build work.
    FastRuntime,
}

/// Options that apply only when producing a final runnable constraint.
#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    pub(super) end_tokens: Vec<u32>,
    pub(super) optimization: Optimization,
}

impl BuildOptions {
    /// Configure tokens that may terminate generation once the final
    /// constraint is accepting.
    pub fn end_tokens(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        self.end_tokens = ids.into_iter().collect();
        self
    }

    /// Select the preferred build/runtime trade-off.
    pub fn optimization(mut self, optimization: Optimization) -> Self {
        self.optimization = optimization;
        self
    }

    pub(crate) fn end_token_ids(&self) -> &[u32] {
        &self.end_tokens
    }

    pub(crate) fn optimization_value(&self) -> Optimization {
        self.optimization
    }
}
