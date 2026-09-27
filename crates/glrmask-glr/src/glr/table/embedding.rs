//! Semantic properties retained when a compiled grammar is embedded.

use serde::{Deserialize, Serialize};

use super::GLRTable;

/// Embedding semantics are independent of diagnostic names and execution rows.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddedStart {
    nullable: bool,
    end_token_ids: Vec<u32>,
}

impl GLRTable {
    /// Whether the source start symbol derives epsilon as an embedded child.
    /// Standalone generation still requires at least one committed token.
    pub fn embedded_start_nullable(&self) -> bool {
        self.embedded_start.nullable
    }

    pub fn set_embedded_start_nullable(&mut self, nullable: bool) {
        self.embedded_start.nullable = nullable;
    }

    /// Model IDs appended as grammar-level end tokens, rather than token slots.
    pub fn embedded_end_token_ids(&self) -> Vec<u32> {
        self.embedded_start.end_token_ids.clone()
    }

    pub fn set_embedded_end_token_ids(&mut self, token_ids: &[u32]) {
        let ids = &mut self.embedded_start.end_token_ids;
        ids.clear();
        ids.extend_from_slice(token_ids);
        ids.sort_unstable();
        ids.dedup();
    }
}

#[cfg(test)]
mod tests;
