//! Backed, lazily decoded compiler metadata retained for later composition.

use crate::automata::regex::Expr;
use std::sync::Arc;
#[derive(Debug, Clone)]
pub(crate) enum DeferredTerminalExprBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
    /// Dynamic worker-transfer artifacts keep the terminal-expression section
    /// compressed until a later composition actually asks for source
    /// expressions. Ordinary mask/commit execution never needs these trees.
    CompressedOwned(Arc<[u8]>),
    /// Same compressed representation, retained directly inside an owned
    /// sectioned transfer backing allocation.
    CompressedBacked {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl DeferredTerminalExprBytes {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
            Self::CompressedOwned(bytes) => bytes,
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }

    pub(crate) fn decode_exprs(&self) -> Result<Vec<Expr>, String> {
        let raw;
        let bytes = match self {
            Self::Owned(bytes) => bytes.as_ref(),
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
            Self::CompressedOwned(bytes) => {
                raw = zstd::stream::decode_all(bytes.as_ref()).map_err(|err| err.to_string())?;
                raw.as_slice()
            }
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => {
                raw = zstd::stream::decode_all(&backing[*start..*start + *len])
                    .map_err(|err| err.to_string())?;
                raw.as_slice()
            }
        };
        bincode::deserialize(bytes).map_err(|err| err.to_string())
    }

    /// Append the canonical uncompressed bincode expression section used by
    /// the self-contained Constraint artifact format. A transfer-loaded
    /// compressed blob therefore remains lazy until either composition or a
    /// later explicit save requests it.
    pub(crate) fn append_raw_serialized(&self, out: &mut Vec<u8>) -> Result<(), String> {
        match self {
            Self::Owned(bytes) => out.extend_from_slice(bytes),
            Self::Backed {
                backing,
                start,
                len,
            } => out.extend_from_slice(&backing[*start..*start + *len]),
            Self::CompressedOwned(bytes) => {
                zstd::stream::copy_decode(bytes.as_ref(), out).map_err(|err| err.to_string())?;
            }
            Self::CompressedBacked {
                backing,
                start,
                len,
            } => {
                zstd::stream::copy_decode(&backing[*start..*start + *len], out)
                    .map_err(|err| err.to_string())?;
            }
        }
        Ok(())
    }
}

/// Opaque current-format composition metadata retained without eagerly
/// rebuilding the large parser-template/characterization graphs. Ordinary
/// runtime masking never needs these bytes; constraint composition materializes
/// them on demand.
#[derive(Debug, Clone)]
pub(crate) enum DeferredCompositionMetadataBytes {
    Owned(Arc<[u8]>),
    Backed {
        backing: Arc<Vec<u8>>,
        start: usize,
        len: usize,
    },
}

impl DeferredCompositionMetadataBytes {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Backed {
                backing,
                start,
                len,
            } => &backing[*start..*start + *len],
        }
    }
}
