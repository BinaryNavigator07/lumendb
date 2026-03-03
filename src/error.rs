use std::fmt;

/// Unified error type for all LumenDB operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LumenError {
    // ── Metric errors (Milestone 1) ───────────────────────────────────────────
    /// Operands have different numbers of dimensions.
    DimensionMismatch { expected: usize, got: usize },

    /// A zero-magnitude vector was supplied where a unit vector is required
    /// (e.g. cosine similarity of the zero vector is undefined).
    ZeroVector,

    /// An empty slice was supplied where a non-empty one is required.
    EmptyVector,

    // ── Storage errors (Milestone 3) ──────────────────────────────────────────
    /// Wraps a Sled I/O error.
    Storage(String),

    /// Serialization / deserialization failure.
    Codec(String),

    /// The database was opened with a different configuration than expected.
    ConfigMismatch(String),
}

impl fmt::Display for LumenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected}-dimensional vector, got {got}"
            ),
            Self::ZeroVector => write!(f, "zero-magnitude vector is not valid for this operation"),
            Self::EmptyVector => write!(f, "vector must not be empty"),
            Self::Storage(msg) => write!(f, "storage error: {msg}"),
            Self::Codec(msg) => write!(f, "codec error: {msg}"),
            Self::ConfigMismatch(msg) => write!(f, "config mismatch: {msg}"),
        }
    }
}

impl std::error::Error for LumenError {}

impl From<sled::Error> for LumenError {
    fn from(e: sled::Error) -> Self {
        LumenError::Storage(e.to_string())
    }
}

impl From<bincode::Error> for LumenError {
    fn from(e: bincode::Error) -> Self {
        LumenError::Codec(e.to_string())
    }
}

impl From<serde_json::Error> for LumenError {
    fn from(e: serde_json::Error) -> Self {
        LumenError::Codec(e.to_string())
    }
}
