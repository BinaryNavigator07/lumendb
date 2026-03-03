use std::fmt;

/// All errors that can be returned by LumenDB operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LumenError {
    /// Operands have different numbers of dimensions.
    DimensionMismatch { expected: usize, got: usize },

    /// A zero-magnitude vector was supplied where a unit vector is required
    /// (e.g. cosine similarity of the zero vector is undefined).
    ZeroVector,

    /// An empty slice was supplied where a non-empty one is required.
    EmptyVector,
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
        }
    }
}

impl std::error::Error for LumenError {}
