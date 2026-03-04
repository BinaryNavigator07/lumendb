use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum LumenError {
    DimensionMismatch { expected: usize, got: usize },
    ZeroVector,
    EmptyVector,
    Storage(String),
    Codec(String),
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
