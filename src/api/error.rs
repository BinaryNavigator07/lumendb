use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::LumenError;

#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    Conflict(String),
    BadRequest(String),
    Internal(String),
    Unauthorized,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, msg) = match &self {
            ApiError::NotFound(m)   => (StatusCode::NOT_FOUND,                    "NOT_FOUND",      m.clone()),
            ApiError::Conflict(m)   => (StatusCode::CONFLICT,                     "CONFLICT",       m.clone()),
            ApiError::BadRequest(m) => (StatusCode::UNPROCESSABLE_ENTITY,         "BAD_REQUEST",    m.clone()),
            ApiError::Internal(m)   => (StatusCode::INTERNAL_SERVER_ERROR,        "INTERNAL_ERROR", m.clone()),
            ApiError::Unauthorized  => (StatusCode::UNAUTHORIZED,                 "UNAUTHORIZED",
                                        "invalid or missing X-API-KEY header".to_string()),
        };
        (status, Json(serde_json::json!({ "error": msg, "code": code }))).into_response()
    }
}

impl From<LumenError> for ApiError {
    fn from(e: LumenError) -> Self {
        match e {
            LumenError::DimensionMismatch { expected, got } =>
                ApiError::BadRequest(format!("dimension mismatch: expected {expected}, got {got}")),
            LumenError::ZeroVector =>
                ApiError::BadRequest("zero-magnitude vector is not valid for this operation".into()),
            LumenError::EmptyVector =>
                ApiError::BadRequest("vector must not be empty".into()),
            LumenError::ConfigMismatch(s) =>
                ApiError::BadRequest(s),
            LumenError::Storage(s) =>
                ApiError::Internal(format!("storage: {s}")),
            LumenError::Codec(s) =>
                ApiError::Internal(format!("codec: {s}")),
        }
    }
}
