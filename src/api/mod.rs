//! The Nexus — Axum REST gateway (Milestone 4).
//!
//! # Architecture
//!
//! ```text
//!  HTTP client
//!       │
//!  ┌────▼──────────────────────────────────────────┐
//!  │             Axum Router                        │
//!  │                                                │
//!  │  GET  /health             ─── handlers::health │
//!  │                                                │
//!  │  ── auth middleware (X-API-KEY) ──────────────  │
//!  │                                                │
//!  │  POST   /v1/collections                        │
//!  │  GET    /v1/collections/:name                  │
//!  │  DELETE /v1/collections/:name                  │
//!  │  POST   /v1/collections/:name/vectors          │
//!  │  POST   /v1/collections/:name/search           │
//!  │  GET    /v1/collections/:name/vectors/:id      │
//!  └────┬──────────────────────────────────────────┘
//!       │  Arc<LumenEngine>  (per collection)
//!  ┌────▼──────────────────────────────────────────┐
//!  │  LumenEngine  (HNSW + Sled Vault)             │
//!  └───────────────────────────────────────────────┘
//! ```

pub mod error;
pub mod handlers;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{Request, State},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Router,
};
use parking_lot::RwLock;

use crate::LumenEngine;

use error::ApiError;

// ── Shared state ──────────────────────────────────────────────────────────────

/// Thread-safe map of `collection_name → LumenEngine`.
pub type CollectionMap = Arc<RwLock<HashMap<String, Arc<LumenEngine>>>>;

/// Cloned into every Axum handler via `State<AppState>`.
#[derive(Clone)]
pub struct AppState {
    pub collections: CollectionMap,
    pub base_dir:    PathBuf,
    pub api_key:     Option<String>,
}

impl AppState {
    pub fn new(base_dir: PathBuf, api_key: Option<String>) -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            base_dir,
            api_key,
        }
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Construct the complete Axum [`Router`].
///
/// `/health` is always public.  All `/v1/…` routes are protected by the
/// `auth_layer` middleware when an API key is configured.
pub fn build_router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/v1/collections",
            post(handlers::create_collection))
        .route("/v1/collections/:name",
            get(handlers::get_collection)
            .delete(handlers::delete_collection))
        .route("/v1/collections/:name/vectors",
            post(handlers::insert_vector))
        .route("/v1/collections/:name/search",
            post(handlers::search_vectors))
        .route("/v1/collections/:name/vectors/:id",
            get(handlers::get_vector_meta))
        .layer(middleware::from_fn_with_state(state.clone(), auth_layer));

    Router::new()
        .route("/health", get(handlers::health))
        .merge(v1)
        .with_state(state)
}

// ── Auth middleware ───────────────────────────────────────────────────────────

/// Validates the `X-API-KEY` request header.
///
/// If no API key is configured on the server (`state.api_key == None`), every
/// request is allowed through.  This makes development frictionless while
/// keeping production secure.
async fn auth_layer(
    State(state): State<AppState>,
    request:      Request,
    next:         Next,
) -> Result<Response, ApiError> {
    if let Some(ref expected) = state.api_key {
        let provided = request
            .headers()
            .get("X-API-KEY")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if provided != expected.as_str() {
            return Err(ApiError::Unauthorized);
        }
    }
    Ok(next.run(request).await)
}
