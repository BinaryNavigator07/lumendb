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

pub type CollectionMap = Arc<RwLock<HashMap<String, Arc<LumenEngine>>>>;

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
