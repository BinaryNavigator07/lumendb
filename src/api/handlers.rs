use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{index::params::HnswParams, metrics::Metric, LumenEngine};

use super::{error::ApiError, AppState};

#[derive(Serialize)]
pub struct HealthResp {
    status:  &'static str,
    version: &'static str,
}

pub async fn health() -> Json<HealthResp> {
    Json(HealthResp {
        status:  "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[derive(Deserialize)]
pub struct CreateCollectionReq {
    pub name:            String,
    pub dim:             usize,
    pub metric:          Metric,
    #[serde(default)]
    pub m:               Option<usize>,
    #[serde(default)]
    pub ef_construction: Option<usize>,
    #[serde(default)]
    pub ef_search:       Option<usize>,
}

#[derive(Serialize)]
pub struct CollectionInfo {
    name:   String,
    dim:    usize,
    metric: Metric,
    count:  usize,
}

pub async fn create_collection(
    State(state): State<AppState>,
    Json(req):    Json<CreateCollectionReq>,
) -> Result<(StatusCode, Json<CollectionInfo>), ApiError> {
    if req.name.is_empty() || req.name.contains(['/', '\\', '.']) {
        return Err(ApiError::BadRequest(
            "collection name must be non-empty and must not contain '/', '\\' or '.'".into(),
        ));
    }
    if req.dim == 0 {
        return Err(ApiError::BadRequest("dim must be > 0".into()));
    }

    let col_path = state.base_dir.join(&req.name);

    {
        let cols = state.collections.read();
        if cols.contains_key(&req.name) {
            return Err(ApiError::Conflict(format!(
                "collection '{}' is already loaded",
                req.name
            )));
        }
    }
    if col_path.exists() {
        return Err(ApiError::Conflict(format!(
            "collection '{}' already exists on disk — use GET /v1/collections/{} to inspect it",
            req.name, req.name
        )));
    }

    let params = HnswParams::builder()
        .m(req.m.unwrap_or(16))
        .ef_construction(req.ef_construction.unwrap_or(200))
        .ef_search(req.ef_search.unwrap_or(50))
        .build();

    let metric = req.metric;
    let dim    = req.dim;
    let name   = req.name.clone();

    let engine = tokio::task::spawn_blocking(move || {
        LumenEngine::open(&col_path, params, metric, dim)
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))?
    .map_err(ApiError::from)?;

    let engine = Arc::new(engine);
    let info   = CollectionInfo {
        name:   name.clone(),
        dim:    engine.dim(),
        metric: engine.index.metric,
        count:  engine.len(),
    };

    state.collections.write().insert(name, engine);
    Ok((StatusCode::CREATED, Json(info)))
}

pub async fn get_collection(
    State(state): State<AppState>,
    Path(name):   Path<String>,
) -> Result<Json<CollectionInfo>, ApiError> {
    let engine = require_collection(&state, &name)?;
    Ok(Json(CollectionInfo {
        name,
        dim:    engine.dim(),
        metric: engine.index.metric,
        count:  engine.len(),
    }))
}

pub async fn delete_collection(
    State(state): State<AppState>,
    Path(name):   Path<String>,
) -> Result<StatusCode, ApiError> {
    let removed = state.collections.write().remove(&name);
    if removed.is_none() {
        return Err(ApiError::NotFound(format!("collection '{name}' not found")));
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
pub struct InsertReq {
    pub vector:   Vec<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Serialize)]
pub struct InsertResp {
    pub id: usize,
}

pub async fn insert_vector(
    State(state): State<AppState>,
    Path(name):   Path<String>,
    Json(req):    Json<InsertReq>,
) -> Result<Json<InsertResp>, ApiError> {
    let engine = require_collection(&state, &name)?;
    let id = tokio::task::spawn_blocking(move || engine.insert(req.vector, req.metadata))
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .map_err(ApiError::from)?;
    Ok(Json(InsertResp { id }))
}

#[derive(Deserialize)]
pub struct SearchReq {
    pub vector: Vec<f32>,
    pub k:      usize,
}

#[derive(Serialize)]
pub struct SearchHitResp {
    pub id:       usize,
    pub distance: f32,
    pub metadata: Value,
}

#[derive(Serialize)]
pub struct SearchResp {
    pub results: Vec<SearchHitResp>,
}

pub async fn search_vectors(
    State(state): State<AppState>,
    Path(name):   Path<String>,
    Json(req):    Json<SearchReq>,
) -> Result<Json<SearchResp>, ApiError> {
    if req.k == 0 {
        return Err(ApiError::BadRequest("k must be > 0".into()));
    }
    let engine = require_collection(&state, &name)?;
    let hits = tokio::task::spawn_blocking(move || engine.search(&req.vector, req.k))
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .map_err(ApiError::from)?;

    let results = hits
        .into_iter()
        .map(|h| SearchHitResp { id: h.id, distance: h.distance, metadata: h.metadata })
        .collect();
    Ok(Json(SearchResp { results }))
}

#[derive(Serialize)]
pub struct VectorMetaResp {
    pub id:       usize,
    pub metadata: Value,
}

pub async fn get_vector_meta(
    State(state):     State<AppState>,
    Path((name, id)): Path<(String, usize)>,
) -> Result<Json<VectorMetaResp>, ApiError> {
    let engine = require_collection(&state, &name)?;
    let meta = tokio::task::spawn_blocking(move || engine.vault.get_metadata(id))
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .map_err(ApiError::from)?
        .unwrap_or(Value::Null);
    Ok(Json(VectorMetaResp { id, metadata: meta }))
}

fn require_collection(state: &AppState, name: &str) -> Result<Arc<LumenEngine>, ApiError> {
    state
        .collections
        .read()
        .get(name)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("collection '{name}' not found")))
}
