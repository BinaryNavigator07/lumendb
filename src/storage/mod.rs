//! The Vault — persistence layer (Milestone 3).
//!
//! # Abstraction layers
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     LumenEngine  (engine.rs)                        │
//! │                insert(vec, meta)  •  search(query, k)               │
//! └───────────────────────┬─────────────────────────────────────────────┘
//!                         │ uses
//!            ┌────────────▼────────────┐
//!            │  Vault trait (this mod) │  ← swappable: Sled / Memory / S3
//!            └────────────┬────────────┘
//!                         │ implemented by
//!                ┌────────▼────────┐
//!                │   SledVault     │  (sled_vault.rs)
//!                └─────────────────┘
//! ```

pub mod codec;
pub mod sled_vault;

pub use codec::StoredConfig;
pub use sled_vault::SledVault;

use std::path::Path;

use crate::index::node::{Node, NodeId};
use crate::LumenError;
use codec::StoredNode;

/// Pluggable persistence back-end contract.
///
/// Every method is synchronous and blocking.  Async wrappers will be added
/// in Milestone 4 when the Axum layer is integrated.
pub trait Vault: Send + Sync {
    // ── Write ─────────────────────────────────────────────────────────────────

    /// Persist a vector, its metadata, and its graph node state.
    /// **Must** complete an `fsync` before returning `Ok` — this is the WAL
    /// guarantee that makes crash recovery possible.
    fn put(
        &self,
        id: NodeId,
        vector: &[f32],
        meta: &serde_json::Value,
        node: &Node,
    ) -> Result<(), LumenError>;

    /// Update only the graph adjacency for an existing node.
    /// Called after bidirectional HNSW wiring to keep the snapshot current.
    fn update_graph_node(&self, node: &Node) -> Result<(), LumenError>;

    /// Persist the collection-level configuration.
    fn save_config(&self, cfg: &StoredConfig) -> Result<(), LumenError>;

    /// Update the entry-point and max-layer fields in the stored config.
    fn update_config_header(
        &self,
        entry_point: Option<NodeId>,
        max_layer: usize,
    ) -> Result<(), LumenError>;

    // ── Read ──────────────────────────────────────────────────────────────────

    fn get_vector(&self, id: NodeId) -> Result<Option<Vec<f32>>, LumenError>;
    fn get_metadata(&self, id: NodeId) -> Result<Option<serde_json::Value>, LumenError>;
    fn load_config(&self) -> Result<Option<StoredConfig>, LumenError>;

    /// Number of stored vectors.
    fn count(&self) -> usize;

    // ── Recovery ──────────────────────────────────────────────────────────────

    /// Drive the warm-boot recovery loop.
    ///
    /// The callback receives `(id, vector, Option<StoredNode>)`:
    /// - `Some(node)` → fast path: graph state is intact, no HNSW re-insert needed.
    /// - `None`       → slow path: graph state is missing, HNSW must re-insert.
    ///
    /// Returns the total number of entries processed.
    fn replay<F>(&self, callback: F) -> Result<usize, LumenError>
    where
        F: FnMut(NodeId, Vec<f32>, Option<StoredNode>) -> Result<(), LumenError>;

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Force all buffered writes to the OS journal.
    fn flush(&self) -> Result<(), LumenError>;

    /// Export a crash-consistent snapshot to `dest` (FR-5).
    fn snapshot_to(&self, dest: &Path) -> Result<(), LumenError>;
}

/// Implement `Vault` for `SledVault` via delegation.
impl Vault for SledVault {
    fn put(&self, id: NodeId, vec: &[f32], meta: &serde_json::Value, node: &Node) -> Result<(), LumenError> {
        SledVault::put(self, id, vec, meta, node)
    }
    fn update_graph_node(&self, node: &Node) -> Result<(), LumenError> {
        SledVault::update_graph_node(self, node)
    }
    fn save_config(&self, cfg: &StoredConfig) -> Result<(), LumenError> {
        SledVault::save_config(self, cfg)
    }
    fn update_config_header(&self, ep: Option<NodeId>, ml: usize) -> Result<(), LumenError> {
        SledVault::update_config_header(self, ep, ml)
    }
    fn get_vector(&self, id: NodeId) -> Result<Option<Vec<f32>>, LumenError> {
        SledVault::get_vector(self, id)
    }
    fn get_metadata(&self, id: NodeId) -> Result<Option<serde_json::Value>, LumenError> {
        SledVault::get_metadata(self, id)
    }
    fn load_config(&self) -> Result<Option<StoredConfig>, LumenError> {
        SledVault::load_config(self)
    }
    fn count(&self) -> usize {
        SledVault::count(self)
    }
    fn replay<F>(&self, callback: F) -> Result<usize, LumenError>
    where
        F: FnMut(NodeId, Vec<f32>, Option<StoredNode>) -> Result<(), LumenError>,
    {
        SledVault::replay(self, callback)
    }
    fn flush(&self) -> Result<(), LumenError> {
        SledVault::flush(self)
    }
    fn snapshot_to(&self, dest: &Path) -> Result<(), LumenError> {
        SledVault::snapshot_to(self, dest)
    }
}
