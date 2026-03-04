pub mod codec;
pub mod sled_vault;

pub use codec::StoredConfig;
pub use sled_vault::SledVault;

use std::path::Path;

use crate::index::node::{Node, NodeId};
use crate::LumenError;
use codec::StoredNode;

pub trait Vault: Send + Sync {

    fn put(
        &self,
        id: NodeId,
        vector: &[f32],
        meta: &serde_json::Value,
        node: &Node,
    ) -> Result<(), LumenError>;

    fn update_graph_node(&self, node: &Node) -> Result<(), LumenError>;

    fn save_config(&self, cfg: &StoredConfig) -> Result<(), LumenError>;

    fn update_config_header(
        &self,
        entry_point: Option<NodeId>,
        max_layer: usize,
    ) -> Result<(), LumenError>;

    fn get_vector(&self, id: NodeId) -> Result<Option<Vec<f32>>, LumenError>;
    fn get_metadata(&self, id: NodeId) -> Result<Option<serde_json::Value>, LumenError>;
    fn load_config(&self) -> Result<Option<StoredConfig>, LumenError>;

    fn count(&self) -> usize;

    fn replay<F>(&self, callback: F) -> Result<usize, LumenError>
    where
        F: FnMut(NodeId, Vec<f32>, Option<StoredNode>) -> Result<(), LumenError>;

    fn flush(&self) -> Result<(), LumenError>;

    fn snapshot_to(&self, dest: &Path) -> Result<(), LumenError>;
}

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
