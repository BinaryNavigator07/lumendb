use std::path::Path;

use crate::index::node::{Node, NodeId};
use crate::LumenError;
use super::codec::{
    decode_config, decode_meta, decode_node, decode_vector,
    encode_config, encode_meta, encode_node, encode_vector,
    id_to_key, key_to_id, StoredConfig, StoredNode,
};

const CONFIG_KEY: &[u8] = b"v1";

pub struct SledVault {
    #[allow(dead_code)]
    db:          sled::Db,
    vectors:     sled::Tree,
    metadata:    sled::Tree,
    graph_nodes: sled::Tree,
    config_tree: sled::Tree,
}

impl SledVault {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LumenError> {
        let db = sled::open(path)?;
        Ok(Self {
            vectors:     db.open_tree("vectors")?,
            metadata:    db.open_tree("metadata")?,
            graph_nodes: db.open_tree("graph_nodes")?,
            config_tree: db.open_tree("config")?,
            db,
        })
    }

    pub fn put(
        &self,
        id: NodeId,
        vector: &[f32],
        meta: &serde_json::Value,
        node: &Node,
    ) -> Result<(), LumenError> {
        let key = id_to_key(id);

        self.vectors.insert(key, encode_vector(vector))?;
        self.metadata.insert(key, encode_meta(meta)?)?;

        let stored = StoredNode {
            level:     node.level,
            neighbors: node.neighbors.clone(),
        };
        self.graph_nodes.insert(key, encode_node(&stored)?)?;

        self.db.flush()?;

        Ok(())
    }

    pub fn update_graph_node(&self, node: &Node) -> Result<(), LumenError> {
        let stored = StoredNode {
            level:     node.level,
            neighbors: node.neighbors.clone(),
        };
        self.graph_nodes.insert(id_to_key(node.id), encode_node(&stored)?)?;
        Ok(())
    }

    pub fn save_config(&self, cfg: &StoredConfig) -> Result<(), LumenError> {
        self.config_tree.insert(CONFIG_KEY, encode_config(cfg)?)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn load_config(&self) -> Result<Option<StoredConfig>, LumenError> {
        match self.config_tree.get(CONFIG_KEY)? {
            Some(bytes) => Ok(Some(decode_config(&bytes)?)),
            None        => Ok(None),
        }
    }

    pub fn update_config_header(
        &self,
        entry_point: Option<NodeId>,
        max_layer: usize,
    ) -> Result<(), LumenError> {
        if let Some(mut cfg) = self.load_config()? {
            cfg.entry_point = entry_point.map(|id| id as u64);
            cfg.max_layer   = max_layer;
            self.config_tree.insert(CONFIG_KEY, encode_config(&cfg)?)?;
        }
        Ok(())
    }

    pub fn get_vector(&self, id: NodeId) -> Result<Option<Vec<f32>>, LumenError> {
        Ok(self.vectors.get(id_to_key(id))?.map(|b| decode_vector(&b)))
    }

    pub fn get_metadata(&self, id: NodeId) -> Result<Option<serde_json::Value>, LumenError> {
        match self.metadata.get(id_to_key(id))? {
            Some(b) => Ok(Some(decode_meta(&b)?)),
            None    => Ok(None),
        }
    }

    pub fn get_graph_node(&self, id: NodeId) -> Result<Option<StoredNode>, LumenError> {
        match self.graph_nodes.get(id_to_key(id))? {
            Some(b) => Ok(Some(decode_node(&b)?)),
            None    => Ok(None),
        }
    }

    pub fn count(&self) -> usize {
        self.vectors.len()
    }

    pub fn replay<F>(&self, mut callback: F) -> Result<usize, LumenError>
    where
        F: FnMut(NodeId, Vec<f32>, Option<StoredNode>) -> Result<(), LumenError>,
    {
        let mut count = 0;

        for result in self.vectors.iter() {
            let (key, val) = result?;
            let id     = key_to_id(&key);
            let vector = decode_vector(&val);

            let graph_node = self.get_graph_node(id)?;

            callback(id, vector, graph_node)?;
            count += 1;
        }

        Ok(count)
    }

    pub fn flush(&self) -> Result<(), LumenError> {
        self.db.flush()?;
        Ok(())
    }

    pub fn snapshot_to<P: AsRef<Path>>(&self, dest: P) -> Result<(), LumenError> {
        self.db.flush()?;
        let export  = self.db.export();
        let dest_db = sled::open(dest)?;
        dest_db.import(export);
        dest_db.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn encode_decode_round_trip() {
        use crate::storage::codec::{decode_vector, encode_vector};
        let v = vec![1.0f32, -0.5, 3.14];
        assert_eq!(v, decode_vector(&encode_vector(&v)));
    }
}
