//! Sled-backed implementation of the Vault (FR-4, FR-5, NFR-2).
//!
//! # Sled tree layout
//!
//! ```text
//! ┌─────────────────┬──────────────────────┬────────────────────────────────┐
//! │ Tree name       │ Key                  │ Value                          │
//! ├─────────────────┼──────────────────────┼────────────────────────────────┤
//! │ vectors         │ NodeId (8 BE bytes)  │ packed LE f32 bytes            │
//! │ metadata        │ NodeId (8 BE bytes)  │ JSON bytes                     │
//! │ graph_nodes     │ NodeId (8 BE bytes)  │ bincode(StoredNode)            │
//! │ config          │ "v1" (literal bytes) │ bincode(StoredConfig)          │
//! └─────────────────┴──────────────────────┴────────────────────────────────┘
//! ```
//!
//! # WAL guarantee
//!
//! Sled maintains its own internal write-ahead log and `flush()` forces an
//! `fsync`.  The insertion order in [`SledVault::put`] is:
//!
//! 1. Write vector to `vectors` tree          (Sled WAL)
//! 2. Write metadata to `metadata` tree       (Sled WAL)
//! 3. Write graph node to `graph_nodes` tree  (Sled WAL)
//! 4. `flush()` — blocks until all three writes reach the OS journal
//!
//! If the process crashes between steps, the recovery scanner in
//! [`SledVault::replay`] handles partial writes:
//! - Entry in `vectors` but missing `graph_nodes` → re-insert via HNSW
//! - Entry in both → restore graph state directly (fast path)

use std::path::Path;

use crate::index::node::{Node, NodeId};
use crate::LumenError;
use super::codec::{
    decode_config, decode_meta, decode_node, decode_vector,
    encode_config, encode_meta, encode_node, encode_vector,
    id_to_key, key_to_id, StoredConfig, StoredNode,
};

const CONFIG_KEY: &[u8] = b"v1";

/// The persistent backing store for a single LumenDB collection.
pub struct SledVault {
    #[allow(dead_code)]
    db:          sled::Db,
    vectors:     sled::Tree,
    metadata:    sled::Tree,
    graph_nodes: sled::Tree,
    config_tree: sled::Tree,
}

impl SledVault {
    /// Open (or create) a vault at `path`.
    ///
    /// Returns a fresh, empty vault if the directory does not exist yet.
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

    // ── Write path ────────────────────────────────────────────────────────────

    /// Atomically persist a vector + metadata + graph state, then flush.
    ///
    /// The graph state (`node`) is written last.  The recovery scanner treats
    /// "vector present, graph_node absent" as a pending insertion and re-runs
    /// the HNSW algorithm for that node on warm boot.
    pub fn put(
        &self,
        id: NodeId,
        vector: &[f32],
        meta: &serde_json::Value,
        node: &Node,
    ) -> Result<(), LumenError> {
        let key = id_to_key(id);

        // 1 — Vector (must be first — presence in this tree = "committed vector")
        self.vectors.insert(key, encode_vector(vector))?;

        // 2 — Metadata
        self.metadata.insert(key, encode_meta(meta)?)?;

        // 3 — Graph adjacency (written last; absence = "needs replay")
        let stored = StoredNode {
            level:     node.level,
            neighbors: node.neighbors.clone(),
        };
        self.graph_nodes.insert(key, encode_node(&stored)?)?;

        // 4 — fsync: blocks until all three writes reach the OS journal
        self.db.flush()?;

        Ok(())
    }

    /// Overwrite only the graph state for a node that already has a vector.
    /// Called after the HNSW completes an insertion to keep the snapshot fresh.
    pub fn update_graph_node(&self, node: &Node) -> Result<(), LumenError> {
        let stored = StoredNode {
            level:     node.level,
            neighbors: node.neighbors.clone(),
        };
        self.graph_nodes.insert(id_to_key(node.id), encode_node(&stored)?)?;
        Ok(())
    }

    // ── Config ────────────────────────────────────────────────────────────────

    /// Persist the collection-level configuration.  Called once on first open.
    pub fn save_config(&self, cfg: &StoredConfig) -> Result<(), LumenError> {
        self.config_tree.insert(CONFIG_KEY, encode_config(cfg)?)?;
        self.db.flush()?;
        Ok(())
    }

    /// Load the stored configuration, or `None` if the vault is brand new.
    pub fn load_config(&self) -> Result<Option<StoredConfig>, LumenError> {
        match self.config_tree.get(CONFIG_KEY)? {
            Some(bytes) => Ok(Some(decode_config(&bytes)?)),
            None        => Ok(None),
        }
    }

    /// Persist updated entry_point and max_layer after each insertion.
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

    // ── Read path ─────────────────────────────────────────────────────────────

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

    /// Total number of stored vectors.
    pub fn count(&self) -> usize {
        self.vectors.len()
    }

    // ── Recovery (warm boot) ──────────────────────────────────────────────────

    /// Drive the warm-boot recovery loop (FR-4 / the "Replay").
    ///
    /// Iterates every entry in the `vectors` tree in insertion order (0…N).
    /// For each entry two paths are taken:
    ///
    /// - **Fast path** (graph state intact): supplies the stored `Node`
    ///   directly so the caller can bypass the HNSW insertion algorithm.
    ///
    /// - **Slow path** (graph state missing, e.g. crash mid-write):
    ///   supplies only the raw vector so the caller re-runs HNSW insertion.
    ///
    /// The callback returns `true` if it consumed the entry successfully.
    /// Returns the total number of entries replayed.
    pub fn replay<F>(&self, mut callback: F) -> Result<usize, LumenError>
    where
        F: FnMut(NodeId, Vec<f32>, Option<StoredNode>) -> Result<(), LumenError>,
    {
        let mut count = 0;

        for result in self.vectors.iter() {
            let (key, val) = result?;
            let id     = key_to_id(&key);
            let vector = decode_vector(&val);

            // Check for matching graph-node entry
            let graph_node = self.get_graph_node(id)?;

            callback(id, vector, graph_node)?;
            count += 1;
        }

        Ok(count)
    }

    /// Force all pending writes to disk (`fsync`).
    pub fn flush(&self) -> Result<(), LumenError> {
        self.db.flush()?;
        Ok(())
    }

    // ── Hot snapshot (FR-5) ───────────────────────────────────────────────────

    /// Export the entire database to a single `.lumen` directory snapshot.
    ///
    /// Sled's `export()` / `import()` API is used internally.
    /// The snapshot is crash-consistent: Sled flushes before exporting.
    pub fn snapshot_to<P: AsRef<Path>>(&self, dest: P) -> Result<(), LumenError> {
        self.db.flush()?;
        let export  = self.db.export();
        let dest_db = sled::open(dest)?;
        dest_db.import(export);
        dest_db.flush()?;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[test]
    fn encode_decode_round_trip() {
        use crate::storage::codec::{decode_vector, encode_vector};
        let v = vec![1.0f32, -0.5, 3.14];
        assert_eq!(v, decode_vector(&encode_vector(&v)));
    }
}
