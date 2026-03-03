//! `LumenEngine` — the top-level coordinator (Milestone 3).
//!
//! Wires the HNSW index (Milestone 2) to the Sled Vault (Milestone 3),
//! implementing the WAL guarantee and the warm-boot recovery loop.
//!
//! # Insertion flow
//!
//! ```text
//!  insert(vector, metadata)
//!       │
//!       ├─ 1. vault.put(id, vec, meta, node)  ←── fsync here (WAL)
//!       │
//!       └─ 2. index already updated by HNSW   ←── in-memory only
//! ```
//!
//! # Recovery flow (warm boot)
//!
//! ```text
//!  LumenEngine::open(path, params, metric)
//!       │
//!       ├─ Load StoredConfig → verify dim / metric / params
//!       │
//!       └─ vault.replay():
//!               ├─ graph_nodes entry present  →  FAST PATH
//!               │       restore Node directly into HnswInner
//!               │       (no HNSW algorithm, O(N) startup)
//!               │
//!               └─ graph_nodes entry missing  →  SLOW PATH
//!                       re-insert via HNSW algorithm
//!                       (handles crash mid-write)
//! ```

use std::path::Path;
use std::sync::Arc;

use crate::index::node::NodeId;
use crate::index::params::HnswParams;
use crate::index::HnswIndex;
use crate::metrics::Metric;
use crate::storage::codec::{StoredConfig, StoredNode};
use crate::storage::SledVault;
use crate::LumenError;

/// The production entry-point to LumenDB.
///
/// All persistent collections should go through `LumenEngine`; direct access
/// to `HnswIndex` is available for in-memory-only workloads (e.g. tests).
pub struct LumenEngine {
    pub index:   Arc<HnswIndex>,
    pub vault:   SledVault,
    dim:         usize,
}

impl LumenEngine {
    // ── Constructor / warm boot ───────────────────────────────────────────────

    /// Open (or create) a LumenDB collection at `path`.
    ///
    /// If the directory exists and contains a valid Sled database, all stored
    /// vectors are recovered into the in-memory HNSW index before returning.
    ///
    /// If the directory is new, a fresh collection is created with `params`
    /// and `metric`.
    ///
    /// # Errors
    /// - `LumenError::ConfigMismatch` if the stored dimension or metric
    ///   differs from the supplied arguments.
    /// - `LumenError::Storage` for I/O failures.
    pub fn open<P: AsRef<Path>>(
        path: P,
        params: HnswParams,
        metric: Metric,
        dim: usize,
    ) -> Result<Self, LumenError> {
        let vault = SledVault::open(path)?;

        // ── Validate or create config ────────────────────────────────────────
        match vault.load_config()? {
            Some(cfg) => {
                // Existing database: verify compatibility
                if cfg.dim != dim {
                    return Err(LumenError::ConfigMismatch(format!(
                        "stored dim={} but requested dim={dim}",
                        cfg.dim
                    )));
                }
                let stored_metric = cfg.metric()?;
                if stored_metric != metric {
                    return Err(LumenError::ConfigMismatch(format!(
                        "stored metric={stored_metric:?} but requested metric={metric:?}"
                    )));
                }
            }
            None => {
                // New database: persist config
                let cfg = StoredConfig::new(dim, metric, &params);
                vault.save_config(&cfg)?;
            }
        }

        let index = Arc::new(HnswIndex::new(params, metric));

        // ── Warm boot: replay all stored vectors ─────────────────────────────
        let index_ref = Arc::clone(&index);
        let count = vault.replay(|id, vector, graph_node| {
            recover_node(&index_ref, id, vector, graph_node);
            Ok(())
        })?;

        if count > 0 {
            // Sync the in-memory header (entry_point, max_layer) from the store
            if let Some(cfg) = vault.load_config()? {
                if let Some(ep) = cfg.entry_point {
                    index.restore_header(ep as NodeId, cfg.max_layer);
                }
            }
            eprintln!(
                "[LumenDB] Warm boot complete — recovered {count} vectors (index len={})",
                index.len()
            );
        }

        Ok(Self { index, vault, dim })
    }

    // ── Insert ────────────────────────────────────────────────────────────────

    /// Insert a vector with associated metadata.
    ///
    /// **Durability guarantee**: the function does not return `Ok` until the
    /// vector is fsynced to disk.  If the process crashes after this call
    /// returns, the vector will be recovered on the next `open()`.
    ///
    /// # Errors
    /// Returns `LumenError::DimensionMismatch` if `vector.len() != self.dim`.
    pub fn insert(
        &self,
        vector: Vec<f32>,
        meta: serde_json::Value,
    ) -> Result<NodeId, LumenError> {
        if vector.len() != self.dim {
            return Err(LumenError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        // ── Step 1: insert into HNSW (in memory) ────────────────────────────
        // Also capture the pre-existing nodes whose neighbor lists were mutated
        // by bidirectional wiring so we can persist their updated graph state.
        let (id, modified_nodes) = self.index.insert_and_get_modified(vector.clone());

        // ── Step 2: persist to Sled with fsync (WAL guarantee) ──────────────
        {
            let inner = self.index.inner.read();
            let node  = &inner.nodes[id];
            self.vault.put(id, &vector, &meta, node)?;

            // Persist the updated neighbor lists of every existing node that
            // received a back-edge from this insertion.  Their vector and
            // metadata records are already in the vault; only graph_nodes needs
            // updating.  We flush once after all writes to amortise the fsync.
            for &nb_id in &modified_nodes {
                self.vault.update_graph_node(&inner.nodes[nb_id])?;
            }
            if !modified_nodes.is_empty() {
                self.vault.flush()?;
            }
        }

        // ── Step 3: update config header (entry_point, max_layer) ───────────
        {
            let inner = self.index.inner.read();
            self.vault.update_config_header(inner.entry_point, inner.max_layer)?;
        }

        Ok(id)
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Return the `k` approximate nearest neighbors of `query`.
    ///
    /// This is a pure in-memory operation — no storage I/O involved.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchHit>, LumenError> {
        if query.len() != self.dim {
            return Err(LumenError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let raw = self.index.search(query, k);
        let mut hits = Vec::with_capacity(raw.len());

        for r in raw {
            let meta = self.vault.get_metadata(r.id)?;
            hits.push(SearchHit {
                id:       r.id,
                distance: r.distance,
                metadata: meta.unwrap_or(serde_json::Value::Null),
            });
        }

        Ok(hits)
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Trigger a hot backup of the entire database to `dest` (FR-5).
    pub fn snapshot_to<P: AsRef<Path>>(&self, dest: P) -> Result<(), LumenError> {
        self.vault.snapshot_to(dest.as_ref())
    }

    pub fn len(&self)      -> usize { self.index.len() }
    pub fn is_empty(&self) -> bool  { self.index.is_empty() }
    pub fn dim(&self)      -> usize { self.dim }
}

// ── Recovery helpers ──────────────────────────────────────────────────────────

/// Restore a single node into the index during warm boot.
fn recover_node(
    index: &HnswIndex,
    id: NodeId,
    vector: Vec<f32>,
    graph_node: Option<StoredNode>,
) {
    match graph_node {
        // Fast path: graph state intact — bypass HNSW algorithm
        Some(stored) => {
            index.restore_node(id, vector, stored.level, stored.neighbors);
        }
        // Slow path: crash mid-write — re-run HNSW insertion
        None => {
            let assigned = index.insert(vector);
            debug_assert_eq!(
                assigned, id,
                "ID mismatch during slow-path recovery: expected {id}, got {assigned}"
            );
        }
    }
}

// ── SearchHit ─────────────────────────────────────────────────────────────────

/// A single result from [`LumenEngine::search`], enriched with metadata.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id:       NodeId,
    pub distance: f32,
    /// The JSON metadata stored alongside this vector at insertion time.
    pub metadata: serde_json::Value,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::params::HnswParams;
    use crate::metrics::Metric;

    fn open_temp(dir: &std::path::Path) -> LumenEngine {
        LumenEngine::open(dir, HnswParams::new(8, 40, 20), Metric::Cosine, 4).unwrap()
    }

    #[test]
    fn insert_then_search() {
        let dir = tempfile::tempdir().unwrap();
        let eng = open_temp(dir.path());

        let id = eng.insert(vec![1.0, 0.0, 0.0, 0.0], serde_json::json!({"tag": "x"})).unwrap();
        let _  = eng.insert(vec![0.0, 1.0, 0.0, 0.0], serde_json::json!({"tag": "y"})).unwrap();

        let hits = eng.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, id);
        assert!(hits[0].distance < 1e-4);
        assert_eq!(hits[0].metadata["tag"], "x");
    }

    #[test]
    fn warm_boot_recovers_all_vectors() {
        let dir = tempfile::tempdir().unwrap();

        // Session 1 — insert 20 vectors
        {
            let eng = open_temp(dir.path());
            for i in 0..20u32 {
                eng.insert(
                    vec![i as f32, 0.0, 0.0, 0.0],
                    serde_json::json!({"i": i}),
                )
                .unwrap();
            }
            assert_eq!(eng.len(), 20);
        } // eng is dropped here (Sled flushes on drop)

        // Session 2 — re-open and verify
        {
            let eng = open_temp(dir.path());
            assert_eq!(eng.len(), 20, "all 20 vectors should survive warm boot");

            // Search should work correctly after recovery
            let hits = eng.search(&[10.0, 0.0, 0.0, 0.0], 3).unwrap();
            assert!(!hits.is_empty());
        }
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let eng = open_temp(dir.path()); // dim = 4
        let err = eng.insert(vec![1.0, 2.0, 3.0], serde_json::Value::Null);
        assert!(matches!(err, Err(LumenError::DimensionMismatch { expected: 4, got: 3 })));
    }

    #[test]
    fn config_mismatch_rejected_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            open_temp(dir.path()); // creates db with dim=4, Cosine
        }
        // Reopen with wrong dimension
        let result = LumenEngine::open(
            dir.path(),
            HnswParams::new(8, 40, 20),
            Metric::Cosine,
            8, // wrong dim
        );
        assert!(matches!(result, Err(LumenError::ConfigMismatch(_))));
    }
}
