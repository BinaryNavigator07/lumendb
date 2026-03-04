use std::path::Path;
use std::sync::Arc;

use crate::index::node::NodeId;
use crate::index::params::HnswParams;
use crate::index::HnswIndex;
use crate::metrics::Metric;
use crate::storage::codec::{StoredConfig, StoredNode};
use crate::storage::SledVault;
use crate::LumenError;

pub struct LumenEngine {
    pub index:   Arc<HnswIndex>,
    pub vault:   SledVault,
    dim:         usize,
}

impl LumenEngine {

    pub fn open<P: AsRef<Path>>(
        path: P,
        params: HnswParams,
        metric: Metric,
        dim: usize,
    ) -> Result<Self, LumenError> {
        let vault = SledVault::open(path)?;

        match vault.load_config()? {
            Some(cfg) => {
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
                let cfg = StoredConfig::new(dim, metric, &params);
                vault.save_config(&cfg)?;
            }
        }

        let index = Arc::new(HnswIndex::new(params, metric));

        let index_ref = Arc::clone(&index);
        let count = vault.replay(|id, vector, graph_node| {
            recover_node(&index_ref, id, vector, graph_node);
            Ok(())
        })?;

        if count > 0 {
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

    pub fn reopen<P: AsRef<Path>>(path: P) -> Result<Self, LumenError> {
        let peek   = SledVault::open(path.as_ref())?;
        let cfg    = peek.load_config()?.ok_or_else(|| {
            LumenError::ConfigMismatch(
                "no stored config — directory is not a LumenDB collection".into(),
            )
        })?;
        let metric = cfg.metric()?;
        let params = cfg.hnsw_params();
        let dim    = cfg.dim;
        drop(peek);
        Self::open(path, params, metric, dim)
    }

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

        let (id, modified_nodes) = self.index.insert_and_get_modified(vector.clone());

        {
            let inner = self.index.inner.read();
            let node  = &inner.nodes[id];
            self.vault.put(id, &vector, &meta, node)?;

            for &nb_id in &modified_nodes {
                self.vault.update_graph_node(&inner.nodes[nb_id])?;
            }
            if !modified_nodes.is_empty() {
                self.vault.flush()?;
            }
        }

        {
            let inner = self.index.inner.read();
            self.vault.update_config_header(inner.entry_point, inner.max_layer)?;
        }

        Ok(id)
    }

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

    pub fn snapshot_to<P: AsRef<Path>>(&self, dest: P) -> Result<(), LumenError> {
        self.vault.snapshot_to(dest.as_ref())
    }

    pub fn len(&self)      -> usize { self.index.len() }
    pub fn is_empty(&self) -> bool  { self.index.is_empty() }
    pub fn dim(&self)      -> usize { self.dim }
}

fn recover_node(
    index: &HnswIndex,
    id: NodeId,
    vector: Vec<f32>,
    graph_node: Option<StoredNode>,
) {
    match graph_node {
        Some(stored) => {
            index.restore_node(id, vector, stored.level, stored.neighbors);
        }
        None => {
            let assigned = index.insert(vector);
            debug_assert_eq!(
                assigned, id,
                "ID mismatch during slow-path recovery: expected {id}, got {assigned}"
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id:       NodeId,
    pub distance: f32,
    pub metadata: serde_json::Value,
}

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
        }

        {
            let eng = open_temp(dir.path());
            assert_eq!(eng.len(), 20, "all 20 vectors should survive warm boot");

            let hits = eng.search(&[10.0, 0.0, 0.0, 0.0], 3).unwrap();
            assert!(!hits.is_empty());
        }
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let eng = open_temp(dir.path());
        let err = eng.insert(vec![1.0, 2.0, 3.0], serde_json::Value::Null);
        assert!(matches!(err, Err(LumenError::DimensionMismatch { expected: 4, got: 3 })));
    }

    #[test]
    fn config_mismatch_rejected_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            open_temp(dir.path());
        }
        let result = LumenEngine::open(
            dir.path(),
            HnswParams::new(8, 40, 20),
            Metric::Cosine,
            8,
        );
        assert!(matches!(result, Err(LumenError::ConfigMismatch(_))));
    }
}
