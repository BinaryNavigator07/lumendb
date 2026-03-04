use serde::{Deserialize, Serialize};

use crate::index::params::HnswParams;
use crate::index::node::NodeId;
use crate::metrics::Metric;
use crate::LumenError;

#[inline]
pub fn id_to_key(id: NodeId) -> [u8; 8] {
    (id as u64).to_be_bytes()
}

#[inline]
pub fn key_to_id(bytes: &[u8]) -> NodeId {
    let arr: [u8; 8] = bytes.try_into().expect("NodeId key must be 8 bytes");
    u64::from_be_bytes(arr) as NodeId
}

pub fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn encode_meta(m: &serde_json::Value) -> Result<Vec<u8>, LumenError> {
    serde_json::to_vec(m).map_err(LumenError::from)
}

pub fn decode_meta(bytes: &[u8]) -> Result<serde_json::Value, LumenError> {
    serde_json::from_slice(bytes).map_err(LumenError::from)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredConfig {
    pub dim: usize,
    pub metric: u8,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub entry_point: Option<u64>,
    pub max_layer: usize,
}

impl StoredConfig {
    pub fn new(dim: usize, metric: Metric, params: &HnswParams) -> Self {
        Self {
            dim,
            metric: metric_to_u8(metric),
            m: params.m,
            ef_construction: params.ef_construction,
            ef_search: params.ef_search,
            entry_point: None,
            max_layer: 0,
        }
    }

    pub fn metric(&self) -> Result<Metric, LumenError> {
        u8_to_metric(self.metric)
    }

    pub fn hnsw_params(&self) -> HnswParams {
        HnswParams::new(self.m, self.ef_construction, self.ef_search)
    }
}

pub fn encode_config(c: &StoredConfig) -> Result<Vec<u8>, LumenError> {
    bincode::serialize(c).map_err(LumenError::from)
}

pub fn decode_config(bytes: &[u8]) -> Result<StoredConfig, LumenError> {
    bincode::deserialize(bytes).map_err(LumenError::from)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredNode {
    pub level: usize,
    pub neighbors: Vec<Vec<NodeId>>,
}

pub fn encode_node(n: &StoredNode) -> Result<Vec<u8>, LumenError> {
    bincode::serialize(n).map_err(LumenError::from)
}

pub fn decode_node(bytes: &[u8]) -> Result<StoredNode, LumenError> {
    bincode::deserialize(bytes).map_err(LumenError::from)
}

fn metric_to_u8(m: Metric) -> u8 {
    match m {
        Metric::DotProduct => 0,
        Metric::Euclidean  => 1,
        Metric::Cosine     => 2,
    }
}

fn u8_to_metric(v: u8) -> Result<Metric, LumenError> {
    match v {
        0 => Ok(Metric::DotProduct),
        1 => Ok(Metric::Euclidean),
        2 => Ok(Metric::Cosine),
        _ => Err(LumenError::Codec(format!("unknown metric byte: {v}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_vector() {
        let v: Vec<f32> = vec![1.0, -2.5, 3.14, f32::MAX, 0.0];
        let encoded = encode_vector(&v);
        let decoded = decode_vector(&encoded);
        assert_eq!(v, decoded);
    }

    #[test]
    fn round_trip_id_key() {
        for id in [0usize, 1, 255, 256, 65535, usize::MAX / 2] {
            assert_eq!(key_to_id(&id_to_key(id)), id);
        }
    }

    #[test]
    fn id_key_lexicographic_order() {
        let keys: Vec<[u8; 8]> = (0u64..10).map(|i| i.to_be_bytes()).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn round_trip_config() {
        let cfg = StoredConfig::new(1536, Metric::Cosine, &HnswParams::default());
        let bytes = encode_config(&cfg).unwrap();
        let back  = decode_config(&bytes).unwrap();
        assert_eq!(back.dim, 1536);
        assert_eq!(back.metric, 2);
    }

    #[test]
    fn round_trip_metadata() {
        let meta = serde_json::json!({ "source": "chat_history", "user_id": 42, "score": 0.95 });
        let bytes = encode_meta(&meta).unwrap();
        let back  = decode_meta(&bytes).unwrap();
        assert_eq!(meta, back);
    }
}
