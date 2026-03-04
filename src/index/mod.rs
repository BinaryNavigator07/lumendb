pub mod layers;
pub mod node;
pub mod params;

use std::collections::{BinaryHeap, HashSet};

use parking_lot::RwLock;

use layers::{dist, search_layer, select_neighbors_heuristic};
use node::{DistancedNode, Node, NodeId, SearchResult};
use params::HnswParams;

use crate::metrics::Metric;

struct Rng {
    state: u64,
}

impl Rng {
    fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs() << 20))
            .unwrap_or(0xcafe_babe_dead_beef);
        Self { state: seed ^ 0x9e37_79b9_7f4a_7c15 }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

pub(crate) struct HnswInner {
    params: HnswParams,
    pub(crate) nodes: Vec<Node>,
    pub(crate) entry_point: Option<NodeId>,
    pub(crate) max_layer: usize,
    rng: Rng,
}

impl HnswInner {
    fn new(params: HnswParams) -> Self {
        Self {
            params,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            rng: Rng::new(),
        }
    }

    fn random_level(&mut self) -> usize {
        let r = self.rng.next_f64();
        if r == 0.0 {
            return 0;
        }
        (-r.ln() * self.params.ml).floor() as usize
    }
}

fn insert_inner(
    inner: &mut HnswInner,
    vector: Vec<f32>,
    metric: Metric,
) -> (NodeId, Vec<NodeId>) {
    let id = inner.nodes.len();
    let level = inner.random_level();

    if inner.nodes.is_empty() {
        inner.nodes.push(Node::new(id, vector, level));
        inner.entry_point = Some(id);
        inner.max_layer = level;
        return (id, vec![]);
    }

    let query = vector.clone();
    inner.nodes.push(Node::new(id, vector, level));

    let max_l = inner.max_layer;
    let mut ep: Vec<NodeId> = vec![inner.entry_point.unwrap()];
    let mut modified: HashSet<NodeId> = HashSet::new();

    if level < max_l {
        for lc in (level + 1..=max_l).rev() {
            let w = search_layer(&inner.nodes, &query, &ep, 1, lc, metric);
            ep = w.into_sorted_vec()
                .into_iter()
                .take(1)
                .map(|dn| dn.id)
                .collect();
            if ep.is_empty() {
                ep = vec![inner.entry_point.unwrap()];
            }
        }
    }

    for lc in (0..=level.min(max_l)).rev() {
        let ef = inner.params.ef_construction;
        let w = search_layer(&inner.nodes, &query, &ep, ef, lc, metric);

        ep = w.iter().map(|dn| dn.id).collect();

        let m = inner.params.m_max(lc);
        let neighbors = select_neighbors_heuristic(&inner.nodes, &query, &w, m, metric);

        inner.nodes[id].neighbors[lc] = neighbors.clone();

        for &nb in &neighbors {
            inner.nodes[nb].neighbors[lc].push(id);
            modified.insert(nb);

            if inner.nodes[nb].neighbors[lc].len() > m {
                let nb_vec = inner.nodes[nb].vector.clone();
                let current: Vec<NodeId> = inner.nodes[nb].neighbors[lc].clone();

                let w_nb: BinaryHeap<DistancedNode> = current
                    .iter()
                    .map(|&nid| DistancedNode {
                        id: nid,
                        distance: dist(&inner.nodes[nid].vector, &nb_vec, metric),
                    })
                    .collect();

                inner.nodes[nb].neighbors[lc] =
                    select_neighbors_heuristic(&inner.nodes, &nb_vec, &w_nb, m, metric);
            }
        }
    }

    if level > max_l {
        inner.entry_point = Some(id);
        inner.max_layer = level;
    }

    (id, modified.into_iter().collect())
}

fn search_inner(
    inner: &HnswInner,
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Vec<SearchResult> {
    if inner.nodes.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut ep: Vec<NodeId> = vec![inner.entry_point.unwrap()];

    for lc in (1..=inner.max_layer).rev() {
        let w = search_layer(&inner.nodes, query, &ep, 1, lc, metric);
        ep = w.into_sorted_vec()
            .into_iter()
            .take(1)
            .map(|dn| dn.id)
            .collect();
        if ep.is_empty() {
            ep = vec![inner.entry_point.unwrap()];
        }
    }

    let ef = inner.params.ef_search.max(k);
    let w = search_layer(&inner.nodes, query, &ep, ef, 0, metric);

    w.into_sorted_vec()
        .into_iter()
        .take(k)
        .map(|dn| SearchResult { id: dn.id, distance: dn.distance })
        .collect()
}

pub struct HnswIndex {
    pub(crate) inner: RwLock<HnswInner>,
    pub metric: Metric,
}

impl HnswIndex {
    pub fn new(params: HnswParams, metric: Metric) -> Self {
        Self {
            inner: RwLock::new(HnswInner::new(params)),
            metric,
        }
    }

    pub fn insert(&self, vector: Vec<f32>) -> NodeId {
        let mut inner = self.inner.write();
        let (id, _) = insert_inner(&mut inner, vector, self.metric);
        id
    }

    pub(crate) fn insert_and_get_modified(&self, vector: Vec<f32>) -> (NodeId, Vec<NodeId>) {
        let mut inner = self.inner.write();
        insert_inner(&mut inner, vector, self.metric)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let inner = self.inner.read();
        search_inner(&inner, query, k, self.metric)
    }

    pub fn len(&self) -> usize {
        self.inner.read().nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().nodes.is_empty()
    }

    pub fn dim(&self) -> Option<usize> {
        self.inner.read().nodes.first().map(|n| n.vector.len())
    }

    pub(crate) fn restore_node(
        &self,
        id: NodeId,
        vector: Vec<f32>,
        level: usize,
        neighbors: Vec<Vec<NodeId>>,
    ) {
        let mut inner = self.inner.write();
        debug_assert_eq!(
            inner.nodes.len(), id,
            "restore_node: expected id={id}, but nodes.len()={}",
            inner.nodes.len()
        );
        let mut node = Node::new(id, vector, level);
        node.neighbors = neighbors;
        inner.nodes.push(node);
    }

    pub(crate) fn restore_header(&self, entry_point: NodeId, max_layer: usize) {
        let mut inner = self.inner.write();
        inner.entry_point = Some(entry_point);
        inner.max_layer   = max_layer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metric;

    fn make_index() -> HnswIndex {
        HnswIndex::new(
            HnswParams::new(8, 40, 20),
            Metric::Cosine,
        )
    }

    fn brute_knn(
        index: &HnswIndex,
        query: &[f32],
        k: usize,
    ) -> Vec<NodeId> {
        let inner = index.inner.read();
        let mut dists: Vec<(NodeId, f32)> = inner
            .nodes
            .iter()
            .map(|n| (n.id, dist(query, &n.vector, index.metric)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.into_iter().take(k).map(|(id, _)| id).collect()
    }

    #[test]
    fn empty_search_returns_empty() {
        let idx = make_index();
        assert!(idx.search(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn single_insert_found() {
        let idx = make_index();
        let id = idx.insert(vec![1.0, 0.0, 0.0]);
        let results = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
        assert!(results[0].distance < 1e-5);
    }

    #[test]
    fn nearest_neighbor_exact_match() {
        let idx = make_index();
        let _a = idx.insert(vec![1.0, 0.0, 0.0]);
        let b  = idx.insert(vec![0.0, 1.0, 0.0]);
        let _c = idx.insert(vec![0.0, 0.0, 1.0]);

        let results = idx.search(&[0.05, 1.0, 0.05], 1);
        assert_eq!(results[0].id, b);
    }

    #[test]
    fn top_k_respects_limit() {
        let idx = make_index();
        let n = 50usize;
        for i in 0..n {
            let angle = std::f32::consts::TAU * i as f32 / n as f32;
            idx.insert(vec![angle.cos(), angle.sin()]);
        }
        let results = idx.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn results_sorted_ascending_by_distance() {
        let idx = make_index();
        for i in 0..30 {
            idx.insert(vec![i as f32, 0.0]);
        }
        let results = idx.search(&[15.0, 0.0], 5);
        for w in results.windows(2) {
            assert!(
                w[0].distance <= w[1].distance,
                "results not sorted: {} > {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    #[test]
    fn recall_on_small_dataset() {
        let idx = HnswIndex::new(HnswParams::new(16, 200, 50), Metric::Cosine);
        let dim = 128;

        let mut state: u64 = 0xdead_beef_cafe_1234;
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect();

        for v in &vectors {
            idx.insert(v.clone());
        }

        let query: Vec<f32> = vectors[0].clone();
        let k = 10;
        let hnsw_ids: std::collections::HashSet<NodeId> =
            idx.search(&query, k).into_iter().map(|r| r.id).collect();
        let true_ids: std::collections::HashSet<NodeId> =
            brute_knn(&idx, &query, k).into_iter().collect();

        let overlap = hnsw_ids.intersection(&true_ids).count();
        let recall = overlap as f32 / k as f32;
        assert!(
            recall >= 0.8,
            "Recall@{k} = {recall:.2} (want ≥ 0.80) — hnsw={hnsw_ids:?} true={true_ids:?}"
        );
    }

    #[test]
    fn concurrent_insert_and_search() {
        use std::sync::Arc;

        let idx = Arc::new(HnswIndex::new(HnswParams::new(8, 40, 20), Metric::Euclidean));

        let idx_w = Arc::clone(&idx);
        let writer = std::thread::spawn(move || {
            for i in 0..200u32 {
                idx_w.insert(vec![i as f32, 0.0, 0.0]);
            }
        });

        let idx_r = Arc::clone(&idx);
        let reader = std::thread::spawn(move || {
            for _ in 0..50 {
                let _ = idx_r.search(&[100.0, 0.0, 0.0], 5);
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        assert_eq!(idx.len(), 200);
    }
}
