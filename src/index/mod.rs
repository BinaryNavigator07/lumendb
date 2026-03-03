//! The Weaver — HNSW index
//!
//! # Threading model
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              HnswIndex  (Send + Sync)            │
//! │                                                  │
//! │  inner: RwLock<HnswInner>                        │
//! │                                                  │
//! │  search() ── read  lock ── concurrent reads ✓   │
//! │  insert() ── write lock ── exclusive write  ✓   │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! Multiple threads can search simultaneously.  Writes (inserts) are
//! serialised through the write lock.  This is the **Phase 1** threading
//! model; Phase 2 (Milestone 4) will introduce per-node `RwLock` to allow
//! concurrent inserts that only contend on the specific nodes being wired.

pub mod layers;
pub mod node;
pub mod params;

use std::collections::{BinaryHeap, HashSet};

use parking_lot::RwLock;

use layers::{dist, search_layer, select_neighbors_heuristic};
use node::{DistancedNode, Node, NodeId, SearchResult};
use params::HnswParams;

use crate::metrics::Metric;

// ── Rng (no external crate needed) ───────────────────────────────────────────

/// Minimal Xorshift-64 PRNG seeded from the system clock.
/// Used only to assign layer levels, so cryptographic quality is unnecessary.
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
        // XOR with a constant to avoid the all-zero state
        Self { state: seed ^ 0x9e37_79b9_7f4a_7c15 }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ── HnswInner — unsynchronised state ─────────────────────────────────────────

pub(crate) struct HnswInner {
    params: HnswParams,
    /// All nodes, indexed by `NodeId`.
    pub(crate) nodes: Vec<Node>,
    /// Entry point for every search: the node with the highest assigned level.
    pub(crate) entry_point: Option<NodeId>,
    /// Current maximum layer level across all nodes.
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

    /// Sample the maximum layer for a newly inserted node.
    ///
    /// Uses the formula from the HNSW paper:
    /// `level = ⌊−ln(uniform(0,1)) × mL⌋`
    ///
    /// where `mL = 1/ln(M)`.  This produces an exponential distribution so
    /// that most nodes appear only in layer 0 and very few reach high layers.
    fn random_level(&mut self) -> usize {
        let r = self.rng.next_f64();
        if r == 0.0 {
            return 0;
        }
        (-r.ln() * self.params.ml).floor() as usize
    }
}

// ── Insert algorithm ──────────────────────────────────────────────────────────

/// Returns `(new_node_id, modified_existing_node_ids)`.
///
/// `modified_existing_node_ids` is the set of pre-existing nodes whose
/// neighbor lists were changed by bidirectional wiring or heuristic pruning.
/// Callers that persist graph state must also re-save those nodes.
fn insert_inner(
    inner: &mut HnswInner,
    vector: Vec<f32>,
    metric: Metric,
) -> (NodeId, Vec<NodeId>) {
    let id = inner.nodes.len();
    let level = inner.random_level();

    // ── First node: becomes the entry point for all layers ────────────────────
    if inner.nodes.is_empty() {
        inner.nodes.push(Node::new(id, vector, level));
        inner.entry_point = Some(id);
        inner.max_layer = level;
        return (id, vec![]);
    }

    // Clone the query vector to avoid borrow conflicts while mutating `nodes`
    let query = vector.clone();
    inner.nodes.push(Node::new(id, vector, level));

    let max_l = inner.max_layer;
    let mut ep: Vec<NodeId> = vec![inner.entry_point.unwrap()];
    // Track every pre-existing node whose neighbor list is mutated so that
    // callers can persist their updated graph state.
    let mut modified: HashSet<NodeId> = HashSet::new();

    // ── Phase 1: coarse greedy descent from max_layer to level + 1 ───────────
    // We use ef = 1 here — we only need the single nearest neighbor per layer
    // to steer the entry point downward.
    if level < max_l {
        for lc in (level + 1..=max_l).rev() {
            let w = search_layer(&inner.nodes, &query, &ep, 1, lc, metric);
            // Use only the nearest element as the entry point for the next layer
            ep = w.into_sorted_vec() // ascending: closest first
                .into_iter()
                .take(1)
                .map(|dn| dn.id)
                .collect();
            if ep.is_empty() {
                ep = vec![inner.entry_point.unwrap()]; // safety fallback
            }
        }
    }

    // ── Phase 2: insert into layers 0 ..= min(level, max_l) ─────────────────
    for lc in (0..=level.min(max_l)).rev() {
        let ef = inner.params.ef_construction;
        let w = search_layer(&inner.nodes, &query, &ep, ef, lc, metric);

        // Collect entry points for the next (lower) layer BEFORE consuming `w`
        ep = w.iter().map(|dn| dn.id).collect();

        let m = inner.params.m_max(lc);
        let neighbors = select_neighbors_heuristic(&inner.nodes, &query, &w, m, metric);

        // Wire new node → selected neighbors
        inner.nodes[id].neighbors[lc] = neighbors.clone();

        // Wire selected neighbors → new node (bidirectional), then prune if needed
        for &nb in &neighbors {
            inner.nodes[nb].neighbors[lc].push(id);
            modified.insert(nb); // back-edge: nb's neighbor list changed

            if inner.nodes[nb].neighbors[lc].len() > m {
                // Rebuild neighbor list for `nb` using the heuristic
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
                // nb is already in `modified`
            }
        }
    }

    // ── Promote entry point if new node occupies higher layers ────────────────
    if level > max_l {
        inner.entry_point = Some(id);
        inner.max_layer = level;
    }

    (id, modified.into_iter().collect())
}

// ── Search algorithm ──────────────────────────────────────────────────────────

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

    // ── Phase 1: coarse descent from max_layer to 1 (ef = 1) ─────────────────
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

    // ── Phase 2: exhaustive beam search at layer 0 ────────────────────────────
    // ef = max(k, ef_search) ensures we explore enough candidates to reliably
    // surface the true top-k even when the graph is imperfectly connected.
    let ef = inner.params.ef_search.max(k);
    let w = search_layer(&inner.nodes, query, &ep, ef, 0, metric);

    // Return top-k sorted nearest-first
    w.into_sorted_vec()
        .into_iter()
        .take(k)
        .map(|dn| SearchResult { id: dn.id, distance: dn.distance })
        .collect()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// The HNSW vector index — thread-safe, zero-copy-search.
///
/// # Example
/// ```rust
/// use lumendb::index::{HnswIndex, params::HnswParams};
/// use lumendb::metrics::Metric;
///
/// let index = HnswIndex::new(HnswParams::default(), Metric::Cosine);
/// let id = index.insert(vec![1.0, 0.0, 0.0]);
/// let results = index.search(&[1.0, 0.0, 0.0], 1);
/// assert_eq!(results[0].id, id);
/// ```
pub struct HnswIndex {
    /// `pub(crate)` so `LumenEngine` can hold a read guard during `vault.put`.
    pub(crate) inner: RwLock<HnswInner>,
    /// The distance metric used for all operations on this collection.
    pub metric: Metric,
}

impl HnswIndex {
    /// Create a new, empty index.
    pub fn new(params: HnswParams, metric: Metric) -> Self {
        Self {
            inner: RwLock::new(HnswInner::new(params)),
            metric,
        }
    }

    /// Insert a vector and return its `NodeId`.
    ///
    /// Acquires the **write lock** — concurrent searches pause until this
    /// returns.  For bulk loading, call this in a tight loop; the write lock
    /// overhead is small compared to the graph-building work.
    pub fn insert(&self, vector: Vec<f32>) -> NodeId {
        let mut inner = self.inner.write();
        let (id, _) = insert_inner(&mut inner, vector, self.metric);
        id
    }

    /// Like [`insert`] but also returns the IDs of all pre-existing nodes
    /// whose neighbor lists were mutated by bidirectional wiring/pruning.
    ///
    /// Used by `LumenEngine` to persist back-edge updates to the vault so
    /// that warm-boot recovery sees a fully connected graph.
    pub(crate) fn insert_and_get_modified(&self, vector: Vec<f32>) -> (NodeId, Vec<NodeId>) {
        let mut inner = self.inner.write();
        insert_inner(&mut inner, vector, self.metric)
    }

    /// Return the `k` approximate nearest neighbors of `query`.
    ///
    /// Acquires the **read lock** — multiple threads may search concurrently.
    ///
    /// Results are sorted by ascending distance (most similar first).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let inner = self.inner.read();
        search_inner(&inner, query, k, self.metric)
    }

    /// Total number of vectors in the index.
    pub fn len(&self) -> usize {
        self.inner.read().nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().nodes.is_empty()
    }

    /// Dimension of the vectors stored in the index (`None` if empty).
    pub fn dim(&self) -> Option<usize> {
        self.inner.read().nodes.first().map(|n| n.vector.len())
    }

    // ── Recovery API (called by LumenEngine during warm boot) ─────────────────

    /// **Fast-path recovery**: directly restore a node that was previously
    /// persisted with its full graph state, bypassing the HNSW insertion
    /// algorithm entirely.
    ///
    /// Nodes *must* be restored in ascending `id` order (0, 1, 2, …) so that
    /// `id == nodes.len()` holds at every call — matching the invariant that
    /// `insert_inner` relies on.
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

    /// Restore the global entry-point and max-layer after all nodes have
    /// been fast-path recovered.
    pub(crate) fn restore_header(&self, entry_point: NodeId, max_layer: usize) {
        let mut inner = self.inner.write();
        inner.entry_point = Some(entry_point);
        inner.max_layer   = max_layer;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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

    // ── Brute-force reference for recall testing ──────────────────────────────

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

    // ── Basic correctness ─────────────────────────────────────────────────────

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
        // Insert the zero-degree axis vectors
        let _a = idx.insert(vec![1.0, 0.0, 0.0]);
        let b  = idx.insert(vec![0.0, 1.0, 0.0]);
        let _c = idx.insert(vec![0.0, 0.0, 1.0]);

        // Query close to b — expect b as nearest
        let results = idx.search(&[0.05, 1.0, 0.05], 1);
        assert_eq!(results[0].id, b);
    }

    #[test]
    fn top_k_respects_limit() {
        // Spread 50 vectors evenly around the unit circle so every pair has a
        // distinct cosine distance — collinear vectors all have distance 0, which
        // makes the diversity heuristic skip them and produces an under-connected
        // graph that cannot reliably return k results.
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

    // ── Recall test: HNSW must agree with brute force on small dataset ────────

    #[test]
    fn recall_on_small_dataset() {
        let idx = HnswIndex::new(HnswParams::new(16, 200, 50), Metric::Cosine);
        let dim = 128;

        // Generate 500 reproducible pseudo-random vectors using the LCG trick
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

        // We expect ≥ 80% recall on a 500-vector dataset with these params
        let overlap = hnsw_ids.intersection(&true_ids).count();
        let recall = overlap as f32 / k as f32;
        assert!(
            recall >= 0.8,
            "Recall@{k} = {recall:.2} (want ≥ 0.80) — hnsw={hnsw_ids:?} true={true_ids:?}"
        );
    }

    // ── Concurrent read / write ───────────────────────────────────────────────

    #[test]
    fn concurrent_insert_and_search() {
        use std::sync::Arc;

        let idx = Arc::new(HnswIndex::new(HnswParams::new(8, 40, 20), Metric::Euclidean));

        // Writer thread
        let idx_w = Arc::clone(&idx);
        let writer = std::thread::spawn(move || {
            for i in 0..200u32 {
                idx_w.insert(vec![i as f32, 0.0, 0.0]);
            }
        });

        // Reader thread (concurrent)
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
