//! Core node types for the HNSW graph.

use std::cmp::Ordering;

/// Stable index into `HnswInner::nodes`.
pub type NodeId = usize;

/// A single vertex in the HNSW graph.
///
/// Stores the raw vector inline (Milestone 2).  In Milestone 3 the vector will
/// be moved to the Sled `Vault` and replaced here with a `VectorId` reference.
#[derive(Debug)]
pub struct Node {
    /// Globally unique identifier (= position in the `nodes` Vec).
    pub id: NodeId,

    /// The embedding vector.
    pub vector: Vec<f32>,

    /// Highest layer this node participates in (0-indexed).
    /// Assigned at insertion time via an exponential distribution.
    pub level: usize,

    /// `neighbors[layer]` holds the NodeIds connected to this node at `layer`.
    /// Only layers `0 ..= level` are populated.
    pub neighbors: Vec<Vec<NodeId>>,
}

impl Node {
    pub fn new(id: NodeId, vector: Vec<f32>, level: usize) -> Self {
        // Pre-allocate neighbor lists for every layer the node participates in
        let neighbors = vec![Vec::new(); level + 1];
        Self { id, vector, level, neighbors }
    }

    /// Borrow the neighbor list at `layer`.
    /// Returns an empty slice if the node does not participate at `layer`.
    #[inline]
    pub fn neighbors_at(&self, layer: usize) -> &[NodeId] {
        self.neighbors.get(layer).map(Vec::as_slice).unwrap_or(&[])
    }
}

// ── DistancedNode ─────────────────────────────────────────────────────────────

/// A `(NodeId, distance)` pair used in the priority queues inside HNSW.
///
/// **Ordering** is deliberately chosen for a **max-heap** (`BinaryHeap`):
/// the node with the *largest* distance floats to the top so that
/// `heap.pop()` efficiently removes the *farthest* candidate — used when
/// pruning the dynamic list W down to `ef` entries.
///
/// For a **min-heap** (nearest-first traversal), wrap in `std::cmp::Reverse`.
#[derive(Clone, Debug, PartialEq)]
pub struct DistancedNode {
    pub id: NodeId,
    /// Distance from the query to this node (smaller = more similar).
    pub distance: f32,
}

/// Safety: all distances in the index are finite (non-NaN).
/// The insertion path rejects NaN distances before they enter the heap.
impl Eq for DistancedNode {}

impl PartialOrd for DistancedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistancedNode {
    /// Larger distance → "greater" → pops first from `BinaryHeap` (max-heap).
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

// ── SearchResult ──────────────────────────────────────────────────────────────

/// One entry in a KNN result set returned by [`HnswIndex::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The node identifier — use this to retrieve metadata from the Vault.
    pub id: NodeId,

    /// Distance from the query vector to this result.
    ///
    /// Interpretation per metric:
    /// - `Cosine`     → cosine distance ∈ [0, 2] (0 = identical direction)
    /// - `Euclidean`  → L2 distance ≥ 0
    /// - `DotProduct` → negated dot product (smaller = higher similarity)
    pub distance: f32,
}
