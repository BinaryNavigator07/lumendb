//! Core graph-traversal algorithms for the HNSW index (FR-2).
//!
//! Both functions are pure: they take an immutable slice of nodes and return
//! results without any locking or state mutation.  The calling code in
//! `mod.rs` is responsible for acquiring the appropriate `RwLock` guard.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use super::node::{DistancedNode, Node, NodeId};
use crate::metrics::{self, Metric};

// ── Distance helper ───────────────────────────────────────────────────────────

/// Returns a non-negative "distance" for any of the three metrics so that
/// **smaller always means more similar** — which is what the HNSW heaps expect.
///
/// | Metric       | Formula                        |
/// |--------------|--------------------------------|
/// | Cosine       | `1 − cosine_similarity`        |
/// | Euclidean    | `sqrt(Σ(aᵢ − bᵢ)²)`           |
/// | DotProduct   | `−dot(a, b)`                   |
///
/// Returns `f32::MAX` on any error (length mismatch, zero vector) so a bad
/// vector degrades gracefully rather than panicking in the middle of a search.
#[inline]
pub(super) fn dist(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => metrics::cosine_distance(a, b).unwrap_or(f32::MAX),
        Metric::Euclidean => metrics::euclidean_distance(a, b).unwrap_or(f32::MAX),
        Metric::DotProduct => -metrics::dot_product(a, b).unwrap_or(f32::MIN),
    }
}

// ── search_layer ──────────────────────────────────────────────────────────────

/// Greedy beam search within a single HNSW layer (Algorithm 2 from the paper).
///
/// Starting from `entry_points`, expands the nearest candidates first and
/// maintains a dynamic result set `W` of at most `ef` nearest elements seen.
///
/// # Returns
/// A **max-heap** (`BinaryHeap<DistancedNode>`) where the farthest element is
/// at the top.  Callers use `into_sorted_vec()` for nearest-first ordering.
pub(super) fn search_layer(
    nodes: &[Node],
    query: &[f32],
    entry_points: &[NodeId],
    ef: usize,
    layer: usize,
    metric: Metric,
) -> BinaryHeap<DistancedNode> {
    // W — dynamic result set (max-heap, farthest on top for O(1) pruning)
    let mut w: BinaryHeap<DistancedNode> = BinaryHeap::with_capacity(ef + 1);
    // C — exploration frontier (min-heap, nearest on top for greedy traversal)
    let mut candidates: BinaryHeap<Reverse<DistancedNode>> =
        BinaryHeap::with_capacity(ef * 4);
    // Visited guard — prevents re-expanding the same node
    let mut visited: HashSet<NodeId> = HashSet::with_capacity(ef * 4);

    // Seed heaps from entry points
    for &ep in entry_points {
        if visited.insert(ep) {
            let d = dist(query, &nodes[ep].vector, metric);
            w.push(DistancedNode { id: ep, distance: d });
            candidates.push(Reverse(DistancedNode { id: ep, distance: d }));
        }
    }

    while let Some(Reverse(c)) = candidates.pop() {
        // If the nearest unvisited candidate is already farther than the
        // farthest element in W (and W is full), we cannot improve W further.
        let f_dist = w.peek().map_or(f32::MAX, |f| f.distance);
        if c.distance > f_dist && w.len() >= ef {
            break;
        }

        // Expand c's neighbors at this layer
        for &nb in nodes[c.id].neighbors_at(layer) {
            if !visited.insert(nb) {
                continue;
            }
            let d = dist(query, &nodes[nb].vector, metric);
            let current_f = w.peek().map_or(f32::MAX, |f| f.distance);

            // Add nb to W if it improves the result set
            if d < current_f || w.len() < ef {
                candidates.push(Reverse(DistancedNode { id: nb, distance: d }));
                w.push(DistancedNode { id: nb, distance: d });
                // Keep W bounded at ef
                if w.len() > ef {
                    w.pop(); // removes the farthest element
                }
            }
        }
    }

    w
}

// ── select_neighbors_heuristic ────────────────────────────────────────────────

/// Heuristic neighbor selection (Algorithm 4 from the paper).
///
/// Unlike the simple "take the M closest" approach, this heuristic prefers
/// **diverse** neighbors: a candidate `e` is selected only if it is closer to
/// the query `q` than it is to any already-selected neighbor.
///
/// This produces a better-distributed neighbourhood and improves recall,
/// especially in high-dimensional spaces.
///
/// # Arguments
/// * `candidates` — the `W` heap returned by `search_layer`.
/// * `m` — maximum number of neighbors to return.
pub(super) fn select_neighbors_heuristic(
    nodes: &[Node],
    _query: &[f32],
    candidates: &BinaryHeap<DistancedNode>,
    m: usize,
    metric: Metric,
) -> Vec<NodeId> {
    // Sort candidates ascending by distance to query (closest first)
    let mut sorted: Vec<DistancedNode> = candidates.iter().cloned().collect();
    sorted.sort_unstable_by(|a, b| {
        a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
    });

    let mut result: Vec<DistancedNode> = Vec::with_capacity(m);

    'outer: for candidate in sorted {
        if result.len() >= m {
            break;
        }
        // Accept the first candidate unconditionally (result set is empty)
        if result.is_empty() {
            result.push(candidate);
            continue;
        }
        // Diversity check: accept candidate only if it is closer to the query
        // than it is to *every* already-selected neighbor.
        // This is the key heuristic from §4 of the HNSW paper.
        for r in &result {
            let d_to_selected = dist(
                &nodes[candidate.id].vector,
                &nodes[r.id].vector,
                metric,
            );
            if candidate.distance >= d_to_selected {
                // candidate is closer to an already-selected neighbor than to q
                // → it would create a "duplicate" direction → skip
                continue 'outer;
            }
        }
        result.push(candidate);
    }

    result.into_iter().map(|dn| dn.id).collect()
}
