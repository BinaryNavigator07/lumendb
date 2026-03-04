use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use super::node::{DistancedNode, Node, NodeId};
use crate::metrics::{self, Metric};

#[inline]
pub(super) fn dist(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => metrics::cosine_distance(a, b).unwrap_or(f32::MAX),
        Metric::Euclidean => metrics::euclidean_distance(a, b).unwrap_or(f32::MAX),
        Metric::DotProduct => -metrics::dot_product(a, b).unwrap_or(f32::MIN),
    }
}

pub(super) fn search_layer(
    nodes: &[Node],
    query: &[f32],
    entry_points: &[NodeId],
    ef: usize,
    layer: usize,
    metric: Metric,
) -> BinaryHeap<DistancedNode> {
    let mut w: BinaryHeap<DistancedNode> = BinaryHeap::with_capacity(ef + 1);
    let mut candidates: BinaryHeap<Reverse<DistancedNode>> =
        BinaryHeap::with_capacity(ef * 4);
    let mut visited: HashSet<NodeId> = HashSet::with_capacity(ef * 4);

    for &ep in entry_points {
        if visited.insert(ep) {
            let d = dist(query, &nodes[ep].vector, metric);
            w.push(DistancedNode { id: ep, distance: d });
            candidates.push(Reverse(DistancedNode { id: ep, distance: d }));
        }
    }

    while let Some(Reverse(c)) = candidates.pop() {
        let f_dist = w.peek().map_or(f32::MAX, |f| f.distance);
        if c.distance > f_dist && w.len() >= ef {
            break;
        }

        for &nb in nodes[c.id].neighbors_at(layer) {
            if !visited.insert(nb) {
                continue;
            }
            let d = dist(query, &nodes[nb].vector, metric);
            let current_f = w.peek().map_or(f32::MAX, |f| f.distance);

            if d < current_f || w.len() < ef {
                candidates.push(Reverse(DistancedNode { id: nb, distance: d }));
                w.push(DistancedNode { id: nb, distance: d });
                if w.len() > ef {
                    w.pop();
                }
            }
        }
    }

    w
}

pub(super) fn select_neighbors_heuristic(
    nodes: &[Node],
    _query: &[f32],
    candidates: &BinaryHeap<DistancedNode>,
    m: usize,
    metric: Metric,
) -> Vec<NodeId> {
    let mut sorted: Vec<DistancedNode> = candidates.iter().cloned().collect();
    sorted.sort_unstable_by(|a, b| {
        a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
    });

    let mut result: Vec<DistancedNode> = Vec::with_capacity(m);

    'outer: for candidate in sorted {
        if result.len() >= m {
            break;
        }
        if result.is_empty() {
            result.push(candidate);
            continue;
        }
        for r in &result {
            let d_to_selected = dist(
                &nodes[candidate.id].vector,
                &nodes[r.id].vector,
                metric,
            );
            if candidate.distance >= d_to_selected {
                continue 'outer;
            }
        }
        result.push(candidate);
    }

    result.into_iter().map(|dn| dn.id).collect()
}
