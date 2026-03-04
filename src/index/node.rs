use std::cmp::Ordering;

pub type NodeId = usize;

#[derive(Debug)]
pub struct Node {
    pub id: NodeId,
    pub vector: Vec<f32>,
    pub level: usize,
    pub neighbors: Vec<Vec<NodeId>>,
}

impl Node {
    pub fn new(id: NodeId, vector: Vec<f32>, level: usize) -> Self {
        let neighbors = vec![Vec::new(); level + 1];
        Self { id, vector, level, neighbors }
    }

    #[inline]
    pub fn neighbors_at(&self, layer: usize) -> &[NodeId] {
        self.neighbors.get(layer).map(Vec::as_slice).unwrap_or(&[])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DistancedNode {
    pub id: NodeId,
    pub distance: f32,
}

impl Eq for DistancedNode {}

impl PartialOrd for DistancedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistancedNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: NodeId,
    pub distance: f32,
}
