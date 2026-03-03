//! HNSW hyper-parameters (NFR-1 tuning knobs).
//!
//! | Parameter         | Typical value | Effect                               |
//! |-------------------|---------------|--------------------------------------|
//! | `m`               | 16            | Connectivity; higher → better recall, more RAM |
//! | `ef_construction` | 200           | Build quality; higher → slower insert, better graph |
//! | `ef_search`       | 50            | Query quality; higher → slower query, better recall |
//!
//! **Rule of thumb:** `ef_construction >= m`, `ef_search >= k` (your top-k).

/// HNSW hyper-parameters.
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Max outgoing edges per node per layer (except layer 0 which uses `2 * m`).
    pub m: usize,

    /// Dynamic candidate list size used during graph construction.
    /// Higher values produce a better-connected graph at the cost of slower inserts.
    pub ef_construction: usize,

    /// Dynamic candidate list size used during search.
    /// Higher values increase recall at the cost of slower queries.
    pub ef_search: usize,

    /// Level normalisation factor: `1 / ln(m)`.
    /// Controls the expected number of nodes per layer (exponential distribution).
    pub(crate) ml: f64,
}

impl HnswParams {
    /// Create a validated parameter set.
    ///
    /// # Panics
    /// Panics if `m < 2` or `ef_construction < m`.
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        assert!(m >= 2, "m must be ≥ 2");
        assert!(
            ef_construction >= m,
            "ef_construction ({ef_construction}) must be ≥ m ({m})"
        );
        Self {
            m,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Maximum allowed edges for a node at `layer`.
    ///
    /// Layer 0 uses `2 * m` to give the base graph extra connectivity,
    /// following the recommendation from the original HNSW paper (§4.1).
    #[inline]
    pub(crate) fn m_max(&self, layer: usize) -> usize {
        if layer == 0 { self.m * 2 } else { self.m }
    }
}

impl Default for HnswParams {
    /// Production-safe defaults: `m=16`, `ef_construction=200`, `ef_search=50`.
    fn default() -> Self {
        Self::new(16, 200, 50)
    }
}

/// Fluent builder for `HnswParams`.
///
/// ```rust
/// use lumendb::index::params::HnswParams;
/// let p = HnswParams::builder().m(32).ef_construction(400).ef_search(100).build();
/// ```
pub struct HnswParamsBuilder {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
}

impl HnswParams {
    pub fn builder() -> HnswParamsBuilder {
        HnswParamsBuilder { m: 16, ef_construction: 200, ef_search: 50 }
    }
}

impl HnswParamsBuilder {
    pub fn m(mut self, m: usize) -> Self { self.m = m; self }
    pub fn ef_construction(mut self, ef: usize) -> Self { self.ef_construction = ef; self }
    pub fn ef_search(mut self, ef: usize) -> Self { self.ef_search = ef; self }
    pub fn build(self) -> HnswParams {
        HnswParams::new(self.m, self.ef_construction, self.ef_search)
    }
}
