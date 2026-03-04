#[derive(Debug, Clone)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub(crate) ml: f64,
}

impl HnswParams {
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

    #[inline]
    pub(crate) fn m_max(&self, layer: usize) -> usize {
        if layer == 0 { self.m * 2 } else { self.m }
    }
}

impl Default for HnswParams {
    fn default() -> Self {
        Self::new(16, 200, 50)
    }
}

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
