pub mod scalar;
pub mod simd;

use crate::LumenError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    DotProduct,
    Euclidean,
    Cosine,
}

#[inline]
fn check_same_len(a: &[f32], b: &[f32]) -> Result<(), LumenError> {
    if a.is_empty() || b.is_empty() {
        return Err(LumenError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(LumenError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(())
}

#[allow(unreachable_code)]
#[inline]
fn raw_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return unsafe { simd::dot_neon(a, b) };

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { simd::dot_avx2(a, b) };
    }

    scalar::dot(a, b)
}

#[allow(unreachable_code)]
#[inline]
fn raw_norm_sq(a: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return unsafe { simd::norm_sq_neon(a) };

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { simd::norm_sq_avx2(a) };
    }

    scalar::norm_sq(a)
}

#[allow(unreachable_code)]
#[inline]
fn raw_l2_sq(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return unsafe { simd::l2_sq_neon(a, b) };

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { simd::l2_sq_avx2(a, b) };
    }

    scalar::l2_sq(a, b)
}

pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, LumenError> {
    check_same_len(a, b)?;
    Ok(raw_dot(a, b))
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, LumenError> {
    check_same_len(a, b)?;
    Ok(raw_l2_sq(a, b).sqrt())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, LumenError> {
    check_same_len(a, b)?;

    let dot = raw_dot(a, b);
    let norm_a = raw_norm_sq(a).sqrt();
    let norm_b = raw_norm_sq(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(LumenError::ZeroVector);
    }

    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, LumenError> {
    Ok(1.0 - cosine_similarity(a, b)?)
}

pub fn compute(metric: Metric, a: &[f32], b: &[f32]) -> Result<f32, LumenError> {
    match metric {
        Metric::DotProduct => dot_product(a, b),
        Metric::Euclidean => euclidean_distance(a, b),
        Metric::Cosine => cosine_similarity(a, b),
    }
}

pub fn normalize(v: &mut [f32]) -> Result<(), LumenError> {
    if v.is_empty() {
        return Err(LumenError::EmptyVector);
    }
    let norm = raw_norm_sq(v).sqrt();
    if norm == 0.0 {
        return Err(LumenError::ZeroVector);
    }
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
    Ok(())
}

pub fn active_backend() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    return "NEON (AArch64)";

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "AVX2 + FMA (x86-64)";
        }
        return "scalar (x86-64, no AVX2)";
    }

    #[allow(unreachable_code)]
    "scalar (generic)"
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn nearly(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn dot_orthogonal() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(nearly(dot_product(&a, &b).unwrap(), 0.0));
    }

    #[test]
    fn dot_parallel() {
        let a = [2.0f32, 0.0, 0.0];
        assert!(nearly(dot_product(&a, &a).unwrap(), 4.0));
    }

    #[test]
    fn dot_dim_mismatch() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32];
        assert_eq!(
            dot_product(&a, &b),
            Err(LumenError::DimensionMismatch { expected: 2, got: 1 })
        );
    }

    #[test]
    fn euclidean_pythagorean() {
        let a = [0.0f32, 0.0];
        let b = [3.0f32, 4.0];
        assert!(nearly(euclidean_distance(&a, &b).unwrap(), 5.0));
    }

    #[test]
    fn euclidean_same_vector() {
        let v = [1.0f32, 2.0, 3.0];
        assert!(nearly(euclidean_distance(&v, &v).unwrap(), 0.0));
    }

    #[test]
    fn cosine_identical() {
        let v = [1.0f32, 2.0, 3.0];
        assert!(nearly(cosine_similarity(&v, &v).unwrap(), 1.0));
    }

    #[test]
    fn cosine_opposite() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert!(nearly(cosine_similarity(&a, &b).unwrap(), -1.0));
    }

    #[test]
    fn cosine_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(nearly(cosine_similarity(&a, &b).unwrap(), 0.0));
    }

    #[test]
    fn cosine_zero_vector_err() {
        let a = [0.0f32, 0.0];
        let b = [1.0f32, 0.0];
        assert_eq!(cosine_similarity(&a, &b), Err(LumenError::ZeroVector));
    }

    #[test]
    fn normalize_produces_unit_vector() {
        let mut v = [3.0f32, 4.0];
        normalize(&mut v).unwrap();
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!(nearly(norm, 1.0));
    }

    #[test]
    fn normalize_then_dot_equals_cosine() {
        let a_raw = [1.0f32, 2.0, 3.0, 4.0];
        let b_raw = [4.0f32, 3.0, 2.0, 1.0];

        let cosine = cosine_similarity(&a_raw, &b_raw).unwrap();

        let mut a = a_raw;
        let mut b = b_raw;
        normalize(&mut a).unwrap();
        normalize(&mut b).unwrap();
        let dot = dot_product(&a, &b).unwrap();

        assert!(nearly(dot, cosine), "dot={dot}, cosine={cosine}");
    }

    #[test]
    fn compute_all_metrics_run() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(compute(Metric::DotProduct, &a, &b).is_ok());
        assert!(compute(Metric::Euclidean, &a, &b).is_ok());
        assert!(compute(Metric::Cosine, &a, &b).is_ok());
    }

    #[test]
    fn high_dim_1536_smoke() {
        let a: Vec<f32> = (0..1536).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i as f32).cos()).collect();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim >= -1.0 && sim <= 1.0);
    }
}
