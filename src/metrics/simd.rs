#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut result = vaddvq_f32(acc);

    for i in (chunks * 4)..len {
        result += a[i] * b[i];
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn norm_sq_neon(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        acc = vfmaq_f32(acc, va, va);
    }

    let mut result = vaddvq_f32(acc);
    for i in (chunks * 4)..len {
        result += a[i] * a[i];
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }

    let mut result = vaddvq_f32(acc);
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);

    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);

    let shuf2 = _mm_movehl_ps(shuf, sum64);
    _mm_cvtss_f32(_mm_add_ss(sum64, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut result = hsum256(acc);
    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn norm_sq_avx2(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, va, acc);
    }

    let mut result = hsum256(acc);
    for i in (chunks * 8)..len {
        result += a[i] * a[i];
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    let mut result = hsum256(acc);
    for i in (chunks * 8)..len {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::scalar;

    const EPS: f32 = 1e-5;

    fn nearly_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn sample() -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (1..=128).map(|x| x as f32 * 0.01).collect();
        let b: Vec<f32> = (1..=128).map(|x| (129 - x) as f32 * 0.01).collect();
        (a, b)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_dot_matches_scalar() {
        let (a, b) = sample();
        let expected = scalar::dot(&a, &b);
        let got = unsafe { dot_neon(&a, &b) };
        assert!(nearly_eq(got, expected), "NEON dot: {got} vs scalar: {expected}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_norm_sq_matches_scalar() {
        let (a, _) = sample();
        let expected = scalar::norm_sq(&a);
        let got = unsafe { norm_sq_neon(&a) };
        assert!(nearly_eq(got, expected), "NEON norm_sq: {got} vs scalar: {expected}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_l2_sq_matches_scalar() {
        let (a, b) = sample();
        let expected = scalar::l2_sq(&a, &b);
        let got = unsafe { l2_sq_neon(&a, &b) };
        assert!(nearly_eq(got, expected), "NEON l2_sq: {got} vs scalar: {expected}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_dot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("AVX2/FMA not available — skipping");
            return;
        }
        let (a, b) = sample();
        let expected = scalar::dot(&a, &b);
        let got = unsafe { dot_avx2(&a, &b) };
        assert!(nearly_eq(got, expected), "AVX2 dot: {got} vs scalar: {expected}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_l2_sq_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (a, b) = sample();
        let expected = scalar::l2_sq(&a, &b);
        let got = unsafe { l2_sq_avx2(&a, &b) };
        assert!(nearly_eq(got, expected), "AVX2 l2_sq: {got} vs scalar: {expected}");
    }
}
