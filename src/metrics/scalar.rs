#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn norm_sq(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_identity() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert_eq!(dot(&a, &b), 0.0);
        assert_eq!(dot(&a, &a), 1.0);
    }

    #[test]
    fn norm_sq_basic() {
        let v = [3.0f32, 4.0];
        assert!((norm_sq(&v) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn l2_sq_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((l2_sq(&a, &b) - 27.0).abs() < 1e-6);
    }
}
