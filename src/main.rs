//! LumenDB — CLI entry point.
//!
//! Currently serves as a smoke-test harness for Milestone 1.
//! The Axum REST layer (Milestone 4) will replace this binary's body.

use lumendb::metrics::{self, Metric};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════╗");
    println!("║         LumenDB  v0.1.0              ║");
    println!("║   Vector Search Engine — Milestone 1 ║");
    println!("╚══════════════════════════════════════╝");
    println!();
    println!("SIMD backend : {}", metrics::active_backend());
    println!();

    // ── 1. Basic metric demo ──────────────────────────────────────────────────
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let c = vec![-1.0f32, 0.0, 0.0];

    println!("── Metric demo (3-dim) ──────────────────");
    println!(
        "  cosine(a, a)  = {:.6}  (expect  1.0)",
        metrics::cosine_similarity(&a, &a).unwrap()
    );
    println!(
        "  cosine(a, b)  = {:.6}  (expect  0.0)",
        metrics::cosine_similarity(&a, &b).unwrap()
    );
    println!(
        "  cosine(a, c)  = {:.6}  (expect -1.0)",
        metrics::cosine_similarity(&a, &c).unwrap()
    );
    println!(
        "  euclidean(a, b) = {:.6}  (expect  1.414…)",
        metrics::euclidean_distance(&a, &b).unwrap()
    );
    println!(
        "  dot(a, a)     = {:.6}  (expect  1.0)",
        metrics::dot_product(&a, &a).unwrap()
    );
    println!();

    // ── 2. Pre-normalisation shortcut ─────────────────────────────────────────
    let raw_a = vec![3.0f32, 4.0, 0.0];
    let raw_b = vec![0.0f32, 3.0, 4.0];

    let cosine_before = metrics::cosine_similarity(&raw_a, &raw_b).unwrap();

    let mut norm_a = raw_a.clone();
    let mut norm_b = raw_b.clone();
    metrics::normalize(&mut norm_a).unwrap();
    metrics::normalize(&mut norm_b).unwrap();
    let dot_after = metrics::dot_product(&norm_a, &norm_b).unwrap();

    println!("── Pre-normalisation shortcut ───────────");
    println!("  cosine(raw_a, raw_b)        = {cosine_before:.6}");
    println!("  dot(normalised_a, norm_b)   = {dot_after:.6}");
    println!("  Δ = {:.2e}  (should be ~0)", (cosine_before - dot_after).abs());
    println!();

    // ── 3. Throughput benchmark — 1536-dim (OpenAI embedding size) ────────────
    let dim = 1536;
    let iterations = 100_000u32;

    let vec_a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
    let vec_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).cos()).collect();

    // Cosine
    let t0 = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        sink += metrics::compute(Metric::Cosine, &vec_a, &vec_b).unwrap();
    }
    let elapsed_cosine = t0.elapsed();

    // Euclidean
    let t1 = Instant::now();
    for _ in 0..iterations {
        sink += metrics::compute(Metric::Euclidean, &vec_a, &vec_b).unwrap();
    }
    let elapsed_l2 = t1.elapsed();

    // Dot (fastest when vectors are pre-normalised)
    let t2 = Instant::now();
    for _ in 0..iterations {
        sink += metrics::compute(Metric::DotProduct, &vec_a, &vec_b).unwrap();
    }
    let elapsed_dot = t2.elapsed();

    let _ = sink; // prevent dead-code elimination

    let ns_per_op = |d: std::time::Duration| d.as_nanos() as f64 / iterations as f64;
    let mops = |d: std::time::Duration| iterations as f64 / d.as_secs_f64() / 1_000_000.0;

    println!("── Throughput benchmark  ({dim}-dim × {iterations} iterations) ──");
    println!(
        "  Cosine      {:>8.1} ns/op  ({:.2} Mops/s)",
        ns_per_op(elapsed_cosine),
        mops(elapsed_cosine)
    );
    println!(
        "  Euclidean   {:>8.1} ns/op  ({:.2} Mops/s)",
        ns_per_op(elapsed_l2),
        mops(elapsed_l2)
    );
    println!(
        "  DotProduct  {:>8.1} ns/op  ({:.2} Mops/s)",
        ns_per_op(elapsed_dot),
        mops(elapsed_dot)
    );
    println!();
    println!("Milestone 1 complete.  Next: HNSW index (Milestone 2).");
}
