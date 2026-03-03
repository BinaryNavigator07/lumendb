
use lumendb::{
    index::{params::HnswParams, HnswIndex},
    metrics::{self, Metric},
};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════╗");
    println!("║           LumenDB  v0.1.0                ║");
    println!("║  Milestone 2 — HNSW Index (The Weaver)   ║");
    println!("╚══════════════════════════════════════════╝");
    println!();
    println!("SIMD backend : {}", metrics::active_backend());
    println!();

    // ── Configuration ─────────────────────────────────────────────────────────
    let params = HnswParams::builder()
        .m(16)
        .ef_construction(200)
        .ef_search(50)
        .build();
    let metric = Metric::Cosine;
    let dim    = 1536; // OpenAI text-embedding-3-small dimensions
    let n      = 10_000u32;
    let k      = 10;

    // ── Build phase ───────────────────────────────────────────────────────────
    println!("── Build  ({n} × {dim}-dim vectors, M=16, ef_c=200) ────────");
    let index = HnswIndex::new(params, metric);

    // Reproducible pseudo-random vectors via Xorshift-64
    let mut rng: u64 = 0xfeed_c0de_babe_cafe;
    let mut next = move || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| next()).collect())
        .collect();

    let t_build = Instant::now();
    for v in &vectors {
        index.insert(v.clone());
    }
    let build_ms = t_build.elapsed().as_millis();
    let ins_per_sec = (n as f64 / t_build.elapsed().as_secs_f64()) as u64;
    println!("  Indexed {n} vectors in {build_ms} ms  ({ins_per_sec} inserts/s)");
    println!("  Index size    : {} nodes", index.len());
    println!("  Vector dim    : {}",       index.dim().unwrap());
    println!();

    // ── Query phase ───────────────────────────────────────────────────────────
    println!("── Query  (top-{k}, ef_search=50) ──────────────────────────");

    // Query 1: exact lookup — expect 0.0 cosine distance for the first vector
    let query = vectors[0].clone();
    let t_q = Instant::now();
    let results = index.search(&query, k);
    let query_us = t_q.elapsed().as_micros();

    println!("  Query time    : {query_us} µs");
    println!("  Top-{k} results (expect id=0 at distance≈0.0):");
    for (rank, r) in results.iter().enumerate() {
        println!("    #{:<2}  id={:<5}  dist={:.6}", rank + 1, r.id, r.distance);
    }
    println!();

    // Verify the closest result is the query vector itself
    assert_eq!(
        results[0].id, 0,
        "Expected id=0 as nearest to itself, got id={}",
        results[0].id
    );
    assert!(
        results[0].distance < 1e-4,
        "Expected near-zero distance, got {}",
        results[0].distance
    );
    println!("  ✓ Exact-match recall confirmed (id=0, dist≈0)");
    println!();

    // ── Throughput benchmark ──────────────────────────────────────────────────
    println!("── Throughput benchmark  ({n} random queries × 1 iter) ──────");

    // Pre-generate query vectors
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..dim).map(|_| next()).collect())
        .collect();

    let t_bench = Instant::now();
    let mut sink = 0usize;
    for q in &queries {
        sink += index.search(q, k).len();
    }
    let bench_ms = t_bench.elapsed().as_millis();
    let qps = (queries.len() as f64 / t_bench.elapsed().as_secs_f64()) as u64;
    let _ = sink;

    println!("  100 queries in {bench_ms} ms  ({qps} QPS)");
    println!();
    println!("Milestone 2 complete.  Next: Sled Vault + WAL (Milestone 3).");
}
