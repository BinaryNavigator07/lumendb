//! LumenDB — CLI demo (Milestones 1, 2 & 3).

use lumendb::{
    engine::LumenEngine,
    index::params::HnswParams,
    metrics::{self, Metric},
};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║             LumenDB  v0.1.0                  ║");
    println!("║  Milestone 3 — The Vault  (WAL + Warm Boot)  ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("SIMD backend : {}", metrics::active_backend());
    println!();

    // ── Configuration ─────────────────────────────────────────────────────────
    let db_path   = "/tmp/lumendb_demo";
    let params    = HnswParams::builder().m(16).ef_construction(200).ef_search(50).build();
    let metric    = Metric::Cosine;
    let dim: usize = 256; // smaller dim for a fast demo
    let n: u32     = 5_000;
    let k: usize   = 5;

    // Xorshift-64 RNG (no external crate)
    let mut rng: u64 = 0xdead_c0de_cafe_beef;
    let mut rand_f32 = move || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rand_f32()).collect())
        .collect();

    // ── Session 1: insert and persist ─────────────────────────────────────────
    println!("── Session 1: Insert + persist  ({n} × {dim}-dim) ────────────");
    {
        // Clean up any previous run so the demo is reproducible
        let _ = std::fs::remove_dir_all(db_path);

        let engine = LumenEngine::open(db_path, params.clone(), metric, dim)
            .expect("failed to open engine");

        let t = Instant::now();
        for (i, v) in vectors.iter().enumerate() {
            let meta = serde_json::json!({ "index": i, "source": "demo" });
            engine.insert(v.clone(), meta).expect("insert failed");
        }
        let ms  = t.elapsed().as_millis();
        let ips = n as f64 / t.elapsed().as_secs_f64();
        println!("  Inserted {n} vectors in {ms} ms  ({ips:.0} inserts/s)");
        println!("  Every write fsynced — data is durable.");
        println!("  Index size: {} nodes", engine.len());
        println!();

        // ── Search before shutdown ─────────────────────────────────────────────
        let query  = &vectors[0];
        let hits   = engine.search(query, k).expect("search failed");
        println!("  Pre-shutdown top-{k} for vectors[0]:");
        for (rank, h) in hits.iter().enumerate() {
            println!("    #{:<2}  id={:<5}  dist={:.6}  meta={}", rank+1, h.id, h.distance, h.metadata);
        }
        assert_eq!(hits[0].id, 0, "vectors[0] must be its own nearest neighbour");
        println!("  ✓ Exact-match recall confirmed before shutdown.");
    } // `engine` dropped here; Sled flushes on drop
    println!();

    // ── Session 2: warm boot ──────────────────────────────────────────────────
    println!("── Session 2: Warm boot (process restart simulation) ─────────");
    {
        let t = Instant::now();
        let engine = LumenEngine::open(db_path, params.clone(), metric, dim)
            .expect("failed to reopen engine");
        let boot_ms = t.elapsed().as_millis();

        println!("  Recovered {} vectors in {boot_ms} ms", engine.len());
        assert_eq!(
            engine.len(), n as usize,
            "all {n} vectors must survive the restart"
        );
        println!("  ✓ All {n} vectors recovered.");
        println!();

        // ── Search after warm boot ─────────────────────────────────────────────
        let query  = &vectors[0];
        let t_q    = Instant::now();
        let hits   = engine.search(query, k).expect("post-boot search failed");
        let qus    = t_q.elapsed().as_micros();

        println!("  Post-boot top-{k} for vectors[0]  ({qus} µs):");
        for (rank, h) in hits.iter().enumerate() {
            println!("    #{:<2}  id={:<5}  dist={:.6}  meta={}", rank+1, h.id, h.distance, h.metadata);
        }
        assert_eq!(hits[0].id, 0, "exact-match recall must hold after warm boot");
        println!("  ✓ Exact-match recall confirmed after warm boot.");
        println!();

        // ── Metadata round-trip ───────────────────────────────────────────────
        let meta = engine.vault.get_metadata(42).expect("get_metadata failed");
        println!("  Metadata for id=42: {}", meta.unwrap_or(serde_json::Value::Null));

        // ── Snapshot (FR-5) ───────────────────────────────────────────────────
        let snap_path = "/tmp/lumendb_snapshot";
        let _ = std::fs::remove_dir_all(snap_path);
        engine.snapshot_to(snap_path).expect("snapshot failed");
        println!("  ✓ Hot snapshot written to {snap_path}");
        println!();

        // ── Throughput benchmark ──────────────────────────────────────────────
        println!("── Query throughput  (100 queries × top-{k}) ────────────────");
        let queries: Vec<Vec<f32>> = vectors.iter().take(100).cloned().collect();
        let t_bench = Instant::now();
        let mut sink = 0usize;
        for q in &queries {
            sink += engine.search(q, k).unwrap().len();
        }
        let bench_ms = t_bench.elapsed().as_millis();
        let qps = 100.0 / t_bench.elapsed().as_secs_f64();
        let _ = sink;
        println!("  100 queries in {bench_ms} ms  ({qps:.0} QPS)");
    }

    println!();
    println!("Milestone 3 complete.  Next: Axum REST gateway (Milestone 4).");
}
