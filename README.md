<div align="center">

# LumenDB

**A vector search engine that ships as a single 1.7 MB binary.**

*Why is your vector database a 500 MB Docker image?*

[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](#license)
[![Binary](https://img.shields.io/badge/Binary-1.7_MB_(Linux_arm64)-brightgreen)](#download)
[![API](https://img.shields.io/badge/API-REST_(Axum)-blue)](#api-reference)

</div>

---

```
╔══════════════════════════════════════════════╗
║             LumenDB  v0.1.0                  ║
║         Zero-Ops Vector Search               ║
╚══════════════════════════════════════════════╝

  Data dir  : ./lumendb_data
  Listening : http://0.0.0.0:7070
  Auth      : X-API-KEY header required
  Collections loaded from disk: 3
```

---

## The Footprint Problem

Every major vector database requires a runtime, a container, and an ops team.

| Database        | Deployment footprint            | Dependencies         |
|-----------------|---------------------------------|----------------------|
| Pinecone        | Managed cloud only              | Internet + credit card |
| Qdrant          | ~150 MB Docker image            | Docker runtime       |
| Weaviate        | ~500 MB Docker image + JVM      | Java, Docker, config |
| Chroma          | Python process + SQLite         | Python 3.8+, pip     |
| **LumenDB**     | **1.7 MB static binary**        | **Nothing**          |

```bash
$ ls -lh target/release/lumendb
-rwxr-xr-x  1.5M  lumendb          ← macOS arm64

$ ls -lh target/x86_64-unknown-linux-musl/release/lumendb
-rwxr-xr-x  1.9M  lumendb          ← Linux x86_64 (statically linked, no glibc)

$ ls -lh target/aarch64-unknown-linux-musl/release/lumendb
-rwxr-xr-x  1.7M  lumendb          ← Linux arm64 / AWS Graviton
```

> `file target/x86_64-unknown-linux-musl/release/lumendb`
> → *ELF 64-bit LSB executable, x86-64, **statically linked**, stripped*

Drop the binary on any Linux server. No Docker. No Python. No JVM. No glibc version hell.

---

## 30-Second Quickstart

```bash
# 1. Start the server
./lumendb --port 7070 --api-key my-secret

# 2. Create a collection
curl -s -X POST http://localhost:7070/v1/collections \
  -H 'X-API-KEY: my-secret' \
  -H 'Content-Type: application/json' \
  -d '{"name":"docs","dim":3,"metric":"cosine"}' | jq

# 3. Insert a vector
curl -s -X POST http://localhost:7070/v1/collections/docs/vectors \
  -H 'X-API-KEY: my-secret' \
  -H 'Content-Type: application/json' \
  -d '{"vector":[1.0,0.0,0.0],"metadata":{"label":"x-axis"}}' | jq

# 4. Search
curl -s -X POST http://localhost:7070/v1/collections/docs/search \
  -H 'X-API-KEY: my-secret' \
  -H 'Content-Type: application/json' \
  -d '{"vector":[0.99,0.1,0.0],"k":5}' | jq
```

```python
# Or use the Python client
from lumendb_client import LumenDB

with LumenDB("http://localhost:7070", api_key="my-secret") as db:
    db.create_collection("docs", dim=1536, metric="cosine")
    db.insert("docs", my_embedding, metadata={"source": "doc_42.pdf"})
    hits = db.search("docs", query_embedding, k=10)
    for h in hits:
        print(f"  [{h.distance:.4f}] {h.metadata}")
```

---

## How the Search Works

LumenDB uses **HNSW** — Hierarchical Navigable Small World graphs. Think of it
as a skip-list for vector space: instead of scanning every vector on a query,
the graph shortcuts you to the answer in *O(log N)* hops.

```
Layer 2 (sparse)   EP ──────────────────────────────► [42]
                    │
Layer 1 (medium)  [0]──[7]──────────[42]──[61]──[88]
                    │                  │
Layer 0 (dense)   [0][1][2]...[7]...[42][43]...[61]...[88]...[N]
                                       ▲
                                  Query lands here
                                  after log(N) hops
```

**Phase 1 — Coarse descent:** Greedily descend from the top layer using beam
width `ef = 1`, narrowing the entry point one layer at a time.

**Phase 2 — Exhaustive beam search at layer 0:** Expand candidates with beam
width `ef_search` (default 50), keeping only the best `k` results.

**Heuristic neighbor selection (Algorithm 4):** When wiring new nodes, LumenDB
prefers *diverse* neighbors over simply the nearest ones. This prevents
"tunnel vision" in the graph and meaningfully improves recall in
high-dimensional spaces.

---

## SIMD-Accelerated Distance Kernels

The bottleneck of any vector search is the distance function. LumenDB uses
hand-written SIMD intrinsics with automatic runtime dispatch — no configuration
needed.

| Architecture       | ISA            | Lane width | Kernel                  |
|--------------------|----------------|------------|-------------------------|
| Apple M1 – M4      | NEON (baseline)| 4 × f32    | `vfmaq_f32` FMA         |
| AWS Graviton 2 / 3 | NEON (baseline)| 4 × f32    | `vfmaq_f32` FMA         |
| Intel / AMD modern | AVX2 + FMA     | 8 × f32    | `_mm256_fmadd_ps`       |
| Any other target   | Scalar         | 1 × f32    | Compiler auto-vectorised |

NEON is **mandatory** on AArch64 — zero runtime overhead for the dispatch.
On x86-64, the CPU feature flags are read once at startup and cached.

### Supported Metrics

| Metric        | Formula                           | Use case                        |
|---------------|-----------------------------------|---------------------------------|
| `cosine`      | `1 − (a · b) / (‖a‖ ‖b‖)`        | Text and image embeddings       |
| `euclidean`   | `√Σ(aᵢ − bᵢ)²`                   | Geospatial, structured features |
| `dot_product` | `−(a · b)`                        | Pre-normalised embeddings       |

---

## Benchmarks

Tested on **Apple M2, 16 GB RAM**, `--release` build (`lto = fat`, `opt-level = 3`).

| Scenario                         | Result        | Notes                                  |
|----------------------------------|---------------|----------------------------------------|
| Insert throughput (256-dim)      | ~12 000 vec/s | Single-threaded, Sled WAL flushed      |
| Warm boot — 5 000 × 256-dim      | ~38 ms        | Full graph recovery from Sled on disk  |
| Search P50 — 5 000 vectors       | < 0.3 ms      | `ef_search=50`, NEON kernels           |
| Binary size (Linux arm64, musl)  | 1.7 MB        | `strip = symbols`, `panic = abort`     |
| Memory at rest (empty server)    | ~4 MB RSS     | tokio + axum idle overhead             |

> Benchmarks are measured workloads, not synthetic micro-benchmarks.
> Recall@10 ≥ 90% at `m=16, ef_construction=200, ef_search=50` on 500-vector
> Cosine datasets verified against exact brute-force KNN in CI.

---

## Durability Guarantees

### Write-Ahead Log

Every insert follows a strict ordering before confirming to the caller:

```
  Client                  LumenDB                 Sled WAL
    │                        │                        │
    │──POST /vectors ────────►│                        │
    │                        │──1. Write vector ──────►│
    │                        │──2. Write metadata ─────►│
    │                        │──3. Write graph node ───►│
    │                        │──4. fsync() ────────────►│
    │                        │◄─ confirmed ────────────│
    │◄─ 200 {"id": 42} ──────│                        │
```

If the process crashes between steps 1–3, the recovery scanner in `replay()`
re-runs the HNSW insertion for any vector whose graph node is absent — the
slow-path recovery. Vectors with intact graph state are restored directly
without touching the HNSW algorithm — the fast-path.

### Warm Boot

```bash
# Kill and restart — all collections reload automatically
pkill lumendb
./lumendb --data-dir ./lumendb_data

  Loaded 'products'  (50000 vectors)
  Loaded 'articles'  (12000 vectors)
  Loaded 'users'     (8000 vectors)
```

No rebuild. No re-indexing. The graph is persisted alongside the vectors.

---

## Deployment

### Single Binary (Recommended)

```bash
# Copy to any Linux server — no runtime required
scp target/x86_64-unknown-linux-musl/release/lumendb user@server:/usr/local/bin/

# Run with a systemd unit or just in a screen session
lumendb --host 0.0.0.0 --port 7070 --data-dir /var/lib/lumendb --api-key $SECRET
```

### CLI Reference

| Flag          | Default            | Description                                    |
|---------------|--------------------|------------------------------------------------|
| `--host`      | `127.0.0.1`        | Bind address (`0.0.0.0` for external access)   |
| `--port`      | `7070`             | TCP port                                       |
| `--data-dir`  | `./lumendb_data`   | Root directory for all collections             |
| `--api-key`   | *(none)*           | Require `X-API-KEY` header on all `/v1/` routes|

### Building from Source

```bash
git clone https://github.com/BinaryNavigator07/lumendb
cd lumendb
cargo build --release

# Cross-compile for Linux servers from macOS
brew install zig
cargo install cargo-zigbuild
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
cargo zigbuild --release --target x86_64-unknown-linux-musl
cargo zigbuild --release --target aarch64-unknown-linux-musl
```

---

## API Reference

All `/v1/` routes require `X-API-KEY: <key>` when the server is started with
`--api-key`. The `/health` route is always public.

### Health

```http
GET /health
→ {"status": "ok", "version": "0.1.0"}
```

### Collections

```http
POST /v1/collections
Body: {
  "name":            "products",   // required: no '/', '\', or '.'
  "dim":             1536,         // required: > 0
  "metric":          "cosine",     // "cosine" | "euclidean" | "dot_product"
  "m":               16,           // optional HNSW param (default 16)
  "ef_construction": 200,          // optional HNSW param (default 200)
  "ef_search":       50            // optional HNSW param (default 50)
}
→ 201 {"name": "products", "dim": 1536, "metric": "cosine", "count": 0}

GET  /v1/collections/:name
→ 200 {"name": "products", "dim": 1536, "metric": "cosine", "count": 42000}

DELETE /v1/collections/:name
→ 204 No Content
```

### Vectors

```http
POST /v1/collections/:name/vectors
Body: {
  "vector":   [0.12, -0.34, ...],  // must match collection dim
  "metadata": {"sku": "ABC", "price": 9.99}  // any JSON object
}
→ 200 {"id": 42}

GET /v1/collections/:name/vectors/:id
→ 200 {"id": 42, "metadata": {"sku": "ABC", "price": 9.99}}

POST /v1/collections/:name/search
Body: {
  "vector": [0.11, -0.33, ...],  // query embedding
  "k": 10                        // top-k results (k ≥ 1)
}
→ 200 {
    "results": [
      {"id": 42, "distance": 0.0031, "metadata": {"sku": "ABC", "price": 9.99}},
      ...
    ]
  }
```

### Error Responses

All errors return a consistent JSON body with an HTTP status code:

```json
{"error": "collection 'foo' not found", "code": "NOT_FOUND"}
```

| HTTP Status | Code             | Meaning                                     |
|-------------|------------------|---------------------------------------------|
| 401         | `UNAUTHORIZED`   | Missing or invalid `X-API-KEY`              |
| 404         | `NOT_FOUND`      | Collection or vector does not exist         |
| 409         | `CONFLICT`       | Collection name already in use              |
| 422         | `BAD_REQUEST`    | Dimension mismatch, zero vector, invalid k  |
| 500         | `INTERNAL_ERROR` | Storage failure                             |

---

## Python Client

```bash
pip install requests
```

```python
from lumendb_client import LumenDB
import pandas as pd

# ── Context manager — connection is closed automatically ──────────────────────
with LumenDB("http://localhost:7070", api_key="my-secret") as db:

    # Create once
    db.create_collection("articles", dim=1536, metric="cosine")

    # Insert a single vector
    id = db.insert("articles",
                   vector=my_embedding,
                   metadata={"title": "LumenDB is fast", "year": 2025})

    # Search → list[SearchHit]
    hits = db.search("articles", query_embedding, k=10)
    for h in hits:
        print(f"  {h.id:>6}  dist={h.distance:.4f}  {h.metadata['title']}")

    # ── Pandas bulk-insert ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "embedding": embeddings,          # list[list[float]] or ndarray
        "title":     titles,
        "category":  categories,
    })
    ids = db.insert_dataframe(
        "articles", df,
        vector_col="embedding",           # which column holds the vectors
        metadata_cols=["title", "category"],
        show_progress=True,               # prints "Inserted 1000/5000"
    )

    # ── Search → DataFrame ────────────────────────────────────────────────────
    result_df = db.search_to_dataframe("articles", query_embedding, k=5)
    print(result_df[["id", "distance", "title"]])
```

**Client API surface:**

| Class / Method                          | Description                                    |
|-----------------------------------------|------------------------------------------------|
| `LumenDB(url, *, api_key, timeout)`     | Main client. Implements `__enter__`/`__exit__` |
| `db.create_collection(name, dim, ...)`  | Create collection → `CollectionInfo`           |
| `db.collection(name)`                   | Scoped `Collection` handle                     |
| `db.insert(col, vector, metadata)`      | Insert one vector → `int` ID                   |
| `db.search(col, query, k)`             | KNN search → `list[SearchHit]`                 |
| `db.insert_dataframe(col, df, ...)`    | Bulk insert from pandas → `list[int]`          |
| `db.search_to_dataframe(col, query, k)`| Search results as `pd.DataFrame`               |
| `SearchHit.id / .distance / .metadata` | Frozen dataclass, one result row               |
| `LumenDBError.status_code / .code`     | Typed exception for non-2xx responses          |

---

## HNSW Tuning Guide

| Parameter        | Default | Effect                                              |
|------------------|---------|-----------------------------------------------------|
| `m`              | 16      | Edges per node. ↑ = better recall, more RAM         |
| `ef_construction`| 200     | Build quality. ↑ = slower inserts, better graph     |
| `ef_search`      | 50      | Query beam width. ↑ = slower queries, better recall |

**Rule of thumb:**
- `ef_construction ≥ m` (enforced by the server)
- `ef_search ≥ k` (your top-k) for reliable results
- For 1536-dim OpenAI embeddings: `m=32, ef_construction=400, ef_search=100`

---

## Roadmap

### Free — Open Core

- [x] HNSW index with SIMD kernels (NEON / AVX2)
- [x] Sled-backed WAL + warm boot
- [x] Axum REST API with API key auth
- [x] Python client with Pandas integration
- [x] Hot snapshot export / import
- [x] Statically-linked Linux binaries (x86_64 + arm64)
- [ ] Filtered search (metadata predicate push-down)
- [ ] Batch insert endpoint (`POST /v1/collections/:name/vectors/batch`)
- [ ] OpenAPI / Swagger spec

### Pro — Planned Commercial Features

| Feature                     | Why enterprises pay for it                                    |
|-----------------------------|---------------------------------------------------------------|
| **S3 Snapshot Sync**        | Automatic offsite backups — no custom scripts needed          |
| **Web UI**                  | Visual collection browser, search playground, metrics graphs  |
| **Multi-node Clustering**   | Horizontal scale beyond a single machine                      |
| **Role-based API Keys**     | Read-only keys for inference, write keys for ingestion        |
| **Prometheus Metrics**      | `/metrics` endpoint for Grafana dashboards                    |
| **Disk-based ANN**          | Collections larger than RAM using memory-mapped files         |

---

## Architecture Overview

```
  HTTP Client
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │                  Axum REST Gateway                  │
  │   auth_layer (X-API-KEY) → route → handler          │
  └────────────────────────┬────────────────────────────┘
                           │  Arc<LumenEngine>
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │                   LumenEngine                       │
  │                                                     │
  │   insert()  ──►  HnswIndex (write lock)             │
  │                       │                             │
  │                  insert_inner()                     │
  │                  returns (id, modified_nodes)       │
  │                       │                             │
  │                  SledVault.put()                    │
  │                  SledVault.update_graph_node()  ×N  │
  │                  SledVault.flush()   ← fsync        │
  │                                                     │
  │   search()  ──►  HnswIndex (read lock, concurrent)  │
  │                  search_inner()  →  SledVault.get_metadata() │
  └─────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │              Sled (embedded KV store)               │
  │                                                     │
  │  "vectors"      NodeId → packed f32 bytes           │
  │  "metadata"     NodeId → JSON bytes                 │
  │  "graph_nodes"  NodeId → bincode(level, neighbors)  │
  │  "config"       "v1"   → bincode(dim, metric, HNSW) │
  └─────────────────────────────────────────────────────┘
```

---

## License

Copyright © 2025 Umair Sajid. All rights reserved.

The source code in this repository is **not open source**. Viewing the source
for evaluation purposes is permitted. Redistribution, modification, or use in
commercial products requires a written license.

For licensing enquiries: **umaiesajid@gmail.com*
