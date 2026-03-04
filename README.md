# LumenDB

LumenDB is a pure-Rust, zero-C-dependency vector search engine with a built-in REST API gateway.

## Features

- **Pure-Rust & Zero C-dependencies**: Simple to build and run anywhere.
- **HNSW Indexing**: Uses Hierarchical Navigable Small World (HNSW) graphs for fast approximate nearest neighbor search.
- **SIMD-Optimized Metrics**: Distance kernels (like Euclidean, Cosine, Dot Product) are optimized using SIMD instructions (NEON/AVX2).
- **Persistent Storage**: Sled-backed Write-Ahead Log (WAL) and hot snapshots ensure that collections survive server restarts automatically.
- **REST API Subsystem (The Nexus)**: Axum-based HTTP gateway for simple integration.

## Architecture Milestones

The engine is built around several core modules:

| # | Module | Status | Description                                 |
| - | ------ | ------ | ------------------------------------------- |
| 1 | `metrics` | Done | SIMD-optimised distance kernels (NEON/AVX2) |
| 2 | `index` | Done | HNSW graph index (The Weaver)               |
| 3 | `storage` | Done | Sled-backed WAL + hot snapshots             |
| 4 | `api` | Done | Axum REST gateway (The Nexus)               |

## Getting Started

### Prerequisites

- Rust (edition 2021) and Cargo installed.

### Installation & Execution

LumenDB exposes a built-in HTTP server listening on port `7070` by default. Data is stored in `./lumendb_data`.

```bash
# Start with defaults (port 7070, no auth, ./lumendb_data)
cargo run --release

# Run with custom port and an API key for authentication
cargo run --release -- --port 8080 --api-key secret123

# Persist collections in a specific directory
cargo run --release -- --data-dir /var/lib/lumendb
```

Any existing collections found in the designated data directory are automatically loaded upon server restart.

### CLI Options

- `--port`: TCP port to listen on (default `7070`).
- `--host`: Host address to bind (use `0.0.0.0` to accept external connections, default `127.0.0.1`).
- `--data-dir`: Base directory where collections are stored on disk (default `./lumendb_data`).
- `--api-key`: Require this value in the `X-API-KEY` request header. If omitted, the API is unauthenticated.

## API Quick-start

LumenDB provides a simple REST API to interact with vector collections.

### 1. Create a Collection

Create a new collection specifying its dimensionality and the distance metric (e.g., `cosine`, `euclidean`, `dot`).

```bash
curl -s -X POST http://localhost:7070/v1/collections \
  -H 'Content-Type: application/json' \
  -d '{"name":"demo","dim":3,"metric":"cosine"}' | jq
```

### 2. Insert a Vector

Insert a vector along with any arbitrary JSON metadata.

```bash
curl -s -X POST http://localhost:7070/v1/collections/demo/vectors \
  -H 'Content-Type: application/json' \
  -d '{"vector":[1.0, 0.0, 0.0], "metadata": {"label": "x-axis"}}' | jq
```

### 3. Search for Nearest Neighbors

Query the collection to find nearest neighbors based on the defined distance metric. `k` indicates the number of neighbors to return.

```bash
curl -s -X POST http://localhost:7070/v1/collections/demo/search \
  -H 'Content-Type: application/json' \
  -d '{"vector":[1.0, 0.1, 0.0], "k": 1}' | jq
```

## License

Copyright (c) Rizwan Nisar.
