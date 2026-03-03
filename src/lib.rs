//! # LumenDB
//!
//! A pure-Rust, zero-C-dependency vector search engine.
//!
//! ## Milestones
//! | # | Module      | Status  | Description                              |
//! |---|-------------|---------|------------------------------------------|
//! | 1 | `metrics`   | Done    | SIMD-optimised distance kernels (NEON/AVX2) |
//! | 2 | `index`     | Done    | HNSW graph index (The Weaver)            |
//! | 3 | `storage`   | Done    | Sled-backed WAL + hot snapshots          |
//! | 4 | `api`       | Done    | Axum REST gateway (The Nexus)            |

pub mod api;
pub mod engine;
pub mod error;
pub mod index;
pub mod metrics;
pub mod storage;

pub use engine::{LumenEngine, SearchHit};
pub use error::LumenError;
pub use index::HnswIndex;
