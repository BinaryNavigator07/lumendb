//! # LumenDB
//!
//! A pure-Rust, zero-C-dependency vector search engine.
//!
//! ## Milestones
//! | # | Module      | Status     | Description                         |
//! |---|-------------|------------|-------------------------------------|
//! | 1 | `metrics`   | Done       | SIMD-optimised distance kernels      |
//! | 2 | `index`     | Done       | HNSW graph index (The Weaver)        |
//! | 3 | *(storage)* | Pending    | Sled-backed WAL + snapshots          |
//! | 4 | *(api)*     | Pending    | Axum REST / gRPC gateway             |

pub mod error;
pub mod index;
pub mod metrics;

pub use error::LumenError;
pub use index::HnswIndex;
