//! # LumenDB
//!
//! A pure-Rust, zero-dependency vector search engine designed for embedding
//! inside AI applications.  Milestone 1 ships the low-level distance-metric
//! kernel, SIMD-optimised for Apple Silicon (NEON) and modern x86-64 (AVX2 +
//! FMA), with a portable scalar fallback for every other target.

pub mod error;
pub mod metrics;

pub use error::LumenError;
