//! LumenDB — Milestone 4: The Nexus (Axum REST gateway).
//!
//! # Usage
//!
//! ```sh
//! # Start with defaults (port 7070, no auth, ./lumendb_data)
//! cargo run --release
//!
//! # Custom port + API key
//! cargo run --release -- --port 8080 --api-key secret123
//!
//! # Persist collections somewhere else
//! cargo run --release -- --data-dir /var/lib/lumendb
//! ```
//!
//! # Quick-start (curl)
//!
//! ```sh
//! # Create a 3-dim cosine collection
//! curl -s -X POST http://localhost:7070/v1/collections \
//!   -H 'Content-Type: application/json' \
//!   -d '{"name":"demo","dim":3,"metric":"cosine"}' | jq
//!
//! # Insert a vector
//! curl -s -X POST http://localhost:7070/v1/collections/demo/vectors \
//!   -H 'Content-Type: application/json' \
//!   -d '{"vector":[1.0,0.0,0.0],"metadata":{"label":"x-axis"}}' | jq
//!
//! # Search
//! curl -s -X POST http://localhost:7070/v1/collections/demo/search \
//!   -H 'Content-Type: application/json' \
//!   -d '{"vector":[1.0,0.1,0.0],"k":1}' | jq
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

use lumendb::{
    api::{build_router, AppState},
    LumenEngine,
};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name    = "lumendb",
    version,
    about   = "LumenDB — pure-Rust vector search engine with REST API",
    long_about = None,
)]
struct Args {
    /// TCP port to listen on.
    #[arg(long, default_value = "7070")]
    port: u16,

    /// Host address to bind (use 0.0.0.0 to accept external connections).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Base directory where collections are stored on disk.
    #[arg(long, default_value = "./lumendb_data")]
    data_dir: PathBuf,

    /// Require this value in the `X-API-KEY` request header.
    /// If omitted, the API is unauthenticated (development mode).
    #[arg(long)]
    api_key: Option<String>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Ensure the data directory exists
    std::fs::create_dir_all(&args.data_dir)
        .unwrap_or_else(|e| panic!("cannot create data directory {:?}: {e}", args.data_dir));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("invalid host/port combination");

    let state = AppState::new(args.data_dir.clone(), args.api_key.clone());

    // ── Auto-load existing collections from disk ──────────────────────────────
    // Any subdirectory that contains a valid LumenDB Sled database is opened
    // automatically so collections survive server restarts without needing to
    // be explicitly re-created via the API.
    let mut loaded = 0usize;
    if let Ok(entries) = std::fs::read_dir(&args.data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            match LumenEngine::reopen(&path) {
                Ok(engine) => {
                    let n = engine.len();
                    state.collections.write().insert(name.clone(), Arc::new(engine));
                    eprintln!("  Loaded '{name}'  ({n} vectors)");
                    loaded += 1;
                }
                Err(_) => {} // not a LumenDB directory — silently skip
            }
        }
    }

    let app = build_router(state);

    // ── Banner ────────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════╗");
    println!("║             LumenDB  v{}                  ║", env!("CARGO_PKG_VERSION"));
    println!("║          Milestone 4 — The Nexus             ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("  Data dir  : {:?}", args.data_dir);
    println!("  Listening : http://{addr}");
    if args.api_key.is_some() {
        println!("  Auth      : X-API-KEY header required");
    } else {
        println!("  Auth      : disabled (development mode)");
    }
    if loaded > 0 {
        println!("  Collections loaded from disk: {loaded}");
    }
    println!();
    println!("  Press Ctrl+C to stop.");
    println!();

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("cannot bind to {addr}: {e}"));

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    println!("\nServer shut down gracefully.");
}

// ── Graceful shutdown ─────────────────────────────────────────────────────────

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c    => { eprintln!("Received Ctrl+C — shutting down…"); }
        _ = terminate => { eprintln!("Received SIGTERM — shutting down…"); }
    }
}
