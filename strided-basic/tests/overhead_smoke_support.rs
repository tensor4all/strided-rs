//! Runs the shared `overhead_smoke` bench plumbing's own unit tests (#306).
//!
//! The benches live in this crate and in `strided-fused`, and `cargo test` does
//! not build bench targets, so the module's tests would otherwise never run. The
//! file stays in the crate that owns `benches/support/mod.rs`: the check used to
//! live in `strided-fused`, where it reached into `strided-basic/benches/` and so
//! could not compile from the packaged `.crate` (tensor4all/strided-rs#399).
// Only the bench plumbing's own tests run here; the bench entry points stay unused.
#![allow(dead_code)]

#[path = "../benches/support/mod.rs"]
mod support;
