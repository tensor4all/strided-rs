# strided-rs Publish Preparation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a user-facing `strided-rs` facade crate and prepare the workspace metadata, dependencies, README examples, and CI checks for a future crates.io release without publishing anything.

**Architecture:** The workspace will continue to publish individual low-level crates, while `strided-rs` becomes the recommended entry point that re-exports the core APIs. README examples will live in `strided-rs/README.md` and be included in crate docs so `cargo test -p strided-rs --doc` verifies them.

**Tech Stack:** Rust workspace, Cargo workspace metadata/dependencies, rustdoc doctests, GitHub Actions.

---

### Task 1: Add the Facade Crate

**Files:**
- Create: `strided-rs/Cargo.toml`
- Create: `strided-rs/src/lib.rs`
- Create: `strided-rs/README.md`
- Modify: `Cargo.toml`

**Steps:**
1. Add `strided-rs` to `workspace.members`.
2. Create a package named `strided-rs` with library crate `strided_rs`.
3. Re-export core APIs from `strided-view`, `strided-kernel`, `strided-perm`, `strided-einsum2`, and `strided-opteinsum`.
4. Add optional `mdarray` and `ndarray` features that re-export `mdarray-opteinsum` and `ndarray-opteinsum`.
5. Include `README.md` as crate docs with `#![doc = include_str!("../README.md")]`.

### Task 2: Centralize Cargo Metadata and Dependencies

**Files:**
- Modify: `Cargo.toml`
- Modify: all workspace member `Cargo.toml` files

**Steps:**
1. Add `[workspace.package]` for version, authors, license, repository, and edition.
2. Add `[workspace.dependencies]` for shared external dependencies and internal crates with `version + path`.
3. Convert package metadata and dependency declarations to `workspace = true` where practical.
4. Keep explicit `rust-version` where current crates already need different MSRV values.
5. Do not run `cargo publish`.

### Task 3: Make README Examples Executable

**Files:**
- Modify: `README.md`
- Create/Modify: `strided-rs/README.md`
- Modify: `.github/workflows/ci.yml`

**Steps:**
1. Move the user-facing Quick Start to `strided-rs/README.md`.
2. Keep root README as workspace guidance and link to the facade crate README.
3. Ensure Rust examples in `strided-rs/README.md` are complete doctests.
4. Add an explicit CI step for `cargo test -p strided-rs --doc`.

### Task 4: Verify

**Commands:**
- `cargo fmt --check`
- `cargo test --workspace`
- `cargo test -p strided-rs --doc`
- `cargo publish --workspace --dry-run` only as a dry-run, never without `--dry-run`

**Expected Result:** Formatting and tests pass. Publish dry-run either passes or reports only issues that require actual registry state or final release decisions.
