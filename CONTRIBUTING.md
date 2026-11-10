<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to strided-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `strided-rs`:

## Local facts

- **MSRV**: none is declared — no `rust-version` in the workspace `Cargo.toml` or in any member crate, and no
  `rust-toolchain.toml`. CI pins `stable` (`dtolnay/rust-toolchain@stable`) and `nightly` for the two Miri lanes only,
  so there is no MSRV claim for CI to hold and no `msrv` job.
- **Feature combinations CI builds**: default features; then four lanes, each asserting the feature set it actually
  compiled (`scripts/assert-lane-features.py`): `--no-default-features`; the parallel set
  (`strided-basic/parallel,strided-kernel/parallel,strided-fused/parallel,strided-perm/parallel`) with
  `strided-basic`'s default `simd` on, on both Linux and macOS; and that parallel set under
  `--no-default-features`, i.e. no SIMD. Clippy lints `--workspace --all-targets` and the `parallel` feature, and
  `package-tests` builds each published crate with `--all-features`.
- **Platforms claimed**: `ubuntu-latest` and `macos-latest` for the `test` job and the feature lanes. `fmt`, `clippy`,
  `doc`, `coverage`, the Miri lanes and `package-tests` run on `ubuntu-latest` only. Windows is neither built nor
  tested.
- **Local gate**: `cargo test --workspace`.

## Anything here that departs from the shared rules

One does: the `msrv` gate is not run, because no `rust-version` is declared for it to hold CI to. Declaring one and
adding the job is the way to close that gap; this file records the state and does not exempt the repository from the
rest of the gates.

This repository adds: the `--profile ci` build profile the test and feature lanes use (locally the default profiles
are fine), the Miri lanes, per-file coverage thresholds (`coverage-thresholds.json`), and a `placement` lane that
rejects a direct `rayon` dependency outside `threadteam`.
