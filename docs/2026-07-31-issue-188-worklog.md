# Issue 188 worklog

## Scope

Add overwrite-only contraction entry points whose destination is borrowed as
`MaybeUninit<T>`. The initialized beta APIs remain unchanged.

## 2026-07-31

- Raised the workspace `cblas-inject` minimum to `0.1.2`.
- Added explicit-context `einsum2_into_uninit`,
  `einsum2_into_owned_uninit`, `dot_general_into_uninit`, and
  `bgemm_raw_strided_into_uninit`.
- Added checked output injectivity, checked offset traversal, conservative
  input/output overlap rejection, empty-output no-write behavior, trace-axis
  reduction, and tests for matrix output and pre-write validation failure.
- Migrated `strided-opteinsum` intermediate pool acquisition to
  `MaybeUninit` and finalize only after a successful overwrite call.
- Kept the default Faer-only `strided-opteinsum` path on initialized storage
  until Faer provides a typed overwrite API; direct Faer uninitialized entry
  points now return a typed `Unsupported` error instead of silently routing to
  the naive provider.
- Added a shared raw-GEMM preflight before label construction, temporary
  allocation, or provider dispatch; it checks full dimension agreement,
  injectivity, conservative aliasing, checked products, and BLAS sizes.
- Made column-major uninitialized allocation and opteinsum pool sizing
  propagate checked overflow errors instead of using unchecked products.

## Verification

- `cargo fmt --all`
- Added coverage tests for the Faer-gated validation/naive fallback, the
  `dot_general_into_uninit` forwarding boundary, and both direct and
  temporary raw-output finalize paths.
- `CARGO_BUILD_JOBS=4 RUSTFLAGS='-C link-arg=-Wl,--threads=1' cargo llvm-cov --workspace --json --output-path /tmp/coverage-188.json`
- `python3 scripts/check-coverage.py /tmp/coverage-188.json` (54/54 files;
  `contiguous.rs` 86.98%, `dot_general.rs` 86.53%, `uninit.rs` 82.17%)
- `cargo test -p strided-einsum2 --lib`
- `cargo test -p strided-opteinsum --lib`
- `cargo test -p strided-einsum2 --no-default-features`
- `cargo test -p strided-einsum2 --no-default-features --features faer`
- `cargo test -p strided-einsum2 --no-default-features --features blas`
- `cargo test -p strided-einsum2 --no-default-features --features blas-inject`
- `cargo test -p strided-opteinsum --no-default-features --features faer`
- `cargo test -p strided-opteinsum --no-default-features --features blas`
- `cargo test -p strided-opteinsum --no-default-features --features blas-inject`
- `cargo check -p strided-einsum2 --no-default-features`
- `cargo check -p strided-einsum2 --no-default-features --features faer`
- `cargo check -p strided-einsum2 --no-default-features --features blas-inject`
- `cargo check -p strided-einsum2 --no-default-features --features blas`
- `cargo test -p strided-einsum2 --no-default-features --features blas-inject --test blas_inject_fallback -- --test-threads=1`
- `cargo fmt --all -- --check`
- `git diff --check`
- `CARGO_BUILD_JOBS=4 RUSTFLAGS='-C link-arg=-Wl,--threads=1' cargo test --workspace`

The first unrestricted workspace run reached the doctest linker but hit a
Rust 1.97 `rust-lld` bus error under the full local parallel load. The same
workspace command passed after limiting Cargo jobs and linker threads. The
repository has no configured rules-review script; package clippy remains
non-clean on the current Rust 1.97 toolchain because of pre-existing lints in
the workspace, so it is not used as the PR gate.

## Remaining work

- Faer uninitialized output remains blocked by #195: the current Faer
  `MatMut::from_raw_parts_mut` contract does not permit forming `MatMut<T>` over
  `MaybeUninit<T>`, including for `Accum::Replace`. No cast or zero-fill
  workaround is allowed.
- The naive, system BLAS, and injected BLAS overwrite paths are wired through
  the public uninitialized APIs. Their feature-specific tests cover direct and
  non-contiguous temporary/writeback execution, including a poisoned-C
  zgemm regression. Faer remains explicitly blocked on strided-rs#195; its
  initialized compatibility path is tested separately.
