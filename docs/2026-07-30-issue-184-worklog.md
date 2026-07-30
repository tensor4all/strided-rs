# Issue #184 worklog

## RED

Added `strided-kernel/tests/issue_184_uninit_elementwise.rs` before production
changes. The focused compile/test command failed because the requested APIs did
not exist:

```text
cargo test -p strided-kernel --test issue_184_uninit_elementwise
```

Observed compiler failures included unresolved imports for
`batched_outer_product_into_uninit`, `broadcast_mul_into_uninit`,
`compare_into_uninit`, and `mul_into_uninit`, plus no
`ErasedFusedPlan::execute_uninit` method.

## Implementation

- Added the five requested public APIs and reused the existing bounded layout
  validation and fused traversal planners.
- Added typed `MaybeUninit` write kernels and erased replay without forming an
  initialized destination slice; erased replay checks dtype, overlap, and
  injectivity before writes and preserves the explicit execution context.
- Added focused differential, bool-validity, wrapping-integer, bounded-parallel,
  and pre-write sentinel tests.
- Extended the existing counting-allocator integration test. The uninitialized
  fused replay did not allocate beyond its initialized counterpart in the
  measured serial case (initialized 216 allocations / 8 calls, uninitialized
  48 allocations / 8 calls in that test binary).

## Verification and benchmarks

- `cargo fmt --all -- --check`: pass.
- `cargo test -p strided-kernel --features parallel --test issue_184_uninit_elementwise`: pass (4 tests).
- `cargo test -p strided-kernel --features parallel --test copy_plan_alloc -- --exact execute_is_allocation_free_up_to_rank_limit`: pass.
- `cargo test --workspace`: pass.
- `cargo doc --workspace --no-deps`: pass with the existing broken-link warning for `StridedError::DimensionMismatch`.
- `cargo miri test -p strided-kernel --test issue_184_uninit_elementwise --features parallel`: unavailable because the installed stable toolchain lacks the Miri component; the focused test is structured to drop `MaybeUninit` storage without assuming failed writes are initialized.
- Release smoke benchmark: `cargo bench -p strided-kernel --features parallel --bench mul_pytorch_compare`, `STRIDED_KERNEL_MUL_BENCH_PROFILE=smoke`, f64, pinned `RAYON_NUM_THREADS=1` and `4`.
  Representative medians (ms): elementwise 0.004600 / 0.002980; outer 0.004420 / 0.002920; compact batched outer 0.002040 / 0.001570; noncompact batched outer 0.005141 / 0.003580 (t1 / t4). These are existing initialized upstream rows; exact initialized-vs-uninitialized public rows must be adopted by the downstream benchmark suite.
