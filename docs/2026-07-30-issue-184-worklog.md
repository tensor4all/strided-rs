# Issue #184 worklog

## RED

The original test-first compile failed because the five requested APIs did not
exist:

```text
cargo test -p strided-kernel --test issue_184_uninit_elementwise
```

The review revision added pointer-input call sites before changing
`ErasedFusedPlan::execute_uninit`; the focused test then failed with an expected
`ErasedRawStridedRef` versus `ErasedRawStridedPtr` signature mismatch.

The all-dtype differential test subsequently exposed a real scalar-path bug:
`i32::MAX * 2` panicked in debug mode at `map_view.rs` instead of preserving the
required wrapping integer semantics. The shared initialized/uninitialized
multiply primitive now handles `i32` and `i64` with wrapping multiplication.

## Implementation

- `ErasedFusedPlan::execute_uninit` now accepts `ErasedRawStridedPtr`, validates
  dtype, shape, injectivity, bounds, and overlap before forming typed shared
  descriptors, and dispatches fixed arities without a replay-time input `Vec`.
- Serial fused replay bypasses every ambient map/zip path. Nonserial replay runs
  under the explicit `ExecContext`.
- Static fused classification and traversal are shared between initialized and
  uninitialized outputs through a private output writer. The existing unary,
  binary, ternary, add-mul, three-input, four-input, and interpreter paths are
  preserved.
- Identity multiplication, broadcast multiplication, and batched outer product
  share the initialized planner, contiguous-range handling, and raw-pointer SIMD
  stores. No initialized destination slice is formed over `MaybeUninit`.
- Typed uninitialized APIs use one allocation-free reachable-byte overlap
  validator. Public docs state pre-write validation, full initialization on
  `Ok`, and partial-initialization/drop-safety behavior after panic.
- `/.codegraph/` is tracked in the repository root `.gitignore`; the local index
  remains present and ignored.

## Acceptance tests

- Initialized/uninitialized differential coverage includes all seven
  `KernelDType` variants accepted by fused replay and every static fused family.
- Typed mul/broadcast/outer covers f32, f64, i32, i64, Complex32, and Complex64;
  compare covers all five operations over f32, f64, i32, i64, and bool.
- Raw erased overlap, dtype, shape, and injectivity errors preserve destination
  sentinels.
- A threshold-crossing test uses `MINTHREADLENGTH + 65` elements and observes
  worker IDs: serial remains on the caller despite an outer Rayon policy;
  `max_threads(2)` uses more than one and at most two workers.
- The partial-write panic test drops the `MaybeUninit` output without assuming
  unwritten elements initialized and is suitable for Miri.
- The counting allocator compares specialization and fallback at serial and
  bounded-parallel t4; uninitialized replay never exceeds initialized replay.

## Paired release benchmark

The temporary focused runner kept setup/allocation outside timed regions,
alternated initialized/uninitialized order for 31 pairs, and reported the
paired log-ratio 95 percent upper bound. It was pinned to CPU 60 for t1 and
CPUs 60-63 for t4. `mpstat` showed 99 percent or greater idle on those CPUs
before the run.

```text
taskset -c 60   ./target/release/examples/issue_184_pair_bench 1
taskset -c 60-63 ./target/release/examples/issue_184_pair_bench 4
```

| family | t1 initialized / uninit ms | t1 ratio / upper95 | t4 initialized / uninit ms | t4 ratio / upper95 |
|---|---:|---:|---:|---:|
| fused large add-mul | 163.662 / 159.383 | 0.974 / 0.985 | 4.584 / 4.589 | 1.000 / 1.004 |
| mul | 7.578 / 7.617 | 1.006 / 1.018 | 4.614 / 4.642 | 1.005 / 1.008 |
| broadcast mul | 5.015 / 4.576 | 0.914 / 0.920 | 3.023 / 3.023 | 0.999 / 1.006 |
| outer | 1.470 / 1.568 | 1.060 / 1.087 | 0.801 / 0.779 | 0.966 / 0.998 |
| compare | 4.759 / 4.759 | 0.999 / 1.007 | 2.862 / 2.848 | 0.999 / 1.005 |

All upper bounds are below the +20 percent gate. The temporary runner was
removed after recording the evidence; permanent benchmark programs remain in
the benchmark suite per repository policy.

## Verification

- `cargo fmt --all -- --check`
- `cargo test -p strided-kernel --features parallel --test issue_184_uninit_elementwise`
- `cargo test -p strided-kernel --no-default-features --features parallel --test issue_184_uninit_elementwise`
- `cargo test -p strided-kernel --features parallel --test copy_plan_alloc -- --test-threads=1`
- `cargo test -p strided-kernel --features parallel --lib`
- `cargo test --workspace`
- `cargo doc --workspace --no-deps`

Miri remains unavailable on the installed stable toolchain because its component
is not installed. A diagnostic `cargo clippy -- -D warnings` run also reaches
pre-existing workspace lint failures outside this change; clippy is not a
configured repository gate.
