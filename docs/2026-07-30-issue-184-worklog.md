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

The follow-up static-path test was first compiled before its test-only
observation functions existed. It failed with missing
`reset_uninitialized_static_family_hits` and
`uninitialized_static_family_hits`, proving that the prior serial-only
differential test could not satisfy the static-specialization acceptance.

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
  `KernelDType` variants accepted by fused replay and every static fused family
  under both `ExecContext::serial()` and `ExecContext::max_threads(2)`.
- A thread-local test-only counter proves nonserial uninitialized replay selects
  unary, binary, ternary, both add-mul forms, three-input, and four-input static
  specializations. The separate serial-context test continues to prove the
  interpreter stays serial under an ambient parallel policy.
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

The checked-in `issue_184_uninit_replay` bench target keeps setup/allocation
outside timed regions, consumes initialized output values with `black_box`,
alternates initialized/uninitialized order for 31 pairs after eight warmups,
emits every raw pair, and reports the paired log-ratio 95 percent upper bound.
Exact shapes are fixed in the runner: 8,388,608 elements for
mul/compare/fused, 1024x8192 for broadcast mul, and 2048x2048 for outer
product.

The rerun was pinned to CPU 60 for t1 and CPUs 60-63 for t4. Before t1,
`mpstat -P 60,61,62,63 1 2` reported 100 percent idle on all four CPUs; the
same was true immediately before t4. The raw 31-pair samples are included in
the PR implementation response. Reproduce them by compiling without affinity,
extracting the exact executable from Cargo's current JSON artifact output, and
pinning only execution:

```bash
cargo bench -p strided-kernel --features parallel \
  --bench issue_184_uninit_replay --no-run --message-format=json \
  > /tmp/issue-184-artifacts.json
bench_exe="$(
  jq -er '
    select(
      .reason == "compiler-artifact"
      and .target.name == "issue_184_uninit_replay"
      and (.target.kind | index("bench"))
      and .executable != null
    )
    | .executable
  ' /tmp/issue-184-artifacts.json | tail -n1
)"
test -x "$bench_exe"
taskset -c 60 "$bench_exe" 1
taskset -c 60-63 "$bench_exe" 4
```

Overwriting the JSON file and filtering the exact target avoids selecting a
stale hashed executable. Main's independent rerun found maximum upper95 values
of 1.170936 at t1 and 1.018668 at t4, also below the +20 percent gate.

| family | t1 initialized / uninit ms | t1 ratio / upper95 | t4 initialized / uninit ms | t4 ratio / upper95 |
|---|---:|---:|---:|---:|
| fused large add-mul | 160.249 / 154.394 | 0.961 / 0.974 | 4.584 / 4.553 | 0.998 / 1.002 |
| mul | 7.946 / 7.951 | 1.002 / 1.008 | 4.556 / 4.583 | 1.003 / 1.007 |
| broadcast mul | 2.994 / 3.245 | 1.088 / 1.111 | 2.318 / 2.335 | 0.995 / 1.013 |
| outer | 1.406 / 1.554 | 1.116 / 1.133 | 0.835 / 0.810 | 0.962 / 0.989 |
| compare | 4.776 / 4.766 | 0.995 / 1.001 | 2.829 / 2.828 | 0.997 / 1.003 |

All candidate-relative upper bounds are below the +20 percent gate; the maximum
is 1.133 for t1 outer product.

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
