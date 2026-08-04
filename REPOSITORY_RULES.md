# strided-rs Repository Rules

These rules are adapted from `tenferro-rs/REPOSITORY_RULES.md` for the current
strided-rs workspace. Apply them in addition to the shared tensor4all rules.

## Retired Crate Freeze

- `strided-einsum2`, `strided-opteinsum`, `mdarray-opteinsum`,
  `ndarray-opteinsum`, and everything under `deprecated/` are retired per
  [#199](https://github.com/tensor4all/strided-rs/issues/199). Contraction is
  owned by tenferro (`tenferro-einsum` plans, `tenferro-cpu` executes).
- Do not land new features, refactors, or performance work in the retired
  crates. Only fixes that protect the current tenferro pin belong here, and
  only when the tenferro-side absorption cannot deliver them first.
- Deprecation notices are exempt: README banners, crate-level and item-level
  doc comments, `#[deprecated]` attributes, and `Cargo.toml` metadata may
  change freely.
- A maintainer waiver label is the escape hatch for a pin-protecting fix.

## Public Surface Discipline

- Keep public APIs intentionally small. Implementation modules, planning
  helpers, loop-order utilities, macro kernels, execution trees, backend glue,
  and test/benchmark helpers should be private or `pub(crate)` unless external
  users are expected to call them directly.
- Public APIs are durable contracts. Before adding or keeping a `pub` item,
  check whether it is useful outside this repository and whether the crate is
  prepared to support its semantics.
- `#[doc(hidden)] pub` is not a substitute for privacy. Use it only for
  explicitly supported macro output, required trait contracts, or documented
  extension contracts.
- When the public API changes, audit README, rustdoc, examples, and benchmark
  code for stale names, deleted paths, and stale capability claims.

## Public Boundary Safety

- User-reachable tensor/view/kernel APIs must validate rank, shape, dtype,
  stride/layout, output shape, and aliasing preconditions before no-op
  shortcuts, allocation, launch planning, or unsafe pointer loops.
- Shape products, byte lengths, strides, offsets, and allocation sizes must use
  checked or otherwise justified arithmetic before conversion to pointer
  offsets or allocation lengths.
- Publicly reachable library paths must not turn invalid input into `panic`,
  `unwrap`, `expect`, unchecked indexing, or debug-only assertions. Return a
  crate error type unless the invariant is truly internal and proven locally.
- Repeated public-boundary validation should live in shared helpers or prepared
  metadata types when sibling operations need the same checks.

## Unsafe And Fast-Path Boundaries

- Keep unsafe pointer arithmetic close to the validation that proves it safe,
  and cover new unsafe branches with focused tests.
- Fast paths must have explicit fallback behavior. For copy/transpose/scale
  paths, cover zero, identity/copy, tiled/specialized, parallel, and generic
  fallback branches where applicable.
- Do not preserve a fast path that is systematically slower than the raw
  pointer naive baseline for the same layout and dtype without documenting why
  it remains useful.
- After validation, hot loops should not repeat avoidable per-element range
  checks. Prefer direct slice iteration, pre-loop assertions, or localized
  unchecked access only when the invariant is clear and tested.

## Materialization And Copies

- Prefer metadata-only views and strided/backend-native operations over hidden
  dense materialization.
- Do not allocate dense temporary buffers whose memory or time scales with an
  unconstrained tensor product unless the API explicitly documents that copy
  boundary.
- Do not zero-initialize buffers that are immediately fully overwritten.
- When a copy or materialization is required by an output contiguity contract or
  external ABI boundary, make that boundary explicit and benchmark it.

## Layout And Copy Semantics

- Preserve column-major semantics unless a function explicitly documents a
  different layout contract.
- Public flat-buffer constructors, exports, examples, FFI contracts, and docs
  must state or preserve the active layout semantics.
- Avoid per-element flat-to-multi-index decoding in tensor-sized loops when
  incremental offsets, blocked traversal, or precomputed stride tables can be
  used.

## CPU Threading Contract

- Tensor-sized CPU kernels compiled with a `parallel` feature must use the
  repository threading threshold consistently. Do not introduce unrelated
  thresholds for similar kernels.
- If the active thread count is one, call the serial kernel directly rather
  than entering Rayon/OpenMP parallel machinery. Avoid thread startup and
  scheduler overhead for `RAYON_NUM_THREADS=1` or single-thread pools.
- If a tensor-sized CPU operation remains a dedicated sequential loop because no
  strided/backend-native parallel primitive fits the indexing pattern yet, add
  a nearby comment naming that rationale.
- Provider-owned threading such as BLAS/OpenMP must be controlled by the
  provider's thread variables. Do not mix independent thread policies inside a
  single benchmark run without documenting it.

## Performance And Benchmark Discipline

- This workspace's own regression benchmarks live in `<crate>/benches/`. Keep
  them there. The rule is about location, not about which harness they use.
- Cross-repository comparisons, competitor and cross-language baselines, and
  any *published* benchmark results belong in
  `tensor4all/strided-rs-benchmark-suite`, not in this repository.
- Crate READMEs and rustdoc must not carry performance tables. Numbers go stale
  as soon as the hardware or the kernel changes; document usage, features, and
  API contracts, and link to the benchmark suite for results. Dated worklogs and
  design records under `docs/` may quote measurements as evidence for a
  decision, provided they state the date and the machine.
- Use release-mode benchmarks for performance claims. Pin thread counts and
  backend configuration, and do not run benchmark jobs concurrently.
- Benchmark scaling across representative tensor sizes, shapes, layouts, dtypes,
  and thread counts. A single fixed-size speedup is not enough evidence for a
  performance-sensitive change.
- Naive baselines must be credible. For contiguous hot loops, prefer raw
  pointer baselines over high-level indexing baselines.
- Keep setup and allocation out of timed regions unless the benchmark name and
  documentation explicitly say setup cost is included.

