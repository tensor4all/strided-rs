# strided-rs Repository Rules

These rules are adapted from `tenferro-rs/REPOSITORY_RULES.md` for the current
strided-rs workspace. Apply them in addition to the shared tensor4all rules.

## Einsum Maintenance Ownership

- `strided-einsum2` is the minimum binary CPU einsum implementation.
- `strided-opteinsum` is its maintained N-ary frontend.
- `mdarray-opteinsum` and `ndarray-opteinsum` remain maintained adapters.

## Deprecated Tree Freeze

- Everything under `deprecated/` remains retired per
  [#199](https://github.com/tensor4all/strided-rs/issues/199).
- Do not land new features, refactors, or performance work there. Only fixes
  that protect the current tenferro pin belong there, and only when the
  tenferro-side absorption cannot deliver them first.
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
- Dtype-erased and op-erased entries (`erased_*`, `Erased*Plan`) must resolve
  the runtime operation, dtype, and conjugation flags once per call, or once
  per worker range, and then run a loop monomorphized for that operation. Do
  not pass a closure that matches on a runtime op value (for example
  `|a, b| T::zip(op, a, b)`) into a loop whose trip count scales with tensor
  elements: the per-element op reload and branch prevents vectorization. This
  applies to serial fast paths and to parallel leaves alike
  ([#269](https://github.com/tensor4all/strided-rs/issues/269)).
- An erased entry must stay within 1.25x of the typed entry for the same
  operation, dtype, layout, and thread count. A larger gap is a defect, not an
  accepted cost of type erasure.

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
- After plan validation, do not rebuild full coordinates or call rank-scanning
  checked-offset helpers inside loops whose trip count scales with tensor
  elements, windows, or reduced elements. A serial traversal may decode once
  at traversal start; parallel or blocked traversal may decode once per worker
  range or traversal block, then must advance coordinates and source/destination
  offsets incrementally. Plan-time injectivity checks, offset-table
  construction, and worker-range/block initialization are exempt from this
  per-element restriction. Data-dependent index reads are allowed; static
  layout mapping remains subject to the restriction. A deliberate replay
  exception requires a nearby `// INVARIANT:` rationale plus dated
  worklog/benchmark evidence, or a narrowly scoped claim linked to a residual
  issue.

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
- The parallel branch must run the same inner kernel quality as the serial
  branch: lane or SIMD leaves, multiple accumulators for reductions, and
  per-op monomorphized loops. A tuned kernel that is reachable only under a
  serial execution context, while the parallel branch falls back to a generic
  single-accumulator or runtime-dispatch loop, is a defect. For tensor-sized
  benchmark cases, four threads must not be slower than one thread.
- Early-return fast paths (contiguous `copy_nonoverlapping`, contiguous run
  copies, single-segment plans) must not bypass the parallel branch for
  tensor-sized inputs. Every prepared plan whose work scales with tensor
  elements (copy, slice, reverse, concatenate, pad, dynamic slice, gather,
  reductions) either has a parallel branch under the repository threshold or
  carries the sequential-rationale comment required above.
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
- Work that adds or specializes a fast path must include a generic
  fast-path-miss case and representative rank scaling. Other performance claims
  must include representative rank scaling and the relevant fallback or layout
  cases without inventing a nonexistent fast-path miss. Alternatively,
  explicitly scope the claim and link a residual generic-path issue. Routine
  production kernel refactors without a performance claim have no benchmark
  obligation solely because they touch kernel code.
- Benchmark scaling across representative tensor sizes, shapes, layouts, dtypes,
  and thread counts. A single fixed-size speedup is not enough evidence for a
  performance-sensitive change.
- Naive baselines must be credible. For contiguous hot loops, prefer raw
  pointer baselines over high-level indexing baselines.
- Every public erased operation family (elementwise unary and binary ops,
  reductions including max and min over all axes and single axes, structural
  copy plans) must have rows in the benchmark suite's kernel scaling page at one
  and four threads, paired against the typed entry, a raw pointer baseline,
  and Julia. Adding an operation family, or routing an existing one through a
  new entry point, adds its rows in the same change or links a benchmark suite
  PR. Defects in #269 surfaced only downstream in tenferro because the suite
  measured typed entries at one thread only.
- Ternary and predicated families (`select`, `clamp`) count as operation
  families too. Their regression rows compare against a plain slice loop rather
  than the typed `zip_map3_into`, so a slow loop shared by both paths still
  shows, and include a size well past the last level cache: a per-step NaN test
  in `clamp` matched the loop at 2^20 elements and cost 1.7x at 2^23.
- Validation that reads operand data, such as the `bool` byte check, is part of
  the kernel's cost. Write it as a fold without an early exit so it vectorizes,
  and skip it where the type already guarantees validity (the sealed typed
  constructors). A byte by byte `find` over the predicate once cost more than
  the `select` it guarded, and no binary row could see it.
- Benchmark harnesses must enforce and verify the thread count they report: use
  a bounded pool or an explicit execution context and assert the effective
  count at startup. A thread flag or environment variable that is requested but
  not verified does not count as pinning.
- A claim that a contiguous hot loop is vectorized or dispatch-free should be
  backed by codegen evidence (disassembly of the hot loop) in the PR, not only
  by a single timing.
- Keep setup and allocation out of timed regions unless the benchmark name and
  documentation explicitly say setup cost is included.

