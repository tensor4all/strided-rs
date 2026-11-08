# strided-rs Repository Rules

Durable rules for this workspace. The cross-repository rules are **not vendored
here**: read them from
[`tensor4all/tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
(or the sibling checkout `../tensor4all-agent-rules`), starting at
[`rules/index.md`](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md),
and load only the files the change needs.

Each section below names the shared section it extends and carries only the
strided-rs part. Where a section here and the shared text differ, this
repository's more specific rule wins
([loading policy](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md)).
`AGENTS.md` holds the workspace layout, commands, and constants;
`docs/PROVENANCE_AND_CITATION_POLICY.md` holds the per-crate provenance.

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

Shared: [`common/repository.md` § Public Surface](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#public-surface),
[§ Layering](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#layering),
[§ API Evolution](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#api-evolution).

- Keep public APIs intentionally small. Implementation modules, planning
  helpers, loop-order utilities, macro kernels, execution trees, backend glue,
  and test/benchmark helpers should be private or `pub(crate)` unless external
  users are expected to call them directly.
- `#[doc(hidden)] pub` is not a substitute for privacy. Use it only for
  explicitly supported macro output, required trait contracts, or documented
  extension contracts.
- When the public API changes, audit README, rustdoc, examples, and benchmark
  code for stale names, deleted paths, and stale capability claims.
- Facade policy: `strided-kernel` is the top of the stack and its crate root is
  a thin re-export — `pub use strided_basic::*` plus its own `erased` layer. An
  item therefore enters that root by being declared in `strided-basic` (or in
  the erased layer), never by being declared in the facade. `strided-fused` and
  `strided-nn` build on `strided-basic` instead of re-exporting it wholesale, and
  none of them duplicates an implementation that `strided-basic` already owns.
- A compatibility re-export is an exception with a removal condition, not a
  pattern: it names the issue that removes it and the release line it survives
  (the `strided-fused` names kept at the `strided-kernel` root for the 0.4 line,
  [#261](https://github.com/tensor4all/strided-rs/issues/261)).

## Suffix Vocabulary

Strided-specific naming rule from
[#220](https://github.com/tensor4all/strided-rs/issues/220) (maintainer-ruled
2026-11-08, kept as a strided rule by
[#216](https://github.com/tensor4all/strided-rs/issues/216); not a copy of a
shared tensor4all rule). Mutation semantics are visible in the name, and no two
names may suggest the same operation. Pre-1.0, a rename deletes the old name
outright: no compatibility shim, alias, or deprecated wrapper. The shared
principle it specializes is
[`common/repository.md` § Output-Update Naming Principle](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#output-update-naming-principle).

| Suffix | Semantics | Names |
|--------|-----------|-------|
| `*_acc` | Read-modify-write: reads `dest[i]` and accumulates into it | `add_acc`, `mul_acc`, `axpy_acc`, `fma_acc` |
| `*_into` | Overwrite: never reads the previous `dest[i]` | `copy_into`, `copy_scale_into`, `copy_conj_into`, `map_into`, `mul_into` |
| `*_view` | Metadata-only: reads and writes no element | `diagonal_view` |

- An accumulation must not be named `*_into`, and an overwrite must not be
  named `*_acc`. A pair such as `mul` / `mul_into` that names two different
  operations is a defect.
- `*_into` means "overwrite" except in the update family, which reads the
  previous `dest[i]` and updates it in place by definition: `map_update_into`,
  `zip_update2_into`, `zip_update3_into`. Their rustdoc states the read.
- A `*_view.rs` module name means "operates on views", not metadata-only; only
  the `_view` suffix carries that meaning.
- Residual names that do not follow the table yet are follow-ups, not
  exemptions: `axpy_raw` / `axpy_conj_raw` read and accumulate into `dest`,
  `axpby_accum` spells accumulation `_accum`, and `copy_scale_raw` /
  `copy_scale_conj_raw` overwrite (`_raw` marks the borrowed-layout family,
  not the mutation semantics).
- The `_uninit` siblings are deferred to after v0.3.0 and are not part of this
  rename: `map_into_uninit`, `zip_map*_into_uninit`, `copy_into_uninit`,
  `reduce*_uninit`, `ErasedCopyPlan::execute_uninit`.

## Public Boundary Safety

Shared: [`rust/numerical.md` § Error Handling](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/rust/numerical.md#error-handling),
[§ Typed Errors](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/rust/numerical.md#typed-errors),
[`common/docs-and-tests.md` § Public Result Error Documentation Gate](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/docs-and-tests.md#public-result-error-documentation-gate).

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

Shared: [`rust/index.md` § Unsafe Code Boundary](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/rust/index.md#unsafe-code-boundary),
[`rust/performance.md` § Indexing And Slicing](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/rust/performance.md#indexing-and-slicing),
[`common/repository.md` § Invariant Markers](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#invariant-markers).

- Fast paths must have explicit fallback behavior. For copy/transpose/scale
  paths, cover zero, identity/copy, tiled/specialized, parallel, and generic
  fallback branches where applicable.
- Do not preserve a fast path that is systematically slower than the raw
  pointer naive baseline for the same layout and dtype without documenting why
  it remains useful.
- Dtype-erased and op-erased entries (`erased_*`, `Erased*Plan`) must resolve
  the runtime operation, dtype, and conjugation flags once per call, or once
  per worker range, and then run a loop monomorphized for that operation. Do
  not pass a closure that matches on a runtime op value (for example
  `|a, b| T::zip(op, a, b)`) into a loop whose trip count scales with tensor
  elements: the per-element op reload and branch prevents vectorization. This
  applies to serial fast paths and to parallel leaves alike
  ([#269](https://github.com/tensor4all/strided-rs/issues/269)).
- Erased and typed APIs mirror each other. Every `erased_*` entry and every
  `Erased*Plan` has a typed counterpart with the same operand order, `Exec`
  position, and output contract; adding one side of a pair alone is an
  incomplete change. The pair spans two crates — the erased entry in
  `strided-kernel/src/erased.rs` (`erased_map_into`, `:81`) and the typed one in
  `strided-basic` (`map_into`, `strided-basic/src/map_view.rs:1251`) — so the
  signature and the doc comment are what keep them in step, not co-location.
- An erased entry must stay within 1.25x of the typed entry for the same
  operation, dtype, layout, and thread count. A larger gap is a defect, not an
  accepted cost of type erasure.

## Materialization And Copies

Shared: [`rust/performance.md` § Allocation And Copies](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/rust/performance.md#allocation-and-copies),
[`rust/performance.md` § Uninitialized And Scratch Acquisition](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/rust/performance.md#uninitialized-and-scratch-acquisition).

- When a copy or materialization is required by an output contiguity contract or
  external ABI boundary, make that boundary explicit and benchmark it.
- Metadata transformations (`permute`, `transpose_2d`, `adjoint_2d`, `conj`,
  `broadcast`) never touch element data and stay allocation-free up to
  `strided_view::INLINE_RANK`. They change dims/strides/offset, and `conj` and
  `adjoint` additionally compose an `ElementOp` (`conj` returns
  `StridedView<T, Op::ComposeConj>`), so the whole metadata contract is the half
  that stays; keep it when adding one of them.
- A method that could alias a mutable view takes `self` by value or returns a
  borrowing view so the borrow checker rejects the alias
  (`permute(self, ...)` at `strided-view/src/view.rs:1049`, `as_view(&self)` at
  `:1090`). Do not answer a borrow-checker complaint about a mutable view with a
  copy.

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
  issue. The marker and its review contract are
  [`common/repository.md` § Invariant Markers](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/repository.md#invariant-markers).

## CPU Threading Contract

Shared: [`rust/performance.md` § CPU Parallelism](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/rust/performance.md#cpu-parallelism),
[`common/performance.md` § Parallel Scaling](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/performance.md#parallel-scaling).
The strided side of the placement contract, including its thresholds and tests,
is owned by [`docs/design/cpu-exec-placement.md`](docs/design/cpu-exec-placement.md).

- The `threadteam::Exec` placement token is the only threading mechanism here.
  Only `threadteam` starts tasks. Every other crate fans out through the token
  it is handed (`fork`, `fork_sized`, `for_ranges`, `broadcast`), takes it by
  value as the last parameter of each entry that can fan out, and never stores
  it. A foreign thread pool is reached through `Exec::external`, and a caller
  that hands a team to a third-party library holds it with `Exec::team` /
  `Exec::team_with`; no strided crate holds one. `cargo deny check bans` rejects
  a direct `rayon` dependency elsewhere (`deny.toml`).
- There is no element-count threading threshold. A kernel multiplies its element
  count by the serial throughput of its class
  (`strided_basic::execution::KernelClass::ns_per_element`) and asks the token:
  `Exec::pays` for the gate, `Exec::width_for` for the width, through
  `strided_basic::execution::{parallel_width, tree_width}`. Do not add an unrelated
  threshold for a similar kernel, and do not read
  `strided_basic::execution::SMALL_TENSOR_THRESHOLD` as one: that constant bounds
  the cache-blocking fast path, not parallelism.
- If the active thread count is one, call the serial kernel directly rather
  than entering pool machinery (`Exec::is_serial`). Avoid startup and scheduler
  overhead for `Exec::serial()` or single-thread pools.
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
  reductions) either asks the token for a width or carries the
  sequential-rationale comment required above.
- Provider-owned threading such as BLAS/OpenMP must be controlled by the
  provider's thread variables. Do not mix independent thread policies inside a
  single benchmark run without documenting it.

## HPTT And faer Integration

Shared: [`common/provenance.md` § Record References In The Source](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/provenance.md#record-references-in-the-source).

- `strided-perm/src/hptt/` is a port of HPTT (Springer, Su, Bientinesi), which
  is why the crate is `(MIT OR Apache-2.0) AND BSD-3-Clause`
  (`strided-perm/Cargo.toml:5`). The port keeps the upstream attribution and the
  reproduced BSD-3-Clause text in `THIRD-PARTY-LICENSES`, and a new ported file
  states the upstream source it came from in its file-level comment.
- Keep the port's structure — `plan/`, `execute/`, `macro_kernel/`,
  `micro_kernel/` mirror the upstream blocking scheme. Changing the blocking or
  the micro/macro kernel shape is a performance change and needs the evidence
  the section below requires.
- `faer`, `t4a-faer` and `spindle` are denied dependencies (`deny.toml`). A
  consumer that wants faer's parallelism gets a plain width through
  `Exec::external`; the dependency stays with the consumer, never with a
  strided crate.
- `docs/faer-kernel-writing-guide.md` and `docs/faer_design.md` are design
  analysis for writing kernels. They are not a dependency, and nothing there is
  an API or licence commitment.

## Performance And Benchmark Discipline

Shared: [`common/performance.md` § General Checklist](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/performance.md#general-checklist),
[§ Evidence](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/performance.md#evidence),
[`common/docs-and-tests.md` § Benchmarks](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/docs-and-tests.md#benchmarks),
[`benchmark/measurement.md` § One measurement at a time](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/benchmark/measurement.md#one-measurement-at-a-time).

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
- Work that adds or specializes a fast path must include a generic
  fast-path-miss case and representative rank scaling. Other performance claims
  must include representative rank scaling and the relevant fallback or layout
  cases without inventing a nonexistent fast-path miss. Alternatively,
  explicitly scope the claim and link a residual generic-path issue. Routine
  production kernel refactors without a performance claim have no benchmark
  obligation solely because they touch kernel code.
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
