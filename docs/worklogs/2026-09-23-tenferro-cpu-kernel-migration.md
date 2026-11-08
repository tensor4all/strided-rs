# tenferro CPU elementwise and max/min kernels moved into strided

Date: 2026-09-23. Machine: Apple M5 Max (18 cores, 128 GB), macOS 26.5.1,
rustc 1.96.0, release builds, `RAYON_NUM_THREADS=1`.

## Scope

Moved from tenferro-rs:

- `tenferro-internal-cpu-kernels/src/elementwise.rs`: the typed replay of
  unary maps, binary zips, comparisons, select, clamp, broadcast multiply and
  the lazy outer product layout.
- `tenferro-cpu/src/reduction.rs`: the `reduce_axis` based max/min fold.

Not moved: GEMM/dot, linalg, FFT, runtime and resource management, and the C
ABI. Those are separate efforts.

## Design

The new entries follow the existing erased descriptor style:

- `erased_{map,zip,compare,select,clamp,broadcast_mul}_into_uninit` in
  `strided-kernel/src/erased/uninit.rs` take a `KernelDType`, an explicit
  `ExecContext`, an `ErasedRawStridedUninitMut` destination and
  `ErasedRawStridedPtr` sources. Every check (dtype, shape, injective
  destination, destination/source overlap, integer zero divisor) completes
  before the first write, so an `Err` leaves the destination untouched. A
  successful call initializes every destination element.
- `plan_lazy_outer_product` in `strided-basic/src/outer_product.rs` is the
  dtype-free layout planner that lets an outer product be written in the
  operands' physical stride order.
- `ReduceOp::Max` and `ReduceOp::Min` extend `ErasedReducePlan` for
  f32/f64/i32/i64 with NaN propagation; complex and bool are rejected at
  compile time.

Each runtime op is matched once, outside the element loop, so every arm runs
a monomorphic closure through the typed kernel. The first version matched per
element and was 2.9x to 5.5x slower than the typed kernel on contiguous
inputs; see the perf section.

## Semantics notes

- Shape mismatch is reported before an integer zero divisor.
- Integer multiply and broadcast multiply wrap, matching the release behavior
  tenferro already had; debug builds no longer panic.
- Complex `Sign` divides the components by the modulus, which fixes
  underflow for tiny magnitudes.
- `raw_any` above the fused rank limit now uses an incremental odometer
  instead of rebuilding coordinates per element.

## Acceptance

- strided: `cargo fmt --all -- --check` and
  `RAYON_NUM_THREADS=1 cargo test --workspace` (1019 passed, 0 failed,
  9 ignored).
- tenferro branch `agent/adopt-strided-kernel-migration` with a `[patch]` path
  override to this tree: workspace tests excluding `tenferro-gpu`, 5215
  passed, with the only failures being two trybuild UI tests that fail
  identically on tenferro main (rustc 1.96 diagnostic rendering).

## Performance

tenferro `elementwise_fusion` bench, `prepared_graph` filter (it also matches
`unprepared_graph`), median times. Before is tenferro main with its pinned
strided; control is tenferro main with strided main d775510; after is the
migration branch with the op dispatch hoisted.

| Case | Before | Control | After |
|---|---|---|---|
| add_mul prepared 4096 | 5.996 us | 6.154 us | 6.158 us |
| add_mul prepared 65536 | 265.5 us | 268.2 us | 269.3 us |
| add_mul prepared 1M | 4.302 ms | 4.195 ms | 4.242 ms |
| add_mul unprepared 4096 | 15.13 us | 15.42 us | 15.41 us |
| add_mul unprepared 65536 | 288.8 us | 286.3 us | 294.0 us |
| add_mul unprepared 1M | 4.463 ms | 4.414 ms | 4.516 ms |
| broadcast_mul prepared 256x256 | 40.99 us | 41.23 us | 42.63 us |
| broadcast_mul prepared 1024x1024 | 517.2 us | 516.5 us | 571.6 us |
| broadcast_mul unprepared 256x256 | 51.95 us | 52.65 us | 52.48 us |
| broadcast_mul unprepared 1024x1024 | 593.0 us | 531.2 us | 534.4 us |
| broadcast_mul_add prepared 256x256 | 271.1 us | 279.8 us | 275.2 us |
| broadcast_mul_add prepared 1024x1024 | 4.117 ms | 4.528 ms | 4.260 ms |
| broadcast_mul_add unprepared 256x256 | 292.5 us | 297.0 us | 298.0 us |
| broadcast_mul_add unprepared 1024x1024 | 4.309 ms | 4.376 ms | 4.521 ms |

Two reruns of broadcast_mul prepared 1024x1024 on the after build gave +10.3%
and then 518.9 us (p = 0.88 against before), so the 571.6 us row is run to run
noise. Before the dispatch fix the after build was 53% slower on add_mul
prepared 4096 and 87% slower on broadcast_mul prepared 1024x1024.

strided `erased_uninit_elementwise` bench (new, paired typed vs erased, f64,
serial context), geometric mean of `erased / typed`:

| Layout | Per element dispatch | Hoisted dispatch |
|---|---|---|
| contiguous vector and matrix, 4096 | 3.8 to 5.5 | 0.81 to 1.10 |
| contiguous vector and matrix, 65536 | 2.6 to 4.9 | 0.99 to 1.01 |
| contiguous vector and matrix, 1M | 2.8 to 3.7 | 1.00 |
| transposed lhs, all sizes | 1.15 to 1.29 | 1.00 to 1.04 |

No existing bench covers reduce max/min, and no perf claim is made for it.

## Known follow-ups

- The initialized `erased_map_into` and `erased_zip_into` in
  `strided-kernel/src/erased.rs` still match the op per element and likely
  have the same 3x to 5x gap. They are untouched here.
- From #264: `total_len` in strided-basic multiplies dims unchecked, and the
  `CopyPlan` odometer in `raw_ops.rs` steps one stride past an axis before
  rewinding. Both panic in debug only on layouts near `isize::MAX`.
- `cargo clippy --workspace` fails on the pre-existing `uninit_vec` deny in
  strided-view.
