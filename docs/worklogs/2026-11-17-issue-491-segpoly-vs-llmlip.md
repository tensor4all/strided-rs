# 2026-11-17: #491 — `strided-segpoly` vs llmlip on the MACE-MP-0b3 layer-1 CG

Issue: [tensor4all/strided-rs#491](https://github.com/tensor4all/strided-rs/issues/491).
Upstream motivation: [tensor4all/tenferro-rs#2412](https://github.com/tensor4all/tenferro-rs/issues/2412)
profiles MACE-MP-0b3 and finds the layer-1 tensor-product block
(`conv_tp`: CG contraction, forward + VJP) plus the symmetric contraction group
dominating the step.

This change is **a measurement, not a library change**: no file under any `src/`
is touched, `strided-segpoly` keeps the contract PR #490 gave it. What is added
is the A/B harness (`benchmarks/strided-491-mace-ab/`), its raw data
(`benchmarks/strided-491-mace-ab/raw/`), and this report.

## 1. What is compared

| side | code | entry point |
| --- | --- | --- |
| strided | `strided-segpoly` @ `443e9b3` (PR #490 merged) | four `SegPolyPlan`s + `execute_into` |
| llmlip | `mizukami-group/llmlip.rs` @ `6bfb041` (the baseline pinned by #2412) | `mace_lite` layer-1 `conv_tp` forward / VJP, its own `layer1-fast` kernel |

llmlip's layer-1 kernel is private, so the harness carries a **local, never
pushed** accessor behind a new `bench-api` feature
(`raw/llmlip-bench-api.diff`, 430 lines, `crates/llmlip/src/mace_lite/bench_api.rs`
and `crates/llmlip/src/mace/bench_api.rs`). It reshapes nothing and reimplements
nothing: it returns the descriptor (baked CG tables, block dims, `weight_numel`),
a `TensorProductPlan` built from those tables, and calls the existing private
fns. The accessor also reports which kernel answered, and every cell below
recorded `layer1-fast` — i.e. the comparison is against llmlip's fast path, not
its generic fallback.

The graph is a diamond-Si supercell with llmlip's own neighbour list
(`4x2x2 = 128` atoms / `5 888` edges, `4x4x8 = 1024` atoms / `47 488` edges,
`r_max = 6.0 Å`). Both sides consume **the same buffers** (`linear_up [N,512]`,
`edge_attrs [E,16]`, `tp_weights [E,1280]`, `bar_scatter [N,5120]`), built once
per cell.

## 2. The layer-1 block, pinned by counters

The descriptor is not reconstructed from the model file (that needs
`python + torch + mace`, absent here — `build.rs:506-513`); it is read from the
baked tables in llmlip source (`MP0B_LAYER1_TP_CG{0..9}_F32`, `mace_lite.rs:34528`)
and pinned by `tests/fidelity.rs`:

* 10 blocks, CG paths per block `1,3,3,3,11,5,11,21,7,21` = **86 per feature**,
  matching the issue's `86 x 128` element count;
* `in1 = 512` (`128x0e + 128x1o`), `in2 = 16`, `out = 5 120` (`10 x 128`),
  `weight_numel = 1 280`;
* the four blocks with a stored w3j tensor have `11, 11, 21, 21` nonzeros,
  matching `mace_mp0b_inspection.json`;
* the VJP `bar_edge_attrs` reduction has **11 008 paths** (`86 x 128`), i.e. its
  lane is not a feature vector but the whole `(feature, m)` plane.

Four segpoly plans cover the block: CG forward (`u = 128`), `bar_x` (`u = 128`,
two views `x0e [N,1,128]` and `x1o [N,3,128]`), `bar_w` (`u = 128`), `bar_y`
(`u = 1`, 11 008 paths — the lane reduction cannot be a lane extent).

## 3. Method

Pinned `taskset`, one L3 domain for the measured cell (`0` for 1T, `0,1,2,3` for
4T, all inside `index3/cache0 = 0-7,16-23`) and the observer (campaign shell,
recorder) on the other domain (`8-15,24-31`). Governor `performance`, boost on.
llmlip's rayon **global** pool is pinned to the same thread count
(`harness/src/main.rs:fix_llmlip_threads`) — without that, "1T" is 32 threads on
one core, which inflated the first pass by ~1.5x and made llmlip look
superlinear at 4T. Warmup + 15 samples, median; the two sides are interleaved per
cell; allocator counters and `VmHWM` are read in the same process.
`raw/host.txt`, `raw/load.txt`, `raw/commands.txt` record the host and the exact
invocations.

## 4. Forward: `conv_tp` (`y = Σ cg · w · x · y`)

Median wall time; ratio = strided / llmlip.

| dtype | T | atoms | llmlip | strided | ratio |
| --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 78.2 ms | 68.8 ms | **0.88** |
| f64 | 1 | 1024 | 636.0 ms | 558.5 ms | **0.88** |
| f64 | 4 | 128 | 10.2 ms | 17.2 ms | 1.69 |
| f64 | 4 | 1024 | 85.7 ms | 142.8 ms | 1.67 |
| f32 | 1 | 128 | 34.9 ms | 64.8 ms | 1.86 |
| f32 | 1 | 1024 | 282.4 ms | 529.4 ms | 1.87 |
| f32 | 4 | 128 | 8.9 ms | 16.2 ms | 1.83 |
| f32 | 4 | 1024 | 71.3 ms | 133.6 ms | 1.87 |

Single-thread f64 is the one cell where the executor wins: **12 % faster than
llmlip's fast kernel, identical at 128 and 1024 atoms** (so it is a kernel
property, not a cache artifact). Three reasons it does not extend:

* **f32 lanes.** llmlip's f32 kernel is 2.2x faster than its own f64 kernel
  (34.9 vs 78.2 ms); segpoly gains nothing (64.8 vs 68.8 ms). The `u = 128` f32
  lane is 512 B per row — the executor's lane lowering does not get the SIMD
  width out of it that llmlip's hand-written f32 paths do.
* **4T.** llmlip scales 7.7x on 4 cores, segpoly 4.0x. The executor splits
  *units* (nodes) with `BLOCK_ROWS = 256`; at 128 nodes there are fewer units than
  useful chunks, and per-node work is uneven (degree 30-70), so 4 workers idle.
  llmlip's rayon fan-out is per edge-range.
* **No intermediate edge array.** segpoly never materializes the
  `[E, 5 120]` conv_tp output — that is 230 MiB at 128 atoms / 1 840 MiB at 1024
  in f64 (920 MiB / 20 MiB f32) that llmlip does allocate.

## 5. VJP: `bar_x`, `bar_w`, `bar_y`

llmlip returns all three gradients in one fused pass; segpoly needs three plans
(no shared traversal), so the comparable number is the sum. Per-plan ratios are
also given because they say *why*.

| dtype | T | atoms | llmlip | `bar_x` | `bar_w` | `bar_y` | sum | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 196.2 ms | 67.4 (0.34) | 61.2 (0.31) | 423.9 (2.16) | 552.5 | 2.82 |
| f64 | 1 | 1024 | 1556.3 ms | 561.6 (0.36) | 512.0 (0.33) | 3441.3 (2.21) | 4514.8 | 2.90 |
| f64 | 4 | 128 | 33.6 ms | 26.2 (0.78) | 15.4 (0.46) | 106.6 (3.17) | 148.2 | 4.41 |
| f64 | 4 | 1024 | 257.7 ms | 223.5 (0.87) | 126.8 (0.49) | 882.1 (3.42) | 1232.4 | 4.78 |
| f32 | 1 | 128 | 46.1 ms | 61.0 (1.32) | 60.0 (1.30) | 416.6 (9.04) | 537.6 | 11.67 |
| f32 | 1 | 1024 | 438.6 ms | 514.5 (1.17) | 488.3 (1.11) | 3359.6 (7.66) | 4362.5 | 9.95 |
| f32 | 4 | 128 | 12.0 ms | 24.4 (2.04) | 15.1 (1.26) | 105.0 (8.78) | 144.5 | 12.07 |
| f32 | 4 | 1024 | 163.5 ms | 204.9 (1.25) | 122.9 (0.75) | 990.5 (6.06) | 1318.3 | 8.06 |

Two vectorised plans beat the whole fused llmlip call: `bar_x` 0.34x and
`bar_w` 0.31x in f64 1T (f32 loses its edge, as in the forward). The
`bar_edge_attrs` reduction is the opposite extreme — 2.2x in f64 1T, 8-9x in f32
— and alone accounts for **77-78 %** of the segment's time. It is the only
gradient whose lane is a reduction over `(feature, m)` rather than a feature
vector, so it lowers to the `u = 1` scalar plan with 11 008 paths over
`N x 5 120` rows: the CG work per byte of output is 128x that of the vectorised
plans.

## 6. Allocation and memory

| | strided | llmlip |
| --- | --- | --- |
| bytes / allocation calls per forward call | **0 / 0** | 2.6-40.2 MiB / 15-1 069 |
| bytes / calls per VJP call | **0 / 0** | 29.4-474.2 MiB / 17-1 230 |
| `VmHWM`, 1024 atoms f64 fwd | 646 MiB | 647 MiB |
| `VmHWM`, 1024 atoms f64 vjp | 1 514 MiB | 1 510 MiB |

`execute_into` holds the zero-allocation contract of PR #490 for all four plans,
serially and on a pool (`tests/fidelity.rs::execute_allocates_nothing`). llmlip
allocates per call, and its count depends on the thread count (238 -> 1 230 at
4T) because rayon chunks allocate.

`VmHWM` is per *process*, so equal peaks above do not mean equal traffic: the
peak is set by the caller-owned buffers both sides share. The difference is the
`[E, 5 120]` intermediate, present only on the llmlip side (230/920 MiB at 128
atoms, 1 840 MiB f64 / 920 MiB f32 at 1024). segpoly's peak is
`tp_weights + grad_w` because the harness has to hold the outputs it asked for.

## 7. Numerics

Both sides agree to the precision of the accumulator, so the comparison is of
kernels, not of models:

| cell | max abs difference | note |
| --- | --- | --- |
| f64 fwd, 128 / 1024 atoms | 4.4e-16 / 1.8e-12 | elementwise relative up to 2e-5 where terms cancel |
| f64 VJP `bar_x` / `bar_w` / `bar_y` | 1.4e-13 / 0 / 1.4e-13 | `bar_w` reached 0 in 1T cells |
| f32 fwd, 128 / 1024 atoms | 2.4e-7 / 7.3e-4 | f32 rounding of a 1 024-atom sum |
| f32 VJP | elementwise relative 5.7e-7..1.1e-5 | |

Elementwise *relative* error is not the right statistic here (it explodes where a
sum cancels); the gate in `tests/fidelity.rs` is `max_abs / max|reference|`, and
it also checks that 1T and 4T produce bit-identical output.

## 8. What this says about the executor

1. **The `u = 128` vectorised lowering is competitive at 1T f64 and better than
   competitive for two of three VJP gradients** — no library change needed to
   claim the layer-1 forward on that configuration.
2. **The lane reduction is the gap.** `bar_edge_attrs` needs `u = 1` and pays
   128x the CG work per output byte. A lowering that reduces over the lane (or a
   transposed plan whose lane is `m` rather than the feature) is the single
   change with the largest payoff in this block, worth ~2.2x on the VJP segment
   in f64 and ~8x in f32.
3. **Thread scaling, not single-thread throughput, is the 4T gap**: the executor
   splits nodes, and MACE layer-1 nodes are few and uneven. Edge-range splitting
   (what rayon does for llmlip) would need a different `IndexMode`/unit choice.
4. **f32 gains nothing from the lane width.** On the strided side f32 ≈ f64 wall
   time, while llmlip's f32 is 2.2x its f64. Worth a separate look; it is not a
   correctness issue (both sides agree to f32 precision).

The second half of #491 — the symmetric contraction group — is **not** covered
here. Its llmlip cost is dominated by per-node pre-weighting outside the
contraction, so a shape-level A/B would not isolate the kernel; it needs a
fixture with real weights, which this host cannot produce (see §9). Reporting
the CG block alone rather than a number that measures the wrong thing.

## 9. Limitations, stated as limitations

* **Shape-level, not model-level.** No `.e3rs` MACE artifact exists on this host
  and the exporter needs `python + torch + mace`. Buffers are deterministic
  synthetic values of the right shape, so the *kernel* comparison holds and the
  end-to-end step time does not appear anywhere here.
* **llmlip is patched locally** (never pushed) to reach its private layer-1
  kernel; the patch is `raw/llmlip-bench-api.diff` and it changes no existing
  code path (7 lines of `mod`/feature wiring plus two new files).
* **Four plans vs one fused call** is inherent (segpoly shares no traversal
  between gradients). Charging llmlip one call and segpoly three is the fair
  decomposition of the same work; it is also reported per plan so a reader can
  re-weight it.
* **No GPU**: no NVIDIA device and no CUDA toolkit on this host, and this block
  has no GPU lowering to compare.
* f32 CG values come from llmlip's baked `*_F32` tables; the f64 path rebuilds
  them in f64, which is why the f32 and f64 cells are not a dtype-scaling study.

## 10. Reproduction

```sh
# 1. the local-only llmlip accessor (never pushed)
cd llmlip.rs && git checkout 6bfb041 && patch -p1 < ../raw/llmlip-bench-api.diff
cargo build --release -p llmlip --features parallel,bench-api

# 2. the harness: edit the two path deps to your checkouts, then
cd benchmarks/strided-491-mace-ab && cargo test --release && cargo build --release

# 3. one cell (prints the JSON record, appends it with --raw-dir)
taskset -c 0 harness/target/release/mace-ab --cell cg-vjp --dtype f64 \
  --threads 1 --atoms 1024 --samples 15 --raw-dir raw

# 4. all 16 cells, observer pinned off the measured L3 domain
taskset -c 8-15,24-31 ./campaign.sh
```

Reviewer record for the implementation plan (required by the session's plan
gate): `ollama-cloud/glm-5.3-flash:high`, two rounds, both `VERDICT: CHANGES`
(round 1: `in1` is 512 not 256, a single plan cannot hold the lane-1 `grad_y`,
`bench_api` submodule placement; round 2: the sym accessor belongs to
`src/mace/bench_api.rs`, `bar_x`/`bar_y` VJP plans may not declare an unused
operand, `grad_m` must be `Gather(receivers)`, `x` is two affine views). All
findings are folded into the plan and into the plans above.
