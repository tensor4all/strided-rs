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
(`4x2x2 = 128` atoms / `5 888` edges, `4x4x8 = 1024` atoms / `47 104` edges,
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

Pinned `taskset`, one L3 domain for the measured cell (`4` for 1T, `4,5,6,7` for
4T, all inside `index3/cache0 = 0-7,16-23`) and the observer (campaign shell,
recorder) on the other domain (`8-15,24-31`). Governor `performance`, boost on.
Core `0` is deliberately not one of them: a first attempt pinned `0` / `0,1,2,3`
while two foreign processes (`fbench verify --threads 1`, `tlbench-before`) were
pinned to core `0`, and every figure of that attempt came out ~2x slow (f64 1T
forward 156.3 ms llmlip / 138.3 ms segpoly instead of 78 / 69) with the ratios
intact (0.88). That attempt was discarded, not published; `campaign.sh` now also
re-measures one anchor cell before and after the 16 cells and refuses the run if
the two ends disagree by more than 15 % (this run: strided 68.7 -> 71.3 ms, 3.7 %;
llmlip 77.9 -> 78.1 ms, 0.1 %).
llmlip's rayon **global** pool is pinned to the same thread count
(`harness/src/main.rs:fix_llmlip_threads`) — without that, "1T" is 32 threads on
one core, which inflated the first pass by ~1.5x and made llmlip look
superlinear at 4T.

Each cell is one process and runs the **round-robin** of
`measure::interleaved`: every side is warmed until one timed round spans
`--target-ms` (20 ms), then for each of the 15 samples the sides are timed once
each, in a fixed order, so no side sees a different slice of the machine's drift.
`raw/host.txt`, `raw/load.txt`, `raw/commands.txt` and the full `raw/campaign.txt`
(two lines per cell: the header and the record) are committed — `.txt`, because the
repo's `.gitignore` drops `*.log`.

Allocations are recorded as **exact totals** over the timed calls
(`alloc_bytes_total`, `alloc_calls_total`, `timed_calls`) next to the per-call
figures, and the per-call figure is `total / timed_calls` as an `f64`, so a few
bytes per call cannot truncate to zero. The counter is process-wide and lives in
`src/lib.rs`, so the gate tests count too.

The strided side must mint an `Exec` per call (`Exec` is neither `Copy` nor
`Clone`, and `execute_into` consumes it). That mint happens **inside** the timed
closure, so its cost is part of every strided wall time in §4-§5; `exec_new`
reports that share — **2.0-2.4 ns and 0 bytes per mint** — rather than removing
it.

## 4. Forward: `conv_tp` (`y = Σ cg · w · x · y`)

Median wall time; ratio = strided / llmlip.

| dtype | T | atoms | llmlip | strided | ratio |
| --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 78.0 ms | 68.7 ms | **0.88** |
| f64 | 1 | 1024 | 638.9 ms | 555.0 ms | **0.87** |
| f64 | 4 | 128 | 10.2 ms | 17.4 ms | 1.70 |
| f64 | 4 | 1024 | 86.0 ms | 141.5 ms | 1.65 |
| f32 | 1 | 128 | 34.9 ms | 65.6 ms | 1.88 |
| f32 | 1 | 1024 | 282.8 ms | 531.8 ms | 1.88 |
| f32 | 4 | 128 | 9.1 ms | 16.6 ms | 1.83 |
| f32 | 4 | 1024 | 72.3 ms | 134.2 ms | 1.86 |

Single-thread f64 is the one cell where the executor wins: **12 % faster than
llmlip's fast kernel, identical at 128 and 1024 atoms** (so it is a kernel
property, not a cache artifact). Three reasons it does not extend:

* **f32 lanes.** llmlip's f32 kernel is 2.2x faster than its own f64 kernel
  (34.9 vs 78.0 ms); segpoly gains 1.05x (65.6 vs 68.7 ms). The `u = 128` f32 lane
  is 512 B per row — the executor's lane lowering does not get the SIMD width out
  of it that llmlip's hand-written f32 paths do.
* **4T.** llmlip scales 7.4x on 4 cores (638.9 -> 86.0 ms), segpoly 3.9x
  (555.0 -> 141.5 ms). The executor splits *units* (nodes) with
  `BLOCK_ROWS = 256`; at 128 nodes there are fewer units than useful chunks, and
  per-node work is uneven (degree 30-70), so 4 workers idle. llmlip's rayon
  fan-out is per edge-range.
* **No intermediate edge array.** segpoly never materializes the
  `[E, 5 120]` conv_tp output — 230 MiB f64 / 115 MiB f32 at 128 atoms and
  1 840 MiB / 920 MiB at 1024, which llmlip does allocate.

## 5. VJP: `bar_x`, `bar_w`, `bar_y`

llmlip returns all three gradients in one fused pass; segpoly needs three plans
(no shared traversal), so the comparable number is the sum. Per-plan ratios are
also given because they say *why*.

| dtype | T | atoms | llmlip | `bar_x` | `bar_w` | `bar_y` | sum | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 193.5 ms | 67.2 (0.35) | 61.5 (0.32) | 426.4 (2.20) | 555.2 | 2.87 |
| f64 | 1 | 1024 | 1568.2 ms | 578.0 (0.37) | 520.2 (0.33) | 3496.6 (2.23) | 4594.9 | 2.93 |
| f64 | 4 | 128 | 32.1 ms | 26.6 (0.83) | 15.5 (0.48) | 107.8 (3.36) | 150.0 | 4.67 |
| f64 | 4 | 1024 | 285.6 ms | 227.2 (0.80) | 128.5 (0.45) | 926.7 (3.24) | 1282.3 | 4.49 |
| f32 | 1 | 128 | 45.9 ms | 62.1 (1.35) | 60.6 (1.32) | 417.6 (9.10) | 540.3 | 11.78 |
| f32 | 1 | 1024 | 430.6 ms | 517.9 (1.20) | 497.2 (1.15) | 3729.4 (8.66) | 4744.5 | 11.02 |
| f32 | 4 | 128 | 11.9 ms | 28.6 (2.40) | 15.4 (1.30) | 106.2 (8.92) | 150.2 | 12.62 |
| f32 | 4 | 1024 | 171.7 ms | 213.7 (1.24) | 124.9 (0.73) | 849.4 (4.95) | 1188.0 | 6.92 |

Two vectorised plans beat the whole fused llmlip call: `bar_x` 0.35x and
`bar_w` 0.32x in f64 1T (f32 loses its edge, as in the forward). The
`bar_edge_attrs` reduction is the opposite extreme — 2.2x in f64 1T, 5-9x in f32
— and alone accounts for **70-79 %** of the segment's time. It is the only
gradient whose lane is a reduction over `(feature, m)` rather than a feature
vector, so it lowers to the `u = 1` scalar plan with 11 008 paths over
`N x 5 120` rows: the CG work per byte of output is 128x that of the vectorised
plans. The feature-expanded form of the same reduction would need an
`[E, 16, 128]` tensor to hold its partial sums (92 MiB f64 / 46 MiB f32 at 128
atoms, 736 / 368 MiB at 1024); the `u = 1` plan re-derives the terms per path and
allocates neither.

## 6. Allocation and memory

| | strided | llmlip |
| --- | --- | --- |
| bytes per forward call | **0** | 2.6-40.6 MiB |
| bytes per VJP call | **0** | 29.4-474.2 MiB |
| allocation calls per call | **0** | 15-1 230 |
| `Exec` mint per call (in the figure above) | 2.0-2.4 ns, 0 bytes | n/a (rayon dispatch) |

`execute_into` holds the zero-allocation contract of PR #490 for all four plans:
over every timed call of every cell the strided side recorded
`alloc_bytes_total = 0` and `alloc_calls_total = 0`, at 1T and at 4T, in f32 and
f64. Two 4T VJP cells (128 atoms) are the exception: one 1 520-byte allocation in
one of the 30 timed `bar_w` calls (51 B/call amortized over that cell). What that
is, precisely: it is **lazy per-worker initialization** on the pool's first use of
that plan — each worker thread allocates its own scratch on its first task and
never again, so the 30 timed calls of that cell contain exactly one such event and
every later execution is free. `tests/alloc.rs` warms the plan and then asserts the
next pooled call allocates exactly 0 bytes. Because the counter is process-wide it
cannot attribute the bytes to a specific structure, only show that they are
per-worker first-touch and not per call. That
gate also proves it is not vacuous: it first requires the counter to see a 1 MiB
allocation. llmlip allocates per call, and its count depends on the thread count
(46 at 1T, 238-1 230 at 4T) because rayon chunks allocate.

`VmHWM` is per *process* and both sides run in one process per cell, so it is
reported per cell rather than per side: 85 MiB (f64) / 93 MiB (f32) at 128 atoms
and 646 / 718 MiB at 1024 for the forward; 192-193 / 99-100 MiB at 128 atoms and
1 510-1 513 / 764-765 MiB at 1024 for the VJP. That peak is set by the
caller-owned buffers both sides share — `tp_weights` plus the outputs — not by the
kernel. What differs is the `[E, 5 120]` intermediate, present only on the llmlip
side: 230 MiB f64 / 115 MiB f32 at 128 atoms and 1 840 / 920 MiB at 1024.
segpoly's peak is `tp_weights + grad_w` because the harness has to hold the
outputs it asked for.

## 7. Numerics

Both sides agree to the precision of the accumulator, so the comparison is of
kernels, not of models. `abs` is the max absolute difference over **every node**
(§"gates"), and `of scale` is the same number divided by the largest magnitude on
the reference side — the only meaningful relative figure, because elementwise
relative error explodes where a sum cancels.

| cell | max abs (128 atoms) | of scale | max abs (1024 atoms) | of scale |
| --- | --- | --- | --- | --- |
| f64 fwd | 1.8e-15 | 6.4e-16 | 4.55e-12 | 4.6e-16 |
| f64 VJP `bar_x` | 8.5e-14 (1T) / 1.1e-13 (4T) | 3.2e-16 / 4.3e-16 | 5.8e-11 / 1.2e-10 | 2.8e-16 / 5.5e-16 |
| f64 VJP `bar_w` | 0 (1T) / 1.3e-15 (4T) | 0 / 5.0e-16 | 0 / 2.3e-12 | 0 / 6.1e-16 |
| f64 VJP `bar_y` | 1.4e-13 / 1.9e-13 | 1.7e-15 / 2.3e-15 | 2.3e-10 / 3.4e-10 | 2.5e-15 / 3.6e-15 |
| f32 fwd | 7.2e-7 | 2.6e-7 | 2.9e-3 | 3.0e-7 |
| f32 VJP `bar_x` | 4.6e-5 | 1.7e-7 | 6.25e-2 | 3.0e-7 |
| f32 VJP `bar_w` | 7.2e-7 | 2.7e-7 | 9.8e-4 | 2.6e-7 |
| f32 VJP `bar_y` | 2.9e-4 | 3.4e-6 | 1.2e-1 | 1.3e-6 |

f64 agrees to 2.8e-16-3.6e-15 of scale, i.e. at the accumulator's precision; the
f32 figures are a handful of f32 ulps of an 11 008-term reduction, which is why
`bar_y` is the loosest cell. `bar_w` came out bit-identical in the 1T cells.

**Gates** (`tests/fidelity.rs`, 10 tests): the published counters above; both
sides agree on **every node** in f32 and f64 (the forward comparison strides the
node axis explicitly — it used to read only node 0, which a perturbation test now
catches, corrupting one element at a time in the narrow middle, the narrow tail
and the wide tail); 1T and 4T produce **bit-identical** output for the forward and
all three VJPs, in f32 and f64. `tests/alloc.rs` (1 test, its own binary because
the counters are process-wide) asserts `execute_into` allocates exactly 0 bytes
and 0 calls serially and on every pooled execution after the per-worker first touch, and starts by requiring
the counting allocator to see a 1 MiB allocation — otherwise the assertion would
be vacuous.

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
4. **f32 gains nothing from the lane width.** On the strided side f32 is 1.05x its
   f64 wall time, while llmlip's f32 is 2.2x its f64. Worth a separate look; it is
   not a correctness issue (both sides agree to f32 precision).

The second half of #491 — the symmetric contraction group — is **not** covered
here. Its llmlip cost is dominated by per-node pre-weighting outside the
contraction, so a shape-level A/B would not isolate the kernel; it needs a
fixture with real weights, which this host cannot produce (see §9). Reporting the
CG block alone is better than a number that measures the wrong thing.

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
* The timings are one host, one L3 domain, one governor setting. The direction of
  each gap is structural (lane width, unit granularity, reduction shape); the
  size of each gap is not a hardware-independent constant.

## 10. Reproduction

```sh
# 1. the local-only llmlip accessor (never pushed)
cd llmlip.rs && git checkout 6bfb041 && patch -p1 < ../raw/llmlip-bench-api.diff
cargo build --release -p llmlip --features parallel,bench-api

# 2. the harness: edit the two path deps to your checkouts, then
cd benchmarks/strided-491-mace-ab && cargo test --release && cargo build --release

# 3. one cell (prints the JSON record, appends it with --raw-dir)
taskset -c 4,5,6,7 target/release/mace-ab --cell cg-vjp --dtype f64 \
  --threads 4 --atoms 1024 --samples 15 --target-ms 20 --raw-dir raw

# 4. all 16 cells, observer pinned off the measured L3 domain
taskset -c 8-15,24-31 ./campaign.sh
grep -c '^{' raw/campaign.txt     # 18 = 16 cells + 2 anchor records
grep -c FAILED raw/campaign.txt   # 0
./check_report.py                 # every table row, size field and range vs raw/
```

`campaign.sh` exits nonzero when a cell fails or when the two anchors disagree by
more than 15 %, so the transcript cannot look finished while a record is missing.

Every number in §4-§7 is a field of one of the 16 records in `raw/*.jsonl` (the
same 16 lines are in `raw/campaign.txt`). `check_report.py` is the mechanical form
of that statement: it parses the tables out of this file, compares each cell
to the record it claims to come from, re-derives `grad_x_mib` as
`N*(128 + 3*f)*sizeof`, checks the prose ranges against the records and the
transcript, and exits nonzero on any disagreement — a mutated table value, a
deleted row, a truncated transcript or an off-by-one range all fail it. Its output
on this tree: 8/8 forward rows, 8/8 VJP rows, 8/8 `grad_x_mib` cells, 32/32 numeric
columns, 18/18 transcript lines, `OK`.

Reviewer record for the implementation plan (required by the session's plan
gate): `ollama-cloud/glm-5.3-flash:high`, two rounds, both `VERDICT: CHANGES`
(round 1: `in1` is 512 not 256, a single plan cannot hold the lane-1 `grad_y`,
`bench_api` submodule placement; round 2: the sym accessor belongs to
`src/mace/bench_api.rs`, `bar_x`/`bar_y` VJP plans may not declare an unused
operand, `grad_m` must be `Gather(receivers)`, `x` is two affine views). All
findings are folded into the plan and into the plans above.

Post-implementation review of this PR found five defects, all fixed here:
integer truncation in the allocation accounting (now exact totals plus `f64`
per-call ratios, `exec_new` measured separately); f64 VJP numerics and the
1024-atom edge count quoted from memory instead of from the records (now
`47 104` edges and the per-cell values of §7); the forward comparison only read
node 0 (now every node, with a perturbation gate), and thread invariance covered
only the f64 forward (now forward + all three VJPs, f32 and f64); "interleaved"
was claimed but not done (now a real round-robin); and the campaign log the body
grepped was not committed — and once renamed to `raw/campaign.txt`, because
`.gitignore` drops `*.log`, it was; the allocation gates moved to their own test
binary after the installed counter showed they were being charged for their
neighbours' allocations; and the mutation test now corrupts a narrow middle
element, the narrow tail and the wide tail, one at a time, instead of an index that
was already `wide[0]`.

The last review round added: `grad_x_mib` derived from the buffer the `Vec` is
actually allocated with (`N*(128 + 3*f)` = `4*N*f` here, it was half that); the
1 520 B event described as lazy **per-worker** initialization instead of a "first
pool dispatch"; the f64 agreement lower bound corrected to `2.8e-16`; the
contaminated attempt (§3, two foreign processes pinned to cpu 0) reported instead
of retried; `campaign.sh` exiting nonzero on a failed cell as well as on anchor
drift; and this checker committed, because a verification claim that is not a
command in the tree is not a claim anyone can repeat.
