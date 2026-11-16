# `strided-491-mace-ab` — A/B harness for strided-rs#491

Compares `strided-segpoly` with `mizukami-group/llmlip.rs` on the MACE-MP-0b3
layer-1 tensor-product block (`conv_tp`: CG forward + VJP), the hotspot reported
in [tensor4all/tenferro-rs#2412](https://github.com/tensor4all/tenferro-rs/issues/2412).
The report with numbers is
[`docs/worklogs/2026-11-17-issue-491-segpoly-vs-llmlip.md`](../../docs/worklogs/2026-11-17-issue-491-segpoly-vs-llmlip.md).

Nothing here is part of the library: the crate is its own workspace
(`[workspace]` in `Cargo.toml`), so `cargo test` at the repo root does not build
it and it cannot affect a released artifact.

## Files

| path | what it is |
| --- | --- |
| `src/topo.rs` | diamond-Si supercells (128 / 1024 atoms) and llmlip's neighbour list |
| `src/cg.rs` | the layer-1 descriptor, the four segpoly plans, both sides' call sites |
| `src/lib.rs` | shared buffers, sequence-valued inputs, comparison helpers |
| `src/measure.rs` | counting allocator, `VmHWM`, warmup + median-of-15 |
| `src/main.rs` | one cell per process: `--cell cg-fwd\|cg-vjp --dtype f32\|f64 --threads N --atoms 128\|1024` |
| `tests/fidelity.rs` | the gates: published counters, both sides agree, 1T/4T bitwise, zero allocation |
| `campaign.sh` | all 16 cells, observer pinned off the measured L3 domain |
| `raw/` | the recorded cells, host/load/command logs, and the llmlip patch |

## Prerequisite: the llmlip checkout

llmlip's layer-1 kernel is private, so `raw/llmlip-bench-api.diff` adds a
`bench-api` feature that exposes it (descriptor + `TensorProductPlan` + the
forward/VJP entry points). It changes no existing code path and was never pushed
upstream.

```sh
git clone https://github.com/mizukami-group/llmlip.rs
cd llmlip.rs && git checkout 6bfb041          # the baseline pinned by #2412
patch -p1 < <this dir>/raw/llmlip-bench-api.diff
cargo build --release -p llmlip --features parallel,bench-api
```

`Cargo.toml` points at `../../../llmlip.rs`; edit that path if your checkout
lives elsewhere. `strided-basic` / `strided-segpoly` are path deps on the parent
repo, so this harness always measures the tree it sits in.

## Running

```sh
cargo test --release    # the gates; fails if either side changes meaning
./campaign.sh           # 16 cells -> raw/*.jsonl
```

Each cell prints one JSON record. `--raw-dir` also appends it to
`raw/<cell>-<dtype>-<N>T-<atoms>.jsonl`, so a re-run keeps history instead of
overwriting it.

## Measurement rules this harness enforces

* **One L3 domain per cell.** Measured CPUs come from `taskset` (`0` for 1T,
  `0,1,2,3` for 4T, all inside `index3/cache0`); the campaign shell and the
  observer are pinned to the other domain (`8-15,24-31`). Derive the sets from
  `/sys/devices/system/cpu/*/cache/index3/shared_cpu_list` on a new host.
* **Thread count, not affinity, on both sides.** llmlip dispatches through
  rayon's global pool, so `main.rs` pins that pool to `--threads`; otherwise
  "1T" would be 32 threads on one core and llmlip's 4T scaling would look
  superlinear.
* **Identical work.** Both sides read the same buffers; the harness never
  materializes the `[E, 5120]` conv_tp intermediate for strided (that is the
  point of the comparison) and reports its size per cell.
* **Fresh process per cell** so `VmHWM` is a peak, not a residue.
