# Runtime SIMD dispatch

tensor4all/strided-rs#340. tenferro ships prebuilt binaries for Julia and Python, built for the
x86-64 baseline. Auto-vectorization only uses the target features enabled at compile time, so a
leaf loop that relies on it alone runs with SSE2 on every user's machine: 2 `f64` lanes, where
AVX2 gives 4 and AVX-512 gives 8. The elementwise leaf loops in this workspace therefore pick
their level at run time: strided-basic's map/zip/mul, ops, update and reduce leaves and its line
kernels (norm, scan), the strided-fused chunk replay and static specializations, and strided-nn's
softmax. Not yet covered: the `raw_ops` copy-scale loops, `outer_product`, `arg_reduce`, and the
gather/indexing plans (which are index-bound).

## Distribution builds

Distribution builds (and anything built for another machine) never use `-C target-cpu=native`
or another `target-cpu` above the baseline. The runtime dispatch reaches AVX2/FMA and AVX-512 on
hosts that have them. A native build is only for local experiments, such as the baseline-vs-native
comparison below.

## Mechanism

A leaf loop runs inside `pulp::Arch::new().dispatch(...)`. pulp compiles the dispatched body once
per level (`V4` = AVX-512, `V3` = AVX2/FMA, scalar; NEON on aarch64), each inside a function with
those target features enabled, and picks one at run time. `Arch::new()` caches the detection.
The workspace enables pulp's `x86-v4` feature. AVX-512 target features are stable since Rust 1.89.
The loop bodies stay unchanged, and LLVM auto-vectorizes them at the dispatched level.

- Dispatch once per leaf call, outside the element loop: `strided_basic::simd::dispatch_if_large`
  per contiguous run in `map_view`, `ops_view`, `update_view` and the reduce kernels, one dispatch
  per worker range in `erased/line.rs`, one per block in the fused chunk replay, and one per
  group in `strided-nn`.
- The dispatched body must be inlined into pulp's target-feature function, or it keeps the build
  target's instructions. pulp calls a closure through a plain call. A small closure (one loop) is
  inlined, but a large one is not (see the note on `map_contiguous`, tensor4all/strided-rs#301).
  A large body goes into a struct that implements `pulp::WithSimd` with an `#[inline(always)]`
  `with_simd`, and every helper with an element loop that it calls is `#[inline(always)]` (plain
  copies can stay `memcpy`). A loop written with zipped iterators instead of indexed slices also
  keeps a closure small enough to inline, because there are no bounds checks. The fused chunk replay
  (`ReplayBlock`) and the `strided-nn` group kernels do this.
- `strided-nn`'s `f32` max/sum reductions do not auto-vectorize without fast-math, so they are
  written over pulp's generic vectors (`splat`/`max`/`add`/`div`/`reduce_*`). One kernel serves
  every level. Its sum order depends on the lane count, as it did with the earlier AVX2 kernel.

## Regression check

`python3 scripts/check-runtime-simd-dispatch.py` builds `strided-kernel/examples/runtime_simd_probe.rs`
in release mode with `RUSTFLAGS` cleared, runs it, and disassembles it. For each dispatched body
(map `sqrt`, zip `add`, fused chunk `sqrt`, the four-input fused `sqrt`/`rsqrt` specialization,
softmax `f32` max and divide, softmax `f64` divide), the
`V4` version must use the named instruction on `zmm` registers and the `V3` version on `ymm`. CI
runs it on the ubuntu test job. A body that pulp does not inline fails it. A new dispatched path
should get a probe call and a `CHECKS` entry.

## Baseline vs native comparison (evo)

This is not measured on shared machines. On an AVX2 host and an AVX-512 host, follow the campaign
protocol: boost and frequency scaling off, one run at a time, nothing else on the measured cores.
Build the same rows twice in separate target directories and compare the rows within the noise band:

```bash
for mode in baseline native; do
  if [ "$mode" = native ]; then flags="-C target-cpu=native"; else flags=""; fi
  RUSTFLAGS="$flags" CARGO_TARGET_DIR=target/simd-$mode \
    cargo bench -p strided-kernel --bench erased_uninit_elementwise | tee simd-$mode-kernel.csv
  RUSTFLAGS="$flags" CARGO_TARGET_DIR=target/simd-$mode \
    cargo bench -p strided-fused --bench fused_chunked_replay | tee simd-$mode-fused.txt
done
```

Neither command needs a thread pin, and neither reads an ambient thread variable. The kernel
benchmark runs every measured call on `Exec::serial()` and prints `context=serial` in its `CONFIG`
line, so the harness fixes the width. The fused benchmark builds its own `Pool` at the fixed widths
it reports (one and four workers with `parallel`, one otherwise) and asserts the effective budget on
each row, so it ignores the shell as well. The workspace has no Rayon dependency:
`RAYON_NUM_THREADS` would select nothing even if it were set.

The kernel rows cover contiguous map/zip, select and clamp. The fused rows cover the chunk replay,
including `ten_sqrt` (the scalar-method `sqrt` that tenferro's `sqrt`/`rsqrt` reach through
`strided_kernel::map_into`).
