# 2026-11-09: #364 — a `CopyPlan` prices its own footprint

Scope: `CopyPlan`'s serial estimate, the width and the fork decisions it drives. No
kernel, no traversal order, no result changes, and no row of the class table moves.

Machine: **AMD Ryzen AI MAX+ 395** (Zen 5, 16 cores / 32 threads, two L3 domains
of 32 MiB, 1 MiB L2 per core, AVX-512, governor `performance` on all 32 CPUs,
boost on, rustc 1.99.1, `--release`). Isolation: the measured process is
`taskset -c 0-7,16-23` (one L3 domain, `/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list`),
the observer (`pi`, this session) `taskset -apc 8-15,24-31`. One measured process
at a time; the load average and a co-tenant, when there was one, are recorded with
each run in the appendix.

## 1. What the plan priced, and what the same plan costs

Measured here with the harness in the appendix: `CopyPlan::execute`, `n x n`
`f64`, `Exec::serial()`, nanoseconds per element, median of 41 warm repetitions
(4 discarded warmups). Ranges are the spread over the runs listed in the appendix
(six runs at 15:44-17:35, load average 0.13-7.0), not a confidence interval:

| n | transpose, `origin/main` | transpose, this change | contiguous (same layout), both |
|---|---|---|---|
| 64 | 0.439-0.479 | 0.454-0.466 | 0.139-0.146 |
| 128 | 0.526-0.664 | 0.539-0.663 | 0.123-0.125 |
| 256 | 1.141-1.332 | 1.158-1.339 | 0.156-0.171 |
| 512 | 2.411-2.757 | 2.321-2.429 | 0.175-0.177 |
| 1024 | 2.333-2.450 | 2.344-2.446 | 0.176-0.179 |

The row `KernelClass::Copy` applies to every one of them is 0.171 (unchanged;
`strided-perm/` is byte-identical to `origin/main` in this change). The serial
times are the same in both trees within that spread: the change moves the price and
the width, not the replay. **Quoted**, not measured here: the worklog of the
withdrawn head of this branch (not part of this change) records `6.97-7.60` at
512; measured here on that head (1417eb5) with this harness it is 7.34-7.67. The
difference from the values above is strided-rs#422, which tiled the transposed
path, not the harness.

An untiled layout past L2, measured here the same way (rank-1, 1024 elements,
stride 256 on both sides, 4 MiB of source and destination): 9.58 and 12.66 ns per
element in the two runs of 17:35, taken while a co-tenant process was running
inside the measured CPU set (`erased_reduce_k`, ~110% of one CPU), so the values
can only be inflated. Both are far above the 3.0 ns per element this change prices
that regime at.

## 2. The change: one compile-time plan property

- `CopyPlan::compile` records `1 + sum((dim - 1) * |stride|)` for each side, so
  the working set in bytes is `dst_span * size_of::<D>() + src_span * size_of::<S>()`
  at replay: each side at its own element size, and each operand counted once.
  The element size is the one fact `compile` cannot know; nothing is probed at run
  time.
- `cost::copy_ns_per_element_for_footprint(host, bytes)` (crate-internal) answers
  the row while the working set fits the class's L2 cache, and the class's
  beyond-L2 copy rate past it: on `x86_64-avx512` 3.0 ns per element past 1 MiB
  (the L2 of both hosts behind the row), the row at every size on every class
  without such a measurement.
- A plan that streams as one dense run keeps the row at every footprint. Fusion
  folds a forward dense run into one axis; a reversed buffer is not folded
  (`[-1, -512]` over `[512, 512]` stays rank two, because fusion sorts by signed
  destination stride), so the test is dense-by-ascending-stride with one sign per
  operand, and a sign flip between axes is not one run. Measured here: 0.176 at
  2 MiB and 0.177 at 16 MiB (`n` = 512, 1024), which is the row within 4%. That is
  #364 item 3: the row stays the class of contiguous runs.
- The rate reaches every scheduling decision the replay makes: the width of the
  replay that is not tiled (`cost::for_len_at_rate`, `copy_plan.rs`'s
  `replay_fused_raw` / `replay_spilled_raw`) and both the fork gate and every node
  of the recursive splitter of the tiled replay of #422 (`cost::is_parallel_at_rate`,
  `threading::mapreduce_threaded_at_rate`). A caller's budget is not narrowed for
  the tiled replay, so only the decision to fork changes, not the width a caller
  asked for.

## 3. Acceptance, with the arithmetic written out

The criterion (#363): a 512x512 transposing `f64` copy, serial, within a factor of
about 2 of what the reporting host measures, 3.09-3.24 ns per element (that table
is **quoted** from #364's own acceptance table; it was not measured here).

| case | measured | source of the measurement | plan estimate | ratio (estimate / measured) | verdict |
|---|---|---|---|---|---|
| 512x512 | 3.09-3.24 | quoted, #364 table (Xeon Platinum 9242) | 3.0 | 3.0 / 3.24 = 0.93, 3.0 / 3.09 = 0.97 | met (1.08x low) |
| 512x512 | 2.32-2.76 | measured here (sec. 1) | 3.0 | 3.0 / 2.76 = 1.09, 3.0 / 2.32 = 1.29 | met |
| 1024x1024 | 2.34-2.45 | measured here | 3.0 | 3.0 / 2.45 = 1.22, 3.0 / 2.34 = 1.28 | met |
| 256x256 | 1.158-1.339 | measured here | 0.171 (1 MiB working set, the row) | 1.158 / 0.171 = **6.8**, 1.339 / 0.171 = **7.8** | **not met** |
| 128x128 | 0.539-0.663 | measured here | 0.171 (256 KiB) | 0.539 / 0.171 = **3.2**, 0.663 / 0.171 = **3.9** | **not met** |
| 64x64 | 0.454-0.466 | measured here | 0.171 (64 KiB) | 0.454 / 0.171 = **2.7**, 0.466 / 0.171 = **2.7** | **not met** |
| 256x256 | 1.56-1.59 | quoted, withdrawn head's worklog (this host, before #422) | 0.171 | 1.56 / 0.171 = 9.1, 1.59 / 0.171 = 9.3 | **not met** |

So the 512x512 acceptance case of #364 is met (0.93-1.29x), and the rows that fit
the class's L2 are **not**: they keep the row and are 2.7-7.8x low on today's
measurements here, 9.1-9.3x on the pre-#422 numbers of the withdrawn worklog. Those
rows are what #364 item 1 (a per-segment term, `segments * per_segment +
elements * per_element_inside_segment`) is for; item 1 is not in this change, and
the constant is deliberately not raised to cover them (item 3), which the two
arithmetic rows below the line show: 3.09 / 0.500 = 6.18x and 1.56 / 0.500 = 3.12x
would still miss the 2x criterion while moving every contiguous caller's width.

## 4. Why the row is not raised (the withdrawn head)

The earlier head of this branch raised `copy_ns_per_element_of(X86_64Avx512)` from
0.171 to 0.500. #364 forbids it, and it does not reach the acceptance above
(3.09 / 0.500 = 6.18). Measured here, with the same harness and placement, on three
trees — `origin/main` (b2358fb), the withdrawn head (1417eb5) and this rebuild —
`strided_perm::copy_into`, the HPTT engine, `Exec::new(pool, 16)` over a 16-thread
pool with the host's default dispatch costs, nanoseconds per element (microseconds
in brackets; the element counts are 16384 for 128x128, 4096, 262144 for 512x512 and
1048576 for 1024x1024):

| case | main | withdrawn head (row 0.5) | rebuild |
|---|---|---|---|
| 128x128 transpose | 0.318-0.320 [5.21-5.24 us] | 0.319-0.323 [5.23-5.29 us] | 0.315-0.319 [5.16-5.23 us] |
| 4096 contiguous | 0.151-0.156 [0.62-0.64 us] | 0.151-0.154 [0.62-0.63 us] | 0.151 [0.62 us] |
| 512x512 transpose | 0.347-0.376 | 0.271 | 0.355-0.393 |
| 1024x1024 transpose | 0.196-0.204 | 0.287 | 0.199-0.201 |

**Quoted** from the withdrawn head's worklog (not measured here): 5.00 -> 6.16 us
(+23.2%, 6.16 / 5.00 = 1.232) at 128x128 and 0.70 -> 0.90 us (+28.6%, 0.90 / 0.70 =
1.286) at 4096 for this engine. Those two rows did not reproduce here on either
tree: with the host's default dispatch costs the engine's first section costs 9.7 us
(the `Outside` row) against its 2.8 us estimate, so it stays at width 1 on both rows
and both trees and the row cannot reach its width. At the sizes where the row does
reach the engine's width, that width moves and so does the time, in both
directions (6 -> 10 at 512x512, 11 -> 16 at 1024x1024). #364 item 3 is about not
moving it, and this rebuild does not: `strided-perm/` is untouched, so the engine
and its widths are main's by construction.

## 5. Red then green, including three mutations

Tests (all in `strided-basic/src/copy_plan/tests/tests.rs` unless named):
`a_strided_plan_past_l2_prices_itself_above_the_contiguous_row` (acceptance, the
1 MiB boundary, the contiguous width), `a_strided_plan_past_l2_forks_untiled_at_its_own_rate`
(executes a rank-1 stride-256 plan of 4 MiB and counts the workers `threadteam`'s
ledger installed), `a_strided_plan_past_l2_splits_its_tiled_replay_at_its_own_rate`
(executes a 128x32 transposing plan of 2.6 MiB against a pool whose first fork asks
3 us, and requires **more than one** recorded splitter leaf and at least one worker
install), `a_reversed_run_past_l2_keeps_the_copy_row` (131072 elements at stride -1
on both sides, 2 MiB together), `a_reversed_dense_buffer_keeps_the_copy_row` (512 x
512 over `[-1, -512]`, with a mixed-sign control), plus
`cost::tests::a_footprint_past_l2_is_priced_above_the_row_of_its_class` (the
AVX-512 branch, `HostClass::X86_64Avx512` given explicitly, so it runs everywhere)
and `the_splitter_asks_its_gate_at_the_rate_it_is_given` (drives
`mapreduce_threaded_at_rate` with an explicit rate, so it runs everywhere too).

The two executed forwarding tests need a class that *has* a beyond-L2 rate, so they
skip with a printed line on a host class that prices everything at the row
(`x86_64-avx2`, `aarch64-apple`, the hosted runners): a one-row class cannot
exercise a size-dependent price, and the forwarding *is* that price. The policy and
the splitter seam are what run on every host.

**Missing footprint branch (the pre-change state)** — the acceptance test:

```
thread '...a_strided_plan_past_l2_prices_itself_above_the_contiguous_row' panicked at strided-basic/src/copy_plan/tests/tests.rs:641:9:
the plan prices itself at 0.171 ns per element, outside a factor of two of the 3.09-3.24 measured for this case
test result: FAILED. 0 passed; 1 failed
```

**Mutation 1 — the rate `CopyPlan::replay` hands to the replay pinned to
`KernelClass::Copy.ns_per_element()`** (the reviewer's first mutation: only the
executed tests can see it; the acceptance test reads the plan's private rate and
still passes):

```
test copy_plan::tests::a_strided_plan_past_l2_prices_itself_above_the_contiguous_row ... ok
test copy_plan::tests::a_strided_plan_past_l2_splits_its_tiled_replay_at_its_own_rate ... FAILED
test copy_plan::tests::a_strided_plan_past_l2_forks_untiled_at_its_own_rate ... FAILED
thread '...a_strided_plan_past_l2_splits_its_tiled_replay_at_its_own_rate' panicked at .../tests.rs:800:5:
the splitter did not fork: 0 leaf/leaves and 0 worker install(s) for a plan priced at 3.000 ns per element
thread '...a_strided_plan_past_l2_forks_untiled_at_its_own_rate' panicked at .../tests.rs:727:5:
the plan forked nowhere: the replay priced it at the row, not at 3.000 ns per element
test result: FAILED. 29 passed; 2 failed
```

**Mutation 2 — only the splitter `mapreduce_threaded_at_rate` is handed the row,
while the outer gate of `replay_tiled` keeps the plan's rate** (the reviewer's
second mutation: one leaf, no install):

```
test copy_plan::tests::a_strided_plan_past_l2_splits_its_tiled_replay_at_its_own_rate ... FAILED
test copy_plan::tests::a_strided_plan_past_l2_forks_untiled_at_its_own_rate ... ok
thread '...a_strided_plan_past_l2_splits_its_tiled_replay_at_its_own_rate' panicked at .../tests.rs:800:5:
the splitter did not fork: 1 leaf/leaves and 0 worker install(s) for a plan priced at 3.000 ns per element
test result: FAILED. 30 passed; 1 failed
```

**Mutation 3 — `is_contiguous_run`'s `unsigned_abs() == 1` replaced by `== 1`** (the
reversed 2 MiB run is then priced at the beyond-L2 rate):

```
test copy_plan::tests::a_reversed_run_past_l2_keeps_the_copy_row ... FAILED
thread '...a_reversed_run_past_l2_keeps_the_copy_row' panicked at .../tests.rs:784:5:
assertion `left == right` failed: a reversed single run is a streamed copy, not a strided one
test result: FAILED. 3 passed; 1 failed
```

With all three restored, the module is green (`cargo test --release --lib
-p strided-basic --features parallel copy_plan::tests`: 31 passed, 0 failed) and the
workspace is green (`cargo test --workspace`: 78 test binaries, 1144 passed, 0
failed).

## 6. Residual

- One rate per class, not per layout. The calibration is the *tiled* transposing
  path (sec. 3). The replay that is not tiled is not calibrated row by row: the one
  untiled beyond-L2 layout measured here (rank-1, stride 256) costs 9.6-12.7 ns per
  element under a co-tenant, i.e. the 3.0 rate under-prices that extreme by 3-4x,
  and a run long enough to fill a cache line was not measured. The earlier "up to 3x
  high" bound had no measurement behind it and is withdrawn.
- A transposing plan inside L2 keeps the row and is 2.7-7.8x low (sec. 3). #364 item
  1 (a per-segment term) is what interpolates between the two regimes; it is not in
  this change.
- The beyond-L2 rate is calibrated on two hosts (2.32-2.76 here, 2.91-3.24 on the
  reporting Xeon, quoted) and is 3.2-4.2x low against the 9.6-12.7 above and the
  7.2-9.7 this host measured before #422 tiled the transposed path (quoted).

## Appendix: harness, commands and runs

Command, in each tree (the file is an `examples/` entry, copied in and removed
again; it is not part of the change):

```
cargo build --release --example repro424 --features parallel
taskset -c 0-7,16-23 ./target/release/examples/repro424
```

`Exec::serial()` for every `CopyPlan` row; `Exec::new(pool, 16)` over
`Pool::builder().threads(16)` (host default dispatch costs) for every `copy_into`
row; `REPS = 41`, four warmup repetitions discarded, median of the 41 samples; each
case asserts one copied element so a wrong traversal fails the run instead of being
timed. The three trees were separate worktrees: `~/work/pr424-main` at b2358fb
(`origin/main`), `~/work/pr424-head` at 1417eb5 (the withdrawn head) and
`~/work/strided-424` (this change, rebuilt after each push).

Runs (median ns/element; load average from `uptime` after the run; the measured set
is cpus 0-7,16-23, the observer 8-15,24-31):

| time | tree | load | 512 transpose | 512 contiguous | copy_into 128x128 | copy_into 4096 | untiled r1 |
|---|---|---|---|---|---|---|---|
| 15:44 | main | 0.13 | 2.411 | 0.177 | 0.318 | 0.154 | - |
| 15:52 | withdrawn head | 0.13 | 7.665 | 0.176 | 0.319 | 0.151 | - |
| 15:52 | rebuild | 0.13 | 2.421 | 0.175 | 0.319 | 0.151 | - |
| 15:54 | main | 0.88 | 2.418 | 0.176 | 0.320 | 0.154 | - |
| 15:54 | withdrawn head | 0.88 | 7.397 | 0.177 | 0.321 | 0.151 | - |
| 15:54 | rebuild | 0.88 | 2.423 | 0.176 | 0.319 | 0.151 | - |
| 16:17 | main | 2.56 | 2.757 | 0.176 | 0.320 | 0.156 | - |
| 16:17 | rebuild | 2.56 | 2.429 | 0.175 | 0.315 | 0.151 | - |
| 17:01 | rebuild | 7.01 | 2.321 | 0.177 | 0.320 | 0.151 | 9.755 |
| 17:35 | main | 4.26 | 2.426 | 0.176 | 0.320 | 0.154 | 9.580 |
| 17:35 | rebuild | 4.21 | 2.322 | 0.176 | 0.319 | 0.151 | 12.656 |

The 16:17 and 17:35 runs share the measured CPU set with another session's
benchmark (`erased_reduce_k`, ~110% of one CPU, `ps -o pcpu`), which can only
inflate a measured time; the quiet window is 15:43-15:54 (load 0.13-0.88). The
`untiled r1` column is the rank-1 stride-256 layout of sec. 1 and sec. 6: it is a
serial replay time, identical in both trees, so it measures what the 3.0 rate should
price, not a rate.

Harness source (`strided-basic/examples/repro424.rs` while measuring):

```rust
//! Evidence harness for strided-rs#424 (issue #364): median serial cost of a
//! `CopyPlan` replay and of the `strided_perm::copy_into` engine, in ns per
//! element. Same binary in every tree it is copied into; not part of the change.

use std::time::{Duration, Instant};

use strided_basic::execution::parallel_width;
use strided_basic::execution::KernelClass;
use strided_basic::{
    CopyPlan, Exec, Pool, RawStridedMut, RawStridedRef, StridedView, StridedViewMut,
};

const REPS: usize = 41;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// `n x n` `f64` copy through `CopyPlan::execute`, serial.
fn plan_case(n: usize, transpose: bool) -> f64 {
    let dims = [n, n];
    let src_strides = [1isize, n as isize];
    let dst_strides = if transpose {
        [n as isize, 1]
    } else {
        [1isize, n as isize]
    };
    let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
    let src = vec![1.5f64; n * n];
    let mut dst = vec![0.0f64; n * n];
    let mut samples = Vec::with_capacity(REPS);
    for rep in 0..REPS + 4 {
        // SAFETY: both buffers hold `n * n` elements, the views' layouts are the
        // compiled ones, and the two buffers do not overlap.
        let src_ref = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
        let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
        let start = Instant::now();
        plan.execute(&mut dst_mut, &src_ref, Exec::serial())
            .unwrap();
        let elapsed = start.elapsed();
        drop(dst_mut);
        if rep >= 4 {
            samples.push(elapsed.as_secs_f64() * 1e9 / (n * n) as f64);
        }
    }
    // One checked element, so a wrong traversal fails the run instead of timing it.
    assert_eq!(dst[0], 1.5);
    assert_eq!(dst[n * n - 1], 1.5);
    median(samples)
}

/// A plan of an arbitrary layout pair, serial, in ns per element.
fn plan_layout_case(dims: &[usize], dst_strides: &[isize], src_strides: &[isize], len: usize) -> f64 {
    let plan = CopyPlan::compile(dims, dst_strides, src_strides).unwrap();
    let total: usize = dims.iter().product();
    let src = vec![1.5f64; len];
    let mut dst = vec![0.0f64; len];
    let mut samples = Vec::with_capacity(REPS);
    for rep in 0..REPS + 4 {
        let src_ref = RawStridedRef::new(&src, dims, src_strides, 0).unwrap();
        let mut dst_mut = RawStridedMut::new(&mut dst, dims, dst_strides, 0).unwrap();
        let start = Instant::now();
        plan.execute(&mut dst_mut, &src_ref, Exec::serial()).unwrap();
        let elapsed = start.elapsed();
        drop(dst_mut);
        if rep >= 4 {
            samples.push(elapsed.as_secs_f64() * 1e9 / total as f64);
        }
    }
    assert_eq!(dst[0], 1.5);
    median(samples)
}

/// `strided_perm::copy_into` through a 16-slot pool.
fn perm_case(dims: &[usize], src_strides: &[isize], dst_strides: &[isize], pool: &Pool) -> f64 {
    let total: usize = dims.iter().product();
    let src = vec![1.5f64; total];
    let mut dst = vec![0.0f64; total];
    let mut samples = Vec::with_capacity(REPS);
    for rep in 0..REPS + 4 {
        let source = StridedView::new(&src, dims, src_strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut dst, dims, dst_strides, 0).unwrap();
        let exec = Exec::new(pool, 16).unwrap();
        let start = Instant::now();
        strided_perm::copy_into(&mut dest, &source, exec).unwrap();
        let elapsed = start.elapsed();
        drop(dest);
        if rep >= 4 {
            samples.push(elapsed.as_secs_f64() * 1e9 / total as f64);
        }
    }
    assert_eq!(dst[0], 1.5);
    assert_eq!(dst[total - 1], 1.5);
    median(samples)
}

fn main() {
    for n in [64usize, 128, 256, 512, 1024] {
        let transpose = plan_case(n, true);
        let contiguous = plan_case(n, false);
        println!(
            "CopyPlan n={n:4} transpose={transpose:7.3} ns/el contiguous={contiguous:7.3} ns/el"
        );
    }
    let pool = Pool::builder().threads(16).build().unwrap();
    let t = perm_case(&[128, 128], &[1, 128], &[128, 1], &pool);
    let c = perm_case(&[4096], &[1], &[1], &pool);
    println!("copy_into 128x128 transpose={t:7.3} ns/el 4096 contiguous={c:7.3} ns/el");
    for n in [512usize, 1024] {
        let m = n as isize;
        let r = perm_case(&[n, n], &[1, m], &[m, 1], &pool);
        println!("copy_into {n}x{n} transpose={r:7.3} ns/el");
    }

    // The width the flat row asks for, at the pool width: `Exec::new(&pool, 96)`
    // clamps to the 16 slots the builder made, so the budget is 16 here.
    let exec = Exec::new(&pool, 96).unwrap();
    assert_eq!(exec.width(), 16, "the pool in this harness has 16 slots");
    for (label, total) in [("512x512", 512usize * 512), ("4096", 4096)] {
        println!(
            "{label} flat width at budget 16 (the pool width) = {}",
            parallel_width(&exec, KernelClass::Copy, total).get()
        );
    }
    // The replay that is not tiled: a rank-1 strided plan (4 MiB of source and
    // destination, no tiles) and a rank-2 plan with long runs.
    let r1 = plan_layout_case(&[1024], &[256], &[256], 1 + 1023 * 256);
    let r2 = plan_layout_case(&[128, 32], &[1, 2048], &[2048, 1], 128 * 2048);
    println!("untiled r1-1024-s256={r1:7.3} ns/el  r2-128x32-A2048(tiled)={r2:7.3} ns/el");

    // The row the public permutation engine reads, and the width it asks of a
    // 16-slot token for the two rows of the worklog.
    let exec = Exec::new(&pool, 16).unwrap();
    let row = strided_perm::copy_ns_per_element();
    for (label, total) in [
        ("128x128", 128usize * 128),
        ("4096", 4096),
        ("512x512", 512 * 512),
        ("1024x1024", 1024 * 1024),
    ] {
        let estimate = Duration::from_secs_f64(total as f64 * row * 1e-9);
        println!(
            "{label} copy_into row={row} ns/el width at budget 16 = {}",
            exec.width_for(estimate, 1).get()
        );
    }
}
```
