# 2026-11-09: #427 — whole-plan concatenate tiles its permuted segments

Raw evidence for tensor4all/strided-rs#432. It covers the red-first and mutation runs of the regression tests, and the timings of `strided-basic/benches/concat_parallel.rs` against the PR's base. Section 1 interprets the timings; the appendices hold the unedited output.

The SHAs below are those of the branch rebased onto `31099b8` (origin/main on 2026-11-10). Measurements and mutation runs were taken on the rebased tree `f43e2a1`, whose non-test code is that of the PR head: later commits on the branch change only this worklog.

Machine: **Apple M5 Max** (18 logical CPUs, macOS 26.7.2), rustc 1.96.0, `cargo bench` profile, `--features parallel`. This is a shared interactive machine; nothing was pinned or isolated. The load average at the start of every session is recorded in Appendix A, and it was 5.4–7.0 throughout. The benchmark host evo2 was not used.

## 1. Timings

Command per session: `CONCAT_BENCH_REPS=201 <binary>`. Each session runs the five cases. Within a case, every rep times the three paths in an order that rotates per rep, with tokens and destination views built before the timer. Each row reports the median of 201 reps after 3 warm-up reps.

The three binaries:

- `head`: built from `f43e2a1`.
- `base`: built from `31099b8`, which includes #428 and #436, with this branch's bench file and its `[[bench]]` entry copied in.
- `base-again`: the same `base` binary run a second time. This is the A/A control, which measures what the machine alone moves.

There were five rounds, and each round ran the three binaries in a rotated order.

Table 1 gives, per case and binary, the median of the five session medians, with the min–max of the session medians. The `whole/per` column is the ratio of the medians, and the closing lines give the case-level ratio of those medians between binaries:

```
case                     binary                 whole-plan 4T           per-segment 4T whole/per
4x[1024x1024] all T      base          4.167 (3.502-4.788)   1.474 (1.270-1.489)     2.827
4x[1024x1024] all T      base-again    4.391 (3.450-5.221)   1.483 (1.240-1.500)     2.961
4x[1024x1024] all T      head          1.206 (1.192-1.214)   1.269 (1.249-1.274)     0.950
4x[1024x1024] T,C,T,C    base          3.127 (2.978-3.217)   0.985 (0.963-0.993)     3.175
4x[1024x1024] T,C,T,C    base-again    2.966 (2.917-3.017)   0.981 (0.968-0.995)     3.023
4x[1024x1024] T,C,T,C    head          0.732 (0.726-0.752)   0.787 (0.778-0.818)     0.930
4x[256x256] all T        base          0.137 (0.137-0.139)   0.138 (0.137-0.139)     0.993
4x[256x256] all T        base-again    0.138 (0.136-0.146)   0.138 (0.137-0.146)     1.000
4x[256x256] all T        head          0.137 (0.137-0.138)   0.138 (0.137-0.138)     0.993
8x[512x512] all T        base          0.909 (0.883-0.956)   1.478 (1.474-1.571)     0.615
8x[512x512] all T        base-again    0.929 (0.895-0.938)   1.519 (1.481-1.610)     0.612
8x[512x512] all T        head          0.458 (0.455-0.472)   1.586 (1.577-1.627)     0.289
4x[1024x1024] all C      base          0.251 (0.248-0.265)   0.316 (0.310-0.326)     0.794
4x[1024x1024] all C      base-again    0.256 (0.251-0.262)   0.315 (0.312-0.321)     0.813
4x[1024x1024] all C      head          0.264 (0.263-0.270)   0.290 (0.285-0.293)     0.910

case-level ratio of medians (whole-plan, per-segment):
4x[1024x1024] all T      base-again/base 1.054 1.006   head/base 0.289 0.861
4x[1024x1024] T,C,T,C    base-again/base 0.949 0.996   head/base 0.234 0.799
4x[256x256] all T        base-again/base 1.007 1.000   head/base 1.000 1.000
8x[512x512] all T        base-again/base 1.022 1.028   head/base 0.504 1.073
4x[1024x1024] all C      base-again/base 1.020 0.997   head/base 1.052 0.918
```

Table 2 pairs the binaries **within each round** (the same round's sessions) and gives the min..max of the five per-round ratios, with their median in brackets:

```
per-round paired ratios (same round), min..max over the 5 rounds [median]:
case                             A/A whole-plan        A/A per-segment        head/base whole      head/base per-seg
4x[1024x1024] all T        0.721..1.453 [0.984]   0.841..1.168 [1.009]   0.252..0.344 [0.286]   0.847..1.003 [0.854]
4x[1024x1024] T,C,T,C      0.914..0.995 [0.965]   0.978..1.022 [0.996]   0.227..0.252 [0.232]   0.790..0.841 [0.799]
4x[256x256] all T          0.993..1.066 [1.007]   0.993..1.058 [1.007]   0.986..1.007 [1.000]   0.986..1.007 [1.000]
8x[512x512] all T          0.976..1.059 [1.008]   1.005..1.066 [1.027]   0.494..0.515 [0.504]   1.036..1.075 [1.067]
4x[1024x1024] all C        0.985..1.032 [1.008]   0.966..1.035 [0.991]   0.996..1.089 [1.048]   0.890..0.945 [0.908]
```

What this establishes:

- **Machine variation (A/A).** The largest **case-level ratio of medians** for the same binary run twice is 1.054 (whole-plan, `4x[1024x1024] all T`), with 0.949 the smallest. That figure is not a bound on how far one session moves.
  - Paired within rounds, the same binary moved whole-plan by 0.721..1.453 on `all T`, which is −28% to +45%, and by 0.914..1.066 on the other four cases.
  - Per-segment moved by 0.841..1.168 on `all T`, and by 0.966..1.066 on the other cases.
- **Permuted segments.** Head whole-plan is 0.29–0.95 of per-segment (Table 1). Paired within rounds, head/base whole-plan is 0.252..0.344, 0.227..0.252 and 0.494..0.515 on the three permuted cases that leave the serial gate. Every one of those per-round ratios lies below the smallest within-round A/A ratio of any case (0.721). So the gain is outside what the machine alone produced in these runs, round by round.
- **4x[256x256].** The case stays under the serial gate on every binary (head/base 0.986..1.007).
- **Contiguous only (`all C`).**
  - Head whole-plan is 1.052 of base by the case-level ratio of medians, and 0.996..1.089 (median 1.048) paired within rounds.
  - The within-round A/A range on this case is 0.985..1.032, which head/base exceeds in 3 of 5 rounds.
  - The per-segment path is the same source in both binaries (`CopyPlan` is unchanged apart from two `pub(crate)` accessors). Even so, it moves by 0.890..0.945 head/base on this case, and by 0.790..1.075 across cases, paired within rounds.
  - So differences of this size also appear between the two builds in code this change does not touch. These runs do not separate a +5% code effect from a build-layout effect.
  - Since `c8046e3`, the contiguous route is the pre-#427 loop (`replay_flat_parallel`), and the only added work per call is the stored `may_tile` flag test. **Equal contiguous throughput is not claimed.** The measured head/base ratio is 1.052 at the case level and at most 1.089 in any round.

Earlier runs on the pre-rebase tree, at that time's base `03a291b`, gave the same picture: permuted whole-plan at 0.22–0.57 of base per round, and contiguous whole-plan at 1.034 of base. They are superseded by the run above and not reproduced here.

## 2. Regression tests: red first and the review mutants

Appendix B is the unedited output of the script in Appendix C, extracted from this file and run on `0309a97`. That tree differs from the PR head only in this worklog. The script enforces every outcome. The unmutated tests must return 0 with `1 passed`. Each mutant, and the red-first commit `22bde1b` (found by its subject and run in a temporary worktree), must return 101, show `1 failed` and the expected panic message in stdout, and show `error: test failed` and no compiler error in stderr. Otherwise the script restores the mutated file and exits 1. Appendix D shows it doing so for a syntax error in the tree and for a mutant that does not compile.

| run | test | expected | observed |
|---|---|---|---|
| head, unmutated | whole-plan and `tile_ranges_partition_the_tiled_box_walk` | return 0, `1 passed` | return 0, `1 passed` (both) |
| r1: tiled branch → `apply_fused_range` over the element range | whole-plan | 101, tile-count assertion | 101, `must walk the serial path's tiles` |
| r2: tiled dispatch serialized (closure once over `0..width`) | whole-plan | 101, dispatch assertion | 101, `dispatched 0 chunks` |
| r3: only `execute` serial, `execute_uninit` parallel | whole-plan | 101, per-entry dispatch assertion | 101, `execute: ... dispatched 0 chunks` |
| r5: every tile of each segment in the range that starts it | whole-plan | 101, per-segment tiled-range assertion | 101, `tiles of segment 0 ... (walked by 1)` |
| r9: segment 0 whole in the first range, segment 2 whole in the last | whole-plan | 101, per-segment tiled-range assertion | 101, `tiles of segment 0 ... (walked by 1)` |
| r6: the two ranges of a cut walked second range first | `tile_ranges_partition_the_tiled_box_walk` | 101, visit-order assertion | 101, `cut at 1 must visit the slots in the whole-box order` |
| r7: `apply_tile_range` numbers and advances its tile grid last axis fastest (start decoding and odometer alike) | `tile_ranges_partition_the_tiled_box_walk` | 101, specified-order assertion | 101, `the whole-box walk must visit the slots in the specified tile order` |
| red first, `22bde1b` | whole-plan | 101, tile-count assertion | 101, `must walk the serial path's tiles` |

"whole-plan" is `concatenate_whole_plan_tiles_permuted_segments_like_serial_segments`. Since `0309a97` it records, per segment, the dispatched ranges that walked that segment's tiles. It requires at least two for each of the two transposed segments, and none for the contiguous one. r9 is the counterexample from the review of `f62d82e`: it passes the earlier test, which summed the ranges over all segments, and fails this one (Appendix E).

What the order checks establish, and what they do not:

- `tile_ranges_partition_the_tiled_box_walk` compares the walk of `0..cut` followed by `cut..count` with `apply_tiled_box` over the whole box. That is a comparison **within the same implementation**: `apply_tiled_box` itself walks through `apply_tile_range`, so the check is not an independent reference. It shows that splitting the grid does not change the walk (r6).
- Since `0309a97`, the test also compares the whole-box sequence with `specified_tile_order`. That reference is built from the documented order alone: the tile grid column-major with axis 0 fastest, tiles anchored at the origin and clipped, and each tile column-major. It catches a consistent change of the grid order in `apply_tile_range` (r7).
- It is a reference built from the specification, not from the implementation before this change. That the pre-change `apply_tiled_box` walked exactly that order rests on its documentation, which this PR keeps unchanged, not on a test run against the old code.

## Appendix A: raw benchmark output

```
# date: 2026-11-10 01:08:06 +0900
# host: Apple M5 Max, 18 logical CPUs, macOS 26.7.2
# rustc 1.96.0 (ac68faa20 2026-05-25)
# cargo 1.96.0 (30a34c682 2026-05-25)
# head binary: strided-basic at f43e2a1 (cargo bench profile, --features parallel), sha256 59ade9d6fa32d935
# base binary: strided-basic at 31099b8 (origin/main) + benches/concat_parallel.rs and its Cargo.toml [[bench]] entry from f43e2a1, sha256 cc7e3b4fd3b96e6c
# A/A: the base binary run a second time in each round (label base-again)
# CONCAT_BENCH_REPS=201, CONCAT_BENCH_THREADS=4 (default); 5 rounds, order rotated per round
## round 1 base  load: 6.95 5.88 5.23
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.073 ms    16.5 GB/s   1.030 Gelem/s  p10..p90 2.842..5.559 ms
4x[1024x1024] all T      per-segment 4T      1.487 ms    45.1 GB/s   2.821 Gelem/s  p10..p90 1.242..1.605 ms
4x[1024x1024] all T      serial 1T           3.992 ms    16.8 GB/s   1.051 Gelem/s  p10..p90 3.864..4.221 ms
4x[1024x1024] all T      whole/per-segment = 2.740
4x[1024x1024] T,C,T,C    whole-plan 4T       2.982 ms    22.5 GB/s   1.406 Gelem/s  p10..p90 1.482..3.601 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.986 ms    68.1 GB/s   4.254 Gelem/s  p10..p90 0.872..1.152 ms
4x[1024x1024] T,C,T,C    serial 1T           2.119 ms    31.7 GB/s   1.979 Gelem/s  p10..p90 2.034..2.264 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.025
4x[256x256] all T        whole-plan 4T       0.137 ms    30.5 GB/s   1.907 Gelem/s  p10..p90 0.129..0.144 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.5 GB/s   1.905 Gelem/s  p10..p90 0.129..0.145 ms
4x[256x256] all T        serial 1T           0.137 ms    30.7 GB/s   1.916 Gelem/s  p10..p90 0.129..0.144 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.956 ms    35.1 GB/s   2.194 Gelem/s  p10..p90 0.829..1.206 ms
8x[512x512] all T        per-segment 4T      1.571 ms    21.4 GB/s   1.335 Gelem/s  p10..p90 1.470..1.686 ms
8x[512x512] all T        serial 1T           1.583 ms    21.2 GB/s   1.325 Gelem/s  p10..p90 1.494..1.723 ms
8x[512x512] all T        whole/per-segment = 0.609
4x[1024x1024] all C      whole-plan 4T       0.248 ms   270.4 GB/s  16.901 Gelem/s  p10..p90 0.217..0.288 ms
4x[1024x1024] all C      per-segment 4T      0.310 ms   216.3 GB/s  13.521 Gelem/s  p10..p90 0.271..0.353 ms
4x[1024x1024] all C      serial 1T           0.525 ms   127.9 GB/s   7.992 Gelem/s  p10..p90 0.505..0.543 ms
4x[1024x1024] all C      whole/per-segment = 0.800
## round 1 head  load: 6.72 5.85 5.22
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.206 ms    55.6 GB/s   3.477 Gelem/s  p10..p90 1.168..1.269 ms
4x[1024x1024] all T      per-segment 4T      1.270 ms    52.8 GB/s   3.303 Gelem/s  p10..p90 1.228..1.335 ms
4x[1024x1024] all T      serial 1T           4.029 ms    16.7 GB/s   1.041 Gelem/s  p10..p90 3.892..4.159 ms
4x[1024x1024] all T      whole/per-segment = 0.950
4x[1024x1024] T,C,T,C    whole-plan 4T       0.752 ms    89.3 GB/s   5.579 Gelem/s  p10..p90 0.711..0.822 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.818 ms    82.0 GB/s   5.126 Gelem/s  p10..p90 0.766..0.960 ms
4x[1024x1024] T,C,T,C    serial 1T           2.288 ms    29.3 GB/s   1.833 Gelem/s  p10..p90 2.185..2.394 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.919
4x[256x256] all T        whole-plan 4T       0.137 ms    30.6 GB/s   1.915 Gelem/s  p10..p90 0.129..0.141 ms
4x[256x256] all T        per-segment 4T      0.137 ms    30.6 GB/s   1.911 Gelem/s  p10..p90 0.130..0.142 ms
4x[256x256] all T        serial 1T           0.137 ms    30.6 GB/s   1.911 Gelem/s  p10..p90 0.130..0.142 ms
4x[256x256] all T        whole/per-segment = 0.998
8x[512x512] all T        whole-plan 4T       0.472 ms    71.1 GB/s   4.446 Gelem/s  p10..p90 0.446..0.538 ms
8x[512x512] all T        per-segment 4T      1.627 ms    20.6 GB/s   1.289 Gelem/s  p10..p90 1.528..1.748 ms
8x[512x512] all T        serial 1T           1.627 ms    20.6 GB/s   1.289 Gelem/s  p10..p90 1.540..1.750 ms
8x[512x512] all T        whole/per-segment = 0.290
4x[1024x1024] all C      whole-plan 4T       0.270 ms   248.9 GB/s  15.556 Gelem/s  p10..p90 0.254..0.292 ms
4x[1024x1024] all C      per-segment 4T      0.293 ms   228.7 GB/s  14.297 Gelem/s  p10..p90 0.275..0.343 ms
4x[1024x1024] all C      serial 1T           0.535 ms   125.5 GB/s   7.841 Gelem/s  p10..p90 0.516..0.555 ms
4x[1024x1024] all C      whole/per-segment = 0.919
## round 1 base-again  load: 6.42 5.81 5.21
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       3.718 ms    18.1 GB/s   1.128 Gelem/s  p10..p90 2.826..5.008 ms
4x[1024x1024] all T      per-segment 4T      1.500 ms    44.7 GB/s   2.796 Gelem/s  p10..p90 1.424..1.603 ms
4x[1024x1024] all T      serial 1T           3.977 ms    16.9 GB/s   1.055 Gelem/s  p10..p90 3.813..4.216 ms
4x[1024x1024] all T      whole/per-segment = 2.479
4x[1024x1024] T,C,T,C    whole-plan 4T       2.966 ms    22.6 GB/s   1.414 Gelem/s  p10..p90 1.991..3.619 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.968 ms    69.3 GB/s   4.334 Gelem/s  p10..p90 0.858..1.161 ms
4x[1024x1024] T,C,T,C    serial 1T           2.109 ms    31.8 GB/s   1.988 Gelem/s  p10..p90 2.006..2.234 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.064
4x[256x256] all T        whole-plan 4T       0.136 ms    30.8 GB/s   1.923 Gelem/s  p10..p90 0.129..0.140 ms
4x[256x256] all T        per-segment 4T      0.137 ms    30.7 GB/s   1.916 Gelem/s  p10..p90 0.129..0.141 ms
4x[256x256] all T        serial 1T           0.137 ms    30.7 GB/s   1.917 Gelem/s  p10..p90 0.129..0.142 ms
4x[256x256] all T        whole/per-segment = 0.996
8x[512x512] all T        whole-plan 4T       0.938 ms    35.8 GB/s   2.237 Gelem/s  p10..p90 0.832..1.180 ms
8x[512x512] all T        per-segment 4T      1.602 ms    20.9 GB/s   1.309 Gelem/s  p10..p90 1.464..1.798 ms
8x[512x512] all T        serial 1T           1.620 ms    20.7 GB/s   1.294 Gelem/s  p10..p90 1.471..1.837 ms
8x[512x512] all T        whole/per-segment = 0.585
4x[1024x1024] all C      whole-plan 4T       0.256 ms   262.6 GB/s  16.411 Gelem/s  p10..p90 0.224..0.295 ms
4x[1024x1024] all C      per-segment 4T      0.321 ms   209.0 GB/s  13.063 Gelem/s  p10..p90 0.272..0.373 ms
4x[1024x1024] all C      serial 1T           0.531 ms   126.4 GB/s   7.897 Gelem/s  p10..p90 0.517..0.545 ms
4x[1024x1024] all C      whole/per-segment = 0.796
## round 2 head  load: 6.42 5.81 5.21
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.214 ms    55.3 GB/s   3.454 Gelem/s  p10..p90 1.174..1.272 ms
4x[1024x1024] all T      per-segment 4T      1.274 ms    52.7 GB/s   3.292 Gelem/s  p10..p90 1.229..1.382 ms
4x[1024x1024] all T      serial 1T           4.035 ms    16.6 GB/s   1.039 Gelem/s  p10..p90 3.905..4.187 ms
4x[1024x1024] all T      whole/per-segment = 0.953
4x[1024x1024] T,C,T,C    whole-plan 4T       0.751 ms    89.4 GB/s   5.587 Gelem/s  p10..p90 0.704..0.807 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.810 ms    82.8 GB/s   5.175 Gelem/s  p10..p90 0.757..0.896 ms
4x[1024x1024] T,C,T,C    serial 1T           2.275 ms    29.5 GB/s   1.844 Gelem/s  p10..p90 2.164..2.376 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.926
4x[256x256] all T        whole-plan 4T       0.138 ms    30.4 GB/s   1.902 Gelem/s  p10..p90 0.131..0.144 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.4 GB/s   1.900 Gelem/s  p10..p90 0.133..0.142 ms
4x[256x256] all T        serial 1T           0.138 ms    30.4 GB/s   1.903 Gelem/s  p10..p90 0.131..0.143 ms
4x[256x256] all T        whole/per-segment = 0.998
8x[512x512] all T        whole-plan 4T       0.458 ms    73.3 GB/s   4.583 Gelem/s  p10..p90 0.440..0.493 ms
8x[512x512] all T        per-segment 4T      1.599 ms    21.0 GB/s   1.312 Gelem/s  p10..p90 1.534..1.689 ms
8x[512x512] all T        serial 1T           1.604 ms    20.9 GB/s   1.307 Gelem/s  p10..p90 1.525..1.689 ms
8x[512x512] all T        whole/per-segment = 0.286
4x[1024x1024] all C      whole-plan 4T       0.264 ms   254.1 GB/s  15.883 Gelem/s  p10..p90 0.249..0.291 ms
4x[1024x1024] all C      per-segment 4T      0.293 ms   229.3 GB/s  14.329 Gelem/s  p10..p90 0.276..0.347 ms
4x[1024x1024] all C      serial 1T           0.542 ms   123.9 GB/s   7.745 Gelem/s  p10..p90 0.522..0.555 ms
4x[1024x1024] all C      whole/per-segment = 0.902
## round 2 base-again  load: 6.14 5.76 5.20
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.391 ms    15.3 GB/s   0.955 Gelem/s  p10..p90 3.038..5.882 ms
4x[1024x1024] all T      per-segment 4T      1.483 ms    45.3 GB/s   2.829 Gelem/s  p10..p90 1.349..1.591 ms
4x[1024x1024] all T      serial 1T           4.132 ms    16.2 GB/s   1.015 Gelem/s  p10..p90 3.867..4.249 ms
4x[1024x1024] all T      whole/per-segment = 2.961
4x[1024x1024] T,C,T,C    whole-plan 4T       2.917 ms    23.0 GB/s   1.438 Gelem/s  p10..p90 1.457..4.102 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.984 ms    68.2 GB/s   4.261 Gelem/s  p10..p90 0.815..1.167 ms
4x[1024x1024] T,C,T,C    serial 1T           2.252 ms    29.8 GB/s   1.863 Gelem/s  p10..p90 2.176..2.335 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 2.964
4x[256x256] all T        whole-plan 4T       0.146 ms    28.8 GB/s   1.798 Gelem/s  p10..p90 0.139..0.155 ms
4x[256x256] all T        per-segment 4T      0.146 ms    28.7 GB/s   1.796 Gelem/s  p10..p90 0.140..0.156 ms
4x[256x256] all T        serial 1T           0.145 ms    28.9 GB/s   1.804 Gelem/s  p10..p90 0.139..0.154 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.929 ms    36.1 GB/s   2.257 Gelem/s  p10..p90 0.838..1.007 ms
8x[512x512] all T        per-segment 4T      1.610 ms    20.8 GB/s   1.303 Gelem/s  p10..p90 1.573..1.677 ms
8x[512x512] all T        serial 1T           1.623 ms    20.7 GB/s   1.292 Gelem/s  p10..p90 1.579..1.717 ms
8x[512x512] all T        whole/per-segment = 0.577
4x[1024x1024] all C      whole-plan 4T       0.260 ms   257.7 GB/s  16.106 Gelem/s  p10..p90 0.227..0.304 ms
4x[1024x1024] all C      per-segment 4T      0.321 ms   209.4 GB/s  13.085 Gelem/s  p10..p90 0.281..0.396 ms
4x[1024x1024] all C      serial 1T           0.550 ms   122.1 GB/s   7.629 Gelem/s  p10..p90 0.533..0.573 ms
4x[1024x1024] all C      whole/per-segment = 0.812
## round 2 base  load: 6.45 5.83 5.22
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.463 ms    15.0 GB/s   0.940 Gelem/s  p10..p90 3.017..5.949 ms
4x[1024x1024] all T      per-segment 4T      1.270 ms    52.8 GB/s   3.302 Gelem/s  p10..p90 1.190..1.496 ms
4x[1024x1024] all T      serial 1T           4.039 ms    16.6 GB/s   1.038 Gelem/s  p10..p90 3.835..4.228 ms
4x[1024x1024] all T      whole/per-segment = 3.514
4x[1024x1024] T,C,T,C    whole-plan 4T       2.978 ms    22.5 GB/s   1.409 Gelem/s  p10..p90 1.408..3.591 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.963 ms    69.7 GB/s   4.357 Gelem/s  p10..p90 0.792..1.145 ms
4x[1024x1024] T,C,T,C    serial 1T           2.067 ms    32.5 GB/s   2.029 Gelem/s  p10..p90 1.979..2.192 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.093
4x[256x256] all T        whole-plan 4T       0.137 ms    30.5 GB/s   1.909 Gelem/s  p10..p90 0.129..0.141 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.5 GB/s   1.905 Gelem/s  p10..p90 0.129..0.142 ms
4x[256x256] all T        serial 1T           0.138 ms    30.4 GB/s   1.902 Gelem/s  p10..p90 0.130..0.143 ms
4x[256x256] all T        whole/per-segment = 0.998
8x[512x512] all T        whole-plan 4T       0.909 ms    36.9 GB/s   2.307 Gelem/s  p10..p90 0.794..1.015 ms
8x[512x512] all T        per-segment 4T      1.511 ms    22.2 GB/s   1.388 Gelem/s  p10..p90 1.433..1.667 ms
8x[512x512] all T        serial 1T           1.528 ms    22.0 GB/s   1.372 Gelem/s  p10..p90 1.437..1.692 ms
8x[512x512] all T        whole/per-segment = 0.602
4x[1024x1024] all C      whole-plan 4T       0.264 ms   253.8 GB/s  15.865 Gelem/s  p10..p90 0.227..0.310 ms
4x[1024x1024] all C      per-segment 4T      0.324 ms   207.2 GB/s  12.947 Gelem/s  p10..p90 0.286..0.383 ms
4x[1024x1024] all C      serial 1T           0.533 ms   126.0 GB/s   7.874 Gelem/s  p10..p90 0.522..0.547 ms
4x[1024x1024] all C      whole/per-segment = 0.816
## round 3 base-again  load: 6.26 5.80 5.22
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       3.450 ms    19.5 GB/s   1.216 Gelem/s  p10..p90 2.831..4.898 ms
4x[1024x1024] all T      per-segment 4T      1.493 ms    44.9 GB/s   2.809 Gelem/s  p10..p90 1.413..1.629 ms
4x[1024x1024] all T      serial 1T           4.018 ms    16.7 GB/s   1.044 Gelem/s  p10..p90 3.879..4.190 ms
4x[1024x1024] all T      whole/per-segment = 2.310
4x[1024x1024] T,C,T,C    whole-plan 4T       3.017 ms    22.2 GB/s   1.390 Gelem/s  p10..p90 1.386..3.890 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.981 ms    68.4 GB/s   4.276 Gelem/s  p10..p90 0.821..1.160 ms
4x[1024x1024] T,C,T,C    serial 1T           2.149 ms    31.2 GB/s   1.952 Gelem/s  p10..p90 2.045..2.282 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.076
4x[256x256] all T        whole-plan 4T       0.140 ms    29.9 GB/s   1.870 Gelem/s  p10..p90 0.138..0.143 ms
4x[256x256] all T        per-segment 4T      0.141 ms    29.8 GB/s   1.865 Gelem/s  p10..p90 0.139..0.144 ms
4x[256x256] all T        serial 1T           0.140 ms    29.9 GB/s   1.870 Gelem/s  p10..p90 0.139..0.143 ms
4x[256x256] all T        whole/per-segment = 0.997
8x[512x512] all T        whole-plan 4T       0.895 ms    37.5 GB/s   2.342 Gelem/s  p10..p90 0.796..0.988 ms
8x[512x512] all T        per-segment 4T      1.516 ms    22.1 GB/s   1.383 Gelem/s  p10..p90 1.434..1.629 ms
8x[512x512] all T        serial 1T           1.531 ms    21.9 GB/s   1.370 Gelem/s  p10..p90 1.446..1.627 ms
8x[512x512] all T        whole/per-segment = 0.590
4x[1024x1024] all C      whole-plan 4T       0.251 ms   267.4 GB/s  16.713 Gelem/s  p10..p90 0.226..0.285 ms
4x[1024x1024] all C      per-segment 4T      0.315 ms   213.2 GB/s  13.326 Gelem/s  p10..p90 0.275..0.366 ms
4x[1024x1024] all C      serial 1T           0.522 ms   128.6 GB/s   8.038 Gelem/s  p10..p90 0.502..0.540 ms
4x[1024x1024] all C      whole/per-segment = 0.797
## round 3 base  load: 6.31 5.82 5.23
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.788 ms    14.0 GB/s   0.876 Gelem/s  p10..p90 2.879..5.921 ms
4x[1024x1024] all T      per-segment 4T      1.331 ms    50.4 GB/s   3.151 Gelem/s  p10..p90 1.200..1.573 ms
4x[1024x1024] all T      serial 1T           3.945 ms    17.0 GB/s   1.063 Gelem/s  p10..p90 3.824..4.108 ms
4x[1024x1024] all T      whole/per-segment = 3.596
4x[1024x1024] T,C,T,C    whole-plan 4T       3.127 ms    21.5 GB/s   1.341 Gelem/s  p10..p90 1.469..3.672 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.985 ms    68.1 GB/s   4.258 Gelem/s  p10..p90 0.821..1.153 ms
4x[1024x1024] T,C,T,C    serial 1T           2.152 ms    31.2 GB/s   1.949 Gelem/s  p10..p90 2.040..2.295 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.174
4x[256x256] all T        whole-plan 4T       0.139 ms    30.1 GB/s   1.880 Gelem/s  p10..p90 0.138..0.144 ms
4x[256x256] all T        per-segment 4T      0.139 ms    30.1 GB/s   1.881 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        serial 1T           0.139 ms    30.1 GB/s   1.881 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        whole/per-segment = 1.000
8x[512x512] all T        whole-plan 4T       0.917 ms    36.6 GB/s   2.286 Gelem/s  p10..p90 0.820..1.017 ms
8x[512x512] all T        per-segment 4T      1.476 ms    22.7 GB/s   1.420 Gelem/s  p10..p90 1.426..1.544 ms
8x[512x512] all T        serial 1T           1.497 ms    22.4 GB/s   1.401 Gelem/s  p10..p90 1.430..1.561 ms
8x[512x512] all T        whole/per-segment = 0.621
4x[1024x1024] all C      whole-plan 4T       0.249 ms   269.8 GB/s  16.862 Gelem/s  p10..p90 0.223..0.285 ms
4x[1024x1024] all C      per-segment 4T      0.311 ms   215.8 GB/s  13.485 Gelem/s  p10..p90 0.271..0.363 ms
4x[1024x1024] all C      serial 1T           0.525 ms   127.9 GB/s   7.994 Gelem/s  p10..p90 0.507..0.542 ms
4x[1024x1024] all C      whole/per-segment = 0.800
## round 3 head  load: 5.89 5.74 5.20
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.207 ms    55.6 GB/s   3.476 Gelem/s  p10..p90 1.174..1.271 ms
4x[1024x1024] all T      per-segment 4T      1.269 ms    52.9 GB/s   3.305 Gelem/s  p10..p90 1.230..1.392 ms
4x[1024x1024] all T      serial 1T           4.046 ms    16.6 GB/s   1.037 Gelem/s  p10..p90 3.897..4.204 ms
4x[1024x1024] all T      whole/per-segment = 0.951
4x[1024x1024] T,C,T,C    whole-plan 4T       0.726 ms    92.4 GB/s   5.776 Gelem/s  p10..p90 0.703..0.764 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.778 ms    86.3 GB/s   5.392 Gelem/s  p10..p90 0.756..0.820 ms
4x[1024x1024] T,C,T,C    serial 1T           2.220 ms    30.2 GB/s   1.889 Gelem/s  p10..p90 2.148..2.305 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.934
4x[256x256] all T        whole-plan 4T       0.137 ms    30.6 GB/s   1.909 Gelem/s  p10..p90 0.129..0.144 ms
4x[256x256] all T        per-segment 4T      0.137 ms    30.6 GB/s   1.914 Gelem/s  p10..p90 0.129..0.141 ms
4x[256x256] all T        serial 1T           0.137 ms    30.6 GB/s   1.912 Gelem/s  p10..p90 0.129..0.142 ms
4x[256x256] all T        whole/per-segment = 1.002
8x[512x512] all T        whole-plan 4T       0.455 ms    73.7 GB/s   4.608 Gelem/s  p10..p90 0.441..0.487 ms
8x[512x512] all T        per-segment 4T      1.586 ms    21.2 GB/s   1.322 Gelem/s  p10..p90 1.514..1.685 ms
8x[512x512] all T        serial 1T           1.605 ms    20.9 GB/s   1.307 Gelem/s  p10..p90 1.531..1.702 ms
8x[512x512] all T        whole/per-segment = 0.287
4x[1024x1024] all C      whole-plan 4T       0.263 ms   255.0 GB/s  15.938 Gelem/s  p10..p90 0.247..0.278 ms
4x[1024x1024] all C      per-segment 4T      0.285 ms   235.3 GB/s  14.704 Gelem/s  p10..p90 0.273..0.325 ms
4x[1024x1024] all C      serial 1T           0.542 ms   123.8 GB/s   7.737 Gelem/s  p10..p90 0.524..0.559 ms
4x[1024x1024] all C      whole/per-segment = 0.923
## round 4 base  load: 5.89 5.74 5.20
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.167 ms    16.1 GB/s   1.006 Gelem/s  p10..p90 2.894..5.750 ms
4x[1024x1024] all T      per-segment 4T      1.474 ms    45.5 GB/s   2.846 Gelem/s  p10..p90 1.237..1.570 ms
4x[1024x1024] all T      serial 1T           3.986 ms    16.8 GB/s   1.052 Gelem/s  p10..p90 3.836..4.207 ms
4x[1024x1024] all T      whole/per-segment = 2.828
4x[1024x1024] T,C,T,C    whole-plan 4T       3.217 ms    20.9 GB/s   1.304 Gelem/s  p10..p90 1.464..4.357 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.993 ms    67.6 GB/s   4.224 Gelem/s  p10..p90 0.819..1.167 ms
4x[1024x1024] T,C,T,C    serial 1T           2.228 ms    30.1 GB/s   1.882 Gelem/s  p10..p90 2.054..2.347 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.239
4x[256x256] all T        whole-plan 4T       0.137 ms    30.6 GB/s   1.913 Gelem/s  p10..p90 0.129..0.142 ms
4x[256x256] all T        per-segment 4T      0.137 ms    30.7 GB/s   1.916 Gelem/s  p10..p90 0.130..0.141 ms
4x[256x256] all T        serial 1T           0.137 ms    30.7 GB/s   1.916 Gelem/s  p10..p90 0.129..0.141 ms
4x[256x256] all T        whole/per-segment = 1.001
8x[512x512] all T        whole-plan 4T       0.899 ms    37.3 GB/s   2.331 Gelem/s  p10..p90 0.795..0.976 ms
8x[512x512] all T        per-segment 4T      1.478 ms    22.7 GB/s   1.419 Gelem/s  p10..p90 1.423..1.586 ms
8x[512x512] all T        serial 1T           1.504 ms    22.3 GB/s   1.394 Gelem/s  p10..p90 1.433..1.607 ms
8x[512x512] all T        whole/per-segment = 0.609
4x[1024x1024] all C      whole-plan 4T       0.265 ms   252.9 GB/s  15.808 Gelem/s  p10..p90 0.234..0.319 ms
4x[1024x1024] all C      per-segment 4T      0.326 ms   206.1 GB/s  12.882 Gelem/s  p10..p90 0.289..0.384 ms
4x[1024x1024] all C      serial 1T           0.535 ms   125.4 GB/s   7.839 Gelem/s  p10..p90 0.522..0.550 ms
4x[1024x1024] all C      whole/per-segment = 0.815
## round 4 head  load: 5.82 5.72 5.20
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.192 ms    56.3 GB/s   3.520 Gelem/s  p10..p90 1.160..1.247 ms
4x[1024x1024] all T      per-segment 4T      1.249 ms    53.7 GB/s   3.359 Gelem/s  p10..p90 1.220..1.322 ms
4x[1024x1024] all T      serial 1T           3.988 ms    16.8 GB/s   1.052 Gelem/s  p10..p90 3.868..4.141 ms
4x[1024x1024] all T      whole/per-segment = 0.954
4x[1024x1024] T,C,T,C    whole-plan 4T       0.732 ms    91.7 GB/s   5.729 Gelem/s  p10..p90 0.707..0.783 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.784 ms    85.6 GB/s   5.349 Gelem/s  p10..p90 0.758..0.839 ms
4x[1024x1024] T,C,T,C    serial 1T           2.218 ms    30.3 GB/s   1.891 Gelem/s  p10..p90 2.146..2.311 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.934
4x[256x256] all T        whole-plan 4T       0.137 ms    30.5 GB/s   1.907 Gelem/s  p10..p90 0.130..0.144 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.5 GB/s   1.904 Gelem/s  p10..p90 0.130..0.144 ms
4x[256x256] all T        serial 1T           0.137 ms    30.6 GB/s   1.911 Gelem/s  p10..p90 0.130..0.145 ms
4x[256x256] all T        whole/per-segment = 0.998
8x[512x512] all T        whole-plan 4T       0.459 ms    73.1 GB/s   4.567 Gelem/s  p10..p90 0.442..0.493 ms
8x[512x512] all T        per-segment 4T      1.577 ms    21.3 GB/s   1.330 Gelem/s  p10..p90 1.507..1.677 ms
8x[512x512] all T        serial 1T           1.596 ms    21.0 GB/s   1.314 Gelem/s  p10..p90 1.510..1.682 ms
8x[512x512] all T        whole/per-segment = 0.291
4x[1024x1024] all C      whole-plan 4T       0.264 ms   253.8 GB/s  15.863 Gelem/s  p10..p90 0.248..0.284 ms
4x[1024x1024] all C      per-segment 4T      0.290 ms   231.7 GB/s  14.482 Gelem/s  p10..p90 0.274..0.318 ms
4x[1024x1024] all C      serial 1T           0.543 ms   123.7 GB/s   7.730 Gelem/s  p10..p90 0.523..0.555 ms
4x[1024x1024] all C      whole/per-segment = 0.913
## round 4 base-again  load: 5.51 5.66 5.18
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       5.221 ms    12.9 GB/s   0.803 Gelem/s  p10..p90 3.143..6.652 ms
4x[1024x1024] all T      per-segment 4T      1.240 ms    54.1 GB/s   3.382 Gelem/s  p10..p90 1.175..1.476 ms
4x[1024x1024] all T      serial 1T           3.916 ms    17.1 GB/s   1.071 Gelem/s  p10..p90 3.753..4.090 ms
4x[1024x1024] all T      whole/per-segment = 4.210
4x[1024x1024] T,C,T,C    whole-plan 4T       2.939 ms    22.8 GB/s   1.427 Gelem/s  p10..p90 1.341..3.369 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.971 ms    69.1 GB/s   4.320 Gelem/s  p10..p90 0.779..1.150 ms
4x[1024x1024] T,C,T,C    serial 1T           2.062 ms    32.5 GB/s   2.034 Gelem/s  p10..p90 1.977..2.183 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.027
4x[256x256] all T        whole-plan 4T       0.137 ms    30.6 GB/s   1.912 Gelem/s  p10..p90 0.129..0.146 ms
4x[256x256] all T        per-segment 4T      0.137 ms    30.6 GB/s   1.909 Gelem/s  p10..p90 0.129..0.144 ms
4x[256x256] all T        serial 1T           0.137 ms    30.6 GB/s   1.915 Gelem/s  p10..p90 0.129..0.143 ms
4x[256x256] all T        whole/per-segment = 0.998
8x[512x512] all T        whole-plan 4T       0.906 ms    37.0 GB/s   2.314 Gelem/s  p10..p90 0.819..0.978 ms
8x[512x512] all T        per-segment 4T      1.519 ms    22.1 GB/s   1.381 Gelem/s  p10..p90 1.440..1.587 ms
8x[512x512] all T        serial 1T           1.524 ms    22.0 GB/s   1.376 Gelem/s  p10..p90 1.440..1.613 ms
8x[512x512] all T        whole/per-segment = 0.597
4x[1024x1024] all C      whole-plan 4T       0.262 ms   256.3 GB/s  16.022 Gelem/s  p10..p90 0.231..0.289 ms
4x[1024x1024] all C      per-segment 4T      0.315 ms   213.2 GB/s  13.328 Gelem/s  p10..p90 0.283..0.367 ms
4x[1024x1024] all C      serial 1T           0.537 ms   125.0 GB/s   7.815 Gelem/s  p10..p90 0.521..0.553 ms
4x[1024x1024] all C      whole/per-segment = 0.832
## round 5 head  load: 5.47 5.65 5.18
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.205 ms    55.7 GB/s   3.481 Gelem/s  p10..p90 1.167..1.271 ms
4x[1024x1024] all T      per-segment 4T      1.264 ms    53.1 GB/s   3.318 Gelem/s  p10..p90 1.229..1.404 ms
4x[1024x1024] all T      serial 1T           4.040 ms    16.6 GB/s   1.038 Gelem/s  p10..p90 3.914..4.219 ms
4x[1024x1024] all T      whole/per-segment = 0.953
4x[1024x1024] T,C,T,C    whole-plan 4T       0.728 ms    92.2 GB/s   5.764 Gelem/s  p10..p90 0.699..0.782 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.787 ms    85.3 GB/s   5.330 Gelem/s  p10..p90 0.755..0.859 ms
4x[1024x1024] T,C,T,C    serial 1T           2.216 ms    30.3 GB/s   1.893 Gelem/s  p10..p90 2.150..2.318 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.925
4x[256x256] all T        whole-plan 4T       0.138 ms    30.4 GB/s   1.902 Gelem/s  p10..p90 0.130..0.144 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.3 GB/s   1.896 Gelem/s  p10..p90 0.131..0.143 ms
4x[256x256] all T        serial 1T           0.138 ms    30.4 GB/s   1.902 Gelem/s  p10..p90 0.131..0.143 ms
4x[256x256] all T        whole/per-segment = 0.996
8x[512x512] all T        whole-plan 4T       0.455 ms    73.7 GB/s   4.609 Gelem/s  p10..p90 0.436..0.484 ms
8x[512x512] all T        per-segment 4T      1.582 ms    21.2 GB/s   1.326 Gelem/s  p10..p90 1.502..1.676 ms
8x[512x512] all T        serial 1T           1.607 ms    20.9 GB/s   1.305 Gelem/s  p10..p90 1.517..1.692 ms
8x[512x512] all T        whole/per-segment = 0.288
4x[1024x1024] all C      whole-plan 4T       0.263 ms   255.0 GB/s  15.940 Gelem/s  p10..p90 0.248..0.276 ms
4x[1024x1024] all C      per-segment 4T      0.287 ms   234.2 GB/s  14.638 Gelem/s  p10..p90 0.274..0.314 ms
4x[1024x1024] all C      serial 1T           0.541 ms   124.0 GB/s   7.747 Gelem/s  p10..p90 0.524..0.557 ms
4x[1024x1024] all C      whole/per-segment = 0.918
## round 5 base-again  load: 5.47 5.65 5.18
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       5.087 ms    13.2 GB/s   0.825 Gelem/s  p10..p90 3.043..6.253 ms
4x[1024x1024] all T      per-segment 4T      1.264 ms    53.1 GB/s   3.319 Gelem/s  p10..p90 1.199..1.525 ms
4x[1024x1024] all T      serial 1T           3.956 ms    17.0 GB/s   1.060 Gelem/s  p10..p90 3.826..4.130 ms
4x[1024x1024] all T      whole/per-segment = 4.025
4x[1024x1024] T,C,T,C    whole-plan 4T       2.999 ms    22.4 GB/s   1.399 Gelem/s  p10..p90 2.641..3.632 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.995 ms    67.4 GB/s   4.215 Gelem/s  p10..p90 0.871..1.174 ms
4x[1024x1024] T,C,T,C    serial 1T           2.107 ms    31.9 GB/s   1.991 Gelem/s  p10..p90 2.021..2.266 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.014
4x[256x256] all T        whole-plan 4T       0.138 ms    30.4 GB/s   1.902 Gelem/s  p10..p90 0.130..0.146 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.5 GB/s   1.907 Gelem/s  p10..p90 0.130..0.143 ms
4x[256x256] all T        serial 1T           0.137 ms    30.5 GB/s   1.908 Gelem/s  p10..p90 0.130..0.144 ms
4x[256x256] all T        whole/per-segment = 1.002
8x[512x512] all T        whole-plan 4T       0.935 ms    35.9 GB/s   2.243 Gelem/s  p10..p90 0.821..1.059 ms
8x[512x512] all T        per-segment 4T      1.481 ms    22.7 GB/s   1.416 Gelem/s  p10..p90 1.430..1.569 ms
8x[512x512] all T        serial 1T           1.501 ms    22.4 GB/s   1.398 Gelem/s  p10..p90 1.437..1.592 ms
8x[512x512] all T        whole/per-segment = 0.631
4x[1024x1024] all C      whole-plan 4T       0.253 ms   265.3 GB/s  16.581 Gelem/s  p10..p90 0.223..0.291 ms
4x[1024x1024] all C      per-segment 4T      0.312 ms   215.4 GB/s  13.463 Gelem/s  p10..p90 0.272..0.359 ms
4x[1024x1024] all C      serial 1T           0.526 ms   127.6 GB/s   7.976 Gelem/s  p10..p90 0.510..0.543 ms
4x[1024x1024] all C      whole/per-segment = 0.812
## round 5 base  load: 5.35 5.62 5.17
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       3.502 ms    19.2 GB/s   1.198 Gelem/s  p10..p90 2.804..4.678 ms
4x[1024x1024] all T      per-segment 4T      1.489 ms    45.1 GB/s   2.816 Gelem/s  p10..p90 1.400..1.636 ms
4x[1024x1024] all T      serial 1T           3.932 ms    17.1 GB/s   1.067 Gelem/s  p10..p90 3.795..4.061 ms
4x[1024x1024] all T      whole/per-segment = 2.352
4x[1024x1024] T,C,T,C    whole-plan 4T       3.212 ms    20.9 GB/s   1.306 Gelem/s  p10..p90 1.408..3.944 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.985 ms    68.1 GB/s   4.259 Gelem/s  p10..p90 0.823..1.161 ms
4x[1024x1024] T,C,T,C    serial 1T           2.188 ms    30.7 GB/s   1.917 Gelem/s  p10..p90 2.039..2.290 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.261
4x[256x256] all T        whole-plan 4T       0.137 ms    30.5 GB/s   1.907 Gelem/s  p10..p90 0.129..0.142 ms
4x[256x256] all T        per-segment 4T      0.137 ms    30.5 GB/s   1.908 Gelem/s  p10..p90 0.130..0.142 ms
4x[256x256] all T        serial 1T           0.137 ms    30.5 GB/s   1.908 Gelem/s  p10..p90 0.129..0.141 ms
4x[256x256] all T        whole/per-segment = 1.000
8x[512x512] all T        whole-plan 4T       0.883 ms    38.0 GB/s   2.374 Gelem/s  p10..p90 0.793..0.958 ms
8x[512x512] all T        per-segment 4T      1.474 ms    22.8 GB/s   1.423 Gelem/s  p10..p90 1.426..1.540 ms
8x[512x512] all T        serial 1T           1.498 ms    22.4 GB/s   1.400 Gelem/s  p10..p90 1.437..1.583 ms
8x[512x512] all T        whole/per-segment = 0.599
4x[1024x1024] all C      whole-plan 4T       0.251 ms   267.5 GB/s  16.719 Gelem/s  p10..p90 0.225..0.282 ms
4x[1024x1024] all C      per-segment 4T      0.316 ms   212.4 GB/s  13.277 Gelem/s  p10..p90 0.269..0.365 ms
4x[1024x1024] all C      serial 1T           0.522 ms   128.6 GB/s   8.038 Gelem/s  p10..p90 0.505..0.539 ms
4x[1024x1024] all C      whole/per-segment = 0.794
```

## Appendix B: raw mutation output

```
$ python3 mutants.py ~/work/strided-rs
--- head 0309a97, unmutated: concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 0
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... ok
stdout| test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.31s
as expected

--- head 0309a97, unmutated: tile_ranges_partition_the_tiled_box_walk
$ cargo test -p strided-basic --lib tile_ranges_partition_the_tiled_box_walk
returncode: 0
stdout| test raw_ops::tiled_tests::tile_ranges_partition_the_tiled_box_walk ... ok
stdout| test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 261 filtered out; finished in 0.07s
as expected

--- mutant r1: tiled branch -> apply_fused_range over the element range
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
stdout| thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (147411781) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:645:9:
stdout| assertion `left == right` failed: execute: the whole-plan replay must walk the serial path's tiles
stdout| left: 0
stdout| right: 234
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.18s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

--- mutant r2: tiled dispatch serialized
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
stdout| thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (147412538) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:653:9:
stdout| execute: the whole-plan replay must dispatch the tiled segments across the budget (dispatched 0 chunks)
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.17s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

--- mutant r3: execute serial, execute_uninit parallel
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
stdout| thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (147413260) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:654:9:
stdout| execute: the whole-plan replay must dispatch the tiled segments across the budget (dispatched 0 chunks)
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.18s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

--- mutant r5: all tiles of a segment in its first range
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
stdout| thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (147413850) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:677:17:
stdout| execute: the tiles of segment 0 must be walked by at least two dispatched ranges (walked by 1)
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.18s
stderr| warning: unused variable: `segment_len`
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

--- mutant r9: each tiled segment whole in one range, segments in different ranges
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
stdout| thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (147414390) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:677:17:
stdout| execute: the tiles of segment 0 must be walked by at least two dispatched ranges (walked by 1)
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.18s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

--- mutant r6: the two ranges of each cut walked second range first
$ cargo test -p strided-basic --lib tile_ranges_partition_the_tiled_box_walk
returncode: 101
stdout| test raw_ops::tiled_tests::tile_ranges_partition_the_tiled_box_walk ... FAILED
stdout| thread 'raw_ops::tiled_tests::tile_ranges_partition_the_tiled_box_walk' (147414819) panicked at strided-basic/src/raw_ops/tests/tiled_tests.rs:1283:13:
stdout| 67x45 transpose: cut at 1 must visit the slots in the whole-box order
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 261 filtered out; finished in 0.01s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

--- mutant r7: tile grid last axis fastest in apply_tile_range
$ cargo test -p strided-basic --lib tile_ranges_partition_the_tiled_box_walk
returncode: 101
stdout| test raw_ops::tiled_tests::tile_ranges_partition_the_tiled_box_walk ... FAILED
stdout| thread 'raw_ops::tiled_tests::tile_ranges_partition_the_tiled_box_walk' (147415269) panicked at strided-basic/src/raw_ops/tests/tiled_tests.rs:1238:9:
stdout| 67x45 transpose: the whole-box walk must visit the slots in the specified tile order
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 261 filtered out; finished in 0.00s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

--- red first: worktree at 22bde1b (test added, fix absent)
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
stdout| thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (147416019) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:617:9:
stdout| assertion `left == right` failed: execute: the whole-plan replay must walk the serial path's tiles
stdout| left: 0
stdout| right: 234
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 270 filtered out; finished in 0.16s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
as expected

all expectations held: 2 unmutated passes, 7 mutants and red first failing in their assertions
exit: 0
```

## Appendix C: the mutation script

`python3 mutants.py <repo>`; it touches only the files it mutates and restores them. Exit 0 means every expectation held; it stops with exit 1 at the first one that did not.

```python
#!/usr/bin/env python3
"""Red-first and review-mutant runs for strided-rs#432, with enforced outcomes.

Usage: python3 mutants.py <repo>

Every expectation is checked, and the script exits 1 at the first one that
does not hold, after restoring the file it mutated:

- the unmutated tests must pass: return code 0 and `1 passed`;
- each mutant, and the red-first commit, must fail *in the intended
  assertion*: return code 101 (a failing test), `1 failed` in stdout, the
  expected panic message in stdout, `error: test failed` in stderr, and no
  compiler error (`error[E`, `could not compile`) in stderr.

A compilation failure therefore stops the script instead of passing for a
caught mutant. Exit 0 means every expectation held.
"""
import pathlib
import subprocess
import sys
import tempfile

repo = pathlib.Path(sys.argv[1]).resolve()
WHOLE = "concatenate_whole_plan_tiles_permuted_segments_like_serial_segments"
TILES = "tile_ranges_partition_the_tiled_box_walk"
SRC = repo / "strided-basic/src/static_indexing_plan.rs"
HELP = repo / "strided-basic/src/static_indexing_plan/tests/parallel_tests.rs"
TILED = repo / "strided-basic/src/raw_ops/tests/tiled_tests.rs"
RAW = repo / "strided-basic/src/raw_ops.rs"
FEATURES = {WHOLE: ["--features", "parallel"], TILES: []}


class Unexpected(Exception):
    pass


def cargo_test(test, cwd):
    cmd = ["cargo", "test", "-p", "strided-basic", *FEATURES[test], "--lib", test]
    print("$ " + " ".join(cmd), flush=True)
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return out.returncode, out.stdout, out.stderr


def show(rc, stdout, stderr):
    print(f"returncode: {rc}")
    for line in stdout.splitlines():
        if line.startswith(("test ", "test result")) or "panicked at" in line \
                or line.strip().startswith(("left:", "right:")) or " must " in line:
            print("stdout| " + line.strip())
    for line in stderr.splitlines():
        if line.startswith(("error", "warning: unused")):
            print("stderr| " + line.strip())


def expect_pass(label, test, cwd=repo):
    print(f"--- {label}", flush=True)
    rc, stdout, stderr = cargo_test(test, cwd)
    show(rc, stdout, stderr)
    problems = []
    if rc != 0:
        problems.append(f"return code {rc}, expected 0")
    if "test result: ok. 1 passed; 0 failed" not in stdout:
        problems.append("no `test result: ok. 1 passed; 0 failed` in stdout")
    check(label, problems, stderr)


def expect_fail(label, test, message, cwd=repo):
    print(f"--- {label}", flush=True)
    rc, stdout, stderr = cargo_test(test, cwd)
    show(rc, stdout, stderr)
    problems = []
    if rc != 101:
        problems.append(f"return code {rc}, expected 101 (a failing test)")
    if "error[E" in stderr or "could not compile" in stderr:
        problems.append("compiler error in stderr: the mutant did not build")
    if "error: test failed" not in stderr:
        problems.append("no `error: test failed` in stderr")
    if "test result: FAILED. 0 passed; 1 failed" not in stdout:
        problems.append("no `test result: FAILED. 0 passed; 1 failed` in stdout")
    if message not in stdout:
        problems.append(f"expected panic message not in stdout: {message!r}")
    check(label, problems, stderr)


def check(label, problems, stderr):
    if problems:
        print(f"UNEXPECTED ({label}):")
        for problem in problems:
            print("  - " + problem)
        print("stderr tail:")
        for line in stderr.splitlines()[-15:]:
            print("  " + line)
        raise Unexpected(label)
    print("as expected\n", flush=True)


def mutated(path, pairs, run):
    original = path.read_text()
    text = original
    for old, new in pairs:
        if text.count(old) != 1:
            raise Unexpected(f"mutation anchor not found exactly once in {path.name}: {old[:60]!r}")
        text = text.replace(old, new)
    path.write_text(text)
    try:
        run()
    finally:
        path.write_text(original)


def main():
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    expect_pass(f"head {head}, unmutated: {WHOLE}", WHOLE)
    expect_pass(f"head {head}, unmutated: {TILES}", TILES)

    walk_tiles = "execute: the whole-plan replay must walk the serial path's tiles"
    dispatch = ("execute: the whole-plan replay must dispatch the tiled segments across "
                "the budget (dispatched 0 chunks)")

    # r1: the tiled walk replaced by the flat walk of the same element range.
    s = SRC.read_text()
    a = s.index("                    crate::raw_ops::apply_tile_range(")
    b = s.index("                    );\n", a) + len("                    );\n")
    mutated(SRC, [(s[a:b], "                    let _ = (first, end);\n"
                           "                    crate::raw_ops::apply_fused_range(dest_ptr.as_ptr(), "
                           "dest_offsets[position], input.data().as_ptr(), input.offset(), layout, "
                           "local_start, local_end - local_start, &apply);\n")],
            lambda: expect_fail("mutant r1: tiled branch -> apply_fused_range over the element range",
                                WHOLE, walk_tiles))

    # r2: the tiled dispatch serialized (its closure called once over 0..width).
    s = SRC.read_text()
    opening = "crate::threading::parallel_for_each(0..width, exec, &|slots| {"
    i = s.index(opening)
    closing = "            });\n        } else {"
    j = s.index(closing, i)
    mutated(SRC, [(s[i:j + len(closing)],
                   "let _ = exec; (|slots: core::ops::Range<usize>| {" + s[i + len(opening):j]
                   + "            })(0..width);\n        } else {")],
            lambda: expect_fail("mutant r2: tiled dispatch serialized", WHOLE, dispatch))

    # r3: only `execute` runs serially in the test helper.
    mutated(HELP, [("    plan.execute(&mut dest, inputs, exec.reborrow()).unwrap();",
                    "    let _ = exec.reborrow();\n"
                    "    plan.execute(&mut dest, inputs, Exec::serial()).unwrap();")],
            lambda: expect_fail("mutant r3: execute serial, execute_uninit parallel", WHOLE, dispatch))

    # r5: every tile of a segment goes to the range that starts it.
    mutated(SRC, [("let first = tile_share(local_start, count, segment_len);",
                   "let first = if local_start == 0 { 0 } else { count };"),
                  ("let end = tile_share(local_end, count, segment_len);",
                   "let end = count; let _ = local_end;")],
            lambda: expect_fail("mutant r5: all tiles of a segment in its first range", WHOLE,
                                "execute: the tiles of segment 0 must be walked by at least two dispatched ranges (walked by 1)"))

    # r9: each tiled segment kept whole in one range, the two segments in two
    # different ranges (segment 0 in the first range, segment 2 in the last),
    # so that two ranges walk tiles in total but neither segment is split.
    mutated(SRC, [("let first = tile_share(local_start, count, segment_len);",
                   "let owner = if position % 4 == 0 { local_start == 0 } else { local_end == segment_len };\n"
                   "                    let first = if owner { 0 } else { count };"),
                  ("let end = tile_share(local_end, count, segment_len);",
                   "let end = count;")],
            lambda: expect_fail("mutant r9: each tiled segment whole in one range, segments in different ranges",
                                WHOLE,
                                "execute: the tiles of segment 0 must be walked by at least two dispatched ranges (walked by 1)"))

    # r6: the two ranges of each cut walked second range first.
    mutated(TILED, [("for (first, len) in [(0, cut), (cut, count - cut)] {",
                     "for (first, len) in [(cut, count - cut), (0, cut)] {")],
            lambda: expect_fail("mutant r6: the two ranges of each cut walked second range first", TILES,
                                "cut at 1 must visit the slots in the whole-box order"))

    # r7: the tile grid numbered and advanced last axis fastest, consistently in
    # both the start decoding and the odometer of `apply_tile_range`, so that the
    # whole-box and split walks still agree with each other.
    mutated(RAW, [("    for axis in 0..rank {\n        let along = tile_box.dims[axis].div_ceil(blocks[axis]);",
                   "    for axis in (0..rank).rev() {\n        let along = tile_box.dims[axis].div_ceil(blocks[axis]);"),
                  ("        let mut axis = 0;\n        loop {\n            debug_assert!(axis < rank, \"tile range past the end of the grid\");",
                   "        let mut axis = rank - 1;\n        loop {\n            debug_assert!(axis < rank, \"tile range past the end of the grid\");"),
                  ("            origin[axis] = 0;\n            tile.dims[axis] = blocks[axis].min(tile_box.dims[axis]);\n            axis += 1;\n        }\n    }\n}\n\n/// Replay every logical index of `box_dims`",
                   "            origin[axis] = 0;\n            tile.dims[axis] = blocks[axis].min(tile_box.dims[axis]);\n            axis -= 1;\n        }\n    }\n}\n\n/// Replay every logical index of `box_dims`")],
            lambda: expect_fail("mutant r7: tile grid last axis fastest in apply_tile_range", TILES,
                                "the whole-box walk must visit the slots in the specified tile order"))

    # Red first: the commit that added the test, before the fix, found by its
    # subject so that the script survives a rebase.
    red = subprocess.run(["git", "log", "--format=%h", "--fixed-strings",
                          "--grep=test: whole-plan concatenate must tile permuted segments like the serial segments",
                          "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.split()
    if len(red) != 1:
        raise Unexpected(f"red-first commit not unique: {red}")
    with tempfile.TemporaryDirectory() as tmp:
        wt = pathlib.Path(tmp) / "wt"
        subprocess.run(["git", "worktree", "add", "-q", str(wt), red[0]], cwd=repo, check=True)
        try:
            expect_fail(f"red first: worktree at {red[0]} (test added, fix absent)", WHOLE, walk_tiles, cwd=wt)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo, check=True)

    print("all expectations held: 2 unmutated passes, 7 mutants and red first failing in their assertions")


if __name__ == "__main__":
    try:
        main()
    except Unexpected as stop:
        print(f"STOPPED: {stop}")
        sys.exit(1)
```

## Appendix D: the script stops on a build failure

A syntax error in the tree is rejected at the unmutated run, and the tree is then restored:

```
$ printf '
fn broken( {
' >> strided-basic/src/static_indexing_plan.rs   # syntax error
$ python3 mutants.py ~/work/strided-rs
--- head 0309a97, unmutated: concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stderr| error: this file contains an unclosed delimiter
stderr| error: could not compile `strided-basic` (lib test) due to 1 previous error
UNEXPECTED (head 0309a97, unmutated: concatenate_whole_plan_tiles_permuted_segments_like_serial_segments):
  - return code 101, expected 0
  - no `test result: ok. 1 passed; 0 failed` in stdout
stderr tail:
     Compiling strided-basic v0.14.0 (/Users/hiroshi/work/strided-rs/strided-basic)
  error: this file contains an unclosed delimiter
      --> strided-basic/src/static_indexing_plan.rs:2583:14
       |
  2583 | fn broken( {
       |          - -^
       |          | |
       |          | unclosed delimiter
       |          unclosed delimiter
  
  error: could not compile `strided-basic` (lib test) due to 1 previous error
STOPPED: head 0309a97, unmutated: concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
exit: 1
$ git checkout -- strided-basic/src/static_indexing_plan.rs   # restored
```

A copy of the script whose r1 mutant does not compile (`fn broken( {` added to the replacement) stops at r1. It does not report the mutant as caught:

```
$ python3 mutants_with_r1_not_compiling.py ~/work/strided-rs   # copy whose r1 mutant has a syntax error
--- head 0309a97, unmutated: concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 0
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... ok
stdout| test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.32s
as expected
--- head 0309a97, unmutated: tile_ranges_partition_the_tiled_box_walk
$ cargo test -p strided-basic --lib tile_ranges_partition_the_tiled_box_walk
returncode: 0
stdout| test raw_ops::tiled_tests::tile_ranges_partition_the_tiled_box_walk ... ok
stdout| test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 261 filtered out; finished in 0.08s
as expected
--- mutant r1: tiled branch -> apply_fused_range over the element range
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stderr| error: mismatched closing delimiter: `}`
stderr| error: this file contains an unclosed delimiter
stderr| error: could not compile `strided-basic` (lib test) due to 2 previous errors
UNEXPECTED (mutant r1: tiled branch -> apply_fused_range over the element range):
  - compiler error in stderr: the mutant did not build
  - no `error: test failed` in stderr
  - no `test result: FAILED. 0 passed; 1 failed` in stdout
  - expected panic message not in stdout: "execute: the whole-plan replay must walk the serial path's tiles"
stderr tail:
  1328 | impl ConcatenatePlan {
       |                      - unclosed delimiter
  ...
  1566 |     {
       |     - unclosed delimiter
  ...
  1653 |  fn broken( {{
       |              - this delimiter might not be properly closed...
  1654 |                 } else {
       |                 - ...as it matches this but it has different indentation
  ...
  2573 | mod parallel_tests;
       |                    ^
  error: could not compile `strided-basic` (lib test) due to 2 previous errors
STOPPED: mutant r1: tiled branch -> apply_fused_range over the element range
exit: 1
```

## Appendix E: the r9 counterexample on the previous and the current test

The same mutation (segment 0 whole in the first range, segment 2 whole in the last), applied to `f62d82e` in a temporary worktree and to `0309a97`:

```
--- counterexample on the previous test (f62d82e: tiled ranges summed over segments)
$ git diff --stat; git diff -U0 strided-basic/src/static_indexing_plan.rs
diff --git a/strided-basic/src/static_indexing_plan.rs b/strided-basic/src/static_indexing_plan.rs
index 002e51f..3604c0d 100644
--- a/strided-basic/src/static_indexing_plan.rs
+++ b/strided-basic/src/static_indexing_plan.rs
@@ -1645,2 +1645,3 @@ impl ConcatenatePlan {
-                    let first = tile_share(local_start, count, segment_len);
-                    let end = tile_share(local_end, count, segment_len);
+                    let owner = if position % 4 == 0 { local_start == 0 } else { local_end == segment_len };
+                    let first = if owner { 0 } else { count };
+                    let end = count;
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 0
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... ok
stdout| test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.32s

--- counterexample on the new test (0309a97: tiled ranges per segment)
$ git diff --stat; git diff -U0 strided-basic/src/static_indexing_plan.rs
diff --git a/strided-basic/src/static_indexing_plan.rs b/strided-basic/src/static_indexing_plan.rs
index b793f7b..7736ce6 100644
--- a/strided-basic/src/static_indexing_plan.rs
+++ b/strided-basic/src/static_indexing_plan.rs
@@ -1649,2 +1649,3 @@ impl ConcatenatePlan {
-                    let first = tile_share(local_start, count, segment_len);
-                    let end = tile_share(local_end, count, segment_len);
+                    let owner = if position % 4 == 0 { local_start == 0 } else { local_end == segment_len };
+                    let first = if owner { 0 } else { count };
+                    let end = count;
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
returncode: 101
stdout| test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
stdout| thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (147408893) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:677:17:
stdout| execute: the tiles of segment 0 must be walked by at least two dispatched ranges (walked by 1)
stdout| test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 272 filtered out; finished in 0.17s
stderr| error: test failed, to rerun pass `-p strided-basic --lib`
```
