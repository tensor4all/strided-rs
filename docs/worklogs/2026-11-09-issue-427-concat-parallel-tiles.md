# 2026-11-09: #427 — whole-plan concatenate tiles its permuted segments

Raw evidence for tensor4all/strided-rs#432. It covers the red-first and mutation runs of the regression test, and the timings of `strided-basic/benches/concat_parallel.rs` against the base `03a291b`. Section 1 interprets the timings; the appendices hold the unedited output.

Machine: **Apple M5 Max** (18 logical CPUs, macOS 26.7.2), rustc 1.96.0, `cargo bench` profile, `--features parallel`. This is a shared interactive machine; nothing was pinned or isolated. The load average at the start of every session is recorded in Appendix A, and it was 5.0–6.2 throughout. The benchmark host evo2 was not used.

## 1. Timings

Command per session: `CONCAT_BENCH_REPS=201 <binary>`. Each session runs the five cases. Within a case, every rep times the three paths in an order that rotates per rep. Each row reports the median of 201 reps after 3 warm-up reps.

The three binaries:

- `head`: built from `49ec66e`.
- `base`: built from `03a291b` with this branch's bench file and its `[[bench]]` entry copied in.
- `base-again`: the same `base` binary run a second time. This is the A/A control, which measures what the machine alone moves.

There were five rounds, and each round ran the three binaries in a rotated order. The table gives the median of the five session medians, with the min–max of the session medians:

```
case                     binary                 whole-plan 4T           per-segment 4T whole/per
4x[1024x1024] all T      base          4.057 (3.676-4.849)   1.498 (1.299-1.513)     2.708
4x[1024x1024] all T      base-again    4.406 (3.836-5.100)   1.482 (1.445-1.534)     2.973
4x[1024x1024] all T      head          1.217 (1.187-1.243)   1.281 (1.246-1.319)     0.950
4x[1024x1024] T,C,T,C    base          3.043 (2.958-3.449)   0.986 (0.974-1.064)     3.086
4x[1024x1024] T,C,T,C    base-again    3.069 (2.950-3.575)   0.985 (0.971-0.995)     3.116
4x[1024x1024] T,C,T,C    head          0.752 (0.734-0.766)   0.803 (0.795-0.830)     0.936
4x[256x256] all T        base          0.138 (0.138-0.142)   0.138 (0.138-0.142)     1.000
4x[256x256] all T        base-again    0.140 (0.138-0.140)   0.140 (0.138-0.140)     1.000
4x[256x256] all T        head          0.140 (0.139-0.141)   0.140 (0.139-0.141)     1.000
8x[512x512] all T        base          0.888 (0.825-0.914)   1.558 (1.530-1.713)     0.570
8x[512x512] all T        base-again    0.943 (0.867-0.952)   1.543 (1.519-1.571)     0.611
8x[512x512] all T        head          0.474 (0.466-0.496)   1.660 (1.645-1.704)     0.286
4x[1024x1024] all C      base          0.263 (0.253-0.269)   0.324 (0.314-0.334)     0.812
4x[1024x1024] all C      base-again    0.260 (0.254-0.265)   0.319 (0.309-0.324)     0.815
4x[1024x1024] all C      head          0.272 (0.263-0.283)   0.306 (0.288-0.317)     0.889

ratio of session medians, median over the 5 rounds (whole-plan, per-segment):
4x[1024x1024] all T      base-again/base 1.086 0.989   head/base 0.300 0.855
4x[1024x1024] T,C,T,C    base-again/base 1.009 0.999   head/base 0.247 0.814
4x[256x256] all T        base-again/base 1.014 1.014   head/base 1.014 1.014
8x[512x512] all T        base-again/base 1.062 0.990   head/base 0.534 1.065
4x[1024x1024] all C      base-again/base 0.989 0.985   head/base 1.034 0.944
```

What this establishes:

- **Permuted segments.** Head whole-plan is 0.29–0.95 of per-segment, and 0.25–0.53 of base whole-plan. The A/A control moves base whole-plan by at most 8.6% on these cases, so the gains are far outside the machine's own variation.
- **4x[256x256].** The case stays under the serial gate on every binary.
- **Contiguous only (`all C`).** Head whole-plan is 1.034 of base, with session medians of 0.263–0.283 ms against 0.253–0.269 ms. The A/A ratio on this case is 0.989, so the +3.4% is larger than what the machine alone moved here.
  - The per-segment path is the same source in both binaries (`CopyPlan` is unchanged apart from two `pub(crate)` accessors). Even so, it moves by 0.944 (head/base) on this case, and by 0.814–1.065 across the cases.
  - So differences of this size also appear between the two builds in code this change does not touch. These runs do not separate a +3.4% code effect from a build-layout effect.
  - Since `233d23a`, the contiguous route is the pre-#427 loop (`replay_flat_parallel`), and the only added work per call is the stored `may_tile` flag test. **Equal contiguous throughput is not claimed.** The measured head/base ratio is 1.034, with a session-extreme bound of 0.283/0.253 = 1.12.

An earlier set of five alternating sessions, at `233d23a` (the same non-test code as `49ec66e`), gave 1.044 for contiguous whole-plan and 0.946 for contiguous per-segment (head/base). That run is quoted, not reproduced here.

## 2. Regression test: red first and the review mutants

Appendix B was produced by the script in Appendix C, run on the committed tree `49ec66e`. Each step applies one mutation, runs the focused test, and restores the file. The "red first" step runs the test commit `80415f8`, where the test is present and the fix absent, in a temporary worktree.

| run | expected | observed |
|---|---|---|
| head, unmutated | pass | ok |
| r1: tiled branch → `apply_fused_range` over the element range | fail | `left: 0, right: 234` (tile count) |
| r2: tiled dispatch serialized (closure once over `0..width`) | fail | `dispatched 0 chunks` |
| r3: only `execute` serial, `execute_uninit` parallel | fail | `execute: ... dispatched 0 chunks` |
| r5: every tile of a segment in the range that starts it | fail | `walked by 1` |
| red first, `80415f8` | fail | `left: 0, right: 234` |

## Appendix A: raw benchmark output

```
# date: 2026-11-09 23:32:42 +0900
# host: Apple M5 Max, 18 logical CPUs, macOS 26.7.2
# rustc 1.96.0 (ac68faa20 2026-05-25)
# cargo 1.96.0 (30a34c682 2026-05-25)
# head binary: strided-basic at 49ec66e (cargo bench profile, --features parallel), sha256 d6d93da859d26572
# base binary: strided-basic at 03a291b + benches/concat_parallel.rs and Cargo.toml [[bench]] from 49ec66e, sha256 3bb82e8cffd27014
# A/A: the base binary run a second time in each round (label base-again)
# CONCAT_BENCH_REPS=201, CONCAT_BENCH_THREADS=4 (default); 5 rounds, order rotated per round
## round 1 base  load: 5.26 6.12 6.27
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.755 ms    14.1 GB/s   0.882 Gelem/s  p10..p90 3.057..6.146 ms
4x[1024x1024] all T      per-segment 4T      1.299 ms    51.6 GB/s   3.228 Gelem/s  p10..p90 1.213..1.549 ms
4x[1024x1024] all T      serial 1T           4.012 ms    16.7 GB/s   1.046 Gelem/s  p10..p90 3.900..4.197 ms
4x[1024x1024] all T      whole/per-segment = 3.659
4x[1024x1024] T,C,T,C    whole-plan 4T       3.036 ms    22.1 GB/s   1.381 Gelem/s  p10..p90 1.430..3.966 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.986 ms    68.0 GB/s   4.252 Gelem/s  p10..p90 0.801..1.169 ms
4x[1024x1024] T,C,T,C    serial 1T           2.212 ms    30.3 GB/s   1.896 Gelem/s  p10..p90 2.097..2.337 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.078
4x[256x256] all T        whole-plan 4T       0.138 ms    30.3 GB/s   1.896 Gelem/s  p10..p90 0.136..0.143 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.3 GB/s   1.894 Gelem/s  p10..p90 0.136..0.142 ms
4x[256x256] all T        serial 1T           0.138 ms    30.3 GB/s   1.896 Gelem/s  p10..p90 0.136..0.142 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.883 ms    38.0 GB/s   2.374 Gelem/s  p10..p90 0.801..0.971 ms
8x[512x512] all T        per-segment 4T      1.530 ms    21.9 GB/s   1.371 Gelem/s  p10..p90 1.458..1.669 ms
8x[512x512] all T        serial 1T           1.551 ms    21.6 GB/s   1.352 Gelem/s  p10..p90 1.490..1.681 ms
8x[512x512] all T        whole/per-segment = 0.578
4x[1024x1024] all C      whole-plan 4T       0.253 ms   264.9 GB/s  16.554 Gelem/s  p10..p90 0.228..0.277 ms
4x[1024x1024] all C      per-segment 4T      0.314 ms   213.5 GB/s  13.343 Gelem/s  p10..p90 0.275..0.372 ms
4x[1024x1024] all C      serial 1T           0.533 ms   126.0 GB/s   7.872 Gelem/s  p10..p90 0.516..0.544 ms
4x[1024x1024] all C      whole/per-segment = 0.806
## round 1 head  load: 5.00 6.06 6.25
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.187 ms    56.6 GB/s   3.535 Gelem/s  p10..p90 1.154..1.226 ms
4x[1024x1024] all T      per-segment 4T      1.246 ms    53.8 GB/s   3.365 Gelem/s  p10..p90 1.214..1.280 ms
4x[1024x1024] all T      serial 1T           3.998 ms    16.8 GB/s   1.049 Gelem/s  p10..p90 3.897..4.087 ms
4x[1024x1024] all T      whole/per-segment = 0.952
4x[1024x1024] T,C,T,C    whole-plan 4T       0.734 ms    91.5 GB/s   5.717 Gelem/s  p10..p90 0.705..0.794 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.795 ms    84.4 GB/s   5.277 Gelem/s  p10..p90 0.760..0.927 ms
4x[1024x1024] T,C,T,C    serial 1T           2.245 ms    29.9 GB/s   1.868 Gelem/s  p10..p90 2.178..2.338 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.923
4x[256x256] all T        whole-plan 4T       0.140 ms    29.9 GB/s   1.870 Gelem/s  p10..p90 0.139..0.145 ms
4x[256x256] all T        per-segment 4T      0.140 ms    29.9 GB/s   1.870 Gelem/s  p10..p90 0.139..0.144 ms
4x[256x256] all T        serial 1T           0.140 ms    29.9 GB/s   1.871 Gelem/s  p10..p90 0.139..0.144 ms
4x[256x256] all T        whole/per-segment = 1.000
8x[512x512] all T        whole-plan 4T       0.477 ms    70.3 GB/s   4.396 Gelem/s  p10..p90 0.449..0.516 ms
8x[512x512] all T        per-segment 4T      1.675 ms    20.0 GB/s   1.252 Gelem/s  p10..p90 1.597..1.750 ms
8x[512x512] all T        serial 1T           1.676 ms    20.0 GB/s   1.251 Gelem/s  p10..p90 1.610..1.750 ms
8x[512x512] all T        whole/per-segment = 0.285
4x[1024x1024] all C      whole-plan 4T       0.263 ms   255.4 GB/s  15.966 Gelem/s  p10..p90 0.247..0.278 ms
4x[1024x1024] all C      per-segment 4T      0.288 ms   233.0 GB/s  14.564 Gelem/s  p10..p90 0.275..0.317 ms
4x[1024x1024] all C      serial 1T           0.549 ms   122.1 GB/s   7.634 Gelem/s  p10..p90 0.540..0.565 ms
4x[1024x1024] all C      whole/per-segment = 0.912
## round 1 base-again  load: 5.96 6.24 6.31
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       3.836 ms    17.5 GB/s   1.094 Gelem/s  p10..p90 2.910..5.195 ms
4x[1024x1024] all T      per-segment 4T      1.534 ms    43.7 GB/s   2.733 Gelem/s  p10..p90 1.442..2.231 ms
4x[1024x1024] all T      serial 1T           4.180 ms    16.1 GB/s   1.003 Gelem/s  p10..p90 3.981..4.947 ms
4x[1024x1024] all T      whole/per-segment = 2.500
4x[1024x1024] T,C,T,C    whole-plan 4T       2.993 ms    22.4 GB/s   1.401 Gelem/s  p10..p90 2.370..3.447 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.995 ms    67.4 GB/s   4.214 Gelem/s  p10..p90 0.831..1.171 ms
4x[1024x1024] T,C,T,C    serial 1T           2.156 ms    31.1 GB/s   1.945 Gelem/s  p10..p90 2.089..2.266 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.008
4x[256x256] all T        whole-plan 4T       0.138 ms    30.5 GB/s   1.905 Gelem/s  p10..p90 0.135..0.143 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.5 GB/s   1.905 Gelem/s  p10..p90 0.135..0.143 ms
4x[256x256] all T        serial 1T           0.137 ms    30.6 GB/s   1.911 Gelem/s  p10..p90 0.135..0.142 ms
4x[256x256] all T        whole/per-segment = 1.000
8x[512x512] all T        whole-plan 4T       0.944 ms    35.6 GB/s   2.223 Gelem/s  p10..p90 0.824..1.081 ms
8x[512x512] all T        per-segment 4T      1.519 ms    22.1 GB/s   1.380 Gelem/s  p10..p90 1.467..1.592 ms
8x[512x512] all T        serial 1T           1.535 ms    21.9 GB/s   1.366 Gelem/s  p10..p90 1.477..1.621 ms
8x[512x512] all T        whole/per-segment = 0.621
4x[1024x1024] all C      whole-plan 4T       0.254 ms   264.3 GB/s  16.516 Gelem/s  p10..p90 0.220..0.297 ms
4x[1024x1024] all C      per-segment 4T      0.309 ms   217.0 GB/s  13.561 Gelem/s  p10..p90 0.269..0.361 ms
4x[1024x1024] all C      serial 1T           0.530 ms   126.6 GB/s   7.913 Gelem/s  p10..p90 0.516..0.546 ms
4x[1024x1024] all C      whole/per-segment = 0.821
## round 2 head  load: 5.96 6.24 6.31
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.206 ms    55.6 GB/s   3.477 Gelem/s  p10..p90 1.173..1.255 ms
4x[1024x1024] all T      per-segment 4T      1.265 ms    53.1 GB/s   3.316 Gelem/s  p10..p90 1.230..1.344 ms
4x[1024x1024] all T      serial 1T           4.074 ms    16.5 GB/s   1.029 Gelem/s  p10..p90 4.000..4.178 ms
4x[1024x1024] all T      whole/per-segment = 0.954
4x[1024x1024] T,C,T,C    whole-plan 4T       0.752 ms    89.2 GB/s   5.576 Gelem/s  p10..p90 0.718..0.812 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.803 ms    83.6 GB/s   5.223 Gelem/s  p10..p90 0.772..0.877 ms
4x[1024x1024] T,C,T,C    serial 1T           2.279 ms    29.4 GB/s   1.840 Gelem/s  p10..p90 2.228..2.351 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.937
4x[256x256] all T        whole-plan 4T       0.141 ms    29.8 GB/s   1.862 Gelem/s  p10..p90 0.140..0.144 ms
4x[256x256] all T        per-segment 4T      0.141 ms    29.7 GB/s   1.857 Gelem/s  p10..p90 0.140..0.145 ms
4x[256x256] all T        serial 1T           0.141 ms    29.7 GB/s   1.858 Gelem/s  p10..p90 0.140..0.144 ms
4x[256x256] all T        whole/per-segment = 0.997
8x[512x512] all T        whole-plan 4T       0.471 ms    71.3 GB/s   4.456 Gelem/s  p10..p90 0.451..0.504 ms
8x[512x512] all T        per-segment 4T      1.660 ms    20.2 GB/s   1.263 Gelem/s  p10..p90 1.598..1.740 ms
8x[512x512] all T        serial 1T           1.666 ms    20.1 GB/s   1.258 Gelem/s  p10..p90 1.597..1.745 ms
8x[512x512] all T        whole/per-segment = 0.284
4x[1024x1024] all C      whole-plan 4T       0.263 ms   255.2 GB/s  15.948 Gelem/s  p10..p90 0.249..0.280 ms
4x[1024x1024] all C      per-segment 4T      0.289 ms   232.0 GB/s  14.499 Gelem/s  p10..p90 0.276..0.318 ms
4x[1024x1024] all C      serial 1T           0.539 ms   124.5 GB/s   7.782 Gelem/s  p10..p90 0.529..0.551 ms
4x[1024x1024] all C      whole/per-segment = 0.909
## round 2 base-again  load: 6.12 6.27 6.32
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.864 ms    13.8 GB/s   0.862 Gelem/s  p10..p90 2.929..6.163 ms
4x[1024x1024] all T      per-segment 4T      1.452 ms    46.2 GB/s   2.889 Gelem/s  p10..p90 1.197..1.576 ms
4x[1024x1024] all T      serial 1T           4.018 ms    16.7 GB/s   1.044 Gelem/s  p10..p90 3.879..4.189 ms
4x[1024x1024] all T      whole/per-segment = 3.350
4x[1024x1024] T,C,T,C    whole-plan 4T       3.069 ms    21.9 GB/s   1.367 Gelem/s  p10..p90 1.448..3.724 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.971 ms    69.1 GB/s   4.320 Gelem/s  p10..p90 0.802..1.143 ms
4x[1024x1024] T,C,T,C    serial 1T           2.165 ms    31.0 GB/s   1.937 Gelem/s  p10..p90 2.088..2.284 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.161
4x[256x256] all T        whole-plan 4T       0.140 ms    29.9 GB/s   1.869 Gelem/s  p10..p90 0.136..0.148 ms
4x[256x256] all T        per-segment 4T      0.140 ms    29.9 GB/s   1.871 Gelem/s  p10..p90 0.136..0.148 ms
4x[256x256] all T        serial 1T           0.140 ms    29.9 GB/s   1.871 Gelem/s  p10..p90 0.137..0.148 ms
4x[256x256] all T        whole/per-segment = 1.001
8x[512x512] all T        whole-plan 4T       0.952 ms    35.3 GB/s   2.204 Gelem/s  p10..p90 0.822..1.269 ms
8x[512x512] all T        per-segment 4T      1.543 ms    21.7 GB/s   1.359 Gelem/s  p10..p90 1.484..1.648 ms
8x[512x512] all T        serial 1T           1.555 ms    21.6 GB/s   1.349 Gelem/s  p10..p90 1.497..1.674 ms
8x[512x512] all T        whole/per-segment = 0.617
4x[1024x1024] all C      whole-plan 4T       0.254 ms   263.9 GB/s  16.491 Gelem/s  p10..p90 0.223..0.294 ms
4x[1024x1024] all C      per-segment 4T      0.321 ms   208.9 GB/s  13.054 Gelem/s  p10..p90 0.279..0.375 ms
4x[1024x1024] all C      serial 1T           0.533 ms   125.8 GB/s   7.864 Gelem/s  p10..p90 0.517..0.548 ms
4x[1024x1024] all C      whole/per-segment = 0.792
## round 2 base  load: 6.19 6.28 6.32
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       3.676 ms    18.3 GB/s   1.141 Gelem/s  p10..p90 2.874..4.932 ms
4x[1024x1024] all T      per-segment 4T      1.513 ms    44.4 GB/s   2.773 Gelem/s  p10..p90 1.414..1.604 ms
4x[1024x1024] all T      serial 1T           4.026 ms    16.7 GB/s   1.042 Gelem/s  p10..p90 3.936..4.168 ms
4x[1024x1024] all T      whole/per-segment = 2.430
4x[1024x1024] T,C,T,C    whole-plan 4T       3.043 ms    22.1 GB/s   1.378 Gelem/s  p10..p90 1.390..3.521 ms
4x[1024x1024] T,C,T,C    per-segment 4T      1.064 ms    63.1 GB/s   3.942 Gelem/s  p10..p90 0.959..1.227 ms
4x[1024x1024] T,C,T,C    serial 1T           2.179 ms    30.8 GB/s   1.925 Gelem/s  p10..p90 2.088..2.293 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 2.860
4x[256x256] all T        whole-plan 4T       0.138 ms    30.4 GB/s   1.901 Gelem/s  p10..p90 0.133..0.143 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.4 GB/s   1.899 Gelem/s  p10..p90 0.134..0.144 ms
4x[256x256] all T        serial 1T           0.137 ms    30.5 GB/s   1.908 Gelem/s  p10..p90 0.134..0.142 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.825 ms    40.7 GB/s   2.541 Gelem/s  p10..p90 0.769..0.875 ms
8x[512x512] all T        per-segment 4T      1.540 ms    21.8 GB/s   1.362 Gelem/s  p10..p90 1.484..1.630 ms
8x[512x512] all T        serial 1T           1.556 ms    21.6 GB/s   1.348 Gelem/s  p10..p90 1.493..1.641 ms
8x[512x512] all T        whole/per-segment = 0.536
4x[1024x1024] all C      whole-plan 4T       0.263 ms   255.5 GB/s  15.971 Gelem/s  p10..p90 0.231..0.329 ms
4x[1024x1024] all C      per-segment 4T      0.334 ms   200.9 GB/s  12.556 Gelem/s  p10..p90 0.287..0.472 ms
4x[1024x1024] all C      serial 1T           0.517 ms   129.8 GB/s   8.115 Gelem/s  p10..p90 0.499..0.537 ms
4x[1024x1024] all C      whole/per-segment = 0.786
## round 3 base-again  load: 6.01 6.24 6.31
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.406 ms    15.2 GB/s   0.952 Gelem/s  p10..p90 2.883..5.923 ms
4x[1024x1024] all T      per-segment 4T      1.482 ms    45.3 GB/s   2.831 Gelem/s  p10..p90 1.262..1.615 ms
4x[1024x1024] all T      serial 1T           4.109 ms    16.3 GB/s   1.021 Gelem/s  p10..p90 3.965..4.280 ms
4x[1024x1024] all T      whole/per-segment = 2.974
4x[1024x1024] T,C,T,C    whole-plan 4T       2.950 ms    22.7 GB/s   1.422 Gelem/s  p10..p90 1.412..4.015 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.985 ms    68.1 GB/s   4.258 Gelem/s  p10..p90 0.810..1.152 ms
4x[1024x1024] T,C,T,C    serial 1T           2.197 ms    30.6 GB/s   1.909 Gelem/s  p10..p90 2.089..2.319 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 2.995
4x[256x256] all T        whole-plan 4T       0.140 ms    30.1 GB/s   1.879 Gelem/s  p10..p90 0.138..0.143 ms
4x[256x256] all T        per-segment 4T      0.140 ms    30.1 GB/s   1.879 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        serial 1T           0.139 ms    30.1 GB/s   1.880 Gelem/s  p10..p90 0.138..0.143 ms
4x[256x256] all T        whole/per-segment = 1.000
8x[512x512] all T        whole-plan 4T       0.867 ms    38.7 GB/s   2.418 Gelem/s  p10..p90 0.798..0.960 ms
8x[512x512] all T        per-segment 4T      1.522 ms    22.0 GB/s   1.378 Gelem/s  p10..p90 1.470..1.605 ms
8x[512x512] all T        serial 1T           1.535 ms    21.9 GB/s   1.366 Gelem/s  p10..p90 1.482..1.636 ms
8x[512x512] all T        whole/per-segment = 0.570
4x[1024x1024] all C      whole-plan 4T       0.260 ms   258.4 GB/s  16.153 Gelem/s  p10..p90 0.232..0.284 ms
4x[1024x1024] all C      per-segment 4T      0.318 ms   210.9 GB/s  13.181 Gelem/s  p10..p90 0.281..0.370 ms
4x[1024x1024] all C      serial 1T           0.535 ms   125.5 GB/s   7.842 Gelem/s  p10..p90 0.521..0.546 ms
4x[1024x1024] all C      whole/per-segment = 0.816
## round 3 base  load: 5.69 6.17 6.28
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       3.779 ms    17.8 GB/s   1.110 Gelem/s  p10..p90 2.788..5.219 ms
4x[1024x1024] all T      per-segment 4T      1.494 ms    44.9 GB/s   2.808 Gelem/s  p10..p90 1.407..1.648 ms
4x[1024x1024] all T      serial 1T           4.103 ms    16.4 GB/s   1.022 Gelem/s  p10..p90 3.970..4.376 ms
4x[1024x1024] all T      whole/per-segment = 2.530
4x[1024x1024] T,C,T,C    whole-plan 4T       3.415 ms    19.6 GB/s   1.228 Gelem/s  p10..p90 1.512..4.224 ms
4x[1024x1024] T,C,T,C    per-segment 4T      1.012 ms    66.3 GB/s   4.144 Gelem/s  p10..p90 0.830..1.208 ms
4x[1024x1024] T,C,T,C    serial 1T           2.290 ms    29.3 GB/s   1.832 Gelem/s  p10..p90 2.180..2.485 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.374
4x[256x256] all T        whole-plan 4T       0.139 ms    30.1 GB/s   1.882 Gelem/s  p10..p90 0.138..0.141 ms
4x[256x256] all T        per-segment 4T      0.139 ms    30.1 GB/s   1.882 Gelem/s  p10..p90 0.138..0.141 ms
4x[256x256] all T        serial 1T           0.139 ms    30.1 GB/s   1.884 Gelem/s  p10..p90 0.138..0.141 ms
4x[256x256] all T        whole/per-segment = 1.000
8x[512x512] all T        whole-plan 4T       0.914 ms    36.7 GB/s   2.293 Gelem/s  p10..p90 0.820..1.010 ms
8x[512x512] all T        per-segment 4T      1.558 ms    21.5 GB/s   1.346 Gelem/s  p10..p90 1.521..1.653 ms
8x[512x512] all T        serial 1T           1.578 ms    21.3 GB/s   1.329 Gelem/s  p10..p90 1.522..1.694 ms
8x[512x512] all T        whole/per-segment = 0.587
4x[1024x1024] all C      whole-plan 4T       0.269 ms   249.6 GB/s  15.597 Gelem/s  p10..p90 0.234..0.319 ms
4x[1024x1024] all C      per-segment 4T      0.324 ms   207.3 GB/s  12.959 Gelem/s  p10..p90 0.285..0.381 ms
4x[1024x1024] all C      serial 1T           0.536 ms   125.2 GB/s   7.825 Gelem/s  p10..p90 0.524..0.553 ms
4x[1024x1024] all C      whole/per-segment = 0.831
## round 3 head  load: 5.48 6.12 6.26
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.240 ms    54.1 GB/s   3.383 Gelem/s  p10..p90 1.207..1.307 ms
4x[1024x1024] all T      per-segment 4T      1.319 ms    50.9 GB/s   3.180 Gelem/s  p10..p90 1.267..1.530 ms
4x[1024x1024] all T      serial 1T           4.140 ms    16.2 GB/s   1.013 Gelem/s  p10..p90 4.061..4.259 ms
4x[1024x1024] all T      whole/per-segment = 0.940
4x[1024x1024] T,C,T,C    whole-plan 4T       0.766 ms    87.6 GB/s   5.474 Gelem/s  p10..p90 0.732..0.817 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.830 ms    80.9 GB/s   5.056 Gelem/s  p10..p90 0.782..0.935 ms
4x[1024x1024] T,C,T,C    serial 1T           2.306 ms    29.1 GB/s   1.819 Gelem/s  p10..p90 2.249..2.378 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.924
4x[256x256] all T        whole-plan 4T       0.139 ms    30.1 GB/s   1.883 Gelem/s  p10..p90 0.138..0.141 ms
4x[256x256] all T        per-segment 4T      0.139 ms    30.1 GB/s   1.881 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        serial 1T           0.139 ms    30.1 GB/s   1.881 Gelem/s  p10..p90 0.138..0.141 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.496 ms    67.6 GB/s   4.226 Gelem/s  p10..p90 0.456..0.558 ms
8x[512x512] all T        per-segment 4T      1.704 ms    19.7 GB/s   1.231 Gelem/s  p10..p90 1.616..1.804 ms
8x[512x512] all T        serial 1T           1.715 ms    19.6 GB/s   1.223 Gelem/s  p10..p90 1.611..1.811 ms
8x[512x512] all T        whole/per-segment = 0.291
4x[1024x1024] all C      whole-plan 4T       0.283 ms   236.7 GB/s  14.797 Gelem/s  p10..p90 0.259..0.309 ms
4x[1024x1024] all C      per-segment 4T      0.317 ms   211.8 GB/s  13.240 Gelem/s  p10..p90 0.282..0.365 ms
4x[1024x1024] all C      serial 1T           0.537 ms   125.0 GB/s   7.815 Gelem/s  p10..p90 0.525..0.550 ms
4x[1024x1024] all C      whole/per-segment = 0.895
## round 4 base  load: 5.48 6.12 6.26
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.849 ms    13.8 GB/s   0.865 Gelem/s  p10..p90 3.298..6.629 ms
4x[1024x1024] all T      per-segment 4T      1.498 ms    44.8 GB/s   2.800 Gelem/s  p10..p90 1.378..1.620 ms
4x[1024x1024] all T      serial 1T           4.265 ms    15.7 GB/s   0.983 Gelem/s  p10..p90 4.111..4.388 ms
4x[1024x1024] all T      whole/per-segment = 3.238
4x[1024x1024] T,C,T,C    whole-plan 4T       3.449 ms    19.5 GB/s   1.216 Gelem/s  p10..p90 1.468..4.332 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.978 ms    68.7 GB/s   4.291 Gelem/s  p10..p90 0.821..1.142 ms
4x[1024x1024] T,C,T,C    serial 1T           2.261 ms    29.7 GB/s   1.855 Gelem/s  p10..p90 2.172..2.353 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.528
4x[256x256] all T        whole-plan 4T       0.142 ms    29.6 GB/s   1.850 Gelem/s  p10..p90 0.139..0.158 ms
4x[256x256] all T        per-segment 4T      0.142 ms    29.6 GB/s   1.852 Gelem/s  p10..p90 0.139..0.159 ms
4x[256x256] all T        serial 1T           0.142 ms    29.6 GB/s   1.850 Gelem/s  p10..p90 0.139..0.162 ms
4x[256x256] all T        whole/per-segment = 1.001
8x[512x512] all T        whole-plan 4T       0.888 ms    37.8 GB/s   2.360 Gelem/s  p10..p90 0.784..1.122 ms
8x[512x512] all T        per-segment 4T      1.713 ms    19.6 GB/s   1.224 Gelem/s  p10..p90 1.533..2.047 ms
8x[512x512] all T        serial 1T           1.740 ms    19.3 GB/s   1.205 Gelem/s  p10..p90 1.544..2.144 ms
8x[512x512] all T        whole/per-segment = 0.519
4x[1024x1024] all C      whole-plan 4T       0.262 ms   255.9 GB/s  15.994 Gelem/s  p10..p90 0.232..0.322 ms
4x[1024x1024] all C      per-segment 4T      0.317 ms   212.0 GB/s  13.250 Gelem/s  p10..p90 0.284..0.384 ms
4x[1024x1024] all C      serial 1T           0.531 ms   126.4 GB/s   7.899 Gelem/s  p10..p90 0.521..0.542 ms
4x[1024x1024] all C      whole/per-segment = 0.828
## round 4 head  load: 5.60 6.13 6.27
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.243 ms    54.0 GB/s   3.375 Gelem/s  p10..p90 1.201..1.314 ms
4x[1024x1024] all T      per-segment 4T      1.312 ms    51.1 GB/s   3.196 Gelem/s  p10..p90 1.263..1.506 ms
4x[1024x1024] all T      serial 1T           4.143 ms    16.2 GB/s   1.012 Gelem/s  p10..p90 4.055..4.329 ms
4x[1024x1024] all T      whole/per-segment = 0.947
4x[1024x1024] T,C,T,C    whole-plan 4T       0.765 ms    87.7 GB/s   5.483 Gelem/s  p10..p90 0.738..0.810 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.814 ms    82.4 GB/s   5.152 Gelem/s  p10..p90 0.785..0.897 ms
4x[1024x1024] T,C,T,C    serial 1T           2.291 ms    29.3 GB/s   1.831 Gelem/s  p10..p90 2.244..2.358 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.940
4x[256x256] all T        whole-plan 4T       0.140 ms    30.0 GB/s   1.878 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        per-segment 4T      0.140 ms    30.0 GB/s   1.875 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        serial 1T           0.140 ms    30.1 GB/s   1.879 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.474 ms    70.9 GB/s   4.428 Gelem/s  p10..p90 0.450..0.513 ms
8x[512x512] all T        per-segment 4T      1.645 ms    20.4 GB/s   1.275 Gelem/s  p10..p90 1.592..1.735 ms
8x[512x512] all T        serial 1T           1.656 ms    20.3 GB/s   1.266 Gelem/s  p10..p90 1.591..1.740 ms
8x[512x512] all T        whole/per-segment = 0.288
4x[1024x1024] all C      whole-plan 4T       0.272 ms   246.6 GB/s  15.411 Gelem/s  p10..p90 0.260..0.293 ms
4x[1024x1024] all C      per-segment 4T      0.306 ms   219.2 GB/s  13.701 Gelem/s  p10..p90 0.282..0.354 ms
4x[1024x1024] all C      serial 1T           0.528 ms   127.1 GB/s   7.943 Gelem/s  p10..p90 0.521..0.541 ms
4x[1024x1024] all C      whole/per-segment = 0.889
## round 4 base-again  load: 5.55 6.11 6.26
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       5.100 ms    13.2 GB/s   0.822 Gelem/s  p10..p90 3.245..6.573 ms
4x[1024x1024] all T      per-segment 4T      1.445 ms    46.5 GB/s   2.903 Gelem/s  p10..p90 1.235..1.593 ms
4x[1024x1024] all T      serial 1T           4.089 ms    16.4 GB/s   1.026 Gelem/s  p10..p90 3.962..4.251 ms
4x[1024x1024] all T      whole/per-segment = 3.531
4x[1024x1024] T,C,T,C    whole-plan 4T       3.575 ms    18.8 GB/s   1.173 Gelem/s  p10..p90 1.553..4.813 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.982 ms    68.3 GB/s   4.269 Gelem/s  p10..p90 0.849..1.132 ms
4x[1024x1024] T,C,T,C    serial 1T           2.258 ms    29.7 GB/s   1.858 Gelem/s  p10..p90 2.147..2.346 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.639
4x[256x256] all T        whole-plan 4T       0.140 ms    30.0 GB/s   1.877 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        per-segment 4T      0.140 ms    30.0 GB/s   1.876 Gelem/s  p10..p90 0.138..0.143 ms
4x[256x256] all T        serial 1T           0.139 ms    30.1 GB/s   1.880 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.900 ms    37.3 GB/s   2.331 Gelem/s  p10..p90 0.805..0.988 ms
8x[512x512] all T        per-segment 4T      1.556 ms    21.6 GB/s   1.348 Gelem/s  p10..p90 1.516..1.649 ms
8x[512x512] all T        serial 1T           1.572 ms    21.3 GB/s   1.334 Gelem/s  p10..p90 1.521..1.703 ms
8x[512x512] all T        whole/per-segment = 0.578
4x[1024x1024] all C      whole-plan 4T       0.263 ms   255.3 GB/s  15.956 Gelem/s  p10..p90 0.231..0.304 ms
4x[1024x1024] all C      per-segment 4T      0.319 ms   210.1 GB/s  13.133 Gelem/s  p10..p90 0.282..0.380 ms
4x[1024x1024] all C      serial 1T           0.532 ms   126.0 GB/s   7.878 Gelem/s  p10..p90 0.522..0.544 ms
4x[1024x1024] all C      whole/per-segment = 0.823
## round 5 head  load: 5.51 6.09 6.25
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       1.217 ms    55.1 GB/s   3.447 Gelem/s  p10..p90 1.180..1.275 ms
4x[1024x1024] all T      per-segment 4T      1.281 ms    52.4 GB/s   3.275 Gelem/s  p10..p90 1.239..1.460 ms
4x[1024x1024] all T      serial 1T           4.063 ms    16.5 GB/s   1.032 Gelem/s  p10..p90 3.976..4.194 ms
4x[1024x1024] all T      whole/per-segment = 0.950
4x[1024x1024] T,C,T,C    whole-plan 4T       0.743 ms    90.4 GB/s   5.648 Gelem/s  p10..p90 0.718..0.785 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.795 ms    84.4 GB/s   5.277 Gelem/s  p10..p90 0.766..0.853 ms
4x[1024x1024] T,C,T,C    serial 1T           2.239 ms    30.0 GB/s   1.873 Gelem/s  p10..p90 2.191..2.303 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 0.934
4x[256x256] all T        whole-plan 4T       0.139 ms    30.1 GB/s   1.880 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        per-segment 4T      0.140 ms    30.1 GB/s   1.879 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        serial 1T           0.140 ms    30.1 GB/s   1.879 Gelem/s  p10..p90 0.138..0.142 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.466 ms    72.0 GB/s   4.501 Gelem/s  p10..p90 0.444..0.505 ms
8x[512x512] all T        per-segment 4T      1.646 ms    20.4 GB/s   1.274 Gelem/s  p10..p90 1.587..1.727 ms
8x[512x512] all T        serial 1T           1.651 ms    20.3 GB/s   1.270 Gelem/s  p10..p90 1.583..1.750 ms
8x[512x512] all T        whole/per-segment = 0.283
4x[1024x1024] all C      whole-plan 4T       0.275 ms   243.9 GB/s  15.243 Gelem/s  p10..p90 0.259..0.300 ms
4x[1024x1024] all C      per-segment 4T      0.316 ms   212.1 GB/s  13.254 Gelem/s  p10..p90 0.288..0.383 ms
4x[1024x1024] all C      serial 1T           0.536 ms   125.2 GB/s   7.827 Gelem/s  p10..p90 0.526..0.552 ms
4x[1024x1024] all C      whole/per-segment = 0.870
## round 5 base-again  load: 5.51 6.09 6.25
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.138 ms    16.2 GB/s   1.014 Gelem/s  p10..p90 2.881..5.532 ms
4x[1024x1024] all T      per-segment 4T      1.515 ms    44.3 GB/s   2.768 Gelem/s  p10..p90 1.290..1.634 ms
4x[1024x1024] all T      serial 1T           4.097 ms    16.4 GB/s   1.024 Gelem/s  p10..p90 3.988..4.219 ms
4x[1024x1024] all T      whole/per-segment = 2.731
4x[1024x1024] T,C,T,C    whole-plan 4T       3.093 ms    21.7 GB/s   1.356 Gelem/s  p10..p90 1.610..3.719 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.985 ms    68.2 GB/s   4.260 Gelem/s  p10..p90 0.875..1.166 ms
4x[1024x1024] T,C,T,C    serial 1T           2.219 ms    30.2 GB/s   1.890 Gelem/s  p10..p90 2.129..2.306 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.142
4x[256x256] all T        whole-plan 4T       0.138 ms    30.3 GB/s   1.893 Gelem/s  p10..p90 0.136..0.145 ms
4x[256x256] all T        per-segment 4T      0.139 ms    30.3 GB/s   1.892 Gelem/s  p10..p90 0.137..0.145 ms
4x[256x256] all T        serial 1T           0.139 ms    30.3 GB/s   1.892 Gelem/s  p10..p90 0.137..0.146 ms
4x[256x256] all T        whole/per-segment = 0.999
8x[512x512] all T        whole-plan 4T       0.943 ms    35.6 GB/s   2.224 Gelem/s  p10..p90 0.837..1.112 ms
8x[512x512] all T        per-segment 4T      1.571 ms    21.4 GB/s   1.335 Gelem/s  p10..p90 1.515..1.683 ms
8x[512x512] all T        serial 1T           1.591 ms    21.1 GB/s   1.318 Gelem/s  p10..p90 1.520..1.719 ms
8x[512x512] all T        whole/per-segment = 0.600
4x[1024x1024] all C      whole-plan 4T       0.265 ms   252.9 GB/s  15.805 Gelem/s  p10..p90 0.233..0.308 ms
4x[1024x1024] all C      per-segment 4T      0.324 ms   207.4 GB/s  12.960 Gelem/s  p10..p90 0.281..0.381 ms
4x[1024x1024] all C      serial 1T           0.537 ms   125.0 GB/s   7.815 Gelem/s  p10..p90 0.526..0.551 ms
4x[1024x1024] all C      whole/per-segment = 0.820
## round 5 base  load: 5.38 6.06 6.24
threads 4, median of 201 reps after 3 warm-up
4x[1024x1024] all T      whole-plan 4T       4.057 ms    16.5 GB/s   1.034 Gelem/s  p10..p90 2.879..5.239 ms
4x[1024x1024] all T      per-segment 4T      1.509 ms    44.5 GB/s   2.779 Gelem/s  p10..p90 1.426..1.611 ms
4x[1024x1024] all T      serial 1T           4.125 ms    16.3 GB/s   1.017 Gelem/s  p10..p90 4.004..4.261 ms
4x[1024x1024] all T      whole/per-segment = 2.688
4x[1024x1024] T,C,T,C    whole-plan 4T       2.958 ms    22.7 GB/s   1.418 Gelem/s  p10..p90 1.420..3.468 ms
4x[1024x1024] T,C,T,C    per-segment 4T      0.974 ms    68.9 GB/s   4.306 Gelem/s  p10..p90 0.817..1.158 ms
4x[1024x1024] T,C,T,C    serial 1T           2.209 ms    30.4 GB/s   1.899 Gelem/s  p10..p90 2.130..2.313 ms
4x[1024x1024] T,C,T,C    whole/per-segment = 3.036
4x[256x256] all T        whole-plan 4T       0.138 ms    30.4 GB/s   1.898 Gelem/s  p10..p90 0.136..0.144 ms
4x[256x256] all T        per-segment 4T      0.138 ms    30.4 GB/s   1.901 Gelem/s  p10..p90 0.136..0.143 ms
4x[256x256] all T        serial 1T           0.138 ms    30.5 GB/s   1.905 Gelem/s  p10..p90 0.136..0.144 ms
4x[256x256] all T        whole/per-segment = 1.001
8x[512x512] all T        whole-plan 4T       0.896 ms    37.5 GB/s   2.342 Gelem/s  p10..p90 0.811..0.978 ms
8x[512x512] all T        per-segment 4T      1.563 ms    21.5 GB/s   1.341 Gelem/s  p10..p90 1.512..1.666 ms
8x[512x512] all T        serial 1T           1.577 ms    21.3 GB/s   1.330 Gelem/s  p10..p90 1.510..1.708 ms
8x[512x512] all T        whole/per-segment = 0.573
4x[1024x1024] all C      whole-plan 4T       0.266 ms   252.6 GB/s  15.785 Gelem/s  p10..p90 0.232..0.340 ms
4x[1024x1024] all C      per-segment 4T      0.332 ms   202.3 GB/s  12.643 Gelem/s  p10..p90 0.283..0.435 ms
4x[1024x1024] all C      serial 1T           0.535 ms   125.4 GB/s   7.835 Gelem/s  p10..p90 0.522..0.550 ms
4x[1024x1024] all C      whole/per-segment = 0.801
```

## Appendix B: raw mutation output

```
--- head 49ec66e, unmutated
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 271 filtered out; finished in 0.31s

--- mutant r1: tiled branch -> apply_fused_range over the element range
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (145993374) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:644:9:
assertion `left == right` failed: execute: the whole-plan replay must walk the serial path's tiles
left: 0
right: 234
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 271 filtered out; finished in 0.17s

--- mutant r2: tiled dispatch serialized
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (145994052) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:652:9:
execute: the whole-plan replay must dispatch the tiled segments across the budget (dispatched 0 chunks)
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 271 filtered out; finished in 0.18s

--- mutant r3: execute serial, execute_uninit parallel
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (145994564) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:653:9:
execute: the whole-plan replay must dispatch the tiled segments across the budget (dispatched 0 chunks)
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 271 filtered out; finished in 0.18s

--- mutant r5: all tiles of a segment in its first range
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (145995040) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:660:9:
execute: the tiles must be walked by at least two dispatched ranges (walked by 1)
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 271 filtered out; finished in 0.18s

--- red first: worktree at 80415f8 (test added, fix absent)
$ cargo test -p strided-basic --features parallel --lib concatenate_whole_plan_tiles_permuted_segments_like_serial_segments
test static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments ... FAILED
thread 'static_indexing_plan::parallel_tests::concatenate_whole_plan_tiles_permuted_segments_like_serial_segments' (145995604) panicked at strided-basic/src/static_indexing_plan/tests/parallel_tests.rs:617:9:
assertion `left == right` failed: execute: the whole-plan replay must walk the serial path's tiles
left: 0
right: 234
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 269 filtered out; finished in 0.17s
```

## Appendix C: the mutation script

`python3 mutants.py <repo>`; it touches only the two files it mutates and restores them.

```python
#!/usr/bin/env python3
"""Rerun red-first and the four review mutants of strided-rs#432 against the
working tree; restores every file it touches. Usage: mutants.py <repo>"""
import subprocess, sys, pathlib, shutil, tempfile
repo = pathlib.Path(sys.argv[1])
T = "concatenate_whole_plan_tiles_permuted_segments_like_serial_segments"
SRC = repo / "strided-basic/src/static_indexing_plan.rs"
HELP = repo / "strided-basic/src/static_indexing_plan/tests/parallel_tests.rs"

def run(label, cwd=repo):
    print(f"--- {label}", flush=True)
    print(f"$ cargo test -p strided-basic --features parallel --lib {T}", flush=True)
    out = subprocess.run(["cargo", "test", "-p", "strided-basic", "--features", "parallel", "--lib", T],
                         cwd=cwd, capture_output=True, text=True)
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith(("test ", "test result")) or "panicked at" in line or line.strip().startswith(("left:", "right:")) \
           or "must " in line:
            print(line.strip())
    print(flush=True)

def mutate(path, pairs, label):
    orig = path.read_text()
    s = orig
    for old, new in pairs:
        assert s.count(old) == 1, (label, old)
        s = s.replace(old, new)
    path.write_text(s)
    try:
        run(label)
    finally:
        path.write_text(orig)

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
run(f"head {head}, unmutated")

# 1. first review: tiled walk replaced by the flat walk of the same element range.
tile_call_start = "                    crate::raw_ops::apply_tile_range("
s = SRC.read_text(); a = s.index(tile_call_start); b = s.index("                    );\n", a) + len("                    );\n")
mutate(SRC, [(s[a:b], "                    let _ = (first, end);\n                    crate::raw_ops::apply_fused_range(dest_ptr.as_ptr(), dest_offsets[position], input.data().as_ptr(), input.offset(), layout, local_start, local_end - local_start, &apply);\n")],
       "mutant r1: tiled branch -> apply_fused_range over the element range")

# 2. second review: tiled dispatch serialized (closure once over 0..width).
s = SRC.read_text()
open_ = "crate::threading::parallel_for_each(0..width, exec, &|slots| {"
i = s.index(open_); j = s.index("            });\n        } else {", i)
mutate(SRC, [(s[i:j + len("            });\n        } else {")],
              "let _ = exec; (|slots: core::ops::Range<usize>| {" + s[i + len(open_):j] + "            })(0..width);\n        } else {")],
       "mutant r2: tiled dispatch serialized")

# 3. third review: only `execute` runs serially in the test helper.
mutate(HELP, [("    plan.execute(&mut dest, inputs, exec.reborrow()).unwrap();",
               "    let _ = exec.reborrow();\n    plan.execute(&mut dest, inputs, Exec::serial()).unwrap();")],
       "mutant r3: execute serial, execute_uninit parallel")

# 4. fifth review: every tile of a segment goes to the range that starts it.
mutate(SRC, [("let first = tile_share(local_start, count, segment_len);", "let first = if local_start == 0 { 0 } else { count };"),
             ("let end = tile_share(local_end, count, segment_len);", "let end = count; let _ = local_end;")],
       "mutant r5: all tiles of a segment in its first range")

# red first: the test commit 80415f8 with the fix absent.
with tempfile.TemporaryDirectory() as tmp:
    wt = pathlib.Path(tmp) / "wt"
    subprocess.run(["git", "worktree", "add", "-q", str(wt), "80415f8"], cwd=repo, check=True)
    try:
        run("red first: worktree at 80415f8 (test added, fix absent)", cwd=wt)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo)
```
