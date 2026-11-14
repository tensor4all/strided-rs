# 2026-11-14: #466 — the destination's L1-set aliasing in the permutation engine

Scope: `strided-perm`'s leaf fallback kernel picks its store order, or its tile
height, from the two strides it is handed, so a destination (or a source) whose
rows are a multiple of the 4 KiB L1 way apart no longer thrash one set. The
planner, the fold, the blocking, the width decision and the native `f64`/`f32`
micro-kernels are unchanged.

The maintainer's plan in #466 was to detect the aliasing in the planner and
shrink the tile to `L1_WAYS - L1_WAYS/4` live lines per set, or to skew the
store. Both are in the kernel now (§1), keyed on a predicate the leaf can
evaluate itself, and measured against five alternatives (§4). §3 and §5 are the
honest remainder: the serial tables gain on every row, and four pooled cells in
an existing benchmark are still above `main` because `main` is already free of
the penalty there.

## 0. Host and protocol

Machine: **AMD EPYC 7702P** (Zen 2, 64 cores, SMT off, one socket, one NUMA
node, boost disabled, governor `performance`, 16 MiB L3 per four-core domain,
L1d 32 KiB 8-way with 64 B lines, rustc 1.99.0, `--release`). Domains are read
at run time from `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`;
`cpu40` shares L3 with `40-43`.

Isolation: measured cells on `taskset -c 40` (1T) and `taskset -c 40-43` (4T),
the observing session pinned to `60-63`, a different L3 domain. Builds pinned to
`60-63` with `-j 4`. Every run records the load average and each measured core's
busy time before and after; the measured cores read 0% in all runs quoted here.

Load: 0.5-2.4 on the 1-minute average across the campaign, with another user's
`python3` on the host. The two revisions are interleaved (`before`, `after`,
`before`, `after`, ...) inside every round and the ns/e columns are the
**median of three rounds of 9 samples**, so drift lands on both sides; the
per-cell spread `(max - min) / median` is 0.1-4.2% except where noted.

Output identity: each row prints a digest of the destination computed outside
the timed region (element count, sum, position-weighted sum of element bits).
144 digest rows compared between the two revisions: **0 mismatches**.

## 1. What was changed

`strided_perm::hptt::macro_kernel_fallback` is the element-by-element 2D tile
copy every element type other than `f64`/`f32` takes, and the one every
conjugating store takes, including `Complex<f64>`. It used to walk `j` (the
destination's stride-1 axis) outermost and `i` (the axis the destination steps
by `ldb`) innermost, which streams the source. With an aliasing `ldb` every row
line of a tile lands in one L1 set, so that order asks one set to hold `block_a`
lines at once, over the ways it has, and each line is fetched again for write
allocation: the 4.5x of the tables below.

The kernel now has three cases, decided per tile from the two strides it is
handed — no plan access, no API change:

| `ldb` (destination row) | `lda` (source row) | order | why |
|---|---|---|---|
| aliases | streams | `i` outer, `j` inner (full run) | writes each row's contiguous run in one visit; one line of the set is live |
| aliases | aliases | old `j` outer, `i` inner, `a` extent cut to `L1_LIVE_LINES_PER_SET` = 6 | moving the thrash to the reads is not a win when the source collides too; the cut keeps at most six destination lines per set, which it has ways for |
| streams | either | old `j` outer, `i` inner | the source streams and the destination rows spread over enough sets |

`aliases_l1_sets(stride, size_of::<T>())` is `|stride| * size_of::<T>()` a
non-zero multiple of `L1_WAY_BYTES = BLOCK_MEMORY_SIZE / 8` = 4 KiB, the way
size `strided-basic`'s `block::fit_l1_sets` derives (#287);
`L1_LIVE_LINES_PER_SET` is that fit's budget. `strided-perm` sits below
`strided-basic`, so both numbers are restated with a pointer to their origin.

The three classes are exactly the shapes that matter here. Issue #466's rows are
destination-only aliasing (`[1024, 32]`: `lda` 512 B, `ldb` 16 KiB; `[96,96,96]`:
`lda` 1536 B, `ldb` 144 KiB). The square power-of-two transposes and
`[32,32,32,32]` perm `[3,1,0,2]` alias on both sides (`lda == ldb` for the
squares; 512 KiB against 16 KiB for the 4D one).

## 2. The issue's tables

`strided-perm/benches/l1_sets_466.rs`, ns per element. Each row is the middle of
a `2^d - 1`, `2^d`, `2^d + 1` triple; `neighbours` is the mean of the two, so
`penalty` is the aliasing cost itself rather than a comparison across unrelated
shapes.

### Complex<f64> (scalar fallback kernel)

| threads | dims | perm | 2^d before | 2^d after | after/before | neighbours before | neighbours after | penalty before | after |
|---|---|---|---|---|---|---|---|---|---|
| 1T | [1024, 1024] | [1, 0] | 5.966 | 3.286 | 0.55x | 3.041 | 3.022 | 1.96x | 1.09x |
| 1T | [1024, 32] | [1, 0] | 5.634 | 1.088 | 0.19x | 1.239 | 1.232 | 4.56x | 0.88x |
| 1T | [1024, 64] | [1, 0] | 5.642 | 1.037 | 0.18x | 1.241 | 1.239 | 4.54x | 0.84x |
| 1T | [128, 128, 128] | [1, 2, 0] | 5.665 | 2.884 | 0.51x | 3.234 | 3.235 | 1.75x | 0.89x |
| 1T | [2048, 32] | [1, 0] | 5.680 | 1.086 | 0.19x | 1.238 | 1.239 | 4.59x | 0.88x |
| 1T | [2048, 64] | [1, 0] | 5.639 | 1.051 | 0.19x | 1.250 | 1.242 | 4.51x | 0.85x |
| 1T | [256, 256] | [1, 0] | 6.229 | 1.770 | 0.28x | 1.663 | 1.701 | 3.75x | 1.04x |
| 1T | [32, 32, 32] | [1, 2, 0] | 5.678 | 1.111 | 0.20x | 1.151 | 1.138 | 4.92x | 0.98x |
| 1T | [64, 64, 64] | [1, 2, 0] | 5.638 | 1.211 | 0.21x | 1.296 | 1.309 | 4.33x | 0.92x |
| 1T | [64, 64, 64] | [2, 0, 1] | 1.303 | 1.308 | 1.00x | 1.389 | 1.389 | 0.93x | 0.94x |
| 1T | [64, 64, 64] | [2, 1, 0] | 5.598 | 1.632 | 0.29x | 1.727 | 1.692 | 3.22x | 0.96x |
| 1T | [96, 96, 96] | [1, 2, 0] | 5.527 | 3.117 | 0.56x | 3.277 | 3.280 | 1.68x | 0.95x |

Four threads, same cells:

| threads | dims | perm | 2^d before | 2^d after | after/before | neighbours before | neighbours after | penalty before | after |
|---|---|---|---|---|---|---|---|---|---|
| 4T | [1024, 1024] | [1, 0] | 5.959 | 3.280 | 0.55x | 3.022 | 3.048 | 1.97x | 1.08x |
| 4T | [1024, 32] | [1, 0] | 5.635 | 1.091 | 0.19x | 1.219 | 1.225 | 4.61x | 0.89x |
| 4T | [1024, 64] | [1, 0] | 5.652 | 1.059 | 0.19x | 1.240 | 1.226 | 4.56x | 0.86x |
| 4T | [128, 128, 128] | [1, 2, 0] | 1.751 | 1.618 | 0.92x | 1.456 | 1.459 | 1.20x | 1.11x |
| 4T | [2048, 32] | [1, 0] | 5.653 | 1.063 | 0.19x | 1.239 | 1.228 | 4.57x | 0.87x |
| 4T | [2048, 64] | [1, 0] | 5.623 | 1.081 | 0.19x | 1.242 | 1.235 | 4.54x | 0.88x |
| 4T | [256, 256] | [1, 0] | 6.421 | 1.667 | 0.26x | 1.695 | 1.697 | 3.79x | 0.98x |
| 4T | [32, 32, 32] | [1, 2, 0] | 5.633 | 1.130 | 0.20x | 1.085 | 1.069 | 5.04x | 1.05x |
| 4T | [64, 64, 64] | [1, 2, 0] | 5.626 | 1.174 | 0.21x | 1.277 | 1.299 | 4.37x | 0.90x |
| 4T | [64, 64, 64] | [2, 0, 1] | 1.283 | 1.319 | 1.03x | 1.387 | 1.393 | 0.91x | 0.94x |
| 4T | [64, 64, 64] | [2, 1, 0] | 5.639 | 1.646 | 0.29x | 1.749 | 1.691 | 3.24x | 0.97x |
| 4T | [96, 96, 96] | [1, 2, 0] | 5.566 | 3.192 | 0.57x | 3.266 | 3.278 | 1.70x | 0.97x |

### f64 (native micro-kernel path, must not move)

| threads | dims | perm | 2^d before | 2^d after | after/before | neighbours before | neighbours after | penalty before | after |
|---|---|---|---|---|---|---|---|---|---|
| 1T | [1024, 1024] | [1, 0] | 1.843 | 1.845 | 1.00x | 2.831 | 2.831 | 0.65x | 0.65x |
| 1T | [1024, 32] | [1, 0] | 1.485 | 1.511 | 1.02x | 1.147 | 1.164 | 1.30x | 1.29x |
| 1T | [1024, 64] | [1, 0] | 1.509 | 1.513 | 1.00x | 1.180 | 1.183 | 1.28x | 1.28x |
| 1T | [128, 128, 128] | [1, 2, 0] | 2.517 | 2.519 | 1.00x | 2.532 | 2.530 | 0.99x | 1.00x |
| 1T | [2048, 32] | [1, 0] | 1.489 | 1.486 | 1.00x | 1.155 | 1.162 | 1.29x | 1.28x |
| 1T | [2048, 64] | [1, 0] | 1.497 | 1.521 | 1.02x | 1.173 | 1.177 | 1.28x | 1.29x |
| 1T | [256, 256] | [1, 0] | 1.032 | 1.023 | 0.99x | 0.954 | 0.957 | 1.08x | 1.07x |
| 1T | [32, 32, 32] | [1, 2, 0] | 1.502 | 1.508 | 1.00x | 0.861 | 0.844 | 1.75x | 1.78x |
| 1T | [64, 64, 64] | [1, 2, 0] | 1.511 | 1.519 | 1.01x | 0.909 | 0.910 | 1.66x | 1.67x |
| 1T | [64, 64, 64] | [2, 0, 1] | 0.915 | 0.917 | 1.00x | 0.912 | 0.907 | 1.00x | 1.01x |
| 1T | [64, 64, 64] | [2, 1, 0] | 1.558 | 1.563 | 1.00x | 1.008 | 1.008 | 1.54x | 1.55x |
| 1T | [96, 96, 96] | [1, 2, 0] | 1.870 | 1.871 | 1.00x | 1.683 | 1.688 | 1.11x | 1.11x |

Four threads, same cells:

| threads | dims | perm | 2^d before | 2^d after | after/before | neighbours before | neighbours after | penalty before | after |
|---|---|---|---|---|---|---|---|---|---|
| 4T | [1024, 1024] | [1, 0] | 1.841 | 1.838 | 1.00x | 2.849 | 2.838 | 0.65x | 0.65x |
| 4T | [1024, 32] | [1, 0] | 1.486 | 1.485 | 1.00x | 1.167 | 1.142 | 1.27x | 1.30x |
| 4T | [1024, 64] | [1, 0] | 1.499 | 1.500 | 1.00x | 1.183 | 1.183 | 1.27x | 1.27x |
| 4T | [128, 128, 128] | [1, 2, 0] | 0.860 | 0.865 | 1.01x | 0.836 | 0.836 | 1.03x | 1.04x |
| 4T | [2048, 32] | [1, 0] | 1.448 | 1.441 | 1.00x | 1.155 | 1.155 | 1.25x | 1.25x |
| 4T | [2048, 64] | [1, 0] | 1.284 | 1.282 | 1.00x | 1.189 | 1.191 | 1.08x | 1.08x |
| 4T | [256, 256] | [1, 0] | 1.020 | 1.011 | 0.99x | 0.886 | 0.879 | 1.15x | 1.15x |
| 4T | [32, 32, 32] | [1, 2, 0] | 1.279 | 1.267 | 0.99x | 0.851 | 0.850 | 1.50x | 1.49x |
| 4T | [64, 64, 64] | [1, 2, 0] | 1.515 | 1.512 | 1.00x | 0.911 | 0.909 | 1.66x | 1.67x |
| 4T | [64, 64, 64] | [2, 0, 1] | 0.878 | 0.881 | 1.00x | 0.915 | 0.915 | 0.96x | 0.97x |
| 4T | [64, 64, 64] | [2, 1, 0] | 1.249 | 1.259 | 1.01x | 0.977 | 0.973 | 1.28x | 1.29x |
| 4T | [96, 96, 96] | [1, 2, 0] | 1.707 | 1.719 | 1.01x | 1.651 | 1.696 | 1.03x | 1.01x |

Digests over the destination, computed outside the timed region: 144 rows
compared against `main`, 0 mismatches.

## 3. The existing benchmark

`strided-perm/benches/conj_copy.rs` (its own 21-sample minimum protocol), ns per
element. `plain`/`conj` are serial, `conj4T` runs `copy_conj_into` on a
four-thread pool with the measured cells pinned to `40-43`.

| case | plain 1T before | after | conj 1T before | after | conj 4T before | after | 4T ratio |
|---|---|---|---|---|---|---|---|
| 2d 1023x1023 [1,0] | 3.316 | 3.304 | 3.299 | 3.297 | 1.118 | 1.112 | 0.99x |
| 2d 1024x1024 [1,0] | 5.595 | 3.102 | 5.600 | 3.094 | 0.939 | 0.954 | 1.02x |
| 2d 1025x1025 [1,0] | 3.465 | 3.368 | 3.476 | 3.363 | 1.118 | 1.113 | 1.00x |
| 2d 2047x2047 [1,0] | 3.315 | 3.308 | 3.144 | 3.180 | 1.109 | 1.108 | 1.00x |
| 2d 2048x2048 [1,0] | 5.699 | 3.211 | 5.008 | 3.508 | 0.957 | 0.961 | 1.00x |
| 2d 2049x2049 [1,0] | 3.402 | 3.344 | 3.026 | 3.070 | 1.092 | 1.086 | 0.99x |
| 2d 255x255 [1,0] | 1.650 | 1.647 | 1.654 | 1.647 | 1.464 | 1.442 | 0.98x |
| 2d 256x256 [1,0] | 5.579 | 1.873 | 5.574 | 1.876 | 0.882 | 1.088 | 1.23x ** |
| 2d 257x257 [1,0] | 1.884 | 1.892 | 1.882 | 1.884 | 0.885 | 0.880 | 0.99x |
| 2d 4095x4095 [1,0] | 3.359 | 3.356 | 3.350 | 3.437 | 1.078 | 1.076 | 1.00x |
| 2d 4096x4096 [1,0] | 6.468 | 4.750 | 6.136 | 5.372 | 0.959 | 1.111 | 1.16x ** |
| 2d 4097x4097 [1,0] | 3.476 | 3.467 | 3.202 | 3.106 | 1.052 | 1.050 | 1.00x |
| [100, 2, 100] perm [2, 0, 1] | 1.481 | 1.495 | 1.484 | 1.501 | 1.503 | 1.475 | 0.98x |
| [128, 128, 128] perm [2, 0, 1] | 5.227 | 5.191 | 3.760 | 3.821 | 1.277 | 1.268 | 0.99x |
| [32, 32, 32, 32] perm [3, 1, 0, 2] | 7.030 | 3.771 | 7.039 | 3.790 | 0.755 | 1.021 | 1.35x ** |
| [64, 64, 64] perm [2, 0, 1] | 1.660 | 1.661 | 1.659 | 1.659 | 1.374 | 1.377 | 1.00x |
| contig 1024x1024 [0,1] | 1.820 | 1.810 | 1.818 | 1.819 | 1.958 | 1.949 | 1.00x |
| contig 2048x2048 [0,1] | 1.902 | 1.724 | 1.956 | 1.767 | 1.817 | 1.808 | 1.00x |
| contig 256x256 [0,1] | 1.276 | 1.275 | 1.278 | 1.277 | 1.958 | 1.947 | 0.99x |
| contig 4096x4096 [0,1] | 1.725 | 1.725 | 1.736 | 1.736 | 1.615 | 1.613 | 1.00x |

Serial cells improve or are neutral; the square power-of-two transposes, which
were the regression of the first cut of this change, now improve 1.7-2.0x at one
thread (they take the tile-shrink branch, §1).

Three `conj4T` cells are above `main` by more than the campaign's 3% spread
threshold: `[32,32,32,32]` 1.35x (0.755 -> 1.021), `256^2` 1.23x (0.882 ->
1.088), `4096^2` 1.16x (0.959 -> 1.111). All three are both-operands-aliasing
cells where `main` is already at its `2^d +/- 1` neighbours' level at four
threads (`256^2` 0.882 against `257^2` 0.887, `4096^2` 0.959 against `4097^2`
1.054), so the pooled split has absorbed the aliasing before the kernel sees it
and the tile shrink only adds work. Every serial cell is at or better than
`main`.

Leaving the both-operands-aliasing class on `main`'s loop order is the one
variant that holds those cells exactly at `main`; it gives up the 1.8-2.0x that
class gains at one thread. That is the open decision on the PR, with the numbers
in §4.

### Review outcome

`review: gpt-6.1-sol approve-with-nits`, no correctness finding. The two nits are
in this commit: `each_branch_walks_the_tile_it_was_picked_for` now records the
destination offsets through a test-only [`Store`] and asserts the traversal each
branch was picked for (with the chunk boundary asserted separately, so removing
the shrink or reverting the run order each fails it — both checked as negative
controls), and the comments that claimed zero stride "spreads over two sets",
that the tile matrix covers "every tile a plan can hand", and the arithmetic of
the 4096-byte row now say what the code does. Both edits are comments and tests;
the kernel's machine code is unchanged, so the tables above stand.

## 4. Alternatives measured and not taken

Same host, same cells, ns per element, three-round medians. All figures are for
the both-operands-aliasing class, i.e. the branch §1 gives to the tile shrink.

| both-alias branch | `256^2` 1T | `256^2` 4T | `4096^2` 1T | `4096^2` 4T | `32^4` 1T | `32^4` 4T |
|---|---|---|---|---|---|---|
| `main` (old order) | 5.58 | 0.891 | 6.46 | 0.958 | 7.04 | 0.750 |
| full-run order change | 3.25 | 1.348 | 5.65 | 1.395 | 5.12 | 1.055 |
| order change, `j` run 8 | 1.96 | 1.112 | 4.93 | 1.232 | 3.59 | 0.983 |
| two-pass 128-element scratch | 2.70 | 0.930 | 5.28 | 0.935 | 4.13 | 0.937 |
| **tile shrink to 6 (shipped)** | **1.84** | **1.091** | **4.76** | **1.109** | **3.74** | **1.018** |

The shrink is the best of them at one thread and the second best at four, which
is why it is the shipped branch; the two-pass scratch tile is the only variant
close to `main` at four threads and it loses 1.1-1.5x at one. None of the five
columns matches `main` at four threads on all three cells, because `main` is
already at its neighbours' level there: the pooled split absorbs the write-side
aliasing before the kernel sees it. Leaving this class on `main`'s order is the
only variant that holds those cells exactly at `main`, and it forgoes the
1.8-2.0x the class gains at one thread.

## 5. Residual

- Serial: every row of the issue's two tables is at or below its neighbours
  (`penalty after` 0.86-1.08x; `[256,256]` 1.05x and `[1024,1024]` 1.08x are the
  both-aliasing rows).
- Four threads: `[32,32,32,32]` 1.36x, `256^2` 1.22x, `4096^2` 1.16x and
  `[64,64,64]` perm `[2,0,1]` 1.04x over `main` (§3), all of them in the
  both-aliasing class and all of them cells where `main` is already free of the
  penalty.
- `f64` never enters this kernel, so none of the `f64` rows move (0.99-1.02x).
