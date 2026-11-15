# 2026-11-14: #466 — the destination's L1-set aliasing in the permutation engine

Scope: `strided-perm`'s leaf fallback kernel picks its store order, or its tile
height, from the two strides it is handed, so a destination (or a source) whose
rows are a multiple of the 4 KiB L1 way apart no longer thrash one set. The
planner, the fold, the blocking, the width decision and the native `f64`/`f32`
micro-kernels are unchanged.

The maintainer's plan in #466 was to detect the aliasing in the planner and
shrink the tile to `L1_WAYS - L1_WAYS/4` live lines per set, or to skew the
store. Both are in the kernel now (§1), keyed on a predicate the leaf can
evaluate itself, and measured against four rejected variants (§4). §3 is the
existing benchmark as re-measured against `main` under the corrected protocol,
and §6 records the second round: the rebase and the paired re-measurement of the
shipped kernel. The serial tables gain on every row, and nothing regresses at
four threads either — an earlier run had reported four both-aliasing pooled cells
above `main`, and that column does not reproduce (§3).

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
element, **medians of three interleaved rounds** of the two revisions (`main`
`3fb9e33`, this branch), measured cells on `taskset -c 40-43`, the observing
session on `60-63`. `plain`/`conj` are serial; `conj4T` runs `copy_conj_into`
through `Exec::new(pool, 4)`.

| case | plain 1T main | branch | conj 1T main | branch | conj 4T main | branch | 4T branch/main | elements | fans out |
|---|---|---|---|---|---|---|---|---|---|
| 2d 255x255 [1,0] | 1.201 | 1.206 | 1.669 | 1.644 | 1.668 | 1.649 | 0.99x | 65,025 | no |
| 2d 256x256 [1,0] | 6.371 | 1.702 | 5.590 | 1.877 | 5.608 | 1.877 | 0.33x | 65,536 | no |
| 2d 257x257 [1,0] | 2.175 | 2.173 | 1.909 | 1.889 | 1.905 | 1.893 | 0.99x | 66,049 | no |
| contig 256x256 [0,1] | 0.654 | 0.655 | 1.276 | 1.276 | 1.275 | 1.279 | 1.00x | 65,536 | no |
| 2d 1023x1023 [1,0] | 2.922 | 2.952 | 3.310 | 3.287 | 3.309 | 3.287 | 0.99x | 1,046,529 | no |
| 2d 1024x1024 [1,0] | 5.939 | 3.215 | 5.588 | 3.070 | 5.594 | 3.063 | 0.55x | 1,048,576 | no |
| 2d 1025x1025 [1,0] | 3.021 | 3.028 | 3.383 | 3.354 | 3.388 | 3.361 | 0.99x | 1,050,625 | no |
| contig 1024x1024 [0,1] | 0.936 | 0.938 | 1.810 | 1.820 | 1.819 | 1.825 | 1.00x | 1,048,576 | no |
| 2d 2047x2047 [1,0] | 2.992 | 2.978 | 3.314 | 3.295 | 1.318 | 1.321 | 1.00x | 4,190,209 | yes |
| 2d 2048x2048 [1,0] | 6.002 | 3.313 | 5.716 | 3.203 | 1.545 | 1.338 | 0.87x | 4,194,304 | yes |
| 2d 2049x2049 [1,0] | 3.121 | 3.082 | 3.404 | 3.344 | 1.307 | 1.315 | 1.01x | 4,198,401 | yes |
| contig 2048x2048 [0,1] | 1.088 | 1.087 | 1.915 | 1.725 | 1.052 | 1.053 | 1.00x | 4,194,304 | yes |
| 2d 4095x4095 [1,0] | 3.119 | 3.112 | 3.366 | 3.354 | 1.676 | 1.688 | 1.01x | 16,769,025 | yes |
| 2d 4096x4096 [1,0] | 6.747 | 4.257 | 6.464 | 4.765 | 1.892 | 1.857 | 0.98x | 16,777,216 | yes |
| 2d 4097x4097 [1,0] | 3.297 | 3.295 | 3.467 | 3.470 | 1.681 | 1.696 | 1.01x | 16,785,409 | yes |
| contig 4096x4096 [0,1] | 1.070 | 1.070 | 1.725 | 1.724 | 1.070 | 1.070 | 1.00x | 16,777,216 | yes |
| [64, 64, 64] perm [2, 0, 1] | 1.229 | 1.222 | 1.658 | 1.658 | 1.656 | 1.661 | 1.00x | 262,144 | no |
| [128, 128, 128] perm [2, 0, 1] | 4.093 | 4.111 | 5.232 | 5.212 | 1.259 | 1.266 | 1.01x | 2,097,152 | yes |
| [32, 32, 32, 32] perm [3, 1, 0, 2] | 9.437 | 3.706 | 7.061 | 3.801 | 7.061 | 3.776 | 0.53x | 1,048,576 | no |
| [100, 2, 100] perm [2, 0, 1] | 1.020 | 1.012 | 1.528 | 1.521 | 1.530 | 1.528 | 1.00x | 20,000 | no |

No cell is above `main` outside 1%: the five both-aliasing (tile-shrink) cells
come out at 0.33x / 0.55x / 0.87x / 0.98x / 0.53x, every `contig` row is 1.00x,
and the serial cells move only where the fallback kernel moved. Round-to-round
spread on those five cells is 0.1-1.1%.

**Which cells fan out at all.** `copy_raw` (`strided-perm/src/copy.rs:126-131`)
asks the token for a width from `n * copy_ns_per_element()`; this host is the
`x86_64-avx2` row (0.090 ns/element) and threadteam 0.6.0 charges an outside
dispatch 40 us at `PAYS_INT` 3, so nothing below `3 * 40 us / 0.090 ns` = 1.33M
elements can fan out. The measured column agrees with no exception in 20 rows:
every cell at or below 1.05M elements has `conj4T` equal to its own `conj 1T`,
every cell at 2.1M or more is 2-4x faster. For the small cells the `4T` column is
therefore the **serial** kernel measured a second time, and it moves with the
serial kernel; it is not a second parallel data point.

**This supersedes the previous round's `4T` column.** That column reported
`main` at `0.882` for `256x256` — 6.3x faster than the same cell's own `conj 1T`
(5.574) while the gate above forbids a fan-out 20x below its threshold — and at
the same time this worklog's §2 measured `[256,256]` on four threads as 6.421,
i.e. serial. Both cannot hold, and the fresh run reproduces neither: `main` is
5.608 there, which is exactly its serial kernel. The `4T` ratios derived from
that column (`1.16-1.35x` over `main`) are therefore withdrawn; the fresh paired
column above replaces them, and it shows no regression at four threads.

### Review outcome

`review: gpt-6.1-sol approve-with-nits`, no correctness finding. The two nits are
in this commit: `each_branch_walks_the_tile_it_was_picked_for` now records the
destination offsets through a test-only [`Store`] and asserts the traversal each
branch was picked for (with the chunk boundary asserted separately, so removing
the shrink or reverting the run order each fails it — both checked as negative
controls), and the comments that claimed zero stride "spreads over two sets",
that the tile matrix covers "every tile a plan can hand", and the arithmetic of
the 4096-byte row now say what the code does. Both edits are comments and tests;
the kernel's machine code is unchanged, so §2's serial tables and §3's one-thread
columns stand. (§3's four-thread column is a separate defect, corrected there;
§6 re-measures the shipped kernel against `main` end to end.)

## 4. Alternatives measured and not taken

Same host, same cells, ns per element, three-round medians. All figures are for
the both-operands-aliasing class, i.e. the branch §1 gives to the tile shrink.

**The `4T` columns of this table came from the protocol §3 retires** and are kept
only because they were the basis of the variant choice; the fresh paired `4T`
column in §3, not these, is what the PR reports. The `1T` columns are unaffected:
they are above the clock's noise floor, they do not depend on the dispatch gate,
and they are what the choice actually rests on (the shrink is the best of the
five at one thread). Read the `4T` columns as "measured before §3's correction".

| both-alias branch | `256^2` 1T | `256^2` 4T | `4096^2` 1T | `4096^2` 4T | `32^4` 1T | `32^4` 4T |
|---|---|---|---|---|---|---|
| `main` (old order) | 5.58 | 0.891 | 6.46 | 0.958 | 7.04 | 0.750 |
| full-run order change | 3.25 | 1.348 | 5.65 | 1.395 | 5.12 | 1.055 |
| order change, `j` run 8 | 1.96 | 1.112 | 4.93 | 1.232 | 3.59 | 0.983 |
| two-pass 128-element scratch | 2.70 | 0.930 | 5.28 | 0.935 | 4.13 | 0.937 |
| **tile shrink to 6 (shipped)** | **1.84** | **1.091** | **4.76** | **1.109** | **3.74** | **1.018** |

The shrink is the best of the five at one thread and the second best at four,
which is why it is the shipped branch; the two-pass scratch tile is the only
variant that looked close to `main` at four threads and it loses 1.1-1.5x at one.
Leaving this class on `main`'s order would hold those cells exactly at `main` and
forgo the 1.8-2.0x the class gains at one thread, which is why the maintainer
chose the shrink for this class in #480.

None of these variants is re-measured at four threads in the fresh round: the
shipped one is, in §3, and it clears `main` there on all five cells it applies
to. Re-running the four rejected variants under the corrected protocol is a
different task (it re-opens the variant choice, which the maintainer has
decided) and is deliberately not done here.

## 5. Residual

- Serial: every row of the issue's two tables is at or below its neighbours
  (`penalty after` 0.86-1.08x; `[256,256]` 1.05x and `[1024,1024]` 1.08x are the
  both-aliasing rows).
- Four threads: no `conj_copy` cell is above `main` outside 1% (§3). The five
  both-aliasing cells sit at 0.33x-0.98x of `main`, i.e. the class that this
  change was said to regress does not regress on this host at this revision.
- `f64` never enters this kernel: 70 of its 72 `l1_sets_466` rows are within
  +-3% of `main`, and the two that are not move 5% in opposite directions
  (`1T [31,31,31]` 1.05x slower, `1T [95,95,95]` 0.95x, round spread below 1%).
  The `f64` source is unchanged, so that pair is binary layout, not throughput.
- Open, and *not* this change: the width gate of §3. A permutation of 1M
  `Complex<f64>` elements (16 MB a side) never fans out on this 64-core host,
  because the estimate uses the contiguous-copy rate (0.090 ns/element) while the
  permutation costs 3-6 ns/element. That is the class already tracked as
  [#363](https://github.com/tensor4all/strided-rs/issues/363) ("`KernelClass::Copy`
  rate is below the measured cost of strided transposes, so their width is chosen
  for a much smaller job"); this run adds the `x86_64-avx2` evidence there. It is a
  policy question about `copy_ns_per_element`, not about the loop order, so it is
  not changed here.

## 6. Second round (2026-11-15): the two-branch kernel measured against `main`

The maintainer's decision on #480 was the combination §1 already implements:
destination-only aliasing keeps the full-run order change (`macro_kernel.rs:315`),
both operands aliasing keeps the old order with the six-row shrink
(`macro_kernel.rs:299-314`), and nothing else changes. This round is therefore a
rebase, a paired re-measurement of that exact kernel and a documentation
correction — no `src` change.

- Branch rebased from `6bc47e6` onto `main` `3fb9e33` (three README edits upstream,
  no conflict; `strided-perm/src` unchanged by it).
- Baseline worktree `~/wt/strided-466-base` at `3fb9e33` plus the same two
  `[[bench]]` entries and a `cmp`-identical `l1_sets_466.rs`; only
  `strided-perm/src` differs between the arms.
- Host: AMD EPYC 7702P as in §0, boost off, governor `performance`, rustc 1.99.0,
  `--release`. Measured cells `taskset -c 40` (serial rows) and `taskset -c 40-43`
  (pooled rows); the observing session pinned to `60-63` with `taskset -apc`. Load
  average 0.60-1.35 over the six runs of each round; another user's `python3` was
  on the host but never on `40-43`.
- Three rounds, each running `base`, `branch` back to back for `l1_sets_466` 1T,
  `l1_sets_466` 4T and `conj_copy`; every table entry is the median of the three
  round values, each round itself a median of 9 samples (`l1_sets_466`) or a
  minimum of 21 (`conj_copy`).

The issue's tables, fresh, `Complex<f64>`, ns per element (`before` = `main`, `after`
= this branch, `penalty` = `2^d` over the mean of its two neighbours):

### 1T

| dims | perm | before | after | after/before | penalty before | penalty after |
|---|---|---|---|---|---|---|
| `[1024, 32]` | `[1, 0]` | 5.753 | 1.110 | 0.19x | 4.57x | 0.88x |
| `[1024, 64]` | `[1, 0]` | 5.653 | 1.058 | 0.19x | 4.51x | 0.86x |
| `[2048, 32]` | `[1, 0]` | 5.708 | 1.096 | 0.19x | 4.55x | 0.88x |
| `[2048, 64]` | `[1, 0]` | 5.648 | 1.069 | 0.19x | 4.54x | 0.86x |
| `[256, 256]` | `[1, 0]` | 6.272 | 1.771 | 0.28x | 3.74x | 1.05x |
| `[1024, 1024]` | `[1, 0]` | 5.958 | 3.267 | 0.55x | 1.96x | 1.08x |
| `[32, 32, 32]` | `[1, 2, 0]` | 5.668 | 1.135 | 0.20x | 4.92x | 1.00x |
| `[64, 64, 64]` | `[1, 2, 0]` | 5.630 | 1.202 | 0.21x | 4.38x | 0.93x |
| `[96, 96, 96]` | `[1, 2, 0]` | 5.558 | 3.116 | 0.56x | 1.69x | 0.95x |
| `[128, 128, 128]` | `[1, 2, 0]` | 5.662 | 2.885 | 0.51x | 1.75x | 0.89x |
| `[64, 64, 64]` | `[2, 1, 0]` | 5.600 | 1.635 | 0.29x | 3.22x | 0.96x |
| `[64, 64, 64]` | `[2, 0, 1]` | 1.354 | 1.316 | 0.97x | 0.97x | 0.95x |

### 4T

| dims | perm | before | after | after/before | penalty before | penalty after |
|---|---|---|---|---|---|---|
| `[1024, 32]` | `[1, 0]` | 5.669 | 1.102 | 0.19x | 4.47x | 0.87x |
| `[1024, 64]` | `[1, 0]` | 5.658 | 1.063 | 0.19x | 4.59x | 0.86x |
| `[2048, 32]` | `[1, 0]` | 5.668 | 1.072 | 0.19x | 4.61x | 0.87x |
| `[2048, 64]` | `[1, 0]` | 5.632 | 1.093 | 0.19x | 4.52x | 0.88x |
| `[256, 256]` | `[1, 0]` | 6.430 | 1.676 | 0.26x | 3.78x | 0.98x |
| `[1024, 1024]` | `[1, 0]` | 5.980 | 3.274 | 0.55x | 1.98x | 1.08x |
| `[32, 32, 32]` | `[1, 2, 0]` | 5.650 | 1.152 | 0.20x | 5.24x | 1.05x |
| `[64, 64, 64]` | `[1, 2, 0]` | 5.660 | 1.165 | 0.21x | 4.44x | 0.90x |
| `[96, 96, 96]` | `[1, 2, 0]` | 5.567 | 3.191 | 0.57x | 1.70x | 0.97x |
| `[128, 128, 128]` | `[1, 2, 0]` | 1.750 | 1.617 | 0.92x | 1.20x | 1.10x |
| `[64, 64, 64]` | `[2, 1, 0]` | 5.651 | 1.642 | 0.29x | 3.25x | 0.96x |
| `[64, 64, 64]` | `[2, 0, 1]` | 1.312 | 1.353 | 1.03x | 0.92x | 0.95x |

`f64` (native micro-kernel) `2^d` rows, 1T and 4T: 0.99-1.03x, as in §2. Output
identity: the `digest` and `digest-weighted` columns of all 144 rows are
identical between the two arms in every round, 0 mismatches.

`conj_copy`: §3. The four cells the maintainer named — `256^2`, `1024^2`,
`2048^2`, `4096^2` at four threads — are 0.33x, 0.55x, 0.87x and 0.98x of `main`,
and the fifth shrink-branch cell, `[32,32,32,32]` perm `[3,1,0,2]`, is 0.53x.

Gates run on the rebased branch: `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets -j4 -- -D warnings`,
`cargo clippy --workspace --features parallel -j4 --lib --tests --examples --
-D warnings`, `cargo test -j4 -p strided-perm`, `cargo test -j4 --workspace`,
`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` and the repository
rules review in `--dry-run --worktree` mode. Results are in the PR body.

Plan review for this round (required by the worker gate): `glm-5.3-flash`
(`ollama-cloud/glm-5.3-flash:high`, read-only subagent) returned
`VERDICT: CHANGES` on the first draft — it caught that my acceptance bar quoted
the serial-only penalty range and that the "four cells" were not enumerated —
and `VERDICT: APPROVE` on the corrected plan. Recorded in
`IMPLEMENTATION-PLAN.md` in the worktree (not committed).
