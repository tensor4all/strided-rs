# #467 — routing `strided_basic::copy_conj_into` through the conjugating permutation engine

Delivers the routing gate left open by #478 (engine entry `strided_perm::copy_conj_into` /
`copy_conj_into_uninit`, squash `6bc47e61`). The HPTT kernel, the plan, the width decision and
`strided_perm`'s public surface are untouched; #466/#480 own the kernel's aliasing weakness.

Host: EVO-X2, AMD Ryzen AI MAX+ 395 (16C/32T, two 32 MiB L3 domains `0-7,16-23` and
`8-15,24-31`), rustc 1.100.0, `Complex<f64>`, `Exec::serial()`, `--release`. Measured cell pinned
to `taskset -c 0-7,16-23`; the observer, the agent and the compile before each pass ran on the
other domain (`taskset -apc 8-15,24-31 <agent pid>`), so no observing thread shared an L3 domain
with the measured cell. Load average 0.8-1.0 before and after each pass of the pre-#480 table,
1.7-1.9 during the current-base table (the agent and its build on the other domain).

This worklog now carries **two** measurement tables, and every row says which base it was
measured on: the pre-#480 table of section 2 against `3fb9e33`, and the current-base
table against `4197c6f` (#480) plus this change. The tables are not comparable row by row —
#480 fixed the engine's L1 aliasing, which moved the 2D rows and the `[32;4]` row by more than
the bound they decided.

## 1. Decision

`strided_basic::copy_conj_into` now hands a copy to the engine when all of these hold, and keeps
the element map otherwise ([`conj_perm_engine_route`], `strided-basic/src/ops_view.rs`):

1. `T` is `Complex32`/`Complex64` — a real type's `conj` is the identity, so the engine's
   `ConjStore` would only be its generic element store;
2. the source and destination layouts are not the same contiguous layout — a contiguous copy
   keeps the map's SIMD store loop and its splitting contiguous path, and the engine's
   conjugating store has no `memcpy`;
3. one operand's packed footprint — `total_len(dims) * size_of::<T>()`, not the address span of
   the strided view — is below 24 MiB (`CONJ_PERM_ENGINE_MAX_BYTES`).

No shape list is involved: one measured constant and two layout/type properties, each stated
with the rows that fix it.

**The rank-2 4 KiB row clause is gone.** It stood in for the engine's own aliasing fix (#466) on
the pre-#480 base, where a power-of-two destination row of one 4 KiB set span or longer was
1.27-1.70x faster on the map. With #480 in the base the engine wins those rows outright — 256^2
2.28-2.30x, 512^2 2.21-2.22x, 1024^2 1.19-1.20x, and 1.11-1.12x at 2048^2, which the footprint
bound keeps on the map anyway — so the clause only kept 2.3x of map route on the fastest rows.
It is removed rather than re-tuned: no current-base rank-2 row is a decided map win (section 3
names the one row where the map is still ahead, 1025^2, and why it is not separable), and a
clause fitted to one row shape would be a shape table, not a measured bound.

## 2. Measurement

`strided-basic/benches/conj_route_467.rs` (`harness = false`) times three stores on the same
views, rotating their order inside each sample: `map` is `copy_into(dst, &src.conj(), ..)` (the
route in place before this change), `engine` is `strided_perm::copy_conj_into`, and `prod` is
`strided_basic::copy_conj_into`. ns/element are minima of 21 samples; `map/eng`, `prod/best` and
their `+-` bands are medians and `3 * max(1.4826 * MAD, 0.15 * median)` over the per-sample
ratios, the bound `benches/support/mod.rs` uses for a parity row.

Two passes after the gate, `ns`/element. **Measured on the pre-#480 base (`3fb9e33`)**, with the
4 KiB rank-2 clause in the predicate — this table is the evidence the clause was built on and is
kept as the historical record, not as current behaviour:

| case | map | engine | prod | map/eng | prod/best |
|---|---|---|---|---|---|
| 2d 128x128 | 0.996 / 1.013 | 0.683 / 0.679 | 0.685 / 0.684 | 1.50 | 1.00 |
| 2d 254x254 | 0.822 / 0.825 | 0.705 / 0.715 | 0.710 / 0.720 | 1.17 | 1.01 |
| 2d 255x255 | 0.827 / 0.905 | 0.721 / 0.747 | 0.720 / 0.745 | 1.21 | 1.00 |
| 2d 256x256 | 1.816 / 2.050 | 2.694 / 2.713 | 1.813 / 2.049 | 0.79 | 0.99 |
| 2d 257x257 | 0.880 / 1.038 | 0.819 / 0.948 | 0.871 / 1.020 | 1.11 | 1.09 |
| 2d 512x512 | 1.740 / 1.735 | 2.958 / 2.958 | 1.741 / 1.736 | 0.59 | 1.00 |
| 2d 1023x1023 | 1.673 / 1.744 | 1.750 / 1.758 | 1.728 / 1.766 | 0.97 | 1.00 |
| 2d 1024x1024 | 2.013 / 2.019 | 3.045 / 3.059 | 2.089 / 2.032 | 0.68 | 1.01 |
| 2d 1025x1025 | 1.720 / 1.728 | 2.020 / 2.005 | 1.784 / 1.746 | 0.87 | 1.01 |
| 2d 2048x2048 | 3.106 / 3.133 | 3.380 / 3.368 | 3.135 / 3.130 | 0.95 | 1.00 |
| `[64,64,64]` | 1.744 / 1.741 | 0.707 / 0.729 | 0.711 / 0.720 | 2.38 | 1.00 |
| `[65,65,65]` | 0.913 / 0.911 | 0.718 / 0.717 | 0.711 / 0.719 | 1.25 | 0.99 |
| `[128,128,128]` | 2.661 / 2.683 | 3.103 / 3.133 | 2.684 / 2.684 | 0.85 | 1.01 |
| `[32;4]` | 3.197 / 3.226 | 3.360 / 3.345 | 3.360 / 3.347 | 0.98 | 1.02 |
| `[100,2,100]` | 0.834 / 0.843 | 0.685 / 0.685 | 0.684 / 0.684 | 1.23 | 1.00 |
| 32^3 | 1.888 / 1.885 | 0.706 / 0.706 | 0.712 / 0.710 | 2.65 | 1.01 |
| 48^3 | 1.771 / 1.778 | 0.717 / 0.718 | 0.717 / 0.721 | 2.43 | 1.00 |
| 64^3 | 1.736 / 1.749 | 0.719 / 0.725 | 0.730 / 0.725 | 2.37 | 1.00 |
| 80^3 | 1.871 / 1.879 | 0.753 / 0.749 | 0.762 / 0.755 | 2.43 | 1.00 |
| 96^3 | 2.028 / 2.054 | 1.423 / 1.420 | 1.446 / 1.411 | 1.33 | 0.99 |
| 100^3 | 1.924 / 1.932 | 1.674 / 1.717 | 1.679 / 1.681 | 1.08 | 1.00 |
| 108^3 | 2.266 / 2.218 | 2.136 / 2.181 | 2.202 / 2.177 | 0.99 | 1.00 |
| 112^3 | 3.952 / 4.018 | 2.462 / 2.478 | 2.461 / 2.427 | 1.62 | 1.00 |
| 116^3 | 2.333 / 2.364 | 2.400 / 2.409 | 2.407 / 2.357 | 0.94 | 1.03 |
| 120^3 | 2.373 / 2.403 | 2.635 / 2.626 | 2.426 / 2.437 | 0.91 | 1.01 |
| 124^3 | 2.333 / 2.384 | 2.693 / 2.658 | 2.378 / 2.399 | 0.89 | 1.01 |
| contig 64x64 | 0.195 / 0.195 | 0.508 / 0.515 | 0.198 / 0.198 | 0.39 | 1.01 |
| contig 1024x1024 | 0.425 / 0.428 | 0.502 / 0.512 | 0.425 / 0.424 | 0.84 | 1.01 |

A pre-gate pass (production was the map route on every row) put `map/eng` at 1.50, 1.17, 1.14,
0.80, 1.05, 0.59, 0.97, 0.69, 0.88, 0.95, 2.35, 1.24, 0.83, 1.03, 1.21, 2.65, 2.53, 2.38, 2.52,
1.25, 1.07, 0.99, 1.57, 0.93, 0.94, 0.90, 0.39, 0.81 on the same rows in the same order: the two
pre-gate passes agree with the post-gate `map`/`engine` columns within their own blank-run spread,
so the routing, not a change in the runs, is what moved `prod`.

What the constants rest on:

- **24 MiB operand bound** (`CONJ_PERM_ENGINE_MAX_BYTES`, 25 165 824 B = 1 572 864 `Complex64`
  elements). 112^3 (21.4 MiB) is the last outright engine win at 1.51-1.67x; 116^3 (23.8 MiB) is
  a tie at 0.93-0.96 and stays on the engine side of the bound; 120^3 (26.4 MiB, 0.90-0.93),
  124^3 (29.1 MiB, 0.86-0.89) and 128^3 (32.0 MiB, 0.84-0.97: a tie in the busiest pass, a
  1.03-1.19x map win otherwise) are above it and keep the map. So the bound sits between the last
  engine win and the rows the map wins or ties, and the only row it leaves on the engine while the
  map is ahead is the 116^3 tie. Re-measured on the current base (section 2.1) the same bracket
  holds: 112^3 1.49-1.58 engine, 116^3 0.93-0.94, 120^3 0.91 tie-ish, 124^3 0.87, 128^3 0.89-0.90
  map — the bound is not a pre-#480 artefact.

### 2.1 Current base (`4197c6f` + this change, rank-2 clause removed)

Same harness, same host, same protocol: `taskset -c 0-7,16-23`, observer and build on
`8-15,24-31`, `Exec::serial()` (1 thread), load average 1.66-1.92, `rustc 1.100.0`. Passes 1 and 3
of three consecutive passes; all three exited 0 under `CONJ_ROUTE_GATE=1`.

| case | map | engine | prod | map/eng | prod/best |
|---|---|---|---|---|---|
| 2d 128x128 | 0.998 / 1.019 | 0.674 / 0.675 | 0.674 / 0.678 | 1.48 / 1.51 | 1.00 / 1.00 |
| 2d 254x254 | 0.861 / 0.865 | 0.707 / 0.718 | 0.709 / 0.721 | 1.23 / 1.21 | 1.00 / 1.01 |
| 2d 255x255 | 0.934 / 0.864 | 0.751 / 0.730 | 0.744 / 0.727 | 1.25 / 1.18 | 0.99 / 0.99 |
| 2d 256x256 | 2.140 / 1.798 | 0.951 / 0.788 | 0.959 / 0.790 | 2.30 / 2.28 | 1.01 / 1.00 |
| 2d 257x257 | 1.063 / 0.883 | 0.960 / 0.813 | 0.977 / 0.824 | 1.09 / 1.08 | 1.01 / 1.01 |
| 2d 512x512 | 1.738 / 1.731 | 0.781 / 0.783 | 0.781 / 0.782 | 2.22 / 2.21 | 1.00 / 1.00 |
| 2d 1023x1023 | 1.737 / 1.760 | 1.739 / 1.705 | 1.733 / 1.701 | 1.00 / 1.11 | 1.01 / 1.00 |
| 2d 1024x1024 | 2.052 / 2.113 | 1.723 / 1.717 | 1.716 / 1.715 | 1.20 / 1.20 | 1.00 / 1.00 |
| 2d 1025x1025 | 1.827 / 1.810 | 2.050 / 2.057 | 1.979 / 2.021 | 0.88 / 0.90 | 1.09 / 1.11 |
| 2d 2048x2048 | 3.122 / 3.180 | 2.826 / 2.827 | 3.151 / 3.173 | 1.11 / 1.12 | 1.11 / 1.10 |
| `[64,64,64]` | 1.745 / 1.753 | 0.702 / 0.708 | 0.706 / 0.704 | 2.40 / 2.44 | 1.00 / 0.99 |
| `[65,65,65]` | 0.941 / 0.943 | 0.710 / 0.702 | 0.710 / 0.705 | 1.31 / 1.29 | 1.00 / 0.99 |
| `[128,128,128]` | 2.862 / 2.895 | 3.219 / 3.159 | 2.853 / 2.923 | 0.89 / 0.90 | 1.00 / 1.00 |
| `[32;4]` | 3.081 / 3.086 | 1.627 / 1.636 | 1.624 / 1.639 | 1.87 / 1.87 | 1.00 / 0.99 |
| `[100,2,100]` | 0.881 / 0.872 | 0.683 / 0.671 | 0.684 / 0.665 | 1.29 / 1.29 | 1.00 / 1.00 |
| 32^3 | 1.893 / 1.877 | 0.698 / 0.709 | 0.705 / 0.707 | 2.69 / 2.66 | 1.01 / 1.00 |
| 48^3 | 1.787 / 1.770 | 0.718 / 0.712 | 0.721 / 0.709 | 2.45 / 2.43 | 1.00 / 1.00 |
| 64^3 | 1.759 / 1.754 | 0.718 / 0.711 | 0.713 / 0.714 | 2.40 / 2.37 | 1.00 / 1.00 |
| 80^3 | 1.860 / 1.849 | 0.721 / 0.735 | 0.726 / 0.734 | 2.47 / 2.41 | 0.99 / 0.99 |
| 96^3 | 2.075 / 2.051 | 1.436 / 1.398 | 1.425 / 1.401 | 1.35 / 1.35 | 0.98 / 0.98 |
| 100^3 | 1.937 / 1.916 | 1.729 / 1.649 | 1.690 / 1.641 | 1.09 / 1.12 | 0.99 / 0.99 |
| 108^3 | 2.222 / 2.220 | 2.106 / 2.149 | 2.057 / 2.108 | 1.03 / 1.01 | 1.00 / 0.99 |
| 112^3 | 3.728 / 3.763 | 2.481 / 2.534 | 2.464 / 2.508 | 1.58 / 1.49 | 1.01 / 1.00 |
| 116^3 | 2.353 / 2.381 | 2.438 / 2.409 | 2.436 / 2.443 | 0.93 / 0.94 | 1.07 / 1.05 |
| 120^3 | 2.433 / 2.460 | 2.685 / 2.681 | 2.448 / 2.418 | 0.91 / 0.92 | 1.01 / 0.98 |
| 124^3 | 2.364 / 2.410 | 2.670 / 2.718 | 2.405 / 2.361 | 0.87 / 0.87 | 1.02 / 0.98 |
| contig 64x64 | 0.200 / 0.193 | 0.515 / 0.530 | 0.203 / 0.200 | 0.41 / 0.37 | 1.01 / 1.04 |
| contig 1024x1024 | 0.423 / 0.422 | 0.501 / 0.505 | 0.423 / 0.426 | 0.84 / 0.83 | 1.01 / 1.00 |

What moved, and why the clause could not survive:

- The 2D power-of-two rows inverted. 256^2 was 0.79 map/eng on the pre-#480 base and is 2.28-2.30
  now; 512^2 was 0.59 and is 2.21-2.22. `prod` follows the engine there in both passes (0.959 and
  0.790 against a 2.14/1.80 map) — the 2.3x the clause was costing.
- 1024^2 moved from 0.68 to 1.20 (16 MiB, under the footprint bound: engine) and 2048^2 from 0.95
  to 1.11 (64 MiB, over it: map). The footprint bound, not the row length, is what separates them
  now.
- `[32;4]` moved from 0.98 (map 1.02-1.05x ahead, section 3 of the pre-#480 record) to 1.87 on
  the engine: #480 fixed the both-operands-aliasing class, so the residual that the earlier
  record had to carry is gone.
- Every 3D row the 24 MiB bound decides stays on the same side of it as on the pre-#480 base
  (112^3 engine 1.49-1.58, 116^3 tie 0.93-0.94, 120^3 0.91 map, 124^3 0.87 map, 128^3 0.89-0.90
  map), so the removal did not disturb that bracket.
- The contiguous rows are still map-side, by 2.4-2.7x at 64^2 and 1.19x at 1024^2 (map/eng
  0.37-0.41 and 0.83-0.84), as the contiguous
  clause requires.

## 3. Residual losses (documented, not hidden)

Re-measured on the current base; the pre-#480 residuals that #480 itself fixed are recorded in
section 2, not here.

- **1025^2 (16400 B destination row): map 1.10-1.14x ahead** (map/eng 0.88-0.91), and it now takes
  the engine: `prod/best` is 1.09-1.11 in these passes. The map's advantage sits inside the
  gate's own paired-sample band (1.09 < 1 + 0.50), so it is not a decidable routing statement, and
  the row is the only shape in the set where the map is still ahead. A clause that kept it would
  be fitted to one row (`row_bytes` just over two 4 KiB spans) with no second row behind it — a
  shape table, which is what this predicate exists to avoid. Left on the engine, documented here.
- **2048^2 (64 MiB) stays on the map while the engine is 1.11-1.12x faster** (map/eng 1.11-1.12,
  `prod/best` 1.10-1.11). It is over `CONJ_PERM_ENGINE_MAX_BYTES`, and that bound is set by the 3D
  rows, where the map wins or ties above 24 MiB (120^3-128^3). Moving the bound to cover a 64 MiB
  2D row would put 120^3-128^3 on the engine, where they are 1.10-1.15x slower. The loss is inside
  the band's own width, so the bound stays.
- **116^3 (23.8 MiB) stays on the engine while the map is 1.05-1.07x ahead** — the pre-existing
  tie the 24 MiB bound deliberately leaves on the engine side (unchanged by this work).

## 4. Gates run

### 4.1 For the pre-#480 table and the original predicate

```
cargo fmt --all -- --check
cargo clippy -j 8 --workspace --all-targets -- -D warnings
cargo clippy -j 8 --workspace --features parallel --lib --tests --examples -- -D warnings
cargo test -j 8 --workspace
cargo test -j 8 -p strided-basic -p strided-perm --features parallel --lib --tests
CONJ_ROUTE_GATE=1 taskset -c 0-7,16-23 cargo bench -j 8 -p strided-basic --bench conj_route_467
taskset -c 0-7,16-23 cargo bench -j 8 -p strided-perm --features parallel --bench conj_copy
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
```

`CONJ_ROUTE_GATE=1` exited 0 on three consecutive passes: two before the review fixes and one
after. The third pass shows the host spread on the rows that straddle the bounds — 112^3 1.67,
116^3 0.93, 124^3 0.86, 128^3 0.92, 255^2 1.36, 257^2 1.11, 1023^2 0.99 — and every row lands on the
same side of its bound as in the table above, so the bounds do not rest on one quiet hour.

`--bench conj_copy` (the engine's own conjugating store) is unchanged by construction: this commit
touches no file under `strided-perm/` (`git diff 3fb9e33 HEAD -- strided-perm/` is empty), and the
run above reproduces the state the engine benches recorded in #478 (2d 1023^2 conj/p 1.25,
1024^2 0.92, `[128,128,128]` 1.39, contig 256^2 1.63 on this host).

The appended results are in the PR body. `kache doctor --verify` passed (652 entries, 0
corrupted) before the first build.

### 4.2 For the current-base fix (rank-2 clause removed)

Reproduced the regression on this host before touching the predicate (`taskset -c 0-7,16-23`,
head `e2d2a2a`, load average 0.21-0.90):

```
prod is not on the faster route: 2d 256x256 [1,0] at map/eng 2.299, prod/best 2.302 (band 1.036)
prod is not on the faster route: 2d 512x512 [1,0] at map/eng 2.229, prod/best 2.229 (band 1.003)
error: bench failed          (exit status 1)
```

After the removal, three consecutive passes of
`CONJ_ROUTE_GATE=1 taskset -c 0-7,16-23 cargo bench -j 8 -p strided-basic --bench conj_route_467`
exited 0 (load average 1.66-1.92; the numbers are the section 2.1 table). No row was excused and
no bar moved: `STRONG_ROUTE`, `STRONG_SLACK` and the band formula are untouched by this change.

A fourth, confirmation pass ran on the committed state (`99365d1`) with the host quiet (load
average 0.17) and also exited 0. The rows that straddle a bound keep their side on all four
passes — 256^2 map/eng 2.27-2.30, 512^2 2.21-2.23, 1024^2 1.19-1.20, 2048^2 1.11-1.12, 1025^2
0.88-0.91, 112^3 1.49-1.58, 120^3 0.90-0.92, 128^3 0.89-0.90 — so the table is not one busy
hour's artefact, and the quiet pass lands on the same side as the loaded ones.

```
cargo fmt --all -- --check                                        (clean)
cargo test -j 8 --workspace                                       (86 suites, 1242 passed, 0 failed)
cargo test -j 8 -p strided-basic --test issue_467_conj_route       (8 passed)
cargo test -j 8 -p strided-basic --lib conj_route                        (9 passed)
cargo clippy -j 8 --workspace --all-targets -- -D warnings         (0 errors; the 4 pre-existing
                                                                  `homepage` x3 and unused
                                                                  `strided-kernel` dependency warnings)
cargo clippy -j 8 --workspace --features parallel --lib --tests --examples -- -D warnings  (0 errors)
RUSTDOCFLAGS="-D warnings" cargo doc -j 8 --workspace --no-deps     (no warnings)
```

The route-assertion tests were rewritten, not deleted: `route_takes_long_row_2d_copies_on_the_engine`
now pins 255^2-1024^2 to the engine and 2048^2 to the map by the footprint bound, and
`copy_conj_into_keeps_the_map_route`'s non-contiguous map case moved to 1500^2 (1500^2 x 16 B = 36 MB = 34 MiB per operand), so both
sides of the surviving bound still have a test. Deleting the engine branch still fails
`copy_conj_into_takes_the_engine_route`.

## 5. The regression this change fixes

Every number in the pre-#480 table was measured against `3fb9e33`, before the engine's
L1-aliasing fix landed below this commit (#480, `4197c6f`). On top of it, `CONJ_ROUTE_GATE=1`
failed exactly as reproduced in section 4.2 (same two rows, same band order):

```
prod is not on the faster route: 2d 256x256 [1,0] at map/eng 2.404, prod/best 2.334 (band 1.051)
prod is not on the faster route: 2d 512x512 [1,0] at map/eng 2.236, prod/best 2.229 (band 1.003)
```

The engine now wins the power-of-two 2D rows it used to lose, so the 4 KiB rank-2 clause was
stale: it kept a 2.3x loss on the shipping route and its `shortcut:` note contradicted the doc
block above it. The clause is removed here (sections 1 and 2.1); after the removal 1025^2
(0.88-0.90) is the only rank-2 row the map still wins, and the loss is inside the gate's own
band (section 3).

## 6. Review

Pre-#480 predicate — plan review before any code: `ollama-cloud/glm-5.3-flash` (read-only),
`VERDICT: APPROVE`, three IMPORTANT findings — a contiguous conj copy never takes `memcpy`
(`Conj::IS_IDENTITY` is false) and the clause was unmeasured, the plan's stop rule contradicted
its own acceptance bar, and the two routes can report different error variants on a doubly-broken
destination. All three were verified against source and folded into the plan (contiguous bench
rows; the stop rule is now "no gate lands"; the variant difference is pinned per route in the
integration test).

Pre-#480 predicate — post-implementation review of the finished diff, same model:
`VERDICT: APPROVE`, no BLOCKER, two IMPORTANT findings on the documented numbers — the 24 MiB
bound's evidence sentence quoted a reciprocal and the 116^3 row was described as map-side when
23.8 MiB is under the bound — plus a mislabeled map-route non-injective test case, an unsupported
1.52x figure, and release-note wording that turned ties into wins. All are fixed in this commit;
the reviewer's note that the bench's `STRONG_ROUTE` lane compared asymmetrically (`0.59` vs
`1.70`) is fixed by using the symmetric factor.

Current-base fix — plan review: `ollama-cloud/glm-5.3-flash`, `VERDICT: APPROVE`, no BLOCKER, no
IMPORTANT. It verified the clause and its contradictory comment at `ops_view.rs:1213-1242`, that
`L1_WAYS` stays used at `block.rs:249`, the complete list of affected tests, and the 8 + 9 test
counts, and asked for three MINOR changes: the `Complex32` 300x300 comment in
`issue_467_conj_route.rs` was already wrong before this change (a 2400 B row never reached the
4 KiB clause), the amend-not-stack rule came from the user-level work rules rather than this
repository's files, and the 2048^2 test replacement should be a smaller 1500^2 case. All three are
in this change.

Current-base fix — post-implementation review of the finished diff, same model: `VERDICT: APPROVE`,
no BLOCKER, no IMPORTANT, four MINOR findings on the documented numbers: a 256^2 range that cited
the unprinted middle pass (2.24 -> 2.28-2.30), two ranges rounded past their own best row
(contiguous 2.5-2.7 -> 2.4-2.7 at 64^2, and the CHANGELOG's 1.2-2.8 -> 1.2-2.7), the 1025^2
reciprocal (1.10-1.12 -> 1.11-1.14 ahead), and a byte-unit slip (1500^2 is 36 MB = 34 MiB, not
36 MiB). All four are fixed. The findings it verified instead: the predicate keeps three clauses
with no dangling 4 KiB/#466 reference, all four rewritten route assertions pin the surviving
side of the footprint bound (`non_injective`'s `[1<<20, 2]` case spans 2 elements, so it really
takes the map without allocating 33 MB), the 2.1 table is consistent with the prose in sections
1-3, and the three residual losses are exactly the rows the table shows a decided loss on.

`review: ollama-cloud/glm-5.3-flash approve` (plan, pre-#480), `review: ollama-cloud/glm-5.3-flash
approve` (diff, pre-#480, findings fixed), `review: ollama-cloud/glm-5.3-flash approve` (plan,
current base), `review: ollama-cloud/glm-5.3-flash approve` (diff, current base, findings fixed).

## 7. Not measured

- Pooled tokens: the gate is exec-independent (one bound, one layout property) and both routes
  keep their own width decision, but no >1-thread row was timed for this worklog. The engine's
  pooled behaviour is #478's table; the map's is unmeasured here.
- `Complex<f32>` throughput (covered by the integration test, and by the same bounds in bytes).
  The removed clause is what the `Complex32` test's comment had wrong: a 2D `Complex32` row is
  half the bytes of the same `Complex64` row, so the `Complex64` 4 KiB boundary never decided a
  `Complex32` copy.
- Non-square 2D and non-cubic 3D layouts outside the issue's rows; the footprint bound applies to
  them unmeasured. In particular no non-square 2D row was measured to test whether 1025^2's
  1.10-1.12x map advantage is a "row just over two set spans" class or a single shape.
- Any host other than this one. A machine with a different L3 size would move the 24 MiB bound;
  it is a measured constant, not a topology query.
