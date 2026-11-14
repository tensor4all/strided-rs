# 2026-11-14: #467 — a conjugating copy in the permutation engine

Scope: `strided-perm`'s copy engine gains a conjugating element store, exposed as
`copy_conj_into` / `copy_conj_into_uninit`. The planner, the fold, the blocking
and the native `f64`/`f32` micro-kernels are unchanged; only the element store
differs (HPTT's `conjA`, monomorphized instead of a runtime flag).

`strided_basic::copy_conj_into` is **not** changed: the route to the new entry
was implemented, measured, and dropped (§2). The public entries are the
deliverable; the default route stays with the maintainer.

## 0. Host and protocol

Machine: **AMD Ryzen AI MAX+ 395** (Zen 5, 16 cores / 32 threads, two 32 MiB L3
domains, SMT on, governor `performance`, boost on, rustc 1.100.0, `--release`).
The two domains are `0-7,16-23` and `8-15,24-31`, read at run time from
`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`.

Isolation: the measured binary ran under `taskset -c 0-7,16-23` (one L3 domain),
the observing `pi` session was pinned to `8-15,24-31` for the duration
(`taskset -apc`).

Three passes were taken. The recorded one ran with a **quiet minute**: load
average 0.97 / 12.0 / 21.2 (1/5/15 min) before and 1.13 / 11.5 / 20.9 after.
The two earlier passes ran at 25-30 on the 1-minute average with other
`cargo`/`rustc` processes on the host; their ratio columns read 0.61-1.64 and
0.71-1.66 against 0.60-1.60 here, so the shape of the table did not move. The
5/15-minute columns above are the residue of those earlier sessions, not of this
run.

Every row is 2 warmup samples plus 21 measured samples. The three stores in a
row rotate their order per sample, so all of them see the same background load
in a sample; the ns/e columns report each store's **minimum**, the ratio columns
the **median of the per-sample ratios** (`conj_i / plain_i`, `threaded_i /
conj_i`). Absolute ns/e is indicative only even on this host; the ratios with
their per-sample pairing are the claim.

## 1. The store inside the engine

`strided-perm/benches/conj_copy.rs`, `Complex<f64>`, ns per element.
`plain`/`conj` are `Exec::serial()`; `conj4T` is `Exec::new(&pool, 4)`. `conj/p`
is the median per-sample conj/plain ratio, `4T/1T` the median per-sample
`conj4T/conj`.

| case | plain | conj | conj4T | conj/p | 4T/1T | elements |
|---|---|---|---|---|---|---|
| 2d 255x255 `[1,0]` | 0.550 | 0.728 | 0.731 | 1.33 | 1.00 | 65025 |

| 2d 256x256 `[1,0]` | 2.787 | 2.687 | 2.690 | 0.97 | 1.01 | 65536 |

| 2d 257x257 `[1,0]` | 1.434 | 0.830 | 0.836 | 0.60 | 1.00 | 66049 |

| contig 256x256 `[0,1]` | 0.291 | 0.464 | 0.468 | 1.60 | 1.00 | 65536 |

| 2d 1023x1023 `[1,0]` | 1.381 | 1.709 | 0.617 | 1.28 | 0.34 | 1046529 |

| 2d 1024x1024 `[1,0]` | 3.328 | 3.040 | 0.934 | 0.91 | 0.32 | 1048576 |

| 2d 1025x1025 `[1,0]` | 1.717 | 2.037 | 0.615 | 1.26 | 0.29 | 1050625 |

| contig 1024x1024 `[0,1]` | 0.428 | 0.554 | 0.212 | 1.28 | 0.40 | 1048576 |

| 2d 2047x2047 `[1,0]` | 2.165 | 3.281 | 1.050 | 1.54 | 0.32 | 4190209 |

| 2d 2048x2048 `[1,0]` | 3.511 | 3.318 | 1.166 | 0.95 | 0.36 | 4194304 |

| 2d 2049x2049 `[1,0]` | 2.333 | 3.617 | 1.046 | 1.55 | 0.28 | 4198401 |

| contig 2048x2048 `[0,1]` | 0.630 | 0.766 | 0.472 | 1.22 | 0.64 | 4194304 |

| 2d 4095x4095 `[1,0]` | 2.214 | 3.371 | 1.140 | 1.52 | 0.34 | 16769025 |

| 2d 4096x4096 `[1,0]` | 3.775 | 3.699 | 1.277 | 0.99 | 0.35 | 16777216 |

| 2d 4097x4097 `[1,0]` | 2.415 | 3.820 | 1.151 | 1.56 | 0.31 | 16785409 |

| contig 4096x4096 `[0,1]` | 0.653 | 0.880 | 0.530 | 1.35 | 0.60 | 16777216 |

| `[64,64,64]` `[2,0,1]` | 0.538 | 0.732 | 0.302 | 1.34 | 0.50 | 262144 |

| `[128,128,128]` `[2,0,1]` | 2.285 | 3.186 | 1.078 | 1.40 | 0.34 | 2097152 |

| `[32,32,32,32]` `[3,1,0,2]` | 3.978 | 3.529 | 0.949 | 0.88 | 0.27 | 1048576 |

| `[100,2,100]` `[2,0,1]` | 0.450 | 0.671 | 0.671 | 1.49 | 1.00 | 20000 |

The issue's proposed acceptance bar was "conj route within ~10% of plain
`copy_into`". **It is not met on every row.** Reading the table by shape class:

- Where the plain path is itself slow (`2^d` 2D at 2.8-3.8 ns/e, `[32;4]` at
  4.0), conj is at parity or faster (0.88-0.99). Both sides are bound by the
  same cache behaviour, and the extra sign flip is free.
- Where the plain path is fast, conj costs 1.26-1.56x: the odd 2D extents 255²,
  1023², 1025², 2047², 2049², 4095², 4097² and the 3D/4D rows 64³ (1.34),
  128³ (1.40) and `[100,2,100]` (1.49). The exception is 257² (0.60), where the
  plain store is the slow side (1.43 vs 0.83 ns/e).

  The conjugation is one floating-point sign flip per stored element in the
  generic element store; with both sides already at memory speed it is a
  visible fraction of the work.
- A contiguous conj copy cannot use `memcpy`: 1.22-1.35x on tensors past L2 and
  1.60x on a 256x256 (L2-resident) copy.
- Threading never costs on a genuinely parallel row (`4T/1T` 0.27-0.65 on every
  row above 2^18 elements), which is what the repository requires of a
  tensor-sized case. Below the width decision's bar the copy stays on the
  calling thread, so those rows time the same serial run twice and sit at
  1.00-1.01 — they are the control, not a scaling claim.

So the *engine* delivers the conjugation without an extra pass, but not at plain
copy cost on every layout. Nothing here was tuned further: a complex-element
micro-kernel (HPTT's `conjA` inside the register tile) could remove part of the
gap, and the power-of-two 2D weakness is pre-existing and dtype-independent
(issue #364's worklog measures the same class of shape for `f64`).

## 2. The routing decision in `strided_basic::copy_conj_into`

`strided_basic::copy_conj_into` still conjugates through `src.conj()` and the
element-map copy. A route to the new entry was implemented
(`element_conj_is_nonidentity::<T>()` on `Complex32`/`Complex64`, plus
`same_contiguous_layout(...).is_none()`, plus `threading::copy_conj_permuted`)
and then dropped, because the measurement does not support it.

Three-way comparison on the issue's shapes, `Complex<f64>`, `Exec::serial()`, ns
per element, minimum of 21, same host; taken in the earlier, heavier-load pass,
so treat the ratios as indicative (`map conj` is the old route this change
leaves in place):

| case | perm plain | perm conj (new entry) | map conj (current route) | map/engine |
|---|---|---|---|---|
| 255x255 | 0.568 | 0.734 | 0.850 | 1.18 |
| 256x256 | 2.794 | 2.696 | 1.777 | 0.66 |
| 1023x1023 | 1.295 | 1.568 | 1.588 | 1.01 |
| 1024x1024 | 3.321 | 3.040 | 1.790 | 0.59 |
| 1025x1025 | 1.674 | 1.855 | 1.644 | 0.86 |
| 2048x2048 | 3.504 | 3.302 | 2.871 | 0.87 |
| `[64,64,64]` | 0.639 | 0.833 | 2.034 | 2.32 |
| `[128,128,128]` | 2.156 | 2.915 | 2.409 | 0.81 |
| `[32,32,32,32]` | 3.806 | 3.577 | 3.113 | 0.87 |
| `[100,2,100]` | 0.431 | 0.671 | 0.782 | 1.14 |

The two routes are unordered. The engine wins on 255² (1.18x), `[64,64,64]`
(2.32x) and `[100,2,100]` (1.14x); 1023² is a tie (1.01); the element map wins on
256² (1.5x), 1024² (1.7x), 1025² (1.2x), 2048² (1.15x), `[128,128,128]` (1.23x)
and `[32;4]` (1.15x). No structural property of the layout separates the two
families, so a gate would be a shape heuristic, and this repository runs one
measured threshold per operation instead.

The first version of this worklog kept the routing anyway, arguing that a plain
2D complex transpose already pays the engine's cost through
`copy_transpose_scale_into`. **That argument was wrong**:
`copy_transpose_scale_into` tries `try_copy_transpose_scale_2d_identity_tiled`
(`strided-basic/src/ops_view.rs:1737`) before the permutation route, and that
helper is generic over `T: Copy + Mul`, so a normal column-major 2D complex
transpose never reaches `threading::copy_permuted` (`ops_view.rs:1904-1911`).
Routing the conjugating copy to the engine is therefore a new cost, not a shared
one, and the `2^d` rows above are a real regression rather than an accepted one.

What is left is the public entry and the table. Choosing the default route is a
policy decision for the maintainer in #467; the one-line gate is in the reverted
commit, should it be wanted.

## 3. Gates run

`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -D
warnings`, `cargo test --workspace` (all green, including doctests), the
`strided-perm` `--features parallel` lane, and
`python3 scripts/repository-rules-review.py --base main --worktree --dry-run`.
Builds at `-j 4`; `kache doctor --verify` passed before the first build.
`CONJ_COPY_GATE=1 cargo bench -j 4 --features parallel -p strided-perm --bench
conj_copy` passes on the table in §1 (gate 2.0x, the store falling off the plain
path's kernel, not a few percent).

## 4. Not measured

- A quiet-host run. Every number here was taken with other sessions building on
  the host; the paired protocol is what makes the ratios usable, and no absolute
  throughput claim is made.
- `Complex<f32>` throughput. Tests cover it; the tables above are `Complex<f64>`
  only.
- The old-route comparison in §2 re-run on the quieter host (the reverted gate is
  gone; re-measuring it means restoring the gate by hand).
- The other hosts in the issue's table (Apple M2), and any cross-machine
  comparison.
