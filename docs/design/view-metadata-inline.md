# Inline view metadata

Date: 2026-10-17 (branch `release/0.6-alloc-free`)

This supersedes [`view-metadata-cow.md`](view-metadata-cow.md). The `Cow` design
removed `Arc` and made borrowed construction allocation-free, but it never
considered inline storage: every derived view (`permute`, `broadcast`,
`diagonal_view`, `transpose_2d`, `adjoint_2d`) and every `StridedArray`
constructor still allocated its owned `Vec` metadata, and `clone`/`conj` of a
computed view allocated again. Issue
[#306](https://github.com/tensor4all/strided-rs/issues/306) makes
allocation-freedom a workspace rule (P1–P7) with `INLINE_RANK = 8`.

## Decision

The tensor4all repositories standardize inline integer metadata on `tinyvec`
([tensor4all-agent-rules#22](https://github.com/tensor4all/tensor4all-agent-rules),
[strided-rs#312](https://github.com/tensor4all/strided-rs/issues/312)).
`strided-view` defines the capacity once and stores shape/strides in
`tinyvec::ArrayVec`:

```rust
pub const INLINE_RANK: usize = 8;
pub type StrideVec = tinyvec::TinyVec<[isize; INLINE_RANK]>; // P5 helper returns
pub type AxisVec = tinyvec::TinyVec<[usize; INLINE_RANK]>;

#[derive(Debug)]
struct ViewMeta {
    inline_dims: ArrayVec<[usize; INLINE_RANK]>,
    inline_strides: ArrayVec<[isize; INLINE_RANK]>,
    spilled: Option<Box<SpilledMeta>>, // Some only above INLINE_RANK
}
```

`StridedView`, `StridedViewMut` and `StridedArray` each hold one `ViewMeta`.

- `ArrayVec` is `Copy` for `Copy` items, so `Clone`/`view`/`as_view`/`conj` are
  a fixed-size bitwise copy of the two `ArrayVec`s plus one predictable `None`
  check, not a generic element walk.
- `new`/`new_unchecked`/`from_parts`/array constructors copy at most
  `2 * INLINE_RANK` words into the view. The view's lifetime is tied to the
  element data only; the layout does not have to outlive it.
- `new_owned` / `new_owned_unchecked` are removed. Downstream the same
  breakage is budgeted in #306 (tcontract `tests/batched.rs`,
  `benchmarks/.../exec_entry.rs`).
- `RawStridedRef` / `RawStridedMut` / `ErasedRawStrided*` stay borrowed: they
  are the prepared-replay descriptors and must not copy.
- P5 helpers return `StrideVec`/`AxisVec` (`tinyvec::TinyVec`), which spills
  above `INLINE_RANK` without error.

### Why `ArrayVec` + a separate spill field

`ArrayVec` is fixed-capacity and panics on overflow, so it cannot be the P5
helper return type; the helpers use `TinyVec`, whose heap variant covers the
spill. For the view metadata we use the `Copy` `ArrayVec` plus an explicit
`Option<Box<SpilledMeta>>` instead of `TinyVec` directly:

- `TinyVec` is an owning enum with a `Vec` arm, so it has drop glue and a
  discriminant; cloned into a view it measured ~30% slower and is 8 B larger.
- `ArrayVec` is `Copy`, has no drop glue and no per-collection heap
  discriminator, and the `spilled` slot is `None` for every supported rank.

Earlier iterations of this branch used `smallvec::SmallVec` and a hand-rolled
`MaybeUninit` container; the maintainer decision in #312 replaced both with
`tinyvec`, and no `unsafe`/`MaybeUninit` remains in the container.

### Construction

`arrayvec_from_slice` fills a `[T; INLINE_RANK]` and wraps it with
`ArrayVec::from_array_len`. The prefix is written with literal-count arms
rather than `copy_from_slice` or a loop, because LLVM idiom-recognizes a
runtime-length copy and lowers it to a `memcpy` PLT call, which dominated the
few-word case in disassembly. An `array::from_fn` variant was measured slower
for `new` and was dropped.

### Size

| type | size |
| --- | ---: |
| `StridedView<f64>` | 184 B |
| `StridedViewMut<f64>` | 184 B |
| `StridedArray<f64>` | 184 B |

The old `Cow` view was 80 B. The 8 B over the two 72 B `ArrayVec`s is the
`Option<Box<SpilledMeta>>` niche pointer.

### Spill rule

Rank 1 through `INLINE_RANK` stays inline and the `spilled` slot is `None`.
Above it `SpilledMeta` boxes both vectors; it never errors and never truncates.
`strided-view/tests/view_alloc.rs`
(`above_inline_rank_spills_gracefully`) records rank-9 `new` and `clone` as
three allocations (the `SpilledMeta` box plus one slice per vector), and the
P5 helpers as one each.

## Measured cost

AMD EPYC 7713P (64 cores), Linux x86-64, rustc 1.97.1, release,
`taskset -c 56`, 4e5 iterations, min over eight interleaved rounds of the
worktree binary and an `origin/main` binary. The host is shared and was under
varying load (load average 5–10 during the final run); the reported minima are
robust to the contention, but the absolute numbers are not a controlled
benchmark.

| operation | 0.5.1 (`Cow`) | inline | ratio |
| --- | ---: | ---: | ---: |
| `StridedView::new` rank 2 | 5.63 ns | 15.29 ns | 2.72x |
| `StridedView::new` rank 4 | 7.33 ns | 17.13 ns | 2.34x |
| `StridedView::new` rank 8 | 11.66 ns | 22.00 ns | 1.89x |
| borrowed `clone` rank 4 | 1.62 ns | 2.71 ns | 1.67x |
| borrowed `conj` rank 4 | 1.62 ns | 2.71 ns | 1.67x |
| `permute` rank 3 | 50.78 ns | 61.37 ns | 1.21x |
| `broadcast` rank 3 | 29.86 ns | 39.40 ns | 1.32x |
| `diagonal_view` rank 3 | 63.68 ns | 74.70 ns | 1.17x |
| `StridedArray::col_major` rank 3 | 44.24 ns | 38.06 ns | 0.86x |
| `StridedArray::view` rank 3 | 0.81 ns | 2.71 ns | 3.34x |
| `map_into` 4x4x4, views built per call | 92.73 ns | 88.75 ns | 0.96x |
| `map_into` 4x4x4, views reused | 86.68 ns | 81.14 ns | 0.94x |

Reading of the table:

- **The pointer-copy regressions that motivated this branch are closed within
  budget.** Borrowed `clone`/`conj` and `as_view`/`view` are 2.7 ns, inside the
  3.5 ns target (the old `Cow` paths were O(1) pointer copies of *borrowed*
  metadata; the inline paths copy a `Copy` fixed array).
  `clone`/`conj` of a *derived* view used to allocate twice and is now
  allocation-free.
- **Small kernel calls are at or below parity** (`map_into` 0.96x with views
  built per call, 0.94x with views reused).
- **`new` is 1.9–2.7x.** The ≤1.5x target is waived by the maintainer: `Cow::new`
  stored two borrowed pointers into an 80 B view, while the inline constructor
  must materialize up to 16 words into a 184 B view and the benchmark's result
  cannot be elided. The residual is the required copy, not fixable overhead
  (the `memcpy` PLT calls are gone from the disassembly).
- **Derived views are 1.2–1.3x** (`diagonal_view` 1.17x, `permute` 1.21x,
  `broadcast` 1.32x) instead of the ~1.0x the earlier `SmallVec` layout
  reached. The gap is the `ArrayVec::from_array_len` fill, which initializes
  the full fixed array before `ArrayVec` sets the length; the container is the
  mandated one and the paths stay allocation-free. `col_major` is 0.86x.
- `new` and the derived-view builds pay a one-time metadata construction; the
  per-op `map_into` arms that build views per call are below the `Cow` cost.

No kernel strategy, planner threshold or fast path changed here; the planner
inline work is #306 item 2 (unit B2). The `index_order` callers inside
`strided-basic`/`strided-perm` still convert to `Vec` at the boundary, so those
planner paths are untouched.

## Verification

- `strided-view/tests/view_alloc.rs`: counting global allocator. Zero
  allocations for `new`, `clone`, `conj`, `permute`, `broadcast`,
  `transpose_2d`, `diagonal_view`, `StridedViewMut::permute`, `as_view`, the
  P5 helpers and `StridedArray::view`/`view_mut`/`permuted` at ranks 1..=8;
  the rank-9 spill is recorded.
- Doc examples on `INLINE_RANK`, `StrideVec`, `AxisVec`,
  `col_major_strides`, `row_major_strides`, `index_order`,
  `StridedView::new`/`new_unchecked`, `StridedViewMut::new`/`new_unchecked`/
  `as_view`, `StridedArray::col_major`/`row_major`.
- `cargo fmt --all -- --check`, `cargo test -j 16 --profile ci`,
  `cargo test -j 16 --profile ci -p strided-basic -p strided-kernel -p
  strided-fused --features parallel`, `cargo test -j 16 -p strided-basic
  --no-default-features`, `cargo doc -j 16 --workspace --no-deps`.
