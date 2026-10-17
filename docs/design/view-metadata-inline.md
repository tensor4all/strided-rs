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

`strided-view` defines the capacity once and stores shape/strides in a small
hand-rolled container instead of `smallvec::SmallVec`:

```rust
pub const INLINE_RANK: usize = 8;
pub type StrideVec = SmallVec<[isize; INLINE_RANK]>; // P5 helper return types
pub type AxisVec = SmallVec<[usize; INLINE_RANK]>;

#[derive(Clone, Copy)]
struct InlineVec<T: Copy> {
    len: u8,
    buf: [MaybeUninit<T>; INLINE_RANK],
}

#[derive(Debug)]
struct ViewMeta {
    inline_dims: InlineVec<usize>,
    inline_strides: InlineVec<isize>,
    spilled: Option<Box<SpilledMeta>>, // Some only above INLINE_RANK
}
```

`StridedView`, `StridedViewMut` and `StridedArray` each hold one `ViewMeta`.

- The inline case is a plain `Copy` `{ len, buf }` whose buffer is
  `MaybeUninit`, so only the live prefix is ever written and the value is
  still `Copy`. `Clone`/`view`/`as_view`/`conj` are a fixed-size bitwise copy
  of the two `InlineVec`s plus a `None` check, not a generic `SmallVec`
  traversal or a runtime-length copy.
- `new`/`new_unchecked`/`from_parts`/array constructors copy at most
  `2 * INLINE_RANK` words into the view. The view's lifetime is tied to the
  element data only; the layout does not have to outlive it.
- `new_owned` / `new_owned_unchecked` are removed. Downstream the same
  breakage is budgeted in #306 (tcontract `tests/batched.rs`,
  `benchmarks/.../exec_entry.rs`).
- `RawStridedRef` / `RawStridedMut` / `ErasedRawStrided*` stay borrowed: they
  are the prepared-replay descriptors and must not copy.
- P5 helpers keep returning `StrideVec`/`AxisVec` (`SmallVec`); they are not on
  the hot view path.

### Why not `SmallVec`, and why not an enum

Two alternatives were measured and rejected:

- **`SmallVec<[T; INLINE_RANK]>`.** Its generic `from_slice`/`clone`
  element-walk and capacity bookkeeping made borrowed `clone`/`conj` ~11–13 ns
  and `StridedArray::view` ~12 ns (8x and 14.7x the `Cow` numbers).
- **An owning enum `{ Inline { len, buf }, Heap(Box<[T]>) }`.** This avoids the
  `SmallVec` generic path, but the enum has drop glue and a discriminant, so
  the same `clone`/`conj`/`view` still measured ~11–13 ns. A `Copy` inline
  struct with the heap fallback in a separate `Option<Box<..>>` field, and a
  `MaybeUninit` buffer that only writes the live prefix, brings them to ~1.1 ns
  (faster than the `Cow` baseline).

`ViewMeta::from_slices` dispatches on `dims.len() <= INLINE_RANK`. Inline
construction uses literal-count `MaybeUninit::write` calls rather than
`copy_from_slice` or a loop: LLVM idiom-recognizes a runtime-length copy and
lowers it to a `memcpy` PLT call, which dominated the few-word case in
disassembly.

### Size

| type | size |
| --- | ---: |
| `StridedView<f64>` | 184 B |
| `StridedViewMut<f64>` | 184 B |
| `StridedArray<f64>` | 184 B |

The old `Cow` view was 80 B. The 8 B over the two 72 B `InlineVec`s is the
`Option<Box<SpilledMeta>>` niche pointer; there is no capacity word and no
per-collection heap discriminator.

### Spill rule

Rank 1 through `INLINE_RANK` stays inline and the `spilled` slot is `None`.
Above it `SpilledMeta` boxes both vectors; it never errors and never truncates.
`strided-view/tests/view_alloc.rs`
(`above_inline_rank_spills_gracefully`) records rank-9 `new` and `clone` as
three allocations (the `SpilledMeta` box plus one slice per vector), and the
P5 helpers as one each.

## Measured cost

AMD EPYC 7713P (64 cores), Linux x86-64, rustc 1.97.1, release,
`taskset -c 56`, 3e5 iterations, min over ten interleaved rounds of the
worktree binary and an `origin/main` binary. The host is shared and was under
varying load (load average 13–38 during the run); the reported minima are
robust to the contention, but the absolute numbers are not a controlled
benchmark.

| operation | 0.5.1 (`Cow`) | inline | ratio |
| --- | ---: | ---: | ---: |
| `StridedView::new` rank 2 | 5.15 ns | 16.46 ns | 3.20x |
| `StridedView::new` rank 4 | 7.32 ns | 17.31 ns | 2.36x |
| `StridedView::new` rank 8 | 11.66 ns | 21.32 ns | 1.83x |
| borrowed `clone` rank 4 | 1.62 ns | 1.08 ns | 0.67x |
| borrowed `conj` rank 4 | 1.62 ns | 1.08 ns | 0.67x |
| `permute` rank 3 | 50.74 ns | 52.98 ns | 1.04x |
| `broadcast` rank 3 | 29.85 ns | 31.50 ns | 1.06x |
| `diagonal_view` rank 3 | 63.42 ns | 41.74 ns | 0.66x |
| `StridedArray::col_major` rank 3 | 44.26 ns | 31.50 ns | 0.71x |
| `StridedArray::view` rank 3 | 0.81 ns | 1.08 ns | 1.33x |
| `map_into` 4x4x4, views built per call | 92.29 ns | 87.18 ns | 0.94x |
| `map_into` 4x4x4, views reused | 84.96 ns | 82.21 ns | 0.97x |

Reading of the table:

- **The pointer-copy regressions that motivated this revision are closed.**
  Borrowed `clone`/`conj` and `as_view`/`view` are 1.1 ns — faster than the
  `Cow` baseline. The old `Cow` paths were O(1) pointer copies of *borrowed*
  metadata; the inline paths now copy a `Copy` fixed array bit for bit.
  `clone`/`conj` of a *derived* view used to allocate twice and is now
  allocation-free.
- **Small kernel calls are at or below parity** (`map_into` 0.94x with views
  built per call, 0.97x with views reused).
- **Derived views are at parity or faster** (`permute`/`broadcast` ~1.05x,
  `diagonal_view` 0.66x, `col_major` 0.71x) and all allocate nothing.
- **`new` remains 1.8–3.2x.** This is the one target the custom type did not
  reach. `Cow::new` only stored two borrowed pointers and returned an 80 B
  view; the inline constructor must materialize up to 16 words into a 184 B
  view, and the benchmark's `new` result cannot be elided the way the borrowed
  result could. The metadata build itself (`ViewMeta::from_slices`, two
  `InlineVec` builds) is the remaining cost; a disassembly check confirmed the
  `memcpy` PLT calls are gone, and a control that forces the equivalent raw
  184 B build to materialize measures the same ~15 ns floor. In practice
  the constructor cost is paid once per view and any small-op dispatch floor
  that builds several views per call reads the `map_into` arm, which is at
  parity. Closing the remaining gap would need a borrow-on-rank-`<= INLINE_RANK`
  constructor that the issue explicitly rules out (the view lifetime must
  become the data borrow only).

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
