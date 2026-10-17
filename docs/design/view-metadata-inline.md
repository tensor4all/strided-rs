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

The metadata of `StridedView`, `StridedViewMut` and `StridedArray` is
`SmallVec<[usize; INLINE_RANK]>` / `SmallVec<[isize; INLINE_RANK]>`, with
`smallvec`'s `union` representation and `INLINE_RANK` defined once in
`strided-view`:

```rust
pub const INLINE_RANK: usize = 8;
pub type StrideVec = SmallVec<[isize; INLINE_RANK]>;
pub type AxisVec = SmallVec<[usize; INLINE_RANK]>;
```

- `StridedView::new` keeps its slice-taking signature and copies at most
  `2 * INLINE_RANK` words into the view. The view's lifetime is now tied to
  the element data only; the layout does not have to outlive it.
- `StridedView::new_owned` / `new_owned_unchecked` are removed. Every
  in-tree caller computes a layout next to the view, and the copy makes the
  owned constructor unnecessary. Downstream the same breakage is budgeted in
  #306 (tcontract `tests/batched.rs`, `benchmarks/.../exec_entry.rs`).
- `RawStridedRef` / `RawStridedMut` / `ErasedRawStrided*` stay borrowed: they
  are the prepared-replay descriptors and must not copy.
- P5 helpers return inline vectors: `col_major_strides` / `row_major_strides`
  return `StrideVec` and `auxiliary::index_order` returns `AxisVec`.

### Why inline instead of `Cow`

`Cow` made the borrowed case free but left the owning case as a `Vec`. That is
the wrong split for this workspace: the hot paths are not "construct one view
once and read it forever", they are "build a derived view per call". A
permuted, broadcast or diagonal operand therefore allocated 2–3 times per call,
and a `clone`/`conj` of it allocated twice more. `SmallVec` makes all of those
O(rank) copies with no allocator traffic, which is the 0.6.0 contract.

### Why `union`

`SmallVec`'s `union` representation overlaps the inline array with the heap
pointer/capacity pair, so the two metadata vectors cost `INLINE_RANK` words
plus one length each. Without `union` the same view grows further and the
`new`/`clone` copy gets slower for no benefit. Measured on this machine:

| type | `union` |
| --- | --- |
| `StridedView<f64>` | 176 B |
| `StridedViewMut<f64>` | 176 B |
| `StridedArray<f64>` | 176 B |

The old `Cow` view was 80 B (`ptr`, `data` fat pointer, two `Cow`s, `offset`).

### Spill rule

Rank 1 through `INLINE_RANK` stays on the stack. Above it the `SmallVec`
spills to the heap exactly like a `Vec`; it never errors and never truncates.
The behavior is covered by
`strided-view/tests/view_alloc.rs`
(`above_inline_rank_spills_gracefully`): rank 9 `new` and `clone` make two
metadata allocations, `col_major_strides` and `index_order` one each.

## Measured cost

AMD EPYC 7713P (64 cores), Linux x86-64, rustc 1.97.1, `taskset -c 56`,
release, 1.5e6 iterations, min of three interleaved rounds. The host is shared
and was under varying load (load average 8–36 during the run), so the numbers
are indicative rather than a controlled benchmark; the 0.5.1 column is the same
binary built against `origin/main` in a sibling worktree.

| operation | 0.5.1 (`Cow`) | inline | ratio |
| --- | ---: | ---: | ---: |
| `StridedView::new` rank 2 | 5.15 ns | 11.94 ns | 2.3x |
| `StridedView::new` rank 4 | 7.35 ns | 13.80 ns | 1.9x |
| `StridedView::new` rank 8 | 12.27 ns | 17.91 ns | 1.5x |
| borrowed `clone` rank 4 | 1.63 ns | 12.96 ns | 8.0x |
| borrowed `conj` rank 4 | 1.72 ns | 11.93 ns | 6.9x |
| `permute` rank 3 | 50.79 ns | 31.82 ns | 0.63x |
| `broadcast` rank 3 | 29.85 ns | 13.89 ns | 0.47x |
| `diagonal_view` rank 3 | 63.96 ns | 34.35 ns | 0.54x |
| `StridedArray::col_major` rank 3 | 44.26 ns | 31.76 ns | 0.72x |
| `StridedArray::view` rank 3 | 0.81 ns | 11.92 ns | 14.7x |
| `map_into` 4x4x4, views built per call | 92.79 ns | 97.69 ns | 1.05x |
| `map_into` 4x4x4, views reused | 85.76 ns | 95.19 ns | 1.11x |

The operations that allocated before are now allocation-free and 1.3–2.1x
faster. The operations that were pure pointer copies before are slower, and
this is the deliberate trade-off the issue anticipated ("B1 adds inline-copy
cost; 80 → 176 B view"):

- **`new` (1.5–2.3x).** It used to copy two `Cow` discriminants and borrow the
  caller's layout; it now copies up to 16 words. The copy floor alone, measured
  with a bare `SmallVec::from_slice` pair at rank 4, is about 6.5 ns. Marking
  the constructors `#[inline]` recovered roughly 30–40% of the `new` gap.
- **borrowed `clone`/`conj` (7–8x), `StridedArray::view` (14.7x).** The old
  paths were O(1) pointer/`Cow` copies of *borrowed* metadata. The new paths
  copy up to `INLINE_RANK` words per vector. The same `clone`/`conj` on a
  *derived* view used to allocate twice and is now allocation-free, so the
  loss is confined to views that never allocated in the first place. This is
  the cost of making the type uniform and inline; there is no borrow-only view
  that shares the inline representation.
- **small kernel calls (5–11%).** The `map_into` arm pays the two view
  constructions when they are built per call, and the pre-built-view arm pays
  the copy of a larger view passed by reference/`black_box`. Both remain far
  from the allocation counts that motivated #306.

No kernel strategy, planner threshold or fast path changed here; the planner
inline work is #306 item 2 (unit B2). The `index_order` callers inside
`strided-basic`/`strided-perm` still convert to `Vec` at the boundary, so those
planner paths are untouched.

## Verification

- `strided-view/tests/view_alloc.rs`: counting global allocator. Zero
  allocations for `new`, `clone`, `conj`, `permute`, `broadcast`,
  `transpose_2d`, `diagonal_view`, `StridedViewMut::permute`, `as_view`, the
  P5 helpers and `StridedArray::view`/`view_mut` at ranks 1..=8; the rank-9
  spill is recorded.
- Doc examples on `INLINE_RANK`, `StrideVec`, `AxisVec`,
  `col_major_strides`, `row_major_strides`, `index_order`,
  `StridedView::new`/`new_unchecked`, `StridedViewMut::new`/`new_unchecked`/
  `as_view`, `StridedArray::col_major`/`row_major`.
- `cargo fmt --all -- --check`, `cargo test -j 16 --profile ci`,
  `cargo test -j 16 --profile ci -p strided-basic -p strided-kernel -p
  strided-fused --features parallel`, `cargo test -j 16 -p strided-basic
  --no-default-features`, `cargo doc -j 16 --workspace --no-deps`.
