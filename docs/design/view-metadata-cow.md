# Borrowed view metadata

Date: 2026-10-12 (branch `refactor/view-metadata-cow`)

## Context

`StridedView`, `StridedViewMut` and `StridedArray` stored their shape and
strides as `Arc<[usize]>` / `Arc<[isize]>`. Every construction therefore made
two heap allocations and, on drop, two atomic refcount operations. Measured on
an AMD Ryzen AI 9 HX 470 (`taskset -c 2`, release, 1e6 iterations, drop
included): `StridedView::new` 24-25 ns, of which 22 ns is the two `Arc::from`
calls; the allocation-free `RawStridedRef::new` costs 1-2 ns.

The cost is visible end to end in a host that builds a view per call. One
prepared `BinaryPlan<f64>` for `D[i,l] = sum_j A[i,j] B[j,l]` on 2x2
column-major operands, warm `Exec`, 1T:

| route | ns/call |
| --- | --- |
| `execute_into` with three views built per call | 92.5 |
| `execute_slices` (no view) | 18.3 |

The three views alone cost 74.0 ns. The same effect is recorded in
`cpueinsum-rs/docs/worklogs/2026-10-05-initial-implementation.md` (chi=2 MPS
overlap: 126 ns/step with views, 55 ns/step with slices).

`strided-basic` had already reacted to this by adding `raw_ops.rs`, whose
entry points take the allocation-free `RawStridedRef`/`RawStridedMut` borrowed
descriptors, with the documented rationale that the Arc-owning view's fixed
cost dominates small replays. `docs/omeinsum-merge-analysis.md` states the
principle directly: "Borrowing (`&'a [T]`) achieves the same zero-copy
semantics without atomic refcount overhead. Arc's benefit is ergonomic (no
lifetime annotations), not performance."

## Decision

A view's metadata is `Cow<'a, [usize]>` / `Cow<'a, [isize]>`:

- `StridedView::new`, `StridedViewMut::new` and the `new_unchecked` pair borrow
  the caller's layout: no allocation, no atomic. The returned view's lifetime
  ties the metadata to the data, exactly as `RawStridedRef` already did.
- `StridedView::new_owned` / `new_owned_unchecked` take the layout by value, for
  the callers that compute it next to the view (a helper returning a view, or a
  kernel that derives broadcast strides).
- Derived views (`permute`, `broadcast`, `diagonal_view`, `transpose_2d`,
  `adjoint_2d`, `StridedViewMut::permute`, `StridedArray::permuted`) own what
  they compute, as before, but with `Vec` instead of `Arc`. The atomic's only
  remaining job on that path was making `clone`/`conj`/`as_view` of a computed
  layout cheap; `as_view` now borrows instead, and `clone`/`conj` copy the
  layout, which is O(rank) and only on a computed view. A view over borrowed
  metadata copies nothing at all, where it previously bumped a refcount.
- `StridedArray`'s metadata is `Vec` for the same reason: it owns its data, and
  its `Clone` already deep-copies that data.
- `RawStridedRef`/`RawStridedMut` are unchanged; their `as_view`/`as_view_mut`
  conversions now borrow the descriptor's metadata instead of allocating.

`Arc` no longer appears in `strided-view`.

## Alternatives rejected

- **`Vec` instead of `Arc`, keeping owning construction on every path.** The
  smallest possible diff, and it reaches zero `Arc`, but the hot constructor
  still makes two allocations: the measured `StridedView::new` cost would fall
  from ~25 ns to ~11 ns rather than to ~2 ns, and a host's per-call view cost
  would stay dominant. It does not meet the objective that motivated the change.
- **Borrowed-only metadata with `*_into` derived views and no owning mode.**
  Smaller type, `Copy` views, but it forces every derived-view call site to
  supply metadata storage. `tenferro-cpu/src/structural.rs` alone has six
  `permute`/`broadcast` call sites on views, and `tenferro-cpu-basic` and
  `tenferro-cpu/src/structural.rs` each have a helper that returns a view built
  from locally computed strides; `Cow` keeps all of them working unchanged.
- **Merging `RawStridedRef`/`RawStridedMut` into the view types.** They are
  duplicates in shape, but the split also carries the two kernel strategies
  (`ops_view.rs` traversal plans vs `raw_ops.rs` stack-fused loops). That merge
  is performance-gated work with its own benchmark obligations and is not
  needed to remove `Arc`.

## Consequences

- Views whose metadata is computed inside a helper must use the owned
  constructor; a `&[expr, ...]` temporary that is not a promoted constant no
  longer compiles. Test call sites were hoisted into locals.
- `StridedView::clone()` now copies a borrowed metadata reference instead of
  bumping a refcount, and copies the `Vec` for an owned view. `conj()` behaves
  the same way: free for borrowed metadata, two small allocations for computed
  metadata (reachable from `strided-basic/src/ops_view.rs` `copy_conj` when its
  source is a permuted or broadcast view).
- The layout a view borrows must outlive the view. Callers that keep shapes and
  strides next to their data (the normal case, and what `RawStridedRef` callers
  such as `tlinalg` already do) are unaffected.

## Verification

- `strided-view/tests/view_alloc.rs`: a counting global allocator asserts that
  borrowed construction of an immutable view, a mutable view, a clone and an
  `as_view` reborrow makes zero allocations, and that owned construction makes
  exactly one allocation per metadata vector. The counting-allocator pattern is
  taken from `strided-kernel/tests/copy_plan_alloc.rs` in this repository.
- `cargo test --workspace` and `cargo fmt --all -- --check`.
- No kernel strategy, threshold or fast path changed, so no benchmark row is
  added and no performance claim is made here beyond the construction
  measurements above.

## Measured after

AMD Ryzen AI 9 HX 470, Linux x86-64, rustc 1.99.0, release, `taskset -c 2`
(1T), 1e6 iterations, `strided-view/examples/view_cost_portable.rs` — the same
file compiled against `origin/main` (0.4.8) and this branch (0.5.0), median of
three rounds:

| | 0.4.8 | 0.5.0 |
| --- | --- | --- |
| one `StridedView::new` | 26.8 ns | 2.7 ns |
| three views (a, b, d) | 81.3 ns | 7.7 ns |
| `RawStridedRef::new` (untouched control) | 2.3 ns | 2.3 ns |

End to end in the caller, one warm prepared `BinaryPlan<f64>` over 2x2
column-major operands with `Exec::serial_with_workspace` (explicit 1T),
`cpueinsum-rs/crates/cpueinsum/examples/op_overhead.rs`:

| route | 0.4.8 | 0.5.0 |
| --- | --- | --- |
| `execute_into`, three views built per call | 92.5 ns/call | 20.2 ns/call |
| `execute_slices` (no view) | 18.3 ns/call | 18.7 ns/call |
| wrapper delta | 74.2 ns | 1.6 ns |

The three views alone fall from 74.0 ns to 5.2 ns, which is the whole of the
delta; the call is no longer distinguishable from the slice path.
