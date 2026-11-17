# Changelog

Crates: `strided-traits`, `strided-view`, `strided-perm`, `strided-basic`,
`strided-kernel`, `strided-fused`, `strided-nn`. Per-release worklogs live in
`docs/worklogs/`.

## 0.15.0 (unreleased) — minor

`0.14.0` (and `strided-view` 0.8.0) is the published line. The tree already carries this section's unreleased `0.15.0`
manifests for `strided-basic`, `strided-nn`, `strided-fused`, `strided-perm` and `strided-kernel`, and `strided-view`
0.8.1; the inherited workspace version is `0.7.2`. The parked `0.14.1` patch (#452) was superseded by the #468 bump and
never published, so a new parameter on `copy_into_col_major` is a minor that lands in this unreleased section, and those
manifests move when it is released.

### `strided-perm`: `copy_conj_into` and `copy_conj_into_uninit`

The permutation engine gained a conjugating copy for the complex element types (#467):
`copy_conj_into(dest, src, exec)` and `copy_conj_into_uninit(dest, src, exec)` apply `conj`
as each element is stored, the way HPTT's `conjA` template parameter does, so one plan
serves both stores and the conjugate costs no extra pass over memory. Validation, the
error set, the width decision and the plan are those of `copy_into` / `copy_into_uninit`;
a conjugating copy of a contiguous layout therefore takes the element store instead of
`memcpy`. `T: ElementOpApply` (re-exported by `strided-view`) is the bound, so a real
element type is accepted and copied unchanged through the generic element store: use
`copy_into` for those.

`strided_basic::copy_conj_into` now routes a non-contiguous complex copy to that entry when the engine is measured to
beat the element map, and keeps the element map otherwise (#467): a `Complex32`/`Complex64` copy whose source and
destination layouts differ and whose packed footprint (`total_len(dims) * size_of::<T>()`, not the address span of the
strided view) is below 24 MiB. The bound is measured, not guessed: `[64,64,64]`, the 32^3-96^3 3D rows, `[100,2,100]`,
112^3 and the 128^2-1024^2 2D rows are up to 2.7x faster on the engine, while contiguous copies stay on the map
1.2-2.7x faster there, and the 26-32 MiB 3D rows (120^3-128^3) keep the map where it is 1.09-1.15x faster.
The predicate is [`conj_perm_engine_route`](strided-basic/src/ops_view.rs), one measured constant plus two
layout/type properties. The two measured tables — one on the base the routing was developed against, one on the base
that carries the engine's L1-aliasing fix (#480) — label their rows, and `CONJ_ROUTE_GATE=1` passes on the current
base. `copy_conj_into` gains a `'static` bound (as `copy_into_uninit` and `copy_transpose_scale_into` already have):
this is breaking for a caller whose element type is not `'static`, and it is what lets the type be tested and handed
to the engine. No other signature changes, and real element types keep the map.
The two tables, the protocol and the residual classes are in
[`docs/worklogs/2026-11-15-issue-467-conj-route.md`](docs/worklogs/2026-11-15-issue-467-conj-route.md), and
`strided-basic/benches/conj_route_467.rs` reproduces them (`CONJ_ROUTE_GATE=1` gates on them).

### `strided-perm`: the permutation fallback stops thrashing an L1 set (#466)

`copy_into` was 4.5x slower per element when the destination's strided-axis byte
stride was a multiple of the 4 KiB L1 way: every row of a macro tile then lands
in one set, and the fallback kernel walked the tile with `j` outermost, which
asks that set to hold `block_a` lines at once and refetches each for write
allocation.

The scalar fallback now decides per tile from the two strides it is handed. A
destination-only alias walks `i` outermost, writing each row's contiguous run in
one visit; when the source row aliases as well — a square transpose — the loop
order stays and the `a` extent is cut to six rows instead, the budget
`strided-basic`'s `block::fit_l1_sets` spends (#287), since moving the thrash to
the reads is not a win when the source collides too. Otherwise nothing changes.
Same plan, same blocking, same `f64`/`f32` micro-kernels, no API change.

Every row of #466's 2D and 3D tables now runs at or below its `2^d ± 1`
neighbours (`[1024, 32]` 5.75 -> 1.11 ns/element, `[96,96,96]` 5.56 -> 3.12,
`[32,32,32]` 5.67 -> 1.14, measured on a paired `main`-vs-branch run); `f64`
never enters this kernel and does not move (0.99-1.03x). Nothing regresses at
four threads either: the square power-of-two transposes this branch also had to
protect come out at 0.33x-0.98x of `main` in the `conj_copy` table, and no cell
of that table is above `main` outside 1%. Host, protocol, both tables and the
four rejected variants are in
[`docs/worklogs/2026-11-14-issue-466-l1-set-order.md`](docs/worklogs/2026-11-14-issue-466-l1-set-order.md).

### `strided-basic`: `copy_into_col_major` takes its `Exec` token

`strided_basic::copy_into_col_major(dst, src)` hard-coded `Exec::serial()` and handed that to the permutation engine, so a
4096x4096 transpose asked for on a four-slot pool ran on one core and the signature said nothing about it (#212). It now
takes the token last, like `copy_into` and every other entry that can fan out ([`docs/threading-contract.md`]), and
forwards the caller's token unchanged.

```rust
// before
copy_into_col_major(&mut dst, &src)?;
// after
copy_into_col_major(&mut dst, &src, exec)?;   // Exec::serial() restores the old route
```

Same result, same layout rules; only the route changes. `strided-perm::copy_into_col_major` already took an `Exec`.

[`docs/threading-contract.md`]: https://github.com/tensor4all/strided-rs/blob/main/docs/threading-contract.md

### `strided-basic`: `replay_*_raw` allocate nothing at any rank on a serial token

`replay_copy_raw`, `replay_copy_raw_uninit`, `replay_map_raw` and `replay_map_raw_uninit` allocated rank-sized buffers
above `RAW_FUSED_RANK_LIMIT` (8) axes: the axis list and range cursor, and, when the cache-block model ran, its slices,
stride-order table, costs, blocks and the tile walk's extents and origins (#469). They now hold that state in stack
arrays bounded by the moving-axis count, which a checked element count bounds by `usize::BITS - 1`, and allocate
nothing on a successful serial call at any rank, or on a pooled call whose replay is not tiled. A pooled tiled replay
with more than 8 moving axes may still allocate the dimension splitter's metadata. `RAW_FUSED_RANK_LIMIT` now only
selects the inline or the wide traversal. Serial `CopyPlan`/`UpdatePlan` execution of a spilled layout stops
allocating too; `copy_scale_raw`, `axpy_raw` and plan compilation keep their fusion scratch.

### `strided-basic`: `replay_map_raw_fused` and `replay_map_raw_uninit_fused`

The two additive entries fuse the pair inside the call (`copy_scale_raw`'s
rule, sorted by destination stride and merged where contiguous) and then replay
through the same allocation-free route as `replay_map_raw`, so a caller holding
unfused axes no longer pays its own fusion pass per call (#471). The fusion
scratch is two fixed stack arrays; a successful call allocates nothing at any
rank on a serial token, or on a pooled token whose replay is not tiled above
`RAW_FUSED_RANK_LIMIT` moving axes. The given-axis entries are unchanged and
remain the cheaper choice when the axes are already fused.

### `strided-basic`: `replay_map_raw_serial`

The additive entry `replay_map_raw_serial(dest, src, f)` is `replay_map_raw` on the calling thread, with only
`D: Copy`, `S: Copy` and `F: FnMut(&mut D, S)`: no `Send`, no `Sync`, no `Exec` (#469). A generic `T: Copy` copy is
`replay_map_raw_serial(dst, src, |d, s| *d = s)`. It shares the route and the leaves of `replay_map_raw` under
`Exec::serial()`, whose serial token now walks that same function, so the traversal, the errors and the visit order are
the same: column-major over the given axes, or tile by tile for a destination proven injective that the cache-block
model tiles. A successful call allocates nothing at any rank. There is no `_uninit` form: with `D = MaybeUninit<T>` the
destination is initialized when `f` writes every slot. The parallel entries keep their bounds.

### `strided-basic`: `RunSumPlan`, the sorted-run reduction (#479)

`RunSumPlan` computes `dest[r, ..] (=|+=) Σ_{j in ptr[r]..ptr[r+1]} src[perm[j], ..]` over CSR runs, the edge -> node
accumulation of a message-passing loop. `RunIndex::from_dest(dest, num_runs)` derives that topology from an unsorted
destination list with a stable counting sort, so rows keep their order inside a run; `RunIndex::from_ptr(ptr)` takes
offsets a caller already grouped. Both allocate at topology time, and one index can be replayed by any number of plans.

`RunSumPlan::compile(src_dims, src_strides, dest_dims, dest_strides, &runs)` proves the layout facts it can without the
buffers — matching strides and ranks, the two row extents against `runs`, the trailing extents, a checked element count,
and an injective destination layout — and fuses the payload traversal. `execute_into` / `execute_acc` take a
`RawStridedMut`, a `RawStridedRef`, the index and an `Exec`, and allocate nothing while the payload fuses to at most
eight axes; a payload of more than eight fused axes walks through `kernel_nd_inner_iterative`, whose cursor is a `Vec`
by contract, as in the `replay_*` entries.

A run is summed by exactly one worker, in increasing `j`, and destination rows are disjoint between runs, so the
partition cannot change a result: every budget produces identical bits, on `f32`, `f64` and complex types alike. A run
with no rows writes zeros under `execute_into` and adds nothing under `execute_acc`; the two agree bit for bit on a
destination seeded with `T::zero()`.

Two error variants carry the new preconditions, `StridedError::InvalidRunIndex` for offsets that do not start at zero or
step downwards, and `StridedError::IndexOutOfRange { index, runs }` for a destination index outside `0..runs`. Validation
reuses the existing `StrideLengthMismatch`, `RankMismatch`, `ShapeMismatch`, `OffsetOverflow`,
`NonInjectiveOutputLayout` and `PlanLayoutMismatch`.

The two `strided-view` variants are an addition to a `StridedError` that is not `#[non_exhaustive]`, so they break a
downstream exhaustive `match`. `strided-view` is still unpublished in the tree (`scripts/published_identity.py` reports
`unpublished strided-view 0.8.1`), so no published content changes; whether that body is published as it stands or as a
`strided-view` minor, and whether `StridedError` gains the attribute, are release decisions for the maintainer.

This is the typed half of #479, and the executor that consumes the index for its scattered outputs is the
`strided-segpoly` section below. The dtype-erased replay (`strided-kernel`) is not in it.

### `strided-segpoly`: the segmented-polynomial executor (#479 part 2)

A new unpublished crate (`publish = false`, so no published content changes) executes a lowered segmented-polynomial
descriptor. `SegPolyPlan::compile` takes the operations, paths and per-buffer `IndexMode`s together with the inputs' and
outputs' layouts, and `execute_into` / `execute_acc` compute
`out[..] (=|+=) Σ_path coeff_path · Π_operand operand[..]` over one batch axis, with that axis read out of a gather list,
a `RunIndex` scatter or a single shared row instead of being the row itself. The lowering this consumes is the one a
cuEquivariance-shaped front end produces (one undefined primal per output, `num_segments × u` payload); producing it is
out of scope, reading it is the whole crate.

Contract points that shape the code. At most one batch axis, else `MultiBatchAxis`. `f32`/`f64` only, with coefficients
rounded at compile time to the element type (`SegPolyElem` is sealed). A negative or out-of-range gather/scatter index,
and a batch extent that disagrees between buffers, are typed errors rather than panics. A scattered destination reads
through the existing `RunIndex`, so the row mapping is the stable one `sort_paths` derives. An empty run writes zero
under `execute_into` and adds nothing under `execute_acc`. Scratch is the caller's, sized by
`scratch_len(token budget)`, and a call whose scratch is too small fails with `ScratchTooSmall` instead of truncating a
sum. Inputs may alias each other; an output may not alias any other buffer, including another output. `compile` rejects
what the layouts alone can show — an output whose own elements repeat is `NonInjectiveOutputLayout` — while two buffers
over the same storage stay the caller's obligation, as `run_sum.rs` states it for its plan.

The summation order is part of the result, not an implementation detail. Rows are visited inside their unit in run
order, paths in descriptor order, rows grouped in fixed `BLOCK_ROWS = 256` blocks whose partials are combined left to
right after the fork joins. The result is therefore bitwise invariant under the `Exec` budget, the worker count and the
layout the caller reaches the same array through — not under a change of `BLOCK_ROWS`, which is a constant for that
reason. The sign of zero in a later block's partial is the one documented exception. One
`tree_width(..., KernelClass::Map, macs)` fork per call, children partitioned by cumulative row count, no nested forks,
no atomics and no locks; a counting-allocator test asserts that `execute_into` and `execute_acc` allocate nothing on a
serial token.

Tests are generated, never hand-written: `tests/fixtures/generate.py` drives a pinned cuEquivariance commit
(`9ed8c7613a79861415071027249cc458e423be30`) to lower each descriptor and to compute the expected output with its own
naive executor, and checks in the `.npy` seeds plus a generated `cases.rs`. The cases cover symmetric contractions of
degree 1, 2 and 3, spherical harmonics (uniform lane extent 1), a jvp and the backward pass of a degree-2 case, and a
gather+scatter message-passing case with a run of several rows. On top of those, finite differences against the
analytic derivative, the transpose as a discrete adjoint, one-token and n-token bitwise equality, and the rejection
table. `benches/executor.rs` measures the fused path against the two-pass alternative (materialise every product, then
`RunSumPlan`).

#### A dispatched lane kernel and a lane reduction (#496)

The lane loop is no longer left to auto-vectorisation. Operations with one to three operands run a
`pulp::Arch::new().dispatch(...)` body (`src/simd.rs`: `AxpyF64` / `AxpyF32`, `f64` 8 and `f32` 16 lanes on an
AVX-512 host, `1` on a scalar target), so a baseline `x86-64` build reaches AVX2 and AVX-512 at run time with no
`target_arch` case, no width constant and no `-C target-cpu` requirement; `scripts/check-runtime-simd-dispatch.py`
disassembles both levels, and `lane_width::<T>()` reports the width a call selected. The body multiplies and adds as
two operations -- never `mul_add` -- and keeps the coefficient-first association, so its result is bit-identical to the
scalar loop; an operand with a lane extent of 1 is splatted and one with a lane stride other than 0 or 1 is copied into
a per-worker tile first. Four-or-more-operand operations keep the scalar loop.

A destination whose own lane extent is 1 while the inputs' is `u` used to be rejected (`OutputLaneBroadcast`) and is now
the lane reduction it reads as: it sums each row's `u` lanes in increasing lane order after the block partials are
combined, which extends the documented order by one rule and leaves the rest in place. The VJP wrt the spherical
harmonics of a MACE layer-1 CG block, previously 11 008 scalar paths over `N x 5 120` rows, is 86 paths of 128 lanes:
11-16x faster in the layer-1 A/B and no `[E, 16, 128]` intermediate, with `f64` `bar_y` at the rounding level of the
sum (1.7e-15 -> 2.2e-15 relative to the scale at 128 atoms) and `f32` better (3.4e-6 -> 1.0e-6).
`scratch_len` now also covers one pack area per worker and reports the new floor through `ScratchTooSmall`. The
measurements, the per-cell before/after tables, the allocation evidence and the thread sweep are in
[`docs/worklogs/2026-11-17-issue-496-segpoly-simd-reduce.md`](docs/worklogs/2026-11-17-issue-496-segpoly-simd-reduce.md).

### `strided-view` 0.8.1: `auxiliary::index_order_into`

`index_order_into(strides, out)` writes the stride order into caller storage and returns `StrideLengthMismatch` when
the lengths differ; `index_order` is its allocating form and is unchanged. The injectivity check's difference search no
longer allocates when it enumerates more than eight axes.

## 0.14.1 (superseded by 0.15.0) — patch

`0.14.0` shipped on crates.io on 2026-11-08. Since then `main` has changed shipped files in all six
crates, so the weekly `Packaged crate tests` job's `published content identity` step (#409) reports
`6 mismatch(es)` at `34ec19c` (run 42670871078); the same check run locally
(`python3 scripts/published_identity.py .`) reports the same six. The version number no longer
identified one body of content, so this patch was to move it. It was superseded before publication: #468 moved the
five retained crates straight to `0.15.0`, and the changes listed below are part of that release.

What changed under 0.14.0:

- **SAFETY documentation** (#219 wave): the `unsafe` blocks in `strided-view`, `strided-perm`,
  `strided-basic` (including `map_view.rs` and `ops_view.rs`), `strided-kernel`, `strided-fused` and
  `strided-nn` name the site that validates them (#435, #438, #440–#442, #444, #446, #448, #449).
  That is what those PRs say they changed; the identity check compares bytes, not intent. Apart from
  two new unit tests in #449 and a rustfmt reflow in #440, those diffs change comments only (checked
  by removing comment lines from `git show` of each merge commit).
- **Fixes**: `CopyPlan` costed by its footprint (#424), per-operand element size in the block model
  (#431), tile placement for the transposed, spilled and view paths (#422, #429, #433), tiled
  segments in `ConcatenatePlan`'s whole-plan parallel replay (#432), alignment of zero-length erased
  buffers (#436), and the `get_unchecked` index-arity contract (#439).
- **Docs**: the crate READMEs follow the published 0.14 / view 0.8 line (#417, #418, #420).

`[workspace.dependencies]` moves with the crates, so a dependent packages against the version it
needs. This list is drawn from the commit subjects between the 0.14.0 publish and `34ec19c`; it is
not a re-audit of each diff.

This patch was never published: #468 folded its version move for the published-content identity check (#409, #452) into
`0.15.0` above. It does not carry the `copy_into_col_major` signature change: that is a new minor in `0.15.0`.

## 0.14.0 (2026-11-08) — breaking

Bumped to **0.14.0**: `strided-basic`, `strided-perm`, `strided-kernel`, `strided-fused`, `strided-nn`.
`strided-view` goes to **0.8.0** and `strided-traits` to **0.7.2**; the table below says why each one moves.

### `strided-view`: `MaybeUninit` storage for the `*_uninit` constructors

`StridedArray::col_major_uninit`, `try_col_major_uninit`, `col_major_from_buffer_uninit` and
`try_col_major_from_buffer_uninit` now return `StridedArray<MaybeUninit<T>>` and store their data as
`Vec<MaybeUninit<T>>`. `Vec::set_len` over uninitialised slots is gone.

**Why this is breaking, and why it had to be.** The old constructors built a `Vec<T>` whose `0..len` was not initialised.
`Vec::set_len` is a claim that those slots *are* initialised; deferring that to the caller is not something it permits. With
`T: Copy` admitting `bool`, `char`, `&U` and `NonNull<U>`, the `Vec` invariant was already broken before anyone wrote, so a
later reborrow, reallocation or drop was UB — and a caller writing every element first did not make it sound.

Migration, one line at a time:

```rust
use std::mem::MaybeUninit;

// before
let a: StridedArray<f64> = unsafe { StridedArray::col_major_uninit(&[3]) };
let a = unsafe { a.write_all(...) };          // whatever wrote every element

// after
let mut a = unsafe { StridedArray::<f64>::col_major_uninit(&[3]) };
// write every slot before any of them is read:
a.set(&[0], MaybeUninit::new(1.0));
a.set(&[1], MaybeUninit::new(2.0));
a.set(&[2], MaybeUninit::new(3.0));
// SAFETY: every slot was written above.
let first: f64 = unsafe { a.get(&[0]).assume_init() };
```

There is deliberately no `StridedArray<MaybeUninit<T>> -> StridedArray<T>` conversion in this release: such a
method would be an `unsafe fn` whose contract is "every slot is initialised", which is precisely the claim the old
`set_len` made without evidence. A caller that wants the initialised type can build it from the values it wrote, or use the
safe constructors from the start.

Callers that do not need an uninitialised buffer should use the safe constructors and are unaffected.

### Why the five other crates move too

`strided-view` types are part of four other public APIs, so 0.8.0 is their change as well: `strided-basic` re-exports
the whole crate, `strided-kernel` re-exports `strided-basic`, and `strided-fused` and `strided-perm` take
`&StridedView` / `&mut StridedViewMut` in their own signatures. `strided-nn` follows so the graph keeps one `strided-basic`.

| crate | before | after | why |
| --- | --- | --- | --- |
| `strided-view` | 0.7.2 | **0.8.0** | the four `*_uninit` constructors return `StridedArray<MaybeUninit<T>>` |
| `strided-basic` | 0.13.0 | **0.14.0** | `pub use strided_view::*` (`src/lib.rs:157`): the changed constructors are its API |
| `strided-kernel` | 0.13.0 | **0.14.0** | `pub use strided_basic::*` (`src/lib.rs:5`) |
| `strided-fused` | 0.13.0 | **0.14.0** | `fused_elementwise_into` takes `&[StridedView]` / `&mut [StridedViewMut]` (`src/fused.rs:1868`) |
| `strided-perm` | 0.13.0 | **0.14.0** | `copy_into*` take `&StridedView` / `&mut StridedViewMut` (`src/copy.rs:228`) |
| `strided-nn` | 0.13.0 | **0.14.0** | depends on `strided-basic`; a 0.13.x line would put two of them in one graph |
| `strided-traits` | 0.7.1 | **0.7.2** | #412 rewrote shipped code; no public item changed |

A patch on the 0.13.x line would be the wrong number: in 0.x the minor is the breaking position, and a downstream that
keeps `strided-view = "0.7"` while taking the new `strided-perm` gets two view types and `E0308` at the boundary — the
failure the version number has to announce, and one no workspace test can see, because the workspace builds on paths.
`[workspace.dependencies]` names these versions, so a dependent packages against the dependency it needs. All seven were published bottom-up on 2026-11-08, so there is no
missing version for a dependent to meet; the order was (`traits`, `view`,
`perm`, `basic`, `fused`, `nn`, `kernel`).

### The fused planner no longer overflows its ordering weights (#450)

`compute_importance` packs each array's axis level `n - order` `g = ceil(log2(m + 2))` bits
apart, which needs 98 bits at rank 32 with four planner operands (the highest term is
`2^97`: a 93-bit level shift times the `2^4` output weight). `1u64 << shift` panicked in
debug ("attempt to shift left with overflow") and wrapped in release, where the plan order for a
high-rank pointwise call came out scrambled. The packing is now tried first under checked
arithmetic and kept whenever it fits, so weights and plan order are unchanged wherever the old
code worked; on an actual overflow the per-level width narrows to the widest one that fits,
which keeps every level distinct below rank ~62. Past that, the levels above the cap share the
top weight.

### The unpublished 0.13.1 batch

`strided-view` 0.7.3, `strided-basic`/`strided-perm`/`strided-kernel` 0.13.1 and `strided-traits` 0.7.2 were staged and
never published (crates.io carries 0.7.2 / 0.13.0 / 0.7.1), so that work ships in the numbers above. After 0.13.0 was
published from `25b6dad`, #405 added comments to the `strided-basic`, `strided-fused` and `strided-kernel` sources (a
one-line rationale on each silenced lint), so the published 0.13.0 is not this tree; no code changed. #412
(`cargo clippy -D warnings`) rewrote code in the shipped sources of `strided-basic`, `strided-perm`, `strided-view` and
`strided-traits`. The rewrites are equivalent (`iter().any(|&d| d == 0)` → `contains(&0)`, `x % n != 0` →
`!x.is_multiple_of(n)`, index loops → `enumerate()`, `#[allow]` attributes with rationales, test-only edits); no public
item changed. `strided-view` now calls `usize::is_multiple_of`, so it needs **Rust 1.87 or newer** (no `rust-version` is
declared).

## 0.13.0 (2026-11-08)

This is a **minor** release, not a patch, because 0.12.0 already shipped on
crates.io with the old names: renaming inside a patch version would break every
downstream that runs `cargo update` without a version constraint change.

Bumped to 0.13.0: `strided-basic`, `strided-perm`, `strided-kernel`,
`strided-fused`, `strided-nn`. `strided-traits` goes to **0.7.1**: tests moved to
their own modules and a doc example became a doctest, so the published 0.7.0 is
not this tree; the public API is unchanged.
`strided-view` goes to **0.7.2**: the published 0.7.1 still multiplies strides with
wrapping `isize` arithmetic. This tree replaces that with two behaviours, and they are
not the same:

- a **non-empty** shape whose stride or element count does not fit `isize` now **panics**;
- a shape with a **zero extent** does not panic — it saturates at `isize::MAX`, so
  `col_major_strides(&[usize::MAX, 0])` returns `[1, isize::MAX]` where published 0.7.1
  returns `[1, -1]` (the wrapped value). Callers that indexed a zero-extent shape with a
  negative stride will see the saturation, not an abort.

### Breaking changes

Public functions renamed to the suffix vocabulary of `REPOSITORY_RULES.md`
(`_acc` = accumulate into the destination, `_into` = write the destination,
`_view` = take a view). Semantics are unchanged; only the name is.

| 0.12.0 | 0.13.0 |
| --- | --- |
| `add` | `add_acc` |
| `mul` | `mul_acc` |
| `fma` | `fma_acc` |
| `axpy` | `axpy_acc` |
| `copy_scale` | `copy_scale_into` |
| `copy_conj` | `copy_conj_into` |

All six live in `strided-basic` and are re-exported by `strided-kernel`
(`pub use strided_basic::*`), so both crates change together.

Other breaking changes:

- `strided_perm::HostClass` is now a re-export of `threadteam::HostClass`
  instead of a copy; the variants and `HostClass::detect` are unchanged.
- The four erased `from_raw_parts` constructors in `strided-view`
  (`ErasedRawStridedPtr`, `ErasedRawStridedRef`, `ErasedRawStridedMut`,
  `ErasedRawStridedUninitMut`) now return `StridedError::DataAlignmentMismatch`
  for a zero-length buffer whose base is misaligned (#434). They used to return
  `Ok`, leaving `data_as*` to form an unaligned `&[T]`, which is UB.
- `strided_view::StridedView::get_unchecked` now states `indices.len() ==
  ndim()` in its `# Safety` contract and checks it with `debug_assert_eq!` in
  debug builds (#437). A shorter `indices` never meant "the missing axes are
  indexed with 0": on a view with a zero-extent axis it dereferenced the
  dangling base pointer. Release code generation is unchanged.

Listed for completeness, not a downstream change: `strided-perm`'s own copy of
`SMALL_TENSOR_THRESHOLD` (`src/kernel.rs`) is gone. That module was never public
(`mod kernel`, not re-exported), so no path a downstream could name disappears;
`strided_basic::execution::SMALL_TENSOR_THRESHOLD` is unchanged.

### Migrating

```
cargo add strided-basic@0.13 strided-perm@0.13 strided-kernel@0.13 strided-fused@0.13 strided-nn@0.13
# then rename the resolved calls: `add`→`add_acc`, `mul`→`mul_acc`, `fma`→`fma_acc`,
# `axpy`→`axpy_acc`, `copy_scale`→`copy_scale_into`, `copy_conj`→`copy_conj_into`
# (unqualified matching is unsafe: `ptr.add(1)` and `impl Add` are not these calls)

cargo build   # then fix the remaining call sites the compiler names
```

Code that only calls `strided-traits` / `strided-view` needs no source change.
`strided-view` 0.7.2 replaces a silently wrapping stride with a panic for non-empty shapes,
and with a saturation at `isize::MAX` for zero-extent shapes — the two cases differ, and
only the first aborts.

### Additive

- `strided_basic::replay_copy_raw`, `replay_copy_raw_uninit`,
  `replay_map_raw`, `replay_map_raw_uninit`: replay a copy or map over
  caller-fused dims and strides.
- `#![doc = include_str!("../README.md")]` on the maintained crates, so the
  README examples now run as doctests.
