# Changelog

Crates: `strided-traits`, `strided-view`, `strided-perm`, `strided-basic`,
`strided-kernel`, `strided-fused`, `strided-nn`. Per-release worklogs live in
`docs/worklogs/`.

## 0.15.0 (unreleased) — minor

`0.14.0` is published and the tree already carries the parked `0.14.1` patch (#452), so nothing unpublished can absorb
an API addition: a new parameter on `copy_into_col_major` is a minor. The workspace version still reads `0.14.1` and
moves when this section is released, together with or after that patch.

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

### `strided-view` 0.8.1: `auxiliary::index_order_into`

`index_order_into(strides, out)` writes the stride order into caller storage and returns `StrideLengthMismatch` when
the lengths differ; `index_order` is its allocating form and is unchanged. The injectivity check's difference search no
longer allocates when it enumerates more than eight axes.

## 0.14.1 (unreleased) — patch

`0.14.0` shipped on crates.io on 2026-11-08. Since then `main` has changed shipped files in all six
crates, so the weekly `Packaged crate tests` job's `published content identity` step (#409) reports
`6 mismatch(es)` at `34ec19c` (run 42670871078); the same check run locally
(`python3 scripts/published_identity.py .`) reports the same six. The version number no longer
identified one body of content, so this patch moves it.

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

This patch is the version move for the published-content identity check (#409, #452) and nothing else. It does not carry
the `copy_into_col_major` signature change: that is a new minor and lands in `0.15.0` above.

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
