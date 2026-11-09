# Changelog

Crates: `strided-traits`, `strided-view`, `strided-perm`, `strided-basic`,
`strided-kernel`, `strided-fused`, `strided-nn`. Per-release worklogs live in
`docs/worklogs/`.

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
