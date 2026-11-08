# Changelog

Crates: `strided-traits`, `strided-view`, `strided-perm`, `strided-basic`,
`strided-kernel`, `strided-fused`, `strided-nn`. Per-release worklogs live in
`docs/worklogs/`.

## strided-view 0.8.0 (unreleased) — breaking

`StridedArray::col_major_uninit`, `try_col_major_uninit`, `col_major_from_buffer_uninit` and
`try_col_major_from_buffer_uninit` now return `StridedArray<MaybeUninit<T>>` and store their data as
`Vec<MaybeUninit<T>>`. `Vec::set_len` over uninitialised slots is gone.

**Why this is breaking, and why it had to be.** The old constructors built a `Vec<T>` whose `0..len` was not initialised.
`Vec::set_len` is a claim that those slots *are* initialised; deferring that to the caller is not something it permits. With
`T: Copy` admitting `bool`, `char`, `&U` and `NonNull<U>`, the `Vec` invariant was already broken before anyone wrote, so a
later reborrow, reallocation or drop was UB — and a caller writing every element first did not make it sound.

Migration, one line at a time:

```rust
// before
let a: StridedArray<f64> = unsafe { StridedArray::col_major_uninit(&[3]) };
let a = unsafe { a.write_all(...) };          // whatever wrote every element

// after
let a: StridedArray<MaybeUninit<f64>> = unsafe { StridedArray::col_major_uninit(&[3]) };
// write every element through the uninit view, then read them out:
let first: f64 = unsafe { a.get(&[0]).assume_init() };
```

There is deliberately no `StridedArray<MaybeUninit<T>> -> StridedArray<T>` conversion in this release: such a
method would be an `unsafe fn` whose contract is "every slot is initialised", which is precisely the claim the old
`set_len` made without evidence. A caller that wants the initialised type can build it from the values it wrote, or use the
safe constructors from the start.

Callers that do not need an uninitialised buffer should use the safe constructors and are unaffected.

**`strided-basic` and `strided-kernel` break too**, and this is the part that is easy to miss: `strided-basic/src/lib.rs:157`
does `pub use strided_view::*;` and `strided-kernel/src/lib.rs:5` does `pub use strided_basic::*;`, so the changed
constructor signatures are part of *their* public APIs as well. They go to **0.14.0** and **0.14.0** respectively.

`strided-perm`, `strided-fused` and `strided-nn` stay on the 0.13.x line: none of them re-exports the affected types, and
none has an affected type in a public signature (`grep -rn 'pub fn .*StridedArray'` finds none in the three).

## 0.13.1 (unreleased)

`strided-basic`, `strided-fused` and `strided-kernel` go to **0.13.1**: after
0.13.0 was published from `25b6dad`, #405 added comments to their sources (a
one-line rationale on each silenced lint), so the published 0.13.0 is not this
tree. No code changed. `strided-nn` stays at 0.13.0.

`strided-perm` goes to **0.13.1**, `strided-view` to **0.7.3** and
`strided-traits` to **0.7.2**: #412 (`cargo clippy -D warnings`) rewrote code in
their shipped sources under the published version numbers. The rewrites are
equivalent (`iter().any(|&d| d == 0)` → `contains(&0)`, `x % n != 0` →
`!x.is_multiple_of(n)`, index loops → `enumerate()`, `#[allow]` attributes with
rationales, test-only edits); no public item changed. `strided-view` 0.7.3 now
calls `usize::is_multiple_of`, so it needs **Rust 1.87 or newer** (no
`rust-version` is declared).

The `[workspace.dependencies]` requirements stay at `^0.13.0` / `^0.7.2` /
`^0.7.0`, which accept these versions. Raising them before the new versions are
on crates.io breaks `cargo package` for every dependent (`failed to select a
version for the requirement strided-view = "^0.7.3"`).

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
