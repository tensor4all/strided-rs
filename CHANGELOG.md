# Changelog

Crates: `strided-traits`, `strided-view`, `strided-perm`, `strided-basic`,
`strided-kernel`, `strided-fused`, `strided-nn`. Per-release worklogs live in
`docs/worklogs/`.

## 0.13.0 (2026-11-08)

This is a **minor** release, not a patch, because 0.12.0 already shipped on
crates.io with the old names: renaming inside a patch version would break every
downstream that runs `cargo update` without a version constraint change.

Bumped to 0.13.0: `strided-basic`, `strided-perm`, `strided-kernel`,
`strided-fused`, `strided-nn`. `strided-traits` keeps 0.7.0 (nothing in it changed).
`strided-view` goes to **0.7.2**: the published 0.7.1 multiplies strides with wrapping
`isize` arithmetic, while this tree uses `checked_col_major_strides` /
`checked_row_major_strides` and refuses a stride product that does not fit (see the
migration note at the end).

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
`strided-view` 0.7.2 replaces unchecked dense-stride arithmetic with the checked
`checked_col_major_strides` / `checked_row_major_strides`: a stride product that does
not fit `isize` now panics (or returns `Err(OffsetOverflow)` from the checked helper)
where the published 0.7.1 wrapped. Shapes at or beyond the `isize` limit used to be
constructible with a wrapped stride; they are now rejected, so any caller relying on
the wrapped value will see a panic or an error instead.

### Additive

- `strided_basic::replay_copy_raw`, `replay_copy_raw_uninit`,
  `replay_map_raw`, `replay_map_raw_uninit`: replay a copy or map over
  caller-fused dims and strides.
- `#![doc = include_str!("../README.md")]` on the maintained crates, so the
  README examples now run as doctests.
