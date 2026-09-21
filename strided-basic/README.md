# strided-basic

Shared typed CPU primitives and concrete copy, concatenation and reduction
execution. Generic map/zip, SIMD, indexing plans and their validation stay with
the traversal implementation. Concrete ordinary arithmetic/indexing dispatch
lives in `strided-kernel`; runtime-DAG fusion lives in `strided-fused`.
Neither of those packages is a dependency of this crate.

```rust
use strided_basic::{copy_into, StridedArray};
let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |i| (i[0] + 2*i[1]) as f64);
let mut dst = StridedArray::<f64>::col_major(&[2, 3]);
copy_into(&mut dst.view_mut(), &src.view()).unwrap();
assert_eq!(dst.get(&[1, 2]), 5.0);
```

`copy_into_uninit` copies into `StridedViewMut<MaybeUninit<T>>` without first
initializing the destination. Success initializes every logical element, not
unreachable holes. It preserves the shared bounded-thread policy.

`simd` is enabled by default. `parallel` opts into the shared execution policy;
`ExecContext::serial()` requests the serial path. Features select implementation
capabilities, not operation families.

The `execution` module is a documented low-level kernel-extension contract.
Prevalidated entry points are unsafe, and a layout marker alone does not satisfy
their requirements. Application code should use the checked root APIs.

See [kernel boundaries](../docs/design/cpu-kernel-boundaries.md) and the packaged
`NOTICE` / `THIRD-PARTY-LICENSES` for the Strided.jl lineage.

This new package has not yet been published; use the matching workspace checkout.
