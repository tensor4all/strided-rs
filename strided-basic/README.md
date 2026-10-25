# strided-basic

Shared typed CPU primitives and concrete copy, concatenation and reduction
execution. Generic map/zip, SIMD, indexing plans and their validation stay with
the traversal implementation. Concrete ordinary arithmetic/indexing dispatch
lives in `strided-kernel`; runtime-DAG fusion lives in `strided-fused`.
Neither of those packages is a dependency of this crate.

```rust
use strided_basic::{copy_into, Exec, StridedArray};
let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |i| (i[0] + 2*i[1]) as f64);
let mut dst = StridedArray::<f64>::col_major(&[2, 3]);
copy_into(&mut dst.view_mut(), &src.view(), Exec::serial()).unwrap();
assert_eq!(dst.get(&[1, 2]), 5.0);
```

`copy_into_uninit` copies into `StridedViewMut<MaybeUninit<T>>` without first
initializing the destination. Success initializes every logical element, not
unreachable holes.

Every operation that can fan out takes a `strided_exec::Exec` placement token
as its last parameter (re-exported as `strided_basic::Exec`, with `Pool`).
`Exec::serial()` runs on the calling thread; `Exec::new(&pool, budget)` splits
the work within `budget` slots of a pool the caller owns, and an operation below
the repository threshold stays on the calling thread whatever the budget. This
crate starts no task of its own and never reaches a global pool. Narrow a token
with `Exec::limit` to bound one operation more tightly than its pool.

`simd` is enabled by default. `parallel` lets pooled tokens run on a private
pool; without it every token runs inline. Features select implementation
capabilities, not operation families.

```rust
use strided_basic::{map_into, Exec, Pool, StridedArray};
let pool = Pool::builder().threads(2).build().unwrap();
let src = StridedArray::<f64>::from_fn_col_major(&[4], |i| i[0] as f64);
let mut dst = StridedArray::<f64>::col_major(&[4]);
map_into(&mut dst.view_mut(), &src.view(), |x| x + 1.0, Exec::new(&pool, 2).unwrap()).unwrap();
assert_eq!(dst.get(&[3]), 4.0);
```

The `execution` module is a documented low-level kernel-extension contract.
Prevalidated entry points are unsafe, and a layout marker alone does not satisfy
their requirements. Application code should use the checked root APIs.

See [kernel boundaries](../docs/design/cpu-kernel-boundaries.md) and the packaged
`NOTICE` / `THIRD-PARTY-LICENSES` for the Strided.jl lineage.

This new package has not yet been published; use the matching workspace checkout.
