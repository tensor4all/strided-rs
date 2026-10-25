# strided-perm

Cache-efficient tensor permutation / transpose, inspired by
[HPTT](https://github.com/springer13/hptt) (Springer et al., 2017).

## Usage

Copy from any `strided-view` source view into a mutable destination view with
the same shape. The last argument is a [`threadteam::Exec`] placement token:
the copy splits within that budget and never starts work of its own.

```rust
use threadteam::Exec;
use strided_perm::copy_into;
use strided_view::StridedArray;

let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| {
    (idx[0] + 10 * idx[1]) as f64
});
let transposed = src.view().permute(&[1, 0]).unwrap();

let mut dst = StridedArray::<f64>::col_major(transposed.dims());
copy_into(&mut dst.view_mut(), &transposed, Exec::serial()).unwrap();
```

Use `copy_into_col_major` when the destination is known to be column-major and
you want to make that layout contract explicit.

`copy_into_uninit` takes a `StridedViewMut<MaybeUninit<T>>`
destination over fresh storage. It runs the same engine, selected on `T` rather than on
`MaybeUninit<T>`, so the native `f32`/`f64` kernels still apply; every element the views
describe is written exactly once, through raw pointers only.

## Threading

`Exec::serial()` runs the copy on the calling thread. A token minted from a
`threadteam::Pool` with `Exec::new(&pool, budget)` splits the permutation
across `budget` tasks of that pool. The crate uses exactly the budget it is
given: callers that want small copies to stay on one thread pass a serial token
or narrow the token with `Exec::limit`. Without the `parallel` feature every
token runs inline.

```toml
[dependencies]
strided-perm = { version = "0.7", features = ["parallel"] }
```

```rust
use threadteam::{Exec, Pool};
use strided_perm::copy_into;
use strided_view::StridedArray;

let pool = Pool::builder().threads(2).build().unwrap();
let src = StridedArray::<f64>::col_major(&[2, 3]);
let mut dst = StridedArray::<f64>::col_major(&[2, 3]);
copy_into(&mut dst.view_mut(), &src.view(), Exec::new(&pool, 2).unwrap()).unwrap();
```

Benchmarks live in
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).

## Acknowledgments and License

The transpose engine in `src/hptt/` reimplements the algorithm of
[HPTT](https://github.com/springer13/hptt) by Paul Springer, Tong Su, and
Paolo Bientinesi, following the structure of the original C++ implementation
(P. Springer, T. Su, P. Bientinesi, "HPTT: A High-Performance Tensor
Transposition C++ Library", ARRAY 2017). HPTT is licensed under BSD-3-Clause
(Copyright 2018 Paul Springer); see the packaged `THIRD-PARTY-LICENSES` file
for the full text.

This crate is therefore licensed as `(MIT OR Apache-2.0) AND BSD-3-Clause`.
