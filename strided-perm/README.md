# strided-perm

Cache-efficient tensor permutation / transpose, inspired by
[HPTT](https://github.com/springer13/hptt) (Springer et al., 2017).

## Usage

Copy from any `strided-view` source view into a mutable destination view with
the same shape:

```rust
use strided_perm::copy_into;
use strided_view::StridedArray;

let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| {
    (idx[0] + 10 * idx[1]) as f64
});
let transposed = src.view().permute(&[1, 0]).unwrap();

let mut dst = StridedArray::<f64>::col_major(transposed.dims());
copy_into(&mut dst.view_mut(), &transposed).unwrap();
```

Use `copy_into_col_major` when the destination is known to be column-major and
you want to make that layout contract explicit.

## Parallel Feature

Enable the `parallel` feature to use Rayon-backed copy entry points:

```toml
[dependencies]
strided-perm = { version = "0.1", features = ["parallel"] }
```

```rust
# use strided_perm::copy_into_par;
# use strided_view::StridedArray;
# let src = StridedArray::<f64>::col_major(&[2, 3]);
# let mut dst = StridedArray::<f64>::col_major(&[2, 3]);
copy_into_par(&mut dst.view_mut(), &src.view()).unwrap();
```

Benchmarks live in
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).

## Acknowledgments and License

The transpose engine in `src/hptt/` reimplements the algorithm of
[HPTT](https://github.com/springer13/hptt) by Paul Springer, Tong Su, and
Paolo Bientinesi, following the structure of the original C++ implementation
(P. Springer, T. Su, P. Bientinesi, "HPTT: A High-Performance Tensor
Transposition C++ Library", ARRAY 2017). HPTT is licensed under BSD-3-Clause
(Copyright 2018 Paul Springer); see `THIRD-PARTY-LICENSES` at the workspace
root for the full text.

This crate is therefore licensed as `(MIT OR Apache-2.0) AND BSD-3-Clause`.
