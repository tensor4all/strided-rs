# strided-rs

`strided-rs` is the recommended entry point for the strided-rs workspace. It
re-exports the core strided view, kernel, permutation, and einsum crates so most
users can depend on one package.

## Installation

```toml
[dependencies]
strided-rs = "0.4"
```

Use individual crates such as `strided-perm`, `strided-view`, or
`strided-kernel` directly when you want a smaller dependency surface or a
lower-level API.

## Quick Start

```rust
use strided_rs::{map_into, StridedArray};

let src = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| {
    (idx[0] * 10 + idx[1]) as f64
});
let mut dest = StridedArray::<f64>::row_major(&[2, 3]);

map_into(&mut dest.view_mut(), &src.view(), |x| x * 2.0).unwrap();

assert_eq!(dest.get(&[1, 2]), 24.0);
```

## Feature Flags

- `faer` (default): enables the `faer` backend for einsum contractions.
- `parallel`: enables Rayon-backed parallel kernels where available.
- `blas`: enables the CBLAS backend for einsum contractions.
- `blas-accelerate`: enables the CBLAS backend and links Apple's Accelerate
  provider.
- `blas-openblas`: enables the CBLAS backend and links OpenBLAS.
- `blas-mkl`: enables the CBLAS backend and links Intel MKL dynamic parallel
  libraries.
- `blas-inject`: enables BLAS through `cblas-inject`.
- `mdarray`: re-exports the `mdarray-opteinsum` frontend as `strided_rs::mdarray`.
- `ndarray`: re-exports the `ndarray-opteinsum` frontend as `strided_rs::ndarray`.

`faer`, `blas`, and `blas-inject` are mutually exclusive at the einsum backend
level. Disable default features when selecting a BLAS backend. If default
features are disabled, enable one backend feature when using einsum APIs or the
`mdarray` / `ndarray` frontends:

```toml
[dependencies]
strided-rs = { version = "0.4", default-features = false, features = ["blas"] }
```

Use a provider feature when the BLAS provider should be fixed explicitly:

```toml
[dependencies]
strided-rs = { version = "0.4", default-features = false, features = ["blas-openblas"] }
```

## Namespaced APIs

The facade exposes lower-level crates under modules:

- `strided_rs::traits`
- `strided_rs::view`
- `strided_rs::perm`
- `strided_rs::kernel`
- `strided_rs::einsum2`
- `strided_rs::opteinsum`

The individual crates remain public and can be used directly.
