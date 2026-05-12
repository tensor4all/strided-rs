# strided-rs

`strided-rs` is a Rust workspace for strided tensor views, kernels, and einsum.
It is inspired by Julia's [Strided.jl](https://github.com/Jutho/Strided.jl),
[StridedViews.jl](https://github.com/Jutho/StridedViews.jl), and
[OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

The recommended user-facing crate is [`strided-rs`](strided-rs/README.md).
Use individual crates such as `strided-perm`, `strided-view`, or
`strided-kernel` directly when you need a smaller dependency surface or a
lower-level API.

## Workspace Layout

- [`strided-rs`](strided-rs/README.md): facade crate that re-exports the main workspace APIs
- [`strided-traits`](strided-traits/): shared scalar and element-operation traits
- [`strided-view`](strided-view/README.md): core dynamic-rank strided view/array types and metadata ops
- [`strided-perm`](strided-perm/README.md): cache-efficient tensor permutation / transpose
- [`strided-kernel`](strided-kernel/README.md): cache-optimized elementwise/reduction kernels over strided views
- [`strided-einsum2`](strided-einsum2/README.md): binary einsum (`einsum2_into`) on strided tensors
- [`strided-opteinsum`](strided-opteinsum/README.md): N-ary einsum frontend with nested notation and contraction-order optimization
- [`mdarray-opteinsum`](mdarray-opteinsum/): einsum wrapper for `mdarray` arrays (row-major ↔ column-major transparent conversion)
- [`ndarray-opteinsum`](ndarray-opteinsum/): einsum wrapper for `ndarray` arrays (direct strides passthrough)

## Features

- **Dynamic-rank strided views** (`StridedView` / `StridedViewMut`) over contiguous memory
- **Owned strided arrays** (`StridedArray`) with row-major and column-major constructors
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: permuting, transposing, broadcasting
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Optional multi-threading** via Rayon (`parallel` feature) with recursive dimension splitting

## Installation

These crates are being prepared for crates.io publication, but this repository
does not publish them automatically. Until a release is published, use workspace
path dependencies:

```toml
[dependencies]
strided-rs = { path = "../strided-rs/strided-rs" }
```

After publication, use:

```toml
[dependencies]
strided-rs = "0.1"
```

## Documentation

Generate API docs locally:

```bash
cargo doc --workspace --no-deps
```

Open docs locally:

```bash
open target/doc/index.html
```

CI also builds rustdoc on PRs and deploys workspace docs to GitHub Pages on `main`.

## Quick Start

See the [`strided-rs` Quick Start](strided-rs/README.md#quick-start). The Rust
example there is included in crate docs and verified by doctests in CI.

See each sub-crate README for detailed API examples and benchmarks:
- [`strided-rs`](strided-rs/README.md) — recommended facade crate and executable Quick Start
- [`strided-view`](strided-view/README.md) — types, view operations
- [`strided-perm`](strided-perm/README.md) — permutation and transpose kernels
- [`strided-kernel`](strided-kernel/README.md) — map/reduce/broadcast kernels, [benchmarks](strided-kernel/README.md#benchmarks)
- [`strided-einsum2`](strided-einsum2/README.md) — binary einsum with GEMM backend
- [`strided-opteinsum`](strided-opteinsum/README.md) — N-ary einsum, [benchmarks](strided-opteinsum/README.md#benchmarks)
- [`mdarray-opteinsum`](mdarray-opteinsum/README.md) — einsum wrapper for `mdarray` arrays
- [`ndarray-opteinsum`](ndarray-opteinsum/README.md) — einsum wrapper for `ndarray` arrays

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) for
  `strided-opteinsum` design ideas and reference test-case patterns

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

See `NOTICE` for upstream attribution (Strided.jl / StridedViews.jl are MIT-licensed).
