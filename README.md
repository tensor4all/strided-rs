# strided-rs

`strided-rs` is a Rust workspace for strided tensor views and kernels.
It is inspired by Julia's [Strided.jl](https://github.com/Jutho/Strided.jl),
[StridedViews.jl](https://github.com/Jutho/StridedViews.jl), and
[OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

Depend on the individual crates you need, for example `strided-view` for the
view types, `strided-perm` for permutation and `strided-kernel` for
arithmetic. All workspace crates listed below are maintained. The new basic/fused packages
are currently available from this workspace checkout, not yet from crates.io.

## Workspace Layout

- [`strided-traits`](strided-traits/): shared scalar and element-operation traits
- [`strided-view`](strided-view/README.md): core dynamic-rank strided view/array types and metadata ops
- [`strided-perm`](strided-perm/README.md): cache-efficient tensor permutation / transpose
- [`strided-basic`](strided-basic/README.md): shared typed primitives and copy/concatenation/reduction execution
- [`strided-kernel`](strided-kernel/README.md): concrete ordinary arithmetic/indexing dispatch and typed APIs
- [`strided-fused`](strided-fused/README.md): runtime-DAG fused execution

The `strided-rs` facade and the einsum crates (`strided-einsum2`,
`strided-opteinsum`, `mdarray-opteinsum`, `ndarray-opteinsum`) were removed
from the workspace after 0.4.4; their published releases remain on crates.io.

## Features

- **Dynamic-rank strided views** (`StridedView` / `StridedViewMut`) over contiguous memory
- **Owned strided arrays** (`StridedArray`) with row-major and column-major constructors
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: permuting, transposing, broadcasting
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Optional multi-threading** via Rayon (`parallel` feature) with recursive dimension splitting

## Installation

Add the crates you use from crates.io, for example:

```toml
[dependencies]
strided-view = "0.4"
strided-kernel = "0.4"
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

See the [`strided-basic`](strided-basic/README.md) and
[`strided-fused`](strided-fused/README.md) READMEs; their Rust examples are
included in crate docs and verified by doctests in CI.

See each sub-crate README for usage examples:
- [`strided-view`](strided-view/README.md) — types, view operations
- [`strided-perm`](strided-perm/README.md) — permutation and transpose kernels
- [`strided-basic`](strided-basic/README.md) — shared primitives and lightweight execution
- [`strided-kernel`](strided-kernel/README.md) — ordinary erased dispatch and typed APIs
- [`strided-fused`](strided-fused/README.md) — runtime-DAG fusion

Design notes:
- [CPU kernel boundaries](docs/design/cpu-kernel-boundaries.md) — generic code ownership, execution contracts and concrete dispatch

Performance design notes:
- [`erased execution policy`](docs/design/erased-execution-policy.md) — serial/parallel thresholds, benchmark commands, and evidence flow
- [`faer-kernel-writing-guide`](docs/faer-kernel-writing-guide.md) — practical rules for writing hot strided kernels based on faer
- [`faer_design`](docs/faer_design.md) — SIMD design analysis and optimization plan

Published benchmark programs and current measured results live in
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).
Reports of bottlenecks, failure cases, or workloads where a strided path loses
to a credible naive baseline are welcome; please include shape, strides,
element type, thread count, and a minimal reproducer when possible.

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho
- [HPTT](https://github.com/springer13/hptt) by Paul Springer, Tong Su, and
  Paolo Bientinesi, whose transpose algorithm `strided-perm` reimplements
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) for the design
  ideas and reference test-case patterns of the former `strided-opteinsum`

A per-component table of which external projects each crate builds on, and
the algorithm-origin references, is maintained in the
[Provenance and Citation Policy](docs/PROVENANCE_AND_CITATION_POLICY.md).

## How to Cite

If you use strided-rs in research, please read the
[Provenance and Citation Policy](docs/PROVENANCE_AND_CITATION_POLICY.md)
and cite the original papers of the algorithms your work relies on (for
example the HPTT paper,
[ARRAY 2017](https://doi.org/10.1145/3091966.3091968), when your work relies
on `strided-perm`), and check the citation policies of the upstream projects
the components you use are ported from, applying them recursively. This is
the permanent citation style for this project: a future strided-rs software
paper will add to, not replace, these upstream citations. Until then,
reference strided-rs directly by repository URL and version or commit.

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

See `NOTICE` for upstream attribution (Strided.jl / StridedViews.jl are
MIT-licensed) and `THIRD-PARTY-LICENSES` for the HPTT attribution; the
`strided-perm` crate is licensed as `(MIT OR Apache-2.0) AND BSD-3-Clause`
because its transpose module is derived from HPTT (BSD-3-Clause).
