# Provenance and Citation Policy

This document records which strided-rs components build on which external
projects, which implemented algorithms originate in which publications, and
how we ask research users to cite them. We maintain it because modern
development, human or AI-assisted, makes it easy to write one codebase while
referencing another, and the scientific contribution of the referenced
projects deserves visible credit beyond what licenses require. When a new
component references external code, designs, or published algorithms, add it
here in the same PR.

## Citation policy

Citations should reflect the full lineage of the methods you use, not only
the topmost software layer. This is a permanent style, not a stopgap: when a
strided-rs software paper appears, it will be added to the list below in
addition to, not in place of, the upstream citations.

- Cite the original papers of the algorithms your work relies on (see
  "Algorithm origins" below).
- For components ported from upstream libraries, check the citation policies
  of those upstream projects and apply them recursively.
- strided-rs itself does not yet have a software paper; if you need to
  reference it directly, cite the repository URL and the version or commit
  you used.

## Component provenance

This table records intellectual provenance (designs, algorithms, and code
lineage), not the Cargo dependency graph; see each crate's `Cargo.toml` for
actual dependencies. Relationship vocabulary:

- **Port**: reimplementation of a specific upstream library, following its
  public API and algorithms.
- **Derived (license)**: contains code following the upstream implementation
  closely enough to carry its license terms.
- **Inspired**: data structures or API design modeled on the upstream;
  implementation independent.
- **Compatible**: interoperates with the upstream's conventions or types;
  validated against it.

Components with more than one provenance relationship get one row per
relationship; the role is stated only on the first row.

| Component | Role | Design/code provenance | Relationship |
| --- | --- | --- | --- |
| `strided-traits` | Element-operation and scalar traits (`Identity`, `Conj`, `Transpose`, `Adjoint`) | [Strided.jl](https://github.com/Jutho/Strided.jl) | Port (type-level counterparts of `FN`/`FC`/`FT`/`FA`) |
| `strided-view` | Dynamic-rank strided views and metadata ops | [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) | Port |
| `strided-kernel` | Cache-optimized map/reduce/broadcast kernels, threading | [Strided.jl](https://github.com/Jutho/Strided.jl) | Port (fusion, ordering, blocking, `_mapreduce_threaded!`) |
| `strided-perm` | Cache-efficient tensor permutation / transpose | [HPTT](https://github.com/springer13/hptt) | Derived (BSD-3-Clause): algorithm and structure of the C++ implementation; SIMD kernels and autotuning not ported |
| `strided-einsum2` | Binary einsum via GEMM | — | Original |
| `strided-opteinsum` | N-ary einsum with contraction-order optimization | [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) | Inspired (design ideas and reference test-case patterns) |
| `mdarray-opteinsum` | Einsum adapter for `mdarray` | [mdarray](https://crates.io/crates/mdarray) | Compatible (adapter) |
| `ndarray-opteinsum` | Einsum adapter for `ndarray` | [ndarray](https://crates.io/crates/ndarray) | Compatible (adapter) |
| `strided-rs` | User-facing facade crate | — | Original |

## Algorithm origins

Library provenance above is distinct from the scientific origin of the
algorithms themselves. Where an implemented algorithm has an identifiable
original publication, research using that component should cite it. This
list is best-effort; corrections and additions are welcome.

| Algorithm | Component(s) | Original references |
| --- | --- | --- |
| Blocked tensor transposition (dimension fusion, macro/micro kernels, recursive loop nest) | `strided-perm` | P. Springer, T. Su, P. Bientinesi, "HPTT: A High-Performance Tensor Transposition C++ Library", [ARRAY 2017](https://doi.org/10.1145/3091966.3091968), [arXiv:1704.04374](https://arxiv.org/abs/1704.04374) |
| Cache-blocked strided map/reduce (dimension fusion, stride-ordered loops, L1 blocking) | `strided-kernel` | Strided.jl has no accompanying paper; cite the [Strided.jl repository](https://github.com/Jutho/Strided.jl) |

Every published crate declares its component-specific lineage in its packaged
`NOTICE`. Crates containing ported or license-derived code also package the
applicable complete upstream license text in `THIRD-PARTY-LICENSES`:
`strided-traits` and `strided-kernel` carry Strided.jl's MIT notice,
`strided-view` carries StridedViews.jl's MIT notice, and `strided-perm` carries
HPTT's BSD-3-Clause notice. The `strided-perm` crate is licensed as
`(MIT OR Apache-2.0) AND BSD-3-Clause`.
