# Erased Gather Plan API

## Scope

Add the Rust-side indexed-read family boundary to `strided-kernel`:

- `GatherSpec` models the gather vocabulary shared with tenferro/XLA-style indexing:
  `offset_dims`, `collapsed_slice_dims`, `start_index_map`, `index_vector_dim`,
  and `slice_sizes`.
- `GatherPlan` owns the generic prepared traversal over raw strided value and
  index descriptors.
- `ErasedGatherPlan` owns the dtype-concrete replay boundary so downstream
  crates can call pre-instantiated value/index dtype combinations.
- Value dtypes: `f32`, `f64`, `i32`, `i64`, `bool`, `c32`, `c64`.
- Index dtypes: `i32`, `i64`.

## Contracts

- Plan compilation validates shape vocabulary, stride/rank agreement, output
  shape, output injectivity, and total-size overflow.
- Runtime replay checks descriptor dtype and layout identity before writing.
- Negative and oversized gather starts clamp to the valid window-start range,
  matching tenferro's existing CPU gather behavior.
- `ExecContext` is part of the erased replay signature. The first gather
  implementation is serial; the parameter prevents accidental ambient-thread
  ownership from being frozen into the boundary.
- Rank <= 8 replay scratch is stack-backed; higher ranks may allocate scratch.

## Out Of Scope

- C ABI symbols and ABI layout stabilization.
- tenferro adoption or pin bump.
- scatter/update semantics, because duplicate indices and write conflicts need
  a separate determinism/overlap contract.
- `dynamic_slice`, because it is a fixed-window copy without index-buffer
  descriptor vocabulary. It should be handled as a later read-family sibling.
