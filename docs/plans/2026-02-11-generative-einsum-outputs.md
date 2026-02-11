# Plan: Generative Einsum Outputs (Issue #45)

## Goal

Support output indices not present in any input (e.g., `->ii`, `i->ij`, `i->ii`),
matching OMEinsum.jl's full unary pipeline. Requires an explicit `size_dict` for
dimensions that cannot be inferred from operands.

## API

Add optional `size_dict: Option<&HashMap<char, usize>>` to:

- `einsum()` and `einsum_into()` in `lib.rs`
- `EinsumCode::evaluate()` and `EinsumCode::evaluate_into()` in `expr.rs`

Internally, merge user-provided `size_dict` with operand-inferred sizes into a
unified `HashMap<char, usize>` and thread it through all evaluation functions.

### Usage examples

```rust
// Normal (no size_dict needed)
einsum("ij,jk->ik", vec![a, b], None)?;

// Scalar -> 5×5 identity matrix
einsum("->ii", vec![scalar], Some(&HashMap::from([('i', 5)])))?;

// Vector -> diagonal matrix (size inferred from input)
einsum("i->ii", vec![v], None)?;

// Vector -> broadcast to new axis
einsum("i->ij", vec![v], Some(&HashMap::from([('j', 4)])))?;
```

## Internal Changes

### single_tensor_einsum (5-step pipeline)

Extend current 3-step pipeline to 5 steps matching OMEinsum.jl:

1. **Diag** — repeated input indices → `diagonal_view` (existing)
2. **Sum** — reduce axes not in output (existing)
3. **Permute** — reorder to match output order (existing)
4. **Repeat** — broadcast to NEW dimensions (NEW: stride-0 broadcast)
5. **Duplicate** — repeated output indices (NEW: diagonal write)

Add `size_dict: Option<&HashMap<char, usize>>` parameter.

### expr.rs propagation

Thread `size_dict: &HashMap<char, usize>` (unified) through:
- `eval_node()` → `eval_single()` → `single_tensor_einsum()`
- `out_dims_from_ids()` / `out_dims_from_map()` (fallback to size_dict)
- `execute_nested()` / `execute_nested_into()`

### Error handling

- Output label not in input AND not in `size_dict` → `OrphanOutputAxis` (unchanged)
- `size_dict` conflicts with operand-inferred size → `DimensionMismatch` (already exists)

## Files to modify

| File | Change |
|------|--------|
| `src/lib.rs` | Add `size_dict` param to `einsum()`, `einsum_into()` |
| `src/expr.rs` | Add `size_dict` param to evaluate/eval_node/eval_single/eval_pair/out_dims/execute_nested |
| `src/single_tensor.rs` | Add `size_dict` param, implement Repeat + Duplicate steps |
| `tests/integration.rs` | Tests for generative patterns |
| `benches/*.rs` | Update call sites (pass `None`) |

## Tests

- `->ii` with scalar (identity matrix)
- `->ij` with scalar (broadcast)
- `i->ii` (vector to diagonal)
- `i->ij` (vector broadcast to new axis)
- `ii->i` → `i->ii` roundtrip
- Missing size_dict for generative label → error
- size_dict conflict with operand → error
