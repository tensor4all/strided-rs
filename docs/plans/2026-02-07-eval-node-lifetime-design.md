# eval_node Lifetime Refactor: Propagate Borrowed Views

GitHub Issue: #50

## Problem

`eval_node` returns `EinsumOperand<'static>`, forcing `to_owned_static()` on every Leaf operand. This copies the entire input tensor even when only a small subset is needed (trace, diagonal, ptrace). The `eval_single_ref`/`eval_pair_ref` workarounds only help when children are immediate Leaves of a Contract node.

Remaining copy sites:
- Binary contraction with one Leaf, one Contract child — the Leaf gets copied
- N-ary (3+) contractions — all Leaf children get copied
- Any deeper tree nesting where a Leaf isn't a direct child of its consuming Contract

## Solution

### 1. Core Signature Change

Thread lifetime `'a` through `eval_node` so borrowed views propagate from Leaf to consumer:

```rust
// Before:
fn eval_node(
    node: &EinsumNode,
    operands: &mut Vec<Option<EinsumOperand<'_>>>,
    needed_ids: &HashSet<char>,
) -> Result<(EinsumOperand<'static>, Vec<char>)>

// After:
fn eval_node<'a>(
    node: &EinsumNode,
    operands: &mut Vec<Option<EinsumOperand<'a>>>,
    needed_ids: &HashSet<char>,
) -> Result<(EinsumOperand<'a>, Vec<char>)>
```

Leaf branch returns the operand directly via `.take()` — no `to_owned_static()`.

Contract results are `EinsumOperand<'static>` (freshly allocated), which coerces to `EinsumOperand<'a>` since `'static: 'a`.

Entry point changes accordingly:

```rust
pub fn evaluate<'a>(&self, operands: Vec<EinsumOperand<'a>>) -> Result<EinsumOperand<'a>>
```

### 2. Simplified Contract Branches

All Contract branches become uniform — no Leaf-detection special cases:

**Single-tensor (1 arg):**
```rust
let (child_op, child_ids) = eval_node(&args[0], operands, &needed_ids)?;
// identity/permutation passthrough (see section 3) or:
let result = eval_single_ref(&child_op, &child_ids, &output_ids)?;
```

**Binary (2 args):**
```rust
let (left_op, left_ids) = eval_node(&args[0], operands, &needed_ids)?;
let (right_op, right_ids) = eval_node(&args[1], operands, &needed_ids)?;
let result = eval_pair_ref(&left_op, &left_ids, &right_op, &right_ids, &output_ids)?;
```

**N-ary (3+ args):**
```rust
let mut children: Vec<(EinsumOperand<'a>, Vec<char>)> = args.iter()
    .map(|arg| eval_node(arg, operands, &needed_ids))
    .collect::<Result<_>>()?;
execute_nested(&mut children, &tree, &output_ids)  // signature updated to 'a
```

`execute_nested` updated to accept `EinsumOperand<'a>` and use `eval_pair_ref` internally.

### 3. Additional Zero-Copy Optimizations

**Identity passthrough:** When `input_ids == output_ids`, return the operand directly with no allocation:

```rust
if child_ids == output_ids {
    return Ok((child_op, output_ids));
}
```

**Permutation-only passthrough:** When input and output ids are the same set (same length, no repeats, just reordered), permute dims/strides metadata without touching the data buffer:

```rust
if is_permutation_only(&child_ids, &output_ids) {
    let perm = compute_permutation(&child_ids, &output_ids);
    let permuted = child_op.permuted(perm);
    return Ok((permuted, output_ids));
}
```

Requires new `permuted()` method on `StridedData`/`EinsumOperand` and a new `StridedArray::permuted()` method that reorders dims/strides Arcs (metadata only, no data copy). `StridedView::permute()` already exists and is zero-copy.

Detection: `is_permutation_only` checks that input_ids and output_ids have the same length, same set of chars, and no char appears more than once in either (rules out trace).

### 4. Code Removal

Remove dead code after the refactor:
- `to_owned_static()` method on `EinsumOperand`
- `eval_pair` (consuming version) — replaced by `eval_pair_ref` everywhere
- `eval_single` (consuming version) — replaced by `eval_single_ref` everywhere
- Leaf-detection branches in single-arg and binary-arg Contract cases

### 5. Impact Summary

| Scenario | Before | After |
|----------|--------|-------|
| Leaf in non-trivial tree | full tensor copy | zero-copy |
| Identity single-tensor | allocate + copy | zero-cost passthrough |
| Permutation-only single-tensor | allocate + copy | metadata reorder only |
| Trace/reduction | allocate + compute | same (unavoidable) |
| Binary contraction output | allocate + compute | same (unavoidable) |

## Files Changed

- `strided-opteinsum/src/expr.rs` — `eval_node`, `evaluate`, remove `eval_pair`/`eval_single`/`to_owned_static`, simplify Contract branches
- `strided-opteinsum/src/operand.rs` — add `permuted()` to `StridedData` and `EinsumOperand`
- `strided-view/src/view.rs` — add `StridedArray::permuted()` (metadata-only dim/stride reorder)

## Testing

1. All existing einsum tests pass (baseline correctness)
2. View operands in 3+ operand expressions with non-trivial tree structure
3. Permutation-only passthrough correctness
4. Mixed owned/view operands in the same expression
5. Benchmark validation with `RAYON_NUM_THREADS=1` — no regression, improvement for View inputs in n-ary cases
