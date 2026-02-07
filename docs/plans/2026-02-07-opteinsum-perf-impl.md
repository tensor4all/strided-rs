# Opteinsum Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 6 benchmarks where Rust strided-opteinsum is 3-234x slower than Julia OMEinsum.

**Architecture:** Add early-return fast paths that bypass heavy machinery (GEMM, materialization, to_owned_static) for simple operation patterns. No existing code paths are modified — only new branches added.

**Tech Stack:** Rust, strided-kernel (zip_map2_into, copy_into), strided-einsum2 (einsum2_into), strided-opteinsum (single_tensor_einsum, eval_node).

**Design doc:** `docs/plans/2026-02-07-opteinsum-perf-design.md`

---

### Task 1: Element-wise fast path in einsum2_into (Solution A)

Fixes hadamard 234x → ~6x (combined with Task 2 → ~1x).

When all indices are batch (no contraction, no left-output, no right-output),
bypass GEMM and use `strided_kernel::zip_map2_into`.

**Files:**
- Modify: `strided-einsum2/src/lib.rs:166-242`
- Test: `strided-einsum2/src/lib.rs` (existing tests) + new test

**Step 1: Write the failing test**

Add to `strided-einsum2/src/lib.rs` at end of `mod tests`:

```rust
#[test]
fn test_elementwise_hadamard() {
    // C_ijk = A_ijk * B_ijk — all batch, no contraction
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 4, 5], |idx| {
        (idx[0] * 20 + idx[1] * 5 + idx[2] + 1) as f64
    });
    let b = StridedArray::<f64>::from_fn_row_major(&[3, 4, 5], |idx| {
        (idx[0] * 20 + idx[1] * 5 + idx[2] + 1) as f64 * 0.1
    });
    let mut c = StridedArray::<f64>::row_major(&[3, 4, 5]);

    einsum2_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &['i', 'j', 'k'],
        &['i', 'j', 'k'],
        &['i', 'j', 'k'],
        1.0,
        0.0,
    )
    .unwrap();

    // Spot check: C[0,0,0] = 1 * 0.1 = 0.1
    assert!((c.get(&[0, 0, 0]) - 0.1).abs() < 1e-12);
    // C[2,3,4] = 60 * 6.0 = 360
    assert!((c.get(&[2, 3, 4]) - 360.0).abs() < 1e-10);
}

#[test]
fn test_elementwise_hadamard_with_alpha() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| {
        (idx[0] * 3 + idx[1] + 1) as f64
    });
    let b = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| {
        (idx[0] * 3 + idx[1] + 1) as f64
    });
    let mut c = StridedArray::<f64>::row_major(&[2, 3]);

    einsum2_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &['i', 'j'],
        &['i', 'j'],
        &['i', 'j'],
        2.0,
        0.0,
    )
    .unwrap();

    // C[0,0] = 2.0 * 1 * 1 = 2.0
    assert_eq!(c.get(&[0, 0]), 2.0);
    // C[1,2] = 2.0 * 6 * 6 = 72.0
    assert_eq!(c.get(&[1, 2]), 72.0);
}
```

**Step 2: Run tests to verify they pass (baseline — existing GEMM path handles these)**

Run: `cargo test -p strided-einsum2 test_elementwise`
Expected: PASS (GEMM path already computes correct results)

**Step 3: Add element-wise fast path**

In `strided-einsum2/src/lib.rs`, add `use strided_kernel::zip_map2_into;` at the top imports area (around line 40). Note: `strided_kernel` is already a dependency.

In `einsum2_into`, insert the fast path after the permutation step (after line 209, before the GEMM call at line 212). The fast path goes before both `#[cfg(feature = "faer")]` and `#[cfg(not(feature = "faer"))]` blocks:

```rust
    // 4. Permute to canonical order
    let a_perm = a_view.permute(&plan.left_perm)?;
    let b_perm = b_view.permute(&plan.right_perm)?;
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

    // 5a. Fast path: element-wise operation (all indices are batch, no contraction)
    if plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty() && beta == T::zero() {
        let mul_fn = move |a_val: T, b_val: T| -> T {
            let a_c = if conj_a {
                strided_view::Conj::apply(a_val)
            } else {
                a_val
            };
            let b_c = if conj_b {
                strided_view::Conj::apply(b_val)
            } else {
                b_val
            };
            alpha * a_c * b_c
        };
        zip_map2_into(&mut c_perm, &a_perm, &b_perm, mul_fn)?;
        return Ok(());
    }

    // 5b. Call batched GEMM (existing code follows)
```

**Step 4: Run all einsum2 tests**

Run: `cargo test -p strided-einsum2`
Expected: All tests PASS (new fast path for element-wise; existing path for everything else)

**Step 5: Run hadamard benchmark to verify improvement**

Run: `RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum --bench hadamard`
Expected: Significant improvement from ~45ms (to_owned copy + element-wise via kernel instead of 1M GEMM calls)

**Step 6: Commit**

```bash
git add strided-einsum2/src/lib.rs
git commit -m "perf: add element-wise fast path in einsum2_into for batch-only ops"
```

---

### Task 2: Borrow-from-Leaf optimization in eval_node

Fixes all benchmarks using `from_view_*` by avoiding the full-tensor copy in `to_owned_static()`.

Currently `eval_node` calls `to_owned_static()` on every Leaf operand (expr.rs:337), copying
the entire input tensor. For single-tensor ops (trace: 1M elements copied, only 1K needed)
and element-wise ops (hadamard: 2× 1M copies), this dominates runtime.

Fix: when a Contract node's child is a Leaf, borrow the view from the original operand
instead of copying. The result of `single_tensor_einsum`/`einsum2_into` is already owned,
so no lifetime issues.

**Files:**
- Modify: `strided-opteinsum/src/expr.rs:216-260` (eval_pair and eval_single areas)
- Modify: `strided-opteinsum/src/expr.rs:339-365` (eval_node Contract arms)
- Test: existing tests in `strided-opteinsum/src/expr.rs`

**Step 1: Add `eval_single_ref` that borrows from operand**

In `strided-opteinsum/src/expr.rs`, add after the existing `eval_single` function (around line 260):

```rust
/// Like eval_single but borrows from the operand instead of consuming it.
/// Avoids the to_owned_static() copy for Leaf operands.
fn eval_single_ref(
    operand: &EinsumOperand<'_>,
    input_ids: &[char],
    output_ids: &[char],
) -> crate::Result<EinsumOperand<'static>> {
    match operand {
        EinsumOperand::F64(data) => {
            let view = data.as_view();
            let result = single_tensor_einsum(&view, input_ids, output_ids)?;
            Ok(EinsumOperand::F64(StridedData::Owned(result)))
        }
        EinsumOperand::C64(data) => {
            let view = data.as_view();
            let result = single_tensor_einsum(&view, input_ids, output_ids)?;
            Ok(EinsumOperand::C64(StridedData::Owned(result)))
        }
    }
}
```

**Step 2: Add `eval_pair_ref` that borrows from operands**

In `strided-opteinsum/src/expr.rs`, add after `eval_pair` (around line 237):

```rust
/// Like eval_pair but borrows from operands instead of consuming them.
/// Avoids the to_owned_static() copy for Leaf operands.
fn eval_pair_ref(
    left: &EinsumOperand<'_>,
    left_ids: &[char],
    right: &EinsumOperand<'_>,
    right_ids: &[char],
    output_ids: &[char],
) -> crate::Result<EinsumOperand<'static>> {
    // Determine output type and promote if needed
    match (left, right) {
        (EinsumOperand::F64(ld), EinsumOperand::F64(rd)) => {
            let a_view = ld.as_view();
            let b_view = rd.as_view();
            let mut dim_map: HashMap<char, usize> = HashMap::new();
            for (i, &id) in left_ids.iter().enumerate() {
                dim_map.insert(id, a_view.dims()[i]);
            }
            for (i, &id) in right_ids.iter().enumerate() {
                dim_map.insert(id, b_view.dims()[i]);
            }
            let out_dims = out_dims_from_map(&dim_map, output_ids)?;
            let mut c_arr = StridedArray::<f64>::col_major(&out_dims);
            einsum2_into(
                c_arr.view_mut(),
                &a_view,
                &b_view,
                output_ids,
                left_ids,
                right_ids,
                1.0,
                0.0,
            )?;
            Ok(EinsumOperand::F64(StridedData::Owned(c_arr)))
        }
        (EinsumOperand::C64(ld), EinsumOperand::C64(rd)) => {
            let a_view = ld.as_view();
            let b_view = rd.as_view();
            let mut dim_map: HashMap<char, usize> = HashMap::new();
            for (i, &id) in left_ids.iter().enumerate() {
                dim_map.insert(id, a_view.dims()[i]);
            }
            for (i, &id) in right_ids.iter().enumerate() {
                dim_map.insert(id, b_view.dims()[i]);
            }
            let out_dims = out_dims_from_map(&dim_map, output_ids)?;
            let mut c_arr = StridedArray::<Complex64>::col_major(&out_dims);
            einsum2_into(
                c_arr.view_mut(),
                &a_view,
                &b_view,
                output_ids,
                left_ids,
                right_ids,
                Complex64::new(1.0, 0.0),
                Complex64::zero(),
            )?;
            Ok(EinsumOperand::C64(StridedData::Owned(c_arr)))
        }
        _ => {
            // Mixed types: fall back to consuming path with promotion
            // This requires owned data, so we can't avoid the copy here
            let left_c64 = left.to_c64_owned_ref();
            let right_c64 = right.to_c64_owned_ref();
            eval_pair(left_c64, left_ids, right_c64, right_ids, output_ids)
        }
    }
}
```

Note: The mixed-type fallback needs a `to_c64_owned_ref` method. Add to `EinsumOperand` in `operand.rs`:

```rust
/// Promote to Complex64, creating an owned copy. Works on borrowed data.
pub fn to_c64_owned_ref(&self) -> EinsumOperand<'static> {
    match self {
        EinsumOperand::C64(data) => {
            let view = data.as_view();
            let dims = view.dims().to_vec();
            let mut dest = StridedArray::<Complex64>::col_major(&dims);
            copy_into(&mut dest.view_mut(), &view).expect("copy_into failed");
            EinsumOperand::C64(StridedData::Owned(dest))
        }
        EinsumOperand::F64(data) => {
            let view = data.as_view();
            let dims = view.dims().to_vec();
            let strides = col_major_strides(&dims);
            let f64_view_data: Vec<f64> = {
                let mut dest = StridedArray::<f64>::col_major(&dims);
                copy_into(&mut dest.view_mut(), &view).expect("copy_into failed");
                dest.data().to_vec()
            };
            let c64_data: Vec<Complex64> = f64_view_data
                .iter()
                .map(|&x| Complex64::new(x, 0.0))
                .collect();
            let c64_array = StridedArray::from_parts(c64_data, &dims, &strides, 0)
                .expect("from_parts failed");
            EinsumOperand::C64(StridedData::Owned(c64_array))
        }
    }
}
```

**Step 3: Update eval_node to use borrow paths for Leaf children**

In `strided-opteinsum/src/expr.rs`, replace the Contract match arms (lines ~343-365).

For the 1-arg case:

```rust
1 => {
    // Single-tensor operation.
    // Optimization: if child is a Leaf, borrow view directly
    // to avoid to_owned_static() copy of the full input tensor.
    if let EinsumNode::Leaf { ids, tensor_index } = &args[0] {
        let slot = operands.get(*tensor_index).ok_or_else(|| {
            crate::EinsumError::OperandCountMismatch {
                expected: tensor_index + 1,
                found: operands.len(),
            }
        })?;
        let op = slot.as_ref().ok_or_else(|| {
            crate::EinsumError::Internal(format!(
                "operand {} was already consumed",
                tensor_index
            ))
        })?;
        let result = eval_single_ref(op, ids, &node_output_ids)?;
        operands[*tensor_index].take(); // mark consumed
        return Ok((result, node_output_ids));
    }
    // Fallback: recursive evaluation for nested Contract children
    let child_needed = compute_child_needed_ids(&node_output_ids, 0, args);
    let (operand, input_ids) = eval_node(&args[0], operands, &child_needed)?;
    let result = eval_single(operand, &input_ids, &node_output_ids)?;
    Ok((result, node_output_ids))
}
```

For the 2-arg case:

```rust
2 => {
    // Binary contraction.
    // Optimization: if both children are Leaves, borrow views directly.
    if let (
        EinsumNode::Leaf { ids: left_ids, tensor_index: left_idx },
        EinsumNode::Leaf { ids: right_ids, tensor_index: right_idx },
    ) = (&args[0], &args[1]) {
        let left_op = operands.get(*left_idx)
            .ok_or_else(|| crate::EinsumError::OperandCountMismatch {
                expected: left_idx + 1, found: operands.len(),
            })?
            .as_ref()
            .ok_or_else(|| crate::EinsumError::Internal(
                format!("operand {} was already consumed", left_idx),
            ))?;
        let right_op = operands.get(*right_idx)
            .ok_or_else(|| crate::EinsumError::OperandCountMismatch {
                expected: right_idx + 1, found: operands.len(),
            })?
            .as_ref()
            .ok_or_else(|| crate::EinsumError::Internal(
                format!("operand {} was already consumed", right_idx),
            ))?;
        let result = eval_pair_ref(
            left_op, left_ids,
            right_op, right_ids,
            &node_output_ids,
        )?;
        operands[*left_idx].take();
        operands[*right_idx].take();
        return Ok((result, node_output_ids));
    }
    // Fallback: recursive evaluation
    let left_needed = compute_child_needed_ids(&node_output_ids, 0, args);
    let right_needed = compute_child_needed_ids(&node_output_ids, 1, args);
    let (left, left_ids) = eval_node(&args[0], operands, &left_needed)?;
    let (right, right_ids) = eval_node(&args[1], operands, &right_needed)?;
    let result = eval_pair(left, &left_ids, right, &right_ids, &node_output_ids)?;
    Ok((result, node_output_ids))
}
```

**Step 4: Run all tests**

Run: `cargo test -p strided-opteinsum`
Expected: All tests PASS

**Step 5: Run benchmarks to measure improvement**

Run: `RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum --bench hadamard`
Run: `RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum --bench trace`
Expected: Major improvement — hadamard and trace should now be near Julia levels.

**Step 6: Commit**

```bash
git add strided-opteinsum/src/expr.rs strided-opteinsum/src/operand.rs
git commit -m "perf: borrow views from Leaf operands to avoid to_owned_static copy"
```

---

### Task 3: Specialized trace and partial trace fast paths (Solution C)

Fixes trace 119x → ~1x, ptrace 16x → ~1x.

Add early-return fast paths at the top of `single_tensor_einsum` for common
patterns: full trace `"ii->"` and partial trace `"iij->j"` etc.

**Files:**
- Modify: `strided-opteinsum/src/single_tensor.rs:10-17` (top of function)
- Test: `strided-opteinsum/src/single_tensor.rs` (existing tests cover these patterns)

**Step 1: Verify existing tests pass**

Run: `cargo test -p strided-opteinsum single_tensor`
Expected: All PASS (baseline)

**Step 2: Add full trace fast path**

In `strided-opteinsum/src/single_tensor.rs`, insert at the beginning of `single_tensor_einsum`
(after the `where` clause, before Step 1 comment at line 18):

```rust
    // Fast path: full trace "cc...c -> " (all indices identical, scalar output)
    if output_ids.is_empty() && !input_ids.is_empty() && input_ids.iter().all(|&c| c == input_ids[0]) {
        let n = src.dims()[0];
        let diag_stride: isize = src.strides().iter().sum();
        let ptr = src.data().as_ptr();
        let mut offset = src.offset() as isize;
        let mut acc = T::zero();
        for _ in 0..n {
            acc = acc + unsafe { *ptr.offset(offset) };
            offset += diag_stride;
        }
        let mut out = StridedArray::<T>::col_major(&[]);
        out.data_mut()[0] = acc;
        return Ok(out);
    }

    // Fast path: partial trace with one repeated pair
    // e.g. "iij->j", "iji->j", "jii->j" — one pair of repeated indices, rest go to output
    {
        let mut pair: Option<(char, usize, usize)> = None;
        let mut seen_chars: Vec<(char, usize)> = Vec::new();
        let mut has_triple = false;
        for (i, &ch) in input_ids.iter().enumerate() {
            if let Some(&(_, first)) = seen_chars.iter().find(|(c, _)| *c == ch) {
                if pair.is_some() {
                    // More than one pair — not a simple partial trace
                    has_triple = true;
                    break;
                }
                pair = Some((ch, first, i));
            } else {
                seen_chars.push((ch, i));
            }
        }

        if let Some((repeated_ch, pos0, pos1)) = pair {
            if !has_triple && !output_ids.contains(&repeated_ch) {
                // All non-repeated indices must be in output_ids
                let free_ids: Vec<(char, usize)> = input_ids
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != pos0 && *i != pos1)
                    .map(|(i, &ch)| (ch, i))
                    .collect();
                let all_free_in_output = free_ids.iter().all(|(ch, _)| output_ids.contains(ch));

                if all_free_in_output && free_ids.len() == output_ids.len() {
                    let n = src.dims()[pos0]; // diagonal length
                    let diag_stride = src.strides()[pos0] + src.strides()[pos1];
                    let ptr = src.data().as_ptr();
                    let base_offset = src.offset() as isize;

                    // Compute output permutation: output_ids order vs free_ids order
                    let out_dims: Vec<usize> = output_ids
                        .iter()
                        .map(|oc| {
                            let (_, src_axis) = free_ids.iter().find(|(ch, _)| ch == oc).unwrap();
                            src.dims()[*src_axis]
                        })
                        .collect();
                    let out_strides_src: Vec<isize> = output_ids
                        .iter()
                        .map(|oc| {
                            let (_, src_axis) = free_ids.iter().find(|(ch, _)| ch == oc).unwrap();
                            src.strides()[*src_axis]
                        })
                        .collect();

                    let total_out: usize = out_dims.iter().product::<usize>().max(1);
                    let out_col_strides = strided_view::col_major_strides(&out_dims);
                    let mut out_data = vec![T::zero(); total_out];

                    // Iterate over output elements using col-major order
                    let out_rank = out_dims.len();
                    let mut idx = vec![0usize; out_rank];
                    for flat in 0..total_out {
                        // Compute source offset for this output position
                        let mut src_off = base_offset;
                        for d in 0..out_rank {
                            src_off += idx[d] as isize * out_strides_src[d];
                        }
                        // Sum along diagonal
                        let mut acc = T::zero();
                        let mut diag_off = src_off;
                        for _ in 0..n {
                            acc = acc + unsafe { *ptr.offset(diag_off) };
                            diag_off += diag_stride;
                        }
                        // Write to output using col-major flat index
                        let mut out_flat = 0usize;
                        for d in 0..out_rank {
                            out_flat += idx[d] * out_col_strides[d] as usize;
                        }
                        out_data[out_flat] = acc;

                        // Increment index (col-major order)
                        if flat + 1 < total_out {
                            for d in 0..out_rank {
                                idx[d] += 1;
                                if idx[d] < out_dims[d] {
                                    break;
                                }
                                idx[d] = 0;
                            }
                        }
                    }

                    return StridedArray::from_parts(out_data, &out_dims, &out_col_strides, 0)
                        .map_err(|e| crate::EinsumError::Strided(e));
                }
            }
        }
    }
```

**Step 3: Run all single_tensor tests**

Run: `cargo test -p strided-opteinsum single_tensor`
Expected: All PASS (fast paths return identical results)

**Step 4: Run full opteinsum test suite**

Run: `cargo test -p strided-opteinsum`
Expected: All PASS

**Step 5: Run trace and ptrace benchmarks**

Run: `RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum --bench trace`
Run: `RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum --bench ptrace`
Expected: Near Julia-level performance (~0.003ms for trace, ~0.02ms for ptrace)

**Step 6: Commit**

```bash
git add strided-opteinsum/src/single_tensor.rs
git commit -m "perf: add specialized trace and partial trace fast paths"
```

---

### Task 4: Eliminate double copy in permute-only path (Solution F)

Fixes perm 3.0x → ~1.5x.

Currently permute-only operations (`"ijkl->ljki"`) copy src → owned (line 105-108),
then permute owned → copy to output (line 131-134). Two full copies.
Fix: permute `src` directly and copy once.

**Files:**
- Modify: `strided-opteinsum/src/single_tensor.rs:97-136`
- Test: existing `test_permutation_only`

**Step 1: Verify existing test passes**

Run: `cargo test -p strided-opteinsum test_permutation_only`
Expected: PASS

**Step 2: Replace double-copy path with single-copy**

In `strided-opteinsum/src/single_tensor.rs`, replace lines 97-136 (Step 5 through Step 7):

```rust
    // Step 5: Get the current result view for permutation.
    // For permute-only (no diagonal, no reduction), use src directly to avoid double copy.

    // Step 6: Handle scalar output (output_ids is empty).
    if output_ids.is_empty() {
        let result_arr = if let Some(arr) = current_arr {
            arr
        } else if let Some(arr) = diag_arr {
            arr
        } else {
            // Scalar from source (shouldn't happen — scalar output implies reduction)
            let mut owned = StridedArray::<T>::col_major(&[]);
            owned.data_mut()[0] = unsafe { *src.data().as_ptr().offset(src.offset() as isize) };
            owned
        };
        return Ok(result_arr);
    }

    // Step 7: Permute to output order if needed.
    if current_ids == output_ids {
        let result_arr = if let Some(arr) = current_arr {
            arr
        } else if let Some(arr) = diag_arr {
            arr
        } else {
            // Identity: copy src to col-major owned array
            let dims = src.dims().to_vec();
            let mut owned = StridedArray::<T>::col_major(&dims);
            copy_into(&mut owned.view_mut(), src)?;
            owned
        };
        return Ok(result_arr);
    }

    // Compute permutation: for each output axis, find its position in current_ids.
    let mut perm: Vec<usize> = Vec::with_capacity(output_ids.len());
    for oc in output_ids {
        match current_ids.iter().position(|c| c == oc) {
            Some(pos) => perm.push(pos),
            None => return Err(crate::EinsumError::OrphanOutputAxis(oc.to_string())),
        }
    }

    // Permute from the best available source (avoid intermediate copy)
    let source_view = if let Some(ref arr) = current_arr {
        arr.view()
    } else if let Some(ref arr) = diag_arr {
        arr.view()
    } else {
        // No intermediate — permute src directly (single copy instead of double)
        src.clone()
    };

    let permuted_view = source_view.permute(&perm)?;
    let out_dims = permuted_view.dims().to_vec();
    let mut out = StridedArray::<T>::col_major(&out_dims);
    copy_into(&mut out.view_mut(), &permuted_view)?;

    Ok(out)
```

**Step 3: Run all tests**

Run: `cargo test -p strided-opteinsum`
Expected: All PASS

**Step 4: Run perm benchmark**

Run: `RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum --bench perm`
Expected: Improvement from ~2.5ms to ~1.3ms for ComplexF64

**Step 5: Commit**

```bash
git add strided-opteinsum/src/single_tensor.rs
git commit -m "perf: eliminate double copy in permute-only single_tensor_einsum path"
```

---

### Task 5: Skip materialization before reduce (Solution B)

Complements Task 3 for cases not covered by the trace/ptrace fast paths
(e.g., triple repeated indices, multiple reduction axes after diagonal).

Pass the diagonal view directly to `reduce_axis` instead of materializing first.

**Files:**
- Modify: `strided-opteinsum/src/single_tensor.rs:30-88` (Steps 2-4)
- Test: existing tests

**Step 1: Verify existing tests pass**

Run: `cargo test -p strided-opteinsum single_tensor`
Expected: All PASS

**Step 2: Skip materialization when reduce follows diagonal**

Replace the diagonal handling (after Task 3 and Task 4 have already been applied):

In `single_tensor.rs`, replace the diagonal materialization block. Change the `else` branch
(where `!pairs.is_empty()`) to skip `copy_into` when there are axes to reduce. The key change
is computing `axes_to_reduce` early, before deciding whether to materialize:

```rust
    // Step 2: Apply diagonal_view if any pairs exist, and compute unique_ids.
    let (diag_arr, unique_ids): (Option<StridedArray<T>>, Vec<char>);
    // We need to know axes_to_reduce early to decide whether to materialize.
    let unique_ids_tmp: Vec<char>;
    if pairs.is_empty() {
        diag_arr = None;
        unique_ids = input_ids.to_vec();
    } else {
        let dv = src.diagonal_view(&pairs)?;
        let axes_to_remove: Vec<usize> = pairs.iter().map(|&(_, b)| b).collect();
        unique_ids_tmp = input_ids
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes_to_remove.contains(i))
            .map(|(_, &ch)| ch)
            .collect();
        let dims = dv.dims().to_vec();
        if dims.iter().product::<usize>() == 0 {
            let out_dims: Vec<usize> = output_ids
                .iter()
                .map(|oc| {
                    let pos = unique_ids_tmp.iter().position(|c| c == oc).unwrap();
                    dims[pos]
                })
                .collect();
            return Ok(StridedArray::<T>::col_major(&out_dims));
        }

        // Check if all axes to reduce exist — if so, skip materialization
        let has_reduce = unique_ids_tmp.iter().any(|ch| !output_ids.contains(ch));
        if has_reduce {
            // Don't materialize — reduce_axis will read from the strided diagonal view directly.
            // We need to apply reduce_axis to dv, then continue.
            let mut axes_to_reduce: Vec<usize> = Vec::new();
            for (i, ch) in unique_ids_tmp.iter().enumerate() {
                if !output_ids.contains(ch) {
                    axes_to_reduce.push(i);
                }
            }
            axes_to_reduce.sort_unstable();
            axes_to_reduce.reverse();

            let mut current_arr: Option<StridedArray<T>> = None;
            let mut first = true;
            for &ax in axes_to_reduce.iter() {
                let reduced = if let Some(ref arr) = current_arr {
                    reduce_axis(&arr.view(), ax, |x| x, |a, b| a + b, T::zero())?
                } else {
                    reduce_axis(&dv, ax, |x| x, |a, b| a + b, T::zero())?
                };
                current_arr = Some(reduced);
            }

            // Compute current_ids after reductions
            let mut current_ids = unique_ids_tmp.clone();
            for &ax in axes_to_reduce.iter() {
                current_ids.remove(ax);
            }

            // Continue to permutation (reuse Step 6/7 logic below)
            // Set up for the rest of the function
            unique_ids = unique_ids_tmp;
            diag_arr = current_arr; // reuse diag_arr slot for reduced result
            // Skip the normal reduction loop by clearing axes_to_reduce
            // (already handled above)
        } else {
            // No reduction follows — materialize for output
            let mut owned = StridedArray::<T>::col_major(&dims);
            copy_into(&mut owned.view_mut(), &dv)?;
            diag_arr = Some(owned);
            unique_ids = unique_ids_tmp;
        }
    }
```

Note: This restructuring is complex. An alternative cleaner approach: simply change the
existing `diag_arr` to hold the `dv` view and reduce from it. But since `dv` is a local
`StridedView` borrowing from `src`, it can't be stored in `Option<StridedArray>`. The
approach above handles reduction immediately within the diagonal branch and stores the
result.

**Step 3: Run all tests**

Run: `cargo test -p strided-opteinsum`
Expected: All PASS

**Step 4: Commit**

```bash
git add strided-opteinsum/src/single_tensor.rs
git commit -m "perf: skip diagonal materialization when reduce follows"
```

---

### Task 6: Remove Option wrapping in reduce hot loop (Solution D)

Improves all non-contiguous full reductions in strided-kernel by 10-30%.
Benefits indexsum indirectly through reduced overhead.

**Files:**
- Modify: `strided-kernel/src/reduce_view.rs:129-152`
- Test: existing tests in strided-kernel

**Step 1: Run existing reduce tests**

Run: `cargo test -p strided-kernel reduce`
Expected: All PASS

**Step 2: Replace Option accumulator with plain value**

In `strided-kernel/src/reduce_view.rs`, replace lines 129-152 (the non-contiguous
`for_each_inner_block_preordered` callback):

```rust
    let mut acc = init;
    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            let mut ptr = unsafe { src_ptr.offset(offsets[0]) };
            let stride = strides[0];
            for _ in 0..len {
                let val = Op::apply(unsafe { *ptr });
                let mapped = map_fn(val);
                acc = reduce_fn(acc.clone(), mapped);
                unsafe {
                    ptr = ptr.offset(stride);
                }
            }
            Ok(())
        },
    )?;

    Ok(acc)
```

Note: The original `acc.take().expect(...)` was defensive but the `Option` was never `None`.
The new code uses plain `acc` with `clone()` (since `U: Clone` is already required).
For primitive types (`f64`, `Complex64`), `clone()` is a no-op copy.

**Step 3: Run all strided-kernel tests**

Run: `cargo test -p strided-kernel`
Expected: All PASS

**Step 4: Run full workspace tests**

Run: `cargo test`
Expected: All PASS

**Step 5: Commit**

```bash
git add strided-kernel/src/reduce_view.rs
git commit -m "perf: remove Option wrapping in reduce hot loop"
```

---

### Task 7: Run all benchmarks and update README

**Step 1: Run full benchmark suite**

```bash
RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum
```

**Step 2: Run Julia benchmarks for comparison**

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_hadamard.jl
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_trace.jl
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_ptrace.jl
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_diag.jl
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_perm.jl
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_indexsum.jl
```

**Step 3: Update benchmark table in README**

Update `strided-opteinsum/README.md` with new benchmark numbers.

**Step 4: Formatting check**

Run: `cargo fmt --check`
If fails: `cargo fmt`

**Step 5: Commit**

```bash
git add strided-opteinsum/README.md
git commit -m "docs: update benchmark results after performance optimizations"
```
