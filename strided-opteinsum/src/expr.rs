use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use num_traits::Zero;
use strided_einsum2::{einsum2_into, einsum2_into_owned};
use strided_kernel::copy_scale;
use strided_view::{StridedArray, StridedViewMut};

use crate::operand::{EinsumOperand, EinsumScalar, StridedData};
use crate::parse::{EinsumCode, EinsumNode};
use crate::single_tensor::single_tensor_einsum;

// ---------------------------------------------------------------------------
// Helper: collect all index chars from a subtree, preserving first-seen order
// ---------------------------------------------------------------------------

fn collect_all_ids(node: &EinsumNode) -> Vec<char> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    collect_all_ids_inner(node, &mut result, &mut seen);
    result
}

fn collect_all_ids_inner(node: &EinsumNode, result: &mut Vec<char>, seen: &mut HashSet<char>) {
    match node {
        EinsumNode::Leaf { ids, .. } => {
            for &id in ids {
                if seen.insert(id) {
                    result.push(id);
                }
            }
        }
        EinsumNode::Contract { args } => {
            for arg in args {
                collect_all_ids_inner(arg, result, seen);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: compute which output indices a Contract node should keep
// ---------------------------------------------------------------------------

/// For a Contract node with `args`, decide which index chars to keep.
///
/// An index is kept if it appears in `needed_ids` (what the parent/caller
/// needs from this node) AND it is actually present in at least one child
/// subtree.
fn compute_contract_output_ids(args: &[EinsumNode], needed_ids: &HashSet<char>) -> Vec<char> {
    // Walk args in order and collect ids preserving first-seen order
    let mut all_ids_ordered = Vec::new();
    let mut seen = HashSet::new();
    for arg in args {
        for id in collect_all_ids(arg) {
            if seen.insert(id) {
                all_ids_ordered.push(id);
            }
        }
    }

    // Keep only ids that the parent needs
    all_ids_ordered
        .into_iter()
        .filter(|id| needed_ids.contains(id))
        .collect()
}

/// Compute the set of ids that a child of a Contract node needs to provide.
///
/// A child needs to keep an index if:
///   - It is in the Contract's own `output_ids` (parent needs it), OR
///   - It is shared with at least one sibling subtree (contraction index).
///
/// The child is only responsible for indices in its own subtree, but this
/// function returns the full needed set; the child will naturally intersect
/// with its own ids.
fn compute_child_needed_ids(
    output_ids: &[char],
    child_idx: usize,
    args: &[EinsumNode],
) -> HashSet<char> {
    let mut needed: HashSet<char> = output_ids.iter().cloned().collect();

    // Add indices shared between this child and any sibling
    let child_ids: HashSet<char> = collect_all_ids(&args[child_idx]).into_iter().collect();
    for (j, arg) in args.iter().enumerate() {
        if j == child_idx {
            continue;
        }
        let sibling_ids: HashSet<char> = collect_all_ids(arg).into_iter().collect();
        for &id in &child_ids {
            if sibling_ids.contains(&id) {
                needed.insert(id);
            }
        }
    }

    needed
}

// ---------------------------------------------------------------------------
// Pairwise contraction (borrows operands)
// ---------------------------------------------------------------------------

fn out_dims_from_map(
    dim_map: &HashMap<char, usize>,
    output_ids: &[char],
) -> crate::Result<Vec<usize>> {
    let mut out_dims = Vec::with_capacity(output_ids.len());
    for &id in output_ids {
        match dim_map.get(&id) {
            Some(&dim) => out_dims.push(dim),
            None => return Err(crate::EinsumError::OrphanOutputAxis(id.to_string())),
        }
    }
    Ok(out_dims)
}

/// Look up output dimensions directly from left/right ids without HashMap allocation.
fn out_dims_from_ids(
    left_ids: &[char],
    left_dims: &[usize],
    right_ids: &[char],
    right_dims: &[usize],
    output_ids: &[char],
) -> crate::Result<Vec<usize>> {
    let mut out_dims = Vec::with_capacity(output_ids.len());
    for &id in output_ids {
        if let Some(pos) = left_ids.iter().position(|&c| c == id) {
            out_dims.push(left_dims[pos]);
        } else if let Some(pos) = right_ids.iter().position(|&c| c == id) {
            out_dims.push(right_dims[pos]);
        } else {
            return Err(crate::EinsumError::OrphanOutputAxis(id.to_string()));
        }
    }
    Ok(out_dims)
}

/// Contract two operands, consuming them by value. Promotes to c64 if types are mixed.
///
/// When both operands are `StridedData::Owned`, dispatches to `einsum2_into_owned`
/// which can reuse the input buffers during contiguous preparation, avoiding copies.
fn eval_pair(
    left: EinsumOperand<'_>,
    left_ids: &[char],
    right: EinsumOperand<'_>,
    right_ids: &[char],
    output_ids: &[char],
) -> crate::Result<EinsumOperand<'static>> {
    match (left, right) {
        (EinsumOperand::F64(ld), EinsumOperand::F64(rd)) => {
            let a_dims: Vec<usize> = ld.dims().to_vec();
            let b_dims: Vec<usize> = rd.dims().to_vec();
            let out_dims = out_dims_from_ids(left_ids, &a_dims, right_ids, &b_dims, output_ids)?;
            let mut c_arr = StridedArray::<f64>::col_major(&out_dims);

            match (ld, rd) {
                (StridedData::Owned(a), StridedData::Owned(b)) => {
                    einsum2_into_owned(
                        c_arr.view_mut(),
                        a,
                        b,
                        output_ids,
                        left_ids,
                        right_ids,
                        1.0,
                        0.0,
                        false,
                        false,
                    )?;
                }
                (StridedData::Owned(a), StridedData::View(b)) => {
                    einsum2_into(
                        c_arr.view_mut(),
                        &a.view(),
                        &b,
                        output_ids,
                        left_ids,
                        right_ids,
                        1.0,
                        0.0,
                    )?;
                }
                (StridedData::View(a), StridedData::Owned(b)) => {
                    einsum2_into(
                        c_arr.view_mut(),
                        &a,
                        &b.view(),
                        output_ids,
                        left_ids,
                        right_ids,
                        1.0,
                        0.0,
                    )?;
                }
                (StridedData::View(a), StridedData::View(b)) => {
                    einsum2_into(
                        c_arr.view_mut(),
                        &a,
                        &b,
                        output_ids,
                        left_ids,
                        right_ids,
                        1.0,
                        0.0,
                    )?;
                }
            }
            Ok(EinsumOperand::F64(StridedData::Owned(c_arr)))
        }
        (EinsumOperand::C64(ld), EinsumOperand::C64(rd)) => {
            let a_dims: Vec<usize> = ld.dims().to_vec();
            let b_dims: Vec<usize> = rd.dims().to_vec();
            let out_dims = out_dims_from_ids(left_ids, &a_dims, right_ids, &b_dims, output_ids)?;
            let mut c_arr = StridedArray::<Complex64>::col_major(&out_dims);

            match (ld, rd) {
                (StridedData::Owned(a), StridedData::Owned(b)) => {
                    einsum2_into_owned(
                        c_arr.view_mut(),
                        a,
                        b,
                        output_ids,
                        left_ids,
                        right_ids,
                        Complex64::new(1.0, 0.0),
                        Complex64::zero(),
                        false,
                        false,
                    )?;
                }
                (StridedData::Owned(a), StridedData::View(b)) => {
                    einsum2_into(
                        c_arr.view_mut(),
                        &a.view(),
                        &b,
                        output_ids,
                        left_ids,
                        right_ids,
                        Complex64::new(1.0, 0.0),
                        Complex64::zero(),
                    )?;
                }
                (StridedData::View(a), StridedData::Owned(b)) => {
                    einsum2_into(
                        c_arr.view_mut(),
                        &a,
                        &b.view(),
                        output_ids,
                        left_ids,
                        right_ids,
                        Complex64::new(1.0, 0.0),
                        Complex64::zero(),
                    )?;
                }
                (StridedData::View(a), StridedData::View(b)) => {
                    einsum2_into(
                        c_arr.view_mut(),
                        &a,
                        &b,
                        output_ids,
                        left_ids,
                        right_ids,
                        Complex64::new(1.0, 0.0),
                        Complex64::zero(),
                    )?;
                }
            }
            Ok(EinsumOperand::C64(StridedData::Owned(c_arr)))
        }
        (left, right) => {
            // Mixed types: promote both to c64 by consuming, then recurse (hits C64/C64 branch)
            let left_c64 = left.to_c64_owned();
            let right_c64 = right.to_c64_owned();
            eval_pair(left_c64, left_ids, right_c64, right_ids, output_ids)
        }
    }
}

// ---------------------------------------------------------------------------
// Pairwise contraction into user-provided output (zero-copy for final step)
// ---------------------------------------------------------------------------

/// Contract two operands directly into a user-provided output buffer.
///
/// Unlike `eval_pair`, this writes into `output` with alpha/beta scaling
/// instead of allocating a fresh array. Used for the final contraction in
/// `evaluate_into`.
fn eval_pair_into<T: EinsumScalar>(
    left: EinsumOperand<'_>,
    left_ids: &[char],
    right: EinsumOperand<'_>,
    right_ids: &[char],
    output: StridedViewMut<T>,
    output_ids: &[char],
    alpha: T,
    beta: T,
) -> crate::Result<()> {
    let left_data = T::extract_data(left)?;
    let right_data = T::extract_data(right)?;

    match (left_data, right_data) {
        (StridedData::Owned(a), StridedData::Owned(b)) => {
            einsum2_into_owned(
                output, a, b, output_ids, left_ids, right_ids, alpha, beta, false, false,
            )?;
        }
        (StridedData::Owned(a), StridedData::View(b)) => {
            einsum2_into(
                output,
                &a.view(),
                &b,
                output_ids,
                left_ids,
                right_ids,
                alpha,
                beta,
            )?;
        }
        (StridedData::View(a), StridedData::Owned(b)) => {
            einsum2_into(
                output,
                &a,
                &b.view(),
                output_ids,
                left_ids,
                right_ids,
                alpha,
                beta,
            )?;
        }
        (StridedData::View(a), StridedData::View(b)) => {
            einsum2_into(output, &a, &b, output_ids, left_ids, right_ids, alpha, beta)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Accumulate helper for single-tensor results
// ---------------------------------------------------------------------------

/// Write `output = alpha * result + beta * output`.
///
/// `result` must already have the same shape as `output`.
fn accumulate_into<T: EinsumScalar>(
    output: &mut StridedViewMut<T>,
    result: &StridedArray<T>,
    alpha: T,
    beta: T,
) -> crate::Result<()> {
    let result_view = result.view();
    if beta == T::zero() {
        if alpha == T::one() {
            strided_kernel::copy_into(output, &result_view)?;
        } else {
            copy_scale(output, &result_view, alpha)?;
        }
    } else {
        // General case: output = alpha * result + beta * output
        // axpy does: output += alpha * result, so we need to scale output by beta first.
        // We use a temporary to avoid aliasing issues.
        let dims = output.dims().to_vec();
        let mut temp = StridedArray::<T>::col_major(&dims);
        strided_kernel::copy_into(&mut temp.view_mut(), &result_view)?;
        // temp now holds result data in col-major layout
        // output = beta * output + alpha * temp
        // Using zip_map2_into would need output as both src and dest.
        // Instead: copy_scale output into a second temp, then zip_map2_into.
        // But simpler: use axpy which reads+writes dest.
        // axpy(dest, src, alpha) does: dest[i] = dest[i] + alpha * src[i]
        // So: first scale output by beta, then axpy with alpha.
        // "scale output by beta" = copy_scale into temp2, copy back. Or just
        // use a different approach: compute full result in temp, copy to output.
        //
        // Simplest correct approach for this rare path:
        let mut output_copy = StridedArray::<T>::col_major(&dims);
        strided_kernel::copy_into(&mut output_copy.view_mut(), &output.as_view())?;
        strided_kernel::zip_map2_into(output, &temp.view(), &output_copy.view(), |r, o| {
            alpha * r + beta * o
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Single-tensor dispatch (borrows operand)
// ---------------------------------------------------------------------------

fn eval_single(
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

// ---------------------------------------------------------------------------
// Permutation helpers
// ---------------------------------------------------------------------------

/// Check if the transformation from input_ids to output_ids is a pure
/// permutation (same set of chars, same length, no repeated indices).
fn is_permutation_only(input_ids: &[char], output_ids: &[char]) -> bool {
    if input_ids.len() != output_ids.len() {
        return false;
    }
    let mut seen = HashSet::new();
    for &id in input_ids {
        if !seen.insert(id) {
            return false; // repeated index = trace, not permutation
        }
    }
    for &id in output_ids {
        if !seen.contains(&id) {
            return false;
        }
    }
    true
}

/// Compute the permutation that maps input_ids ordering to output_ids ordering.
fn compute_permutation(input_ids: &[char], output_ids: &[char]) -> Vec<usize> {
    output_ids
        .iter()
        .map(|oid| input_ids.iter().position(|iid| iid == oid).unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// Execute omeco NestedEinsum tree
// ---------------------------------------------------------------------------

/// Execute an omeco-optimized contraction tree by contracting pairs according
/// to the tree structure.
///
/// `children` is a Vec of Option-wrapped (operand, ids) pairs. The omeco
/// `NestedEinsum::Leaf` variant references children by index; we `.take()`
/// each entry to move ownership out exactly once.
fn execute_nested<'a>(
    nested: &omeco::NestedEinsum<char>,
    children: &mut Vec<Option<(EinsumOperand<'a>, Vec<char>)>>,
) -> crate::Result<(EinsumOperand<'a>, Vec<char>)> {
    match nested {
        omeco::NestedEinsum::Leaf { tensor_index } => {
            let slot = children.get_mut(*tensor_index).ok_or_else(|| {
                crate::EinsumError::Internal(format!(
                    "optimizer referenced child index {} out of bounds",
                    tensor_index
                ))
            })?;
            let (op, ids) = slot.take().ok_or_else(|| {
                crate::EinsumError::Internal(format!(
                    "child operand {} was already consumed",
                    tensor_index
                ))
            })?;
            Ok((op, ids))
        }
        omeco::NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return Err(crate::EinsumError::Internal(format!(
                    "optimizer produced non-binary node with {} children",
                    args.len()
                )));
            }
            let (left, left_ids) = execute_nested(&args[0], children)?;
            let (right, right_ids) = execute_nested(&args[1], children)?;
            let output_ids: Vec<char> = eins.iy.clone();
            let result = eval_pair(left, &left_ids, right, &right_ids, &output_ids)?;
            Ok((result, output_ids))
        }
    }
}

/// Execute an omeco-optimized contraction tree, writing the root contraction
/// directly into a user-provided output buffer.
///
/// Inner (non-root) contractions use normal `execute_nested` / `eval_pair`.
/// Only the root `Node`'s contraction is written directly into `output`.
fn execute_nested_into<'a, T: EinsumScalar>(
    nested: &omeco::NestedEinsum<char>,
    children: &mut Vec<Option<(EinsumOperand<'a>, Vec<char>)>>,
    output: StridedViewMut<T>,
    output_ids: &[char],
    alpha: T,
    beta: T,
) -> crate::Result<()> {
    match nested {
        omeco::NestedEinsum::Node { args, eins: _ } => {
            if args.len() != 2 {
                return Err(crate::EinsumError::Internal(format!(
                    "optimizer produced non-binary node with {} children",
                    args.len()
                )));
            }
            // Evaluate children normally (they allocate temporaries)
            let (left, left_ids) = execute_nested(&args[0], children)?;
            let (right, right_ids) = execute_nested(&args[1], children)?;
            // Root contraction writes directly into user's output
            eval_pair_into(
                left, &left_ids, right, &right_ids, output, output_ids, alpha, beta,
            )
        }
        omeco::NestedEinsum::Leaf { tensor_index } => {
            // Root is a single leaf — extract and accumulate into output
            let slot = children.get_mut(*tensor_index).ok_or_else(|| {
                crate::EinsumError::Internal(format!(
                    "optimizer referenced child index {} out of bounds",
                    tensor_index
                ))
            })?;
            let (op, op_ids) = slot.take().ok_or_else(|| {
                crate::EinsumError::Internal(format!(
                    "child operand {} was already consumed",
                    tensor_index
                ))
            })?;
            let data = T::extract_data(op)?;
            let arr = data.into_array();
            // Permute if needed
            if op_ids != output_ids {
                let perm = compute_permutation(&op_ids, output_ids);
                let permuted = arr.permuted(&perm)?;
                accumulate_into(&mut { output }, &permuted, alpha, beta)?;
            } else {
                accumulate_into(&mut { output }, &arr, alpha, beta)?;
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Recursive evaluation
// ---------------------------------------------------------------------------

/// Recursively evaluate an `EinsumNode`, returning the result operand and
/// the index chars labelling its axes.
///
/// Leaf nodes return borrowed views directly (no copy). Contract nodes
/// always produce freshly allocated results (`'static` coerced to `'a`).
///
/// `needed_ids` tells this node which indices the caller needs in the result.
/// For the root call this is the final output indices of the einsum.
fn eval_node<'a>(
    node: &EinsumNode,
    operands: &mut Vec<Option<EinsumOperand<'a>>>,
    needed_ids: &HashSet<char>,
) -> crate::Result<(EinsumOperand<'a>, Vec<char>)> {
    match node {
        EinsumNode::Leaf { ids, tensor_index } => {
            let found = operands.len();
            let slot = operands.get_mut(*tensor_index).ok_or_else(|| {
                crate::EinsumError::OperandCountMismatch {
                    expected: tensor_index + 1,
                    found,
                }
            })?;
            let op = slot.take().ok_or_else(|| {
                crate::EinsumError::Internal(format!(
                    "operand {} was already consumed",
                    tensor_index
                ))
            })?;
            // Return borrowed view directly — no to_owned_static() copy.
            Ok((op, ids.clone()))
        }
        EinsumNode::Contract { args } => {
            // Determine which indices this Contract node should output.
            let node_output_ids = compute_contract_output_ids(args, needed_ids);

            match args.len() {
                0 => unreachable!("empty Contract node"),
                1 => {
                    // Single-tensor operation.
                    let child_needed = compute_child_needed_ids(&node_output_ids, 0, args);
                    let (child_op, child_ids) = eval_node(&args[0], operands, &child_needed)?;

                    // Identity passthrough: no allocation needed.
                    if child_ids == node_output_ids {
                        return Ok((child_op, node_output_ids));
                    }

                    // Permutation-only passthrough: metadata reorder, no data copy.
                    if is_permutation_only(&child_ids, &node_output_ids) {
                        let perm = compute_permutation(&child_ids, &node_output_ids);
                        return Ok((child_op.permuted(&perm)?, node_output_ids));
                    }

                    // General case: trace, reduction, etc.
                    let result = eval_single(&child_op, &child_ids, &node_output_ids)?;
                    Ok((result, node_output_ids))
                }
                2 => {
                    // Binary contraction.
                    let left_needed = compute_child_needed_ids(&node_output_ids, 0, args);
                    let right_needed = compute_child_needed_ids(&node_output_ids, 1, args);
                    let (left, left_ids) = eval_node(&args[0], operands, &left_needed)?;
                    let (right, right_ids) = eval_node(&args[1], operands, &right_needed)?;
                    let result = eval_pair(left, &left_ids, right, &right_ids, &node_output_ids)?;
                    Ok((result, node_output_ids))
                }
                _ => {
                    // 3+ children: use omeco greedy optimizer to find
                    // an efficient pairwise contraction order.

                    // 1. Evaluate all children to get their operands and ids
                    let mut children: Vec<Option<(EinsumOperand<'a>, Vec<char>)>> = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        let child_needed = compute_child_needed_ids(&node_output_ids, i, args);
                        let (op, ids) = eval_node(arg, operands, &child_needed)?;
                        children.push(Some((op, ids)));
                    }

                    // 2. Build dimension sizes map from evaluated operands
                    let mut dim_sizes: HashMap<char, usize> = HashMap::new();
                    for child_opt in &children {
                        if let Some((op, ids)) = child_opt {
                            for (j, &id) in ids.iter().enumerate() {
                                dim_sizes.insert(id, op.dims()[j]);
                            }
                        }
                    }

                    // 3. Build omeco EinCode from child ids and node output ids
                    let input_ids: Vec<Vec<char>> = children
                        .iter()
                        .map(|c| c.as_ref().unwrap().1.clone())
                        .collect();
                    let code = omeco::EinCode::new(input_ids, node_output_ids.clone());

                    // 4. Optimize using omeco greedy method
                    let optimizer = omeco::GreedyMethod::default();
                    let nested = omeco::CodeOptimizer::optimize(&optimizer, &code, &dim_sizes)
                        .ok_or_else(|| {
                            crate::EinsumError::Internal(
                                "optimizer failed to produce a plan".into(),
                            )
                        })?;

                    // 5. Execute the nested contraction tree
                    let (result, result_ids) = execute_nested(&nested, &mut children)?;
                    Ok((result, result_ids))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl EinsumCode {
    /// Evaluate the einsum contraction tree with the given operands.
    ///
    /// Borrowed view operands are propagated through the tree without copying.
    /// The result lifetime matches the input operand lifetime: if all inputs
    /// are owned (`'static`), the result is also `'static`.
    pub fn evaluate<'a>(
        &self,
        operands: Vec<EinsumOperand<'a>>,
    ) -> crate::Result<EinsumOperand<'a>> {
        let expected = leaf_count(&self.root);
        if operands.len() != expected {
            return Err(crate::EinsumError::OperandCountMismatch {
                expected,
                found: operands.len(),
            });
        }

        let final_output: HashSet<char> = self.output_ids.iter().cloned().collect();
        let mut ops: Vec<Option<EinsumOperand<'a>>> = operands.into_iter().map(Some).collect();

        let (result, result_ids) = eval_node(&self.root, &mut ops, &final_output)?;

        // If the result ids already match the desired output, we're done.
        if result_ids == self.output_ids {
            return Ok(result);
        }

        // Permutation-only: reorder metadata, no data copy.
        if is_permutation_only(&result_ids, &self.output_ids) {
            let perm = compute_permutation(&result_ids, &self.output_ids);
            return Ok(result.permuted(&perm)?);
        }

        // General fallback: reduce/trace to match the final output_ids.
        let adjusted = eval_single(&result, &result_ids, &self.output_ids)?;
        Ok(adjusted)
    }
}

fn leaf_count(node: &EinsumNode) -> usize {
    match node {
        EinsumNode::Leaf { .. } => 1,
        EinsumNode::Contract { args } => args.iter().map(leaf_count).sum(),
    }
}

/// Build a dimension map from operands and the parsed tree.
///
/// Maps each index char to its dimension size by walking the tree's Leaf nodes
/// and matching them to the corresponding operands.
fn build_dim_map(
    node: &EinsumNode,
    operands: &[Option<EinsumOperand<'_>>],
) -> HashMap<char, usize> {
    let mut dim_map = HashMap::new();
    build_dim_map_inner(node, operands, &mut dim_map);
    dim_map
}

fn build_dim_map_inner(
    node: &EinsumNode,
    operands: &[Option<EinsumOperand<'_>>],
    dim_map: &mut HashMap<char, usize>,
) {
    match node {
        EinsumNode::Leaf { ids, tensor_index } => {
            if let Some(Some(op)) = operands.get(*tensor_index) {
                for (i, &id) in ids.iter().enumerate() {
                    dim_map.insert(id, op.dims()[i]);
                }
            }
        }
        EinsumNode::Contract { args } => {
            for arg in args {
                build_dim_map_inner(arg, operands, dim_map);
            }
        }
    }
}

impl EinsumCode {
    /// Evaluate the einsum contraction tree, writing the result directly into
    /// a user-provided output buffer with alpha/beta scaling.
    ///
    /// `output = alpha * einsum(operands) + beta * output`
    ///
    /// The output element type `T` must match the computation: use `f64` when
    /// all operands are real, `Complex64` when any operand is complex.
    /// If `T = f64` but any operand is complex, returns `TypeMismatch` error.
    /// If `T = Complex64`, real operands are promoted automatically.
    pub fn evaluate_into<T: EinsumScalar>(
        &self,
        operands: Vec<EinsumOperand<'_>>,
        mut output: StridedViewMut<T>,
        alpha: T,
        beta: T,
    ) -> crate::Result<()> {
        let expected = leaf_count(&self.root);
        if operands.len() != expected {
            return Err(crate::EinsumError::OperandCountMismatch {
                expected,
                found: operands.len(),
            });
        }

        // Validate output type compatibility
        let mut ops: Vec<Option<EinsumOperand<'_>>> = operands.into_iter().map(Some).collect();
        T::validate_operands(&ops)?;

        // Compute expected output shape
        let dim_map = build_dim_map(&self.root, &ops);
        let expected_dims = out_dims_from_map(&dim_map, &self.output_ids)?;
        if output.dims() != expected_dims.as_slice() {
            return Err(crate::EinsumError::OutputShapeMismatch {
                expected: expected_dims,
                got: output.dims().to_vec(),
            });
        }

        let final_output: HashSet<char> = self.output_ids.iter().cloned().collect();

        match &self.root {
            EinsumNode::Leaf { ids, tensor_index } => {
                // Single operand: extract, permute/trace, accumulate
                let op = ops[*tensor_index].take().ok_or_else(|| {
                    crate::EinsumError::Internal("operand already consumed".into())
                })?;
                let single_result = eval_single(&op, ids, &self.output_ids)?;
                let data = T::extract_data(single_result)?;
                accumulate_into(&mut output, &data.into_array(), alpha, beta)?;
            }
            EinsumNode::Contract { args } => match args.len() {
                0 => unreachable!("empty Contract node"),
                1 => {
                    // Single child: evaluate, then accumulate
                    let child_needed = compute_child_needed_ids(&self.output_ids, 0, args);
                    let (child_op, child_ids) = eval_node(&args[0], &mut ops, &child_needed)?;

                    if child_ids == self.output_ids {
                        // Identity: just accumulate
                        let data = T::extract_data(child_op)?;
                        accumulate_into(&mut output, &data.into_array(), alpha, beta)?;
                    } else if is_permutation_only(&child_ids, &self.output_ids) {
                        // Permutation: permute the data, then accumulate
                        let perm = compute_permutation(&child_ids, &self.output_ids);
                        let data = T::extract_data(child_op)?;
                        let arr = data.into_array();
                        let permuted = arr.permuted(&perm)?;
                        accumulate_into(&mut output, &permuted, alpha, beta)?;
                    } else {
                        // General: trace/reduction
                        let result = eval_single(&child_op, &child_ids, &self.output_ids)?;
                        let data = T::extract_data(result)?;
                        accumulate_into(&mut output, &data.into_array(), alpha, beta)?;
                    }
                }
                2 => {
                    // Binary contraction: write directly into output
                    let left_needed = compute_child_needed_ids(&self.output_ids, 0, args);
                    let right_needed = compute_child_needed_ids(&self.output_ids, 1, args);
                    let (left, left_ids) = eval_node(&args[0], &mut ops, &left_needed)?;
                    let (right, right_ids) = eval_node(&args[1], &mut ops, &right_needed)?;
                    eval_pair_into(
                        left,
                        &left_ids,
                        right,
                        &right_ids,
                        output,
                        &self.output_ids,
                        alpha,
                        beta,
                    )?;
                }
                _ => {
                    // 3+ children: use omeco, final contraction into output
                    let node_output_ids = compute_contract_output_ids(args, &final_output);

                    let mut children: Vec<Option<(EinsumOperand<'_>, Vec<char>)>> = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        let child_needed = compute_child_needed_ids(&node_output_ids, i, args);
                        let (op, ids) = eval_node(arg, &mut ops, &child_needed)?;
                        children.push(Some((op, ids)));
                    }

                    let mut dim_sizes: HashMap<char, usize> = HashMap::new();
                    for child_opt in &children {
                        if let Some((op, ids)) = child_opt {
                            for (j, &id) in ids.iter().enumerate() {
                                dim_sizes.insert(id, op.dims()[j]);
                            }
                        }
                    }

                    let input_ids: Vec<Vec<char>> = children
                        .iter()
                        .map(|c| c.as_ref().unwrap().1.clone())
                        .collect();
                    let code = omeco::EinCode::new(input_ids, self.output_ids.clone());

                    let optimizer = omeco::GreedyMethod::default();
                    let nested = omeco::CodeOptimizer::optimize(&optimizer, &code, &dim_sizes)
                        .ok_or_else(|| {
                            crate::EinsumError::Internal(
                                "optimizer failed to produce a plan".into(),
                            )
                        })?;

                    execute_nested_into(
                        &nested,
                        &mut children,
                        output,
                        &self.output_ids,
                        alpha,
                        beta,
                    )?;
                }
            },
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_einsum;
    use approx::assert_abs_diff_eq;
    use strided_view::{row_major_strides, StridedArray};

    fn make_f64(dims: &[usize], data: Vec<f64>) -> EinsumOperand<'static> {
        let strides = row_major_strides(dims);
        StridedArray::from_parts(data, dims, &strides, 0)
            .unwrap()
            .into()
    }

    #[test]
    fn test_matmul() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let result = code.evaluate(vec![a, b]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[2, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0);
                assert_abs_diff_eq!(arr.get(&[0, 1]), 22.0);
                assert_abs_diff_eq!(arr.get(&[1, 0]), 43.0);
                assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_nested_three_tensor() {
        let code = parse_einsum("(ij,jk),kl->il").unwrap();
        // A = [[1,0],[0,1]] (identity), B = [[1,2],[3,4]], C = [[5,6],[7,8]]
        let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let b = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let c = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        // A*B = B, B*C = [[19,22],[43,50]]
        let result = code.evaluate(vec![a, b, c]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[2, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0);
                assert_abs_diff_eq!(arr.get(&[0, 1]), 22.0);
                assert_abs_diff_eq!(arr.get(&[1, 0]), 43.0);
                assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_outer_product() {
        let code = parse_einsum("i,j->ij").unwrap();
        let a = make_f64(&[3], vec![1.0, 2.0, 3.0]);
        let b = make_f64(&[2], vec![10.0, 20.0]);
        let result = code.evaluate(vec![a, b]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[3, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 10.0);
                assert_abs_diff_eq!(arr.get(&[2, 1]), 60.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_dot_product() {
        let code = parse_einsum("i,i->").unwrap();
        let a = make_f64(&[3], vec![1.0, 2.0, 3.0]);
        let b = make_f64(&[3], vec![4.0, 5.0, 6.0]);
        let result = code.evaluate(vec![a, b]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                // 1*4 + 2*5 + 3*6 = 32
                assert_abs_diff_eq!(data.as_array().data()[0], 32.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_single_tensor_permute() {
        let code = parse_einsum("ij->ji").unwrap();
        let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = code.evaluate(vec![a]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[3, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 1.0);
                assert_abs_diff_eq!(arr.get(&[0, 1]), 4.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_single_tensor_trace() {
        let code = parse_einsum("ii->").unwrap();
        let a = make_f64(&[3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        let result = code.evaluate(vec![a]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                assert_abs_diff_eq!(data.as_array().data()[0], 6.0);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_three_tensor_flat_omeco() {
        // ij,jk,kl->il -- flat 3-tensor, should use omeco
        let code = parse_einsum("ij,jk,kl->il").unwrap();
        let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f64(&[3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        // c = identity; AB = [[4,2],[10,5]], AB*I = [[4,2],[10,5]]
        let c = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let result = code.evaluate(vec![a, b, c]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[2, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 4.0, epsilon = 1e-10);
                assert_abs_diff_eq!(arr.get(&[0, 1]), 2.0, epsilon = 1e-10);
                assert_abs_diff_eq!(arr.get(&[1, 0]), 10.0, epsilon = 1e-10);
                assert_abs_diff_eq!(arr.get(&[1, 1]), 5.0, epsilon = 1e-10);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_four_tensor_flat_omeco() {
        // ij,jk,kl,lm->im -- 4-tensor chain
        let code = parse_einsum("ij,jk,kl,lm->im").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
        let b = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let c = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
        let d = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        // I*B = B, B*I = B, B*D = [[19,22],[43,50]]
        let result = code.evaluate(vec![a, b, c, d]).unwrap();
        match result {
            EinsumOperand::F64(data) => {
                let arr = data.as_array();
                assert_eq!(arr.dims(), &[2, 2]);
                assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0, epsilon = 1e-10);
                assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0, epsilon = 1e-10);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_orphan_output_axis_returns_error() {
        let code = parse_einsum("ij,jk->iz").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let err = code.evaluate(vec![a, b]).unwrap_err();
        assert!(matches!(err, crate::EinsumError::OrphanOutputAxis(ref s) if s == "z"));
    }

    #[test]
    fn test_operand_count_mismatch_too_few() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let err = code.evaluate(vec![a]).unwrap_err();
        assert!(matches!(
            err,
            crate::EinsumError::OperandCountMismatch {
                expected: 2,
                found: 1
            }
        ));
    }

    #[test]
    fn test_operand_count_mismatch_too_many() {
        let code = parse_einsum("ij->ji").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let err = code.evaluate(vec![a, b]).unwrap_err();
        assert!(matches!(
            err,
            crate::EinsumError::OperandCountMismatch {
                expected: 1,
                found: 2
            }
        ));
    }

    // -----------------------------------------------------------------------
    // evaluate_into tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_into_matmul() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let mut c = StridedArray::<f64>::col_major(&[2, 2]);
        code.evaluate_into(vec![a, b], c.view_mut(), 1.0, 0.0)
            .unwrap();
        assert_abs_diff_eq!(c.get(&[0, 0]), 19.0);
        assert_abs_diff_eq!(c.get(&[0, 1]), 22.0);
        assert_abs_diff_eq!(c.get(&[1, 0]), 43.0);
        assert_abs_diff_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_into_matmul_alpha_beta() {
        // C = 2 * A*B + 3 * C_old
        let code = parse_einsum("ij,jk->ik").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        // A*B = [[19,22],[43,50]]
        // C_old = [[1,1],[1,1]]
        // result = 2*[[19,22],[43,50]] + 3*[[1,1],[1,1]] = [[41,47],[89,103]]
        let mut c = StridedArray::<f64>::col_major(&[2, 2]);
        for v in c.data_mut().iter_mut() {
            *v = 1.0;
        }
        code.evaluate_into(vec![a, b], c.view_mut(), 2.0, 3.0)
            .unwrap();
        assert_abs_diff_eq!(c.get(&[0, 0]), 41.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c.get(&[0, 1]), 47.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c.get(&[1, 0]), 89.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c.get(&[1, 1]), 103.0, epsilon = 1e-10);
    }

    #[test]
    fn test_into_single_tensor_permute() {
        let code = parse_einsum("ij->ji").unwrap();
        let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut c = StridedArray::<f64>::col_major(&[3, 2]);
        code.evaluate_into(vec![a], c.view_mut(), 1.0, 0.0).unwrap();
        assert_eq!(c.dims(), &[3, 2]);
        assert_abs_diff_eq!(c.get(&[0, 0]), 1.0);
        assert_abs_diff_eq!(c.get(&[0, 1]), 4.0);
        assert_abs_diff_eq!(c.get(&[1, 0]), 2.0);
        assert_abs_diff_eq!(c.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_into_single_tensor_trace() {
        let code = parse_einsum("ii->").unwrap();
        let a = make_f64(&[3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        let mut c = StridedArray::<f64>::col_major(&[]);
        code.evaluate_into(vec![a], c.view_mut(), 1.0, 0.0).unwrap();
        assert_abs_diff_eq!(c.data()[0], 6.0);
    }

    #[test]
    fn test_into_three_tensor_omeco() {
        let code = parse_einsum("ij,jk,kl->il").unwrap();
        let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_f64(&[3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let c_op = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let mut out = StridedArray::<f64>::col_major(&[2, 2]);
        code.evaluate_into(vec![a, b, c_op], out.view_mut(), 1.0, 0.0)
            .unwrap();
        assert_abs_diff_eq!(out.get(&[0, 0]), 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out.get(&[0, 1]), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out.get(&[1, 0]), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out.get(&[1, 1]), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_into_nested() {
        let code = parse_einsum("(ij,jk),kl->il").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let b = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let c_op = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let mut out = StridedArray::<f64>::col_major(&[2, 2]);
        code.evaluate_into(vec![a, b, c_op], out.view_mut(), 1.0, 0.0)
            .unwrap();
        assert_abs_diff_eq!(out.get(&[0, 0]), 19.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out.get(&[0, 1]), 22.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out.get(&[1, 0]), 43.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out.get(&[1, 1]), 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_into_dot_product() {
        let code = parse_einsum("i,i->").unwrap();
        let a = make_f64(&[3], vec![1.0, 2.0, 3.0]);
        let b = make_f64(&[3], vec![4.0, 5.0, 6.0]);
        let mut c = StridedArray::<f64>::col_major(&[]);
        code.evaluate_into(vec![a, b], c.view_mut(), 1.0, 0.0)
            .unwrap();
        assert_abs_diff_eq!(c.data()[0], 32.0);
    }

    #[test]
    fn test_into_type_mismatch_f64_output_c64_input() {
        let code = parse_einsum("ij->ji").unwrap();
        let c64_data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let strides = row_major_strides(&[2, 2]);
        let arr = StridedArray::from_parts(c64_data, &[2, 2], &strides, 0).unwrap();
        let op = EinsumOperand::C64(StridedData::Owned(arr));
        let mut out = StridedArray::<f64>::col_major(&[2, 2]);
        let err = code
            .evaluate_into(vec![op], out.view_mut(), 1.0, 0.0)
            .unwrap_err();
        assert!(matches!(err, crate::EinsumError::TypeMismatch { .. }));
    }

    #[test]
    fn test_into_shape_mismatch() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let mut out = StridedArray::<f64>::col_major(&[3, 3]); // wrong shape
        let err = code
            .evaluate_into(vec![a, b], out.view_mut(), 1.0, 0.0)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::EinsumError::OutputShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_into_c64_output() {
        let code = parse_einsum("ij,jk->ik").unwrap();
        let c64 = |r| Complex64::new(r, 0.0);
        let a_data = vec![c64(1.0), c64(2.0), c64(3.0), c64(4.0)];
        let b_data = vec![c64(5.0), c64(6.0), c64(7.0), c64(8.0)];
        let strides = row_major_strides(&[2, 2]);
        let a = EinsumOperand::C64(StridedData::Owned(
            StridedArray::from_parts(a_data, &[2, 2], &strides, 0).unwrap(),
        ));
        let b = EinsumOperand::C64(StridedData::Owned(
            StridedArray::from_parts(b_data, &[2, 2], &strides, 0).unwrap(),
        ));
        let mut out = StridedArray::<Complex64>::col_major(&[2, 2]);
        code.evaluate_into(vec![a, b], out.view_mut(), c64(1.0), Complex64::zero())
            .unwrap();
        assert_abs_diff_eq!(out.get(&[0, 0]).re, 19.0);
        assert_abs_diff_eq!(out.get(&[1, 1]).re, 50.0);
    }

    #[test]
    fn test_into_mixed_types_c64_output() {
        // f64 + c64 operands -> c64 output (f64 gets promoted)
        let code = parse_einsum("ij,jk->ik").unwrap();
        let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let c64 = |r| Complex64::new(r, 0.0);
        let b_data = vec![c64(5.0), c64(6.0), c64(7.0), c64(8.0)];
        let strides = row_major_strides(&[2, 2]);
        let b = EinsumOperand::C64(StridedData::Owned(
            StridedArray::from_parts(b_data, &[2, 2], &strides, 0).unwrap(),
        ));
        let mut out = StridedArray::<Complex64>::col_major(&[2, 2]);
        code.evaluate_into(vec![a, b], out.view_mut(), c64(1.0), Complex64::zero())
            .unwrap();
        assert_abs_diff_eq!(out.get(&[0, 0]).re, 19.0);
        assert_abs_diff_eq!(out.get(&[1, 1]).re, 50.0);
    }
}
