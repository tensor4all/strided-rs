use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use num_traits::Zero;
use strided_einsum2::{einsum2_into, einsum2_into_owned};
use strided_view::StridedArray;

use crate::operand::{EinsumOperand, StridedData};
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
            // Build dim_map from views before destructuring
            let a_dims: Vec<usize> = ld.dims().to_vec();
            let b_dims: Vec<usize> = rd.dims().to_vec();
            let mut dim_map: HashMap<char, usize> = HashMap::new();
            for (i, &id) in left_ids.iter().enumerate() {
                dim_map.insert(id, a_dims[i]);
            }
            for (i, &id) in right_ids.iter().enumerate() {
                dim_map.insert(id, b_dims[i]);
            }
            let out_dims = out_dims_from_map(&dim_map, output_ids)?;
            let mut c_arr = StridedArray::<f64>::row_major(&out_dims);

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
            let mut dim_map: HashMap<char, usize> = HashMap::new();
            for (i, &id) in left_ids.iter().enumerate() {
                dim_map.insert(id, a_dims[i]);
            }
            for (i, &id) in right_ids.iter().enumerate() {
                dim_map.insert(id, b_dims[i]);
            }
            let out_dims = out_dims_from_map(&dim_map, output_ids)?;
            let mut c_arr = StridedArray::<Complex64>::row_major(&out_dims);

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
}
