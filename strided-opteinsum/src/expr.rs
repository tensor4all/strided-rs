use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use num_traits::Zero;
use strided_einsum2::einsum2_into;
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
// Pairwise contraction helpers
// ---------------------------------------------------------------------------

fn eval_pair_f64(
    left: EinsumOperand<'_>,
    left_ids: &[char],
    right: EinsumOperand<'_>,
    right_ids: &[char],
    output_ids: &[char],
) -> crate::Result<EinsumOperand<'static>> {
    let left_data = match left {
        EinsumOperand::F64(d) => d,
        _ => unreachable!("eval_pair_f64 called with non-f64 left operand"),
    };
    let right_data = match right {
        EinsumOperand::F64(d) => d,
        _ => unreachable!("eval_pair_f64 called with non-f64 right operand"),
    };

    let a_view = left_data.as_view();
    let b_view = right_data.as_view();

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (i, &id) in left_ids.iter().enumerate() {
        dim_map.insert(id, a_view.dims()[i]);
    }
    for (i, &id) in right_ids.iter().enumerate() {
        dim_map.insert(id, b_view.dims()[i]);
    }

    let out_dims = out_dims_from_map(&dim_map, output_ids)?;

    // Allocate output (col-major)
    let mut c_arr = StridedArray::<f64>::col_major(&out_dims);

    // Call einsum2_into
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

fn eval_pair_c64(
    left: EinsumOperand<'_>,
    left_ids: &[char],
    right: EinsumOperand<'_>,
    right_ids: &[char],
    output_ids: &[char],
) -> crate::Result<EinsumOperand<'static>> {
    let left_data = match left {
        EinsumOperand::C64(d) => d,
        _ => unreachable!("eval_pair_c64 called with non-c64 left operand"),
    };
    let right_data = match right {
        EinsumOperand::C64(d) => d,
        _ => unreachable!("eval_pair_c64 called with non-c64 right operand"),
    };

    let a_view = left_data.as_view();
    let b_view = right_data.as_view();

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (i, &id) in left_ids.iter().enumerate() {
        dim_map.insert(id, a_view.dims()[i]);
    }
    for (i, &id) in right_ids.iter().enumerate() {
        dim_map.insert(id, b_view.dims()[i]);
    }

    let out_dims = out_dims_from_map(&dim_map, output_ids)?;

    // Allocate output (col-major)
    let mut c_arr = StridedArray::<Complex64>::col_major(&out_dims);

    // Call einsum2_into
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

/// Contract two operands, promoting to c64 if types are mixed.
fn eval_pair(
    left: EinsumOperand<'_>,
    left_ids: &[char],
    right: EinsumOperand<'_>,
    right_ids: &[char],
    output_ids: &[char],
) -> crate::Result<EinsumOperand<'static>> {
    match (&left, &right) {
        (EinsumOperand::F64(_), EinsumOperand::F64(_)) => {
            eval_pair_f64(left, left_ids, right, right_ids, output_ids)
        }
        (EinsumOperand::C64(_), EinsumOperand::C64(_)) => {
            eval_pair_c64(left, left_ids, right, right_ids, output_ids)
        }
        _ => {
            // Mixed types: promote both to c64
            let left_c64 = left.to_c64_owned();
            let right_c64 = right.to_c64_owned();
            eval_pair_c64(left_c64, left_ids, right_c64, right_ids, output_ids)
        }
    }
}

// ---------------------------------------------------------------------------
// Single-tensor dispatch
// ---------------------------------------------------------------------------

fn eval_single(
    operand: EinsumOperand<'_>,
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
// Execute omeco NestedEinsum tree
// ---------------------------------------------------------------------------

/// Execute an omeco-optimized contraction tree by contracting pairs according
/// to the tree structure.
///
/// `children` is a Vec of Option-wrapped (operand, ids) pairs. The omeco
/// `NestedEinsum::Leaf` variant references children by index; we `.take()`
/// each entry to move ownership out exactly once.
fn execute_nested(
    nested: &omeco::NestedEinsum<char>,
    children: &mut Vec<Option<(EinsumOperand<'static>, Vec<char>)>>,
) -> crate::Result<(EinsumOperand<'static>, Vec<char>)> {
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
/// `needed_ids` tells this node which indices the caller needs in the result.
/// For the root call this is the final output indices of the einsum.
fn eval_node(
    node: &EinsumNode,
    operands: &mut Vec<Option<EinsumOperand<'_>>>,
    needed_ids: &HashSet<char>,
) -> crate::Result<(EinsumOperand<'static>, Vec<char>)> {
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
            Ok((op.to_owned_static(), ids.clone()))
        }
        EinsumNode::Contract { args } => {
            // Determine which indices this Contract node should output.
            let node_output_ids = compute_contract_output_ids(args, needed_ids);

            match args.len() {
                0 => unreachable!("empty Contract node"),
                1 => {
                    // Single-tensor operation: child provides all its ids,
                    // then we apply single_tensor_einsum to transform to
                    // node_output_ids (handling trace, permutation, sum).
                    let child_needed = compute_child_needed_ids(&node_output_ids, 0, args);
                    let (operand, input_ids) = eval_node(&args[0], operands, &child_needed)?;
                    let result = eval_single(operand, &input_ids, &node_output_ids)?;
                    Ok((result, node_output_ids))
                }
                2 => {
                    // Binary contraction via einsum2_into.
                    // Each child needs to keep its contraction indices (shared
                    // with the sibling) plus any indices needed in the output.
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
                    let mut children: Vec<Option<(EinsumOperand<'static>, Vec<char>)>> = Vec::new();
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
// EinsumOperand: helper to convert to 'static lifetime
// ---------------------------------------------------------------------------

impl<'a> EinsumOperand<'a> {
    /// Convert to an owned operand with `'static` lifetime.
    fn to_owned_static(self) -> EinsumOperand<'static> {
        match self {
            EinsumOperand::F64(data) => EinsumOperand::F64(StridedData::Owned(data.into_array())),
            EinsumOperand::C64(data) => EinsumOperand::C64(StridedData::Owned(data.into_array())),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl EinsumCode {
    /// Evaluate the einsum contraction tree with the given operands.
    pub fn evaluate(
        &self,
        operands: Vec<EinsumOperand<'_>>,
    ) -> crate::Result<EinsumOperand<'static>> {
        let expected = leaf_count(&self.root);
        if operands.len() != expected {
            return Err(crate::EinsumError::OperandCountMismatch {
                expected,
                found: operands.len(),
            });
        }

        let final_output: HashSet<char> = self.output_ids.iter().cloned().collect();
        let mut ops: Vec<Option<EinsumOperand<'_>>> = operands.into_iter().map(Some).collect();

        let (result, result_ids) = eval_node(&self.root, &mut ops, &final_output)?;

        // If the result ids already match the desired output, we're done.
        if result_ids == self.output_ids {
            return Ok(result);
        }

        // Otherwise, permute/reduce to match the final output_ids.
        eval_single(result, &result_ids, &self.output_ids)
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
