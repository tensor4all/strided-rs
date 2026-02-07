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

    let out_dims: Vec<usize> = output_ids.iter().map(|id| dim_map[id]).collect();

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

    let out_dims: Vec<usize> = output_ids.iter().map(|id| dim_map[id]).collect();

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
            let op = operands[*tensor_index]
                .take()
                .expect("operand already consumed");
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
                    // 3+ children: left-to-right fallback.
                    // Contract args[0] with args[1], then result with args[2], etc.
                    let child_needed_0 = compute_child_needed_ids(&node_output_ids, 0, args);
                    let (mut current, mut current_ids) =
                        eval_node(&args[0], operands, &child_needed_0)?;

                    for (i, arg) in args[1..].iter().enumerate() {
                        let child_idx = i + 1;
                        let child_needed =
                            compute_child_needed_ids(&node_output_ids, child_idx, args);
                        let (next, next_ids) = eval_node(arg, operands, &child_needed)?;

                        // Compute intermediate output ids:
                        // Keep ids that are in the final node_output_ids OR
                        // appear in any remaining sibling subtree.
                        let remaining_ids: HashSet<char> = args[child_idx + 1..]
                            .iter()
                            .flat_map(|a| collect_all_ids(a))
                            .collect();

                        let intermediate_output: Vec<char> = {
                            let mut all = Vec::new();
                            let mut seen = HashSet::new();
                            for &id in current_ids.iter().chain(next_ids.iter()) {
                                if seen.insert(id) {
                                    all.push(id);
                                }
                            }
                            all.into_iter()
                                .filter(|id| needed_ids.contains(id) || remaining_ids.contains(id))
                                .collect()
                        };

                        current = eval_pair(
                            current,
                            &current_ids,
                            next,
                            &next_ids,
                            &intermediate_output,
                        )?;
                        current_ids = intermediate_output;
                    }

                    Ok((current, current_ids))
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
}
