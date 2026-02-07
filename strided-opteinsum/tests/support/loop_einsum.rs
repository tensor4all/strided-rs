use std::collections::HashMap;

use num_complex::Complex64;
use strided_opteinsum::{parse_einsum, EinsumNode};
use strided_view::{row_major_strides, StridedArray};

#[derive(Clone)]
pub enum LoopTensor {
    F64(StridedArray<f64>),
    C64(StridedArray<Complex64>),
}

impl LoopTensor {
    pub fn dims(&self) -> &[usize] {
        match self {
            LoopTensor::F64(a) => a.dims(),
            LoopTensor::C64(a) => a.dims(),
        }
    }
}

pub fn row_major_f64(dims: &[usize], data: Vec<f64>) -> LoopTensor {
    let strides = row_major_strides(dims);
    let arr = StridedArray::from_parts(data, dims, &strides, 0).expect("valid row-major f64 array");
    LoopTensor::F64(arr)
}

pub fn row_major_c64(dims: &[usize], data: Vec<Complex64>) -> LoopTensor {
    let strides = row_major_strides(dims);
    let arr = StridedArray::from_parts(data, dims, &strides, 0).expect("valid row-major c64 array");
    LoopTensor::C64(arr)
}

fn collect_leaf_ids(node: &EinsumNode, out: &mut Vec<Option<Vec<char>>>) -> Result<(), String> {
    match node {
        EinsumNode::Leaf { ids, tensor_index } => {
            if *tensor_index >= out.len() {
                out.resize_with(*tensor_index + 1, || None);
            }
            if out[*tensor_index].is_some() {
                return Err(format!("duplicate tensor index {tensor_index}"));
            }
            out[*tensor_index] = Some(ids.clone());
            Ok(())
        }
        EinsumNode::Contract { args } => {
            for arg in args {
                collect_leaf_ids(arg, out)?;
            }
            Ok(())
        }
    }
}

fn operand_value(operand: &LoopTensor, positions: &[usize], assignment: &[usize]) -> Complex64 {
    let mut idx = Vec::with_capacity(positions.len());
    for &pos in positions {
        idx.push(assignment[pos]);
    }
    match operand {
        LoopTensor::F64(arr) => Complex64::new(arr.get(&idx), 0.0),
        LoopTensor::C64(arr) => arr.get(&idx),
    }
}

fn accumulate_recursive(
    depth: usize,
    label_dims: &[usize],
    assignment: &mut [usize],
    operands: &[LoopTensor],
    op_positions: &[Vec<usize>],
    out: &mut StridedArray<Complex64>,
    out_positions: &[usize],
) {
    if depth == label_dims.len() {
        let mut term = Complex64::new(1.0, 0.0);
        for (operand, positions) in operands.iter().zip(op_positions.iter()) {
            term *= operand_value(operand, positions, assignment);
        }

        let mut out_idx = Vec::with_capacity(out_positions.len());
        for &pos in out_positions {
            out_idx.push(assignment[pos]);
        }
        let prev = out.get(&out_idx);
        out.set(&out_idx, prev + term);
        return;
    }

    for i in 0..label_dims[depth] {
        assignment[depth] = i;
        accumulate_recursive(
            depth + 1,
            label_dims,
            assignment,
            operands,
            op_positions,
            out,
            out_positions,
        );
    }
}

pub fn loop_einsum_complex(
    notation: &str,
    operands: &[LoopTensor],
) -> Result<StridedArray<Complex64>, String> {
    let code = parse_einsum(notation).map_err(|e| e.to_string())?;

    let mut leaf_ids_opt: Vec<Option<Vec<char>>> = vec![None; operands.len()];
    collect_leaf_ids(&code.root, &mut leaf_ids_opt)?;
    if leaf_ids_opt.len() != operands.len() || leaf_ids_opt.iter().any(|x| x.is_none()) {
        return Err(format!(
            "operand count mismatch in reference evaluator: expected {}, found {}",
            leaf_ids_opt.len(),
            operands.len()
        ));
    }
    let leaf_ids: Vec<Vec<char>> = leaf_ids_opt
        .into_iter()
        .map(|ids| ids.expect("checked above"))
        .collect();

    let mut dim_map: HashMap<char, usize> = HashMap::new();
    let mut label_order: Vec<char> = Vec::new();

    for (i, ids) in leaf_ids.iter().enumerate() {
        let dims = operands[i].dims();
        if ids.len() != dims.len() {
            return Err(format!(
                "rank mismatch for operand {i}: ids={}, dims={}",
                ids.len(),
                dims.len()
            ));
        }
        for (&id, &dim) in ids.iter().zip(dims.iter()) {
            match dim_map.get(&id) {
                Some(&existing) if existing != dim => {
                    return Err(format!(
                        "dimension mismatch for label '{id}': {existing} vs {dim}"
                    ));
                }
                Some(_) => {}
                None => {
                    dim_map.insert(id, dim);
                    label_order.push(id);
                }
            }
        }
    }

    for &id in &code.output_ids {
        if !dim_map.contains_key(&id) {
            return Err(format!(
                "orphan output label '{id}' is not supported in loop_einsum"
            ));
        }
    }

    let mut label_pos: HashMap<char, usize> = HashMap::new();
    for (i, &label) in label_order.iter().enumerate() {
        label_pos.insert(label, i);
    }

    let label_dims: Vec<usize> = label_order.iter().map(|label| dim_map[label]).collect();
    let op_positions: Vec<Vec<usize>> = leaf_ids
        .iter()
        .map(|ids| ids.iter().map(|id| label_pos[id]).collect())
        .collect();
    let out_positions: Vec<usize> = code.output_ids.iter().map(|id| label_pos[id]).collect();
    let out_dims: Vec<usize> = code.output_ids.iter().map(|id| dim_map[id]).collect();

    let mut out = StridedArray::<Complex64>::col_major(&out_dims);
    let mut assignment = vec![0usize; label_order.len()];
    accumulate_recursive(
        0,
        &label_dims,
        &mut assignment,
        operands,
        &op_positions,
        &mut out,
        &out_positions,
    );
    Ok(out)
}
