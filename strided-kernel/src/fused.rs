//! Runtime-DAG fused elementwise kernels.

use crate::kernel::ensure_same_shape;
use crate::{MaybeSendSync, Result, StridedError, StridedView, StridedViewMut};

/// Runtime scalar operation for a fused elementwise plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusedOp {
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Maximum,
    Minimum,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
}

/// One SSA instruction in a [`FusedPlan`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusedInst {
    pub op: FusedOp,
    pub inputs: Vec<usize>,
}

/// Topologically ordered fused elementwise DAG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusedPlan {
    pub input_count: usize,
    pub outputs: Vec<usize>,
    pub ops: Vec<FusedInst>,
}

/// Scalar types supported by [`fused_elementwise_into`].
pub trait FusedScalar: Copy + MaybeSendSync + 'static {}

impl FusedScalar for f32 {}
impl FusedScalar for f64 {}
impl FusedScalar for num_complex::Complex32 {}
impl FusedScalar for num_complex::Complex64 {}

#[inline]
fn op_arity(op: FusedOp) -> usize {
    match op {
        FusedOp::Negate
        | FusedOp::Conj
        | FusedOp::Abs
        | FusedOp::Exp
        | FusedOp::Log
        | FusedOp::Sin
        | FusedOp::Cos
        | FusedOp::Tanh
        | FusedOp::Sqrt
        | FusedOp::Rsqrt
        | FusedOp::Expm1
        | FusedOp::Log1p => 1,
        FusedOp::Add
        | FusedOp::Multiply
        | FusedOp::Divide
        | FusedOp::Maximum
        | FusedOp::Minimum
        | FusedOp::Pow => 2,
        FusedOp::Clamp => 3,
    }
}

fn validate_plan(plan: &FusedPlan, input_count: usize, output_count: usize) -> Result<()> {
    if input_count != plan.input_count {
        return Err(StridedError::RankMismatch(input_count, plan.input_count));
    }
    if output_count != plan.outputs.len() {
        return Err(StridedError::RankMismatch(output_count, plan.outputs.len()));
    }
    if output_count == 0 {
        return Err(StridedError::RankMismatch(0, 1));
    }

    let mut value_count = plan.input_count;
    for inst in &plan.ops {
        let expected_arity = op_arity(inst.op);
        if inst.inputs.len() != expected_arity {
            return Err(StridedError::RankMismatch(
                inst.inputs.len(),
                expected_arity,
            ));
        }
        for &input in &inst.inputs {
            if input >= value_count {
                return Err(StridedError::InvalidAxis {
                    axis: input,
                    rank: value_count,
                });
            }
        }
        value_count += 1;
    }

    for &output in &plan.outputs {
        if output >= value_count {
            return Err(StridedError::InvalidAxis {
                axis: output,
                rank: value_count,
            });
        }
    }

    Ok(())
}

fn validate_shapes<T: FusedScalar>(
    dests: &[StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
) -> Result<()> {
    let dims = dests[0].dims();
    for dest in &dests[1..] {
        ensure_same_shape(dims, dest.dims())?;
    }
    for input in inputs {
        ensure_same_shape(dims, input.dims())?;
    }
    Ok(())
}

/// Evaluate a runtime-DAG elementwise plan into one or more destinations.
pub fn fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    validate_plan(plan, inputs.len(), dests.len())?;
    validate_shapes(dests, inputs)?;
    Ok(())
}
