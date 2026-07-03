//! Runtime-DAG fused elementwise kernels.

use crate::kernel::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    total_len, SMALL_TENSOR_THRESHOLD,
};
use crate::{MaybeSendSync, Result, StridedError, StridedView, StridedViewMut};

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{
    for_each_inner_block_with_offsets, mapreduce_threaded, SendPtr, MINTHREADLENGTH,
};

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
pub trait FusedScalar: Copy + MaybeSendSync + 'static {
    fn fused_add(self, rhs: Self) -> Self;
    fn fused_multiply(self, rhs: Self) -> Self;
    fn fused_negate(self) -> Self;
    fn fused_conj(self) -> Self;
    fn fused_divide(self, rhs: Self) -> Self;
    fn fused_abs(self) -> Self;
    fn fused_maximum(self, rhs: Self) -> Self;
    fn fused_minimum(self, rhs: Self) -> Self;
    fn fused_clamp(self, min: Self, max: Self) -> Self;
    fn fused_exp(self) -> Self;
    fn fused_log(self) -> Self;
    fn fused_sin(self) -> Self;
    fn fused_cos(self) -> Self;
    fn fused_tanh(self) -> Self;
    fn fused_sqrt(self) -> Self;
    fn fused_rsqrt(self) -> Self;
    fn fused_pow(self, rhs: Self) -> Self;
    fn fused_expm1(self) -> Self;
    fn fused_log1p(self) -> Self;
}

macro_rules! impl_real_fused_scalar {
    ($ty:ty) => {
        impl FusedScalar for $ty {
            #[inline(always)]
            fn fused_add(self, rhs: Self) -> Self {
                self + rhs
            }

            #[inline(always)]
            fn fused_multiply(self, rhs: Self) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn fused_negate(self) -> Self {
                -self
            }

            #[inline(always)]
            fn fused_conj(self) -> Self {
                self
            }

            #[inline(always)]
            fn fused_divide(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn fused_abs(self) -> Self {
                self.abs()
            }

            #[inline(always)]
            fn fused_maximum(self, rhs: Self) -> Self {
                self.max(rhs)
            }

            #[inline(always)]
            fn fused_minimum(self, rhs: Self) -> Self {
                self.min(rhs)
            }

            #[inline(always)]
            fn fused_clamp(self, min: Self, max: Self) -> Self {
                self.clamp(min, max)
            }

            #[inline(always)]
            fn fused_exp(self) -> Self {
                self.exp()
            }

            #[inline(always)]
            fn fused_log(self) -> Self {
                self.ln()
            }

            #[inline(always)]
            fn fused_sin(self) -> Self {
                self.sin()
            }

            #[inline(always)]
            fn fused_cos(self) -> Self {
                self.cos()
            }

            #[inline(always)]
            fn fused_tanh(self) -> Self {
                self.tanh()
            }

            #[inline(always)]
            fn fused_sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn fused_rsqrt(self) -> Self {
                1.0 / self.sqrt()
            }

            #[inline(always)]
            fn fused_pow(self, rhs: Self) -> Self {
                self.powf(rhs)
            }

            #[inline(always)]
            fn fused_expm1(self) -> Self {
                self.exp_m1()
            }

            #[inline(always)]
            fn fused_log1p(self) -> Self {
                self.ln_1p()
            }
        }
    };
}

macro_rules! impl_complex_fused_scalar {
    ($ty:ty) => {
        impl FusedScalar for $ty {
            #[inline(always)]
            fn fused_add(self, rhs: Self) -> Self {
                self + rhs
            }

            #[inline(always)]
            fn fused_multiply(self, rhs: Self) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn fused_negate(self) -> Self {
                -self
            }

            #[inline(always)]
            fn fused_conj(self) -> Self {
                num_complex::Complex::conj(&self)
            }

            #[inline(always)]
            fn fused_divide(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn fused_abs(self) -> Self {
                Self::new(self.norm(), 0.0)
            }

            #[inline(always)]
            fn fused_maximum(self, rhs: Self) -> Self {
                if self.norm_sqr() >= rhs.norm_sqr() {
                    self
                } else {
                    rhs
                }
            }

            #[inline(always)]
            fn fused_minimum(self, rhs: Self) -> Self {
                if self.norm_sqr() <= rhs.norm_sqr() {
                    self
                } else {
                    rhs
                }
            }

            #[inline(always)]
            fn fused_clamp(self, min: Self, max: Self) -> Self {
                self.fused_maximum(min).fused_minimum(max)
            }

            #[inline(always)]
            fn fused_exp(self) -> Self {
                self.exp()
            }

            #[inline(always)]
            fn fused_log(self) -> Self {
                self.ln()
            }

            #[inline(always)]
            fn fused_sin(self) -> Self {
                self.sin()
            }

            #[inline(always)]
            fn fused_cos(self) -> Self {
                self.cos()
            }

            #[inline(always)]
            fn fused_tanh(self) -> Self {
                self.tanh()
            }

            #[inline(always)]
            fn fused_sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn fused_rsqrt(self) -> Self {
                Self::new(1.0, 0.0) / self.sqrt()
            }

            #[inline(always)]
            fn fused_pow(self, rhs: Self) -> Self {
                self.powc(rhs)
            }

            #[inline(always)]
            fn fused_expm1(self) -> Self {
                self.exp() - Self::new(1.0, 0.0)
            }

            #[inline(always)]
            fn fused_log1p(self) -> Self {
                (self + Self::new(1.0, 0.0)).ln()
            }
        }
    };
}

impl_real_fused_scalar!(f32);
impl_real_fused_scalar!(f64);
impl_complex_fused_scalar!(num_complex::Complex32);
impl_complex_fused_scalar!(num_complex::Complex64);

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

#[inline(always)]
fn eval_op<T: FusedScalar>(op: FusedOp, regs: &[T], inputs: &[usize]) -> T {
    match op {
        FusedOp::Add => regs[inputs[0]].fused_add(regs[inputs[1]]),
        FusedOp::Multiply => regs[inputs[0]].fused_multiply(regs[inputs[1]]),
        FusedOp::Negate => regs[inputs[0]].fused_negate(),
        FusedOp::Conj => regs[inputs[0]].fused_conj(),
        FusedOp::Divide => regs[inputs[0]].fused_divide(regs[inputs[1]]),
        FusedOp::Abs => regs[inputs[0]].fused_abs(),
        FusedOp::Maximum => regs[inputs[0]].fused_maximum(regs[inputs[1]]),
        FusedOp::Minimum => regs[inputs[0]].fused_minimum(regs[inputs[1]]),
        FusedOp::Clamp => regs[inputs[0]].fused_clamp(regs[inputs[1]], regs[inputs[2]]),
        FusedOp::Exp => regs[inputs[0]].fused_exp(),
        FusedOp::Log => regs[inputs[0]].fused_log(),
        FusedOp::Sin => regs[inputs[0]].fused_sin(),
        FusedOp::Cos => regs[inputs[0]].fused_cos(),
        FusedOp::Tanh => regs[inputs[0]].fused_tanh(),
        FusedOp::Sqrt => regs[inputs[0]].fused_sqrt(),
        FusedOp::Rsqrt => regs[inputs[0]].fused_rsqrt(),
        FusedOp::Pow => regs[inputs[0]].fused_pow(regs[inputs[1]]),
        FusedOp::Expm1 => regs[inputs[0]].fused_expm1(),
        FusedOp::Log1p => regs[inputs[0]].fused_log1p(),
    }
}

unsafe fn interpret_inner_loop<T: FusedScalar>(
    dst_ptrs: &[*mut T],
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
) {
    let output_count = dst_ptrs.len();
    let mut regs = Vec::with_capacity(plan.input_count + plan.ops.len());

    for i in 0..len {
        let i = i as isize;
        regs.clear();

        for (input_index, &input_ptr) in input_ptrs.iter().enumerate() {
            let stride_index = output_count + input_index;
            regs.push(*input_ptr.offset(offsets[stride_index] + i * strides[stride_index]));
        }

        for inst in &plan.ops {
            regs.push(eval_op(inst.op, &regs, &inst.inputs));
        }

        for (output_index, &dst_ptr) in dst_ptrs.iter().enumerate() {
            *dst_ptr.offset(offsets[output_index] + i * strides[output_index]) =
                regs[plan.outputs[output_index]];
        }
    }
}

fn interpret_fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    let dims = dests[0].dims().to_vec();
    if total_len(&dims) == 0 {
        return Ok(());
    }

    let dst_ptrs: Vec<*mut T> = dests.iter_mut().map(|dest| dest.as_mut_ptr()).collect();
    let input_ptrs: Vec<*const T> = inputs.iter().map(StridedView::ptr).collect();

    let mut strides_list: Vec<&[isize]> = Vec::with_capacity(dests.len() + inputs.len());
    for dest in dests.iter() {
        strides_list.push(dest.strides());
    }
    for input in inputs {
        strides_list.push(input.strides());
    }

    let elem_size = std::mem::size_of::<T>();
    let total = total_len(&dims);
    let (fused_dims, ordered_strides, kernel_plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(&dims, &strides_list)
    } else {
        build_plan_fused(&dims, &strides_list, Some(0), elem_size)
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send: Vec<SendPtr<T>> = dst_ptrs.iter().map(|&ptr| SendPtr(ptr)).collect();
            let input_send: Vec<SendPtr<T>> = input_ptrs
                .iter()
                .map(|&ptr| SendPtr(ptr as *mut T))
                .collect();

            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; ordered_strides.len()];
            let nthreads = rayon::current_num_threads();

            return mapreduce_threaded(
                &fused_dims,
                &kernel_plan.block,
                &ordered_strides,
                &initial_offsets,
                &costs,
                nthreads,
                0,
                1,
                &|dims, blocks, strides_list, offsets| {
                    let dst_ptrs: Vec<*mut T> = dst_send.iter().map(|ptr| ptr.as_ptr()).collect();
                    let input_ptrs: Vec<*const T> =
                        input_send.iter().map(|ptr| ptr.as_const()).collect();
                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        strides_list,
                        offsets,
                        |offsets, len, strides| {
                            unsafe {
                                interpret_inner_loop(
                                    &dst_ptrs,
                                    &input_ptrs,
                                    plan,
                                    offsets,
                                    len,
                                    strides,
                                );
                            }
                            Ok(())
                        },
                    )
                },
            );
        }
    }

    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &kernel_plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            unsafe {
                interpret_inner_loop(&dst_ptrs, &input_ptrs, plan, offsets, len, strides);
            }
            Ok(())
        },
    )
}

/// Evaluate a runtime-DAG elementwise plan into one or more destinations.
pub fn fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    validate_plan(plan, inputs.len(), dests.len())?;
    validate_shapes(dests, inputs)?;
    interpret_fused_elementwise_into(dests, inputs, plan)
}
