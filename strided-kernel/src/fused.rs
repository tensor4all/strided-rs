//! Runtime-DAG fused elementwise kernels.

use crate::kernel::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    total_len, SMALL_TENSOR_THRESHOLD,
};
use crate::map_view::{map_into, zip_map2_into, zip_map3_into, zip_map4_into};
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

/// Topologically ordered fused elementwise SSA DAG.
///
/// Values are numbered in evaluation order. Input values occupy
/// `0..input_count`; each instruction appends one value after the previous
/// inputs/instructions. For example, with `input_count == 2`, the first
/// instruction writes value `2`, the second writes value `3`, and so on.
/// `outputs` contains the value ids to write to `dests` in order.
///
/// All inputs and destinations passed to [`fused_elementwise_into`] must have
/// the same shape and scalar type. Broadcast inputs should be represented with
/// `StridedView::broadcast` before building the plan; the fused API does not
/// perform implicit broadcasting.
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
    for dest in dests {
        validate_destination_layout(dest)?;
    }
    for dest in &dests[1..] {
        ensure_same_shape(dims, dest.dims())?;
    }
    for input in inputs {
        ensure_same_shape(dims, input.dims())?;
    }
    Ok(())
}

fn validate_destination_layout<T>(dest: &StridedViewMut<'_, T>) -> Result<()> {
    if is_injective_layout(dest.dims(), dest.strides()) {
        Ok(())
    } else {
        Err(StridedError::NonInjectiveOutputLayout)
    }
}

fn is_injective_layout(dims: &[usize], strides: &[isize]) -> bool {
    if dims.len() != strides.len() {
        return false;
    }

    let Some(total) = dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    else {
        return false;
    };
    if total <= 1 {
        return true;
    }
    if dims
        .iter()
        .zip(strides.iter())
        .any(|(&dim, &stride)| dim > 1 && stride == 0)
    {
        return false;
    }

    const EXACT_CHECK_LIMIT: usize = 4096;
    if total <= EXACT_CHECK_LIMIT {
        return has_unique_offsets_exact(dims, strides, total);
    }

    has_disjoint_stride_spans(dims, strides)
}

fn has_unique_offsets_exact(dims: &[usize], strides: &[isize], total: usize) -> bool {
    let mut seen = std::collections::HashSet::with_capacity(total);
    let mut indices = vec![0usize; dims.len()];
    let mut offset = 0isize;

    for _ in 0..total {
        if !seen.insert(offset) {
            return false;
        }

        for axis in 0..dims.len() {
            indices[axis] += 1;
            offset = match offset.checked_add(strides[axis]) {
                Some(offset) => offset,
                None => return false,
            };
            if indices[axis] < dims[axis] {
                break;
            }

            let rewind = match strides[axis].checked_mul(indices[axis] as isize) {
                Some(rewind) => rewind,
                None => return false,
            };
            offset = match offset.checked_sub(rewind) {
                Some(offset) => offset,
                None => return false,
            };
            indices[axis] = 0;
        }
    }

    true
}

fn has_disjoint_stride_spans(dims: &[usize], strides: &[isize]) -> bool {
    let mut axes = Vec::with_capacity(dims.len());
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim <= 1 {
            continue;
        }
        let stride_abs = match stride.checked_abs() {
            Some(stride_abs) => stride_abs as u128,
            None => return false,
        };
        axes.push((stride_abs, dim as u128 - 1));
    }
    axes.sort_unstable_by_key(|&(stride, _)| stride);

    let mut covered_span = 0u128;
    for (stride, extent) in axes {
        if stride <= covered_span {
            return false;
        }
        covered_span = match stride
            .checked_mul(extent)
            .and_then(|span| covered_span.checked_add(span))
        {
            Some(covered_span) => covered_span,
            None => return false,
        };
    }

    true
}

#[inline(always)]
fn eval_op<T: FusedScalar>(op: FusedOp, regs: &[T], inputs: &[usize]) -> T {
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
        | FusedOp::Log1p => eval_unary(op, regs[inputs[0]]),
        FusedOp::Add
        | FusedOp::Multiply
        | FusedOp::Divide
        | FusedOp::Maximum
        | FusedOp::Minimum
        | FusedOp::Pow => eval_binary(op, regs[inputs[0]], regs[inputs[1]]),
        FusedOp::Clamp => eval_ternary(op, regs[inputs[0]], regs[inputs[1]], regs[inputs[2]]),
    }
}

#[inline(always)]
fn eval_unary<T: FusedScalar>(op: FusedOp, x: T) -> T {
    match op {
        FusedOp::Negate => x.fused_negate(),
        FusedOp::Conj => x.fused_conj(),
        FusedOp::Abs => x.fused_abs(),
        FusedOp::Exp => x.fused_exp(),
        FusedOp::Log => x.fused_log(),
        FusedOp::Sin => x.fused_sin(),
        FusedOp::Cos => x.fused_cos(),
        FusedOp::Tanh => x.fused_tanh(),
        FusedOp::Sqrt => x.fused_sqrt(),
        FusedOp::Rsqrt => x.fused_rsqrt(),
        FusedOp::Expm1 => x.fused_expm1(),
        FusedOp::Log1p => x.fused_log1p(),
        _ => unreachable!("not a unary fused op: {op:?}"),
    }
}

#[inline(always)]
fn eval_binary<T: FusedScalar>(op: FusedOp, a: T, b: T) -> T {
    match op {
        FusedOp::Add => a.fused_add(b),
        FusedOp::Multiply => a.fused_multiply(b),
        FusedOp::Divide => a.fused_divide(b),
        FusedOp::Maximum => a.fused_maximum(b),
        FusedOp::Minimum => a.fused_minimum(b),
        FusedOp::Pow => a.fused_pow(b),
        _ => unreachable!("not a binary fused op: {op:?}"),
    }
}

#[inline(always)]
fn eval_ternary<T: FusedScalar>(op: FusedOp, a: T, b: T, c: T) -> T {
    match op {
        FusedOp::Clamp => a.fused_clamp(b, c),
        _ => unreachable!("not a ternary fused op: {op:?}"),
    }
}

fn try_static_specialization<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<bool> {
    if dests.len() != 1 || plan.outputs.len() != 1 {
        return Ok(false);
    }

    if let [inst] = plan.ops.as_slice() {
        let output_id = plan.input_count;
        if plan.outputs[0] != output_id {
            return Ok(false);
        }

        match op_arity(inst.op) {
            1 => {
                map_into(&mut dests[0], &inputs[inst.inputs[0]], |x| {
                    eval_unary(inst.op, x)
                })?;
                return Ok(true);
            }
            2 => {
                zip_map2_into(
                    &mut dests[0],
                    &inputs[inst.inputs[0]],
                    &inputs[inst.inputs[1]],
                    |a, b| eval_binary(inst.op, a, b),
                )?;
                return Ok(true);
            }
            3 => {
                zip_map3_into(
                    &mut dests[0],
                    &inputs[inst.inputs[0]],
                    &inputs[inst.inputs[1]],
                    &inputs[inst.inputs[2]],
                    |a, b, c| eval_ternary(inst.op, a, b, c),
                )?;
                return Ok(true);
            }
            _ => unreachable!("unsupported fused op arity"),
        }
    }

    if plan.input_count == 2
        && plan.outputs.as_slice() == [3]
        && plan.ops.len() == 2
        && plan.ops[0].op == FusedOp::Add
        && plan.ops[0].inputs.as_slice() == [0, 1]
        && plan.ops[1].op == FusedOp::Multiply
    {
        match plan.ops[1].inputs.as_slice() {
            [2, 0] => {
                zip_map2_into(&mut dests[0], &inputs[0], &inputs[1], |a, b| {
                    a.fused_add(b).fused_multiply(a)
                })?;
                return Ok(true);
            }
            [0, 2] => {
                zip_map2_into(&mut dests[0], &inputs[0], &inputs[1], |a, b| {
                    a.fused_multiply(a.fused_add(b))
                })?;
                return Ok(true);
            }
            _ => {}
        }
    }

    if plan.input_count == 3
        && plan.outputs.as_slice() == [5]
        && plan.ops.len() == 3
        && plan.ops[0].op == FusedOp::Multiply
        && plan.ops[0].inputs.as_slice() == [0, 1]
        && plan.ops[1].op == FusedOp::Add
        && plan.ops[1].inputs.as_slice() == [3, 2]
        && plan.ops[2].op == FusedOp::Exp
        && plan.ops[2].inputs.as_slice() == [4]
    {
        zip_map3_into(
            &mut dests[0],
            &inputs[0],
            &inputs[1],
            &inputs[2],
            |a, b, c| a.fused_multiply(b).fused_add(c).fused_exp(),
        )?;
        return Ok(true);
    }

    if plan.input_count == 4
        && plan.outputs.as_slice() == [8]
        && plan.ops.len() == 5
        && plan.ops[0].op == FusedOp::Divide
        && plan.ops[0].inputs.as_slice() == [0, 1]
        && plan.ops[1].op == FusedOp::Maximum
        && plan.ops[1].inputs.as_slice() == [4, 2]
        && plan.ops[2].op == FusedOp::Minimum
        && plan.ops[2].inputs.as_slice() == [5, 3]
        && plan.ops[3].op == FusedOp::Sqrt
        && plan.ops[3].inputs.as_slice() == [6]
        && plan.ops[4].op == FusedOp::Rsqrt
        && plan.ops[4].inputs.as_slice() == [7]
    {
        zip_map4_into(
            &mut dests[0],
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            |a, b, lo, hi| {
                a.fused_divide(b)
                    .fused_maximum(lo)
                    .fused_minimum(hi)
                    .fused_sqrt()
                    .fused_rsqrt()
            },
        )?;
        return Ok(true);
    }

    Ok(false)
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
///
/// The plan is validated before any destination is written:
///
/// - `inputs.len()` must equal `plan.input_count`;
/// - `dests.len()` must equal `plan.outputs.len()`;
/// - instruction operands must reference earlier SSA values with the right
///   arity for their [`FusedOp`];
/// - every input and destination must have exactly the destination shape;
/// - each mutable destination layout must be injective, so two logical output
///   elements never map to the same memory address.
///
/// The implementation dispatches known single-output plans to existing static
/// `map_into`/`zip_map*_into` kernels and uses a generic interpreter fallback
/// for arbitrary validated DAGs. Overlapping source/destination memory is not
/// supported by the strided kernels generally.
///
/// Real `Maximum`, `Minimum`, and `Clamp` use Rust `f32`/`f64` `max`/`min`
/// semantics. Complex `Abs` returns the norm in the real component; complex
/// `Maximum`, `Minimum`, and `Clamp` compare by squared norm.
pub fn fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    validate_plan(plan, inputs.len(), dests.len())?;
    validate_shapes(dests, inputs)?;
    if try_static_specialization(dests, inputs, plan)? {
        return Ok(());
    }
    interpret_fused_elementwise_into(dests, inputs, plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StridedArray;

    fn input(values: &[f64]) -> StridedArray<f64> {
        StridedArray::from_parts(values.to_vec(), &[values.len()], &[1], 0).unwrap()
    }

    fn run_static(plan: &FusedPlan, arrays: &[StridedArray<f64>]) -> (bool, Vec<f64>) {
        let inputs: Vec<_> = arrays.iter().map(|array| array.view()).collect();
        let mut out = StridedArray::<f64>::col_major(arrays[0].dims());
        let used_static = {
            let mut dests = [out.view_mut()];
            try_static_specialization(&mut dests, &inputs, plan).unwrap()
        };
        (used_static, out.iter().copied().collect())
    }

    fn run_interpreter(plan: &FusedPlan, arrays: &[StridedArray<f64>]) -> Vec<f64> {
        let inputs: Vec<_> = arrays.iter().map(|array| array.view()).collect();
        let mut out = StridedArray::<f64>::col_major(arrays[0].dims());
        {
            let mut dests = [out.view_mut()];
            interpret_fused_elementwise_into(&mut dests, &inputs, plan).unwrap();
        }
        out.iter().copied().collect()
    }

    fn assert_static_matches_interpreter(plan: FusedPlan, arrays: &[StridedArray<f64>]) {
        let (used_static, static_values) = run_static(&plan, arrays);
        let interpreter_values = run_interpreter(&plan, arrays);

        assert!(used_static, "plan should use static specialization");
        assert_eq!(static_values.len(), interpreter_values.len());
        for (actual, expected) in static_values.iter().zip(interpreter_values.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn specializes_unary_exp() {
        let a = input(&[1.0, 2.0, 3.0]);
        let plan = FusedPlan {
            input_count: 1,
            outputs: vec![1],
            ops: vec![FusedInst {
                op: FusedOp::Exp,
                inputs: vec![0],
            }],
        };

        assert_static_matches_interpreter(plan, &[a]);
    }

    #[test]
    fn specializes_binary_add() {
        let a = input(&[1.0, 2.0, 3.0]);
        let b = input(&[10.0, 20.0, 30.0]);
        let plan = FusedPlan {
            input_count: 2,
            outputs: vec![2],
            ops: vec![FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            }],
        };

        assert_static_matches_interpreter(plan, &[a, b]);
    }

    #[test]
    fn specializes_ternary_clamp() {
        let x = input(&[1.0, 2.0, 3.0]);
        let lo = input(&[1.5, 1.5, 1.5]);
        let hi = input(&[2.5, 2.5, 2.5]);
        let plan = FusedPlan {
            input_count: 3,
            outputs: vec![3],
            ops: vec![FusedInst {
                op: FusedOp::Clamp,
                inputs: vec![0, 1, 2],
            }],
        };

        assert_static_matches_interpreter(plan, &[x, lo, hi]);
    }

    #[test]
    fn specializes_add_then_multiply_reusing_input() {
        let a = input(&[1.0, 2.0, 3.0]);
        let b = input(&[10.0, 20.0, 30.0]);
        let plan = FusedPlan {
            input_count: 2,
            outputs: vec![3],
            ops: vec![
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![0, 1],
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![2, 0],
                },
            ],
        };

        assert_static_matches_interpreter(plan, &[a, b]);
    }

    #[test]
    fn specializes_exp_of_multiply_add_chain() {
        let a = input(&[1.0, 2.0, 3.0]);
        let b = input(&[0.5, 1.5, 2.5]);
        let c = input(&[2.0, 2.0, 2.0]);
        let plan = FusedPlan {
            input_count: 3,
            outputs: vec![5],
            ops: vec![
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![0, 1],
                },
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![3, 2],
                },
                FusedInst {
                    op: FusedOp::Exp,
                    inputs: vec![4],
                },
            ],
        };

        assert_static_matches_interpreter(plan, &[a, b, c]);
    }

    #[test]
    fn specializes_divide_clamp_sqrt_rsqrt_chain() {
        let a = input(&[4.0, 9.0, 16.0]);
        let b = input(&[2.0, 3.0, 4.0]);
        let lo = input(&[1.5, 1.5, 1.5]);
        let hi = input(&[8.0, 8.0, 8.0]);
        let plan = FusedPlan {
            input_count: 4,
            outputs: vec![8],
            ops: vec![
                FusedInst {
                    op: FusedOp::Divide,
                    inputs: vec![0, 1],
                },
                FusedInst {
                    op: FusedOp::Maximum,
                    inputs: vec![4, 2],
                },
                FusedInst {
                    op: FusedOp::Minimum,
                    inputs: vec![5, 3],
                },
                FusedInst {
                    op: FusedOp::Sqrt,
                    inputs: vec![6],
                },
                FusedInst {
                    op: FusedOp::Rsqrt,
                    inputs: vec![7],
                },
            ],
        };

        assert_static_matches_interpreter(plan, &[a, b, lo, hi]);
    }
}
