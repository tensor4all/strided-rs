//! Runtime-DAG fused elementwise kernels.

use core::mem::MaybeUninit;

use crate::kernel::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    total_len, SMALL_TENSOR_THRESHOLD,
};
use crate::map_view::{
    map_into_validated, validate_destination_layout_without_alloc, zip_map2_into_validated,
    zip_map3_into_validated, zip_map4_into_validated, ValidatedDestinationLayout,
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

impl FusedOp {
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Multiply => "multiply",
            Self::Negate => "negate",
            Self::Conj => "conj",
            Self::Divide => "divide",
            Self::Abs => "abs",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::Clamp => "clamp",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Pow => "pow",
            Self::Expm1 => "expm1",
            Self::Log1p => "log1p",
        }
    }
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
    fn fused_dtype_label() -> &'static str {
        core::any::type_name::<Self>()
    }

    fn supports_fused_op(_op: FusedOp) -> bool {
        true
    }

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

macro_rules! unsupported_fused_op {
    ($op:literal, $ty:literal) => {
        unreachable!("unsupported fused op {} for dtype {}", $op, $ty)
    };
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

macro_rules! impl_signed_integer_fused_scalar {
    ($ty:ty, $label:literal) => {
        impl FusedScalar for $ty {
            #[inline]
            fn fused_dtype_label() -> &'static str {
                $label
            }

            #[inline]
            fn supports_fused_op(op: FusedOp) -> bool {
                matches!(
                    op,
                    FusedOp::Add
                        | FusedOp::Multiply
                        | FusedOp::Negate
                        | FusedOp::Conj
                        | FusedOp::Abs
                        | FusedOp::Maximum
                        | FusedOp::Minimum
                        | FusedOp::Clamp
                )
            }

            #[inline(always)]
            fn fused_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn fused_multiply(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn fused_negate(self) -> Self {
                self.wrapping_neg()
            }

            #[inline(always)]
            fn fused_conj(self) -> Self {
                self
            }

            #[inline(always)]
            fn fused_divide(self, _rhs: Self) -> Self {
                unsupported_fused_op!("divide", $label)
            }

            #[inline(always)]
            fn fused_abs(self) -> Self {
                self.wrapping_abs()
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
                unsupported_fused_op!("exp", $label)
            }

            #[inline(always)]
            fn fused_log(self) -> Self {
                unsupported_fused_op!("log", $label)
            }

            #[inline(always)]
            fn fused_sin(self) -> Self {
                unsupported_fused_op!("sin", $label)
            }

            #[inline(always)]
            fn fused_cos(self) -> Self {
                unsupported_fused_op!("cos", $label)
            }

            #[inline(always)]
            fn fused_tanh(self) -> Self {
                unsupported_fused_op!("tanh", $label)
            }

            #[inline(always)]
            fn fused_sqrt(self) -> Self {
                unsupported_fused_op!("sqrt", $label)
            }

            #[inline(always)]
            fn fused_rsqrt(self) -> Self {
                unsupported_fused_op!("rsqrt", $label)
            }

            #[inline(always)]
            fn fused_pow(self, _rhs: Self) -> Self {
                unsupported_fused_op!("pow", $label)
            }

            #[inline(always)]
            fn fused_expm1(self) -> Self {
                unsupported_fused_op!("expm1", $label)
            }

            #[inline(always)]
            fn fused_log1p(self) -> Self {
                unsupported_fused_op!("log1p", $label)
            }
        }
    };
}

impl_signed_integer_fused_scalar!(i32, "i32");
impl_signed_integer_fused_scalar!(i64, "i64");

impl FusedScalar for bool {
    #[inline]
    fn fused_dtype_label() -> &'static str {
        "bool"
    }

    #[inline]
    fn supports_fused_op(op: FusedOp) -> bool {
        matches!(op, FusedOp::Conj)
    }

    #[inline(always)]
    fn fused_add(self, _rhs: Self) -> Self {
        unsupported_fused_op!("add", "bool")
    }

    #[inline(always)]
    fn fused_multiply(self, _rhs: Self) -> Self {
        unsupported_fused_op!("multiply", "bool")
    }

    #[inline(always)]
    fn fused_negate(self) -> Self {
        unsupported_fused_op!("negate", "bool")
    }

    #[inline(always)]
    fn fused_conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn fused_divide(self, _rhs: Self) -> Self {
        unsupported_fused_op!("divide", "bool")
    }

    #[inline(always)]
    fn fused_abs(self) -> Self {
        unsupported_fused_op!("abs", "bool")
    }

    #[inline(always)]
    fn fused_maximum(self, _rhs: Self) -> Self {
        unsupported_fused_op!("maximum", "bool")
    }

    #[inline(always)]
    fn fused_minimum(self, _rhs: Self) -> Self {
        unsupported_fused_op!("minimum", "bool")
    }

    #[inline(always)]
    fn fused_clamp(self, _min: Self, _max: Self) -> Self {
        unsupported_fused_op!("clamp", "bool")
    }

    #[inline(always)]
    fn fused_exp(self) -> Self {
        unsupported_fused_op!("exp", "bool")
    }

    #[inline(always)]
    fn fused_log(self) -> Self {
        unsupported_fused_op!("log", "bool")
    }

    #[inline(always)]
    fn fused_sin(self) -> Self {
        unsupported_fused_op!("sin", "bool")
    }

    #[inline(always)]
    fn fused_cos(self) -> Self {
        unsupported_fused_op!("cos", "bool")
    }

    #[inline(always)]
    fn fused_tanh(self) -> Self {
        unsupported_fused_op!("tanh", "bool")
    }

    #[inline(always)]
    fn fused_sqrt(self) -> Self {
        unsupported_fused_op!("sqrt", "bool")
    }

    #[inline(always)]
    fn fused_rsqrt(self) -> Self {
        unsupported_fused_op!("rsqrt", "bool")
    }

    #[inline(always)]
    fn fused_pow(self, _rhs: Self) -> Self {
        unsupported_fused_op!("pow", "bool")
    }

    #[inline(always)]
    fn fused_expm1(self) -> Self {
        unsupported_fused_op!("expm1", "bool")
    }

    #[inline(always)]
    fn fused_log1p(self) -> Self {
        unsupported_fused_op!("log1p", "bool")
    }
}

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

pub(crate) fn validate_plan(
    plan: &FusedPlan,
    input_count: usize,
    output_count: usize,
) -> Result<()> {
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

pub(crate) fn validate_plan_for_scalar<T: FusedScalar>(
    plan: &FusedPlan,
    input_count: usize,
    output_count: usize,
) -> Result<()> {
    validate_plan(plan, input_count, output_count)?;
    for inst in &plan.ops {
        if !T::supports_fused_op(inst.op) {
            return Err(StridedError::UnsupportedOp {
                op: inst.op.label(),
                dtype: T::fused_dtype_label(),
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

pub(crate) fn is_injective_layout(dims: &[usize], strides: &[isize]) -> bool {
    let Some(total) = validate_injective_layout_inputs(dims, strides) else {
        return false;
    };
    if total <= 1 || has_disjoint_stride_spans(dims, strides) {
        return true;
    }

    const EXACT_CHECK_LIMIT: usize = 4096;
    if total <= EXACT_CHECK_LIMIT {
        return has_unique_offsets_exact(dims, strides, total);
    }

    false
}

pub(crate) fn is_injective_layout_without_alloc(dims: &[usize], strides: &[isize]) -> bool {
    let Some(total) = validate_injective_layout_inputs(dims, strides) else {
        return false;
    };
    if total <= 1 || has_disjoint_stride_spans(dims, strides) {
        return true;
    }

    const EXACT_CHECK_LIMIT: usize = 4096;
    total <= EXACT_CHECK_LIMIT && has_unique_offsets_pairwise(dims, strides, total)
}

fn offset_for_linear_index(dims: &[usize], strides: &[isize], mut linear: usize) -> Option<isize> {
    let mut offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        let index = linear % dim;
        linear /= dim;
        offset = offset.checked_add(stride.checked_mul(index as isize)?)?;
    }
    Some(offset)
}

fn has_unique_offsets_pairwise(dims: &[usize], strides: &[isize], total: usize) -> bool {
    for lhs in 0..total {
        let Some(lhs_offset) = offset_for_linear_index(dims, strides, lhs) else {
            return false;
        };
        for rhs in (lhs + 1)..total {
            if offset_for_linear_index(dims, strides, rhs) == Some(lhs_offset) {
                return false;
            }
        }
    }
    true
}

fn validate_injective_layout_inputs(dims: &[usize], strides: &[isize]) -> Option<usize> {
    if dims.len() != strides.len() {
        return None;
    }

    let total = dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))?;
    if total <= 1 {
        return Some(total);
    }
    if dims
        .iter()
        .zip(strides.iter())
        .any(|(&dim, &stride)| dim > 1 && stride == 0)
    {
        return None;
    }

    let mut min_offset = 0isize;
    let mut max_offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim <= 1 {
            continue;
        }
        let extent = isize::try_from(dim - 1).ok()?;
        let span = stride.checked_mul(extent)?;
        if span >= 0 {
            max_offset = max_offset.checked_add(span)?;
        } else {
            min_offset = min_offset.checked_add(span)?;
        }
    }
    Some(total)
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
    let mut covered_span = 0u128;
    let mut previous_axis = None;
    let active_axes = dims.iter().filter(|&&dim| dim > 1).count();
    for _ in 0..active_axes {
        let mut next = None;
        for (axis, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
            if dim <= 1 {
                continue;
            }
            let stride = match stride.checked_abs() {
                Some(stride) => stride as u128,
                None => return false,
            };
            let key = (stride, axis);
            if previous_axis.is_some_and(|previous| key <= previous) {
                continue;
            }
            if next.is_none_or(|(best, _)| key < best) {
                next = Some((key, dim as u128 - 1));
            }
        }
        let Some(((stride, axis), extent)) = next else {
            return false;
        };
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
        previous_axis = Some((stride, axis));
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

#[derive(Clone, Copy)]
enum StaticFusedKind {
    Unary(FusedOp, usize),
    Binary(FusedOp, usize, usize),
    Ternary(FusedOp, usize, usize, usize),
    AddMulLeft,
    AddMulRight,
    MulAddExp,
    DivClampSqrtRsqrt,
}

fn classify_static_specialization(plan: &FusedPlan) -> Option<StaticFusedKind> {
    if plan.outputs.len() != 1 {
        return None;
    }
    if let [inst] = plan.ops.as_slice() {
        if plan.outputs[0] != plan.input_count {
            return None;
        }
        return match (op_arity(inst.op), inst.inputs.as_slice()) {
            (1, [a]) => Some(StaticFusedKind::Unary(inst.op, *a)),
            (2, [a, b]) => Some(StaticFusedKind::Binary(inst.op, *a, *b)),
            (3, [a, b, c]) => Some(StaticFusedKind::Ternary(inst.op, *a, *b, *c)),
            _ => None,
        };
    }
    if plan.input_count == 2
        && plan.outputs.as_slice() == [3]
        && plan.ops.len() == 2
        && plan.ops[0].op == FusedOp::Add
        && plan.ops[0].inputs.as_slice() == [0, 1]
        && plan.ops[1].op == FusedOp::Multiply
    {
        return match plan.ops[1].inputs.as_slice() {
            [2, 0] => Some(StaticFusedKind::AddMulLeft),
            [0, 2] => Some(StaticFusedKind::AddMulRight),
            _ => None,
        };
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
        return Some(StaticFusedKind::MulAddExp);
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
        return Some(StaticFusedKind::DivClampSqrtRsqrt);
    }
    None
}

trait StaticOutput<T: FusedScalar> {
    type Value: Copy + MaybeSendSync;

    fn write(value: T) -> Self::Value;
}

struct InitializedStaticOutput;

impl<T: FusedScalar> StaticOutput<T> for InitializedStaticOutput {
    type Value = T;

    #[inline(always)]
    fn write(value: T) -> T {
        value
    }
}

struct UninitializedStaticOutput;

impl<T: FusedScalar> StaticOutput<T> for UninitializedStaticOutput {
    type Value = MaybeUninit<T>;

    #[inline(always)]
    fn write(value: T) -> MaybeUninit<T> {
        MaybeUninit::new(value)
    }
}

fn try_static_specialization_validated<T, O>(
    dest: &mut StridedViewMut<'_, O::Value>,
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    validated: ValidatedDestinationLayout,
) -> Result<bool>
where
    T: FusedScalar,
    O: StaticOutput<T>,
{
    match classify_static_specialization(plan) {
        Some(StaticFusedKind::Unary(op, a)) => {
            map_into_validated(dest, &inputs[a], |x| O::write(eval_unary(op, x)), validated)?
        }
        Some(StaticFusedKind::Binary(op, a, b)) => zip_map2_into_validated(
            dest,
            &inputs[a],
            &inputs[b],
            |x, y| O::write(eval_binary(op, x, y)),
            validated,
        )?,
        Some(StaticFusedKind::Ternary(op, a, b, c)) => zip_map3_into_validated(
            dest,
            &inputs[a],
            &inputs[b],
            &inputs[c],
            |x, y, z| O::write(eval_ternary(op, x, y, z)),
            validated,
        )?,
        Some(StaticFusedKind::AddMulLeft) => zip_map2_into_validated(
            dest,
            &inputs[0],
            &inputs[1],
            |a, b| O::write(a.fused_add(b).fused_multiply(a)),
            validated,
        )?,
        Some(StaticFusedKind::AddMulRight) => zip_map2_into_validated(
            dest,
            &inputs[0],
            &inputs[1],
            |a, b| O::write(a.fused_multiply(a.fused_add(b))),
            validated,
        )?,
        Some(StaticFusedKind::MulAddExp) => zip_map3_into_validated(
            dest,
            &inputs[0],
            &inputs[1],
            &inputs[2],
            |a, b, c| O::write(a.fused_multiply(b).fused_add(c).fused_exp()),
            validated,
        )?,
        Some(StaticFusedKind::DivClampSqrtRsqrt) => zip_map4_into_validated(
            dest,
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            |a, b, lo, hi| {
                O::write(
                    a.fused_divide(b)
                        .fused_maximum(lo)
                        .fused_minimum(hi)
                        .fused_sqrt()
                        .fused_rsqrt(),
                )
            },
            validated,
        )?,
        None => return Ok(false),
    }
    Ok(true)
}

fn try_static_specialization<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<bool> {
    if dests.len() != 1 {
        return Ok(false);
    }
    let validated = validate_destination_layout_without_alloc(dests[0].dims(), dests[0].strides())?;
    try_static_specialization_validated::<T, InitializedStaticOutput>(
        &mut dests[0],
        inputs,
        plan,
        validated,
    )
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
    #[cfg(feature = "parallel")]
    {
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

        let total: usize = fused_dims.iter().product();
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            let dst_send: Vec<SendPtr<T>> = dst_ptrs.iter().map(|&ptr| SendPtr(ptr)).collect();
            let input_send: Vec<SendPtr<T>> = input_ptrs
                .iter()
                .map(|&ptr| SendPtr(ptr as *mut T))
                .collect();

            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; ordered_strides.len()];
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

    interpret_fused_elementwise_into_serial(dests, inputs, plan)
}

fn interpret_fused_elementwise_into_serial<T: FusedScalar>(
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

pub(crate) fn fused_elementwise_into_serial<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    validate_plan_for_scalar::<T>(plan, inputs.len(), dests.len())?;
    validate_shapes(dests, inputs)?;
    interpret_fused_elementwise_into_serial(dests, inputs, plan)
}

unsafe fn interpret_inner_loop_uninit<T: FusedScalar>(
    dst_ptr: *mut MaybeUninit<T>,
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
) {
    let mut regs = Vec::with_capacity(plan.input_count + plan.ops.len());
    for i in 0..len {
        let i = i as isize;
        regs.clear();
        for (input_index, &input_ptr) in input_ptrs.iter().enumerate() {
            let stride_index = 1 + input_index;
            regs.push(*input_ptr.offset(offsets[stride_index] + i * strides[stride_index]));
        }
        for inst in &plan.ops {
            regs.push(eval_op(inst.op, &regs, &inst.inputs));
        }
        *dst_ptr.offset(offsets[0] + i * strides[0]) = MaybeUninit::new(regs[plan.outputs[0]]);
    }
}

pub(crate) fn fused_elementwise_into_uninit<T: FusedScalar>(
    dest: &mut StridedViewMut<'_, MaybeUninit<T>>,
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    serial: bool,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    #[cfg(not(feature = "parallel"))]
    let _ = serial;
    validate_plan_for_scalar::<T>(plan, inputs.len(), 1)?;
    for input in inputs {
        ensure_same_shape(dest.dims(), input.dims())?;
    }

    if !serial
        && try_static_specialization_validated::<T, UninitializedStaticOutput>(
            dest, inputs, plan, validated,
        )?
    {
        return Ok(());
    }

    let dims = dest.dims();
    if total_len(dims) == 0 {
        return Ok(());
    }
    let dst_ptr = dest.as_mut_ptr();
    let input_ptrs: Vec<*const T> = inputs.iter().map(StridedView::ptr).collect();
    let mut strides_list: Vec<&[isize]> = Vec::with_capacity(1 + inputs.len());
    strides_list.push(dest.strides());
    for input in inputs {
        strides_list.push(input.strides());
    }
    let total = total_len(dims);
    let (fused_dims, ordered_strides, kernel_plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dims, &strides_list)
    } else {
        build_plan_fused(dims, &strides_list, Some(0), core::mem::size_of::<T>())
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = crate::execution_policy::rayon_threads();
        if !serial && total > MINTHREADLENGTH && nthreads > 1 {
            let dst_send = SendPtr(dst_ptr);
            let input_send: Vec<SendPtr<T>> = input_ptrs
                .iter()
                .map(|&ptr| SendPtr(ptr as *mut T))
                .collect();
            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; ordered_strides.len()];
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
                    let input_ptrs: Vec<*const T> =
                        input_send.iter().map(|ptr| ptr.as_const()).collect();
                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        strides_list,
                        offsets,
                        |offsets, len, strides| {
                            unsafe {
                                interpret_inner_loop_uninit(
                                    dst_send.as_ptr(),
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
                interpret_inner_loop_uninit(dst_ptr, &input_ptrs, plan, offsets, len, strides);
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
/// `Maximum`, `Minimum`, and `Clamp` compare by squared norm. Signed integer
/// `Add`, `Multiply`, `Negate`, and `Abs` use wrapping arithmetic. `bool`
/// supports only copy-like identity plans and `Conj`; ambiguous arithmetic and
/// transcendental op/dtype pairs are rejected before any destination is written.
pub fn fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    validate_plan_for_scalar::<T>(plan, inputs.len(), dests.len())?;
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
    #[cfg(feature = "parallel")]
    use std::sync::Mutex;

    #[cfg(feature = "parallel")]
    static UNINIT_WORKER_IDS: Mutex<Vec<std::thread::ThreadId>> = Mutex::new(Vec::new());

    #[cfg(feature = "parallel")]
    #[derive(Clone, Copy)]
    struct ThreadTracked(u64);

    #[cfg(feature = "parallel")]
    impl ThreadTracked {
        fn observed(value: u64) -> Self {
            let id = std::thread::current().id();
            let mut ids = UNINIT_WORKER_IDS.lock().unwrap();
            if !ids.contains(&id) {
                ids.push(id);
            }
            drop(ids);
            for _ in 0..32 {
                std::hint::spin_loop();
            }
            Self(value)
        }
    }

    #[cfg(feature = "parallel")]
    impl FusedScalar for ThreadTracked {
        fn fused_add(self, rhs: Self) -> Self {
            Self::observed(self.0 + rhs.0)
        }

        fn fused_multiply(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }

        fn fused_negate(self) -> Self {
            self
        }

        fn fused_conj(self) -> Self {
            self
        }

        fn fused_divide(self, _rhs: Self) -> Self {
            self
        }

        fn fused_abs(self) -> Self {
            self
        }

        fn fused_maximum(self, _rhs: Self) -> Self {
            self
        }

        fn fused_minimum(self, _rhs: Self) -> Self {
            self
        }

        fn fused_clamp(self, _min: Self, _max: Self) -> Self {
            self
        }

        fn fused_exp(self) -> Self {
            self
        }

        fn fused_log(self) -> Self {
            self
        }

        fn fused_sin(self) -> Self {
            self
        }

        fn fused_cos(self) -> Self {
            self
        }

        fn fused_tanh(self) -> Self {
            self
        }

        fn fused_sqrt(self) -> Self {
            self
        }

        fn fused_rsqrt(self) -> Self {
            self
        }

        fn fused_pow(self, _rhs: Self) -> Self {
            self
        }

        fn fused_expm1(self) -> Self {
            self
        }

        fn fused_log1p(self) -> Self {
            self
        }
    }

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

    #[cfg(feature = "parallel")]
    #[test]
    fn uninitialized_fused_replay_respects_serial_and_bounded_contexts_above_threshold() {
        use crate::{with_execution_policy, ExecContext, ExecutionPolicy};
        use std::num::NonZeroUsize;

        let len = MINTHREADLENGTH + 65;
        let lhs = vec![ThreadTracked(1); len];
        let rhs = vec![ThreadTracked(2); len];
        let lhs = StridedView::new(&lhs, &[len], &[1], 0).unwrap();
        let rhs = StridedView::new(&rhs, &[len], &[1], 0).unwrap();
        let inputs = [lhs, rhs];
        let plan = FusedPlan {
            input_count: 2,
            outputs: vec![2],
            ops: vec![FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            }],
        };
        let four = NonZeroUsize::new(4).unwrap();

        let caller = std::thread::current().id();
        let mut output = vec![MaybeUninit::uninit(); len];
        let mut dest = StridedViewMut::new(&mut output, &[len], &[1], 0).unwrap();
        let validated =
            validate_destination_layout_without_alloc(dest.dims(), dest.strides()).unwrap();
        UNINIT_WORKER_IDS.lock().unwrap().clear();
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
            fused_elementwise_into_uninit(&mut dest, &inputs, &plan, true, validated).unwrap();
        });
        assert_eq!(*UNINIT_WORKER_IDS.lock().unwrap(), vec![caller]);

        let mut output = vec![MaybeUninit::uninit(); len];
        let mut dest = StridedViewMut::new(&mut output, &[len], &[1], 0).unwrap();
        let validated =
            validate_destination_layout_without_alloc(dest.dims(), dest.strides()).unwrap();
        UNINIT_WORKER_IDS.lock().unwrap().clear();
        let ctx = ExecContext::max_threads(2).unwrap();
        ctx.run(|| {
            fused_elementwise_into_uninit(&mut dest, &inputs, &plan, false, validated).unwrap();
        });
        let workers = UNINIT_WORKER_IDS.lock().unwrap();
        assert!(
            workers.len() > 1,
            "bounded replay must cross the parallel threshold"
        );
        assert!(workers.len() <= 2, "bounded replay exceeded max_threads(2)");
    }

    // Single-instruction plan whose sole output is the instruction result. Such
    // plans always hit `try_static_specialization`, and both the static and the
    // interpreter path dispatch through the scalar `FusedScalar` methods, so
    // iterating every op exercises each scalar implementation.
    fn single_op(input_count: usize, op: FusedOp, inputs: Vec<usize>) -> FusedPlan {
        FusedPlan {
            input_count,
            outputs: vec![input_count],
            ops: vec![FusedInst { op, inputs }],
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

    // Real negate/conj/abs were the only real scalar ops not reached by the
    // chains above; cover them so the real `FusedScalar` impl is fully exercised.
    #[test]
    fn specializes_real_negate_conj_abs() {
        for op in [FusedOp::Negate, FusedOp::Conj, FusedOp::Abs] {
            let x = input(&[-1.5, 2.0, -3.5]);
            assert_static_matches_interpreter(single_op(1, op, vec![0]), &[x]);
        }
    }

    // The complex `FusedScalar` impl had no coverage at all (every existing test
    // used f64). Run every op over Complex64 so both the static and interpreter
    // paths dispatch through the complex scalar methods.
    #[test]
    fn specializes_every_op_over_complex() {
        use num_complex::Complex64;

        let c = |re: f64, im: f64| Complex64::new(re, im);
        let cinput = |values: &[Complex64]| {
            StridedArray::from_parts(values.to_vec(), &[values.len()], &[1], 0).unwrap()
        };
        let assert_complex_match = |plan: FusedPlan, arrays: &[StridedArray<Complex64>]| {
            let inputs: Vec<_> = arrays.iter().map(|array| array.view()).collect();
            let mut static_out = StridedArray::<Complex64>::col_major(arrays[0].dims());
            let used_static = {
                let mut dests = [static_out.view_mut()];
                try_static_specialization(&mut dests, &inputs, &plan).unwrap()
            };
            let mut interp_out = StridedArray::<Complex64>::col_major(arrays[0].dims());
            {
                let mut dests = [interp_out.view_mut()];
                interpret_fused_elementwise_into(&mut dests, &inputs, &plan).unwrap();
            }
            assert!(used_static, "single-op plan should specialize");
            for (actual, expected) in static_out.iter().zip(interp_out.iter()) {
                assert!((actual - expected).norm() < 1e-9, "{actual} vs {expected}");
            }
        };

        // Positive-real-part, nonzero operands keep div/log/sqrt/pow well defined.
        let a = cinput(&[c(1.5, 0.5), c(2.0, -1.0), c(0.7, 0.3)]);
        let b = cinput(&[c(1.1, 0.2), c(0.9, 0.4), c(1.3, -0.6)]);
        let d = cinput(&[c(2.0, 0.0), c(2.0, 0.0), c(2.0, 0.0)]);

        for op in [
            FusedOp::Negate,
            FusedOp::Conj,
            FusedOp::Abs,
            FusedOp::Exp,
            FusedOp::Log,
            FusedOp::Sin,
            FusedOp::Cos,
            FusedOp::Tanh,
            FusedOp::Sqrt,
            FusedOp::Rsqrt,
            FusedOp::Expm1,
            FusedOp::Log1p,
        ] {
            assert_complex_match(single_op(1, op, vec![0]), std::slice::from_ref(&a));
        }
        for op in [
            FusedOp::Add,
            FusedOp::Multiply,
            FusedOp::Divide,
            FusedOp::Maximum,
            FusedOp::Minimum,
            FusedOp::Pow,
        ] {
            assert_complex_match(single_op(2, op, vec![0, 1]), &[a.clone(), b.clone()]);
        }
        assert_complex_match(
            single_op(3, FusedOp::Clamp, vec![0, 1, 2]),
            &[a.clone(), b.clone(), d.clone()],
        );
    }

    // Error branches in plan/layout validation that the positive tests skip.
    #[test]
    fn validate_plan_rejects_out_of_range_output() {
        // output id refers to a value that no instruction produces.
        let plan = FusedPlan {
            input_count: 1,
            outputs: vec![5],
            ops: vec![FusedInst {
                op: FusedOp::Exp,
                inputs: vec![0],
            }],
        };
        assert!(validate_plan(&plan, 1, 1).is_err());
    }

    #[test]
    fn validate_plan_rejects_zero_outputs() {
        let plan = FusedPlan {
            input_count: 1,
            outputs: vec![],
            ops: vec![],
        };
        assert!(validate_plan(&plan, 1, 0).is_err());
    }

    #[test]
    fn is_injective_layout_rejects_rank_and_broadcast_mismatch() {
        assert!(!is_injective_layout(&[2, 3], &[1]));
        assert!(!is_injective_layout(&[2, 2], &[0, 1]));
        assert!(is_injective_layout(&[1], &[0]));
    }

    #[test]
    fn is_injective_layout_rejects_unrepresentable_offset_spans() {
        let positive = isize::MAX / 2 + 1;
        let negative = isize::MIN / 2 - 1;
        assert!(!is_injective_layout(&[2, 2], &[positive, isize::MAX]));
        assert!(!is_injective_layout(&[2, 2], &[negative, isize::MIN]));
    }
}
