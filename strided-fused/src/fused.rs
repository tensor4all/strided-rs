//! Runtime-DAG fused elementwise kernels.

use core::mem::MaybeUninit;
use strided_basic::execution::is_injective_layout;

use crate::{MaybeSendSync, Result, StridedError, StridedView, StridedViewMut};
use strided_basic::execution::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    SMALL_TENSOR_THRESHOLD,
};
use strided_basic::execution::{
    map_into_validated, validate_destination_layout_without_alloc, zip_map2_into_validated,
    zip_map3_into_validated, zip_map4_into_validated, ValidatedDestinationLayout,
};

#[cfg(feature = "parallel")]
use strided_basic::execution::compute_costs;
#[cfg(feature = "parallel")]
use strided_basic::execution::{mapreduce_threaded, SendPtr, MINTHREADLENGTH};

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

#[cfg(test)]
std::thread_local! {
    static UNINITIALIZED_STATIC_FAMILY_HITS: core::cell::Cell<[usize; 7]> =
        const { core::cell::Cell::new([0; 7]) };
}

#[cfg(test)]
impl StaticFusedKind {
    fn test_index(self) -> usize {
        match self {
            Self::Unary(..) => 0,
            Self::Binary(..) => 1,
            Self::Ternary(..) => 2,
            Self::AddMulLeft => 3,
            Self::AddMulRight => 4,
            Self::MulAddExp => 5,
            Self::DivClampSqrtRsqrt => 6,
        }
    }
}

#[cfg(all(test, feature = "parallel"))]
fn reset_uninitialized_static_family_hits() {
    UNINITIALIZED_STATIC_FAMILY_HITS.set([0; 7]);
}

#[cfg(all(test, feature = "parallel"))]
fn uninitialized_static_family_hits() -> [usize; 7] {
    UNINITIALIZED_STATIC_FAMILY_HITS.get()
}

#[cfg(test)]
fn record_uninitialized_static_family_hit(kind: StaticFusedKind) {
    UNINITIALIZED_STATIC_FAMILY_HITS.set({
        let mut hits = UNINITIALIZED_STATIC_FAMILY_HITS.get();
        hits[kind.test_index()] += 1;
        hits
    });
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

    #[cfg(test)]
    const IS_UNINITIALIZED: bool;

    fn write(value: T) -> Self::Value;
}

struct InitializedStaticOutput;

impl<T: FusedScalar> StaticOutput<T> for InitializedStaticOutput {
    type Value = T;

    #[cfg(test)]
    const IS_UNINITIALIZED: bool = false;

    #[inline(always)]
    fn write(value: T) -> T {
        value
    }
}

struct UninitializedStaticOutput;

impl<T: FusedScalar> StaticOutput<T> for UninitializedStaticOutput {
    type Value = MaybeUninit<T>;

    #[cfg(test)]
    const IS_UNINITIALIZED: bool = true;

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
    let Some(kind) = classify_static_specialization(plan) else {
        return Ok(false);
    };
    #[cfg(test)]
    if O::IS_UNINITIALIZED {
        record_uninitialized_static_family_hit(kind);
    }

    match kind {
        StaticFusedKind::Unary(op, a) => {
            // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
            unsafe {
                map_into_validated(dest, &inputs[a], |x| O::write(eval_unary(op, x)), validated)
            }?
        }
        // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
        StaticFusedKind::Binary(op, a, b) => unsafe {
            zip_map2_into_validated(
                dest,
                &inputs[a],
                &inputs[b],
                |x, y| O::write(eval_binary(op, x, y)),
                validated,
            )
        }?,
        // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
        StaticFusedKind::Ternary(op, a, b, c) => unsafe {
            zip_map3_into_validated(
                dest,
                &inputs[a],
                &inputs[b],
                &inputs[c],
                |x, y, z| O::write(eval_ternary(op, x, y, z)),
                validated,
            )
        }?,
        // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
        StaticFusedKind::AddMulLeft => unsafe {
            zip_map2_into_validated(
                dest,
                &inputs[0],
                &inputs[1],
                |a, b| O::write(a.fused_add(b).fused_multiply(a)),
                validated,
            )
        }?,
        // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
        StaticFusedKind::AddMulRight => unsafe {
            zip_map2_into_validated(
                dest,
                &inputs[0],
                &inputs[1],
                |a, b| O::write(a.fused_multiply(a.fused_add(b))),
                validated,
            )
        }?,
        // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
        StaticFusedKind::MulAddExp => unsafe {
            zip_map3_into_validated(
                dest,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                |a, b, c| O::write(a.fused_multiply(b).fused_add(c).fused_exp()),
                validated,
            )
        }?,
        // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
        StaticFusedKind::DivClampSqrtRsqrt => unsafe {
            zip_map4_into_validated(
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
            )
        }?,
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
        if dests[0].len() == 0 {
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
        let total = dests[0].len();
        let (fused_dims, ordered_strides, kernel_plan) = if total <= SMALL_TENSOR_THRESHOLD {
            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
            unsafe { build_plan_fused_small(&dims, &strides_list) }
        } else {
            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
            unsafe { build_plan_fused(&dims, &strides_list, Some(0), elem_size) }
        };

        let total: usize = fused_dims.iter().product();
        let nthreads = strided_basic::execution::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            let dst_send: Vec<SendPtr<T>> = dst_ptrs.iter().map(|&ptr| SendPtr(ptr)).collect();
            let input_send: Vec<SendPtr<T>> = input_ptrs
                .iter()
                .map(|&ptr| SendPtr(ptr as *mut T))
                .collect();

            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.

            let costs = unsafe { compute_costs(&ordered_strides) };
            let initial_offsets = vec![0isize; ordered_strides.len()];
            let run_partition = |dims: &[usize],
                                 blocks: &[usize],
                                 strides_list: &[Vec<isize>],
                                 offsets: &[isize]|
             -> Result<()> {
                let dst_ptrs: Vec<*mut T> = dst_send.iter().map(|ptr| ptr.as_ptr()).collect();
                let input_ptrs: Vec<*const T> =
                    input_send.iter().map(|ptr| ptr.as_const()).collect();
                let run_block = |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
                    // SAFETY: validated shapes/layouts and the derived block bound every access.
                    unsafe {
                        interpret_inner_loop(&dst_ptrs, &input_ptrs, plan, offsets, len, strides);
                    }
                    Ok(())
                };
                // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
                unsafe {
                    for_each_inner_block_preordered(dims, blocks, strides_list, offsets, run_block)
                }
            };
            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
            return unsafe {
                mapreduce_threaded(
                    &fused_dims,
                    &kernel_plan.block,
                    &ordered_strides,
                    &initial_offsets,
                    &costs,
                    nthreads,
                    0,
                    1,
                    &run_partition,
                )
            };
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
    if dests[0].len() == 0 {
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
    let total = dests[0].len();
    let (fused_dims, ordered_strides, kernel_plan) = if total <= SMALL_TENSOR_THRESHOLD {
        // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
        unsafe { build_plan_fused_small(&dims, &strides_list) }
    } else {
        // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
        unsafe { build_plan_fused(&dims, &strides_list, Some(0), elem_size) }
    };

    let initial_offsets = vec![0isize; ordered_strides.len()];
    let run_block = |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
        // SAFETY: validated shapes/layouts and the derived block bound every access.
        unsafe {
            interpret_inner_loop(&dst_ptrs, &input_ptrs, plan, offsets, len, strides);
        }
        Ok(())
    };
    // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
    unsafe {
        for_each_inner_block_preordered(
            &fused_dims,
            &kernel_plan.block,
            &ordered_strides,
            &initial_offsets,
            run_block,
        )
    }
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
    if dest.len() == 0 {
        return Ok(());
    }
    let dst_ptr = dest.as_mut_ptr();
    let input_ptrs: Vec<*const T> = inputs.iter().map(StridedView::ptr).collect();
    let mut strides_list: Vec<&[isize]> = Vec::with_capacity(1 + inputs.len());
    strides_list.push(dest.strides());
    for input in inputs {
        strides_list.push(input.strides());
    }
    let total = dest.len();
    let (fused_dims, ordered_strides, kernel_plan) = if total <= SMALL_TENSOR_THRESHOLD {
        // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
        unsafe { build_plan_fused_small(dims, &strides_list) }
    } else {
        // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
        unsafe { build_plan_fused(dims, &strides_list, Some(0), core::mem::size_of::<T>()) }
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = strided_basic::execution::rayon_threads();
        if !serial && total > MINTHREADLENGTH && nthreads > 1 {
            let dst_send = SendPtr(dst_ptr);
            let input_send: Vec<SendPtr<T>> = input_ptrs
                .iter()
                .map(|&ptr| SendPtr(ptr as *mut T))
                .collect();
            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
            let costs = unsafe { compute_costs(&ordered_strides) };
            let initial_offsets = vec![0isize; ordered_strides.len()];
            let run_partition = |dims: &[usize],
                                 blocks: &[usize],
                                 strides_list: &[Vec<isize>],
                                 offsets: &[isize]|
             -> Result<()> {
                let input_ptrs: Vec<*const T> =
                    input_send.iter().map(|ptr| ptr.as_const()).collect();
                let run_block = |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
                    // SAFETY: validated shapes/layouts and the derived block bound every access.
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
                };
                // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
                unsafe {
                    for_each_inner_block_preordered(dims, blocks, strides_list, offsets, run_block)
                }
            };
            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
            return unsafe {
                mapreduce_threaded(
                    &fused_dims,
                    &kernel_plan.block,
                    &ordered_strides,
                    &initial_offsets,
                    &costs,
                    nthreads,
                    0,
                    1,
                    &run_partition,
                )
            };
        }
    }

    let initial_offsets = vec![0isize; ordered_strides.len()];
    let run_block = |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
        // SAFETY: validated shapes/layouts and the derived block bound every access.
        unsafe {
            interpret_inner_loop_uninit(dst_ptr, &input_ptrs, plan, offsets, len, strides);
        }
        Ok(())
    };
    // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
    unsafe {
        for_each_inner_block_preordered(
            &fused_dims,
            &kernel_plan.block,
            &ordered_strides,
            &initial_offsets,
            run_block,
        )
    }
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
#[path = "fused/tests/tests.rs"]
mod tests;
