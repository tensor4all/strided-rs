//! Runtime-DAG fused elementwise kernels.

use core::mem::MaybeUninit;
use strided_basic::execution::validate_destination_layout;
#[cfg(test)]
use strided_basic::execution::{is_injective_layout, validate_destination_layout_without_alloc};

use crate::{MaybeSendSync, Result, StridedError, StridedView, StridedViewMut};
use strided_basic::execution::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    SMALL_TENSOR_THRESHOLD,
};
use strided_basic::execution::{
    map_into_validated, zip_map2_into_validated, zip_map3_into_validated, zip_map4_into_validated,
    ValidatedDestinationLayout,
};
use strided_basic::{with_execution_policy, ExecutionPolicy};

mod chunk;

/// RAII owner of the initialized prefix of a stack buffer of destination
/// views.
///
/// Used by the erased multi-output entry points so their stack-buffer cast stays
/// out of the erased raw-boundary modules. Dropping the guard drops exactly the
/// initialized prefix, so a rank above `INLINE_RANK` does not leak every
/// destination's boxed metadata on success, error, or unwind.
pub(crate) struct InitializedViewsMut<'a, T> {
    storage: *mut MaybeUninit<StridedViewMut<'a, T>>,
    len: usize,
}

impl<'a, T> InitializedViewsMut<'a, T> {
    /// # Safety
    /// `storage` must point to at least `OPERAND_CAP` writable slots that stay
    /// valid until the guard is dropped.
    pub(crate) unsafe fn new(storage: *mut MaybeUninit<StridedViewMut<'a, T>>) -> Self {
        Self { storage, len: 0 }
    }

    /// Write the next initialized view into the prefix.
    pub(crate) fn push(&mut self, view: StridedViewMut<'a, T>) {
        debug_assert!(self.len < strided_basic::OPERAND_CAP);
        // SAFETY: the caller reserved `OPERAND_CAP` slots and
        // `len < OPERAND_CAP` holds.
        unsafe { self.storage.add(self.len).write(MaybeUninit::new(view)) };
        self.len += 1;
    }

    /// Borrow the initialized prefix.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [StridedViewMut<'a, T>] {
        // SAFETY: every slot below `len` was initialized by `push`, and
        // `MaybeUninit<StridedViewMut>` has the same layout as
        // `StridedViewMut`.
        unsafe {
            core::slice::from_raw_parts_mut(self.storage.cast::<StridedViewMut<'a, T>>(), self.len)
        }
    }
}

impl<T> Drop for InitializedViewsMut<'_, T> {
    fn drop(&mut self) {
        for index in 0..self.len {
            // SAFETY: every slot in the prefix was initialized by `push`.
            unsafe { (*self.storage.add(index)).assume_init_drop() };
        }
    }
}

/// A raw operand pointer that is `Default` and `Send + Sync`, so pointer lists
/// can live in an inline operand vector and be captured by worker closures.
///
/// `repr(transparent)` over `*mut T`, so a slice of them can be viewed as a
/// slice of raw pointers without copying.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub(super) struct OperandPtr<T>(*mut T);

impl<T> Default for OperandPtr<T> {
    fn default() -> Self {
        Self(core::ptr::null_mut())
    }
}

// SAFETY: only used to move/copy validated operand addresses to worker threads;
// the owning views establish liveness, aliasing and access at each use.
unsafe impl<T> Send for OperandPtr<T> {}
unsafe impl<T> Sync for OperandPtr<T> {}

impl<T> OperandPtr<T> {
    #[inline]
    fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }
}

/// View an inline operand vector as a `&[*mut T]` slice.
#[inline]
fn operand_ptrs<T>(ptrs: &[OperandPtr<T>]) -> &[*mut T] {
    // SAFETY: `OperandPtr<T>` is `repr(transparent)` over `*mut T`.
    unsafe { core::slice::from_raw_parts(ptrs.as_ptr().cast::<*mut T>(), ptrs.len()) }
}

/// View an inline operand vector as a `&[*const T]` slice.
#[inline]
fn operand_const_ptrs<T>(ptrs: &[OperandPtr<T>]) -> &[*const T] {
    // SAFETY: `OperandPtr<T>` is `repr(transparent)` over `*mut T`, and
    // `*mut T` and `*const T` have the same layout.
    unsafe { core::slice::from_raw_parts(ptrs.as_ptr().cast::<*const T>(), ptrs.len()) }
}

#[cfg(feature = "parallel")]
use strided_basic::execution::compute_costs;
#[cfg(feature = "parallel")]
use strided_basic::execution::{mapreduce_threaded, MINTHREADLENGTH};

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
///
/// `inputs` holds at most [`OPERAND_CAP`](strided_basic::OPERAND_CAP) value ids
/// inline and spills above that, so a bounded program builds without touching
/// the heap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusedInst {
    pub op: FusedOp,
    pub inputs: strided_basic::OperandVec<usize>,
}

impl FusedInst {
    /// Build an instruction from an operation and its input value ids.
    ///
    /// ```
    /// use strided_fused::{FusedInst, FusedOp};
    /// let inst = FusedInst::new(FusedOp::Add, [0, 1]);
    /// assert_eq!(inst.inputs.as_slice(), &[0, 1]);
    /// ```
    pub fn new(op: FusedOp, inputs: impl IntoIterator<Item = usize>) -> Self {
        Self {
            op,
            inputs: inputs.into_iter().collect(),
        }
    }
}

impl Default for FusedOp {
    fn default() -> Self {
        Self::Add
    }
}

impl Default for FusedInst {
    fn default() -> Self {
        Self {
            op: FusedOp::default(),
            inputs: strided_basic::OperandVec::new(),
        }
    }
}

/// Topologically ordered fused elementwise SSA DAG.
///
/// Values are numbered in evaluation order. Input values occupy
/// `0..input_count`; each instruction appends one value after the previous
/// inputs/instructions. For example, with `input_count == 2`, the first
/// instruction writes value `2`, the second writes value `3`, and so on.
/// `outputs` contains the value ids to write to `dests` in order.
///
/// The program is stored inline: at most eight instructions, five outputs and
/// [`OPERAND_CAP`](strided_basic::OPERAND_CAP) inputs per instruction, spilling
/// to the heap only above those bounds. Build one with [`FusedPlan::new`] to
/// keep `vec!`-based construction ergonomic; the `pub` fields accept the inline
/// collection directly. (Changing the field type from `Vec` is a 0.6.0 breaking
/// change.)
///
/// All inputs and destinations passed to [`fused_elementwise_into`] must have
/// the same shape and scalar type. Broadcast inputs should be represented with
/// `StridedView::broadcast` before building the plan; the fused API does not
/// perform implicit broadcasting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusedPlan {
    pub input_count: usize,
    pub outputs: strided_basic::OperandVec<usize>,
    pub ops: strided_basic::InlineVec<FusedInst>,
}

impl FusedPlan {
    /// Build a plan from an input count, output value ids, and instructions.
    ///
    /// ```
    /// use strided_fused::{FusedInst, FusedOp, FusedPlan};
    /// let plan = FusedPlan::new(
    ///     2,
    ///     vec![3],
    ///     vec![
    ///         FusedInst::new(FusedOp::Add, vec![0, 1]),
    ///         FusedInst::new(FusedOp::Negate, vec![2]),
    ///     ],
    /// );
    /// assert_eq!(plan.ops.len(), 2);
    /// ```
    pub fn new(
        input_count: usize,
        outputs: impl IntoIterator<Item = usize>,
        ops: impl IntoIterator<Item = FusedInst>,
    ) -> Self {
        Self {
            input_count,
            outputs: outputs.into_iter().collect(),
            ops: ops.into_iter().collect(),
        }
    }
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
                // Same select as the ordinary binary kernel; see
                // `strided_basic::nan_propagating_maximum`.
                strided_basic::nan_propagating_maximum(self, rhs)
            }

            #[inline(always)]
            fn fused_minimum(self, rhs: Self) -> Self {
                strided_basic::nan_propagating_minimum(self, rhs)
            }

            #[inline(always)]
            fn fused_clamp(self, min: Self, max: Self) -> Self {
                // `minimum(hi, maximum(lo, x))`, the ordinary clamp
                // composition; the argument order keeps signed-zero ties on
                // the bounds and propagates NaN from any operand.
                max.fused_minimum(min.fused_maximum(self))
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
    ($ty:ty, $div:path) => {
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
                $div(self, rhs)
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
impl_complex_fused_scalar!(
    num_complex::Complex32,
    strided_basic::robust_complex_divide_f32
);
impl_complex_fused_scalar!(
    num_complex::Complex64,
    strided_basic::robust_complex_divide_f64
);

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
) -> Result<ValidatedDestinationLayout> {
    let dims = dests[0].dims();
    // Validate the layout once and hand the marker to the specialization, so
    // the injectivity check does not run twice on the hot single-output path.
    let first = validate_destination_layout(dests[0].dims(), dests[0].strides())?;
    for dest in &dests[1..] {
        validate_destination_layout(dest.dims(), dest.strides())?;
        ensure_same_shape(dims, dest.dims())?;
    }
    for input in inputs {
        ensure_same_shape(dims, input.dims())?;
    }
    Ok(first)
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

#[cfg(test)]
fn reset_uninitialized_static_family_hits() {
    UNINITIALIZED_STATIC_FAMILY_HITS.set([0; 7]);
}

#[cfg(test)]
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
            // Dispatch on the op once, outside the element loop, so the
            // specialization is monomorphized per `FusedOp` and vectorizes
            // (tensor4all/strided-rs#305 item 2).
            macro_rules! unary {
                ($apply:expr) => {{
                    // SAFETY: matching shapes and this destination layout were
                    // validated before specialization/replay.
                    unsafe {
                        map_into_validated(dest, &inputs[a], |x| O::write($apply(x)), validated)
                    }?
                }};
            }
            match op {
                FusedOp::Negate => unary!(|x: T| x.fused_negate()),
                FusedOp::Conj => unary!(|x: T| x.fused_conj()),
                FusedOp::Abs => unary!(|x: T| x.fused_abs()),
                FusedOp::Exp => unary!(|x: T| x.fused_exp()),
                FusedOp::Log => unary!(|x: T| x.fused_log()),
                FusedOp::Sin => unary!(|x: T| x.fused_sin()),
                FusedOp::Cos => unary!(|x: T| x.fused_cos()),
                FusedOp::Tanh => unary!(|x: T| x.fused_tanh()),
                FusedOp::Sqrt => unary!(|x: T| x.fused_sqrt()),
                FusedOp::Rsqrt => unary!(|x: T| x.fused_rsqrt()),
                FusedOp::Expm1 => unary!(|x: T| x.fused_expm1()),
                FusedOp::Log1p => unary!(|x: T| x.fused_log1p()),
                _ => unreachable!("not a unary fused op: {op:?}"),
            }
        }
        StaticFusedKind::Binary(op, a, b) => {
            // As above: one monomorphized kernel per `FusedOp`.
            macro_rules! binary {
                ($apply:expr) => {{
                    // SAFETY: matching shapes and this destination layout were
                    // validated before specialization/replay.
                    unsafe {
                        zip_map2_into_validated(
                            dest,
                            &inputs[a],
                            &inputs[b],
                            |x, y| O::write($apply(x, y)),
                            validated,
                        )
                    }?
                }};
            }
            match op {
                FusedOp::Add => binary!(|x: T, y: T| x.fused_add(y)),
                FusedOp::Multiply => binary!(|x: T, y: T| x.fused_multiply(y)),
                FusedOp::Divide => binary!(|x: T, y: T| x.fused_divide(y)),
                FusedOp::Maximum => binary!(|x: T, y: T| x.fused_maximum(y)),
                FusedOp::Minimum => binary!(|x: T, y: T| x.fused_minimum(y)),
                FusedOp::Pow => binary!(|x: T, y: T| x.fused_pow(y)),
                _ => unreachable!("not a binary fused op: {op:?}"),
            }
        }
        StaticFusedKind::Ternary(op, a, b, c) => match op {
            FusedOp::Clamp => {
                // SAFETY: matching shapes and this destination layout were
                // validated before specialization/replay.
                unsafe {
                    zip_map3_into_validated(
                        dest,
                        &inputs[a],
                        &inputs[b],
                        &inputs[c],
                        |x, y, z| O::write(x.fused_clamp(y, z)),
                        validated,
                    )
                }?
            }
            _ => unreachable!("not a ternary fused op: {op:?}"),
        },
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

/// Take the initialized specialization with an already validated destination
/// layout, so the caller does not run the injectivity check twice.
fn try_static_specialization_initialized<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    validated: ValidatedDestinationLayout,
    serial: bool,
) -> Result<bool> {
    if dests.len() != 1 {
        return Ok(false);
    }
    try_static_specialization_validated_scoped::<T, InitializedStaticOutput>(
        &mut dests[0],
        inputs,
        plan,
        validated,
        serial,
    )
}

/// Test-only thin wrapper over `validate_destination_layout_without_alloc` plus
/// [`try_static_specialization_initialized`], matching the helper the
/// specialization tests used before the layout marker was threaded through.
#[cfg(test)]
fn try_static_specialization<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<bool> {
    if dests.len() != 1 {
        return Ok(false);
    }
    let validated = strided_basic::execution::validate_destination_layout_without_alloc(
        dests[0].dims(),
        dests[0].strides(),
    )?;
    try_static_specialization_initialized(dests, inputs, plan, validated, false)
}

/// Static specialization with an explicit serial request.
///
/// The specialized kernels choose their own fanout from the ambient execution
/// policy, so a serial caller pins [`ExecutionPolicy::Sequential`] while they
/// run. Skipping the specialization instead is what made serial replay fall
/// back to the per-element interpreter (tensor4all/strided-rs#305).
fn try_static_specialization_validated_scoped<T, O>(
    dest: &mut StridedViewMut<'_, O::Value>,
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    validated: ValidatedDestinationLayout,
    serial: bool,
) -> Result<bool>
where
    T: FusedScalar,
    O: StaticOutput<T>,
{
    if serial {
        with_execution_policy(ExecutionPolicy::Sequential, || {
            try_static_specialization_validated::<T, O>(dest, inputs, plan, validated)
        })
    } else {
        try_static_specialization_validated::<T, O>(dest, inputs, plan, validated)
    }
}

/// Registers the scalar interpreter keeps inline; a larger program spills to
/// the heap (#306 item 6).
const INLINE_REGS: usize = 12;

unsafe fn interpret_inner_loop<T: FusedScalar>(
    dst_ptrs: &[*mut T],
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
) {
    let register_count = plan.input_count + plan.ops.len();
    if len == 0 || register_count == 0 {
        return;
    }
    // SAFETY: a validated plan with instructions has at least one input, and
    // the enclosing kernel proved every operand addresses a readable element.
    let init = unsafe { *input_ptrs[0].offset(offsets[dst_ptrs.len()]) };
    if register_count <= INLINE_REGS {
        let mut inline = [init; INLINE_REGS];
        // SAFETY: forwarded contract.
        unsafe {
            interpret_loop_body(
                dst_ptrs,
                input_ptrs,
                plan,
                offsets,
                len,
                strides,
                &mut inline[..register_count],
            )
        };
    } else {
        let mut heap = vec![init; register_count];
        // SAFETY: forwarded contract.
        unsafe {
            interpret_loop_body(dst_ptrs, input_ptrs, plan, offsets, len, strides, &mut heap)
        };
    }
}

/// One interpreter pass over a block, writing the register file by index so a
/// stack or heap file is used identically.
///
/// # Safety
/// As [`interpret_inner_loop`].
#[allow(clippy::too_many_arguments)]
unsafe fn interpret_loop_body<T: FusedScalar>(
    dst_ptrs: &[*mut T],
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
    regs: &mut [T],
) {
    let output_count = dst_ptrs.len();
    for i in 0..len {
        let i = i as isize;
        for (input_index, &input_ptr) in input_ptrs.iter().enumerate() {
            let stride_index = output_count + input_index;
            let stride = inner_stride(strides, stride_index);
            // SAFETY: forwarded caller contract.
            regs[input_index] = unsafe { *input_ptr.offset(offsets[stride_index] + i * stride) };
        }
        for (instruction, inst) in plan.ops.iter().enumerate() {
            let output = plan.input_count + instruction;
            regs[output] = eval_op(inst.op, &regs[..output], &inst.inputs);
        }
        for (output_index, &dst_ptr) in dst_ptrs.iter().enumerate() {
            let stride = inner_stride(strides, output_index);
            // SAFETY: forwarded caller contract.
            unsafe {
                *dst_ptr.offset(offsets[output_index] + i * stride) =
                    regs[plan.outputs[output_index]];
            }
        }
    }
}

/// Seed for the chunk register file, or `None` when this scalar has no vector
/// kernels (or there is no input to seed from).
#[inline]
fn chunk_seed<T: FusedScalar>(inputs: &[StridedView<'_, T>]) -> Option<T> {
    if inputs.is_empty() || !chunk::is_real_chunk_scalar::<T>() {
        return None;
    }
    // SAFETY: callers reach this with a non-empty destination, and every input
    // shares the destination shape, so the first input has a readable element.
    Some(unsafe { *inputs[0].ptr() })
}

/// Register file for `seed`, or an empty one when the chunked path is declined.
#[inline]
fn register_file<T: FusedScalar, R>(
    plan: &FusedPlan,
    seed: Option<T>,
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    chunk::with_register_file(plan, seed, f)
}

/// Inputs the chunked path reads straight from the operand, empty when it is
/// declined.
///
/// A stride-0 input is never direct: the register machine reads the broadcast
/// value from its pre-filled register chunk, not from `offset + done`. The
/// innermost stride of each operand (outputs first, then inputs) is read from
/// `ordered_strides[i].first()`.
#[inline]
fn direct_inputs<T: FusedScalar, S: AsRef<[isize]>>(
    plan: &FusedPlan,
    seed: Option<T>,
    ordered_strides: &[S],
    output_count: usize,
) -> strided_basic::OperandVec<bool> {
    match seed {
        Some(_) => {
            let mut direct = chunk::single_use_inputs(plan);
            for (index, flag) in direct.iter_mut().enumerate() {
                let stride = ordered_strides
                    .get(output_count + index)
                    .and_then(|strides| strides.as_ref().first())
                    .copied()
                    .unwrap_or(0);
                if stride != 1 {
                    *flag = false;
                }
            }
            direct
        }
        None => strided_basic::OperandVec::new(),
    }
}

/// Stride of `index` along the block's innermost axis.
///
/// A rank-zero block carries no strides: its single element sits at the block
/// offset, which is what a stride of 0 yields.
#[inline]
fn inner_stride(strides: &[isize], index: usize) -> isize {
    strides.get(index).copied().unwrap_or(0)
}

fn interpret_fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    #[cfg(feature = "parallel")]
    {
        if dests[0].is_empty() {
            return Ok(());
        }
        let dims: strided_basic::InlineVec<usize> = dests[0].dims().iter().copied().collect();

        let dst_ptrs: strided_basic::OperandVec<OperandPtr<T>> = dests
            .iter_mut()
            .map(|dest| OperandPtr::new(dest.as_mut_ptr()))
            .collect();
        let input_ptrs: strided_basic::OperandVec<OperandPtr<T>> = inputs
            .iter()
            .map(|input| OperandPtr::new(input.ptr() as *mut T))
            .collect();

        let mut strides_list: strided_basic::OperandVec<&[isize]> =
            strided_basic::OperandVec::with_capacity(dests.len() + inputs.len());
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
            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
            let costs = unsafe { compute_costs(&ordered_strides) };
            let initial_offsets: strided_basic::OperandVec<isize> =
                ordered_strides.iter().map(|_| 0isize).collect();
            let seed = chunk_seed(inputs);
            let operands = dst_ptrs.len() + input_ptrs.len();
            let run_partition = |dims: &[usize],
                                 blocks: &[usize],
                                 strides_list: &[&[isize]],
                                 offsets: &[isize]|
             -> Result<()> {
                let direct = direct_inputs(plan, seed, strides_list, dst_ptrs.len());
                register_file(plan, seed, |regs| {
                    let run_block =
                        |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
                            if seed.is_some()
                                && chunk::block_replay_eligible(strides, dst_ptrs.len(), operands)
                            {
                                // SAFETY: validated shapes/layouts and the derived block bound every access.
                                unsafe {
                                    chunk::replay_chunked(
                                        operand_ptrs(&dst_ptrs),
                                        operand_const_ptrs(&input_ptrs),
                                        plan,
                                        offsets,
                                        len,
                                        strides,
                                        &direct,
                                        regs,
                                    );
                                }
                                return Ok(());
                            }
                            // SAFETY: validated shapes/layouts and the derived block bound every access.
                            unsafe {
                                interpret_inner_loop(
                                    operand_ptrs(&dst_ptrs),
                                    operand_const_ptrs(&input_ptrs),
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
                        for_each_inner_block_preordered(
                            dims,
                            blocks,
                            strides_list,
                            offsets,
                            run_block,
                        )
                    }
                })
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
    if dests[0].is_empty() {
        return Ok(());
    }
    let dims: strided_basic::InlineVec<usize> = dests[0].dims().iter().copied().collect();

    let dst_ptrs: strided_basic::OperandVec<OperandPtr<T>> = dests
        .iter_mut()
        .map(|dest| OperandPtr::new(dest.as_mut_ptr()))
        .collect();
    let input_ptrs: strided_basic::OperandVec<OperandPtr<T>> = inputs
        .iter()
        .map(|input| OperandPtr::new(input.ptr() as *mut T))
        .collect();

    let mut strides_list: strided_basic::OperandVec<&[isize]> =
        strided_basic::OperandVec::with_capacity(dests.len() + inputs.len());
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

    let initial_offsets: strided_basic::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
    let seed = chunk_seed(inputs);
    let operands = dst_ptrs.len() + input_ptrs.len();
    let direct = direct_inputs(plan, seed, &ordered_strides, dst_ptrs.len());
    register_file(plan, seed, |regs| {
        let run_block = |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
            if seed.is_some() && chunk::block_replay_eligible(strides, dst_ptrs.len(), operands) {
                // SAFETY: validated shapes/layouts and the derived block bound every access.
                unsafe {
                    chunk::replay_chunked(
                        operand_ptrs(&dst_ptrs),
                        operand_const_ptrs(&input_ptrs),
                        plan,
                        offsets,
                        len,
                        strides,
                        &direct,
                        regs,
                    );
                }
                return Ok(());
            }
            // SAFETY: validated shapes/layouts and the derived block bound every access.
            unsafe {
                interpret_inner_loop(
                    operand_ptrs(&dst_ptrs),
                    operand_const_ptrs(&input_ptrs),
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
            for_each_inner_block_preordered(
                &fused_dims,
                &kernel_plan.block,
                &ordered_strides,
                &initial_offsets,
                run_block,
            )
        }
    })
}

pub(crate) fn fused_elementwise_into_serial<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    validate_plan_for_scalar::<T>(plan, inputs.len(), dests.len())?;
    let validated = validate_shapes(dests, inputs)?;
    if try_static_specialization_initialized(dests, inputs, plan, validated, true)? {
        return Ok(());
    }
    interpret_fused_elementwise_into_serial(dests, inputs, plan)
}

unsafe fn interpret_inner_loop_uninit<T: FusedScalar>(
    dst_ptrs: &[*mut MaybeUninit<T>],
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
) {
    let register_count = plan.input_count + plan.ops.len();
    if len == 0 || register_count == 0 {
        return;
    }
    // SAFETY: a validated plan with instructions has at least one input, and
    // the enclosing kernel proved every operand addresses a readable element.
    let init = unsafe { *input_ptrs[0].offset(offsets[dst_ptrs.len()]) };
    if register_count <= INLINE_REGS {
        let mut inline = [init; INLINE_REGS];
        // SAFETY: forwarded contract.
        unsafe {
            interpret_loop_body_uninit(
                dst_ptrs,
                input_ptrs,
                plan,
                offsets,
                len,
                strides,
                &mut inline[..register_count],
            )
        };
    } else {
        let mut heap = vec![init; register_count];
        // SAFETY: forwarded contract.
        unsafe {
            interpret_loop_body_uninit(dst_ptrs, input_ptrs, plan, offsets, len, strides, &mut heap)
        };
    }
}

/// Uninitialized-destination sibling of [`interpret_loop_body`].
///
/// # Safety
/// As [`interpret_inner_loop_uninit`].
#[allow(clippy::too_many_arguments)]
unsafe fn interpret_loop_body_uninit<T: FusedScalar>(
    dst_ptrs: &[*mut MaybeUninit<T>],
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
    regs: &mut [T],
) {
    let output_count = dst_ptrs.len();
    for i in 0..len {
        let i = i as isize;
        for (input_index, &input_ptr) in input_ptrs.iter().enumerate() {
            let stride_index = output_count + input_index;
            let stride = inner_stride(strides, stride_index);
            // SAFETY: forwarded caller contract.
            regs[input_index] = unsafe { *input_ptr.offset(offsets[stride_index] + i * stride) };
        }
        for (instruction, inst) in plan.ops.iter().enumerate() {
            let output = plan.input_count + instruction;
            regs[output] = eval_op(inst.op, &regs[..output], &inst.inputs);
        }
        for (output_index, &dst_ptr) in dst_ptrs.iter().enumerate() {
            let stride = inner_stride(strides, output_index);
            // SAFETY: forwarded caller contract.
            unsafe {
                *dst_ptr.offset(offsets[output_index] + i * stride) =
                    MaybeUninit::new(regs[plan.outputs[output_index]]);
            }
        }
    }
}

/// Single-destination front end for [`fused_elementwise_into_uninit_multi`].
#[cfg(test)]
pub(crate) fn fused_elementwise_into_uninit<T: FusedScalar>(
    dest: &mut StridedViewMut<'_, MaybeUninit<T>>,
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    serial: bool,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    fused_elementwise_into_uninit_multi(
        std::slice::from_mut(dest),
        inputs,
        plan,
        serial,
        std::slice::from_ref(&validated),
    )
}

/// Evaluate a plan into one or more fully-overwritten uninitialized destinations.
///
/// `validated[i]` must come from `validate_destination_layout_without_alloc`
/// for `dests[i]`'s layout. A single destination keeps the static kernel
/// specialization; several destinations always take the interpreter or the
/// chunked register machine.
pub(crate) fn fused_elementwise_into_uninit_multi<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, MaybeUninit<T>>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    serial: bool,
    validated: &[ValidatedDestinationLayout],
) -> Result<()> {
    validate_plan_for_scalar::<T>(plan, inputs.len(), dests.len())?;
    let dims = dests[0].dims();
    for dest in dests.iter() {
        ensure_same_shape(dims, dest.dims())?;
    }
    for input in inputs {
        ensure_same_shape(dims, input.dims())?;
    }

    if dests.len() == 1
        && try_static_specialization_validated_scoped::<T, UninitializedStaticOutput>(
            &mut dests[0],
            inputs,
            plan,
            validated[0],
            serial,
        )?
    {
        return Ok(());
    }

    if dests[0].is_empty() {
        return Ok(());
    }
    interpret_fused_elementwise_into_uninit(dests, inputs, plan, serial)
}

fn interpret_fused_elementwise_into_uninit<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, MaybeUninit<T>>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    _serial: bool,
) -> Result<()> {
    let dims: strided_basic::InlineVec<usize> = dests[0].dims().iter().copied().collect();
    let dst_ptrs: strided_basic::OperandVec<OperandPtr<MaybeUninit<T>>> = dests
        .iter_mut()
        .map(|dest| OperandPtr::new(dest.as_mut_ptr()))
        .collect();
    let input_ptrs: strided_basic::OperandVec<OperandPtr<T>> = inputs
        .iter()
        .map(|input| OperandPtr::new(input.ptr() as *mut T))
        .collect();
    let mut strides_list: strided_basic::OperandVec<&[isize]> =
        strided_basic::OperandVec::with_capacity(dests.len() + inputs.len());
    for dest in dests.iter() {
        strides_list.push(dest.strides());
    }
    for input in inputs {
        strides_list.push(input.strides());
    }
    let total = dests[0].len();
    let (fused_dims, ordered_strides, kernel_plan) = if total <= SMALL_TENSOR_THRESHOLD {
        // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
        unsafe { build_plan_fused_small(&dims, &strides_list) }
    } else {
        // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
        unsafe { build_plan_fused(&dims, &strides_list, Some(0), core::mem::size_of::<T>()) }
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = strided_basic::execution::rayon_threads();
        if !_serial && total > MINTHREADLENGTH && nthreads > 1 {
            // SAFETY: the enclosing checked kernel supplies validated layout metadata and its derived partitions.
            let costs = unsafe { compute_costs(&ordered_strides) };
            let initial_offsets: strided_basic::OperandVec<isize> =
                ordered_strides.iter().map(|_| 0isize).collect();
            let seed = chunk_seed(inputs);
            let operands = dst_ptrs.len() + input_ptrs.len();
            let run_partition = |dims: &[usize],
                                 blocks: &[usize],
                                 strides_list: &[&[isize]],
                                 offsets: &[isize]|
             -> Result<()> {
                let direct = direct_inputs(plan, seed, strides_list, dst_ptrs.len());
                register_file(plan, seed, |regs| {
                    let run_block =
                        |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
                            if seed.is_some()
                                && chunk::block_replay_eligible(strides, dst_ptrs.len(), operands)
                            {
                                // SAFETY: validated shapes/layouts and the derived block bound every access.
                                unsafe {
                                    chunk::replay_chunked_uninit(
                                        operand_ptrs(&dst_ptrs),
                                        operand_const_ptrs(&input_ptrs),
                                        plan,
                                        offsets,
                                        len,
                                        strides,
                                        &direct,
                                        regs,
                                    );
                                }
                                return Ok(());
                            }
                            // SAFETY: validated shapes/layouts and the derived block bound every access.
                            unsafe {
                                interpret_inner_loop_uninit(
                                    operand_ptrs(&dst_ptrs),
                                    operand_const_ptrs(&input_ptrs),
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
                        for_each_inner_block_preordered(
                            dims,
                            blocks,
                            strides_list,
                            offsets,
                            run_block,
                        )
                    }
                })
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

    let initial_offsets: strided_basic::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
    let seed = chunk_seed(inputs);
    let operands = dst_ptrs.len() + input_ptrs.len();
    let direct = direct_inputs(plan, seed, &ordered_strides, dst_ptrs.len());
    register_file(plan, seed, |regs| {
        let run_block = |offsets: &[isize], len: usize, strides: &[isize]| -> Result<()> {
            if seed.is_some() && chunk::block_replay_eligible(strides, dst_ptrs.len(), operands) {
                // SAFETY: validated shapes/layouts and the derived block bound every access.
                unsafe {
                    chunk::replay_chunked_uninit(
                        operand_ptrs(&dst_ptrs),
                        operand_const_ptrs(&input_ptrs),
                        plan,
                        offsets,
                        len,
                        strides,
                        &direct,
                        regs,
                    );
                }
                return Ok(());
            }
            // SAFETY: validated shapes/layouts and the derived block bound every access.
            unsafe {
                interpret_inner_loop_uninit(
                    operand_ptrs(&dst_ptrs),
                    operand_const_ptrs(&input_ptrs),
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
            for_each_inner_block_preordered(
                &fused_dims,
                &kernel_plan.block,
                &ordered_strides,
                &initial_offsets,
                run_block,
            )
        }
    })
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
/// `map_into`/`zip_map*_into` kernels, replays a block that is contiguous for
/// every operand through a chunked register machine, and falls back to a
/// per-element interpreter for arbitrary validated DAGs. Overlapping
/// source/destination memory is not supported by the strided kernels generally.
///
/// # Numerics
///
/// The chunked replay evaluates `Exp`, `Log`, `Sin`, `Cos`, `Tanh` and `Expm1`
/// with the `elem-simd-math` slice kernels — the same kernels a per-instruction
/// dispatch through [`strided_basic::map_into_uninit_with_kernel`] uses, so the
/// fused and the non-fused route agree with each other. Against `std`'s scalar
/// functions those six instructions differ by at most one ULP of their own
/// output magnitude; a chain of them multiplies that difference by the output
/// magnitude, so an exponential chain is not bit-stable to `std`. `Log1p` and
/// every other instruction (`Add`, `Multiply`, `Negate`, `Conj`, `Divide`,
/// `Abs`, `Maximum`, `Minimum`, `Clamp`, `Sqrt`, `Rsqrt`, `Pow`) keep the scalar
/// `FusedScalar` methods and are bit-identical to the per-element replay.
///
/// Real `Maximum`, `Minimum`, and `Clamp` share the NaN-propagating select and
/// signed-zero contract of `strided_basic::nan_propagating_maximum` /
/// `nan_propagating_minimum` (issue #315): a NaN in any operand makes the
/// result NaN, and a tie keeps the left operand, so `maximum(-0.0, 0.0)` and
/// `clamp` on equal bounds preserve the signed zero of the left/bound operand.
/// Complex `Abs` returns the norm in the real component; complex `Maximum`,
/// `Minimum`, and `Clamp` compare by squared norm. Signed integer `Add`,
/// `Multiply`, `Negate`, and `Abs` use wrapping arithmetic. `bool` supports
/// only copy-like identity plans and `Conj`; ambiguous arithmetic and
/// transcendental op/dtype pairs are rejected before any destination is
/// written.
pub fn fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()> {
    validate_plan_for_scalar::<T>(plan, inputs.len(), dests.len())?;
    let validated = validate_shapes(dests, inputs)?;
    if try_static_specialization_initialized(dests, inputs, plan, validated, false)? {
        return Ok(());
    }
    interpret_fused_elementwise_into(dests, inputs, plan)
}

#[cfg(test)]
#[path = "fused/tests/tests.rs"]
mod tests;
