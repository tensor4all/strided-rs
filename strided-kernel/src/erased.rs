//! Dtype-erased prepared kernel entry points.
//!
//! These wrappers keep dtype-specific monomorphization inside `strided-kernel`
//! so downstream runtime crates can replay prepared kernels through stable,
//! non-generic entry points.
//!
//! C ABI symbols are intentionally out of scope here. A future ABI layer must
//! pass an explicit execution context, preserve the non-overlap contract for
//! descriptors used by one replay call, and validate ABI dtype tags before
//! constructing these Rust descriptors.
//!
//! The safe Rust descriptor constructors validate dtype byte layout up front.
//! Mutable erased descriptors only re-scan value-constrained dtypes, currently
//! `bool`, after their raw bytes have escaped through `data_mut`.

use core::ops::Add;

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use crate::{
    fused_elementwise_into, ConcatenatePlan, CopyPlan, DynamicSlicePlan, DynamicUpdateSlicePlan,
    ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, FusedPlan, FusedScalar, GatherIndex,
    GatherPlan, GatherSpec, Identity, KernelDType, PadPlan, RawStridedMut, RawStridedRef, Result,
    ReversePlan, ScatterPlan, ScatterSpec, SlicePlan, StridedError, StridedView, StridedViewMut,
    RAW_FUSED_RANK_LIMIT,
};

const ERASED_FUSED_INPUT_LIMIT: usize = 4;

/// Runtime unary operation for [`erased_map_into`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErasedMapOp {
    Negate,
    Conj,
    Abs,
    Sign,
}

impl ErasedMapOp {
    const fn label(self) -> &'static str {
        match self {
            Self::Negate => "negate",
            Self::Conj => "conj",
            Self::Abs => "abs",
            Self::Sign => "sign",
        }
    }
}

/// Runtime binary operation for [`erased_zip_into`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErasedZipOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Maximum,
    Minimum,
}

impl ErasedZipOp {
    const fn label(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Remainder => "remainder",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
        }
    }
}

/// Apply one runtime-selected unary operation without compiling a plan.
///
/// The destination must not overlap the input. Real and complex dtypes support
/// every [`ErasedMapOp`], signed integers use wrapping negate/abs semantics,
/// and `bool` supports only [`ErasedMapOp::Conj`].
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, overlap,
/// or unsupported dtype/op contracts. Validation completes before any write.
pub fn erased_map_into(
    dtype: KernelDType,
    op: ErasedMapOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    input: &ErasedRawStridedRef<'_>,
) -> Result<()> {
    check_dtype(dtype, dest.dtype())?;
    check_dtype(dtype, input.dtype())?;
    validate_no_overlap(dest, input, 0)?;
    dest.validate_data_if_needed()?;

    let result = ctx.run(|| match dtype {
        KernelDType::F32 => execute_one_shot_map::<f32>(op, dest, input),
        KernelDType::F64 => execute_one_shot_map::<f64>(op, dest, input),
        KernelDType::I32 => execute_one_shot_map::<i32>(op, dest, input),
        KernelDType::I64 => execute_one_shot_map::<i64>(op, dest, input),
        KernelDType::Bool => execute_one_shot_map::<bool>(op, dest, input),
        KernelDType::C32 => execute_one_shot_map::<Complex32>(op, dest, input),
        KernelDType::C64 => execute_one_shot_map::<Complex64>(op, dest, input),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    });
    if result.is_ok() {
        // SAFETY: the scalar map writes valid values of the descriptor dtype.
        unsafe {
            dest.assume_data_valid();
        }
    }
    result
}

/// Apply one runtime-selected binary operation without compiling a plan.
///
/// The destination must not overlap either input. Real dtypes support every
/// [`ErasedZipOp`]. Signed integers support add/subtract/multiply/min/max, and
/// complex dtypes support add/subtract/multiply/divide. `bool` has no binary
/// one-shot operations.
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, overlap,
/// or unsupported dtype/op contracts. Validation completes before any write.
pub fn erased_zip_into(
    dtype: KernelDType,
    op: ErasedZipOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    rhs: &ErasedRawStridedRef<'_>,
) -> Result<()> {
    check_dtype(dtype, dest.dtype())?;
    check_dtype(dtype, lhs.dtype())?;
    check_dtype(dtype, rhs.dtype())?;
    validate_no_overlap(dest, lhs, 0)?;
    validate_no_overlap(dest, rhs, 1)?;
    dest.validate_data_if_needed()?;

    let result = ctx.run(|| match dtype {
        KernelDType::F32 => execute_one_shot_zip::<f32>(op, dest, lhs, rhs),
        KernelDType::F64 => execute_one_shot_zip::<f64>(op, dest, lhs, rhs),
        KernelDType::I32 => execute_one_shot_zip::<i32>(op, dest, lhs, rhs),
        KernelDType::I64 => execute_one_shot_zip::<i64>(op, dest, lhs, rhs),
        KernelDType::Bool => execute_one_shot_zip::<bool>(op, dest, lhs, rhs),
        KernelDType::C32 => execute_one_shot_zip::<Complex32>(op, dest, lhs, rhs),
        KernelDType::C64 => execute_one_shot_zip::<Complex64>(op, dest, lhs, rhs),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    });
    if result.is_ok() {
        // SAFETY: the scalar zip writes valid values of the descriptor dtype.
        unsafe {
            dest.assume_data_valid();
        }
    }
    result
}

/// Dtype-erased wrapper around [`CopyPlan`].
#[derive(Clone, Debug)]
pub struct ErasedCopyPlan {
    dtype: KernelDType,
    plan: CopyPlan,
}

/// Dtype-erased static-slice wrapper.
#[derive(Clone, Debug)]
pub struct ErasedSlicePlan {
    dtype: KernelDType,
    plan: SlicePlan,
}

/// Dtype-erased reverse wrapper.
#[derive(Clone, Debug)]
pub struct ErasedReversePlan {
    dtype: KernelDType,
    plan: ReversePlan,
}

/// Dtype-erased pad wrapper.
#[derive(Clone, Debug)]
pub struct ErasedPadPlan {
    dtype: KernelDType,
    plan: PadPlan,
}

/// Dtype-erased concatenate wrapper.
#[derive(Clone, Debug)]
pub struct ErasedConcatenatePlan {
    dtype: KernelDType,
    plan: ConcatenatePlan,
}

impl ErasedCopyPlan {
    /// Compile a copy plan for one dtype and layout pair.
    pub fn compile(
        dtype: KernelDType,
        dims: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
    ) -> Result<Self> {
        Ok(Self {
            dtype,
            plan: CopyPlan::compile(dims, dst_strides, src_strides)?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    /// `dest = src` through a non-generic dtype-erased replay boundary.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        self.check_dtype(dest.dtype())?;
        self.check_dtype(src.dtype())?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_copy::<f32>(&self.plan, dest, src),
            KernelDType::F64 => execute_copy::<f64>(&self.plan, dest, src),
            KernelDType::I32 => execute_copy::<i32>(&self.plan, dest, src),
            KernelDType::I64 => execute_copy::<i64>(&self.plan, dest, src),
            KernelDType::Bool => execute_copy::<bool>(&self.plan, dest, src),
            KernelDType::C32 => execute_copy::<Complex32>(&self.plan, dest, src),
            KernelDType::C64 => execute_copy::<Complex64>(&self.plan, dest, src),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: `execute_copy` only writes values produced from the
            // already-validated source descriptor for the same dtype.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }

    fn check_dtype(&self, actual: KernelDType) -> Result<()> {
        if actual != self.dtype {
            return Err(StridedError::DTypeMismatch {
                expected: self.dtype.label(),
                actual: actual.label(),
            });
        }
        Ok(())
    }
}

impl ErasedSlicePlan {
    /// Validate and store a static slice plan for one dtype and fixed layout set.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        dtype: KernelDType,
        operand_dims: &[usize],
        operand_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        starts: &[usize],
        limits: &[usize],
        slice_strides: &[usize],
    ) -> Result<Self> {
        check_static_indexing_dtype(dtype)?;
        Ok(Self {
            dtype,
            plan: SlicePlan::compile(
                operand_dims,
                operand_strides,
                dest_dims,
                dest_strides,
                starts,
                limits,
                slice_strides,
            )?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn plan(&self) -> &SlicePlan {
        &self.plan
    }

    /// Execute a static slice into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_slice::<f32>(&self.plan, dest, operand),
            KernelDType::F64 => execute_slice::<f64>(&self.plan, dest, operand),
            KernelDType::I32 => execute_slice::<i32>(&self.plan, dest, operand),
            KernelDType::I64 => execute_slice::<i64>(&self.plan, dest, operand),
            KernelDType::Bool => execute_slice::<bool>(&self.plan, dest, operand),
            KernelDType::C32 => execute_slice::<Complex32>(&self.plan, dest, operand),
            KernelDType::C64 => execute_slice::<Complex64>(&self.plan, dest, operand),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: static slice writes values read from a descriptor with
            // the same dtype and already-validated byte representation.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedReversePlan {
    /// Validate and store a reverse plan for one dtype and fixed layout set.
    pub fn compile(
        dtype: KernelDType,
        operand_dims: &[usize],
        operand_strides: &[isize],
        dest_strides: &[isize],
        axes: &[usize],
    ) -> Result<Self> {
        check_static_indexing_dtype(dtype)?;
        Ok(Self {
            dtype,
            plan: ReversePlan::compile(operand_dims, operand_strides, dest_strides, axes)?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn plan(&self) -> &ReversePlan {
        &self.plan
    }

    /// Execute a reverse into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_reverse::<f32>(&self.plan, dest, operand),
            KernelDType::F64 => execute_reverse::<f64>(&self.plan, dest, operand),
            KernelDType::I32 => execute_reverse::<i32>(&self.plan, dest, operand),
            KernelDType::I64 => execute_reverse::<i64>(&self.plan, dest, operand),
            KernelDType::Bool => execute_reverse::<bool>(&self.plan, dest, operand),
            KernelDType::C32 => execute_reverse::<Complex32>(&self.plan, dest, operand),
            KernelDType::C64 => execute_reverse::<Complex64>(&self.plan, dest, operand),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: reverse writes values read from a descriptor with the
            // same dtype and already-validated byte representation.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedPadPlan {
    /// Validate and store a pad plan for one dtype and fixed layout set.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        dtype: KernelDType,
        operand_dims: &[usize],
        operand_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[i64],
    ) -> Result<Self> {
        check_static_indexing_dtype(dtype)?;
        Ok(Self {
            dtype,
            plan: PadPlan::compile(
                operand_dims,
                operand_strides,
                dest_dims,
                dest_strides,
                edge_padding_low,
                edge_padding_high,
                interior_padding,
            )?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn plan(&self) -> &PadPlan {
        &self.plan
    }

    /// Execute pad into an erased output descriptor using one dtype scalar as fill.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
        fill: &[u8],
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        validate_scalar_bytes(self.dtype, fill)?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_pad::<f32>(&self.plan, dest, operand, fill),
            KernelDType::F64 => execute_pad::<f64>(&self.plan, dest, operand, fill),
            KernelDType::I32 => execute_pad::<i32>(&self.plan, dest, operand, fill),
            KernelDType::I64 => execute_pad::<i64>(&self.plan, dest, operand, fill),
            KernelDType::Bool => execute_pad::<bool>(&self.plan, dest, operand, fill),
            KernelDType::C32 => execute_pad::<Complex32>(&self.plan, dest, operand, fill),
            KernelDType::C64 => execute_pad::<Complex64>(&self.plan, dest, operand, fill),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: pad writes either the validated fill scalar or values
            // read from a descriptor with the same validated dtype.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedConcatenatePlan {
    /// Validate and store a concatenate plan for one dtype and fixed layout set.
    pub fn compile(
        dtype: KernelDType,
        input_dims: &[&[usize]],
        input_strides: &[&[isize]],
        dest_dims: &[usize],
        dest_strides: &[isize],
        axis: usize,
    ) -> Result<Self> {
        check_static_indexing_dtype(dtype)?;
        Ok(Self {
            dtype,
            plan: ConcatenatePlan::compile(
                input_dims,
                input_strides,
                dest_dims,
                dest_strides,
                axis,
            )?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn plan(&self) -> &ConcatenatePlan {
        &self.plan
    }

    /// Execute concatenate into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        inputs: &[ErasedRawStridedRef<'_>],
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        for input in inputs {
            check_dtype(self.dtype, input.dtype())?;
        }
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_concatenate::<f32>(&self.plan, dest, inputs),
            KernelDType::F64 => execute_concatenate::<f64>(&self.plan, dest, inputs),
            KernelDType::I32 => execute_concatenate::<i32>(&self.plan, dest, inputs),
            KernelDType::I64 => execute_concatenate::<i64>(&self.plan, dest, inputs),
            KernelDType::Bool => execute_concatenate::<bool>(&self.plan, dest, inputs),
            KernelDType::C32 => execute_concatenate::<Complex32>(&self.plan, dest, inputs),
            KernelDType::C64 => execute_concatenate::<Complex64>(&self.plan, dest, inputs),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: concatenate writes values read from descriptors with
            // the same dtype and already-validated byte representation.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

/// Dtype-erased single-output wrapper around [`FusedPlan`].
///
/// This is the erased replay boundary for unary map and zip-map elementwise
/// families. It supports the same runtime op-code vocabulary as [`FusedPlan`],
/// but only for the scalar dtypes currently implementing [`FusedScalar`].
#[derive(Clone, Debug)]
pub struct ErasedFusedPlan {
    dtype: KernelDType,
    plan: FusedPlan,
}

/// Runtime reduction operation for dtype-erased full reductions.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceOp {
    Sum,
    Product,
}

/// Dtype-erased reduction wrapper.
///
/// This is the erased replay boundary for full-tensor scalar reductions and
/// axis reductions with a fixed output layout. It supports only operations with
/// an unambiguous identity value in the selected dtype.
#[derive(Clone, Debug)]
pub struct ErasedReducePlan {
    dtype: KernelDType,
    op: ReduceOp,
    layout: ReduceLayout,
}

#[derive(Clone, Debug)]
enum ReduceLayout {
    Full {
        dims: Vec<usize>,
        src_strides: Vec<isize>,
    },
    Axes {
        src_dims: Vec<usize>,
        src_strides: Vec<isize>,
        dest_dims: Vec<usize>,
        dest_strides: Vec<isize>,
        axes: Vec<usize>,
        kept_axes: Vec<usize>,
        reduce_dims: Vec<usize>,
        dest_total: usize,
        reduce_total: usize,
    },
}

impl ReduceLayout {
    fn src_dims(&self) -> &[usize] {
        match self {
            Self::Full { dims, .. } => dims,
            Self::Axes { src_dims, .. } => src_dims,
        }
    }

    fn src_strides(&self) -> &[isize] {
        match self {
            Self::Full { src_strides, .. } | Self::Axes { src_strides, .. } => src_strides,
        }
    }

    fn check_src_layout(&self, src: &ErasedRawStridedRef<'_>) -> Result<()> {
        if src.dims() != self.src_dims() || src.strides() != self.src_strides() {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct AxesLayout<'a> {
    src_dims: &'a [usize],
    src_strides: &'a [isize],
    dest_dims: &'a [usize],
    dest_strides: &'a [isize],
    axes: &'a [usize],
    kept_axes: &'a [usize],
    reduce_dims: &'a [usize],
    dest_total: usize,
    reduce_total: usize,
}

/// Dtype-erased gather wrapper.
///
/// This is the erased replay boundary for indexed reads. Value buffers use the
/// configured value dtype, while the index descriptor must use `i32` or `i64`.
#[derive(Clone, Debug)]
pub struct ErasedGatherPlan {
    dtype: KernelDType,
    index_dtype: KernelDType,
    plan: GatherPlan,
}

/// Dtype-erased fixed-window dynamic-slice wrapper.
#[derive(Clone, Debug)]
pub struct ErasedDynamicSlicePlan {
    dtype: KernelDType,
    index_dtype: KernelDType,
    plan: DynamicSlicePlan,
}

/// Dtype-erased dynamic-update-slice wrapper.
#[derive(Clone, Debug)]
pub struct ErasedDynamicUpdateSlicePlan {
    dtype: KernelDType,
    index_dtype: KernelDType,
    plan: DynamicUpdateSlicePlan,
}

/// Dtype-erased additive scatter wrapper.
#[derive(Clone, Debug)]
pub struct ErasedScatterPlan {
    dtype: KernelDType,
    index_dtype: KernelDType,
    plan: ScatterPlan,
}

impl ErasedFusedPlan {
    /// Validate and store a single-output fused elementwise plan for one dtype.
    pub fn compile(dtype: KernelDType, plan: FusedPlan) -> Result<Self> {
        check_fused_dtype(dtype)?;
        if plan.input_count == 0 || plan.input_count > ERASED_FUSED_INPUT_LIMIT {
            return Err(StridedError::UnsupportedArity {
                arity: plan.input_count,
                max: ERASED_FUSED_INPUT_LIMIT,
            });
        }
        if plan.outputs.len() != 1 {
            return Err(StridedError::RankMismatch(plan.outputs.len(), 1));
        }
        validate_fused_plan_for_dtype(dtype, &plan)?;
        Ok(Self { dtype, plan })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn plan(&self) -> &FusedPlan {
        &self.plan
    }

    /// Execute a single-output fused elementwise plan through erased descriptors.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        inputs: &[ErasedRawStridedRef<'_>],
    ) -> Result<()> {
        if inputs.len() != self.plan.input_count {
            return Err(StridedError::RankMismatch(
                inputs.len(),
                self.plan.input_count,
            ));
        }
        check_dtype(self.dtype, dest.dtype())?;
        for input in inputs {
            check_dtype(self.dtype, input.dtype())?;
        }
        dest.validate_data_if_needed()?;

        let result = match self.dtype {
            KernelDType::F32 => execute_fused::<f32>(&self.plan, ctx, dest, inputs),
            KernelDType::F64 => execute_fused::<f64>(&self.plan, ctx, dest, inputs),
            KernelDType::I32 => execute_fused::<i32>(&self.plan, ctx, dest, inputs),
            KernelDType::I64 => execute_fused::<i64>(&self.plan, ctx, dest, inputs),
            KernelDType::Bool => execute_fused::<bool>(&self.plan, ctx, dest, inputs),
            KernelDType::C32 => execute_fused::<Complex32>(&self.plan, ctx, dest, inputs),
            KernelDType::C64 => execute_fused::<Complex64>(&self.plan, ctx, dest, inputs),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        };
        if result.is_ok() {
            // SAFETY: supported fused elementwise dtypes have no extra byte
            // validity invariant beyond the typed values written by the kernel.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedReducePlan {
    /// Validate and store a full-reduction plan for one dtype and source layout.
    pub fn compile(
        dtype: KernelDType,
        op: ReduceOp,
        dims: &[usize],
        src_strides: &[isize],
    ) -> Result<Self> {
        check_reduce_dtype(dtype)?;
        if dims.len() != src_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        checked_total_len(dims)?;
        Ok(Self {
            dtype,
            op,
            layout: ReduceLayout::Full {
                dims: dims.to_vec(),
                src_strides: src_strides.to_vec(),
            },
        })
    }

    /// Validate and store an axis-reduction plan for one dtype and fixed source/output layouts.
    ///
    /// `axes` names the source axes reduced away. Output dimensions must be the
    /// remaining source dimensions in source-axis order. When all axes are
    /// reduced, any output layout with exactly one reachable element is accepted.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_axes(
        dtype: KernelDType,
        op: ReduceOp,
        src_dims: &[usize],
        src_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        axes: &[usize],
    ) -> Result<Self> {
        check_reduce_dtype(dtype)?;
        if src_dims.len() != src_strides.len() || dest_dims.len() != dest_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        checked_total_len(src_dims)?;
        let dest_total = checked_total_len(dest_dims)?;
        if !crate::fused::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        validate_unique_axes(axes, src_dims.len())?;

        let kept_axes: Vec<usize> = (0..src_dims.len())
            .filter(|axis| !axes.contains(axis))
            .collect();
        let expected_dest_dims: Vec<usize> = kept_axes.iter().map(|&axis| src_dims[axis]).collect();
        if expected_dest_dims.is_empty() {
            if dest_total != 1 {
                return Err(StridedError::ShapeMismatch(
                    dest_dims.to_vec(),
                    expected_dest_dims,
                ));
            }
        } else if dest_dims != expected_dest_dims.as_slice() {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                expected_dest_dims,
            ));
        }

        let reduce_dims = axes.iter().map(|&axis| src_dims[axis]).collect::<Vec<_>>();
        let reduce_total = checked_total_len(&reduce_dims)?;
        Ok(Self {
            dtype,
            op,
            layout: ReduceLayout::Axes {
                src_dims: src_dims.to_vec(),
                src_strides: src_strides.to_vec(),
                dest_dims: dest_dims.to_vec(),
                dest_strides: dest_strides.to_vec(),
                axes: axes.to_vec(),
                kept_axes,
                reduce_dims,
                dest_total,
                reduce_total,
            },
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn op(&self) -> ReduceOp {
        self.op
    }

    /// Execute the reduction into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        self.layout.check_src_layout(src)?;
        match &self.layout {
            ReduceLayout::Full { .. } => {
                let dest_len = checked_total_len(dest.dims())?;
                if dest_len != 1 {
                    return Err(StridedError::RankMismatch(dest_len, 1));
                }
            }
            ReduceLayout::Axes {
                dest_dims,
                dest_strides,
                ..
            } => {
                if dest.dims() != dest_dims.as_slice() || dest.strides() != dest_strides.as_slice()
                {
                    return Err(StridedError::PlanLayoutMismatch);
                }
            }
        }
        dest.validate_data_if_needed()?;

        let result = match self.dtype {
            KernelDType::F32 => dispatch_reduce::<f32>(self.op, &self.layout, ctx, dest, src),
            KernelDType::F64 => dispatch_reduce::<f64>(self.op, &self.layout, ctx, dest, src),
            KernelDType::I32 => dispatch_reduce::<i32>(self.op, &self.layout, ctx, dest, src),
            KernelDType::I64 => dispatch_reduce::<i64>(self.op, &self.layout, ctx, dest, src),
            KernelDType::C32 => dispatch_reduce::<Complex32>(self.op, &self.layout, ctx, dest, src),
            KernelDType::C64 => dispatch_reduce::<Complex64>(self.op, &self.layout, ctx, dest, src),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        };
        if result.is_ok() {
            // SAFETY: supported reduction dtypes have no extra byte validity
            // invariant beyond the typed scalar written by the kernel.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedGatherPlan {
    /// Validate and store a gather plan for one value dtype, index dtype, and layout set.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        dtype: KernelDType,
        index_dtype: KernelDType,
        operand_dims: &[usize],
        operand_strides: &[isize],
        index_dims: &[usize],
        index_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        spec: GatherSpec,
    ) -> Result<Self> {
        check_index_dtype(index_dtype)?;
        check_gather_value_dtype(dtype)?;
        Ok(Self {
            dtype,
            index_dtype,
            plan: GatherPlan::compile(
                operand_dims,
                operand_strides,
                index_dims,
                index_strides,
                dest_dims,
                dest_strides,
                spec,
            )?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn index_dtype(&self) -> KernelDType {
        self.index_dtype
    }

    #[inline]
    pub fn plan(&self) -> &GatherPlan {
        &self.plan
    }

    /// Execute an indexed read into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
        start_indices: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.index_dtype, start_indices.dtype())?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_gather_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::F64 => dispatch_gather_index::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::I32 => dispatch_gather_index::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::I64 => dispatch_gather_index::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::Bool => dispatch_gather_index::<bool>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::C32 => dispatch_gather_index::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::C64 => dispatch_gather_index::<Complex64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: gather writes values read from a descriptor with the
            // same dtype and already-validated byte representation.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedDynamicSlicePlan {
    /// Validate and store a dynamic-slice plan for one value dtype, index dtype, and layout set.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        dtype: KernelDType,
        index_dtype: KernelDType,
        operand_dims: &[usize],
        operand_strides: &[isize],
        start_dims: &[usize],
        start_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        slice_sizes: &[usize],
    ) -> Result<Self> {
        check_index_dtype(index_dtype)?;
        check_gather_value_dtype(dtype)?;
        Ok(Self {
            dtype,
            index_dtype,
            plan: DynamicSlicePlan::compile(
                operand_dims,
                operand_strides,
                start_dims,
                start_strides,
                dest_dims,
                dest_strides,
                slice_sizes,
            )?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn index_dtype(&self) -> KernelDType {
        self.index_dtype
    }

    #[inline]
    pub fn plan(&self) -> &DynamicSlicePlan {
        &self.plan
    }

    /// Execute a fixed-window dynamic slice into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
        starts: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.index_dtype, starts.dtype())?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_dynamic_slice_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::F64 => dispatch_dynamic_slice_index::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::I32 => dispatch_dynamic_slice_index::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::I64 => dispatch_dynamic_slice_index::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::Bool => dispatch_dynamic_slice_index::<bool>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::C32 => dispatch_dynamic_slice_index::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::C64 => dispatch_dynamic_slice_index::<Complex64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: dynamic slice writes values read from a descriptor with
            // the same dtype and already-validated byte representation.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedDynamicUpdateSlicePlan {
    /// Validate and store a dynamic-update-slice plan for one value dtype, index dtype, and layout set.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        dtype: KernelDType,
        index_dtype: KernelDType,
        operand_dims: &[usize],
        operand_strides: &[isize],
        start_dims: &[usize],
        start_strides: &[isize],
        update_dims: &[usize],
        update_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
    ) -> Result<Self> {
        check_index_dtype(index_dtype)?;
        check_gather_value_dtype(dtype)?;
        Ok(Self {
            dtype,
            index_dtype,
            plan: DynamicUpdateSlicePlan::compile(
                operand_dims,
                operand_strides,
                start_dims,
                start_strides,
                update_dims,
                update_strides,
                dest_dims,
                dest_strides,
            )?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn index_dtype(&self) -> KernelDType {
        self.index_dtype
    }

    #[inline]
    pub fn plan(&self) -> &DynamicUpdateSlicePlan {
        &self.plan
    }

    /// Execute a dynamic update slice into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
        update: &ErasedRawStridedRef<'_>,
        starts: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.dtype, update.dtype())?;
        check_dtype(self.index_dtype, starts.dtype())?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_dynamic_update_slice_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::F64 => dispatch_dynamic_update_slice_index::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::I32 => dispatch_dynamic_update_slice_index::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::I64 => dispatch_dynamic_update_slice_index::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::Bool => dispatch_dynamic_update_slice_index::<bool>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::C32 => dispatch_dynamic_update_slice_index::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::C64 => dispatch_dynamic_update_slice_index::<Complex64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: dynamic-update-slice writes either values copied from
            // `operand` or values read from `update`, both with matching dtype.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

impl ErasedScatterPlan {
    /// Validate and store an additive scatter plan for one value dtype, index dtype, and layout set.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        dtype: KernelDType,
        index_dtype: KernelDType,
        operand_dims: &[usize],
        operand_strides: &[isize],
        index_dims: &[usize],
        index_strides: &[isize],
        update_dims: &[usize],
        update_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        spec: ScatterSpec,
    ) -> Result<Self> {
        check_index_dtype(index_dtype)?;
        check_scatter_value_dtype(dtype)?;
        Ok(Self {
            dtype,
            index_dtype,
            plan: ScatterPlan::compile(
                operand_dims,
                operand_strides,
                index_dims,
                index_strides,
                update_dims,
                update_strides,
                dest_dims,
                dest_strides,
                spec,
            )?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn index_dtype(&self) -> KernelDType {
        self.index_dtype
    }

    #[inline]
    pub fn plan(&self) -> &ScatterPlan {
        &self.plan
    }

    /// Execute additive scatter into an erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
        scatter_indices: &ErasedRawStridedRef<'_>,
        updates: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.dtype, updates.dtype())?;
        check_dtype(self.index_dtype, scatter_indices.dtype())?;
        dest.validate_data_if_needed()?;

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_scatter_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
            ),
            KernelDType::F64 => dispatch_scatter_index::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
            ),
            KernelDType::I32 => dispatch_scatter_index::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
            ),
            KernelDType::I64 => dispatch_scatter_index::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
            ),
            KernelDType::C32 => dispatch_scatter_index::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
            ),
            KernelDType::C64 => dispatch_scatter_index::<Complex64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
            ),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        });
        if result.is_ok() {
            // SAFETY: additive scatter copies or adds values with matching,
            // already-validated dtypes.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }
}

fn execute_one_shot_map<T: OneShotScalar>(
    op: ErasedMapOp,
    dest: &mut ErasedRawStridedMut<'_>,
    input: &ErasedRawStridedRef<'_>,
) -> Result<()> {
    if !T::supports_map(op) {
        return Err(StridedError::UnsupportedOp {
            op: op.label(),
            dtype: T::one_shot_dtype_label(),
        });
    }
    validate_one_shot_destination(dest)?;

    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    let input = erased_raw_ref::<T>(input);

    crate::map_view::map_raw_into::<T, T, Identity>(&mut dest, &input, |value| T::map(op, value))
}

fn execute_one_shot_zip<T: OneShotScalar>(
    op: ErasedZipOp,
    dest: &mut ErasedRawStridedMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    rhs: &ErasedRawStridedRef<'_>,
) -> Result<()> {
    if !T::supports_zip(op) {
        return Err(StridedError::UnsupportedOp {
            op: op.label(),
            dtype: T::one_shot_dtype_label(),
        });
    }
    validate_one_shot_destination(dest)?;

    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    let lhs = erased_raw_ref::<T>(lhs);
    let rhs = erased_raw_ref::<T>(rhs);

    crate::map_view::zip_map2_raw_into::<T, T, T, Identity, Identity>(
        &mut dest,
        &lhs,
        &rhs,
        |lhs, rhs| T::zip(op, lhs, rhs),
    )
}

fn validate_one_shot_destination(dest: &ErasedRawStridedMut<'_>) -> Result<()> {
    if crate::fused::is_provably_injective_layout(dest.dims(), dest.strides()) {
        Ok(())
    } else {
        Err(StridedError::NonInjectiveOutputLayout)
    }
}

fn validate_no_overlap(
    dest: &ErasedRawStridedMut<'_>,
    input: &ErasedRawStridedRef<'_>,
    input_index: usize,
) -> Result<()> {
    let dest_start = dest.data().as_ptr() as usize;
    let input_start = input.data().as_ptr() as usize;
    let Some(dest_end) = dest_start.checked_add(dest.data().len()) else {
        return Err(StridedError::OffsetOverflow);
    };
    let Some(input_end) = input_start.checked_add(input.data().len()) else {
        return Err(StridedError::OffsetOverflow);
    };
    if dest_start < input_end && input_start < dest_end {
        Err(StridedError::OverlappingInputOutput { input: input_index })
    } else {
        Ok(())
    }
}

trait OneShotScalar: Copy + crate::MaybeSendSync + 'static {
    fn one_shot_dtype_label() -> &'static str;
    fn supports_map(op: ErasedMapOp) -> bool;
    fn supports_zip(op: ErasedZipOp) -> bool;
    fn map(op: ErasedMapOp, value: Self) -> Self;
    fn zip(op: ErasedZipOp, lhs: Self, rhs: Self) -> Self;
}

macro_rules! impl_real_one_shot_scalar {
    ($ty:ty, $label:literal) => {
        impl OneShotScalar for $ty {
            fn one_shot_dtype_label() -> &'static str {
                $label
            }

            fn supports_map(_op: ErasedMapOp) -> bool {
                true
            }

            fn supports_zip(_op: ErasedZipOp) -> bool {
                true
            }

            #[inline(always)]
            fn map(op: ErasedMapOp, value: Self) -> Self {
                match op {
                    ErasedMapOp::Negate => -value,
                    ErasedMapOp::Conj => value,
                    ErasedMapOp::Abs => value.abs(),
                    ErasedMapOp::Sign => {
                        if value == 0.0 {
                            0.0
                        } else {
                            value.signum()
                        }
                    }
                }
            }

            #[inline(always)]
            fn zip(op: ErasedZipOp, lhs: Self, rhs: Self) -> Self {
                match op {
                    ErasedZipOp::Add => lhs + rhs,
                    ErasedZipOp::Subtract => lhs - rhs,
                    ErasedZipOp::Multiply => lhs * rhs,
                    ErasedZipOp::Divide => lhs / rhs,
                    ErasedZipOp::Remainder => lhs % rhs,
                    ErasedZipOp::Maximum => {
                        if lhs.is_nan() || rhs.is_nan() {
                            <$ty>::NAN
                        } else {
                            lhs.max(rhs)
                        }
                    }
                    ErasedZipOp::Minimum => {
                        if lhs.is_nan() || rhs.is_nan() {
                            <$ty>::NAN
                        } else {
                            lhs.min(rhs)
                        }
                    }
                }
            }
        }
    };
}

macro_rules! impl_integer_one_shot_scalar {
    ($ty:ty, $label:literal) => {
        impl OneShotScalar for $ty {
            fn one_shot_dtype_label() -> &'static str {
                $label
            }

            fn supports_map(_op: ErasedMapOp) -> bool {
                true
            }

            fn supports_zip(op: ErasedZipOp) -> bool {
                !matches!(op, ErasedZipOp::Divide | ErasedZipOp::Remainder)
            }

            #[inline(always)]
            fn map(op: ErasedMapOp, value: Self) -> Self {
                match op {
                    ErasedMapOp::Negate => value.wrapping_neg(),
                    ErasedMapOp::Conj => value,
                    ErasedMapOp::Abs => value.wrapping_abs(),
                    ErasedMapOp::Sign => value.signum(),
                }
            }

            #[inline(always)]
            fn zip(op: ErasedZipOp, lhs: Self, rhs: Self) -> Self {
                match op {
                    ErasedZipOp::Add => lhs.wrapping_add(rhs),
                    ErasedZipOp::Subtract => lhs.wrapping_sub(rhs),
                    ErasedZipOp::Multiply => lhs.wrapping_mul(rhs),
                    ErasedZipOp::Maximum => lhs.max(rhs),
                    ErasedZipOp::Minimum => lhs.min(rhs),
                    ErasedZipOp::Divide | ErasedZipOp::Remainder => {
                        unreachable!("unsupported integer one-shot op")
                    }
                }
            }
        }
    };
}

macro_rules! impl_complex_one_shot_scalar {
    ($ty:ty, $label:literal) => {
        impl OneShotScalar for $ty {
            fn one_shot_dtype_label() -> &'static str {
                $label
            }

            fn supports_map(_op: ErasedMapOp) -> bool {
                true
            }

            fn supports_zip(op: ErasedZipOp) -> bool {
                !matches!(
                    op,
                    ErasedZipOp::Remainder | ErasedZipOp::Maximum | ErasedZipOp::Minimum
                )
            }

            #[inline(always)]
            fn map(op: ErasedMapOp, value: Self) -> Self {
                match op {
                    ErasedMapOp::Negate => -value,
                    ErasedMapOp::Conj => value.conj(),
                    ErasedMapOp::Abs => Self::new(value.norm(), 0.0),
                    ErasedMapOp::Sign => {
                        let norm = value.norm();
                        if norm == 0.0 {
                            value
                        } else {
                            value / norm
                        }
                    }
                }
            }

            #[inline(always)]
            fn zip(op: ErasedZipOp, lhs: Self, rhs: Self) -> Self {
                match op {
                    ErasedZipOp::Add => lhs + rhs,
                    ErasedZipOp::Subtract => lhs - rhs,
                    ErasedZipOp::Multiply => lhs * rhs,
                    ErasedZipOp::Divide => lhs / rhs,
                    ErasedZipOp::Remainder | ErasedZipOp::Maximum | ErasedZipOp::Minimum => {
                        unreachable!("unsupported complex one-shot op")
                    }
                }
            }
        }
    };
}

impl_real_one_shot_scalar!(f32, "f32");
impl_real_one_shot_scalar!(f64, "f64");
impl_integer_one_shot_scalar!(i32, "i32");
impl_integer_one_shot_scalar!(i64, "i64");
impl_complex_one_shot_scalar!(Complex32, "c32");
impl_complex_one_shot_scalar!(Complex64, "c64");

impl OneShotScalar for bool {
    fn one_shot_dtype_label() -> &'static str {
        "bool"
    }

    fn supports_map(op: ErasedMapOp) -> bool {
        matches!(op, ErasedMapOp::Conj)
    }

    fn supports_zip(_op: ErasedZipOp) -> bool {
        false
    }

    fn map(op: ErasedMapOp, value: Self) -> Self {
        match op {
            ErasedMapOp::Conj => value,
            _ => unreachable!("unsupported bool one-shot op"),
        }
    }

    fn zip(_op: ErasedZipOp, _lhs: Self, _rhs: Self) -> Self {
        unreachable!("unsupported bool one-shot op")
    }
}

fn erased_raw_ref<'a, T>(src: &ErasedRawStridedRef<'a>) -> RawStridedRef<'a, T> {
    let data = typed_slice::<T>(src.data());
    unsafe { RawStridedRef::new_unchecked(data, src.dims(), src.strides(), src.offset()) }
}

fn check_dtype(expected: KernelDType, actual: KernelDType) -> Result<()> {
    if actual != expected {
        return Err(StridedError::DTypeMismatch {
            expected: expected.label(),
            actual: actual.label(),
        });
    }
    Ok(())
}

fn check_fused_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::F32
        | KernelDType::F64
        | KernelDType::I32
        | KernelDType::I64
        | KernelDType::Bool
        | KernelDType::C32
        | KernelDType::C64 => Ok(()),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn validate_fused_plan_for_dtype(dtype: KernelDType, plan: &FusedPlan) -> Result<()> {
    match dtype {
        KernelDType::F32 => {
            crate::fused::validate_plan_for_scalar::<f32>(plan, plan.input_count, 1)
        }
        KernelDType::F64 => {
            crate::fused::validate_plan_for_scalar::<f64>(plan, plan.input_count, 1)
        }
        KernelDType::I32 => {
            crate::fused::validate_plan_for_scalar::<i32>(plan, plan.input_count, 1)
        }
        KernelDType::I64 => {
            crate::fused::validate_plan_for_scalar::<i64>(plan, plan.input_count, 1)
        }
        KernelDType::Bool => {
            crate::fused::validate_plan_for_scalar::<bool>(plan, plan.input_count, 1)
        }
        KernelDType::C32 => {
            crate::fused::validate_plan_for_scalar::<Complex32>(plan, plan.input_count, 1)
        }
        KernelDType::C64 => {
            crate::fused::validate_plan_for_scalar::<Complex64>(plan, plan.input_count, 1)
        }
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn check_reduce_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::F32
        | KernelDType::F64
        | KernelDType::I32
        | KernelDType::I64
        | KernelDType::C32
        | KernelDType::C64 => Ok(()),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn check_index_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::I32 | KernelDType::I64 => Ok(()),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn check_gather_value_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::F32
        | KernelDType::F64
        | KernelDType::I32
        | KernelDType::I64
        | KernelDType::Bool
        | KernelDType::C32
        | KernelDType::C64 => Ok(()),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn check_scatter_value_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::F32
        | KernelDType::F64
        | KernelDType::I32
        | KernelDType::I64
        | KernelDType::C32
        | KernelDType::C64 => Ok(()),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn check_static_indexing_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::F32
        | KernelDType::F64
        | KernelDType::I32
        | KernelDType::I64
        | KernelDType::Bool
        | KernelDType::C32
        | KernelDType::C64 => Ok(()),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn validate_scalar_bytes(dtype: KernelDType, bytes: &[u8]) -> Result<()> {
    let element_size = dtype.size_of();
    if bytes.len() != element_size {
        return Err(StridedError::ByteLengthMismatch {
            dtype: dtype.label(),
            byte_len: bytes.len(),
            element_size,
        });
    }
    if dtype.requires_valid_byte_values() {
        if let Some(&value) = bytes.iter().find(|&&value| value > 1) {
            return Err(StridedError::InvalidBoolByte { value });
        }
    }
    Ok(())
}

fn checked_total_len(dims: &[usize]) -> Result<usize> {
    if dims.is_empty() {
        return Ok(1);
    }
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)
}

fn execute_copy<T>(
    plan: &CopyPlan,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    let source_data = typed_slice::<T>(src.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let source = unsafe {
        RawStridedRef::new_unchecked(source_data, src.dims(), src.strides(), src.offset())
    };
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest, &source)
}

fn execute_slice<T>(
    plan: &SlicePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    let operand_data = typed_slice::<T>(operand.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest_ref, &operand_ref)
}

fn execute_reverse<T>(
    plan: &ReversePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    let operand_data = typed_slice::<T>(operand.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest_ref, &operand_ref)
}

fn execute_pad<T>(
    plan: &PadPlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    fill: &[u8],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    let fill = read_unaligned_scalar::<T>(fill);
    let operand_data = typed_slice::<T>(operand.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest_ref, &operand_ref, fill)
}

fn execute_concatenate<T>(
    plan: &ConcatenatePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    inputs: &[ErasedRawStridedRef<'_>],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    if inputs.len() != plan.input_count() {
        return Err(StridedError::RankMismatch(inputs.len(), plan.input_count()));
    }
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.check_dest_layout(&dest_ref)?;

    for (position, input) in inputs.iter().enumerate() {
        let input_data = typed_slice::<T>(input.data());
        let input_ref = unsafe {
            RawStridedRef::new_unchecked(input_data, input.dims(), input.strides(), input.offset())
        };
        plan.check_input_layout(position, &input_ref)?;
        plan.execute_segment(position, &mut dest_ref, &input_ref)?;
    }
    Ok(())
}

fn execute_reduce<T>(
    op: ReduceOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
{
    let source = erased_view::<T>(src);
    let value = if ctx.is_serial() {
        crate::reduce_view::reduce_serial(
            &source,
            |value| value,
            |a, b| reduce_values(op, a, b),
            reduce_identity(op),
        )?
    } else {
        ctx.run(|| {
            crate::reduce(
                &source,
                |value| value,
                |a, b| reduce_values(op, a, b),
                reduce_identity(op),
            )
        })?
    };

    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    unsafe {
        *dest_data.as_mut_ptr().offset(dest_offset) = value;
    }
    Ok(())
}

fn dispatch_reduce<T>(
    op: ReduceOp,
    layout: &ReduceLayout,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
{
    match layout {
        ReduceLayout::Full { .. } => execute_reduce::<T>(op, ctx, dest, src),
        ReduceLayout::Axes {
            src_dims,
            src_strides,
            dest_dims,
            dest_strides,
            axes,
            kept_axes,
            reduce_dims,
            dest_total,
            reduce_total,
        } => execute_reduce_axes::<T>(
            op,
            ctx,
            dest,
            src,
            AxesLayout {
                src_dims,
                src_strides,
                dest_dims,
                dest_strides,
                axes,
                kept_axes,
                reduce_dims,
                dest_total: *dest_total,
                reduce_total: *reduce_total,
            },
        ),
    }
}

fn execute_reduce_axes<T>(
    op: ReduceOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
{
    if layout.kept_axes.is_empty()
        && layout.axes.len() == layout.src_dims.len()
        && layout.dest_total == 1
    {
        return execute_reduce::<T>(op, ctx, dest, src);
    }

    if layout.dest_total == 0 {
        return Ok(());
    }

    if ctx.is_serial() {
        execute_reduce_axes_serial::<T>(op, dest, src, layout)
    } else {
        ctx.run(|| execute_reduce_axes_policy::<T>(op, dest, src, layout))
    }
}

fn execute_reduce_axes_policy<T>(
    op: ReduceOp,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
{
    let source_data = typed_slice::<T>(src.data());
    let dest_offset_base = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    #[cfg(feature = "parallel")]
    {
        let nthreads = crate::threading::parallel_threads_for_len(layout.dest_total);
        if nthreads > 1 {
            return execute_reduce_axes_parallel(
                op,
                dest_offset_base,
                dest_data,
                src.offset(),
                source_data,
                layout,
                nthreads,
            );
        }
    }

    execute_reduce_axes_serial_data(
        op,
        dest_offset_base,
        dest_data,
        src.offset(),
        source_data,
        layout,
    )
}

fn execute_reduce_axes_serial<T>(
    op: ReduceOp,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
{
    let source_data = typed_slice::<T>(src.data());
    let dest_offset_base = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    execute_reduce_axes_serial_data(
        op,
        dest_offset_base,
        dest_data,
        src.offset(),
        source_data,
        layout,
    )
}

fn execute_reduce_axes_serial_data<T>(
    op: ReduceOp,
    dest_offset_base: isize,
    dest_data: &mut [T],
    source_offset_base: isize,
    source_data: &[T],
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
{
    let mut out_idx_storage = CoordScratch::new(layout.dest_dims.len());
    let mut reduce_idx_storage = CoordScratch::new(layout.reduce_dims.len());
    let mut src_idx_storage = CoordScratch::new(layout.src_dims.len());
    let out_idx = out_idx_storage.as_mut_slice();
    let reduce_idx = reduce_idx_storage.as_mut_slice();
    let src_idx = src_idx_storage.as_mut_slice();

    for _ in 0..layout.dest_total {
        src_idx.fill(0);
        for (dest_axis, &src_axis) in layout.kept_axes.iter().enumerate() {
            src_idx[src_axis] = out_idx[dest_axis];
        }

        let mut acc = reduce_identity(op);
        reduce_idx.fill(0);
        for _ in 0..layout.reduce_total {
            for (reduce_axis, &src_axis) in layout.axes.iter().enumerate() {
                src_idx[src_axis] = reduce_idx[reduce_axis];
            }
            let source_offset =
                checked_strided_offset(source_offset_base, layout.src_strides, src_idx)?;
            let value = unsafe { *source_data.as_ptr().offset(source_offset) };
            acc = reduce_values(op, acc, value);
            advance_col_major_index(reduce_idx, layout.reduce_dims);
        }

        let dest_offset = checked_strided_offset(dest_offset_base, layout.dest_strides, out_idx)?;
        unsafe {
            *dest_data.as_mut_ptr().offset(dest_offset) = acc;
        }
        advance_col_major_index(out_idx, layout.dest_dims);
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn execute_reduce_axes_parallel<T>(
    op: ReduceOp,
    dest_offset_base: isize,
    dest_data: &mut [T],
    source_offset_base: isize,
    source_data: &[T],
    layout: AxesLayout<'_>,
    nthreads: usize,
) -> Result<()>
where
    T: ErasedReduceScalar,
{
    let dest_ptr = crate::threading::SendPtr(dest_data.as_mut_ptr());
    let source_ptr = crate::threading::SendPtr(source_data.as_ptr() as *mut T);
    crate::threading::parallel_map_reduce(
        0..layout.dest_total,
        nthreads,
        &|range| {
            let mut out_idx_storage = CoordScratch::new(layout.dest_dims.len());
            let mut reduce_idx_storage = CoordScratch::new(layout.reduce_dims.len());
            let mut src_idx_storage = CoordScratch::new(layout.src_dims.len());
            let out_idx = out_idx_storage.as_mut_slice();
            let reduce_idx = reduce_idx_storage.as_mut_slice();
            let src_idx = src_idx_storage.as_mut_slice();
            fill_col_major_index(range.start, layout.dest_dims, out_idx);
            let dest_ptr = dest_ptr.as_ptr();
            let source_ptr = source_ptr.as_const();

            for _ in range {
                src_idx.fill(0);
                for (dest_axis, &src_axis) in layout.kept_axes.iter().enumerate() {
                    src_idx[src_axis] = out_idx[dest_axis];
                }

                let mut acc = reduce_identity(op);
                reduce_idx.fill(0);
                for _ in 0..layout.reduce_total {
                    for (reduce_axis, &src_axis) in layout.axes.iter().enumerate() {
                        src_idx[src_axis] = reduce_idx[reduce_axis];
                    }
                    let source_offset =
                        checked_strided_offset(source_offset_base, layout.src_strides, src_idx)?;
                    let value = unsafe { *source_ptr.offset(source_offset) };
                    acc = reduce_values(op, acc, value);
                    advance_col_major_index(reduce_idx, layout.reduce_dims);
                }

                let dest_offset =
                    checked_strided_offset(dest_offset_base, layout.dest_strides, out_idx)?;
                unsafe {
                    // SAFETY: axis reduction writes exactly one scalar per
                    // logical output position, and compile rejected
                    // non-injective destination layouts.
                    *dest_ptr.offset(dest_offset) = acc;
                }
                advance_col_major_index(out_idx, layout.dest_dims);
            }
            Ok(())
        },
        &|left, right| left.and(right),
    )
}

#[inline]
fn reduce_identity<T>(op: ReduceOp) -> T
where
    T: One + Zero,
{
    match op {
        ReduceOp::Sum => T::zero(),
        ReduceOp::Product => T::one(),
    }
}

#[inline]
fn reduce_values<T>(op: ReduceOp, a: T, b: T) -> T
where
    T: ErasedReduceScalar,
{
    match op {
        ReduceOp::Sum => T::reduce_sum(a, b),
        ReduceOp::Product => T::reduce_product(a, b),
    }
}

trait ErasedReduceScalar: Copy + One + Zero + crate::MaybeSendSync {
    fn reduce_sum(lhs: Self, rhs: Self) -> Self;
    fn reduce_product(lhs: Self, rhs: Self) -> Self;
}

macro_rules! impl_default_erased_reduce_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ErasedReduceScalar for $ty {
                #[inline(always)]
                fn reduce_sum(lhs: Self, rhs: Self) -> Self {
                    lhs + rhs
                }

                #[inline(always)]
                fn reduce_product(lhs: Self, rhs: Self) -> Self {
                    lhs * rhs
                }
            }
        )*
    };
}

macro_rules! impl_wrapping_erased_reduce_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ErasedReduceScalar for $ty {
                #[inline(always)]
                fn reduce_sum(lhs: Self, rhs: Self) -> Self {
                    lhs.wrapping_add(rhs)
                }

                #[inline(always)]
                fn reduce_product(lhs: Self, rhs: Self) -> Self {
                    lhs.wrapping_mul(rhs)
                }
            }
        )*
    };
}

impl_default_erased_reduce_scalar!(f32, f64, Complex32, Complex64);
impl_wrapping_erased_reduce_scalar!(i32, i64);

fn validate_unique_axes(axes: &[usize], rank: usize) -> Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(StridedError::InvalidAxis { axis, rank });
        }
        if seen[axis] {
            return Err(StridedError::InvalidAxis { axis, rank });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn checked_strided_offset(base: isize, strides: &[isize], index: &[usize]) -> Result<isize> {
    let mut offset = base;
    for (&stride, &coord) in strides.iter().zip(index.iter()) {
        offset = checked_offset_add(offset, stride, coord)?;
    }
    Ok(offset)
}

fn checked_offset_add(base: isize, stride: isize, coord: usize) -> Result<isize> {
    let coord = isize::try_from(coord).map_err(|_| StridedError::OffsetOverflow)?;
    let scaled = stride
        .checked_mul(coord)
        .ok_or(StridedError::OffsetOverflow)?;
    base.checked_add(scaled).ok_or(StridedError::OffsetOverflow)
}

fn advance_col_major_index(index: &mut [usize], shape: &[usize]) {
    for axis in 0..index.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

#[cfg(feature = "parallel")]
fn fill_col_major_index(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    for (axis, coord) in out.iter_mut().enumerate() {
        let dim = shape[axis];
        *coord = linear % dim;
        linear /= dim;
    }
}

struct CoordScratch {
    inline: [usize; RAW_FUSED_RANK_LIMIT],
    heap: Option<Vec<usize>>,
    len: usize,
}

impl CoordScratch {
    fn new(len: usize) -> Self {
        if len <= RAW_FUSED_RANK_LIMIT {
            Self {
                inline: [0; RAW_FUSED_RANK_LIMIT],
                heap: None,
                len,
            }
        } else {
            Self {
                inline: [0; RAW_FUSED_RANK_LIMIT],
                heap: Some(vec![0; len]),
                len,
            }
        }
    }

    fn as_mut_slice(&mut self) -> &mut [usize] {
        match &mut self.heap {
            Some(heap) => heap,
            None => &mut self.inline[..self.len],
        }
    }
}

fn execute_fused<T>(
    plan: &FusedPlan,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    inputs: &[ErasedRawStridedRef<'_>],
) -> Result<()>
where
    T: FusedScalar,
{
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let dest_view =
        unsafe { StridedViewMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };

    match inputs {
        [a] => {
            let input_views = [erased_view::<T>(a)];
            let mut dests = [dest_view];
            execute_fused_views(ctx, &mut dests, &input_views, plan)
        }
        [a, b] => {
            let input_views = [erased_view::<T>(a), erased_view::<T>(b)];
            let mut dests = [dest_view];
            execute_fused_views(ctx, &mut dests, &input_views, plan)
        }
        [a, b, c] => {
            let input_views = [
                erased_view::<T>(a),
                erased_view::<T>(b),
                erased_view::<T>(c),
            ];
            let mut dests = [dest_view];
            execute_fused_views(ctx, &mut dests, &input_views, plan)
        }
        [a, b, c, d] => {
            let input_views = [
                erased_view::<T>(a),
                erased_view::<T>(b),
                erased_view::<T>(c),
                erased_view::<T>(d),
            ];
            let mut dests = [dest_view];
            execute_fused_views(ctx, &mut dests, &input_views, plan)
        }
        _ => Err(StridedError::UnsupportedArity {
            arity: inputs.len(),
            max: ERASED_FUSED_INPUT_LIMIT,
        }),
    }
}

fn dispatch_gather_index<T>(
    plan: &GatherPlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    start_indices: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    match index_dtype {
        KernelDType::I32 => execute_gather::<T, i32>(plan, dest, operand, start_indices),
        KernelDType::I64 => execute_gather::<T, i64>(plan, dest, operand, start_indices),
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_gather<T, I>(
    plan: &GatherPlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    start_indices: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
    I: GatherIndex,
{
    let operand_data = typed_slice::<T>(operand.data());
    let index_data = typed_slice::<I>(start_indices.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let index_ref = unsafe {
        RawStridedRef::new_unchecked(
            index_data,
            start_indices.dims(),
            start_indices.strides(),
            start_indices.offset(),
        )
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest_ref, &operand_ref, &index_ref)
}

fn dispatch_dynamic_slice_index<T>(
    plan: &DynamicSlicePlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    match index_dtype {
        KernelDType::I32 => execute_dynamic_slice::<T, i32>(plan, dest, operand, starts),
        KernelDType::I64 => execute_dynamic_slice::<T, i64>(plan, dest, operand, starts),
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_dynamic_slice<T, I>(
    plan: &DynamicSlicePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
    I: GatherIndex,
{
    let operand_data = typed_slice::<T>(operand.data());
    let start_data = typed_slice::<I>(starts.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let start_ref = unsafe {
        RawStridedRef::new_unchecked(start_data, starts.dims(), starts.strides(), starts.offset())
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest_ref, &operand_ref, &start_ref)
}

fn dispatch_dynamic_update_slice_index<T>(
    plan: &DynamicUpdateSlicePlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    update: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    match index_dtype {
        KernelDType::I32 => {
            execute_dynamic_update_slice::<T, i32>(plan, dest, operand, update, starts)
        }
        KernelDType::I64 => {
            execute_dynamic_update_slice::<T, i64>(plan, dest, operand, update, starts)
        }
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_dynamic_update_slice<T, I>(
    plan: &DynamicUpdateSlicePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    update: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
    I: GatherIndex,
{
    let operand_data = typed_slice::<T>(operand.data());
    let update_data = typed_slice::<T>(update.data());
    let start_data = typed_slice::<I>(starts.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let update_ref = unsafe {
        RawStridedRef::new_unchecked(
            update_data,
            update.dims(),
            update.strides(),
            update.offset(),
        )
    };
    let start_ref = unsafe {
        RawStridedRef::new_unchecked(start_data, starts.dims(), starts.strides(), starts.offset())
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest_ref, &operand_ref, &update_ref, &start_ref)
}

fn dispatch_scatter_index<T>(
    plan: &ScatterPlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    scatter_indices: &ErasedRawStridedRef<'_>,
    updates: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + Add<Output = T> + crate::MaybeSendSync,
{
    match index_dtype {
        KernelDType::I32 => {
            execute_scatter::<T, i32>(plan, dest, operand, scatter_indices, updates)
        }
        KernelDType::I64 => {
            execute_scatter::<T, i64>(plan, dest, operand, scatter_indices, updates)
        }
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_scatter<T, I>(
    plan: &ScatterPlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    scatter_indices: &ErasedRawStridedRef<'_>,
    updates: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + Add<Output = T> + crate::MaybeSendSync,
    I: GatherIndex,
{
    let operand_data = typed_slice::<T>(operand.data());
    let index_data = typed_slice::<I>(scatter_indices.data());
    let update_data = typed_slice::<T>(updates.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let index_ref = unsafe {
        RawStridedRef::new_unchecked(
            index_data,
            scatter_indices.dims(),
            scatter_indices.strides(),
            scatter_indices.offset(),
        )
    };
    let update_ref = unsafe {
        RawStridedRef::new_unchecked(
            update_data,
            updates.dims(),
            updates.strides(),
            updates.offset(),
        )
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest_ref, &operand_ref, &index_ref, &update_ref)
}

fn execute_fused_views<T>(
    ctx: &ExecContext,
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()>
where
    T: FusedScalar,
{
    if ctx.is_serial() {
        crate::fused::fused_elementwise_into_serial(dests, inputs, plan)
    } else {
        ctx.run(|| fused_elementwise_into(dests, inputs, plan))
    }
}

fn erased_view<'a, T>(src: &ErasedRawStridedRef<'a>) -> StridedView<'a, T> {
    let data = typed_slice::<T>(src.data());
    unsafe { StridedView::new_unchecked(data, src.dims(), src.strides(), src.offset()) }
}

fn typed_slice<T>(bytes: &[u8]) -> &[T] {
    if bytes.is_empty() {
        return &[];
    }
    unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr().cast::<T>(),
            bytes.len() / core::mem::size_of::<T>(),
        )
    }
}

fn typed_slice_mut<T>(bytes: &mut [u8]) -> &mut [T] {
    unsafe {
        core::slice::from_raw_parts_mut(
            if bytes.is_empty() {
                core::ptr::NonNull::<T>::dangling().as_ptr()
            } else {
                bytes.as_mut_ptr().cast::<T>()
            },
            bytes.len() / core::mem::size_of::<T>(),
        )
    }
}

fn read_unaligned_scalar<T>(bytes: &[u8]) -> T
where
    T: Copy,
{
    unsafe { core::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) }
}
