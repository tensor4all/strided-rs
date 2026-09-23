use crate::*;
use core::ops::Add;
use num_complex::{Complex32, Complex64};
use strided_basic::execution::check_static_indexing_dtype;
use strided_basic::execution::{check_dtype, validate_uninit_no_overlap};

mod uninit;
pub use uninit::{
    erased_broadcast_mul_into_uninit, erased_clamp_into_uninit, erased_compare_into_uninit,
    erased_map_into_uninit, erased_select_into_uninit, erased_zip_into_uninit,
};
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
/// and `bool` supports only [`ErasedMapOp::Conj`]. Complex absolute value has
/// the real output contract `c32 -> f32` and `c64 -> f64`; all other supported
/// unary operations preserve dtype.
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, overlap,
/// or unsupported dtype/op contracts. Validation completes before any write.
pub fn erased_map_into(
    input_dtype: KernelDType,
    op: ErasedMapOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    input: &ErasedRawStridedPtr<'_>,
) -> Result<()> {
    check_dtype(input_dtype, input.dtype())?;
    check_dtype(map_output_dtype(input_dtype, op)?, dest.dtype())?;
    validate_no_overlap(dest, input, 0)?;
    // SAFETY: input/output overlap was rejected before forming references.
    let input = unsafe { input.try_as_ref_after_no_overlap() }?;

    let result = ctx.run(|| match (input_dtype, op) {
        (KernelDType::C32, ErasedMapOp::Abs) => {
            execute_one_shot_map_with::<f32, Complex32>(dest, &input, |value| value.norm())
        }
        (KernelDType::C64, ErasedMapOp::Abs) => {
            execute_one_shot_map_with::<f64, Complex64>(dest, &input, |value| value.norm())
        }
        (KernelDType::F32, _) => execute_one_shot_map::<f32>(op, dest, &input),
        (KernelDType::F64, _) => execute_one_shot_map::<f64>(op, dest, &input),
        (KernelDType::I32, _) => execute_one_shot_map::<i32>(op, dest, &input),
        (KernelDType::I64, _) => execute_one_shot_map::<i64>(op, dest, &input),
        (KernelDType::Bool, _) => execute_one_shot_map::<bool>(op, dest, &input),
        (KernelDType::C32, _) => execute_one_shot_map::<Complex32>(op, dest, &input),
        (KernelDType::C64, _) => execute_one_shot_map::<Complex64>(op, dest, &input),
        _ => Err(StridedError::UnsupportedDType {
            dtype: input_dtype.label(),
        }),
    });
    result
}

/// Apply one runtime-selected binary operation without compiling a plan.
///
/// The destination must not overlap either input. Real dtypes support every
/// [`ErasedZipOp`]. Signed integers support every operation with wrapping
/// arithmetic and a pre-write zero-divisor check. Complex dtypes support
/// add/subtract/multiply/divide. `bool` has no binary one-shot operations.
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
    lhs: &ErasedRawStridedPtr<'_>,
    rhs: &ErasedRawStridedPtr<'_>,
) -> Result<()> {
    check_dtype(dtype, dest.dtype())?;
    check_dtype(dtype, lhs.dtype())?;
    check_dtype(dtype, rhs.dtype())?;
    validate_no_overlap(dest, lhs, 0)?;
    validate_no_overlap(dest, rhs, 1)?;
    // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
    let lhs = unsafe { lhs.try_as_ref_after_no_overlap() }?;
    // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
    let rhs = unsafe { rhs.try_as_ref_after_no_overlap() }?;

    let result = ctx.run(|| match dtype {
        KernelDType::F32 => execute_one_shot_zip::<f32>(op, dest, &lhs, &rhs),
        KernelDType::F64 => execute_one_shot_zip::<f64>(op, dest, &lhs, &rhs),
        KernelDType::I32 => execute_one_shot_zip::<i32>(op, dest, &lhs, &rhs),
        KernelDType::I64 => execute_one_shot_zip::<i64>(op, dest, &lhs, &rhs),
        KernelDType::Bool => execute_one_shot_zip::<bool>(op, dest, &lhs, &rhs),
        KernelDType::C32 => execute_one_shot_zip::<Complex32>(op, dest, &lhs, &rhs),
        KernelDType::C64 => execute_one_shot_zip::<Complex64>(op, dest, &lhs, &rhs),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    });
    result
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
        result
    }

    /// Execute a static slice as a full overwrite of uninitialized output storage.
    /// On success, every reachable destination slot is fully overwritten;
    /// unreachable holes are neither read nor initialized. Validation errors
    /// are returned before any destination write. A panic during execution
    /// may leave a partially initialized `MaybeUninit` destination, which is
    /// still safely droppable; no readable value is promised for unwritten
    /// reachable slots.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        operand: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        validate_uninit_no_overlap(dest, operand, 0)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let operand = unsafe { operand.try_as_ref_after_no_overlap() }?;

        ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_slice_uninit::<f32>(&self.plan, dest, &operand),
            KernelDType::F64 => execute_slice_uninit::<f64>(&self.plan, dest, &operand),
            KernelDType::I32 => execute_slice_uninit::<i32>(&self.plan, dest, &operand),
            KernelDType::I64 => execute_slice_uninit::<i64>(&self.plan, dest, &operand),
            KernelDType::Bool => execute_slice_uninit::<bool>(&self.plan, dest, &operand),
            KernelDType::C32 => execute_slice_uninit::<Complex32>(&self.plan, dest, &operand),
            KernelDType::C64 => execute_slice_uninit::<Complex64>(&self.plan, dest, &operand),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        })
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
        result
    }

    /// Execute reverse as a full overwrite of uninitialized output storage.
    /// On success, every reachable destination slot is fully overwritten;
    /// unreachable holes are neither read nor initialized. Validation errors
    /// are returned before any destination write. A panic during execution
    /// may leave a partially initialized `MaybeUninit` destination, which is
    /// still safely droppable; no readable value is promised for unwritten
    /// reachable slots.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        operand: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        validate_uninit_no_overlap(dest, operand, 0)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let operand = unsafe { operand.try_as_ref_after_no_overlap() }?;

        ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_reverse_uninit::<f32>(&self.plan, dest, &operand),
            KernelDType::F64 => execute_reverse_uninit::<f64>(&self.plan, dest, &operand),
            KernelDType::I32 => execute_reverse_uninit::<i32>(&self.plan, dest, &operand),
            KernelDType::I64 => execute_reverse_uninit::<i64>(&self.plan, dest, &operand),
            KernelDType::Bool => execute_reverse_uninit::<bool>(&self.plan, dest, &operand),
            KernelDType::C32 => execute_reverse_uninit::<Complex32>(&self.plan, dest, &operand),
            KernelDType::C64 => execute_reverse_uninit::<Complex64>(&self.plan, dest, &operand),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        })
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
        result
    }

    /// Execute pad as a full overwrite of uninitialized output storage.
    /// On success, every reachable destination slot is fully overwritten;
    /// unreachable holes are neither read nor initialized. Validation errors
    /// are returned before any destination write. A panic during execution
    /// may leave a partially initialized `MaybeUninit` destination, which is
    /// still safely droppable; no readable value is promised for unwritten
    /// reachable slots.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        operand: &ErasedRawStridedPtr<'_>,
        fill: &[u8],
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        validate_scalar_bytes(self.dtype, fill)?;
        validate_uninit_no_overlap(dest, operand, 0)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let operand = unsafe { operand.try_as_ref_after_no_overlap() }?;

        ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_pad_uninit::<f32>(&self.plan, dest, &operand, fill),
            KernelDType::F64 => execute_pad_uninit::<f64>(&self.plan, dest, &operand, fill),
            KernelDType::I32 => execute_pad_uninit::<i32>(&self.plan, dest, &operand, fill),
            KernelDType::I64 => execute_pad_uninit::<i64>(&self.plan, dest, &operand, fill),
            KernelDType::Bool => execute_pad_uninit::<bool>(&self.plan, dest, &operand, fill),
            KernelDType::C32 => execute_pad_uninit::<Complex32>(&self.plan, dest, &operand, fill),
            KernelDType::C64 => execute_pad_uninit::<Complex64>(&self.plan, dest, &operand, fill),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        })
    }
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

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_gather_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                &operand,
                &start_indices,
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
        result
    }

    /// Execute gather into a destination whose reachable slots may be
    /// uninitialized. All validation precedes the first destination write.
    /// On success, every reachable destination slot is fully overwritten;
    /// unreachable holes are neither read nor initialized. Validation errors
    /// are returned before any destination write. A panic during execution
    /// may leave a partially initialized `MaybeUninit` destination, which is
    /// still safely droppable; no readable value is promised for unwritten
    /// reachable slots.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        operand: &ErasedRawStridedPtr<'_>,
        start_indices: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.index_dtype, start_indices.dtype())?;
        validate_uninit_no_overlap(dest, operand, 0)?;
        validate_uninit_no_overlap(dest, start_indices, 1)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let operand = &unsafe { operand.try_as_ref_after_no_overlap() }?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let start_indices = &unsafe { start_indices.try_as_ref_after_no_overlap() }?;
        let run = |dest: &mut ErasedRawStridedUninitMut<'_>| match self.dtype {
            KernelDType::F32 => execute_gather_uninit_dispatch::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::F64 => execute_gather_uninit_dispatch::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::I32 => execute_gather_uninit_dispatch::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::I64 => execute_gather_uninit_dispatch::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::Bool => execute_gather_uninit_dispatch::<bool>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::C32 => execute_gather_uninit_dispatch::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            KernelDType::C64 => execute_gather_uninit_dispatch::<Complex64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                start_indices,
            ),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        };
        if ctx.is_serial() {
            run(dest)
        } else {
            ctx.run(|| run(dest))
        }
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

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_dynamic_slice_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                &operand,
                &starts,
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
        result
    }

    /// Execute dynamic slice into a destination whose reachable slots may be
    /// uninitialized.
    /// On success, every reachable destination slot is fully overwritten;
    /// unreachable holes are neither read nor initialized. Validation errors
    /// are returned before any destination write. A panic during execution may
    /// leave reachable slots partially initialized, but the `MaybeUninit`
    /// destination remains safely droppable.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        operand: &ErasedRawStridedPtr<'_>,
        starts: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.index_dtype, starts.dtype())?;
        validate_uninit_no_overlap(dest, operand, 0)?;
        validate_uninit_no_overlap(dest, starts, 1)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let operand = &unsafe { operand.try_as_ref_after_no_overlap() }?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let starts = &unsafe { starts.try_as_ref_after_no_overlap() }?;
        let run = |dest: &mut ErasedRawStridedUninitMut<'_>| match self.dtype {
            KernelDType::F32 => execute_dynamic_slice_uninit_dispatch::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::F64 => execute_dynamic_slice_uninit_dispatch::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::I32 => execute_dynamic_slice_uninit_dispatch::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::I64 => execute_dynamic_slice_uninit_dispatch::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::Bool => execute_dynamic_slice_uninit_dispatch::<bool>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::C32 => execute_dynamic_slice_uninit_dispatch::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            KernelDType::C64 => execute_dynamic_slice_uninit_dispatch::<Complex64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                starts,
            ),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        };
        if ctx.is_serial() {
            run(dest)
        } else {
            ctx.run(|| run(dest))
        }
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

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_dynamic_update_slice_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                &operand,
                &update,
                &starts,
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
        result
    }

    /// On success, the copy phase initializes every reachable destination
    /// slot before the read-modify-write phase. Unreachable holes are neither
    /// read nor initialized. Validation errors before the copy leave the
    /// destination untouched; an error or panic after the copy may leave a
    /// mixture of old and new reachable values, all initialized and safely
    /// droppable.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        operand: &ErasedRawStridedPtr<'_>,
        update: &ErasedRawStridedPtr<'_>,
        starts: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.dtype, update.dtype())?;
        check_dtype(self.index_dtype, starts.dtype())?;
        validate_uninit_no_overlap(dest, operand, 0)?;
        validate_uninit_no_overlap(dest, update, 1)?;
        validate_uninit_no_overlap(dest, starts, 2)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let operand = &unsafe { operand.try_as_ref_after_no_overlap() }?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let update = &unsafe { update.try_as_ref_after_no_overlap() }?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let starts = &unsafe { starts.try_as_ref_after_no_overlap() }?;
        let run = |dest: &mut ErasedRawStridedUninitMut<'_>| match self.dtype {
            KernelDType::F32 => execute_dynamic_update_uninit_dispatch::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::F64 => execute_dynamic_update_uninit_dispatch::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::I32 => execute_dynamic_update_uninit_dispatch::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::I64 => execute_dynamic_update_uninit_dispatch::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::Bool => execute_dynamic_update_uninit_dispatch::<bool>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::C32 => execute_dynamic_update_uninit_dispatch::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                update,
                starts,
            ),
            KernelDType::C64 => execute_dynamic_update_uninit_dispatch::<Complex64>(
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
        };
        if ctx.is_serial() {
            run(dest)
        } else {
            ctx.run(|| run(dest))
        }
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

        let result = ctx.run(|| match self.dtype {
            KernelDType::F32 => dispatch_scatter_index::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                &operand,
                &scatter_indices,
                &updates,
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
        result
    }

    /// On success, the copy phase initializes every reachable destination
    /// slot before the read-modify-write phase. Unreachable holes are neither
    /// read nor initialized. Validation errors before the copy leave the
    /// destination untouched; an error or panic after the copy may leave a
    /// mixture of old and new reachable values, all initialized and safely
    /// droppable.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        operand: &ErasedRawStridedPtr<'_>,
        scatter_indices: &ErasedRawStridedPtr<'_>,
        updates: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.dtype, updates.dtype())?;
        check_dtype(self.index_dtype, scatter_indices.dtype())?;
        validate_uninit_no_overlap(dest, operand, 0)?;
        validate_uninit_no_overlap(dest, scatter_indices, 1)?;
        validate_uninit_no_overlap(dest, updates, 2)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let operand = &unsafe { operand.try_as_ref_after_no_overlap() }?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let scatter_indices = &unsafe { scatter_indices.try_as_ref_after_no_overlap() }?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let updates = &unsafe { updates.try_as_ref_after_no_overlap() }?;
        let run = |dest: &mut ErasedRawStridedUninitMut<'_>| match self.dtype {
            KernelDType::F32 => execute_scatter_uninit_dispatch::<f32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
                add_values::<f32>,
            ),
            KernelDType::F64 => execute_scatter_uninit_dispatch::<f64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
                add_values::<f64>,
            ),
            KernelDType::I32 => execute_scatter_uninit_dispatch::<i32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
                i32::wrapping_add,
            ),
            KernelDType::I64 => execute_scatter_uninit_dispatch::<i64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
                i64::wrapping_add,
            ),
            KernelDType::C32 => execute_scatter_uninit_dispatch::<Complex32>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
                add_values::<Complex32>,
            ),
            KernelDType::C64 => execute_scatter_uninit_dispatch::<Complex64>(
                &self.plan,
                self.index_dtype,
                dest,
                operand,
                scatter_indices,
                updates,
                add_values::<Complex64>,
            ),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        };
        if ctx.is_serial() {
            run(dest)
        } else {
            ctx.run(|| run(dest))
        }
    }
}

fn add_values<T: Add<Output = T>>(lhs: T, rhs: T) -> T {
    lhs + rhs
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
    let validated = strided_basic::execution::validate_destination_layout_without_alloc(
        dest.dims(),
        dest.strides(),
    )?;
    strided_basic::execution::ensure_same_shape(dest.dims(), input.dims())?;
    if dest.dims().contains(&0) {
        return Ok(());
    }

    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    let input = erased_raw_ref::<T>(input)?;

    // SAFETY: matching shapes and this destination layout were validated before specialization/replay.

    unsafe {
        strided_basic::execution::map_raw_into_validated::<T, T, Identity>(
            &mut dest,
            &input,
            |value| T::map(op, value),
            validated,
        )
    }
}

fn execute_one_shot_map_with<D, A>(
    dest: &mut ErasedRawStridedMut<'_>,
    input: &ErasedRawStridedRef<'_>,
    map: impl Fn(A) -> D + crate::MaybeSync,
) -> Result<()>
where
    D: Copy + crate::MaybeSendSync + KernelStorageElement,
    A: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let validated = strided_basic::execution::validate_destination_layout_without_alloc(
        dest.dims(),
        dest.strides(),
    )?;
    strided_basic::execution::ensure_same_shape(dest.dims(), input.dims())?;
    if dest.dims().contains(&0) {
        return Ok(());
    }
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<D>()?;
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    let input = erased_raw_ref::<A>(input)?;
    // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
    unsafe {
        strided_basic::execution::map_raw_into_validated::<D, A, Identity>(
            &mut dest, &input, map, validated,
        )
    }
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
    let validated = strided_basic::execution::validate_destination_layout_without_alloc(
        dest.dims(),
        dest.strides(),
    )?;
    strided_basic::execution::ensure_same_shape(dest.dims(), lhs.dims())?;
    strided_basic::execution::ensure_same_shape(dest.dims(), rhs.dims())?;
    if dest.dims().contains(&0) {
        return Ok(());
    }

    let lhs = erased_raw_ref::<T>(lhs)?;
    let rhs = erased_raw_ref::<T>(rhs)?;
    if matches!(op, ErasedZipOp::Divide | ErasedZipOp::Remainder)
        && T::INTEGER
        && raw_any(&rhs, T::is_zero)?
    {
        return Err(StridedError::IntegerDivisionByZero { op: op.label() });
    }
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    // SAFETY: matching shapes and this destination layout were validated before specialization/replay.
    unsafe {
        strided_basic::execution::zip_map2_raw_into_validated::<T, T, T, Identity, Identity>(
            &mut dest,
            &lhs,
            &rhs,
            |lhs, rhs| T::zip(op, lhs, rhs),
            validated,
        )
    }
}

fn validate_no_overlap(
    dest: &ErasedRawStridedMut<'_>,
    input: &ErasedRawStridedPtr<'_>,
    input_index: usize,
) -> Result<()> {
    if input.overlaps_mut(dest)? {
        Err(StridedError::OverlappingInputOutput { input: input_index })
    } else {
        Ok(())
    }
}

trait OneShotScalar: Copy + crate::MaybeSendSync + KernelStorageElement + 'static {
    const INTEGER: bool = false;
    fn is_zero(_value: Self) -> bool {
        false
    }
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
                        } else if lhs >= rhs {
                            lhs
                        } else {
                            rhs
                        }
                    }
                    ErasedZipOp::Minimum => {
                        if lhs.is_nan() || rhs.is_nan() {
                            <$ty>::NAN
                        } else if lhs <= rhs {
                            lhs
                        } else {
                            rhs
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
            const INTEGER: bool = true;

            fn is_zero(value: Self) -> bool {
                value == 0
            }
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
                    ErasedZipOp::Divide => lhs.wrapping_div(rhs),
                    ErasedZipOp::Remainder => lhs.wrapping_rem(rhs),
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
                        if value.re == 0.0 && value.im == 0.0 {
                            Self::new(0.0, 0.0)
                        } else {
                            // Divide the components by the real modulus: complex
                            // division squares the divisor's components, which
                            // underflows for tiny magnitudes such as 1e-200 in
                            // f64 and yields NaN instead of the unit phase.
                            let norm = value.norm();
                            Self::new(value.re / norm, value.im / norm)
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

fn erased_raw_ref<'a, T: KernelStorageElement>(
    src: &'a ErasedRawStridedRef<'a>,
) -> Result<RawStridedRef<'a, T>> {
    let data = src.data_as::<T>()?;
    Ok(unsafe { RawStridedRef::new_unchecked(data, src.dims(), src.strides(), src.offset()) })
}

fn map_output_dtype(dtype: KernelDType, op: ErasedMapOp) -> Result<KernelDType> {
    match (dtype, op) {
        (KernelDType::C32, ErasedMapOp::Abs) => Ok(KernelDType::F32),
        (KernelDType::C64, ErasedMapOp::Abs) => Ok(KernelDType::F64),
        (KernelDType::Bool, ErasedMapOp::Conj) => Ok(KernelDType::Bool),
        (KernelDType::Bool, _) => Err(StridedError::UnsupportedOp {
            op: op.label(),
            dtype: dtype.label(),
        }),
        _ => Ok(dtype),
    }
}

fn raw_any<T: Copy>(
    src: &RawStridedRef<'_, T>,
    predicate: impl Fn(T) -> bool + Copy,
) -> Result<bool> {
    let total = src
        .dims()
        .iter()
        .try_fold(1usize, |total, &dim| total.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)?;
    if total == 0 {
        return Ok(false);
    }

    let rank = src.dims().len();
    if rank <= RAW_FUSED_RANK_LIMIT {
        let mut coordinates = [0usize; RAW_FUSED_RANK_LIMIT];
        let mut resets = [0isize; RAW_FUSED_RANK_LIMIT];
        raw_any_odometer(
            src,
            predicate,
            &mut coordinates[..rank],
            &mut resets[..rank],
        )
    } else {
        // Ranks above the fused limit keep the same incremental-offset
        // odometer; only the cursor storage moves to the heap.
        let mut coordinates = vec![0usize; rank];
        let mut resets = vec![0isize; rank];
        raw_any_odometer(src, predicate, &mut coordinates, &mut resets)
    }
}

fn raw_any_odometer<T: Copy>(
    src: &RawStridedRef<'_, T>,
    predicate: impl Fn(T) -> bool + Copy,
    coordinates: &mut [usize],
    resets: &mut [isize],
) -> Result<bool> {
    let dims = src.dims();
    let strides = src.strides();
    // INVARIANT: the caller rejected empty extents, so every `dim - 1` is valid.
    for axis in 0..dims.len() {
        let last = isize::try_from(dims[axis] - 1).map_err(|_| StridedError::OffsetOverflow)?;
        resets[axis] = strides[axis]
            .checked_mul(last)
            .and_then(isize::checked_neg)
            .ok_or(StridedError::OffsetOverflow)?;
    }

    let mut offset = src.offset();
    loop {
        // SAFETY: RawStridedRef construction validated every reachable offset.
        if predicate(unsafe { *src.data().as_ptr().offset(offset) }) {
            return Ok(true);
        }

        let mut axis = 0;
        while axis < dims.len() && coordinates[axis] == dims[axis] - 1 {
            axis += 1;
        }
        if axis == dims.len() {
            return Ok(false);
        }
        for reset_axis in 0..axis {
            coordinates[reset_axis] = 0;
            offset = offset
                .checked_add(resets[reset_axis])
                .ok_or(StridedError::OffsetOverflow)?;
        }
        coordinates[axis] = coordinates[axis]
            .checked_add(1)
            .ok_or(StridedError::OffsetOverflow)?;
        offset = offset
            .checked_add(strides[axis])
            .ok_or(StridedError::OffsetOverflow)?;
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

fn execute_slice<T>(
    plan: &SlicePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
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

fn execute_gather_uninit_dispatch<T>(
    plan: &GatherPlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    start_indices: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    match index_dtype {
        KernelDType::I32 => {
            execute_gather_uninit::<T, i32>(plan, index_dtype, dest, operand, start_indices)
        }
        KernelDType::I64 => {
            execute_gather_uninit::<T, i64>(plan, index_dtype, dest, operand, start_indices)
        }
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_gather_uninit<T, I>(
    plan: &GatherPlan,
    _index_dtype: KernelDType,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    start_indices: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let index_data = start_indices.data_as::<I>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
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
    // SAFETY: the erased entry rejected overlap before constructing these descriptors.
    unsafe {
        strided_basic::execution::gather_into_uninit(plan, &mut dest_ref, &operand_ref, &index_ref)
    }
}

fn execute_dynamic_slice_uninit_dispatch<T>(
    plan: &DynamicSlicePlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    match index_dtype {
        KernelDType::I32 => execute_dynamic_slice_uninit::<T, i32>(plan, dest, operand, starts),
        KernelDType::I64 => execute_dynamic_slice_uninit::<T, i64>(plan, dest, operand, starts),
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_dynamic_slice_uninit<T, I>(
    plan: &DynamicSlicePlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let starts_data = starts.data_as::<I>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
    let operand_ref = unsafe {
        RawStridedRef::new_unchecked(
            operand_data,
            operand.dims(),
            operand.strides(),
            operand.offset(),
        )
    };
    let starts_ref = unsafe {
        RawStridedRef::new_unchecked(
            starts_data,
            starts.dims(),
            starts.strides(),
            starts.offset(),
        )
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    // SAFETY: the erased entry rejected overlap before constructing these descriptors.
    unsafe {
        strided_basic::execution::dynamic_slice_into_uninit(
            plan,
            &mut dest_ref,
            &operand_ref,
            &starts_ref,
        )
    }
}

fn execute_dynamic_update_uninit_dispatch<T>(
    plan: &DynamicUpdateSlicePlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    update: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    match index_dtype {
        KernelDType::I32 => {
            execute_dynamic_update_uninit::<T, i32>(plan, dest, operand, update, starts)
        }
        KernelDType::I64 => {
            execute_dynamic_update_uninit::<T, i64>(plan, dest, operand, update, starts)
        }
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_dynamic_update_uninit<T, I>(
    plan: &DynamicUpdateSlicePlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    update: &ErasedRawStridedRef<'_>,
    starts: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let update_data = update.data_as::<T>()?;
    let starts_data = starts.data_as::<I>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
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
    let starts_ref = unsafe {
        RawStridedRef::new_unchecked(
            starts_data,
            starts.dims(),
            starts.strides(),
            starts.offset(),
        )
    };
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    // SAFETY: the erased entry rejected overlap before constructing these descriptors.
    unsafe {
        strided_basic::execution::dynamic_update_into_uninit(
            plan,
            &mut dest_ref,
            &operand_ref,
            &update_ref,
            &starts_ref,
        )
    }
}

fn execute_scatter_uninit_dispatch<T>(
    plan: &ScatterPlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    scatter_indices: &ErasedRawStridedRef<'_>,
    updates: &ErasedRawStridedRef<'_>,
    combine: fn(T, T) -> T,
) -> Result<()>
where
    T: Copy + Add<Output = T> + crate::MaybeSendSync + KernelStorageElement,
{
    match index_dtype {
        KernelDType::I32 => {
            execute_scatter_uninit::<T, i32>(plan, dest, operand, scatter_indices, updates, combine)
        }
        KernelDType::I64 => {
            execute_scatter_uninit::<T, i64>(plan, dest, operand, scatter_indices, updates, combine)
        }
        _ => Err(StridedError::UnsupportedDType {
            dtype: index_dtype.label(),
        }),
    }
}

fn execute_scatter_uninit<T, I>(
    plan: &ScatterPlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    scatter_indices: &ErasedRawStridedRef<'_>,
    updates: &ErasedRawStridedRef<'_>,
    combine: fn(T, T) -> T,
) -> Result<()>
where
    T: Copy + Add<Output = T> + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let indices = scatter_indices;
    let operand_data = operand.data_as::<T>()?;
    let index_data = indices.data_as::<I>()?;
    let update_data = updates.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
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
            indices.dims(),
            indices.strides(),
            indices.offset(),
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
    // SAFETY: the erased entry rejected overlap before constructing these descriptors.
    unsafe {
        strided_basic::execution::scatter_into_uninit(
            plan,
            &mut dest_ref,
            &operand_ref,
            &index_ref,
            &update_ref,
            combine,
        )
    }
}

fn execute_slice_uninit<T>(
    plan: &SlicePlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
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
    plan.execute_uninit(&mut dest_ref, &operand_ref)
}

fn execute_reverse<T>(
    plan: &ReversePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
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

fn execute_reverse_uninit<T>(
    plan: &ReversePlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
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
    plan.execute_uninit(&mut dest_ref, &operand_ref)
}

fn execute_pad<T>(
    plan: &PadPlan,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    fill: &[u8],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let fill = read_unaligned_scalar::<T>(fill);
    let operand_data = operand.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
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

fn execute_pad_uninit<T>(
    plan: &PadPlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    fill: &[u8],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let fill = read_unaligned_scalar::<T>(fill);
    let operand_data = operand.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
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
    plan.execute_uninit(&mut dest_ref, &operand_ref, fill)
}

fn dispatch_gather_index<T>(
    plan: &GatherPlan,
    index_dtype: KernelDType,
    dest: &mut ErasedRawStridedMut<'_>,
    operand: &ErasedRawStridedRef<'_>,
    start_indices: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
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
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let index_data = start_indices.data_as::<I>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
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
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
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
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let start_data = starts.data_as::<I>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
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
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
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
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let update_data = update.data_as::<T>()?;
    let start_data = starts.data_as::<I>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
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
    T: Copy + Add<Output = T> + crate::MaybeSendSync + KernelStorageElement,
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
    T: Copy + Add<Output = T> + crate::MaybeSendSync + KernelStorageElement,
    I: GatherIndex + KernelStorageElement,
{
    let operand_data = operand.data_as::<T>()?;
    let index_data = scatter_indices.data_as::<I>()?;
    let update_data = updates.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
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

fn read_unaligned_scalar<T>(bytes: &[u8]) -> T
where
    T: Copy,
{
    unsafe { core::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) }
}
