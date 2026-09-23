use crate::erased_common::*;
use crate::*;
use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
const SERIAL_REDUCE_LANES: usize = 8;

trait ReduceWriter<T> {
    fn offset(&self) -> isize;
    /// # Safety
    /// The pointer may only be used within the validated destination extent.
    unsafe fn ptr(&mut self) -> *mut T;
    fn extent(&self) -> usize;
    /// # Safety
    /// The offset must be an in-bounds logical reduction destination offset.
    unsafe fn write_at(&mut self, offset: isize, value: T) {
        debug_assert!(offset >= 0 && (offset as usize) < self.extent());
        // SAFETY: reduction layout validation proves the logical offset.
        unsafe { self.ptr().offset(offset).write(value) }
    }
}

struct RawReduceWriter<'a, T> {
    ptr: *mut T,
    extent: usize,
    offset: isize,
    _marker: core::marker::PhantomData<&'a mut [MaybeUninit<T>]>,
}

impl<'a, T> ReduceWriter<T> for RawReduceWriter<'a, T> {
    fn offset(&self) -> isize {
        self.offset
    }
    unsafe fn ptr(&mut self) -> *mut T {
        self.ptr
    }
    fn extent(&self) -> usize {
        self.extent
    }
}

/// Dtype-erased wrapper around [`CopyPlan`].
#[derive(Clone, Debug)]
pub struct ErasedCopyPlan {
    dtype: KernelDType,
    plan: CopyPlan,
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
        result
    }

    /// Execute concatenate as a full overwrite of uninitialized output storage.
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
        inputs: &[ErasedRawStridedPtr<'_>],
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        if inputs.len() != self.plan.input_count() {
            return Err(StridedError::RankMismatch(
                inputs.len(),
                self.plan.input_count(),
            ));
        }
        for input in inputs {
            check_dtype(self.dtype, input.dtype())?;
        }
        for (position, input) in inputs.iter().enumerate() {
            validate_uninit_no_overlap(dest, input, position)?;
        }
        for input in inputs {
            // SAFETY: input/output overlap was rejected before forming references.
            unsafe { input.try_as_ref_after_no_overlap() }?;
        }

        ctx.run(|| match self.dtype {
            KernelDType::F32 => execute_concatenate_uninit::<f32>(&self.plan, dest, inputs),
            KernelDType::F64 => execute_concatenate_uninit::<f64>(&self.plan, dest, inputs),
            KernelDType::I32 => execute_concatenate_uninit::<i32>(&self.plan, dest, inputs),
            KernelDType::I64 => execute_concatenate_uninit::<i64>(&self.plan, dest, inputs),
            KernelDType::Bool => execute_concatenate_uninit::<bool>(&self.plan, dest, inputs),
            KernelDType::C32 => execute_concatenate_uninit::<Complex32>(&self.plan, dest, inputs),
            KernelDType::C64 => execute_concatenate_uninit::<Complex64>(&self.plan, dest, inputs),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        })
    }
}

/// Runtime reduction operation for dtype-erased full reductions.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceOp {
    Sum,
    Product,
    /// Sum of same-dtype rounded squares.
    ///
    /// Each input is first multiplied by itself without FMA contraction, then
    /// accumulated under the same association policy as [`Self::Sum`].
    SumSquares,
    /// NaN-propagating maximum for `f32`/`f64`, ordered maximum for
    /// `i32`/`i64`.
    ///
    /// Any NaN operand makes the result the canonical NaN of the dtype. The
    /// identity written for an empty reduction is `-inf` for floats and the
    /// dtype minimum for integers. Complex and `bool` dtypes are rejected.
    /// When the reduced values contain both `-0.0` and `+0.0`, which zero is
    /// returned is unspecified.
    Max,
    /// NaN-propagating minimum for `f32`/`f64`, ordered minimum for
    /// `i32`/`i64`.
    ///
    /// Any NaN operand makes the result the canonical NaN of the dtype. The
    /// identity written for an empty reduction is `+inf` for floats and the
    /// dtype maximum for integers. Complex and `bool` dtypes are rejected.
    /// When the reduced values contain both `-0.0` and `+0.0`, which zero is
    /// returned is unspecified.
    Min,
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
        outer_axes: Vec<ReduceOuterAxis>,
        inner_axes: Vec<ReduceInnerAxis>,
        dest_total: usize,
        reduce_total: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct ReduceOuterAxis {
    extent: usize,
    source_step: isize,
    source_reset: isize,
    dest_step: isize,
    dest_reset: isize,
}
#[derive(Clone, Copy, Debug)]
struct ReduceInnerAxis {
    extent: usize,
    source_step: isize,
    source_reset: isize,
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
    axes: &'a [usize],
    kept_axes: &'a [usize],
    outer_axes: &'a [ReduceOuterAxis],
    inner_axes: &'a [ReduceInnerAxis],
    dest_total: usize,
    reduce_total: usize,
}

impl ErasedReducePlan {
    /// Validate and store a full-reduction plan for one dtype and source layout.
    pub fn compile(
        dtype: KernelDType,
        op: ReduceOp,
        dims: &[usize],
        src_strides: &[isize],
    ) -> Result<Self> {
        check_reduce_op_dtype(dtype, op)?;
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
        check_reduce_op_dtype(dtype, op)?;
        if src_dims.len() != src_strides.len() || dest_dims.len() != dest_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        checked_total_len(src_dims)?;
        check_reduce_layout_offset_arithmetic(src_dims, src_strides)?;
        let dest_total = checked_total_len(dest_dims)?;
        check_reduce_layout_offset_arithmetic(dest_dims, dest_strides)?;
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
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

        let reduce_total = axes
            .iter()
            .try_fold(1usize, |total, &axis| total.checked_mul(src_dims[axis]))
            .ok_or(StridedError::OffsetOverflow)?;
        let outer_axes = compress_reduce_outer_axes(
            kept_axes
                .iter()
                .enumerate()
                .map(|(dest_axis, &src_axis)| {
                    let extent = src_dims[src_axis];
                    Ok(ReduceOuterAxis {
                        extent,
                        source_step: src_strides[src_axis],
                        source_reset: checked_reduce_reset(extent, src_strides[src_axis])?,
                        dest_step: dest_strides[dest_axis],
                        dest_reset: checked_reduce_reset(extent, dest_strides[dest_axis])?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        let inner_axes = compress_reduce_inner_axes(
            axes.iter()
                .map(|&src_axis| {
                    let extent = src_dims[src_axis];
                    Ok(ReduceInnerAxis {
                        extent,
                        source_step: src_strides[src_axis],
                        source_reset: checked_reduce_reset(extent, src_strides[src_axis])?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
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
                outer_axes,
                inner_axes,
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

        let result = match self.dtype {
            KernelDType::F32 => {
                let mut writer = reduce_writer::<f32>(dest)?;
                dispatch_reduce::<f32, _>(self.op, &self.layout, ctx, &mut writer, src)
            }
            KernelDType::F64 => {
                let mut writer = reduce_writer::<f64>(dest)?;
                dispatch_reduce::<f64, _>(self.op, &self.layout, ctx, &mut writer, src)
            }
            KernelDType::I32 => {
                let mut writer = reduce_writer::<i32>(dest)?;
                dispatch_reduce::<i32, _>(self.op, &self.layout, ctx, &mut writer, src)
            }
            KernelDType::I64 => {
                let mut writer = reduce_writer::<i64>(dest)?;
                dispatch_reduce::<i64, _>(self.op, &self.layout, ctx, &mut writer, src)
            }
            KernelDType::C32 => {
                let mut writer = reduce_writer::<Complex32>(dest)?;
                dispatch_reduce::<Complex32, _>(self.op, &self.layout, ctx, &mut writer, src)
            }
            KernelDType::C64 => {
                let mut writer = reduce_writer::<Complex64>(dest)?;
                dispatch_reduce::<Complex64, _>(self.op, &self.layout, ctx, &mut writer, src)
            }
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        };
        result
    }

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
        src: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        validate_uninit_no_overlap(dest, src, 0)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let src = unsafe { src.try_as_ref_after_no_overlap() }?;
        self.layout.check_src_layout(&src)?;
        match &self.layout {
            ReduceLayout::Full { .. } => {
                let total = checked_total_len(dest.dims())?;
                if total != 1 {
                    return Err(StridedError::RankMismatch(total, 1));
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
        macro_rules! run {
            ($ty:ty) => {{
                let mut writer = reduce_uninit_writer::<$ty>(dest)?;
                dispatch_reduce::<$ty, _>(self.op, &self.layout, ctx, &mut writer, &src)
            }};
        }
        match self.dtype {
            KernelDType::F32 => run!(f32),
            KernelDType::F64 => run!(f64),
            KernelDType::I32 => run!(i32),
            KernelDType::I64 => run!(i64),
            KernelDType::C32 => run!(Complex32),
            KernelDType::C64 => run!(Complex64),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        }
    }
}

fn reduce_writer<'a, T>(dest: &'a mut ErasedRawStridedMut<'_>) -> Result<RawReduceWriter<'a, T>>
where
    T: KernelStorageElement,
{
    let offset = dest.offset();
    let data = dest.data_as_mut::<T>()?;
    let ptr = data.as_mut_ptr();
    let extent = data.len();
    Ok(RawReduceWriter {
        ptr,
        extent,
        offset,
        _marker: core::marker::PhantomData,
    })
}

fn reduce_uninit_writer<'a, T>(
    dest: &'a mut ErasedRawStridedUninitMut<'_>,
) -> Result<RawReduceWriter<'a, T>>
where
    T: KernelStorageElement,
{
    let offset = dest.offset();
    let data = dest.data_as_uninit_mut::<T>()?;
    let ptr = data.as_mut_ptr().cast::<T>();
    let extent = data.len();
    Ok(RawReduceWriter {
        ptr,
        extent,
        offset,
        _marker: core::marker::PhantomData,
    })
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

fn check_reduce_op_dtype(dtype: KernelDType, op: ReduceOp) -> Result<()> {
    if op == ReduceOp::SumSquares && !matches!(dtype, KernelDType::F32 | KernelDType::F64) {
        return Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        });
    }
    if matches!(op, ReduceOp::Max | ReduceOp::Min)
        && !matches!(
            dtype,
            KernelDType::F32 | KernelDType::F64 | KernelDType::I32 | KernelDType::I64
        )
    {
        return Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        });
    }
    check_reduce_dtype(dtype)
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
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let source_data = src.data_as::<T>()?;
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
    let source = unsafe {
        RawStridedRef::new_unchecked(source_data, src.dims(), src.strides(), src.offset())
    };
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest, &source)
}

fn execute_concatenate<T>(
    plan: &ConcatenatePlan,
    dest: &mut ErasedRawStridedMut<'_>,
    inputs: &[ErasedRawStridedRef<'_>],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    if inputs.len() != plan.input_count() {
        return Err(StridedError::RankMismatch(inputs.len(), plan.input_count()));
    }
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.check_dest_layout(&dest_ref)?;

    for (position, input) in inputs.iter().enumerate() {
        let input_data = input.data_as::<T>()?;
        let input_ref = unsafe {
            RawStridedRef::new_unchecked(input_data, input.dims(), input.strides(), input.offset())
        };
        plan.check_input_layout(position, &input_ref)?;
        plan.segment_offset(position, dest_offset)?;
    }
    if plan.prefers_whole_plan() {
        let input_refs = inputs
            .iter()
            .map(|input| {
                let input_data = input.data_as::<T>()?;
                Ok(unsafe {
                    RawStridedRef::new_unchecked(
                        input_data,
                        input.dims(),
                        input.strides(),
                        input.offset(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // Splits the concatenated index space across and within segments.
        return plan.execute(&mut dest_ref, &input_refs);
    }
    for (position, input) in inputs.iter().enumerate() {
        let input_data = input.data_as::<T>()?;
        let input_ref = unsafe {
            RawStridedRef::new_unchecked(input_data, input.dims(), input.strides(), input.offset())
        };
        plan.execute_segment(position, &mut dest_ref, &input_ref)?;
    }
    Ok(())
}

fn execute_concatenate_uninit<T>(
    plan: &ConcatenatePlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    inputs: &[ErasedRawStridedPtr<'_>],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
    let mut dest_ref =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.check_dest_layout(&dest_ref)?;

    for (position, input) in inputs.iter().enumerate() {
        // SAFETY: input/output overlap was rejected before forming references.
        let input = unsafe { input.try_as_ref_after_no_overlap() }?;
        let input_data = input.data_as::<T>()?;
        let input_ref = unsafe {
            RawStridedRef::new_unchecked(input_data, input.dims(), input.strides(), input.offset())
        };
        plan.check_input_layout(position, &input_ref)?;
        plan.segment_offset(position, dest_offset)?;
    }
    if plan.prefers_whole_plan() {
        let inputs = inputs
            .iter()
            // SAFETY: input/output overlap was rejected before forming references.
            .map(|input| unsafe { input.try_as_ref_after_no_overlap() })
            .collect::<Result<Vec<_>>>()?;
        let input_refs = inputs
            .iter()
            .map(|input| {
                let input_data = input.data_as::<T>()?;
                Ok(unsafe {
                    RawStridedRef::new_unchecked(
                        input_data,
                        input.dims(),
                        input.strides(),
                        input.offset(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // Splits the concatenated index space across and within segments.
        return plan.execute_uninit(&mut dest_ref, &input_refs);
    }
    for (position, input) in inputs.iter().enumerate() {
        // SAFETY: input/output overlap was rejected before forming references.
        let input = unsafe { input.try_as_ref_after_no_overlap() }?;
        let input_data = input.data_as::<T>()?;
        let input_ref = unsafe {
            RawStridedRef::new_unchecked(input_data, input.dims(), input.strides(), input.offset())
        };
        plan.execute_segment_uninit(position, &mut dest_ref, &input_ref)?;
    }
    Ok(())
}

fn execute_reduce<T, W>(
    op: ReduceOp,
    ctx: &ExecContext,
    dest: &mut W,
    src: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    let use_serial = ctx.is_serial()
        || ctx
            .max_threads_limit()
            .is_some_and(|max_threads| max_threads.get() == 1);
    let value = if use_serial {
        if let Some(value) = reduce_contiguous_serial(op, src) {
            value
        } else {
            let source = erased_view::<T>(src)?;
            crate::reduce_view::reduce_serial(
                &source,
                |value| reduce_map_value(op, value),
                |a, b| reduce_values(op, a, b),
                reduce_identity(op),
            )?
        }
    } else {
        let source = erased_view::<T>(src)?;
        ctx.run(|| {
            crate::reduce(
                &source,
                |value| reduce_map_value(op, value),
                |a, b| reduce_values(op, a, b),
                reduce_identity(op),
            )
        })?
    };

    // SAFETY: validated rank-zero destination layout proves the offset.
    unsafe { dest.write_at(dest.offset(), value) };
    Ok(())
}

fn reduce_contiguous_serial<T>(op: ReduceOp, src: &ErasedRawStridedRef<'_>) -> Option<T>
where
    T: ErasedReduceScalar,
{
    crate::kernel::same_contiguous_layout(src.dims(), &[src.strides()])?;
    let len = checked_total_len(src.dims()).ok()?;
    if len == 0 {
        return Some(reduce_identity(op));
    }

    let source_data = src.data_as::<T>().ok()?;
    let start = usize::try_from(src.offset()).ok()?;
    let end = start.checked_add(len)?;
    let values = source_data.get(start..end)?;
    Some(match op {
        ReduceOp::Sum => T::try_simd_sum(values)
            .unwrap_or_else(|| reduce_contiguous_lanes(values, T::zero(), T::reduce_sum)),
        ReduceOp::Product => T::try_simd_product(values)
            .unwrap_or_else(|| reduce_contiguous_lanes(values, T::one(), T::reduce_product)),
        ReduceOp::SumSquares => T::try_simd_sum_squares(values).unwrap_or_else(|| {
            reduce_contiguous_mapped_lanes(
                values,
                T::zero(),
                |value| T::reduce_product(value, value),
                T::reduce_sum,
            )
        }),
        ReduceOp::Max => reduce_contiguous_lanes(values, T::max_identity(), T::reduce_max),
        ReduceOp::Min => reduce_contiguous_lanes(values, T::min_identity(), T::reduce_min),
    })
}

#[inline]
fn reduce_contiguous_lanes<T>(values: &[T], identity: T, combine: impl Fn(T, T) -> T) -> T
where
    T: Copy,
{
    reduce_contiguous_mapped_lanes(values, identity, |value| value, combine)
}

#[inline]
fn reduce_contiguous_mapped_lanes<T>(
    values: &[T],
    identity: T,
    map: impl Fn(T) -> T,
    combine: impl Fn(T, T) -> T,
) -> T
where
    T: Copy,
{
    let mut lanes = [identity; SERIAL_REDUCE_LANES];
    let mut chunks = values.chunks_exact(SERIAL_REDUCE_LANES);
    for chunk in chunks.by_ref() {
        for lane in 0..SERIAL_REDUCE_LANES {
            lanes[lane] = combine(lanes[lane], map(chunk[lane]));
        }
    }
    for (lane, &value) in chunks.remainder().iter().enumerate() {
        lanes[lane] = combine(lanes[lane], map(value));
    }
    lanes.into_iter().fold(identity, combine)
}

fn dispatch_reduce<T, W>(
    op: ReduceOp,
    layout: &ReduceLayout,
    ctx: &ExecContext,
    dest: &mut W,
    src: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    match layout {
        ReduceLayout::Full { .. } => execute_reduce::<T, W>(op, ctx, dest, src),
        ReduceLayout::Axes {
            src_dims,
            axes,
            kept_axes,
            outer_axes,
            inner_axes,
            dest_total,
            reduce_total,
            ..
        } => execute_reduce_axes::<T, W>(
            op,
            ctx,
            dest,
            src,
            AxesLayout {
                src_dims,
                axes,
                kept_axes,
                outer_axes,
                inner_axes,
                dest_total: *dest_total,
                reduce_total: *reduce_total,
            },
        ),
    }
}

fn execute_reduce_axes<T, W>(
    op: ReduceOp,
    ctx: &ExecContext,
    dest: &mut W,
    src: &ErasedRawStridedRef<'_>,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    if layout.kept_axes.is_empty()
        && layout.axes.len() == layout.src_dims.len()
        && layout.dest_total == 1
    {
        return execute_reduce::<T, W>(op, ctx, dest, src);
    }

    if layout.dest_total == 0 {
        return Ok(());
    }

    if layout.reduce_total == 0 {
        if ctx.is_serial() {
            execute_reduce_axes_identity_serial(op, dest, layout)
        } else {
            ctx.run(|| execute_reduce_axes_identity_policy(op, dest, layout))
        }
    } else if ctx.is_serial() {
        execute_reduce_axes_serial::<T, W>(op, dest, src, layout)
    } else {
        ctx.run(|| execute_reduce_axes_policy::<T, W>(op, dest, src, layout))
    }
}

fn execute_reduce_axes_policy<T, W>(
    op: ReduceOp,
    dest: &mut W,
    src: &ErasedRawStridedRef<'_>,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    let source_data = src.data_as::<T>()?;
    let dest_offset_base = dest.offset();
    #[cfg(feature = "parallel")]
    {
        let nthreads = crate::threading::parallel_threads_for_len(layout.dest_total);
        if nthreads > 1 {
            return execute_reduce_axes_parallel(
                op,
                dest_offset_base,
                dest,
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
        dest,
        src.offset(),
        source_data,
        layout,
    )
}

fn execute_reduce_axes_serial<T, W>(
    op: ReduceOp,
    dest: &mut W,
    src: &ErasedRawStridedRef<'_>,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    let source_data = src.data_as::<T>()?;
    execute_reduce_axes_serial_data(op, dest.offset(), dest, src.offset(), source_data, layout)
}

fn execute_reduce_axes_serial_data<T, W>(
    op: ReduceOp,
    dest_offset_base: isize,
    dest: &mut W,
    source_offset_base: isize,
    source_data: &[T],
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    let mut outer =
        ReduceOuterCursor::decode(0, source_offset_base, dest_offset_base, layout.outer_axes)?;
    // INVARIANT: (1) compile_axes checked signed source/destination spans and
    // every cursor step/reset, including -(extent-1)*stride; (2) raw input and
    // output descriptors validated every reachable offset; (3) execute checked
    // exact plan-layout equality before dispatch.
    let reduce_inner = |inner: &mut ReduceInnerCursor<'_>| {
        let mut acc = reduce_identity(op);
        for value_index in 0..layout.reduce_total {
            // SAFETY: the three-link layout invariant above proves each source
            // cursor offset is within `source_data`.
            let value = unsafe { *source_data.as_ptr().offset(inner.source_offset) };
            acc = reduce_values(op, acc, reduce_map_value(op, value));
            if value_index + 1 < layout.reduce_total {
                inner.advance();
            }
        }
        acc
    };

    if layout.inner_axes.len() <= RAW_FUSED_RANK_LIMIT {
        for output in 0..layout.dest_total {
            let mut inner = ReduceInnerCursor::new(outer.source_offset, layout.inner_axes);
            let acc = reduce_inner(&mut inner);
            // SAFETY: the three-link layout invariant above proves the destination
            // cursor offset is an in-bounds logical output offset.
            unsafe { dest.write_at(outer.dest_offset, acc) };
            if output + 1 < layout.dest_total {
                outer.advance();
            }
        }
    } else {
        let mut inner = ReduceInnerCursor::new(source_offset_base, layout.inner_axes);
        for output in 0..layout.dest_total {
            inner.reset(outer.source_offset);
            let acc = reduce_inner(&mut inner);
            // SAFETY: the three-link layout invariant above proves the destination
            // cursor offset is an in-bounds logical output offset.
            unsafe { dest.write_at(outer.dest_offset, acc) };
            if output + 1 < layout.dest_total {
                outer.advance();
            }
        }
    }
    Ok(())
}

fn execute_reduce_axes_identity_serial<T, W>(
    op: ReduceOp,
    dest: &mut W,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    let mut outer = ReduceOuterCursor::decode(0, 0, dest.offset(), layout.outer_axes)?;
    for output in 0..layout.dest_total {
        // INVARIANT: (1) compile_axes checked the destination span and every
        // destination step/reset, including -(extent-1)*stride; (2) the raw
        // destination descriptor validated every reachable offset; (3) execute
        // checked exact plan-layout equality before dispatch.
        // SAFETY: the three-link layout invariant proves this destination
        // cursor offset is in bounds; the source pointer is intentionally never
        // formed for an empty reduction domain.
        unsafe { dest.write_at(outer.dest_offset, reduce_identity(op)) };
        if output + 1 < layout.dest_total {
            outer.advance();
        }
    }
    Ok(())
}
fn execute_reduce_axes_identity_policy<T, W>(
    op: ReduceOp,
    dest: &mut W,
    layout: AxesLayout<'_>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    #[cfg(feature = "parallel")]
    {
        let nthreads = crate::threading::parallel_threads_for_len(layout.dest_total);
        if nthreads > 1 {
            return execute_reduce_axes_identity_parallel(op, dest, layout, nthreads);
        }
    }
    execute_reduce_axes_identity_serial(op, dest, layout)
}
#[cfg(feature = "parallel")]
fn execute_reduce_axes_identity_parallel<T, W>(
    op: ReduceOp,
    dest: &mut W,
    layout: AxesLayout<'_>,
    nthreads: usize,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    // SAFETY: the validated reduction writer owns the destination allocation.
    let dest_ptr = crate::threading::SendPtr(unsafe { dest.ptr() });
    let dest_offset_base = dest.offset();
    crate::threading::parallel_map_reduce(
        0..layout.dest_total,
        nthreads,
        &|range| {
            let range_end = range.end;
            let mut outer =
                ReduceOuterCursor::decode(range.start, 0, dest_offset_base, layout.outer_axes)?;
            let dest_ptr = dest_ptr.as_ptr();
            for output in range {
                // INVARIANT: (1) compile_axes checked the destination span and
                // every destination step/reset, including
                // -(extent-1)*stride; (2) the raw destination descriptor
                // validated every reachable pointer offset; (3) execute checked
                // exact plan-layout equality before dispatch.
                // SAFETY: the three-link layout invariant proves this
                // destination offset is in bounds; no source pointer is formed.
                unsafe {
                    dest_ptr
                        .offset(outer.dest_offset)
                        .write(reduce_identity(op))
                };
                if output + 1 < range_end {
                    outer.advance();
                }
            }
            Ok(())
        },
        &|left, right| left.and(right),
    )
}
#[cfg(feature = "parallel")]
fn execute_reduce_axes_parallel<T, W>(
    op: ReduceOp,
    dest_offset_base: isize,
    dest: &mut W,
    source_offset_base: isize,
    source_data: &[T],
    layout: AxesLayout<'_>,
    nthreads: usize,
) -> Result<()>
where
    T: ErasedReduceScalar,
    W: ReduceWriter<T>,
{
    // SAFETY: the validated reduction writer owns the destination allocation.
    let dest_ptr = crate::threading::SendPtr(unsafe { dest.ptr() });
    let source_ptr = crate::threading::SendPtr(source_data.as_ptr() as *mut T);
    crate::threading::parallel_map_reduce(
        0..layout.dest_total,
        nthreads,
        &|range| {
            let range_end = range.end;
            let mut outer = ReduceOuterCursor::decode(
                range.start,
                source_offset_base,
                dest_offset_base,
                layout.outer_axes,
            )?;
            let dest_ptr = dest_ptr.as_ptr();
            let source_ptr = source_ptr.as_const();
            // INVARIANT: (1) compile_axes checked signed source/destination
            // spans and every cursor step/reset, including -(extent-1)*stride;
            // (2) raw descriptors validated every reachable pointer offset;
            // (3) execute checked exact plan-layout equality before dispatch.
            let reduce_inner = |inner: &mut ReduceInnerCursor<'_>| {
                let mut acc = reduce_identity(op);
                for value_index in 0..layout.reduce_total {
                    // SAFETY: the three-link layout invariant above proves each
                    // source cursor offset is within the source allocation.
                    let value = unsafe { *source_ptr.offset(inner.source_offset) };
                    acc = reduce_values(op, acc, reduce_map_value(op, value));
                    if value_index + 1 < layout.reduce_total {
                        inner.advance();
                    }
                }
                acc
            };

            if layout.inner_axes.len() <= RAW_FUSED_RANK_LIMIT {
                for output in range {
                    let mut inner = ReduceInnerCursor::new(outer.source_offset, layout.inner_axes);
                    let acc = reduce_inner(&mut inner);
                    // SAFETY: the three-link layout invariant above proves this
                    // destination cursor offset is in bounds.
                    unsafe { dest_ptr.offset(outer.dest_offset).write(acc) };
                    if output + 1 < range_end {
                        outer.advance();
                    }
                }
            } else {
                let mut inner = ReduceInnerCursor::new(outer.source_offset, layout.inner_axes);
                for output in range {
                    inner.reset(outer.source_offset);
                    let acc = reduce_inner(&mut inner);
                    // SAFETY: the three-link layout invariant above proves this
                    // destination cursor offset is in bounds.
                    unsafe { dest_ptr.offset(outer.dest_offset).write(acc) };
                    if output + 1 < range_end {
                        outer.advance();
                    }
                }
            }
            Ok(())
        },
        &|left, right| left.and(right),
    )
}

#[inline]
fn reduce_identity<T>(op: ReduceOp) -> T
where
    T: ErasedReduceScalar,
{
    match op {
        ReduceOp::Sum => T::zero(),
        ReduceOp::Product => T::one(),
        ReduceOp::SumSquares => T::zero(),
        ReduceOp::Max => T::max_identity(),
        ReduceOp::Min => T::min_identity(),
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
        ReduceOp::SumSquares => T::reduce_sum(a, b),
        ReduceOp::Max => T::reduce_max(a, b),
        ReduceOp::Min => T::reduce_min(a, b),
    }
}

#[inline]
fn reduce_map_value<T>(op: ReduceOp, value: T) -> T
where
    T: ErasedReduceScalar,
{
    match op {
        ReduceOp::Sum | ReduceOp::Product | ReduceOp::Max | ReduceOp::Min => value,
        ReduceOp::SumSquares => T::reduce_product(value, value),
    }
}

trait ErasedReduceScalar:
    KernelStorageElement
    + Copy
    + One
    + Zero
    + crate::MaybeSendSync
    + crate::simd::MaybeSimdOps
    + crate::simd::MaybeSimdProduct
    + crate::simd::MaybeSimdSumSquares
{
    fn reduce_sum(lhs: Self, rhs: Self) -> Self;
    fn reduce_product(lhs: Self, rhs: Self) -> Self;
    /// Identity of [`ReduceOp::Max`]; unreachable for dtypes the plan rejects.
    fn max_identity() -> Self;
    /// Identity of [`ReduceOp::Min`]; unreachable for dtypes the plan rejects.
    fn min_identity() -> Self;
    fn reduce_max(lhs: Self, rhs: Self) -> Self;
    fn reduce_min(lhs: Self, rhs: Self) -> Self;
}

macro_rules! impl_float_erased_reduce_scalar {
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

                #[inline(always)]
                fn max_identity() -> Self {
                    <$ty>::NEG_INFINITY
                }

                #[inline(always)]
                fn min_identity() -> Self {
                    <$ty>::INFINITY
                }

                #[inline(always)]
                fn reduce_max(lhs: Self, rhs: Self) -> Self {
                    if lhs.is_nan() || rhs.is_nan() {
                        <$ty>::NAN
                    } else {
                        lhs.max(rhs)
                    }
                }

                #[inline(always)]
                fn reduce_min(lhs: Self, rhs: Self) -> Self {
                    if lhs.is_nan() || rhs.is_nan() {
                        <$ty>::NAN
                    } else {
                        lhs.min(rhs)
                    }
                }
            }
        )*
    };
}

macro_rules! impl_complex_erased_reduce_scalar {
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

                fn max_identity() -> Self {
                    // INVARIANT: check_reduce_op_dtype rejects complex Max at compile time.
                    unreachable!("complex max reduction is rejected at plan compile")
                }

                fn min_identity() -> Self {
                    // INVARIANT: check_reduce_op_dtype rejects complex Min at compile time.
                    unreachable!("complex min reduction is rejected at plan compile")
                }

                fn reduce_max(_lhs: Self, _rhs: Self) -> Self {
                    // INVARIANT: check_reduce_op_dtype rejects complex Max at compile time.
                    unreachable!("complex max reduction is rejected at plan compile")
                }

                fn reduce_min(_lhs: Self, _rhs: Self) -> Self {
                    // INVARIANT: check_reduce_op_dtype rejects complex Min at compile time.
                    unreachable!("complex min reduction is rejected at plan compile")
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

                #[inline(always)]
                fn max_identity() -> Self {
                    <$ty>::MIN
                }

                #[inline(always)]
                fn min_identity() -> Self {
                    <$ty>::MAX
                }

                #[inline(always)]
                fn reduce_max(lhs: Self, rhs: Self) -> Self {
                    lhs.max(rhs)
                }

                #[inline(always)]
                fn reduce_min(lhs: Self, rhs: Self) -> Self {
                    lhs.min(rhs)
                }
            }
        )*
    };
}

impl_float_erased_reduce_scalar!(f32, f64);

impl_complex_erased_reduce_scalar!(Complex32, Complex64);

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

struct ReduceOuterCursor<'a> {
    axes: &'a [ReduceOuterAxis],
    coords: CoordScratch,
    source_offset: isize,
    dest_offset: isize,
}
impl<'a> ReduceOuterCursor<'a> {
    fn decode(
        mut linear: usize,
        source_base: isize,
        dest_base: isize,
        axes: &'a [ReduceOuterAxis],
    ) -> Result<Self> {
        let mut coords = CoordScratch::new(axes.len());
        let mut source_offset = source_base;
        let mut dest_offset = dest_base;
        for (coord, axis) in coords.as_mut_slice().iter_mut().zip(axes) {
            // INVARIANT: a non-empty destination domain implies every outer
            // extent is nonzero before decode; compile-time span checks make
            // these one-time checked additions represent valid layout offsets.
            debug_assert!(axis.extent != 0);
            *coord = linear % axis.extent;
            linear /= axis.extent;
            source_offset = checked_offset_add(source_offset, axis.source_step, *coord)?;
            dest_offset = checked_offset_add(dest_offset, axis.dest_step, *coord)?;
        }
        Ok(Self {
            axes,
            coords,
            source_offset,
            dest_offset,
        })
    }

    #[inline]
    fn advance(&mut self) {
        // INVARIANT: compile_axes checked every signed step and reset delta;
        // descriptor validation plus exact layout equality proves each cursor
        // state is a reachable source/destination offset.
        for (coord, axis) in self.coords.as_mut_slice().iter_mut().zip(self.axes) {
            let next = *coord + 1;
            if next < axis.extent {
                *coord = next;
                self.source_offset += axis.source_step;
                self.dest_offset += axis.dest_step;
                return;
            }
            *coord = 0;
            self.source_offset += axis.source_reset;
            self.dest_offset += axis.dest_reset;
        }
    }
}
struct ReduceInnerCursor<'a> {
    axes: &'a [ReduceInnerAxis],
    coords: CoordScratch,
    source_offset: isize,
}
impl<'a> ReduceInnerCursor<'a> {
    fn new(source_base: isize, axes: &'a [ReduceInnerAxis]) -> Self {
        Self {
            axes,
            coords: CoordScratch::new(axes.len()),
            source_offset: source_base,
        }
    }

    #[inline]
    fn reset(&mut self, source_base: isize) {
        self.coords.as_mut_slice().fill(0);
        self.source_offset = source_base;
    }

    #[inline]
    fn advance(&mut self) {
        // INVARIANT: compile_axes checked every signed source step and reset
        // delta, and the validated descriptor/layout chain proves each value
        // offset is reachable from the current outer source base.
        for (coord, axis) in self.coords.as_mut_slice().iter_mut().zip(self.axes) {
            let next = *coord + 1;
            if next < axis.extent {
                *coord = next;
                self.source_offset += axis.source_step;
                return;
            }
            *coord = 0;
            self.source_offset += axis.source_reset;
        }
    }
}
fn check_reduce_layout_offset_arithmetic(dims: &[usize], strides: &[isize]) -> Result<()> {
    if dims.len() != strides.len() {
        return Err(StridedError::StrideLengthMismatch);
    }
    let mut min_offset = 0isize;
    let mut max_offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides) {
        let last =
            isize::try_from(dim.saturating_sub(1)).map_err(|_| StridedError::OffsetOverflow)?;
        let extent = stride
            .checked_mul(last)
            .ok_or(StridedError::OffsetOverflow)?;
        if extent < 0 {
            min_offset = min_offset
                .checked_add(extent)
                .ok_or(StridedError::OffsetOverflow)?;
        } else {
            max_offset = max_offset
                .checked_add(extent)
                .ok_or(StridedError::OffsetOverflow)?;
        }
    }
    let _ = (min_offset, max_offset);
    Ok(())
}
fn compress_reduce_outer_axes(axes: Vec<ReduceOuterAxis>) -> Result<Vec<ReduceOuterAxis>> {
    let mut compressed: Vec<ReduceOuterAxis> = Vec::with_capacity(axes.len());
    for axis in axes {
        if let Some(previous) = compressed.last_mut() {
            let previous_extent =
                isize::try_from(previous.extent).map_err(|_| StridedError::OffsetOverflow)?;
            let expected_source = previous
                .source_step
                .checked_mul(previous_extent)
                .ok_or(StridedError::OffsetOverflow)?;
            let expected_dest = previous
                .dest_step
                .checked_mul(previous_extent)
                .ok_or(StridedError::OffsetOverflow)?;
            if axis.source_step == expected_source && axis.dest_step == expected_dest {
                let fused_extent = previous
                    .extent
                    .checked_mul(axis.extent)
                    .ok_or(StridedError::OffsetOverflow)?;
                previous.extent = fused_extent;
                previous.source_reset = checked_reduce_reset(fused_extent, previous.source_step)?;
                previous.dest_reset = checked_reduce_reset(fused_extent, previous.dest_step)?;
                continue;
            }
        }
        compressed.push(axis);
    }
    Ok(compressed)
}
fn compress_reduce_inner_axes(axes: Vec<ReduceInnerAxis>) -> Result<Vec<ReduceInnerAxis>> {
    let mut compressed: Vec<ReduceInnerAxis> = Vec::with_capacity(axes.len());
    for axis in axes {
        if let Some(previous) = compressed.last_mut() {
            let previous_extent =
                isize::try_from(previous.extent).map_err(|_| StridedError::OffsetOverflow)?;
            let expected_source = previous
                .source_step
                .checked_mul(previous_extent)
                .ok_or(StridedError::OffsetOverflow)?;
            if axis.source_step == expected_source {
                let fused_extent = previous
                    .extent
                    .checked_mul(axis.extent)
                    .ok_or(StridedError::OffsetOverflow)?;
                previous.extent = fused_extent;
                previous.source_reset = checked_reduce_reset(fused_extent, previous.source_step)?;
                continue;
            }
        }
        compressed.push(axis);
    }
    Ok(compressed)
}
fn checked_reduce_reset(extent: usize, stride: isize) -> Result<isize> {
    if extent == 0 {
        return Ok(0);
    }
    let last = isize::try_from(extent - 1).map_err(|_| StridedError::OffsetOverflow)?;
    stride
        .checked_mul(last)
        .and_then(isize::checked_neg)
        .ok_or(StridedError::OffsetOverflow)
}
fn checked_offset_add(base: isize, stride: isize, coord: usize) -> Result<isize> {
    let coord = isize::try_from(coord).map_err(|_| StridedError::OffsetOverflow)?;
    let scaled = stride
        .checked_mul(coord)
        .ok_or(StridedError::OffsetOverflow)?;
    base.checked_add(scaled).ok_or(StridedError::OffsetOverflow)
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
