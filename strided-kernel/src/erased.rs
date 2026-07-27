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

use core::ops::{Add, Mul};

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use crate::{
    fused_elementwise_into, CopyPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext,
    FusedPlan, FusedScalar, GatherIndex, GatherPlan, GatherSpec, KernelDType, RawStridedMut,
    RawStridedRef, Result, StridedError, StridedView, StridedViewMut,
};

const ERASED_FUSED_INPUT_LIMIT: usize = 4;

/// Dtype-erased wrapper around [`CopyPlan`].
#[derive(Clone, Debug)]
pub struct ErasedCopyPlan {
    dtype: KernelDType,
    plan: CopyPlan,
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
        _ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        self.check_dtype(dest.dtype())?;
        self.check_dtype(src.dtype())?;
        dest.validate_data_if_needed()?;

        let result = match self.dtype {
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
        };
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

/// Dtype-erased full-reduction wrapper.
///
/// This is the erased replay boundary for full-tensor reductions whose result
/// is one scalar. It supports only operations with an unambiguous identity value
/// in the selected dtype.
#[derive(Clone, Debug)]
pub struct ErasedReducePlan {
    dtype: KernelDType,
    op: ReduceOp,
    dims: Vec<usize>,
    src_strides: Vec<isize>,
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
        crate::fused::validate_plan(&plan, plan.input_count, 1)?;
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
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
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

    /// Execute a full reduction into a scalar erased output descriptor.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        if src.dims() != self.dims.as_slice() || src.strides() != self.src_strides.as_slice() {
            return Err(StridedError::PlanLayoutMismatch);
        }
        let dest_len = checked_total_len(dest.dims())?;
        if dest_len != 1 {
            return Err(StridedError::RankMismatch(dest_len, 1));
        }
        dest.validate_data_if_needed()?;

        let result = match self.dtype {
            KernelDType::F32 => execute_reduce::<f32>(self.op, ctx, dest, src),
            KernelDType::F64 => execute_reduce::<f64>(self.op, ctx, dest, src),
            KernelDType::I32 => execute_reduce::<i32>(self.op, ctx, dest, src),
            KernelDType::I64 => execute_reduce::<i64>(self.op, ctx, dest, src),
            KernelDType::C32 => execute_reduce::<Complex32>(self.op, ctx, dest, src),
            KernelDType::C64 => execute_reduce::<Complex64>(self.op, ctx, dest, src),
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
        _ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        operand: &ErasedRawStridedRef<'_>,
        start_indices: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, operand.dtype())?;
        check_dtype(self.index_dtype, start_indices.dtype())?;
        dest.validate_data_if_needed()?;

        let result = match self.dtype {
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
        };
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
        KernelDType::F32 | KernelDType::F64 | KernelDType::C32 | KernelDType::C64 => Ok(()),
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

fn execute_reduce<T>(
    op: ReduceOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + One + Zero + crate::MaybeSendSync,
{
    let source = erased_view::<T>(src);
    let value = if ctx.is_ambient() {
        crate::reduce(
            &source,
            |value| value,
            |a, b| reduce_values(op, a, b),
            reduce_identity(op),
        )?
    } else {
        crate::reduce_view::reduce_serial(
            &source,
            |value| value,
            |a, b| reduce_values(op, a, b),
            reduce_identity(op),
        )?
    };

    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    unsafe {
        *dest_data.as_mut_ptr().offset(dest_offset) = value;
    }
    Ok(())
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
    T: Add<Output = T> + Mul<Output = T>,
{
    match op {
        ReduceOp::Sum => a + b,
        ReduceOp::Product => a * b,
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

fn execute_fused_views<T>(
    ctx: &ExecContext,
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()>
where
    T: FusedScalar,
{
    if ctx.is_ambient() {
        fused_elementwise_into(dests, inputs, plan)
    } else {
        crate::fused::fused_elementwise_into_serial(dests, inputs, plan)
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
