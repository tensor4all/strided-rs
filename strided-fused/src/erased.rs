use crate::*;
use num_complex::{Complex32, Complex64};
use strided_basic::execution::{check_dtype, erased_view, validate_uninit_no_overlap};
const ERASED_FUSED_INPUT_LIMIT: usize = 4;

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
        result
    }

    /// Execute a single-output fused plan into fully overwritten uninitialized storage.
    ///
    /// Dtype, shape, destination injectivity, bounds, and input/output overlap
    /// are validated before any shared typed input descriptor is formed or any
    /// destination byte is written. On `Ok(())`, every logical destination
    /// element is initialized. An error leaves the destination untouched; a
    /// panic during execution may leave partial initialization, but the backing
    /// `MaybeUninit` storage remains safe to drop.
    ///
    /// # Errors
    ///
    /// Returns a typed dtype, input-count, shape, bounds, destination
    /// injectivity, unsupported-operation, or input/output-overlap error. All
    /// error-producing validation completes before execution starts.
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
        for (index, input) in inputs.iter().enumerate() {
            validate_uninit_no_overlap(dest, input, index)?;
            if input.dims() != dest.dims() {
                return Err(StridedError::ShapeMismatch(
                    input.dims().to_vec(),
                    dest.dims().to_vec(),
                ));
            }
        }
        let validated = strided_basic::execution::validate_destination_layout_without_alloc(
            dest.dims(),
            dest.strides(),
        )?;

        let run = |dest: &mut ErasedRawStridedUninitMut<'_>| match self.dtype {
            KernelDType::F32 => execute_fused_uninit_ptrs::<f32>(
                &self.plan,
                dest,
                inputs,
                ctx.is_serial(),
                validated,
            ),
            KernelDType::F64 => execute_fused_uninit_ptrs::<f64>(
                &self.plan,
                dest,
                inputs,
                ctx.is_serial(),
                validated,
            ),
            KernelDType::I32 => execute_fused_uninit_ptrs::<i32>(
                &self.plan,
                dest,
                inputs,
                ctx.is_serial(),
                validated,
            ),
            KernelDType::I64 => execute_fused_uninit_ptrs::<i64>(
                &self.plan,
                dest,
                inputs,
                ctx.is_serial(),
                validated,
            ),
            KernelDType::Bool => execute_fused_uninit_ptrs::<bool>(
                &self.plan,
                dest,
                inputs,
                ctx.is_serial(),
                validated,
            ),
            KernelDType::C32 => execute_fused_uninit_ptrs::<Complex32>(
                &self.plan,
                dest,
                inputs,
                ctx.is_serial(),
                validated,
            ),
            KernelDType::C64 => execute_fused_uninit_ptrs::<Complex64>(
                &self.plan,
                dest,
                inputs,
                ctx.is_serial(),
                validated,
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

fn execute_fused<T>(
    plan: &FusedPlan,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedMut<'_>,
    inputs: &[ErasedRawStridedRef<'_>],
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = dest.data_as_mut::<T>()?;
    let dest_view =
        unsafe { StridedViewMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };

    match inputs {
        [a] => {
            let input_views = [erased_view::<T>(a)?];
            let mut dests = [dest_view];
            execute_fused_views(ctx, &mut dests, &input_views, plan)
        }
        [a, b] => {
            let input_views = [erased_view::<T>(a)?, erased_view::<T>(b)?];
            let mut dests = [dest_view];
            execute_fused_views(ctx, &mut dests, &input_views, plan)
        }
        [a, b, c] => {
            let input_views = [
                erased_view::<T>(a)?,
                erased_view::<T>(b)?,
                erased_view::<T>(c)?,
            ];
            let mut dests = [dest_view];
            execute_fused_views(ctx, &mut dests, &input_views, plan)
        }
        [a, b, c, d] => {
            let input_views = [
                erased_view::<T>(a)?,
                erased_view::<T>(b)?,
                erased_view::<T>(c)?,
                erased_view::<T>(d)?,
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

fn execute_fused_uninit<T>(
    plan: &FusedPlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    inputs: &[ErasedRawStridedRef<'_>],
    serial: bool,
    validated: strided_basic::execution::ValidatedDestinationLayout,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    let dims = dest.dims();
    let strides = dest.strides();
    let offset = dest.offset();
    let dest_data = dest.data_as_uninit_mut::<T>()?;
    let mut dest_view = unsafe { StridedViewMut::new_unchecked(dest_data, dims, strides, offset) };
    match inputs {
        [a] => {
            let input_views = [erased_view::<T>(a)?];
            crate::fused::fused_elementwise_into_uninit(
                &mut dest_view,
                &input_views,
                plan,
                serial,
                validated,
            )
        }
        [a, b] => {
            let input_views = [erased_view::<T>(a)?, erased_view::<T>(b)?];
            crate::fused::fused_elementwise_into_uninit(
                &mut dest_view,
                &input_views,
                plan,
                serial,
                validated,
            )
        }
        [a, b, c] => {
            let input_views = [
                erased_view::<T>(a)?,
                erased_view::<T>(b)?,
                erased_view::<T>(c)?,
            ];
            crate::fused::fused_elementwise_into_uninit(
                &mut dest_view,
                &input_views,
                plan,
                serial,
                validated,
            )
        }
        [a, b, c, d] => {
            let input_views = [
                erased_view::<T>(a)?,
                erased_view::<T>(b)?,
                erased_view::<T>(c)?,
                erased_view::<T>(d)?,
            ];
            crate::fused::fused_elementwise_into_uninit(
                &mut dest_view,
                &input_views,
                plan,
                serial,
                validated,
            )
        }
        _ => Err(StridedError::UnsupportedArity {
            arity: inputs.len(),
            max: ERASED_FUSED_INPUT_LIMIT,
        }),
    }
}

fn execute_fused_uninit_ptrs<T>(
    plan: &FusedPlan,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    inputs: &[ErasedRawStridedPtr<'_>],
    serial: bool,
    validated: strided_basic::execution::ValidatedDestinationLayout,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    match inputs {
        [a] => {
            // SAFETY: ErasedFusedPlan::execute_uninit checked every input against the destination before this helper.
            let refs = unsafe { [a.try_as_ref_after_no_overlap()?] };
            execute_fused_uninit::<T>(plan, dest, &refs, serial, validated)
        }
        [a, b] => {
            // SAFETY: ErasedFusedPlan::execute_uninit checked every input against the destination before this helper.
            let refs = unsafe {
                [
                    a.try_as_ref_after_no_overlap()?,
                    b.try_as_ref_after_no_overlap()?,
                ]
            };
            execute_fused_uninit::<T>(plan, dest, &refs, serial, validated)
        }
        [a, b, c] => {
            // SAFETY: ErasedFusedPlan::execute_uninit checked every input against the destination before this helper.
            let refs = unsafe {
                [
                    a.try_as_ref_after_no_overlap()?,
                    b.try_as_ref_after_no_overlap()?,
                    c.try_as_ref_after_no_overlap()?,
                ]
            };
            execute_fused_uninit::<T>(plan, dest, &refs, serial, validated)
        }
        [a, b, c, d] => {
            // SAFETY: ErasedFusedPlan::execute_uninit checked every input against the destination before this helper.
            let refs = unsafe {
                [
                    a.try_as_ref_after_no_overlap()?,
                    b.try_as_ref_after_no_overlap()?,
                    c.try_as_ref_after_no_overlap()?,
                    d.try_as_ref_after_no_overlap()?,
                ]
            };
            execute_fused_uninit::<T>(plan, dest, &refs, serial, validated)
        }
        _ => Err(StridedError::UnsupportedArity {
            arity: inputs.len(),
            max: ERASED_FUSED_INPUT_LIMIT,
        }),
    }
}

fn execute_fused_views<T>(
    ctx: &ExecContext,
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    if ctx.is_serial() {
        crate::fused::fused_elementwise_into_serial(dests, inputs, plan)
    } else {
        ctx.run(|| fused_elementwise_into(dests, inputs, plan))
    }
}
