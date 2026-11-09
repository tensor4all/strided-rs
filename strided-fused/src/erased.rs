use crate::*;
use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_basic::execution::{
    check_dtype, ensure_same_shape, erased_view, validate_destination_layout_without_alloc,
    validate_uninit_no_overlap, ValidatedDestinationLayout,
};
const ERASED_FUSED_INPUT_LIMIT: usize = 4;

/// Dtype-erased wrapper around [`FusedPlan`].
///
/// This is the erased replay boundary for unary map and zip-map elementwise
/// families. It supports the same runtime op-code vocabulary as [`FusedPlan`],
/// but only for the scalar dtypes currently implementing [`FusedScalar`]. A
/// plan may name one or more outputs; [`ErasedFusedPlan::execute_outputs`] and
/// [`ErasedFusedPlan::execute_outputs_uninit`] write them all in one replay.
#[derive(Clone, Debug)]
pub struct ErasedFusedPlan {
    dtype: KernelDType,
    plan: FusedPlan,
}

impl ErasedFusedPlan {
    /// Validate and store a fused elementwise plan for one dtype.
    ///
    /// The plan must name at least one output; repeated output value ids are
    /// allowed and write the same value to each named destination.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::UnsupportedArity`] when `plan.input_count` is
    /// outside `1..=4`,
    /// [`StridedError::RankMismatch`] when the plan names no output or an
    /// instruction is given a different operand count than it consumes,
    /// [`StridedError::InvalidAxis`] when an instruction input or a named
    /// output is not a value id defined before it, and
    /// [`StridedError::UnsupportedOp`] when the dtype cannot evaluate a
    /// planned operation.
    pub fn compile(dtype: KernelDType, plan: FusedPlan) -> Result<Self> {
        check_fused_dtype(dtype)?;
        if plan.input_count == 0 || plan.input_count > ERASED_FUSED_INPUT_LIMIT {
            return Err(StridedError::UnsupportedArity {
                arity: plan.input_count,
                max: ERASED_FUSED_INPUT_LIMIT,
            });
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
    ///
    /// Forwarding front end for [`Self::execute_outputs`]; the plan must name
    /// exactly one output.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when the input count or the
    /// output count disagrees with the plan or the plan does not name exactly
    /// one output, [`StridedError::DTypeMismatch`] when a descriptor does not
    /// carry the plan dtype, [`StridedError::ShapeMismatch`] or
    /// [`StridedError::RankMismatch`] when a destination or input does not
    /// have the shape of `dests[0]`, and
    /// [`StridedError::NonInjectiveOutputLayout`] when a destination layout
    /// maps two logical elements onto one offset. Nothing is written unless
    /// every check succeeds.
    pub fn execute(
        &self,
        dest: &mut ErasedRawStridedMut<'_>,
        inputs: &[ErasedRawStridedRef<'_>],
        exec: Exec<'_>,
    ) -> Result<()> {
        self.execute_outputs(std::slice::from_mut(dest), inputs, exec)
    }

    /// Execute a plan into one or more destinations in a single replay.
    ///
    /// `plan.outputs` are written to `dests` in order. Every destination is
    /// validated before any write: dtype, shape against the inputs, and output
    /// injectivity. Overlapping initialized input/output memory is not
    /// supported; the mutable destination slices are exclusive, so only
    /// destination-versus-input overlap is a caller concern.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when `inputs.len()` or
    /// `dests.len()` disagrees with the plan,
    /// [`StridedError::DTypeMismatch`] when a descriptor does not carry the
    /// plan dtype, [`StridedError::ShapeMismatch`] or
    /// [`StridedError::RankMismatch`] when a destination or input does not
    /// have the shape of `dests[0]`, and
    /// [`StridedError::NonInjectiveOutputLayout`] when a destination layout
    /// maps two logical elements onto one offset. Every check runs before any
    /// destination is written.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{
    ///     ErasedRawStridedMut, ErasedRawStridedRef, Exec, KernelDType,
    /// };
    /// use strided_fused::{ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};
    ///
    /// let plan = ErasedFusedPlan::compile(
    ///     KernelDType::F64,
    ///     FusedPlan::new(
    ///         1,
    ///         vec![0, 1],
    ///         vec![FusedInst::new(FusedOp::Negate, vec![0])],
    ///     ),
    /// )
    /// .unwrap();
    /// let input = [1.0_f64, -2.0];
    /// let dims = [2];
    /// let strides = [1];
    /// let inputs = [ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap()];
    /// let mut pass_through = [0.0_f64; 2];
    /// let mut negated = [0.0_f64; 2];
    /// {
    ///     let mut dests = [
    ///         ErasedRawStridedMut::from_slice_mut(&mut pass_through, &dims, &strides, 0).unwrap(),
    ///         ErasedRawStridedMut::from_slice_mut(&mut negated, &dims, &strides, 0).unwrap(),
    ///     ];
    ///     plan.execute_outputs(&mut dests, &inputs, Exec::serial())
    ///         .unwrap();
    /// }
    /// assert_eq!(pass_through, [1.0, -2.0]);
    /// assert_eq!(negated, [-1.0, 2.0]);
    /// ```
    pub fn execute_outputs(
        &self,
        dests: &mut [ErasedRawStridedMut<'_>],
        inputs: &[ErasedRawStridedRef<'_>],
        exec: Exec<'_>,
    ) -> Result<()> {
        if inputs.len() != self.plan.input_count {
            return Err(StridedError::RankMismatch(
                inputs.len(),
                self.plan.input_count,
            ));
        }
        if dests.len() != self.plan.outputs.len() {
            return Err(StridedError::RankMismatch(
                dests.len(),
                self.plan.outputs.len(),
            ));
        }
        for dest in dests.iter() {
            check_dtype(self.dtype, dest.dtype())?;
        }
        for input in inputs {
            check_dtype(self.dtype, input.dtype())?;
        }
        let dims = dests[0].dims();
        for dest in &dests[1..] {
            ensure_same_shape(dims, dest.dims())?;
        }
        for input in inputs {
            ensure_same_shape(dims, input.dims())?;
        }

        match self.dtype {
            KernelDType::F32 => execute_fused_multi::<f32>(&self.plan, dests, inputs, exec),
            KernelDType::F64 => execute_fused_multi::<f64>(&self.plan, dests, inputs, exec),
            KernelDType::I32 => execute_fused_multi::<i32>(&self.plan, dests, inputs, exec),
            KernelDType::I64 => execute_fused_multi::<i64>(&self.plan, dests, inputs, exec),
            KernelDType::Bool => execute_fused_multi::<bool>(&self.plan, dests, inputs, exec),
            KernelDType::C32 => execute_fused_multi::<Complex32>(&self.plan, dests, inputs, exec),
            KernelDType::C64 => execute_fused_multi::<Complex64>(&self.plan, dests, inputs, exec),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        }
    }

    /// Execute a single-output fused plan into fully overwritten uninitialized
    /// storage.
    ///
    /// Forwarding front end for [`Self::execute_outputs_uninit`]; the plan must
    /// name exactly one output.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when the input count or the
    /// output count disagrees with the plan or the plan does not name exactly
    /// one output, [`StridedError::DTypeMismatch`] when a descriptor does not
    /// carry the plan dtype, [`StridedError::ShapeMismatch`] or
    /// [`StridedError::RankMismatch`] when a destination or input does not
    /// have the shape of `dests[0]`,
    /// [`StridedError::NonInjectiveOutputLayout`] when a destination layout
    /// maps two logical elements onto one offset,
    /// [`StridedError::OverlappingInputOutput`] when an input allocation
    /// overlaps a destination, and [`StridedError::InvalidBoolByte`] when a
    /// `bool` input holds a byte other than 0 or 1. Every rejection is
    /// returned before any destination is written.
    pub fn execute_uninit(
        &self,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        inputs: &[ErasedRawStridedPtr<'_>],
        exec: Exec<'_>,
    ) -> Result<()> {
        self.execute_outputs_uninit(std::slice::from_mut(dest), inputs, exec)
    }

    /// Execute a plan into one or more fully overwritten uninitialized
    /// destinations in a single replay.
    ///
    /// Dtype, shape, destination injectivity, bounds, and input/output overlap
    /// are validated for every destination before any shared typed input
    /// descriptor is formed or any destination byte is written. On `Ok(())`,
    /// every logical destination element is initialized. An error leaves every
    /// destination untouched; a panic during execution may leave partial
    /// initialization, but the backing `MaybeUninit` storage remains safe to
    /// drop.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when `inputs.len()` or
    /// `dests.len()` disagrees with the plan,
    /// [`StridedError::DTypeMismatch`] when a descriptor does not carry the
    /// plan dtype, [`StridedError::ShapeMismatch`] or
    /// [`StridedError::RankMismatch`] when a destination or input does not
    /// have the shape of `dests[0]`,
    /// [`StridedError::NonInjectiveOutputLayout`] when a destination layout
    /// maps two logical elements onto one offset,
    /// [`StridedError::OverlappingInputOutput`] when an input allocation
    /// overlaps a destination, and [`StridedError::InvalidBoolByte`] when a
    /// `bool` input holds a byte other than 0 or 1. All error-producing
    /// validation completes before execution starts.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    /// use strided_basic::{
    ///     ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    ///     Exec, KernelDType,
    /// };
    /// use strided_fused::{ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};
    ///
    /// let plan = ErasedFusedPlan::compile(
    ///     KernelDType::F32,
    ///     FusedPlan::new(
    ///         1,
    ///         vec![0, 1],
    ///         vec![FusedInst::new(FusedOp::Negate, vec![0])],
    ///     ),
    /// )
    /// .unwrap();
    /// let input = [1.0_f32, -2.0];
    /// let dims = [2];
    /// let strides = [1];
    /// let input = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    /// let pointers = [ErasedRawStridedPtr::from_ref(&input)];
    /// let mut pass_through = [MaybeUninit::<f32>::uninit(); 2];
    /// let mut negated = [MaybeUninit::<f32>::uninit(); 2];
    /// {
    ///     let mut dests = [
    ///         ErasedRawStridedUninitMut::from_uninit_slice(
    ///             &mut pass_through,
    ///             &dims,
    ///             &strides,
    ///             0,
    ///         )
    ///         .unwrap(),
    ///         ErasedRawStridedUninitMut::from_uninit_slice(&mut negated, &dims, &strides, 0).unwrap(),
    ///     ];
    ///     plan.execute_outputs_uninit(&mut dests, &pointers, Exec::serial())
    ///         .unwrap();
    /// }
    /// // SAFETY: `execute_outputs_uninit` overwrites every reachable element.
    /// assert_eq!(unsafe { pass_through[1].assume_init() }, -2.0);
    /// // SAFETY: as above.
    /// assert_eq!(unsafe { negated[1].assume_init() }, 2.0);
    /// ```
    pub fn execute_outputs_uninit(
        &self,
        dests: &mut [ErasedRawStridedUninitMut<'_>],
        inputs: &[ErasedRawStridedPtr<'_>],
        exec: Exec<'_>,
    ) -> Result<()> {
        if inputs.len() != self.plan.input_count {
            return Err(StridedError::RankMismatch(
                inputs.len(),
                self.plan.input_count,
            ));
        }
        if dests.len() != self.plan.outputs.len() {
            return Err(StridedError::RankMismatch(
                dests.len(),
                self.plan.outputs.len(),
            ));
        }
        for dest in dests.iter() {
            check_dtype(self.dtype, dest.dtype())?;
        }
        for input in inputs {
            check_dtype(self.dtype, input.dtype())?;
        }
        let dims = dests[0].dims();
        for dest in &dests[1..] {
            ensure_same_shape(dims, dest.dims())?;
        }
        for input in inputs {
            ensure_same_shape(dims, input.dims())?;
        }
        for (index, input) in inputs.iter().enumerate() {
            for dest in dests.iter() {
                validate_uninit_no_overlap(dest, input, index)?;
            }
        }
        // Store the validation tokens inline: the destination-count bound is
        // the five-stream operand bound, so two outputs plus one input allocate
        // nothing. Above the bound the vector spills by contract.
        let mut validated: strided_basic::OperandVec<ValidatedDestinationLayout> =
            strided_basic::OperandVec::with_capacity(dests.len());
        for dest in dests.iter() {
            validated.push(validate_destination_layout_without_alloc(
                dest.dims(),
                dest.strides(),
            )?);
        }
        self.dispatch_uninit(dests, inputs, validated.as_slice(), exec)
    }

    fn dispatch_uninit(
        &self,
        dests: &mut [ErasedRawStridedUninitMut<'_>],
        inputs: &[ErasedRawStridedPtr<'_>],
        validated: &[ValidatedDestinationLayout],
        exec: Exec<'_>,
    ) -> Result<()> {
        match self.dtype {
            KernelDType::F32 => {
                execute_fused_uninit_multi::<f32>(&self.plan, dests, inputs, validated, exec)
            }
            KernelDType::F64 => {
                execute_fused_uninit_multi::<f64>(&self.plan, dests, inputs, validated, exec)
            }
            KernelDType::I32 => {
                execute_fused_uninit_multi::<i32>(&self.plan, dests, inputs, validated, exec)
            }
            KernelDType::I64 => {
                execute_fused_uninit_multi::<i64>(&self.plan, dests, inputs, validated, exec)
            }
            KernelDType::Bool => {
                execute_fused_uninit_multi::<bool>(&self.plan, dests, inputs, validated, exec)
            }
            KernelDType::C32 => {
                execute_fused_uninit_multi::<Complex32>(&self.plan, dests, inputs, validated, exec)
            }
            KernelDType::C64 => {
                execute_fused_uninit_multi::<Complex64>(&self.plan, dests, inputs, validated, exec)
            }
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
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
    let output_count = plan.outputs.len();
    match dtype {
        KernelDType::F32 => {
            crate::fused::validate_plan_for_scalar::<f32>(plan, plan.input_count, output_count)
        }
        KernelDType::F64 => {
            crate::fused::validate_plan_for_scalar::<f64>(plan, plan.input_count, output_count)
        }
        KernelDType::I32 => {
            crate::fused::validate_plan_for_scalar::<i32>(plan, plan.input_count, output_count)
        }
        KernelDType::I64 => {
            crate::fused::validate_plan_for_scalar::<i64>(plan, plan.input_count, output_count)
        }
        KernelDType::Bool => {
            crate::fused::validate_plan_for_scalar::<bool>(plan, plan.input_count, output_count)
        }
        KernelDType::C32 => crate::fused::validate_plan_for_scalar::<Complex32>(
            plan,
            plan.input_count,
            output_count,
        ),
        KernelDType::C64 => crate::fused::validate_plan_for_scalar::<Complex64>(
            plan,
            plan.input_count,
            output_count,
        ),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

fn execute_fused_multi<T>(
    plan: &FusedPlan,
    dests: &mut [ErasedRawStridedMut<'_>],
    inputs: &[ErasedRawStridedRef<'_>],
    exec: Exec<'_>,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    // The single-output case uses stack storage so the hot path allocates
    // nothing; several destinations stay inline up to the operand bound and
    // spill above it.
    if dests.len() == 1 {
        let dims = dests[0].dims();
        let strides = dests[0].strides();
        let offset = dests[0].offset();
        let data = dests[0].data_as_mut::<T>()?;
        let mut dest_views =
            // SAFETY: `StridedViewMut::new_unchecked` (strided-view/src/view.rs:957) needs every
            // index combination in bounds. This descriptor checked exactly these dims, strides
            // and offset against its element count when it was built (`validate_bounds`,
            // strided-view/src/view.rs:187), and `data_as_mut` (strided-view/src/raw.rs:710)
            // returned that count after checking the dtype. The `&mut` slice carries exclusivity.
            [unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) }];
        return execute_fused_inputs(&mut dest_views, inputs, plan, exec);
    }

    if dests.len() <= strided_basic::OPERAND_CAP {
        let mut storage =
            [const { MaybeUninit::<StridedViewMut<'_, T>>::uninit() }; strided_basic::OPERAND_CAP];
        // SAFETY: `InitializedViewsMut::new` (fused.rs:38) needs `OPERAND_CAP` writable slots
        // that outlive the guard: `storage` is an `OPERAND_CAP` array declared before `views`, so
        // it is dropped after it, and this branch holds only when `dests.len() <= OPERAND_CAP`
        // (erased.rs:448), which bounds the pushes.
        let mut views = unsafe { crate::fused::InitializedViewsMut::new(storage.as_mut_ptr()) };
        for dest in dests.iter_mut() {
            let dims = dest.dims();
            let strides = dest.strides();
            let offset = dest.offset();
            let data = dest.data_as_mut::<T>()?;
            // SAFETY: as for the single-destination view above: this descriptor's
            // construction-time `validate_bounds` covers these dims, strides and offset.
            views.push(unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) });
        }
        return execute_fused_inputs(views.as_mut_slice(), inputs, plan, exec);
    }

    let mut dest_views = Vec::with_capacity(dests.len());
    for dest in dests.iter_mut() {
        let dims = dest.dims();
        let strides = dest.strides();
        let offset = dest.offset();
        let data = dest.data_as_mut::<T>()?;
        // SAFETY: as for the single-destination view above: this descriptor's construction-time
        // `validate_bounds` covers these dims, strides and offset.
        dest_views.push(unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) });
    }
    execute_fused_inputs(&mut dest_views, inputs, plan, exec)
}

fn execute_fused_inputs<T>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[ErasedRawStridedRef<'_>],
    plan: &FusedPlan,
    exec: Exec<'_>,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    match inputs {
        [a] => {
            let views = [erased_view::<T>(a)?];
            execute_fused_views(dests, &views, plan, exec)
        }
        [a, b] => {
            let views = [erased_view::<T>(a)?, erased_view::<T>(b)?];
            execute_fused_views(dests, &views, plan, exec)
        }
        [a, b, c] => {
            let views = [
                erased_view::<T>(a)?,
                erased_view::<T>(b)?,
                erased_view::<T>(c)?,
            ];
            execute_fused_views(dests, &views, plan, exec)
        }
        [a, b, c, d] => {
            let views = [
                erased_view::<T>(a)?,
                erased_view::<T>(b)?,
                erased_view::<T>(c)?,
                erased_view::<T>(d)?,
            ];
            execute_fused_views(dests, &views, plan, exec)
        }
        _ => Err(StridedError::UnsupportedArity {
            arity: inputs.len(),
            max: ERASED_FUSED_INPUT_LIMIT,
        }),
    }
}

fn execute_fused_uninit_multi<T>(
    plan: &FusedPlan,
    dests: &mut [ErasedRawStridedUninitMut<'_>],
    inputs: &[ErasedRawStridedPtr<'_>],
    validated: &[ValidatedDestinationLayout],
    exec: Exec<'_>,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    if dests.len() == 1 {
        let dims = dests[0].dims();
        let strides = dests[0].strides();
        let offset = dests[0].offset();
        let data = dests[0].data_as_uninit_mut::<T>()?;
        let mut dest_views =
            // SAFETY: `StridedViewMut::new_unchecked` (strided-view/src/view.rs:957) needs every
            // index combination in bounds. This descriptor checked exactly these dims, strides
            // and offset against its element count when it was built (`validate_bounds`,
            // strided-view/src/view.rs:187), and `data_as_uninit_mut`
            // (strided-view/src/raw.rs:882) returned that count after checking the dtype. The
            // `&mut` slice carries exclusivity.
            [unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) }];
        return execute_fused_uninit_inputs(plan, &mut dest_views, inputs, validated, exec);
    }

    if dests.len() <= strided_basic::OPERAND_CAP {
        let mut storage = [const { MaybeUninit::<StridedViewMut<'_, MaybeUninit<T>>>::uninit() };
            strided_basic::OPERAND_CAP];
        // SAFETY: `InitializedViewsMut::new` (fused.rs:38) needs `OPERAND_CAP` writable slots
        // that outlive the guard: `storage` is an `OPERAND_CAP` array declared before `views`, so
        // it is dropped after it, and this branch holds only when `dests.len() <= OPERAND_CAP`
        // (erased.rs:549), which bounds the pushes.
        let mut views = unsafe { crate::fused::InitializedViewsMut::new(storage.as_mut_ptr()) };
        for dest in dests.iter_mut() {
            let dims = dest.dims();
            let strides = dest.strides();
            let offset = dest.offset();
            let data = dest.data_as_uninit_mut::<T>()?;
            // SAFETY: as for the single-destination view above: this descriptor's
            // construction-time `validate_bounds` covers these dims, strides and offset.
            views.push(unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) });
        }
        return execute_fused_uninit_inputs(plan, views.as_mut_slice(), inputs, validated, exec);
    }

    let mut dest_views = Vec::with_capacity(dests.len());
    for dest in dests.iter_mut() {
        let dims = dest.dims();
        let strides = dest.strides();
        let offset = dest.offset();
        let data = dest.data_as_uninit_mut::<T>()?;
        // SAFETY: as for the single-destination view above: this descriptor's construction-time
        // `validate_bounds` covers these dims, strides and offset.
        dest_views.push(unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) });
    }
    execute_fused_uninit_inputs(plan, &mut dest_views, inputs, validated, exec)
}

fn execute_fused_uninit_inputs<T>(
    plan: &FusedPlan,
    dests: &mut [StridedViewMut<'_, MaybeUninit<T>>],
    inputs: &[ErasedRawStridedPtr<'_>],
    validated: &[ValidatedDestinationLayout],
    exec: Exec<'_>,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    match inputs {
        // SAFETY: `ErasedFusedPlan::execute_outputs_uninit` checked every input
        // against every destination before this helper.
        [a] => {
            // SAFETY: `try_as_ref_after_no_overlap` (strided-view/src/raw.rs:288) needs no
            // mutable allocation overlapping `a`, its allocation contract still in force
            // (strided-view/src/raw.rs:204), and no conflicting access while the result is used.
            // This helper's only caller is `execute_fused_uninit_multi`, reached only from
            // `ErasedFusedPlan::execute_outputs_uninit`, which checked every input against every
            // destination with `validate_uninit_no_overlap` (erased.rs:319) before dispatching.
            // The destinations stay borrowed `&mut` for the call.
            let refs = [unsafe { a.try_as_ref_after_no_overlap()? }];
            let views = [erased_view::<T>(&refs[0])?];
            crate::fused::fused_elementwise_into_uninit_multi(dests, &views, plan, validated, exec)
        }
        [a, b] => {
            let refs = [
                // SAFETY: `try_as_ref_after_no_overlap` (strided-view/src/raw.rs:288) needs no
                // mutable allocation overlapping `a`, its allocation contract still in force
                // (strided-view/src/raw.rs:204), and no conflicting access while the result is
                // used. This helper's only caller is `execute_fused_uninit_multi`, reached only
                // from `ErasedFusedPlan::execute_outputs_uninit`, which checked every input
                // against every destination with `validate_uninit_no_overlap` (erased.rs:319)
                // before dispatching. The destinations stay borrowed `&mut` for the call.
                unsafe { a.try_as_ref_after_no_overlap()? },
                // SAFETY: as for `a`: the same overlap check covered `b`.
                unsafe { b.try_as_ref_after_no_overlap()? },
            ];
            let views = [erased_view::<T>(&refs[0])?, erased_view::<T>(&refs[1])?];
            crate::fused::fused_elementwise_into_uninit_multi(dests, &views, plan, validated, exec)
        }
        [a, b, c] => {
            let refs = [
                // SAFETY: `try_as_ref_after_no_overlap` (strided-view/src/raw.rs:288) needs no
                // mutable allocation overlapping `a`, its allocation contract still in force
                // (strided-view/src/raw.rs:204), and no conflicting access while the result is
                // used. This helper's only caller is `execute_fused_uninit_multi`, reached only
                // from `ErasedFusedPlan::execute_outputs_uninit`, which checked every input
                // against every destination with `validate_uninit_no_overlap` (erased.rs:319)
                // before dispatching. The destinations stay borrowed `&mut` for the call.
                unsafe { a.try_as_ref_after_no_overlap()? },
                // SAFETY: as for `a`: the same overlap check covered `b`.
                unsafe { b.try_as_ref_after_no_overlap()? },
                // SAFETY: as for `a`: the same overlap check covered `c`.
                unsafe { c.try_as_ref_after_no_overlap()? },
            ];
            let views = [
                erased_view::<T>(&refs[0])?,
                erased_view::<T>(&refs[1])?,
                erased_view::<T>(&refs[2])?,
            ];
            crate::fused::fused_elementwise_into_uninit_multi(dests, &views, plan, validated, exec)
        }
        [a, b, c, d] => {
            let refs = [
                // SAFETY: `try_as_ref_after_no_overlap` (strided-view/src/raw.rs:288) needs no
                // mutable allocation overlapping `a`, its allocation contract still in force
                // (strided-view/src/raw.rs:204), and no conflicting access while the result is
                // used. This helper's only caller is `execute_fused_uninit_multi`, reached only
                // from `ErasedFusedPlan::execute_outputs_uninit`, which checked every input
                // against every destination with `validate_uninit_no_overlap` (erased.rs:319)
                // before dispatching. The destinations stay borrowed `&mut` for the call.
                unsafe { a.try_as_ref_after_no_overlap()? },
                // SAFETY: as for `a`: the same overlap check covered `b`.
                unsafe { b.try_as_ref_after_no_overlap()? },
                // SAFETY: as for `a`: the same overlap check covered `c`.
                unsafe { c.try_as_ref_after_no_overlap()? },
                // SAFETY: as for `a`: the same overlap check covered `d`.
                unsafe { d.try_as_ref_after_no_overlap()? },
            ];
            let views = [
                erased_view::<T>(&refs[0])?,
                erased_view::<T>(&refs[1])?,
                erased_view::<T>(&refs[2])?,
                erased_view::<T>(&refs[3])?,
            ];
            crate::fused::fused_elementwise_into_uninit_multi(dests, &views, plan, validated, exec)
        }
        _ => Err(StridedError::UnsupportedArity {
            arity: inputs.len(),
            max: ERASED_FUSED_INPUT_LIMIT,
        }),
    }
}

fn execute_fused_views<T>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
    exec: Exec<'_>,
) -> Result<()>
where
    T: FusedScalar + KernelStorageElement,
{
    fused_elementwise_into(dests, inputs, plan, exec)
}
