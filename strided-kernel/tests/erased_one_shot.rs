use num_complex::{Complex32, Complex64};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use strided_fused::{ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};
use strided_kernel::{
    erased_clamp_into_uninit, erased_map_into, erased_zip_into, ErasedMapOp, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedZipOp, Exec,
    KernelDType, StridedError,
};

fn single_op_plan(input_count: usize, op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count,
        outputs: vec![input_count].into_iter().collect(),
        ops: vec![FusedInst {
            op,
            inputs: (0..input_count).collect(),
        }]
        .into_iter()
        .collect(),
    }
}

fn assert_unary_matches_plan<T>(
    dtype: KernelDType,
    input: &[T],
    one_shot_op: ErasedMapOp,
    plan_op: FusedOp,
) where
    T: Copy + Default + PartialEq + core::fmt::Debug + strided_kernel::KernelStorageElement,
{
    let dims = [input.len()];
    let strides = [1isize];
    let mut one_shot = vec![T::default(); input.len()];
    let mut planned = vec![T::default(); input.len()];

    {
        let input = ErasedRawStridedRef::from_slice(input, &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut one_shot, &dims, &strides, 0).unwrap();
        erased_map_into(
            dtype,
            one_shot_op,
            &mut output,
            &ErasedRawStridedPtr::from_ref(&input),
            Exec::serial(),
        )
        .unwrap();
    }

    {
        let input = ErasedRawStridedRef::from_slice(input, &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut planned, &dims, &strides, 0).unwrap();
        ErasedFusedPlan::compile(dtype, single_op_plan(1, plan_op))
            .unwrap()
            .execute(&mut output, &[input], Exec::serial())
            .unwrap();
    }

    assert_eq!(one_shot, planned);
}

fn assert_binary_matches_plan<T>(
    dtype: KernelDType,
    lhs: &[T],
    rhs: &[T],
    one_shot_op: ErasedZipOp,
    plan_op: FusedOp,
) where
    T: Copy + Default + PartialEq + core::fmt::Debug + strided_kernel::KernelStorageElement,
{
    let dims = [lhs.len()];
    let strides = [1isize];
    let mut one_shot = vec![T::default(); lhs.len()];
    let mut planned = vec![T::default(); lhs.len()];

    {
        let lhs = ErasedRawStridedRef::from_slice(lhs, &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::from_slice(rhs, &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut one_shot, &dims, &strides, 0).unwrap();
        erased_zip_into(
            dtype,
            one_shot_op,
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs),
            &ErasedRawStridedPtr::from_ref(&rhs),
            Exec::serial(),
        )
        .unwrap();
    }

    {
        let lhs = ErasedRawStridedRef::from_slice(lhs, &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::from_slice(rhs, &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut planned, &dims, &strides, 0).unwrap();
        ErasedFusedPlan::compile(dtype, single_op_plan(2, plan_op))
            .unwrap()
            .execute(&mut output, &[lhs, rhs], Exec::serial())
            .unwrap();
    }

    assert_eq!(one_shot, planned);
}

/// Bitwise `u64` key so the differential checks can compare NaN payloads and
/// signed zeros, where `PartialEq` would hide or conflate the result.
trait BitKey: Copy {
    fn bit_key(self) -> u64;
}

impl BitKey for f32 {
    fn bit_key(self) -> u64 {
        self.to_bits() as u64
    }
}

impl BitKey for f64 {
    fn bit_key(self) -> u64 {
        self.to_bits()
    }
}

fn bit_keys<T: BitKey>(values: &[T]) -> Vec<u64> {
    values.iter().map(|value| value.bit_key()).collect()
}

/// Differential check of the fused `Maximum`/`Minimum` against the ordinary
/// erased binary kernel for every f32/f64 edge value (NaN in either operand
/// and signed-zero ties).
fn assert_binary_bits_match<T: BitKey + Copy + strided_kernel::KernelStorageElement>(
    dtype: KernelDType,
    lhs: &[T],
    rhs: &[T],
    one_shot_op: ErasedZipOp,
    plan_op: FusedOp,
) {
    let dims = [lhs.len()];
    let strides = [1isize];
    let mut one_shot = lhs.to_vec();
    let mut planned = lhs.to_vec();

    {
        let lhs = ErasedRawStridedRef::from_slice(lhs, &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::from_slice(rhs, &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut one_shot, &dims, &strides, 0).unwrap();
        erased_zip_into(
            dtype,
            one_shot_op,
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs),
            &ErasedRawStridedPtr::from_ref(&rhs),
            Exec::serial(),
        )
        .unwrap();
    }

    {
        let lhs = ErasedRawStridedRef::from_slice(lhs, &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::from_slice(rhs, &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut planned, &dims, &strides, 0).unwrap();
        ErasedFusedPlan::compile(dtype, single_op_plan(2, plan_op))
            .unwrap()
            .execute(&mut output, &[lhs, rhs], Exec::serial())
            .unwrap();
    }

    assert_eq!(bit_keys(&one_shot), bit_keys(&planned));
}

/// Differential check of the fused `Clamp` against the ordinary erased clamp.
fn assert_clamp_bits_match<T: BitKey + Copy + strided_kernel::KernelStorageElement>(
    dtype: KernelDType,
    x: &[T],
    lo: &[T],
    hi: &[T],
) {
    let dims = [x.len()];
    let strides = [1isize];
    let init = |values: Vec<MaybeUninit<T>>| -> Vec<T> {
        values
            .into_iter()
            .map(|value| unsafe { value.assume_init() })
            .collect()
    };

    let mut ordinary_storage = vec![MaybeUninit::<T>::uninit(); x.len()];
    {
        let x = ErasedRawStridedRef::from_slice(x, &dims, &strides, 0).unwrap();
        let lo = ErasedRawStridedRef::from_slice(lo, &dims, &strides, 0).unwrap();
        let hi = ErasedRawStridedRef::from_slice(hi, &dims, &strides, 0).unwrap();
        let mut dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut ordinary_storage, &dims, &strides, 0)
                .unwrap();
        erased_clamp_into_uninit(
            dtype,
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&x),
            &ErasedRawStridedPtr::from_ref(&lo),
            &ErasedRawStridedPtr::from_ref(&hi),
            Exec::serial(),
        )
        .unwrap();
    }

    let mut planned_storage = vec![MaybeUninit::<T>::uninit(); x.len()];
    {
        let x = ErasedRawStridedRef::from_slice(x, &dims, &strides, 0).unwrap();
        let lo = ErasedRawStridedRef::from_slice(lo, &dims, &strides, 0).unwrap();
        let hi = ErasedRawStridedRef::from_slice(hi, &dims, &strides, 0).unwrap();
        let mut dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut planned_storage, &dims, &strides, 0)
                .unwrap();
        ErasedFusedPlan::compile(dtype, single_op_plan(3, FusedOp::Clamp))
            .unwrap()
            .execute_uninit(
                &mut dest,
                &[
                    ErasedRawStridedPtr::from_ref(&x),
                    ErasedRawStridedPtr::from_ref(&lo),
                    ErasedRawStridedPtr::from_ref(&hi),
                ],
                Exec::serial(),
            )
            .unwrap();
    }

    assert_eq!(
        bit_keys(&init(ordinary_storage)),
        bit_keys(&init(planned_storage))
    );
}

#[test]
fn fused_maximum_minimum_and_clamp_agree_with_the_ordinary_kernels_on_nan() {
    let f64_lhs = [f64::NAN, 3.0, f64::NAN, 0.0, -0.0, f64::INFINITY];
    let f64_rhs = [3.0, f64::NAN, f64::NAN, -0.0, 0.0, f64::NEG_INFINITY];
    assert_binary_bits_match(
        KernelDType::F64,
        &f64_lhs,
        &f64_rhs,
        ErasedZipOp::Maximum,
        FusedOp::Maximum,
    );
    assert_binary_bits_match(
        KernelDType::F64,
        &f64_lhs,
        &f64_rhs,
        ErasedZipOp::Minimum,
        FusedOp::Minimum,
    );
    assert_clamp_bits_match(
        KernelDType::F64,
        &[1.0, 2.0, 3.0, f64::NAN, 0.5, 2.0, -0.0],
        &[0.0, 4.0, f64::NAN, 0.0, f64::NAN, 0.0, 0.0],
        &[3.0, 1.0, 2.0, 1.0, 1.0, f64::NAN, 1.0],
    );

    let f32_lhs = [f32::NAN, 3.0, f32::NAN, 0.0, -0.0];
    let f32_rhs = [3.0, f32::NAN, f32::NAN, -0.0, 0.0];
    assert_binary_bits_match(
        KernelDType::F32,
        &f32_lhs,
        &f32_rhs,
        ErasedZipOp::Maximum,
        FusedOp::Maximum,
    );
    assert_binary_bits_match(
        KernelDType::F32,
        &f32_lhs,
        &f32_rhs,
        ErasedZipOp::Minimum,
        FusedOp::Minimum,
    );
    assert_clamp_bits_match(
        KernelDType::F32,
        &[1.0, 2.0, 3.0, f32::NAN],
        &[0.0, 4.0, f32::NAN, 0.0],
        &[3.0, 1.0, 2.0, 1.0],
    );
}

#[test]
fn one_shot_matches_plan_for_every_kernel_dtype() {
    macro_rules! assert_add {
        ($dtype:expr, $lhs:expr, $rhs:expr) => {
            assert_binary_matches_plan($dtype, $lhs, $rhs, ErasedZipOp::Add, FusedOp::Add)
        };
    }

    assert_add!(KernelDType::F32, &[1.0f32, -2.0], &[3.0, 4.0]);
    assert_add!(KernelDType::F64, &[1.0f64, -2.0], &[3.0, 4.0]);
    assert_add!(KernelDType::I32, &[1i32, -2], &[3, 4]);
    assert_add!(KernelDType::I64, &[1i64, -2], &[3, 4]);
    assert_add!(
        KernelDType::C32,
        &[Complex32::new(1.0, 2.0), Complex32::new(-2.0, 1.0)],
        &[Complex32::new(3.0, -1.0), Complex32::new(4.0, 2.0)]
    );
    assert_add!(
        KernelDType::C64,
        &[Complex64::new(1.0, 2.0), Complex64::new(-2.0, 1.0)],
        &[Complex64::new(3.0, -1.0), Complex64::new(4.0, 2.0)]
    );
    assert_unary_matches_plan(
        KernelDType::Bool,
        &[true, false],
        ErasedMapOp::Conj,
        FusedOp::Conj,
    );
}

#[test]
fn one_shot_covers_required_unary_and_binary_ops() {
    let input = [-3.0f64, 0.0, 2.0];
    for (one_shot_op, plan_op) in [
        (ErasedMapOp::Negate, FusedOp::Negate),
        (ErasedMapOp::Conj, FusedOp::Conj),
        (ErasedMapOp::Abs, FusedOp::Abs),
    ] {
        assert_unary_matches_plan(KernelDType::F64, &input, one_shot_op, plan_op);
    }

    let lhs = [5.0f64, -4.0, 9.0];
    let rhs = [2.0f64, 3.0, 4.0];
    for (one_shot_op, plan_op) in [
        (ErasedZipOp::Add, FusedOp::Add),
        (ErasedZipOp::Multiply, FusedOp::Multiply),
        (ErasedZipOp::Divide, FusedOp::Divide),
        (ErasedZipOp::Maximum, FusedOp::Maximum),
        (ErasedZipOp::Minimum, FusedOp::Minimum),
    ] {
        assert_binary_matches_plan(KernelDType::F64, &lhs, &rhs, one_shot_op, plan_op);
    }

    let dims = [3usize];
    let strides = [1isize];
    let input_ref = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut signed = [0.0f64; 3];
    let mut output = ErasedRawStridedMut::from_slice_mut(&mut signed, &dims, &strides, 0).unwrap();
    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Sign,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input_ref),
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(signed, [-1.0, 0.0, 1.0]);

    for (op, expected) in [
        (ErasedZipOp::Subtract, [3.0, -7.0, 5.0]),
        (ErasedZipOp::Remainder, [1.0, -1.0, 1.0]),
    ] {
        let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
        let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
        let mut actual = [0.0f64; 3];
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
        erased_zip_into(
            KernelDType::F64,
            op,
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
            &ErasedRawStridedPtr::from_ref(&rhs_ref),
            Exec::serial(),
        )
        .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn one_shot_handles_borrowed_noncontiguous_metadata() {
    let dims = [2usize, 3];
    let src_strides = [3isize, 1];
    let dst_strides = [1isize, 2];
    let lhs = [0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0];
    let rhs = [100.0f64, 101.0, 102.0, 110.0, 111.0, 112.0];
    let mut dst = [0.0f64; 6];
    let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &src_strides, 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &src_strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &dst_strides, 0).unwrap();

    erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
        Exec::serial(),
    )
    .unwrap();

    assert_eq!(dst, [100.0, 120.0, 102.0, 122.0, 104.0, 124.0]);
}

#[test]
fn one_shot_empty_output_is_a_noop_for_degenerate_strides() {
    let dims = [0usize, 2];
    let strides = [1isize, 0];
    let input: [f64; 0] = [];
    let mut output: [f64; 0] = [];
    let input = ErasedRawStridedRef::from_slice(&input, &dims, &strides, isize::MAX).unwrap();
    let mut output =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dims, &strides, isize::MAX).unwrap();

    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
        Exec::serial(),
    )
    .unwrap();

    erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
        &ErasedRawStridedPtr::from_ref(&input),
        Exec::serial(),
    )
    .unwrap();
}

#[test]
fn one_shot_rejects_unsupported_ops_before_writing() {
    let dims = [2usize];
    let strides = [1isize];
    let lhs = [true, false];
    let rhs = [false, true];
    let mut dst = [true, true];
    let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    let error = erased_zip_into(
        KernelDType::Bool,
        ErasedZipOp::Add,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
        Exec::serial(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        StridedError::UnsupportedOp {
            op: "add",
            dtype: "bool"
        }
    ));
    assert_eq!(dst, [true, true]);
}

#[test]
fn one_shot_complex_abs_uses_real_output_dtype() {
    macro_rules! check {
        ($input_ty:ty, $output_ty:ty, $input_dtype:expr, $output_dtype:expr) => {{
            let dims = [2usize];
            let strides = [1isize];
            let input = [<$input_ty>::new(3.0, 4.0), <$input_ty>::new(5.0, 12.0)];
            let input = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
            let mut actual = [0.0 as $output_ty; 2];
            let mut output =
                ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
            erased_map_into(
                $input_dtype,
                ErasedMapOp::Abs,
                &mut output,
                &ErasedRawStridedPtr::from_ref(&input),
                Exec::serial(),
            )
            .unwrap();
            assert_eq!(actual, [5.0, 13.0]);
        }};
    }

    check!(Complex32, f32, KernelDType::C32, KernelDType::F32);
    check!(Complex64, f64, KernelDType::C64, KernelDType::F64);
}

#[test]
fn one_shot_integer_division_is_wrapping_and_zero_is_preflighted() {
    macro_rules! check {
        ($ty:ty, $dtype:expr) => {{
            let dims = [2usize];
            let strides = [1isize];
            let lhs = [<$ty>::MIN, 7];
            let rhs = [-1 as $ty, 3];
            for (op, expected) in [
                (ErasedZipOp::Divide, [<$ty>::MIN, 2]),
                (ErasedZipOp::Remainder, [0, 1]),
            ] {
                let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
                let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
                let mut actual = [99 as $ty; 2];
                let mut output =
                    ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
                erased_zip_into(
                    $dtype,
                    op,
                    &mut output,
                    &ErasedRawStridedPtr::from_ref(&lhs),
                    &ErasedRawStridedPtr::from_ref(&rhs),
                    Exec::serial(),
                )
                .unwrap();
                assert_eq!(actual, expected);
            }

            for op in [ErasedZipOp::Divide, ErasedZipOp::Remainder] {
                let rhs = [1 as $ty, 0];
                let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
                let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
                let mut actual = [99 as $ty; 2];
                let mut output =
                    ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
                let error = erased_zip_into(
                    $dtype,
                    op,
                    &mut output,
                    &ErasedRawStridedPtr::from_ref(&lhs),
                    &ErasedRawStridedPtr::from_ref(&rhs),
                    Exec::serial(),
                )
                .unwrap_err();
                assert!(matches!(error, StridedError::IntegerDivisionByZero { .. }));
                assert_eq!(actual, [99, 99]);
            }
        }};
    }

    check!(i32, KernelDType::I32);
    check!(i64, KernelDType::I64);
}

#[test]
fn one_shot_integer_preflight_covers_rank_zero_empty_compact_and_fallback() {
    let dims: &[usize] = &[];
    let strides: &[isize] = &[];
    let lhs = ErasedRawStridedRef::from_slice(&[i32::MIN], dims, strides, 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(&[-1i32], dims, strides, 0).unwrap();
    let mut actual = [99i32];
    let mut output = ErasedRawStridedMut::from_slice_mut(&mut actual, dims, strides, 0).unwrap();
    erased_zip_into(
        KernelDType::I32,
        ErasedZipOp::Divide,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(actual, [i32::MIN]);

    let lhs = ErasedRawStridedRef::from_slice(&[7i32], dims, strides, 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(&[0i32], dims, strides, 0).unwrap();
    let mut actual = [123i32];
    let mut output = ErasedRawStridedMut::from_slice_mut(&mut actual, dims, strides, 0).unwrap();
    assert!(matches!(
        erased_zip_into(
            KernelDType::I32,
            ErasedZipOp::Divide,
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs),
            &ErasedRawStridedPtr::from_ref(&rhs),
            Exec::serial()
        ),
        Err(StridedError::IntegerDivisionByZero { op: "divide" })
    ));
    assert_eq!(actual, [123]);

    let empty_dims = [0usize, 2];
    let empty_strides = [1isize, 0];
    let lhs =
        ErasedRawStridedRef::from_slice(&[] as &[i32], &empty_dims, &empty_strides, isize::MAX)
            .unwrap();
    let rhs =
        ErasedRawStridedRef::from_slice(&[] as &[i32], &empty_dims, &empty_strides, isize::MAX)
            .unwrap();
    let mut actual: [i32; 0] = [];
    let mut output =
        ErasedRawStridedMut::from_slice_mut(&mut actual, &empty_dims, &empty_strides, isize::MAX)
            .unwrap();
    erased_zip_into(
        KernelDType::I32,
        ErasedZipOp::Divide,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
        Exec::serial(),
    )
    .unwrap();

    macro_rules! compact_late_zero {
        ($rank:expr, $ty:ty, $dtype:expr) => {{
            let dims = vec![2usize; $rank];
            let mut strides = Vec::with_capacity($rank);
            let mut stride = 1isize;
            for &dim in &dims {
                strides.push(stride);
                stride *= dim as isize;
            }
            let total = 1usize << $rank;
            let lhs_data = vec![1 as $ty; total];
            let mut rhs_data = vec![1 as $ty; total];
            rhs_data[total - 1] = 0;
            let lhs = ErasedRawStridedRef::from_slice(&lhs_data, &dims, &strides, 0).unwrap();
            let rhs = ErasedRawStridedRef::from_slice(&rhs_data, &dims, &strides, 0).unwrap();
            let mut actual = vec![77 as $ty; total];
            let mut output =
                ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
            assert!(matches!(
                erased_zip_into(
                    $dtype,
                    ErasedZipOp::Divide,
                    &mut output,
                    &ErasedRawStridedPtr::from_ref(&lhs),
                    &ErasedRawStridedPtr::from_ref(&rhs),
                    Exec::serial()
                ),
                Err(StridedError::IntegerDivisionByZero { op: "divide" })
            ));
            assert_eq!(actual, vec![77 as $ty; total]);
        }};
    }

    compact_late_zero!(1, i32, KernelDType::I32);
    compact_late_zero!(2, i32, KernelDType::I32);
    compact_late_zero!(4, i32, KernelDType::I32);
    compact_late_zero!(8, i32, KernelDType::I32);
    compact_late_zero!(9, i32, KernelDType::I32);

    macro_rules! negative_nonunit_layout {
        ($ty:ty, $dtype:expr) => {{
            let dims = [2usize, 3];
            let strides = [-2isize, 3];
            let lhs_data = [1 as $ty; 6];
            let rhs_data = [1 as $ty, 0, 1, 1, 0, 1, 1, 0, 1];
            let lhs = ErasedRawStridedRef::from_slice(&lhs_data, &dims, &[1isize, 2], 0).unwrap();
            let rhs = ErasedRawStridedRef::from_slice(&rhs_data, &dims, &strides, 2).unwrap();
            let mut actual = [77 as $ty; 6];
            let mut output =
                ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &[1isize, 2], 0).unwrap();
            erased_zip_into(
                $dtype,
                ErasedZipOp::Divide,
                &mut output,
                &ErasedRawStridedPtr::from_ref(&lhs),
                &ErasedRawStridedPtr::from_ref(&rhs),
                Exec::serial(),
            )
            .unwrap();
            assert_eq!(actual, [1 as $ty; 6]);

            let rhs_data = [1 as $ty, 0, 1, 1, 0, 1, 0, 0, 1];
            let rhs = ErasedRawStridedRef::from_slice(&rhs_data, &dims, &strides, 2).unwrap();
            let mut actual = [77 as $ty; 6];
            let mut output =
                ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &[1isize, 2], 0).unwrap();
            assert!(matches!(
                erased_zip_into(
                    $dtype,
                    ErasedZipOp::Remainder,
                    &mut output,
                    &ErasedRawStridedPtr::from_ref(&lhs),
                    &ErasedRawStridedPtr::from_ref(&rhs),
                    Exec::serial()
                ),
                Err(StridedError::IntegerDivisionByZero { op: "remainder" })
            ));
            assert_eq!(actual, [77 as $ty; 6]);
        }};
    }

    negative_nonunit_layout!(i32, KernelDType::I32);
    negative_nonunit_layout!(i64, KernelDType::I64);
}

#[test]
fn one_shot_preserves_tenferro_numeric_edge_semantics() {
    let dims = [2usize];
    let strides = [1isize];

    for op in [ErasedZipOp::Maximum, ErasedZipOp::Minimum] {
        let lhs = [0.0f64, -0.0];
        let rhs = [-0.0f64, 0.0];
        let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
        let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
        let mut actual = [1.0f64; 2];
        let mut output =
            ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
        erased_zip_into(
            KernelDType::F64,
            op,
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
            &ErasedRawStridedPtr::from_ref(&rhs_ref),
            Exec::serial(),
        )
        .unwrap();
        assert_eq!(actual[0].to_bits(), lhs[0].to_bits());
        assert_eq!(actual[1].to_bits(), lhs[1].to_bits());
    }

    let input = [Complex64::new(-0.0, 0.0), Complex64::new(3.0, 4.0)];
    let input = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut actual = [Complex64::new(1.0, 1.0); 2];
    let mut output = ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
    erased_map_into(
        KernelDType::C64,
        ErasedMapOp::Sign,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(actual[0].re.to_bits(), 0.0f64.to_bits());
    assert_eq!(actual[0].im.to_bits(), 0.0f64.to_bits());
    assert_eq!(actual[1], Complex64::new(0.6, 0.8));
}

#[test]
fn one_shot_rejects_overlap_before_forming_input_reference() {
    let dims = [2usize];
    let strides = [1isize];
    let mut storage = [1.0f64, 2.0];
    let ptr = NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap();
    let input = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::F64,
            ptr,
            core::mem::size_of_val(&storage),
            &dims,
            &strides,
            0,
        )
        .unwrap()
    };
    let mut output = ErasedRawStridedMut::from_slice_mut(&mut storage, &dims, &strides, 0).unwrap();
    let error = erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &mut output,
        &input,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StridedError::OverlappingInputOutput { input: 0 }
    ));
    assert_eq!(storage, [1.0, 2.0]);
}

#[test]
fn one_shot_accepts_small_injective_layout_without_allocating() {
    let dims = [3usize, 2];
    let input_strides = [1isize, 3];
    let output_strides = [2isize, 3];
    let input = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = ErasedRawStridedRef::from_slice(&input, &dims, &input_strides, 0).unwrap();
    let mut actual = [0.0f64; 8];
    let mut output =
        ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &output_strides, 0).unwrap();
    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(actual, [-1.0, 0.0, -2.0, -4.0, -3.0, -5.0, 0.0, -6.0]);
}

#[test]
fn one_shot_rejects_noninjective_destination_before_raw_replay() {
    let dims = [4usize];
    let input_strides = [1isize];
    let output_strides = [0isize];
    let input = [1.0f64, 2.0, 3.0, 4.0];
    let input = ErasedRawStridedRef::from_slice(&input, &dims, &input_strides, 0).unwrap();
    let mut actual = [7.0f64];
    let mut output =
        ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &output_strides, 0).unwrap();

    let error = erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
        Exec::serial(),
    )
    .unwrap_err();

    assert!(matches!(error, StridedError::NonInjectiveOutputLayout));
    assert_eq!(actual, [7.0]);
}

#[test]
fn integer_division_preflight_handles_high_rank_iteratively() {
    const RANK: usize = 20_000;
    let dims = vec![1usize; RANK];
    let strides = vec![0isize; RANK];
    let lhs = [i32::MIN];
    let rhs = [-1i32];
    let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
    let mut actual = [0i32];
    let mut output = ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &strides, 0).unwrap();
    erased_zip_into(
        KernelDType::I32,
        ErasedZipOp::Divide,
        &mut output,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(actual, [i32::MIN]);
}

/// Scale-robust complex division: the textbook `|b|²` denominator overflows
/// here, so the Baudin–Smith path must keep the representable quotient.
#[test]
fn one_shot_complex_divide_is_scale_robust() {
    fn divide_once(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
        let dims = [lhs.len()];
        let strides = [1isize];
        let mut out = vec![Complex64::default(); lhs.len()];
        let lhs_ref = ErasedRawStridedRef::from_slice(lhs, &dims, &strides, 0).unwrap();
        let rhs_ref = ErasedRawStridedRef::from_slice(rhs, &dims, &strides, 0).unwrap();
        let mut out_mut =
            ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &strides, 0).unwrap();
        erased_zip_into(
            KernelDType::C64,
            ErasedZipOp::Divide,
            &mut out_mut,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
            &ErasedRawStridedPtr::from_ref(&rhs_ref),
            Exec::serial(),
        )
        .unwrap();
        out[0]
    }

    // 1 / (2^600 + 2^600 i) == 2^-601 (1 - i)
    let huge = 2f64.powi(600);
    let quotient = divide_once(&[Complex64::new(1.0, 0.0)], &[Complex64::new(huge, huge)]);
    let expected = 2f64.powi(-601);
    assert!(
        (quotient.re - expected).abs() <= expected * 1e-15,
        "{quotient:?}"
    );
    assert!(
        (quotient.im + expected).abs() <= expected * 1e-15,
        "{quotient:?}"
    );

    // 1 / (2^-600 + 2^-600 i) == 2^599 (1 - i)
    let tiny = 2f64.powi(-600);
    let quotient = divide_once(&[Complex64::new(1.0, 0.0)], &[Complex64::new(tiny, tiny)]);
    let expected = 2f64.powi(599);
    assert!(
        (quotient.re - expected).abs() <= expected * 1e-15,
        "{quotient:?}"
    );
    assert!(
        (quotient.im + expected).abs() <= expected * 1e-15,
        "{quotient:?}"
    );

    // The erased one-shot and the fused plan agree on the edge case.
    assert_binary_matches_plan(
        KernelDType::C64,
        &[Complex64::new(1.0, 0.0)],
        &[Complex64::new(huge, huge)],
        ErasedZipOp::Divide,
        FusedOp::Divide,
    );
}
