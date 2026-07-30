use num_complex::{Complex32, Complex64};
use std::ptr::NonNull;
use strided_kernel::{
    erased_map_into, erased_zip_into, ErasedFusedPlan, ErasedMapOp, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedZipOp, ExecContext, FusedInst, FusedOp,
    FusedPlan, KernelDType, StridedError,
};

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data)) }
}

fn as_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(
            data.as_mut_ptr().cast::<u8>(),
            core::mem::size_of_val(data),
        )
    }
}

fn single_op_plan(input_count: usize, op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count,
        outputs: vec![input_count],
        ops: vec![FusedInst {
            op,
            inputs: (0..input_count).collect(),
        }],
    }
}

fn assert_unary_matches_plan<T>(
    dtype: KernelDType,
    input: &[T],
    one_shot_op: ErasedMapOp,
    plan_op: FusedOp,
) where
    T: Copy + Default + PartialEq + core::fmt::Debug,
{
    let dims = [input.len()];
    let strides = [1isize];
    let mut one_shot = vec![T::default(); input.len()];
    let mut planned = vec![T::default(); input.len()];

    {
        let input = ErasedRawStridedRef::new(dtype, as_bytes(input), &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::new(dtype, as_bytes_mut(&mut one_shot), &dims, &strides, 0)
                .unwrap();
        erased_map_into(
            dtype,
            one_shot_op,
            &ExecContext::serial(),
            &mut output,
            &ErasedRawStridedPtr::from_ref(&input),
        )
        .unwrap();
    }

    {
        let input = ErasedRawStridedRef::new(dtype, as_bytes(input), &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::new(dtype, as_bytes_mut(&mut planned), &dims, &strides, 0)
                .unwrap();
        ErasedFusedPlan::compile(dtype, single_op_plan(1, plan_op))
            .unwrap()
            .execute(&ExecContext::serial(), &mut output, &[input])
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
    T: Copy + Default + PartialEq + core::fmt::Debug,
{
    let dims = [lhs.len()];
    let strides = [1isize];
    let mut one_shot = vec![T::default(); lhs.len()];
    let mut planned = vec![T::default(); lhs.len()];

    {
        let lhs = ErasedRawStridedRef::new(dtype, as_bytes(lhs), &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::new(dtype, as_bytes(rhs), &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::new(dtype, as_bytes_mut(&mut one_shot), &dims, &strides, 0)
                .unwrap();
        erased_zip_into(
            dtype,
            one_shot_op,
            &ExecContext::serial(),
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs),
            &ErasedRawStridedPtr::from_ref(&rhs),
        )
        .unwrap();
    }

    {
        let lhs = ErasedRawStridedRef::new(dtype, as_bytes(lhs), &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::new(dtype, as_bytes(rhs), &dims, &strides, 0).unwrap();
        let mut output =
            ErasedRawStridedMut::new(dtype, as_bytes_mut(&mut planned), &dims, &strides, 0)
                .unwrap();
        ErasedFusedPlan::compile(dtype, single_op_plan(2, plan_op))
            .unwrap()
            .execute(&ExecContext::serial(), &mut output, &[lhs, rhs])
            .unwrap();
    }

    assert_eq!(one_shot, planned);
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
    let input_ref =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&input), &dims, &strides, 0).unwrap();
    let mut signed = [0.0f64; 3];
    let mut output = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut signed),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Sign,
        &ExecContext::serial(),
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input_ref),
    )
    .unwrap();
    assert_eq!(signed, [-1.0, 0.0, 1.0]);

    for (op, expected) in [
        (ErasedZipOp::Subtract, [3.0, -7.0, 5.0]),
        (ErasedZipOp::Remainder, [1.0, -1.0, 1.0]),
    ] {
        let lhs_ref =
            ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&lhs), &dims, &strides, 0).unwrap();
        let rhs_ref =
            ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&rhs), &dims, &strides, 0).unwrap();
        let mut actual = [0.0f64; 3];
        let mut output = ErasedRawStridedMut::new(
            KernelDType::F64,
            as_bytes_mut(&mut actual),
            &dims,
            &strides,
            0,
        )
        .unwrap();
        erased_zip_into(
            KernelDType::F64,
            op,
            &ExecContext::serial(),
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
            &ErasedRawStridedPtr::from_ref(&rhs_ref),
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
    let lhs =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&lhs), &dims, &src_strides, 0).unwrap();
    let rhs =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&rhs), &dims, &src_strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &dims,
        &dst_strides,
        0,
    )
    .unwrap();

    erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
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
    let input = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&input),
        &dims,
        &strides,
        isize::MAX,
    )
    .unwrap();
    let mut output = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut output),
        &dims,
        &strides,
        isize::MAX,
    )
    .unwrap();

    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &ExecContext::serial(),
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
    )
    .unwrap();

    erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &ExecContext::serial(),
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
        &ErasedRawStridedPtr::from_ref(&input),
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
    let lhs =
        ErasedRawStridedRef::new(KernelDType::Bool, as_bytes(&lhs), &dims, &strides, 0).unwrap();
    let rhs =
        ErasedRawStridedRef::new(KernelDType::Bool, as_bytes(&rhs), &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::Bool,
        as_bytes_mut(&mut dst),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    let error = erased_zip_into(
        KernelDType::Bool,
        ErasedZipOp::Add,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
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
            let input =
                ErasedRawStridedRef::new($input_dtype, as_bytes(&input), &dims, &strides, 0)
                    .unwrap();
            let mut actual = [0.0 as $output_ty; 2];
            let mut output = ErasedRawStridedMut::new(
                $output_dtype,
                as_bytes_mut(&mut actual),
                &dims,
                &strides,
                0,
            )
            .unwrap();
            erased_map_into(
                $input_dtype,
                ErasedMapOp::Abs,
                &ExecContext::serial(),
                &mut output,
                &ErasedRawStridedPtr::from_ref(&input),
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
                let lhs =
                    ErasedRawStridedRef::new($dtype, as_bytes(&lhs), &dims, &strides, 0).unwrap();
                let rhs =
                    ErasedRawStridedRef::new($dtype, as_bytes(&rhs), &dims, &strides, 0).unwrap();
                let mut actual = [99 as $ty; 2];
                let mut output =
                    ErasedRawStridedMut::new($dtype, as_bytes_mut(&mut actual), &dims, &strides, 0)
                        .unwrap();
                erased_zip_into(
                    $dtype,
                    op,
                    &ExecContext::serial(),
                    &mut output,
                    &ErasedRawStridedPtr::from_ref(&lhs),
                    &ErasedRawStridedPtr::from_ref(&rhs),
                )
                .unwrap();
                assert_eq!(actual, expected);
            }

            let rhs = [1 as $ty, 0];
            let lhs = ErasedRawStridedRef::new($dtype, as_bytes(&lhs), &dims, &strides, 0).unwrap();
            let rhs = ErasedRawStridedRef::new($dtype, as_bytes(&rhs), &dims, &strides, 0).unwrap();
            let mut actual = [99 as $ty; 2];
            let mut output =
                ErasedRawStridedMut::new($dtype, as_bytes_mut(&mut actual), &dims, &strides, 0)
                    .unwrap();
            let error = erased_zip_into(
                $dtype,
                ErasedZipOp::Divide,
                &ExecContext::serial(),
                &mut output,
                &ErasedRawStridedPtr::from_ref(&lhs),
                &ErasedRawStridedPtr::from_ref(&rhs),
            )
            .unwrap_err();
            assert!(matches!(
                error,
                StridedError::IntegerDivisionByZero { op: "divide" }
            ));
            assert_eq!(actual, [99, 99]);
        }};
    }

    check!(i32, KernelDType::I32);
    check!(i64, KernelDType::I64);
}

#[test]
fn one_shot_preserves_tenferro_numeric_edge_semantics() {
    let dims = [2usize];
    let strides = [1isize];

    for op in [ErasedZipOp::Maximum, ErasedZipOp::Minimum] {
        let lhs = [0.0f64, -0.0];
        let rhs = [-0.0f64, 0.0];
        let lhs_ref =
            ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&lhs), &dims, &strides, 0).unwrap();
        let rhs_ref =
            ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&rhs), &dims, &strides, 0).unwrap();
        let mut actual = [1.0f64; 2];
        let mut output = ErasedRawStridedMut::new(
            KernelDType::F64,
            as_bytes_mut(&mut actual),
            &dims,
            &strides,
            0,
        )
        .unwrap();
        erased_zip_into(
            KernelDType::F64,
            op,
            &ExecContext::serial(),
            &mut output,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
            &ErasedRawStridedPtr::from_ref(&rhs_ref),
        )
        .unwrap();
        assert_eq!(actual[0].to_bits(), lhs[0].to_bits());
        assert_eq!(actual[1].to_bits(), lhs[1].to_bits());
    }

    let input = [Complex64::new(-0.0, 0.0), Complex64::new(3.0, 4.0)];
    let input =
        ErasedRawStridedRef::new(KernelDType::C64, as_bytes(&input), &dims, &strides, 0).unwrap();
    let mut actual = [Complex64::new(1.0, 1.0); 2];
    let mut output = ErasedRawStridedMut::new(
        KernelDType::C64,
        as_bytes_mut(&mut actual),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    erased_map_into(
        KernelDType::C64,
        ErasedMapOp::Sign,
        &ExecContext::serial(),
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
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
        ErasedRawStridedPtr::new(
            KernelDType::F64,
            ptr,
            core::mem::size_of_val(&storage),
            &dims,
            &strides,
            0,
        )
        .unwrap()
    };
    let mut output = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut storage),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    let error = erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &ExecContext::serial(),
        &mut output,
        &input,
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
    let input =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&input), &dims, &input_strides, 0)
            .unwrap();
    let mut actual = [0.0f64; 8];
    let mut output = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut actual),
        &dims,
        &output_strides,
        0,
    )
    .unwrap();
    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &ExecContext::serial(),
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
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
    let input =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&input), &dims, &input_strides, 0)
            .unwrap();
    let mut actual = [7.0f64];
    let mut output = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut actual),
        &dims,
        &output_strides,
        0,
    )
    .unwrap();

    let error = erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &ExecContext::serial(),
        &mut output,
        &ErasedRawStridedPtr::from_ref(&input),
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
    let lhs =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&lhs), &dims, &strides, 0).unwrap();
    let rhs =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&rhs), &dims, &strides, 0).unwrap();
    let mut actual = [0i32];
    let mut output = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut actual),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    erased_zip_into(
        KernelDType::I32,
        ErasedZipOp::Divide,
        &ExecContext::serial(),
        &mut output,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
    )
    .unwrap();
    assert_eq!(actual, [i32::MIN]);
}
