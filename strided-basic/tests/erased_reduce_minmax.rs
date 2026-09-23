//! `ReduceOp::Max` / `ReduceOp::Min` semantics and differential checks
//! against a sequential fold.

use strided_basic::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan, ExecContext, KernelDType, ReduceOp,
    StridedError,
};

fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

fn full_reduce<T: Copy + Default + strided_basic::KernelStorageElement>(
    dtype: KernelDType,
    op: ReduceOp,
    input: &[T],
    dims: &[usize],
    strides: &[isize],
    ctx: &ExecContext,
) -> T {
    let plan = ErasedReducePlan::compile(dtype, op, dims, strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(input, dims, strides, 0).unwrap();
    let mut output = [T::default()];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();
    plan.execute(ctx, &mut dest, &source).unwrap();
    output[0]
}

fn pseudo_random_f64(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 200.0 - 100.0
        })
        .collect()
}

#[test]
fn full_max_min_match_sequential_fold_for_contiguous_and_strided_layouts() {
    let input = pseudo_random_f64(97 * 5, 7);
    let layouts: [(&[usize], &[isize]); 3] =
        [(&[485], &[1]), (&[97, 5], &[1, 97]), (&[97, 5], &[5, 1])];
    for ctx in [ExecContext::serial(), ExecContext::max_threads(4).unwrap()] {
        for (dims, strides) in layouts {
            let max = full_reduce(KernelDType::F64, ReduceOp::Max, &input, dims, strides, &ctx);
            let min = full_reduce(KernelDType::F64, ReduceOp::Min, &input, dims, strides, &ctx);
            assert_eq!(max, input.iter().copied().fold(f64::NEG_INFINITY, nan_max));
            assert_eq!(min, input.iter().copied().fold(f64::INFINITY, nan_min));
        }
    }
}

#[test]
fn float_max_min_propagate_nan_in_any_position() {
    for position in 0..9 {
        let mut input = pseudo_random_f64(9, position as u64 + 1);
        input[position] = f64::NAN;
        let max = full_reduce(
            KernelDType::F64,
            ReduceOp::Max,
            &input,
            &[9],
            &[1],
            &ExecContext::serial(),
        );
        let min = full_reduce(
            KernelDType::F64,
            ReduceOp::Min,
            &input,
            &[3, 3],
            &[3, 1],
            &ExecContext::serial(),
        );
        assert!(max.is_nan());
        assert!(min.is_nan());
    }
    let input = [1.0f32, f32::NAN, -3.0];
    let max = full_reduce(
        KernelDType::F32,
        ReduceOp::Max,
        &input,
        &[3],
        &[1],
        &ExecContext::serial(),
    );
    assert!(max.is_nan());
}

#[test]
fn float_max_min_handle_infinities() {
    let input = [f64::NEG_INFINITY, -1.0, f64::INFINITY];
    let ctx = ExecContext::serial();
    assert_eq!(
        full_reduce(KernelDType::F64, ReduceOp::Max, &input, &[3], &[1], &ctx),
        f64::INFINITY
    );
    assert_eq!(
        full_reduce(KernelDType::F64, ReduceOp::Min, &input, &[3], &[1], &ctx),
        f64::NEG_INFINITY
    );
}

#[test]
fn integer_max_min_use_ordered_extremes() {
    let input = [3i64, i64::MIN, 7, i64::MAX, -2];
    let ctx = ExecContext::serial();
    assert_eq!(
        full_reduce(KernelDType::I64, ReduceOp::Max, &input, &[5], &[1], &ctx),
        i64::MAX
    );
    assert_eq!(
        full_reduce(KernelDType::I64, ReduceOp::Min, &input, &[5], &[1], &ctx),
        i64::MIN
    );
    let input = [3i32, -9, 7, 1];
    assert_eq!(
        full_reduce(
            KernelDType::I32,
            ReduceOp::Max,
            &input,
            &[2, 2],
            &[1, 2],
            &ctx
        ),
        7
    );
    assert_eq!(
        full_reduce(
            KernelDType::I32,
            ReduceOp::Min,
            &input,
            &[2, 2],
            &[1, 2],
            &ctx
        ),
        -9
    );
}

#[test]
fn empty_max_min_write_identities() {
    let input: [f64; 0] = [];
    let ctx = ExecContext::serial();
    assert_eq!(
        full_reduce(KernelDType::F64, ReduceOp::Max, &input, &[0], &[1], &ctx),
        f64::NEG_INFINITY
    );
    assert_eq!(
        full_reduce(KernelDType::F64, ReduceOp::Min, &input, &[0], &[1], &ctx),
        f64::INFINITY
    );
    let input: [i32; 0] = [];
    assert_eq!(
        full_reduce(KernelDType::I32, ReduceOp::Max, &input, &[0], &[1], &ctx),
        i32::MIN
    );
    assert_eq!(
        full_reduce(KernelDType::I32, ReduceOp::Min, &input, &[0], &[1], &ctx),
        i32::MAX
    );
}

#[test]
fn max_min_reject_complex_and_bool() {
    for dtype in [KernelDType::C32, KernelDType::C64, KernelDType::Bool] {
        for op in [ReduceOp::Max, ReduceOp::Min] {
            let err = ErasedReducePlan::compile(dtype, op, &[1], &[1]).unwrap_err();
            assert!(
                matches!(err, StridedError::UnsupportedDType { .. }),
                "{err:?}"
            );
            let err =
                ErasedReducePlan::compile_axes(dtype, op, &[2], &[1], &[], &[], &[0]).unwrap_err();
            assert!(
                matches!(err, StridedError::UnsupportedDType { .. }),
                "{err:?}"
            );
        }
    }
}

#[test]
fn axes_max_min_match_per_output_fold() {
    // Column-major 4 x 3 x 5 source with a transposed physical layout.
    let src_dims = [4usize, 3, 5];
    let src_strides = [15isize, 5, 1];
    let input = pseudo_random_f64(60, 11);
    let get = |i: usize, j: usize, k: usize| input[i * 15 + j * 5 + k];
    for ctx in [ExecContext::serial(), ExecContext::max_threads(3).unwrap()] {
        for (op, init, combine) in [
            (
                ReduceOp::Max,
                f64::NEG_INFINITY,
                nan_max as fn(f64, f64) -> f64,
            ),
            (ReduceOp::Min, f64::INFINITY, nan_min as fn(f64, f64) -> f64),
        ] {
            let dest_dims = [3usize];
            let dest_strides = [1isize];
            let plan = ErasedReducePlan::compile_axes(
                KernelDType::F64,
                op,
                &src_dims,
                &src_strides,
                &dest_dims,
                &dest_strides,
                &[0, 2],
            )
            .unwrap();
            let source =
                ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
            let mut output = [0.0f64; 3];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
                    .unwrap();
            plan.execute(&ctx, &mut dest, &source).unwrap();
            for (j, &value) in output.iter().enumerate() {
                let mut expected = init;
                for k in 0..5 {
                    for i in 0..4 {
                        expected = combine(expected, get(i, j, k));
                    }
                }
                assert_eq!(value, expected);
            }
        }
    }
}

#[test]
fn axes_max_propagates_nan_only_into_affected_outputs() {
    let src_dims = [2usize, 3];
    let src_strides = [1isize, 2];
    let input = [1.0f32, 2.0, f32::NAN, 4.0, 5.0, -6.0];
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F32,
        ReduceOp::Max,
        &src_dims,
        &src_strides,
        &[3],
        &[1],
        &[0],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut output = [0.0f32; 3];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[3], &[1], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    assert_eq!(output[0], 2.0);
    assert!(output[1].is_nan());
    assert_eq!(output[2], 5.0);
}
