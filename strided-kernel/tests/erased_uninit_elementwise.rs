//! Dtype-erased uninitialized elementwise entry points: differential checks
//! against the initialized one-shot entries and naive loops, plus semantics
//! and boundary-validation checks.

use core::mem::MaybeUninit;
use core::ptr::NonNull;
use num_complex::{Complex32, Complex64};
use strided_kernel::{
    erased_broadcast_mul_into_uninit, erased_clamp_into_uninit, erased_compare_into_uninit,
    erased_map_into, erased_map_into_uninit, erased_select_into_uninit, erased_zip_into,
    erased_zip_into_uninit, plan_lazy_outer_product, CompareOp, ErasedMapOp, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedZipOp, ExecContext,
    KernelDType, KernelStorageElement, StridedError,
};

const DIMS: [usize; 2] = [3, 4];
/// Transposed (row-major) input layout over 12 elements.
const T_STRIDES: [isize; 2] = [4, 1];
const C_STRIDES: [isize; 2] = [1, 3];

fn contexts() -> [ExecContext; 2] {
    [ExecContext::serial(), ExecContext::max_threads(4).unwrap()]
}

fn f64_values(seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..12)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 8.0 - 4.0
        })
        .collect()
}

fn i64_values(seed: u64) -> Vec<i64> {
    let mut values: Vec<i64> = f64_values(seed).iter().map(|v| *v as i64 + 5).collect();
    values[0] = i64::MAX;
    values[1] = i64::MIN;
    values
}

fn init<T: Copy>(values: Vec<MaybeUninit<T>>) -> Vec<T> {
    // SAFETY: every caller only passes storage a successful call initialized.
    values
        .into_iter()
        .map(|v| unsafe { v.assume_init() })
        .collect()
}

fn run_map<T: KernelStorageElement + Copy + Default, D: KernelStorageElement + Copy + Default>(
    dtype: KernelDType,
    op: ErasedMapOp,
    input: &[T],
    ctx: &ExecContext,
) -> (Vec<D>, Vec<D>) {
    let source = ErasedRawStridedRef::from_slice(input, &DIMS, &T_STRIDES, 0).unwrap();
    let mut expected = vec![D::default(); 12];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut expected, &DIMS, &C_STRIDES, 0).unwrap();
    erased_map_into(
        dtype,
        op,
        ctx,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&source),
    )
    .unwrap();
    let mut out = vec![MaybeUninit::<D>::uninit(); 12];
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &DIMS, &C_STRIDES, 0).unwrap();
    erased_map_into_uninit(
        dtype,
        op,
        ctx,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&source),
    )
    .unwrap();
    (init(out), expected)
}

#[test]
fn map_uninit_matches_initialized_entry_for_every_dtype_and_op() {
    let ops = [
        ErasedMapOp::Negate,
        ErasedMapOp::Conj,
        ErasedMapOp::Abs,
        ErasedMapOp::Sign,
    ];
    for ctx in contexts() {
        let reals = f64_values(3);
        let ints = i64_values(4);
        let complex: Vec<Complex64> = reals
            .iter()
            .zip(f64_values(5))
            .map(|(&re, im)| Complex64::new(re, im))
            .collect();
        for op in ops {
            let (got, want) = run_map::<f64, f64>(KernelDType::F64, op, &reals, &ctx);
            assert_eq!(got, want);
            let (got, want) = run_map::<i64, i64>(KernelDType::I64, op, &ints, &ctx);
            assert_eq!(got, want);
            if op == ErasedMapOp::Abs {
                let (got, want) = run_map::<Complex64, f64>(KernelDType::C64, op, &complex, &ctx);
                assert_eq!(got, want);
            } else {
                let (got, want) =
                    run_map::<Complex64, Complex64>(KernelDType::C64, op, &complex, &ctx);
                assert_eq!(got, want);
            }
        }
        let bools: Vec<bool> = (0..12).map(|i| i % 3 == 0).collect();
        let (got, want) = run_map::<bool, bool>(KernelDType::Bool, ErasedMapOp::Conj, &bools, &ctx);
        assert_eq!(got, want);
    }
}

#[test]
fn complex_sign_keeps_unit_phase_for_tiny_magnitudes() {
    let input = [
        Complex64::new(3e-200, 4e-200),
        Complex64::new(0.0, 0.0),
        Complex64::new(-0.0, 1e-310),
    ];
    let source = ErasedRawStridedRef::from_slice(&input, &[3], &[1], 0).unwrap();
    for uninit in [false, true] {
        let got: Vec<Complex64> = if uninit {
            let mut out = vec![MaybeUninit::<Complex64>::uninit(); 3];
            let mut dest =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[3], &[1], 0).unwrap();
            erased_map_into_uninit(
                KernelDType::C64,
                ErasedMapOp::Sign,
                &ExecContext::serial(),
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&source),
            )
            .unwrap();
            init(out)
        } else {
            let mut out = vec![Complex64::default(); 3];
            let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[3], &[1], 0).unwrap();
            erased_map_into(
                KernelDType::C64,
                ErasedMapOp::Sign,
                &ExecContext::serial(),
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&source),
            )
            .unwrap();
            out
        };
        assert!((got[0].re - 0.6).abs() < 1e-15 && (got[0].im - 0.8).abs() < 1e-15);
        assert_eq!(got[1], Complex64::new(0.0, 0.0));
        assert!(got[2].re.abs() < 1e-15 && (got[2].im - 1.0).abs() < 1e-15);
    }
    let tiny = [Complex32::new(3e-30, -4e-30)];
    let source = ErasedRawStridedRef::from_slice(&tiny, &[1], &[1], 0).unwrap();
    let mut out = [MaybeUninit::<Complex32>::uninit()];
    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[1], &[1], 0).unwrap();
    erased_map_into_uninit(
        KernelDType::C32,
        ErasedMapOp::Sign,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&source),
    )
    .unwrap();
    let got = unsafe { out[0].assume_init() };
    assert!((got.re - 0.6).abs() < 1e-6 && (got.im + 0.8).abs() < 1e-6);
}

fn run_zip<T: KernelStorageElement + Copy + Default>(
    dtype: KernelDType,
    op: ErasedZipOp,
    lhs: &[T],
    rhs: &[T],
    ctx: &ExecContext,
) -> (Result<Vec<T>, StridedError>, Result<Vec<T>, StridedError>) {
    let lhs = ErasedRawStridedRef::from_slice(lhs, &DIMS, &T_STRIDES, 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(rhs, &DIMS, &C_STRIDES, 0).unwrap();
    let mut expected = vec![T::default(); 12];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut expected, &DIMS, &C_STRIDES, 0).unwrap();
    let want = erased_zip_into(
        dtype,
        op,
        ctx,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
    )
    .map(|()| expected);
    let mut out = vec![MaybeUninit::<T>::uninit(); 12];
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &DIMS, &C_STRIDES, 0).unwrap();
    let got = erased_zip_into_uninit(
        dtype,
        op,
        ctx,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
    )
    .map(|()| init(out));
    (got, want)
}

fn same_bits(lhs: &[f64], rhs: &[f64]) -> bool {
    lhs.iter()
        .zip(rhs)
        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
}

#[test]
fn zip_uninit_matches_initialized_entry_including_nan_and_wrapping() {
    let ops = [
        ErasedZipOp::Add,
        ErasedZipOp::Subtract,
        ErasedZipOp::Multiply,
        ErasedZipOp::Divide,
        ErasedZipOp::Remainder,
        ErasedZipOp::Maximum,
        ErasedZipOp::Minimum,
    ];
    for ctx in contexts() {
        let mut lhs = f64_values(7);
        let mut rhs = f64_values(8);
        lhs[2] = f64::NAN;
        rhs[5] = f64::NAN;
        rhs[6] = 0.0;
        lhs[7] = -0.0;
        rhs[7] = 0.0;
        let ints_lhs = i64_values(9);
        let ints_rhs: Vec<i64> = i64_values(10)
            .iter()
            .map(|v| if *v == 0 { 3 } else { *v })
            .collect();
        for op in ops {
            let (got, want) = run_zip(KernelDType::F64, op, &lhs, &rhs, &ctx);
            assert!(same_bits(&got.unwrap(), &want.unwrap()), "{op:?}");
            let (got, want) = run_zip(KernelDType::I64, op, &ints_lhs, &ints_rhs, &ctx);
            assert_eq!(got.unwrap(), want.unwrap(), "{op:?}");
        }
    }
    // Wrapping integer semantics, independent of the initialized entry.
    let (got, _) = run_zip(
        KernelDType::I64,
        ErasedZipOp::Multiply,
        &[i64::MAX; 12],
        &[2; 12],
        &ExecContext::serial(),
    );
    assert!(got.unwrap().iter().all(|&v| v == -2));
}

#[test]
fn zip_uninit_rejects_integer_zero_divisor_before_writes_at_high_rank() {
    // Rank 9 exceeds the fused-rank limit, so the heap odometer is used.
    let dims = [2usize, 1, 2, 1, 2, 1, 2, 1, 2];
    let mut strides = [0isize; 9];
    let mut stride = 1;
    for (axis, &dim) in dims.iter().enumerate() {
        strides[axis] = stride;
        stride *= dim as isize;
    }
    let len = 32;
    let lhs = vec![7i32; len];
    for zero_at in [0usize, 17, len - 1] {
        let mut rhs = vec![3i32; len];
        rhs[zero_at] = 0;
        let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
        for op in [ErasedZipOp::Divide, ErasedZipOp::Remainder] {
            let mut out = vec![MaybeUninit::new(-1i32); len];
            let mut dest =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &dims, &strides, 0).unwrap();
            let err = erased_zip_into_uninit(
                KernelDType::I32,
                op,
                &ExecContext::serial(),
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&lhs),
                &ErasedRawStridedPtr::from_ref(&rhs),
            )
            .unwrap_err();
            assert!(matches!(err, StridedError::IntegerDivisionByZero { .. }));
            assert!(init(out).iter().all(|&v| v == -1));
        }
    }
    // i32::MIN / -1 wraps instead of panicking.
    let (got, _) = run_zip(
        KernelDType::I32,
        ErasedZipOp::Divide,
        &[i32::MIN; 12],
        &[-1; 12],
        &ExecContext::serial(),
    );
    assert!(got.unwrap().iter().all(|&v| v == i32::MIN));
}

fn compare<T: KernelStorageElement + Copy>(
    dtype: KernelDType,
    op: CompareOp,
    lhs: &[T],
    rhs: &[T],
) -> Result<Vec<bool>, StridedError> {
    let n = [lhs.len()];
    let n = &n[..];
    let lhs = ErasedRawStridedRef::from_slice(lhs, n, &[1], 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(rhs, n, &[1], 0).unwrap();
    let mut out = vec![MaybeUninit::<bool>::uninit(); n[0]];
    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, n, &[1], 0).unwrap();
    erased_compare_into_uninit(
        dtype,
        op,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
    )
    .map(|()| init(out))
}

#[test]
fn compare_uninit_follows_partial_ord_and_rejects_complex_ordering() {
    let lhs = [1.0f64, 2.0, f64::NAN, 3.0, -0.0];
    let rhs = [2.0f64, 2.0, 1.0, f64::NAN, 0.0];
    let ops = [
        CompareOp::Eq,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ];
    let naive: [fn(f64, f64) -> bool; 5] = [
        |a, b| a == b,
        |a, b| a < b,
        |a, b| a <= b,
        |a, b| a > b,
        |a, b| a >= b,
    ];
    for (op, f) in ops.into_iter().zip(naive) {
        let want: Vec<bool> = lhs.iter().zip(&rhs).map(|(&a, &b)| f(a, b)).collect();
        assert_eq!(compare(KernelDType::F64, op, &lhs, &rhs).unwrap(), want);
    }
    assert_eq!(
        compare(KernelDType::I32, CompareOp::Ge, &[1i32, -5, 9], &[1, 0, 10]).unwrap(),
        [true, false, false]
    );
    assert_eq!(
        compare(
            KernelDType::Bool,
            CompareOp::Lt,
            &[false, true],
            &[true, true]
        )
        .unwrap(),
        [true, false]
    );
    let a = [Complex32::new(1.0, 2.0), Complex32::new(1.0, 0.0)];
    let b = [Complex32::new(1.0, 2.0), Complex32::new(1.0, 1.0)];
    assert_eq!(
        compare(KernelDType::C32, CompareOp::Eq, &a, &b).unwrap(),
        [true, false]
    );
    assert!(matches!(
        compare(KernelDType::C32, CompareOp::Lt, &a, &b),
        Err(StridedError::UnsupportedOp { .. })
    ));
}

#[test]
fn select_uninit_matches_naive_for_every_dtype_with_strided_and_broadcast_operands() {
    let pred: Vec<bool> = (0..12).map(|i| (i * 7) % 5 < 2).collect();
    let pred_ref = ErasedRawStridedRef::from_slice(&pred, &DIMS, &T_STRIDES, 0).unwrap();
    fn check<T: KernelStorageElement + Copy + PartialEq + core::fmt::Debug>(
        dtype: KernelDType,
        pred: &[bool],
        pred_ref: &ErasedRawStridedRef<'_>,
        on_true: &[T],
        on_false: T,
    ) {
        let on_true_ref = ErasedRawStridedRef::from_slice(on_true, &DIMS, &C_STRIDES, 0).unwrap();
        let on_false_arr = [on_false];
        // A stride-0 descriptor broadcasts one value.
        let on_false_ref =
            ErasedRawStridedRef::from_slice(&on_false_arr, &DIMS, &[0, 0], 0).unwrap();
        for ctx in contexts() {
            let mut out = vec![MaybeUninit::<T>::uninit(); 12];
            let mut dest =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &DIMS, &C_STRIDES, 0)
                    .unwrap();
            erased_select_into_uninit(
                dtype,
                &ctx,
                &mut dest,
                &ErasedRawStridedPtr::from_ref(pred_ref),
                &ErasedRawStridedPtr::from_ref(&on_true_ref),
                &ErasedRawStridedPtr::from_ref(&on_false_ref),
            )
            .unwrap();
            let got = init(out);
            for i in 0..3 {
                for j in 0..4 {
                    let p = pred[i * 4 + j];
                    let want = if p { on_true[i + 3 * j] } else { on_false };
                    assert_eq!(got[i + 3 * j], want);
                }
            }
        }
    }
    check(KernelDType::F32, &pred, &pred_ref, &[1.5f32; 12], -2.0);
    check(KernelDType::F64, &pred, &pred_ref, &f64_values(1), -3.25);
    check(KernelDType::I32, &pred, &pred_ref, &[4i32; 12], -1);
    check(KernelDType::I64, &pred, &pred_ref, &i64_values(2), 0);
    check(KernelDType::Bool, &pred, &pred_ref, &[true; 12], false);
    check(
        KernelDType::C32,
        &pred,
        &pred_ref,
        &[Complex32::new(1.0, -1.0); 12],
        Complex32::new(0.0, 2.0),
    );
    check(
        KernelDType::C64,
        &pred,
        &pred_ref,
        &[Complex64::new(1.0, -1.0); 12],
        Complex64::new(0.0, 2.0),
    );
}

fn clamp<T: KernelStorageElement + Copy>(
    dtype: KernelDType,
    x: &[T],
    lo: &[T],
    hi: &[T],
) -> Result<Vec<T>, StridedError> {
    let n = [x.len()];
    let n = &n[..];
    let x = ErasedRawStridedRef::from_slice(x, n, &[1], 0).unwrap();
    let lo = ErasedRawStridedRef::from_slice(lo, n, &[1], 0).unwrap();
    let hi = ErasedRawStridedRef::from_slice(hi, n, &[1], 0).unwrap();
    let mut out = vec![MaybeUninit::<T>::uninit(); n[0]];
    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, n, &[1], 0).unwrap();
    erased_clamp_into_uninit(
        dtype,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&x),
        &ErasedRawStridedPtr::from_ref(&lo),
        &ErasedRawStridedPtr::from_ref(&hi),
    )
    .map(|()| init(out))
}

#[test]
fn clamp_uninit_propagates_nan_and_prefers_hi_when_bounds_cross() {
    let got = clamp(
        KernelDType::F64,
        &[-5.0, 0.5, 5.0, f64::NAN, 0.5, 0.5, 2.0],
        &[0.0, 0.0, 0.0, 0.0, f64::NAN, 0.0, 3.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0, f64::NAN, 1.0],
    )
    .unwrap();
    assert_eq!(&got[..3], &[0.0, 0.5, 1.0]);
    assert!(got[3].is_nan() && got[4].is_nan() && got[5].is_nan());
    assert_eq!(got[6], 1.0);
    assert_eq!(
        clamp(
            KernelDType::I64,
            &[i64::MIN, 3, i64::MAX],
            &[-1, -1, -1],
            &[4, 4, 4]
        )
        .unwrap(),
        [-1, 3, 4]
    );
    assert_eq!(
        clamp(KernelDType::F32, &[-0.0f32], &[0.0], &[1.0]).unwrap()[0].to_bits(),
        0.0f32.to_bits(),
        "maximum(lo, x) keeps lo on ties"
    );
    assert!(matches!(
        clamp(KernelDType::Bool, &[true], &[false], &[true]),
        Err(StridedError::UnsupportedDType { .. })
    ));
    assert!(matches!(
        clamp(
            KernelDType::C64,
            &[Complex64::default()],
            &[Complex64::default()],
            &[Complex64::default()]
        ),
        Err(StridedError::UnsupportedDType { .. })
    ));
}

#[allow(clippy::too_many_arguments)]
fn naive_broadcast_mul<T: Copy>(
    out_dims: &[usize],
    lhs: &[T],
    lhs_strides: &[isize],
    lhs_axes: &[usize],
    rhs: &[T],
    rhs_strides: &[isize],
    rhs_axes: &[usize],
    mul: impl Fn(T, T) -> T,
) -> Vec<T> {
    let total: usize = out_dims.iter().product();
    let mut out = Vec::with_capacity(total);
    for linear in 0..total {
        let mut rest = linear;
        let mut coord = vec![0usize; out_dims.len()];
        for (axis, &dim) in out_dims.iter().enumerate() {
            coord[axis] = rest % dim;
            rest /= dim;
        }
        let offset = |strides: &[isize], axes: &[usize]| -> usize {
            axes.iter()
                .zip(strides)
                .map(|(&axis, &stride)| coord[axis] as isize * stride)
                .sum::<isize>() as usize
        };
        out.push(mul(
            lhs[offset(lhs_strides, lhs_axes)],
            rhs[offset(rhs_strides, rhs_axes)],
        ));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn broadcast_mul<T: KernelStorageElement + Copy>(
    dtype: KernelDType,
    ctx: &ExecContext,
    out_dims: &[usize],
    out_strides: &[isize],
    out_len: usize,
    lhs: &ErasedRawStridedRef<'_>,
    lhs_axes: &[usize],
    rhs: &ErasedRawStridedRef<'_>,
    rhs_axes: &[usize],
) -> Result<Vec<T>, StridedError> {
    let mut out = vec![MaybeUninit::<T>::uninit(); out_len];
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut out, out_dims, out_strides, 0).unwrap();
    erased_broadcast_mul_into_uninit(
        dtype,
        ctx,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(lhs),
        lhs_axes,
        &ErasedRawStridedPtr::from_ref(rhs),
        rhs_axes,
    )
    .map(|()| init(out))
}

#[test]
fn broadcast_mul_uninit_matches_naive_batched_outer_product() {
    // out[i, j, b] = lhs[b, i] * rhs[j, b] with a transposed lhs.
    let out_dims = [3usize, 4, 2];
    let out_strides = [1isize, 3, 12];
    let lhs: Vec<f64> = f64_values(21)[..6].to_vec();
    let rhs: Vec<f64> = f64_values(22)[..8].to_vec();
    let lhs_strides = [1isize, 2];
    let rhs_strides = [1isize, 4];
    let want = naive_broadcast_mul(
        &out_dims,
        &lhs,
        &lhs_strides,
        &[2, 0],
        &rhs,
        &rhs_strides,
        &[1, 2],
        |a, b| a * b,
    );
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &[2, 3], &lhs_strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &[4, 2], &rhs_strides, 0).unwrap();
    for ctx in contexts() {
        let got = broadcast_mul::<f64>(
            KernelDType::F64,
            &ctx,
            &out_dims,
            &out_strides,
            24,
            &lhs_ref,
            &[2, 0],
            &rhs_ref,
            &[1, 2],
        )
        .unwrap();
        assert_eq!(got, want);
    }
    let lhs_c: Vec<Complex64> = lhs.iter().map(|&v| Complex64::new(v, -v)).collect();
    let rhs_c: Vec<Complex64> = rhs.iter().map(|&v| Complex64::new(0.5, v)).collect();
    let want = naive_broadcast_mul(
        &out_dims,
        &lhs_c,
        &lhs_strides,
        &[2, 0],
        &rhs_c,
        &rhs_strides,
        &[1, 2],
        |a, b| a * b,
    );
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs_c, &[2, 3], &lhs_strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs_c, &[4, 2], &rhs_strides, 0).unwrap();
    let got = broadcast_mul::<Complex64>(
        KernelDType::C64,
        &ExecContext::serial(),
        &out_dims,
        &out_strides,
        24,
        &lhs_ref,
        &[2, 0],
        &rhs_ref,
        &[1, 2],
    )
    .unwrap();
    assert_eq!(got, want);
}

#[test]
fn broadcast_mul_uninit_wraps_integer_overflow_and_broadcasts_scalars() {
    let lhs = [i32::MAX, 3, i32::MIN];
    let rhs = [2i32];
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &[3], &[1], 0).unwrap();
    // Rank-0 operand broadcast over the output.
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &[], &[], 0).unwrap();
    for ctx in contexts() {
        let got = broadcast_mul::<i32>(
            KernelDType::I32,
            &ctx,
            &[3],
            &[1],
            3,
            &lhs_ref,
            &[0],
            &rhs_ref,
            &[],
        )
        .unwrap();
        assert_eq!(got, [-2, 6, 0]);
    }
    let lhs = [i64::MAX, -7];
    let rhs = [i64::MAX, 2, 5];
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &[2], &[1], 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &[3], &[1], 0).unwrap();
    let got = broadcast_mul::<i64>(
        KernelDType::I64,
        &ExecContext::serial(),
        &[2, 3],
        &[1, 2],
        6,
        &lhs_ref,
        &[0],
        &rhs_ref,
        &[1],
    )
    .unwrap();
    let want = naive_broadcast_mul(
        &[2, 3],
        &lhs,
        &[1],
        &[0],
        &rhs,
        &[1],
        &[1],
        i64::wrapping_mul,
    );
    assert_eq!(got, want);
}

#[test]
fn broadcast_mul_uninit_rejects_bool_and_invalid_axes() {
    let values = [true, false];
    let bools = ErasedRawStridedRef::from_slice(&values, &[2], &[1], 0).unwrap();
    assert!(matches!(
        broadcast_mul::<bool>(
            KernelDType::Bool,
            &ExecContext::serial(),
            &[2],
            &[1],
            2,
            &bools,
            &[0],
            &bools,
            &[0],
        ),
        Err(StridedError::UnsupportedDType { .. })
    ));
    let values = [1.0f64, 2.0];
    let operand = ErasedRawStridedRef::from_slice(&values, &[2], &[1], 0).unwrap();
    for axes in [&[1usize][..], &[0, 0][..]] {
        assert!(broadcast_mul::<f64>(
            KernelDType::F64,
            &ExecContext::serial(),
            &[2],
            &[1],
            2,
            &operand,
            axes,
            &operand,
            &[0],
        )
        .is_err());
    }
}

#[test]
fn lazy_outer_product_layout_filled_by_broadcast_mul_matches_logical_product() {
    // lhs is 3 x 2 stored row-major, rhs is 4 x 2 stored row-major, batch last.
    let out_dims = [3usize, 4, 2];
    let lhs: Vec<f64> = f64_values(31)[..6].to_vec();
    let rhs: Vec<f64> = f64_values(32)[..8].to_vec();
    let (lhs_dims, lhs_strides) = ([3usize, 2], [2isize, 1]);
    let (rhs_dims, rhs_strides) = ([4usize, 2], [2isize, 1]);
    let (lhs_axes, rhs_axes) = ([0usize, 2], [1usize, 2]);
    // Physical order already matches logical order for single free axes, so
    // make lhs free rank 2 to trigger a reordering.
    assert!(plan_lazy_outer_product(
        &out_dims,
        &lhs_dims,
        &lhs_strides,
        &lhs_axes,
        &rhs_dims,
        &rhs_strides,
        &rhs_axes
    )
    .unwrap()
    .is_none());

    let out_dims = [2usize, 3, 4];
    let (lhs_dims, lhs_strides, lhs_axes) = ([2usize, 3], [3isize, 1], [0usize, 1]);
    let (rhs_dims, rhs_strides, rhs_axes) = ([4usize], [1isize], [2usize]);
    let layout = plan_lazy_outer_product(
        &out_dims,
        &lhs_dims,
        &lhs_strides,
        &lhs_axes,
        &rhs_dims,
        &rhs_strides,
        &rhs_axes,
    )
    .unwrap()
    .unwrap();
    assert_eq!(layout.base_dims, [3, 2, 4]);
    let base_len: usize = layout.base_dims.iter().product();
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &lhs_dims, &lhs_strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs[..4], &rhs_dims, &rhs_strides, 0).unwrap();
    let base = broadcast_mul::<f64>(
        KernelDType::F64,
        &ExecContext::serial(),
        &out_dims,
        &layout.output_strides,
        base_len,
        &lhs_ref,
        &lhs_axes,
        &rhs_ref,
        &rhs_axes,
    )
    .unwrap();
    // The base is written in physical order and read back through the strides.
    for i in 0..2 {
        for j in 0..3 {
            for (k, &rhs_k) in rhs[..4].iter().enumerate() {
                let at = i as isize * layout.output_strides[0]
                    + j as isize * layout.output_strides[1]
                    + k as isize * layout.output_strides[2];
                assert_eq!(base[at as usize], lhs[i * 3 + j] * rhs_k);
            }
        }
    }
}

#[test]
fn uninit_entries_reject_overlap_and_dtype_mismatch_before_writes() {
    let dims = [4usize];
    let strides = [1isize];
    let mut output = vec![MaybeUninit::new(99.0f64); 4];
    let ptr = NonNull::new(output.as_mut_ptr().cast::<u8>()).unwrap();
    // SAFETY: the pointer covers `output`, which outlives the descriptor, and
    // the entry under test must reject the overlap before reading it.
    let overlapping = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::F64,
            ptr,
            core::mem::size_of_val(output.as_slice()),
            &dims,
            &strides,
            0,
        )
        .unwrap()
    };
    let other = [1.0f64; 4];
    let other = ErasedRawStridedRef::from_slice(&other, &dims, &strides, 0).unwrap();
    let other = ErasedRawStridedPtr::from_ref(&other);
    let flags = [true; 4];
    let flags = ErasedRawStridedRef::from_slice(&flags, &dims, &strides, 0).unwrap();
    let flags = ErasedRawStridedPtr::from_ref(&flags);
    let ctx = ExecContext::serial();
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut output, &dims, &strides, 0).unwrap();
    let overlap = |result: Result<(), StridedError>, input: usize| {
        assert!(
            matches!(result, Err(StridedError::OverlappingInputOutput { input: i }) if i == input),
            "{result:?}"
        );
    };
    overlap(
        erased_map_into_uninit(
            KernelDType::F64,
            ErasedMapOp::Negate,
            &ctx,
            &mut dest,
            &overlapping,
        ),
        0,
    );
    overlap(
        erased_zip_into_uninit(
            KernelDType::F64,
            ErasedZipOp::Add,
            &ctx,
            &mut dest,
            &other,
            &overlapping,
        ),
        1,
    );
    overlap(
        erased_select_into_uninit(
            KernelDType::F64,
            &ctx,
            &mut dest,
            &flags,
            &other,
            &overlapping,
        ),
        2,
    );
    overlap(
        erased_clamp_into_uninit(
            KernelDType::F64,
            &ctx,
            &mut dest,
            &overlapping,
            &other,
            &other,
        ),
        0,
    );
    overlap(
        erased_broadcast_mul_into_uninit(
            KernelDType::F64,
            &ctx,
            &mut dest,
            &other,
            &[0],
            &overlapping,
            &[0],
        ),
        1,
    );
    assert!(matches!(
        erased_zip_into_uninit(
            KernelDType::F32,
            ErasedZipOp::Add,
            &ctx,
            &mut dest,
            &other,
            &other
        ),
        Err(StridedError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        erased_compare_into_uninit(
            KernelDType::F64,
            CompareOp::Eq,
            &ctx,
            &mut dest,
            &other,
            &other
        ),
        Err(StridedError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        erased_zip_into_uninit(
            KernelDType::Bool,
            ErasedZipOp::Add,
            &ctx,
            &mut dest,
            &flags,
            &flags
        ),
        Err(StridedError::DTypeMismatch { .. })
    ));
    assert!(init(output).iter().all(|&v| v == 99.0));
}

#[test]
fn uninit_entries_reject_shape_mismatch_and_non_injective_output() {
    let values = [1.0f64; 4];
    let input = ErasedRawStridedRef::from_slice(&values, &[4], &[1], 0).unwrap();
    let input = ErasedRawStridedPtr::from_ref(&input);
    let ctx = ExecContext::serial();
    let mut out = vec![MaybeUninit::<f64>::uninit(); 3];
    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[3], &[1], 0).unwrap();
    assert!(matches!(
        erased_map_into_uninit(
            KernelDType::F64,
            ErasedMapOp::Negate,
            &ctx,
            &mut dest,
            &input
        ),
        Err(StridedError::ShapeMismatch(..) | StridedError::RankMismatch(..))
    ));
    let mut out = vec![MaybeUninit::<f64>::uninit(); 1];
    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[4], &[0], 0).unwrap();
    assert!(matches!(
        erased_zip_into_uninit(
            KernelDType::F64,
            ErasedZipOp::Add,
            &ctx,
            &mut dest,
            &input,
            &input
        ),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
}
