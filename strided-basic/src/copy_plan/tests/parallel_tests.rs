//! Serial versus parallel `ExecContext` equivalence for `CopyPlan` replay.

use super::*;
use crate::threading::run_serial_and_parallel;
use num_complex::Complex64;

/// Smallest backing length covering every reachable offset from `offset`.
fn span(dims: &[usize], strides: &[isize], offset: isize) -> usize {
    if dims.iter().any(|&d| d == 0) {
        return 0;
    }
    let mut max = offset;
    for (&d, &s) in dims.iter().zip(strides) {
        if s > 0 {
            max += (d as isize - 1) * s;
        }
    }
    max as usize + 1
}

/// Offset that makes a layout with negative strides start in bounds.
fn base_offset(dims: &[usize], strides: &[isize]) -> isize {
    dims.iter()
        .zip(strides)
        .filter(|(_, &s)| s < 0)
        .map(|(&d, &s)| (d as isize - 1) * -s)
        .sum()
}

fn reference<T: Copy>(
    dst: &mut [T],
    src: &[T],
    dims: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    src_strides: &[isize],
    src_offset: isize,
    op: impl Fn(T) -> T,
) {
    let total: usize = dims.iter().product();
    for linear in 0..total {
        let mut rest = linear;
        let mut d = dst_offset;
        let mut s = src_offset;
        for axis in 0..dims.len() {
            let coord = (rest % dims[axis]) as isize;
            rest /= dims[axis];
            d += coord * dst_strides[axis];
            s += coord * src_strides[axis];
        }
        dst[d as usize] = op(src[s as usize]);
    }
}

struct Case {
    dims: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
}

fn cases() -> Vec<Case> {
    let case = |dims: &[usize], dst: &[isize], src: &[isize]| Case {
        dims: dims.to_vec(),
        dst_strides: dst.to_vec(),
        src_strides: src.to_vec(),
    };
    vec![
        // rank 1 contiguous, odd length above the threshold
        case(&[40_001], &[1], &[1]),
        // rank 1 reversed source
        case(&[40_003], &[1], &[-1]),
        // rank 1 stride-2 source (static slice with step 2)
        case(&[33_331], &[1], &[2]),
        // rank 2: fewer outer positions than workers, long inner runs
        case(&[20_011, 3], &[1, 20_011], &[1, 20_011]),
        // rank 2 transposed destination, reversed outer source axis
        case(&[211, 199], &[199, 1], &[1, -211]),
        // rank 6 with mixed, negative and non-unit strides
        case(
            &[3, 5, 7, 9, 11, 5],
            &[1, 3, 15, 105, 945, 10_395],
            &[-1, 6, -3, 90, 810, 8_910],
        ),
        // zero-size
        case(&[0, 50_000], &[1, 1], &[1, 1]),
        case(&[50_000, 0], &[1, 50_000], &[-1, 50_000]),
        // rank 0
        case(&[], &[], &[]),
    ]
}

#[test]
fn copy_plan_serial_and_parallel_contexts_agree() {
    for case in cases() {
        let Case {
            dims,
            dst_strides,
            src_strides,
        } = &case;
        let plan = CopyPlan::compile(dims, dst_strides, src_strides).unwrap();
        let src_offset = base_offset(dims, src_strides);
        let dst_offset = base_offset(dims, dst_strides);
        let src_len = span(dims, src_strides, src_offset);
        let dst_len = span(dims, dst_strides, dst_offset).max(1);
        let src: Vec<f64> = (0..src_len).map(|v| v as f64 * 0.5 - 7.0).collect();
        let source = RawStridedRef::new(&src, dims, src_strides, src_offset).unwrap();

        let mut expected = vec![-1.0f64; dst_len];
        reference(
            &mut expected,
            &src,
            dims,
            dst_strides,
            dst_offset,
            src_strides,
            src_offset,
            |v| v,
        );
        let mut expected_scaled = vec![-1.0f64; dst_len];
        reference(
            &mut expected_scaled,
            &src,
            dims,
            dst_strides,
            dst_offset,
            src_strides,
            src_offset,
            |v| 3.0 * v,
        );

        let (serial, parallel) = run_serial_and_parallel(|| {
            let mut copied = vec![-1.0f64; dst_len];
            let mut dest = RawStridedMut::new(&mut copied, dims, dst_strides, dst_offset).unwrap();
            plan.execute(&mut dest, &source).unwrap();

            let mut scaled = vec![-1.0f64; dst_len];
            let mut dest = RawStridedMut::new(&mut scaled, dims, dst_strides, dst_offset).unwrap();
            plan.execute_scale(&mut dest, &source, 3.0).unwrap();

            let mut storage = vec![MaybeUninit::new(-1.0f64); dst_len];
            let mut dest = RawStridedMut::new(&mut storage, dims, dst_strides, dst_offset).unwrap();
            plan.execute_uninit(&mut dest, &source).unwrap();
            // SAFETY: every slot was initialized to -1.0 before the copy.
            let uninit: Vec<f64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
            (copied, scaled, uninit)
        });
        assert_eq!(serial.0, expected, "execute serial {dims:?}");
        assert_eq!(parallel.0, expected, "execute parallel {dims:?}");
        assert_eq!(serial.1, expected_scaled, "execute_scale serial {dims:?}");
        assert_eq!(
            parallel.1, expected_scaled,
            "execute_scale parallel {dims:?}"
        );
        assert_eq!(serial.2, expected, "execute_uninit serial {dims:?}");
        assert_eq!(parallel.2, expected, "execute_uninit parallel {dims:?}");
    }
}

#[test]
fn copy_plan_uninit_parallel_initializes_every_reachable_slot() {
    // Truly uninitialized, dense destination: every slot must be written.
    let dims = [40_009usize];
    let plan = CopyPlan::compile(&dims, &[1], &[-1]).unwrap();
    let src: Vec<i64> = (0..40_009).collect();
    let source = RawStridedRef::new(&src, &dims, &[-1], 40_008).unwrap();
    let (serial, parallel) = run_serial_and_parallel(|| {
        let mut storage = vec![MaybeUninit::<i64>::uninit(); 40_009];
        let mut dest = RawStridedMut::new(&mut storage, &dims, &[1], 0).unwrap();
        plan.execute_uninit(&mut dest, &source).unwrap();
        // SAFETY: the dense unit-stride layout reaches every slot and
        // `execute_uninit` initializes every reachable slot on success.
        storage
            .iter()
            .map(|v| unsafe { v.assume_init() })
            .collect::<Vec<_>>()
    });
    let expected: Vec<i64> = (0..40_009).rev().collect();
    assert_eq!(serial, expected);
    assert_eq!(parallel, expected);
}

#[test]
fn copy_plan_conj_parallel_matches_serial() {
    let dims = [181usize, 187];
    let dst_strides = [1isize, 181];
    let src_strides = [187isize, 1];
    let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
    let src: Vec<Complex64> = (0..181 * 187)
        .map(|v| Complex64::new(v as f64, -(v as f64) * 0.25))
        .collect();
    let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
    let (serial, parallel) = run_serial_and_parallel(|| {
        let mut out = vec![Complex64::new(0.0, 0.0); 181 * 187];
        let mut dest = RawStridedMut::new(&mut out, &dims, &dst_strides, 0).unwrap();
        plan.execute_conj(&mut dest, &source).unwrap();
        out
    });
    assert_eq!(serial, parallel);
    assert_eq!(serial[1], src[187].conj());
}
