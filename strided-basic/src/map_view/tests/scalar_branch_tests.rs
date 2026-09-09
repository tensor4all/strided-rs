use super::*;
use crate::{RawStridedMut, RawStridedRef, StridedArray};
use strided_view::Identity;

#[test]
fn raw_map_rejects_noninjective_destination_before_write() {
    let dims = [4usize];
    let source_strides = [1isize];
    let dest_strides = [0isize];
    let lhs = [1.0f64, 2.0, 3.0, 4.0];
    let lhs = RawStridedRef::new(&lhs, &dims, &source_strides, 0).unwrap();

    let mut output = [7.0f64];
    let mut dest = RawStridedMut::new(&mut output, &dims, &dest_strides, 0).unwrap();
    let error = map_raw_into::<f64, f64, Identity>(&mut dest, &lhs, |value| -value).unwrap_err();
    assert!(matches!(error, StridedError::NonInjectiveOutputLayout));
    assert_eq!(output, [7.0]);
}

#[test]
fn test_inner_loop_map2_stride_specializations() {
    let a = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
    let b = [17.0, 19.0, 23.0, 29.0, 31.0, 37.0];

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_map2::<f64, f64, f64, Identity, Identity>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            3,
            &|x, y| x + y,
        );
    }
    assert_eq!(out, [19.0, 22.0, 28.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_map2::<f64, f64, f64, Identity, Identity>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            1,
            b.as_ptr(),
            0,
            3,
            &|x, y| x * y,
        );
    }
    assert_eq!(out, [34.0, 51.0, 85.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_map2::<f64, f64, f64, Identity, Identity>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            0,
            b.as_ptr(),
            1,
            3,
            &|x, y| x * y,
        );
    }
    assert_eq!(out, [34.0, 38.0, 46.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_map2::<f64, f64, f64, Identity, Identity>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            0,
            b.as_ptr(),
            0,
            3,
            &|x, y| x + y,
        );
    }
    assert_eq!(out, [19.0, 19.0, 19.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_map2::<f64, f64, f64, Identity, Identity>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            2,
            b.as_ptr(),
            0,
            3,
            &|x, y| x + y,
        );
    }
    assert_eq!(out, [19.0, 22.0, 28.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_map2::<f64, f64, f64, Identity, Identity>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            0,
            b.as_ptr(),
            2,
            3,
            &|x, y| x + y,
        );
    }
    assert_eq!(out, [19.0, 25.0, 33.0]);
}

#[test]
fn test_inner_loop_mul2_stride_specializations() {
    let a = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
    let b = [17.0, 19.0, 23.0, 29.0, 31.0, 37.0];

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_mul2::<InitializedOutput, f64, f64, f64>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            3,
        );
    }
    assert_eq!(out, [34.0, 57.0, 115.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_mul2::<InitializedOutput, f64, f64, f64>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            0,
            b.as_ptr(),
            1,
            3,
        );
    }
    assert_eq!(out, [34.0, 38.0, 46.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_mul2::<InitializedOutput, f64, f64, f64>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            0,
            b.as_ptr(),
            0,
            3,
        );
    }
    assert_eq!(out, [34.0, 34.0, 34.0]);

    let mut out = [0.0; 3];
    unsafe {
        inner_loop_mul2::<InitializedOutput, f64, f64, f64>(
            out.as_mut_ptr(),
            1,
            a.as_ptr(),
            0,
            b.as_ptr(),
            2,
            3,
        );
    }
    assert_eq!(out, [34.0, 46.0, 62.0]);
}

#[test]
fn test_broadcast_mul_into_error_branches_and_non_identity_ops() {
    let lhs = StridedArray::<f64>::row_major(&[2, 3]);
    let rhs = StridedArray::<f64>::row_major(&[2, 3]);
    let mut out = StridedArray::<f64>::row_major(&[2, 3]);

    let err = broadcast_mul_into(&mut out.view_mut(), &lhs.view(), &[0], &rhs.view(), &[0, 1])
        .unwrap_err();
    assert!(matches!(err, StridedError::RankMismatch(2, 1)));

    let err = broadcast_mul_into(
        &mut out.view_mut(),
        &lhs.view(),
        &[0, 3],
        &rhs.view(),
        &[0, 1],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        StridedError::InvalidAxis { axis: 3, rank: 2 }
    ));

    let err = broadcast_mul_into(
        &mut out.view_mut(),
        &lhs.view(),
        &[0, 0],
        &rhs.view(),
        &[0, 1],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        StridedError::InvalidAxis { axis: 0, rank: 2 }
    ));

    let rhs_bad = StridedArray::<f64>::row_major(&[2, 4]);
    let err = broadcast_mul_into(
        &mut out.view_mut(),
        &lhs.view(),
        &[0, 1],
        &rhs_bad.view(),
        &[0, 1],
    )
    .unwrap_err();
    assert!(matches!(err, StridedError::ShapeMismatch(_, _)));

    let lhs_conj = lhs.view().conj();
    broadcast_mul_into(
        &mut out.view_mut(),
        &lhs_conj,
        &[0, 1],
        &rhs.view(),
        &[0, 1],
    )
    .unwrap();
}

#[test]
fn contiguous_mul_range_plan_available_without_parallel_feature() {
    let dims = [3usize; 16];
    let dst = [
        1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969,
        14348907,
    ];
    let lhs = [1isize, 3, 9, 27, 81, 243, 729, 2187, 0, 0, 0, 0, 0, 0, 0, 0];
    let rhs = [0isize, 0, 0, 0, 0, 0, 0, 0, 1, 3, 9, 27, 81, 243, 729, 2187];

    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

    assert_eq!(plan.inner_len, 6561);
    assert_eq!(plan.row_len, 3);
    assert_eq!(plan.fast_axis, 0);
}

#[test]
fn contiguous_range_mul_single_thread_computes_large_broadcast_mul() {
    let dims = [3usize; 10];
    let dst = [1isize, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683];
    let lhs = [1isize, 3, 9, 27, 81, 0, 0, 0, 0, 0];
    let rhs = [0isize, 0, 0, 0, 0, 1, 3, 9, 27, 81];
    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();
    let total = total_len(&dims);
    let block_len = plan.inner_len.max(1).saturating_mul(plan.row_len.max(1));
    let outer_groups = total.div_ceil(block_len);

    let a = vec![2.0; 243];
    let b = vec![3.0; 243];
    let mut out = vec![0.0; total];

    assert!(run_contiguous_range_mul_single_thread::<
        InitializedOutput,
        f64,
        f64,
        f64,
    >(
        out.as_mut_ptr(),
        &dims,
        a.as_ptr(),
        &lhs,
        b.as_ptr(),
        &rhs,
        &plan,
        total,
        block_len,
        outer_groups,
    ));
    assert!(out.iter().all(|&x| x == 6.0));
}
