//! Coverage and correctness for the strided-destination route of
//! `reduce_axis_into` (`reduce_axis_range_strided`) and for the validation
//! errors of the public entry.
//!
//! `reduce_axis` always allocates a column-major output, so the general
//! strided cursor is only reachable through `reduce_axis_into` with a
//! destination whose layout is injective but not column-major. These cases
//! exercise the column sweep, the per-output fold, the multi-axis carry, the
//! empty-axis fill, and the rank spill of the cursor. The reference is a
//! naive loop over the logical indices, and the data is exact `i64` so any
//! lane or block order is exact.

#[cfg(feature = "parallel")]
mod common;
#[cfg(feature = "parallel")]
use common::Budget;
use strided_basic::Exec;
use strided_basic::{reduce_axis_into, StridedError, StridedView, StridedViewMut};

/// A strided layout over a flat buffer.
struct Layout {
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    len: usize,
}

fn layout(dims: &[usize], strides: &[isize], offset: isize, len: usize) -> Layout {
    Layout {
        dims: dims.to_vec(),
        strides: strides.to_vec(),
        offset,
        len,
    }
}

/// Visits every multi-index in column-major order (axis 0 fastest).
fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    if dims.contains(&0) {
        return;
    }
    let mut index = vec![0usize; dims.len()];
    loop {
        f(&index);
        let mut axis = 0;
        loop {
            if axis == dims.len() {
                return;
            }
            index[axis] += 1;
            if index[axis] < dims[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
    }
}

fn offset_of(layout: &Layout, index: &[usize]) -> usize {
    let mut offset = layout.offset;
    for (&i, &s) in index.iter().zip(&layout.strides) {
        offset += i as isize * s;
    }
    offset as usize
}

/// Naive per-output wrapping sum seeded with `init`, in destination order.
fn expected_values(
    src_data: &[i64],
    src: &Layout,
    dest: &Layout,
    axis: usize,
    init: i64,
) -> Vec<i64> {
    let mut expected = Vec::new();
    for_each_index(&dest.dims, |kept| {
        let mut acc = init;
        for k in 0..src.dims[axis] {
            let mut index = kept.to_vec();
            index.insert(axis, k);
            acc = acc.wrapping_add(src_data[offset_of(src, &index)]);
        }
        expected.push(acc);
    });
    expected
}

fn assert_matches(dest_data: &[i64], dest: &Layout, expected: &[i64], label: &str) {
    let mut iter = expected.iter();
    for_each_index(&dest.dims, |kept| {
        let offset = offset_of(dest, kept);
        assert_eq!(
            dest_data[offset],
            *iter.next().unwrap(),
            "{label}: destination index {kept:?}"
        );
    });
    assert!(
        iter.next().is_none(),
        "{label}: expected longer than layout"
    );
}

fn src_data_for(src: &Layout) -> Vec<i64> {
    (0..src.len)
        .map(|i| (i as i64).wrapping_mul(7).wrapping_sub(3))
        .collect()
}

/// Runs `reduce_axis_into` into `dest` serially and checks the naive result.
fn check_case(src: &Layout, dest: &Layout, axis: usize, init: i64, label: &str) {
    let src_data = src_data_for(src);
    let src_view = StridedView::<i64>::new(&src_data, &src.dims, &src.strides, src.offset).unwrap();
    let expected = expected_values(&src_data, src, dest, axis, init);
    let mut dest_data = vec![i64::MIN; dest.len];
    {
        let mut dest_view =
            StridedViewMut::<i64>::new(&mut dest_data, &dest.dims, &dest.strides, dest.offset)
                .unwrap();
        reduce_axis_into(
            &mut dest_view,
            &src_view,
            axis,
            |x| x,
            i64::wrapping_add,
            init,
            Exec::serial(),
        )
        .unwrap();
    }
    assert_matches(&dest_data, dest, &expected, label);
}

// ============================================================================
// Validation errors of `reduce_axis_into`
// ============================================================================

#[test]
fn reduce_axis_into_rejects_axis_out_of_range() {
    let src_data = [0i64; 6];
    let src = StridedView::<i64>::new(&src_data, &[2, 3], &[3, 1], 0).unwrap();
    let mut dest_data = [0i64; 3];
    let mut dest = StridedViewMut::new(&mut dest_data, &[3], &[1], 0).unwrap();
    let error = reduce_axis_into(
        &mut dest,
        &src,
        2,
        |x| x,
        i64::wrapping_add,
        0,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StridedError::InvalidAxis { axis: 2, rank: 2 }
    ));
}

#[test]
fn reduce_axis_into_rejects_multi_element_full_reduction() {
    // A rank-1 source reduces to a single output element.
    let src_data = [1i64, 2, 3];
    let src = StridedView::<i64>::new(&src_data, &[3], &[1], 0).unwrap();
    let mut dest_data = [0i64; 2];
    let mut dest = StridedViewMut::new(&mut dest_data, &[2], &[1], 0).unwrap();
    let error = reduce_axis_into(
        &mut dest,
        &src,
        0,
        |x| x,
        i64::wrapping_add,
        0,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StridedError::ShapeMismatch(ref dims, ref expected)
            if dims == &[2] && expected == &[1]
    ));
}

#[test]
fn reduce_axis_into_rejects_destination_rank_mismatch() {
    // Rank-3 source reduced along axis 0 keeps two axes; a rank-1 dest
    // cannot hold them.
    let src_data = [0i64; 24];
    let src = StridedView::<i64>::new(&src_data, &[2, 3, 4], &[1, 2, 6], 0).unwrap();
    let mut dest_data = [0i64; 3];
    let mut dest = StridedViewMut::new(&mut dest_data, &[3], &[1], 0).unwrap();
    let error = reduce_axis_into(
        &mut dest,
        &src,
        0,
        |x| x,
        i64::wrapping_add,
        0,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(error, StridedError::RankMismatch(1, 2)));
}

#[test]
fn reduce_axis_into_rejects_destination_shape_mismatch() {
    // Rank-2 [2, 3] reduced along axis 0 expects a [3] destination.
    let src_data = [0i64; 6];
    let src = StridedView::<i64>::new(&src_data, &[2, 3], &[3, 1], 0).unwrap();
    let mut dest_data = [0i64; 4];
    let mut dest = StridedViewMut::new(&mut dest_data, &[4], &[1], 0).unwrap();
    let error = reduce_axis_into(
        &mut dest,
        &src,
        0,
        |x| x,
        i64::wrapping_add,
        0,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        StridedError::ShapeMismatch(ref dims, ref expected)
            if dims == &[4] && expected == &[3]
    ));
}

#[test]
fn reduce_axis_into_rejects_non_injective_destination() {
    // A broadcast destination writes several outputs to one element.
    let src_data = [0i64; 6];
    let src = StridedView::<i64>::new(&src_data, &[2, 3], &[3, 1], 0).unwrap();
    let mut dest_data = [0i64; 1];
    let mut dest = StridedViewMut::new(&mut dest_data, &[3], &[0], 0).unwrap();
    let error = reduce_axis_into(
        &mut dest,
        &src,
        0,
        |x| x,
        i64::wrapping_add,
        0,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(error, StridedError::NonInjectiveOutputLayout));
}

// ============================================================================
// Strided destination, serial: sweep, fold and multi-axis carry
// ============================================================================

#[test]
fn strided_destination_sweeps_contiguous_columns_rank3() {
    // Source [16, 8, 4] contiguous, reduced along axis 1. The kept axes are
    // (axis 0, stride 1) and (axis 2, stride 128), both unit lead strides on
    // the source and destination, so the column sweep runs. The padded
    // destination stride 19 keeps the destination non-column-major, which is
    // what routes it to the strided cursor.
    let src = layout(&[16, 8, 4], &[1, 16, 128], 0, 16 * 8 * 4);
    let dest = layout(&[16, 4], &[1, 19], 0, 15 + 3 * 19 + 1);
    check_case(&src, &dest, 1, 0, "sweep rank3");
}

#[test]
fn strided_destination_folds_each_output_rank3() {
    // Same source, but the destination lead stride is 2, so the sweep is
    // disabled and every output folds its own reduced column.
    let src = layout(&[16, 8, 4], &[1, 16, 128], 0, 16 * 8 * 4);
    let dest = layout(&[16, 4], &[2, 35], 0, 15 * 2 + 3 * 35 + 1);
    check_case(&src, &dest, 1, 0, "fold rank3");
}

#[test]
fn strided_destination_folds_each_output_rank2() {
    // Rank-2 destination with a lead stride of 3: non-linear, so the strided
    // cursor folds each output separately.
    let src = layout(&[16, 8], &[1, 16], 0, 16 * 8);
    let dest = layout(&[16], &[3], 0, 15 * 3 + 1);
    check_case(&src, &dest, 1, 0, "fold rank2");
}

#[test]
fn strided_destination_fold_over_unit_stride_axis() {
    // The reduced source axis has stride 1, so the per-output fold uses the
    // contiguous lane fold. The kept axis (stride 8) is not unit stride, so
    // the sweep stays off.
    let src = layout(&[8, 16], &[1, 8], 0, 8 * 16);
    let dest = layout(&[16], &[3], 0, 15 * 3 + 1);
    check_case(&src, &dest, 0, 0, "fold unit axis");
    // Same layout with a non-zero init and a non-additive identity check.
    check_case(&src, &dest, 0, 12345, "fold unit axis init");
}

#[test]
fn strided_destination_carries_across_kept_axes() {
    // A small lead extent forces a carry into the second kept axis after
    // every lead row; the folded sum is still exact.
    let src = layout(&[3, 5, 4], &[1, 3, 15], 0, 3 * 5 * 4);
    let dest = layout(&[3, 4], &[1, 7], 0, 2 + 3 * 7 + 1);
    check_case(&src, &dest, 1, 0, "carry rank3");
}

#[test]
fn strided_destination_spills_cursor_above_inline_rank() {
    // Nine kept axes exceed `INLINE_RANK`, so the strided cursor spills its
    // coordinates to the heap. The separated destination strides keep the
    // layout injective.
    let src = layout(
        &[1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        &[1, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        0,
        512,
    );
    let dest_strides = [1isize, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000];
    let dest_len: usize = 1 + 1000 + 2000 + 4000 + 8000 + 16000 + 32000 + 64000 + 128000 + 1;
    let dest = layout(&[2; 9], &dest_strides, 0, dest_len);
    check_case(&src, &dest, 0, 0, "spill rank9");
}

// ============================================================================
// Empty reduced axis: the fill path
// ============================================================================

#[test]
fn empty_reduced_axis_skips_an_empty_destination() {
    // Source [0, 0] reduced along axis 0 keeps an empty axis, so the
    // destination has no element to fill and no source pointer is formed.
    let src_data: [i64; 0] = [];
    let src = StridedView::<i64>::new(&src_data, &[0, 0], &[0, 1], 0).unwrap();
    let mut dest_data: [i64; 0] = [];
    let mut dest = StridedViewMut::new(&mut dest_data, &[0], &[1], 0).unwrap();
    reduce_axis_into(
        &mut dest,
        &src,
        0,
        |x| x,
        i64::wrapping_add,
        9,
        Exec::serial(),
    )
    .unwrap();
}

#[test]
fn empty_reduced_axis_fills_a_rank0_destination() {
    // A rank-1 empty source reduces to a single rank-0 output.
    let src_data: [i64; 0] = [];
    let src = StridedView::<i64>::new(&src_data, &[0], &[1], 0).unwrap();
    let mut dest_data = [123i64];
    let mut dest = StridedViewMut::new(&mut dest_data, &[], &[], 0).unwrap();
    reduce_axis_into(
        &mut dest,
        &src,
        0,
        |x| x,
        i64::wrapping_add,
        77,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(dest_data, [77]);
}

#[test]
fn empty_reduced_axis_spills_the_fill_cursor_above_inline_rank() {
    // Ten source axes with an empty reduced axis and nine non-empty kept axes
    // give a rank-9 destination; the fill odometer spills to the heap.
    let src_data: [i64; 0] = [];
    let src =
        StridedView::<i64>::new(&src_data, &[0, 2, 2, 2, 2, 2, 2, 2, 2, 2], &[1; 10], 0).unwrap();
    let mut dest_data = vec![0i64; 512];
    let dest_dims = [2usize; 9];
    let mut dest_strides = [0isize; 9];
    let mut step = 1isize;
    for (stride, &dim) in dest_strides.iter_mut().zip(dest_dims.iter()) {
        *stride = step;
        step *= dim as isize;
    }
    let mut dest = StridedViewMut::new(&mut dest_data, &dest_dims, &dest_strides, 0).unwrap();
    reduce_axis_into(
        &mut dest,
        &src,
        0,
        |x| x,
        i64::wrapping_add,
        5,
        Exec::serial(),
    )
    .unwrap();
    assert!(dest_data.iter().all(|&value| value == 5));
}

// ============================================================================
// Strided destination, parallel above the threading threshold
// ============================================================================

#[cfg(feature = "parallel")]
#[test]
fn strided_destination_splits_across_two_workers() {
    // 32 * 32 outputs over a 64-element reduced axis exceeds MINTHREADLENGTH
    // (1 << 15) by enough to open a second worker.
    let (n, m, k) = (32usize, 64usize, 32usize);
    let src = layout(&[n, m, k], &[1, n as isize, (n * m) as isize], 0, n * m * k);
    let dest = layout(
        &[n, k],
        &[1, (n + 3) as isize],
        0,
        (n - 1) + (k - 1) * (n + 3) + 1,
    );
    assert!(dest.dims[0] * dest.dims[1] * m > 1 << 15);

    let src_data = src_data_for(&src);
    let src_view = StridedView::<i64>::new(&src_data, &src.dims, &src.strides, src.offset).unwrap();
    let expected = expected_values(&src_data, &src, &dest, 1, 0);

    let mut dest_data = vec![i64::MIN; dest.len];
    {
        let budget = Budget::Pooled(2);
        let mut dest_view =
            StridedViewMut::<i64>::new(&mut dest_data, &dest.dims, &dest.strides, 0).unwrap();
        reduce_axis_into(
            &mut dest_view,
            &src_view,
            1,
            |x| x,
            i64::wrapping_add,
            0,
            budget.exec(),
        )
        .unwrap();
    }
    assert_matches(&dest_data, &dest, &expected, "parallel strided sweep");
}
