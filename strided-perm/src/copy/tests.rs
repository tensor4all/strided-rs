use super::*;
use strided_view::StridedArray;

#[test]
fn test_copy_into_contiguous() {
    let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut dst = StridedArray::<f64>::col_major(&[2, 3]);
    copy_into(&mut dst.view_mut(), &src.view(), Exec::serial()).unwrap();
    assert_eq!(dst.get(&[0, 0]), 0.0);
    assert_eq!(dst.get(&[1, 2]), 12.0);
}

#[test]
fn test_copy_into_transposed() {
    // src is row-major [3,2], dst is col-major [3,2]
    let src = StridedArray::<f64>::from_parts(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[3, 2],
        &[2, 1], // row-major
        0,
    )
    .unwrap();
    let mut dst = StridedArray::<f64>::col_major(&[3, 2]);
    copy_into(&mut dst.view_mut(), &src.view(), Exec::serial()).unwrap();
    assert_eq!(dst.get(&[0, 0]), 0.0);
    assert_eq!(dst.get(&[0, 1]), 1.0);
    assert_eq!(dst.get(&[1, 0]), 2.0);
    assert_eq!(dst.get(&[2, 1]), 5.0);
}

#[test]
fn test_copy_into_col_major_basic() {
    let src = StridedArray::<f64>::from_fn_col_major(&[4, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut dst = StridedArray::<f64>::col_major(&[4, 3]);
    copy_into_col_major(&mut dst.view_mut(), &src.view(), Exec::serial()).unwrap();
    assert_eq!(dst.get(&[0, 0]), 0.0);
    assert_eq!(dst.get(&[3, 2]), 32.0);
}

#[test]
fn test_copy_into_col_major_permuted_src() {
    // src has scattered strides, dst is col-major
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let src = StridedArray::<f64>::from_parts(
        data,
        &[2, 3, 4],
        &[12, 4, 1], // row-major
        0,
    )
    .unwrap();
    let mut dst = StridedArray::<f64>::col_major(&[2, 3, 4]);
    copy_into_col_major(&mut dst.view_mut(), &src.view(), Exec::serial()).unwrap();
    // Verify element-by-element
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(
                    dst.get(&[i, j, k]),
                    src.get(&[i, j, k]),
                    "mismatch at [{},{},{}]",
                    i,
                    j,
                    k
                );
            }
        }
    }
}

#[test]
fn test_copy_into_grouped_high_rank_binary_permutation() {
    let dims = vec![2usize; 12];
    let total: usize = dims.iter().product();
    let src = StridedArray::<u32>::from_fn_col_major(&dims, |idx| {
        idx.iter()
            .enumerate()
            .map(|(axis, &coord)| (coord as u32) << axis)
            .sum::<u32>()
    });
    let perm = vec![2, 4, 5, 6, 0, 3, 7, 1, 8, 9, 10, 11];
    let src_view = src.view().permute(&perm).unwrap();
    let mut dst = StridedArray::<u32>::col_major(&dims);

    copy_into(&mut dst.view_mut(), &src_view, Exec::serial()).unwrap();

    for flat in 0..total {
        let mut rem = flat;
        let mut src_offset = 0usize;
        for &stride in src_view.strides() {
            let coord = rem & 1;
            rem >>= 1;
            src_offset += coord * stride as usize;
        }
        assert_eq!(dst.data()[flat], src.data()[src_offset], "flat={flat}");
    }
}

#[test]
fn parallel_transpose_splits_the_full_tile_grid_for_a_short_row_extent() {
    // The source's unit-stride extent is the short side of the transpose, so
    // the whole kernel would be a single row tile over `block`, and the budget is
    // two. The parallel entry must split the
    // column tiles too; a row-only split would leave the work on one worker.
    // The result is checked element by element.
    let rows = 20000usize;
    let cols = 3usize;
    let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();
    let src = StridedArray::<f64>::from_parts(data, &[rows, cols], &[cols as isize, 1], 0).unwrap();
    let pool = threadteam::Pool::builder().threads(2).build().unwrap();
    let mut dst = StridedArray::<f64>::col_major(&[rows, cols]);
    let exec = Exec::new(&pool, 2).unwrap();
    copy_into(&mut dst.view_mut(), &src.view(), exec).unwrap();
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(dst.get(&[r, c]), (r * cols + c) as f64);
        }
    }
}

#[test]
fn test_copy_shape_mismatch() {
    let src = StridedArray::<f64>::col_major(&[2, 3]);
    let mut dst = StridedArray::<f64>::col_major(&[3, 2]);
    let result = copy_into(&mut dst.view_mut(), &src.view(), Exec::serial());
    assert!(result.is_err());
}
