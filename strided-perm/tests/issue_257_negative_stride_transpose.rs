//! Regression tests for issue #257: a source whose smallest-|stride| axis is
//! reversed (stride -1) must not enter the unit-stride Transpose micro-kernel.
//! Every case is checked against a naive coordinate walk, serially and through
//! the parallel executor.

use strided_perm::copy_into;
use strided_view::{StridedView, StridedViewMut};

fn expected_row_major(
    source: &[f64],
    dims: &[usize; 2],
    strides: &[isize; 2],
    offset: isize,
) -> Vec<f64> {
    let mut out = vec![0.0; dims[0] * dims[1]];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let idx = offset + i as isize * strides[0] + j as isize * strides[1];
            out[i * dims[1] + j] = source[idx as usize];
        }
    }
    out
}

fn check(dims: [usize; 2], strides: [isize; 2], offset: isize, source_len: usize) {
    let source: Vec<f64> = (0..source_len).map(|i| i as f64).collect();
    let expected = expected_row_major(&source, &dims, &strides, offset);
    let src = StridedView::new(&source, &dims, &strides, offset).unwrap();
    let dst_strides = [dims[1] as isize, 1];

    let mut out = vec![f64::NAN; dims[0] * dims[1]];
    copy_into(
        &mut StridedViewMut::new(&mut out, &dims, &dst_strides, 0).unwrap(),
        &src,
    )
    .unwrap();
    assert_eq!(out, expected, "serial dims={dims:?} strides={strides:?}");

    #[cfg(feature = "parallel")]
    {
        out.fill(f64::NAN);
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                strided_perm::copy_into_par(
                    &mut StridedViewMut::new(&mut out, &dims, &dst_strides, 0).unwrap(),
                    &src,
                )
                .unwrap();
            });
        assert_eq!(out, expected, "parallel dims={dims:?} strides={strides:?}");
    }
}

#[test]
fn reversed_inner_source_axis_issue_reproducer() {
    // The exact layout from issue #257: 39-element buffer, base 2.
    check([3, 4], [-1, 12], 2, 39);
}

#[test]
fn reversed_inner_source_axis_at_upper_boundary() {
    // The base sits at the last element, so a +1 forward read would leave
    // the buffer.
    check([4, 5], [-1, 4], 3, 20);
}

#[test]
fn reversed_inner_source_axis_tiled_and_parallel() {
    // Large enough for the tiled transpose area and the parallel threshold.
    let n = 256usize;
    let m = 96usize;
    check([n, m], [-1, n as isize], n as isize - 1, n * m);
}
