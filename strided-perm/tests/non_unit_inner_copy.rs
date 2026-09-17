use strided_perm::copy_into;
use strided_view::{StridedView, StridedViewMut};

fn check(dims: &[usize], strides: &[isize], offset: isize, source_len: usize) {
    let source: Vec<f64> = (0..source_len).map(|i| i as f64).collect();
    let mut dest_strides = Vec::new();
    let mut count = 1;
    for &dim in dims {
        dest_strides.push(count as isize);
        count *= dim;
    }
    let expected: Vec<_> = (0..count)
        .map(|mut linear| {
            let mut index = offset;
            for (&dim, &stride) in dims.iter().zip(strides) {
                index += (linear % dim) as isize * stride;
                linear /= dim;
            }
            source[index as usize]
        })
        .collect();
    let src = StridedView::new(&source, dims, strides, offset).unwrap();
    let mut output = vec![f64::NAN; count];
    copy_into(
        &mut StridedViewMut::new(&mut output, dims, &dest_strides, 0).unwrap(),
        &src,
    )
    .unwrap();
    assert_eq!(output, expected);
    #[cfg(feature = "parallel")]
    {
        output.fill(f64::NAN);
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                strided_perm::copy_into_par(
                    &mut StridedViewMut::new(&mut output, dims, &dest_strides, 0).unwrap(),
                    &src,
                )
                .unwrap();
            });
        assert_eq!(output, expected);
    }
}

#[test]
fn non_unit_destination_inner_axis_preserves_holes() {
    let source = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    for (strides, offset, expected) in [
        ([-1, 2], 1, vec![3., 0., 4., 1., 5., 2.]),
        (
            [2, 4],
            0,
            vec![0., -1., 3., -1., 1., -1., 4., -1., 2., -1., 5.],
        ),
    ] {
        let mut output = vec![-1.; expected.len()];
        copy_into(
            &mut StridedViewMut::new(&mut output, &[2, 3], &strides, offset).unwrap(),
            &StridedView::new(&source, &[2, 3], &[3, 1], 0).unwrap(),
        )
        .unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
fn non_unit_inner_axes_preserve_source_coordinates() {
    check(&[3, 2, 2], &[1, 3, 0], 0, 6);
    check(&[2, 3, 2], &[0, 1, 3], 0, 6);
    check(&[2, 3], &[3, -1], 2, 6);
    check(&[2, 3], &[6, 2], 0, 11);
    // Exceed the parallel executor's threshold as well.
    check(&[16384, 3], &[1, 0], 0, 16384);
    check(&[2, 17000], &[17000, -1], 16999, 34000);
}
