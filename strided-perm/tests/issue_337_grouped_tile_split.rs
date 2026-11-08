//! Grouped tiles larger than the inline offset-table bound walk the rest of
//! the group as outer loops (tensor4all/strided-rs#337). These layouts take
//! that split path; the copy must match a direct gather in serial and pooled
//! execution.

use strided_perm::copy_into;
use strided_view::{StridedView, StridedViewMut};
use threadteam::{Exec, Pool};

fn check(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) {
    let span = |strides: &[isize]| -> usize {
        1 + dims
            .iter()
            .zip(strides)
            .map(|(&n, &s)| (n - 1) * s as usize)
            .sum::<usize>()
    };
    let source: Vec<f64> = (0..span(src_strides)).map(|i| i as f64).collect();
    let mut expected = vec![f64::NAN; span(dst_strides)];
    let count: usize = dims.iter().product();
    for mut linear in 0..count {
        let (mut s, mut d) = (0isize, 0isize);
        for ((&n, &ss), &ds) in dims.iter().zip(src_strides).zip(dst_strides) {
            s += (linear % n) as isize * ss;
            d += (linear % n) as isize * ds;
            linear /= n;
        }
        expected[d as usize] = source[s as usize];
    }
    let src = StridedView::new(&source, dims, src_strides, 0).unwrap();
    let pool = Pool::builder().threads(2).build().unwrap();
    for exec in [Exec::serial(), Exec::new(&pool, 2).unwrap()] {
        let mut output = vec![f64::NAN; expected.len()];
        let mut dst = StridedViewMut::new(&mut output, dims, dst_strides, 0).unwrap();
        copy_into(&mut dst, &src, exec).unwrap();
        // NaN holes compare by bits.
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&output), bits(&expected), "{dims:?}");
    }
}

#[test]
fn hundred_element_source_group_is_split_into_table_and_outer_loop() {
    check(
        &[100, 2, 2, 2, 2, 2, 2, 2],
        &[1, 101, 1000, 1001, 1002, 1003, 1004, 1005],
        &[128, 1, 2, 4, 8, 16, 32, 64],
    );
}

#[test]
fn prime_extent_group_is_walked_as_an_outer_loop() {
    // 101 has no divisor up to the table bound, so the whole axis is outer.
    check(
        &[101, 2, 2, 2, 2, 2, 2, 2],
        &[1, 102, 1000, 1001, 1002, 1003, 1004, 1005],
        &[128, 1, 2, 4, 8, 16, 32, 64],
    );
}

#[test]
fn mixed_extent_rank8_swap_splits_its_destination_group() {
    let dims = [2usize, 3, 4, 5, 6, 7, 8, 9];
    let mut dst_strides = [0isize; 8];
    let mut stride = 1;
    for (s, &n) in dst_strides.iter_mut().zip(&dims) {
        *s = stride;
        stride *= n as isize;
    }
    let mut src_strides = dst_strides;
    src_strides.swap(0, 7);
    check(&dims, &src_strides, &dst_strides);
}
