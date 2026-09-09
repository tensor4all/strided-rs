use super::*;
use crate::StridedArray;

#[test]
fn copy_into_col_major_copies_from_non_col_major_source() {
    let source =
        StridedArray::<usize>::from_fn_row_major(&[3, 4], |index| index[0] * 10 + index[1]);
    let mut destination = StridedArray::<usize>::col_major(&[3, 4]);

    copy_into_col_major(&mut destination.view_mut(), &source.view()).unwrap();

    for row in 0..3 {
        for column in 0..4 {
            assert_eq!(destination.get(&[row, column]), source.get(&[row, column]));
        }
    }
}

#[test]
fn copy_permuted_serial_copies_transposed_view() {
    let source =
        StridedArray::<usize>::from_fn_col_major(&[3, 4], |index| index[0] * 10 + index[1]);
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut destination = StridedArray::<usize>::row_major(&[4, 3]);

    copy_permuted_serial(&mut destination.view_mut(), &transposed).unwrap();

    for row in 0..3 {
        for column in 0..4 {
            assert_eq!(destination.get(&[column, row]), source.get(&[row, column]));
        }
    }
}
