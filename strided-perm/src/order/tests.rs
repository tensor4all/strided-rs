use super::*;

#[test]
fn test_compute_order_column_major() {
    let dims = [4usize, 5];
    let strides = [1isize, 4];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let order = compute_order(&dims, &strides_list, Some(0));
    assert_eq!(order[0], 0);
    assert_eq!(order[1], 1);
}

#[test]
fn test_compute_order_row_major() {
    let dims = [4usize, 5];
    let strides = [5isize, 1];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let order = compute_order(&dims, &strides_list, Some(0));
    assert_eq!(order[0], 1);
    assert_eq!(order[1], 0);
}

#[test]
fn test_compute_order_mixed() {
    let dims = [4usize, 5];
    let out_strides = [1isize, 4];
    let in_strides = [5isize, 1];
    let strides_list: Vec<&[isize]> = vec![&out_strides, &in_strides];
    let order = compute_order(&dims, &strides_list, Some(0));
    assert_eq!(order[0], 0);
    assert_eq!(order[1], 1);
}

#[test]
fn test_compute_order_empty() {
    let dims: [usize; 0] = [];
    let strides: [isize; 0] = [];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let order = compute_order(&dims, &strides_list, Some(0));
    assert!(order.is_empty());
}

#[test]
fn test_compute_order_dest_reorder() {
    // dest_index=1: should swap so destination is first (gets 2x weight)
    let dims = [4usize, 5];
    let src_strides = [1isize, 4]; // col-major
    let dst_strides = [5isize, 1]; // row-major
    let strides_list: Vec<&[isize]> = vec![&src_strides, &dst_strides];
    let order = compute_order(&dims, &strides_list, Some(1));
    // With dst (row-major) weighted 2x, dim 1 (stride 1 in dst) should be innermost
    assert_eq!(order.len(), 2);
}

#[test]
fn test_compute_order_no_dest() {
    let dims = [4usize, 5];
    let strides = [1isize, 4];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let order = compute_order(&dims, &strides_list, None);
    assert_eq!(order.len(), 2);
}

#[test]
fn test_compute_order_dest_out_of_bounds() {
    // dest_index >= strides_list.len(): should skip reordering
    let dims = [4usize, 5];
    let strides = [1isize, 4];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let order = compute_order(&dims, &strides_list, Some(5));
    assert_eq!(order.len(), 2);
}

#[test]
fn test_compute_order_empty_strides_list() {
    let dims = [4usize, 5, 6];
    let strides_list: Vec<&[isize]> = vec![];
    let order = compute_order(&dims, &strides_list, None);
    assert_eq!(order, vec![0, 1, 2]);
}

#[test]
fn test_compute_order_3d() {
    // 3D col-major
    let dims = [4usize, 5, 6];
    let strides = [1isize, 4, 20];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let order = compute_order(&dims, &strides_list, Some(0));
    assert_eq!(order.len(), 3);
    // Col-major: innermost should be dim 0
    assert_eq!(order[0], 0);
}
