use super::*;

#[test]
fn test_index_order() {
    // strides [4, 1, 2]: order is [3, 1, 2] (4 is largest, 1 is smallest, 2 is middle)
    let order = index_order(&[4, 1, 2]);
    assert_eq!(order.as_slice(), &[3, 1, 2]);
}

#[test]
fn test_index_order_with_zero() {
    // Zero strides have order 1
    let order = index_order(&[4, 0, 2]);
    assert_eq!(order.as_slice(), &[2, 1, 1]);
}

#[test]
fn test_index_order_negative_strides() {
    let order = index_order(&[-4, 1, -2]);
    assert_eq!(order.as_slice(), &[3, 1, 2]);
}

#[test]
fn test_index_order_tied_strides() {
    let order = index_order(&[2, 2, 1]);
    assert_eq!(order.as_slice(), &[2, 2, 1]);
}

#[test]
fn test_index_order_all_same() {
    let order = index_order(&[3, 3, 3]);
    assert_eq!(order.as_slice(), &[1, 1, 1]);
}

#[test]
fn index_order_into_matches_index_order() {
    let strides = [-4isize, 0, 2, 2, 1, -9, 0, 3, 7, 11];
    let mut out = [0usize; 10];
    index_order_into(&strides, &mut out).unwrap();
    assert_eq!(out[..], index_order(&strides)[..]);
}

#[test]
fn index_order_into_rejects_a_length_mismatch_without_writing() {
    let mut out = [7usize; 2];
    assert!(matches!(
        index_order_into(&[4, 1, 2], &mut out),
        Err(crate::StridedError::StrideLengthMismatch)
    ));
    assert_eq!(out, [7, 7]);
}
