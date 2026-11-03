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
