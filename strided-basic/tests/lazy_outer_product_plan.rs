//! Layout planning for lazily ordered outer products.

use strided_basic::{plan_lazy_outer_product, LazyOuterProductLayout, StridedError};

fn plan(
    out: &[usize],
    lhs: (&[usize], &[isize], &[usize]),
    rhs: (&[usize], &[isize], &[usize]),
) -> Result<Option<LazyOuterProductLayout>, StridedError> {
    plan_lazy_outer_product(out, lhs.0, lhs.1, lhs.2, rhs.0, rhs.1, rhs.2)
}

#[test]
fn lhs_prefix_orders_base_by_lhs_physical_strides() {
    let layout = plan(&[2, 3, 4], (&[2, 3], &[3, 1], &[0, 1]), (&[4], &[1], &[2]))
        .unwrap()
        .unwrap();
    assert_eq!(layout.base_dims, [3, 2, 4]);
    assert_eq!(layout.output_strides, [3, 1, 6]);
}

#[test]
fn rhs_prefix_puts_rhs_free_axes_first_in_physical_order() {
    // out[j0, j1, i, b] = lhs[i, b] * rhs[j1, j0, b]; rhs axis 0 (output
    // axis 1) is its fastest axis, so the base stores output axis 1 first.
    let layout = plan(
        &[2, 3, 5, 4],
        (&[5, 4], &[1, 5], &[2, 3]),
        (&[3, 2, 4], &[1, 3, 6], &[1, 0, 3]),
    )
    .unwrap()
    .unwrap();
    assert_eq!(layout.base_dims, [3, 2, 5, 4]);
    assert_eq!(layout.output_strides, [3, 1, 6, 30]);
}

#[test]
fn trailing_group_is_also_reordered() {
    // Leading lhs group is already physical; the trailing rhs group is not.
    let layout = plan(&[3, 2, 4], (&[3], &[1], &[0]), (&[2, 4], &[4, 1], &[1, 2]))
        .unwrap()
        .unwrap();
    assert_eq!(layout.base_dims, [3, 4, 2]);
    assert_eq!(layout.output_strides, [1, 12, 3]);
}

#[test]
fn ineligible_inputs_return_none() {
    let none = |result: Result<Option<LazyOuterProductLayout>, StridedError>| {
        assert_eq!(result.unwrap(), None);
    };
    // Already in physical order.
    none(plan(
        &[2, 3, 4],
        (&[2, 3], &[1, 2], &[0, 1]),
        (&[4], &[1], &[2]),
    ));
    // Free group of size one.
    none(plan(
        &[2, 3, 1],
        (&[2, 3], &[3, 1], &[0, 1]),
        (&[1], &[1], &[2]),
    ));
    // Negative stride.
    none(plan(
        &[2, 3, 4],
        (&[2, 3], &[3, -1], &[0, 1]),
        (&[4], &[1], &[2]),
    ));
    // Extent differs from the output (broadcast extents are not planned).
    none(plan(
        &[2, 3, 4],
        (&[2, 1], &[1, 1], &[0, 1]),
        (&[4], &[1], &[2]),
    ));
    // An output axis covered by neither operand.
    none(plan(
        &[2, 3, 4, 5],
        (&[2, 3], &[3, 1], &[0, 1]),
        (&[4], &[1], &[2]),
    ));
    // Interleaved free groups: [lhs, rhs, lhs].
    none(plan(
        &[2, 4, 3],
        (&[2, 3], &[3, 1], &[0, 2]),
        (&[4], &[1], &[1]),
    ));
    // Batch axis not last.
    none(plan(
        &[5, 2, 3, 4],
        (&[5, 2, 3], &[6, 3, 1], &[0, 1, 2]),
        (&[5, 4], &[1, 5], &[0, 3]),
    ));
    // Duplicate or out-of-range axis maps.
    none(plan(
        &[2, 3, 4],
        (&[2, 3], &[3, 1], &[0, 0]),
        (&[4], &[1], &[2]),
    ));
    none(plan(
        &[2, 3, 4],
        (&[2, 3], &[3, 1], &[0, 7]),
        (&[4], &[1], &[2]),
    ));
}

#[test]
fn malformed_ranks_return_errors() {
    assert!(matches!(
        plan(&[2, 3], (&[2], &[1, 1], &[0]), (&[3], &[1], &[1])),
        Err(StridedError::RankMismatch(1, 2))
    ));
    assert!(matches!(
        plan(&[2, 3], (&[2], &[1], &[0]), (&[3], &[1], &[1, 0])),
        Err(StridedError::RankMismatch(1, 2))
    ));
}
