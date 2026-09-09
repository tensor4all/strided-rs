use super::*;

fn compact_strides_for_axis_order<const N: usize>(
    dims: [usize; N],
    axis_order: [usize; N],
) -> [isize; N] {
    let mut strides = [0isize; N];
    let mut stride = 1isize;
    for &axis in &axis_order {
        strides[axis] = stride;
        stride *= dims[axis] as isize;
    }
    strides
}

#[test]
fn test_contiguous_mul_range_plan_pure_outer() {
    let dims = [7usize, 11];
    let dst = [1isize, 7];
    let lhs = [1isize, 0];
    let rhs = [0isize, 1];

    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

    assert_eq!(plan.inner_len, 7);
    assert_eq!(plan.row_len, 11);
    assert_eq!(plan.fast_axis, 0);
    assert_eq!(plan.a_fast_stride, 1);
    assert_eq!(plan.b_fast_stride, 0);
    assert_eq!(plan.a_row_stride, 0);
    assert_eq!(plan.b_row_stride, 1);
}

#[test]
fn test_compact_axis_order_accepts_all_rank4_axis_permutations() {
    fn visit(dims: [usize; 4], axes: &mut [usize; 4], pos: usize, count: &mut usize) {
        if pos == axes.len() {
            let dst = compact_strides_for_axis_order(dims, *axes);
            let axis_order = compact_axis_order(&dims, &dst).unwrap();
            assert_eq!(&axis_order[..], &axes[..]);
            *count += 1;
            return;
        }

        for i in pos..axes.len() {
            axes.swap(pos, i);
            visit(dims, axes, pos + 1, count);
            axes.swap(pos, i);
        }
    }

    let dims = [2usize, 3, 5, 7];
    let mut axes = [0usize, 1, 2, 3];
    let mut count = 0usize;
    visit(dims, &mut axes, 0, &mut count);

    assert_eq!(count, 24);
}

#[test]
fn test_compact_axis_order_rejects_strided_layout_with_holes() {
    let dims = [2usize, 3, 5];
    let strides = [1isize, 4, 2];

    assert_eq!(compact_axis_order(&dims, &strides), None);
}

#[test]
fn test_contiguous_mul_range_plan_uses_permuted_compact_output_for_unrelated_shape() {
    let dims = [2usize, 3, 5, 7, 11];
    let dst = compact_strides_for_axis_order(dims, [2usize, 0, 4, 1, 3]);
    let lhs = [5isize, 0, 1, 0, 10];
    let rhs = [0isize, 1, 0, 3, 0];

    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

    assert_eq!(&plan.axis_order[..], &[2, 0, 4, 1, 3]);
    assert_eq!(plan.inner_len, 110);
    assert_eq!(plan.row_len, 3);
    assert_eq!(plan.fast_axis, 2);
    assert_eq!(plan.a_fast_stride, 1);
    assert_eq!(plan.b_fast_stride, 0);
    assert_eq!(plan.a_row_stride, 0);
    assert_eq!(plan.b_row_stride, 1);
    assert_eq!(transposed_scalar_tile_kind(&plan), None);
}

#[test]
fn test_contiguous_mul_range_plan_compact_batched_outer() {
    let dims = [3usize, 5, 7, 11];
    let dst = [1isize, 3, 15, 105];
    let lhs = [1isize, 3, 0, 15];
    let rhs = [0isize, 0, 1, 7];

    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

    assert_eq!(plan.inner_len, 15);
    assert_eq!(plan.row_len, 7);
    assert_eq!(plan.fast_axis, 0);
    assert_eq!(plan.a_fast_stride, 1);
    assert_eq!(plan.b_fast_stride, 0);
    assert_eq!(plan.a_row_stride, 0);
    assert_eq!(plan.b_row_stride, 1);
}

#[test]
fn test_contiguous_mul_range_plan_noncompact_batched_outer() {
    let dims = [5usize, 5, 7, 11];
    let dst = [1isize, 5, 25, 175];
    let lhs = [5isize, 1, 0, 25];
    let rhs = [0isize, 0, 1, 7];

    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

    assert_eq!(plan.inner_len, 5);
    assert_eq!(plan.row_len, 5);
    assert_eq!(plan.fast_axis, 0);
    assert_eq!(plan.a_fast_stride, 5);
    assert_eq!(plan.b_fast_stride, 0);
    assert_eq!(plan.a_row_stride, 1);
    assert_eq!(plan.b_row_stride, 0);
}

#[test]
fn test_contiguous_mul_range_plan_noncompact_row_major_output() {
    let dims = [5usize, 5, 7, 11];
    let dst = [5isize, 1, 25, 175];
    let lhs = [5isize, 1, 0, 25];
    let rhs = [0isize, 0, 1, 7];

    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

    assert_eq!(plan.inner_len, 25);
    assert_eq!(plan.row_len, 7);
    assert_eq!(plan.fast_axis, 1);
    assert_eq!(plan.a_fast_stride, 1);
    assert_eq!(plan.b_fast_stride, 0);
    assert_eq!(plan.a_row_stride, 0);
    assert_eq!(plan.b_row_stride, 1);
    assert_eq!(transposed_scalar_tile_kind(&plan), None);
}

#[test]
fn test_broadcast_strides_for_axes_batched_outer() {
    let target_dims = [3usize, 5, 7, 11];
    let lhs_dims = [3usize, 5, 11];
    let lhs_strides = [3isize, 1, 15];
    let rhs_dims = [7usize, 11];
    let rhs_strides = [1isize, 7];

    let lhs =
        broadcast_strides_for_axes(&lhs_dims, &lhs_strides, &target_dims, &[0, 1, 3]).unwrap();
    let rhs = broadcast_strides_for_axes(&rhs_dims, &rhs_strides, &target_dims, &[2, 3]).unwrap();

    assert_eq!(&lhs[..], &[3, 1, 0, 15]);
    assert_eq!(&rhs[..], &[0, 0, 1, 7]);
}

#[test]
fn test_broadcast_strides_for_axes_uses_zero_stride_for_size_one_source_dim() {
    let target_dims = [8usize, 4];
    let source_dims = [1usize, 4];
    let source_strides = [1isize, 1];

    let strides =
        broadcast_strides_for_axes(&source_dims, &source_strides, &target_dims, &[0, 1]).unwrap();

    assert_eq!(&strides[..], &[0, 1]);
}

#[test]
fn test_transposed_scalar_tile_kind_detects_noncompact_rhs_scalar() {
    let dims = [5usize, 5, 7, 11];
    let dst = [1isize, 5, 25, 175];
    let lhs = [5isize, 1, 0, 25];
    let rhs = [0isize, 0, 1, 7];

    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

    assert_eq!(
        transposed_scalar_tile_kind(&plan),
        Some(TransposedScalarTileKind::RhsScalar)
    );
}

#[test]
fn test_contiguous_mul_outer_cursor_matches_linear_offsets() {
    let dims = [16usize, 16, 64, 64];
    let dst = [1isize, 16, 256, 16_384];
    let lhs = [16isize, 1, 0, 256];
    let rhs = [0isize, 0, 1, 64];
    let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();
    let mut cursor = ContiguousMulOuterCursor::new(&dims, &lhs, &rhs, &plan, 13);
    let block_len = plan.inner_len * plan.row_len;

    for group in 13..80 {
        let index = group * block_len;
        assert_eq!(
            cursor.a_offset,
            strided_offset_for_contiguous_linear_index(&dims, &lhs, &plan.axis_order, index)
        );
        assert_eq!(
            cursor.b_offset,
            strided_offset_for_contiguous_linear_index(&dims, &rhs, &plan.axis_order, index)
        );
        cursor.advance();
    }
}
