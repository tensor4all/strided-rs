use super::*;

#[test]
fn test_kernel_inner_block() {
    let dims = vec![2, 4];
    let strides1 = vec![4isize, 1];
    let strides2 = vec![4isize, 1];
    let strides_list: Vec<&[isize]> = vec![&strides1, &strides2];
    let plan = build_plan(&dims, &strides_list, Some(0), 8);

    let mut total_elements = 0usize;
    for_each_inner_block(&dims, &plan, &strides_list, |_offsets, len, _strides| {
        total_elements += len;
        Ok(())
    })
    .unwrap();

    assert_eq!(total_elements, 8);
}

#[test]
fn test_contiguous_layout_row_vs_col() {
    let dims = [3usize, 4];
    let row = [4isize, 1];
    let col = [1isize, 3];
    assert_eq!(
        contiguous_layout(&dims, &row),
        Some(ContiguousLayout::RowMajor)
    );
    assert_eq!(
        contiguous_layout(&dims, &col),
        Some(ContiguousLayout::ColMajor)
    );
    assert!(contiguous_layout(&dims, &row).is_some());
    assert!(contiguous_layout(&dims, &col).is_some());
}

#[test]
fn test_contiguous_layout_ignores_dim1_axes() {
    let dims = [2usize, 1, 3];
    // Middle stride is irrelevant since that axis never varies.
    let strides = [3isize, 999, 1];
    assert_eq!(
        contiguous_layout(&dims, &strides),
        Some(ContiguousLayout::RowMajor)
    );
}

// ---- same_contiguous_layout tests ----

#[test]
fn test_same_contiguous_layout_all_row_major() {
    let dims = [3usize, 4];
    let s1 = [4isize, 1];
    let s2 = [4isize, 1];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s1, &s2]),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_same_contiguous_layout_all_col_major() {
    let dims = [3usize, 4];
    let s1 = [1isize, 3];
    let s2 = [1isize, 3];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s1, &s2]),
        Some(ContiguousLayout::ColMajor)
    );
}

#[test]
fn test_same_contiguous_layout_mixed_layouts() {
    let dims = [3usize, 4];
    let row = [4isize, 1];
    let col = [1isize, 3];
    assert_eq!(same_contiguous_layout(&dims, &[&row, &col]), None);
}

#[test]
fn test_same_contiguous_layout_one_noncontiguous() {
    let dims = [3usize, 4];
    let row = [4isize, 1];
    let bad = [8isize, 2];
    assert_eq!(same_contiguous_layout(&dims, &[&row, &bad]), None);
}

#[test]
fn test_same_contiguous_layout_empty_strides_list() {
    let dims = [3usize, 4];
    let empty: &[&[isize]] = &[];
    assert_eq!(same_contiguous_layout(&dims, empty), None);
}

#[test]
fn test_same_contiguous_layout_single_array() {
    let dims = [3usize, 4];
    let s = [4isize, 1];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s[..]]),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_same_contiguous_layout_many_arrays() {
    let dims = [2usize, 3];
    let s = [3isize, 1];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s[..], &s[..], &s[..], &s[..], &s[..]]),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_same_contiguous_layout_empty_dims() {
    let dims: [usize; 0] = [];
    let s: [isize; 0] = [];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s[..], &s[..]]),
        Some(ContiguousLayout::RowMajor)
    );
}

// ---- sequential_contiguous_layout tests ----

#[test]
fn test_sequential_contiguous_layout_small_array() {
    let dims = [3usize, 4];
    let s1 = [4isize, 1];
    let s2 = [4isize, 1];
    assert_eq!(
        sequential_contiguous_layout(&dims, &[&s1, &s2]),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_sequential_contiguous_layout_noncontiguous() {
    let dims = [3usize, 4];
    let s1 = [4isize, 1];
    let s2 = [8isize, 2];
    assert_eq!(sequential_contiguous_layout(&dims, &[&s1, &s2]), None);
}

#[test]
fn test_sequential_contiguous_layout_col_major() {
    let dims = [3usize, 4];
    let col = [1isize, 3];
    assert_eq!(
        sequential_contiguous_layout(&dims, &[&col]),
        Some(ContiguousLayout::ColMajor)
    );
}

#[test]
fn test_build_plan_fused_compresses() {
    // A contiguous 2x3 column-major array fuses [2,3] -> [6,1] -> compress -> [6]
    let dims = [2usize, 3];
    let strides = [1isize, 2];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
    // After fusion + compression, should be 1D
    assert_eq!(fused_dims, vec![6]);
    assert_eq!(fused_strides.len(), 1);
    assert_eq!(fused_strides[0], vec![1]);
    assert_eq!(plan.block.len(), 1);
}

#[test]
fn test_build_plan_fused_keeps_broadcast_compact_inner_loop() {
    let dims = [16usize, 16, 64, 64];
    let out = [1isize, 16, 256, 16_384];
    let lhs = [1isize, 16, 0, 256];
    let rhs = [0isize, 0, 1, 64];
    let strides_list: Vec<&[isize]> = vec![&out, &lhs, &rhs];

    let (fused_dims, fused_strides, _) = build_plan_fused(&dims, &strides_list, Some(0), 8);

    assert_eq!(fused_dims[0], 256);
    assert_eq!(fused_strides[0][0], 1);
    assert_eq!(fused_strides[1][0], 1);
    assert_eq!(fused_strides[2][0], 0);
}

#[test]
fn test_kernel_nd_iterative_total_elements_match() {
    // Use rank 9 to test kernel_nd_inner_iterative (fallback for rank >= 9)
    let dims = vec![2usize, 2, 2, 2, 2, 2, 2, 2, 3];
    let blocks = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 1];
    let mut stride_val = 1isize;
    let mut sv = Vec::new();
    for &d in &dims {
        sv.push(stride_val);
        stride_val *= d as isize;
    }
    let strides = vec![sv.clone(), sv];
    let mut offsets = vec![0isize, 0isize];
    let mut total = 0usize;

    kernel_nd_inner_iterative(
        &dims,
        &blocks,
        &strides,
        &mut offsets,
        &mut |_off, len, _s| {
            total += len;
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(total, dims.iter().product::<usize>());
    assert_eq!(offsets, vec![0isize, 0isize]);
}

/// Helper: run a macro-generated kernel and verify total elements and offset reset.
fn verify_kernel_total_and_offsets(rank: usize) {
    // Build dims: dim[0] = 3 for block variety, rest = 2
    let mut dims = vec![3usize];
    for _ in 1..rank {
        dims.push(2);
    }
    let blocks = vec![2usize; rank];

    // Column-major strides
    let mut stride_val = 1isize;
    let mut sv = Vec::new();
    for &d in &dims {
        sv.push(stride_val);
        stride_val *= d as isize;
    }
    let strides = vec![sv.clone(), sv];
    let mut offsets = vec![0isize, 0isize];
    let mut total = 0usize;
    let expected: usize = dims.iter().product();

    let result = match rank {
        1 => kernel_1d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        2 => kernel_2d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        3 => kernel_3d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        4 => kernel_4d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        5 => kernel_5d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        6 => kernel_6d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        7 => kernel_7d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        8 => kernel_8d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        _ => panic!("unsupported rank"),
    };
    result.unwrap();
    assert_eq!(total, expected, "rank={rank}: total mismatch");
    assert_eq!(offsets, vec![0, 0], "rank={rank}: offsets not reset");
}

#[test]
fn test_macro_kernels_total_elements_1d() {
    verify_kernel_total_and_offsets(1);
}

#[test]
fn test_macro_kernels_total_elements_2d() {
    verify_kernel_total_and_offsets(2);
}

#[test]
fn test_macro_kernels_total_elements_3d() {
    verify_kernel_total_and_offsets(3);
}

#[test]
fn test_macro_kernels_total_elements_4d() {
    verify_kernel_total_and_offsets(4);
}

#[test]
fn test_macro_kernels_total_elements_5d() {
    verify_kernel_total_and_offsets(5);
}

#[test]
fn test_macro_kernels_total_elements_6d() {
    verify_kernel_total_and_offsets(6);
}

#[test]
fn test_macro_kernels_total_elements_7d() {
    verify_kernel_total_and_offsets(7);
}

#[test]
fn test_macro_kernels_total_elements_8d() {
    verify_kernel_total_and_offsets(8);
}

/// Verify that macro-generated kernels visit every element exactly once
/// by collecting all linear offsets and checking them against the expected set.
fn verify_kernel_visits_all_elements(rank: usize) {
    assert!(rank >= 2 && rank <= 8);
    let mut dims = vec![3usize];
    for _ in 1..rank {
        dims.push(2);
    }
    let blocks = vec![2usize; rank];

    // Column-major strides (single array for simplicity)
    let mut stride_val = 1isize;
    let mut sv = Vec::new();
    for &d in &dims {
        sv.push(stride_val);
        stride_val *= d as isize;
    }
    let strides = vec![sv];

    // Collect all linear offsets visited
    let mut visited = std::collections::HashSet::new();
    let mut offsets = vec![0isize];

    let result = match rank {
        2 => kernel_2d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        3 => kernel_3d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        4 => kernel_4d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        5 => kernel_5d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        6 => kernel_6d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        7 => kernel_7d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        8 => kernel_8d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        _ => unreachable!(),
    };
    result.unwrap();

    // Expected: all offsets 0..total
    let total: usize = dims.iter().product();
    let expected: std::collections::HashSet<isize> = (0..total as isize).collect();
    assert_eq!(
        visited, expected,
        "rank={rank}: not all elements visited exactly once"
    );
}

#[test]
fn test_macro_kernel_5d_visits_all_elements() {
    verify_kernel_visits_all_elements(5);
}

#[test]
fn test_macro_kernel_6d_visits_all_elements() {
    verify_kernel_visits_all_elements(6);
}

#[test]
fn test_macro_kernel_7d_visits_all_elements() {
    verify_kernel_visits_all_elements(7);
}

#[test]
fn test_macro_kernel_8d_visits_all_elements() {
    verify_kernel_visits_all_elements(8);
}
