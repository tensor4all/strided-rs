use super::*;

#[test]
fn test_find_stride1_dim_basic() {
    assert_eq!(find_stride1_dim(&[4, 5], &[1, 4]), 0);
    assert_eq!(find_stride1_dim(&[4, 5], &[5, 1]), 1);
}

#[test]
fn test_find_stride1_dim_skips_size1() {
    // dim 0 has stride 1 but size 1 — should pick dim 1
    assert_eq!(find_stride1_dim(&[1, 5], &[1, 2]), 1);
}

#[test]
fn test_build_plan_identity() {
    // Identity: src and dst both col-major → fuses to single dim → ConstStride1
    let plan = build_permute_plan(&[2, 3, 4], &[1, 2, 6], &[1, 2, 6], 8).unwrap();
    assert_eq!(&plan.fused_dims[..], &[24]);
    assert!(matches!(plan.mode, ExecMode::ConstStride1 { .. }));
}

#[test]
fn test_build_plan_transpose_2d() {
    // 2D transpose: src [1, 4], dst [5, 1]
    let plan = build_permute_plan(&[4, 5], &[1, 4], &[5, 1], 8).unwrap();
    assert_eq!(&plan.fused_dims[..], &[4, 5]);
    match plan.mode {
        ExecMode::Transpose { dim_a, dim_b } => {
            assert_eq!(dim_a, 0); // src stride-1
            assert_eq!(dim_b, 1); // dst stride-1
        }
        _ => panic!("expected Transpose mode"),
    }
    assert_eq!(plan.block, 16); // f64 BLOCK
    assert_eq!(plan.lda_inner, 4); // src stride along dim_b
    assert_eq!(plan.ldb_inner, 5); // dst stride along dim_a
                                   // No loop nodes (only 2 dims, both consumed by macro_kernel)
    assert!(plan.levels.is_empty());
}

#[test]
fn test_build_plan_3d_permute() {
    // 3D: dims [4,2,3], src strides [6,1,2], dst [1,4,8]
    // Bilateral fusion: dims 1-2 fuse (src: 2*1=2 == strides[2], dst: 2*4=8 == strides[2])
    // After fusion: dims [4, 6], src [6, 1], dst [1, 4]
    let plan = build_permute_plan(&[4, 2, 3], &[6, 1, 2], &[1, 4, 8], 8).unwrap();
    assert_eq!(&plan.fused_dims[..], &[4, 6]);
    match plan.mode {
        ExecMode::Transpose { dim_a, dim_b } => {
            // dim_a: min |src_stride| → dim 1 (stride 1)
            assert_eq!(dim_a, 1);
            // dim_b: min |dst_stride| → dim 0 (stride 1)
            assert_eq!(dim_b, 0);
        }
        _ => panic!("expected Transpose mode"),
    }
    // Only 2 fused dims, both consumed by macro_kernel → no outer loops
    assert!(plan.levels.is_empty());
}

#[test]
fn test_build_plan_scattered_strides() {
    // Simplified scattered case: 4 dims of size 2
    let dims = vec![2, 2, 2, 2];
    let src_strides = vec![1, 8, 2, 4]; // scattered
    let dst_strides = vec![1, 2, 4, 8]; // col-major

    let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8).unwrap();

    // Bilateral fusion: dims 2-3 fuse (src: 2→4 contiguous, dst: 4→8 contiguous)
    // Result: 3 fused dims
    assert_eq!(plan.fused_dims.len(), 3);

    // dim_a and dim_b should be identified correctly
    match plan.mode {
        ExecMode::Transpose { .. } | ExecMode::ConstStride1 { .. } => {
            // After bilateral fusion, the mode depends on which dims fuse
        }
        _ => panic!("unexpected mode"),
    }
}

#[test]
fn test_build_plan_step408_uses_grouped_tile() {
    let dims = vec![2usize; 24];
    let src_strides = vec![
        16, 1024, 8388608, 4096, 1048576, 1, 8, 131072, 2, 4, 64, 128, 256, 512, 2048, 8192, 16384,
        32768, 262144, 2097152, 4194304, 32, 65536, 524288,
    ];
    let dst_strides = (0..24).map(|i| 1isize << i).collect::<Vec<_>>();

    let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8).unwrap();

    assert!(matches!(plan.mode, ExecMode::GroupedTranspose));
    let grouped = plan.grouped_tile.as_ref().expect("grouped tile plan");
    assert!(grouped.group_a.total_len() >= 16);
    assert!(grouped.group_b.total_len() >= 16);
    assert!(grouped.group_a.total_len() * grouped.group_b.total_len() >= 256);
}

#[test]
fn test_build_plan_rank8_scattered_uses_inline_grouped_tile() {
    // The rank-24 grouped case capped at the inline rank. Both offset
    // tables must fit the explicit grouped bound so `copy_into` allocates
    // nothing at rank 8.
    let dims = vec![2usize; 8];
    let src_strides = [16isize, 1024, 8388608, 4096, 1048576, 1, 8, 131072];
    let dst_strides = [1isize, 2, 4, 8, 16, 32, 64, 128];

    let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8).unwrap();

    assert!(matches!(plan.mode, ExecMode::GroupedTranspose));
    let grouped = plan.grouped_tile.as_ref().expect("grouped tile plan");
    assert!(grouped.group_a.outer.is_empty() && grouped.group_b.outer.is_empty());
    assert!(grouped.group_a.len >= 4 && grouped.group_b.len >= 4);
    assert!(!plan.levels.is_empty());
}

#[test]
fn test_build_plan_splits_groups_above_the_table_bound() {
    // tensor4all/strided-rs#337: groups of 100 and 120 elements keep
    // tables of at most GROUP_OFFSETS_INLINE entries and cover the rest
    // with outer loops; the group as a whole is unchanged.
    let cases: [([usize; 8], [isize; 8], [isize; 8]); 3] = [
        (
            [101, 2, 2, 2, 2, 2, 2, 2],
            [1, 102, 1000, 1001, 1002, 1003, 1004, 1005],
            [128, 1, 2, 4, 8, 16, 32, 64],
        ),
        (
            [100, 2, 2, 2, 2, 2, 2, 2],
            [1, 101, 1000, 1001, 1002, 1003, 1004, 1005],
            [128, 1, 2, 4, 8, 16, 32, 64],
        ),
        (
            [2, 3, 4, 5, 6, 7, 8, 9],
            [40320, 2, 6, 24, 120, 720, 5040, 1],
            [1, 2, 6, 24, 120, 720, 5040, 40320],
        ),
    ];
    for (dims, src_strides, dst_strides) in cases {
        let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8).unwrap();
        assert!(matches!(plan.mode, ExecMode::GroupedTranspose), "{dims:?}");
        let grouped = plan.grouped_tile.as_ref().expect("grouped tile plan");
        let groups = [&grouped.group_a, &grouped.group_b];
        assert!(groups.iter().any(|g| !g.outer.is_empty()), "{dims:?}");
        for group in groups {
            assert!(group.len <= GROUP_OFFSETS_INLINE);
            assert_eq!(group.src_offsets.len(), group.len);
        }
    }
    // A prime 101-element axis has no table divisor and is walked whole;
    // the 100-element axis is divided as 50 table steps times 2.
    let group_a = |case: usize| {
        let (dims, src, dst) = &cases[case];
        let plan = build_permute_plan(dims, src, dst, 8).unwrap();
        plan.grouped_tile.unwrap().group_a
    };
    let prime = group_a(0);
    assert_eq!(
        (prime.len, prime.outer.as_slice()),
        (1, &[(101, 1, 128)][..])
    );
    let split = group_a(1);
    assert_eq!(
        (split.len, split.outer.as_slice()),
        (50, &[(2, 50, 6400)][..])
    );
}

#[test]
fn test_build_plan_large_3d_transpose_uses_legacy_tile() {
    let dims = vec![256usize, 256, 256];
    let src_strides = vec![256isize, 1, 65536];
    let dst_strides = vec![1isize, 256, 65536];

    let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8).unwrap();

    assert!(matches!(plan.mode, ExecMode::Transpose { .. }));
    assert!(plan.grouped_tile.is_none());
}

#[test]
fn test_build_plan_rank0() {
    let plan = build_permute_plan(&[], &[], &[], 8).unwrap();
    assert!(matches!(plan.mode, ExecMode::Scalar));
    assert!(plan.levels.is_empty());
}

#[test]
fn test_compute_loop_order_transpose() {
    let dims = [4, 5, 3, 7];
    let src_s = [1isize, 4, 100, 300];
    let dst_s = [35isize, 1, 7, 21];
    // dim_a=0 (min src stride), dim_b=1 (min dst stride)
    let order = compute_loop_order_transpose(&dims, &src_s, &dst_s, 0, 1);
    // Remaining: dims 2 and 3
    // cost[2] = 100 + 7 = 107, cost[3] = 300 + 21 = 321
    // Descending: [3, 2]
    assert_eq!(&order[..], &[3, 2]);
}

#[test]
fn test_build_loop_levels_chain() {
    let dims = [10, 5, 3];
    let src_s = [1isize, 10, 50];
    let dst_s = [15isize, 1, 5];
    let loop_order = vec![2]; // only dim 2 in the loop

    let levels = build_loop_levels(&dims, &src_s, &dst_s, &loop_order);
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].end, 3);
    assert_eq!(levels[0].lda, 50);
    assert_eq!(levels[0].ldb, 5);
    assert!(build_loop_levels(&dims, &src_s, &dst_s, &[]).is_empty());
}
