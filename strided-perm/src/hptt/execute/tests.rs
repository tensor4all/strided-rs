use super::*;
use crate::hptt::plan::build_permute_plan;
use crate::hptt::PlainStore;

#[test]
fn test_execute_identity_copy() {
    let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut dst = vec![0.0f64; 6];
    let plan = build_permute_plan(&[2, 3], &[1, 2], &[1, 2], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is 1 + 2 * 2 = 5 on both
    // sides, below both lengths (6). The token is serial.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }
    assert_eq!(dst, src);
}

#[test]
fn terminal_cursor_advance_may_leave_the_allocation() {
    // dims [2, 2] with source strides [1, 3] over a 5-element buffer: the
    // outer loop's terminal advance reaches offset 6 after its last copy,
    // past the allocation. That cursor is never read, so the advance must
    // wrap; Miri flags the pre-fix `offset`.
    let src = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let mut dst = vec![0.0f64; 4];
    let plan = build_permute_plan(&[2, 2], &[1, 3], &[1, 2], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest source offset is 1 + 3 = 4 < 5 and
    // the largest destination offset 1 + 2 = 3 < 4. The token is serial.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }
    // dst(i, j) = src(i + 3j) in col-major dst: [1, 2, 4, 5].
    assert_eq!(dst, vec![1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn test_execute_transpose_2d() {
    // src [3, 2] col-major: [1,2,3,4,5,6]
    // Permuted view: dims [2, 3], strides [3, 1]
    // dst col-major [2, 3]: strides [1, 2]
    let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut dst = vec![0.0f64; 6];
    let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is 3 + 2 = 5 (source) and 1
    // + 4 = 5 (destination), below both lengths (6). The token is serial.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }
    // Expected: dst = [1, 4, 2, 5, 3, 6]
    assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_execute_3d_permute() {
    // src [2,3,4] col-major, permute [2,0,1]
    let dims = [2usize, 3, 4];
    let total: usize = dims.iter().product();
    let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; total];

    // Permuted: dims [4,2,3], strides [6,1,2], dst col-major [1,4,8]
    let plan = build_permute_plan(&[4, 2, 3], &[6, 1, 2], &[1, 4, 8], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is 18 + 1 + 4 = 23 (source)
    // and 3 + 4 + 16 = 23 (destination), below both lengths (24). The token is serial.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }

    for k in 0..4 {
        for i in 0..2 {
            for j in 0..3 {
                let dst_idx = k + i * 4 + j * 8;
                let src_idx = i + j * 2 + k * 6;
                assert_eq!(
                    dst[dst_idx], src[src_idx],
                    "mismatch at k={k}, i={i}, j={j}"
                );
            }
        }
    }
}

#[test]
fn test_execute_4d_permute() {
    let dims = [2usize, 3, 4, 5];
    let total: usize = dims.iter().product();
    let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; total];

    // Permuted [3,1,0,2]: dims [5,3,2,4], strides [24,2,1,6], dst [1,5,15,30]
    let plan = build_permute_plan(&[5, 3, 2, 4], &[24, 2, 1, 6], &[1, 5, 15, 30], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is 96 + 4 + 1 + 18 = 119
    // (source) and 4 + 10 + 15 + 90 = 119 (destination), below both lengths (120). The token is
    // serial.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }

    for i0 in 0..5 {
        for i1 in 0..3 {
            for i2 in 0..2 {
                for i3 in 0..4 {
                    let src_idx = i0 * 24 + i1 * 2 + i2 + i3 * 6;
                    let dst_idx = i0 + i1 * 5 + i2 * 15 + i3 * 30;
                    assert_eq!(
                        dst[dst_idx], src[src_idx],
                        "4D mismatch at ({i0},{i1},{i2},{i3})"
                    );
                }
            }
        }
    }
}

#[test]
fn test_execute_5d_permute() {
    let dims = [2usize, 2, 2, 2, 3];
    let total: usize = dims.iter().product();
    let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; total];

    // Permuted [4,0,1,2,3]: dims [3,2,2,2,2], strides [16,1,2,4,8], dst [1,3,6,12,24]
    let plan =
        build_permute_plan(&[3, 2, 2, 2, 2], &[16, 1, 2, 4, 8], &[1, 3, 6, 12, 24], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is 32 + 1 + 2 + 4 + 8 = 47
    // (source) and 2 + 3 + 6 + 12 + 24 = 47 (destination), below both lengths (48). The token is
    // serial.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }

    for i0 in 0..3 {
        for i1 in 0..2 {
            for i2 in 0..2 {
                for i3 in 0..2 {
                    for i4 in 0..2 {
                        let src_idx = i0 * 16 + i1 + i2 * 2 + i3 * 4 + i4 * 8;
                        let dst_idx = i0 + i1 * 3 + i2 * 6 + i3 * 12 + i4 * 24;
                        assert_eq!(
                            dst[dst_idx], src[src_idx],
                            "5D mismatch at ({i0},{i1},{i2},{i3},{i4})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_execute_rank0_scalar() {
    let src = [42.0f64];
    let mut dst = vec![0.0f64];
    let plan = build_permute_plan(&[], &[], &[], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; a rank-0 plan reaches only offset 0, which both
    // one-element buffers hold. The token is serial.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }
    assert_eq!(dst[0], 42.0);
}

#[test]
fn native_float_kind_rejects_weakly_aligned_same_size_types() {
    // Exact primitives keep the fast path.
    assert_eq!(native_float_kind::<f64>(), Some(8));
    assert_eq!(native_float_kind::<i64>(), Some(8));
    assert_eq!(native_float_kind::<u64>(), Some(8));
    assert_eq!(native_float_kind::<f32>(), Some(4));
    assert_eq!(native_float_kind::<i32>(), Some(4));
    assert_eq!(native_float_kind::<u32>(), Some(4));
    // Size-only matches must not qualify: complex/array/packed structs.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct EightAlign4([u32; 2]);
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct EightAlign2([u16; 4]);
    #[repr(C, packed)]
    #[derive(Clone, Copy)]
    struct EightAlign1([u8; 8]);
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FourAlign1([u8; 4]);
    assert_eq!(std::mem::size_of::<EightAlign4>(), 8);
    assert_eq!(native_float_kind::<EightAlign4>(), None);
    assert_eq!(std::mem::size_of::<EightAlign2>(), 8);
    assert_eq!(native_float_kind::<EightAlign2>(), None);
    assert_eq!(std::mem::size_of::<EightAlign1>(), 8);
    assert_eq!(native_float_kind::<EightAlign1>(), None);
    assert_eq!(std::mem::size_of::<FourAlign1>(), 4);
    assert_eq!(native_float_kind::<FourAlign1>(), None);
}

fn pool(threads: usize) -> threadteam::Pool {
    threadteam::Pool::builder()
        .threads(threads)
        .build()
        .unwrap()
}

#[test]
fn test_execute_par_transpose_2d() {
    let pool = pool(2);
    let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut dst = vec![0.0f64; 6];
    let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is 3 + 2 = 5 (source) and 1
    // + 4 = 5 (destination), below both lengths (6). The destination layout is column-major,
    // hence injective, so the parallel token is allowed.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::new(&pool, 2).unwrap(),
        );
    }
    assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_execute_par_large() {
    let pool = pool(4);
    let n = 256;
    let total = n * n * n;
    let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; total];

    // [256, 256, 256] col-major, transpose [2, 0, 1]
    let plan = build_permute_plan(&[n, n, n], &[65536, 1, 256], &[1, 256, 65536], 8).unwrap();
    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is 255 * (65536 + 1 + 256) =
    // n^3 - 1 on both sides, below both lengths (n^3). The destination layout is column-major,
    // hence injective, so the parallel token is allowed.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            &plan,
            Exec::new(&pool, 4).unwrap(),
        );
    }

    for i0 in [0, 1, 127, 255] {
        for i1 in [0, 1, 127, 255] {
            for i2 in [0, 1, 127, 255] {
                let dst_idx = i0 + i1 * n + i2 * n * n;
                let src_idx = i0 * 65536 + i1 + i2 * 256;
                assert_eq!(
                    dst[dst_idx], src[src_idx],
                    "mismatch at i0={i0}, i1={i1}, i2={i2}"
                );
            }
        }
    }
}

#[test]
fn test_transpose_tile_rows_cover_rootless_3d_transpose() {
    let n = 256;
    let plan = build_permute_plan(&[n, n, n], &[65536, 1, 256], &[1, 256, 65536], 8).unwrap();
    assert!(plan.levels.is_empty());

    let ExecMode::Transpose { dim_a, dim_b } = plan.mode else {
        panic!("expected transpose mode");
    };
    assert_eq!(
        transpose_tile_rows(plan.fused_dims[dim_a], plan.block),
        4096
    );
    assert_eq!(transpose_tile_rows(plan.fused_dims[dim_b], plan.block), 16);
}

#[test]
fn test_execute_budget_one_token_matches_serial_on_the_calling_thread() {
    let pool = pool(4);
    let n = 128;
    let total = n * n;
    let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
    let mut dst_one = vec![0.0f64; total];
    let mut dst_serial = vec![0.0f64; total];
    let plan = build_permute_plan(&[n, n], &[n as isize, 1], &[1, n as isize], 8).unwrap();

    // SAFETY: (execute.rs:117) `src` and `dst` are separate live buffers, and the plan is
    // `build_permute_plan` over the layout below; the largest offset is (n - 1) * (n + 1) = n^2 -
    // 1 on both sides, below each buffer's length (n^2); the two calls write separate
    // destinations. The destination layout is column-major, hence injective, so the parallel
    // token is allowed.
    unsafe {
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst_one.as_mut_ptr(),
            &plan,
            Exec::new(&pool, 1).unwrap(),
        );
        execute_permute_blocked::<_, PlainStore>(
            src.as_ptr(),
            dst_serial.as_mut_ptr(),
            &plan,
            Exec::serial(),
        );
    }

    assert_eq!(dst_one, dst_serial);
}
