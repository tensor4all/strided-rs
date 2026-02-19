//! Execution engine: recursive loop nest dispatching to macro_kernel.
//!
//! Mirrors HPTT C++'s `transpose_int` (lines 602-681) and
//! `transpose_int_constStride1` (lines 683-720).

use crate::hptt::macro_kernel::{
    const_stride1_copy, macro_kernel_f32, macro_kernel_f64, macro_kernel_fallback,
};
use crate::hptt::plan::{ComputeNode, ExecMode, PermutePlan};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Minimum elements to justify multi-threaded execution.
#[cfg(feature = "parallel")]
const MINTHREADLENGTH: usize = 1 << 15; // 32768

/// Execute the permutation plan (single-threaded).
///
/// # Safety
/// - `src` must be valid for reads at all offsets determined by dims/src_strides
/// - `dst` must be valid for writes at all offsets determined by dims/dst_strides
/// - src and dst must not overlap
pub unsafe fn execute_permute_blocked<T: Copy>(src: *const T, dst: *mut T, plan: &PermutePlan) {
    match plan.mode {
        ExecMode::Scalar => {
            *dst = *src;
        }
        ExecMode::ConstStride1 { inner_dim } => {
            let count = plan.fused_dims[inner_dim];
            let src_stride = plan.src_strides[inner_dim];
            let dst_stride = plan.dst_strides[inner_dim];
            match &plan.root {
                Some(root) => {
                    const_stride1_recursive(src, dst, root, count, src_stride, dst_stride);
                }
                None => {
                    const_stride1_copy(src, dst, count, src_stride, dst_stride);
                }
            }
        }
        ExecMode::Transpose { dim_a, dim_b } => {
            let size_a = plan.fused_dims[dim_a];
            let size_b = plan.fused_dims[dim_b];
            let lda = plan.lda_inner;
            let ldb = plan.ldb_inner;
            let block = plan.block;
            let elem_size = std::mem::size_of::<T>();

            match &plan.root {
                Some(root) => {
                    transpose_recursive(src, dst, root, size_a, size_b, lda, ldb, block, elem_size);
                }
                None => {
                    // No outer loops — just the 2D blocked transpose
                    dispatch_blocked_2d(src, dst, size_a, size_b, lda, ldb, block, elem_size);
                }
            }
        }
    }
}

/// Execute the permutation plan with Rayon parallelism.
///
/// Parallelizes over the outermost ComputeNode's dimension.
/// Falls back to single-threaded for small tensors.
///
/// # Safety
/// Same requirements as `execute_permute_blocked`.
#[cfg(feature = "parallel")]
pub unsafe fn execute_permute_blocked_par<T: Copy + Send + Sync>(
    src: *const T,
    dst: *mut T,
    plan: &PermutePlan,
) {
    let total: usize = plan.fused_dims.iter().product();

    if total < MINTHREADLENGTH {
        execute_permute_blocked(src, dst, plan);
        return;
    }

    let root = match &plan.root {
        Some(r) => r,
        None => {
            execute_permute_blocked(src, dst, plan);
            return;
        }
    };

    let outer_dim = root.end;
    if outer_dim <= 1 {
        execute_permute_blocked(src, dst, plan);
        return;
    }

    let src_addr = src as usize;
    let dst_addr = dst as usize;
    let lda_root = root.lda;
    let ldb_root = root.ldb;
    let elem_size = std::mem::size_of::<T>();
    let inner = root.next.clone();

    match plan.mode {
        ExecMode::Transpose { dim_a, dim_b } => {
            let size_a = plan.fused_dims[dim_a];
            let size_b = plan.fused_dims[dim_b];
            let lda = plan.lda_inner;
            let ldb = plan.ldb_inner;
            let block = plan.block;

            (0..outer_dim).into_par_iter().for_each(|i| {
                let s = (src_addr as isize + (i as isize) * lda_root * (elem_size as isize))
                    as *const T;
                let d =
                    (dst_addr as isize + (i as isize) * ldb_root * (elem_size as isize)) as *mut T;

                unsafe {
                    match &inner {
                        Some(next) => {
                            transpose_recursive(
                                s, d, next, size_a, size_b, lda, ldb, block, elem_size,
                            );
                        }
                        None => {
                            dispatch_blocked_2d(s, d, size_a, size_b, lda, ldb, block, elem_size);
                        }
                    }
                }
            });
        }
        ExecMode::ConstStride1 { inner_dim } => {
            let count = plan.fused_dims[inner_dim];
            let src_stride = plan.src_strides[inner_dim];
            let dst_stride = plan.dst_strides[inner_dim];

            (0..outer_dim).into_par_iter().for_each(|i| {
                let s = (src_addr as isize + (i as isize) * lda_root * (elem_size as isize))
                    as *const T;
                let d =
                    (dst_addr as isize + (i as isize) * ldb_root * (elem_size as isize)) as *mut T;

                unsafe {
                    match &inner {
                        Some(next) => {
                            const_stride1_recursive(s, d, next, count, src_stride, dst_stride);
                        }
                        None => {
                            const_stride1_copy(s, d, count, src_stride, dst_stride);
                        }
                    }
                }
            });
        }
        ExecMode::Scalar => {
            execute_permute_blocked(src, dst, plan);
        }
    }
}

// ---------------------------------------------------------------------------
// Transpose mode: recursive execution
// ---------------------------------------------------------------------------

/// Recursive loop nest for Transpose mode.
///
/// Mirrors HPTT's `transpose_int`. Each ComputeNode iterates its dimension
/// with inc=1. At the leaf, runs the 2D blocked transpose over dim_A × dim_B.
unsafe fn transpose_recursive<T: Copy>(
    src: *const T,
    dst: *mut T,
    node: &ComputeNode,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
    elem_size: usize,
) {
    let end = node.end;
    let node_lda = node.lda;
    let node_ldb = node.ldb;

    match &node.next {
        Some(next) => {
            let mut s = src;
            let mut d = dst;
            for _ in 0..end {
                transpose_recursive(s, d, next, size_a, size_b, lda, ldb, block, elem_size);
                s = s.offset(node_lda);
                d = d.offset(node_ldb);
            }
        }
        None => {
            // Leaf: iterate this dim, calling blocked 2D transpose at each position
            let mut s = src;
            let mut d = dst;
            for _ in 0..end {
                dispatch_blocked_2d(s, d, size_a, size_b, lda, ldb, block, elem_size);
                s = s.offset(node_lda);
                d = d.offset(node_ldb);
            }
        }
    }
}

/// 2D blocked transpose over dim_A × dim_B.
///
/// Tiles both dimensions by BLOCK and calls the appropriate macro_kernel.
#[inline]
unsafe fn dispatch_blocked_2d<T: Copy>(
    src: *const T,
    dst: *mut T,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
    elem_size: usize,
) {
    match elem_size {
        8 => blocked_transpose_2d_f64(
            src as *const f64,
            dst as *mut f64,
            size_a,
            size_b,
            lda,
            ldb,
            block,
        ),
        4 => blocked_transpose_2d_f32(
            src as *const f32,
            dst as *mut f32,
            size_a,
            size_b,
            lda,
            ldb,
            block,
        ),
        _ => blocked_transpose_2d_fallback(src, dst, size_a, size_b, lda, ldb, block),
    }
}

#[inline]
unsafe fn blocked_transpose_2d_f64(
    src: *const f64,
    dst: *mut f64,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    let mut ib = 0usize;
    while ib < size_b {
        let bb = block.min(size_b - ib);
        let mut ia = 0usize;
        while ia < size_a {
            let ba = block.min(size_a - ia);
            macro_kernel_f64(
                src.offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                dst.offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
            ia += block;
        }
        ib += block;
    }
}

#[inline]
unsafe fn blocked_transpose_2d_f32(
    src: *const f32,
    dst: *mut f32,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    let mut ib = 0usize;
    while ib < size_b {
        let bb = block.min(size_b - ib);
        let mut ia = 0usize;
        while ia < size_a {
            let ba = block.min(size_a - ia);
            macro_kernel_f32(
                src.offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                dst.offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
            ia += block;
        }
        ib += block;
    }
}

#[inline]
unsafe fn blocked_transpose_2d_fallback<T: Copy>(
    src: *const T,
    dst: *mut T,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    let mut ib = 0usize;
    while ib < size_b {
        let bb = block.min(size_b - ib);
        let mut ia = 0usize;
        while ia < size_a {
            let ba = block.min(size_a - ia);
            macro_kernel_fallback(
                src.offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                dst.offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
            ia += block;
        }
        ib += block;
    }
}

// ---------------------------------------------------------------------------
// ConstStride1 mode: recursive execution
// ---------------------------------------------------------------------------

/// Recursive loop nest for ConstStride1 mode.
///
/// Mirrors HPTT's `transpose_int_constStride1`. Each ComputeNode iterates
/// its dimension. At the leaf, calls `const_stride1_copy` for the inner dim.
unsafe fn const_stride1_recursive<T: Copy>(
    src: *const T,
    dst: *mut T,
    node: &ComputeNode,
    count: usize,
    src_stride: isize,
    dst_stride: isize,
) {
    let end = node.end;
    let node_lda = node.lda;
    let node_ldb = node.ldb;

    match &node.next {
        Some(next) => {
            let mut s = src;
            let mut d = dst;
            for _ in 0..end {
                const_stride1_recursive(s, d, next, count, src_stride, dst_stride);
                s = s.offset(node_lda);
                d = d.offset(node_ldb);
            }
        }
        None => {
            let mut s = src;
            let mut d = dst;
            for _ in 0..end {
                const_stride1_copy(s, d, count, src_stride, dst_stride);
                s = s.offset(node_lda);
                d = d.offset(node_ldb);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hptt::plan::build_permute_plan;

    #[test]
    fn test_execute_identity_copy() {
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        let plan = build_permute_plan(&[2, 3], &[1, 2], &[1, 2], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        assert_eq!(dst, src);
    }

    #[test]
    fn test_execute_transpose_2d() {
        // src [3, 2] col-major: [1,2,3,4,5,6]
        // Permuted view: dims [2, 3], strides [3, 1]
        // dst col-major [2, 3]: strides [1, 2]
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
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
        let plan = build_permute_plan(&[4, 2, 3], &[6, 1, 2], &[1, 4, 8], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
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
        let plan = build_permute_plan(&[5, 3, 2, 4], &[24, 2, 1, 6], &[1, 5, 15, 30], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
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
        let plan = build_permute_plan(&[3, 2, 2, 2, 2], &[16, 1, 2, 4, 8], &[1, 3, 6, 12, 24], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
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
        let src = vec![42.0f64];
        let mut dst = vec![0.0f64];
        let plan = build_permute_plan(&[], &[], &[], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        assert_eq!(dst[0], 42.0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_execute_par_transpose_2d() {
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8);
        unsafe {
            execute_permute_blocked_par(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_execute_par_large() {
        let n = 256;
        let total = n * n * n;
        let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; total];

        // [256, 256, 256] col-major, transpose [2, 0, 1]
        let plan = build_permute_plan(&[n, n, n], &[65536, 1, 256], &[1, 256, 65536], 8);
        unsafe {
            execute_permute_blocked_par(src.as_ptr(), dst.as_mut_ptr(), &plan);
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
}
