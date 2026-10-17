//! Execution engine: recursive loop nest dispatching to macro_kernel.
//!
//! Mirrors HPTT C++'s `transpose_int` (lines 602-681) and
//! `transpose_int_constStride1` (lines 683-720).

use crate::hptt::macro_kernel::{
    const_stride1_copy, macro_kernel_f32, macro_kernel_f64, macro_kernel_fallback,
};
use crate::hptt::plan::{ExecMode, GroupedTilePlan, LoopLevel, PermutePlan};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Minimum elements to justify multi-threaded execution.
#[cfg(feature = "parallel")]
const MINTHREADLENGTH: usize = 1 << 15; // 32768

/// The native float width `T` may be reinterpreted as, or `None` for the
/// generic fallback.
///
/// INVARIANT: the permutation micro-kernels read and write the buffer as
/// `f64`/`f32`. Reinterpreting by `size_of::<T>()` alone is unsound for an
/// arbitrary `Copy` type with a weaker alignment (for example
/// `num_complex::Complex32`, size 8 / align 4, or `[u16; 4]`, size 8 / align
/// 2): a `Vec` of those is only guaranteed their own alignment, so the float
/// access can be misaligned. Restrict the reinterpretation to the exact
/// primitive types whose alignment matches the float width (#289); every
/// other type takes the generic element-wise path.
#[inline]
fn native_float_kind<T: 'static>() -> Option<usize> {
    let id = std::any::TypeId::of::<T>();
    if (id == std::any::TypeId::of::<f64>()
        || id == std::any::TypeId::of::<i64>()
        || id == std::any::TypeId::of::<u64>())
        && std::mem::align_of::<T>() >= 8
    {
        Some(8)
    } else if (id == std::any::TypeId::of::<f32>()
        || id == std::any::TypeId::of::<i32>()
        || id == std::any::TypeId::of::<u32>())
        && std::mem::align_of::<T>() >= 4
    {
        Some(4)
    } else {
        None
    }
}

/// Execute the permutation plan (single-threaded).
///
/// # Safety
/// - `src` must be valid for reads at all offsets determined by dims/src_strides
/// - `dst` must be valid for writes at all offsets determined by dims/dst_strides
/// - src and dst must not overlap
pub unsafe fn execute_permute_blocked<T: Copy + 'static>(
    src: *const T,
    dst: *mut T,
    plan: &PermutePlan,
) {
    match plan.mode {
        ExecMode::Scalar => {
            *dst = *src;
        }
        ExecMode::ConstStride1 { inner_dim } => {
            let count = plan.fused_dims[inner_dim];
            let src_stride = plan.src_strides[inner_dim];
            let dst_stride = plan.dst_strides[inner_dim];
            if plan.levels.is_empty() {
                const_stride1_copy(src, dst, count, src_stride, dst_stride);
            } else {
                const_stride1_recursive(src, dst, &plan.levels, count, src_stride, dst_stride);
            }
        }
        ExecMode::GroupedTranspose => {
            let grouped = plan
                .grouped_tile
                .as_ref()
                .expect("grouped transpose mode requires grouped tile plan");
            if plan.levels.is_empty() {
                grouped_tile_copy(src, dst, grouped);
            } else {
                grouped_recursive(src, dst, &plan.levels, grouped);
            }
        }
        ExecMode::Transpose { dim_a, dim_b } => {
            let size_a = plan.fused_dims[dim_a];
            let size_b = plan.fused_dims[dim_b];
            let lda = plan.lda_inner;
            let ldb = plan.ldb_inner;
            let block = plan.block;

            if plan.levels.is_empty() {
                // No outer loops — just the 2D blocked transpose
                dispatch_blocked_2d(src, dst, size_a, size_b, lda, ldb, block);
            } else {
                transpose_recursive(src, dst, &plan.levels, size_a, size_b, lda, ldb, block);
            }
        }
    }
}

/// Execute the permutation plan with Rayon parallelism.
///
/// Parallelizes over the outermost loop level.
/// Falls back to single-threaded for small tensors.
///
/// # Safety
/// Same requirements as `execute_permute_blocked`.
#[cfg(feature = "parallel")]
pub unsafe fn execute_permute_blocked_par<T: Copy + Send + Sync + 'static>(
    src: *const T,
    dst: *mut T,
    plan: &PermutePlan,
) {
    let total: usize = plan.fused_dims.iter().product();

    if total < MINTHREADLENGTH || rayon::current_num_threads() <= 1 {
        execute_permute_blocked(src, dst, plan);
        return;
    }

    if plan.levels.is_empty() {
        match plan.mode {
            ExecMode::Transpose { dim_a, dim_b } => {
                dispatch_blocked_2d_par(
                    src,
                    dst,
                    plan.fused_dims[dim_a],
                    plan.fused_dims[dim_b],
                    plan.lda_inner,
                    plan.ldb_inner,
                    plan.block,
                );
            }
            ExecMode::ConstStride1 { inner_dim } => {
                // One strided stream with no enclosing loop: split its
                // elements across workers instead of running it on one.
                let count = plan.fused_dims[inner_dim];
                let src_stride = plan.src_strides[inner_dim];
                let dst_stride = plan.dst_strides[inner_dim];
                let nthreads = rayon::current_num_threads().max(1);
                let chunk = count.div_ceil(nthreads).max(1);
                let src_addr = src as usize;
                let dst_addr = dst as usize;
                let chunks = count.div_ceil(chunk).max(1);
                (0..chunks).into_par_iter().for_each(|c| {
                    let start = c * chunk;
                    if start >= count {
                        return;
                    }
                    let len = chunk.min(count - start);
                    let s = (src_addr as isize
                        + start as isize * src_stride * (std::mem::size_of::<T>() as isize))
                        as *const T;
                    let d = (dst_addr as isize
                        + start as isize * dst_stride * (std::mem::size_of::<T>() as isize))
                        as *mut T;
                    unsafe { const_stride1_copy(s, d, len, src_stride, dst_stride) };
                });
            }
            _ => {
                execute_permute_blocked(src, dst, plan);
            }
        }
        return;
    }

    let root = &plan.levels[0];
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
    let inner: &[LoopLevel] = &plan.levels[1..];

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
                    if inner.is_empty() {
                        dispatch_blocked_2d(s, d, size_a, size_b, lda, ldb, block);
                    } else {
                        transpose_recursive(s, d, inner, size_a, size_b, lda, ldb, block);
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
                    if inner.is_empty() {
                        const_stride1_copy(s, d, count, src_stride, dst_stride);
                    } else {
                        const_stride1_recursive(s, d, inner, count, src_stride, dst_stride);
                    }
                }
            });
        }
        ExecMode::GroupedTranspose => {
            let grouped = plan
                .grouped_tile
                .as_ref()
                .expect("grouped transpose mode requires grouped tile plan");

            (0..outer_dim).into_par_iter().for_each(|i| {
                let s = (src_addr as isize + (i as isize) * lda_root * (elem_size as isize))
                    as *const T;
                let d =
                    (dst_addr as isize + (i as isize) * ldb_root * (elem_size as isize)) as *mut T;

                unsafe {
                    if inner.is_empty() {
                        grouped_tile_copy(s, d, grouped);
                    } else {
                        grouped_recursive(s, d, inner, grouped);
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
// Grouped transpose mode: recursive execution
// ---------------------------------------------------------------------------

unsafe fn grouped_recursive<T: Copy>(
    src: *const T,
    dst: *mut T,
    levels: &[LoopLevel],
    grouped: &GroupedTilePlan,
) {
    let (node, rest) = levels
        .split_first()
        .expect("grouped_recursive requires a loop level");
    let mut s = src;
    let mut d = dst;
    for _ in 0..node.end {
        if rest.is_empty() {
            grouped_tile_copy(s, d, grouped);
        } else {
            grouped_recursive(s, d, rest, grouped);
        }
        // The advance after the last iteration may leave the allocation; the
        // cursor is never dereferenced there, so it must not use `offset`.
        s = s.wrapping_offset(node.lda);
        d = d.wrapping_offset(node.ldb);
    }
}

#[inline]
unsafe fn grouped_tile_copy<T: Copy>(src: *const T, dst: *mut T, grouped: &GroupedTilePlan) {
    let group_a = &grouped.group_a;
    let group_b = &grouped.group_b;
    // Borrow the offset tables once; indexing the `TinyVec` per access would
    // pay its inline/heap match in the element-count loop.
    let a_src = group_a.src_offsets.as_slice();
    let a_dst = group_a.dst_offsets.as_slice();
    let b_src = group_b.src_offsets.as_slice();
    let b_dst = group_b.dst_offsets.as_slice();

    for a in 0..group_a.len {
        let src_a = src.offset(a_src[a]);
        let dst_a = dst.offset(a_dst[a]);
        for b in 0..group_b.len {
            *dst_a.offset(b_dst[b]) = *src_a.offset(b_src[b]);
        }
    }
}

// ---------------------------------------------------------------------------
// Transpose mode: recursive execution
// ---------------------------------------------------------------------------

/// Recursive loop nest for Transpose mode.
///
/// Mirrors HPTT's `transpose_int`. Each [`LoopLevel`] iterates its dimension
/// with inc=1. At the leaf, runs the 2D blocked transpose over dim_A × dim_B.
unsafe fn transpose_recursive<T: Copy + 'static>(
    src: *const T,
    dst: *mut T,
    levels: &[LoopLevel],
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    let (node, rest) = levels
        .split_first()
        .expect("transpose_recursive requires a loop level");
    let mut s = src;
    let mut d = dst;
    for _ in 0..node.end {
        if rest.is_empty() {
            dispatch_blocked_2d(s, d, size_a, size_b, lda, ldb, block);
        } else {
            transpose_recursive(s, d, rest, size_a, size_b, lda, ldb, block);
        }
        // Last-iteration advance may leave the allocation; see
        // `grouped_recursive`.
        s = s.wrapping_offset(node.lda);
        d = d.wrapping_offset(node.ldb);
    }
}

/// 2D blocked transpose over dim_A × dim_B.
///
/// The native-float micro-kernel is used only for the exact 4/8-byte types the
/// guard in [`native_float_kind`] allows; every other `T` goes through the
/// generic element-wise path.
#[inline]
unsafe fn dispatch_blocked_2d<T: Copy + 'static>(
    src: *const T,
    dst: *mut T,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    match native_float_kind::<T>() {
        Some(8) => blocked_transpose_2d_f64(
            src as *const f64,
            dst as *mut f64,
            size_a,
            size_b,
            lda,
            ldb,
            block,
        ),
        Some(4) => blocked_transpose_2d_f32(
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

#[cfg(feature = "parallel")]
#[inline]
fn transpose_tile_rows(size: usize, block: usize) -> usize {
    debug_assert!(block > 0);
    if size == 0 {
        0
    } else {
        (size + block - 1) / block
    }
}

#[cfg(feature = "parallel")]
#[inline]
unsafe fn dispatch_blocked_2d_par<T: Copy + Send + Sync + 'static>(
    src: *const T,
    dst: *mut T,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    match native_float_kind::<T>() {
        Some(8) => parallel_blocked_transpose_2d_f64(
            src as *const f64,
            dst as *mut f64,
            size_a,
            size_b,
            lda,
            ldb,
            block,
        ),
        Some(4) => parallel_blocked_transpose_2d_f32(
            src as *const f32,
            dst as *mut f32,
            size_a,
            size_b,
            lda,
            ldb,
            block,
        ),
        _ => parallel_blocked_transpose_2d_fallback(src, dst, size_a, size_b, lda, ldb, block),
    }
}

#[cfg(feature = "parallel")]
#[inline]
unsafe fn parallel_blocked_transpose_2d_f64(
    src: *const f64,
    dst: *mut f64,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    let row_tiles = transpose_tile_rows(size_a, block);
    let col_tiles = transpose_tile_rows(size_b, block);
    let src_addr = src as usize;
    let dst_addr = dst as usize;

    // Split the whole row x column tile grid, not only the row tiles. A
    // near-vector-shaped transpose (for example 3 x N) has a single row tile, so
    // a row-only split would run the entire kernel on one worker.
    (0..row_tiles * col_tiles).into_par_iter().for_each(|tile| {
        let tile_a = tile / col_tiles;
        let tile_b = tile % col_tiles;
        let ia = tile_a * block;
        let ba = block.min(size_a - ia);
        let ib = tile_b * block;
        let bb = block.min(size_b - ib);
        let src_base = src_addr as *const f64;
        let dst_base = dst_addr as *mut f64;
        unsafe {
            macro_kernel_f64(
                src_base.offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                dst_base.offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
        }
    });
}

#[cfg(feature = "parallel")]
#[inline]
unsafe fn parallel_blocked_transpose_2d_f32(
    src: *const f32,
    dst: *mut f32,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    let row_tiles = transpose_tile_rows(size_a, block);
    let col_tiles = transpose_tile_rows(size_b, block);
    let src_addr = src as usize;
    let dst_addr = dst as usize;

    // Split the whole row x column tile grid, not only the row tiles. A
    // near-vector-shaped transpose (for example 3 x N) has a single row tile, so
    // a row-only split would run the entire kernel on one worker.
    (0..row_tiles * col_tiles).into_par_iter().for_each(|tile| {
        let tile_a = tile / col_tiles;
        let tile_b = tile % col_tiles;
        let ia = tile_a * block;
        let ba = block.min(size_a - ia);
        let ib = tile_b * block;
        let bb = block.min(size_b - ib);
        let src_base = src_addr as *const f32;
        let dst_base = dst_addr as *mut f32;
        unsafe {
            macro_kernel_f32(
                src_base.offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                dst_base.offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
        }
    });
}

#[cfg(feature = "parallel")]
#[inline]
unsafe fn parallel_blocked_transpose_2d_fallback<T: Copy + Send + Sync>(
    src: *const T,
    dst: *mut T,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
) {
    let row_tiles = transpose_tile_rows(size_a, block);
    let col_tiles = transpose_tile_rows(size_b, block);
    let src_addr = src as usize;
    let dst_addr = dst as usize;

    // Split the whole row x column tile grid, not only the row tiles. A
    // near-vector-shaped transpose (for example 3 x N) has a single row tile, so
    // a row-only split would run the entire kernel on one worker.
    (0..row_tiles * col_tiles).into_par_iter().for_each(|tile| {
        let tile_a = tile / col_tiles;
        let tile_b = tile % col_tiles;
        let ia = tile_a * block;
        let ba = block.min(size_a - ia);
        let ib = tile_b * block;
        let bb = block.min(size_b - ib);
        let src_base = src_addr as *const T;
        let dst_base = dst_addr as *mut T;
        unsafe {
            macro_kernel_fallback(
                src_base.offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                dst_base.offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
        }
    });
}

// ---------------------------------------------------------------------------
// ConstStride1 mode: recursive execution
// ---------------------------------------------------------------------------

/// Recursive loop nest for ConstStride1 mode.
///
/// Mirrors HPTT's `transpose_int_constStride1`. Each [`LoopLevel`] iterates
/// its dimension. At the leaf, calls `const_stride1_copy` for the inner dim.
unsafe fn const_stride1_recursive<T: Copy>(
    src: *const T,
    dst: *mut T,
    levels: &[LoopLevel],
    count: usize,
    src_stride: isize,
    dst_stride: isize,
) {
    let (node, rest) = levels
        .split_first()
        .expect("const_stride1_recursive requires a loop level");
    let mut s = src;
    let mut d = dst;
    for _ in 0..node.end {
        if rest.is_empty() {
            const_stride1_copy(s, d, count, src_stride, dst_stride);
        } else {
            const_stride1_recursive(s, d, rest, count, src_stride, dst_stride);
        }
        // Last-iteration advance may leave the allocation; see
        // `grouped_recursive`.
        s = s.wrapping_offset(node.lda);
        d = d.wrapping_offset(node.ldb);
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
        let plan = build_permute_plan(&[2, 3], &[1, 2], &[1, 2], 8).unwrap();
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        assert_eq!(dst, src);
    }

    #[test]
    fn terminal_cursor_advance_may_leave_the_allocation() {
        // dims [2, 2] with source strides [1, 3] over a 5-element buffer: the
        // outer loop's terminal advance reaches offset 6 after its last copy,
        // past the allocation. That cursor is never read, so the advance must
        // wrap; Miri flags the pre-fix `offset`.
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0f64; 4];
        let plan = build_permute_plan(&[2, 2], &[1, 3], &[1, 2], 8).unwrap();
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        // dst(i, j) = src(i + 3j) in col-major dst: [1, 2, 4, 5].
        assert_eq!(dst, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_execute_transpose_2d() {
        // src [3, 2] col-major: [1,2,3,4,5,6]
        // Permuted view: dims [2, 3], strides [3, 1]
        // dst col-major [2, 3]: strides [1, 2]
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8).unwrap();
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
        let plan = build_permute_plan(&[4, 2, 3], &[6, 1, 2], &[1, 4, 8], 8).unwrap();
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
        let plan = build_permute_plan(&[5, 3, 2, 4], &[24, 2, 1, 6], &[1, 5, 15, 30], 8).unwrap();
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
        let plan =
            build_permute_plan(&[3, 2, 2, 2, 2], &[16, 1, 2, 4, 8], &[1, 3, 6, 12, 24], 8).unwrap();
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
        let plan = build_permute_plan(&[], &[], &[], 8).unwrap();
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
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

    #[cfg(feature = "parallel")]
    #[test]
    fn test_execute_par_transpose_2d() {
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8).unwrap();
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
        let plan = build_permute_plan(&[n, n, n], &[65536, 1, 256], &[1, 256, 65536], 8).unwrap();
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

    #[cfg(feature = "parallel")]
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

    #[cfg(feature = "parallel")]
    #[test]
    fn test_execute_par_one_thread_pool_uses_serial_path() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let n = 128;
            let total = n * n;
            let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
            let mut dst_par = vec![0.0f64; total];
            let mut dst_serial = vec![0.0f64; total];
            let plan = build_permute_plan(&[n, n], &[n as isize, 1], &[1, n as isize], 8).unwrap();

            unsafe {
                execute_permute_blocked_par(src.as_ptr(), dst_par.as_mut_ptr(), &plan);
                execute_permute_blocked(src.as_ptr(), dst_serial.as_mut_ptr(), &plan);
            }

            assert_eq!(dst_par, dst_serial);
        });
    }
}
