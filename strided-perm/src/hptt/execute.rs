//! Execution engine: recursive loop nest dispatching to macro_kernel.
//!
//! Mirrors HPTT C++'s `transpose_int` (lines 602-681) and
//! `transpose_int_constStride1` (lines 683-720).

use crate::hptt::macro_kernel::{
    const_stride1_copy, macro_kernel_f32, macro_kernel_f64, macro_kernel_fallback,
};
use crate::hptt::plan::{ExecMode, GroupedTilePlan, LoopLevel, PermutePlan};

use core::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use threadteam::Exec;

const ONE: NonZeroUsize = NonZeroUsize::MIN;

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

/// Execute the permutation plan on the calling thread.
///
/// # Safety
/// Same requirements as [`execute_permute_blocked`].
unsafe fn execute_serial<T: Copy + 'static>(src: *const T, dst: *mut T, plan: &PermutePlan) {
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

/// Execute the permutation plan within the budget of `exec`.
///
/// The outermost loop level is split into contiguous ranges, one per task; a
/// plan without an enclosing loop splits its 2D tile grid or its single strided
/// stream instead. The function uses exactly the budget it is given and never
/// narrows it: callers that want small copies to stay on one thread pass a
/// serial token or narrow it with [`Exec::limit`]. A serial token runs the whole
/// plan on the calling thread.
///
/// # Safety
/// - `src` must be valid for reads at all offsets determined by dims/src_strides
/// - `dst` must be valid for writes at all offsets determined by dims/dst_strides
/// - src and dst must not overlap
/// - no other thread may access the destination range during the call
/// - when `exec` runs more than one task, distinct logical positions must map to
///   distinct destination offsets. The partitions opened below are disjoint in
///   logical index space, and that alone does not keep two workers off one byte
///   when `dst_strides` is not injective. The crate's public entry proves this
///   with `is_injective_layout` at `copy.rs:184` before calling here.
pub unsafe fn execute_permute_blocked<T: Copy + Send + Sync + 'static>(
    src: *const T,
    dst: *mut T,
    plan: &PermutePlan,
    exec: Exec<'_>,
) {
    if exec.is_serial() {
        execute_serial(src, dst, plan);
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
                    exec,
                );
            }
            ExecMode::ConstStride1 { inner_dim } => {
                // One strided stream with no enclosing loop: split its
                // elements across the budget instead of running it on one task.
                let count = plan.fused_dims[inner_dim];
                let src_stride = plan.src_strides[inner_dim];
                let dst_stride = plan.dst_strides[inner_dim];
                let src_addr = src as usize;
                let dst_addr = dst as usize;
                exec.for_ranges(count, ONE, |range| {
                    let s = (src_addr as isize
                        + range.start as isize * src_stride * (std::mem::size_of::<T>() as isize))
                        as *const T;
                    let d = (dst_addr as isize
                        + range.start as isize * dst_stride * (std::mem::size_of::<T>() as isize))
                        as *mut T;
                    // SAFETY: `range` is a disjoint subrange of `0..count`, and
                    // `count`/`src_stride`/`dst_stride` are the plan's inner extent
                    // and strides, so `s`/`d` span exactly the reachable elements
                    // whose validity the caller proved under this function's
                    // `# Safety` (execute.rs:109). Range disjointness is logical
                    // only: two ranges land on distinct destination offsets solely
                    // because the caller supplied an injective destination, proved
                    // with `is_injective_layout` at `copy.rs:184`, not because the
                    // split itself separates them.
                    unsafe { const_stride1_copy(s, d, range.len(), src_stride, dst_stride) };
                });
            }
            _ => execute_serial(src, dst, plan),
        }
        return;
    }

    let root = &plan.levels[0];
    let outer_dim = root.end;
    if outer_dim <= 1 {
        execute_serial(src, dst, plan);
        return;
    }

    let src_addr = src as usize;
    let dst_addr = dst as usize;
    let lda_root = root.lda;
    let ldb_root = root.ldb;
    let elem_size = std::mem::size_of::<T>();
    let inner: &[LoopLevel] = &plan.levels[1..];
    let at = |i: usize| {
        (
            (src_addr as isize + (i as isize) * lda_root * (elem_size as isize)) as *const T,
            (dst_addr as isize + (i as isize) * ldb_root * (elem_size as isize)) as *mut T,
        )
    };

    match plan.mode {
        ExecMode::Transpose { dim_a, dim_b } => {
            let size_a = plan.fused_dims[dim_a];
            let size_b = plan.fused_dims[dim_b];
            let lda = plan.lda_inner;
            let ldb = plan.ldb_inner;
            let block = plan.block;

            exec.for_ranges(outer_dim, ONE, |range| {
                for i in range {
                    let (s, d) = at(i);
                    // SAFETY: `i` comes from a disjoint subrange of `0..outer_dim`,
                    // and `at` (execute.rs:184) advances `src`/`dst` by the plan's
                    // own `lda_root`/`ldb_root` for that axis, so `s`/`d` and the
                    // tile `dispatch_blocked_2d`/`transpose_recursive` write stay
                    // inside the buffers the caller proved valid under
                    // `execute_permute_blocked`'s `# Safety` (execute.rs:104).
                    unsafe {
                        if inner.is_empty() {
                            dispatch_blocked_2d(s, d, size_a, size_b, lda, ldb, block);
                        } else {
                            transpose_recursive(s, d, inner, size_a, size_b, lda, ldb, block);
                        }
                    }
                }
            });
        }
        ExecMode::ConstStride1 { inner_dim } => {
            let count = plan.fused_dims[inner_dim];
            let src_stride = plan.src_strides[inner_dim];
            let dst_stride = plan.dst_strides[inner_dim];

            exec.for_ranges(outer_dim, ONE, |range| {
                for i in range {
                    let (s, d) = at(i);
                    // SAFETY: as the Transpose arm above — `i` is a logical position
                    // on the root axis, so `s`/`d` advance by the plan's own strides
                    // for that axis and stay inside the caller-proved buffers
                    // (`execute_permute_blocked`'s `# Safety`, execute.rs:104).
                    unsafe {
                        if inner.is_empty() {
                            const_stride1_copy(s, d, count, src_stride, dst_stride);
                        } else {
                            const_stride1_recursive(s, d, inner, count, src_stride, dst_stride);
                        }
                    }
                }
            });
        }
        ExecMode::GroupedTranspose => {
            let grouped = plan
                .grouped_tile
                .as_ref()
                .expect("grouped transpose mode requires grouped tile plan");

            exec.for_ranges(outer_dim, ONE, |range| {
                for i in range {
                    let (s, d) = at(i);
                    // SAFETY: as the Transpose arm above — `s`/`d` are the tile
                    // origins for logical position `i` on the root axis, and the
                    // grouped offset tables were built by
                    // `try_build_grouped_tile_plan` (plan.rs:274) from the same
                    // validated layout, so every access stays inside the
                    // caller-proved buffers (`execute_permute_blocked`'s `# Safety`,
                    // execute.rs:104).
                    unsafe {
                        if inner.is_empty() {
                            grouped_tile_copy(s, d, grouped);
                        } else {
                            grouped_recursive(s, d, inner, grouped);
                        }
                    }
                }
            });
        }
        ExecMode::Scalar => execute_serial(src, dst, plan),
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
        // INVARIANT: the advance after the last iteration may leave the
        // allocation — the cursor is never dereferenced there, so it must use
        // `wrapping_offset` rather than `offset`. Rewriting it as an `offset`
        // would be UB for a tile whose last origin sits within one stride of the
        // buffer end, which is the common full-buffer case.
        s = s.wrapping_offset(node.lda);
        d = d.wrapping_offset(node.ldb);
    }
}

#[inline]
unsafe fn grouped_tile_copy<T: Copy>(src: *const T, dst: *mut T, grouped: &GroupedTilePlan) {
    let group_a = &grouped.group_a;
    let group_b = &grouped.group_b;
    // Borrow the offset tables once; indexing the `ArrayVec` per access would
    // pay its length check in the element-count loop.
    let a_src = group_a.src_offsets.as_slice();
    let a_dst = group_a.dst_offsets.as_slice();
    let b_src = group_b.src_offsets.as_slice();
    let b_dst = group_b.dst_offsets.as_slice();

    let copy_b = |src_a: *const T, dst_a: *mut T| {
        for b in 0..group_b.len {
            *dst_a.offset(b_dst[b]) = *src_a.offset(b_src[b]);
        }
    };
    if group_a.outer.is_empty() && group_b.outer.is_empty() {
        // Both groups fit their tables: the plain two-table tile.
        for a in 0..group_a.len {
            copy_b(src.offset(a_src[a]), dst.offset(a_dst[a]));
        }
        return;
    }
    walk_outer(&group_a.outer, src, dst, &mut |src, dst| {
        for a in 0..group_a.len {
            let (src_a, dst_a) = (src.offset(a_src[a]), dst.offset(a_dst[a]));
            walk_outer(&group_b.outer, src_a, dst_a, &mut |s, d| copy_b(s, d));
        }
    });
}

/// Visit every position of a group's outer axes, slowest axis outermost, so
/// the tile keeps the group's flat order.
unsafe fn walk_outer<T>(
    outer: &[(usize, isize, isize)],
    src: *const T,
    dst: *mut T,
    f: &mut impl FnMut(*const T, *mut T),
) {
    let Some((&(n, src_stride, dst_stride), rest)) = outer.split_last() else {
        f(src, dst);
        return;
    };
    let (mut s, mut d) = (src, dst);
    for _ in 0..n {
        walk_outer(rest, s, d, f);
        // INVARIANT: the last-iteration advance may leave the allocation; the
        // cursor is not dereferenced after it, so `wrapping_offset` is required
        // here (same reasoning as `grouped_recursive`).
        s = s.wrapping_offset(src_stride);
        d = d.wrapping_offset(dst_stride);
    }
}

// ---------------------------------------------------------------------------
// Transpose mode: recursive execution
// ---------------------------------------------------------------------------

/// Recursive loop nest for Transpose mode.
///
/// Mirrors HPTT's `transpose_int`. Each [`LoopLevel`] iterates its dimension
/// with inc=1. At the leaf, runs the 2D blocked transpose over dim_A × dim_B.
#[allow(clippy::too_many_arguments)] // INVARIANT: raw cursors plus the tile geometry the leaf needs.
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
        // INVARIANT: the last-iteration advance may leave the allocation; the
        // cursor is not dereferenced after it, so `wrapping_offset` is required
        // here (same reasoning as `grouped_recursive`).
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

#[inline]
fn transpose_tile_rows(size: usize, block: usize) -> usize {
    debug_assert!(block > 0);
    size.div_ceil(block)
}

/// Runs `tile_kernel(ia, ba, ib, bb)` for every tile of the `size_a` x `size_b`
/// grid within the budget of `exec`; `(ia, ib)` is the tile origin and
/// `(ba, bb)` its extent.
///
/// Splits the whole row x column tile grid, not only the row tiles: a
/// near-vector-shaped transpose (for example 3 x N) has a single row tile, so a
/// row-only split would run the entire kernel on one task.
fn for_each_tile_par(
    exec: Exec<'_>,
    size_a: usize,
    size_b: usize,
    block: usize,
    tile_kernel: impl Fn(usize, usize, usize, usize) + Sync,
) {
    let row_tiles = transpose_tile_rows(size_a, block);
    let col_tiles = transpose_tile_rows(size_b, block);
    let tiles = row_tiles * col_tiles;
    // One task per budget slot, each pulling tiles from a shared counter, so a
    // slow worker (a busy core, an efficiency core) does not leave the others
    // idle behind a static split.
    let next = AtomicUsize::new(0);
    let tasks = exec.budget().get().min(tiles);
    // Contiguous chunks, about eight per task, keep neighbouring tiles on one worker.
    let chunk = tiles.div_ceil(tasks * 8).max(1);
    exec.for_ranges(tasks, ONE, |_| loop {
        let first = next.fetch_add(chunk, Ordering::Relaxed);
        if first >= tiles {
            break;
        }
        for tile in first..(first + chunk).min(tiles) {
            let ia = (tile / col_tiles) * block;
            let ib = (tile % col_tiles) * block;
            tile_kernel(ia, block.min(size_a - ia), ib, block.min(size_b - ib));
        }
    });
}

#[inline]
// INVARIANT: the eight parameters are the prepared plan's raw tile geometry,
// forwarded unchanged from the `Transpose` arm of `execute_permute_blocked`;
// bundling them into a struct would add a per-call aggregate without removing
// any decision, so the lint is suppressed here rather than the call shape
// changed.
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_blocked_2d_par<T: Copy + Send + Sync + 'static>(
    src: *const T,
    dst: *mut T,
    size_a: usize,
    size_b: usize,
    lda: isize,
    ldb: isize,
    block: usize,
    exec: Exec<'_>,
) {
    let src_addr = src as usize;
    let dst_addr = dst as usize;
    match native_float_kind::<T>() {
        // SAFETY: `for_each_tile_par` (execute.rs:533) hands out tile origins in
        // `0..size_a` x `0..size_b` with extents clamped to the grid, so each
        // sub-tile stays inside the `size_a` x `size_b` region whose validity the
        // caller proved under `execute_permute_blocked`'s `# Safety`
        // (execute.rs:109). Tiles are disjoint as logical index sets; that they
        // reach distinct destination offsets, and so that no two workers write one
        // offset, rests on the caller's injective destination (established at
        // `is_injective_layout`, `copy.rs:184`), not on the tile partition.
        Some(8) => for_each_tile_par(exec, size_a, size_b, block, |ia, ba, ib, bb| unsafe {
            macro_kernel_f64(
                (src_addr as *const f64).offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                (dst_addr as *mut f64).offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
        }),
        // SAFETY: as the `f64` arm above. `native_float_kind` (execute.rs:29)
        // admits `f32` here only for types whose alignment is at least 4, so the
        // cast is aligned at every tile offset.
        Some(4) => for_each_tile_par(exec, size_a, size_b, block, |ia, ba, ib, bb| unsafe {
            macro_kernel_f32(
                (src_addr as *const f32).offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                (dst_addr as *mut f32).offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
        }),
        // SAFETY: as the `f64` arm above, on `T` itself, which needs no alignment
        // or representation assumption.
        _ => for_each_tile_par(exec, size_a, size_b, block, |ia, ba, ib, bb| unsafe {
            macro_kernel_fallback(
                (src_addr as *const T).offset(ia as isize + ib as isize * lda),
                lda,
                ba,
                (dst_addr as *mut T).offset(ib as isize + ia as isize * ldb),
                ldb,
                bb,
            );
        }),
    }
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
        // INVARIANT: the last-iteration advance may leave the allocation; the
        // cursor is not dereferenced after it, so `wrapping_offset` is required
        // here (same reasoning as `grouped_recursive`).
        s = s.wrapping_offset(node.lda);
        d = d.wrapping_offset(node.ldb);
    }
}

#[cfg(test)]
mod tests;
