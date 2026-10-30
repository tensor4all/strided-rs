//! Plan construction for HPTT-faithful tensor permutation.
//!
//! Mirrors HPTT C++'s plan construction: bilateral fusion → identify stride-1
//! dims → determine execution mode → compute loop order → build ComputeNode chain.

use crate::fuse::{plan_bilateral_fusion, FusionPlanError};
use crate::hptt::micro_kernel::{MicroKernel, ScalarKernel};
use strided_view::INLINE_RANK;
use tinyvec::{ArrayVec, TinyVec};

/// Fused axis extents, inline up to [`INLINE_RANK`].
pub(crate) type PlanDimVec = TinyVec<[usize; INLINE_RANK]>;
/// Fused strides, inline up to [`INLINE_RANK`].
pub(crate) type PlanStrideVec = TinyVec<[isize; INLINE_RANK]>;

/// One loop level of the flattened HPTT loop nest.
///
/// This replaces the previous `Box<ComputeNode>` linked list so a plan keeps
/// its loop structure inline up to [`INLINE_RANK`] levels with no per-node
/// allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LoopLevel {
    /// End index for this loop (loop runs 0..end).
    pub(crate) end: usize,
    /// Source stride for this dimension.
    pub(crate) lda: isize,
    /// Destination stride for this dimension.
    pub(crate) ldb: isize,
}

/// Flattened loop nest, outermost first; empty means all work is at the leaf.
pub(crate) type LoopLevels = TinyVec<[LoopLevel; INLINE_RANK]>;

/// Execution mode determined at plan time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecMode {
    /// dim_A != dim_B: 2D micro-kernel transpose path.
    Transpose {
        /// Dimension with smallest |src_stride| (stride-1 in source).
        dim_a: usize,
        /// Dimension with smallest |dst_stride| (stride-1 in dest).
        dim_b: usize,
    },
    /// dim_A == dim_B (perm[0]==0 equivalent): memcpy/strided-copy path.
    ConstStride1 {
        /// The shared stride-1 dimension.
        inner_dim: usize,
    },
    /// High-rank transpose path using two virtual flattened tile groups.
    GroupedTranspose,
    /// Rank 0: single element copy.
    Scalar,
}

/// One virtual flattened tile group.
///
/// The group's flat index is split into an inner part held as offset tables
/// and an outer part walked as plain strided loops. The tables cover the
/// leading group axes up to [`GROUP_OFFSETS_INLINE`] entries; an axis that
/// does not fit is divided at its largest divisor that does, and the rest of
/// the group goes to `outer`. The tables are therefore always inline, and a
/// rank `<= INLINE_RANK` plan is allocation-free whatever its tile extents
/// (tensor4all/strided-rs#337).
#[derive(Debug, Clone)]
pub(crate) struct OffsetGroup {
    pub(crate) src_offsets: ArrayVec<[isize; GROUP_OFFSETS_INLINE]>,
    pub(crate) dst_offsets: ArrayVec<[isize; GROUP_OFFSETS_INLINE]>,
    /// Number of table entries (the inner part of the group).
    pub(crate) len: usize,
    /// Outer group axes as `(extent, src_stride, dst_stride)`, fastest first.
    pub(crate) outer: TinyVec<[(usize, isize, isize); INLINE_RANK]>,
}

impl OffsetGroup {
    /// Number of group elements: the table length times the outer extents.
    #[cfg(test)]
    pub(crate) fn total_len(&self) -> usize {
        self.outer.iter().map(|&(n, _, _)| n).product::<usize>() * self.len
    }
}

/// Inline capacity of a grouped-tile offset table.
///
/// INVARIANT: bounds the inner part of every grouped tile, so the tables
/// never leave the plan's inline storage; the planner's group size is set by
/// the cache-tile rule in `select_contiguous_group`, not by this bound, and
/// the part of a group above it is walked as outer loops.
pub(crate) const GROUP_OFFSETS_INLINE: usize = 64;

/// Offset-table tile plan for high-rank scattered permutations.
#[derive(Debug, Clone)]
pub(crate) struct GroupedTilePlan {
    pub(crate) group_a: OffsetGroup,
    pub(crate) group_b: OffsetGroup,
}

/// Complete permutation plan.
#[derive(Debug)]
pub(crate) struct PermutePlan {
    /// Fused dimensions (after bilateral fusion).
    pub(crate) fused_dims: PlanDimVec,
    /// Fused source strides.
    pub(crate) src_strides: PlanStrideVec,
    /// Fused destination strides.
    pub(crate) dst_strides: PlanStrideVec,
    /// Flattened loop nest, outermost first (empty for Scalar / rootless).
    pub(crate) levels: LoopLevels,
    /// Execution mode.
    pub(crate) mode: ExecMode,
    /// Source stride along dim_B — the "lda" for macro_kernel.
    /// (In the 2D view for the macro-kernel, this is the stride that
    /// steps between columns of the source tile.)
    pub(crate) lda_inner: isize,
    /// Dest stride along dim_A — the "ldb" for macro_kernel.
    pub(crate) ldb_inner: isize,
    /// Macro-kernel tile size (= BLOCK, e.g. 16 for f64).
    pub(crate) block: usize,
    /// Optional grouped tile metadata for high-rank scattered permutations.
    pub(crate) grouped_tile: Option<GroupedTilePlan>,
}

/// Build a permutation plan using bilateral fusion and HPTT-style blocking.
///
/// This is the main entry point. The returned plan is consumed by
/// `execute_permute_blocked`.
///
/// # Errors
///
/// Propagates [`FusionPlanError`] from bilateral fusion. Legal stride-0
/// broadcast views can carry extents whose fused product overflows `usize`
/// (issue #263), so this is a recoverable planning failure, not an invariant.
pub(crate) fn build_permute_plan(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    elem_size: usize,
) -> Result<PermutePlan, FusionPlanError> {
    // Phase 1: Bilateral dimension fusion
    let fusion = plan_bilateral_fusion(dims, src_strides, dst_strides)?;
    let fused_dims = fusion.dims;
    let fused_src = fusion.src_strides;
    let fused_dst = fusion.dst_strides;

    let rank = fused_dims.len();
    if rank == 0 {
        return Ok(PermutePlan {
            fused_dims,
            src_strides: fused_src,
            dst_strides: fused_dst,
            levels: LoopLevels::new(),
            mode: ExecMode::Scalar,
            lda_inner: 0,
            ldb_inner: 0,
            block: 0,
            grouped_tile: None,
        });
    }

    // Phase 2: Identify stride-1 dimensions
    let dim_a = find_stride1_dim(&fused_dims, &fused_src);
    let dim_b = find_stride1_dim(&fused_dims, &fused_dst);

    // Phase 3: Determine execution mode and blocking
    let block = block_for_elem_size(elem_size);

    if dim_a == dim_b || fused_src[dim_a] != 1 || fused_dst[dim_b] != 1 {
        // The transpose kernels assume positive unit inner strides on both
        // operands. Broadcast, negative, and gapped inner axes instead use the
        // existing strided copy loop, which honors both explicit strides.
        let inner_dim = dim_a;
        let mode = ExecMode::ConstStride1 { inner_dim };

        let loop_order = compute_loop_order_const(&fused_dims, &fused_src, &fused_dst, inner_dim);
        let levels = build_loop_levels(&fused_dims, &fused_src, &fused_dst, &loop_order);

        Ok(PermutePlan {
            fused_dims,
            src_strides: fused_src.clone(),
            dst_strides: fused_dst.clone(),
            levels,
            mode,
            lda_inner: fused_src[inner_dim],
            ldb_inner: fused_dst[inner_dim],
            block: 0,
            grouped_tile: None,
        })
    } else {
        // Transpose path: 2D micro-kernel
        let legacy_area = fused_dims[dim_a].saturating_mul(fused_dims[dim_b]);
        if legacy_area < block.saturating_mul(block) {
            // The grouped path covers the dims the 2D micro-kernel cannot, by
            // tiling a source-contiguous group against a destination-contiguous
            // one. Its `target` is the smallest per-group extent worth taking,
            // so it sets the length of the destination-contiguous run written in
            // the inner loop. The micro-kernel block (16 elements = 128 bytes for
            // f64) leaves runs shorter than a page, which defeats sequential
            // first-touch and streaming on a fresh output; a small multiple of it
            // restores long runs without pushing the tile out of cache.
            // INVARIANT: `block` is the micro-kernel extent for this element size,
            // so `block * 2` keeps the inner run at several cache lines without
            // enlarging the outer step so far that it defeats the tile's locality.
            let grouped_target = block.saturating_mul(2);
            if let Some((levels, grouped_tile)) = try_build_grouped_tile_plan(
                &fused_dims,
                &fused_src,
                &fused_dst,
                grouped_target,
                legacy_area,
            ) {
                return Ok(PermutePlan {
                    fused_dims,
                    src_strides: fused_src,
                    dst_strides: fused_dst,
                    levels,
                    mode: ExecMode::GroupedTranspose,
                    lda_inner: 0,
                    ldb_inner: 0,
                    block,
                    grouped_tile: Some(grouped_tile),
                });
            }
        }

        let mode = ExecMode::Transpose { dim_a, dim_b };

        // lda_inner = src stride along dim_B (steps between rows in the 2D micro-kernel view)
        // ldb_inner = dst stride along dim_A (steps between rows in the transposed view)
        let lda_inner = fused_src[dim_b];
        let ldb_inner = fused_dst[dim_a];

        let loop_order =
            compute_loop_order_transpose(&fused_dims, &fused_src, &fused_dst, dim_a, dim_b);
        let levels = build_loop_levels(&fused_dims, &fused_src, &fused_dst, &loop_order);

        Ok(PermutePlan {
            fused_dims,
            src_strides: fused_src,
            dst_strides: fused_dst,
            levels,
            mode,
            lda_inner,
            ldb_inner,
            block,
            grouped_tile: None,
        })
    }
}

/// Find the dimension with the smallest absolute stride among non-trivial dims.
fn find_stride1_dim(dims: &[usize], strides: &[isize]) -> usize {
    dims.iter()
        .zip(strides.iter())
        .enumerate()
        .filter(|(_, (&d, _))| d > 1)
        .min_by_key(|(_, (_, &s))| s.unsigned_abs())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Upper bound on a grouped tile dimension group, as a multiple of the group
/// target. Keeps a grouped tile a cache-sized tile instead of letting one huge
/// axis absorb the rest of the permutation.
const MAX_GROUP_LEN_MULTIPLE: usize = 64;

/// BLOCK size for a given element size (matches HPTT's blocking_ = micro * 4).
fn block_for_elem_size(elem_size: usize) -> usize {
    match elem_size {
        8 => <ScalarKernel as MicroKernel<f64>>::BLOCK, // 16
        4 => <ScalarKernel as MicroKernel<f32>>::BLOCK, // 32
        _ => 16,                                        // default
    }
}

fn try_build_grouped_tile_plan(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    target: usize,
    legacy_area: usize,
) -> Option<(LoopLevels, GroupedTilePlan)> {
    let group_a = select_contiguous_group(dims, src_strides, &[], target)?;
    let group_a_len = group_len(dims, &group_a);
    if group_a_len < 4 {
        return None;
    }

    let group_b = select_contiguous_group(dims, dst_strides, &group_a, target)?;
    let group_b_len = group_len(dims, &group_b);
    let area = group_a_len.saturating_mul(group_b_len);
    if group_b_len < 4 || area < 128 || area <= legacy_area.saturating_mul(4) {
        return None;
    }

    let mut consumed: TinyVec<[bool; INLINE_RANK]> = TinyVec::with_capacity(dims.len());
    for _ in 0..dims.len() {
        consumed.push(false);
    }
    for &d in group_a.iter().chain(group_b.iter()) {
        consumed[d] = true;
    }

    let mut outer: PlanDimVec = TinyVec::with_capacity(dims.len());
    for d in 0..dims.len() {
        if !consumed[d] && dims[d] > 1 {
            outer.push(d);
        }
    }
    outer.sort_by(|&a, &b| {
        let cost_a = src_strides[a].unsigned_abs() + dst_strides[a].unsigned_abs();
        let cost_b = src_strides[b].unsigned_abs() + dst_strides[b].unsigned_abs();
        cost_b.cmp(&cost_a)
    });

    let levels = build_loop_levels(dims, src_strides, dst_strides, &outer);
    let grouped_tile = GroupedTilePlan {
        group_a: build_offset_group(dims, src_strides, dst_strides, &group_a),
        group_b: build_offset_group(dims, src_strides, dst_strides, &group_b),
    };
    Some((levels, grouped_tile))
}

/// Select the longest contiguous source or destination group within the
/// bounded cache-tile growth rule.
///
/// The group size is driven by `target` and the bounded-multiple growth rule,
/// not by the offset-table capacity: a group larger than
/// [`GROUP_OFFSETS_INLINE`] still builds, and `build_offset_group` walks the
/// part above the table bound as outer loops.
fn select_contiguous_group(
    dims: &[usize],
    strides: &[isize],
    excluded: &[usize],
    target: usize,
) -> Option<PlanDimVec> {
    let mut is_excluded: TinyVec<[bool; INLINE_RANK]> = TinyVec::with_capacity(dims.len());
    for _ in 0..dims.len() {
        is_excluded.push(false);
    }
    for &d in excluded {
        is_excluded[d] = true;
    }

    let mut order: PlanDimVec = TinyVec::with_capacity(dims.len());
    for d in 0..dims.len() {
        if dims[d] > 1 && !is_excluded[d] && strides[d] > 0 {
            order.push(d);
        }
    }
    order.sort_by_key(|&d| strides[d].unsigned_abs());

    let mut best = PlanDimVec::with_capacity(dims.len());
    let mut best_len = 0usize;

    for start in 0..order.len() {
        let first = order[start];
        let base = strides[first];
        let mut group = PlanDimVec::with_capacity(order.len());
        group.push(first);
        let mut len = dims[first];
        let mut expected = base.saturating_mul(len as isize);

        for &d in &order[start + 1..] {
            match strides[d].cmp(&expected) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => {
                    if len >= target {
                        break;
                    }
                    // `target` is a stopping threshold, not an upper bound. Stop
                    // before an axis whose inclusion would grow the group past a
                    // bounded multiple of it, so the tile stays a cache tile and
                    // the remaining axes keep driving the outer loop (and with it
                    // the parallel domain). A first axis larger than the bound is
                    // still taken: dimensions are atomic.
                    let next_len = len.saturating_mul(dims[d]);
                    if next_len > target.saturating_mul(MAX_GROUP_LEN_MULTIPLE) {
                        break;
                    }
                    group.push(d);
                    len = next_len;
                    expected = base.saturating_mul(len as isize);
                }
                std::cmp::Ordering::Greater => break,
            }
        }

        if len > best_len {
            best_len = len;
            best = group;
        }
        if best_len >= target {
            break;
        }
    }

    if best.is_empty() {
        None
    } else {
        Some(best)
    }
}

fn group_len(dims: &[usize], group: &[usize]) -> usize {
    group.iter().map(|&d| dims[d]).product()
}

fn build_offset_group(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    group: &[usize],
) -> OffsetGroup {
    // Split the group, fastest axis first, into table axes and outer axes.
    let mut inner: TinyVec<[(usize, isize, isize); INLINE_RANK]> = TinyVec::new();
    let mut outer: TinyVec<[(usize, isize, isize); INLINE_RANK]> = TinyVec::new();
    let mut len = 1usize;
    for &d in group {
        let (n, s, t) = (dims[d], src_strides[d], dst_strides[d]);
        if !outer.is_empty() {
            outer.push((n, s, t));
        } else if len * n <= GROUP_OFFSETS_INLINE {
            inner.push((n, s, t));
            len *= n;
        } else {
            // Divide the axis as `c` table steps times `n / c` outer steps of
            // stride `c * stride`; `c == 1` leaves the whole axis outer.
            let cap = GROUP_OFFSETS_INLINE / len;
            let c = (1..=cap).rev().find(|&c| n % c == 0).unwrap_or(1);
            if c > 1 {
                inner.push((c, s, t));
                len *= c;
            }
            outer.push((n / c, s * c as isize, t * c as isize));
        }
    }

    let mut src_offsets = ArrayVec::new();
    let mut dst_offsets = ArrayVec::new();
    for flat in 0..len {
        let mut rem = flat;
        let mut src_offset = 0isize;
        let mut dst_offset = 0isize;
        for &(n, s, t) in &inner {
            let coord = (rem % n) as isize;
            rem /= n;
            src_offset += coord * s;
            dst_offset += coord * t;
        }
        src_offsets.push(src_offset);
        dst_offsets.push(dst_offset);
    }

    OffsetGroup {
        src_offsets,
        dst_offsets,
        len,
        outer,
    }
}

/// Compute loop order for Transpose mode.
///
/// Excludes dim_a and dim_b (consumed by macro_kernel).
/// Remaining dims sorted by stride cost descending (largest strides outermost).
fn compute_loop_order_transpose(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    dim_a: usize,
    dim_b: usize,
) -> PlanDimVec {
    let mut loop_dims: PlanDimVec = TinyVec::with_capacity(dims.len());
    for d in 0..dims.len() {
        if d != dim_a && d != dim_b && dims[d] > 1 {
            loop_dims.push(d);
        }
    }
    loop_dims.sort_by(|&a, &b| {
        let cost_a = src_strides[a].unsigned_abs() + dst_strides[a].unsigned_abs();
        let cost_b = src_strides[b].unsigned_abs() + dst_strides[b].unsigned_abs();
        cost_b.cmp(&cost_a)
    });
    loop_dims
}

/// Compute loop order for ConstStride1 mode.
///
/// Excludes inner_dim (handled by memcpy at leaf).
/// Remaining dims sorted by |dst_stride| descending: largest dst stride outermost,
/// smallest innermost. This ensures the innermost loops advance by the smallest
/// dst offsets, building up contiguous blocks that tile perfectly with the
/// stride-1 inner copy. For a column-major dst (common case), this gives
/// fully sequential write access.
fn compute_loop_order_const(
    dims: &[usize],
    _src_strides: &[isize],
    dst_strides: &[isize],
    inner_dim: usize,
) -> PlanDimVec {
    let mut loop_dims: PlanDimVec = TinyVec::with_capacity(dims.len());
    for d in 0..dims.len() {
        if d != inner_dim && dims[d] > 1 {
            loop_dims.push(d);
        }
    }
    loop_dims.sort_by(|&a, &b| {
        dst_strides[b]
            .unsigned_abs()
            .cmp(&dst_strides[a].unsigned_abs())
    });
    loop_dims
}

/// Flatten a loop order into an inline array of loop levels, outermost first.
///
/// All levels iterate a single dimension. The two stride-1 dims are not in
/// the chain; they are handled by macro_kernel or memcpy at the leaf. An
/// empty result means all work is done at the leaf.
fn build_loop_levels(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    loop_order: &[usize],
) -> LoopLevels {
    let mut levels: LoopLevels = TinyVec::with_capacity(loop_order.len());
    for &d in loop_order {
        levels.push(LoopLevel {
            end: dims[d],
            lda: src_strides[d],
            ldb: dst_strides[d],
        });
    }
    levels
}

#[cfg(test)]
mod tests {
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
            16, 1024, 8388608, 4096, 1048576, 1, 8, 131072, 2, 4, 64, 128, 256, 512, 2048, 8192,
            16384, 32768, 262144, 2097152, 4194304, 32, 65536, 524288,
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
}
