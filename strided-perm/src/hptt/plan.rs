//! Plan construction for HPTT-faithful tensor permutation.
//!
//! Mirrors HPTT C++'s plan construction: bilateral fusion → identify stride-1
//! dims → determine execution mode → compute loop order → build ComputeNode chain.

use crate::fuse::fuse_dims_bilateral;
use crate::hptt::micro_kernel::{MicroKernel, ScalarKernel};

/// A node in the recursive loop structure.
///
/// Mirrors HPTT's ComputeNode linked list. Each node represents one
/// loop level in the execution nest.
#[derive(Debug, Clone)]
pub(crate) struct ComputeNode {
    /// End index for this loop (loop runs 0..end).
    pub(crate) end: usize,
    /// Source stride for this dimension.
    pub(crate) lda: isize,
    /// Destination stride for this dimension.
    pub(crate) ldb: isize,
    /// Next node in the chain (None = leaf → calls macro_kernel or memcpy).
    pub(crate) next: Option<Box<ComputeNode>>,
}

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
#[derive(Debug, Clone)]
pub(crate) struct OffsetGroup {
    pub(crate) src_offsets: Vec<isize>,
    pub(crate) dst_offsets: Vec<isize>,
    pub(crate) len: usize,
}

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
    pub(crate) fused_dims: Vec<usize>,
    /// Fused source strides.
    pub(crate) src_strides: Vec<isize>,
    /// Fused destination strides.
    pub(crate) dst_strides: Vec<isize>,
    /// Root of the recursive loop structure (None for Scalar mode).
    pub(crate) root: Option<ComputeNode>,
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
pub(crate) fn build_permute_plan(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    elem_size: usize,
) -> PermutePlan {
    // Phase 1: Bilateral dimension fusion
    let (fused_dims, fused_src, fused_dst) = fuse_dims_bilateral(dims, src_strides, dst_strides);

    let rank = fused_dims.len();
    if rank == 0 {
        return PermutePlan {
            fused_dims,
            src_strides: fused_src,
            dst_strides: fused_dst,
            root: None,
            mode: ExecMode::Scalar,
            lda_inner: 0,
            ldb_inner: 0,
            block: 0,
            grouped_tile: None,
        };
    }

    // Phase 2: Identify stride-1 dimensions
    let dim_a = find_stride1_dim(&fused_dims, &fused_src);
    let dim_b = find_stride1_dim(&fused_dims, &fused_dst);

    // Phase 3: Determine execution mode and blocking
    let block = block_for_elem_size(elem_size);

    if dim_a == dim_b {
        // ConstStride1 path: both stride-1 dims are the same
        let inner_dim = dim_a;
        let mode = ExecMode::ConstStride1 { inner_dim };

        let loop_order = compute_loop_order_const(&fused_dims, &fused_src, &fused_dst, inner_dim);
        let root = build_compute_nodes(&fused_dims, &fused_src, &fused_dst, &loop_order);

        PermutePlan {
            fused_dims,
            src_strides: fused_src.clone(),
            dst_strides: fused_dst.clone(),
            root,
            mode,
            lda_inner: fused_src[inner_dim],
            ldb_inner: fused_dst[inner_dim],
            block: 0,
            grouped_tile: None,
        }
    } else {
        // Transpose path: 2D micro-kernel
        let legacy_area = fused_dims[dim_a].saturating_mul(fused_dims[dim_b]);
        if legacy_area < block.saturating_mul(block) {
            if let Some((root, grouped_tile)) =
                try_build_grouped_tile_plan(&fused_dims, &fused_src, &fused_dst, block, legacy_area)
            {
                return PermutePlan {
                    fused_dims,
                    src_strides: fused_src,
                    dst_strides: fused_dst,
                    root,
                    mode: ExecMode::GroupedTranspose,
                    lda_inner: 0,
                    ldb_inner: 0,
                    block,
                    grouped_tile: Some(grouped_tile),
                };
            }
        }

        let mode = ExecMode::Transpose { dim_a, dim_b };

        // lda_inner = src stride along dim_B (steps between rows in the 2D micro-kernel view)
        // ldb_inner = dst stride along dim_A (steps between rows in the transposed view)
        let lda_inner = fused_src[dim_b];
        let ldb_inner = fused_dst[dim_a];

        let loop_order =
            compute_loop_order_transpose(&fused_dims, &fused_src, &fused_dst, dim_a, dim_b);
        let root = build_compute_nodes(&fused_dims, &fused_src, &fused_dst, &loop_order);

        PermutePlan {
            fused_dims,
            src_strides: fused_src,
            dst_strides: fused_dst,
            root,
            mode,
            lda_inner,
            ldb_inner,
            block,
            grouped_tile: None,
        }
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
) -> Option<(Option<ComputeNode>, GroupedTilePlan)> {
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

    let mut consumed = vec![false; dims.len()];
    for &d in group_a.iter().chain(group_b.iter()) {
        consumed[d] = true;
    }

    let mut outer: Vec<usize> = (0..dims.len())
        .filter(|&d| !consumed[d] && dims[d] > 1)
        .collect();
    outer.sort_by(|&a, &b| {
        let cost_a = src_strides[a].unsigned_abs() + dst_strides[a].unsigned_abs();
        let cost_b = src_strides[b].unsigned_abs() + dst_strides[b].unsigned_abs();
        cost_b.cmp(&cost_a)
    });

    let root = build_compute_nodes(dims, src_strides, dst_strides, &outer);
    let grouped_tile = GroupedTilePlan {
        group_a: build_offset_group(dims, src_strides, dst_strides, &group_a),
        group_b: build_offset_group(dims, src_strides, dst_strides, &group_b),
    };
    Some((root, grouped_tile))
}

fn select_contiguous_group(
    dims: &[usize],
    strides: &[isize],
    excluded: &[usize],
    target: usize,
) -> Option<Vec<usize>> {
    let mut is_excluded = vec![false; dims.len()];
    for &d in excluded {
        is_excluded[d] = true;
    }

    let mut order: Vec<usize> = (0..dims.len())
        .filter(|&d| dims[d] > 1 && !is_excluded[d] && strides[d] > 0)
        .collect();
    order.sort_by_key(|&d| strides[d].unsigned_abs());

    let mut best = Vec::new();
    let mut best_len = 0usize;

    for start in 0..order.len() {
        let first = order[start];
        let base = strides[first];
        let mut group = vec![first];
        let mut len = dims[first];
        let mut expected = base.saturating_mul(len as isize);

        for &d in &order[start + 1..] {
            match strides[d].cmp(&expected) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => {
                    group.push(d);
                    len = len.saturating_mul(dims[d]);
                    if len >= target {
                        break;
                    }
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
    let len = group_len(dims, group);
    let mut src_offsets = Vec::with_capacity(len);
    let mut dst_offsets = Vec::with_capacity(len);

    for flat in 0..len {
        let mut rem = flat;
        let mut src_offset = 0isize;
        let mut dst_offset = 0isize;
        for &d in group {
            let coord = rem % dims[d];
            rem /= dims[d];
            src_offset += coord as isize * src_strides[d];
            dst_offset += coord as isize * dst_strides[d];
        }
        src_offsets.push(src_offset);
        dst_offsets.push(dst_offset);
    }

    OffsetGroup {
        src_offsets,
        dst_offsets,
        len,
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
) -> Vec<usize> {
    let mut loop_dims: Vec<usize> = (0..dims.len())
        .filter(|&d| d != dim_a && d != dim_b && dims[d] > 1)
        .collect();
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
) -> Vec<usize> {
    let mut loop_dims: Vec<usize> = (0..dims.len())
        .filter(|&d| d != inner_dim && dims[d] > 1)
        .collect();
    loop_dims.sort_by(|&a, &b| {
        dst_strides[b]
            .unsigned_abs()
            .cmp(&dst_strides[a].unsigned_abs())
    });
    loop_dims
}

/// Build a linked-list ComputeNode chain from loop_order.
///
/// All nodes have inc=1 (the two stride-1 dims are not in the chain;
/// they are handled by macro_kernel or memcpy at the leaf).
/// Returns None if loop_order is empty (all work done at the leaf).
fn build_compute_nodes(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    loop_order: &[usize],
) -> Option<ComputeNode> {
    let mut current: Option<ComputeNode> = None;

    // Build from innermost (last in loop_order) to outermost (first)
    for &d in loop_order.iter().rev() {
        let node = ComputeNode {
            end: dims[d],
            lda: src_strides[d],
            ldb: dst_strides[d],
            next: current.map(Box::new),
        };
        current = Some(node);
    }

    current
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
        let plan = build_permute_plan(&[2, 3, 4], &[1, 2, 6], &[1, 2, 6], 8);
        assert_eq!(plan.fused_dims, vec![24]);
        assert!(matches!(plan.mode, ExecMode::ConstStride1 { .. }));
    }

    #[test]
    fn test_build_plan_transpose_2d() {
        // 2D transpose: src [1, 4], dst [5, 1]
        let plan = build_permute_plan(&[4, 5], &[1, 4], &[5, 1], 8);
        assert_eq!(plan.fused_dims, vec![4, 5]);
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
        assert!(plan.root.is_none());
    }

    #[test]
    fn test_build_plan_3d_permute() {
        // 3D: dims [4,2,3], src strides [6,1,2], dst [1,4,8]
        // Bilateral fusion: dims 1-2 fuse (src: 2*1=2 == strides[2], dst: 2*4=8 == strides[2])
        // After fusion: dims [4, 6], src [6, 1], dst [1, 4]
        let plan = build_permute_plan(&[4, 2, 3], &[6, 1, 2], &[1, 4, 8], 8);
        assert_eq!(plan.fused_dims, vec![4, 6]);
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
        assert!(plan.root.is_none());
    }

    #[test]
    fn test_build_plan_scattered_strides() {
        // Simplified scattered case: 4 dims of size 2
        let dims = vec![2, 2, 2, 2];
        let src_strides = vec![1, 8, 2, 4]; // scattered
        let dst_strides = vec![1, 2, 4, 8]; // col-major

        let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8);

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

        let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8);

        assert!(matches!(plan.mode, ExecMode::GroupedTranspose));
        let grouped = plan.grouped_tile.as_ref().expect("grouped tile plan");
        assert!(grouped.group_a.len >= 16);
        assert!(grouped.group_b.len >= 16);
        assert!(grouped.group_a.len * grouped.group_b.len >= 256);
    }

    #[test]
    fn test_build_plan_large_3d_transpose_uses_legacy_tile() {
        let dims = vec![256usize, 256, 256];
        let src_strides = vec![256isize, 1, 65536];
        let dst_strides = vec![1isize, 256, 65536];

        let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8);

        assert!(matches!(plan.mode, ExecMode::Transpose { .. }));
        assert!(plan.grouped_tile.is_none());
    }

    #[test]
    fn test_build_plan_rank0() {
        let plan = build_permute_plan(&[], &[], &[], 8);
        assert!(matches!(plan.mode, ExecMode::Scalar));
        assert!(plan.root.is_none());
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
        assert_eq!(order, vec![3, 2]);
    }

    #[test]
    fn test_build_compute_nodes_chain() {
        let dims = [10, 5, 3];
        let src_s = [1isize, 10, 50];
        let dst_s = [15isize, 1, 5];
        let loop_order = vec![2]; // only dim 2 in the loop

        let root = build_compute_nodes(&dims, &src_s, &dst_s, &loop_order);
        assert!(root.is_some());
        let root = root.unwrap();
        assert_eq!(root.end, 3);
        assert_eq!(root.lda, 50);
        assert_eq!(root.ldb, 5);
        assert!(root.next.is_none());
    }
}
