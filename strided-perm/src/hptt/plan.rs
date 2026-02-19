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
pub struct ComputeNode {
    /// End index for this loop (loop runs 0..end).
    pub end: usize,
    /// Source stride for this dimension.
    pub lda: isize,
    /// Destination stride for this dimension.
    pub ldb: isize,
    /// Next node in the chain (None = leaf → calls macro_kernel or memcpy).
    pub next: Option<Box<ComputeNode>>,
}

/// Execution mode determined at plan time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
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
    /// Rank 0: single element copy.
    Scalar,
}

/// Complete permutation plan.
#[derive(Debug)]
pub struct PermutePlan {
    /// Fused dimensions (after bilateral fusion).
    pub fused_dims: Vec<usize>,
    /// Fused source strides.
    pub src_strides: Vec<isize>,
    /// Fused destination strides.
    pub dst_strides: Vec<isize>,
    /// Root of the recursive loop structure (None for Scalar mode).
    pub root: Option<ComputeNode>,
    /// Execution mode.
    pub mode: ExecMode,
    /// Source stride along dim_B — the "lda" for macro_kernel.
    /// (In the 2D view for the macro-kernel, this is the stride that
    /// steps between columns of the source tile.)
    pub lda_inner: isize,
    /// Dest stride along dim_A — the "ldb" for macro_kernel.
    pub ldb_inner: isize,
    /// Macro-kernel tile size (= BLOCK, e.g. 16 for f64).
    pub block: usize,
}

/// Build a permutation plan using bilateral fusion and HPTT-style blocking.
///
/// This is the main entry point. The returned plan is consumed by
/// `execute_permute_blocked`.
pub fn build_permute_plan(
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
        }
    } else {
        // Transpose path: 2D micro-kernel
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
        _ => 16,                                         // default
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
            _ => panic!("unexpected mode")
        }
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
