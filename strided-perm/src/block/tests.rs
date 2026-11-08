use super::*;

#[test]
fn test_total_memory_region_contiguous() {
    let dims = [100usize];
    let strides = [8u128];
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let region = total_memory_region(&dims, &byte_strides);
    assert_eq!(region, 832);
}

#[test]
fn test_compute_blocks_small() {
    let dims = [10usize, 10];
    let costs = [2isize, 2];
    let strides = [8u128, 80];
    let orders = [1usize, 2];
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];
    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );
    assert_eq!(blocks, vec![10, 10]);
}

#[test]
fn test_compute_blocks_large() {
    let dims = [1000usize, 1000];
    let costs = [2isize, 2];
    let strides = [8u128, 8000];
    let orders = [1usize, 2];
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];
    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );
    assert!(blocks[0] <= dims[0]);
    assert!(blocks[1] <= dims[1]);
    assert!(blocks[0] >= 1);
    assert!(blocks[1] >= 1);
}

#[test]
fn test_compute_blocks_empty() {
    let blocks = compute_blocks(&[], &[], &[], &[], BLOCK_MEMORY_SIZE);
    assert!(blocks.is_empty());
}

#[test]
fn test_last_argmax_weighted_basic() {
    // blocks [10, 5], costs [2, 4]
    // scores: (10-1)*2=18, (5-1)*4=16 → first wins
    assert_eq!(last_argmax_weighted(&[10, 5], &[2, 4]), Some(0));
}

#[test]
fn test_last_argmax_weighted_ties() {
    // Equal scores: last wins (>= semantics)
    assert_eq!(last_argmax_weighted(&[5, 5], &[2, 2]), Some(1));
}

#[test]
fn test_last_argmax_weighted_all_one() {
    // All blocks are 1: no valid candidate
    assert_eq!(last_argmax_weighted(&[1, 1, 1], &[1, 1, 1]), None);
}

#[test]
fn test_last_argmax_weighted_empty() {
    assert_eq!(last_argmax_weighted(&[], &[]), None);
}

#[test]
fn test_total_memory_region_multi_array() {
    // Two arrays with different strides
    let dims = [100usize, 100];
    let s1 = [8u128, 800]; // col-major f64
    let s2 = [800u128, 8]; // row-major f64
    let byte_strides: Vec<&[u128]> = vec![&s1, &s2];
    let region = total_memory_region(&dims, &byte_strides);
    // Should sum contributions from both arrays
    assert!(region > 0);
}

#[test]
fn test_total_memory_region_large_stride() {
    // Stride >= cache line triggers block multiplication
    let dims = [10usize, 10];
    let strides = [8u128, 800]; // second dim stride 800 >= 64
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let region = total_memory_region(&dims, &byte_strides);
    assert!(region > 0);
}

#[test]
fn test_compute_blocks_min_stride_exceeds_block_size() {
    // Both strides very large (> block_size=64): min_stride > block_size → all blocks = 1
    // Two arrays with conflicting stride orders to prevent recursive tail path
    let dims = [10usize, 10];
    let costs = [1isize, 1];
    let s1 = [100000u128, 1000000]; // array 1: order [0, 1]
    let s2 = [1000000u128, 100000]; // array 2: order [1, 0]
    let o1 = [0usize, 1];
    let o2 = [1usize, 0]; // conflicting: o1[0]=0, o2[0]=1, min_order=0 but o2[0]!=0
    let byte_strides: Vec<&[u128]> = vec![&s1, &s2];
    let stride_orders: Vec<&[usize]> = vec![&o1, &o2];
    let blocks = compute_blocks(&dims, &costs, &byte_strides, &stride_orders, 64);
    assert_eq!(blocks, vec![1, 1]);
}

#[test]
fn test_compute_block_sizes_3d() {
    // 3D col-major
    let dims = [10usize, 20, 30];
    let strides = [8isize, 80, 1600];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let blocks = compute_block_sizes(&dims, &strides_list, 8);
    assert_eq!(blocks.len(), 3);
    for i in 0..3 {
        assert!(blocks[i] >= 1 && blocks[i] <= dims[i]);
    }
}

#[test]
fn test_compute_blocks_first_stride_order_matches() {
    // stride_orders[0][0] == min_order → triggers recursive tail path
    let dims = [4usize, 100, 100];
    let costs = [1isize, 10, 10];
    // Stride order: [0, 1, 2] and min_order = 0
    let strides = [8u128, 32, 3200];
    let orders = [0usize, 1, 2];
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];
    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );
    // First dim should be kept at full extent
    assert_eq!(blocks[0], 4);
}

// ========== Exact arithmetic contract ==========

// Copy, not a `#[path]` into the sibling crate: a path outside this package is
// absent from the published `.crate`, so `cargo test` failed in the unpacked
// package. Keep it identical to `strided-basic/src/block/tests/oracle.rs` —
// `scripts/check-oracle-copy.sh` diffs the two, and CI runs it.
mod oracle;

use oracle::{legacy_blocks, model_blocks, model_costs, LegacyCosts};

#[test]
fn compute_block_sizes_equals_the_exact_model_on_the_sweep() {
    let (mut fits, mut overflows) = (0, 0);
    for (dims, strides, elem_size) in oracle::sweep() {
        let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
        let model = model_blocks(&dims, &strides, elem_size);
        assert_eq!(
            compute_block_sizes(&dims, &strides, elem_size),
            model,
            "{dims:?} {strides:?} e={elem_size}"
        );
        let costs: Vec<u128> = compute_costs(&strides).iter().map(|&c| c as u128).collect();
        assert_eq!(costs, model_costs(&strides));
        match legacy_blocks(&dims, &strides, elem_size, LegacyCosts::Unchecked) {
            Some(legacy) => {
                assert_eq!(legacy, model, "{dims:?} {strides:?} e={elem_size}");
                fits += 1;
            }
            None => overflows += 1,
        }
    }
    assert!(
        fits > 0 && overflows > 0,
        "{fits} fitting, {overflows} overflowing"
    );
}

#[test]
fn costs_saturate_instead_of_overflowing() {
    // R3-4: 2 * 2^62 overflowed `isize`, and so did `abs(isize::MIN)`.
    let dims = [1024usize, 1024];
    let a = [1isize << 62, 1];
    let b = [1isize << 62, 2];
    assert_eq!(compute_costs(&[&a, &b]), [isize::MAX, 2]);
    assert_eq!(compute_block_sizes(&dims, &[&a, &b], 1), [10, 1024]);
    assert_eq!(model_blocks(&dims, &[&a, &b], 1), [10, 1024]);
    assert_eq!(
        legacy_blocks(&dims, &[&a, &b], 1, LegacyCosts::Unchecked),
        None
    );
    assert_eq!(compute_costs(&[[isize::MIN, 1]]), [isize::MAX, 2]);
}

/// The perm front end, the oracle and the legacy replay on one preordered case.
fn check(
    dims: &[usize],
    strides: &[&[isize]],
    elem_size: usize,
    expected: &[usize],
    legacy: Option<&[usize]>,
) {
    assert_eq!(compute_block_sizes(dims, strides, elem_size), expected);
    assert_eq!(model_blocks(dims, strides, elem_size), expected);
    assert_eq!(
        legacy_blocks(dims, strides, elem_size, LegacyCosts::Unchecked).as_deref(),
        legacy
    );
}

#[test]
fn the_traced_overflow_fixtures_follow_the_exact_model() {
    // R1-A: region 2^66 per operand, which wrapped to 0.
    let a = [8isize, 8, 8, 8, 8, 16];
    let b = [8isize, 8, 8, 8, 16, 8];
    check(
        &[1024; 6],
        &[&a, &b],
        8,
        &[1024, 1024, 1024, 1024, 16, 16],
        None,
    );
    // R1-B: byte stride 2^61 * 8 = 2^64, which wrapped to 0.
    check(&[2, 8192], &[&[1, 2], &[1 << 61, 1]], 8, &[2, 1024], None);
    // R3-3: score (3 - 1) * 2^62 = 2^63, which wrapped to -2^63.
    check(
        &[3, 8192],
        &[&[1 << 61, 1], &[1 << 61, 2]],
        1,
        &[1, 8192],
        None,
    );
}

#[test]
fn a_zero_element_size_keeps_every_axis_for_any_operand_count() {
    // R3-1: 513 operands, region 32832 above the budget, all byte orders 1.
    let mut strides = vec![vec![1isize, 2]; 512];
    strides.push(vec![2, 1]);
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    check(&[2, 2], &strides, 0, &[2, 2], Some(&[2, 2]));
}

#[test]
fn tied_and_strictly_increasing_strides() {
    // F1: ties run the full core.
    let dst = [1isize, 64, 4096];
    for src in [[0isize, 1, 1], [1, 1, 1]] {
        check(
            &[64, 64, 64],
            &[&dst, &src],
            8,
            &[64, 31, 16],
            Some(&[64, 31, 16]),
        );
    }
    // crop r2 and reverse r2: every axis kept.
    check(
        &[3840, 4096],
        &[&[1, 3840], &[1, 4096]],
        8,
        &[3840, 4096],
        Some(&[3840, 4096]),
    );
    check(
        &[4096, 4096],
        &[&[1, 4096], &[-1, 4096]],
        8,
        &[4096, 4096],
        Some(&[4096, 4096]),
    );
}
