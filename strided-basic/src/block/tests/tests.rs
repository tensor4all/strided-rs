use super::*;

#[test]
fn test_total_memory_region_contiguous() {
    // Contiguous array: 100 elements * 8 bytes = 800 bytes
    // Should be about 800 bytes (plus cache line rounding)
    let dims = [100usize];
    let strides = [8u128]; // 8 bytes per element
    let byte_strides: Vec<&[u128]> = vec![&strides];

    let region = total_memory_region(&dims, &byte_strides);

    // 99 * 8 = 792 bytes contiguous
    // 792 / 64 + 1 = 13 cache lines
    // 64 * 13 = 832 bytes
    assert_eq!(region, 832);
}

#[test]
fn test_total_memory_region_strided() {
    // Non-contiguous: stride larger than cache line
    let dims = [10usize];
    let strides = [128u128]; // Stride >= cache line
    let byte_strides: Vec<&[u128]> = vec![&strides];

    let region = total_memory_region(&dims, &byte_strides);

    // Each element touches separate cache line blocks
    // 64 * 1 * 10 = 640 bytes
    assert_eq!(region, 640);
}

#[test]
fn test_compute_blocks_small() {
    // Small array that fits in cache
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

    // Should use full dimensions since it fits
    assert_eq!(blocks.as_slice(), vec![10, 10]);
}

#[test]
fn test_compute_blocks_large() {
    // Large array that needs blocking
    let dims = [1000usize, 1000];
    let costs = [2isize, 2];
    let strides = [8u128, 8000]; // 8 bytes per f64
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

    // Should reduce block sizes
    assert!(blocks[0] <= dims[0]);
    assert!(blocks[1] <= dims[1]);
    // But should still be reasonable
    assert!(blocks[0] >= 1);
    assert!(blocks[1] >= 1);
}

#[test]
fn test_last_argmax_weighted() {
    let blocks = [10usize, 20, 5];
    let costs = [1isize, 1, 2];

    let idx = last_argmax_weighted(&blocks, &costs);

    // (10-1)*1=9, (20-1)*1=19, (5-1)*2=8
    // Maximum is 19 at index 1
    assert_eq!(idx, Some(1));
}

#[test]
fn test_last_argmax_weighted_tie() {
    // When tied, returns last index
    let blocks = [10usize, 10];
    let costs = [1isize, 1];

    let idx = last_argmax_weighted(&blocks, &costs);

    // Both have score 9, return last (index 1)
    assert_eq!(idx, Some(1));
}

#[test]
fn test_compute_block_sizes_full_pipeline() {
    // Test the full compute_block_sizes function
    let dims = [100usize, 100];
    let strides = [1isize, 100];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let blocks = compute_block_sizes(&dims, &strides_list, 8);

    // Should return valid block sizes
    assert_eq!(blocks.len(), 2);
    assert!(blocks[0] >= 1 && blocks[0] <= 100);
    assert!(blocks[1] >= 1 && blocks[1] <= 100);
}

// ========== Julia-comparison tests ==========

#[test]
fn test_total_memory_region_julia_match_2d() {
    // Julia: dims=(10,10), bytestrides=((8,80),)
    // numcontigeouscachelines: 9*8 + 9*80 = 72 + 720 = 792 (but 80 >= 64!)
    // Actually: stride 8 < 64, so += (10-1)*8 = 72
    //          stride 80 >= 64, so numcachelineblocks *= 10
    // contiguous_lines = 72/64 + 1 = 2
    // memory = 64 * 2 * 10 = 1280
    let dims = [10usize, 10];
    let strides = [8u128, 80];
    let byte_strides: Vec<&[u128]> = vec![&strides];

    let region = total_memory_region(&dims, &byte_strides);
    assert_eq!(region, 1280);
}

#[test]
fn test_total_memory_region_julia_match_multiple_arrays() {
    // Two arrays with different strides
    let dims = [10usize, 10];
    let strides1 = [8u128, 80]; // Column-major-ish
    let strides2 = [80u128, 8]; // Row-major-ish
    let byte_strides: Vec<&[u128]> = vec![&strides1, &strides2];

    let region = total_memory_region(&dims, &byte_strides);

    // Array 1: stride 8 < 64 -> contiguous += 72, stride 80 >= 64 -> blocks *= 10
    //          contiguous_lines = 2, region1 = 64 * 2 * 10 = 1280
    // Array 2: stride 80 >= 64 -> blocks *= 10, stride 8 < 64 -> contiguous += 72
    //          contiguous_lines = 2, region2 = 64 * 2 * 10 = 1280
    // Total = 2560
    assert_eq!(region, 2560);
}

#[test]
fn test_total_memory_region_all_contiguous() {
    // All strides < cache line
    let dims = [10usize, 5];
    let strides = [8u128, 40]; // Both < 64
    let byte_strides: Vec<&[u128]> = vec![&strides];

    let region = total_memory_region(&dims, &byte_strides);

    // contiguous = (10-1)*8 + (5-1)*40 = 72 + 160 = 232
    // contiguous_lines = 232/64 + 1 = 4
    // blocks = 1 (no large strides)
    // region = 64 * 4 * 1 = 256
    assert_eq!(region, 256);
}

#[test]
fn test_total_memory_region_all_large_strides() {
    // All strides >= cache line
    let dims = [5usize, 4];
    let strides = [64u128, 320]; // Both >= 64
    let byte_strides: Vec<&[u128]> = vec![&strides];

    let region = total_memory_region(&dims, &byte_strides);

    // contiguous = 0
    // contiguous_lines = 0/64 + 1 = 1
    // blocks = 5 * 4 = 20
    // region = 64 * 1 * 20 = 1280
    assert_eq!(region, 1280);
}

#[test]
fn test_compute_blocks_first_dim_smallest_stride() {
    // When first dimension has smallest stride for all arrays,
    // Julia keeps d1 and recurses on tail
    let dims = [100usize, 10, 10];
    let costs = [2isize, 2, 2];
    // All arrays have first dim with smallest stride
    let strides1 = [8u128, 800, 8000];
    let orders1 = [1usize, 2, 3]; // First dim has order 1 (smallest)
    let byte_strides: Vec<&[u128]> = vec![&strides1];
    let stride_orders: Vec<&[usize]> = vec![&orders1];

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    // First dimension should be kept as-is (100)
    assert_eq!(blocks[0], 100);
}

#[test]
fn test_compute_blocks_min_stride_larger_than_blocksize() {
    // When minimum stride > blocksize AND total memory > blocksize, return all 1s
    // Need:
    // 1. Total memory > blocksize
    // 2. First dim does NOT have smallest stride (to skip the special case)
    // 3. All strides > blocksize
    let dims = [100usize, 100];
    let costs = [2isize, 2];
    // Stride order: dim 1 has smaller stride than dim 0, but both > blocksize
    let strides = [40000000u128, 40000]; // dim 1 is smaller but still > 32768
    let orders = [2usize, 1]; // dim 1 has smaller stride order
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    // Verify total memory exceeds blocksize first
    let initial_mem = total_memory_region(&dims, &byte_strides);
    assert!(
        initial_mem > BLOCK_MEMORY_SIZE as u128,
        "Initial memory {} should exceed {}",
        initial_mem,
        BLOCK_MEMORY_SIZE
    );

    // Verify minimum stride > blocksize
    let min_stride = strides.iter().copied().min().unwrap();
    assert!(
        min_stride > BLOCK_MEMORY_SIZE as u128,
        "Min stride {} should exceed {}",
        min_stride,
        BLOCK_MEMORY_SIZE
    );

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    // With minimum stride > blocksize, should return all 1s
    assert_eq!(blocks.as_slice(), vec![1, 1]);
}

#[test]
fn test_compute_blocks_4d_array_first_dim_smallest() {
    // 4D array where first dim has smallest stride (column-major case)
    // Julia's algorithm keeps first dimension when it has smallest stride for cache locality
    let dims = [10usize, 10, 10, 10];
    let costs = [2isize, 2, 2, 2];
    // Column-major strides: first dim has smallest stride
    let strides = [8u128, 80, 800, 8000];
    let orders = [1usize, 2, 3, 4];
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    // When first dim has smallest stride, Julia keeps full dims via recursion
    // This is by design for cache efficiency
    assert_eq!(blocks.len(), 4);
    for (i, &b) in blocks.iter().enumerate() {
        assert!(b >= 1 && b <= dims[i], "Block {} out of range", i);
    }
}

#[test]
fn test_compute_blocks_4d_needs_reduction() {
    // 4D array where blocking reduction is actually needed
    // Mix strides so first dim doesn't have globally smallest stride
    let dims = [100usize, 100, 100, 100];
    let costs = [2isize, 2, 2, 2];
    // Mixed strides: dim 1 has smallest stride, not dim 0
    let strides = [80u128, 8, 8000, 800];
    let orders = [2usize, 1, 4, 3]; // Reflects actual stride magnitudes
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    // Verify total memory exceeds blocksize
    let initial_mem = total_memory_region(&dims, &byte_strides);
    assert!(initial_mem > BLOCK_MEMORY_SIZE as u128);

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    // Blocks should be reduced from original dims
    assert_eq!(blocks.len(), 4);
    let total_elements: usize = blocks.iter().product();
    let original_elements: usize = dims.iter().product();
    assert!(
        total_elements < original_elements,
        "Blocks should be smaller than original"
    );
}

#[test]
fn test_compute_blocks_4d_permuted() {
    // 4D array with permuted strides (Issue #5 scenario)
    let dims = [10usize, 10, 10, 10];
    let costs = [2isize, 4, 8, 16]; // Different costs
                                    // Permuted strides: last dim has smallest stride
    let strides = [8000u128, 800, 80, 8];
    let orders = [4usize, 3, 2, 1];
    let byte_strides: Vec<&[u128]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    assert_eq!(blocks.len(), 4);
    // Verify blocks are valid
    for (i, &b) in blocks.iter().enumerate() {
        assert!(b >= 1 && b <= dims[i]);
    }
}

#[test]
fn test_compute_blocks_mixed_strides_two_arrays() {
    // Two arrays with conflicting stride patterns
    let dims = [100usize, 100];
    let costs = [2isize, 2];
    let strides1 = [8u128, 800]; // Column-major
    let strides2 = [800u128, 8]; // Row-major
    let orders1 = [1usize, 2];
    let orders2 = [2usize, 1];
    let byte_strides: Vec<&[u128]> = vec![&strides1, &strides2];
    let stride_orders: Vec<&[usize]> = vec![&orders1, &orders2];

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    // Should not use the first-dim-smallest special case
    // since arrays have different stride orders
    assert_eq!(blocks.len(), 2);
    assert!(blocks[0] >= 1 && blocks[0] <= 100);
    assert!(blocks[1] >= 1 && blocks[1] <= 100);
}

#[test]
fn test_last_argmax_weighted_all_ones() {
    // When all blocks are 1, should return None
    let blocks = [1usize, 1, 1];
    let costs = [1isize, 2, 3];

    let idx = last_argmax_weighted(&blocks, &costs);
    assert_eq!(idx, None);
}

#[test]
fn test_last_argmax_weighted_mixed() {
    // Mix of 1s and larger values
    let blocks = [1usize, 5, 3];
    let costs = [100isize, 1, 1];

    let idx = last_argmax_weighted(&blocks, &costs);

    // (1-1)*100=0, (5-1)*1=4, (3-1)*1=2
    // Max is 4 at index 1
    assert_eq!(idx, Some(1));
}

#[test]
fn test_last_argmax_weighted_cost_matters() {
    // Higher cost can outweigh larger block
    let blocks = [3usize, 10];
    let costs = [10isize, 1];

    let idx = last_argmax_weighted(&blocks, &costs);

    // (3-1)*10=20, (10-1)*1=9
    // Max is 20 at index 0
    assert_eq!(idx, Some(0));
}

#[test]
fn test_compute_blocks_negative_strides() {
    // Negative strides are handled via their magnitude: same blocks as the
    // positive layout.
    let dims = [100usize, 100];
    let positive = compute_block_sizes(&dims, &[&[1, 100], &[100, 1]], 8);
    let negative = compute_block_sizes(&dims, &[&[-1, -100], &[-100, 1]], 8);
    assert_eq!(negative, positive);
}

#[test]
fn test_compute_block_sizes_4d_column_major() {
    // Full pipeline test for 4D arrays (Issue #5)
    // Column-major layout: first dim has smallest stride
    let dims = [32usize, 32, 32, 32];
    let strides = [1isize, 32, 1024, 32768]; // Column-major
    let strides_list: Vec<&[isize]> = vec![&strides];

    let blocks = compute_block_sizes(&dims, &strides_list, 8);

    assert_eq!(blocks.len(), 4);
    // Verify blocks are valid
    for (i, &b) in blocks.iter().enumerate() {
        assert!(b >= 1 && b <= dims[i], "Block {} = {} out of range", i, b);
    }
}

#[test]
fn test_compute_block_sizes_4d_permuted_strides() {
    // Row-major strides [32768, 1024, 32, 1] in the reversed iteration order
    // [3, 2, 1, 0]: the caller passes the metadata already in that order, so
    // the strides strictly increase and every axis is kept whole.
    let dims = [32usize, 32, 32, 32];
    let strides = [1isize, 32, 1024, 32768];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let blocks = compute_block_sizes(&dims, &strides_list, 8);

    assert_eq!(blocks.as_slice(), [32, 32, 32, 32]);
}

// ========== Exact arithmetic contract ==========

#[path = "oracle.rs"]
mod oracle;

use crate::kernel::{build_plan, build_plan_fused};
use oracle::{legacy_blocks, model_blocks, LegacyCosts};

fn refs(strides: &[Vec<isize>]) -> Vec<&[isize]> {
    strides.iter().map(|s| s.as_slice()).collect()
}

#[test]
fn compute_block_sizes_equals_the_exact_model_on_the_sweep() {
    let (mut fits, mut overflows) = (0, 0);
    for (dims, strides, elem_size) in oracle::sweep() {
        let strides = refs(&strides);
        let model = model_blocks(&dims, &strides, elem_size);
        let blocks = compute_block_sizes(&dims, &strides, elem_size);
        assert_eq!(
            blocks.as_slice(),
            model,
            "{dims:?} {strides:?} e={elem_size}"
        );
        if elem_size == 0 {
            assert_eq!(model, dims, "a zero element size keeps every axis");
        }
        match legacy_blocks(&dims, &strides, elem_size, LegacyCosts::Saturating) {
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
    // Rank 0 returns before any conversion of the element size.
    assert!(compute_block_sizes(&[], &[&[]], usize::MAX).is_empty());
    assert_eq!(
        legacy_blocks(&[], &[&[]], usize::MAX, LegacyCosts::Saturating),
        Some(vec![])
    );
}

#[test]
fn a_zip_reduce_region_beyond_usize_is_exact() {
    // R1-A: two f64 operands over 1024^6, whose region 2^66 per operand
    // wrapped to 0 ("fits") before the exact contract.
    let dims = [1024usize; 6];
    let a = [8isize, 16, 8, 8, 8, 8];
    let b = [16isize, 8, 8, 8, 8, 8];
    let (dims, strides, plan) = build_plan_fused(&dims, &[&a, &b], None, 8);
    assert_eq!(dims.as_slice(), [1024; 6]);
    assert_eq!(strides[0].as_slice(), [8, 8, 8, 8, 8, 16]);
    assert_eq!(strides[1].as_slice(), [8, 8, 8, 8, 16, 8]);
    assert_eq!(plan.block.as_slice(), [1024, 1024, 1024, 1024, 16, 16]);
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    assert_eq!(model_blocks(&dims, &strides, 8), plan.block.as_slice());
    assert_eq!(
        legacy_blocks(&dims, &strides, 8, LegacyCosts::Saturating),
        None
    );
}

/// Zero-sized source element for the mixed-size `map_into` fixture.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Zst;

impl crate::ElementOpApply for Zst {}

static ZSTS: [Zst; 1 << 62] = [Zst; 1 << 62];

#[test]
fn a_zero_sized_source_byte_stride_beyond_isize_is_exact() {
    // R1-B: f64 destination from a zero-sized source. The planner converts
    // both with the larger element size, so the source byte stride
    // 2^61 * 8 = 2^64 wrapped to 0 before the exact contract.
    let dims = [2usize, 8192];
    let dst_strides = [1isize, 2];
    let src_strides = [1isize << 61, 1];
    let (_, _, plan) = build_plan_fused(&dims, &[&dst_strides, &src_strides], Some(0), 8);
    assert_eq!(plan.block.as_slice(), [2, 1024]);
    assert_eq!(
        legacy_blocks(
            &dims,
            &[&dst_strides, &src_strides],
            8,
            LegacyCosts::Saturating
        ),
        None
    );

    let src = crate::StridedView::<Zst>::new(&ZSTS[..], &dims, &src_strides, 0).unwrap();
    let mut data = vec![0.0f64; 2 * 8192];
    let mut dst = crate::StridedViewMut::new(&mut data, &dims, &dst_strides, 0).unwrap();
    crate::map_into(&mut dst, &src, |_: Zst| 1.5f64, crate::Exec::serial()).unwrap();
    assert!(data.iter().all(|&x| x == 1.5));
}

#[test]
fn a_zero_element_size_keeps_every_axis_for_any_operand_count() {
    // R3-1: 513 operands, region 513 * 64 = 32832 above the budget, but every
    // byte stride is 0, so every stride order is 1 and both axes are kept.
    let dims = [2usize, 2];
    let mut strides = vec![vec![1isize, 2]; 512];
    strides.push(vec![2, 1]);
    let strides = refs(&strides);
    assert_eq!(compute_block_sizes(&dims, &strides, 0).as_slice(), [2, 2]);
    assert_eq!(model_blocks(&dims, &strides, 0), [2, 2]);
    assert_eq!(
        legacy_blocks(&dims, &strides, 0, LegacyCosts::Saturating),
        Some(vec![2, 2])
    );
    // Stride orders taken from the element strides instead would shrink both.
    let orders: Vec<InlineVec<usize>> = strides.iter().map(|s| index_order(s)).collect();
    let orders: Vec<&[usize]> = orders.iter().map(|o| o.as_slice()).collect();
    let zeros = [0u128; 2];
    let byte_strides = vec![&zeros[..]; 513];
    let costs = compute_costs(&strides);
    assert_eq!(
        compute_blocks(&dims, &costs, &byte_strides, &orders, BLOCK_MEMORY_SIZE).as_slice(),
        [1, 1]
    );
}

#[test]
fn a_halving_score_beyond_isize_selects_the_exact_maximum() {
    // R3-3: costs [2^62, 2]; the score (3 - 1) * 2^62 = 2^63 wrapped to
    // -2^63, which made the old core shrink axis 1 to [3, 3616].
    let dims = [3usize, 8192];
    let a = [1isize << 61, 1];
    let b = [1isize << 61, 2];
    assert_eq!(compute_costs(&[&a, &b]).as_slice(), [1 << 62, 2]);
    // Regions 73728 -> 49152 (halved) -> 24576 (decremented).
    assert_eq!(
        compute_block_sizes(&dims, &[&a, &b], 1).as_slice(),
        [1, 8192]
    );
    assert_eq!(model_blocks(&dims, &[&a, &b], 1), [1, 8192]);
    assert_eq!(
        legacy_blocks(&dims, &[&a, &b], 1, LegacyCosts::Saturating),
        None
    );
}

#[test]
fn tied_strides_run_the_full_core() {
    // F1: the source ties (or is broadcast on) the destination's leading axes,
    // so the strictly increasing exit does not apply.
    let dims = [64usize, 64, 64];
    let dst = [1isize, 64, 4096];
    for src in [[0isize, 1, 1], [1, 1, 1]] {
        assert_eq!(
            compute_block_sizes(&dims, &[&dst, &src], 8).as_slice(),
            [64, 31, 16]
        );
        assert_eq!(model_blocks(&dims, &[&dst, &src], 8), [64, 31, 16]);
    }
}

#[test]
fn strictly_increasing_strides_keep_every_axis_in_the_model() {
    // crop r2 and reverse r2: the model without exits keeps every axis, as
    // the strictly increasing exit does.
    let cases: [(&[usize], [isize; 2], [isize; 2]); 2] = [
        (&[3840, 4096], [1, 3840], [1, 4096]),
        (&[4096, 4096], [1, 4096], [-1, 4096]),
    ];
    for (dims, dst, src) in cases {
        assert_eq!(model_blocks(dims, &[&dst, &src], 8), dims);
        assert_eq!(compute_block_sizes(dims, &[&dst, &src], 8).as_slice(), dims);
    }
}

#[test]
fn build_plan_forwards_ordered_dims_and_strides() {
    // R3-5: [3, 5] row-major iterates in order [1, 0], so the block core
    // sees dims [5, 3] and strides [1, 5].
    let plan = build_plan(&[3, 5], &[&[5, 1]], Some(0), 8);
    assert_eq!(plan.order.as_slice(), [1, 0]);
    assert_eq!(plan.block.as_slice(), [5, 3]);
    // [3, 1000]: forwarded strides [1, 1000] fit (region 24000); unreordered
    // strides [1000, 1] would not (64000) and give blocks [512, 3].
    let plan = build_plan(&[3, 1000], &[&[1000, 1]], Some(0), 8);
    assert_eq!(plan.order.as_slice(), [1, 0]);
    assert_eq!(plan.block.as_slice(), [1000, 3]);
}

/// Column-major strides of `dims` laid out in the axis order `perm`.
fn permuted_strides(dims: &[usize], perm: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    let mut stride = 1isize;
    for &axis in perm {
        strides[axis] = stride;
        stride *= dims[axis] as isize;
    }
    strides
}

#[test]
fn flat_scratch_matches_the_model_across_its_spill_boundaries() {
    // T9: ranks around INLINE_RANK, operand counts around OPERAND_CAP, and
    // flat tables of 16, 18, 40, 45 and 72 entries; each operand walks the
    // axes in a different order, so every case reaches the core.
    let cases: [(usize, usize); 5] = [(8, 2), (9, 2), (8, 5), (9, 5), (12, 6)];
    for (rank, operands) in cases {
        let dims: Vec<usize> = (0..rank).map(|i| 2 + i % 3).collect();
        let strides: Vec<Vec<isize>> = (0..operands)
            .map(|k| {
                // Rotate by k, reversed for odd k: a distinct order per operand.
                let mut perm: Vec<usize> = (0..rank).map(|i| (i + k) % rank).collect();
                if k % 2 == 1 {
                    perm.reverse();
                }
                permuted_strides(&dims, &perm)
            })
            .collect();
        let strides = refs(&strides);
        assert!(!strides.iter().all(|s| strictly_increasing(s)));
        let blocks = compute_block_sizes(&dims, &strides, 8);
        assert_ne!(
            blocks.as_slice(),
            dims.as_slice(),
            "rank {rank}, {operands} operands"
        );
        assert_eq!(
            blocks.as_slice(),
            model_blocks(&dims, &strides, 8),
            "rank {rank}, {operands} operands"
        );
    }

    // Keep-first recursion: axes 0 and 1 lead every operand's stride order,
    // so the core keeps them at offsets 0 and 1 and reduces the suffix at
    // offset 2, where the operands disagree; the suffix rows are read at the
    // original pitch.
    let dims = [8usize, 5, 6, 7, 4, 3];
    let strides: Vec<Vec<isize>> = [
        [0usize, 1, 2, 3, 4, 5],
        [0, 1, 5, 4, 3, 2],
        [0, 1, 3, 5, 2, 4],
    ]
    .iter()
    .map(|perm| permuted_strides(&dims, perm))
    .collect();
    let strides = refs(&strides);
    let blocks = compute_block_sizes(&dims, &strides, 8);
    assert_eq!(blocks[..2], dims[..2]);
    assert_ne!(blocks.as_slice(), dims.as_slice());
    assert_eq!(blocks.as_slice(), model_blocks(&dims, &strides, 8));
    assert_eq!(blocks.as_slice(), [8, 5, 5, 5, 2, 3]);
}
