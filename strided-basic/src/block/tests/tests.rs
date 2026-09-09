use super::*;

#[test]
fn test_total_memory_region_contiguous() {
    // Contiguous array: 100 elements * 8 bytes = 800 bytes
    // Should be about 800 bytes (plus cache line rounding)
    let dims = [100usize];
    let strides = [8isize]; // 8 bytes per element
    let byte_strides: Vec<&[isize]> = vec![&strides];

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
    let strides = [128isize]; // Stride >= cache line
    let byte_strides: Vec<&[isize]> = vec![&strides];

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
    let strides = [8isize, 80];
    let orders = [1usize, 2];
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    // Should use full dimensions since it fits
    assert_eq!(blocks, vec![10, 10]);
}

#[test]
fn test_compute_blocks_large() {
    // Large array that needs blocking
    let dims = [1000usize, 1000];
    let costs = [2isize, 2];
    let strides = [8isize, 8000]; // 8 bytes per f64
    let orders = [1usize, 2];
    let byte_strides: Vec<&[isize]> = vec![&strides];
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
    let order = [0usize, 1];
    let strides = [1isize, 100];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let blocks = compute_block_sizes(&dims, &order, &strides_list, 8);

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
    let strides = [8isize, 80];
    let byte_strides: Vec<&[isize]> = vec![&strides];

    let region = total_memory_region(&dims, &byte_strides);
    assert_eq!(region, 1280);
}

#[test]
fn test_total_memory_region_julia_match_multiple_arrays() {
    // Two arrays with different strides
    let dims = [10usize, 10];
    let strides1 = [8isize, 80]; // Column-major-ish
    let strides2 = [80isize, 8]; // Row-major-ish
    let byte_strides: Vec<&[isize]> = vec![&strides1, &strides2];

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
    let strides = [8isize, 40]; // Both < 64
    let byte_strides: Vec<&[isize]> = vec![&strides];

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
    let strides = [64isize, 320]; // Both >= 64
    let byte_strides: Vec<&[isize]> = vec![&strides];

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
    let strides1 = [8isize, 800, 8000];
    let orders1 = [1usize, 2, 3]; // First dim has order 1 (smallest)
    let byte_strides: Vec<&[isize]> = vec![&strides1];
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
    let strides = [40000000isize, 40000]; // dim 1 is smaller but still > 32768
    let orders = [2usize, 1]; // dim 1 has smaller stride order
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    // Verify total memory exceeds blocksize first
    let initial_mem = total_memory_region(&dims, &byte_strides);
    assert!(
        initial_mem > BLOCK_MEMORY_SIZE,
        "Initial memory {} should exceed {}",
        initial_mem,
        BLOCK_MEMORY_SIZE
    );

    // Verify minimum stride > blocksize
    let min_stride = strides.iter().map(|s| s.unsigned_abs()).min().unwrap();
    assert!(
        min_stride > BLOCK_MEMORY_SIZE,
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
    assert_eq!(blocks, vec![1, 1]);
}

#[test]
fn test_compute_blocks_4d_array_first_dim_smallest() {
    // 4D array where first dim has smallest stride (column-major case)
    // Julia's algorithm keeps first dimension when it has smallest stride for cache locality
    let dims = [10usize, 10, 10, 10];
    let costs = [2isize, 2, 2, 2];
    // Column-major strides: first dim has smallest stride
    let strides = [8isize, 80, 800, 8000];
    let orders = [1usize, 2, 3, 4];
    let byte_strides: Vec<&[isize]> = vec![&strides];
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
    let strides = [80isize, 8, 8000, 800];
    let orders = [2usize, 1, 4, 3]; // Reflects actual stride magnitudes
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    // Verify total memory exceeds blocksize
    let initial_mem = total_memory_region(&dims, &byte_strides);
    assert!(initial_mem > BLOCK_MEMORY_SIZE);

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
    let strides = [8000isize, 800, 80, 8];
    let orders = [4usize, 3, 2, 1];
    let byte_strides: Vec<&[isize]> = vec![&strides];
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
    let strides1 = [8isize, 800]; // Column-major
    let strides2 = [800isize, 8]; // Row-major
    let orders1 = [1usize, 2];
    let orders2 = [2usize, 1];
    let byte_strides: Vec<&[isize]> = vec![&strides1, &strides2];
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
    // Negative strides should be handled via absolute value
    let dims = [10usize, 10];
    let costs = [2isize, 2];
    let strides = [-8isize, -80]; // Negative strides
    let orders = [1usize, 2];
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];

    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );

    // Should behave same as positive strides
    assert_eq!(blocks.len(), 2);
    assert!(blocks[0] >= 1 && blocks[0] <= 10);
    assert!(blocks[1] >= 1 && blocks[1] <= 10);
}

#[test]
fn test_compute_block_sizes_4d_column_major() {
    // Full pipeline test for 4D arrays (Issue #5)
    // Column-major layout: first dim has smallest stride
    let dims = [32usize, 32, 32, 32];
    let order = [0usize, 1, 2, 3]; // Natural order
    let strides = [1isize, 32, 1024, 32768]; // Column-major
    let strides_list: Vec<&[isize]> = vec![&strides];

    let blocks = compute_block_sizes(&dims, &order, &strides_list, 8);

    assert_eq!(blocks.len(), 4);
    // Verify blocks are valid
    for (i, &b) in blocks.iter().enumerate() {
        assert!(b >= 1 && b <= dims[i], "Block {} = {} out of range", i, b);
    }
}

#[test]
fn test_compute_block_sizes_4d_permuted_strides() {
    // Full pipeline test for 4D arrays with permuted strides
    // This exercises the halving/decrementing reduction path
    let dims = [32usize, 32, 32, 32];
    let order = [3usize, 2, 1, 0]; // Reversed order
                                   // Permuted strides: smallest stride is in last position of original
    let strides = [32768isize, 1024, 32, 1];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let blocks = compute_block_sizes(&dims, &order, &strides_list, 8);

    assert_eq!(blocks.len(), 4);
    for (i, &b) in blocks.iter().enumerate() {
        assert!(b >= 1, "Block {} must be >= 1", i);
    }
}
