use crate::BLOCK_MEMORY_SIZE;

pub(crate) fn compute_block_sizes(
    dims: &[usize],
    order: &[usize],
    strides_list: &[&[isize]],
    elem_size: usize,
) -> Vec<usize> {
    if order.is_empty() {
        return Vec::new();
    }

    let arrays = strides_list.len().max(1);
    let per_array_bytes = BLOCK_MEMORY_SIZE / arrays;
    let per_array_elems = (per_array_bytes / elem_size).max(1);

    let mut block_sizes = vec![1; order.len()];
    let mut max_offsets: Vec<usize> = vec![0; arrays];

    for (rev_level, &dim) in order.iter().rev().enumerate() {
        let level = order.len() - 1 - rev_level;
        let dim_len = dims[dim].max(1);

        let mut block_len = dim_len;
        for (a_idx, strides) in strides_list.iter().enumerate() {
            let stride = strides[dim].unsigned_abs();
            if stride == 0 {
                continue;
            }
            let used = max_offsets[a_idx];
            let available = per_array_elems.saturating_sub(used + 1);
            let allowed = 1 + available / stride;
            block_len = block_len.min(allowed);
        }

        block_len = block_len.max(1);
        block_sizes[level] = block_len;

        for (a_idx, strides) in strides_list.iter().enumerate() {
            let stride = strides[dim].unsigned_abs();
            if stride == 0 {
                continue;
            }
            max_offsets[a_idx] = max_offsets[a_idx].saturating_add((block_len - 1) * stride);
        }
    }

    block_sizes
}
