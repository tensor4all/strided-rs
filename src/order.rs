pub(crate) fn compute_order(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
) -> Vec<usize> {
    let rank = dims.len();
    if rank == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..rank).collect();
    order.sort_by(|&a, &b| {
        let score_a = dim_score(a, strides_list, dest_index);
        let score_b = dim_score(b, strides_list, dest_index);
        score_b.cmp(&score_a).then_with(|| a.cmp(&b))
    });
    order
}

fn dim_score(dim: usize, strides_list: &[&[isize]], dest_index: Option<usize>) -> usize {
    let mut score = 0usize;
    for (i, strides) in strides_list.iter().enumerate() {
        let weight = if dest_index == Some(i) { 2 } else { 1 };
        let stride = strides[dim].unsigned_abs();
        score = score.saturating_add(weight * stride);
    }
    score
}
