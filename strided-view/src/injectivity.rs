//! Output-layout injectivity validation.
//!
//! Kernel families that write through a strided destination must prove that
//! distinct logical positions map to distinct memory offsets. This is the one
//! owner of that analysis: `strided-basic` re-exports it for its kernel
//! families and `strided-perm` calls it before its unsafe permutation loops.
//!
//! The common case is validated without allocating: axes are reduced in
//! ascending `|stride|` order and only the interleaved block of small-stride
//! axes is compared pairwise. A block larger than the pairwise budget cannot
//! be settled pairwise, so it spills to the exact (allocating) enumeration.
//! Both checks conservatively reject metadata whose spans do not fit the
//! analysis, which keeps the entry points panic-free.

/// Decide whether distinct logical positions map to distinct offsets.
///
/// Runs the allocation-free pairwise check first and falls back to the exact
/// enumeration only when the interleaved block exceeds the pairwise budget.
/// This accepts every injective layout the exact check can settle, including
/// the interleaved `[3, 1500]` / `[2000, 3]` case, without allocating for the
/// common separated or small-block layouts.
///
/// # Examples
///
/// ```
/// use strided_view::is_injective_layout;
/// assert!(is_injective_layout(&[2, 3], &[1, 2]));
/// assert!(!is_injective_layout(&[2, 3], &[0, 1]));
/// // Interleaved but injective: 2000 = 2 (mod 3) keeps the rows apart.
/// assert!(is_injective_layout(&[3, 1500], &[2000, 3]));
/// ```
pub fn is_injective_layout(dims: &[usize], strides: &[isize]) -> bool {
    is_injective_layout_without_alloc(dims, strides) || is_injective_layout_exact(dims, strides)
}

/// Allocation-free injectivity check.
///
/// Exact for interleaved blocks of at most `PAIRWISE_BLOCK_BUDGET` logical
/// elements; larger blocks are conservatively rejected (callers that must keep
/// every injective layout use [`is_injective_layout`] instead, which spills to
/// the exact enumeration). Exposed for the kernel adapters that validate a
/// destination on a per-call path.
#[doc(hidden)]
pub fn is_injective_layout_without_alloc(dims: &[usize], strides: &[isize]) -> bool {
    let Some(total) = validate_injective_layout_inputs(dims, strides) else {
        return false;
    };
    if total <= 1 {
        return true;
    }
    match interleaved_block(dims, strides) {
        None => false,
        Some(None) => true,
        Some(Some(block)) => {
            block.may_be_injective()
                && block.total <= PAIRWISE_BLOCK_BUDGET as u128
                && block_offsets_unique_pairwise(dims, strides, &block)
        }
    }
}

/// Conservative, allocation-free proof that every non-singleton axis is
/// separated from the offset span of the smaller-stride axes.
///
/// A layout whose axes interleave is rejected even when it is injective, so a
/// parallel engine that splits the outermost loop can never let two workers
/// write one offset. Exposed for `strided-perm`'s parallel entry.
#[doc(hidden)]
pub fn axes_are_separated(dims: &[usize], strides: &[isize]) -> bool {
    // Negative and zero strides on a non-singleton axis are left unresolved:
    // the conservative proof only reasons about positive, non-collapsed axes.
    if dims
        .iter()
        .zip(strides.iter())
        .any(|(&dim, &stride)| dim > 1 && stride <= 0)
    {
        return false;
    }
    matches!(interleaved_block(dims, strides), Some(None))
}

/// Exact injectivity check used when the pairwise budget is exceeded.
fn is_injective_layout_exact(dims: &[usize], strides: &[isize]) -> bool {
    let Some(total) = validate_injective_layout_inputs(dims, strides) else {
        return false;
    };
    if total <= 1 {
        return true;
    }
    match interleaved_block(dims, strides) {
        None => false,
        Some(None) => true,
        Some(Some(block)) => {
            block.may_be_injective()
                && block.total <= EXACT_BLOCK_BUDGET as u128
                && block_offsets_unique(dims, strides, &block)
        }
    }
}

/// Largest interleaved block enumerated exactly by the spill path.
///
/// INVARIANT: bounds the auxiliary memory of the exact check. The block is
/// visited once in O(block) time; its seen-set is a bitmap over the block's
/// offset span when that span is at most 64 offsets per element, otherwise a
/// sorted offset list, so it never exceeds 8 bytes per block element
/// (128 MiB at this bound). Separated axes never count toward the budget.
const EXACT_BLOCK_BUDGET: usize = 1 << 24;

/// Largest interleaved block compared pairwise by the allocation-free check.
///
/// INVARIANT: keeps the O(block^2) pairwise comparison bounded; larger blocks
/// use the exact spill path. Separated axes never count toward the budget.
const PAIRWISE_BLOCK_BUDGET: usize = 4096;

/// The smallest-stride axes that must be checked by enumeration.
struct InterleavedBlock {
    last_key: (u128, usize),
    /// Product of the block's extents.
    total: u128,
    /// Largest offset reachable inside the block after normalising strides to
    /// their absolute values (the smallest is zero).
    span: u128,
}

impl InterleavedBlock {
    fn contains(&self, axis: usize, dim: usize, stride: isize) -> bool {
        dim > 1 && (stride.unsigned_abs() as u128, axis) <= self.last_key
    }

    /// Pigeonhole: more elements than reachable offsets always alias.
    fn may_be_injective(&self) -> bool {
        self.total <= self.span + 1
    }
}

/// Find the interleaved block of a layout.
///
/// Returns `None` when a stride or span cannot be represented, `Some(None)`
/// when every axis is separated (so the layout is injective), and the block
/// otherwise. Negating a stride only reflects the coordinate along that axis
/// and shifts all offsets by a constant, so the analysis uses `|stride|`.
fn interleaved_block(dims: &[usize], strides: &[isize]) -> Option<Option<InterleavedBlock>> {
    let mut covered_span = 0u128;
    let mut covered_total = 1u128;
    let mut previous_key = None;
    let mut block = None;
    let active_axes = dims.iter().filter(|&&dim| dim > 1).count();
    for _ in 0..active_axes {
        let mut next = None;
        for (axis, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
            if dim <= 1 {
                continue;
            }
            let key = (stride.checked_abs()? as u128, axis);
            if previous_key.is_some_and(|previous| key <= previous) {
                continue;
            }
            if next.is_none_or(|(best, _)| key < best) {
                next = Some((key, dim as u128));
            }
        }
        let ((stride, axis), dim) = next?;
        let separated = stride > covered_span;
        covered_span = stride
            .checked_mul(dim - 1)
            .and_then(|span| covered_span.checked_add(span))?;
        covered_total = covered_total.checked_mul(dim)?;
        if !separated {
            block = Some(InterleavedBlock {
                last_key: (stride, axis),
                total: covered_total,
                span: covered_span,
            });
        }
        previous_key = Some((stride, axis));
    }
    Some(block)
}

/// Enumerate the block's offsets once and report whether they are distinct.
fn block_offsets_unique(dims: &[usize], strides: &[isize], block: &InterleavedBlock) -> bool {
    let axes: Vec<(usize, usize)> = dims
        .iter()
        .zip(strides.iter())
        .enumerate()
        .filter(|&(axis, (&dim, &stride))| block.contains(axis, dim, stride))
        .map(|(_, (&dim, &stride))| (dim, stride.unsigned_abs()))
        .collect();
    let (Ok(total), Ok(span)) = (usize::try_from(block.total), usize::try_from(block.span)) else {
        return false;
    };

    // Every visited offset and every partial axis span lies within
    // [0, span], which fits `usize` (checked above), so the incremental
    // updates below cannot overflow.
    let mut indices = vec![0usize; axes.len()];
    let mut offset = 0usize;
    let mut advance = |offset: &mut usize| {
        for (index, &(dim, stride)) in indices.iter_mut().zip(axes.iter()) {
            if *index + 1 < dim {
                *index += 1;
                *offset += stride;
                return;
            }
            *offset -= stride * (dim - 1);
            *index = 0;
        }
    };

    if block.span / 64 <= block.total {
        let mut seen = vec![0u64; span / 64 + 1];
        for _ in 0..total {
            let (word, bit) = (offset / 64, 1u64 << (offset % 64));
            if seen[word] & bit != 0 {
                return false;
            }
            seen[word] |= bit;
            advance(&mut offset);
        }
        true
    } else {
        let mut offsets = Vec::with_capacity(total);
        for _ in 0..total {
            offsets.push(offset);
            advance(&mut offset);
        }
        offsets.sort_unstable();
        offsets.windows(2).all(|pair| pair[0] != pair[1])
    }
}

fn block_offset_for_linear_index(
    dims: &[usize],
    strides: &[isize],
    block: &InterleavedBlock,
    mut linear: usize,
) -> u128 {
    let mut offset = 0u128;
    for (axis, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
        if !block.contains(axis, dim, stride) {
            continue;
        }
        offset += stride.unsigned_abs() as u128 * (linear % dim) as u128;
        linear /= dim;
    }
    offset
}

fn block_offsets_unique_pairwise(
    dims: &[usize],
    strides: &[isize],
    block: &InterleavedBlock,
) -> bool {
    let Ok(total) = usize::try_from(block.total) else {
        return false;
    };
    for lhs in 0..total {
        let lhs_offset = block_offset_for_linear_index(dims, strides, block, lhs);
        for rhs in (lhs + 1)..total {
            if block_offset_for_linear_index(dims, strides, block, rhs) == lhs_offset {
                return false;
            }
        }
    }
    true
}

fn validate_injective_layout_inputs(dims: &[usize], strides: &[isize]) -> Option<usize> {
    if dims.len() != strides.len() {
        return None;
    }

    // A zero-sized layout is trivially injective even when the product of its
    // other extents would overflow.
    let total = total_len(dims)?;
    if total <= 1 {
        return Some(total);
    }
    if dims
        .iter()
        .zip(strides.iter())
        .any(|(&dim, &stride)| dim > 1 && stride == 0)
    {
        return None;
    }

    let mut min_offset = 0isize;
    let mut max_offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim <= 1 {
            continue;
        }
        let extent = isize::try_from(dim - 1).ok()?;
        let span = stride.checked_mul(extent)?;
        if span >= 0 {
            max_offset = max_offset.checked_add(span)?;
        } else {
            min_offset = min_offset.checked_add(span)?;
        }
    }
    Some(total)
}

/// Checked product of extents; `None` on overflow.
fn total_len(dims: &[usize]) -> Option<usize> {
    // A zero extent wins before the product, matching the kernel helper: a
    // zero-sized layout is trivially injective even when its other extents
    // would overflow.
    if dims.contains(&0) {
        return Some(0);
    }
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_axes_are_injective() {
        assert!(is_injective_layout(&[2, 3], &[1, 2]));
        assert!(is_injective_layout(&[2, 3], &[3, 1]));
        assert!(is_injective_layout(&[1, 5], &[0, 1]));
        assert!(is_injective_layout(&[], &[]));
    }

    #[test]
    fn overlapping_axes_are_rejected() {
        assert!(!is_injective_layout(&[2, 3], &[0, 1]));
        assert!(!is_injective_layout(&[2, 2], &[1, 1]));
        assert!(!is_injective_layout(&[256, 256], &[256, 256]));
        // Stride 0 on a non-singleton axis collapses offsets.
        assert!(!is_injective_layout(&[3, 3], &[1, 0]));
    }

    #[test]
    fn interleaved_injective_layout_needs_the_exact_spill_path() {
        // The allocation-free pairwise budget is 4096; this block has 4500
        // elements, so the first pass rejects it and the exact path accepts.
        assert!(is_injective_layout(&[3, 1500], &[2000, 3]));
        assert!(is_injective_layout_without_alloc(&[3, 1500], &[2000, 3]) == false);
        assert!(is_injective_layout(&[1500, 3], &[3, 2000]));
        assert!(!is_injective_layout(&[3, 1500], &[2000, 2]));
    }

    #[test]
    fn separated_axes_gate_matches_the_conservative_parallel_rule() {
        assert!(axes_are_separated(&[3, 16384], &[1, 3]));
        assert!(!axes_are_separated(&[3, 16384], &[2, 1]));
        assert!(!axes_are_separated(&[4], &[-1]));
        assert!(!axes_are_separated(&[4], &[0]));
        assert!(axes_are_separated(&[1, 1], &[0, 1]));
    }

    #[test]
    fn checked_total_len_detects_overflow() {
        assert_eq!(total_len(&[usize::MAX, 2]), None);
        assert_eq!(total_len(&[3, 4]), Some(12));
        // A zero extent wins before the overflowing product.
        assert_eq!(total_len(&[usize::MAX, 2, 0]), Some(0));
    }
}
