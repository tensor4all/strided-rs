//! Output-layout injectivity validation.
//!
//! Kernel families that write through a strided destination must prove that
//! distinct logical positions map to distinct memory offsets. This is the one
//! owner of that analysis: `strided-basic` re-exports it for its kernel
//! families and `strided-perm` calls it before its unsafe permutation loops.
//!
//! The common case is validated without allocating: axes are reduced in
//! ascending `|stride|` order and only the interleaved block of small-stride
//! axes is compared pairwise. A two-axis block is settled exactly by a gcd
//! argument regardless of size. A block of three or more axes is settled
//! exactly by searching for a colliding difference vector over all but its two
//! largest axes, which is cheap whenever the other axes are short, regardless
//! of the block's element count. A block whose difference search exceeds its
//! budget falls back to pairwise comparison, then to the exact (allocating)
//! enumeration. All paths conservatively reject metadata whose spans do not
//! fit the analysis, which keeps the entry points panic-free.

/// Decide whether distinct logical positions map to distinct offsets.
///
/// Runs the allocation-free proof first (separated axes, the two-axis gcd
/// argument, the difference search, then the pairwise budget) and falls back
/// to the exact enumeration only when all of those are out of budget. This
/// accepts the interleaved `[3, 1500]` / `[2000, 3]` and
/// `[2, 2, 5_000_000]` / `[4, 5, 6]` cases without allocating.
///
/// `true` always means injective. `false` means the layout aliases, except
/// for one documented residual: an interleaved block of three or more axes
/// whose difference search (half the product of `2 * extent - 1` over all but
/// its two longest axes) exceeds 2^22 vectors *and* whose element count
/// exceeds 2^24 is not decided and is reported as `false`. Such a block needs
/// at least three interleaved axes with long extents.
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
/// Exact for a two-axis interleaved block (any size), for a block whose
/// difference search fits `DIFFERENCE_SEARCH_BUDGET`, and for interleaved
/// blocks of at most `PAIRWISE_BLOCK_BUDGET` logical elements; other blocks
/// are conservatively rejected (callers that must keep
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
            if !block.may_be_injective() {
                // Pigeonhole: more elements than reachable offsets always alias.
                return false;
            }
            if let Some(injective) = two_axis_block_is_injective(dims, strides, &block) {
                return injective;
            }
            if let Some(injective) = block_differences_avoid_zero(dims, strides, &block) {
                return injective;
            }
            block.total <= PAIRWISE_BLOCK_BUDGET as u128
                && block_offsets_unique_pairwise(dims, strides, &block)
        }
    }
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
            if !block.may_be_injective() {
                // Pigeonhole: more elements than reachable offsets always alias.
                return false;
            }
            if let Some(injective) = two_axis_block_is_injective(dims, strides, &block) {
                return injective;
            }
            // The difference search already ran in the allocation-free path
            // and was out of budget.
            block.total <= EXACT_BLOCK_BUDGET as u128 && block_offsets_unique(dims, strides, &block)
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

/// Largest number of difference vectors visited by the difference search.
///
/// INVARIANT: bounds the time of the allocation-free difference search in
/// [`block_differences_avoid_zero`]. Each visit is O(1) integer arithmetic
/// after an O(rank) setup, and the count is half the product of
/// `2 * extent - 1` over every block axis except the two largest, so the
/// bound is independent of the block's element count.
const DIFFERENCE_SEARCH_BUDGET: u128 = 1 << 22;

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

/// Exact, allocation-free injectivity decision for a two-axis interleaved
/// block.
///
/// For strides `s1 <= s2` and extents `d1`, `d2`, the sums `i*s1 + j*s2` are
/// all distinct iff `s2/g >= d1` or `s1/g >= d2`, where `g = gcd(s1, s2)`: a
/// non-zero `a*s1 + b*s2 = 0` must have `|a| >= s2/g` and `|b| >= s1/g`, so a
/// collision exists exactly when both bounds fit inside the extents. The
/// result is exact, so it is decidable without enumeration even for a block
/// far above the enumeration budget. Returns `None` when the block does not
/// hold exactly two axes.
fn two_axis_block_is_injective(
    dims: &[usize],
    strides: &[isize],
    block: &InterleavedBlock,
) -> Option<bool> {
    let mut axes = [(0usize, 0usize); 2]; // (extent, |stride|)
    let mut count = 0;
    for (axis, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
        if !block.contains(axis, dim, stride) {
            continue;
        }
        if count == 2 {
            return None;
        }
        axes[count] = (dim, stride.unsigned_abs());
        count += 1;
    }
    if count != 2 {
        return None;
    }
    // Order by stride so the pairing of the gcd bounds matches the proof.
    let (extent_lo, stride_lo, extent_hi, stride_hi) = if axes[0].1 <= axes[1].1 {
        (
            axes[0].0 as u128,
            axes[0].1 as u128,
            axes[1].0 as u128,
            axes[1].1 as u128,
        )
    } else {
        (
            axes[1].0 as u128,
            axes[1].1 as u128,
            axes[0].0 as u128,
            axes[0].1 as u128,
        )
    };
    let g = gcd(stride_lo, stride_hi);
    debug_assert!(g > 0, "stride zero on a non-singleton axis was rejected");
    Some(stride_hi / g >= extent_lo || stride_lo / g >= extent_hi)
}

/// Exact, allocation-free injectivity decision for a block of three or more
/// axes, by searching for a colliding difference vector.
///
/// Two positions collide iff a non-zero `delta` with `|delta[k]| < dims[k]`
/// solves `sum(delta[k] * strides[k]) == 0`. The two block axes with the
/// largest extents are solved in closed form: for a fixed difference `r` of
/// the other axes, `x * a + y * b == -r` with `|x| <= X`, `|y| <= Y` is a
/// bounded two-variable linear Diophantine equation, settled by its gcd and
/// one modular inverse. Only the other axes are enumerated, and `delta` and
/// `-delta` are the same collision, so half of their box is visited. The
/// search is therefore independent of the two largest extents, which is
/// what lets a large block with small cross axes be decided exactly
/// (tensor4all/strided-rs#342). Returns `None` when the block has fewer than
/// three axes or the search exceeds `DIFFERENCE_SEARCH_BUDGET`.
fn block_differences_avoid_zero(
    dims: &[usize],
    strides: &[isize],
    block: &InterleavedBlock,
) -> Option<bool> {
    let in_block = |axis: usize| block.contains(axis, dims[axis], strides[axis]);
    // The two block axes with the largest extents.
    let (mut first, mut second) = (None::<usize>, None::<usize>);
    let mut count = 0usize;
    for axis in (0..dims.len()).filter(|&axis| in_block(axis)) {
        count += 1;
        if first.is_none_or(|f| dims[axis] > dims[f]) {
            second = first;
            first = Some(axis);
        } else if second.is_none_or(|s| dims[axis] > dims[s]) {
            second = Some(axis);
        }
    }
    if count < 3 {
        return None;
    }
    let (first, second) = (first?, second?);
    let enumerated = |axis: usize| axis != first && axis != second && in_block(axis);

    // The difference box of the enumerated axes holds `boxed` vectors, an
    // odd number with the zero vector in the middle.
    let mut boxed = 1u128;
    for axis in (0..dims.len()).filter(|&axis| enumerated(axis)) {
        boxed = boxed.checked_mul(2 * dims[axis] as u128 - 1)?;
        if boxed / 2 > DIFFERENCE_SEARCH_BUDGET {
            return None;
        }
    }

    // The block span fits `isize` (checked by the caller's input
    // validation), so every partial sum below fits `i128` comfortably.
    let a = strides[first].unsigned_abs() as i128;
    let b = strides[second].unsigned_abs() as i128;
    let (x_bound, y_bound) = ((dims[first] - 1) as i128, (dims[second] - 1) as i128);
    let (g, inverse) = gcd_and_inverse(a, b);
    let (a, b) = (a / g, b / g);
    // `x * a + y * b == c` (after dividing by `g`) has a solution in the box.
    let solvable = |c: i128| -> bool {
        // All solutions: x = x0 + t * b, y = (c - x * a) / b.
        let x0 = (c.rem_euclid(b) * inverse).rem_euclid(b);
        // |y| <= Y  <=>  c - Y b <= x a <= c + Y b.
        let lo = (c - y_bound * b).div_euclid(a) + i128::from((c - y_bound * b).rem_euclid(a) != 0);
        let hi = (c + y_bound * b).div_euclid(a);
        let (lo, hi) = (lo.max(-x_bound), hi.min(x_bound));
        lo <= hi && lo + (x0 - lo).rem_euclid(b) <= hi
    };

    // Collisions with the enumerated axes all zero: the two-axis gcd bound.
    if b <= x_bound && a <= y_bound {
        return Some(false);
    }

    // Walk the upper half of the box in mixed radix, starting just past the
    // zero vector. Each axis is (delta, extent - 1, |stride|) and `r` is the
    // running `sum(delta * stride)`; negating every enumerated stride is a
    // bijection on the box, so `|stride|` is used throughout.
    let enumerated_count = count - 2;
    let fill = |axes: &mut [DifferenceAxis]| {
        for (slot, axis) in axes
            .iter_mut()
            .zip((0..dims.len()).filter(|&axis| enumerated(axis)))
        {
            *slot = (
                0,
                (dims[axis] - 1) as i128,
                strides[axis].unsigned_abs() as i128,
            );
        }
    };
    let collides = |r: i128| r % g == 0 && solvable(-r / g);
    let found = if enumerated_count <= crate::INLINE_RANK {
        let mut axes = [(0i128, 0i128, 0i128); crate::INLINE_RANK];
        let axes = &mut axes[..enumerated_count];
        fill(axes);
        difference_walk_collides(axes, boxed, &collides)
    } else {
        wide_difference_walk_collides(enumerated_count, &fill, boxed, &collides)?
    };
    Some(!found)
}

/// One enumerated axis of the difference walk: `(delta, extent - 1, |stride|)`.
type DifferenceAxis = (i128, i128, i128);

/// Most axes the difference walk can enumerate. Every enumerated axis has an
/// extent above 1, so it multiplies the box by at least 3, and the budget
/// guard runs before the walk: `3^14 / 2 <= 2^22 < 3^15 / 2` admits at most 14.
const WIDE_DIFFERENCE_AXES: usize = 16;

/// [`difference_walk_collides`] for more than [`crate::INLINE_RANK`]
/// enumerated axes, kept out of line so the common walk's frame keeps its size.
/// `None` only for a count the budget guard excludes.
#[inline(never)]
fn wide_difference_walk_collides(
    count: usize,
    fill: &impl Fn(&mut [DifferenceAxis]),
    boxed: u128,
    collides: &impl Fn(i128) -> bool,
) -> Option<bool> {
    // INVARIANT: the budget guard bounds `count` by 14 (see
    // `WIDE_DIFFERENCE_AXES`); a larger count answers undecided.
    if count > WIDE_DIFFERENCE_AXES {
        return None;
    }
    let mut axes = [(0i128, 0i128, 0i128); WIDE_DIFFERENCE_AXES];
    let axes = &mut axes[..count];
    fill(axes);
    Some(difference_walk_collides(axes, boxed, collides))
}

/// Whether the upper half of the difference box, walked in mixed radix from
/// just past the zero vector, holds a difference `r` for which `collides`.
fn difference_walk_collides(
    axes: &mut [DifferenceAxis],
    boxed: u128,
    collides: &impl Fn(i128) -> bool,
) -> bool {
    let mut r = 0i128;
    for _ in 0..boxed / 2 {
        for (delta, bound, stride) in axes.iter_mut() {
            if *delta < *bound {
                *delta += 1;
                r += *stride;
                break;
            }
            r -= *stride * (*delta + *bound);
            *delta = -*bound;
        }
        if collides(r) {
            return true;
        }
    }
    false
}

/// `gcd(a, b)` and the inverse of `a / g` modulo `b / g` (`0` when
/// `b / g == 1`), for positive `a` and `b`.
fn gcd_and_inverse(a: i128, b: i128) -> (i128, i128) {
    // Extended Euclid: invariant `old_r == old_s * a (mod b)`.
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    let modulus = b / old_r;
    (old_r, old_s.rem_euclid(modulus))
}

/// Greatest common divisor of two non-negative integers.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
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
mod tests;
