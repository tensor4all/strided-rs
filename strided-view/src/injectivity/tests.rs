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
fn three_axis_interleaved_injective_layout_is_decided_without_alloc() {
    // Three-axis interleaved block above the pairwise budget; no two-axis
    // proof applies, but the difference search settles it without
    // enumerating the block.
    assert!(is_injective_layout_without_alloc(&[2, 2, 2000], &[4, 5, 6]));
    assert!(is_injective_layout(&[2, 2, 2000], &[4, 5, 6]));
    // The two-axis gcd proof now settles the old 4500-element case
    // allocation-free.
    assert!(is_injective_layout_without_alloc(&[3, 1500], &[2000, 3]));
    assert!(is_injective_layout(&[1500, 3], &[3, 2000]));
    assert!(!is_injective_layout(&[3, 1500], &[2000, 2]));
}

#[test]
fn large_two_axis_injective_layout_is_decided_without_enumeration() {
    // Post-review regression: 6_000_001 = 1 (mod 3), so the rows stay
    // distinct. The block has 18M elements, above the exact enumeration
    // budget, and must be settled by the allocation-free gcd proof instead
    // of being rejected as overlapping.
    assert!(is_injective_layout(&[3, 6_000_000], &[6_000_001, 3]));
    assert!(is_injective_layout_without_alloc(
        &[3, 6_000_000],
        &[6_000_001, 3]
    ));
    // The gcd proof is exact: with a stride-3 axis of extent 6M and a
    // 6_000_003 axis of extent 3, `2000001 * 3 == 6000003`, so the layout
    // aliases and is rejected without enumeration.
    assert!(!is_injective_layout(&[3, 6_000_000], &[6_000_003, 3]));
    assert!(!is_injective_layout_without_alloc(
        &[3, 6_000_000],
        &[6_000_003, 3]
    ));
}

#[test]
fn large_three_axis_injective_layout_is_decided_without_enumeration() {
    // tensor4all/strided-rs#342: 20M elements, above the exact
    // enumeration budget, three interleaved axes. 6c = 4a + 5b has no
    // non-zero solution with |a|, |b| <= 1, so the layout is injective.
    assert!(is_injective_layout(&[2, 2, 5_000_000], &[4, 5, 6]));
    assert!(is_injective_layout_without_alloc(
        &[2, 2, 5_000_000],
        &[4, 5, 6]
    ));
    // Same size, aliasing: 4 + 6 == 2 * 5.
    assert!(!is_injective_layout(&[2, 2, 5_000_000], &[4, 6, 5]));
}

#[test]
fn difference_search_matches_enumeration_on_small_layouts() {
    // Every three- and four-axis layout with small extents and strides is
    // decided by the difference search and by direct enumeration; the two
    // must agree, including on negative strides.
    let mut checked = 0;
    for rank in 3..=4usize {
        let cases = 3usize.pow(rank as u32) * 5usize.pow(rank as u32);
        for case in 0..cases {
            let mut code = case;
            let mut dims = [0usize; 4];
            let mut strides = [0isize; 4];
            for axis in 0..rank {
                dims[axis] = 2 + code % 3;
                code /= 3;
                let stride = 1 + (code % 5) as isize;
                code /= 5;
                strides[axis] = if axis == 1 { -stride } else { stride };
            }
            let (dims, strides) = (&dims[..rank], &strides[..rank]);
            let Some(Some(block)) = interleaved_block(dims, strides) else {
                continue;
            };
            let Some(fast) = block_differences_avoid_zero(dims, strides, &block) else {
                continue;
            };
            assert_eq!(
                fast,
                block_offsets_unique(dims, strides, &block),
                "{dims:?} {strides:?}"
            );
            checked += 1;
        }
    }
    assert!(checked > 1000);
}

#[test]
fn checked_total_len_detects_overflow() {
    assert_eq!(total_len(&[usize::MAX, 2]), None);
    assert_eq!(total_len(&[3, 4]), Some(12));
    // A zero extent wins before the overflowing product.
    assert_eq!(total_len(&[usize::MAX, 2, 0]), Some(0));
}
