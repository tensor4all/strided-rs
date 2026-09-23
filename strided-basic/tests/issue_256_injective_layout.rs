//! Regression tests for issues #255 / #256: `is_injective_layout` must not
//! flip from "injective" to "overlapping" just because a valid layout exceeds
//! 4096 elements.

use std::collections::HashSet;
use strided_basic::execution::{is_injective_layout, validate_destination_layout_without_alloc};
use strided_basic::{map_into, StridedArray, StridedError, StridedViewMut};

/// Brute-force oracle over every logical coordinate.
fn oracle(dims: &[usize], strides: &[isize]) -> bool {
    let total: usize = dims.iter().product();
    let mut seen = HashSet::with_capacity(total);
    for mut linear in 0..total {
        let mut offset = 0isize;
        for (&dim, &stride) in dims.iter().zip(strides) {
            offset += (linear % dim) as isize * stride;
            linear /= dim;
        }
        if !seen.insert(offset) {
            return false;
        }
    }
    true
}

#[test]
fn interleaved_injective_layout_above_4096_is_accepted() {
    // Issue #256 reproducer: 2000 = 2 (mod 3), so the three rows occupy
    // distinct residues and all 4500 offsets are distinct.
    assert!(oracle(&[3, 1500], &[2000, 3]));
    assert!(is_injective_layout(&[3, 1500], &[2000, 3]));
    // The same family below the old 4096 cap was already accepted.
    assert!(is_injective_layout(&[3, 1365], &[2000, 3]));
    // Negative strides reflect coordinates and keep injectivity.
    assert!(is_injective_layout(&[3, 1500], &[-2000, 3]));
    assert!(is_injective_layout(&[3, 1500], &[2000, -3]));
}

#[test]
fn interleaved_overlapping_layout_above_4096_is_rejected() {
    // 3000 = 0 (mod 3): offset 3000 is hit by (1, 0) and (0, 1000).
    assert!(!oracle(&[3, 1500], &[3000, 3]));
    assert!(!is_injective_layout(&[3, 1500], &[3000, 3]));
    // Pigeonhole: 2 * 5000 elements over a 5000-offset span.
    assert!(!is_injective_layout(&[2, 5000], &[1, 1]));
}

#[test]
fn separated_outer_axes_do_not_count_toward_the_exact_block() {
    // The interleaved block is [3, 1500]; the outer axis is separated.
    let dims = [3, 1500, 4];
    assert!(is_injective_layout(&dims, &[2000, 3, 1 << 20]));
    assert!(!is_injective_layout(&dims, &[3000, 3, 1 << 20]));
    // A separated inner axis below an interleaved pair: offsets
    // i + 2 * (j * 2000 + 3 * k) stay distinct.
    assert!(is_injective_layout(&[2, 3, 1500], &[1, 4000, 6]));
}

#[test]
fn randomized_layouts_match_the_oracle() {
    // Deterministic LCG over small interleaved layouts, both signs.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = |bound: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    for _ in 0..2000 {
        let rank = 1 + next(4) as usize;
        let dims: Vec<usize> = (0..rank).map(|_| 1 + next(7) as usize).collect();
        let strides: Vec<isize> = (0..rank)
            .map(|_| {
                let magnitude = 1 + next(24) as isize;
                if next(2) == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();
        let expected = oracle(&dims, &strides);
        assert_eq!(
            is_injective_layout(&dims, &strides),
            expected,
            "dims={dims:?} strides={strides:?}"
        );
        // The allocation-free variant is exact on these small blocks too.
        assert_eq!(
            validate_destination_layout_without_alloc(&dims, &strides).is_ok(),
            expected,
            "without_alloc dims={dims:?} strides={strides:?}"
        );
    }
}

#[test]
fn without_alloc_variant_accepts_separated_large_layouts() {
    // 3 * 1000 interleaved elements plus a separated outer axis: 12000
    // logical elements, only 3000 of them in the pairwise block.
    assert!(validate_destination_layout_without_alloc(&[3, 1000, 4], &[2000, 3, 1 << 20]).is_ok());
    // 2997 = 3 * 999, so (1, 0, 0) and (0, 999, 0) alias.
    assert!(!oracle(&[3, 1000, 4], &[2997, 3, 1 << 20]));
    assert!(validate_destination_layout_without_alloc(&[3, 1000, 4], &[2997, 3, 1 << 20]).is_err());
}

#[test]
fn map_into_accepts_interleaved_injective_destination() {
    let dims = [3usize, 1500];
    let strides = [2000isize, 3];
    let src =
        StridedArray::<f64>::from_fn_col_major(&dims, |idx| (idx[0] * 10_000 + idx[1]) as f64);
    let mut buffer = vec![-1.0f64; 8498];
    {
        let mut dest = StridedViewMut::new(&mut buffer, &dims, &strides, 0).unwrap();
        let result: Result<(), StridedError> = map_into(&mut dest, &src.view(), |x| x);
        result.unwrap();
    }
    for i in 0..3 {
        for j in 0..1500 {
            assert_eq!(buffer[i * 2000 + 3 * j], (i * 10_000 + j) as f64);
        }
    }
}
