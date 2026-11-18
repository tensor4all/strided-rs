//! The dispatched kernel's own contract: every (arity, mask) the executor can
//! carry, against the scalar statement of `acc[i] = acc[i] + ((c·a_0)·a_1)·a_2`.
//!
//! The mask is the only thing the dispatch selects by value, so each arity runs
//! its whole mask space here; the lane count is 201 so every vector level runs at
//! least one full `CHUNK_VECTORS` chunk and a partial chunk, and has a non-empty
//! scalar remainder. (The scalar level's whole slice is one "vector", so its
//! remainder is the partial chunk and its tail is empty.) A row here holds one
//! operation with one path per segment, which is the per-path statement the
//! replay composes into a row.

use super::{lanes_f32, lanes_f64, op_row_f32, op_row_f64, OpRow};
use crate::plan::MAX_INPUTS;

/// Lanes per case: at the widest vector (16 f32 lanes) 201 lanes are 12 vectors,
/// i.e. one full eight-vector chunk, a four-vector partial chunk and nine scalar
/// tail lanes; every narrower level has its own non-empty partial chunk and
/// tail. A count that is an exact multiple of eight vectors would leave `rest`
/// empty on that level and stop testing the partial-chunk path.
const LANES: usize = 201;

/// One row of one path against the scalar order, over every mask of the arity.
///
/// `splat_mask` is the complement of the `slices` mask the replay passes: bit `t`
/// set means operand `t` is one splatted scalar.
macro_rules! check_arity {
    ($name:ident, $elem:ty, $n:literal, [$($mask:literal),+]) => {{
        let coefficient: $elem = 0.75;
        // Operand lanes differ per lane, and a splat operand is a *single* element
        // at its base, so the two sources carry different values: a mask that
        // selected the wrong source changes the result.
        let pool: [Vec<$elem>; 3] = [
            (0..LANES).map(|i| 1.0 + i as $elem).collect(),
            (0..LANES).map(|i| 2.0 - i as $elem * 0.25).collect(),
            (0..LANES).map(|i| 0.5 + i as $elem * 0.125).collect(),
        ];
        let splats: [$elem; 3] = core::array::from_fn(|t| -0.5 + t as $elem * 0.25);
        let start: Vec<$elem> = (0..LANES).map(|i| i as $elem * 0.5).collect();
        for mask in [$($mask),+] {
            let mut bases = [core::ptr::null::<$elem>(); MAX_INPUTS];
            for (t, base) in bases.iter_mut().enumerate().take($n) {
                *base = if mask & (1 << t) == 0 {
                    core::ptr::addr_of!(splats[t])
                } else {
                    pool[t].as_ptr()
                };
            }
            let offsets = [0isize; $n];
            let coeffs = [coefficient];
            let seg_starts = [0u32];
            let seg_ends = [1u32];
            let mut acc = start.clone();
            let row = OpRow {
                acc: &mut acc,
                u: LANES,
                seg_starts: &seg_starts,
                seg_ends: &seg_ends,
                coeffs: &coeffs,
                offsets: &offsets,
                bases,
                arity: $n,
                splat_mask: !mask & ((1 << $n) - 1),
            };
            // SAFETY: every base points at the operand or the splat above, `u =
            // LANES`, and one path per segment covers the whole row.
            unsafe { $name(pulp::Arch::new(), row) };
            let mut want = start.clone();
            for (lane, slot) in want.iter_mut().enumerate() {
                let mut value = coefficient;
                for t in 0..$n {
                    let source = if mask & (1 << t) == 0 {
                        splats[t]
                    } else {
                        pool[t][lane]
                    };
                    value *= source;
                }
                *slot += value;
            }
            assert_eq!(acc, want, "arity {} mask {mask:#b}", $n);
        }
    }};
}

#[test]
fn every_mask_matches_the_scalar_order() {
    check_arity!(op_row_f64, f64, 1, [0, 1]);
    check_arity!(op_row_f64, f64, 2, [0, 1, 2, 3]);
    check_arity!(op_row_f64, f64, 3, [0, 1, 2, 3, 4, 5, 6, 7]);
    check_arity!(op_row_f32, f32, 1, [0, 1]);
    check_arity!(op_row_f32, f32, 2, [0, 1, 2, 3]);
    check_arity!(op_row_f32, f32, 3, [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_zero_lane_call_touches_nothing() {
    // `u` comes from a buffer's lane extent, which can be empty; the tail loop is
    // then the whole body and must not touch the empty slices.
    let operands: [f64; 3] = [1.0, 2.0, 3.0];
    let coeffs = [2.0f64];
    let seg_starts = [0u32];
    let seg_ends = [1u32];
    let offsets = [0isize; 3];
    let mut bases = [core::ptr::null::<f64>(); MAX_INPUTS];
    for (t, value) in operands.iter().enumerate() {
        bases[t] = value;
    }
    let mut acc: Vec<f64> = Vec::new();
    let row = OpRow {
        acc: &mut acc,
        u: 0,
        seg_starts: &seg_starts,
        seg_ends: &seg_ends,
        coeffs: &coeffs,
        offsets: &offsets,
        bases,
        arity: 3,
        splat_mask: 0b111,
    };
    // SAFETY: the bases point at single initialised values and `u` is 0.
    unsafe { op_row_f64(pulp::Arch::new(), row) };
    let mut empty: Vec<f32> = Vec::new();
    let operands: [f32; 2] = [1.0, 2.0];
    let coeffs = [2.0f32];
    let offsets = [0isize; 2];
    let mut bases = [core::ptr::null::<f32>(); MAX_INPUTS];
    for (t, value) in operands.iter().enumerate() {
        bases[t] = value;
    }
    let row = OpRow {
        acc: &mut empty,
        u: 0,
        seg_starts: &seg_starts,
        seg_ends: &seg_ends,
        coeffs: &coeffs,
        offsets: &offsets,
        bases,
        arity: 2,
        splat_mask: 0b11,
    };
    // SAFETY: as above.
    unsafe { op_row_f32(pulp::Arch::new(), row) };
    assert!(acc.is_empty());
}

#[test]
fn the_reported_widths_are_real() {
    // Whatever level the dispatch picked has to agree with `pulp`'s own constants
    // for that level; both are powers of two, and f32 packs at least as many.
    let (f64_lanes, f32_lanes) = (lanes_f64(), lanes_f32());
    assert!(f64_lanes.is_power_of_two(), "f64 lanes {f64_lanes}");
    assert!(f32_lanes.is_power_of_two(), "f32 lanes {f32_lanes}");
    assert!(f32_lanes >= f64_lanes, "{f32_lanes} < {f64_lanes}");
}

/// A row with several paths per segment, each at its own offset, and payloads
/// that no exact binary arithmetic absorbs.
///
/// The per-segment mask test above pins the *mask*; this one pins the row loop the
/// #498 rewrite moved inside the dispatch: the segment order, the path order within
/// a segment, the per-path offset stride (`arity + 1` across the inputs' entries),
/// and the two roundings of `((c·a_0)·a_1)·a_2` — payloads and coefficients are
/// deliberately non-dyadic, so an FMA contraction or a re-association changes the
/// bits and the comparison catches it.
macro_rules! check_row_order {
    ($name:ident, $elem:ty, $lanes:literal) => {{
        let segments = 2usize;
        let paths = 5usize;
        let arity = 3usize;
        let slots = [[1usize, 0, 2], [0, 1, 1], [2, 2, 0], [1, 2, 0], [0, 0, 2]];
        let seg_starts: [u32; 2] = [0, 3];
        let seg_ends: [u32; 2] = [3, 5];
        let coefficients: [f64; 5] = [0.1, 0.3, -0.7, 1.0 / 3.0, 2.5e-3];
        // Three slots per operand, `$lanes` lanes each, from an irrational-ish
        // generator so no product or sum is exact.
        let payload = |seed: f64| -> Vec<$elem> {
            (0..3 * $lanes)
                .map(|index| ((index as f64 + 1.0).sqrt() * seed + index as f64 * 0.1) as $elem)
                .collect()
        };
        let operands: [Vec<$elem>; 3] = [payload(0.37), payload(-0.61), payload(1.13)];
        // Per path: the three input slots, then the output's own segment offset --
        // the slot the scalar replay shares and the kernel skips.
        let mut offsets = Vec::with_capacity(paths * (arity + 1));
        for path in 0..paths {
            for t in 0..arity {
                offsets.push(slots[path][t] as isize * $lanes as isize);
            }
            offsets.push(path as isize);
        }
        let coefficients: Vec<$elem> = coefficients.iter().map(|&c| c as $elem).collect();
        let mut bases = [core::ptr::null::<$elem>(); MAX_INPUTS];
        for (t, base) in bases.iter_mut().enumerate().take(arity) {
            *base = operands[t].as_ptr();
        }
        let mut acc = vec![0 as $elem; segments * $lanes];
        let row = OpRow {
            acc: &mut acc,
            u: $lanes,
            seg_starts: &seg_starts,
            seg_ends: &seg_ends,
            coeffs: &coefficients,
            offsets: &offsets,
            bases,
            arity,
            splat_mask: 0,
        };
        // SAFETY: every base reaches three `$lanes` slots, the offsets stay inside
        // them, and `u` is the lane extent.
        unsafe { $name(pulp::Arch::new(), row) };
        let mut want = vec![0 as $elem; segments * $lanes];
        for segment in 0..segments {
            for path in seg_starts[segment] as usize..seg_ends[segment] as usize {
                for lane in 0..$lanes {
                    let mut value = coefficients[path];
                    for t in 0..arity {
                        value *= operands[t][slots[path][t] * $lanes + lane];
                    }
                    want[segment * $lanes + lane] += value;
                }
            }
        }
        for (index, (actual, expected)) in acc.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "element {index}: {actual:?} is not {expected:?}"
            );
        }
    }};
}

#[test]
fn a_row_visits_every_segment_and_path_at_its_own_offset() {
    check_row_order!(op_row_f64, f64, 201);
    check_row_order!(op_row_f32, f32, 201);
}
