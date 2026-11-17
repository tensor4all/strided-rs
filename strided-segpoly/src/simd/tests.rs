//! The dispatched kernel's own contract: every mask of every arity, against the
//! scalar statement of `acc[i] = acc[i] + ((c·a_0)·a_1)·a_2`.
//!
//! The mask is the only thing the dispatch selects by value, so each arity runs
//! its whole mask space here; the lane count is 19 so the vector loop and its
//! scalar remainder both run on every supported width.

use super::{
    axpy1_f32, axpy1_f64, axpy2_f32, axpy2_f64, axpy3_f32, axpy3_f64, lanes_f32, lanes_f64,
};

/// Lanes per case: more than the widest vector (16 f32), so the remainder runs.
const LANES: usize = 19;

/// One entry point against the scalar order, over every mask of its arity.
macro_rules! check_arity {
    ($name:ident, $elem:ty, $n:literal, [$($mask:literal),+]) => {{
        let coefficient: $elem = 0.75;
        // Operand values differ per lane and from the splats, so a mask that
        // selected the wrong source changes the result.
        let pool: [Vec<$elem>; 3] = [
            (0..LANES).map(|i| 1.0 + i as $elem).collect(),
            (0..LANES).map(|i| 2.0 - i as $elem * 0.25).collect(),
            (0..LANES).map(|i| 0.5 + i as $elem * 0.125).collect(),
        ];
        let operands: [&[$elem]; $n] = core::array::from_fn(|t| &pool[t][..]);
        let splats: [$elem; $n] = core::array::from_fn(|t| -0.5 + t as $elem * 0.25);
        let start: Vec<$elem> = (0..LANES).map(|i| i as $elem * 0.5).collect();
        for mask in [$($mask),+] {
            let mut acc = start.clone();
            $name(&mut acc, coefficient, operands, splats, mask);
            let mut want = start.clone();
            for (lane, slot) in want.iter_mut().enumerate() {
                let mut value = coefficient;
                for (t, operand) in operands.iter().enumerate() {
                    let source = if mask & (1 << t) == 0 {
                        splats[t]
                    } else {
                        operand[lane]
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
    check_arity!(axpy1_f64, f64, 1, [0, 1]);
    check_arity!(axpy2_f64, f64, 2, [0, 1, 2, 3]);
    check_arity!(axpy3_f64, f64, 3, [0, 1, 2, 3, 4, 5, 6, 7]);
    check_arity!(axpy1_f32, f32, 1, [0, 1]);
    check_arity!(axpy2_f32, f32, 2, [0, 1, 2, 3]);
    check_arity!(axpy3_f32, f32, 3, [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_zero_lane_call_touches_nothing() {
    // `u` comes from a buffer's lane extent, which can be empty; the tail loop is
    // then the whole body and must not touch the empty slices.
    let mut acc: Vec<f64> = Vec::new();
    axpy3_f64(&mut acc, 2.0, [&[], &[], &[]], [1.0, 1.0, 1.0], 0b111);
    axpy2_f32(&mut [], 2.0, [&[], &[]], [1.0, 1.0], 0b11);
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
