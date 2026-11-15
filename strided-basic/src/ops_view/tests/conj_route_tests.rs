//! Route assertions for `copy_conj_into` (tensor4all/strided-rs#467).
//!
//! The elements `copy_conj_into` writes are identical on both routes, so a value
//! check cannot see which one ran. These tests assert the decision itself:
//! `conj_perm_engine_route` for the clause boundaries, and
//! `threading::take_conj_permuted` for the route `copy_conj_into` actually took.

use super::*;
use crate::threading::take_conj_permuted;
use crate::view::StridedArray;
use num_complex::{Complex32, Complex64};

/// A `[rows, cols]` complex source permuted to `[cols, rows]`, i.e. the 2D
/// transposing copy the issue measures.
fn transposing_pair(
    rows: usize,
    cols: usize,
) -> (StridedArray<Complex64>, StridedArray<Complex64>) {
    let src = StridedArray::<Complex64>::from_fn_col_major(&[rows, cols], |idx| {
        Complex64::new(idx[0] as f64, idx[1] as f64)
    });
    let dst = StridedArray::<Complex64>::col_major(&[cols, rows]);
    (src, dst)
}

#[test]
fn route_takes_a_small_strided_complex_permutation() {
    // [5,3] permuted [1,0]: destination strides [1,5], source strides [3,1].
    assert!(conj_perm_engine_route::<Complex64>(
        &[3, 5],
        &[1, 3],
        &[5, 1]
    ));
    assert!(conj_perm_engine_route::<Complex32>(
        &[3, 5],
        &[1, 3],
        &[5, 1]
    ));
}

#[test]
fn route_keeps_contiguous_copies_on_the_map() {
    // Equal contiguous layouts: the map's SIMD store loop, not the engine.
    assert!(!conj_perm_engine_route::<Complex64>(
        &[4, 4],
        &[1, 4],
        &[1, 4]
    ));
}

#[test]
fn route_takes_long_row_2d_copies_on_the_engine() {
    // The rows the removed 4 KiB destination-row clause kept on the map: #480
    // fixed the engine's L1 aliasing, and every one of them is under the 24 MiB
    // footprint bound, so the engine wins them (2.2x at 256^2/512^2 on the base
    // that removed the clause).
    for side in [255usize, 256, 257, 512, 1023, 1024] {
        assert!(
            conj_perm_engine_route::<Complex64>(
                &[side, side],
                &[1, side as isize],
                &[side as isize, 1]
            ),
            "{side}^2"
        );
    }
    // The clause is gone, not the footprint bound: 2048^2 is 64 MiB per operand
    // and still keeps the map.
    assert!(!conj_perm_engine_route::<Complex64>(
        &[2048, 2048],
        &[1, 2048],
        &[2048, 1]
    ));
    // Other 3D/4D layouts are decided by the same two clauses.
    assert!(conj_perm_engine_route::<Complex64>(
        &[64, 64, 64],
        &[1, 64, 4096],
        &[64, 1, 64]
    ));
    assert!(conj_perm_engine_route::<Complex64>(
        &[32, 32, 32, 32],
        &[1, 32, 1024, 32768],
        &[32, 1024, 1, 32768]
    ));
}

#[test]
fn route_keeps_an_operand_at_the_cache_bound_on_the_map() {
    // 2_097_152 Complex64 is 32 MiB, at the measured crossing; 1_404_928
    // (112^3, 22.4 MiB) is below it. Rank 3, so only this bound decides.
    assert!(conj_perm_engine_route::<Complex64>(
        &[112, 112, 112],
        &[1, 112, 12544],
        &[12544, 1, 112]
    ));
    // 116^3 is 23.8 MiB, the last row under the bound: still the engine.
    assert!(conj_perm_engine_route::<Complex64>(
        &[116, 116, 116],
        &[1, 116, 13456],
        &[13456, 1, 116]
    ));
    // 120^3 is 26.4 MiB, the first row over it.
    assert!(!conj_perm_engine_route::<Complex64>(
        &[120, 120, 120],
        &[1, 120, 14400],
        &[14400, 1, 120]
    ));
    assert!(!conj_perm_engine_route::<Complex64>(
        &[128, 128, 128],
        &[1, 128, 16384],
        &[16384, 1, 128]
    ));
}

#[test]
fn route_keeps_real_element_types_on_the_map() {
    // A real type's `conj` is the identity; the engine's conjugating store would
    // only be the generic element store.
    assert!(!conj_perm_engine_route::<f64>(&[3, 5], &[1, 3], &[5, 1]));
    assert!(!conj_perm_engine_route::<f32>(&[3, 5], &[1, 3], &[5, 1]));
}

#[test]
fn route_leaves_an_overflowing_byte_count_to_the_map_path() {
    // A single axis of 2^61 complex elements overflows the byte count, not the
    // element count.
    assert!(!conj_perm_engine_route::<Complex64>(&[1 << 61], &[2], &[1]));
}

#[test]
fn route_leaves_an_overflowing_extent_to_the_map_path() {
    // Huge broadcast: total_len overflows, and the map path owns that error.
    assert!(!conj_perm_engine_route::<Complex64>(
        &[usize::MAX, usize::MAX],
        &[0, 0],
        &[0, 0]
    ));
}

#[test]
fn copy_conj_into_takes_the_engine_route() {
    let (src, mut dst) = transposing_pair(5, 3);
    let _ = take_conj_permuted();

    let src_t = src.view().permute(&[1, 0]).unwrap();
    copy_conj_into(&mut dst.view_mut(), &src_t, Exec::serial()).unwrap();

    // The decision, not the bits: deleting the engine branch leaves the values
    // correct and this count at zero.
    assert_eq!(take_conj_permuted(), 1);
    for i in 0..5 {
        for j in 0..3 {
            assert_eq!(dst.get(&[j, i]), src.get(&[i, j]).conj());
        }
    }
}

#[test]
fn copy_conj_into_keeps_the_map_route() {
    // Contiguous, over the footprint bound, and real: the counter must stay at
    // zero on each, so the gate is not "always the engine".
    let mut buf = StridedArray::<Complex64>::col_major(&[4, 4]);
    let src = StridedArray::<Complex64>::from_fn_col_major(&[4, 4], |idx| {
        Complex64::new(idx[0] as f64, idx[1] as f64)
    });
    let _ = take_conj_permuted();
    copy_conj_into(&mut buf.view_mut(), &src.view(), Exec::serial()).unwrap();
    assert_eq!(take_conj_permuted(), 0);

    // 1500^2 Complex64 is 1500^2 * 16 B = 36 MB (34 MiB) per operand: over
    // `CONJ_PERM_ENGINE_MAX_BYTES`.
    let (src, mut dst) = transposing_pair(1500, 1500);
    let src_t = src.view().permute(&[1, 0]).unwrap();
    copy_conj_into(&mut dst.view_mut(), &src_t, Exec::serial()).unwrap();
    assert_eq!(take_conj_permuted(), 0);

    let src = StridedArray::<f64>::from_fn_col_major(&[5, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut dst = StridedArray::<f64>::col_major(&[3, 5]);
    let src_t = src.view().permute(&[1, 0]).unwrap();
    copy_conj_into(&mut dst.view_mut(), &src_t, Exec::serial()).unwrap();
    assert_eq!(take_conj_permuted(), 0);
}
