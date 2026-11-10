//! The pointwise planner must not overflow its importance weights at high rank
//! (tensor4all/strided-rs#450).
//!
//! `compute_importance` packs one level per axis, `g = ceil(log2(m + 2))` bits
//! apart, so the four planner operands of `zip_map3_into` (destination plus
//! three inputs, `g = 3`) need 98 bits at rank 32 where the score holds 64: the
//! highest term is `2^97`, a 93-bit level shift times the `2^4` output weight. The
//! case below is the reported layout: 2048 elements over 32 axes with a gapped
//! inner stride, so it takes the fused plan (above `SMALL_TENSOR_THRESHOLD`)
//! rather than the contiguous fast path.

use strided_basic::{zip_map3_into, Exec, Identity, StridedView, StridedViewMut};

#[test]
fn test_zip_map3_high_rank_plan_does_not_overflow() {
    const RANK: usize = 32;
    const LEN: usize = 2048;
    const BUF: usize = 2 * LEN;

    let mut dims = [1usize; RANK];
    dims[0] = LEN;
    // Gapped inner stride: a contiguous layout would take the sequential copy
    // path instead of the fused plan this test is about.
    let mut strides = [LEN as isize; RANK];
    strides[0] = 2;

    let a: Vec<f64> = (0..BUF).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..BUF).map(|i| 2.0 * i as f64).collect();
    let c: Vec<f64> = (0..BUF).map(|i| 3.0 * i as f64).collect();
    let mut out = vec![f64::NAN; BUF];

    let view_a = StridedView::<f64, Identity>::new(&a, &dims, &strides, 0).unwrap();
    let view_b = StridedView::<f64, Identity>::new(&b, &dims, &strides, 0).unwrap();
    let view_c = StridedView::<f64, Identity>::new(&c, &dims, &strides, 0).unwrap();
    let mut view_out = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();

    zip_map3_into(
        &mut view_out,
        &view_a,
        &view_b,
        &view_c,
        |x, y, z| x + y + z,
        Exec::serial(),
    )
    .unwrap();

    for i in 0..LEN {
        assert_eq!(out[2 * i], 12.0 * i as f64, "element {i}");
    }
}
