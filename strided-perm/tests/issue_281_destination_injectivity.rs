//! Regression tests for tensor4all/strided-rs#281.
//!
//! `copy_into` / `copy_into_par` must reject a non-injective destination
//! layout (overlapping writes, and an unsynchronized data race under
//! `parallel`) before entering their unsafe write loops, while every layout
//! that is in fact injective stays accepted. The interleaved `[3, 1500]` /
//! `[2000, 3]` case is accepted only by the allocating exact spill path of the
//! validator.

#[cfg(feature = "parallel")]
use strided_perm::copy_into_par;
use strided_perm::{copy_into, copy_into_col_major};
use strided_view::{StridedError, StridedView, StridedViewMut};

/// The interleaved-but-injective case from the issue: `copy_into` must accept
/// it (the allocation-free validator's pairwise budget is 4096 elements, so it
/// hands the 4500-element block to the exact spill path).
#[test]
fn interleaved_injective_destination_is_accepted() {
    let dims = [3usize, 1500];
    let src_data: Vec<f64> = (0..4500).map(|value| value as f64).collect();
    // Max reachable dest offset: 2 * 2000 + 1499 * 3 = 8497.
    let mut dst_data = vec![0.0f64; 8498];
    let src = StridedView::new(&src_data, &dims, &[1, 3], 0).unwrap();
    let mut dst = StridedViewMut::new(&mut dst_data, &dims, &[2000, 3], 0).unwrap();

    copy_into(&mut dst, &src).unwrap();
    for i in 0..3 {
        for j in 0..1500 {
            assert_eq!(dst_data[i * 2000 + j * 3], src_data[i + j * 3]);
        }
    }
}

/// A destination whose logical positions collide must be rejected, serial and
/// parallel, and the destination must be left untouched.
#[test]
fn overlapping_destination_is_rejected() {
    // (i, j) -> 256 i + 256 j collides: max offset 130560 < 131072, so the
    // safe view constructor accepts it.
    let dims = [256usize, 256];
    let src_data = vec![1.0f64; 65536];
    let mut dst_data = vec![0.0f64; 131072];
    let src = StridedView::new(&src_data, &dims, &[1, 256], 0).unwrap();
    let mut dst = StridedViewMut::new(&mut dst_data, &dims, &[256, 256], 0).unwrap();

    assert!(matches!(
        copy_into(&mut dst, &src),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
    assert!(matches!(
        copy_into_col_major(&mut dst, &src),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
    drop(dst);
    assert!(dst_data.iter().all(|&value| value == 0.0));
}

/// The overlapping destination stays rejected under the parallel entry, where
/// the collision would be an unsynchronized write.
#[cfg(feature = "parallel")]
#[test]
fn overlapping_destination_is_rejected_parallel() {
    let dims = [256usize, 256];
    let src_data = vec![1.0f64; 65536];
    let mut dst_data = vec![0.0f64; 131072];
    let src = StridedView::new(&src_data, &dims, &[1, 256], 0).unwrap();
    let mut dst = StridedViewMut::new(&mut dst_data, &dims, &[256, 256], 0).unwrap();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let result = pool.install(|| copy_into_par(&mut dst, &src));
    assert!(matches!(
        result,
        Err(StridedError::NonInjectiveOutputLayout)
    ));
    drop(dst);
    assert!(dst_data.iter().all(|&value| value == 0.0));
}

/// A stride-0 non-singleton axis also overlaps; it must be rejected like the
/// interleaved case.
#[test]
fn collapsed_destination_axis_is_rejected() {
    let dims = [3usize, 3];
    let src_data = vec![1.0f64; 9];
    let mut dst_data = vec![0.0f64; 4];
    let src = StridedView::new(&src_data, &dims, &[1, 3], 0).unwrap();
    let mut dst = StridedViewMut::new(&mut dst_data, &dims, &[1, 0], 0).unwrap();

    assert!(matches!(
        copy_into(&mut dst, &src),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
}
