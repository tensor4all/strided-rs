//! `map_into_uninit_with_kernel`: the vectorized-kernel entry point.
//!
//! The contract under test is that the kernel is offered a flat run exactly when one exists — the
//! whole buffer for a sequential contiguous pair, one block at a time on the parallel path — that
//! declining a run leaves the destination untouched so the scalar path can write it, and that the
//! layout checks still reject shapes and destinations the other map entry points reject.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use strided_basic::{
    map_into, map_into_uninit_with_kernel, Identity, StridedError, StridedView, StridedViewMut,
};

/// A value the operation cannot produce, so that a slot the layout leaves unreachable stays
/// visibly untouched instead of being read as uninitialized memory.
const SENTINEL: f64 = -999.0;

/// Runs the entry point with a kernel that counts its calls, doubles every element and either
/// accepts or declines, and returns the counted calls plus the resulting values.
fn run(
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    extent: usize,
    accept: bool,
) -> (usize, Vec<f64>) {
    let source: Vec<f64> = (0..extent).map(|i| i as f64 + 1.0).collect();
    let mut out = vec![MaybeUninit::new(SENTINEL); extent];
    let calls = AtomicUsize::new(0);
    {
        let mut dest = StridedViewMut::new(&mut out, dims, strides, offset).unwrap();
        let view = StridedView::new(&source, dims, strides, offset).unwrap();
        map_into_uninit_with_kernel(
            &mut dest,
            &view,
            |src: &[f64], dst: &mut [MaybeUninit<f64>]| {
                calls.fetch_add(1, Ordering::Relaxed);
                if !accept {
                    return false;
                }
                for (slot, value) in dst.iter_mut().zip(src) {
                    slot.write(value * 2.0);
                }
                true
            },
            |x| x * 2.0,
        )
        .unwrap();
    }
    // SAFETY: every slot is either written by the kernel or the scalar fallback, or still holds
    // the sentinel it was initialized with.
    let values = out
        .into_iter()
        .map(|value| unsafe { value.assume_init() })
        .collect();
    (calls.load(Ordering::Relaxed), values)
}

/// The scalar reference for the same layout, so the two entry points are compared on equal terms.
fn reference(dims: &[usize], strides: &[isize], offset: isize, extent: usize) -> Vec<f64> {
    let source: Vec<f64> = (0..extent).map(|i| i as f64 + 1.0).collect();
    let mut out = vec![SENTINEL; extent];
    {
        let mut dest = StridedViewMut::new(&mut out, dims, strides, offset).unwrap();
        let view = StridedView::<f64, Identity>::new(&source, dims, strides, offset).unwrap();
        map_into(&mut dest, &view, |x| x * 2.0).unwrap();
    }
    out
}

#[test]
fn a_sequential_contiguous_pair_offers_the_whole_buffer_once() {
    let (calls, values) = run(&[4], &[1], 0, 4, true);
    assert_eq!(calls, 1);
    assert_eq!(values, reference(&[4], &[1], 0, 4));
    assert_eq!(values, [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn a_contiguous_pair_of_any_rank_offers_one_run() {
    // The predicate accepts a shared row-major layout too, not only column-major; both sides agree
    // on the order, so the flat pair is still element for element.
    let (calls, values) = run(&[2, 3], &[3, 1], 0, 6, true);
    assert_eq!(calls, 1);
    assert_eq!(values, reference(&[2, 3], &[3, 1], 0, 6));

    // An offset does not break the contiguous run.
    let (calls, values) = run(&[3], &[1], 2, 8, true);
    assert_eq!(calls, 1);
    assert_eq!(values, reference(&[3], &[1], 2, 8));
}

#[test]
fn a_stride_that_is_not_one_never_reaches_the_kernel() {
    let (calls, values) = run(&[2], &[2], 0, 4, true);
    assert_eq!(calls, 0, "a strided run has no flat slice to offer");
    assert_eq!(values, reference(&[2], &[2], 0, 4));
}

#[test]
fn declining_a_run_leaves_the_scalar_path_to_write_every_element() {
    let (calls, values) = run(&[4], &[1], 0, 4, false);
    assert_eq!(calls, 1, "the kernel was offered the run and declined it");
    assert_eq!(values, [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn the_shape_and_layout_checks_still_apply() {
    let source = vec![1.0f64, 2.0];
    let mut out = vec![MaybeUninit::<f64>::uninit(); 4];
    let mut dest = StridedViewMut::new(&mut out, &[4], &[1], 0).unwrap();
    let view = StridedView::new(&source, &[2], &[1], 0).unwrap();
    let kernel = |_src: &[f64], _dst: &mut [MaybeUninit<f64>]| true;
    // The kernel accepts, but a mismatched shape must be rejected before it is consulted.
    assert!(matches!(
        map_into_uninit_with_kernel(&mut dest, &view, kernel, |x| x),
        Err(StridedError::ShapeMismatch { .. })
    ));

    // A destination layout that cannot be proven injective is rejected too.
    let source = vec![1.0f64, 2.0];
    let mut out = vec![MaybeUninit::<f64>::uninit(); 4];
    let mut dest = StridedViewMut::new(&mut out, &[2, 2], &[1, 0], 0).unwrap();
    let view = StridedView::new(&source, &[2, 2], &[1, 0], 0).unwrap();
    assert!(matches!(
        map_into_uninit_with_kernel(&mut dest, &view, kernel, |x| x),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
}

#[cfg(feature = "parallel")]
#[test]
fn the_parallel_path_offers_each_block_a_run() {
    use std::num::NonZeroUsize;

    use strided_basic::{with_execution_policy, ExecutionPolicy};

    let extent = 1usize << 20;
    let source: Vec<f64> = (0..extent).map(|i| i as f64 + 1.0).collect();
    let mut out = vec![MaybeUninit::new(SENTINEL); extent];
    let calls = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        with_execution_policy(
            ExecutionPolicy::Rayon {
                max_threads: NonZeroUsize::new(4).unwrap(),
            },
            || {
                let mut dest = StridedViewMut::new(&mut out, &[extent], &[1], 0).unwrap();
                let view = StridedView::new(&source, &[extent], &[1], 0).unwrap();
                map_into_uninit_with_kernel(
                    &mut dest,
                    &view,
                    |src: &[f64], dst: &mut [MaybeUninit<f64>]| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        for (slot, value) in dst.iter_mut().zip(src) {
                            slot.write(value * 2.0);
                        }
                        true
                    },
                    |x| x * 2.0,
                )
                .unwrap();
            },
        );
    });

    assert!(
        calls.load(Ordering::Relaxed) > 1,
        "the parallel path splits the buffer into blocks"
    );
    for (index, value) in out.into_iter().enumerate() {
        // SAFETY: every block was written by the kernel.
        let value = unsafe { value.assume_init() };
        assert_eq!(value, (index as f64 + 1.0) * 2.0);
    }
}
