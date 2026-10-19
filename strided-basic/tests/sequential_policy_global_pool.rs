//! A sequential execution policy must never initialize Rayon's global pool.
//!
//! This lives in its own integration-test binary with a single `#[test]` so no
//! other test can initialize the global registry first.

#![cfg(feature = "parallel")]

use std::mem::MaybeUninit;
use strided_basic::execution::parallel_map_reduce;
use strided_basic::{copy_into_uninit, ExecContext, StridedArray, StridedViewMut};

const ROWS: usize = 257;
const COLS: usize = 257;
// 257 * 257 = 66049, comfortably above MINTHREADLENGTH (1 << 15).
const _: () = assert!(ROWS * COLS > (1 << 15));

#[test]
fn sequential_policy_never_initializes_the_global_pool() {
    let source = StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] * COLS + index[1]) as f64
    });
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut storage = vec![MaybeUninit::<f64>::uninit(); ROWS * COLS];
    let mut dest =
        StridedViewMut::new(&mut storage, &[COLS, ROWS], &[1, COLS as isize], 0).unwrap();

    // This mirrors the CPU backend's serial entry: `ExecContext::serial()`.
    let sum = ExecContext::serial().run(|| {
        // Non-contiguous permutation copy: the permutation-engine dispatch must
        // decide from the policy before it probes the ambient Rayon pool.
        copy_into_uninit(&mut dest, &transposed).unwrap();

        // A low-level caller may pass a thread budget above one even under the
        // sequential policy. The splitter must still stay on the calling thread.
        parallel_map_reduce(0..10_000usize, 4, &|range| range.sum::<usize>(), &|a, b| {
            a + b
        })
    });

    assert_eq!(sum, (0..10_000usize).sum::<usize>());
    for i in 0..COLS {
        for j in 0..ROWS {
            // SAFETY: the successful copy initialized every destination element.
            let got = unsafe { storage[i + j * COLS].assume_init() };
            assert_eq!(got, source.get(&[j, i]), "mismatch at [{i}, {j}]");
        }
    }

    assert!(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .is_ok(),
        "the sequential policy initialized Rayon's global pool"
    );
}
