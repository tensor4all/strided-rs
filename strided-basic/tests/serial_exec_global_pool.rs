//! Serial work, and low-level calls with a budget above one, must never
//! start the process-global pool. A serial token runs inline; a pooled token
//! uses only its private strided pool.
//!
//! This lives in its own integration-test binary with a single `#[test]` so no
//! other test can initialize the global registry first.

#![cfg(feature = "parallel")]

use std::mem::MaybeUninit;

use strided_basic::execution::parallel_map_reduce;
use strided_basic::{copy_into_uninit, Exec, Pool, StridedArray, StridedViewMut};

const ROWS: usize = 257;
const COLS: usize = 257;
// 257 * 257 = 66049, comfortably above MINTHREADLENGTH (1 << 15).
const _: () = assert!(ROWS * COLS > (1 << 15));

#[test]
fn serial_exec_never_initializes_the_global_pool() {
    let source = StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] * COLS + index[1]) as f64
    });
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut storage = vec![MaybeUninit::<f64>::uninit(); ROWS * COLS];
    let mut dest =
        StridedViewMut::new(&mut storage, &[COLS, ROWS], &[1, COLS as isize], 0).unwrap();

    // Non-contiguous permutation copy on the calling thread.
    copy_into_uninit(&mut dest, &transposed, Exec::serial()).unwrap();

    let sum = parallel_map_reduce(
        0..10_000usize,
        Exec::serial(),
        &|range| range.sum::<usize>(),
        &|a, b| a + b,
    );
    assert_eq!(sum, (0..10_000usize).sum::<usize>());

    // A pooled token with a budget above one, on a private pool: the reduction
    // result is the same, and the process-global pool is still unstarted.
    let pool = Pool::builder().threads(4).build().unwrap();
    let pooled_sum = parallel_map_reduce(
        0..10_000usize,
        Exec::new(&pool, 4).unwrap(),
        &|range| range.sum::<usize>(),
        &|a, b| a + b,
    );
    assert_eq!(pooled_sum, sum);

    for i in 0..COLS {
        for j in 0..ROWS {
            // SAFETY: the successful copy initialized every destination element.
            let got = unsafe { storage[i + j * COLS].assume_init() };
            assert_eq!(got, source.get(&[j, i]), "mismatch at [{i}, {j}]");
        }
    }

    assert!(
        threadteam::ledger::global_pool_unstarted(),
        "the serial token started the process-global pool"
    );
}
