//! Nothing run on a private strided pool may start the process-global pool.
//!
//! This lives in its own integration-test binary with a single `#[test]` so no
//! other test can start the process-global pool first. The final check reads
//! `threadteam::ledger::global_pool_unstarted`.

#![cfg(feature = "parallel")]

use std::mem::MaybeUninit;

use strided_basic::{copy_into_uninit, map_into, Exec, Pool, StridedArray, StridedViewMut};

const ROWS: usize = 257;
const COLS: usize = 257;
const _: () = assert!(ROWS * COLS > (1 << 15));

fn large_map(exec: Exec<'_>) {
    let source = StridedArray::<f64>::from_fn_col_major(&[ROWS * COLS], |index| index[0] as f64);
    let mut destination = StridedArray::<f64>::col_major(&[ROWS * COLS]);
    map_into(
        &mut destination.view_mut(),
        &source.view(),
        |value| value + 1.0,
        exec,
    )
    .unwrap();
    assert_eq!(destination.get(&[ROWS * COLS - 1]), (ROWS * COLS) as f64);
}

#[test]
fn pooled_work_never_initializes_the_global_pool() {
    let pool = Pool::builder().threads(4).build().unwrap();

    let source = StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] * COLS + index[1]) as f64
    });
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut storage = vec![MaybeUninit::<f64>::uninit(); ROWS * COLS];
    let mut dest =
        StridedViewMut::new(&mut storage, &[COLS, ROWS], &[1, COLS as isize], 0).unwrap();

    // Small op: runs inline on the caller, never installs.
    let small_source = StridedArray::<f64>::from_fn_col_major(&[4], |index| index[0] as f64);
    let mut small_destination = StridedArray::<f64>::col_major(&[4]);
    map_into(
        &mut small_destination.view_mut(),
        &small_source.view(),
        |value| value + 1.0,
        Exec::new(&pool, 4).unwrap(),
    )
    .unwrap();

    // Large elementwise op: fans out on the private pool.
    large_map(Exec::new(&pool, 4).unwrap());

    // Non-contiguous native-float copy: the permutation engine path, also run on
    // the private pool.
    copy_into_uninit(&mut dest, &transposed, Exec::new(&pool, 4).unwrap()).unwrap();

    for i in 0..COLS {
        for j in 0..ROWS {
            // SAFETY: the successful copy initialized every destination element.
            let got = unsafe { storage[i + j * COLS].assume_init() };
            assert_eq!(got, source.get(&[j, i]), "mismatch at [{i}, {j}]");
        }
    }

    assert!(
        threadteam::ledger::global_pool_unstarted(),
        "the private strided pool started the process-global pool"
    );
}
