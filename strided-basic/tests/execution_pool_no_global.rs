//! Nothing inside a target-pool scope may initialize Rayon's global pool.
//!
//! This lives in its own integration-test binary with a single `#[test]` so no
//! other test can initialize the global registry first.

#![cfg(feature = "parallel")]

use std::mem::MaybeUninit;
use std::num::NonZeroUsize;

use strided_basic::{
    copy_into_uninit, map_into, with_execution_pool, StridedArray, StridedViewMut,
};

const ROWS: usize = 257;
const COLS: usize = 257;
const _: () = assert!(ROWS * COLS > (1 << 15));

fn large_map() {
    let source = StridedArray::<f64>::from_fn_col_major(&[ROWS * COLS], |index| index[0] as f64);
    let mut destination = StridedArray::<f64>::col_major(&[ROWS * COLS]);
    map_into(&mut destination.view_mut(), &source.view(), |value| {
        value + 1.0
    })
    .unwrap();
    assert_eq!(destination.get(&[ROWS * COLS - 1]), (ROWS * COLS) as f64);
}

#[test]
fn target_pool_scope_never_initializes_the_global_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let source = StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] * COLS + index[1]) as f64
    });
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut storage = vec![MaybeUninit::<f64>::uninit(); ROWS * COLS];
    let mut dest =
        StridedViewMut::new(&mut storage, &[COLS, ROWS], &[1, COLS as isize], 0).unwrap();

    with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
        // Small op: no install.
        let small_source = StridedArray::<f64>::from_fn_col_major(&[4], |index| index[0] as f64);
        let mut small_destination = StridedArray::<f64>::col_major(&[4]);
        map_into(
            &mut small_destination.view_mut(),
            &small_source.view(),
            |value| value + 1.0,
        )
        .unwrap();

        // Large elementwise op: installs the target pool.
        large_map();

        // Non-contiguous native-float copy: the permutation engine path, also
        // installed into the target pool rather than the ambient one.
        copy_into_uninit(&mut dest, &transposed).unwrap();
    });

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
        "the target-pool scope initialized Rayon's global pool"
    );
}
