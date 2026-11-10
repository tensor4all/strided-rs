//! The permutation engine takes its placement from the token, never from an
//! ambient pool (tensor4all/strided-rs#209).
//!
//! A serial token keeps a large non-contiguous permutation on the calling
//! thread and never starts the process-global pool. A pooled token whose
//! dispatch cost the copy cannot repay narrows to one slot and installs
//! nothing, while the same copy under a free section installs work. Both
//! layouts take the HPTT engine, not the contiguous `memcpy` path that
//! `contiguous_exec.rs` covers.
//!
//! One test in its own binary: `ledger::global_pool_unstarted` starts the
//! process-global pool itself, so nothing else may run in this process.

#![cfg(feature = "parallel")]

use std::time::Duration;

use strided_perm::copy_into;
use strided_view::StridedArray;
use threadteam::{ledger, DispatchCosts, Exec, Pool, WidthCosts};

/// 257 x 257 = 66049 elements: past the `1 << 15` gate the engine used before
/// #209, so a size-based fan-out would open here.
const N: usize = 257;

/// The destination must hold the transpose of `source`. Called after each copy
/// so that every route is observed on its own output, not on whatever the next
/// copy leaves behind.
fn assert_transposed(destination: &StridedArray<f64>, source: &StridedArray<f64>, route: &str) {
    for i in 0..N {
        for j in 0..N {
            assert_eq!(
                destination.get(&[i, j]),
                source.get(&[j, i]),
                "{route}: mismatch at [{i}, {j}]"
            );
        }
    }
}

fn pool_with(cost: Duration) -> Pool {
    let costs = WidthCosts::flat(cost);
    Pool::builder()
        .threads(4)
        .dispatch_costs(DispatchCosts::new(costs, costs, costs))
        .build()
        .unwrap()
}

#[test]
fn the_permutation_runs_on_the_token_it_is_handed() {
    let source =
        StridedArray::<f64>::from_fn_col_major(&[N, N], |index| (index[0] * N + index[1]) as f64);
    // Swapped axes: the two layouts disagree, so the blocked engine runs.
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut destination = StridedArray::<f64>::col_major(&[N, N]);

    // A section this expensive is never repaid: the token narrows to one slot
    // and the copy stays on the calling thread.
    let cold = pool_with(Duration::from_secs(1));
    let exec = Exec::new(&cold, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    copy_into(&mut destination.view_mut(), &transposed, exec).unwrap();
    assert_eq!(
        ledger::installs(root),
        0,
        "a copy too small to repay a fan-out installed work"
    );
    assert_transposed(&destination, &source, "the narrow-to-one-slot copy");

    // The same copy under a free section fans out: the token's dispatch cost
    // decides, not the element count.
    let hot = pool_with(Duration::ZERO);
    let exec = Exec::new(&hot, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    copy_into(&mut destination.view_mut(), &transposed, exec).unwrap();
    assert!(
        ledger::installs(root) > 0,
        "a free section never fanned out: the token was ignored"
    );
    // Check the pooled copy *before* the serial copy replaces its output: with
    // only the final comparison (review of cf164de), a pooled copy that wrote
    // the wrong values still passed because the serial copy overwrote it.
    assert_transposed(&destination, &source, "the pooled copy");

    // The serial route writes the transposed source and never starts the
    // process-global pool (the rayon guard of #209). Poison the destination
    // first: otherwise the pooled copy's output already satisfies the value
    // check and deleting this copy would go unnoticed (review of 4168cea).
    for i in 0..N {
        for j in 0..N {
            destination.set(&[i, j], f64::NAN);
        }
    }
    copy_into(&mut destination.view_mut(), &transposed, Exec::serial()).unwrap();
    assert_transposed(&destination, &source, "the serial copy");
    assert!(
        ledger::global_pool_unstarted(),
        "the permutation reached the process-global pool"
    );
}
