//! `copy_into_col_major` must do its work inside the token it is handed.
//!
//! The bytes a copy produces are the same on either route, so this asserts the
//! *route*: a large permutation copy called from outside its pool must install
//! work into that pool. The pre-#212 wrapper hard-coded `Exec::serial()`, so it
//! never installed anything and this failed.

#![cfg(feature = "parallel")]

mod common;

use strided_basic::{copy_into_col_major, StridedArray};

/// A 1024x1024 `f64` transpose: about 94 us of serial copy work, which repays
/// the 100 ns sections of the shared test pool four times over.
const N: usize = 1024;

#[test]
fn copy_into_col_major_uses_the_callers_token() {
    let source =
        StridedArray::<f64>::from_fn_col_major(&[N, N], |index| (index[0] + N * index[1]) as f64);
    // A transposed source keeps the copy on the permutation engine instead of
    // a contiguous fast path.
    let source = source.view().permute(&[1, 0]).unwrap();
    let mut destination = StridedArray::<f64>::col_major(&[N, N]);

    let exec = common::Budget::Pooled(4).exec();
    let root = exec.ledger_root().unwrap();
    copy_into_col_major(&mut destination.view_mut(), &source, exec).unwrap();

    // The caller is outside the pool, so installing work into it is the proof
    // that the copy used the token. A serial token installs nothing, and the
    // counter does not depend on how the workers happen to interleave.
    let installs = threadteam::ledger::installs(root);
    assert!(
        installs > 0,
        "the copy never left the calling thread ({installs} installs): it did not use its token"
    );
    // The parallel route must still produce the copy (the byte-for-byte check on
    // the serial route is `ops_view::tests::tiled_tests`).
    assert_eq!(destination.get(&[N - 1, N - 1]), (N * N - 1) as f64);
    // Off-diagonal, so a source that was not transposed fails here: symmetry made
    // the corner element above pass either way.
    assert_eq!(destination.get(&[1, 0]), N as f64);
}
