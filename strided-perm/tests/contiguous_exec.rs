//! Contiguous copies must consume the caller's placement token (#347).
use std::{mem::MaybeUninit, time::Duration};

use strided_perm::{copy_into, copy_into_col_major, copy_into_uninit};
use strided_view::{StridedView, StridedViewMut};
use threadteam::{ledger, DispatchCosts, Exec, Pool, WidthCosts};

#[test]
fn contiguous_copies_use_caller_exec_and_match_serial() {
    // Force a deterministic width decision, independent of host cost tables.
    let zero = WidthCosts::flat(Duration::ZERO);
    let pool = Pool::builder()
        .threads(2)
        .dispatch_costs(DispatchCosts::new(zero, zero, zero))
        .build()
        .unwrap();
    let len = 257 * 509; // An uneven split must still write every element once.
    let source: Vec<u64> = (0..len)
        .map(|i| (i as u64).wrapping_mul(u64::MAX / 3))
        .collect();
    for (dims, strides) in [
        (vec![len], vec![1]),
        (vec![257, 509], vec![1, 257]),
        (vec![257, 509], vec![509, 1]),
        (vec![257, 1, 509, 1], vec![1, -17, 257, 0]),
    ] {
        // Nonzero origins and sentinels guard the edges of the flat ranges.
        let src = StridedView::new(&source, &dims, &strides, 0).unwrap();
        let mut serial = vec![17; len + 2];
        let mut parallel = serial.clone();
        copy_into(
            &mut StridedViewMut::new(&mut serial, &dims, &strides, 1).unwrap(),
            &src,
            Exec::serial(),
        )
        .unwrap();
        assert_eq!(&serial[1..=len], source);
        for copy in [copy_into::<u64>, copy_into_col_major::<u64>] {
            parallel.fill(17);
            let exec = Exec::new(&pool, 2).unwrap();
            let root = exec.ledger_root().unwrap();
            copy(
                &mut StridedViewMut::new(&mut parallel, &dims, &strides, 1).unwrap(),
                &src,
                exec,
            )
            .unwrap();
            assert_eq!(parallel, serial);
            // A plain outside-pool token installs exactly once iff it splits.
            assert_eq!(
                ledger::installs(root),
                usize::from(cfg!(feature = "parallel"))
            );
        }
        for budget in [1, 2] {
            let mut uninit = vec![MaybeUninit::<u64>::uninit(); len + 2];
            uninit[0].write(17);
            uninit[len + 1].write(17);
            let exec = Exec::new(&pool, budget).unwrap();
            let root = exec.ledger_root().unwrap();
            copy_into_uninit(
                &mut StridedViewMut::new(&mut uninit, &dims, &strides, 1).unwrap(),
                &src,
                exec,
            )
            .unwrap();
            for (slot, expected) in uninit.iter().zip(&serial) {
                // SAFETY: the copy initialized the view; both sentinels were initialized above.
                assert_eq!(unsafe { slot.assume_init() }, *expected);
            }
            assert_eq!(
                ledger::installs(root),
                usize::from(cfg!(feature = "parallel") && budget > 1)
            );
        }
    }
}

#[test]
fn contiguous_copy_keeps_narrowed_serial_token_inline() {
    // Even a multi-slot token must stay inline when its dispatch cost is too high.
    let high = WidthCosts::flat(Duration::from_secs(1));
    let pool = Pool::builder()
        .threads(2)
        .dispatch_costs(DispatchCosts::new(high, high, high))
        .build()
        .unwrap();
    let src = [3u64; 1024];
    let mut dst = [0u64; 1024];
    let exec = Exec::new(&pool, 2).unwrap();
    let root = exec.ledger_root().unwrap();
    copy_into(
        &mut StridedViewMut::new(&mut dst, &[1024], &[1], 0).unwrap(),
        &StridedView::new(&src, &[1024], &[1], 0).unwrap(),
        exec,
    )
    .unwrap();
    assert_eq!(dst, src);
    assert_eq!(ledger::installs(root), 0);
}
