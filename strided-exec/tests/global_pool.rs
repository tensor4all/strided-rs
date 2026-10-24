//! The process-global Rayon pool must never be started implicitly.
//!
//! This file holds a single test so that the process is pristine: after serial
//! and pooled work, installing a global pool must still succeed, which fails
//! once anything has started the global registry.
#![cfg(feature = "parallel")]

use std::num::NonZeroUsize;
use strided_exec::{Exec, Pool};

#[test]
fn global_pool_is_never_started() {
    let nz = NonZeroUsize::MIN;
    Exec::serial().fork(4, |_, child| child.for_ranges(8, nz, |_| {}));
    Exec::serial().for_ranges(16, nz, |_| {});
    Exec::serial().broadcast(|_| {}).unwrap();

    let pool = Pool::builder().threads(3).build().unwrap();
    assert!(pool.width().get() >= 1);
    let mut exec = Exec::new(&pool, 3).unwrap();
    exec.reborrow()
        .fork(5, |_, child| child.for_ranges(32, nz, |_| {}));
    exec.reborrow().for_ranges(100, nz, |_| {});
    exec.reborrow().broadcast(|_| {}).unwrap();
    #[cfg(feature = "faer")]
    exec.reborrow().faer(|_par| ());
    Exec::new(&pool, 1).unwrap().fork(3, |_, _| {});

    // A worker of the pool calling back into its own pool.
    Exec::new(&pool, 2).unwrap().fork(2, |_, _| {
        Exec::new(&pool, 2).unwrap().for_ranges(8, nz, |_| {});
    });

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .expect("the global pool was started by strided-exec");
}
