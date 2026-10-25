//! Dropping a `Pool` joins its workers.
#![cfg(feature = "parallel")]

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use strided_exec::{Exec, Pool};

#[test]
fn drop_of_an_unused_pool_returns() {
    drop(Pool::builder().threads(3).build().unwrap());
}

struct Flag(Arc<AtomicUsize>);
impl Drop for Flag {
    fn drop(&mut self) {
        // Slow destructor: a pool that did not join would return first.
        std::thread::sleep(std::time::Duration::from_millis(50));
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

thread_local! {
    static GUARD: Cell<Option<Flag>> = const { Cell::new(None) };
}

#[test]
fn drop_returns_after_every_worker_destructor_ran() {
    let started = Arc::new(AtomicUsize::new(0));
    let exited = Arc::new(AtomicUsize::new(0));
    let (s, e) = (Arc::clone(&started), Arc::clone(&exited));
    let pool = Pool::builder()
        .threads(4)
        .on_start(move |_| {
            s.fetch_add(1, Ordering::SeqCst);
            let flag = Flag(Arc::clone(&e));
            GUARD.with(|g| g.set(Some(flag)));
        })
        .build()
        .unwrap();
    Exec::new(&pool, 4).unwrap().fork(8, |_, _| {});
    let n = pool.width().get();
    drop(pool);
    assert_eq!(exited.load(Ordering::SeqCst), n);
    assert_eq!(started.load(Ordering::SeqCst), n);
}
