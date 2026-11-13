//! `replay_map_raw_serial` needs neither `Send` nor `Sync`: an element type
//! that is neither, and an `FnMut` that captures non-`Send` state, compile and
//! replay every element once.

use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

use strided_basic::{replay_map_raw_serial, RawStridedMut, RawStridedRef};

/// `Copy`, but neither `Send` nor `Sync`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NotSync(f64, PhantomData<*const ()>);

#[test]
fn non_send_elements_and_callbacks_replay() {
    let dims = [3usize, 4];
    let src: Vec<NotSync> = (0..12).map(|v| NotSync(v as f64, PhantomData)).collect();
    let source = RawStridedRef::new(&src, &dims, &[4, 1], 0).unwrap();
    let mut out = vec![NotSync(-1.0, PhantomData); 12];
    let calls = Rc::new(Cell::new(0usize));
    let counter = Rc::clone(&calls);
    let mut seen = Vec::new();
    replay_map_raw_serial(
        &mut RawStridedMut::new(&mut out, &dims, &[1, 3], 0).unwrap(),
        &source,
        |d: &mut NotSync, s: NotSync| {
            counter.set(counter.get() + 1);
            seen.push(s.0);
            *d = s;
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 12);
    // Column-major over the given axes: axis 0 (source stride 4) fastest.
    let expected: Vec<f64> = (0..4)
        .flat_map(|j| (0..3).map(move |i| (4 * i + j) as f64))
        .collect();
    assert_eq!(seen, expected);
    assert_eq!(out.iter().map(|v| v.0).collect::<Vec<_>>(), expected);
}
