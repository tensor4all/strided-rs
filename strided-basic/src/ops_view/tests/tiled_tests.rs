use super::*;
use crate::view::StridedArray;

#[test]
fn test_f64_tiled_transpose_scale_handles_remainders() {
    let rows = 7;
    let cols = 9;
    let a =
        StridedArray::<f64>::from_fn_col_major(&[rows, cols], |idx| (idx[0] * 100 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::col_major(&[cols, rows]);

    let used_tiled = {
        let src = a.view();
        let mut dst = out.view_mut();
        unsafe { try_copy_transpose_scale_2d_f64_tiled(&mut dst, &src, 3.0) }
    };

    assert!(used_tiled);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(out.get(&[j, i]), 3.0 * a.get(&[i, j]));
        }
    }
}

#[test]
fn test_identity_tiled_transpose_scale_handles_integer_remainders() {
    let rows = 6;
    let cols = 5;
    let a =
        StridedArray::<u64>::from_fn_col_major(&[rows, cols], |idx| (idx[0] * 100 + idx[1]) as u64);
    let mut out = StridedArray::<u64>::col_major(&[cols, rows]);

    let used_tiled = {
        let src = a.view();
        let mut dst = out.view_mut();
        unsafe { try_copy_transpose_scale_2d_identity_tiled(&mut dst, &src, 2) }
    };

    assert!(used_tiled);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(out.get(&[j, i]), 2 * a.get(&[i, j]));
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_bounded_identity_tiled_transpose_uses_two_partitions_exactly_once() {
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    struct TileState {
        active: AtomicUsize,
        max_active: AtomicUsize,
        released: AtomicBool,
        coverage: Box<[AtomicUsize]>,
    }

    impl TileState {
        fn observe(&self, index: usize) {
            self.coverage[index].fetch_add(1, Ordering::SeqCst);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            if active >= 2 {
                self.released.store(true, Ordering::Release);
            } else if !self.released.load(Ordering::Acquire) {
                let deadline = Instant::now() + Duration::from_secs(2);
                while !self.released.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::hint::spin_loop();
                }
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Clone, Copy)]
    struct TrackedTile {
        index: usize,
        value: usize,
        state: &'static TileState,
    }

    impl std::ops::Mul for TrackedTile {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self.state.observe(rhs.index);
            Self {
                index: rhs.index,
                value: self.value * rhs.value,
                state: self.state,
            }
        }
    }

    const ROWS: usize = 257;
    const COLS: usize = 129;
    const LEN: usize = ROWS * COLS;
    let state = Box::leak(Box::new(TileState {
        active: AtomicUsize::new(0),
        max_active: AtomicUsize::new(0),
        released: AtomicBool::new(false),
        coverage: (0..LEN)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    }));
    let source: Vec<_> = (0..LEN)
        .map(|index| TrackedTile {
            index,
            value: index + 1,
            state,
        })
        .collect();
    let mut destination = vec![
        TrackedTile {
            index: usize::MAX,
            value: 0,
            state,
        };
        LEN
    ];
    let scale = TrackedTile {
        index: usize::MAX,
        value: 3,
        state,
    };
    let two = NonZeroUsize::new(2).unwrap();

    crate::threading::test_pool(4).install(|| {
        crate::with_execution_policy(
            crate::ExecutionPolicy::Rayon { max_threads: two },
            || unsafe {
                copy_transpose_scale_2d_identity_tiled_raw(
                    destination.as_mut_ptr(),
                    1,
                    COLS as isize,
                    source.as_ptr(),
                    1,
                    ROWS as isize,
                    ROWS,
                    COLS,
                    scale,
                );
            },
        );
    });

    assert_eq!(state.max_active.load(Ordering::SeqCst), 2);
    for (index, count) in state.coverage.iter().enumerate() {
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "tiled transpose source index {index} did not execute exactly once"
        );
    }
    for i in 0..ROWS {
        for j in 0..COLS {
            let source_index = i + ROWS * j;
            assert_eq!(destination[j + COLS * i].value, 3 * (source_index + 1));
        }
    }
}

#[test]
fn test_zero_scale_fills_non_contiguous_destination() {
    let rows = 3;
    let cols = 4;
    let a = StridedArray::<u64>::from_fn_col_major(&[rows, cols], |idx| {
        (idx[0] * 10 + idx[1] + 1) as u64
    });
    let mut out_base = StridedArray::<u64>::from_fn_col_major(&[rows, cols], |_| 99);
    let mut out_t = out_base.view_mut().permute(&[1, 0]).unwrap();

    copy_transpose_scale_into(&mut out_t, &a.view(), 0).unwrap();

    for i in 0..cols {
        for j in 0..rows {
            assert_eq!(out_t.get(&[i, j]), 0);
        }
    }
}
