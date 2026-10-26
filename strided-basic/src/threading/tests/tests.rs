use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use threadteam::ledger;

fn pooled(pool: &threadteam::Pool, budget: usize) -> (Exec<'_>, u64) {
    let exec = Exec::new(pool, budget).unwrap();
    let root = exec.ledger_root().unwrap();
    (exec, root)
}

#[test]
fn split_axis_breaks_ties_towards_the_last_axis() {
    let blocks = [1usize; 3];
    // scores (10-1)*2, (20-1)*1, (5-1)*3 = 18, 19, 12: the maximum is axis 1.
    assert_eq!(split_axis(&[10, 20, 5], &blocks, &[2, 1, 3]), Some(1));
    // Equal scores: the last axis wins (`>=`).
    assert_eq!(split_axis(&[10, 10, 10], &blocks, &[1, 1, 1]), Some(2));
    // Single dimension.
    assert_eq!(split_axis(&[1 << 20], &[1 << 10], &[2]), Some(0));
}

#[test]
fn split_axis_declines_a_zero_cost_or_one_block_dimension() {
    // The best axis has zero cost.
    assert_eq!(split_axis(&[1 << 20], &[1 << 10], &[0]), None);
    // The best axis fits in one block.
    assert_eq!(split_axis(&[512], &[1024], &[2]), None);
    // All extents one: scores are all zero, the last axis is picked and declined.
    assert_eq!(split_axis(&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]), None);
}

/// A region length the test pools (100 ns per section) always open for.
const LARGE: usize = 1 << 16;

#[test]
fn parallel_width_follows_the_serial_time_against_the_section_cost() {
    use crate::cost::{parallel_width, tree_width};
    let pool = test_pool(4);
    let (exec, _) = pooled(&pool, 4);
    // 100 ns sections: a copy pays at 3 * 100 ns / 0.09 ns = 3334 elements on two slots.
    assert_eq!(parallel_width(&exec, KernelClass::Copy, 100).get(), 1);
    assert_eq!(parallel_width(&exec, KernelClass::Copy, LARGE).get(), 4);
    // A transcendental kernel pays at a much smaller length than a copy.
    assert!(
        parallel_width(&exec, KernelClass::Transcendental, 1000).get()
            > parallel_width(&exec, KernelClass::Copy, 1000).get()
    );
    // A tree of width 4 holds two sections, so it needs twice the work of width 2.
    assert_eq!(tree_width(&exec, KernelClass::Reduce, 100).get(), 1);
    assert_eq!(tree_width(&exec, KernelClass::Reduce, LARGE).get(), 4);
    // A serial token and a width one budget never fan out.
    assert_eq!(
        parallel_width(&Exec::serial(), KernelClass::Copy, usize::MAX >> 8).get(),
        1
    );
}

#[test]
fn for_len_narrows_with_the_estimate_and_keeps_the_budget_above_it() {
    let pool = test_pool(4);
    let (exec, _) = pooled(&pool, 4);
    assert!(for_len(exec, KernelClass::Copy, 100).is_serial());
    let (exec, _) = pooled(&pool, 4);
    let wide = for_len(exec, KernelClass::Copy, LARGE);
    assert_eq!(wide.budget().get(), pool.width().get().min(4));
}

#[test]
fn test_mapreduce_threaded_single_slot() {
    // With a serial token, the leaf is called once on the whole region.
    let dims = vec![10, 10];
    let blocks = vec![10, 10];
    let strides = vec![vec![1isize, 10], vec![1, 10]];
    let offsets = vec![0isize, 0];
    let costs = vec![2, 20];

    let calls = AtomicUsize::new(0);
    mapreduce_threaded(
        &dims,
        &blocks,
        &strides,
        &offsets,
        &costs,
        KernelClass::Map,
        Exec::serial(),
        &|_dims, _blocks, _strides, _offsets| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn test_mapreduce_threaded_splits_cover_all_elements() {
    // The pooled splits visit every element exactly once.
    let pool = test_pool(4);
    let (exec, root) = pooled(&pool, 4);
    let budget = exec.budget().get();
    let dims = vec![256, 256];
    let blocks = vec![256, 256];
    let strides = vec![vec![1isize, 256], vec![1, 256]];
    let offsets = vec![0isize, 0];
    let costs = vec![2, 512];
    assert!(dims.iter().product::<usize>() >= LARGE);

    let total_elements = AtomicUsize::new(0);
    mapreduce_threaded(
        &dims,
        &blocks,
        &strides,
        &offsets,
        &costs,
        KernelClass::Map,
        exec,
        &|dims, _blocks, _strides, _offsets| {
            let n: usize = dims.iter().product();
            total_elements.fetch_add(n, Ordering::Relaxed);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(total_elements.load(Ordering::SeqCst), 256 * 256);
    assert!(ledger::max_live(root) <= budget);
}

#[test]
fn mapreduce_threaded_leaves_never_exceed_the_budget() {
    let pool = test_pool(4);
    let (exec, root) = pooled(&pool, 4);
    let exec = exec.limit(std::num::NonZeroUsize::new(2).unwrap());
    let leaves = AtomicUsize::new(0);
    let covered = AtomicUsize::new(0);
    let dims = [1usize << 17];
    let blocks = [1usize << 10];
    let strides = [vec![1isize]];
    let offsets = [0isize];
    let costs = [2isize];

    mapreduce_threaded(
        &dims,
        &blocks,
        &strides,
        &offsets,
        &costs,
        KernelClass::Map,
        exec,
        &|dims, _blocks, _strides, _offsets| {
            leaves.fetch_add(1, Ordering::SeqCst);
            covered.fetch_add(dims.iter().product::<usize>(), Ordering::SeqCst);
            Ok(())
        },
    )
    .unwrap();

    assert!(
        leaves.load(Ordering::SeqCst) <= 2,
        "mapreduce_threaded created {} leaves with a budget of 2",
        leaves.load(Ordering::SeqCst)
    );
    assert_eq!(covered.load(Ordering::SeqCst), 1usize << 17);
    assert!(ledger::max_live(root) <= 2);
}

#[test]
fn parallel_map_reduce_matches_the_serial_sum() {
    let pool = test_pool(4);
    let len = LARGE * 2;
    let expected: usize = (0..len).sum();
    let serial = parallel_map_reduce(
        0..len,
        Exec::serial(),
        &|range| range.sum::<usize>(),
        &|left, right| left + right,
    );
    let (exec, root) = pooled(&pool, 4);
    let budget = exec.budget().get();
    let pooled_sum = parallel_map_reduce(
        0..len,
        exec,
        &|range| range.sum::<usize>(),
        &|left, right| left + right,
    );
    assert_eq!(serial, expected);
    assert_eq!(pooled_sum, expected);
    assert!(ledger::max_live(root) <= budget);
}

#[test]
fn parallel_map_reduce_leaves_never_exceed_the_budget() {
    let pool = test_pool(4);
    let (exec, root) = pooled(&pool, 4);
    let exec = exec.limit(std::num::NonZeroUsize::new(2).unwrap());
    let callback_count = AtomicUsize::new(0);
    let active = AtomicUsize::new(0);
    let max_active = AtomicUsize::new(0);

    let sum = parallel_map_reduce(
        0..4096,
        exec,
        &|range| {
            callback_count.fetch_add(1, Ordering::SeqCst);
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(now, Ordering::SeqCst);
            for _ in 0..64 {
                std::hint::spin_loop();
            }
            active.fetch_sub(1, Ordering::SeqCst);
            range.sum::<usize>()
        },
        &|left, right| left + right,
    );

    assert_eq!(sum, (0..4096usize).sum::<usize>());
    assert!(
        callback_count.load(Ordering::SeqCst) <= 2,
        "the splitter created {} leaf partitions with a budget of 2",
        callback_count.load(Ordering::SeqCst)
    );
    assert!(max_active.load(Ordering::SeqCst) <= 2);
    assert!(ledger::max_live(root) <= 2);
}

#[test]
fn budget_one_token_of_a_wide_pool_stays_on_the_calling_thread() {
    // A budget of one runs inline: nothing is installed and no worker runs.
    let pool = test_pool(4);
    let (exec, root) = pooled(&pool, 1);
    let caller = std::thread::current().id();
    let off_caller = AtomicUsize::new(0);

    let sum = parallel_map_reduce(
        0..(LARGE * 2),
        exec,
        &|range| {
            if std::thread::current().id() != caller {
                off_caller.fetch_add(range.len(), Ordering::SeqCst);
            }
            range.sum::<usize>()
        },
        &|left, right| left + right,
    );

    assert_eq!(sum, (0..(LARGE * 2)).sum::<usize>());
    assert_eq!(off_caller.load(Ordering::SeqCst), 0);
    assert_eq!(ledger::installs(root), 0);
}

#[test]
fn small_copy_stays_on_the_calling_thread_whatever_the_budget() {
    let pool = test_pool(4);
    let (exec, root) = pooled(&pool, 4);
    let src = vec![1.0f64; 1024];
    let mut dst = vec![0.0f64; 1024];
    // SAFETY: both buffers hold 1024 elements and do not overlap.
    unsafe { copy_contiguous(src.as_ptr(), dst.as_mut_ptr(), 1024, exec) };
    assert_eq!(dst, src);
    assert_eq!(ledger::installs(root), 0);
    assert_eq!(ledger::max_live(root), 0);
}

#[test]
fn large_copy_runs_on_pool_workers_when_the_budget_allows() {
    let pool = test_pool(4);
    let (exec, root) = pooled(&pool, 4);
    let parallel = exec.budget().get() > 1;
    let len = LARGE * 2;
    let src: Vec<f64> = (0..len).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; len];
    // SAFETY: both buffers hold `len` elements and do not overlap.
    unsafe { copy_contiguous(src.as_ptr(), dst.as_mut_ptr(), len, exec) };
    assert_eq!(dst, src);
    assert!(ledger::max_live(root) <= 4);
    if parallel {
        assert!(ledger::max_live(root) >= 1);
        assert_eq!(ledger::installs(root), 1);
    }
}

const ROWS: usize = 257;
const COLS: usize = 129;
const _: () = assert!(ROWS * COLS >= LARGE / 2);

fn transposed_pair() -> (crate::StridedArray<f64>, crate::StridedArray<f64>) {
    let source = crate::StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] + ROWS * index[1]) as f64
    });
    // A column-major destination against the permuted source is a real transpose;
    // a row-major one would be a flat copy.
    let destination = crate::StridedArray::<f64>::col_major(&[COLS, ROWS]);
    (source, destination)
}

fn assert_transposed(source: &crate::StridedArray<f64>, destination: &crate::StridedArray<f64>) {
    for row in 0..ROWS {
        for column in 0..COLS {
            assert_eq!(destination.get(&[column, row]), source.get(&[row, column]));
        }
    }
}

/// The permutation engine takes the token directly, so a pooled copy enters
/// the pool exactly once and uses at most its budget.
#[test]
fn pooled_permutation_copy_installs_once_within_the_budget() {
    let pool = test_pool(4);
    let (source, mut destination) = transposed_pair();
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let (exec, root) = pooled(&pool, 4);
    let budget = exec.budget().get();

    copy_permuted(&mut destination.view_mut(), &transposed, exec).unwrap();

    assert_transposed(&source, &destination);
    assert!(ledger::max_live(root) <= budget);
    assert_eq!(ledger::installs(root), usize::from(budget > 1));
}

/// A token narrower than the pool bounds the permutation copy as well: the
/// engine has no route that ignores the budget.
#[test]
fn permutation_copy_honours_a_budget_below_the_pool_width() {
    let pool = test_pool(4);
    let (source, mut destination) = transposed_pair();
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let (exec, root) = pooled(&pool, 2);

    copy_permuted(&mut destination.view_mut(), &transposed, exec).unwrap();

    assert_transposed(&source, &destination);
    assert!(
        ledger::max_live(root) <= 2,
        "permutation copy ran {} tasks at once with a budget of 2",
        ledger::max_live(root)
    );
}

#[test]
fn permutation_copy_into_uninit_matches_the_initialized_copy() {
    let pool = test_pool(4);
    let (source, mut want) = transposed_pair();
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let (exec, root) = pooled(&pool, 4);
    let budget = exec.budget().get();
    copy_permuted(&mut want.view_mut(), &transposed, Exec::serial()).unwrap();

    let mut data = vec![core::mem::MaybeUninit::<f64>::uninit(); ROWS * COLS];
    let mut view =
        crate::StridedViewMut::new(&mut data, &[COLS, ROWS], &[1, COLS as isize], 0).unwrap();
    copy_permuted_uninit(&mut view, &transposed, exec).unwrap();
    // SAFETY: the copy wrote every destination element.
    let got: Vec<f64> = data
        .into_iter()
        .map(|x| unsafe { x.assume_init() })
        .collect();
    assert_eq!(got, want.data());
    assert!(ledger::max_live(root) <= budget);
}

/// A caller that already is a worker of the target pool spawns into the
/// current scope instead of installing a second time.
#[cfg(feature = "parallel")]
#[test]
fn a_call_from_a_pool_worker_does_not_install() {
    let pool = test_pool(4);
    let (source, mut destination) = transposed_pair();
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let inner_root = std::sync::Mutex::new(None);
    let destination_view = std::sync::Mutex::new(destination.view_mut());

    // A two-task fork enters the pool, so its first job runs on a worker.
    Exec::new(&pool, 2).unwrap().fork(2, |index, _| {
        if index != 0 {
            return;
        }
        let exec = Exec::new(&pool, 4).unwrap();
        *inner_root.lock().unwrap() = exec.ledger_root();
        let mut view = destination_view.lock().unwrap();
        copy_permuted(&mut view, &transposed, exec).unwrap();
    });
    drop(destination_view);

    let root = inner_root.lock().unwrap().expect("the closure ran");
    assert_eq!(ledger::installs(root), 0);
    assert_transposed(&source, &destination);
}

/// A length where a fork pays for a map (3 sections * 100 ns / 0.100 ns = 3000
/// elements) but not yet for a copy (3 * 100 / 0.090 = 3334 elements) on the
/// test pools.
#[cfg(feature = "parallel")]
const MAP_PAYS_COPY_NOT: usize = 3200;

#[cfg(feature = "parallel")]
#[test]
fn the_contiguous_gate_asks_with_the_class_of_the_caller() {
    use crate::kernel::sequential_contiguous_layout;
    let pool = test_pool(4);
    let (exec, _) = pooled(&pool, 4);
    let strides = [1isize];
    let dims = [MAP_PAYS_COPY_NOT];
    // The fork pays for a map, so the sequential shortcut is declined...
    assert!(
        sequential_contiguous_layout(KernelClass::Map, &dims, &[&strides, &strides], &exec)
            .unwrap()
            .is_none()
    );
    // ...and still taken for a copy and a reduction of the same length.
    assert!(
        sequential_contiguous_layout(KernelClass::Copy, &dims, &[&strides, &strides], &exec)
            .unwrap()
            .is_some()
    );
    assert!(
        sequential_contiguous_layout(KernelClass::Reduce, &dims, &[&strides], &exec)
            .unwrap()
            .is_some()
    );
}

/// `copy_into` of a contiguous run the fork does not pay for as a copy stays on
/// the calling thread; it used to ask the gate as `Map` and enter the pool.
#[cfg(feature = "parallel")]
#[test]
fn contiguous_copy_into_is_gated_as_a_copy() {
    let pool = test_pool(4);
    let src: Vec<f64> = (0..MAP_PAYS_COPY_NOT).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; MAP_PAYS_COPY_NOT];
    let (exec, root) = pooled(&pool, 4);
    crate::copy_into(
        &mut crate::StridedViewMut::new(&mut dst, &[MAP_PAYS_COPY_NOT], &[1], 0).unwrap(),
        &crate::StridedView::<f64>::new(&src, &[MAP_PAYS_COPY_NOT], &[1], 0).unwrap(),
        exec,
    )
    .unwrap();
    assert_eq!(dst, src);
    assert_eq!(ledger::installs(root), 0);

    // The same length as a map does fork, which pins the length to the window.
    let (exec, root) = pooled(&pool, 4);
    crate::map_into(
        &mut crate::StridedViewMut::new(&mut dst, &[MAP_PAYS_COPY_NOT], &[1], 0).unwrap(),
        &crate::StridedView::<f64>::new(&src, &[MAP_PAYS_COPY_NOT], &[1], 0).unwrap(),
        |x| x,
        exec,
    )
    .unwrap();
    assert_eq!(ledger::installs(root), 1);
}

/// A length where a fork pays for a transcendental element function
/// (3 * 100 ns / 1.38 ns = 218 elements) but not for a plain map (3000 elements)
/// on the test pools.
#[cfg(feature = "parallel")]
const TRANSCENDENTAL_PAYS_MAP_NOT: usize = 1000;

/// The class rides with the function passed to `map_into`: a closure is a `Map`
/// and stays on the calling thread, the same closure declared `Transcendental`
/// enters the pool, and one declared `Map` explicitly behaves as the bare closure.
#[cfg(feature = "parallel")]
#[test]
fn map_into_gates_on_the_class_of_the_function() {
    use crate::{Classed, MapFn};
    let pool = test_pool(4);
    let n = TRANSCENDENTAL_PAYS_MAP_NOT;
    let src: Vec<f64> = (0..n).map(|i| i as f64 * 1e-3).collect();
    let expected: Vec<f64> = src.iter().map(|x| x.exp()).collect();
    let run = |f: &dyn Fn(&mut [f64], crate::Exec<'_>)| {
        let mut dst = vec![0.0f64; n];
        let (exec, root) = pooled(&pool, 4);
        f(&mut dst, exec);
        assert_eq!(dst, expected);
        ledger::installs(root)
    };
    let view_src = || crate::StridedView::<f64>::new(&src, &[n], &[1], 0).unwrap();

    let plain = run(&|dst, exec| {
        let mut d = crate::StridedViewMut::new(dst, &[n], &[1], 0).unwrap();
        crate::map_into(&mut d, &view_src(), |x: f64| x.exp(), exec).unwrap();
    });
    assert_eq!(plain, 0);

    let map_class = run(&|dst, exec| {
        let mut d = crate::StridedViewMut::new(dst, &[n], &[1], 0).unwrap();
        let f = Classed::new(KernelClass::Map, |x: f64| x.exp());
        crate::map_into(&mut d, &view_src(), f, exec).unwrap();
    });
    assert_eq!(map_class, 0);

    let transcendental = run(&|dst, exec| {
        let mut d = crate::StridedViewMut::new(dst, &[n], &[1], 0).unwrap();
        let f = KernelClass::Transcendental.with(|x: f64| x.exp());
        crate::map_into(&mut d, &view_src(), f, exec).unwrap();
    });
    assert_eq!(transcendental, 1);

    // The class a function reports.
    assert_eq!(MapFn::<f64, f64>::class(&|x: f64| x), KernelClass::Map);
    assert_eq!(
        MapFn::<f64, f64>::class(&KernelClass::Reduce.with(|x: f64| x)),
        KernelClass::Reduce
    );
}
