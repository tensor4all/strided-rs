//! P6 counting-allocator matrix for the strided-basic planner and fast paths
//! (tensor4all/strided-rs#306 items 2-4, tensor4all/strided-rs#312).
//!
//! Asserts 0 library-owned allocations at rank 1..=8 for every touched entry,
//! on both sides of `SMALL_TENSOR_THRESHOLD`, with contiguous, permuted and
//! stride-0 operands, serial and inside an explicit two-worker pool. Output
//! data of array-returning APIs and allocations made inside user callbacks are
//! out of scope (P1) and are not asserted. The rank-9 spill is executed but not
//! gated: above the inline rank bound the planner may spill by contract.
//!
//! The counter is process global, so the whole matrix runs in one test; the
//! setup and warm-up calls stay outside every counted window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use strided_basic::{
    add, axpy, batched_outer_product_into, col_major_strides, dot, embed_diagonal_into_uninit, fma,
    map_into, map_update_into, mul, reduce_axis_into, symmetrize_into, with_execution_policy,
    zip_map2_into, zip_map3_into, zip_map4_into, zip_update2_into, zip_update3_into,
    ExecutionPolicy, Identity, StridedArray, StridedView, StridedViewMut,
};

const SMALL_TENSOR_THRESHOLD: usize = 1024;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    let _ = THREAD_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocations made by every thread while `run` executes.
fn count_allocations(run: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    run();
    ALLOCATIONS.load(Ordering::SeqCst) - before
}

/// Allocations made by the calling thread while `run` executes.
///
/// Serial windows use this so a concurrently idle worker pool or the global
/// Rayon pool cannot leak background allocations into the count.
fn count_thread_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get) - before
}

#[derive(Clone, Copy, Debug)]
enum LayoutKind {
    Contiguous,
    Permuted,
    Broadcast,
}

/// Immutable identity view over test data.
fn sv<'a>(
    data: &'a [f64],
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<StridedView<'a, f64, Identity>, strided_basic::StridedError> {
    StridedView::new(data, dims, strides, offset)
}

fn strides_for(dims: &[usize], kind: LayoutKind) -> Vec<isize> {
    match kind {
        LayoutKind::Contiguous => col_major_strides(dims).into_vec(),
        LayoutKind::Permuted => {
            let mut strides = col_major_strides(dims).into_vec();
            strides.reverse();
            strides
        }
        LayoutKind::Broadcast => vec![0; dims.len()],
    }
}

fn kind_index(kind: LayoutKind) -> usize {
    match kind {
        LayoutKind::Contiguous => 0,
        LayoutKind::Permuted => 1,
        LayoutKind::Broadcast => 2,
    }
}

fn total(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Backing length needed to make every layout in `strides` in-bounds.
fn required_len(dims: &[usize], strides: &[isize]) -> usize {
    let mut max_offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides) {
        if dim > 1 {
            max_offset += (dim as isize - 1) * stride.abs();
        }
    }
    (max_offset + 1) as usize
}

/// Runs one entry serially and in a two-worker pool, asserting no allocations.
/// `entry` is warmed up first so one-time runtime state stays outside the
/// counted windows. Every input layout in `kinds` is exercised.
fn assert_entry<'a>(
    name: &str,
    pool: &rayon::ThreadPool,
    dims: &[usize],
    kinds: &[LayoutKind],
    mut entry: impl FnMut(LayoutKind) + Send,
) {
    for &kind in kinds {
        // Warm up outside the counted windows, under both execution modes, so
        // the ambient Rayon global pool is never initialized and cannot leak
        // background allocations into the process-wide counter.
        with_execution_policy(ExecutionPolicy::Sequential, || entry(kind));
        pool.install(|| {
            with_execution_policy(
                ExecutionPolicy::Rayon {
                    max_threads: NonZeroUsize::new(2).unwrap(),
                },
                || entry(kind),
            );
        });
        let serial = count_thread_allocations(|| {
            with_execution_policy(ExecutionPolicy::Sequential, || entry(kind));
        });
        assert_eq!(
            serial, 0,
            "{name} {kind:?} serial at rank {dims:?} allocated {serial}"
        );
        let parallel = pool.install(|| {
            count_allocations(|| {
                with_execution_policy(
                    ExecutionPolicy::Rayon {
                        max_threads: NonZeroUsize::new(2).unwrap(),
                    },
                    || entry(kind),
                );
            })
        });
        assert_eq!(
            parallel, 0,
            "{name} {kind:?} two-worker at rank {dims:?} allocated {parallel}"
        );
    }
}

fn run_map_entries(dims: &[usize], pool: &rayon::ThreadPool) {
    let layouts = [
        strides_for(dims, LayoutKind::Contiguous),
        strides_for(dims, LayoutKind::Permuted),
        strides_for(dims, LayoutKind::Broadcast),
    ];
    // Every layout must index inside the same backing buffers, so size them by
    // the widest reachable offset.
    let len = layouts
        .iter()
        .map(|strides| required_len(dims, strides))
        .max()
        .unwrap_or(0)
        .max(total(dims));
    let dst_strides = strides_for(dims, LayoutKind::Contiguous);
    let src: Vec<f64> = (0..len).map(|v| v as f64 + 1.0).collect();
    let a: Vec<f64> = (0..len).map(|v| v as f64 + 2.0).collect();
    let b: Vec<f64> = (0..len).map(|v| v as f64 + 3.0).collect();
    let c: Vec<f64> = (0..len).map(|v| v as f64 + 4.0).collect();
    let mut out = vec![0.0f64; len];

    let kinds = [
        LayoutKind::Contiguous,
        LayoutKind::Permuted,
        LayoutKind::Broadcast,
    ];

    assert_entry("map_into", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let source = sv(&src, dims, &strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        map_into(&mut dest, &source, |x| x + 1.0).unwrap();
    });

    assert_entry("zip_map2_into", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let vb = sv(&b, dims, &dst_strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        zip_map2_into(&mut dest, &va, &vb, |x, y| x + y).unwrap();
    });

    assert_entry("zip_map3_into", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let vb = sv(&b, dims, &dst_strides, 0).unwrap();
        let vc = sv(&c, dims, &dst_strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        zip_map3_into(&mut dest, &va, &vb, &vc, |x, y, z| x + y + z).unwrap();
    });

    assert_entry("zip_map4_into", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let vb = sv(&b, dims, &dst_strides, 0).unwrap();
        let vc = sv(&c, dims, &dst_strides, 0).unwrap();
        let vd = sv(&src, dims, &dst_strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        zip_map4_into(&mut dest, &va, &vb, &vc, &vd, |x, y, z, w| x + y + z + w).unwrap();
    });

    assert_entry("add", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let source = sv(&src, dims, &strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        add(&mut dest, &source).unwrap();
    });

    assert_entry("mul", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let source = sv(&src, dims, &strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        mul(&mut dest, &source).unwrap();
    });

    assert_entry("axpy", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let source = sv(&src, dims, &strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        axpy(&mut dest, &source, 1.5).unwrap();
    });

    assert_entry("fma", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let vb = sv(&b, dims, &dst_strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        fma(&mut dest, &va, &vb).unwrap();
    });

    assert_entry("dot", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let vb = sv(&b, dims, &dst_strides, 0).unwrap();
        let _ = dot::<f64, f64, f64, Identity, Identity>(&va, &vb).unwrap();
    });

    assert_entry("map_update_into", pool, dims, &kinds, |_kind| {
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        map_update_into::<f64, Identity>(&mut dest, |x| x + 1.0).unwrap();
    });

    assert_entry("zip_update2_into", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        zip_update2_into::<f64, f64, Identity, Identity>(&mut dest, &va, |d, a| d + a).unwrap();
    });

    assert_entry("zip_update3_into", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let vb = sv(&b, dims, &dst_strides, 0).unwrap();
        let mut dest = StridedViewMut::new(&mut out, dims, &dst_strides, 0).unwrap();
        zip_update3_into::<f64, f64, f64, Identity, Identity, Identity>(
            &mut dest,
            &va,
            &vb,
            |d, a, b| d + a + b,
        )
        .unwrap();
    });

    // `reduce_axis_into` writes caller storage, so the output is in scope.
    if dims.len() >= 2 {
        let reduced: Vec<usize> = dims[1..].to_vec();
        let reduced_len = total(&reduced);
        let reduced_strides = col_major_strides(&reduced);
        assert_entry("reduce_axis_into", pool, dims, &kinds, |kind| {
            let strides = &layouts[kind_index(kind)];
            let source = sv(&src, dims, &strides, 0).unwrap();
            let mut dest =
                StridedViewMut::new(&mut out[..reduced_len], &reduced, &reduced_strides, 0)
                    .unwrap();
            reduce_axis_into(&mut dest, &source, 0, |x| x, |x, y| x + y, 0.0).unwrap();
        });
    }
}

fn run_composites(pool: &rayon::ThreadPool) {
    // symmetrize, rank 2.
    let dims = [8usize, 8];
    let len = 64;
    let src: Vec<f64> = (0..len).map(|v| v as f64 + 1.0).collect();
    let strides = col_major_strides(&dims);
    let mut out = vec![0.0f64; len];
    assert_entry(
        "symmetrize_into",
        pool,
        &dims,
        &[LayoutKind::Permuted],
        |_kind| {
            let source = sv(&src, &dims, &strides, 0).unwrap();
            let mut dest = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
            symmetrize_into(&mut dest, &source).unwrap();
        },
    );

    // batched outer product: dest [2, 3, n], lhs [2, n], rhs [3, n].
    let batch = 8usize;
    let dest_dims = [2usize, 3, batch];
    let lhs_dims = [2usize, batch];
    let rhs_dims = [3usize, batch];
    let dest_strides = col_major_strides(&dest_dims);
    let lhs_strides = col_major_strides(&lhs_dims);
    let rhs_strides = col_major_strides(&rhs_dims);
    let lhs: Vec<f64> = (0..2 * batch).map(|v| v as f64).collect();
    let rhs: Vec<f64> = (0..3 * batch).map(|v| v as f64 + 1.0).collect();
    let mut dest_data = vec![0.0f64; dest_dims.iter().product()];
    assert_entry(
        "batched_outer_product_into",
        pool,
        &dest_dims,
        &[LayoutKind::Contiguous],
        |_kind| {
            let lhs_view = sv(&lhs, &lhs_dims, &lhs_strides, 0).unwrap();
            let rhs_view = sv(&rhs, &rhs_dims, &rhs_strides, 0).unwrap();
            let mut dest =
                StridedViewMut::new(&mut dest_data, &dest_dims, &dest_strides, 0).unwrap();
            batched_outer_product_into(&mut dest, &lhs_view, &rhs_view, 1, 1).unwrap();
        },
    );

    // embed diagonal, rank 2 -> rank 3.
    let shape = [8usize, 8];
    let len: usize = 64;
    let input: Vec<f64> = (0..len).map(|v| v as f64).collect();
    let mut out: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len * shape[0]];
    assert_entry(
        "embed_diagonal_into_uninit",
        pool,
        &shape,
        &[LayoutKind::Contiguous],
        |_kind| {
            embed_diagonal_into_uninit(&mut out, &input, &shape, 0, 0, 0.0).unwrap();
        },
    );
}

#[test]
fn typed_entries_are_allocation_free_up_to_rank_limit() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| assert_eq!(rayon::current_num_threads(), 2));

    for rank in 1..=8usize {
        let small = vec![2usize; rank];
        let mut large = vec![2usize; rank];
        // Push the element count just past SMALL_TENSOR_THRESHOLD.
        large[0] = (SMALL_TENSOR_THRESHOLD >> (rank - 1)) + 1;
        assert!(total(&small) <= SMALL_TENSOR_THRESHOLD);
        assert!(total(&large) > SMALL_TENSOR_THRESHOLD);

        for dims in [&small, &large] {
            run_map_entries(dims, &pool);
        }
    }

    run_composites(&pool);
    check_reduce_axis(&pool);
    record_rank9_spill(&pool);
}

/// `reduce_axis_into` must be allocation-free; the array-returning
/// `reduce_axis` wrapper must allocate only its column-major output.
fn check_reduce_axis(pool: &rayon::ThreadPool) {
    use strided_basic::reduce_axis;

    let dims = [4usize, 4, 4];
    let src: Vec<f64> = (0..64).map(|v| v as f64).collect();
    let strides = col_major_strides(&dims);
    let source = sv(&src, &dims, &strides, 0).unwrap();
    // Warm up under a serial policy so the ambient global pool is untouched.
    with_execution_policy(ExecutionPolicy::Sequential, || {
        let _ = reduce_axis(&source, 0, |x| x, |a, b| a + b, 0.0).unwrap();
    });
    let wrapper = count_thread_allocations(|| {
        let out = reduce_axis(&source, 0, |x| x, |a, b| a + b, 0.0).unwrap();
        std::hint::black_box(out);
    });
    assert_eq!(
        wrapper, 1,
        "reduce_axis must allocate exactly its column-major output"
    );

    let reduced = [4usize, 4];
    let mut out = StridedArray::<f64>::col_major(&reduced);
    let mut dest = out.view_mut();
    reduce_axis_into(&mut dest, &source, 0, |x| x, |a, b| a + b, 0.0).unwrap();
    let into = count_thread_allocations(|| {
        reduce_axis_into(&mut dest, &source, 0, |x| x, |a, b| a + b, 0.0).unwrap();
    });
    assert_eq!(into, 0, "reduce_axis_into must not allocate");
    let _ = pool;
}

/// Above `INLINE_RANK` the planner spills by contract. This records the spill
/// instead of gating it, so a change from inline to unconditional heap use
/// above the bound is visible without making rank 9 a failure.
fn record_rank9_spill(pool: &rayon::ThreadPool) {
    let dims = vec![2usize; 9];
    let len = total(&dims);
    let src: Vec<f64> = (0..len).map(|v| v as f64).collect();
    let mut out = vec![0.0f64; len];
    let dest_strides = strides_for(&dims, LayoutKind::Contiguous);
    let src_strides = strides_for(&dims, LayoutKind::Permuted);
    let source = sv(&src, &dims, &src_strides, 0).unwrap();
    let mut dest = StridedViewMut::new(&mut out, &dims, &dest_strides, 0).unwrap();
    // Warm up under a serial policy so the ambient global pool is untouched.
    with_execution_policy(ExecutionPolicy::Sequential, || {
        map_into(&mut dest, &source, |x| x + 1.0).unwrap();
    });
    let serial = count_thread_allocations(|| {
        with_execution_policy(ExecutionPolicy::Sequential, || {
            map_into(&mut dest, &source, |x| x + 1.0).unwrap();
        });
    });
    let parallel = pool.install(|| {
        count_allocations(|| {
            with_execution_policy(
                ExecutionPolicy::Rayon {
                    max_threads: NonZeroUsize::new(2).unwrap(),
                },
                || {
                    map_into(&mut dest, &source, |x| x + 1.0).unwrap();
                },
            );
        })
    });
    eprintln!("rank-9 permuted map_into spill: serial={serial}, parallel={parallel}");
}
