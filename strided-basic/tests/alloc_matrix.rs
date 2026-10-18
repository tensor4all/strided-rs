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
    map_into, map_update_into, mul, reduce, reduce_axis_into, sum, symmetrize_into,
    with_execution_policy, zip_map2_into, zip_map3_into, zip_map4_into, zip_reduce,
    zip_update2_into, zip_update3_into, ExecutionPolicy, Identity, StridedArray, StridedView,
    StridedViewMut,
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

/// Steady-state allocations made by every thread while `run` executes.
///
/// The window runs three times and reports the minimum. A pool worker the
/// warm-up never scheduled pays one-time first-touch costs (thread-local
/// registration allocates on macOS) the first time it picks up a split, which
/// is not a per-call allocation; a per-call allocation shows up in every round.
/// (#321)
fn count_allocations(mut run: impl FnMut()) -> usize {
    (0..3)
        .map(|_| {
            let before = ALLOCATIONS.load(Ordering::SeqCst);
            run();
            ALLOCATIONS.load(Ordering::SeqCst) - before
        })
        .min()
        .unwrap_or(0)
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

    // Single- and two-operand full reductions (#280). `dot` above is the typed
    // two-operand entry; `zip_reduce` is the mapped one, and `reduce`/`sum` are
    // the one-operand entries.
    assert_entry("reduce", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let source = sv(&src, dims, &strides, 0).unwrap();
        let _ = reduce(&source, |x| x, |x, y| x + y, 0.0).unwrap();
    });

    assert_entry("sum", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let source = sv(&src, dims, &strides, 0).unwrap();
        let _ = sum(&source).unwrap();
    });

    assert_entry("zip_reduce", pool, dims, &kinds, |kind| {
        let strides = &layouts[kind_index(kind)];
        let va = sv(&a, dims, &strides, 0).unwrap();
        let vb = sv(&b, dims, &dst_strides, 0).unwrap();
        let _: f64 = zip_reduce(&va, &vb, |x, y| x * y, |x, y| x + y, 0.0).unwrap();
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
    check_parallel_threshold(&pool);
    check_full_reduction(&pool);
    check_plans(&pool);
    record_rank9_spill(&pool);
}

/// Tensor-sized cases above the 32768-element threading threshold, so the
/// two-worker counting windows exercise the real parallel branches rather than
/// the serial fallback that the `large` shapes (1025..1152 elements) take.
fn check_parallel_threshold(pool: &rayon::ThreadPool) {
    let shapes: [&[usize]; 3] = [
        &[33, 33, 33],
        &[3, 3, 3, 3, 3, 3, 3, 21],
        &[5, 4, 4, 4, 4, 4, 4, 2],
    ];
    for dims in shapes {
        assert!(
            total(dims) > 32_768,
            "threshold case {dims:?} is not tensor-sized"
        );
        run_map_entries(dims, pool);
    }
}

/// The recursive full-reduction fold (`reduce`, whose parallel path combines
/// per-worker partials in a tree) must also be allocation-free on every worker
/// for a non-contiguous source above the threading threshold.
fn check_full_reduction(pool: &rayon::ThreadPool) {
    use strided_basic::reduce;

    let dims = [257usize, 128];
    let strides = [1isize, 300];
    assert!(
        total(&dims) > 32_768,
        "full-reduction case is not tensor-sized"
    );
    let len = required_len(&dims, &strides);
    let src: Vec<f64> = (0..len).map(|v| v as f64).collect();
    // A gapped stride keeps the source non-contiguous so the recursive
    // partial-combination fold runs instead of the contiguous lane fold.
    let source = sv(&src, &dims, &strides, 0).unwrap();

    // Warm up under both execution modes outside the counted windows.
    with_execution_policy(ExecutionPolicy::Sequential, || {
        let _ = reduce(&source, |x| x, |a, b| a + b, 0.0).unwrap();
    });
    pool.install(|| {
        with_execution_policy(
            ExecutionPolicy::Rayon {
                max_threads: NonZeroUsize::new(2).unwrap(),
            },
            || {
                let _ = reduce(&source, |x| x, |a, b| a + b, 0.0).unwrap();
            },
        );
    });

    let serial = count_thread_allocations(|| {
        with_execution_policy(ExecutionPolicy::Sequential, || {
            let _ = reduce(&source, |x| x, |a, b| a + b, 0.0).unwrap();
        });
    });
    assert_eq!(serial, 0, "serial full reduction allocated {serial}");

    let parallel = pool.install(|| {
        count_allocations(|| {
            with_execution_policy(
                ExecutionPolicy::Rayon {
                    max_threads: NonZeroUsize::new(2).unwrap(),
                },
                || {
                    let _ = reduce(&source, |x| x, |a, b| a + b, 0.0).unwrap();
                },
            );
        })
    });
    assert_eq!(
        parallel, 0,
        "two-worker full reduction allocated {parallel}"
    );
}

/// Compiled plans must not allocate on compile or execute within the inline
/// rank/operand bounds (P4), so downstream code that compiles per call and
/// executes per accumulate does not touch the heap.
fn check_plans(pool: &rayon::ThreadPool) {
    use strided_basic::{
        ConcatenatePlan, ErasedConcatenatePlan, ErasedRawStridedMut, ErasedRawStridedPtr,
        ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedReducePlan, ErasedUpdatePlan,
        ExecContext, KernelDType, RawStridedMut, RawStridedRef, ReduceOp, UpdateOp, UpdatePlan,
    };
    for rank in 1..=8usize {
        let dims = vec![2usize; rank];
        let strides = col_major_strides(&dims).into_vec();
        let len = total(&dims);
        let src: Vec<f64> = (0..len).map(|v| v as f64 + 1.0).collect();
        let mut dst = vec![0.0f64; len];

        // UpdatePlan: compile and execute allocation-free.
        let plan = with_execution_policy(ExecutionPolicy::Sequential, || {
            UpdatePlan::compile(&dims, &strides, &strides).unwrap()
        });
        {
            let src_ref = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
            let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
            plan.execute(&mut dst_mut, &src_ref, 2.0, 3.0, |x| x)
                .unwrap();
        }
        let update_compile = count_thread_allocations(|| {
            let plan = UpdatePlan::compile(&dims, &strides, &strides).unwrap();
            std::hint::black_box(plan);
        });
        assert_eq!(
            update_compile, 0,
            "UpdatePlan::compile rank {rank} allocated {update_compile}"
        );
        let src_ref = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
        let update_execute = count_thread_allocations(|| {
            plan.execute(&mut dst_mut, &src_ref, 2.0, 3.0, |x| x)
                .unwrap();
        });
        assert_eq!(
            update_execute, 0,
            "UpdatePlan::execute rank {rank} allocated {update_execute}"
        );

        // Write-only uninitialized destination: same allocation-free replay.
        let mut uninit_dst = vec![MaybeUninit::<f64>::uninit(); len];
        let mut uninit_mut = RawStridedMut::new(&mut uninit_dst, &dims, &strides, 0).unwrap();
        plan.execute_uninit(&mut uninit_mut, &src_ref, 1.0, |x| x)
            .unwrap();
        let update_uninit = count_thread_allocations(|| {
            plan.execute_uninit(&mut uninit_mut, &src_ref, 1.0, |x| x)
                .unwrap();
        });
        assert_eq!(
            update_uninit, 0,
            "UpdatePlan::execute_uninit rank {rank} allocated {update_uninit}"
        );

        // Erased update plan: same traversal behind a dtype-erased boundary.
        let erased_update =
            ErasedUpdatePlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
        {
            let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
            let mut erased_dst = vec![0.0f64; len];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut erased_dst, &dims, &strides, 0).unwrap();
            erased_update
                .execute(
                    &ExecContext::serial(),
                    &mut dest,
                    &source,
                    2.0,
                    3.0,
                    UpdateOp::Identity,
                )
                .unwrap();
        }
        let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
        let mut erased_dst = vec![0.0f64; len];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut erased_dst, &dims, &strides, 0).unwrap();
        let erased_update_execute = count_thread_allocations(|| {
            erased_update
                .execute(
                    &ExecContext::serial(),
                    &mut dest,
                    &source,
                    2.0,
                    3.0,
                    UpdateOp::Identity,
                )
                .unwrap();
        });
        assert_eq!(
            erased_update_execute, 0,
            "ErasedUpdatePlan::execute rank {rank} allocated {erased_update_execute}"
        );

        // Erased write-only uninitialized destination.
        let mut erased_uninit = vec![MaybeUninit::<f64>::uninit(); len];
        let mut uninit_dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut erased_uninit, &dims, &strides, 0)
                .unwrap();
        let source_ptr = ErasedRawStridedPtr::from_ref(&source);
        erased_update
            .execute_uninit(
                &ExecContext::serial(),
                &mut uninit_dest,
                &source_ptr,
                1.0,
                UpdateOp::Identity,
            )
            .unwrap();
        let erased_update_uninit = count_thread_allocations(|| {
            erased_update
                .execute_uninit(
                    &ExecContext::serial(),
                    &mut uninit_dest,
                    &source_ptr,
                    1.0,
                    UpdateOp::Identity,
                )
                .unwrap();
        });
        assert_eq!(
            erased_update_uninit, 0,
            "ErasedUpdatePlan::execute_uninit rank {rank} allocated {erased_update_uninit}"
        );

        // ConcatenatePlan with two inputs: compile and execute allocation-free.
        let mut dest_dims = dims.clone();
        dest_dims[0] *= 2;
        let dest_strides = col_major_strides(&dest_dims).into_vec();
        let dest_len = total(&dest_dims);
        let mut concat_dst = vec![0.0f64; dest_len];
        let concat_compile = count_thread_allocations(|| {
            let plan = ConcatenatePlan::compile(
                &[&dims, &dims],
                &[&strides, &strides],
                &dest_dims,
                &dest_strides,
                0,
            )
            .unwrap();
            std::hint::black_box(plan);
        });
        assert_eq!(
            concat_compile, 0,
            "ConcatenatePlan::compile rank {rank} allocated {concat_compile}"
        );
        let concat_plan = ConcatenatePlan::compile(
            &[&dims, &dims],
            &[&strides, &strides],
            &dest_dims,
            &dest_strides,
            0,
        )
        .unwrap();
        {
            let a = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
            let b = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
            let mut dest =
                RawStridedMut::new(&mut concat_dst, &dest_dims, &dest_strides, 0).unwrap();
            concat_plan.execute(&mut dest, &[a, b]).unwrap();
        }
        let a = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        let b = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        let mut dest = RawStridedMut::new(&mut concat_dst, &dest_dims, &dest_strides, 0).unwrap();
        let concat_execute = count_thread_allocations(|| {
            concat_plan.execute(&mut dest, &[a, b]).unwrap();
        });
        assert_eq!(
            concat_execute, 0,
            "ConcatenatePlan::execute rank {rank} allocated {concat_execute}"
        );

        // Erased concatenate execute (typed scalar, erased dtype tag).
        let erased_plan = ErasedConcatenatePlan::compile(
            KernelDType::F64,
            &[&dims, &dims],
            &[&strides, &strides],
            &dest_dims,
            &dest_strides,
            0,
        )
        .unwrap();
        let mut erased_dst = vec![0.0f64; dest_len];
        {
            let a = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
            let b = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut erased_dst, &dest_dims, &dest_strides, 0)
                    .unwrap();
            erased_plan
                .execute(&strided_basic::ExecContext::serial(), &mut dest, &[a, b])
                .unwrap();
        }
        let a = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
        let b = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut erased_dst, &dest_dims, &dest_strides, 0)
                .unwrap();
        let erased_concat_execute = count_thread_allocations(|| {
            erased_plan
                .execute(&strided_basic::ExecContext::serial(), &mut dest, &[a, b])
                .unwrap();
        });
        assert_eq!(
            erased_concat_execute, 0,
            "ErasedConcatenatePlan::execute rank {rank} allocated {erased_concat_execute}"
        );

        // Erased reduction compile (full and one axis) allocation-free.
        let full_compile = count_thread_allocations(|| {
            let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides)
                .unwrap();
            std::hint::black_box(plan);
        });
        assert_eq!(
            full_compile, 0,
            "ErasedReducePlan::compile rank {rank} allocated {full_compile}"
        );
        let reduced_dims: Vec<usize> = dims[1..].to_vec();
        let reduced_strides = col_major_strides(&reduced_dims).into_vec();
        let axes_compile = count_thread_allocations(|| {
            let plan = ErasedReducePlan::compile_axes(
                KernelDType::F64,
                ReduceOp::Sum,
                &dims,
                &strides,
                &reduced_dims,
                &reduced_strides,
                &[0],
            )
            .unwrap();
            std::hint::black_box(plan);
        });
        assert_eq!(
            axes_compile, 0,
            "ErasedReducePlan::compile_axes rank {rank} allocated {axes_compile}"
        );
    }

    // Axis-reduction execute for each kernel strategy, allocation-free. The
    // `OutputBlocks` path exercises the per-thread reusable accumulator scratch.
    let dims = [16usize, 4];
    let strides = col_major_strides(&dims).into_vec();
    let src: Vec<f64> = (0..64).map(|v| v as f64).collect();
    for reduced_dims in [vec![4usize], vec![16usize]] {
        let reduce_axis = if reduced_dims[0] == 4 { 0 } else { 1 };
        let reduced_strides = col_major_strides(&reduced_dims).into_vec();
        let plan = ErasedReducePlan::compile_axes(
            KernelDType::F64,
            ReduceOp::Sum,
            &dims,
            &strides,
            &reduced_dims,
            &reduced_strides,
            &[reduce_axis],
        )
        .unwrap();
        let mut out = vec![0.0f64; reduced_dims.iter().product::<usize>()];
        {
            let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out, &reduced_dims, &reduced_strides, 0)
                    .unwrap();
            plan.execute(&strided_basic::ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
        let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut out, &reduced_dims, &reduced_strides, 0)
                .unwrap();
        let execute = count_thread_allocations(|| {
            plan.execute(&strided_basic::ExecContext::serial(), &mut dest, &source)
                .unwrap();
        });
        assert_eq!(
            execute, 0,
            "ErasedReducePlan::execute {reduced_dims:?} allocated {execute}"
        );
    }
    let full = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let mut scalar = [0.0f64; 1];
    {
        let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut scalar, &[1], &[1], 0).unwrap();
        full.execute(&strided_basic::ExecContext::serial(), &mut dest, &source)
            .unwrap();
    }
    let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut scalar, &[1], &[1], 0).unwrap();
    let full_execute = count_thread_allocations(|| {
        full.execute(&strided_basic::ExecContext::serial(), &mut dest, &source)
            .unwrap();
    });
    assert_eq!(
        full_execute, 0,
        "ErasedReducePlan full execute allocated {full_execute}"
    );
    let _ = pool;
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
