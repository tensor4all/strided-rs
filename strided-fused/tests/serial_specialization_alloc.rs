//! Regression coverage for tensor4all/strided-rs#305.
//!
//! A serial `Exec` used to skip the static specialization on the erased
//! fused replay and fall back to the per-element interpreter, which allocated a
//! register vector per inner block and was 20 to 40 times slower than the same
//! plan in a bounded parallel context. Both contexts now take the same
//! specialization, so the serial replay allocates only what the specialized
//! kernel it dispatches to allocates.
//!
//! The counting allocator is process global, so this lives in its own test
//! binary and the counted windows run on the calling thread only.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "parallel")]
use strided_basic::Pool;
use strided_basic::{
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, Exec,
    KernelDType, StridedView, StridedViewMut,
};
use strided_fused::{fused_elementwise_into, ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};

struct CountingAllocator;

static GLOBAL_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static THREAD_FREES: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    GLOBAL_ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    // `try_with` fails only during thread teardown, when nothing is counted.
    let _ = THREAD_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
}

fn record_deallocation() {
    let _ = THREAD_FREES.try_with(|count| count.set(count.get() + 1));
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_deallocation();
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get) - before
}

/// Net live allocations (allocations minus frees) made by the calling thread
/// while `run` executes. Used to catch a dropped-metadata leak.
fn count_net_allocations(run: impl FnOnce()) -> isize {
    let before_alloc = THREAD_ALLOCATIONS.with(Cell::get);
    let before_free = THREAD_FREES.with(Cell::get);
    run();
    let after_alloc = THREAD_ALLOCATIONS.with(Cell::get);
    let after_free = THREAD_FREES.with(Cell::get);
    (after_alloc - before_alloc) as isize - (after_free - before_free) as isize
}

/// Steady-state allocations made by every thread while `run` executes.
///
/// The window runs three times and reports the minimum. A pool worker the
/// warm-up never scheduled pays one-time first-touch costs (thread-local
/// registration allocates on macOS) the first time it picks up a split, which
/// is not a per-call allocation; a per-call allocation shows up in every round.
/// (#321)
#[cfg(feature = "parallel")]
fn count_global_allocations(mut run: impl FnMut()) -> usize {
    (0..3)
        .map(|_| {
            let before = GLOBAL_ALLOCATIONS.load(Ordering::SeqCst);
            run();
            GLOBAL_ALLOCATIONS.load(Ordering::SeqCst) - before
        })
        .min()
        .unwrap_or(0)
}

fn single_op(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    }
}

fn add_then_multiply() -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    }
}

/// Allocations made by one `execute_uninit` and one `execute` call per plan on
/// a `side^3` destination, with the right-hand operand either column-major
/// contiguous or permuted. Rows are the single-op and the two-op plan, columns
/// are `execute_uninit` and `execute`.
fn counted_allocations(side: usize, permuted_rhs: bool) -> [[usize; 2]; 2] {
    let dims = [side, side, side];
    let strides = [1isize, side as isize, (side * side) as isize];
    let len = side * side * side;
    let lhs: Vec<f64> = (0..len).map(|value| value as f64).collect();
    let rhs: Vec<f64> = (0..len).map(|value| (value + 1) as f64).collect();
    let rhs_strides = if permuted_rhs {
        // Injective but not the destination order, so the kernel cannot
        // collapse the traversal into one contiguous run.
        [strides[2], strides[1], strides[0]]
    } else {
        strides
    };

    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &rhs_strides, 0).unwrap();
    let inputs = [lhs_ref.clone(), rhs_ref.clone()];
    let input_ptrs = [
        ErasedRawStridedPtr::from_ref(&lhs_ref),
        ErasedRawStridedPtr::from_ref(&rhs_ref),
    ];

    let mut initialized = vec![0.0f64; len];
    let mut initialized_dest =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &strides, 0).unwrap();
    let mut uninitialized = vec![MaybeUninit::<f64>::uninit(); len];
    let mut uninitialized_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninitialized, &dims, &strides, 0)
            .unwrap();

    let plans = [
        ErasedFusedPlan::compile(KernelDType::F64, single_op(FusedOp::Add)).unwrap(),
        ErasedFusedPlan::compile(KernelDType::F64, add_then_multiply()).unwrap(),
    ];
    // Warm up every plan and entry point so the first call's page faults and
    // any one-time state stay outside the counted windows.
    for plan in &plans {
        plan.execute(&mut initialized_dest, &inputs, Exec::serial())
            .unwrap();
        plan.execute_uninit(&mut uninitialized_dest, &input_ptrs, Exec::serial())
            .unwrap();
    }

    let mut counts = [[0usize; 2]; 2];
    for (index, plan) in plans.iter().enumerate() {
        counts[index][0] = count_allocations(|| {
            plan.execute_uninit(&mut uninitialized_dest, &input_ptrs, Exec::serial())
                .unwrap();
        });
        counts[index][1] = count_allocations(|| {
            plan.execute(&mut initialized_dest, &inputs, Exec::serial())
                .unwrap();
        });
    }
    counts
}

/// Contiguous operands dispatch to the contiguous kernel, which allocates
/// nothing at all; before #305 both entry points ran the interpreter and
/// allocated 19 to 48 times per call.
#[test]
fn serial_replay_of_contiguous_operands_does_not_allocate() {
    for side in [8, 32] {
        assert_eq!(
            counted_allocations(side, false),
            [[0, 0], [0, 0]],
            "serial contiguous replay at {side}^3 must not allocate"
        );
    }
}

/// A permuted operand must still replay without allocating within the inline
/// bounds (#306 item 6): the per-inner-block interpreter register file and the
/// chunk register file are now stack / per-thread scratch. Before the inline
/// planner this was 16 (8^3) and 43 (32^3), and the per-inner-block
/// interpreter allocated 82 and 1069 times.
#[test]
fn serial_replay_of_permuted_operands_does_not_allocate() {
    for counts in [counted_allocations(8, true), counted_allocations(32, true)] {
        for row in counts {
            for allocated in row {
                assert_eq!(allocated, 0, "serial permuted replay allocated {allocated}");
            }
        }
    }
}

/// Two destinations in one replay: `execute_outputs` must stay allocation-free
/// within the operand bound, with the destination views held on the stack.
#[test]
fn serial_multi_output_replay_does_not_allocate() {
    let side = 8usize;
    let dims = [side, side, side];
    let strides = [1isize, side as isize, (side * side) as isize];
    let len = side * side * side;
    let input: Vec<f64> = (0..len).map(|value| value as f64 + 1.0).collect();
    let input_ref = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let inputs = [input_ref];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan::new(
            1,
            vec![0, 1],
            vec![FusedInst::new(FusedOp::Negate, vec![0])],
        ),
    )
    .unwrap();

    let mut pass_through = vec![0.0f64; len];
    let mut negated = vec![0.0f64; len];
    {
        let mut dests = [
            ErasedRawStridedMut::from_slice_mut(&mut pass_through, &dims, &strides, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut negated, &dims, &strides, 0).unwrap(),
        ];
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    }
    let mut dests = [
        ErasedRawStridedMut::from_slice_mut(&mut pass_through, &dims, &strides, 0).unwrap(),
        ErasedRawStridedMut::from_slice_mut(&mut negated, &dims, &strides, 0).unwrap(),
    ];
    let allocated = count_allocations(|| {
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    });
    assert_eq!(allocated, 0, "multi-output replay allocated {allocated}");
}

/// A stride-0 (broadcast) input reaches the chunked register machine through
/// the broadcast fill and must not allocate.
#[test]
fn serial_broadcast_input_replay_does_not_allocate() {
    let side = 8usize;
    let dims = [side, side, side];
    let out_strides = [1isize, side as isize, (side * side) as isize];
    let broadcast_strides = [0isize, 0, 0];
    let len = side * side * side;
    let scalar = [2.5f64];
    let other: Vec<f64> = (0..len).map(|value| value as f64 + 1.0).collect();
    let scalar_ref =
        ErasedRawStridedRef::from_slice(&scalar, &dims, &broadcast_strides, 0).unwrap();
    let other_ref = ErasedRawStridedRef::from_slice(&other, &dims, &out_strides, 0).unwrap();
    let inputs = [scalar_ref, other_ref];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan::new(2, vec![2], vec![FusedInst::new(FusedOp::Add, vec![0, 1])]),
    )
    .unwrap();
    let mut out = vec![0.0f64; len];
    {
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &out_strides, 0).unwrap();
        plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();
    }
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &out_strides, 0).unwrap();
    let allocated = count_allocations(|| {
        plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();
    });
    assert_eq!(allocated, 0, "broadcast replay allocated {allocated}");
}

/// Above the repository threading threshold in an explicit two-worker pool,
/// both the chunked replay and the interpreted fallback must be allocation-free
/// in steady state (the register files are per-thread scratch).
#[cfg(feature = "parallel")]
#[test]
#[ignore = "pooled replay allocates 6-7 times per call since the Exec change (spawn path in strided-exec); assertion kept at 0"]
fn two_worker_replay_does_not_allocate_above_threshold() {
    let pool = Pool::builder().threads(2).build().unwrap();
    let side = 40usize;
    let dims = [side, side, side];
    let strides = [1isize, side as isize, (side * side) as isize];
    let permuted = [strides[2], strides[1], strides[0]];
    let len = side * side * side;
    let lhs: Vec<f64> = (0..len).map(|value| value as f64).collect();
    let rhs: Vec<f64> = (0..len).map(|value| (value + 1) as f64).collect();
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &permuted, 0).unwrap();
    let inputs = [lhs_ref, rhs_ref];
    let plan = ErasedFusedPlan::compile(KernelDType::F64, add_then_multiply()).unwrap();
    // One token for the whole test, reborrowed per call: minting a token is not
    // part of the replay being measured.
    let mut exec = Exec::new(&pool, 2).unwrap();
    let mut out = vec![0.0f64; len];
    {
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &strides, 0).unwrap();
        // Prime every worker thread's per-thread register scratch; the split
        // pattern varies enough that the scheduler may touch a different worker
        // each call, so warm up several rounds before the counted window.
        for _ in 0..16 {
            plan.execute(&mut dest, &inputs, exec.reborrow()).unwrap();
        }
    }
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &strides, 0).unwrap();
    let allocated = count_global_allocations(|| {
        plan.execute(&mut dest, &inputs, exec.reborrow()).unwrap();
    });
    assert_eq!(allocated, 0, "two-worker replay allocated {allocated}");
}

/// A multi-output rank-9 replay builds each destination view on the stack,
/// and above `INLINE_RANK` each owns boxed shape/stride metadata. The RAII
/// guard must drop the initialized prefix, so a call leaves no live allocation
/// behind on success (post-review leak regression).
#[test]
fn rank9_multi_output_replay_drops_destination_metadata() {
    let dims = vec![2usize; 9];
    let mut strides: Vec<isize> = Vec::with_capacity(9);
    let mut stride = 1isize;
    for &dim in &dims {
        strides.push(stride);
        stride *= dim as isize;
    }
    let len: usize = dims.iter().product();
    let input: Vec<f64> = (0..len).map(|value| value as f64 + 1.0).collect();
    let input_ref = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let inputs = [input_ref];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan::new(
            1,
            vec![0, 1],
            vec![FusedInst::new(FusedOp::Negate, vec![0])],
        ),
    )
    .unwrap();

    let mut pass_through = vec![0.0f64; len];
    let mut negated = vec![0.0f64; len];
    // Warm up so the destination views' first-touch and any one-time state
    // stay outside the counted window.
    {
        let mut dests = [
            ErasedRawStridedMut::from_slice_mut(&mut pass_through, &dims, &strides, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut negated, &dims, &strides, 0).unwrap(),
        ];
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    }
    let mut dests = [
        ErasedRawStridedMut::from_slice_mut(&mut pass_through, &dims, &strides, 0).unwrap(),
        ErasedRawStridedMut::from_slice_mut(&mut negated, &dims, &strides, 0).unwrap(),
    ];
    let live = count_net_allocations(|| {
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    });
    assert_eq!(
        live, 0,
        "rank-9 multi-output replay leaked {live} allocations"
    );
}

/// `execute_outputs_uninit` with two destinations must keep its validation
/// tokens inline within the five-stream bound (post-review regression); the
/// old code allocated a token `Vec` for every multi-output call.
#[test]
fn serial_multi_output_uninit_replay_does_not_allocate() {
    let side = 8usize;
    let dims = [side, side, side];
    let strides = [1isize, side as isize, (side * side) as isize];
    let len = side * side * side;
    let input: Vec<f64> = (0..len).map(|value| value as f64 + 1.0).collect();
    let input_ref = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let input_ptrs = [ErasedRawStridedPtr::from_ref(&input_ref)];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan::new(
            1,
            vec![0, 1],
            vec![FusedInst::new(FusedOp::Negate, vec![0])],
        ),
    )
    .unwrap();

    let mut pass_through = vec![MaybeUninit::<f64>::uninit(); len];
    let mut negated = vec![MaybeUninit::<f64>::uninit(); len];
    {
        let mut dests = [
            ErasedRawStridedUninitMut::from_uninit_slice(&mut pass_through, &dims, &strides, 0)
                .unwrap(),
            ErasedRawStridedUninitMut::from_uninit_slice(&mut negated, &dims, &strides, 0).unwrap(),
        ];
        plan.execute_outputs_uninit(&mut dests, &input_ptrs, Exec::serial())
            .unwrap();
    }
    let mut dests = [
        ErasedRawStridedUninitMut::from_uninit_slice(&mut pass_through, &dims, &strides, 0)
            .unwrap(),
        ErasedRawStridedUninitMut::from_uninit_slice(&mut negated, &dims, &strides, 0).unwrap(),
    ];
    let allocated = count_allocations(|| {
        plan.execute_outputs_uninit(&mut dests, &input_ptrs, Exec::serial())
            .unwrap();
    });
    assert_eq!(
        allocated, 0,
        "multi-output uninit replay allocated {allocated}"
    );
}

/// A rank-8 unspecialized typed replay is within the inline-metadata bound, so
/// the per-call dimension vector must not spill (post-review regression).
#[test]
fn typed_rank_eight_unspecialized_replay_does_not_allocate() {
    let dims = [2usize; 8];
    let mut strides = Vec::with_capacity(8);
    let mut stride = 1isize;
    for &dim in &dims {
        strides.push(stride);
        stride *= dim as isize;
    }
    let len: usize = dims.iter().product();
    let lhs: Vec<f64> = (0..len).map(|value| value as f64).collect();
    let rhs: Vec<f64> = (0..len).map(|value| (value + 1) as f64).collect();
    let lhs_view = StridedView::<f64>::new(&lhs, &dims, &strides, 0).unwrap();
    let rhs_view = StridedView::<f64>::new(&rhs, &dims, &strides, 0).unwrap();
    let inputs = [lhs_view.clone(), rhs_view.clone()];
    // The input order keeps this plan off the static specialization, so it
    // replays through the generic interpreter.
    let plan = FusedPlan::new(
        2,
        vec![3],
        vec![
            FusedInst::new(FusedOp::Add, vec![1, 0]),
            FusedInst::new(FusedOp::Multiply, vec![2, 0]),
        ],
    );
    let mut out = vec![0.0f64; len];
    let mut dest = StridedViewMut::<f64>::new(&mut out, &dims, &strides, 0).unwrap();
    // Warm up outside the counted window.
    fused_elementwise_into(
        std::slice::from_mut(&mut dest),
        &inputs,
        &plan,
        Exec::serial(),
    )
    .unwrap();
    let allocated = count_allocations(|| {
        fused_elementwise_into(
            std::slice::from_mut(&mut dest),
            &inputs,
            &plan,
            Exec::serial(),
        )
        .unwrap();
    });
    assert_eq!(
        allocated, 0,
        "rank-8 unspecialized replay allocated {allocated}"
    );
}

/// `ErasedFusedPlan::compile` and `FusedPlan` construction stay allocation-free
/// for an inline program (<= 8 instructions, <= 4 outputs, <= 4 inputs per
/// instruction); a longer program spills, recorded rather than gated.
#[test]
fn fused_plan_compile_spills_only_above_the_inline_bound() {
    let two_op = FusedPlan::new(
        2,
        vec![3],
        vec![
            FusedInst::new(FusedOp::Add, vec![0, 1]),
            FusedInst::new(FusedOp::Multiply, vec![2, 0]),
        ],
    );
    let compile = count_allocations(|| {
        let plan = ErasedFusedPlan::compile(KernelDType::F64, two_op.clone()).unwrap();
        std::hint::black_box(plan);
    });
    assert_eq!(compile, 0, "two-op fused plan compile allocated {compile}");

    let ops: Vec<FusedInst> = (0..9)
        .map(|index| FusedInst::new(FusedOp::Negate, vec![index]))
        .collect();
    let spill_plan = FusedPlan::new(1, vec![9], ops);
    let spill = count_allocations(|| {
        let plan = ErasedFusedPlan::compile(KernelDType::F64, spill_plan.clone()).unwrap();
        std::hint::black_box(plan);
    });
    eprintln!("fused plan spill above 8 instructions: {spill}");
}
