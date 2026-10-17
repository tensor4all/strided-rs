//! Regression coverage for tensor4all/strided-rs#305.
//!
//! A serial `ExecContext` used to skip the static specialization on the erased
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

use strided_basic::{
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    ExecContext, KernelDType,
};
use strided_fused::{ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};

struct CountingAllocator;

thread_local! {
    static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    // `try_with` fails only during thread teardown, when nothing is counted.
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

fn count_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get) - before
}

fn single_op(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op,
            inputs: vec![0, 1],
        }],
    }
}

fn add_then_multiply() -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![3],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0],
            },
        ],
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
    let serial = ExecContext::serial();

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
        plan.execute(&serial, &mut initialized_dest, &inputs)
            .unwrap();
        plan.execute_uninit(&serial, &mut uninitialized_dest, &input_ptrs)
            .unwrap();
    }

    let mut counts = [[0usize; 2]; 2];
    for (index, plan) in plans.iter().enumerate() {
        counts[index][0] = count_allocations(|| {
            plan.execute_uninit(&serial, &mut uninitialized_dest, &input_ptrs)
                .unwrap();
        });
        counts[index][1] = count_allocations(|| {
            plan.execute(&serial, &mut initialized_dest, &inputs)
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

/// A permuted operand still pays at most one wrapper allocation around the
/// inline planner (the interpreter register file that B4/
/// tensor4all/strided-rs#306 removes); before the inline planner this was 16
/// (8^3) and 43 (32^3), and the per-inner-block interpreter allocated 82 and
/// 1069 times. Debug builds can keep one lazy allocation that release drops, so
/// the guard is a small upper bound rather than an exact count.
#[test]
fn serial_replay_of_permuted_operands_allocates_like_the_typed_kernel() {
    for counts in [counted_allocations(8, true), counted_allocations(32, true)] {
        for row in counts {
            for allocated in row {
                assert!(
                    allocated <= 1,
                    "serial permuted replay regressed to {allocated} allocations"
                );
            }
        }
    }
}
