//! Asserts the zero-allocation contract of prepared erased/raw replay for ranks
//! at most `RAW_FUSED_RANK_LIMIT`.
//!
//! This lives in its own integration-test binary because it installs a
//! counting global allocator; keeping it isolated avoids counting noise from
//! unrelated tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex64;
use strided_kernel::{
    erased_map_into, erased_zip_into, map_into, zip_map2_into, CopyPlan, ErasedConcatenatePlan,
    ErasedCopyPlan, ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedFusedPlan,
    ErasedMapOp, ErasedPadPlan, ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef,
    ErasedRawStridedUninitMut, ErasedReducePlan, ErasedReversePlan, ErasedScatterPlan,
    ErasedSlicePlan, ErasedZipOp, ExecContext, FusedInst, FusedOp, FusedPlan, Identity,
    KernelDType, RawStridedMut, RawStridedRef, ReduceOp, ScatterSpec, StridedView, StridedViewMut,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            data.as_ptr().cast::<u8>(),
            data.len() * core::mem::size_of::<T>(),
        )
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations(run: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    run();
    ALLOCATIONS.load(Ordering::SeqCst) - before
}

fn assert_fused_uninit_allocation_parity(ctx: &ExecContext, fused_plan: FusedPlan) {
    let dims = [1usize << 16];
    let strides = [1isize];
    let lhs = vec![1.25f64; dims[0]];
    let rhs = vec![2.0f64; dims[0]];
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
    let inputs = [lhs_ref.clone(), rhs_ref.clone()];
    let input_ptrs = [
        ErasedRawStridedPtr::from_ref(&lhs_ref),
        ErasedRawStridedPtr::from_ref(&rhs_ref),
    ];
    let plan = ErasedFusedPlan::compile(KernelDType::F64, fused_plan).unwrap();
    let mut initialized = vec![0.0f64; dims[0]];
    let mut initialized_dest =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &strides, 0).unwrap();
    plan.execute(ctx, &mut initialized_dest, &inputs).unwrap();
    let initialized_allocations = count_allocations(|| {
        for _ in 0..8 {
            plan.execute(ctx, &mut initialized_dest, &inputs).unwrap();
        }
    });
    let mut uninitialized = vec![MaybeUninit::<f64>::uninit(); dims[0]];
    let mut uninitialized_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninitialized, &dims, &strides, 0)
            .unwrap();
    plan.execute_uninit(ctx, &mut uninitialized_dest, &input_ptrs)
        .unwrap();
    let uninitialized_allocations = count_allocations(|| {
        for _ in 0..8 {
            plan.execute_uninit(ctx, &mut uninitialized_dest, &input_ptrs)
                .unwrap();
        }
    });
    assert!(
        uninitialized_allocations <= initialized_allocations,
        "uninitialized={uninitialized_allocations}, initialized={initialized_allocations}"
    );
}

// One test function: the counter is process-global, so concurrently running
// sibling tests would leak their setup allocations into the counted window.
#[test]
fn execute_is_allocation_free_up_to_rank_limit() {
    #[cfg(feature = "parallel")]
    {
        let compile_allocations = count_allocations(|| {
            let plan = CopyPlan::compile(&[2, 2], &[1, 2], &[1, 2]).unwrap();
            std::hint::black_box(plan);
        });
        assert_eq!(
            compile_allocations, 0,
            "provably disjoint small CopyPlan compile must not allocate"
        );
    }

    // Rank 8 (the RAW_FUSED_RANK_LIMIT boundary), permuted strides so fusion
    // cannot collapse everything into one contiguous run.
    let dims = [2usize; 8];
    let dst_strides: Vec<isize> = (0..8).map(|axis| 1isize << axis).collect();
    let src_strides: Vec<isize> = (0..8).rev().map(|axis| 1isize << axis).collect();
    let src: Vec<f64> = (0..256).map(|value| value as f64).collect();
    let mut dst = vec![0.0f64; 256];

    let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();

    // Warm up (first call may fault pages but must not allocate either).
    plan.execute(&mut dest, &source).unwrap();

    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&mut dest, &source).unwrap();
            plan.execute_scale(&mut dest, &source, 2.0).unwrap();
        }
    });
    assert_eq!(allocations, 0, "execute/execute_scale must not allocate");

    let dims = [4usize, 4];
    let strides = [1isize, 4];
    let src: Vec<Complex64> = (0..16)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let mut dst = vec![Complex64::default(); 16];

    let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    plan.execute_conj(&mut dest, &source).unwrap();

    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute_conj(&mut dest, &source).unwrap();
        }
    });
    assert_eq!(allocations, 0, "execute_conj must not allocate");

    let dims = [2usize; 8];
    let dst_strides: Vec<isize> = (0..8).map(|axis| 1isize << axis).collect();
    let src_strides: Vec<isize> = (0..8).rev().map(|axis| 1isize << axis).collect();
    let src: Vec<f64> = (0..256).map(|value| value as f64).collect();
    let mut dst = vec![0.0f64; 256];

    let plan =
        ErasedCopyPlan::compile(KernelDType::F64, &dims, &dst_strides, &src_strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&src, &dims, &src_strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &dst_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased execute must not allocate");

    let dims = [256usize];
    let strides = [1isize];
    let lhs: Vec<f64> = (0..256).map(|value| value as f64).collect();
    let rhs: Vec<f64> = (0..256).map(|value| (value + 1) as f64).collect();
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
    let mut dst = vec![0.0f64; 256];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    let mut typed_dst = vec![0.0f64; 256];
    let lhs_view: StridedView<'_, f64, Identity> =
        StridedView::new(&lhs, &dims, &strides, 0).unwrap();
    let rhs_view: StridedView<'_, f64, Identity> =
        StridedView::new(&rhs, &dims, &strides, 0).unwrap();
    let mut typed_dest = StridedViewMut::new(&mut typed_dst, &dims, &strides, 0).unwrap();
    // Warm each exact counted sequence symmetrically so TLS/coverage runtime
    // initialization is outside both allocation windows.
    zip_map2_into(&mut typed_dest, &lhs_view, &rhs_view, |lhs, rhs| lhs + rhs).unwrap();
    map_into(&mut typed_dest, &lhs_view, |value| -value).unwrap();
    erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs_ref),
        &ErasedRawStridedPtr::from_ref(&rhs_ref),
    )
    .unwrap();
    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs_ref),
    )
    .unwrap();
    let kernel_allocations = count_allocations(|| {
        for _ in 0..16 {
            zip_map2_into(&mut typed_dest, &lhs_view, &rhs_view, |lhs, rhs| lhs + rhs).unwrap();
            map_into(&mut typed_dest, &lhs_view, |value| -value).unwrap();
        }
    });
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            erased_zip_into(
                KernelDType::F64,
                ErasedZipOp::Add,
                &ExecContext::serial(),
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&lhs_ref),
                &ErasedRawStridedPtr::from_ref(&rhs_ref),
            )
            .unwrap();
            erased_map_into(
                KernelDType::F64,
                ErasedMapOp::Negate,
                &ExecContext::serial(),
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&lhs_ref),
            )
            .unwrap();
        }
    });
    assert_eq!(
        allocations, kernel_allocations,
        "one-shot erased map/zip must not allocate beyond the typed kernels"
    );

    let add_mul = FusedPlan {
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
    };
    let fallback = FusedPlan {
        input_count: 2,
        outputs: vec![3],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![2],
            },
        ],
    };
    for ctx in [ExecContext::serial(), ExecContext::max_threads(4).unwrap()] {
        assert_fused_uninit_allocation_parity(&ctx, add_mul.clone());
        assert_fused_uninit_allocation_parity(&ctx, fallback.clone());
    }

    let dims = [3usize, 2];
    let input_strides = [1isize, 3];
    let output_strides = [2isize, 3];
    let lhs = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [6.0f64, 5.0, 4.0, 3.0, 2.0, 1.0];
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &input_strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &input_strides, 0).unwrap();
    let mut dst = [0.0f64; 8];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &output_strides, 0).unwrap();
    let reference_strides = [2isize, 7];
    let mut reference_dst = [0.0f64; 12];
    let mut reference_dest =
        ErasedRawStridedMut::from_slice_mut(&mut reference_dst, &dims, &reference_strides, 0)
            .unwrap();
    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &ExecContext::serial(),
        &mut reference_dest,
        &ErasedRawStridedPtr::from_ref(&lhs_ref),
    )
    .unwrap();
    erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &ExecContext::serial(),
        &mut reference_dest,
        &ErasedRawStridedPtr::from_ref(&lhs_ref),
        &ErasedRawStridedPtr::from_ref(&rhs_ref),
    )
    .unwrap();
    erased_map_into(
        KernelDType::F64,
        ErasedMapOp::Negate,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs_ref),
    )
    .unwrap();
    erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs_ref),
        &ErasedRawStridedPtr::from_ref(&rhs_ref),
    )
    .unwrap();
    let kernel_allocations = count_allocations(|| {
        erased_map_into(
            KernelDType::F64,
            ErasedMapOp::Negate,
            &ExecContext::serial(),
            &mut reference_dest,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
        )
        .unwrap();
        erased_zip_into(
            KernelDType::F64,
            ErasedZipOp::Add,
            &ExecContext::serial(),
            &mut reference_dest,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
            &ErasedRawStridedPtr::from_ref(&rhs_ref),
        )
        .unwrap();
    });
    let allocations = count_allocations(|| {
        erased_map_into(
            KernelDType::F64,
            ErasedMapOp::Negate,
            &ExecContext::serial(),
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
        )
        .unwrap();
        erased_zip_into(
            KernelDType::F64,
            ErasedZipOp::Add,
            &ExecContext::serial(),
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&lhs_ref),
            &ErasedRawStridedPtr::from_ref(&rhs_ref),
        )
        .unwrap();
    });
    assert_eq!(
        allocations, kernel_allocations,
        "validated one-shot raw replay must not repeat allocating injectivity checks"
    );

    let src_dims = [2usize; 8];
    let src_strides: Vec<isize> = (0..8).map(|axis| 1isize << axis).collect();
    let dest_dims = [2usize; 4];
    let dest_strides: Vec<isize> = (0..4).map(|axis| 1isize << axis).collect();
    let axes = [1usize, 3, 5, 7];
    let src: Vec<f64> = (0..256).map(|value| value as f64).collect();
    let mut dst = vec![0.0f64; 16];

    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &axes,
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&src, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased reduce-axis execute must not allocate"
    );

    let sum_squares_axis_plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::SumSquares,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &axes,
    )
    .unwrap();
    sum_squares_axis_plan
        .execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        sum_squares_axis_plan
            .execute(&ExecContext::serial(), &mut dest, &source)
            .unwrap();
    });
    assert_eq!(
        allocations, 0,
        "erased sum-squares axis execute must not allocate"
    );

    let sum_squares_dims = [4096usize];
    let sum_squares_strides = [1isize];
    let sum_squares_src = vec![1.25f64; sum_squares_dims[0]];
    let mut sum_squares_dst = [0.0f64];
    let sum_squares_plan = ErasedReducePlan::compile(
        KernelDType::F64,
        ReduceOp::SumSquares,
        &sum_squares_dims,
        &sum_squares_strides,
    )
    .unwrap();
    let sum_squares_source = ErasedRawStridedRef::from_slice(
        &sum_squares_src,
        &sum_squares_dims,
        &sum_squares_strides,
        0,
    )
    .unwrap();
    let mut sum_squares_dest =
        ErasedRawStridedMut::from_slice_mut(&mut sum_squares_dst, &[], &[], 0).unwrap();

    sum_squares_plan
        .execute(
            &ExecContext::serial(),
            &mut sum_squares_dest,
            &sum_squares_source,
        )
        .unwrap();
    let allocations = count_allocations(|| {
        sum_squares_plan
            .execute(
                &ExecContext::serial(),
                &mut sum_squares_dest,
                &sum_squares_source,
            )
            .unwrap();
    });
    assert_eq!(
        allocations, 0,
        "compact sum-squares replay must not allocate"
    );

    let src_dims = [2usize; 8];
    let src_strides: Vec<isize> = (0..8).map(|axis| 1isize << axis).collect();
    let start_dims = [8usize];
    let start_strides = [1isize];
    let slice_dims = [2usize; 8];
    let slice_strides: Vec<isize> = (0..8).map(|axis| 1isize << axis).collect();
    let starts = [0i64; 8];
    let src: Vec<f64> = (0..256).map(|value| value as f64).collect();
    let mut dst = vec![0.0f64; 256];
    let plan = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &src_dims,
        &src_strides,
        &start_dims,
        &start_strides,
        &slice_dims,
        &slice_strides,
        &slice_dims,
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&src, &src_dims, &src_strides, 0).unwrap();
    let starts_ref =
        ErasedRawStridedRef::from_slice(&starts, &start_dims, &start_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &slice_dims, &slice_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source, &starts_ref)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source, &starts_ref)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased dynamic-slice execute must not allocate"
    );

    let update_dims = [1usize; 8];
    let update_strides: Vec<isize> = (0..8).map(|_| 0isize).collect();
    let starts = [1i64; 8];
    let update = [7.0f64];
    let mut dst = vec![0.0f64; 256];
    let plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &src_dims,
        &src_strides,
        &start_dims,
        &start_strides,
        &update_dims,
        &update_strides,
        &src_dims,
        &src_strides,
    )
    .unwrap();
    let starts_ref =
        ErasedRawStridedRef::from_slice(&starts, &start_dims, &start_strides, 0).unwrap();
    let update_ref =
        ErasedRawStridedRef::from_slice(&update, &update_dims, &update_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &src_dims, &src_strides, 0).unwrap();

    plan.execute(
        &ExecContext::serial(),
        &mut dest,
        &source,
        &update_ref,
        &starts_ref,
    )
    .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(
                &ExecContext::serial(),
                &mut dest,
                &source,
                &update_ref,
                &starts_ref,
            )
            .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased dynamic-update-slice execute must not allocate"
    );

    let index_dims = [4usize, 8];
    let index_strides = [1isize, 4];
    let indices: Vec<i64> = (0..8)
        .flat_map(|axis| (0..4).map(move |batch| ((axis + batch) & 1) as i64))
        .collect();
    let update_dims = [4usize];
    let update_strides = [1isize];
    let updates = [1.0f64, 2.0, 3.0, 4.0];
    let mut dst = vec![0.0f64; 256];
    let plan = ErasedScatterPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &src_dims,
        &src_strides,
        &index_dims,
        &index_strides,
        &update_dims,
        &update_strides,
        &src_dims,
        &src_strides,
        ScatterSpec {
            update_window_dims: vec![],
            inserted_window_dims: (0..8).collect(),
            scatter_dims_to_operand_dims: (0..8).collect(),
            index_vector_dim: 1,
        },
    )
    .unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let update_ref =
        ErasedRawStridedRef::from_slice(&updates, &update_dims, &update_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &src_dims, &src_strides, 0).unwrap();

    plan.execute(
        &ExecContext::serial(),
        &mut dest,
        &source,
        &index_ref,
        &update_ref,
    )
    .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(
                &ExecContext::serial(),
                &mut dest,
                &source,
                &index_ref,
                &update_ref,
            )
            .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased scatter execute must not allocate");

    let starts_usize = [0usize; 8];
    let limits = [2usize; 8];
    let slice_steps = [1usize; 8];
    let mut dst = vec![0.0f64; 256];
    let plan = ErasedSlicePlan::compile(
        KernelDType::F64,
        &src_dims,
        &src_strides,
        &src_dims,
        &src_strides,
        &starts_usize,
        &limits,
        &slice_steps,
    )
    .unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &src_dims, &src_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased slice execute must not allocate");
    let mut uninit_dst = vec![MaybeUninit::<f64>::uninit(); 256];
    let mut uninit_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninit_dst, &src_dims, &src_strides, 0)
            .unwrap();
    let source_ptr = ErasedRawStridedPtr::from_ref(&source);
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute_uninit(&ExecContext::serial(), &mut uninit_dest, &source_ptr)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased uninitialized slice execute must not allocate"
    );

    let axes: Vec<usize> = (0..8).collect();
    let mut dst = vec![0.0f64; 256];
    let plan = ErasedReversePlan::compile(
        KernelDType::F64,
        &src_dims,
        &src_strides,
        &src_strides,
        &axes,
    )
    .unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &src_dims, &src_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased reverse execute must not allocate");
    let mut uninit_dst = vec![MaybeUninit::<f64>::uninit(); 256];
    let mut uninit_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninit_dst, &src_dims, &src_strides, 0)
            .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute_uninit(&ExecContext::serial(), &mut uninit_dest, &source_ptr)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased uninitialized reverse execute must not allocate"
    );

    let edge = [0i64; 8];
    let interior = [0i64; 8];
    let fill = [0.0f64];
    let mut dst = vec![0.0f64; 256];
    let plan = ErasedPadPlan::compile(
        KernelDType::F64,
        &src_dims,
        &src_strides,
        &src_dims,
        &src_strides,
        &edge,
        &edge,
        &interior,
    )
    .unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &src_dims, &src_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source, as_bytes(&fill))
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source, as_bytes(&fill))
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased pad execute must not allocate");
    let mut uninit_dst = vec![MaybeUninit::<f64>::uninit(); 256];
    let mut uninit_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninit_dst, &src_dims, &src_strides, 0)
            .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute_uninit(
                &ExecContext::serial(),
                &mut uninit_dest,
                &source_ptr,
                as_bytes(&fill),
            )
            .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased uninitialized pad execute must not allocate"
    );

    let mut left_dims = [2usize; 8];
    left_dims[0] = 1;
    let right_dims = left_dims;
    let left_strides = col_major_strides(&left_dims);
    let right_strides = left_strides.clone();
    let left: Vec<f64> = (0..128).map(|value| value as f64).collect();
    let right: Vec<f64> = (128..256).map(|value| value as f64).collect();
    let mut dst = vec![0.0f64; 256];
    let input_dims = [&left_dims[..], &right_dims[..]];
    let input_strides = [&left_strides[..], &right_strides[..]];
    let plan = ErasedConcatenatePlan::compile(
        KernelDType::F64,
        &input_dims,
        &input_strides,
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();
    let left_ref = ErasedRawStridedRef::from_slice(&left, &left_dims, &left_strides, 0).unwrap();
    let right_ref =
        ErasedRawStridedRef::from_slice(&right, &right_dims, &right_strides, 0).unwrap();
    let inputs = [left_ref, right_ref];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut dst, &src_dims, &src_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &inputs)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased concatenate execute must not allocate"
    );
    let input_ptrs = [
        ErasedRawStridedPtr::from_ref(&inputs[0]),
        ErasedRawStridedPtr::from_ref(&inputs[1]),
    ];
    let mut uninit_dst = vec![MaybeUninit::<f64>::uninit(); 256];
    let mut uninit_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninit_dst, &src_dims, &src_strides, 0)
            .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute_uninit(&ExecContext::serial(), &mut uninit_dest, &input_ptrs)
                .unwrap();
        }
    });
    assert_eq!(
        allocations, 0,
        "erased uninitialized concatenate execute must not allocate"
    );

    let dims = [32usize];
    let strides = [1isize];
    let lhs = [1.0f64; 32];
    let rhs = [2.0f64; 32];
    let inputs = [
        ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap(),
    ];
    let input_ptrs = inputs
        .each_ref()
        .map(|input| ErasedRawStridedPtr::from_ref(input));
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 2,
            outputs: vec![2],
            ops: vec![FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            }],
        },
    )
    .unwrap();
    let ctx = ExecContext::serial();

    let mut initialized = vec![0.0f64; 32];
    let mut initialized_dest =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &strides, 0).unwrap();
    plan.execute(&ctx, &mut initialized_dest, &inputs).unwrap();
    let initialized_allocations = count_allocations(|| {
        for _ in 0..8 {
            plan.execute(&ctx, &mut initialized_dest, &inputs).unwrap();
        }
    });

    let mut uninitialized = vec![MaybeUninit::<f64>::uninit(); 32];
    let mut uninitialized_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninitialized, &dims, &strides, 0)
            .unwrap();
    plan.execute_uninit(&ctx, &mut uninitialized_dest, &input_ptrs)
        .unwrap();
    let uninitialized_allocations = count_allocations(|| {
        for _ in 0..8 {
            plan.execute_uninit(&ctx, &mut uninitialized_dest, &input_ptrs)
                .unwrap();
        }
    });

    assert!(
        uninitialized_allocations <= initialized_allocations,
        "uninitialized replay allocated beyond initialized path: initialized={initialized_allocations}, uninitialized={uninitialized_allocations}"
    );
}

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut stride = 1isize;
    dims.iter()
        .map(|&dim| {
            let current = stride;
            stride *= dim as isize;
            current
        })
        .collect()
}
