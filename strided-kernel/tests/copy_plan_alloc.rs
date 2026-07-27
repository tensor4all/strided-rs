//! Asserts the zero-allocation contract of prepared erased/raw replay for ranks
//! at most `RAW_FUSED_RANK_LIMIT`.
//!
//! This lives in its own integration-test binary because it installs a
//! counting global allocator; keeping it isolated avoids counting noise from
//! unrelated tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex64;
use strided_kernel::{
    CopyPlan, ErasedConcatenatePlan, ErasedCopyPlan, ErasedDynamicSlicePlan,
    ErasedDynamicUpdateSlicePlan, ErasedPadPlan, ErasedRawStridedMut, ErasedRawStridedRef,
    ErasedReducePlan, ErasedReversePlan, ErasedScatterPlan, ErasedSlicePlan, ExecContext,
    KernelDType, RawStridedMut, RawStridedRef, ReduceOp, ScatterSpec,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

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

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            data.as_ptr().cast::<u8>(),
            data.len() * core::mem::size_of::<T>(),
        )
    }
}

fn as_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(
            data.as_mut_ptr().cast::<u8>(),
            data.len() * core::mem::size_of::<T>(),
        )
    }
}

// One test function: the counter is process-global, so concurrently running
// sibling tests would leak their setup allocations into the counted window.
#[test]
fn execute_is_allocation_free_up_to_rank_limit() {
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
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&src), &dims, &src_strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &dims,
        &dst_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased execute must not allocate");

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
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&src), &src_dims, &src_strides, 0)
            .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

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
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&src), &src_dims, &src_strides, 0)
            .unwrap();
    let starts_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&starts),
        &start_dims,
        &start_strides,
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &slice_dims,
        &slice_strides,
        0,
    )
    .unwrap();

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
    let starts_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&starts),
        &start_dims,
        &start_strides,
        0,
    )
    .unwrap();
    let update_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&update),
        &update_dims,
        &update_strides,
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();

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
    let index_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&indices),
        &index_dims,
        &index_strides,
        0,
    )
    .unwrap();
    let update_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&updates),
        &update_dims,
        &update_strides,
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();

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
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased slice execute must not allocate");

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
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source)
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased reverse execute must not allocate");

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
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source, as_bytes(&fill))
        .unwrap();
    let allocations = count_allocations(|| {
        for _ in 0..16 {
            plan.execute(&ExecContext::serial(), &mut dest, &source, as_bytes(&fill))
                .unwrap();
        }
    });
    assert_eq!(allocations, 0, "erased pad execute must not allocate");

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
    let left_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&left),
        &left_dims,
        &left_strides,
        0,
    )
    .unwrap();
    let right_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&right),
        &right_dims,
        &right_strides,
        0,
    )
    .unwrap();
    let inputs = [left_ref, right_ref];
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();

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
