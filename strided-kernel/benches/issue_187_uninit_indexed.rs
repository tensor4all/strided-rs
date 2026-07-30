use core::mem::MaybeUninit;
use std::{
    env,
    hint::black_box,
    process,
    time::{Duration, Instant},
};
use strided_kernel::{
    ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedGatherPlan, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedReducePlan,
    ErasedScatterPlan, ExecContext, GatherSpec, KernelDType, ReduceOp, ScatterSpec,
};

fn bytes<T>(value: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(value.as_ptr().cast(), core::mem::size_of_val(value)) }
}
fn bytes_mut<T>(value: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(value.as_mut_ptr().cast(), core::mem::size_of_val(value))
    }
}

/// View aligned f64 storage as erased MaybeUninit bytes without initializing it.
///
/// # Safety
/// The cast preserves the allocation and byte length; callers retain the
/// exclusive borrow for the returned view.
unsafe fn uninit_f64_bytes(value: &mut [MaybeUninit<f64>]) -> &mut [MaybeUninit<u8>] {
    core::slice::from_raw_parts_mut(
        value.as_mut_ptr().cast::<MaybeUninit<u8>>(),
        core::mem::size_of_val(value),
    )
}

fn sample(
    ctx: &ExecContext,
    plan: &ErasedReducePlan,
    source: &ErasedRawStridedRef<'_>,
    init: &mut [f64],
    raw: &mut [MaybeUninit<f64>],
    initialized_first: bool,
) -> (Duration, Duration) {
    let initialized = || {
        let mut dest =
            ErasedRawStridedMut::new(KernelDType::F64, bytes_mut(init), &[], &[], 0).unwrap();
        let start = Instant::now();
        plan.execute(ctx, &mut dest, source).unwrap();
        black_box(init);
        start.elapsed()
    };
    let uninitialized = || {
        let mut dest = ErasedRawStridedUninitMut::new(
            KernelDType::F64,
            unsafe { uninit_f64_bytes(raw) },
            &[],
            &[],
            0,
        )
        .unwrap();
        let ptr = ErasedRawStridedPtr::from_ref(source);
        let start = Instant::now();
        plan.execute_uninit(ctx, &mut dest, &ptr).unwrap();
        black_box(raw);
        start.elapsed()
    };
    if initialized_first {
        (initialized(), uninitialized())
    } else {
        let uninit = uninitialized();
        let init = initialized();
        (init, uninit)
    }
}

fn report(label: &str, initialized: &[Duration], uninitialized: &[Duration]) -> bool {
    let mut logs = initialized
        .iter()
        .zip(uninitialized)
        .map(|(init, uninit)| (uninit.as_secs_f64() / init.as_secs_f64()).ln())
        .collect::<Vec<_>>();
    logs.sort_by(f64::total_cmp);
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    let variance = logs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (logs.len() - 1) as f64;
    let upper95 = (mean + 1.645 * variance.sqrt() / (logs.len() as f64).sqrt()).exp();
    let median = logs[logs.len() / 2].exp();
    println!(
        "{label}: median={median:.4} upper95={upper95:.4} n={}",
        logs.len()
    );
    upper95 <= 1.20
}

fn run_pairs(
    label: &str,
    mut initialized: impl FnMut() -> Duration,
    mut uninitialized: impl FnMut() -> Duration,
) -> bool {
    let mut init_times = Vec::with_capacity(31);
    let mut uninit_times = Vec::with_capacity(31);
    for sample in 0..31 {
        if sample % 2 == 0 {
            init_times.push(initialized());
            uninit_times.push(uninitialized());
        } else {
            let uninit = uninitialized();
            let init = initialized();
            init_times.push(init);
            uninit_times.push(uninit);
        }
    }
    report(label, &init_times, &uninit_times)
}

fn main() {
    let threads = env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(1);
    let ctx = ExecContext::max_threads(threads).expect("threads must be positive");
    let n = 131_073;
    let dims = [n];
    let strides = [1isize];
    let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25).collect();
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, bytes(&input), &dims, &strides, 0).unwrap();
    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let mut init = vec![0.0f64; 1];
    let mut raw = vec![MaybeUninit::<f64>::uninit(); 1];
    let mut initialized = Vec::with_capacity(31);
    let mut uninitialized = Vec::with_capacity(31);
    for sample_index in 0..31 {
        let (a, b) = sample(
            &ctx,
            &plan,
            &source,
            &mut init,
            &mut raw,
            sample_index % 2 == 0,
        );
        initialized.push(a);
        uninitialized.push(b);
    }
    let mut ok = report("reduce", &initialized, &uninitialized);

    let operand = (0..8192).map(|i| i as f64).collect::<Vec<_>>();
    let operand_dims = [8192usize];
    let index_dims = [2048usize];
    let dest_dims = [2048usize, 4];
    let indices = (0..2048).map(|i| (i * 3 % 8192) as i64).collect::<Vec<_>>();
    let gather = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &[1],
        &index_dims,
        &[1],
        &dest_dims,
        &[1, 2048],
        GatherSpec {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![4],
        },
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::new(KernelDType::F64, bytes(&operand), &operand_dims, &[1], 0)
            .unwrap();
    let index_ref =
        ErasedRawStridedRef::new(KernelDType::I64, bytes(&indices), &index_dims, &[1], 0).unwrap();
    let mut gather_init = vec![0.0f64; 8192];
    let mut gather_raw = vec![MaybeUninit::<f64>::uninit(); 8192];
    ok &= run_pairs(
        "gather_window",
        || {
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                bytes_mut(&mut gather_init),
                &dest_dims,
                &[1, 2048],
                0,
            )
            .unwrap();
            let start = Instant::now();
            gather
                .execute(&ctx, &mut dest, &operand_ref, &index_ref)
                .unwrap();
            black_box(&gather_init);
            start.elapsed()
        },
        || {
            let mut dest = ErasedRawStridedUninitMut::new(
                KernelDType::F64,
                unsafe { uninit_f64_bytes(&mut gather_raw) },
                &dest_dims,
                &[1, 2048],
                0,
            )
            .unwrap();
            let operand = ErasedRawStridedPtr::from_ref(&operand_ref);
            let index = ErasedRawStridedPtr::from_ref(&index_ref);
            let start = Instant::now();
            gather
                .execute_uninit(&ctx, &mut dest, &operand, &index)
                .unwrap();
            black_box(&gather_raw);
            start.elapsed()
        },
    );

    let starts = [128i64];
    let start_dims = [1usize];
    let update_dims = [4096usize];
    let update_values = (0..4096).map(|i| i as f64).collect::<Vec<_>>();
    let slice = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &[8192],
        &[1],
        &start_dims,
        &[1],
        &update_dims,
        &[1],
        &[4096],
    )
    .unwrap();
    let starts_ref =
        ErasedRawStridedRef::new(KernelDType::I64, bytes(&starts), &start_dims, &[1], 0).unwrap();
    let mut slice_init = vec![0.0f64; 4096];
    let mut slice_raw = vec![MaybeUninit::<f64>::uninit(); 4096];
    ok &= run_pairs(
        "dynamic_slice",
        || {
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                bytes_mut(&mut slice_init),
                &update_dims,
                &[1],
                0,
            )
            .unwrap();
            let start = Instant::now();
            slice
                .execute(&ctx, &mut dest, &operand_ref, &starts_ref)
                .unwrap();
            black_box(&slice_init);
            start.elapsed()
        },
        || {
            let mut dest = ErasedRawStridedUninitMut::new(
                KernelDType::F64,
                unsafe { uninit_f64_bytes(&mut slice_raw) },
                &update_dims,
                &[1],
                0,
            )
            .unwrap();
            let operand = ErasedRawStridedPtr::from_ref(&operand_ref);
            let starts = ErasedRawStridedPtr::from_ref(&starts_ref);
            let start = Instant::now();
            slice
                .execute_uninit(&ctx, &mut dest, &operand, &starts)
                .unwrap();
            black_box(&slice_raw);
            start.elapsed()
        },
    );

    let update_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        bytes(&update_values),
        &update_dims,
        &[1],
        0,
    )
    .unwrap();
    let update_plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &[8192],
        &[1],
        &start_dims,
        &[1],
        &update_dims,
        &[1],
        &[8192],
        &[1],
    )
    .unwrap();
    let mut update_init = vec![0.0f64; 8192];
    let mut update_raw = vec![MaybeUninit::<f64>::uninit(); 8192];
    ok &= run_pairs(
        "dynamic_update_slice",
        || {
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                bytes_mut(&mut update_init),
                &[8192],
                &[1],
                0,
            )
            .unwrap();
            let start = Instant::now();
            update_plan
                .execute(&ctx, &mut dest, &operand_ref, &update_ref, &starts_ref)
                .unwrap();
            black_box(&update_init);
            start.elapsed()
        },
        || {
            let mut dest = ErasedRawStridedUninitMut::new(
                KernelDType::F64,
                unsafe { uninit_f64_bytes(&mut update_raw) },
                &[8192],
                &[1],
                0,
            )
            .unwrap();
            let operand = ErasedRawStridedPtr::from_ref(&operand_ref);
            let update = ErasedRawStridedPtr::from_ref(&update_ref);
            let starts = ErasedRawStridedPtr::from_ref(&starts_ref);
            let start = Instant::now();
            update_plan
                .execute_uninit(&ctx, &mut dest, &operand, &update, &starts)
                .unwrap();
            black_box(&update_raw);
            start.elapsed()
        },
    );

    let scatter_indices = (0..4096).map(|i| (i % 8192) as i64).collect::<Vec<_>>();
    let scatter_updates = vec![1.0f64; 4096];
    let scatter_dims = [8192usize];
    let scatter_index_dims = [4096usize, 1];
    let scatter_update_dims = [4096usize];
    let scatter = ErasedScatterPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &scatter_dims,
        &[1],
        &scatter_index_dims,
        &[1, 4096],
        &scatter_update_dims,
        &[1],
        &scatter_dims,
        &[1],
        ScatterSpec {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
    )
    .unwrap();
    let scatter_index_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        bytes(&scatter_indices),
        &scatter_index_dims,
        &[1, 4096],
        0,
    )
    .unwrap();
    let scatter_update_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        bytes(&scatter_updates),
        &scatter_update_dims,
        &[1],
        0,
    )
    .unwrap();
    let mut scatter_init = vec![0.0f64; 8192];
    let mut scatter_raw = vec![MaybeUninit::<f64>::uninit(); 8192];
    let serial = ExecContext::serial();
    ok &= run_pairs(
        "scatter_serial",
        || {
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                bytes_mut(&mut scatter_init),
                &scatter_dims,
                &[1],
                0,
            )
            .unwrap();
            let start = Instant::now();
            scatter
                .execute(
                    &serial,
                    &mut dest,
                    &operand_ref,
                    &scatter_index_ref,
                    &scatter_update_ref,
                )
                .unwrap();
            black_box(&scatter_init);
            start.elapsed()
        },
        || {
            let mut dest = ErasedRawStridedUninitMut::new(
                KernelDType::F64,
                unsafe { uninit_f64_bytes(&mut scatter_raw) },
                &scatter_dims,
                &[1],
                0,
            )
            .unwrap();
            let operand = ErasedRawStridedPtr::from_ref(&operand_ref);
            let index = ErasedRawStridedPtr::from_ref(&scatter_index_ref);
            let update = ErasedRawStridedPtr::from_ref(&scatter_update_ref);
            let start = Instant::now();
            scatter
                .execute_uninit(&serial, &mut dest, &operand, &index, &update)
                .unwrap();
            black_box(&scatter_raw);
            start.elapsed()
        },
    );
    if !ok {
        process::exit(1);
    }
}
