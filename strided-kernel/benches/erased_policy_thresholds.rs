use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use strided_kernel::{
    col_major_strides, ErasedCopyPlan, ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan,
    ErasedGatherPlan, ErasedPadPlan, ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan,
    ErasedScatterPlan, ExecContext, GatherSpec, KernelDType, ReduceOp, ScatterSpec,
};

#[derive(Clone, Copy)]
struct BenchCase {
    label: &'static str,
    len: usize,
    threads: usize,
}

fn profile_cases() -> Vec<BenchCase> {
    let profile = std::env::var("STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE")
        .unwrap_or_else(|_| "quick".to_string());
    let default_threads = rayon::current_num_threads().min(2);
    let threads = std::env::var("STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_threads);
    let sizes: &[(&str, usize)] = match profile.as_str() {
        "smoke" => &[("small", 1 << 12)],
        "threshold" => &[
            ("small", 1 << 12),
            ("near_threshold", 1 << 15),
            ("medium", 1 << 18),
            ("large", 1 << 20),
        ],
        _ => &[("small", 1 << 12), ("near_threshold", 1 << 15)],
    };

    sizes
        .iter()
        .flat_map(|&(label, len)| {
            let serial = BenchCase {
                label,
                len,
                threads: 1,
            };
            if threads == 1 {
                vec![serial]
            } else {
                vec![
                    serial,
                    BenchCase {
                        label,
                        len,
                        threads,
                    },
                ]
            }
        })
        .collect()
}

fn context(case: BenchCase) -> ExecContext {
    if case.threads == 1 {
        ExecContext::serial()
    } else {
        ExecContext::max_threads(case.threads).unwrap()
    }
}

fn context_label(case: BenchCase) -> String {
    if case.threads == 1 {
        "serial".to_string()
    } else {
        format!("max_threads_{}", case.threads)
    }
}

fn bench_id(case: BenchCase) -> BenchmarkId {
    BenchmarkId::new(context_label(case), format!("{}_n{}", case.label, case.len))
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

fn patterned_f64(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| (index % 251) as f64 * 0.25 + 1.0)
        .collect()
}

fn patterned_i32(len: usize) -> Vec<i32> {
    (0..len).map(|index| (index % 97) as i32 - 31).collect()
}

fn bench_axis_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_axis_reduce_sum");
    for case in profile_cases() {
        let rows = 64usize;
        let columns = (case.len / rows).max(1);
        let src_dims = [rows, columns];
        let src_strides = col_major_strides(&src_dims);
        let dest_dims = [columns];
        let dest_strides = [1isize];
        let input = patterned_f64(rows * columns);
        let plan = ErasedReducePlan::compile_axes(
            KernelDType::F64,
            ReduceOp::Sum,
            &src_dims,
            &src_strides,
            &dest_dims,
            &dest_strides,
            &[0],
        )
        .unwrap();
        let source = ErasedRawStridedRef::new(
            KernelDType::F64,
            as_bytes(&input),
            &src_dims,
            &src_strides,
            0,
        )
        .unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; columns];
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                as_bytes_mut(&mut output),
                &dest_dims,
                &dest_strides,
                0,
            )
            .unwrap();
            bencher.iter(|| {
                plan.execute(&ctx, &mut dest, &source).unwrap();
                black_box(&mut dest);
            });
        });
    }
    group.finish();
}

fn bench_gather_take(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_gather_take");
    for case in profile_cases() {
        let operand_dims = [case.len];
        let operand_strides = [1isize];
        let index_dims = [case.len];
        let index_strides = [1isize];
        let dest_dims = [case.len];
        let dest_strides = [1isize];
        let operand = patterned_f64(case.len);
        let indices = (0..case.len)
            .map(|index| (case.len - 1 - index) as i64)
            .collect::<Vec<_>>();
        let spec = GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        };
        let plan = ErasedGatherPlan::compile(
            KernelDType::F64,
            KernelDType::I64,
            &operand_dims,
            &operand_strides,
            &index_dims,
            &index_strides,
            &dest_dims,
            &dest_strides,
            spec,
        )
        .unwrap();
        let operand_ref = ErasedRawStridedRef::new(
            KernelDType::F64,
            as_bytes(&operand),
            &operand_dims,
            &operand_strides,
            0,
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
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                as_bytes_mut(&mut output),
                &dest_dims,
                &dest_strides,
                0,
            )
            .unwrap();
            bencher.iter(|| {
                plan.execute(&ctx, &mut dest, &operand_ref, &index_ref)
                    .unwrap();
                black_box(&mut dest);
            });
        });
    }
    group.finish();
}

fn bench_dynamic_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_dynamic_slice");
    for case in profile_cases() {
        let operand_dims = [case.len + 128];
        let operand_strides = [1isize];
        let starts_dims = [1usize];
        let starts_strides = [1isize];
        let dest_dims = [case.len];
        let dest_strides = [1isize];
        let slice_sizes = [case.len];
        let operand = patterned_f64(case.len + 128);
        let starts = [64i64];
        let plan = ErasedDynamicSlicePlan::compile(
            KernelDType::F64,
            KernelDType::I64,
            &operand_dims,
            &operand_strides,
            &starts_dims,
            &starts_strides,
            &dest_dims,
            &dest_strides,
            &slice_sizes,
        )
        .unwrap();
        let operand_ref = ErasedRawStridedRef::new(
            KernelDType::F64,
            as_bytes(&operand),
            &operand_dims,
            &operand_strides,
            0,
        )
        .unwrap();
        let starts_ref = ErasedRawStridedRef::new(
            KernelDType::I64,
            as_bytes(&starts),
            &starts_dims,
            &starts_strides,
            0,
        )
        .unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                as_bytes_mut(&mut output),
                &dest_dims,
                &dest_strides,
                0,
            )
            .unwrap();
            bencher.iter(|| {
                plan.execute(&ctx, &mut dest, &operand_ref, &starts_ref)
                    .unwrap();
                black_box(&mut dest);
            });
        });
    }
    group.finish();
}

fn bench_dynamic_update_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_dynamic_update_slice");
    for case in profile_cases() {
        let operand_dims = [case.len + 128];
        let operand_strides = [1isize];
        let starts_dims = [1usize];
        let starts_strides = [1isize];
        let update_dims = [case.len];
        let update_strides = [1isize];
        let dest_dims = [case.len + 128];
        let dest_strides = [1isize];
        let operand = patterned_i32(case.len + 128);
        let update = patterned_i32(case.len);
        let starts = [64i32];
        let plan = ErasedDynamicUpdateSlicePlan::compile(
            KernelDType::I32,
            KernelDType::I32,
            &operand_dims,
            &operand_strides,
            &starts_dims,
            &starts_strides,
            &update_dims,
            &update_strides,
            &dest_dims,
            &dest_strides,
        )
        .unwrap();
        let operand_ref = ErasedRawStridedRef::new(
            KernelDType::I32,
            as_bytes(&operand),
            &operand_dims,
            &operand_strides,
            0,
        )
        .unwrap();
        let update_ref = ErasedRawStridedRef::new(
            KernelDType::I32,
            as_bytes(&update),
            &update_dims,
            &update_strides,
            0,
        )
        .unwrap();
        let starts_ref = ErasedRawStridedRef::new(
            KernelDType::I32,
            as_bytes(&starts),
            &starts_dims,
            &starts_strides,
            0,
        )
        .unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0i32; case.len + 128];
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::I32,
                as_bytes_mut(&mut output),
                &dest_dims,
                &dest_strides,
                0,
            )
            .unwrap();
            bencher.iter(|| {
                plan.execute(&ctx, &mut dest, &operand_ref, &update_ref, &starts_ref)
                    .unwrap();
                black_box(&mut dest);
            });
        });
    }
    group.finish();
}

fn bench_pad_fill_and_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_pad_fill_and_copy");
    for case in profile_cases() {
        let operand_len = (case.len / 2).max(1);
        let edge = (case.len - operand_len) / 2;
        let operand_dims = [operand_len];
        let operand_strides = [1isize];
        let dest_dims = [operand_len + edge * 2];
        let dest_strides = [1isize];
        let edge_low = [edge as i64];
        let edge_high = [edge as i64];
        let interior = [0i64];
        let fill = [-1i32];
        let operand = patterned_i32(operand_len);
        let plan = ErasedPadPlan::compile(
            KernelDType::I32,
            &operand_dims,
            &operand_strides,
            &dest_dims,
            &dest_strides,
            &edge_low,
            &edge_high,
            &interior,
        )
        .unwrap();
        let operand_ref = ErasedRawStridedRef::new(
            KernelDType::I32,
            as_bytes(&operand),
            &operand_dims,
            &operand_strides,
            0,
        )
        .unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0i32; dest_dims[0]];
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::I32,
                as_bytes_mut(&mut output),
                &dest_dims,
                &dest_strides,
                0,
            )
            .unwrap();
            bencher.iter(|| {
                plan.execute(&ctx, &mut dest, &operand_ref, as_bytes(&fill))
                    .unwrap();
                black_box(&mut dest);
            });
        });
    }
    group.finish();
}

fn bench_copy_raw_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_copy_raw_path");
    for case in profile_cases() {
        let dims = [case.len];
        let strides = [1isize];
        let source_data = patterned_f64(case.len);
        let plan = ErasedCopyPlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
        let source =
            ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&source_data), &dims, &strides, 0)
                .unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                as_bytes_mut(&mut output),
                &dims,
                &strides,
                0,
            )
            .unwrap();
            bencher.iter(|| {
                plan.execute(&ctx, &mut dest, &source).unwrap();
                black_box(&mut dest);
            });
        });
    }
    group.finish();
}

fn bench_scatter_additive(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_scatter_additive_serial_semantics");
    for case in profile_cases() {
        let operand_dims = [case.len];
        let operand_strides = [1isize];
        let index_dims = [case.len, 1];
        let index_strides = [1isize, case.len as isize];
        let update_dims = [case.len];
        let update_strides = [1isize];
        let dest_dims = [case.len];
        let dest_strides = [1isize];
        let operand = patterned_f64(case.len);
        let updates = patterned_f64(case.len);
        let indices = (0..case.len)
            .map(|index| (index % case.len.max(1)) as i64)
            .collect::<Vec<_>>();
        let spec = ScatterSpec {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        };
        let plan = ErasedScatterPlan::compile(
            KernelDType::F64,
            KernelDType::I64,
            &operand_dims,
            &operand_strides,
            &index_dims,
            &index_strides,
            &update_dims,
            &update_strides,
            &dest_dims,
            &dest_strides,
            spec,
        )
        .unwrap();
        let operand_ref = ErasedRawStridedRef::new(
            KernelDType::F64,
            as_bytes(&operand),
            &operand_dims,
            &operand_strides,
            0,
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
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest = ErasedRawStridedMut::new(
                KernelDType::F64,
                as_bytes_mut(&mut output),
                &dest_dims,
                &dest_strides,
                0,
            )
            .unwrap();
            bencher.iter(|| {
                plan.execute(&ctx, &mut dest, &operand_ref, &index_ref, &update_ref)
                    .unwrap();
                black_box(&mut dest);
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1));
    targets =
        bench_axis_reduce,
        bench_gather_take,
        bench_dynamic_slice,
        bench_dynamic_update_slice,
        bench_pad_fill_and_copy,
        bench_copy_raw_path,
        bench_scatter_additive
}
criterion_main!(benches);
