use criterion::measurement::Measurement;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion,
};
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
        let rows = 2usize;
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
        let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; columns];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
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
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let index_ref =
            ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
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

#[derive(Clone, Copy)]
enum GatherVariant {
    Compact(usize),
    NonunitRank2,
    NegativeRank2,
}

impl GatherVariant {
    fn label(self) -> String {
        match self {
            Self::Compact(rank) => format!("compact_rank{rank}"),
            Self::NonunitRank2 => "rank2_nonunit_operand_index".to_string(),
            Self::NegativeRank2 => "rank2_negative_operand".to_string(),
        }
    }
}

fn varying_starts(batch: usize) -> Vec<i64> {
    (0..batch)
        .map(|index| ((index * 5 + 1) % batch) as i64)
        .collect()
}

fn bench_gather_variant<M: Measurement>(
    group: &mut BenchmarkGroup<'_, M>,
    case: BenchCase,
    variant: GatherVariant,
) {
    let (operand_dims, operand_strides, operand_len, operand_offset, index_strides, index_len) =
        match variant {
            GatherVariant::Compact(rank) => {
                let window = 1usize << (rank - 1);
                let batch = case.len / window;
                let mut dims = vec![2; rank - 1];
                dims.push(batch);
                let strides = col_major_strides(&dims);
                (dims, strides, case.len, 0, vec![1, batch as isize], batch)
            }
            GatherVariant::NonunitRank2 => {
                let batch = case.len / 2;
                (
                    vec![2, batch],
                    vec![2, 4],
                    4 * batch - 1,
                    0,
                    vec![2, (2 * batch) as isize],
                    2 * batch - 1,
                )
            }
            GatherVariant::NegativeRank2 => {
                let batch = case.len / 2;
                (
                    vec![2, batch],
                    vec![1, -2],
                    case.len,
                    2 * (batch - 1) as isize,
                    vec![1, batch as isize],
                    batch,
                )
            }
        };
    let batch = operand_dims[operand_dims.len() - 1];
    let index_dims = vec![batch, 1];
    let dest_dims = operand_dims.clone();
    let dest_strides = col_major_strides(&dest_dims);
    let indices = varying_starts(batch);
    let mut index_data = vec![0i64; index_len];
    for (index, &start) in indices.iter().enumerate() {
        index_data[index * (index_strides[0] as usize)] = start;
    }
    let operand = patterned_f64(operand_len);
    let spec = GatherSpec {
        offset_dims: (0..operand_dims.len() - 1).collect(),
        collapsed_slice_dims: vec![operand_dims.len() - 1],
        start_index_map: vec![operand_dims.len() - 1],
        index_vector_dim: 1,
        slice_sizes: operand_dims
            .iter()
            .enumerate()
            .map(|(axis, _)| if axis + 1 == operand_dims.len() { 1 } else { 2 })
            .collect(),
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
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, operand_offset)
            .unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&index_data, &index_dims, &index_strides, 0).unwrap();
    let mut output = vec![0.0f64; case.len];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();
    let ctx = context(case);
    let id = BenchmarkId::new(
        format!("{}_{}", variant.label(), context_label(case)),
        format!("{}_n{}", case.label, case.len),
    );

    group.bench_function(id, |bencher| {
        bencher.iter(|| {
            plan.execute(&ctx, &mut dest, &operand_ref, &index_ref)
                .unwrap();
            black_box(&mut dest);
        });
    });
}

fn bench_erased_gather_generic_rank_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_gather_generic_rank_layout");
    for case in profile_cases() {
        for variant in [
            GatherVariant::Compact(2),
            GatherVariant::Compact(4),
            GatherVariant::Compact(8),
            GatherVariant::NonunitRank2,
            GatherVariant::NegativeRank2,
        ] {
            bench_gather_variant(&mut group, case, variant);
        }
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
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let starts_ref =
            ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
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
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let update_ref =
            ErasedRawStridedRef::from_slice(&update, &update_dims, &update_strides, 0).unwrap();
        let starts_ref =
            ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0i32; case.len + 128];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
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
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0i32; dest_dims[0]];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
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
        let source = ErasedRawStridedRef::from_slice(&source_data, &dims, &strides, 0).unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dims, &strides, 0).unwrap();
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
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let index_ref =
            ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
        let update_ref =
            ErasedRawStridedRef::from_slice(&updates, &update_dims, &update_strides, 0).unwrap();
        let ctx = context(case);

        group.bench_function(bench_id(case), |bencher| {
            let mut output = vec![0.0f64; case.len];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
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
        bench_erased_gather_generic_rank_layout,
        bench_dynamic_slice,
        bench_dynamic_update_slice,
        bench_pad_fill_and_copy,
        bench_copy_raw_path,
        bench_scatter_additive
}
criterion_main!(benches);
