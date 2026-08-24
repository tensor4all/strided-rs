use criterion::measurement::Measurement;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion,
};
use std::time::Duration;
use strided_kernel::{
    col_major_strides, erased_zip_into, ErasedCopyPlan, ErasedDynamicSlicePlan,
    ErasedDynamicUpdateSlicePlan, ErasedGatherPlan, ErasedPadPlan, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedReducePlan, ErasedScatterPlan, ErasedZipOp,
    ExecContext, GatherSpec, KernelDType, RawStridedRef, ReduceOp, ScatterSpec, StridedError,
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

fn nonzero_i32(len: usize) -> Vec<i32> {
    (0..len).map(|index| 1 + (index % 97) as i32).collect()
}

struct AnyLayout {
    label: String,
    dims: Vec<usize>,
    strides: Vec<isize>,
    data: Vec<i32>,
    offset: isize,
}

fn any_layouts(len: usize) -> Vec<AnyLayout> {
    let mut layouts = Vec::new();
    for rank in [1usize, 2, 4, 8] {
        let mut dims = vec![2usize; rank.saturating_sub(1)];
        dims.push(len >> rank.saturating_sub(1));
        layouts.push(AnyLayout {
            label: format!("compact_rank{rank}"),
            strides: col_major_strides(&dims),
            dims,
            data: nonzero_i32(len),
            offset: 0,
        });
    }
    layouts.push(AnyLayout {
        label: "rank2_negative".to_string(),
        dims: vec![2, len / 2],
        strides: vec![-1, 2],
        data: nonzero_i32(len),
        offset: 1,
    });
    layouts.push(AnyLayout {
        label: "rank2_nonunit".to_string(),
        dims: vec![2, len / 2],
        strides: vec![2, 4],
        data: nonzero_i32(2 * len),
        offset: 0,
    });
    layouts
}

fn current_any_scan(src: &RawStridedRef<'_, i32>) -> Result<bool, StridedError> {
    let total = src
        .dims()
        .iter()
        .try_fold(1usize, |total, &dim| total.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)?;
    for linear in 0..total {
        let mut remainder = linear;
        let mut offset = src.offset();
        for (&dim, &stride) in src.dims().iter().zip(src.strides()) {
            let index = remainder % dim;
            remainder /= dim;
            offset = offset
                .checked_add(
                    stride
                        .checked_mul(index as isize)
                        .ok_or(StridedError::OffsetOverflow)?,
                )
                .ok_or(StridedError::OffsetOverflow)?;
        }
        if unsafe { *src.data().as_ptr().offset(offset) } == 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn incremental_any_scan(src: &RawStridedRef<'_, i32>) -> Result<bool, StridedError> {
    let total = src
        .dims()
        .iter()
        .try_fold(1usize, |total, &dim| total.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)?;
    assert!(src.dims().len() <= 8);
    let mut coords = [0usize; 8];
    let mut offset = src.offset();
    for _ in 0..total {
        if unsafe { *src.data().as_ptr().offset(offset) } == 0 {
            return Ok(true);
        }
        for axis in 0..src.dims().len() {
            let next = coords[axis] + 1;
            if next < src.dims()[axis] {
                coords[axis] = next;
                offset = offset
                    .checked_add(src.strides()[axis])
                    .ok_or(StridedError::OffsetOverflow)?;
                break;
            }
            coords[axis] = 0;
            let reset = src.strides()[axis]
                .checked_mul((src.dims()[axis] - 1) as isize)
                .and_then(isize::checked_neg)
                .ok_or(StridedError::OffsetOverflow)?;
            offset = offset
                .checked_add(reset)
                .ok_or(StridedError::OffsetOverflow)?;
        }
    }
    Ok(false)
}

fn bench_raw_any_integer_preflight(c: &mut Criterion) {
    let cases = profile_cases();
    let mut scan_group = c.benchmark_group("erased_raw_any_scan");
    for case in cases.iter().copied().filter(|case| case.threads == 1) {
        for layout in any_layouts(case.len) {
            let src =
                RawStridedRef::new(&layout.data, &layout.dims, &layout.strides, layout.offset)
                    .unwrap();
            assert_eq!(current_any_scan(&src), incremental_any_scan(&src));
            for (algorithm, scan) in [
                (
                    "current_scan",
                    current_any_scan as fn(&RawStridedRef<'_, i32>) -> Result<bool, StridedError>,
                ),
                (
                    "incremental_scan",
                    incremental_any_scan
                        as fn(&RawStridedRef<'_, i32>) -> Result<bool, StridedError>,
                ),
            ] {
                scan_group.bench_function(
                    BenchmarkId::new(
                        format!("{}_{}", layout.label, algorithm),
                        format!("{}_n{}", case.label, case.len),
                    ),
                    |bencher| {
                        bencher.iter(|| black_box(scan(black_box(&src)).unwrap()));
                    },
                );
            }
        }
    }
    scan_group.finish();

    let mut public_group = c.benchmark_group("erased_integer_zip_preflight");
    for case in cases {
        for rank in [1usize, 8] {
            let layout = any_layouts(case.len)
                .into_iter()
                .find(|layout| layout.label == format!("compact_rank{rank}"))
                .unwrap();
            let lhs_data = nonzero_i32(case.len);
            let lhs = ErasedRawStridedRef::from_slice(&lhs_data, &layout.dims, &layout.strides, 0)
                .unwrap();
            let rhs = ErasedRawStridedRef::from_slice(
                &layout.data,
                &layout.dims,
                &layout.strides,
                layout.offset,
            )
            .unwrap();
            let lhs_ptr = ErasedRawStridedPtr::from_ref(&lhs);
            let rhs_ptr = ErasedRawStridedPtr::from_ref(&rhs);
            let ctx = context(case);
            for op in [ErasedZipOp::Add, ErasedZipOp::Divide] {
                public_group.bench_function(
                    BenchmarkId::new(
                        format!("compact_rank{rank}_{op:?}_{}", context_label(case)),
                        format!("{}_n{}", case.label, case.len),
                    ),
                    |bencher| {
                        let mut output = vec![0i32; case.len];
                        let mut dest = ErasedRawStridedMut::from_slice_mut(
                            &mut output,
                            &layout.dims,
                            &layout.strides,
                            0,
                        )
                        .unwrap();
                        bencher.iter(|| {
                            erased_zip_into(
                                KernelDType::I32,
                                op,
                                &ctx,
                                &mut dest,
                                &lhs_ptr,
                                &rhs_ptr,
                            )
                            .unwrap();
                            black_box(&mut dest);
                        });
                    },
                );
            }
        }
    }
    public_group.finish();
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

#[derive(Clone, Copy)]
enum ReduceLayoutVariant {
    CompactSingle(usize),
    CompactMulti(usize),
    NonunitRank4,
    NegativeRank4,
}

impl ReduceLayoutVariant {
    fn label(self) -> String {
        match self {
            Self::CompactSingle(rank) => format!("compact_single_rank{rank}"),
            Self::CompactMulti(rank) => format!("compact_multi_rank{rank}"),
            Self::NonunitRank4 => "rank4_nonunit_source".to_string(),
            Self::NegativeRank4 => "rank4_negative_source".to_string(),
        }
    }
}

fn bench_erased_reduce_variant<M: Measurement>(
    group: &mut BenchmarkGroup<'_, M>,
    case: BenchCase,
    variant: ReduceLayoutVariant,
) {
    let (rank, reduce_axes) = match variant {
        ReduceLayoutVariant::CompactSingle(rank) => (rank, vec![0]),
        ReduceLayoutVariant::CompactMulti(rank) => (rank, (0..rank - 1).collect()),
        ReduceLayoutVariant::NonunitRank4 | ReduceLayoutVariant::NegativeRank4 => (4, vec![0]),
    };
    let mut src_dims = vec![2; rank - 1];
    src_dims.push(case.len / (1usize << (rank - 1)));
    let src_strides = match variant {
        ReduceLayoutVariant::CompactSingle(_) | ReduceLayoutVariant::CompactMulti(_) => {
            col_major_strides(&src_dims)
        }
        ReduceLayoutVariant::NonunitRank4 => vec![2, 4, 8, 16],
        ReduceLayoutVariant::NegativeRank4 => vec![1, 2, 4, -8],
    };
    let kept_axes: Vec<_> = (0..rank)
        .filter(|axis| !reduce_axes.contains(axis))
        .collect();
    let dest_dims: Vec<_> = kept_axes.iter().map(|&axis| src_dims[axis]).collect();
    let dest_strides = col_major_strides(&dest_dims);
    let source_offset = match variant {
        ReduceLayoutVariant::NegativeRank4 => 8 * (src_dims[3] - 1) as isize,
        _ => 0,
    };
    let source_len = match variant {
        ReduceLayoutVariant::CompactSingle(_) | ReduceLayoutVariant::CompactMulti(_) => case.len,
        ReduceLayoutVariant::NonunitRank4 => {
            src_dims
                .iter()
                .zip(src_strides.iter())
                .map(|(&dim, &stride)| (dim - 1) * stride as usize)
                .sum::<usize>()
                + 1
        }
        ReduceLayoutVariant::NegativeRank4 => case.len,
    };
    let input = patterned_f64(source_len);
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &reduce_axes,
    )
    .unwrap();
    let source =
        ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, source_offset).unwrap();
    let mut output = vec![0.0f64; dest_dims.iter().product()];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();
    let ctx = context(case);
    let id = BenchmarkId::new(
        format!("{}_{}", variant.label(), context_label(case)),
        format!("{}_n{}", case.label, case.len),
    );

    group.bench_function(id, |bencher| {
        bencher.iter(|| {
            plan.execute(&ctx, &mut dest, &source).unwrap();
            black_box(&mut dest);
        });
    });
}

fn bench_erased_axis_reduce_generic_rank_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_axis_reduce_generic_rank_layout");
    for case in profile_cases() {
        for variant in [
            ReduceLayoutVariant::CompactSingle(2),
            ReduceLayoutVariant::CompactSingle(4),
            ReduceLayoutVariant::CompactSingle(8),
            ReduceLayoutVariant::CompactMulti(4),
            ReduceLayoutVariant::CompactMulti(8),
            ReduceLayoutVariant::NonunitRank4,
            ReduceLayoutVariant::NegativeRank4,
        ] {
            bench_erased_reduce_variant(&mut group, case, variant);
        }
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
enum LayoutVariant {
    Compact(usize),
    NonunitRank2,
    NegativeRank2,
}

impl LayoutVariant {
    fn label(self) -> String {
        match self {
            Self::Compact(rank) => format!("compact_rank{rank}"),
            Self::NonunitRank2 => "rank2_nonunit_source".to_string(),
            Self::NegativeRank2 => "rank2_negative_source".to_string(),
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
    variant: LayoutVariant,
) {
    let (operand_dims, operand_strides, operand_len, operand_offset, index_strides, index_len) =
        match variant {
            LayoutVariant::Compact(rank) => {
                let window = 1usize << (rank - 1);
                let batch = case.len / window;
                let mut dims = vec![2; rank - 1];
                dims.push(batch);
                let strides = col_major_strides(&dims);
                (dims, strides, case.len, 0, vec![1, batch as isize], batch)
            }
            LayoutVariant::NonunitRank2 => {
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
            LayoutVariant::NegativeRank2 => {
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
            LayoutVariant::Compact(2),
            LayoutVariant::Compact(4),
            LayoutVariant::Compact(8),
            LayoutVariant::NonunitRank2,
            LayoutVariant::NegativeRank2,
        ] {
            bench_gather_variant(&mut group, case, variant);
        }
    }
    group.finish();
}

fn bench_dynamic_slice_variant<M: Measurement>(
    group: &mut BenchmarkGroup<'_, M>,
    case: BenchCase,
    variant: LayoutVariant,
) {
    let rank = match variant {
        LayoutVariant::Compact(rank) => rank,
        LayoutVariant::NonunitRank2 | LayoutVariant::NegativeRank2 => 2,
    };
    let batch = case.len / (1usize << (rank - 1));
    let mut window_dims = vec![2; rank - 1];
    window_dims.push(batch);
    let mut operand_dims = window_dims.clone();
    *operand_dims.last_mut().unwrap() += 128;
    let (operand_strides, operand_len, operand_offset) = match variant {
        LayoutVariant::Compact(_) => (
            col_major_strides(&operand_dims),
            operand_dims.iter().product(),
            0,
        ),
        LayoutVariant::NonunitRank2 => (vec![2, 4], 4 * (batch + 128) - 1, 0),
        LayoutVariant::NegativeRank2 => (
            vec![1, -2],
            2 * (batch + 127) + 2,
            2 * (batch + 127) as isize,
        ),
    };
    let dest_strides = col_major_strides(&window_dims);
    let starts_dims = vec![rank];
    let starts_strides = vec![1isize];
    let mut starts = vec![0i64; rank];
    *starts.last_mut().unwrap() = 64;
    let operand = patterned_f64(operand_len);
    let plan = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &window_dims,
        &dest_strides,
        &window_dims,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, operand_offset)
            .unwrap();
    let starts_ref =
        ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
    let mut output = vec![0.0f64; case.len];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &window_dims, &dest_strides, 0).unwrap();
    let ctx = context(case);
    let id = BenchmarkId::new(
        format!("{}_{}", variant.label(), context_label(case)),
        format!("{}_n{}", case.label, case.len),
    );
    group.bench_function(id, |bencher| {
        bencher.iter(|| {
            plan.execute(&ctx, &mut dest, &operand_ref, &starts_ref)
                .unwrap();
            black_box(&mut dest);
        });
    });
}

fn bench_erased_dynamic_slice_generic_rank_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_dynamic_slice_generic_rank_layout");
    for case in profile_cases() {
        for variant in [
            LayoutVariant::Compact(2),
            LayoutVariant::Compact(4),
            LayoutVariant::Compact(8),
            LayoutVariant::NonunitRank2,
            LayoutVariant::NegativeRank2,
        ] {
            bench_dynamic_slice_variant(&mut group, case, variant);
        }
    }
    group.finish();
}

fn bench_dynamic_update_variant<M: Measurement>(
    group: &mut BenchmarkGroup<'_, M>,
    case: BenchCase,
    variant: LayoutVariant,
) {
    let rank = match variant {
        LayoutVariant::Compact(rank) => rank,
        LayoutVariant::NonunitRank2 | LayoutVariant::NegativeRank2 => 2,
    };
    let batch = case.len / (1usize << (rank - 1));
    let mut update_dims = vec![2; rank - 1];
    update_dims.push(batch);
    let mut padded_dims = update_dims.clone();
    *padded_dims.last_mut().unwrap() += 128;
    let operand_strides = col_major_strides(&padded_dims);
    let update_strides = match variant {
        LayoutVariant::Compact(_) => col_major_strides(&update_dims),
        LayoutVariant::NonunitRank2 => vec![2, 4],
        LayoutVariant::NegativeRank2 => vec![1, -2],
    };
    let (update_len, update_offset) = match variant {
        LayoutVariant::Compact(_) => (case.len, 0),
        LayoutVariant::NonunitRank2 => (4 * batch - 1, 0),
        LayoutVariant::NegativeRank2 => (2 * (batch - 1) + 2, 2 * (batch - 1) as isize),
    };
    let starts_dims = vec![rank];
    let starts_strides = vec![1isize];
    let mut starts = vec![0i32; rank];
    *starts.last_mut().unwrap() = 64;
    let operand = patterned_i32(padded_dims.iter().product());
    let update = patterned_i32(update_len);
    let plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &padded_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &update_dims,
        &update_strides,
        &padded_dims,
        &operand_strides,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &padded_dims, &operand_strides, 0).unwrap();
    let update_ref =
        ErasedRawStridedRef::from_slice(&update, &update_dims, &update_strides, update_offset)
            .unwrap();
    let starts_ref =
        ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
    let mut output = vec![0i32; padded_dims.iter().product()];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &padded_dims, &operand_strides, 0)
            .unwrap();
    let ctx = context(case);
    let id = BenchmarkId::new(
        format!("{}_{}", variant.label(), context_label(case)),
        format!("{}_n{}", case.label, case.len),
    );
    group.bench_function(id, |bencher| {
        bencher.iter(|| {
            plan.execute(&ctx, &mut dest, &operand_ref, &update_ref, &starts_ref)
                .unwrap();
            black_box(&mut dest);
        });
    });
}

fn bench_erased_dynamic_update_generic_rank_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_dynamic_update_generic_rank_layout");
    for case in profile_cases() {
        for variant in [
            LayoutVariant::Compact(2),
            LayoutVariant::Compact(4),
            LayoutVariant::Compact(8),
            LayoutVariant::NonunitRank2,
            LayoutVariant::NegativeRank2,
        ] {
            bench_dynamic_update_variant(&mut group, case, variant);
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

fn bench_erased_pad_generic_rank_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("erased_pad_generic_rank_layout");
    for case in profile_cases() {
        for (label, rank, crop, nonunit) in [
            ("compact_rank2", 2usize, false, false),
            ("compact_rank4", 4, false, false),
            ("compact_rank8", 8, false, false),
            ("rank2_negative_crop", 2, true, false),
            ("rank2_nonunit", 2, false, true),
        ] {
            let mut operand_dims = vec![2usize; rank - 1];
            operand_dims.push((case.len / (3 * (1usize << (rank - 2)))).max(1));
            let interior = vec![1i64; 1]
                .into_iter()
                .chain(vec![0; rank - 1])
                .collect::<Vec<_>>();
            let mut edge_low = vec![0i64; rank];
            let mut edge_high = vec![0i64; rank];
            if crop {
                edge_low[1] = -1;
                edge_high[1] = 1;
            }
            let operand_strides = if nonunit {
                vec![2isize, 4]
            } else {
                col_major_strides(&operand_dims)
            };
            let dest_dims: Vec<_> = operand_dims
                .iter()
                .zip(interior.iter())
                .zip(edge_low.iter().zip(edge_high.iter()))
                .map(|((&dim, &inner), (&low, &high))| {
                    (low + (dim as i64 - 1) * (inner + 1) + high + 1) as usize
                })
                .collect();
            let dest_strides = if nonunit {
                vec![2isize, 6]
            } else {
                col_major_strides(&dest_dims)
            };
            let span = |dims: &[usize], strides: &[isize]| {
                dims.iter()
                    .zip(strides)
                    .map(|(&dim, &stride)| (dim - 1) * stride as usize)
                    .sum::<usize>()
                    + 1
            };
            let operand = patterned_i32(span(&operand_dims, &operand_strides));
            let fill = [-1i32];
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
                ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0)
                    .unwrap();
            let ctx = context(case);
            let id = BenchmarkId::new(
                format!("{label}_{}", context_label(case)),
                format!("n{}", case.len),
            );
            let mut output = vec![0i32; span(&dest_dims, &dest_strides)];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0)
                    .unwrap();
            group.bench_function(id, |bencher| {
                bencher.iter(|| {
                    plan.execute(&ctx, &mut dest, &operand_ref, as_bytes(&fill))
                        .unwrap();
                    black_box(&mut dest);
                });
            });
        }
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
        bench_erased_axis_reduce_generic_rank_layout,
        bench_gather_take,
        bench_erased_gather_generic_rank_layout,
        bench_dynamic_slice,
        bench_dynamic_update_slice,
        bench_erased_dynamic_slice_generic_rank_layout,
        bench_erased_dynamic_update_generic_rank_layout,
        bench_pad_fill_and_copy,
        bench_erased_pad_generic_rank_layout,
        bench_raw_any_integer_preflight,
        bench_copy_raw_path,
        bench_scatter_additive
}
criterion_main!(benches);
