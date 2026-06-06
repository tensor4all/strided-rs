use std::env;
use std::hint::black_box;
use std::ops::Mul;
use std::time::{Duration, Instant};

use strided_kernel::MaybeSendSync;
use strided_kernel::{
    batched_outer_product_into, broadcast_mul_into, mul_into, StridedArray, StridedViewMut,
};

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_RUNS: usize = 15;

#[derive(Clone, Copy)]
enum BenchCase {
    Elementwise {
        n: usize,
    },
    OuterProduct {
        n: usize,
    },
    BatchedOuterCompact {
        j: usize,
        k: usize,
        o: usize,
        t: usize,
    },
    BatchedOuterNoncompact {
        j: usize,
        k: usize,
        o: usize,
        t: usize,
    },
    BatchedOuterNoncompactTorchlikeOutput {
        j: usize,
        k: usize,
        o: usize,
        t: usize,
    },
    BatchedOuterNoncompactLhsScalar {
        j: usize,
        k: usize,
        o: usize,
        t: usize,
    },
    BatchedOuterNoncompactSingleOuterGroup {
        j: usize,
        k: usize,
        o: usize,
        t: usize,
    },
    PermutedElementwise {
        rank: usize,
        extent: usize,
    },
}

#[derive(Clone, Copy)]
enum BenchDType {
    F64,
    C64,
    C128,
}

impl BenchDType {
    fn label(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        }
    }
}

impl BenchCase {
    fn benchmark(self) -> String {
        match self {
            Self::Elementwise { n } => format!("bin_elementwise_mul_{n}x{n}"),
            Self::OuterProduct { n } => format!("bin_outer_product_{n}"),
            Self::BatchedOuterCompact { j, k, o, t } => {
                format!("bin_batched_outer_product_compact_j{j}_k{k}_o{o}_t{t}")
            }
            Self::BatchedOuterNoncompact { j, k, o, t } => {
                format!("bin_batched_outer_product_noncompact_j{j}_k{k}_o{o}_t{t}")
            }
            Self::BatchedOuterNoncompactTorchlikeOutput { j, k, o, t } => {
                format!("bin_batched_outer_product_noncompact_torchlike_output_j{j}_k{k}_o{o}_t{t}")
            }
            Self::BatchedOuterNoncompactLhsScalar { j, k, o, t } => {
                format!("bin_batched_outer_product_noncompact_lhs_scalar_j{j}_k{k}_o{o}_t{t}")
            }
            Self::BatchedOuterNoncompactSingleOuterGroup { j, k, o, t } => {
                format!(
                    "bin_batched_outer_product_noncompact_single_outer_group_j{j}_k{k}_o{o}_t{t}"
                )
            }
            Self::PermutedElementwise { rank, extent } => {
                format!("bin_permuted_elementwise_mul_rank{rank}_extent{extent}")
            }
        }
    }

    fn shape_label(self) -> String {
        match self {
            Self::Elementwise { n } => format!("{n}x{n}"),
            Self::OuterProduct { n } => format!("{n}x{n}"),
            Self::BatchedOuterCompact { j, k, o, t }
            | Self::BatchedOuterNoncompact { j, k, o, t }
            | Self::BatchedOuterNoncompactTorchlikeOutput { j, k, o, t }
            | Self::BatchedOuterNoncompactLhsScalar { j, k, o, t }
            | Self::BatchedOuterNoncompactSingleOuterGroup { j, k, o, t } => {
                format!("j={j};k={k};o={o};t={t}")
            }
            Self::PermutedElementwise { rank, extent } => {
                format!("rank={rank};extent={extent}")
            }
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn profile_cases() -> Vec<BenchCase> {
    match env::var("STRIDED_KERNEL_MUL_BENCH_PROFILE")
        .unwrap_or_else(|_| "full".to_string())
        .as_str()
    {
        "smoke" => vec![
            BenchCase::Elementwise { n: 64 },
            BenchCase::OuterProduct { n: 64 },
            BenchCase::BatchedOuterCompact {
                j: 4,
                k: 4,
                o: 8,
                t: 8,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 4,
                k: 4,
                o: 8,
                t: 8,
            },
            BenchCase::BatchedOuterNoncompactTorchlikeOutput {
                j: 4,
                k: 4,
                o: 8,
                t: 8,
            },
            BenchCase::BatchedOuterNoncompactLhsScalar {
                j: 4,
                k: 4,
                o: 8,
                t: 8,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 2,
                k: 8,
                o: 8,
                t: 8,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 8,
                k: 2,
                o: 8,
                t: 8,
            },
            BenchCase::BatchedOuterNoncompactSingleOuterGroup {
                j: 32,
                k: 32,
                o: 1,
                t: 1,
            },
            BenchCase::PermutedElementwise { rank: 6, extent: 3 },
        ],
        "quick" => vec![
            BenchCase::Elementwise { n: 1024 },
            BenchCase::OuterProduct { n: 2048 },
            BenchCase::BatchedOuterCompact {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompactTorchlikeOutput {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompactLhsScalar {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 8,
                k: 32,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 32,
                k: 8,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompactSingleOuterGroup {
                j: 256,
                k: 256,
                o: 1,
                t: 1,
            },
            BenchCase::PermutedElementwise {
                rank: 12,
                extent: 3,
            },
        ],
        _ => vec![
            BenchCase::Elementwise { n: 2048 },
            BenchCase::OuterProduct { n: 4096 },
            BenchCase::BatchedOuterCompact {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompactTorchlikeOutput {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompactLhsScalar {
                j: 16,
                k: 16,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 8,
                k: 32,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompact {
                j: 32,
                k: 8,
                o: 64,
                t: 64,
            },
            BenchCase::BatchedOuterNoncompactSingleOuterGroup {
                j: 1024,
                k: 1024,
                o: 1,
                t: 1,
            },
            BenchCase::PermutedElementwise {
                rank: 16,
                extent: 3,
            },
        ],
    }
}

fn profile_dtypes() -> Vec<BenchDType> {
    let dtypes: Vec<_> = env::var("STRIDED_KERNEL_MUL_BENCH_DTYPES")
        .unwrap_or_else(|_| "f64".to_string())
        .split(',')
        .filter_map(|value| match value.trim() {
            "f64" => Some(BenchDType::F64),
            "c64" => Some(BenchDType::C64),
            "c128" => Some(BenchDType::C128),
            _ => None,
        })
        .collect();

    if dtypes.is_empty() {
        vec![BenchDType::F64]
    } else {
        dtypes
    }
}

fn fill_value(indices: &[usize], salt: usize) -> f64 {
    let mut acc = salt.wrapping_mul(1_099);
    for (axis, &idx) in indices.iter().enumerate() {
        acc = acc.wrapping_add((axis + 1).wrapping_mul(1_003).wrapping_mul(idx + 1));
    }
    ((acc % 1024) as f64 - 512.0) / 512.0
}

trait BenchScalar: Copy + Default + MaybeSendSync + Mul<Output = Self> + 'static {
    fn from_indices(indices: &[usize], salt: usize) -> Self;
}

impl BenchScalar for f64 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        fill_value(indices, salt)
    }
}

impl BenchScalar for num_complex::Complex32 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        Self::new(
            fill_value(indices, salt) as f32,
            fill_value(indices, salt + 17) as f32,
        )
    }
}

impl BenchScalar for num_complex::Complex64 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        Self::new(fill_value(indices, salt), fill_value(indices, salt + 17))
    }
}

fn make_col_major<T: BenchScalar>(dims: &[usize], salt: usize) -> StridedArray<T> {
    StridedArray::<T>::from_fn_col_major(dims, |idx| T::from_indices(idx, salt))
}

fn duration_stats(mut durations: Vec<Duration>) -> (f64, f64) {
    durations.sort_unstable();
    let median = durations[durations.len() / 2];
    let q1 = durations[durations.len() / 4];
    let q3 = durations[3 * durations.len() / 4];
    (
        median.as_secs_f64() * 1e3,
        q3.saturating_sub(q1).as_secs_f64() * 1e3,
    )
}

fn measure(mut f: impl FnMut()) -> (f64, f64) {
    let warmups = env_usize("STRIDED_KERNEL_MUL_BENCH_WARMUPS", DEFAULT_WARMUPS);
    let runs = env_usize("STRIDED_KERNEL_MUL_BENCH_RUNS", DEFAULT_RUNS);
    for _ in 0..warmups {
        f();
    }

    let mut durations = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        f();
        durations.push(start.elapsed());
    }
    duration_stats(durations)
}

fn run_elementwise<T: BenchScalar>(n: usize) -> (f64, f64) {
    let lhs = make_col_major::<T>(&[n, n], 1);
    let rhs = make_col_major::<T>(&[n, n], 2);
    let mut out = StridedArray::<T>::col_major(&[n, n]);
    measure(|| {
        mul_into(&mut out.view_mut(), &lhs.view(), &rhs.view()).unwrap();
        black_box(out.data().as_ptr());
    })
}

fn run_outer_product<T: BenchScalar>(n: usize) -> (f64, f64) {
    let lhs = make_col_major::<T>(&[n], 3);
    let rhs = make_col_major::<T>(&[n], 4);
    let mut out = StridedArray::<T>::col_major(&[n, n]);
    measure(|| {
        broadcast_mul_into(&mut out.view_mut(), &lhs.view(), &[0], &rhs.view(), &[1]).unwrap();
        black_box(out.data().as_ptr());
    })
}

fn run_batched_outer_compact<T: BenchScalar>(j: usize, k: usize, o: usize, t: usize) -> (f64, f64) {
    let lhs = make_col_major::<T>(&[j, k, t], 5);
    let rhs = make_col_major::<T>(&[o, t], 6);
    let mut out = StridedArray::<T>::col_major(&[j, k, o, t]);
    measure(|| {
        batched_outer_product_into(&mut out.view_mut(), &lhs.view(), &rhs.view(), 2, 1).unwrap();
        black_box(out.data().as_ptr());
    })
}

fn run_batched_outer_noncompact<T: BenchScalar>(
    j: usize,
    k: usize,
    o: usize,
    t: usize,
) -> (f64, f64) {
    let lhs_base = make_col_major::<T>(&[k, j, t], 7);
    let lhs = lhs_base.view().permute(&[1, 0, 2]).unwrap();
    let rhs = make_col_major::<T>(&[o, t], 8);
    let mut out = StridedArray::<T>::col_major(&[j, k, o, t]);
    measure(|| {
        batched_outer_product_into(&mut out.view_mut(), &lhs, &rhs.view(), 2, 1).unwrap();
        black_box(out.data().as_ptr());
    })
}

fn run_batched_outer_noncompact_torchlike_output<T: BenchScalar>(
    j: usize,
    k: usize,
    o: usize,
    t: usize,
) -> (f64, f64) {
    let lhs_base = make_col_major::<T>(&[k, j, t], 17);
    let lhs = lhs_base.view().permute(&[1, 0, 2]).unwrap();
    let rhs = make_col_major::<T>(&[o, t], 18);
    let mut out = StridedArray::<T>::col_major(&[j, k, o, t]);
    let out_strides = [j as isize, 1, (j * k) as isize, (j * k * o) as isize];
    let mut out_view = StridedViewMut::new(out.data_mut(), &[j, k, o, t], &out_strides, 0).unwrap();
    measure(|| {
        batched_outer_product_into(&mut out_view, &lhs, &rhs.view(), 2, 1).unwrap();
        black_box(out_view.as_mut_ptr());
    })
}

fn run_batched_outer_noncompact_lhs_scalar<T: BenchScalar>(
    j: usize,
    k: usize,
    o: usize,
    t: usize,
) -> (f64, f64) {
    let lhs = make_col_major::<T>(&[o, t], 9);
    let rhs_base = make_col_major::<T>(&[k, j, t], 10);
    let rhs = rhs_base.view().permute(&[1, 0, 2]).unwrap();
    let mut out = StridedArray::<T>::col_major(&[j, k, o, t]);
    measure(|| {
        // Keep this as broadcast_mul_into: the case intentionally fixes operand
        // order so the scalar-on-lhs branch is benchmarked.
        broadcast_mul_into(&mut out.view_mut(), &lhs.view(), &[2, 3], &rhs, &[0, 1, 3]).unwrap();
        black_box(out.data().as_ptr());
    })
}

fn permuted_elementwise_axes(rank: usize) -> Vec<usize> {
    if rank == 16 {
        vec![7, 4, 10, 5, 12, 2, 9, 13, 1, 3, 6, 15, 14, 11, 8, 0]
    } else {
        (0..rank).rev().collect()
    }
}

fn run_permuted_elementwise<T: BenchScalar>(rank: usize, extent: usize) -> (f64, f64) {
    let dims = vec![extent; rank];
    let rhs_base = make_col_major::<T>(&dims, 11);
    let rhs_perm = permuted_elementwise_axes(rank);
    let lhs = make_col_major::<T>(&dims, 12);
    let rhs = rhs_base.view().permute(&rhs_perm).unwrap();
    let mut out = StridedArray::<T>::col_major(&dims);
    measure(|| {
        mul_into(&mut out.view_mut(), &lhs.view(), &rhs).unwrap();
        black_box(out.data().as_ptr());
    })
}

fn run_case_typed<T: BenchScalar>(case: BenchCase) -> (f64, f64) {
    match case {
        BenchCase::Elementwise { n } => run_elementwise::<T>(n),
        BenchCase::OuterProduct { n } => run_outer_product::<T>(n),
        BenchCase::BatchedOuterCompact { j, k, o, t } => run_batched_outer_compact::<T>(j, k, o, t),
        BenchCase::BatchedOuterNoncompact { j, k, o, t } => {
            run_batched_outer_noncompact::<T>(j, k, o, t)
        }
        BenchCase::BatchedOuterNoncompactTorchlikeOutput { j, k, o, t } => {
            run_batched_outer_noncompact_torchlike_output::<T>(j, k, o, t)
        }
        BenchCase::BatchedOuterNoncompactLhsScalar { j, k, o, t } => {
            run_batched_outer_noncompact_lhs_scalar::<T>(j, k, o, t)
        }
        BenchCase::BatchedOuterNoncompactSingleOuterGroup { j, k, o, t } => {
            run_batched_outer_noncompact::<T>(j, k, o, t)
        }
        BenchCase::PermutedElementwise { rank, extent } => {
            run_permuted_elementwise::<T>(rank, extent)
        }
    }
}

fn run_case_for_dtype(dtype: BenchDType, case: BenchCase) -> (f64, f64) {
    match dtype {
        BenchDType::F64 => run_case_typed::<f64>(case),
        BenchDType::C64 => run_case_typed::<num_complex::Complex32>(case),
        BenchDType::C128 => run_case_typed::<num_complex::Complex64>(case),
    }
}

fn threads_label() -> String {
    env::var("RAYON_NUM_THREADS")
        .or_else(|_| env::var("OMP_NUM_THREADS"))
        .unwrap_or_else(|_| "unset".to_string())
}

fn main() {
    println!("suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status");
    let threads = threads_label();
    for dtype in profile_dtypes() {
        for case in profile_cases() {
            let (median_ms, iqr_ms) = run_case_for_dtype(dtype, case);
            println!(
                "mul,{},{},{},{},{},{:.6},{:.6},ok",
                case.benchmark(),
                dtype.label(),
                threads,
                case.shape_label(),
                "strided-kernel",
                median_ms,
                iqr_ms
            );
        }
    }
}
