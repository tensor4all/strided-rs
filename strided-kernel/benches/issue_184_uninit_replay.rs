//! Reproducible paired benchmark for issue #184 initialized/uninitialized replay.
//!
//! Build without CPU affinity, extract this target's exact executable from
//! Cargo's current JSON artifact stream, then pin only benchmark execution:
//! `cargo bench -p strided-kernel --features parallel --bench issue_184_uninit_replay --no-run --message-format=json > /tmp/issue-184-artifacts.json`
//! `bench_exe="$(jq -er 'select(.reason == "compiler-artifact" and .target.name == "issue_184_uninit_replay" and (.target.kind | index("bench")) and .executable != null) | .executable' /tmp/issue-184-artifacts.json | tail -n1)"`
//! `test -x "$bench_exe"`
//! `taskset -c 60 "$bench_exe" 1`
//! `taskset -c 60-63 "$bench_exe" 4`

use core::mem::MaybeUninit;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use strided_kernel::{
    batched_outer_product_into, batched_outer_product_into_uninit, broadcast_mul_into,
    broadcast_mul_into_uninit, compare_into, compare_into_uninit, with_execution_policy, CompareOp,
    ErasedFusedPlan, ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef,
    ErasedRawStridedUninitMut, ExecContext, ExecutionPolicy, FusedInst, FusedOp, FusedPlan,
    Identity, KernelDType, StridedView, StridedViewMut,
};

const WARMUPS: usize = 8;
const SAMPLES: usize = 31;
const VECTOR_LEN: usize = 1 << 23;
const BROADCAST_ROWS: usize = 1 << 10;
const BROADCAST_COLUMNS: usize = 1 << 13;
const OUTER_LEN: usize = 1 << 11;

#[derive(Clone, Copy)]
enum PairOrder {
    InitializedFirst,
    UninitializedFirst,
}

impl PairOrder {
    fn label(self) -> &'static str {
        match self {
            Self::InitializedFirst => "initialized_first",
            Self::UninitializedFirst => "uninitialized_first",
        }
    }
}

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), core::mem::size_of_val(data)) }
}

fn as_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), core::mem::size_of_val(data))
    }
}

fn as_uninit_bytes<T>(data: &mut [MaybeUninit<T>]) -> &mut [MaybeUninit<u8>] {
    unsafe {
        core::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), core::mem::size_of_val(data))
    }
}

fn elapsed(operation: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    operation();
    start.elapsed()
}

fn median(samples: &[f64]) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[ordered.len() / 2]
}

fn paired_upper95(initialized: &[f64], uninitialized: &[f64]) -> (f64, f64) {
    let log_ratios: Vec<_> = initialized
        .iter()
        .zip(uninitialized)
        .map(|(&initialized, &uninitialized)| (uninitialized / initialized).ln())
        .collect();
    let mean = log_ratios.iter().sum::<f64>() / log_ratios.len() as f64;
    let variance = log_ratios
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (log_ratios.len() - 1) as f64;
    let standard_error = (variance / log_ratios.len() as f64).sqrt();
    (mean.exp(), (mean + 1.96 * standard_error).exp())
}

fn measure_pair(family: &str, mut initialized: impl FnMut(), mut uninitialized: impl FnMut()) {
    for warmup in 0..WARMUPS {
        if warmup % 2 == 0 {
            initialized();
            uninitialized();
        } else {
            uninitialized();
            initialized();
        }
    }

    let mut initialized_samples = Vec::with_capacity(SAMPLES);
    let mut uninitialized_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        let order = if sample % 2 == 0 {
            PairOrder::InitializedFirst
        } else {
            PairOrder::UninitializedFirst
        };
        let (initialized_elapsed, uninitialized_elapsed) = match order {
            PairOrder::InitializedFirst => {
                let initialized_elapsed = elapsed(&mut initialized);
                let uninitialized_elapsed = elapsed(&mut uninitialized);
                (initialized_elapsed, uninitialized_elapsed)
            }
            PairOrder::UninitializedFirst => {
                let uninitialized_elapsed = elapsed(&mut uninitialized);
                let initialized_elapsed = elapsed(&mut initialized);
                (initialized_elapsed, uninitialized_elapsed)
            }
        };
        let initialized_ms = initialized_elapsed.as_secs_f64() * 1e3;
        let uninitialized_ms = uninitialized_elapsed.as_secs_f64() * 1e3;
        initialized_samples.push(initialized_ms);
        uninitialized_samples.push(uninitialized_ms);
        println!(
            "RAW,{family},{sample},{},{initialized_ms:.9},{uninitialized_ms:.9},{:.9}",
            order.label(),
            uninitialized_ms / initialized_ms
        );
    }

    let (ratio, upper95) = paired_upper95(&initialized_samples, &uninitialized_samples);
    println!(
        "SUMMARY,{family},{:.9},{:.9},{ratio:.9},{upper95:.9}",
        median(&initialized_samples),
        median(&uninitialized_samples)
    );
}

fn vector_values(len: usize, offset: f64) -> Vec<f64> {
    (0..len)
        .map(|index| (index % 251) as f64 * 0.00390625 + offset)
        .collect()
}

fn bench_mul() {
    let dims = [VECTOR_LEN];
    let strides = [1];
    let lhs = vector_values(VECTOR_LEN, 1.0);
    let rhs = vector_values(VECTOR_LEN, 2.0);
    let lhs = StridedView::<_, Identity>::new(&lhs, &dims, &strides, 0).unwrap();
    let rhs = StridedView::<_, Identity>::new(&rhs, &dims, &strides, 0).unwrap();
    let mut initialized = vec![0.0; VECTOR_LEN];
    let mut uninitialized = vec![MaybeUninit::<f64>::uninit(); VECTOR_LEN];
    let mut initialized = StridedViewMut::new(&mut initialized, &dims, &strides, 0).unwrap();
    let mut uninitialized = StridedViewMut::new(&mut uninitialized, &dims, &strides, 0).unwrap();

    measure_pair(
        "mul",
        || {
            strided_kernel::mul_into(&mut initialized, &lhs, &rhs).unwrap();
            black_box(initialized.data()[0]);
        },
        || {
            strided_kernel::mul_into_uninit(&mut uninitialized, &lhs, &rhs).unwrap();
            black_box(unsafe { uninitialized.data()[0].assume_init() });
        },
    );
}

fn bench_broadcast_mul() {
    let dest_dims = [BROADCAST_ROWS, BROADCAST_COLUMNS];
    let dest_strides = [1, BROADCAST_ROWS as isize];
    let lhs_dims = [BROADCAST_ROWS];
    let rhs_dims = [BROADCAST_COLUMNS];
    let lhs = vector_values(BROADCAST_ROWS, 1.0);
    let rhs = vector_values(BROADCAST_COLUMNS, 2.0);
    let lhs = StridedView::<_, Identity>::new(&lhs, &lhs_dims, &[1], 0).unwrap();
    let rhs = StridedView::<_, Identity>::new(&rhs, &rhs_dims, &[1], 0).unwrap();
    let mut initialized = vec![0.0; BROADCAST_ROWS * BROADCAST_COLUMNS];
    let mut uninitialized = vec![MaybeUninit::uninit(); BROADCAST_ROWS * BROADCAST_COLUMNS];
    let mut initialized =
        StridedViewMut::new(&mut initialized, &dest_dims, &dest_strides, 0).unwrap();
    let mut uninitialized =
        StridedViewMut::new(&mut uninitialized, &dest_dims, &dest_strides, 0).unwrap();

    measure_pair(
        "broadcast_mul",
        || {
            broadcast_mul_into(&mut initialized, &lhs, &[0], &rhs, &[1]).unwrap();
            black_box(initialized.data()[0]);
        },
        || {
            broadcast_mul_into_uninit(&mut uninitialized, &lhs, &[0], &rhs, &[1]).unwrap();
            black_box(unsafe { uninitialized.data()[0].assume_init() });
        },
    );
}

fn bench_outer() {
    let input_dims = [OUTER_LEN];
    let dest_dims = [OUTER_LEN, OUTER_LEN];
    let dest_strides = [1, OUTER_LEN as isize];
    let lhs = vector_values(OUTER_LEN, 1.0);
    let rhs = vector_values(OUTER_LEN, 2.0);
    let lhs = StridedView::<_, Identity>::new(&lhs, &input_dims, &[1], 0).unwrap();
    let rhs = StridedView::<_, Identity>::new(&rhs, &input_dims, &[1], 0).unwrap();
    let mut initialized = vec![0.0; OUTER_LEN * OUTER_LEN];
    let mut uninitialized = vec![MaybeUninit::uninit(); OUTER_LEN * OUTER_LEN];
    let mut initialized =
        StridedViewMut::new(&mut initialized, &dest_dims, &dest_strides, 0).unwrap();
    let mut uninitialized =
        StridedViewMut::new(&mut uninitialized, &dest_dims, &dest_strides, 0).unwrap();

    measure_pair(
        "outer",
        || {
            batched_outer_product_into(&mut initialized, &lhs, &rhs, 1, 1).unwrap();
            black_box(initialized.data()[0]);
        },
        || {
            batched_outer_product_into_uninit(&mut uninitialized, &lhs, &rhs, 1, 1).unwrap();
            black_box(unsafe { uninitialized.data()[0].assume_init() });
        },
    );
}

fn bench_compare() {
    let dims = [VECTOR_LEN];
    let strides = [1];
    let lhs = vector_values(VECTOR_LEN, 1.0);
    let rhs = vector_values(VECTOR_LEN, 1.5);
    let lhs = StridedView::<_, Identity>::new(&lhs, &dims, &strides, 0).unwrap();
    let rhs = StridedView::<_, Identity>::new(&rhs, &dims, &strides, 0).unwrap();
    let mut initialized = vec![false; VECTOR_LEN];
    let mut uninitialized = vec![MaybeUninit::uninit(); VECTOR_LEN];
    let mut initialized = StridedViewMut::new(&mut initialized, &dims, &strides, 0).unwrap();
    let mut uninitialized = StridedViewMut::new(&mut uninitialized, &dims, &strides, 0).unwrap();

    measure_pair(
        "compare",
        || {
            compare_into(&mut initialized, &lhs, &rhs, CompareOp::Lt).unwrap();
            black_box(initialized.data()[0]);
        },
        || {
            compare_into_uninit(&mut uninitialized, &lhs, &rhs, CompareOp::Lt).unwrap();
            black_box(unsafe { uninitialized.data()[0].assume_init() });
        },
    );
}

fn bench_fused(context: ExecContext) {
    let dims = [VECTOR_LEN];
    let strides = [1];
    let lhs = vector_values(VECTOR_LEN, 1.0);
    let rhs = vector_values(VECTOR_LEN, 2.0);
    let lhs_ref =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&lhs), &dims, &strides, 0).unwrap();
    let rhs_ref =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&rhs), &dims, &strides, 0).unwrap();
    let refs = [lhs_ref, rhs_ref];
    let ptrs = refs.map(|input| ErasedRawStridedPtr::from_ref(&input));
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
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
        },
    )
    .unwrap();
    let mut initialized = vec![0.0; VECTOR_LEN];
    let mut uninitialized = vec![MaybeUninit::<f64>::uninit(); VECTOR_LEN];
    let mut initialized = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut initialized),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    let mut uninitialized = ErasedRawStridedUninitMut::new(
        KernelDType::F64,
        as_uninit_bytes(&mut uninitialized),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    measure_pair(
        "fused_add_mul",
        || {
            plan.execute(&context, &mut initialized, &refs).unwrap();
            black_box(initialized.data()[0]);
        },
        || {
            plan.execute_uninit(&context, &mut uninitialized, &ptrs)
                .unwrap();
            black_box(unsafe { uninitialized.data_mut()[0].assume_init() });
        },
    );
}

fn main() {
    let threads = std::env::args()
        .nth(1)
        .expect("usage: issue_184_uninit_replay <threads>")
        .parse::<usize>()
        .expect("threads must be a positive integer");
    let max_threads = NonZeroUsize::new(threads).expect("threads must be nonzero");
    let context = if threads == 1 {
        ExecContext::serial()
    } else {
        ExecContext::max_threads(threads).unwrap()
    };
    let policy = if threads == 1 {
        ExecutionPolicy::Sequential
    } else {
        ExecutionPolicy::Rayon { max_threads }
    };

    println!(
        "CONFIG,threads={threads},warmups={WARMUPS},samples={SAMPLES},vector_len={VECTOR_LEN},broadcast={BROADCAST_ROWS}x{BROADCAST_COLUMNS},outer={OUTER_LEN}x{OUTER_LEN}"
    );
    println!("HEADER,family,sample,order,initialized_ms,uninitialized_ms,ratio");
    with_execution_policy(policy, || {
        bench_mul();
        bench_broadcast_mul();
        bench_outer();
        bench_compare();
        bench_fused(context);
    });
}
