//! Paired regression benchmark for the erased product reductions.
//!
//! The default `ReduceOp::Product` is the reassociated (PyTorch-compatible)
//! product and `ReduceOp::StableProduct` is the exponent-tracked product with a
//! source-order fallback. Both must stay within the reduction policy's budget
//! on in-range normal inputs and must not regress on the extreme-exponent case
//! that the stable variant exists to handle.
//!
//! Every case runs at one and four workers in an explicitly bounded strided pool
//! whose thread count is asserted, so the number on a `SUMMARY` line always
//! belongs to the configuration named there. Setup is outside the timed
//! region: inputs are built once and each plan is compiled before the timed
//! calls.
//!
//! ```text
//! cargo bench -p strided-basic --features parallel --bench product_reduce
//! ```

use std::hint::black_box;
use std::time::Instant;

use strided_basic::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan, KernelDType, KernelStorageElement,
    ReduceOp,
};
use strided_basic::{Exec, Pool};

const WARMUPS: usize = 2;
const SAMPLES: usize = 9;
const THREAD_COUNTS: [usize; 2] = [1, 4];
/// One full-reduction sample; the matrix is fixed, so there is one pass per
/// sample and the reported number is the plan execution itself.
const ROWS: usize = 8192;
const COLS: usize = 4096;

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn time_ms(mut operation: impl FnMut()) -> f64 {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    median(&mut samples)
}

fn plan_full(dtype: KernelDType, op: ReduceOp, n: usize) -> ErasedReducePlan {
    ErasedReducePlan::compile(dtype, op, &[n], &[1]).unwrap()
}

/// Full reduction through the erased plan for the concrete dtype.
fn full_reduce<T: KernelStorageElement + Default>(
    plan: &ErasedReducePlan,
    values: &[T],
    pool: &Pool,
    threads: usize,
) -> T {
    let dims = [values.len()];
    let strides = [1isize];
    let source = ErasedRawStridedRef::from_slice(values, &dims, &strides, 0).unwrap();
    let mut output = [T::default()];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();
    plan.execute(&mut dest, &source, Exec::new(pool, threads).unwrap())
        .unwrap();
    output[0]
}

fn measure_case<T: KernelStorageElement + Default>(
    pool: &Pool,
    label: &str,
    threads: usize,
    n: usize,
    values: &[T],
    fast: &ErasedReducePlan,
    stable: &ErasedReducePlan,
) {
    let run = |label: &str, plan: &ErasedReducePlan| {
        let mut operation = || {
            black_box(full_reduce(plan, values, pool, threads));
        };
        for _ in 0..WARMUPS {
            operation();
        }
        let ms = time_ms(&mut operation);
        println!(
            "SUMMARY {label:<14} {threads}T n={n:<9} full \
             {ms:>9.3}ms {:>7.3} ns/el",
            ms * 1e6 / n as f64,
        );
    };
    run(&format!("{label} Product"), fast);
    run(&format!("{label} StableProduct"), stable);
}

fn main() {
    let n = ROWS * COLS;
    let near = |i: usize| 1.0 + ((i % 1013) as f64 - 506.0) * 1e-9;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let near_f64: Vec<f64> = (0..n).map(near).collect();
    let near_f32: Vec<f32> = (0..n).map(|i| near(i) as f32).collect();
    let wild_f64: Vec<f64> = (0..n)
        .map(|_| {
            let word = next();
            let exponent = (word >> 12) % 601 + 723;
            f64::from_bits((exponent << 52) | (word & ((1 << 52) - 1)))
        })
        .collect();
    let wild_f32: Vec<f32> = (0..n)
        .map(|_| {
            let word = next() as u32;
            let exponent = (word >> 13) % 121 + 63;
            f32::from_bits((exponent << 23) | (word & ((1 << 23) - 1)))
        })
        .collect();

    for threads in THREAD_COUNTS {
        let pool = Pool::builder().threads(threads).build().unwrap();
        {
            {
                let f64_fast = plan_full(KernelDType::F64, ReduceOp::Product, n);
                let f64_stable = plan_full(KernelDType::F64, ReduceOp::StableProduct, n);
                let f32_fast = plan_full(KernelDType::F32, ReduceOp::Product, n);
                let f32_stable = plan_full(KernelDType::F32, ReduceOp::StableProduct, n);
                measure_case(
                    &pool,
                    "f64 near1",
                    threads,
                    n,
                    &near_f64,
                    &f64_fast,
                    &f64_stable,
                );
                measure_case(
                    &pool,
                    "f64 wild-exp",
                    threads,
                    n,
                    &wild_f64,
                    &f64_fast,
                    &f64_stable,
                );
                measure_case(
                    &pool,
                    "f32 near1",
                    threads,
                    n,
                    &near_f32,
                    &f32_fast,
                    &f32_stable,
                );
                measure_case(
                    &pool,
                    "f32 wild-exp",
                    threads,
                    n,
                    &wild_f32,
                    &f32_fast,
                    &f32_stable,
                );
            };
        };
    }
}
