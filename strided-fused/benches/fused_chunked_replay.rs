//! Paired benchmark for the chunked fused replay (tensor4all/strided-rs#304).
//!
//! The acceptance in #304 is that a fused region is at least as fast as
//! dispatching the same instructions one at a time through the workspace's
//! vector kernels. Both sides run the same ten-instruction chains with the same
//! `elem-simd-math` kernels, on the reported `1024x64` region and a larger one,
//! at one worker thread.
//!
//! Run it as:
//!
//! ```text
//! RAYON_NUM_THREADS=1 cargo bench -p strided-fused --bench fused_chunked_replay
//! ```
//!
//! The harness asserts the worker count and owns every buffer, view and plan,
//! so the timed region holds only the operation itself (plus whatever the
//! operation allocates internally, which is part of the measured path for both
//! sides).

use core::mem::MaybeUninit;
use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use strided_basic::{
    map_into_uninit_with_kernel, mul_into_uninit, ErasedRawStridedPtr, ErasedRawStridedRef,
    ErasedRawStridedUninitMut, ExecContext, KernelDType, KernelStorageElement, MaybeSendSync,
    StridedView, StridedViewMut,
};
use strided_fused::{
    fused_elementwise_into, ErasedFusedPlan, FusedInst, FusedOp, FusedPlan, FusedScalar,
};

/// The chains #304 reports: ten `tanh` over one input, ten `mul` by a second one.
#[derive(Clone, Copy)]
enum Chain {
    Tanh,
    Mul,
}

impl Chain {
    fn plan(self) -> FusedPlan {
        match self {
            Self::Tanh => FusedPlan {
                input_count: 1,
                outputs: vec![10],
                ops: (0..10)
                    .map(|index| FusedInst {
                        op: FusedOp::Tanh,
                        inputs: vec![index],
                    })
                    .collect(),
            },
            Self::Mul => FusedPlan {
                input_count: 2,
                outputs: vec![11],
                ops: (0..10)
                    .map(|index| FusedInst {
                        op: FusedOp::Multiply,
                        inputs: vec![if index == 0 { 0 } else { 1 + index }, 1],
                    })
                    .collect(),
            },
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Tanh => "ten_tanh",
            Self::Mul => "ten_mul",
        }
    }

    fn inputs(self) -> usize {
        match self {
            Self::Tanh => 1,
            Self::Mul => 2,
        }
    }
}

trait BenchScalar:
    FusedScalar
    + KernelStorageElement
    + MaybeSendSync
    + Send
    + Sync
    + std::ops::Mul<Self, Output = Self>
    + 'static
{
    fn kernel_dtype() -> KernelDType;
    /// The vector kernel the per-instruction dispatch side calls.
    fn tanh_slice(src: &[Self], dst: &mut [MaybeUninit<Self>]);
    /// `dst = a * b`, the arithmetic leaf of the `mul` chain.

    fn value(index: usize) -> Self;
}

impl BenchScalar for f64 {
    fn kernel_dtype() -> KernelDType {
        KernelDType::F64
    }

    fn tanh_slice(src: &[f64], dst: &mut [MaybeUninit<f64>]) {
        elem_simd_math::f64::tanh(src, dst).unwrap();
    }

    fn value(index: usize) -> Self {
        ((index % 251) as f64 - 125.0) * 0.01
    }
}

impl BenchScalar for f32 {
    fn kernel_dtype() -> KernelDType {
        KernelDType::F32
    }

    fn tanh_slice(src: &[f32], dst: &mut [MaybeUninit<f32>]) {
        elem_simd_math::f32::tanh(src, dst).unwrap();
    }

    fn value(index: usize) -> Self {
        ((index % 251) as f32 - 125.0) * 0.01
    }
}

/// Ten passes over the whole region, alternating between two preallocated
/// buffers: exactly what dispatching the plan one instruction at a time
/// through the workspace's kernels costs. The `tanh` chain goes through
/// `map_into_uninit_with_kernel` with the `elem-simd-math` kernel, the `mul`
/// chain through the workspace's own SIMD multiply. Pass 0 reads the input,
/// pass `p` reads the buffer pass `p - 1` wrote.
fn dispatch_ten<T: BenchScalar>(
    chain: Chain,
    a: &[T],
    b: &[T],
    buffers: &mut [Vec<MaybeUninit<T>>; 2],
) {
    let len = a.len();
    let dims = [len];
    let strides = [1isize];
    let b_view = StridedView::<T>::new(b, &dims, &strides, 0).unwrap();
    for pass in 0..10 {
        // SAFETY: pass 0 reads `a`; later passes read the buffer the previous
        // pass wrote in full.
        let source: &[T] = unsafe {
            let pointer = if pass == 0 {
                a.as_ptr()
            } else {
                buffers[1 - pass % 2].as_ptr().cast::<T>()
            };
            std::slice::from_raw_parts(pointer, len)
        };
        let destination = &mut buffers[pass % 2];
        let mut dest = StridedViewMut::new(destination, &dims, &strides, 0).unwrap();
        let src = StridedView::<T>::new(source, &dims, &strides, 0).unwrap();
        match chain {
            Chain::Tanh => map_into_uninit_with_kernel(
                &mut dest,
                &src,
                |src: &[T], dst: &mut [MaybeUninit<T>]| {
                    T::tanh_slice(src, dst);
                    true
                },
                |value| value,
            ),
            Chain::Mul => mul_into_uninit(&mut dest, &src, &b_view),
        }
        .unwrap();
    }
}

fn bench_size<T: BenchScalar>(c: &mut Criterion, label: &str, rows: usize, cols: usize) {
    let len = rows * cols;
    let dims = [rows, cols];
    let strides = [1isize, rows as isize];

    let a: Vec<T> = (0..len).map(T::value).collect();
    let b: Vec<T> = (0..len).map(|index| T::value(index + 7)).collect();

    for chain in [Chain::Tanh, Chain::Mul] {
        let plan = chain.plan();
        let group_name = format!("{label}_{}_{}x{}", chain.name(), rows, cols);
        let mut group = c.benchmark_group(group_name);
        group.measurement_time(Duration::from_secs(3));

        // Typed initialized entry.
        let mut typed_out = vec![T::value(0); len];
        let mut typed_dests =
            vec![StridedViewMut::new(&mut typed_out, &dims, &strides, 0).unwrap()];
        let views: Vec<StridedView<'_, T>> = if chain.inputs() == 1 {
            vec![StridedView::new(&a, &dims, &strides, 0).unwrap()]
        } else {
            vec![
                StridedView::new(&a, &dims, &strides, 0).unwrap(),
                StridedView::new(&b, &dims, &strides, 0).unwrap(),
            ]
        };
        group.bench_function("fused_typed", |bencher| {
            bencher.iter(|| {
                fused_elementwise_into(&mut typed_dests, &views, &plan).unwrap();
                black_box(typed_dests[0].as_mut_ptr());
            });
        });

        // Erased uninitialized entry, which is what tenferro's fused lane calls.
        let erased = ErasedFusedPlan::compile(T::kernel_dtype(), plan.clone()).unwrap();
        let refs: Vec<ErasedRawStridedRef<'_>> = if chain.inputs() == 1 {
            vec![ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap()]
        } else {
            vec![
                ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap(),
                ErasedRawStridedRef::from_slice(&b, &dims, &strides, 0).unwrap(),
            ]
        };
        let ptrs: Vec<ErasedRawStridedPtr<'_>> =
            refs.iter().map(ErasedRawStridedPtr::from_ref).collect();
        let mut uninit_out = vec![MaybeUninit::<T>::uninit(); len];
        let mut uninit_dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut uninit_out, &dims, &strides, 0)
                .unwrap();
        let ctx = ExecContext::serial();
        group.bench_function("fused_erased_uninit", |bencher| {
            bencher.iter(|| {
                erased
                    .execute_uninit(&ctx, &mut uninit_dest, &ptrs)
                    .unwrap();
                black_box(uninit_dest.dims());
            });
        });

        // One instruction at a time over preallocated buffers.
        let mut buffers = [
            vec![MaybeUninit::<T>::uninit(); len],
            vec![MaybeUninit::<T>::uninit(); len],
        ];
        group.bench_function("per_instruction_dispatch", |bencher| {
            bencher.iter(|| {
                dispatch_ten(chain, &a, &b, &mut buffers);
                black_box(buffers[0].as_ptr());
            });
        });

        group.finish();
    }
}

fn benchmarks(c: &mut Criterion) {
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "run with RAYON_NUM_THREADS=1: the numbers must belong to one worker thread"
    );
    for (rows, cols) in [(1024, 64), (4096, 256)] {
        bench_size::<f64>(c, "f64", rows, cols);
        bench_size::<f32>(c, "f32", rows, cols);
    }
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
