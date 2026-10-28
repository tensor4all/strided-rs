//! Paired benchmark for the chunked fused replay (tensor4all/strided-rs#304).
//!
//! The acceptance in #304 is that a fused region is at least as fast as
//! dispatching the same instructions one at a time through the workspace's
//! vector kernels. Both sides run the same instruction chains over the reported
//! `1024x64` and `4096x256` regions at one and four worker threads.
//!
//! The rows cover the two shapes that used to force the fused chain onto its
//! interpreter: a stride-0 broadcast input on the blocked axis, and a plan whose
//! shared chain feeds two live outputs.
//!
//! Run it as:
//!
//! ```text
//! RAYON_NUM_THREADS=8 cargo bench -p strided-fused --features parallel --bench fused_chunked_replay
//! ```
//!
//! The harness asserts the worker count it reports and owns every buffer, view
//! and plan, so the timed region holds only the operation itself (plus whatever
//! the operation allocates internally, which is part of the measured path for
//! both sides).

use core::mem::MaybeUninit;
use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use strided_basic::{
    map_into_uninit_with_kernel, mul_into_uninit, zip_map2_into, ErasedRawStridedPtr,
    ErasedRawStridedRef, ErasedRawStridedUninitMut, Exec, KernelDType, KernelStorageElement, Pool,
    StridedView, StridedViewMut,
};
use strided_fused::{
    fused_elementwise_into, ErasedFusedPlan, FusedInst, FusedOp, FusedPlan, FusedScalar,
};

/// The instruction chains measured for each region.
#[derive(Clone, Copy)]
enum Row {
    /// Ten `tanh` over one input.
    Tanh,
    /// Ten `sqrt` over one input: the compile-target-sensitive chunk loop
    /// (tensor4all/strided-rs#340). Negative inputs give NaN, which the vector
    /// square root handles at the same cost.
    Sqrt,
    /// Ten `mul` by a second input.
    Mul,
    /// Ten `mul` by a second input, with the first input broadcast on the
    /// blocked axis (stride 0).
    BroadcastMul,
    /// Ten shared `mul` instructions plus `add(input0)`: two live outputs over
    /// the same chain.
    TwoOutput,
}

impl Row {
    fn plan(self) -> FusedPlan {
        match self {
            Self::Tanh | Self::Sqrt => FusedPlan {
                input_count: 1,
                outputs: vec![10].into_iter().collect(),
                ops: (0..10)
                    .map(|index| FusedInst {
                        op: if matches!(self, Self::Tanh) {
                            FusedOp::Tanh
                        } else {
                            FusedOp::Sqrt
                        },
                        inputs: vec![index].into_iter().collect(),
                    })
                    .collect(),
            },
            Self::Mul | Self::BroadcastMul => FusedPlan {
                input_count: 2,
                outputs: vec![11].into_iter().collect(),
                ops: (0..10)
                    .map(|index| FusedInst {
                        op: FusedOp::Multiply,
                        inputs: vec![if index == 0 { 0 } else { 1 + index }, 1]
                            .into_iter()
                            .collect(),
                    })
                    .collect(),
            },
            Self::TwoOutput => {
                let mut ops: Vec<FusedInst> = (0..10)
                    .map(|index| FusedInst {
                        op: FusedOp::Multiply,
                        inputs: vec![if index == 0 { 0 } else { 1 + index }, 1]
                            .into_iter()
                            .collect(),
                    })
                    .collect();
                ops.push(FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![11, 0].into_iter().collect(),
                });
                FusedPlan {
                    input_count: 2,
                    outputs: vec![11, 12].into_iter().collect(),
                    ops: ops.into_iter().collect(),
                }
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Tanh => "ten_tanh",
            Self::Sqrt => "ten_sqrt",
            Self::Mul => "ten_mul",
            Self::BroadcastMul => "ten_mul_broadcast",
            Self::TwoOutput => "ten_mul_add_two_output",
        }
    }

    fn inputs(self) -> usize {
        match self {
            Self::Tanh | Self::Sqrt => 1,
            Self::Mul | Self::BroadcastMul | Self::TwoOutput => 2,
        }
    }

    fn outputs(self) -> usize {
        match self {
            Self::TwoOutput => 2,
            _ => 1,
        }
    }

    fn broadcast(self) -> bool {
        matches!(self, Self::BroadcastMul)
    }
}

/// Mints one token per call: serial for one thread, else `threads` slots of `pool`.
#[derive(Clone, Copy)]
struct Placement<'p> {
    pool: &'p Pool,
    threads: usize,
}

impl<'p> Placement<'p> {
    fn exec(self) -> Exec<'p> {
        if self.threads == 1 {
            Exec::serial()
        } else {
            Exec::new(self.pool, self.threads).unwrap()
        }
    }
}

trait BenchScalar:
    FusedScalar
    + KernelStorageElement
    + Send
    + Sync
    + std::ops::Mul<Self, Output = Self>
    + std::ops::Add<Self, Output = Self>
    + 'static
{
    fn kernel_dtype() -> KernelDType;
    /// The vector kernel the per-instruction dispatch side calls.
    fn tanh_slice(src: &[Self], dst: &mut [MaybeUninit<Self>]);
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

/// Ten multiplies over preallocated buffers, ping-ponging between `first` and
/// `second`. The first pass reads `a_view` (which may be broadcast); pass `p`
/// reads the buffer pass `p - 1` wrote. The result ends in `second`.
fn ping_pong_mul<T: BenchScalar>(
    a_view: &StridedView<'_, T>,
    b_view: &StridedView<'_, T>,
    dims: &[usize],
    strides: &[isize],
    len: usize,
    first: &mut [MaybeUninit<T>],
    second: &mut [MaybeUninit<T>],
    place: Placement<'_>,
) {
    {
        let mut dest = StridedViewMut::new(first, dims, strides, 0).unwrap();
        mul_into_uninit(&mut dest, a_view, b_view, place.exec()).unwrap();
    }
    for pass in 1..10 {
        // SAFETY: `first`/`second` are distinct buffers; the raw read of one and
        // the mutable borrow of the other do not overlap.
        if pass % 2 == 1 {
            let src_data: &[T] =
                unsafe { std::slice::from_raw_parts(first.as_ptr().cast::<T>(), len) };
            let src = StridedView::<T>::new(src_data, dims, strides, 0).unwrap();
            let mut dest = StridedViewMut::new(second, dims, strides, 0).unwrap();
            mul_into_uninit(&mut dest, &src, b_view, place.exec()).unwrap();
        } else {
            let src_data: &[T] =
                unsafe { std::slice::from_raw_parts(second.as_ptr().cast::<T>(), len) };
            let src = StridedView::<T>::new(src_data, dims, strides, 0).unwrap();
            let mut dest = StridedViewMut::new(first, dims, strides, 0).unwrap();
            mul_into_uninit(&mut dest, &src, b_view, place.exec()).unwrap();
        }
    }
}

/// Ten `tanh` (or `sqrt`) passes over preallocated buffers, ping-ponging as
/// [`ping_pong_mul`]. The result ends in `second`. `tanh` goes through its
/// vector kernel; `sqrt` has none and maps the scalar method, as tenferro's
/// `sqrt` does.
#[allow(clippy::too_many_arguments)]
fn ping_pong_unary<T: BenchScalar>(
    sqrt: bool,
    a_view: &StridedView<'_, T>,
    dims: &[usize],
    strides: &[isize],
    len: usize,
    first: &mut [MaybeUninit<T>],
    second: &mut [MaybeUninit<T>],
    place: Placement<'_>,
) {
    let map = |dest: &mut [MaybeUninit<T>], src: &StridedView<'_, T>| {
        let mut dest = StridedViewMut::new(dest, dims, strides, 0).unwrap();
        map_into_uninit_with_kernel(
            &mut dest,
            src,
            |src: &[T], dst: &mut [MaybeUninit<T>]| {
                if sqrt {
                    return false;
                }
                T::tanh_slice(src, dst);
                true
            },
            |value: T| if sqrt { value.fused_sqrt() } else { value },
            place.exec(),
        )
        .unwrap();
    };
    map(first, a_view);
    for pass in 1..10 {
        // SAFETY: as for `ping_pong_mul`.
        let src_data: &[T] = unsafe {
            let pointer = if pass % 2 == 1 {
                first.as_ptr()
            } else {
                second.as_ptr()
            };
            std::slice::from_raw_parts(pointer.cast::<T>(), len)
        };
        let src = StridedView::<T>::new(src_data, dims, strides, 0).unwrap();
        if pass % 2 == 1 {
            map(second, &src);
        } else {
            map(first, &src);
        }
    }
}

/// The per-instruction dispatch cost: the same instructions one at a time
/// through the workspace's vector kernels. For [`Row::TwoOutput`] the shared
/// chain runs once; its buffer is both the first output and the input of the
/// final `add(input0)` that produces the second output, and every op goes
/// through a threaded kernel.
#[allow(clippy::too_many_arguments)]
fn per_instruction<T: BenchScalar>(
    row: Row,
    a_data: &[T],
    a_strides: &[isize],
    b_data: &[T],
    b_strides: &[isize],
    dims: &[usize],
    len: usize,
    buffers: &mut [Vec<MaybeUninit<T>>; 3],
    place: Placement<'_>,
) {
    let [buffer0, buffer1, buffer2] = buffers;
    let a_view = StridedView::<T>::new(a_data, dims, a_strides, 0).unwrap();
    let b_view = StridedView::<T>::new(b_data, dims, b_strides, 0).unwrap();
    match row {
        Row::Tanh | Row::Sqrt => ping_pong_unary(
            matches!(row, Row::Sqrt),
            &a_view,
            dims,
            b_strides,
            len,
            buffer0,
            buffer1,
            place,
        ),
        Row::Mul | Row::BroadcastMul => ping_pong_mul(
            &a_view, &b_view, dims, b_strides, len, buffer0, buffer1, place,
        ),
        Row::TwoOutput => {
            // The shared chain runs once and ends in `buffer1`.
            ping_pong_mul(
                &a_view, &b_view, dims, b_strides, len, buffer0, buffer1, place,
            );
            // SAFETY: `buffer1` is read through a raw pointer while `buffer2`
            // is written; they are distinct buffers.
            let chain_data: &[T] =
                unsafe { std::slice::from_raw_parts(buffer1.as_ptr().cast::<T>(), len) };
            let chain = StridedView::<T>::new(chain_data, dims, b_strides, 0).unwrap();
            let mut dest = StridedViewMut::new(buffer2, dims, b_strides, 0).unwrap();
            zip_map2_into(
                &mut dest,
                &chain,
                &a_view,
                |x, y| MaybeUninit::new(x + y),
                place.exec(),
            )
            .unwrap();
        }
    }
}

fn bench_size<T: BenchScalar>(
    c: &mut Criterion,
    label: &str,
    rows: usize,
    cols: usize,
    place: Placement<'_>,
) {
    let threads = place.threads;
    let len = rows * cols;
    let dims = [rows, cols];
    let cont = [1isize, rows as isize];
    let broadcast = [0isize, 1isize];

    let a_full: Vec<T> = (0..len).map(T::value).collect();
    let a_broadcast: Vec<T> = (0..cols).map(T::value).collect();
    let b: Vec<T> = (0..len).map(|index| T::value(index + 7)).collect();

    for row in [
        Row::Tanh,
        Row::Sqrt,
        Row::Mul,
        Row::BroadcastMul,
        Row::TwoOutput,
    ] {
        let plan = row.plan();
        let group_name = format!("{label}_{}_{}x{}_{}t", row.name(), rows, cols, threads);
        let mut group = c.benchmark_group(group_name);
        group.measurement_time(Duration::from_secs(3));

        let a_data: &[T] = if row.broadcast() {
            &a_broadcast
        } else {
            &a_full
        };
        let a_strides: &[isize] = if row.broadcast() { &broadcast } else { &cont };

        // Typed initialized entry.
        let output_count = row.outputs();
        let mut typed_outputs: Vec<Vec<T>> =
            (0..output_count).map(|_| vec![T::value(0); len]).collect();
        let input_views: Vec<StridedView<'_, T>> = if row.inputs() == 1 {
            vec![StridedView::<T>::new(a_data, &dims, a_strides, 0).unwrap()]
        } else {
            vec![
                StridedView::<T>::new(a_data, &dims, a_strides, 0).unwrap(),
                StridedView::<T>::new(&b, &dims, &cont, 0).unwrap(),
            ]
        };
        {
            let mut typed_views: Vec<StridedViewMut<'_, T>> = typed_outputs
                .iter_mut()
                .map(|data| StridedViewMut::new(data, &dims, &cont, 0).unwrap())
                .collect();
            group.bench_function("fused_typed", |bencher| {
                bencher.iter(|| {
                    fused_elementwise_into(&mut typed_views, &input_views, &plan, place.exec())
                        .unwrap();
                    black_box(typed_views[0].as_mut_ptr());
                });
            });
        }

        // Erased uninitialized entry, which is what tenferro's fused lane calls.
        let erased = ErasedFusedPlan::compile(T::kernel_dtype(), plan.clone()).unwrap();
        let a_ref = ErasedRawStridedRef::from_slice(a_data, &dims, a_strides, 0).unwrap();
        let b_ref = ErasedRawStridedRef::from_slice(&b, &dims, &cont, 0).unwrap();
        let refs = if row.inputs() == 1 {
            vec![a_ref]
        } else {
            vec![a_ref, b_ref]
        };
        let pointers: Vec<ErasedRawStridedPtr<'_>> =
            refs.iter().map(ErasedRawStridedPtr::from_ref).collect();
        let mut uninitialized_outputs: Vec<Vec<MaybeUninit<T>>> = (0..output_count)
            .map(|_| vec![MaybeUninit::uninit(); len])
            .collect();
        let mut erased_views: Vec<ErasedRawStridedUninitMut<'_>> = uninitialized_outputs
            .iter_mut()
            .map(|data| {
                ErasedRawStridedUninitMut::from_uninit_slice(data, &dims, &cont, 0).unwrap()
            })
            .collect();
        group.bench_function("fused_erased_uninit", |bencher| {
            bencher.iter(|| {
                erased
                    .execute_outputs_uninit(&mut erased_views, &pointers, place.exec())
                    .unwrap();
                black_box(erased_views[0].dims());
            });
        });

        // One instruction at a time over preallocated buffers.
        let mut buffers: [Vec<MaybeUninit<T>>; 3] =
            std::array::from_fn(|_| vec![MaybeUninit::uninit(); len]);
        group.bench_function("per_instruction_dispatch", |bencher| {
            bencher.iter(|| {
                per_instruction(
                    row,
                    a_data,
                    a_strides,
                    &b,
                    &cont,
                    &dims,
                    len,
                    &mut buffers,
                    place,
                );
                black_box(buffers[0].as_ptr());
            });
        });

        group.finish();
    }
}

#[cfg(feature = "parallel")]
fn thread_counts() -> [usize; 2] {
    [1, 4]
}

#[cfg(not(feature = "parallel"))]
fn thread_counts() -> [usize; 1] {
    [1]
}

fn benchmarks(c: &mut Criterion) {
    for threads in thread_counts() {
        let pool = Pool::builder().threads(threads).build().unwrap();
        if threads > 1 {
            assert!(
                Exec::new(&pool, threads).unwrap().budget().get() >= threads,
                "the {threads}-thread rows need a pool of at least {threads} workers"
            );
        }
        let place = Placement {
            pool: &pool,
            threads,
        };
        for (rows, cols) in [(1024usize, 64usize), (4096, 256)] {
            bench_size::<f64>(c, "f64", rows, cols, place);
            bench_size::<f32>(c, "f32", rows, cols, place);
        }
    }
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
