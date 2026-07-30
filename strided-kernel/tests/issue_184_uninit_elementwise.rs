use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use strided_kernel::{
    batched_outer_product_into, batched_outer_product_into_uninit, broadcast_mul_into,
    broadcast_mul_into_uninit, compare_into, compare_into_uninit, mul_into, mul_into_uninit,
    CompareOp, ErasedFusedPlan, ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef,
    ErasedRawStridedUninitMut, ExecContext, FusedInst, FusedOp, FusedPlan, Identity, KernelDType,
    StridedError, StridedView, StridedViewMut,
};

fn uninit_view<'a, T>(
    data: &'a mut [MaybeUninit<T>],
    dims: &[usize],
) -> StridedViewMut<'a, MaybeUninit<T>> {
    StridedViewMut::new(data, dims, &[1], 0).unwrap()
}

fn assume_init<T>(data: Vec<MaybeUninit<T>>) -> Vec<T> {
    let mut data = core::mem::ManuallyDrop::new(data);
    unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast(), data.len(), data.capacity()) }
}

fn assert_mul_dtype<T>(lhs: &[T], rhs: &[T], expected: &[T])
where
    T: Copy
        + core::fmt::Debug
        + PartialEq
        + core::ops::Mul<Output = T>
        + strided_kernel::MaybeSendSync
        + 'static,
{
    let dims = [lhs.len()];
    let lhs_view = StridedView::<_, Identity>::new(lhs, &dims, &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(rhs, &dims, &[1], 0).unwrap();
    let mut output = vec![MaybeUninit::uninit(); lhs.len()];
    mul_into_uninit(&mut uninit_view(&mut output, &dims), &lhs_view, &rhs_view).unwrap();
    assert_eq!(assume_init(output), expected);
}

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), core::mem::size_of_val(data)) }
}

fn as_uninit_bytes<T>(data: &mut [MaybeUninit<T>]) -> &mut [MaybeUninit<u8>] {
    unsafe {
        core::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), core::mem::size_of_val(data))
    }
}

fn as_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), core::mem::size_of_val(data))
    }
}

fn assert_mul_broadcast_outer_differential<T>(lhs: &[T], rhs: &[T])
where
    T: Copy
        + core::fmt::Debug
        + PartialEq
        + core::ops::Mul<Output = T>
        + strided_kernel::MaybeSendSync
        + 'static,
{
    let dims = [lhs.len()];
    let lhs_view = StridedView::<_, Identity>::new(lhs, &dims, &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(rhs, &dims, &[1], 0).unwrap();

    let mut initialized = vec![lhs[0]; lhs.len()];
    mul_into(
        &mut StridedViewMut::new(&mut initialized, &dims, &[1], 0).unwrap(),
        &lhs_view,
        &rhs_view,
    )
    .unwrap();
    let mut uninitialized = vec![MaybeUninit::uninit(); lhs.len()];
    mul_into_uninit(
        &mut uninit_view(&mut uninitialized, &dims),
        &lhs_view,
        &rhs_view,
    )
    .unwrap();
    assert_eq!(assume_init(uninitialized), initialized);

    let mut initialized = vec![lhs[0]; lhs.len()];
    broadcast_mul_into(
        &mut StridedViewMut::new(&mut initialized, &dims, &[1], 0).unwrap(),
        &lhs_view,
        &[0],
        &rhs_view,
        &[0],
    )
    .unwrap();
    let mut uninitialized = vec![MaybeUninit::uninit(); lhs.len()];
    broadcast_mul_into_uninit(
        &mut uninit_view(&mut uninitialized, &dims),
        &lhs_view,
        &[0],
        &rhs_view,
        &[0],
    )
    .unwrap();
    assert_eq!(assume_init(uninitialized), initialized);

    let outer_dims = [2usize, 2];
    let lhs_view = StridedView::<_, Identity>::new(&lhs[..2], &[2], &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(&rhs[..2], &[2], &[1], 0).unwrap();
    let mut initialized = vec![lhs[0]; 4];
    batched_outer_product_into(
        &mut StridedViewMut::new(&mut initialized, &outer_dims, &[1, 2], 0).unwrap(),
        &lhs_view,
        &rhs_view,
        1,
        1,
    )
    .unwrap();
    let mut uninitialized = vec![MaybeUninit::uninit(); 4];
    batched_outer_product_into_uninit(
        &mut StridedViewMut::new(&mut uninitialized, &outer_dims, &[1, 2], 0).unwrap(),
        &lhs_view,
        &rhs_view,
        1,
        1,
    )
    .unwrap();
    assert_eq!(assume_init(uninitialized), initialized);
}

fn assert_compare_differential<T>(lhs: &[T], rhs: &[T])
where
    T: Copy + core::fmt::Debug + PartialOrd + strided_kernel::MaybeSendSync,
{
    let dims = [lhs.len()];
    let lhs_view = StridedView::<_, Identity>::new(lhs, &dims, &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(rhs, &dims, &[1], 0).unwrap();
    for op in [
        CompareOp::Eq,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ] {
        let mut initialized = vec![false; lhs.len()];
        compare_into(
            &mut StridedViewMut::new(&mut initialized, &dims, &[1], 0).unwrap(),
            &lhs_view,
            &rhs_view,
            op,
        )
        .unwrap();
        let mut uninitialized = vec![MaybeUninit::uninit(); lhs.len()];
        compare_into_uninit(
            &mut uninit_view(&mut uninitialized, &dims),
            &lhs_view,
            &rhs_view,
            op,
        )
        .unwrap();
        assert_eq!(assume_init(uninitialized), initialized);
    }
}

fn assert_fused_differential<T>(dtype: KernelDType, inputs: &[Vec<T>], plan: FusedPlan)
where
    T: Copy + core::fmt::Debug + PartialEq,
{
    let dims = [inputs[0].len()];
    let strides = [1isize];
    let refs: Vec<_> = inputs
        .iter()
        .map(|input| ErasedRawStridedRef::new(dtype, as_bytes(input), &dims, &strides, 0).unwrap())
        .collect();
    let ptrs: Vec<_> = refs.iter().map(ErasedRawStridedPtr::from_ref).collect();
    let plan = ErasedFusedPlan::compile(dtype, plan).unwrap();

    for context in [ExecContext::serial(), ExecContext::max_threads(2).unwrap()] {
        let mut initialized = vec![inputs[0][0]; dims[0]];
        let mut initialized_dest =
            ErasedRawStridedMut::new(dtype, as_bytes_mut(&mut initialized), &dims, &strides, 0)
                .unwrap();
        plan.execute(&context, &mut initialized_dest, &refs)
            .unwrap();

        let mut uninitialized = vec![MaybeUninit::<T>::uninit(); dims[0]];
        let mut uninitialized_dest = ErasedRawStridedUninitMut::new(
            dtype,
            as_uninit_bytes(&mut uninitialized),
            &dims,
            &strides,
            0,
        )
        .unwrap();
        plan.execute_uninit(&context, &mut uninitialized_dest, &ptrs)
            .unwrap();
        assert_eq!(assume_init(uninitialized), initialized);
    }
}

#[test]
fn typed_uninitialized_full_overwrite_siblings_match_initialized_paths() {
    let lhs = [1i32, 2, 3, 4];
    let rhs = [5i32, 6, 7, 8];
    let lhs_view = StridedView::<_, Identity>::new(&lhs, &[4], &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(&rhs, &[4], &[1], 0).unwrap();

    let mut mul_out = vec![MaybeUninit::uninit(); 4];
    mul_into_uninit(&mut uninit_view(&mut mul_out, &[4]), &lhs_view, &rhs_view).unwrap();
    assert_eq!(assume_init(mul_out), [5, 12, 21, 32]);

    let mut compare_out = vec![MaybeUninit::uninit(); 4];
    compare_into_uninit(
        &mut uninit_view(&mut compare_out, &[4]),
        &lhs_view,
        &rhs_view,
        CompareOp::Lt,
    )
    .unwrap();
    assert_eq!(assume_init(compare_out), [true; 4]);

    let mut broadcast_out = vec![MaybeUninit::uninit(); 4];
    broadcast_mul_into_uninit(
        &mut uninit_view(&mut broadcast_out, &[4]),
        &lhs_view,
        &[0],
        &rhs_view,
        &[0],
    )
    .unwrap();
    assert_eq!(assume_init(broadcast_out), [5, 12, 21, 32]);

    let lhs_outer = [2i32, 3];
    let rhs_outer = [10i32, 20, 30];
    let lhs_outer_view = StridedView::<_, Identity>::new(&lhs_outer, &[2], &[1], 0).unwrap();
    let rhs_outer_view = StridedView::<_, Identity>::new(&rhs_outer, &[3], &[1], 0).unwrap();
    let mut outer_out = vec![MaybeUninit::uninit(); 6];
    let mut outer_view = StridedViewMut::new(&mut outer_out, &[2, 3], &[1, 2], 0).unwrap();
    batched_outer_product_into_uninit(&mut outer_view, &lhs_outer_view, &rhs_outer_view, 1, 1)
        .unwrap();
    assert_eq!(assume_init(outer_out), [20, 30, 40, 60, 60, 90]);

    assert_mul_dtype(&[1.0f32, -2.0], &[3.0, 4.0], &[3.0, -8.0]);
    assert_mul_dtype(&[1.0f64, -2.0], &[3.0, 4.0], &[3.0, -8.0]);
    assert_mul_dtype(&[3i64, -4], &[5, 6], &[15, -24]);
    assert_mul_dtype(
        &[Complex32::new(1.0, 2.0), Complex32::new(3.0, -1.0)],
        &[Complex32::new(2.0, -1.0), Complex32::new(0.5, 2.0)],
        &[Complex32::new(4.0, 3.0), Complex32::new(3.5, 5.5)],
    );
    assert_mul_dtype(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        &[Complex64::new(2.0, -1.0), Complex64::new(0.5, 2.0)],
        &[Complex64::new(4.0, 3.0), Complex64::new(3.5, 5.5)],
    );

    assert_mul_broadcast_outer_differential(&[1.0f32, -2.0], &[3.0, 4.0]);
    assert_mul_broadcast_outer_differential(&[1.0f64, -2.0], &[3.0, 4.0]);
    assert_mul_broadcast_outer_differential(&[i32::MAX, -4], &[2, 6]);
    assert_mul_broadcast_outer_differential(&[i64::MAX, -4], &[2, 6]);
    assert_mul_broadcast_outer_differential(
        &[Complex32::new(1.0, 2.0), Complex32::new(3.0, -1.0)],
        &[Complex32::new(2.0, -1.0), Complex32::new(0.5, 2.0)],
    );
    assert_mul_broadcast_outer_differential(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        &[Complex64::new(2.0, -1.0), Complex64::new(0.5, 2.0)],
    );

    assert_compare_differential(&[1.0f32, -2.0], &[3.0, -2.0]);
    assert_compare_differential(&[1.0f64, -2.0], &[3.0, -2.0]);
    assert_compare_differential(&[1i32, -2], &[3, -2]);
    assert_compare_differential(&[1i64, -2], &[3, -2]);
    assert_compare_differential(&[true, false], &[false, false]);
}

#[test]
fn erased_fused_uninitialized_matches_initialized_for_every_dtype_and_specialization_family() {
    let unary = |op| FusedPlan {
        input_count: 1,
        outputs: vec![1],
        ops: vec![FusedInst {
            op,
            inputs: vec![0],
        }],
    };
    let binary = |op| FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op,
            inputs: vec![0, 1],
        }],
    };

    assert_fused_differential(
        KernelDType::Bool,
        &[vec![true, false]],
        unary(FusedOp::Conj),
    );
    assert_fused_differential(
        KernelDType::F32,
        &[vec![1.0f32, 2.0], vec![3.0, 4.0]],
        binary(FusedOp::Add),
    );
    assert_fused_differential(
        KernelDType::F64,
        &[vec![1.0f64, 2.0], vec![3.0, 4.0]],
        binary(FusedOp::Add),
    );
    assert_fused_differential(
        KernelDType::I32,
        &[vec![i32::MAX, 2], vec![1, 4]],
        binary(FusedOp::Add),
    );
    assert_fused_differential(
        KernelDType::I64,
        &[vec![i64::MAX, 2], vec![1, 4]],
        binary(FusedOp::Add),
    );
    assert_fused_differential(
        KernelDType::C32,
        &[
            vec![Complex32::new(1.0, 2.0), Complex32::new(2.0, 1.0)],
            vec![Complex32::new(3.0, 1.0), Complex32::new(4.0, 2.0)],
        ],
        binary(FusedOp::Multiply),
    );
    assert_fused_differential(
        KernelDType::C64,
        &[
            vec![Complex64::new(1.0, 2.0), Complex64::new(2.0, 1.0)],
            vec![Complex64::new(3.0, 1.0), Complex64::new(4.0, 2.0)],
        ],
        binary(FusedOp::Multiply),
    );

    assert_fused_differential(
        KernelDType::F64,
        &[vec![1.0, 4.0], vec![0.0, 3.0], vec![2.0, 2.0]],
        FusedPlan {
            input_count: 3,
            outputs: vec![3],
            ops: vec![FusedInst {
                op: FusedOp::Clamp,
                inputs: vec![0, 1, 2],
            }],
        },
    );
    assert_fused_differential(
        KernelDType::F64,
        &[vec![1.0, 2.0], vec![3.0, 4.0]],
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
    );
    assert_fused_differential(
        KernelDType::F64,
        &[vec![1.0, 2.0], vec![3.0, 4.0]],
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
                    inputs: vec![0, 2],
                },
            ],
        },
    );
    assert_fused_differential(
        KernelDType::F64,
        &[vec![1.0, 2.0], vec![2.0, 3.0], vec![0.5, 1.0]],
        FusedPlan {
            input_count: 3,
            outputs: vec![5],
            ops: vec![
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![0, 1],
                },
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![3, 2],
                },
                FusedInst {
                    op: FusedOp::Exp,
                    inputs: vec![4],
                },
            ],
        },
    );
    assert_fused_differential(
        KernelDType::F64,
        &[
            vec![4.0, 9.0],
            vec![2.0, 3.0],
            vec![1.0, 1.0],
            vec![4.0, 4.0],
        ],
        FusedPlan {
            input_count: 4,
            outputs: vec![8],
            ops: vec![
                FusedInst {
                    op: FusedOp::Divide,
                    inputs: vec![0, 1],
                },
                FusedInst {
                    op: FusedOp::Maximum,
                    inputs: vec![4, 2],
                },
                FusedInst {
                    op: FusedOp::Minimum,
                    inputs: vec![5, 3],
                },
                FusedInst {
                    op: FusedOp::Sqrt,
                    inputs: vec![6],
                },
                FusedInst {
                    op: FusedOp::Rsqrt,
                    inputs: vec![7],
                },
            ],
        },
    );
}

#[test]
fn erased_fused_uninitialized_replay_writes_valid_bool_output() {
    let lhs = [true, false, true, false];
    let dims = [4usize];
    let strides = [1isize];
    let plan = ErasedFusedPlan::compile(
        KernelDType::Bool,
        FusedPlan {
            input_count: 1,
            outputs: vec![1],
            ops: vec![FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0],
            }],
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::Bool, as_bytes(&lhs), &dims, &strides, 0).unwrap(),
    ];
    let mut output = vec![MaybeUninit::<bool>::uninit(); 4];
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::Bool,
        as_uninit_bytes(&mut output),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    let inputs = inputs.map(|input| ErasedRawStridedPtr::from_ref(&input));
    plan.execute_uninit(&ExecContext::max_threads(1).unwrap(), &mut dest, &inputs)
        .unwrap();
    assert_eq!(assume_init(output), lhs);
}

#[test]
fn typed_uninitialized_validation_errors_leave_sentinels_untouched() {
    let lhs = [2i32, 3];
    let rhs = [5i32, 7];
    let lhs_view = StridedView::<_, Identity>::new(&lhs, &[2], &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(&rhs, &[2], &[1], 0).unwrap();
    let sentinel = MaybeUninit::new(99i32);
    let mut output = vec![sentinel; 2];
    let err = mul_into_uninit(
        &mut StridedViewMut::new(&mut output, &[2], &[0], 0).unwrap(),
        &lhs_view,
        &rhs_view,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        strided_kernel::StridedError::NonInjectiveOutputLayout
    ));
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == 99 }));
}

#[test]
fn erased_fused_validation_errors_precede_writes_and_overlap_precedes_shared_refs() {
    let dims = [4usize];
    let strides = [1isize];
    let plan = ErasedFusedPlan::compile(
        KernelDType::I32,
        FusedPlan {
            input_count: 1,
            outputs: vec![1],
            ops: vec![FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0],
            }],
        },
    )
    .unwrap();

    let mut output = vec![MaybeUninit::new(99i32); 4];
    let ptr = NonNull::new(output.as_mut_ptr().cast::<u8>()).unwrap();
    let overlapping = unsafe {
        ErasedRawStridedPtr::new(
            KernelDType::I32,
            ptr,
            core::mem::size_of_val(output.as_slice()),
            &dims,
            &strides,
            0,
        )
        .unwrap()
    };
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes(&mut output),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    assert!(matches!(
        plan.execute_uninit(&ExecContext::serial(), &mut dest, &[overlapping]),
        Err(StridedError::OverlappingInputOutput { input: 0 })
    ));
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == 99 }));

    let wrong_dtype = [1.0f32; 4];
    let wrong_dtype_ref =
        ErasedRawStridedRef::new(KernelDType::F32, as_bytes(&wrong_dtype), &dims, &strides, 0)
            .unwrap();
    let mut output = vec![MaybeUninit::new(99i32); 4];
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes(&mut output),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    assert!(matches!(
        plan.execute_uninit(
            &ExecContext::serial(),
            &mut dest,
            &[ErasedRawStridedPtr::from_ref(&wrong_dtype_ref)]
        ),
        Err(StridedError::DTypeMismatch { .. })
    ));
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == 99 }));

    let short_dims = [2usize];
    let short_input = [1i32; 2];
    let short_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&short_input),
        &short_dims,
        &strides,
        0,
    )
    .unwrap();
    let mut output = vec![MaybeUninit::new(99i32); 4];
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes(&mut output),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    assert!(matches!(
        plan.execute_uninit(
            &ExecContext::serial(),
            &mut dest,
            &[ErasedRawStridedPtr::from_ref(&short_ref)]
        ),
        Err(StridedError::ShapeMismatch(_, _))
    ));
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == 99 }));

    let input = [1i32; 4];
    let input_ref =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&input), &dims, &strides, 0).unwrap();
    let input_ptr = ErasedRawStridedPtr::from_ref(&input_ref);
    let mut output = vec![MaybeUninit::new(99i32); 4];
    let mut noninjective = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes(&mut output),
        &dims,
        &[0],
        0,
    )
    .unwrap();
    assert!(matches!(
        plan.execute_uninit(&ExecContext::serial(), &mut noninjective, &[input_ptr]),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == 99 }));
}

static PANIC_MUL_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct PanicMul(i32);

impl core::ops::Mul for PanicMul {
    type Output = i32;

    fn mul(self, rhs: Self) -> i32 {
        if PANIC_MUL_CALLS.fetch_add(1, Ordering::SeqCst) == 2 {
            panic!("intentional partial-write probe");
        }
        self.0 * rhs.0
    }
}

#[test]
fn partial_write_panic_leaves_maybe_uninit_destination_safe_to_drop() {
    PANIC_MUL_CALLS.store(0, Ordering::SeqCst);
    let lhs = [PanicMul(2); 8];
    let rhs = [PanicMul(3); 8];
    let lhs_view = StridedView::<_, Identity>::new(&lhs, &[8], &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(&rhs, &[8], &[1], 0).unwrap();
    let mut output = vec![MaybeUninit::new(99i32); 8];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mul_into_uninit(&mut uninit_view(&mut output, &[8]), &lhs_view, &rhs_view).unwrap();
    }));
    assert!(result.is_err());
    assert_eq!(unsafe { output[0].assume_init() }, 6);
    assert_eq!(unsafe { output[1].assume_init() }, 6);
    assert_eq!(unsafe { output[3].assume_init() }, 99);
    drop(output);
}

#[cfg(feature = "parallel")]
#[test]
fn erased_fused_uninitialized_parallel_replay_preserves_wrapping_integer_semantics() {
    let len = (32 * 1024) + 65;
    let lhs = vec![i32::MAX; len];
    let rhs = vec![2i32; len];
    let dims = [len];
    let strides = [1isize];
    let plan = ErasedFusedPlan::compile(
        KernelDType::I32,
        FusedPlan {
            input_count: 2,
            outputs: vec![2],
            ops: vec![FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1],
            }],
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&lhs), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&rhs), &dims, &strides, 0).unwrap(),
    ];
    let mut output = vec![MaybeUninit::<i32>::uninit(); len];
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes(&mut output),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    let inputs = inputs.map(|input| ErasedRawStridedPtr::from_ref(&input));
    plan.execute_uninit(&ExecContext::max_threads(2).unwrap(), &mut dest, &inputs)
        .unwrap();
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == -2 }));
}
