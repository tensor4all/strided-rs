#[cfg(feature = "parallel")]
use std::num::NonZeroUsize;

use strided_kernel::{
    compare_into, CompareOp, Identity, StridedArray, StridedError, StridedView, StridedViewMut,
};
#[cfg(feature = "parallel")]
use strided_kernel::{with_execution_policy, ExecutionPolicy};

macro_rules! assert_all_compare_ops {
    ($ty:ty, $lhs:expr, $rhs:expr) => {{
        let lhs = StridedArray::from_parts($lhs, &[4], &[1], 0).unwrap();
        let rhs = StridedArray::from_parts($rhs, &[4], &[1], 0).unwrap();
        let cases = [
            (CompareOp::Eq, [false, true, false, true]),
            (CompareOp::Lt, [true, false, false, false]),
            (CompareOp::Le, [true, true, false, true]),
            (CompareOp::Gt, [false, false, true, false]),
            (CompareOp::Ge, [false, true, true, true]),
        ];

        for (op, expected) in cases {
            let mut out = StridedArray::<bool>::col_major(&[4]);
            compare_into(&mut out.view_mut(), &lhs.view(), &rhs.view(), op).unwrap();
            assert_eq!(out.data(), &expected, "operation {op:?}");
        }
    }};
}

#[test]
fn compare_into_covers_ordered_kernel_dtypes_and_operations() {
    assert_all_compare_ops!(f32, vec![1.0, 2.0, 3.0, 2.0], vec![2.0, 2.0, 2.0, 2.0]);
    assert_all_compare_ops!(f64, vec![1.0, 2.0, 3.0, 2.0], vec![2.0, 2.0, 2.0, 2.0]);
    assert_all_compare_ops!(i32, vec![1, 2, 3, 2], vec![2, 2, 2, 2]);
    assert_all_compare_ops!(i64, vec![1, 2, 3, 2], vec![2, 2, 2, 2]);

    let lhs = StridedArray::from_parts(vec![false, false, true, true], &[4], &[1], 0).unwrap();
    let rhs = StridedArray::from_parts(vec![false, true, false, true], &[4], &[1], 0).unwrap();
    let cases = [
        (CompareOp::Eq, [true, false, false, true]),
        (CompareOp::Lt, [false, true, false, false]),
        (CompareOp::Le, [true, true, false, true]),
        (CompareOp::Gt, [false, false, true, false]),
        (CompareOp::Ge, [true, false, true, true]),
    ];
    for (op, expected) in cases {
        let mut out = StridedArray::<bool>::col_major(&[4]);
        compare_into(&mut out.view_mut(), &lhs.view(), &rhs.view(), op).unwrap();
        assert_eq!(out.data(), &expected, "operation {op:?}");
    }
}

#[test]
fn compare_into_preserves_nan_unordered_semantics() {
    let lhs = StridedArray::from_parts(vec![f32::NAN], &[1], &[1], 0).unwrap();
    let rhs = StridedArray::from_parts(vec![1.0_f32], &[1], &[1], 0).unwrap();

    for op in [
        CompareOp::Eq,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ] {
        let mut out = StridedArray::<bool>::col_major(&[1]);
        compare_into(&mut out.view_mut(), &lhs.view(), &rhs.view(), op).unwrap();
        assert_eq!(out.data(), &[false], "operation {op:?}");
    }
}

#[test]
fn compare_into_preserves_noncontiguous_view_semantics() {
    let dims = [5];
    let lhs_data = [1_i64, 2, 3, 4, 5];
    let rhs = StridedArray::from_parts(vec![0_i64, 3, 3, 3, 6], &dims, &[1], 0).unwrap();
    let lhs = StridedView::<i64, Identity>::new(&lhs_data, &dims, &[-1], 4).unwrap();
    let mut out = StridedArray::<bool>::col_major(&dims);

    compare_into(&mut out.view_mut(), &lhs, &rhs.view(), CompareOp::Gt).unwrap();

    assert_eq!(out.data(), &[true, true, false, false, false]);
}

#[test]
fn compare_into_rejects_noninjective_destination_before_mutation() {
    let lhs = StridedArray::from_parts(vec![1_i64; 4], &[4], &[1], 0).unwrap();
    let rhs = StridedArray::from_parts(vec![2_i64; 4], &[4], &[1], 0).unwrap();
    let mut output = [true];
    let mut dest = StridedViewMut::new(&mut output, &[4], &[0], 0).unwrap();

    let error = compare_into(&mut dest, &lhs.view(), &rhs.view(), CompareOp::Lt);

    assert!(matches!(error, Err(StridedError::NonInjectiveOutputLayout)));
    assert_eq!(output, [true]);
}

#[cfg(feature = "parallel")]
#[test]
fn compare_into_rejects_large_noninjective_destination_before_parallel_mutation() {
    const LEN: usize = 100_000;
    let lhs = StridedArray::from_parts(vec![1_i64; LEN], &[LEN], &[1], 0).unwrap();
    let rhs = StridedArray::from_parts(vec![2_i64; LEN], &[LEN], &[1], 0).unwrap();
    let mut output = [true];
    let mut dest = StridedViewMut::new(&mut output, &[LEN], &[0], 0).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    let error = pool.install(|| {
        with_execution_policy(
            ExecutionPolicy::Rayon {
                max_threads: NonZeroUsize::new(2).unwrap(),
            },
            || compare_into(&mut dest, &lhs.view(), &rhs.view(), CompareOp::Lt),
        )
    });

    assert!(matches!(error, Err(StridedError::NonInjectiveOutputLayout)));
    assert_eq!(output, [true]);
}
