//! Two-input mapped reduction (`zip_reduce`) correctness, widening
//! accumulation, execution-order determinism and the threading threshold
//! (tensor4all/strided-rs#280).
//!
//! The widening case is the reason the entry exists: `dot` pins the result type
//! to `A * B`, so `f32` operands cannot accumulate in `f64` there. The
//! determinism case pins the serial token: its result must be bitwise
//! identical whatever the pool width it runs inside, for inputs above the threading threshold on
//! both the contiguous and the gapped layout. The empty-layout cases pin the
//! identity return that keeps a zero-extent view with an arbitrary stride out
//! of the block planner.

use num_complex::Complex;
use strided_basic::Exec;
use strided_basic::{
    dot, reduce, sum, zip_reduce, Conj, ElementOp, Identity, StridedArray, StridedView,
};

/// Threading threshold of the crate (`threading::MINTHREADLENGTH`); inputs
/// above it take the parallel branches on a pooled token.
#[cfg(feature = "parallel")]
const THREADING_THRESHOLD: usize = 1 << 15;

/// Naive reference: fold `map` over every coordinate of the two views.
fn naive_reference<A, B, OpA, OpB, M, R, U>(
    a: &StridedView<'_, A, OpA>,
    b: &StridedView<'_, B, OpB>,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> U
where
    A: Copy,
    B: Copy,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    M: Fn(A, B) -> U,
    R: Fn(U, U) -> U,
{
    assert_eq!(a.dims(), b.dims());
    let mut acc = init;
    let dims = a.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut coords = vec![0usize; dims.len()];
    for _ in 0..total {
        acc = reduce_fn(acc, map_fn(a.get(&coords), b.get(&coords)));
        for axis in 0..dims.len() {
            coords[axis] += 1;
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    acc
}

/// `f32` operands whose products are exactly representable, laid out so that a
/// left-to-right `f32` accumulation drops every unit term while an `f64`
/// accumulation keeps all of them.
///
/// The first half of the pair list has product `2^24` and the second half
/// product `1`. In `f32` the running total reaches `2^24` after a single large
/// element (ulp `2`, so unit terms are lost) and `2^35` in the half-total (ulp
/// `4096`, so unit terms are certainly lost). In `f64` the total is an exact
/// integer below `2^53`.
fn widening_case(len: usize) -> (Vec<f32>, Vec<f32>, f64) {
    assert_eq!(len % 2, 0);
    let half = len / 2;
    let mut a = vec![1.0f32; len];
    let mut b = vec![1.0f32; len];
    for i in 0..half {
        a[i] = 4096.0;
        b[i] = 4096.0;
    }
    let expected = half as f64 * 16_777_216.0 + half as f64;
    (a, b, expected)
}

#[test]
fn widening_dot_accumulates_f32_operands_in_f64() {
    let dims = [8usize, 512];
    let len: usize = dims.iter().product();
    let (a, b, expected) = widening_case(len);

    let array_a = StridedArray::<f32>::from_parts(a.clone(), &dims, &[1, 8], 0).unwrap();
    let array_b = StridedArray::<f32>::from_parts(b.clone(), &dims, &[1, 8], 0).unwrap();

    // Contiguous operands: the serial contiguous path.
    let wide: f64 = zip_reduce(
        &array_a.view(),
        &array_b.view(),
        |x: f32, y: f32| x as f64 * y as f64,
        |x, y| x + y,
        0.0f64,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(wide, expected, "f64 accumulation must keep every unit term");

    // The narrow route through `dot` cannot widen, and `f32` accumulation
    // loses the unit terms.
    let narrow: f32 =
        dot::<f32, f32, f32, Identity, Identity>(&array_a.view(), &array_b.view()).unwrap();
    assert!(
        expected - narrow as f64 >= 1.0,
        "expected the f32 dot to lose unit terms, got {narrow} for {expected}"
    );

    // Permuted operands: the general plan path must widen identically.
    let permuted_a = array_a.view().permute(&[1, 0]).unwrap();
    let permuted_b = array_b.view().permute(&[1, 0]).unwrap();
    let permuted_wide: f64 = zip_reduce(
        &permuted_a,
        &permuted_b,
        |x: f32, y: f32| x as f64 * y as f64,
        |x, y| x + y,
        0.0f64,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(permuted_wide, expected);

    // A gapped layout (every other element, rank 2) goes through the fused
    // plan with non-unit strides on both operands.
    let gapped_dims = [4usize, 512];
    let gapped_a = StridedView::<f32, Identity>::new(&a, &gapped_dims, &[2, 8], 0).unwrap();
    let gapped_b = StridedView::<f32, Identity>::new(&b, &gapped_dims, &[2, 8], 0).unwrap();
    let gapped_wide: f64 = zip_reduce(
        &gapped_a,
        &gapped_b,
        |x: f32, y: f32| x as f64 * y as f64,
        |x, y| x + y,
        0.0f64,
        Exec::serial(),
    )
    .unwrap();
    // Only the even indices of every contiguous block are visited.
    let reference: f64 = naive_reference(
        &gapped_a,
        &gapped_b,
        |x: f32, y: f32| x as f64 * y as f64,
        |x, y| x + y,
        0.0,
    );
    assert_eq!(gapped_wide, reference);
}

#[test]
fn matches_naive_reference_for_every_layout() {
    // Small integers keep every fold order exact, so the check is a real
    // equality and not a tolerance.
    let data_a: Vec<i64> = (0..4096).map(|i| (i % 17) as i64 - 6).collect();
    let data_b: Vec<i64> = (0..4096).map(|i| (i % 11) as i64 - 4).collect();
    let map = |x: i64, y: i64| x * y;
    let combine = |x: i64, y: i64| x + y;

    let cases: [(&str, Vec<usize>, Vec<isize>, Vec<isize>, isize); 6] = [
        ("contiguous rank 2", vec![7, 13], vec![1, 7], vec![1, 7], 0),
        // Permuted operands: the second axis is contiguous.
        ("permuted rank 2", vec![13, 7], vec![1, 13], vec![1, 13], 0),
        // Gapped with a non-zero offset.
        (
            "gapped offset rank 2",
            vec![5, 9],
            vec![3, 15],
            vec![2, 45],
            21,
        ),
        // Negative stride.
        (
            "negative rank 2",
            vec![6, 8],
            vec![-1, -6],
            vec![-1, -6],
            47,
        ),
        // Stride 0 (broadcast operand on both sides).
        (
            "stride-0 rank 3",
            vec![3, 4, 5],
            vec![0, 0, 0],
            vec![0, 1, 0],
            3,
        ),
        // Rank 3 with mixed strides.
        (
            "mixed rank 3",
            vec![4, 3, 6],
            vec![1, 8, 32],
            vec![2, 4, 48],
            5,
        ),
    ];

    for (name, dims, strides_a, strides_b, offset) in cases {
        let a = StridedView::<i64, Identity>::new(&data_a, &dims, &strides_a, offset).unwrap();
        let b = StridedView::<i64, Identity>::new(&data_b, &dims, &strides_b, offset).unwrap();
        let expected = naive_reference(&a, &b, map, combine, 0i64);
        let actual = zip_reduce(&a, &b, map, combine, 0i64, Exec::serial()).unwrap();
        assert_eq!(actual, expected, "{name}");
    }
}

#[test]
fn honours_each_operand_element_op() {
    let dims = [4usize, 6];
    let data_a: Vec<Complex<f64>> = (0..24)
        .map(|i| Complex::new(i as f64 + 1.0, i as f64 * 0.5))
        .collect();
    let data_b: Vec<Complex<f64>> = (0..24)
        .map(|i| Complex::new(0.25 * i as f64, -(i as f64) - 2.0))
        .collect();
    let strides = [1isize, 4];

    let a = StridedView::<Complex<f64>, Conj>::new(&data_a, &dims, &strides, 0).unwrap();
    let b = StridedView::<Complex<f64>, Identity>::new(&data_b, &dims, &strides, 0).unwrap();
    let map = |x: Complex<f64>, y: Complex<f64>| x * y;
    let combine = |x: Complex<f64>, y: Complex<f64>| x + y;

    let expected = naive_reference(&a, &b, map, combine, Complex::new(0.0, 0.0));
    let actual = zip_reduce(&a, &b, map, combine, Complex::new(0.0, 0.0), Exec::serial()).unwrap();
    assert_eq!(actual, expected);

    // The conjugated left operand is really conjugated, not a copy of `a`.
    let plain_a = StridedView::<Complex<f64>, Identity>::new(&data_a, &dims, &strides, 0).unwrap();
    let plain = zip_reduce(
        &plain_a,
        &b,
        map,
        combine,
        Complex::new(0.0, 0.0),
        Exec::serial(),
    )
    .unwrap();
    assert_ne!(actual, plain);
}

#[test]
fn empty_operand_returns_init() {
    let data_a = [1.0f64, 2.0, 3.0];
    let dims = [0usize, 3];
    let a = StridedView::<f64, Identity>::new(&data_a, &dims, &[1, 1], 0).unwrap();
    let b = StridedView::<f64, Identity>::new(&data_a, &dims, &[1, 1], 0).unwrap();
    let result = zip_reduce(
        &a,
        &b,
        |x: f64, y: f64| x * y,
        |x, y| x + y,
        7.0,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(result, 7.0);
}

/// A zero-extent dimension makes any stride representable, including one the
/// block planner cannot multiply out. Every reduction entry must return its
/// identity before planning (`tensor4all/strided-rs#280` post-review).
#[test]
fn empty_non_contiguous_view_returns_the_identity() {
    let data: [f64; 0] = [];
    let view = StridedView::<f64, Identity>::new(&data, &[0, 2], &[0, isize::MAX], 0).unwrap();
    let zipped = zip_reduce(
        &view,
        &view,
        |x: f64, y: f64| x * y,
        |x, y| x + y,
        7.0,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(zipped, 7.0);
    assert_eq!(
        reduce(&view, |x| x, |x, y| x + y, 7.0, Exec::serial()).unwrap(),
        7.0
    );
    assert_eq!(sum(&view, Exec::serial()).unwrap(), 0.0);
    let dotted = dot::<f64, f64, f64, Identity, Identity>(&view, &view).unwrap();
    assert_eq!(dotted, 0.0);
}

#[test]
fn shape_mismatch_is_rejected() {
    let data_a = [1.0f64, 2.0, 3.0, 4.0];
    let a = StridedView::<f64, Identity>::new(&data_a, &[2, 2], &[1, 2], 0).unwrap();
    let same_rank = StridedView::<f64, Identity>::new(&data_a, &[4, 1], &[1, 1], 0).unwrap();
    let error = zip_reduce(
        &a,
        &same_rank,
        |x: f64, y: f64| x * y,
        |x, y| x + y,
        0.0,
        Exec::serial(),
    );
    assert!(matches!(
        error,
        Err(strided_basic::StridedError::ShapeMismatch(_, _))
    ));
    let other_rank = StridedView::<f64, Identity>::new(&data_a, &[4], &[1], 0).unwrap();
    let error = zip_reduce(
        &a,
        &other_rank,
        |x: f64, y: f64| x * y,
        |x, y| x + y,
        0.0,
        Exec::serial(),
    );
    assert!(matches!(
        error,
        Err(strided_basic::StridedError::RankMismatch(_, _))
    ));
}

/// One pair of operands above the threading threshold.
///
/// `exact` integer-valued data make every fold order exact, which lets a
/// split fold be compared bitwise with the naive reference. The inexact data
/// are the ones that can detect a reordered fold, and the
/// `fold_order_is_observable` control pins that property instead of assuming
/// it.
#[cfg(feature = "parallel")]
struct ThresholdCase {
    name: &'static str,
    a: StridedArray<f64>,
    b: StridedArray<f64>,
}

/// Both above-threshold layouts: the contiguous pair (serial contiguous kernel
/// and the parallel contiguous leaf) and the gapped pair (general fused plan
/// with a non-unit stride on both operands).
#[cfg(feature = "parallel")]
fn threshold_cases(exact: bool) -> Vec<ThresholdCase> {
    const DIMS: [usize; 2] = [257, 128];
    assert!(DIMS.iter().product::<usize>() > THREADING_THRESHOLD);
    [("contiguous", [1isize, 257]), ("gapped", [1, 300])]
        .into_iter()
        .map(|(name, strides)| {
            let len = (DIMS[0] - 1) + (DIMS[1] - 1) * strides[1] as usize + 1;
            let (src_a, src_b): (Vec<f64>, Vec<f64>) = if exact {
                let a = (0..len).map(|i| (i % 13) as f64 - 6.0).collect();
                let b = (0..len).map(|i| (i % 7) as f64 - 3.0).collect();
                (a, b)
            } else {
                let a = (0..len).map(|i| 1.0 / ((i % 97) as f64 + 1.0)).collect();
                let b = (0..len).map(|i| (i % 89) as f64 * 0.1 + 0.3).collect();
                (a, b)
            };
            ThresholdCase {
                name,
                a: StridedArray::from_parts(src_a, &DIMS, &strides, 0).unwrap(),
                b: StridedArray::from_parts(src_b, &DIMS, &strides, 0).unwrap(),
            }
        })
        .collect()
}

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use std::sync::Mutex;
    use strided_basic::Pool;

    /// Pool widths the determinism tests run on. A fresh private pool per width
    /// keeps every width's slots independent of the others.
    const WIDTHS: [usize; 3] = [1, 2, 4];

    fn pool_of(width: usize) -> Pool {
        Pool::builder().threads(width).build().unwrap()
    }

    /// Positive control for the determinism tests: the inexact threshold data
    /// must be able to distinguish fold orders, otherwise a leaked parallel
    /// branch would go unnoticed. A reversed traversal and a left-to-right naive
    /// fold both see a different order from the forward reduction.
    #[test]
    fn fold_order_is_observable() {
        // The gapped case, reversed on both axes: negative strides and an
        // offset at the far corner visit the same elements in the opposite
        // order.
        let case = &threshold_cases(false)[1];
        let forward = zip_reduce_sum(case, Exec::serial());
        let dims = [257usize, 128];
        let strides = [-1isize, -300];
        let offset = 256 + 127 * 300;
        let reversed_a =
            StridedView::<f64, Identity>::new(case.a.data(), &dims, &strides, offset).unwrap();
        let reversed_b =
            StridedView::<f64, Identity>::new(case.b.data(), &dims, &strides, offset).unwrap();
        let reversed = zip_reduce(
            &reversed_a,
            &reversed_b,
            |x, y| x * y,
            |x, y| x + y,
            0.0f64,
            Exec::serial(),
        )
        .unwrap();
        let left_to_right = naive_reference(
            &case.a.view(),
            &case.b.view(),
            |x: f64, y: f64| x * y,
            |x, y| x + y,
            0.0f64,
        );
        assert_ne!(
            forward.to_bits(),
            reversed.to_bits(),
            "the determinism data cannot see a reordered fold"
        );
        assert_ne!(forward.to_bits(), left_to_right.to_bits());
    }

    fn zip_reduce_sum(case: &ThresholdCase, exec: Exec<'_>) -> f64 {
        zip_reduce(
            &case.a.view(),
            &case.b.view(),
            |x, y| x * y,
            |x, y| x + y,
            0.0f64,
            exec,
        )
        .unwrap()
    }

    /// Above the threading threshold `zip_reduce` splits work across workers on
    /// both the contiguous leaf path and the general fused plan; exact
    /// integer-valued data makes every split order exact, so each parallel
    /// branch must equal the naive reference bitwise.
    #[test]
    fn parallel_branch_matches_the_naive_reference_above_the_threshold() {
        for case in threshold_cases(true) {
            let reference = naive_reference(
                &case.a.view(),
                &case.b.view(),
                |x: f64, y: f64| x * y,
                |x, y| x + y,
                0.0f64,
            );
            let serial = zip_reduce_sum(&case, Exec::serial());
            assert_eq!(serial, reference, "{} serial", case.name);
            for width in WIDTHS.into_iter().filter(|&width| width > 1) {
                let pool = pool_of(width);
                let exec = Exec::new(&pool, width).unwrap();
                let root = exec.ledger_root().unwrap();
                let actual = zip_reduce_sum(&case, exec);
                assert_eq!(actual, reference, "{} width {width}", case.name);
                assert!(
                    threadteam::ledger::max_live(root) <= width,
                    "{} width {width} exceeded its slot budget",
                    case.name
                );
            }
        }
    }

    /// The serial token must produce one bitwise result no matter which pool
    /// the call runs inside: each job of a pooled fork runs the serial route on
    /// a pool worker, and the result must equal the calling-thread reference.
    #[test]
    fn serial_result_is_pool_width_independent() {
        for case in threshold_cases(false) {
            let a = &case.a;
            let b = &case.b;
            let pool_free = zip_reduce_sum(&case, Exec::serial());
            let reference_reduce =
                reduce(&a.view(), |x| x, |x, y| x + y, 0.0f64, Exec::serial()).unwrap();
            let reference_sum = sum(&a.view(), Exec::serial()).unwrap();
            let reference_dot =
                dot::<f64, f64, f64, Identity, Identity>(&a.view(), &b.view()).unwrap();

            for width in WIDTHS {
                let pool = pool_of(width);
                let seen = Mutex::new(Vec::new());
                Exec::new(&pool, width).unwrap().fork(2, |_job, _child| {
                    let zipped = zip_reduce_sum(&case, Exec::serial());
                    let reduced =
                        reduce(&a.view(), |x| x, |x, y| x + y, 0.0f64, Exec::serial()).unwrap();
                    let summed = sum(&a.view(), Exec::serial()).unwrap();
                    let dotted =
                        dot::<f64, f64, f64, Identity, Identity>(&a.view(), &b.view()).unwrap();
                    seen.lock().unwrap().push((zipped, reduced, summed, dotted));
                });
                for (zipped, reduced, summed, dotted) in seen.into_inner().unwrap() {
                    assert_eq!(
                        zipped.to_bits(),
                        pool_free.to_bits(),
                        "{} zip_reduce differed inside a {width}-worker pool",
                        case.name
                    );
                    assert_eq!(
                        reduced.to_bits(),
                        reference_reduce.to_bits(),
                        "{} reduce differed inside a {width}-worker pool",
                        case.name
                    );
                    assert_eq!(
                        summed.to_bits(),
                        reference_sum.to_bits(),
                        "{} sum differed inside a {width}-worker pool",
                        case.name
                    );
                    assert_eq!(
                        dotted.to_bits(),
                        reference_dot.to_bits(),
                        "{} dot differed inside a {width}-worker pool",
                        case.name
                    );
                }
            }
        }
    }
}
