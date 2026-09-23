//! Per-op reduction kernels against a naive reference.
//!
//! Every op is checked on contiguous, strided (including negative strides)
//! and broadcast (stride zero) layouts, under a serial and a bounded parallel
//! context. Sizes are odd so the eight lane, SIMD and output tile remainder
//! loops run. Inputs are chosen so that every partial result is exactly
//! representable; the reference and the kernels therefore agree bitwise for
//! any evaluation order. Separate tests pin the orders that are specified.

use num_complex::Complex64;
use strided_basic::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan, ExecContext, KernelDType,
    KernelStorageElement, ReduceOp,
};

/// Larger than `MINTHREADLENGTH` so that parallel contexts split the work.
const LARGE: usize = (1 << 15) + 65;

const ALL_OPS: [ReduceOp; 5] = [
    ReduceOp::Sum,
    ReduceOp::Product,
    ReduceOp::SumSquares,
    ReduceOp::Max,
    ReduceOp::Min,
];

trait Scalar: KernelStorageElement + Copy + PartialEq + core::fmt::Debug + 'static {
    const KIND: KernelDType;
    /// Ops the erased plan accepts for this dtype.
    const OPS: &'static [ReduceOp];
    /// Value for element `i` whose partial sums and squares stay exact.
    fn additive(i: usize) -> Self;
    /// Value for element `i` whose partial products stay exact.
    fn multiplicative(i: usize) -> Self;
    fn identity(op: ReduceOp) -> Self;
    fn combine(op: ReduceOp, lhs: Self, rhs: Self) -> Self;
    fn map(op: ReduceOp, value: Self) -> Self;
    fn same(lhs: Self, rhs: Self) -> bool {
        lhs == rhs
    }
}

macro_rules! float_scalar {
    ($ty:ty, $dtype:ident) => {
        impl Scalar for $ty {
            const KIND: KernelDType = KernelDType::$dtype;
            const OPS: &'static [ReduceOp] = &ALL_OPS;
            fn additive(i: usize) -> Self {
                ((i * 7919) % 17) as $ty - 8.0
            }
            fn multiplicative(i: usize) -> Self {
                if (i * 31) % 5 < 2 {
                    -1.0
                } else {
                    1.0
                }
            }
            fn identity(op: ReduceOp) -> Self {
                match op {
                    ReduceOp::Sum | ReduceOp::SumSquares => 0.0,
                    ReduceOp::Product => 1.0,
                    ReduceOp::Max => <$ty>::NEG_INFINITY,
                    ReduceOp::Min => <$ty>::INFINITY,
                    _ => unreachable!(),
                }
            }
            fn combine(op: ReduceOp, lhs: Self, rhs: Self) -> Self {
                match op {
                    ReduceOp::Sum | ReduceOp::SumSquares => lhs + rhs,
                    ReduceOp::Product => lhs * rhs,
                    ReduceOp::Max if lhs.is_nan() || rhs.is_nan() => <$ty>::NAN,
                    ReduceOp::Min if lhs.is_nan() || rhs.is_nan() => <$ty>::NAN,
                    ReduceOp::Max => lhs.max(rhs),
                    ReduceOp::Min => lhs.min(rhs),
                    _ => unreachable!(),
                }
            }
            fn map(op: ReduceOp, value: Self) -> Self {
                if op == ReduceOp::SumSquares {
                    value * value
                } else {
                    value
                }
            }
            fn same(lhs: Self, rhs: Self) -> bool {
                (lhs.is_nan() && rhs.is_nan()) || lhs.to_bits() == rhs.to_bits()
            }
        }
    };
}
float_scalar!(f32, F32);
float_scalar!(f64, F64);

macro_rules! int_scalar {
    ($ty:ty, $dtype:ident) => {
        impl Scalar for $ty {
            const KIND: KernelDType = KernelDType::$dtype;
            const OPS: &'static [ReduceOp] = &[
                ReduceOp::Sum,
                ReduceOp::Product,
                ReduceOp::Max,
                ReduceOp::Min,
            ];
            fn additive(i: usize) -> Self {
                // Large magnitudes so sums and squares wrap.
                (i as $ty)
                    .wrapping_mul(<$ty>::MAX / 7)
                    .wrapping_add(i as $ty)
            }
            fn multiplicative(i: usize) -> Self {
                // Wrapping products are exact in any order.
                [1, -1, 2, -2, 3, -3, 1][i % 7]
            }
            fn identity(op: ReduceOp) -> Self {
                match op {
                    ReduceOp::Sum | ReduceOp::SumSquares => 0,
                    ReduceOp::Product => 1,
                    ReduceOp::Max => <$ty>::MIN,
                    ReduceOp::Min => <$ty>::MAX,
                    _ => unreachable!(),
                }
            }
            fn combine(op: ReduceOp, lhs: Self, rhs: Self) -> Self {
                match op {
                    ReduceOp::Sum | ReduceOp::SumSquares => lhs.wrapping_add(rhs),
                    ReduceOp::Product => lhs.wrapping_mul(rhs),
                    ReduceOp::Max => lhs.max(rhs),
                    ReduceOp::Min => lhs.min(rhs),
                    _ => unreachable!(),
                }
            }
            fn map(op: ReduceOp, value: Self) -> Self {
                if op == ReduceOp::SumSquares {
                    value.wrapping_mul(value)
                } else {
                    value
                }
            }
        }
    };
}
int_scalar!(i32, I32);
int_scalar!(i64, I64);

impl Scalar for Complex64 {
    const KIND: KernelDType = KernelDType::C64;
    const OPS: &'static [ReduceOp] = &[ReduceOp::Sum, ReduceOp::Product];
    fn additive(i: usize) -> Self {
        Complex64::new(f64::additive(i), f64::additive(i + 3))
    }
    fn multiplicative(i: usize) -> Self {
        // Powers of the imaginary unit keep every partial product exact.
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ][(i * 7) % 4]
    }
    fn identity(op: ReduceOp) -> Self {
        match op {
            ReduceOp::Product => Complex64::new(1.0, 0.0),
            _ => Complex64::new(0.0, 0.0),
        }
    }
    fn combine(op: ReduceOp, lhs: Self, rhs: Self) -> Self {
        match op {
            ReduceOp::Product => lhs * rhs,
            _ => lhs + rhs,
        }
    }
    fn map(op: ReduceOp, value: Self) -> Self {
        if op == ReduceOp::SumSquares {
            value * value
        } else {
            value
        }
    }
    // Complex products create signed zeros whose sign depends on the
    // association order, so complex results compare numerically.
}

fn ops_for<T: Scalar>() -> &'static [ReduceOp] {
    T::OPS
}

fn contexts() -> Vec<(&'static str, ExecContext)> {
    vec![
        ("serial", ExecContext::serial()),
        ("max_threads(4)", ExecContext::max_threads(4).unwrap()),
    ]
}

fn value_for<T: Scalar>(op: ReduceOp, i: usize) -> T {
    if op == ReduceOp::Product {
        T::multiplicative(i)
    } else {
        T::additive(i)
    }
}

/// Buffer offset of the multi-index `coords` under `strides` and `offset`.
fn offset_of(coords: &[usize], strides: &[isize], offset: isize) -> usize {
    let offset = coords
        .iter()
        .zip(strides)
        .fold(offset, |acc, (&c, &s)| acc + c as isize * s);
    usize::try_from(offset).unwrap()
}

/// Column-major multi-indices of `dims`.
fn coords_of(dims: &[usize]) -> Vec<Vec<usize>> {
    let total: usize = dims.iter().product();
    (0..total)
        .map(|mut linear| {
            dims.iter()
                .map(|&d| {
                    let c = linear % d;
                    linear /= d;
                    c
                })
                .collect()
        })
        .collect()
}

/// Buffer large enough for `dims`, `strides`, `offset`, filled by index.
fn buffer<T: Scalar>(
    op: ReduceOp,
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    special: Option<(usize, T)>,
) -> Vec<T> {
    let len = coords_of(dims)
        .iter()
        .map(|c| offset_of(c, strides, offset) + 1)
        .max()
        .unwrap_or(offset as usize + 1);
    let mut data: Vec<T> = (0..len).map(|i| value_for::<T>(op, i)).collect();
    if let Some((index, value)) = special {
        data[index] = value;
    }
    data
}

fn naive_full<T: Scalar>(
    op: ReduceOp,
    data: &[T],
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) -> T {
    coords_of(dims).iter().fold(T::identity(op), |acc, c| {
        T::combine(op, acc, T::map(op, data[offset_of(c, strides, offset)]))
    })
}

fn run_full<T: Scalar>(
    op: ReduceOp,
    ctx: &ExecContext,
    data: &[T],
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) -> T {
    let plan = ErasedReducePlan::compile(T::KIND, op, dims, strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(data, dims, strides, offset).unwrap();
    let mut output = [T::identity(op)];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();
    plan.execute(ctx, &mut dest, &source).unwrap();
    output[0]
}

struct FullCase {
    name: &'static str,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

fn full_cases(len: usize) -> Vec<FullCase> {
    let rows = 7usize;
    let cols = len / rows;
    vec![
        FullCase {
            name: "contiguous",
            dims: vec![len],
            strides: vec![1],
            offset: 0,
        },
        FullCase {
            name: "strided",
            dims: vec![len],
            strides: vec![3],
            offset: 1,
        },
        FullCase {
            name: "negative stride",
            dims: vec![len],
            strides: vec![-2],
            offset: 2 * (len as isize - 1),
        },
        FullCase {
            name: "transposed",
            dims: vec![rows, cols],
            strides: vec![cols as isize, 1],
            offset: 0,
        },
        FullCase {
            name: "padded columns",
            dims: vec![rows, cols],
            strides: vec![1, rows as isize + 2],
            offset: 0,
        },
        FullCase {
            name: "broadcast",
            dims: vec![rows, cols],
            strides: vec![0, 1],
            offset: 0,
        },
    ]
}

fn check_full<T: Scalar>(len: usize) {
    for &op in ops_for::<T>() {
        for case in full_cases(len) {
            let data = buffer::<T>(op, &case.dims, &case.strides, case.offset, None);
            let expected = naive_full(op, &data, &case.dims, &case.strides, case.offset);
            for (ctx_name, ctx) in contexts() {
                let got = run_full(op, &ctx, &data, &case.dims, &case.strides, case.offset);
                assert!(
                    T::same(got, expected),
                    "{op:?} {} {ctx_name} len={len}: got {got:?}, expected {expected:?}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn full_reduce_matches_naive_small_odd() {
    for len in [1usize, 7, 9, 63, 64, 65, 1001] {
        check_full::<f32>(len);
        check_full::<f64>(len);
        check_full::<i32>(len);
        check_full::<i64>(len);
        check_full::<Complex64>(len);
    }
}

#[test]
fn full_reduce_matches_naive_parallel_sizes() {
    check_full::<f64>(LARGE);
    check_full::<f32>(LARGE);
    check_full::<i64>(LARGE);
    check_full::<Complex64>(LARGE);
}

#[test]
fn full_reduce_propagates_nan_at_every_lane_position() {
    for len in [9usize, 65, LARGE] {
        for position in [0, 3, 7, len / 2, len - 1] {
            for op in ALL_OPS {
                let dims = [len];
                let strides = [1isize];
                let data = buffer::<f64>(op, &dims, &strides, 0, Some((position, f64::NAN)));
                for (ctx_name, ctx) in contexts() {
                    let got = run_full(op, &ctx, &data, &dims, &strides, 0);
                    assert!(
                        got.is_nan(),
                        "{op:?} {ctx_name} len={len} nan at {position}: {got}"
                    );
                }
            }
        }
    }
}

#[test]
fn full_reduce_signed_zero_and_infinities() {
    let dims = [37usize];
    let strides = [1isize];
    let zeros = vec![-0.0f64; 37];
    for (_, ctx) in contexts() {
        assert_eq!(
            run_full(ReduceOp::Max, &ctx, &zeros, &dims, &strides, 0).to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            run_full(ReduceOp::Min, &ctx, &zeros, &dims, &strides, 0).to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            run_full(ReduceOp::Product, &ctx, &zeros, &dims, &strides, 0).to_bits(),
            (-0.0f64).to_bits(),
            "37 negative zeros multiply to -0.0"
        );
    }
    let mut data = vec![1.0f64; 37];
    data[5] = f64::INFINITY;
    data[20] = f64::NEG_INFINITY;
    for (_, ctx) in contexts() {
        assert_eq!(
            run_full(ReduceOp::Max, &ctx, &data, &dims, &strides, 0),
            f64::INFINITY
        );
        assert_eq!(
            run_full(ReduceOp::Min, &ctx, &data, &dims, &strides, 0),
            f64::NEG_INFINITY
        );
        assert!(run_full(ReduceOp::Sum, &ctx, &data, &dims, &strides, 0).is_nan());
    }
}

#[test]
fn full_reduce_integer_overflow_wraps() {
    let dims = [LARGE];
    let strides = [1isize];
    let data = vec![i32::MAX; LARGE];
    let expected_sum = (0..LARGE).fold(0i32, |acc, _| acc.wrapping_add(i32::MAX));
    let expected_product = (0..LARGE).fold(1i32, |acc, _| acc.wrapping_mul(i32::MAX));
    for (_, ctx) in contexts() {
        assert_eq!(
            run_full(ReduceOp::Sum, &ctx, &data, &dims, &strides, 0),
            expected_sum
        );
        assert_eq!(
            run_full(ReduceOp::Product, &ctx, &data, &dims, &strides, 0),
            expected_product
        );
    }
}

#[test]
fn full_reduce_empty_writes_identity() {
    let dims = [0usize, 5];
    let strides = [1isize, 0];
    let data = [0.0f64];
    for op in ALL_OPS {
        for (_, ctx) in contexts() {
            let got = run_full(op, &ctx, &data, &dims, &strides, 0);
            assert!(f64::same(got, f64::identity(op)), "{op:?}: {got}");
        }
    }
}

#[test]
fn full_parallel_reduce_is_repeatable_for_inexact_values() {
    let dims = [LARGE];
    let strides = [1isize];
    let data: Vec<f64> = (0..LARGE).map(|i| (i as f64 * 0.37).sin()).collect();
    let ctx = ExecContext::max_threads(4).unwrap();
    let first = run_full(ReduceOp::Sum, &ctx, &data, &dims, &strides, 0);
    for _ in 0..4 {
        let again = run_full(ReduceOp::Sum, &ctx, &data, &dims, &strides, 0);
        assert_eq!(first.to_bits(), again.to_bits());
    }
    let naive: f64 = data.iter().sum();
    assert!((first - naive).abs() <= 1e-9 * LARGE as f64);
}

struct AxesCase {
    name: &'static str,
    src_dims: Vec<usize>,
    src_strides: Vec<isize>,
    src_offset: isize,
    dest_strides: Vec<isize>,
    axes: Vec<usize>,
}

/// Destination dims, and the source coordinate of output `out` at reduced
/// element `k` (reduced axes in caller order, first fastest).
fn axes_reference<T: Scalar>(op: ReduceOp, data: &[T], case: &AxesCase) -> (Vec<usize>, Vec<T>) {
    let kept: Vec<usize> = (0..case.src_dims.len())
        .filter(|a| !case.axes.contains(a))
        .collect();
    let dest_dims: Vec<usize> = kept.iter().map(|&a| case.src_dims[a]).collect();
    let reduced_dims: Vec<usize> = case.axes.iter().map(|&a| case.src_dims[a]).collect();
    let values = coords_of(&dest_dims)
        .iter()
        .map(|out| {
            coords_of(&reduced_dims)
                .iter()
                .fold(T::identity(op), |acc, red| {
                    let mut coord = vec![0usize; case.src_dims.len()];
                    for (&axis, &c) in kept.iter().zip(out) {
                        coord[axis] = c;
                    }
                    for (&axis, &c) in case.axes.iter().zip(red) {
                        coord[axis] = c;
                    }
                    let index = offset_of(&coord, &case.src_strides, case.src_offset);
                    T::combine(op, acc, T::map(op, data[index]))
                })
        })
        .collect();
    (dest_dims, values)
}

fn run_axes<T: Scalar>(
    op: ReduceOp,
    ctx: &ExecContext,
    data: &[T],
    case: &AxesCase,
    dest_dims: &[usize],
) -> Vec<T> {
    let plan = ErasedReducePlan::compile_axes(
        T::KIND,
        op,
        &case.src_dims,
        &case.src_strides,
        dest_dims,
        &case.dest_strides,
        &case.axes,
    )
    .unwrap();
    let source =
        ErasedRawStridedRef::from_slice(data, &case.src_dims, &case.src_strides, case.src_offset)
            .unwrap();
    let dest_len = coords_of(dest_dims)
        .iter()
        .map(|c| offset_of(c, &case.dest_strides, 0) + 1)
        .max()
        .unwrap_or(1);
    let mut output = vec![T::identity(op); dest_len];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, dest_dims, &case.dest_strides, 0).unwrap();
    plan.execute(ctx, &mut dest, &source).unwrap();
    coords_of(dest_dims)
        .iter()
        .map(|c| output[offset_of(c, &case.dest_strides, 0)])
        .collect()
}

fn axes_cases(outputs: usize) -> Vec<AxesCase> {
    let reduce = 37usize;
    vec![
        // Unit-stride reduced run of length >= 16: contiguous run kernel.
        AxesCase {
            name: "contiguous reduced axis, strided output",
            src_dims: vec![reduce, outputs],
            src_strides: vec![1, reduce as isize],
            src_offset: 0,
            dest_strides: vec![3],
            axes: vec![0],
        },
        // Unit-stride kept axis: output tiles with a remainder.
        AxesCase {
            name: "contiguous kept axis",
            src_dims: vec![outputs, reduce],
            src_strides: vec![1, outputs as isize],
            src_offset: 0,
            dest_strides: vec![1],
            axes: vec![1],
        },
        AxesCase {
            name: "contiguous kept axis, strided output",
            src_dims: vec![outputs, reduce],
            src_strides: vec![1, outputs as isize + 1],
            src_offset: 0,
            dest_strides: vec![2],
            axes: vec![1],
        },
        // No unit stride anywhere: sequential fold.
        AxesCase {
            name: "strided both",
            src_dims: vec![reduce, outputs],
            src_strides: vec![2, -(2 * reduce as isize)],
            src_offset: 2 * reduce as isize * (outputs as isize - 1),
            dest_strides: vec![1],
            axes: vec![0],
        },
        AxesCase {
            name: "broadcast reduced axis",
            src_dims: vec![reduce, outputs],
            src_strides: vec![0, 1],
            src_offset: 0,
            dest_strides: vec![1],
            axes: vec![0],
        },
        AxesCase {
            name: "broadcast kept axis",
            src_dims: vec![outputs, reduce],
            src_strides: vec![0, 1],
            src_offset: 0,
            dest_strides: vec![1],
            axes: vec![1],
        },
        // Two unfused kept axes: output blocks stop at the leading axis end.
        AxesCase {
            name: "unfused kept axes",
            src_dims: vec![outputs, 3, 19],
            src_strides: vec![1, outputs as isize + 1, 3 * (outputs as isize + 1)],
            src_offset: 0,
            dest_strides: vec![1, outputs as isize],
            axes: vec![2],
        },
        AxesCase {
            name: "rank3 reordered axes",
            src_dims: vec![5, outputs, 3],
            src_strides: vec![1, 5, 5 * outputs as isize],
            src_offset: 0,
            dest_strides: vec![1],
            axes: vec![2, 0],
        },
    ]
}

fn check_axes<T: Scalar>(outputs: usize) {
    for &op in ops_for::<T>() {
        for case in axes_cases(outputs) {
            let data = buffer::<T>(op, &case.src_dims, &case.src_strides, case.src_offset, None);
            let (dest_dims, expected) = axes_reference(op, &data, &case);
            for (ctx_name, ctx) in contexts() {
                let got = run_axes(op, &ctx, &data, &case, &dest_dims);
                for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        T::same(g, e),
                        "{op:?} {} {ctx_name} outputs={outputs} out[{index}]: got {g:?}, expected {e:?}",
                        case.name
                    );
                }
            }
        }
    }
}

#[test]
fn axes_reduce_matches_naive_small_odd() {
    for outputs in [1usize, 5, 16, 17, 33, 51] {
        check_axes::<f32>(outputs);
        check_axes::<f64>(outputs);
        check_axes::<i32>(outputs);
        check_axes::<i64>(outputs);
        check_axes::<Complex64>(outputs);
    }
}

#[test]
fn axes_reduce_matches_naive_parallel_sizes() {
    // Parallel worker ranges start in the middle of output tiles.
    let outputs = (1 << 15) + 37;
    check_axes::<f64>(outputs);
    check_axes::<i32>(outputs);
}

#[test]
fn axes_reduce_propagates_nan() {
    for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
        for case in axes_cases(33) {
            let mut data =
                buffer::<f64>(op, &case.src_dims, &case.src_strides, case.src_offset, None);
            // Poison the element used by output 20 at reduced position 30.
            let (dest_dims, _) = axes_reference(op, &data, &case);
            let kept: Vec<usize> = (0..case.src_dims.len())
                .filter(|a| !case.axes.contains(a))
                .collect();
            let mut coord = vec![0usize; case.src_dims.len()];
            coord[kept[0]] = 20;
            coord[case.axes[0]] = case.src_dims[case.axes[0]] - 1;
            let poisoned = offset_of(&coord, &case.src_strides, case.src_offset);
            data[poisoned] = f64::NAN;
            let (_, expected) = axes_reference(op, &data, &case);
            for (ctx_name, ctx) in contexts() {
                let got = run_axes(op, &ctx, &data, &case, &dest_dims);
                for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        f64::same(g, e),
                        "{op:?} {} {ctx_name} out[{index}]: got {g}, expected {e}",
                        case.name
                    );
                }
                assert!(got[20].is_nan(), "{op:?} {}", case.name);
            }
        }
    }
}

/// Output tiles and the sequential fold keep the caller reduced-axis order
/// exactly, so they must agree bitwise with a sequential fold even for
/// inexact values, in every context.
#[test]
fn tiled_and_sequential_axes_keep_sequential_order() {
    for (outputs, reduce) in [(37usize, 9usize), ((1 << 15) + 37, 3)] {
        let src_dims = vec![outputs, reduce];
        let src_strides = vec![1isize, outputs as isize];
        let data: Vec<f64> = (0..outputs * reduce)
            .map(|i| (i as f64 * 0.731).sin() * 1e3)
            .collect();
        for axes_first_strided in [false, true] {
            let case = if axes_first_strided {
                AxesCase {
                    name: "sequential",
                    src_dims: vec![reduce, outputs],
                    src_strides: vec![outputs as isize, 1],
                    src_offset: 0,
                    dest_strides: vec![1],
                    axes: vec![0],
                }
            } else {
                AxesCase {
                    name: "tiles",
                    src_dims: src_dims.clone(),
                    src_strides: src_strides.clone(),
                    src_offset: 0,
                    dest_strides: vec![1],
                    axes: vec![1],
                }
            };
            let (dest_dims, expected) = axes_reference(ReduceOp::Sum, &data, &case);
            for (ctx_name, ctx) in contexts() {
                let got = run_axes(ReduceOp::Sum, &ctx, &data, &case, &dest_dims);
                for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        e.to_bits(),
                        "{} {ctx_name} out[{index}]",
                        case.name
                    );
                }
            }
        }
    }
}

/// Axis reductions never depend on the context: the contiguous run kernel
/// gives the same bits serially and with any thread count.
#[test]
fn contiguous_run_axes_are_context_independent() {
    let reduce = 41usize;
    let outputs = (1 << 15) / 8 + 3;
    let case = AxesCase {
        name: "runs",
        src_dims: vec![reduce, outputs],
        src_strides: vec![1, reduce as isize],
        src_offset: 0,
        dest_strides: vec![1],
        axes: vec![0],
    };
    let data: Vec<f64> = (0..reduce * outputs)
        .map(|i| (i as f64 * 0.917).cos())
        .collect();
    let (dest_dims, naive) = axes_reference(ReduceOp::Sum, &data, &case);
    let serial = run_axes(
        ReduceOp::Sum,
        &ExecContext::serial(),
        &data,
        &case,
        &dest_dims,
    );
    for threads in [1usize, 2, 3, 4] {
        let ctx = ExecContext::max_threads(threads).unwrap();
        let got = run_axes(ReduceOp::Sum, &ctx, &data, &case, &dest_dims);
        for (index, (&g, &s)) in got.iter().zip(&serial).enumerate() {
            assert_eq!(g.to_bits(), s.to_bits(), "threads={threads} out[{index}]");
        }
    }
    for (&s, &n) in serial.iter().zip(&naive) {
        assert!((s - n).abs() <= 1e-12 * reduce as f64);
    }
}

/// Reducing every axis through `compile_axes` uses the full traversal and
/// matches `compile`.
#[test]
fn all_axes_reduction_matches_full_plan() {
    let dims = [7usize, 9, 5];
    let strides = [45isize, 5, 1];
    let data: Vec<f64> = (0..315).map(|i| (i as f64 * 0.3).sin()).collect();
    for op in ALL_OPS {
        let full = run_full(op, &ExecContext::serial(), &data, &dims, &strides, 0);
        let plan = ErasedReducePlan::compile_axes(
            KernelDType::F64,
            op,
            &dims,
            &strides,
            &[],
            &[],
            &[2, 0, 1],
        )
        .unwrap();
        let source = ErasedRawStridedRef::from_slice(&data, &dims, &strides, 0).unwrap();
        let mut output = [0.0f64];
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();
        plan.execute(&ExecContext::serial(), &mut dest, &source)
            .unwrap();
        assert_eq!(output[0].to_bits(), full.to_bits(), "{op:?}");
    }
}

/// Column-major matrix reduced along its strided axis: the output block
/// kernel sweeps contiguous columns into a block of outputs. Blocks cross the
/// 512 output limit, and the parallel context splits the rows because the
/// work (rows times columns) exceeds the threshold although the output
/// count does not.
#[test]
fn strided_axis_of_column_major_matrix_uses_column_sweep() {
    for (rows, cols) in [(600usize, 61usize), (2048, 33)] {
        for &op in &ALL_OPS {
            let case = AxesCase {
                name: "matrix axis 1",
                src_dims: vec![rows, cols],
                src_strides: vec![1, rows as isize],
                src_offset: 0,
                dest_strides: vec![1],
                axes: vec![1],
            };
            let data = buffer::<f64>(op, &case.src_dims, &case.src_strides, 0, None);
            let (dest_dims, expected) = axes_reference(op, &data, &case);
            for (ctx_name, ctx) in contexts() {
                let got = run_axes(op, &ctx, &data, &case, &dest_dims);
                for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        f64::same(g, e),
                        "{op:?} {rows}x{cols} {ctx_name} out[{index}]: got {g}, expected {e}"
                    );
                }
            }
        }
        // Inexact values: the block kernel keeps the sequential order, so
        // every context agrees bitwise with the naive fold.
        let case = AxesCase {
            name: "matrix axis 1 inexact",
            src_dims: vec![rows, cols],
            src_strides: vec![1, rows as isize],
            src_offset: 0,
            dest_strides: vec![2],
            axes: vec![1],
        };
        let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64 * 0.61).sin()).collect();
        let (dest_dims, expected) = axes_reference(ReduceOp::Sum, &data, &case);
        for threads in [1usize, 2, 3, 4] {
            let ctx = ExecContext::max_threads(threads).unwrap();
            let got = run_axes(ReduceOp::Sum, &ctx, &data, &case, &dest_dims);
            for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(g.to_bits(), e.to_bits(), "threads={threads} out[{index}]");
            }
        }
    }
}
