//! Differential tests for the two erased product variants.
//!
//! `ReduceOp::Product` is the reassociated (PyTorch-compatible) product and
//! `ReduceOp::StableProduct` is the exponent-tracked product with a
//! source-order fallback for values it cannot represent. Both are checked
//! against a high-precision reference or against the sequential fold over the
//! full-reduction and axis-reduction routes, in a serial context and a verified
//! two-worker pool.

use num_complex::Complex64;
use strided_basic::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan, ExecContext, KernelDType,
    KernelStorageElement, ReduceOp,
};

/// Largest tensor used per case; sized past `MINTHREADLENGTH` so the parallel
/// axis route really partitions outputs.
const LARGE_AXIS_OUTPUTS: usize = (1 << 15) + 37;

fn run_full<T: KernelStorageElement + Default>(
    dtype: KernelDType,
    op: ReduceOp,
    ctx: &ExecContext,
    data: &[T],
    dims: &[usize],
    strides: &[isize],
) -> T {
    let plan = ErasedReducePlan::compile(dtype, op, dims, strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(data, dims, strides, 0).unwrap();
    let mut output = [T::default()];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();
    plan.execute(ctx, &mut dest, &source).unwrap();
    output[0]
}

#[allow(clippy::too_many_arguments)]
fn run_axes<T: KernelStorageElement + Default>(
    dtype: KernelDType,
    op: ReduceOp,
    ctx: &ExecContext,
    data: &[T],
    dims: &[usize],
    strides: &[isize],
    dest_dims: &[usize],
    dest_strides: &[isize],
    axes: &[usize],
) -> Vec<T> {
    let plan =
        ErasedReducePlan::compile_axes(dtype, op, dims, strides, dest_dims, dest_strides, axes)
            .unwrap();
    let source = ErasedRawStridedRef::from_slice(data, dims, strides, 0).unwrap();
    let mut output = vec![T::default(); dest_dims.iter().product::<usize>().max(1)];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, dest_dims, dest_strides, 0).unwrap();
    plan.execute(ctx, &mut dest, &source).unwrap();
    output
}

/// Bitwise comparison that treats any two NaNs as equal.
fn same_f64(lhs: f64, rhs: f64) -> bool {
    (lhs.is_nan() && rhs.is_nan()) || lhs.to_bits() == rhs.to_bits()
}
fn same_f32(lhs: f32, rhs: f32) -> bool {
    (lhs.is_nan() && rhs.is_nan()) || lhs.to_bits() == rhs.to_bits()
}

/// `m * 2^e` for `f64`, matching the implementation's overflow/underflow
/// behaviour. Test-only reference helper.
fn ldexp_ref(mut m: f64, mut e: i64) -> f64 {
    const TWO_POW_M1022: u64 = 1u64 << 52;
    const TWO_POW_1023: u64 = 2046u64 << 52;
    while e > 1023 {
        m *= f64::from_bits(TWO_POW_1023);
        e -= 1023;
        if m.is_infinite() {
            return m;
        }
    }
    while e < -1022 {
        m *= f64::from_bits(TWO_POW_M1022);
        e += 1022;
        if m == 0.0 {
            return m;
        }
    }
    m * f64::from_bits(((e + 1023) as u64) << 52)
}

/// High-precision reference product of the `f64` factors, tracking the exact
/// integer exponent. Overflows and underflows are reported exactly.
fn reference_product_f64(values: &[f64]) -> f64 {
    let mut m = 1.0f64;
    let mut e = 0i64;
    for &value in values {
        let bits = value.to_bits();
        let raw = ((bits >> 52) & 0x7ff) as i64;
        m *= f64::from_bits((bits & !(0x7ffu64 << 52)) | (1023u64 << 52));
        e += raw - 1023;
        let (mantissa, exponent, _) = split_ref_f64(m);
        m = mantissa;
        e += exponent;
    }
    ldexp_ref(m, e)
}

fn reference_product_f32(values: &[f32]) -> f32 {
    let mut m = 1.0f64;
    let mut e = 0i64;
    for &value in values {
        let bits = value.to_bits();
        let raw = ((bits >> 23) & 0xff) as i64;
        m *= f32::from_bits((bits & !(0xffu32 << 23)) | (127u32 << 23)) as f64;
        e += raw - 127;
        let (mantissa, exponent, _) = split_ref_f64(m);
        m = mantissa;
        e += exponent;
    }
    ldexp_ref(m, e) as f32
}

/// High-precision reference along one axis, used when a plain fold would
/// underflow.
fn axis_reference_hp_f64(data: &[f64], dims: &[usize], strides: &[isize], axis: usize) -> Vec<f64> {
    let kept: Vec<usize> = (0..dims.len()).filter(|&a| a != axis).collect();
    let outputs: usize = kept.iter().map(|&a| dims[a]).product::<usize>().max(1);
    (0..outputs)
        .map(|mut index| {
            let mut coords = vec![0usize; dims.len()];
            for &a in &kept {
                coords[a] = index % dims[a];
                index /= dims[a];
            }
            let mut factors = Vec::with_capacity(dims[axis]);
            for red in 0..dims[axis] {
                coords[axis] = red;
                let offset: isize = coords
                    .iter()
                    .zip(strides)
                    .map(|(&c, &s)| c as isize * s)
                    .sum();
                factors.push(data[offset as usize]);
            }
            reference_product_f64(&factors)
        })
        .collect()
}

fn axis_reference_hp_f32(data: &[f32], dims: &[usize], strides: &[isize], axis: usize) -> Vec<f32> {
    let kept: Vec<usize> = (0..dims.len()).filter(|&a| a != axis).collect();
    let outputs: usize = kept.iter().map(|&a| dims[a]).product::<usize>().max(1);
    (0..outputs)
        .map(|mut index| {
            let mut coords = vec![0usize; dims.len()];
            for &a in &kept {
                coords[a] = index % dims[a];
                index /= dims[a];
            }
            let mut factors = Vec::with_capacity(dims[axis]);
            for red in 0..dims[axis] {
                coords[axis] = red;
                let offset: isize = coords
                    .iter()
                    .zip(strides)
                    .map(|(&c, &s)| c as isize * s)
                    .sum();
                factors.push(data[offset as usize]);
            }
            reference_product_f32(&factors)
        })
        .collect()
}

fn split_ref_f64(value: f64) -> (f64, i64, bool) {
    let bits = value.to_bits();
    let raw = ((bits >> 52) & 0x7ff) as i64;
    (
        f64::from_bits((bits & !(0x7ffu64 << 52)) | (1023u64 << 52)),
        raw - 1023,
        raw == 0 || raw == 0x7ff,
    )
}

/// Deterministic LCG so the tests do not depend on the `rand` stream.
fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn wild_f64(state: &mut u64) -> f64 {
    let word = next(state);
    let sign = word >> 63;
    let exponent = (((word >> 12) % 601) as i64) - 300; // [-300, 300]
    let mantissa = word & ((1 << 52) - 1);
    f64::from_bits((sign << 63) | (((exponent + 1023) as u64) << 52) | mantissa)
}

fn wild_f32(state: &mut u64) -> f32 {
    let word = next(state) as u32;
    let sign = word >> 31;
    let exponent = (((word >> 13) % 121) as i32) - 60; // [-60, 60]
    let mantissa = word & ((1 << 23) - 1);
    f32::from_bits((sign << 31) | (((exponent + 127) as u32) << 23) | mantissa)
}

/// Runs `f` in a pool holding exactly two workers and verifies both the pool
/// and the context budget before executing.
#[cfg(feature = "parallel")]
fn with_two_workers<R: Send>(f: impl FnOnce(&ExecContext) -> R + Send) -> R {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    ExecContext::run_on(&pool, 2, |ctx| {
        assert_eq!(rayon::current_num_threads(), 2);
        assert_eq!(ctx.max_threads_limit().map(|value| value.get()), Some(2));
        f(&ctx)
    })
    .unwrap()
}

/// The #2150 shape: alternating `2.0`/`0.5`. The stable product must be exactly
/// `1.0` because every factor is a power of two and the exponent sum is zero,
/// while a raw partial product can run away.
#[test]
fn stable_product_alternating_two_and_half_is_exactly_one() {
    for count in [1000usize, 1_000_000] {
        let f64_input: Vec<f64> = (0..count)
            .map(|index| if index % 2 == 0 { 2.0 } else { 0.5 })
            .collect();
        let f32_input: Vec<f32> = (0..count)
            .map(|index| if index % 2 == 0 { 2.0 } else { 0.5 })
            .collect();
        let dims = [count];
        let strides = [1isize];

        let serial = ExecContext::serial();
        assert_eq!(
            run_full(
                KernelDType::F64,
                ReduceOp::StableProduct,
                &serial,
                &f64_input,
                &dims,
                &strides
            ),
            1.0
        );
        assert_eq!(
            run_full(
                KernelDType::F32,
                ReduceOp::StableProduct,
                &serial,
                &f32_input,
                &dims,
                &strides
            ),
            1.0
        );

        #[cfg(feature = "parallel")]
        with_two_workers(|ctx| {
            assert_eq!(
                run_full(
                    KernelDType::F64,
                    ReduceOp::StableProduct,
                    ctx,
                    &f64_input,
                    &dims,
                    &strides
                ),
                1.0
            );
            assert_eq!(
                run_full(
                    KernelDType::F32,
                    ReduceOp::StableProduct,
                    ctx,
                    &f32_input,
                    &dims,
                    &strides
                ),
                1.0
            );
        });
    }
}

/// Factors spanning many orders of magnitude, compared to the high-precision
/// reference within a few ulps per factor, on the full and both axis routes.
#[test]
fn stable_product_extreme_exponents_match_reference() {
    let mut state = 0x0f0e_0d0c_0b0a_0908u64;
    let f64_input: Vec<f64> = (0..2048).map(|_| wild_f64(&mut state)).collect();
    let f32_input: Vec<f32> = (0..2048).map(|_| wild_f32(&mut state)).collect();
    let dims = [64usize, 32];
    let strides = [1isize, 64];

    let check_f64 = |ctx: &ExecContext, label: &str| {
        let expected = reference_product_f64(&f64_input);
        let got = run_full(
            KernelDType::F64,
            ReduceOp::StableProduct,
            ctx,
            &f64_input,
            &dims,
            &strides,
        );
        assert!(
            same_f64(got, expected) || relative_close_f64(got, expected, f64_input.len()),
            "f64 full {label}: got {got:e}, reference {expected:e}"
        );
    };
    let check_f32 = |ctx: &ExecContext, label: &str| {
        let expected = reference_product_f32(&f32_input);
        let got = run_full(
            KernelDType::F32,
            ReduceOp::StableProduct,
            ctx,
            &f32_input,
            &dims,
            &strides,
        );
        assert!(
            same_f32(got, expected) || relative_close_f32(got, expected, f32_input.len()),
            "f32 full {label}: got {got:e}, reference {expected:e}"
        );
    };

    check_f64(&ExecContext::serial(), "serial");
    check_f32(&ExecContext::serial(), "serial");
    #[cfg(feature = "parallel")]
    with_two_workers(|ctx| {
        check_f64(ctx, "2T");
        check_f32(ctx, "2T");
    });

    // Axis routes on a matrix of the same values.
    let matrix_f64: Vec<f64> = f64_input
        .iter()
        .copied()
        .chain(f64_input.iter().copied())
        .collect();
    let matrix_f32: Vec<f32> = f32_input
        .iter()
        .copied()
        .chain(f32_input.iter().copied())
        .collect();
    let mdims = [32usize, 128];
    let mstrides = [1isize, 32];
    for axis in [0usize, 1] {
        let kept = if axis == 0 { mdims[1] } else { mdims[0] };
        let expected: Vec<f64> = axis_reference_hp_f64(&matrix_f64, &mdims, &mstrides, axis);
        let got = run_axes(
            KernelDType::F64,
            ReduceOp::StableProduct,
            &ExecContext::serial(),
            &matrix_f64,
            &mdims,
            &mstrides,
            &[kept],
            &[1isize],
            &[axis],
        );
        for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                same_f64(g, e) || relative_close_f64(g, e, mdims[axis]),
                "f64 axis {axis} out[{index}]: got {g:e}, reference {e:e}"
            );
        }
        let expected32: Vec<f32> = axis_reference_hp_f32(&matrix_f32, &mdims, &mstrides, axis);
        let got32 = run_axes(
            KernelDType::F32,
            ReduceOp::StableProduct,
            &ExecContext::serial(),
            &matrix_f32,
            &mdims,
            &mstrides,
            &[kept],
            &[1isize],
            &[axis],
        );
        for (index, (&g, &e)) in got32.iter().zip(&expected32).enumerate() {
            assert!(
                same_f32(g, e) || relative_close_f32(g, e, mdims[axis]),
                "f32 axis {axis} out[{index}]: got {g:e}, reference {e:e}"
            );
        }
    }

    // A parallel axis route over long reductions.
    let parallel_dims = [31usize, LARGE_AXIS_OUTPUTS];
    let parallel_strides = [1isize, 31];
    let parallel_f64: Vec<f64> = (0..31 * LARGE_AXIS_OUTPUTS)
        .map(|_| wild_f64(&mut state))
        .collect();
    let parallel_f32: Vec<f32> = (0..31 * LARGE_AXIS_OUTPUTS)
        .map(|_| wild_f32(&mut state))
        .collect();
    let _expected: Vec<f64> =
        axis_reference_hp_f64(&parallel_f64, &parallel_dims, &parallel_strides, 0);
    let _expected32: Vec<f32> =
        axis_reference_hp_f32(&parallel_f32, &parallel_dims, &parallel_strides, 0);
    #[cfg(feature = "parallel")]
    with_two_workers(|ctx| {
        let got = run_axes(
            KernelDType::F64,
            ReduceOp::StableProduct,
            ctx,
            &parallel_f64,
            &parallel_dims,
            &parallel_strides,
            &[LARGE_AXIS_OUTPUTS],
            &[1isize],
            &[0],
        );
        for (index, (&g, &e)) in got.iter().zip(&_expected).enumerate() {
            assert!(
                same_f64(g, e) || relative_close_f64(g, e, 31),
                "f64 parallel axis out[{index}]: got {g:e}, reference {e:e}"
            );
        }
        let got32 = run_axes(
            KernelDType::F32,
            ReduceOp::StableProduct,
            ctx,
            &parallel_f32,
            &parallel_dims,
            &parallel_strides,
            &[LARGE_AXIS_OUTPUTS],
            &[1isize],
            &[0],
        );
        for (index, (&g, &e)) in got32.iter().zip(&_expected32).enumerate() {
            assert!(
                same_f32(g, e) || relative_close_f32(g, e, 31),
                "f32 parallel axis out[{index}]: got {g:e}, reference {e:e}"
            );
        }
    });
}

/// A strided full reduction and a parallel output-block axis reduction cover
/// the remaining layout strategies.
#[test]
fn stable_product_strided_and_output_block_routes() {
    // Gapped column-major matrix: no unit stride, so the full traversal splits
    // into many strided runs.
    let dims = [64usize, 32];
    let strides = [3isize, 192];
    let span = (dims[0] - 1) * strides[0] as usize + (dims[1] - 1) * strides[1] as usize + 1;
    let mut storage = vec![1.0f64; span];
    let mut factors = Vec::with_capacity(dims[0] * dims[1]);
    let mut state = 0x1122_3344_5566_7788u64;
    for column in 0..dims[1] {
        for row in 0..dims[0] {
            let value = wild_f64(&mut state);
            storage[row * 3 + column * 192] = value;
            factors.push(value);
        }
    }
    let expected = reference_product_f64(&factors);
    let got = run_full(
        KernelDType::F64,
        ReduceOp::StableProduct,
        &ExecContext::serial(),
        &storage,
        &dims,
        &strides,
    );
    assert!(
        same_f64(got, expected) || relative_close_f64(got, expected, factors.len()),
        "strided full: got {got:e}, reference {expected:e}"
    );

    // Column-major matrix reduced along its strided axis: the output-block
    // kernel, split across a two-worker pool.
    let rows = LARGE_AXIS_OUTPUTS;
    let columns = 31usize;
    let dims = [rows, columns];
    let strides = [1isize, rows as isize];
    let data: Vec<f64> = (0..rows * columns).map(|_| wild_f64(&mut state)).collect();
    let _expected = axis_reference_hp_f64(&data, &dims, &strides, 1);
    #[cfg(feature = "parallel")]
    with_two_workers(|ctx| {
        let got = run_axes(
            KernelDType::F64,
            ReduceOp::StableProduct,
            ctx,
            &data,
            &dims,
            &strides,
            &[rows],
            &[1isize],
            &[1],
        );
        for (index, (&g, &e)) in got.iter().zip(&_expected).enumerate() {
            assert!(
                same_f64(g, e) || relative_close_f64(g, e, columns),
                "output-block axis out[{index}]: got {g:e}, reference {e:e}"
            );
        }
    });
}

fn relative_close_f64(got: f64, expected: f64, count: usize) -> bool {
    if !got.is_finite() || !expected.is_finite() || expected == 0.0 {
        return got == expected;
    }
    (got - expected).abs() <= 8.0 * f64::EPSILON * count as f64 * expected.abs()
}

fn relative_close_f32(got: f32, expected: f32, count: usize) -> bool {
    if !got.is_finite() || !expected.is_finite() || expected == 0.0 {
        return got == expected;
    }
    (got - expected).abs() <= 8.0 * f32::EPSILON * count as f64 as f32 * expected.abs()
}

/// Sequential-fold reference along one axis; reduced axes in increasing order.
fn axis_reference_f64(data: &[f64], dims: &[usize], strides: &[isize], axis: usize) -> Vec<f64> {
    let kept: Vec<usize> = (0..dims.len()).filter(|&a| a != axis).collect();
    let outputs: usize = kept.iter().map(|&a| dims[a]).product::<usize>().max(1);
    (0..outputs)
        .map(|mut index| {
            let mut coords = vec![0usize; dims.len()];
            for &a in &kept {
                coords[a] = index % dims[a];
                index /= dims[a];
            }
            let mut acc = 1.0f64;
            for red in 0..dims[axis] {
                coords[axis] = red;
                let offset: isize = coords
                    .iter()
                    .zip(strides)
                    .map(|(&c, &s)| c as isize * s)
                    .sum();
                acc *= data[offset as usize];
            }
            acc
        })
        .collect()
}

fn axis_reference_f32(data: &[f32], dims: &[usize], strides: &[isize], axis: usize) -> Vec<f32> {
    let kept: Vec<usize> = (0..dims.len()).filter(|&a| a != axis).collect();
    let outputs: usize = kept.iter().map(|&a| dims[a]).product::<usize>().max(1);
    (0..outputs)
        .map(|mut index| {
            let mut coords = vec![0usize; dims.len()];
            for &a in &kept {
                coords[a] = index % dims[a];
                index /= dims[a];
            }
            let mut acc = 1.0f32;
            for red in 0..dims[axis] {
                coords[axis] = red;
                let offset: isize = coords
                    .iter()
                    .zip(strides)
                    .map(|(&c, &s)| c as isize * s)
                    .sum();
                acc *= data[offset as usize];
            }
            acc
        })
        .collect()
}

/// Zero, infinity and NaN factors must propagate exactly like the sequential
/// fold: the stable accumulator cannot represent them and recomputes the output
/// in source order.
#[test]
fn stable_product_specials_match_ordered_fold() {
    let cases: [Vec<f64>; 6] = [
        vec![-0.0, 2.0, 3.0, -4.0],
        vec![f64::INFINITY, 2.0, -3.0],
        vec![f64::NEG_INFINITY, -2.0],
        vec![0.0, f64::INFINITY, 2.0],
        vec![f64::NAN, 2.0, 3.0],
        vec![2.0, f64::INFINITY, 0.0, -1.0],
    ];
    for case in &cases {
        let expected = case.iter().copied().fold(1.0f64, |acc, value| acc * value);
        let dims = [case.len()];
        let strides = [1isize];
        let serial = run_full(
            KernelDType::F64,
            ReduceOp::StableProduct,
            &ExecContext::serial(),
            case,
            &dims,
            &strides,
        );
        assert!(
            same_f64(serial, expected),
            "f64 special full: got {serial}, expected {expected}"
        );
        #[cfg(feature = "parallel")]
        with_two_workers(|ctx| {
            let got = run_full(
                KernelDType::F64,
                ReduceOp::StableProduct,
                ctx,
                case,
                &dims,
                &strides,
            );
            assert!(
                same_f64(got, expected),
                "f64 special 2T: {got} vs {expected}"
            );
        });
    }

    // The same specials on an axis route: one row per special pattern.
    let patterns = cases.len();
    let columns = 37usize;
    let mut data = vec![1.0f64; patterns * columns];
    for (row, case) in cases.iter().enumerate() {
        for (column, &value) in case.iter().enumerate() {
            data[column * patterns + row] = value;
        }
    }
    let dims = [patterns, columns];
    let strides = [1isize, patterns as isize];
    let expected = axis_reference_f64(&data, &dims, &strides, 1);
    let got = run_axes(
        KernelDType::F64,
        ReduceOp::StableProduct,
        &ExecContext::serial(),
        &data,
        &dims,
        &strides,
        &[patterns],
        &[1isize],
        &[1],
    );
    for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
        assert!(same_f64(g, e), "f64 special axis out[{index}]: {g} vs {e}");
    }

    // `f32` full and axis routes.
    for case in &cases {
        let case32: Vec<f32> = case.iter().map(|&value| value as f32).collect();
        let expected = case32
            .iter()
            .copied()
            .fold(1.0f32, |acc, value| acc * value);
        let dims = [case32.len()];
        let strides = [1isize];
        let got = run_full(
            KernelDType::F32,
            ReduceOp::StableProduct,
            &ExecContext::serial(),
            &case32,
            &dims,
            &strides,
        );
        assert!(
            same_f32(got, expected),
            "f32 special full: got {got}, expected {expected}"
        );
    }
    let data32: Vec<f32> = data.iter().map(|&value| value as f32).collect();
    let expected32 = axis_reference_f32(&data32, &dims, &strides, 1);
    let got32 = run_axes(
        KernelDType::F32,
        ReduceOp::StableProduct,
        &ExecContext::serial(),
        &data32,
        &dims,
        &strides,
        &[patterns],
        &[1isize],
        &[1],
    );
    for (index, (&g, &e)) in got32.iter().zip(&expected32).enumerate() {
        assert!(same_f32(g, e), "f32 special axis out[{index}]: {g} vs {e}");
    }
}

/// Genuine subnormal factors force the source-order fallback; the result must
/// match the sequential fold bit for bit (apart from NaN payload).
#[test]
fn stable_product_subnormals_match_ordered_fold() {
    let smallest = f64::from_bits(1);
    let mut state = 0xdead_beef_cafe_babeu64;
    let cases = [
        vec![smallest; 65],
        vec![smallest, 2.0, smallest, 2.0, smallest],
        (0..1000)
            .map(|_| {
                if next(&mut state) % 2 == 0 {
                    smallest
                } else {
                    3.0
                }
            })
            .collect(),
    ];
    for case in &cases {
        let expected = case.iter().copied().fold(1.0f64, |acc, value| acc * value);
        let dims = [case.len()];
        let strides = [1isize];
        let got = run_full(
            KernelDType::F64,
            ReduceOp::StableProduct,
            &ExecContext::serial(),
            case,
            &dims,
            &strides,
        );
        assert!(
            same_f64(got, expected),
            "subnormal full: {got} vs {expected}"
        );
    }

    // Subnormal f32 factors.
    let tiny = f32::from_bits(1);
    let case: Vec<f32> = (0..257)
        .map(|index| if index % 2 == 0 { tiny } else { 1.5 })
        .collect();
    let expected = case.iter().copied().fold(1.0f32, |acc, value| acc * value);
    let got = run_full(
        KernelDType::F32,
        ReduceOp::StableProduct,
        &ExecContext::serial(),
        &case,
        &[case.len()],
        &[1isize],
    );
    assert!(
        same_f32(got, expected),
        "subnormal f32: {got} vs {expected}"
    );
}

/// The reassociated product is not required to match the fold bitwise, but on
/// in-range normal inputs it must stay within a rounding-scale relative error.
#[test]
fn reassociated_product_matches_ordered_fold_on_normals() {
    let mut state = 0x0102_0304_0506_0708u64;
    let f64_input: Vec<f64> = (0..2048)
        .map(|_| 1.0 + (next(&mut state) % 2001) as f64 * 1e-4)
        .collect();
    let f32_input: Vec<f32> = (0..2048)
        .map(|_| 1.0 + (next(&mut state) % 2001) as f32 * 1e-4)
        .collect();
    let dims = [64usize, 32];
    let strides = [1isize, 64];

    let expected = f64_input
        .iter()
        .copied()
        .fold(1.0f64, |acc, value| acc * value);
    let got = run_full(
        KernelDType::F64,
        ReduceOp::Product,
        &ExecContext::serial(),
        &f64_input,
        &dims,
        &strides,
    );
    assert!(
        relative_close_f64(got, expected, f64_input.len()),
        "f64 reassociated full: {got} vs {expected}"
    );

    let expected32 = f32_input
        .iter()
        .copied()
        .fold(1.0f32, |acc, value| acc * value);
    let got32 = run_full(
        KernelDType::F32,
        ReduceOp::Product,
        &ExecContext::serial(),
        &f32_input,
        &dims,
        &strides,
    );
    assert!(
        relative_close_f32(got32, expected32, f32_input.len()),
        "f32 reassociated full: {got32} vs {expected32}"
    );

    // Axis route.
    let mdims = [32usize, 64];
    let mstrides = [1isize, 32];
    let expected = axis_reference_f64(&f64_input, &mdims, &mstrides, 1);
    let got = run_axes(
        KernelDType::F64,
        ReduceOp::Product,
        &ExecContext::serial(),
        &f64_input,
        &mdims,
        &mstrides,
        &[32],
        &[1isize],
        &[1],
    );
    for (index, (&g, &e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            relative_close_f64(g, e, 64),
            "f64 reassociated axis out[{index}]: {g} vs {e}"
        );
    }

    // Integer and empty reductions are exact regardless of order.
    let integers = [i64::MAX, 3, -5, 7, i64::MIN, 2];
    let expected = integers
        .iter()
        .fold(1i64, |acc, &value| acc.wrapping_mul(value));
    let got = run_full(
        KernelDType::I64,
        ReduceOp::Product,
        &ExecContext::serial(),
        &integers,
        &[integers.len()],
        &[1isize],
    );
    assert_eq!(got, expected);
    let got = run_full(
        KernelDType::F64,
        ReduceOp::Product,
        &ExecContext::serial(),
        &[] as &[f64],
        &[0usize],
        &[1isize],
    );
    assert_eq!(got, 1.0);
}

/// A complex product keeps the source order: a power-of-two factor sequence
/// that cancels to one is exact, and values with signed-zero components still
/// match the fold.
#[test]
fn stable_product_complex_is_source_order() {
    let input: Vec<Complex64> = (0..300)
        .map(|index| {
            if index % 2 == 0 {
                Complex64::new(2.0, 0.0)
            } else {
                Complex64::new(0.5, 0.0)
            }
        })
        .collect();
    let expected = input
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, &v| acc * v);
    let got = run_full(
        KernelDType::C64,
        ReduceOp::StableProduct,
        &ExecContext::serial(),
        &input,
        &[input.len()],
        &[1isize],
    );
    assert_eq!((got.re, got.im), (expected.re, expected.im));
}
