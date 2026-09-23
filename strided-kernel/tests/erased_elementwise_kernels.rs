//! Per-op elementwise kernels against a naive reference.
//!
//! The erased map, zip, compare, select and clamp entries match their op once
//! per call and replay a loop specialized for that op. These tests pin every
//! op on contiguous, strided and broadcast layouts, under a serial and a
//! bounded parallel context, with odd lengths (remainder loops) and a length
//! above the threading threshold. Special values cover NaN, signed zeros,
//! integer overflow and complex inputs.

use core::mem::MaybeUninit;
use num_complex::Complex64;
use strided_kernel::{
    erased_clamp_into_uninit, erased_compare_into_uninit, erased_map_into,
    erased_select_into_uninit, erased_zip_into, CompareOp, ErasedMapOp, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedZipOp, ExecContext,
    KernelDType, KernelStorageElement,
};

/// Larger than `MINTHREADLENGTH` so that parallel contexts split the work.
const LARGE: usize = (1 << 15) + 65;
const LENGTHS: [usize; 5] = [1, 7, 17, 67, LARGE];

fn contexts() -> Vec<(&'static str, ExecContext)> {
    vec![
        ("serial", ExecContext::serial()),
        ("max_threads(4)", ExecContext::max_threads(4).unwrap()),
    ]
}

/// One input layout over a logical `[rows, cols]` tensor with `rows * cols`
/// close to the requested length.
#[derive(Clone, Debug)]
struct Layout {
    name: &'static str,
    dims: Vec<usize>,
    strides: Vec<isize>,
}

fn layouts(len: usize) -> Vec<Layout> {
    let rows = 3usize;
    let cols = len.div_ceil(rows);
    vec![
        Layout {
            name: "contiguous",
            dims: vec![rows, cols],
            strides: vec![1, rows as isize],
        },
        Layout {
            name: "strided",
            dims: vec![rows, cols],
            strides: vec![cols as isize * 2, 2],
        },
        Layout {
            name: "broadcast",
            dims: vec![rows, cols],
            strides: vec![0, 1],
        },
    ]
}

fn dest_layout(dims: &[usize]) -> Vec<isize> {
    vec![1, dims[0] as isize]
}

fn buffer_len(layout: &Layout) -> usize {
    layout
        .dims
        .iter()
        .zip(&layout.strides)
        .map(|(&d, &s)| (d as isize - 1) * s)
        .sum::<isize>() as usize
        + 1
}

fn coords(dims: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..dims[1]).flat_map(move |c| (0..dims[0]).map(move |r| (r, c)))
}

fn at<T: Copy>(data: &[T], layout: &Layout, (r, c): (usize, usize)) -> T {
    data[(r as isize * layout.strides[0] + c as isize * layout.strides[1]) as usize]
}

fn f64_values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| match (i + seed) % 23 {
            0 => f64::NAN,
            1 => -0.0,
            2 => 0.0,
            3 => f64::INFINITY,
            4 => f64::NEG_INFINITY,
            k => ((i * 7 + seed * 13) % 19) as f64 * 0.75 - 6.0 + k as f64 * 1e-3,
        })
        .collect()
}

fn i64_values(len: usize, seed: usize) -> Vec<i64> {
    (0..len)
        .map(|i| match (i + seed) % 11 {
            0 => i64::MAX,
            1 => i64::MIN,
            2 => -1,
            k => ((i * 31 + seed) % 97) as i64 - 48 + k as i64,
        })
        .collect()
}

fn i32_values(len: usize, seed: usize) -> Vec<i32> {
    (0..len)
        .map(|i| match (i + seed) % 11 {
            0 => i32::MAX,
            1 => i32::MIN,
            2 => -1,
            k => ((i * 31 + seed) % 97) as i32 - 48 + k as i32,
        })
        .collect()
}

fn c64_values(len: usize, seed: usize) -> Vec<Complex64> {
    let re = f64_values(len, seed);
    let im = f64_values(len, seed + 5);
    re.into_iter()
        .zip(im)
        .map(|(re, im)| Complex64::new(re, im))
        .collect()
}

trait Same: Copy + core::fmt::Debug {
    fn same(self, other: Self) -> bool;
}
impl Same for f64 {
    fn same(self, other: Self) -> bool {
        (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
    }
}
impl Same for i32 {
    fn same(self, other: Self) -> bool {
        self == other
    }
}
impl Same for i64 {
    fn same(self, other: Self) -> bool {
        self == other
    }
}
impl Same for bool {
    fn same(self, other: Self) -> bool {
        self == other
    }
}
impl Same for Complex64 {
    fn same(self, other: Self) -> bool {
        self.re.same(other.re) && self.im.same(other.im)
    }
}

fn run_map<T: KernelStorageElement + Copy + Default>(
    dtype: KernelDType,
    op: ErasedMapOp,
    ctx: &ExecContext,
    data: &[T],
    layout: &Layout,
) -> Vec<T> {
    let source = ErasedRawStridedRef::from_slice(data, &layout.dims, &layout.strides, 0).unwrap();
    let dest_strides = dest_layout(&layout.dims);
    let mut out = vec![T::default(); layout.dims.iter().product()];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut out, &layout.dims, &dest_strides, 0).unwrap();
    erased_map_into(
        dtype,
        op,
        ctx,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&source),
    )
    .unwrap();
    out
}

fn run_zip<T: KernelStorageElement + Copy + Default>(
    dtype: KernelDType,
    op: ErasedZipOp,
    ctx: &ExecContext,
    lhs: (&[T], &Layout),
    rhs: (&[T], &Layout),
) -> Vec<T> {
    let l = ErasedRawStridedRef::from_slice(lhs.0, &lhs.1.dims, &lhs.1.strides, 0).unwrap();
    let r = ErasedRawStridedRef::from_slice(rhs.0, &rhs.1.dims, &rhs.1.strides, 0).unwrap();
    let dims = &lhs.1.dims;
    let dest_strides = dest_layout(dims);
    let mut out = vec![T::default(); dims.iter().product()];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, dims, &dest_strides, 0).unwrap();
    erased_zip_into(
        dtype,
        op,
        ctx,
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&l),
        &ErasedRawStridedPtr::from_ref(&r),
    )
    .unwrap();
    out
}

fn check<T: Same>(got: &[T], expected: impl Iterator<Item = T>, what: &str) {
    let mut count = 0;
    for (index, (g, e)) in got.iter().copied().zip(expected).enumerate() {
        assert!(g.same(e), "{what} [{index}]: got {g:?}, expected {e:?}");
        count += 1;
    }
    assert_eq!(count, got.len(), "{what}: length");
}

fn f64_map(op: ErasedMapOp, v: f64) -> f64 {
    match op {
        ErasedMapOp::Negate => -v,
        ErasedMapOp::Conj => v,
        ErasedMapOp::Abs => v.abs(),
        ErasedMapOp::Sign if v == 0.0 => 0.0,
        ErasedMapOp::Sign => v.signum(),
        _ => unreachable!(),
    }
}

fn f64_zip(op: ErasedZipOp, a: f64, b: f64) -> f64 {
    match op {
        ErasedZipOp::Add => a + b,
        ErasedZipOp::Subtract => a - b,
        ErasedZipOp::Multiply => a * b,
        ErasedZipOp::Divide => a / b,
        ErasedZipOp::Remainder => a % b,
        ErasedZipOp::Maximum | ErasedZipOp::Minimum if a.is_nan() || b.is_nan() => f64::NAN,
        ErasedZipOp::Maximum if a >= b => a,
        ErasedZipOp::Maximum => b,
        ErasedZipOp::Minimum if a <= b => a,
        ErasedZipOp::Minimum => b,
        _ => unreachable!(),
    }
}

fn i64_map(op: ErasedMapOp, v: i64) -> i64 {
    match op {
        ErasedMapOp::Negate => v.wrapping_neg(),
        ErasedMapOp::Conj => v,
        ErasedMapOp::Abs => v.wrapping_abs(),
        ErasedMapOp::Sign => v.signum(),
        _ => unreachable!(),
    }
}

fn i64_zip(op: ErasedZipOp, a: i64, b: i64) -> i64 {
    match op {
        ErasedZipOp::Add => a.wrapping_add(b),
        ErasedZipOp::Subtract => a.wrapping_sub(b),
        ErasedZipOp::Multiply => a.wrapping_mul(b),
        ErasedZipOp::Divide => a.wrapping_div(b),
        ErasedZipOp::Remainder => a.wrapping_rem(b),
        ErasedZipOp::Maximum => a.max(b),
        ErasedZipOp::Minimum => a.min(b),
        _ => unreachable!(),
    }
}

const MAP_OPS: [ErasedMapOp; 4] = [
    ErasedMapOp::Negate,
    ErasedMapOp::Conj,
    ErasedMapOp::Abs,
    ErasedMapOp::Sign,
];
const ZIP_OPS: [ErasedZipOp; 7] = [
    ErasedZipOp::Add,
    ErasedZipOp::Subtract,
    ErasedZipOp::Multiply,
    ErasedZipOp::Divide,
    ErasedZipOp::Remainder,
    ErasedZipOp::Maximum,
    ErasedZipOp::Minimum,
];

#[test]
fn map_matches_naive_for_every_op_layout_and_context() {
    for len in LENGTHS {
        for layout in layouts(len) {
            let n = buffer_len(&layout);
            let f = f64_values(n, 1);
            let i = i64_values(n, 1);
            let c = c64_values(n, 1);
            for (ctx_name, ctx) in contexts() {
                for op in MAP_OPS {
                    let what = format!("{op:?} {} {ctx_name} len={len}", layout.name);
                    let got = run_map(KernelDType::F64, op, &ctx, &f, &layout);
                    check(
                        &got,
                        coords(&layout.dims).map(|p| f64_map(op, at(&f, &layout, p))),
                        &format!("f64 {what}"),
                    );
                    let got = run_map(KernelDType::I64, op, &ctx, &i, &layout);
                    check(
                        &got,
                        coords(&layout.dims).map(|p| i64_map(op, at(&i, &layout, p))),
                        &format!("i64 {what}"),
                    );
                    // Complex abs and sign are checked in `complex_sign_matches_naive`
                    // and the existing one-shot tests.
                    if matches!(op, ErasedMapOp::Negate | ErasedMapOp::Conj) {
                        let got = run_map(KernelDType::C64, op, &ctx, &c, &layout);
                        check(
                            &got,
                            coords(&layout.dims).map(|p| {
                                let v = at(&c, &layout, p);
                                match op {
                                    ErasedMapOp::Negate => -v,
                                    ErasedMapOp::Conj => v.conj(),
                                    _ => v,
                                }
                            }),
                            &format!("c64 {what}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_sign_matches_naive() {
    let layout = &layouts(67)[1];
    let c = c64_values(buffer_len(layout), 3);
    for (_, ctx) in contexts() {
        let got = run_map(KernelDType::C64, ErasedMapOp::Sign, &ctx, &c, layout);
        for ((g, p), index) in got.iter().zip(coords(&layout.dims)).zip(0..) {
            let v = at(&c, layout, p);
            if v.re.is_nan() || v.im.is_nan() {
                assert!(g.re.is_nan() || g.im.is_nan(), "[{index}] {g:?}");
            } else if v.re == 0.0 && v.im == 0.0 {
                assert_eq!(*g, Complex64::new(0.0, 0.0));
            } else if v.norm().is_finite() {
                assert!((g.norm() - 1.0).abs() < 1e-12, "[{index}] {v:?} -> {g:?}");
            }
        }
    }
}

#[test]
fn zip_matches_naive_for_every_op_layout_and_context() {
    for len in LENGTHS {
        let lhs_layouts = layouts(len);
        for (lhs_layout, rhs_layout) in lhs_layouts.iter().zip(lhs_layouts.iter().cycle().skip(1)) {
            let fl = f64_values(buffer_len(lhs_layout), 2);
            let fr = f64_values(buffer_len(rhs_layout), 9);
            let il = i64_values(buffer_len(lhs_layout), 2);
            // Nonzero divisors: integer division by zero is rejected before
            // any write, which is tested elsewhere.
            let ir: Vec<i64> = i64_values(buffer_len(rhs_layout), 9)
                .into_iter()
                .map(|v| if v == 0 { 3 } else { v })
                .collect();
            let jl = i32_values(buffer_len(lhs_layout), 2);
            let jr: Vec<i32> = i32_values(buffer_len(rhs_layout), 9)
                .into_iter()
                .map(|v| if v == 0 { 3 } else { v })
                .collect();
            let cl = c64_values(buffer_len(lhs_layout), 2);
            let cr = c64_values(buffer_len(rhs_layout), 9);
            for (ctx_name, ctx) in contexts() {
                for op in ZIP_OPS {
                    let what = format!(
                        "{op:?} {}x{} {ctx_name} len={len}",
                        lhs_layout.name, rhs_layout.name
                    );
                    let got = run_zip(
                        KernelDType::F64,
                        op,
                        &ctx,
                        (&fl, lhs_layout),
                        (&fr, rhs_layout),
                    );
                    check(
                        &got,
                        coords(&lhs_layout.dims)
                            .map(|p| f64_zip(op, at(&fl, lhs_layout, p), at(&fr, rhs_layout, p))),
                        &format!("f64 {what}"),
                    );
                    let got = run_zip(
                        KernelDType::I64,
                        op,
                        &ctx,
                        (&il, lhs_layout),
                        (&ir, rhs_layout),
                    );
                    check(
                        &got,
                        coords(&lhs_layout.dims)
                            .map(|p| i64_zip(op, at(&il, lhs_layout, p), at(&ir, rhs_layout, p))),
                        &format!("i64 {what}"),
                    );
                    let got = run_zip(
                        KernelDType::I32,
                        op,
                        &ctx,
                        (&jl, lhs_layout),
                        (&jr, rhs_layout),
                    );
                    check(
                        &got,
                        coords(&lhs_layout.dims).map(|p| {
                            i64_zip(
                                op,
                                at(&jl, lhs_layout, p) as i64,
                                at(&jr, rhs_layout, p) as i64,
                            ) as i32
                        }),
                        &format!("i32 {what}"),
                    );
                    if matches!(
                        op,
                        ErasedZipOp::Add
                            | ErasedZipOp::Subtract
                            | ErasedZipOp::Multiply
                            | ErasedZipOp::Divide
                    ) {
                        let got = run_zip(
                            KernelDType::C64,
                            op,
                            &ctx,
                            (&cl, lhs_layout),
                            (&cr, rhs_layout),
                        );
                        check(
                            &got,
                            coords(&lhs_layout.dims).map(|p| {
                                let (a, b) = (at(&cl, lhs_layout, p), at(&cr, rhs_layout, p));
                                match op {
                                    ErasedZipOp::Add => a + b,
                                    ErasedZipOp::Subtract => a - b,
                                    ErasedZipOp::Multiply => a * b,
                                    _ => a / b,
                                }
                            }),
                            &format!("c64 {what}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn maximum_minimum_signed_zero_ties_return_lhs() {
    let dims = [5usize];
    let strides = [1isize];
    let lhs = [0.0f64, -0.0, 0.0, f64::NAN, 1.0];
    let rhs = [-0.0f64, 0.0, 0.0, 1.0, f64::NAN];
    for (_, ctx) in contexts() {
        for op in [ErasedZipOp::Maximum, ErasedZipOp::Minimum] {
            let l = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
            let r = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
            let mut out = [0.0f64; 5];
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &strides, 0).unwrap();
            erased_zip_into(
                KernelDType::F64,
                op,
                &ctx,
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&l),
                &ErasedRawStridedPtr::from_ref(&r),
            )
            .unwrap();
            assert_eq!(out[0].to_bits(), 0.0f64.to_bits(), "{op:?}");
            assert_eq!(out[1].to_bits(), (-0.0f64).to_bits(), "{op:?}");
            assert!(out[3].is_nan() && out[4].is_nan(), "{op:?}");
        }
    }
}

fn init<T: Copy>(values: Vec<MaybeUninit<T>>) -> Vec<T> {
    // SAFETY: callers only pass storage that a successful call initialized.
    values
        .into_iter()
        .map(|v| unsafe { v.assume_init() })
        .collect()
}

#[test]
fn compare_select_clamp_match_naive() {
    let compare_ops = [
        CompareOp::Eq,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ];
    for len in LENGTHS {
        let all = layouts(len);
        for (a_layout, b_layout) in all.iter().zip(all.iter().cycle().skip(2)) {
            let dims = &a_layout.dims;
            let total: usize = dims.iter().product();
            let dest_strides = dest_layout(dims);
            let a = f64_values(buffer_len(a_layout), 4);
            let b = f64_values(buffer_len(b_layout), 6);
            let a_ref = ErasedRawStridedRef::from_slice(&a, dims, &a_layout.strides, 0).unwrap();
            let b_ref =
                ErasedRawStridedRef::from_slice(&b, &b_layout.dims, &b_layout.strides, 0).unwrap();
            for (ctx_name, ctx) in contexts() {
                for op in compare_ops {
                    let mut out = vec![MaybeUninit::<bool>::uninit(); total];
                    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(
                        &mut out,
                        dims,
                        &dest_strides,
                        0,
                    )
                    .unwrap();
                    erased_compare_into_uninit(
                        KernelDType::F64,
                        op,
                        &ctx,
                        &mut dest,
                        &ErasedRawStridedPtr::from_ref(&a_ref),
                        &ErasedRawStridedPtr::from_ref(&b_ref),
                    )
                    .unwrap();
                    let got = init(out);
                    check(
                        &got,
                        coords(dims).map(|p| {
                            let (x, y) = (at(&a, a_layout, p), at(&b, b_layout, p));
                            match op {
                                CompareOp::Eq => x == y,
                                CompareOp::Lt => x < y,
                                CompareOp::Le => x <= y,
                                CompareOp::Gt => x > y,
                                _ => x >= y,
                            }
                        }),
                        &format!(
                            "compare {op:?} {}x{} {ctx_name} len={len}",
                            a_layout.name, b_layout.name
                        ),
                    );
                }

                // Select with a strided predicate.
                let pred: Vec<bool> = (0..buffer_len(b_layout)).map(|k| k % 3 == 1).collect();
                let pred_ref =
                    ErasedRawStridedRef::from_slice(&pred, dims, &b_layout.strides, 0).unwrap();
                let mut out = vec![MaybeUninit::<f64>::uninit(); total];
                let mut dest =
                    ErasedRawStridedUninitMut::from_uninit_slice(&mut out, dims, &dest_strides, 0)
                        .unwrap();
                erased_select_into_uninit(
                    KernelDType::F64,
                    &ctx,
                    &mut dest,
                    &ErasedRawStridedPtr::from_ref(&pred_ref),
                    &ErasedRawStridedPtr::from_ref(&a_ref),
                    &ErasedRawStridedPtr::from_ref(&b_ref),
                )
                .unwrap();
                check(
                    &init(out),
                    coords(dims).map(|p| {
                        if at(&pred, b_layout, p) {
                            at(&a, a_layout, p)
                        } else {
                            at(&b, b_layout, p)
                        }
                    }),
                    &format!("select {} {ctx_name} len={len}", a_layout.name),
                );

                // Clamp: x from `a`, bounds broadcast from short vectors.
                let lo = vec![-1.5f64; buffer_len(b_layout)];
                let hi = vec![2.25f64; buffer_len(b_layout)];
                let lo_ref =
                    ErasedRawStridedRef::from_slice(&lo, dims, &b_layout.strides, 0).unwrap();
                let hi_ref =
                    ErasedRawStridedRef::from_slice(&hi, dims, &b_layout.strides, 0).unwrap();
                let mut out = vec![MaybeUninit::<f64>::uninit(); total];
                let mut dest =
                    ErasedRawStridedUninitMut::from_uninit_slice(&mut out, dims, &dest_strides, 0)
                        .unwrap();
                erased_clamp_into_uninit(
                    KernelDType::F64,
                    &ctx,
                    &mut dest,
                    &ErasedRawStridedPtr::from_ref(&a_ref),
                    &ErasedRawStridedPtr::from_ref(&lo_ref),
                    &ErasedRawStridedPtr::from_ref(&hi_ref),
                )
                .unwrap();
                check(
                    &init(out),
                    coords(dims).map(|p| {
                        let x = at(&a, a_layout, p);
                        f64_zip(
                            ErasedZipOp::Minimum,
                            2.25,
                            f64_zip(ErasedZipOp::Maximum, -1.5, x),
                        )
                    }),
                    &format!("clamp {} {ctx_name} len={len}", a_layout.name),
                );
            }
        }
    }
}
