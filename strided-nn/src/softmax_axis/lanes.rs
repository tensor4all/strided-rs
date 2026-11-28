//! Leaf loops of the strided softmax over pulp vectors.
//!
//! Every leaf runs inside one `pulp::Arch::new().dispatch(...)`, so the level (AVX-512,
//! AVX2, NEON, scalar) is picked at run time and a baseline build still reaches AVX-512
//! (`scripts/check-runtime-simd-dispatch.py`). The leaves are elementwise across the
//! contiguous columns of one row; the only reductions are the per-column running max and
//! sum across rows (source order), and the per-row dot of the contiguous VJP.

use core::mem::MaybeUninit;

use strided_basic::Exec;

use crate::error::SoftmaxError;
use crate::real::Real;

pub(crate) trait Lanes: Real {
    const NEG_INFINITY: Self;
    fn softmax_axis0(
        axis_len: usize,
        input: &[Self],
        output: &mut [MaybeUninit<Self>],
        scratch: &mut [Self],
        exec: Exec<'_>,
    ) -> Result<(), SoftmaxError>;
    fn exp(self) -> Self;
    /// `acc = max(acc, row)`.
    fn max_into(acc: &mut [Self], row: &[Self]);
    /// `temp = row - max`.
    fn shift_into(row: &[Self], max: &[Self], temp: &mut [Self]);
    /// `acc += row`.
    fn add_into(acc: &mut [Self], row: &[Self]);
    /// `row /= sum`.
    fn div_into(row: &mut [Self], sum: &[Self]);
    /// `acc += y * dy`.
    fn dot_into(acc: &mut [Self], y: &[Self], dy: &[Self]);
    /// `dx = y * (dy - dot)`.
    fn apply_into(y: &[Self], dy: &[Self], dot: &[Self], dx: &mut [Self]);
    /// Contiguous VJP over whole rows of `axis_len`.
    fn vjp_rows(y: &[Self], dy: &[Self], dx: &mut [Self], axis_len: usize);
    /// The shift the contiguous kernel uses: a `-inf` max becomes `MIN`, NaN stays NaN.
    fn clamp_max(max: &mut [Self]);
    /// The divisor the contiguous kernel uses: at least `MIN_POSITIVE`, NaN stays NaN.
    fn clamp_sum(sum: &mut [Self]);
    /// `exp(temp)` into `out`, every element written.
    fn exp_into_slots(temp: &[Self], out: &mut [MaybeUninit<Self>]) {
        if !Self::exp_into(temp, out) {
            for (slot, &value) in out.iter_mut().zip(temp) {
                slot.write(value.exp());
            }
        }
    }
}

macro_rules! lanes {
    ($t:ident, $axis0:path, $as:ident, $as_mut:ident, $splat:ident, $max:ident, $add:ident,
     $sub:ident, $mul:ident, $div:ident, $reduce:ident) => {
        impl Lanes for $t {
            const NEG_INFINITY: Self = $t::NEG_INFINITY;
            fn softmax_axis0(
                axis_len: usize,
                input: &[Self],
                output: &mut [MaybeUninit<Self>],
                scratch: &mut [Self],
                exec: Exec<'_>,
            ) -> Result<(), SoftmaxError> {
                $axis0(axis_len, input, output, scratch, exec)
            }
            fn exp(self) -> Self {
                $t::exp(self)
            }
            fn max_into(acc: &mut [Self], row: &[Self]) {
                struct Op<'a>(&'a mut [$t], &'a [$t]);
                impl pulp::WithSimd for Op<'_> {
                    type Output = ();
                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) {
                        let (ah, at) = S::$as_mut(self.0);
                        let (rh, rt) = S::$as(self.1);
                        for (a, &r) in ah.iter_mut().zip(rh) {
                            *a = simd.$max(*a, r);
                        }
                        for (a, &r) in at.iter_mut().zip(rt) {
                            *a = a.max(r);
                        }
                    }
                }
                pulp::Arch::new().dispatch(Op(acc, row));
            }
            fn shift_into(row: &[Self], max: &[Self], temp: &mut [Self]) {
                struct Op<'a>(&'a [$t], &'a [$t], &'a mut [$t]);
                impl pulp::WithSimd for Op<'_> {
                    type Output = ();
                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) {
                        let (rh, rt) = S::$as(self.0);
                        let (mh, mt) = S::$as(self.1);
                        let (oh, ot) = S::$as_mut(self.2);
                        for ((o, &r), &m) in oh.iter_mut().zip(rh).zip(mh) {
                            *o = simd.$sub(r, m);
                        }
                        for ((o, &r), &m) in ot.iter_mut().zip(rt).zip(mt) {
                            *o = r - m;
                        }
                    }
                }
                pulp::Arch::new().dispatch(Op(row, max, temp));
            }
            fn add_into(acc: &mut [Self], row: &[Self]) {
                struct Op<'a>(&'a mut [$t], &'a [$t]);
                impl pulp::WithSimd for Op<'_> {
                    type Output = ();
                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) {
                        let (ah, at) = S::$as_mut(self.0);
                        let (rh, rt) = S::$as(self.1);
                        for (a, &r) in ah.iter_mut().zip(rh) {
                            *a = simd.$add(*a, r);
                        }
                        for (a, &r) in at.iter_mut().zip(rt) {
                            *a += r;
                        }
                    }
                }
                pulp::Arch::new().dispatch(Op(acc, row));
            }
            fn div_into(row: &mut [Self], sum: &[Self]) {
                struct Op<'a>(&'a mut [$t], &'a [$t]);
                impl pulp::WithSimd for Op<'_> {
                    type Output = ();
                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) {
                        let (rh, rt) = S::$as_mut(self.0);
                        let (sh, st) = S::$as(self.1);
                        for (r, &s) in rh.iter_mut().zip(sh) {
                            *r = simd.$div(*r, s);
                        }
                        for (r, &s) in rt.iter_mut().zip(st) {
                            *r /= s;
                        }
                    }
                }
                pulp::Arch::new().dispatch(Op(row, sum));
            }
            fn dot_into(acc: &mut [Self], y: &[Self], dy: &[Self]) {
                struct Op<'a>(&'a mut [$t], &'a [$t], &'a [$t]);
                impl pulp::WithSimd for Op<'_> {
                    type Output = ();
                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) {
                        let (ah, at) = S::$as_mut(self.0);
                        let (yh, yt) = S::$as(self.1);
                        let (gh, gt) = S::$as(self.2);
                        for ((a, &y), &g) in ah.iter_mut().zip(yh).zip(gh) {
                            *a = simd.$add(*a, simd.$mul(y, g));
                        }
                        for ((a, &y), &g) in at.iter_mut().zip(yt).zip(gt) {
                            *a += y * g;
                        }
                    }
                }
                pulp::Arch::new().dispatch(Op(acc, y, dy));
            }
            fn apply_into(y: &[Self], dy: &[Self], dot: &[Self], dx: &mut [Self]) {
                struct Op<'a>(&'a [$t], &'a [$t], &'a [$t], &'a mut [$t]);
                impl pulp::WithSimd for Op<'_> {
                    type Output = ();
                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) {
                        let (yh, yt) = S::$as(self.0);
                        let (gh, gt) = S::$as(self.1);
                        let (dh, dt) = S::$as(self.2);
                        let (oh, ot) = S::$as_mut(self.3);
                        for (((o, &y), &g), &d) in oh.iter_mut().zip(yh).zip(gh).zip(dh) {
                            *o = simd.$mul(y, simd.$sub(g, d));
                        }
                        for (((o, &y), &g), &d) in ot.iter_mut().zip(yt).zip(gt).zip(dt) {
                            *o = y * (g - d);
                        }
                    }
                }
                pulp::Arch::new().dispatch(Op(y, dy, dot, dx));
            }
            fn vjp_rows(y: &[Self], dy: &[Self], dx: &mut [Self], axis_len: usize) {
                struct Op<'a>(&'a [$t], &'a [$t], &'a mut [$t], usize);
                impl pulp::WithSimd for Op<'_> {
                    type Output = ();
                    #[inline(always)]
                    fn with_simd<S: pulp::Simd>(self, simd: S) {
                        let rows = self.0.chunks(self.3).zip(self.1.chunks(self.3));
                        for ((y, dy), dx) in rows.zip(self.2.chunks_mut(self.3)) {
                            let (yh, yt) = S::$as(y);
                            let (gh, gt) = S::$as(dy);
                            let mut acc = simd.$splat(0.0);
                            for (&y, &g) in yh.iter().zip(gh) {
                                acc = simd.$add(acc, simd.$mul(y, g));
                            }
                            let mut dot = simd.$reduce(acc);
                            for (&y, &g) in yt.iter().zip(gt) {
                                dot += y * g;
                            }
                            let d = simd.$splat(dot);
                            let (oh, ot) = S::$as_mut(dx);
                            for ((o, &y), &g) in oh.iter_mut().zip(yh).zip(gh) {
                                *o = simd.$mul(y, simd.$sub(g, d));
                            }
                            for ((o, &y), &g) in ot.iter_mut().zip(yt).zip(gt) {
                                *o = y * (g - dot);
                            }
                        }
                    }
                }
                pulp::Arch::new().dispatch(Op(y, dy, dx, axis_len));
            }
            fn clamp_max(max: &mut [Self]) {
                for m in max {
                    if !m.is_nan() {
                        *m = m.max($t::MIN);
                    }
                }
            }
            fn clamp_sum(sum: &mut [Self]) {
                for s in sum {
                    if !s.is_nan() {
                        *s = s.max($t::MIN_POSITIVE);
                    }
                }
            }
        }
    };
}

lanes!(
    f32,
    crate::softmax_axis0_f32,
    as_simd_f32s,
    as_mut_simd_f32s,
    splat_f32s,
    max_f32s,
    add_f32s,
    sub_f32s,
    mul_f32s,
    div_f32s,
    reduce_sum_f32s
);
lanes!(
    f64,
    crate::softmax_axis0_f64,
    as_simd_f64s,
    as_mut_simd_f64s,
    splat_f64s,
    max_f64s,
    add_f64s,
    sub_f64s,
    mul_f64s,
    div_f64s,
    reduce_sum_f64s
);
