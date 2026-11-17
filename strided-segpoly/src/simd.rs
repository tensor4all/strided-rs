//! The lane kernel, dispatched at run time.
//!
//! `acc[i] = acc[i] + ((c * a0[i]) * a1[i]) * a2[i]` for every lane `i` is the
//! inner loop of every operation whose arity is one to three, and it is the only
//! loop whose trip count is the lane extent `u`. It ran on auto-vectorisation
//! alone, so a baseline x86-64 build used SSE2 on every machine and `f32` bought
//! nothing over `f64` (tensor4all/strided-rs#496).
//!
//! This module runs the loop inside `pulp::Arch::new().dispatch(...)`, which
//! compiles the body once per SIMD level (`V4` = AVX-512, `V3` = AVX2/FMA, the
//! scalar level, NEON on aarch64) and picks one at run time; see
//! `docs/design/runtime-simd-dispatch.md`. How many lanes one iteration handles
//! is `S::F64_LANES` / `S::F32_LANES`, never a literal.
//!
//! **The result does not depend on the level.** Each lane is `((c * a0) * a1) *
//! a2` followed by one add, in the operand order the plan stores; the vector loop
//! and the scalar remainder do the same operations in the same order, and
//! `mul_add` is deliberately not used — the scalar path rounds twice, so the
//! vector path must too. The scalar level has `S::F64_LANES == 1`, which leaves
//! the whole body to the remainder loop: that *is* the fallback, not a second
//! implementation.
//!
//! An operand is either `u` contiguous lanes (a lane-unit-stride buffer, or one
//! the replay packed into scratch) or a unit-lane buffer whose single value
//! applies to every lane. The second kind is a *splat*, selected by a const mask
//! so the choice folds away per instantiation instead of branching per lane.

use pulp::{Simd, WithSimd};

struct AxpyF64<'a, const N: usize, const MASK: u8> {
    acc: &'a mut [f64],
    coefficient: f64,
    operands: [&'a [f64]; N],
    splats: [f64; N],
}

impl<const N: usize, const MASK: u8> WithSimd for AxpyF64<'_, N, MASK> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let total = self.acc.len();
        let (acc_head, acc_tail) = S::as_mut_simd_f64s(self.acc);
        let lanes_in_head = total - acc_tail.len();
        let coefficient = simd.splat_f64s(self.coefficient);
        // Bit `t` of `MASK` is set when operand `t` is a lane slice; the mask is a
        // const parameter, so one side of every branch below is dead and removed.
        let splats: [S::f64s; N] = core::array::from_fn(|t| simd.splat_f64s(self.splats[t]));
        let heads: [&[S::f64s]; N] = core::array::from_fn(|t| {
            if MASK & (1 << t) == 0 {
                &[]
            } else {
                S::as_simd_f64s(self.operands[t]).0
            }
        });
        for (index, slot) in acc_head.iter_mut().enumerate() {
            let mut value = coefficient;
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    splats[t]
                } else {
                    heads[t][index]
                };
                value = simd.mul_f64s(value, operand);
            }
            *slot = simd.add_f64s(*slot, value);
        }
        for (index, slot) in acc_tail.iter_mut().enumerate() {
            let mut value = self.coefficient;
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    self.splats[t]
                } else {
                    self.operands[t][lanes_in_head + index]
                };
                value *= operand;
            }
            *slot += value;
        }
    }
}

struct AxpyF32<'a, const N: usize, const MASK: u8> {
    acc: &'a mut [f32],
    coefficient: f32,
    operands: [&'a [f32]; N],
    splats: [f32; N],
}

impl<const N: usize, const MASK: u8> WithSimd for AxpyF32<'_, N, MASK> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let total = self.acc.len();
        let (acc_head, acc_tail) = S::as_mut_simd_f32s(self.acc);
        let lanes_in_head = total - acc_tail.len();
        let coefficient = simd.splat_f32s(self.coefficient);
        let splats: [S::f32s; N] = core::array::from_fn(|t| simd.splat_f32s(self.splats[t]));
        let heads: [&[S::f32s]; N] = core::array::from_fn(|t| {
            if MASK & (1 << t) == 0 {
                &[]
            } else {
                S::as_simd_f32s(self.operands[t]).0
            }
        });
        for (index, slot) in acc_head.iter_mut().enumerate() {
            let mut value = coefficient;
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    splats[t]
                } else {
                    heads[t][index]
                };
                value = simd.mul_f32s(value, operand);
            }
            *slot = simd.add_f32s(*slot, value);
        }
        for (index, slot) in acc_tail.iter_mut().enumerate() {
            let mut value = self.coefficient;
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    self.splats[t]
                } else {
                    self.operands[t][lanes_in_head + index]
                };
                value *= operand;
            }
            *slot += value;
        }
    }
}

/// One dispatch entry per (dtype, arity): every mask the arity can carry.
macro_rules! axpy_arity {
    ($name:ident, $body:ident, $elem:ty, $n:literal, [$($mask:literal),+]) => {
        pub(crate) fn $name(
            acc: &mut [$elem],
            coefficient: $elem,
            operands: [&[$elem]; $n],
            splats: [$elem; $n],
            slices: u32,
        ) {
            let mask = slices & ((1 << $n) - 1);
            match mask {
                $(
                    $mask => pulp::Arch::new().dispatch($body::<$n, $mask> {
                        acc,
                        coefficient,
                        operands,
                        splats,
                    }),
                )+
                _ => unreachable!("slice mask {mask} is above 2^{}", $n),
            }
        }
    };
}

axpy_arity!(axpy1_f64, AxpyF64, f64, 1, [0, 1]);
axpy_arity!(axpy2_f64, AxpyF64, f64, 2, [0, 1, 2, 3]);
axpy_arity!(axpy3_f64, AxpyF64, f64, 3, [0, 1, 2, 3, 4, 5, 6, 7]);
axpy_arity!(axpy1_f32, AxpyF32, f32, 1, [0, 1]);
axpy_arity!(axpy2_f32, AxpyF32, f32, 2, [0, 1, 2, 3]);
axpy_arity!(axpy3_f32, AxpyF32, f32, 3, [0, 1, 2, 3, 4, 5, 6, 7]);

/// The `f64` lanes the run-time ISA's vector holds.
pub(crate) fn lanes_f64() -> usize {
    pulp::Arch::new().dispatch(Lanes { f64_lanes: true })
}

/// The `f32` lanes the run-time ISA's vector holds.
pub(crate) fn lanes_f32() -> usize {
    pulp::Arch::new().dispatch(Lanes { f64_lanes: false })
}

struct Lanes {
    f64_lanes: bool,
}

impl WithSimd for Lanes {
    type Output = usize;

    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) -> usize {
        if self.f64_lanes {
            S::F64_LANES
        } else {
            S::F32_LANES
        }
    }
}

#[cfg(test)]
mod tests;
