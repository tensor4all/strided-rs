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
//!
//! **The dispatch is per operation row, not per path.** One row of the CG plan
//! carries 86 paths, so resolving the ISA and the mask once per path cost more
//! than the arithmetic of the paths themselves (tensor4all/strided-rs#498). The
//! path loop lives inside the dispatched body here: same operations in the same
//! order, one dispatch per row.

use pulp::{Simd, WithSimd};

use crate::plan::MAX_INPUTS;

/// One operation row: every path of one operation at one batch point.
///
/// `acc` is `seg_starts.len() * u` elements. `offsets` holds `arity + 1` entries
/// per path, the inputs' segment offsets first and the output's last (the layout
/// the scalar replay shares); `arity` is the input count, so a path strides
/// `arity + 1`, and input `t` starts at `bases[t].offset(offsets[path * (arity +
/// 1) + t])`. Bit `t` of `splat_mask` is set when input `t` carries one element
/// for the whole lane axis instead of `u` lanes.
// `pub` inside a private module: the sealing trait's method carries it, and the
// module itself is not reachable from outside the crate.
pub struct OpRow<'a, T> {
    pub acc: &'a mut [T],
    pub u: usize,
    pub seg_starts: &'a [u32],
    pub seg_ends: &'a [u32],
    pub coeffs: &'a [T],
    pub offsets: &'a [isize],
    pub bases: [*const T; MAX_INPUTS],
    pub arity: usize,
    pub splat_mask: u32,
}

struct RowF64<'a, const N: usize, const MASK: u8> {
    row: OpRow<'a, f64>,
}

impl<const N: usize, const MASK: u8> WithSimd for RowF64<'_, N, MASK> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let row = self.row;
        let u = row.u;
        for (segment, (&start, &end)) in row.seg_starts.iter().zip(row.seg_ends.iter()).enumerate()
        {
            if start == end {
                continue;
            }
            let slots = &mut row.acc[segment * u..segment * u + u];
            for local in start as usize..end as usize {
                let coefficient = row.coeffs[local];
                let path = &row.offsets[local * (N + 1)..local * (N + 1) + N];
                let operands: [&[f64]; N] = core::array::from_fn(|t| {
                    if MASK & (1 << t) == 0 {
                        &[]
                    } else {
                        // SAFETY: the caller's contract; the operand is `u`
                        // contiguous lanes from this path's segment base.
                        unsafe { core::slice::from_raw_parts(row.bases[t].offset(path[t]), u) }
                    }
                });
                let splats: [f64; N] = core::array::from_fn(|t| {
                    if MASK & (1 << t) != 0 {
                        0.0
                    } else {
                        // SAFETY: the caller's contract; a unit-lane operand reads
                        // one element for the whole lane axis.
                        unsafe { *row.bases[t].offset(path[t]) }
                    }
                });
                axpy_f64::<N, MASK, S>(simd, slots, coefficient, operands, splats);
            }
        }
    }
}

struct RowF32<'a, const N: usize, const MASK: u8> {
    row: OpRow<'a, f32>,
}

impl<const N: usize, const MASK: u8> WithSimd for RowF32<'_, N, MASK> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let row = self.row;
        let u = row.u;
        for (segment, (&start, &end)) in row.seg_starts.iter().zip(row.seg_ends.iter()).enumerate()
        {
            if start == end {
                continue;
            }
            let slots = &mut row.acc[segment * u..segment * u + u];
            for local in start as usize..end as usize {
                let coefficient = row.coeffs[local];
                let path = &row.offsets[local * (N + 1)..local * (N + 1) + N];
                let operands: [&[f32]; N] = core::array::from_fn(|t| {
                    if MASK & (1 << t) == 0 {
                        &[]
                    } else {
                        // SAFETY: as in `RowF64`.
                        unsafe { core::slice::from_raw_parts(row.bases[t].offset(path[t]), u) }
                    }
                });
                let splats: [f32; N] = core::array::from_fn(|t| {
                    if MASK & (1 << t) != 0 {
                        0.0
                    } else {
                        // SAFETY: as in `RowF64`.
                        unsafe { *row.bases[t].offset(path[t]) }
                    }
                });
                axpy_f32::<N, MASK, S>(simd, slots, coefficient, operands, splats);
            }
        }
    }
}

/// One path: `acc[i] = acc[i] + ((c * a0[i]) * a1[i]) * a2[i]` over the lanes.
///
/// The vector loop and the scalar remainder are the same statement, and neither
/// uses `mul_add`: the rounding must not depend on the selected level.
#[inline(always)]
fn axpy_f64<const N: usize, const MASK: u8, S: Simd>(
    simd: S,
    acc: &mut [f64],
    coefficient: f64,
    operands: [&[f64]; N],
    splats: [f64; N],
) {
    let total = acc.len();
    let (acc_head, acc_tail) = S::as_mut_simd_f64s(acc);
    let lanes_in_head = total - acc_tail.len();
    let head_coefficient = simd.splat_f64s(coefficient);
    // Bit `t` of `MASK` is set when operand `t` is a lane slice; the mask is a
    // const parameter, so one side of every branch below is dead and removed.
    let head_splats: [S::f64s; N] = core::array::from_fn(|t| simd.splat_f64s(splats[t]));
    let heads: [&[S::f64s]; N] = core::array::from_fn(|t| {
        if MASK & (1 << t) == 0 {
            &[]
        } else {
            S::as_simd_f64s(operands[t]).0
        }
    });
    for (index, slot) in acc_head.iter_mut().enumerate() {
        let mut value = head_coefficient;
        for t in 0..N {
            let operand = if MASK & (1 << t) == 0 {
                head_splats[t]
            } else {
                heads[t][index]
            };
            value = simd.mul_f64s(value, operand);
        }
        *slot = simd.add_f64s(*slot, value);
    }
    for (index, slot) in acc_tail.iter_mut().enumerate() {
        let mut value = coefficient;
        for t in 0..N {
            let operand = if MASK & (1 << t) == 0 {
                splats[t]
            } else {
                operands[t][lanes_in_head + index]
            };
            value *= operand;
        }
        *slot += value;
    }
}

#[inline(always)]
fn axpy_f32<const N: usize, const MASK: u8, S: Simd>(
    simd: S,
    acc: &mut [f32],
    coefficient: f32,
    operands: [&[f32]; N],
    splats: [f32; N],
) {
    let total = acc.len();
    let (acc_head, acc_tail) = S::as_mut_simd_f32s(acc);
    let lanes_in_head = total - acc_tail.len();
    let head_coefficient = simd.splat_f32s(coefficient);
    let head_splats: [S::f32s; N] = core::array::from_fn(|t| simd.splat_f32s(splats[t]));
    let heads: [&[S::f32s]; N] = core::array::from_fn(|t| {
        if MASK & (1 << t) == 0 {
            &[]
        } else {
            S::as_simd_f32s(operands[t]).0
        }
    });
    for (index, slot) in acc_head.iter_mut().enumerate() {
        let mut value = head_coefficient;
        for t in 0..N {
            let operand = if MASK & (1 << t) == 0 {
                head_splats[t]
            } else {
                heads[t][index]
            };
            value = simd.mul_f32s(value, operand);
        }
        *slot = simd.add_f32s(*slot, value);
    }
    for (index, slot) in acc_tail.iter_mut().enumerate() {
        let mut value = coefficient;
        for t in 0..N {
            let operand = if MASK & (1 << t) == 0 {
                splats[t]
            } else {
                operands[t][lanes_in_head + index]
            };
            value *= operand;
        }
        *slot += value;
    }
}

/// One dispatch entry per dtype: every (arity, mask) the executor can carry.
macro_rules! op_row_entry {
    ($name:ident, $row:ident, $elem:ty; $(($n:literal, [$($mask:literal),+])),+ $(,)?) => {
        /// Sum one operation row through the dispatched lane kernel.
        ///
        /// `arch` is the caller's resolved [`pulp::Arch`], so the CPU-feature
        /// lookup happens once per replay rather than once per path.
        ///
        /// # Safety
        ///
        /// `row.acc` holds `row.seg_starts.len() * row.u` elements; for every path
        /// `p` and every operand `t < row.arity`, `row.bases[t].offset(row.offsets[p
        /// * (arity + 1) + t])` reaches either `u` contiguous, initialised elements
        /// (bit `t` of `!row.splat_mask`) or one initialised element (bit `t` of
        /// `row.splat_mask`), and `row.coeffs`/`row.seg_ends` cover every path.
        pub(crate) unsafe fn $name(arch: pulp::Arch, row: OpRow<'_, $elem>) {
            let mask = !row.splat_mask & ((1u32 << row.arity) - 1);
            match (row.arity, mask) {
                $($(
                    ($n, $mask) => arch.dispatch($row::<$n, $mask> { row }),
                )+)+
                (arity, mask) => unreachable!("arity {arity} with slice mask {mask:#b}"),
            }
        }
    };
}

op_row_entry!(op_row_f64, RowF64, f64; (1, [0, 1]), (2, [0, 1, 2, 3]), (3, [0, 1, 2, 3, 4, 5, 6, 7]));
op_row_entry!(op_row_f32, RowF32, f32; (1, [0, 1]), (2, [0, 1, 2, 3]), (3, [0, 1, 2, 3, 4, 5, 6, 7]));

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
