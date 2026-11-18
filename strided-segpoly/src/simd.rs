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

/// SIMD vectors one accumulator chunk keeps in registers while the path loop
/// runs. The per-path form loaded and stored `acc` once per path per vector; a
/// chunk fetches the vectors once and walks every path inside them, so those
/// loads and stores happen once per chunk instead of once per path. Eight
/// vectors fit the register file of every level the dispatch can select (NEON,
/// AVX2, AVX-512), and the remainder plus the scalar tail take the one-vector
/// form, which keeps the same saving for a short run.
///
/// The arithmetic does not change: for every lane the accumulator starts at its
/// current value and receives `((c * a_0) * a_1) * a_2` in path order, exactly
/// the sequence the per-path form executed. Only the moment its value is fetched
/// and written back moves.
const CHUNK_VECTORS: usize = 8;

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
            axpy_segment_f64::<N, MASK, S>(
                simd,
                slots,
                row.coeffs,
                row.offsets,
                &row.bases,
                start as usize,
                end as usize,
            );
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
            axpy_segment_f32::<N, MASK, S>(
                simd,
                slots,
                row.coeffs,
                row.offsets,
                &row.bases,
                start as usize,
                end as usize,
            );
        }
    }
}

/// Sum one segment's paths into `acc`, keeping [`CHUNK_VECTORS`] vectors of the
/// accumulator in registers while the path loop runs.
///
/// Per lane the statement is the scalar contract: start from the accumulator,
/// then for every path in order add `((c * a_0) * a_1) * a_2`. Only when the
/// accumulator is fetched and written back changes, so the result is bit
/// identical to fetching it per path.
#[inline(always)]
fn axpy_segment_f64<const N: usize, const MASK: u8, S: Simd>(
    simd: S,
    acc: &mut [f64],
    coeffs: &[f64],
    offsets: &[isize],
    bases: &[*const f64; MAX_INPUTS],
    start: usize,
    end: usize,
) {
    let u = acc.len();
    let (head, tail) = S::as_mut_simd_f64s(acc);
    let full = head.len() / CHUNK_VECTORS * CHUNK_VECTORS;
    let (groups, rest) = head.split_at_mut(full);

    // The per-path values one vector slice of this path needs: the coefficient as
    // a vector, each splat operand as a vector, and each slice operand as the
    // `len` vectors starting at `index`. The mask is a const, so the unused side
    // of every branch below is removed per instantiation.
    macro_rules! path {
        ($local:expr, $index:expr, $len:expr, $co:ident, $splats:ident, $heads:ident) => {
            let path = &offsets[$local * (N + 1)..$local * (N + 1) + N];
            let $co = simd.splat_f64s(coeffs[$local]);
            let $splats: [S::f64s; N] = core::array::from_fn(|t| {
                if MASK & (1 << t) != 0 {
                    simd.splat_f64s(0.0)
                } else {
                    // SAFETY: the caller's contract; a unit-lane operand reads one
                    // element for the whole lane axis.
                    simd.splat_f64s(unsafe { *bases[t].offset(path[t]) })
                }
            });
            let $heads: [&[S::f64s]; N] = core::array::from_fn(|t| {
                if MASK & (1 << t) == 0 {
                    &[]
                } else {
                    // SAFETY: the caller's contract; the operand is `u` contiguous
                    // lanes from this path's segment base.
                    let lanes = unsafe { core::slice::from_raw_parts(bases[t].offset(path[t]), u) };
                    &S::as_simd_f64s(lanes).0[$index..$index + $len]
                }
            });
        };
    }

    let (groups, _) = groups.as_chunks_mut::<CHUNK_VECTORS>();
    for (chunk, group) in groups.iter_mut().enumerate() {
        let index = chunk * CHUNK_VECTORS;
        let mut accumulators: [S::f64s; CHUNK_VECTORS] = *group;
        for local in start..end {
            path!(local, index, CHUNK_VECTORS, coefficient, splats, heads);
            for k in 0..CHUNK_VECTORS {
                let mut value = coefficient;
                for t in 0..N {
                    let operand = if MASK & (1 << t) == 0 {
                        splats[t]
                    } else {
                        heads[t][k]
                    };
                    value = simd.mul_f64s(value, operand);
                }
                accumulators[k] = simd.add_f64s(accumulators[k], value);
            }
        }
        *group = accumulators;
    }

    // Fewer than `CHUNK_VECTORS` vectors remain: one register-resident
    // accumulator per vector, the same saving over a short run.
    for (offset, slot) in rest.iter_mut().enumerate() {
        let index = full + offset;
        let mut accumulator = *slot;
        for local in start..end {
            path!(local, index, 1, coefficient, splats, heads);
            let mut value = coefficient;
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    splats[t]
                } else {
                    heads[t][0]
                };
                value = simd.mul_f64s(value, operand);
            }
            accumulator = simd.add_f64s(accumulator, value);
        }
        *slot = accumulator;
    }

    // Lanes past the last whole vector, in increasing lane order.
    let first_tail = u - tail.len();
    for (offset, slot) in tail.iter_mut().enumerate() {
        let lane = first_tail + offset;
        let mut accumulator = *slot;
        for local in start..end {
            let path = &offsets[local * (N + 1)..local * (N + 1) + N];
            let mut value = coeffs[local];
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    // SAFETY: a unit-lane operand reads one element.
                    unsafe { *bases[t].offset(path[t]) }
                } else {
                    // SAFETY: the lane is inside this path's `u` lanes.
                    unsafe { *bases[t].offset(path[t] + lane as isize) }
                };
                value *= operand;
            }
            accumulator += value;
        }
        *slot = accumulator;
    }
}

/// The `f32` form of [`axpy_segment_f64`]; same statement, same chunking.
#[inline(always)]
fn axpy_segment_f32<const N: usize, const MASK: u8, S: Simd>(
    simd: S,
    acc: &mut [f32],
    coeffs: &[f32],
    offsets: &[isize],
    bases: &[*const f32; MAX_INPUTS],
    start: usize,
    end: usize,
) {
    let u = acc.len();
    let (head, tail) = S::as_mut_simd_f32s(acc);
    let full = head.len() / CHUNK_VECTORS * CHUNK_VECTORS;
    let (groups, rest) = head.split_at_mut(full);

    macro_rules! path {
        ($local:expr, $index:expr, $len:expr, $co:ident, $splats:ident, $heads:ident) => {
            let path = &offsets[$local * (N + 1)..$local * (N + 1) + N];
            let $co = simd.splat_f32s(coeffs[$local]);
            let $splats: [S::f32s; N] = core::array::from_fn(|t| {
                if MASK & (1 << t) != 0 {
                    simd.splat_f32s(0.0)
                } else {
                    // SAFETY: as in `axpy_segment_f64`.
                    simd.splat_f32s(unsafe { *bases[t].offset(path[t]) })
                }
            });
            let $heads: [&[S::f32s]; N] = core::array::from_fn(|t| {
                if MASK & (1 << t) == 0 {
                    &[]
                } else {
                    // SAFETY: as in `axpy_segment_f64`.
                    let lanes = unsafe { core::slice::from_raw_parts(bases[t].offset(path[t]), u) };
                    &S::as_simd_f32s(lanes).0[$index..$index + $len]
                }
            });
        };
    }

    let (groups, _) = groups.as_chunks_mut::<CHUNK_VECTORS>();
    for (chunk, group) in groups.iter_mut().enumerate() {
        let index = chunk * CHUNK_VECTORS;
        let mut accumulators: [S::f32s; CHUNK_VECTORS] = *group;
        for local in start..end {
            path!(local, index, CHUNK_VECTORS, coefficient, splats, heads);
            for k in 0..CHUNK_VECTORS {
                let mut value = coefficient;
                for t in 0..N {
                    let operand = if MASK & (1 << t) == 0 {
                        splats[t]
                    } else {
                        heads[t][k]
                    };
                    value = simd.mul_f32s(value, operand);
                }
                accumulators[k] = simd.add_f32s(accumulators[k], value);
            }
        }
        *group = accumulators;
    }

    for (offset, slot) in rest.iter_mut().enumerate() {
        let index = full + offset;
        let mut accumulator = *slot;
        for local in start..end {
            path!(local, index, 1, coefficient, splats, heads);
            let mut value = coefficient;
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    splats[t]
                } else {
                    heads[t][0]
                };
                value = simd.mul_f32s(value, operand);
            }
            accumulator = simd.add_f32s(accumulator, value);
        }
        *slot = accumulator;
    }

    let first_tail = u - tail.len();
    for (offset, slot) in tail.iter_mut().enumerate() {
        let lane = first_tail + offset;
        let mut accumulator = *slot;
        for local in start..end {
            let path = &offsets[local * (N + 1)..local * (N + 1) + N];
            let mut value = coeffs[local];
            for t in 0..N {
                let operand = if MASK & (1 << t) == 0 {
                    // SAFETY: as in `axpy_segment_f64`.
                    unsafe { *bases[t].offset(path[t]) }
                } else {
                    // SAFETY: as in `axpy_segment_f64`.
                    unsafe { *bases[t].offset(path[t] + lane as isize) }
                };
                value *= operand;
            }
            accumulator += value;
        }
        *slot = accumulator;
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
