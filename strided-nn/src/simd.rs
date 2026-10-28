//! Group kernels for the `f32` softmax reductions over pulp vectors.
//!
//! LLVM vectorizes the `max` fold but not the `+=` sum fold without fast-math,
//! so the `f32` reductions and scale run over explicit vectors. The `f64` group
//! functions keep their scalar source order (a source-order sum) and rely on
//! auto-vectorization of the shift and divide. Both run inside one pulp dispatch
//! per group call, which selects the level at run time (AVX-512, AVX2, NEON or
//! scalar), so a baseline build still reaches AVX-512.

/// `dst = src - max(src)` per slice of `axis_len` elements, vectorized.
pub(crate) fn shift_group_f32(src: &[f32], dst: &mut [f32], axis_len: usize) {
    pulp::Arch::new().dispatch(ShiftGroup { src, dst, axis_len });
}

/// `data /= sum(data)` per slice of `axis_len` elements, with a vectorized sum.
pub(crate) fn scale_group_f32(data: &mut [f32], axis_len: usize) {
    pulp::Arch::new().dispatch(ScaleGroup { data, axis_len });
}

struct ShiftGroup<'a> {
    src: &'a [f32],
    dst: &'a mut [f32],
    axis_len: usize,
}

impl pulp::WithSimd for ShiftGroup<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        for (in_slice, out_slice) in self
            .src
            .chunks(self.axis_len)
            .zip(self.dst.chunks_mut(self.axis_len))
        {
            let (head, tail) = S::as_simd_f32s(in_slice);
            let mut acc = simd.splat_f32s(f32::NEG_INFINITY);
            for &value in head {
                acc = simd.max_f32s(acc, value);
            }
            let mut max = simd.reduce_max_f32s(acc);
            for &value in tail {
                max = max.max(value);
            }
            let safe_max = if max.is_nan() { max } else { max.max(f32::MIN) };
            let shift = simd.splat_f32s(safe_max);
            let (out_head, out_tail) = S::as_mut_simd_f32s(out_slice);
            for (out, &value) in out_head.iter_mut().zip(head) {
                *out = simd.sub_f32s(value, shift);
            }
            for (out, &value) in out_tail.iter_mut().zip(tail) {
                *out = value - safe_max;
            }
        }
    }
}

struct ScaleGroup<'a> {
    data: &'a mut [f32],
    axis_len: usize,
}

impl pulp::WithSimd for ScaleGroup<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        for slice in self.data.chunks_mut(self.axis_len) {
            let (head, tail) = S::as_mut_simd_f32s(slice);
            let mut acc = simd.splat_f32s(0.0);
            for &value in head.iter() {
                acc = simd.add_f32s(acc, value);
            }
            let mut sum = simd.reduce_sum_f32s(acc);
            for &value in tail.iter() {
                sum += value;
            }
            let safe_sum = if sum.is_nan() {
                sum
            } else {
                sum.max(f32::MIN_POSITIVE)
            };
            let divisor = simd.splat_f32s(safe_sum);
            for value in head.iter_mut() {
                *value = simd.div_f32s(*value, divisor);
            }
            for value in tail.iter_mut() {
                *value /= safe_sum;
            }
        }
    }
}

/// `dst = src - max(src)` per slice of `axis_len` elements, in source order.
pub(crate) fn shift_group_f64(src: &[f64], dst: &mut [f64], axis_len: usize) {
    pulp::Arch::new().dispatch(ShiftGroupF64 { src, dst, axis_len });
}

/// `data /= sum(data)` per slice of `axis_len` elements, summed in source order.
pub(crate) fn scale_group_f64(data: &mut [f64], axis_len: usize) {
    pulp::Arch::new().dispatch(ScaleGroupF64 { data, axis_len });
}

struct ShiftGroupF64<'a> {
    src: &'a [f64],
    dst: &'a mut [f64],
    axis_len: usize,
}

impl pulp::WithSimd for ShiftGroupF64<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) {
        for (in_slice, out_slice) in self
            .src
            .chunks(self.axis_len)
            .zip(self.dst.chunks_mut(self.axis_len))
        {
            let max = in_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let safe_max = if max.is_nan() { max } else { max.max(f64::MIN) };
            for (out, &value) in out_slice.iter_mut().zip(in_slice) {
                *out = value - safe_max;
            }
        }
    }
}

struct ScaleGroupF64<'a> {
    data: &'a mut [f64],
    axis_len: usize,
}

impl pulp::WithSimd for ScaleGroupF64<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) {
        for slice in self.data.chunks_mut(self.axis_len) {
            let sum: f64 = slice.iter().copied().sum();
            let safe_sum = if sum.is_nan() {
                sum
            } else {
                sum.max(f64::MIN_POSITIVE)
            };
            for value in slice.iter_mut() {
                *value /= safe_sum;
            }
        }
    }
}
