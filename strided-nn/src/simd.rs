//! AVX2 group kernels for the `f32` softmax reductions.
//!
//! LLVM vectorizes the `max` fold but not the `+=` sum fold without fast-math,
//! so both reductions and the scale run over explicit `__m256` lanes. Everything
//! is runtime-detected: a host without AVX2 keeps the scalar group functions.

/// Whether this host can run the AVX2 group kernels.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

/// `dst = src - max(src)` per slice, vectorized.
///
/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn shift_group_f32_avx2(src: &[f32], dst: &mut [f32], axis_len: usize) {
    use std::arch::x86_64::*;
    for (in_slice, out_slice) in src.chunks(axis_len).zip(dst.chunks_mut(axis_len)) {
        let n = in_slice.len();
        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: the caller guarantees AVX2 and the loop bound keeps the
            // load inside `in_slice`.
            acc = _mm256_max_ps(acc, unsafe { _mm256_loadu_ps(in_slice.as_ptr().add(i)) });
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        // SAFETY: a full 8-lane store into a stack array.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut max = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &value in &in_slice[i..] {
            max = max.max(value);
        }
        let safe_max = if max.is_nan() { max } else { max.max(f32::MIN) };
        let shift = _mm256_set1_ps(safe_max);
        let mut j = 0;
        while j + 8 <= n {
            // SAFETY: as above, for the matching load and store.
            unsafe {
                let value = _mm256_loadu_ps(in_slice.as_ptr().add(j));
                _mm256_storeu_ps(out_slice.as_mut_ptr().add(j), _mm256_sub_ps(value, shift));
            }
            j += 8;
        }
        for (out, &input) in out_slice[j..].iter_mut().zip(&in_slice[j..]) {
            *out = input - safe_max;
        }
    }
}

/// `data /= sum(data)` per slice, with a vectorized sum, vectorized.
///
/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn scale_group_f32_avx2(data: &mut [f32], axis_len: usize) {
    use std::arch::x86_64::*;
    for slice in data.chunks_mut(axis_len) {
        let n = slice.len();
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: the caller guarantees AVX2 and the loop bound keeps the
            // load inside `slice`.
            acc = _mm256_add_ps(acc, unsafe { _mm256_loadu_ps(slice.as_ptr().add(i)) });
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        // SAFETY: a full 8-lane store into a stack array.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut sum: f32 = lanes.iter().sum();
        for &value in &slice[i..] {
            sum += value;
        }
        let safe_sum = if sum.is_nan() {
            sum
        } else {
            sum.max(f32::MIN_POSITIVE)
        };
        let divisor = _mm256_set1_ps(safe_sum);
        let mut j = 0;
        while j + 8 <= n {
            // SAFETY: as above, for the matching load and store.
            unsafe {
                let value = _mm256_loadu_ps(slice.as_ptr().add(j));
                _mm256_storeu_ps(slice.as_mut_ptr().add(j), _mm256_div_ps(value, divisor));
            }
            j += 8;
        }
        for value in &mut slice[j..] {
            *value /= safe_sum;
        }
    }
}
