use bytemuck::{Pod, Zeroable};
use num_complex::Complex;

/// POD representation of `Complex<f64>`: `[re, im]` with `repr(C)`.
#[repr(C)]
#[derive(Copy, Clone, Zeroable, Pod)]
pub struct PodComplexF64 {
    pub re: f64,
    pub im: f64,
}

/// POD representation of `Complex<f32>`: `[re, im]` with `repr(C)`.
#[repr(C)]
#[derive(Copy, Clone, Zeroable, Pod)]
pub struct PodComplexF32 {
    pub re: f32,
    pub im: f32,
}

impl From<Complex<f64>> for PodComplexF64 {
    fn from(c: Complex<f64>) -> Self {
        PodComplexF64 { re: c.re, im: c.im }
    }
}

impl From<PodComplexF64> for Complex<f64> {
    fn from(p: PodComplexF64) -> Self {
        Complex { re: p.re, im: p.im }
    }
}

impl From<Complex<f32>> for PodComplexF32 {
    fn from(c: Complex<f32>) -> Self {
        PodComplexF32 { re: c.re, im: c.im }
    }
}

impl From<PodComplexF32> for Complex<f32> {
    fn from(p: PodComplexF32) -> Self {
        Complex { re: p.re, im: p.im }
    }
}

/// Cast a slice of `Complex<f64>` to a slice of `PodComplexF64` by reinterpreting
/// the underlying bytes. Unsafe: caller must ensure layout compatibility.
pub unsafe fn cast_complex_slice_to_pod_f64(src: &[Complex<f64>]) -> &[PodComplexF64] {
    debug_assert_eq!(
        std::mem::size_of::<Complex<f64>>(),
        std::mem::size_of::<PodComplexF64>()
    );
    debug_assert_eq!(
        std::mem::align_of::<Complex<f64>>(),
        std::mem::align_of::<PodComplexF64>()
    );
    let byte_ptr = src.as_ptr() as *const u8;
    let byte_len = src.len() * std::mem::size_of::<Complex<f64>>();
    let bytes = std::slice::from_raw_parts(byte_ptr, byte_len);
    bytemuck::cast_slice(bytes)
}

/// Mutable variant for `Complex<f64>`.
pub unsafe fn cast_complex_slice_mut_to_pod_f64(dst: &mut [Complex<f64>]) -> &mut [PodComplexF64] {
    debug_assert_eq!(
        std::mem::size_of::<Complex<f64>>(),
        std::mem::size_of::<PodComplexF64>()
    );
    debug_assert_eq!(
        std::mem::align_of::<Complex<f64>>(),
        std::mem::align_of::<PodComplexF64>()
    );
    let byte_ptr = dst.as_mut_ptr() as *mut u8;
    let byte_len = dst.len() * std::mem::size_of::<Complex<f64>>();
    let bytes = std::slice::from_raw_parts_mut(byte_ptr, byte_len);
    bytemuck::cast_slice_mut(bytes)
}

/// Cast helpers for `Complex<f32>`.
pub unsafe fn cast_complex_slice_to_pod_f32(src: &[Complex<f32>]) -> &[PodComplexF32] {
    debug_assert_eq!(
        std::mem::size_of::<Complex<f32>>(),
        std::mem::size_of::<PodComplexF32>()
    );
    debug_assert_eq!(
        std::mem::align_of::<Complex<f32>>(),
        std::mem::align_of::<PodComplexF32>()
    );
    let byte_ptr = src.as_ptr() as *const u8;
    let byte_len = src.len() * std::mem::size_of::<Complex<f32>>();
    let bytes = std::slice::from_raw_parts(byte_ptr, byte_len);
    bytemuck::cast_slice(bytes)
}

pub unsafe fn cast_complex_slice_mut_to_pod_f32(dst: &mut [Complex<f32>]) -> &mut [PodComplexF32] {
    debug_assert_eq!(
        std::mem::size_of::<Complex<f32>>(),
        std::mem::size_of::<PodComplexF32>()
    );
    debug_assert_eq!(
        std::mem::align_of::<Complex<f32>>(),
        std::mem::align_of::<PodComplexF32>()
    );
    let byte_ptr = dst.as_mut_ptr() as *mut u8;
    let byte_len = dst.len() * std::mem::size_of::<Complex<f32>>();
    let bytes = std::slice::from_raw_parts_mut(byte_ptr, byte_len);
    bytemuck::cast_slice_mut(bytes)
}
