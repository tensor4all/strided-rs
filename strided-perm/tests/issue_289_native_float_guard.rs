//! Regression tests for tensor4all/strided-rs#289.
//!
//! The HPTT 4/8-byte path reinterprets the typed buffer as native `f32`/`f64`.
//! It must be restricted to the exact primitive types with a matching
//! alignment; an arbitrary same-size `Copy` type with weaker alignment (for
//! example an 8-byte / align-2 array) has to take the generic element-wise
//! path instead of a misaligned float access.

use strided_perm::copy_into;
use strided_view::{StridedView, StridedViewMut};

/// A 4-byte type with alignment 1.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct U32Align1([u8; 4]);

/// An 8-byte type with alignment 2.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct U64Align2([u16; 4]);

/// An 8-byte type with alignment 1.
#[repr(C, packed)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct U64Align1([u8; 8]);

/// Copy `[3, 4]` with a transposing layout: source row-major `[4, 1]`,
/// destination col-major `[1, 3]`. The planner selects the 2D micro-kernel
/// path, which for the size-only match would reinterpret `T` as `f32`/`f64`.
fn transposing_copy<T: Copy + 'static>(src_data: &[T], dst_data: &mut [T]) {
    let dims = [3usize, 4];
    let src = StridedView::new(src_data, &dims, &[4, 1], 0).unwrap();
    let mut dst = StridedViewMut::new(dst_data, &dims, &[1, 3], 0).unwrap();
    copy_into(&mut dst, &src).unwrap();
}

fn assert_transposing_copy<T: Copy + 'static + PartialEq + core::fmt::Debug>(
    src_data: &[T],
    dst_data: &mut [T],
) {
    transposing_copy(src_data, dst_data);
    // logical (i, j) maps to src[4 i + j] and dst[i + 3 j].
    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(dst_data[i + 3 * j], src_data[4 * i + j], "at ({i}, {j})");
        }
    }
}

#[test]
fn four_byte_alignment_one_struct_uses_the_generic_path() {
    assert_eq!(std::mem::size_of::<U32Align1>(), 4);
    assert_eq!(std::mem::align_of::<U32Align1>(), 1);
    let src_data: Vec<U32Align1> = (0..12).map(|i| U32Align1([i as u8; 4])).collect();
    let mut dst_data = vec![U32Align1::default(); 12];
    assert_transposing_copy(&src_data, &mut dst_data);
}

#[test]
fn eight_byte_alignment_two_struct_uses_the_generic_path() {
    assert_eq!(std::mem::size_of::<U64Align2>(), 8);
    assert_eq!(std::mem::align_of::<U64Align2>(), 2);
    let src_data: Vec<U64Align2> = (0..12).map(|i| U64Align2([i as u16; 4])).collect();
    let mut dst_data = vec![U64Align2::default(); 12];
    assert_transposing_copy(&src_data, &mut dst_data);
}

#[test]
fn eight_byte_alignment_one_struct_uses_the_generic_path() {
    assert_eq!(std::mem::size_of::<U64Align1>(), 8);
    assert_eq!(std::mem::align_of::<U64Align1>(), 1);
    let src_data: Vec<U64Align1> = (0..12).map(|i| U64Align1([i as u8; 8])).collect();
    let mut dst_data = vec![U64Align1::default(); 12];
    assert_transposing_copy(&src_data, &mut dst_data);
}

/// The primitive types the guard admits keep the existing fast path and stay
/// correct, including the integer types whose alignment matches their size.
#[test]
fn matching_primitive_types_still_copy() {
    let src_f64: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let mut dst_f64 = vec![0.0f64; 12];
    assert_transposing_copy(&src_f64, &mut dst_f64);

    let src_f32: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let mut dst_f32 = vec![0.0f32; 12];
    assert_transposing_copy(&src_f32, &mut dst_f32);

    let src_i64: Vec<i64> = (0..12).collect();
    let mut dst_i64 = vec![0i64; 12];
    assert_transposing_copy(&src_i64, &mut dst_i64);

    let src_u32: Vec<u32> = (0..12).collect();
    let mut dst_u32 = vec![0u32; 12];
    assert_transposing_copy(&src_u32, &mut dst_u32);
}
