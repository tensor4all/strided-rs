//! Zero-length erased buffers must still be aligned (#434).
//!
//! `ErasedRawStridedRef::data_as`, `ErasedRawStridedMut::{data_as, data_as_mut}`
//! and `ErasedRawStridedUninitMut::data_as_uninit_mut` form `&[T]`/`&mut [T]`
//! from the descriptor's base pointer, and `slice::from_raw_parts` requires an
//! aligned pointer even when the length is zero. A misaligned empty buffer
//! therefore used to be UB, and aborted under debug assertions.

use core::ptr::NonNull;
use strided_view::{
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    KernelDType,
};

#[test]
fn empty_erased_buffers_must_still_be_aligned() {
    let dims = [0usize];
    let strides = [1isize];
    let mut storage = [1.0f64; 2];
    // One byte into an `f64` allocation: the provenance is valid, the address
    // is not `f64`-aligned.
    let misaligned = NonNull::new(storage.as_mut_ptr().cast::<u8>().wrapping_add(1)).unwrap();

    // Every constructor rejects it, so no `data_as*` ever forms a `&[f64]` from
    // an unaligned base.
    macro_rules! rejects {
        ($constructor:expr) => {
            assert!(matches!(
                $constructor,
                Err(strided_view::StridedError::DataAlignmentMismatch { alignment: 8, .. })
            ))
        };
    }
    rejects!(unsafe {
        ErasedRawStridedPtr::from_raw_parts(KernelDType::F64, misaligned, 0, &dims, &strides, 0)
    });
    rejects!(unsafe {
        ErasedRawStridedRef::from_raw_parts(KernelDType::F64, misaligned, 0, &dims, &strides, 0)
    });
    rejects!(unsafe {
        ErasedRawStridedMut::from_raw_parts(KernelDType::F64, misaligned, 0, &dims, &strides, 0)
    });
    rejects!(unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(
            KernelDType::F64,
            misaligned,
            0,
            &dims,
            &strides,
            0,
        )
    });

    // An aligned zero-length buffer stays accepted by all four accessors.
    let aligned = NonNull::<f64>::dangling().cast::<u8>();
    let pointer = unsafe {
        ErasedRawStridedPtr::from_raw_parts(KernelDType::F64, aligned, 0, &dims, &strides, 0)
    }
    .unwrap();
    let reference = unsafe { pointer.try_as_ref_after_no_overlap() }.unwrap();
    assert!(reference.data_as::<f64>().unwrap().is_empty());
    let mut mutable = unsafe {
        ErasedRawStridedMut::from_raw_parts(KernelDType::F64, aligned, 0, &dims, &strides, 0)
    }
    .unwrap();
    assert!(mutable.data_as::<f64>().unwrap().is_empty());
    assert!(mutable.data_as_mut::<f64>().unwrap().is_empty());
    let mut uninitialized = unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(KernelDType::F64, aligned, 0, &dims, &strides, 0)
    }
    .unwrap();
    assert!(uninitialized
        .data_as_uninit_mut::<f64>()
        .unwrap()
        .is_empty());
}
