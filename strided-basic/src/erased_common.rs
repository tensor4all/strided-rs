use crate::*;
/// Reject an input overlapping any byte of the output backing allocation.
///
/// # Examples
///
/// ```
/// use strided_basic::{ErasedRawStridedRef, ErasedRawStridedPtr, ErasedRawStridedUninitMut, execution::validate_uninit_no_overlap};
/// use core::mem::MaybeUninit;
/// let input = ErasedRawStridedRef::from_slice(&[2_i32], &[1], &[1], 0).unwrap();
/// let input = ErasedRawStridedPtr::from_ref(&input);
/// let mut values = [MaybeUninit::<i32>::uninit()];
/// let output = ErasedRawStridedUninitMut::from_uninit_slice(&mut values, &[1], &[1], 0).unwrap();
/// validate_uninit_no_overlap(&output, &input, 0).unwrap();
/// ```
///
/// # Errors
/// Returns an overlap or byte-range overflow error without reading element values.
pub fn validate_uninit_no_overlap(
    dest: &ErasedRawStridedUninitMut<'_>,
    input: &ErasedRawStridedPtr<'_>,
    input_index: usize,
) -> Result<()> {
    if input.overlaps_uninit_mut(dest)? {
        Err(StridedError::OverlappingInputOutput { input: input_index })
    } else {
        Ok(())
    }
}
/// Check a descriptor dtype against a prepared operation.
///
/// # Examples
///
/// ```
/// use strided_basic::{KernelDType, execution::check_dtype};
/// check_dtype(KernelDType::F64, KernelDType::F64).unwrap();
/// assert!(check_dtype(KernelDType::F64, KernelDType::F32).is_err());
/// ```
///
/// # Errors
/// Returns `DTypeMismatch` when the tags differ.
pub fn check_dtype(expected: KernelDType, actual: KernelDType) -> Result<()> {
    if actual != expected {
        return Err(StridedError::DTypeMismatch {
            expected: expected.label(),
            actual: actual.label(),
        });
    }
    Ok(())
}
/// Check whether a dtype has a static-indexing implementation.
///
/// # Examples
///
/// ```
/// use strided_basic::{KernelDType, execution::check_static_indexing_dtype};
/// check_static_indexing_dtype(KernelDType::Bool).unwrap();
/// ```
///
/// # Errors
/// Returns `UnsupportedDType` for unimplemented tags.
pub fn check_static_indexing_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::F32
        | KernelDType::F64
        | KernelDType::I32
        | KernelDType::I64
        | KernelDType::Bool
        | KernelDType::C32
        | KernelDType::C64 => Ok(()),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}
/// Borrow initialized element storage with owning view metadata.
///
/// # Examples
///
/// ```
/// use strided_basic::{ErasedRawStridedRef, execution::erased_view};
/// let values = [2.0_f64, 3.0];
/// let raw = ErasedRawStridedRef::from_slice(&values, &[2], &[1], 0).unwrap();
/// let view = erased_view::<f64>(&raw).unwrap();
/// assert_eq!(view.get(&[1]), 3.0);
/// ```
///
/// # Errors
/// Returns a dtype mismatch if `T` differs from the descriptor tag.
pub fn erased_view<'a, T: KernelStorageElement>(
    src: &'a ErasedRawStridedRef<'a>,
) -> Result<StridedView<'a, T>> {
    let data = src.data_as::<T>()?;
    Ok(unsafe { StridedView::new_unchecked(data, src.dims(), src.strides(), src.offset()) })
}
