use num_complex::Complex64;
use num_traits::Zero;
use strided_einsum2::Scalar;
use strided_kernel::copy_into;
use strided_view::{col_major_strides, ElementOpApply, StridedArray, StridedView};

use crate::typed_tensor::TypedTensor;

/// Type-erased strided data that can be either owned or borrowed.
#[derive(Debug)]
pub enum StridedData<'a, T> {
    Owned(StridedArray<T>),
    View(StridedView<'a, T>),
}

impl<'a, T> StridedData<'a, T> {
    /// Return the dimensions of the underlying data.
    pub fn dims(&self) -> &[usize] {
        match self {
            StridedData::Owned(arr) => arr.dims(),
            StridedData::View(view) => view.dims(),
        }
    }

    /// Return an immutable strided view over the data.
    pub fn as_view(&self) -> StridedView<'_, T> {
        match self {
            StridedData::Owned(arr) => arr.view(),
            StridedData::View(view) => view.clone(),
        }
    }

    /// Return a reference to the owned array.
    ///
    /// # Panics
    /// Panics if this is a `View` variant.
    pub fn as_array(&self) -> &StridedArray<T> {
        match self {
            StridedData::Owned(arr) => arr,
            StridedData::View(_) => panic!("StridedData::as_array called on a View variant"),
        }
    }
}

impl<'a, T> StridedData<'a, T> {
    /// Permute dimensions (metadata-only reorder, no data copy).
    pub fn permuted(self, perm: &[usize]) -> strided_view::Result<Self> {
        match self {
            StridedData::Owned(arr) => Ok(StridedData::Owned(arr.permuted(perm)?)),
            StridedData::View(view) => Ok(StridedData::View(view.permute(perm)?)),
        }
    }
}

impl<'a, T> StridedData<'a, T>
where
    T: Copy + ElementOpApply + Send + Sync + Zero + Default,
{
    /// Convert into an owned `StridedArray`.
    ///
    /// If already owned, returns the inner array directly.
    /// If a view, copies the data into a new column-major array.
    pub fn into_array(self) -> StridedArray<T> {
        match self {
            StridedData::Owned(arr) => arr,
            StridedData::View(view) => {
                let dims = view.dims().to_vec();
                let mut dest = StridedArray::<T>::col_major(&dims);
                let mut dest_view = dest.view_mut();
                copy_into(&mut dest_view, &view).expect("copy_into failed in into_array");
                dest
            }
        }
    }
}

/// A type-erased einsum operand holding either f64 or Complex64 strided data.
#[derive(Debug)]
pub enum EinsumOperand<'a> {
    F64(StridedData<'a, f64>),
    C64(StridedData<'a, Complex64>),
}

impl<'a> EinsumOperand<'a> {
    /// Returns `true` if this operand holds f64 data.
    pub fn is_f64(&self) -> bool {
        matches!(self, EinsumOperand::F64(_))
    }

    /// Returns `true` if this operand holds Complex64 data.
    pub fn is_c64(&self) -> bool {
        matches!(self, EinsumOperand::C64(_))
    }

    /// Return the dimensions of the underlying data.
    pub fn dims(&self) -> &[usize] {
        match self {
            EinsumOperand::F64(data) => data.dims(),
            EinsumOperand::C64(data) => data.dims(),
        }
    }

    /// Permute dimensions (metadata-only reorder, no data copy).
    pub fn permuted(self, perm: &[usize]) -> crate::Result<Self> {
        match self {
            EinsumOperand::F64(data) => Ok(EinsumOperand::F64(data.permuted(perm)?)),
            EinsumOperand::C64(data) => Ok(EinsumOperand::C64(data.permuted(perm)?)),
        }
    }

    /// Create an `EinsumOperand` from a borrowed strided view.
    ///
    /// Type inference selects the correct variant (`F64` or `C64`) from the view's element type.
    pub fn from_view<T: EinsumScalar>(view: &StridedView<'a, T>) -> Self {
        T::wrap_data(StridedData::View(view.clone()))
    }

    /// Promote to an owned Complex64 operand by borrowing the data.
    ///
    /// Unlike `to_c64_owned`, this works on `&self` and always copies.
    pub fn to_c64_owned_ref(&self) -> EinsumOperand<'static> {
        match self {
            EinsumOperand::C64(data) => {
                let view = data.as_view();
                let dims = view.dims().to_vec();
                let mut dest = StridedArray::<Complex64>::col_major(&dims);
                copy_into(&mut dest.view_mut(), &view).expect("copy_into failed");
                EinsumOperand::C64(StridedData::Owned(dest))
            }
            EinsumOperand::F64(data) => {
                let view = data.as_view();
                let dims = view.dims().to_vec();
                let strides = col_major_strides(&dims);
                let mut f64_dest = StridedArray::<f64>::col_major(&dims);
                copy_into(&mut f64_dest.view_mut(), &view).expect("copy_into failed");
                let c64_data: Vec<Complex64> = f64_dest
                    .data()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                let c64_array = StridedArray::from_parts(c64_data, &dims, &strides, 0)
                    .expect("from_parts failed");
                EinsumOperand::C64(StridedData::Owned(c64_array))
            }
        }
    }

    /// Promote to an owned Complex64 operand.
    ///
    /// - If already C64 and owned, returns as-is.
    /// - If C64 view, copies into an owned array.
    /// - If F64, converts each element to `Complex64` and returns an owned array.
    pub fn to_c64_owned(self) -> EinsumOperand<'static> {
        match self {
            EinsumOperand::C64(data) => EinsumOperand::C64(StridedData::Owned(data.into_array())),
            EinsumOperand::F64(data) => {
                let view = data.as_view();
                let dims = view.dims().to_vec();
                let strides = col_major_strides(&dims);
                // Build a new col-major array by copying data through the view
                // First materialize f64 into a col-major owned array, then convert
                let f64_array = match data {
                    StridedData::Owned(arr) => arr,
                    StridedData::View(v) => {
                        let mut dest = StridedArray::<f64>::col_major(v.dims());
                        let mut dest_view = dest.view_mut();
                        copy_into(&mut dest_view, &v).expect("copy_into failed in to_c64_owned");
                        dest
                    }
                };
                let c64_data: Vec<Complex64> = f64_array
                    .data()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                let c64_array = StridedArray::from_parts(c64_data, &dims, &strides, 0)
                    .expect("from_parts failed in to_c64_owned");
                EinsumOperand::C64(StridedData::Owned(c64_array))
            }
        }
    }
}

impl From<StridedArray<f64>> for EinsumOperand<'static> {
    fn from(arr: StridedArray<f64>) -> Self {
        EinsumOperand::F64(StridedData::Owned(arr))
    }
}

impl From<StridedArray<Complex64>> for EinsumOperand<'static> {
    fn from(arr: StridedArray<Complex64>) -> Self {
        EinsumOperand::C64(StridedData::Owned(arr))
    }
}

// ---------------------------------------------------------------------------
// EinsumScalar trait — sealed, implemented for f64 and Complex64
// ---------------------------------------------------------------------------

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex64 {}
}

/// Scalar types that can be used as the output element type for `einsum_into`.
///
/// Sealed trait: only implemented for `f64` and `Complex64`.
pub trait EinsumScalar: private::Sealed + Scalar + Default + 'static {
    /// Human-readable type name for error messages.
    fn type_name() -> &'static str;

    /// Wrap typed `StridedData` into a type-erased `EinsumOperand`.
    fn wrap_data(data: StridedData<'_, Self>) -> EinsumOperand<'_>;

    /// Wrap an owned `StridedArray` into a type-erased `EinsumOperand`.
    fn wrap_array(arr: StridedArray<Self>) -> EinsumOperand<'static>;

    /// Wrap an owned `StridedArray` into a `TypedTensor`.
    fn wrap_typed_tensor(arr: StridedArray<Self>) -> TypedTensor;

    /// Extract typed data from a type-erased `EinsumOperand`, promoting if needed.
    ///
    /// For `f64`: returns error if operand is `C64`.
    /// For `Complex64`: promotes `F64` operands to `C64`.
    fn extract_data<'a>(op: EinsumOperand<'a>) -> crate::Result<StridedData<'a, Self>>;

    /// Check whether any operand requires this type or is incompatible.
    /// Returns error early if `T = f64` but any operand is `C64`.
    fn validate_operands(ops: &[Option<EinsumOperand<'_>>]) -> crate::Result<()>;
}

impl EinsumScalar for f64 {
    fn type_name() -> &'static str {
        "f64"
    }

    fn wrap_data(data: StridedData<'_, Self>) -> EinsumOperand<'_> {
        EinsumOperand::F64(data)
    }

    fn wrap_array(arr: StridedArray<Self>) -> EinsumOperand<'static> {
        EinsumOperand::F64(StridedData::Owned(arr))
    }

    fn wrap_typed_tensor(arr: StridedArray<Self>) -> TypedTensor {
        TypedTensor::F64(arr)
    }

    fn extract_data<'a>(op: EinsumOperand<'a>) -> crate::Result<StridedData<'a, f64>> {
        match op {
            EinsumOperand::F64(data) => Ok(data),
            EinsumOperand::C64(_) => Err(crate::EinsumError::TypeMismatch {
                output_type: "f64",
                computed_type: "Complex64",
            }),
        }
    }

    fn validate_operands(ops: &[Option<EinsumOperand<'_>>]) -> crate::Result<()> {
        for op in ops.iter().flatten() {
            if op.is_c64() {
                return Err(crate::EinsumError::TypeMismatch {
                    output_type: "f64",
                    computed_type: "Complex64",
                });
            }
        }
        Ok(())
    }
}

impl EinsumScalar for Complex64 {
    fn type_name() -> &'static str {
        "Complex64"
    }

    fn wrap_data(data: StridedData<'_, Self>) -> EinsumOperand<'_> {
        EinsumOperand::C64(data)
    }

    fn wrap_array(arr: StridedArray<Self>) -> EinsumOperand<'static> {
        EinsumOperand::C64(StridedData::Owned(arr))
    }

    fn wrap_typed_tensor(arr: StridedArray<Self>) -> TypedTensor {
        TypedTensor::C64(arr)
    }

    fn extract_data<'a>(op: EinsumOperand<'a>) -> crate::Result<StridedData<'a, Complex64>> {
        match op {
            EinsumOperand::C64(data) => Ok(data),
            EinsumOperand::F64(data) => {
                // Promote f64 → Complex64.
                // First materialize to col-major f64, then convert elements.
                let view = data.as_view();
                let dims = view.dims().to_vec();
                let strides = col_major_strides(&dims);
                let mut f64_col = StridedArray::<f64>::col_major(&dims);
                copy_into(&mut f64_col.view_mut(), &view)
                    .expect("copy_into failed in extract_data");
                let c64_data: Vec<Complex64> = f64_col
                    .data()
                    .iter()
                    .map(|&x| Complex64::new(x, 0.0))
                    .collect();
                let c64_array = StridedArray::from_parts(c64_data, &dims, &strides, 0)
                    .expect("from_parts failed in extract_data");
                Ok(StridedData::Owned(c64_array))
            }
        }
    }

    fn validate_operands(_ops: &[Option<EinsumOperand<'_>>]) -> crate::Result<()> {
        // Complex64 output accepts both f64 and c64 operands
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use strided_view::StridedArray;

    #[test]
    fn test_f64_owned() {
        let arr = StridedArray::<f64>::col_major(&[2, 3]);
        let op = EinsumOperand::from(arr);
        assert!(op.is_f64());
        assert!(!op.is_c64());
        assert_eq!(op.dims(), &[2, 3]);
    }

    #[test]
    fn test_c64_owned() {
        let arr = StridedArray::<Complex64>::col_major(&[4, 5]);
        let op = EinsumOperand::from(arr);
        assert!(op.is_c64());
        assert_eq!(op.dims(), &[4, 5]);
    }

    #[test]
    fn test_f64_view() {
        let arr = StridedArray::<f64>::col_major(&[2, 3]);
        let view = arr.view();
        let op = EinsumOperand::from_view(&view);
        assert!(op.is_f64());
        assert_eq!(op.dims(), &[2, 3]);
    }

    #[test]
    fn test_promote_f64_to_c64() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 2]);
        arr.data_mut()[0] = 1.0;
        arr.data_mut()[1] = 2.0;
        arr.data_mut()[2] = 3.0;
        arr.data_mut()[3] = 4.0;
        let op = EinsumOperand::from(arr);
        let promoted = op.to_c64_owned();
        assert!(promoted.is_c64());
        match &promoted {
            EinsumOperand::C64(StridedData::Owned(arr)) => {
                assert_eq!(arr.data()[0], Complex64::new(1.0, 0.0));
                assert_eq!(arr.data()[1], Complex64::new(2.0, 0.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }

    // -----------------------------------------------------------------------
    // to_c64_owned_ref tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_c64_owned_ref_from_f64_owned() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 2]);
        arr.data_mut()[0] = 1.0;
        arr.data_mut()[1] = 2.0;
        arr.data_mut()[2] = 3.0;
        arr.data_mut()[3] = 4.0;
        let op = EinsumOperand::from(arr);
        let promoted = op.to_c64_owned_ref();
        assert!(promoted.is_c64());
        match &promoted {
            EinsumOperand::C64(StridedData::Owned(arr)) => {
                assert_eq!(arr.dims(), &[2, 2]);
                assert_eq!(arr.data()[0], Complex64::new(1.0, 0.0));
                assert_eq!(arr.data()[1], Complex64::new(2.0, 0.0));
                assert_eq!(arr.data()[2], Complex64::new(3.0, 0.0));
                assert_eq!(arr.data()[3], Complex64::new(4.0, 0.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }

    #[test]
    fn test_to_c64_owned_ref_from_f64_view() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 2]);
        arr.data_mut()[0] = 5.0;
        arr.data_mut()[1] = 6.0;
        arr.data_mut()[2] = 7.0;
        arr.data_mut()[3] = 8.0;
        let view = arr.view();
        let op = EinsumOperand::from_view(&view);
        let promoted = op.to_c64_owned_ref();
        assert!(promoted.is_c64());
        match &promoted {
            EinsumOperand::C64(StridedData::Owned(c_arr)) => {
                assert_eq!(c_arr.dims(), &[2, 2]);
                assert_eq!(c_arr.data()[0], Complex64::new(5.0, 0.0));
                assert_eq!(c_arr.data()[1], Complex64::new(6.0, 0.0));
                assert_eq!(c_arr.data()[2], Complex64::new(7.0, 0.0));
                assert_eq!(c_arr.data()[3], Complex64::new(8.0, 0.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }

    #[test]
    fn test_to_c64_owned_ref_from_c64_owned() {
        let mut arr = StridedArray::<Complex64>::col_major(&[2, 2]);
        arr.data_mut()[0] = Complex64::new(1.0, 2.0);
        arr.data_mut()[1] = Complex64::new(3.0, 4.0);
        arr.data_mut()[2] = Complex64::new(5.0, 6.0);
        arr.data_mut()[3] = Complex64::new(7.0, 8.0);
        let op = EinsumOperand::from(arr);
        let copied = op.to_c64_owned_ref();
        assert!(copied.is_c64());
        match &copied {
            EinsumOperand::C64(StridedData::Owned(c_arr)) => {
                assert_eq!(c_arr.dims(), &[2, 2]);
                assert_eq!(c_arr.data()[0], Complex64::new(1.0, 2.0));
                assert_eq!(c_arr.data()[1], Complex64::new(3.0, 4.0));
                assert_eq!(c_arr.data()[2], Complex64::new(5.0, 6.0));
                assert_eq!(c_arr.data()[3], Complex64::new(7.0, 8.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }

    #[test]
    fn test_to_c64_owned_ref_from_c64_view() {
        let mut arr = StridedArray::<Complex64>::col_major(&[3]);
        arr.data_mut()[0] = Complex64::new(1.0, -1.0);
        arr.data_mut()[1] = Complex64::new(2.0, -2.0);
        arr.data_mut()[2] = Complex64::new(3.0, -3.0);
        let view = arr.view();
        let op = EinsumOperand::from_view(&view);
        let copied = op.to_c64_owned_ref();
        assert!(copied.is_c64());
        match &copied {
            EinsumOperand::C64(StridedData::Owned(c_arr)) => {
                assert_eq!(c_arr.dims(), &[3]);
                assert_eq!(c_arr.data()[0], Complex64::new(1.0, -1.0));
                assert_eq!(c_arr.data()[1], Complex64::new(2.0, -2.0));
                assert_eq!(c_arr.data()[2], Complex64::new(3.0, -3.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }

    // -----------------------------------------------------------------------
    // StridedData::into_array tests (View variant)
    // -----------------------------------------------------------------------

    #[test]
    fn test_strided_data_into_array_from_owned() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 3]);
        for (i, v) in arr.data_mut().iter_mut().enumerate() {
            *v = i as f64;
        }
        let data = StridedData::Owned(arr);
        let result = data.into_array();
        assert_eq!(result.dims(), &[2, 3]);
        assert_eq!(result.data()[0], 0.0);
        assert_eq!(result.data()[5], 5.0);
    }

    #[test]
    fn test_strided_data_into_array_from_view() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 3]);
        for (i, v) in arr.data_mut().iter_mut().enumerate() {
            *v = (i as f64) * 10.0;
        }
        let view = arr.view();
        let data = StridedData::<f64>::View(view);
        let result = data.into_array();
        assert_eq!(result.dims(), &[2, 3]);
        // Values should be copied correctly
        assert_eq!(result.get(&[0, 0]), 0.0);
        assert_eq!(result.get(&[1, 0]), 10.0);
    }

    #[test]
    fn test_strided_data_into_array_from_view_c64() {
        let mut arr = StridedArray::<Complex64>::col_major(&[2, 2]);
        arr.data_mut()[0] = Complex64::new(1.0, 2.0);
        arr.data_mut()[1] = Complex64::new(3.0, 4.0);
        arr.data_mut()[2] = Complex64::new(5.0, 6.0);
        arr.data_mut()[3] = Complex64::new(7.0, 8.0);
        let view = arr.view();
        let data = StridedData::<Complex64>::View(view);
        let result = data.into_array();
        assert_eq!(result.dims(), &[2, 2]);
        assert_eq!(result.get(&[0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(result.get(&[1, 1]), Complex64::new(7.0, 8.0));
    }

    // -----------------------------------------------------------------------
    // StridedData::as_array tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strided_data_as_array_owned() {
        let arr = StridedArray::<f64>::col_major(&[3, 2]);
        let data = StridedData::Owned(arr);
        let array_ref = data.as_array();
        assert_eq!(array_ref.dims(), &[3, 2]);
    }

    #[test]
    #[should_panic(expected = "StridedData::as_array called on a View variant")]
    fn test_strided_data_as_array_view_panics() {
        let arr = StridedArray::<f64>::col_major(&[3, 2]);
        let view = arr.view();
        let data = StridedData::<f64>::View(view);
        let _ = data.as_array(); // should panic
    }

    // -----------------------------------------------------------------------
    // EinsumScalar::validate_operands tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_operands_f64_all_f64() {
        let arr1 = StridedArray::<f64>::col_major(&[2, 2]);
        let arr2 = StridedArray::<f64>::col_major(&[2, 2]);
        let ops: Vec<Option<EinsumOperand>> = vec![
            Some(EinsumOperand::from(arr1)),
            Some(EinsumOperand::from(arr2)),
        ];
        assert!(f64::validate_operands(&ops).is_ok());
    }

    #[test]
    fn test_validate_operands_f64_with_none() {
        // None entries should be skipped without error
        let arr = StridedArray::<f64>::col_major(&[2, 2]);
        let ops: Vec<Option<EinsumOperand>> = vec![Some(EinsumOperand::from(arr)), None];
        assert!(f64::validate_operands(&ops).is_ok());
    }

    #[test]
    fn test_validate_operands_f64_with_c64_returns_error() {
        let f64_arr = StridedArray::<f64>::col_major(&[2, 2]);
        let c64_arr = StridedArray::<Complex64>::col_major(&[2, 2]);
        let ops: Vec<Option<EinsumOperand>> = vec![
            Some(EinsumOperand::from(f64_arr)),
            Some(EinsumOperand::from(c64_arr)),
        ];
        let err = f64::validate_operands(&ops).unwrap_err();
        assert!(matches!(
            err,
            crate::EinsumError::TypeMismatch {
                output_type: "f64",
                computed_type: "Complex64",
            }
        ));
    }

    #[test]
    fn test_validate_operands_c64_accepts_anything() {
        let f64_arr = StridedArray::<f64>::col_major(&[2, 2]);
        let c64_arr = StridedArray::<Complex64>::col_major(&[2, 2]);
        let ops: Vec<Option<EinsumOperand>> = vec![
            Some(EinsumOperand::from(f64_arr)),
            Some(EinsumOperand::from(c64_arr)),
        ];
        assert!(Complex64::validate_operands(&ops).is_ok());
    }

    #[test]
    fn test_validate_operands_c64_all_f64() {
        let arr1 = StridedArray::<f64>::col_major(&[2, 2]);
        let arr2 = StridedArray::<f64>::col_major(&[2, 2]);
        let ops: Vec<Option<EinsumOperand>> = vec![
            Some(EinsumOperand::from(arr1)),
            Some(EinsumOperand::from(arr2)),
        ];
        assert!(Complex64::validate_operands(&ops).is_ok());
    }

    // -----------------------------------------------------------------------
    // EinsumScalar::extract_data tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_data_f64_from_f64() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 2]);
        arr.data_mut()[0] = 42.0;
        let op = EinsumOperand::from(arr);
        let data = f64::extract_data(op).unwrap();
        assert_eq!(data.as_view().get(&[0, 0]), 42.0);
    }

    #[test]
    fn test_extract_data_f64_from_c64_returns_error() {
        let arr = StridedArray::<Complex64>::col_major(&[2, 2]);
        let op = EinsumOperand::from(arr);
        let err = f64::extract_data(op).unwrap_err();
        assert!(matches!(
            err,
            crate::EinsumError::TypeMismatch {
                output_type: "f64",
                computed_type: "Complex64",
            }
        ));
    }

    #[test]
    fn test_extract_data_c64_from_c64() {
        let mut arr = StridedArray::<Complex64>::col_major(&[2, 2]);
        arr.data_mut()[0] = Complex64::new(1.0, 2.0);
        let op = EinsumOperand::from(arr);
        let data = Complex64::extract_data(op).unwrap();
        assert_eq!(data.as_view().get(&[0, 0]), Complex64::new(1.0, 2.0));
    }

    #[test]
    fn test_extract_data_c64_from_f64_promotes() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 2]);
        arr.data_mut()[0] = 5.0;
        arr.data_mut()[1] = 6.0;
        arr.data_mut()[2] = 7.0;
        arr.data_mut()[3] = 8.0;
        let op = EinsumOperand::from(arr);
        let data = Complex64::extract_data(op).unwrap();
        // Data should be promoted from f64 to Complex64
        match &data {
            StridedData::Owned(c_arr) => {
                assert_eq!(c_arr.dims(), &[2, 2]);
                assert_eq!(c_arr.data()[0], Complex64::new(5.0, 0.0));
                assert_eq!(c_arr.data()[1], Complex64::new(6.0, 0.0));
                assert_eq!(c_arr.data()[2], Complex64::new(7.0, 0.0));
                assert_eq!(c_arr.data()[3], Complex64::new(8.0, 0.0));
            }
            StridedData::View(_) => panic!("expected Owned after promotion"),
        }
    }

    // -----------------------------------------------------------------------
    // EinsumScalar::type_name tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_name() {
        assert_eq!(f64::type_name(), "f64");
        assert_eq!(Complex64::type_name(), "Complex64");
    }

    // -----------------------------------------------------------------------
    // StridedData::dims and as_view tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strided_data_dims_and_as_view() {
        let mut arr = StridedArray::<f64>::col_major(&[3, 4]);
        for (i, v) in arr.data_mut().iter_mut().enumerate() {
            *v = i as f64;
        }
        // Test Owned variant
        let owned = StridedData::Owned(arr.clone());
        assert_eq!(owned.dims(), &[3, 4]);
        let owned_view = owned.as_view();
        assert_eq!(owned_view.dims(), &[3, 4]);

        // Test View variant
        let view = arr.view();
        let data_view = StridedData::<f64>::View(view);
        assert_eq!(data_view.dims(), &[3, 4]);
        let view_again = data_view.as_view();
        assert_eq!(view_again.dims(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // StridedData::permuted tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strided_data_permuted_owned() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 3]);
        for (i, v) in arr.data_mut().iter_mut().enumerate() {
            *v = i as f64;
        }
        let data = StridedData::Owned(arr);
        let permuted = data.permuted(&[1, 0]).unwrap();
        assert_eq!(permuted.dims(), &[3, 2]);
    }

    #[test]
    fn test_strided_data_permuted_view() {
        let mut arr = StridedArray::<f64>::col_major(&[2, 3]);
        for (i, v) in arr.data_mut().iter_mut().enumerate() {
            *v = i as f64;
        }
        let view = arr.view();
        let data = StridedData::<f64>::View(view);
        let permuted = data.permuted(&[1, 0]).unwrap();
        assert_eq!(permuted.dims(), &[3, 2]);
    }

    // -----------------------------------------------------------------------
    // EinsumOperand::permuted tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_operand_permuted_f64() {
        let arr = StridedArray::<f64>::col_major(&[2, 3]);
        let op = EinsumOperand::from(arr);
        let permuted = op.permuted(&[1, 0]).unwrap();
        assert!(permuted.is_f64());
        assert_eq!(permuted.dims(), &[3, 2]);
    }

    #[test]
    fn test_einsum_operand_permuted_c64() {
        let arr = StridedArray::<Complex64>::col_major(&[4, 5]);
        let op = EinsumOperand::from(arr);
        let permuted = op.permuted(&[1, 0]).unwrap();
        assert!(permuted.is_c64());
        assert_eq!(permuted.dims(), &[5, 4]);
    }

    // -----------------------------------------------------------------------
    // to_c64_owned edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_c64_owned_c64_view() {
        // C64 View variant should be materialized into Owned
        let mut arr = StridedArray::<Complex64>::col_major(&[2, 2]);
        arr.data_mut()[0] = Complex64::new(1.0, -1.0);
        arr.data_mut()[1] = Complex64::new(2.0, -2.0);
        arr.data_mut()[2] = Complex64::new(3.0, -3.0);
        arr.data_mut()[3] = Complex64::new(4.0, -4.0);
        let view = arr.view();
        let op = EinsumOperand::from_view(&view);
        let owned = op.to_c64_owned();
        assert!(owned.is_c64());
        match &owned {
            EinsumOperand::C64(StridedData::Owned(c_arr)) => {
                assert_eq!(c_arr.dims(), &[2, 2]);
                assert_eq!(c_arr.data()[0], Complex64::new(1.0, -1.0));
                assert_eq!(c_arr.data()[3], Complex64::new(4.0, -4.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }

    #[test]
    fn test_to_c64_owned_c64_already_owned() {
        // C64 Owned should pass through without reallocation
        let mut arr = StridedArray::<Complex64>::col_major(&[2]);
        arr.data_mut()[0] = Complex64::new(10.0, 20.0);
        arr.data_mut()[1] = Complex64::new(30.0, 40.0);
        let op = EinsumOperand::from(arr);
        let owned = op.to_c64_owned();
        assert!(owned.is_c64());
        match &owned {
            EinsumOperand::C64(StridedData::Owned(c_arr)) => {
                assert_eq!(c_arr.data()[0], Complex64::new(10.0, 20.0));
                assert_eq!(c_arr.data()[1], Complex64::new(30.0, 40.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }

    #[test]
    fn test_to_c64_owned_f64_view() {
        // F64 View should be materialized and promoted
        let mut arr = StridedArray::<f64>::col_major(&[3]);
        arr.data_mut()[0] = 10.0;
        arr.data_mut()[1] = 20.0;
        arr.data_mut()[2] = 30.0;
        let view = arr.view();
        let op = EinsumOperand::from_view(&view);
        let promoted = op.to_c64_owned();
        assert!(promoted.is_c64());
        match &promoted {
            EinsumOperand::C64(StridedData::Owned(c_arr)) => {
                assert_eq!(c_arr.dims(), &[3]);
                assert_eq!(c_arr.data()[0], Complex64::new(10.0, 0.0));
                assert_eq!(c_arr.data()[1], Complex64::new(20.0, 0.0));
                assert_eq!(c_arr.data()[2], Complex64::new(30.0, 0.0));
            }
            _ => panic!("expected C64 Owned"),
        }
    }
}
