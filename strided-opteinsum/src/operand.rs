use num_complex::Complex64;
use num_traits::Zero;
use strided_kernel::copy_into;
use strided_view::{col_major_strides, ElementOpApply, StridedArray, StridedView};

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

    /// Create an `EinsumOperand` from a borrowed f64 strided view.
    pub fn from_view_f64(view: &StridedView<'a, f64>) -> Self {
        EinsumOperand::F64(StridedData::View(view.clone()))
    }

    /// Create an `EinsumOperand` from a borrowed Complex64 strided view.
    pub fn from_view_c64(view: &StridedView<'a, Complex64>) -> Self {
        EinsumOperand::C64(StridedData::View(view.clone()))
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
        let op = EinsumOperand::from_view_f64(&view);
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
}
