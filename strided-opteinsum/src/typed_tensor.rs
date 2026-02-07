use num_complex::Complex64;
use strided_view::StridedArray;

/// A type-erased tensor that dispatches over f64 and Complex64 at runtime.
pub enum TypedTensor {
    F64(StridedArray<f64>),
    C64(StridedArray<Complex64>),
}

const DEMOTE_THRESHOLD: f64 = 1e-15;

impl TypedTensor {
    /// Returns `true` if this tensor holds `f64` data.
    pub fn is_f64(&self) -> bool {
        matches!(self, TypedTensor::F64(_))
    }
    /// Returns `true` if this tensor holds `Complex64` data.
    pub fn is_c64(&self) -> bool {
        matches!(self, TypedTensor::C64(_))
    }

    /// Returns the dimensions of the underlying array.
    pub fn dims(&self) -> &[usize] {
        match self {
            TypedTensor::F64(a) => a.dims(),
            TypedTensor::C64(a) => a.dims(),
        }
    }

    /// Promote to Complex64. If already C64, returns self unchanged.
    pub fn to_c64(self) -> TypedTensor {
        match self {
            TypedTensor::C64(_) => self,
            TypedTensor::F64(a) => {
                let c64_data: Vec<Complex64> =
                    a.data().iter().map(|&x| Complex64::new(x, 0.0)).collect();
                let arr = StridedArray::from_parts(c64_data, a.dims(), a.strides(), 0).unwrap();
                TypedTensor::C64(arr)
            }
        }
    }

    /// Try to demote a Complex64 array to f64 if all imaginary parts are negligible.
    pub fn try_demote_to_f64(arr: StridedArray<Complex64>) -> TypedTensor {
        let all_real = arr.data().iter().all(|c| c.im.abs() < DEMOTE_THRESHOLD);
        if all_real {
            let f64_data: Vec<f64> = arr.data().iter().map(|c| c.re).collect();
            let f_arr = StridedArray::from_parts(f64_data, arr.dims(), arr.strides(), 0).unwrap();
            TypedTensor::F64(f_arr)
        } else {
            TypedTensor::C64(arr)
        }
    }
}

/// Returns true if any input is Complex64 (triggering promotion for all).
pub fn needs_c64_promotion(inputs: &[&TypedTensor]) -> bool {
    inputs.iter().any(|t| t.is_c64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_typed_f64() {
        let arr = StridedArray::<f64>::col_major(&[2, 3]);
        let t = TypedTensor::F64(arr);
        assert!(t.is_f64());
        assert!(!t.is_c64());
        assert_eq!(t.dims(), &[2, 3]);
    }

    #[test]
    fn test_typed_c64() {
        let arr = StridedArray::<Complex64>::col_major(&[3, 4]);
        let t = TypedTensor::C64(arr);
        assert!(t.is_c64());
    }

    #[test]
    fn test_promote_to_c64() {
        let mut arr = StridedArray::<f64>::col_major(&[2]);
        arr.data_mut()[0] = 3.0;
        arr.data_mut()[1] = 7.0;
        let t = TypedTensor::F64(arr);
        let promoted = t.to_c64();
        match promoted {
            TypedTensor::C64(a) => {
                assert_abs_diff_eq!(a.data()[0].re, 3.0);
                assert_abs_diff_eq!(a.data()[1].re, 7.0);
                assert_abs_diff_eq!(a.data()[0].im, 0.0);
            }
            _ => panic!("expected C64"),
        }
    }

    #[test]
    fn test_demote_to_f64() {
        let mut arr = StridedArray::<Complex64>::col_major(&[2]);
        arr.data_mut()[0] = Complex64::new(1.0, 0.0);
        arr.data_mut()[1] = Complex64::new(2.0, 1e-16);
        let t = TypedTensor::try_demote_to_f64(arr);
        assert!(t.is_f64());
    }

    #[test]
    fn test_no_demote_if_complex() {
        let mut arr = StridedArray::<Complex64>::col_major(&[2]);
        arr.data_mut()[0] = Complex64::new(1.0, 0.5);
        arr.data_mut()[1] = Complex64::new(2.0, 0.0);
        let t = TypedTensor::try_demote_to_f64(arr);
        assert!(t.is_c64());
    }

    #[test]
    fn test_needs_c64_promotion() {
        let f = TypedTensor::F64(StridedArray::<f64>::col_major(&[1]));
        let c = TypedTensor::C64(StridedArray::<Complex64>::col_major(&[1]));
        assert!(!needs_c64_promotion(&[&f]));
        assert!(needs_c64_promotion(&[&f, &c]));
        assert!(needs_c64_promotion(&[&c]));
    }
}
