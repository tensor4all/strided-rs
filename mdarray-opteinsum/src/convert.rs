use std::mem::ManuallyDrop;

use mdarray::{Array, DynRank, Shape, View, ViewMut};
use strided_kernel::copy_into;
use strided_view::{StridedArray, StridedView, StridedViewMut, row_major_strides};

/// Reverse the index labels in an einsum notation string.
///
/// Each operand's labels and the output labels are reversed to convert
/// between row-major (mdarray) and column-major (strided) conventions.
///
/// Example: `"(ij,jk),kl->il"` becomes `"(ji,kj),lk->li"`
pub fn reverse_notation(notation: &str) -> String {
    let s: String = notation.chars().filter(|c| !c.is_whitespace()).collect();

    let arrow_pos = s.find("->").expect("missing '->' in einsum notation");
    let lhs = &s[..arrow_pos];
    let rhs = &s[arrow_pos + 2..];

    let reversed_lhs = reverse_lhs(lhs);
    let reversed_rhs: String = rhs.chars().rev().collect();

    format!("{}->{}", reversed_lhs, reversed_rhs)
}

/// Reverse index labels in the LHS of einsum notation, preserving structure.
fn reverse_lhs(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut label_buf = String::new();

    for c in s.chars() {
        match c {
            '(' | ')' | ',' => {
                // Flush accumulated labels (reversed)
                if !label_buf.is_empty() {
                    result.extend(label_buf.chars().rev());
                    label_buf.clear();
                }
                result.push(c);
            }
            _ => {
                label_buf.push(c);
            }
        }
    }
    // Flush trailing labels
    if !label_buf.is_empty() {
        result.extend(label_buf.chars().rev());
    }
    result
}

/// Wrap an mdarray `Array<T, DynRank>` as a `StridedView` with reversed dims.
///
/// Zero-copy: the view borrows the array's data.
pub fn array_to_strided_view<T>(arr: &Array<T, DynRank>) -> StridedView<'_, T> {
    let dims: Vec<usize> = arr.dims().iter().rev().copied().collect();
    let strides = row_major_strides(arr.dims());
    let reversed_strides: Vec<isize> = strides.iter().rev().copied().collect();
    let len = arr.len();
    let data = unsafe { std::slice::from_raw_parts(arr.as_ptr(), len) };
    StridedView::new(data, &dims, &reversed_strides, 0).expect("valid bounds")
}

/// Wrap an mdarray `View<T, DynRank>` as a `StridedView` with reversed dims.
///
/// Zero-copy: the view borrows the same data.
pub fn view_to_strided_view<'a, T>(view: &'a View<'a, T, DynRank>) -> StridedView<'a, T> {
    let dims: Vec<usize> = view.dims().iter().rev().copied().collect();
    let strides = row_major_strides(view.dims());
    let reversed_strides: Vec<isize> = strides.iter().rev().copied().collect();
    let len = view.len();
    let data = unsafe { std::slice::from_raw_parts(view.as_ptr(), len) };
    StridedView::new(data, &dims, &reversed_strides, 0).expect("valid bounds")
}

/// Wrap an mdarray `ViewMut<T, DynRank>` as a `StridedViewMut` with reversed dims.
///
/// Zero-copy: the mutable view borrows the same data.
pub fn view_mut_to_strided_view_mut<'a, T>(
    view: &'a mut ViewMut<'a, T, DynRank>,
) -> StridedViewMut<'a, T> {
    let dims: Vec<usize> = view.dims().iter().rev().copied().collect();
    let strides = row_major_strides(view.dims());
    let reversed_strides: Vec<isize> = strides.iter().rev().copied().collect();
    let len = view.len();
    let data = unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), len) };
    StridedViewMut::new(data, &dims, &reversed_strides, 0).expect("valid bounds")
}

/// Convert a `StridedArray` result into an mdarray `Array<T, DynRank>`.
///
/// The result's dims are reversed to match mdarray's row-major convention.
/// A copy is performed to materialize the data into dense row-major order,
/// since mdarray only supports dense (implicit-stride) layout.
pub fn strided_array_to_mdarray<T>(arr: StridedArray<T>) -> Array<T, DynRank>
where
    T: Copy + strided_view::ElementOpApply + Send + Sync + num_traits::Zero + Default,
{
    let src_view = arr.view();
    let reversed_dims: Vec<usize> = arr.dims().iter().rev().copied().collect();
    let reversed_strides: Vec<isize> = arr.strides().iter().rev().copied().collect();

    // Create a dest StridedViewMut with row-major layout (= reversed strides)
    // and copy from the source view, letting copy_into handle arbitrary strides.
    let total: usize = reversed_dims.iter().product();
    let mut buf = vec![T::default(); total];
    let rm_strides = row_major_strides(&reversed_dims);
    {
        let mut dest =
            StridedViewMut::new(&mut buf, &reversed_dims, &rm_strides, 0).expect("valid dest");
        // Wrap source data with reversed dims/strides so copy_into can match shapes.
        let src_reversed: StridedView<'_, T> =
            StridedView::new(src_view.data(), &reversed_dims, &reversed_strides, 0)
                .expect("valid source");
        copy_into(&mut dest, &src_reversed).expect("copy_into failed");
    }

    let shape: DynRank = Shape::from_dims(&reversed_dims);
    let mut buf = ManuallyDrop::new(buf);
    let capacity = buf.capacity();
    let ptr = buf.as_mut_ptr();
    unsafe { Array::from_raw_parts(ptr, shape, capacity) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_flat() {
        assert_eq!(reverse_notation("ij,jk->ik"), "ji,kj->ki");
    }

    #[test]
    fn test_reverse_nested() {
        assert_eq!(reverse_notation("(ij,jk),kl->il"), "(ji,kj),lk->li");
    }

    #[test]
    fn test_reverse_deep_nested() {
        assert_eq!(
            reverse_notation("((ij,jk),(kl,lm))->im"),
            "((ji,kj),(lk,ml))->mi"
        );
    }

    #[test]
    fn test_reverse_trace() {
        assert_eq!(reverse_notation("ii->"), "ii->");
    }

    #[test]
    fn test_reverse_single_operand() {
        assert_eq!(reverse_notation("ijk->kji"), "kji->ijk");
    }

    #[test]
    fn test_reverse_scalar_output() {
        assert_eq!(reverse_notation("ij,ji->"), "ji,ij->");
    }

    #[test]
    fn test_array_to_strided_view_2d() {
        // Create a row-major 2x3 mdarray: [[1,2,3],[4,5,6]]
        // Memory: [1,2,3,4,5,6]
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape: DynRank = Shape::from_dims(&[2, 3]);
        let arr: Array<f64, DynRank> = unsafe {
            let mut data = ManuallyDrop::new(data);
            Array::from_raw_parts(data.as_mut_ptr(), shape, data.capacity())
        };

        let view = array_to_strided_view(&arr);
        // Reversed dims: [3, 2]
        assert_eq!(view.dims(), &[3, 2]);
        // Row-major strides for [2,3] are [3,1], reversed: [1,3]
        assert_eq!(view.strides(), &[1, 3]);

        // Verify element access via pointer arithmetic:
        // view[col, row] maps to arr[row, col] (transposed indexing)
        let ptr = view.ptr();
        unsafe {
            // view[0,0] = arr[0,0] = 1.0
            assert_eq!(*ptr, 1.0);
            // view[1,0] = arr[0,1] = 2.0 (stride[0]=1)
            assert_eq!(*ptr.offset(1), 2.0);
            // view[0,1] = arr[1,0] = 4.0 (stride[1]=3)
            assert_eq!(*ptr.offset(3), 4.0);
            // view[2,1] = arr[1,2] = 6.0 (2*1 + 1*3 = 5)
            assert_eq!(*ptr.offset(5), 6.0);
        }
    }

    #[test]
    fn test_strided_array_to_mdarray_roundtrip() {
        // Create a col-major StridedArray [3, 2] (result from einsum)
        // col-major strides: [1, 3]
        // Memory: [a00, a10, a20, a01, a11, a21]
        let data = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let arr = StridedArray::from_parts(data, &[3, 2], &[1, 3], 0).expect("valid strided array");

        let md = strided_array_to_mdarray(arr);
        // Reversed dims: [2, 3]
        assert_eq!(md.dims(), &[2, 3]);
        // Row-major mdarray [2,3] with memory [a00, a10, a20, a01, a11, a21]
        // md[0,0] = memory[0] = 1.0, md[0,1] = memory[1] = 4.0, ...
        // This is correct because col-major [3,2] data = row-major [2,3] data
        // (same memory layout when dims are reversed)
    }
}
