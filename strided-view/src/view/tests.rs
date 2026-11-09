use super::*;

#[test]
fn empty_views_use_dangling_base_pointers_for_extreme_offsets() {
    let dims = [0usize];
    let strides = [1isize];
    let data: [f64; 0] = [];
    let view = StridedView::<f64>::new(&data, &dims, &strides, isize::MAX).unwrap();
    assert_eq!(view.ptr(), empty_const_ptr());

    let mut data: [f64; 0] = [];
    let view = StridedViewMut::new(&mut data, &dims, &strides, isize::MAX).unwrap();
    assert_eq!(view.ptr(), empty_const_ptr());
    assert_eq!(view.as_mut_ptr(), empty_mut_ptr());

    let array = StridedArray::<f64>::from_parts(Vec::new(), &dims, &strides, isize::MAX).unwrap();
    assert_eq!(array.view().ptr(), empty_const_ptr());

    let mut array =
        StridedArray::<f64>::from_parts(Vec::new(), &dims, &strides, isize::MAX).unwrap();
    let view = array.view_mut();
    assert_eq!(view.ptr(), empty_const_ptr());
    assert_eq!(view.as_mut_ptr(), empty_mut_ptr());
}
use num_complex::Complex64;

#[test]
fn test_col_major_strides() {
    assert_eq!(col_major_strides(&[3, 4]).as_slice(), &[1, 3]);
    assert_eq!(col_major_strides(&[2, 3, 4]).as_slice(), &[1, 2, 6]);
}

#[test]
fn test_row_major_strides() {
    assert_eq!(row_major_strides(&[3, 4]).as_slice(), &[4, 1]);
    assert_eq!(row_major_strides(&[2, 3, 4]).as_slice(), &[12, 4, 1]);
}

#[test]
fn test_strided_view_new() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
    assert_eq!(view.ndim(), 2);
    assert_eq!(view.dims(), &[2, 3]);
    assert_eq!(view.strides(), &[3, 1]);
    assert_eq!(view.len(), 6);
}

#[test]
fn test_strided_view_get() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
    assert_eq!(view.get(&[0, 0]), 1.0);
    assert_eq!(view.get(&[0, 1]), 2.0);
    assert_eq!(view.get(&[0, 2]), 3.0);
    assert_eq!(view.get(&[1, 0]), 4.0);
    assert_eq!(view.get(&[1, 2]), 6.0);
}

#[test]
fn test_strided_view_col_major() {
    // Column-major: strides [1, 2] for 2x3
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = StridedView::<f64>::new(&data, &[2, 3], &[1, 2], 0).unwrap();
    assert_eq!(view.get(&[0, 0]), 1.0); // data[0]
    assert_eq!(view.get(&[1, 0]), 2.0); // data[1]
    assert_eq!(view.get(&[0, 1]), 3.0); // data[2]
    assert_eq!(view.get(&[1, 1]), 4.0); // data[3]
    assert_eq!(view.get(&[0, 2]), 5.0); // data[4]
    assert_eq!(view.get(&[1, 2]), 6.0); // data[5]
}

#[test]
fn test_strided_view_permute() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
    let perm = view.permute(&[1, 0]).unwrap();
    assert_eq!(perm.dims(), &[3, 2]);
    assert_eq!(perm.strides(), &[1, 3]);
    assert_eq!(perm.get(&[0, 0]), 1.0);
    assert_eq!(perm.get(&[1, 0]), 2.0);
    assert_eq!(perm.get(&[0, 1]), 4.0);
}

#[test]
fn test_strided_view_transpose_2d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
    let t = view.transpose_2d().unwrap();
    assert_eq!(t.dims(), &[3, 2]);
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[1, 0]), 2.0);
    assert_eq!(t.get(&[0, 1]), 4.0);
}

#[test]
fn test_strided_view_conj() {
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let view = StridedView::<Complex64>::new(&data, &[2], &[1], 0).unwrap();
    let c = view.conj();
    assert_eq!(c.get(&[0]), Complex64::new(1.0, -2.0));
    assert_eq!(c.get(&[1]), Complex64::new(3.0, -4.0));
}

#[test]
fn test_strided_view_adjoint_2d() {
    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    // 2x2 row-major
    let view = StridedView::<Complex64>::new(&data, &[2, 2], &[2, 1], 0).unwrap();
    let adj = view.adjoint_2d().unwrap();
    assert_eq!(adj.dims(), &[2, 2]);
    // Adjoint: conj + transpose
    assert_eq!(adj.get(&[0, 0]), Complex64::new(1.0, -2.0));
    assert_eq!(adj.get(&[1, 0]), Complex64::new(3.0, -4.0));
    assert_eq!(adj.get(&[0, 1]), Complex64::new(5.0, -6.0));
}

#[test]
fn test_strided_view_broadcast() {
    let data = vec![1.0, 2.0, 3.0];
    let view = StridedView::<f64>::new(&data, &[1, 3], &[3, 1], 0).unwrap();
    let broad = view.broadcast(&[4, 3]).unwrap();
    assert_eq!(broad.dims(), &[4, 3]);
    for i in 0..4 {
        assert_eq!(broad.get(&[i, 0]), 1.0);
        assert_eq!(broad.get(&[i, 1]), 2.0);
        assert_eq!(broad.get(&[i, 2]), 3.0);
    }
}

#[test]
fn test_strided_view_mut() {
    let mut data = vec![0.0; 6];
    {
        let mut view = StridedViewMut::<f64>::new(&mut data, &[2, 3], &[3, 1], 0).unwrap();
        view.set(&[0, 0], 1.0);
        view.set(&[1, 2], 6.0);
    }
    assert_eq!(data[0], 1.0);
    assert_eq!(data[5], 6.0);
}

#[test]
fn test_strided_view_mut_as_view() {
    let mut data = vec![1.0, 2.0, 3.0];
    let vm = StridedViewMut::<f64>::new(&mut data, &[3], &[1], 0).unwrap();
    let v = vm.as_view();
    assert_eq!(v.get(&[0]), 1.0);
    assert_eq!(v.get(&[2]), 3.0);
}

#[test]
fn test_strided_tensor_col_major() {
    let t = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.strides(), &[1, 2]); // column-major
    assert_eq!(t.get(&[0, 0]), 0.0);
    assert_eq!(t.get(&[1, 0]), 3.0);
    assert_eq!(t.get(&[0, 1]), 1.0);
    assert_eq!(t.get(&[1, 2]), 5.0);
}

#[test]
fn test_strided_tensor_row_major() {
    let t = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.strides(), &[3, 1]); // row-major
    assert_eq!(t.get(&[0, 0]), 0.0);
    assert_eq!(t.get(&[0, 1]), 1.0);
    assert_eq!(t.get(&[1, 0]), 3.0);
    assert_eq!(t.get(&[1, 2]), 5.0);
}

#[test]
fn test_strided_tensor_view() {
    let t = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
    let v = t.view();
    assert_eq!(v.get(&[0, 0]), 0.0);
    assert_eq!(v.get(&[1, 0]), 10.0);
    assert_eq!(v.get(&[0, 2]), 2.0);
}

#[test]
fn test_strided_tensor_view_mut() {
    let mut t = StridedArray::<f64>::col_major(&[2, 3]);
    {
        let mut vm = t.view_mut();
        vm.set(&[1, 2], 42.0);
    }
    assert_eq!(t.get(&[1, 2]), 42.0);
}

#[test]
fn test_strided_tensor_index() {
    let t = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 10 + idx[1]) as f64);
    assert_eq!(t[&[0usize, 0] as &[usize]], 0.0);
    assert_eq!(t[&[2usize, 3] as &[usize]], 23.0);
}

#[test]
fn test_strided_tensor_index_mut() {
    let mut t = StridedArray::<f64>::row_major(&[2, 3]);
    t[&[1usize, 2] as &[usize]] = 99.0;
    assert_eq!(t.get(&[1, 2]), 99.0);
}

#[test]
fn test_validate_bounds_ok() {
    assert!(validate_bounds(6, &[2, 3], &[3, 1], 0).is_ok());
    assert!(validate_bounds(6, &[2, 3], &[1, 2], 0).is_ok());
}

#[test]
fn test_validate_bounds_out_of_range() {
    assert!(validate_bounds(5, &[2, 3], &[3, 1], 0).is_err());
}

#[test]
fn test_validate_bounds_empty() {
    assert!(validate_bounds(0, &[0, 3], &[3, 1], 0).is_ok());
}

#[test]
fn test_validate_bounds_with_offset() {
    assert!(validate_bounds(7, &[2, 3], &[3, 1], 1).is_ok());
    assert!(validate_bounds(6, &[2, 3], &[3, 1], 1).is_err());
}

#[test]
fn test_strided_tensor_3d() {
    let t = StridedArray::<f64>::from_fn_col_major(&[2, 3, 4], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
    });
    assert_eq!(t.ndim(), 3);
    assert_eq!(t.strides(), &[1, 2, 6]); // column-major
    assert_eq!(t.get(&[0, 0, 0]), 0.0);
    assert_eq!(t.get(&[1, 0, 0]), 100.0);
    assert_eq!(t.get(&[0, 1, 0]), 10.0);
    assert_eq!(t.get(&[0, 0, 1]), 1.0);
    assert_eq!(t.get(&[1, 2, 3]), 123.0);
}

// tensor4all/strided-rs#205: shape products and dense strides are checked.
const HUGE: usize = 1 << 62;

fn assert_overflow<T: std::fmt::Debug>(result: Result<T>) {
    assert!(
        matches!(result, Err(StridedError::OffsetOverflow)),
        "{result:?}"
    );
}

#[test]
fn test_try_constructors_reject_overflowing_shapes() {
    // [1 << 62, 4] wraps to 0 elements unchecked; [1 << 62, 2] overflows
    // only isize; the single extent above isize::MAX overflows the
    // stride conversion.
    for dims in [
        &[HUGE, 4][..],
        &[HUGE, 2],
        &[2, HUGE],
        &[isize::MAX as usize + 2, 1],
        &[usize::MAX, usize::MAX],
    ] {
        assert_overflow(StridedArray::<f64>::try_col_major(dims));
        assert_overflow(StridedArray::<f64>::try_row_major(dims));
        assert_overflow(StridedArray::<f64>::try_from_fn_col_major(dims, |_| {
            unreachable!("f is not called on error")
        }));
        assert_overflow(StridedArray::<f64>::try_from_fn_row_major(dims, |_| {
            unreachable!("f is not called on error")
        }));
        assert_overflow(StridedArray::<f64>::try_col_major_from_buffer(
            vec![1.0; 3],
            dims,
        ));
        // SAFETY: the uninit constructors' only contract (view.rs:1618) is
        // to read a slot after writing it; these calls fail and read nothing.
        assert_overflow(unsafe { StridedArray::<f64>::try_col_major_uninit(dims) });
        // SAFETY: as above.
        assert_overflow(unsafe {
            StridedArray::<f64>::try_col_major_from_buffer_uninit(Vec::new(), dims)
        });
    }
}

#[test]
fn uninit_col_major_writes_bool_slots_before_reading_them() {
    // `bool` is the element `set_len` storage could not hold: a byte that is
    // neither 0 nor 1 is not a `bool`, so an uninitialized slot in a `Vec<bool>`
    // was already invalid before the caller wrote anything (#413).
    let dims = [2usize, 3];
    // SAFETY: (view.rs:1618) every slot is written below before it is read.
    let mut a = unsafe { StridedArray::<bool>::try_col_major_uninit(&dims).unwrap() };
    for j in 0..dims[1] {
        for i in 0..dims[0] {
            a.set(&[i, j], MaybeUninit::new((i + j) % 2 == 0));
        }
    }
    for j in 0..dims[1] {
        for i in 0..dims[0] {
            // SAFETY: every slot was written above, before this read.
            let got = unsafe { a.get(&[i, j]).assume_init() };
            assert_eq!(got, (i + j) % 2 == 0, "slot ({i}, {j})");
        }
    }
}

#[test]
fn uninit_from_buffer_holds_reference_elements() {
    // `&u32` is the other element `set_len` storage broke, and this drives the
    // buffer constructor: the empty buffer grows to the shape.
    let values = [11u32, 22, 33];
    // SAFETY: (view.rs:1618) every slot is written below before it is read.
    let mut a = unsafe {
        StridedArray::<&u32>::try_col_major_from_buffer_uninit(Vec::new(), &[3]).unwrap()
    };
    for (i, v) in values.iter().enumerate() {
        a.set(&[i], MaybeUninit::new(v));
    }
    for (i, v) in values.iter().enumerate() {
        // SAFETY: every slot was written above, before this read.
        let got = unsafe { a.get(&[i]).assume_init() };
        assert!(std::ptr::eq(got, v), "slot {i} does not point at {v}");
    }
}

#[test]
fn uninit_from_buffer_resizes_without_touching_retained_slots() {
    // Spare capacity: the slot the caller already wrote survives the call.
    let mut buf: Vec<MaybeUninit<u8>> = Vec::with_capacity(6);
    buf.push(MaybeUninit::new(7));
    // SAFETY: (view.rs:1618) only slot 0, written above, is read.
    let a = unsafe { StridedArray::try_col_major_from_buffer_uninit(buf, &[2, 3]).unwrap() };
    assert_eq!((a.len(), a.data().len()), (6, 6));
    // SAFETY: slot 0 was written before the call; the others are never read.
    assert_eq!(unsafe { a.data()[0].assume_init() }, 7);

    // A longer buffer is truncated to the shape.
    let buf = vec![MaybeUninit::new(1u8); 10];
    // SAFETY: (view.rs:1618) no slot is read.
    let a = unsafe { StridedArray::try_col_major_from_buffer_uninit(buf, &[2, 3]).unwrap() };
    assert_eq!(a.data().len(), 6);

    // A zero extent makes the array empty whatever the other extents.
    let buf = vec![MaybeUninit::new(1u8); 3];
    // SAFETY: (view.rs:1618) no slot is read.
    let a = unsafe { StridedArray::try_col_major_from_buffer_uninit(buf, &[0, 4]).unwrap() };
    assert!(a.is_empty() && a.data().is_empty());

    // A zero-sized element still tracks its element count.
    // SAFETY: (view.rs:1618) no slot is read.
    let a =
        unsafe { StridedArray::<()>::try_col_major_from_buffer_uninit(Vec::new(), &[3]).unwrap() };
    assert_eq!(a.data().len(), 3);
}

#[test]
fn test_try_constructors_accept_valid_and_empty_shapes() {
    let a = StridedArray::<f64>::try_col_major(&[2, 3]).unwrap();
    assert_eq!((a.strides(), a.len()), (&[1isize, 2][..], 6));
    let a = StridedArray::<f64>::try_row_major(&[2, 3]).unwrap();
    assert_eq!(a.strides(), &[3, 1]);
    let a = StridedArray::<usize>::try_from_fn_row_major(&[2, 3], |i| 10 * i[0] + i[1]).unwrap();
    assert_eq!(a.get(&[1, 2]), 12);
    // A zero extent makes the array empty whatever the other extents.
    for dims in [&[HUGE, 4, 0][..], &[0, usize::MAX, usize::MAX]] {
        let a = StridedArray::<f64>::try_col_major(dims).unwrap();
        assert_eq!((a.len(), a.data().len()), (0, 0));
        assert!(a.is_empty());
        let a = StridedArray::<f64>::try_row_major(dims).unwrap();
        assert_eq!(a.len(), 0);
    }
}

#[test]
#[should_panic(expected = "col_major: shape overflows isize")]
fn test_col_major_panics_instead_of_wrapping() {
    // Unchecked, release builds wrapped the product to 0 and returned an
    // empty buffer behind a [1 << 62, 4] view.
    let _ = StridedArray::<f64>::col_major(&[HUGE, 4]);
}

#[test]
#[should_panic(expected = "column-major strides overflow isize")]
fn test_col_major_strides_panic_on_overflow() {
    let _ = col_major_strides(&[HUGE, 4]);
}

#[test]
fn test_dense_strides_of_an_empty_shape_do_not_panic() {
    assert_eq!(col_major_strides(&[3, 0, 5]).as_slice(), &[1, 3, 0]);
    assert_eq!(row_major_strides(&[3, 0, 5]).as_slice(), &[0, 5, 1]);
    // The overflowing step of an empty shape saturates.
    assert_eq!(
        col_major_strides(&[HUGE, 4, 0]).as_slice(),
        &[1, HUGE as isize, isize::MAX]
    );
    assert_eq!(
        row_major_strides(&[0, 4, HUGE]).as_slice(),
        &[isize::MAX, HUGE as isize, 1]
    );
}

#[test]
fn test_len_of_a_zero_sized_broadcast_view_is_zero() {
    // A zero extent wins before the product of the others.
    let data = [0.0f64; 1];
    let view = StridedView::<f64>::new(&data, &[usize::MAX, 2, 0], &[0, 0, 0], 0).unwrap();
    assert_eq!(view.len(), 0);
}

#[test]
#[should_panic(expected = "element count overflows usize")]
fn test_len_of_an_overflowing_broadcast_view_panics() {
    let data = [0.0f64; 1];
    let view = StridedView::<f64>::new(&data, &[HUGE, 4], &[0, 0], 0).unwrap();
    let _ = view.len();
}

#[test]
fn test_diagonal_view_2d() {
    // A[i,i] shape=[3,3] row-major strides=[3,1]
    // diagonal: shape=[3] strides=[4] (3+1)
    let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[3, 3], &[3, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 1)]).unwrap();
    assert_eq!(diag.dims(), &[3]);
    assert_eq!(diag.strides(), &[4]);
    assert_eq!(diag.get(&[0]), 0.0); // A[0,0]
    assert_eq!(diag.get(&[1]), 4.0); // A[1,1]
    assert_eq!(diag.get(&[2]), 8.0); // A[2,2]
}

#[test]
fn test_diagonal_view_3d_adjacent() {
    // A[i,i,j] shape=[2,2,3] row-major strides=[6,3,1]
    // diagonal over (0,1): shape=[2,3] strides=[9,1] (6+3)
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 2, 3], &[6, 3, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 1)]).unwrap();
    assert_eq!(diag.dims(), &[2, 3]);
    assert_eq!(diag.strides(), &[9, 1]);
    assert_eq!(diag.get(&[0, 0]), 0.0);
    assert_eq!(diag.get(&[0, 2]), 2.0);
    assert_eq!(diag.get(&[1, 0]), 9.0);
}

#[test]
fn test_diagonal_view_3d_non_adjacent() {
    // A[i,j,i] shape=[2,3,2] row-major strides=[6,2,1]
    // diagonal over (0,2): shape=[2,3] strides=[7,2] (6+1)
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 3, 2], &[6, 2, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 2)]).unwrap();
    assert_eq!(diag.dims(), &[2, 3]);
    assert_eq!(diag.strides(), &[7, 2]);
    assert_eq!(diag.get(&[0, 0]), 0.0);
    assert_eq!(diag.get(&[0, 1]), 2.0);
    assert_eq!(diag.get(&[1, 0]), 7.0);
    assert_eq!(diag.get(&[1, 2]), 11.0);
}

#[test]
fn test_diagonal_view_two_pairs() {
    // A[i,j,i,j] shape=[2,3,2,3] -> A_diag[i,j] shape=[2,3]
    let data: Vec<f64> = (0..36).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 3, 2, 3], &[18, 6, 3, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 2), (1, 3)]).unwrap();
    assert_eq!(diag.dims(), &[2, 3]);
    assert_eq!(diag.strides(), &[21, 7]);
    assert_eq!(diag.get(&[0, 0]), 0.0);
    assert_eq!(diag.get(&[1, 1]), 28.0);
}

#[test]
fn test_diagonal_view_pairs_sharing_a_high_axis_fuse_all_three() {
    // Axis 2 is tied to both axis 0 and axis 1, so all three axes are one
    // diagonal: A[i,i,i] with stride 4 + 2 + 1. Folding axis 2 into both
    // survivors separately would reach offset 8 of a length-8 backing.
    let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 2, 2], &[4, 2, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 2), (1, 2)]).unwrap();
    assert_eq!(diag.dims(), &[2]);
    assert_eq!(diag.strides(), &[7]);
    assert_eq!(diag.get(&[1]), 7.0);
}

#[test]
fn test_diagonal_view_repeated_pair_is_idempotent() {
    // Tying the same two axes twice is the same constraint as once; the
    // stride contribution must not be counted twice.
    let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 2, 2], &[4, 2, 1], 0).unwrap();
    let once = view.diagonal_view(&[(0, 1)]).unwrap();
    let twice = view.diagonal_view(&[(0, 1), (1, 0)]).unwrap();
    assert_eq!(twice.dims(), once.dims());
    assert_eq!(twice.strides(), once.strides());
    assert_eq!(twice.strides(), &[6, 1]);
}

#[test]
fn test_diagonal_view_chained_pairs_are_order_independent() {
    // tensor4all/strided-rs#329: [(0, 1), (1, 2)] and [(1, 2), (0, 1)]
    // both describe A[i,i,i]. The second pair's low axis is the first
    // pair's removed axis, and its contribution must still reach axis 0.
    let data: Vec<f64> = (0..27).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[3, 3, 3], &[1, 3, 9], 0).unwrap();
    for pairs in [
        [(0, 1), (1, 2)],
        [(1, 2), (0, 1)],
        [(2, 1), (1, 0)],
        [(0, 2), (2, 1)],
    ] {
        let diag = view.diagonal_view(&pairs).unwrap();
        assert_eq!(diag.dims(), &[3], "{pairs:?}");
        assert_eq!(diag.strides(), &[13], "{pairs:?}");
        assert_eq!(diag.get(&[2]), 26.0, "{pairs:?}");
    }
}

#[test]
fn test_diagonal_view_chained_pairs_keep_free_axes_in_order() {
    // A[i,j,i,i] with a chain through axis 2: the free axis j stays second
    // and the fused dim is the minimum over the whole class.
    let dims = [3usize, 4, 2, 3];
    let strides = [1isize, 3, 12, 24];
    let data: Vec<f64> = (0..72).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &dims, &strides, 0).unwrap();
    let a = view.diagonal_view(&[(2, 3), (0, 2)]).unwrap();
    let b = view.diagonal_view(&[(0, 2), (2, 3)]).unwrap();
    for diag in [a, b] {
        assert_eq!(diag.dims(), &[2, 4]);
        assert_eq!(diag.strides(), &[37, 3]);
        assert_eq!(diag.get(&[1, 2]), 43.0);
    }
}

#[test]
fn test_diagonal_view_fuses_a_shared_low_axis_in_bounds() {
    // A[i,i,i] is [(0, 1), (0, 2)]: axis 0 carries both fused strides and
    // the result stays inside the backing.
    let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
    let view = StridedView::<f64>::new(&data, &[2, 2, 2], &[4, 2, 1], 0).unwrap();
    let diag = view.diagonal_view(&[(0, 1), (0, 2)]).unwrap();
    assert_eq!(diag.dims(), &[2]);
    assert_eq!(diag.strides(), &[7]);
    assert_eq!(diag.get(&[0]), 0.0);
    assert_eq!(diag.get(&[1]), 7.0);
}

#[test]
fn test_custom_type_without_element_op_apply() {
    // Custom Copy type that does NOT implement ElementOpApply.
    // Should work with Identity views (StridedView<T> default).
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct MyScalar(f64);

    impl Default for MyScalar {
        fn default() -> Self {
            MyScalar(0.0)
        }
    }

    // Create array with custom type
    let data = vec![MyScalar(1.0), MyScalar(2.0), MyScalar(3.0), MyScalar(4.0)];
    let arr = StridedArray::from_parts(data, &[2, 2], &[2, 1], 0).unwrap();

    // Access via Identity view (default)
    assert_eq!(arr.get(&[0, 0]), MyScalar(1.0));
    assert_eq!(arr.get(&[1, 0]), MyScalar(3.0));

    // Create view and access elements
    let view: StridedView<MyScalar> = arr.view();
    assert_eq!(view.get(&[0, 1]), MyScalar(2.0));

    // Permute view (metadata-only, no ElementOpApply needed)
    let perm = view.permute(&[1, 0]).unwrap();
    assert_eq!(perm.get(&[0, 0]), MyScalar(1.0));
    assert_eq!(perm.get(&[0, 1]), MyScalar(3.0)); // transposed
}

#[test]
#[should_panic(expected = "wrong number of indices")]
fn test_index_rejects_a_short_index() {
    let arr = StridedArray::from_parts(vec![0.0f64; 4], &[2, 2], &[2, 1], 0).unwrap();
    // Without the rank check this reads an element the index does not name.
    let _ = arr[&[1][..]];
}

#[test]
#[should_panic(expected = "wrong number of indices")]
fn test_view_get_rejects_a_short_index() {
    let data = vec![0.0f64; 4];
    let view = StridedView::<f64>::new(&data, &[2, 2], &[2, 1], 0).unwrap();
    let _ = view.get(&[1]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_view_set_rejects_an_out_of_bounds_index() {
    let mut data = vec![0.0f64; 4];
    let mut view = StridedViewMut::<f64>::new(&mut data, &[2, 2], &[2, 1], 0).unwrap();
    view.set(&[2, 0], 1.0);
}

// `get_unchecked` requires one index per axis. A shorter `indices` used to be
// accepted silently: the loop skipped the missing axes, which read index 0 of
// them instead of rejecting the call (#437).

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "wrong number of indices")]
fn test_view_get_unchecked_rejects_a_short_index() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let view = StridedView::<f64>::new(&data, &[2, 2], &[2, 1], 0).unwrap();
    // SAFETY: the call violates the arity contract on purpose (#437). In debug
    // builds `get_unchecked` panics on its `debug_assert_eq!` before it forms
    // an offset, so the violating dereference never runs; the test is gated on
    // `debug_assertions` and cannot exist where that check is compiled out.
    let _ = unsafe { view.get_unchecked(&[1]) };
}

// The counterexample from #437: with a zero-extent axis the view's base pointer
// is dangling, so dropping that axis used to dereference it.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "wrong number of indices")]
fn test_view_get_unchecked_rejects_a_short_index_on_a_zero_extent_axis() {
    let data: [u8; 0] = [];
    let view = StridedView::<u8>::new(&data, &[0], &[1], 0).unwrap();
    // SAFETY: as above; the omitted axis has extent 0, so without the arity
    // check the call would dereference the view's dangling base pointer.
    let _ = unsafe { view.get_unchecked(&[]) };
}
