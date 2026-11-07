use super::*;

#[test]
fn typed_erased_storage_covers_all_kernel_dtypes() {
    macro_rules! check {
        ($ty:ty, $value:expr) => {{
            let mut initialized = [$value, $value];
            let dims = [2usize];
            let strides = [1isize];
            let reference =
                ErasedRawStridedRef::from_slice(&initialized, &dims, &strides, 0).unwrap();
            assert_eq!(reference.data_as::<$ty>().unwrap().len(), 2);
            let mut mutable =
                ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &strides, 0).unwrap();
            assert_eq!(mutable.data_as::<$ty>().unwrap().len(), 2);
            assert_eq!(mutable.data_as_mut::<$ty>().unwrap().len(), 2);
            let mut uninitialized = [MaybeUninit::<$ty>::uninit(); 2];
            let mut destination = ErasedRawStridedUninitMut::from_uninit_slice(
                &mut uninitialized,
                &dims,
                &strides,
                0,
            )
            .unwrap();
            assert_eq!(destination.data_as_uninit_mut::<$ty>().unwrap().len(), 2);
        }};
    }

    check!(f32, 1.0f32);
    check!(f64, 1.0f64);
    check!(i32, 1i32);
    check!(i64, 1i64);
    check!(bool, true);
    check!(Complex32, Complex32::new(1.0, 0.0));
    check!(Complex64, Complex64::new(1.0, 0.0));
}

#[test]
fn raw_bool_descriptor_reports_the_first_invalid_byte() {
    let mut bytes = vec![1u8; 4096];
    bytes[4000] = 3;
    bytes[4001] = 9;
    let dims = [bytes.len()];
    let strides = [1isize];
    let ptr = NonNull::new(bytes.as_mut_ptr()).unwrap();
    let erased = unsafe {
        ErasedRawStridedPtr::from_raw_parts(KernelDType::Bool, ptr, bytes.len(), &dims, &strides, 0)
    }
    .unwrap();
    assert!(matches!(
        unsafe { erased.try_as_ref_after_no_overlap() },
        Err(crate::StridedError::InvalidBoolByte { value: 3 })
    ));

    // Write through the handed off pointer so no reborrow of `bytes`
    // invalidates it.
    unsafe {
        *ptr.as_ptr().add(4000) = 0;
        *ptr.as_ptr().add(4001) = 1;
    }
    assert!(unsafe { erased.try_as_ref_after_no_overlap() }.is_ok());
}

#[test]
fn rejects_usize_max_dimension_before_isize_truncation() {
    let data = [false; 1];
    let dims = [usize::MAX];
    let strides = [-1isize];
    assert!(matches!(
        RawStridedRef::new(&data, &dims, &strides, 0),
        Err(crate::StridedError::OffsetOverflow)
    ));
    let mut data = [false; 1];
    assert!(matches!(
        RawStridedMut::new(&mut data, &dims, &strides, 0),
        Err(crate::StridedError::OffsetOverflow)
    ));
    assert!(matches!(
        ErasedRawStridedRef::from_slice(&data, &dims, &strides, 0),
        Err(crate::StridedError::OffsetOverflow)
    ));
    let mut data = [false; 1];
    assert!(matches!(
        ErasedRawStridedMut::from_slice_mut(&mut data, &dims, &strides, 0),
        Err(crate::StridedError::OffsetOverflow)
    ));
    let ptr = NonNull::new(data.as_mut_ptr()).unwrap();
    assert!(matches!(
        unsafe {
            ErasedRawStridedPtr::from_raw_parts(
                KernelDType::Bool,
                ptr.cast(),
                data.len(),
                &dims,
                &strides,
                0,
            )
        },
        Err(crate::StridedError::OffsetOverflow)
    ));
    let mut data = vec![MaybeUninit::<bool>::uninit(); 1];
    assert!(matches!(
        ErasedRawStridedUninitMut::from_uninit_slice(&mut data, &dims, &strides, 0),
        Err(crate::StridedError::OffsetOverflow)
    ));
}

#[test]
fn empty_raw_views_use_dangling_base_pointers_for_extreme_offsets() {
    let dims = [0usize];
    let strides = [1isize];
    let data: [f64; 0] = [];
    let raw = RawStridedRef::new(&data, &dims, &strides, isize::MAX).unwrap();
    assert_eq!(raw.ptr(), NonNull::<f64>::dangling().as_ptr());

    let mut data: [f64; 0] = [];
    let mut raw = RawStridedMut::new(&mut data, &dims, &strides, isize::MAX).unwrap();
    assert_eq!(raw.ptr(), NonNull::<f64>::dangling().as_ptr());
    assert_eq!(raw.as_mut_ptr(), NonNull::<f64>::dangling().as_ptr());
}

#[test]
fn raw_ref_rejects_out_of_bounds_layout() {
    let data = [0.0f64; 4];
    let err = RawStridedRef::new(&data, &[2, 3], &[3, 1], 0).unwrap_err();
    assert!(matches!(err, crate::StridedError::OffsetOverflow));
}

#[test]
fn raw_mut_can_reborrow_as_view() {
    let mut data = [1, 2, 3, 4];
    let mut raw = RawStridedMut::new(&mut data, &[2, 2], &[2, 1], 0).unwrap();
    {
        let view = raw.as_view();
        assert_eq!(view.dims(), &[2, 2]);
    }
    let view_mut = raw.as_view_mut();
    assert_eq!(view_mut.get(&[1, 1]), 4);
}

#[test]
fn raw_descriptors_debug_print_layout_without_the_buffer() {
    let dims = [64usize];
    let strides = [1isize];
    let values = [1.5f64; 64];
    let reference = ErasedRawStridedRef::from_slice(&values, &dims, &strides, 0).unwrap();
    let typed = RawStridedRef::new(&values, &dims, &strides, 0).unwrap();

    let mut mutable_values = [1.5f64; 64];
    let mutable =
        ErasedRawStridedMut::from_slice_mut(&mut mutable_values, &dims, &strides, 0).unwrap();
    let mut typed_values = [1.5f64; 64];
    let typed_mut = RawStridedMut::new(&mut typed_values, &dims, &strides, 0).unwrap();

    let mut uninit_values = [MaybeUninit::<f64>::uninit(); 64];
    let uninit_mut =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninit_values, &dims, &strides, 0)
            .unwrap();

    for text in [
        format!("{reference:?}"),
        format!("{mutable:?}"),
        format!("{uninit_mut:?}"),
    ] {
        assert!(text.contains("dims: [64]"), "{text}");
        assert!(text.contains("byte_len: 512"), "{text}");
        assert!(!text.contains("1.5"), "{text}");
    }
    for text in [format!("{typed:?}"), format!("{typed_mut:?}")] {
        assert!(text.contains("dims: [64]"), "{text}");
        assert!(text.contains("data_len: 64"), "{text}");
        assert!(!text.contains("1.5"), "{text}");
    }
}
