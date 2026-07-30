use std::mem::MaybeUninit;
use std::{fs, path::Path, ptr::NonNull};

use strided_kernel::{
    ErasedConcatenatePlan, ErasedRawStridedPtr, ErasedRawStridedUninitMut, ExecContext,
    KernelDType, StridedError,
};

#[test]
fn concatenate_checks_all_overlaps_before_deferred_bool_validation() {
    let input_dims: [&[usize]; 2] = [&[1], &[1]];
    let input_strides: [&[isize]; 2] = [&[1], &[1]];
    let dest_dims = [2usize];
    let dest_strides = [1isize];
    let plan = ErasedConcatenatePlan::compile(
        KernelDType::Bool,
        &input_dims,
        &input_strides,
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    let mut invalid = vec![2u8];
    let invalid_ptr = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(invalid.as_mut_ptr()).unwrap(),
            invalid.len(),
            &[1],
            &[1],
            0,
        )
    }
    .unwrap();

    let mut destination = vec![2u8; 2];
    let overlapping_ptr = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(destination.as_mut_ptr()).unwrap(),
            1,
            &[1],
            &[1],
            0,
        )
    }
    .unwrap();
    let mut output = unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(destination.as_mut_ptr()).unwrap(),
            destination.len(),
            &dest_dims,
            &dest_strides,
            0,
        )
    }
    .unwrap();

    let result = plan.execute_uninit(
        &ExecContext::serial(),
        &mut output,
        &[invalid_ptr, overlapping_ptr],
    );
    // The first input is invalid, but the second input overlaps the output.
    // This exact error proves every overlap is checked before deferred Bool
    // validation of any input.
    assert!(matches!(
        result,
        Err(StridedError::OverlappingInputOutput { input: 1 })
    ));
}

#[test]
fn bool_mutation_after_raw_pointer_handoff_is_validated_before_writes() {
    let dims = [&[1usize][..]];
    let strides = [&[1isize][..]];
    let dest_dims = [1usize];
    let dest_strides = [1isize];
    let plan = ErasedConcatenatePlan::compile(
        KernelDType::Bool,
        &dims,
        &strides,
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    let mut input = [1u8];
    let input_ptr = input.as_mut_ptr();
    let handed_off = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(input_ptr).unwrap(),
            input.len(),
            &dims[0],
            &strides[0],
            0,
        )
    }
    .unwrap();
    // The raw contract permits sequential mutation through the original
    // pointer before execution; no access through `handed_off` occurs here.
    unsafe { *input_ptr = 2 };

    let mut destination = [MaybeUninit::new(7u8)];
    let mut output = unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(destination.as_mut_ptr().cast()).unwrap(),
            destination.len(),
            &dest_dims,
            &dest_strides,
            0,
        )
    }
    .unwrap();
    let result = plan.execute_uninit(&ExecContext::serial(), &mut output, &[handed_off]);
    assert!(matches!(result, Err(StridedError::InvalidBoolByte { .. })));
    assert_eq!(unsafe { destination[0].assume_init() }, 7);
}

#[test]
fn bool_uninitialized_replay_does_not_read_strided_holes() {
    let input_dims = [&[2usize][..]];
    let input_strides = [&[1isize][..]];
    let dest_dims = [2usize];
    let dest_strides = [2isize];
    let plan = ErasedConcatenatePlan::compile(
        KernelDType::Bool,
        &input_dims,
        &input_strides,
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    let input = [1u8, 0u8];
    let input_ptr = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(input.as_ptr() as *mut u8).unwrap(),
            input.len(),
            &input_dims[0],
            &input_strides[0],
            0,
        )
    }
    .unwrap();
    let mut destination = [
        MaybeUninit::<u8>::uninit(),
        MaybeUninit::<u8>::uninit(),
        MaybeUninit::<u8>::uninit(),
    ];
    let mut output = unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(destination.as_mut_ptr().cast()).unwrap(),
            destination.len(),
            &dest_dims,
            &dest_strides,
            0,
        )
    }
    .unwrap();

    plan.execute_uninit(&ExecContext::serial(), &mut output, &[input_ptr])
        .unwrap();
    // Only offsets 0 and 2 are reachable. The hole at offset 1 remains
    // MaybeUninit and is deliberately never read or assumed initialized.
    assert_eq!(unsafe { destination[0].assume_init() }, 1);
    assert_eq!(unsafe { destination[2].assume_init() }, 0);
}

#[test]
fn production_source_contract_has_no_legacy_erased_storage_apis() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let production = [
        root.join("strided-view/src"),
        root.join("strided-kernel/src"),
        root.join("strided-einsum2/src"),
        root.join("strided-opteinsum/src"),
        root.join("mdarray-opteinsum/src"),
        root.join("ndarray-opteinsum/src"),
        root.join("strided-rs/src"),
    ];
    let forbidden = [
        "ErasedRawStridedRef::new",
        "ErasedRawStridedMut::new",
        "ErasedRawStridedUninitMut::new",
        "ErasedRawStridedPtr::new",
        "typed_slice",
        "data_unchecked",
    ];
    for directory in production {
        let mut files = vec![directory];
        while let Some(path) = files.pop() {
            if path.is_dir() {
                files.extend(
                    fs::read_dir(path)
                        .unwrap()
                        .map(|entry| entry.unwrap().path()),
                );
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            for pattern in forbidden {
                assert!(!source.contains(pattern), "{pattern} remains in {path:?}");
            }
            if path.ends_with("strided-view/src/raw.rs") {
                continue;
            }
            if path.ends_with("strided-kernel/src/erased.rs") {
                assert!(!source.contains("std::slice::from_raw_parts"));
                assert!(!source.contains("std::slice::from_raw_parts_mut"));
                assert!(!source.contains("core::slice::from_raw_parts"));
                assert!(!source.contains("core::slice::from_raw_parts_mut"));
            }
            assert!(
                !source.contains("ErasedRawStridedRef::from_raw_parts")
                    && !source.contains("ErasedRawStridedMut::from_raw_parts")
                    && !source.contains("ErasedRawStridedUninitMut::from_raw_parts"),
                "erased raw construction escaped the designated raw boundary: {path:?}"
            );
        }
    }
}
