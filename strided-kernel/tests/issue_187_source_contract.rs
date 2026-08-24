#[test]
fn erased_axis_reduction_uses_prepared_incremental_cursors() {
    let source = include_str!("../src/erased.rs");
    assert!(source.contains("ReduceOuterAxis"));
    assert!(source.contains("ReduceInnerAxis"));
    assert!(source.contains("check_reduce_layout_offset_arithmetic"));
    assert!(source.contains("checked_reduce_reset"));
    assert!(source.contains("compress_reduce_outer_axes"));
    assert!(source.contains("compress_reduce_inner_axes"));
    let axes = source
        .split_once("fn execute_reduce_axes_serial_data")
        .and_then(|(_, rest)| rest.split_once("#[cfg(feature = \"parallel\")]"))
        .map(|(body, _)| body)
        .expect("axis serial replay remains ordered");
    assert!(!axes.contains("checked_strided_offset"));
}

#[test]
fn reduction_uninit_has_no_initialized_backing_conversion() {
    let source = include_str!("../src/erased.rs");
    let reduce = source
        .split_once("impl ErasedReducePlan")
        .and_then(|(_, rest)| rest.split_once("impl ErasedGatherPlan"))
        .expect("reduction and gather impls remain ordered")
        .0;
    assert!(!reduce.contains("from_raw_parts_mut"));
    assert!(!reduce.contains("ErasedRawStridedMut::new"));
    assert!(reduce.contains("reduce_uninit_writer"));
    for forbidden in [
        "ErasedRawStridedMut::from_slice_mut",
        "RawStridedMut::new",
        "typed_slice_mut",
        "data_as_mut",
        "data_as::<",
        "from_slice_mut",
    ] {
        assert!(
            !reduce.contains(forbidden),
            "initialized conversion remains: {forbidden}"
        );
    }
}

#[test]
fn indexed_uninit_receipt_is_private_and_writer_is_not_additive() {
    let source = include_str!("../src/gather_plan.rs");
    assert!(!source.contains("pub fn execute_uninit"));
    assert!(source.contains("pub(crate) fn execute_uninit"));
    let copy = include_str!("../src/copy_plan.rs");
    let maybe_uninit = copy
        .split_once("impl<'a, T> OverwriteWriter<T> for RawStridedMut<'a, MaybeUninit<T>>")
        .and_then(|(_, rest)| rest.split_once("pub(crate) struct InitializedRawDest"))
        .expect("MaybeUninit writer and receipt remain ordered")
        .0;
    assert!(!maybe_uninit.contains("add_at"));
    assert!(copy.contains("for<'b> FnOnce(InitializedRawDest<'b, T>)"));
    assert!(copy.contains("pub(crate) fn execute_uninit_then"));
    assert!(copy.contains("extent: usize"));
    assert!(copy.contains("PhantomData<&'a mut [MaybeUninit<T>]>"));
    assert!(copy.contains(".write(value)"));
    assert!(copy.contains("unsafe fn data_ptr"));
    assert!(copy.contains("unsafe fn write_at"));
    assert!(copy.contains("unsafe fn add_at"));
    assert!(copy.contains("# Safety"));
    let erased = include_str!("../src/erased.rs");
    assert!(erased.contains("unsafe fn ptr"));
    assert!(erased.contains("unsafe fn write_at"));
    assert!(!source.contains("InitializedRawDest"));
}

#[test]
fn uninitialized_parallel_stores_use_write() {
    let gather = include_str!("../src/gather_plan.rs");
    let erased = include_str!("../src/erased.rs");
    for source in [gather, erased] {
        for line in source.lines().filter(|line| line.contains("*dest_ptr")) {
            assert!(
                !line.contains(" = "),
                "direct destination assignment remains: {line}"
            );
        }
    }
}

#[test]
fn indexed_uninit_dispatches_only_prevalidated_inputs() {
    let source = include_str!("../src/erased.rs");
    for name in [
        "execute_gather_uninit_dispatch",
        "execute_dynamic_slice_uninit_dispatch",
        "execute_dynamic_update_uninit_dispatch",
        "execute_scatter_uninit_dispatch",
    ] {
        let section = source
            .split_once(&format!("fn {name}"))
            .and_then(|(_, rest)| rest.split_once("\nfn "))
            .map(|(body, _)| body)
            .expect("dispatch helper exists");
        assert!(!section.contains("validated_input_ref"));
        assert!(!section.contains("typed_slice_mut"));
    }
}

#[test]
fn receipt_and_typed_uninit_boundaries_remain_private() {
    let copy = include_str!("../src/copy_plan.rs");
    let lib = include_str!("../src/lib.rs");
    assert!(copy.contains("pub(crate) struct InitializedRawDest"));
    assert!(!copy.contains("pub struct InitializedRawDest"));
    assert!(!lib.contains("pub use crate::copy_plan::InitializedRawDest"));
    assert!(!lib.contains("pub use copy_plan::InitializedRawDest"));
    assert!(!copy.contains("pub fn execute_uninit_then"));
    let erased = include_str!("../src/erased.rs");
    assert!(erased.contains("fn reduce_uninit_writer"));
    assert!(erased.contains("data_as_uninit_mut"));
    let reduce_uninit = erased
        .split_once("pub fn execute_uninit")
        .and_then(|(_, rest)| rest.split_once("impl ErasedGatherPlan"))
        .map(|(body, _)| body)
        .expect("reduce uninitialized entry point remains");
    assert!(!reduce_uninit.contains("typed_slice_mut"));
    assert!(reduce_uninit.contains("reduce_uninit_writer"));
    assert!(!reduce_uninit.contains("from_slice_mut"));
    let writer = erased
        .split_once("fn reduce_uninit_writer")
        .and_then(|(_, rest)| rest.split_once("\nfn "))
        .map(|(body, _)| body)
        .expect("reduction uninitialized writer remains");
    assert!(writer.contains("data_as_uninit_mut"));
    for forbidden in [
        "ErasedRawStridedMut<",
        "from_slice_mut",
        "typed_slice_mut",
        "data_as_mut",
        "RawStridedMut<",
        "assume_init",
    ] {
        assert!(
            !writer.contains(forbidden),
            "initialized conversion remains: {forbidden}"
        );
    }
    for helper in [
        "execute_gather_uninit_dispatch",
        "execute_dynamic_slice_uninit_dispatch",
        "execute_dynamic_update_uninit_dispatch",
        "execute_scatter_uninit_dispatch",
    ] {
        let body = erased
            .split_once(&format!("fn {helper}"))
            .and_then(|(_, rest)| rest.split_once("\nfn "))
            .map(|(body, _)| body)
            .expect("uninitialized dispatch helper remains");
        for forbidden in ["data_as_mut", "RawStridedMut<", "assume_init", "*dest"] {
            assert!(
                !body.contains(forbidden),
                "destination read remains in {helper}: {forbidden}"
            );
        }
    }
    for helper in [
        "execute_gather_uninit<T, I>",
        "execute_dynamic_slice_uninit<T, I>",
        "execute_dynamic_update_uninit<T, I>",
        "execute_scatter_uninit<T, I>",
    ] {
        let body = erased
            .split_once(&format!("fn {helper}"))
            .and_then(|(_, rest)| rest.split_once("\nfn "))
            .map(|(body, _)| body)
            .expect("generic uninitialized helper remains");
        assert!(
            body.contains("data_as_uninit_mut"),
            "missing typed uninit accessor in {helper}"
        );
        for forbidden in [
            "data_as_mut",
            "RawStridedMut::<T>",
            "RawStridedMut<T>",
            "StridedViewMut",
            "from_slice_mut",
            "assume_init",
            "dest_data.as_ptr",
            "dest_data[",
            "*dest_ptr",
        ] {
            assert!(
                !body.contains(forbidden),
                "destination read/conversion in {helper}: {forbidden}"
            );
        }
    }
}
