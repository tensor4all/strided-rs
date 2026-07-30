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
    assert!(erased.contains("typed_uninit_slice_mut"));
    let reduce_uninit = erased
        .split_once("pub fn execute_uninit")
        .and_then(|(_, rest)| rest.split_once("impl ErasedGatherPlan"))
        .map(|(body, _)| body)
        .expect("reduce uninitialized entry point remains");
    assert!(!reduce_uninit.contains("typed_slice_mut"));
}
