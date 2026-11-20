//! The initialized additive-scatter entry must use the same integer combine as
//! its uninitialized sibling (`wrapping_add`, not `+`), so a value at the dtype
//! extrema cannot panic in debug/test builds and wrap in release.
//!
//! #508: the typed `ScatterPlan::execute` (reachable as
//! `strided_kernel::ScatterPlan`) is bound to the sealed `ScatterScalar` and
//! uses its `wrapping_add`, so it agrees with the erased initialized entry and
//! with the uninitialized entries on the same input.

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedScatterPlan, Exec, KernelDType, Pool,
    RawStridedMut, RawStridedRef, ScatterPlan, ScatterScalar, ScatterSpec,
};

fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Pool::builder().threads(2).build().unwrap())
}

fn bounded(threads: usize) -> Exec<'static> {
    Exec::new(pool(), threads).unwrap()
}

#[test]
fn initialized_scatter_wraps_integer_extrema_like_uninit() {
    macro_rules! case {
        ($ty:ty, $dtype:expr, $ity:ty, $idtype:expr) => {{
            let dims = [3usize];
            let ids = [3usize, 1];
            let operand = [<$ty>::MAX, 1, 2];
            let indices = [0 as $ity, 0 as $ity, 1 as $ity];
            let updates = [1 as $ty, 2 as $ty, <$ty>::MAX];
            let spec = ScatterSpec {
                update_window_dims: vec![],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            };
            let plan = ErasedScatterPlan::compile(
                $dtype,
                $idtype,
                &dims,
                &[1],
                &ids,
                &[1, 3],
                &dims,
                &[1],
                &dims,
                &[1],
                spec,
            )
            .unwrap();
            let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
            let index = ErasedRawStridedRef::from_slice(&indices, &ids, &[1, 3], 0).unwrap();
            let update = ErasedRawStridedRef::from_slice(&updates, &dims, &[1], 0).unwrap();
            let expected = [
                operand[0].wrapping_add(updates[0]).wrapping_add(updates[1]),
                operand[1].wrapping_add(updates[2]),
                operand[2],
            ];
            for ctx in [Exec::serial(), bounded(2)] {
                let mut dest = [0 as $ty; 3];
                let mut out =
                    ErasedRawStridedMut::from_slice_mut(&mut dest, &dims, &[1], 0).unwrap();
                plan.execute(&mut out, &source, &index, &update, ctx)
                    .unwrap();
                assert_eq!(dest, expected);
            }
        }};
    }
    case!(i32, KernelDType::I32, i32, KernelDType::I32);
    case!(i64, KernelDType::I64, i64, KernelDType::I64);
}

fn scalar_spec() -> ScatterSpec {
    ScatterSpec {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}

/// Runs the typed initialized entry, the typed uninitialized entry and the
/// erased initialized entry on one input and checks that all three agree with
/// `expected` under every `Exec`.
fn assert_entries_agree<T, I>(
    dtype: KernelDType,
    index_dtype: KernelDType,
    operand: [T; 3],
    indices: [I; 3],
    updates: [T; 3],
    expected: [T; 3],
) where
    T: ScatterScalar
        + PartialEq
        + core::fmt::Debug
        + Default
        + strided_kernel::KernelStorageElement,
    I: strided_kernel::GatherIndex + strided_kernel::KernelStorageElement,
{
    let dims = [3usize];
    let ids = [3usize, 1];
    let plan = ScatterPlan::compile(
        &dims,
        &[1],
        &ids,
        &[1, 3],
        &dims,
        &[1],
        &dims,
        &[1],
        scalar_spec(),
    )
    .unwrap();
    let erased = ErasedScatterPlan::compile(
        dtype,
        index_dtype,
        &dims,
        &[1],
        &ids,
        &[1, 3],
        &dims,
        &[1],
        &dims,
        &[1],
        scalar_spec(),
    )
    .unwrap();
    let source = RawStridedRef::new(&operand, &dims, &[1], 0).unwrap();
    let index = RawStridedRef::new(&indices, &ids, &[1, 3], 0).unwrap();
    let update = RawStridedRef::new(&updates, &dims, &[1], 0).unwrap();
    let erased_source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
    let erased_index = ErasedRawStridedRef::from_slice(&indices, &ids, &[1, 3], 0).unwrap();
    let erased_update = ErasedRawStridedRef::from_slice(&updates, &dims, &[1], 0).unwrap();
    for threads in [None, Some(2)] {
        let ctx = || threads.map_or_else(Exec::serial, bounded);

        let mut typed = [T::default(); 3];
        let mut dest = RawStridedMut::new(&mut typed, &dims, &[1], 0).unwrap();
        plan.execute(&mut dest, &source, &index, &update, ctx())
            .unwrap();
        assert_eq!(typed, expected, "typed execute, threads {threads:?}");

        let mut storage = [MaybeUninit::<T>::uninit(); 3];
        let mut dest = RawStridedMut::new(&mut storage, &dims, &[1], 0).unwrap();
        // SAFETY: every input is an initialized local array disjoint from `storage`, and
        // `ScatterPlan::compile` above fixed the layouts these views carry.
        unsafe {
            strided_kernel::execution::scatter_into_uninit(
                &plan,
                &mut dest,
                &source,
                &index,
                &update,
                <T as ScatterScalar>::wrapping_add,
                ctx(),
            )
        }
        .unwrap();
        // SAFETY: the operand copy initialized every slot of the full-coverage destination.
        let uninit = storage.map(|slot| unsafe { slot.assume_init() });
        assert_eq!(
            uninit, expected,
            "typed execute_uninit, threads {threads:?}"
        );

        let mut erased_out = [T::default(); 3];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut erased_out, &dims, &[1], 0).unwrap();
        erased
            .execute(
                &mut dest,
                &erased_source,
                &erased_index,
                &erased_update,
                ctx(),
            )
            .unwrap();
        assert_eq!(erased_out, expected, "erased execute, threads {threads:?}");
    }
}

#[test]
fn typed_scatter_execute_wraps_integer_extrema_like_erased_and_uninit() {
    macro_rules! case {
        ($ty:ty, $dtype:expr, $ity:ty, $idtype:expr) => {{
            let operand = [<$ty>::MAX, <$ty>::MIN, 2];
            let indices = [0 as $ity, 0 as $ity, 1 as $ity];
            let updates = [1 as $ty, <$ty>::MAX, -1 as $ty];
            let expected = [
                operand[0].wrapping_add(updates[0]).wrapping_add(updates[1]),
                operand[1].wrapping_add(updates[2]),
                operand[2],
            ];
            assert_eq!(expected, [-1, <$ty>::MAX, 2]);
            assert_entries_agree::<$ty, $ity>($dtype, $idtype, operand, indices, updates, expected);
        }};
    }
    case!(i32, KernelDType::I32, i32, KernelDType::I32);
    case!(i32, KernelDType::I32, i64, KernelDType::I64);
    case!(i64, KernelDType::I64, i32, KernelDType::I32);
    case!(i64, KernelDType::I64, i64, KernelDType::I64);
}

#[test]
fn typed_scatter_execute_adds_floats_and_complex_like_erased_and_uninit() {
    assert_entries_agree::<f32, i64>(
        KernelDType::F32,
        KernelDType::I64,
        [1.0, 2.0, 3.0],
        [2, 0, 2],
        [0.5, 4.0, 0.25],
        [5.0, 2.0, 3.75],
    );
    assert_entries_agree::<f64, i32>(
        KernelDType::F64,
        KernelDType::I32,
        [f64::MAX, 2.0, 3.0],
        [0, 0, 1],
        [f64::MAX, -1.0, 0.5],
        [f64::INFINITY, 2.5, 3.0],
    );
    let c32 = Complex32::new;
    assert_entries_agree::<Complex32, i64>(
        KernelDType::C32,
        KernelDType::I64,
        [c32(1.0, 1.0), c32(0.0, 0.0), c32(2.0, -2.0)],
        [1, 1, 2],
        [c32(1.0, 0.0), c32(0.0, 1.0), c32(-2.0, 2.0)],
        [c32(1.0, 1.0), c32(1.0, 1.0), c32(0.0, 0.0)],
    );
    let c64 = Complex64::new;
    assert_entries_agree::<Complex64, i32>(
        KernelDType::C64,
        KernelDType::I32,
        [c64(1.0, 1.0), c64(0.0, 0.0), c64(2.0, -2.0)],
        [0, 2, 0],
        [c64(0.5, 0.5), c64(1.0, 1.0), c64(0.5, -0.5)],
        [c64(2.0, 1.0), c64(0.0, 0.0), c64(3.0, -1.0)],
    );
}
