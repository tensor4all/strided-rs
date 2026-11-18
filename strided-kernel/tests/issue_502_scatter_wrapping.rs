//! The initialized additive-scatter entry must use the same integer combine as
//! its uninitialized sibling (`wrapping_add`, not `+`), so a value at the dtype
//! extrema cannot panic in debug/test builds and wrap in release.

use strided_kernel::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedScatterPlan, Exec, KernelDType, Pool,
    ScatterSpec,
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
