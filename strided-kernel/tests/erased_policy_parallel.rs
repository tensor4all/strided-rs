#![cfg(feature = "parallel")]

use strided_kernel::{
    erased_zip_into, ErasedCopyPlan, ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan,
    ErasedGatherPlan, ErasedPadPlan, ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef,
    ErasedReducePlan, ErasedScatterPlan, ErasedZipOp, ExecContext, GatherSpec, KernelDType,
    ReduceOp, ScatterSpec, StridedError,
};

const LARGE_LEN: usize = (1 << 15) + 65;

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            data.as_ptr().cast::<u8>(),
            data.len() * core::mem::size_of::<T>(),
        )
    }
}

fn bounded_context() -> ExecContext {
    ExecContext::max_threads(2).unwrap()
}

#[test]
fn large_one_shot_zip_matches_serial() {
    let dims = [LARGE_LEN];
    let strides = [1isize];
    let lhs: Vec<f64> = (0..LARGE_LEN).map(|index| index as f64).collect();
    let rhs: Vec<f64> = (0..LARGE_LEN)
        .map(|index| (LARGE_LEN - index) as f64)
        .collect();

    let run = |ctx: ExecContext| {
        let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
        let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
        let mut output = vec![0.0f64; LARGE_LEN];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &dims, &strides, 0).unwrap();
        erased_zip_into(
            KernelDType::F64,
            ErasedZipOp::Add,
            &ctx,
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&lhs),
            &ErasedRawStridedPtr::from_ref(&rhs),
        )
        .unwrap();
        output
    };

    assert_eq!(run(bounded_context()), run(ExecContext::serial()));
}

#[test]
fn bounded_one_shot_rejects_noninjective_destination_before_raw_replay() {
    let dims = [LARGE_LEN];
    let source_strides = [1isize];
    let dest_strides = [0isize];
    let lhs = vec![1.0f64; LARGE_LEN];
    let rhs = vec![2.0f64; LARGE_LEN];
    let lhs = ErasedRawStridedRef::from_slice(&lhs, &dims, &source_strides, 0).unwrap();
    let rhs = ErasedRawStridedRef::from_slice(&rhs, &dims, &source_strides, 0).unwrap();
    let mut actual = [7.0f64];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut actual, &dims, &dest_strides, 0).unwrap();

    let error = erased_zip_into(
        KernelDType::F64,
        ErasedZipOp::Add,
        &bounded_context(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&lhs),
        &ErasedRawStridedPtr::from_ref(&rhs),
    )
    .unwrap_err();

    assert!(matches!(error, StridedError::NonInjectiveOutputLayout));
    assert_eq!(actual, [7.0]);
}

#[test]
fn large_erased_copy_matches_serial() {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    const LEN: usize = ROWS * COLS;
    let dims = [ROWS, COLS];
    let src_strides = [COLS as isize, 1isize];
    let dest_strides = [1isize, ROWS as isize];
    let source: Vec<i64> = (0..LEN).map(|index| index as i64 - 17).collect();
    let plan =
        ErasedCopyPlan::compile(KernelDType::I64, &dims, &dest_strides, &src_strides).unwrap();

    let run = |ctx: ExecContext| {
        let source_ref = ErasedRawStridedRef::from_slice(&source, &dims, &src_strides, 0).unwrap();
        let mut output = vec![0i64; LEN];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &dims, &dest_strides, 0).unwrap();
        plan.execute(&ctx, &mut dest, &source_ref).unwrap();
        output
    };

    assert_eq!(run(bounded_context()), run(ExecContext::serial()));
}

#[test]
fn large_erased_axis_reduce_matches_serial() {
    let src_dims = [LARGE_LEN, 2usize];
    let src_strides = [1isize, LARGE_LEN as isize];
    let dest_dims = [LARGE_LEN];
    let dest_strides = [1isize];
    let source: Vec<i32> = (0..LARGE_LEN * 2)
        .map(|index| (index % 251) as i32 - 113)
        .collect();
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::I32,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[1],
    )
    .unwrap();

    let run = |ctx: ExecContext| {
        let source_ref =
            ErasedRawStridedRef::from_slice(&source, &src_dims, &src_strides, 0).unwrap();
        let mut output = vec![0i32; LARGE_LEN];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();
        plan.execute(&ctx, &mut dest, &source_ref).unwrap();
        output
    };

    assert_eq!(run(bounded_context()), run(ExecContext::serial()));
}

#[test]
fn large_erased_gather_matches_serial() {
    let dims = [LARGE_LEN];
    let strides = [1isize];
    let operand: Vec<f64> = (0..LARGE_LEN).map(|index| index as f64 * 0.5).collect();
    let indices: Vec<i64> = (0..LARGE_LEN)
        .map(|index| (LARGE_LEN - 1 - index) as i64)
        .collect();
    let spec = GatherSpec {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let plan = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &dims,
        &strides,
        &dims,
        &strides,
        &dims,
        &strides,
        spec,
    )
    .unwrap();

    let run = |ctx: ExecContext| {
        let operand_ref = ErasedRawStridedRef::from_slice(&operand, &dims, &strides, 0).unwrap();
        let index_ref = ErasedRawStridedRef::from_slice(&indices, &dims, &strides, 0).unwrap();
        let mut output = vec![0.0f64; LARGE_LEN];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &dims, &strides, 0).unwrap();
        plan.execute(&ctx, &mut dest, &operand_ref, &index_ref)
            .unwrap();
        output
    };

    assert_eq!(run(bounded_context()), run(ExecContext::serial()));
}

#[test]
fn large_erased_dynamic_slice_and_update_match_serial() {
    let operand_dims = [LARGE_LEN + 128];
    let operand_strides = [1isize];
    let starts_dims = [1usize];
    let starts_strides = [1isize];
    let window_dims = [LARGE_LEN];
    let window_strides = [1isize];
    let operand: Vec<i32> = (0..LARGE_LEN + 128)
        .map(|index| (index % 997) as i32 - 411)
        .collect();
    let update: Vec<i32> = (0..LARGE_LEN)
        .map(|index| 1000 + (index % 31) as i32)
        .collect();
    let starts = [64i64];

    let slice = ErasedDynamicSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &window_dims,
        &window_strides,
        &window_dims,
    )
    .unwrap();
    let update_slice = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &window_dims,
        &window_strides,
        &operand_dims,
        &operand_strides,
    )
    .unwrap();

    let run_slice = |ctx: ExecContext| {
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let starts_ref =
            ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
        let mut output = vec![0i32; LARGE_LEN];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &window_dims, &window_strides, 0)
                .unwrap();
        slice
            .execute(&ctx, &mut dest, &operand_ref, &starts_ref)
            .unwrap();
        output
    };
    let run_update = |ctx: ExecContext| {
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let update_ref =
            ErasedRawStridedRef::from_slice(&update, &window_dims, &window_strides, 0).unwrap();
        let starts_ref =
            ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
        let mut output = vec![0i32; LARGE_LEN + 128];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &operand_dims, &operand_strides, 0)
                .unwrap();
        update_slice
            .execute(&ctx, &mut dest, &operand_ref, &update_ref, &starts_ref)
            .unwrap();
        output
    };

    assert_eq!(
        run_slice(bounded_context()),
        run_slice(ExecContext::serial())
    );
    assert_eq!(
        run_update(bounded_context()),
        run_update(ExecContext::serial())
    );
}

#[test]
fn large_erased_pad_matches_serial() {
    let operand_dims = [LARGE_LEN];
    let operand_strides = [1isize];
    let dest_dims = [LARGE_LEN + 128];
    let dest_strides = [1isize];
    let edge_low = [64i64];
    let edge_high = [64i64];
    let interior = [0i64];
    let fill = [-7i32];
    let operand: Vec<i32> = (0..LARGE_LEN).map(|index| index as i32).collect();
    let plan = ErasedPadPlan::compile(
        KernelDType::I32,
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &edge_low,
        &edge_high,
        &interior,
    )
    .unwrap();

    let run = |ctx: ExecContext| {
        let operand_ref =
            ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
        let mut output = vec![0i32; LARGE_LEN + 128];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();
        plan.execute(&ctx, &mut dest, &operand_ref, as_bytes(&fill))
            .unwrap();
        output
    };

    assert_eq!(run(bounded_context()), run(ExecContext::serial()));
}

#[test]
fn large_erased_scatter_matches_serial_with_overlaps() {
    let dims = [LARGE_LEN];
    let strides = [1isize];
    let index_dims = [LARGE_LEN, 1usize];
    let index_strides = [1isize, LARGE_LEN as isize];
    let operand: Vec<f64> = (0..LARGE_LEN).map(|index| index as f64).collect();
    let updates: Vec<f64> = (0..LARGE_LEN)
        .map(|index| (index % 17) as f64 - 3.0)
        .collect();
    let indices: Vec<i64> = (0..LARGE_LEN).map(|index| (index % 1024) as i64).collect();
    let spec = ScatterSpec {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let plan = ErasedScatterPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &dims,
        &strides,
        &index_dims,
        &index_strides,
        &dims,
        &strides,
        &dims,
        &strides,
        spec,
    )
    .unwrap();

    let run = |ctx: ExecContext| {
        let operand_ref = ErasedRawStridedRef::from_slice(&operand, &dims, &strides, 0).unwrap();
        let index_ref =
            ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
        let update_ref = ErasedRawStridedRef::from_slice(&updates, &dims, &strides, 0).unwrap();
        let mut output = vec![0.0f64; LARGE_LEN];
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut output, &dims, &strides, 0).unwrap();
        plan.execute(&ctx, &mut dest, &operand_ref, &index_ref, &update_ref)
            .unwrap();
        output
    };

    assert_eq!(run(bounded_context()), run(ExecContext::serial()));
}
