use super::{DynamicSlicePlan, DynamicUpdateSlicePlan, ScatterPlan, ScatterSpec, WindowReplay};

#[test]
fn window_replay_fuses_only_bilaterally_contiguous_axes() {
    let compact = WindowReplay::compile(&[2, 3, 4], &[1, 2, 6], &[1, 2, 6]).unwrap();
    assert_eq!(&compact.shape[..], &[24]);
    assert_eq!(compact.axes.len(), 1);

    let negative_source = WindowReplay::compile(&[2, 3], &[1, -2], &[1, 2]).unwrap();
    assert_eq!(&negative_source.shape[..], &[2, 3]);
    assert_eq!(negative_source.axes.len(), 2);
}

#[test]
fn scatter_fast_path_is_limited_to_rank_one_scalar_updates() {
    let rank_one = ScatterPlan::compile(
        &[16],
        &[1],
        &[16, 1],
        &[1, 16],
        &[16],
        &[1],
        &[16],
        &[1],
        ScatterSpec {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
    )
    .unwrap();
    assert!(rank_one.uses_rank_one_scalar_update_path());

    let generic = ScatterPlan::compile(
        &[4, 2],
        &[1, 4],
        &[4, 1],
        &[1, 4],
        &[4, 2],
        &[1, 4],
        &[4, 2],
        &[1, 4],
        ScatterSpec {
            update_window_dims: vec![1],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
    )
    .unwrap();
    assert!(!generic.uses_rank_one_scalar_update_path());
}

#[test]
fn dynamic_slice_fast_path_is_limited_to_rank_one_contiguous_layouts() {
    let contiguous = DynamicSlicePlan::compile(&[16], &[1], &[1], &[1], &[8], &[1], &[8]).unwrap();
    assert!(contiguous.uses_rank_one_contiguous_path());

    let higher_rank =
        DynamicSlicePlan::compile(&[4, 4], &[1, 4], &[2], &[1], &[2, 2], &[1, 2], &[2, 2]).unwrap();
    assert!(!higher_rank.uses_rank_one_contiguous_path());

    let strided = DynamicSlicePlan::compile(&[16], &[2], &[1], &[1], &[8], &[2], &[8]).unwrap();
    assert!(!strided.uses_rank_one_contiguous_path());
}

#[test]
fn dynamic_update_fast_path_is_limited_to_rank_one_contiguous_layouts() {
    let contiguous =
        DynamicUpdateSlicePlan::compile(&[16], &[1], &[1], &[1], &[8], &[1], &[16], &[1]).unwrap();
    assert!(contiguous.uses_rank_one_contiguous_path());

    let higher_rank = DynamicUpdateSlicePlan::compile(
        &[4, 4],
        &[1, 4],
        &[2],
        &[1],
        &[2, 2],
        &[1, 2],
        &[4, 4],
        &[1, 4],
    )
    .unwrap();
    assert!(!higher_rank.uses_rank_one_contiguous_path());

    let strided =
        DynamicUpdateSlicePlan::compile(&[16], &[2], &[1], &[1], &[8], &[2], &[16], &[2]).unwrap();
    assert!(!strided.uses_rank_one_contiguous_path());
}
