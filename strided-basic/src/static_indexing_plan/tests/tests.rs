use core::fmt::Debug;
use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};

use super::{PadPlan, RawStridedMut, RawStridedRef};

fn assert_contiguous_pad_matches_scalar<T>(operand_data: &[T], fill: T)
where
    T: Copy + Debug + PartialEq + super::MaybeSendSync,
{
    let operand_dims = [3usize, 2];
    let operand_strides = [1isize, -3];
    let operand_offset = 3isize;
    let dest_dims = [4usize, 4];
    let dest_strides = [1isize, 4];
    let dest_offset = 2isize;
    let edge_low = [-1i64, 1];
    let edge_high = [2i64, 0];
    let interior = [0i64, 1];
    let plan = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &edge_low,
        &edge_high,
        &interior,
    )
    .unwrap();
    assert!(plan.contiguous_axis0_run.is_some());

    let mut scalar_plan = plan.clone();
    scalar_plan.contiguous_dest_fill = false;
    scalar_plan.contiguous_axis0_run = None;
    scalar_plan.contiguous_axis0_columns = None;
    let mut fast_dest = vec![fill; 20];
    let mut scalar_dest = fast_dest.clone();
    let operand = RawStridedRef::new(
        operand_data,
        &operand_dims,
        &operand_strides,
        operand_offset,
    )
    .unwrap();
    {
        let mut dest =
            RawStridedMut::new(&mut fast_dest, &dest_dims, &dest_strides, dest_offset).unwrap();
        plan.execute(&mut dest, &operand, fill).unwrap();
    }
    {
        let mut dest =
            RawStridedMut::new(&mut scalar_dest, &dest_dims, &dest_strides, dest_offset).unwrap();
        scalar_plan.execute(&mut dest, &operand, fill).unwrap();
    }
    assert_eq!(fast_dest, scalar_dest);
}

#[test]
fn pad_plan_selects_contiguous_axis0_run_for_dense_edge_padding() {
    let plan =
        PadPlan::compile(&[2_097_152], &[1], &[2_097_408], &[1], &[128], &[128], &[0]).unwrap();

    assert_eq!(plan.contiguous_axis0_run(), Some((0, 128, 2_097_152)));
    assert!(plan.has_contiguous_dest_fill());
}

#[test]
fn contiguous_pad_matches_scalar_for_every_erased_scalar_type() {
    assert_contiguous_pad_matches_scalar(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], -1.0);
    assert_contiguous_pad_matches_scalar(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], -1.0);
    assert_contiguous_pad_matches_scalar(&[1i32, 2, 3, 4, 5, 6], -1);
    assert_contiguous_pad_matches_scalar(&[1i64, 2, 3, 4, 5, 6], -1);
    assert_contiguous_pad_matches_scalar(&[true, false, true, false, true, false], false);
    assert_contiguous_pad_matches_scalar(
        &[
            Complex32::new(1.0, -1.0),
            Complex32::new(2.0, -2.0),
            Complex32::new(3.0, -3.0),
            Complex32::new(4.0, -4.0),
            Complex32::new(5.0, -5.0),
            Complex32::new(6.0, -6.0),
        ],
        Complex32::new(-1.0, 0.0),
    );
    assert_contiguous_pad_matches_scalar(
        &[
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
            Complex64::new(5.0, -5.0),
            Complex64::new(6.0, -6.0),
        ],
        Complex64::new(-1.0, 0.0),
    );
}

#[test]
fn pad_plan_selects_axis0_columns_only_for_dense_non_cropping_edge_padding() {
    let operand_dims = [5usize, 4];
    let operand_strides = [1isize, 5];
    let dest_dims = [9usize, 8];
    let dest_strides = [1isize, 9];

    let selected = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &[2, 2],
        &[2, 2],
        &[0, 0],
    )
    .unwrap();
    assert!(selected.has_contiguous_axis0_columns());

    // Interior spacing on any axis keeps the fill-then-copy fallback.
    let interior = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &[9, 11],
        &[1, 9],
        &[2, 2],
        &[2, 2],
        &[0, 1],
    )
    .unwrap();
    assert!(!interior.has_contiguous_axis0_columns());

    // A non-dense destination keeps the fallback.
    let strided = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &[9, 1],
        &[2, 2],
        &[2, 2],
        &[0, 0],
    )
    .unwrap();
    assert!(!strided.has_contiguous_axis0_columns());

    // Cropping on either axis keeps the fallback.
    let crop_axis0 = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &[7, 8],
        &[1, 7],
        &[-1, 2],
        &[3, 2],
        &[0, 0],
    )
    .unwrap();
    assert!(!crop_axis0.has_contiguous_axis0_columns());
    let crop_axis1 = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &[9, 6],
        &[1, 9],
        &[2, -1],
        &[2, 3],
        &[0, 0],
    )
    .unwrap();
    assert!(!crop_axis1.has_contiguous_axis0_columns());

    // Unit axis-0 operand stride and dense destination with no padding at all
    // still selects the single-pass write (each element is written once).
    let exact = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &operand_dims,
        &operand_strides,
        &[0, 0],
        &[0, 0],
        &[0, 0],
    )
    .unwrap();
    assert!(exact.has_contiguous_axis0_columns());
}

/// Compare the single-pass column write against the generic fill-then-copy
/// fallback for a dense column-major destination.
fn assert_axis0_columns_match_fallback(
    operand_dims: &[usize],
    operand_strides: &[isize],
    dest_dims: &[usize],
    edge_low: &[i64],
    edge_high: &[i64],
) {
    let dest_strides = col_major_strides(dest_dims);
    let plan = PadPlan::compile(
        operand_dims,
        operand_strides,
        dest_dims,
        &dest_strides,
        edge_low,
        edge_high,
        &vec![0i64; operand_dims.len()],
    )
    .unwrap();
    assert!(
        plan.has_contiguous_axis0_columns(),
        "dense non-cropping layout should select the single-pass write"
    );
    let mut fallback = plan.clone();
    fallback.contiguous_dest_fill = false;
    fallback.contiguous_axis0_run = None;
    fallback.contiguous_axis0_columns = None;

    let operand_total: usize = operand_dims.iter().product();
    let operand: Vec<f64> = (0..operand_total).map(|value| value as f64 + 1.0).collect();
    let source = RawStridedRef::new(&operand, operand_dims, operand_strides, 0).unwrap();
    let dest_total: usize = dest_dims.iter().product();
    let fill = -1.0f64;

    let mut fast = vec![fill; dest_total];
    let mut generic = vec![fill; dest_total];
    {
        let mut dest = RawStridedMut::new(&mut fast, dest_dims, &dest_strides, 0).unwrap();
        plan.execute(&mut dest, &source, fill).unwrap();
    }
    {
        let mut dest = RawStridedMut::new(&mut generic, dest_dims, &dest_strides, 0).unwrap();
        fallback.execute(&mut dest, &source, fill).unwrap();
    }
    assert_eq!(fast, generic, "rank {:?}", operand_dims.len());
}

#[test]
fn axis0_columns_match_fallback_for_ranks_one_to_three() {
    // rank 1, low/high pad > 0, = 0
    assert_axis0_columns_match_fallback(&[7], &[1], &[7], &[0], &[0]);
    assert_axis0_columns_match_fallback(&[7], &[1], &[12], &[3], &[2]);
    assert_axis0_columns_match_fallback(&[7], &[1], &[9], &[0], &[2]);

    // rank 2, dense operand and destination
    assert_axis0_columns_match_fallback(&[5, 4], &[1, 5], &[5, 4], &[0, 0], &[0, 0]);
    assert_axis0_columns_match_fallback(&[5, 4], &[1, 5], &[9, 8], &[2, 2], &[2, 2]);
    assert_axis0_columns_match_fallback(&[5, 4], &[1, 5], &[8, 4], &[3, 0], &[0, 0]);

    // rank 3, dense operand and destination
    assert_axis0_columns_match_fallback(
        &[3, 4, 5],
        &[1, 3, 12],
        &[3, 4, 5],
        &[0, 0, 0],
        &[0, 0, 0],
    );
    assert_axis0_columns_match_fallback(
        &[3, 4, 5],
        &[1, 3, 12],
        &[6, 7, 9],
        &[1, 2, 3],
        &[2, 1, 1],
    );
}

#[test]
fn axis0_columns_uninit_initializes_every_poisoned_slot() {
    let operand_dims = [5usize, 4];
    let operand_strides = [1isize, 5];
    let dest_dims = [9usize, 8];
    let dest_strides = [1isize, 9];
    let plan = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &[2, 2],
        &[2, 2],
        &[0, 0],
    )
    .unwrap();
    assert!(plan.has_contiguous_axis0_columns());

    let operand: Vec<f64> = (1..=20).map(|value| value as f64).collect();
    let source = RawStridedRef::new(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let mut storage = vec![MaybeUninit::new(f64::NAN); dest_dims.iter().product()];
    let mut dest = RawStridedMut::new(&mut storage, &dest_dims, &dest_strides, 0).unwrap();
    plan.execute_uninit(&mut dest, &source, -1.0).unwrap();
    // SAFETY: every reachable destination slot was written by the plan.
    let values: Vec<f64> = storage
        .iter()
        .map(|slot| unsafe { slot.assume_init() })
        .collect();
    assert!(values.iter().all(|value| !value.is_nan()));

    let mut reference = vec![-1.0f64; dest_dims.iter().product()];
    let mut dest = RawStridedMut::new(&mut reference, &dest_dims, &dest_strides, 0).unwrap();
    plan.execute(&mut dest, &source, -1.0).unwrap();
    assert_eq!(values, reference);
}

#[test]
fn pad_benchmark_recipes_use_generic_replay() {
    for (label, rank, crop, nonunit) in [
        ("compact_rank2", 2usize, false, false),
        ("compact_rank4", 4, false, false),
        ("compact_rank8", 8, false, false),
        ("rank2_negative_crop", 2, true, false),
        ("rank2_nonunit", 2, false, true),
    ] {
        let mut operand_dims = vec![2usize; rank - 1];
        operand_dims.push(8);
        let interior = std::iter::once(1i64)
            .chain(std::iter::repeat_n(0i64, rank - 1))
            .collect::<Vec<_>>();
        let mut edge_low = vec![0i64; rank];
        let mut edge_high = vec![0i64; rank];
        if crop {
            edge_low[1] = -1;
            edge_high[1] = 1;
        }
        let operand_strides = if nonunit {
            vec![2isize, 4]
        } else {
            col_major_strides(&operand_dims)
        };
        let dest_dims = operand_dims
            .iter()
            .zip(&interior)
            .zip(edge_low.iter().zip(&edge_high))
            .map(|((&dim, &inner), (&low, &high))| {
                (low + (dim as i64 - 1) * (inner + 1) + high + 1) as usize
            })
            .collect::<Vec<_>>();
        let dest_strides = if nonunit {
            vec![2isize, 6]
        } else {
            col_major_strides(&dest_dims)
        };
        let plan = PadPlan::compile(
            &operand_dims,
            &operand_strides,
            &dest_dims,
            &dest_strides,
            &edge_low,
            &edge_high,
            &interior,
        )
        .unwrap();
        assert!(
            plan.contiguous_axis0_run.is_none(),
            "{label} unexpectedly selected the axis-0 fast path"
        );
        assert!(plan.generic_copy.total > 0, "{label} has no copy domain");
    }
}

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut stride = 1isize;
    dims.iter()
        .map(|&dim| {
            let current = stride;
            stride *= dim as isize;
            current
        })
        .collect()
}

#[test]
fn axis0_columns_match_fallback_for_rank_five_small_axis0() {
    // Axis-0 extent 1 and 2 keep the single-pass column write selected while
    // the outer odometer has four axes to carry, so the incremental operand
    // offset must track entering and leaving the in-bounds interval per axis.
    assert_axis0_columns_match_fallback(
        &[1, 3, 4, 2, 5],
        &[1, 1, 3, 12, 24],
        &[1, 5, 6, 5, 8],
        &[0, 1, 1, 2, 1],
        &[0, 1, 1, 1, 2],
    );
    assert_axis0_columns_match_fallback(
        &[2, 3, 4, 2, 5],
        &[1, 2, 6, 24, 48],
        &[4, 5, 6, 5, 8],
        &[1, 1, 1, 2, 1],
        &[1, 1, 1, 1, 2],
    );
}

#[test]
fn axis0_columns_uninit_writes_bool_slots() {
    // A `bool` destination with genuinely uninitialized backing storage: the
    // single-pass column write must fill pad spans through `MaybeUninit<bool>`
    // and never form a `&mut [bool]` over uninitialized memory.
    let operand_dims = [1usize, 3, 4, 2, 5];
    let operand_strides = [1isize, 1, 3, 12, 24];
    let dest_dims = [2usize, 5, 6, 5, 8];
    let dest_strides = col_major_strides(&dest_dims);
    let edge_low = [1i64, 1, 1, 2, 1];
    let edge_high = [0i64, 1, 1, 1, 2];
    let plan = PadPlan::compile(
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &edge_low,
        &edge_high,
        &[0i64; 5],
    )
    .unwrap();
    assert!(plan.has_contiguous_axis0_columns());

    let operand: Vec<bool> = (0..120).map(|index| index % 2 == 0).collect();
    let source = RawStridedRef::new(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let dest_total = dest_dims.iter().product();
    let mut storage = vec![MaybeUninit::<bool>::uninit(); dest_total];
    let mut dest = RawStridedMut::new(&mut storage, &dest_dims, &dest_strides, 0).unwrap();
    plan.execute_uninit(&mut dest, &source, false).unwrap();
    // SAFETY: every reachable destination slot was written by the plan.
    let values: Vec<bool> = storage
        .iter()
        .map(|slot| unsafe { slot.assume_init() })
        .collect();

    let mut reference = vec![false; dest_total];
    let mut dest = RawStridedMut::new(&mut reference, &dest_dims, &dest_strides, 0).unwrap();
    plan.execute(&mut dest, &source, false).unwrap();
    assert_eq!(values, reference);
}
