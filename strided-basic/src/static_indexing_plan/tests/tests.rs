use core::fmt::Debug;

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
