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

/// Naive column-major window reference for the fused window replay tests.
fn window_reference(
    operand: &[i64],
    operand_strides: &[isize],
    operand_offset: isize,
    starts: &[usize],
    window: &[usize],
) -> Vec<i64> {
    let total: usize = window.iter().product();
    (0..total)
        .map(|linear| {
            let mut rest = linear;
            let mut offset = operand_offset;
            for axis in 0..window.len() {
                let coord = rest % window[axis];
                rest /= window[axis];
                offset += (starts[axis] + coord) as isize * operand_strides[axis];
            }
            operand[offset as usize]
        })
        .collect()
}

#[test]
fn dynamic_slice_fused_window_matches_reference_for_general_layouts() {
    use core::mem::MaybeUninit;

    // (operand dims, operand strides, offset, window, starts (unclamped))
    type Case = (Vec<usize>, Vec<isize>, isize, Vec<usize>, Vec<i64>);
    let cases: Vec<Case> = vec![
        (vec![9, 7], vec![1, 9], 0, vec![4, 3], vec![2, 1]),
        (vec![9, 7], vec![-1, 9], 8, vec![4, 3], vec![-3, 9]),
        (vec![9, 7], vec![7, 1], 0, vec![9, 2], vec![0, 4]),
        (
            vec![5, 4, 3],
            vec![12, -3, 1],
            9,
            vec![2, 4, 2],
            vec![3, 0, 1],
        ),
        // rank 9: above the fused rank limit, uses the window replay fallback
        (
            vec![2; 9],
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            0,
            vec![1; 9],
            vec![1; 9],
        ),
    ];
    for (dims, strides, offset, window, starts) in cases {
        let rank = dims.len();
        let len = 1 + dims
            .iter()
            .zip(&strides)
            .map(|(&d, &s)| (d as isize - 1) * s.abs())
            .sum::<isize>() as usize;
        let operand: Vec<i64> = (0..len as i64).map(|v| v * 3 + 1).collect();
        let dest_strides: Vec<isize> = {
            // reversed axis order (transposed for rank 2)
            let mut out = vec![0isize; rank];
            let mut scale = 1isize;
            for axis in (0..rank).rev() {
                out[axis] = scale;
                scale *= window[axis] as isize;
            }
            out
        };
        let plan = DynamicSlicePlan::compile(
            &dims,
            &strides,
            &[rank],
            &[1],
            &window,
            &dest_strides,
            &window,
        )
        .unwrap();
        let clamped: Vec<usize> = (0..rank)
            .map(|a| starts[a].clamp(0, (dims[a] - window[a]) as i64) as usize)
            .collect();
        let expected = window_reference(&operand, &strides, offset, &clamped, &window);
        let total = expected.len();
        let source = super::RawStridedRef::new(&operand, &dims, &strides, offset).unwrap();
        let start_dims = [rank];
        let start_ref = super::RawStridedRef::new(&starts, &start_dims, &[1], 0).unwrap();
        let read_back = |data: &[i64]| -> Vec<i64> {
            (0..total)
                .map(|linear| {
                    let mut rest = linear;
                    let mut off = 0isize;
                    for axis in 0..rank {
                        off += (rest % window[axis]) as isize * dest_strides[axis];
                        rest /= window[axis];
                    }
                    data[off as usize]
                })
                .collect()
        };

        let mut out = vec![-1i64; total];
        let mut dest = super::RawStridedMut::new(&mut out, &window, &dest_strides, 0).unwrap();
        plan.execute(&mut dest, &source, &start_ref).unwrap();
        assert_eq!(
            read_back(&out),
            expected,
            "dims {dims:?} strides {strides:?}"
        );

        let mut storage = vec![MaybeUninit::<i64>::uninit(); total];
        let mut dest = super::RawStridedMut::new(&mut storage, &window, &dest_strides, 0).unwrap();
        plan.execute_uninit(&mut dest, &source, &start_ref).unwrap();
        // SAFETY: the dense destination reaches every slot and a successful
        // uninit execute initializes every reachable slot.
        let uninit: Vec<i64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
        assert_eq!(read_back(&uninit), expected, "uninit dims {dims:?}");
    }
}

#[test]
fn dynamic_update_slice_fused_window_matches_reference() {
    use core::mem::MaybeUninit;

    let dims = [9usize, 7];
    let strides = [-1isize, 9];
    let operand: Vec<i64> = (0..63).collect();
    let update: Vec<i64> = (0..12).map(|v| -v - 1).collect();
    let update_dims = [4usize, 3];
    let update_strides = [3isize, 1];
    let dest_strides = [7isize, 1];
    let plan = DynamicUpdateSlicePlan::compile(
        &dims,
        &strides,
        &[2],
        &[1],
        &update_dims,
        &update_strides,
        &dims,
        &dest_strides,
    )
    .unwrap();
    let starts = [7i64, -2];
    let (s0, s1) = (5usize, 0usize);
    let mut expected = vec![0i64; 63];
    for j in 0..7 {
        for i in 0..9 {
            expected[7 * i + j] = operand[8 - i + 9 * j];
        }
    }
    for j in 0..3 {
        for i in 0..4 {
            expected[7 * (s0 + i) + (s1 + j)] = update[3 * i + j];
        }
    }
    let source = super::RawStridedRef::new(&operand, &dims, &strides, 8).unwrap();
    let update_ref = super::RawStridedRef::new(&update, &update_dims, &update_strides, 0).unwrap();
    let start_ref = super::RawStridedRef::new(&starts, &[2], &[1], 0).unwrap();

    let mut out = vec![0i64; 63];
    let mut dest = super::RawStridedMut::new(&mut out, &dims, &dest_strides, 0).unwrap();
    plan.execute(&mut dest, &source, &update_ref, &start_ref)
        .unwrap();
    assert_eq!(out, expected);

    let mut storage = vec![MaybeUninit::<i64>::uninit(); 63];
    let mut dest = super::RawStridedMut::new(&mut storage, &dims, &dest_strides, 0).unwrap();
    plan.execute_uninit(&mut dest, &source, &update_ref, &start_ref)
        .unwrap();
    // SAFETY: the dense destination reaches every slot and a successful
    // uninit execute initializes every reachable slot.
    let uninit: Vec<i64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
    assert_eq!(uninit, expected);
}
