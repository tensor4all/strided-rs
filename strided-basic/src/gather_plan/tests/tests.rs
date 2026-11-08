use super::{
    DynamicSlicePlan, DynamicUpdateSlicePlan, Exec, ScatterPlan, ScatterSpec, WindowReplay,
};

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
        // rank 9 original, but its singleton window axes collapse: fusion now
        // happens before the fused/inline choice, so this stays on the fused path
        (
            vec![2; 9],
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            0,
            vec![1; 9],
            vec![1; 9],
        ),
        // rank 9 and genuinely non-fusible (padded operand strides): the fused
        // window is unavailable, so the window replay fallback runs
        (
            vec![2; 9],
            vec![1, 3, 7, 15, 31, 63, 127, 255, 511],
            0,
            vec![2; 9],
            vec![0; 9],
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
        plan.execute(&mut dest, &source, &start_ref, Exec::serial())
            .unwrap();
        assert_eq!(
            read_back(&out),
            expected,
            "dims {dims:?} strides {strides:?}"
        );

        let mut storage = vec![MaybeUninit::<i64>::uninit(); total];
        let mut dest = super::RawStridedMut::new(&mut storage, &window, &dest_strides, 0).unwrap();
        plan.execute_uninit(&mut dest, &source, &start_ref, Exec::serial())
            .unwrap();
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
    plan.execute(&mut dest, &source, &update_ref, &start_ref, Exec::serial())
        .unwrap();
    assert_eq!(out, expected);

    let mut storage = vec![MaybeUninit::<i64>::uninit(); 63];
    let mut dest = super::RawStridedMut::new(&mut storage, &dims, &dest_strides, 0).unwrap();
    plan.execute_uninit(&mut dest, &source, &update_ref, &start_ref, Exec::serial())
        .unwrap();
    // SAFETY: the dense destination reaches every slot and a successful
    // uninit execute initializes every reachable slot.
    let uninit: Vec<i64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
    assert_eq!(uninit, expected);
}

/// The fused window is chosen by *fused* rank, so an original rank above the
/// limit still uses it when its window collapses, and a genuinely non-fusible
/// window keeps the compiled non-fused replay.
#[test]
fn dynamic_slice_window_uses_fused_replay_only_when_it_fuses() {
    let dims = vec![2usize; 9];
    let starts = [9usize];
    let start_strides = [1isize];

    // Singleton window axes collapse to one fused axis.
    let window = vec![1usize; 9];
    let dest_strides = vec![1isize; 9];
    let plan = DynamicSlicePlan::compile(
        &dims,
        &[1, 2, 4, 8, 16, 32, 64, 128, 256],
        &starts,
        &start_strides,
        &window,
        &dest_strides,
        &window,
    )
    .unwrap();
    assert!(
        plan.window.is_some(),
        "a rank-9 singleton window fuses and must not fall back"
    );

    // Padded operand strides cannot merge any window axis.
    let padded = [1isize, 3, 7, 15, 31, 63, 127, 255, 511];
    let window = vec![2usize; 9];
    let dest_strides: Vec<isize> = {
        let mut out = vec![0isize; 9];
        let mut scale = 1isize;
        for axis in (0..9).rev() {
            out[axis] = scale;
            scale *= window[axis] as isize;
        }
        out
    };
    let plan = DynamicSlicePlan::compile(
        &dims,
        &padded,
        &starts,
        &start_strides,
        &window,
        &dest_strides,
        &window,
    )
    .unwrap();
    assert!(
        plan.window.is_none(),
        "a non-fusible rank-9 window keeps the non-fused replay"
    );
}

/// Reference for a full-rank dynamic update: copy the operand, then overwrite
/// the clamped update window.
// Test reference: the operand, update and destination strides stay separate, with
// the data and start vectors spelled out, so each call reads as the case it builds.
#[allow(clippy::too_many_arguments)]
pub(super) fn dynamic_update_reference(
    dims: &[usize],
    operand_strides: &[isize],
    operand: &[i64],
    update_dims: &[usize],
    update_strides: &[isize],
    update: &[i64],
    starts: &[i64],
    dest_strides: &[isize],
) -> Vec<i64> {
    let rank = dims.len();
    let clamped: Vec<usize> = (0..rank)
        .map(|axis| starts[axis].clamp(0, (dims[axis] - update_dims[axis]) as i64) as usize)
        .collect();
    let total: usize = dims.iter().product();
    let mut expected = vec![0i64; total];
    let mut coords = vec![0usize; rank];
    let mut window_coords = vec![0usize; rank];
    for linear in 0..total {
        let mut rest = linear;
        for (coord, &dim) in coords.iter_mut().zip(dims.iter()) {
            *coord = rest % dim;
            rest /= dim;
        }
        let in_window = (0..rank)
            .all(|axis| (clamped[axis]..clamped[axis] + update_dims[axis]).contains(&coords[axis]));
        let (data, strides, source): (&[i64], &[isize], &[usize]) = if in_window {
            for (axis, (window_coord, &coord)) in
                window_coords.iter_mut().zip(coords.iter()).enumerate()
            {
                *window_coord = coord - clamped[axis];
            }
            (update, update_strides, &window_coords)
        } else {
            (operand, operand_strides, &coords)
        };
        let dest_offset: isize = (0..rank)
            .map(|axis| coords[axis] as isize * dest_strides[axis])
            .sum();
        let source_offset: isize = (0..rank)
            .map(|axis| source[axis] as isize * strides[axis])
            .sum();
        expected[dest_offset as usize] = data[source_offset as usize];
    }
    expected
}

/// The compiled non-fused window replay (a fused rank above
/// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT)) copies the operand and
/// then overwrites the clamped window from its preflighted replay bases.
#[test]
fn dynamic_update_generic_replay_matches_reference() {
    let dims = [2usize, 2, 2, 2, 2, 2, 2, 2, 4];
    let operand_strides = [1isize, 3, 7, 15, 31, 63, 127, 255, 511];
    let update_strides = [511isize, 255, 127, 63, 31, 15, 7, 3, 1];
    let dest_strides = [512isize, 256, 128, 64, 32, 16, 8, 4, 1];
    let update_dims = [2usize; 9];
    let starts = [0i64, 1, 0, 1, 0, 1, 0, i64::MIN, 3];
    let plan = DynamicUpdateSlicePlan::compile(
        &dims,
        &operand_strides,
        &[9],
        &[1],
        &update_dims,
        &update_strides,
        &dims,
        &dest_strides,
    )
    .unwrap();
    assert!(
        plan.window.is_none(),
        "a non-fusible rank-9 window keeps the compiled non-fused replay"
    );
    assert!(!plan.uses_rank_one_contiguous_path());

    let len = |dims: &[usize], strides: &[isize]| -> usize {
        1 + dims
            .iter()
            .zip(strides.iter())
            .map(|(&dim, &stride)| (dim as isize - 1) * stride.abs())
            .sum::<isize>() as usize
    };
    let operand: Vec<i64> = (0..len(&dims, &operand_strides) as i64).collect();
    let update: Vec<i64> = (0..len(&update_dims, &update_strides) as i64)
        .map(|value| -value - 1)
        .collect();
    let expected = dynamic_update_reference(
        &dims,
        &operand_strides,
        &operand,
        &update_dims,
        &update_strides,
        &update,
        &starts,
        &dest_strides,
    );

    let source = super::RawStridedRef::new(&operand, &dims, &operand_strides, 0).unwrap();
    let update_ref = super::RawStridedRef::new(&update, &update_dims, &update_strides, 0).unwrap();
    let start_ref = super::RawStridedRef::new(&starts, &[9], &[1], 0).unwrap();
    let mut out = vec![-1i64; expected.len()];
    let mut dest = super::RawStridedMut::new(&mut out, &dims, &dest_strides, 0).unwrap();
    plan.execute(&mut dest, &source, &update_ref, &start_ref, Exec::serial())
        .unwrap();
    assert_eq!(out, expected);
}

/// The preflight guards the whole window span, not just its base. These cases
/// call the preflight directly, so a missing guard fails an assertion instead
/// of sending an overflowing descriptor into the pointer replay, and each span
/// has its own case so deleting either `covers` call turns one of them red.
#[test]
fn preflight_rejects_a_window_that_overflows_the_update_span() {
    use crate::{RawStridedMut, RawStridedRef, StridedError};

    for (update_stride, update_offset) in [(1isize, isize::MAX), (-1, isize::MIN)] {
        let dims = [2usize];
        let update_storage = [0i32];
        let starts = [0i64];
        let update_strides = [update_stride];
        let plan = DynamicUpdateSlicePlan::compile(
            &dims,
            &[1],
            &[1],
            &[1],
            &dims,
            &update_strides,
            &dims,
            &[1],
        )
        .unwrap();
        // SAFETY: the preflight rejects this window, so the unreachable update
        // offset is never dereferenced.
        let update_ref = unsafe {
            RawStridedRef::new_unchecked(&update_storage[..], &dims, &update_strides, update_offset)
        };
        let start_ref = RawStridedRef::new(&starts, &[1], &[1], 0).unwrap();
        let mut out = [-1i32; 2];
        let dest = RawStridedMut::new(&mut out, &dims, &[1], 0).unwrap();
        let result = plan.preflight_update_window(&dest, &update_ref, &start_ref);
        assert!(
            matches!(result, Err(StridedError::OffsetOverflow)),
            "a window past the update span must be rejected"
        );
    }
}

/// The same whole-window guard on the destination side: a destination base that
/// fits `isize` is still rejected when the window reaches past it.
#[test]
fn preflight_rejects_a_window_that_overflows_the_dest_span() {
    use crate::{RawStridedMut, RawStridedRef, StridedError};

    for (dest_stride, dest_offset) in [(1isize, isize::MAX), (-1, isize::MIN)] {
        let dims = [2usize];
        let update = [3i32, 4];
        let starts = [0i64];
        let dest_strides = [dest_stride];
        let plan = DynamicUpdateSlicePlan::compile(
            &dims,
            &[1],
            &[1],
            &[1],
            &dims,
            &[1],
            &dims,
            &dest_strides,
        )
        .unwrap();
        let update_ref = RawStridedRef::new(&update, &dims, &[1], 0).unwrap();
        let start_ref = RawStridedRef::new(&starts, &[1], &[1], 0).unwrap();
        let mut out = [0i32; 2];
        // SAFETY: the preflight rejects this window before the copy writes
        // through the destination, so the unreachable offset is never used.
        let dest = unsafe {
            RawStridedMut::new_unchecked(&mut out[..], &dims, &dest_strides, dest_offset)
        };
        let result = plan.preflight_update_window(&dest, &update_ref, &start_ref);
        assert!(
            matches!(result, Err(StridedError::OffsetOverflow)),
            "a window past the destination span must be rejected"
        );
    }
}
