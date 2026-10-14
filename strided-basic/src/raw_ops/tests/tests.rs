use super::*;
use crate::ops_view::copy_scale;
use crate::{StridedView, StridedViewMut};

fn reference_copy_scale(
    dst: &mut [f64],
    src: &[f64],
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    scale: f64,
) {
    let mut dest_view = StridedViewMut::new(dst, dims, dst_strides, 0).unwrap();
    let src_view: StridedView<'_, f64> = StridedView::new(src, dims, src_strides, 0).unwrap();
    copy_scale(&mut dest_view, &src_view, scale).unwrap();
}

#[test]
fn raw_copy_scale_matches_view_kernel() {
    let dims = [2usize, 3, 2];
    let src_strides = [1isize, 2, 6];
    let dst_strides = [6isize, 2, 1];
    let src: Vec<f64> = (0..12).map(|value| value as f64 - 3.0).collect();
    let mut expected = vec![0.0; 12];
    reference_copy_scale(&mut expected, &src, &dims, &dst_strides, &src_strides, 1.5);

    let mut actual = vec![0.0; 12];
    let mut dest = RawStridedMut::new(&mut actual, &dims, &dst_strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
    copy_scale_raw(&mut dest, &source, 1.5).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn raw_axpy_accumulates() {
    let dims = [4usize];
    let strides = [1isize];
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let mut dst = [10.0f64, 20.0, 30.0, 40.0];
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    axpy_raw(&mut dest, &source, 2.0).unwrap();

    assert_eq!(dst, [12.0, 24.0, 36.0, 48.0]);
}

#[test]
fn raw_copy_scale_conjugates_complex_sources() {
    use num_complex::Complex64;
    let dims = [2usize];
    let strides = [1isize];
    let src = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
    let mut dst = [Complex64::new(0.0, 0.0); 2];
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    copy_scale_conj_raw(&mut dest, &source, Complex64::new(2.0, 0.0)).unwrap();

    assert_eq!(dst[0], Complex64::new(2.0, -4.0));
    assert_eq!(dst[1], Complex64::new(-6.0, -8.0));
}

/// Naive per-element reference: walk logical indices column-major.
fn reference_strided_copy(
    dst: &mut [i64],
    src: &[i64],
    dims: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    src_strides: &[isize],
    src_offset: isize,
) {
    let total: usize = dims.iter().product();
    for linear in 0..total {
        let mut rest = linear;
        let mut d = dst_offset;
        let mut s = src_offset;
        for axis in 0..dims.len() {
            let coord = (rest % dims[axis]) as isize;
            rest /= dims[axis];
            d += coord * dst_strides[axis];
            s += coord * src_strides[axis];
        }
        dst[d as usize] = src[s as usize];
    }
}

/// Every split of the logical range into three consecutive pieces must replay
/// exactly the serial result: this covers worker ranges that start and end in
/// the middle of an inner run and of outer axes.
#[test]
fn fused_range_replay_matches_reference_for_every_split() {
    // (dims, dst_strides, dst_offset, src_strides, src_offset, dst_len, src_len)
    type Case = (
        Vec<usize>,
        Vec<isize>,
        isize,
        Vec<isize>,
        isize,
        usize,
        usize,
    );
    let cases: Vec<Case> = vec![
        // contiguous rank 1
        (vec![7], vec![1], 0, vec![1], 0, 7, 7),
        // reversed source (stride -1)
        (vec![7], vec![1], 0, vec![-1], 6, 7, 7),
        // stride-2 source (slice with step)
        (vec![5, 3], vec![1, 5], 0, vec![2, 12], 1, 15, 40),
        // strided destination (transpose)
        (vec![3, 4], vec![4, 1], 0, vec![1, 3], 0, 12, 12),
        // negative outer source stride, reversed middle axis
        (vec![3, 2, 3], vec![1, 3, 6], 2, vec![1, -3, 6], 3, 20, 18),
        // negative strides on every axis
        (
            vec![2, 3, 2],
            vec![1, 2, 6],
            0,
            vec![-1, -2, -6],
            11,
            12,
            12,
        ),
        // stride-0 broadcast source
        (vec![4, 3], vec![1, 4], 0, vec![0, 1], 0, 12, 3),
    ];
    for (dims, dst_strides, dst_offset, src_strides, src_offset, dst_len, src_len) in cases {
        let src: Vec<i64> = (0..src_len as i64).map(|v| v * 10 + 1).collect();
        let mut expected = vec![-1i64; dst_len];
        reference_strided_copy(
            &mut expected,
            &src,
            &dims,
            &dst_strides,
            dst_offset,
            &src_strides,
            src_offset,
        );
        let layout = fuse_pair_layout(&dims, &dst_strides, &src_strides)
            .expect("fused rank fits the inline representation");
        let total = fused_total(&layout);
        assert_eq!(total, dims.iter().product::<usize>());

        let mut serial = vec![-1i64; dst_len];
        {
            let mut dest =
                RawStridedMut::new(&mut serial, &dims, &dst_strides, dst_offset).unwrap();
            let source = RawStridedRef::new(&src, &dims, &src_strides, src_offset).unwrap();
            apply_fused_pair(&mut dest, &source, &layout, |d, v| *d = v, |v| v);
        }
        assert_eq!(serial, expected, "serial replay for dims {dims:?}");

        for first in 0..=total {
            for second in first..=total {
                let mut actual = vec![-1i64; dst_len];
                for (start, end) in [(0, first), (first, second), (second, total)] {
                    // SAFETY: the offsets come from validated raw views of the
                    // same layout (checked by `RawStridedMut::new` above).
                    unsafe {
                        apply_fused_range(
                            actual.as_mut_ptr(),
                            dst_offset,
                            src.as_ptr(),
                            src_offset,
                            &layout,
                            start,
                            end - start,
                            &|d: &mut i64, v| *d = v,
                            &|v| v,
                        );
                    }
                }
                assert_eq!(actual, expected, "dims {dims:?} split at {first}, {second}");
            }
        }
    }
}

/// Column-major strides with one padding element between consecutive blocks, so
/// no two axes are contiguous and fusion cannot reduce the rank.
fn padded_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut next = 1isize;
    for &dim in dims {
        strides.push(next);
        next = next * dim as isize + 1;
    }
    strides
}

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut next = 1isize;
    for &dim in dims {
        strides.push(next);
        next *= dim as isize;
    }
    strides
}

/// Elements spanned by a layout from `offset`.
fn layout_span(dims: &[usize], strides: &[isize]) -> usize {
    let mut max = 0isize;
    let mut min = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides) {
        if dim == 0 {
            continue;
        }
        let end = (dim as isize - 1) * stride;
        if end > 0 {
            max += end;
        } else {
            min += end;
        }
    }
    (max - min + 1).max(1) as usize
}

/// Naive reference that walks the logical indices column-major and combines
/// each source value into its destination slot.
fn reference_apply(
    dims: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    dst_len: usize,
    src_strides: &[isize],
    src_offset: isize,
    src: &[f64],
    initial: f64,
    combine: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let mut dst = vec![initial; dst_len];
    let total: usize = dims.iter().product();
    for linear in 0..total {
        let mut rest = linear;
        let mut d = dst_offset;
        let mut s = src_offset;
        for axis in 0..dims.len() {
            let coord = (rest % dims[axis]) as isize;
            rest /= dims[axis];
            d += coord * dst_strides[axis];
            s += coord * src_strides[axis];
        }
        dst[d as usize] = combine(dst[d as usize], src[s as usize]);
    }
    dst
}

/// (dims, dst_strides, src_strides, dst_offset, src_offset, injective_dst)
type RankCase = (Vec<usize>, Vec<isize>, Vec<isize>, isize, isize, bool);

fn above_limit_rank_cases() -> Vec<(&'static str, RankCase)> {
    let padded9 = vec![2usize, 2, 2, 2, 2, 2, 2, 2, 16];
    let rank12 = vec![2usize; 12];
    let rank16 = vec![2usize; 16];
    vec![
        (
            // rank 8 boundary: non-fusible, still inline
            "rank8 padded",
            (
                padded9[..8].to_vec(),
                padded_strides(&padded9[..8]),
                padded_strides(&padded9[..8]),
                0,
                0,
                true,
            ),
        ),
        (
            // rank 9 that cannot fuse: the motivating cliff case
            "rank9 padded non-fusible",
            (
                padded9.clone(),
                padded_strides(&padded9),
                padded_strides(&padded9),
                0,
                0,
                true,
            ),
        ),
        (
            // rank 9, destination and source permuted against each other
            "rank12 permuted",
            (
                rank12.clone(),
                (0..12).map(|axis| 1isize << axis).collect(),
                (0..12).rev().map(|axis| 1isize << axis).collect(),
                0,
                0,
                true,
            ),
        ),
        (
            "rank16 padded",
            (
                rank16.clone(),
                padded_strides(&rank16),
                padded_strides(&rank16),
                0,
                0,
                true,
            ),
        ),
        (
            // rank 12 with every source stride negative
            "rank12 reversed source",
            {
                let strides = col_major_strides(&rank12);
                let offset: isize = rank12
                    .iter()
                    .zip(&strides)
                    .map(|(&dim, &stride)| (dim as isize - 1) * stride)
                    .sum();
                (
                    rank12.clone(),
                    strides.clone(),
                    strides.iter().map(|&stride| -stride).collect(),
                    0,
                    offset,
                    true,
                )
            },
        ),
        (
            // broadcast source over an injective destination
            "rank9 broadcast source",
            {
                let dims = vec![2usize, 2, 2, 2, 2, 2, 2, 2, 3];
                let dst = col_major_strides(&dims);
                let mut src = vec![0isize; 9];
                src[0] = 1;
                (dims, dst, src, 0, 0, false)
            },
        ),
        (
            // zero extent: nothing may be written
            "rank9 zero extent",
            {
                let mut dims = vec![2usize; 9];
                dims[4] = 0;
                let strides = col_major_strides(&dims);
                (dims, strides.clone(), strides, 0, 0, true)
            },
        ),
    ]
}

/// Ranks above the inline limit must fuse (not fall back) and replay the same
/// result as a dense logical-index reference for copy/scale/axpy.
#[test]
fn raw_ops_above_inline_limit_match_reference() {
    for (name, (dims, dst_strides, src_strides, dst_offset, src_offset, injective_dst)) in
        above_limit_rank_cases()
    {
        let total: usize = dims.iter().product();
        let dst_len = layout_span(&dims, &dst_strides) + dst_offset as usize + 1;
        let src_len = layout_span(&dims, &src_strides) + src_offset as usize + 1;
        let src: Vec<f64> = (0..src_len).map(|value| value as f64 * 0.5 - 1.0).collect();

        // Copy and scale (overwriting) are only defined for an injective
        // destination; axpy accumulates and is order-independent per slot.
        if injective_dst {
            let expected = reference_apply(
                &dims,
                &dst_strides,
                dst_offset,
                dst_len,
                &src_strides,
                src_offset,
                &src,
                0.0,
                |_, value| 2.0 * value,
            );
            let mut actual = vec![0.0f64; dst_len];
            let mut dest =
                RawStridedMut::new(&mut actual, &dims, &dst_strides, dst_offset).unwrap();
            let source = RawStridedRef::new(&src, &dims, &src_strides, src_offset).unwrap();
            copy_scale_raw(&mut dest, &source, 2.0).unwrap();
            assert_eq!(actual, expected, "{name}: copy_scale_raw");
        }

        let expected = reference_apply(
            &dims,
            &dst_strides,
            dst_offset,
            dst_len,
            &src_strides,
            src_offset,
            &src,
            1.0,
            |dst, value| dst + 2.0 * value,
        );
        let mut actual = vec![1.0f64; dst_len];
        let mut dest = RawStridedMut::new(&mut actual, &dims, &dst_strides, dst_offset).unwrap();
        let source = RawStridedRef::new(&src, &dims, &src_strides, src_offset).unwrap();
        axpy_raw(&mut dest, &source, 2.0).unwrap();
        assert_eq!(actual, expected, "{name}: axpy_raw");
        assert_eq!(total, dims.iter().product::<usize>());

        // The conjugating variants take the same path with an extra op.
        let expected = reference_apply(
            &dims,
            &dst_strides,
            dst_offset,
            dst_len,
            &src_strides,
            src_offset,
            &src,
            1.0,
            |dst, value| dst + value,
        );
        let mut actual = vec![1.0f64; dst_len];
        let mut dest = RawStridedMut::new(&mut actual, &dims, &dst_strides, dst_offset).unwrap();
        let source = RawStridedRef::new(&src, &dims, &src_strides, src_offset).unwrap();
        axpy_conj_raw(&mut dest, &source, 1.0).unwrap();
        assert_eq!(actual, expected, "{name}: axpy_conj_raw");
    }
}

/// Fusion happens before the storage representation is chosen: an original
/// rank above the limit that collapses to at most [`RAW_FUSED_RANK_LIMIT`]
/// fused axes stays inline, while a genuinely non-fusible high rank spills.
#[test]
fn fused_layout_fuses_before_choosing_storage() {
    // Rank 10, but four singleton axes and a contiguous block: one fused axis.
    let dims = vec![1usize, 1, 1, 1, 2, 2, 3, 4, 5, 6];
    let strides = col_major_strides(&dims);
    let inline = fuse_pair_layout(&dims, &strides, &strides).expect("fuses to one axis");
    assert_eq!(inline.rank, 1);
    assert_eq!(inline.dims[0], 1440);

    // Rank 9, every axis padded: fusion cannot reduce the rank, so the window
    // plans keep their non-fused replay and the raw path keeps the fused one.
    let dims = vec![2usize; 9];
    let strides = padded_strides(&dims);
    assert!(fuse_pair_layout(&dims, &strides, &strides).is_none());
    match fuse_pair_traversal(&dims, &strides, &strides) {
        PairTraversal::Spilled(axes) => {
            assert_eq!(axes.len(), 9);
            assert_eq!(axes.iter().map(|axis| axis.dim).product::<usize>(), 512);
        }
        PairTraversal::Inline(_) => panic!("a non-fusible rank 9 must spill"),
    }

    // Ranks within the limit keep the zero-heap representation.
    let dims = vec![2usize; 8];
    let strides = padded_strides(&dims);
    assert_eq!(fuse_pair_traversal(&dims, &strides, &strides).rank(), 8);
}

/// A spilled traversal must accumulate into a non-injective destination exactly
/// like the serial logical-index walk does.
#[test]
fn spilled_replay_accumulates_into_a_non_injective_destination() {
    let dims = vec![2usize; 9];
    // Padded strides never fuse, and making axis 3 repeat axis 2's destination
    // stride keeps that while sending two logical positions to every slot.
    let mut dst_strides = padded_strides(&dims);
    dst_strides[3] = dst_strides[2];
    let src_strides = padded_strides(&dims);
    let src_len = layout_span(&dims, &src_strides);
    let dst_len = layout_span(&dims, &dst_strides);
    let src: Vec<f64> = (0..src_len).map(|value| value as f64 * 0.25).collect();

    let layout = fuse_pair_layout(&dims, &dst_strides, &src_strides);
    assert!(
        layout.is_none(),
        "padded axes must not fuse, so the traversal spills"
    );
    assert!(
        !crate::layout_check::is_injective_layout(&dims, &dst_strides),
        "the destination repeats offsets"
    );

    let expected = reference_apply(
        &dims,
        &dst_strides,
        0,
        dst_len,
        &src_strides,
        0,
        &src,
        1.0,
        |dst, value| dst + 2.0 * value,
    );
    let mut actual = vec![1.0f64; dst_len];
    let mut dest = RawStridedMut::new(&mut actual, &dims, &dst_strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
    axpy_raw(&mut dest, &source, 2.0).unwrap();
    assert_eq!(actual, expected);

    // The same layout through a prepared plan.
    let plan = crate::CopyPlan::compile(&dims, &dst_strides, &src_strides);
    assert!(plan.is_err(), "an injective destination is required");
}

/// Every split of the logical range into three consecutive pieces must replay
/// exactly the serial result for a *spilled* layout too, including splits that
/// start and end inside the inner axis.
#[test]
fn spilled_range_replay_matches_reference_for_every_split() {
    let dims = vec![2usize; 9];
    let dst_strides: Vec<isize> = (0..9).map(|axis| 1isize << axis).collect();
    let src_strides: Vec<isize> = (0..9).rev().map(|axis| 1isize << axis).collect();
    let src: Vec<i64> = (0..512).map(|value| value * 10 + 1).collect();
    let mut expected = vec![-1i64; 512];
    reference_strided_copy(&mut expected, &src, &dims, &dst_strides, 0, &src_strides, 0);

    let layout = fuse_pair_traversal(&dims, &dst_strides, &src_strides);
    let axes = match &layout {
        PairTraversal::Spilled(axes) => axes,
        PairTraversal::Inline(_) => panic!("a non-fusible rank 9 must spill"),
    };
    assert_eq!(axes.len(), 9);

    let mut serial = vec![-1i64; 512];
    {
        let mut dest = RawStridedMut::new(&mut serial, &dims, &dst_strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
        apply_spilled_pair(&mut dest, &source, axes, |d, v| *d = v, |v| v);
    }
    assert_eq!(serial, expected, "serial spilled replay");

    // Exhaustive over every three-way split; 512 elements keeps this bounded.
    let total = 512usize;
    for first in 0..=total {
        for second in first..=total {
            let mut actual = vec![-1i64; 512];
            for (start, end) in [(0, first), (first, second), (second, total)] {
                // SAFETY: the offsets come from validated raw views of the same
                // layout (checked by `RawStridedMut::new` above).
                unsafe {
                    apply_spilled_range(
                        actual.as_mut_ptr(),
                        0,
                        src.as_ptr(),
                        0,
                        axes,
                        start,
                        end - start,
                        &|d: &mut i64, v| *d = v,
                        &|v| v,
                    );
                }
            }
            assert_eq!(actual, expected, "spilled split at {first}, {second}");
        }
    }
}
