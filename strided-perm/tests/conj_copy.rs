//! `copy_conj_into` / `copy_conj_into_uninit` conjugate on the permutation
//! engine's store (tensor4all/strided-rs#467): same plan and validation as
//! `copy_into`, one conjugation per stored element and no extra pass.
//!
//! Reference values are computed by index arithmetic over the flat buffers, so
//! the expected conjugation never reuses the code under test.

use core::mem::MaybeUninit;
use std::time::Duration;

use num_complex::Complex;
use strided_perm::{copy_conj_into, copy_conj_into_uninit, copy_into};
use strided_view::{col_major_strides, ElementOpApply, StridedView, StridedViewMut};
use threadteam::{ledger, DispatchCosts, Exec, Pool, WidthCosts};

/// A pool whose section is free, so the width decision cannot narrow a copy away
/// from the parallel store arms on a host with few cores. Same construction as
/// `policy_guard.rs`; with the default costs a 2-core CI runner keeps the copy
/// on the calling thread and the fan-out assertions below never see a task.
fn forcing_pool() -> Pool {
    let free = WidthCosts::flat(Duration::ZERO);
    Pool::builder()
        .threads(4)
        .dispatch_costs(DispatchCosts::new(free, free, free))
        .build()
        .unwrap()
}

/// Column-major strides of `dims`.
fn strides_of(dims: &[usize]) -> Vec<isize> {
    col_major_strides(dims).to_vec()
}

/// Flat column-major offset of `coord` in `dims`.
fn offset_of(dims: &[usize], coord: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&d, &c) in dims.iter().zip(coord) {
        offset += c * stride;
        stride *= d;
    }
    offset
}

/// Reference conjugation of `src_data[dims]` permuted by `perm`, at every flat
/// destination offset in `view_dims`.
fn reference<T: Copy + ElementOpApply>(
    src_data: &[T],
    dims: &[usize],
    perm: &[usize],
    view_dims: &[usize],
) -> Vec<T> {
    let total: usize = view_dims.iter().product();
    let mut expected = Vec::with_capacity(total);
    for flat in 0..total {
        // Unravel `flat` in the permuted (view) dims.
        let mut rem = flat;
        let mut view_coord = vec![0usize; view_dims.len()];
        for (c, &d) in view_coord.iter_mut().zip(view_dims) {
            *c = rem % d;
            rem /= d;
        }
        // Undo the permutation to get the source coordinates.
        let mut src_coord = vec![0usize; dims.len()];
        for (k, &p) in perm.iter().enumerate() {
            src_coord[p] = view_coord[k];
        }
        expected.push(src_data[offset_of(dims, &src_coord)].conj());
    }
    expected
}

/// Copy the `perm`-permuted view of a dense column-major `dims` tensor with both
/// conjugating entries, serial and pooled, and compare with the reference.
fn check_perm<T>(
    dims: &[usize],
    perm: &[usize],
    lift: impl Fn(&[usize]) -> T + Copy,
    parallel: bool,
    label: &str,
) where
    T: Copy + PartialEq + core::fmt::Debug + Send + Sync + 'static + ElementOpApply,
{
    let total: usize = dims.iter().product();
    let mut src_data = Vec::with_capacity(total);
    for flat in 0..total {
        let mut rem = flat;
        let mut coord = vec![0usize; dims.len()];
        for (c, &d) in coord.iter_mut().zip(dims) {
            *c = rem % d;
            rem /= d;
        }
        src_data.push(lift(&coord));
    }
    let src_dense = StridedView::new(&src_data, dims, &strides_of(dims), 0).unwrap();
    let src = src_dense.permute(perm).unwrap();
    let view_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
    let expected = reference(&src_data, dims, perm, &view_dims);

    let pool = forcing_pool();
    let execs = if parallel {
        vec![Exec::serial(), Exec::new(&pool, 4).unwrap()]
    } else {
        vec![Exec::serial()]
    };
    for mut exec in execs {
        let root = exec.ledger_root();
        let width = exec.budget().get();

        let mut got = vec![src_data[0]; expected.len()];
        let mut got_view =
            StridedViewMut::new(&mut got, &view_dims, &strides_of(&view_dims), 0).unwrap();
        copy_conj_into(&mut got_view, &src, exec.reborrow()).unwrap();
        assert_eq!(got, expected, "{label} copy_conj_into width {width}");
        // Snapshot here, before any other call on this token: rebinding a token
        // keeps its ledger, so the plain copy below would otherwise install the
        // work this assertion is meant to attribute to the conjugating copy.
        let main_installs = root.map(ledger::installs);

        let mut fresh: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); expected.len()];
        let mut fresh_view =
            StridedViewMut::new(&mut fresh, &view_dims, &strides_of(&view_dims), 0).unwrap();
        // A second *token*, not a second `ledger_root()` on this one: a root is
        // per token, so a shared root would let the first call's fan-out stand
        // in for the second's.
        let fresh_exec = Exec::new(&pool, width).unwrap();
        let fresh_parallel = !fresh_exec.is_serial();
        let fresh_root = fresh_exec.ledger_root();
        copy_conj_into_uninit(&mut fresh_view, &src, fresh_exec).unwrap();
        let fresh_installs = fresh_root.map(ledger::installs);
        // SAFETY: the copy wrote every destination element.
        let fresh: Vec<T> = fresh
            .into_iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();
        assert_eq!(
            fresh, expected,
            "{label} copy_conj_into_uninit width {width}"
        );

        // A plain copy of the same view must differ wherever the conjugate does,
        // so the assertions above cannot pass on a copy that ignored the store.
        let differs = expected
            .iter()
            .zip(&src_data)
            .any(|(conj, plain)| conj != plain);
        if differs {
            let mut plain = vec![src_data[0]; expected.len()];
            let mut plain_view =
                StridedViewMut::new(&mut plain, &view_dims, &strides_of(&view_dims), 0).unwrap();
            copy_into(&mut plain_view, &src, exec.reborrow()).unwrap();
            assert_ne!(plain, expected, "{label} copy_into must not conjugate");
        }

        if parallel {
            // A free section always fans out, so this asserts the parallel store
            // arms ran rather than assuming them. `installs`, not `max_live`:
            // how many tasks are live at once is the host's scheduling, not the
            // engine's, and a 2-core runner decided this test's first version.
            for (entry, is_parallel, installs) in [
                ("copy_conj_into", !exec.is_serial(), main_installs),
                ("copy_conj_into_uninit", fresh_parallel, fresh_installs),
            ] {
                if !is_parallel {
                    continue;
                }
                let installs = installs.expect("a token from `new` is a ledger root");
                assert!(
                    installs > 0,
                    "{label} {entry} installed no work at width {width}"
                );
            }
        }
    }
}

/// Both element widths over 2D, 3D, 4D, unit-extent, rank-0 and gapped layouts.
fn check_both_widths() {
    // 2D transpose.
    check_perm(
        &[4, 3],
        &[1, 0],
        |c| Complex::new(c[0] as f64 * 0.5, 7.0 - c[1] as f64),
        false,
        "c64 2d transpose",
    );
    check_perm(
        &[4, 3],
        &[1, 0],
        |c| Complex::new(c[0] as f32 * 0.5, 7.0 - c[1] as f32),
        false,
        "c32 2d transpose",
    );
    // 3D cyclic permutation.
    check_perm(
        &[3, 4, 5],
        &[2, 0, 1],
        |c| Complex::new(c[0] as f64, (c[1] * 10 + c[2]) as f64),
        false,
        "c64 3d cyclic",
    );
    check_perm(
        &[3, 4, 5],
        &[2, 0, 1],
        |c| Complex::new(c[0] as f32, (c[1] * 10 + c[2]) as f32),
        false,
        "c32 3d cyclic",
    );
    // 4D permutation.
    check_perm(
        &[2, 3, 2, 5],
        &[3, 1, 0, 2],
        |c| Complex::new(c[3] as f64 + 0.25, c[0] as f64),
        false,
        "c64 4d",
    );
    // A unit extent on the source.
    check_perm(
        &[5, 1, 7],
        &[2, 0, 1],
        |c| Complex::new(c[0] as f64, c[2] as f64),
        false,
        "c64 unit extent",
    );
    // Rank 0: one element, conjugated.
    check_perm(
        &[],
        &[],
        |_| Complex::new(1.5f64, -2.5),
        false,
        "c64 rank 0",
    );
    // Contiguous same-layout copy: the conj store must not take the memcpy path.
    check_perm(
        &[8, 8],
        &[0, 1],
        |c| Complex::new(c[0] as f64, c[1] as f64),
        false,
        "c64 contiguous",
    );
    // 2^18 elements: above the width decision's one-section bar, so both entries
    // run on the pooled token too — the only place `_uninit` sees the parallel
    // store arms.
    check_perm(
        &[512, 512],
        &[1, 0],
        |c| Complex::new(c[0] as f64, c[1] as f64 * 0.5),
        true,
        "c64 2d transpose parallel",
    );
}

/// Layouts that take an explicit stride pair instead of a permutation: a gapped
/// source, a negative source stride, the contiguous parallel stream and a
/// transposed parallel tile grid.
fn check_layout<T>(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    lift: impl Fn(usize) -> T + Copy,
    parallel: bool,
    label: &str,
) where
    T: Copy + PartialEq + core::fmt::Debug + Send + Sync + 'static + ElementOpApply,
{
    let span = |strides: &[isize]| -> usize {
        1 + dims
            .iter()
            .zip(strides)
            .map(|(&n, &s)| (n - 1) * s.unsigned_abs())
            .sum::<usize>()
    };
    // A negative source stride reads below the origin, so the view is placed at
    // the lowest reachable offset.
    let src_low: isize = dims
        .iter()
        .zip(src_strides)
        .map(|(&n, &s)| (n as isize - 1) * s.min(0))
        .sum();
    let source: Vec<T> = (0..span(src_strides)).map(lift).collect();
    let mut expected = vec![source[0]; span(dst_strides)];
    let count: usize = dims.iter().product();
    for flat in 0..count {
        let (mut s, mut d) = (-src_low, 0isize);
        let mut rem = flat;
        for (&n, (&ss, &ds)) in dims.iter().zip(src_strides.iter().zip(dst_strides)) {
            let c = (rem % n) as isize;
            rem /= n;
            s += c * ss;
            d += c * ds;
        }
        expected[d as usize] = source[s as usize].conj();
    }
    let src = StridedView::new(&source, dims, src_strides, -src_low).unwrap();

    let pool = forcing_pool();
    let execs = if parallel {
        vec![Exec::serial(), Exec::new(&pool, 4).unwrap()]
    } else {
        vec![Exec::serial()]
    };
    for mut exec in execs {
        let root = exec.ledger_root();
        let mut got = vec![source[0]; expected.len()];
        let mut got_view = StridedViewMut::new(&mut got, dims, dst_strides, 0).unwrap();
        copy_conj_into(&mut got_view, &src, exec.reborrow()).unwrap();
        assert_eq!(got, expected, "{label}");
        if parallel && !exec.is_serial() {
            // A free section forces the fan-out, so this asserts the parallel
            // store arm ran instead of trusting the element count to earn one.
            let root = root.expect("a token from `new` is a ledger root");
            assert!(
                ledger::installs(root) > 0,
                "{label} installed no work on a pooled token"
            );
        }
    }
}

#[test]
fn matches_the_reference_on_every_permutation_shape() {
    check_both_widths();
}

#[test]
fn gapped_source_strides_conjugate() {
    // Non-unit inner source stride: ConstStride1 with an explicit stride.
    check_layout(
        &[6, 4],
        &[2, 12],
        &[1, 6],
        |i| Complex::new(i as f64, -(i as f64) * 0.5),
        false,
        "c64 gapped",
    );
}

#[test]
fn broadcast_source_conjugates() {
    // A stride-0 source dimension: one element read into six destinations.
    check_layout(
        &[6, 4],
        &[0, 1],
        &[1, 6],
        |i| Complex::new(i as f64 + 0.5, 2.0),
        false,
        "c64 broadcast",
    );
}

#[test]
fn non_unit_destination_stride_conjugates() {
    // Non-unit inner destination stride: the store cannot stream a row.
    check_layout(
        &[5, 3],
        &[1, 5],
        &[2, 10],
        |i| Complex::new(i as f64, -(i as f64)),
        false,
        "c64 gapped destination",
    );
}

#[test]
fn empty_view_conjugates_nothing() {
    let mut dst = [Complex::new(9.0f64, 9.0); 3];
    let src_data = [Complex::new(1.0f64, 2.0); 3];
    let mut dst_view = StridedViewMut::new(&mut dst, &[3, 0], &[1, 3], 0).unwrap();
    let src = StridedView::new(&src_data, &[3, 0], &[1, 3], 0).unwrap();
    copy_conj_into(&mut dst_view, &src, Exec::serial()).unwrap();
    assert_eq!(dst, [Complex::new(9.0f64, 9.0); 3]);
}

#[test]
fn negative_source_stride_conjugates() {
    check_layout(
        &[3, 4],
        &[-1, 3],
        &[1, 3],
        |i| Complex::new(i as f32, -3.0),
        false,
        "c32 reversed",
    );
}

#[test]
fn contiguous_parallel_stream_conjugates() {
    // 2^20 elements: the token's width decision fans this out, so the parallel
    // ConstStride1 store runs (asserted through the ledger).
    check_layout(
        &[1024, 1024],
        &[1, 1024],
        &[1, 1024],
        |i| Complex::new((i % 97) as f64, (i % 13) as f64),
        true,
        "c64 contiguous parallel",
    );
}

#[test]
fn transposed_parallel_tile_grid_conjugates() {
    // 2^d + 7: a 2D transpose whose tile grid is split across the budget.
    check_layout(
        &[1024, 1031],
        &[256, 1],
        &[1, 1024],
        |i| Complex::new(i as f32, 1.5),
        true,
        "c32 transposed parallel",
    );
}

#[test]
fn grouped_tile_layout_conjugates() {
    // The rank-8 scattered layout `hptt::plan` selects GroupedTranspose for.
    check_layout(
        &[2, 2, 2, 2, 2, 2, 2, 2],
        &[16, 1024, 8388608, 4096, 1048576, 1, 8, 131072],
        &[1, 2, 4, 8, 16, 32, 64, 128],
        |i| Complex::new(i as f64, -1.0),
        false,
        "c64 grouped tile",
    );
}

#[test]
fn shape_mismatch_is_rejected_without_writing() {
    let src_data = [Complex::new(1.0f64, 2.0); 6];
    let src = StridedView::new(&src_data, &[2, 3], &[1, 2], 0).unwrap();
    let mut dst = [Complex::new(9.0f64, 9.0); 4];
    let mut dst_view = StridedViewMut::new(&mut dst, &[2, 2], &[1, 2], 0).unwrap();
    assert!(matches!(
        copy_conj_into(&mut dst_view, &src, Exec::serial()),
        Err(strided_view::StridedError::ShapeMismatch(..))
    ));
    assert_eq!(dst, [Complex::new(9.0f64, 9.0); 4]);
}

#[test]
fn non_injective_destination_is_rejected() {
    let src_data = [Complex::new(1.0f64, 2.0); 4];
    let src = StridedView::new(&src_data, &[2, 2], &[1, 2], 0).unwrap();
    let mut dst = [Complex::new(0.0f64, 0.0); 4];
    let mut dst_view = StridedViewMut::new(&mut dst, &[2, 2], &[1, 0], 0).unwrap();
    assert!(matches!(
        copy_conj_into(&mut dst_view, &src, Exec::serial()),
        Err(strided_view::StridedError::NonInjectiveOutputLayout)
    ));
    assert_eq!(dst, [Complex::new(0.0f64, 0.0); 4]);
}
