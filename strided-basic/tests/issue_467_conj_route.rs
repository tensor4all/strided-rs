//! Output correctness of `strided_basic::copy_conj_into` on both routes
//! (tensor4all/strided-rs#467).
//!
//! The route is asserted in-crate (`ops_view::conj_route_tests`); these tests
//! pin that whichever route the gate takes writes the same elements as the other
//! one and as a naive index-arithmetic reference, including the power-of-two and
//! aliasing shapes of the issue, gapped and negative-stride destinations, unit
//! extents, rank 0 and empty views.

use num_complex::{Complex32, Complex64};
use strided_basic::{copy_conj_into, copy_into, Exec, StridedError, StridedView, StridedViewMut};
use strided_view::ElementOpApply;

/// Elements a layout touches from its lowest to its highest offset.
fn span(dims: &[usize], strides: &[isize]) -> usize {
    let reach: usize = dims
        .iter()
        .zip(strides)
        .map(|(&d, &s)| d.saturating_sub(1) * s.unsigned_abs())
        .sum();
    reach + 1
}

/// The origin that keeps a negative-stride layout inside its span.
fn origin(dims: &[usize], strides: &[isize]) -> isize {
    dims.iter()
        .zip(strides)
        .map(|(&d, &s)| if s < 0 { (d as isize - 1) * -s } else { 0 })
        .sum()
}

/// A mask of the offsets the layout touches, so the others can be checked
/// untouched. Linear in the element count, not in its square.
fn touched(dims: &[usize], strides: &[isize], origin: isize, span: usize) -> Vec<bool> {
    let mut mask = vec![false; span];
    let mut idx = vec![0usize; dims.len()];
    loop {
        let offset = origin
            + idx
                .iter()
                .zip(strides)
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        mask[offset as usize] = true;
        if next_index(&mut idx, dims).is_none() {
            break;
        }
    }
    mask
}

/// The logical index after `idx`, or `None` at the last one.
fn next_index(idx: &mut [usize], dims: &[usize]) -> Option<()> {
    let axis = (0..dims.len()).rev().find(|&a| idx[a] + 1 < dims[a])?;
    idx[axis] += 1;
    for later in idx.iter_mut().skip(axis + 1) {
        *later = 0;
    }
    Some(())
}

/// One layout on all three routes plus the reference.
///
/// `make` turns a distinct number into an element, so two offsets never share a
/// value; `sentinel` is the prefill and is unreachable from `make`.
fn check_layout<T>(
    label: &str,
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    make: impl Fn(f64) -> T,
    sentinel: T,
) where
    T: ElementOpApply + Copy + PartialEq + std::fmt::Debug + Send + Sync + 'static,
{
    let src_origin = origin(dims, src_strides);
    let dst_origin = origin(dims, dst_strides);
    let src_buf: Vec<T> = (0..span(dims, src_strides))
        .map(|i| make(i as f64 + 0.5))
        .collect();
    let dst_init = vec![sentinel; span(dims, dst_strides)];
    let src_view = StridedView::new(&src_buf, dims, src_strides, src_origin).unwrap();

    let mut prod = dst_init.clone();
    copy_conj_into(
        &mut StridedViewMut::new(&mut prod, dims, dst_strides, dst_origin).unwrap(),
        &src_view,
        Exec::serial(),
    )
    .unwrap_or_else(|e| panic!("{label}: production route rejected the copy: {e:?}"));

    let mut map = dst_init.clone();
    let src_conj = src_view.conj();
    copy_into(
        &mut StridedViewMut::new(&mut map, dims, dst_strides, dst_origin).unwrap(),
        &src_conj,
        Exec::serial(),
    )
    .unwrap_or_else(|e| panic!("{label}: element map rejected the copy: {e:?}"));

    let mut engine = dst_init.clone();
    strided_perm::copy_conj_into(
        &mut StridedViewMut::new(&mut engine, dims, dst_strides, dst_origin).unwrap(),
        &src_view,
        Exec::serial(),
    )
    .unwrap_or_else(|e| panic!("{label}: engine rejected the copy: {e:?}"));

    assert_eq!(prod, map, "{label}: production and element map differ");
    assert_eq!(prod, engine, "{label}: production and engine differ");

    // Every logical index against `conj` of the source element.
    let dst_view: StridedView<'_, T> =
        StridedView::new(&prod, dims, dst_strides, dst_origin).unwrap();
    let mut idx = vec![0usize; dims.len()];
    loop {
        assert_eq!(
            dst_view.get(&idx),
            src_view.get(&idx).conj(),
            "{label}: wrong element at {idx:?}"
        );
        if next_index(&mut idx, dims).is_none() {
            break;
        }
    }

    // Nothing outside the layout was written.
    let written = touched(dims, dst_strides, dst_origin, prod.len());
    for (offset, &value) in prod.iter().enumerate() {
        if !written[offset] {
            assert_eq!(value, sentinel, "{label}: wrote outside the layout");
        }
    }
}

fn c64(x: f64) -> Complex64 {
    Complex64::new(x, -2.0 * x)
}

fn c32(x: f64) -> Complex32 {
    Complex32::new(x as f32, -2.0 * x as f32)
}

/// Both complex sentinels are unreachable from `c64`/`c32` (real part `-1`).
fn sentinel_c64() -> Complex64 {
    Complex64::new(-1.0, 7.0)
}

fn sentinel_c32() -> Complex32 {
    Complex32::new(-1.0, 7.0)
}

/// The source strides of a column-major array whose axes are reordered by
/// `perm` into the destination's axis order.
fn permuted_src_strides(dims: &[usize], perm: &[usize]) -> Vec<isize> {
    let cm = strided_view::col_major_strides(dims);
    perm.iter().map(|&p| cm[p]).collect()
}

#[test]
fn complex64_2d_transpose_and_its_neighbours() {
    // Every size takes the engine: all four are under the 24 MiB footprint bound
    // and the destination row length no longer decides the route (#480 fixed the
    // engine's L1 aliasing that the old 4 KiB clause stood in for). `check_layout`
    // asserts the production route and the engine agree on the values.
    for side in [5usize, 255, 256, 257] {
        check_layout(
            &format!("c64 2d {side}^2"),
            &[side, side],
            &[side as isize, 1],
            &[1, side as isize],
            c64,
            sentinel_c64(),
        );
    }
}

#[test]
fn complex64_power_of_two_and_aliasing_shapes() {
    // The issue's 3D and 4D rows, plus its asymmetric 3D row and an odd one.
    for (dims, perm) in [
        (vec![64usize, 64, 64], vec![2usize, 0, 1]),
        (vec![32, 32, 32, 32], vec![3, 1, 0, 2]),
        (vec![100, 2, 100], vec![2, 0, 1]),
        (vec![7, 3, 5], vec![1, 2, 0]),
    ] {
        let src_strides = permuted_src_strides(&dims, &perm);
        let dst_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        let dst_strides = strided_view::col_major_strides(&dst_dims);
        check_layout(
            &format!("c64 {dims:?} perm {perm:?}"),
            &dst_dims,
            &src_strides,
            &dst_strides,
            c64,
            sentinel_c64(),
        );
    }
}

#[test]
fn complex32_is_conj_correct_on_the_engine_route() {
    // Both sizes are engine-side: 300^2 Complex32 is a 2400 B destination row,
    // under the 4 KiB clause this change removed, so `Complex32` never had a
    // 2D map-side size here. The `Complex64` map side is the same element-map
    // code path.
    for side in [5usize, 300] {
        check_layout(
            &format!("c32 {side}^2"),
            &[side, side],
            &[side as isize, 1],
            &[1, side as isize],
            c32,
            sentinel_c32(),
        );
    }
}

#[test]
fn real_types_stay_on_the_map_and_are_conj_correct() {
    for side in [5usize, 256] {
        check_layout(
            &format!("f64 {side}^2"),
            &[side, side],
            &[side as isize, 1],
            &[1, side as isize],
            |x| x,
            -1.0,
        );
    }
    check_layout("f32 4x6", &[6, 4], &[4, 1], &[1, 6], |x| x as f32, -1.0);
}

#[test]
fn strided_and_negative_stride_destinations() {
    // A gapped destination: every other element of the target span.
    check_layout(
        "c64 gapped destination",
        &[5, 4],
        &[4, 1],
        &[2, 10],
        c64,
        sentinel_c64(),
    );
    // A negative-stride destination, written backwards.
    check_layout(
        "c64 negative-stride destination",
        &[4, 5],
        &[5, 1],
        &[-5, -1],
        c64,
        sentinel_c64(),
    );
    // A negative-stride source into a contiguous destination.
    check_layout(
        "c64 negative-stride source",
        &[4, 5],
        &[-1, -4],
        &[1, 4],
        c64,
        sentinel_c64(),
    );
}

#[test]
fn unit_extent_rank_zero_and_empty_views() {
    check_layout(
        "c64 unit extent",
        &[1, 7],
        &[7, 1],
        &[1, 1],
        c64,
        sentinel_c64(),
    );
    check_layout("c64 rank 0", &[], &[], &[], c64, sentinel_c64());
    for dims in [vec![0usize, 3], vec![3, 0, 2]] {
        let strides = strided_view::col_major_strides(&dims);
        let src_buf = vec![Complex64::new(1.0, 2.0); span(&dims, &strides)];
        let mut dst_buf = vec![sentinel_c64(); span(&dims, &strides)];
        let src = StridedView::new(&src_buf, &dims, &strides, 0).unwrap();
        {
            let mut dst = StridedViewMut::new(&mut dst_buf, &dims, &strides, 0).unwrap();
            copy_conj_into(&mut dst, &src, Exec::serial()).unwrap();
        }
        assert!(dst_buf.iter().all(|v| *v == sentinel_c64()));
        let mut dst = StridedViewMut::new(&mut dst_buf, &dims, &strides, 0).unwrap();
        assert!(strided_perm::copy_conj_into(&mut dst, &src, Exec::serial()).is_ok());
    }
}

#[test]
fn non_injective_destination_is_rejected_on_both_routes() {
    // A small rank-3 layout takes the engine; a rank-2 layout over the 24 MiB
    // footprint bound takes the map (its zero-stride axis is what makes it
    // non-injective, so the buffers stay tiny). Both reject the same way and
    // leave the destination untouched.
    for (label, dims, strides) in [
        ("c64 engine route", vec![3usize, 3, 3], vec![1isize, 1, 1]),
        ("c64 map route", vec![1 << 20, 2], vec![0, 1]),
    ] {
        let mut buf = vec![sentinel_c64(); span(&dims, &strides)];
        let src_buf = vec![Complex64::new(1.0, 2.0); span(&dims, &strides)];
        let src = StridedView::new(&src_buf, &dims, &strides, 0).unwrap();
        let mut dst = StridedViewMut::new(&mut buf, &dims, &strides, 0).unwrap();
        let err = copy_conj_into(&mut dst, &src, Exec::serial()).unwrap_err();
        assert!(
            matches!(err, StridedError::NonInjectiveOutputLayout),
            "{label}: {err:?}"
        );
        assert!(
            buf.iter().all(|v| *v == sentinel_c64()),
            "{label}: a rejected copy wrote to the destination"
        );
    }
}

#[test]
fn overflowing_shapes_report_an_error_on_both_routes() {
    // The gate leaves an overflowing count to the map path, which owns that
    // error; the engine may report it as OffsetOverflow or
    // NonInjectiveOutputLayout (it plans before the injectivity check, #263).
    // 2^80 elements: the product overflows `usize`, the stride-0 span does not.
    let dims = [1usize << 40, 1 << 40];
    let strides = [0isize, 0];
    let mut buf = [Complex64::new(1.0, 2.0); 4];
    let src_buf = [Complex64::new(1.0, 2.0); 4];
    let mut dst = StridedViewMut::new(&mut buf, &dims, &strides, 0).unwrap();
    let src = StridedView::new(&src_buf, &dims, &strides, 0).unwrap();
    assert!(copy_conj_into(&mut dst, &src, Exec::serial()).is_err());
    let engine = strided_perm::copy_conj_into(&mut dst, &src, Exec::serial()).unwrap_err();
    assert!(matches!(
        engine,
        StridedError::OffsetOverflow | StridedError::NonInjectiveOutputLayout
    ));
    assert!(buf.iter().all(|v| *v == Complex64::new(1.0, 2.0)));
}
