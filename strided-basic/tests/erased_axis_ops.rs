//! Scan, arg-reduction and fused norm plans against naive references.
//!
//! Every case builds a logical tensor in one of several physical layouts
//! (column major, row major, permuted with negative strides and an offset,
//! a broadcast source axis, and a destination layout differing from the
//! source), so the line, panel and strided fallback kernels are all reached.

use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_basic::{
    ArgReduceOp, ErasedArgReducePlan, ErasedNormPlan, ErasedRawStridedMut, ErasedRawStridedPtr,
    ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedScanPlan, ExecContext, KernelDType,
    KernelStorageElement, NormKind, NormSpec, ScanOp, ScanOptions, StridedError,
};

// ---------------------------------------------------------------- layouts

/// A physical layout of a logical tensor.
#[derive(Clone, Debug)]
struct Layout {
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    len: usize,
}

/// `order` lists axes fastest first; `negative` flips the listed axes;
/// `pad` adds unused elements before the first reachable one. Axes in
/// `broadcast` get stride zero.
fn layout(
    dims: &[usize],
    order: &[usize],
    negative: &[usize],
    broadcast: &[usize],
    pad: usize,
) -> Layout {
    let mut strides = vec![0isize; dims.len()];
    let mut step = 1isize;
    for &axis in order {
        if broadcast.contains(&axis) {
            continue;
        }
        strides[axis] = step;
        step *= dims[axis].max(1) as isize;
    }
    let mut offset = pad as isize;
    for &axis in negative {
        if dims[axis] > 0 {
            offset += strides[axis] * (dims[axis] as isize - 1);
        }
        strides[axis] = -strides[axis];
    }
    let len = pad + step as usize + 2;
    Layout {
        dims: dims.to_vec(),
        strides,
        offset,
        len,
    }
}

fn col_major(dims: &[usize]) -> Layout {
    let order: Vec<usize> = (0..dims.len()).collect();
    layout(dims, &order, &[], &[], 0)
}

fn row_major(dims: &[usize]) -> Layout {
    let order: Vec<usize> = (0..dims.len()).rev().collect();
    layout(dims, &order, &[], &[], 0)
}

/// Representative source/destination layout pairs for a rank-3 shape.
fn layout_pairs(dims: &[usize; 3]) -> Vec<(Layout, Layout)> {
    vec![
        (col_major(dims), col_major(dims)),
        (row_major(dims), row_major(dims)),
        (col_major(dims), row_major(dims)),
        (
            layout(dims, &[1, 2, 0], &[0, 2], &[], 5),
            layout(dims, &[2, 0, 1], &[1], &[], 3),
        ),
        (layout(dims, &[0, 1, 2], &[], &[1], 2), col_major(dims)),
        (
            layout(dims, &[2, 1, 0], &[0, 1, 2], &[], 1),
            layout(dims, &[0, 1, 2], &[0], &[], 4),
        ),
    ]
}

fn total(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Calls `f` with every coordinate of `dims` in column-major order.
fn for_each_coord(dims: &[usize], mut f: impl FnMut(&[usize])) {
    if dims.contains(&0) {
        return;
    }
    let mut coord = vec![0usize; dims.len()];
    loop {
        f(&coord);
        let mut axis = 0;
        loop {
            if axis == dims.len() {
                return;
            }
            coord[axis] += 1;
            if coord[axis] < dims[axis] {
                break;
            }
            coord[axis] = 0;
            axis += 1;
        }
    }
}

fn offset_of(layout: &Layout, coord: &[usize]) -> usize {
    let offset = coord
        .iter()
        .zip(&layout.strides)
        .fold(layout.offset, |acc, (&c, &s)| acc + c as isize * s);
    usize::try_from(offset).unwrap()
}

/// Materializes `value(coord)` in `layout`, filling unreachable slots with `fill`.
fn materialize<T: Copy>(layout: &Layout, fill: T, value: impl Fn(&[usize]) -> T) -> Vec<T> {
    let mut data = vec![fill; layout.len];
    for_each_coord(&layout.dims, |coord| {
        data[offset_of(layout, coord)] = value(coord)
    });
    data
}

fn read<T: Copy>(layout: &Layout, data: &[T], coord: &[usize]) -> T {
    data[offset_of(layout, coord)]
}

/// Deterministic pseudo-random value in `[-1, 1)` for a coordinate.
fn noise(coord: &[usize], salt: u64) -> f64 {
    let mut h = salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for &c in coord {
        h ^= c as u64;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 31;
    }
    (h >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Source coordinates of the line through `outer` (an output coordinate
/// without the axis) at position `k` along `axis`.
fn with_axis(outer: &[usize], axis: usize, k: usize) -> Vec<usize> {
    let mut coord = outer.to_vec();
    coord.insert(axis, k);
    coord
}

fn without_axis(dims: &[usize], axis: usize) -> Vec<usize> {
    let mut out = dims.to_vec();
    out.remove(axis);
    out
}

// ------------------------------------------------------------------- scan

fn reference_scan<T: Copy>(
    layout: &Layout,
    data: &[T],
    axis: usize,
    options: ScanOptions,
    identity: T,
    combine: impl Fn(T, T) -> T,
) -> Vec<(Vec<usize>, T)> {
    let n = layout.dims[axis];
    let outer = without_axis(&layout.dims, axis);
    let mut out = Vec::new();
    for_each_coord(&outer, |outer| {
        let order: Vec<usize> = if options.is_reverse() {
            (0..n).rev().collect()
        } else {
            (0..n).collect()
        };
        let mut acc = identity;
        for k in order {
            let coord = with_axis(outer, axis, k);
            let x = read(layout, data, &coord);
            if options.is_exclusive() {
                out.push((coord, acc));
                acc = combine(acc, x);
            } else {
                acc = combine(acc, x);
                out.push((coord, acc));
            }
        }
    });
    out
}

fn all_scan_options() -> [ScanOptions; 4] {
    [
        ScanOptions::new(),
        ScanOptions::new().exclusive(true),
        ScanOptions::new().reverse(true),
        ScanOptions::new().exclusive(true).reverse(true),
    ]
}

/// Runs a scan through both entries and checks it against the reference
/// with `same` (bitwise for every dtype: the fold order is fixed).
#[allow(clippy::too_many_arguments)]
fn check_scan<T: KernelStorageElement + core::fmt::Debug>(
    dtype: KernelDType,
    op: ScanOp,
    src_layout: &Layout,
    dest_layout: &Layout,
    axis: usize,
    options: ScanOptions,
    value: impl Fn(&[usize]) -> T,
    fill: T,
    identity: T,
    combine: impl Fn(T, T) -> T,
    same: impl Fn(T, T) -> bool,
) {
    let src = materialize(src_layout, fill, &value);
    let plan = ErasedScanPlan::compile(
        dtype,
        op,
        &src_layout.dims,
        &src_layout.strides,
        &dest_layout.strides,
        axis,
        options,
    )
    .unwrap();
    let expected = reference_scan(src_layout, &src, axis, options, identity, &combine);
    let src_ref = ErasedRawStridedRef::from_slice(
        &src,
        &src_layout.dims,
        &src_layout.strides,
        src_layout.offset,
    )
    .unwrap();

    let mut out = vec![fill; dest_layout.len];
    {
        let mut dest = ErasedRawStridedMut::from_slice_mut(
            &mut out,
            &dest_layout.dims,
            &dest_layout.strides,
            dest_layout.offset,
        )
        .unwrap();
        plan.execute(&ExecContext::serial(), &mut dest, &src_ref)
            .unwrap();
    }
    let mut uninit = vec![MaybeUninit::new(fill); dest_layout.len];
    {
        let src_ptr = ErasedRawStridedPtr::from_ref(&src_ref);
        let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(
            &mut uninit,
            &dest_layout.dims,
            &dest_layout.strides,
            dest_layout.offset,
        )
        .unwrap();
        plan.execute_uninit(&ExecContext::ambient(), &mut dest, &src_ptr)
            .unwrap();
    }
    let uninit: Vec<T> = uninit.iter().map(|v| unsafe { v.assume_init() }).collect();
    for (coord, want) in expected {
        for got in [
            read(dest_layout, &out, &coord),
            read(dest_layout, &uninit, &coord),
        ] {
            assert!(
                same(got, want),
                "{op:?} {options:?} axis {axis} at {coord:?}: got {got:?}, want {want:?}\nsrc {src_layout:?}\ndest {dest_layout:?}"
            );
        }
    }
}

fn f64_bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

#[test]
fn scan_f64_matches_reference_across_layouts_axes_and_options() {
    for dims in [[5usize, 4, 3], [1, 70, 2], [3, 1, 67]] {
        for (src_layout, dest_layout) in layout_pairs(&dims) {
            for axis in 0..3 {
                for options in all_scan_options() {
                    check_scan(
                        KernelDType::F64,
                        ScanOp::Sum,
                        &src_layout,
                        &dest_layout,
                        axis,
                        options,
                        |c| noise(c, 1) * 1e3,
                        -7.0,
                        0.0,
                        |a, b| a + b,
                        f64_bits_eq,
                    );
                    check_scan(
                        KernelDType::F64,
                        ScanOp::Product,
                        &src_layout,
                        &dest_layout,
                        axis,
                        options,
                        |c| 1.0 + noise(c, 2) * 0.5,
                        -7.0,
                        1.0,
                        |a, b| a * b,
                        f64_bits_eq,
                    );
                }
            }
        }
    }
}

#[test]
fn scan_panel_mode_covers_partial_panels() {
    // Axis 1 of a column-major (130, 9) matrix is strided while axis 0 is
    // contiguous: the scan runs in panels of 64, 64 and 2 lines.
    let dims = [130usize, 9];
    for (src_layout, dest_layout) in [
        (col_major(&dims), col_major(&dims)),
        (
            layout(&dims, &[0, 1], &[1], &[], 3),
            layout(&dims, &[0, 1], &[1], &[], 1),
        ),
    ] {
        for options in all_scan_options() {
            check_scan(
                KernelDType::F32,
                ScanOp::Sum,
                &src_layout,
                &dest_layout,
                1,
                options,
                |c| noise(c, 3) as f32,
                9.0,
                0.0,
                |a, b| a + b,
                |a: f32, b: f32| a.to_bits() == b.to_bits(),
            );
        }
    }
}

#[test]
fn scan_integer_dtypes_wrap() {
    let dims = [6usize, 3, 2];
    for (src_layout, dest_layout) in layout_pairs(&dims) {
        for axis in 0..3 {
            for options in all_scan_options() {
                check_scan(
                    KernelDType::I32,
                    ScanOp::Sum,
                    &src_layout,
                    &dest_layout,
                    axis,
                    options,
                    |c| i32::MAX - (c[0] + c[1] + c[2]) as i32,
                    0,
                    0,
                    i32::wrapping_add,
                    |a, b| a == b,
                );
                check_scan(
                    KernelDType::I64,
                    ScanOp::Product,
                    &src_layout,
                    &dest_layout,
                    axis,
                    options,
                    |c| 1 << 20 | (c[0] as i64 + 3 * c[1] as i64),
                    0,
                    1,
                    i64::wrapping_mul,
                    |a, b| a == b,
                );
            }
        }
    }
}

#[test]
fn scan_complex_dtypes_match_reference() {
    let dims = [4usize, 3, 2];
    for (src_layout, dest_layout) in layout_pairs(&dims) {
        for axis in 0..3 {
            for options in all_scan_options() {
                check_scan(
                    KernelDType::C64,
                    ScanOp::Product,
                    &src_layout,
                    &dest_layout,
                    axis,
                    options,
                    |c| Complex64::new(noise(c, 4), noise(c, 5)),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    |a, b| a * b,
                    |a: Complex64, b: Complex64| {
                        a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
                    },
                );
                check_scan(
                    KernelDType::C32,
                    ScanOp::Sum,
                    &src_layout,
                    &dest_layout,
                    axis,
                    options,
                    |c| Complex32::new(noise(c, 6) as f32, noise(c, 7) as f32),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    |a, b| a + b,
                    |a: Complex32, b: Complex32| a == b,
                );
            }
        }
    }
}

#[test]
fn scan_propagates_nonfinite_values_like_the_sequential_fold() {
    let values = [1.0, f64::INFINITY, -f64::INFINITY, 2.0, f64::NAN, 3.0];
    let dims = [6usize, 2];
    for options in all_scan_options() {
        for op in [ScanOp::Sum, ScanOp::Product] {
            let (identity, combine): (f64, fn(f64, f64) -> f64) = match op {
                ScanOp::Sum => (0.0, |a, b| a + b),
                _ => (1.0, |a, b| a * b),
            };
            check_scan(
                KernelDType::F64,
                op,
                &col_major(&dims),
                &row_major(&dims),
                0,
                options,
                |c| values[c[0]] * (c[1] as f64 + 1.0),
                0.0,
                identity,
                combine,
                |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b,
            );
        }
    }
}

#[test]
fn scan_empty_axis_and_empty_lines_write_nothing() {
    for dims in [[0usize, 3], [3, 0]] {
        for axis in 0..2 {
            let plan = ErasedScanPlan::compile(
                KernelDType::F64,
                ScanOp::Sum,
                &dims,
                &[1, 3],
                &[1, 3],
                axis,
                ScanOptions::new(),
            )
            .unwrap();
            let src: [f64; 0] = [];
            let mut out: [f64; 0] = [];
            let src = ErasedRawStridedRef::from_slice(&src, &dims, &[1, 3], 0).unwrap();
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &[1, 3], 0).unwrap();
            plan.execute(&ExecContext::serial(), &mut dest, &src)
                .unwrap();
        }
    }
}

#[test]
fn scan_rank_one_and_extent_one_layouts() {
    for dims in [[1usize, 1, 1], [1, 1, 5], [5, 1, 1]] {
        for axis in 0..3 {
            check_scan(
                KernelDType::F64,
                ScanOp::Sum,
                &col_major(&dims),
                &row_major(&dims),
                axis,
                ScanOptions::new().reverse(true),
                |c| (c[0] + 2 * c[1] + 3 * c[2]) as f64,
                0.0,
                0.0,
                |a, b| a + b,
                f64_bits_eq,
            );
        }
    }
}

#[test]
fn scan_rejects_invalid_contracts() {
    let o = ScanOptions::new();
    let err = |r: Result<ErasedScanPlan, StridedError>| r.unwrap_err();
    assert!(matches!(
        err(ErasedScanPlan::compile(
            KernelDType::Bool,
            ScanOp::Sum,
            &[2],
            &[1],
            &[1],
            0,
            o
        )),
        StridedError::UnsupportedDType { .. }
    ));
    assert!(matches!(
        err(ErasedScanPlan::compile(
            KernelDType::F64,
            ScanOp::Sum,
            &[2],
            &[1],
            &[1],
            1,
            o
        )),
        StridedError::InvalidAxis { .. }
    ));
    assert!(matches!(
        err(ErasedScanPlan::compile(
            KernelDType::F64,
            ScanOp::Sum,
            &[2],
            &[1, 1],
            &[1],
            0,
            o
        )),
        StridedError::StrideLengthMismatch
    ));
    assert!(matches!(
        err(ErasedScanPlan::compile(
            KernelDType::F64,
            ScanOp::Sum,
            &[2, 2],
            &[1, 2],
            &[1, 1],
            0,
            o
        )),
        StridedError::NonInjectiveOutputLayout
    ));
    assert!(matches!(
        err(ErasedScanPlan::compile(
            KernelDType::F64,
            ScanOp::Sum,
            &[3, 2],
            &[1, isize::MAX],
            &[1, 3],
            0,
            o
        )),
        StridedError::OffsetOverflow
    ));

    let plan =
        ErasedScanPlan::compile(KernelDType::F64, ScanOp::Sum, &[2], &[1], &[1], 0, o).unwrap();
    let src = [1.0f64, 2.0];
    let src32 = [1.0f32, 2.0];
    let src_ref = ErasedRawStridedRef::from_slice(&src, &[2], &[1], 0).unwrap();
    let mut out = [5.0f64; 4];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[2], &[2], 0).unwrap();
    assert!(matches!(
        plan.execute(&ExecContext::serial(), &mut dest, &src_ref),
        Err(StridedError::PlanLayoutMismatch)
    ));
    let src32_ref = ErasedRawStridedRef::from_slice(&src32, &[2], &[1], 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[2], &[1], 0).unwrap();
    assert!(matches!(
        plan.execute(&ExecContext::serial(), &mut dest, &src32_ref),
        Err(StridedError::DTypeMismatch { .. })
    ));
    assert_eq!(out, [5.0; 4]);

    // The uninit entry rejects a source inside the destination allocation.
    let mut storage = [MaybeUninit::new(1.0f64); 4];
    let src_ptr = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::F64,
            core::ptr::NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap(),
            32,
            &[2],
            &[1],
            0,
        )
    }
    .unwrap();
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, &[2], &[1], 2).unwrap();
    assert!(matches!(
        plan.execute_uninit(&ExecContext::serial(), &mut dest, &src_ptr),
        Err(StridedError::OverlappingInputOutput { input: 0 })
    ));
}

// -------------------------------------------------------- arg-reduction

/// Reference: first NaN, else the lowest index of the strict extremum.
fn reference_arg(keys: &[f64], max: bool) -> usize {
    if let Some(index) = keys.iter().position(|k| k.is_nan()) {
        return index;
    }
    let mut best = 0;
    for (index, &key) in keys.iter().enumerate() {
        if (max && key > keys[best]) || (!max && key < keys[best]) {
            best = index;
        }
    }
    best
}

fn op_max(op: ArgReduceOp) -> bool {
    matches!(op, ArgReduceOp::Max | ArgReduceOp::MaxAbs)
}

/// Runs an arg-reduction through both entries and both index dtypes.
fn check_arg<T: KernelStorageElement>(
    dtype: KernelDType,
    op: ArgReduceOp,
    src_layout: &Layout,
    axis: usize,
    dest_order_reversed: bool,
    value: impl Fn(&[usize]) -> T,
    fill: T,
    key: impl Fn(T) -> f64,
) {
    let src = materialize(src_layout, fill, &value);
    let out_dims = without_axis(&src_layout.dims, axis);
    let dest_layout = if dest_order_reversed {
        row_major(&out_dims)
    } else {
        layout(
            &out_dims,
            &(0..out_dims.len()).collect::<Vec<_>>(),
            &[0],
            &[],
            2,
        )
    };
    let src_ref = ErasedRawStridedRef::from_slice(
        &src,
        &src_layout.dims,
        &src_layout.strides,
        src_layout.offset,
    )
    .unwrap();
    let n = src_layout.dims[axis];
    let mut expected = Vec::new();
    for_each_coord(&out_dims, |outer| {
        let keys: Vec<f64> = (0..n)
            .map(|k| key(read(src_layout, &src, &with_axis(outer, axis, k))))
            .collect();
        expected.push((outer.to_vec(), reference_arg(&keys, op_max(op))));
    });

    for index_dtype in [KernelDType::I64, KernelDType::I32] {
        let plan = ErasedArgReducePlan::compile(
            dtype,
            index_dtype,
            op,
            &src_layout.dims,
            &src_layout.strides,
            &dest_layout.dims,
            &dest_layout.strides,
            axis,
        )
        .unwrap();
        let got: Vec<Vec<i64>> = if index_dtype == KernelDType::I64 {
            let mut out = vec![-1i64; dest_layout.len];
            let mut dest = ErasedRawStridedMut::from_slice_mut(
                &mut out,
                &dest_layout.dims,
                &dest_layout.strides,
                dest_layout.offset,
            )
            .unwrap();
            plan.execute(&ExecContext::serial(), &mut dest, &src_ref)
                .unwrap();
            let mut uninit = vec![MaybeUninit::new(-1i64); dest_layout.len];
            let src_ptr = ErasedRawStridedPtr::from_ref(&src_ref);
            let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(
                &mut uninit,
                &dest_layout.dims,
                &dest_layout.strides,
                dest_layout.offset,
            )
            .unwrap();
            plan.execute_uninit(&ExecContext::ambient(), &mut dest, &src_ptr)
                .unwrap();
            let uninit: Vec<i64> = uninit.iter().map(|v| unsafe { v.assume_init() }).collect();
            vec![out, uninit]
        } else {
            let mut out = vec![-1i32; dest_layout.len];
            let mut dest = ErasedRawStridedMut::from_slice_mut(
                &mut out,
                &dest_layout.dims,
                &dest_layout.strides,
                dest_layout.offset,
            )
            .unwrap();
            plan.execute(&ExecContext::serial(), &mut dest, &src_ref)
                .unwrap();
            vec![out.into_iter().map(i64::from).collect()]
        };
        for out in got {
            for (coord, want) in &expected {
                let got = read(&dest_layout, &out, coord);
                assert_eq!(
                    got, *want as i64,
                    "{op:?} axis {axis} at {coord:?}\nsrc {src_layout:?}"
                );
            }
        }
    }
}

/// Values with many ties, signed zeros, infinities and (optionally) NaN.
fn tricky(coord: &[usize], salt: u64, nan: bool) -> f64 {
    let h = ((noise(coord, salt) + 1.0) * 1000.0) as u64;
    match h % 13 {
        0 => 0.0,
        1 => -0.0,
        2 => f64::INFINITY,
        3 => f64::NEG_INFINITY,
        4 if nan => f64::NAN,
        5 | 6 => 2.5,
        7 => -2.5,
        _ => (h % 7) as f64 - 3.0,
    }
}

#[test]
fn argreduce_real_matches_reference_across_layouts() {
    let ops = [
        ArgReduceOp::Max,
        ArgReduceOp::Min,
        ArgReduceOp::MaxAbs,
        ArgReduceOp::MinAbs,
    ];
    for dims in [[5usize, 4, 3], [2, 70, 3], [67, 1, 2]] {
        for (src_layout, _) in layout_pairs(&dims) {
            for axis in 0..3 {
                for op in ops {
                    for (salt, nan) in [(11, false), (12, true)] {
                        let abs = matches!(op, ArgReduceOp::MaxAbs | ArgReduceOp::MinAbs);
                        let key = move |x: f64| if abs { x.abs() } else { x };
                        check_arg(
                            KernelDType::F64,
                            op,
                            &src_layout,
                            axis,
                            salt == 11,
                            |c| tricky(c, salt, nan),
                            0.0,
                            key,
                        );
                        check_arg(
                            KernelDType::F32,
                            op,
                            &src_layout,
                            axis,
                            salt == 12,
                            |c| tricky(c, salt, nan) as f32,
                            0.0,
                            move |x: f32| key(x as f64),
                        );
                    }
                    check_arg(
                        KernelDType::I32,
                        op,
                        &src_layout,
                        axis,
                        false,
                        |c| match (c[0] + c[1] + c[2]) % 5 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            other => other as i32 - 2,
                        },
                        0,
                        move |x: i32| {
                            if matches!(op, ArgReduceOp::MaxAbs | ArgReduceOp::MinAbs) {
                                x.unsigned_abs() as f64
                            } else {
                                x as f64
                            }
                        },
                    );
                    check_arg(
                        KernelDType::I64,
                        op,
                        &src_layout,
                        axis,
                        true,
                        |c| (c[0] as i64 * 7 + c[1] as i64 * 3 + c[2] as i64) % 5 - 2,
                        0,
                        move |x: i64| {
                            if matches!(op, ArgReduceOp::MaxAbs | ArgReduceOp::MinAbs) {
                                x.unsigned_abs() as f64
                            } else {
                                x as f64
                            }
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn argreduce_complex_magnitude_is_overflow_safe_and_nan_aware() {
    // |1e300 + 1e300 i| and |1e300| differ although their squared moduli
    // both overflow to infinity.
    let values = [
        Complex64::new(1e300, 0.0),
        Complex64::new(1e300, 1e300),
        Complex64::new(-1e300, -1e300),
        Complex64::new(0.0, -0.0),
    ];
    let dims = [4usize];
    let check = |op, want: i64, values: &[Complex64]| {
        let plan = ErasedArgReducePlan::compile(
            KernelDType::C64,
            KernelDType::I64,
            op,
            &dims,
            &[1],
            &[],
            &[],
            0,
        )
        .unwrap();
        let src = ErasedRawStridedRef::from_slice(values, &dims, &[1], 0).unwrap();
        let mut out = [-1i64];
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[], &[], 0).unwrap();
        plan.execute(&ExecContext::serial(), &mut dest, &src)
            .unwrap();
        assert_eq!(out[0], want, "{op:?} {values:?}");
    };
    check(ArgReduceOp::MaxAbs, 1, &values);
    check(ArgReduceOp::MinAbs, 3, &values);
    // A NaN component wins even next to an infinite one.
    let with_nan = [
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(f64::INFINITY, f64::NAN),
        Complex64::new(f64::NAN, 0.0),
    ];
    check(ArgReduceOp::MaxAbs, 2, &with_nan);
    check(ArgReduceOp::MinAbs, 2, &with_nan);

    for dims in [[5usize, 4, 3], [2, 70, 3]] {
        for (src_layout, _) in layout_pairs(&dims) {
            for axis in 0..3 {
                for op in [ArgReduceOp::MaxAbs, ArgReduceOp::MinAbs] {
                    let key = |z: Complex32| {
                        if z.re.is_nan() || z.im.is_nan() {
                            f64::NAN
                        } else {
                            z.re.hypot(z.im) as f64
                        }
                    };
                    check_arg(
                        KernelDType::C32,
                        op,
                        &src_layout,
                        axis,
                        axis == 1,
                        // Small integer components give exact magnitude ties.
                        |c| Complex32::new(tricky(c, 21, true) as f32, ((c[0] + c[2]) % 3) as f32),
                        Complex32::new(0.0, 0.0),
                        key,
                    );
                }
            }
        }
    }
}

#[test]
fn argreduce_rejects_invalid_contracts() {
    let compile =
        |dtype, index, op, src: &[usize], ss: &[isize], dd: &[usize], ds: &[isize], axis| {
            ErasedArgReducePlan::compile(dtype, index, op, src, ss, dd, ds, axis).map(|_| ())
        };
    use ArgReduceOp::*;
    use KernelDType::*;
    assert!(matches!(
        compile(Bool, I64, Max, &[2], &[1], &[], &[], 0),
        Err(StridedError::UnsupportedDType { .. })
    ));
    assert!(matches!(
        compile(C32, I64, Min, &[2], &[1], &[], &[], 0),
        Err(StridedError::UnsupportedOp { .. })
    ));
    assert!(matches!(
        compile(F64, F64, Max, &[2], &[1], &[], &[], 0),
        Err(StridedError::UnsupportedDType { .. })
    ));
    assert!(matches!(
        compile(F64, I64, Max, &[2], &[1, 1], &[], &[], 0),
        Err(StridedError::StrideLengthMismatch)
    ));
    assert!(matches!(
        compile(F64, I64, Max, &[2], &[1], &[], &[], 1),
        Err(StridedError::InvalidAxis { .. })
    ));
    assert!(matches!(
        compile(F64, I64, Max, &[2, 3], &[1, 2], &[2], &[1], 0),
        Err(StridedError::ShapeMismatch(..))
    ));
    assert!(matches!(
        compile(F64, I64, Max, &[0, 3], &[1, 1], &[3], &[1], 0),
        Err(StridedError::UnsupportedOp { .. })
    ));
    assert!(matches!(
        compile(F64, I64, Max, &[2, 3, 2], &[1, 2, 6], &[3, 2], &[1, 1], 0),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
    // An i32 index cannot address the last element of a 2^31 + 1 axis.
    let long = (1usize << 31) + 1;
    assert!(matches!(
        compile(F64, I32, Max, &[long], &[0], &[], &[], 0),
        Err(StridedError::OffsetOverflow)
    ));
    assert!(compile(F64, I64, Max, &[long], &[0], &[], &[], 0).is_ok());
    // An empty set of lines is valid and writes nothing.
    let plan =
        ErasedArgReducePlan::compile(F64, I64, Max, &[3, 0], &[1, 3], &[0], &[1], 0).unwrap();
    let src: [f64; 0] = [];
    let mut out: [i64; 0] = [];
    let src = ErasedRawStridedRef::from_slice(&src, &[3, 0], &[1, 3], 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[0], &[1], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &src)
        .unwrap();

    let plan = ErasedArgReducePlan::compile(F64, I64, Max, &[2], &[1], &[], &[], 0).unwrap();
    let src = [1.0f64, 2.0];
    let src_ref = ErasedRawStridedRef::from_slice(&src, &[2], &[1], 0).unwrap();
    let mut wrong_index = [0i32];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut wrong_index, &[], &[], 0).unwrap();
    assert!(matches!(
        plan.execute(&ExecContext::serial(), &mut dest, &src_ref),
        Err(StridedError::DTypeMismatch { .. })
    ));
    let reversed = ErasedRawStridedRef::from_slice(&src, &[2], &[-1], 1).unwrap();
    let mut out = [7i64];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[], &[], 0).unwrap();
    assert!(matches!(
        plan.execute(&ExecContext::serial(), &mut dest, &reversed),
        Err(StridedError::PlanLayoutMismatch)
    ));
    assert_eq!(out, [7]);
    let mut uninit = [MaybeUninit::<i64>::uninit()];
    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut uninit, &[], &[], 0).unwrap();
    let reversed_ptr = ErasedRawStridedPtr::from_ref(&reversed);
    assert!(matches!(
        plan.execute_uninit(&ExecContext::serial(), &mut dest, &reversed_ptr),
        Err(StridedError::PlanLayoutMismatch)
    ));
}

// ------------------------------------------------------------------- norm

/// Reference in `f64` with the documented shifted two-pass formula.
fn reference_norm(
    kind: NormKind,
    xs: &[f64],
    eps: f64,
    w: Option<&[f64]>,
    b: Option<&[f64]>,
) -> Vec<f64> {
    let n = xs.len() as f64;
    let (shift, mean) = match kind {
        NormKind::Layer => (xs[0], xs.iter().map(|x| x - xs[0]).sum::<f64>() / n),
        _ => (0.0, 0.0),
    };
    let dev = |x: f64| (x - shift) - mean;
    let var = xs.iter().map(|&x| dev(x) * dev(x)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    xs.iter()
        .enumerate()
        .map(|(k, &x)| {
            let mut y = dev(x) * inv;
            if let Some(w) = w {
                y *= w[k];
            }
            if let Some(b) = b {
                y += b[k];
            }
            y
        })
        .collect()
}

trait NormElem: KernelStorageElement + core::fmt::Debug {
    const KIND: KernelDType;
    const TOL: f64;
    fn from(x: f64) -> Self;
    fn to(self) -> f64;
}
impl NormElem for f32 {
    const KIND: KernelDType = KernelDType::F32;
    const TOL: f64 = 2e-5;
    fn from(x: f64) -> Self {
        x as f32
    }
    fn to(self) -> f64 {
        self as f64
    }
}
impl NormElem for f64 {
    const KIND: KernelDType = KernelDType::F64;
    const TOL: f64 = 1e-12;
    fn from(x: f64) -> Self {
        x
    }
    fn to(self) -> f64 {
        self
    }
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    if want.is_infinite() {
        return got == want;
    }
    (got - want).abs() <= tol * (1.0 + want.abs())
}

#[allow(clippy::too_many_arguments)]
fn check_norm<T: NormElem>(
    kind: NormKind,
    src_layout: &Layout,
    dest_layout: &Layout,
    axis: usize,
    eps: f64,
    affine: (Option<isize>, Option<isize>),
    value: impl Fn(&[usize]) -> f64,
) {
    let n = src_layout.dims[axis];
    let src = materialize(src_layout, T::from(-9.0), |c| T::from(value(c)));
    let mut spec = match kind {
        NormKind::Layer => NormSpec::layer_norm(eps),
        _ => NormSpec::rms_norm(eps),
    };
    // Affine vectors: stride s, with a negative stride starting at the end.
    let vector = |stride: isize, salt: u64| -> (Vec<T>, isize, Vec<f64>) {
        let len = n * stride.unsigned_abs() + 3;
        let logical: Vec<f64> = (0..n).map(|k| 1.0 + noise(&[k], salt)).collect();
        let offset = if stride < 0 {
            (n.max(1) - 1) as isize * -stride + 1
        } else {
            2
        };
        let mut data = vec![T::from(f64::NAN); len];
        for (k, &v) in logical.iter().enumerate() {
            data[(offset + k as isize * stride) as usize] = T::from(v);
        }
        let logical = logical.iter().map(|&v| T::from(v).to()).collect();
        (data, offset, logical)
    };
    let weight = affine.0.map(|s| {
        spec = spec.with_weight(s);
        (vector(s, 31), s)
    });
    let bias = affine.1.map(|s| {
        spec = spec.with_bias(s);
        (vector(s, 32), s)
    });
    let plan = ErasedNormPlan::compile(
        T::KIND,
        spec,
        &src_layout.dims,
        &src_layout.strides,
        &dest_layout.strides,
        axis,
    )
    .unwrap();
    let dims_n = [n];
    let w_strides = weight.as_ref().map(|(_, s)| [*s]);
    let b_strides = bias.as_ref().map(|(_, s)| [*s]);
    let w_ref = weight.as_ref().map(|((data, off, _), _)| {
        ErasedRawStridedRef::from_slice(data, &dims_n, w_strides.as_ref().unwrap(), *off).unwrap()
    });
    let b_ref = bias.as_ref().map(|((data, off, _), _)| {
        ErasedRawStridedRef::from_slice(data, &dims_n, b_strides.as_ref().unwrap(), *off).unwrap()
    });
    let src_ref = ErasedRawStridedRef::from_slice(
        &src,
        &src_layout.dims,
        &src_layout.strides,
        src_layout.offset,
    )
    .unwrap();

    let mut out = vec![T::from(-5.0); dest_layout.len];
    {
        let mut dest = ErasedRawStridedMut::from_slice_mut(
            &mut out,
            &dest_layout.dims,
            &dest_layout.strides,
            dest_layout.offset,
        )
        .unwrap();
        plan.execute(
            &ExecContext::serial(),
            &mut dest,
            &src_ref,
            w_ref.as_ref(),
            b_ref.as_ref(),
        )
        .unwrap();
    }
    let mut uninit = vec![MaybeUninit::new(T::from(-5.0)); dest_layout.len];
    {
        let src_ptr = ErasedRawStridedPtr::from_ref(&src_ref);
        let w_ptr = w_ref.as_ref().map(ErasedRawStridedPtr::from_ref);
        let b_ptr = b_ref.as_ref().map(ErasedRawStridedPtr::from_ref);
        let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(
            &mut uninit,
            &dest_layout.dims,
            &dest_layout.strides,
            dest_layout.offset,
        )
        .unwrap();
        plan.execute_uninit(
            &ExecContext::ambient(),
            &mut dest,
            &src_ptr,
            w_ptr.as_ref(),
            b_ptr.as_ref(),
        )
        .unwrap();
    }
    let uninit: Vec<T> = uninit.iter().map(|v| unsafe { v.assume_init() }).collect();
    // Untouched slots keep their fill.
    let mut reachable = vec![false; dest_layout.len];
    for_each_coord(&dest_layout.dims, |c| {
        reachable[offset_of(dest_layout, c)] = true
    });
    for (slot, &hit) in reachable.iter().enumerate() {
        if !hit {
            assert_eq!(out[slot].to(), -5.0);
        }
    }

    let outer = without_axis(&src_layout.dims, axis);
    for_each_coord(&outer, |outer| {
        let xs: Vec<f64> = (0..n)
            .map(|k| read(src_layout, &src, &with_axis(outer, axis, k)).to())
            .collect();
        let want = reference_norm(
            kind,
            &xs,
            T::from(eps).to(),
            weight.as_ref().map(|((_, _, l), _)| l.as_slice()),
            bias.as_ref().map(|((_, _, l), _)| l.as_slice()),
        );
        for (k, &want) in want.iter().enumerate() {
            let coord = with_axis(outer, axis, k);
            for got in [
                read(dest_layout, &out, &coord),
                read(dest_layout, &uninit, &coord),
            ] {
                assert!(
                    close(got.to(), want, T::TOL),
                    "{kind:?} axis {axis} {coord:?}: got {got:?}, want {want}\nsrc {src_layout:?}\ndest {dest_layout:?}"
                );
            }
        }
    });
}

#[test]
fn norm_matches_reference_across_layouts_kinds_and_affine() {
    let affines = [
        (None, None),
        (Some(1), None),
        (None, Some(-2)),
        (Some(-1), Some(3)),
    ];
    for dims in [[9usize, 4, 3], [3, 70, 2], [66, 1, 2]] {
        for (src_layout, dest_layout) in layout_pairs(&dims) {
            for axis in 0..3 {
                for kind in [NormKind::Layer, NormKind::Rms] {
                    for affine in affines {
                        check_norm::<f64>(
                            kind,
                            &src_layout,
                            &dest_layout,
                            axis,
                            1e-5,
                            affine,
                            |c| 3.0 + noise(c, 41) * 2.0,
                        );
                        check_norm::<f32>(
                            kind,
                            &src_layout,
                            &dest_layout,
                            axis,
                            1e-6,
                            affine,
                            |c| noise(c, 42),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn norm_feature_first_decode_shapes() {
    // The #2006 workload: (d, length, batch) normalized over d.
    for (d, len, batch) in [(1024usize, 8usize, 1usize), (64, 3, 2), (17, 1, 1)] {
        let dims = [d, len, batch];
        let l = col_major(&dims);
        for kind in [NormKind::Layer, NormKind::Rms] {
            check_norm::<f32>(kind, &l, &l, 0, 1e-5, (Some(1), Some(1)), |c| {
                noise(c, 43) * 4.0
            });
        }
    }
}

#[test]
fn norm_zero_variance_eps_and_nonfinite_lines() {
    let dims = [5usize, 6];
    let l = col_major(&dims);
    let r = row_major(&dims);
    // Line 0: constant (zero variance); line 1: zeros; line 2: NaN; line 3:
    // +inf; line 4: huge finite values; line 5: ordinary.
    let value = |c: &[usize]| match c[1] {
        0 => 4.0,
        1 => 0.0,
        2 => {
            if c[0] == 3 {
                f64::NAN
            } else {
                1.0
            }
        }
        3 => {
            if c[0] == 1 {
                f64::INFINITY
            } else {
                1.0
            }
        }
        4 => 1e200 * (c[0] as f64 + 1.0),
        _ => c[0] as f64,
    };
    for (src, dest) in [(&l, &l), (&r, &r), (&l, &r)] {
        for kind in [NormKind::Layer, NormKind::Rms] {
            for eps in [0.0, 1e-5] {
                for affine in [(None, None), (Some(1), Some(1))] {
                    check_norm::<f64>(kind, src, dest, 0, eps, affine, value);
                }
            }
        }
    }

    // Spot-check the documented zero-variance results directly.
    let x = [4.0f64; 3];
    let spec = NormSpec::layer_norm(1e-5);
    let plan = ErasedNormPlan::compile(KernelDType::F64, spec, &[3], &[1], &[1], 0).unwrap();
    let x_ref = ErasedRawStridedRef::from_slice(&x, &[3], &[1], 0).unwrap();
    let mut y = [1.0f64; 3];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut y, &[3], &[1], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &x_ref, None, None)
        .unwrap();
    assert_eq!(y, [0.0; 3]);
    let plan = ErasedNormPlan::compile(
        KernelDType::F64,
        NormSpec::layer_norm(0.0),
        &[3],
        &[1],
        &[1],
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut y, &[3], &[1], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &x_ref, None, None)
        .unwrap();
    assert!(y.iter().all(|v| v.is_nan()));
}

#[test]
fn norm_empty_axis_and_empty_lines_write_nothing() {
    for dims in [[0usize, 3], [3, 0]] {
        for axis in 0..2 {
            let spec = NormSpec::layer_norm(1e-5).with_weight(1);
            let plan =
                ErasedNormPlan::compile(KernelDType::F32, spec, &dims, &[1, 3], &[1, 3], axis)
                    .unwrap();
            let src: [f32; 0] = [];
            let mut out: [f32; 0] = [];
            let w = [0.0f32; 3];
            let src = ErasedRawStridedRef::from_slice(&src, &dims, &[1, 3], 0).unwrap();
            let w = ErasedRawStridedRef::from_slice(&w[..dims[axis]], &dims[axis..=axis], &[1], 0)
                .unwrap();
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &[1, 3], 0).unwrap();
            plan.execute(&ExecContext::serial(), &mut dest, &src, Some(&w), None)
                .unwrap();
        }
    }
}

#[test]
fn norm_rejects_invalid_contracts() {
    let spec = NormSpec::layer_norm(1e-5);
    let compile = |dtype, spec, dims: &[usize], ss: &[isize], ds: &[isize], axis| {
        ErasedNormPlan::compile(dtype, spec, dims, ss, ds, axis).map(|_| ())
    };
    for dtype in [KernelDType::I32, KernelDType::C32, KernelDType::Bool] {
        assert!(matches!(
            compile(dtype, spec, &[2], &[1], &[1], 0),
            Err(StridedError::UnsupportedDType { .. })
        ));
    }
    for eps in [-1e-5, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            compile(
                KernelDType::F64,
                NormSpec::rms_norm(eps),
                &[2],
                &[1],
                &[1],
                0
            ),
            Err(StridedError::UnsupportedOp { .. })
        ));
    }
    assert!(matches!(
        compile(KernelDType::F64, spec, &[2], &[1], &[1, 1], 0),
        Err(StridedError::StrideLengthMismatch)
    ));
    assert!(matches!(
        compile(KernelDType::F64, spec, &[2], &[1], &[1], 1),
        Err(StridedError::InvalidAxis { .. })
    ));
    assert!(matches!(
        compile(KernelDType::F64, spec, &[2, 2], &[1, 2], &[0, 1], 0),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
    assert!(matches!(
        compile(
            KernelDType::F64,
            spec.with_weight(isize::MAX),
            &[3],
            &[1],
            &[1],
            0
        ),
        Err(StridedError::OffsetOverflow)
    ));

    let plan = ErasedNormPlan::compile(KernelDType::F64, spec.with_weight(1), &[2], &[1], &[1], 0)
        .unwrap();
    let x = [1.0f64, 2.0];
    let w32 = [1.0f32, 1.0];
    let w3 = [1.0f64; 3];
    let x_ref = ErasedRawStridedRef::from_slice(&x, &[2], &[1], 0).unwrap();
    let w32_ref = ErasedRawStridedRef::from_slice(&w32, &[2], &[1], 0).unwrap();
    let w3_ref = ErasedRawStridedRef::from_slice(&w3, &[3], &[1], 0).unwrap();
    let mut y = [7.0f64; 2];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut y, &[2], &[1], 0).unwrap();
    let ctx = ExecContext::serial();
    assert!(matches!(
        plan.execute(&ctx, &mut dest, &x_ref, None, None),
        Err(StridedError::PlanLayoutMismatch)
    ));
    assert!(matches!(
        plan.execute(&ctx, &mut dest, &x_ref, Some(&x_ref), Some(&x_ref)),
        Err(StridedError::PlanLayoutMismatch)
    ));
    assert!(matches!(
        plan.execute(&ctx, &mut dest, &x_ref, Some(&w32_ref), None),
        Err(StridedError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        plan.execute(&ctx, &mut dest, &x_ref, Some(&w3_ref), None),
        Err(StridedError::PlanLayoutMismatch)
    ));
    assert_eq!(y, [7.0; 2]);

    // The uninit entry names the overlapping input.
    let mut storage = [MaybeUninit::new(1.0f64); 4];
    let inside = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::F64,
            core::ptr::NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap(),
            32,
            &[2],
            &[1],
            2,
        )
    }
    .unwrap();
    let x_ptr = ErasedRawStridedPtr::from_ref(&x_ref);
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, &[2], &[1], 0).unwrap();
    assert!(matches!(
        plan.execute_uninit(&ctx, &mut dest, &x_ptr, Some(&inside), None),
        Err(StridedError::OverlappingInputOutput { input: 1 })
    ));
    let plan_b = ErasedNormPlan::compile(
        KernelDType::F64,
        NormSpec::rms_norm(0.0).with_bias(1),
        &[2],
        &[1],
        &[1],
        0,
    )
    .unwrap();
    assert!(matches!(
        plan_b.execute_uninit(&ctx, &mut dest, &x_ptr, None, Some(&inside)),
        Err(StridedError::OverlappingInputOutput { input: 2 })
    ));
}
