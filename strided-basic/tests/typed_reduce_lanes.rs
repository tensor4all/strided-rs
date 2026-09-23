//! The typed `reduce` and `reduce_axis` fold with independent lanes, sweep
//! contiguous columns and split outputs across threads. These tests check
//! every path against a naive reference on exact integer data (so any order
//! is exact) and on floats with NaN, serially and in parallel.

use std::num::NonZeroUsize;
use strided_basic::{reduce, reduce_axis, with_execution_policy, ExecutionPolicy, StridedView};

fn policies() -> [ExecutionPolicy; 2] {
    [
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Rayon {
            max_threads: NonZeroUsize::new(4).unwrap(),
        },
    ]
}

/// A layout over `data`: dims, strides and offset.
struct Layout {
    name: &'static str,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    len: usize,
}

fn col_major(name: &'static str, dims: &[usize]) -> Layout {
    let mut strides = Vec::new();
    let mut step = 1isize;
    for &d in dims {
        strides.push(step);
        step *= d as isize;
    }
    Layout {
        name,
        dims: dims.to_vec(),
        strides,
        offset: 0,
        len: dims.iter().product(),
    }
}

fn layouts() -> Vec<Layout> {
    let mut out = vec![
        col_major("col 1x1", &[1, 1]),
        col_major("col 7x5", &[7, 5]),
        col_major("col 33x70", &[33, 70]),
        col_major("col 8200x5 crosses the sweep block", &[8200, 5]),
        col_major("col 17x3000", &[17, 3000]),
        col_major("col rank3 19x6x7", &[19, 6, 7]),
        col_major("col rank3 3x40x300", &[3, 40, 300]),
    ];
    // Row-major 300x131: the second axis is contiguous.
    out.push(Layout {
        name: "row 300x131",
        dims: vec![300, 131],
        strides: vec![131, 1],
        offset: 0,
        len: 300 * 131,
    });
    // Padded columns: leading stride 1, column stride larger than rows.
    out.push(Layout {
        name: "padded 40x900",
        dims: vec![40, 900],
        strides: vec![1, 45],
        offset: 3,
        len: 3 + 45 * 900,
    });
    // Negative strides on both axes.
    out.push(Layout {
        name: "negative 50x700",
        dims: vec![50, 700],
        strides: vec![-1, -50],
        offset: 50 * 700 - 1,
        len: 50 * 700,
    });
    // Broadcast along each axis.
    out.push(Layout {
        name: "broadcast rows 64x600",
        dims: vec![64, 600],
        strides: vec![1, 0],
        offset: 0,
        len: 64,
    });
    out.push(Layout {
        name: "broadcast cols 64x600",
        dims: vec![64, 600],
        strides: vec![0, 1],
        offset: 0,
        len: 600,
    });
    out
}

fn offset_of(layout: &Layout, index: &[usize]) -> usize {
    let mut off = layout.offset as isize;
    for (&i, &s) in index.iter().zip(&layout.strides) {
        off += i as isize * s;
    }
    off as usize
}

/// Visits every multi-index in column-major order.
fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    if dims.contains(&0) {
        return;
    }
    let mut index = vec![0usize; dims.len()];
    loop {
        f(&index);
        let mut axis = 0;
        loop {
            if axis == dims.len() {
                return;
            }
            index[axis] += 1;
            if index[axis] < dims[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
    }
}

fn int_data(len: usize) -> Vec<i64> {
    (0..len)
        .map(|i| ((i as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)) >> 7)
        .collect()
}

/// Naive per-output wrapping sum of `3 * x`, seeded with `init`.
fn naive_axis_i64(layout: &Layout, data: &[i64], axis: usize, init: i64) -> Vec<i64> {
    let out_dims: Vec<usize> = (0..layout.dims.len())
        .filter(|&a| a != axis)
        .map(|a| layout.dims[a])
        .collect();
    let mut out = Vec::new();
    for_each_index(&out_dims, |kept| {
        let mut acc = init;
        for k in 0..layout.dims[axis] {
            let mut index = kept.to_vec();
            index.insert(axis, k);
            acc = acc.wrapping_add(data[offset_of(layout, &index)].wrapping_mul(3));
        }
        out.push(acc);
    });
    if out_dims.is_empty() {
        out.truncate(1);
    }
    out
}

#[test]
fn typed_reduce_axis_matches_naive_integer_reference() {
    for layout in layouts() {
        let data = int_data(layout.len);
        let view =
            StridedView::<i64>::new(&data, &layout.dims, &layout.strides, layout.offset as isize)
                .unwrap();
        for axis in 0..layout.dims.len() {
            let want = naive_axis_i64(&layout, &data, axis, 11);
            for policy in policies() {
                let got = with_execution_policy(policy, || {
                    reduce_axis(&view, axis, |x| x.wrapping_mul(3), i64::wrapping_add, 11).unwrap()
                });
                assert_eq!(
                    got.data(),
                    &want[..],
                    "{} axis {axis} {policy:?}",
                    layout.name
                );
            }
        }
    }
}

#[test]
fn typed_reduce_full_matches_naive_integer_reference() {
    for layout in layouts() {
        let data = int_data(layout.len);
        let view =
            StridedView::<i64>::new(&data, &layout.dims, &layout.strides, layout.offset as isize)
                .unwrap();
        let mut want = 5i64;
        for_each_index(&layout.dims, |index| {
            want = want.wrapping_add(data[offset_of(&layout, index)].wrapping_mul(3));
        });
        for policy in policies() {
            let got = with_execution_policy(policy, || {
                reduce(&view, |x| x.wrapping_mul(3), i64::wrapping_add, 0).unwrap()
            });
            // `init` must be an identity once threads may split the work.
            assert_eq!(got.wrapping_add(5), want, "{} {policy:?}", layout.name);
        }
    }
}

#[test]
fn typed_contiguous_fold_seeds_init_once() {
    // Serial folds combine `init` exactly once for every length around the
    // lane count and the vector width.
    for len in [0usize, 1, 15, 16, 31, 32, 33, 47, 48, 100, 1001] {
        let data = int_data(len);
        let view = StridedView::<i64>::new(&data, &[len], &[1], 0).unwrap();
        let want = data.iter().fold(1000i64, |acc, &x| acc.wrapping_add(x));
        let got = with_execution_policy(ExecutionPolicy::Sequential, || {
            reduce(&view, |x| x, i64::wrapping_add, 1000).unwrap()
        });
        assert_eq!(got, want, "len {len}");
        if len > 0 {
            let got = with_execution_policy(ExecutionPolicy::Sequential, || {
                reduce_axis(&view, 0, |x| x, i64::wrapping_add, 1000).unwrap()
            });
            assert_eq!(got.data(), &[want], "axis len {len}");
        }
    }
}

fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

#[test]
fn typed_float_max_propagates_nan_on_every_path() {
    for layout in layouts() {
        if layout.dims.len() != 2 {
            continue;
        }
        let mut data: Vec<f64> = (0..layout.len)
            .map(|i| ((i * 37) % 101) as f64 - 50.0)
            .collect();
        // One NaN at the last element of output 1 along each axis.
        let poison = offset_of(&layout, &[layout.dims[0] - 1, layout.dims[1] - 1]);
        data[poison] = f64::NAN;
        let view =
            StridedView::<f64>::new(&data, &layout.dims, &layout.strides, layout.offset as isize)
                .unwrap();
        for policy in policies() {
            let full = with_execution_policy(policy, || {
                reduce(&view, |x| x, nan_max, f64::NEG_INFINITY).unwrap()
            });
            assert!(full.is_nan(), "{} full {policy:?}", layout.name);
            for axis in 0..2 {
                let got = with_execution_policy(policy, || {
                    reduce_axis(&view, axis, |x| x, nan_max, f64::NEG_INFINITY).unwrap()
                });
                let kept = 1 - axis;
                for_each_index(&[layout.dims[kept]], |k| {
                    let mut want = f64::NEG_INFINITY;
                    for r in 0..layout.dims[axis] {
                        let mut index = [0usize; 2];
                        index[kept] = k[0];
                        index[axis] = r;
                        want = nan_max(want, data[offset_of(&layout, &index)]);
                    }
                    let value = got.data()[k[0]];
                    assert!(
                        value == want || (value.is_nan() && want.is_nan()),
                        "{} axis {axis} output {} {policy:?}: {value} vs {want}",
                        layout.name,
                        k[0]
                    );
                });
            }
        }
    }
}

#[test]
fn typed_reduce_axis_column_sweep_is_thread_count_independent() {
    // The column sweep keeps the left to right order per output, so float
    // sums are bitwise equal to a sequential fold for any thread count.
    let dims = [2048usize, 33];
    let data: Vec<f64> = (0..dims[0] * dims[1])
        .map(|i| 1.0 / (1.0 + i as f64))
        .collect();
    let view = StridedView::<f64>::new(&data, &dims, &[1, dims[0] as isize], 0).unwrap();
    let want: Vec<f64> = (0..dims[0])
        .map(|i| (0..dims[1]).fold(0.25, |acc, j| acc + data[i + j * dims[0]]))
        .collect();
    for threads in 1..=4 {
        let policy = ExecutionPolicy::Rayon {
            max_threads: NonZeroUsize::new(threads).unwrap(),
        };
        let got = with_execution_policy(policy, || {
            reduce_axis(&view, 1, |x| x, |a, b| a + b, 0.25).unwrap()
        });
        let same = got
            .data()
            .iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "threads {threads}");
    }
}
