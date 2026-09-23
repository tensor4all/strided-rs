//! Serial versus parallel `ExecContext` equivalence for the static structural
//! plans (slice, reverse, concatenate, pad).

use core::mem::MaybeUninit;

use super::{ConcatenatePlan, PadPlan, RawStridedMut, RawStridedRef, ReversePlan, SlicePlan};
use crate::threading::run_serial_and_parallel;

/// Visit every multi-index of `dims` in column-major order.
fn for_each_index(dims: &[usize], mut visit: impl FnMut(&[usize])) {
    if dims.iter().any(|&d| d == 0) {
        return;
    }
    let mut idx = vec![0usize; dims.len()];
    loop {
        visit(&idx);
        let mut axis = 0;
        loop {
            if axis == dims.len() {
                return;
            }
            idx[axis] += 1;
            if idx[axis] < dims[axis] {
                break;
            }
            idx[axis] = 0;
            axis += 1;
        }
    }
}

fn linear(dims: &[usize], idx: &[usize]) -> usize {
    let mut out = 0;
    let mut scale = 1;
    for (&d, &i) in dims.iter().zip(idx) {
        out += i * scale;
        scale *= d;
    }
    out
}

fn offset_of(strides: &[isize], base: isize, idx: &[usize]) -> usize {
    (base
        + idx
            .iter()
            .zip(strides)
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>()) as usize
}

/// Storage length and base offset for a strided layout, handling negative strides.
fn layout_extent(dims: &[usize], strides: &[isize]) -> (usize, isize) {
    if dims.iter().any(|&d| d == 0) {
        return (1, 0);
    }
    let mut lo = 0isize;
    let mut hi = 0isize;
    for (&d, &s) in dims.iter().zip(strides) {
        let reach = (d as isize - 1) * s;
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    ((hi - lo + 1) as usize, -lo)
}

/// Operand storage whose logical element at `idx` is `tag + linear(idx)`.
fn operand(dims: &[usize], strides: &[isize], tag: i64) -> (Vec<i64>, isize) {
    let (len, base) = layout_extent(dims, strides);
    let mut data = vec![-7i64; len];
    for_each_index(dims, |idx| {
        data[offset_of(strides, base, idx)] = tag + linear(dims, idx) as i64;
    });
    (data, base)
}

/// Read a strided destination back in logical column-major order.
fn read_logical(data: &[i64], dims: &[usize], strides: &[isize], base: isize) -> Vec<i64> {
    let mut out = Vec::new();
    for_each_index(dims, |idx| out.push(data[offset_of(strides, base, idx)]));
    out
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut scale = 1isize;
    for &d in dims {
        strides.push(scale);
        scale *= d.max(1) as isize;
    }
    strides
}

/// Run `plan` into a freshly allocated destination under both contexts, via
/// the initialized and the uninit entry point, and assert all four agree with
/// `expected` (logical column-major order).
fn check_both<F, G>(
    dest_dims: &[usize],
    dest_strides: &[isize],
    expected: &[i64],
    label: &str,
    run: F,
    run_uninit: G,
) where
    F: Fn(&mut RawStridedMut<'_, i64>) + Sync,
    G: Fn(&mut RawStridedMut<'_, MaybeUninit<i64>>) + Sync,
{
    let (len, base) = layout_extent(dest_dims, dest_strides);
    let (serial, parallel) = run_serial_and_parallel(|| {
        let mut init = vec![-1i64; len];
        let mut dest = RawStridedMut::new(&mut init, dest_dims, dest_strides, base).unwrap();
        run(&mut dest);

        let mut storage = vec![MaybeUninit::new(-1i64); len];
        let mut dest = RawStridedMut::new(&mut storage, dest_dims, dest_strides, base).unwrap();
        run_uninit(&mut dest);
        // SAFETY: every slot was initialized to -1 before execution.
        let uninit: Vec<i64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
        (
            read_logical(&init, dest_dims, dest_strides, base),
            read_logical(&uninit, dest_dims, dest_strides, base),
        )
    });
    assert_eq!(serial.0, expected, "{label}: serial execute");
    assert_eq!(parallel.0, expected, "{label}: parallel execute");
    assert_eq!(serial.1, expected, "{label}: serial execute_uninit");
    assert_eq!(parallel.1, expected, "{label}: parallel execute_uninit");
}

#[test]
fn slice_plan_serial_and_parallel_contexts_agree() {
    // (operand dims, operand strides, starts, limits, steps)
    type SliceCase = (Vec<usize>, Vec<isize>, Vec<usize>, Vec<usize>, Vec<usize>);
    let cases: Vec<SliceCase> = vec![
        (vec![80_005], vec![1], vec![3], vec![80_004], vec![2]),
        (vec![70_001], vec![-1], vec![1], vec![70_001], vec![1]),
        (
            vec![7, 9, 11, 13, 5, 3],
            vec![1, 7, 63, 693, 9009, -45045],
            vec![1, 0, 2, 1, 0, 0],
            vec![7, 9, 11, 13, 5, 3],
            vec![1, 2, 1, 1, 2, 1],
        ),
        (
            vec![0, 50_000],
            vec![1, 1],
            vec![0, 0],
            vec![0, 50_000],
            vec![1, 1],
        ),
        (vec![], vec![], vec![], vec![], vec![]),
    ];
    for (dims, strides, starts, limits, steps) in cases {
        let dest_dims: Vec<usize> = (0..dims.len())
            .map(|a| (limits[a] - starts[a]).div_ceil(steps[a]))
            .collect();
        let dest_strides = col_major(&dest_dims);
        let plan = SlicePlan::compile(
            &dims,
            &strides,
            &dest_dims,
            &dest_strides,
            &starts,
            &limits,
            &steps,
        )
        .unwrap();
        let (data, base) = operand(&dims, &strides, 1);
        let source = RawStridedRef::new(&data, &dims, &strides, base).unwrap();
        let mut expected = Vec::new();
        for_each_index(&dest_dims, |idx| {
            let src: Vec<usize> = (0..dims.len())
                .map(|a| starts[a] + idx[a] * steps[a])
                .collect();
            expected.push(1 + linear(&dims, &src) as i64);
        });
        check_both(
            &dest_dims,
            &dest_strides,
            &expected,
            &format!("slice {dims:?}"),
            |dest| plan.execute(dest, &source).unwrap(),
            |dest| plan.execute_uninit(dest, &source).unwrap(),
        );
    }
}

#[test]
fn reverse_plan_serial_and_parallel_contexts_agree() {
    // (dims, operand strides, dest strides, reversed axes)
    type ReverseCase = (Vec<usize>, Vec<isize>, Vec<isize>, Vec<usize>);
    let cases: Vec<ReverseCase> = vec![
        (vec![65_537], vec![1], vec![1], vec![0]),
        (vec![65_537], vec![-1], vec![1], vec![0]),
        (vec![257, 131], vec![1, 257], vec![1, 257], vec![0]),
        (vec![257, 131], vec![131, 1], vec![1, 257], vec![1]),
        (vec![3, 20_001], vec![1, 3], vec![-1, 3], vec![0, 1]),
        (
            vec![3, 5, 7, 9, 11, 5],
            vec![1, 3, 15, 105, 945, 10_395],
            vec![-1, 3, -15, 105, 945, -10_395],
            vec![0, 2, 5],
        ),
        (vec![50_000, 0], vec![1, 50_000], vec![1, 50_000], vec![0]),
        (vec![], vec![], vec![], vec![]),
    ];
    for (dims, strides, dest_strides, axes) in cases {
        let plan = ReversePlan::compile(&dims, &strides, &dest_strides, &axes).unwrap();
        let (data, base) = operand(&dims, &strides, 1);
        let source = RawStridedRef::new(&data, &dims, &strides, base).unwrap();
        let mut expected = Vec::new();
        for_each_index(&dims, |idx| {
            let src: Vec<usize> = (0..dims.len())
                .map(|a| {
                    if axes.contains(&a) {
                        dims[a] - 1 - idx[a]
                    } else {
                        idx[a]
                    }
                })
                .collect();
            expected.push(1 + linear(&dims, &src) as i64);
        });
        check_both(
            &dims,
            &dest_strides,
            &expected,
            &format!("reverse {dims:?} axes {axes:?}"),
            |dest| plan.execute(dest, &source).unwrap(),
            |dest| plan.execute_uninit(dest, &source).unwrap(),
        );
    }
}

#[test]
fn concatenate_plan_serial_and_parallel_contexts_agree() {
    // (per input dims, per input strides, axis, dest strides or None for col-major)
    type ConcatCase = (Vec<Vec<usize>>, Vec<Vec<isize>>, usize, Option<Vec<isize>>);
    let many_small: Vec<Vec<usize>> = (0..41).map(|i| vec![1_000 + i]).collect();
    let many_small_strides: Vec<Vec<isize>> = (0..41)
        .map(|i| vec![if i % 3 == 0 { -1 } else { 1 }])
        .collect();
    let cases: Vec<ConcatCase> = vec![
        // two big segments, one reversed
        (
            vec![vec![40_001], vec![30_007]],
            vec![vec![1], vec![-1]],
            0,
            None,
        ),
        // many segments each below the threshold, total above it
        (many_small, many_small_strides, 0, None),
        // zero-size segment in the middle
        (
            vec![vec![17, 2_001], vec![17, 0], vec![17, 1_999]],
            vec![vec![1, 17], vec![1, 17], vec![2_001 * 17, -17]],
            1,
            None,
        ),
        // concatenate along axis 0 of rank 2, transposed dest
        (
            vec![vec![301, 150], vec![199, 150]],
            vec![vec![1, 301], vec![150, 1]],
            0,
            Some(vec![150, 1]),
        ),
        // rank 6
        (
            vec![vec![3, 5, 7, 9, 11, 2], vec![3, 5, 7, 9, 11, 3]],
            vec![
                vec![1, 3, 15, 105, 945, 10_395],
                vec![-1, 3, 15, 105, 945, 10_395],
            ],
            5,
            None,
        ),
        // all segments empty
        (
            vec![vec![0, 40_000], vec![0, 40_000]],
            vec![vec![1, 1], vec![1, 1]],
            1,
            None,
        ),
    ];
    for (input_dims, input_strides, axis, dest_strides) in cases {
        let mut dest_dims = input_dims[0].clone();
        dest_dims[axis] = input_dims.iter().map(|d| d[axis]).sum();
        let dest_strides = dest_strides.unwrap_or_else(|| col_major(&dest_dims));
        let dims_refs: Vec<&[usize]> = input_dims.iter().map(Vec::as_slice).collect();
        let strides_refs: Vec<&[isize]> = input_strides.iter().map(Vec::as_slice).collect();
        let plan =
            ConcatenatePlan::compile(&dims_refs, &strides_refs, &dest_dims, &dest_strides, axis)
                .unwrap();
        let storages: Vec<(Vec<i64>, isize)> = input_dims
            .iter()
            .zip(&input_strides)
            .enumerate()
            .map(|(k, (d, s))| operand(d, s, 1_000_000 * (k as i64 + 1)))
            .collect();
        let inputs: Vec<RawStridedRef<'_, i64>> = storages
            .iter()
            .zip(input_dims.iter().zip(&input_strides))
            .map(|((data, base), (d, s))| RawStridedRef::new(data, d, s, *base).unwrap())
            .collect();
        let mut expected = Vec::new();
        for_each_index(&dest_dims, |idx| {
            let mut local = idx.to_vec();
            let mut k = 0;
            while local[axis] >= input_dims[k][axis] {
                local[axis] -= input_dims[k][axis];
                k += 1;
            }
            expected.push(1_000_000 * (k as i64 + 1) + linear(&input_dims[k], &local) as i64);
        });
        check_both(
            &dest_dims,
            &dest_strides,
            &expected,
            &format!("concatenate {dest_dims:?} axis {axis}"),
            |dest| plan.execute(dest, &inputs).unwrap(),
            |dest| plan.execute_uninit(dest, &inputs).unwrap(),
        );
    }
}

#[test]
fn pad_plan_serial_and_parallel_contexts_agree() {
    // (operand dims, operand strides, low, high, interior, dest strides or None)
    type PadCase = (
        Vec<usize>,
        Vec<isize>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Option<Vec<isize>>,
    );
    let cases: Vec<PadCase> = vec![
        // rank 1, single long contiguous run
        (vec![70_001], vec![1], vec![3], vec![5], vec![0], None),
        // rank 1, negative low padding crops the run
        (vec![70_001], vec![1], vec![-2], vec![7], vec![0], None),
        // many contiguous runs, outer total well above worker count
        (
            vec![301, 257],
            vec![1, 301],
            vec![2, 1],
            vec![3, 0],
            vec![0, 0],
            None,
        ),
        // two long runs: outer total below worker count
        (
            vec![40_003, 2],
            vec![1, 40_003],
            vec![1, 1],
            vec![2, 1],
            vec![0, 0],
            None,
        ),
        // interior padding and negative operand stride (generic path)
        (
            vec![1_001, 41],
            vec![-1, 1_001],
            vec![1, -1],
            vec![0, 2],
            vec![1, 0],
            None,
        ),
        // non-contiguous dest (transposed)
        (
            vec![257, 131],
            vec![1, 257],
            vec![1, 2],
            vec![1, 0],
            vec![0, 0],
            Some(vec![133, 1]),
        ),
        // rank 6 with mixed padding
        (
            vec![3, 5, 7, 9, 11, 5],
            vec![1, 3, 15, 105, 945, 10_395],
            vec![1, 0, -1, 0, 1, 0],
            vec![0, 1, 0, 0, 1, 1],
            vec![0, 0, 1, 0, 0, 0],
            None,
        ),
        // zero-size operand, large dest (fill only)
        (
            vec![0, 3],
            vec![1, 1],
            vec![40_001, 0],
            vec![0, 0],
            vec![0, 0],
            None,
        ),
        // rank 0
        (vec![], vec![], vec![], vec![], vec![], None),
    ];
    for (dims, strides, low, high, interior, dest_strides) in cases {
        let rank = dims.len();
        let dest_dims: Vec<usize> = (0..rank)
            .map(|a| {
                let n = dims[a] as i64;
                let body = if n == 0 { 0 } else { n + (n - 1) * interior[a] };
                (low[a] + high[a] + body) as usize
            })
            .collect();
        let dest_strides = dest_strides.unwrap_or_else(|| col_major(&dest_dims));
        let plan = PadPlan::compile(
            &dims,
            &strides,
            &dest_dims,
            &dest_strides,
            &low,
            &high,
            &interior,
        )
        .unwrap();
        let (data, base) = operand(&dims, &strides, 1);
        let source = RawStridedRef::new(&data, &dims, &strides, base).unwrap();
        let fill = -5i64;
        let mut expected = Vec::new();
        for_each_index(&dest_dims, |idx| {
            let mut src = Vec::with_capacity(rank);
            for a in 0..rank {
                let shifted = idx[a] as i64 - low[a];
                let step = interior[a] + 1;
                if shifted < 0 || shifted % step != 0 || shifted / step >= dims[a] as i64 {
                    break;
                }
                src.push((shifted / step) as usize);
            }
            expected.push(if src.len() == rank {
                1 + linear(&dims, &src) as i64
            } else {
                fill
            });
        });
        check_both(
            &dest_dims,
            &dest_strides,
            &expected,
            &format!("pad {dims:?} low {low:?}"),
            |dest| plan.execute(dest, &source, fill).unwrap(),
            |dest| plan.execute_uninit(dest, &source, fill).unwrap(),
        );
    }
}
