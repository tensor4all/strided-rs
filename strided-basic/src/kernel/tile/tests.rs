use super::*;
use crate::kernel::{
    fold_plan, for_each_inner_block_preordered, for_each_pointwise_block_preordered, pointwise_plan,
};
use crate::threading::test_pool;
use crate::{
    add_acc, copy_into, copy_into_uninit, map_into, zip_map2_into, zip_map3_into, zip_map4_into,
    Exec, StridedView, StridedViewMut,
};
use num_complex::Complex64;
use std::mem::MaybeUninit;

/// Column-major `[n, n, batch]` strides.
fn col_major(n: usize) -> [isize; 3] {
    [1, n as isize, (n * n) as isize]
}

/// The serial token, a one-lane token of a one-worker pool and a four-lane
/// token of a four-worker pool, each passed to `operation`.
fn each_exec<R>(mut operation: impl FnMut(Exec<'_>) -> R) -> [R; 3] {
    let serial = operation(Exec::serial());
    let one = test_pool(1);
    let one_lane = operation(Exec::new(&one, 1).unwrap());
    let four = test_pool(4);
    let four_lanes = operation(Exec::new(&four, 4).unwrap());
    [serial, one_lane, four_lanes]
}

fn f64_values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 13 + seed) % 31) as f64 - 15.0 + 0.25)
        .collect()
}

fn c64_values(len: usize, seed: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            Complex64::new(
                ((i * 13 + seed) % 31) as f64 - 15.0,
                ((i * 5 + seed) % 17) as f64 + 0.5,
            )
        })
        .collect()
}

/// `out[i, j, k] = f(a[i, j, k], b[j, i, k])`: the naive loop of the issue.
fn naive_transposed<T: Copy>(
    n: usize,
    batch: usize,
    a: &[T],
    b: &[T],
    f: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut out = a.to_vec();
    for k in 0..batch {
        let o = k * n * n;
        for j in 0..n {
            for i in 0..n {
                out[o + i + n * j] = f(a[o + i + n * j], b[o + j + n * i]);
            }
        }
    }
    out
}

fn check_transposed_zip<T>(n: usize, batch: usize, values: impl Fn(usize, usize) -> Vec<T>)
where
    T: Copy + Send + Sync + PartialEq + std::fmt::Debug + std::ops::Sub<Output = T> + 'static,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let len = n * n * batch;
    let dims = [n, n, batch];
    let strides = col_major(n);
    let a = values(len, 1);
    let b = values(len, 7);
    let c = values(len, 11);
    let e = values(len, 3);
    let av = StridedView::<T>::new(&a, &dims, &strides, 0).unwrap();
    let bt = StridedView::<T>::new(&b, &dims, &strides, 0)
        .unwrap()
        .permute(&[1, 0, 2])
        .unwrap();
    let cv = StridedView::<T>::new(&c, &dims, &strides, 0).unwrap();
    let ev = StridedView::<T>::new(&e, &dims, &strides, 0).unwrap();

    let sub = naive_transposed(n, batch, &a, &b, |x, y| x - y);
    let copied = naive_transposed(n, batch, &a, &b, |_, y| y);
    let zip3 = {
        let ab = naive_transposed(n, batch, &a, &b, |x, y| x * y);
        ab.iter().zip(&c).map(|(&p, &q)| p + q).collect::<Vec<_>>()
    };
    let zip4 = {
        let ab = naive_transposed(n, batch, &a, &b, |x, y| x - y);
        ab.iter()
            .zip(&c)
            .zip(&e)
            .map(|((&p, &q), &r)| p * q + r)
            .collect::<Vec<_>>()
    };
    let accumulated = {
        let bt_dense = naive_transposed(n, batch, &a, &b, |_, y| y);
        c.iter()
            .zip(&bt_dense)
            .map(|(&p, &q)| p + q)
            .collect::<Vec<_>>()
    };

    for out in each_exec(|exec| {
        let mut out = vec![T::default(); len];
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        zip_map2_into(&mut dv, &av, &bt, |x, y| x - y, exec).unwrap();
        out
    }) {
        assert_eq!(out, sub, "zip_map2_into n={n} batch={batch}");
    }
    for out in each_exec(|exec| {
        let mut out = vec![T::default(); len];
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        copy_into(&mut dv, &bt, exec).unwrap();
        out
    }) {
        assert_eq!(out, copied, "copy_into n={n} batch={batch}");
    }
    for out in each_exec(|exec| {
        let mut out = vec![T::default(); len];
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        map_into(&mut dv, &bt, |y| y, exec).unwrap();
        out
    }) {
        assert_eq!(out, copied, "map_into n={n} batch={batch}");
    }
    for out in each_exec(|exec| {
        let mut out = vec![MaybeUninit::<T>::uninit(); len];
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        copy_into_uninit(&mut dv, &bt, exec).unwrap();
        // SAFETY: a successful copy initialized every element of `out`.
        out.iter()
            .map(|v| unsafe { v.assume_init() })
            .collect::<Vec<_>>()
    }) {
        assert_eq!(out, copied, "copy_into_uninit n={n} batch={batch}");
    }
    for out in each_exec(|exec| {
        let mut out = vec![T::default(); len];
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        zip_map3_into(&mut dv, &av, &bt, &cv, |x, y, z| x * y + z, exec).unwrap();
        out
    }) {
        assert_eq!(out, zip3, "zip_map3_into n={n} batch={batch}");
    }
    for out in each_exec(|exec| {
        let mut out = vec![T::default(); len];
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        zip_map4_into(
            &mut dv,
            &av,
            &bt,
            &cv,
            &ev,
            |x, y, z, w| (x - y) * z + w,
            exec,
        )
        .unwrap();
        out
    }) {
        assert_eq!(out, zip4, "zip_map4_into n={n} batch={batch}");
    }
    for out in each_exec(|exec| {
        let mut out = c.clone();
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        add_acc(&mut dv, &bt, exec).unwrap();
        out
    }) {
        assert_eq!(out, accumulated, "add_acc n={n} batch={batch}");
    }
}

#[test]
fn transposed_small_blocks_equal_the_naive_loop() {
    // Batch 4096 takes the planned path; 100 is at most 1024 elements for
    // n <= 3 and takes the small-tensor plan, whose blocks are the full
    // dimensions. 37 is not a multiple of any block.
    for n in [2usize, 3, 4] {
        for batch in [4096usize, 100, 37] {
            check_transposed_zip(n, batch, f64_values);
            check_transposed_zip(n, batch, c64_values);
        }
    }
}

#[test]
fn transposed_small_block_plan_keeps_the_long_axis_outermost() {
    for n in [2usize, 3, 4] {
        let batch = 4096;
        let dims = [n, n, batch];
        let dst = col_major(n);
        let bt = [n as isize, 1, (n * n) as isize];
        for elem in [8usize, 16] {
            let (fused, strides, plan) = pointwise_plan(
                &dims,
                &[&dst, &dst, &bt],
                Some(0),
                &[elem; 3],
                n * n * batch,
            );
            // Neither leading pair fuses; the batch axis is the outermost
            // level, with the same stride `n²` in every operand.
            assert_eq!(fused.as_slice(), [n, n, batch], "n={n}");
            for s in &strides {
                assert_eq!(s[2], (n * n) as isize, "n={n}");
            }
            // The tile levels stay whole; the long axis is blocked to the
            // budget over all three operands.
            let expected = (TILE_BUDGET_BYTES / (n * n * 3 * elem)).clamp(MIN_RUN, batch);
            assert_eq!(plan.block.as_slice(), [n, n, expected], "n={n} elem={elem}");
            let refs: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
            assert!(SmallTile::of(&fused, &plan.block, &refs).is_some());
        }
    }
}

#[test]
fn the_order_exact_walk_and_folds_are_unchanged() {
    // The order-exact walk still visits one innermost run per tile column,
    // tile inside, long axis outside: exactly the nested loops, with the
    // plan's blocks of the long axis making no difference to the order.
    let n = 3usize;
    let batch = 200usize;
    let dims = [n, n, batch];
    let dst = col_major(n);
    let bt = [n as isize, 1, (n * n) as isize];
    let (fused, strides, plan) =
        pointwise_plan(&dims, &[&dst, &bt], Some(0), &[8, 8], n * n * batch);
    assert!(plan.block[2] < batch, "the long axis is blocked");
    let mut seen = Vec::new();
    for_each_inner_block_preordered(
        &fused,
        &plan.block,
        &strides,
        &[0, 0],
        |offsets, len, inner| {
            seen.push((offsets[0], offsets[1], len, inner[0], inner[1]));
            Ok(())
        },
    )
    .unwrap();
    let mut expected = Vec::new();
    for k in 0..batch as isize {
        for j in 0..n as isize {
            let nn = (n * n) as isize;
            expected.push((k * nn + 3 * j, k * nn + j, n, 1, 3));
        }
    }
    assert_eq!(seen, expected);

    // A fold takes the block model, never the tile blocks.
    let (_, _, folded) = fold_plan(&dims, &[&dst, &bt], 0, &[8, 8], n * n * batch);
    let model = crate::block::compute_block_sizes(&fused, &[&strides[0], &strides[1]], &[8, 8]);
    assert_eq!(folded.block, model);
}

#[test]
fn the_pointwise_walk_visits_every_element_once() {
    // Shapes with and without a tile, odd extents and blocks that do not
    // divide the long axis.
    let cases: [(&[usize], &[usize]); 6] = [
        (&[2, 2, 100], &[2, 2, 33]),
        (&[3, 5, 70], &[3, 5, 16]),
        (&[16, 50], &[16, 17]),
        (&[2, 2, 10], &[2, 2, 10]), // long axis below MIN_RUN: block walk
        (&[8, 8, 40], &[8, 8, 40]), // 64-element tile: block walk
        (&[2, 2, 40], &[1, 2, 40]), // blocked tile level: block walk
    ];
    for (dims, blocks) in cases {
        let rank = dims.len();
        // A dense column-major operand and one whose two leading axes are
        // stored transposed, so nothing fuses.
        let mut col = vec![1isize; rank];
        for i in 1..rank {
            col[i] = col[i - 1] * dims[i - 1] as isize;
        }
        let mut swapped = col.clone();
        swapped[0] = dims[1] as isize;
        swapped[1] = 1;
        let total: usize = dims.iter().product();
        let mut hits = vec![0u8; total];
        let mut second = vec![0u8; total];
        for_each_pointwise_block_preordered(
            dims,
            blocks,
            &[&col[..], &swapped[..]],
            &[0, 0],
            |offsets, len, inner| {
                for t in 0..len as isize {
                    hits[(offsets[0] + t * inner[0]) as usize] += 1;
                    second[(offsets[1] + t * inner[1]) as usize] += 1;
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(hits.iter().all(|&h| h == 1), "dims={dims:?}");
        assert!(second.iter().all(|&h| h == 1), "dims={dims:?}");
    }
}

#[test]
fn tiles_are_only_taken_for_a_small_tile_under_one_long_axis() {
    let s3: [&[isize]; 1] = [&[1, 2, 4]];
    assert!(SmallTile::of(&[2, 2, 64], &[2, 2, 64], &s3).is_some());
    // 64 elements in the leading levels.
    assert!(SmallTile::of(&[8, 8, 64], &[8, 8, 64], &[&[1, 8, 64]]).is_none());
    // Two levels above the tile.
    assert!(SmallTile::of(&[2, 2, 64, 3], &[2, 2, 64, 3], &[&[1, 2, 4, 256]]).is_none());
    // A blocked tile level, and a long axis block below MIN_RUN.
    assert!(SmallTile::of(&[2, 2, 64], &[1, 2, 64], &s3).is_none());
    assert!(SmallTile::of(&[2, 2, 64], &[2, 2, 8], &s3).is_none());
    // An empty axis.
    assert!(SmallTile::of(&[2, 0, 64], &[2, 0, 64], &s3).is_none());

    assert_eq!(tile_blocks(&[2, 2, 8], &[8, 8]), None);
    assert_eq!(
        tile_blocks(&[2, 2, 100_000], &[8, 8, 8])
            .unwrap()
            .as_slice(),
        [2, 2, TILE_BUDGET_BYTES / (4 * 24)]
    );
    // Zero-sized operands only: nothing to budget, the long axis stays whole.
    assert_eq!(
        tile_blocks(&[2, 2, 100], &[0, 0]).unwrap().as_slice(),
        [2, 2, 100]
    );
}
