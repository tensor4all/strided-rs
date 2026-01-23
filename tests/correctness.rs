use approx::assert_relative_eq;
use mdarray::{DynRank, Tensor};
use mdarray_strided::{
    copy_into_uninit, copy_transpose_scale_into, copy_transpose_scale_into_fast, dot, map_into,
    reduce, reduce_axis, zip_map2_into, zip_map4_into,
};
use std::mem::MaybeUninit;

fn make_tensor(rows: usize, cols: usize) -> Tensor<f64, DynRank> {
    Tensor::from_fn([rows, cols], |idx| (idx[0] * cols + idx[1]) as f64).into_dyn()
}

#[test]
fn test_map_into_transposed() {
    let a = make_tensor(8, 5);
    let a_t = a.as_ref().permute([1, 0]);
    let mut out = Tensor::zeros([5, 8]).into_dyn();

    map_into(&mut out, &a_t, |x| x * 2.0).unwrap();

    let expected = Tensor::from_fn([5, 8], |idx| a_t[[idx[0], idx[1]]] * 2.0).into_dyn();
    assert_eq!(out.shape().dims(), expected.shape().dims());
    for i in 0..5 {
        for j in 0..8 {
            assert_relative_eq!(out[[i, j]], expected[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_zip_map2_mixed_strides() {
    let a = make_tensor(6, 4);
    let b = make_tensor(6, 4);
    let a_t = a.as_ref().permute([1, 0]);
    let mut out = Tensor::zeros([4, 6]).into_dyn();

    let b_t = b.as_ref().permute([1, 0]);
    zip_map2_into(&mut out, &a_t, &b_t, |x, y| x + y).unwrap();

    let expected =
        Tensor::from_fn([4, 6], |idx| a_t[[idx[0], idx[1]]] + b[[idx[1], idx[0]]]).into_dyn();
    for i in 0..4 {
        for j in 0..6 {
            assert_relative_eq!(out[[i, j]], expected[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_reduce_sum() {
    let a = make_tensor(10, 12);
    let result = reduce(&a, |x| *x, |a, b| a + b, 0.0).unwrap();
    let expected: f64 = a.iter().copied().sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_reduce_axis_sum() {
    let a: Tensor<f64, DynRank> =
        Tensor::from_fn([4, 3, 2], |idx| (idx[0] + 2 * idx[1] + 3 * idx[2]) as f64).into_dyn();

    let result = reduce_axis(&a, 1, |x| *x, |a, b| a + b, 0.0).unwrap();
    let expected = sum_axis_expected(&a, 1);

    assert_eq!(result.shape().dims(), expected.shape().dims());
    for i in 0..result.dim(0) {
        for j in 0..result.dim(1) {
            assert_relative_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_dot() {
    let a = make_tensor(7, 3);
    let b = make_tensor(7, 3);
    let result = dot(&a, &b).unwrap();
    let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_copy_transpose_scale_into() {
    let a: Tensor<f64, DynRank> =
        Tensor::from_fn([2, 3], |idx| (idx[0] * 10 + idx[1]) as f64).into_dyn();
    let mut out = Tensor::zeros([3, 2]).into_dyn();

    copy_transpose_scale_into(&mut out, &a, 3.0).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_relative_eq!(out[[j, i]], 3.0 * a[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_copy_transpose_scale_into_fast_small() {
    let a: Tensor<f64, DynRank> =
        Tensor::from_fn([2, 3], |idx| (idx[0] * 10 + idx[1]) as f64).into_dyn();
    let mut out = Tensor::zeros([3, 2]).into_dyn();

    copy_transpose_scale_into_fast(&mut out, &a, 3.0).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_relative_eq!(out[[j, i]], 3.0 * a[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_copy_transpose_scale_into_fast_large() {
    // Test with a size that exercises the 4x4 micro-kernel and edge cases
    let a: Tensor<f64, DynRank> =
        Tensor::from_fn([17, 13], |idx| (idx[0] * 100 + idx[1]) as f64).into_dyn();
    let mut out = Tensor::zeros([13, 17]).into_dyn();

    copy_transpose_scale_into_fast(&mut out, &a, 2.5).unwrap();

    for i in 0..17 {
        for j in 0..13 {
            assert_relative_eq!(out[[j, i]], 2.5 * a[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_copy_transpose_scale_into_fast_matches_original() {
    // Verify that fast version produces same results as original
    let a: Tensor<f64, DynRank> =
        Tensor::from_fn([100, 80], |idx| (idx[0] as f64 * 0.1 + idx[1] as f64 * 0.01)).into_dyn();

    let mut out_orig = Tensor::zeros([80, 100]).into_dyn();
    let mut out_fast = Tensor::zeros([80, 100]).into_dyn();

    copy_transpose_scale_into(&mut out_orig, &a, 3.14).unwrap();
    copy_transpose_scale_into_fast(&mut out_fast, &a, 3.14).unwrap();

    for i in 0..80 {
        for j in 0..100 {
            assert_relative_eq!(out_fast[[i, j]], out_orig[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_copy_into_uninit_contiguous() {
    let a = make_tensor(4, 5);
    let dims = [4usize, 5usize];
    let strides = [5isize, 1isize];
    let mut buf: Vec<MaybeUninit<f64>> = Vec::with_capacity(20);
    buf.resize_with(20, MaybeUninit::uninit);

    unsafe {
        copy_into_uninit(buf.as_mut_ptr(), &dims, &strides, &a).unwrap();
    }

    let out: Vec<f64> = unsafe { buf.into_iter().map(|v| v.assume_init()).collect() };
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let idx = i * dims[1] + j;
            assert_relative_eq!(out[idx], a[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_zip_map4_into_contiguous() {
    let a = make_tensor(4, 5);
    let b = make_tensor(4, 5);
    let c = make_tensor(4, 5);
    let d = make_tensor(4, 5);
    let mut out = Tensor::zeros([4, 5]).into_dyn();

    zip_map4_into(&mut out, &a, &b, &c, &d, |a, b, c, d| a + b + c + d).unwrap();

    for i in 0..4 {
        for j in 0..5 {
            let expected = a[[i, j]] + b[[i, j]] + c[[i, j]] + d[[i, j]];
            assert_relative_eq!(out[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_zip_map4_into_permuted() {
    // Test with 4D arrays and various permutations (like the Julia benchmark)
    let size = 8usize;
    let a: Tensor<f64, DynRank> =
        Tensor::from_fn([size, size, size, size], |idx| {
            (idx[0] + 2 * idx[1] + 3 * idx[2] + 4 * idx[3]) as f64
        })
        .into_dyn();

    // Julia (1,2,3,4) -> Rust [0,1,2,3] (identity)
    // Julia (2,3,4,1) -> Rust [1,2,3,0]
    // Julia (3,4,1,2) -> Rust [2,3,0,1]
    // Julia (4,1,2,3) -> Rust [3,0,1,2]
    let p1 = a.as_ref().permute([0, 1, 2, 3]); // identity
    let p2 = a.as_ref().permute([1, 2, 3, 0]);
    let p3 = a.as_ref().permute([2, 3, 0, 1]);
    let p4 = a.as_ref().permute([3, 0, 1, 2]);

    let mut out = Tensor::zeros([size, size, size, size]).into_dyn();

    zip_map4_into(&mut out, &p1, &p2, &p3, &p4, |a, b, c, d| a + b + c + d).unwrap();

    // Verify against naive computation
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                for l in 0..size {
                    let expected =
                        p1[[i, j, k, l]] + p2[[i, j, k, l]] + p3[[i, j, k, l]] + p4[[i, j, k, l]];
                    assert_relative_eq!(out[[i, j, k, l]], expected, epsilon = 1e-10);
                }
            }
        }
    }
}

#[test]
fn test_zip_map4_into_mixed_ops() {
    let a = make_tensor(3, 4);
    let b = make_tensor(3, 4);
    let c = make_tensor(3, 4);
    let d = make_tensor(3, 4);
    let mut out = Tensor::zeros([3, 4]).into_dyn();

    // Test with a more complex combining function
    zip_map4_into(&mut out, &a, &b, &c, &d, |a, b, c, d| a * b + c * d).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let expected = a[[i, j]] * b[[i, j]] + c[[i, j]] * d[[i, j]];
            assert_relative_eq!(out[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

fn sum_axis_expected(tensor: &Tensor<f64, DynRank>, axis: usize) -> Tensor<f64, DynRank> {
    let mut dims: Vec<usize> = (0..tensor.rank()).map(|i| tensor.dim(i)).collect();
    let axis_dim = dims.remove(axis);
    if dims.is_empty() {
        dims.push(1);
    }
    let mut out = Tensor::from_fn(dims.as_slice(), |_| 0.0).into_dyn();

    let out_rank = out.rank();
    let mut idx = vec![0usize; out_rank];
    let total: usize = dims.iter().product();
    for _ in 0..total {
        let mut sum = 0.0;
        for k in 0..axis_dim {
            let mut full_idx = idx.clone();
            full_idx.insert(axis, k);
            sum += tensor[&full_idx[..]];
        }
        out[&idx[..]] = sum;

        for i in (0..out_rank).rev() {
            idx[i] += 1;
            if idx[i] < dims[i] {
                break;
            }
            idx[i] = 0;
        }
    }

    out
}
