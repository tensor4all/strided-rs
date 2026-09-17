use core::mem::MaybeUninit;
use strided_basic::{axpby_accum, embed_diagonal_into_uninit, triangular_mask_into_uninit};

fn initialized<T: Copy>(out: &[MaybeUninit<T>]) -> Vec<T> {
    // SAFETY: tests call this only after a successful full-overwrite kernel.
    out.iter().map(|x| unsafe { x.assume_init() }).collect()
}

#[test]
fn axpby_values_and_error_preserve_destination() {
    for n in [0, 1, 3, 8, 32769, 65537] {
        let x: Vec<f64> = (0..n).map(|i| i as f64 - 7.0).collect();
        let mut y = vec![3.0; n];
        axpby_accum(&mut y, &x, 2.0, -3.0).unwrap();
        assert_eq!(y, x.iter().map(|x| 2.0 * x - 9.0).collect::<Vec<_>>());
    }
    let mut y = [7.0];
    assert!(axpby_accum(&mut y, &[], 1.0, 1.0).is_err());
    assert_eq!(y, [7.0]);
    axpby_accum(&mut y, &[f64::NAN], 0.0, 1.0).unwrap();
    assert!(y[0].is_nan());
    let mut y = [f64::NAN];
    axpby_accum(&mut y, &[1.0], 1.0, 0.0).unwrap();
    assert!(y[0].is_nan());
    use num_complex::Complex64;
    let x = [Complex64::new(2.0, 3.0)];
    let mut y = [Complex64::new(4.0, -1.0)];
    let alpha = Complex64::new(0.0, 1.0);
    let beta = Complex64::new(-1.0, 0.0);
    let expected = alpha * x[0] + beta * y[0];
    axpby_accum(&mut y, &x, alpha, beta).unwrap();
    assert_eq!(y[0], expected);
}

#[test]
fn triangular_matches_elementwise_reference() {
    for shape in [
        vec![0, 3],
        vec![3, 0, 2],
        vec![3, 5],
        vec![5, 3, 2],
        vec![129, 257, 2],
    ] {
        let n = shape.iter().product();
        let input: Vec<i32> = (0..n as i32).map(|x| x + 1).collect();
        for k in [i64::MIN, -4, -1, 0, 1, 4, i64::MAX] {
            for upper in [false, true] {
                let mut output = vec![MaybeUninit::uninit(); n];
                triangular_mask_into_uninit(&mut output, &input, &shape, k, upper, -99).unwrap();
                let expected: Vec<_> = input
                    .iter()
                    .enumerate()
                    .map(|(flat, &v)| {
                        let row = (flat % shape[0]) as i128;
                        let col = ((flat / shape[0]) % shape[1]) as i128;
                        let keep = if upper {
                            row <= col - k as i128
                        } else {
                            row >= col - k as i128
                        };
                        if keep {
                            v
                        } else {
                            -99
                        }
                    })
                    .collect();
                assert_eq!(initialized(&output), expected, "{shape:?} {k} {upper}");
            }
        }
    }
}

#[test]
fn embedding_all_axes_and_ranks() {
    for shape in [
        vec![0],
        vec![1],
        vec![3],
        vec![2, 3],
        vec![2, 3, 4],
        vec![3, 1, 2, 2],
        vec![37, 43],
    ] {
        let n: usize = shape.iter().product();
        let input: Vec<i64> = (0..n as i64).map(|x| x + 1).collect();
        for a in 0..shape.len() {
            for b in 0..=shape.len() {
                let mut out_shape = shape.clone();
                out_shape.insert(b, shape[a]);
                let mut output = vec![MaybeUninit::uninit(); n * shape[a]];
                embed_diagonal_into_uninit(&mut output, &input, &shape, a, b, 0).unwrap();
                let mut expected = vec![0; output.len()];
                for (i, &value) in input.iter().enumerate() {
                    let mut rem = i;
                    let mut coords = Vec::new();
                    for &dim in &shape {
                        coords.push(rem % dim);
                        rem /= dim;
                    }
                    let d = coords[a];
                    coords.insert(b, d);
                    let mut stride = 1;
                    let mut offset = 0;
                    for (&coord, &dim) in coords.iter().zip(&out_shape) {
                        offset += coord * stride;
                        stride *= dim;
                    }
                    expected[offset] = value;
                }
                assert_eq!(initialized(&output), expected, "{shape:?} {a} {b}");
            }
        }
    }
}

#[test]
fn invalid_shapes_do_not_write() {
    let mut output = [MaybeUninit::new(71_i32); 4];
    assert!(triangular_mask_into_uninit(&mut output, &[1; 4], &[4], 0, true, 0).is_err());
    assert!(triangular_mask_into_uninit(&mut output, &[1; 4], &[2, 3], 0, true, 0).is_err());
    assert!(
        triangular_mask_into_uninit(&mut output, &[1; 4], &[usize::MAX, 2], 0, true, 0).is_err()
    );
    assert!(embed_diagonal_into_uninit(&mut output, &[1, 2], &[2], 1, 1, 0).is_err());
    assert!(embed_diagonal_into_uninit(&mut output, &[1, 2], &[2], 0, 2, 0).is_err());
    assert!(embed_diagonal_into_uninit(&mut output, &[1, 2], &[3], 0, 1, 0).is_err());
    assert_eq!(initialized(&output), [71; 4]);
}

#[cfg(feature = "parallel")]
#[test]
fn bounded_and_sequential_policy_inside_pool() {
    use std::num::NonZeroUsize;
    use strided_basic::{with_execution_policy, ExecutionPolicy};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        for policy in [
            ExecutionPolicy::Sequential,
            ExecutionPolicy::Rayon {
                max_threads: NonZeroUsize::new(2).unwrap(),
            },
        ] {
            with_execution_policy(policy, || {
                triangular_matches_elementwise_reference();
                embedding_all_axes_and_ranks();
                axpby_values_and_error_preserve_destination();
            });
        }
    });
}
