use strided_basic::Exec;

use super::*;

fn grid(n: usize) -> Vec<f64> {
    // Spans saturation on both sides, includes 0.
    (0..n)
        .map(|i| (i as f64 / n.max(2) as f64 - 0.5) * 120.0 * (i % 3 + 1) as f64 / 3.0)
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

macro_rules! dtype_tests {
    ($m:ident, $t:ty, $tol:expr, $silu:ident, $sig:ident, $silu_vjp:ident, $sig_vjp:ident) => {
        mod $m {
            use super::*;
            type T = $t;

            fn close(got: &[T], want: impl Iterator<Item = f64>) {
                for (i, (g, w)) in got.iter().zip(want).enumerate() {
                    assert!(
                        (*g as f64 - w).abs() <= $tol * (1.0 + w.abs()),
                        "elem {i}: {g} vs {w}"
                    );
                }
            }

            #[test]
            fn forward_values_and_derivatives() {
                // 1, non-lane multiples, and more than one chunk.
                for n in [1usize, 2, 7, 8, 9, 1023, 1024, 1025, 3000] {
                    let xs = grid(n);
                    let x: Vec<T> = xs.iter().map(|&v| v as T).collect();
                    let xr: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                    for is_silu in [false, true] {
                        let mut y = vec![0.0 as T; n];
                        let mut d = vec![0.0 as T; n];
                        let mut scratch = vec![0.0 as T; scratch_len(n, 1)];
                        let f = if is_silu { $silu } else { $sig };
                        f(&x, &mut y, Some(&mut d), &mut scratch, Exec::serial()).unwrap();
                        let h = 1e-5;
                        let val = |v: f64| if is_silu { silu(v) } else { sigmoid(v) };
                        close(&y, xr.iter().map(|&v| val(v)));
                        // Derivative against a central difference of the f64 definition.
                        close(
                            &d,
                            xr.iter().map(|&v| (val(v + h) - val(v - h)) / (2.0 * h)),
                        );
                        let mut y2 = vec![0.0 as T; n];
                        f(&x, &mut y2, None, &mut scratch, Exec::serial()).unwrap();
                        assert_eq!(y, y2);
                    }
                }
            }

            #[test]
            fn vjp_multiplies_cached_derivative() {
                let n = 2500;
                let x: Vec<T> = grid(n).iter().map(|&v| v as T).collect();
                let dy: Vec<T> = (0..n).map(|i| ((i * 7 % 11) as f64 - 5.0) as T).collect();
                let (mut y, mut d, mut dx) =
                    (vec![0.0 as T; n], vec![0.0 as T; n], vec![0.0 as T; n]);
                let mut scratch = vec![0.0 as T; scratch_len(n, 1)];
                $silu(&x, &mut y, Some(&mut d), &mut scratch, Exec::serial()).unwrap();
                $silu_vjp(&dy, &d, &mut dx, Exec::serial()).unwrap();
                for i in 0..n {
                    assert_eq!(dx[i], dy[i] * d[i]);
                }
                $sig(&x, &mut y, Some(&mut d), &mut scratch, Exec::serial()).unwrap();
                $sig_vjp(&dy, &d, &mut dx, Exec::serial()).unwrap();
                for i in 0..n {
                    assert_eq!(dx[i], dy[i] * d[i]);
                }
            }

            #[test]
            fn thread_count_does_not_change_bits() {
                let pool = threadteam::Pool::builder().threads(4).build().unwrap();
                let n = 200_003;
                let x: Vec<T> = (0..n)
                    .map(|i| ((i % 401) as f64 - 200.0) as T * 0.05)
                    .collect();
                let run = |exec: Exec<'_>| {
                    let (mut y, mut d) = (vec![0.0 as T; n], vec![0.0 as T; n]);
                    let mut scratch = vec![0.0 as T; scratch_len(n, 4)];
                    $silu(&x, &mut y, Some(&mut d), &mut scratch, exec).unwrap();
                    (y, d)
                };
                assert_eq!(run(Exec::serial()), run(Exec::new(&pool, 4).unwrap()));
            }

            #[test]
            fn rejects_bad_shapes() {
                let x = [1.0 as T; 4];
                let mut y = [0.0 as T; 4];
                let mut scratch = [0.0 as T; 4];
                assert!(matches!(
                    $silu(&x, &mut y[..3], None, &mut scratch, Exec::serial()),
                    Err(ActivationError::LengthMismatch { name: "y", .. })
                ));
                assert!(matches!(
                    $sig(
                        &x,
                        &mut y,
                        Some(&mut [0.0; 2]),
                        &mut scratch,
                        Exec::serial()
                    ),
                    Err(ActivationError::LengthMismatch { name: "deriv", .. })
                ));
                assert_eq!(
                    $silu(&x, &mut y, None, &mut [], Exec::serial()),
                    Err(ActivationError::ScratchTooSmall { need: 4, got: 0 })
                );
                assert!(matches!(
                    $silu_vjp(&x, &x[..3], &mut y, Exec::serial()),
                    Err(ActivationError::LengthMismatch { name: "deriv", .. })
                ));
                assert_eq!($silu(&[], &mut [], None, &mut [], Exec::serial()), Ok(()));
            }
        }
    };
}

dtype_tests!(
    f32_kernels,
    f32,
    2e-5,
    silu_forward_f32,
    sigmoid_forward_f32,
    silu_vjp_f32,
    sigmoid_vjp_f32
);
dtype_tests!(
    f64_kernels,
    f64,
    1e-8,
    silu_forward_f64,
    sigmoid_forward_f64,
    silu_vjp_f64,
    sigmoid_vjp_f64
);
