use strided_basic::Exec;

use super::*;

fn rand(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

const EPS: f64 = 1e-5;

/// Naive definition-level reference in f64: `(y, mean, inv_std)`; `layer = false` is RMSNorm
/// (`mean` is then zero and `inv_std` is `inv_rms`).
fn reference(
    layer: bool,
    width: usize,
    x: &[f64],
    w: Option<&[f64]>,
    b: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut y, mut means, mut invs) = (vec![], vec![], vec![]);
    for row in x.chunks(width) {
        let n = width as f64;
        let mean = if layer {
            row.iter().sum::<f64>() / n
        } else {
            0.0
        };
        let denom = if layer {
            row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
        } else {
            row.iter().map(|v| v * v).sum::<f64>() / n
        };
        let inv = 1.0 / (denom + EPS).sqrt();
        for (j, v) in row.iter().enumerate() {
            let mut o = (v - mean) * inv;
            if let Some(w) = w {
                o *= w[j];
            }
            if let Some(b) = b {
                o += b[j];
            }
            y.push(o);
        }
        means.push(mean);
        invs.push(inv);
    }
    (y, means, invs)
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (p, q)) in a.iter().zip(b).enumerate() {
        assert!(
            (p - q).abs() <= tol * (1.0 + q.abs()),
            "elem {i}: {p} vs {q}"
        );
    }
}

const WIDTHS: [usize; 9] = [1, 2, 3, 7, 8, 9, 33, 64, 257];

/// Central finite difference of `loss(x, w, b) = sum(c * norm(x, w, b))` along each input.
fn fd(
    layer: bool,
    width: usize,
    x: &[f64],
    w: Option<&[f64]>,
    b: Option<&[f64]>,
    c: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let loss = |x: &[f64], w: Option<&[f64]>, b: Option<&[f64]>| {
        let (y, _, _) = reference(layer, width, x, w, b);
        y.iter().zip(c).map(|(a, b)| a * b).sum::<f64>()
    };
    let h = 1e-6;
    let diff = |v: &[f64], f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> {
        (0..v.len())
            .map(|i| {
                let (mut p, mut m) = (v.to_vec(), v.to_vec());
                p[i] += h;
                m[i] -= h;
                (f(&p) - f(&m)) / (2.0 * h)
            })
            .collect()
    };
    let dx = diff(x, &|x| loss(x, w, b));
    let dw = w.map_or(vec![], |w| diff(w, &|w| loss(x, Some(w), b)));
    let db = b.map_or(vec![], |b| diff(b, &|b| loss(x, w, Some(b))));
    (dx, dw, db)
}

macro_rules! dtype_tests {
    ($m:ident, $t:ty, $tol:expr, $ln_f:ident, $ln_i:ident, $ln_p:ident, $rms_f:ident, $rms_i:ident, $rms_w:ident) => {
        mod $m {
            use super::*;
            type T = $t;
            fn cast(v: &[f64]) -> Vec<T> {
                v.iter().map(|&a| a as T).collect()
            }
            fn up(v: &[T]) -> Vec<f64> {
                v.iter().map(|&a| a as f64).collect()
            }
            fn cases() -> Vec<(usize, bool, bool)> {
                WIDTHS
                    .iter()
                    .flat_map(|&w| {
                        [
                            (w, false, false),
                            (w, true, false),
                            (w, false, true),
                            (w, true, true),
                        ]
                    })
                    .collect()
            }

            #[test]
            fn layer_norm_forward_and_stats() {
                for (width, aw, ab) in cases() {
                    let x = rand(5 * width, 1);
                    let w = rand(width, 2);
                    let b = rand(width, 3);
                    let (wr, br) = (aw.then_some(&w[..]), ab.then_some(&b[..]));
                    let (want, wm, wi) = reference(true, width, &x, wr, br);
                    let (xt, wt, bt) = (cast(&x), cast(&w), cast(&b));
                    let mut y = vec![0.0 as T; x.len()];
                    let (mut m, mut s) = (vec![0.0 as T; 5], vec![0.0 as T; 5]);
                    $ln_f(
                        width,
                        EPS as T,
                        &xt,
                        aw.then_some(&wt[..]),
                        ab.then_some(&bt[..]),
                        &mut y,
                        Some((&mut m, &mut s)),
                        Exec::serial(),
                    )
                    .unwrap();
                    close(&up(&y), &want, $tol);
                    close(&up(&m), &wm, $tol);
                    close(&up(&s), &wi, $tol);
                    // Without stats the output is the same.
                    let mut y2 = vec![0.0 as T; x.len()];
                    $ln_f(
                        width,
                        EPS as T,
                        &xt,
                        aw.then_some(&wt[..]),
                        ab.then_some(&bt[..]),
                        &mut y2,
                        None,
                        Exec::serial(),
                    )
                    .unwrap();
                    assert_eq!(y, y2);
                }
            }

            #[test]
            fn rms_norm_forward_and_stats() {
                for (width, aw, _) in cases() {
                    let x = rand(4 * width, 4);
                    let w = rand(width, 5);
                    let (want, _, wi) = reference(false, width, &x, aw.then_some(&w[..]), None);
                    let (xt, wt) = (cast(&x), cast(&w));
                    let mut y = vec![0.0 as T; x.len()];
                    let mut s = vec![0.0 as T; 4];
                    $rms_f(
                        width,
                        EPS as T,
                        &xt,
                        aw.then_some(&wt[..]),
                        &mut y,
                        Some(&mut s),
                        Exec::serial(),
                    )
                    .unwrap();
                    close(&up(&y), &want, $tol);
                    close(&up(&s), &wi, $tol);
                }
            }

            #[test]
            fn layer_norm_vjp_matches_finite_differences() {
                for width in [1, 3, 9, 16] {
                    for (aw, ab) in [(false, false), (true, false), (false, true), (true, true)] {
                        let rows = 3;
                        let x = rand(rows * width, 6);
                        let w = rand(width, 7);
                        let b = rand(width, 8);
                        let c = rand(rows * width, 9);
                        let (wr, br) = (aw.then_some(&w[..]), ab.then_some(&b[..]));
                        let (dx_fd, dw_fd, db_fd) = fd(true, width, &x, wr, br, &c);
                        let (_, mean, inv) = reference(true, width, &x, wr, br);
                        let (xt, wt, ct) = (cast(&x), cast(&w), cast(&c));
                        let (mt, it) = (cast(&mean), cast(&inv));
                        let mut dx = vec![0.0 as T; x.len()];
                        $ln_i(
                            width,
                            &xt,
                            &mt,
                            &it,
                            aw.then_some(&wt[..]),
                            &ct,
                            &mut dx,
                            Exec::serial(),
                        )
                        .unwrap();
                        close(&up(&dx), &dx_fd, 50.0 * $tol);
                        let mut dw = vec![9.0 as T; width];
                        let mut db = vec![9.0 as T; width];
                        let mut scratch =
                            vec![0.0 as T; layer_norm_params_scratch_len(rows, width).unwrap()];
                        $ln_p(
                            width,
                            &xt,
                            &mt,
                            &it,
                            &ct,
                            aw.then_some(&mut dw[..]),
                            ab.then_some(&mut db[..]),
                            &mut scratch,
                            Exec::serial(),
                        )
                        .unwrap();
                        if aw {
                            close(&up(&dw), &dw_fd, 50.0 * $tol);
                        }
                        if ab {
                            close(&up(&db), &db_fd, 50.0 * $tol);
                        }
                    }
                }
            }

            #[test]
            fn rms_norm_vjp_matches_finite_differences() {
                for width in [1, 3, 9, 16] {
                    for aw in [false, true] {
                        let rows = 3;
                        let x = rand(rows * width, 10);
                        let w = rand(width, 11);
                        let c = rand(rows * width, 12);
                        let wr = aw.then_some(&w[..]);
                        let (dx_fd, dw_fd, _) = fd(false, width, &x, wr, None, &c);
                        let (_, _, inv) = reference(false, width, &x, wr, None);
                        let (xt, wt, ct, it) = (cast(&x), cast(&w), cast(&c), cast(&inv));
                        let mut dx = vec![0.0 as T; x.len()];
                        $rms_i(
                            width,
                            &xt,
                            &it,
                            aw.then_some(&wt[..]),
                            &ct,
                            &mut dx,
                            Exec::serial(),
                        )
                        .unwrap();
                        close(&up(&dx), &dx_fd, 50.0 * $tol);
                        if aw {
                            let mut dw = vec![9.0 as T; width];
                            let mut scratch =
                                vec![0.0 as T; rms_norm_weight_scratch_len(rows, width).unwrap()];
                            $rms_w(width, &xt, &it, &ct, &mut dw, &mut scratch, Exec::serial())
                                .unwrap();
                            close(&up(&dw), &dw_fd, 50.0 * $tol);
                        }
                    }
                }
            }

            #[test]
            fn params_reduce_over_many_chunks() {
                // 300 rows span three 128-row chunks; width 70 spans two column blocks.
                let (rows, width) = (300, 70);
                let x = rand(rows * width, 13);
                let dy = rand(rows * width, 14);
                let (_, mean, inv) = reference(true, width, &x, None, None);
                let (mut want_w, mut want_b) = (vec![0.0; width], vec![0.0; width]);
                for r in 0..rows {
                    for j in 0..width {
                        let g = dy[r * width + j];
                        want_w[j] += g * (x[r * width + j] - mean[r]) * inv[r];
                        want_b[j] += g;
                    }
                }
                let (xt, dt, mt, it) = (cast(&x), cast(&dy), cast(&mean), cast(&inv));
                let (mut dw, mut db) = (vec![0.0 as T; width], vec![0.0 as T; width]);
                let mut scratch =
                    vec![0.0 as T; layer_norm_params_scratch_len(rows, width).unwrap()];
                $ln_p(
                    width,
                    &xt,
                    &mt,
                    &it,
                    &dt,
                    Some(&mut dw),
                    Some(&mut db),
                    &mut scratch,
                    Exec::serial(),
                )
                .unwrap();
                close(&up(&dw), &want_w, 100.0 * $tol);
                close(&up(&db), &want_b, 100.0 * $tol);
            }

            #[test]
            fn zero_rows_overwrite_requested_outputs_with_zero() {
                let width = 5;
                let mut scratch = vec![0.0 as T; 1];
                let (mut dw, mut db) = (vec![7.0 as T; width], vec![7.0 as T; width]);
                $ln_p(
                    width,
                    &[],
                    &[],
                    &[],
                    &[],
                    Some(&mut dw),
                    Some(&mut db),
                    &mut scratch,
                    Exec::serial(),
                )
                .unwrap();
                assert!(dw.iter().chain(&db).all(|&v| v == 0.0));

                // Only one output requested: the other stays untouched.
                let (dw, mut db) = (vec![7.0 as T; width], vec![7.0 as T; width]);
                $ln_p(
                    width,
                    &[],
                    &[],
                    &[],
                    &[],
                    None,
                    Some(&mut db),
                    &mut scratch,
                    Exec::serial(),
                )
                .unwrap();
                assert!(db.iter().all(|&v| v == 0.0));
                assert!(dw.iter().all(|&v| v == 7.0));

                let mut dw = vec![7.0 as T; width];
                $rms_w(width, &[], &[], &[], &mut dw, &mut scratch, Exec::serial()).unwrap();
                assert!(dw.iter().all(|&v| v == 0.0));
            }

            #[test]
            fn thread_count_does_not_change_bits() {
                let pool = threadteam::Pool::builder().threads(4).build().unwrap();
                let (rows, width) = (20_000, 33);
                let x = cast(&rand(rows * width, 15));
                let w = cast(&rand(width, 16));
                let b = cast(&rand(width, 17));
                let dy = cast(&rand(rows * width, 18));
                let run = |exec: Exec<'_>| {
                    let mut exec = exec;
                    let mut y = vec![0.0 as T; x.len()];
                    let (mut m, mut s) = (vec![0.0 as T; rows], vec![0.0 as T; rows]);
                    $ln_f(
                        width,
                        EPS as T,
                        &x,
                        Some(&w),
                        Some(&b),
                        &mut y,
                        Some((&mut m, &mut s)),
                        exec.reborrow(),
                    )
                    .unwrap();
                    let mut dx = vec![0.0 as T; x.len()];
                    $ln_i(width, &x, &m, &s, Some(&w), &dy, &mut dx, exec.reborrow()).unwrap();
                    let (mut dw, mut db) = (vec![0.0 as T; width], vec![0.0 as T; width]);
                    let mut scratch =
                        vec![0.0 as T; layer_norm_params_scratch_len(rows, width).unwrap()];
                    $ln_p(
                        width,
                        &x,
                        &m,
                        &s,
                        &dy,
                        Some(&mut dw),
                        Some(&mut db),
                        &mut scratch,
                        exec,
                    )
                    .unwrap();
                    (y, m, s, dx, dw, db)
                };
                assert_eq!(run(Exec::serial()), run(Exec::new(&pool, 4).unwrap()));
            }

            #[test]
            fn rejects_bad_shapes() {
                let x = [1.0 as T; 6];
                let mut y = [0.0 as T; 6];
                let e = Exec::serial;
                assert_eq!(
                    $ln_f(0, 1e-5, &x, None, None, &mut y, None, e()),
                    Err(NormError::EmptyWidth)
                );
                assert_eq!(
                    $ln_f(4, 1e-5, &x, None, None, &mut y, None, e()),
                    Err(NormError::UnevenRows { len: 6, width: 4 })
                );
                assert!(matches!(
                    $ln_f(3, 1e-5, &x, Some(&[1.0; 2]), None, &mut y, None, e()),
                    Err(NormError::LengthMismatch { name: "weight", .. })
                ));
                assert!(matches!(
                    $rms_f(3, 1e-5, &x, None, &mut y[..5], None, e()),
                    Err(NormError::LengthMismatch { name: "y", .. })
                ));
                let (m, s) = ([0.0 as T; 2], [1.0 as T; 2]);
                let mut dw = [0.0 as T; 3];
                assert_eq!(
                    $ln_p(3, &x, &m, &s, &x, Some(&mut dw), None, &mut [], e()),
                    Err(NormError::ScratchTooSmall { need: 3, got: 0 })
                );
                // Nothing requested: no scratch needed.
                assert_eq!($ln_p(3, &x, &m, &s, &x, None, None, &mut [], e()), Ok(()));
                // Empty input is a no-op once the shape is valid.
                assert_eq!($rms_f(3, 1e-5, &[], None, &mut [], None, e()), Ok(()));
            }
        }
    };
}

dtype_tests!(
    f32_kernels,
    f32,
    2e-5,
    layer_norm_forward_f32,
    layer_norm_vjp_input_f32,
    layer_norm_vjp_params_f32,
    rms_norm_forward_f32,
    rms_norm_vjp_input_f32,
    rms_norm_vjp_weight_f32
);
dtype_tests!(
    f64_kernels,
    f64,
    1e-9,
    layer_norm_forward_f64,
    layer_norm_vjp_input_f64,
    layer_norm_vjp_params_f64,
    rms_norm_forward_f64,
    rms_norm_vjp_input_f64,
    rms_norm_vjp_weight_f64
);
