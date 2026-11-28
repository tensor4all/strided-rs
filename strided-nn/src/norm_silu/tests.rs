use strided_basic::Exec;

use crate::activation as act;
use crate::norm as nm;

fn rand(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
        })
        .collect()
}

// Widths straddle the SIMD lane counts and the 1024-element epilogue chunk.
const WIDTHS: [usize; 11] = [1, 2, 7, 8, 9, 33, 257, 1023, 1024, 1025, 2049];
const ROWS: [usize; 4] = [1, 2, 5, 131];

macro_rules! suite {
    ($m:ident, $t:ty,
     $f_ln:ident, $f_ln_in:ident, $f_ln_par:ident, $f_rms:ident, $f_rms_in:ident, $f_rms_w:ident,
     $a_ln:ident, $a_ln_in:ident, $a_ln_par:ident, $a_rms:ident, $a_rms_in:ident, $a_rms_w:ident,
     $silu:ident, $silu_vjp:ident) => {
        mod $m {
            use super::*;
            use crate::norm_silu::*;

            fn vec(n: usize, seed: u64) -> Vec<$t> {
                rand(n, seed).into_iter().map(|v| v as $t).collect()
            }
            fn bits(v: &[$t]) -> Vec<u64> {
                v.iter().map(|&x| u64::from(x.to_bits())).collect()
            }
            const EPS: $t = 1e-5;

            /// Fused forward + VJP versus the two-pass composition, compared bit for bit.
            fn check(layer: bool, rows: usize, width: usize, aff: (bool, bool), mut exec: Exec<'_>) {
                let n = rows * width;
                let x = vec(n, 1);
                let w = vec(width, 2);
                let b = vec(width, 3);
                let dy = vec(n, 4);
                let (w, b) = (aff.0.then_some(&w[..]), (aff.1 && layer).then_some(&b[..]));
                let ctx = format!("layer={layer} rows={rows} width={width} aff={aff:?}");

                // Two-pass reference: A1 into an intermediate, then A2.
                let mut z = vec![0.0 as $t; n];
                let (mut m0, mut s0) = (vec![0.0 as $t; rows], vec![0.0 as $t; rows]);
                let mut y0 = vec![0.0 as $t; n];
                let mut d0 = vec![0.0 as $t; n];
                if layer {
                    nm::$a_ln(width, EPS, &x, w, b, &mut z, Some((&mut m0, &mut s0)), Exec::serial())
                        .unwrap();
                } else {
                    nm::$a_rms(width, EPS, &x, w, &mut z, Some(&mut s0), Exec::serial()).unwrap();
                }
                let mut sc = vec![0.0 as $t; act::scratch_len(n, 1)];
                act::$silu(&z, &mut y0, Some(&mut d0), &mut sc, Exec::serial()).unwrap();

                // Fused forward.
                let mut y1 = vec![0.0 as $t; n];
                let mut d1 = vec![0.0 as $t; n];
                let (mut m1, mut s1) = (vec![0.0 as $t; rows], vec![0.0 as $t; rows]);
                let mut sc = vec![0.0 as $t; norm_silu_scratch_len(rows, width, 8).unwrap()];
                if layer {
                    $f_ln(width, EPS, &x, w, b, &mut y1, Some((&mut m1, &mut s1)), Some(&mut d1), &mut sc, exec.reborrow())
                        .unwrap();
                } else {
                    $f_rms(width, EPS, &x, w, &mut y1, Some(&mut s1), Some(&mut d1), &mut sc, exec.reborrow())
                        .unwrap();
                }
                assert_eq!(bits(&y0), bits(&y1), "y {ctx}");
                assert_eq!(bits(&d0), bits(&d1), "deriv {ctx}");
                assert_eq!(bits(&s0), bits(&s1), "inv {ctx}");
                if layer {
                    assert_eq!(bits(&m0), bits(&m1), "mean {ctx}");
                }

                // Forward without the optional caches gives the same y.
                let mut y2 = vec![0.0 as $t; n];
                if layer {
                    $f_ln(width, EPS, &x, w, b, &mut y2, None, None, &mut sc, exec.reborrow()).unwrap();
                } else {
                    $f_rms(width, EPS, &x, w, &mut y2, None, None, &mut sc, exec.reborrow()).unwrap();
                }
                assert_eq!(bits(&y0), bits(&y2), "y(no cache) {ctx}");

                // VJP: A2 vjp then A1 input / parameter VJPs.
                let mut dz = vec![0.0 as $t; n];
                act::$silu_vjp(&dy, &d0, &mut dz, Exec::serial()).unwrap();
                let (mut dx0, mut dx1) = (vec![0.0 as $t; n], vec![0.0 as $t; n]);
                let planes = if layer { 2 } else { 1 };
                let plen = nm::layer_norm_params_scratch_len(rows, width).unwrap();
                let (mut dw0, mut dw1) = (vec![9.0 as $t; width], vec![9.0 as $t; width]);
                let (mut db0, mut db1) = (vec![9.0 as $t; width], vec![9.0 as $t; width]);
                let mut psc = vec![0.0 as $t; plen];
                if layer {
                    nm::$a_ln_in(width, &x, &m0, &s0, w, &dz, &mut dx0, Exec::serial()).unwrap();
                    nm::$a_ln_par(width, &x, &m0, &s0, &dz, Some(&mut dw0), Some(&mut db0), &mut psc, Exec::serial())
                        .unwrap();
                    $f_ln_in(width, &x, &m1, &s1, w, &d1, &dy, &mut dx1, exec.reborrow()).unwrap();
                    $f_ln_par(width, &x, &m1, &s1, &d1, &dy, Some(&mut dw1), Some(&mut db1), &mut psc, exec.reborrow())
                        .unwrap();
                    assert_eq!(bits(&db0), bits(&db1), "dbias {ctx}");
                } else {
                    nm::$a_rms_in(width, &x, &s0, w, &dz, &mut dx0, Exec::serial()).unwrap();
                    nm::$a_rms_w(width, &x, &s0, &dz, &mut dw0, &mut psc, Exec::serial()).unwrap();
                    $f_rms_in(width, &x, &s1, w, &d1, &dy, &mut dx1, exec.reborrow()).unwrap();
                    $f_rms_w(width, &x, &s1, &d1, &dy, &mut dw1, &mut psc, exec.reborrow()).unwrap();
                }
                let _ = planes;
                assert_eq!(bits(&dx0), bits(&dx1), "dx {ctx}");
                assert_eq!(bits(&dw0), bits(&dw1), "dweight {ctx}");
            }

            #[test]
            fn bit_identical_to_two_pass_serial() {
                for layer in [true, false] {
                    for &rows in &ROWS {
                        for &width in &WIDTHS {
                            for aff in [(false, false), (true, false), (false, true), (true, true)] {
                                check(layer, rows, width, aff, Exec::serial());
                            }
                        }
                    }
                }
            }

            #[test]
            fn bit_identical_to_two_pass_four_threads() {
                let pool = threadteam::Pool::builder().threads(4).build().unwrap();
                for layer in [true, false] {
                    for &rows in &[5usize, 131] {
                        for &width in &[7usize, 257, 1025] {
                            check(layer, rows, width, (true, true), Exec::new(&pool, 4).unwrap());
                        }
                    }
                }
            }

            #[test]
            fn errors_before_output() {
                let x = vec![1.0 as $t; 6];
                let mut y = vec![7.0 as $t; 6];
                let mut sc = vec![0.0 as $t; 64];
                let mut e = Exec::serial();
                assert_eq!(
                    $f_rms(0, EPS, &x, None, &mut y, None, None, &mut sc, e.reborrow()),
                    Err(crate::NormError::EmptyWidth)
                );
                assert!(matches!(
                    $f_rms(4, EPS, &x, None, &mut y, None, None, &mut sc, e.reborrow()),
                    Err(crate::NormError::UnevenRows { .. })
                ));
                assert!(matches!(
                    $f_rms(3, EPS, &x, None, &mut y, None, Some(&mut [0.0; 5]), &mut sc, e.reborrow()),
                    Err(crate::NormError::LengthMismatch { name: "deriv", .. })
                ));
                assert!(matches!(
                    $f_rms(3, EPS, &x, None, &mut y, None, None, &mut sc[..5], e.reborrow()),
                    Err(crate::NormError::ScratchTooSmall { need: 6, got: 5 })
                ));
                assert!(y.iter().all(|&v| v == 7.0));
            }
        }
    };
}

suite!(
    f32_suite, f32,
    layer_norm_silu_forward_f32, layer_norm_silu_vjp_input_f32, layer_norm_silu_vjp_params_f32,
    rms_norm_silu_forward_f32, rms_norm_silu_vjp_input_f32, rms_norm_silu_vjp_weight_f32,
    layer_norm_forward_f32, layer_norm_vjp_input_f32, layer_norm_vjp_params_f32,
    rms_norm_forward_f32, rms_norm_vjp_input_f32, rms_norm_vjp_weight_f32,
    silu_forward_f32, silu_vjp_f32
);
suite!(
    f64_suite, f64,
    layer_norm_silu_forward_f64, layer_norm_silu_vjp_input_f64, layer_norm_silu_vjp_params_f64,
    rms_norm_silu_forward_f64, rms_norm_silu_vjp_input_f64, rms_norm_silu_vjp_weight_f64,
    layer_norm_forward_f64, layer_norm_vjp_input_f64, layer_norm_vjp_params_f64,
    rms_norm_forward_f64, rms_norm_vjp_input_f64, rms_norm_vjp_weight_f64,
    silu_forward_f64, silu_vjp_f64
);
