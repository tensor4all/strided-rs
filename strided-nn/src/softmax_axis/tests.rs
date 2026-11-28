use core::mem::MaybeUninit;

use strided_basic::Exec;

use super::*;
use crate::{softmax_axis0_f32, softmax_axis0_f64};

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 24.0
        })
        .collect()
}

/// f64 oracle over `[inner, axis_len, outer]`.
fn oracle(x: &[f64], inner: usize, a: usize, outer: usize) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| i + inner * (j + a * o);
            let m = (0..a).map(|j| x[at(j)]).fold(f64::NEG_INFINITY, f64::max);
            let s: f64 = (0..a).map(|j| (x[at(j)] - m).exp()).sum();
            for j in 0..a {
                y[at(j)] = (x[at(j)] - m).exp() / s;
            }
        }
    }
    y
}

fn oracle_vjp(y: &[f64], dy: &[f64], inner: usize, a: usize, outer: usize) -> Vec<f64> {
    let mut dx = vec![0.0; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| i + inner * (j + a * o);
            let d: f64 = (0..a).map(|j| y[at(j)] * dy[at(j)]).sum();
            for j in 0..a {
                dx[at(j)] = y[at(j)] * (dy[at(j)] - d);
            }
        }
    }
    dx
}

macro_rules! fwd {
    ($f:ident, $t:ty, $inner:expr, $a:expr, $outer:expr, $x:expr, $exec:expr, $budget:expr) => {{
        let mut out = vec![MaybeUninit::<$t>::uninit(); $x.len()];
        let mut sc = vec![0 as $t; softmax_axis_scratch_len::<$t>(sh($inner, $a, $outer), $budget)];
        $f(sh($inner, $a, $outer), $x, &mut out, &mut sc, $exec).unwrap();
        // SAFETY: the call returned `Ok` for a non-empty input, so every slot is written.
        out.iter()
            .map(|v| unsafe { v.assume_init() })
            .collect::<Vec<$t>>()
    }};
}

fn sh(inner: usize, axis_len: usize, outer: usize) -> AxisShape {
    AxisShape {
        inner,
        axis_len,
        outer,
    }
}

const SHAPES: &[(usize, usize, usize)] = &[
    (1, 7, 5),
    (1, 512, 3),
    (2, 3, 4),
    (16, 64, 3),
    (17, 33, 5),
    (33, 1, 2),
    (100, 513, 2),
    (1000, 17, 3),
    (5, 4096, 2),
];

#[test]
fn forward_matches_oracle_and_axis0_on_the_overlap() {
    for &(inner, a, outer) in SHAPES {
        let x = data(inner * a * outer, 7 + inner as u64);
        let want = oracle(&x, inner, a, outer);
        let got = fwd!(
            softmax_axis_f64,
            f64,
            inner,
            a,
            outer,
            &x,
            Exec::serial(),
            1
        );
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-14 * w + 1e-300,
                "f64 {inner},{a},{outer}: {g} {w}"
            );
        }
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let got32 = fwd!(
            softmax_axis_f32,
            f32,
            inner,
            a,
            outer,
            &x32,
            Exec::serial(),
            1
        );
        let want32 = oracle(
            &x32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            inner,
            a,
            outer,
        );
        for (g, w) in got32.iter().zip(&want32) {
            assert!(
                (f64::from(*g) - w).abs() <= 1e-5 * w + 1e-37,
                "f32 {inner},{a},{outer}: {g} {w}"
            );
        }
        if inner == 1 {
            // Contiguous axis: bit-identical to the axis-0 kernel.
            let mut out = vec![MaybeUninit::<f32>::uninit(); x32.len()];
            let mut sc = vec![0f32; crate::scratch_len::<f32>(a, x32.len(), 1)];
            softmax_axis0_f32(a, &x32, &mut out, &mut sc, Exec::serial()).unwrap();
            for (g, o) in got32.iter().zip(&out) {
                assert_eq!(g.to_bits(), unsafe { o.assume_init() }.to_bits());
            }
        }
    }
}

#[test]
fn strided_f64_is_bit_identical_to_axis0_on_the_transposed_layout() {
    // Softmax over a strided axis of [inner, a, outer] equals the axis-0 kernel on the same
    // columns gathered contiguously, bit for bit in f64 (source-order sum, same exp).
    for &(inner, a, outer) in &[
        (16usize, 64usize, 3usize),
        (17, 33, 5),
        (100, 513, 2),
        (5, 4096, 2),
    ] {
        let x = data(inner * a * outer, 99);
        let got = fwd!(
            softmax_axis_f64,
            f64,
            inner,
            a,
            outer,
            &x,
            Exec::serial(),
            1
        );
        let mut cols = Vec::new();
        for o in 0..outer {
            for i in 0..inner {
                for j in 0..a {
                    cols.push(x[i + inner * (j + a * o)]);
                }
            }
        }
        let mut out = vec![MaybeUninit::<f64>::uninit(); cols.len()];
        let mut sc = vec![0f64; crate::scratch_len::<f64>(a, cols.len(), 1)];
        softmax_axis0_f64(a, &cols, &mut out, &mut sc, Exec::serial()).unwrap();
        let mut k = 0;
        for o in 0..outer {
            for i in 0..inner {
                for j in 0..a {
                    let want = unsafe { out[k].assume_init() };
                    assert_eq!(
                        got[i + inner * (j + a * o)].to_bits(),
                        want.to_bits(),
                        "{inner},{a},{outer} at {i},{j},{o}"
                    );
                    k += 1;
                }
            }
        }
    }
}

#[test]
fn threads_do_not_change_the_result() {
    let (inner, a, outer) = (300usize, 129usize, 7usize);
    let x = data(inner * a * outer, 5);
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let serial = fwd!(
        softmax_axis_f32,
        f32,
        inner,
        a,
        outer,
        &x32,
        Exec::serial(),
        1
    );
    let pool = threadteam::Pool::builder().threads(4).build().unwrap();
    let exec = Exec::new(&pool, 4).unwrap();
    let par = fwd!(softmax_axis_f32, f32, inner, a, outer, &x32, exec, 4);
    assert!(serial
        .iter()
        .zip(&par)
        .all(|(s, p)| s.to_bits() == p.to_bits()));
}

#[test]
fn vjp_matches_oracle_and_rows_sum_to_zero() {
    for &(inner, a, outer) in SHAPES {
        let n = inner * a * outer;
        let x = data(n, 3);
        let y = oracle(&x, inner, a, outer);
        let dy = data(n, 11);
        let want = oracle_vjp(&y, &dy, inner, a, outer);
        let mut dx = vec![0.0f64; n];
        let mut sc = vec![0.0f64; softmax_vjp_axis_scratch_len::<f64>(sh(inner, a, outer), 1)];
        softmax_vjp_axis_f64(
            sh(inner, a, outer),
            &y,
            &dy,
            &mut dx,
            &mut sc,
            Exec::serial(),
        )
        .unwrap();
        for (g, w) in dx.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-13 * (1.0 + w.abs()),
                "f64 vjp {inner},{a},{outer}: {g} {w}"
            );
        }
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let dy32: Vec<f32> = dy.iter().map(|&v| v as f32).collect();
        let want32 = oracle_vjp(
            &y32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &dy32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            inner,
            a,
            outer,
        );
        let mut dx32 = vec![0.0f32; n];
        let mut sc32 = vec![0.0f32; softmax_vjp_axis_scratch_len::<f32>(sh(inner, a, outer), 1)];
        softmax_vjp_axis_f32(
            sh(inner, a, outer),
            &y32,
            &dy32,
            &mut dx32,
            &mut sc32,
            Exec::serial(),
        )
        .unwrap();
        for (g, w) in dx32.iter().zip(&want32) {
            assert!(
                (f64::from(*g) - w).abs() <= 1e-5 * (1.0 + w.abs()),
                "f32 vjp {inner},{a},{outer}: {g} {w}"
            );
        }
    }
}

#[test]
fn vjp_threads_do_not_change_the_result() {
    for &(inner, a, outer) in &[(1usize, 129usize, 40usize), (300, 129, 7)] {
        let n = inner * a * outer;
        let y: Vec<f32> = oracle(&data(n, 3), inner, a, outer)
            .iter()
            .map(|&v| v as f32)
            .collect();
        let dy: Vec<f32> = data(n, 11).iter().map(|&v| v as f32).collect();
        let run = |exec: Exec<'_>, budget: usize| {
            let mut dx = vec![0.0f32; n];
            let mut sc =
                vec![0.0f32; softmax_vjp_axis_scratch_len::<f32>(sh(inner, a, outer), budget)];
            softmax_vjp_axis_f32(sh(inner, a, outer), &y, &dy, &mut dx, &mut sc, exec).unwrap();
            dx
        };
        let pool = threadteam::Pool::builder().threads(4).build().unwrap();
        let (s, p) = (run(Exec::serial(), 1), run(Exec::new(&pool, 4).unwrap(), 4));
        assert!(s.iter().zip(&p).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[test]
fn errors_and_empty() {
    let x = [0.0f32; 12];
    let mut out = [MaybeUninit::<f32>::uninit(); 12];
    let mut sc = [0.0f32; 64];
    let e = Exec::serial;
    assert_eq!(
        softmax_axis_f32(sh(2, 0, 6), &x, &mut out, &mut sc, e()),
        Err(SoftmaxError::EmptyAxis)
    );
    assert!(matches!(
        softmax_axis_f32(sh(2, 3, 3), &x, &mut out, &mut sc, e()),
        Err(SoftmaxError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        softmax_axis_f32(sh(2, 3, 2), &x, &mut out[..11], &mut sc, e()),
        Err(SoftmaxError::LengthMismatch { .. })
    ));
    assert!(matches!(
        softmax_axis_f32(sh(2, 3, 2), &x, &mut out, &mut sc[..0], e()),
        Err(SoftmaxError::ScratchTooSmall { .. })
    ));
    assert_eq!(
        softmax_axis_f32(sh(0, 3, 2), &[], &mut [], &mut sc, e()),
        Ok(())
    );
}

#[test]
fn extreme_logits_stay_finite() {
    let (inner, a) = (20usize, 8usize);
    let mut x = vec![0.0f32; inner * a];
    for i in 0..inner {
        x[i] = 1.0e30;
        x[i + inner] = -1.0e30;
        x[i + 2 * inner] = f32::NEG_INFINITY;
    }
    let y = fwd!(softmax_axis_f32, f32, inner, a, 1, &x, Exec::serial(), 1);
    for i in 0..inner {
        assert_eq!(y[i], 1.0);
        assert_eq!(y[i + inner], 0.0);
    }
}

#[test]
fn strided_f32_differs_from_axis0_only_by_summation_order() {
    // Same columns, gathered contiguously: only the sum order differs (lane order vs source
    // order), so the results agree to a few ulp of the probability.
    for &(inner, a, outer) in &[(16usize, 64usize, 3usize), (17, 513, 2), (100, 4096, 1)] {
        let mut worst = 0.0f64;
        let (mut err_strided, mut err_axis0) = (0.0f64, 0.0f64);
        let x: Vec<f32> = data(inner * a * outer, 21)
            .iter()
            .map(|&v| v as f32)
            .collect();
        let got = fwd!(
            softmax_axis_f32,
            f32,
            inner,
            a,
            outer,
            &x,
            Exec::serial(),
            1
        );
        let mut cols = Vec::new();
        for o in 0..outer {
            for i in 0..inner {
                cols.extend((0..a).map(|j| x[i + inner * (j + a * o)]));
            }
        }
        let mut out = vec![MaybeUninit::<f32>::uninit(); cols.len()];
        let mut sc = vec![0f32; crate::scratch_len::<f32>(a, cols.len(), 1)];
        softmax_axis0_f32(a, &cols, &mut out, &mut sc, Exec::serial()).unwrap();
        let truth = oracle(
            &x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            inner,
            a,
            outer,
        );
        let mut k = 0;
        for o in 0..outer {
            for i in 0..inner {
                for j in 0..a {
                    // SAFETY: `softmax_axis0_f32` wrote every slot.
                    let want = unsafe { out[k].assume_init() };
                    let g = got[i + inner * (j + a * o)];
                    worst = worst
                        .max(f64::from((g - want).abs()) / f64::from(want.max(f32::MIN_POSITIVE)));
                    let t = truth[i + inner * (j + a * o)];
                    err_strided = err_strided.max((f64::from(g) - t).abs() / t);
                    err_axis0 = err_axis0.max((f64::from(want) - t).abs() / t);
                    k += 1;
                }
            }
        }
        println!("{inner},{a},{outer}: strided-vs-axis0 {worst:e}; vs f64 oracle strided {err_strided:e}, axis0 {err_axis0:e}");
        assert!(worst < 1e-5, "{worst}");
    }
}
