use super::*;
use crate::{Conj, ExecContext, Identity};
use num_complex::Complex64 as C64;

fn vals(seed: u32, n: usize) -> Vec<C64> {
    (0..n)
        .map(|i| {
            let x = (i as u32 * 7 + seed * 13 + 3) % 17;
            C64::new(x as f64 * 0.31 - 2.0, (x * x % 11) as f64 * 0.17 - 1.0)
        })
        .collect()
}

fn op<O: ElementOp<C64>>(x: C64) -> C64 {
    O::apply(x)
}

/// Check every conjugation combination of the three-operand form against a
/// naive loop, over a reversed A, a broadcast B and a permuted D.
fn check3<OD: ElementOp<C64>, OA: ElementOp<C64>, OB: ElementOp<C64>>() {
    let (m, n) = (3usize, 4usize);
    let dims = [m, n];
    let a = vals(1, m * n);
    let b = vals(2, n);
    let start = vals(3, m * n);
    // A reversed on both axes, B broadcast along axis 0, D row-major.
    let av =
        StridedView::<C64, OA>::new(&a, &dims, &[-1, -(m as isize)], (m * n - 1) as isize).unwrap();
    let bv = StridedView::<C64, OB>::new(&b, &dims, &[0, 1], 0).unwrap();
    let ds = [n as isize, 1];
    let mut got = start.clone();
    {
        let mut dv = StridedViewMut::new(&mut got, &dims, &ds, 0).unwrap();
        zip_update3_into::<_, _, _, OD, OA, OB>(&mut dv, &av, &bv, |d, a, b| {
            C64::new(0.5, -0.25) * a * b + C64::new(-0.3, 0.2) * d
        })
        .unwrap();
    }
    for i in 0..m {
        for j in 0..n {
            let d = start[i * n + j];
            let ai = a[(m * n - 1) - i - m * j];
            let want = C64::new(0.5, -0.25) * op::<OA>(ai) * op::<OB>(b[j])
                + C64::new(-0.3, 0.2) * op::<OD>(d);
            assert!((got[i * n + j] - want).norm() < 1e-13, "({i}, {j})");
        }
    }
}

#[test]
fn zip_update3_matches_a_naive_loop_for_every_conjugation() {
    check3::<Identity, Identity, Identity>();
    check3::<Identity, Identity, Conj>();
    check3::<Identity, Conj, Identity>();
    check3::<Identity, Conj, Conj>();
    check3::<Conj, Identity, Identity>();
    check3::<Conj, Identity, Conj>();
    check3::<Conj, Conj, Identity>();
    check3::<Conj, Conj, Conj>();
}

#[test]
fn zip_update2_and_map_update_match_a_naive_loop() {
    let dims = [5usize, 4];
    let a = vals(4, 20);
    let start = vals(5, 20);
    let av = StridedView::<C64, Conj>::new(&a, &dims, &[4, 1], 0).unwrap();
    let mut got = start.clone();
    {
        let mut dv = StridedViewMut::new(&mut got, &dims, &[1, 5], 0).unwrap();
        zip_update2_into::<_, _, Conj, Conj>(&mut dv, &av, |d, a| a + d * d).unwrap();
    }
    for i in 0..5 {
        for j in 0..4 {
            let d = start[i + 5 * j].conj();
            let want = a[i * 4 + j].conj() + d * d;
            assert!((got[i + 5 * j] - want).norm() < 1e-13);
        }
    }
    let mut got = start.clone();
    {
        let mut dv = StridedViewMut::new(&mut got, &dims, &[1, 5], 0).unwrap();
        map_update_into::<_, Conj>(&mut dv, |d| d * 2.0).unwrap();
    }
    for (g, s) in got.iter().zip(&start) {
        assert_eq!(*g, s.conj() * 2.0);
    }
}

#[test]
fn threaded_update_equals_serial_and_formula() {
    let (m, n) = (300usize, 300usize); // above the parallel threshold
    let dims = [m, n];
    let a = vals(6, m * n);
    let b = vals(7, m * n);
    let start = vals(8, m * n);
    let run = |ctx: ExecContext| {
        let mut out = start.clone();
        let av = StridedView::<C64, Identity>::new(&a, &dims, &[n as isize, 1], 0).unwrap();
        let bv = StridedView::<C64, Conj>::new(&b, &dims, &[1, m as isize], 0).unwrap();
        ctx.run(|| {
            let mut dv = StridedViewMut::new(&mut out, &dims, &[1, m as isize], 0).unwrap();
            zip_update3_into::<_, _, _, Conj, Identity, Conj>(&mut dv, &av, &bv, |d, a, b| {
                d + a * b
            })
            .unwrap();
        });
        out
    };
    let serial = run(ExecContext::serial());
    let threaded = run(ExecContext::max_threads(4).unwrap());
    assert_eq!(serial, threaded);
    for i in 0..m {
        for j in 0..n {
            let want = start[i + m * j].conj() + a[i * n + j] * b[i + m * j].conj();
            assert_eq!(serial[i + m * j], want);
        }
    }
}

#[test]
fn invalid_calls_are_rejected_before_any_write() {
    let a = [1.0f64; 4];
    let mut d = [9.0f64; 4];
    // A non-injective destination (stride 0 over extent 4).
    let mut dv = StridedViewMut::new(&mut d, &[4], &[0], 0).unwrap();
    assert!(matches!(
        map_update_into::<_, Identity>(&mut dv, |x| x + 1.0),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
    let mut d2 = [9.0f64; 4];
    let av = StridedView::<f64, Identity>::new(&a, &[4], &[1], 0).unwrap();
    let mut dv2 = StridedViewMut::new(&mut d2, &[2, 2], &[1, 2], 0).unwrap();
    assert!(zip_update2_into::<_, _, Identity, Identity>(&mut dv2, &av, |d, a| d + a).is_err());
    assert_eq!(d, [9.0; 4]);
    assert_eq!(d2, [9.0; 4]);
}

#[test]
fn empty_and_scalar_shapes() {
    let a: [f64; 0] = [];
    let mut d: [f64; 0] = [];
    let av = StridedView::<f64, Identity>::new(&a, &[0], &[1], 0).unwrap();
    let mut dv = StridedViewMut::new(&mut d, &[0], &[1], 0).unwrap();
    zip_update2_into::<_, _, Identity, Identity>(&mut dv, &av, |d, a| d + a).unwrap();
    // Rank 0: one element.
    let a = [2.0f64];
    let mut d = [1.0f64];
    let av = StridedView::<f64, Identity>::new(&a, &[], &[], 0).unwrap();
    let mut dv = StridedViewMut::new(&mut d, &[], &[], 0).unwrap();
    zip_update2_into::<_, _, Identity, Identity>(&mut dv, &av, |d, a| d + 3.0 * a).unwrap();
    assert_eq!(d, [7.0]);
}
