//! The parallel branches of the strided element-wise entries (`add`, `mul`, `axpy`, `fma`,
//! `sum`, `copy_transpose_scale_into`) agree with the serial ones. The pool of
//! `common::shared_pool` has 100 ns dispatch costs, so regions of tens of thousands of
//! elements open the parallel branches.

mod common;

use common::Budget;
use strided_basic::{
    add, axpy, copy_transpose_scale_into, fma, mul, sum, StridedArray, StridedView,
};

const R: usize = 300;
const C: usize = 260;

fn src_f64(row_major: bool) -> StridedArray<f64> {
    let f = |idx: &[usize]| (idx[0] * 7 + idx[1] * 3) as f64 * 0.5 + 1.0;
    if row_major {
        StridedArray::from_fn_row_major(&[R, C], f)
    } else {
        StridedArray::from_fn_col_major(&[R, C], f)
    }
}

fn values(a: &StridedArray<f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(R * C);
    for i in 0..a.dims()[0] {
        for j in 0..a.dims()[1] {
            v.push(a.get(&[i, j]));
        }
    }
    v
}

/// Runs `op` with a serial token and a four-slot token on identical destinations and compares.
fn same_result(op: impl Fn(&mut StridedArray<f64>, Budget)) {
    for dst_row_major in [true, false] {
        let make = || {
            let f = |idx: &[usize]| (idx[0] + 2 * idx[1]) as f64;
            if dst_row_major {
                StridedArray::<f64>::from_fn_row_major(&[R, C], f)
            } else {
                StridedArray::<f64>::from_fn_col_major(&[R, C], f)
            }
        };
        let mut serial = make();
        let mut parallel = make();
        op(&mut serial, Budget::Serial);
        op(&mut parallel, Budget::Pooled(4));
        assert_eq!(values(&serial), values(&parallel));
    }
}

#[test]
fn add_mul_axpy_fma_match_the_serial_result() {
    for src_row_major in [true, false] {
        let a = src_f64(src_row_major);
        let b = src_f64(!src_row_major);
        // A source whose layout differs from the destination's takes the strided path.
        let (av, bv): (StridedView<f64>, StridedView<f64>) = (a.view(), b.view());
        same_result(|d, bud| add(&mut d.view_mut(), &av, bud.exec()).unwrap());
        same_result(|d, bud| mul(&mut d.view_mut(), &av, bud.exec()).unwrap());
        same_result(|d, bud| axpy(&mut d.view_mut(), &av, 1.5, bud.exec()).unwrap());
        same_result(|d, bud| fma(&mut d.view_mut(), &av, &bv, bud.exec()).unwrap());
    }
}

#[test]
fn sum_of_a_contiguous_and_a_strided_view_matches() {
    let big = StridedArray::<f64>::from_fn_row_major(&[1 << 18], |idx| (idx[0] % 17) as f64);
    let s0 = sum(&big.view(), Budget::Serial.exec()).unwrap();
    let s1 = sum(&big.view(), Budget::Pooled(4).exec()).unwrap();
    assert_eq!(s0, s1);
    let m = src_f64(true);
    let t = m.view().permute(&[1, 0]).unwrap();
    assert_eq!(
        sum(&t, Budget::Serial.exec()).unwrap(),
        sum(&t, Budget::Pooled(4).exec()).unwrap()
    );
}

fn transpose_scale<T>(scale: T, make: impl Fn(&[usize]) -> T + Copy)
where
    T: Copy
        + strided_basic::ElementOpApply
        + std::ops::Mul<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq
        + Default
        + Send
        + Sync
        + std::fmt::Debug
        + 'static,
{
    for src_row_major in [true, false] {
        for dst_row_major in [true, false] {
            let src = if src_row_major {
                StridedArray::<T>::from_fn_row_major(&[R, C], make)
            } else {
                StridedArray::<T>::from_fn_col_major(&[R, C], make)
            };
            let run = |bud: Budget| {
                let mut dst = if dst_row_major {
                    StridedArray::<T>::from_fn_row_major(&[C, R], |_| make(&[1, 1]))
                } else {
                    StridedArray::<T>::from_fn_col_major(&[C, R], |_| make(&[1, 1]))
                };
                copy_transpose_scale_into(&mut dst.view_mut(), &src.view(), scale, bud.exec())
                    .unwrap();
                let mut v = Vec::with_capacity(R * C);
                for i in 0..C {
                    for j in 0..R {
                        v.push(dst.get(&[i, j]));
                    }
                }
                v
            };
            assert_eq!(run(Budget::Serial), run(Budget::Pooled(4)));
        }
    }
}

#[test]
fn copy_transpose_scale_matches_for_every_layout_and_scale() {
    for scale in [2.0_f64, 1.0, 0.0] {
        transpose_scale(scale, |idx| (idx[0] * 5 + idx[1]) as f64);
    }
    for scale in [2.0_f32, 1.0, 0.0] {
        transpose_scale(scale, |idx| (idx[0] * 5 + idx[1]) as f32);
    }
    for scale in [3_i64, 1, 0] {
        transpose_scale(scale, |idx| (idx[0] * 5 + idx[1]) as i64);
    }
}
