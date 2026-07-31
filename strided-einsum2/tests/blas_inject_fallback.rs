#![cfg(feature = "blas-inject")]

use std::os::raw::c_char;
use std::sync::Once;

use cblas_inject::{register_dgemm, BlasInt32};
use num_complex::Complex64;
use std::mem::MaybeUninit;
use strided_einsum2::{einsum2_into, einsum2_into_uninit};
use strided_kernel::ExecContext;
use strided_view::StridedArray;

static REGISTER: Once = Once::new();

unsafe extern "C" fn test_dgemm(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    b: *const f64,
    ldb: *const BlasInt32,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt32,
) {
    let (m, n, k, lda, ldb, ldc) = (
        *m as usize,
        *n as usize,
        *k as usize,
        *lda as usize,
        *ldb as usize,
        *ldc as usize,
    );
    let ta = (*transa as u8).to_ascii_uppercase() as char;
    let tb = (*transb as u8).to_ascii_uppercase() as char;
    let av = |row: usize, col: usize| unsafe {
        if ta == 'N' {
            *a.add(row + col * lda)
        } else {
            *a.add(col + row * lda)
        }
    };
    let bv = |row: usize, col: usize| unsafe {
        if tb == 'N' {
            *b.add(row + col * ldb)
        } else {
            *b.add(col + row * ldb)
        }
    };
    for col in 0..n {
        for row in 0..m {
            let mut value = 0.0;
            for inner in 0..k {
                value += av(row, inner) * bv(inner, col);
            }
            let out = c.add(row + col * ldc);
            if *beta == 0.0 {
                *out = *alpha * value;
            } else {
                *out = *alpha * value + *beta * *out;
            }
        }
    }
}

fn register_test_provider() {
    REGISTER.call_once(|| unsafe { register_dgemm(test_dgemm) });
}

#[test]
fn test_blas_inject_works_with_explicit_registration() {
    register_test_provider();
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
    });
    let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
    });
    let mut c = StridedArray::<f64>::row_major(&[2, 2]);

    einsum2_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &['i', 'k'],
        &['i', 'j'],
        &['j', 'k'],
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_blas_inject_uninitialized_overwrite_uses_registered_provider() {
    register_test_provider();
    let a = StridedArray::<f64>::from_fn_col_major(&[2, 2], |idx| (idx[0] + 2 * idx[1] + 1) as f64);
    let b = StridedArray::<f64>::from_fn_col_major(&[2, 2], |idx| (idx[0] + 2 * idx[1] + 5) as f64);
    let mut storage = vec![MaybeUninit::<f64>::uninit(); 4];
    let dims = [2usize, 2];
    let strides = [1isize, 2];
    let mut c = strided_view::RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap();
    einsum2_into_uninit(
        &mut c,
        &a.view(),
        &b.view(),
        &['i', 'j'],
        &['i', 'k'],
        &['k', 'j'],
        1.0,
        &ExecContext::serial(),
    )
    .unwrap();
    let values: Vec<f64> = storage
        .into_iter()
        .map(|x| unsafe { x.assume_init() })
        .collect();
    assert_eq!(values, vec![23.0, 34.0, 31.0, 46.0]);
}

#[test]
fn test_blas_inject_zgemm_overwrite_does_not_read_poisoned_c() {
    let a = StridedArray::<Complex64>::from_fn_col_major(&[2, 2], |idx| {
        Complex64::new((idx[0] + 2 * idx[1] + 1) as f64, 0.0)
    });
    let b = StridedArray::<Complex64>::from_fn_col_major(&[2, 2], |idx| {
        Complex64::new((idx[0] + 2 * idx[1] + 5) as f64, 0.0)
    });
    let poison = Complex64::new(f64::NAN, f64::NAN);
    let mut storage = vec![MaybeUninit::new(poison); 4];
    let dims = [2usize, 2];
    let strides = [1isize, 2];
    let mut c = strided_view::RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap();

    einsum2_into_uninit(
        &mut c,
        &a.view(),
        &b.view(),
        &['i', 'j'],
        &['i', 'k'],
        &['k', 'j'],
        Complex64::new(1.0, 0.0),
        &ExecContext::serial(),
    )
    .unwrap();

    let values: Vec<Complex64> = storage
        .into_iter()
        .map(|x| unsafe { x.assume_init() })
        .collect();
    assert_eq!(
        values,
        vec![
            Complex64::new(23.0, 0.0),
            Complex64::new(34.0, 0.0),
            Complex64::new(31.0, 0.0),
            Complex64::new(46.0, 0.0),
        ]
    );
}
