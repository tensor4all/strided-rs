use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use strided_einsum2::backend::{ActiveBackend, Backend};
use strided_einsum2::contiguous::{
    prepare_input_view, prepare_output_view, ContiguousOperand, ContiguousOperandMut,
};
use strided_einsum2::Scalar;
use strided_view::StridedArray;

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_RUNS: usize = 15;

#[derive(Clone, Copy)]
enum BenchCase {
    BatchedMatmul {
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        layout: MatmulLayout,
    },
}

#[derive(Clone, Copy)]
enum MatmulLayout {
    MemoryMatched,
    NN,
    TN,
    NT,
    TT,
}

impl MatmulLayout {
    fn label(self) -> &'static str {
        match self {
            Self::MemoryMatched => "memory_matched",
            Self::NN => "NN",
            Self::TN => "TN",
            Self::NT => "NT",
            Self::TT => "TT",
        }
    }

    fn benchmark_suffix(self) -> &'static str {
        match self {
            Self::MemoryMatched => "",
            Self::NN => "_nn",
            Self::TN => "_tn",
            Self::NT => "_nt",
            Self::TT => "_tt",
        }
    }

    fn lhs_row_major(self) -> bool {
        matches!(self, Self::TN | Self::TT)
    }

    fn rhs_row_major(self) -> bool {
        matches!(self, Self::NT | Self::TT)
    }
}

#[derive(Clone, Copy)]
enum BenchDType {
    F64,
    C64,
    C128,
}

impl BenchDType {
    fn label(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        }
    }
}

impl BenchCase {
    fn benchmark(self) -> String {
        match self {
            Self::BatchedMatmul {
                batch,
                m,
                n,
                k,
                layout,
            } => {
                format!(
                    "bin_batched_matmul_b{batch}_m{m}_n{n}_k{k}{}",
                    layout.benchmark_suffix()
                )
            }
        }
    }

    fn shape_label(self) -> String {
        match self {
            Self::BatchedMatmul {
                batch,
                m,
                n,
                k,
                layout,
            } => {
                format!("b={batch};m={m};n={n};k={k};layout={}", layout.label())
            }
        }
    }
}

trait BenchScalar: Scalar + Copy + Default + 'static {
    fn from_indices(indices: &[usize], salt: usize) -> Self;
    fn one() -> Self;
}

impl BenchScalar for f64 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        fill_value(indices, salt)
    }

    fn one() -> Self {
        1.0
    }
}

impl BenchScalar for num_complex::Complex32 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        Self::new(
            fill_value(indices, salt) as f32,
            fill_value(indices, salt + 17) as f32,
        )
    }

    fn one() -> Self {
        Self::new(1.0, 0.0)
    }
}

impl BenchScalar for num_complex::Complex64 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        Self::new(fill_value(indices, salt), fill_value(indices, salt + 17))
    }

    fn one() -> Self {
        Self::new(1.0, 0.0)
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_bool(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn fill_value(indices: &[usize], salt: usize) -> f64 {
    let mut acc = salt.wrapping_mul(1_099);
    for (axis, &idx) in indices.iter().enumerate() {
        acc = acc.wrapping_add((axis + 1).wrapping_mul(1_003).wrapping_mul(idx + 1));
    }
    ((acc % 1024) as f64 - 512.0) / 512.0
}

fn make_col_major<T: BenchScalar>(dims: &[usize], salt: usize) -> StridedArray<T> {
    StridedArray::<T>::from_fn_col_major(dims, |idx| T::from_indices(idx, salt))
}

fn make_batched_matrix<T: BenchScalar>(
    rows: usize,
    cols: usize,
    batch: usize,
    row_major: bool,
    salt: usize,
) -> StridedArray<T> {
    let dims = [rows, cols, batch];
    let strides = if row_major {
        [cols as isize, 1, (rows * cols) as isize]
    } else {
        [1, rows as isize, (rows * cols) as isize]
    };
    let len = rows * cols * batch;
    let data = (0..len)
        .map(|i| T::from_indices(&[i], salt))
        .collect::<Vec<_>>();
    StridedArray::<T>::from_parts(data, &dims, &strides, 0).unwrap()
}

fn matmul_cases(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    layouts: &[MatmulLayout],
) -> Vec<BenchCase> {
    layouts
        .iter()
        .copied()
        .map(|layout| BenchCase::BatchedMatmul {
            batch,
            m,
            n,
            k,
            layout,
        })
        .collect()
}

fn profile_cases() -> Vec<BenchCase> {
    match env::var("STRIDED_EINSUM2_DOT_GENERAL_BENCH_PROFILE")
        .unwrap_or_else(|_| "full".to_string())
        .as_str()
    {
        "smoke" => matmul_cases(2, 8, 8, 8, &[MatmulLayout::MemoryMatched, MatmulLayout::TN]),
        "quick" => matmul_cases(
            32,
            64,
            64,
            64,
            &[
                MatmulLayout::MemoryMatched,
                MatmulLayout::NN,
                MatmulLayout::TN,
                MatmulLayout::NT,
                MatmulLayout::TT,
            ],
        ),
        _ => {
            let layouts = [
                MatmulLayout::MemoryMatched,
                MatmulLayout::NN,
                MatmulLayout::TN,
                MatmulLayout::NT,
                MatmulLayout::TT,
            ];
            let mut cases = matmul_cases(32, 64, 64, 64, &layouts);
            cases.extend(matmul_cases(32, 128, 128, 128, &layouts));
            cases
        }
    }
}

fn profile_dtypes() -> Vec<BenchDType> {
    let dtypes: Vec<_> = env::var("STRIDED_EINSUM2_DOT_GENERAL_BENCH_DTYPES")
        .unwrap_or_else(|_| "f64".to_string())
        .split(',')
        .filter_map(|value| match value.trim() {
            "f64" => Some(BenchDType::F64),
            "c64" => Some(BenchDType::C64),
            "c128" => Some(BenchDType::C128),
            _ => None,
        })
        .collect();

    if dtypes.is_empty() {
        vec![BenchDType::F64]
    } else {
        dtypes
    }
}

fn duration_stats(mut durations: Vec<Duration>) -> (f64, f64) {
    durations.sort_unstable();
    let median = durations[durations.len() / 2];
    let q1 = durations[durations.len() / 4];
    let q3 = durations[3 * durations.len() / 4];
    (
        median.as_secs_f64() * 1e3,
        q3.saturating_sub(q1).as_secs_f64() * 1e3,
    )
}

fn measure(mut f: impl FnMut()) -> (f64, f64) {
    let warmups = env_usize("STRIDED_EINSUM2_DOT_GENERAL_BENCH_WARMUPS", DEFAULT_WARMUPS);
    let runs = env_usize("STRIDED_EINSUM2_DOT_GENERAL_BENCH_RUNS", DEFAULT_RUNS);
    for _ in 0..warmups {
        f();
    }

    let mut durations = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        f();
        durations.push(start.elapsed());
    }
    duration_stats(durations)
}

fn run_batched_matmul<T: BenchScalar>(batch: usize, m: usize, n: usize, k: usize) -> (f64, f64)
where
    ActiveBackend: Backend<T>,
{
    // tenferro-benchmark's row-major `bij,bjk->bik` instance maps to
    // col-major buffers as `jib,kjb->kib`. This prepared benchmark excludes
    // allocation/setup from the timed loop: operands are converted to GEMM-ready
    // metadata once, then the backend batched GEMM is called repeatedly.
    let lhs_row_memory = make_col_major::<T>(&[k, m, batch], 11);
    let rhs_row_memory = make_col_major::<T>(&[n, k, batch], 12);
    let mut out = StridedArray::<T>::col_major(&[n, m, batch]);
    let a_op = prepare_input_view(
        &rhs_row_memory.view(),
        1,
        1,
        false,
        ActiveBackend::REQUIRES_UNIT_STRIDE,
        true,
        None,
    )
    .unwrap();
    let b_op = prepare_input_view(
        &lhs_row_memory.view(),
        1,
        1,
        false,
        ActiveBackend::REQUIRES_UNIT_STRIDE,
        true,
        None,
    )
    .unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(
            &mut out_view,
            1,
            1,
            T::default(),
            ActiveBackend::REQUIRES_UNIT_STRIDE,
            true,
        )
        .unwrap()
    };

    measure(|| {
        ActiveBackend::bgemm_contiguous_into(
            &mut c_op,
            &a_op,
            &b_op,
            &[batch],
            n,
            m,
            k,
            <T as BenchScalar>::one(),
            T::default(),
        )
        .unwrap();
        black_box(out.data().as_ptr());
    })
}

fn run_batched_matmul_layout<T: BenchScalar>(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    layout: MatmulLayout,
) -> (f64, f64)
where
    ActiveBackend: Backend<T>,
{
    if matches!(layout, MatmulLayout::MemoryMatched) {
        return run_batched_matmul::<T>(batch, m, n, k);
    }

    let lhs = make_batched_matrix::<T>(m, k, batch, layout.lhs_row_major(), 11);
    let rhs = make_batched_matrix::<T>(k, n, batch, layout.rhs_row_major(), 12);
    let mut out = StridedArray::<T>::col_major(&[m, n, batch]);

    let a_op = prepare_input_view(
        &lhs.view(),
        1,
        1,
        false,
        ActiveBackend::REQUIRES_UNIT_STRIDE,
        true,
        None,
    )
    .unwrap();
    let b_op = prepare_input_view(
        &rhs.view(),
        1,
        1,
        false,
        ActiveBackend::REQUIRES_UNIT_STRIDE,
        true,
        None,
    )
    .unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(
            &mut out_view,
            1,
            1,
            T::default(),
            ActiveBackend::REQUIRES_UNIT_STRIDE,
            true,
        )
        .unwrap()
    };

    measure(|| {
        ActiveBackend::bgemm_contiguous_into(
            &mut c_op,
            &a_op,
            &b_op,
            &[batch],
            m,
            n,
            k,
            <T as BenchScalar>::one(),
            T::default(),
        )
        .unwrap();
        black_box(out.data().as_ptr());
    })
}

#[cfg(feature = "blas")]
fn raw_cblas_flip_transpose(t: cblas_sys::CBLAS_TRANSPOSE) -> cblas_sys::CBLAS_TRANSPOSE {
    match t {
        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans => cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
        cblas_sys::CBLAS_TRANSPOSE::CblasTrans => cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
        other => other,
    }
}

#[cfg(feature = "blas")]
fn raw_cblas_operand_layout(
    row_stride: isize,
    col_stride: isize,
    nrows: usize,
    ncols: usize,
) -> (cblas_sys::CBLAS_TRANSPOSE, i32) {
    if row_stride == 1 || row_stride == 0 {
        let lda = col_stride.max(nrows as isize).max(1) as i32;
        (cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans, lda)
    } else if col_stride == 1 || col_stride == 0 {
        let lda = row_stride.max(ncols as isize).max(1) as i32;
        (cblas_sys::CBLAS_TRANSPOSE::CblasTrans, lda)
    } else {
        panic!(
            "raw cblas benchmark input has non-unit strides (row={row_stride}, col={col_stride})"
        );
    }
}

#[cfg(feature = "blas")]
fn raw_cblas_dgemm_batched(
    c: &mut ContiguousOperandMut<f64>,
    a: &ContiguousOperand<f64>,
    b: &ContiguousOperand<f64>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let (trans_a, lda) = raw_cblas_operand_layout(a.row_stride(), a.col_stride(), m, k);
    let (trans_b, ldb) = raw_cblas_operand_layout(b.row_stride(), b.col_stride(), k, n);
    let c_is_col_major = c.row_stride() == 1 || c.row_stride() == 0;
    let a_batch_step = a.batch_strides().first().copied().unwrap_or(0);
    let b_batch_step = b.batch_strides().first().copied().unwrap_or(0);
    let c_batch_step = c.batch_strides().first().copied().unwrap_or(0);

    for batch_idx in 0..batch {
        let a_off = batch_idx as isize * a_batch_step;
        let b_off = batch_idx as isize * b_batch_step;
        let c_off = batch_idx as isize * c_batch_step;
        unsafe {
            if c_is_col_major {
                let ldc = c.col_stride().max(m as isize).max(1) as i32;
                cblas_sys::cblas_dgemm(
                    cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                    trans_a,
                    trans_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.ptr().offset(a_off),
                    lda,
                    b.ptr().offset(b_off),
                    ldb,
                    0.0,
                    c.ptr().offset(c_off),
                    ldc,
                );
            } else {
                let ldc = c.row_stride().max(n as isize).max(1) as i32;
                cblas_sys::cblas_dgemm(
                    cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                    raw_cblas_flip_transpose(trans_b),
                    raw_cblas_flip_transpose(trans_a),
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b.ptr().offset(b_off),
                    ldb,
                    a.ptr().offset(a_off),
                    lda,
                    0.0,
                    c.ptr().offset(c_off),
                    ldc,
                );
            }
        }
    }
}

#[cfg(feature = "blas")]
fn raw_trait_dgemm_batched(
    c: &mut ContiguousOperandMut<f64>,
    a: &ContiguousOperand<f64>,
    b: &ContiguousOperand<f64>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let (trans_a, lda) = raw_cblas_operand_layout(a.row_stride(), a.col_stride(), m, k);
    let (trans_b, ldb) = raw_cblas_operand_layout(b.row_stride(), b.col_stride(), k, n);
    let c_is_col_major = c.row_stride() == 1 || c.row_stride() == 0;
    let a_batch_step = a.batch_strides().first().copied().unwrap_or(0);
    let b_batch_step = b.batch_strides().first().copied().unwrap_or(0);
    let c_batch_step = c.batch_strides().first().copied().unwrap_or(0);

    for batch_idx in 0..batch {
        let a_off = batch_idx as isize * a_batch_step;
        let b_off = batch_idx as isize * b_batch_step;
        let c_off = batch_idx as isize * c_batch_step;
        unsafe {
            if c_is_col_major {
                let ldc = c.col_stride().max(m as isize).max(1) as i32;
                <f64 as strided_einsum2::bgemm_blas::BlasGemm>::gemm(
                    trans_a,
                    trans_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.ptr().offset(a_off),
                    lda,
                    b.ptr().offset(b_off),
                    ldb,
                    0.0,
                    c.ptr().offset(c_off),
                    ldc,
                );
            } else {
                let ldc = c.row_stride().max(n as isize).max(1) as i32;
                <f64 as strided_einsum2::bgemm_blas::BlasGemm>::gemm(
                    raw_cblas_flip_transpose(trans_b),
                    raw_cblas_flip_transpose(trans_a),
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b.ptr().offset(b_off),
                    ldb,
                    a.ptr().offset(a_off),
                    lda,
                    0.0,
                    c.ptr().offset(c_off),
                    ldc,
                );
            }
        }
    }
}

#[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
extern "C" {
    fn dgemm_(
        transa: *const std::os::raw::c_char,
        transb: *const std::os::raw::c_char,
        m: *const std::os::raw::c_int,
        n: *const std::os::raw::c_int,
        k: *const std::os::raw::c_int,
        alpha: *const f64,
        a: *const f64,
        lda: *const std::os::raw::c_int,
        b: *const f64,
        ldb: *const std::os::raw::c_int,
        beta: *const f64,
        c: *mut f64,
        ldc: *const std::os::raw::c_int,
    );
}

#[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
fn raw_fortran_transpose(t: cblas_sys::CBLAS_TRANSPOSE) -> std::os::raw::c_char {
    match t {
        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans => b'N' as std::os::raw::c_char,
        cblas_sys::CBLAS_TRANSPOSE::CblasTrans => b'T' as std::os::raw::c_char,
        cblas_sys::CBLAS_TRANSPOSE::CblasConjTrans => b'C' as std::os::raw::c_char,
    }
}

#[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
fn raw_fortran_dgemm_batched(
    c: &mut ContiguousOperandMut<f64>,
    a: &ContiguousOperand<f64>,
    b: &ContiguousOperand<f64>,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let (trans_a, lda) = raw_cblas_operand_layout(a.row_stride(), a.col_stride(), m, k);
    let (trans_b, ldb) = raw_cblas_operand_layout(b.row_stride(), b.col_stride(), k, n);
    let c_is_col_major = c.row_stride() == 1 || c.row_stride() == 0;
    let a_batch_step = a.batch_strides().first().copied().unwrap_or(0);
    let b_batch_step = b.batch_strides().first().copied().unwrap_or(0);
    let c_batch_step = c.batch_strides().first().copied().unwrap_or(0);

    for batch_idx in 0..batch {
        let a_off = batch_idx as isize * a_batch_step;
        let b_off = batch_idx as isize * b_batch_step;
        let c_off = batch_idx as isize * c_batch_step;
        unsafe {
            if c_is_col_major {
                let trans_a = raw_fortran_transpose(trans_a);
                let trans_b = raw_fortran_transpose(trans_b);
                let m = m as std::os::raw::c_int;
                let n = n as std::os::raw::c_int;
                let k = k as std::os::raw::c_int;
                let alpha = 1.0;
                let beta = 0.0;
                let ldc = c.col_stride().max(m as isize).max(1) as std::os::raw::c_int;
                dgemm_(
                    &trans_a,
                    &trans_b,
                    &m,
                    &n,
                    &k,
                    &alpha,
                    a.ptr().offset(a_off),
                    &lda,
                    b.ptr().offset(b_off),
                    &ldb,
                    &beta,
                    c.ptr().offset(c_off),
                    &ldc,
                );
            } else {
                let trans_a_flipped = raw_fortran_transpose(raw_cblas_flip_transpose(trans_b));
                let trans_b_flipped = raw_fortran_transpose(raw_cblas_flip_transpose(trans_a));
                let m_i32 = n as std::os::raw::c_int;
                let n_i32 = m as std::os::raw::c_int;
                let k_i32 = k as std::os::raw::c_int;
                let alpha = 1.0;
                let beta = 0.0;
                let ldc = c.row_stride().max(n as isize).max(1) as std::os::raw::c_int;
                dgemm_(
                    &trans_a_flipped,
                    &trans_b_flipped,
                    &m_i32,
                    &n_i32,
                    &k_i32,
                    &alpha,
                    b.ptr().offset(b_off),
                    &ldb,
                    a.ptr().offset(a_off),
                    &lda,
                    &beta,
                    c.ptr().offset(c_off),
                    &ldc,
                );
            }
        }
    }
}

#[cfg(feature = "blas")]
fn run_batched_matmul_raw_cblas_f64(batch: usize, m: usize, n: usize, k: usize) -> (f64, f64) {
    let lhs_row_memory = make_col_major::<f64>(&[k, m, batch], 11);
    let rhs_row_memory = make_col_major::<f64>(&[n, k, batch], 12);
    let mut out = StridedArray::<f64>::col_major(&[n, m, batch]);
    let a_op = prepare_input_view(&rhs_row_memory.view(), 1, 1, false, true, true, None).unwrap();
    let b_op = prepare_input_view(&lhs_row_memory.view(), 1, 1, false, true, true, None).unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(&mut out_view, 1, 1, 0.0, true, true).unwrap()
    };

    measure(|| {
        raw_cblas_dgemm_batched(&mut c_op, &a_op, &b_op, batch, n, m, k);
        black_box(out.data().as_ptr());
    })
}

#[cfg(feature = "blas")]
fn run_batched_matmul_raw_trait_f64(batch: usize, m: usize, n: usize, k: usize) -> (f64, f64) {
    let lhs_row_memory = make_col_major::<f64>(&[k, m, batch], 11);
    let rhs_row_memory = make_col_major::<f64>(&[n, k, batch], 12);
    let mut out = StridedArray::<f64>::col_major(&[n, m, batch]);
    let a_op = prepare_input_view(&rhs_row_memory.view(), 1, 1, false, true, true, None).unwrap();
    let b_op = prepare_input_view(&lhs_row_memory.view(), 1, 1, false, true, true, None).unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(&mut out_view, 1, 1, 0.0, true, true).unwrap()
    };

    measure(|| {
        raw_trait_dgemm_batched(&mut c_op, &a_op, &b_op, batch, n, m, k);
        black_box(out.data().as_ptr());
    })
}

#[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
fn run_batched_matmul_raw_fortran_f64(batch: usize, m: usize, n: usize, k: usize) -> (f64, f64) {
    let lhs_row_memory = make_col_major::<f64>(&[k, m, batch], 11);
    let rhs_row_memory = make_col_major::<f64>(&[n, k, batch], 12);
    let mut out = StridedArray::<f64>::col_major(&[n, m, batch]);
    let a_op = prepare_input_view(&rhs_row_memory.view(), 1, 1, false, true, true, None).unwrap();
    let b_op = prepare_input_view(&lhs_row_memory.view(), 1, 1, false, true, true, None).unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(&mut out_view, 1, 1, 0.0, true, true).unwrap()
    };

    measure(|| {
        raw_fortran_dgemm_batched(&mut c_op, &a_op, &b_op, batch, n, m, k);
        black_box(out.data().as_ptr());
    })
}

#[cfg(feature = "blas")]
fn run_batched_matmul_layout_raw_cblas_f64(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    layout: MatmulLayout,
) -> (f64, f64) {
    if matches!(layout, MatmulLayout::MemoryMatched) {
        return run_batched_matmul_raw_cblas_f64(batch, m, n, k);
    }

    let lhs = make_batched_matrix::<f64>(m, k, batch, layout.lhs_row_major(), 11);
    let rhs = make_batched_matrix::<f64>(k, n, batch, layout.rhs_row_major(), 12);
    let mut out = StridedArray::<f64>::col_major(&[m, n, batch]);
    let a_op = prepare_input_view(&lhs.view(), 1, 1, false, true, true, None).unwrap();
    let b_op = prepare_input_view(&rhs.view(), 1, 1, false, true, true, None).unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(&mut out_view, 1, 1, 0.0, true, true).unwrap()
    };

    measure(|| {
        raw_cblas_dgemm_batched(&mut c_op, &a_op, &b_op, batch, m, n, k);
        black_box(out.data().as_ptr());
    })
}

#[cfg(feature = "blas")]
fn run_batched_matmul_layout_raw_trait_f64(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    layout: MatmulLayout,
) -> (f64, f64) {
    if matches!(layout, MatmulLayout::MemoryMatched) {
        return run_batched_matmul_raw_trait_f64(batch, m, n, k);
    }

    let lhs = make_batched_matrix::<f64>(m, k, batch, layout.lhs_row_major(), 11);
    let rhs = make_batched_matrix::<f64>(k, n, batch, layout.rhs_row_major(), 12);
    let mut out = StridedArray::<f64>::col_major(&[m, n, batch]);
    let a_op = prepare_input_view(&lhs.view(), 1, 1, false, true, true, None).unwrap();
    let b_op = prepare_input_view(&rhs.view(), 1, 1, false, true, true, None).unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(&mut out_view, 1, 1, 0.0, true, true).unwrap()
    };

    measure(|| {
        raw_trait_dgemm_batched(&mut c_op, &a_op, &b_op, batch, m, n, k);
        black_box(out.data().as_ptr());
    })
}

#[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
fn run_batched_matmul_layout_raw_fortran_f64(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    layout: MatmulLayout,
) -> (f64, f64) {
    if matches!(layout, MatmulLayout::MemoryMatched) {
        return run_batched_matmul_raw_fortran_f64(batch, m, n, k);
    }

    let lhs = make_batched_matrix::<f64>(m, k, batch, layout.lhs_row_major(), 11);
    let rhs = make_batched_matrix::<f64>(k, n, batch, layout.rhs_row_major(), 12);
    let mut out = StridedArray::<f64>::col_major(&[m, n, batch]);
    let a_op = prepare_input_view(&lhs.view(), 1, 1, false, true, true, None).unwrap();
    let b_op = prepare_input_view(&rhs.view(), 1, 1, false, true, true, None).unwrap();
    let mut c_op = {
        let mut out_view = out.view_mut();
        prepare_output_view(&mut out_view, 1, 1, 0.0, true, true).unwrap()
    };

    measure(|| {
        raw_fortran_dgemm_batched(&mut c_op, &a_op, &b_op, batch, m, n, k);
        black_box(out.data().as_ptr());
    })
}

fn run_case_typed<T: BenchScalar>(case: BenchCase) -> (f64, f64)
where
    ActiveBackend: Backend<T>,
{
    match case {
        BenchCase::BatchedMatmul {
            batch,
            m,
            n,
            k,
            layout,
        } => run_batched_matmul_layout::<T>(batch, m, n, k, layout),
    }
}

#[cfg(feature = "blas")]
fn run_raw_cblas_case_for_dtype(dtype: BenchDType, case: BenchCase) -> Option<(f64, f64)> {
    if !matches!(dtype, BenchDType::F64) {
        return None;
    }
    match case {
        BenchCase::BatchedMatmul {
            batch,
            m,
            n,
            k,
            layout,
        } => Some(run_batched_matmul_layout_raw_cblas_f64(
            batch, m, n, k, layout,
        )),
    }
}

#[cfg(feature = "blas")]
fn run_raw_trait_case_for_dtype(dtype: BenchDType, case: BenchCase) -> Option<(f64, f64)> {
    if !matches!(dtype, BenchDType::F64) {
        return None;
    }
    match case {
        BenchCase::BatchedMatmul {
            batch,
            m,
            n,
            k,
            layout,
        } => Some(run_batched_matmul_layout_raw_trait_f64(
            batch, m, n, k, layout,
        )),
    }
}

#[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
fn run_raw_fortran_case_for_dtype(dtype: BenchDType, case: BenchCase) -> Option<(f64, f64)> {
    if !matches!(dtype, BenchDType::F64) {
        return None;
    }
    match case {
        BenchCase::BatchedMatmul {
            batch,
            m,
            n,
            k,
            layout,
        } => Some(run_batched_matmul_layout_raw_fortran_f64(
            batch, m, n, k, layout,
        )),
    }
}

fn run_case_for_dtype(dtype: BenchDType, case: BenchCase) -> (f64, f64) {
    match dtype {
        BenchDType::F64 => run_case_typed::<f64>(case),
        BenchDType::C64 => run_case_typed::<num_complex::Complex32>(case),
        BenchDType::C128 => run_case_typed::<num_complex::Complex64>(case),
    }
}

fn threads_label() -> String {
    env::var("RAYON_NUM_THREADS")
        .or_else(|_| env::var("OMP_NUM_THREADS"))
        .unwrap_or_else(|_| "unset".to_string())
}

fn backend_label() -> &'static str {
    #[cfg(feature = "blas-accelerate")]
    {
        "strided-einsum2-accelerate-prepared"
    }
    #[cfg(feature = "blas-openblas")]
    {
        "strided-einsum2-openblas-prepared"
    }
    #[cfg(feature = "blas-mkl")]
    {
        "strided-einsum2-mkl-prepared"
    }
    #[cfg(all(
        feature = "blas",
        not(any(
            feature = "blas-accelerate",
            feature = "blas-openblas",
            feature = "blas-mkl",
            feature = "blas-inject"
        )),
        not(feature = "blas-inject"),
        target_os = "macos"
    ))]
    {
        "strided-einsum2-accelerate-prepared"
    }
    #[cfg(all(
        feature = "blas",
        not(any(
            feature = "blas-accelerate",
            feature = "blas-openblas",
            feature = "blas-mkl",
            feature = "blas-inject"
        )),
        not(feature = "blas-inject"),
        not(target_os = "macos")
    ))]
    {
        "strided-einsum2-blas-prepared"
    }
    #[cfg(all(feature = "blas-inject", not(feature = "blas")))]
    {
        "strided-einsum2-blas-inject-prepared"
    }
    #[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
    {
        "strided-einsum2-faer-prepared"
    }
    #[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
    {
        "strided-einsum2-naive-prepared"
    }
}

fn main() {
    println!("suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status");
    let threads = threads_label();
    let backend = backend_label();
    let emit_diagnostics = env_bool("STRIDED_EINSUM2_DOT_GENERAL_BENCH_DIAGNOSTICS");
    for dtype in profile_dtypes() {
        for case in profile_cases() {
            let (median_ms, iqr_ms) = run_case_for_dtype(dtype, case);
            println!(
                "dot_general,{},{},{},{},{},{:.6},{:.6},ok",
                case.benchmark(),
                dtype.label(),
                threads,
                case.shape_label(),
                backend,
                median_ms,
                iqr_ms
            );
            if !emit_diagnostics {
                continue;
            }
            #[cfg(feature = "blas")]
            if let Some((median_ms, iqr_ms)) = run_raw_cblas_case_for_dtype(dtype, case) {
                println!(
                    "dot_general,{},{},{},{},{},{:.6},{:.6},ok",
                    case.benchmark(),
                    dtype.label(),
                    threads,
                    case.shape_label(),
                    "raw-cblas-dgemm",
                    median_ms,
                    iqr_ms
                );
            }
            #[cfg(feature = "blas")]
            if let Some((median_ms, iqr_ms)) = run_raw_trait_case_for_dtype(dtype, case) {
                println!(
                    "dot_general,{},{},{},{},{},{:.6},{:.6},ok",
                    case.benchmark(),
                    dtype.label(),
                    threads,
                    case.shape_label(),
                    "raw-trait-dgemm",
                    median_ms,
                    iqr_ms
                );
            }
            #[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
            if let Some((median_ms, iqr_ms)) = run_raw_fortran_case_for_dtype(dtype, case) {
                println!(
                    "dot_general,{},{},{},{},{},{:.6},{:.6},ok",
                    case.benchmark(),
                    dtype.label(),
                    threads,
                    case.shape_label(),
                    "raw-fortran-dgemm",
                    median_ms,
                    iqr_ms
                );
            }
        }
    }
}
