use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_perm::{copy_into, copy_into_col_major};
use strided_view::{col_major_strides, StridedArray};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2
    }
}

fn bench_n(label: &str, warmup: usize, iters: usize, bytes: usize, mut f: impl FnMut()) {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    let med = median(&mut samples);
    let ms = med.as_secs_f64() * 1e3;
    let gbps = (bytes as f64) / med.as_secs_f64() / 1e9;
    // p25 and p75
    let p25 = samples[samples.len() / 4].as_secs_f64() * 1e3;
    let p75 = samples[samples.len() * 3 / 4].as_secs_f64() * 1e3;
    println!("  {label:30} {ms:8.3} ms  ({p25:.3} / {p75:.3})  {gbps:6.2} GB/s");
}

fn make_random_array(dims: &[usize], strides: &[isize], seed: u64) -> StridedArray<f64> {
    let total: usize = dims.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f64> = (0..total).map(|_| rng.sample(StandardNormal)).collect();
    StridedArray::from_parts(data, dims, strides, 0).unwrap()
}

/// Generic strided copy using odometer iteration.
///
/// Iterates over `dims` in col-major order, reading from `src_ptr` using
/// `src_strides` and writing to `dst_ptr` using `dst_strides`.
unsafe fn naive_strided_copy(
    src_ptr: *const f64,
    dst_ptr: *mut f64,
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) {
    let rank = dims.len();
    let total: usize = dims.iter().product();

    let mut idx = vec![0usize; rank];
    let mut src_off = 0isize;
    let mut dst_off = 0isize;

    for _ in 0..total {
        *dst_ptr.offset(dst_off) = *src_ptr.offset(src_off);
        for d in 0..rank {
            idx[d] += 1;
            src_off += src_strides[d];
            dst_off += dst_strides[d];
            if idx[d] < dims[d] {
                break;
            }
            src_off -= (idx[d] as isize) * src_strides[d];
            dst_off -= (idx[d] as isize) * dst_strides[d];
            idx[d] = 0;
        }
    }
}

/// Convenience: naive permuted copy from col-major source.
///
/// B[i_0, ..., i_{N-1}] = A[i_{perm[0]}, ..., i_{perm[N-1]}]
unsafe fn naive_permute_colmajor(
    a_ptr: *const f64,
    b_ptr: *mut f64,
    dims: &[usize],
    perm: &[usize],
) {
    let rank = dims.len();
    let cm = col_major_strides(dims);
    let src_perm: Vec<isize> = (0..rank).map(|d| cm[perm[d]]).collect();
    naive_strided_copy(a_ptr, b_ptr, dims, &src_perm, &cm);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Scenario 1: Scattered stride → col-major (the critical bottleneck)
///
/// 24 dimensions, all size 2, 16,777,216 elements (128 MB as f64).
/// Source strides from tensornetwork_permutation_light_415 Step 408.
fn scenario_scattered_to_colmajor() {
    println!("=== Scenario 1: Scattered stride → col-major (24 dims, 16M elems) ===");

    let rank = 24;
    let dims = vec![2usize; rank];
    let total: usize = 1 << rank; // 2^24 = 16,777,216
    let bytes = total * std::mem::size_of::<f64>() * 2; // read + write

    // Scattered source strides from the real workload
    let src_strides: Vec<isize> = vec![
        1, 2, 4, 8, 4194304, 16, 8388608, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
    ];

    // The inverse permutation: maps src stride order to col-major order
    let perm_inv: Vec<usize> = vec![
        0, 1, 2, 3, 22, 4, 23, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    ];

    let src = make_random_array(&dims, &src_strides, 42);
    let mut dst = StridedArray::<f64>::col_major(&dims);
    let dst_strides = col_major_strides(&dims);

    // Build the permuted view: permute src so that dst is col-major
    let src_view = src.view();
    let src_perm = src_view.permute(&perm_inv).unwrap();

    // Naive baseline (using actual permuted strides)
    {
        let a_ptr = src.data().as_ptr();
        let b_ptr = dst.data_mut().as_mut_ptr();
        let perm_strides: Vec<isize> = src_perm.strides().to_vec();
        let dst_cm = col_major_strides(&dims);
        bench_n("naive_odometer", 2, 30, bytes, || {
            unsafe { naive_strided_copy(a_ptr, b_ptr, &dims, &perm_strides, &dst_cm) };
            black_box(b_ptr);
        });
    }

    // strided-perm copy_into
    bench_n("strided_perm::copy_into", 2, 30, bytes, || {
        copy_into(&mut dst.view_mut(), &src_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    // strided-perm copy_into_col_major
    bench_n("strided_perm::copy_into_col_major", 2, 30, bytes, || {
        copy_into_col_major(&mut dst.view_mut(), &src_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    // Also bench with contiguous source permuted view (what copy_into sees)
    // to separate "scattered stride" effect from "permutation" effect
    let src_contig = make_random_array(&dims, &dst_strides, 42);
    let src_contig_perm = src_contig.view().permute(&perm_inv).unwrap();
    bench_n("copy_into (contig src, same perm)", 2, 30, bytes, || {
        copy_into(&mut dst.view_mut(), &src_contig_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    println!();
}

/// Scenario 2: Contiguous → contiguous permutation
///
/// Same shape as Scenario 1 but source is col-major.
/// Measures "always materialize" strategy cost.
fn scenario_contig_to_contig_perm() {
    println!("=== Scenario 2: Contiguous → contiguous permutation (24 dims, 16M elems) ===");

    let rank = 24;
    let dims = vec![2usize; rank];
    let total: usize = 1 << rank;
    let bytes = total * std::mem::size_of::<f64>() * 2;
    let strides = col_major_strides(&dims);

    let perm_inv: Vec<usize> = vec![
        0, 1, 2, 3, 22, 4, 23, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    ];

    let src = make_random_array(&dims, &strides, 123);
    let mut dst = StridedArray::<f64>::col_major(&dims);

    let src_perm = src.view().permute(&perm_inv).unwrap();

    bench_n("strided_perm::copy_into", 2, 30, bytes, || {
        copy_into(&mut dst.view_mut(), &src_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    bench_n("strided_perm::copy_into_col_major", 2, 30, bytes, || {
        copy_into_col_major(&mut dst.view_mut(), &src_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    // Naive baseline
    {
        let a_ptr = src.data().as_ptr();
        let b_ptr = dst.data_mut().as_mut_ptr();
        bench_n("naive_odometer", 2, 30, bytes, || {
            unsafe { naive_permute_colmajor(a_ptr, b_ptr, &dims, &perm_inv) };
            black_box(b_ptr);
        });
    }

    println!();
}

/// Scenario 3: memcpy baseline
///
/// Contiguous copy (no permutation) for bandwidth reference.
fn scenario_memcpy_baseline() {
    println!("=== Scenario 3: memcpy baseline (16M f64 elems) ===");

    let total: usize = 1 << 24; // 16,777,216
    let bytes = total * std::mem::size_of::<f64>() * 2;

    let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; total];

    bench_n("std::ptr::copy_nonoverlapping", 2, 30, bytes, || {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), total) };
        black_box(dst.as_ptr());
    });

    // Also test copy_into with both contiguous (should hit fast path)
    let dims = vec![2usize; 24];
    let strides = col_major_strides(&dims);
    let src_arr = make_random_array(&dims, &strides, 42);
    let mut dst_arr = StridedArray::<f64>::col_major(&dims);
    bench_n("copy_into (contiguous fast path)", 2, 30, bytes, || {
        copy_into(&mut dst_arr.view_mut(), &src_arr.view()).unwrap();
        black_box(dst_arr.data().as_ptr());
    });

    println!();
}

/// Scenario 4: Small tensor permutation
///
/// 13 dimensions, all size 2, 8,192 elements.
/// Tests overhead for small tensors.
fn scenario_small_tensor() {
    println!("=== Scenario 4: Small tensor permutation (13 dims, 8K elems) ===");

    let rank = 13;
    let dims = vec![2usize; rank];
    let total: usize = 1 << rank; // 8192
    let bytes = total * std::mem::size_of::<f64>() * 2;
    let strides = col_major_strides(&dims);

    // Reverse permutation (worst case)
    let perm_rev: Vec<usize> = (0..rank).rev().collect();

    let src = make_random_array(&dims, &strides, 42);
    let mut dst = StridedArray::<f64>::col_major(&dims);

    let src_perm = src.view().permute(&perm_rev).unwrap();

    bench_n("strided_perm::copy_into (reverse)", 5, 30, bytes, || {
        copy_into(&mut dst.view_mut(), &src_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    // Cyclic shift
    let perm_cyc: Vec<usize> = {
        let mut p: Vec<usize> = (1..rank).collect();
        p.push(0);
        p
    };
    let src_cyc = src.view().permute(&perm_cyc).unwrap();
    bench_n("strided_perm::copy_into (cyclic)", 5, 30, bytes, || {
        copy_into(&mut dst.view_mut(), &src_cyc).unwrap();
        black_box(dst.data().as_ptr());
    });

    // Naive
    {
        let a_ptr = src.data().as_ptr();
        let b_ptr = dst.data_mut().as_mut_ptr();
        bench_n("naive_odometer (reverse)", 5, 30, bytes, || {
            unsafe { naive_permute_colmajor(a_ptr, b_ptr, &dims, &perm_rev) };
            black_box(b_ptr);
        });
    }

    println!();
}

/// Scenario 5: Fewer large dimensions
///
/// 3 dimensions, sizes [256, 256, 256] (16M elements).
/// Transpose: [2, 0, 1]. Traditional blocking shines here.
fn scenario_large_dims() {
    println!("=== Scenario 5: Fewer large dimensions ([256, 256, 256], transpose [2,0,1]) ===");

    let dims = vec![256, 256, 256];
    let total: usize = dims.iter().product(); // 16,777,216
    let bytes = total * std::mem::size_of::<f64>() * 2;
    let strides = col_major_strides(&dims);
    let perm = vec![2, 0, 1];

    let src = make_random_array(&dims, &strides, 42);
    let mut dst = StridedArray::<f64>::col_major(&dims);

    let src_perm = src.view().permute(&perm).unwrap();

    bench_n("strided_perm::copy_into", 2, 30, bytes, || {
        copy_into(&mut dst.view_mut(), &src_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    bench_n("strided_perm::copy_into_col_major", 2, 30, bytes, || {
        copy_into_col_major(&mut dst.view_mut(), &src_perm).unwrap();
        black_box(dst.data().as_ptr());
    });

    // Naive baseline with precomputed strides
    {
        let a_ptr = src.data().as_ptr();
        let b_ptr = dst.data_mut().as_mut_ptr();
        bench_n("naive_odometer", 2, 30, bytes, || {
            unsafe { naive_permute_colmajor(a_ptr, b_ptr, &dims, &perm) };
            black_box(b_ptr);
        });
    }

    // Also test simple transpose [1, 0, 2]
    let perm2 = vec![1, 0, 2];
    let src_perm2 = src.view().permute(&perm2).unwrap();
    bench_n("copy_into (transpose [1,0,2])", 2, 30, bytes, || {
        copy_into(&mut dst.view_mut(), &src_perm2).unwrap();
        black_box(dst.data().as_ptr());
    });

    println!();
}

// ---------------------------------------------------------------------------
// Correctness verification
// ---------------------------------------------------------------------------

fn verify_scenario(label: &str, dims: &[usize], src_strides: &[isize], perm: &[usize]) {
    let total: usize = dims.iter().product();
    let src = make_random_array(dims, src_strides, 99);
    let mut dst_naive = StridedArray::<f64>::col_major(dims);
    let mut dst_strided = StridedArray::<f64>::col_major(dims);

    // Build the same permuted view that copy_into will see
    let src_perm = src.view().permute(perm).unwrap();
    let perm_src_strides: Vec<isize> = src_perm.strides().to_vec();
    let dst_cm = col_major_strides(dims);

    unsafe {
        naive_strided_copy(
            src.data().as_ptr(),
            dst_naive.data_mut().as_mut_ptr(),
            dims,
            &perm_src_strides,
            &dst_cm,
        );
    }

    copy_into(&mut dst_strided.view_mut(), &src_perm).unwrap();

    for i in 0..total {
        assert_eq!(
            dst_naive.data()[i],
            dst_strided.data()[i],
            "Mismatch at element {i} for '{label}'"
        );
    }
    println!("  [OK] {label}");
}

fn run_correctness_checks() {
    println!("--- Correctness verification ---");

    // Scenario 1: scattered strides
    let dims24 = vec![2usize; 24];
    let scattered_strides: Vec<isize> = vec![
        1, 2, 4, 8, 4194304, 16, 8388608, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
    ];
    let perm_inv: Vec<usize> = vec![
        0, 1, 2, 3, 22, 4, 23, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    ];
    verify_scenario(
        "scattered→colmajor (24d)",
        &dims24,
        &scattered_strides,
        &perm_inv,
    );

    // Scenario 2: contiguous + permutation
    let cm_strides = col_major_strides(&dims24);
    verify_scenario("contig→contig perm (24d)", &dims24, &cm_strides, &perm_inv);

    // Scenario 4: small tensor
    let dims13 = vec![2usize; 13];
    let sm_strides = col_major_strides(&dims13);
    let perm_rev13: Vec<usize> = (0..13).rev().collect();
    verify_scenario(
        "small tensor reverse (13d)",
        &dims13,
        &sm_strides,
        &perm_rev13,
    );

    // Scenario 5: large dims
    let dims3 = vec![256, 256, 256];
    let strides3 = col_major_strides(&dims3);
    verify_scenario("large dims [2,0,1]", &dims3, &strides3, &[2, 0, 1]);
    verify_scenario("large dims [1,0,2]", &dims3, &strides3, &[1, 0, 2]);

    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("strided-perm permutation benchmarks");
    println!("====================================");
    println!(
        "Element size: {} bytes, 16M elements = {} MB",
        std::mem::size_of::<f64>(),
        (1 << 24) * std::mem::size_of::<f64>() / (1024 * 1024)
    );
    println!("Format: label  median_ms  (p25 / p75)  bandwidth_GB/s");
    println!();

    run_correctness_checks();
    scenario_memcpy_baseline();
    scenario_scattered_to_colmajor();
    scenario_contig_to_contig_perm();
    scenario_small_tensor();
    scenario_large_dims();

    println!("Done.");
}
