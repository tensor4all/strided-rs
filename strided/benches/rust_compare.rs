use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided::{
    copy_into, copy_transpose_scale_into, map_into, zip_map2_into, zip_map4_into, StridedArray,
};

fn mean(durations: &[Duration]) -> Duration {
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn bench_n(label: &str, warmup_iters: usize, iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..warmup_iters {
        f();
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }

    let avg = mean(&samples);
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

/// Create a column-major random 2D array (matches Julia's Array layout).
fn make_random_2d(n: usize, seed: u64) -> StridedArray<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    StridedArray::<f64>::from_fn_col_major(&[n, n], |_| rng.sample(StandardNormal))
}

/// Create a column-major random 4D array (matches Julia's Array layout).
fn make_random_4d(n: usize, seed: u64) -> StridedArray<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    StridedArray::<f64>::from_fn_col_major(&[n, n, n, n], |_| rng.sample(StandardNormal))
}

fn main() {
    println!("Rust runner: benches/rust_compare.rs");
    println!("Note: single-threaded runner for parity with Julia. Column-major layout.");
    println!();

    // 1) symmetrize_4000: B[i,j] = 0.5 * (A[i,j] + A[j,i])
    //    Col-major [n,n]: strides [1, n]. Inner loop over i is contiguous.
    {
        println!("=== Benchmark 1: symmetrize_4000 ===");
        let n = 4000usize;
        let a = make_random_2d(n, 0);
        let a_view = a.view();
        let a_t = a_view.permute(&[1, 0]).unwrap();
        let mut b = StridedArray::<f64>::col_major(&[n, n]);

        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();

        bench_n("rust_naive", 1, 3, || {
            // Col-major: A[i,j] = a[i + j*n], A[j,i] = a[j + i*n]
            for j in 0..n {
                let col = j * n;
                for i in 0..n {
                    unsafe {
                        let aij = *a_ptr.add(col + i);
                        let aji = *a_ptr.add(i * n + j);
                        *b_ptr.add(col + i) = 0.5 * (aij + aji);
                    }
                }
            }
            black_box(b_ptr);
        });

        bench_n("rust_strided", 1, 3, || {
            zip_map2_into(&mut b.view_mut(), &a_view, &a_t, |x, y| (x + y) * 0.5).unwrap();
            black_box(b_ptr);
        });
        println!();
    }

    // 2) scale_transpose_1000: B[i,j] = 3.0 * A[j,i]
    {
        println!("=== Benchmark 2: scale_transpose_1000 ===");
        let n = 1000usize;
        let a = make_random_2d(n, 1);
        let a_view = a.view();
        let mut b = StridedArray::<f64>::col_major(&[n, n]);

        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();

        bench_n("rust_naive", 5, 10, || {
            // B[i,j] = 3.0 * A[j,i], col-major: A[j,i] at i*n+j
            for j in 0..n {
                let col_b = j * n;
                for i in 0..n {
                    unsafe {
                        let aji = *a_ptr.add(i * n + j);
                        *b_ptr.add(col_b + i) = 3.0 * aji;
                    }
                }
            }
            black_box(b_ptr);
        });

        bench_n("rust_strided", 5, 10, || {
            copy_transpose_scale_into(&mut b.view_mut(), &a_view, 3.0).unwrap();
            black_box(b_ptr);
        });
        println!();
    }

    // 2a) mwe_stridedview_scale_transpose_1000 (map_into path)
    {
        println!("=== Benchmark 2a: mwe_stridedview_scale_transpose_1000 ===");
        let n = 1000usize;
        let mut rng = StdRng::seed_from_u64(11);
        let a = StridedArray::<f64>::from_fn_col_major(&[n, n], |_| rng.gen::<f64>());
        let a_view = a.view();
        let a_t = a_view.permute(&[1, 0]).unwrap();
        let mut b = StridedArray::<f64>::col_major(&[n, n]);

        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();

        bench_n("rust_naive", 5, 10, || {
            for j in 0..n {
                let col_b = j * n;
                for i in 0..n {
                    unsafe {
                        let aji = *a_ptr.add(i * n + j);
                        *b_ptr.add(col_b + i) = 3.0 * aji;
                    }
                }
            }
            black_box(b_ptr);
        });

        bench_n("rust_strided_map", 5, 10, || {
            map_into(&mut b.view_mut(), &a_t, |x| 3.0 * x).unwrap();
            black_box(b_ptr);
        });

        println!();
    }

    // 3) complex_elementwise_1000: B[i,j] = f(A[i,j]) where f(x) = x*exp(-2x) + sin(x^2)
    //    Both A and B are contiguous col-major, so naive inner loop is contiguous.
    {
        println!("=== Benchmark 3: complex_elementwise_1000 (Float64) ===");
        let n = 1000usize;
        let a = make_random_2d(n, 2);
        let a_view = a.view();
        let mut b = StridedArray::<f64>::col_major(&[n, n]);

        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();
        let total = n * n;

        bench_n("rust_naive", 3, 6, || {
            for k in 0..total {
                unsafe {
                    let x = *a_ptr.add(k);
                    *b_ptr.add(k) = x * (-2.0 * x).exp() + (x * x).sin();
                }
            }
            black_box(b_ptr);
        });

        bench_n("rust_strided", 3, 6, || {
            map_into(&mut b.view_mut(), &a_view, |x| {
                x * (-2.0 * x).exp() + (x * x).sin()
            })
            .unwrap();
            black_box(b_ptr);
        });
        println!();
    }

    // 4) permute_32_4d: B = permutedims(A, (4,3,2,1))
    //    Col-major 4D [n,n,n,n]: strides [1, n, n^2, n^3].
    {
        println!("=== Benchmark 4: permute_32_4d ===");
        let n = 32usize;
        let a = make_random_4d(n, 3);
        let a_view = a.view();
        let a_perm = a_view.permute(&[3, 2, 1, 0]).unwrap();
        let mut b = StridedArray::<f64>::col_major(&[n, n, n, n]);

        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();
        let n2 = n * n;
        let n3 = n2 * n;

        bench_n("rust_naive", 20, 50, || {
            // B[i,j,k,l] = A[l,k,j,i] (permutation (4,3,2,1) in 1-based)
            // Col-major: B at i + j*n + k*n^2 + l*n^3
            //            A[l,k,j,i] at l + k*n + j*n^2 + i*n^3
            for l in 0..n {
                for k in 0..n {
                    for j in 0..n {
                        let b_base = j * n + k * n2 + l * n3;
                        let a_base = l + k * n + j * n2;
                        for i in 0..n {
                            unsafe {
                                let val = *a_ptr.add(a_base + i * n3);
                                *b_ptr.add(b_base + i) = val;
                            }
                        }
                    }
                }
            }
            black_box(b_ptr);
        });

        bench_n("rust_strided", 20, 50, || {
            copy_into(&mut b.view_mut(), &a_perm).unwrap();
            black_box(b_ptr);
        });
        println!();
    }

    // 5) multiple_permute_sum_32_4d: B = A + perm(A,p1) + perm(A,p2) + perm(A,p3)
    {
        println!("=== Benchmark 5: multiple_permute_sum_32_4d ===");
        let n = 32usize;
        let a = make_random_4d(n, 4);
        let a_view = a.view();

        let p1 = a_view.permute(&[0, 1, 2, 3]).unwrap();
        let p2 = a_view.permute(&[1, 2, 3, 0]).unwrap();
        let p3 = a_view.permute(&[2, 3, 0, 1]).unwrap();
        let p4 = a_view.permute(&[3, 0, 1, 2]).unwrap();
        let mut b = StridedArray::<f64>::col_major(&[n, n, n, n]);

        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();
        let n2 = n * n;
        let n3 = n2 * n;

        bench_n("rust_naive", 10, 30, || {
            // p1 = identity: A[i,j,k,l]       -> i + j*n + k*n2 + l*n3
            // p2 = (1,2,3,0): A[j,k,l,i]      -> j + k*n + l*n2 + i*n3
            // p3 = (2,3,0,1): A[k,l,i,j]      -> k + l*n + i*n2 + j*n3
            // p4 = (3,0,1,2): A[l,i,j,k]      -> l + i*n + j*n2 + k*n3
            for l in 0..n {
                for k in 0..n {
                    for j in 0..n {
                        let b_off = j * n + k * n2 + l * n3;
                        let p1_off = j * n + k * n2 + l * n3; // i + ...
                        let p2_off = k * n + l * n2; // j + ... + i*n3
                        let p3_off = l * n + j * n3; // k + ... + i*n2
                        let p4_off = j * n2 + k * n3; // l + i*n + ...
                        for i in 0..n {
                            unsafe {
                                let v1 = *a_ptr.add(p1_off + i);
                                let v2 = *a_ptr.add(p2_off + j + i * n3);
                                let v3 = *a_ptr.add(p3_off + k + i * n2);
                                let v4 = *a_ptr.add(p4_off + l + i * n);
                                *b_ptr.add(b_off + i) = v1 + v2 + v3 + v4;
                            }
                        }
                    }
                }
            }
            black_box(b_ptr);
        });

        bench_n("rust_strided_fused", 10, 30, || {
            zip_map4_into(&mut b.view_mut(), &p1, &p2, &p3, &p4, |a, b, c, d| {
                a + b + c + d
            })
            .unwrap();
            black_box(b_ptr);
        });
        println!();
    }
}
