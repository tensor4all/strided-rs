use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_kernel::{col_major_strides, copy_into, StridedArray};

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

/// Create a rank-25 random column-major array with each dimension of size 2.
///
/// Uses `from_parts` with a pre-built Vec to avoid the multi-index bookkeeping
/// overhead of `from_fn_col_major` for 25 dimensions.
fn make_random_rank25(seed: u64) -> StridedArray<f64> {
    let rank = 25;
    let dims = vec![2usize; rank];
    let total: usize = 1 << rank; // 2^25 = 33,554,432
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f64> = (0..total).map(|_| rng.sample(StandardNormal)).collect();
    let strides = col_major_strides(&dims);
    StridedArray::from_parts(data, &dims, &strides, 0).unwrap()
}

/// Generic N-dimensional naive permuted copy using odometer iteration.
///
/// B[i_0, i_1, ..., i_{N-1}] = A[i_{perm[0]}, i_{perm[1]}, ..., i_{perm[N-1]}]
///
/// Uses precomputed strides and raw pointers to avoid high-level indexing overhead.
unsafe fn naive_permute_nd(a_ptr: *const f64, b_ptr: *mut f64, dims: &[usize], perm: &[usize]) {
    let rank = dims.len();
    let total: usize = dims.iter().product();

    // Precompute column-major strides for source: [1, 2, 4, 8, ..., 2^(N-1)]
    let mut a_strides = vec![1usize; rank];
    for i in 1..rank {
        a_strides[i] = a_strides[i - 1] * dims[i - 1];
    }

    // Permuted source strides: for B's multi-index [i0,i1,...], the source offset
    // contribution from dimension d is i_d * a_strides[perm[d]].
    let a_perm_strides: Vec<usize> = (0..rank).map(|d| a_strides[perm[d]]).collect();

    // Destination column-major strides (same shape as source)
    let mut b_strides = vec![1usize; rank];
    for i in 1..rank {
        b_strides[i] = b_strides[i - 1] * dims[i - 1];
    }

    // Odometer iteration
    let mut idx = vec![0usize; rank];
    let mut a_offset = 0usize;
    let mut b_offset = 0usize;

    for _ in 0..total {
        *b_ptr.add(b_offset) = *a_ptr.add(a_offset);

        // Increment multi-index (column-major order: fastest dimension first)
        for d in 0..rank {
            idx[d] += 1;
            a_offset += a_perm_strides[d];
            b_offset += b_strides[d];
            if idx[d] < dims[d] {
                break;
            }
            // Reset this dimension and carry to next
            a_offset -= idx[d] * a_perm_strides[d];
            b_offset -= idx[d] * b_strides[d];
            idx[d] = 0;
        }
    }
}

/// Verify that naive and strided implementations produce identical results.
fn verify_correctness(a: &StridedArray<f64>, dims: &[usize], perm: &[usize], label: &str) {
    let total: usize = dims.iter().product();
    let mut b_naive = StridedArray::<f64>::col_major(dims);
    let mut b_strided = StridedArray::<f64>::col_major(dims);

    unsafe {
        naive_permute_nd(
            a.data().as_ptr(),
            b_naive.data_mut().as_mut_ptr(),
            dims,
            perm,
        );
    }

    let a_perm = a.view().permute(perm).unwrap();
    copy_into(&mut b_strided.view_mut(), &a_perm).unwrap();

    for i in 0..total {
        assert_eq!(
            b_naive.data()[i],
            b_strided.data()[i],
            "Mismatch at element {} for permutation '{}': naive={} strided={}",
            i,
            label,
            b_naive.data()[i],
            b_strided.data()[i]
        );
    }
    println!("  correctness check passed for '{label}'");
}

fn benchmark_permutation(
    a: &StridedArray<f64>,
    b: &mut StridedArray<f64>,
    perm: &[usize],
    label: &str,
) {
    println!("=== permute_rank25_{label} ===");
    println!("Permutation: {perm:?}");

    let a_view = a.view();
    let a_perm = a_view.permute(perm).unwrap();
    let a_ptr = a.data().as_ptr();
    let b_ptr = b.data_mut().as_mut_ptr();
    let dims = a.dims().to_vec();

    bench_n("rust_naive", 1, 3, || {
        unsafe { naive_permute_nd(a_ptr, b_ptr, &dims, perm) };
        black_box(b_ptr);
    });

    bench_n("rust_strided", 1, 3, || {
        copy_into(&mut b.view_mut(), &a_perm).unwrap();
        black_box(b.data().as_ptr());
    });
    println!();
}

fn main() {
    let rank = 25;
    let total: usize = 1 << rank;
    let dims = vec![2usize; rank];

    println!("Rank-25 tensor permutation benchmarks (Issue #37)");
    println!(
        "Tensor shape: [2; 25], total elements: 2^25 = {}, memory per array: {} MB",
        total,
        total * 8 / (1024 * 1024)
    );
    println!();

    let a = make_random_rank25(42);
    let mut b = StridedArray::<f64>::col_major(&dims);

    // --- Contiguous copy baseline (memcpy lower bound) ---
    {
        println!("=== contiguous_copy_baseline ===");
        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();
        bench_n("memcpy", 1, 3, || {
            unsafe { std::ptr::copy_nonoverlapping(a_ptr, b_ptr, total) };
            black_box(b_ptr);
        });
        println!();
    }

    // --- Define permutations ---
    // 1) Full reversal: worst-case, no dimension fusion possible
    let perm_reverse: Vec<usize> = (0..rank).rev().collect();

    // 2) Cyclic shift by 1: near-best, most dims fuse into ~2D operation
    let perm_cyclic: Vec<usize> = {
        let mut p: Vec<usize> = (1..rank).collect();
        p.push(0);
        p
    };

    // 3) Pairwise swap: quantum-circuit-relevant (2-qubit gate pattern)
    let perm_pairwise: Vec<usize> = {
        let mut p = vec![0usize; rank];
        for i in (0..rank - 1).step_by(2) {
            p[i] = i + 1;
            p[i + 1] = i;
        }
        // Last dimension (24) stays in place (odd rank)
        p[rank - 1] = rank - 1;
        p
    };

    // --- Correctness checks ---
    println!("--- Correctness verification ---");
    verify_correctness(&a, &dims, &perm_reverse, "reverse");
    verify_correctness(&a, &dims, &perm_cyclic, "cyclic_shift_1");
    verify_correctness(&a, &dims, &perm_pairwise, "pairwise_swap");
    println!();

    // --- Benchmarks ---
    benchmark_permutation(&a, &mut b, &perm_reverse, "reverse");
    benchmark_permutation(&a, &mut b, &perm_cyclic, "cyclic_shift_1");
    benchmark_permutation(&a, &mut b, &perm_pairwise, "pairwise_swap");
}
