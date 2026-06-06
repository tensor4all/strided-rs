use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_kernel::{
    batched_outer_product_into, zip_map2_into, Identity, StridedArray, StridedView, StridedViewMut,
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

fn make_random_col_major(dims: &[usize], seed: u64) -> StridedArray<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    StridedArray::<f64>::from_fn_col_major(dims, |_| rng.gen::<f64>())
}

fn main() {
    let j = 16usize;
    let k = 16usize;
    let o = 64usize;
    let t = 64usize;

    println!("Rust runner: benches/outer_product.rs");
    println!("Case: C[j,k,o,t] = A[k,j,t] * B[o,t], column-major logical output");
    println!();

    let lhs_compact = make_random_col_major(&[j, k, t], 1);
    let lhs_noncompact_base = make_random_col_major(&[k, j, t], 2);
    let lhs_noncompact = lhs_noncompact_base.view().permute(&[1, 0, 2]).unwrap();
    let rhs = make_random_col_major(&[o, t], 3);
    let mut out = StridedArray::<f64>::col_major(&[j, k, o, t]);
    let mut out_torchlike = StridedArray::<f64>::col_major(&[j, k, o, t]);
    let torchlike_out_strides = [j as isize, 1, (j * k) as isize, (j * k * o) as isize];

    println!("lhs compact strides: {:?}", lhs_compact.view().strides());
    println!("lhs noncompact strides: {:?}", lhs_noncompact.strides());
    println!("rhs strides: {:?}", rhs.view().strides());
    println!("out strides: {:?}", out.view().strides());
    println!();

    bench_n("batched_outer compact", 20, 100, || {
        batched_outer_product_into(&mut out.view_mut(), &lhs_compact.view(), &rhs.view(), 2, 1)
            .unwrap();
        black_box(out.data().as_ptr());
    });

    bench_n("batched_outer noncompact", 20, 100, || {
        batched_outer_product_into(&mut out.view_mut(), &lhs_noncompact, &rhs.view(), 2, 1)
            .unwrap();
        black_box(out.data().as_ptr());
    });

    bench_n("batched_outer noncompact torchlike output", 20, 100, || {
        let mut out_view = StridedViewMut::new(
            out_torchlike.data_mut(),
            &[j, k, o, t],
            &torchlike_out_strides,
            0,
        )
        .unwrap();
        batched_outer_product_into(&mut out_view, &lhs_noncompact, &rhs.view(), 2, 1).unwrap();
        black_box(out_torchlike.data().as_ptr());
    });

    let lhs_ptr = lhs_noncompact.ptr();
    let rhs_view = rhs.view();
    let rhs_ptr = rhs_view.ptr();
    let out_ptr = out.data_mut().as_mut_ptr();
    let out_torchlike_ptr = out_torchlike.data_mut().as_mut_ptr();
    let lhs_row_offsets: Vec<isize> = (0..j * k)
        .map(|row| ((row % j) * j + (row / j)) as isize)
        .collect();
    let mut lhs_values = Vec::with_capacity(j * k);
    bench_n("manual packed lhs noncompact", 20, 100, || {
        for batch in 0..t {
            lhs_values.clear();
            let lhs_batch = (batch * j * k) as isize;
            for &row_offset in &lhs_row_offsets {
                unsafe {
                    lhs_values.push(*lhs_ptr.offset(lhs_batch + row_offset));
                }
            }
            for col in 0..o {
                let rhs_value = unsafe { *rhs_ptr.offset((batch * o + col) as isize) };
                let dst_base = batch * j * k * o + col * j * k;
                for row in 0..j * k {
                    unsafe {
                        *out_ptr.add(dst_base + row) = lhs_values[row] * rhs_value;
                    }
                }
            }
        }
        black_box(out_ptr);
    });

    bench_n("manual torchlike output", 20, 100, || {
        for batch in 0..t {
            let lhs_batch = batch * j * k;
            let rhs_batch = batch * o;
            let dst_batch = batch * j * k * o;
            for col in 0..o {
                let rhs_value = unsafe { *rhs_ptr.add(rhs_batch + col) };
                let dst_col = dst_batch + col * j * k;
                for row_j in 0..j {
                    let lhs_row = lhs_batch + row_j * j;
                    let dst_row = dst_col + row_j * j;
                    for row_k in 0..k {
                        unsafe {
                            *out_torchlike_ptr.add(dst_row + row_k) =
                                *lhs_ptr.add(lhs_row + row_k) * rhs_value;
                        }
                    }
                }
            }
        }
        black_box(out_torchlike_ptr);
    });

    bench_n("manual torchlike output flat row", 20, 100, || {
        let rows = j * k;
        for batch in 0..t {
            let lhs_batch = batch * rows;
            let rhs_batch = batch * o;
            let dst_batch = batch * rows * o;
            for col in 0..o {
                let rhs_value = unsafe { *rhs_ptr.add(rhs_batch + col) };
                let dst_col = dst_batch + col * rows;
                for row in 0..rows {
                    unsafe {
                        *out_torchlike_ptr.add(dst_col + row) =
                            *lhs_ptr.add(lhs_batch + row) * rhs_value;
                    }
                }
            }
        }
        black_box(out_torchlike_ptr);
    });

    bench_n("manual torchlike output slices", 20, 100, || {
        let rows = j * k;
        let lhs_data = lhs_noncompact_base.data();
        let rhs_data = rhs.data();
        let dst_data = out_torchlike.data_mut();
        for batch in 0..t {
            let lhs_slice = &lhs_data[batch * rows..(batch + 1) * rows];
            let rhs_batch = batch * o;
            let dst_batch = batch * rows * o;
            for col in 0..o {
                let rhs_value = rhs_data[rhs_batch + col];
                let dst_col = dst_batch + col * rows;
                let dst_slice = &mut dst_data[dst_col..dst_col + rows];
                for (d, &x) in dst_slice.iter_mut().zip(lhs_slice.iter()) {
                    *d = x * rhs_value;
                }
            }
        }
        black_box(out_torchlike.data().as_ptr());
    });

    bench_n("manual colmajor output j-inner", 20, 100, || {
        let rows = j * k;
        for batch in 0..t {
            let lhs_batch = batch * rows;
            let rhs_batch = batch * o;
            let dst_batch = batch * rows * o;
            for col in 0..o {
                let rhs_value = unsafe { *rhs_ptr.add(rhs_batch + col) };
                let dst_col = dst_batch + col * rows;
                for row_k in 0..k {
                    let lhs_k = lhs_batch + row_k;
                    let dst_k = dst_col + row_k * j;
                    for row_j in 0..j {
                        unsafe {
                            *out_ptr.add(dst_k + row_j) =
                                *lhs_ptr.add(lhs_k + row_j * k) * rhs_value;
                        }
                    }
                }
            }
        }
        black_box(out_ptr);
    });

    bench_n("manual colmajor output k-inner", 20, 100, || {
        let rows = j * k;
        for batch in 0..t {
            let lhs_batch = batch * rows;
            let rhs_batch = batch * o;
            let dst_batch = batch * rows * o;
            for col in 0..o {
                let rhs_value = unsafe { *rhs_ptr.add(rhs_batch + col) };
                let dst_col = dst_batch + col * rows;
                for row_j in 0..j {
                    let lhs_j = lhs_batch + row_j * k;
                    let dst_j = dst_col + row_j;
                    for row_k in 0..k {
                        unsafe {
                            *out_ptr.add(dst_j + row_k * j) =
                                *lhs_ptr.add(lhs_j + row_k) * rhs_value;
                        }
                    }
                }
            }
        }
        black_box(out_ptr);
    });

    let lhs_broadcast_seed: StridedView<'_, f64, Identity> = StridedView::new(
        lhs_noncompact_base.data(),
        &[j, k, 1, t],
        &[j as isize, 1, 0, (j * k) as isize],
        0,
    )
    .unwrap();
    let lhs_broadcast = lhs_broadcast_seed.broadcast(&[j, k, o, t]).unwrap();
    let rhs_broadcast_seed: StridedView<'_, f64, Identity> =
        StridedView::new(rhs.data(), &[1, 1, o, t], &[0, 0, 1, o as isize], 0).unwrap();
    let rhs_broadcast = rhs_broadcast_seed.broadcast(&[j, k, o, t]).unwrap();

    bench_n("zip_map2 broadcast noncompact", 20, 100, || {
        zip_map2_into(
            &mut out.view_mut(),
            &lhs_broadcast,
            &rhs_broadcast,
            |x, y| x * y,
        )
        .unwrap();
        black_box(out.data().as_ptr());
    });

    bench_n(
        "zip_map2 broadcast noncompact torchlike output",
        20,
        100,
        || {
            let mut out_view = StridedViewMut::new(
                out_torchlike.data_mut(),
                &[j, k, o, t],
                &torchlike_out_strides,
                0,
            )
            .unwrap();
            zip_map2_into(&mut out_view, &lhs_broadcast, &rhs_broadcast, |x, y| x * y).unwrap();
            black_box(out_torchlike.data().as_ptr());
        },
    );
}
