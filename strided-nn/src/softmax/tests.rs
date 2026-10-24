use core::mem::MaybeUninit;

use strided_basic::ExecContext;

use super::*;

/// Run one dtype's entry and read back the output.
fn run<T: Copy>(
    call: impl FnOnce(&mut [MaybeUninit<T>]) -> Result<(), SoftmaxError>,
    len: usize,
) -> Vec<T> {
    let mut output = vec![MaybeUninit::<T>::uninit(); len];
    call(&mut output).expect("valid call");
    // SAFETY: the kernel wrote every element.
    unsafe { core::slice::from_raw_parts(output.as_ptr().cast::<T>(), len) }.to_vec()
}

fn reference(input: &[f64], axis_len: usize) -> Vec<f64> {
    let mut out = input.to_vec();
    for column in out.chunks_mut(axis_len) {
        let max = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for value in column.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in column.iter_mut() {
            *value /= sum;
        }
    }
    out
}

#[test]
fn columns_normalize_in_both_dtypes() {
    let xs = [
        1.0f64, 2.0, 3.0, -1.0, 0.0, 1.0, 0.5, -0.5, 4.0, 2.0, 1.0, 0.0,
    ];
    let want = reference(&xs, 3);
    for (got, &want) in run(
        |out| {
            let mut scratch = vec![0.0f64; scratch_len::<f64>(3, xs.len(), 2)];
            softmax_axis0_f64(ExecContext::serial(), 3, &xs, out, &mut scratch)
        },
        xs.len(),
    )
    .iter()
    .zip(&want)
    {
        assert!((got - want).abs() < 1e-12, "f64: {got} vs {want}");
    }
    let xs32: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
    let got = run(
        |out| {
            let mut scratch = vec![0.0f32; scratch_len::<f32>(3, xs32.len(), 2)];
            softmax_axis0_f32(ExecContext::serial(), 3, &xs32, out, &mut scratch)
        },
        xs32.len(),
    );
    for (got, &want) in got.iter().zip(&want) {
        assert!(
            (f64::from(*got) - want).abs() < 1e-6,
            "f32: {got} vs {want}"
        );
    }
}

#[test]
fn multi_worker_scratch_matches_the_serial_result() {
    // Enough columns for several tasks and a ragged tail.
    let axis_len = 17;
    let columns = 4_000;
    let xs: Vec<f32> = (0..axis_len * columns)
        .map(|i| ((i % axis_len) as f32) * 0.25 - 1.0)
        .collect();
    let want = run(
        |out| {
            let mut scratch = vec![0.0f32; scratch_len::<f32>(axis_len, xs.len(), 1)];
            softmax_axis0_f32(ExecContext::serial(), axis_len, &xs, out, &mut scratch)
        },
        xs.len(),
    );
    let got = run(
        |out| {
            let mut scratch = vec![0.0f32; scratch_len::<f32>(axis_len, xs.len(), 4)];
            softmax_axis0_f32(
                ExecContext::max_threads(4).unwrap(),
                axis_len,
                &xs,
                out,
                &mut scratch,
            )
        },
        xs.len(),
    );
    assert_eq!(got, want);
    for column in got.chunks(axis_len) {
        let sum: f32 = column.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "column sums to {sum}");
    }
}

#[test]
fn a_nan_column_stays_nan_and_leaves_the_others_alone() {
    let axis_len = 3;
    let xs = [f32::NAN, 1.0, 2.0, 1.0, 2.0, 3.0];
    let got = run(
        |out| {
            let mut scratch = vec![0.0f32; scratch_len::<f32>(axis_len, xs.len(), 1)];
            softmax_axis0_f32(ExecContext::serial(), axis_len, &xs, out, &mut scratch)
        },
        xs.len(),
    );
    assert!(got[..3].iter().all(|value| value.is_nan()));
    let sum: f32 = got[3..].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn contract_violations_are_typed_errors() {
    let xs = [1.0f32, 2.0, 3.0];
    let mut out = [MaybeUninit::<f32>::uninit(); 3];
    let mut scratch = vec![0.0f32; 1024];
    assert_eq!(
        softmax_axis0_f32(ExecContext::serial(), 0, &xs, &mut out, &mut scratch),
        Err(SoftmaxError::EmptyAxis)
    );
    assert_eq!(
        softmax_axis0_f32(ExecContext::serial(), 2, &xs, &mut out, &mut scratch),
        Err(SoftmaxError::UnevenColumns {
            len: 3,
            axis_len: 2
        })
    );
    let mut short = [MaybeUninit::<f32>::uninit(); 2];
    assert_eq!(
        softmax_axis0_f32(ExecContext::serial(), 3, &xs, &mut short, &mut scratch),
        Err(SoftmaxError::LengthMismatch {
            input: 3,
            output: 2
        })
    );
    let mut empty: [f32; 0] = [];
    assert!(matches!(
        softmax_axis0_f32(ExecContext::serial(), 3, &xs, &mut out, &mut empty),
        Err(SoftmaxError::ScratchTooSmall { .. })
    ));
    // An empty input is a valid no-op, even without scratch.
    assert_eq!(
        softmax_axis0_f32(ExecContext::serial(), 3, &[], &mut [], &mut empty),
        Ok(())
    );
}

#[test]
fn scratch_len_covers_one_group_per_task() {
    // A long column stays whole, so one group is one column.
    assert_eq!(scratch_len::<f64>(1_000_000, 1_000_000, 1), 1_000_000);
    assert_eq!(scratch_len::<f64>(1_000_000, 3_000_000, 3), 3_000_000);
    // A wide pool on one column reserves one task, not one region per worker.
    assert_eq!(scratch_len::<f64>(1_000_000, 1_000_000, 64), 1_000_000);
    // A short column is grouped up to the cache-sized bound.
    assert_eq!(scratch_len::<f32>(64, 8_192 * 2, 2), 2 * 8_192);
    assert_eq!(scratch_len::<f64>(64, 4_096, 1), 4_096);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_group_kernels_match_the_scalar_reference() {
    if !simd::avx2_available() {
        return;
    }
    let src = [
        1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0, 5.0, 0.5, -2.0, 1.0, 1.0, 1.0,
    ];
    let mut scalar = [0.0f32; 12];
    shift_group_f32(&src, &mut scalar, 3);
    let mut vector = [0.0f32; 12];
    // SAFETY: guarded by the runtime AVX2 check above.
    unsafe { simd::shift_group_f32_avx2(&src, &mut vector, 3) };
    assert_eq!(scalar, vector);

    let mut a = src;
    let mut b = src;
    // SAFETY: guarded by the runtime AVX2 check above.
    unsafe { simd::scale_group_f32_avx2(&mut a, 3) };
    scale_group_f32(&mut b, 3);
    assert_eq!(a, b);
}
