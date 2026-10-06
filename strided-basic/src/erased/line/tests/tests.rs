//! Serial and parallel replay of the single-axis plans agree bitwise.

use crate::threading::run_serial_and_parallel;
use crate::*;

/// Column-major strides of `dims`.
fn strides(dims: &[usize]) -> Vec<isize> {
    let mut stride = 1isize;
    dims.iter()
        .map(|&dim| {
            let current = stride;
            stride *= dim as isize;
            current
        })
        .collect()
}

fn data(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 7919) % 1009) as f64 / 97.0 - 5.0)
        .collect()
}

/// Shapes above the threading threshold in line mode (axis 0) and panel mode
/// (axis 1, with a contiguous leading axis wider than one panel).
const CASES: [([usize; 2], usize); 3] = [([4096, 16], 0), ([130, 512], 1), ([600, 70], 1)];

#[test]
fn scan_parallel_matches_serial() {
    for (dims, axis) in CASES {
        let st = strides(&dims);
        let src = data(dims.iter().product());
        let plan = ErasedScanPlan::compile(
            KernelDType::F64,
            ScanOp::Sum,
            &dims,
            &st,
            &st,
            axis,
            ScanOptions::new().reverse(true),
        )
        .unwrap();
        let (serial, parallel) = run_serial_and_parallel(|| {
            let mut out = vec![0.0f64; src.len()];
            let src = ErasedRawStridedRef::from_slice(&src, &dims, &st, 0).unwrap();
            let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &st, 0).unwrap();
            plan.execute(&ExecContext::ambient(), &mut dest, &src)
                .unwrap();
            out
        });
        assert_eq!(serial, parallel);
    }
}

#[test]
fn argreduce_parallel_matches_serial() {
    for (dims, axis) in CASES {
        let st = strides(&dims);
        let src = data(dims.iter().product());
        let out_dims = [dims[1 - axis]];
        let plan = ErasedArgReducePlan::compile(
            KernelDType::F64,
            KernelDType::I64,
            ArgReduceOp::MaxAbs,
            &dims,
            &st,
            &out_dims,
            &[1],
            axis,
        )
        .unwrap();
        let (serial, parallel) = run_serial_and_parallel(|| {
            let mut out = vec![-1i64; out_dims[0]];
            let src = ErasedRawStridedRef::from_slice(&src, &dims, &st, 0).unwrap();
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out, &out_dims, &[1], 0).unwrap();
            plan.execute(&ExecContext::ambient(), &mut dest, &src)
                .unwrap();
            out
        });
        assert_eq!(serial, parallel);
        assert!(serial.iter().all(|&i| i >= 0));
    }
}

#[test]
fn norm_parallel_matches_serial() {
    for (dims, axis) in CASES {
        let st = strides(&dims);
        let src = data(dims.iter().product());
        let weight = data(dims[axis]);
        let spec = NormSpec::layer_norm(1e-5).with_weight(1).with_bias(1);
        let plan = ErasedNormPlan::compile(KernelDType::F64, spec, &dims, &st, &st, axis).unwrap();
        let (serial, parallel) = run_serial_and_parallel(|| {
            let mut out = vec![0.0f64; src.len()];
            let src = ErasedRawStridedRef::from_slice(&src, &dims, &st, 0).unwrap();
            let w = ErasedRawStridedRef::from_slice(&weight, &dims[axis..=axis], &[1], 0).unwrap();
            let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &st, 0).unwrap();
            plan.execute(&ExecContext::ambient(), &mut dest, &src, Some(&w), Some(&w))
                .unwrap();
            out
        });
        assert_eq!(
            serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
