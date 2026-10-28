//! Probe for `scripts/check-runtime-simd-dispatch.py` (tensor4all/strided-rs#340).
//!
//! It runs one representative arithmetic body of each runtime-dispatched SIMD
//! path: a `map_into` square root (the path tenferro's `sqrt` takes), a
//! `zip_map2_into` sum, a chunk-replayed fused chain, the four-input fused
//! `sqrt`/`rsqrt` specialization and both softmax dtypes. A
//! baseline release build (no `-C target-cpu`) must hold AVX2 and AVX-512
//! versions of each body; the script checks that in the disassembly. Running the
//! binary checks only the results.

use strided_fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan};
use strided_kernel::{map_into, zip_map2_into, Exec, StridedView, StridedViewMut};

fn main() {
    let n = 4096;
    let x: Vec<f64> = (0..n)
        .map(|index| ((index % 64) * (index % 64)) as f64)
        .collect();
    let source = StridedView::<f64>::new(&x, &[n], &[1], 0).unwrap();

    let mut roots = vec![0.0f64; n];
    let mut dest = StridedViewMut::new(&mut roots, &[n], &[1], 0).unwrap();
    map_into(
        &mut dest,
        &source,
        |value: f64| value.sqrt(),
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(roots[67], 3.0);

    let mut sums = vec![0.0f64; n];
    let mut dest = StridedViewMut::new(&mut sums, &[n], &[1], 0).unwrap();
    zip_map2_into(
        &mut dest,
        &source,
        &source,
        |a: f64, b: f64| a + b,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(sums[3], 18.0);

    // sqrt(x) * x + x: three instructions, so no static specialization takes the
    // plan and the contiguous block goes through the chunk replay.
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![3].into_iter().collect(),
        ops: vec![
            FusedInst::new(FusedOp::Sqrt, [0]),
            FusedInst::new(FusedOp::Multiply, [1, 0]),
            FusedInst::new(FusedOp::Add, [2, 0]),
        ]
        .into_iter()
        .collect(),
    };
    let mut fused = vec![0.0f64; n];
    let dest = StridedViewMut::new(&mut fused, &[n], &[1], 0).unwrap();
    let views = [StridedView::<f64>::new(&x, &[n], &[1], 0).unwrap()];
    fused_elementwise_into(&mut [dest], &views, &plan, Exec::serial()).unwrap();
    assert_eq!(fused[2], 4.0 * 2.0 + 4.0);

    // rsqrt(sqrt(clamp(x / y, lo, hi))): the four-input static specialization
    // runs a contiguous `zip_map4` leaf.
    let plan = FusedPlan {
        input_count: 4,
        outputs: vec![8].into_iter().collect(),
        ops: vec![
            FusedInst::new(FusedOp::Divide, [0, 1]),
            FusedInst::new(FusedOp::Maximum, [4, 2]),
            FusedInst::new(FusedOp::Minimum, [5, 3]),
            FusedInst::new(FusedOp::Sqrt, [6]),
            FusedInst::new(FusedOp::Rsqrt, [7]),
        ]
        .into_iter()
        .collect(),
    };
    let ones = vec![1.0f64; n];
    let lo = vec![1.0f64; n];
    let hi = vec![16.0f64; n];
    let views = [
        source,
        StridedView::<f64>::new(&ones, &[n], &[1], 0).unwrap(),
        StridedView::<f64>::new(&lo, &[n], &[1], 0).unwrap(),
        StridedView::<f64>::new(&hi, &[n], &[1], 0).unwrap(),
    ];
    let mut clamped = vec![0.0f64; n];
    let dest = StridedViewMut::new(&mut clamped, &[n], &[1], 0).unwrap();
    fused_elementwise_into(&mut [dest], &views, &plan, Exec::serial()).unwrap();
    // x[3] = 9 stays inside [1, 16]: rsqrt(sqrt(9)) = 1/sqrt(3).
    assert!((clamped[3] - 1.0 / 3.0f64.sqrt()).abs() < 1e-15);

    let axis_len = 64;
    let logits32: Vec<f32> = (0..n).map(|index| (index % 7) as f32).collect();
    let mut probs32 = vec![core::mem::MaybeUninit::<f32>::uninit(); n];
    let mut scratch32 = vec![0.0f32; strided_nn::scratch_len::<f32>(axis_len, n, 1)];
    strided_nn::softmax_axis0_f32(
        axis_len,
        &logits32,
        &mut probs32,
        &mut scratch32,
        Exec::serial(),
    )
    .unwrap();
    let logits64: Vec<f64> = logits32.iter().map(|&value| f64::from(value)).collect();
    let mut probs64 = vec![core::mem::MaybeUninit::<f64>::uninit(); n];
    let mut scratch64 = vec![0.0f64; strided_nn::scratch_len::<f64>(axis_len, n, 1)];
    strided_nn::softmax_axis0_f64(
        axis_len,
        &logits64,
        &mut probs64,
        &mut scratch64,
        Exec::serial(),
    )
    .unwrap();
    // SAFETY: the kernels wrote every element.
    let column: f64 = probs64[..axis_len]
        .iter()
        .map(|value| unsafe { value.assume_init() })
        .sum();
    assert!((column - 1.0).abs() < 1e-12);
    println!("runtime SIMD probe: ok");
}
