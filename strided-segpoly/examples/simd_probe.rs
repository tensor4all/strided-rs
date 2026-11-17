//! Probe for `scripts/check-runtime-simd-dispatch.py` (tensor4all/strided-rs#496).
//!
//! It runs one representative lane body of each runtime-dispatched SIMD path in
//! `strided_segpoly::simd`: the `f64` body of a plan whose operand lanes are
//! contiguous, and the `f32` body of a plan whose operand lanes are strided (so
//! the replay packs them first). A baseline release build (no `-C target-cpu`)
//! must hold AVX2 and AVX-512 versions of each body; the script checks that in
//! the disassembly. Running the binary only checks the results.

use strided_basic::{Exec, RawStridedMut, RawStridedRef};
use strided_segpoly::{
    lane_width, IndexMode, Layout, Operation, Path, SegPolyElem, SegPolyPlan, Spec,
};

const BATCH: usize = 4;
const SEGMENTS: usize = 3;
const U: usize = 12;
const COEFFS: [f64; SEGMENTS] = [1.0, -0.5, 2.0];

fn main() {
    let f64_stride = 1;
    let f64_values: Vec<f64> = (0..BATCH * SEGMENTS * U)
        .map(|index| 0.5 + (index % 17) as f64)
        .collect();
    run(f64_values, f64_stride, "f64");

    // A lane stride the vector loop cannot take in place: the executor packs it.
    let f32_stride = 3;
    let f32_values: Vec<f32> = (0..BATCH * SEGMENTS * ((U - 1) * f32_stride + 1))
        .map(|index| 0.5 + (index % 17) as f32)
        .collect();
    run(f32_values, f32_stride, "f32");

    println!(
        "lane widths: f64 {} f32 {}",
        lane_width::<f64>(),
        lane_width::<f32>()
    );
}

fn run<T: SegPolyElem + PartialEq>(x: Vec<T>, lane_stride: usize, label: &str) {
    let span = (U - 1) * lane_stride + 1;
    let w: Vec<T> = (0..SEGMENTS * U)
        .map(|index| T::from_coefficient(0.25 + (index % 11) as f64))
        .collect();
    let mut out = vec![T::zero(); BATCH * SEGMENTS * U];
    execute(&x, &w, lane_stride as isize, &mut out);
    // The same sum in the documented order: paths, then rows, then lanes.
    let mut expected = vec![T::zero(); BATCH * SEGMENTS * U];
    for segment in 0..SEGMENTS {
        let coefficient = T::from_coefficient(COEFFS[segment]);
        for batch in 0..BATCH {
            for lane in 0..U {
                let index = batch * SEGMENTS * span + segment * span + lane * lane_stride;
                let term = coefficient * x[index] * w[segment * U + lane];
                let slot = batch * SEGMENTS * U + segment * U + lane;
                expected[slot] = if segment == 0 {
                    term
                } else {
                    expected[slot] + term
                };
            }
        }
    }
    assert_eq!(
        out, expected,
        "the dispatched {label} lane body moved the result"
    );
}

/// The plan both probes run through: three paths, one per output segment.
fn execute<T: SegPolyElem>(x: &[T], w: &[T], lane_stride: isize, out: &mut [T]) {
    let span = (U as isize - 1) * lane_stride + 1;
    let x_dims = [BATCH, SEGMENTS, U];
    let x_strides = [SEGMENTS as isize * span, span, lane_stride];
    let w_dims = [1usize, SEGMENTS, U];
    let w_strides = [U as isize, U as isize, 1];
    let out_dims = [BATCH, SEGMENTS, U];
    let out_strides = [(SEGMENTS * U) as isize, U as isize, 1];
    let operations = [Operation {
        buffers: &[0, 1, 2],
        path_start: 0,
        num_paths: SEGMENTS as u32,
    }];
    let segment_ids: Vec<[u32; SEGMENTS]> = (0..SEGMENTS).map(|s| [s as u32; 3]).collect();
    let segments: Vec<&[u32]> = segment_ids.iter().map(|ids| &ids[..]).collect();
    let paths: Vec<Path<'_>> = segments
        .iter()
        .zip(COEFFS.iter())
        .map(|(&segments, &coefficient)| Path {
            segments,
            coefficient,
        })
        .collect();
    let indexes = [
        &[IndexMode::Batched][..],
        &[IndexMode::Shared][..],
        &[IndexMode::Batched][..],
    ];
    let descriptor = Spec {
        num_batch_axes: 1,
        operations: &operations,
        paths: &paths,
        indexes: &indexes,
    };
    let plan = SegPolyPlan::<T>::compile(
        &descriptor,
        &[
            Layout {
                dims: &x_dims,
                strides: &x_strides,
            },
            Layout {
                dims: &w_dims,
                strides: &w_strides,
            },
        ],
        &[Layout {
            dims: &out_dims,
            strides: &out_strides,
        }],
    )
    .expect("the probe plan compiles");
    let mut scratch = vec![T::zero(); plan.scratch_len(1)];
    let inputs = [
        RawStridedRef::new(x, &x_dims, &x_strides, 0).expect("x view"),
        RawStridedRef::new(w, &w_dims, &w_strides, 0).expect("w view"),
    ];
    let mut outputs = [RawStridedMut::new(out, &out_dims, &out_strides, 0).expect("out view")];
    plan.execute_into(
        &mut outputs,
        &inputs,
        descriptor.indexes,
        &mut scratch,
        Exec::serial(),
    )
    .expect("the probe plan executes");
}
