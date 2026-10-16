//! Chunked (vectorized) replay of a fused plan, against the per-element replay.
//!
//! Every plan here has ten instructions and therefore no static specialization
//! kind, so the interpreter — and with it the chunked replay — is the only path.
//! The per-element replay is produced by giving one operand a stride of 2, which
//! makes every block non-contiguous for that operand and leaves the plan on the
//! scalar path (tensor4all/strided-rs#304).

use core::mem::MaybeUninit;
use num_traits::Float;
use strided_basic::{StridedView, StridedViewMut};
use strided_fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan, FusedScalar};

/// Ten unary instructions over the same input; `op_arity(op)` is 1.
fn ten_unary(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 1,
        outputs: vec![10],
        ops: (0..10)
            .map(|index| FusedInst {
                op,
                inputs: vec![index],
            })
            .collect(),
    }
}

/// Ten binary instructions, each multiplying the running value by the second
/// input.
fn ten_binary(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![11],
        ops: (0..10)
            .map(|index| FusedInst {
                op,
                // Instruction `k` writes register `2 + k` and reads the previous
                // running value plus the second input.
                inputs: vec![if index == 0 { 0 } else { 1 + index }, 1],
            })
            .collect(),
    }
}

/// The same logical values with a stride of 2, so no block is contiguous for
/// this operand.
fn gapped<T: FusedScalar>(values: &[T]) -> Vec<T> {
    let mut buffer = values.to_vec();
    buffer.resize(values.len() * 2 - 1, values[0]);
    for index in (0..values.len()).rev() {
        buffer[index * 2] = values[index];
    }
    buffer
}

/// `op` over the plan's inputs, preceded by an exact `Conj` of the first input:
/// two instructions are never a static specialization kind, so the plan must go
/// through the chunked replay.
fn unspecialized(op: FusedOp, arity: usize) -> FusedPlan {
    let conj = arity;
    let inputs: Vec<usize> = (0..arity)
        .map(|index| if index == 0 { conj } else { index })
        .collect();
    FusedPlan {
        input_count: arity,
        outputs: vec![conj + 1],
        ops: vec![
            FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0],
            },
            FusedInst { op, inputs },
        ],
    }
}

/// Replay `plan` over the operands, each with its own element stride.
fn replay<T: FusedScalar>(plan: &FusedPlan, len: usize, operands: &[(&[T], isize)]) -> Vec<T> {
    let dims = [len];
    let views: Vec<StridedView<'_, T>> = operands
        .iter()
        .map(|(data, stride)| {
            StridedView::new_owned(data, dims.to_vec(), vec![*stride], 0).unwrap()
        })
        .collect();
    let mut output = vec![operands[0].0[0]; len];
    {
        let mut dest = StridedViewMut::new(&mut output, &dims, &[1isize], 0).unwrap();
        fused_elementwise_into(&mut [dest], &views, plan).unwrap();
    }
    output
}

/// Relative distance between `a` and `b` in units of `T::epsilon`.
fn ulps<T: Float>(a: T, b: T) -> f64 {
    let scale = a.abs().max(b.abs()).max(T::min_positive_value());
    ((a - b).abs() / scale / T::epsilon()).to_f64().unwrap()
}

fn assert_exact<T: FusedScalar + PartialEq + std::fmt::Debug>(chunked: &[T], scalar: &[T]) {
    assert_eq!(chunked.len(), scalar.len());
    for (index, (a, b)) in chunked.iter().zip(scalar).enumerate() {
        assert_eq!(a, b, "element {index}");
    }
}

/// The chunked replay must stay within two ULP of the per-element replay for a
/// single analytic kernel. A ten-instruction chain multiplies the per-kernel
/// difference by the output magnitude (an exponential chain grows the absolute
/// error with it), so a chain is compared on the dtype-scaled budget below.
fn assert_within_ulps<T: FusedScalar + Float + std::fmt::Display>(
    chunked: &[T],
    scalar: &[T],
    budget: f64,
) {
    assert_eq!(chunked.len(), scalar.len());
    for (index, (a, b)) in chunked.iter().zip(scalar).enumerate() {
        if a == b || (a.is_nan() && b.is_nan()) {
            continue;
        }
        let measured = ulps(*a, *b);
        assert!(
            measured <= budget,
            "element {index}: {a} vs {b} is {measured} ULP"
        );
    }
}

/// Lengths below, at, just above and well above one chunk, plus a full
/// chunk followed by a tail.
const LENGTHS: [usize; 6] = [63, 64, 65, 1000, 1025, 2049];

#[test]
fn chunked_replay_is_bitwise_identical_for_arithmetic_chains() {
    for length in LENGTHS {
        let a: Vec<f64> = (0..length).map(|v| v as f64 * 0.5 - 3.0).collect();
        let b: Vec<f64> = (0..length).map(|v| (v % 7) as f64 - 2.5).collect();
        let gapped_a = gapped(&a);
        let plan = ten_binary(FusedOp::Multiply);
        let chunked = replay(&plan, length, &[(&a, 1), (&b, 1)]);
        let scalar = replay(&plan, length, &[(&gapped_a, 2), (&b, 1)]);
        assert_exact(&chunked, &scalar);

        let plan = ten_binary(FusedOp::Add);
        let chunked = replay(&plan, length, &[(&a, 1), (&b, 1)]);
        let scalar = replay(&plan, length, &[(&gapped_a, 2), (&b, 1)]);
        assert_exact(&chunked, &scalar);
    }
}

#[test]
fn chunked_replay_is_bitwise_identical_for_exact_unary_ops() {
    for op in [FusedOp::Negate, FusedOp::Conj, FusedOp::Abs, FusedOp::Sqrt] {
        for length in LENGTHS {
            let a: Vec<f64> = (0..length).map(|v| v as f64 + 0.25).collect();
            let gapped_a = gapped(&a);
            let plan = ten_unary(op);
            let chunked = replay(&plan, length, &[(&a, 1)]);
            let scalar = replay(&plan, length, &[(&gapped_a, 2)]);
            assert_exact(&chunked, &scalar);
        }
    }
}

#[test]
fn chunked_replay_tracks_the_scalar_replay_for_analytic_ops() {
    for op in [
        FusedOp::Exp,
        FusedOp::Log,
        FusedOp::Sin,
        FusedOp::Cos,
        FusedOp::Tanh,
        FusedOp::Expm1,
        FusedOp::Log1p,
    ] {
        for length in LENGTHS {
            let a: Vec<f64> = (0..length).map(|v| 0.05 + (v % 11) as f64 * 0.1).collect();
            let gapped_a = gapped(&a);
            // Two ULP per instruction; a ten-instruction chain amplifies the
            // per-kernel difference by the output magnitude.
            let chain_budget = 512.0;
            let plan = ten_unary(op);
            let chunked = replay(&plan, length, &[(&a, 1)]);
            let scalar = replay(&plan, length, &[(&gapped_a, 2)]);
            assert_within_ulps(&chunked, &scalar, chain_budget);

            // `Log1p` is exact here because it keeps the scalar chunk loop.
            // A single instruction stays inside two ULP. `Conj` is exact over
            // `f64`, so the plan applies `op` and still cannot be a static
            // specialization kind.
            let single = unspecialized(op, 1);
            let chunked = replay(&single, length, &[(&a, 1)]);
            let scalar = replay(&single, length, &[(&gapped_a, 2)]);
            assert_within_ulps(&chunked, &scalar, 2.0);
        }
    }
}

#[test]
fn chunked_replay_of_the_reported_f32_chain_is_correct() {
    let length = 1024 * 64;
    let a: Vec<f32> = (0..length)
        .map(|v| ((v % 97) as f32 - 48.0) * 0.1)
        .collect();
    let gapped_a = gapped(&a);
    for op in [FusedOp::Tanh, FusedOp::Multiply] {
        let plan = if op == FusedOp::Multiply {
            ten_binary(op)
        } else {
            ten_unary(op)
        };
        let operands: Vec<(&[f32], isize)> = if op == FusedOp::Multiply {
            vec![(&a, 1), (&a, 1)]
        } else {
            vec![(&a, 1)]
        };
        let chunked = replay(&plan, length, &operands);
        let operands: Vec<(&[f32], isize)> = if op == FusedOp::Multiply {
            vec![(&gapped_a, 2), (&a, 1)]
        } else {
            vec![(&gapped_a, 2)]
        };
        let scalar = replay(&plan, length, &operands);
        if op == FusedOp::Multiply {
            assert_exact(&chunked, &scalar);
        } else {
            assert_within_ulps(&chunked, &scalar, 20.0);
            // The acceptance comparison of #304: the fused replay agrees
            // bitwise with dispatching the same instruction ten times through
            // the same vector kernel.
            let reference = dispatch_ten_tanh_f32(&a);
            for (index, (fused, dispatched)) in chunked.iter().zip(&reference).enumerate() {
                assert_eq!(
                    fused.to_bits(),
                    dispatched.to_bits(),
                    "element {index}: {fused} vs {dispatched}"
                );
            }
        }
    }
}

/// Ten `elem_simd_math::f32::tanh` passes over the whole array: what a
/// per-instruction dispatch through the workspace's vector kernels does.
fn dispatch_ten_tanh_f32(values: &[f32]) -> Vec<f32> {
    let mut current = values.to_vec();
    let mut next = current.clone();
    for _ in 0..10 {
        let mut uninit = vec![MaybeUninit::<f32>::uninit(); current.len()];
        elem_simd_math::f32::tanh(&current, &mut uninit).unwrap();
        for (slot, value) in next.iter_mut().zip(uninit) {
            // SAFETY: the kernel writes every element before returning `Ok`.
            *slot = unsafe { value.assume_init() };
        }
        std::mem::swap(&mut current, &mut next);
    }
    current
}

#[test]
fn chunked_replay_propagates_special_values() {
    let specials = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        1.0,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    let b: Vec<f64> = specials.iter().map(|v| v * 2.0 + 0.5).collect();
    let gapped_a = gapped(&specials);
    for (op, arity) in [
        (FusedOp::Divide, 2),
        (FusedOp::Maximum, 2),
        (FusedOp::Minimum, 2),
        (FusedOp::Pow, 2),
        (FusedOp::Sqrt, 1),
        (FusedOp::Rsqrt, 1),
        (FusedOp::Tanh, 1),
        (FusedOp::Log, 1),
        (FusedOp::Expm1, 1),
        (FusedOp::Log1p, 1),
    ] {
        let plan = unspecialized(op, arity);
        let chunked = if arity == 2 {
            replay(&plan, specials.len(), &[(&specials, 1), (&b, 1)])
        } else {
            replay(&plan, specials.len(), &[(&specials, 1)])
        };
        let scalar = if arity == 2 {
            replay(&plan, specials.len(), &[(&gapped_a, 2), (&b, 1)])
        } else {
            replay(&plan, specials.len(), &[(&gapped_a, 2)])
        };
        // The analytic kernels are worth one ULP over the scalar functions;
        // everything else is exact in both replays.
        // `Log1p` is on the scalar chunk loop: `elem-simd-math 0.1.0` overflows
        // at `f64::MAX`, where `f64::ln_1p` returns `709.78`.
        let analytic = matches!(
            op,
            FusedOp::Exp
                | FusedOp::Log
                | FusedOp::Sin
                | FusedOp::Cos
                | FusedOp::Tanh
                | FusedOp::Expm1
        );
        for (index, (a, expected)) in chunked.iter().zip(&scalar).enumerate() {
            if expected.is_nan() {
                assert!(a.is_nan(), "{op:?} element {index}: {a} is not NaN");
                continue;
            }
            if !analytic {
                assert_eq!(
                    a.to_bits(),
                    expected.to_bits(),
                    "{op:?} element {index}: {a} vs {expected}"
                );
                continue;
            }
            if a == expected {
                continue;
            }
            let measured = ulps(*a, *expected);
            assert!(
                measured <= 2.0,
                "{op:?} element {index}: {a} vs {expected} is {measured} ULP"
            );
        }
    }
    // Clamp is the ternary instruction.
    let lo = vec![-1.0f64; specials.len()];
    let hi = vec![1.0f64; specials.len()];
    let plan = unspecialized(FusedOp::Clamp, 3);
    let chunked = replay(&plan, specials.len(), &[(&specials, 1), (&lo, 1), (&hi, 1)]);
    let scalar = replay(&plan, specials.len(), &[(&gapped_a, 2), (&lo, 1), (&hi, 1)]);
    assert_eq!(chunked.len(), scalar.len());
    for (index, (a, expected)) in chunked.iter().zip(&scalar).enumerate() {
        if expected.is_nan() {
            assert!(a.is_nan(), "clamp element {index}");
        } else {
            assert_eq!(a.to_bits(), expected.to_bits(), "clamp element {index}");
        }
    }
}

/// Replay `plan` over the operands, returning every destination.
fn replay_multi<T: FusedScalar>(
    plan: &FusedPlan,
    len: usize,
    operands: &[(&[T], isize)],
) -> Vec<Vec<T>> {
    let dims = [len];
    let views: Vec<StridedView<'_, T>> = operands
        .iter()
        .map(|(data, stride)| {
            StridedView::new_owned(data, dims.to_vec(), vec![*stride], 0).unwrap()
        })
        .collect();
    let mut outputs: Vec<Vec<T>> = (0..plan.outputs.len())
        .map(|_| vec![operands[0].0[0]; len])
        .collect();
    {
        let mut dests: Vec<StridedViewMut<'_, T>> = outputs
            .iter_mut()
            .map(|data| StridedViewMut::new(data, &dims, &[1isize], 0).unwrap())
            .collect();
        fused_elementwise_into(&mut dests, &views, plan).unwrap();
    }
    outputs
}

#[test]
fn chunked_replay_handles_multiple_outputs_and_reused_inputs() {
    let len = 130;
    let a: Vec<f64> = (0..len).map(|v| v as f64 * 0.25 + 0.5).collect();
    let gapped_a = gapped(&a);
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![0, 2, 4],
        ops: vec![
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 0],
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![1, 0],
            },
            FusedInst {
                op: FusedOp::Sqrt,
                inputs: vec![2],
            },
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![3],
            },
        ],
    };
    let chunked = replay_multi(&plan, len, &[(&a, 1)]);
    let scalar = replay_multi(&plan, len, &[(&gapped_a, 2)]);
    assert_eq!(chunked.len(), scalar.len());
    for (output, (chunked, scalar)) in chunked.iter().zip(&scalar).enumerate() {
        for (index, (a, b)) in chunked.iter().zip(scalar).enumerate() {
            assert_eq!(a, b, "output {output} element {index}");
        }
    }
}

/// Several callbacks, each with its own base offset and a tail chunk: the inner
/// axis is unit-stride, so every block is contiguous, while the outer stride is
/// gapped, so the traversal walks one run per column with a different base.
#[test]
fn chunked_replay_handles_several_blocks_with_different_bases() {
    let (rows, cols) = (7usize, 300usize);
    let row_gap = rows + 5;
    let dims = [rows, cols];
    let gapped_strides = [1isize, row_gap as isize];
    let dense_strides = [1isize, rows as isize];
    let plan = ten_binary(FusedOp::Add);

    let mut gapped_lhs = vec![0.0f64; cols * row_gap];
    let mut gapped_rhs = vec![0.0f64; cols * row_gap];
    let mut dense_lhs = vec![0.0f64; rows * cols];
    let mut dense_rhs = vec![0.0f64; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            let value_lhs = (col * rows + row) as f64 * 0.01 - 1.0;
            let value_rhs = 1.5 - (col as f64) * 0.002;
            gapped_lhs[col * row_gap + row] = value_lhs;
            gapped_rhs[col * row_gap + row] = value_rhs;
            dense_lhs[col * rows + row] = value_lhs;
            dense_rhs[col * rows + row] = value_rhs;
        }
    }

    let run = |lhs: &[f64], rhs: &[f64], strides: [isize; 2]| -> Vec<f64> {
        let mut output = vec![0.0f64; cols * row_gap];
        {
            let inputs = [
                StridedView::new_owned(lhs, dims.to_vec(), strides.to_vec(), 0).unwrap(),
                StridedView::new_owned(rhs, dims.to_vec(), strides.to_vec(), 0).unwrap(),
            ];
            let mut dest = StridedViewMut::new(&mut output, &dims, &gapped_strides, 0).unwrap();
            fused_elementwise_into(&mut [dest], &inputs, &plan).unwrap();
        }
        output
    };

    let chunked = run(&dense_lhs, &dense_rhs, dense_strides);
    let scalar = run(&gapped_lhs, &gapped_rhs, gapped_strides);
    for col in 0..cols {
        for row in 0..rows {
            let index = col * row_gap + row;
            assert_eq!(chunked[index], scalar[index], "element ({row}, {col})");
        }
    }
}

/// The erased uninitialized entry, which is what tenferro's fused lane calls,
/// agrees bitwise with ten per-instruction `tanh` passes.
#[test]
fn erased_uninitialized_replay_matches_per_instruction_dispatch() {
    use strided_basic::{
        ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ExecContext,
        KernelDType,
    };
    use strided_fused::ErasedFusedPlan;

    let length = 1024 + 7;
    let values: Vec<f32> = (0..length)
        .map(|v| ((v % 97) as f32 - 48.0) * 0.1)
        .collect();
    let dims = [length];
    let strides = [1isize];
    let plan = ten_unary(FusedOp::Tanh);
    let erased = ErasedFusedPlan::compile(KernelDType::F32, plan).unwrap();
    let input = ErasedRawStridedRef::from_slice(&values, &dims, &strides, 0).unwrap();
    let pointers = [ErasedRawStridedPtr::from_ref(&input)];
    let mut output = vec![MaybeUninit::<f32>::uninit(); length];
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut output, &dims, &strides, 0).unwrap();
    erased
        .execute_uninit(&ExecContext::serial(), &mut dest, &pointers)
        .unwrap();

    let reference = dispatch_ten_tanh_f32(&values);
    for (index, (slot, expected)) in output.iter().zip(&reference).enumerate() {
        // SAFETY: `execute_uninit` writes every reachable element.
        let got = unsafe { slot.assume_init() };
        assert_eq!(got.to_bits(), expected.to_bits(), "element {index}");
    }
}

/// `outputs = [0]` copies the first input while the instruction negates the
/// second: the first input is read exactly once, but the store step reads that
/// register too, so it must stay in the register file.
#[test]
fn a_destination_that_names_an_input_register_reads_the_loaded_chunk() {
    let a: Vec<f64> = (0..2050).map(|v| v as f64 * 0.25).collect();
    let b = vec![1.0f64; a.len()];
    let gapped_a = gapped(&a);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![0],
        ops: vec![FusedInst {
            op: FusedOp::Negate,
            inputs: vec![0],
        }],
    };
    let chunked = replay(&plan, a.len(), &[(&a, 1), (&b, 1)]);
    let scalar = replay(&plan, a.len(), &[(&gapped_a, 2), (&b, 1)]);
    assert_exact(&chunked, &scalar);
    assert_exact(&chunked, &a);
}
