//! Chunked (vectorized) replay of a fused plan, against the per-element replay.
//!
//! Every plan here has ten instructions and therefore no static specialization
//! kind, so the interpreter — and with it the chunked replay — is the only path.
//! The per-element replay is produced by giving one operand a stride of 2, which
//! makes every block non-contiguous for that operand and leaves the plan on the
//! scalar path (tensor4all/strided-rs#304).

use core::mem::MaybeUninit;
use num_traits::Float;
use strided_basic::{Exec, StridedView, StridedViewMut};
use strided_fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan, FusedScalar};

/// Ten unary instructions over the same input; `op_arity(op)` is 1.
fn ten_unary(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 1,
        outputs: vec![10].into_iter().collect(),
        ops: (0..10)
            .map(|index| FusedInst {
                op,
                inputs: vec![index].into_iter().collect(),
            })
            .collect(),
    }
}

/// Ten binary instructions, each multiplying the running value by the second
/// input.
fn ten_binary(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![11].into_iter().collect(),
        ops: (0..10)
            .map(|index| FusedInst {
                op,
                // Instruction `k` writes register `2 + k` and reads the previous
                // running value plus the second input.
                inputs: vec![if index == 0 { 0 } else { 1 + index }, 1]
                    .into_iter()
                    .collect(),
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
        outputs: vec![conj + 1].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst::new(op, inputs),
        ]
        .into_iter()
        .collect(),
    }
}

/// Replay `plan` over the operands, each with its own element stride.
fn replay<T: FusedScalar>(plan: &FusedPlan, len: usize, operands: &[(&[T], isize)]) -> Vec<T> {
    let dims = [len];
    let views: Vec<StridedView<'_, T>> = operands
        .iter()
        .map(|(data, stride)| StridedView::new(data, &dims, &[*stride], 0).unwrap())
        .collect();
    let mut output = vec![operands[0].0[0]; len];
    {
        let dest = StridedViewMut::new(&mut output, &dims, &[1isize], 0).unwrap();
        fused_elementwise_into(&mut [dest], &views, plan, Exec::serial()).unwrap();
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
        .map(|(data, stride)| StridedView::new(data, &dims, &[*stride], 0).unwrap())
        .collect();
    let mut outputs: Vec<Vec<T>> = (0..plan.outputs.len())
        .map(|_| vec![operands[0].0[0]; len])
        .collect();
    {
        let mut dests: Vec<StridedViewMut<'_, T>> = outputs
            .iter_mut()
            .map(|data| StridedViewMut::new(data, &dims, &[1isize], 0).unwrap())
            .collect();
        fused_elementwise_into(&mut dests, &views, plan, Exec::serial()).unwrap();
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
        outputs: vec![0, 2, 4].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![1, 0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Sqrt,
                inputs: vec![2].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![3].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
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
                StridedView::new(lhs, &dims, &strides, 0).unwrap(),
                StridedView::new(rhs, &dims, &strides, 0).unwrap(),
            ];
            let dest = StridedViewMut::new(&mut output, &dims, &gapped_strides, 0).unwrap();
            fused_elementwise_into(&mut [dest], &inputs, &plan, Exec::serial()).unwrap();
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
        ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, KernelDType,
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
        .execute_uninit(&mut dest, &pointers, Exec::serial())
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
        outputs: vec![0].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Negate,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };
    let chunked = replay(&plan, a.len(), &[(&a, 1), (&b, 1)]);
    let scalar = replay(&plan, a.len(), &[(&gapped_a, 2), (&b, 1)]);
    assert_exact(&chunked, &scalar);
    assert_exact(&chunked, &a);
}

/// Replay a rank-1 plan over operands with explicit strides. Each destination's
/// stride is `dest_strides[i]`; the backing buffer is returned so a caller can
/// read logical element `j` at `buffer[j * stride]`.
fn run_rank1<T: FusedScalar>(
    plan: &FusedPlan,
    len: usize,
    operands: &[(&[T], isize)],
    dest_strides: &[isize],
) -> Vec<Vec<T>> {
    let dims = [len];
    let views: Vec<StridedView<'_, T>> = operands
        .iter()
        .map(|(data, stride)| StridedView::new(data, &dims, &[*stride], 0).unwrap())
        .collect();
    let seed = operands[0].0[0];
    let mut outputs: Vec<Vec<T>> = dest_strides
        .iter()
        .map(|&stride| {
            let span = if len == 0 {
                0
            } else {
                (len - 1) * stride as usize + 1
            };
            vec![seed; span]
        })
        .collect();
    {
        let dest_layouts: Vec<[isize; 1]> = dest_strides.iter().map(|&stride| [stride]).collect();
        let mut dests: Vec<StridedViewMut<'_, T>> = outputs
            .iter_mut()
            .zip(&dest_layouts)
            .map(|(data, strides)| StridedViewMut::new(data, &dims, strides, 0).unwrap())
            .collect();
        fused_elementwise_into(&mut dests, &views, plan, Exec::serial()).unwrap();
    }
    outputs
}

/// An input with stride 0 along the blocked axis must be replayed through the
/// chunk machine (not the interpreter) and must read the one broadcast value.
#[test]
fn chunked_replay_handles_inner_broadcast_inputs_for_f64_and_f32() {
    for length in LENGTHS {
        let b: Vec<f64> = (0..length).map(|v| (v % 13) as f64 * 0.25 - 1.0).collect();
        let a = [2.5f64];

        // Single use: `conj(a) * b` with `a` broadcast on the blocked axis.
        let single = FusedPlan {
            input_count: 2,
            outputs: vec![3].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Conj,
                    inputs: vec![0].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![2, 1].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        };
        let got = run_rank1(&single, length, &[(&a, 0), (&b, 1)], &[1]);
        let want: Vec<f64> = b.iter().map(|&x| 2.5 * x).collect();
        assert_exact(&got[0], &want);

        // Reused broadcast plus a pass-through output that names the broadcast
        // register: outputs `[0, 3]` with `3 = (conj(a) + a) * a`.
        let reused = FusedPlan {
            input_count: 1,
            outputs: vec![0, 3].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Conj,
                    inputs: vec![0].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![1, 0].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![2, 0].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        };
        let got = run_rank1(&reused, length, &[(&a, 0)], &[1, 1]);
        assert_exact(&got[0], &vec![2.5f64; length]);
        assert_exact(&got[1], &vec![12.5f64; length]);
    }

    // One f32 chain over a broadcast input, so the f32 chunk kernels see a
    // broadcast register too.
    let length = 1025;
    let a = [1.25f32];
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![2].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    let got = run_rank1(&plan, length, &[(&a, 0)], &[1]);
    assert_exact(&got[0], &vec![-1.25f32; length]);
}

/// Rank-2 inner broadcast: the broadcast input is constant along the blocked
/// axis 0 and contiguous along axis 1.
#[test]
fn chunked_replay_handles_rank2_inner_broadcast() {
    let (rows, cols) = (5usize, 300usize);
    let dims = [rows, cols];
    let out_strides = [1isize, rows as isize];
    let row: Vec<f64> = (0..cols).map(|c| c as f64 * 0.5 + 1.0).collect();
    let peer: Vec<f64> = (0..rows * cols).map(|i| (i % 17) as f64 - 8.0).collect();
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![2, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    let views = [
        StridedView::new(&row, &dims, &[0isize, 1], 0).unwrap(),
        StridedView::new(&peer, &dims, &[1isize, rows as isize], 0).unwrap(),
    ];
    let mut output = vec![0.0f64; rows * cols];
    {
        let dest = StridedViewMut::new(&mut output, &dims, &out_strides, 0).unwrap();
        fused_elementwise_into(&mut [dest], &views, &plan, Exec::serial()).unwrap();
    }
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(
                output[c * rows + r],
                row[c] + peer[c * rows + r],
                "({r},{c})"
            );
        }
    }
}

/// `[0, 1]` outer-varying broadcast: the input is constant along the outer axis
/// and contiguous along the blocked axis, so the block load is a normal
/// contiguous read with a stride-0 outer coordinate.
#[test]
fn chunked_replay_handles_outer_varying_broadcast() {
    let (rows, cols) = (5usize, 300usize);
    let dims = [rows, cols];
    let out_strides = [cols as isize, 1isize];
    let row: Vec<f64> = (0..cols).map(|c| c as f64 * 0.5 + 1.0).collect();
    let peer: Vec<f64> = (0..rows * cols).map(|i| (i % 19) as f64 - 9.0).collect();
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    let views = [
        StridedView::new(&row, &dims, &[0isize, 1], 0).unwrap(),
        StridedView::new(&peer, &dims, &[cols as isize, 1], 0).unwrap(),
    ];
    let mut output = vec![0.0f64; rows * cols];
    {
        let dest = StridedViewMut::new(&mut output, &dims, &out_strides, 0).unwrap();
        fused_elementwise_into(&mut [dest], &views, &plan, Exec::serial()).unwrap();
    }
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(
                output[r * cols + c],
                row[c] * peer[r * cols + c],
                "({r},{c})"
            );
        }
    }
}

/// Rank-3 inner broadcast, so the fill happens once per outer `(j, k)` block.
#[test]
fn chunked_replay_handles_rank3_inner_broadcast() {
    let dims = [4usize, 3, 200];
    let (a, b, c) = (dims[0], dims[1], dims[2]);
    let out_strides = [1isize, a as isize, (a * b) as isize];
    let plane: Vec<f64> = (0..b * c).map(|i| (i % 23) as f64 * 0.25 - 2.0).collect();
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![2].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    let views = [StridedView::new(&plane, &dims, &[0isize, 1, b as isize], 0).unwrap()];
    let mut output = vec![0.0f64; a * b * c];
    {
        let dest = StridedViewMut::new(&mut output, &dims, &out_strides, 0).unwrap();
        fused_elementwise_into(&mut [dest], &views, &plan, Exec::serial()).unwrap();
    }
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                assert_eq!(
                    output[i + j * a + k * a * b],
                    -plane[j + k * b],
                    "({i},{j},{k})"
                );
            }
        }
    }
}

/// A rank-zero DAG with several outputs: the block callback has no strides, so
/// the interpreter must treat every operand as stride 0 instead of indexing.
#[test]
fn chunked_replay_handles_rank_zero_multi_output() {
    let dims: [usize; 0] = [];
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![0, 1].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Negate,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };
    let a = [3.5f64];
    let views = [StridedView::new(&a, &dims, &[], 0).unwrap()];
    let mut pass_through = [0.0f64; 1];
    let mut negated = [0.0f64; 1];
    {
        let mut dests = [
            StridedViewMut::new(&mut pass_through, &dims, &[], 0).unwrap(),
            StridedViewMut::new(&mut negated, &dims, &[], 0).unwrap(),
        ];
        fused_elementwise_into(&mut dests, &views, &plan, Exec::serial()).unwrap();
    }
    assert_eq!(pass_through, [3.5]);
    assert_eq!(negated, [-3.5]);
}

/// Multiple destinations with differing outer strides: all inner strides stay 1
/// (so the chunk machine stores them) while the offsets differ per destination.
#[test]
fn chunked_replay_stores_multiple_outputs_with_different_outer_layouts() {
    let (rows, cols) = (4usize, 130usize);
    let dims = [rows, cols];
    let row: Vec<f64> = (0..cols).map(|c| c as f64 * 0.5 + 0.25).collect();
    let input = StridedView::new(&row, &dims, &[0isize, 1], 0).unwrap();
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![0, 1, 2].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![1, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    let gap = rows + 2;
    let out0_strides = [1isize, rows as isize];
    let out1_strides = [1isize, gap as isize];
    let out2_strides = [1isize, rows as isize];
    let mut out0 = vec![0.0f64; rows * cols];
    let mut out1 = vec![0.0f64; gap * cols];
    let mut out2 = vec![0.0f64; rows * cols];
    {
        let mut dests = [
            StridedViewMut::new(&mut out0, &dims, &out0_strides, 0).unwrap(),
            StridedViewMut::new(&mut out1, &dims, &out1_strides, 0).unwrap(),
            StridedViewMut::new(&mut out2, &dims, &out2_strides, 0).unwrap(),
        ];
        fused_elementwise_into(&mut dests, &[input], &plan, Exec::serial()).unwrap();
    }
    for c in 0..cols {
        for r in 0..rows {
            assert_eq!(out0[c * rows + r], row[c]);
            assert_eq!(out1[c * gap + r], -row[c]);
            assert_eq!(out2[c * rows + r], row[c] * row[c]);
        }
    }
}

/// Differing inner strides across destinations: the chunk machine requires every
/// output stride to be 1, so this must fall back to the interpreter and still
/// agree with an unfused reference.
#[test]
fn chunked_replay_falls_back_for_differing_output_inner_strides() {
    let (rows, cols) = (4usize, 130usize);
    let dims = [rows, cols];
    let row: Vec<f64> = (0..cols).map(|c| c as f64 * 0.5 + 0.25).collect();
    let input = StridedView::new(&row, &dims, &[0isize, 1], 0).unwrap();
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![0, 1].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    // out0 is column-major (inner axis 0), out1 is row-major (inner axis 1).
    let col_major = [1isize, rows as isize];
    let row_major = [cols as isize, 1isize];
    let mut out0 = vec![0.0f64; rows * cols];
    let mut out1 = vec![0.0f64; rows * cols];
    {
        let mut dests = [
            StridedViewMut::new(&mut out0, &dims, &col_major, 0).unwrap(),
            StridedViewMut::new(&mut out1, &dims, &row_major, 0).unwrap(),
        ];
        fused_elementwise_into(&mut dests, &[input], &plan, Exec::serial()).unwrap();
    }
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(out0[c * rows + r], row[c]);
            assert_eq!(out1[r * cols + c], -row[c]);
        }
    }
}

/// Parallel replay of a broadcast, multi-output plan in an explicit two-worker
/// pool, over a length that is above the threading threshold and not divisible
/// by the worker count.
#[cfg(feature = "parallel")]
#[test]
fn chunked_replay_parallel_pool_matches_reference() {
    use strided_basic::execution::MINTHREADLENGTH;
    use strided_basic::{
        ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, KernelDType,
    };
    use strided_fused::ErasedFusedPlan;

    use strided_basic::Pool;

    let pool = Pool::builder().threads(2).build().unwrap();

    let len = MINTHREADLENGTH + 65;
    let b: Vec<f64> = (0..len).map(|i| (i % 29) as f64 - 14.0).collect();
    let a = [2.5f64];
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![0, 3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    let erased = ErasedFusedPlan::compile(KernelDType::F64, plan).unwrap();
    let dims = [len];
    let unit = [1isize];
    let broadcast = [0isize];
    let mut pass_through = vec![MaybeUninit::<f64>::uninit(); len];
    let mut product = vec![MaybeUninit::<f64>::uninit(); len];

    let exec = Exec::new(&pool, 2).unwrap();
    assert_eq!(
        exec.budget().get(),
        2,
        "verify the token really has two slots"
    );
    let root = exec.ledger_root().unwrap();
    {
        let a_ref = ErasedRawStridedRef::from_slice(&a, &dims, &broadcast, 0).unwrap();
        let b_ref = ErasedRawStridedRef::from_slice(&b, &dims, &unit, 0).unwrap();
        let pointers = [
            ErasedRawStridedPtr::from_ref(&a_ref),
            ErasedRawStridedPtr::from_ref(&b_ref),
        ];
        let mut dests = [
            ErasedRawStridedUninitMut::from_uninit_slice(&mut pass_through, &dims, &unit, 0)
                .unwrap(),
            ErasedRawStridedUninitMut::from_uninit_slice(&mut product, &dims, &unit, 0).unwrap(),
        ];
        erased
            .execute_outputs_uninit(&mut dests, &pointers, exec)
            .unwrap();
    }
    assert!(threadteam::ledger::max_live(root) <= 2);

    for index in 0..len {
        // SAFETY: `execute_outputs_uninit` writes every reachable element.
        let pass = unsafe { pass_through[index].assume_init() };
        // SAFETY: as above.
        let value = unsafe { product[index].assume_init() };
        assert_eq!(pass, a[0], "pass-through element {index}");
        assert_eq!(value, a[0] * b[index], "product element {index}");
    }
}
