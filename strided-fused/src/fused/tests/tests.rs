use super::*;
use crate::StridedArray;

/// A pool of `threads` workers for unit tests.
fn test_pool(threads: usize) -> strided_exec::Pool {
    strided_exec::Pool::builder()
        .threads(threads)
        .build()
        .unwrap()
}

#[cfg(feature = "parallel")]
use std::sync::Mutex;

#[cfg(feature = "parallel")]
static UNINIT_WORKER_IDS: Mutex<Vec<std::thread::ThreadId>> = Mutex::new(Vec::new());

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct ThreadTracked(u64);

#[cfg(feature = "parallel")]
impl ThreadTracked {
    fn observed(value: u64) -> Self {
        let id = std::thread::current().id();
        let mut ids = UNINIT_WORKER_IDS.lock().unwrap();
        if !ids.contains(&id) {
            ids.push(id);
        }
        drop(ids);
        for _ in 0..32 {
            std::hint::spin_loop();
        }
        Self(value)
    }
}

#[cfg(feature = "parallel")]
impl FusedScalar for ThreadTracked {
    fn fused_add(self, rhs: Self) -> Self {
        Self::observed(self.0 + rhs.0)
    }

    fn fused_multiply(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }

    fn fused_negate(self) -> Self {
        self
    }

    fn fused_conj(self) -> Self {
        self
    }

    fn fused_divide(self, _rhs: Self) -> Self {
        self
    }

    fn fused_abs(self) -> Self {
        self
    }

    fn fused_maximum(self, _rhs: Self) -> Self {
        self
    }

    fn fused_minimum(self, _rhs: Self) -> Self {
        self
    }

    fn fused_clamp(self, _min: Self, _max: Self) -> Self {
        self
    }

    fn fused_exp(self) -> Self {
        self
    }

    fn fused_log(self) -> Self {
        self
    }

    fn fused_sin(self) -> Self {
        self
    }

    fn fused_cos(self) -> Self {
        self
    }

    fn fused_tanh(self) -> Self {
        self
    }

    fn fused_sqrt(self) -> Self {
        self
    }

    fn fused_rsqrt(self) -> Self {
        self
    }

    fn fused_pow(self, _rhs: Self) -> Self {
        self
    }

    fn fused_expm1(self) -> Self {
        self
    }

    fn fused_log1p(self) -> Self {
        self
    }
}

fn input(values: &[f64]) -> StridedArray<f64> {
    StridedArray::from_parts(values.to_vec(), &[values.len()], &[1], 0).unwrap()
}

fn run_static(plan: &FusedPlan, arrays: &[StridedArray<f64>]) -> (bool, Vec<f64>) {
    let inputs: Vec<_> = arrays.iter().map(|array| array.view()).collect();
    let mut out = StridedArray::<f64>::col_major(arrays[0].dims());
    let used_static = {
        let mut dests = [out.view_mut()];
        try_static_specialization(&mut dests, &inputs, plan).unwrap()
    };
    (used_static, out.iter().copied().collect())
}

fn run_interpreter(plan: &FusedPlan, arrays: &[StridedArray<f64>]) -> Vec<f64> {
    let inputs: Vec<_> = arrays.iter().map(|array| array.view()).collect();
    let mut out = StridedArray::<f64>::col_major(arrays[0].dims());
    {
        let mut dests = [out.view_mut()];
        interpret_fused_elementwise_into(&mut dests, &inputs, plan, Exec::serial()).unwrap();
    }
    out.iter().copied().collect()
}

fn assert_static_matches_interpreter(plan: FusedPlan, arrays: &[StridedArray<f64>]) {
    assert_static_matches_interpreter_within(plan, arrays, 0.0)
}

/// `ulp_budget` is relative to the larger magnitude, in units of
/// `f64::EPSILON`; `0.0` demands bit equality. The interpreter's analytic
/// instructions now run through the vector kernels, documented at 1.0 ULP,
/// while the static specialization still calls the scalar functions; every
/// other instruction kind is exact in both.
fn assert_static_matches_interpreter_within(
    plan: FusedPlan,
    arrays: &[StridedArray<f64>],
    ulp_budget: f64,
) {
    let (used_static, static_values) = run_static(&plan, arrays);
    let interpreter_values = run_interpreter(&plan, arrays);

    assert!(used_static, "plan should use static specialization");
    assert_eq!(static_values.len(), interpreter_values.len());
    for (actual, expected) in static_values.iter().zip(interpreter_values.iter()) {
        if actual == expected || (actual.is_nan() && expected.is_nan()) {
            continue;
        }
        let scale = actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE);
        let measured = (actual - expected).abs() / scale / f64::EPSILON;
        assert!(
            measured <= ulp_budget,
            "{actual} vs {expected} is {measured} ULP"
        );
    }
}

/// Replay one plan per static family through tokens from `mint`, reporting each
/// destination.
fn replay_all_static_families<'p>(mint: impl Fn() -> Exec<'p>, mut collect: impl FnMut(&[f64])) {
    let mut run = |plan: FusedPlan, arrays: &[StridedArray<f64>]| {
        let inputs: Vec<_> = arrays.iter().map(|array| array.view()).collect();
        let mut output = vec![MaybeUninit::uninit(); arrays[0].len()];
        {
            let mut dest = StridedViewMut::new(&mut output, arrays[0].dims(), &[1], 0).unwrap();
            let validated =
                validate_destination_layout_without_alloc(dest.dims(), dest.strides()).unwrap();
            fused_elementwise_into_uninit(&mut dest, &inputs, &plan, validated, mint()).unwrap();
        }
        // SAFETY: every plan here has a single output and fully overwrites the destination.
        collect(
            &output
                .iter()
                .map(|value| unsafe { value.assume_init() })
                .collect::<Vec<_>>(),
        );
    };

    let a = input(&[4.0, 9.0, 16.0]);
    let b = input(&[2.0, 3.0, 4.0]);
    let c = input(&[1.0, 1.5, 2.0]);
    let d = input(&[4.0, 4.0, 4.0]);

    run(
        single_op(1, FusedOp::Sqrt, vec![0]),
        std::slice::from_ref(&a),
    );
    run(
        single_op(2, FusedOp::Add, vec![0, 1]),
        &[a.clone(), b.clone()],
    );
    run(
        single_op(3, FusedOp::Clamp, vec![0, 1, 2]),
        &[a.clone(), c.clone(), d.clone()],
    );
    run(
        FusedPlan {
            input_count: 2,
            outputs: vec![3].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![0, 1].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![2, 0].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
        &[a.clone(), b.clone()],
    );
    run(
        FusedPlan {
            input_count: 2,
            outputs: vec![3].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![0, 1].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![0, 2].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
        &[a.clone(), b.clone()],
    );
    run(
        FusedPlan {
            input_count: 3,
            outputs: vec![5].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![0, 1].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![3, 2].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Exp,
                    inputs: vec![4].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
        &[a.clone(), b.clone(), c.clone()],
    );
    run(
        FusedPlan {
            input_count: 4,
            outputs: vec![8].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Divide,
                    inputs: vec![0, 1].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Maximum,
                    inputs: vec![4, 2].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Minimum,
                    inputs: vec![5, 3].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Sqrt,
                    inputs: vec![6].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Rsqrt,
                    inputs: vec![7].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
        &[a, b, c, d],
    );
}

#[test]
fn uninitialized_nonserial_replay_selects_every_static_family() {
    let pool = test_pool(2);

    reset_uninitialized_static_family_hits();
    replay_all_static_families(|| Exec::new(&pool, 2).unwrap(), |_| {});
    assert_eq!(uninitialized_static_family_hits(), [1; 7]);
}

#[test]
fn uninitialized_serial_replay_selects_every_static_family() {
    reset_uninitialized_static_family_hits();
    replay_all_static_families(Exec::serial, |_| {});
    assert_eq!(uninitialized_static_family_hits(), [1; 7]);
}

#[test]
fn uninitialized_replay_agrees_between_serial_and_bounded_contexts() {
    let pool = test_pool(2);

    let mut serial = Vec::new();
    replay_all_static_families(Exec::serial, |values| serial.push(values.to_vec()));
    let mut bounded = Vec::new();
    replay_all_static_families(
        || Exec::new(&pool, 2).unwrap(),
        |values| bounded.push(values.to_vec()),
    );
    assert_eq!(serial, bounded);
}

#[cfg(feature = "parallel")]
#[test]
fn uninitialized_fused_replay_respects_serial_and_bounded_contexts_above_threshold() {
    let len = strided_basic::execution::MINTHREADLENGTH + 65;
    let lhs = vec![ThreadTracked(1); len];
    let rhs = vec![ThreadTracked(2); len];
    let dims = [len];
    let unit = [1isize];
    let lhs = StridedView::new(&lhs, &dims, &unit, 0).unwrap();
    let rhs = StridedView::new(&rhs, &dims, &unit, 0).unwrap();
    let inputs = [lhs, rhs];
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };
    let pool = test_pool(4);

    let caller = std::thread::current().id();
    let mut output = vec![MaybeUninit::uninit(); len];
    let mut dest = StridedViewMut::new(&mut output, &dims, &unit, 0).unwrap();
    let validated = validate_destination_layout_without_alloc(dest.dims(), dest.strides()).unwrap();
    UNINIT_WORKER_IDS.lock().unwrap().clear();
    fused_elementwise_into_uninit(&mut dest, &inputs, &plan, validated, Exec::serial()).unwrap();
    assert_eq!(*UNINIT_WORKER_IDS.lock().unwrap(), vec![caller]);

    let mut output = vec![MaybeUninit::uninit(); len];
    let mut dest = StridedViewMut::new(&mut output, &dims, &unit, 0).unwrap();
    let validated = validate_destination_layout_without_alloc(dest.dims(), dest.strides()).unwrap();
    UNINIT_WORKER_IDS.lock().unwrap().clear();
    let exec = Exec::new(&pool, 2).unwrap();
    fused_elementwise_into_uninit(&mut dest, &inputs, &plan, validated, exec).unwrap();
    let workers = UNINIT_WORKER_IDS.lock().unwrap();
    assert!(
        workers.len() > 1,
        "bounded replay must cross the parallel threshold"
    );
    assert!(workers.len() <= 2, "bounded replay exceeded a budget of 2");
}

// Single-instruction plan whose sole output is the instruction result. Such
// plans always hit `try_static_specialization`, and both the static and the
// interpreter path dispatch through the scalar `FusedScalar` methods, so
// iterating every op exercises each scalar implementation.
fn single_op(input_count: usize, op: FusedOp, inputs: Vec<usize>) -> FusedPlan {
    FusedPlan {
        input_count,
        outputs: vec![input_count].into_iter().collect(),
        ops: vec![FusedInst::new(op, inputs)].into_iter().collect(),
    }
}

#[test]
fn specializes_unary_exp() {
    let a = input(&[1.0, 2.0, 3.0]);
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![1].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Exp,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    assert_static_matches_interpreter(plan, &[a]);
}

#[test]
fn specializes_binary_add() {
    let a = input(&[1.0, 2.0, 3.0]);
    let b = input(&[10.0, 20.0, 30.0]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    assert_static_matches_interpreter(plan, &[a, b]);
}

#[test]
fn specializes_ternary_clamp() {
    let x = input(&[1.0, 2.0, 3.0]);
    let lo = input(&[1.5, 1.5, 1.5]);
    let hi = input(&[2.5, 2.5, 2.5]);
    let plan = FusedPlan {
        input_count: 3,
        outputs: vec![3].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Clamp,
            inputs: vec![0, 1, 2].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    assert_static_matches_interpreter(plan, &[x, lo, hi]);
}

#[test]
fn specializes_add_then_multiply_reusing_input() {
    let a = input(&[1.0, 2.0, 3.0]);
    let b = input(&[10.0, 20.0, 30.0]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };

    assert_static_matches_interpreter(plan, &[a, b]);
}

#[test]
fn specializes_exp_of_multiply_add_chain() {
    let a = input(&[1.0, 2.0, 3.0]);
    let b = input(&[0.5, 1.5, 2.5]);
    let c = input(&[2.0, 2.0, 2.0]);
    let plan = FusedPlan {
        input_count: 3,
        outputs: vec![5].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![3, 2].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![4].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };

    // The one analytic instruction is worth one ULP of its own output
    // magnitude, which the scalar comparison rejects at `exp(13.5)`.
    assert_static_matches_interpreter_within(plan, &[a, b, c], 2.0);
}

#[test]
fn specializes_divide_clamp_sqrt_rsqrt_chain() {
    let a = input(&[4.0, 9.0, 16.0]);
    let b = input(&[2.0, 3.0, 4.0]);
    let lo = input(&[1.5, 1.5, 1.5]);
    let hi = input(&[8.0, 8.0, 8.0]);
    let plan = FusedPlan {
        input_count: 4,
        outputs: vec![8].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Divide,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Maximum,
                inputs: vec![4, 2].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Minimum,
                inputs: vec![5, 3].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Sqrt,
                inputs: vec![6].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Rsqrt,
                inputs: vec![7].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };

    assert_static_matches_interpreter(plan, &[a, b, lo, hi]);
}

// Real negate/conj/abs were the only real scalar ops not reached by the
// chains above; cover them so the real `FusedScalar` impl is fully exercised.
#[test]
fn specializes_real_negate_conj_abs() {
    for op in [FusedOp::Negate, FusedOp::Conj, FusedOp::Abs] {
        let x = input(&[-1.5, 2.0, -3.5]);
        assert_static_matches_interpreter(single_op(1, op, vec![0]), &[x]);
    }
}

// The complex `FusedScalar` impl had no coverage at all (every existing test
// used f64). Run every op over Complex64 so both the static and interpreter
// paths dispatch through the complex scalar methods.
#[test]
fn specializes_every_op_over_complex() {
    use num_complex::Complex64;

    let c = |re: f64, im: f64| Complex64::new(re, im);
    let cinput = |values: &[Complex64]| {
        StridedArray::from_parts(values.to_vec(), &[values.len()], &[1], 0).unwrap()
    };
    let assert_complex_match = |plan: FusedPlan, arrays: &[StridedArray<Complex64>]| {
        let inputs: Vec<_> = arrays.iter().map(|array| array.view()).collect();
        let mut static_out = StridedArray::<Complex64>::col_major(arrays[0].dims());
        let used_static = {
            let mut dests = [static_out.view_mut()];
            try_static_specialization(&mut dests, &inputs, &plan).unwrap()
        };
        let mut interp_out = StridedArray::<Complex64>::col_major(arrays[0].dims());
        {
            let mut dests = [interp_out.view_mut()];
            interpret_fused_elementwise_into(&mut dests, &inputs, &plan, Exec::serial()).unwrap();
        }
        assert!(used_static, "single-op plan should specialize");
        for (actual, expected) in static_out.iter().zip(interp_out.iter()) {
            assert!((actual - expected).norm() < 1e-9, "{actual} vs {expected}");
        }
    };

    // Positive-real-part, nonzero operands keep div/log/sqrt/pow well defined.
    let a = cinput(&[c(1.5, 0.5), c(2.0, -1.0), c(0.7, 0.3)]);
    let b = cinput(&[c(1.1, 0.2), c(0.9, 0.4), c(1.3, -0.6)]);
    let d = cinput(&[c(2.0, 0.0), c(2.0, 0.0), c(2.0, 0.0)]);

    for op in [
        FusedOp::Negate,
        FusedOp::Conj,
        FusedOp::Abs,
        FusedOp::Exp,
        FusedOp::Log,
        FusedOp::Sin,
        FusedOp::Cos,
        FusedOp::Tanh,
        FusedOp::Sqrt,
        FusedOp::Rsqrt,
        FusedOp::Expm1,
        FusedOp::Log1p,
    ] {
        assert_complex_match(single_op(1, op, vec![0]), std::slice::from_ref(&a));
    }
    for op in [
        FusedOp::Add,
        FusedOp::Multiply,
        FusedOp::Divide,
        FusedOp::Maximum,
        FusedOp::Minimum,
        FusedOp::Pow,
    ] {
        assert_complex_match(single_op(2, op, vec![0, 1]), &[a.clone(), b.clone()]);
    }
    assert_complex_match(
        single_op(3, FusedOp::Clamp, vec![0, 1, 2]),
        &[a.clone(), b.clone(), d.clone()],
    );
}

// Error branches in plan/layout validation that the positive tests skip.
#[test]
fn validate_plan_rejects_out_of_range_output() {
    // output id refers to a value that no instruction produces.
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![5].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Exp,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };
    assert!(validate_plan(&plan, 1, 1).is_err());
}

#[test]
fn uninitialized_serial_replay_accepts_an_empty_destination() {
    let dims = [0usize, 3];
    let strides = [3isize, 1];
    let a = StridedArray::<f64>::from_parts(vec![], &dims, &strides, 0).unwrap();
    let b = StridedArray::<f64>::from_parts(vec![], &dims, &strides, 0).unwrap();
    let inputs = [a.view(), b.view()];
    let mut output: Vec<MaybeUninit<f64>> = Vec::new();
    let mut dest = StridedViewMut::new(&mut output, &dims, &strides, 0).unwrap();
    let validated = validate_destination_layout_without_alloc(dest.dims(), dest.strides()).unwrap();
    let plan = single_op(2, FusedOp::Add, vec![0, 1]);
    fused_elementwise_into_uninit(&mut dest, &inputs, &plan, validated, Exec::serial()).unwrap();
}

#[test]
fn validate_plan_rejects_zero_outputs() {
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![].into_iter().collect(),
        ops: vec![].into_iter().collect(),
    };
    assert!(validate_plan(&plan, 1, 0).is_err());
}

#[test]
fn is_injective_layout_rejects_rank_and_broadcast_mismatch() {
    assert!(!is_injective_layout(&[2, 3], &[1]));
    assert!(!is_injective_layout(&[2, 2], &[0, 1]));
    assert!(is_injective_layout(&[1], &[0]));
}

#[test]
fn is_injective_layout_rejects_unrepresentable_offset_spans() {
    let positive = isize::MAX / 2 + 1;
    let negative = isize::MIN / 2 - 1;
    assert!(!is_injective_layout(&[2, 2], &[positive, isize::MAX]));
    assert!(!is_injective_layout(&[2, 2], &[negative, isize::MIN]));
}

/// Observation state for the broadcast partition regression. It is separate
/// from `UNINIT_WORKER_IDS` so the two parallel tests do not race.
#[cfg(feature = "parallel")]
static BROADCAST_WORKER_IDS: Mutex<Vec<std::thread::ThreadId>> = Mutex::new(Vec::new());

#[cfg(feature = "parallel")]
static BROADCAST_ADDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct BroadcastTracked(u64);

#[cfg(feature = "parallel")]
impl FusedScalar for BroadcastTracked {
    fn fused_add(self, rhs: Self) -> Self {
        BROADCAST_ADDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let id = std::thread::current().id();
        {
            let mut ids = BROADCAST_WORKER_IDS.lock().unwrap();
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        for _ in 0..64 {
            std::hint::spin_loop();
        }
        Self(self.0.wrapping_add(rhs.0))
    }

    fn fused_multiply(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }

    fn fused_negate(self) -> Self {
        self
    }

    fn fused_conj(self) -> Self {
        self
    }

    fn fused_divide(self, _rhs: Self) -> Self {
        self
    }

    fn fused_abs(self) -> Self {
        self
    }

    fn fused_maximum(self, _rhs: Self) -> Self {
        self
    }

    fn fused_minimum(self, _rhs: Self) -> Self {
        self
    }

    fn fused_clamp(self, _min: Self, _max: Self) -> Self {
        self
    }

    fn fused_exp(self) -> Self {
        self
    }

    fn fused_log(self) -> Self {
        self
    }

    fn fused_sin(self) -> Self {
        self
    }

    fn fused_cos(self) -> Self {
        self
    }

    fn fused_tanh(self) -> Self {
        self
    }

    fn fused_sqrt(self) -> Self {
        self
    }

    fn fused_rsqrt(self) -> Self {
        self
    }

    fn fused_pow(self, _rhs: Self) -> Self {
        self
    }

    fn fused_expm1(self) -> Self {
        self
    }

    fn fused_log1p(self) -> Self {
        self
    }
}

/// Regression (R-performance-09): the fused replay of an inner-axis broadcast
/// must split across workers, not run in one partition behind the whole-tensor
/// cache block. `[1023, 67]` is odd, above the threading threshold, and below
/// the splitter's 1024 guard, so it exercises the shared block normalization.
#[cfg(feature = "parallel")]
#[test]
fn broadcast_fused_replay_splits_and_covers_each_destination_once() {
    use crate::{StridedView, StridedViewMut};
    use core::mem::MaybeUninit;
    use std::sync::atomic::Ordering;

    let rows = 1023usize;
    let cols = 67usize;
    let len = rows * cols;
    let dims = [rows, cols];
    let out_strides = [1isize, rows as isize];
    let broadcast_strides = [0isize, 1isize];
    let peer_strides = [1isize, rows as isize];

    let a: Vec<BroadcastTracked> = (0..cols).map(|c| BroadcastTracked(c as u64)).collect();
    let b: Vec<BroadcastTracked> = (0..len).map(|i| BroadcastTracked(i as u64)).collect();
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

    let pool = test_pool(4);
    {
        let a_view = StridedView::new(&a, &dims, &broadcast_strides, 0).unwrap();
        let b_view = StridedView::new(&b, &dims, &peer_strides, 0).unwrap();
        let inputs = [a_view, b_view];

        BROADCAST_WORKER_IDS.lock().unwrap().clear();
        BROADCAST_ADDS.store(0, Ordering::SeqCst);
        let mut output = vec![BroadcastTracked(0); len];
        {
            let dest = StridedViewMut::new(&mut output, &dims, &out_strides, 0).unwrap();
            fused_elementwise_into(&mut [dest], &inputs, &plan, Exec::new(&pool, 4).unwrap())
                .unwrap();
        }
        assert_eq!(
            BROADCAST_ADDS.load(Ordering::SeqCst),
            len,
            "initialized replay must evaluate Add exactly once per element"
        );
        let workers = BROADCAST_WORKER_IDS.lock().unwrap().len();
        assert!(
            workers > 1,
            "initialized broadcast replay must split across workers, saw {workers}"
        );

        BROADCAST_WORKER_IDS.lock().unwrap().clear();
        BROADCAST_ADDS.store(0, Ordering::SeqCst);
        let mut raw = vec![MaybeUninit::<BroadcastTracked>::uninit(); len];
        {
            let mut dest = StridedViewMut::new(&mut raw, &dims, &out_strides, 0).unwrap();
            let validated =
                validate_destination_layout_without_alloc(dest.dims(), dest.strides()).unwrap();
            let exec = Exec::new(&pool, 4).unwrap();
            fused_elementwise_into_uninit(&mut dest, &inputs, &plan, validated, exec).unwrap();
        }
        assert_eq!(
            BROADCAST_ADDS.load(Ordering::SeqCst),
            len,
            "uninitialized replay must evaluate Add exactly once per element"
        );
        let workers = BROADCAST_WORKER_IDS.lock().unwrap().len();
        assert!(
            workers > 1,
            "uninitialized broadcast replay must split across workers, saw {workers}"
        );
    }
}
