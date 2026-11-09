use super::*;
use crate::fused::{FusedInst, FusedOp};

#[test]
fn register_file_is_seeded_before_the_live_slice_is_formed() {
    // The per-thread buffer starts uninitialized; the seed must be written
    // through `MaybeUninit` before `&mut [T]` is formed. Miri flags reading
    // or referencing the uninitialized buffer otherwise.
    let plan = FusedPlan::new(1, vec![2], vec![FusedInst::new(FusedOp::Negate, vec![0])]);
    with_register_file::<f64, _>(&plan, Some(7.0), |regs| {
        assert_eq!(regs.len(), register_file_len(&plan));
        assert!(regs.iter().all(|&value| value == 7.0));
    });
    // A second call reuses the grown buffer and re-seeds it.
    with_register_file::<f64, _>(&plan, Some(-3.5), |regs| {
        assert!(regs.iter().all(|&value| value == -3.5));
    });
}

/// A plan that both outputs input 0 and reads it in exactly one instruction.
/// `replay_chunked` requires that `direct` never marks an input that
/// `plan.outputs` names (a direct input's register is neither loaded nor
/// computed), so this input must not be direct.
fn pass_through_and_negate() -> FusedPlan {
    FusedPlan::new(
        1,
        vec![0, 1],
        vec![FusedInst::new(FusedOp::Negate, vec![0])],
    )
}

#[test]
fn direct_inputs_never_mark_an_output_register() {
    let plan = pass_through_and_negate();
    assert_eq!(single_use_inputs(&plan).as_slice(), &[false]);
    let strides: [&[isize]; 3] = [&[1], &[1], &[1]];
    assert_eq!(
        crate::fused::direct_inputs(&plan, Some(1.0f64), &strides, 2).as_slice(),
        &[false]
    );
    // Without the pass-through output the same single-use input is direct.
    let reader_only = FusedPlan::new(1, vec![1], vec![FusedInst::new(FusedOp::Negate, vec![0])]);
    assert_eq!(single_use_inputs(&reader_only).as_slice(), &[true]);
}

#[test]
fn chunked_replay_stores_an_input_that_is_also_an_output() {
    use crate::{StridedView, StridedViewMut};
    // Distinct values: if the pass-through input were treated as direct, its
    // register would keep the seed (the first input element) and the first
    // output would read `[1.0, 1.0, 1.0]`.
    let input = [1.0f64, 2.0, 3.0];
    let dims = [3usize];
    let unit = [1isize];
    let mut pass = [0.0f64; 3];
    let mut negated = [0.0f64; 3];
    {
        let view = StridedView::<f64>::new(&input, &dims, &unit, 0).unwrap();
        let mut dests = [
            StridedViewMut::new(&mut pass, &dims, &unit, 0).unwrap(),
            StridedViewMut::new(&mut negated, &dims, &unit, 0).unwrap(),
        ];
        crate::fused::fused_elementwise_into(
            &mut dests,
            &[view],
            &pass_through_and_negate(),
            strided_basic::Exec::serial(),
        )
        .unwrap();
    }
    assert_eq!(pass, input);
    assert_eq!(negated, [-1.0, -2.0, -3.0]);
}
