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
