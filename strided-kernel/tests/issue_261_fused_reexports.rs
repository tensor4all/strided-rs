//! Regression test for issue #261: strided-kernel 0.4.1 dropped the fused API
//! that 0.4.0 exported from its crate root. These paths must keep resolving on
//! the 0.4 line so downstream crates pinned to `strided-kernel = "0.4"` build.

use strided_kernel::{
    fused_elementwise_into, ErasedFusedPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext,
    FusedInst, FusedOp, FusedPlan, FusedScalar, KernelDType, StridedView, StridedViewMut,
};

fn add_plan() -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1],
        }],
    }
}

fn assert_fused_scalar<T: FusedScalar>() {}

#[test]
fn fused_api_is_reexported_at_0_4_0_paths() {
    assert_fused_scalar::<f64>();

    let a = [1.0f64, 2.0, 3.0];
    let b = [10.0f64, 20.0, 30.0];
    let mut out = [0.0f64; 3];
    {
        let av = StridedView::new(&a, &[3], &[1], 0).unwrap();
        let bv = StridedView::new(&b, &[3], &[1], 0).unwrap();
        let mut dests = [StridedViewMut::new(&mut out, &[3], &[1], 0).unwrap()];
        fused_elementwise_into(&mut dests, &[av, bv], &add_plan()).unwrap();
    }
    assert_eq!(out, [11.0, 22.0, 33.0]);

    let mut erased_out = [0.0f64; 3];
    {
        let inputs = [
            ErasedRawStridedRef::from_slice(&a, &[3], &[1], 0).unwrap(),
            ErasedRawStridedRef::from_slice(&b, &[3], &[1], 0).unwrap(),
        ];
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut erased_out, &[3], &[1], 0).unwrap();
        ErasedFusedPlan::compile(KernelDType::F64, add_plan())
            .unwrap()
            .execute(&ExecContext::serial(), &mut dest, &inputs)
            .unwrap();
    }
    assert_eq!(erased_out, [11.0, 22.0, 33.0]);
}

#[test]
fn fused_reexports_are_the_strided_fused_types() {
    // The compatibility paths must be the same items as strided-fused, not
    // copies, so values flow freely between crates that import either path.
    let plan: strided_fused::FusedPlan = add_plan();
    let _: FusedPlan = plan;
    let op: strided_fused::FusedOp = FusedOp::Multiply;
    assert_eq!(op.label(), strided_fused::FusedOp::Multiply.label());
}
