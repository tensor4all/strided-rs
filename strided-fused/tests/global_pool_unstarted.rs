//! Serial and pooled work never starts the process-global Rayon pool.
//!
//! This binary holds a single test: the check installs a one-thread global
//! pool, which succeeds only if nothing in the process started it before.

use strided_basic::{map_into, Exec, Pool, StridedArray};
use strided_fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan};

const LEN: usize = (1 << 17) + 65;

fn double_map(exec: Exec<'_>) {
    let source = StridedArray::<f64>::from_fn_col_major(&[LEN], |index| index[0] as f64);
    let mut destination = StridedArray::<f64>::col_major(&[LEN]);
    map_into(
        &mut destination.view_mut(),
        &source.view(),
        |v| v * 2.0,
        exec,
    )
    .unwrap();
    assert_eq!(destination.get(&[LEN - 1]), 2.0 * (LEN - 1) as f64);
}

fn fused_add(exec: Exec<'_>) {
    let lhs = StridedArray::<f64>::from_fn_col_major(&[LEN], |index| index[0] as f64);
    let rhs = StridedArray::<f64>::from_fn_col_major(&[LEN], |_| 1.0);
    let mut destination = StridedArray::<f64>::col_major(&[LEN]);
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
    fused_elementwise_into(
        &mut [destination.view_mut()],
        &[lhs.view(), rhs.view()],
        &plan,
        exec,
    )
    .unwrap();
    assert_eq!(destination.get(&[LEN - 1]), LEN as f64);
}

#[test]
fn serial_and_pooled_work_leave_the_global_pool_unstarted() {
    double_map(Exec::serial());
    fused_add(Exec::serial());

    let pool = Pool::builder().threads(4).build().unwrap();
    double_map(Exec::new(&pool, 2).unwrap());
    fused_add(Exec::new(&pool, 4).unwrap());
    drop(pool);

    assert!(strided_exec::ledger::global_pool_unstarted());
}
