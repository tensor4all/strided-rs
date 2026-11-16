//! The allocation gates, alone in their own test binary.
//!
//! The counting allocator is process-wide and `cargo test` runs the tests of one
//! target in parallel threads, so these live in a file of their own: a
//! concurrent gate would be charged for its neighbours' allocations and the
//! measurement would mean nothing.

use mace_ab::cg;

/// If the counting allocator is not installed, every counter reads 0 for
/// anything and the gates below would pass vacuously. Prove it sees 1 MiB first.
fn assert_counting_allocator_live() {
    mace_ab::measure::reset_alloc();
    let probe = vec![0u8; 1 << 20];
    std::hint::black_box(&probe);
    let seen = mace_ab::measure::alloc_snapshot();
    assert!(
        seen.bytes >= 1 << 20 && seen.calls >= 1,
        "the counting global allocator is not installed, so an allocation gate \
         cannot see a 1 MiB allocation and means nothing: {seen:?}"
    );
}

#[test]
fn the_executor_allocates_nothing_per_call() {
    let topology = mace_ab::topo::diamond_si([2, 2, 1], 6.0).expect("topology");
    let cg = cg::build::<f64>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 9);
    let pool = cg::pool(4).expect("pool");
    assert_counting_allocator_live();

    // Serial forward. Output and `Exec` are created before the counter resets.
    let mut out = vec![0.0f64; cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut out, exec).expect("warm");
    let exec = mace_ab::exec_for(1, None);
    mace_ab::measure::reset_alloc();
    cg.strided_forward(&buffers, &mut out, exec).expect("timed");
    let serial = mace_ab::measure::alloc_snapshot();
    assert_eq!(
        (serial.bytes, serial.calls),
        (0, 0),
        "the serial executor must not allocate, got {serial:?}"
    );

    // Pooled `bar_w`. Two 4T cells record one 1 520-byte allocation in one of
    // their 30 timed calls; this shows that the first pooled execution of a plan
    // may pay it and every later one is free. The counter is process-wide, so it
    // cannot say which component paid it — only that it is not per call.
    let mut grad_w = vec![0.0f64; cg.num_edges * 10 * cg.feature_dim()];
    let mut again = vec![0.0f64; grad_w.len()];
    let exec = mace_ab::exec_for(4, Some(&pool));
    cg.strided_vjp_w(&buffers, &mut grad_w, exec).expect("warm");
    let exec = mace_ab::exec_for(4, Some(&pool));
    mace_ab::measure::reset_alloc();
    cg.strided_vjp_w(&buffers, &mut again, exec)
        .expect("second");
    let pooled = mace_ab::measure::alloc_snapshot();
    assert_eq!(
        (pooled.bytes, pooled.calls),
        (0, 0),
        "the first pooled execution of a plan may allocate, later ones may not, got {pooled:?}"
    );
}
