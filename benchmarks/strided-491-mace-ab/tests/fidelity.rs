//! The fidelity gate: the descriptor reproduces the published layer-1 counters,
//! both sides compute the same numbers, the bits are thread-invariant, and the
//! strided plans allocate nothing per call.

use mace_ab::cg::{self, Buffers, Cg};
use mace_ab::topo::Topology;

/// Largest magnitude on either side, so the error bar can be scale relative.
fn scale(reference: &[f64]) -> f64 {
    reference
        .iter()
        .fold(0.0f64, |acc, &value| acc.max(value.abs()))
}

fn small() -> Topology {
    // 32 atoms: 2x2x1 conventional cells of diamond Si.
    mace_ab::topo::diamond_si([2, 2, 1], 6.0).expect("topology")
}

#[test]
fn descriptor_reproduces_published_layer1_counters() {
    let desc = cg::layer1_descriptor_for_test();
    assert_eq!(desc.feature_dim, 128);
    assert_eq!(desc.in1_dim, 512, "128x0e + 128x1o");
    assert_eq!(desc.in2_dim, 16, "1x0e+1x1o+1x2e+1x3o");
    assert_eq!(desc.out_dim, 5120, "10 blocks of 128 features");
    assert_eq!(desc.weight_numel, 1280, "10 blocks x 128 features");
    assert_eq!(desc.blocks.len(), 10);
    let lens: Vec<usize> = desc.blocks.iter().map(|block| block.cg.len()).collect();
    assert_eq!(lens, vec![1, 3, 3, 3, 11, 5, 11, 21, 7, 21]);
    assert_eq!(lens.iter().sum::<usize>(), 86);
    // The four blocks with a stored w3j tensor, whose published nonzero counts
    // are 11, 11, 21 and 21 in mace_mp0b_inspection.json.
    let w3j: Vec<usize> = desc
        .blocks
        .iter()
        .filter(|block| block.x_dim == 3 && block.out_dim > 1)
        .map(|block| block.cg.len())
        .filter(|len| [11, 21].contains(len))
        .collect();
    assert_eq!(w3j, vec![11, 11, 21, 21]);
    // Every CG index has to fit its block.
    for block in &desc.blocks {
        for &(i_m, j_m, k_m, _) in &block.cg {
            assert!(i_m < block.x_dim);
            assert!(k_m < block.out_dim);
            assert!(j_m < 7);
        }
    }
}

#[test]
fn both_sides_compute_the_same_forward() {
    let topology = small();
    let mut cg = cg::build::<f64>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 3);
    let mut strided = vec![0.0f64; cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut strided, exec)
        .expect("fwd");
    let (path, llmlip) = cg.llmlip_forward(&buffers).expect("llmlip fwd");
    assert_eq!(
        path, "layer1-fast",
        "llmlip must answer with its fast kernel"
    );
    let (max_abs, _max_rel) = mace_ab::compare_forward(&cg, &llmlip, &strided);
    // Different summation orders cancel differently, so the bar is on the
    // absolute error relative to the scale of the result, not per element.
    assert!(
        max_abs / scale(&llmlip) < 1e-14,
        "fwd max_abs {max_abs:e} on scale {:e}",
        scale(&llmlip)
    );
}

#[test]
fn both_sides_compute_the_same_vjp() {
    let topology = small();
    let mut cg = cg::build::<f64>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 5);
    let (path, (bar_x, bar_y, bar_w)) = cg.llmlip_vjp(&buffers).expect("llmlip vjp");
    assert_eq!(path, "layer1-fast");

    let mut grad_x = vec![0.0f64; cg.num_nodes * (128 + 3 * cg.feature_dim())];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_x(&buffers, &mut grad_x, exec)
        .expect("vjp-x");
    let (narrow, wide) = grad_x.split_at(cg.num_nodes * cg.feature_dim());
    let (abs_x, _rel_x) = mace_ab::compare_grad_x(&cg, &bar_x, narrow, wide);
    assert!(abs_x / scale(&bar_x) < 1e-14, "bar_x max_abs {abs_x:e}");

    let mut grad_w = vec![0.0f64; cg.num_edges * 10 * cg.feature_dim()];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_w(&buffers, &mut grad_w, exec)
        .expect("vjp-w");
    let (abs_w, _rel_w) = mace_ab::compare_same_order(&bar_w, &grad_w);
    assert!(abs_w / scale(&bar_w) < 1e-14, "bar_w max_abs {abs_w:e}");

    let mut grad_y = vec![0.0f64; cg.num_edges * 16];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_y(&buffers, &mut grad_y, exec)
        .expect("vjp-y");
    let (abs_y, _rel_y) = mace_ab::compare_same_order(&bar_y, &grad_y);
    assert!(abs_y / scale(&bar_y) < 1e-14, "bar_y max_abs {abs_y:e}");
}

#[test]
fn bits_are_thread_invariant() {
    let topology = small();
    let mut cg = cg::build::<f64>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 7);
    let pool = cg::pool(4).expect("pool");
    let mut serial = vec![0.0f64; cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut serial, exec)
        .expect("serial");
    let mut pooled = vec![0.0f64; serial.len()];
    let exec = mace_ab::exec_for(4, Some(&pool));
    cg.strided_forward(&buffers, &mut pooled, exec)
        .expect("pooled");
    assert_eq!(serial, pooled, "1T and 4T must agree bitwise");
}

#[test]
fn execute_allocates_nothing() {
    let topology = small();
    let mut cg = cg::build::<f64>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 9);
    let mut out = vec![0.0f64; cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
    // Warm, then measure.
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut out, exec).expect("warm");
    mace_ab::measure::reset_alloc();
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut out, exec).expect("timed");
    let allocated = mace_ab::measure::alloc_snapshot();
    assert_eq!(allocated.bytes, 0, "the executor must not allocate");
}

#[test]
fn buffers_have_the_layout_the_plans_were_compiled_against() {
    let topology = small();
    let cg = cg::build::<f64>(&topology).expect("plans");
    let buffers: Buffers<f64> = mace_ab::buffers(&cg, 1);
    assert_eq!(buffers.linear_up.len(), cg.num_nodes * 512);
    assert_eq!(buffers.edge_attrs.len(), cg.num_edges * 16);
    assert_eq!(buffers.tp_weights.len(), cg.num_edges * 1280);
    assert_eq!(buffers.bar_scatter.len(), cg.num_nodes * 5120);
    assert_eq!(buffers.bar_scatter_major.len(), cg.num_nodes * 5120);
}
