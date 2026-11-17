//! The fidelity gates: the descriptor reproduces the published layer-1 counters,
//! both sides compute the same numbers on **every node**, the bits are
//! thread-invariant in both dtypes, and the comparisons actually look at the
//! whole buffer. The allocation gates live in `tests/alloc.rs`, which runs in its
//! own process: the counters are process-wide, so they cannot be read while the
//! other gates are allocating beside them.

use mace_ab::cg::{self, Buffers, Cg, Scalar};
use mace_ab::topo::Topology;

fn small() -> Topology {
    // 32 atoms: 2x2x1 conventional cells of diamond Si.
    mace_ab::topo::diamond_si([2, 2, 1], 6.0).expect("topology")
}

/// Largest magnitude on the reference side, so an error bar can be scale
/// relative: elementwise relative error is meaningless where terms cancel.
fn scale<T: Scalar>(reference: &[T]) -> f64 {
    reference
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.to_f64().abs()))
}

fn identity<T: Scalar>() -> &'static str {
    if std::mem::size_of::<T>() == 4 {
        "f32"
    } else {
        "f64"
    }
}

/// The agreement bar per dtype, scale relative. f64 lands at the accumulator's
/// precision; f32 is 1e-5 because `bar_y` sums 11 008 terms per value and the
/// measured worst case on these graphs is 3.4e-6.
fn tolerance<T: Scalar>() -> f64 {
    if std::mem::size_of::<T>() == 4 {
        1e-5
    } else {
        1e-14
    }
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
    let w3j: Vec<usize> = desc
        .blocks
        .iter()
        .filter(|block| block.x_dim == 3 && block.out_dim > 1)
        .map(|block| block.cg.len())
        .filter(|len| [11, 21].contains(len))
        .collect();
    assert_eq!(w3j, vec![11, 11, 21, 21]);
    for block in &desc.blocks {
        for &(i_m, j_m, k_m, _) in &block.cg {
            assert!(i_m < block.x_dim);
            assert!(k_m < block.out_dim);
            assert!(j_m < 7);
        }
    }
}

fn check_forward<T: Scalar>() {
    let topology = small();
    let cg = cg::build::<T>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 3);
    let mut strided = vec![T::from_f64(0.0); cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut strided, exec)
        .expect("fwd");
    let (path, llmlip) = cg.llmlip_forward(&buffers).expect("llmlip fwd");
    assert_eq!(
        path, "layer1-fast",
        "llmlip must answer with its fast kernel"
    );
    let (max_abs, _) = mace_ab::compare_forward(&cg, &llmlip, &strided);
    assert!(
        max_abs / scale(&llmlip) < tolerance::<T>(),
        "{} fwd max_abs {max_abs:e} on scale {:e}",
        identity::<T>(),
        scale(&llmlip)
    );
}

fn check_vjp<T: Scalar>() {
    let topology = small();
    let cg = cg::build::<T>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 5);
    let (path, (bar_x, bar_y, bar_w)) = cg.llmlip_vjp(&buffers).expect("llmlip vjp");
    assert_eq!(path, "layer1-fast");
    let tolerance = tolerance::<T>();

    let mut grad_x = vec![T::from_f64(0.0); cg.num_nodes * (128 + 3 * cg.feature_dim())];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_x(&buffers, &mut grad_x, exec)
        .expect("vjp-x");
    let (narrow, wide) = grad_x.split_at(cg.num_nodes * cg.feature_dim());
    let (abs_x, _) = mace_ab::compare_grad_x(&cg, &bar_x, narrow, wide);
    assert!(
        abs_x / scale(&bar_x) < tolerance,
        "{} bar_x max_abs {abs_x:e} scale {:e} rel {:e}",
        identity::<T>(),
        scale(&bar_x),
        abs_x / scale(&bar_x)
    );

    let mut grad_w = vec![T::from_f64(0.0); cg.num_edges * 10 * cg.feature_dim()];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_w(&buffers, &mut grad_w, exec)
        .expect("vjp-w");
    let (abs_w, _) = mace_ab::compare_same_order(&bar_w, &grad_w);
    assert!(
        abs_w / scale(&bar_w) < tolerance,
        "{} bar_w max_abs {abs_w:e} scale {:e} rel {:e}",
        identity::<T>(),
        scale(&bar_w),
        abs_w / scale(&bar_w)
    );

    let mut grad_y = vec![T::from_f64(0.0); cg.num_edges * 16];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_y(&buffers, &mut grad_y, exec)
        .expect("vjp-y");
    let (abs_y, _) = mace_ab::compare_same_order(&bar_y, &grad_y);
    assert!(
        abs_y / scale(&bar_y) < tolerance,
        "{} bar_y max_abs {abs_y:e} scale {:e} rel {:e}",
        identity::<T>(),
        scale(&bar_y),
        abs_y / scale(&bar_y)
    );
}

#[test]
fn both_sides_compute_the_same_forward_f64() {
    check_forward::<f64>();
}

#[test]
fn both_sides_compute_the_same_forward_f32() {
    check_forward::<f32>();
}

#[test]
fn both_sides_compute_the_same_vjp_f64() {
    check_vjp::<f64>();
}

#[test]
fn both_sides_compute_the_same_vjp_f32() {
    check_vjp::<f32>();
}

/// A comparison that only looked at node 0 would pass every gate above, so this
/// perturbs the *last* node of each buffer and requires the comparison to fail.
#[test]
fn comparisons_cover_every_node() {
    let topology = small();
    let cg = cg::build::<f64>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 13);
    let f = cg.feature_dim();
    let slots = cg.out_m_total();
    let last = cg.num_nodes - 1;

    let mut strided = vec![0.0f64; cg.num_nodes * slots * f];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut strided, exec)
        .expect("fwd");
    let (_, llmlip) = cg.llmlip_forward(&buffers).expect("llmlip fwd");
    let (base, _) = mace_ab::compare_forward(&cg, &llmlip, &strided);
    strided[last * slots * f] += 1.0;
    let (moved, _) = mace_ab::compare_forward(&cg, &llmlip, &strided);
    assert!(base < 1e-3, "unperturbed forward must agree, got {base:e}");
    assert!(
        moved > 0.5,
        "compare_forward ignored node {last} (base {base:e}, moved {moved:e})"
    );

    // `bar_x` is compared dense, so a corruption anywhere has to show up. One at
    // a time and each restored: a combined corruption would not say which element
    // the comparison missed — the narrow view is `[node][feature]` over the first
    // `N*f` elements, so `N*f` itself is already wide[0], not a narrow slot.
    let (_, (bar_x, bar_y, bar_w)) = cg.llmlip_vjp(&buffers).expect("llmlip vjp");
    let mut grad_x = vec![0.0f64; cg.num_nodes * (128 + 3 * f)];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_x(&buffers, &mut grad_x, exec)
        .expect("vjp-x");
    let (base_x, _) = {
        let (narrow, wide) = grad_x.split_at(cg.num_nodes * f);
        mace_ab::compare_grad_x(&cg, &bar_x, narrow, wide)
    };
    assert!(
        base_x < 1e-3,
        "unperturbed bar_x must agree, got {base_x:e}"
    );
    let narrow_mid = (cg.num_nodes / 2) * f + f / 2;
    let narrow_tail = cg.num_nodes * f - 1;
    let wide_tail = grad_x.len() - 1;
    for (label, index) in [
        ("narrow middle", narrow_mid),
        ("narrow tail", narrow_tail),
        ("wide tail", wide_tail),
    ] {
        grad_x[index] += 1.0;
        let moved = {
            let (narrow, wide) = grad_x.split_at(cg.num_nodes * f);
            mace_ab::compare_grad_x(&cg, &bar_x, narrow, wide).0
        };
        grad_x[index] -= 1.0;
        assert!(
            moved > 0.5,
            "compare_grad_x ignored a corruption of bar_x at {index} ({label}), got {moved:e}"
        );
    }

    let mut grad_w = vec![0.0f64; cg.num_edges * 10 * f];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_w(&buffers, &mut grad_w, exec)
        .expect("vjp-w");
    let (base_w, _) = mace_ab::compare_same_order(&bar_w, &grad_w);
    if let Some(last_value) = grad_w.last_mut() {
        *last_value += 1.0;
    }
    let (moved_w, _) = mace_ab::compare_same_order(&bar_w, &grad_w);
    assert!(
        base_w < 1e-3,
        "unperturbed bar_w must agree, got {base_w:e}"
    );
    assert!(
        moved_w > 0.5,
        "compare_same_order ignored the tail of bar_w"
    );

    let mut grad_y = vec![0.0f64; cg.num_edges * 16];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_y(&buffers, &mut grad_y, exec)
        .expect("vjp-y");
    let (base_y, _) = mace_ab::compare_same_order(&bar_y, &grad_y);
    if let Some(last_value) = grad_y.last_mut() {
        *last_value += 1.0;
    }
    let (moved_y, _) = mace_ab::compare_same_order(&bar_y, &grad_y);
    assert!(
        base_y < 1e-3,
        "unperturbed bar_y must agree, got {base_y:e}"
    );
    assert!(
        moved_y > 0.5,
        "compare_same_order ignored the tail of bar_y"
    );
}

fn check_thread_invariance<T: Scalar + PartialEq + std::fmt::Debug>() {
    let topology = small();
    let cg = cg::build::<T>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 7);
    let pool = cg::pool(4).expect("pool");

    let mut serial = vec![T::from_f64(0.0); cg.num_nodes * cg.out_m_total() * cg.feature_dim()];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_forward(&buffers, &mut serial, exec)
        .expect("serial");
    let mut pooled = vec![T::from_f64(0.0); serial.len()];
    let exec = mace_ab::exec_for(4, Some(&pool));
    cg.strided_forward(&buffers, &mut pooled, exec)
        .expect("pooled");
    assert_eq!(
        serial,
        pooled,
        "{}: 1T and 4T forward must agree bitwise",
        identity::<T>()
    );

    let x_len = cg.num_nodes * (128 + 3 * cg.feature_dim());
    let mut serial_x = vec![T::from_f64(0.0); x_len];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_x(&buffers, &mut serial_x, exec)
        .expect("vjp-x serial");
    let mut pooled_x = vec![T::from_f64(0.0); x_len];
    let exec = mace_ab::exec_for(4, Some(&pool));
    cg.strided_vjp_x(&buffers, &mut pooled_x, exec)
        .expect("vjp-x pooled");
    assert_eq!(
        serial_x,
        pooled_x,
        "{}: 1T and 4T bar_x must agree bitwise",
        identity::<T>()
    );

    let w_len = cg.num_edges * 10 * cg.feature_dim();
    let mut serial_w = vec![T::from_f64(0.0); w_len];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_w(&buffers, &mut serial_w, exec)
        .expect("vjp-w serial");
    let mut pooled_w = vec![T::from_f64(0.0); w_len];
    let exec = mace_ab::exec_for(4, Some(&pool));
    cg.strided_vjp_w(&buffers, &mut pooled_w, exec)
        .expect("vjp-w pooled");
    assert_eq!(
        serial_w,
        pooled_w,
        "{}: 1T and 4T bar_w must agree bitwise",
        identity::<T>()
    );

    let y_len = cg.num_edges * 16;
    let mut serial_y = vec![T::from_f64(0.0); y_len];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_y(&buffers, &mut serial_y, exec)
        .expect("vjp-y serial");
    let mut pooled_y = vec![T::from_f64(0.0); y_len];
    let exec = mace_ab::exec_for(4, Some(&pool));
    cg.strided_vjp_y(&buffers, &mut pooled_y, exec)
        .expect("vjp-y pooled");
    assert_eq!(
        serial_y,
        pooled_y,
        "{}: 1T and 4T bar_y must agree bitwise",
        identity::<T>()
    );
}

#[test]
fn bits_are_thread_invariant_f64() {
    check_thread_invariance::<f64>();
}

#[test]
fn bits_are_thread_invariant_f32() {
    check_thread_invariance::<f32>();
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

#[test]
fn cg_and_layers_are_correctly_sized() {
    let topology = small();
    let cg: Cg<f64> = cg::build(&topology).expect("plans");
    assert_eq!(cg.num_nodes, 32);
    assert!(cg.num_edges > 0);
}

/// The fused VJP is the same three gradients computed in one traversal: it must
/// reproduce the three split plans bit for bit (1T and 4T), and still agree with
/// llmlip within the dtype's scale-relative tolerance.
/// `assert_eq!` on floats is numerical equality (`-0.0 == 0.0`); the gate below is
/// bit identity, so compare the bit patterns. An `f32` widens to `f64` exactly.
fn assert_same_bits<T: Scalar>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual.to_f64().to_bits(),
            expected.to_f64().to_bits(),
            "{what}: element {index} is {}, not {}",
            actual.to_f64(),
            expected.to_f64()
        );
    }
}

fn check_fused_vjp<T: Scalar + PartialEq + std::fmt::Debug>() {
    let topology = small();
    let cg = cg::build::<T>(&topology).expect("plans");
    let buffers = mace_ab::buffers(&cg, 17);
    let (path, (bar_x, bar_y, bar_w)) = cg.llmlip_vjp(&buffers).expect("llmlip vjp");
    assert_eq!(path, "layer1-fast");
    let x_len = cg.num_nodes * (128 + 3 * cg.feature_dim());
    let w_len = cg.num_edges * 10 * cg.feature_dim();
    let y_len = cg.num_edges * 16;

    // Reference: the three split plans, each its own call, serial and fresh.
    let mut split_x = vec![T::from_f64(0.0); x_len];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_x(&buffers, &mut split_x, exec)
        .expect("vjp-x");
    let mut split_w = vec![T::from_f64(0.0); w_len];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_w(&buffers, &mut split_w, exec)
        .expect("vjp-w");
    let mut split_y = vec![T::from_f64(0.0); y_len];
    let exec = mace_ab::exec_for(1, None);
    cg.strided_vjp_y(&buffers, &mut split_y, exec)
        .expect("vjp-y");

    let pool = cg::pool(4).expect("pool");
    let mut fused_x = vec![T::from_f64(0.0); x_len];
    let mut fused_w = vec![T::from_f64(0.0); w_len];
    let mut fused_y = vec![T::from_f64(0.0); y_len];
    for threads in [1usize, 4] {
        fused_x.iter_mut().for_each(|value| *value = T::from_f64(0.0));
        fused_w.iter_mut().for_each(|value| *value = T::from_f64(0.0));
        fused_y.iter_mut().for_each(|value| *value = T::from_f64(0.0));
        let exec = mace_ab::exec_for(threads, (threads > 1).then_some(&pool));
        cg.strided_vjp_fused(&buffers, &mut fused_x, &mut fused_w, &mut fused_y, exec)
            .expect("vjp fused");
        assert_same_bits(
            &fused_x,
            &split_x,
            &format!("{}: {threads}T fused bar_x", identity::<T>()),
        );
        assert_same_bits(
            &fused_w,
            &split_w,
            &format!("{}: {threads}T fused bar_w", identity::<T>()),
        );
        assert_same_bits(
            &fused_y,
            &split_y,
            &format!("{}: {threads}T fused bar_y", identity::<T>()),
        );
    }

    let tolerance = tolerance::<T>();
    let (narrow, wide) = fused_x.split_at(cg.num_nodes * cg.feature_dim());
    let (abs_x, _) = mace_ab::compare_grad_x(&cg, &bar_x, narrow, wide);
    let (abs_w, _) = mace_ab::compare_same_order(&bar_w, &fused_w);
    let (abs_y, _) = mace_ab::compare_same_order(&bar_y, &fused_y);
    for (name, abs, reference) in [
        ("bar_x", abs_x, scale(&bar_x)),
        ("bar_w", abs_w, scale(&bar_w)),
        ("bar_y", abs_y, scale(&bar_y)),
    ] {
        assert!(
            abs / reference < tolerance,
            "{} fused {name} max_abs {abs:e} on scale {reference:e}",
            identity::<T>()
        );
    }
}

#[test]
fn the_fused_vjp_matches_the_split_plans_f64() {
    check_fused_vjp::<f64>();
}

#[test]
fn the_fused_vjp_matches_the_split_plans_f32() {
    check_fused_vjp::<f32>();
}
