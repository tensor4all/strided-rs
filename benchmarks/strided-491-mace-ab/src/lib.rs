//! strided-rs#491: `strided-multilinear` against llmlip.rs on the MACE-MP-0b3
//! layer-1 CG forward and VJP, same descriptor, same graph, same inputs.

pub mod cg;
pub mod measure;
pub mod topo;

/// Installed here, not in the binary: `tests/` links this library, and a counter
/// that is never installed reads 0 for every allocation — which would make every
/// allocation gate pass vacuously.
#[global_allocator]
static ALLOCATOR: measure::Counting = measure::Counting;

use cg::{Buffers, Cg, Scalar};
use strided_basic::{Exec, Pool};

/// Deterministic spread values; only the shape and layout matter for timing.
fn sequence(len: usize, start: f64, step: f64) -> Vec<f64> {
    (0..len).map(|index| start + step * index as f64).collect()
}

/// Inputs of one cell. The weights carry the model's per-block prescale
/// (`MP0B_LAYER1_HIDDEN_PATH_WEIGHTS`), which the kernel expects folded in
/// (`mace_lite.rs:107-112`).
pub fn buffers<T: Scalar>(cg: &Cg<T>, seed: u64) -> Buffers<T> {
    let desc = &cg.desc;
    let f = desc.feature_dim;
    let n = cg.num_nodes;
    let e = cg.num_edges;
    let scale = 1.0 + (seed % 7) as f64 * 0.1;
    let linear_up = convert(sequence(n * desc.in1_dim, -0.3 * scale, 0.000_011));
    let edge_attrs = convert(sequence(e * desc.in2_dim, 0.2 * scale, -0.000_013));
    let mut tp_weights = sequence(e * desc.weight_numel, 0.015 * scale, 0.000_001);
    for block in &desc.blocks {
        for feature in 0..f {
            for edge in 0..e {
                tp_weights[edge * desc.weight_numel + block.weight_base + feature] *=
                    block.path_weight;
            }
        }
    }
    let tp_weights = convert(tp_weights);
    let bar_scatter = convert(sequence(n * 5120, 0.05 * scale, 0.000_002));
    let bar_scatter_major = cg::permute_to_major(&bar_scatter, n, desc);
    Buffers {
        linear_up,
        edge_attrs,
        tp_weights,
        bar_scatter,
        bar_scatter_major,
    }
}

fn convert<T: Scalar>(values: Vec<f64>) -> Vec<T> {
    values.into_iter().map(T::from_f64).collect()
}

/// Largest magnitude in a reference buffer, so an absolute error can be read
/// scale relative (elementwise relative error is meaningless where terms cancel).
#[must_use]
pub fn scale<T: Scalar>(reference: &[T]) -> f64 {
    reference
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.to_f64().abs()))
}

fn diff(value: f64, reference: f64) -> (f64, f64) {
    let abs = (value - reference).abs();
    (abs, abs / reference.abs().max(1e-30))
}

/// `(max |Δ|, max |Δ|/|ref|)` of the forward, strided against llmlip, over
/// every node (llmlip packs a node as `[slot][feature]`, the plan as
/// `[node][slot][feature]`, so the node stride is explicit here).
pub fn compare_forward<T: Scalar>(cg: &Cg<T>, llmlip: &[T], strided_major: &[T]) -> (f64, f64) {
    let f = cg.desc.feature_dim;
    let slots = cg.out_m_total();
    let mut worst = (0.0f64, 0.0f64);
    for node in 0..cg.num_nodes {
        let mut block_offset = 0usize;
        for block in &cg.desc.blocks {
            for feature in 0..f {
                for k_m in 0..block.out_dim {
                    let reference = llmlip
                        [node * slots * f + block_offset + feature * block.out_dim + k_m]
                        .to_f64();
                    let slot = block.out_base / f + k_m;
                    let value = strided_major[node * slots * f + slot * f + feature].to_f64();
                    let (abs, rel) = diff(value, reference);
                    worst = (worst.0.max(abs), worst.1.max(rel));
                }
            }
            block_offset += f * block.out_dim;
        }
    }
    worst
}

/// `(max |Δ|, max |Δ|/|ref|)` of `bar_linear_up`, in llmlip's own order.
pub fn compare_grad_x<T: Scalar>(cg: &Cg<T>, llmlip: &[T], narrow: &[T], wide: &[T]) -> (f64, f64) {
    let f = cg.desc.feature_dim;
    let mut worst = (0.0f64, 0.0f64);
    for node in 0..cg.num_nodes {
        for feature in 0..f {
            let (abs, rel) = diff(
                narrow[node * f + feature].to_f64(),
                llmlip[node * 512 + feature].to_f64(),
            );
            worst = (worst.0.max(abs), worst.1.max(rel));
            for m in 0..3 {
                let (abs, rel) = diff(
                    wide[node * 3 * f + feature * 3 + m].to_f64(),
                    llmlip[node * 512 + 128 + feature * 3 + m].to_f64(),
                );
                worst = (worst.0.max(abs), worst.1.max(rel));
            }
        }
    }
    worst
}

/// `(max |Δ|, max |Δ|/|ref|)` of one same-order buffer against llmlip.
pub fn compare_same_order<T: Scalar>(reference: &[T], value: &[T]) -> (f64, f64) {
    let mut worst = (0.0f64, 0.0f64);
    for (reference, value) in reference.iter().zip(value) {
        let (abs, rel) = diff(value.to_f64(), reference.to_f64());
        worst = (worst.0.max(abs), worst.1.max(rel));
    }
    worst
}

#[must_use]
pub fn exec_for(threads: usize, pool: Option<&Pool>) -> Exec<'_> {
    match pool {
        Some(pool) if threads > 1 => Exec::new(pool, threads).expect("an exec token"),
        _ => Exec::serial(),
    }
}
