//! The layer-1 `conv_tp` (CG message) A/B: llmlip's fused kernel against
//! `strided-segpoly` executing the same descriptor.
//!
//! Layouts. llmlip addresses a row as `base + feature*component_dim + m`
//! (`mace_lite.rs:34416-34519`, `:36078-36082`), i.e.
//! `in1 = [0e: 128 scalars][1o: 128 vectors]`, `out = [block][feature][m]`.
//! `strided-segpoly` needs `[batch, num_segments, u]` with one stride per axis,
//! so the strided side views the same buffers with `u = 128 features`:
//!
//! | buffer | dims | strides | llmlip memory |
//! | --- | --- | --- | --- |
//! | `x0e` | `[N, 1, 128]` | `[512, 1, 1]` | affine view |
//! | `x1o` | `[N, 3, 128]` | `[512, 1, 3]` | affine view |
//! | `y` | `[E, 16, 1]` | `[16, 1, 1]` | affine view |
//! | `w` | `[E, 10, 128]` | `[1280, 128, 1]` | affine view |
//! | `m` | `[N, 40, 128]` | `[5120, 128, 1]` | permutation of `[block][feature][m]` |
//!
//! `grad_w` and `grad_y` keep llmlip's own order, so they compare element for
//! element; `m`/`grad_m` need one untimed permutation (`permute_*`).

use std::cell::RefCell;

use llmlip::mace_lite::bench_api::{self, Layer1Descriptor, Layer1Plan};
use strided_basic::{Exec, Pool, RawStridedMut, RawStridedRef, RunIndex};
use strided_segpoly::{IndexMode, Layout, Operation, Path, SegPolyElem, SegPolyPlan, Spec};

use crate::topo::Topology;

/// The element types both sides support.
pub trait Scalar: SegPolyElem + CgScalar {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

/// The llmlip call, dispatched on the element type.
pub trait CgScalar: Copy {
    fn call_forward(
        plan: &Layer1Plan,
        linear_up: &[Self],
        edge_attrs: &[Self],
        tp_weights: &[Self],
        senders: &[usize],
        receivers: &[usize],
        num_nodes: usize,
    ) -> Result<(&'static str, Vec<Self>), String>;

    fn call_vjp(
        plan: &Layer1Plan,
        linear_up: &[Self],
        edge_attrs: &[Self],
        tp_weights: &[Self],
        bar: &[Self],
        senders: &[usize],
        receivers: &[usize],
    ) -> Result<(&'static str, (Vec<Self>, Vec<Self>, Vec<Self>)), String>;
}

impl CgScalar for f64 {
    fn call_forward(
        plan: &Layer1Plan,
        linear_up: &[Self],
        edge_attrs: &[Self],
        tp_weights: &[Self],
        senders: &[usize],
        receivers: &[usize],
        num_nodes: usize,
    ) -> Result<(&'static str, Vec<Self>), String> {
        bench_api::layer1_fwd(
            plan, linear_up, edge_attrs, tp_weights, senders, receivers, num_nodes,
        )
        .map_err(|error| format!("llmlip fwd: {error}"))
    }

    fn call_vjp(
        plan: &Layer1Plan,
        linear_up: &[Self],
        edge_attrs: &[Self],
        tp_weights: &[Self],
        bar: &[Self],
        senders: &[usize],
        receivers: &[usize],
    ) -> Result<(&'static str, (Vec<Self>, Vec<Self>, Vec<Self>)), String> {
        bench_api::layer1_vjp(
            plan, linear_up, edge_attrs, tp_weights, bar, senders, receivers,
        )
        .map_err(|error| format!("llmlip vjp: {error}"))
    }
}

impl CgScalar for f32 {
    fn call_forward(
        plan: &Layer1Plan,
        linear_up: &[Self],
        edge_attrs: &[Self],
        tp_weights: &[Self],
        senders: &[usize],
        receivers: &[usize],
        num_nodes: usize,
    ) -> Result<(&'static str, Vec<Self>), String> {
        bench_api::layer1_fwd_f32(
            plan, linear_up, edge_attrs, tp_weights, senders, receivers, num_nodes,
        )
        .map_err(|error| format!("llmlip fwd f32: {error}"))
    }

    fn call_vjp(
        plan: &Layer1Plan,
        linear_up: &[Self],
        edge_attrs: &[Self],
        tp_weights: &[Self],
        bar: &[Self],
        senders: &[usize],
        receivers: &[usize],
    ) -> Result<(&'static str, (Vec<Self>, Vec<Self>, Vec<Self>)), String> {
        bench_api::layer1_vjp_f32(
            plan, linear_up, edge_attrs, tp_weights, bar, senders, receivers,
        )
        .map_err(|error| format!("llmlip vjp f32: {error}"))
    }
}

impl Scalar for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl Scalar for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

/// Every buffer of one cell, in llmlip's own memory order where possible.
pub struct Buffers<T: Scalar> {
    /// in1: `[N, 512]`, `[0e: 128][1o: 128 x 3]`.
    pub linear_up: Vec<T>,
    /// SH: `[E, 16]`.
    pub edge_attrs: Vec<T>,
    /// Prescaled CG weights: `[E, 1280]`, `[block][feature]`.
    pub tp_weights: Vec<T>,
    /// Per-node message gradient for the VJP: llmlip order `[N, 5120]`.
    pub bar_scatter: Vec<T>,
    /// The same gradient in the strided major layout `[N, 40 slots, 128]`.
    pub bar_scatter_major: Vec<T>,
}

/// The four plans plus the shared index data.
pub struct Cg<T: Scalar> {
    pub desc: Layer1Descriptor,
    pub num_nodes: usize,
    pub num_edges: usize,
    senders: Vec<i32>,
    receivers: Vec<i32>,
    runs_source: RunIndex,
    runs_receiver: RunIndex,
    pub fwd: SegPolyPlan<T>,
    pub vjp_x: SegPolyPlan<T>,
    pub vjp_w: SegPolyPlan<T>,
    pub vjp_y: SegPolyPlan<T>,
    /// Each plan's scratch. Behind a `RefCell` so both sides of a cell can hold
    /// their closures at once: the measured cell has to interleave them.
    pub fwd_scratch: RefCell<Vec<T>>,
    pub vjp_x_scratch: RefCell<Vec<T>>,
    pub vjp_w_scratch: RefCell<Vec<T>>,
    pub vjp_y_scratch: RefCell<Vec<T>>,
    plan: Layer1Plan,
    senders_usize: Vec<usize>,
    receivers_usize: Vec<usize>,
}

#[must_use]
pub fn out_m_total(desc: &Layer1Descriptor) -> usize {
    desc.blocks.iter().map(|block| block.out_dim).sum()
}

impl<T: Scalar> Cg<T> {
    #[must_use]
    pub fn out_m_total(&self) -> usize {
        out_m_total(&self.desc)
    }

    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.desc.feature_dim
    }

    pub fn llmlip_forward(&self, buffers: &Buffers<T>) -> Result<(&'static str, Vec<T>), String> {
        T::call_forward(
            &self.plan,
            &buffers.linear_up,
            &buffers.edge_attrs,
            &buffers.tp_weights,
            &self.senders_usize,
            &self.receivers_usize,
            self.num_nodes,
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn llmlip_vjp(
        &self,
        buffers: &Buffers<T>,
    ) -> Result<(&'static str, (Vec<T>, Vec<T>, Vec<T>)), String> {
        T::call_vjp(
            &self.plan,
            &buffers.linear_up,
            &buffers.edge_attrs,
            &buffers.tp_weights,
            &buffers.bar_scatter,
            &self.senders_usize,
            &self.receivers_usize,
        )
    }

    pub fn strided_forward(
        &self,
        buffers: &Buffers<T>,
        m: &mut [T],
        exec: Exec<'_>,
    ) -> Result<(), String> {
        let n = self.num_nodes;
        let e = self.num_edges;
        let f = self.desc.feature_dim;
        let slots = self.out_m_total();
        let x0e_dims = [n, 1, f];
        let x0e_strides = [4 * f as isize, 1, 1];
        let x1o_dims = [n, 3, f];
        let x1o_strides = [4 * f as isize, 1, 3];
        let y_dims = [e, 16, 1];
        let y_strides = [16, 1isize, 1];
        let w_dims = [e, 10, f];
        let w_strides = [10 * f as isize, f as isize, 1];
        let m_dims = [n, slots, f];
        let m_strides = [(slots * f) as isize, f as isize, 1];
        let x = &buffers.linear_up;
        let inputs = [
            RawStridedRef::new(x, &x0e_dims, &x0e_strides, 0).map_err(err)?,
            RawStridedRef::new(x, &x1o_dims, &x1o_strides, 128).map_err(err)?,
            RawStridedRef::new(&buffers.edge_attrs, &y_dims, &y_strides, 0).map_err(err)?,
            RawStridedRef::new(&buffers.tp_weights, &w_dims, &w_strides, 0).map_err(err)?,
        ];
        let mut outputs = [RawStridedMut::new(m, &m_dims, &m_strides, 0).map_err(err)?];
        let gather = [IndexMode::Gather(&self.senders)];
        let batched = [IndexMode::Batched];
        let scatter = [IndexMode::Scatter(&self.runs_receiver)];
        let indexes: [&[IndexMode<'_>]; 5] = [&gather, &gather, &batched, &batched, &scatter];
        self.fwd
            .execute_into(
                &mut outputs,
                &inputs,
                &indexes,
                &mut *self.fwd_scratch.borrow_mut(),
                exec,
            )
            .map_err(|error| format!("segpoly fwd: {error}"))
    }

    pub fn strided_vjp_x(
        &self,
        buffers: &Buffers<T>,
        grad_x: &mut [T],
        exec: Exec<'_>,
    ) -> Result<(), String> {
        let n = self.num_nodes;
        let e = self.num_edges;
        let f = self.desc.feature_dim;
        let slots = self.out_m_total();
        let y_dims = [e, 16, 1];
        let y_strides = [16, 1isize, 1];
        let w_dims = [e, 10, f];
        let w_strides = [10 * f as isize, f as isize, 1];
        let gm_dims = [n, slots, f];
        let gm_strides = [(slots * f) as isize, f as isize, 1];
        let gx0e_dims = [n, 1, f];
        let gx0e_strides = [f as isize, 1isize, 1];
        let gx1o_dims = [n, 3, f];
        let gx1o_strides = [3 * f as isize, 1isize, 3];
        let grad_m = &buffers.bar_scatter_major;
        let inputs = [
            RawStridedRef::new(&buffers.edge_attrs, &y_dims, &y_strides, 0).map_err(err)?,
            RawStridedRef::new(&buffers.tp_weights, &w_dims, &w_strides, 0).map_err(err)?,
            RawStridedRef::new(grad_m, &gm_dims, &gm_strides, 0).map_err(err)?,
        ];
        let (narrow, wide) = grad_x.split_at_mut(n * f);
        let mut outputs = [
            RawStridedMut::new(narrow, &gx0e_dims, &gx0e_strides, 0).map_err(err)?,
            RawStridedMut::new(wide, &gx1o_dims, &gx1o_strides, 0).map_err(err)?,
        ];
        let batched = [IndexMode::Batched];
        let gather_r = [IndexMode::Gather(&self.receivers)];
        let scatter = [IndexMode::Scatter(&self.runs_source)];
        let indexes: [&[IndexMode<'_>]; 5] = [&batched, &batched, &gather_r, &scatter, &scatter];
        self.vjp_x
            .execute_into(
                &mut outputs,
                &inputs,
                &indexes,
                &mut *self.vjp_x_scratch.borrow_mut(),
                exec,
            )
            .map_err(|error| format!("segpoly vjp-x: {error}"))
    }

    pub fn strided_vjp_w(
        &self,
        buffers: &Buffers<T>,
        grad_w: &mut [T],
        exec: Exec<'_>,
    ) -> Result<(), String> {
        let n = self.num_nodes;
        let e = self.num_edges;
        let f = self.desc.feature_dim;
        let slots = self.out_m_total();
        let x0e_dims = [n, 1, f];
        let x0e_strides = [4 * f as isize, 1, 1];
        let x1o_dims = [n, 3, f];
        let x1o_strides = [4 * f as isize, 1, 3];
        let y_dims = [e, 16, 1];
        let y_strides = [16, 1isize, 1];
        let gm_dims = [n, slots, f];
        let gm_strides = [(slots * f) as isize, f as isize, 1];
        let gw_dims = [e, 10, f];
        let gw_strides = [10 * f as isize, f as isize, 1];
        let x = &buffers.linear_up;
        let grad_m = &buffers.bar_scatter_major;
        let inputs = [
            RawStridedRef::new(x, &x0e_dims, &x0e_strides, 0).map_err(err)?,
            RawStridedRef::new(x, &x1o_dims, &x1o_strides, 128).map_err(err)?,
            RawStridedRef::new(&buffers.edge_attrs, &y_dims, &y_strides, 0).map_err(err)?,
            RawStridedRef::new(grad_m, &gm_dims, &gm_strides, 0).map_err(err)?,
        ];
        let mut outputs = [RawStridedMut::new(grad_w, &gw_dims, &gw_strides, 0).map_err(err)?];
        let gather_s = [IndexMode::Gather(&self.senders)];
        let batched = [IndexMode::Batched];
        let gather_r = [IndexMode::Gather(&self.receivers)];
        let indexes: [&[IndexMode<'_>]; 5] = [&gather_s, &gather_s, &batched, &gather_r, &batched];
        self.vjp_w
            .execute_into(
                &mut outputs,
                &inputs,
                &indexes,
                &mut *self.vjp_w_scratch.borrow_mut(),
                exec,
            )
            .map_err(|error| format!("segpoly vjp-w: {error}"))
    }

    pub fn strided_vjp_y(
        &self,
        buffers: &Buffers<T>,
        grad_y: &mut [T],
        exec: Exec<'_>,
    ) -> Result<(), String> {
        let n = self.num_nodes;
        let e = self.num_edges;
        let f = self.desc.feature_dim;
        let w_len = 10 * f;
        let x_dims = [n, 512, 1];
        let x_strides = [512, 1isize, 1];
        let w_dims = [e, w_len, 1];
        let w_strides = [w_len as isize, 1isize, 1];
        let gm_dims = [n, self.out_m_total() * f, 1];
        let gm_strides = [(self.out_m_total() * f) as isize, 1isize, 1];
        let gy_dims = [e, 16, 1];
        let gy_strides = [16, 1isize, 1];
        let grad_m = &buffers.bar_scatter_major;
        let inputs = [
            RawStridedRef::new(&buffers.linear_up, &x_dims, &x_strides, 0).map_err(err)?,
            RawStridedRef::new(&buffers.tp_weights, &w_dims, &w_strides, 0).map_err(err)?,
            RawStridedRef::new(grad_m, &gm_dims, &gm_strides, 0).map_err(err)?,
        ];
        let mut outputs = [RawStridedMut::new(grad_y, &gy_dims, &gy_strides, 0).map_err(err)?];
        let gather_s = [IndexMode::Gather(&self.senders)];
        let batched = [IndexMode::Batched];
        let gather_r = [IndexMode::Gather(&self.receivers)];
        let indexes: [&[IndexMode<'_>]; 4] = [&gather_s, &batched, &gather_r, &batched];
        self.vjp_y
            .execute_into(
                &mut outputs,
                &inputs,
                &indexes,
                &mut *self.vjp_y_scratch.borrow_mut(),
                exec,
            )
            .map_err(|error| format!("segpoly vjp-y: {error}"))
    }
}

fn err<E: std::fmt::Display>(error: E) -> String {
    format!("{error}")
}

/// llmlip order `[block][feature][m]` -> strided order `[slot=(block,m)][feature]`.
#[must_use]
pub fn permute_to_major<T: Scalar>(
    llmlip_order: &[T],
    num_nodes: usize,
    desc: &Layer1Descriptor,
) -> Vec<T> {
    let f = desc.feature_dim;
    let slots: usize = desc.blocks.iter().map(|block| block.out_dim).sum();
    let mut out = vec![T::from_f64(0.0); num_nodes * slots * f];
    for node in 0..num_nodes {
        let mut block_offset = 0usize;
        for block in &desc.blocks {
            for feature in 0..f {
                for k_m in 0..block.out_dim {
                    let src = node * slots * f + block_offset + feature * block.out_dim + k_m;
                    let slot = block.out_base / f + k_m;
                    out[node * slots * f + slot * f + feature] = llmlip_order[src];
                }
            }
            block_offset += f * block.out_dim;
        }
    }
    out
}

fn leak_segments(paths: &[[u32; 4]]) -> Vec<&'static [u32]> {
    // `compile` copies what it needs; these leak only for the process lifetime
    // of a benchmark binary with a handful of plans.
    paths
        .iter()
        .map(|segments| &*Box::leak(segments.to_vec().into_boxed_slice()))
        .collect()
}

/// Build the four plans for one topology.
/// The descriptor alone, for tests that pin the published counters.
pub fn layer1_descriptor_for_test() -> Layer1Descriptor {
    llmlip::mace_lite::bench_api::layer1_descriptor()
}

pub fn build<T: Scalar>(topo: &Topology) -> Result<Cg<T>, String> {
    let desc = bench_api::layer1_descriptor();
    let plan = bench_api::layer1_plan().map_err(|error| format!("llmlip plan: {error}"))?;
    let n = topo.num_nodes;
    let e = topo.num_edges();
    let f = desc.feature_dim;
    let slots = out_m_total(&desc);
    let senders: Vec<i32> = topo.senders.iter().map(|&s| s as i32).collect();
    let receivers: Vec<i32> = topo.receivers.iter().map(|&r| r as i32).collect();
    let runs_source =
        RunIndex::from_dest(&senders, n).map_err(|error| format!("source runs: {error}"))?;
    let runs_receiver =
        RunIndex::from_dest(&receivers, n).map_err(|error| format!("receiver runs: {error}"))?;

    let x0e_dims = [n, 1, f];
    let x0e_strides = [4 * f as isize, 1, 1];
    let x1o_dims = [n, 3, f];
    let x1o_strides = [4 * f as isize, 1, 3];
    let y_dims = [e, 16, 1];
    let y_strides = [16, 1isize, 1];
    let w_dims = [e, 10, f];
    let w_strides = [10 * f as isize, f as isize, 1];
    let m_dims = [n, slots, f];
    let m_strides = [(slots * f) as isize, f as isize, 1];
    let gw_dims = [e, 10, f];
    let gy_dims = [e, 16, 1];
    let xflat_dims = [n, 512, 1];
    let x0e_out_dims = [n, 1, f];
    let x0e_out_strides = [f as isize, 1isize, 1];
    let x1o_out_dims = [n, 3, f];
    let x1o_out_strides = [3 * f as isize, 1isize, 3];
    let wflat_dims = [e, 10 * f, 1];
    let gmflat_dims = [n, slots * f, 1];

    // ---- forward -----------------------------------------------------------
    let (fwd_segments, fwd_coeffs) = build_paths(&desc, Kind::Forward);
    let fwd_split = count_width_one(&desc);
    let fwd_segs = leak_segments(&fwd_segments);
    let fwd_paths = mk_paths(&fwd_segs, &fwd_coeffs);
    let fwd_ops = [
        Operation {
            buffers: &[0, 2, 3, 4],
            path_start: 0,
            num_paths: fwd_split as u32,
        },
        Operation {
            buffers: &[1, 2, 3, 4],
            path_start: fwd_split as u32,
            num_paths: (fwd_paths.len() - fwd_split) as u32,
        },
    ];
    let fwd_indexes = [
        [IndexMode::Gather(&senders)],
        [IndexMode::Gather(&senders)],
        [IndexMode::Batched],
        [IndexMode::Batched],
        [IndexMode::Scatter(&runs_receiver)],
    ];
    let fwd_spec = Spec {
        num_batch_axes: 1,
        operations: &fwd_ops,
        paths: &fwd_paths,
        indexes: &[
            &fwd_indexes[0],
            &fwd_indexes[1],
            &fwd_indexes[2],
            &fwd_indexes[3],
            &fwd_indexes[4],
        ],
    };
    let fwd = SegPolyPlan::compile(
        &fwd_spec,
        &[
            Layout {
                dims: &x0e_dims,
                strides: &x0e_strides,
            },
            Layout {
                dims: &x1o_dims,
                strides: &x1o_strides,
            },
            Layout {
                dims: &y_dims,
                strides: &y_strides,
            },
            Layout {
                dims: &w_dims,
                strides: &w_strides,
            },
        ],
        &[Layout {
            dims: &m_dims,
            strides: &m_strides,
        }],
    )
    .map_err(|error| format!("fwd compile: {error}"))?;

    // ---- VJP wrt in1 -------------------------------------------------------
    let (vx_segments, vx_coeffs) = build_paths(&desc, Kind::VjpX);
    let vx_segs = leak_segments(&vx_segments);
    let vx_paths = mk_paths(&vx_segs, &vx_coeffs);
    let vx_split = count_width_one(&desc);
    let vx_ops = [
        Operation {
            buffers: &[0, 1, 2, 3],
            path_start: 0,
            num_paths: vx_split as u32,
        },
        Operation {
            buffers: &[0, 1, 2, 4],
            path_start: vx_split as u32,
            num_paths: (vx_paths.len() - vx_split) as u32,
        },
    ];
    let vx_indexes = [
        [IndexMode::Batched],
        [IndexMode::Batched],
        [IndexMode::Gather(&receivers)],
        [IndexMode::Scatter(&runs_source)],
        [IndexMode::Scatter(&runs_source)],
    ];
    let vx_spec = Spec {
        num_batch_axes: 1,
        operations: &vx_ops,
        paths: &vx_paths,
        indexes: &[
            &vx_indexes[0],
            &vx_indexes[1],
            &vx_indexes[2],
            &vx_indexes[3],
            &vx_indexes[4],
        ],
    };
    let vjp_x = SegPolyPlan::compile(
        &vx_spec,
        &[
            Layout {
                dims: &y_dims,
                strides: &y_strides,
            },
            Layout {
                dims: &w_dims,
                strides: &w_strides,
            },
            Layout {
                dims: &m_dims,
                strides: &m_strides,
            },
        ],
        &[
            Layout {
                dims: &x0e_out_dims,
                strides: &x0e_out_strides,
            },
            Layout {
                dims: &x1o_out_dims,
                strides: &x1o_out_strides,
            },
        ],
    )
    .map_err(|error| format!("vjp-x compile: {error}"))?;

    // ---- VJP wrt the per-edge CG weights -----------------------------------
    let (vw_segments, vw_coeffs) = build_paths(&desc, Kind::VjpW);
    let vw_segs = leak_segments(&vw_segments);
    let vw_paths = mk_paths(&vw_segs, &vw_coeffs);
    let vw_split = count_width_one(&desc);
    let vw_ops = [
        Operation {
            buffers: &[0, 2, 3, 4],
            path_start: 0,
            num_paths: vw_split as u32,
        },
        Operation {
            buffers: &[1, 2, 3, 4],
            path_start: vw_split as u32,
            num_paths: (vw_paths.len() - vw_split) as u32,
        },
    ];
    let vw_indexes = [
        [IndexMode::Gather(&senders)],
        [IndexMode::Gather(&senders)],
        [IndexMode::Batched],
        [IndexMode::Gather(&receivers)],
        [IndexMode::Batched],
    ];
    let vw_spec = Spec {
        num_batch_axes: 1,
        operations: &vw_ops,
        paths: &vw_paths,
        indexes: &[
            &vw_indexes[0],
            &vw_indexes[1],
            &vw_indexes[2],
            &vw_indexes[3],
            &vw_indexes[4],
        ],
    };
    let vjp_w = SegPolyPlan::compile(
        &vw_spec,
        &[
            Layout {
                dims: &x0e_dims,
                strides: &x0e_strides,
            },
            Layout {
                dims: &x1o_dims,
                strides: &x1o_strides,
            },
            Layout {
                dims: &y_dims,
                strides: &y_strides,
            },
            Layout {
                dims: &m_dims,
                strides: &m_strides,
            },
        ],
        &[Layout {
            dims: &gw_dims,
            strides: &w_strides,
        }],
    )
    .map_err(|error| format!("vjp-w compile: {error}"))?;

    // ---- VJP wrt the spherical harmonics (a lane reduction, u = 1) ---------
    let (vy_segments, vy_coeffs) = build_paths(&desc, Kind::VjpY);
    let vy_segs = leak_segments(&vy_segments);
    let vy_paths = mk_paths(&vy_segs, &vy_coeffs);
    let vy_ops = [Operation {
        buffers: &[0, 1, 2, 3],
        path_start: 0,
        num_paths: vy_paths.len() as u32,
    }];
    let vy_indexes = [
        [IndexMode::Gather(&senders)],
        [IndexMode::Batched],
        [IndexMode::Gather(&receivers)],
        [IndexMode::Batched],
    ];
    let vy_spec = Spec {
        num_batch_axes: 1,
        operations: &vy_ops,
        paths: &vy_paths,
        indexes: &[
            &vy_indexes[0],
            &vy_indexes[1],
            &vy_indexes[2],
            &vy_indexes[3],
        ],
    };
    let vjp_y = SegPolyPlan::compile(
        &vy_spec,
        &[
            Layout {
                dims: &xflat_dims,
                strides: &[512, 1, 1],
            },
            Layout {
                dims: &wflat_dims,
                strides: &[(10 * f) as isize, 1, 1],
            },
            Layout {
                dims: &gmflat_dims,
                strides: &[(slots * f) as isize, 1, 1],
            },
        ],
        &[Layout {
            dims: &gy_dims,
            strides: &[16, 1, 1],
        }],
    )
    .map_err(|error| format!("vjp-y compile: {error}"))?;

    let features = 8usize;
    Ok(Cg {
        fwd_scratch: RefCell::new(vec![T::from_f64(0.0); fwd.scratch_len(features)]),
        vjp_x_scratch: RefCell::new(vec![T::from_f64(0.0); vjp_x.scratch_len(features)]),
        vjp_w_scratch: RefCell::new(vec![T::from_f64(0.0); vjp_w.scratch_len(features)]),
        vjp_y_scratch: RefCell::new(vec![T::from_f64(0.0); vjp_y.scratch_len(features)]),
        desc,
        num_nodes: n,
        num_edges: e,
        senders_usize: topo.senders.clone(),
        receivers_usize: topo.receivers.clone(),
        senders,
        receivers,
        runs_source,
        runs_receiver,
        fwd,
        vjp_x,
        vjp_w,
        vjp_y,
        plan,
    })
}

fn mk_paths(segments: &[&'static [u32]], coeffs: &[f64]) -> Vec<Path<'static>> {
    segments
        .iter()
        .zip(coeffs)
        .map(|(&segments, &coefficient)| Path {
            segments,
            coefficient,
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Kind {
    Forward,
    VjpX,
    VjpW,
    VjpY,
}

/// `[segment; 4]` per path, grouped by in1 block width (`x_dim == 1` first).
fn build_paths(desc: &Layer1Descriptor, kind: Kind) -> (Vec<[u32; 4]>, Vec<f64>) {
    let f = desc.feature_dim;
    let mut narrow = Vec::new();
    let mut wide = Vec::new();
    for (block_index, block) in desc.blocks.iter().enumerate() {
        let narrow_block = block.x_dim == 1;
        let target = if narrow_block { &mut narrow } else { &mut wide };
        match kind {
            Kind::Forward => {
                for &(i_m, j_m, k_m, scale) in &block.cg {
                    let x_seg = if narrow_block { 0 } else { i_m };
                    target.push((
                        [
                            x_seg as u32,
                            (block.y_base + j_m) as u32,
                            block_index as u32,
                            (block.out_base / f + k_m) as u32,
                        ],
                        scale,
                    ));
                }
            }
            Kind::VjpX => {
                for &(i_m, j_m, k_m, scale) in &block.cg {
                    let x_seg = if narrow_block { 0 } else { i_m };
                    target.push((
                        [
                            (block.y_base + j_m) as u32,
                            block_index as u32,
                            (block.out_base / f + k_m) as u32,
                            x_seg as u32,
                        ],
                        scale,
                    ));
                }
            }
            Kind::VjpW => {
                for &(i_m, j_m, k_m, scale) in &block.cg {
                    let x_seg = if narrow_block { 0 } else { i_m };
                    target.push((
                        [
                            x_seg as u32,
                            (block.y_base + j_m) as u32,
                            (block.out_base / f + k_m) as u32,
                            block_index as u32,
                        ],
                        scale,
                    ));
                }
            }
            Kind::VjpY => {
                for feature in 0..f {
                    for &(i_m, j_m, k_m, scale) in &block.cg {
                        target.push((
                            [
                                (block.x_base + feature * block.x_dim + i_m) as u32,
                                (block.weight_base + feature) as u32,
                                (((block.out_base / f + k_m) * f) + feature) as u32,
                                (block.y_base + j_m) as u32,
                            ],
                            scale,
                        ));
                    }
                }
            }
        }
    }
    let count_narrow = narrow.len();
    let mut segments = Vec::with_capacity(count_narrow + wide.len());
    let mut coeffs = Vec::with_capacity(count_narrow + wide.len());
    for (segments_entry, coefficient) in narrow.into_iter().chain(wide) {
        segments.push(segments_entry);
        coeffs.push(coefficient);
    }
    (segments, coeffs)
}

/// Number of paths whose in1 block is `l = 0` (`x_dim == 1`).
fn count_width_one(desc: &Layer1Descriptor) -> usize {
    desc.blocks
        .iter()
        .filter(|block| block.x_dim == 1)
        .map(|block| block.cg.len())
        .sum()
}

/// A pool for the strided side's multi-thread cells.
pub fn pool(threads: usize) -> Result<Pool, String> {
    Pool::builder()
        .threads(threads)
        .build()
        .map_err(|error| format!("pool: {error}"))
}
