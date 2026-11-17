//! What the four plans ask the caller to supply, per worker budget (strided-rs#496).
//!
//! The VJP wrt the spherical harmonics is a lane reduction, so its scratch now
//! carries one lane tile per worker plus the pack area of its lane-strided
//! operand; the other three plans are unchanged in shape. These are the numbers
//! `Cg::new` hands to `vec![0.0; plan.scratch_len(features)]`.

use mace_ab::cg;

/// The layer-1 A/B descriptors, at the smallest and the largest graph the
/// campaign measures.
fn plans(atoms: usize) -> cg::Cg<f64> {
    let repeats = match atoms {
        128 => [2, 2, 4],
        _ => [4, 4, 8],
    };
    let topology = mace_ab::topo::diamond_si(repeats, 6.0).expect("topology");
    cg::build::<f64>(&topology).expect("the A/B graph compiles")
}

#[test]
fn the_vjp_y_lane_reduction_scratch_is_a_tile_per_worker() {
    let features = 8;
    for atoms in [128, 1024] {
        let cg = plans(atoms);
        let fwd = cg.fwd_scratch.borrow().len();
        let vjp_y = cg.vjp_y_scratch.borrow().len();
        println!(
            "atoms {atoms}: scratch elements per call -- fwd {} (budget 1) / {} (8), \
             vjp_x {} / {}, vjp_w {} / {}, vjp_y {} / {}; buffers {fwd} and {vjp_y}",
            cg.fwd.scratch_len(1),
            cg.fwd.scratch_len(features),
            cg.vjp_x.scratch_len(1),
            cg.vjp_x.scratch_len(features),
            cg.vjp_w.scratch_len(1),
            cg.vjp_w.scratch_len(features),
            cg.vjp_y.scratch_len(1),
            cg.vjp_y.scratch_len(features)
        );
        // Both buffers are sized from `scratch_len(features)` and must cover the
        // one-worker floor.
        assert!(fwd >= cg.fwd.scratch_len(1));
        assert!(vjp_y >= cg.vjp_y.scratch_len(1));
        // Growing the budget only grows the tile part, so the floor is the same
        // plan at budget 1.
        assert!(cg.vjp_y.scratch_len(1) <= cg.vjp_y.scratch_len(features));
    }
}

/// The pack area exists because one operand of the u = 128 plan has lane stride 3;
/// the plan that reads the same array with lane stride 1 must be smaller.
#[test]
fn the_lane_strided_operand_costs_one_pack_area() {
    for atoms in [128, 1024] {
        let cg = plans(atoms);
        let per_worker = cg.vjp_y.scratch_len(2) - cg.vjp_y.scratch_len(1);
        let tile = cg.vjp_y.tile_elems();
        assert!(
            per_worker > tile,
            "atoms {atoms}: one worker costs {per_worker} elements, tile is {tile}"
        );
        println!("atoms {atoms}: vjp_y tile {tile}, per worker {per_worker}");
    }
}
