//! The graph both sides share: a diamond-Si supercell and its directed edge list.

use llmlip::geometry::Cell;
use llmlip::neighbor::cell_list_multi_image_neighbor_list;
use llmlip::structure::Structure;

/// One directed-edge list, used unchanged by both libraries.
#[derive(Clone, Debug)]
pub struct Topology {
    pub num_nodes: usize,
    pub senders: Vec<usize>,
    pub receivers: Vec<usize>,
}

impl Topology {
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.senders.len()
    }

    #[must_use]
    pub fn avg_neighbors(&self) -> f64 {
        self.num_edges() as f64 / self.num_nodes as f64
    }
}

/// Diamond-Si conventional cell, 8 atoms, cubic lattice constant 5.431 Å.
fn diamond_basis(lattice: f64) -> Vec<[f64; 3]> {
    let fcc = [
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ];
    let basis_shift = (0.25, 0.25, 0.25);
    let mut positions = Vec::with_capacity(8);
    for &(x, y, z) in &fcc {
        positions.push([x * lattice, y * lattice, z * lattice]);
        positions.push([
            (x + basis_shift.0) * lattice,
            (y + basis_shift.1) * lattice,
            (z + basis_shift.2) * lattice,
        ]);
    }
    positions
}

/// `repeats` copies of the 8-atom cell along each axis.
pub fn diamond_si(repeats: [usize; 3], cutoff: f64) -> Result<Topology, String> {
    let lattice = 5.431;
    let basis = diamond_basis(lattice);
    let (nx, ny, nz) = (repeats[0], repeats[1], repeats[2]);
    let mut species = Vec::new();
    let mut positions = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for position in &basis {
                    positions.push([
                        position[0] + ix as f64 * lattice,
                        position[1] + iy as f64 * lattice,
                        position[2] + iz as f64 * lattice,
                    ]);
                    species.push("Si".to_owned());
                }
            }
        }
    }
    let cell = Cell::from_vectors(
        [nx as f64 * lattice, 0.0, 0.0],
        [0.0, ny as f64 * lattice, 0.0],
        [0.0, 0.0, nz as f64 * lattice],
    );
    let structure = Structure::new(species, positions).with_cell(cell, [true; 3]);
    let neighbors = cell_list_multi_image_neighbor_list(&structure, cutoff)
        .map_err(|error| format!("neighbor list: {error}"))?;
    let num_nodes = structure.len();
    let senders = neighbors.edges.iter().map(|edge| edge.src).collect();
    let receivers = neighbors.edges.iter().map(|edge| edge.dst).collect();
    Ok(Topology {
        num_nodes,
        senders,
        receivers,
    })
}
