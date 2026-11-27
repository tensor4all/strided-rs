//! Small tiles under one long axis (strided-rs#526).
//!
//! A plan whose innermost axes cannot fuse and are all tiny, under one long
//! axis, is the shape of a per-block transpose over `[n, n, batch]` (strides
//! `[n, 1, n²]` against `[1, n, n²]`). The block walk reaches the per-operand
//! callback once per innermost run of `n` elements, and that call costs far
//! more than the `n` elements it moves.
//!
//! Two pieces handle the shape:
//!
//! - [`tile_blocks`] (planning): the tile axes stay whole and the long axis is
//!   blocked so that one block of every operand fits [`TILE_BUDGET_BYTES`],
//!   from the operands' element sizes. It replaces the block model for this
//!   shape only, whose second phase shrinks the long axis one position at a
//!   time on every call.
//! - [`SmallTile`] (pointwise walk): a per-operand offset table with one entry
//!   per tile element is built once per call. For every block of the long axis
//!   the callback is called once per tile element, with the block's length and
//!   the long axis' strides. Every element of the block is visited once, from
//!   lines the first tile pass brought into L1.
//!
//! The order of the elements inside one block changes (tile element outer,
//! long axis inner), so the walk is only for pointwise operations: each
//! destination element is computed from its own inputs and written once, and
//! the result does not depend on the order. Folds keep the block walk; with
//! the blocks of [`tile_blocks`] that walk still visits a whole tile per
//! position of the long axis.

use super::BlockFn;
use crate::inline::{InlineVec, OperandVec, OPERAND_CAP};
use crate::Result;

/// Largest tile, in elements, that is replayed from an offset table.
///
/// Sixteen covers the `4 x 4` blocks of the issue. A wider leading run is long
/// enough that one callback per run is already a small share of its elements.
pub(crate) const TILE_MAX: usize = 16;

/// Shortest block of the long axis the tile walk accepts.
///
/// Below it a callback per tile element moves fewer elements than the block
/// walk's callback per innermost run (at most [`TILE_MAX`] elements).
const MIN_RUN: usize = TILE_MAX;

/// Bytes one block of the long axis may span, summed over the operands:
/// three quarters of the 32 KiB L1 data cache
/// ([`BLOCK_MEMORY_SIZE`](crate::BLOCK_MEMORY_SIZE)).
///
/// The tile passes revisit every line of the block once per tile element, so
/// the block has to stay in L1, and at the strides of this shape (`n²`
/// elements, a power of two for `n = 2, 4`) the lines fall in a subset of the
/// sets. Measured on x86-64 (AMD EPYC 7702P) with a hand-written loop in the
/// same order, for `n = 2..4`, f64 and `Complex<f64>`: the time per element is
/// flat from a quarter to three quarters of L1 and rises sharply past it.
const TILE_BUDGET_BYTES: usize = crate::BLOCK_MEMORY_SIZE / 4 * 3;

/// Leading levels of `dims` whose extents multiply to at most [`TILE_MAX`],
/// with that product.
fn tile_levels(dims: &[usize]) -> (usize, usize) {
    let mut levels = 0usize;
    let mut elements = 1usize;
    while levels < dims.len() {
        match elements.checked_mul(dims[levels]) {
            Some(next) if next <= TILE_MAX => {
                elements = next;
                levels += 1;
            }
            _ => break,
        }
    }
    (levels, elements)
}

/// The tile levels of `dims` when it is exactly a small tile under one long
/// axis: every level but the last multiplies to at most [`TILE_MAX`]
/// elements, the last is at least [`MIN_RUN`] long, and no axis is empty.
fn tile_shape(dims: &[usize]) -> Option<(usize, usize)> {
    let rank = dims.len();
    if rank < 2 || dims.contains(&0) {
        return None;
    }
    let (levels, elements) = tile_levels(dims);
    (levels == rank - 1 && dims[levels] >= MIN_RUN).then_some((levels, elements))
}

/// Blocks for a compressed, ordered destination plan that is exactly a small
/// tile under one long axis, or `None` for any other plan.
///
/// The tile levels keep their full extent; the long axis is cut so that one
/// block spans at most [`TILE_BUDGET_BYTES`] over all operands, and never
/// below [`MIN_RUN`] positions.
pub(crate) fn tile_blocks(dims: &[usize], elem_sizes: &[usize]) -> Option<InlineVec<usize>> {
    let (levels, elements) = tile_shape(dims)?;
    let mut blocks = InlineVec::from(dims);
    let bytes_per_position = elem_sizes
        .iter()
        .try_fold(0usize, |sum, &size| sum.checked_add(size))
        .and_then(|sum| sum.checked_mul(elements));
    if let Some(per_position) = bytes_per_position.filter(|&bytes| bytes > 0) {
        blocks[levels] = (TILE_BUDGET_BYTES / per_position).clamp(MIN_RUN, dims[levels]);
    }
    Some(blocks)
}

/// The per-operand offsets of each element of a small tile.
pub(crate) struct SmallTile {
    /// Leading levels that form the tile: levels `0..levels`.
    levels: usize,
    /// Elements in the tile (`prod(dims[..levels])`).
    elements: usize,
    /// Operands.
    operands: usize,
    /// `table[t][k]`: offset of tile element `t` of operand `k` from the tile
    /// origin.
    table: [[isize; OPERAND_CAP]; TILE_MAX],
}

impl SmallTile {
    /// The tile of a pre-ordered plan, or `None` when the plan is not exactly
    /// a small tile under one long axis.
    ///
    /// Besides the shape, the tile levels must be unblocked
    /// (`blocks[i] >= dims[i]`, so one block of the long axis holds whole
    /// tiles), the long axis' block at least [`MIN_RUN`] long, and there must
    /// be at most [`OPERAND_CAP`] operands.
    pub(crate) fn of(dims: &[usize], blocks: &[usize], strides: &[&[isize]]) -> Option<Self> {
        let operands = strides.len();
        if operands == 0 || operands > OPERAND_CAP || blocks.len() != dims.len() {
            return None;
        }
        if strides.iter().any(|s| s.len() != dims.len()) {
            return None;
        }
        let (levels, elements) = tile_shape(dims)?;
        if (0..levels).any(|i| blocks[i] < dims[i]) || blocks[levels] < MIN_RUN {
            return None;
        }

        // Element `t` in column-major order over the tile levels. Every entry
        // is the displacement of a reachable element from the tile origin, so
        // the sums are representable (issue #243).
        let mut table = [[0isize; OPERAND_CAP]; TILE_MAX];
        for (t, row) in table.iter_mut().enumerate().take(elements) {
            let mut rest = t;
            for level in 0..levels {
                let index = (rest % dims[level]) as isize;
                rest /= dims[level];
                for (slot, s) in row.iter_mut().zip(strides) {
                    *slot += index * s[level];
                }
            }
        }
        Some(Self {
            levels,
            elements,
            operands,
            table,
        })
    }

    /// Walk `dims` with this tile: the long axis through the shared block
    /// walk, and per block of it one call of `f` per tile element over that
    /// block.
    #[inline]
    pub(crate) fn walk<F>(
        &self,
        dims: &[usize],
        blocks: &[usize],
        strides: &[&[isize]],
        initial_offsets: &[isize],
        f: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
    {
        let levels = self.levels;
        let operands = self.operands;
        let table = &self.table[..self.elements];
        let outer: OperandVec<&[isize]> = strides.iter().map(|s| &s[levels..]).collect();
        let mut along = |origin: &[isize], count: usize, step: &[isize]| -> Result<()> {
            let mut offsets = [0isize; OPERAND_CAP];
            for row in table {
                for k in 0..operands {
                    offsets[k] = origin[k] + row[k];
                }
                f(&offsets[..operands], count, step)?;
            }
            Ok(())
        };
        let along: BlockFn<'_> = &mut along;
        super::preordered_walk(
            &dims[levels..],
            &blocks[levels..],
            &outer,
            initial_offsets,
            along,
        )
    }
}

#[cfg(test)]
#[path = "tile/tests.rs"]
mod tests;
