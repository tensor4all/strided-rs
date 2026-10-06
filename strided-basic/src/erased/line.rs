//! Shared single-axis line layout for the scan, arg-reduction and norm plans.
//!
//! Every plan in this family treats the source as independent *lines* along
//! one axis. The remaining (outer) axes are compiled once into a cursor, and
//! execution visits one *unit* per cursor position:
//!
//! * In **line mode** a unit is one line.
//! * In **panel mode** a unit is a block of up to [`PANEL`] adjacent lines
//!   whose elements are contiguous in the source (the leading outer axis has
//!   unit source stride). The kernels then loop over the axis outermost and
//!   over the block innermost, so the inner loop walks contiguous memory even
//!   when the line axis itself is strided.
//!
//! Panel mode is selected only when the line axis does not have unit source
//! stride, so a unit-stride line axis (the column-major `(d, ...)` feature
//! first layout) always runs the contiguous line kernels.
//!
//! Serial execution and every parallel worker range call the same per-unit
//! kernel, so a parallel chunk runs exactly the loop the serial path runs.

use super::{checked_reduce_reset, compress_reduce_outer_axes, ReduceOuterAxis, ReduceOuterCursor};
use crate::{ExecContext, Result, StridedError};
use core::ops::Range;

/// Maximum number of adjacent lines processed together in panel mode.
pub(super) const PANEL: usize = 64;

/// Compiled outer traversal of a single-axis plan.
#[derive(Clone, Debug)]
pub(super) struct LineLayout {
    /// Extent of the line axis.
    pub(super) axis_len: usize,
    /// Source stride along the line axis.
    pub(super) src_axis_stride: isize,
    /// Destination stride along the line axis (zero when the destination has
    /// no line axis, as for arg-reductions).
    pub(super) dest_axis_stride: isize,
    /// Destination stride between adjacent lines of a panel.
    pub(super) dest_lane_stride: isize,
    /// Compressed outer axes; in panel mode the first one is the panel axis.
    outer_axes: Vec<ReduceOuterAxis>,
    /// Number of units (cursor positions).
    units: usize,
    /// Extent of the panel axis, when panel mode is selected.
    panel_extent: Option<usize>,
    /// Number of source elements, the threading threshold domain.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    total: usize,
}

impl LineLayout {
    /// Compile the outer traversal.
    ///
    /// `dest_outer_strides` holds one destination stride per source axis other
    /// than `axis`, in source axis order. `panel_needs_unit_dest` requires the
    /// destination to be contiguous across a panel as well (the scan and norm
    /// kernels store a whole panel row per axis step).
    pub(super) fn compile(
        src_dims: &[usize],
        src_strides: &[isize],
        dest_outer_strides: &[isize],
        dest_axis_stride: isize,
        axis: usize,
        panel_needs_unit_dest: bool,
    ) -> Result<Self> {
        let rank = src_dims.len();
        if axis >= rank {
            return Err(StridedError::InvalidAxis { axis, rank });
        }
        debug_assert_eq!(dest_outer_strides.len() + 1, rank);
        let axis_len = src_dims[axis];
        let src_axis_stride = src_strides[axis];
        let total = src_dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(StridedError::OffsetOverflow)?;

        let mut outer: Vec<(usize, isize, isize)> = (0..rank)
            .filter(|&source_axis| source_axis != axis)
            .zip(dest_outer_strides)
            .map(|(source_axis, &dest_step)| {
                (src_dims[source_axis], src_strides[source_axis], dest_step)
            })
            .collect();
        let units_before_panel = outer
            .iter()
            .try_fold(1usize, |acc, &(extent, _, _)| acc.checked_mul(extent))
            .ok_or(StridedError::OffsetOverflow)?;
        // Extent-one axes never move the cursor; dropping them keeps the
        // cursor short and lets the panel axis be found.
        outer.retain(|&(extent, _, _)| extent != 1);
        // Visit outer axes in increasing source stride so consecutive units
        // touch nearby memory; broadcast (zero stride) axes go last. Lines are
        // independent, so the order never changes a result.
        outer.sort_by_key(|&(_, source_step, _)| match source_step.unsigned_abs() {
            0 => usize::MAX,
            step => step,
        });

        let mut panel_extent = None;
        if src_axis_stride != 1 && axis_len > 1 {
            if let Some(&(extent, source_step, dest_step)) = outer.first() {
                if source_step == 1 && (!panel_needs_unit_dest || dest_step == 1) && extent > 1 {
                    panel_extent = Some(extent);
                }
            }
        }

        let axes = outer
            .iter()
            .enumerate()
            .map(|(index, &(extent, source_step, dest_step))| {
                let (extent, source_step, dest_step) = match panel_extent {
                    Some(_) if index == 0 => {
                        let panel =
                            isize::try_from(PANEL).map_err(|_| StridedError::OffsetOverflow)?;
                        (
                            extent.div_ceil(PANEL),
                            source_step
                                .checked_mul(panel)
                                .ok_or(StridedError::OffsetOverflow)?,
                            dest_step
                                .checked_mul(panel)
                                .ok_or(StridedError::OffsetOverflow)?,
                        )
                    }
                    _ => (extent, source_step, dest_step),
                };
                Ok(ReduceOuterAxis {
                    extent,
                    source_step,
                    source_reset: checked_reduce_reset(extent, source_step)?,
                    dest_step,
                    dest_reset: checked_reduce_reset(extent, dest_step)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let dest_lane_stride = match panel_extent {
            Some(_) => outer[0].2,
            None => 0,
        };
        // The panel axis must stay first, so only the axes behind it fuse.
        let outer_axes = match panel_extent {
            Some(_) => {
                let mut axes = axes.into_iter();
                let mut fused = vec![axes.next().expect("panel axis exists")];
                fused.extend(compress_reduce_outer_axes(axes.collect())?);
                fused
            }
            None => compress_reduce_outer_axes(axes)?,
        };
        let units = if units_before_panel == 0 {
            0
        } else {
            outer_axes
                .iter()
                .try_fold(1usize, |acc, axis| acc.checked_mul(axis.extent))
                .ok_or(StridedError::OffsetOverflow)?
        };
        Ok(Self {
            axis_len,
            src_axis_stride,
            dest_axis_stride,
            dest_lane_stride,
            outer_axes,
            units,
            panel_extent,
            total,
        })
    }

    /// Whether any unit exists and the line axis is non-empty.
    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.units == 0 || self.axis_len == 0
    }

    /// Number of lines in the unit at the given panel-axis coordinate.
    #[inline]
    fn width(&self, leading_coord: usize) -> usize {
        match self.panel_extent {
            Some(extent) => (extent - leading_coord * PANEL).min(PANEL),
            None => 1,
        }
    }
}

/// Visits every unit of `layout`, serially or across workers.
///
/// `unit(source_offset, dest_offset, width)` is called once per unit with the
/// element offsets of the unit's first line; `width` is 1 in line mode.
///
/// # Safety
///
/// The caller guarantees that `unit` is safe to call for every offset pair
/// the compiled layout produces from the given bases (the plan validated the
/// source and destination descriptors against the layout), and that distinct
/// units write disjoint destination elements.
pub(super) unsafe fn for_each_unit<F>(
    ctx: &ExecContext,
    layout: &LineLayout,
    source_base: isize,
    dest_base: isize,
    unit: &F,
) -> Result<()>
where
    F: Fn(isize, isize, usize) + crate::MaybeSync,
{
    if layout.is_empty() {
        return Ok(());
    }
    if super::reduce_context_is_serial(ctx) {
        return run_units(layout, source_base, dest_base, 0..layout.units, unit);
    }
    ctx.run(|| {
        #[cfg(feature = "parallel")]
        {
            let nthreads =
                crate::threading::parallel_threads_for_len(layout.total).min(layout.units);
            if nthreads > 1 {
                return crate::threading::parallel_map_reduce(
                    0..layout.units,
                    nthreads,
                    &|range| run_units(layout, source_base, dest_base, range, unit),
                    &|left, right| left.and(right),
                );
            }
        }
        run_units(layout, source_base, dest_base, 0..layout.units, unit)
    })
}

/// Runs one contiguous range of units, decoding the cursor once.
fn run_units<F>(
    layout: &LineLayout,
    source_base: isize,
    dest_base: isize,
    range: Range<usize>,
    unit: &F,
) -> Result<()>
where
    F: Fn(isize, isize, usize),
{
    let end = range.end;
    let mut cursor =
        ReduceOuterCursor::decode(range.start, source_base, dest_base, &layout.outer_axes)?;
    for index in range {
        unit(
            cursor.source_offset,
            cursor.dest_offset,
            layout.width(cursor.leading_coord()),
        );
        if index + 1 < end {
            cursor.advance();
        }
    }
    Ok(())
}

/// Raw pointer that may be captured by a worker closure.
///
/// Workers only touch disjoint destination units and shared read-only
/// sources; the owning plan establishes that before dispatch.
#[derive(Clone, Copy, Debug)]
pub(super) struct UnitPtr<T>(pub(super) *mut T);

// SAFETY: see the type docs; the plan proves disjoint writes and shared reads.
unsafe impl<T> Send for UnitPtr<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for UnitPtr<T> {}

impl<T> UnitPtr<T> {
    #[inline(always)]
    pub(super) fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(all(test, feature = "parallel"))]
#[path = "line/tests/tests.rs"]
mod tests;
