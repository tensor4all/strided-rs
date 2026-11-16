//! Sorted-run reduction: destination rows that are already grouped by a
//! destination index ([#479]).
//!
//! `RunSumPlan` computes
//!
//! ```text
//! dest[r, ..] (=|+=) Σ_{j in ptr[r]..ptr[r+1]} src[perm[j], ..]
//! ```
//!
//! for a source `[E, ..]` and a destination `[R, ..]` over the CSR runs
//! `ptr[0..=R]` and the row permutation `perm`. It is the edge → node
//! accumulation of a sparse message-passing loop (`out[r] += message[e]` for
//! every edge `e` received by `r`), where `ScatterPlan`'s additive scatter
//! would be serial and order-sensitive.
//!
//! # Topology and replay
//!
//! [`RunIndex`] owns the grouped-row topology: a stable counting sort of an
//! unsorted destination list, or offsets a caller already has. Sorting is
//! **topology time**: it allocates, and `RunSumPlan::compile` allocates its
//! fused traversal, while `execute_into` / `execute_acc` allocate nothing as
//! long as the payload fuses to at most eight axes, which is where the
//! rank-specialized kernels take over. A payload of more than eight fused axes
//! walks through `kernel_nd_inner_iterative`, whose cursor is a `Vec` by
//! contract. One `RunIndex` can be replayed by several plans until the topology
//! changes.
//!
//! # Determinism
//!
//! A run is summed by exactly one worker, in increasing `j`, and destination
//! rows are disjoint between runs (`compile` requires an injective destination
//! layout). The partition therefore cannot change a result: every `Exec` budget
//! produces identical bits. No atomics, no partial sums.
//!
//! # Contract
//!
//! - `execute_into` writes the sum; a run with no rows writes zeros.
//! - `execute_acc` adds the sum to the existing value; a run with no rows adds
//!   nothing.
//! - Both seed the accumulation with `T::zero()` (or the existing value) and add
//!   the rows in order, so `execute_into` and `execute_acc` agree bit for bit on
//!   a destination seeded with `T::zero()`.
//! - `dest` and `src` must not overlap; this is the caller's obligation, as for
//!   every plan taking `RawStridedRef` / `RawStridedMut`.
//!
//! # Not in this plan
//!
//! This module is the typed side only: the dtype-erased replay does not exist, and
//! no consumer of [#479] needs one. `strided-segpoly` reads the same `RunIndex`
//! for its scattered outputs and runs its own payload walk, so a third copy of
//! this plan that erases `T` would add a code path nothing calls.
//!
//! The payload walk is not blocked or SIMD-dispatched, and a single long run is
//! one worker's work.
//!
//! [#479]: https://github.com/tensor4all/strided-rs/issues/479

use core::ops::{Add, Range};

use num_traits::Zero;

use crate::cost::{tree_width, KernelClass};
use crate::fuse::{compress_dims, fuse_dims};
use crate::gather_plan::GatherIndex;
use crate::inline::{InlineVec, OperandVec};
use crate::threading::{for_each_inner_block_with_offsets, SendPtr};
use crate::{Exec, RawStridedMut, RawStridedRef, Result, StridedError};

/// Rows grouped by destination: CSR offsets plus a stable row permutation.
///
/// `ptr` holds `runs + 1` offsets and `perm` is either the identity (rows are
/// already grouped, so the plan walks `0..rows`) or as long as the row count.
#[derive(Debug, Clone)]
pub struct RunIndex {
    ptr: Vec<usize>,
    perm: Option<Vec<usize>>,
}

impl RunIndex {
    /// Group an unsorted destination list with a stable counting sort.
    ///
    /// Row `e` joins run `dest[e]`, and rows keep their original order inside a
    /// run. A destination that is already non-decreasing keeps the identity
    /// permutation and only pays for the offsets. Allocates; call it when the
    /// topology changes, never per execution.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::IndexOutOfRange`] when an index is negative, is
    /// not below `num_runs`, or is present while `num_runs` is zero, and
    /// [`StridedError::OffsetOverflow`] when the row count does not fit `usize`.
    pub fn from_dest<I: GatherIndex>(dest: &[I], num_runs: usize) -> Result<Self> {
        // The counting sort allocates one `usize` per run, so a run count whose
        // byte size is not representable is `OffsetOverflow` here rather than a
        // capacity panic in the allocation below.
        if num_runs > isize::MAX as usize / core::mem::size_of::<usize>() {
            return Err(StridedError::OffsetOverflow);
        }
        let counts = counts_of(dest, num_runs)?;

        // Prefix sums become the offsets; the counts are reused as the
        // placement cursor when a permutation is needed.
        let mut ptr: Vec<usize> = Vec::with_capacity(num_runs + 1);
        let mut grouped = 0usize;
        ptr.push(0);
        for &count in counts.iter() {
            grouped = grouped
                .checked_add(count)
                .ok_or(StridedError::OffsetOverflow)?;
            ptr.push(grouped);
        }
        if is_grouped(dest) {
            return Ok(Self { ptr, perm: None });
        }

        let mut cursor = counts;
        let mut start = 0usize;
        for slot in cursor.iter_mut() {
            let count = *slot;
            *slot = start;
            start += count;
        }
        let mut perm = vec![0usize; dest.len()];
        for (row, &index) in dest.iter().enumerate() {
            let run = run_of(index, num_runs)?;
            let slot = cursor[run];
            perm[slot] = row;
            cursor[run] = slot + 1;
        }
        Ok(Self {
            ptr,
            perm: Some(perm),
        })
    }

    /// Take rows a caller has already grouped.
    ///
    /// `ptr` is the CSR offset list: `runs + 1` non-decreasing entries starting
    /// at zero, so `ptr[r]..ptr[r + 1]` is the row range of run `r` and
    /// `ptr[last]` is the row count. Holds the `Vec` as given and allocates
    /// nothing.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::InvalidRunIndex`] when `ptr` is empty, does not
    /// start at zero, or steps downwards.
    pub fn from_ptr(ptr: Vec<usize>) -> Result<Self> {
        if ptr.first() != Some(&0) || ptr.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(StridedError::InvalidRunIndex);
        }
        Ok(Self { ptr, perm: None })
    }

    /// The CSR offsets: `runs + 1` non-decreasing entries starting at zero.
    pub fn ptr(&self) -> &[usize] {
        &self.ptr
    }

    /// The row permutation, or `None` when the rows are already in run order.
    pub fn perm(&self) -> Option<&[usize]> {
        self.perm.as_deref()
    }

    fn rows(&self) -> usize {
        self.ptr[self.ptr.len() - 1]
    }

    fn runs(&self) -> usize {
        self.ptr.len() - 1
    }
}

/// Count the rows of each run and reject an index outside `0..num_runs`.
fn counts_of<I: GatherIndex>(dest: &[I], num_runs: usize) -> Result<Vec<usize>> {
    let mut counts = vec![0usize; num_runs];
    for &index in dest {
        let run = run_of(index, num_runs)?;
        counts[run] += 1;
    }
    Ok(counts)
}

fn run_of<I: GatherIndex>(index: I, num_runs: usize) -> Result<usize> {
    let value = index.to_i64();
    if value < 0 || value as u64 >= num_runs as u64 {
        return Err(StridedError::IndexOutOfRange {
            index: value,
            runs: num_runs,
        });
    }
    Ok(value as usize)
}

fn is_grouped<I: GatherIndex>(dest: &[I]) -> bool {
    dest.windows(2)
        .all(|pair| pair[0].to_i64() <= pair[1].to_i64())
}

/// `dest[r, ..] (=|+=) Σ_{j in ptr[r]..ptr[r+1]} src[perm[j], ..]`.
///
/// Compiled once for a source layout, a destination layout and a
/// [`RunIndex`]'s row count; replayed with [`execute_into`](Self::execute_into)
/// or [`execute_acc`](Self::execute_acc).
#[derive(Debug)]
pub struct RunSumPlan {
    src_dims: InlineVec<usize>,
    src_strides: InlineVec<isize>,
    dest_dims: InlineVec<usize>,
    dest_strides: InlineVec<isize>,
    /// Payload traversal, fused in declaration order: `[dest, src]` strides.
    payload_dims: InlineVec<usize>,
    payload_strides: OperandVec<InlineVec<isize>>,
    src_row_stride: isize,
    dest_row_stride: isize,
    rows: usize,
    runs: usize,
    payload_elems: usize,
}

impl RunSumPlan {
    /// Compile a replay for one source layout, one destination layout and the
    /// run count the index describes.
    ///
    /// The layout facts a plan can prove without the buffers are proved here;
    /// buffer bounds, alignment and the absence of `dest`/`src` overlap are the
    /// caller's `RawStridedRef` / `RawStridedMut` obligations. Allocates the
    /// fused payload traversal; never called from the execute path.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when a stride list does not
    /// match its dimension list, [`StridedError::RankMismatch`] when the ranks
    /// differ or there is no row axis, [`StridedError::ShapeMismatch`] when the
    /// trailing extents differ or the row counts disagree with `runs`,
    /// [`StridedError::OffsetOverflow`] when an element count does not fit
    /// `usize`, and [`StridedError::NonInjectiveOutputLayout`] when the
    /// destination layout cannot be proven injective.
    pub fn compile(
        src_dims: &[usize],
        src_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        runs: &RunIndex,
    ) -> Result<Self> {
        if src_dims.len() != src_strides.len() || dest_dims.len() != dest_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        if src_dims.len() != dest_dims.len() {
            return Err(StridedError::RankMismatch(src_dims.len(), dest_dims.len()));
        }
        // The row axis is the one this plan exists for.
        if src_dims.is_empty() {
            return Err(StridedError::RankMismatch(0, 1));
        }
        if src_dims[0] != runs.rows() {
            return Err(StridedError::ShapeMismatch(
                vec![src_dims[0]],
                vec![runs.rows()],
            ));
        }
        if dest_dims[0] != runs.runs() {
            return Err(StridedError::ShapeMismatch(
                vec![dest_dims[0]],
                vec![runs.runs()],
            ));
        }
        if src_dims[1..] != dest_dims[1..] {
            return Err(StridedError::ShapeMismatch(
                src_dims.to_vec(),
                dest_dims.to_vec(),
            ));
        }
        crate::kernel::total_len(src_dims)?;
        crate::kernel::total_len(dest_dims)?;
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }

        let payload_dims = &src_dims[1..];
        let payload_strides = [&dest_strides[1..], &src_strides[1..]];
        let fused = fuse_dims(payload_dims, &payload_strides);
        let (fused_dims, fused_strides) = compress_dims(&fused, &payload_strides);
        let payload_elems = crate::kernel::total_len(&fused_dims)?;

        Ok(Self {
            src_dims: src_dims.into(),
            src_strides: src_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            payload_dims: fused_dims,
            payload_strides: fused_strides,
            src_row_stride: src_strides[0],
            dest_row_stride: dest_strides[0],
            rows: runs.rows(),
            runs: runs.runs(),
            payload_elems,
        })
    }

    /// Execute the overwriting form: `dest[r, ..] = Σ` over the run, zeros for a
    /// run with no rows. Never reads the previous destination value.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when a view does not carry
    /// its compiled extents and strides, and [`StridedError::ShapeMismatch`]
    /// when `runs` describes a different row or run count than `compile` was
    /// given.
    pub fn execute_into<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        runs: &RunIndex,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Zero + Add<Output = T> + Send + Sync,
    {
        self.execute(dest, src, runs, true, exec)
    }

    /// Execute the accumulating form: `dest[r, ..] += Σ` over the run, nothing
    /// for a run with no rows.
    ///
    /// # Errors
    ///
    /// As [`execute_into`](Self::execute_into).
    pub fn execute_acc<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        runs: &RunIndex,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Zero + Add<Output = T> + Send + Sync,
    {
        self.execute(dest, src, runs, false, exec)
    }

    fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        runs: &RunIndex,
        into: bool,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Zero + Add<Output = T> + Send + Sync,
    {
        self.check_call(dest, src, runs)?;
        if self.runs == 0 || self.payload_elems == 0 {
            return Ok(());
        }
        // `_into` and `_acc` both pay the destination's writes.
        let elements = self
            .rows
            .saturating_add(self.runs)
            .saturating_mul(self.payload_elems);
        let width = tree_width(&exec, KernelClass::Reduce, elements);
        if self.rows == 0 {
            // No source element is reachable, so an empty source's payload
            // strides carry no bound (`RawStridedRef` validates the reach of the
            // whole layout, which a zero row extent makes empty). Walking them
            // would overflow the odometer, so the source is not walked at all:
            // `_into` zeroes the destination rows, `_acc` adds nothing.
            if into {
                self.zero_destination(dest)?;
            }
            return Ok(());
        }
        let leaf = Leaf {
            plan: self,
            // `as_mut_ptr`/`ptr` already fold each layout's base offset in.
            dest: SendPtr(dest.as_mut_ptr()),
            src: SendPtr(src.ptr() as *mut T),
            into,
            ptr: runs.ptr(),
            perm: runs.perm(),
        };
        let body = |range: Range<usize>| {
            for run in range {
                // SAFETY: `check_call` matched the compiled layouts,
                // `RawStridedMut`/`RawStridedRef` validated the reach of those
                // layouts, no extent is zero (returned above), the destination
                // layout is injective so its rows are disjoint, and every run
                // belongs to exactly one leaf.
                unsafe { leaf.run(run) };
            }
        };
        self.dispatch(runs.ptr(), 0..self.runs, exec.limit(width), &body);
        Ok(())
    }

    /// `dest[r, ..] = 0` for every run, walking the destination alone.
    ///
    /// Only reached when the source has no rows, so the fused payload axes and
    /// their destination strides are the whole traversal.
    fn zero_destination<T: Copy + Zero>(&self, dest: &mut RawStridedMut<'_, T>) -> Result<()> {
        let base = SendPtr(dest.as_mut_ptr());
        let mut apply = |offsets: &[isize], len: usize, inner: &[isize]| -> Result<()> {
            let dest_inner = inner.first().copied().unwrap_or(0);
            for run in 0..self.runs {
                let block = run as isize * self.dest_row_stride + offsets[0];
                // SAFETY: `check_call` matched the compiled layout,
                // `RawStridedMut` validated the reach of it, and the destination
                // layout is injective, so this block is the run's own slots.
                unsafe { zero_block(base.as_ptr(), block, len, dest_inner) };
            }
            Ok(())
        };
        for_each_inner_block_with_offsets(
            &self.payload_dims,
            &self.payload_dims,
            &self.payload_strides[..1],
            &[0],
            &mut apply,
        )
    }

    fn check_call<T>(
        &self,
        dest: &RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        runs: &RunIndex,
    ) -> Result<()> {
        if dest.dims() != &self.dest_dims[..]
            || dest.strides() != &self.dest_strides[..]
            || src.dims() != &self.src_dims[..]
            || src.strides() != &self.src_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        if runs.rows() != self.rows || runs.runs() != self.runs {
            return Err(StridedError::ShapeMismatch(
                vec![runs.rows(), runs.runs()],
                vec![self.rows, self.runs],
            ));
        }
        Ok(())
    }

    /// Halve the run range at its row-count midpoint until the token's budget
    /// is spent, one run per leaf at the leaves.
    ///
    /// Splitting only between runs is what keeps the result independent of the
    /// budget: no run is ever assembled from two leaves.
    fn dispatch(
        &self,
        ptr: &[usize],
        runs: Range<usize>,
        exec: Exec<'_>,
        body: &(impl Fn(Range<usize>) + Sync),
    ) {
        if exec.budget().get() <= 1 || runs.len() <= 1 {
            body(runs);
            return;
        }
        let middle = row_midpoint(ptr, &runs);
        exec.fork(2, |job, child| {
            let half = if job == 0 {
                runs.start..middle
            } else {
                middle..runs.end
            };
            self.dispatch(ptr, half, child, body);
        });
    }
}

/// The run index that halves `runs`' row count, strictly inside the range so
/// the split always makes progress.
fn row_midpoint(ptr: &[usize], runs: &Range<usize>) -> usize {
    let target = ptr[runs.start] + (ptr[runs.end] - ptr[runs.start]) / 2;
    // `ptr` is non-decreasing (both constructors enforce it), so the first run
    // starting at or after the midpoint is the split; the clamp keeps it inside
    // the range even when several runs share a start.
    ptr.partition_point(|&start| start < target)
        .clamp(runs.start + 1, runs.end - 1)
}

/// One execution's leaf data: the compiled plan, the two bases, and the shared
/// topology.
struct Leaf<'a, 'plan, T> {
    plan: &'plan RunSumPlan,
    dest: SendPtr<T>,
    src: SendPtr<T>,
    into: bool,
    ptr: &'a [usize],
    perm: Option<&'a [usize]>,
}

impl<T> Leaf<'_, '_, T>
where
    T: Copy + Zero + Add<Output = T>,
{
    /// Sum one run into its destination row.
    ///
    /// # Safety
    ///
    /// The caller proved the buffers and both layouts, no extent is zero, and
    /// this run's destination row is not touched by another worker.
    unsafe fn run(&self, run: usize) {
        let plan = self.plan;
        let start = self.ptr[run];
        let end = self.ptr[run + 1];
        let dest_base = run as isize * plan.dest_row_stride;
        let mut apply = |offsets: &[isize], len: usize, inner: &[isize]| -> Result<()> {
            let dest_block = dest_base + offsets[0];
            let src_block = offsets[1];
            // A rank-0 payload reaches this once with `len == 1` and no strides.
            let dest_inner = inner.first().copied().unwrap_or(0);
            let src_inner = inner.get(1).copied().unwrap_or(0);
            // SAFETY: the walk visits the payload inside the validated layouts,
            // and the run's blocks are inside this leaf's destination row.
            unsafe {
                if self.into {
                    zero_block(self.dest.as_ptr(), dest_block, len, dest_inner);
                }
                for row in start..end {
                    let source_row = match self.perm {
                        Some(perm) => perm[row],
                        None => row,
                    };
                    add_block(
                        self.dest.as_ptr(),
                        dest_block,
                        self.src.as_const(),
                        src_block + source_row as isize * plan.src_row_stride,
                        len,
                        dest_inner,
                        src_inner,
                    );
                }
            }
            Ok(())
        };
        let walk = for_each_inner_block_with_offsets(
            &plan.payload_dims,
            &plan.payload_dims,
            &plan.payload_strides,
            &[0, 0],
            &mut apply,
        );
        // The walk propagates the leaf callback's own result, and this leaf
        // never fails; a failure would be a defect in the walk itself.
        debug_assert!(walk.is_ok(), "the sorted-run leaf is infallible");
    }
}

/// Write `T::zero()` over the block `len` runs of `dest_inner`.
///
/// # Safety
///
/// Every `dest.add(block + index * dest_inner)` for `index < len` must be a
/// writable, `T`-aligned slot this call owns.
unsafe fn zero_block<T: Copy + Zero>(dest: *mut T, block: isize, len: usize, dest_inner: isize) {
    let base = dest.offset(block);
    if dest_inner == 1 {
        // SAFETY: a unit inner stride makes the block one contiguous run the
        // caller proved writable.
        let out = unsafe { core::slice::from_raw_parts_mut(base, len) };
        for slot in out {
            *slot = T::zero();
        }
        return;
    }
    for index in 0..len {
        // Address arithmetic from the base, never past the final slot: the last
        // write is at `(len - 1) * dest_inner`.
        let slot = unsafe { base.offset(index as isize * dest_inner) };
        // SAFETY: as above, one strided slot at a time.
        unsafe { slot.write(T::zero()) };
    }
}

/// `dest_block += src_block` over the block `len` runs of the inner strides.
///
/// # Safety
///
/// Both blocks must be `len` reachable, `T`-aligned slots of their own
/// allocation; the destination slots must be initialized and exclusively owned
/// by this call.
unsafe fn add_block<T: Copy + Add<Output = T>>(
    dest: *mut T,
    dest_block: isize,
    src: *const T,
    src_block: isize,
    len: usize,
    dest_inner: isize,
    src_inner: isize,
) {
    if dest_inner == 1 && src_inner == 1 {
        // SAFETY: both unit-stride blocks are contiguous runs inside their
        // validated layouts.
        let out = unsafe { core::slice::from_raw_parts_mut(dest.offset(dest_block), len) };
        let input = unsafe { core::slice::from_raw_parts(src.offset(src_block), len) };
        for (slot, &value) in out.iter_mut().zip(input.iter()) {
            *slot = *slot + value;
        }
        return;
    }
    for index in 0..len {
        // Address arithmetic from the base, never past the final slot: the last
        // element is at `(len - 1) * inner`.
        let step = index as isize;
        let destination = unsafe { dest.offset(dest_block + step * dest_inner) };
        let source = unsafe { src.offset(src_block + step * src_inner) };
        // SAFETY: as above, one strided element at a time.
        unsafe { *destination = *destination + *source };
    }
}

#[cfg(test)]
#[path = "run_sum/tests/tests.rs"]
mod tests;

// Not under miri: `threading::test_pool` reads the host topology through the
// filesystem, which miri's isolation refuses. The serial tests in `run_sum/tests`
// are the miri surface for this module.
#[cfg(all(test, not(miri)))]
#[path = "run_sum/tests/parallel_tests.rs"]
mod parallel_tests;
