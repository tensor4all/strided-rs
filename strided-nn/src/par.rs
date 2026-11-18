//! Shared task splitting for the row kernels.

use strided_basic::execution::{tree_width, KernelClass, SendPtr};
use strided_basic::Exec;

use crate::softmax::fan_out;

/// Split `units` equal work units over the width the token's dispatch cost justifies for
/// `elements` of `class`, and call `body(first_unit, end_unit)` once per task.
///
/// The split depends on the token only, never on the data, and a unit is always handled
/// whole by one task, so per-unit results do not depend on the thread count.
pub(crate) fn for_unit_ranges<F: Fn(usize, usize) + Sync>(
    units: usize,
    elements: usize,
    class: KernelClass,
    exec: Exec<'_>,
    max_tasks: usize,
    body: F,
) {
    if units == 0 {
        return;
    }
    let width = tree_width(&exec, class, elements)
        .get()
        .min(max_tasks.max(1));
    let per_task = units.div_ceil(width.min(units));
    let tasks = units.div_ceil(per_task);
    let limit = core::num::NonZeroUsize::new(width).unwrap_or(core::num::NonZeroUsize::MIN);
    fan_out(tasks, exec.limit(limit), |task| {
        let start = task * per_task;
        body(start, (start + per_task).min(units));
    });
}

/// `len` elements of the buffer behind `ptr` starting at `start`.
///
/// # Safety
/// `start..start + len` must lie inside the live `&mut` buffer `ptr` was taken from, and no
/// other live reference or concurrent task may touch that span.
pub(crate) unsafe fn span_mut<'a, T>(ptr: SendPtr<T>, start: usize, len: usize) -> &'a mut [T] {
    // SAFETY: the caller guarantees the span is in bounds and exclusively owned.
    unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr().add(start), len) }
}
