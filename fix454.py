import re
p = 'strided-fused/tests/common/mod.rs'
s = open(p, encoding='utf-8').read()

old_head = """pub(crate) fn count_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get).wrapping_sub(before)
}"""

new_head = """/// Modular delta between two readings of a counter that may have wrapped.
///
/// This is the only place a counter reading is turned into a delta. Keeping it
/// in one function means no call site can reintroduce a plain subtraction, and
/// `counter_delta_wraps_instead_of_underflowing` covers the wrap
/// deterministically, without depending on the harness making no allocation
/// between a saturation and a snapshot.
pub(crate) fn counter_delta(after: usize, before: usize) -> usize {
    after.wrapping_sub(before)
}

pub(crate) fn count_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    counter_delta(THREAD_ALLOCATIONS.with(Cell::get), before)
}"""

assert old_head in s
s = s.replace(old_head, new_head, 1)

old_net = """    let alloc_delta = after_alloc.wrapping_sub(before_alloc) as isize;
    let free_delta = after_free.wrapping_sub(before_free) as isize;"""
new_net = """    let alloc_delta = counter_delta(after_alloc, before_alloc) as isize;
    let free_delta = counter_delta(after_free, before_free) as isize;"""
assert old_net in s
s = s.replace(old_net, new_net, 1)

old_glob = """            GLOBAL_ALLOCATIONS
                .load(Ordering::SeqCst)
                .wrapping_sub(before)"""
new_glob = """            counter_delta(GLOBAL_ALLOCATIONS.load(Ordering::SeqCst), before)"""
assert old_glob in s
s = s.replace(old_glob, new_glob, 1)
open(p, 'w', encoding='utf-8').write(s)
print('common ok')
