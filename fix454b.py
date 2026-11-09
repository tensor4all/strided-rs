p = 'strided-fused/tests/serial_specialization_alloc.rs'
s = open(p, encoding='utf-8').read()

old_use = """use common::{
    count_allocations, count_net_allocations, record_allocation, record_deallocation,
    THREAD_ALLOCATIONS, THREAD_FREES,"""
new_use = """use common::{
    count_allocations, count_net_allocations, counter_delta, record_allocation,
    record_deallocation, THREAD_ALLOCATIONS, THREAD_FREES,"""
assert old_use in s, "use block"
s = s.replace(old_use, new_use, 1)

anchor = """/// The signed combination wraps as well."""
new_test = """/// The modular delta on its own, with no counter involved: a window that crosses
/// the wrap reports the small true delta. Deterministic by construction — no
/// allocation can land between the two readings, unlike the process-wide atomic
/// window in `global_counter_wrap.rs`, where an intervening allocation from
/// another thread turns the window into an ordinary increasing one.
#[test]
fn counter_delta_wraps_instead_of_underflowing() {
    assert_eq!(counter_delta(0, usize::MAX), 1);
    assert_eq!(counter_delta(1, usize::MAX), 2);
    assert_eq!(counter_delta(usize::MAX, 0), usize::MAX);
    assert_eq!(counter_delta(4, 4), 0);
}

/// The signed combination wraps as well."""
assert anchor in s
s = s.replace(anchor, new_test, 1)
open(p, 'w', encoding='utf-8').write(s)
print('ok')
