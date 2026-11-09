p = 'strided-basic/src/fuse.rs'
s = open(p, encoding='utf-8').read()
old = """            // One bit per level still overflows from rank ~62 on: share the top
            // weight, with the widest level cap that fits.
            (1..=61u64)
                .rev()
                .find_map(|level_cap| weights(1, level_cap))"""
new = """            // One bit per level still overflows from rank ~62 on: share the top
            // weight, with the widest level cap that fits. The bound is `u64`,
            // not a fixed 61: the cap that fits depends on the shape (a cap of
            // 61 is forced only when the output stream is live, because its
            // weight is `4 * 2^cap`; with an all-zero output stride `63` fits),
            // so let the checked arithmetic choose per shape.
            (1..=63u64)
                .rev()
                .find_map(|level_cap| weights(1, level_cap))"""
assert old in s
open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))

p2 = 'strided-basic/src/fuse/tests/tests.rs'
t = open(p2, encoding='utf-8').read()
anchor = """#[test]
fn test_compute_importance_matches_exact_reference() {"""
new_test = """#[test]
fn test_compute_importance_rank_65_broadcast_output_keeps_live_axes_distinct() {
    // Review of #450: the fallback searched level caps only up to 61, so a
    // shape whose live axes are on the input side (zero-strided output) lost a
    // distinction that does fit: cap 63 gives 2^63 and 2^62 here, cap 61 gives
    // 2^61 and 2^61. The cap must be chosen per shape, not capped at 61.
    let n = 65;
    let mut dims = vec![1usize; n];
    dims[0] = 2;
    dims[2] = 2;
    let out_strides = vec![0isize; n];
    let in_strides: Vec<isize> = (0..n).map(|i| i as isize + 1).collect();
    let all_strides: [&[isize]; 2] = [&out_strides, &in_strides];
    let index_orders: Vec<Vec<usize>> = [&out_strides[..], &in_strides[..]]
        .iter()
        .map(|s| index_order(s).into_vec())
        .collect();

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    assert!(
        importance[2] > importance[0],
        "axis 2 (stride 3) must outrank axis 0 (stride 1): {:?}",
        &importance[..3]
    );
    assert!(importance[2..].iter().all(|&w| w == 0));
}

#[test]
fn test_compute_importance_matches_exact_reference() {"""
assert anchor in t
open(p2, 'w', encoding='utf-8').write(t.replace(anchor, new_test, 1))
print('ok')
