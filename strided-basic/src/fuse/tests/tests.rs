use super::*;
use strided_view::auxiliary::index_order;

#[test]
fn test_fuse_dims_contiguous() {
    // Two contiguous dimensions: [3, 4] with strides [1, 3] -> fused to [12, 1]
    let dims = [3, 4];
    let strides1 = [1isize, 3];
    let strides2 = [1isize, 3];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused.as_slice(), vec![12, 1]);
}

#[test]
fn test_fuse_dims_allows_broadcast_operand_across_fused_axes() {
    let dims = [16usize, 16, 64, 64];
    let out = [1isize, 16, 256, 16_384];
    let lhs = [1isize, 16, 0, 256];
    let rhs = [0isize, 0, 1, 64];
    let all_strides: Vec<&[isize]> = vec![&out, &lhs, &rhs];

    let fused = fuse_dims(&dims, &all_strides);

    assert_eq!(fused.as_slice(), vec![256, 1, 64, 64]);
}

#[test]
fn test_fuse_dims_non_contiguous() {
    // Non-contiguous: strides don't match
    let dims = [3, 4];
    let strides1 = [1isize, 10]; // Not contiguous (should be 3)
    let all_strides: Vec<&[isize]> = vec![&strides1];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused.as_slice(), vec![3, 4]); // No fusion
}

#[test]
fn test_fuse_dims_partial() {
    // 3D: first two fuse, third doesn't
    let dims = [2, 3, 4];
    let strides = [1isize, 2, 100]; // dims[0]*strides[0]=2=strides[1], but 6≠100
    let all_strides: Vec<&[isize]> = vec![&strides];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused.as_slice(), vec![6, 1, 4]); // Fused first two
}

#[test]
fn test_fuse_dims_multiple_arrays() {
    // Only fuse if ALL arrays are contiguous
    let dims = [3, 4];
    let strides1 = [1isize, 3]; // Contiguous
    let strides2 = [1isize, 10]; // Not contiguous
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused.as_slice(), vec![3, 4]); // No fusion because strides2 isn't contiguous
}

#[test]
fn test_compute_importance_2_arrays() {
    // Example with 2 arrays, dims [4, 5]
    let dims = [4usize, 5];
    let strides1 = [1isize, 4]; // Column-major output
    let strides2 = [5isize, 1]; // Row-major input
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let order1 = index_order(&strides1);
    let order2 = index_order(&strides2);
    let index_orders = vec![order1.into_vec(), order2.into_vec()];

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // With output weighted 2x, dimension 0 should have higher importance
    // since it has smaller stride in the output array
    assert!(importance[0] > importance[1]);
}

#[test]
fn test_sort_by_importance() {
    let importance = vec![100u64, 50, 200, 10];
    let perm = sort_by_importance(&importance);
    assert_eq!(perm.as_slice(), vec![2, 0, 1, 3]); // Indices sorted by descending importance
}

#[test]
fn test_compute_costs() {
    let strides1 = [1isize, 4, 0];
    let strides2 = [2isize, 1, 0];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let costs = compute_costs(&all_strides);
    // min strides: [1, 1, 0], transformed: [2, 2, 1]
    assert_eq!(costs.as_slice(), vec![2, 2, 1]);
}

#[test]
fn test_compute_importance_with_zero_stride() {
    // Zero stride (broadcast) still gets index_order = 1 for compatibility,
    // but it does not contribute to ordering importance.
    let dims = [4usize, 5];
    let strides1 = [0isize, 1]; // First dim is broadcast
    let all_strides: Vec<&[isize]> = vec![&strides1];

    let order1 = index_order(&strides1);
    // For stride 0: order = 1 (zero strides always get 1)
    // For stride 1: order = 1 (no non-zero stride < 1)
    assert_eq!(order1.as_slice(), &[1, 1]);

    let index_orders = vec![order1.into_vec()];
    let importance = compute_importance(&dims, &all_strides, &index_orders);

    assert_eq!(importance[0], 0);
    assert!(importance[1] > 0);
}

#[test]
fn test_compute_importance_size_one_dim() {
    // Size-1 dimensions get zero importance
    let dims = [4usize, 1, 5];
    let strides1 = [1isize, 4, 4];
    let all_strides: Vec<&[isize]> = vec![&strides1];

    let order1 = index_order(&strides1);
    let index_orders = vec![order1.into_vec()];
    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // Dimension 1 has size 1 -> importance = 0
    assert_eq!(importance[1], 0);
    // Other dimensions should have non-zero importance
    assert!(importance[0] > 0);
    assert!(importance[2] > 0);
}

#[test]
fn test_compute_importance_output_weight() {
    // Output (first array) is strongly weighted
    // With same strides, dimension with smaller stride in output wins
    let dims = [4usize, 5];
    let out_strides = [1isize, 4]; // Column-major output
    let in_strides = [5isize, 1]; // Row-major input
    let all_strides: Vec<&[isize]> = vec![&out_strides, &in_strides];

    let order_out = index_order(&out_strides); // [1, 2]
    let order_in = index_order(&in_strides); // [2, 1]
    let index_orders = vec![order_out.into_vec(), order_in.into_vec()];

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // Output weighting makes dimension 0 (smaller stride in output) win.
    assert!(importance[0] > importance[1]);
}

#[test]
fn test_compute_costs_owned_vecs() {
    // Ported from threading.rs: verify compute_costs works with Vec<Vec<isize>>
    let strides_list: Vec<Vec<isize>> = vec![vec![1, 0, 3], vec![2, 0, 4]];
    let costs = compute_costs(&strides_list);
    assert_eq!(costs.as_slice(), vec![2, 1, 6]);
}

#[test]
fn test_compute_costs_with_zero() {
    // Zero strides become cost 1, non-zero become 2*abs
    let strides1 = [0isize, 2, -3];
    let strides2 = [1isize, 0, 2];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let costs = compute_costs(&all_strides);
    // min abs: [0, 0, 2]
    // transform: [1, 1, 4]
    assert_eq!(costs.as_slice(), vec![1, 1, 4]);
}

// ---- compress_dims tests ----

/// Compare the compressed per-operand stride lists against expected slices.
fn assert_strides(cs: &[crate::inline::InlineVec<isize>], expected: &[&[isize]]) {
    assert_eq!(cs.len(), expected.len());
    for (actual, expected) in cs.iter().zip(expected) {
        assert_eq!(actual.as_slice(), *expected);
    }
}

#[test]
fn test_compress_dims_removes_fused() {
    let dims = vec![12usize, 1];
    let strides = vec![vec![1isize, 3]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd.as_slice(), vec![12]);
    assert_strides(&cs, &[&[1]]);
}

#[test]
fn test_compress_dims_removes_multiple() {
    let dims = vec![6usize, 1, 4];
    let strides = vec![vec![1isize, 2, 100]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd.as_slice(), vec![6, 4]);
    assert_strides(&cs, &[&[1, 100]]);
}

#[test]
fn test_compress_dims_no_removal() {
    let dims = vec![3usize, 4];
    let strides = vec![vec![1isize, 3]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd.as_slice(), vec![3, 4]);
    assert_strides(&cs, &[&[1, 3]]);
}

#[test]
fn test_compress_dims_all_ones() {
    let dims = vec![1usize, 1, 1];
    let strides = vec![vec![1isize, 1, 1]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd.as_slice(), vec![1]);
    assert_strides(&cs, &[&[1]]);
}

#[test]
fn test_compress_dims_multi_arrays() {
    let dims = vec![6usize, 1, 4];
    let strides = vec![vec![1isize, 6, 6], vec![4isize, 24, 1]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd.as_slice(), vec![6, 4]);
    assert_strides(&cs, &[&[1, 6], &[4, 1]]);
}

#[test]
fn test_compress_dims_single_dim() {
    let dims = vec![5usize];
    let strides = vec![vec![1isize]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd.as_slice(), vec![5]);
    assert_strides(&cs, &[&[1]]);
}

#[test]
fn test_compress_dims_single_dim_one() {
    let dims = vec![1usize];
    let strides = vec![vec![1isize]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd.as_slice(), vec![1]);
    assert_strides(&cs, &[&[1]]);
}

#[test]
fn test_compress_dims_empty() {
    let dims: Vec<usize> = vec![];
    let strides: Vec<Vec<isize>> = vec![];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert!(cd.is_empty());
    assert!(cs.is_empty());
}

#[test]
fn test_compute_importance_high_rank_levels_fit_in_u64() {
    // tensor4all/strided-rs#450: `g * (n - order)` reached 93 bits at rank 32
    // with 3 operands, so `1u64 << shift` panicked in debug and wrapped in
    // release. Weights must stay exact enough to keep the level order.
    let mut dims = vec![1usize; 32];
    dims[0] = 2048;
    let mut strides0 = vec![2048isize; 32];
    strides0[0] = 1;
    let strides = [strides0.clone(), strides0.clone(), strides0];
    let all_strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    let index_orders: Vec<Vec<usize>> = strides.iter().map(|s| index_order(s).into_vec()).collect();

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    assert!(importance[0] > 0);
    // Singleton axes carry no ordering information.
    assert!(importance[1..].iter().all(|&w| w == 0));
}

#[test]
fn test_compute_importance_high_rank_keeps_level_order() {
    // Column-major strides on 32 live axes with 3 operands: 98 bits naive (the
    // highest term is 2^97), so a
    // wrapping shift scrambles the level order instead of panicking (#450).
    let n = 32;
    let dims = vec![2usize; n];
    let strides: Vec<isize> = (0..n).map(|i| 1isize << i).collect();
    let all_strides: Vec<&[isize]> = (0..3).map(|_| strides.as_slice()).collect();
    let index_orders: Vec<Vec<usize>> = (0..3).map(|_| index_order(&strides).into_vec()).collect();

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // Importance drops with the stride order, so it must be strictly decreasing.
    for i in 1..n {
        assert!(
            importance[i - 1] > importance[i],
            "axis {} ({}) is not above axis {} ({})",
            i - 1,
            importance[i - 1],
            i,
            importance[i]
        );
    }
}

#[test]
fn test_compute_importance_keeps_the_original_width_when_it_fits() {
    // Rank 32, two operands, striding only in the input array: the original
    // arithmetic fits (2^62), so the per-level width must stay `g` and the
    // weight must stay exact. Narrowing here would reorder the plan (review of
    // #450: a width of 1 bit gives 2^31).
    let n = 32;
    let mut dims = vec![1usize; n];
    dims[0] = 2048;
    let out_strides = vec![0isize; n];
    let mut in_strides = vec![2048isize; n];
    in_strides[0] = 2;
    let all_strides: [&[isize]; 2] = [&out_strides, &in_strides];
    let index_orders: Vec<Vec<usize>> = [&out_strides[..], &in_strides[..]]
        .iter()
        .map(|s| index_order(s).into_vec())
        .collect();

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    assert_eq!(importance[0], 1u64 << 62);
    assert!(importance[1..].iter().all(|&w| w == 0));
}

#[test]
fn test_compute_importance_rank_64_shares_the_top_weight() {
    // Past rank ~61 even one bit per level overflows, so the top levels share a
    // weight: the plan still runs, and the two live axes come out equal.
    let n = 64;
    let mut dims = vec![1usize; n];
    dims[0] = 1024;
    dims[1] = 2;
    let mut strides = vec![4096isize; n];
    strides[0] = 4;
    strides[1] = 1;
    let all_strides: Vec<&[isize]> = (0..3).map(|_| strides.as_slice()).collect();
    let index_orders: Vec<Vec<usize>> = (0..3).map(|_| index_order(&strides).into_vec()).collect();

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // The widest cap that fits is 61: 4 * 2^61 from the output plus 2 * 2^61.
    assert_eq!(importance[0], 6u64 << 61);
    assert_eq!(importance[0], importance[1]);
    assert!(importance[2..].iter().all(|&w| w == 0));
}

#[test]
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

    assert_eq!(
        importance[0],
        1u64 << 63,
        "axis 0 (stride 1) takes the top level"
    );
    assert_eq!(
        importance[2],
        1u64 << 62,
        "axis 2 (stride 3) keeps the level below it"
    );
    assert!(importance
        .iter()
        .enumerate()
        .all(|(i, &w)| i == 0 || i == 2 || w == 0));
}

#[test]
fn test_compute_importance_matches_exact_reference() {
    // The original formula evaluated without overflow (u128). Wherever it fits,
    // the 64-bit weights must be identical, so the plan order is unchanged.
    //
    // The grid, for every rank `n = 1..=4` and operand count `m = 1..=4`:
    //   * every independent stride permutation of every operand (the two
    //     budget notes below say where `n = 4, m >= 3` drops to every pair),
    //   * every zero-stride mask (all `2^n` subsets) on the output, on each
    //     input in turn, and on every operand at once,
    //   * every size-1 variant with at most two singleton axes, including the
    //     non-adjacent `[2, 1, 2]` shape.
    fn reference(dims: &[usize], strides: &[&[isize]], orders: &[Vec<usize>]) -> Vec<u128> {
        let (n, m) = (dims.len(), strides.len());
        if n == 0 || m == 0 {
            return Vec::new();
        }
        let g = 64 - (m as u64 + 1).leading_zeros();
        let mut weights = vec![0u128; n];
        for i in 0..n {
            if strides[0][i] != 0 {
                let shift = g * (n - orders[0][i]) as u32;
                weights[i] = (1u128 << (g + 1)) * (1u128 << shift);
            }
        }
        for k in 1..m {
            for i in 0..n {
                if strides[k][i] != 0 {
                    weights[i] += 1u128 << (g * (n - orders[k][i]) as u32);
                }
            }
        }
        for i in 0..n {
            if dims[i] <= 1 {
                weights[i] = 0;
            }
        }
        weights
    }

    fn permutations(items: &[isize]) -> Vec<Vec<isize>> {
        fn rec(items: &mut [isize], pos: usize, out: &mut Vec<Vec<isize>>) {
            if pos == items.len() {
                out.push(items.to_vec());
                return;
            }
            for i in pos..items.len() {
                items.swap(pos, i);
                rec(items, pos + 1, out);
                items.swap(pos, i);
            }
        }
        let mut items = items.to_vec();
        let mut out = Vec::new();
        rec(&mut items, 0, &mut out);
        out
    }

    /// One operand stride tuple per combination: the operands at `crossed`
    /// range over every permutation of the power-of-two strides independently,
    /// the rest keep the column-major (identity) permutation.
    fn stride_tuples(perms: &[Vec<isize>], m: usize, crossed: &[usize]) -> Vec<Vec<Vec<isize>>> {
        let identity = perms[0].clone();
        let mut tuples = Vec::new();
        let mut pick = vec![0usize; crossed.len()];
        loop {
            let mut tuple = vec![identity.clone(); m];
            for (slot, &operand) in crossed.iter().enumerate() {
                tuple[operand] = perms[pick[slot]].clone();
            }
            tuples.push(tuple);
            let mut carry = true;
            for p in pick.iter_mut() {
                if !carry {
                    break;
                }
                *p += 1;
                if *p == perms.len() {
                    *p = 0;
                } else {
                    carry = false;
                }
            }
            if carry {
                break;
            }
        }
        tuples
    }

    fn check(dims: &[usize], strides: &[Vec<isize>]) {
        let all_strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
        let orders: Vec<Vec<usize>> = strides.iter().map(|s| index_order(s).into_vec()).collect();

        let got: Vec<u128> = compute_importance(dims, &all_strides, &orders)
            .as_slice()
            .iter()
            .map(|&w| w as u128)
            .collect();
        let want = reference(dims, &all_strides, &orders);
        assert_eq!(
            got.as_slice(),
            want.as_slice(),
            "dims {:?} strides {:?}",
            dims,
            strides
        );
    }

    let mut broadcast_cases = 0usize;

    for n in 1..=4usize {
        let powers: Vec<isize> = (0..n).map(|i| 1isize << i).collect();
        let perms = permutations(&powers);

        // Every size-1 variant: none, each single axis, and the non-adjacent
        // pair that makes `[2, 1, 2]` reachable.
        let mut dim_grids: Vec<Vec<usize>> = vec![vec![2usize; n]];
        for i in 0..n {
            let mut dims = vec![2usize; n];
            dims[i] = 1;
            dim_grids.push(dims);
        }
        if n >= 3 {
            let mut dims = vec![2usize; n];
            dims[0] = 1;
            dims[n - 1] = 1;
            dim_grids.push(dims);
        }

        for m in 1..=4usize {
            // Stride permutations for the size-1 sweep: every operand
            // independent while that stays within the test budget (n <= 3 any
            // m, n = 4 up to m = 2). The exhaustive `24^3 + 24^4 = 345 600`
            // grid at n = 4 costs ~4.6 s in a debug test build, so
            // `n = 4, m >= 3` crosses every pair of operands instead
            // (`24^2` per pair, the rest column-major).
            let independent: Vec<Vec<usize>> = if perms.len().pow(m as u32) <= 2000 {
                vec![(0..m).collect()]
            } else {
                (0..m)
                    .flat_map(|i| ((i + 1)..m).map(move |j| vec![i, j]))
                    .collect()
            };
            let mut tuples: Vec<Vec<Vec<isize>>> = Vec::new();
            for crossed in &independent {
                tuples.extend(stride_tuples(&perms, m, crossed));
            }

            // Every stride permutation against every size-1 variant.
            for dims in &dim_grids {
                for strides in &tuples {
                    check(dims, strides);
                }
            }

            // Every zero-stride mask (a broadcast axis) on every operand
            // position: the output, each input in turn, and all operands at
            // once. This sweep multiplies by `1 + (2^n - 1) * (m + 1)`, so it
            // keeps full permutation independence only up to `(n!)^m <= 600`
            // and permutes one operand at a time above that. The size-1
            // variants rotate across the grid, so every broadcast case is also
            // checked against several dims shapes.
            let broadcast_tuples: Vec<Vec<Vec<isize>>> = if perms.len().pow(m as u32) <= 600 {
                tuples.clone()
            } else {
                let mut t = Vec::new();
                for operand in 0..m {
                    t.extend(stride_tuples(&perms, m, &[operand]));
                }
                t
            };
            for strides in &broadcast_tuples {
                for zero_mask in 0..(1u32 << n) {
                    // Every mask on every position; a zero mask is the same
                    // tuple for every position, so it is checked once.
                    let receivers = if zero_mask == 0 { 1 } else { m + 1 };
                    for role in 0..receivers {
                        let mut strides = strides.clone();
                        for (k, operand) in strides.iter_mut().enumerate() {
                            if role == m || role == k {
                                for (i, s) in operand.iter_mut().enumerate() {
                                    if zero_mask & (1 << i) != 0 {
                                        *s = 0;
                                    }
                                }
                            }
                        }
                        let dims = &dim_grids[broadcast_cases % dim_grids.len()];
                        broadcast_cases += 1;
                        check(dims, &strides);
                    }
                }
            }
        }
    }
}
