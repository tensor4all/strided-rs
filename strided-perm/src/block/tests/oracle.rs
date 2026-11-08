//! Test-only exact model of the shared block planner, used by the block tests
//! of both `strided-basic` and `strided-perm`.
//!
//! [`model_blocks`] transliterates Strided.jl 2.6.1 `_computeblocks` and
//! `totalmemoryregion` (`src/mapreduce.jl:505-558`) over mathematical
//! integers, held in checked `u128` (any overflow aborts the test), with the
//! Rust conventions the planners keep: byte stride magnitudes `|s| * e`,
//! costs `1` for a zero minimum stride and `min(2 * min|s|, isize::MAX)`
//! otherwise, and no halving candidate once every block is 1. Stride orders
//! rank the exact byte magnitudes. It has no early exit, no reordering, and
//! shares no code with either planner.
//!
//! [`legacy_blocks`] replays the planner arithmetic before the exact contract,
//! with every conversion and operation checked; it is `None` exactly when one
//! of them was not representable.

const BUDGET: u128 = 32 * 1024;
const LINE: u128 = 64;

fn exact(value: Option<u128>) -> u128 {
    value.expect("block oracle: intermediate overflow")
}

/// The model's per-axis costs.
pub fn model_costs(strides: &[&[isize]]) -> Vec<u128> {
    (0..strides[0].len())
        .map(|i| {
            let min = strides
                .iter()
                .map(|s| s[i].unsigned_abs() as u128)
                .min()
                .unwrap();
            if min == 0 {
                1
            } else {
                (2 * min).min(isize::MAX as u128)
            }
        })
        .collect()
}

/// `1` for a zero magnitude, otherwise one more than the number of nonzero
/// magnitudes strictly below it.
fn rank(magnitudes: &[u128]) -> Vec<usize> {
    magnitudes
        .iter()
        .map(|&m| {
            if m == 0 {
                1
            } else {
                1 + magnitudes.iter().filter(|&&x| x != 0 && x < m).count()
            }
        })
        .collect()
}

fn region(dims: &[usize], magnitudes: &[Vec<u128>]) -> u128 {
    let mut total = 0u128;
    for strides in magnitudes {
        let mut contiguous = 0u128;
        let mut blocks = 1u128;
        for (&d, &s) in dims.iter().zip(strides) {
            if s < LINE {
                contiguous = exact(contiguous.checked_add(exact((d as u128 - 1).checked_mul(s))));
            } else {
                blocks = exact(blocks.checked_mul(d as u128));
            }
        }
        let lines = contiguous / LINE + 1;
        let operand = exact(exact(LINE.checked_mul(lines)).checked_mul(blocks));
        total = exact(total.checked_add(operand));
    }
    total
}

fn last_argmax(blocks: &[usize], costs: &[u128]) -> Option<usize> {
    let mut best = None;
    let mut best_score = 0u128;
    for (i, (&b, &c)) in blocks.iter().zip(costs).enumerate() {
        if b <= 1 {
            continue;
        }
        let score = exact((b as u128 - 1).checked_mul(c));
        if score >= best_score {
            best_score = score;
            best = Some(i);
        }
    }
    best
}

fn computeblocks(
    dims: &[usize],
    costs: &[u128],
    magnitudes: &[Vec<u128>],
    orders: &[Vec<usize>],
) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    if region(dims, magnitudes) <= BUDGET {
        return dims.to_vec();
    }
    let min_order = orders.iter().flat_map(|o| o.iter()).min().copied().unwrap();
    if orders.iter().all(|o| o[0] == min_order) {
        let tail_magnitudes: Vec<Vec<u128>> = magnitudes.iter().map(|m| m[1..].to_vec()).collect();
        let tail_orders: Vec<Vec<usize>> = orders.iter().map(|o| o[1..].to_vec()).collect();
        let mut out = vec![dims[0]];
        out.extend(computeblocks(
            &dims[1..],
            &costs[1..],
            &tail_magnitudes,
            &tail_orders,
        ));
        return out;
    }
    let min_stride = magnitudes
        .iter()
        .flat_map(|m| m.iter())
        .min()
        .copied()
        .unwrap();
    if min_stride > BUDGET {
        return vec![1; dims.len()];
    }
    let mut blocks = dims.to_vec();
    while region(&blocks, magnitudes) >= 2 * BUDGET {
        let Some(i) = last_argmax(&blocks, costs) else {
            break;
        };
        blocks[i] = blocks[i].div_ceil(2);
    }
    while region(&blocks, magnitudes) > BUDGET {
        let Some(i) = last_argmax(&blocks, costs) else {
            break;
        };
        blocks[i] -= 1;
    }
    blocks
}

/// The model's blocks for axes already in iteration order, each extent at
/// least 1.
pub fn model_blocks(dims: &[usize], strides: &[&[isize]], elem_size: usize) -> Vec<usize> {
    let magnitudes: Vec<Vec<u128>> = strides
        .iter()
        .map(|s| {
            s.iter()
                .map(|&x| exact((x.unsigned_abs() as u128).checked_mul(elem_size as u128)))
                .collect()
        })
        .collect();
    let orders: Vec<Vec<usize>> = magnitudes.iter().map(|m| rank(m)).collect();
    computeblocks(dims, &model_costs(strides), &magnitudes, &orders)
}

/// How the replayed legacy front end computed its costs.
#[derive(Clone, Copy)]
#[allow(dead_code)] // each crate's tests replay one of the two
pub enum LegacyCosts {
    /// `strided-basic`'s `compute_costs`: checked absolute value, saturating
    /// doubling. It never overflows.
    Saturating,
    /// `strided-perm`'s former `compute_costs`: `abs` and `*= 2`.
    Unchecked,
}

fn legacy_region(dims: &[usize], byte_strides: &[Vec<isize>]) -> Option<usize> {
    let mut total = 0usize;
    for strides in byte_strides {
        let mut contiguous = 0isize;
        let mut blocks = 1usize;
        for (&d, &s) in dims.iter().zip(strides) {
            let s_abs = s.unsigned_abs();
            if s_abs < 64 {
                let term = isize::try_from(d.saturating_sub(1))
                    .ok()?
                    .checked_mul(s_abs as isize)?;
                contiguous = contiguous.checked_add(term)?;
            } else {
                blocks = blocks.checked_mul(d)?;
            }
        }
        let lines = contiguous as usize / 64 + 1;
        total = total.checked_add(64usize.checked_mul(lines)?.checked_mul(blocks)?)?;
    }
    Some(total)
}

fn legacy_argmax(blocks: &[usize], costs: &[isize]) -> Option<Option<usize>> {
    let mut best = None;
    let mut best_score = 0isize;
    for (i, (&b, &c)) in blocks.iter().zip(costs).enumerate() {
        if b <= 1 {
            continue;
        }
        let score = (isize::try_from(b).ok()? - 1).checked_mul(c)?;
        if score >= best_score {
            best_score = score;
            best = Some(i);
        }
    }
    Some(best)
}

fn legacy_core(
    dims: &[usize],
    costs: &[isize],
    byte_strides: &[Vec<isize>],
    orders: &[Vec<usize>],
) -> Option<Vec<usize>> {
    if dims.is_empty() {
        return Some(Vec::new());
    }
    let budget = 32 * 1024usize;
    if legacy_region(dims, byte_strides)? <= budget {
        return Some(dims.to_vec());
    }
    let min_order = orders.iter().flat_map(|o| o.iter()).min().copied().unwrap();
    if orders.iter().all(|o| o[0] == min_order) {
        let tail_strides: Vec<Vec<isize>> = byte_strides.iter().map(|s| s[1..].to_vec()).collect();
        let tail_orders: Vec<Vec<usize>> = orders.iter().map(|o| o[1..].to_vec()).collect();
        let mut out = vec![dims[0]];
        out.extend(legacy_core(
            &dims[1..],
            &costs[1..],
            &tail_strides,
            &tail_orders,
        )?);
        return Some(out);
    }
    let min_stride = byte_strides
        .iter()
        .flat_map(|s| s.iter().map(|x| x.unsigned_abs()))
        .min()
        .unwrap();
    if min_stride > budget {
        return Some(vec![1; dims.len()]);
    }
    let mut blocks = dims.to_vec();
    while legacy_region(&blocks, byte_strides)? >= 2 * budget {
        let Some(i) = legacy_argmax(&blocks, costs)? else {
            break;
        };
        blocks[i] = blocks[i].div_ceil(2);
    }
    while legacy_region(&blocks, byte_strides)? > budget {
        let Some(i) = legacy_argmax(&blocks, costs)? else {
            break;
        };
        blocks[i] -= 1;
    }
    Some(blocks)
}

/// The legacy planner's blocks, or `None` when one of its operations
/// (`elem_size as isize`, the byte-stride products, the cost doubling in
/// [`LegacyCosts::Unchecked`], the region sums, products and `d - 1` casts,
/// the score casts and products) was not representable.
pub fn legacy_blocks(
    dims: &[usize],
    strides: &[&[isize]],
    elem_size: usize,
    costs: LegacyCosts,
) -> Option<Vec<usize>> {
    if dims.is_empty() {
        return Some(Vec::new());
    }
    let e = isize::try_from(elem_size).ok()?;
    let byte_strides: Vec<Vec<isize>> = strides
        .iter()
        .map(|s| s.iter().map(|&x| x.checked_mul(e)).collect::<Option<_>>())
        .collect::<Option<_>>()?;
    let orders: Vec<Vec<usize>> = byte_strides
        .iter()
        .map(|s| {
            rank(
                &s.iter()
                    .map(|x| x.unsigned_abs() as u128)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let mut legacy_costs = vec![isize::MAX; dims.len()];
    for s in strides {
        for (cost, &x) in legacy_costs.iter_mut().zip(s.iter()) {
            let abs = match costs {
                LegacyCosts::Saturating => x.checked_abs().unwrap_or(isize::MAX),
                LegacyCosts::Unchecked => x.checked_abs()?,
            };
            *cost = (*cost).min(abs);
        }
    }
    for cost in &mut legacy_costs {
        *cost = match (*cost, costs) {
            (0, _) => 1,
            (c, LegacyCosts::Saturating) => c.saturating_mul(2),
            (c, LegacyCosts::Unchecked) => c.checked_mul(2)?,
        };
    }
    legacy_core(dims, &legacy_costs, &byte_strides, &orders)
}

/// One sweep case: dims, per-operand strides, element size.
pub type Case = (Vec<usize>, Vec<Vec<isize>>, usize);

/// The deterministic sweep: ranks 1-5, extents {1, 2, 3, 5, 64, 1000},
/// strides {0, ±1, ±7, ±64, ±4096, ±2^40, isize::MIN, isize::MAX} in random,
/// tied, permuted and broadcast patterns, element sizes {0, 1, 8, 16}, and 2
/// or 3 operands.
pub fn sweep() -> Vec<Case> {
    fn signed(m: isize, next: &mut dyn FnMut(usize) -> usize) -> isize {
        if m == isize::MAX && next(2) == 0 {
            isize::MIN
        } else if next(2) == 0 {
            -m
        } else {
            m
        }
    }
    const EXTENTS: [usize; 6] = [1, 2, 3, 5, 64, 1000];
    const MAGNITUDES: [isize; 6] = [1, 7, 64, 4096, 1 << 40, isize::MAX];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let mut cases = Vec::new();
    for rank in 1..=5 {
        for elem_size in [0usize, 1, 8, 16] {
            for operands in [2usize, 3] {
                for case in 0..60 {
                    let dims: Vec<usize> = (0..rank).map(|_| EXTENTS[next(6)]).collect();
                    let mut strides = Vec::new();
                    for _ in 0..operands {
                        let s: Vec<isize> = match case % 4 {
                            // random, zeros included
                            0 => (0..rank)
                                .map(|_| match next(7) {
                                    6 => 0,
                                    k => signed(MAGNITUDES[k], &mut next),
                                })
                                .collect(),
                            // tied: one magnitude on every axis
                            1 => {
                                let m = MAGNITUDES[next(6)];
                                (0..rank).map(|_| signed(m, &mut next)).collect()
                            }
                            // permuted: distinct magnitudes in a shuffled order
                            2 => {
                                let mut pool: Vec<isize> = MAGNITUDES.to_vec();
                                (0..rank)
                                    .map(|_| {
                                        let m = pool.remove(next(pool.len()));
                                        signed(m, &mut next)
                                    })
                                    .collect()
                            }
                            // broadcast: zero on a random subset of axes
                            _ => (0..rank)
                                .map(|_| {
                                    if next(2) == 0 {
                                        0
                                    } else {
                                        signed(MAGNITUDES[next(6)], &mut next)
                                    }
                                })
                                .collect(),
                        };
                        strides.push(s);
                    }
                    cases.push((dims, strides, elem_size));
                }
            }
        }
    }
    cases
}
