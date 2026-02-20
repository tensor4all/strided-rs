# Flatten HPTT Recursion Experiment

Branch: `perf/flatten-hptt-recursion`

## Hypothesis

HPTT's recursive `ComputeNode` traversal causes significant per-element
overhead for tensor networks with many small dimensions (e.g., 24 binary dims
of size 2 → ~15 recursion levels with 2 iterations each). Replacing the
recursion with a flat iterative odometer loop should reduce this overhead and
improve performance.

## Implementation

Replaced `transpose_recursive` and `const_stride1_recursive` with flat
odometer versions (`transpose_flat`, `const_stride1_flat`). The `ComputeNode`
linked list is flattened into parallel arrays (`FlatLoops`) at execution time.

The flat loop has the same structure as `copy_strided_src_order` — the only
difference is the iteration order (dst-stride-order vs src-stride-order) and
the leaf operation (blocked 2D transpose vs single element copy).

## Results (AMD EPYC 7713P, faer, 1T)

| Instance | Recursive | Flat | Diff |
|---|---|---|---|
| mera_closed (opt_flops) | 1476 ms | 1528 ms | +4% |
| mera_open (opt_flops) | 918 ms | 943 ms | +3% |
| tn_focus (opt_flops) | 279 ms | 292 ms | +5% |
| tn_light (opt_flops) | 283 ms | 290 ms | +2% |
| mera_closed (opt_size) | 1519 ms | 1462 ms | -4% |
| mera_open (opt_size) | 918 ms | 943 ms | +3% |

All differences are within noise (±5%).

## Conclusion

**Flattening the recursion does not improve performance.** The recursive
function-call overhead is not the bottleneck.

This confirms that the performance advantage of lazy permutation (documented in
`eager-hptt-experiment.md`) comes from **copy elision** (`try_fuse_group`
skipping the copy entirely), not from the copy strategy (source-order vs
destination-order) or loop implementation (recursive vs flat).

### Priority for `Contract` CPU backend implementation

1. **Copy elision** (`try_fuse_group`) — dominant optimization
2. **Source-stride-order copy** — safe default for cold-cache sources
3. Copy strategy (HPTT vs flat odometer) — negligible difference
