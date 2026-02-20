# Eager HPTT Permutation Experiment

Branch: `perf/eager-hptt-permute`

## Motivation

strided-opteinsum uses **lazy permutation**: between contraction steps, only
metadata (dims/strides arrays) is reordered — no data copy. When a subsequent
step needs contiguous input, a source-stride-order copy is performed.

This experiment tests the alternative: **eager HPTT permutation** at every
permutation-only step, using the cache-efficient HPTT transpose from
`strided-perm`. The eager approach is simpler (no deferred permutation state)
but may do unnecessary work when `try_fuse_group` would have skipped the copy.

See [permutation-optimization.md](permutation-optimization.md) for background
on lazy permutation and source-order copy.

## Implementation

A feature flag `eager-permute` gates the behavior. When enabled,
permutation-only nodes in `eval_node` and the final output permutation in
`evaluate` call `materialize_permuted()` instead of the metadata-only
`permuted()`.

Modified files:
- `strided-opteinsum/src/operand.rs`: added `materialize_permuted` method
- `strided-opteinsum/src/expr.rs`: `#[cfg(feature = "eager-permute")]` at two
  permutation sites

All 82 tests pass with both `eager-permute` and `eager-permute,parallel`.

## Results (AMD EPYC 7713P, faer backend)

Median of 15 runs, 3 warmup. Positive diff = eager is slower.

### opt_flops

| Instance | Lazy 1T | Eager 1T | Lazy 4T | Eager 4T | 1T diff | 4T diff |
|---|---|---|---|---|---|---|
| mera_closed | 1476 ms | 1498 ms | 786 ms | 773 ms | +1% | -2% |
| **mera_open** | **918 ms** | **1199 ms** | **509 ms** | **598 ms** | **+31%** | **+17%** |
| tn_focus | 279 ms | 287 ms | 223 ms | 219 ms | +3% | -2% |
| tn_light | 283 ms | 287 ms | 222 ms | 218 ms | +1% | -2% |
| queen5_5 | 5542 ms | — | 5632 ms | 5684 ms | — | +1% |
| mps_200 | 16 ms | — | 42 ms | 46 ms | — | +10% |

### opt_size

| Instance | Lazy 1T | Eager 1T | Lazy 4T | Eager 4T | 1T diff | 4T diff |
|---|---|---|---|---|---|---|
| mera_closed | 1519 ms | 1439 ms | 697 ms | 689 ms | -5% | -1% |
| **mera_open** | **918 ms** | **1159 ms** | **502 ms** | **614 ms** | **+26%** | **+22%** |
| tn_focus | 283 ms | 284 ms | 220 ms | 232 ms | +0% | +5% |
| tn_light | 287 ms | 290 ms | 226 ms | 222 ms | +1% | -2% |

## Conclusion

- **`mera_open` is consistently and significantly worse with eager HPTT** —
  +26–31% at 1T, +17–22% at 4T. This is the clearest signal.
- All other instances are within noise (~±5%).
- **Lazy permutation wins overall**: eager HPTT never provides a meaningful
  benefit, but it clearly hurts `mera_open`.

The likely explanation: `mera_open` has many intermediate permutation-only
steps where `try_fuse_group` in the lazy path can skip the copy entirely (the
strides are already compatible for the next GEMM). Eager HPTT forces a full
data copy at every permutation step, paying the cost unnecessarily.

**Decision**: Keep lazy permutation as the default strategy. The `eager-permute`
feature flag remains available for future investigation.
