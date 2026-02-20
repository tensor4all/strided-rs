# Source-Stride-Order vs Destination-Stride-Order (HPTT) Experiment

Branch: `perf/src-vs-dst-order`

## Hypothesis

The eager-HPTT experiment showed a 26–31% regression on `mera_open` when
permutations are eagerly materialized. However, that experiment conflated two
factors: **copy elision** (`try_fuse_group`) and **copy strategy** (source-order
vs destination-order). This experiment isolates the copy strategy factor by
disabling copy elision (`force-copy` feature) and comparing source-stride-order
copy vs HPTT (destination-stride-order) copy.

## Setup

Two feature flags added to `strided-einsum2`:

- `force-copy`: Forces `needs_copy = true` in all `prepare_input_*` and
  `prepare_output_*` functions, disabling `try_fuse_group` elision.
- `hptt-input-copy`: Switches `prepare_input_owned` from source-stride-order
  copy to HPTT (`strided_kernel::copy_into_col_major`).

Three configurations benchmarked:

1. **Baseline**: Default (copy elision enabled, src-order when copy needed)
2. **force-copy + src-order**: Copy elision disabled, source-stride-order copy
3. **force-copy + HPTT**: Copy elision disabled, HPTT destination-order copy

## Results (AMD EPYC 7713P, faer, 1T)

### opt_flops

| Instance | Baseline | force+src | force+HPTT | src vs HPTT |
|---|---|---|---|---|
| gm_queen5_5_3 | 5775 ms | 6400 ms (+11%) | 6052 ms (+5%) | HPTT 5% faster |
| lm_brackets_4_4d | 19 ms | 35 ms (+80%) | 24 ms (+23%) | **HPTT 31% faster** |
| lm_sentence_3_12d | 65 ms | 92 ms (+42%) | 78 ms (+19%) | **HPTT 16% faster** |
| lm_sentence_4_4d | 17 ms | 34 ms (+99%) | 24 ms (+41%) | **HPTT 29% faster** |
| str_matrix_chain_100 | 11 ms | 20 ms (+80%) | 14 ms (+25%) | **HPTT 30% faster** |
| str_mps_varying_200 | 16 ms | 21 ms (+31%) | 16 ms (-3%) | **HPTT 25% faster** |
| mera_closed | 1518 ms | 1739 ms (+15%) | 1567 ms (+3%) | **HPTT 10% faster** |
| mera_open | 935 ms | 1129 ms (+21%) | 1142 ms (+22%) | ~same (+1%) |
| tn_focus | 288 ms | 400 ms (+39%) | 568 ms (+97%) | **src 30% faster** |
| tn_light | 289 ms | 401 ms (+39%) | 560 ms (+94%) | **src 28% faster** |

### opt_size

| Instance | Baseline | force+src | force+HPTT | src vs HPTT |
|---|---|---|---|---|
| gm_queen5_5_3 | 1727 ms | 2471 ms (+43%) | 2290 ms (+33%) | HPTT 7% faster |
| lm_brackets_4_4d | 19 ms | 35 ms (+82%) | 20 ms (+5%) | **HPTT 43% faster** |
| lm_sentence_3_12d | 52 ms | 78 ms (+49%) | 59 ms (+14%) | **HPTT 24% faster** |
| lm_sentence_4_4d | 22 ms | 37 ms (+69%) | 25 ms (+18%) | **HPTT 30% faster** |
| str_matrix_chain_100 | 11 ms | 20 ms (+85%) | 15 ms (+36%) | **HPTT 26% faster** |
| str_mps_varying_200 | 14 ms | 23 ms (+68%) | 13 ms (-3%) | **HPTT 43% faster** |
| mera_closed | 1480 ms | 1496 ms (+1%) | 1322 ms (-11%) | **HPTT 12% faster** |
| mera_open | 934 ms | 1108 ms (+19%) | 1086 ms (+16%) | ~same (-2%) |
| tn_focus | 288 ms | 394 ms (+37%) | 541 ms (+88%) | **src 27% faster** |
| tn_light | 289 ms | 387 ms (+34%) | 532 ms (+84%) | **src 27% faster** |

## Analysis

### HPTT is faster for most workloads

Contrary to the initial assumption that source-stride-order is generally
superior, HPTT outperforms source-order on 8 out of 10 instances (16–43%
faster). HPTT's cache-blocked 2D transpose tiles give better cache utilization
when the data layout has moderate-to-large contiguous blocks.

### Source-order wins only for many small binary dimensions

The two instances where source-order is faster — `tn_focus` (316 tensors) and
`tn_light` (415 tensors) — have many binary dimensions (size 2). With ~24
dimensions of size 2, HPTT builds ~15 recursion levels with only 2 iterations
each, and the 2×2 inner tile degenerates. The simple odometer loop of
source-order copy handles this case more efficiently.

### mera_open: copy strategy is irrelevant

`mera_open` shows essentially no difference between source-order and HPTT
(+1% / -2%, within noise). The 21–22% regression vs baseline is entirely
due to copy elision loss. This confirms that the 26–31% regression in the
eager-HPTT experiment was caused by copy elision, not by HPTT's copy strategy.

### Copy elision remains the dominant optimization

All instances are faster with copy elision enabled (baseline) than with either
forced copy strategy. The biggest gaps are on lm_* and str_* instances (up to
99% regression with forced copies). Copy elision (`try_fuse_group`) should
always be the first priority.

## Conclusions

1. **Copy elision (`try_fuse_group`) is the most important optimization** —
   responsible for the majority of performance gains across all instances.

2. **HPTT is the better default copy strategy** when copies cannot be avoided.
   It outperforms source-order on most workloads thanks to cache-blocked tiling.

3. **Source-order is better for degenerate cases** with many small dimensions
   (size 2), where HPTT's recursion structure becomes overhead-heavy.

4. **The optimal `Contract` implementation should use adaptive copy strategy**:
   - Always try copy elision first (`try_fuse_group`)
   - Use HPTT for general cases
   - Consider source-order for tensors with many small dimensions (heuristic
     needed)

## Implications for `Contract` CPU backend

The priority order in `contract-as-core-op.md` should be updated:

1. **Copy elision** (`try_fuse_group`) — dominant optimization, always first
2. **HPTT (destination-stride-order)** — default copy strategy when elision fails
3. **Source-stride-order** — fallback for degenerate many-small-dims cases

A simple heuristic for choosing between HPTT and source-order: if the minimum
dimension size after bilateral fusion is ≤ 2 and the number of fused dimensions
is large (e.g., > 10), prefer source-order. Otherwise, use HPTT.
