# 2026-10-17: B4 compiled plans + fused inline state timing (strided-rs#306 items 5-6)

Scope: the 1T small-op instruction table for the B4 diff on this branch
(`wip/0.6-b4-plans`), measured with `perf stat -e instructions:u` on the shared
host (EPYC-class, `RAYON_NUM_THREADS=1`, f64, release). The base is the state
this branch started from after the Step 0 merge, `16d4411`; the head is the B4
commits `6edf084` + `a80c009`. The harness is a throwaway
`strided-kernel/examples/b4_small_ops.rs` (not kept) that runs one op in a
fixed-iteration loop; setup, plan compile and allocation are outside the
measured loop. Both revisions used the same source, with the fused plan built
through the base `Vec` fields and the head `FusedPlan::new`.

## 4x4x4, 200,000 iterations (instructions, user space)

| op | base | head | delta |
|---|---:|---:|---:|
| `map_into` | 299,161,688 | 300,361,716 | +0.4% |
| `zip_map2_into` | 354,362,844 | 355,564,133 | +0.3% |
| `add` | 157,361,721 | 158,762,543 | +0.9% |
| `reduce_axis` 0 | 461,161 | 463,623 | +0.5% |
| `reduce_axis` 1 | 460,675 | 462,983 | +0.5% |
| `reduce_axis` 2 | 462,665 | 463,300 | +0.1% |
| `ConcatenatePlan::execute` | 576,770,229 | 595,971,648 | +3.3% |
| fused 2-op | 570,962,693 | 586,766,866 | +2.8% |
| fused 10-op | 5,276,168,681 | 4,867,768,064 | -7.7% |

The absolute counts include a fixed process/setup offset; the per-call deltas
are `delta / iters` and the largest is the concatenate row at about +96
instructions per call.

## 32x32x32, 1,000 iterations

| op | base | head | delta |
|---|---:|---:|---:|
| `map_into` | 24,800,602 | 24,798,335 | -0.01% |
| `zip_map2_into` | 33,257,909 | 33,256,308 | -0.005% |
| `add` | 32,267,251 | 32,269,092 | +0.006% |
| `reduce_axis` 0 | 919,319 | 923,890 | +0.5% |
| `reduce_axis` 1 | 919,570 | 921,388 | +0.2% |
| `reduce_axis` 2 | 921,913 | 924,000 | +0.2% |
| `ConcatenatePlan::execute` | 208,802,163 | 212,913,987 | +2.0% |
| fused 2-op | 42,510,039 | 42,590,880 | +0.2% |
| fused 10-op | 967,425,999 | 964,847,856 | -0.3% |

Every row is within 5% and 32^3 is within noise, so the B4 allocation-free
inlining (inline plan fields, per-thread register scratch, inline replay
metadata) does not regress the small-op instruction counts. The concatenate
serial replay pays a small fixed cost for indexing the now-inline per-segment
collections; the 10-op fused chain improves because the interpreter register
file no longer grows a `Vec` and reuses one stack slice.
