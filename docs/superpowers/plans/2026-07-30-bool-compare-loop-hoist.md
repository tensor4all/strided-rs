# Boolean comparison loop-hoist plan

Issue: #179

## Problem

Downstream comparison closures match a runtime operation for every tensor
element. On the 33,554,432-element f64 public workload, this prevents the
single-threaded loop from reaching the throughput of a fixed comparison.

## Design

Add `CompareOp` and `compare_into` to the typed kernel surface.
`compare_into` selects one fixed comparison closure before entering the
existing `zip_map2_into` traversal. Shape validation, strided fallback,
threading policy, and thresholds remain owned by that traversal.
The new boundary rejects non-injective mutable layouts before traversal, using
the allocation-free injectivity validator, so bounded parallel execution
cannot assign overlapping writes to workers.

The initial API supports ordered scalar types through `PartialOrd`. Complex
equality remains on the downstream generic path. A SIMD mask-to-byte-store
kernel is explicitly deferred because loop hoisting is sufficient and does not
add unsafe code or an architecture-specific contract.

## Evidence

An uncommitted same-process probe compared the old runtime-match closure with
`compare_into` on 33,554,432 f64 elements, using 3 warmups and 15 samples:

| Threads | Runtime match | Loop hoisted | Change |
|---:|---:|---:|---:|
| 1 | 42.064 ms | 19.551 ms | -53.5% |
| 4 | 13.276 ms | 13.066 ms | -1.6% |

CPUs 60-63 were idle before the run. The four-thread result is already
bandwidth-limited, so no SIMD implementation is justified by this evidence.

The exact downstream tenferro public API row was then measured with the local
candidate wired into both owned-tensor and borrowed-view comparison paths.
The run used the same CPUs, 3 warmups, and 15 samples:

| Threads | Current tenferro | Local adoption | Change |
|---:|---:|---:|---:|
| 1 | 60.752 ms | 40.337 ms | -33.6% |
| 4 | 20.440 ms | 17.833 ms | -12.8% |

A repeated candidate run measured 41.607 ms at one thread and 17.908 ms at
four threads. The upstream PR adds only the fixed-operation entry point; the
temporary downstream wiring is not part of this repository and will land in a
separate tenferro adoption PR after the upstream commit is merged.

## Verification

- Differential tests for `Eq`, `Lt`, `Le`, `Gt`, and `Ge` across f32, f64,
  i32, i64, and bool.
- NaN unordered behavior.
- Noncontiguous reverse-stride traversal.
- Typed rejection of a large stride-zero destination before mutation under a
  bounded Rayon policy.
- Full workspace format and tests.
- Exact downstream `compare_lt` public API row measured at one and four
  threads before updating the git pin.
