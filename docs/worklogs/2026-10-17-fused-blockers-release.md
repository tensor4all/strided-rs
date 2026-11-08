# 2026-10-17: strided-basic 0.5.4 and strided-fused 0.5.2

| Crate | Published | This release | Library change |
|---|---|---|---|
| `strided-basic` | 0.5.3 | **0.5.4** | #313: whole-dimension blocks are halved before they reach the threaded mapreduce, so small tensors (for example a 1024x64 broadcast) split across workers instead of running one partition. This applies to map, zip and fused callers. |
| `strided-fused` | 0.5.1 | **0.5.2** | #313: the chunked replay accepts stride-0 (broadcast) inputs. Multi-output plans run in one replay through the additive `ErasedFusedPlan::execute_outputs` / `execute_outputs_uninit`. A rank-0 scalar no longer indexes empty strides. |
| other crates | unchanged | unchanged | none |

There are no breaking changes. The public additions are `ErasedFusedPlan::execute_outputs` and `execute_outputs_uninit`, and `ErasedFusedPlan::compile` now accepts plans with more than one output.

Workspace requirements become `strided-basic 0.5.4` and `strided-fused 0.5.2`, as caret ranges. The `strided-view` requirement stays at `0.5.0`, for the same reason as in the 0.5.3 release note.

Publish order: `strided-basic`, then `strided-fused`.

## Downstream

`tcontract 0.2.3` pins `strided-basic =0.5.3`, so tenferro-rs taking `strided-basic 0.5.4` needs `tcontract 0.2.4`, followed by `cpueinsum` / `cpueinsum-blas 0.2.4`. tenferro-rs then compiles one multi-output plan per region and re-runs its predeclared fusion rule (tenferro-rs#2161).

Evidence (fused / per-instruction, EPYC 7713P, 1T and 4T, f32 and f64): every row is 0.40–1.05 at 4T. The f64 1024x64 broadcast row, which was 16x slower than per-instruction on 0.5.1, is now 0.68–0.85. A two-output region, which ran the chain twice on 0.5.1, is now 0.50–0.94. See the PR for the full table.
