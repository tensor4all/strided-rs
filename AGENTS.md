# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before acting, read the latest shared tensor4all agent rules from the
[`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
repository. Start from:

- `https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`

If internet access is unavailable or the remote cannot be resolved, use the
sibling checkout:

- `../tensor4all-agent-rules/rules/index.md`

Load only the common, Rust, performance, numerical, docs, or benchmark rule
files relevant to the task. In particular, `rules/common/provenance.md`
applies whenever code is written while referencing third-party code.

Then read `REPOSITORY_RULES.md`, which holds the durable repository-specific
rules (public surface discipline, unsafe and fast-path boundaries,
performance and benchmark discipline, layout and copy semantics).

## Project Overview

strided-rs is a Rust workspace providing dynamic-rank strided views and
cache-optimized kernels for strided multidimensional array operations. It is
the CPU foundation used by
[tenferro-rs](https://github.com/tensor4all/tenferro-rs). The view and kernel
layers are ports of Julia's Strided.jl / StridedViews.jl; the permutation
engine follows HPTT. See `docs/PROVENANCE_AND_CITATION_POLICY.md` for the
full per-crate provenance and `NOTICE` / `THIRD-PARTY-LICENSES` for the
license-bearing attributions (`strided-perm` is
`(MIT OR Apache-2.0) AND BSD-3-Clause`).

## Workspace Layout

| Crate | Role |
|-------|------|
| `strided-traits` | Element-operation and scalar traits (`Identity`, `Conj`, `Transpose`, `Adjoint`) |
| `strided-view` | Dynamic-rank strided views (`StridedView`, `StridedViewMut`, `StridedArray`) and metadata ops |
| `strided-kernel` | Cache-optimized map/reduce/broadcast kernels, Rayon threading, pulp SIMD (feature `simd`) |
| `strided-perm` | Cache-efficient permutation / transpose (HPTT-derived, `src/hptt/`), feature `parallel` |
| `strided-einsum2` | Binary einsum via GEMM backends |
| `strided-opteinsum` | N-ary einsum with contraction-order optimization |
| `mdarray-opteinsum` | Einsum adapter for `mdarray` (row-major conversion at the boundary) |
| `ndarray-opteinsum` | Einsum adapter for `ndarray` (direct stride passthrough) |
| `strided-rs` | User-facing facade re-exporting the workspace APIs |

Dense flat-buffer APIs are column-major; see `REPOSITORY_RULES.md` for the
layout and copy-semantics contracts.

## Pre-Push / PR Checklist

All of the following must pass before pushing or creating a pull request:

```bash
cargo fmt --all -- --check   # run `cargo fmt --all` to fix
cargo test --workspace       # all tests
```

## Repository Rules Review Bot

`.github/workflows/review_bot.yml` reviews every PR diff against
`REPOSITORY_RULES.md`. It runs from the trusted base revision and treats PR
contents as data: the PR head is fetched for `git diff` only, never checked out
or executed. Findings are posted as a single updating PR comment; only
`block`-severity findings fail CI.

Preview the review locally before pushing:

```bash
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
python3 scripts/test-repository-rules-review.py   # the script's own tests
```

Drop the `--dry-run` to include the LLM pass; it needs `DEEPSEEK_API_KEY` in the
environment or in a repo-root `.env` (`pip install -r scripts/requirements-dev.txt`).

The system prompt lives in `ai/prompts/repository-rules-review.md`. Two
deterministic checks run before the LLM and independently of it: secret-shaped
text in added lines blocks the upload entirely, and the **Deprecated Tree
Freeze** rejects source changes under `deprecated/`.

Maintainer escape hatches, both requiring the `maintain`/`admin` role and
reapplication after the latest push:

| Label | Effect |
|-------|--------|
| `rules-review:no-llm` | Skips the LLM pass; deterministic checks still run |
| `rules-review:waive` | Waives the review entirely |

When adding a `## ` section to `REPOSITORY_RULES.md`, also route it in
`SECTION_TRIGGERS` (or `ALWAYS_SECTIONS` / `HUMAN_ONLY_SECTIONS`); an unrouted
section is never shown to the reviewer, and
`test_every_rule_section_is_reachable` fails.

## Build And Test Commands

```bash
cargo build
cargo test --workspace
cargo test -p strided-perm             # single crate
cargo test test_name                   # single test
cargo bench                            # benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench   # enable AVX2/NEON auto-vectorization
```

## Benchmarking Notes

- Start with [`docs/design/erased-execution-policy.md`](docs/design/erased-execution-policy.md) when changing CPU threading thresholds or prepared-plan fast paths; it links the policy, benchmark, worklog, issue, and external results flow.
- This workspace's own regression benchmarks live in `<crate>/benches/`.
  Cross-repository comparisons and published results go to
  [`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).
- Crate READMEs and rustdoc document usage and API contracts, not performance
  tables. Dated worklogs under `docs/worklogs/` may quote measurements as evidence.
- Naive baselines must be credible: pointer-based loops with precomputed
  strides, not per-element high-level indexing.
- Keep setup out of timed regions; use `black_box`.
- For parity with Julia, pin threads (`RAYON_NUM_THREADS=1`,
  `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) unless testing threading.

## Key Constants

| Constant | Value | Origin |
|----------|-------|--------|
| `BLOCK_MEMORY_SIZE` | 32 KB (L1) | Strided.jl `BLOCKMEMORYSIZE` |
| `CACHE_LINE_SIZE` | 64 bytes | Strided.jl `_cachelinelength` |
| transpose micro/macro blocks | 4x4 / 16x16 (f64), 8x8 / 32x32 (f32) | HPTT `blocking_micro_`, `blocking_` |
