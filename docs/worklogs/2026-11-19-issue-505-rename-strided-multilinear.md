# 2026-11-19: #505 — rename `strided-segpoly` to `strided-multilinear`

Scope: the mechanical rename only, before the crate's first publish. The
executor computes `Σ_paths c · Π_operands x[segment]` — a sparse multilinear
form whose sparsity is a list of paths — and "segpoly" names only the special
case where one buffer feeds several operands (where cuEquivariance's
`SegmentedPolynomial` gets its name). The crate is `0.1.0`, `publish = false`,
never published, so the rename costs in-repo edits and needs no alias
(REPOSITORY_RULES.md § Suffix Vocabulary: a pre-1.0 rename deletes the old name
outright).

## What changed

- Directory `strided-segpoly/` → `strided-multilinear/` (`git mv`, 51 tracked
  files), package name, `documentation` URL, description reframed per the issue
  ("cuEquivariance-style segmented polynomials, i.e. sparse multilinear forms").
  No `[lib] name` existed; the lib name follows the package name.
- Workspace member, both lockfiles regenerated (root and the standalone
  `benchmarks/strided-491-mace-ab` workspace; diffs are exactly the rename).
- All `use strided_segpoly::` → `strided_multilinear` (crate tests/examples/
  benches, benchmark harness `src/{cg,lib,main}.rs`), `AGENTS.md`,
  `docs/PROVENANCE_AND_CITATION_POLICY.md`, `docs/design/runtime-simd-dispatch.md`,
  `scripts/check-runtime-simd-dispatch.py` (`PROBES` pair and the demangled
  `CHECKS` fragments `strided_multilinear::simd::RowF64/RowF32`, in lockstep with
  the symbol rename), `strided-basic/src/run_sum.rs` docs.
- Benchmark harness: path dep, README, error labels `"segpoly fwd:"` →
  `"multilinear fwd:"` (labels only).
- README/lib.rs framing per the issue: the segmented-polynomial explanation and
  the cuEquivariance credit stay, led by "executor for cuEquivariance-style
  segmented polynomials, i.e. sparse multilinear forms". The never-existing
  `segpoly-desc` future crate is now "a separate descriptor-lowering crate"
  (no new coined name).
- `CHANGELOG.md` 0.15.0 section: heading renamed, one paragraph records the
  rename with the old name.
- Type names (`SegPolyPlan`, `SegPolyError`, `SegPolyElem`, …) keep the
  cuEquivariance vocabulary, per the issue; public API, `publish = false`,
  features and versions unchanged.

## Deliberately kept (old name remains)

- `docs/worklogs/*.md` and their filenames, `benchmarks/.../raw/*/commands.txt`
  (records of runs that were executed), the `check_report*.py` strings and
  README/CHANGELOG links that point at those kept worklog files, and the
  CHANGELOG rename sentence. Renaming a string that references a kept file, or a
  recorded command, would falsify the record.

## Plan review

Reviewed before implementation by GLM 5.3 Flash (pi `glm-brainstormer`
subagent, read-only). Verdict CHANGES with two IMPORTANT findings, both
plan-text only: (1) the acceptance tolerance list must include CHANGELOG's
unchanged worklog filename links; (2) the `check_report*.py` / raw deviation
needed explicit statement rather than a buried mention. Both addressed in the
plan before any code was written; two minor wording findings also addressed.
`review: glm-5.3-flash CHANGES → plan text fixed, no change-set impact`.
Post-review of the finished diff: gpt-6.1-sol unavailable (OpenAI OAuth
expired, invalid_grant), so the independent post-review ran on DeepSeek V4.1
Flash: no BLOCKER, no IMPORTANT; two MINOR wording nits (this file and the
worklog fmt note) fixed in the amend. `review: deepseek-v4.1-flash no-blocker`.

## Downstream check

`mizukami-group/llmlip.rs` (only branch: `main`, pinned `6bfb041`): no branch
depends on this crate by path or git — the dependency runs the other way (the
benchmark harness depends on llmlip by path). Nothing to rename downstream.

## Verification (x86-64 host, kache-backed builds)

- `cargo fmt --all -- --check` — clean (the longer `strided_multilinear::`
  qualification pushed a handful of lines in the moved tests over the limit;
  `cargo fmt --all` rewrapped `tests/kernels.rs` and `tests/support/mod.rs`).
- `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- `cargo test --workspace` — 1333 passed, 94 suites.
- `python3 scripts/check-runtime-simd-dispatch.py` — ok: 9 dispatched bodies
  reach AVX2 and AVX-512, lane widths f64 8 / f32 16.
- `cargo check` in `benchmarks/strided-491-mace-ab` (with llmlip.rs @ `6bfb041`
  patched by `raw/llmlip-bench-api.diff`) — compiles; its lock renamed.
  Clippy there reports 3 findings in harness code untouched by this diff
  (`measure.rs`, `cg.rs:55`, `lib.rs:133`), pre-existing and outside the repo
  gate (the harness is its own workspace).
- Acceptance grep (`git grep segpoly` + plain `grep -rn`): hits only the kept
  records named above.

Behaviour is unchanged by construction: no logic lines moved, only names,
docs and labels.
