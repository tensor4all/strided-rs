# strided-rs#218 — `strided-perm`'s test-gated trio removed

## Scope

[#218](https://github.com/tensor4all/strided-rs/issues/218) has three parts.
Part 1 (inline `#[cfg(test)] mod` bodies out of `src/`) landed earlier, in #362
and the `strided-kernel` → `strided-basic` split: the maintained crates now hold
36 `#[cfg(test)] mod tests;` declarations and no inline `#[cfg(test)] mod x { .. }`
body in `strided-*/src`. This worklog covers part 2, the decision about
`strided-perm`'s `#[cfg(test)]`-gated `block`, `kernel` and `order` modules.
Part 3 (split the files still over 1000 lines) stays open; the re-measured
count is at the end.

## Decision: delete the trio (option b)

The three modules are unreachable from any shipped path: their only callers are
each other and their own tests, and `strided-perm`'s live copy path plans
through `src/hptt/` (`copy_raw` → `hptt::build_permute_plan` →
`execute_permute_blocked`). The same policy — dimension fusion, loop order,
block sizes — is implemented and shipped by `strided-basic`
(`src/fuse.rs`, `src/order.rs`, `src/block.rs`, `src/kernel.rs`), which is where
`strided-fused` and the typed APIs reach it.

So the trio was a second implementation of a policy another crate owns, kept
alive only by its own tests. Deleting it follows the "one owner per policy"
rule, and R-performance-25 ("delete legacy or obsolete paths that only survive
through artificial references"). Folding it into test-support (option a) would
preserve exactly that duplicate while still not making it an oracle for the
shipped engine, and documenting why it ships (option c) would not give it a
purpose.

Deleted with the modules: the test-only items only they and their tests kept
alive — `fuse_dims`, `compress_dims`, `compute_importance`, `sort_by_importance`,
`compute_costs`, `fuse_dims_bilateral` in `fuse.rs`, `try_fuse_group` in
`copy.rs` — and the tests pinning them. Three bilateral-fusion tests stay, now
calling the live `plan_bilateral_fusion` directly. `scripts/check-oracle-copy.sh`
and its `ci.yml` step are retired: the script existed only to keep the deleted
`strided-perm/src/block/tests/oracle.rs` byte-identical to
`strided-basic/src/block/tests/oracle.rs`.

Left alone deliberately: `strided-perm`'s `pub const BLOCK_MEMORY_SIZE` and
`pub const CACHE_LINE_SIZE` lose their last in-crate user with the trio. Both are
public items, so removing them is a public-API decision, not part of this
test-only cleanup.

## Measurements

`cargo test --workspace -j8` (`CARGO_BUILD_JOBS=8`), same tree except this diff:

| | passed | failed | test binaries |
|---|---|---|---|
| before | 1183 | 0 | 78 |
| after | 1127 | 0 | 78 |

The −56 is exactly the deleted tests: 43 in the trio, 8 for the deleted `fuse`
helpers, 5 for `try_fuse_group`. `cargo test -p strided-perm -j8` goes from 101
lib unit tests to 45. `cargo clippy --workspace --all-targets -- -D warnings`,
the `--features parallel --lib --tests --examples -D warnings` lane, and
`cargo fmt --all -- --check` are clean.

## What is left of #218

`scripts/count-audit-metrics.py .` still reports `files_over_1000 22` (and
`inline_mods 1`, the last inline body, which is in
`strided-basic/benches/support/mod.rs`, not in `src/`). Thirteen source files are
over 1000 lines, `strided-kernel/src/erased.rs` (2935) and
`strided-basic/src/map_view.rs` (2777) largest among them; the other nine are
test and bench files under `tests/`, `benches/`. Splitting those is its own
change: a mechanical move across 22 files is not reviewable beside a deletion,
and `strided-basic/src/erased/` already shows the intended shape (submodules
under a module directory). The 30 `#[cfg(test)]` items left in non-test files
(`strided-basic`'s block/kernel/map_view/ops_view/raw_ops/static_indexing_plan/
threading helpers, `strided-fused/src/fused.rs`'s static-family accessors,
`strided-perm/src/hptt/plan.rs`'s `total_len`) belong to that same follow-up:
moving them means a crate-level test-support module and edits to the production
files that import them.
