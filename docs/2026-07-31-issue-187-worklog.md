# Issue #187 Worklog

## RED History

- RED: indexed uninitialized destinations required a safe copy-then-update
  boundary; direct initialized backing views were forbidden.
- GREEN work introduced private raw-pointer writers and a closure-scoped
  post-copy receipt.
- Acceptance work adds differential lifecycle and indexed coverage.

## Architecture and Safety Contract

- Only CopyPlan constructs the private post-copy receipt.
- Receipt construction is closure-scoped and cannot escape the HRTB helper.
- Uninitialized writers use MaybeUninit storage and full-overwrite or
  copy-then-update proofs before any typed read.
- Reduction terminal writers use raw pointers plus validated extents.
- Integer erased scatter uses wrapping i32/i64 combine functions; typed public
  scatter semantics remain unchanged.

## Verification

Commands:

    cargo fmt --all
    cargo test -p strided-kernel --test issue_187_uninit_indexed
    cargo test -p strided-kernel --features parallel --test issue_187_uninit_indexed
    cargo bench -p strided-kernel --features parallel --bench issue_187_uninit_indexed --no-run

Benchmark results: not run in this pass. The runner records 31 alternating
initialized/uninitialized samples and reports no invented timing values.

Remaining verification: Miri lifecycle execution and the full repository gate.
