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

Initial fixture pass did not run benchmark timing; later affinity benchmark
evidence is recorded below with exact candidate-relative upper bounds.

## Typed-storage migration after issue #190

The indexed and reduction fixtures now use concrete `Vec<MaybeUninit<T>>`
storage and `from_uninit_slice`; typed accessors are used for post-replay
inspection. The all-dtype macros instantiate the concrete `$ty`, and hole
tests do not read unreachable elements. The only raw pointer construction
left in these fixtures is the narrow, documented stale-invalid `Bool` input
case.

Verification on the issue-187 worktree:

```text
cargo fmt --all
cargo test -p strided-kernel --test issue_187_uninit_indexed --test issue_187_uninit_reduce
cargo test -p strided-kernel --features parallel --test issue_187_uninit_indexed --test issue_187_uninit_reduce
cargo test -p strided-kernel --test issue_187_source_contract
cargo check -p strided-kernel --all-targets --all-features
cargo bench -p strided-kernel --features parallel --bench issue_187_uninit_indexed --no-run
```

Results: indexed 68/68 and reduction 11/11 passed in both default and
parallel configurations; source contract 5/5 passed; all-target and bench
no-run checks passed.

Focused strict-provenance Miri passed:

```text
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed aligned_uninit_lifecycle_all_indexed_families
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed bool_gather_invalid_operand_rejects_before_mutation
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed dynamic_update_hole_layout_preserves_unreachable_bytes
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed gather_validation_errors_preserve_sentinel
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_reduce validation_errors_leave_uninitialized_bytes_untouched
```

The first Miri pass exposed a fixture bug: an invalid-Bool destination was
being compared before initialization. It was changed to initialized typed
Bool sentinel storage; no production code was changed. The corrected run
passed all five filters.

## Sol high follow-up

The dynamic-update hole coverage now has both an initialized-canary test and
a separate strict-Miri test with genuinely uninitialized unreachable holes;
the latter only inspects reachable slots. Scatter extrema coverage executes
the uninitialized scatter path for i32 and i64 with repeated indices under
Serial and bounded 1/2/4-thread contexts. The lifecycle coverage executes
gather, dynamic slice, dynamic update, scatter, reduction, and the
copy-then-update receipt path; the large threshold benchmark remains outside
the focused Miri filters.

Fresh coverage: `53/53 files passed` using:

```text
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Affinity benchmark evidence, all candidate-relative 95% upper bounds at or
below the 1.20 gate:

```text
t1 taskset CPU60 upper95: reduce 1.0169, gather 1.0415,
  dynamic_slice 1.1028, dynamic_update 1.1966, scatter 0.7869
t4 taskset CPUs60-63 upper95: reduce 1.0456, gather 1.0337,
  dynamic_slice 0.9308, dynamic_update 1.1312, scatter 0.7856
```

Sol high focused Miri passed with strict provenance and symbolic alignment:

```text
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed aligned_uninit_lifecycle_all_indexed_families
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed uninit_lifecycle_executes_dynamic_and_scatter_families
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed dynamic_update_uninitialized_holes_are_never_read
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_indexed scatter_integer_extrema_wrap_in_uninit_path
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_reduce axis_holes_negative_stride_and_identity_match
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_reduce uninit_reduce_product_and_nonfinite_match_initialized_replay
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo +nightly miri test -p strided-kernel --test issue_187_uninit_reduce uninit_reduce_simd_tail_and_sum_squares_match_initialized_replay
```
