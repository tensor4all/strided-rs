# Cargo Build-Artifact Reduction

## Test-profile debuginfo

On 2026-08-09, a cold `cargo test --workspace --no-run` comparison measured
the effect of disabling debuginfo in the test profile. Both builds used fresh
target directories, four Cargo jobs, disabled incremental compilation, and an
empty `RUSTC_WRAPPER`.

Environment:

- macOS Darwin 25.5.0 on Apple Silicon (`arm64`)
- `rustc 1.97.0 (2d8144b78 2026-07-07)`
- `cargo 1.97.0 (c980f4866 2026-06-30)`

Commands:

```bash
env CARGO_TARGET_DIR=<fresh-target> CARGO_BUILD_JOBS=4 \
  CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
  cargo test --workspace --no-run

env CARGO_TARGET_DIR=<fresh-target> CARGO_BUILD_JOBS=4 \
  CARGO_INCREMENTAL=0 CARGO_PROFILE_TEST_DEBUG=0 RUSTC_WRAPPER= \
  cargo test --workspace --no-run
```

Results use allocated size from `du -sk`:

| Configuration | Target size | Cold build time |
|---|---:|---:|
| Cargo default test profile | 1,717,360 KiB | 36.62 s |
| Test profile with `debug = 0` | 824,372 KiB | 30.61 s |

Disabling test debuginfo reduced allocated target size by 892,988 KiB
(52.0%) and reduced this cold build sample by 6.01 seconds (16.4%). The test
profile therefore sets `debug = 0`; debug assertions and overflow checks retain
their Cargo defaults.

The build produced 33 test executables. The hosted-CI profile and
integration-test consolidation were measured independently below.

## Development-profile debuginfo

The same method was applied independently to `cargo build --workspace`, using
`CARGO_PROFILE_DEV_DEBUG=0` for the experimental build.

| Configuration | Target size | Cold build time |
|---|---:|---:|
| Cargo default development profile | 1,082,416 KiB | 26.18 s |
| Development profile with `debug = 0` | 497,244 KiB | 23.59 s |

Disabling development-profile debuginfo reduced allocated target size by
585,172 KiB (54.1%) and reduced this cold build sample by 2.59 seconds (9.9%).
The development profile therefore also sets `debug = 0`. Incremental
compilation, debug assertions, and overflow checks retain their Cargo defaults.

## Hosted-CI profile

A named `ci` profile was then measured against the debug-free test profile:

```toml
[profile.ci]
inherits = "test"
incremental = false
strip = "symbols"
```

Both test builds used fresh target directories and produced the same 33 test
executables.

| Configuration | Target size | Cold build time |
|---|---:|---:|
| Test profile with `debug = 0` | 824,372 KiB | 30.61 s |
| CI profile with stripped symbols | 703,520 KiB | 30.98 s |

The CI profile reduced allocated size by another 120,852 KiB (14.7%). The
0.37-second time difference is noise for a single cold sample. Hosted test jobs
use this profile and report `du` and `df` after testing so Linux and macOS runner
usage remains visible. Coverage keeps its instrumentation-owned profile and is
not stripped.

## Rejected integration-test consolidation

The `strided-kernel` crate has 18 integration-test executables. An experiment
consolidated eleven general suites into one explicit harness while retaining
the issue-specific Miri/source-contract targets and two process-isolated
suites. This would have reduced the workspace from 33 test executables to 23.
A fresh CI-profile comparison measured:

| Configuration | Target size | Cold build time |
|---|---:|---:|
| Separate general integration targets | 703,520 KiB | 30.98 s |
| Consolidated general integration target | 678,756 KiB | 29.93 s |

The 24,764 KiB (3.5%) reduction did not justify changing focused `--test`
commands and increasing harness coupling. The experiment also demonstrated
two required isolation boundaries: `copy_plan_alloc` uses a process-wide
allocation counter, while `execution_policy` tests Rayon and thread-local
policy isolation. Both failed when unrelated tests ran concurrently in the
same process. Integration-test consolidation was therefore rejected and the
existing targets were retained.
