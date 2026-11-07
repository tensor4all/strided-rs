# Issue #198 worklog

## 2026-07-31

- Reproduced the `strided-einsum2` `faer + parallel` feature combination failure
  after the merged uninitialized GEMM work (#188/#196).
- Added the existing `MaybeSendSync` contract to the uninitialized contiguous
  output wrapper and its preparation entry point. This is a type-bound fix only;
  it does not change kernel behavior or threading policy.
- Verification: `cargo fmt --all -- --check`; `cargo test -p strided-einsum2`.
