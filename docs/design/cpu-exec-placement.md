# CPU placement in strided-rs

The placement token itself (`Pool`, `PoolBuilder`, `Exec`, `fork`, `for_ranges`,
`external`, `broadcast`, the test ledger) lives in
[threadteam-rs](https://github.com/tensor4all/threadteam-rs). Its design, invariants,
behaviour decisions and the cross-repository ownership table are in
[`docs/design/placement.md`](https://github.com/tensor4all/threadteam-rs/blob/main/docs/design/placement.md)
there. This page records only what is specific to the strided crates.

## How strided kernels take the token

- Every entry that can fan out takes a `threadteam::Exec` by value as its last
  parameter and never stores it. Plans (`CopyPlan`, `UpdatePlan`, the erased
  plans) never hold an `Exec`; replay takes it per call.
- `strided-basic` re-exports `Exec`, `Pool` and `PoolBuilder`, so a consumer that
  depends on `strided-basic` does not need its own `threadteam` dependency.
- `strided-perm` and `strided-basic` use the budget they are handed and never
  choose a width. No thread-local scope exists in any strided crate.
- Only `threadteam` depends on `rayon` in this workspace; `deny.toml` rejects a
  direct `rayon` dependency elsewhere (`cargo deny check bans` in CI). The check
  that no `rayon::` path appears in `threadteam`'s public API lives in
  threadteam-rs.

## Thresholds

- `strided_perm::MINTHREADLENGTH` is the single element-count threshold, re-exported
  by `strided-basic`. Up to it a kernel runs serially; above it the kernel asks for
  `parallel_threads_for_len(len, budget)` tasks.
- A serial token (`Exec::is_serial()`) runs the serial kernel directly, without
  entering any pool.
- Without the `parallel` feature the types and signatures are unchanged; every
  pool has width 1 and every entry runs inline. The `parallel` features of
  `strided-perm` and `strided-basic` forward to `threadteam/parallel`.

## Tests

Dev-dependencies enable `threadteam`'s `ledger` feature; the placement tests in
`strided-basic`, `strided-fused` and `strided-perm` assert that the maximum number of
live tasks per root token stays within its budget, and the "global pool not started"
tests assert that no kernel reaches Rayon's global registry.
