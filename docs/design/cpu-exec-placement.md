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
- `strided-perm` and `strided-basic` never exceed the budget they are handed;
  within it the width is the token's decision (see Thresholds). No thread-local
  scope exists in any strided crate.
- Only `threadteam` depends on `rayon` in this workspace; `deny.toml` rejects a
  direct `rayon` dependency elsewhere (`cargo deny check bans` in CI). The check
  that no `rayon::` path appears in `threadteam`'s public API lives in
  threadteam-rs.

## Thresholds

- There is no element-count threshold. A kernel multiplies its element count by
  the serial throughput of its class on the host class
  (`strided_basic::execution::KernelClass::ns_per_element`; `x86_64-avx512`: copy 0.171, map 0.173, exp-like 5.60,
  reduce 0.104 ns per element; `aarch64-apple` and the `generic` row of every unmeasured class: 0.090, 0.100, 1.38,
  0.081; the table, with its derivation, lives in `strided-basic/src/cost.rs` and
  `strided_perm::copy_ns_per_element_of`; rates and the `threadteam` dispatch costs of a class come from the same
  machine and clock state)
  and asks the token: `Exec::pays(estimate, sections, width)` for a gate and
  `Exec::width_for(estimate, sections)` for a width. A region runs in parallel
  when its serial time is at least `PAYS` times the dispatch cost of every
  section; the cost depends on the kind of token (team, plain token inside its
  pool, plain token outside it) and is read at call time.
  The class of an element function rides with the function passed to `map_into`
  and its siblings: a closure is a `Map`, and a caller declares an expensive one
  with `KernelClass::Transcendental.with(f)` (`strided_basic::Classed`, trait
  `MapFn`).
- `sections` is the number of dispatches the entry performs: 1 for a flat entry
  (`for_ranges`: copies, fills, the permutation engine, axis-chunked kernels),
  `ceil(log2 width)` for a tree entry (`parallel_map_reduce`), and 1 per node for
  the dimension splitter, which decides again for the work each child has left.
  The softmax of `strided-nn` is a tree entry of the transcendental class.
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
