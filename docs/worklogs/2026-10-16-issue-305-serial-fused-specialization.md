# strided-rs#305 — a serial fused replay takes the static specialization

## Scope

`ErasedFusedPlan::execute` and `ErasedFusedPlan::execute_uninit` never reached
`try_static_specialization_validated` under `ExecContext::serial()`. The
initialized path went through `fused_elementwise_into_serial`, which called the
per-element interpreter directly, and the uninitialized path guarded the
specialization with `!serial`. Every single-threaded fused elementwise op in
tenferro-rs (`crates/tenferro-cpu/src/provider.rs`, `strided_exec_context`, and
the fused lane's `execute_uninit`) therefore ran the interpreter, which
allocates a register vector per inner block.

The serial entry points now try the same specialization as the bounded parallel
ones. The specialized kernels (`map_into_validated`, `zip_map2_into_validated`,
…) choose their own fanout from the ambient execution policy, so the serial
request is passed as an explicit `ExecutionPolicy::Sequential` scope around the
specialization instead of being dropped; that keeps the "no Rayon fanout" part
of `ExecContext::serial()` and the `serial` argument of
`fused_elementwise_into_uninit` intact.

## Verification

- `strided-fused/tests/serial_specialization_alloc.rs` (new, counting global
  allocator, one test function per window): serial `execute` and
  `execute_uninit` of a single-op plan and of a two-op DAG allocate `0` at
  `8^3` and `32^3` with contiguous operands, and `16` (`8^3`) / `43` (`32^3`)
  per call with a permuted right-hand operand — the strided-basic planner
  metadata the specialized kernel itself builds, tracked by
  tensor4all/strided-rs#306. Before the change the same windows counted `19` /
  `21` contiguous and `82` / `1069` permuted.
- `strided-fused/src/fused/tests/tests.rs`: the seven static families are now
  replayed under a serial context as well as a bounded one, each family is
  asserted to be selected (`[1; 7]` hits), and the two contexts are asserted to
  produce identical destinations. The serial family test fails with `[0; 7]`
  hits on the previous revision.
- `cargo fmt --all -- --check` and `cargo test --workspace --profile ci` pass.

## Measured

AMD EPYC 7713P, Linux 6.8.0-101-generic, rustc 1.97.1, release profile with the
`parallel` feature, `ExecContext::serial()`, two-op `add` then `multiply`, f64,
one warm-up call then 2000 timed calls (indicative, not a benchmark-suite run).
Before is `origin/main` `d9cbae8`, after is this change.

| case | before (ns/call) | after (ns/call) | factor |
|:-|-:|-:|-:|
| `8^3` contiguous | 11,485 | 342 | 34x |
| `8^3` permuted rhs | 12,793 | 1,752 | 7x |
| `32^3` contiguous | 694,087 | 12,914 | 54x |
| `32^3` permuted rhs | 732,077 | 125,853 | 6x |

The permuted residual is the per-call planner metadata in `strided-basic`
(one fused plan and one offset vector per call), not interpreter state; the
allocation-free planner in #306 removes it.
