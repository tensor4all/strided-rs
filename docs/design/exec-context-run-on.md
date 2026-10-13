# Running strided kernels on a caller-supplied pool

Date: 2026-10-12 (branch `refactor/view-metadata-cow`)

## Context

`strided-basic` carries no pool. `ExecutionPolicy` "controls fanout only" and
"does not create a Rayon pool or choose CPU placement; callers that need
placement control should install the operation in their chosen executor and then
select `Sequential` or `Rayon`" (`strided-basic/src/execution_policy.rs`). The
policy is `Copy` and lives in a thread-local `Cell`, and `ExecContext` holds only
`Serial`/`MaxThreads`/`Ambient` (`strided-basic/src/exec_context.rs`), so a pool
cannot be stored in either.

The consequence is that a host must `install` its pool at every call site, and a
caller that forgets silently gets `AmbientRayon` — "the current installed Rayon
pool, or the global pool when no pool is installed", i.e. a second worker team
beside the host's. The capability to run on a given pool already exists one layer
up, as `tcontract_exec::strided::run_with_exec`, whose whole reason to exist is
that strided cannot be handed a pool.

## Decision

Add one entry point to the owning abstraction:

```rust
// strided-basic, feature = "parallel"
impl ExecContext {
    pub fn run_on<R: Send>(
        pool: &rayon::ThreadPool,
        max_threads: usize,
        operation: impl FnOnce(ExecContext) -> R + Send,
    ) -> Result<R>;
}
```

- The width is clamped to `pool.current_num_threads()`.
- A width of one enters no Rayon machinery and `operation` receives
  `ExecContext::serial`, matching the repository rule that one thread runs the
  serial kernel rather than the parallel machinery.
- A call from a worker of the *same* pool runs in place
  (`pool.current_thread_index().is_some()`), so nested operations do not nest a
  second team. Inside the pool, `ExecContext::run` still bounds fanout through
  `with_execution_policy`, whose `restrict` keeps nested budgets conservative.
- Zero is rejected with `StridedError::InvalidThreadBudget`, as
  `ExecContext::max_threads` already does.

This does not replace `tcontract_exec::strided::run_with_exec`, which counts
pool entries and in-place runs into `PoolStats` before entering. strided-rs owns
the scheduling primitive; the host keeps its own instrumentation.

## Alternatives rejected

- **A pool field in `ExecutionPolicy`/`ExecContext`.** `ExecutionPolicy` is
  `Copy` and stored in a thread-local `Cell`; `ExecContext` is `Copy` and
  documented as hiding its representation so replay families can add scopes
  without a closed enum. A borrowed pool needs a lifetime, so this has to be a
  scope entry, not a policy value.
- **Keep requiring `install` at every call site.** It works (tenferro does it),
  but it makes pool placement ambient state that a caller can silently fail to
  set, and the failure mode is a second pool rather than an error.
- **Pool creation in strided-rs.** Out of scope by layering: the host owns the
  pool, and the ABI-level story belongs to the executor vocabulary one layer up
  (tensor4all/tcontract-rs#112).

## Constraints recorded

- BLAS/LAPACK providers (`cpueinsum-blas`, `tlinalg-blas`) run on their own
  threads (provider thread variables) and are outside this pool; documentation
  must not imply one uniform thread policy.
- `rayon` is already a `parallel`-feature dependency of `strided-basic`.
- No kernel or fast path changes, so the obligation is a focused test rather
  than a benchmark family.

## Verification

`strided-basic/src/exec_context/tests/tests.rs` (`mod run_on`, feature
`parallel`):

- the pool the operation observes is the one it was handed (2 workers), not the
  ambient pool;
- width one stays on the calling thread with `is_serial()` true, and a
  one-worker pool clamps a request of four to serial;
- a nested call from inside the pool runs in place with the same bound;
- zero is rejected.

Plus the doc example (executed by `cargo test --doc`), `cargo test --workspace`,
`cargo test --profile ci -p strided-basic -p strided-kernel --features parallel`
and `cargo check --workspace --all-targets --all-features`.
