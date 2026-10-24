# strided-exec

Splittable CPU placement for the strided stack. `strided-exec` is the one place
that starts tasks: kernels take an `Exec` token and split it, and no scheduler
type appears in any public signature.

Design record: `docs/design/cpu-exec-placement.md` (placement proposal for
tenferro-rs#2206, #2272, #2273).

## Usage

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use strided_exec::{Exec, Pool};

// The owner builds a pool once and keeps it.
let pool = Pool::builder().threads(4).build()?;

// Per call, the owner mints a token for `budget` slots of the pool.
let mut exec = Exec::new(&pool, 4)?;

// Several phases in sequence: reborrow, because entries consume the token.
let total = AtomicUsize::new(0);
exec.reborrow().for_ranges(1000, 64.try_into().unwrap(), |r| {
    total.fetch_add(r.len(), Ordering::Relaxed);
});

// Independent parts: each job gets a child token with its share of the slots.
exec.fork(2, |_job, child| {
    let _ = child.budget();
});
# Ok::<(), strided_exec::Error>(())
```

`Exec::serial()` needs no pool and runs everything inline.

## Intended consumer pattern

The owner of the CPU resources (an engine, `tcontract-exec`'s pool) holds a
`strided_exec::Pool` and mints `Exec::new(&pool, budget)` for each call, passing
the token to kernels by value. Kernels never store a token and never mint one.
Today `tcontract-exec`'s `Exec` is `Copy` and its `Pool::rayon_pool()` is public;
both go away when tcontract migrates to this crate.

## API

| Item | Role |
| --- | --- |
| `Pool`, `PoolBuilder` | fixed set of workers; `threads`, `thread_name`, `stack_size` (default 16 MiB, sized for recursive LU), `on_start`, `build` |
| `Exec::new(&pool, budget)` | mint a token; the only clamp (to the pool width); `budget == 0` is an error |
| `Exec::serial()` | the calling thread only; `const` |
| `Exec::budget`, `is_serial` | inspect the token (`is_serial` means budget 1) |
| `Exec::reborrow(&mut self)` | same pool and budget while the parent is exclusively borrowed |
| `Exec::limit(max)` | narrow to `min(budget, max)` |
| `Exec::fork(jobs, f)` | `min(jobs, budget)` tasks; shares differ by at most one and sum to the budget; job `i` gets a child token |
| `Exec::for_ranges(len, grain, f)` | at most `budget` tasks, ranges of at least `grain` except possibly the last |
| `Exec::broadcast(f)` | `f(i)` for `i in 0..budget`, all concurrent, or `Err(Refused)` |
| `Exec::faer(f)` (feature `faer`) | enter the pool once, hand faer a `Par` of the largest power of two not above the budget |

## Contracts

* **No duplication.** `Exec` is neither `Clone` nor `Copy` (compile-fail
  doctests), and every parallel entry consumes it.
* **Slots are conserved.** Under one root token the number of tasks running at
  once is at most its budget at every nesting depth. A parent waiting for its
  children keeps no slot.
* **Lazy and inline.** A serial token, budget one, or one task runs inline on the
  calling thread and makes no scheduler call, so the process-global Rayon pool is
  never started implicitly.
* **Placement by caller.** A worker of the target pool spawns into the current
  scope without installing; a plain thread or a worker of another pool enters the
  target pool once and waits. No thread-local state exists in the crate.
* **`broadcast`** refuses with `Refused::ForkChild` for a token derived from
  `fork` (through `reborrow` or `limit` too), and with `Refused::CallerInPool`
  when the caller is a worker of the same pool. Roots, and their `reborrow`s and
  `limit`s, are eligible. Callers must handle `Refused` by falling back to a
  barrier-free path. Concurrent broadcasts on one pool queue in the same order on every
  worker (no interleaving); participants are the pool's workers `0..budget`, so
  blocking work of another token on those workers delays the broadcast.
* **No scheduler type is public.** There is no constructor from, and no accessor
  returning, a scheduler pool. `scripts/check-no-public-rayon.sh` runs
  `cargo public-api -p strided-exec` (when installed) and fails if `rayon::`
  appears in the output.

## Features

| Feature | Effect |
| --- | --- |
| `parallel` (default) | run on a private pool. Without it the same types exist, `PoolBuilder::build` returns a width-1 pool and every entry runs inline |
| `faer` | `Exec::faer`; depends on `t4a-faer` (published under the name `faer`), mirroring tenferro-rs |
| `ledger` | test support: `ledger::max_live(root)`, `live`, `installs`, and `Exec::ledger_root`. Records live tasks per root token and pool installs; compiled out when off |

Consumers assert `ledger::max_live(root) <= budget` in their own tests, and keep
a test binary that runs serial and pooled work and then checks that
`rayon::ThreadPoolBuilder::new().build_global()` still succeeds.
