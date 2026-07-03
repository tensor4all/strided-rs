# Fused Elementwise DAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #136 end to end: fused runtime-DAG elementwise kernels in `strided-kernel`, performance benchmarks, and follow-up benchmark-suite support.

**Architecture:** Add a focused `strided-kernel/src/fused.rs` module with public plan types, scalar semantics, validation, static specialization, and interpreter fallback. The interpreter reuses existing shape/block/parallel traversal; the specialization path reuses `map_into` and `zip_map*_into`. Benchmark-suite changes are made only after the `strided-rs` implementation is committed so the exact hash can be recorded.

**Tech Stack:** Rust 2021, `strided-view`, `strided-kernel`, `num-complex`, Criterion benchmarks, sibling `strided-rs-benchmark-suite`.

## Global Constraints

- Follow `AGENTS.md`: run `du -s .` before work; if over 100GB, propose cleanup before continuing.
- Follow `strided-rs/AGENTS.md`: before push or PR, `cargo fmt --check` and `cargo test` must pass.
- Use CodeGraph before grep/find for code understanding when `.codegraph/` exists.
- Keep public API small; expose only `FusedOp`, `FusedInst`, `FusedPlan`, `FusedScalar`, and `fused_elementwise_into`.
- Validate every plan, shape, rank, and output condition before unsafe pointer loops.
- No BLAS dependency and no mixed-dtype promotion.
- Benchmarks must not run concurrently.
- Benchmark-suite result notes must record the exact committed `strided-rs` hash; on macOS, state that CPU pinning is unavailable.

---

## File Structure

- Create `strided-kernel/src/fused.rs`: public fused API, scalar semantics, validation, static matcher, interpreter execution, and private unit tests.
- Modify `strided-kernel/src/lib.rs`: export the new fused module API.
- Create `strided-kernel/tests/fused_elementwise.rs`: integration tests for public behavior and edge cases.
- Modify `strided-kernel/Cargo.toml`: add `fused_elementwise` benchmark target.
- Create `strided-kernel/benches/fused_elementwise.rs`: Criterion comparisons against per-op baselines.
- Modify `strided-rs-benchmark-suite/Cargo.toml` and `src/main.rs` or add a dedicated benchmark binary if the suite structure makes that cleaner.
- Modify `strided-rs-benchmark-suite/README.md` or `benchmarks/README.md`: add fused benchmark usage and recorded hash/result notes.

## Task 1: Public API And Validation

**Files:**
- Create: `strided-kernel/src/fused.rs`
- Modify: `strided-kernel/src/lib.rs`
- Test: `strided-kernel/tests/fused_elementwise.rs`

**Interfaces:**
- Produces:
  - `pub enum FusedOp`
  - `pub struct FusedInst { pub op: FusedOp, pub inputs: Vec<usize> }`
  - `pub struct FusedPlan { pub input_count: usize, pub outputs: Vec<usize>, pub ops: Vec<FusedInst> }`
  - `pub trait FusedScalar`
  - `pub fn fused_elementwise_into<T: FusedScalar>(dests: &mut [StridedViewMut<'_, T>], inputs: &[StridedView<'_, T>], plan: &FusedPlan) -> Result<()>`

- [ ] **Step 1: Write failing validation tests**

Add tests that import `fused_elementwise_into`, `FusedInst`, `FusedOp`, and `FusedPlan`. Cover:

```rust
#[test]
fn fused_rejects_input_count_mismatch() { /* inputs.len() != plan.input_count */ }

#[test]
fn fused_rejects_destination_count_mismatch() { /* dests.len() != plan.outputs.len() */ }

#[test]
fn fused_rejects_bad_value_id_before_writing() { /* op references future value */ }

#[test]
fn fused_rejects_wrong_op_arity_before_writing() { /* Add with one input */ }

#[test]
fn fused_rejects_shape_mismatch_before_writing() { /* input shape differs */ }
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p strided-kernel --test fused_elementwise
```

Expected: compile failure because the fused API is not exported yet.

- [ ] **Step 3: Implement minimal public API and validation**

Create `fused.rs`, add the public types, implement `validate_plan`, and make `fused_elementwise_into` call validation then return `Ok(())` only for zero-length shapes or panic-free no-op behavior. Export from `lib.rs`.

- [ ] **Step 4: Verify GREEN for validation**

Run:

```bash
cargo test -p strided-kernel --test fused_elementwise
```

Expected: validation tests pass or fail only because outputs are not computed yet; if computation assertions are absent in this task, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add strided-kernel/src/fused.rs strided-kernel/src/lib.rs strided-kernel/tests/fused_elementwise.rs
git commit -m "feat: add fused elementwise plan validation"
```

## Task 2: Interpreter Fallback Correctness

**Files:**
- Modify: `strided-kernel/src/fused.rs`
- Modify: `strided-kernel/tests/fused_elementwise.rs`

**Interfaces:**
- Consumes: Task 1 public API.
- Produces: correct interpreter execution for arbitrary valid DAGs and multiple outputs.

- [ ] **Step 1: Write failing interpreter tests**

Add tests for:

```rust
#[test]
fn fused_interprets_reused_dag_value() {
    // out = (a + b) * a
}

#[test]
fn fused_interprets_multiple_outputs() {
    // out0 = a + b, out1 = a * b
}

#[test]
fn fused_interprets_broadcast_stride_zero_inputs() {
    // y = exp(a * b + c_broadcast)
}

#[test]
fn fused_interprets_noncontiguous_inputs_and_outputs() {
    // permuted views for both input and destination
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p strided-kernel --test fused_elementwise
```

Expected: new tests fail because output values remain uncomputed.

- [ ] **Step 3: Implement interpreter execution**

Use `build_plan_fused_small` / `build_plan_fused`, collect destination and input strides, and use `for_each_inner_block_preordered`. For each element in an inner block:

```rust
regs[0..input_count] = loaded input values;
for inst in &plan.ops {
    regs.push(eval_op(inst.op, &regs, &inst.inputs));
}
for (dst, &value_id) in dest_ptrs.iter().zip(&plan.outputs) {
    *dst = regs[value_id];
}
```

For parallel builds, mirror the existing `map_view` pattern with `mapreduce_threaded` and `SendPtr`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p strided-kernel --test fused_elementwise
cargo test -p strided-kernel
```

Expected: all `strided-kernel` tests pass.

- [ ] **Step 5: Commit**

```bash
git add strided-kernel/src/fused.rs strided-kernel/tests/fused_elementwise.rs
git commit -m "feat: interpret fused elementwise dag plans"
```

## Task 3: Scalar Semantics Edge Cases

**Files:**
- Modify: `strided-kernel/src/fused.rs`
- Modify: `strided-kernel/tests/fused_elementwise.rs`

**Interfaces:**
- Consumes: interpreter execution from Task 2.
- Produces: documented real and complex semantics for all `FusedOp` variants.

- [ ] **Step 1: Write failing scalar semantics tests**

Add tests for:

```rust
#[test]
fn fused_real_maximum_minimum_match_nan_contract() {
    // max(NaN, 3.0) == 3.0; min(3.0, NaN) == 3.0; max(NaN, NaN).is_nan()
}

#[test]
fn fused_complex_divide_matches_native_complex_division() {
    // out = a / b for Complex64
}

#[test]
fn fused_complex_abs_returns_norm_in_real_component() {
    // abs(3+4i) == 5+0i
}

#[test]
fn fused_all_real_ops_have_basic_parity() {
    // Divide, Clamp, Log, Sin, Cos, Tanh, Sqrt, Rsqrt, Pow, Expm1, Log1p
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p strided-kernel --test fused_elementwise
```

Expected: tests fail for unimplemented or incorrect ops.

- [ ] **Step 3: Complete `FusedScalar` implementations**

Implement `FusedScalar` for `f32`, `f64`, `Complex32`, and `Complex64`. Real `maximum` / `minimum` use `max` / `min`. Complex ordering ops compare `norm_sqr()` and return one operand; complex `abs` returns `Complex::new(norm, 0.0)`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p strided-kernel --test fused_elementwise
cargo test -p strided-kernel
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add strided-kernel/src/fused.rs strided-kernel/tests/fused_elementwise.rs
git commit -m "feat: define fused scalar semantics"
```

## Task 4: Static Specialization

**Files:**
- Modify: `strided-kernel/src/fused.rs`
- Modify: `strided-kernel/tests/fused_elementwise.rs`

**Interfaces:**
- Consumes: validated plans and semantic interpreter.
- Produces: static dispatch for common one-output patterns through existing map kernels.

- [ ] **Step 1: Write failing specialization parity tests**

In `fused.rs` unit tests, compare `try_static_specialization` output against direct interpreter execution for:

```rust
// unary Exp
// binary Add
// ternary Clamp
// chain (a + b) * a
// chain exp(a * b + c)
```

Add an integration test that verifies a static-specialized plan still handles broadcast inputs.

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p strided-kernel fused
```

Expected: tests fail because the matcher returns false.

- [ ] **Step 3: Implement static matcher**

Add `try_static_specialization` before interpreter fallback. Match only valid single-output plans with exact input ids and op sequences. Dispatch to:

```rust
map_into(&mut dests[0], &inputs[0], |x| eval_unary(op, x))
zip_map2_into(&mut dests[0], &inputs[0], &inputs[1], |a, b| eval_binary(op, a, b))
zip_map3_into(&mut dests[0], &inputs[0], &inputs[1], &inputs[2], |a, b, c| {
    eval_ternary(op, a, b, c)
})
zip_map4_into(&mut dests[0], &inputs[0], &inputs[1], &inputs[2], &inputs[3], |a, b, c, d| {
    eval_four_input_chain(a, b, c, d)
})
```

Return `Ok(true)` if a specialized path ran and `Ok(false)` otherwise.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p strided-kernel
cargo test -p strided-kernel --features parallel
```

Expected: all tests pass with and without parallel feature.

- [ ] **Step 5: Commit**

```bash
git add strided-kernel/src/fused.rs strided-kernel/tests/fused_elementwise.rs
git commit -m "perf: specialize common fused elementwise plans"
```

## Task 5: `strided-kernel` Benchmark And Performance Check

**Files:**
- Modify: `strided-kernel/Cargo.toml`
- Create: `strided-kernel/benches/fused_elementwise.rs`

**Interfaces:**
- Consumes: public fused API.
- Produces: release benchmark evidence against per-op baselines.

- [ ] **Step 1: Write benchmark target**

Create Criterion benchmarks for:

```rust
// contiguous_per_op_add_mul
// contiguous_fused_add_mul
// broadcast_per_op_exp_mul_add
// broadcast_fused_exp_mul_add
// long_chain_per_op
// long_chain_fused_interpreter
```

Use preallocated arrays inside benchmark setup, not inside timed loops except for explicit per-op intermediate buffers that represent baseline cost. Use `black_box`.

- [ ] **Step 2: Run benchmark sequentially**

Run on macOS:

```bash
RAYON_NUM_THREADS=1 cargo bench -p strided-kernel --bench fused_elementwise
```

Expected: benchmark completes. Record that CPU pinning was unavailable on macOS.

- [ ] **Step 3: Verify repository gates**

Run:

```bash
cargo fmt --check
cargo test -p strided-kernel
cargo test
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add strided-kernel/Cargo.toml strided-kernel/benches/fused_elementwise.rs
git commit -m "bench: compare fused elementwise dag kernels"
```

Record the resulting `git rev-parse HEAD` hash for benchmark-suite notes.

## Task 6: Benchmark Suite Support

**Files:**
- Read first: `strided-rs-benchmark-suite/AGENTS.md`
- Modify: `strided-rs-benchmark-suite/Cargo.toml`
- Modify or create: `strided-rs-benchmark-suite/src/main.rs` or `strided-rs-benchmark-suite/src/bin/fused_elementwise.rs`
- Modify: `strided-rs-benchmark-suite/README.md` or `benchmarks/README.md`

**Interfaces:**
- Consumes: committed sibling `../strided-rs` hash with fused API.
- Produces: a reproducible benchmark-suite entry for fused elementwise DAGs.

- [ ] **Step 1: Inspect suite structure and choose binary shape**

Prefer a dedicated binary if `main.rs` is tightly focused on einsum benchmarks:

```toml
[[bin]]
name = "fused-elementwise"
path = "src/bin/fused_elementwise.rs"
```

- [ ] **Step 2: Implement benchmark-suite benchmark**

Benchmark the same three scenarios as `strided-kernel`, but as a standalone suite command using the path dependency on `../strided-rs`.

- [ ] **Step 3: Run benchmark sequentially**

Run:

```bash
cargo run --release --bin fused-elementwise
```

Expected: command prints timings for per-op, specialized fused, and interpreter fused scenarios. On macOS, output or README notes state CPU pinning unavailable.

- [ ] **Step 4: Record exact `strided-rs` hash**

Run in `strided-rs`:

```bash
git rev-parse HEAD
```

Add that hash to the benchmark-suite result note.

- [ ] **Step 5: Commit benchmark-suite changes**

```bash
git add Cargo.toml src/bin/fused_elementwise.rs README.md benchmarks/README.md
git commit -m "bench: add fused elementwise suite"
```

## Final Verification

- [ ] Run in `strided-rs`:

```bash
cargo fmt --check
cargo test
RAYON_NUM_THREADS=1 cargo bench -p strided-kernel --bench fused_elementwise
git log --oneline --decorate -8
```

- [ ] Run in `strided-rs-benchmark-suite`:

```bash
cargo fmt --check
cargo test
cargo run --release --bin fused-elementwise
git log --oneline --decorate -5
```

- [ ] Confirm no `.codegraph/` content is staged.
- [ ] Confirm one `strided-rs` feature branch contains all implementation commits for one PR.
