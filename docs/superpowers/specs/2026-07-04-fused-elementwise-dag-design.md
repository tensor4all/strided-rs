# Fused Elementwise DAG Kernel Design

## Goal

Add the full issue #136 fused elementwise runtime-DAG API to `strided-kernel`, then add matching benchmarks in `strided-rs-benchmark-suite`.

The feature provides one public entry point that evaluates a runtime-built scalar expression DAG over broadcast-aware strided inputs and one or more outputs. It must avoid main-memory intermediates, reuse the existing cache-blocked and parallel traversal machinery, and opportunistically dispatch common shapes to existing static map kernels.

## Public API

`strided-kernel` exposes a new `fused` module through `lib.rs`:

```rust
pub enum FusedOp {
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Maximum,
    Minimum,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
}

pub struct FusedInst {
    pub op: FusedOp,
    pub inputs: Vec<usize>,
}

pub struct FusedPlan {
    pub input_count: usize,
    pub outputs: Vec<usize>,
    pub ops: Vec<FusedInst>,
}

pub trait FusedScalar: Copy + MaybeSendSync + 'static {
    fn fused_add(self, rhs: Self) -> Self;
    fn fused_multiply(self, rhs: Self) -> Self;
    fn fused_negate(self) -> Self;
    fn fused_conj(self) -> Self;
    fn fused_divide(self, rhs: Self) -> Self;
    fn fused_abs(self) -> Self;
    fn fused_maximum(self, rhs: Self) -> Self;
    fn fused_minimum(self, rhs: Self) -> Self;
    fn fused_clamp(self, min: Self, max: Self) -> Self;
    fn fused_exp(self) -> Self;
    fn fused_log(self) -> Self;
    fn fused_sin(self) -> Self;
    fn fused_cos(self) -> Self;
    fn fused_tanh(self) -> Self;
    fn fused_sqrt(self) -> Self;
    fn fused_rsqrt(self) -> Self;
    fn fused_pow(self, rhs: Self) -> Self;
    fn fused_expm1(self) -> Self;
    fn fused_log1p(self) -> Self;
}

pub fn fused_elementwise_into<T: FusedScalar>(
    dests: &mut [StridedViewMut<'_, T>],
    inputs: &[StridedView<'_, T>],
    plan: &FusedPlan,
) -> Result<()>;
```

The first implementation targets `f32`, `f64`, `Complex32`, and `Complex64`. All operations are same-dtype. Mixed dtype promotion remains out of scope.

## Semantics

For each logical element position:

1. Load all `inputs` using their existing strides. Broadcast is represented by stride-0 views created before calling `fused_elementwise_into`.
2. Evaluate `plan.ops` in topological order into a per-element register file. Values `0..plan.input_count` are inputs; op `i` appends value id `plan.input_count + i`.
3. Store `plan.outputs[j]` into `dests[j]`.

Validation happens before any output write:

- `inputs.len() == plan.input_count`.
- `dests.len() == plan.outputs.len()`.
- At least one destination exists.
- All inputs and destinations have the same logical shape.
- Every instruction input id refers to an existing earlier value id.
- Each op receives its required arity: unary, binary, or ternary for `Clamp`.
- Every output id refers to an existing value.

Validation errors use existing `StridedError` variants where they fit: `ShapeMismatch`, `RankMismatch`, and `InvalidAxis`. Invalid plan structure uses `StridedError::InvalidAxis { axis, rank }`, where `rank` is the number of values available at that validation point.

`Maximum` and `Minimum` use Rust float `max` / `min` semantics for real scalars: if exactly one operand is NaN, the non-NaN operand is returned; if both are NaN, the result is NaN. Complex maximum/minimum/clamp are not ordered mathematically, so their `FusedScalar` implementation compares magnitudes (`norm`) and returns one of the original complex operands. Complex `Abs` returns a complex value whose real component is the norm and whose imaginary component is zero. Complex `Divide` is native complex division.

## Execution Paths

The public function has two paths behind the same validation and API:

1. Static specialization:
   - Match a small set of common one-output DAGs.
   - Dispatch to existing `map_into`, `zip_map2_into`, `zip_map3_into`, or `zip_map4_into` closures.
   - This preserves the current contiguous fast paths, SIMD dispatch, blocking, and parallel execution.
   - Initial recognized patterns:
     - unary: `Negate`, `Conj`, `Abs`, `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Sqrt`, `Rsqrt`, `Expm1`, `Log1p`
     - binary: `Add`, `Multiply`, `Divide`, `Maximum`, `Minimum`, `Pow`
     - ternary: `Clamp`
     - chains: `(a + b) * a`, `exp(a * b + c)`, and equivalent output ids produced by these exact DAGs.
2. Runtime interpreter fallback:
   - Handles arbitrary arity, arbitrary DAGs, multiple outputs, repeated value use, and operations not recognized by the static matcher.
   - Reuses `build_plan_fused`, `build_plan_fused_small`, `for_each_inner_block_preordered`, and the existing parallel `mapreduce_threaded` path.
   - Allocates the per-element register file once per inner block and reuses it across elements.

The interpreter is the semantic authority. Static specialization is an optimization only; every specialized pattern has a parity test against the same plan evaluated through a non-specialized fallback plan.

## Testing

Tests live primarily in `strided-kernel/tests/fused_elementwise.rs` with focused unit tests in `strided-kernel/src/fused.rs` when private helpers need coverage.

Coverage requirements:

- Basic unary, binary, ternary, and chained real operations.
- DAG reuse such as `(a + b) * a`.
- Multiple outputs.
- Broadcast stride-0 inputs with no materialization.
- Shape mismatch, destination count mismatch, bad value id, wrong arity, and empty output errors.
- Parity against explicit per-op `map_into` / `zip_map*_into` baselines for `f32`, `f64`, `Complex32`, and `Complex64`.
- `Maximum` / `Minimum` NaN behavior for real floats.
- Complex `Divide` and `Abs` behavior.
- Static-specialized results matching interpreter fallback.

Repository gates:

```bash
cargo fmt --check
cargo test -p strided-kernel
cargo test
```

## Benchmarks

In `strided-rs`, add a Criterion benchmark target under `strided-kernel/benches/fused_elementwise.rs` comparing:

- sequential per-op baseline with intermediate arrays
- `fused_elementwise_into` static-specialized plans
- `fused_elementwise_into` interpreter fallback plans

Benchmark scenarios include contiguous and broadcast inputs for:

- `(a + b) * a`
- `exp(a * b + c)`
- a longer SVD-backward-like chain using `Divide`, `Maximum`, `Minimum`, `Sqrt`, and `Rsqrt`

After implementation commits are in place and benchmark output is captured, update `strided-rs-benchmark-suite` with a corresponding benchmark program or mode that depends on the sibling `../strided-rs` path. Its result notes must include the exact `strided-rs` git hash. On macOS, record that CPU pinning is unavailable.

## Commit Strategy

Use one feature branch and one eventual PR for `strided-rs`. Keep commits staged:

1. Spec and implementation plan.
2. Public fused API and validation tests.
3. Interpreter fallback and correctness tests.
4. Scalar semantics edge cases.
5. Static specialization and parity tests.
6. `strided-kernel` benchmark and recorded performance check.
7. Benchmark-suite support in `strided-rs-benchmark-suite` after the `strided-rs` hash is committed.

Each implementation commit must be backed by tests. Benchmark-related commits must include release benchmark evidence gathered sequentially.
