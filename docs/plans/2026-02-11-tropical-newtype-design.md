# Tropical Newtype Design

## Goal

Support non-standard semirings (starting with tropical/max-plus algebra) in the existing einsum stack with zero changes to existing code.

## Approach: Newtype Wrapper

Instead of introducing a `SemiringOps<T>` trait and parameterizing the entire einsum stack, define a newtype `Tropical(f64)` that implements Rust's standard arithmetic traits with tropical semantics. The existing type system handles everything:

- `Tropical` satisfies `ScalarBase` (the existing einsum trait bound)
- `Tropical` does NOT satisfy `faer::ComplexField` or `BlasGemm` -- GEMM is automatically excluded by the type system, falling back to naive loops
- No API changes, no new parameters, no refactoring

### Alternatives Considered

**A. Parallel API (two entry points):** Add `einsum2_semiring_into` alongside existing `einsum2_into`, parameterized by `SemiringOps<T>`. Keeps existing API stable but duplicates code.

**B. Unified API (SemiringOps-centric):** Refactor `einsum2_into` to take `ops: &S` where `S: SemiringOps<T>`. Clean but massive breaking change -- every call site needs updating, GEMM specialization becomes complex.

**C. Newtype wrapper (chosen):** Zero changes to existing code. The type system provides backend dispatch for free. Simplest possible design.

## Design

### Crate: `strided-traits`

New lightweight crate for types and traits supporting the strided ecosystem.

```
strided-traits/
  Cargo.toml      # deps: num-traits, strided-view
  src/
    lib.rs         # pub mod tropical;
    tropical.rs    # Tropical type + tests
```

### Tropical Type

```rust
/// Tropical semiring element (max-plus algebra).
///
/// - Addition: max(a, b)
/// - Multiplication: a + b (standard)
/// - Zero (additive identity): -inf
/// - One (multiplicative identity): 0.0
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Tropical(pub f64);
```

`#[repr(transparent)]` ensures identical memory layout to `f64`, enabling potential zero-cost view transmutation in the future.

### Trait Implementations

| Trait | Implementation | Notes |
|-------|---------------|-------|
| `Add` | `Tropical(self.0.max(rhs.0))` | Tropical addition = max |
| `Mul` | `Tropical(self.0 + rhs.0)` | Tropical multiplication = standard addition |
| `Zero` | `Tropical(f64::NEG_INFINITY)` | Additive identity |
| `One` | `Tropical(0.0)` | Multiplicative identity |
| `Copy`, `Clone` | derive | Required by `ScalarBase` |
| `PartialEq` | derive (inner f64 comparison) | Required by `ScalarBase` for optimization branches (`beta == zero`) |
| `Send + Sync` | auto-derived | `f64` is Send+Sync |
| `ElementOpApply` | identity (default impl) | Real-valued; conj/transpose/adjoint = self |

### ElementOpApply Default Implementation

Change in `strided-view/src/element_op.rs` -- add default methods:

```rust
pub trait ElementOpApply: Copy {
    fn conj(self) -> Self { self }
    fn transpose(self) -> Self { self }
    fn adjoint(self) -> Self { self }
}
```

Backwards compatible: existing explicit impls (f64, Complex) override defaults. New real-valued newtypes need only `impl ElementOpApply for MyType {}`.

### Usage

```rust
use strided_traits::Tropical;
use strided_einsum2::einsum2_into;

// Tropical matrix multiply: C_ik = max_j (A_ij + B_jk)
einsum2_into(
    c.view_mut(), &a.view(), &b.view(),
    &['i','k'], &['i','j'], &['j','k'],
    Tropical::one(),   // alpha = 0.0 (multiplicative identity)
    Tropical::zero(),  // beta = -inf (additive identity = overwrite)
)?;
```

### PartialEq Note

`PartialEq` is not a semiring axiom. It is required by the current `ScalarBase` trait bound for performance optimizations in `bgemm_naive` (`beta == T::zero()`, `alpha == T::one()` checks). For `Tropical`, derived `PartialEq` compares inner f64 values and works correctly with these checks.

## Tests

Verify semiring laws using a shared helper:

```rust
fn assert_semiring_laws<T: Copy + PartialEq + Debug>(
    a: T, b: T, c: T,
    add: impl Fn(T, T) -> T,
    mul: impl Fn(T, T) -> T,
    zero: T, one: T,
) {
    // Additive identity: a + 0 = a
    // Multiplicative identity: a * 1 = a
    // Additive commutativity: a + b = b + a
    // Additive associativity: (a + b) + c = a + (b + c)
    // Multiplicative associativity: (a * b) * c = a * (b * c)
    // Distributivity: a * (b + c) = a*b + a*c
    // Zero annihilation: a * 0 = 0
}
```

Test cases:
- `test_tropical_semiring_laws` -- verify all 7 laws with representative values
- `test_tropical_zero_one` -- identity elements
- `test_tropical_special_values` -- NEG_INFINITY, 0.0, positive, negative values

## Scope

- **In scope:** `Tropical` newtype, trait impls, semiring law tests, `ElementOpApply` default impl
- **Out of scope:** Integration tests with einsum2/opteinsum, TropicalMin, zero-cost view transmutation, benchmarks
