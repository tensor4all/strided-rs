# Custom Scalar Type Support: Trait Refactoring Design

## Goal

Make strided-rs's trait/type system flexible enough for custom scalar types (e.g., tropical semiring from `tropical-gemm`) to work with einsum, without requiring `ElementOpApply` for types that only need `Identity` views.

## Problems with Current Design

1. **`ElementOp::apply<T: ElementOpApply>`** requires `ElementOpApply` even for `Identity` (which just returns the value), forcing all kernel functions to require it.
2. **`ScalarBase` includes `ElementOpApply`** — custom types that don't need conj/transpose/adjoint can't satisfy it.
3. **`Scalar` (with faer) requires `ComplexField`** — custom types can't be used with einsum at all.
4. **Orphan rule** — `ElementOpApply` in `strided-view` means external crates can't implement it for their types.

## Design

### New Crate: `strided-traits`

Lightweight crate with shared traits. External crates (e.g., `tropical-gemm`) can depend on it.

```
strided-traits/
  Cargo.toml      # deps: num-traits, num-complex
  src/
    lib.rs
    element_op.rs  # ElementOp<T>, ComposableElementOp<T>, ElementOpApply, marker types
    scalar.rs      # ScalarBase
```

### `ElementOp<T>` — Generic Over T

The key change: make `ElementOp` generic over the element type.

```rust
pub trait ElementOp<T>: Copy + Default + 'static {
    const IS_IDENTITY: bool = false;
    fn apply(value: T) -> T;
}

// Identity: works with ANY Copy type (no ElementOpApply needed)
impl<T: Copy> ElementOp<T> for Identity {
    const IS_IDENTITY: bool = true;
    fn apply(value: T) -> T { value }
}

// Conj/Transpose/Adjoint: only work with ElementOpApply types
impl<T: ElementOpApply> ElementOp<T> for Conj {
    fn apply(value: T) -> T { value.conj() }
}
```

### `ComposableElementOp<T>` — Composition for Complex Types

Composition associated types are separated into a supertrait, only available when `T: ElementOpApply`:

```rust
pub trait ComposableElementOp<T: ElementOpApply>: ElementOp<T> {
    type Inverse: ComposableElementOp<T>;
    type ComposeConj: ComposableElementOp<T>;
    type ComposeTranspose: ComposableElementOp<T>;
    type ComposeAdjoint: ComposableElementOp<T>;
}
```

This means `.conj()`, `.transpose_2d()`, `.adjoint_2d()` on `StridedView` are only available when the element type supports complex operations.

### `ElementOpApply` — Default Identity Methods

```rust
pub trait ElementOpApply: Copy {
    fn conj(self) -> Self { self }
    fn transpose(self) -> Self { self }
    fn adjoint(self) -> Self { self }
}
```

External crates can write `impl ElementOpApply for TropicalMaxPlus<f64> {}` with zero code. Existing explicit impls (f64, Complex64) override the defaults.

### `ScalarBase` — Without `ElementOpApply`

```rust
pub trait ScalarBase:
    Copy + Send + Sync
    + Mul<Output = Self> + Add<Output = Self>
    + num_traits::Zero + num_traits::One
    + PartialEq
{ }
```

### Impact on `StridedView`

```rust
// Struct: no Op bound
pub struct StridedView<'a, T, Op = Identity> { ... }

// Metadata methods: no bounds needed
impl<'a, T, Op> StridedView<'a, T, Op> {
    pub fn dims(&self) -> &[usize] { ... }
    pub fn permute(&self, ...) -> StridedView<'a, T, Op> { ... }
}

// Element access: needs Op: ElementOp<T>
impl<'a, T: Copy, Op: ElementOp<T>> StridedView<'a, T, Op> {
    pub fn get(&self, indices: &[usize]) -> T { Op::apply(raw) }
}

// Composition: needs T: ElementOpApply, Op: ComposableElementOp<T>
impl<'a, T: Copy + ElementOpApply, Op: ComposableElementOp<T>> StridedView<'a, T, Op> {
    pub fn conj(&self) -> StridedView<'a, T, Op::ComposeConj> { ... }
    pub fn transpose_2d(&self) -> Result<StridedView<'a, T, Op::ComposeTranspose>> { ... }
}
```

### Impact on Kernel Functions

Mechanical: `Op: ElementOp` → `Op: ElementOp<T>`, remove `T: ElementOpApply`:

```rust
// Before
pub fn map_into<D, A: Copy + ElementOpApply, Op: ElementOp, F>(...)
// After
pub fn map_into<D, A: Copy, Op: ElementOp<A>, F>(...)
```

### Impact on Einsum

- `Scalar` keeps `ElementOpApply` in its bounds (backward compatible for f64/Complex64)
- New `einsum2_naive_into<T: ScalarBase>`: closure-based element mapping, naive GEMM, works with custom types
- `bgemm_naive` gets closure-based variant (no `ElementOpApply` in bounds)

### Usage with Tropical Types

```rust
use tropical_gemm::TropicalMaxPlus;
use strided_einsum2::einsum2_naive_into;

// Tropical matrix multiply: C_ik = max_j (A_ij + B_jk)
einsum2_naive_into(
    c.view_mut(), &a.view(), &b.view(),
    &['i','k'], &['i','j'], &['j','k'],
    TropicalMaxPlus::one(),    // alpha = multiplicative identity
    TropicalMaxPlus::zero(),   // beta = additive identity (overwrite)
    |x| x,                    // map_a: identity
    |x| x,                    // map_b: identity
)?;
```

### Orphan Rule Analysis

| Scenario | Trait location | Type location | Valid? |
|----------|---------------|---------------|--------|
| `impl ElementOpApply for TropicalMaxPlus` | strided-traits | tropical-gemm | Yes (local type) |
| `impl ScalarBase for TropicalMaxPlus` | strided-traits | tropical-gemm | Yes (blanket impl) |
| Future `impl FastGemm for TropicalMaxPlus` | strided-traits | tropical-gemm | Yes (local type) |

### Dependency Graph

```
strided-traits (NEW, no strided-* deps)
    ↑
strided-view
    ↑
strided-kernel
    ↑
strided-einsum2
    ↑
strided-opteinsum

tropical-gemm ──→ strided-traits (optional dep)
```

## Scope

- **In scope:** `strided-traits` crate, `ElementOp<T>` refactor, `ComposableElementOp<T>`, `ScalarBase` without `ElementOpApply`, `einsum2_naive_into`, kernel bound relaxation, tests with local custom types
- **Out of scope:** tropical-gemm integration (separate PR), `EinsumScalar`/`EinsumOperand` extension for custom types, `FastGemm` trait, benchmarks
