# BLAS Backend for strided-einsum2

**Issue:** #54
**Date:** 2026-02-07

## Goal

Add a BLAS (OpenBLAS, MKL, etc.) backend to strided-einsum2 alongside the existing faer backend, enabling fair benchmark comparisons with Julia OMEinsum and supporting function pointer injection for Julia/Python interop.

## Design Decisions

| Decision | Choice |
|----------|--------|
| FFI layer | `cblas-sys` with `extern crate` alias |
| Provider selection | `blas-src` (delegated to downstream) |
| Julia/Python interop | `cblas-inject` via same alias pattern |
| Backend exclusivity | Mutual exclusion: `faer` vs `blas` vs `blas-inject` |
| Supported types | `f64` + `Complex64` only |
| Entry point | Same `bgemm_contiguous_into` signature, reuse `ContiguousOperand` |
| Conjugation | Pre-conjugate during copy-in (`prepare_input_*`), CBLAS always `CblasNoTrans` |
| Default feature | `faer` (unchanged) |

## Feature Flags and Dependencies

```toml
# strided-einsum2/Cargo.toml
[features]
default = ["faer", "faer-traits"]
faer = ["dep:faer", "dep:faer-traits"]
blas = ["dep:cblas-sys", "dep:blas-src"]
blas-inject = ["dep:cblas-inject"]

[dependencies]
cblas-sys = { version = "0.1", optional = true }
blas-src = { version = "0.10", optional = true }
cblas-inject = { version = "0.1", optional = true }
```

## Mutual Exclusion

Enforced via `compile_error!` in `lib.rs`:

```rust
#[cfg(all(feature = "faer", feature = "blas"))]
compile_error!("Features `faer` and `blas` are mutually exclusive");

#[cfg(all(feature = "faer", feature = "blas-inject"))]
compile_error!("Features `faer` and `blas-inject` are mutually exclusive");

#[cfg(all(feature = "blas", feature = "blas-inject"))]
compile_error!("Features `blas` and `blas-inject` are mutually exclusive");
```

## Crate Alias Pattern

In `lib.rs`, alias whichever CBLAS crate is active so `bgemm_blas.rs` uses a unified `cblas::` path:

```rust
#[cfg(feature = "blas")]
extern crate cblas_sys as cblas;
#[cfg(feature = "blas-inject")]
extern crate cblas_inject as cblas;
```

## Scalar Trait

Three cfg variants (mutually exclusive by compile_error above):

- `#[cfg(feature = "faer")]` — adds `faer_traits::ComplexField` bound
- `#[cfg(any(feature = "blas", feature = "blas-inject"))]` — basic bounds only
- `#[cfg(not(any(...)))]` — basic bounds only (naive fallback)

The blas/blas-inject and naive variants share the same trait bounds. Type-level CBLAS dispatch uses a separate `BlasGemm` trait.

## New File: `bgemm_blas.rs`

### BlasGemm Trait

```rust
trait BlasGemm {
    unsafe fn cblas_gemm(
        layout: u32, trans_a: u32, trans_b: u32,
        m: i32, n: i32, k: i32,
        alpha: Self, a: *const Self, lda: i32,
        b: *const Self, ldb: i32,
        beta: Self, c: *mut Self, ldc: i32,
    );
}

impl BlasGemm for f64 { /* cblas::cblas_dgemm */ }
impl BlasGemm for Complex64 { /* cblas::cblas_zgemm */ }
```

### bgemm_contiguous_into

Same signature as `bgemm_faer::bgemm_contiguous_into`:

```rust
pub fn bgemm_contiguous_into<T: Scalar + BlasGemm>(
    c: &mut ContiguousOperandMut<T>,
    a: &ContiguousOperand<T>,
    b: &ContiguousOperand<T>,
    batch_dims: &[usize],
    m: usize, n: usize, k: usize,
    alpha: T, beta: T,
) -> strided_view::Result<()>
```

- Iterates over batch dimensions using `MultiIndex`
- Calls `BlasGemm::cblas_gemm` with `CblasColMajor` + `CblasNoTrans` per batch slice
- Conjugation is already resolved during copy-in (conj flag is always false)

## Conjugation Handling

In `contiguous.rs`, `prepare_input_view` and `prepare_input_owned` are modified:

- When `#[cfg(any(feature = "blas", feature = "blas-inject"))]` and `conj=true`:
  - If copy is already needed (non-contiguous): conjugate elements during copy, set `conj=false`
  - If contiguous but `conj=true`: force a copy-with-conjugation, set `conj=false`
- When `#[cfg(feature = "faer")]`: unchanged (conj flag passed through to `matmul_with_conj`)
- Result: BLAS backend always receives `conj=false` operands

## Integration in lib.rs

`einsum2_gemm_dispatch` is gated on `any(feature = "faer", feature = "blas", feature = "blas-inject")`. The GEMM call is the only cfg branch point:

```rust
fn einsum2_gemm_dispatch<T: Scalar>(...) -> Result<()> {
    // 1. Permute to canonical order (shared)
    // 2. Element-wise fast path (shared)
    // 3. Prepare contiguous operands (shared, conj resolved for blas)

    // 4. GEMM (backend branch)
    #[cfg(feature = "faer")]
    bgemm_faer::bgemm_contiguous_into(...)?;

    #[cfg(any(feature = "blas", feature = "blas-inject"))]
    bgemm_blas::bgemm_contiguous_into(...)?;

    // 5. Finalize (shared)
}
```

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` | Add optional deps: `cblas-sys`, `blas-src`, `cblas-inject` |
| `src/lib.rs` | Add compile_error!, extern crate alias, cfg gates for bgemm_blas module, Scalar trait variant |
| `src/bgemm_blas.rs` | **New** — `BlasGemm` trait + `bgemm_contiguous_into` |
| `src/contiguous.rs` | cfg-gated conj resolution in `prepare_input_view` / `prepare_input_owned` |

No changes to: `plan.rs`, `trace.rs`, `util.rs`, `bgemm_naive.rs`, `bgemm_faer.rs`.

## Testing

Existing tests cover all backends via feature flag switching:

```bash
cargo test -p strided-einsum2                                    # faer (default)
cargo test -p strided-einsum2 --no-default-features --features blas         # blas
cargo test -p strided-einsum2 --no-default-features --features blas-inject  # blas-inject
cargo test -p strided-einsum2 --no-default-features                         # naive
```

Additional: unit tests in `bgemm_blas.rs` for `BlasGemm` dispatch on `f64` / `Complex64`.

## Benchmarking

```bash
# Rust faer (current)
RAYON_NUM_THREADS=1 cargo bench -p strided-einsum2

# Rust BLAS (OpenBLAS)
RAYON_NUM_THREADS=1 cargo bench -p strided-einsum2 --no-default-features --features blas

# Julia OMEinsum (OpenBLAS)
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches ...
```
