//! Backend abstraction for batched GEMM dispatch.
//!
//! This module defines the [`Backend`] trait, marker structs for each backend,
//! and the [`ActiveBackend`] type alias that serves as the single point of
//! backend selection based on Cargo features.

/// Trait for backends that can execute batched GEMM on contiguous operands.
///
/// Each backend declares its configuration (conjugation materialization,
/// stride requirements) and provides a GEMM implementation.
///
/// Implementations are provided by each backend module (faer, blas).
/// External crates can implement this trait for custom scalar types
/// (e.g., tropical semiring) and pass the backend to [`einsum2_with_backend_into`].
///
/// [`einsum2_with_backend_into`]: crate::einsum2_with_backend_into
pub trait Backend<T: crate::ScalarBase> {
    /// Whether the backend needs conjugation materialized into the data
    /// before GEMM (e.g., CBLAS has no conjugation flag for `?gemm`).
    const MATERIALIZES_CONJ: bool;

    /// Whether the backend requires at least one unit stride per matrix
    /// dimension (row or column stride must be 1). CBLAS `?gemm` requires
    /// this; faer does not.
    const REQUIRES_UNIT_STRIDE: bool;

    /// Execute batched GEMM: `C = alpha * A * B + beta * C` for each batch.
    ///
    /// - `c`: mutable output operand (batch x m x n)
    /// - `a`: input operand (batch x m x k)
    /// - `b`: input operand (batch x k x n)
    /// - `batch_dims`: sizes of the batch dimensions
    /// - `m`, `n`, `k`: fused matrix dimensions
    /// - `alpha`, `beta`: scaling factors
    fn bgemm_contiguous_into(
        c: &mut crate::contiguous::ContiguousOperandMut<T>,
        a: &crate::contiguous::ContiguousOperand<T>,
        b: &crate::contiguous::ContiguousOperand<T>,
        batch_dims: &[usize],
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        beta: T,
    ) -> strided_view::Result<()>;
}

// ---------------------------------------------------------------------------
// Marker structs
// ---------------------------------------------------------------------------

/// Batched GEMM backend using the [`faer`] library.
///
/// `Backend<T>` is implemented in `bgemm_faer.rs`.
#[cfg(feature = "faer")]
pub struct FaerBackend;

/// Batched GEMM backend using CBLAS (via `cblas-sys` or `cblas-inject`).
///
/// `Backend<T>` is implemented in `bgemm_blas.rs`.
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub struct BlasBackend;

/// Fallback batched GEMM backend using explicit loops (no external library).
///
/// This backend is used as `ActiveBackend` when no GEMM feature is enabled.
/// The GEMM dispatch in `einsum2_into` calls `bgemm_naive` directly rather
/// than going through the `Backend` trait, so `bgemm_contiguous_into` is
/// unreachable.
#[allow(dead_code)]
pub struct NaiveBackend;

impl<T: crate::ScalarBase> Backend<T> for NaiveBackend {
    const MATERIALIZES_CONJ: bool = false;
    const REQUIRES_UNIT_STRIDE: bool = false;

    fn bgemm_contiguous_into(
        _c: &mut crate::contiguous::ContiguousOperandMut<T>,
        _a: &crate::contiguous::ContiguousOperand<T>,
        _b: &crate::contiguous::ContiguousOperand<T>,
        _batch_dims: &[usize],
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: T,
        _beta: T,
    ) -> strided_view::Result<()> {
        unreachable!("NaiveBackend GEMM is dispatched directly, not through Backend trait")
    }
}

// ---------------------------------------------------------------------------
// ActiveBackend type alias -- the SINGLE point of backend selection
// ---------------------------------------------------------------------------

/// The active GEMM backend, selected by Cargo features.
///
/// - `faer` (without blas/blas-inject) -> [`FaerBackend`]
/// - `blas` or `blas-inject` (without faer) -> [`BlasBackend`]
/// - no backend feature -> [`NaiveBackend`]
/// - invalid combos -> [`NaiveBackend`] (placeholder; `compile_error!` fires first)
#[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
pub type ActiveBackend = FaerBackend;

#[cfg(all(
    not(feature = "faer"),
    any(
        all(feature = "blas", not(feature = "blas-inject")),
        all(feature = "blas-inject", not(feature = "blas"))
    )
))]
pub type ActiveBackend = BlasBackend;

#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub type ActiveBackend = NaiveBackend;

/// Placeholder for invalid mutually-exclusive feature combinations.
///
/// The crate emits `compile_error!` for these combinations (in `lib.rs`), so this
/// alias only suppresses cascading type-resolution errors.
#[cfg(any(
    all(feature = "faer", any(feature = "blas", feature = "blas-inject")),
    all(feature = "blas", feature = "blas-inject")
))]
pub type ActiveBackend = NaiveBackend;
