//! Backend abstraction for batched GEMM dispatch.
//!
//! This module defines the [`BackendConfig`] and [`BgemmBackend`] traits, marker
//! structs for each backend, and the [`ActiveBackend`] type alias that serves as
//! the single point of backend selection based on Cargo features.

/// Static configuration for a GEMM backend.
///
/// Each backend declares its requirements so that operand preparation can
/// adapt without per-call `cfg` checks.
pub trait BackendConfig {
    /// Whether the backend needs conjugation materialized into the data
    /// before GEMM (e.g., CBLAS has no conjugation flag for `?gemm`).
    const MATERIALIZES_CONJ: bool;

    /// Whether the backend requires at least one unit stride per matrix
    /// dimension (row or column stride must be 1). CBLAS `?gemm` requires
    /// this; faer does not.
    const REQUIRES_UNIT_STRIDE: bool;
}

/// Trait for backends that can execute batched GEMM on contiguous operands.
///
/// Implementations are provided by each backend module (faer, blas).
/// External crates can implement this trait for custom scalar types
/// (e.g., tropical semiring) and pass the backend to [`einsum2_with_backend_into`].
///
/// [`einsum2_with_backend_into`]: crate::einsum2_with_backend_into
pub trait BgemmBackend<T: crate::ScalarBase> {
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
// Marker structs and BackendConfig implementations
// ---------------------------------------------------------------------------

/// Batched GEMM backend using the [`faer`] library.
#[cfg(feature = "faer")]
pub struct FaerBackend;

#[cfg(feature = "faer")]
impl BackendConfig for FaerBackend {
    const MATERIALIZES_CONJ: bool = false;
    const REQUIRES_UNIT_STRIDE: bool = false;
}

/// Batched GEMM backend using CBLAS (via `cblas-sys` or `cblas-inject`).
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub struct BlasBackend;

#[cfg(any(feature = "blas", feature = "blas-inject"))]
impl BackendConfig for BlasBackend {
    const MATERIALIZES_CONJ: bool = true;
    const REQUIRES_UNIT_STRIDE: bool = true;
}

/// Fallback batched GEMM backend using explicit loops (no external library).
#[allow(dead_code)]
pub struct NaiveBackend;

impl BackendConfig for NaiveBackend {
    const MATERIALIZES_CONJ: bool = false;
    const REQUIRES_UNIT_STRIDE: bool = false;
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
