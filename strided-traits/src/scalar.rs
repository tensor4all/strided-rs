//! Scalar type bounds for strided operations and einsum.

/// Shared trait bounds for all element types usable with einsum, independent
/// of GEMM backend.
///
/// Unlike the previous design, `ScalarBase` does **not** require
/// `ElementOpApply`. This allows custom scalar types (e.g., tropical
/// semiring types) to satisfy einsum bounds without implementing
/// conj/transpose/adjoint.
pub trait ScalarBase:
    Copy
    + Send
    + Sync
    + 'static
    + std::ops::Mul<Output = Self>
    + std::ops::Add<Output = Self>
    + num_traits::Zero
    + num_traits::One
    + PartialEq
{
}

impl<T> ScalarBase for T where
    T: Copy
        + Send
        + Sync
        + 'static
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq
{
}

#[cfg(test)]
mod tests;
