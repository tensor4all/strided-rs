#![doc = include_str!("../README.md")]

/// Shared scalar and element-operation traits.
pub mod traits {
    pub use strided_traits::*;
}

/// Strided view and owned strided array types.
pub mod view {
    pub use strided_view::*;
}

/// Cache-efficient tensor permutation and transpose routines.
pub mod perm {
    pub use strided_perm::*;
}

/// Cache-optimized map, reduce, and elementwise kernels.
pub mod kernel {
    pub use strided_kernel::*;
}

/// Binary einsum contractions on strided views.
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub mod einsum2 {
    pub use strided_einsum2::*;
}

/// N-ary optimized einsum frontend.
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub mod opteinsum {
    pub use strided_opteinsum::*;
}

/// `mdarray` einsum frontend.
#[cfg(feature = "mdarray")]
pub mod mdarray {
    pub use mdarray_opteinsum::*;
}

/// `ndarray` einsum frontend.
#[cfg(feature = "ndarray")]
pub mod ndarray {
    pub use ndarray_opteinsum::*;
}

pub use strided_kernel::{
    add, axpy, copy_conj, copy_into, copy_into_col_major, copy_scale, copy_transpose_scale_into,
    dot, fma, map_into, mul, reduce, reduce_axis, sum, symmetrize_conj_into, symmetrize_into,
    zip_map2_into, zip_map3_into, zip_map4_into, MaybeSimdOps,
};
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub use strided_opteinsum::{
    einsum, einsum_into, einsum_into_with_pool, einsum_with_pool, BufferPool, EinsumCode,
    EinsumError, EinsumNode, EinsumOperand, EinsumScalar, StridedData, TypedTensor,
};
pub use strided_traits::{
    Adjoint, ComposableElementOp, Compose, Conj, ElementOp, ElementOpApply, Identity, ScalarBase,
    Transpose,
};
pub use strided_view::{
    col_major_strides, row_major_strides, StridedArray, StridedError, StridedView, StridedViewMut,
};
