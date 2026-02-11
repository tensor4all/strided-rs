//! Element-wise operations applied lazily to strided views.
//!
//! Re-exported from [`strided_traits`]. See that crate for full documentation.

pub use strided_traits::element_op::{
    Adjoint, ComposableElementOp, Compose, Conj, ElementOp, ElementOpApply, Identity, Transpose,
};
