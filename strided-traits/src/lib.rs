//! Shared traits for the strided-rs ecosystem.
//!
//! This crate provides the core trait definitions that are shared across
//! `strided-view`, `strided-kernel`, `strided-einsum2`, and external crates
//! (e.g., `tropical-gemm`).
//!
//! External crates can depend on `strided-traits` to implement traits for
//! their types without orphan rule violations.

pub mod element_op;
pub mod scalar;

pub use element_op::{
    Adjoint, ComposableElementOp, Compose, Conj, ElementOp, ElementOpApply, Identity, Transpose,
};
pub use scalar::ScalarBase;
