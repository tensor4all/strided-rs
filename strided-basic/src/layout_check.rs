//! Output layout validation shared by kernel families.
//!
//! The injectivity analysis lives in [`strided_view::injectivity`], the lowest
//! crate both `strided-basic` and `strided-perm` depend on, so there is one
//! owner for the validator. This module re-exports it under the path the
//! kernel families already use.
//!
//! [`is_injective_layout`] already runs the allocation-free pairwise check
//! first and spills to the exact enumeration only when the interleaved block
//! exceeds the pairwise budget, so a successful validation on the common
//! small/separated layouts does not allocate (#306 P1).

pub use strided_view::is_injective_layout;
pub(crate) use strided_view::is_injective_layout_without_alloc;
