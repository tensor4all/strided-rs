//! Output layout validation shared by kernel families.
//!
//! The injectivity analysis lives in [`strided_view::injectivity`], the lowest
//! crate both `strided-basic` and `strided-perm` depend on, so there is one
//! owner for the validator. This module re-exports it under the path the
//! kernel families already use.

pub use strided_view::is_injective_layout;
pub(crate) use strided_view::is_injective_layout_without_alloc;
