//! Feature-gated Send/Sync marker traits.
//!
//! When the `parallel` feature is enabled, [`MaybeSend`] ≡ [`Send`],
//! [`MaybeSync`] ≡ [`Sync`], and [`MaybeSendSync`] ≡ [`Send`] + [`Sync`].
//!
//! When `parallel` is disabled, these traits are blanket-implemented
//! for all types, so non-thread-safe types can use the kernel APIs.

// ---- parallel enabled: alias to real Send/Sync ----

/// Equivalent to [`Send`] when `parallel` is enabled; blanket-impl otherwise.
#[cfg(feature = "parallel")]
pub trait MaybeSend: Send {}
#[cfg(feature = "parallel")]
impl<T: Send> MaybeSend for T {}

/// Equivalent to [`Sync`] when `parallel` is enabled; blanket-impl otherwise.
#[cfg(feature = "parallel")]
pub trait MaybeSync: Sync {}
#[cfg(feature = "parallel")]
impl<T: Sync> MaybeSync for T {}

/// Equivalent to [`Send`] + [`Sync`] when `parallel` is enabled; blanket-impl otherwise.
#[cfg(feature = "parallel")]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(feature = "parallel")]
impl<T: Send + Sync> MaybeSendSync for T {}

// ---- parallel disabled: blanket impl for all types ----

/// Equivalent to [`Send`] when `parallel` is enabled; blanket-impl otherwise.
#[cfg(not(feature = "parallel"))]
pub trait MaybeSend {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSend for T {}

/// Equivalent to [`Sync`] when `parallel` is enabled; blanket-impl otherwise.
#[cfg(not(feature = "parallel"))]
pub trait MaybeSync {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSync for T {}

/// Equivalent to [`Send`] + [`Sync`] when `parallel` is enabled; blanket-impl otherwise.
#[cfg(not(feature = "parallel"))]
pub trait MaybeSendSync {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSendSync for T {}

#[cfg(test)]
#[path = "maybe_sync/tests/tests.rs"]
mod tests;
