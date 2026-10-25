//! Send/Sync bound aliases used by the kernel APIs.
//!
//! Every parallel entry hands its closures to a `strided_exec::Exec`, which
//! requires `Send`/`Sync` whether or not the `parallel` feature runs them on a
//! pool. [`MaybeSend`] is therefore [`Send`], [`MaybeSync`] is [`Sync`] and
//! [`MaybeSendSync`] is [`Send`] + [`Sync`], in every feature configuration.

/// Alias of [`Send`].
pub trait MaybeSend: Send {}
impl<T: Send> MaybeSend for T {}

/// Alias of [`Sync`].
pub trait MaybeSync: Sync {}
impl<T: Sync> MaybeSync for T {}

/// Alias of [`Send`] + [`Sync`].
pub trait MaybeSendSync: Send + Sync {}
impl<T: Send + Sync> MaybeSendSync for T {}

#[cfg(test)]
#[path = "maybe_sync/tests/tests.rs"]
mod tests;
