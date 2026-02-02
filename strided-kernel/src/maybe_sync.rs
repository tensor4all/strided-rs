//! Feature-gated Send/Sync marker traits.
//!
//! When the `parallel` feature is enabled, [`MaybeSend`] ≡ [`Send`],
//! [`MaybeSync`] ≡ [`Sync`], and [`MaybeSendSync`] ≡ [`Send`] + [`Sync`].
//!
//! When `parallel` is disabled, these traits are blanket-implemented
//! for all types, so non-thread-safe types can use the kernel APIs.

// ---- parallel enabled: alias to real Send/Sync ----

#[cfg(feature = "parallel")]
pub trait MaybeSend: Send {}
#[cfg(feature = "parallel")]
impl<T: Send> MaybeSend for T {}

#[cfg(feature = "parallel")]
pub trait MaybeSync: Sync {}
#[cfg(feature = "parallel")]
impl<T: Sync> MaybeSync for T {}

#[cfg(feature = "parallel")]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(feature = "parallel")]
impl<T: Send + Sync> MaybeSendSync for T {}

// ---- parallel disabled: blanket impl for all types ----

#[cfg(not(feature = "parallel"))]
pub trait MaybeSend {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSend for T {}

#[cfg(not(feature = "parallel"))]
pub trait MaybeSync {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSync for T {}

#[cfg(not(feature = "parallel"))]
pub trait MaybeSendSync {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSendSync for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_satisfies_maybe_traits() {
        fn _check_send<T: MaybeSend>() {}
        fn _check_sync<T: MaybeSync>() {}
        fn _check_send_sync<T: MaybeSendSync>() {}
        _check_send::<f64>();
        _check_sync::<f64>();
        _check_send_sync::<f64>();
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_rc_satisfies_maybe_traits_without_parallel() {
        use std::rc::Rc;
        fn _check_send<T: MaybeSend>() {}
        fn _check_sync<T: MaybeSync>() {}
        fn _check_send_sync<T: MaybeSendSync>() {}
        _check_send::<Rc<f64>>();
        _check_sync::<Rc<f64>>();
        _check_send_sync::<Rc<f64>>();
    }
}
