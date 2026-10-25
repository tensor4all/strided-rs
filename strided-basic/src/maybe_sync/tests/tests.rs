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
