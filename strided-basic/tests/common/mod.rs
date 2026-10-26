//! Shared test helpers: a copyable thread-budget description that mints a fresh
//! `Exec` token for each call, since `Exec` is neither `Clone` nor `Copy`.

#![allow(dead_code)]

use std::sync::OnceLock;

use strided_basic::{Exec, Pool};

/// A four-slot pool shared by the tests in one binary.
pub fn shared_pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        Pool::builder()
            .dispatch_costs(cheap_dispatch_costs())
            .threads(4)
            .build()
            .unwrap()
    })
}

/// Which token a test call receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Budget {
    /// `Exec::serial()`: the calling thread only.
    Serial,
    /// `Exec::new(shared_pool(), n)`: up to `n` slots of the shared pool.
    Pooled(usize),
}

impl Budget {
    /// Mints a fresh token for one call.
    pub fn exec(self) -> Exec<'static> {
        match self {
            Budget::Serial => Exec::serial(),
            Budget::Pooled(n) => Exec::new(shared_pool(), n).unwrap(),
        }
    }
}

/// Dispatch costs of the test pools: 100 ns per section whatever the token, so that the
/// regions of tens of thousands of elements these tests use open the parallel branches
/// (the host default costs would keep them serial).
fn cheap_dispatch_costs() -> threadteam::DispatchCosts {
    use threadteam::{DispatchKind, WidthCosts};
    let cheap = WidthCosts::flat(std::time::Duration::from_nanos(100));
    threadteam::DispatchCosts::for_class(threadteam::HostClass::Generic)
        .with(DispatchKind::Team, cheap)
        .with(DispatchKind::InPool, cheap)
        .with(DispatchKind::Outside, cheap)
}
