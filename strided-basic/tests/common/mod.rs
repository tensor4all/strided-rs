//! Shared test helpers: a copyable thread-budget description that mints a fresh
//! `Exec` token for each call, since `Exec` is neither `Clone` nor `Copy`.

#![allow(dead_code)]

use std::sync::OnceLock;

use strided_basic::{Exec, Pool};

/// A four-slot pool shared by the tests in one binary.
pub fn shared_pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::builder().threads(4).build().unwrap())
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
