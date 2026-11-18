//! Private float abstraction and fixed-order lane reductions shared by the row kernels.
//!
//! Every reduction uses eight independent accumulators combined in a fixed tree, so a
//! sum has one order regardless of the ISA level, the token, or the thread count.

use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

pub(crate) trait Real:
    Copy
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    const ONE: Self;
    fn from_usize(n: usize) -> Self;
    fn sqrt(self) -> Self;
    /// Vectorized `exp(src)` into `dst`, or `false` when the slice lengths differ.
    fn exp_into(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> bool;
}

macro_rules! real {
    ($t:ident) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline(always)]
            fn from_usize(n: usize) -> Self {
                n as $t
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            fn exp_into(src: &[Self], dst: &mut [MaybeUninit<Self>]) -> bool {
                elem_simd_math::$t::exp(src, dst).is_ok()
            }
        }
    };
}
real!(f32);
real!(f64);

/// `sum_i f(i)` for `i in 0..n` over eight fixed lanes.
#[inline(always)]
pub(crate) fn lane_sum<T: Real>(n: usize, f: impl Fn(usize) -> T) -> T {
    let mut acc = [T::ZERO; 8];
    let body = n - n % 8;
    let mut i = 0;
    while i < body {
        for (k, a) in acc.iter_mut().enumerate() {
            *a += f(i + k);
        }
        i += 8;
    }
    for (k, a) in acc.iter_mut().enumerate().take(n - body) {
        *a += f(body + k);
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}
