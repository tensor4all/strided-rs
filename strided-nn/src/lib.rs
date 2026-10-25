#![doc = include_str!("../README.md")]

//! Fused neural-network kernels over contiguous strided CPU slices.
//!
//! The crate owns fused kernels whose contract is tensor-free: flat contiguous
//! storage, an explicit [`strided_basic::Exec`] token for placement, and
//! caller-provided scratch. Tensor semantics, dtype dispatch, allocation and
//! pooling stay with the host runtime.
//!
//! Every entry decides its worker count and chunking on the calling thread from
//! its arguments; nothing here creates a pool or sizes work from the ambient
//! Rayon registry. Under a serial token the kernel runs inline on the calling
//! thread.

mod error;
mod simd;
mod softmax;

pub use error::SoftmaxError;
pub use softmax::{scratch_len, softmax_axis0_f32, softmax_axis0_f64};
