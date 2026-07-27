//! Dtype-erased prepared kernel entry points.
//!
//! These wrappers keep dtype-specific monomorphization inside `strided-kernel`
//! so downstream runtime crates can replay prepared kernels through stable,
//! non-generic entry points.
//!
//! C ABI symbols are intentionally out of scope here. A future ABI layer must
//! pass an explicit execution context, preserve the non-overlap contract for
//! descriptors used by one replay call, and validate ABI dtype tags before
//! constructing these Rust descriptors.
//!
//! The safe Rust descriptor constructors validate dtype byte layout up front.
//! Mutable erased descriptors only re-scan value-constrained dtypes, currently
//! `bool`, after their raw bytes have escaped through `data_mut`.

use num_complex::{Complex32, Complex64};

use crate::{
    CopyPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, KernelDType, RawStridedMut,
    RawStridedRef, Result, StridedError,
};

/// Dtype-erased wrapper around [`CopyPlan`].
#[derive(Clone, Debug)]
pub struct ErasedCopyPlan {
    dtype: KernelDType,
    plan: CopyPlan,
}

impl ErasedCopyPlan {
    /// Compile a copy plan for one dtype and layout pair.
    pub fn compile(
        dtype: KernelDType,
        dims: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
    ) -> Result<Self> {
        Ok(Self {
            dtype,
            plan: CopyPlan::compile(dims, dst_strides, src_strides)?,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    /// `dest = src` through a non-generic dtype-erased replay boundary.
    pub fn execute(
        &self,
        _ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        self.check_dtype(dest.dtype())?;
        self.check_dtype(src.dtype())?;
        dest.validate_data_if_needed()?;

        let result = match self.dtype {
            KernelDType::F32 => execute_copy::<f32>(&self.plan, dest, src),
            KernelDType::F64 => execute_copy::<f64>(&self.plan, dest, src),
            KernelDType::I32 => execute_copy::<i32>(&self.plan, dest, src),
            KernelDType::I64 => execute_copy::<i64>(&self.plan, dest, src),
            KernelDType::Bool => execute_copy::<bool>(&self.plan, dest, src),
            KernelDType::C32 => execute_copy::<Complex32>(&self.plan, dest, src),
            KernelDType::C64 => execute_copy::<Complex64>(&self.plan, dest, src),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        };
        if result.is_ok() {
            // SAFETY: `execute_copy` only writes values produced from the
            // already-validated source descriptor for the same dtype.
            unsafe {
                dest.assume_data_valid();
            }
        }
        result
    }

    fn check_dtype(&self, actual: KernelDType) -> Result<()> {
        if actual != self.dtype {
            return Err(StridedError::DTypeMismatch {
                expected: self.dtype.label(),
                actual: actual.label(),
            });
        }
        Ok(())
    }
}

fn execute_copy<T>(
    plan: &CopyPlan,
    dest: &mut ErasedRawStridedMut<'_>,
    src: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync,
{
    let source_data = typed_slice::<T>(src.data());
    let dest_dims = dest.dims();
    let dest_strides = dest.strides();
    let dest_offset = dest.offset();
    let dest_data = typed_slice_mut::<T>(dest.data_mut());
    let source = unsafe {
        RawStridedRef::new_unchecked(source_data, src.dims(), src.strides(), src.offset())
    };
    let mut dest =
        unsafe { RawStridedMut::new_unchecked(dest_data, dest_dims, dest_strides, dest_offset) };
    plan.execute(&mut dest, &source)
}

fn typed_slice<T>(bytes: &[u8]) -> &[T] {
    if bytes.is_empty() {
        return &[];
    }
    unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr().cast::<T>(),
            bytes.len() / core::mem::size_of::<T>(),
        )
    }
}

fn typed_slice_mut<T>(bytes: &mut [u8]) -> &mut [T] {
    unsafe {
        core::slice::from_raw_parts_mut(
            if bytes.is_empty() {
                core::ptr::NonNull::<T>::dangling().as_ptr()
            } else {
                bytes.as_mut_ptr().cast::<T>()
            },
            bytes.len() / core::mem::size_of::<T>(),
        )
    }
}
