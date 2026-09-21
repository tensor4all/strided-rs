#[inline(always)]
pub(crate) fn dispatch<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "simd")]
    {
        pulp::Arch::new().dispatch(f)
    }
    #[cfg(not(feature = "simd"))]
    {
        f()
    }
}

#[inline(always)]
pub(crate) fn dispatch_if_large<R>(len: usize, f: impl FnOnce() -> R) -> R {
    // Avoid runtime-dispatch overhead for tiny loops (especially common for small-array cases).
    // This is a heuristic; correctness does not depend on it.
    if len >= 64 {
        dispatch(f)
    } else {
        f()
    }
}

#[cfg(feature = "simd")]
#[inline(always)]
unsafe fn cast_slice<T, U>(src: &[T]) -> &[U] {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<U>(), src.len()) }
}

#[cfg(feature = "simd")]
macro_rules! impl_simd_mul_ptr {
    (
        $mul_into:ident,
        $ty:ty,
        $split:ident,
        $split_uninit:ident,
        $load:ident,
        $mask:ident,
        $store_ptr:ident,
        $mul:ident,
        $mask_scale:expr
    ) => {
        unsafe fn $mul_into(dst: *mut $ty, len: usize, a: &[$ty], b: &[$ty]) {
            struct Mul<'a> {
                dst: *mut $ty,
                len: usize,
                a: &'a [$ty],
                b: &'a [$ty],
            }

            impl<'a> pulp::WithSimd for Mul<'a> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    debug_assert_eq!(self.len, self.a.len());
                    debug_assert_eq!(self.len, self.b.len());

                    // SAFETY: the caller supplies len writable, nonaliasing
                    // elements. MaybeUninit does not assert initialized output.
                    let output = unsafe {
                        core::slice::from_raw_parts_mut(
                            self.dst.cast::<core::mem::MaybeUninit<$ty>>(),
                            self.len,
                        )
                    };
                    let (dest, tail) = S::$split_uninit(output);
                    let (a, a_tail) = S::$split(self.a);
                    let (b, b_tail) = S::$split(self.b);
                    // Full vectors use ordinary stores. Pulp's partial-access
                    // machinery is needed only for the final incomplete vector.
                    for ((dst, &a), &b) in dest.iter_mut().zip(a).zip(b) {
                        dst.write(simd.$mul(a, b));
                    }
                    if !tail.is_empty() {
                        let va = simd.$load(a_tail);
                        let vb = simd.$load(b_tail);
                        // SAFETY: the mask writes exactly the remaining lanes.
                        unsafe {
                            simd.$store_ptr(
                                simd.$mask(0, (tail.len() * $mask_scale) as _),
                                tail.as_mut_ptr().cast::<$ty>(),
                                simd.$mul(va, vb),
                            );
                        }
                    }
                }
            }

            pulp::Arch::new().dispatch(Mul { dst, len, a, b });
        }
    };
}

#[cfg(feature = "simd")]
impl_simd_mul_ptr!(
    simd_mul_f32_into,
    f32,
    as_simd_f32s,
    as_uninit_mut_simd_f32s,
    partial_load_f32s,
    mask_between_m32s,
    mask_store_ptr_f32s,
    mul_f32s,
    1
);

#[cfg(feature = "simd")]
impl_simd_mul_ptr!(
    simd_mul_f64_into,
    f64,
    as_simd_f64s,
    as_uninit_mut_simd_f64s,
    partial_load_f64s,
    mask_between_m64s,
    mask_store_ptr_f64s,
    mul_f64s,
    1
);

#[cfg(feature = "simd")]
impl_simd_mul_ptr!(
    simd_mul_c32_into,
    num_complex::Complex32,
    as_simd_c32s,
    as_uninit_mut_simd_c32s,
    partial_load_c32s,
    mask_between_m32s,
    mask_store_ptr_c32s,
    mul_e_c32s,
    2
);

#[cfg(feature = "simd")]
impl_simd_mul_ptr!(
    simd_mul_c64_into,
    num_complex::Complex64,
    as_simd_c64s,
    as_uninit_mut_simd_c64s,
    partial_load_c64s,
    mask_between_m64s,
    mask_store_ptr_c64s,
    mul_e_c64s,
    2
);

#[cfg(feature = "simd")]
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn try_mul_contiguous<D: 'static, A: 'static, B: 'static>(
    dst: &mut [D],
    a: &[A],
    b: &[B],
) -> bool {
    unsafe { try_mul_contiguous_ptr(dst.as_mut_ptr(), dst.len(), a, b) }
}

#[cfg(feature = "simd")]
#[inline]
pub(crate) unsafe fn try_mul_contiguous_ptr<D: 'static, A: 'static, B: 'static>(
    dst: *mut D,
    len: usize,
    a: &[A],
    b: &[B],
) -> bool {
    use std::any::TypeId;

    macro_rules! try_same_type {
        ($ty:ty, $mul_into:ident) => {
            if TypeId::of::<D>() == TypeId::of::<$ty>()
                && TypeId::of::<A>() == TypeId::of::<$ty>()
                && TypeId::of::<B>() == TypeId::of::<$ty>()
            {
                unsafe { $mul_into(dst.cast::<$ty>(), len, cast_slice(a), cast_slice(b)) };
                return true;
            }
        };
    }

    try_same_type!(f64, simd_mul_f64_into);
    try_same_type!(f32, simd_mul_f32_into);
    try_same_type!(num_complex::Complex64, simd_mul_c64_into);
    try_same_type!(num_complex::Complex32, simd_mul_c32_into);

    false
}

#[cfg(not(feature = "simd"))]
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn try_mul_contiguous<D: 'static, A: 'static, B: 'static>(
    _dst: &mut [D],
    _a: &[A],
    _b: &[B],
) -> bool {
    false
}

#[cfg(not(feature = "simd"))]
#[inline]
pub(crate) unsafe fn try_mul_contiguous_ptr<D: 'static, A: 'static, B: 'static>(
    _dst: *mut D,
    _len: usize,
    _a: &[A],
    _b: &[B],
) -> bool {
    false
}

#[cfg(feature = "parallel")]
const TRANSPOSE_TILE: usize = 8;

#[cfg(feature = "parallel")]
unsafe fn mul_transposed_scalar_rhs_source_contiguous<T>(
    dst: *mut T,
    src: *const T,
    scalar: T,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) where
    T: Copy + std::ops::Mul<Output = T>,
{
    let mut inner0 = 0usize;

    while inner0 < inner_len {
        let inner_count = TRANSPOSE_TILE.min(inner_len - inner0);
        let mut row0 = 0usize;

        while row0 < row_len {
            let row_count = TRANSPOSE_TILE.min(row_len - row0);

            for inner in 0..inner_count {
                let src_base = src.offset((inner0 + inner) as isize * src_fast_stride);
                for row in 0..row_count {
                    let row_index = row0 + row;
                    let src_offset = row_index as isize * src_row_stride;
                    *dst.add(row_index * inner_len + inner0 + inner) =
                        *src_base.offset(src_offset) * scalar;
                }
            }

            row0 += TRANSPOSE_TILE;
        }

        inner0 += TRANSPOSE_TILE;
    }
}

#[cfg(feature = "parallel")]
unsafe fn mul_transposed_scalar_rhs_dst_contiguous<T>(
    dst: *mut T,
    src: *const T,
    scalar: T,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) where
    T: Copy + std::ops::Mul<Output = T>,
{
    for row in 0..row_len {
        let dst_base = dst.add(row * inner_len);
        let src_base = src.offset(row as isize * src_row_stride);
        for inner in 0..inner_len {
            *dst_base.add(inner) = *src_base.offset(inner as isize * src_fast_stride) * scalar;
        }
    }
}

#[cfg(feature = "parallel")]
unsafe fn mul_transposed_scalar_lhs_source_contiguous<T>(
    dst: *mut T,
    scalar: T,
    src: *const T,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) where
    T: Copy + std::ops::Mul<Output = T>,
{
    let mut inner0 = 0usize;

    while inner0 < inner_len {
        let inner_count = TRANSPOSE_TILE.min(inner_len - inner0);
        let mut row0 = 0usize;

        while row0 < row_len {
            let row_count = TRANSPOSE_TILE.min(row_len - row0);

            for inner in 0..inner_count {
                let src_base = src.offset((inner0 + inner) as isize * src_fast_stride);
                for row in 0..row_count {
                    let row_index = row0 + row;
                    let src_offset = row_index as isize * src_row_stride;
                    *dst.add(row_index * inner_len + inner0 + inner) =
                        scalar * *src_base.offset(src_offset);
                }
            }

            row0 += TRANSPOSE_TILE;
        }

        inner0 += TRANSPOSE_TILE;
    }
}

#[cfg(feature = "parallel")]
unsafe fn mul_transposed_scalar_lhs_dst_contiguous<T>(
    dst: *mut T,
    scalar: T,
    src: *const T,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) where
    T: Copy + std::ops::Mul<Output = T>,
{
    for row in 0..row_len {
        let dst_base = dst.add(row * inner_len);
        let src_base = src.offset(row as isize * src_row_stride);
        for inner in 0..inner_len {
            *dst_base.add(inner) = scalar * *src_base.offset(inner as isize * src_fast_stride);
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) unsafe fn mul_transposed_scalar_rhs_2d_typed<T>(
    dst: *mut T,
    src: *const T,
    scalar: T,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) where
    T: Copy + std::ops::Mul<Output = T>,
{
    if row_len > inner_len {
        unsafe {
            mul_transposed_scalar_rhs_source_contiguous(
                dst,
                src,
                scalar,
                inner_len,
                row_len,
                src_fast_stride,
                src_row_stride,
            );
        }
    } else {
        unsafe {
            mul_transposed_scalar_rhs_dst_contiguous(
                dst,
                src,
                scalar,
                inner_len,
                row_len,
                src_fast_stride,
                src_row_stride,
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) unsafe fn mul_transposed_scalar_lhs_2d_typed<T>(
    dst: *mut T,
    scalar: T,
    src: *const T,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) where
    T: Copy + std::ops::Mul<Output = T>,
{
    if row_len > inner_len {
        unsafe {
            mul_transposed_scalar_lhs_source_contiguous(
                dst,
                scalar,
                src,
                inner_len,
                row_len,
                src_fast_stride,
                src_row_stride,
            );
        }
    } else {
        unsafe {
            mul_transposed_scalar_lhs_dst_contiguous(
                dst,
                scalar,
                src,
                inner_len,
                row_len,
                src_fast_stride,
                src_row_stride,
            );
        }
    }
}

#[inline]
#[cfg(feature = "parallel")]
pub(crate) unsafe fn try_mul_transposed_scalar_rhs_2d<D: 'static, A: 'static, B: 'static>(
    dst: *mut D,
    src: *const A,
    scalar: *const B,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) -> bool {
    use std::any::TypeId;

    macro_rules! try_same_type {
        ($ty:ty) => {
            if TypeId::of::<D>() == TypeId::of::<$ty>()
                && TypeId::of::<A>() == TypeId::of::<$ty>()
                && TypeId::of::<B>() == TypeId::of::<$ty>()
            {
                unsafe {
                    mul_transposed_scalar_rhs_2d_typed(
                        dst.cast::<$ty>(),
                        src.cast::<$ty>(),
                        *scalar.cast::<$ty>(),
                        inner_len,
                        row_len,
                        src_fast_stride,
                        src_row_stride,
                    );
                }
                return true;
            }
        };
    }

    try_same_type!(f64);
    try_same_type!(f32);
    try_same_type!(num_complex::Complex64);
    try_same_type!(num_complex::Complex32);

    false
}

#[inline]
#[cfg(feature = "parallel")]
pub(crate) unsafe fn try_mul_transposed_scalar_lhs_2d<D: 'static, A: 'static, B: 'static>(
    dst: *mut D,
    scalar: *const A,
    src: *const B,
    inner_len: usize,
    row_len: usize,
    src_fast_stride: isize,
    src_row_stride: isize,
) -> bool {
    use std::any::TypeId;

    macro_rules! try_same_type {
        ($ty:ty) => {
            if TypeId::of::<D>() == TypeId::of::<$ty>()
                && TypeId::of::<A>() == TypeId::of::<$ty>()
                && TypeId::of::<B>() == TypeId::of::<$ty>()
            {
                unsafe {
                    mul_transposed_scalar_lhs_2d_typed(
                        dst.cast::<$ty>(),
                        *scalar.cast::<$ty>(),
                        src.cast::<$ty>(),
                        inner_len,
                        row_len,
                        src_fast_stride,
                        src_row_stride,
                    );
                }
                return true;
            }
        };
    }

    try_same_type!(f64);
    try_same_type!(f32);
    try_same_type!(num_complex::Complex64);
    try_same_type!(num_complex::Complex32);

    false
}

/// Trait for types that may have SIMD-accelerated sum/dot operations.
///
/// Default implementations return `None` (no SIMD available).
/// f32/f64 override these with SIMD kernels when the `simd` feature is enabled.
pub trait MaybeSimdOps: Copy + Sized {
    fn try_simd_sum(_src: &[Self]) -> Option<Self> {
        None
    }
    fn try_simd_dot(_a: &[Self], _b: &[Self]) -> Option<Self> {
        None
    }
}

pub(crate) trait MaybeSimdProduct: Copy + Sized {
    fn try_simd_product(_src: &[Self]) -> Option<Self> {
        None
    }
}

pub(crate) trait MaybeSimdSumSquares: Copy + Sized {
    fn try_simd_sum_squares(_src: &[Self]) -> Option<Self> {
        None
    }
}

// Default (no-op) impls for integer types and Complex
macro_rules! impl_no_simd {
    ($($t:ty),*) => {
        $(
            impl MaybeSimdOps for $t {}
            impl MaybeSimdProduct for $t {}
            impl MaybeSimdSumSquares for $t {}
        )*
    };
}

impl_no_simd!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl<T: num_traits::Num + Copy + Clone + std::ops::Neg<Output = T>> MaybeSimdOps
    for num_complex::Complex<T>
{
}
impl<T: num_traits::Num + Copy + Clone + std::ops::Neg<Output = T>> MaybeSimdProduct
    for num_complex::Complex<T>
{
}
impl<T: num_traits::Num + Copy + Clone + std::ops::Neg<Output = T>> MaybeSimdSumSquares
    for num_complex::Complex<T>
{
}

// f32/f64: SIMD-accelerated when feature enabled, no-op otherwise
#[cfg(not(feature = "simd"))]
impl MaybeSimdOps for f32 {}
#[cfg(not(feature = "simd"))]
impl MaybeSimdProduct for f32 {}
#[cfg(not(feature = "simd"))]
impl MaybeSimdSumSquares for f32 {}

#[cfg(not(feature = "simd"))]
impl MaybeSimdOps for f64 {}
#[cfg(not(feature = "simd"))]
impl MaybeSimdProduct for f64 {}
#[cfg(not(feature = "simd"))]
impl MaybeSimdSumSquares for f64 {}

#[cfg(feature = "simd")]
mod simd_impls {
    use super::{MaybeSimdOps, MaybeSimdProduct, MaybeSimdSumSquares};
    use pulp::{Simd, WithSimd};

    impl MaybeSimdOps for f32 {
        fn try_simd_sum(src: &[f32]) -> Option<f32> {
            struct Sum<'a>(&'a [f32]);
            impl<'a> WithSimd for Sum<'a> {
                type Output = f32;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f32s(self.0);

                    let mut acc0 = simd.splat_f32s(0.0);
                    let mut acc1 = simd.splat_f32s(0.0);
                    let mut acc2 = simd.splat_f32s(0.0);
                    let mut acc3 = simd.splat_f32s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.add_f32s(acc0, head[i]);
                        acc1 = simd.add_f32s(acc1, head[i + 1]);
                        acc2 = simd.add_f32s(acc2, head[i + 2]);
                        acc3 = simd.add_f32s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &v in &head[i..] {
                        acc0 = simd.add_f32s(acc0, v);
                    }

                    let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f32s(acc);
                    for &x in tail {
                        sum += x;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(Sum(src)))
        }

        fn try_simd_dot(a: &[f32], b: &[f32]) -> Option<f32> {
            try_simd_dot_f32(a, b)
        }
    }

    impl MaybeSimdProduct for f32 {
        fn try_simd_product(src: &[f32]) -> Option<f32> {
            struct Product<'a>(&'a [f32]);
            impl<'a> WithSimd for Product<'a> {
                type Output = f32;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f32s(self.0);

                    let mut acc0 = simd.splat_f32s(1.0);
                    let mut acc1 = simd.splat_f32s(1.0);
                    let mut acc2 = simd.splat_f32s(1.0);
                    let mut acc3 = simd.splat_f32s(1.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.mul_f32s(acc0, head[i]);
                        acc1 = simd.mul_f32s(acc1, head[i + 1]);
                        acc2 = simd.mul_f32s(acc2, head[i + 2]);
                        acc3 = simd.mul_f32s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &value in &head[i..] {
                        acc0 = simd.mul_f32s(acc0, value);
                    }

                    let acc = simd.mul_f32s(simd.mul_f32s(acc0, acc1), simd.mul_f32s(acc2, acc3));
                    let mut product = simd.reduce_product_f32s(acc);
                    for &value in tail {
                        product *= value;
                    }
                    product
                }
            }

            Some(pulp::Arch::new().dispatch(Product(src)))
        }
    }

    impl MaybeSimdSumSquares for f32 {
        fn try_simd_sum_squares(src: &[f32]) -> Option<f32> {
            struct SumSquares<'a>(&'a [f32]);
            impl<'a> WithSimd for SumSquares<'a> {
                type Output = f32;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f32s(self.0);
                    let mut acc0 = simd.splat_f32s(0.0);
                    let mut acc1 = simd.splat_f32s(0.0);
                    let mut acc2 = simd.splat_f32s(0.0);
                    let mut acc3 = simd.splat_f32s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        let square0 = simd.mul_f32s(head[i], head[i]);
                        let square1 = simd.mul_f32s(head[i + 1], head[i + 1]);
                        let square2 = simd.mul_f32s(head[i + 2], head[i + 2]);
                        let square3 = simd.mul_f32s(head[i + 3], head[i + 3]);
                        acc0 = simd.add_f32s(acc0, square0);
                        acc1 = simd.add_f32s(acc1, square1);
                        acc2 = simd.add_f32s(acc2, square2);
                        acc3 = simd.add_f32s(acc3, square3);
                        i += 4;
                    }
                    for &value in &head[i..] {
                        let square = simd.mul_f32s(value, value);
                        acc0 = simd.add_f32s(acc0, square);
                    }

                    let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f32s(acc);
                    for &value in tail {
                        let square = value * value;
                        sum += square;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(SumSquares(src)))
        }
    }

    fn try_simd_dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        struct Dot<'a> {
            a: &'a [f32],
            b: &'a [f32],
        }
        impl<'a> WithSimd for Dot<'a> {
            type Output = f32;

            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                debug_assert_eq!(self.a.len(), self.b.len());
                let (a_head, a_tail) = S::as_simd_f32s(self.a);
                let (b_head, b_tail) = S::as_simd_f32s(self.b);
                debug_assert_eq!(a_head.len(), b_head.len());
                debug_assert_eq!(a_tail.len(), b_tail.len());

                let mut acc0 = simd.splat_f32s(0.0);
                let mut acc1 = simd.splat_f32s(0.0);
                let mut acc2 = simd.splat_f32s(0.0);
                let mut acc3 = simd.splat_f32s(0.0);

                let mut i = 0usize;
                while i + 4 <= a_head.len() {
                    acc0 = simd.mul_add_f32s(a_head[i], b_head[i], acc0);
                    acc1 = simd.mul_add_f32s(a_head[i + 1], b_head[i + 1], acc1);
                    acc2 = simd.mul_add_f32s(a_head[i + 2], b_head[i + 2], acc2);
                    acc3 = simd.mul_add_f32s(a_head[i + 3], b_head[i + 3], acc3);
                    i += 4;
                }
                for j in i..a_head.len() {
                    acc0 = simd.mul_add_f32s(a_head[j], b_head[j], acc0);
                }

                let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
                let mut sum = simd.reduce_sum_f32s(acc);
                for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
                    sum += x * y;
                }
                sum
            }
        }

        Some(pulp::Arch::new().dispatch(Dot { a, b }))
    }

    impl MaybeSimdOps for f64 {
        fn try_simd_sum(src: &[f64]) -> Option<f64> {
            struct Sum<'a>(&'a [f64]);
            impl<'a> WithSimd for Sum<'a> {
                type Output = f64;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f64s(self.0);

                    let mut acc0 = simd.splat_f64s(0.0);
                    let mut acc1 = simd.splat_f64s(0.0);
                    let mut acc2 = simd.splat_f64s(0.0);
                    let mut acc3 = simd.splat_f64s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.add_f64s(acc0, head[i]);
                        acc1 = simd.add_f64s(acc1, head[i + 1]);
                        acc2 = simd.add_f64s(acc2, head[i + 2]);
                        acc3 = simd.add_f64s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &v in &head[i..] {
                        acc0 = simd.add_f64s(acc0, v);
                    }

                    let acc = simd.add_f64s(simd.add_f64s(acc0, acc1), simd.add_f64s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f64s(acc);
                    for &x in tail {
                        sum += x;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(Sum(src)))
        }

        fn try_simd_dot(a: &[f64], b: &[f64]) -> Option<f64> {
            try_simd_dot_f64(a, b)
        }
    }

    impl MaybeSimdProduct for f64 {
        fn try_simd_product(src: &[f64]) -> Option<f64> {
            struct Product<'a>(&'a [f64]);
            impl<'a> WithSimd for Product<'a> {
                type Output = f64;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f64s(self.0);

                    let mut acc0 = simd.splat_f64s(1.0);
                    let mut acc1 = simd.splat_f64s(1.0);
                    let mut acc2 = simd.splat_f64s(1.0);
                    let mut acc3 = simd.splat_f64s(1.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.mul_f64s(acc0, head[i]);
                        acc1 = simd.mul_f64s(acc1, head[i + 1]);
                        acc2 = simd.mul_f64s(acc2, head[i + 2]);
                        acc3 = simd.mul_f64s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &value in &head[i..] {
                        acc0 = simd.mul_f64s(acc0, value);
                    }

                    let acc = simd.mul_f64s(simd.mul_f64s(acc0, acc1), simd.mul_f64s(acc2, acc3));
                    let mut product = simd.reduce_product_f64s(acc);
                    for &value in tail {
                        product *= value;
                    }
                    product
                }
            }

            Some(pulp::Arch::new().dispatch(Product(src)))
        }
    }

    impl MaybeSimdSumSquares for f64 {
        fn try_simd_sum_squares(src: &[f64]) -> Option<f64> {
            struct SumSquares<'a>(&'a [f64]);
            impl<'a> WithSimd for SumSquares<'a> {
                type Output = f64;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f64s(self.0);
                    let mut acc0 = simd.splat_f64s(0.0);
                    let mut acc1 = simd.splat_f64s(0.0);
                    let mut acc2 = simd.splat_f64s(0.0);
                    let mut acc3 = simd.splat_f64s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        let square0 = simd.mul_f64s(head[i], head[i]);
                        let square1 = simd.mul_f64s(head[i + 1], head[i + 1]);
                        let square2 = simd.mul_f64s(head[i + 2], head[i + 2]);
                        let square3 = simd.mul_f64s(head[i + 3], head[i + 3]);
                        acc0 = simd.add_f64s(acc0, square0);
                        acc1 = simd.add_f64s(acc1, square1);
                        acc2 = simd.add_f64s(acc2, square2);
                        acc3 = simd.add_f64s(acc3, square3);
                        i += 4;
                    }
                    for &value in &head[i..] {
                        let square = simd.mul_f64s(value, value);
                        acc0 = simd.add_f64s(acc0, square);
                    }

                    let acc = simd.add_f64s(simd.add_f64s(acc0, acc1), simd.add_f64s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f64s(acc);
                    for &value in tail {
                        let square = value * value;
                        sum += square;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(SumSquares(src)))
        }
    }

    fn try_simd_dot_f64(a: &[f64], b: &[f64]) -> Option<f64> {
        struct Dot<'a> {
            a: &'a [f64],
            b: &'a [f64],
        }
        impl<'a> WithSimd for Dot<'a> {
            type Output = f64;

            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                debug_assert_eq!(self.a.len(), self.b.len());
                let (a_head, a_tail) = S::as_simd_f64s(self.a);
                let (b_head, b_tail) = S::as_simd_f64s(self.b);
                debug_assert_eq!(a_head.len(), b_head.len());
                debug_assert_eq!(a_tail.len(), b_tail.len());

                let mut acc0 = simd.splat_f64s(0.0);
                let mut acc1 = simd.splat_f64s(0.0);
                let mut acc2 = simd.splat_f64s(0.0);
                let mut acc3 = simd.splat_f64s(0.0);

                let mut i = 0usize;
                while i + 4 <= a_head.len() {
                    acc0 = simd.mul_add_f64s(a_head[i], b_head[i], acc0);
                    acc1 = simd.mul_add_f64s(a_head[i + 1], b_head[i + 1], acc1);
                    acc2 = simd.mul_add_f64s(a_head[i + 2], b_head[i + 2], acc2);
                    acc3 = simd.mul_add_f64s(a_head[i + 3], b_head[i + 3], acc3);
                    i += 4;
                }
                for j in i..a_head.len() {
                    acc0 = simd.mul_add_f64s(a_head[j], b_head[j], acc0);
                }

                let acc = simd.add_f64s(simd.add_f64s(acc0, acc1), simd.add_f64s(acc2, acc3));
                let mut sum = simd.reduce_sum_f64s(acc);
                for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
                    sum += x * y;
                }
                sum
            }
        }

        Some(pulp::Arch::new().dispatch(Dot { a, b }))
    }
}

#[cfg(test)]
#[path = "simd/tests/tests.rs"]
mod tests;
