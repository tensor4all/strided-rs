use crate::kernel::{
    build_plan_fused, ensure_same_shape, for_each_inner_block, for_each_offset, is_contiguous,
    total_len, StridedView, StridedViewMut,
};
use crate::broadcast::Consume;
use crate::element_op::{ElementOp, ElementOpApply, Identity};
use crate::promote::{
    broadcast_shape, broadcast_shape as broadcast_shape_dyn, promote_strides_to_shape,
};
use crate::view::{StridedArrayView, StridedArrayViewMut};
use crate::{Result, StridedError};
use mdarray::{DynRank, Layout, Shape, Slice, Tensor};

pub fn reduce<T, S, L, M, R, U>(src: &Slice<T, S, L>, map_fn: M, reduce_fn: R, init: U) -> Result<U>
where
    S: Shape,
    L: Layout,
    M: Fn(&T) -> U,
    R: Fn(U, U) -> U,
{
    let src_view = StridedView::from_slice(src)?;
    if is_contiguous(&src_view.dims, &src_view.strides) {
        let len = total_len(&src_view.dims);
        let mut ptr = src_view.ptr;
        let mut acc = init;
        for _ in 0..len {
            let mapped = map_fn(unsafe { &*ptr });
            acc = reduce_fn(acc, mapped);
            unsafe {
                ptr = ptr.add(1);
            }
        }
        return Ok(acc);
    }

    let strides_list = [&src_view.strides[..]];
    let (fused_dims, plan) =
        build_plan_fused(&src_view.dims, &strides_list, None, std::mem::size_of::<T>());

    let mut acc = Some(init);
    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut ptr = unsafe { src_view.ptr.offset(offsets[0]) };
        let stride = strides[0];
        for _ in 0..len {
            let mapped = map_fn(unsafe { &*ptr });
            let current = acc.take().ok_or(StridedError::OffsetOverflow)?;
            acc = Some(reduce_fn(current, mapped));
            unsafe {
                ptr = ptr.offset(stride);
            }
        }
        Ok(())
    })?;

    acc.ok_or(StridedError::OffsetOverflow)
}

pub fn reduce_axis<T, S, L, M, R, U>(
    src: &Slice<T, S, L>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<Tensor<U, DynRank>>
where
    S: Shape,
    L: Layout,
    M: Fn(&T) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    let src_view = StridedView::from_slice(src)?;
    let rank = src_view.dims.len();
    if axis >= rank {
        return Err(StridedError::InvalidAxis { axis, rank });
    }

    let out_dims: Vec<usize> = src_view
        .dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();

    let axis_len = src_view.dims[axis];
    let axis_stride = src_view.strides[axis];

    if out_dims.is_empty() {
        let mut acc = init;
        let mut offset = 0isize;
        for _ in 0..axis_len {
            let ptr = unsafe { src_view.ptr.offset(offset) };
            let mapped = map_fn(unsafe { &*ptr });
            acc = reduce_fn(acc, mapped);
            offset = offset
                .checked_add(axis_stride)
                .ok_or(StridedError::OffsetOverflow)?;
        }
        let out = Tensor::from_fn([1], |_| acc.clone()).into_dyn();
        return Ok(out);
    }

    let mut out = Tensor::from_fn(out_dims.as_slice(), |_| init.clone()).into_dyn();
    let mut in_strides_reduced = Vec::with_capacity(out_dims.len());
    for (i, &stride) in src_view.strides.iter().enumerate() {
        if i != axis {
            in_strides_reduced.push(stride);
        }
    }

    let in_reduced = StridedView::from_parts(src_view.ptr, &out_dims, &in_strides_reduced)?;
    let out_view = StridedViewMut::from_slice(&mut out)?;
    ensure_same_shape(&in_reduced.dims, &out_view.dims)?;

    let strides_list = [&in_reduced.strides[..], &out_view.strides[..]];
    let (fused_dims, plan) =
        build_plan_fused(&out_view.dims, &strides_list, Some(1), std::mem::size_of::<U>());

    for_each_offset(&fused_dims, &plan, &strides_list, |offsets| {
        let mut acc = init.clone();
        let mut offset = offsets[0];
        for _ in 0..axis_len {
            let ptr = unsafe { src_view.ptr.offset(offset) };
            let mapped = map_fn(unsafe { &*ptr });
            acc = reduce_fn(acc, mapped);
            offset = offset
                .checked_add(axis_stride)
                .ok_or(StridedError::OffsetOverflow)?;
        }
        let out_ptr = unsafe { out_view.ptr.offset(offsets[1]) };
        unsafe {
            *out_ptr = acc;
        }
        Ok(())
    })?;

    Ok(out)
}

/// Map-reduce along specified dimensions into a pre-allocated destination array.
///
/// This is the Rust equivalent of Julia's `Base.mapreducedim!`. It:
/// 1. Applies `map_fn` to each element of the source
/// 2. Reduces using `reduce_fn` along dimensions where `dest` has size 1
/// 3. Writes results into the pre-shaped destination array
///
/// # Arguments
/// - `dest`: Pre-allocated destination with reduced dimensions (size 1 for reduction dims)
/// - `src`: Source array to reduce
/// - `map_fn`: Function to map each element before reduction
/// - `reduce_fn`: Binary reduction operation (e.g., `|a, b| a + b`)
/// - `init_op`: Optional initialization operation applied to destination before reduction.
///   - `None`: accumulate into existing values (dest = dest + mapped)
///   - `Some(f)`: apply f to dest first (dest = f(dest)), then accumulate
///
/// # Shape Requirements
/// - `dest` and `src` must be broadcastable
/// - `dest` shape must match `src` shape except for reduction dims which must be 1
///
/// # Julia equivalent
/// ```julia
/// function Base.mapreducedim!(f, op, b::StridedView{<:Any,N},
///                             a1::StridedView{<:Any,N}, ...) where {N}
/// ```
///
/// # Example
/// ```ignore
/// // Sum along axis 1 of a 3x4 array into a 3x1 result
/// let mut dest = /* 3x1 array initialized to zero */;
/// let src = /* 3x4 array */;
/// mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None)?;
/// ```
pub fn mapreducedim_into<T, SD, SS, LD, LS, M, R>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
    map_fn: M,
    reduce_fn: R,
    init_op: Option<fn(&T) -> T>,
) -> Result<()>
where
    T: Clone,
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
    M: Fn(&T) -> T,
    R: Fn(T, T) -> T,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let src_view = StridedView::from_slice(src)?;

    // Compute broadcast-compatible dimensions (dest may have size-1 reduction dims)
    let dims = broadcast_shape_dyn(&[&dst_view.dims, &src_view.dims])?;

    let total_len = dims.iter().product::<usize>();
    if total_len == 0 {
        // Handle empty arrays - apply init_op if provided
        if let Some(init) = init_op {
            let dst_total = dst_view.dims.iter().product::<usize>();
            if dst_total > 0 {
                // Apply init_op to all dest elements
                for i in 0..dst_total {
                    let ptr = unsafe { dst_view.ptr.add(i) };
                    unsafe {
                        let val = init(&*ptr);
                        *ptr = val;
                    }
                }
            }
        }
        return Ok(());
    }

    // Build promoted strides for broadcast/reduction iteration
    // - Any size-1 dim in dest becomes stride-0 (reduction)
    // - Any size-1 dim in src becomes stride-0 (broadcast)
    let dst_promoted_strides =
        promote_strides_to_shape(&dims, &dst_view.dims, &dst_view.strides)?;
    let src_promoted_strides =
        promote_strides_to_shape(&dims, &src_view.dims, &src_view.strides)?;

    // If init_op is provided, apply it first to all dest elements
    if let Some(init) = init_op {
        let dst_total = dst_view.dims.iter().product::<usize>();
        if dst_total > 0 {
            apply_init_op(&dst_view, init)?;
        }
    }

    // Now iterate and perform the map-reduce
    let strides_list = [&dst_promoted_strides[..], &src_promoted_strides[..]];
    let (fused_dims, plan) =
        build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut dst_ptr = unsafe { dst_view.ptr.offset(offsets[0]) };
        let mut src_ptr = unsafe { src_view.ptr.offset(offsets[1]) };
        let dst_stride = strides[0];
        let src_stride = strides[1];

        // Check if this is a reduction (stride 0 for dest in inner loop)
        if dst_stride == 0 && len > 0 {
            // Reduction: accumulate all src elements into single dest
            let mut acc = unsafe { (*dst_ptr).clone() };
            for _ in 0..len {
                let mapped = map_fn(unsafe { &*src_ptr });
                acc = reduce_fn(acc, mapped);
                src_ptr = unsafe { src_ptr.offset(src_stride) };
            }
            unsafe {
                *dst_ptr = acc;
            }
        } else {
            // Non-reduction: apply map and combine with dest
            for _ in 0..len {
                let mapped = map_fn(unsafe { &*src_ptr });
                let current = unsafe { (*dst_ptr).clone() };
                let result = reduce_fn(current, mapped);
                unsafe {
                    *dst_ptr = result;
                    dst_ptr = dst_ptr.offset(dst_stride);
                    src_ptr = src_ptr.offset(src_stride);
                }
            }
        }
        Ok(())
    })
}

/// Apply initialization operation to all elements of a strided view.
fn apply_init_op<T: Clone>(view: &StridedViewMut<T>, init: fn(&T) -> T) -> Result<()> {
    if is_contiguous(&view.dims, &view.strides) {
        let len = total_len(&view.dims);
        let mut ptr = view.ptr;
        for _ in 0..len {
            unsafe {
                let val = init(&*ptr);
                *ptr = val;
                ptr = ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [&view.strides[..]];
    let (fused_dims, plan) =
        build_plan_fused(&view.dims, &strides_list, Some(0), std::mem::size_of::<T>());

    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut ptr = unsafe { view.ptr.offset(offsets[0]) };
        let stride = strides[0];
        for _ in 0..len {
            unsafe {
                let val = init(&*ptr);
                *ptr = val;
                ptr = ptr.offset(stride);
            }
        }
        Ok(())
    })
}

fn apply_init_op_view<T: Copy, const N: usize>(
    view: &mut StridedArrayViewMut<'_, T, N, Identity>,
    init: fn(&T) -> T,
) -> Result<()> {
    let dims: Vec<usize> = view.size().to_vec();
    let strides: Vec<isize> = view.strides().to_vec();

    if is_contiguous(&dims, &strides) {
        let len = total_len(&dims);
        let mut ptr = view.as_mut_ptr();
        for _ in 0..len {
            unsafe {
                let val = init(&*ptr);
                *ptr = val;
                ptr = ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [&strides[..]];
    let (fused_dims, plan) =
        build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    let base = view.as_mut_ptr();
    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, inner_strides| {
        let mut ptr = unsafe { base.offset(offsets[0]) };
        let step = inner_strides[0];
        for _ in 0..len {
            unsafe {
                let val = init(&*ptr);
                *ptr = val;
                ptr = ptr.offset(step);
            }
        }
        Ok(())
    })
}

/// View-based Julia-style `mapreducedim!` for an arbitrary number of source arrays.
///
/// - `dest` may have size-1 dimensions which represent reduction dimensions.
/// - `sources` are broadcasted to the common shape (stride-0) before iteration.
/// - `capture` is evaluated per element (can be nested, include scalars, etc.).
pub fn mapreducedim_capture_views_into<'a, T, const N: usize, Op, C, R>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    sources: &[&StridedArrayView<'a, T, N, Op>],
    capture: &C,
    reduce_fn: R,
    init_op: Option<fn(&T) -> T>,
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op: ElementOp,
    C: Consume<T, Output = T>,
    R: Fn(T, T) -> T,
{
    // Common broadcast shape between dest and all sources.
    let mut dims_list: Vec<&[usize]> = Vec::with_capacity(1 + sources.len());
    dims_list.push(&dest.size()[..]);
    for &src in sources {
        dims_list.push(&src.size()[..]);
    }
    let dims = broadcast_shape(&dims_list)?;

    if total_len(&dims) == 0 {
        if let Some(init) = init_op {
            apply_init_op_view(dest, init)?;
        }
        return Ok(());
    }

    if let Some(init) = init_op {
        apply_init_op_view(dest, init)?;
    }

    let dst_promoted_strides =
        promote_strides_to_shape(&dims, &dest.size()[..], &dest.strides()[..])?;

    let mut src_promoted_strides: Vec<Vec<isize>> = Vec::with_capacity(sources.len());
    for &src in sources {
        src_promoted_strides.push(promote_strides_to_shape(
            &dims,
            &src.size()[..],
            &src.strides()[..],
        )?);
    }

    let mut strides_list: Vec<&[isize]> = Vec::with_capacity(1 + sources.len());
    strides_list.push(&dst_promoted_strides[..]);
    for v in &src_promoted_strides {
        strides_list.push(v.as_slice());
    }

    let (fused_dims, plan) =
        build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    let num_src = sources.len();
    let dst_base = dest.as_mut_ptr();
    let src_bases: Vec<*const T> = sources.iter().map(|s| s.as_ptr()).collect();

    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, inner_strides| {
        let mut dst_ptr = unsafe { dst_base.offset(offsets[0]) };
        let dst_step = inner_strides[0];

        match num_src {
            0 => {
                if dst_step == 0 && len > 0 {
                    let mut acc = unsafe { *dst_ptr };
                    for _ in 0..len {
                        let mut it = [].into_iter();
                        let mapped = capture.consume(&mut it);
                        acc = reduce_fn(acc, mapped);
                    }
                    unsafe { *dst_ptr = acc };
                } else {
                    for _ in 0..len {
                        let mut it = [].into_iter();
                        let mapped = capture.consume(&mut it);
                        unsafe {
                            let current = *dst_ptr;
                            *dst_ptr = reduce_fn(current, mapped);
                            dst_ptr = dst_ptr.offset(dst_step);
                        }
                    }
                }
            }
            1 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let s0 = inner_strides[1];

                if dst_step == 0 && len > 0 {
                    let mut acc = unsafe { *dst_ptr };
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        p0 = unsafe { p0.offset(s0) };
                        let mut it = [v0].into_iter();
                        let mapped = capture.consume(&mut it);
                        acc = reduce_fn(acc, mapped);
                    }
                    unsafe { *dst_ptr = acc };
                } else {
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        p0 = unsafe { p0.offset(s0) };
                        let mut it = [v0].into_iter();
                        let mapped = capture.consume(&mut it);
                        unsafe {
                            let current = *dst_ptr;
                            *dst_ptr = reduce_fn(current, mapped);
                            dst_ptr = dst_ptr.offset(dst_step);
                        }
                    }
                }
            }
            2 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let mut p1 = unsafe { src_bases[1].offset(offsets[2]) };
                let s0 = inner_strides[1];
                let s1 = inner_strides[2];

                if dst_step == 0 && len > 0 {
                    let mut acc = unsafe { *dst_ptr };
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        let v1 = unsafe { Op::apply(*p1) };
                        p0 = unsafe { p0.offset(s0) };
                        p1 = unsafe { p1.offset(s1) };
                        let mut it = [v0, v1].into_iter();
                        let mapped = capture.consume(&mut it);
                        acc = reduce_fn(acc, mapped);
                    }
                    unsafe { *dst_ptr = acc };
                } else {
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        let v1 = unsafe { Op::apply(*p1) };
                        p0 = unsafe { p0.offset(s0) };
                        p1 = unsafe { p1.offset(s1) };
                        let mut it = [v0, v1].into_iter();
                        let mapped = capture.consume(&mut it);
                        unsafe {
                            let current = *dst_ptr;
                            *dst_ptr = reduce_fn(current, mapped);
                            dst_ptr = dst_ptr.offset(dst_step);
                        }
                    }
                }
            }
            3 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let mut p1 = unsafe { src_bases[1].offset(offsets[2]) };
                let mut p2 = unsafe { src_bases[2].offset(offsets[3]) };
                let s0 = inner_strides[1];
                let s1 = inner_strides[2];
                let s2 = inner_strides[3];

                if dst_step == 0 && len > 0 {
                    let mut acc = unsafe { *dst_ptr };
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        let v1 = unsafe { Op::apply(*p1) };
                        let v2 = unsafe { Op::apply(*p2) };
                        p0 = unsafe { p0.offset(s0) };
                        p1 = unsafe { p1.offset(s1) };
                        p2 = unsafe { p2.offset(s2) };
                        let mut it = [v0, v1, v2].into_iter();
                        let mapped = capture.consume(&mut it);
                        acc = reduce_fn(acc, mapped);
                    }
                    unsafe { *dst_ptr = acc };
                } else {
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        let v1 = unsafe { Op::apply(*p1) };
                        let v2 = unsafe { Op::apply(*p2) };
                        p0 = unsafe { p0.offset(s0) };
                        p1 = unsafe { p1.offset(s1) };
                        p2 = unsafe { p2.offset(s2) };
                        let mut it = [v0, v1, v2].into_iter();
                        let mapped = capture.consume(&mut it);
                        unsafe {
                            let current = *dst_ptr;
                            *dst_ptr = reduce_fn(current, mapped);
                            dst_ptr = dst_ptr.offset(dst_step);
                        }
                    }
                }
            }
            4 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let mut p1 = unsafe { src_bases[1].offset(offsets[2]) };
                let mut p2 = unsafe { src_bases[2].offset(offsets[3]) };
                let mut p3 = unsafe { src_bases[3].offset(offsets[4]) };
                let s0 = inner_strides[1];
                let s1 = inner_strides[2];
                let s2 = inner_strides[3];
                let s3 = inner_strides[4];

                if dst_step == 0 && len > 0 {
                    let mut acc = unsafe { *dst_ptr };
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        let v1 = unsafe { Op::apply(*p1) };
                        let v2 = unsafe { Op::apply(*p2) };
                        let v3 = unsafe { Op::apply(*p3) };
                        p0 = unsafe { p0.offset(s0) };
                        p1 = unsafe { p1.offset(s1) };
                        p2 = unsafe { p2.offset(s2) };
                        p3 = unsafe { p3.offset(s3) };
                        let mut it = [v0, v1, v2, v3].into_iter();
                        let mapped = capture.consume(&mut it);
                        acc = reduce_fn(acc, mapped);
                    }
                    unsafe { *dst_ptr = acc };
                } else {
                    for _ in 0..len {
                        let v0 = unsafe { Op::apply(*p0) };
                        let v1 = unsafe { Op::apply(*p1) };
                        let v2 = unsafe { Op::apply(*p2) };
                        let v3 = unsafe { Op::apply(*p3) };
                        p0 = unsafe { p0.offset(s0) };
                        p1 = unsafe { p1.offset(s1) };
                        p2 = unsafe { p2.offset(s2) };
                        p3 = unsafe { p3.offset(s3) };
                        let mut it = [v0, v1, v2, v3].into_iter();
                        let mapped = capture.consume(&mut it);
                        unsafe {
                            let current = *dst_ptr;
                            *dst_ptr = reduce_fn(current, mapped);
                            dst_ptr = dst_ptr.offset(dst_step);
                        }
                    }
                }
            }
            _ => {
                let mut src_ptrs: Vec<*const T> = Vec::with_capacity(num_src);
                let mut src_steps: Vec<isize> = Vec::with_capacity(num_src);
                let mut values: Vec<T> = Vec::with_capacity(num_src);

                src_ptrs.clear();
                src_steps.clear();
                for i in 0..num_src {
                    src_ptrs.push(unsafe { src_bases[i].offset(offsets[i + 1]) });
                    src_steps.push(inner_strides[i + 1]);
                }

                if dst_step == 0 && len > 0 {
                    let mut acc = unsafe { *dst_ptr };
                    for _ in 0..len {
                        values.clear();
                        for i in 0..num_src {
                            unsafe {
                                values.push(Op::apply(*src_ptrs[i]));
                                src_ptrs[i] = src_ptrs[i].offset(src_steps[i]);
                            }
                        }
                        let mut it = values.iter().copied();
                        let mapped = capture.consume(&mut it);
                        acc = reduce_fn(acc, mapped);
                    }
                    unsafe { *dst_ptr = acc };
                } else {
                    for _ in 0..len {
                        values.clear();
                        for i in 0..num_src {
                            unsafe {
                                values.push(Op::apply(*src_ptrs[i]));
                                src_ptrs[i] = src_ptrs[i].offset(src_steps[i]);
                            }
                        }
                        let mut it = values.iter().copied();
                        let mapped = capture.consume(&mut it);
                        unsafe {
                            let current = *dst_ptr;
                            *dst_ptr = reduce_fn(current, mapped);
                            dst_ptr = dst_ptr.offset(dst_step);
                        }
                    }
                }
            }
        }

        Ok(())
    })
}


// Intentionally omitted: dense strides helper not needed in v0.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broadcast::{Arg, CaptureArgs};
    use crate::element_op::Conj;
    use crate::view::{StridedArrayView, StridedArrayViewMut};
    use mdarray::Tensor;
    use num_complex::Complex;

    #[test]
    fn test_mapreducedim_sum_rows() {
        // Sum along rows: [3, 4] -> [1, 4]
        let src_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let src = Tensor::from_fn([3, 4], |idx| src_data[idx[0] * 4 + idx[1]]);

        let mut dest = Tensor::from_fn([1, 4], |_| 0.0);

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None).unwrap();

        // Column sums: [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
        assert_eq!(dest[[0, 0]], 15.0);
        assert_eq!(dest[[0, 1]], 18.0);
        assert_eq!(dest[[0, 2]], 21.0);
        assert_eq!(dest[[0, 3]], 24.0);
    }

    #[test]
    fn test_mapreducedim_sum_cols() {
        // Sum along columns: [3, 4] -> [3, 1]
        let src_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let src = Tensor::from_fn([3, 4], |idx| src_data[idx[0] * 4 + idx[1]]);

        let mut dest = Tensor::from_fn([3, 1], |_| 0.0);

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None).unwrap();

        // Row sums: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
        assert_eq!(dest[[0, 0]], 10.0);
        assert_eq!(dest[[1, 0]], 26.0);
        assert_eq!(dest[[2, 0]], 42.0);
    }

    #[test]
    fn test_mapreducedim_capture1_reduce_rows() {
        let a_data: Vec<f64> = (0..12)
            .map(|i| {
                let row = i / 4;
                let col = i % 4;
                (row * 10 + col + 1) as f64
            })
            .collect();
        let mut out_data = vec![0.0f64; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [3, 4], [4, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut out_data, [1, 4], [4, 1], 0).unwrap();

        let capture = CaptureArgs::new(|x: f64| x, (Arg,));
        mapreducedim_capture_views_into(&mut dest, &[&a], &capture, |a, b| a + b, None).unwrap();

        // Column-wise sums of:
        // [ 1  2  3  4
        //  11 12 13 14
        //  21 22 23 24 ]
        assert_eq!(dest.get([0, 0]), 33.0);
        assert_eq!(dest.get([0, 1]), 36.0);
        assert_eq!(dest.get([0, 2]), 39.0);
        assert_eq!(dest.get([0, 3]), 42.0);
    }

    #[test]
    fn test_mapreducedim_capture2_broadcast_rhs() {
        // Reduce rows: [3, 4] -> [1, 4]
        // b is [1,4] and broadcasts to [3,4]
        let a_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let b_data: Vec<f64> = (0..4).map(|i| (i as f64 + 1.0) * 10.0).collect(); // [10,20,30,40]
        let mut out_data = vec![0.0f64; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [3, 4], [4, 1], 0).unwrap();
        let b_row: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [1, 4], [4, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut out_data, [1, 4], [4, 1], 0).unwrap();

        let capture = CaptureArgs::new(|x: f64, y: f64| x + y, (Arg, Arg));
        mapreducedim_capture_views_into(&mut dest, &[&a, &b_row], &capture, |a, b| a + b, None)
            .unwrap();

        // sum(a) cols = [15,18,21,24]; b contributes 3*[10,20,30,40]
        assert_eq!(dest.get([0, 0]), 45.0);
        assert_eq!(dest.get([0, 1]), 78.0);
        assert_eq!(dest.get([0, 2]), 111.0);
        assert_eq!(dest.get([0, 3]), 144.0);
    }

    #[test]
    fn test_mapreducedim_capture_views_into_reduce_rows() {
        // Equivalent to test_mapreducedim_capture2_reduce_rows but via StridedArrayView.
        let a_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let b_data = vec![1.0f64; 12];
        let mut out_data = vec![0.0f64; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [3, 4], [4, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [3, 4], [4, 1], 0).unwrap();
        let mut out: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut out_data, [1, 4], [4, 1], 0).unwrap();

        let capture = CaptureArgs::new(|x: f64, y: f64| x + y, (Arg, Arg));
        mapreducedim_capture_views_into(&mut out, &[&a, &b], &capture, |a, b| a + b, None)
            .unwrap();

        assert_eq!(out.get([0, 0]), 18.0);
        assert_eq!(out.get([0, 1]), 21.0);
        assert_eq!(out.get([0, 2]), 24.0);
        assert_eq!(out.get([0, 3]), 27.0);
    }

    #[test]
    fn test_mapreducedim_capture_views_into_with_conj() {
        let a_data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(5.0, 4.0),
            Complex::new(7.0, -2.0),
        ];
        let mut out_data = vec![Complex::new(0.0, 0.0); 1];

        let a: StridedArrayView<'_, Complex<f64>, 1, Conj> =
            StridedArrayView::<Complex<f64>, 1, Identity>::new(&a_data, [4], [1], 0)
                .unwrap()
                .conj();
        let mut out: StridedArrayViewMut<'_, Complex<f64>, 1, Identity> =
            StridedArrayViewMut::new(&mut out_data, [1], [1], 0).unwrap();

        let capture = CaptureArgs::new(|x: Complex<f64>| x, (Arg,));
        mapreducedim_capture_views_into(&mut out, &[&a], &capture, |a, b| a + b, None).unwrap();

        // Sum of conjugated values:
        // conj(1+2i) + conj(3-1i) + conj(5+4i) + conj(7-2i)
        // = (1-2i) + (3+1i) + (5-4i) + (7+2i) = 16 - 3i
        assert_eq!(out.get([0]), Complex::new(16.0, -3.0));
    }

    #[test]
    fn test_mapreducedim_with_map_fn() {
        // Sum of squares along rows
        let src = Tensor::from_fn([2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let mut dest = Tensor::from_fn([1, 3], |_| 0.0);

        mapreducedim_into(&mut dest, &src, |&x| x * x, |a, b| a + b, None).unwrap();

        // Column sums of squares: [1²+4², 2²+5², 3²+6²] = [17, 29, 45]
        assert_eq!(dest[[0, 0]], 17.0);
        assert_eq!(dest[[0, 1]], 29.0);
        assert_eq!(dest[[0, 2]], 45.0);
    }

    #[test]
    fn test_mapreducedim_with_init_op() {
        // Test init_op that zeros the destination first
        let src = Tensor::from_fn([2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let mut dest = Tensor::from_fn([1, 3], |_| 100.0); // Start with non-zero

        // Init with zero function
        fn zero_init(_: &f64) -> f64 {
            0.0
        }

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, Some(zero_init)).unwrap();

        // Should be same as starting from zero
        assert_eq!(dest[[0, 0]], 5.0); // 1 + 4
        assert_eq!(dest[[0, 1]], 7.0); // 2 + 5
        assert_eq!(dest[[0, 2]], 9.0); // 3 + 6
    }

    #[test]
    fn test_mapreducedim_same_shape() {
        // No reduction, just map-combine
        let src = Tensor::from_fn([2, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
        let mut dest = Tensor::from_fn([2, 2], |_| 10.0);

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None).unwrap();

        // dest = dest + src: [11, 12, 13, 14]
        assert_eq!(dest[[0, 0]], 11.0);
        assert_eq!(dest[[0, 1]], 12.0);
        assert_eq!(dest[[1, 0]], 13.0);
        assert_eq!(dest[[1, 1]], 14.0);
    }

    #[test]
    fn test_mapreducedim_product() {
        // Product along columns
        let src = Tensor::from_fn([2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let mut dest = Tensor::from_fn([2, 1], |_| 1.0); // Start with 1 for product

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a * b, None).unwrap();

        // Row products: [1*2*3, 4*5*6] = [6, 120]
        assert_eq!(dest[[0, 0]], 6.0);
        assert_eq!(dest[[1, 0]], 120.0);
    }

    #[test]
    fn test_mapreducedim_1d() {
        // Full reduction: [4] -> [1]
        let src = Tensor::from_fn([4], |idx| (idx[0] + 1) as f64);
        let mut dest = Tensor::from_fn([1], |_| 0.0);

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None).unwrap();

        assert_eq!(dest[[0]], 10.0); // 1+2+3+4
    }

    #[test]
    fn test_mapreducedim_3d() {
        // 3D reduction: [2, 3, 4] -> [2, 1, 4] (reduce middle dim)
        let src = Tensor::from_fn([2, 3, 4], |idx| {
            (idx[0] * 12 + idx[1] * 4 + idx[2] + 1) as f64
        });
        let mut dest = Tensor::from_fn([2, 1, 4], |_| 0.0);

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None).unwrap();

        // Sum along dim 1 for each (i, k) position
        // dest[0, 0, 0] = src[0,0,0] + src[0,1,0] + src[0,2,0] = 1 + 5 + 9 = 15
        assert_eq!(dest[[0, 0, 0]], 15.0);
        // dest[1, 0, 3] = src[1,0,3] + src[1,1,3] + src[1,2,3] = 16 + 20 + 24 = 60
        assert_eq!(dest[[1, 0, 3]], 60.0);
    }

    #[test]
    fn test_mapreducedim_shape_mismatch() {
        let src = Tensor::from_fn([2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
        let mut dest = Tensor::from_fn([2, 2], |_| 0.0); // Incompatible shape

        let result = mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mapreducedim_rank_mismatch() {
        let src = Tensor::from_fn([2, 3, 4], |_| 1.0);
        let mut dest = Tensor::from_fn([2, 3], |_| 0.0);

        let result = mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None);
        assert!(result.is_err());
    }
}
