use crate::kernel::{
    build_plan, ensure_same_shape, for_each_inner_block, for_each_offset, is_contiguous, total_len,
    StridedView, StridedViewMut,
};
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
    let plan = build_plan(
        &src_view.dims,
        &strides_list,
        None,
        std::mem::size_of::<T>(),
    );

    let mut acc = Some(init);
    for_each_inner_block(
        &src_view.dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
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
        },
    )?;

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
    let plan = build_plan(
        &out_view.dims,
        &strides_list,
        Some(1),
        std::mem::size_of::<U>(),
    );
    for_each_offset(&out_view.dims, &plan, &strides_list, |offsets| {
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

// Intentionally omitted: dense strides helper not needed in v0.
