//! Reduce operations on dynamic-rank strided views.

use crate::element_op::{ElementOp, ElementOpApply};
use crate::kernel::{build_plan_fused, for_each_inner_block, is_contiguous, total_len};
use crate::strided_view::{col_major_strides, StridedArray, StridedView};
use crate::{Result, StridedError};

/// Full reduction with map function: `reduce(init, op, map.(src))`.
pub fn reduce<T: Copy + ElementOpApply, Op: ElementOp, M, R, U>(
    src: &StridedView<T, Op>,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<U>
where
    M: Fn(T) -> U,
    R: Fn(U, U) -> U,
{
    let src_ptr = src.ptr();
    let src_dims = src.dims();
    let src_strides = src.strides();

    if is_contiguous(src_dims, src_strides) {
        let len = total_len(src_dims);
        let mut ptr = src_ptr;
        let mut acc = init;
        for _ in 0..len {
            let val = Op::apply(unsafe { *ptr });
            let mapped = map_fn(val);
            acc = reduce_fn(acc, mapped);
            unsafe {
                ptr = ptr.add(1);
            }
        }
        return Ok(acc);
    }

    let src_strides_v = src_strides.to_vec();
    let src_dims_v = src_dims.to_vec();
    let strides_list: [&[isize]; 1] = [&src_strides_v];

    let (fused_dims, plan) =
        build_plan_fused(&src_dims_v, &strides_list, None, std::mem::size_of::<T>());

    let mut acc = Some(init);
    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut ptr = unsafe { src_ptr.offset(offsets[0]) };
            let stride = strides[0];
            for _ in 0..len {
                let val = Op::apply(unsafe { *ptr });
                let mapped = map_fn(val);
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

/// Reduce along a single axis, returning a new StridedArray.
pub fn reduce_axis<T: Copy + ElementOpApply, Op: ElementOp, M, R, U>(
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<StridedArray<U>>
where
    M: Fn(T) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    let rank = src.ndim();
    if axis >= rank {
        return Err(StridedError::InvalidAxis { axis, rank });
    }

    let src_dims = src.dims();
    let src_strides = src.strides();
    let src_ptr = src.ptr();

    let out_dims: Vec<usize> = src_dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();

    let axis_len = src_dims[axis];
    let axis_stride = src_strides[axis];

    if out_dims.is_empty() {
        // Reduce to scalar
        let mut acc = init;
        let mut offset = 0isize;
        for _ in 0..axis_len {
            let val = Op::apply(unsafe { *src_ptr.offset(offset) });
            let mapped = map_fn(val);
            acc = reduce_fn(acc, mapped);
            offset += axis_stride;
        }
        let strides = col_major_strides(&[1]);
        return StridedArray::from_parts(vec![acc], &[1], &strides, 0);
    }

    let total_out: usize = out_dims.iter().product();
    let out_strides = col_major_strides(&out_dims);
    let mut data = vec![init.clone(); total_out];

    // Build strides for iteration over non-axis dimensions
    let iter_strides: Vec<isize> = src_strides
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &s)| s)
        .collect();

    // Iterate over all output positions
    let out_rank = out_dims.len();
    let mut idx = vec![0usize; out_rank];
    for out_i in 0..total_out {
        // Compute source base offset for this output position
        let mut base_offset = 0isize;
        for (d, &index) in idx.iter().enumerate() {
            base_offset += index as isize * iter_strides[d];
        }

        // Reduce along the axis
        let mut acc = init.clone();
        let mut offset = base_offset;
        for _ in 0..axis_len {
            let val = Op::apply(unsafe { *src_ptr.offset(offset) });
            let mapped = map_fn(val);
            acc = reduce_fn(acc, mapped);
            offset += axis_stride;
        }
        data[out_i] = acc;

        // Increment multi-index (column-major order to match out_strides)
        for d in 0..out_rank {
            idx[d] += 1;
            if idx[d] < out_dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }

    StridedArray::from_parts(data, &out_dims, &out_strides, 0)
}
