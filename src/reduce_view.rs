//! Reduce operations on dynamic-rank strided views.

use crate::element_op::{ElementOp, ElementOpApply};
use crate::kernel::{
    build_plan_fused, for_each_inner_block_preordered, is_contiguous, total_len,
    use_sequential_fast_path,
};
use crate::strided_view::{col_major_strides, StridedArray, StridedView};
use crate::{Result, StridedError};

#[cfg(feature = "parallel")]
use crate::threading::{
    compute_costs, for_each_inner_block_with_offsets, mapreduce_threaded, SendPtr, MINTHREADLENGTH,
};

/// Full reduction with map function: `reduce(init, op, map.(src))`.
pub fn reduce<T: Copy + ElementOpApply + Send + Sync, Op: ElementOp, M, R, U>(
    src: &StridedView<T, Op>,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<U>
where
    M: Fn(T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
{
    let src_ptr = src.ptr();
    let src_dims = src.dims();
    let src_strides = src.strides();

    if use_sequential_fast_path(total_len(src_dims)) && is_contiguous(src_dims, src_strides) {
        let len = total_len(src_dims);
        let src = unsafe { std::slice::from_raw_parts(src_ptr, len) };
        let mut acc = init;
        for &val in src.iter() {
            acc = reduce_fn(acc, map_fn(Op::apply(val)));
        }
        return Ok(acc);
    }

    let src_strides_v = src_strides.to_vec();
    let src_dims_v = src_dims.to_vec();
    let strides_list: [&[isize]; 1] = [&src_strides_v];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(&src_dims_v, &strides_list, None, std::mem::size_of::<T>());
    #[cfg(feature = "parallel")]
    let ordered_strides_refs: Vec<&[isize]> =
        ordered_strides.iter().map(|s| s.as_slice()).collect();

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            let nthreads = rayon::current_num_threads();
            // False sharing avoidance: space output slots by cache line size
            let spacing = (64 / std::mem::size_of::<U>()).max(1);
            let mut threadedout = vec![init.clone(); spacing * nthreads];
            let threadedout_ptr = SendPtr(threadedout.as_mut_ptr());
            let src_send = SendPtr(src_ptr as *mut T);

            let costs = compute_costs(&ordered_strides_refs, fused_dims.len());

            // For complete reduction, strides_list has 2 entries:
            // [0] = threadedout (stride 0 everywhere — broadcasting), [1] = src
            // The spacing/taskindex mechanism addresses output slots.
            let ndim = fused_dims.len();
            let mut threaded_strides = Vec::with_capacity(ordered_strides.len() + 1);
            threaded_strides.push(vec![0isize; ndim]); // threadedout: stride 0 (broadcast)
            for s in &ordered_strides {
                threaded_strides.push(s.clone());
            }
            let initial_offsets = vec![0isize; threaded_strides.len()];

            // Mask costs for threadedout stride=0 dims (all dims, since it's fully broadcast)
            // This means: do NOT split on dims where output stride is 0 — but for complete
            // reduction, ALL output strides are 0, so costs would all be masked to 0.
            // Julia handles this with the spacing mechanism: each task writes to its own slot.
            // We keep costs unmasked so splitting still works.

            mapreduce_threaded(
                &fused_dims,
                &plan.block,
                &threaded_strides,
                &initial_offsets,
                &costs,
                nthreads,
                spacing as isize,
                1,
                &|dims, blocks, strides_list, offsets| {
                    // offsets[0] = spacing * (taskindex - 1) for threadedout
                    // offsets[1] = offset into src
                    let out_offset = offsets[0] as usize;
                    let src_offsets = &offsets[1..];

                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        &strides_list[1..],
                        src_offsets,
                        |offsets, len, strides| {
                            let mut ptr = unsafe { src_send.as_const().offset(offsets[0]) };
                            let stride = strides[0];
                            let slot = unsafe { &mut *threadedout_ptr.as_ptr().add(out_offset) };
                            for _ in 0..len {
                                let val = Op::apply(unsafe { *ptr });
                                let mapped = map_fn(val);
                                *slot = reduce_fn(slot.clone(), mapped);
                                unsafe {
                                    ptr = ptr.offset(stride);
                                }
                            }
                            Ok(())
                        },
                    )
                },
            )?;

            // Merge thread-local results
            let mut result = init;
            for i in 0..nthreads {
                result = reduce_fn(result, threadedout[i * spacing].clone());
            }
            return Ok(result);
        }
    }

    let mut acc = Some(init);
    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
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
pub fn reduce_axis<T: Copy + ElementOpApply + Send + Sync, Op: ElementOp, M, R, U>(
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<StridedArray<U>>
where
    M: Fn(T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
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
