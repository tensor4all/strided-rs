//! Reduce operations on dynamic-rank strided views.

#[cfg(feature = "parallel")]
use crate::kernel::same_contiguous_layout;
use crate::kernel::{
    build_plan_fused, for_each_inner_block_preordered, sequential_contiguous_layout, total_len,
};
use crate::maybe_sync::{MaybeSendSync, MaybeSync};
use crate::simd;
use crate::view::{col_major_strides, StridedArray, StridedView};
use crate::{Result, StridedError};
use strided_view::{ElementOp, ElementOpApply};

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{
    for_each_inner_block_with_offsets, mapreduce_threaded, SendPtr, MINTHREADLENGTH,
};

/// Full reduction with map function: `reduce(init, op, map.(src))`.
pub fn reduce<T: Copy + ElementOpApply + MaybeSendSync, Op: ElementOp, M, R, U>(
    src: &StridedView<T, Op>,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<U>
where
    M: Fn(T) -> U + MaybeSync,
    R: Fn(U, U) -> U + MaybeSync,
    U: Clone + MaybeSendSync,
{
    let src_ptr = src.ptr();
    let src_dims = src.dims();
    let src_strides = src.strides();

    if sequential_contiguous_layout(src_dims, &[src_strides]).is_some() {
        let len = total_len(src_dims);
        let src = unsafe { std::slice::from_raw_parts(src_ptr, len) };
        return Ok(simd::dispatch_if_large(len, || {
            let mut acc = init;
            for &val in src.iter() {
                acc = reduce_fn(acc, map_fn(Op::apply(val)));
            }
            acc
        }));
    }

    // Parallel contiguous fast path: split into rayon chunks with slice-based iteration.
    // This enables LLVM auto-vectorization on each chunk, unlike the general threaded path
    // which uses scalar pointer-offset loops.
    #[cfg(feature = "parallel")]
    {
        let total = total_len(src_dims);
        if total > MINTHREADLENGTH && same_contiguous_layout(src_dims, &[src_strides]).is_some() {
            let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, total) };
            use rayon::prelude::*;
            let nthreads = rayon::current_num_threads();
            let chunk_size = (total + nthreads - 1) / nthreads;
            let result = src_slice
                .par_chunks(chunk_size)
                .map(|chunk| {
                    simd::dispatch_if_large(chunk.len(), || {
                        let mut acc = init.clone();
                        for &val in chunk.iter() {
                            acc = reduce_fn(acc, map_fn(Op::apply(val)));
                        }
                        acc
                    })
                })
                .reduce(|| init.clone(), |a, b| reduce_fn(a, b));
            return Ok(result);
        }
    }

    let strides_list: [&[isize]; 1] = [src_strides];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(src_dims, &strides_list, None, std::mem::size_of::<T>());

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

            let costs = compute_costs(&ordered_strides);

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

    let mut acc = init;
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
                acc = reduce_fn(acc.clone(), mapped);
                unsafe {
                    ptr = ptr.offset(stride);
                }
            }
            Ok(())
        },
    )?;

    Ok(acc)
}

/// Reduce along a single axis, returning a new StridedArray.
pub fn reduce_axis<T: Copy + ElementOpApply + MaybeSendSync, Op: ElementOp, M, R, U>(
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<StridedArray<U>>
where
    M: Fn(T) -> U + MaybeSync,
    R: Fn(U, U) -> U + MaybeSync,
    U: Clone + MaybeSendSync,
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
    let mut out =
        StridedArray::from_parts(vec![init.clone(); total_out], &out_dims, &out_strides, 0)?;

    // Build source strides for iteration over non-axis dimensions (same rank as out_dims)
    let src_kept_strides: Vec<isize> = src_strides
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &s)| s)
        .collect();

    let elem_size = std::mem::size_of::<T>().max(std::mem::size_of::<U>());
    let strides_list: [&[isize]; 2] = [&out_strides, &src_kept_strides];
    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(&out_dims, &strides_list, Some(0), elem_size);

    let out_ptr = out.view_mut().as_mut_ptr();

    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            let out_step = strides[0];
            let src_step = strides[1];

            // Fast path: when both output and source have stride 1, swap to
            // reduction-outer / output-inner with slices so LLVM can
            // auto-vectorize the contiguous inner loop.
            if out_step == 1 && src_step == 1 && axis_len > 1 {
                let n = len as usize;
                let out_slice =
                    unsafe { std::slice::from_raw_parts_mut(out_ptr.offset(offsets[0]), n) };
                // First reduction element → initialize output
                let src0 = unsafe { std::slice::from_raw_parts(src_ptr.offset(offsets[1]), n) };
                for i in 0..n {
                    out_slice[i] = map_fn(Op::apply(src0[i]));
                }
                // Remaining reduction elements → accumulate
                for k in 1..axis_len {
                    let src_k = unsafe {
                        std::slice::from_raw_parts(
                            src_ptr.offset(offsets[1] + k as isize * axis_stride),
                            n,
                        )
                    };
                    for i in 0..n {
                        out_slice[i] = reduce_fn(out_slice[i].clone(), map_fn(Op::apply(src_k[i])));
                    }
                }
                return Ok(());
            }

            // General path: output-outer, reduction-inner
            let mut out_off = offsets[0];
            let mut src_off = offsets[1];
            for _ in 0..len {
                let mut acc = init.clone();
                let mut ptr = unsafe { src_ptr.offset(src_off) };
                for _ in 0..axis_len {
                    let val = Op::apply(unsafe { *ptr });
                    let mapped = map_fn(val);
                    acc = reduce_fn(acc, mapped);
                    unsafe {
                        ptr = ptr.offset(axis_stride);
                    }
                }
                unsafe {
                    *out_ptr.offset(out_off) = acc;
                }
                out_off += out_step;
                src_off += src_step;
            }
            Ok(())
        },
    )?;

    Ok(out)
}
