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
///              - `None`: accumulate into existing values (dest = dest + mapped)
///              - `Some(f)`: apply f to dest first (dest = f(dest)), then accumulate
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

    // Validate shapes: they must match except where dest has size 1
    if dst_view.dims.len() != src_view.dims.len() {
        return Err(StridedError::RankMismatch(
            dst_view.dims.len(),
            src_view.dims.len(),
        ));
    }

    // Compute broadcast-compatible dimensions
    let mut dims = Vec::with_capacity(src_view.dims.len());
    for i in 0..src_view.dims.len() {
        let d_dim = dst_view.dims[i];
        let s_dim = src_view.dims[i];
        if d_dim == s_dim {
            dims.push(s_dim);
        } else if d_dim == 1 {
            // Reduction dimension
            dims.push(s_dim);
        } else if s_dim == 1 {
            // Source broadcasts to dest
            dims.push(d_dim);
        } else {
            return Err(StridedError::ShapeMismatch(
                dst_view.dims.clone(),
                src_view.dims.clone(),
            ));
        }
    }

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

    // Build promoted strides for broadcast iteration
    let mut dst_promoted_strides = Vec::with_capacity(dims.len());
    let mut src_promoted_strides = Vec::with_capacity(dims.len());
    for i in 0..dims.len() {
        if dst_view.dims[i] == 1 && dims[i] > 1 {
            // Reduction dimension: stride 0 for dest
            dst_promoted_strides.push(0isize);
        } else {
            dst_promoted_strides.push(dst_view.strides[i]);
        }

        if src_view.dims[i] == 1 && dims[i] > 1 {
            // Broadcast dimension: stride 0 for src
            src_promoted_strides.push(0isize);
        } else {
            src_promoted_strides.push(src_view.strides[i]);
        }
    }

    // If init_op is provided, apply it first to all dest elements
    if let Some(init) = init_op {
        let dst_total = dst_view.dims.iter().product::<usize>();
        if dst_total > 0 {
            apply_init_op(&dst_view, init)?;
        }
    }

    // Now iterate and perform the map-reduce
    let strides_list = [&dst_promoted_strides[..], &src_promoted_strides[..]];
    let plan = build_plan(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    for_each_inner_block(&dims, &plan, &strides_list, |offsets, len, strides| {
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
    let plan = build_plan(&view.dims, &strides_list, Some(0), std::mem::size_of::<T>());

    for_each_inner_block(&view.dims, &plan, &strides_list, |offsets, len, strides| {
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

// Intentionally omitted: dense strides helper not needed in v0.

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::Tensor;

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
        fn zero_init(_: &f64) -> f64 { 0.0 }

        mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, Some(zero_init)).unwrap();

        // Should be same as starting from zero
        assert_eq!(dest[[0, 0]], 5.0);  // 1 + 4
        assert_eq!(dest[[0, 1]], 7.0);  // 2 + 5
        assert_eq!(dest[[0, 2]], 9.0);  // 3 + 6
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
        let src = Tensor::from_fn([2, 3, 4], |idx| (idx[0] * 12 + idx[1] * 4 + idx[2] + 1) as f64);
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
