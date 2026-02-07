use strided_kernel::{copy_into, reduce_axis};
use strided_view::{ElementOpApply, StridedArray, StridedView};

/// Execute a single-tensor einsum operation.
///
/// Given input with axis IDs `input_ids` and desired `output_ids`:
/// 1. Identify repeated indices -> diagonal_view (stride trick, zero-copy)
/// 2. Identify indices to sum out -> reduce_axis
/// 3. Permute to output order -> copy_into
pub fn single_tensor_einsum<T>(
    src: &StridedView<T>,
    input_ids: &[char],
    output_ids: &[char],
) -> crate::Result<StridedArray<T>>
where
    T: Copy + ElementOpApply + Send + Sync + std::ops::Add<Output = T> + num_traits::Zero + Default,
{
    // Step 1: Find repeated index pairs for diagonal_view.
    // Scan input_ids left-to-right. If a char appears twice, record (first_pos, second_pos).
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut seen: Vec<(char, usize)> = Vec::new();
    for (i, &ch) in input_ids.iter().enumerate() {
        if let Some(&(_, first)) = seen.iter().find(|(c, _)| *c == ch) {
            pairs.push((first, i));
        } else {
            seen.push((ch, i));
        }
    }

    // Step 2: Apply diagonal_view if any pairs exist, and compute unique_ids.
    // If diagonal is needed, materialize into an owned array.
    let (diag_arr, unique_ids): (Option<StridedArray<T>>, Vec<char>);
    if pairs.is_empty() {
        diag_arr = None;
        unique_ids = input_ids.to_vec();
    } else {
        let dv = src.diagonal_view(&pairs)?;
        // Compute unique_ids: remove the higher-indexed axis of each pair from input_ids.
        let axes_to_remove: Vec<usize> = pairs.iter().map(|&(_, b)| b).collect();
        unique_ids = input_ids
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes_to_remove.contains(i))
            .map(|(_, &ch)| ch)
            .collect();
        let dims = dv.dims().to_vec();
        if dims.iter().product::<usize>() == 0 {
            // Empty tensor: return immediately
            let out_dims: Vec<usize> = output_ids
                .iter()
                .map(|oc| {
                    let pos = unique_ids.iter().position(|c| c == oc).unwrap();
                    dims[pos]
                })
                .collect();
            return Ok(StridedArray::<T>::col_major(&out_dims));
        }
        // Materialize the diagonal view into an owned array
        let mut owned = StridedArray::<T>::col_major(&dims);
        copy_into(&mut owned.view_mut(), &dv)?;
        diag_arr = Some(owned);
    }

    // Step 3: Find axes to sum out -- indices in unique_ids that are NOT in output_ids.
    let mut axes_to_reduce: Vec<usize> = Vec::new();
    for (i, ch) in unique_ids.iter().enumerate() {
        if !output_ids.contains(ch) {
            axes_to_reduce.push(i);
        }
    }

    // Step 4: Reduce axes from back to front (to preserve axis indices).
    // Sort axes in descending order so removing higher axes first doesn't shift lower indices.
    axes_to_reduce.sort_unstable();
    axes_to_reduce.reverse();

    let mut current_arr: Option<StridedArray<T>> = None;

    for &ax in axes_to_reduce.iter() {
        let reduced = if let Some(ref arr) = current_arr {
            reduce_axis(&arr.view(), ax, |x| x, |a, b| a + b, T::zero())?
        } else if let Some(ref arr) = diag_arr {
            reduce_axis(&arr.view(), ax, |x| x, |a, b| a + b, T::zero())?
        } else {
            reduce_axis(src, ax, |x| x, |a, b| a + b, T::zero())?
        };
        current_arr = Some(reduced);
    }

    // Compute current_ids after reductions.
    let mut current_ids = unique_ids.clone();
    // Remove reduced axes (already sorted descending, so indices stay valid).
    for &ax in axes_to_reduce.iter() {
        current_ids.remove(ax);
    }

    // Step 5: Get the current result array.
    let result_arr = if let Some(arr) = current_arr {
        arr
    } else if let Some(arr) = diag_arr {
        arr
    } else {
        // No reduction, no diagonal -- just permutation or identity.
        // We need to materialize from the source.
        let dims = src.dims().to_vec();
        let mut owned = StridedArray::<T>::col_major(&dims);
        copy_into(&mut owned.view_mut(), src)?;
        owned
    };

    // Step 6: Handle scalar output (output_ids is empty).
    if output_ids.is_empty() {
        return Ok(result_arr);
    }

    // Step 7: Permute to output order if needed.
    if current_ids == output_ids {
        // Already in the right order.
        return Ok(result_arr);
    }

    // Compute permutation: for each output axis, find its position in current_ids.
    let mut perm: Vec<usize> = Vec::with_capacity(output_ids.len());
    for oc in output_ids {
        match current_ids.iter().position(|c| c == oc) {
            Some(pos) => perm.push(pos),
            None => return Err(crate::EinsumError::OrphanOutputAxis(oc.to_string())),
        }
    }

    let permuted_view = result_arr.view().permute(&perm)?;
    let out_dims = permuted_view.dims().to_vec();
    let mut out = StridedArray::<T>::col_major(&out_dims);
    copy_into(&mut out.view_mut(), &permuted_view)?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use strided_view::StridedArray;

    #[test]
    fn test_permutation_only() {
        // ijk -> kji
        let arr = StridedArray::<f64>::from_fn_row_major(&[2, 3, 4], |idx| {
            (idx[0] * 12 + idx[1] * 4 + idx[2]) as f64
        });
        let result = single_tensor_einsum(&arr.view(), &['i', 'j', 'k'], &['k', 'j', 'i']).unwrap();
        assert_eq!(result.dims(), &[4, 3, 2]);
        assert_abs_diff_eq!(result.get(&[0, 0, 0]), 0.0);
        assert_abs_diff_eq!(result.get(&[3, 2, 1]), 23.0);
    }

    #[test]
    fn test_full_trace() {
        // ii -> (scalar)
        let mut arr = StridedArray::<f64>::col_major(&[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                arr.set(&[i, j], (i * 10 + j) as f64);
            }
        }
        // trace = A[0,0] + A[1,1] + A[2,2] = 0 + 11 + 22 = 33
        let result = single_tensor_einsum(&arr.view(), &['i', 'i'], &[]).unwrap();
        assert_abs_diff_eq!(result.data()[0], 33.0);
    }

    #[test]
    fn test_partial_trace() {
        // iij -> j  (sum over diagonal i)
        let arr = StridedArray::<f64>::from_fn_row_major(&[2, 2, 3], |idx| {
            (idx[0] * 6 + idx[1] * 3 + idx[2]) as f64
        });
        // A[0,0,:] = [0,1,2], A[1,1,:] = [9,10,11]
        // result[j] = A[0,0,j] + A[1,1,j] = [9, 11, 13]
        let result = single_tensor_einsum(&arr.view(), &['i', 'i', 'j'], &['j']).unwrap();
        assert_eq!(result.len(), 3);
        let values: Vec<f64> = (0..3).map(|j| result.data()[j]).collect();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(sorted[0], 9.0);
        assert_abs_diff_eq!(sorted[1], 11.0);
        assert_abs_diff_eq!(sorted[2], 13.0);
    }

    #[test]
    fn test_diagonal_extraction() {
        // ijj -> ij
        let arr = StridedArray::<f64>::from_fn_row_major(&[2, 3, 3], |idx| {
            (idx[0] * 9 + idx[1] * 3 + idx[2]) as f64
        });
        // result[i,j] = A[i,j,j]
        let result = single_tensor_einsum(&arr.view(), &['i', 'j', 'j'], &['i', 'j']).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
        assert_abs_diff_eq!(result.get(&[0, 0]), 0.0); // A[0,0,0]
        assert_abs_diff_eq!(result.get(&[0, 1]), 4.0); // A[0,1,1]
        assert_abs_diff_eq!(result.get(&[0, 2]), 8.0); // A[0,2,2]
        assert_abs_diff_eq!(result.get(&[1, 0]), 9.0); // A[1,0,0]
        assert_abs_diff_eq!(result.get(&[1, 1]), 13.0); // A[1,1,1]
        assert_abs_diff_eq!(result.get(&[1, 2]), 17.0); // A[1,2,2]
    }

    #[test]
    fn test_sum_axis() {
        // ij -> i (sum over j)
        let arr =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
        // result[0] = 0+1+2 = 3, result[1] = 3+4+5 = 12
        let result = single_tensor_einsum(&arr.view(), &['i', 'j'], &['i']).unwrap();
        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result.data()[0], 3.0);
        assert_abs_diff_eq!(result.data()[1], 12.0);
    }
}
