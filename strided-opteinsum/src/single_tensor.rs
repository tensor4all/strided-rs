use std::collections::HashMap;

use strided_kernel::{copy_into, reduce_axis};
use strided_view::{ElementOpApply, Identity, StridedArray, StridedView};

/// Execute a single-tensor einsum operation (5-step pipeline).
///
/// Given input with axis IDs `input_ids` and desired `output_ids`:
/// 1. Identify repeated input indices -> diagonal_view (stride trick, zero-copy)
/// 2. Identify indices to sum out -> reduce_axis
/// 3. Permute to output order -> copy_into
/// 4. Repeat: broadcast to NEW dimensions (output labels not in input, via size_dict)
/// 5. Duplicate: repeated output indices (e.g. "i->ii", diagonal write)
///
/// Pass `size_dict` to specify sizes for output indices not present in the input.
pub fn single_tensor_einsum<T>(
    src: &StridedView<T>,
    input_ids: &[char],
    output_ids: &[char],
    size_dict: Option<&HashMap<char, usize>>,
) -> crate::Result<StridedArray<T>>
where
    T: Copy + ElementOpApply + Send + Sync + std::ops::Add<Output = T> + num_traits::Zero + Default,
{
    // Decompose output_ids into unique labels and detect repeated output labels (Duplicate).
    // Also identify labels not present in input (Repeat/generative).
    let mut unique_output_ids: Vec<char> = Vec::new();
    let mut duplicate_map: Vec<Vec<usize>> = Vec::new(); // for each unique output label, positions in output_ids
    for (pos, &ch) in output_ids.iter().enumerate() {
        if let Some(idx) = unique_output_ids.iter().position(|&c| c == ch) {
            duplicate_map[idx].push(pos);
        } else {
            unique_output_ids.push(ch);
            duplicate_map.push(vec![pos]);
        }
    }
    let has_duplicate = duplicate_map.iter().any(|positions| positions.len() > 1);

    // Identify generative (Repeat) labels: in unique_output_ids but not in input_ids
    let generative_labels: Vec<char> = unique_output_ids
        .iter()
        .filter(|ch| !input_ids.contains(ch))
        .copied()
        .collect();
    let has_repeat = !generative_labels.is_empty();

    // If we have generative or duplicate labels, we need a different pipeline.
    // Otherwise, fall through to the existing optimized pipeline.
    if !has_repeat && !has_duplicate {
        return single_tensor_einsum_classic(src, input_ids, output_ids);
    }

    // For generative labels, look up sizes from size_dict
    let empty_dict = HashMap::new();
    let sd = size_dict.unwrap_or(&empty_dict);
    let mut gen_sizes: HashMap<char, usize> = HashMap::new();
    for &ch in &generative_labels {
        match sd.get(&ch) {
            Some(&sz) => {
                gen_sizes.insert(ch, sz);
            }
            None => return Err(crate::EinsumError::OrphanOutputAxis(ch.to_string())),
        }
    }

    // Step 1-3: Run classic pipeline on unique_output_ids minus generative labels.
    // This gives us the "core" result before Repeat and Duplicate.
    let core_output_ids: Vec<char> = unique_output_ids
        .iter()
        .filter(|ch| !generative_labels.contains(ch))
        .copied()
        .collect();

    let core_result = if core_output_ids.is_empty() && input_ids.is_empty() {
        // Scalar input, scalar core output — just copy the scalar
        let mut out = StridedArray::<T>::col_major(&[]);
        out.data_mut()[0] = unsafe { *src.data().as_ptr().offset(src.offset() as isize) };
        out
    } else if core_output_ids.is_empty() && !input_ids.is_empty() {
        // Need to reduce everything (full trace/sum)
        single_tensor_einsum_classic(src, input_ids, &[])?
    } else {
        single_tensor_einsum_classic(src, input_ids, &core_output_ids)?
    };

    // Step 4 (Repeat): broadcast core_result to include generative dimensions.
    // Build the intermediate shape: unique_output_ids ordering, with generative dims added.
    let intermediate = if has_repeat {
        let mut inter_dims: Vec<usize> = Vec::new();
        let mut inter_strides: Vec<isize> = Vec::new();
        let core_view = core_result.view();
        for &ch in &unique_output_ids {
            if generative_labels.contains(&ch) {
                inter_dims.push(gen_sizes[&ch]);
                inter_strides.push(0); // stride-0 = broadcast
            } else {
                let core_pos = core_output_ids.iter().position(|&c| c == ch).unwrap();
                inter_dims.push(core_view.dims()[core_pos]);
                inter_strides.push(core_view.strides()[core_pos]);
            }
        }
        // Create a broadcasted view and materialize it
        let broadcast_view: StridedView<'_, T, Identity> = StridedView::new(
            core_view.data(),
            &inter_dims,
            &inter_strides,
            core_view.offset() as isize,
        )?;
        let mut materialized = StridedArray::<T>::col_major(&inter_dims);
        copy_into(&mut materialized.view_mut(), &broadcast_view)?;
        materialized
    } else {
        // No repeat needed; but we may need to permute core_result to unique_output_ids order
        if core_output_ids == unique_output_ids {
            core_result
        } else {
            // Permute
            let perm: Vec<usize> = unique_output_ids
                .iter()
                .map(|ch| core_output_ids.iter().position(|c| c == ch).unwrap())
                .collect();
            let permuted_view = core_result.view().permute(&perm)?;
            let mut out = StridedArray::<T>::col_major(permuted_view.dims());
            copy_into(&mut out.view_mut(), &permuted_view)?;
            out
        }
    };

    // Step 5 (Duplicate): write to diagonal positions for repeated output labels.
    if !has_duplicate {
        return Ok(intermediate);
    }

    // Build the full output shape
    let out_dims: Vec<usize> = output_ids
        .iter()
        .map(|ch| {
            let idx = unique_output_ids.iter().position(|c| c == ch).unwrap();
            intermediate.dims()[idx]
        })
        .collect();
    let total: usize = out_dims.iter().product::<usize>().max(1);
    let mut out_data = vec![T::zero(); total];
    let out_strides = strided_view::col_major_strides(&out_dims);

    // Iterate over the intermediate (unique_output_ids shape) and write to
    // all matching positions in the output.
    let inter_dims = intermediate.dims().to_vec();
    let inter_total: usize = inter_dims.iter().product::<usize>().max(1);
    let inter_strides_cm = strided_view::col_major_strides(&inter_dims);

    let mut inter_idx = vec![0usize; inter_dims.len()];
    for flat in 0..inter_total {
        // Read value from intermediate
        let mut inter_flat = 0usize;
        for d in 0..inter_dims.len() {
            inter_flat += inter_idx[d] * inter_strides_cm[d] as usize;
        }
        let val = intermediate.data()[inter_flat];

        // Compute output flat index: for each output position, look up the
        // corresponding unique_output_id index from inter_idx
        let mut out_flat = 0usize;
        for (out_pos, &ch) in output_ids.iter().enumerate() {
            let inter_pos = unique_output_ids.iter().position(|&c| c == ch).unwrap();
            out_flat += inter_idx[inter_pos] * out_strides[out_pos] as usize;
        }
        out_data[out_flat] = val;

        // Increment inter_idx (col-major order)
        if flat + 1 < inter_total {
            for d in 0..inter_dims.len() {
                inter_idx[d] += 1;
                if inter_idx[d] < inter_dims[d] {
                    break;
                }
                inter_idx[d] = 0;
            }
        }
    }

    StridedArray::from_parts(out_data, &out_dims, &out_strides, 0)
        .map_err(|e| crate::EinsumError::Strided(e))
}

/// Classic single-tensor einsum (steps 1-3 only: Diag, Sum, Permute).
///
/// No generative or duplicate output labels.
fn single_tensor_einsum_classic<T>(
    src: &StridedView<T>,
    input_ids: &[char],
    output_ids: &[char],
) -> crate::Result<StridedArray<T>>
where
    T: Copy + ElementOpApply + Send + Sync + std::ops::Add<Output = T> + num_traits::Zero + Default,
{
    // Fast path: full trace "cc...c -> " (all indices identical, scalar output)
    if output_ids.is_empty()
        && !input_ids.is_empty()
        && input_ids.iter().all(|&c| c == input_ids[0])
    {
        let n = src.dims()[0];
        let diag_stride: isize = src.strides().iter().sum();
        let ptr = src.data().as_ptr();
        let mut offset = src.offset() as isize;
        let mut acc = T::zero();
        for _ in 0..n {
            acc = acc + unsafe { *ptr.offset(offset) };
            offset += diag_stride;
        }
        let mut out = StridedArray::<T>::col_major(&[]);
        out.data_mut()[0] = acc;
        return Ok(out);
    }

    // Fast path: partial trace with one repeated pair
    // e.g. "iij->j", "iji->j", "jii->j" — one pair of repeated indices, rest go to output
    {
        let mut pair: Option<(char, usize, usize)> = None;
        let mut seen_chars: Vec<(char, usize)> = Vec::new();
        let mut has_triple = false;
        for (i, &ch) in input_ids.iter().enumerate() {
            if let Some(&(_, first)) = seen_chars.iter().find(|(c, _)| *c == ch) {
                if pair.is_some() {
                    // More than one pair — not a simple partial trace
                    has_triple = true;
                    break;
                }
                pair = Some((ch, first, i));
            } else {
                seen_chars.push((ch, i));
            }
        }

        if let Some((repeated_ch, pos0, pos1)) = pair {
            if !has_triple && !output_ids.contains(&repeated_ch) {
                // All non-repeated indices must be in output_ids
                let free_ids: Vec<(char, usize)> = input_ids
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != pos0 && *i != pos1)
                    .map(|(i, &ch)| (ch, i))
                    .collect();
                let all_free_in_output = free_ids.iter().all(|(ch, _)| output_ids.contains(ch));

                if all_free_in_output && free_ids.len() == output_ids.len() {
                    let n = src.dims()[pos0]; // diagonal length
                    let diag_stride = src.strides()[pos0] + src.strides()[pos1];
                    let ptr = src.data().as_ptr();
                    let base_offset = src.offset() as isize;

                    // Compute output permutation: output_ids order vs free_ids order
                    let out_dims: Vec<usize> = output_ids
                        .iter()
                        .map(|oc| {
                            let (_, src_axis) = free_ids.iter().find(|(ch, _)| ch == oc).unwrap();
                            src.dims()[*src_axis]
                        })
                        .collect();
                    let out_strides_src: Vec<isize> = output_ids
                        .iter()
                        .map(|oc| {
                            let (_, src_axis) = free_ids.iter().find(|(ch, _)| ch == oc).unwrap();
                            src.strides()[*src_axis]
                        })
                        .collect();

                    let total_out: usize = out_dims.iter().product::<usize>().max(1);
                    let out_col_strides = strided_view::col_major_strides(&out_dims);
                    let mut out_data = vec![T::zero(); total_out];

                    // Iterate over output elements using col-major order
                    let out_rank = out_dims.len();
                    let mut idx = vec![0usize; out_rank];
                    for flat in 0..total_out {
                        // Compute source offset for this output position
                        let mut src_off = base_offset;
                        for d in 0..out_rank {
                            src_off += idx[d] as isize * out_strides_src[d];
                        }
                        // Sum along diagonal
                        let mut acc = T::zero();
                        let mut diag_off = src_off;
                        for _ in 0..n {
                            acc = acc + unsafe { *ptr.offset(diag_off) };
                            diag_off += diag_stride;
                        }
                        // Write to output using col-major flat index
                        let mut out_flat = 0usize;
                        for d in 0..out_rank {
                            out_flat += idx[d] * out_col_strides[d] as usize;
                        }
                        out_data[out_flat] = acc;

                        // Increment index (col-major order)
                        if flat + 1 < total_out {
                            for d in 0..out_rank {
                                idx[d] += 1;
                                if idx[d] < out_dims[d] {
                                    break;
                                }
                                idx[d] = 0;
                            }
                        }
                    }

                    return StridedArray::from_parts(out_data, &out_dims, &out_col_strides, 0)
                        .map_err(|e| crate::EinsumError::Strided(e));
                }
            }
        }
    }

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
    // If diagonal is needed and reduction follows, reduce directly from the diagonal
    // view (skip materialization). Otherwise materialize into an owned array.
    let (diag_arr, unique_ids): (Option<StridedArray<T>>, Vec<char>);
    let mut diag_reduce_done = false;
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

        // Check if there are axes to reduce — if so, reduce from the diagonal view
        // directly without materializing first, saving one full copy.
        let has_reduce = unique_ids.iter().any(|ch| !output_ids.contains(ch));
        if has_reduce {
            // Compute axes to reduce (within diagonal view's axes)
            let mut axes_to_reduce_diag: Vec<usize> = Vec::new();
            for (i, ch) in unique_ids.iter().enumerate() {
                if !output_ids.contains(ch) {
                    axes_to_reduce_diag.push(i);
                }
            }
            axes_to_reduce_diag.sort_unstable();
            axes_to_reduce_diag.reverse();

            // First reduction reads directly from diagonal view (no copy!)
            let mut reduced =
                reduce_axis(&dv, axes_to_reduce_diag[0], |x| x, |a, b| a + b, T::zero())?;
            // Subsequent reductions read from the owned result of the previous
            for &ax in &axes_to_reduce_diag[1..] {
                reduced = reduce_axis(&reduced.view(), ax, |x| x, |a, b| a + b, T::zero())?;
            }
            diag_arr = Some(reduced);
            diag_reduce_done = true;
        } else {
            // No reduction follows — materialize for output
            let mut owned = StridedArray::<T>::col_major(&dims);
            copy_into(&mut owned.view_mut(), &dv)?;
            diag_arr = Some(owned);
        }
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

    if !diag_reduce_done {
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
    }

    // Compute current_ids after reductions.
    let mut current_ids = unique_ids.clone();
    if diag_reduce_done {
        // Reductions already happened in the diagonal branch.
        // Remove all reduced axes from current_ids.
        let mut reduced_axes: Vec<usize> = Vec::new();
        for (i, ch) in unique_ids.iter().enumerate() {
            if !output_ids.contains(ch) {
                reduced_axes.push(i);
            }
        }
        reduced_axes.sort_unstable();
        reduced_axes.reverse();
        for ax in reduced_axes {
            current_ids.remove(ax);
        }
    } else {
        // Remove reduced axes (already sorted descending, so indices stay valid).
        for &ax in axes_to_reduce.iter() {
            current_ids.remove(ax);
        }
    }

    // Step 5: Get the current result view for permutation.
    // For permute-only (no diagonal, no reduction), use src directly to avoid double copy.

    // Step 6: Handle scalar output (output_ids is empty).
    if output_ids.is_empty() {
        let result_arr = if let Some(arr) = current_arr {
            arr
        } else if let Some(arr) = diag_arr {
            arr
        } else {
            // Scalar from source (shouldn't normally happen — scalar output implies reduction)
            let mut owned = StridedArray::<T>::col_major(&[]);
            owned.data_mut()[0] = unsafe { *src.data().as_ptr().offset(src.offset() as isize) };
            owned
        };
        return Ok(result_arr);
    }

    // Step 7: Permute to output order if needed.
    if current_ids == output_ids {
        let result_arr = if let Some(arr) = current_arr {
            arr
        } else if let Some(arr) = diag_arr {
            arr
        } else {
            // Identity: copy src to col-major owned array
            let dims = src.dims().to_vec();
            let mut owned = StridedArray::<T>::col_major(&dims);
            copy_into(&mut owned.view_mut(), src)?;
            owned
        };
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

    // Permute from the best available source (avoid intermediate copy)
    let source_view = if let Some(ref arr) = current_arr {
        arr.view()
    } else if let Some(ref arr) = diag_arr {
        arr.view()
    } else {
        // No intermediate — permute src directly (single copy instead of double)
        src.clone()
    };

    let permuted_view = source_view.permute(&perm)?;
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
        let result =
            single_tensor_einsum(&arr.view(), &['i', 'j', 'k'], &['k', 'j', 'i'], None).unwrap();
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
        let result = single_tensor_einsum(&arr.view(), &['i', 'i'], &[], None).unwrap();
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
        let result = single_tensor_einsum(&arr.view(), &['i', 'i', 'j'], &['j'], None).unwrap();
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
        let result =
            single_tensor_einsum(&arr.view(), &['i', 'j', 'j'], &['i', 'j'], None).unwrap();
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
        let result = single_tensor_einsum(&arr.view(), &['i', 'j'], &['i'], None).unwrap();
        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result.data()[0], 3.0);
        assert_abs_diff_eq!(result.data()[1], 12.0);
    }
}
