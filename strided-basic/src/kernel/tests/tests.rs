use super::*;
use crate::cost::KernelClass;

#[test]
fn test_kernel_inner_block() {
    let dims = vec![2, 4];
    let strides1 = vec![4isize, 1];
    let strides2 = vec![4isize, 1];
    let strides_list: Vec<&[isize]> = vec![&strides1, &strides2];
    let plan = build_plan(&dims, &strides_list, Some(0), &[8, 8]);

    let mut total_elements = 0usize;
    for_each_inner_block(&dims, &plan, &strides_list, |_offsets, len, _strides| {
        total_elements += len;
        Ok(())
    })
    .unwrap();

    assert_eq!(total_elements, 8);
}

#[test]
fn test_contiguous_layout_row_vs_col() {
    let dims = [3usize, 4];
    let row = [4isize, 1];
    let col = [1isize, 3];
    assert_eq!(
        contiguous_layout(&dims, &row),
        Some(ContiguousLayout::RowMajor)
    );
    assert_eq!(
        contiguous_layout(&dims, &col),
        Some(ContiguousLayout::ColMajor)
    );
    assert!(contiguous_layout(&dims, &row).is_some());
    assert!(contiguous_layout(&dims, &col).is_some());
}

#[test]
fn test_contiguous_layout_ignores_dim1_axes() {
    let dims = [2usize, 1, 3];
    // Middle stride is irrelevant since that axis never varies.
    let strides = [3isize, 999, 1];
    assert_eq!(
        contiguous_layout(&dims, &strides),
        Some(ContiguousLayout::RowMajor)
    );
}

// ---- same_contiguous_layout tests ----

#[test]
fn test_same_contiguous_layout_all_row_major() {
    let dims = [3usize, 4];
    let s1 = [4isize, 1];
    let s2 = [4isize, 1];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s1, &s2]),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_same_contiguous_layout_all_col_major() {
    let dims = [3usize, 4];
    let s1 = [1isize, 3];
    let s2 = [1isize, 3];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s1, &s2]),
        Some(ContiguousLayout::ColMajor)
    );
}

#[test]
fn test_same_contiguous_layout_mixed_layouts() {
    let dims = [3usize, 4];
    let row = [4isize, 1];
    let col = [1isize, 3];
    assert_eq!(same_contiguous_layout(&dims, &[&row, &col]), None);
}

#[test]
fn test_same_contiguous_layout_one_noncontiguous() {
    let dims = [3usize, 4];
    let row = [4isize, 1];
    let bad = [8isize, 2];
    assert_eq!(same_contiguous_layout(&dims, &[&row, &bad]), None);
}

#[test]
fn test_same_contiguous_layout_empty_strides_list() {
    let dims = [3usize, 4];
    let empty: &[&[isize]] = &[];
    assert_eq!(same_contiguous_layout(&dims, empty), None);
}

#[test]
fn test_same_contiguous_layout_single_array() {
    let dims = [3usize, 4];
    let s = [4isize, 1];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s[..]]),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_same_contiguous_layout_many_arrays() {
    let dims = [2usize, 3];
    let s = [3isize, 1];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s[..], &s[..], &s[..], &s[..], &s[..]]),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_same_contiguous_layout_empty_dims() {
    let dims: [usize; 0] = [];
    let s: [isize; 0] = [];
    assert_eq!(
        same_contiguous_layout(&dims, &[&s[..], &s[..]]),
        Some(ContiguousLayout::RowMajor)
    );
}

// ---- sequential_contiguous_layout tests ----

#[test]
fn test_sequential_contiguous_layout_small_array() {
    let dims = [3usize, 4];
    let s1 = [4isize, 1];
    let s2 = [4isize, 1];
    assert_eq!(
        sequential_contiguous_layout(KernelClass::Map, &dims, &[&s1, &s2], &Exec::serial())
            .unwrap(),
        Some(ContiguousLayout::RowMajor)
    );
}

#[test]
fn test_sequential_contiguous_layout_noncontiguous() {
    let dims = [3usize, 4];
    let s1 = [4isize, 1];
    let s2 = [8isize, 2];
    assert_eq!(
        sequential_contiguous_layout(KernelClass::Map, &dims, &[&s1, &s2], &Exec::serial())
            .unwrap(),
        None
    );
}

#[test]
fn test_sequential_contiguous_layout_col_major() {
    let dims = [3usize, 4];
    let col = [1isize, 3];
    assert_eq!(
        sequential_contiguous_layout(KernelClass::Map, &dims, &[&col], &Exec::serial()).unwrap(),
        Some(ContiguousLayout::ColMajor)
    );
}

#[test]
fn test_build_plan_fused_compresses() {
    // A contiguous 2x3 column-major array fuses [2,3] -> [6,1] -> compress -> [6]
    let dims = [2usize, 3];
    let strides = [1isize, 2];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), &[8]);
    // After fusion + compression, should be 1D
    assert_eq!(fused_dims.as_slice(), vec![6]);
    assert_eq!(fused_strides.len(), 1);
    assert_eq!(fused_strides[0].as_slice(), vec![1]);
    assert_eq!(plan.block.len(), 1);
}

#[test]
fn test_build_plan_fused_keeps_broadcast_compact_inner_loop() {
    let dims = [16usize, 16, 64, 64];
    let out = [1isize, 16, 256, 16_384];
    let lhs = [1isize, 16, 0, 256];
    let rhs = [0isize, 0, 1, 64];
    let strides_list: Vec<&[isize]> = vec![&out, &lhs, &rhs];

    let (fused_dims, fused_strides, _) = build_plan_fused(&dims, &strides_list, Some(0), &[8; 3]);

    assert_eq!(fused_dims[0], 256);
    assert_eq!(fused_strides[0][0], 1);
    assert_eq!(fused_strides[1][0], 1);
    assert_eq!(fused_strides[2][0], 0);
}

#[test]
fn test_kernel_nd_iterative_total_elements_match() {
    // Use rank 9 to test kernel_nd_inner_iterative (fallback for rank >= 9)
    let dims = vec![2usize, 2, 2, 2, 2, 2, 2, 2, 3];
    let blocks = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 1];
    let mut stride_val = 1isize;
    let mut sv = Vec::new();
    for &d in &dims {
        sv.push(stride_val);
        stride_val *= d as isize;
    }
    let strides = [sv.clone(), sv];
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    let mut offsets = vec![0isize, 0isize];
    let mut total = 0usize;

    kernel_nd_inner_iterative(
        &dims,
        &blocks,
        &strides,
        &mut offsets,
        &mut |_off, len, _s| {
            total += len;
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(total, dims.iter().product::<usize>());
    assert_eq!(offsets, vec![0isize, 0isize]);
}

/// Helper: run a macro-generated kernel and verify total elements and offset reset.
fn verify_kernel_total_and_offsets(rank: usize) {
    // Build dims: dim[0] = 3 for block variety, rest = 2
    let mut dims = vec![3usize];
    dims.resize(rank, 2);
    let blocks = vec![2usize; rank];

    // Column-major strides
    let mut stride_val = 1isize;
    let mut sv = Vec::new();
    for &d in &dims {
        sv.push(stride_val);
        stride_val *= d as isize;
    }
    let strides = [sv.clone(), sv];
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    let mut offsets = vec![0isize, 0isize];
    let mut total = 0usize;
    let expected: usize = dims.iter().product();

    let result = match rank {
        1 => kernel_1d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        2 => kernel_2d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        3 => kernel_3d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        4 => kernel_4d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        5 => kernel_5d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        6 => kernel_6d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        7 => kernel_7d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        8 => kernel_8d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
            total += l;
            Ok(())
        }),
        _ => panic!("unsupported rank"),
    };
    result.unwrap();
    assert_eq!(total, expected, "rank={rank}: total mismatch");
    assert_eq!(offsets, vec![0, 0], "rank={rank}: offsets not reset");
}

#[test]
fn test_macro_kernels_total_elements_1d() {
    verify_kernel_total_and_offsets(1);
}

#[test]
fn test_macro_kernels_total_elements_2d() {
    verify_kernel_total_and_offsets(2);
}

#[test]
fn test_macro_kernels_total_elements_3d() {
    verify_kernel_total_and_offsets(3);
}

#[test]
fn test_macro_kernels_total_elements_4d() {
    verify_kernel_total_and_offsets(4);
}

#[test]
fn test_macro_kernels_total_elements_5d() {
    verify_kernel_total_and_offsets(5);
}

#[test]
fn test_macro_kernels_total_elements_6d() {
    verify_kernel_total_and_offsets(6);
}

#[test]
fn test_macro_kernels_total_elements_7d() {
    verify_kernel_total_and_offsets(7);
}

#[test]
fn test_macro_kernels_total_elements_8d() {
    verify_kernel_total_and_offsets(8);
}

/// Verify that macro-generated kernels visit every element exactly once
/// by collecting all linear offsets and checking them against the expected set.
fn verify_kernel_visits_all_elements(rank: usize) {
    assert!((2..=8).contains(&rank));
    let mut dims = vec![3usize];
    dims.resize(rank, 2);
    let blocks = vec![2usize; rank];

    // Column-major strides (single array for simplicity)
    let mut stride_val = 1isize;
    let mut sv = Vec::new();
    for &d in &dims {
        sv.push(stride_val);
        stride_val *= d as isize;
    }
    let strides = [sv];
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();

    // Collect all linear offsets visited
    let mut visited = std::collections::HashSet::new();
    let mut offsets = vec![0isize];

    let result = match rank {
        2 => kernel_2d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        3 => kernel_3d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        4 => kernel_4d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        5 => kernel_5d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        6 => kernel_6d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        7 => kernel_7d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        8 => kernel_8d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
            for i in 0..len {
                visited.insert(o[0] + (i as isize) * s[0]);
            }
            Ok(())
        }),
        _ => unreachable!(),
    };
    result.unwrap();

    // Expected: all offsets 0..total
    let total: usize = dims.iter().product();
    let expected: std::collections::HashSet<isize> = (0..total as isize).collect();
    assert_eq!(
        visited, expected,
        "rank={rank}: not all elements visited exactly once"
    );
}

#[test]
fn test_macro_kernel_5d_visits_all_elements() {
    verify_kernel_visits_all_elements(5);
}

#[test]
fn test_macro_kernel_6d_visits_all_elements() {
    verify_kernel_visits_all_elements(6);
}

#[test]
fn test_macro_kernel_7d_visits_all_elements() {
    verify_kernel_visits_all_elements(7);
}

#[test]
fn test_macro_kernel_8d_visits_all_elements() {
    verify_kernel_visits_all_elements(8);
}

// ============================================================================
// L1 set fit of the planned tiles (#287)
// ============================================================================

/// Lines a tile may keep live in one 8-way L1 set.
const LIVE_LINES: u128 = 6;

/// The per-set load of the lines a tile keeps live, computed independently of
/// the planner from the L1d geometry the tiles target: 32 KiB, 8 ways, so a
/// 4 KiB way and 64-byte lines.
///
/// For operand `k`, `c_k` is its first axis whose byte stride is below a line;
/// the lines along the axes before it are revisited on the next step of
/// `c_k`. An axis whose byte stride is a multiple of the 4 KiB way puts all
/// `b` of its lines in one set; other axes spread theirs and are not counted.
/// A tile may keep [`LIVE_LINES`] of them: the 8 ways less a quarter of
/// headroom for pseudo-LRU replacement and the stride prefetcher.
fn l1_set_load(blocks: &[usize], strides_list: &[&[isize]], elem_sizes: &[usize]) -> u128 {
    const LINE: u128 = 64;
    const WAY: u128 = 32 * 1024 / 8;
    strides_list
        .iter()
        .zip(elem_sizes)
        .filter(|&(_, &e)| e > 0)
        .filter_map(|(strides, &e)| {
            let byte = |i: usize| strides[i].unsigned_abs() as u128 * e as u128;
            // `c_k = 0`: the operand streams its lines and keeps none live.
            let c = (0..strides.len())
                .find(|&i| byte(i) < LINE)
                .filter(|&c| c > 0)?;
            Some(
                (0..c)
                    .filter(|&i| byte(i) % WAY == 0)
                    .fold(1u128, |load, i| load * blocks[i] as u128),
            )
        })
        .sum()
}

/// Column-major strides of `dims`.
fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut acc = 1isize;
    dims.iter()
        .map(|&d| {
            let s = acc;
            acc *= d as isize;
            s
        })
        .collect()
}

/// Strides of `dims` stored with axis `storage[0]` fastest, then `storage[1]`, ...
fn stored(dims: &[usize], storage: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    let mut acc = 1isize;
    for &axis in storage {
        strides[axis] = acc;
        acc *= dims[axis] as isize;
    }
    strides
}

/// `D = A op B` layouts whose transposed or permuted operands walk
/// power-of-two strides along the destination's run: the tiles of #287.
/// A named layout: dims and one stride vector per operand, destination first.
type NamedLayout = (&'static str, Vec<usize>, Vec<Vec<isize>>);

fn transposed_operand_layouts() -> Vec<NamedLayout> {
    let mut cases = Vec::new();
    for (m, n) in [(2048, 2048), (1024, 4096), (4096, 1024)] {
        let d = col_major(&[m, n]);
        let t = stored(&[m, n], &[1, 0]);
        cases.push((
            "B transposed",
            vec![m, n],
            vec![d.clone(), d.clone(), t.clone()],
        ));
        cases.push((
            "A and B transposed",
            vec![m, n],
            vec![d.clone(), t.clone(), t.clone()],
        ));
        cases.push((
            "B transposed, A row broadcast",
            vec![m, n],
            vec![d, vec![0, 1], t],
        ));
    }
    let dims = [128usize, 128, 256];
    let d = col_major(&dims);
    for storage in [[2, 0, 1], [1, 2, 0], [2, 1, 0]] {
        cases.push((
            "B permuted",
            dims.to_vec(),
            vec![d.clone(), d.clone(), stored(&dims, &storage)],
        ));
    }
    cases
}

/// Copy-shaped (one destination, one source) layouts whose source is
/// transposed or permuted along the destination's run.
fn transposed_pair_layouts() -> Vec<NamedLayout> {
    let mut cases = Vec::new();
    for (m, n) in [(2048, 2048), (1024, 4096), (4096, 1024)] {
        let d = col_major(&[m, n]);
        let t = stored(&[m, n], &[1, 0]);
        cases.push(("source transposed", vec![m, n], vec![d, t]));
    }
    let dims = [128usize, 128, 256];
    for storage in [[1, 2, 0], [2, 1, 0]] {
        cases.push((
            "source permuted",
            dims.to_vec(),
            vec![col_major(&dims), stored(&dims, &storage)],
        ));
    }
    cases
}

#[test]
fn transposed_pair_tiles_fit_the_l1_sets() {
    for elem in [8usize, 4] {
        for (name, dims, strides) in transposed_pair_layouts() {
            let refs: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
            let sizes = vec![elem; refs.len()];
            let (fused, ordered, plan) = build_plan_fused(&dims, &refs, Some(0), &sizes);
            let ordered: Vec<&[isize]> = ordered.iter().map(|s| s.as_slice()).collect();
            let load = l1_set_load(&plan.block, &ordered, &sizes);
            assert!(
                load <= LIVE_LINES,
                "{name} {dims:?} elem {elem}: tiles {:?} over {:?} keep {load} lines live in one 8-way L1 set",
                plan.block.as_slice(),
                fused.as_slice(),
            );
        }
    }
}

/// Walks with more than two operands keep the model's tiles: the #287 gate
/// measured the fit regressing three-operand `mul_into`.
#[test]
fn wider_walks_keep_the_model_tiles() {
    for elem in [8usize, 4] {
        for (name, dims, strides) in transposed_operand_layouts() {
            let refs: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
            let sizes = vec![elem; refs.len()];
            let (fused, ordered, plan) = build_plan_fused(&dims, &refs, Some(0), &sizes);
            let ordered: Vec<&[isize]> = ordered.iter().map(|s| s.as_slice()).collect();
            let model = crate::block::compute_block_sizes(&fused, &ordered, &sizes);
            assert_eq!(
                plan.block.as_slice(),
                model.as_slice(),
                "{name} {dims:?} elem {elem}"
            );
        }
    }
}

#[test]
fn transposed_copy_pair_tiles_fit_the_l1_sets() {
    use crate::raw_ops::{pair_tiles, FusedPairLayout, RAW_FUSED_RANK_LIMIT};
    for (m, n) in [(2048usize, 2048usize), (1024, 4096), (4096, 1024)] {
        let mut layout = FusedPairLayout {
            rank: 2,
            dims: [1; RAW_FUSED_RANK_LIMIT],
            dst_strides: [0; RAW_FUSED_RANK_LIMIT],
            src_strides: [0; RAW_FUSED_RANK_LIMIT],
        };
        layout.dims[..2].copy_from_slice(&[m, n]);
        layout.dst_strides[..2].copy_from_slice(&[1, m as isize]);
        layout.src_strides[..2].copy_from_slice(&[n as isize, 1]);
        let tiles = pair_tiles::<f64, f64>(&layout).expect("a transposed copy is tiled");
        let blocks = &tiles.blocks[..2];
        let load = l1_set_load(blocks, &[&[1, m as isize], &[n as isize, 1]], &[8, 8]);
        assert!(
            load <= LIVE_LINES,
            "copy [{m}, {n}] transposed: tiles {blocks:?} keep {load} lines live in one 8-way L1 set"
        );
    }
}

/// Layouts without a large-stride run under a reused operand, a destination
/// with zero-stride (reduced) axes, and the transposed set: the planner's tiles
/// never exceed the model's, equal them where no line is kept live, and keep
/// every destination zero-stride axis at the model's block.
#[test]
fn planned_tiles_never_exceed_the_model_and_keep_the_reduced_axes() {
    let mut cases = transposed_operand_layouts();
    cases.extend(transposed_pair_layouts());
    for (m, n) in [(2048usize, 2048usize), (2047, 2047), (1000, 4100)] {
        let d = col_major(&[m, n]);
        let t = stored(&[m, n], &[1, 0]);
        // No operand keeps a line live: the model's tiles exactly.
        cases.push((
            "column-major",
            vec![m, n],
            vec![d.clone(), d.clone(), d.clone()],
        ));
        cases.push((
            "row broadcast",
            vec![m, n],
            vec![d.clone(), d.clone(), vec![0, 1]],
        ));
        cases.push((
            "column broadcast",
            vec![m, n],
            vec![d.clone(), d.clone(), vec![1, 0]],
        ));
        // A reduction into a vector: the destination's zero-stride axis.
        cases.push(("sum over columns", vec![m, n], vec![vec![1, 0], t.clone()]));
        cases.push(("sum over rows", vec![m, n], vec![vec![0, 1], d.clone()]));
        cases.push((
            "sum over rows of a transpose",
            vec![m, n],
            vec![vec![0, 1], t],
        ));
    }
    for elem in [8usize, 4] {
        for (name, dims, strides) in &cases {
            let refs: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
            let sizes = vec![elem; refs.len()];
            let (fused, ordered, plan) = build_plan_fused(dims, &refs, Some(0), &sizes);
            let ordered: Vec<&[isize]> = ordered.iter().map(|s| s.as_slice()).collect();
            let model = crate::block::compute_block_sizes(&fused, &ordered, &sizes);
            for axis in 0..fused.len() {
                assert!(
                    (1..=model[axis]).contains(&plan.block[axis]),
                    "{name} {dims:?} elem {elem}: tiles {:?} outside 1..=model {:?}",
                    plan.block.as_slice(),
                    model.as_slice(),
                );
                if ordered[0][axis] == 0 {
                    assert_eq!(
                        plan.block[axis], model[axis],
                        "{name} {dims:?} elem {elem}: reduced axis {axis} moved from the model"
                    );
                }
            }
            if l1_set_load(&model, &ordered, &sizes) <= LIVE_LINES {
                assert_eq!(
                    plan.block.as_slice(),
                    model.as_slice(),
                    "{name} {dims:?} elem {elem}: tiles within the sets must stay the model's"
                );
            }
            // A fold has no destination; its tiles order the accumulator and
            // stay the model's.
            let (fold_dims, fold_strides, fold_plan) = build_plan_fused(dims, &refs, None, &sizes);
            let fold_strides: Vec<&[isize]> = fold_strides.iter().map(|s| s.as_slice()).collect();
            assert_eq!(
                fold_plan.block.as_slice(),
                crate::block::compute_block_sizes(&fold_dims, &fold_strides, &sizes).as_slice(),
                "{name} {dims:?} elem {elem}: a fold's tiles moved from the model"
            );
        }
    }
}

/// `dot` folds into one accumulator in tile order, so its plan keeps the
/// model's tiles even where a destination-writing plan of the same layout is
/// fitted to the L1 sets (#287): its result bits stay those of the model.
#[test]
fn a_fold_keeps_the_model_tiles_where_a_destination_is_fitted() {
    for (m, n) in [(2048usize, 2048usize), (1024, 4096)] {
        let a = col_major(&[m, n]);
        let b = stored(&[m, n], &[1, 0]);
        let refs: [&[isize]; 2] = [&a, &b];
        let total = m * n;
        let (fused, ordered, fold) = fold_plan(&[m, n], &refs, 0, &[8, 8], total);
        let ordered: Vec<&[isize]> = ordered.iter().map(|s| s.as_slice()).collect();
        let model = crate::block::compute_block_sizes(&fused, &ordered, &[8, 8]);
        assert_eq!(fold.block.as_slice(), model.as_slice(), "[{m}, {n}]");
        let (_, _, fitted) = pointwise_plan(&[m, n], &refs, Some(0), &[8, 8], total);
        assert_ne!(
            fitted.block, fold.block,
            "[{m}, {n}]: the layout must be one the fit moves"
        );
    }
}
