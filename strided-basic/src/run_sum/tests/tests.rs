//! Sorted-run reduction against a scalar reference: random topologies, named
//! edge shapes (empty runs, one run, degree skew, strided rows), payload rank
//! 0..3, `f32`/`f64`/`Complex64`, `_into` vs `_acc`, and the typed errors.

use core::ops::Add;

use num_complex::Complex64;
use num_traits::Zero;
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{RunIndex, RunSumPlan};
use crate::{Exec, RawStridedMut, RawStridedRef, StridedError};

/// Random topologies per property test, reduced under Miri so that lane stays
/// runnable.
fn iterations() -> usize {
    if cfg!(miri) {
        4
    } else {
        500
    }
}

/// One scalar type of the reduction, with the bitwise comparison the budget and
/// `_into`/`_acc` claims need.
pub(super) trait Sample:
    Copy + Zero + Add<Output = Self> + PartialEq + core::fmt::Debug + Send + Sync
{
    fn sample(rng: &mut StdRng) -> Self;
    /// A deterministic value whose magnitude varies with `index`, so a
    /// reordered sum of two different indices need not carry the same bits.
    fn from_index(index: usize) -> Self;
    fn bits_eq(self, other: Self) -> bool;
}

impl Sample for f32 {
    fn sample(rng: &mut StdRng) -> Self {
        rng.gen_range(-1.0f32..1.0)
    }
    fn from_index(index: usize) -> Self {
        (index as f32).sin() * (1.0 + (index % 13) as f32)
    }
    fn bits_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl Sample for f64 {
    fn sample(rng: &mut StdRng) -> Self {
        rng.gen_range(-1.0f64..1.0)
    }
    fn from_index(index: usize) -> Self {
        (index as f64).sin() * (1.0 + (index % 13) as f64)
    }
    fn bits_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl Sample for Complex64 {
    fn sample(rng: &mut StdRng) -> Self {
        Complex64::new(f64::sample(rng), f64::sample(rng))
    }
    fn from_index(index: usize) -> Self {
        Complex64::new(
            <f64 as Sample>::from_index(index),
            <f64 as Sample>::from_index(index + 1),
        )
    }
    fn bits_eq(self, other: Self) -> bool {
        self.re.to_bits() == other.re.to_bits() && self.im.to_bits() == other.im.to_bits()
    }
}

/// A strided layout of `dims`: contiguous in a rotated axis order, with random
/// gaps between axes when asked, and optionally a reversed axis.
struct Layout {
    strides: Vec<isize>,
    offset: isize,
    len: usize,
}

fn make_layout(rng: &mut StdRng, dims: &[usize], gapped: bool, reversed: bool) -> Layout {
    let rank = dims.len();
    let mut order: Vec<usize> = (0..rank).collect();
    if rank > 1 {
        let rotation = rng.gen_range(0..rank);
        order.rotate_left(rotation);
    }
    let mut strides = vec![0isize; rank];
    let mut next = 1usize;
    for (position, &axis) in order.iter().enumerate() {
        if gapped && position > 0 && rng.gen_bool(0.4) {
            next += rng.gen_range(1..=2);
        }
        strides[axis] = next as isize;
        next *= dims[axis];
    }
    if reversed && rank > 0 {
        let axis = order[rank - 1];
        strides[axis] = -strides[axis];
    }
    layout_of(dims, strides)
}

fn layout_of(dims: &[usize], strides: Vec<isize>) -> Layout {
    let mut min_offset = 0isize;
    let mut max_offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        let reach = (dim as isize - 1) * stride;
        if reach < 0 {
            min_offset += reach;
        } else {
            max_offset += reach;
        }
    }
    Layout {
        strides,
        offset: -min_offset,
        len: (max_offset - min_offset) as usize + 1,
    }
}

/// Every payload element's offset, in destination order.
fn payload_offsets(dims: &[usize], strides: &[isize]) -> Vec<isize> {
    let mut offsets = vec![0isize];
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        let mut next = Vec::with_capacity(offsets.len() * dim);
        for &base in &offsets {
            for index in 0..dim {
                next.push(base + index as isize * stride);
            }
        }
        offsets = next;
    }
    offsets
}

/// One case: the payload extents, the destination run of every source row, and
/// the run count.
struct Case {
    payload: Vec<usize>,
    dest_of_row: Vec<i32>,
    runs: usize,
}

impl Case {
    fn rows(&self) -> usize {
        self.dest_of_row.len()
    }

    fn src_dims(&self) -> Vec<usize> {
        let mut dims = vec![self.rows()];
        dims.extend_from_slice(&self.payload);
        dims
    }

    fn dest_dims(&self) -> Vec<usize> {
        let mut dims = vec![self.runs];
        dims.extend_from_slice(&self.payload);
        dims
    }

    fn is_grouped(&self) -> bool {
        self.dest_of_row.windows(2).all(|pair| pair[0] <= pair[1])
    }

    fn grouped_ptr(&self) -> Vec<usize> {
        let mut ptr = vec![0usize];
        for run in 0..self.runs {
            ptr.push(
                self.dest_of_row
                    .iter()
                    .filter(|&&destination| destination as usize == run)
                    .count()
                    + ptr[run],
            );
        }
        ptr
    }
}

/// The destination slots the reduction writes, with the seed each element
/// accumulates into: `T::zero()` for `_into`, the previous destination value
/// for `_acc`.
struct Expected<T> {
    values: Vec<T>,
    touched: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn reference<T: Sample>(
    case: &Case,
    into: bool,
    src: &[T],
    src_strides: &[isize],
    src_offset: isize,
    dest_before: &[T],
    dest_strides: &[isize],
    dest_offset: isize,
) -> Expected<T> {
    let dest_payload = payload_offsets(&case.payload, &dest_strides[1..]);
    let src_payload = payload_offsets(&case.payload, &src_strides[1..]);
    let mut values = dest_before.to_vec();
    let mut touched = Vec::new();
    for run in 0..case.runs {
        for (&d_off, &s_off) in dest_payload.iter().zip(src_payload.iter()) {
            let dest_slot = (dest_offset + run as isize * dest_strides[0] + d_off) as usize;
            touched.push(dest_slot);
            let mut accumulator = if into { T::zero() } else { values[dest_slot] };
            for (row, &run_of_row) in case.dest_of_row.iter().enumerate() {
                if run_of_row as usize != run {
                    continue;
                }
                let src_slot = (src_offset + row as isize * src_strides[0] + s_off) as usize;
                accumulator = accumulator + src[src_slot];
            }
            values[dest_slot] = accumulator;
        }
    }
    Expected { values, touched }
}

/// Check one case against the scalar reference; returns the number of
/// destination slots the reduction owns.
///
/// The buffers, both `_acc` variants and the untouched slots are compared here,
/// so a caller only decides what the slot count should be.
fn run_case<T: Sample>(rng: &mut StdRng, case: &Case) -> usize {
    let src_dims = case.src_dims();
    let dest_dims = case.dest_dims();
    let (src_gapped, src_reversed) = (rng.gen_bool(0.5), rng.gen_bool(0.3));
    let (dest_gapped, dest_reversed) = (rng.gen_bool(0.5), rng.gen_bool(0.3));
    let src_layout = make_layout(rng, &src_dims, src_gapped, src_reversed);
    let dest_layout = make_layout(rng, &dest_dims, dest_gapped, dest_reversed);

    let from_offsets = case.is_grouped() && rng.gen_bool(0.5);
    let runs = match from_offsets {
        true => RunIndex::from_ptr(case.grouped_ptr()).unwrap(),
        _ => RunIndex::from_dest(&case.dest_of_row, case.runs).unwrap(),
    };
    let plan = RunSumPlan::compile(
        &src_dims,
        &src_layout.strides,
        &dest_dims,
        &dest_layout.strides,
        &runs,
    )
    .unwrap();

    let src: Vec<T> = (0..src_layout.len).map(|_| T::sample(rng)).collect();
    let garbage: Vec<T> = (0..dest_layout.len).map(|_| T::sample(rng)).collect();
    let src_ref =
        RawStridedRef::new(&src, &src_dims, &src_layout.strides, src_layout.offset).unwrap();

    // `_into` overwrites its rows and leaves everything else alone.
    let mut into = garbage.clone();
    let mut into_mut = RawStridedMut::new(
        &mut into,
        &dest_dims,
        &dest_layout.strides,
        dest_layout.offset,
    )
    .unwrap();
    plan.execute_into(&mut into_mut, &src_ref, &runs, Exec::serial())
        .unwrap();

    // `_acc` from a `T::zero()` seed carries the same bits as `_into`.
    let mut acc_from_zero = vec![T::zero(); dest_layout.len];
    let mut acc_mut = RawStridedMut::new(
        &mut acc_from_zero,
        &dest_dims,
        &dest_layout.strides,
        dest_layout.offset,
    )
    .unwrap();
    plan.execute_acc(&mut acc_mut, &src_ref, &runs, Exec::serial())
        .unwrap();

    // `_acc` into an arbitrary seed adds to it.
    let mut acc_seeded = garbage.clone();
    let mut seeded_mut = RawStridedMut::new(
        &mut acc_seeded,
        &dest_dims,
        &dest_layout.strides,
        dest_layout.offset,
    )
    .unwrap();
    plan.execute_acc(&mut seeded_mut, &src_ref, &runs, Exec::serial())
        .unwrap();

    let into_expected = reference::<T>(
        case,
        true,
        &src,
        &src_layout.strides,
        src_layout.offset,
        &garbage,
        &dest_layout.strides,
        dest_layout.offset,
    );
    let seeded_expected = reference::<T>(
        case,
        false,
        &src,
        &src_layout.strides,
        src_layout.offset,
        &garbage,
        &dest_layout.strides,
        dest_layout.offset,
    );

    for &slot in &into_expected.touched {
        assert!(
            into[slot].bits_eq(into_expected.values[slot]),
            "_into at slot {slot}: payload {:?} dest_of_row {:?} runs {} src {:?}/{}/{} dest {:?}/{}/{} -> got {:?} want {:?}",
            case.payload,
            case.dest_of_row,
            case.runs,
            src_layout.strides,
            src_layout.offset,
            src_layout.len,
            dest_layout.strides,
            dest_layout.offset,
            dest_layout.len,
            into[slot],
            into_expected.values[slot]
        );
        assert!(
            acc_from_zero[slot].bits_eq(into_expected.values[slot]),
            "_acc from zero at slot {slot}"
        );
        assert!(
            acc_seeded[slot].bits_eq(seeded_expected.values[slot]),
            "_acc into a seed at slot {slot}"
        );
    }
    let touched = into_expected.touched;
    let mut is_touched = vec![false; dest_layout.len];
    for &slot in &touched {
        is_touched[slot] = true;
    }
    for (slot, &was_touched) in is_touched.iter().enumerate() {
        if !was_touched {
            assert!(
                into[slot].bits_eq(garbage[slot]),
                "`_into` wrote slot {slot} outside its rows"
            );
            assert!(
                acc_seeded[slot].bits_eq(garbage[slot]),
                "`_acc` wrote slot {slot} outside its rows"
            );
        }
    }

    touched.len()
}

fn random_case(rng: &mut StdRng) -> Case {
    let rank = rng.gen_range(0..=3);
    let payload: Vec<usize> = (0..rank).map(|_| rng.gen_range(1..=4)).collect();
    let rows = rng.gen_range(0..=12);
    // At least one run: a case with no runs has no rows either, and is covered
    // by `zero_rows_and_zero_runs_are_empty_plans` instead.
    let runs = rng.gen_range(1..=rows.min(5) + 1);
    let mut dest_of_row: Vec<i32> = (0..rows).map(|_| rng.gen_range(0..runs) as i32).collect();
    dest_of_row.sort_unstable();
    if !dest_of_row.is_empty() && rng.gen_bool(0.5) {
        // A rotation of the destinations keeps the multiset and breaks the
        // grouping, so `from_dest` has to place a permutation.
        let shift = rng.gen_range(0..dest_of_row.len());
        dest_of_row.rotate_left(shift);
    }
    Case {
        payload,
        dest_of_row,
        runs,
    }
}

#[test]
fn property_matches_scalar_reference() {
    let mut rng = StdRng::seed_from_u64(479);
    for _ in 0..iterations() {
        let case = random_case(&mut rng);
        run_case::<f64>(&mut rng, &case);
    }
}

#[test]
fn property_holds_for_f32_and_complex() {
    let mut rng = StdRng::seed_from_u64(4790);
    for _ in 0..iterations() {
        let case = random_case(&mut rng);
        run_case::<f32>(&mut rng, &case);
        run_case::<Complex64>(&mut rng, &case);
    }
}

#[test]
fn one_run_holds_every_row() {
    let mut rng = StdRng::seed_from_u64(4791);
    let case = Case {
        payload: vec![3, 2],
        dest_of_row: vec![0; 9],
        runs: 1,
    };
    let touched = run_case::<f64>(&mut rng, &case);
    assert_eq!(touched, 6);
}

#[test]
fn empty_runs_write_zero_and_accumulate_nothing() {
    // Runs 0, 2 and 4 are empty. Destination dims `[5, 2]`, strides `[1, 5]`, so
    // run `r` writes slots `r` and `r + 5` and the buffer is readable here.
    let runs = RunIndex::from_dest(&[1i32, 1, 3], 5).unwrap();
    let plan = RunSumPlan::compile(&[3, 2], &[1, 3], &[5, 2], &[1, 5], &runs).unwrap();
    let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src_ref = RawStridedRef::new(&src, &[3, 2], &[1, 3], 0).unwrap();

    let mut into = [f64::NAN; 10];
    let mut dest = RawStridedMut::new(&mut into, &[5, 2], &[1, 5], 0).unwrap();
    plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();
    assert_eq!(into, [0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 9.0, 0.0, 6.0, 0.0]);

    let mut acc = [1.0f64; 10];
    let mut dest = RawStridedMut::new(&mut acc, &[5, 2], &[1, 5], 0).unwrap();
    plan.execute_acc(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();
    assert_eq!(acc, [1.0, 4.0, 1.0, 4.0, 1.0, 1.0, 10.0, 1.0, 7.0, 1.0]);
}

#[test]
fn degree_skew_balances_by_rows() {
    let mut rng = StdRng::seed_from_u64(4793);
    // One hub run of 4000 rows and 200 singletons.
    let mut dest_of_row = vec![0i32; 4000];
    for run in 1..=200 {
        dest_of_row.push(run);
    }
    let case = Case {
        payload: vec![4],
        dest_of_row,
        runs: 201,
    };
    let touched = run_case::<f64>(&mut rng, &case);
    assert_eq!(touched, 201 * 4);
}

#[test]
fn reversed_and_gapped_rows_match_the_reference() {
    // A reversed source row axis and a transposed payload, both with gaps.
    let case = Case {
        payload: vec![2, 3],
        dest_of_row: vec![0, 0, 1, 2, 2, 2],
        runs: 3,
    };
    let src_dims = case.src_dims();
    let dest_dims = case.dest_dims();
    let src_layout = layout_of(&src_dims, vec![-7, 6, 2]);
    let dest_layout = layout_of(&dest_dims, vec![6, 1, 2]);
    let runs = RunIndex::from_dest(&case.dest_of_row, case.runs).unwrap();
    let plan = RunSumPlan::compile(
        &src_dims,
        &src_layout.strides,
        &dest_dims,
        &dest_layout.strides,
        &runs,
    )
    .unwrap();
    let src: Vec<f64> = (0..src_layout.len)
        .map(|value| value as f64 * 0.5)
        .collect();
    let src_ref =
        RawStridedRef::new(&src, &src_dims, &src_layout.strides, src_layout.offset).unwrap();
    let mut into = vec![f64::NAN; dest_layout.len];
    let mut dest = RawStridedMut::new(
        &mut into,
        &dest_dims,
        &dest_layout.strides,
        dest_layout.offset,
    )
    .unwrap();
    plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();

    let expected = reference::<f64>(
        &case,
        true,
        &src,
        &src_layout.strides,
        src_layout.offset,
        &into,
        &dest_layout.strides,
        dest_layout.offset,
    );
    for &slot in &expected.touched {
        assert_eq!(into[slot].to_bits(), expected.values[slot].to_bits());
    }
}

#[test]
fn zero_rows_and_zero_runs_are_empty_plans() {
    let runs = RunIndex::from_ptr(vec![0]).unwrap();
    let plan = RunSumPlan::compile(&[0, 3], &[1, 0], &[0, 3], &[1, 0], &runs).unwrap();
    let mut into: Vec<f64> = Vec::new();
    let mut dest = RawStridedMut::new(&mut into, &[0, 3], &[1, 0], 0).unwrap();
    let src: Vec<f64> = Vec::new();
    let src_ref = RawStridedRef::new(&src, &[0, 3], &[1, 0], 0).unwrap();
    plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();
    plan.execute_acc(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();

    // Runs with no rows at all: every run is empty, so `_into` writes zeros.
    let runs = RunIndex::from_ptr(vec![0, 0, 0]).unwrap();
    let plan = RunSumPlan::compile(&[0], &[1], &[2], &[1], &runs).unwrap();
    let mut into = vec![1.0f64, 2.0];
    let mut dest = RawStridedMut::new(&mut into, &[2], &[1], 0).unwrap();
    let src: Vec<f64> = Vec::new();
    let src_ref = RawStridedRef::new(&src, &[0], &[1], 0).unwrap();
    plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();
    assert_eq!(into, vec![0.0, 0.0]);
}

#[test]
fn scalar_payload_rows_sum_their_runs() {
    let runs = RunIndex::from_dest(&[0i32, 1, 0], 2).unwrap();
    assert_eq!(runs.ptr(), &[0, 2, 3]);
    let plan = RunSumPlan::compile(&[3], &[1], &[2], &[1], &runs).unwrap();
    let src = [1.5f64, 3.0, 4.5];
    let mut into = [f64::NAN, f64::NAN];
    let mut dest = RawStridedMut::new(&mut into, &[2], &[1], 0).unwrap();
    let src_ref = RawStridedRef::new(&src, &[3], &[1], 0).unwrap();
    plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();
    assert_eq!(into, [6.0, 3.0]);
}

#[test]
fn from_ptr_accepts_grouped_rows_and_rejects_bad_offsets() {
    let grouped = RunIndex::from_ptr(vec![0, 2, 5]).unwrap();
    assert_eq!(grouped.ptr(), &[0, 2, 5]);
    assert!(grouped.perm().is_none());

    // A sorted destination takes the identity permutation from either entry.
    let sorted = RunIndex::from_dest(&[0i64, 0, 1, 1, 1], 2).unwrap();
    assert_eq!(sorted.ptr(), &[0, 2, 5]);
    assert!(sorted.perm().is_none());

    let unsorted = RunIndex::from_dest(&[1i64, 0, 1, 0], 2).unwrap();
    assert_eq!(unsorted.ptr(), &[0, 2, 4]);
    assert_eq!(unsorted.perm(), Some(&[1usize, 3, 0, 2][..]));
}

#[test]
fn empty_source_rows_zero_the_destination_without_walking_it() {
    // A zero row extent makes the source empty, so `RawStridedRef` validates no
    // bound for its payload strides: the storage is empty whatever they say.
    // The destination rows must still be zeroed, and the source must not be
    // walked (`isize::MAX` would overflow the walk's odometer).
    let src: [f64; 0] = [];
    let src_dims = [0, 2, 3];
    let src_strides = [0, 1, isize::MAX];
    let dest_dims = [1, 2, 3];
    let dest_strides = [6, 1, 2];
    let src_ref = RawStridedRef::new(&src, &src_dims, &src_strides, 0).unwrap();
    let runs = RunIndex::from_ptr(vec![0, 0]).unwrap();
    let plan =
        RunSumPlan::compile(&src_dims, &src_strides, &dest_dims, &dest_strides, &runs).unwrap();

    let mut into = [f64::NAN; 6];
    let mut dest = RawStridedMut::new(&mut into, &dest_dims, &dest_strides, 0).unwrap();
    plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();
    assert_eq!(into, [0.0; 6]);

    let mut acc = [1.5f64; 6];
    let mut dest = RawStridedMut::new(&mut acc, &dest_dims, &dest_strides, 0).unwrap();
    plan.execute_acc(&mut dest, &src_ref, &runs, Exec::serial())
        .unwrap();
    assert_eq!(acc, [1.5; 6]);
}

#[test]
fn from_dest_rejects_a_run_count_it_cannot_allocate() {
    assert!(matches!(
        RunIndex::from_dest::<i32>(&[], usize::MAX),
        Err(StridedError::OffsetOverflow)
    ));
}

#[test]
fn run_index_rejects_out_of_range_indices() {
    assert!(matches!(
        RunIndex::from_dest(&[0i32, -1], 4),
        Err(StridedError::IndexOutOfRange { index: -1, runs: 4 })
    ));
    assert!(matches!(
        RunIndex::from_dest(&[4i64], 4),
        Err(StridedError::IndexOutOfRange { index: 4, runs: 4 })
    ));
    assert!(matches!(
        RunIndex::from_dest(&[0i32], 0),
        Err(StridedError::IndexOutOfRange { index: 0, runs: 0 })
    ));
    assert!(matches!(
        RunIndex::from_ptr(Vec::new()),
        Err(StridedError::InvalidRunIndex)
    ));
    assert!(matches!(
        RunIndex::from_ptr(vec![1, 2]),
        Err(StridedError::InvalidRunIndex)
    ));
    assert!(matches!(
        RunIndex::from_ptr(vec![0, 2, 1]),
        Err(StridedError::InvalidRunIndex)
    ));
}

#[test]
fn compile_rejects_layouts_that_do_not_describe_the_runs() {
    let runs = RunIndex::from_ptr(vec![0, 2]).unwrap();
    // Stride list length.
    assert!(matches!(
        RunSumPlan::compile(&[2, 3], &[1], &[1, 3], &[1, 3], &runs),
        Err(StridedError::StrideLengthMismatch)
    ));
    // Rank.
    assert!(matches!(
        RunSumPlan::compile(&[2, 3], &[1, 2], &[1], &[1], &runs),
        Err(StridedError::RankMismatch(2, 1))
    ));
    assert!(matches!(
        RunSumPlan::compile(&[], &[], &[], &[], &runs),
        Err(StridedError::RankMismatch(0, 1))
    ));
    // Row extents against the run index.
    assert!(matches!(
        RunSumPlan::compile(&[3, 3], &[1, 3], &[1, 3], &[1, 3], &runs),
        Err(StridedError::ShapeMismatch(_, _))
    ));
    assert!(matches!(
        RunSumPlan::compile(&[2, 3], &[1, 3], &[2, 3], &[1, 3], &runs),
        Err(StridedError::ShapeMismatch(_, _))
    ));
    // Payload extents.
    assert!(matches!(
        RunSumPlan::compile(&[2, 3], &[1, 2], &[1, 4], &[1, 2], &runs),
        Err(StridedError::ShapeMismatch(_, _))
    ));
    // A destination that maps two rows onto one slot: two runs, stride 0.
    let two_runs = RunIndex::from_ptr(vec![0, 1, 2]).unwrap();
    assert!(matches!(
        RunSumPlan::compile(&[2, 3], &[1, 2], &[2, 3], &[0, 1], &two_runs),
        Err(StridedError::NonInjectiveOutputLayout)
    ));
}

#[test]
fn execute_rejects_a_view_or_topology_that_moved() {
    let runs = RunIndex::from_ptr(vec![0, 2]).unwrap();
    let plan = RunSumPlan::compile(&[2, 3], &[1, 2], &[1, 3], &[1, 2], &runs).unwrap();
    let src = [1.0f64; 6];

    let mut into = [0.0f64; 6];
    let mut dest = RawStridedMut::new(&mut into, &[1, 3], &[2, 1], 0).unwrap();
    let src_ref = RawStridedRef::new(&src, &[2, 3], &[1, 2], 0).unwrap();
    assert!(matches!(
        plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial()),
        Err(StridedError::PlanLayoutMismatch)
    ));

    let mut dest = RawStridedMut::new(&mut into, &[1, 3], &[1, 2], 0).unwrap();
    let wider_src = [1.0f64; 6];
    let src_ref = RawStridedRef::new(&wider_src, &[2, 3], &[3, 1], 0).unwrap();
    assert!(matches!(
        plan.execute_into(&mut dest, &src_ref, &runs, Exec::serial()),
        Err(StridedError::PlanLayoutMismatch)
    ));

    let src_ref = RawStridedRef::new(&wider_src, &[2, 3], &[1, 2], 0).unwrap();
    let other = RunIndex::from_ptr(vec![0, 1, 2]).unwrap();
    assert!(matches!(
        plan.execute_acc(&mut dest, &src_ref, &other, Exec::serial()),
        Err(StridedError::ShapeMismatch(_, _))
    ));
}
