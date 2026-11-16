//! Bitwise equality of the sorted-run reduction across `Exec` budgets 1/2/4/8,
//! for `f64`, `f32` and `Complex64`, including an uneven degree distribution and
//! one hub run.

use num_complex::Complex64;

use super::tests::Sample;
use super::{RunIndex, RunSumPlan};
#[cfg(feature = "parallel")]
use crate::cost::{tree_width, KernelClass};
use crate::threading::test_pool;
use crate::{Exec, RawStridedMut, RawStridedRef};

const PAYLOAD: usize = 8;
const HUB: i32 = 400;

/// Runs over rows grouped by destination, with a rotated row order so the
/// topology needs a permutation. The last run is a hub.
fn topology() -> (Vec<i32>, usize) {
    let mut dest_of_row = Vec::new();
    for run in 0..400usize {
        let degree = 1 + (run * 37) % 97;
        for _ in 0..degree {
            dest_of_row.push(run as i32);
        }
    }
    let rows = dest_of_row.len();
    dest_of_row.extend(std::iter::repeat_n(HUB, rows / 2));
    let runs = 401;
    let shift = 1234 % dest_of_row.len();
    dest_of_row.rotate_left(shift);
    (dest_of_row, runs)
}

/// Bitwise comparison of two results, reported at the first differing slot.
fn assert_same_bits<T: Sample>(reference: &[T], other: &[T], context: &str) {
    assert_eq!(reference.len(), other.len(), "{context}: length");
    for (slot, (&want, &got)) in reference.iter().zip(other.iter()).enumerate() {
        assert!(
            got.bits_eq(want),
            "{context}: slot {slot}: {want:?} vs {got:?}"
        );
    }
}

fn run_with_budget<T: Sample>(budget: usize) -> Vec<T> {
    let (dest_of_row, runs_count) = topology();
    let rows = dest_of_row.len();
    let src_dims = [rows, PAYLOAD];
    let dest_dims = [runs_count, PAYLOAD];
    let strides = [PAYLOAD as isize, 1];
    let src: Vec<T> = (0..rows * PAYLOAD).map(T::from_index).collect();
    let src_ref = RawStridedRef::new(&src, &src_dims, &strides, 0).unwrap();

    let runs = RunIndex::from_dest(&dest_of_row, runs_count).unwrap();
    let plan = RunSumPlan::compile(&src_dims, &strides, &dest_dims, &strides, &runs).unwrap();
    assert_eq!(runs.ptr().len() - 1, runs_count);

    let mut out = vec![T::zero(); runs_count * PAYLOAD];
    let mut dest = RawStridedMut::new(&mut out, &dest_dims, &strides, 0).unwrap();
    let pool = test_pool(budget);
    let exec = Exec::new(&pool, budget).unwrap();
    // Without the `parallel` feature every token is serial, so the gate has no
    // width to open to and the equivalence below holds trivially; with it, the
    // window has to be big enough for the split to happen at all.
    #[cfg(feature = "parallel")]
    if budget > 1 {
        let elements = rows * PAYLOAD;
        assert_eq!(
            tree_width(&exec, KernelClass::Reduce, elements).get(),
            budget,
            "the gate has to open for the split to exist"
        );
    }
    plan.execute_into(&mut dest, &src_ref, &runs, exec).unwrap();
    out
}

fn budgets_agree<T: Sample>() {
    let reference = run_with_budget::<T>(1);
    for budget in [2usize, 4, 8] {
        let other = run_with_budget::<T>(budget);
        assert_same_bits(&reference, &other, &format!("budget {budget}"));
    }
}

#[test]
fn budgets_one_two_four_eight_agree_bit_for_bit() {
    budgets_agree::<f64>();
    budgets_agree::<f32>();
    budgets_agree::<Complex64>();
}

/// The bit-for-bit assertions above can only bite if regrouping a run's rows
/// moves the bits, so the hub payload they run on must not be accidentally
/// order-insensitive.
#[test]
fn regrouping_the_hub_would_move_the_bits() {
    let (dest_of_row, _) = topology();
    let rows = dest_of_row.len();
    let hub: Vec<usize> = (0..rows)
        .filter(|&row| dest_of_row[row] == HUB)
        .map(|row| row * PAYLOAD)
        .collect();
    assert!(hub.len() > 64, "the hub run must be long enough");

    let regrouped = |sum: &dyn Fn(&[usize]) -> f64| -> (u64, u64) {
        let split = hub.len() / 2;
        let whole = sum(&hub);
        let halves = sum(&hub[..split]) + sum(&hub[split..]);
        (whole.to_bits(), halves.to_bits())
    };
    let in_order = |indices: &[usize]| -> f64 {
        indices.iter().fold(0.0f64, |acc, &index| {
            acc + <f64 as Sample>::from_index(index)
        })
    };
    let (whole, halves) = regrouped(&in_order);
    assert_ne!(
        whole, halves,
        "the hub's rows must not sum to the same bits when regrouped"
    );
}

/// `_into` and `_acc` from a `T::zero()` seed hold the same bits at every
/// budget, for every scalar type.
#[test]
fn acc_matches_into_at_every_budget_from_a_zero_seed() {
    fn check<T: Sample>() {
        let (dest_of_row, runs_count) = topology();
        let rows = dest_of_row.len();
        let src_dims = [rows, 4];
        let dest_dims = [runs_count, 4];
        let strides = [4isize, 1];
        let src: Vec<T> = (0..rows * 4).map(T::from_index).collect();
        let src_ref = RawStridedRef::new(&src, &src_dims, &strides, 0).unwrap();
        let runs = RunIndex::from_dest(&dest_of_row, runs_count).unwrap();
        let plan = RunSumPlan::compile(&src_dims, &strides, &dest_dims, &strides, &runs).unwrap();

        for budget in [1usize, 2, 4, 8] {
            let mut into = vec![T::zero(); runs_count * 4];
            let pool = test_pool(budget);
            let exec = Exec::new(&pool, budget).unwrap();
            let mut dest = RawStridedMut::new(&mut into, &dest_dims, &strides, 0).unwrap();
            plan.execute_into(&mut dest, &src_ref, &runs, exec).unwrap();

            let mut acc = vec![T::zero(); runs_count * 4];
            let pool = test_pool(budget);
            let exec = Exec::new(&pool, budget).unwrap();
            let mut dest = RawStridedMut::new(&mut acc, &dest_dims, &strides, 0).unwrap();
            plan.execute_acc(&mut dest, &src_ref, &runs, exec).unwrap();

            assert_same_bits(&into, &acc, &format!("budget {budget}"));
        }
    }
    check::<f64>();
    check::<f32>();
    check::<Complex64>();
}
