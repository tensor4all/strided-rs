//! Serial versus pooled `Exec` equivalence for dynamic slice and
//! dynamic update slice, including the rank-1 contiguous memcpy path.

use core::mem::MaybeUninit;

use super::{DynamicSlicePlan, DynamicUpdateSlicePlan, RawStridedMut, RawStridedRef};
use crate::cost::KernelClass;
use crate::threading::run_serial_and_parallel;

#[test]
fn dynamic_slice_rank_one_contiguous_parallel_matches_serial() {
    let operand: Vec<f64> = (0..100_003).map(|v| v as f64 * 0.25).collect();
    // Odd lengths above the threshold; a start that needs clamping.
    for (len, start) in [(40_001usize, 7i64), (65_539, 99_999), (0, 5), (1, 3)] {
        let plan =
            DynamicSlicePlan::compile(&[100_003], &[1], &[1], &[1], &[len], &[1], &[len]).unwrap();
        assert!(plan.uses_rank_one_contiguous_path());
        let clamped = (start as usize).min(100_003 - len);
        let expected = operand[clamped..clamped + len].to_vec();
        let starts = [start];
        let source = RawStridedRef::new(&operand, &[100_003], &[1], 0).unwrap();
        let start_ref = RawStridedRef::new(&starts, &[1], &[1], 0).unwrap();
        let dest_dims = [len];
        let (serial, parallel) = run_serial_and_parallel(|mut exec| {
            let mut out = vec![-1.0f64; len];
            let mut dest = RawStridedMut::new(&mut out, &dest_dims, &[1], 0).unwrap();
            plan.execute(&mut dest, &source, &start_ref, exec.reborrow())
                .unwrap();

            let mut storage = vec![MaybeUninit::<f64>::uninit(); len];
            let mut dest = RawStridedMut::new(&mut storage, &dest_dims, &[1], 0).unwrap();
            plan.execute_uninit(&mut dest, &source, &start_ref, exec.reborrow())
                .unwrap();
            // SAFETY: the dense destination reaches every slot and a
            // successful `execute_uninit` initializes every reachable slot.
            let uninit: Vec<f64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
            (out, uninit)
        });
        assert_eq!(serial.0, expected, "len {len}");
        assert_eq!(parallel.0, expected, "len {len}");
        assert_eq!(serial.1, expected, "len {len}");
        assert_eq!(parallel.1, expected, "len {len}");
    }
}

#[test]
fn dynamic_update_slice_parallel_matches_serial() {
    // Rank 1 contiguous (memcpy path) and rank 2 with a reversed operand axis.
    let n = 90_001usize;
    let operand: Vec<i64> = (0..n as i64).collect();
    let update: Vec<i64> = (0..40_001).map(|v| -v - 1).collect();
    let plan = DynamicUpdateSlicePlan::compile(&[n], &[1], &[1], &[1], &[40_001], &[1], &[n], &[1])
        .unwrap();
    assert!(plan.uses_rank_one_contiguous_path());
    let starts = [12_345i64];
    let full = [n];
    let source = RawStridedRef::new(&operand, &full, &[1], 0).unwrap();
    let update_ref = RawStridedRef::new(&update, &[40_001], &[1], 0).unwrap();
    let start_ref = RawStridedRef::new(&starts, &[1], &[1], 0).unwrap();
    let mut expected = operand.clone();
    expected[12_345..12_345 + 40_001].copy_from_slice(&update);
    let (serial, parallel) = run_serial_and_parallel(|mut exec| {
        let mut out = vec![0i64; n];
        let mut dest = RawStridedMut::new(&mut out, &full, &[1], 0).unwrap();
        plan.execute(&mut dest, &source, &update_ref, &start_ref, exec.reborrow())
            .unwrap();
        out
    });
    assert_eq!(serial, expected);
    assert_eq!(parallel, expected);

    let dims = [301usize, 211];
    let op_strides = [-1isize, 301];
    let op_offset = 300isize;
    let operand: Vec<i64> = (0..(301 * 211) as i64).collect();
    let update: Vec<i64> = (0..(101 * 51) as i64).map(|v| -v - 1).collect();
    let plan = DynamicUpdateSlicePlan::compile(
        &dims,
        &op_strides,
        &[2],
        &[1],
        &[101, 51],
        &[1, 101],
        &dims,
        &[1, 301],
    )
    .unwrap();
    let starts = [17i64, 190];
    let source = RawStridedRef::new(&operand, &dims, &op_strides, op_offset).unwrap();
    let update_ref = RawStridedRef::new(&update, &[101, 51], &[1, 101], 0).unwrap();
    let start_ref = RawStridedRef::new(&starts, &[2], &[1], 0).unwrap();
    let mut expected = vec![0i64; 301 * 211];
    for j in 0..211 {
        for i in 0..301 {
            expected[i + 301 * j] = operand[(op_offset - i as isize + 301 * j as isize) as usize];
        }
    }
    // Start 190 on axis 1 clamps to 211 - 51 = 160.
    for j in 0..51 {
        for i in 0..101 {
            expected[(17 + i) + 301 * (160 + j)] = update[i + 101 * j];
        }
    }
    let (serial, parallel) = run_serial_and_parallel(|mut exec| {
        let mut out = vec![0i64; 301 * 211];
        let mut dest = RawStridedMut::new(&mut out, &dims, &[1, 301], 0).unwrap();
        plan.execute(&mut dest, &source, &update_ref, &start_ref, exec.reborrow())
            .unwrap();
        out
    });
    assert_eq!(serial, expected);
    assert_eq!(parallel, expected);
}

#[test]
fn dynamic_slice_rank_six_parallel_matches_serial() {
    let dims = [3usize, 5, 7, 9, 11, 5];
    let strides = [1isize, 3, 15, 105, 945, 10_395];
    let total: usize = dims.iter().product();
    let operand: Vec<i64> = (0..total as i64).collect();
    let slice = [3usize, 4, 7, 8, 11, 5];
    let dest_strides = [1isize, 3, 12, 84, 672, 7_392];
    let plan =
        DynamicSlicePlan::compile(&dims, &strides, &[6], &[1], &slice, &dest_strides, &slice)
            .unwrap();
    let starts = [0i64, 1, 0, 1, 0, 0];
    let source = RawStridedRef::new(&operand, &dims, &strides, 0).unwrap();
    let start_ref = RawStridedRef::new(&starts, &[6], &[1], 0).unwrap();
    let out_len: usize = slice.iter().product();
    let (serial, parallel) = run_serial_and_parallel(|mut exec| {
        let mut out = vec![-1i64; out_len];
        let mut dest = RawStridedMut::new(&mut out, &slice, &dest_strides, 0).unwrap();
        plan.execute(&mut dest, &source, &start_ref, exec.reborrow())
            .unwrap();
        out
    });
    assert_eq!(serial, parallel);
    // Spot check: dest index (0,0,0,0,0,0) maps to operand (0,1,0,1,0,0).
    assert_eq!(serial[0], 3 + 105);
    assert!(!serial.contains(&-1));
}

/// The compiled non-fused window replay above the copy threshold runs on both
/// `Exec` contexts, so the per-worker replay starts are compared against the
/// same reference.
#[test]
fn dynamic_update_generic_replay_parallel_matches_serial() {
    let dims = [2usize, 2, 2, 2, 2, 2, 2, 2, 320];
    let operand_strides = [1isize, 2, 4, 8, 16, 32, 64, 128, 256];
    let update_dims = [2usize, 2, 2, 2, 2, 2, 2, 2, 256];
    let update_strides = [1isize, 3, 7, 15, 31, 63, 127, 255, 511];
    let dest_strides = [40960isize, 20480, 10240, 5120, 2560, 1280, 640, 320, 1];
    let starts = [i64::MAX, 0, 5, i64::MIN, 3, 0, 1, 2, 40];
    let plan = DynamicUpdateSlicePlan::compile(
        &dims,
        &operand_strides,
        &[9],
        &[1],
        &update_dims,
        &update_strides,
        &dims,
        &dest_strides,
    )
    .unwrap();
    assert!(
        plan.window.is_none(),
        "a non-fusible rank-9 window keeps the compiled non-fused replay"
    );
    assert!(!plan.uses_rank_one_contiguous_path());

    let len = |dims: &[usize], strides: &[isize]| -> usize {
        1 + dims
            .iter()
            .zip(strides.iter())
            .map(|(&dim, &stride)| (dim as isize - 1) * stride.abs())
            .sum::<isize>() as usize
    };
    let operand: Vec<i64> = (0..len(&dims, &operand_strides) as i64).collect();
    let update: Vec<i64> = (0..len(&update_dims, &update_strides) as i64)
        .map(|value| -value - 1)
        .collect();
    let expected = super::tests::dynamic_update_reference(
        &dims,
        &operand_strides,
        &operand,
        &update_dims,
        &update_strides,
        &update,
        &starts,
        &dest_strides,
    );

    let source = RawStridedRef::new(&operand, &dims, &operand_strides, 0).unwrap();
    let update_ref = RawStridedRef::new(&update, &update_dims, &update_strides, 0).unwrap();
    let start_ref = RawStridedRef::new(&starts, &[9], &[1], 0).unwrap();
    let update_len: usize = update_dims.iter().product();
    // The update replay is gated on the window length, not the operand length:
    // execute_update_with_writer dispatches with plan.total.
    assert_eq!(update_len, plan.total);
    let (serial, parallel) = run_serial_and_parallel(|mut exec| {
        // The same gate the plan applies: a pooled context must really fan out
        // for this length, or this test would silently re-run the serial path.
        if !exec.is_serial() {
            assert!(
                !crate::cost::for_len_tree(exec.reborrow(), KernelClass::Copy, update_len)
                    .is_serial(),
                "the pooled run must fan out for {update_len} update elements"
            );
        }
        let mut out = vec![-1i64; expected.len()];
        let mut dest = RawStridedMut::new(&mut out, &dims, &dest_strides, 0).unwrap();
        plan.execute(&mut dest, &source, &update_ref, &start_ref, exec.reborrow())
            .unwrap();

        let mut storage = vec![MaybeUninit::<i64>::uninit(); expected.len()];
        let mut dest = RawStridedMut::new(&mut storage, &dims, &dest_strides, 0).unwrap();
        plan.execute_uninit(&mut dest, &source, &update_ref, &start_ref, exec.reborrow())
            .unwrap();
        // SAFETY: the dense destination reaches every slot and a successful
        // `execute_uninit` initializes every reachable slot.
        let uninit: Vec<i64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
        (out, uninit)
    });
    assert_eq!(serial.0, expected);
    assert_eq!(parallel.0, expected);
    assert_eq!(serial.1, expected);
    assert_eq!(parallel.1, expected);
}
