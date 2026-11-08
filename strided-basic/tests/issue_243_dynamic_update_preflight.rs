use rand::{rngs::StdRng, Rng, SeedableRng};
use strided_basic::{DynamicUpdateSlicePlan, Exec, RawStridedMut, RawStridedRef};

fn pick_stride(rng: &mut StdRng) -> isize {
    let m = isize::MAX;
    let c = [
        1,
        2,
        3,
        7,
        m / 2,
        m / 3,
        m / 4,
        m / 5,
        m - 1,
        m,
        m / 2 + 1,
        m / 6,
    ];
    let v = c[rng.gen_range(0..c.len())];
    if rng.gen_bool(0.5) {
        -v
    } else {
        v
    }
}
fn pick_off(rng: &mut StdRng) -> isize {
    let m = isize::MAX;
    let c = [
        0,
        1,
        m / 2,
        m / 3,
        2 * (m / 3),
        m / 4,
        3 * (m / 4),
        m - 1,
        m,
        m / 2 + 1,
        m / 5 * 4,
    ];
    c[rng.gen_range(0..c.len())]
}

/// Issue #243 asked whether a validated plan can fail after the operand copy
/// has already mutated `dest`. Every fallible step after the copy (clamped
/// start read, update-window base, replay decode) stays inside the
/// already-validated reachable span of `dest`, `update`, and `starts`, so a
/// plan whose `compile` and `check_call` succeed must never return `Err` from
/// the post-copy phase. Zero-sized elements let the probe use offsets and
/// strides near `isize::MAX`, including layouts whose reachable span ends
/// within one stride of it (the operand copy must not step past the last
/// position of an axis before rewinding).
#[test]
fn issue_243_validated_layouts_never_fail_after_copy() {
    let mut rng = StdRng::seed_from_u64(243);
    let big = [(); usize::MAX];
    let mut dest_data = [(); usize::MAX];
    let mut tried = 0usize;
    let mut extreme = 0usize;
    for _ in 0..60_000 {
        let rank = rng.gen_range(1..=3);
        let dims: Vec<usize> = (0..rank).map(|_| rng.gen_range(1..=3)).collect();
        let ds: Vec<isize> = (0..rank).map(|_| pick_stride(&mut rng)).collect();
        let os: Vec<isize> = (0..rank).map(|_| pick_stride(&mut rng)).collect();
        let ud: Vec<usize> = dims.iter().map(|&d| rng.gen_range(1..=d)).collect();
        let us: Vec<isize> = (0..rank).map(|_| pick_stride(&mut rng)).collect();
        let (doff, ooff, uoff) = (pick_off(&mut rng), pick_off(&mut rng), pick_off(&mut rng));
        let Ok(dest) = RawStridedMut::new(&mut dest_data[..], &dims, &ds, doff) else {
            continue;
        };
        let Ok(op) = RawStridedRef::new(&big[..], &dims, &os, ooff) else {
            continue;
        };
        let Ok(up) = RawStridedRef::new(&big[..], &ud, &us, uoff) else {
            continue;
        };
        let starts: Vec<i64> = (0..rank)
            .map(|_| [i64::MIN, -1, 0, 1, 2, i64::MAX][rng.gen_range(0..6)])
            .collect();
        let sd = [rank];
        let ss = [1isize];
        let st = RawStridedRef::new(&starts[..], &sd, &ss, 0).unwrap();
        let Ok(plan) = DynamicUpdateSlicePlan::compile(&dims, &os, &sd, &ss, &ud, &us, &dims, &ds)
        else {
            continue;
        };
        tried += 1;
        if dims
            .iter()
            .zip(ds.iter())
            .any(|(&d, &s)| d > 1 && s.unsigned_abs() >= (isize::MAX / 6) as usize)
        {
            extreme += 1;
        }
        let mut dest = dest;
        if let Err(e) = plan.execute(&mut dest, &op, &up, &st, Exec::serial()) {
            panic!("post-check failure {e:?} dims={dims:?} ds={ds:?} os={os:?} ud={ud:?} us={us:?} offs={doff},{ooff},{uoff} starts={starts:?}");
        }
    }
    eprintln!("executed {tried} valid cases, {extreme} with extreme dest strides");
    assert!(tried > 1000);
}

/// The clamped window start and the destination window base are resolved
/// before the operand copy, so a rejected call leaves `dest` untouched. The
/// rank-one contiguous path converts the update offset to a buffer index; a
/// negative update offset is the minimal way to reach that rejection, and no
/// checked view can carry one. The descriptor below therefore takes the
/// update view one element into its buffer, so the rejected offset points at
/// the preceding element of the same allocation: even an accidental read stays
/// inside a buffer.
#[test]
fn issue_243_preflight_rejection_leaves_initialized_dest_untouched() {
    let dims = [3usize];
    let operand = [7i32, 8, 9];
    let update_storage = [5i32, 6];
    let update = &update_storage[1..];
    let starts = [0i64];
    let plan =
        DynamicUpdateSlicePlan::compile(&[3], &[1], &[1], &[1], &[1], &[1], &[3], &[1]).unwrap();
    let operand_ref = RawStridedRef::new(&operand, &dims, &[1], 0).unwrap();
    // SAFETY: the view starts outside `update` but inside its buffer, and the
    // preflight rejects the offset before the replay reads the value.
    let update_ref = unsafe { RawStridedRef::new_unchecked(update, &[1], &[1], -1) };
    let start_ref = RawStridedRef::new(&starts, &[1], &[1], 0).unwrap();

    let mut dest_data = [-1i32; 3];
    let mut dest = RawStridedMut::new(&mut dest_data, &dims, &[1], 0).unwrap();
    let result = plan.execute(
        &mut dest,
        &operand_ref,
        &update_ref,
        &start_ref,
        Exec::serial(),
    );
    assert!(
        matches!(result, Err(strided_basic::StridedError::OffsetOverflow)),
        "the rejected window must be reported as an offset overflow, got {result:?}"
    );
    assert_eq!(
        dest_data,
        [-1, -1, -1],
        "a rejected call must not run the operand copy"
    );
}

/// The same ordering guarantee for the uninitialized lifecycle: the rejected
/// window is reported before the copy, so no destination slot is initialized
/// by the failed call.
#[test]
fn issue_243_preflight_rejection_leaves_uninit_dest_untouched() {
    use core::mem::MaybeUninit;
    use strided_basic::execution::dynamic_update_into_uninit;

    let dims = [3usize];
    let operand = [7i32, 8, 9];
    let update_storage = [5i32, 6];
    let update = &update_storage[1..];
    let starts = [0i64];
    let plan =
        DynamicUpdateSlicePlan::compile(&[3], &[1], &[1], &[1], &[1], &[1], &[3], &[1]).unwrap();
    let operand_ref = RawStridedRef::new(&operand, &dims, &[1], 0).unwrap();
    // SAFETY: the view starts outside `update` but inside its buffer, and the
    // preflight rejects the offset before the replay reads the value.
    let update_ref = unsafe { RawStridedRef::new_unchecked(update, &[1], &[1], -1) };
    let start_ref = RawStridedRef::new(&starts, &[1], &[1], 0).unwrap();

    let mut values = [MaybeUninit::new(-1i32); 3];
    let mut dest = RawStridedMut::new(&mut values, &dims, &[1], 0).unwrap();
    // SAFETY: the sentinel-initialized destination does not overlap the
    // inputs, and the rejected call writes no output slot.
    let result = unsafe {
        dynamic_update_into_uninit(
            &plan,
            &mut dest,
            &operand_ref,
            &update_ref,
            &start_ref,
            Exec::serial(),
        )
    };
    assert!(matches!(
        result,
        Err(strided_basic::StridedError::OffsetOverflow)
    ));
    // SAFETY: every slot was initialized with the sentinel before the call and
    // the failed call wrote none of them.
    let read: Vec<i32> = values.iter().map(|v| unsafe { v.assume_init() }).collect();
    assert_eq!(read, [-1, -1, -1], "a rejected call must not run the copy");
}

/// Zero rank and an empty update window keep their copy-then-window semantics
/// without any offset arithmetic over an empty dimension: a rank-0 window
/// covers the single destination element, and a zero-extent window leaves the
/// copied operand in place.
#[test]
fn issue_243_zero_rank_and_empty_window_copy_only() {
    let mut scalar = [-1i32];
    let plan = DynamicUpdateSlicePlan::compile(&[], &[], &[0], &[1], &[], &[], &[], &[]).unwrap();
    let operand = [7i32];
    let update = [5i32];
    let operand_ref = RawStridedRef::new(&operand, &[], &[], 0).unwrap();
    let update_ref = RawStridedRef::new(&update, &[], &[], 0).unwrap();
    let no_starts: [i64; 0] = [];
    let start_ref = RawStridedRef::new(&no_starts, &[0], &[1], 0).unwrap();
    let mut dest = RawStridedMut::new(&mut scalar, &[], &[], 0).unwrap();
    plan.execute(
        &mut dest,
        &operand_ref,
        &update_ref,
        &start_ref,
        Exec::serial(),
    )
    .unwrap();

    // A rank-0 update window covers the single destination element, so the
    // scalar ends up as the update value.
    assert_eq!(scalar, [5]);

    // An empty update window still copies the operand.
    let operand = [1i32, 2, 3];
    let plan =
        DynamicUpdateSlicePlan::compile(&[3], &[1], &[1], &[1], &[0], &[1], &[3], &[1]).unwrap();
    let operand_ref = RawStridedRef::new(&operand, &[3], &[1], 0).unwrap();
    let no_update: [i32; 0] = [];
    let update_ref = RawStridedRef::new(&no_update, &[0], &[1], 0).unwrap();
    let start_ref = RawStridedRef::new(&[9i64], &[1], &[1], 0).unwrap();
    let mut out = [-1i32; 3];
    let mut dest = RawStridedMut::new(&mut out, &[3], &[1], 0).unwrap();
    plan.execute(
        &mut dest,
        &operand_ref,
        &update_ref,
        &start_ref,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(out, operand);
}

/// Negative destination and update strides and both signed extremes of the
/// start range keep the documented copy-then-clamped-window semantics.
#[test]
fn issue_243_negative_strides_and_extreme_starts_match_reference() {
    let dest_dims = [4usize, 3];
    let dest_strides = [-3isize, 1];
    let operand_dims = [4usize, 3];
    let operand_strides = [1isize, 4];
    let update_dims = [2usize, 2];
    let update_strides = [-1isize, -2];
    let operand: Vec<i32> = (1..=12).collect();
    let update = [10i32, 20, 30, 40];
    let starts = [i64::MIN, i64::MAX];

    let plan = DynamicUpdateSlicePlan::compile(
        &operand_dims,
        &operand_strides,
        &[2],
        &[1],
        &update_dims,
        &update_strides,
        &dest_dims,
        &dest_strides,
    )
    .unwrap();

    let clamped: Vec<usize> = starts
        .iter()
        .zip(operand_dims.iter().zip(update_dims.iter()))
        .map(|(&start, (&dim, &window))| start.clamp(0, (dim - window) as i64) as usize)
        .collect();
    let mut expected = vec![-1i32; 12];
    for j in 0..dest_dims[1] {
        for i in 0..dest_dims[0] {
            let in_window = (clamped[0]..clamped[0] + update_dims[0]).contains(&i)
                && (clamped[1]..clamped[1] + update_dims[1]).contains(&j);
            let value = if in_window {
                let (a, b) = (i - clamped[0], j - clamped[1]);
                // The update view maps (a, b) to offset 3 - a - 2 * b.
                update[3 - a - 2 * b]
            } else {
                // Column-major operand: logical (i, j) at offset i + 4 * j.
                operand[i + 4 * j]
            };
            expected[(9 - 3 * i as isize + j as isize) as usize] = value;
        }
    }

    let operand_ref = RawStridedRef::new(&operand, &operand_dims, &operand_strides, 0).unwrap();
    // The update view's negative strides reach offsets 3, 2, 1 and 0.
    let update_ref = RawStridedRef::new(&update, &update_dims, &update_strides, 3).unwrap();
    let start_ref = RawStridedRef::new(&starts, &[2], &[1], 0).unwrap();
    let mut out = vec![-1i32; 12];
    let mut dest = RawStridedMut::new(&mut out, &dest_dims, &dest_strides, 9).unwrap();
    plan.execute(
        &mut dest,
        &operand_ref,
        &update_ref,
        &start_ref,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(out, expected);
}

/// The uninitialized lifecycle completes on success: the copy initializes every
/// reachable slot before the window overwrite, so reading the whole destination
/// back is defined.
#[test]
fn issue_243_uninit_lifecycle_initializes_every_slot() {
    use core::mem::MaybeUninit;
    use strided_basic::execution::dynamic_update_into_uninit;

    let dims = [4usize, 3];
    let operand: Vec<i32> = (1..=12).collect();
    let update = [10i32, 20];
    let starts = [2i64, -5];
    let plan = DynamicUpdateSlicePlan::compile(
        &dims,
        &[1, 4],
        &[2],
        &[1],
        &[2, 1],
        &[1, 2],
        &dims,
        &[3, 1],
    )
    .unwrap();

    let mut expected = vec![0i32; 12];
    for j in 0..3 {
        for i in 0..4 {
            expected[3 * i + j] = operand[i + 4 * j];
        }
    }
    // Start 2 clamps to 4 - 2 = 2 on axis 0, and the negative start clamps to
    // 0 on axis 1.
    expected[3 * 2] = update[0];
    expected[3 * 3] = update[1];

    let operand_ref = RawStridedRef::new(&operand, &dims, &[1, 4], 0).unwrap();
    let update_ref = RawStridedRef::new(&update, &[2, 1], &[1, 2], 0).unwrap();
    let start_ref = RawStridedRef::new(&starts, &[2], &[1], 0).unwrap();
    let mut storage = vec![MaybeUninit::<i32>::uninit(); 12];
    let mut dest = RawStridedMut::new(&mut storage, &dims, &[3, 1], 0).unwrap();
    // SAFETY: the destination does not overlap the inputs and a successful call
    // initializes every reachable slot.
    unsafe {
        dynamic_update_into_uninit(
            &plan,
            &mut dest,
            &operand_ref,
            &update_ref,
            &start_ref,
            Exec::serial(),
        )
    }
    .unwrap();
    // SAFETY: the successful call initialized every slot of the dense
    // destination.
    let read: Vec<i32> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
    assert_eq!(read, expected);
}
