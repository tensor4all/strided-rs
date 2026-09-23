use rand::{rngs::StdRng, Rng, SeedableRng};
use strided_basic::{DynamicUpdateSlicePlan, RawStridedMut, RawStridedRef};

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
        if let Err(e) = plan.execute(&mut dest, &op, &up, &st) {
            panic!("post-check failure {e:?} dims={dims:?} ds={ds:?} os={os:?} ud={ud:?} us={us:?} offs={doff},{ooff},{uoff} starts={starts:?}");
        }
    }
    eprintln!("executed {tried} valid cases, {extreme} with extreme dest strides");
    assert!(tried > 1000);
}
