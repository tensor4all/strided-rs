//! The plan-free `replay_*_raw` entries against the compiled plans, on serial
//! and team tokens.

use super::*;
use crate::{CopyPlan, StridedError, UpdatePlan};
use num_complex::Complex64;
use num_traits::{One, Zero};
use threadteam::{ledger, Team};

/// A four-member team with the cheap unit-test dispatch costs.
fn team() -> Team {
    // A loaded host (the whole suite runs in parallel) can keep workers from
    // waking within the default acquisition timeout.
    threadteam::Pool::builder()
        .threads(4)
        .dispatch_costs(crate::threading::test_dispatch_costs())
        .acquire_timeout(core::time::Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(4)
        .expect("an idle unit-test pool")
}

/// A team token for a region of `n` copied elements that must fan out: under
/// the `parallel` feature the crate's width rule has to grant it more than
/// one slot, otherwise the comparison would only exercise the serial branch.
fn fan_out_token(team: &Team, n: usize) -> (Exec<'_>, u64) {
    let exec = team.exec(4);
    assert!(exec.carries_team());
    let root = exec.ledger_root().expect("unit-test pools keep a ledger");
    crate::threading::take_dispatched_chunks();
    #[cfg(feature = "parallel")]
    assert!(
        crate::cost::parallel_width(&exec, KernelClass::Copy, n).get() > 1,
        "{n} elements must fan out on the team token"
    );
    #[cfg(not(feature = "parallel"))]
    let _ = n;
    (exec, root)
}

/// The tasks a fan-out replay keeps live stay within the team's four slots,
/// and, with the `parallel` feature, the replay was split into more than one
/// chunk. The chunk count, not `max_live`, proves the split: a chunk may finish
/// before the next one starts, so the live high-water mark can stay at one.
fn assert_fanned_out(root: u64) {
    assert!(ledger::max_live(root) <= 4);
    let chunks = crate::threading::take_dispatched_chunks();
    #[cfg(feature = "parallel")]
    assert!(chunks >= 2, "the replay did not fan out ({chunks} chunks)");
    #[cfg(not(feature = "parallel"))]
    let _ = chunks;
}

/// Smallest backing length covering every reachable offset from `offset`.
fn span(dims: &[usize], strides: &[isize], offset: isize) -> usize {
    if dims.contains(&0) {
        return 0;
    }
    let mut max = offset;
    for (&dim, &stride) in dims.iter().zip(strides) {
        if stride > 0 {
            max += (dim as isize - 1) * stride;
        }
    }
    max as usize + 1
}

/// Destination offset of every logical index, column-major over `dims`.
fn offsets(dims: &[usize], strides: &[isize], base: isize) -> Vec<usize> {
    let total: usize = dims.iter().product();
    (0..total)
        .map(|linear| {
            let mut rest = linear;
            let mut offset = base;
            for (&dim, &stride) in dims.iter().zip(strides) {
                offset += (rest % dim) as isize * stride;
                rest /= dim;
            }
            offset as usize
        })
        .collect()
}

/// Bit patterns, so that NaN payloads and signed zeros compare exactly.
trait Bits: Copy {
    fn bits(self) -> [u64; 2];
}

impl Bits for f64 {
    fn bits(self) -> [u64; 2] {
        [self.to_bits(), 0]
    }
}

impl Bits for Complex64 {
    fn bits(self) -> [u64; 2] {
        [self.re.to_bits(), self.im.to_bits()]
    }
}

fn bits<T: Bits>(values: &[T]) -> Vec<[u64; 2]> {
    values.iter().map(|&v| v.bits()).collect()
}

fn bits_uninit<T: Bits>(values: &[MaybeUninit<T>]) -> Vec<[u64; 2]> {
    // SAFETY: every test storage starts as `MaybeUninit::new(sentinel)`, and
    // the replays only ever write initialized values into it.
    values
        .iter()
        .map(|v| unsafe { v.assume_init() }.bits())
        .collect()
}

/// One layout pair with its offsets and backing lengths.
struct Fixture {
    dims: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
    dst_offset: isize,
    src_offset: isize,
    dst_len: usize,
    src_len: usize,
}

impl Fixture {
    fn new(
        dims: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        src_offset: isize,
    ) -> Self {
        let dst_offset = dims
            .iter()
            .zip(dst_strides)
            .filter(|(_, &s)| s < 0)
            .map(|(&d, &s)| (d as isize - 1) * -s)
            .sum();
        Self {
            dims: dims.to_vec(),
            dst_strides: dst_strides.to_vec(),
            src_strides: src_strides.to_vec(),
            dst_offset,
            src_offset,
            dst_len: span(dims, dst_strides, dst_offset).max(1),
            src_len: span(dims, src_strides, src_offset).max(1),
        }
    }

    fn total(&self) -> usize {
        self.dims.iter().product()
    }

    fn src<'a, T>(&'a self, data: &'a [T]) -> RawStridedRef<'a, T> {
        RawStridedRef::new(data, &self.dims, &self.src_strides, self.src_offset).unwrap()
    }

    fn dst<'a, T>(&'a self, data: &'a mut [T]) -> RawStridedMut<'a, T> {
        RawStridedMut::new(data, &self.dims, &self.dst_strides, self.dst_offset).unwrap()
    }
}

/// A stored `[16, 12, 300]` block cropped to the outer band `20..276`,
/// replayed into a permuted destination with holes.
fn cropped_band() -> Fixture {
    let stored = [16isize, 12 * 16];
    Fixture::new(
        &[16, 12, 256],
        &[3120, 260, 1],
        &[1, stored[0], stored[1]],
        20 * 192,
    )
}

/// A source with a reversed inner axis and a reversed outer axis.
fn negative_source() -> Fixture {
    let src_offset = 63 + 32 * 1536;
    Fixture::new(&[64, 24, 33], &[792, 33, 1], &[-1, 64, -1536], src_offset)
}

/// Scalars with which the row comparisons are generic over `f64` and
/// `Complex64`; `special` is a non-finite value placed into one slot.
trait Value:
    Bits
    + Send
    + Sync
    + Zero
    + One
    + PartialEq
    + ElementOpApply
    + Add<Output = Self>
    + Mul<Output = Self>
{
    fn value(k: usize) -> Self;
    fn special() -> Self;
    fn sentinel() -> Self;
    fn scalar(re: f64, im: f64) -> Self;
}

impl Value for f64 {
    fn value(k: usize) -> Self {
        (k % 97) as f64 * 0.25 - 7.0
    }
    fn special() -> Self {
        f64::INFINITY
    }
    fn sentinel() -> Self {
        f64::from_bits(0x7ff8_dead_beef_0001)
    }
    fn scalar(re: f64, _im: f64) -> Self {
        re
    }
}

impl Value for Complex64 {
    fn value(k: usize) -> Self {
        Complex64::new((k % 89) as f64 * 0.5 - 3.0, (k % 13) as f64 - 6.0)
    }
    fn special() -> Self {
        Complex64::new(f64::INFINITY, 0.0)
    }
    fn sentinel() -> Self {
        Complex64::new(f64::from_bits(0x7ff8_dead_beef_0002), -0.0)
    }
    fn scalar(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

/// Runs every row of the plan-equivalence table once on `exec_for(row)` and
/// asserts bit equality with the compiled plan on a serial token. The plan
/// and the matching entry evaluate the same expression per slot, so the
/// comparison holds for the non-finite slots too.
fn check_rows<'e, T: Value>(
    fixture: &Fixture,
    mut exec_for: impl FnMut() -> (Exec<'e>, Option<u64>),
) {
    let mut src: Vec<T> = (0..fixture.src_len).map(T::value).collect();
    src[fixture.src_len / 2] = T::special();
    let mut init: Vec<T> = (0..fixture.dst_len).map(|k| T::value(k + 5)).collect();
    init[fixture.dst_len / 3] = T::special();
    let scale = T::scalar(1.5, -0.5);
    let k = T::scalar(-2.0, 0.75);
    let g = move |x: T| x * k;
    let copy =
        CopyPlan::compile(&fixture.dims, &fixture.dst_strides, &fixture.src_strides).unwrap();
    let update =
        UpdatePlan::compile(&fixture.dims, &fixture.dst_strides, &fixture.src_strides).unwrap();
    let source = fixture.src(&src);

    let run = |plan_side: &dyn Fn(&mut RawStridedMut<'_, T>),
               entry_side: &dyn Fn(&mut RawStridedMut<'_, T>, Exec<'e>),
               exec: Exec<'e>,
               label: &str| {
        let mut expected = init.clone();
        plan_side(&mut fixture.dst(&mut expected));
        let mut actual = init.clone();
        entry_side(&mut fixture.dst(&mut actual), exec);
        assert_eq!(bits(&actual), bits(&expected), "{label}");
    };

    let (exec, root) = exec_for();
    run(
        &|d| copy.execute(d, &source, Exec::serial()).unwrap(),
        &|d, e| replay_copy_raw(d, &source, e).unwrap(),
        exec,
        "copy",
    );
    if let Some(root) = root {
        assert_fanned_out(root);
    }
    let (exec, _) = exec_for();
    run(
        &|d| {
            copy.execute_scale(d, &source, scale, Exec::serial())
                .unwrap()
        },
        &|d, e| replay_map_raw(d, &source, |d: &mut T, s: T| *d = scale * s, e).unwrap(),
        exec,
        "scale",
    );
    let (exec, _) = exec_for();
    run(
        &|d| copy.execute_conj(d, &source, Exec::serial()).unwrap(),
        &|d, e| replay_map_raw(d, &source, |d: &mut T, s: T| *d = s.conj(), e).unwrap(),
        exec,
        "conj",
    );
    for (alpha, beta) in [
        (T::one(), T::zero()),
        (T::scalar(2.5, 0.5), T::zero()),
        (T::scalar(2.5, 0.5), T::scalar(-0.5, 0.25)),
    ] {
        let (exec, _) = exec_for();
        run(
            &|d| {
                update
                    .execute(d, &source, alpha, beta, g, Exec::serial())
                    .unwrap()
            },
            &|d, e| {
                // The same branch selection as `UpdatePlan::execute`.
                if beta.is_zero() && alpha.is_one() {
                    replay_map_raw(d, &source, |d: &mut T, s: T| *d = g(s), e).unwrap()
                } else if beta.is_zero() {
                    replay_map_raw(d, &source, |d: &mut T, s: T| *d = alpha * g(s), e).unwrap()
                } else {
                    replay_map_raw(
                        d,
                        &source,
                        |d: &mut T, s: T| *d = alpha * g(s) + beta * *d,
                        e,
                    )
                    .unwrap()
                }
            },
            exec,
            "update",
        );
    }

    // Uninitialized destinations: equal to the plan, and the holes keep the
    // sentinel the storage started with.
    let reachable = offsets(&fixture.dims, &fixture.dst_strides, fixture.dst_offset);
    let mut hole = vec![true; fixture.dst_len];
    for &offset in &reachable {
        hole[offset] = false;
    }
    let run_uninit = |plan_side: &dyn Fn(&mut RawStridedMut<'_, MaybeUninit<T>>),
                      entry_side: &dyn Fn(&mut RawStridedMut<'_, MaybeUninit<T>>, Exec<'e>),
                      exec: Exec<'e>,
                      label: &str| {
        let fresh = || vec![MaybeUninit::new(T::sentinel()); fixture.dst_len];
        let mut expected = fresh();
        plan_side(&mut fixture.dst(&mut expected));
        let mut actual = fresh();
        entry_side(&mut fixture.dst(&mut actual), exec);
        let actual = bits_uninit(&actual);
        assert_eq!(actual, bits_uninit(&expected), "{label}");
        for (slot, value) in actual.iter().enumerate() {
            if hole[slot] {
                assert_eq!(*value, T::sentinel().bits(), "{label}: hole {slot} written");
            }
        }
    };
    let (exec, _) = exec_for();
    run_uninit(
        &|d| copy.execute_uninit(d, &source, Exec::serial()).unwrap(),
        &|d, e| replay_copy_raw_uninit(d, &source, e).unwrap(),
        exec,
        "copy uninit",
    );
    for alpha in [T::one(), T::scalar(2.5, 0.5)] {
        let (exec, _) = exec_for();
        run_uninit(
            &|d| {
                update
                    .execute_uninit(d, &source, alpha, g, Exec::serial())
                    .unwrap()
            },
            &|d, e| {
                // The same branch selection as `UpdatePlan::execute_uninit`.
                if alpha.is_one() {
                    replay_map_raw_uninit(d, &source, g, e).unwrap()
                } else {
                    replay_map_raw_uninit(d, &source, |s: T| alpha * g(s), e).unwrap()
                }
            },
            exec,
            "update uninit",
        );
    }
}

/// Runs the rows serially and on a fan-out team token.
fn check_rows_serial_and_team<T: Value>(fixture: &Fixture) {
    check_rows::<T>(fixture, || (Exec::serial(), None));
    let team = team();
    let n = fixture.total();
    check_rows::<T>(fixture, || {
        let (exec, root) = fan_out_token(&team, n);
        (exec, Some(root))
    });
}

#[test]
fn cropped_outer_band_matches_the_compiled_plans() {
    check_rows_serial_and_team::<f64>(&cropped_band());
    check_rows_serial_and_team::<Complex64>(&cropped_band());
}

#[test]
fn negative_stride_source_matches_the_compiled_plans() {
    check_rows_serial_and_team::<f64>(&negative_source());
    check_rows_serial_and_team::<Complex64>(&negative_source());
}

#[test]
fn rank_nine_spilled_layout_matches_the_compiled_plans() {
    let dims = [3usize; 9];
    let dst: Vec<isize> = (0..9).map(|axis| 3isize.pow(axis)).collect();
    let src: Vec<isize> = dst.iter().rev().copied().collect();
    let fixture = Fixture::new(&dims, &dst, &src, 0);
    assert!(matches!(
        PairTraversal::given(&dims, &dst, &src),
        PairTraversal::Spilled(_)
    ));
    assert!(
        CopyPlan::compile(&dims, &dst, &src)
            .unwrap()
            .fused_layout()
            .is_none(),
        "the plan must spill too"
    );
    check_rows_serial_and_team::<f64>(&fixture);
}

#[test]
fn zero_beta_never_reads_a_nan_destination_and_unit_alpha_copies_exactly() {
    // `UpdatePlan(alpha = 2, beta = 0)` over a NaN destination writes `2 * s`;
    // the matching entry row does the same.
    let dims = [3usize];
    let strides = [1isize];
    let src = [3.0f64, -1.0, 0.5];
    let plan = UpdatePlan::compile(&dims, &strides, &strides).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    let mut expected = [f64::NAN; 3];
    plan.execute(
        &mut RawStridedMut::new(&mut expected, &dims, &strides, 0).unwrap(),
        &source,
        2.0,
        0.0,
        |x| x,
        Exec::serial(),
    )
    .unwrap();
    let mut actual = [f64::NAN; 3];
    replay_map_raw(
        &mut RawStridedMut::new(&mut actual, &dims, &strides, 0).unwrap(),
        &source,
        |d: &mut f64, s: f64| *d = 2.0 * s,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(actual, [6.0, -2.0, 1.0]);
    assert_eq!(bits(&actual), bits(&expected));

    // `UpdatePlan::execute_uninit(alpha = 1)` copies `(inf, 0)` unchanged,
    // as does the matching row; `1 * (inf + 0i)` would give an imaginary NaN.
    let src = [Complex64::new(f64::INFINITY, 0.0)];
    let plan = UpdatePlan::compile(&[1], &[1], &[1]).unwrap();
    let source = RawStridedRef::new(&src, &[1], &[1], 0).unwrap();
    let mut expected = [MaybeUninit::new(Complex64::sentinel())];
    plan.execute_uninit(
        &mut RawStridedMut::new(&mut expected, &[1], &[1], 0).unwrap(),
        &source,
        Complex64::one(),
        |x| x,
        Exec::serial(),
    )
    .unwrap();
    let mut actual = [MaybeUninit::new(Complex64::sentinel())];
    replay_map_raw_uninit(
        &mut RawStridedMut::new(&mut actual, &[1], &[1], 0).unwrap(),
        &source,
        |s| s,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(bits_uninit(&actual), [[f64::INFINITY.to_bits(), 0]]);
    assert_eq!(bits_uninit(&actual), bits_uninit(&expected));
}

#[test]
fn exact_accumulate_keeps_complex_infinities() {
    let src = [
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(1.0, -1.0),
    ];
    let mut acc = [Complex64::new(2.0, 3.0), Complex64::new(0.5, 0.5)];
    replay_map_raw(
        &mut RawStridedMut::new(&mut acc, &[2], &[1], 0).unwrap(),
        &RawStridedRef::new(&src, &[2], &[1], 0).unwrap(),
        |d: &mut Complex64, s| *d += s,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(
        bits(&acc),
        [
            [f64::INFINITY.to_bits(), 3.0f64.to_bits()],
            [1.5f64.to_bits(), (-0.5f64).to_bits()]
        ]
    );
}

/// A non-float `Copy` element with padding.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rgb(u8, u16, u8);

fn check_bulk<T: Copy + Send + Sync + PartialEq + core::fmt::Debug>(make: impl Fn(usize) -> T) {
    let team = team();
    for (dims, strides) in [(vec![1000usize], vec![1isize]), (vec![37, 5], vec![1, 37])] {
        let total: usize = dims.iter().product();
        let src: Vec<T> = (0..total).map(&make).collect();
        let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();
        let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        let mut expected = vec![make(usize::MAX); total];
        plan.execute(
            &mut RawStridedMut::new(&mut expected, &dims, &strides, 0).unwrap(),
            &source,
            Exec::serial(),
        )
        .unwrap();
        assert_eq!(expected, src);
        for exec in [Exec::serial(), team.exec(4)] {
            let mut actual = vec![make(usize::MAX); total];
            replay_copy_raw(
                &mut RawStridedMut::new(&mut actual, &dims, &strides, 0).unwrap(),
                &source,
                exec,
            )
            .unwrap();
            assert_eq!(actual, expected);
        }
        for exec in [Exec::serial(), team.exec(4)] {
            let mut storage = vec![MaybeUninit::<T>::uninit(); total];
            replay_copy_raw_uninit(
                &mut RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap(),
                &source,
                exec,
            )
            .unwrap();
            // SAFETY: the layout is dense, so the successful replay
            // initialized every slot.
            let actual: Vec<T> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn bulk_copy_runs_move_every_element() {
    check_bulk(|k| Complex64::new(k as f64, -(k as f64)));
    check_bulk(|k| Rgb(k as u8, k as u16, (k >> 8) as u8));
}

/// Every entry on `dims`, serial and on a team token, expecting `Ok` and no
/// write; the boundary cases cannot fan out, so no width is asserted.
fn check_empty(dims: &[usize], strides: &[isize]) {
    let team = team();
    let src: [f64; 0] = [];
    let source = RawStridedRef::new(&src, dims, strides, 0).unwrap();
    for exec_kind in 0..2 {
        let exec = || {
            if exec_kind == 0 {
                Exec::serial()
            } else {
                team.exec(4)
            }
        };
        let mut dst: [f64; 0] = [];
        let mut dest = RawStridedMut::new(&mut dst, dims, strides, 0).unwrap();
        replay_copy_raw(&mut dest, &source, exec()).unwrap();
        replay_map_raw(&mut dest, &source, |_: &mut f64, _| panic!("empty"), exec()).unwrap();
        let mut uninit: [MaybeUninit<f64>; 0] = [];
        let mut dest = RawStridedMut::new(&mut uninit, dims, strides, 0).unwrap();
        replay_copy_raw_uninit(&mut dest, &source, exec()).unwrap();
        replay_map_raw_uninit(
            &mut dest,
            &source,
            |_: f64| -> f64 { panic!("empty") },
            exec(),
        )
        .unwrap();
    }
    let plan = CopyPlan::compile(dims, strides, strides).unwrap();
    let mut dst: [f64; 0] = [];
    plan.execute(
        &mut RawStridedMut::new(&mut dst, dims, strides, 0).unwrap(),
        &source,
        Exec::serial(),
    )
    .unwrap();
}

#[test]
fn empty_layouts_with_an_overflowing_prefix_replay_nothing() {
    let mut dims = vec![usize::MAX, 2, 0];
    check_empty(&dims, &[0, 0, 0]);
    dims.extend([1; 6]);
    assert_eq!(dims.len(), 9);
    check_empty(&dims, &[0; 9]);
    check_empty(&[3, 0], &[1, 3]);
}

#[test]
fn rank_zero_replays_one_element() {
    let team = team();
    let src = [0.0f64, 4.5];
    for exec in [Exec::serial(), team.exec(4)] {
        let mut dst = [-1.0f64; 3];
        replay_copy_raw(
            &mut RawStridedMut::new(&mut dst, &[], &[], 2).unwrap(),
            &RawStridedRef::new(&src, &[], &[], 1).unwrap(),
            exec,
        )
        .unwrap();
        assert_eq!(dst, [-1.0, -1.0, 4.5]);
    }
}

#[test]
fn errors_precede_any_write() {
    let src = [1.0f64; 8];
    let mut dst = [-1.0f64; 8];
    // Bounds and rank agreement are checked by the views themselves.
    assert!(matches!(
        RawStridedRef::new(&src, &[4, 3], &[1, 4], 0),
        Err(StridedError::OffsetOverflow)
    ));
    assert!(matches!(
        RawStridedRef::new(&src, &[2, 2], &[1, -4], 0),
        Err(StridedError::OffsetOverflow)
    ));
    assert!(matches!(
        RawStridedRef::new(&src, &[2, 2], &[1], 0),
        Err(StridedError::StrideLengthMismatch)
    ));
    // Mismatched dims.
    let source = RawStridedRef::new(&src, &[2, 4], &[1, 2], 0).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &[4, 2], &[1, 4], 0).unwrap();
    assert!(matches!(
        replay_copy_raw(&mut dest, &source, Exec::serial()),
        Err(StridedError::ShapeMismatch(..))
    ));
    assert!(matches!(
        replay_map_raw(&mut dest, &source, |d: &mut f64, s| *d = s, Exec::serial()),
        Err(StridedError::ShapeMismatch(..))
    ));
    assert_eq!(dst, [-1.0; 8]);
    // A nonzero count that overflows `usize` (broadcast axes).
    let big = [1usize << 62, 8];
    let zero = [0isize, 0];
    let source = RawStridedRef::new(&src, &big, &zero, 0).unwrap();
    let mut uninit = [MaybeUninit::new(-1.0f64)];
    let mut dest = RawStridedMut::new(&mut uninit, &big, &zero, 0).unwrap();
    assert!(matches!(
        replay_copy_raw_uninit(&mut dest, &source, Exec::serial()),
        Err(StridedError::OffsetOverflow)
    ));
    assert!(matches!(
        replay_map_raw_uninit(&mut dest, &source, |s| s, Exec::serial()),
        Err(StridedError::OffsetOverflow)
    ));
    // SAFETY: initialized at construction and never written.
    assert_eq!(unsafe { uninit[0].assume_init() }, -1.0);
}

#[test]
fn non_injective_destination_replays_serially_on_a_team_token() {
    // Every column of a `[4096, 3]` source accumulates into one slot.
    let dims = [4096usize, 3];
    let dst_strides = [0isize, 1];
    let src_strides = [1isize, 4096];
    let src: Vec<i64> = (0..4096 * 3).map(|k| k * 7 - 1000).collect();
    let mut expected = [5i64, 6, 7];
    for column in 0..3 {
        for row in 0..4096 {
            expected[column] += src[row + 4096 * column];
        }
    }
    let team = team();
    let (exec, root) = fan_out_token(&team, 4096 * 3);
    let mut acc = [5i64, 6, 7];
    replay_map_raw(
        &mut RawStridedMut::new(&mut acc, &dims, &dst_strides, 0).unwrap(),
        &RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap(),
        |d: &mut i64, s| *d += s,
        exec,
    )
    .unwrap();
    assert_eq!(acc, expected);
    assert_eq!(
        ledger::max_live(root),
        0,
        "a non-injective replay must not fan out"
    );
}

#[test]
fn non_injective_destination_keeps_the_given_axis_order() {
    // Destination slot 2 is reached by (1, 0) with source 1 and by (0, 2) with
    // source 4. The given axes visit (1, 0) first, so the overwrite leaves 4;
    // sorting the axes by destination stride would visit (0, 2) first.
    let dims = [2usize, 3];
    let dst_strides = [2isize, 1];
    let src_strides = [1isize, 2];
    let src = [0i64, 1, 2, 3, 4, 5];
    let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
    let team = team();
    for exec_kind in 0..2 {
        let exec = || {
            if exec_kind == 0 {
                Exec::serial()
            } else {
                team.exec(4)
            }
        };
        let mut copied = [-1i64; 5];
        replay_copy_raw(
            &mut RawStridedMut::new(&mut copied, &dims, &dst_strides, 0).unwrap(),
            &source,
            exec(),
        )
        .unwrap();
        assert_eq!(copied, [0, 2, 4, 3, 5]);
        let mut mapped = [-1i64; 5];
        replay_map_raw(
            &mut RawStridedMut::new(&mut mapped, &dims, &dst_strides, 0).unwrap(),
            &source,
            |d: &mut i64, s| *d = s,
            exec(),
        )
        .unwrap();
        assert_eq!(mapped, [0, 2, 4, 3, 5]);
    }
}
