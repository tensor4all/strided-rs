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
fn assert_fanned_out(root: Option<u64>) {
    let Some(root) = root else { return };
    assert!(ledger::max_live(root) <= 4);
    let chunks = crate::threading::take_dispatched_chunks();
    #[cfg(feature = "parallel")]
    assert!(chunks >= 2, "the replay did not fan out ({chunks} chunks)");
    #[cfg(not(feature = "parallel"))]
    let _ = chunks;
}

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut stride = 1isize;
    dims.iter()
        .map(|&dim| {
            let out = stride;
            stride *= dim as isize;
            out
        })
        .collect()
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
    assert_fanned_out(root);
    let (exec, root) = exec_for();
    run(
        &|d| {
            copy.execute_scale(d, &source, scale, Exec::serial())
                .unwrap()
        },
        &|d, e| replay_map_raw(d, &source, |d: &mut T, s: T| *d = scale * s, e).unwrap(),
        exec,
        "scale",
    );
    assert_fanned_out(root);
    let (exec, root) = exec_for();
    run(
        &|d| copy.execute_conj(d, &source, Exec::serial()).unwrap(),
        &|d, e| replay_map_raw(d, &source, |d: &mut T, s: T| *d = s.conj(), e).unwrap(),
        exec,
        "conj",
    );
    assert_fanned_out(root);
    for (alpha, beta) in [
        (T::one(), T::zero()),
        (T::scalar(2.5, 0.5), T::zero()),
        (T::scalar(2.5, 0.5), T::scalar(-0.5, 0.25)),
    ] {
        let (exec, root) = exec_for();
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
        assert_fanned_out(root);
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
    let (exec, root) = exec_for();
    run_uninit(
        &|d| copy.execute_uninit(d, &source, Exec::serial()).unwrap(),
        &|d, e| replay_copy_raw_uninit(d, &source, e).unwrap(),
        exec,
        "copy uninit",
    );
    assert_fanned_out(root);
    for alpha in [T::one(), T::scalar(2.5, 0.5)] {
        let (exec, root) = exec_for();
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
        assert_fanned_out(root);
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

/// The fused entry on the same fixtures: the destination is injective, so the
/// fused visit order cannot change the result, and every element must match the
/// given-axis replay bit for bit.
fn assert_bits_eq(label: &str, got: &[[u64; 2]], want: &[[u64; 2]]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    if let Some((index, (g, w))) = got.iter().zip(want).enumerate().find(|(_, (g, w))| g != w) {
        panic!("{label}: first mismatch at {index}: {g:?} != {w:?}");
    }
}

fn check_fused_rows<T: Value>(fixture: &Fixture) {
    let src: Vec<T> = (0..fixture.src_len).map(T::value).collect();
    let source = fixture.src(&src);
    let mut expected = vec![T::sentinel(); fixture.dst_len];
    replay_copy_raw(&mut fixture.dst(&mut expected), &source, Exec::serial()).unwrap();

    let mut got = vec![T::sentinel(); fixture.dst_len];
    replay_map_raw_fused(
        &mut fixture.dst(&mut got),
        &source,
        |d: &mut T, s| *d = s,
        Exec::serial(),
    )
    .unwrap();
    assert_bits_eq("copy", &bits(&got), &bits(&expected));

    let mut uninit = vec![MaybeUninit::new(T::sentinel()); fixture.dst_len];
    replay_map_raw_uninit_fused(
        &mut fixture.dst(&mut uninit),
        &source,
        |s: T| s,
        Exec::serial(),
    )
    .unwrap();
    assert_bits_eq("uninit", &bits_uninit(&uninit), &bits(&expected));

    // Accumulating callbacks keep their arithmetic under the fused route. Only
    // the reachable destination slots are written; holes keep the filler.
    let mut acc = vec![T::one(); fixture.dst_len];
    replay_map_raw_fused(
        &mut fixture.dst(&mut acc),
        &fixture.src(&vec![T::one(); fixture.src_len]),
        |d: &mut T, s| *d = *d + s,
        Exec::serial(),
    )
    .unwrap();
    let want = (T::one() + T::one()).bits();
    for &slot in &offsets(&fixture.dims, &fixture.dst_strides, fixture.dst_offset) {
        assert_eq!(
            acc[slot].bits(),
            want,
            "accumulate at destination slot {slot}"
        );
    }
}

#[test]
fn fused_replay_matches_the_given_axis_replay() {
    let fixtures = [cropped_band(), negative_source()];
    for fixture in &fixtures {
        check_fused_rows::<f64>(fixture);
        check_fused_rows::<Complex64>(fixture);
    }
    let dims = [3usize; 9];
    let dst: Vec<isize> = (0..9).map(|axis| 3isize.pow(axis)).collect();
    let src: Vec<isize> = dst.iter().rev().copied().collect();
    let fixture = Fixture::new(&dims, &dst, &src, 0);
    assert!(matches!(
        PairTraversal::given(&dims, &dst, &src),
        PairTraversal::Spilled(_)
    ));
    check_fused_rows::<f64>(&fixture);
}

/// Block sides 2 and 8 at ranks 2-4: the shape the downstream gate measures.
/// The source is the reversed destination, so fusion has something to merge and
/// the fused route must still write every slot.
#[test]
fn fused_replay_matches_on_small_blocks() {
    for side in [2usize, 8] {
        for rank in 2..=4usize {
            let dims = vec![side; rank];
            let dst = col_major_strides(&dims);
            let src: Vec<isize> = dst.iter().rev().copied().collect();
            let fixture = Fixture::new(&dims, &dst, &src, 0);
            check_fused_rows::<f64>(&fixture);
        }
    }
}

/// A rank above the inline limit with extent-1 axes: the compaction and the
/// fusion scratch both stay on the stack and the result is unchanged.
#[test]
fn fused_replay_handles_a_rank_above_the_inline_limit() {
    let moving = [2usize, 3, 4];
    let (mut dims, mut dst, mut src) = (Vec::new(), Vec::new(), Vec::new());
    for axis in 0..40usize {
        if axis % 13 == 0 && axis / 13 < moving.len() {
            let k = axis / 13;
            dims.push(moving[k]);
            dst.push([1isize, 2, 6][k]);
            src.push([12isize, 4, 1][k]);
        } else {
            dims.push(1);
            dst.push(axis as isize);
            src.push(-(axis as isize));
        }
    }
    let fixture = Fixture::new(&dims, &dst, &src, 0);
    check_fused_rows::<f64>(&fixture);
}

#[test]
fn fused_replay_rejects_a_shape_mismatch_before_writing() {
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let mut out = [7.0f64; 4];
    let source = RawStridedRef::new(&src, &[2, 2], &[1, 2], 0).unwrap();
    let mut dest = RawStridedMut::new(&mut out, &[4], &[1], 0).unwrap();
    assert!(matches!(
        replay_map_raw_fused(&mut dest, &source, |d: &mut f64, s| *d = s, Exec::serial()),
        Err(StridedError::ShapeMismatch(_, _))
    ));
    assert_eq!(out, [7.0; 4]);
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

/// The moving-axis bound is what the wide route's stack arrays are sized by:
/// `usize::BITS - 1` extent-2 axes are the most a checked count admits, and
/// they fill exactly that many slots.
#[test]
fn the_wide_route_holds_the_largest_checked_moving_rank() {
    let moving = usize::BITS as usize - 1;
    let dims = vec![2usize; moving];
    let strides = vec![0isize; moving];
    crate::kernel::total_len(&dims).expect("2^(BITS - 1) elements fit usize");
    let PairTraversal::Spilled(axes) = PairTraversal::given(&dims, &strides, &strides) else {
        panic!("{moving} moving axes must spill");
    };
    assert_eq!(axes.len(), moving);
    assert!(axes.len() < MAX_MOVING_AXES);
}

/// One more moving axis than the bound overflows the count: the entries reject
/// it before any route is built. The constructors accept the broadcast, whose
/// only reachable offset is 0.
#[test]
fn a_count_beyond_the_moving_axis_bound_is_rejected_before_the_route() {
    let dims = vec![2usize; usize::BITS as usize];
    let strides = vec![0isize; dims.len()];
    let src = [1.0f64];
    let mut dst = [0.0f64];
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    assert!(matches!(
        replay_copy_raw(&mut dest, &source, Exec::serial()),
        Err(StridedError::OffsetOverflow)
    ));
    assert!(matches!(
        replay_map_raw(&mut dest, &source, |d: &mut f64, s| *d += s, Exec::serial()),
        Err(StridedError::OffsetOverflow)
    ));
    assert_eq!(dst, [0.0]);
}

/// The wide untiled replay of the allocation test (`alloc_matrix.rs`,
/// `check_pooled_wide_replay`) dispatches more than one range on a team token,
/// so every worker range runs its own cursor.
#[test]
fn wide_untiled_replay_fans_out_on_a_team_token() {
    let team = team();
    let dims = vec![2usize; 17];
    let strides: Vec<isize> = (0..17).map(|k| 1isize << k).collect();
    let n: usize = dims.iter().product();
    let PairTraversal::Spilled(_) = PairTraversal::given(&dims, &strides, &strides) else {
        panic!("seventeen moving axes must spill");
    };
    assert!(PairTraversal::given(&dims, &strides, &strides)
        .tiles::<f64, f64>()
        .is_none());
    let src: Vec<f64> = (0..n).map(|v| v as f64).collect();
    let mut out = vec![0.0f64; n];
    let (exec, root) = fan_out_token(&team, n);
    replay_copy_raw(
        &mut RawStridedMut::new(&mut out, &dims, &strides, 0).unwrap(),
        &RawStridedRef::new(&src, &dims, &strides, 0).unwrap(),
        exec,
    )
    .unwrap();
    assert_fanned_out(Some(root));
    assert_eq!(out, src);
}

/// Destination slots in the order the tiled walk is specified to visit them,
/// from the specification alone: the tile grid column-major, tiles of extents
/// `blocks` anchored at the origin and clipped to the box, each tile
/// column-major.
fn tile_order(dims: &[usize], dst_strides: &[isize], blocks: &[usize], base: isize) -> Vec<usize> {
    let rank = dims.len();
    let grid: Vec<usize> = dims
        .iter()
        .zip(blocks)
        .map(|(&d, &b)| d.div_ceil(b))
        .collect();
    let mut order = Vec::new();
    for tile in 0..grid.iter().product::<usize>() {
        let mut rest = tile;
        let origin: Vec<usize> = (0..rank)
            .map(|axis| {
                let coord = rest % grid[axis];
                rest /= grid[axis];
                coord * blocks[axis]
            })
            .collect();
        let extent: Vec<usize> = (0..rank)
            .map(|axis| blocks[axis].min(dims[axis] - origin[axis]))
            .collect();
        for element in 0..extent.iter().product::<usize>() {
            let mut rest = element;
            let mut slot = base;
            for axis in 0..rank {
                slot += (origin[axis] + rest % extent[axis]) as isize * dst_strides[axis];
                rest /= extent[axis];
            }
            order.push(slot as usize);
        }
    }
    order
}

/// `replay_map_raw_serial` walks what `replay_map_raw` walks on a serial
/// token: the same `(destination slot, source value)` sequence and the same
/// final buffer, on every route. The sequence is also checked against an
/// order built from the specification alone: column-major over the given axes
/// for an untiled route, [`tile_order`] for a tiled one.
#[test]
fn serial_entry_visits_what_the_serial_token_visits() {
    use std::sync::Mutex;

    struct Case {
        label: &'static str,
        dims: Vec<usize>,
        dst: Vec<isize>,
        src: Vec<isize>,
        src_offset: isize,
        /// Whether the route must come out tiled (and be kept: an injective
        /// destination).
        tiled: bool,
    }
    let reversed = |dims: &[usize]| {
        let mut strides = col_major_strides(&dims.iter().rev().copied().collect::<Vec<_>>());
        strides.reverse();
        strides
    };
    let mut cases = vec![
        Case {
            label: "rank 0",
            dims: vec![],
            dst: vec![],
            src: vec![],
            src_offset: 0,
            tiled: false,
        },
        Case {
            label: "rank 1 reversed",
            dims: vec![5],
            dst: vec![1],
            src: vec![-1],
            src_offset: 4,
            tiled: false,
        },
        Case {
            label: "inline untiled",
            dims: vec![3, 4, 2],
            dst: col_major_strides(&[3, 4, 2]),
            src: vec![8, 2, 1],
            src_offset: 0,
            tiled: false,
        },
        Case {
            label: "inline tiled",
            dims: vec![16, 16, 16],
            dst: col_major_strides(&[16, 16, 16]),
            src: reversed(&[16, 16, 16]),
            src_offset: 0,
            tiled: true,
        },
        Case {
            label: "non-injective accumulate",
            dims: vec![64, 3],
            dst: vec![0, 1],
            src: vec![1, 64],
            src_offset: 0,
            tiled: false,
        },
        Case {
            label: "zero extent",
            dims: vec![3, 0, 2],
            dst: vec![1, 3, 3],
            src: vec![1, 3, 3],
            src_offset: 0,
            tiled: false,
        },
        Case {
            label: "rank 9 issue layout (spilled untiled)",
            dims: vec![2; 9],
            dst: col_major_strides(&[2; 9]),
            src: reversed(&[2; 9]),
            src_offset: 0,
            tiled: false,
        },
        Case {
            label: "4^9 reversed (spilled tiled)",
            dims: vec![4; 9],
            dst: col_major_strides(&[4; 9]),
            src: reversed(&[4; 9]),
            src_offset: 0,
            tiled: true,
        },
    ];
    // Rank 12 with eight moving axes: the wide route compacts to inline.
    let mut wide = Case {
        label: "rank 12, eight moving (inline after compaction)",
        dims: vec![],
        dst: vec![],
        src: vec![],
        src_offset: 0,
        tiled: false,
    };
    let moving = [3usize, 2, 2, 3, 2, 2, 2, 3];
    let moving_src = reversed(&moving);
    let moving_dst = col_major_strides(&moving);
    for axis in 0..12 {
        if axis % 3 == 2 {
            wide.dims.push(1);
            wide.dst.push(1000);
            wide.src.push(-1000);
        } else {
            let k = axis - axis / 3;
            wide.dims.push(moving[k]);
            wide.dst.push(moving_dst[k]);
            wide.src.push(moving_src[k]);
        }
    }
    cases.push(wide);

    for case in &cases {
        let label = case.label;
        let route_tiles = if case.dims.contains(&0) {
            None
        } else {
            PairTraversal::given(&case.dims, &case.dst, &case.src).tiles::<f64, f64>()
        };
        assert_eq!(route_tiles.is_some(), case.tiled, "{label}: route");
        let src_len = span(&case.dims, &case.src, case.src_offset);
        let src: Vec<f64> = (0..src_len).map(|v| v as f64 * 0.5 - 3.0).collect();
        let dst_len = span(&case.dims, &case.dst, 0).max(1);
        let initial: Vec<f64> = (0..dst_len).map(|v| v as f64 * 100.0).collect();
        let source = RawStridedRef::new(&src, &case.dims, &case.src, case.src_offset).unwrap();

        let mut by_token = initial.clone();
        let base = by_token.as_ptr() as usize;
        let token_visits = Mutex::new(Vec::new());
        replay_map_raw(
            &mut RawStridedMut::new(&mut by_token, &case.dims, &case.dst, 0).unwrap(),
            &source,
            |d: &mut f64, s: f64| {
                let slot = (d as *mut f64 as usize - base) / size_of::<f64>();
                token_visits.lock().unwrap().push((slot, s.to_bits()));
                *d += s;
            },
            Exec::serial(),
        )
        .unwrap();
        let token_visits = token_visits.into_inner().unwrap();

        let mut by_entry = initial.clone();
        let base = by_entry.as_ptr() as usize;
        let mut entry_visits = Vec::new();
        replay_map_raw_serial(
            &mut RawStridedMut::new(&mut by_entry, &case.dims, &case.dst, 0).unwrap(),
            &source,
            |d: &mut f64, s: f64| {
                let slot = (d as *mut f64 as usize - base) / size_of::<f64>();
                entry_visits.push((slot, s.to_bits()));
                *d += s;
            },
        )
        .unwrap();

        assert!(entry_visits == token_visits, "{label}: visit sequence");
        assert_eq!(bits(&by_entry), bits(&by_token), "{label}: result");
        let slots: Vec<usize> = entry_visits.iter().map(|&(slot, _)| slot).collect();
        let specified = match &route_tiles {
            None => offsets(&case.dims, &case.dst, 0),
            Some(Tiles::Inline(tiles)) => {
                let rank = tiles.layout.rank;
                tile_order(
                    &tiles.layout.dims[..rank],
                    &tiles.layout.dst_strides[..rank],
                    &tiles.blocks[..rank],
                    0,
                )
            }
            Some(Tiles::Spilled(tiles)) => tile_order(tiles.dims(), tiles.dst(), tiles.blocks(), 0),
        };
        assert!(slots == specified, "{label}: specified visit order");
    }
}

/// The serial entry fails as the serial token does, before `f` first runs and
/// before any write.
#[test]
fn serial_entry_errors_match_the_serial_token() {
    let src = [1.0f64; 8];
    let mut dst = [-1.0f64; 8];
    let source = RawStridedRef::new(&src, &[2, 4], &[1, 2], 0).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &[4, 2], &[1, 4], 0).unwrap();
    let mut calls = 0;
    assert!(matches!(
        replay_map_raw(&mut dest, &source, |d: &mut f64, s| *d = s, Exec::serial()),
        Err(StridedError::ShapeMismatch(..))
    ));
    assert!(matches!(
        replay_map_raw_serial(&mut dest, &source, |d: &mut f64, s| {
            calls += 1;
            *d = s
        }),
        Err(StridedError::ShapeMismatch(..))
    ));
    assert_eq!(dst, [-1.0; 8]);

    let dims = vec![2usize; usize::BITS as usize];
    let strides = vec![0isize; dims.len()];
    let one = [1.0f64];
    let mut out = [0.0f64];
    let source = RawStridedRef::new(&one, &dims, &strides, 0).unwrap();
    let mut dest = RawStridedMut::new(&mut out, &dims, &strides, 0).unwrap();
    assert!(matches!(
        replay_map_raw(&mut dest, &source, |d: &mut f64, s| *d += s, Exec::serial()),
        Err(StridedError::OffsetOverflow)
    ));
    assert!(matches!(
        replay_map_raw_serial(&mut dest, &source, |d: &mut f64, s| {
            calls += 1;
            *d += s
        }),
        Err(StridedError::OffsetOverflow)
    ));
    assert_eq!(out, [0.0]);
    assert_eq!(calls, 0);
}
