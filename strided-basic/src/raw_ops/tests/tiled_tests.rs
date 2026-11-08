//! The cache-tiled plan and replay walk: every entry equals an independent
//! per-element decode bit for bit, on serial and team tokens; layouts whose
//! tiles would not change the visit order keep the column-major walk; split
//! leaves partition the box; non-injective destinations stay serial.

use super::*;
use crate::{CopyPlan, UpdatePlan};
use num_complex::Complex64;
use threadteam::{ledger, Team};

fn team() -> Team {
    threadteam::Pool::builder()
        .threads(4)
        .dispatch_costs(crate::threading::test_dispatch_costs())
        .acquire_timeout(core::time::Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(4)
        .expect("an idle unit-test pool")
}

/// One layout with its base offsets and backing lengths.
struct Layout {
    dims: Vec<usize>,
    dst: Vec<isize>,
    src: Vec<isize>,
    dst_offset: isize,
    src_offset: isize,
}

impl Layout {
    fn new(dims: &[usize], dst: &[isize], src: &[isize], src_offset: isize) -> Self {
        Self {
            dims: dims.to_vec(),
            dst: dst.to_vec(),
            src: src.to_vec(),
            dst_offset: lowest(dims, dst),
            src_offset,
        }
    }

    fn total(&self) -> usize {
        self.dims.iter().product()
    }

    /// Destination and source offset of every logical index, column-major
    /// over the given axes. Extent-1 axes contribute nothing whatever their
    /// stride.
    fn offsets(&self) -> Vec<(usize, usize)> {
        (0..self.total())
            .map(|linear| {
                let mut rest = linear;
                let (mut d, mut s) = (self.dst_offset, self.src_offset);
                for axis in 0..self.dims.len() {
                    let coord = rest % self.dims[axis];
                    rest /= self.dims[axis];
                    if coord > 0 {
                        d += coord as isize * self.dst[axis];
                        s += coord as isize * self.src[axis];
                    }
                }
                (d as usize, s as usize)
            })
            .collect()
    }

    fn lens(&self) -> (usize, usize) {
        let offsets = self.offsets();
        let d = offsets.iter().map(|o| o.0).max().unwrap() + 1;
        let s = offsets.iter().map(|o| o.1).max().unwrap() + 1;
        (d, s)
    }

    fn src_view<'a, T>(&'a self, data: &'a [T]) -> RawStridedRef<'a, T> {
        RawStridedRef::new(data, &self.dims, &self.src, self.src_offset).unwrap()
    }

    fn dst_view<'a, T>(&'a self, data: &'a mut [T]) -> RawStridedMut<'a, T> {
        RawStridedMut::new(data, &self.dims, &self.dst, self.dst_offset).unwrap()
    }

    fn given_tiles<D, S>(&self) -> Option<PairTiles> {
        PairTraversal::given(&self.dims, &self.dst, &self.src)
            .inline()
            .and_then(pair_tiles::<D, S>)
    }
}

/// Offset that makes the lowest reachable slot 0.
fn lowest(dims: &[usize], strides: &[isize]) -> isize {
    dims.iter()
        .zip(strides)
        .filter(|(&d, &s)| d > 1 && s < 0)
        .map(|(&d, &s)| (d as isize - 1) * -s)
        .sum()
}

/// `dims` with `strides(axis)` reversed over the axes.
fn reversed(dims: &[usize]) -> Layout {
    let mut stride = 1isize;
    let dst: Vec<isize> = dims
        .iter()
        .map(|&d| {
            let s = stride;
            stride *= d as isize;
            s
        })
        .collect();
    let mut stride = 1isize;
    let mut src = vec![0isize; dims.len()];
    for axis in (0..dims.len()).rev() {
        src[axis] = stride;
        stride *= dims[axis] as isize;
    }
    Layout::new(dims, &dst, &src, 0)
}

fn layouts() -> Vec<(&'static str, Layout)> {
    vec![
        // Clipped tiles: blocks [43, 42] do not divide [67, 45].
        (
            "67x45 transpose",
            Layout::new(&[67, 45], &[1, 67], &[45, 1], 0),
        ),
        // A stored [37, 25, 41] source, permuted, reversed along its unit
        // axis and cropped at a nonzero base.
        (
            "rank 3 permuted, negative, cropped",
            Layout::new(&[40, 36, 24], &[1, 40, 1440], &[925, -1, 37], 36 + 37),
        ),
        ("16^4 reversed", reversed(&[16; 4])),
        ("4^8 reversed", reversed(&[4; 8])),
        // An extent-1 axis that `pair_tiles` drops.
        (
            "[1, 65, 65] transpose",
            Layout::new(&[1, 65, 65], &[3, 1, 65], &[5, 65, 1], 0),
        ),
    ]
}

trait Value: Copy + Send + Sync + PartialEq + core::fmt::Debug {
    fn value(k: usize) -> Self;
    fn bits(self) -> [u64; 2];
}

impl Value for f64 {
    fn value(k: usize) -> Self {
        (k % 97) as f64 * 0.25 - 7.0
    }
    fn bits(self) -> [u64; 2] {
        [self.to_bits(), 0]
    }
}

impl Value for Complex64 {
    fn value(k: usize) -> Self {
        Complex64::new((k % 89) as f64 * 0.5 - 3.0, (k % 13) as f64 - 6.0)
    }
    fn bits(self) -> [u64; 2] {
        [self.re.to_bits(), self.im.to_bits()]
    }
}

fn bits<T: Value>(values: &[T]) -> Vec<[u64; 2]> {
    values.iter().map(|v| v.bits()).collect()
}

/// `f(dst_slot, src_slot)` applied per logical index in column-major order.
fn oracle<D: Copy, S: Copy>(layout: &Layout, dst: &mut [D], src: &[S], f: impl Fn(&mut D, S)) {
    for (d, s) in layout.offsets() {
        f(&mut dst[d], src[s]);
    }
}

fn uninit_bits<T: Value>(values: &[MaybeUninit<T>]) -> Vec<[u64; 2]> {
    // SAFETY: test storage starts initialized and only receives initialized values.
    values
        .iter()
        .map(|v| unsafe { v.assume_init() }.bits())
        .collect()
}

/// Every typed entry on `layout` with token `exec()` against [`oracle`].
fn check_entries<'e, T>(layout: &Layout, exec: impl Fn() -> Exec<'e>, label: &str)
where
    T: Value
        + Mul<Output = T>
        + Add<Output = T>
        + ElementOpApply
        + num_traits::Zero
        + num_traits::One,
{
    let (dst_len, src_len) = layout.lens();
    let src: Vec<T> = (0..src_len).map(T::value).collect();
    let init: Vec<T> = (0..dst_len).map(|k| T::value(k + 11)).collect();
    let source = layout.src_view(&src);
    let scale = T::value(3);
    let beta = T::value(40);

    let check = |name: &str, f: &dyn Fn(&mut T, T), run: &dyn Fn(&mut RawStridedMut<'_, T>)| {
        let mut expected = init.clone();
        oracle(layout, &mut expected, &src, f);
        let mut actual = init.clone();
        run(&mut layout.dst_view(&mut actual));
        assert_eq!(bits(&actual), bits(&expected), "{label}: {name}");
    };
    check("replay_copy_raw", &|d, s| *d = s, &|d| {
        replay_copy_raw(d, &source, exec()).unwrap()
    });
    check("replay_map_raw +=", &|d, s| *d = *d + s, &|d| {
        replay_map_raw(d, &source, |d: &mut T, s: T| *d = *d + s, exec()).unwrap()
    });
    let copy = CopyPlan::compile(&layout.dims, &layout.dst, &layout.src).unwrap();
    let update = UpdatePlan::compile(&layout.dims, &layout.dst, &layout.src).unwrap();
    check("CopyPlan::execute", &|d, s| *d = s, &|d| {
        copy.execute(d, &source, exec()).unwrap()
    });
    check("CopyPlan::execute_scale", &|d, s| *d = scale * s, &|d| {
        copy.execute_scale(d, &source, scale, exec()).unwrap()
    });
    check("CopyPlan::execute_conj", &|d, s| *d = s.conj(), &|d| {
        copy.execute_conj(d, &source, exec()).unwrap()
    });
    check(
        "UpdatePlan::execute beta != 0",
        &|d, s| *d = scale * (s * scale) + beta * *d,
        &|d| {
            update
                .execute(d, &source, scale, beta, |x| x * scale, exec())
                .unwrap()
        },
    );

    // Uninitialized destinations: the holes keep their sentinel.
    let sentinel = T::value(1000);
    let check_uninit =
        |name: &str,
         f: &dyn Fn(&mut T, T),
         run: &dyn Fn(&mut RawStridedMut<'_, MaybeUninit<T>>)| {
            let mut expected = vec![sentinel; dst_len];
            oracle(layout, &mut expected, &src, f);
            let mut actual = vec![MaybeUninit::new(sentinel); dst_len];
            run(&mut layout.dst_view(&mut actual));
            assert_eq!(uninit_bits(&actual), bits(&expected), "{label}: {name}");
        };
    check_uninit("replay_copy_raw_uninit", &|d, s| *d = s, &|d| {
        replay_copy_raw_uninit(d, &source, exec()).unwrap()
    });
    check_uninit("CopyPlan::execute_uninit", &|d, s| *d = s, &|d| {
        copy.execute_uninit(d, &source, exec()).unwrap()
    });
    check_uninit("UpdatePlan::execute_uninit", &|d, s| *d = scale * s, &|d| {
        update
            .execute_uninit(d, &source, scale, |x| x, exec())
            .unwrap()
    });
}

/// The `f64`-only rows: beta = 0 over a NaN destination, and a mixed-type
/// uninitialized map.
fn check_f64_rows<'e>(layout: &Layout, exec: impl Fn() -> Exec<'e>, label: &str) {
    let (dst_len, src_len) = layout.lens();
    let src: Vec<f64> = (0..src_len).map(f64::value).collect();
    let update = UpdatePlan::compile(&layout.dims, &layout.dst, &layout.src).unwrap();
    let mut expected = vec![f64::NAN; dst_len];
    oracle(layout, &mut expected, &src, |d, s| *d = 2.0 * s);
    let mut actual = vec![f64::NAN; dst_len];
    update
        .execute(
            &mut layout.dst_view(&mut actual),
            &layout.src_view(&src),
            2.0,
            0.0,
            |x| x,
            exec(),
        )
        .unwrap();
    assert_eq!(bits(&actual), bits(&expected), "{label}: beta = 0");

    let ints: Vec<i32> = (0..src_len as i32).map(|k| k * 3 - 50).collect();
    let mut expected = vec![-0.5f64; dst_len];
    oracle(layout, &mut expected, &ints, |d, s| *d = f64::from(s) * 0.5);
    let mut actual = vec![MaybeUninit::new(-0.5f64); dst_len];
    replay_map_raw_uninit(
        &mut layout.dst_view(&mut actual),
        &layout.src_view(&ints),
        |s| f64::from(s) * 0.5,
        exec(),
    )
    .unwrap();
    assert_eq!(uninit_bits(&actual), bits(&expected), "{label}: i32 -> f64");
}

#[test]
fn tiled_replays_equal_the_column_major_oracle() {
    let team = team();
    for (label, layout) in layouts() {
        assert!(
            layout.given_tiles::<f64, f64>().is_some(),
            "{label} must take the tiled walk"
        );
        for parallel in [false, true] {
            let exec = || {
                if parallel {
                    team.exec(4)
                } else {
                    Exec::serial()
                }
            };
            check_entries::<f64>(&layout, exec, label);
            check_entries::<Complex64>(&layout, exec, label);
            check_f64_rows(&layout, exec, label);
        }
    }
    // One clipped tile: 67 = 43 + 24 and 45 = 42 + 3.
    let tiles = layouts()[0].1.given_tiles::<f64, f64>().unwrap();
    assert_eq!(tiles.blocks[..2], [43, 42]);
    // The extent-1 axis is dropped before the model.
    let tiles = layouts()[4].1.given_tiles::<f64, f64>().unwrap();
    assert_eq!(tiles.layout.rank, 2);
    assert_eq!(tiles.layout.dims[..2], [65, 65]);
}

#[test]
fn tiled_dynamic_slice_window_equals_the_oracle() {
    use crate::DynamicSlicePlan;
    // A transposing window of a [100, 90] operand at runtime start (7, 11).
    let operand_dims = [100usize, 90];
    let operand_strides = [90isize, 1];
    let sizes = [67usize, 45];
    let dest_strides = [1isize, 67];
    assert!(pair_tiles::<MaybeUninit<f64>, f64>(
        &fuse_pair_layout(&sizes, &dest_strides, &operand_strides).unwrap()
    )
    .is_some());
    let plan = DynamicSlicePlan::compile(
        &operand_dims,
        &operand_strides,
        &[2],
        &[1],
        &sizes,
        &dest_strides,
        &sizes,
    )
    .unwrap();
    let data: Vec<f64> = (0..9000).map(f64::value).collect();
    let operand = RawStridedRef::new(&data, &operand_dims, &operand_strides, 0).unwrap();
    let starts_data = [7i64, 11];
    let starts = RawStridedRef::new(&starts_data, &[2], &[1], 0).unwrap();
    let mut expected = vec![0.0f64; 67 * 45];
    for j in 0..45 {
        for i in 0..67 {
            expected[i + 67 * j] = data[(i + 7) * 90 + (j + 11)];
        }
    }
    let team = team();
    for exec in [Exec::serial(), team.exec(4)] {
        let mut out = vec![0.0f64; 67 * 45];
        plan.execute(
            &mut RawStridedMut::new(&mut out, &sizes, &dest_strides, 0).unwrap(),
            &operand,
            &starts,
            exec,
        )
        .unwrap();
        assert_eq!(bits(&out), bits(&expected));
    }
}

#[test]
fn layouts_whose_tiles_keep_the_order_take_the_single_pass() {
    let none = |dims: &[usize], dst: &[isize], src: &[isize]| {
        PairTraversal::given(dims, dst, src)
            .inline()
            .and_then(pair_tiles::<f64, f64>)
            .is_none()
    };
    // Strictly increasing strides in both operands.
    assert!(none(&[3840, 512], &[1, 3840], &[1, 4096]));
    assert!(none(&[4096, 512], &[1, 4096], &[-1, 4096]));
    // Zero-length and single-element disagreeing layouts.
    assert!(none(&[0, 64], &[64, 1], &[1, 64]));
    assert!(none(&[1, 1], &[64, 1], &[1, 64]));
    // Every byte stride above the budget: all blocks are 1.
    assert!(none(&[64, 64], &[5000, 320_000], &[320_000, 5000]));
    // One run: rank one, or one moving axis among extent-1 axes.
    assert!(none(&[4096], &[1], &[-3]));
    assert!(none(&[1, 4096, 1], &[7, 1, 9], &[5, -3, 11]));

    // On a team token the empty and single-element cases dispatch nothing.
    let team = team();
    let src = [2.5f64; 1];
    let mut dst = [0.0f64; 1];
    crate::threading::take_dispatched_chunks();
    replay_copy_raw(
        &mut RawStridedMut::new(&mut dst, &[1, 1], &[64, 1], 0).unwrap(),
        &RawStridedRef::new(&src, &[1, 1], &[1, 64], 0).unwrap(),
        team.exec(4),
    )
    .unwrap();
    assert_eq!(dst, [2.5]);
    let empty: [f64; 0] = [];
    let mut out: [f64; 0] = [];
    replay_copy_raw(
        &mut RawStridedMut::new(&mut out, &[0, 64], &[64, 1], 0).unwrap(),
        &RawStridedRef::new(&empty, &[0, 64], &[1, 64], 0).unwrap(),
        team.exec(4),
    )
    .unwrap();
    assert_eq!(crate::threading::take_dispatched_chunks(), 0);
}

/// Whether `pair_tiles::<D, S>` selects tiles for the given axes.
fn tiles<D, S>(dims: &[usize], dst: &[isize], src: &[isize]) -> bool {
    PairTraversal::given(dims, dst, src)
        .inline()
        .and_then(pair_tiles::<D, S>)
        .is_some()
}

#[test]
fn slabs_that_share_no_cache_line_keep_the_flat_walk() {
    // F3: both operands stream whole 32 KiB runs along axis 0 and the tail
    // axes only permute those runs; the model still offers [4096, 16, 16].
    let f3 = Layout::new(&[4096, 17, 33], &[1, 4096, 69_632], &[1, 135_168, 4096], 0);
    let model = crate::block::compute_block_sizes(&f3.dims, &[&f3.dst, &f3.src], 8);
    assert_eq!(model.as_slice(), [4096, 16, 16]);
    assert!(f3.given_tiles::<f64, f64>().is_none());
    // A zero-sized source moves no line; the destination alone decides.
    assert!(f3.given_tiles::<f64, ()>().is_none());
    // Sparse prefixes whose bounding intervals are separated and line-sized,
    // the second with a whole-layout span above the model's 64 * N region.
    assert!(!tiles::<f64, f64>(
        &[8, 17, 33],
        &[81, 568, 9656],
        &[81, 18744, 568]
    ));
    assert!(!tiles::<f64, f64>(
        &[8, 17, 33],
        &[9, 72, 1224],
        &[9, 2376, 72]
    ));

    // Not declined: equal source tail strides re-read the same slab, so
    // per-axis separation alone is not enough.
    assert!(tiles::<f64, f64>(
        &[4096, 17, 16],
        &[1, 65_536, 4096],
        &[1, 4096, 4096]
    ));
    // Not declined: under the model's 8-byte size the slabs look line-sized,
    // but the u8 source's 8-byte slabs share lines with their neighbours.
    let mixed = Layout::new(&[8, 512, 512], &[1, 8, 4096], &[1, 4096, 8], 0);
    assert!(mixed.given_tiles::<u64, u8>().is_some());
    assert!(mixed.given_tiles::<f64, f64>().is_none());

    let team = team();
    let (dst_len, src_len) = f3.lens();
    let src: Vec<f64> = (0..src_len).map(f64::value).collect();
    let mut expected = vec![0.0f64; dst_len];
    oracle(&f3, &mut expected, &src, |d, s| *d = s);
    let mut actual = vec![0.0f64; dst_len];
    crate::threading::take_tiled_leaves();
    crate::threading::take_dispatched_chunks();
    replay_copy_raw(
        &mut f3.dst_view(&mut actual),
        &f3.src_view(&src),
        team.exec(4),
    )
    .unwrap();
    assert_eq!(bits(&actual), bits(&expected));
    assert!(crate::threading::take_tiled_leaves().is_empty());
    let chunks = crate::threading::take_dispatched_chunks();
    #[cfg(feature = "parallel")]
    assert!(chunks >= 2, "F3 runs the flat split ({chunks} chunks)");
    #[cfg(not(feature = "parallel"))]
    let _ = chunks;

    let (dst_len, src_len) = mixed.lens();
    let bytes: Vec<u8> = (0..src_len).map(|k| (k * 7) as u8).collect();
    let mut expected = vec![0u64; dst_len];
    oracle(&mixed, &mut expected, &bytes, |d, s| *d = u64::from(s) * 3);
    let mut actual = vec![0u64; dst_len];
    replay_map_raw(
        &mut mixed.dst_view(&mut actual),
        &mixed.src_view(&bytes),
        |d: &mut u64, s: u8| *d = u64::from(s) * 3,
        team.exec(4),
    )
    .unwrap();
    assert_eq!(actual, expected);
}

/// Every logical index of `layout` covered by the recorded `leaves`, counted
/// independently of the replay: each origin is decoded from its destination
/// offset through the layout's dense destination strides and checked against
/// its source offset.
#[cfg(feature = "parallel")]
fn leaf_visits(layout: &Layout, leaves: &[crate::threading::TiledLeaf]) -> Vec<u8> {
    let rank = layout.dims.len();
    let mut order: Vec<usize> = (0..rank).collect();
    order.sort_by_key(|&axis| layout.dst[axis]);
    let mut visits = vec![0u8; layout.total()];
    for (dst_offset, src_offset, extents) in leaves {
        assert_eq!(extents.len(), rank);
        let mut rest = dst_offset - layout.dst_offset;
        let mut origin = vec![0usize; rank];
        for &axis in order.iter().rev() {
            origin[axis] = (rest / layout.dst[axis]) as usize;
            rest %= layout.dst[axis];
        }
        assert_eq!(rest, 0);
        let src_at: isize = layout.src_offset
            + origin
                .iter()
                .zip(&layout.src)
                .map(|(&c, &s)| c as isize * s)
                .sum::<isize>();
        assert_eq!(*src_offset, src_at, "source offset of leaf at {origin:?}");
        let count: usize = extents.iter().product();
        for linear in 0..count {
            let mut rest = linear;
            let mut index = 0usize;
            let mut pitch = 1usize;
            for axis in 0..rank {
                let coord = origin[axis] + rest % extents[axis];
                rest /= extents[axis];
                assert!(coord < layout.dims[axis], "leaf beyond the box");
                index += coord * pitch;
                pitch *= layout.dims[axis];
            }
            visits[index] += 1;
        }
    }
    visits
}

#[test]
fn split_leaves_partition_the_box() {
    // A dense permutation that tiles with blocks [35, 17, 2]; at width 4 the
    // split cuts axis 2 into origins 0, 8, 16, 24.
    let dims = [64usize, 17, 33];
    let layout = Layout::new(&dims, &[1, 64, 1088], &[561, 1, 17], 0);
    let tiles = layout.given_tiles::<f64, f64>().unwrap();
    assert_eq!(tiles.blocks[..3], [35, 17, 2]);
    let (dst_len, src_len) = layout.lens();
    let src: Vec<f64> = (0..src_len).map(f64::value).collect();
    let mut expected = vec![0.0f64; dst_len];
    oracle(&layout, &mut expected, &src, |d, s| *d = s);
    let mut actual = vec![0.0f64; dst_len];
    let team = team();
    crate::threading::take_tiled_leaves();
    replay_copy_raw(
        &mut layout.dst_view(&mut actual),
        &layout.src_view(&src),
        team.exec(4),
    )
    .unwrap();
    assert_eq!(bits(&actual), bits(&expected));
    let leaves = crate::threading::take_tiled_leaves();
    #[cfg(feature = "parallel")]
    {
        assert!(leaf_visits(&layout, &leaves).iter().all(|&v| v == 1));
        assert_eq!(leaves.len(), 4);
        // Each cut keeps at most ceil(d / 2) / d of its parent.
        let n = layout.total() as f64;
        for (_, _, extents) in &leaves {
            let mut bound = n / 4.0;
            for (axis, &extent) in extents.iter().enumerate() {
                let mut d = dims[axis];
                while d > extent {
                    bound *= 2.0 * d.div_ceil(2) as f64 / d as f64;
                    d = d.div_ceil(2);
                }
            }
            assert!(
                extents.iter().product::<usize>() as f64 <= bound,
                "{leaves:?}"
            );
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = leaves;
}

#[test]
fn tiled_replay_keeps_the_callers_budget() {
    // With D(2) = estimate / 100 and D(4) = D(6) = 10 * estimate the flat
    // width rule grants 2, while every node of the tree pays a fork at
    // width 2: a tiled replay must split into 4 leaves, as the plan does.
    use threadteam::{DispatchCosts, DispatchKind, HostClass, WidthCosts};
    let layout = Layout::new(&[128, 128], &[1, 128], &[128, 1], 0);
    assert!(layout.given_tiles::<f64, f64>().is_some());
    let estimate = KernelClass::Copy.estimate(layout.total());
    let costs = WidthCosts::new([estimate / 100, estimate * 10, estimate * 10]);
    let pool = threadteam::Pool::builder()
        .threads(4)
        .dispatch_costs(
            DispatchCosts::for_class(HostClass::Generic)
                .with(DispatchKind::Team, costs)
                .with(DispatchKind::InPool, costs)
                .with(DispatchKind::Outside, costs),
        )
        .acquire_timeout(core::time::Duration::from_secs(10))
        .build()
        .unwrap();
    let team = pool.into_team(4).expect("an idle unit-test pool");
    #[cfg(feature = "parallel")]
    assert_eq!(
        crate::cost::parallel_width(&team.exec(4), KernelClass::Copy, layout.total()).get(),
        2
    );
    let (dst_len, src_len) = layout.lens();
    let src: Vec<f64> = (0..src_len).map(f64::value).collect();
    let mut expected = vec![0.0f64; dst_len];
    oracle(&layout, &mut expected, &src, |d, s| *d = s);
    let update = UpdatePlan::compile(&layout.dims, &layout.dst, &layout.src).unwrap();
    for entry in ["replay_map_raw", "UpdatePlan::execute"] {
        let mut actual = vec![0.0f64; dst_len];
        crate::threading::take_tiled_leaves();
        if entry == "replay_map_raw" {
            replay_map_raw(
                &mut layout.dst_view(&mut actual),
                &layout.src_view(&src),
                |d: &mut f64, s| *d = s,
                team.exec(4),
            )
            .unwrap();
        } else {
            update
                .execute(
                    &mut layout.dst_view(&mut actual),
                    &layout.src_view(&src),
                    1.0,
                    0.0,
                    |x| x,
                    team.exec(4),
                )
                .unwrap();
        }
        assert_eq!(bits(&actual), bits(&expected), "{entry}");
        let leaves = crate::threading::take_tiled_leaves();
        #[cfg(feature = "parallel")]
        {
            let mut extents: Vec<Vec<usize>> = leaves.iter().map(|l| l.2.clone()).collect();
            extents.sort();
            assert_eq!(extents, vec![vec![64, 64]; 4], "{entry}");
            assert!(leaf_visits(&layout, &leaves).iter().all(|&v| v == 1));
        }
        #[cfg(not(feature = "parallel"))]
        let _ = leaves;
    }
}

#[test]
fn non_injective_destination_replays_serially_in_column_major_order() {
    // Axis 2 writes every slot twice; the source disagrees with the
    // destination, so an injective destination of this shape would tile.
    let dims = [64usize, 64, 2];
    let dst = [1isize, 64, 0];
    let src = [64isize, 1, 4096];
    let layout = Layout::new(&dims, &dst, &src, 0);
    assert!(layout.given_tiles::<i64, i64>().is_some());
    let (dst_len, src_len) = layout.lens();
    let values: Vec<i64> = (0..src_len as i64).map(|k| k * 7 - 1000).collect();
    let team = team();
    for parallel in [false, true] {
        let exec = || {
            let exec = if parallel {
                team.exec(4)
            } else {
                Exec::serial()
            };
            let root = exec.ledger_root();
            (exec, root)
        };
        let assert_serial = |root: Option<u64>| {
            if let Some(root) = root {
                assert_eq!(ledger::max_live(root), 0, "must not fan out");
            }
        };
        let mut expected = vec![3i64; dst_len];
        oracle(&layout, &mut expected, &values, |d, s| *d = s);
        let mut actual = vec![3i64; dst_len];
        let (e, root) = exec();
        replay_copy_raw(
            &mut layout.dst_view(&mut actual),
            &layout.src_view(&values),
            e,
        )
        .unwrap();
        assert_eq!(actual, expected, "last writer");
        assert_serial(root);
        let mut expected = vec![3i64; dst_len];
        oracle(&layout, &mut expected, &values, |d, s| *d = *d * 3 + s);
        let mut actual = vec![3i64; dst_len];
        let (e, root) = exec();
        replay_map_raw(
            &mut layout.dst_view(&mut actual),
            &layout.src_view(&values),
            |d: &mut i64, s| *d = *d * 3 + s,
            e,
        )
        .unwrap();
        assert_eq!(actual, expected, "order-sensitive accumulate");
        assert_serial(root);
    }
}

#[test]
fn inert_singleton_strides_stay_out_of_the_arithmetic() {
    // A stride of `isize::MAX` on an extent-1 axis is valid and unused.
    let m = isize::MAX;
    let layout = Layout::new(&[1, 65, 65], &[m, 1, 65], &[m, 65, 1], 0);
    let tiles = layout.given_tiles::<f64, f64>().unwrap();
    assert_eq!(tiles.blocks[..2], [56, 33]);
    let (dst_len, src_len) = layout.lens();
    let src: Vec<f64> = (0..src_len).map(f64::value).collect();
    let mut expected = vec![0.0f64; dst_len];
    oracle(&layout, &mut expected, &src, |d, s| *d = s);
    let team = team();
    for parallel in [false, true] {
        let exec = || {
            if parallel {
                team.exec(4)
            } else {
                Exec::serial()
            }
        };
        let mut actual = vec![0.0f64; dst_len];
        replay_copy_raw(
            &mut layout.dst_view(&mut actual),
            &layout.src_view(&src),
            exec(),
        )
        .unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        let mut actual = vec![1.0f64; dst_len];
        let mut accumulated = vec![1.0f64; dst_len];
        oracle(&layout, &mut accumulated, &src, |d, s| *d += s);
        replay_map_raw(
            &mut layout.dst_view(&mut actual),
            &layout.src_view(&src),
            |d: &mut f64, s| *d += s,
            exec(),
        )
        .unwrap();
        assert_eq!(bits(&actual), bits(&accumulated));
    }
}

/// Isolated arithmetic self-check of the tile odometer, not route coverage:
/// hand-built tiles over zero-sized layouts whose last slot lies within one
/// stride of `isize::MAX` visit every index once without overflowing.
#[test]
fn tile_odometer_stays_within_the_reachable_offsets() {
    let m = isize::MAX;
    // (dims, destination strides, source strides, both bases, blocks)
    type Case<'a> = (&'a [usize], &'a [isize], &'a [isize], isize, &'a [usize]);
    let cases: [Case<'_>; 3] = [
        (&[2, 2], &[1, m - 1], &[m - 1, 1], 0, &[1, 1]),
        (&[2, 2], &[-1, -(m - 1)], &[-(m - 1), -1], m, &[1, 2]),
        (&[2, 2, 2], &[1, 2, m - 3], &[m - 3, 2, 1], 0, &[1, 2, 1]),
    ];
    for (dims, dst, src, base, blocks) in cases {
        let axes: Vec<FusedAxis> = dims
            .iter()
            .zip(dst)
            .zip(src)
            .map(|((&dim, &dst), &src)| FusedAxis { dim, dst, src })
            .collect();
        let layout = FusedPairLayout::from_axes(&axes);
        let mut tile_blocks = [1usize; RAW_FUSED_RANK_LIMIT];
        tile_blocks[..blocks.len()].copy_from_slice(blocks);
        let visited = core::cell::Cell::new(0usize);
        let mut unit = ();
        // SAFETY: zero-sized elements make every offset of the dangling
        // pointers a valid access.
        unsafe {
            apply_tiled_box(
                &mut unit as *mut (),
                base,
                &() as *const (),
                base,
                &layout,
                &tile_blocks,
                &|_: &mut (), _: ()| visited.set(visited.get() + 1),
            );
        }
        assert_eq!(visited.get(), dims.iter().product::<usize>(), "{dims:?}");
    }
}
