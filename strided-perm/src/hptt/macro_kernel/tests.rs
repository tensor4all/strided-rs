use super::*;
use crate::hptt::PlainStore;
use std::cell::RefCell;

#[test]
fn test_macro_kernel_f64_full_block() {
    // 16×16 tile, lda=16, ldb=16 (both square)
    let n = 16;
    let src: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; n * n];

    // SAFETY: (macro_kernel.rs:22) separate buffers; the largest offset is 15 + 15 * 16 = 255 on
    // both sides, below both lengths (256).
    unsafe {
        macro_kernel_f64(src.as_ptr(), n as isize, n, dst.as_mut_ptr(), n as isize, n);
    }

    for j in 0..n {
        for i in 0..n {
            assert_eq!(dst[j + i * n], src[i + j * n], "mismatch at i={i}, j={j}");
        }
    }
}

#[test]
fn test_macro_kernel_f64_with_remainder() {
    // 15×17 tile (both have remainders w.r.t. MICRO=4)
    let block_a = 15;
    let block_b = 17;
    let lda = 20isize; // src leading dim
    let ldb = 18isize; // dst leading dim

    let src: Vec<f64> = (0..(block_a as isize + (block_b - 1) as isize * lda + 1) as usize)
        .map(|i| i as f64)
        .collect();
    let mut dst =
        vec![0.0f64; ((block_b - 1) as isize + (block_a - 1) as isize * ldb + 1) as usize];

    // SAFETY: (macro_kernel.rs:22) separate buffers; the largest offset is 14 + 16 * 20 = 334 <
    // 336 (source) and 16 + 14 * 18 = 268 < 269 (destination).
    unsafe {
        macro_kernel_f64(src.as_ptr(), lda, block_a, dst.as_mut_ptr(), ldb, block_b);
    }

    for j in 0..block_b {
        for i in 0..block_a {
            let s = src[(i as isize + j as isize * lda) as usize];
            let d = dst[(j as isize + i as isize * ldb) as usize];
            assert_eq!(d, s, "mismatch at i={i}, j={j}");
        }
    }
}

#[test]
fn test_macro_kernel_f64_small() {
    // 3×5 tile (smaller than MICRO=4)
    let block_a = 3;
    let block_b = 5;
    let lda = 8isize;
    let ldb = 6isize;

    let src_len = ((block_a - 1) as isize + (block_b - 1) as isize * lda + 1) as usize;
    let dst_len = ((block_b - 1) as isize + (block_a - 1) as isize * ldb + 1) as usize;
    let src: Vec<f64> = (0..src_len).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; dst_len];

    // SAFETY: (macro_kernel.rs:22) separate buffers sized to the largest offset plus one: 2 + 4 *
    // 8 = 34 (source) and 4 + 2 * 6 = 16 (destination).
    unsafe {
        macro_kernel_f64(src.as_ptr(), lda, block_a, dst.as_mut_ptr(), ldb, block_b);
    }

    for j in 0..block_b {
        for i in 0..block_a {
            let s = src[(i as isize + j as isize * lda) as usize];
            let d = dst[(j as isize + i as isize * ldb) as usize];
            assert_eq!(d, s, "mismatch at i={i}, j={j}");
        }
    }
}

#[test]
fn test_const_stride1_copy_contiguous() {
    let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let mut dst = vec![0.0f64; 5];
    // SAFETY: (macro_kernel.rs:172) separate five-element buffers; offsets 0..5 on both sides.
    unsafe {
        const_stride1_copy::<_, PlainStore>(src.as_ptr(), dst.as_mut_ptr(), 5, 1, 1);
    }
    assert_eq!(dst, src);
}

#[test]
fn test_const_stride1_copy_strided() {
    let src = [1.0f64, 0.0, 2.0, 0.0, 3.0];
    let mut dst = vec![0.0f64; 5];
    // SAFETY: (macro_kernel.rs:172) separate buffers; source offsets 0, 2, 4 < 5, destination
    // offsets 0, 1, 2 < 5.
    unsafe {
        const_stride1_copy::<_, PlainStore>(src.as_ptr(), dst.as_mut_ptr(), 3, 2, 1);
    }
    assert_eq!(dst[0], 1.0);
    assert_eq!(dst[1], 2.0);
    assert_eq!(dst[2], 3.0);
}

#[test]
fn const_stride1_leaf_advances_wrap_at_the_allocation_end() {
    // Two elements with source stride 3 live at offsets 0 and 3 in a
    // 4-element buffer; the copy's terminal advance would read offset 6.
    // The cursor is not dereferenced after the last copy, so the advance
    // must be a wrapping one; Miri flags the pre-fix `offset`.
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let mut dst = vec![0.0f64; 2];
    // SAFETY: (macro_kernel.rs:172) separate buffers; source offsets 0, 3 < 4, destination
    // offsets 0, 1 < 2. The terminal advance wraps and is not read.
    unsafe {
        const_stride1_copy::<_, PlainStore>(src.as_ptr(), dst.as_mut_ptr(), 2, 3, 1);
    }
    assert_eq!(dst, vec![1.0, 4.0]);
}

thread_local! {
    /// Destination offsets, relative to [`ORIGIN`], in the order
    /// [`RecordingStore`] wrote them.
    static WRITES: RefCell<Vec<isize>> = const { RefCell::new(Vec::new()) };
    /// The tile's destination origin, so a recorded offset is `j + i * ldb`.
    static ORIGIN: std::cell::Cell<isize> = const { std::cell::Cell::new(0) };
}

/// A [`Store`] that records the destination offset of each store, so a test can
/// assert the traversal the kernel chose rather than only its result.
struct RecordingStore;

impl Store<u8> for RecordingStore {
    const PLAIN: bool = false;

    #[inline(always)]
    unsafe fn store(dst: *mut u8, src: *const u8) {
        let off = dst as isize - ORIGIN.with(|o| o.get());
        WRITES.with(|w| w.borrow_mut().push(off));
        // SAFETY: the caller guarantees both accesses.
        unsafe { *dst = *src };
    }
}

/// Take the recorded offsets as `(i, j)` pairs, decoded from a destination whose
/// `j` axis is packed into `block_b`: every offset is `j + i * span`.
fn recorded(span: isize, block_b: usize) -> Vec<(isize, isize)> {
    WRITES.with(|w| {
        w.borrow()
            .iter()
            .map(|&off| (off / span, off % span))
            .filter(|&(_, j)| (0..block_b as isize).contains(&j))
            .collect()
    })
}

/// The three branches must not only write the same tile, they must walk it in
/// the order the predicate chose: this is what a wrong branch looks like.
#[test]
fn each_branch_walks_the_tile_it_was_picked_for() {
    const BLOCK_B: usize = 6;
    for block_a in [1usize, 5, 6, 7, 12, 13] {
        for (lda, ldb, walk) in [
            // `u8` elements: a stride of 4096 of them is a 4096-byte row, the
            // aliasing one. Both sides alias, so the shrink branch.
            (4096isize, 4096isize, Walk::Shrink),
            // Destination only: the full-run branch.
            (19, 4096, Walk::Rows),
            // Source only, and neither: the original order.
            (4096, 17, Walk::Cols),
            (19, 19, Walk::Cols),
        ] {
            let src_len = (block_a as isize - 1) + (BLOCK_B as isize - 1) * lda + 1;
            let dst_len = (BLOCK_B as isize - 1) + (block_a as isize - 1) * ldb + 1;
            let mut src = vec![0u8; src_len as usize];
            let mut dst = vec![0u8; dst_len as usize];
            WRITES.with(|w| w.borrow_mut().clear());
            ORIGIN.with(|o| o.set(dst.as_ptr() as isize));
            // SAFETY: (macro_kernel.rs:231) both buffers hold every offset this
            // tile reaches, computed above from the same extents and strides.
            unsafe {
                macro_kernel_fallback::<u8, RecordingStore>(
                    src.as_mut_ptr(),
                    lda,
                    block_a,
                    dst.as_mut_ptr(),
                    ldb,
                    BLOCK_B,
                );
            }
            let got = recorded(ldb, BLOCK_B);
            assert_eq!(
                got.len(),
                block_a * BLOCK_B,
                "every tile element is written once: {walk:?} block_a={block_a}"
            );
            let head = |n: usize| got.iter().take(n).copied().collect::<Vec<_>>();
            match walk {
                // `i` runs through its six-row chunk before `j` advances.
                Walk::Shrink => {
                    let chunk = 6.min(block_a);
                    assert_eq!(
                        head(chunk).iter().map(|&(i, _)| i).collect::<Vec<_>>(),
                        (0..chunk as isize).collect::<Vec<_>>(),
                        "shrink branch: block_a={block_a}"
                    );
                    if block_a > chunk {
                        // Shrinking is the whole point: the seventh write must
                        // still be inside the first `i` chunk, not the next
                        // `i` of an unchunked `j` sweep. Unchunked it would be
                        // `(chunk, 0)`.
                        assert_eq!(
                            got[chunk],
                            (0, 1),
                            "shrink branch does not restart j per chunk: block_a={block_a}"
                        );
                    }
                }
                // `j` sweeps the whole run for a fixed `i`.
                Walk::Rows => assert_eq!(
                    head(BLOCK_B).iter().map(|&(_, j)| j).collect::<Vec<_>>(),
                    (0..BLOCK_B as isize).collect::<Vec<_>>(),
                    "rows branch: block_a={block_a}"
                ),
                // `i` sweeps the whole extent for a fixed `j`.
                Walk::Cols => assert_eq!(
                    head(block_a).iter().map(|&(i, _)| i).collect::<Vec<_>>(),
                    (0..block_a as isize).collect::<Vec<_>>(),
                    "cols branch: block_a={block_a}"
                ),
            }
        }
    }
}

/// The traversal a test expects [`macro_kernel_fallback`] to pick.
#[derive(Clone, Copy, Debug)]
enum Walk {
    Shrink,
    Rows,
    Cols,
}

/// One [`macro_kernel_fallback`] tile of `E`-typed elements against the
/// reference transpose, over `block_a` x `block_b` with destination row stride
/// `ldb`.
fn check_fallback_tile<E: Copy + core::fmt::Debug + Eq>(
    label: &str,
    lda: isize,
    block_a: usize,
    ldb: isize,
    block_b: usize,
    make: impl Fn(usize) -> E + Copy,
) {
    // Offsets reachable on each side, `i < block_a` and `j < block_b`, against
    // the two stride pairs; a negative stride moves the base pointer instead.
    let span = |si: isize, sj: isize| {
        let lo = (block_a as isize - 1) * si.min(0) + (block_b as isize - 1) * sj.min(0);
        let hi = (block_a as isize - 1) * si.max(0) + (block_b as isize - 1) * sj.max(0);
        (lo, hi)
    };
    let (src_lo, src_hi) = span(1, lda);
    let (dst_lo, dst_hi) = span(ldb, 1);
    let src: Vec<E> = (0..(src_hi - src_lo + 1) as usize).map(make).collect();
    let mut dst: Vec<E> = (0..(dst_hi - dst_lo + 1) as usize).map(make).collect();
    // SAFETY: (macro_kernel.rs:231) both buffers are separate, and a base
    // shifted by `-lo` leaves every offset the kernel reaches inside them.
    unsafe {
        macro_kernel_fallback::<E, PlainStore>(
            src.as_ptr().offset(-src_lo),
            lda,
            block_a,
            dst.as_mut_ptr().offset(-dst_lo),
            ldb,
            block_b,
        );
    }
    for j in 0..block_b {
        for i in 0..block_a {
            let want = src[(i as isize + j as isize * lda - src_lo) as usize];
            let got = dst[(j as isize + i as isize * ldb - dst_lo) as usize];
            assert_eq!(
                got, want,
                "{label}: lda={lda} ldb={ldb} tile {block_a}x{block_b} at i={i}, j={j}"
            );
        }
    }
}

/// The three branches and the shapes around them: an aliasing `ldb`
/// (4096-byte rows), a near-miss `ldb` that spreads over two sets, both block
/// edge cases, tiles both smaller and larger than the 16-element macro block,
/// and both `lda` classes so the shrink branch and the full-run branch are each
/// exercised. `u64` covers the 8-byte store of the #466 shapes; `[u64; 2]` the
/// 16-byte one. Which branch was *taken* is asserted separately, by
/// [`each_branch_walks_the_tile_it_was_picked_for`]; this test only pins the
/// values. A zero `ldb` is not reachable for a non-singleton destination
/// dimension: `copy_into` rejects a non-injective destination before planning
/// (`copy.rs`).
#[test]
fn fallback_transposes_every_aliasing_and_non_aliasing_tile() {
    let tiles = [
        (1usize, 1usize),
        (1, 17),
        (17, 1),
        (3, 5),
        (16, 16),
        (15, 17),
    ];
    for (block_a, block_b) in tiles {
        // `lda` aliasing selects the shrink branch, `19` the full-run branch.
        for lda in [19isize, 512, 1024] {
            for ldb in [512isize, 513, 256] {
                check_fallback_tile::<u64>("u64", lda, block_a, ldb, block_b, |i| i as u64);
            }
            // Negative strides keep the same set of sets; the orders must agree.
            check_fallback_tile::<u64>("u64 negative", -lda, block_a, -512, block_b, |i| i as u64);
            // 16-byte elements: `ldb = 256` is a 4096-byte row and aliases.
            for ldb in [256isize, 257] {
                check_fallback_tile::<[u64; 2]>("[u64; 2]", lda, block_a, ldb, block_b, |i| {
                    [i as u64, i as u64 + 1]
                });
            }
        }
    }
}

/// The order predicate: a row that is a multiple of the 4 KiB way aliases; a
/// stride of 0 and anything else does not. A row of 256 `u64` elements (16
/// bytes each) is 4096 bytes, which is the aliasing case of #466.
#[test]
fn only_a_multiple_of_the_l1_way_aliases() {
    for (ldb, elem_size, expected) in [
        (512isize, 8usize, true), // 4096 bytes
        (1024, 8, true),          // 8192
        (9216, 8, true),          // [96,96,96], 73728 bytes
        (513, 8, false),          // 4104
        (511, 8, false),
        (1, 8, false),
        (0, 8, false),
        (256, 16, true), // 1024 * 16 in [32,1024]
        (257, 16, false),
        (-512, 8, true), // a negative stride aliases the same sets
        (4096, 1, true),
        (4095, 1, false),
    ] {
        assert_eq!(
            aliases_l1_sets(ldb, elem_size),
            expected,
            "ldb={ldb} elem={elem_size}"
        );
    }
}
