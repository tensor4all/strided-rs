//! Wall time, allocation and peak-RSS measurement shared by both sides.
//!
//! Two rules this module exists for:
//!
//! * A per-call number is never an integer division of a total. Totals and the
//!   exact timed-call count are kept, and the per-call figure is the ratio of
//!   the two as an `f64`, so a handful of bytes per call cannot truncate to 0.
//! * The sides of a cell are timed in a round-robin, one timed round per side per
//!   sample, so neither side sits in a different slice of the machine's drift.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A counting allocator: bytes and calls allocated since the last reset.
pub struct Counting;

static BYTES: AtomicU64 = AtomicU64::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
            let live =
                LIVE.fetch_add(layout.size() as u64, Ordering::Relaxed) + layout.size() as u64;
            PEAK_LIVE.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
            if new_size > layout.size() {
                LIVE.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub((layout.size() - new_size) as u64, Ordering::Relaxed);
            }
        }
        new_pointer
    }
}

/// Exact allocation totals over every timed call of one side.
#[derive(Clone, Copy, Debug, Default)]
pub struct Alloc {
    /// Bytes allocated, summed over the timed calls.
    pub bytes: u64,
    /// Allocation calls, summed over the timed calls.
    pub calls: u64,
    /// Peak live bytes inside one timed call.
    pub peak_live: u64,
}

pub fn reset_alloc() {
    BYTES.store(0, Ordering::Relaxed);
    CALLS.store(0, Ordering::Relaxed);
    PEAK_LIVE.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

#[must_use]
pub fn alloc_snapshot() -> Alloc {
    Alloc {
        bytes: BYTES.load(Ordering::Relaxed),
        calls: CALLS.load(Ordering::Relaxed),
        peak_live: PEAK_LIVE.load(Ordering::Relaxed),
    }
}

/// `VmHWM` in bytes, i.e. the process's peak resident set.
#[must_use]
pub fn peak_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(0, |kib| kib * 1024)
}

/// One measured side: wall time per call and the allocations inside it.
#[derive(Clone, Debug, Default)]
pub struct Measured {
    pub median_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    /// Exact totals over `calls` timed calls.
    pub alloc: Alloc,
    /// Timed calls this side ran: `samples * inner`.
    pub calls: u64,
    pub samples: usize,
    /// Calls timed together per sample.
    pub inner: usize,
}

impl Measured {
    /// Bytes per call, unrounded.
    #[must_use]
    pub fn bytes_per_call(&self) -> f64 {
        self.alloc.bytes as f64 / self.calls.max(1) as f64
    }

    /// Allocation calls per call, unrounded.
    #[must_use]
    pub fn alloc_calls_per_call(&self) -> f64 {
        self.alloc.calls as f64 / self.calls.max(1) as f64
    }
}

/// Time every side in a round-robin and return one [`Measured`] per side.
///
/// Each side is warmed until one timed round spans `target`; the sample loop
/// then times side 0, side 1, ... once per sample, so all sides see the same
/// sequence of machine states.
#[must_use]
pub fn interleaved(
    target: Duration,
    samples: usize,
    sides: &mut [&mut dyn FnMut()],
) -> Vec<Measured> {
    let count = sides.len();
    let mut inner = vec![1usize; count];
    for (index, side) in sides.iter_mut().enumerate() {
        loop {
            let start = Instant::now();
            for _ in 0..inner[index] {
                side();
            }
            if start.elapsed() >= target || inner[index] >= 1 << 20 {
                break;
            }
            inner[index] *= 2;
        }
    }

    let mut per_sample = vec![Vec::with_capacity(samples); count];
    let mut totals = vec![Alloc::default(); count];
    for _ in 0..samples {
        for (index, side) in sides.iter_mut().enumerate() {
            reset_alloc();
            let start = Instant::now();
            for _ in 0..inner[index] {
                side();
            }
            let elapsed = start.elapsed();
            let alloc = alloc_snapshot();
            per_sample[index].push(elapsed.as_secs_f64() * 1e9 / inner[index] as f64);
            totals[index].bytes += alloc.bytes;
            totals[index].calls += alloc.calls;
            totals[index].peak_live = totals[index].peak_live.max(alloc.peak_live);
        }
    }

    per_sample
        .into_iter()
        .zip(totals)
        .enumerate()
        .map(|(index, (mut nanoseconds, alloc))| {
            nanoseconds.sort_by(|left, right| left.total_cmp(right));
            Measured {
                median_ns: median(&mut nanoseconds.clone()),
                min_ns: nanoseconds.first().copied().unwrap_or(0.0),
                max_ns: nanoseconds.last().copied().unwrap_or(0.0),
                alloc,
                calls: (samples * inner[index]) as u64,
                samples,
                inner: inner[index],
            }
        })
        .collect()
}

/// Time one side. Thin wrapper over [`interleaved`] for single-side figures.
#[must_use]
pub fn measure(target: Duration, samples: usize, run: &mut dyn FnMut()) -> Measured {
    interleaved(target, samples, &mut [run])
        .pop()
        .expect("one side")
}

#[must_use]
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}
