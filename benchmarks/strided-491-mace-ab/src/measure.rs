//! Wall time, allocation and peak-RSS measurement shared by both sides.

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

/// Allocation counters at one instant.
#[derive(Clone, Copy, Debug, Default)]
pub struct Alloc {
    /// Bytes allocated since the last reset.
    pub bytes: u64,
    /// Allocation calls since the last reset.
    pub calls: u64,
    /// Peak live bytes since the last reset.
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
#[derive(Clone, Debug)]
pub struct Measured {
    pub median_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    pub alloc: Alloc,
    pub samples: usize,
}

/// Time `run` until one sample spans `target`, then take `samples` medians.
pub fn measure<F: FnMut()>(target: Duration, samples: usize, mut run: F) -> Measured {
    // Warm up: at least one call, then until one call is long enough to time.
    let mut inner = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..inner {
            run();
        }
        if start.elapsed() >= target || inner >= 1 << 20 {
            break;
        }
        inner *= 2;
    }
    let mut per_sample = Vec::with_capacity(samples);
    reset_alloc();
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..inner {
            run();
        }
        let elapsed = start.elapsed();
        per_sample.push(elapsed.as_secs_f64() * 1e9 / inner as f64);
    }
    let alloc = alloc_snapshot();
    let mut sorted = per_sample.clone();
    sorted.sort_by(|left, right| left.total_cmp(right));
    Measured {
        median_ns: median(&mut per_sample.clone()),
        min_ns: sorted.first().copied().unwrap_or(0.0),
        max_ns: sorted.last().copied().unwrap_or(0.0),
        alloc: Alloc {
            bytes: alloc.bytes / (samples * inner) as u64,
            calls: alloc.calls / (samples * inner) as u64,
            peak_live: alloc.peak_live,
        },
        samples,
    }
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
