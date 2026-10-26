//! Placement of the fan-out entries under an `Exec` token: concurrency caps,
//! inline execution for budget one, exactly-once coverage, and pool placement.
#![cfg(feature = "parallel")]

use std::collections::HashMap;
use std::ops::Mul;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

use strided_basic::{
    broadcast_mul_into, copy_into, copy_transpose_scale_into, map_into, mul_into, reduce,
    reduce_axis, zip_map2_into, zip_map3_into, ElementOp, Exec, Identity, Pool, StridedArray,
    StridedView, StridedViewMut,
};
use strided_fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan, FusedScalar};
use threadteam::ledger;

const LARGE_LEN: usize = 1 << 17;
const NON_DIVISIBLE_LEN: usize = (1 << 15) + 65;

#[derive(Default)]
struct Participants {
    active: AtomicUsize,
    max_active: AtomicUsize,
    thread_ids: Mutex<Vec<ThreadId>>,
    thread_names: Mutex<Vec<String>>,
    required_concurrency: usize,
    rendezvous_released: AtomicBool,
    rendezvous_lock: Mutex<()>,
    rendezvous: Condvar,
}

impl Participants {
    fn requiring(required_concurrency: usize) -> Self {
        assert!(required_concurrency >= 2);
        Self {
            required_concurrency,
            ..Self::default()
        }
    }

    fn observe(&self) {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        {
            let id = std::thread::current().id();
            let mut ids = self.thread_ids.lock().unwrap();
            if !ids.contains(&id) {
                ids.push(id);
                let name = std::thread::current().name().unwrap_or("").to_owned();
                self.thread_names.lock().unwrap().push(name);
            }
        }
        if self.required_concurrency >= 2 && !self.rendezvous_released.load(Ordering::Acquire) {
            let guard = self.rendezvous_lock.lock().unwrap();
            if active >= self.required_concurrency {
                self.rendezvous_released.store(true, Ordering::Release);
                self.rendezvous.notify_all();
            } else if !self.rendezvous_released.load(Ordering::Acquire) {
                let _ = self
                    .rendezvous
                    .wait_timeout_while(guard, Duration::from_secs(2), |_| {
                        !self.rendezvous_released.load(Ordering::Acquire)
                    })
                    .unwrap();
            }
        }
        for _ in 0..32 {
            std::hint::spin_loop();
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
    }

    fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }

    fn thread_ids(&self) -> Vec<ThreadId> {
        self.thread_ids.lock().unwrap().clone()
    }

    fn thread_names(&self) -> Vec<String> {
        self.thread_names.lock().unwrap().clone()
    }
}

#[derive(Default)]
struct ActiveThreads {
    depths: Mutex<HashMap<ThreadId, usize>>,
    max_active: AtomicUsize,
}

impl ActiveThreads {
    fn enter(&self) -> ActiveThreadGuard<'_> {
        let thread_id = std::thread::current().id();
        let active = {
            let mut depths = self.depths.lock().unwrap();
            *depths.entry(thread_id).or_default() += 1;
            depths.len()
        };
        self.max_active.fetch_max(active, Ordering::SeqCst);
        ActiveThreadGuard {
            threads: self,
            thread_id,
        }
    }

    fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }
}

struct ActiveThreadGuard<'a> {
    threads: &'a ActiveThreads,
    thread_id: ThreadId,
}

impl Drop for ActiveThreadGuard<'_> {
    fn drop(&mut self) {
        let mut depths = self.threads.depths.lock().unwrap();
        let depth = depths.get_mut(&self.thread_id).unwrap();
        *depth -= 1;
        if *depth == 0 {
            depths.remove(&self.thread_id);
        }
    }
}

#[derive(Clone, Copy)]
struct TrackedValue {
    value: f64,
    participants: &'static Participants,
}

impl TrackedValue {
    fn observed(self, value: f64) -> Self {
        self.participants.observe();
        Self {
            value,
            participants: self.participants,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct TrackingElementOp;

impl ElementOp<TrackedValue> for TrackingElementOp {
    fn apply(value: TrackedValue) -> TrackedValue {
        value.observed(value.value)
    }
}

impl FusedScalar for TrackedValue {
    fn fused_add(self, rhs: Self) -> Self {
        self.observed(self.value + rhs.value)
    }

    fn fused_multiply(self, rhs: Self) -> Self {
        self.observed(self.value * rhs.value)
    }

    fn fused_negate(self) -> Self {
        self.observed(-self.value)
    }

    fn fused_conj(self) -> Self {
        self.observed(self.value)
    }

    fn fused_divide(self, rhs: Self) -> Self {
        self.observed(self.value / rhs.value)
    }

    fn fused_abs(self) -> Self {
        self.observed(self.value.abs())
    }

    fn fused_maximum(self, rhs: Self) -> Self {
        self.observed(self.value.max(rhs.value))
    }

    fn fused_minimum(self, rhs: Self) -> Self {
        self.observed(self.value.min(rhs.value))
    }

    fn fused_clamp(self, min: Self, max: Self) -> Self {
        self.observed(self.value.clamp(min.value, max.value))
    }

    fn fused_exp(self) -> Self {
        self.observed(self.value.exp())
    }

    fn fused_log(self) -> Self {
        self.observed(self.value.ln())
    }

    fn fused_sin(self) -> Self {
        self.observed(self.value.sin())
    }

    fn fused_cos(self) -> Self {
        self.observed(self.value.cos())
    }

    fn fused_tanh(self) -> Self {
        self.observed(self.value.tanh())
    }

    fn fused_sqrt(self) -> Self {
        self.observed(self.value.sqrt())
    }

    fn fused_rsqrt(self) -> Self {
        self.observed(self.value.sqrt().recip())
    }

    fn fused_pow(self, rhs: Self) -> Self {
        self.observed(self.value.powf(rhs.value))
    }

    fn fused_expm1(self) -> Self {
        self.observed(self.value.exp_m1())
    }

    fn fused_log1p(self) -> Self {
        self.observed(self.value.ln_1p())
    }
}

fn leaked_participants(required_concurrency: usize) -> &'static Participants {
    let participants = if required_concurrency >= 2 {
        Participants::requiring(required_concurrency)
    } else {
        Participants::default()
    };
    Box::leak(Box::new(participants))
}

fn exact_coverage(len: usize) -> Arc<[AtomicUsize]> {
    (0..len)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>()
        .into()
}

fn assert_exactly_once(coverage: &[AtomicUsize]) {
    for (index, count) in coverage.iter().enumerate() {
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "logical index {index} did not execute exactly once"
        );
    }
}

struct TrackedMulState {
    participants: Participants,
    coverage: Box<[AtomicUsize]>,
}

fn leaked_mul_state(len: usize, required_concurrency: usize) -> &'static TrackedMulState {
    let participants = if required_concurrency >= 2 {
        Participants::requiring(required_concurrency)
    } else {
        Participants::default()
    };
    let coverage = (0..len)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Box::leak(Box::new(TrackedMulState {
        participants,
        coverage,
    }))
}

#[derive(Clone, Copy)]
struct TrackedMulLhs {
    index_component: usize,
    value: usize,
    state: &'static TrackedMulState,
}

#[derive(Clone, Copy)]
struct TrackedMulRhs {
    index_component: usize,
    value: usize,
}

impl Mul<TrackedMulRhs> for TrackedMulLhs {
    type Output = usize;

    fn mul(self, rhs: TrackedMulRhs) -> Self::Output {
        let index = self.index_component + rhs.index_component;
        self.state.coverage[index].fetch_add(1, Ordering::SeqCst);
        self.state.participants.observe();
        self.value * rhs.value
    }
}

const POOL_PREFIX: &str = "placement-test-";

/// A four-worker pool whose workers carry a recognisable name.
fn named_pool() -> Pool {
    Pool::builder()
        .dispatch_costs(cheap_dispatch_costs())
        .threads(4)
        .thread_name(|index| format!("{POOL_PREFIX}{index}"))
        .build()
        .unwrap()
}

/// A token for `budget` slots of `pool` and the ledger root that counts its tasks.
fn pooled(pool: &Pool, budget: usize) -> (Exec<'_>, u64) {
    let exec = Exec::new(pool, budget).unwrap();
    let root = exec.ledger_root().unwrap();
    (exec, root)
}

/// No more tasks than slots ever ran at once under `root`, and none leaked.
fn assert_within_budget(root: u64, budget: usize) {
    assert!(ledger::max_live(root) <= budget);
    assert_eq!(ledger::live(root), 0);
}

/// Every observed thread is a worker of the named pool, never the caller.
fn assert_on_pool_workers(participants: &Participants) {
    let caller = std::thread::current().id();
    assert!(!participants.thread_ids().contains(&caller));
    for name in participants.thread_names() {
        assert!(name.starts_with(POOL_PREFIX), "ran on thread {name:?}");
    }
}

fn run_large_map(exec: Exec<'_>, required_concurrency: usize) -> (Participants, Vec<f64>) {
    let source = StridedArray::<f64>::from_fn_col_major(&[LARGE_LEN], |index| index[0] as f64);
    let mut destination = StridedArray::<f64>::col_major(&[LARGE_LEN]);
    let participants = Arc::new(if required_concurrency >= 2 {
        Participants::requiring(required_concurrency)
    } else {
        Participants::default()
    });
    let observed = Arc::clone(&participants);

    map_into(
        &mut destination.view_mut(),
        &source.view(),
        |value| {
            observed.observe();
            value + 1.0
        },
        exec,
    )
    .unwrap();

    drop(observed);
    (
        Arc::into_inner(participants).unwrap(),
        destination.into_data(),
    )
}

fn run_large_reduce(exec: Exec<'_>, required_concurrency: usize) -> (Participants, usize) {
    let source = StridedArray::<u8>::from_fn_col_major(&[LARGE_LEN], |_| 1);
    let participants = Arc::new(if required_concurrency >= 2 {
        Participants::requiring(required_concurrency)
    } else {
        Participants::default()
    });
    let observed = Arc::clone(&participants);

    let result = reduce(
        &source.view(),
        |value| {
            observed.observe();
            value as usize
        },
        |left, right| left + right,
        0usize,
        exec,
    )
    .unwrap();

    drop(observed);
    (Arc::into_inner(participants).unwrap(), result)
}

fn run_large_transposed_copy(
    exec: Exec<'_>,
    required_concurrency: usize,
) -> (&'static Participants, Vec<TrackedValue>) {
    let participants = leaked_participants(required_concurrency);
    let dims = [512usize, 256];
    let source_strides = [256isize, 1];
    let destination_strides = [1isize, 512];
    let source: Vec<_> = (0..LARGE_LEN)
        .map(|index| TrackedValue {
            value: index as f64,
            participants,
        })
        .collect();
    let mut destination = vec![
        TrackedValue {
            value: -1.0,
            participants,
        };
        LARGE_LEN
    ];
    let source =
        StridedView::<TrackedValue, TrackingElementOp>::new(&source, &dims, &source_strides, 0)
            .unwrap();
    let mut destination_view =
        StridedViewMut::new(&mut destination, &dims, &destination_strides, 0).unwrap();

    copy_into(&mut destination_view, &source, exec).unwrap();
    (participants, destination)
}

fn run_large_fused_add(
    exec: Exec<'_>,
    required_concurrency: usize,
) -> (&'static Participants, Vec<TrackedValue>) {
    let participants = leaked_participants(required_concurrency);
    let dims = [LARGE_LEN];
    let strides = [1isize];
    let lhs: Vec<_> = (0..LARGE_LEN)
        .map(|index| TrackedValue {
            value: index as f64,
            participants,
        })
        .collect();
    let rhs = vec![
        TrackedValue {
            value: 2.0,
            participants,
        };
        LARGE_LEN
    ];
    let mut destination = vec![
        TrackedValue {
            value: -1.0,
            participants,
        };
        LARGE_LEN
    ];
    let lhs = StridedView::new(&lhs, &dims, &strides, 0).unwrap();
    let rhs = StridedView::new(&rhs, &dims, &strides, 0).unwrap();
    let destination_view = StridedViewMut::new(&mut destination, &dims, &strides, 0).unwrap();
    let mut destinations = [destination_view];
    let inputs = [lhs, rhs];
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    fused_elementwise_into(&mut destinations, &inputs, &plan, exec).unwrap();
    (participants, destination)
}

fn assert_row_major_identity_permutation_copy(pool_threads: usize, budget: usize) {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    let source = StridedArray::<f64>::from_fn_row_major(&[ROWS, COLS], |index| {
        (index[0] * COLS + index[1]) as f64
    });
    let mut destination = StridedArray::<f64>::row_major(&[COLS, ROWS]);
    let pool = Pool::builder().threads(pool_threads).build().unwrap();
    let (exec, root) = pooled(&pool, budget);

    copy_transpose_scale_into(&mut destination.view_mut(), &source.view(), 1.0, exec).unwrap();

    assert_within_budget(root, budget.min(pool_threads));
    for row in 0..ROWS {
        for column in 0..COLS {
            assert_eq!(destination.get(&[column, row]), source.get(&[row, column]));
        }
    }
}

#[test]
fn identity_permutation_copy_is_correct_for_parallel_eligible_and_serial_fallback() {
    assert_row_major_identity_permutation_copy(2, 2);
    assert_row_major_identity_permutation_copy(4, 2);
}

#[test]
fn serial_map_stays_on_the_calling_thread() {
    let caller = std::thread::current().id();
    let (participants, output) = run_large_map(Exec::serial(), 1);

    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_eq!(output[0], 1.0);
    assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
}

#[test]
fn pooled_map_caps_concurrent_participants_inside_a_larger_pool() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 2);
    let (participants, output) = run_large_map(exec, 2);

    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_ids().len(), 2);
    assert_on_pool_workers(&participants);
    assert_within_budget(root, 2);
    assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
}

#[test]
fn budget_one_uses_no_fanout_and_stays_on_the_calling_thread() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 1);
    let caller = std::thread::current().id();
    let (participants, output) = run_large_map(exec, 1);

    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_eq!(ledger::installs(root), 0);
    assert_within_budget(root, 1);
    assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
}

#[test]
fn bounded_zip2_and_zip3_cover_non_divisible_inputs_exactly_once() {
    let pool = named_pool();
    let dims = [NON_DIVISIBLE_LEN];
    let reverse_strides = [-1isize];
    let forward_strides = [1isize];
    let a_data: Vec<_> = (0..NON_DIVISIBLE_LEN).collect();
    let b_data: Vec<_> = (0..NON_DIVISIBLE_LEN).map(|index| index * 3).collect();
    let c_data: Vec<_> = (0..NON_DIVISIBLE_LEN).map(|index| index * 5).collect();
    let a = StridedView::<usize, Identity>::new(
        &a_data,
        &dims,
        &reverse_strides,
        (NON_DIVISIBLE_LEN - 1) as isize,
    )
    .unwrap();
    let b = StridedView::<usize, Identity>::new(&b_data, &dims, &forward_strides, 0).unwrap();
    let c = StridedView::<usize, Identity>::new(&c_data, &dims, &forward_strides, 0).unwrap();

    let mut zip2_destination = StridedArray::<usize>::col_major(&dims);
    let zip2_coverage = exact_coverage(NON_DIVISIBLE_LEN);
    let zip2_covered = Arc::clone(&zip2_coverage);
    let zip2_participants = Arc::new(Participants::requiring(2));
    let zip2_observed = Arc::clone(&zip2_participants);
    let (exec, root) = pooled(&pool, 2);
    zip_map2_into(
        &mut zip2_destination.view_mut(),
        &a,
        &b,
        |a_index, b_value| {
            zip2_covered[a_index].fetch_add(1, Ordering::SeqCst);
            zip2_observed.observe();
            a_index + b_value
        },
        exec,
    )
    .unwrap();
    assert_eq!(zip2_participants.max_active(), 2);
    assert_eq!(zip2_participants.thread_ids().len(), 2);
    assert_within_budget(root, 2);
    assert_exactly_once(&zip2_coverage);
    assert_eq!(zip2_destination.get(&[0]), NON_DIVISIBLE_LEN - 1);
    assert_eq!(
        zip2_destination.get(&[NON_DIVISIBLE_LEN - 1]),
        3 * (NON_DIVISIBLE_LEN - 1)
    );

    let mut zip3_destination = StridedArray::<usize>::col_major(&dims);
    let zip3_coverage = exact_coverage(NON_DIVISIBLE_LEN);
    let zip3_covered = Arc::clone(&zip3_coverage);
    let zip3_participants = Arc::new(Participants::requiring(2));
    let zip3_observed = Arc::clone(&zip3_participants);
    let (exec, root) = pooled(&pool, 2);
    zip_map3_into(
        &mut zip3_destination.view_mut(),
        &a,
        &b,
        &c,
        |a_index, b_value, c_value| {
            zip3_covered[a_index].fetch_add(1, Ordering::SeqCst);
            zip3_observed.observe();
            a_index + b_value + c_value
        },
        exec,
    )
    .unwrap();
    assert_eq!(zip3_participants.max_active(), 2);
    assert_eq!(zip3_participants.thread_ids().len(), 2);
    assert_within_budget(root, 2);
    assert_exactly_once(&zip3_coverage);
    assert_eq!(zip3_destination.get(&[0]), NON_DIVISIBLE_LEN - 1);
    assert_eq!(
        zip3_destination.get(&[NON_DIVISIBLE_LEN - 1]),
        8 * (NON_DIVISIBLE_LEN - 1)
    );
}

#[test]
fn bounded_mul_and_broadcast_mul_cover_range_partitions_exactly_once() {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    const LEN: usize = ROWS * COLS;

    let pool = named_pool();

    let mul_state = leaked_mul_state(LEN, 2);
    let lhs_data: Vec<_> = (0..LEN)
        .map(|linear| TrackedMulLhs {
            index_component: linear,
            value: linear + 1,
            state: mul_state,
        })
        .collect();
    let lhs = StridedView::<TrackedMulLhs, Identity>::new(
        &lhs_data,
        &[ROWS, COLS],
        &[1, ROWS as isize],
        0,
    )
    .unwrap();
    let rhs_data = [TrackedMulRhs {
        index_component: 0,
        value: 3,
    }];
    let rhs =
        StridedView::<TrackedMulRhs, Identity>::new(&rhs_data, &[ROWS, COLS], &[0, 0], 0).unwrap();
    let mut destination = StridedArray::<usize>::col_major(&[ROWS, COLS]);
    let (exec, root) = pooled(&pool, 2);
    mul_into(&mut destination.view_mut(), &lhs, &rhs, exec).unwrap();
    assert_eq!(mul_state.participants.max_active(), 2);
    assert_eq!(mul_state.participants.thread_ids().len(), 2);
    assert_within_budget(root, 2);
    assert_exactly_once(&mul_state.coverage);
    assert_eq!(destination.get(&[0, 0]), 3);
    assert_eq!(destination.get(&[ROWS - 1, COLS - 1]), LEN * 3);

    let broadcast_state = leaked_mul_state(LEN, 2);
    let lhs_data: Vec<_> = (0..ROWS)
        .map(|index| TrackedMulLhs {
            index_component: index,
            value: index + 1,
            state: broadcast_state,
        })
        .collect();
    let rhs_data: Vec<_> = (0..COLS)
        .map(|index| TrackedMulRhs {
            index_component: ROWS * index,
            value: index + 2,
        })
        .collect();
    let lhs = StridedView::<TrackedMulLhs, Identity>::new(&lhs_data, &[ROWS], &[1], 0).unwrap();
    let rhs = StridedView::<TrackedMulRhs, Identity>::new(&rhs_data, &[COLS], &[1], 0).unwrap();
    let mut destination = StridedArray::<usize>::col_major(&[ROWS, COLS]);
    let (exec, root) = pooled(&pool, 2);
    broadcast_mul_into(&mut destination.view_mut(), &lhs, &[0], &rhs, &[1], exec).unwrap();
    assert_eq!(broadcast_state.participants.max_active(), 2);
    assert_eq!(broadcast_state.participants.thread_ids().len(), 2);
    assert_within_budget(root, 2);
    assert_exactly_once(&broadcast_state.coverage);
    assert_eq!(destination.get(&[0, 0]), 2);
    assert_eq!(destination.get(&[ROWS - 1, COLS - 1]), ROWS * (COLS + 1));
}

const REDUCE_AXIS_ROWS: usize = 257;
const REDUCE_AXIS_COLS: usize = 129;

fn run_tracked_reduce_axis(
    exec: Exec<'_>,
    required_concurrency: usize,
) -> (Participants, StridedArray<usize>, Arc<[AtomicUsize]>) {
    let source =
        StridedArray::<usize>::from_fn_col_major(&[REDUCE_AXIS_ROWS, REDUCE_AXIS_COLS], |index| {
            index[0] + REDUCE_AXIS_ROWS * index[1]
        });
    let participants = Arc::new(if required_concurrency >= 2 {
        Participants::requiring(required_concurrency)
    } else {
        Participants::default()
    });
    let observed = Arc::clone(&participants);
    let coverage = exact_coverage(REDUCE_AXIS_ROWS * REDUCE_AXIS_COLS);
    let covered = Arc::clone(&coverage);
    let output = reduce_axis(
        &source.view(),
        1,
        |index| {
            covered[index].fetch_add(1, Ordering::SeqCst);
            observed.observe();
            index
        },
        |left, right| left + right,
        0usize,
        exec,
    )
    .unwrap();
    drop(observed);
    (Arc::into_inner(participants).unwrap(), output, coverage)
}

fn assert_reduce_axis_rows(output: &StridedArray<usize>) {
    for row in 0..REDUCE_AXIS_ROWS {
        let expected = REDUCE_AXIS_COLS * row
            + REDUCE_AXIS_ROWS * REDUCE_AXIS_COLS * (REDUCE_AXIS_COLS - 1) / 2;
        assert_eq!(output.get(&[row]), expected);
    }
}

/// Serial `reduce_axis` never fans out: it stays on the calling thread and
/// reads every source element exactly once.
#[test]
fn serial_reduce_axis_stays_on_the_calling_thread() {
    let caller = std::thread::current().id();
    let (participants, output, coverage) = run_tracked_reduce_axis(Exec::serial(), 1);

    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_exactly_once(&coverage);
    assert_reduce_axis_rows(&output);
}

/// Above the threading threshold `reduce_axis` partitions outputs across at
/// most `budget` workers. Each output is computed by one worker, so every
/// source element is read exactly once and the result does not depend on the
/// thread count.
#[test]
fn pooled_reduce_axis_respects_the_thread_budget() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 2);
    let (participants, output, coverage) = run_tracked_reduce_axis(exec, 2);

    assert_eq!(participants.max_active(), 2);
    assert!(participants.thread_ids().len() <= 2);
    assert_within_budget(root, 2);
    assert_exactly_once(&coverage);
    assert_reduce_axis_rows(&output);
}

#[test]
fn bounded_tiled_transpose_scale_handles_non_tile_remainders() {
    const ROWS: usize = 257;
    const COLS: usize = 129;

    let pool = named_pool();
    let source = StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] + ROWS * index[1]) as f64
    });
    let mut destination = StridedArray::<f64>::col_major(&[COLS, ROWS]);
    let (exec, root) = pooled(&pool, 2);
    copy_transpose_scale_into(&mut destination.view_mut(), &source.view(), 2.0, exec).unwrap();
    assert_within_budget(root, 2);

    for i in 0..ROWS {
        for j in 0..COLS {
            assert_eq!(destination.get(&[j, i]), 2.0 * (i + ROWS * j) as f64);
        }
    }
}

/// A user callback receives no token, so a nested strided operation inside it
/// must be given one. Giving it `Exec::serial()` keeps the whole computation
/// within the outer budget: at most two threads ever run user code.
#[test]
fn nested_strided_operations_given_a_serial_token_stay_within_the_outer_budget() {
    const NESTED_LEN: usize = (1 << 16) + 37;

    let pool = named_pool();
    let active_threads = Arc::new(ActiveThreads::default());
    let outer_rendezvous = Arc::new(Barrier::new(2));
    let nested_waited = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
    let coverage: Arc<[AtomicUsize]> = (0..2 * NESTED_LEN)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>()
        .into();

    let source = StridedArray::<usize>::from_fn_col_major(&[LARGE_LEN], |index| index[0]);
    let mut destination = StridedArray::<usize>::col_major(&[LARGE_LEN]);
    let (exec, root) = pooled(&pool, 2);
    map_into(
        &mut destination.view_mut(),
        &source.view(),
        |value| {
            let nested_operation = if value == 0 {
                Some(0)
            } else if value == LARGE_LEN / 2 {
                Some(1)
            } else {
                None
            };
            if let Some(nested_operation) = nested_operation {
                let _outer_guard = active_threads.enter();
                outer_rendezvous.wait();
                let nested_source =
                    StridedArray::<usize>::from_fn_col_major(&[NESTED_LEN], |index| index[0]);
                let mut nested_destination = StridedArray::<usize>::col_major(&[NESTED_LEN]);
                map_into(
                    &mut nested_destination.view_mut(),
                    &nested_source.view(),
                    |nested_index: usize| {
                        let _nested_guard = active_threads.enter();
                        coverage[nested_operation * NESTED_LEN + nested_index]
                            .fetch_add(1, Ordering::SeqCst);
                        if !nested_waited[nested_operation].swap(true, Ordering::SeqCst) {
                            let deadline = std::time::Instant::now() + Duration::from_millis(200);
                            while active_threads.max_active() < 3
                                && std::time::Instant::now() < deadline
                            {
                                std::hint::spin_loop();
                            }
                        }
                        nested_index
                    },
                    Exec::serial(),
                )
                .unwrap();
                assert_eq!(nested_destination.get(&[NESTED_LEN - 1]), NESTED_LEN - 1);
            }
            value
        },
        exec,
    )
    .unwrap();
    assert_eq!(destination.get(&[LARGE_LEN - 1]), LARGE_LEN - 1);

    assert_eq!(
        active_threads.max_active(),
        2,
        "nested strided work exceeded the outer operation's aggregate budget"
    );
    assert_within_budget(root, 2);
    for (index, count) in coverage.iter().enumerate() {
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "nested logical index {index} did not execute exactly once"
        );
    }
}

/// A panic in a worker leaf reaches the caller, leaks no task slot, and leaves
/// the pool usable for the next token.
#[test]
fn a_panicking_worker_leaf_leaks_no_slot_and_leaves_the_pool_usable() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 2);

    let source_data: Vec<_> = (0..NON_DIVISIBLE_LEN).collect();
    let source = StridedView::<usize, Identity>::new(
        &source_data,
        &[NON_DIVISIBLE_LEN],
        &[-1],
        (NON_DIVISIBLE_LEN - 1) as isize,
    )
    .unwrap();
    let mut destination = StridedArray::<usize>::col_major(&[NON_DIVISIBLE_LEN]);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        map_into(
            &mut destination.view_mut(),
            &source,
            |value| {
                assert_ne!(value, 0, "worker leaf panic test");
                value
            },
            exec,
        )
        .unwrap();
    }));
    assert!(panic.is_err());
    assert_eq!(ledger::live(root), 0);

    let (exec, root) = pooled(&pool, 2);
    let (participants, output) = run_large_map(exec, 2);
    assert!(participants.max_active() > 1);
    assert!(participants.thread_ids().len() > 1);
    assert_within_budget(root, 2);
    assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
}

#[test]
fn serial_reduction_stays_on_the_calling_thread() {
    let caller = std::thread::current().id();
    let (participants, result) = run_large_reduce(Exec::serial(), 1);
    assert_eq!(result, LARGE_LEN);
    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_ids(), vec![caller]);
}

#[test]
fn pooled_reduction_caps_concurrent_participants() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 2);
    let (participants, result) = run_large_reduce(exec, 2);
    assert_eq!(result, LARGE_LEN);
    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_ids().len(), 2);
    assert_within_budget(root, 2);
}

#[test]
fn serial_transposed_copy_stays_on_the_calling_thread() {
    let caller = std::thread::current().id();
    let (participants, output) = run_large_transposed_copy(Exec::serial(), 1);
    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_eq!(output[1 + 2 * 512].value, 258.0);
}

#[test]
fn pooled_transposed_copy_caps_concurrent_participants() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 2);
    let (participants, output) = run_large_transposed_copy(exec, 2);
    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_ids().len(), 2);
    assert_within_budget(root, 2);
    assert_eq!(output[1 + 2 * 512].value, 258.0);
}

#[test]
fn serial_fused_path_stays_on_the_calling_thread() {
    let caller = std::thread::current().id();
    let (participants, output) = run_large_fused_add(Exec::serial(), 1);
    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_eq!(output[LARGE_LEN - 1].value, LARGE_LEN as f64 + 1.0);
}

#[test]
fn pooled_fused_path_caps_concurrent_participants() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 2);
    let (participants, output) = run_large_fused_add(exec, 2);
    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_ids().len(), 2);
    assert_within_budget(root, 2);
    assert_eq!(output[LARGE_LEN - 1].value, LARGE_LEN as f64 + 1.0);
}

/// The fused scheduler runs on the token's pool: the element operations run on
/// that pool's workers and never on the calling thread.
#[test]
fn fused_add_runs_on_pool_workers() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 4);
    let (participants, output) = run_large_fused_add(exec, 2);

    assert!(participants.max_active() >= 2);
    assert!(participants.thread_ids().len() >= 2);
    assert_on_pool_workers(participants);
    assert_within_budget(root, 4);
    assert_eq!(output[LARGE_LEN - 1].value, LARGE_LEN as f64 + 1.0);
}

/// The static unary family states the class of its op: `exp` over a length where a
/// fork pays for a transcendental (3 * 100 ns / 1.38 ns = 218 elements) but not for
/// arithmetic (3000 elements) enters the pool, and `negate` of the same length stays
/// on the calling thread.
#[test]
fn static_unary_exp_is_gated_as_transcendental() {
    const MID_LEN: usize = 1000;
    let pool = named_pool();
    let source = StridedArray::<f64>::from_fn_col_major(&[MID_LEN], |index| index[0] as f64 * 1e-3);
    let run = |op: FusedOp| {
        let mut destination = StridedArray::<f64>::col_major(&[MID_LEN]);
        let (exec, root) = pooled(&pool, 4);
        let plan = FusedPlan {
            input_count: 1,
            outputs: vec![1].into_iter().collect(),
            ops: vec![FusedInst {
                op,
                inputs: vec![0].into_iter().collect(),
            }]
            .into_iter()
            .collect(),
        };
        fused_elementwise_into(&mut [destination.view_mut()], &[source.view()], &plan, exec)
            .unwrap();
        ledger::installs(root)
    };
    assert_eq!(run(FusedOp::Negate), 0);
    assert!(run(FusedOp::Exp) >= 1);
}

/// A large non-contiguous copy that fans out through the map scheduler is
/// placed on the token's pool as well.
#[test]
fn large_transposed_copy_runs_on_pool_workers() {
    let pool = named_pool();
    let (exec, root) = pooled(&pool, 4);
    let (participants, output) = run_large_transposed_copy(exec, 2);

    assert!(participants.max_active() >= 2);
    assert!(participants.thread_ids().len() >= 2);
    assert_on_pool_workers(participants);
    assert_within_budget(root, 4);
    assert_eq!(output[1 + 2 * 512].value, 258.0);
}

/// Dispatch costs of the test pools: 100 ns per section whatever the token, so that the
/// regions of tens of thousands of elements these tests use open the parallel branches
/// (the host default costs would keep them serial).
fn cheap_dispatch_costs() -> threadteam::DispatchCosts {
    use threadteam::{DispatchKind, WidthCosts};
    let cheap = WidthCosts::flat(std::time::Duration::from_nanos(100));
    threadteam::DispatchCosts::generic()
        .with(DispatchKind::Team, cheap)
        .with(DispatchKind::InPool, cheap)
        .with(DispatchKind::Outside, cheap)
}
