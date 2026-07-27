#![cfg(feature = "parallel")]

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::ops::Mul;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

use strided_kernel::{
    broadcast_mul_into, copy_into, copy_transpose_scale_into, fused_elementwise_into, map_into,
    mul_into, reduce, reduce_axis, with_execution_policy, zip_map2_into, zip_map3_into, ElementOp,
    ExecutionPolicy, FusedInst, FusedOp, FusedPlan, FusedScalar, Identity, StridedArray,
    StridedView, StridedViewMut,
};

const LARGE_LEN: usize = 1 << 17;
const NON_DIVISIBLE_LEN: usize = (1 << 15) + 65;

#[derive(Default)]
struct Participants {
    active: AtomicUsize,
    max_active: AtomicUsize,
    thread_ids: Mutex<Vec<ThreadId>>,
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

fn run_large_map(policy: ExecutionPolicy, required_concurrency: usize) -> (Participants, Vec<f64>) {
    let source = StridedArray::<f64>::from_fn_col_major(&[LARGE_LEN], |index| index[0] as f64);
    let mut destination = StridedArray::<f64>::col_major(&[LARGE_LEN]);
    let participants = Arc::new(if required_concurrency >= 2 {
        Participants::requiring(required_concurrency)
    } else {
        Participants::default()
    });
    let observed = Arc::clone(&participants);

    with_execution_policy(policy, || {
        map_into(&mut destination.view_mut(), &source.view(), |value| {
            observed.observe();
            value + 1.0
        })
        .unwrap();
    });

    drop(observed);
    (
        Arc::into_inner(participants).unwrap(),
        destination.into_data(),
    )
}

fn run_large_reduce(policy: ExecutionPolicy, required_concurrency: usize) -> (Participants, usize) {
    let source = StridedArray::<u8>::from_fn_col_major(&[LARGE_LEN], |_| 1);
    let participants = Arc::new(if required_concurrency >= 2 {
        Participants::requiring(required_concurrency)
    } else {
        Participants::default()
    });
    let observed = Arc::clone(&participants);

    let result = with_execution_policy(policy, || {
        reduce(
            &source.view(),
            |value| {
                observed.observe();
                value as usize
            },
            |left, right| left + right,
            0usize,
        )
        .unwrap()
    });

    drop(observed);
    (Arc::into_inner(participants).unwrap(), result)
}

fn run_large_transposed_copy(
    policy: ExecutionPolicy,
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

    with_execution_policy(policy, || {
        copy_into(&mut destination_view, &source).unwrap()
    });
    (participants, destination)
}

fn run_large_fused_add(
    policy: ExecutionPolicy,
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
        outputs: vec![2],
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1],
        }],
    };

    with_execution_policy(policy, || {
        fused_elementwise_into(&mut destinations, &inputs, &plan).unwrap()
    });
    (participants, destination)
}

fn assert_row_major_identity_permutation_copy(pool_threads: usize, policy: ExecutionPolicy) {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    let source = StridedArray::<f64>::from_fn_row_major(&[ROWS, COLS], |index| {
        (index[0] * COLS + index[1]) as f64
    });
    let mut destination = StridedArray::<f64>::row_major(&[COLS, ROWS]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .build()
        .unwrap();

    pool.install(|| {
        with_execution_policy(policy, || {
            copy_transpose_scale_into(&mut destination.view_mut(), &source.view(), 1.0).unwrap();
        });
    });

    for row in 0..ROWS {
        for column in 0..COLS {
            assert_eq!(destination.get(&[column, row]), source.get(&[row, column]));
        }
    }
}

#[test]
fn explicit_execution_policy_scope_returns_the_operation_result() {
    let value = with_execution_policy(ExecutionPolicy::Sequential, || 42usize);
    assert_eq!(value, 42);

    let two = NonZeroUsize::new(2).unwrap();
    let value = with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || 7usize);
    assert_eq!(value, 7);
}

#[test]
fn identity_permutation_copy_is_correct_for_parallel_eligible_and_serial_fallback() {
    let two = NonZeroUsize::new(2).unwrap();
    assert_row_major_identity_permutation_copy(2, ExecutionPolicy::Rayon { max_threads: two });
    assert_row_major_identity_permutation_copy(4, ExecutionPolicy::Rayon { max_threads: two });
}

#[test]
fn sequential_map_stays_on_the_calling_thread_inside_a_large_rayon_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let caller = std::thread::current().id();
        let (participants, output) = run_large_map(ExecutionPolicy::Sequential, 1);

        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids(), vec![caller]);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
    });
}

#[test]
fn rayon_map_caps_concurrent_participants_inside_a_larger_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();

    pool.install(|| {
        let (participants, output) = run_large_map(ExecutionPolicy::Rayon { max_threads: two }, 2);

        assert_eq!(participants.max_active(), 2);
        assert_eq!(participants.thread_ids().len(), 2);
        assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
    });
}

#[test]
fn rayon_budget_one_uses_no_fanout_and_stays_on_the_calling_thread() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let one = NonZeroUsize::new(1).unwrap();

    pool.install(|| {
        let caller = std::thread::current().id();
        let (participants, output) = run_large_map(ExecutionPolicy::Rayon { max_threads: one }, 1);
        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids(), vec![caller]);
        assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
    });
}

#[test]
fn sequential_policy_ignores_a_multithreaded_global_rayon_pool() {
    assert!(
        rayon::current_num_threads() > 1,
        "this test requires a multithreaded global Rayon pool"
    );
    let caller = std::thread::current().id();
    let source = StridedArray::<usize>::from_fn_col_major(&[NON_DIVISIBLE_LEN], |index| index[0]);
    let mut destination = StridedArray::<usize>::col_major(&[NON_DIVISIBLE_LEN]);
    let participants = Arc::new(Participants::default());
    let observed = Arc::clone(&participants);
    let coverage = exact_coverage(NON_DIVISIBLE_LEN);
    let covered = Arc::clone(&coverage);

    with_execution_policy(ExecutionPolicy::Sequential, || {
        map_into(&mut destination.view_mut(), &source.view(), |index| {
            covered[index].fetch_add(1, Ordering::SeqCst);
            observed.observe();
            index + 1
        })
        .unwrap();
    });

    assert_eq!(participants.max_active(), 1);
    assert_eq!(participants.thread_ids(), vec![caller]);
    assert_exactly_once(&coverage);
    assert_eq!(destination.get(&[NON_DIVISIBLE_LEN - 1]), NON_DIVISIBLE_LEN);
}

#[test]
fn bounded_zip2_and_zip3_cover_non_divisible_inputs_exactly_once() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();
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

    pool.install(|| {
        let mut zip2_destination = StridedArray::<usize>::col_major(&dims);
        let zip2_coverage = exact_coverage(NON_DIVISIBLE_LEN);
        let zip2_covered = Arc::clone(&zip2_coverage);
        let zip2_participants = Arc::new(Participants::requiring(2));
        let zip2_observed = Arc::clone(&zip2_participants);
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            zip_map2_into(
                &mut zip2_destination.view_mut(),
                &a,
                &b,
                |a_index, b_value| {
                    zip2_covered[a_index].fetch_add(1, Ordering::SeqCst);
                    zip2_observed.observe();
                    a_index + b_value
                },
            )
            .unwrap();
        });
        assert_eq!(zip2_participants.max_active(), 2);
        assert_eq!(zip2_participants.thread_ids().len(), 2);
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
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
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
            )
            .unwrap();
        });
        assert_eq!(zip3_participants.max_active(), 2);
        assert_eq!(zip3_participants.thread_ids().len(), 2);
        assert_exactly_once(&zip3_coverage);
        assert_eq!(zip3_destination.get(&[0]), NON_DIVISIBLE_LEN - 1);
        assert_eq!(
            zip3_destination.get(&[NON_DIVISIBLE_LEN - 1]),
            8 * (NON_DIVISIBLE_LEN - 1)
        );
    });
}

#[test]
fn bounded_mul_and_broadcast_mul_cover_range_partitions_exactly_once() {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    const LEN: usize = ROWS * COLS;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();

    pool.install(|| {
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
        let rhs = StridedView::<TrackedMulRhs, Identity>::new(&rhs_data, &[ROWS, COLS], &[0, 0], 0)
            .unwrap();
        let mut destination = StridedArray::<usize>::col_major(&[ROWS, COLS]);
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            mul_into(&mut destination.view_mut(), &lhs, &rhs).unwrap();
        });
        assert_eq!(mul_state.participants.max_active(), 2);
        assert_eq!(mul_state.participants.thread_ids().len(), 2);
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
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            broadcast_mul_into(&mut destination.view_mut(), &lhs, &[0], &rhs, &[1]).unwrap();
        });
        assert_eq!(broadcast_state.participants.max_active(), 2);
        assert_eq!(broadcast_state.participants.thread_ids().len(), 2);
        assert_exactly_once(&broadcast_state.coverage);
        assert_eq!(destination.get(&[0, 0]), 2);
        assert_eq!(destination.get(&[ROWS - 1, COLS - 1]), ROWS * (COLS + 1));
    });
}

#[test]
fn reduce_axis_remains_caller_threaded_under_all_explicit_policies() {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    const LEN: usize = ROWS * COLS;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();

    pool.install(|| {
        let caller = std::thread::current().id();
        for policy in [
            ExecutionPolicy::Sequential,
            ExecutionPolicy::Rayon { max_threads: two },
        ] {
            let source = StridedArray::<usize>::from_fn_col_major(&[ROWS, COLS], |index| {
                index[0] + ROWS * index[1]
            });
            let participants = Arc::new(Participants::default());
            let observed = Arc::clone(&participants);
            let coverage = exact_coverage(LEN);
            let covered = Arc::clone(&coverage);
            let output = with_execution_policy(policy, || {
                reduce_axis(
                    &source.view(),
                    1,
                    |index| {
                        covered[index].fetch_add(1, Ordering::SeqCst);
                        observed.observe();
                        index
                    },
                    |left, right| left + right,
                    0usize,
                )
                .unwrap()
            });

            assert_eq!(participants.max_active(), 1);
            assert_eq!(participants.thread_ids(), vec![caller]);
            assert_exactly_once(&coverage);
            for row in 0..ROWS {
                let expected = COLS * row + ROWS * COLS * (COLS - 1) / 2;
                assert_eq!(output.get(&[row]), expected);
            }
        }
    });
}

#[test]
fn bounded_tiled_transpose_scale_handles_non_tile_remainders() {
    const ROWS: usize = 257;
    const COLS: usize = 129;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();
    pool.install(|| {
        let source = StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
            (index[0] + ROWS * index[1]) as f64
        });
        let mut destination = StridedArray::<f64>::col_major(&[COLS, ROWS]);
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            copy_transpose_scale_into(&mut destination.view_mut(), &source.view(), 2.0).unwrap();
        });

        for i in 0..ROWS {
            for j in 0..COLS {
                assert_eq!(destination.get(&[j, i]), 2.0 * (i + ROWS * j) as f64);
            }
        }
    });
}

#[test]
fn nested_rayon_operation_is_sequential_inside_an_outer_worker_partition() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();
    let four = NonZeroUsize::new(4).unwrap();

    pool.install(|| {
        let outer_source =
            StridedArray::<f64>::from_fn_col_major(&[LARGE_LEN], |index| index[0] as f64);
        let mut outer_destination = StridedArray::<f64>::col_major(&[LARGE_LEN]);
        let nested_started = AtomicBool::new(false);
        let nested_participants = Arc::new(Participants::default());
        let observed = Arc::clone(&nested_participants);
        let scope_caller = std::thread::current().id();

        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            map_into(
                &mut outer_destination.view_mut(),
                &outer_source.view(),
                |value| {
                    if std::thread::current().id() != scope_caller
                        && !nested_started.swap(true, Ordering::SeqCst)
                    {
                        let source =
                            StridedArray::<f64>::from_fn_col_major(&[LARGE_LEN], |index| {
                                index[0] as f64
                            });
                        let mut destination = StridedArray::<f64>::col_major(&[LARGE_LEN]);
                        with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
                            map_into(
                                &mut destination.view_mut(),
                                &source.view(),
                                |nested_value| {
                                    observed.observe();
                                    nested_value + 2.0
                                },
                            )
                            .unwrap();
                        });
                        assert_eq!(destination.get(&[LARGE_LEN - 1]), LARGE_LEN as f64 + 1.0);
                    }
                    value + 1.0
                },
            )
            .unwrap();
        });

        assert!(nested_started.load(Ordering::SeqCst));
        drop(observed);
        let participants = Arc::into_inner(nested_participants).unwrap();
        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids().len(), 1);
        assert_eq!(outer_destination.get(&[LARGE_LEN - 1]), LARGE_LEN as f64);
    });
}

#[test]
fn nested_sequential_policy_dominates_a_rayon_request() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let four = NonZeroUsize::new(4).unwrap();

    pool.install(|| {
        let caller = std::thread::current().id();
        let (participants, _) = with_execution_policy(ExecutionPolicy::Sequential, || {
            run_large_map(ExecutionPolicy::Rayon { max_threads: four }, 1)
        });
        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids(), vec![caller]);
    });
}

#[test]
fn bounded_outer_partitions_force_nested_strided_operations_to_run_sequentially() {
    const NESTED_LEN: usize = MINTHREADLENGTH_FOR_TEST + 37;
    const MINTHREADLENGTH_FOR_TEST: usize = 1 << 15;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();
    let active_threads = Arc::new(ActiveThreads::default());
    let outer_rendezvous = Arc::new(Barrier::new(2));
    let nested_waited = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
    let coverage: Arc<[AtomicUsize]> = (0..2 * NESTED_LEN)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>()
        .into();

    pool.install(|| {
        let source = StridedArray::<usize>::from_fn_col_major(&[LARGE_LEN], |index| index[0]);
        let mut destination = StridedArray::<usize>::col_major(&[LARGE_LEN]);
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            map_into(&mut destination.view_mut(), &source.view(), |value| {
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
                    with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
                        map_into(
                            &mut nested_destination.view_mut(),
                            &nested_source.view(),
                            |nested_index| {
                                let _nested_guard = active_threads.enter();
                                coverage[nested_operation * NESTED_LEN + nested_index]
                                    .fetch_add(1, Ordering::SeqCst);
                                if !nested_waited[nested_operation].swap(true, Ordering::SeqCst) {
                                    let deadline =
                                        std::time::Instant::now() + Duration::from_millis(200);
                                    while active_threads.max_active() < 3
                                        && std::time::Instant::now() < deadline
                                    {
                                        std::hint::spin_loop();
                                    }
                                }
                                nested_index
                            },
                        )
                        .unwrap();
                    });
                    assert_eq!(nested_destination.get(&[NESTED_LEN - 1]), NESTED_LEN - 1);
                }
                value
            })
            .unwrap();
        });
        assert_eq!(destination.get(&[LARGE_LEN - 1]), LARGE_LEN - 1);
    });

    assert_eq!(
        active_threads.max_active(),
        2,
        "nested strided work exceeded the outer operation's aggregate budget"
    );
    for (index, count) in coverage.iter().enumerate() {
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "nested logical index {index} did not execute exactly once"
        );
    }
}

#[test]
fn a_panicking_scope_restores_the_previous_policy() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let panic = std::panic::catch_unwind(|| {
            with_execution_policy(ExecutionPolicy::Sequential, || panic!("policy scope test"));
        });
        assert!(panic.is_err());

        let (participants, _) = run_large_map(ExecutionPolicy::AmbientRayon, 2);
        assert!(participants.max_active() > 1);
        assert!(participants.thread_ids().len() > 1);
    });
}

#[test]
fn a_panicking_worker_leaf_restores_policy_before_later_ambient_work() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();

    pool.install(|| {
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
            with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
                map_into(&mut destination.view_mut(), &source, |value| {
                    assert_ne!(value, 0, "worker leaf policy restoration test");
                    value
                })
                .unwrap();
            });
        }));
        assert!(panic.is_err());

        let (participants, output) = run_large_map(ExecutionPolicy::AmbientRayon, 2);
        assert!(participants.max_active() > 1);
        assert!(participants.thread_ids().len() > 1);
        assert_eq!(output[LARGE_LEN - 1], LARGE_LEN as f64);
    });
}

#[test]
fn sequential_reduction_stays_on_the_calling_thread() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let caller = std::thread::current().id();
        let (participants, result) = run_large_reduce(ExecutionPolicy::Sequential, 1);
        assert_eq!(result, LARGE_LEN);
        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids(), vec![caller]);
    });
}

#[test]
fn rayon_reduction_caps_concurrent_participants() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();

    pool.install(|| {
        let (participants, result) =
            run_large_reduce(ExecutionPolicy::Rayon { max_threads: two }, 2);
        assert_eq!(result, LARGE_LEN);
        assert_eq!(participants.max_active(), 2);
        assert_eq!(participants.thread_ids().len(), 2);
    });
}

#[test]
fn sequential_transposed_copy_stays_on_the_calling_thread() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let caller = std::thread::current().id();
        let (participants, output) = run_large_transposed_copy(ExecutionPolicy::Sequential, 1);
        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids(), vec![caller]);
        assert_eq!(output[1 + 2 * 512].value, 258.0);
    });
}

#[test]
fn rayon_transposed_copy_caps_concurrent_participants() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();

    pool.install(|| {
        let (participants, output) =
            run_large_transposed_copy(ExecutionPolicy::Rayon { max_threads: two }, 2);
        assert_eq!(participants.max_active(), 2);
        assert_eq!(participants.thread_ids().len(), 2);
        assert_eq!(output[1 + 2 * 512].value, 258.0);
    });
}

#[test]
fn sequential_fused_path_stays_on_the_calling_thread() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let caller = std::thread::current().id();
        let (participants, output) = run_large_fused_add(ExecutionPolicy::Sequential, 1);
        assert_eq!(participants.max_active(), 1);
        assert_eq!(participants.thread_ids(), vec![caller]);
        assert_eq!(output[LARGE_LEN - 1].value, LARGE_LEN as f64 + 1.0);
    });
}

#[test]
fn rayon_fused_path_caps_concurrent_participants() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let two = NonZeroUsize::new(2).unwrap();

    pool.install(|| {
        let (participants, output) =
            run_large_fused_add(ExecutionPolicy::Rayon { max_threads: two }, 2);
        assert_eq!(participants.max_active(), 2);
        assert_eq!(participants.thread_ids().len(), 2);
        assert_eq!(output[LARGE_LEN - 1].value, LARGE_LEN as f64 + 1.0);
    });
}

fn parallel_source_violations(relative: &std::path::Path, source: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let threading_module = std::path::Path::new("threading.rs");
    let approved_central_identifiers = [
        ("rayon::join(", 2usize),
        ("rayon::current_num_threads()", 1),
        ("rayon::ThreadPool>", 1),
        ("rayon::ThreadPoolBuilder::new()", 1),
        ("strided_perm::copy_into_par(", 1),
        ("strided_perm::copy_into(", 1),
        ("strided_perm::copy_into_col_major(", 1),
    ];
    let mut central_counts = vec![0usize; approved_central_identifiers.len()];

    for (line_index, line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let code = line.split("//").next().unwrap_or("");
        if code.trim().is_empty() {
            continue;
        }
        let has_owned_identifier = ["rayon", "strided_perm"].iter().any(|identifier| {
            code.split(|character: char| !character.is_alphanumeric() && character != '_')
                .any(|token| token == *identifier)
        });
        if !has_owned_identifier {
            continue;
        }

        if relative != threading_module {
            violations.push(format!(
                "{}:{line_number}: parallel-runtime identifier exists outside threading.rs",
                relative.display()
            ));
            continue;
        }

        let mut unapproved = code.to_owned();
        for (index, (approved, _)) in approved_central_identifiers.iter().enumerate() {
            let count = unapproved.matches(approved).count();
            central_counts[index] += count;
            unapproved = unapproved.replace(approved, "");
        }
        if ["rayon", "strided_perm"].iter().any(|identifier| {
            unapproved
                .split(|character: char| !character.is_alphanumeric() && character != '_')
                .any(|token| token == *identifier)
        }) {
            violations.push(format!(
                "{}:{line_number}: unapproved central parallel-runtime API",
                relative.display()
            ));
        }
    }

    if relative == threading_module {
        for ((approved, expected), actual) in
            approved_central_identifiers.iter().zip(central_counts)
        {
            if actual != *expected {
                violations.push(format!(
                    "threading.rs: expected {expected} call(s) to {approved}, found {actual}"
                ));
            }
        }
    }

    violations
}

#[test]
fn scanner_rejects_parallel_bypass_mutations() {
    let bypasses = [
        "use rayon::slice::ParallelSliceMut; values.par_sort_unstable();",
        "rayon::in_place_scope(|scope| work(scope));",
        "rayon::yield_now();",
        "use strided_perm as perm; perm::copy_into(dest, src);",
    ];
    for mutation in bypasses {
        assert!(
            !parallel_source_violations(std::path::Path::new("mutated.rs"), mutation).is_empty(),
            "scanner accepted parallel bypass mutation: {mutation}"
        );
    }

    let source_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let threading = std::fs::read_to_string(source_root.join("threading.rs")).unwrap();
    let extra_central_call = format!("{threading}\nfn bypass() {{ rayon::join(|| (), || ()); }}\n");
    assert!(
        !parallel_source_violations(std::path::Path::new("threading.rs"), &extra_central_call)
            .is_empty()
    );
}

#[test]
fn production_parallelism_is_routed_through_the_execution_policy_layer() {
    fn visit(directory: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    let source_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    visit(&source_root, &mut files);

    let mut violations = Vec::new();
    for path in files {
        let source = std::fs::read_to_string(&path).unwrap();
        let relative = path.strip_prefix(&source_root).unwrap();
        violations.extend(parallel_source_violations(relative, &source));
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}
