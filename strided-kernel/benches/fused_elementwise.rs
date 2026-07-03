use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use strided_kernel::{
    fused_elementwise_into, map_into, zip_map2_into, FusedInst, FusedOp, FusedPlan, StridedArray,
};

const N: usize = 512;
const DIMS: [usize; 2] = [N, N];

fn make_input(seed: f64) -> StridedArray<f64> {
    StridedArray::<f64>::from_fn_col_major(&DIMS, |idx| {
        seed + 0.001 * idx[0] as f64 + 0.000_01 * idx[1] as f64
    })
}

fn make_constant(value: f64) -> StridedArray<f64> {
    StridedArray::<f64>::from_fn_col_major(&DIMS, |_| value)
}

fn bench_add_mul(c: &mut Criterion) {
    let a = make_input(1.0);
    let b = make_input(2.0);
    let mut tmp = StridedArray::<f64>::col_major(&DIMS);
    let mut out = StridedArray::<f64>::col_major(&DIMS);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![3],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0],
            },
        ],
    };

    let mut group = c.benchmark_group("fused_add_mul");
    group.bench_function("per_op_reused_buffers", |bch| {
        bch.iter(|| {
            zip_map2_into(&mut tmp.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
            zip_map2_into(&mut out.view_mut(), &tmp.view(), &a.view(), |x, y| x * y).unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.bench_function("fused_static", |bch| {
        bch.iter(|| {
            fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.finish();
}

fn bench_broadcast_exp_mul_add(c: &mut Criterion) {
    let a = make_input(0.25);
    let b = make_input(0.5);
    let c_scalar = StridedArray::<f64>::from_parts(vec![0.125], &[1, 1], &[1, 1], 0).unwrap();
    let c_view = c_scalar.view();
    let c_broadcast = c_view.broadcast(&DIMS).unwrap();
    let mut tmp_mul = StridedArray::<f64>::col_major(&DIMS);
    let mut tmp_add = StridedArray::<f64>::col_major(&DIMS);
    let mut out = StridedArray::<f64>::col_major(&DIMS);
    let plan = FusedPlan {
        input_count: 3,
        outputs: vec![5],
        ops: vec![
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![3, 2],
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![4],
            },
        ],
    };

    let mut group = c.benchmark_group("fused_broadcast_exp_mul_add");
    group.bench_function("per_op_reused_buffers", |bch| {
        bch.iter(|| {
            zip_map2_into(&mut tmp_mul.view_mut(), &a.view(), &b.view(), |x, y| x * y).unwrap();
            zip_map2_into(
                &mut tmp_add.view_mut(),
                &tmp_mul.view(),
                &c_broadcast,
                |x, y| x + y,
            )
            .unwrap();
            map_into(&mut out.view_mut(), &tmp_add.view(), |x| x.exp()).unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.bench_function("fused_static", |bch| {
        bch.iter(|| {
            fused_elementwise_into(
                &mut [out.view_mut()],
                &[a.view(), b.view(), c_broadcast.clone()],
                &plan,
            )
            .unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.finish();
}

fn bench_long_chain(c: &mut Criterion) {
    let a = make_input(2.0);
    let b = make_input(1.0);
    let lo = make_constant(0.25);
    let hi = make_constant(4.0);
    let mut tmp_div = StridedArray::<f64>::col_major(&DIMS);
    let mut tmp_max = StridedArray::<f64>::col_major(&DIMS);
    let mut tmp_min = StridedArray::<f64>::col_major(&DIMS);
    let mut tmp_sqrt = StridedArray::<f64>::col_major(&DIMS);
    let mut out = StridedArray::<f64>::col_major(&DIMS);
    let plan = FusedPlan {
        input_count: 4,
        outputs: vec![8],
        ops: vec![
            FusedInst {
                op: FusedOp::Divide,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Maximum,
                inputs: vec![4, 2],
            },
            FusedInst {
                op: FusedOp::Minimum,
                inputs: vec![5, 3],
            },
            FusedInst {
                op: FusedOp::Sqrt,
                inputs: vec![6],
            },
            FusedInst {
                op: FusedOp::Rsqrt,
                inputs: vec![7],
            },
        ],
    };

    let mut group = c.benchmark_group("fused_long_chain");
    group.bench_function("per_op_reused_buffers", |bch| {
        bch.iter(|| {
            zip_map2_into(&mut tmp_div.view_mut(), &a.view(), &b.view(), |x, y| x / y).unwrap();
            zip_map2_into(
                &mut tmp_max.view_mut(),
                &tmp_div.view(),
                &lo.view(),
                |x, y| x.max(y),
            )
            .unwrap();
            zip_map2_into(
                &mut tmp_min.view_mut(),
                &tmp_max.view(),
                &hi.view(),
                |x, y| x.min(y),
            )
            .unwrap();
            map_into(&mut tmp_sqrt.view_mut(), &tmp_min.view(), |x| x.sqrt()).unwrap();
            map_into(&mut out.view_mut(), &tmp_sqrt.view(), |x| 1.0 / x.sqrt()).unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.bench_function("fused_static", |bch| {
        bch.iter(|| {
            fused_elementwise_into(
                &mut [out.view_mut()],
                &[a.view(), b.view(), lo.view(), hi.view()],
                &plan,
            )
            .unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.finish();
}

fn bench_interpreter_fallback(c: &mut Criterion) {
    let a = make_input(0.25);
    let b = make_input(0.5);
    let mut tmp_add = StridedArray::<f64>::col_major(&DIMS);
    let mut tmp_neg = StridedArray::<f64>::col_major(&DIMS);
    let mut out = StridedArray::<f64>::col_major(&DIMS);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![4],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Negate,
                inputs: vec![2],
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![3],
            },
        ],
    };

    let mut group = c.benchmark_group("fused_interpreter_fallback");
    group.bench_function("per_op_reused_buffers", |bch| {
        bch.iter(|| {
            zip_map2_into(&mut tmp_add.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();
            map_into(&mut tmp_neg.view_mut(), &tmp_add.view(), |x| -x).unwrap();
            map_into(&mut out.view_mut(), &tmp_neg.view(), |x| x.exp()).unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.bench_function("fused_interpreter", |bch| {
        bch.iter(|| {
            fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();
            black_box(out.data().as_ptr());
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_add_mul, bench_broadcast_exp_mul_add, bench_long_chain, bench_interpreter_fallback
}
criterion_main!(benches);
