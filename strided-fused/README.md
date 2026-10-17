# strided-fused

Runtime-DAG fused elementwise execution. The typed `FusedPlan` and dtype-erased
`ErasedFusedPlan` implementations use `strided-basic`'s shared execution policy
and generic traversal without depending on `strided-kernel`.

```rust
use strided_basic::StridedArray;
use strided_fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan};
let input = StridedArray::<f64>::from_parts(vec![2.0, 3.0], &[2], &[1], 0).unwrap();
let mut output = StridedArray::<f64>::col_major(&[2]);
let plan = FusedPlan::new(
    1,
    vec![1],
    vec![FusedInst::new(FusedOp::Negate, vec![0])],
);
fused_elementwise_into(&mut [output.view_mut()], &[input.view()], &plan).unwrap();
assert_eq!(output.get(&[1]), -3.0);
```

`simd` is enabled by default; `parallel` enables bounded parallel replay and
forwards to the shared foundation. This crate does not create an independent
thread pool. An assembling backend can retain automatic fusion by calling this
crate with its existing context and output storage; users need not select plans
manually when their runtime already performs that planning.

See [kernel boundaries](../docs/design/cpu-kernel-boundaries.md) and the packaged
license notices. This new package has not yet been published.
