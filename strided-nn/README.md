# strided-nn

Fused neural-network kernels over contiguous strided CPU slices.

The first kernel is axis-0 softmax: per-slice max, shift, `exp`, sum and scale in
one pass over cache-sized groups, with no full-size intermediate. The API is
tensor-free: contiguous slices in and out, an explicit
[`strided_basic::Exec`] token, and caller-provided scratch. Placement and
threading stay with the caller, and `exp` comes from `elem-simd-math`.

```rust
use core::mem::MaybeUninit;
use strided_basic::Exec;
use strided_nn::{scratch_len, softmax_axis0_f32};

// Two columns of three, column-major.
let input = [1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0];
let mut output = [MaybeUninit::<f32>::uninit(); 6];
let mut scratch = vec![0.0f32; scratch_len::<f32>(3, 6, 1)];
softmax_axis0_f32(3, &input, &mut output, &mut scratch, Exec::serial())
.unwrap();
for slot in &mut output {
    slot.write(0.0);
}
```
