# Complex Mul Benchmark Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PyTorch-comparable complex multiplication benchmarks to `strided-kernel`, then add complex fast paths only where the benchmark shows a gap.

**Architecture:** Keep the existing `mul_pytorch_compare` benchmark shape matrix and add a dtype dimension (`f64`, `c64`, `c128`) so real and complex numbers are measured under identical layout/threading conditions. Implement same-type complex fast paths in `strided-kernel/src/simd.rs` using `pulp::Simd` complex lane APIs, matching the mechanism faer uses internally.

**Tech Stack:** Rust `strided-kernel`, `num-complex`, optional `pulp` SIMD, Rayon for multi-thread runs, Python/PyTorch reference runner.

---

## Faer Finding

faer does have special SIMD support for complex values.

- `/Users/hiroshi/projects/tensor4all/faer-rs/faer-traits/src/lib.rs:1224` defines `ComplexField` with SIMD hooks such as `type SimdVec<S: Simd>`, `SIMD_CAPABILITIES`, `simd_mul`, `simd_conj_mul`, `simd_mul_add`, and `simd_conj_mul_add`.
- `/Users/hiroshi/projects/tensor4all/faer-rs/faer-traits/src/lib.rs:2538` implements `ComplexField` for generic `num_complex::Complex<T>`. This uses `Complex<T::SimdVec<S>>` and composes real SIMD operations manually.
- `/Users/hiroshi/projects/tensor4all/faer-rs/faer-traits/src/lib.rs:3263` implements `ComplexField` for `ComplexImpl<f32>` with `type SimdVec<S> = S::c32s`, `SIMD_CAPABILITIES = Simd`, and `ctx.mul_e_c32s`.
- `/Users/hiroshi/projects/tensor4all/faer-rs/faer-traits/src/lib.rs:3930` implements `ComplexField` for `ComplexImpl<f64>` with `type SimdVec<S> = S::c64s`, `SIMD_CAPABILITIES = Simd`, and `ctx.mul_e_c64s`.
- `/Users/hiroshi/projects/tensor4all/faer-rs/faer/src/lib.rs:201` dispatches public native complex scalar types through `ComplexImpl<f32>` or `ComplexImpl<f64>` before entering kernels.
- `pulp` exposes the needed APIs directly: `partial_load_c32s`, `partial_load_c64s`, `partial_store_c32s`, `partial_store_c64s`, `mul_e_c32s`, `mul_e_c64s`, `conj_mul_e_c32s`, and `conj_mul_e_c64s`.

Decision: `strided-kernel` should not copy faer's whole `ComplexField` abstraction. For this work, use the same low-level `pulp::Simd` complex lane operations directly in the existing TypeId-based fast path.

## File Map

- Modify `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/mul_pytorch_compare.rs`
  - Add dtype selection.
  - Run existing benchmark cases for `f64`, `num_complex::Complex32`, and `num_complex::Complex64`.
  - Emit dtype values that match PyTorch rows: `f64`, `c64`, `c128`.

- Modify `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/mul_pytorch_compare.py`
  - Add `--dtype` and `STRIDED_KERNEL_MUL_BENCH_DTYPES`.
  - Generate PyTorch tensors with `torch.float64`, `torch.complex64`, and `torch.complex128`.
  - Keep `torch.mul(..., out=...)` in the timed loop.

- Modify `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/run_mul_pytorch_compare.sh`
  - Iterate over dtype list and thread list.
  - Keep uv setup in the measurement procedure.

- Modify `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/src/simd.rs`
  - Add contiguous same-type complex SIMD multiplication for `Complex32` and `Complex64`.
  - Add complex coverage to the transposed-scalar 2D source-contiguous path.
  - Add unit tests for complex contiguous SIMD and complex transposed scalar cases.

- Update `/Users/hiroshi/projects/tensor4all/strided-rs/docs/faer-kernel-writing-guide.md`
  - Add a short note that faer uses `ComplexImpl<f32/f64>` plus `pulp::Simd` native complex lanes.

## Task 1: Add Complex Benchmark Dimension

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/mul_pytorch_compare.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/mul_pytorch_compare.py`
- Modify: `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/run_mul_pytorch_compare.sh`

- [ ] **Step 1: Add dtype enum to Rust benchmark**

Add this near `BenchCase`:

```rust
#[derive(Clone, Copy)]
enum BenchDType {
    F64,
    C64,
    C128,
}

impl BenchDType {
    fn label(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        }
    }
}
```

- [ ] **Step 2: Parse dtype list in Rust benchmark**

Add:

```rust
fn profile_dtypes() -> Vec<BenchDType> {
    env::var("STRIDED_KERNEL_MUL_BENCH_DTYPES")
        .unwrap_or_else(|_| "f64".to_string())
        .split(',')
        .filter_map(|value| match value.trim() {
            "f64" => Some(BenchDType::F64),
            "c64" => Some(BenchDType::C64),
            "c128" => Some(BenchDType::C128),
            _ => None,
        })
        .collect()
}
```

- [ ] **Step 3: Make Rust benchmark scalar construction generic**

Replace `make_col_major` with:

```rust
trait BenchScalar: Copy + 'static {
    fn from_indices(indices: &[usize], salt: usize) -> Self;
    fn make_out(dims: &[usize]) -> StridedArray<Self>;
}

impl BenchScalar for f64 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        fill_value(indices, salt)
    }

    fn make_out(dims: &[usize]) -> StridedArray<Self> {
        StridedArray::<Self>::col_major(dims)
    }
}

impl BenchScalar for num_complex::Complex32 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        Self::new(fill_value(indices, salt) as f32, fill_value(indices, salt + 17) as f32)
    }

    fn make_out(dims: &[usize]) -> StridedArray<Self> {
        StridedArray::<Self>::col_major(dims)
    }
}

impl BenchScalar for num_complex::Complex64 {
    fn from_indices(indices: &[usize], salt: usize) -> Self {
        Self::new(fill_value(indices, salt), fill_value(indices, salt + 17))
    }

    fn make_out(dims: &[usize]) -> StridedArray<Self> {
        StridedArray::<Self>::col_major(dims)
    }
}

fn make_col_major<T: BenchScalar>(dims: &[usize], salt: usize) -> StridedArray<T> {
    StridedArray::<T>::from_fn_col_major(dims, |idx| T::from_indices(idx, salt))
}
```

- [ ] **Step 4: Genericize Rust case runners**

Change runners from `fn run_elementwise(n: usize) -> (f64, f64)` to `fn run_elementwise<T: BenchScalar + std::ops::Mul<Output = T>>(n: usize) -> (f64, f64)` and use `StridedArray::<T>` outputs. Apply the same bound to outer-product and batched outer-product runners.

- [ ] **Step 5: Dispatch Rust dtype at runtime**

Replace `run_case(case)` with:

```rust
fn run_case_for_dtype(dtype: BenchDType, case: BenchCase) -> (f64, f64) {
    match dtype {
        BenchDType::F64 => run_case_typed::<f64>(case),
        BenchDType::C64 => run_case_typed::<num_complex::Complex32>(case),
        BenchDType::C128 => run_case_typed::<num_complex::Complex64>(case),
    }
}
```

Then loop over `for dtype in profile_dtypes() { for case in profile_cases() { ... } }` and emit `dtype.label()`.

- [ ] **Step 6: Add Python dtype parsing**

Add:

```python
def parse_dtypes(value: str) -> list[str]:
    dtypes = []
    for item in value.split(","):
        item = item.strip()
        if item in {"f64", "c64", "c128"}:
            dtypes.append(item)
        elif item:
            raise argparse.ArgumentTypeError(f"unknown dtype: {item}")
    return dtypes or ["f64"]


def torch_dtype(torch, dtype: str):
    if dtype == "f64":
        return torch.float64
    if dtype == "c64":
        return torch.complex64
    if dtype == "c128":
        return torch.complex128
    raise ValueError(f"unknown dtype: {dtype}")


def randn_tensor(torch, shape: tuple[int, ...], dtype: str):
    tdtype = torch_dtype(torch, dtype)
    if dtype == "f64":
        return torch.randn(shape, dtype=tdtype)
    real_dtype = torch.float32 if dtype == "c64" else torch.float64
    return torch.randn(shape, dtype=real_dtype) + 1j * torch.randn(shape, dtype=real_dtype)
```

- [ ] **Step 7: Thread dtype through Python case builders**

Change every `make_*_case(torch, ...)` to accept `dtype: str`, replace `torch.randn(..., dtype=torch.float64)` with `randn_tensor(torch, shape, dtype)`, and replace output allocation with `torch.empty(shape, dtype=torch_dtype(torch, dtype))`.

- [ ] **Step 8: Emit Python rows for all dtypes**

Add `--dtypes` with default `STRIDED_KERNEL_MUL_BENCH_DTYPES`:

```python
parser.add_argument("--dtypes", default=os.environ.get("STRIDED_KERNEL_MUL_BENCH_DTYPES", "f64"))
```

Then in `run`, use:

```python
for dtype in parse_dtypes(args.dtypes):
    for kind, dims in case_specs(args.profile):
        shape, fn = make_case(torch, dtype, kind, dims)
        ...
        writer.writerow({"dtype": dtype, ...})
```

- [ ] **Step 9: Verify benchmark plumbing**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
STRIDED_KERNEL_MUL_BENCH_PROFILE=smoke STRIDED_KERNEL_MUL_BENCH_DTYPES=f64,c64,c128 cargo bench -p strided-kernel --bench mul_pytorch_compare --features parallel -- --nocapture
python3 -m py_compile /Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/mul_pytorch_compare.py
```

Expected: Rust emits rows for three dtypes; Python compiles without errors.

## Task 2: Measure Complex Baseline Before Fast Paths

**Files:**
- Read: `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/run_mul_pytorch_compare.sh`

- [ ] **Step 1: Run 1T smoke baseline**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
STRIDED_KERNEL_MUL_BENCH_PROFILE=smoke STRIDED_KERNEL_MUL_BENCH_DTYPES=c64,c128 STRIDED_KERNEL_MUL_BENCH_RUNS=15 STRIDED_KERNEL_MUL_BENCH_WARMUPS=3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo bench -p strided-kernel --bench mul_pytorch_compare --features parallel -- --nocapture
uv run python strided-kernel/benches/mul_pytorch_compare.py --num-threads 1 --profile smoke --dtypes c64,c128 --runs 15 --warmups 3
```

Expected: All rows have `status=ok`.

- [ ] **Step 2: Identify gaps**

Compute ratio `strided-kernel median / pytorch median` by `benchmark,dtype,threads`. Treat `ratio >= 1.10` as a gap worth optimizing. Record cases separately for contiguous elementwise and broadcast/noncompact cases.

## Task 3: Add Contiguous Complex SIMD Mul

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/src/simd.rs`

- [ ] **Step 1: Write complex contiguous unit tests**

Add tests under `#[cfg(test)] mod tests`:

```rust
#[cfg(feature = "simd")]
#[test]
fn test_try_mul_contiguous_complex64() {
    let a = vec![
        num_complex::Complex64::new(1.0, 2.0),
        num_complex::Complex64::new(-3.0, 4.0),
        num_complex::Complex64::new(0.5, -0.25),
    ];
    let b = vec![
        num_complex::Complex64::new(5.0, -1.0),
        num_complex::Complex64::new(2.0, 0.25),
        num_complex::Complex64::new(-4.0, 3.0),
    ];
    let mut dst = vec![num_complex::Complex64::new(0.0, 0.0); a.len()];
    assert!(super::try_mul_contiguous(&mut dst, &a, &b));
    for i in 0..a.len() {
        assert_eq!(dst[i], a[i] * b[i]);
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_try_mul_contiguous_complex32() {
    let a = vec![
        num_complex::Complex32::new(1.0, 2.0),
        num_complex::Complex32::new(-3.0, 4.0),
        num_complex::Complex32::new(0.5, -0.25),
    ];
    let b = vec![
        num_complex::Complex32::new(5.0, -1.0),
        num_complex::Complex32::new(2.0, 0.25),
        num_complex::Complex32::new(-4.0, 3.0),
    ];
    let mut dst = vec![num_complex::Complex32::new(0.0, 0.0); a.len()];
    assert!(super::try_mul_contiguous(&mut dst, &a, &b));
    for i in 0..a.len() {
        assert_eq!(dst[i], a[i] * b[i]);
    }
}
```

- [ ] **Step 2: Add complex SIMD macro**

Add next to `impl_simd_mul!`:

```rust
#[cfg(feature = "simd")]
macro_rules! impl_simd_mul_complex {
    (
        $mul_into:ident,
        $ty:ty,
        $lanes:ident,
        $load:ident,
        $store:ident,
        $mul:ident
    ) => {
        fn $mul_into(dst: &mut [$ty], a: &[$ty], b: &[$ty]) {
            struct Mul<'a> {
                dst: &'a mut [$ty],
                a: &'a [$ty],
                b: &'a [$ty],
            }

            impl<'a> pulp::WithSimd for Mul<'a> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let lanes = S::$lanes;
                    let mut i = 0usize;
                    while i + lanes <= self.dst.len() {
                        let va = simd.$load(&self.a[i..i + lanes]);
                        let vb = simd.$load(&self.b[i..i + lanes]);
                        simd.$store(&mut self.dst[i..i + lanes], simd.$mul(va, vb));
                        i += lanes;
                    }
                    if i < self.dst.len() {
                        let va = simd.$load(&self.a[i..]);
                        let vb = simd.$load(&self.b[i..]);
                        simd.$store(&mut self.dst[i..], simd.$mul(va, vb));
                    }
                }
            }

            pulp::Arch::new().dispatch(Mul { dst, a, b });
        }
    };
}
```

- [ ] **Step 3: Instantiate complex SIMD functions**

Add:

```rust
#[cfg(feature = "simd")]
impl_simd_mul_complex!(
    simd_mul_c32_into,
    num_complex::Complex32,
    C32_LANES,
    partial_load_c32s,
    partial_store_c32s,
    mul_e_c32s
);

#[cfg(feature = "simd")]
impl_simd_mul_complex!(
    simd_mul_c64_into,
    num_complex::Complex64,
    C64_LANES,
    partial_load_c64s,
    partial_store_c64s,
    mul_e_c64s
);
```

- [ ] **Step 4: Add TypeId branches**

Add these branches to `try_mul_contiguous` after the real branches:

```rust
if TypeId::of::<D>() == TypeId::of::<num_complex::Complex64>()
    && TypeId::of::<A>() == TypeId::of::<num_complex::Complex64>()
    && TypeId::of::<B>() == TypeId::of::<num_complex::Complex64>()
{
    unsafe { simd_mul_c64_into(cast_slice_mut(dst), cast_slice(a), cast_slice(b)) };
    return true;
}
if TypeId::of::<D>() == TypeId::of::<num_complex::Complex32>()
    && TypeId::of::<A>() == TypeId::of::<num_complex::Complex32>()
    && TypeId::of::<B>() == TypeId::of::<num_complex::Complex32>()
{
    unsafe { simd_mul_c32_into(cast_slice_mut(dst), cast_slice(a), cast_slice(b)) };
    return true;
}
```

- [ ] **Step 5: Verify tests**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo test -p strided-kernel --features parallel
cargo test -p strided-kernel --no-default-features
```

Expected: both pass. `--no-default-features` confirms complex support still falls back cleanly without SIMD.

## Task 4: Add Complex Transposed-Scalar Coverage

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/src/simd.rs`

- [ ] **Step 1: Add complex transposed scalar tests**

Add:

```rust
#[cfg(feature = "parallel")]
#[test]
fn test_transposed_scalar_rhs_2d_complex64_source_contiguous() {
    let inner_len = 5usize;
    let row_len = 7usize;
    let src: Vec<num_complex::Complex64> = (0..inner_len * row_len)
        .map(|i| num_complex::Complex64::new(i as f64 + 0.25, i as f64 * -0.5))
        .collect();
    let scalar = num_complex::Complex64::new(2.0, -0.25);
    let mut dst = vec![num_complex::Complex64::new(0.0, 0.0); inner_len * row_len];

    let used = unsafe {
        super::try_mul_transposed_scalar_rhs_2d::<
            num_complex::Complex64,
            num_complex::Complex64,
            num_complex::Complex64,
        >(
            dst.as_mut_ptr(),
            src.as_ptr(),
            &scalar,
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };

    assert!(used);
    for row in 0..row_len {
        for inner in 0..inner_len {
            assert_eq!(dst[row * inner_len + inner], src[inner * row_len + row] * scalar);
        }
    }
}
```

- [ ] **Step 2: Add TypeId branches for complex transposed scalar**

Add same-type `Complex64` and `Complex32` branches in both `try_mul_transposed_scalar_rhs_2d` and `try_mul_transposed_scalar_lhs_2d`, using the existing generic `mul_transposed_scalar_rhs_source_contiguous` function.

For `Complex64` rhs-scalar branch:

```rust
if TypeId::of::<D>() == TypeId::of::<num_complex::Complex64>()
    && TypeId::of::<A>() == TypeId::of::<num_complex::Complex64>()
    && TypeId::of::<B>() == TypeId::of::<num_complex::Complex64>()
{
    unsafe {
        mul_transposed_scalar_rhs_source_contiguous(
            dst.cast::<num_complex::Complex64>(),
            src.cast::<num_complex::Complex64>(),
            *scalar.cast::<num_complex::Complex64>(),
            inner_len,
            row_len,
            src_fast_stride,
            src_row_stride,
        );
    }
    return true;
}
```

Repeat with `num_complex::Complex32`, and mirror the branches in the lhs-scalar function.

- [ ] **Step 3: Verify tests**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo test -p strided-kernel --features parallel
```

Expected: complex transposed-scalar tests pass.

## Task 5: Benchmark Complex 1T and 4T

**Files:**
- Read: `/Users/hiroshi/projects/tensor4all/strided-rs/strided-kernel/benches/run_mul_pytorch_compare.sh`

- [ ] **Step 1: Run 1T full comparison**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
STRIDED_KERNEL_MUL_BENCH_PROFILE=full STRIDED_KERNEL_MUL_BENCH_DTYPES=c64,c128 STRIDED_KERNEL_MUL_BENCH_RUNS=50 STRIDED_KERNEL_MUL_BENCH_WARMUPS=10 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 bash strided-kernel/benches/run_mul_pytorch_compare.sh
```

Expected: CSV contains both `strided-kernel` and `pytorch-cpu` rows for `c64` and `c128`.

- [ ] **Step 2: Run 4T full comparison**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
STRIDED_KERNEL_MUL_BENCH_PROFILE=full STRIDED_KERNEL_MUL_BENCH_DTYPES=c64,c128 STRIDED_KERNEL_MUL_BENCH_RUNS=50 STRIDED_KERNEL_MUL_BENCH_WARMUPS=10 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 bash strided-kernel/benches/run_mul_pytorch_compare.sh
```

Expected: 4T ratios should not regress real `f64` behavior. If complex 4T is slower than 1T for small cases, keep that as a scheduling threshold issue rather than adding another fast path immediately.

- [ ] **Step 3: Summarize gaps**

Report:

```text
benchmark,dtype,threads,strided_ms,pytorch_ms,ratio,classification
```

Classify:

- `contiguous`: `bin_elementwise_mul_*`
- `broadcast compact`: `bin_outer_product_*`, `bin_batched_outer_product_compact_*`
- `broadcast noncompact`: `bin_batched_outer_product_noncompact_*`
- `source-transposed scalar`: `*_lhs_scalar_*`

## Task 6: Document Complex SIMD Rule

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/strided-rs/docs/faer-kernel-writing-guide.md`

- [ ] **Step 1: Add complex SIMD note**

Add:

```markdown
### Complex SIMD

faer has two complex SIMD layers. The generic `num_complex::Complex<T>` implementation composes real SIMD lanes as `Complex<T::SimdVec<S>>`. For native `c32`/`c64` kernels, faer dispatches through `ComplexImpl<f32>` or `ComplexImpl<f64>` and uses `pulp::Simd` native complex lanes (`S::c32s`, `S::c64s`) plus operations such as `mul_e_c32s`, `mul_e_c64s`, `conj_mul_e_c32s`, and `conj_mul_e_c64s`.

For strided-kernel elementwise complex multiplication, prefer direct `pulp::Simd` complex operations at the TypeId dispatch boundary. Do not add a broad trait abstraction unless multiple complex kernels need the same dispatch surface.
```

- [ ] **Step 2: Check docs formatting**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
git diff --check
```

Expected: no whitespace errors.

## Self-Review

- Spec coverage: faer complex SIMD was checked first; benchmark expansion and implementation plan are separated; 1T and 4T comparisons are included.
- Placeholder scan: no `TBD`, `TODO`, or unspecified files remain in this plan.
- Type consistency: `c64` maps to `num_complex::Complex32` and `torch.complex64`; `c128` maps to `num_complex::Complex64` and `torch.complex128`.
