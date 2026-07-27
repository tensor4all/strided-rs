## Erased Copy-Plan API

Issue: tensor4all/strided-rs#149

Goal: add the first non-generic replay boundary for tenferro CPU-kernel migration by wrapping the existing `CopyPlan` in a dtype-erased, dtype-concrete dispatch layer owned by `strided-kernel`.

Scope:

- Add a small public dtype enum for the scalar set needed by tensor callers.
- Add borrowed byte-slice raw descriptors with element-based dims, strides, and offsets.
- Add an erased copy plan that stores a `CopyPlan` plus dtype and dispatches to pre-instantiated concrete functions inside `strided-kernel`.
- Pass an explicit `ExecContext` through the erased replay boundary, even while copy replay itself stays serial, so downstream runtimes do not rely on ambient thread-pool state.
- Validate dtype equality, byte length divisibility, and pointer alignment before constructing typed raw views.
- Keep `KernelDType` non-exhaustive and make bool byte revalidation dirty-bit based rather than scanning output buffers on every replay.
- Keep `CopyPlan::execute<T>` unchanged for existing Rust users.

Out of scope:

- C ABI symbols.
- Thread-pool handles, affinity controls, and execution-scope ownership for the later C ABI.
- scale/conj erased operations.
- indexed gather/scatter/dynamic_slice kernels.
- tenferro dependency migration.
- new performance tuning beyond preserving the existing prepared-copy zero-allocation path.

Verification:

- Add integration tests for transposed f64 copy, dtype mismatch rejection, supported dtype dispatch, and allocation-free replay for rank <= `RAW_FUSED_RANK_LIMIT`.
- Run targeted `strided-kernel` tests first, then workspace tests.
