# Issue #190 worklog

## Design

Issue #190 replaces safe erased byte storage construction with typed storage
witnesses. Integer arithmetic and `align_offset` can check offsets and relative
alignment, but neither can certify that an allocation itself was created with
the alignment required by `f64` or complex storage. The safe constructors
therefore accept only sealed, known kernel element types and derive the dtype,
byte length, alignment, and validity from `&[T]`, `&mut [T]`, or
`&mut [MaybeUninit<T>]`.

`KernelStorageElement` is public but sealed, `Copy`, `'static`, and limited to
`f32`, `f64`, `i32`, `i64`, `bool`, `Complex32`, and `Complex64`. Erased raw
construction remains an explicitly unsafe boundary with documented allocation,
extent, provenance, lifetime, initialization, alignment, and aliasing
requirements. It is used only where storage is genuinely erased.

## Bool and lifetime contracts

Raw `Bool` bytes are initialized bytes; only the byte value may be temporarily
invalid. Pointer consumers first perform dtype checks, then all overlap checks,
then deferred Bool validation and typed conversion. The concatenate regression
uses an earlier invalid byte and a later overlapping input, proving that global
overlap ordering wins over invalid-Bool reporting.

Raw pointer metadata uses `PhantomData<&'a [MaybeUninit<u8>]>`. Conversion after
the caller has proved no mutable overlap returns a borrow tied to the method
borrow, and requires the complete byte extent to be initialized and dtype-valid.
The raw pointer contract permits sequential mutation through the original raw
pointer before execution, provided no concurrent access occurs; the mutation
regression verifies invalidation is detected before destination writes.

## Migration scope

Production erased kernel paths now use descriptor-owned typed accessors and
preserve dtype, overlap, deferred Bool validation, conversion, and write
ordering. Workspace unit/integration tests, benchmarks, and examples were
migrated to typed constructors where storage is known. Obsolete safe byte
constructors, byte exposure, typed-slice helpers, and erased data accessors were
removed. No benchmark tuning, threshold changes, performance claim, or intended
semantic behavior change was made.

## Verification

The following checks passed during the implementation and final passes:

```text
cargo fmt --all
cargo fmt --all -- --check
cargo check --workspace --lib
cargo check --workspace --benches --examples
cargo check --workspace --all-targets
cargo test --workspace --no-run
cargo test --workspace
cargo test -p strided-kernel --test issue_190_bool_overlap
cargo doc --workspace --no-deps
git diff --check
```

Focused strict-provenance Miri passed:

```text
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' \
  cargo +nightly miri test -p strided-view \
  raw::tests::typed_erased_storage_covers_all_kernel_dtypes
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' \
  cargo +nightly miri test -p strided-kernel --test issue_190_bool_overlap \
  concatenate_checks_all_overlaps_before_deferred_bool_validation
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' \
  cargo +nightly miri test -p strided-kernel --test issue_190_bool_overlap \
  bool_mutation_after_raw_pointer_handoff_is_validated_before_writes
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' \
  cargo +nightly miri test -p strided-kernel --test issue_190_bool_overlap \
  bool_uninitialized_replay_does_not_read_strided_holes
```

The strided Bool hole regression verifies only reachable offsets are assumed
initialized after uninitialized replay; the unreachable backing hole remains
`MaybeUninit` and is never read.

The broad `strided-view raw::tests` Miri filter still fails in the pre-existing
`raw_mut_can_reborrow_as_view` test with a Stacked Borrows violation in the
older view reborrow path. This was recorded separately and was not fixed or
claimed as fixed by issue #190. The source-contract scanner passes normally;
filesystem traversal is not Miri-isolated and is therefore not included in the
focused Miri run.

Coverage passed exactly `53/53 files passed` with:

```text
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```
