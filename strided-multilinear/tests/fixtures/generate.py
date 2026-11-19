#!/usr/bin/env python3
"""Generate the `strided-multilinear` fixtures from cuEquivariance.

Pinned to NVIDIA/cuEquivariance `9ed8c7613a79861415071027249cc458e423be30`
(Apache-2.0). Neither the descriptors nor the numbers in `tests/fixtures/` are
hand-written here:

* the **descriptor** of a case is the one cuEquivariance's `uniform_1d` lowering
  produces. `cuequivariance_ops_jax` is replaced by a stub that records the
  lowered operations, paths and reshaped buffers instead of launching the CUDA
  kernel, so the recorded form is the lowering's own output.
* the **expectation** is the same polynomial evaluated by cuEquivariance's
  `naive` method on the same buffers.

Usage (needs `jax[cpu]` and a checkout of cuequivariance, but no GPU):

    python3 tests/fixtures/generate.py

Writes `tests/fixtures/<case>/*.npy` and `tests/fixtures/cases.rs`.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

# The script's own `Path` descriptor class shadows this below, so keep the real one.
PATHLIB = Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import cuequivariance as cue
import cuequivariance_jax as cuex
import cuequivariance_jax.segmented_polynomials.segmented_polynomial as spmod
from cuequivariance_jax.segmented_polynomials.segmented_polynomial_uniform_1d import (
    execute_uniform_1d,
)

PIN = "9ed8c7613a79861415071027249cc458e423be30"
HERE = Path(__file__).resolve().parent


class _Stop(Exception):
    """Ends a capture: the run is never meant to reach a kernel."""


LOWERED: dict = {}

# The output-seed derivation per case, fixed so regeneration is deterministic.
OUTPUT_SEEDS = {
    "sym_deg1": 1,
    "sym_deg2": 2,
    "sym_deg3": 3,
    "sh": 4,
    "jvp_deg2": 5,
    "backward_deg2": 6,
    "indexed": 7,
}


class Operation:
    def __init__(self, buffers, path_start, num_paths):
        self.buffers = tuple(int(b) for b in buffers)
        self.path_start = int(path_start)
        self.num_paths = int(num_paths)


class Path:
    def __init__(self, indices, coefficient):
        self.indices = tuple(int(i) for i in indices)
        self.coefficient = float(coefficient)


def require_pinned_upstream() -> None:
    """Refuse to generate from a checkout that is not the pinned revision."""
    source = cuex.__file__
    root = PATHLIB(source).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():
            head = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if head != PIN:
                raise SystemExit(f"{parent} is at {head}, but the fixtures are pinned to {PIN}")
            return
    raise SystemExit(f"cannot find the checkout of {source} to verify {PIN}")


def install_fakes() -> None:
    def uniform_1d(inputs, outputs, indices, index_configuration, **kwargs):
        LOWERED.update(
            dict(
                inputs=list(inputs),
                outputs=list(outputs),
                indices=list(indices),
                index_configuration=np.asarray(index_configuration),
                **kwargs,
            )
        )
        raise _Stop()

    stub = types.ModuleType("cuequivariance_ops_jax")
    # `execute_uniform_1d` refuses anything below 0.4.0.dev.
    stub.__version__ = "0.6.0"
    stub.Operation = Operation
    stub.Path = Path
    stub.uniform_1d = uniform_1d
    sys.modules["cuequivariance_ops_jax"] = stub


def lower(polynomial, inputs, outputs_shape_dtype, indices, index_configuration):
    """The descriptor cuEquivariance's `uniform_1d` lowering produces."""
    # The same pre-transform `segmented_polynomial_impl` applies before dispatch.
    try:
        polynomial = polynomial.unsymmetrize_for_identical_operands()
    except NotImplementedError:
        pass
    LOWERED.clear()
    try:
        execute_uniform_1d(
            inputs=inputs,
            outputs_shape_dtype=outputs_shape_dtype,
            indices=indices,
            index_configuration=index_configuration,
            polynomial=polynomial,
            options={},
            name="fixture",
        )
    except _Stop:
        pass
    lowered = dict(LOWERED)
    if not lowered:
        raise SystemExit("the lowering did not reach the kernel")
    return lowered


def modes_of(index_configuration, num_buffers, batch):
    """Map cuEquivariance's index configuration to this crate's index modes."""
    modes = []
    for buffer in range(num_buffers):
        row = index_configuration[buffer]
        assert len(row) == 1, "the executor accepts exactly one batch axis"
        index = int(row[0])
        modes.append(-1 if index < 0 else index)
    return modes


def emit_case(
    name, polynomial, arrays, index_arrays, batch, output_rows, gather_of, scatter_of
):
    """Lower one case, evaluate it, and write its payloads."""
    num_inputs = polynomial.num_inputs
    outputs_shape_dtype = [
        jax.ShapeDtypeStruct((rows, operand.size), jnp.float64)
        for rows, operand in zip(output_rows, polynomial.outputs)
    ]
    num_buffers = num_inputs + polynomial.num_outputs
    indices = [None] * num_buffers
    for buffer, index in gather_of.items():
        indices[buffer] = np.s_[index_arrays[index]]
    for buffer, index in scatter_of.items():
        indices[buffer] = np.s_[index_arrays[index]]

    # Rebuild what the prim would have prepared: one batch axis, one entry per
    # buffer, and a unique index array per distinct destination.
    positions = {}
    index_configuration = []
    for buffer in range(num_buffers):
        index = None
        if buffer in gather_of:
            index = gather_of[buffer]
        elif buffer in scatter_of:
            index = scatter_of[buffer]
        if index is None:
            index_configuration.append([-1])
        else:
            if index not in positions:
                positions[index] = len(positions)
            index_configuration.append([positions[index]])
    unique = [index_arrays[slot] for slot, _ in sorted(positions.items(), key=lambda kv: kv[1])]

    lowered = lower(
        polynomial,
        [jnp.asarray(array) for array in arrays],
        outputs_shape_dtype,
        [jnp.asarray(array) for array in unique],
        index_configuration,
    )
    expected = cuex.segmented_polynomial(
        polynomial, arrays, outputs_shape_dtype, indices, method="naive"
    )
    lowered_inputs = lowered["inputs"]
    lowered_outputs = lowered["outputs"]
    configuration = lowered["index_configuration"]
    num_buffers = len(lowered_inputs) + len(lowered_outputs)
    assert configuration.shape[0] >= num_buffers

    directory = HERE / name
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.npy"):
        stale.unlink()

    # Buffers: inputs first, outputs last, exactly the order the descriptor uses.
    dims = [tuple(int(d) for d in array.shape) for array in lowered_inputs]
    dims += [tuple(int(d) for d in array.shape) for array in lowered_outputs]
    payloads = [np.asarray(array, dtype=np.float64).reshape(shape)
                for array, shape in zip(arrays, dims[:num_inputs])]
    for buffer, payload in enumerate(payloads):
        np.save(directory / f"buffer{buffer}.npy", payload)
    # An output's payload is only its *seed*: `execute_acc` starts from it, so it
    # must not be the expectation.
    # A fixed seed per case, so regenerating into a partial tree produces the same
    # files: a seed derived from the directory count would not.
    rng = np.random.default_rng(10_000 + OUTPUT_SEEDS[name])
    for dense, (value, shape) in enumerate(zip(expected, dims[num_inputs:])):
        buffer = num_inputs + dense
        seed = rng.normal(size=shape).astype(np.float64)
        np.save(directory / f"buffer{buffer}.npy", seed)
        np.save(directory / f"expected{dense}.npy", np.asarray(value, dtype=np.float64).reshape(shape))

    # Index arrays, in the order `index_configuration` numbers them.
    unique_indices = lowered["indices"]
    index_files = []
    for slot, index in enumerate(unique_indices):
        values = np.asarray(index, dtype=np.int32).reshape(-1)
        assert values.shape == (batch,), (name, slot, values.shape)
        np.save(directory / f"index{slot}.npy", values)
        index_files.append((slot, values))

    modes = modes_of(configuration, num_buffers, batch)
    kinds = []
    for buffer, mode in enumerate(modes):
        extent = dims[buffer][0]
        if mode >= 0:
            kinds.append(("gather" if buffer < num_inputs else "scatter", mode, extent))
        elif extent == 1 and batch > 1:
            if buffer >= num_inputs:
                raise SystemExit(
                    f"{name}: output {buffer} is shared (batch reduction); the naive "
                    "method clamps instead of summing, so it cannot be the reference"
                )
            kinds.append(("shared", None, extent))
        else:
            assert extent == batch, (name, buffer, extent, batch)
            kinds.append(("batched", None, extent))

    operations = lowered["operations"]
    paths = lowered["paths"]
    assert sum(operation.num_paths for operation in operations) == len(paths)

    return dict(
        name=name,
        batch=batch,
        num_inputs=num_inputs,
        dims=dims,
        kinds=kinds,
        operations=[
            dict(
                buffers=operation.buffers,
                path_start=operation.path_start,
                num_paths=operation.num_paths,
            )
            for operation in operations
        ],
        paths=[
            dict(
                coefficient=path.coefficient,
                segments=[int(s) for s in path.indices],
            )
            for path in paths
        ],
        index_files=index_files,
    )


def rust_case(case) -> str:
    """One `Case` literal for `tests/fixtures/cases.rs`."""
    name = case["name"]
    lines = [
        f"    // {name}: batch {case['batch']}, {len(case['dims'])} buffers, "
        f"{len(case['operations'])} operations"
    ]
    lines.append("    Case {")
    lines.append(f"        batch: {case['batch']},")
    lines.append("        inputs: vec![")
    for buffer, (dims, kind) in enumerate(
        zip(case["dims"][: case["num_inputs"]], case["kinds"][: case["num_inputs"]])
    ):
        lines.append(
            "            Buffer::dense("
            f"{list(dims)}, {rust_mode(kind, buffer, case)}, "
            f"npy(\"{name}/buffer{buffer}.npy\", &{list(dims)})),"
        )
    lines.append("        ],")
    lines.append("        outputs: vec![")
    for dense, (dims, kind) in enumerate(
        zip(case["dims"][case["num_inputs"] :], case["kinds"][case["num_inputs"] :])
    ):
        buffer = case["num_inputs"] + dense
        lines.append(
            "            Buffer::dense("
            f"{list(dims)}, {rust_mode(kind, buffer, case)}, "
            f"npy(\"{name}/buffer{buffer}.npy\", &{list(dims)})),"
        )
    lines.append("        ],")
    lines.append("        ops: vec![")
    for operation in case["operations"]:
        lines.append("            Op {")
        lines.append(f"                buffers: vec!{list(operation['buffers'])},")
        lines.append("                paths: vec![")
        for path in case["paths"][
            operation["path_start"] : operation["path_start"] + operation["num_paths"]
        ]:
            segments = ", ".join(str(s) for s in path["segments"])
            lines.append(
                "                    PathSpec { segments: vec!["
                f"{segments}], coefficient: {path['coefficient']!r} }},"
            )
        lines.append("                ],")
        lines.append("            },")
    lines.append("        ],")
    lines.append("        expected: vec![")
    for dense in range(len(case["dims"]) - case["num_inputs"]):
        dims = case["dims"][case["num_inputs"] + dense]
        lines.append(
            f"            npy(\"{name}/expected{dense}.npy\", &{list(dims)}),"
        )
    lines.append("        ],")
    lines.append("    },")
    return "\n".join(lines)


def rust_mode(kind, buffer, case) -> str:
    name, slot, extent = kind
    if name == "batched":
        return "Mode::Batched"
    if name == "shared":
        return "Mode::Shared"
    # An index array runs over the batch, not over the rows of the buffer it
    # addresses.
    index = f"npy_i32(\"{case['name']}/index{slot}.npy\", &[{case['batch']}])"
    if name == "gather":
        return f"Mode::Gather({index})"
    return f"Mode::Scatter({index})"


def main() -> None:
    require_pinned_upstream()
    install_fakes()

    same_mul = 4  # the descriptor requires one multiplicity, and it becomes `u`

    def scaled(irreps):
        return cue.Irreps(
            cue.O3, "+".join(f"{same_mul}x{term.strip()}" for term in irreps.split("+"))
        )

    def symmetric(degrees, irreps_in, irreps_out):
        return cue.descriptors.symmetric_contraction(
            scaled(irreps_in), scaled(irreps_out), degrees
        ).polynomial

    def spherical():
        return cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial

    cases = []

    batch = 6

    def add(name, polynomial, seed, rows, output_rows=None, gather_of=None, scatter_of=None):
        rng = np.random.default_rng(seed)
        if output_rows is None:
            output_rows = [batch] * polynomial.num_outputs
        examples = [
            rng.normal(size=(count, operand.size)).astype(np.float64)
            for count, operand in zip(rows, polynomial.inputs)
        ]
        index_arrays, gather_of, scatter_of = {}, dict(gather_of or {}), dict(scatter_of or {})
        for buffer, (slots, low, high) in {
            **{b: (s, lo, hi) for b, (s, lo, hi) in (gather_of or {}).items()},
            **{b: (s, lo, hi) for b, (s, lo, hi) in (scatter_of or {}).items()},
        }.items():
            index_arrays[slots] = rng.integers(low, high, batch).astype(np.int32)
        cases.append(
            emit_case(
                name,
                polynomial,
                examples,
                index_arrays,
                batch,
                output_rows,
                {b: s for b, (s, _, _) in (gather_of or {}).items()},
                {b: s for b, (s, _, _) in (scatter_of or {}).items()},
            )
        )

    # `symmetric_contraction(irreps_in, irreps_out, degrees)` is
    # `weights[u], input[u] (degree times), output[u]`: buffer 0 is the
    # (shared) weight tensor, buffer 1 the batched data, the last the output.
    add("sym_deg1", symmetric([1], "1o+0e", "1o"), seed=1, rows=[1, batch])
    add("sym_deg2", symmetric([2], "1o", "0e"), seed=2, rows=[1, batch])
    add("sym_deg3", symmetric([3], "1o", "1o"), seed=3, rows=[1, batch])
    add("sh", spherical(), seed=4, rows=[batch])

    degree2 = symmetric([2], "1o", "0e")
    jvp_polynomial, _ = degree2.jvp([False, True])
    # (w, x, dx): the tangent is a batched buffer too.
    add("jvp_deg2", jvp_polynomial, seed=5, rows=[1, batch, batch])

    # Reverse mode goes through the linearised polynomial: transposing the
    # non-linear one would need the `x` operand twice and is rejected upstream.
    backward_polynomial, _ = jvp_polynomial.transpose([False, False, True], [True])
    # (w, x, dy) -> (dx).
    add("backward_deg2", backward_polynomial, seed=6, rows=[1, batch, batch])

    # Indexed message passing: per-edge weights (batched), the source features
    # gathered by `src`, the output scattered into the receivers' rows.
    add(
        "indexed",
        symmetric([1], "1o+0e", "1o"),
        seed=7,
        rows=[batch, 5],
        output_rows=[3],
        gather_of={1: (0, 0, 5)},
        scatter_of={2: (1, 0, 3)},
    )

    lines = [
        "// @generated by tests/fixtures/generate.py -- do not edit.",
        f"//",
        f"// Descriptors: cuEquivariance's `uniform_1d` lowering at {PIN}.",
        "// Expectations: its `naive` method on the same buffers.",
        "",
        "#[allow(unused_imports)]",
        "use crate::support::{assert_close, npy, npy_i32, Buffer, Case, Mode, Op, PathSpec};",
        "",
        "pub fn cases() -> Vec<Case> {",
        "    vec![",
    ]
    for case in cases:
        lines.append(rust_case(case))
    lines += ["    ]", "}", ""]
    (HERE / "cases.rs").write_text("\n".join(lines))

    for case in cases:
        paths = sum(operation["num_paths"] for operation in case["operations"])
        print(
            f"{case['name']:14s} batch {case['batch']:2d}  "
            f"buffers {len(case['dims'])}  ops {len(case['operations'])}  paths {paths}  "
            f"dims {case['dims']}"
        )
    print(f"\nwrote {len(cases)} cases to {HERE}")


if __name__ == "__main__":
    main()
