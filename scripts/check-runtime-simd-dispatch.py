#!/usr/bin/env python3
"""Check that a baseline x86-64 build reaches AVX2 and AVX-512 at run time.

The elementwise leaf loops of strided-basic, strided-fused, strided-nn and
strided-multilinear run inside `pulp::Arch::new().dispatch(...)`, which compiles the
inlined loop body once per SIMD level and picks one at run time
(tensor4all/strided-rs#340, #496). This script builds each crate's probe example
in release mode with `RUSTFLAGS` cleared, so the build target is the x86-64
baseline, runs it, and disassembles it. Each check names a dispatched body and one arithmetic
instruction it must hold: the pulp `V4` version must use it on `zmm` registers
and the `V3` version on `ymm` registers. A body that pulp calls without
inlining keeps the baseline instructions and fails the check.

Run with `python3 scripts/check-runtime-simd-dispatch.py` on an x86-64 host with
`objdump`. `--asm FILE` checks an existing `objdump -d --no-show-raw-insn -C`
listing instead of building.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (package, example) pairs whose probes exercise the dispatched bodies below.
PROBES = [
    ("strided-kernel", "runtime_simd_probe"),
    ("strided-multilinear", "simd_probe"),
]

# (dispatched body, instruction): the body's demangled name fragment and one
# arithmetic instruction its loop must vectorize.
CHECKS = [
    ("strided_basic::map_view::map_contiguous", "vsqrtpd"),
    ("strided_basic::map_view::inner_loop_map2", "vaddpd"),
    ("strided_fused::fused::chunk::ReplayBlock<f64>", "vsqrtpd"),
    ("strided_basic::map_view::inner_loop_map4<f64", "vsqrtpd"),
    ("strided_nn::simd::ShiftGroup>", "vmaxps"),
    ("strided_nn::simd::ScaleGroup>", "vdivps"),
    ("strided_nn::simd::ScaleGroupF64>", "vdivpd"),
    ("<f32 as strided_nn::softmax_axis::lanes::Lanes>::max_into", "vmaxps"),
    ("<f64 as strided_nn::softmax_axis::lanes::Lanes>::max_into", "vmaxpd"),
    ("<f32 as strided_nn::softmax_axis::lanes::Lanes>::shift_into", "vsubps"),
    ("<f64 as strided_nn::softmax_axis::lanes::Lanes>::shift_into", "vsubpd"),
    ("<f32 as strided_nn::softmax_axis::lanes::Lanes>::add_into", "vaddps"),
    ("<f64 as strided_nn::softmax_axis::lanes::Lanes>::add_into", "vaddpd"),
    ("<f32 as strided_nn::softmax_axis::lanes::Lanes>::div_into", "vdivps"),
    ("<f64 as strided_nn::softmax_axis::lanes::Lanes>::div_into", "vdivpd"),
    ("<f32 as strided_nn::softmax_axis::lanes::Lanes>::dot_into", "vmulps"),
    ("<f64 as strided_nn::softmax_axis::lanes::Lanes>::dot_into", "vmulpd"),
    ("<f32 as strided_nn::softmax_axis::lanes::Lanes>::apply_into", "vmulps"),
    ("<f64 as strided_nn::softmax_axis::lanes::Lanes>::apply_into", "vmulpd"),
    ("<f32 as strided_nn::softmax_axis::lanes::Lanes>::vjp_rows", "vmulps"),
    ("<f64 as strided_nn::softmax_axis::lanes::Lanes>::vjp_rows", "vmulpd"),
    ("strided_multilinear::simd::RowF64<", "vmulpd"),
    ("strided_multilinear::simd::RowF32<", "vmulps"),
]

LEVELS = [("V4", "zmm"), ("V3", "ymm")]

FUNCTION = re.compile(r"^[0-9a-f]+ <(.*)>:$")


def parse_functions(listing: str) -> dict[str, list[str]]:
    """Map each function of an objdump listing to its instruction texts."""
    functions: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in listing.splitlines():
        match = FUNCTION.match(line)
        if match:
            current = functions.setdefault(match.group(1), [])
            continue
        if current is not None and "\t" in line:
            current.append(line.split("\t", 1)[1].strip())
    return functions


def failures(functions: dict[str, list[str]]) -> list[str]:
    """Every (body, level) pair whose dispatched version lacks its instruction."""
    missing = []
    for body, instruction in CHECKS:
        for level, register in LEVELS:
            candidates = [
                name
                for name in functions
                if f"::{level}>::vectorize" in name and body in name
            ]
            found = any(
                text.startswith(instruction) and register in text
                for name in candidates
                for text in functions[name]
            )
            if not found:
                missing.append(
                    f"{body}: no {level} version uses {instruction} on {register} "
                    f"({len(candidates)} dispatched instances)"
                )
    return missing


def build_probes() -> list[pathlib.Path]:
    """Build every probe example against the baseline target."""
    env = dict(os.environ)
    # The baseline target: no `-C target-cpu`, from the environment or config.
    env["RUSTFLAGS"] = ""
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.pop("CARGO_BUILD_RUSTFLAGS", None)
    jobs = os.environ.get("CHECK_SIMD_JOBS")
    executables: list[pathlib.Path] = []
    for package, probe in PROBES:
        command = [
            "cargo",
            "build",
            "--release",
            "-p",
            package,
            "--example",
            probe,
            "--message-format=json",
        ]
        if jobs:
            command += ["-j", jobs]
        output = subprocess.run(
            command, cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE, text=True
        ).stdout
        found = None
        for line in output.splitlines():
            message = json.loads(line)
            if (
                message.get("reason") == "compiler-artifact"
                and message.get("target", {}).get("name") == probe
                and message.get("executable")
            ):
                found = pathlib.Path(message["executable"])
        if found is None:
            raise RuntimeError(f"cargo did not report the {probe} executable")
        executables.append(found)
    return executables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asm", type=pathlib.Path, help="check an existing listing")
    args = parser.parse_args()
    if args.asm:
        listing = args.asm.read_text()
    else:
        listing = ""
        for probe in build_probes():
            subprocess.run([str(probe)], check=True)
            listing += subprocess.run(
                ["objdump", "-d", "--no-show-raw-insn", "-C", str(probe)],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout
    missing = failures(parse_functions(listing))
    for line in missing:
        print(f"FAIL {line}")
    if missing:
        return 1
    print(f"ok: {len(CHECKS)} dispatched bodies reach AVX2 and AVX-512")
    return 0


if __name__ == "__main__":
    sys.exit(main())
