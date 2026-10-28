#!/usr/bin/env python3
"""Self-contained tests for scripts/check-runtime-simd-dispatch.py.

Run with `python3 scripts/test-check-runtime-simd-dispatch.py`; no pytest needed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "check-runtime-simd-dispatch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_runtime_simd", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def listing(bodies: dict[str, list[str]]) -> str:
    lines = []
    for address, (name, instructions) in enumerate(bodies.items()):
        lines.append(f"{address:016x} <{name}>:")
        lines += [f"  {address:x}:\t{text}" for text in instructions]
        lines.append("")
    return "\n".join(lines)


def full_listing(module, drop: str | None = None) -> str:
    bodies = {}
    for body, instruction in module.CHECKS:
        for level, register in module.LEVELS:
            name = f"<pulp::x86::{level}>::vectorize::imp::<Impl<{body}<f64>>>"
            text = f"{instruction} %{register}1,%{register}2"
            if (body, level) == drop:
                text = "jmp    1234 <inner>"
            bodies[name] = [text, "ret"]
    return listing(bodies)


def test_a_complete_listing_passes(module) -> None:
    assert module.failures(module.parse_functions(full_listing(module))) == []


def test_a_body_left_at_the_baseline_fails(module) -> None:
    body = module.CHECKS[2][0]
    missing = module.failures(module.parse_functions(full_listing(module, (body, "V4"))))
    assert len(missing) == 1 and body in missing[0] and "V4" in missing[0], missing


def test_the_instruction_must_use_the_level_register(module) -> None:
    body, instruction = module.CHECKS[0]
    text = listing(
        {
            f"<pulp::x86::V4>::vectorize::imp::<Impl<{body}>>": [f"{instruction} %xmm1,%xmm2"],
        }
    )
    missing = module.failures(module.parse_functions(text))
    assert any(body in line and "V4" in line for line in missing), missing


def main() -> int:
    module = load_module()
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test(module)
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
