#!/usr/bin/env python3
"""PyTorch reference runner for strided-einsum2 dot_general benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections.abc import Callable


DEFAULT_WARMUPS = 3
DEFAULT_RUNS = 15

BATCHED_MATMUL = "batched_matmul"
LAYOUTS_FULL = ("memory_matched", "NN", "TN", "NT", "TT")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_dtypes(value: str) -> list[str]:
    dtypes = []
    for item in value.split(","):
        item = item.strip()
        if item in {"f64", "c64", "c128"}:
            dtypes.append(item)
        elif item:
            raise argparse.ArgumentTypeError(f"unknown dtype: {item}")
    return dtypes or ["f64"]


def configure_thread_env(num_threads: int) -> None:
    value = str(num_threads)
    xla_multi_thread = "true" if num_threads > 1 else "false"
    os.environ.update(
        {
            "OMP_NUM_THREADS": value,
            "OMP_THREAD_LIMIT": value,
            "OMP_DYNAMIC": "FALSE",
            "RAYON_NUM_THREADS": value,
            "OPENBLAS_NUM_THREADS": value,
            "GOTO_NUM_THREADS": value,
            "MKL_NUM_THREADS": value,
            "VECLIB_MAXIMUM_THREADS": value,
            "VECLIB_NUM_THREADS": value,
            "NUMEXPR_NUM_THREADS": value,
            "BLIS_NUM_THREADS": value,
            "XLA_FLAGS": (
                f"--xla_cpu_multi_thread_eigen={xla_multi_thread} "
                f"intra_op_parallelism_threads={value}"
            ),
        }
    )


def case_specs(profile: str) -> list[tuple[str, tuple[int, ...]]]:
    if profile == "smoke":
        return [
            (BATCHED_MATMUL, (2, 8, 8, 8, "memory_matched")),
            (BATCHED_MATMUL, (2, 8, 8, 8, "TN")),
        ]
    if profile == "quick":
        return [(BATCHED_MATMUL, (32, 64, 64, 64, layout)) for layout in LAYOUTS_FULL]
    cases = []
    for dims in ((32, 64, 64, 64), (32, 128, 128, 128)):
        for layout in LAYOUTS_FULL:
            cases.append((BATCHED_MATMUL, (*dims, layout)))
    return cases


def benchmark_name(kind: str, dims: tuple[int, ...]) -> str:
    if kind == BATCHED_MATMUL:
        batch, m, n, k, layout = dims
        suffix = "" if layout == "memory_matched" else f"_{layout.lower()}"
        return f"bin_batched_matmul_b{batch}_m{m}_n{n}_k{k}{suffix}"
    raise ValueError(f"unknown benchmark kind: {kind}")


def shape_label(kind: str, dims: tuple[int, ...]) -> str:
    if kind == BATCHED_MATMUL:
        batch, m, n, k, layout = dims
        return f"b={batch};m={m};n={n};k={k};layout={layout}"
    raise ValueError(f"unknown benchmark kind: {kind}")


def median_iqr(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[(3 * len(ordered)) // 4] - ordered[len(ordered) // 4]


def bench(fn: Callable[[], object], runs: int, warmups: int) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        value = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
        if hasattr(value, "numel"):
            value.numel()
    return median_iqr(samples)


def torch_dtype(torch, dtype: str):
    if dtype == "f64":
        return torch.float64
    if dtype == "c64":
        return torch.complex64
    if dtype == "c128":
        return torch.complex128
    raise ValueError(f"unknown dtype: {dtype}")


def randn_tensor(torch, shape: tuple[int, ...], dtype: str):
    if dtype == "f64":
        return torch.randn(shape, dtype=torch_dtype(torch, dtype))
    real_dtype = torch.float32 if dtype == "c64" else torch.float64
    return torch.randn(shape, dtype=real_dtype) + 1j * torch.randn(shape, dtype=real_dtype)


def empty_tensor(torch, shape: tuple[int, ...], dtype: str):
    return torch.empty(shape, dtype=torch_dtype(torch, dtype))


def strided_randn_matrix(torch, dtype: str, batch: int, rows: int, cols: int, row_major: bool):
    if row_major:
        return randn_tensor(torch, (batch, rows, cols), dtype).contiguous()
    base = randn_tensor(torch, (batch * rows * cols,), dtype)
    return torch.as_strided(base, (batch, rows, cols), (rows * cols, 1, rows))


def empty_strided_matrix(torch, dtype: str, batch: int, rows: int, cols: int, row_major: bool):
    if row_major:
        return empty_tensor(torch, (batch, rows, cols), dtype).contiguous()
    return torch.empty_strided((batch, rows, cols), (rows * cols, 1, rows), dtype=torch_dtype(torch, dtype))


def make_batched_matmul_case(torch, dtype: str, batch: int, m: int, n: int, k: int, layout: str):
    if layout == "memory_matched":
        return make_memory_matched_batched_matmul_case(torch, dtype, batch, m, n, k)

    lhs = strided_randn_matrix(torch, dtype, batch, m, k, row_major=layout[0] == "T")
    rhs = strided_randn_matrix(torch, dtype, batch, k, n, row_major=layout[1] == "T")
    out = empty_strided_matrix(torch, dtype, batch, m, n, row_major=False)
    shape = f"b={batch};m={m};n={n};k={k};layout={layout}"
    return shape, {"pytorch-bmm": lambda: torch.bmm(lhs, rhs, out=out)}


def make_memory_matched_batched_matmul_case(torch, dtype: str, batch: int, m: int, n: int, k: int):
    lhs = randn_tensor(torch, (batch, m, k), dtype).contiguous()
    rhs = randn_tensor(torch, (batch, k, n), dtype).contiguous()
    out = empty_tensor(torch, (batch, m, n), dtype)
    shape = f"b={batch};m={m};n={n};k={k};layout=memory_matched"
    return shape, {"pytorch-bmm": lambda: torch.bmm(lhs, rhs, out=out)}


def make_case(torch, dtype: str, kind: str, dims: tuple[int, ...]):
    if kind == BATCHED_MATMUL:
        return make_batched_matmul_case(torch, dtype, *dims)
    raise ValueError(f"unknown benchmark kind: {kind}")


def emit_error_rows(writer: csv.DictWriter, args: argparse.Namespace, status: str) -> None:
    for dtype in parse_dtypes(args.dtypes):
        for kind, dims in case_specs(args.profile):
            for backend in ("pytorch-bmm",):
                writer.writerow(
                    {
                        "suite": "dot_general",
                        "benchmark": benchmark_name(kind, dims),
                        "dtype": dtype,
                        "threads": args.num_threads,
                        "shape": shape_label(kind, dims),
                        "backend": backend,
                        "median_ms": "",
                        "iqr_ms": "",
                        "status": status,
                    }
                )


def run(args: argparse.Namespace, writer: csv.DictWriter) -> None:
    try:
        import torch
    except Exception as exc:
        emit_error_rows(writer, args, f"error: {exc}")
        return

    torch.manual_seed(0)
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    for dtype in parse_dtypes(args.dtypes):
        for kind, dims in case_specs(args.profile):
            try:
                shape, cases = make_case(torch, dtype, kind, dims)
            except Exception as exc:
                shape = ""
                cases = {
                    "pytorch-bmm": None,
                }
                setup_status = f"error: {exc}"
            else:
                setup_status = "ok"

            for backend, fn in cases.items():
                if fn is None:
                    status = setup_status
                    median = ""
                    iqr = ""
                else:
                    try:
                        median_ms, iqr_ms = bench(fn, args.runs, args.warmups)
                        status = "ok"
                        median = f"{median_ms:.6f}"
                        iqr = f"{iqr_ms:.6f}"
                    except Exception as exc:
                        status = f"error: {exc}"
                        median = ""
                        iqr = ""

                writer.writerow(
                    {
                        "suite": "dot_general",
                        "benchmark": benchmark_name(kind, dims),
                        "dtype": dtype,
                        "threads": args.num_threads,
                        "shape": shape,
                        "backend": backend,
                        "median_ms": median,
                        "iqr_ms": iqr,
                        "status": status,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=positive_int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument(
        "--profile",
        choices=["smoke", "quick", "full"],
        default=os.environ.get("STRIDED_EINSUM2_DOT_GENERAL_BENCH_PROFILE", "full"),
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=int(os.environ.get("STRIDED_EINSUM2_DOT_GENERAL_BENCH_RUNS", DEFAULT_RUNS)),
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=int(os.environ.get("STRIDED_EINSUM2_DOT_GENERAL_BENCH_WARMUPS", DEFAULT_WARMUPS)),
    )
    parser.add_argument("--dtypes", default=os.environ.get("STRIDED_EINSUM2_DOT_GENERAL_BENCH_DTYPES", "f64"))
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    configure_thread_env(args.num_threads)

    fieldnames = [
        "suite",
        "benchmark",
        "dtype",
        "threads",
        "shape",
        "backend",
        "median_ms",
        "iqr_ms",
        "status",
    ]
    if args.output:
        file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            if not file_exists:
                writer.writeheader()
            run(args, writer)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        run(args, writer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
